from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401
"""
historical_backfill.py — 歷史預測回填腳本
=====================================================================
對 stock_price 中每個實際交易日（stock_id=2330）執行模型推論，
將未來 30 個交易日的逐日預測軌跡批量寫入 stock_forecast_daily。

用途：
  - 建立歷史預測基準，供 backtest_audit.py 做校準驗證（Calibration）
  - 填補 predict.py 尚未累積的 live 資料

設計重點：
  - 特徵工程只做一次（全量），再按 predict_date 切片推論，速度極快
  - 使用現有 run_prediction() 邏輯，預測語意完全一致
  - TFT 預設關閉（每次需 30 秒，306 日 = 2.5 小時）；--with-tft 可啟用
  - 所有 9,180 筆一次性批量寫入（executemany），減少 DB round-trip

注意：本腳本使用的模型（ensemble_final.pkl）是以 2026/04 資料訓練
的最終版本，對 2025 年歷史日期而言存在 look-ahead bias。
這對「校準分析」（模型現在的機率品質）是可接受的；若需純 Walk-Forward
歷史回測，請用 train_evaluate.py 的 OOF 輸出（wf_fold_metrics.csv）。

執行：
    python historical_backfill.py                        # 2025/01/01 起
    python historical_backfill.py --since 2024-01-01     # 自訂起始日
    python historical_backfill.py --with-tft             # 啟用 TFT（慢）
    python historical_backfill.py --dry-run              # 試跑不寫入 DB
    python historical_backfill.py --workers 4            # 並行（不含 TFT）
"""


import argparse
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# ── scripts/ 目錄加入 path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import ALL_FEATURES, DB_CONFIG, HORIZON, MODEL_DIR, TFT_PARAMS
from data_pipeline import build_daily_frame, get_conn
from feature_engineering import build_features
from predict import (
    classify_confidence, detect_macro_shock, explain_prediction,
    load_ensemble, load_tft,
)

logger = logging.getLogger(__name__)

# [P1 修復 2.6] 移除硬編碼 DB 憑證，統一引用 config.DB_CONFIG
# 保留 DB_CONFIG 在 module 層級暴露，向後兼容外部 import


# ─────────────────────────────────────────────
# 取得待回填的交易日清單
# ─────────────────────────────────────────────

def get_predict_dates(since: str, until: Optional[str] = None,
                      stock_id: str = "2330") -> list[pd.Timestamp]:
    """從 stock_price 取得 [since, until] 內的所有實際交易日。"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            sql = """
                SELECT date FROM stock_price
                WHERE stock_id = %s AND date >= %s
            """
            params = [stock_id, since]
            if until:
                sql += " AND date <= %s"
                params.append(until)
            sql += " ORDER BY date"
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [pd.Timestamp(r[0]) for r in rows]


# ─────────────────────────────────────────────
# 單日推論（核心邏輯，與 predict.py 對齊）
# ─────────────────────────────────────────────

def predict_for_date(
    predict_date: pd.Timestamp,
    df_feat: pd.DataFrame,
    ensemble,
    tft=None,
    stock_id: str = "2330",
) -> list[dict]:
    """
    以 predict_date 為基準，對 df_feat 切片後執行推論，
    回傳 30 筆 trajectory dict（可直接寫入 stock_forecast_daily）。
    """
    # 切片：只使用 predict_date 當天及之前的歷史資料
    df_ctx = df_feat.loc[:predict_date]
    if df_ctx.empty or df_ctx.index[-1] != predict_date:
        return []

    today_str     = predict_date.strftime("%Y-%m-%d")
    current_close = float(df_ctx["close"].iloc[-1])

    # ── 特徵矩陣 ──
    feat_cols = [c for c in ALL_FEATURES if c in df_ctx.columns]
    X_latest  = df_ctx[feat_cols].fillna(0).iloc[[-1]]

    # ── TFT（可選，_prepare_dataset 已自動截取 min_rows 加速）──
    tft_prob_up = None
    tft_result  = None
    if tft is not None:
        try:
            tft_result  = tft.predict(df_ctx)
            tft_prob_up = tft_result["prob_up"]
        except Exception as e:
            logger.debug(f"  [{today_str}] TFT 失敗：{e}，略過")

    # ── Ensemble ──
    pred_dict   = ensemble.predict(X_latest, tft_pred=tft_prob_up)
    prob_up_raw = float(pred_dict["ensemble"][0])
    xgb_prob_raw = float(pred_dict["xgb"][0])
    lgb_prob_raw = float(pred_dict["lgb"][0])

    # 校準後個別機率（語意對齊）
    xgb_prob = float(pred_dict.get("xgb_cal", pred_dict["xgb"])[0])
    lgb_prob = float(pred_dict.get("lgb_cal", pred_dict["lgb"])[0])
    tft_cal  = float(pred_dict.get("tft_cal", pred_dict["tft"])[0]) if tft_prob_up is not None else None

    # ── Shrinkage（極端估值抑制）──
    per_pct_rank = float(df_ctx["per_pct_rank_252"].iloc[-1]) \
        if "per_pct_rank_252" in df_ctx.columns else 0.5
    shrinkage = 0.0
    if per_pct_rank > 0.95:
        shrinkage = (per_pct_rank - 0.95) / 0.05
    elif per_pct_rank < 0.05:
        shrinkage = (0.05 - per_pct_rank) / 0.05
    if shrinkage > 0:
        fallback = tft_prob_up if tft_prob_up is not None else 0.5
        prob_up  = prob_up_raw * (1 - shrinkage) + fallback * shrinkage
    else:
        prob_up = prob_up_raw

    # ── 模型一致性（校準後機率）──
    cal_probs = [prob_up, xgb_prob, lgb_prob]
    if tft_cal is not None:
        cal_probs.append(tft_cal)
    elif tft_prob_up is not None:
        cal_probs.append(tft_prob_up)

    all_bull = all(p >= 0.5 for p in cal_probs)
    all_bear = all(p <  0.5 for p in cal_probs)
    if all_bull or all_bear:
        std_dev   = float(np.std(cal_probs))
        agreement = max(0.0, min(1.0, 1.0 - (std_dev / 0.25)))
    else:
        agreement = 0.0

    # ── 宏觀衝擊 ──
    macro_shock = detect_macro_shock(df_ctx)

    # ── 信心度 ──
    confidence = classify_confidence(prob_up, agreement, macro_shock)
    if per_pct_rank > 0.95:
        if "高信心" in confidence:
            confidence = "🟡 中等信心（極端高估值降級）"
        elif "中等信心" in confidence:
            confidence = "🔴 低信心（極端高估值降級）"

    # ── 量化區間 ──
    if tft_result and "quantiles" in tft_result:
        q = tft_result["quantiles"]
        exp_ret_q = q   # [q10, q25, q50, q75, q90]
    else:
        hist_vol = float(df_ctx["realized_vol_20d"].iloc[-1]) / np.sqrt(252) * np.sqrt(HORIZON) \
            if "realized_vol_20d" in df_ctx.columns else 0.08
        adj    = 2.0 if prob_up > 0.5 else -2.0
        center = (prob_up - 0.5) * adj * hist_vol
        exp_ret_q = [
            center - hist_vol,          # q10
            center - hist_vol * 0.5,    # q25
            center,                     # q50
            center + hist_vol * 0.5,    # q75
            center + hist_vol,          # q90
        ]

    # ── 逐日 30-day 軌跡 ──
    is_extreme  = bool(shrinkage > 0)
    is_macro    = bool(macro_shock)
    tft_prob_db = round(tft_prob_up, 4) if tft_prob_up is not None else None
    xgb_prob_db = round(xgb_prob, 4)
    lgb_prob_db = round(lgb_prob, 4)

    hist_vol_daily = float(df_ctx["realized_vol_20d"].iloc[-1]) / np.sqrt(252) \
        if "realized_vol_20d" in df_ctx.columns else 0.01
    daily_drift = (prob_up - 0.5) * 2 * hist_vol_daily

    biz_dates = pd.bdate_range(
        start=predict_date + pd.Timedelta(days=1), periods=HORIZON
    )

    tft_daily_q = tft_result.get("daily_quantiles") if tft_result else None

    trajectory = []
    for i, bdate in enumerate(biz_dates):
        day_offset = i + 1
        cum_drift  = daily_drift * day_offset
        ens_price  = round(current_close * (1 + cum_drift), 2)

        if tft_daily_q is not None and i < len(tft_daily_q):
            dq      = tft_daily_q[i]
            prices  = [round(current_close * (1 + v), 2) for v in dq]
            p_q10, p_q25, p_q50, p_q75, p_q90 = prices
        else:
            p_q50 = ens_price
            p_q10 = round(current_close * (1 + cum_drift - hist_vol_daily * np.sqrt(day_offset)), 2)
            p_q25 = round(current_close * (1 + cum_drift - hist_vol_daily * np.sqrt(day_offset) * 0.5), 2)
            p_q75 = round(current_close * (1 + cum_drift + hist_vol_daily * np.sqrt(day_offset) * 0.5), 2)
            p_q90 = round(current_close * (1 + cum_drift + hist_vol_daily * np.sqrt(day_offset)), 2)

        trajectory.append({
            "predict_date":      today_str,
            "stock_id":          stock_id,
            "forecast_date":     bdate.strftime("%Y-%m-%d"),
            "day_offset":        day_offset,
            "price_q10":         p_q10,
            "price_q25":         p_q25,
            "price_q50":         p_q50,
            "price_q75":         p_q75,
            "price_q90":         p_q90,
            "ensemble_price":    ens_price,
            "current_close":     current_close,
            "prob_up":           round(prob_up, 4),
            "confidence_level":  confidence,
            "model_agreement":   round(agreement, 4),
            "xgb_prob":          xgb_prob_db,
            "lgb_prob":          lgb_prob_db,
            "tft_prob":          tft_prob_db,
            "extreme_valuation": is_extreme,
            "macro_shock":       is_macro,
            "is_backfill":        True,  # [P0 修復 2.2] 標記為歷史回填
        })

    return trajectory


# ─────────────────────────────────────────────
# 批量寫入 stock_forecast_daily
# ─────────────────────────────────────────────

UPSERT_SQL = """
    INSERT INTO stock_forecast_daily (
        predict_date, stock_id, forecast_date, day_offset,
        price_q10, price_q25, price_q50, price_q75, price_q90,
        ensemble_price,
        current_close, prob_up, confidence_level, model_agreement,
        xgb_prob, lgb_prob, tft_prob,
        extreme_valuation, macro_shock, is_backfill
    ) VALUES (
        %(predict_date)s, %(stock_id)s, %(forecast_date)s, %(day_offset)s,
        %(price_q10)s, %(price_q25)s, %(price_q50)s, %(price_q75)s, %(price_q90)s,
        %(ensemble_price)s,
        %(current_close)s, %(prob_up)s, %(confidence_level)s, %(model_agreement)s,
        %(xgb_prob)s, %(lgb_prob)s, %(tft_prob)s,
        %(extreme_valuation)s, %(macro_shock)s, %(is_backfill)s
    )
    ON CONFLICT (predict_date, stock_id, forecast_date) DO UPDATE SET
        day_offset        = EXCLUDED.day_offset,
        price_q10         = EXCLUDED.price_q10,
        price_q25         = EXCLUDED.price_q25,
        price_q50         = EXCLUDED.price_q50,
        price_q75         = EXCLUDED.price_q75,
        price_q90         = EXCLUDED.price_q90,
        ensemble_price    = EXCLUDED.ensemble_price,
        current_close     = EXCLUDED.current_close,
        prob_up           = EXCLUDED.prob_up,
        confidence_level  = EXCLUDED.confidence_level,
        model_agreement   = EXCLUDED.model_agreement,
        xgb_prob          = EXCLUDED.xgb_prob,
        lgb_prob          = EXCLUDED.lgb_prob,
        tft_prob          = EXCLUDED.tft_prob,
        extreme_valuation = EXCLUDED.extreme_valuation,
        macro_shock       = EXCLUDED.macro_shock,
        is_backfill       = EXCLUDED.is_backfill,
        created_at        = CURRENT_TIMESTAMP
"""


def bulk_upsert(all_records: list[dict], dry_run: bool = False) -> int:
    """將所有 trajectory 記錄批量 Upsert。"""
    if not all_records:
        return 0
    if dry_run:
        logger.info(f"[DRY-RUN] 跳過寫入 {len(all_records):,} 筆資料")
        return len(all_records)

    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_SQL, all_records, page_size=500)
        conn.commit()
    return len(all_records)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    p = argparse.ArgumentParser(description="2330 歷史預測回填（stock_forecast_daily）")
    p.add_argument("--since",     default="2025-01-01", help="回填起始日（預設 2025-01-01）")
    p.add_argument("--until",     default=None,          help="回填結束日（預設最新交易日）")
    p.add_argument("--with-tft",  action="store_true",   help="啟用 TFT（慢，每日~30s）")
    p.add_argument("--dry-run",   action="store_true",   help="試跑，不寫入資料庫")
    p.add_argument("--batch-log", default=10, type=int,  help="每 N 筆日期印一次進度")
    args = p.parse_args()

    t0 = time.time()
    logger.info("=" * 65)
    logger.info(f"  台積電 2330 歷史預測回填")
    logger.info(f"  期間：{args.since} ~ {args.until or '最新交易日'}")
    logger.info(f"  TFT : {'啟用（慢）' if args.with_tft else '停用（快）'}")
    logger.info(f"  模式：{'DRY-RUN（不寫入）' if args.dry_run else '正式寫入 stock_forecast_daily'}")
    logger.info("=" * 65)

    # ── Step 1：取得待回填交易日 ────────────────────────────────
    predict_dates = get_predict_dates(args.since, args.until)
    total = len(predict_dates)
    logger.info(f"\n[Step 1] 待回填交易日：{total} 天"
                f"（{predict_dates[0].date()} ~ {predict_dates[-1].date()}）"
                f"  預計產生 {total * HORIZON:,} 筆")

    # ── Step 2：特徵工程（全量，只做一次）──────────────────────
    logger.info("\n[Step 2] 載入全量特徵框架（只做一次）…")
    raw     = build_daily_frame(stock_id="2330")
    df_feat = build_features(raw, for_inference=True)
    logger.info(f"  特徵框架：{len(df_feat):,} 天 × {df_feat.shape[1]} 欄")

    # ── Step 3：載入模型 ─────────────────────────────────────────
    logger.info("\n[Step 3] 載入模型…")
    ensemble = load_ensemble()
    tft      = load_tft() if args.with_tft else None
    if tft is None and args.with_tft:
        logger.warning("  TFT 載入失敗，將以 tree 平均替代 TFT 訊號")

    # ── Step 4：逐日 TFT 推論 + 組裝 trajectory ──────────────────
    # 效能說明：tft_model.py 的 _prepare_dataset 已加入 min_rows 截取優化，
    # 每次 TFT 推論只處理最後 282 行（vs 之前的 ~7000 行），省去 25x DataSet
    # 建立開銷。306 天 × ~1.5s/call ≈ 7~10 分鐘。
    logger.info(f"\n[Step 4] 開始逐日推論（{total} 天）"
                f"{'  [含 TFT 分位數]' if tft else '  [無 TFT，使用 vol 估算]'}…")

    all_records = []
    skipped     = 0
    tft_success = 0

    for idx, predict_date in enumerate(predict_dates, 1):
        try:
            trajectory = predict_for_date(
                predict_date, df_feat, ensemble, tft=tft
            )
            if not trajectory:
                logger.warning(f"  [{predict_date.date()}] 無預測結果，略過")
                skipped += 1
                continue

            all_records.extend(trajectory)
            if tft and trajectory[0].get("tft_prob") is not None:
                tft_success += 1

            if idx % args.batch_log == 0 or idx == total:
                elapsed = time.time() - t0
                rate    = idx / elapsed
                eta     = (total - idx) / rate if rate > 0 else 0
                tft_tag = f"TFT✅({tft_success})" if tft else "no-TFT"
                logger.info(
                    f"  進度：{idx:3d}/{total}"
                    f"  date={predict_date.date()}"
                    f"  prob_up={trajectory[0]['prob_up']:.1%}"
                    f"  agree={trajectory[0]['model_agreement']:.0%}"
                    f"  [{tft_tag}]"
                    f"  ETA={eta:.0f}s"
                )

        except Exception as e:
            logger.error(f"  [{predict_date.date()}] 推論失敗：{e}")
            skipped += 1

    # ── Step 5：批量寫入 DB ─────────────────────────────────────
    written = bulk_upsert(all_records, dry_run=args.dry_run)

    # ── 完成摘要 ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info(f"\n{'='*65}")
    logger.info(f"  回填完成！")
    logger.info(f"  寫入筆數  : {written:,} 筆（{total - skipped} 天 × {HORIZON} 天/筆）")
    logger.info(f"  跳過天數  : {skipped}")
    logger.info(f"  總耗時    : {elapsed:.1f} 秒（平均 {elapsed/total:.2f} s/天）")
    logger.info(f"  下一步    : python backtest_audit.py --since {args.since}")
    logger.info(f"{'='*65}")


if __name__ == "__main__":
    main()
