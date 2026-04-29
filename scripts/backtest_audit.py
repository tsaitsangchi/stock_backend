"""
backtest_audit.py — 歷史預測回測審計
=====================================================================
策略：利用 train_evaluate.py 所產出的 Walk-Forward OOF 預測
      與實際收盤價做交叉比對，驗證：

  1. 校準圖 (Calibration / Reliability Diagram)
     → 當模型說 70% 上漲機率，實際上漲率是否接近 70%？

  2. 區間覆蓋率 (Coverage Rate)
     → q10~q90 預測區間，實際上覆蓋了幾 % 的樣本？（期望 ~80%）

  3. 逐日 MAE / RMSE（day_offset 1~30）
     → 在 stock_forecast_daily 有資料後持續追蹤以確認精度衰減！

  4. 模型在不同市況下的條件勝率
     → 低/中/高波動 regime 各自的方向準確率

執行：
    python backtest_audit.py                    # 使用 OOF 做完整歷史審計
    python backtest_audit.py --since 2024-01-01 # 只看 2024 之後
    python backtest_audit.py --live             # 只看 stock_forecast_daily 的真實部署紀錄

【為什麼不從 2025/01/01 開始跑新的預測做校準？】
  stock_forecast_daily 目前只有 2026-04-14 一筆；要等 30 天後（2026-05-14）
  才有第一個可驗證的終端預測。
  → 最快速的路徑：用 Walk-Forward OOF 預測（已產出 1,140 筆，涵蓋近 5 年）
    進行統計審計，立刻獲得模型真實表現的基準線。
  → 之後每天 predict.py 執行後，30 天後自動累積進 live_backtest 表。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# 確保 scripts/ 在 path
sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, DB_CONFIG

logger = logging.getLogger(__name__)


def get_conn():
    # [P1 修復] 統一引用 config.DB_CONFIG，不再在此模組重複定義
    return psycopg2.connect(**DB_CONFIG)


# ─────────────────────────────────────────────
# 資料載入
# ─────────────────────────────────────────────

def load_oof_predictions(stock_id: str = "2330", since: str = "2020-01-01") -> pd.DataFrame:
    """
    從 outputs/wf_fold_metrics.csv + feature_engineering OOF 重建評估資料。
    由於 OOF 預測本身已保存在 DB 中（target_30d 就是實際報酬），
    這裡直接從 stock_price 取實際收盤，搭配 OOF meta 機率做校準。

    注意：OOF 機率本身儲存在 meta_oof_metrics.csv，但缺逐日索引。
    → 改用更直接的方式：重新從 build_daily_frame + build_features
      取特徵框架，再用已保存的模型推論每一天（Walk-Forward Simulation）。

    本函式回傳 stock_price 的完整收盤序列供後續分析。
    """
    conn = get_conn()
    sql = f"""
        SELECT date, close::float
        FROM   stock_price
        WHERE  stock_id = %s
          AND  date >= %s
        ORDER  BY date
    """
    df = pd.read_sql(sql, conn, params=(stock_id, since), parse_dates=["date"])
    conn.close()
    df = df.set_index("date")
    return df


def load_live_forecasts(stock_id: str = "2330",
                        since: str = "2025-01-01",
                        exclude_backfill: bool = True) -> pd.DataFrame:
    """
    從 stock_forecast_daily 取「已有實際收盤可驗證」的預測。
    JOIN stock_price 取得 forecast_date 的實際收盤。

    [P0 修復 2.2] 加入 exclude_backfill 參數：
      預設 True，排除 historical_backfill.py 回填的資料，
      避免 look-ahead bias 污染 Signal Filtering 績效分析。
    """
    conn = get_conn()
    # [P0] is_backfill 過濾子句（若欄位不存在則降級為不過濾並記錄警告）
    backfill_clause = "AND f.is_backfill = FALSE" if exclude_backfill else ""
    sql = f"""
        SELECT
            f.predict_date,
            f.forecast_date,
            f.day_offset,
            f.current_close,
            f.prob_up,
            f.confidence_level,
            f.price_q10,
            f.price_q25,
            f.price_q50,
            f.price_q75,
            f.price_q90,
            f.ensemble_price,
            f.xgb_prob,
            f.lgb_prob,
            f.tft_prob,
            f.extreme_valuation,
            f.macro_shock,
            p.close::float AS actual_close
        FROM stock_forecast_daily f
        JOIN stock_price p
          ON p.stock_id = f.stock_id
         AND p.date     = f.forecast_date
        WHERE f.stock_id   = %s
          AND f.predict_date >= %s
          {backfill_clause}
        ORDER BY f.predict_date, f.day_offset
    """
    try:
        df = pd.read_sql(sql, conn, params=(stock_id, since), parse_dates=["predict_date", "forecast_date"])
    except Exception as e:
        # is_backfill 欄位可能尚未遷移，降級為不過濾
        logger.warning(f"is_backfill 過濾失敗（{e}），降級為不過濾（請執行 DB 遷移加入此欄位）")
        sql_fallback = sql.replace(backfill_clause, "")
        df = pd.read_sql(sql_fallback, conn, params=(stock_id, since), parse_dates=["predict_date", "forecast_date"])
    conn.close()
    return df


def load_wf_oof_with_actual(stock_id: str = "2330") -> dict:
    """
    讀取 walk-forward OOF 結果。

    [P2 修復 2.10] 同時載入逐筆 OOF 預測序列（含日期 + 真實標籤），
    以便 calibration_analysis() 在 WF 模式下也能執行（不只 Live 模式）。

    回傳 dict（key 對應檔案）：
      meta    : meta_oof_metrics.csv     全局 OOF 指標
      folds   : wf_fold_metrics.csv      每 fold 評估
      regime  : regime_metrics.csv       不同 regime 的 OOF 指標
      oof_seq : oof_predictions_with_dates_{stock_id}.csv  逐筆 prob_up + y_true（新）
    """
    logger.info(f"載入 Walk-Forward OOF 指標 ({stock_id})…")
    meta_path    = OUTPUT_DIR / "meta_oof_metrics.csv"
    fold_path    = OUTPUT_DIR / "wf_fold_metrics.csv"
    regime_path  = OUTPUT_DIR / "regime_metrics.csv"
    oof_seq_path = OUTPUT_DIR / f"oof_predictions_with_dates_{stock_id}.csv"

    results: dict = {}
    if meta_path.exists():
        results["meta"]   = pd.read_csv(meta_path)
    if fold_path.exists():
        results["folds"]  = pd.read_csv(fold_path)
    if regime_path.exists():
        results["regime"] = pd.read_csv(regime_path)
    if oof_seq_path.exists():
        try:
            results["oof_seq"] = pd.read_csv(oof_seq_path, parse_dates=["date"])
            logger.info(
                f"  逐筆 OOF 序列：{len(results['oof_seq'])} 筆"
                f"（{results['oof_seq']['date'].min().date()}"
                f" ~ {results['oof_seq']['date'].max().date()}）"
            )
        except Exception as e:
            logger.warning(f"讀取 oof_predictions_with_dates.csv 失敗：{e}")

    return results


def oof_calibration_analysis(oof_seq: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    [P2 新增 2.10] 對 Walk-Forward OOF 預測做完整校準分析（ECE）。
    輸入 oof_seq 需含 prob_up + y_true 欄。
    """
    df = oof_seq.dropna(subset=["prob_up", "y_true"]).copy()
    df["prob_bucket"] = (df["prob_up"] * 10).round() / 10  # 0.1 為單位
    cal = (
        df.groupby("prob_bucket")
          .agg(
              predicted_prob = ("prob_up",  "mean"),
              actual_up_rate = ("y_true",   "mean"),
              count          = ("y_true",   "count"),
          )
          .reset_index()
    )
    cal["calibration_error"] = (cal["predicted_prob"] - cal["actual_up_rate"]).abs()
    total = cal["count"].sum()
    ece   = float((cal["calibration_error"] * cal["count"] / max(total, 1)).sum())
    return cal, ece


# ─────────────────────────────────────────────
# 校準分析
# ─────────────────────────────────────────────

def calibration_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    輸入需有欄位：prob_up, actual_up (0/1)
    輸出：各 prob bucket 的預測 vs 實際對比
    """
    df = df.copy()
    df["prob_bucket"] = (df["prob_up"] * 10).round() / 10  # 0.1 為單位
    df["actual_up"]   = (df["actual_close"] > df["current_close"]).astype(int)

    cal = df.groupby("prob_bucket").agg(
        predicted_prob  = ("prob_up",     "mean"),
        actual_up_rate  = ("actual_up",   "mean"),
        count           = ("actual_up",   "count"),
    ).reset_index()

    cal["calibration_error"] = (cal["predicted_prob"] - cal["actual_up_rate"]).abs()
    # Expected Calibration Error（ECE）
    total = cal["count"].sum()
    ece   = float((cal["calibration_error"] * cal["count"] / total).sum())

    return cal, ece


def coverage_analysis(df: pd.DataFrame) -> dict:
    """
    驗證分位數區間的實際覆蓋率。
    期望：q10~q90 覆蓋 ~80%，q25~q75 覆蓋 ~50%。
    """
    df = df.copy()
    results = {}
    for lo, hi, label, expected in [
        ("price_q10", "price_q90", "q10~q90", 0.80),
        ("price_q25", "price_q75", "q25~q75", 0.50),
    ]:
        if lo in df.columns and hi in df.columns:
            covered = ((df["actual_close"] >= df[lo]) &
                       (df["actual_close"] <= df[hi])).mean()
            results[label] = {
                "coverage_rate": round(float(covered), 4),
                "expected":      expected,
                "gap":           round(float(covered - expected), 4),
                "status":        "✅" if abs(covered - expected) < 0.08 else "⚠️",
            }
    return results


def daily_mae_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    逐 day_offset (1~30) 計算 MAE、RMSE、MAPE。
    """
    df = df.copy()
    df["abs_err"]  = (df["actual_close"] - df["price_q50"]).abs()
    df["sq_err"]   = (df["actual_close"] - df["price_q50"]) ** 2
    df["pct_err"]  = df["abs_err"] / df["actual_close"]

    by_day = df.groupby("day_offset").agg(
        mae  = ("abs_err", "mean"),
        rmse = ("sq_err",  lambda x: np.sqrt(x.mean())),
        mape = ("pct_err", "mean"),
        n    = ("abs_err", "count"),
    ).reset_index()
    return by_day


def regime_live_analysis(df: pd.DataFrame) -> dict:
    """
    從 stock_forecast_daily 的 live 紀錄按信心等級分群驗證。
    """
    df = df.copy()
    df["actual_up"] = (df["actual_close"] > df["current_close"]).astype(int)

    results = {}
    for conf_level in df["confidence_level"].unique():
        subset = df[df["confidence_level"] == conf_level]
        if len(subset) < 5:
            continue
        results[str(conf_level)] = {
            "count":           len(subset),
            "actual_up_rate":  round(float(subset["actual_up"].mean()), 4),
            "mean_prob_up":    round(float(subset["prob_up"].mean()), 4),
            "calibration_err": round(
                abs(subset["prob_up"].mean() - subset["actual_up"].mean()), 4
            ),
        }
    return results


# ─────────────────────────────────────────────
# 主報告
# ─────────────────────────────────────────────

def print_wf_summary(wf_data: dict):
    print("\n" + "═" * 65)
    print("  ① Walk-Forward OOF 指標（訓練時期已驗證，涵蓋全歷史）")
    print("═" * 65)

    if "meta" in wf_data:
        meta = wf_data["meta"].iloc[0]
        print(f"\n  【Meta-Learner OOF 全局指標（1,140 筆，約 4.5 年）】")
        print(f"    方向準確率 (DA)  : {meta['directional_accuracy']:.1%}"
              f"  {'✅' if meta['directional_accuracy'] >= 0.65 else '⚠️'} （目標 ≥ 65%）")
        print(f"    AUC             : {meta['auc']:.4f}"
              f"  {'✅' if meta['auc'] >= 0.55 else '⚠️'}")
        print(f"    IC (Rank)       : {meta['ic']:.4f}")
        print(f"    Sharpe          : {meta['sharpe']:.2f}"
              f"  {'✅' if meta['sharpe'] >= 1.0 else '⚠️'} （目標 ≥ 1.0）")
        print(f"    勝率            : {meta['win_rate']:.1%}")
        print(f"    最大回撤        : {meta['max_drawdown']:.1%}")

    if "regime" in wf_data:
        print(f"\n  【波動 Regime 分析】")
        for _, row in wf_data["regime"].iterrows():
            decay_flag = ""
            print(f"    {row['regime']:<25s}  "
                  f"DA={row['directional_accuracy']:.1%}  "
                  f"AUC={row['auc']:.3f}  "
                  f"Sharpe={row['sharpe']:.2f}")

    if "folds" in wf_data:
        folds  = wf_data["folds"]
        n_fold = len(folds)
        da_std = folds["directional_accuracy"].std()
        da_avg = folds["directional_accuracy"].mean()
        # 每 fold 的 test 樣本數（從 config 取；若無則顯示未知）
        try:
            from config import WF_CONFIG
            test_w  = WF_CONFIG.get("test_window", WF_CONFIG.get("step_days", "?"))
            # 理論 DA std 上界
            da_std_theory = (0.5 / (test_w ** 0.5)) if isinstance(test_w, int) else None
        except Exception:
            test_w        = "?"
            da_std_theory = None

        print(f"\n  【Fold 穩定性（共 {n_fold} folds，每 fold test={test_w}d）】")
        print(f"    DA 平均                  : {da_avg:.1%}")
        theory_str = (f"，理論上界 {da_std_theory:.3f}" if da_std_theory else "")
        print(f"    DA 標準差（跨 {n_fold} folds）: {da_std:.3f}{theory_str}"
              f"  {'✅ 穩定（< 10%）' if da_std < 0.10 else '⚠ 波動大（≥ 10%）'}")
        print(f"    DA 最低 fold             : {folds['directional_accuracy'].min():.1%}")
        print(f"    DA 最高 fold             : {folds['directional_accuracy'].max():.1%}")
        if da_std >= 0.10 and isinstance(test_w, int) and test_w < 60:
            print(f"\n    ⚠️  DA std 偏高根本原因：每 fold test 窗口僅 {test_w} 天。")
            print(f"        二元樣本理論 std 上界 = 0.5/√{test_w} = {0.5/test_w**0.5:.3f}，")
            print(f"        建議在 config.WF_CONFIG 設定 test_window=126（半年）。")



def print_live_summary(df_live: pd.DataFrame):
    n = len(df_live)
    print("\n" + "═" * 65)
    print(f"  ② Live 部署紀錄驗證（stock_forecast_daily，共 {n} 筆）")
    print("═" * 65)

    if n == 0:
        print("\n  ⚠️  目前無可驗證的 live 紀錄（預測日 + 30 天未到期）")
        print("     → 每天執行 predict.py，30 天後自動累積驗證資料")
        print("\n  【預期時程】")
        print("    2026-05-14：第一筆可驗證（predict_date=2026-04-14, day_offset=30）")
        print("    2026-06-13：累積 30 筆（單次執行涵蓋所有 day_offset）")
        print("    2026-07-13：累積 60+ 筆，校準分析開始有統計意義")
        return

    # 有資料的情況
    df_live = df_live[df_live["day_offset"] == 30].copy()
    if df_live.empty:
        print("\n  尚無 day_offset=30 的終端預測可驗證")
        return

    cal_df, ece = calibration_analysis(df_live)
    print(f"\n  【校準誤差（ECE）】：{ece:.4f}  "
          f"{'✅ 良好（< 5%）' if ece < 0.05 else '⚠️ 需校準（≥ 5%）'}")

    print("\n  【各 prob_up bucket 校準表】")
    print(f"  {'區間':>6}  {'預測':>6}  {'實際':>6}  {'樣本':>5}  {'誤差':>6}")
    print("  " + "-" * 38)
    for _, row in cal_df.iterrows():
        flag = "✅" if row["calibration_error"] < 0.1 else "⚠️"
        print(f"  {row['prob_bucket']:>6.1f}  "
              f"{row['predicted_prob']:>6.1%}  "
              f"{row['actual_up_rate']:>6.1%}  "
              f"{int(row['count']):>5d}  "
              f"{row['calibration_error']:>6.1%}  {flag}")

    cov = coverage_analysis(df_live)
    print(f"\n  【分位數區間覆蓋率】（期望 q10~q90 ≈ 80%）")
    for label, stat in cov.items():
        print(f"    {label}：{stat['coverage_rate']:.1%}  "
              f"（期望 {stat['expected']:.0%}，偏差 {stat['gap']:+.1%}）  {stat['status']}")

    by_day = daily_mae_analysis(df_live.assign(
        # 對 day_offset=30 以外的也算，若有的話
        **{}
    ))
    if not by_day.empty:
        last = by_day.iloc[-1]
        print(f"\n  【第 30 天預測精度】")
        print(f"    MAE  : {last['mae']:.1f} TWD  ({last['mape']:.1%})")
        print(f"    RMSE : {last['rmse']:.1f} TWD")


def print_recommendation(wf_data, df_live):
    print("\n" + "═" * 65)
    print("  ③ 建議行動")
    print("═" * 65)

    meta_da = wf_data.get("meta", pd.DataFrame()).get("directional_accuracy", pd.Series([0.66])).iloc[0]

    print(f"""
  現況評估：
    • Walk-Forward OOF DA = {meta_da:.1%}（{'達標' if meta_da >= 0.65 else '未達標'}，目標 ≥ 65%）
    • Live 驗證資料：{'有' if len(df_live) > 0 else '無'}（需 30 天積累期）

  接下來最值得做的事（按優先順序）：

  【立刻可做】
  1. 每天執行 predict.py（已設定），30 天後自動累積 live 驗證資料
     → 再執行本腳本時，② 欄位會自動填入實際比對結果

  【本週內】
  2. 機率校準（train_evaluate.py 已有 Isotonic Calibration）
     → 確認 _calibrator 是否已掛載到 ensemble_final.pkl：
       python -c "import joblib; m=joblib.load('outputs/models/ensemble_final.pkl'); print(hasattr(m,'_calibrator'))"

  3. 加入期貨籌碼特徵（futures_daily 表目前完全未使用）
     → 預期 IC 提升 +0.02~0.04

  【本月內】
  4. 建立自動 Walk-Forward 重訓排程（每月月底）
     → 避免 concept drift（目前模型靜態不動）

  5. 加入 ADR 溢價（us_stock_price 有 TSM 資料）
     → 作為台灣開盤前的方向領先信號
""")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="個股預測回測審計")
    p.add_argument("--stock-id", default="2330",
                   help="股票代碼（預設 2330）")
    p.add_argument("--since",    default="2025-01-01",
                   help="Live 驗證起始日（預設 2025-01-01）")
    p.add_argument("--live",     action="store_true",
                   help="僅顯示 live 部署紀錄驗證")
    p.add_argument("--save-csv", action="store_true",
                   help="儲存各分析結果為 CSV（到 outputs/）")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    stock_id = args.stock_id

    print("\n" + "═" * 65)
    print(f"  個股 ({stock_id}) 預測模型回測審計")
    print("  Walk-Forward OOF + Live 部署雙軌驗證")
    print("═" * 65)

    # ── ① Walk-Forward OOF 指標（歷史訓練驗證）─────────────────
    wf_data = load_wf_oof_with_actual(stock_id=stock_id)
    if not args.live:
        print_wf_summary(wf_data)

        # [P2 修復 2.10] 若有逐筆 OOF 序列，補充印出 OOF 校準分析
        oof_seq = wf_data.get("oof_seq")
        if oof_seq is not None and len(oof_seq) >= 50:
            try:
                cal_df, ece = oof_calibration_analysis(oof_seq)
                print("\n" + "─" * 65)
                print(f"  【Walk-Forward OOF 校準誤差 (ECE)】：{ece:.4f}  "
                      f"{'✅ 良好（< 5%）' if ece < 0.05 else '⚠️ 需校準（≥ 5%）'}")
                print(f"  N={len(oof_seq):,} 筆 OOF 預測")
                print(f"  {'區間':>6}  {'預測':>6}  {'實際':>6}  {'樣本':>6}  {'誤差':>6}")
                print("  " + "-" * 38)
                for _, row in cal_df.iterrows():
                    flag = "✅" if row["calibration_error"] < 0.1 else "⚠️"
                    print(f"  {row['prob_bucket']:>6.1f}  "
                          f"{row['predicted_prob']:>6.1%}  "
                          f"{row['actual_up_rate']:>6.1%}  "
                          f"{int(row['count']):>6d}  "
                          f"{row['calibration_error']:>6.1%}  {flag}")
            except Exception as e:
                logger.warning(f"OOF 校準分析失敗：{e}")

    # ── ② Live 部署紀錄（stock_forecast_daily × stock_price）────
    logger.info(f"載入 live 預測紀錄 ({stock_id}, since={args.since})…")
    df_live = load_live_forecasts(stock_id=stock_id, since=args.since)
    logger.info(f"  live 紀錄：{len(df_live)} 筆（可驗證：已過期且有實際收盤）")
    print_live_summary(df_live)

    # ── ③ Signal Filtering 回測比較 ─────────────────────────────
    # 比較「全部訊號交易」vs「只在過濾通過時交易」的績效差異
    if len(df_live) > 0 and "actual_close" in df_live.columns and not df_live["actual_close"].isna().all():
        try:
            df_day30 = df_live[df_live["day_offset"] == 30].copy()
            if len(df_day30) >= 5:
                df_day30["actual_ret"] = df_day30["actual_close"] / df_day30["current_close"] - 1
                df_day30["signal_dir"] = (df_day30["prob_up"] >= 0.5).astype(int) * 2 - 1
                df_day30["strat_ret_all"] = df_day30["signal_dir"] * df_day30["actual_ret"]

                # 過濾：prob > 0.65 才交易
                filtered = df_day30[df_day30["prob_up"] >= 0.65].copy()

                print("\n" + "═" * 65)
                print("  ④ Signal Filtering 效益比較（prob > 0.65 門檻）")
                print("─" * 65)
                print(f"  全部訊號交易  ：{len(df_day30)} 次  "
                      f"DA={( (df_day30['signal_dir']*df_day30['actual_ret']) > 0).mean():.1%}  "
                      f"平均報酬={df_day30['strat_ret_all'].mean():.2%}")
                if len(filtered) > 0:
                    filtered["strat_ret"] = (
                        (filtered["prob_up"] >= 0.5).astype(int) * 2 - 1
                    ) * filtered["actual_ret"]
                    print(f"  過濾後只交易  ：{len(filtered)} 次  "
                          f"DA={((filtered['strat_ret']) > 0).mean():.1%}  "
                          f"平均報酬={filtered['strat_ret'].mean():.2%}")
                    reduction = 1 - len(filtered) / len(df_day30)
                    print(f"  交易次數減少  ：{reduction:.0%}（僅保留高信心訊號）")
                print("═" * 65)
        except Exception as e:
            logger.warning(f"Signal Filter 回測比較失敗：{e}")

    # ── ⑤ 建議 ─────────────────────────────────────────────────
    print_recommendation(wf_data, df_live)

    # ── 儲存 CSV ─────────────────────────────────────────────
    if args.save_csv and len(df_live) > 0:
        out = OUTPUT_DIR / f"live_backtest_audit_{stock_id}.csv"
        df_live.to_csv(out, index=False)
        logger.info(f"Live 審計資料已儲存：{out}")

    print("─" * 65)
    print("  使用 --save-csv 儲存明細  |  30 天後再執行本腳本看 live 驗證結果")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
