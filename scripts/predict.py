"""
predict.py — 每日推論 Pipeline
台股收盤後（每日凌晨或收盤後自動執行）：
  1. 讀取最新資料 CSV（或從 DB 載入）
  2. 特徵工程（增量）
  3. 載入最新模型
  4. 輸出「30 天趨勢預測報告」＋ JSON
  5. 可選：資料漂移偵測（Evidently）

執行：
    python predict.py
    python predict.py --output-json results/pred_2026.json
    python predict.py --drift-check   # 開啟資料漂移監控
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ALL_FEATURES, HORIZON, MODEL_DIR, OUTPUT_DIR,
    TFT_PARAMS, EVAL_TARGETS, STOCK_CONFIGS, get_all_features,
    CONFIDENCE_THRESHOLD
)
from data_pipeline import build_daily_frame
from feature_engineering import build_features, build_features_with_medium_term
from signal_filter import SignalFilter, FilterResult

from utils.model_loader import safe_load

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 載入已訓練模型
# ─────────────────────────────────────────────

def load_ensemble(stock_id: str, path: Optional[Path] = None):
    path = path or (MODEL_DIR / f"ensemble_{stock_id}.pkl")
    if not path.exists():
        # Fallback to general final model if specific one doesn't exist
        path = MODEL_DIR / "ensemble_final.pkl"
        
    if not path.exists():
        raise FileNotFoundError(
            f"找不到股票 {stock_id} 的模型，請先執行 train_evaluate.py --stock-id {stock_id}"
        )
    # [P3 修復] 使用 safe_load (File Locking) 避免與訓練進程衝突
    model = safe_load(path)
    logger.info(f"模型載入 (Safe)：{path}")
    return model


def load_tft(stock_id: str, path: Optional[Path] = None):
    path = path or (MODEL_DIR / f"tft_{stock_id}_final.ckpt")
    if not path.exists():
        # Fallback to legacy path
        legacy_path = MODEL_DIR / "tft_final.ckpt"
        if legacy_path.exists():
            path = legacy_path
        else:
            logger.warning(f"找不到 TFT checkpoint {path}，跳過 TFT")
            return None
    try:
        from models.tft_model import TFTPredictor
        return TFTPredictor.load(str(path), TFT_PARAMS)
    except Exception as e:
        logger.warning(f"TFT 載入失敗：{e}")
        return None


# ─────────────────────────────────────────────
# 信心度分類
# ─────────────────────────────────────────────

def classify_confidence(prob_up: float,
                        model_agreement: float,
                        macro_shock: bool) -> str:
    """
    prob_up         ：上漲機率
    model_agreement ：三個模型方向一致性（0~1）
    macro_shock     ：是否偵測到宏觀衝擊（利率急升、匯率劇變等）
    """
    if macro_shock:
        return "🔴 高不確定（宏觀衝擊）"

    # 八二法則：極端高信心區間
    if prob_up >= CONFIDENCE_THRESHOLD:
        return "🔥 強烈買進 (STRONG_BUY)"
    elif prob_up <= (1 - CONFIDENCE_THRESHOLD):
        return "❄️ 強烈賣出 (STRONG_SELL)"

    abs_signal = abs(prob_up - 0.5) * 2   # 轉換為 0~1 的確定性
    if abs_signal >= 0.4 and model_agreement >= 0.8:
        return "🟢 高信心"
    elif abs_signal >= 0.2 and model_agreement >= 0.6:
        return "🟡 中等信心"
    else:
        return "🔴 低信心（建議觀望）"


def detect_macro_shock(df: pd.DataFrame) -> bool:
    """
    簡易宏觀衝擊偵測：
      - FED 利率 30 天變化 > 0.5%
      - USD/TWD 10 天變化 > 3%
      - TAIEX 5 天跌幅 > 5%
    """
    latest = df.iloc[-1]

    fed_shock = abs(latest.get("fed_rate_chg_30d", 0)) > 0.5
    fx_shock  = abs(latest.get("usd_twd_chg_10d", 0))  > 0.03
    idx_shock = latest.get("taiex_ret_5d", 0) < -0.05

    if fed_shock:
        logger.warning("宏觀衝擊偵測：FED 利率急變")
    if fx_shock:
        logger.warning("宏觀衝擊偵測：USD/TWD 匯率劇變")
    if idx_shock:
        logger.warning("宏觀衝擊偵測：TAIEX 急跌")

    return fed_shock or fx_shock or idx_shock


# ─────────────────────────────────────────────
# 資料漂移偵測（Evidently）
# ─────────────────────────────────────────────

def check_data_drift(reference_df: pd.DataFrame,
                     current_df: pd.DataFrame,
                     output_path: Path) -> bool:
    """
    使用 Evidently 偵測特徵分佈漂移。
    回傳 True 表示漂移顯著（需要重新訓練）。
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        feat_cols = [c for c in ALL_FEATURES if c in reference_df.columns]
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df[feat_cols].tail(252),
            current_data=current_df[feat_cols].tail(30),
        )
        report.save_html(str(output_path / "drift_report.html"))

        # 取得漂移摘要
        result_dict = report.as_dict()
        drift_detected = result_dict["metrics"][0]["result"].get(
            "dataset_drift", False
        )
        logger.info(f"資料漂移偵測：{'⚠️ 漂移顯著，建議重新訓練' if drift_detected else '✅ 無顯著漂移'}")
        return drift_detected

    except ImportError:
        logger.warning("evidently 未安裝，跳過資料漂移偵測（pip install evidently）")
        return False
    except Exception as e:
        logger.warning(f"漂移偵測失敗：{e}")
        return False


# ─────────────────────────────────────────────
# SHAP 解釋（第一性驗證）
# ─────────────────────────────────────────────

def explain_prediction(ensemble, X_latest: pd.DataFrame) -> dict:
    """
    取得最新一天的 SHAP 解釋（top 10 正/負貢獻因子）。
    """
    try:
        shap_dict = ensemble.shap_analysis(X_latest)
        xgb_shap  = shap_dict.get("xgb_shap")
        lgb_shap  = shap_dict.get("lgb_shap")

        if xgb_shap is None and lgb_shap is None:
            return {}

        shap_vals = xgb_shap if xgb_shap is not None else lgb_shap
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]    # 二分類取正類

        feat_cols = [c for c in ALL_FEATURES if c in X_latest.columns]
        shap_series = pd.Series(
            shap_vals[-1] if shap_vals.ndim > 1 else shap_vals,
            index=feat_cols[:len(shap_vals[-1])]
        ).sort_values(key=abs, ascending=False)

        return {
            "top_positive": shap_series[shap_series > 0].head(5).to_dict(),
            "top_negative": shap_series[shap_series < 0].head(5).to_dict(),
        }
    except Exception as e:
        logger.debug(f"SHAP 解釋失敗：{e}")
        return {}


# ─────────────────────────────────────────────
# 異常偵測 (Outlier Detection)
# ─────────────────────────────────────────────

def validate_prediction_sanity(report: dict, df_feat: pd.DataFrame) -> List[str]:
    """
    執行統計常理性檢查。若發現異常，回傳警告訊息列表。
    """
    warnings = []
    prob_up = report["prob_up"]
    current_close = report["current_close"]
    
    # 1. 預期報酬異常 (Magnitude Check)
    # 若 30 天預期漲跌幅超過 50%，在一般市場情況下極不合理
    exp_ret_mid = (report["target_price"]["mid"] / current_close) - 1
    if abs(exp_ret_mid) > 0.5:
        warnings.append(f"異常預期報酬: {exp_ret_mid:.1%}")

    # 2. 訊號退化檢查 (Entropy/Degeneration Check)
    # 如果 prob_up 剛好是 0.5000，且模型一致性極高，可能是特徵全被填為 0 導致的偏誤
    if abs(prob_up - 0.5) < 1e-4:
        warnings.append("預測訊號退化 (Prob=0.5)")

    # 3. 資料時效性檢查 (Staleness Check)
    # 如果最後一筆特徵資料距離現在超過 3 天，推論結果不具時效性
    last_data_date = df_feat.index[-1]
    if (datetime.now() - last_data_date).days > 3:
        warnings.append(f"資料過度陳舊: {last_data_date.strftime('%Y-%m-%d')}")

    return warnings


# ─────────────────────────────────────────────
# 主推論函式
# ─────────────────────────────────────────────

def run_prediction(
    df_feat:     pd.DataFrame,
    ensemble,
    stock_id:    str,
    tft=None,
    drift_check: bool = False,
) -> dict:
    """
    核心推論：
      輸入：特徵 DataFrame（最新歷史）
      輸出：預測報告 dict
    """
    today_str = df_feat.index[-1].strftime("%Y-%m-%d")
    target_str = pd.Timestamp(df_feat.index[-1]) + pd.Timedelta(days=HORIZON)
    target_str = target_str.strftime("%Y-%m-%d")

    current_close = float(df_feat["close"].iloc[-1])

    # ── 準備特徵 ──
    feat_cols = [c for c in ALL_FEATURES if c in df_feat.columns]
    X_latest  = df_feat[feat_cols].fillna(0).iloc[[-1]]    # 最後一行

    # ── TFT 推論（可選）──
    tft_result = None
    tft_prob_up = None
    if tft is not None:
        try:
            tft_result = tft.predict(df_feat)
            tft_prob_up = tft_result["prob_up"]
            logger.info(f"  TFT prob_up={tft_prob_up:.3f}")
        except Exception as e:
            logger.warning(f"TFT 推論失敗：{e}")

    # ── Ensemble 推論 ──
    pred_dict    = ensemble.predict(X_latest, tft_pred=tft_prob_up)
    prob_up_raw  = float(pred_dict["ensemble"][0])
    xgb_prob_raw = float(pred_dict["xgb"][0])   # 压縮前原始機率（僅供參考）
    lgb_prob_raw = float(pred_dict["lgb"][0])

    # 邏語對齊的「隔離貢獻校準」機率（經過 scaler + meta + calibrator 技紹）
    # 用於一致性計算：語意和 ensemble 對齊，0.5 為方向臨界
    xgb_prob = float(pred_dict.get("xgb_cal", pred_dict["xgb"])[0])
    lgb_prob = float(pred_dict.get("lgb_cal", pred_dict["lgb"])[0])
    tft_cal  = float(pred_dict.get("tft_cal", pred_dict["tft"])[0]) if tft_prob_up is not None else None

    # ── 極端估值動態權重抑制 (Dynamic Shrinkage) ──
    per_pct_rank = float(df_feat["per_pct_rank_252"].iloc[-1]) if "per_pct_rank_252" in df_feat.columns else 0.5
    shrinkage = 0.0
    if per_pct_rank > 0.95:
        shrinkage = (per_pct_rank - 0.95) / 0.05
    elif per_pct_rank < 0.05:
        shrinkage = (0.05 - per_pct_rank) / 0.05
        
    if shrinkage > 0:
        xgb_prob = xgb_prob * (1 - shrinkage) + 0.5 * shrinkage
        lgb_prob = lgb_prob * (1 - shrinkage) + 0.5 * shrinkage
        # 高度依賴樹模型組建的 ensemble，在極端區應向 TFT 靠攏，或向均衡點 0.5 退守
        fallback_tgt = tft_prob_up if tft_prob_up is not None else 0.5
        prob_up = prob_up_raw * (1 - shrinkage) + fallback_tgt * shrinkage
        logger.warning(f"歷史本益比達極端值 ({per_pct_rank*100:.1f}%)，啟動 {shrinkage*100:.0f}% Shrinkage 抑制外插發散。")
    else:
        prob_up = prob_up_raw

    # ── 模型一致性（使用校準後的個別機率，語意和 ensemble 對齊）──
    # 模型分此三層：
    #   raw 機率：xgb_prob_raw/lgb_prob_raw → 未經 scaler center，語意跟 ensemble 不同
    #   校準機率：xgb_prob/lgb_prob/tft_cal → transparency對齊，可直接比較方向
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

    # ── 宏觀衝擊偵測 ──
    macro_shock = detect_macro_shock(df_feat)

    # ── 信心度分類 ──
    confidence = classify_confidence(prob_up, agreement, macro_shock)

    # ── 極端估值強制降級控管 ──
    try:
        per_pct_rank = float(df_feat["per_pct_rank_252"].iloc[-1])
        if per_pct_rank > 0.95:
            if "高信心" in confidence:
                confidence = "🟡 中等信心（極端高估值降級）"
                logger.warning("估值位於歷史 >95% 分位，強制將高信心降級。")
            elif "中等信心" in confidence:
                confidence = "🔴 低信心（極端高估值降級）"
                logger.warning("估值位於歷史 >95% 分位，強制將中等信心降級。")
    except KeyError:
        pass

    # ── 量化預測區間（來自 TFT quantile，或簡化估計）──
    if tft_result and "quantiles" in tft_result:
        q = tft_result["quantiles"]
        exp_ret_low  = q[0]   # q10
        exp_ret_mid  = q[2]   # q50
        exp_ret_high = q[4]   # q90
    else:
        # 無 TFT 時，用歷史波動率估計區間
        hist_vol = df_feat["realized_vol_20d"].iloc[-1] / np.sqrt(252) * np.sqrt(HORIZON)
        adj = 2.0 if prob_up > 0.5 else -2.0
        center = (prob_up - 0.5) * adj * hist_vol
        exp_ret_low  = center - hist_vol
        exp_ret_mid  = center
        exp_ret_high = center + hist_vol

    target_price_low  = current_close * (1 + exp_ret_low)
    target_price_mid  = current_close * (1 + exp_ret_mid)
    target_price_high = current_close * (1 + exp_ret_high)

    # ── SHAP 解釋 ──
    shap_explanation = explain_prediction(ensemble, X_latest)

    # ── 最新關鍵指標 ──
    latest = df_feat.iloc[-1]
    key_indicators = {
        "close":             current_close,
        "rsi_14":            round(float(latest.get("rsi_14", 0)), 2),
        "foreign_net_ma5":   round(float(latest.get("foreign_net_ma5", 0)), 0),
        "per":               round(float(latest.get("per", 0)), 2),
        "per_pct_rank_252":  round(float(latest.get("per_pct_rank_252", 0)), 3),
        "fed_rate":          round(float(latest.get("fed_rate", 0)), 4),
        "revenue_yoy":       f"{float(latest.get('revenue_yoy', 0))*100:.1f}%",
        "gross_margin":      f"{float(latest.get('gross_margin', 0))*100:.1f}%",
        "days_to_ex_div":    int(latest.get("days_to_next_ex_dividend", 999)),
        "taiex_rel_strength": round(float(latest.get("taiex_rel_strength", 0)), 4),
    }

    # ── 資料漂移（可選）──
    drift_detected = False
    if drift_check:
        drift_detected = check_data_drift(
            df_feat.iloc[:-30], df_feat.iloc[-30:],
            OUTPUT_DIR,
        )

    # ── 組裝 30 天逐日預測軌跡 ──
    # 用 pandas bdate_range 取未來 30 個台灣交易日（近似，忽略台灣假日）
    as_of_dt = df_feat.index[-1]
    biz_dates = pd.bdate_range(start=as_of_dt + pd.Timedelta(days=1), periods=HORIZON)

    # TFT 每日 log-return 轉實際價格（分位數欄位是 log-return 尺度)
    tft_daily_q = None
    if tft_result and "daily_quantiles" in tft_result:
        # daily_quantiles: [[q10,q25,q50,q75,q90] * 30]
        tft_daily_q = tft_result["daily_quantiles"]

    # Ensemble prob_up 線性插值：累積漲跌構成價格軌跡
    # 以 30 日 prob_up 和歷史波動率建立每日線性預期路徑作為 ensemble_price
    hist_vol_daily = float(df_feat["realized_vol_20d"].iloc[-1]) / np.sqrt(252) \
        if "realized_vol_20d" in df_feat.columns else 0.01
    daily_drift = (prob_up - 0.5) * 2 * hist_vol_daily  # 方向 × 波動率

    is_extreme = bool(shrinkage > 0)
    is_macro   = bool(macro_shock)
    tft_prob_db = round(tft_prob_up, 4) if tft_prob_up else None
    # DB 儲存校準後的个別機率（可從建立對說性）
    xgb_prob_db = round(xgb_prob, 4)       # xgb_cal
    lgb_prob_db = round(lgb_prob, 4)       # lgb_cal

    trajectory = []
    for i, bdate in enumerate(biz_dates):
        day_offset = i + 1
        cum_drift  = daily_drift * day_offset
        ens_price  = round(current_close * (1 + cum_drift), 2)

        if tft_daily_q is not None and i < len(tft_daily_q):
            dq = tft_daily_q[i]          # [q10, q25, q50, q75, q90] as log-returns
            row_prices = [round(current_close * (1 + v), 2) for v in dq]
            p_q10, p_q25, p_q50, p_q75, p_q90 = row_prices
        else:
            # TFT 不可用時，以 Ensemble 路徑 ± vol 填充
            p_q50 = ens_price
            p_q10 = round(current_close * (1 + cum_drift - hist_vol_daily * np.sqrt(day_offset)), 2)
            p_q25 = round(current_close * (1 + cum_drift - hist_vol_daily * np.sqrt(day_offset) * 0.5), 2)
            p_q75 = round(current_close * (1 + cum_drift + hist_vol_daily * np.sqrt(day_offset) * 0.5), 2)
            p_q90 = round(current_close * (1 + cum_drift + hist_vol_daily * np.sqrt(day_offset)), 2)

        trajectory.append({
            "predict_date":     today_str,
            "stock_id":         stock_id,
            "forecast_date":    bdate.strftime("%Y-%m-%d"),
            "day_offset":       day_offset,
            "price_q10":        p_q10,
            "price_q25":        p_q25,
            "price_q50":        p_q50,
            "price_q75":        p_q75,
            "price_q90":        p_q90,
            "ensemble_price":   ens_price,
            "current_close":    current_close,
            "prob_up":          round(prob_up, 4),
            "confidence_level": confidence,
            "model_agreement":  round(agreement, 4),
            "xgb_prob":         xgb_prob_db,
            "lgb_prob":         lgb_prob_db,
            "tft_prob":         tft_prob_db,
            "extreme_valuation": is_extreme,
            "macro_shock":       is_macro,
            "warning_flag":      "", # 預留欄位
        })

    # ── 異常偵測 (Outlier Detection) ──
    sanity_warnings = validate_prediction_sanity(report, df_feat)
    if sanity_warnings:
        warning_str = "; ".join(sanity_warnings)
        logger.warning(f"⚠️ [{stock_id}] 偵測到異常信號: {warning_str}")
        for day in trajectory:
            day["warning_flag"] = warning_str

    # ── 組裝報告 ──
    report = {
        "as_of_date":        today_str,
        "target_date":       target_str,
        "horizon_days":      HORIZON,
        "stock_id":          stock_id,
        "current_close":     current_close,

        # 核心預測
        "prob_up":           round(prob_up, 4),
        "direction":         "\U0001f4c8 上漲" if prob_up > 0.5 else "\U0001f4c9 下跌",
        "confidence":        confidence,
        "model_agreement":   round(agreement, 4),

        # 預期報酬區間（q10~q90）
        "expected_return": {
            "low":    f"{exp_ret_low*100:.1f}%",
            "mid":    f"{exp_ret_mid*100:.1f}%",
            "high":   f"{exp_ret_high*100:.1f}%",
        },
        "target_price": {
            "low":  round(target_price_low, 1),
            "mid":  round(target_price_mid, 1),
            "high": round(target_price_high, 1),
        },

        # 模型明細（展示校準後的個別機率）
        "model_breakdown": {
            "ensemble": round(prob_up, 4),
            "xgb_cal":  round(xgb_prob, 4),      # 校準後 XGB 隔離貢獻機率
            "lgb_cal":  round(lgb_prob, 4),      # 校準後 LGB 隔離貢獻機率
            "tft":      round(tft_prob_up, 4) if tft_prob_up else "N/A",
            # 以下為 raw 機率，供進階檢視
            "xgb_raw":  round(xgb_prob_raw, 4),
            "lgb_raw":  round(lgb_prob_raw, 4),
        },

        # 逐日軌跡（30 筆，供 DB 寫入）
        "daily_trajectory":  trajectory,

        # 第一性因子解釋
        "shap_explanation":  shap_explanation,
        "key_indicators":    key_indicators,

        # 警示
        "warnings": {
            "macro_shock":     macro_shock,
            "data_drift":      drift_detected,
            "low_confidence":  "低信心" in confidence or "高不確定" in confidence,
        },

        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── 訊號過濾（Signal Filtering）───────────────────────────
    try:
        sf = SignalFilter()
        filter_result = sf.evaluate(report, df_feat)
        report["signal_filter"] = {
            "decision":       filter_result.decision,
            "overall_score":  round(filter_result.overall_score, 1),
            "is_tradeable":   filter_result.is_tradeable,
            "stop_loss":      filter_result.stop_loss,
            "take_profit":    filter_result.take_profit,
            "blocking_reasons": filter_result.blocking_reasons,
            "boosting_reasons": filter_result.boosting_reasons,
            "dimensions": [
                {
                    "name": d.name, "passed": d.passed,
                    "score": round(d.score, 3), "detail": d.detail,
                }
                for d in filter_result.dimensions
            ],
        }
        report["_filter_result"] = filter_result   # 供 print_report 使用
        logger.info(
            f"[SignalFilter] decision={filter_result.decision}  "
            f"score={filter_result.overall_score:.0f}/100"
        )
    except Exception as e:
        logger.warning(f"[SignalFilter] 過濾失敗，略過：{e}")
        report["signal_filter"] = {"decision": "UNKNOWN", "overall_score": 0}

    return report

# ─────────────────────────────────────────────
# 輸出格式化
# ─────────────────────────────────────────────

def print_report(report: dict):
    stock_id = report.get("stock_id", "Unknown")
    stock_name = STOCK_CONFIGS.get(stock_id, {}).get("name", "")
    print("\n" + "═" * 60)
    print(f"  {stock_name} ({stock_id}) 30 天趨勢預測報告")
    print(f"  基準日：{report['as_of_date']} → 目標日：{report['target_date']}")
    print("═" * 60)

    print(f"\n  當前收盤價：{report['current_close']:,.0f} TWD")
    print(f"\n  ▶ 預測方向   : {report['direction']}")
    print(f"  ▶ 上漲機率   : {report['prob_up']*100:.1f}%")
    print(f"  ▶ 信心等級   : {report['confidence']}")
    print(f"  ▶ 模型一致性 : {report['model_agreement']*100:.0f}%")

    print(f"\n  預期報酬區間（q10~q90）：")
    er = report["expected_return"]
    tp = report["target_price"]
    print(f"    低端  ({er['low']})  →  {tp['low']:,.0f} TWD")
    print(f"    中位  ({er['mid']})  →  {tp['mid']:,.0f} TWD  ← 點預測")
    print(f"    高端  ({er['high']}) →  {tp['high']:,.0f} TWD")

    print(f"\n  模型分解（校準後個別機率）：")
    mb = report["model_breakdown"]
    print(f"    {'ensemble':12s}: {mb['ensemble']}")
    print(f"    {'xgb_cal':12s}: {mb['xgb_cal']}  (raw: {mb['xgb_raw']})")
    print(f"    {'lgb_cal':12s}: {mb['lgb_cal']}  (raw: {mb['lgb_raw']})")
    tft_val = mb['tft'] if isinstance(mb['tft'], str) else f"{mb['tft']}"
    print(f"    {'tft':12s}: {tft_val}")

    if report["shap_explanation"]:
        print(f"\n  主要驅動因子（SHAP）：")
        pos = report["shap_explanation"].get("top_positive", {})
        neg = report["shap_explanation"].get("top_negative", {})
        print("    ↑ 支撐上漲：", list(pos.keys())[:3])
        print("    ↓ 壓制下跌：", list(neg.keys())[:3])

    print(f"\n  關鍵指標：")
    ki = report["key_indicators"]
    print(f"    RSI(14)：{ki['rsi_14']}  |  PER：{ki['per']}（歷史 {ki['per_pct_rank_252']*100:.0f}% 分位）")
    print(f"    外資淨買超 MA5：{ki['foreign_net_ma5']:,.0f}")
    print(f"    月營收 YoY：{ki['revenue_yoy']}  |  毛利率：{ki['gross_margin']}")
    print(f"    FED 利率：{ki['fed_rate']}%  |  距除息日：{ki['days_to_ex_div']} 天")

    warnings = report["warnings"]
    if any(warnings.values()):
        print(f"\n  ⚠️  警示：")
        if warnings["macro_shock"]:    print("    ・宏觀衝擊偵測（利率/匯率/指數急變）")
        if warnings["data_drift"]:    print("    ・特徵分佈漂移（建議重新訓練）")
        if warnings["low_confidence"]: print("    ・預測信心不足，建議謹慎")

    # ── 訊號過濾決策（Signal Filtering）──────────────────────
    filter_result: FilterResult | None = report.get("_filter_result")
    if filter_result is not None:
        print("")
        print(filter_result.summary())
    elif "signal_filter" in report:
        sf_data = report["signal_filter"]
        icon = {"LONG": "🟢", "HOLD_CASH": "🔴", "WATCH": "🟡"}.get(
            sf_data.get("decision", ""), "⚪"
        )
        print(f"\n  訊號過濾：{icon} {sf_data.get('decision')}  "
              f"（綜合評分：{sf_data.get('overall_score', 0):.0f}/100）")

    print("\n" + "─" * 60)
    print(f"  生成時間：{report['generated_at']}")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TSMC 30 天趨勢每日推論")
    parser.add_argument("--stock-id", default="2330", help="股票代碼")
    parser.add_argument("--output-json", default=None, help="輸出 JSON 路徑")
    parser.add_argument("--drift-check", action="store_true", help="啟用資料漂移偵測")
    parser.add_argument("--no-tft", action="store_true", help="跳過 TFT 推論")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    logger.info("=== TSMC 30 天趨勢預測 — 推論開始 ===")

    # 資料層
    from data_pipeline import load_features_from_store
    df_feat = load_features_from_store(stock_id=args.stock_id)
    
    if df_feat.empty:
        logger.warning(f"Feature Store 為空，嘗試即時計算特徵...")
        raw     = build_daily_frame(stock_id=args.stock_id)
        df_feat = build_features_with_medium_term(raw, stock_id=args.stock_id, for_inference=True)

    # 載入模型
    ensemble = load_ensemble(stock_id=args.stock_id)
    tft      = None if args.no_tft else load_tft(stock_id=args.stock_id)

    # 推論
    report = run_prediction(
        df_feat,
        ensemble,
        stock_id=args.stock_id,
        tft=tft,
        drift_check=args.drift_check,
    )

    # 寫入資料庫（逐日 30 筆）
    from data_pipeline import save_forecast_daily
    try:
        save_forecast_daily(report)
    except Exception as e:
        logger.error(f"寫入資料庫失敗：{e}")

    # 輸出
    print_report(report)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 已儲存：{out_path}")

    logger.info("=== 推論完成 ===")


if __name__ == "__main__":
    main()
