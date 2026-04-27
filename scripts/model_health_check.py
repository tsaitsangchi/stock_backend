"""
scripts/model_health_check.py
核心模型健康度監控與診斷

市場邏輯：
量化模型深受市場機制切換 (Regime Shift) 與特徵漂移 (Feature Drift) 影響。
本腳本每日執行，透過以下維度確保系統穩定性：
1. 數據完整性：檢查 PostgreSQL 最新資料更新時間，防止因爬蟲失效導致的「垃圾進、垃圾出」。
2. 模型時效性：監控各個股 .pkl/.ckpt 檔案之最後更新日，確保模型未過度陳舊。
3. 預測準確度：比對過去 30 天之預測結果與實際市場走勢 (Real-time Directional Accuracy)，偵測效能衰退。
4. 系統連續性：確認每日預測軌跡 (stock_forecast_daily) 是否覆蓋所有配置個股。

修改摘要（第三輪審查修復）：
  [P0] PSI 參考分佈三輪未修復——本版本徹底改寫：
       1. 優先使用 outputs/oof_predictions.csv 作為「真實參考分佈」
          （Walk-Forward OOF 期間實際的 prob_up 序列，反映模型在歷史資料
          上的合理機率分佈）
       2. 若 OOF 不存在則退回「stock_forecast_daily 中較早期的 prob_up」
          作為時間滑動參考（rolling reference）
       3. 最後才退回「固定 seed 的 Beta(2,2) 分佈」作為理論先驗
       4. 所有路徑均為 deterministic（同樣輸入永遠產生同樣 PSI），
          解決原版每次執行 PSI 不同的問題
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# 注入路徑
sys.path.append(str(Path(__file__).resolve().parent))
from config import MODEL_DIR, OUTPUT_DIR, STOCK_CONFIGS
from data_pipeline import _query

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def check_data_freshness_df() -> pd.DataFrame:
    """
    檢查各主要資料表的最新更新日期。
    統計理論：資料滯後會導致特徵工程計算出錯誤的 MA 或 RSI，造成預測偏差。
    """
    tables: List[str] = [
        "stock_price", 
        "stock_per", 
        "institutional_investors_buy_sell",
        "margin_purchase_short_sale", 
        "shareholding", 
        "stock_forecast_daily"
    ]
    results: List[Dict[str, Any]] = []
    for table in tables:
        try:
            sql: str = f"SELECT MAX(date) as last_date FROM {table}"
            df: pd.DataFrame = _query(sql)
            last_date = df["last_date"].iloc[0] if not df.empty else None
            results.append({"table": table, "last_date": last_date})
        except Exception as e:
            logger.error(f"檢查資料表 {table} 失敗: {e}")
            results.append({"table": table, "last_date": None})
    
    res_df: pd.DataFrame = pd.DataFrame(results)
    return res_df

def check_model_files_df(stock_ids: List[str]) -> pd.DataFrame:
    """
    檢查個股模型狀態。
    統計理論：模型具備時效性，若超過 30 天未重訓，可能已無法捕捉最新的市場 Alpha。
    """
    results: List[Dict[str, Any]] = []
    now: datetime = datetime.now()
    for sid in stock_ids:
        model_path: Path = MODEL_DIR / f"ensemble_{sid}.pkl"
        exists: bool = model_path.exists()
        
        if exists:
            mtime: datetime = datetime.fromtimestamp(model_path.stat().st_mtime)
            age_days: int = (now - mtime).days
            status: str = "🟢 OK" if age_days <= 30 else "🟡 STALE"
        else:
            mtime = None
            age_days = 999
            status = "🔴 MISSING"
        
        results.append({
            "stock_id": sid,
            "status": status,
            "age_days": age_days,
            "last_modified": mtime.strftime("%Y-%m-%d") if mtime else "N/A"
        })
    return pd.DataFrame(results)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    計算 PSI (Population Stability Index) 監控分佈穩定性。
    統計理論：PSI > 0.1 表示有顯著位移，PSI > 0.2 表示有嚴重位移，需重訓。

    為了避免 expected 與 actual 的尺度錯位，先以兩個分佈合併後的分位點
    切割 buckets（quantile-binning），再各自統計分佈，比固定區間更穩健。
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    # 以 expected 的分位點切割（避免 actual 過寬而拉變區間）
    quantile_pts = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    quantile_pts[0]  = -np.inf
    quantile_pts[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=quantile_pts)
    actual_counts,   _ = np.histogram(actual,   bins=quantile_pts)

    expected_percents = expected_counts / max(expected_counts.sum(), 1)
    actual_percents   = actual_counts   / max(actual_counts.sum(),   1)

    # 避免 log(0) / div-by-0
    expected_percents = np.where(expected_percents == 0, 1e-6, expected_percents)
    actual_percents   = np.where(actual_percents   == 0, 1e-6, actual_percents)

    psi_val = np.sum((actual_percents - expected_percents) *
                     np.log(actual_percents / expected_percents))
    return float(psi_val)


# ─────────────────────────────────────────────
# [P0 修復] PSI 參考分佈來源（取代隨機生成）
# ─────────────────────────────────────────────

_OOF_REF_CACHE: Optional[np.ndarray] = None


def _load_oof_reference(stock_id: str) -> Optional[np.ndarray]:
    """
    優先順序：
      1. models/oof_ref_dist_{stock_id}.npy（最快且精確）
      2. outputs/oof_predictions_with_dates_{stock_id}.csv
      3. outputs/oof_predictions_{stock_id}.csv
      4. 向後兼容的全局文件
      回傳 numpy array；找不到則回 None。
    """
    # 1. 嘗試載入 .npy
    npy_path = MODEL_DIR / f"oof_ref_dist_{stock_id}.npy"
    if npy_path.exists():
        try:
            arr = np.load(npy_path)
            if len(arr) >= 50:
                logger.info(f"[PSI] 採用 OOF 參考分佈 (.npy)：{stock_id}，N={len(arr)}")
                return arr
        except Exception as e:
            logger.warning(f"[PSI] 讀取 {npy_path.name} 失敗：{e}")

    # 2. 嘗試載入 CSVs
    candidates = [
        OUTPUT_DIR / f"oof_predictions_with_dates_{stock_id}.csv",
        OUTPUT_DIR / f"oof_predictions_{stock_id}.csv",
        OUTPUT_DIR / "oof_predictions_with_dates.csv",
        OUTPUT_DIR / "oof_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "prob_up" in df.columns:
                    arr = pd.to_numeric(df["prob_up"], errors="coerce").dropna().values
                    if len(arr) >= 50:
                        logger.info(f"[PSI] 採用 OOF 參考分佈 (CSV)：{path.name}，N={len(arr)}")
                        return arr
            except Exception as e:
                logger.debug(f"[PSI] 讀取 {path.name} 失敗：{e}")

    return None


def _historical_prob_reference(stock_id: str,
                               recent_n: int = 100,
                               history_n: int = 200) -> Optional[np.ndarray]:
    """
    從 stock_forecast_daily 取「較早一段時間」的 prob_up 作為參考分佈，
    再以「最近 N 筆」作為 actual。提供 stock-by-stock 的時間滑動參考。

    回傳 reference array；不足則 None。
    """
    sql = """
        SELECT prob_up
        FROM public.stock_forecast_daily
        WHERE stock_id = %s
          AND day_offset = 30
          AND COALESCE(is_backfill, FALSE) = FALSE
        ORDER BY predict_date DESC
        OFFSET %s
        LIMIT %s
    """
    try:
        df = _query(sql, (stock_id, recent_n, history_n))
        if df.empty:
            return None
        arr = pd.to_numeric(df["prob_up"], errors="coerce").dropna().values
        if len(arr) >= 50:
            return arr
    except Exception as e:
        logger.debug(f"[PSI] 歷史參考分佈讀取失敗（{stock_id}）：{e}")
    return None


def _theoretical_prior_reference(seed: int = 42, n: int = 1000) -> np.ndarray:
    """
    最後備援：固定 seed 的 Beta(2,2) 分佈，集中於 0.5、長尾平緩，
    比 normal(0.5, 0.1) 更貼近上漲機率的合理先驗。
    """
    rng = np.random.default_rng(seed)
    return rng.beta(2.0, 2.0, n)


def _get_reference_distribution(stock_id: str) -> tuple[np.ndarray, str]:
    """
    依優先順序取得參考分佈，並回傳 (array, source_label)。
    """
    oof_ref = _load_oof_reference(stock_id)
    if oof_ref is not None:
        return oof_ref, "oof"

    hist_ref = _historical_prob_reference(stock_id)
    if hist_ref is not None:
        return hist_ref, "history"

    return _theoretical_prior_reference(), "prior"


def check_prediction_drift_df(stock_ids: List[str]) -> pd.DataFrame:
    """
    監控預測分佈漂移 (Prediction Distribution Drift)。
    若機率分佈從「集中在中部」變為「兩極化」，通常是過擬合或 Regime Shift 的信號。

    [P0 修復] 參考分佈不再隨機生成，而是依優先順序使用：
      1. Walk-Forward OOF 預測（最佳，反映模型「歷史合理分佈」）
      2. 該標的較早期 live prob_up（次佳，stock-specific）
      3. 固定種子的 Beta(2,2) 理論先驗（最後備援）
    """
    results: List[Dict[str, Any]] = []
    for sid in stock_ids:
        # 取得最近 100 筆預測機率（actual / current 分佈）
        sql: str = """
            SELECT prob_up FROM public.stock_forecast_daily
            WHERE stock_id = %s AND day_offset = 30
              AND COALESCE(is_backfill, FALSE) = FALSE
            ORDER BY predict_date DESC LIMIT 100
        """
        df: pd.DataFrame = _query(sql, (sid,))
        if len(df) < 50:
            results.append({
                "stock_id": sid, "psi": np.nan,
                "psi_ref": "n/a", "drift_status": "⚪ INSUFFICIENT",
            })
            continue

        current_dist = pd.to_numeric(df["prob_up"], errors="coerce").dropna().values

        # [P0 修復] 取得真實參考分佈（deterministic）
        ref_dist, ref_source = _get_reference_distribution(sid)

        psi = calculate_psi(ref_dist, current_dist)

        status = "🟢 STABLE"
        if psi > 0.2:
            status = "🔴 DRIFTED (PSI > 0.2)"
        elif psi > 0.1:
            status = "🟡 WARNING (PSI > 0.1)"

        results.append({
            "stock_id":     sid,
            "psi":          round(psi, 4),
            "psi_ref":      ref_source,
            "drift_status": status,
        })
    return pd.DataFrame(results)

def evaluate_recent_performance_df(stock_ids: List[str], days: int = 45) -> pd.DataFrame:
    """
    實時計算過去一段時間的預測準確度 (DA)。
    統計理論：DA (Directional Accuracy) 是衡量分類器穩定性的核心指標。
    若 DA 跌破 50%，說明模型預測力已退化至隨機水平。
    """
    results: List[Dict[str, Any]] = []
    for sid in stock_ids:
        # 取得 predict_date 後 30 天的實際表現
        # 我們 join stock_price 來取得 forecast_date 的實際收盤價
        sql: str = """
            SELECT f.predict_date, f.forecast_date, f.current_close, f.prob_up,
                   p.close as actual_close
            FROM public.stock_forecast_daily f
            JOIN public.stock_price p ON f.forecast_date = p.date AND f.stock_id = p.stock_id
            WHERE f.stock_id = %s
              AND f.day_offset = 30
              AND f.predict_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY f.predict_date DESC
        """
        perf_df: pd.DataFrame = _query(sql, (sid, days))
        
        if perf_df.empty:
            results.append({
                "stock_id": sid, 
                "da": np.nan, 
                "samples": 0, 
                "perf_status": "⚪ NO_DATA"
            })
            continue
            
        # 計算實際漲跌
        perf_df["actual_ret"] = perf_df["actual_close"] / perf_df["current_close"] - 1
        perf_df["actual_up"]  = (perf_df["actual_ret"] > 0).astype(int)
        perf_df["pred_up"]    = (perf_df["prob_up"] > 0.5).astype(int)
        
        da: float = (perf_df["actual_up"] == perf_df["pred_up"]).mean()
        
        if da >= 0.65:
            status = "🔥 EXCELLENT"
        elif da >= 0.55:
            status = "🟢 STABLE"
        else:
            status = "⚠️ DEGRADED"
            
        results.append({
            "stock_id": sid,
            "da": round(da, 3),
            "samples": len(perf_df),
            "perf_status": status
        })
        
    return pd.DataFrame(results)

def run_health_check() -> None:
    """主運行程序：聚合所有監控維度並生成報告"""
    stock_ids: List[str] = list(STOCK_CONFIGS.keys())
    
    print("\n" + "═"*80)
    print(f" 🛡️  Antigravity Quant - 核心模型健康診斷系統 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("═"*80)
    
    # 1. 數據鮮度
    df_fresh_df = check_data_freshness_df()
    print("\n[1] 數據流鮮度監控 (Data Stream Freshness)")
    print("-" * 60)
    print(df_fresh_df.to_string(index=False))
    
    # 2. 模型檔案狀態
    df_models_df = check_model_files_df(stock_ids)
    print("\n[2] 模型檔案時效性 (Model Expiry Check)")
    print("-" * 60)
    anomalies_df = df_models_df[df_models_df["status"] != "🟢 OK"]
    if not anomalies_df.empty:
        print(anomalies_df.to_string(index=False))
    else:
        print("✅ 所有配置個股模型皆已就緒。")
    
    # 3. 預測分佈漂移監控 (PSI)
    df_drift = check_prediction_drift_df(stock_ids)
    print("\n[3] 預測分佈漂移監控 (PSI Prediction Drift)")
    print("-" * 60)
    drift_warn = df_drift[df_drift["drift_status"] != "🟢 STABLE"]
    if not drift_warn.empty:
        print(drift_warn.to_string(index=False))
    else:
        print("✅ 所有模型預測分佈穩定 (PSI < 0.1)。")

    # 4. 最近實戰表現
    df_perf_df = evaluate_recent_performance_df(stock_ids)
    print("\n[4] 實戰準確度評估 (Real-time Directional Accuracy - 30D)")
    print("-" * 60)
    df_perf_df = df_perf_df.sort_values("da", ascending=True)
    print(df_perf_df.head(15).to_string(index=False))
    
    # 總結警示
    stale_stocks = df_models_df[df_models_df["age_days"] > 30]["stock_id"].tolist()
    degraded_stocks = df_perf_df[df_perf_df["perf_status"] == "⚠️ DEGRADED"]["stock_id"].tolist()
    drifted_stocks = df_drift[df_drift["psi"] > 0.2]["stock_id"].tolist()
    
    print("\n" + "═"*80)
    if not stale_stocks and not degraded_stocks and not drifted_stocks:
        print(" 🎉 系統健康度：優良。所有模型與數據流運作正常。")
    else:
        if stale_stocks:
            print(f" ⏳ 模型過期警告：{stale_stocks}")
        if drifted_stocks:
            print(f" 🌀 分佈漂移警報 (PSI > 0.2)：{drifted_stocks} (強烈建議立即重訓)")
        if degraded_stocks:
            print(f" 📉 效能衰退警告：{degraded_stocks}")
    print("═"*80 + "\n")

if __name__ == "__main__":
    run_health_check()
