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
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# 注入路徑
sys.path.append("/home/hugo/project/stock_backend/scripts")
from config import MODEL_DIR, STOCK_CONFIGS
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
    # 只顯示異常個股或前 10 筆
    anomalies_df = df_models_df[df_models_df["status"] != "🟢 OK"]
    if not anomalies_df.empty:
        print(anomalies_df.to_string(index=False))
    else:
        print("✅ 所有配置個股模型皆已就緒且在有效期限內。")
    
    # 3. 最近實戰表現
    df_perf_df = evaluate_recent_performance_df(stock_ids)
    print("\n[3] 實戰準確度評估 (Real-time Directional Accuracy - 30D)")
    print("-" * 60)
    # 排序：優先顯示警告項
    df_perf_df = df_perf_df.sort_values("da", ascending=True)
    print(df_perf_df.head(15).to_string(index=False))
    
    # 總結警示
    stale_stocks = df_models_df[df_models_df["age_days"] > 30]["stock_id"].tolist()
    degraded_stocks = df_perf_df[df_perf_df["perf_status"] == "⚠️ DEGRADED"]["stock_id"].tolist()
    
    print("\n" + "═"*80)
    if not stale_stocks and not degraded_stocks:
        print(" 🎉 系統健康度：優良。所有模型與數據流運作正常。")
    else:
        if stale_stocks:
            print(f" ⏳ 模型過期警告：{stale_stocks} (建議執行 parallel_train.py)")
        if degraded_stocks:
            print(f" 📉 效能衰退警告：{degraded_stocks} (建議檢查特徵漂移或市場 Regime)")
    print("═"*80 + "\n")

if __name__ == "__main__":
    run_health_check()
