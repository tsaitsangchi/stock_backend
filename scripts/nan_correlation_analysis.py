"""
scripts/nan_correlation_analysis.py
資料完整性 vs 特徵缺失值 關聯分析工具 (Trinity Edition)

解決風險：
8. 資料完整性 vs 特徵 NaN 率沒有關聯分析。
   - 建立「資料缺失」到「模型失效」的因果鏈條。
   - 幫助量化分析師判斷：某個特徵失效是因為 fetch 沒抓好，還是 business 邏輯問題。
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 注入路徑
sys.path.append(str(Path(__file__).resolve().parent))
from config import STOCK_CONFIGS, TABLE_REGISTRY, OUTPUT_DIR
from data_integrity_check import run_trinity_audit
from data_pipeline import _query

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_feature_nan_rates(stock_id: str, days: int = 250) -> pd.Series:
    """獲取特定標的的特徵 NaN 比率"""
    # 這裡模擬從資料庫讀取已計算好的特徵，或直接計算關鍵欄位
    # 為了演示，我們檢查關鍵表的關鍵欄位
    tables = ["stock_price", "institutional_investors_buy_sell", "month_revenue", "financial_statements"]
    nan_counts = {}
    
    for table in tables:
        sql = f"SELECT * FROM {table} WHERE stock_id = %s ORDER BY date DESC LIMIT %s"
        df = _query(sql, (stock_id, days))
        if df.empty:
            nan_counts[table] = 1.0
        else:
            # 計算所有非 date/stock_id 欄位的平均 NaN 率
            cols = [c for c in df.columns if c not in ["date", "stock_id"]]
            nan_rate = df[cols].isna().mean().mean()
            nan_counts[table] = nan_rate
            
    return pd.Series(nan_counts)

def analyze_correlation():
    """執行關聯分析"""
    logger.info("正在執行全系統完整性審計...")
    integrity_matrix = run_trinity_audit(summary=False)
    
    # 將狀態標記轉為分數
    def status_to_score(s):
        if "✅" in str(s): return 1.0
        if "🔴" in str(s): return 0.7
        if "⚠️" in str(s): return 0.4
        return 0.0
    
    stock_ids = integrity_matrix["stock_id"].tolist()
    tables = [c for c in integrity_matrix.columns if c != "stock_id"]
    
    integrity_scores = integrity_matrix.copy()
    for t in tables:
        integrity_scores[t] = integrity_matrix[t].apply(status_to_score)
        
    logger.info("正在計算特徵 NaN 率...")
    nan_results = []
    for sid in stock_ids:
        nan_rates = get_feature_nan_rates(sid)
        nan_rates["stock_id"] = sid
        nan_results.append(nan_rates)
        
    df_nan = pd.DataFrame(nan_results)
    
    # 關聯分析 (Integrity Score vs 1 - NaN Rate)
    # 我們預期：Integrity Score 越高，NaN Rate 越低
    correlations = {}
    for t in ["stock_price", "institutional_investors_buy_sell", "month_revenue"]:
        if t in integrity_scores.columns and t in df_nan.columns:
            x = integrity_scores[t]
            y = 1.0 - df_nan[t] # 1 - NaN Rate = Valid Rate
            corr = np.corrcoef(x, y)[0, 1]
            correlations[t] = corr
            
    print("\n" + "="*80)
    print("  Correlation: Data Integrity Score vs Feature Validity (1 - NaN Rate)")
    print("="*80)
    for t, c in correlations.items():
        print(f"{t:<40} Correlation: {c:.4f}")
    print("="*80 + "\n")
    
    if any(c < 0.8 for c in correlations.values()):
        logger.warning("🚩 注意：部分資料表的完整性得分與特徵有效率關聯度低，可能存在隱性斷層或資料清洗邏輯問題。")

if __name__ == "__main__":
    analyze_correlation()
