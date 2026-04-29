"""
scripts/data_integrity_check.py
資料完整性校驗系統 v2.0 (Trinity Edition)

修復風險：
1. 解決「二維問答」：提供 Stock x Table 矩陣。
2. 解決「日期連續性」：精確偵測日期空洞 (Gap Detection)。
3. 解決「資料表漏監控」：動態從 config.TABLE_REGISTRY 載入 20+ 張表。
4. 解決「應更新未更新」：結合 DATA_LAG_CONFIG 進行過時告警。
"""

import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# 注入路徑
sys.path.append(str(Path(__file__).resolve().parent))
from config import STOCK_CONFIGS, TABLE_REGISTRY, DATA_LAG_CONFIG
from data_pipeline import _query

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_expected_trading_days(days: int = 60) -> List[date]:
    """獲取台股預期交易日 (基準為 stock_price)"""
    sql = "SELECT DISTINCT date FROM stock_price ORDER BY date DESC LIMIT %s"
    df = _query(sql, (days,))
    return sorted(df["date"].tolist())

def detect_gaps(stock_id: str, table: str, expected_dates: List[date]) -> List[date]:
    """偵測特定標在特定表中的日期斷層"""
    reg = TABLE_REGISTRY[table]
    if reg["type"] != "daily":
        return [] # 季報/事件不適用連續性檢查
        
    id_col = reg["id_col"]
    where_clause = f"WHERE {id_col} = %s" if id_col else ""
    params = (stock_id,) if id_col else ()
    
    # 取得實際日期
    sql = f"SELECT date FROM {table} {where_clause} AND date >= %s"
    df_actual = _query(sql, params + (min(expected_dates),))
    actual_dates = set(df_actual["date"].tolist()) if not df_actual.empty else set()
    
    gaps = [d for d in expected_dates if d not in actual_dates]
    return gaps

def check_staleness(table: str, last_date: Optional[date]) -> str:
    """檢查資料是否過時"""
    if last_date is None:
        return "❌ 缺失"
    
    lag_limit = TABLE_REGISTRY[table]["lag"]
    # 財報類通常 lag 較大，這裡簡單處理
    diff = (datetime.now().date() - last_date).days
    
    if diff > lag_limit + 3: # 給 3 天寬限期
        return f"⚠️ 延遲({diff}d)"
    return "✅ 正常"

def get_integrity_report(stock_id: str) -> Dict[str, Any]:
    """產出單一標的的全面健康報告"""
    report = {"stock_id": stock_id}
    expected_60d = get_expected_trading_days(60)
    
    for table, meta in TABLE_REGISTRY.items():
        if meta["type"] not in ["daily", "monthly", "quarterly"]:
            continue
            
        id_col = meta["id_col"]
        where_clause = f"WHERE {id_col} = %s" if id_col else ""
        params = (stock_id,) if id_col else ()
        
        # 1. 查最新日期與總量
        sql = f"SELECT MAX(date) as last_date, COUNT(*) as total_cnt FROM {table} {where_clause}"
        df = _query(sql, params)
        last_date = df["last_date"].iloc[0] if not df.empty else None
        
        # 2. 狀態判斷
        status = check_staleness(table, last_date)
        
        # 3. 斷層偵測 (僅限日更)
        gaps = []
        if meta["type"] == "daily" and status != "❌ 缺失":
            gaps = detect_gaps(stock_id, table, expected_60d)
            if gaps:
                status = f"🔴 斷層({len(gaps)}d)"
        
        report[table] = status
        
    return report

def run_trinity_audit():
    """執行全系統二維審計"""
    stock_ids = list(STOCK_CONFIGS.keys())
    # 為了畫面整潔，挑選最核心的 10 張表做 Matrix 顯示，其餘做背後檢查
    display_tables = [
        "stock_price", "institutional_investors_buy_sell", "margin_purchase_short_sale",
        "shareholding", "securities_lending", "month_revenue", "financial_statements",
        "stock_news"
    ]
    
    logger.info(f"🚀 Trinity 資料審計啟動：標的={len(stock_ids)}, 監控表={len(TABLE_REGISTRY)}")
    
    results = []
    for sid in stock_ids:
        results.append(get_integrity_report(sid))
    
    df_results = pd.DataFrame(results)
    
    # 輸出矩陣
    print("\n" + "═"*100)
    print(f"║ {'Trinity Data Integrity Matrix (Snapshot)':^96} ║")
    print("═"*100)
    # 格式化輸出
    matrix_cols = ["stock_id"] + [t for t in display_tables if t in df_results.columns]
    print(df_results[matrix_cols].to_string(index=False))
    print("═"*100 + "\n")
    
    # 總結風險
    total_gaps = df_results.apply(lambda x: x.str.contains("🔴").sum()).sum()
    total_stale = df_results.apply(lambda x: x.str.contains("⚠️").sum()).sum()
    total_missing = df_results.apply(lambda x: x.str.contains("❌").sum()).sum()
    
    if total_gaps + total_stale + total_missing > 0:
        logger.error(f"🚩 發現關鍵風險：斷層={total_gaps}, 延遲={total_stale}, 缺失={total_missing}")
    else:
        logger.info("✅ 全系統資料完整性良好。")
        
    return df_results

if __name__ == "__main__":
    run_trinity_audit()
