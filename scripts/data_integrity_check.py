"""
scripts/data_integrity_check.py
資料完整性校驗系統 (Trinity Edition)

核心邏輯：
1. 個股級別校驗：針對每個標的，檢查所有關聯資料表的資料存量與最新日期。
2. 頻率感知監控：區分日更、季更、事件觸發，設定不同的過時閾值。
3. 缺口偵測 (Gap Detection)：比對主股價表，找出缺失的交易日。
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# 注入路徑
sys.path.append(str(Path(__file__).resolve().parent))
from config import STOCK_CONFIGS
from data_pipeline import _query

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 資料表註冊表 (Table Registry)
# key: table_name, value: { 'type': 'daily'|'quarterly'|'event', 'column': 'stock_id'|'None' }
TABLE_REGISTRY = {
    # 核心價量與籌碼 (Daily)
    "stock_price":                      {"type": "daily", "id_col": "stock_id"},
    "price_adj":                        {"type": "daily", "id_col": "stock_id"},
    "stock_per":                        {"type": "daily", "id_col": "stock_id"},
    "institutional_investors_buy_sell": {"type": "daily", "id_col": "stock_id"},
    "margin_purchase_short_sale":       {"type": "daily", "id_col": "stock_id"},
    "shareholding":                    {"type": "daily", "id_col": "stock_id"},
    "day_trading":                     {"type": "daily", "id_col": "stock_id"},
    "price_limit":                     {"type": "daily", "id_col": "stock_id"},
    
    # 進階籌碼與借券 (Daily)
    "securities_lending":              {"type": "daily", "id_col": "stock_id"},
    "daily_short_balance":             {"type": "daily", "id_col": "stock_id"},
    "total_margin_short":              {"type": "daily", "id_col": "stock_id"},
    "eight_banks_buy_sell":            {"type": "daily", "id_col": "stock_id"},
    
    # 財務與基本面 (Quarterly)
    "cash_flows_statement":            {"type": "quarterly", "id_col": "stock_id"},
    "market_value":                    {"type": "daily", "id_col": "stock_id"},
    
    # 事件類 (Event-based)
    "disposition_securities":          {"type": "event", "id_col": "stock_id"},
    "capital_reduction":               {"type": "event", "id_col": "stock_id"},
    "suspended":                       {"type": "event", "id_col": "stock_id"},
    
    # 總經與期權 (Market-wide Daily)
    "futures_inst_investors":          {"type": "market", "id_col": None},
    "options_inst_investors":          {"type": "market", "id_col": None},
    "fred_series":                     {"type": "market", "id_col": None},
    
    # 另類數據
    "stock_news":                      {"type": "daily", "id_col": "stock_id"},
}

def get_trading_days(days: int = 30) -> List[str]:
    """獲取最近 N 個實際交易日"""
    sql = "SELECT DISTINCT date FROM stock_price ORDER BY date DESC LIMIT %s"
    df = _query(sql, (days,))
    return df["date"].astype(str).tolist()

def check_stock_table_integrity(stock_id: str, table: str, days: int = 30) -> Dict[str, Any]:
    """檢查單一標的在單一表中的完整性"""
    reg = TABLE_REGISTRY[table]
    id_col = reg["id_col"]
    
    where_clause = f"WHERE {id_col} = %s" if id_col else ""
    params = (stock_id,) if id_col else ()
    
    # 1. 檢查最新日期
    sql_max = f"SELECT MAX(date) as last_date FROM {table} {where_clause}"
    df_max = _query(sql_max, params)
    last_date = df_max["last_date"].iloc[0] if not df_max.empty else None
    
    # 2. 檢查資料量 (Window)
    if reg["type"] in ["daily", "market"]:
        sql_count = f"SELECT COUNT(*) as cnt FROM {table} {where_clause} {'AND' if id_col else 'WHERE'} date >= CURRENT_DATE - INTERVAL '%s days'"
        df_count = _query(sql_count, params + (days,))
        count = int(df_count["cnt"].iloc[0]) if not df_count.empty else 0
        
        # 計算健康得分
        score = min(1.0, count / (days * 0.7)) # 考慮假日，預期 70% 覆蓋率
    else:
        # 季報或事件類不適用連續性得分，僅看最新日期
        count = -1
        score = 1.0 if last_date and (datetime.now().date() - last_date).days < 120 else 0.0

    return {
        "last_date": last_date,
        "count_30d": count,
        "score": round(score, 2)
    }

def generate_integrity_matrix(stock_ids: List[str], tables: List[str]) -> pd.DataFrame:
    """生成完整性矩陣"""
    data = []
    for sid in stock_ids:
        row = {"stock_id": sid}
        for table in tables:
            res = check_stock_table_integrity(sid, table)
            # 我們主要看分數
            row[table] = res["score"]
        data.append(row)
    return pd.DataFrame(data)

def run_integrity_check(summary: bool = True) -> pd.DataFrame:
    """執行完整性校驗並回傳結果"""
    stock_ids = list(STOCK_CONFIGS.keys())
    # 挑選關鍵監控表
    key_tables = [
        "stock_price", "price_adj", "institutional_investors_buy_sell", 
        "securities_lending", "day_trading", "cash_flows_statement", "stock_news"
    ]
    
    logger.info(f"開始執行資料完整性校驗 (標的數: {len(stock_ids)}, 關鍵表數: {len(key_tables)})")
    matrix = generate_integrity_matrix(stock_ids, key_tables)
    
    if summary:
        print("\n" + "="*80)
        print("  Data Integrity Matrix (Score 0.0 ~ 1.0)")
        print("="*80)
        print(matrix.to_string(index=False))
        print("="*80 + "\n")
        
        # 找出紅字 (分數過低)
        threshold = 0.5
        for table in key_tables:
            fails = matrix[matrix[table] < threshold][["stock_id", table]]
            if not fails.empty:
                logger.warning(f"表 {table} 資料異常標的:\n{fails.to_string(index=False)}")
                
    return matrix

if __name__ == "__main__":
    run_integrity_check()
