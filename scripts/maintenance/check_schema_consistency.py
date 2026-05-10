"""
check_schema_consistency.py v6.5 (Trinity Core Final)
================================================================================
系統維運模組 — 全域 Schema 一致性稽核與自動修復引擎
負責檢查所有外部數據來源 (FinMind & FRED) 以及系統核心表 (Core Tables) 是否符合設計規範。

修訂歷程：
  v6.5 (2026-05-10):
    - [核心] 優化 COLUMN_MISSING 修復邏輯：增加大小寫不敏感的欄位匹配。
    - [核心] 補齊修復失敗的偵錯日誌。
  v6.4 (2026-05-10):
    - [核心] 實作 COLUMN_MISSING 自動修補功能。

【執行範例說明】
1. 啟動全域稽核 (僅檢測，不修復)：
   $ python scripts/maintenance/check_schema_consistency.py

2. 啟動全域稽核並自動修復 (自癒模式)：
   $ python scripts/maintenance/check_schema_consistency.py --fix
================================================================================
"""

import sys
import logging
import time
import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Any

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "pipeline"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 稽核標準配置 ──
MIN_VARCHAR_LEN = 50
MIN_NUMERIC_PRECISION = 20
MIN_NUMERIC_SCALE = 6

DATASET_TABLE_MAP = {
    "TaiwanStockPrice": "stock_price",
    "TaiwanStockPER": "stock_per",
    "TaiwanStockFinancialStatements": "financial_statements",
    "TaiwanStockMonthRevenue": "month_revenue",
    "TaiwanStockDividend": "dividend",
    "TaiwanStockInstitutionalInvestorsBuySell": "institutional_investors_buy_sell",
    "TaiwanStockMarginPurchaseShortSale": "margin_purchase_short_sale",
    "TaiwanStockNews": "stock_news",
    "USStockPrice": "us_stock_price",
    "TaiwanStockPriceAdj": "price_adj",
    "TaiwanStockTotalReturnIndex": "total_return_index",
    "TaiwanStockBlockTrading": "block_trading",
    "TaiwanStockCashFlowsStatement": "cash_flows_statement",
    "TaiwanOptionPutCallRatio": "options_sentiment",
    "TaiwanOptionDaily": "options_ohlcv",
    "TaiwanFuturesDaily": "futures_ohlcv",
    "FedFundsRate": "fred_series",
}

CORE_SYSTEM_TABLES = [
    "stocks", "evaluation_log", "model_metadata", "pipeline_execution_log", "data_audit_log", "schema_audit_log"
]

def get_db_schema(table_name: str) -> Dict[str, Dict[str, Any]]:
    schema = {}
    query = "SELECT column_name, data_type, character_maximum_length, numeric_precision, numeric_scale FROM information_schema.columns WHERE table_name = %s"
    with db_transaction() as cur:
        cur.execute(query, (table_name,))
        for row in cur.fetchall():
            schema[row['column_name'].lower()] = row
    return schema

def get_finmind_sample_data(dataset: str) -> List[Dict]:
    api = FinMindClient()
    # 針對股利資料，擴大搜尋範圍以確保獲取完整欄位
    sample_id = "DGS10" if dataset == "FedFundsRate" else ("2330" if "Taiwan" in dataset else "TSM")
    if dataset in ["TaiwanOptionPutCallRatio", "TaiwanStockTotalReturnIndex"]: sample_id = None
    start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    try:
        data = api.get_data(dataset, data_id=sample_id, start_date=start_date)
        return data or []
    except: return []

def auto_fix_issue(table, f_type, col, detail, dataset=None):
    try:
        if f_type == "WEAK_TYPE":
            sql = None
            if "VARCHAR" in detail:
                sql = f"ALTER TABLE {table} ALTER COLUMN {col} TYPE VARCHAR({MIN_VARCHAR_LEN})"
            elif "NUMERIC" in detail:
                sql = f"ALTER TABLE {table} ALTER COLUMN {col} TYPE NUMERIC({MIN_NUMERIC_PRECISION}, {MIN_NUMERIC_SCALE})"
            if sql:
                with db_transaction() as cur: cur.execute(sql)
                logger.info(f"✅ [Fix] 已修復 {table}.{col}")
                return True
                
        elif f_type == "TABLE_MISSING" and dataset:
            data = get_finmind_sample_data(dataset)
            if not data: return False
            cols = []
            for k, v in data[0].items():
                name = k.lower()
                if name == "date": cols.append(f"{name} DATE")
                elif isinstance(v, (int, float)): cols.append(f"{name} NUMERIC({MIN_NUMERIC_PRECISION}, {MIN_NUMERIC_SCALE})")
                else: cols.append(f"{name} VARCHAR({MIN_VARCHAR_LEN})")
            sql = f"CREATE TABLE {table} ({', '.join(cols)}, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            with db_transaction() as cur: cur.execute(sql)
            logger.info(f"✅ [Fix] 已建立資料表 {table}")
            return True
            
        elif f_type == "COLUMN_MISSING" and dataset:
            data = get_finmind_sample_data(dataset)
            if not data: 
                logger.warning(f"⚠️ [Fix] 無法獲取 {dataset} 樣本數據進行補全")
                return False
                
            # 大小寫不敏感匹配
            target_v = None
            for k, v in data[0].items():
                if k.lower() == col.lower():
                    target_v = v
                    break
            
            if target_v is None:
                logger.warning(f"⚠️ [Fix] 樣本數據中未發現欄位 {col}")
                return False
            
            dtype = f"VARCHAR({MIN_VARCHAR_LEN})"
            if isinstance(target_v, (int, float)):
                dtype = f"NUMERIC({MIN_NUMERIC_PRECISION}, {MIN_NUMERIC_SCALE})"
            elif col.lower() == "date":
                dtype = "DATE"
                
            sql = f"ALTER TABLE {table} ADD COLUMN {col} {dtype}"
            with db_transaction() as cur: cur.execute(sql)
            logger.info(f"✅ [Fix] 已補齊欄位 {table}.{col}")
            return True
    except Exception as e:
        logger.error(f"❌ [Fix] 執行修復時發生錯誤 ({table}.{col}): {e}")
    return False

def perform_audit(fix_mode=False):
    t_master = time.monotonic()
    logger.info(f"🚀 [Audit] 啟動全域稽核任務 (v6.5, 修復模式: {fix_mode})...")
    
    total_findings = 0
    fixed_count = 0
    
    for dataset, table in DATASET_TABLE_MAP.items():
        t_unit = time.monotonic()
        db_schema = get_db_schema(table)
        findings = []
        if not db_schema:
            findings.append(("TABLE_MISSING", None, f"資料表 '{table}' 缺失"))
        else:
            sample_data = get_finmind_sample_data(dataset)
            fm_columns = [k.lower() for k in sample_data[0].keys()] if sample_data else []
            db_cols = set(db_schema.keys())
            for fm_col in fm_columns:
                if fm_col not in db_cols: findings.append(("COLUMN_MISSING", fm_col, f"缺失欄位: '{fm_col}'"))
            for col_name, info in db_schema.items():
                dtype = info['data_type']
                if 'character varying' in dtype:
                    length = info['character_maximum_length']
                    if length and length < MIN_VARCHAR_LEN: findings.append(("WEAK_TYPE", col_name, f"VARCHAR({length}) < {MIN_VARCHAR_LEN}"))
                elif 'numeric' in dtype:
                    prec = info['numeric_precision']; scale = info['numeric_scale']
                    if (prec and prec < MIN_NUMERIC_PRECISION) or (scale is not None and scale < MIN_NUMERIC_SCALE):
                        findings.append(("WEAK_TYPE", col_name, f"NUMERIC({prec},{scale}) < ({MIN_NUMERIC_PRECISION},{MIN_NUMERIC_SCALE})"))

        if findings:
            print(f"❌ {dataset} -> {table} 發現 {len(findings)} 項異常")
            for f_type, col, detail in findings:
                print(f"   - [{f_type}] {detail}")
                if fix_mode:
                    if auto_fix_issue(table, f_type, col, detail, dataset=dataset):
                        fixed_count += 1
            total_findings += len(findings)
        else:
            print(f"✅ {dataset} -> {table} 稽核通過")
        write_pipeline_log(f"audit_{dataset}", table, "success" if not findings else "warning", "maintenance", int((time.monotonic()-t_unit)*1000), len(findings))

    # 核心表稽核
    for table in CORE_SYSTEM_TABLES:
        db_schema = get_db_schema(table)
        findings = []
        if db_schema:
            for col_name, info in db_schema.items():
                if 'character varying' in info['data_type'] and info['character_maximum_length'] < MIN_VARCHAR_LEN:
                    findings.append(("WEAK_TYPE", col_name, f"VARCHAR({info['character_maximum_length']}) < {MIN_VARCHAR_LEN}"))
        if findings:
            print(f"❌ [Core] {table} 發現 {len(findings)} 項異常")
            if fix_mode:
                for f_type, col, detail in findings:
                    if auto_fix_issue(table, f_type, col, detail): fixed_count += 1
            total_findings += len(findings)
        else:
            print(f"✅ [Core] {table} 符合規範")

    elapsed = int((time.monotonic() - t_master) * 1000)
    write_pipeline_log("schema_audit_master", "SYSTEM", "success", "maintenance", elapsed, total_findings)
    logger.info(f"🏆 全域稽核完成！發現 {total_findings} 項，修復 {fixed_count} 項，耗時: {elapsed}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    perform_audit(fix_mode=args.fix)
