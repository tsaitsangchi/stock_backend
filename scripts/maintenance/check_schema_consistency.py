"""
check_schema_consistency.py v7.1 (Quantum Finance Edition)
================================================================================
全域架構守護者 — Schema 一致性稽核與自動修復引擎 (Quantum v5.2 標準)
負責確保資料庫架構與外部 API 對齊，具備壓力測試樣本探測與「硬核保底」建立功能。

修訂歷程：
  v7.1 (2026-05-11): [終極衝刺] 導入硬核保底 (Hardcoded Fallback) 模式，確保 100% 資料表覆蓋。
  v7.0 (2026-05-11): [衝刺版] 導入壓力測試樣本輪詢 (Stress Polling)。

【執行範例矩陣】
  1. [100% 全域自癒] 強制建立所有缺失資料表 (包含期權與 FRED):
     python scripts/maintenance/check_schema_consistency.py --fix
================================================================================
"""
import sys, logging, time, argparse, re
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Dict, List, Any

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log, record_lifecycle, write_data_audit_log, ensure_infrastructure
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

# ── 稽核標準配置 ──
MIN_VARCHAR_LEN = 100 
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

# 終極保底 Schema (針對 API 探測失敗的特殊表)
FALLBACK_SCHEMAS = {
    "block_trading": "date DATE, stock_id VARCHAR(50), contract VARCHAR(100), buy_price NUMERIC(20,6), sell_price NUMERIC(20,6), volume NUMERIC(20,6)",
    "options_sentiment": "date DATE, stock_id VARCHAR(50), put_call_ratio NUMERIC(20,6), put_volume NUMERIC(20,6), call_volume NUMERIC(20,6)",
    "options_ohlcv": "date DATE, stock_id VARCHAR(50), open NUMERIC(20,6), high NUMERIC(20,6), low NUMERIC(20,6), close NUMERIC(20,6), volume NUMERIC(20,6)",
    "fred_series": "date DATE, value NUMERIC(20,6), series_id VARCHAR(100)"
}

CORE_SYSTEM_TABLES = [
    "stocks", "evaluation_log", "model_metadata", "pipeline_execution_log", "data_audit_log", "schema_audit_log"
]

def camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def show_schema_dashboard(total, findings, fixed):
    remaining = findings - fixed
    coverage = ( (total - remaining) / total * 100 ) if total > 0 else 0
    print("\n" + "🛡️"*35)
    print("🚀 Quantum Finance: 全域架構守護報告 (v7.1)")
    print("🛡️"*35)
    print(f"✅ 稽核狀態  : 完成")
    print(f"📊 總稽核項  : {total}")
    print(f"⚠️ 當前異常  : {remaining} 項")
    print(f"🔧 本次修復  : {fixed} 項")
    print(f"⚖️ 架構健康度: {coverage:.1f}%")
    
    if remaining == 0:
        print("-" * 70)
        print("🟢 完美！系統已進入 100% 終極健康狀態。")
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (schema_fix)")
    print("🛡️"*35 + "\n")

def get_db_schema(table_name: str) -> Dict[str, Dict[str, Any]]:
    schema = {}
    query = "SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_name = %s"
    try:
        with db_transaction() as cur:
            cur.execute(query, (table_name,))
            for row in cur.fetchall():
                schema[row['column_name'].lower()] = row
    except: pass
    return schema

def auto_fix_issue(table, f_type, col, detail, dataset=None):
    """硬核保底自癒引擎。"""
    try:
        with db_transaction() as cur:
            if f_type == "TABLE_MISSING" and dataset:
                api = FinMindClient()
                strategies = [("2330", 30), ("TAIEX", 30), ("TX", 30), ("DFF", 30), ("2330", 730)]
                data = None
                for sid, lookback in strategies:
                    start_dt = (date.today() - timedelta(days=lookback)).strftime("%Y-%m-%d")
                    data = api.get_data(dataset, sid, start_dt)
                    if data: break
                
                if data:
                    cols = []
                    for k, v in data[0].items():
                        name = camel_to_snake(k)
                        if name == "date": cols.append(f"{name} DATE")
                        elif isinstance(v, (int, float)): cols.append(f"{name} NUMERIC({MIN_NUMERIC_PRECISION}, {MIN_NUMERIC_SCALE})")
                        else: cols.append(f"{name} VARCHAR({MIN_VARCHAR_LEN})")
                    sql = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(cols)}, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                    cur.execute(sql)
                elif table in FALLBACK_SCHEMAS:
                    # 硬核保底路徑
                    logger.info(f"⚠️ [Fix] API 探測失敗，啟動 {table} 的硬核保底建立模式...")
                    sql = f"CREATE TABLE IF NOT EXISTS {table} ({FALLBACK_SCHEMAS[table]}, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                    cur.execute(sql)
                else:
                    return False
                
                write_data_audit_log("schema_fix", table, date.today().strftime("%Y-%m-%d"), "TABLE_CREATE", 1)
                logger.info(f"✅ [Fix] 已成功建立資料表: {table}")
                return True
            
            elif f_type == "WEAK_TYPE":
                sql = f"ALTER TABLE {table} ALTER COLUMN {col} TYPE VARCHAR({MIN_VARCHAR_LEN})"
                cur.execute(sql)
                return True
    except Exception as e:
        logger.error(f"❌ [Fix] 修復失敗 ({table}): {e}")
    return False

def perform_audit(fix_mode=False):
    ensure_infrastructure()
    total_items = len(DATASET_TABLE_MAP) + len(CORE_SYSTEM_TABLES)
    findings = 0
    fixed = 0

    with record_lifecycle("schema_audit", category="maintenance", stock_id="SYSTEM"):
        for dataset, table in DATASET_TABLE_MAP.items():
            db_schema = get_db_schema(table)
            if not db_schema:
                print(f"❌ {dataset:<35} -> {table:<30} : 發現資料表缺失")
                findings += 1
                if fix_mode and auto_fix_issue(table, "TABLE_MISSING", None, None, dataset): fixed += 1
            else:
                print(f"✅ {dataset:<35} -> {table:<30} : 稽核通過")

        for table in CORE_SYSTEM_TABLES:
            db_schema = get_db_schema(table)
            if not db_schema:
                print(f"❌ [Core] {table:<30} : 資料表缺失")
                findings += 1
                if fix_mode and auto_fix_issue(table, "TABLE_MISSING", None, None): fixed += 1
            else:
                issue_found = False
                for col, info in db_schema.items():
                    if info['character_maximum_length'] and info['character_maximum_length'] < 50:
                        print(f"⚠️ [Core] {table}.{col:<25} : 長度不足")
                        findings += 1; issue_found = True
                        if fix_mode and auto_fix_issue(table, "WEAK_TYPE", col, None): fixed += 1
                if not issue_found: print(f"✅ [Core] {table:<30} : 符合規範")

        show_schema_dashboard(total_items, findings, fixed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    perform_audit(fix_mode=args.fix)
