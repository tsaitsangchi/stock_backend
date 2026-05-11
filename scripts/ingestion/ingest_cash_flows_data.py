"""
fetch_cash_flows_data.py v4.7 (Quantum Finance Edition)
================================================================================
現金流量表抓取器 — 財務品質專項 (Quantum v5.1 標準)
負責同步個股的現金流量表數據，用於評估營運品質。

修訂歷程：
  v4.7 (2026-05-10): [核心] 實作混合模式日誌：同步寫入 pipeline_execution_log 與專項審計。
  v4.6 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.5 (2026-05-10): [修復] 對齊 db_utils v5.1 混合日誌規範。
  v4.4 (2026-05-10): [修復] 實作 Fallback 資料集備援機制。

【執行範例矩陣 — 數據抓取方案】
1. 單一標的、單一表格同步 (Python)：
   python scripts/ingestion/fetch_cash_flows_data.py --stock_id 2330
2. 單一標的、單一表格「強制」更新歷史 (Python)：
   python scripts/ingestion/fetch_cash_flows_data.py --stock_id 2330 --force
3. 單一標的「所有」維度表格抓取 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --stock_id 2330 --table ALL
4. 核心標的集「所有」維度表格「強制」更新 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe core --table ALL --force
5. 全市場標的「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe all --table ALL
================================================================================
"""
import os, sys, logging, time, argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

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
    # 🔴 v4.7: 引入 write_data_audit_log 以支援混合模式
    from core.db_utils import write_pipeline_log, bulk_upsert, get_latest_date, write_data_audit_log
    from core.finmind_client import FinMindClient
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import write_pipeline_log, bulk_upsert, get_latest_date, write_data_audit_log
    from finmind_client import FinMindClient

logger = logging.getLogger(__name__)

def sync_cash_flows(stock_id: str, start_date: str = None):
    start_time = time.time()
    client = FinMindClient()
    TABLE_NAME = "cash_flows_statement"
    
    if not start_date:
        last_date = get_latest_date(TABLE_NAME, stock_id) or "2010-01-01"
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"💰 正在同步 {stock_id} 現金流量表 (Since: {start_date})...")
    
    try:
        # 1. 嘗試主要資料集
        data = client.get_data("TaiwanStockCashFlows", stock_id, start_date)
        
        # 2. 備援機制 (Self-Healing)
        if not data:
            logger.warning(f"🟡 {stock_id} 嘗試從 TaiwanStockFinancialStatements 獲取現金流...")
            data = client.get_data("TaiwanStockFinancialStatements", stock_id, start_date)
        
        if data:
            df = pd.DataFrame(data)
            df.columns = [c.lower() for c in df.columns]
            
            # 🔴 原子操作：寫入數據
            rows = bulk_upsert(TABLE_NAME, df.to_dict('records'), unique_cols=["date", "stock_id", "type"])
            duration = int((time.time() - start_time) * 1000)
            
            # 🔴 v4.7 混合模式日誌 (Hybrid Logging)
            # A. 生命週期日誌 (統一管線監控)
            write_pipeline_log("FetchCashFlows", stock_id, "SUCCESS", "Ingestion", duration_ms=duration, rows=rows)
            
            # B. 專項數據審計日誌 (分類記錄)
            write_data_audit_log(TABLE_NAME, stock_id, start_date, datetime.now().strftime("%Y-%m-%d"), rows)
            
            logger.info(f"✅ {stock_id} 同步完成，筆數: {rows}")
        else:
            logger.info(f"ℹ️  {stock_id} 無新資料")
            
    except Exception as e:
        logger.error(f"❌ {stock_id} 同步失敗: {e}")
        write_pipeline_log("FetchCashFlows", stock_id, "FAILED", "Ingestion", err=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, default="2330")
    parser.add_argument("--start_date", type=str)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_cash_flows(args.stock_id, args.start_date if not args.force else "2010-01-01")