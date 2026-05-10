"""
fetch_fundamental_data.py v4.4 (Quantum Finance Edition)
================================================================================
綜合財務報表抓取器 — 財報核心 (Quantum v5.1 標準)
負責同步個股的損益表、資產負債表等核心財務數據。

修訂歷程：
  v4.4 (2026-05-10): [核心] 實作混合模式日誌：同步寫入 pipeline_execution_log 與專項審計。
  v4.3 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.2 (2026-05-10): [修復] 對齊 db_utils v5.1 混合日誌規範。

【執行範例矩陣 — 數據抓取方案】
1. 單一標的、單一表格同步 (Python)：
   python scripts/ingestion/fetch_fundamental_data.py --stock_id 2330
2. 單一標的、單一表格「強制」更新歷史 (Python)：
   python scripts/ingestion/fetch_fundamental_data.py --stock_id 2330 --force
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
    from core.db_utils import write_pipeline_log, bulk_upsert, get_latest_date, write_data_audit_log
    from core.finmind_client import FinMindClient
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import write_pipeline_log, bulk_upsert, get_latest_date, write_data_audit_log
    from finmind_client import FinMindClient

logger = logging.getLogger(__name__)

def sync_fundamental(stock_id: str, start_date: str = None):
    start_time = time.time()
    client = FinMindClient()
    TABLE_NAME = "financial_statements"
    
    if not start_date:
        last_date = get_latest_date(TABLE_NAME, stock_id) or "2010-01-01"
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"📊 正在同步 {stock_id} 綜合財務報表 (Since: {start_date})...")
    
    try:
        raw_data = client.get_data("TaiwanStockFinancialStatements", stock_id, start_date)
        if raw_data:
            df = pd.DataFrame(raw_data)
            df.columns = [c.lower() for c in df.columns]
            rows = bulk_upsert(TABLE_NAME, df.to_dict('records'), unique_cols=["stock_id", "date", "type", "origin_name"])
            duration = int((time.time() - start_time) * 1000)
            
            # 🔴 混合模式日誌
            write_pipeline_log("FetchFundamental", stock_id, "SUCCESS", "Ingestion", duration_ms=duration, rows=rows)
            write_data_audit_log(TABLE_NAME, stock_id, start_date, datetime.now().strftime("%Y-%m-%d"), rows)
            
            logger.info(f"✅ {stock_id} 財報同步完成，筆數: {rows}")
        else:
            logger.info(f"ℹ️  {stock_id} 無新資料")
    except Exception as e:
        logger.error(f"❌ {stock_id} 同步失敗: {e}")
        write_pipeline_log("FetchFundamental", stock_id, "FAILED", "Ingestion", err=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, default="2330")
    parser.add_argument("--start_date", type=str)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_fundamental(args.stock_id, args.start_date if not args.force else "2010-01-01")