"""
fetch_stock_info.py v4.3 (Quantum Finance Edition)
================================================================================
市場標的抓取器 — 基礎定義專項 (Quantum v5.1 標準)
負責同步全市場股票的基本資訊（名稱、代號、產業別、上市日期）。

修訂歷程：
  v4.3 (2026-05-10): [核心] 實作混合模式日誌：同步寫入 pipeline_execution_log 與專項審計。
  v4.2 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.1 (2026-05-10): [修復] 對齊 db_utils v5.1 混合日誌規範。

【執行範例矩陣 — 數據抓取方案】
1. 全市場標的資訊同步 (Python)：
   python scripts/ingestion/fetch_stock_info.py
2. 單一標的「所有」維度表格抓取 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --stock_id 2330 --table ALL
3. 核心標的集「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe core --table ALL
4. 核心標的集「所有」維度表格「強制」更新 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe core --table ALL --force
5. 全市場標的「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe all --table ALL
================================================================================
"""
import os, sys, logging, time, argparse
import pandas as pd
from datetime import datetime
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
    from core.db_utils import write_pipeline_log, bulk_upsert, write_data_audit_log
    from core.finmind_client import FinMindClient
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import write_pipeline_log, bulk_upsert, write_data_audit_log
    from finmind_client import FinMindClient

logger = logging.getLogger(__name__)

def sync_stock_info():
    start_time = time.time()
    client = FinMindClient()
    TABLE_NAME = "stocks"
    
    logger.info("🏢 正在同步市場標的資訊...")
    
    try:
        raw_data = client.get_data("TaiwanStockInfo", "", "")
        if raw_data:
            df = pd.DataFrame(raw_data)
            df.columns = [c.lower() for c in df.columns]
            rows = bulk_upsert(TABLE_NAME, df.to_dict('records'), unique_cols=["stock_id"])
            duration = int((time.time() - start_time) * 1000)
            
            # 🔴 混合模式日誌
            write_pipeline_log("FetchStockInfo", "MARKET", "SUCCESS", "Ingestion", duration_ms=duration, rows=rows)
            write_data_audit_log(TABLE_NAME, "MARKET", "1900-01-01", datetime.now().strftime("%Y-%m-%d"), rows)
            
            logger.info(f"✅ 市場標的資訊同步完成，筆數: {rows}")
        else:
            logger.info("ℹ️  無新資料")
    except Exception as e:
        logger.error(f"❌ 市場標的資訊同步失敗: {e}")
        write_pipeline_log("FetchStockInfo", "MARKET", "FAILED", "Ingestion", err=str(e))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_stock_info()