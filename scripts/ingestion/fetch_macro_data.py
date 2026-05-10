"""
fetch_macro_data.py v4.3 (Quantum Finance Edition)
================================================================================
台灣宏觀數據抓取器 — 景氣循環專項 (Quantum v5.1 標準)
負責同步台灣景氣對策信號、領先指標等關鍵宏觀數據。

修訂歷程：
  v4.3 (2026-05-10): [核心] 實作混合模式日誌：同步寫入 pipeline_execution_log 與專項審計。
  v4.2 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.1 (2026-05-10): [修復] 對齊 db_utils v5.1 混合日誌規範。

【執行範例矩陣 — 數據抓取方案】
1. 景氣指標、單一表格同步 (Python)：
   python scripts/ingestion/fetch_macro_data.py
2. 景氣指標、單一表格「強制」更新歷史 (Python)：
   python scripts/ingestion/fetch_macro_data.py --start_date 2010-01-01
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

def sync_macro(start_date: str = None):
    start_time = time.time()
    client = FinMindClient()
    TABLE_NAME = "business_indicator"
    
    if not start_date:
        last_date = get_latest_date(TABLE_NAME) or "2010-01-01"
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"🇹🇼 正在同步台灣景氣指標 (Since: {start_date})...")
    
    try:
        raw_data = client.get_data("TaiwanEconomicIndicators", "", start_date)
        if raw_data:
            df = pd.DataFrame(raw_data)
            df.columns = [c.lower() for c in df.columns]
            rows = bulk_upsert(TABLE_NAME, df.to_dict('records'), unique_cols=["date"])
            duration = int((time.time() - start_time) * 1000)
            
            # 🔴 混合模式日誌
            write_pipeline_log("FetchMacro", "TW", "SUCCESS", "Ingestion", duration_ms=duration, rows=rows)
            write_data_audit_log(TABLE_NAME, "TW", start_date, datetime.now().strftime("%Y-%m-%d"), rows)
            
            logger.info(f"✅ 台灣景氣指標同步完成，筆數: {rows}")
        else:
            logger.info(f"ℹ️  無新資料")
    except Exception as e:
        logger.error(f"❌ 台灣景氣指標同步失敗: {e}")
        write_pipeline_log("FetchMacro", "TW", "FAILED", "Ingestion", err=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_macro(args.start_date)