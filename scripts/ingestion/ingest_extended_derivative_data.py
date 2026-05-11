"""
fetch_extended_derivative_data.py v4.3 (Quantum Finance Edition)
================================================================================
擴展衍生品抓取器 — 選擇權專項 (Quantum v5.1 標準)
負責同步台指選擇權的日線 (OHLCV) 數據。

修訂歷程：
  v4.3 (2026-05-10): [核心] 實作混合模式日誌：同步寫入 pipeline_execution_log 與專項審計。
  v4.2 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.1 (2026-05-10): [修復] 對齊 db_utils v5.1 混合日誌規範。

【執行範例矩陣 — 數據抓取方案】
1. 市場指標、單一表格同步 (Python)：
   python scripts/ingestion/fetch_extended_derivative_data.py
2. 市場指標、單一表格「強制」更新歷史 (Python)：
   python scripts/ingestion/fetch_extended_derivative_data.py --start_date 2024-01-01
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

def sync_ext_deriv(start_date: str = None):
    start_time = time.time()
    client = FinMindClient()
    TABLE_NAME = "options_ohlcv"
    
    if not start_date:
        last_date = get_latest_date(TABLE_NAME) or "2010-01-01"
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"🌀 正在同步擴展期權資料 (Since: {start_date})...")
    
    try:
        raw_data = client.get_data("TaiwanOptionDaily", "", start_date)
        if raw_data:
            df = pd.DataFrame(raw_data)
            df.columns = [c.lower() for c in df.columns]
            unique_cols = ["date", "option_id", "contract_date", "strike_price", "call_put", "trading_session"]
            rows = bulk_upsert(TABLE_NAME, df.to_dict('records'), unique_cols=unique_cols)
            duration = int((time.time() - start_time) * 1000)
            
            # 🔴 混合模式日誌
            write_pipeline_log("FetchExtDeriv", "MARKET", "SUCCESS", "Ingestion", duration_ms=duration, rows=rows)
            write_data_audit_log(TABLE_NAME, "MARKET", start_date, datetime.now().strftime("%Y-%m-%d"), rows)
            
            logger.info(f"✅ 擴展期權同步完成，筆數: {rows}")
        else:
            logger.info(f"ℹ️  擴展期權無新資料")
    except Exception as e:
        logger.error(f"❌ 擴展期權同步失敗: {e}")
        write_pipeline_log("FetchExtDeriv", "MARKET", "FAILED", "Ingestion", err=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_ext_deriv(args.start_date)