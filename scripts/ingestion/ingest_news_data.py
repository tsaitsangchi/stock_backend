"""
fetch_news_data.py v4.5 (Quantum Finance Edition)
================================================================================
個股新聞抓取器 — 輿情數據專項 (Quantum v5.1 標準)
負責同步個股的即時與歷史新聞，支援後續情緒分析管線。

修訂歷程：
  v4.5 (2026-05-10): [核心] 實作混合模式日誌：同步寫入 pipeline_execution_log 與專項審計。
  v4.4 (2026-05-10): [文件] 完善五維度執行範例矩陣，確保範例完整性。
  v4.3 (2026-05-10): [修復] 強制對標題 strip() 並標準化日期，解決 DB 衝突。
  v4.2 (2026-05-10): [修復] 在 Upsert 前進行 DataFrame 去重。

【執行範例矩陣 — 數據抓取方案】
1. 單一標的、單一表格同步 (Python)：
   python scripts/ingestion/fetch_news_data.py --stock_id 2330
2. 強制更新特定標的歷史新聞 (Python)：
   python scripts/ingestion/fetch_news_data.py --stock_id 2330 --start_date 2024-01-01
3. 單一標的「所有」維度表格抓取 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --stock_id 2330 --table ALL
4. 核心標的集「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe core --table News
5. 全市場標的「所有」維度表格同步 (透過編排器)：
   python scripts/ingestion/parallel_fetch.py --universe all --table News
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

def sync_news(stock_id: str, start_date: str = None):
    start_time = time.time()
    client = FinMindClient()
    TABLE_NAME = "stock_news"
    
    if not start_date:
        last_date = get_latest_date(TABLE_NAME, stock_id) or (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        start_date = last_date
    
    logger.info(f"📰 正在同步 {stock_id} 個股新聞 (Since: {start_date})...")
    
    try:
        raw_data = client.get_data("TaiwanStockNews", stock_id, start_date)
        if raw_data:
            df = pd.DataFrame(raw_data)
            df.columns = [c.lower() for c in df.columns]
            df['title'] = df['title'].astype(str).str.strip()
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df.drop_duplicates(subset=["date", "stock_id", "title"])
            
            rows = bulk_upsert(TABLE_NAME, df.to_dict('records'), unique_cols=["date", "stock_id", "title"])
            duration = int((time.time() - start_time) * 1000)
            
            # 🔴 混合模式日誌
            write_pipeline_log("FetchNews", stock_id, "SUCCESS", "Ingestion", duration_ms=duration, rows=rows)
            write_data_audit_log(TABLE_NAME, stock_id, start_date, datetime.now().strftime("%Y-%m-%d"), rows)
            
            logger.info(f"✅ {stock_id} 新聞同步完成，筆數: {rows}")
        else:
            logger.info(f"ℹ️  {stock_id} 無新資料")
    except Exception as e:
        logger.error(f"❌ {stock_id} 新聞同步失敗: {e}")
        write_pipeline_log("FetchNews", stock_id, "FAILED", "Ingestion", err=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, default="2330")
    parser.add_argument("--start_date", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_news(args.stock_id, args.start_date)