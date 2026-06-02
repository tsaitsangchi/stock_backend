"""
ingest_advanced_chip_data.py v5.1 (Quantum Finance Edition)
================================================================================
融資融券數據抓取器 — 籌碼硬核版 (Quantum v5.2 標準)
負責同步個股的融資買進、賣出、融券買進、賣出等信用交易數據。

修訂歷程：
  v5.1 (2026-05-11): [修復] 實作 camel_to_snake 轉換器，修正資料庫欄位不匹配問題。
  v5.0 (2026-05-11): [標準化] 更名為 ingest_、導入 record_lifecycle 與數據入庫儀表板。

執行範例 (Comprehensive Usage Examples):
  1. [個股單表同步] 同步台積電(2330)最新融資融券數據:
     python scripts/ingestion/ingest_advanced_chip_data.py --stock_id 2330

  2. [單個股全表同步] (透過編排器) 同步 2330 的所有 Ingestion 任務:
     python scripts/ingestion/parallel_ingestion.py --stock_id 2330 --table ALL

  3. [核心股遍歷同步] (透過編排器) 同步所有活躍核心標的的進階籌碼:
     python scripts/ingestion/parallel_ingestion.py --universe core --table AdvancedChip

  4. [強制重灌全量歷史] 強制回刷 2330 自 2010 年起的歷史數據:
     python scripts/ingestion/ingest_advanced_chip_data.py --stock_id 2330 --force
================================================================================
"""
import os, sys, logging, time, argparse, re
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
    from core.db_utils import write_pipeline_log, bulk_upsert, get_latest_date, record_lifecycle, write_data_audit_log, ensure_infrastructure
    from core.finmind_client import FinMindClient
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import write_pipeline_log, bulk_upsert, get_latest_date, record_lifecycle, write_data_audit_log, ensure_infrastructure
    from finmind_client import FinMindClient

logger = logging.getLogger(__name__)

def camel_to_snake(name: str) -> str:
    """將 CamelCase 轉換為 snake_case。"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def show_ingest_dashboard(stock_id: str, rows: int, duration: int):
    """入庫任務後的摘要儀表板。"""
    print("\n" + "="*50)
    print(f"💎 Quantum Finance: 融資融券數據同步報告 ({stock_id})")
    print("="*50)
    print(f"✅ 同步狀態  : 成功 (欄位已自動映射)")
    print(f"📥 寫入筆數  : {rows} 筆")
    print(f"⏱️  執行耗時  : {duration} ms")
    print("-" * 50)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log")
    print("="*50 + "\n")

def sync_advanced_chip(stock_id: str, start_date: str = None):
    """同步融資融券資料。"""
    TABLE_NAME = "margin_purchase_short_sale"
    t0_all = time.monotonic()
    
    # 執行資料庫自癒，確保表格與欄位存在
    ensure_infrastructure()
    
    with record_lifecycle("ingest_advanced_chip", category="ingestion", stock_id=stock_id):
        client = FinMindClient()
        
        if not start_date:
            last_date = get_latest_date(TABLE_NAME, stock_id) or "2010-01-01"
            start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        logger.info(f"💎 正在同步 {stock_id} 融資融券資料 (Since: {start_date})...")
        
        try:
            raw_data = client.get_data("TaiwanStockMarginPurchaseShortSale", stock_id, start_date)
            if raw_data:
                df = pd.DataFrame(raw_data)
                # 核心修正：將 CamelCase 轉換為 snake_case
                df.columns = [camel_to_snake(c) for c in df.columns]
                
                # 執行 Upsert
                rows = bulk_upsert(TABLE_NAME, df.to_dict('records'), unique_cols=["date", "stock_id"])
                
                # 分類審計紀錄
                write_data_audit_log(TABLE_NAME, stock_id, start_date, datetime.now().strftime("%Y-%m-%d"), rows)
                
                duration = int((time.monotonic() - t0_all) * 1000)
                show_ingest_dashboard(stock_id, rows, duration)
                return rows
            else:
                logger.info(f"ℹ️  {stock_id} 無新資料")
                return 0
        except Exception as e:
            logger.error(f"❌ {stock_id} 同步失敗: {e}")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, default="2330")
    parser.add_argument("--start_date", type=str)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_advanced_chip(args.stock_id, args.start_date if not args.force else "2010-01-01")