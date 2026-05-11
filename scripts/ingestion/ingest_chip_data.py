"""
fetch_chip_data.py v4.8 (Quantum Finance Edition)
================================================================================
三大法人籌碼抓取器 — 核心籌碼專項 (Quantum v5.2 標準)
負責同步個股的外資、投信、自營商買賣超數據，具備自動日期偏移與抓取儀表板。

修訂歷程：
  v4.8 (2026-05-11): [標準化] 導入 record_lifecycle、抓取儀表板與更健壯的自癒路徑。
  v4.7 (2026-05-10): [核心] 實作基礎混合模式日誌。

執行範例 (Comprehensive Usage Examples):
  1. [個股同步] 同步台積電(2330)最新籌碼數據:
     python scripts/ingestion/fetch_chip_data.py --stock_id 2330

  2. [強制更新] 重新抓取 2330 自 2010 年起的歷史籌碼:
     python scripts/ingestion/fetch_chip_data.py --stock_id 2330 --force

  3. [自定義區間] 抓取 2024 年至今的數據:
     python scripts/ingestion/fetch_chip_data.py --stock_id 2330 --start_date 2024-01-01

  4. [批量執行] 配合 parallel_fetch 執行核心股全量更新:
     python scripts/ingestion/parallel_fetch.py --universe core --table institutional_investors_buy_sell
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
    from core.db_utils import write_pipeline_log, bulk_upsert, get_latest_date, write_data_audit_log, record_lifecycle
    from core.finmind_client import FinMindClient
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from db_utils import write_pipeline_log, bulk_upsert, get_latest_date, write_data_audit_log, record_lifecycle
    from finmind_client import FinMindClient

logger = logging.getLogger(__name__)

def show_fetch_dashboard(stock_id: str, rows: int, duration: int):
    """抓取任務後的摘要儀表板。"""
    print("\n" + "="*50)
    print(f"🤝 Quantum Finance: 籌碼數據同步報告 ({stock_id})")
    print("="*50)
    print(f"✅ 同步狀態  : 成功")
    print(f"📥 寫入筆數  : {rows} 筆")
    print(f"⏱️  執行耗時  : {duration} ms")
    print("-" * 50)
    print("📝 日誌寫入: pipeline_execution_log & data_audit_log")
    print("="*50 + "\n")

def sync_chip(stock_id: str, start_date: str = None):
    """同步三大法人籌碼數據。"""
    TABLE_NAME = "institutional_investors_buy_sell"
    t0_all = time.monotonic()
    
    # 整合生命週期監測
    with record_lifecycle("fetch_chip", category="ingestion", stock_id=stock_id):
        client = FinMindClient()
        
        if not start_date:
            last_date = get_latest_date(TABLE_NAME, stock_id) or "2010-01-01"
            start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        logger.info(f"🤝 正在同步 {stock_id} 籌碼數據 (起始日: {start_date})...")
        
        try:
            raw_data = client.get_data("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_date)
            if raw_data:
                df = pd.DataFrame(raw_data)
                df.columns = [c.lower() for c in df.columns]
                # 執行 Upsert
                rows = bulk_upsert(TABLE_NAME, df.to_dict('records'), unique_cols=["date", "stock_id", "name"])
                
                # 分類審計紀錄
                write_data_audit_log(TABLE_NAME, stock_id, start_date, datetime.now().strftime("%Y-%m-%d"), rows)
                
                duration = int((time.monotonic() - t0_all) * 1000)
                show_fetch_dashboard(stock_id, rows, duration)
                return rows
            else:
                logger.info(f"ℹ️  {stock_id} 查無新資料。")
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
    sync_chip(args.stock_id, args.start_date if not args.force else "2010-01-01")