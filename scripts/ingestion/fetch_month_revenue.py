"""
fetch_month_revenue.py v6.4 (Trinity Core Final)
================================================================================
資料抓取模組 — 營收面 (每月營收)
負責同步 FinMind 每月營收數據 (TaiwanStockMonthRevenue) 至資料庫。

修訂歷程：
  v6.4 (2026-05-10): [核心] 導入終極路徑自癒 Bootstrap。
  v6.3 (2026-05-10): [功能] 支援 start 參數，對齊強制更新矩陣。

【執行範例矩陣 — 營收同步方案】
1. 單一個股同步： $ python scripts/ingestion/fetch_month_revenue.py --stock_id 2330
2. 全核心股同步： $ python scripts/ingestion/fetch_month_revenue.py --all
3. 強制日期重刷： $ python scripts/ingestion/fetch_month_revenue.py --stock_id 2330 --start 2010-01-01
================================================================================
"""
import sys, logging, time, argparse
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS = None
for p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent, _THIS_DIR.parent.parent.parent]:
    if p.name == "scripts" or (p / "scripts").exists():
        _SCRIPTS = p if p.name == "scripts" else (p / "scripts")
        break
if _SCRIPTS:
    if str(_SCRIPTS) not in sys.path: sys.path.insert(0, str(_SCRIPTS))
    if str(_SCRIPTS.parent) not in sys.path: sys.path.insert(0, str(_SCRIPTS.parent))

try:
    from core.db_utils import write_pipeline_log, get_latest_date, get_db_stock_ids, bulk_upsert
    from core.finmind_client import FinMindClient
except ImportError:
    from db_utils import write_pipeline_log, get_latest_date, get_db_stock_ids, bulk_upsert
    from finmind_client import FinMindClient

def fetch_month_revenue(stock_id: str, start: str = None):
    t0 = time.monotonic(); api = FinMindClient()
    last_date = start or get_latest_date("month_revenue", stock_id) or "2010-01-01"
    logging.info(f"💰 正在同步 {stock_id} 營收數據 (Since: {last_date})...")
    data = api.get_data("TaiwanStockMonthRevenue", stock_id, last_date)
    rows = bulk_upsert("month_revenue", data, ["date", "stock_id"]) if data else 0
    write_pipeline_log("fetch_month_revenue", stock_id, "success" if data is not None else "failed", "ingestion", int((time.monotonic()-t0)*1000), rows)
    return rows

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(); parser.add_argument("--stock_id", type=str); parser.add_argument("--all", action="store_true"); parser.add_argument("--start", type=str); args = parser.parse_args()
    if args.all:
        for sid in get_db_stock_ids(): fetch_month_revenue(sid, args.start)
    else: fetch_month_revenue(args.stock_id or "2330", args.start)
