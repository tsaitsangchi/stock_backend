"""
parallel_fetch.py v6.5 (Trinity Core Final)
================================================================================
數據抓取單元指揮官 — 混合模式日誌標準版
負責單一標的全量數據 (價格, 籌碼, 基本面, 營收) 的整合同步邏輯。

修訂歷程：
  v6.5 (2026-05-10): [修正] 強化對 fetch_chip 參數傳遞的嚴謹性。
  v6.4 (2026-05-10): [核心] 導入終極路徑自癒 Bootstrap。

【執行範例矩陣 — 數據同步方案】
1. 個股 x 所有表： $ python scripts/ingestion/parallel_fetch.py --stock_id 2330
2. 強制深層同步： $ python scripts/ingestion/parallel_fetch.py --stock_id 2330 --start 2010-01-01
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
    from core.db_utils import write_pipeline_log, get_db_stock_ids
    from ingestion.fetch_technical_data import fetch_tech
    from ingestion.fetch_chip_data import fetch_chip
    from ingestion.fetch_fundamental_data import fetch_fundamental
    from ingestion.fetch_month_revenue import fetch_month_revenue
except ImportError:
    from db_utils import write_pipeline_log, get_db_stock_ids
    from fetch_technical_data import fetch_tech
    from fetch_chip_data import fetch_chip
    from fetch_fundamental_data import fetch_fundamental
    from fetch_month_revenue import fetch_month_revenue

def fetch_stock_data_unit(stock_id: str, start_date: str = None):
    """ 執行單一標的之多維度數據補齊任務 """
    t0 = time.monotonic()
    logging.info(f"🚀 [Fetch] 正在同步 {stock_id} 多維度數據 (Start: {start_date})...")
    total_rows = 0; success_count = 0
    fetchers = [("Price", fetch_tech), ("Chip", fetch_chip), ("Fundamental", fetch_fundamental), ("Revenue", fetch_month_revenue)]
    for name, func in fetchers:
        try:
            # 💡 確保傳遞 start 參數以對齊強制更新邏輯
            rows = func(stock_id, start=start_date)
            total_rows += (rows or 0); success_count += 1
        except Exception as e: logging.error(f"  - [{name}] 失敗: {e}")
    write_pipeline_log("stock_data_unit_master", stock_id, "success" if success_count>0 else "failed", "ingestion", int((time.monotonic()-t0)*1000), total_rows)
    return total_rows

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(); parser.add_argument("--stock_id", type=str, default="2330"); parser.add_argument("--start", type=str); args = parser.parse_args()
    fetch_stock_data_unit(args.stock_id, args.start)
