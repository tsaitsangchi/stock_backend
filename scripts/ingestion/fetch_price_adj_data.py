"""
fetch_price_adj_data.py v6.0 (Trinity Core Final)
================================================================================
資料抓取模組 — 混合模式日誌標準版
負責將 FinMind 原始數據 (TaiwanStockPriceAdj) 同步至資料庫。

修訂歷程：
  v6.0 (2026-05-10):
    - [核心] 升級至 Trinity Core v6.0 標準，確保 price_adj 表對齊。
  v5.5.7 (2026-05-09):
    - [文檔] 補齊「大規模並行調度」與「手動單點調試」執行範例。

【執行範例說明】
1. 手動抓取特定標的還原價格 (例如 台積電 2330)：
   $ python scripts/ingestion/fetch_price_adj_data.py --stock_id 2330
================================================================================
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.1) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "pipeline"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, get_latest_date
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fetch_adj(stock_id: str):
    """
    抓取特定標的之還原價格數據。
    
    執行範例：
    $ python scripts/ingestion/fetch_price_adj_data.py --stock_id 2330
    """
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：price_adj
    last_date = get_latest_date("price_adj", stock_id) or "2010-01-01"
    
    logger.info(f"🔄 正在同步 {stock_id} 還原價格資料 (Since: {last_date})...")
    data = api.get_data("TaiwanStockPriceAdj", stock_id, start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    # 🔴 混合日誌紀錄 (Category: ingestion)
    write_pipeline_log(
        task_name="fetch_price_adj",
        stock_id=stock_id,
        status="success" if data is not None else "failed",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_id", type=str, default="2330")
    args = parser.parse_args()
    fetch_adj(args.stock_id)