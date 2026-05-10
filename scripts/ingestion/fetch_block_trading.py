"""
fetch_block_trading.py v6.0 (Trinity Core Final)
================================================================================
資料抓取模組 — 混合模式日誌標準版
負責將 FinMind 原始數據 (TaiwanStockBlockTrading) 同步至資料庫。

修訂歷程：
  v6.0 (2026-05-10):
    - [核心] 建立 Trinity Core v6.0 標準抓取腳本，對應 block_trading 表。

【執行範例說明】
1. 手動抓取特定標的鉅額交易 (例如 台積電 2330)：
   $ python scripts/ingestion/fetch_block_trading.py --stock_id 2330
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

def fetch_block_trading(stock_id: str):
    """
    抓取特定標的之鉅額交易數據。
    
    執行範例：
    $ python scripts/ingestion/fetch_block_trading.py --stock_id 2330
    """
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：block_trading
    last_date = get_latest_date("block_trading", stock_id) or "2010-01-01"
    
    logger.info(f"🧱 正在同步 {stock_id} 鉅額交易數據 (Since: {last_date})...")
    data = api.get_data("TaiwanStockBlockTrading", stock_id, start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    # 🔴 混合日誌紀錄 (Category: ingestion)
    write_pipeline_log(
        task_name="fetch_block_trading",
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
    fetch_block_trading(args.stock_id)
