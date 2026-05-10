"""
fetch_extended_derivative_data.py v6.0 (Trinity Core Final)
================================================================================
資料抓取模組 — 混合模式日誌標準版
負責將 FinMind 原始數據 (TaiwanOptionDaily) 同步至資料庫。

修訂歷程：
  v6.0 (2026-05-10):
    - [核心] 升級至 Trinity Core v6.0 標準，確保 options_ohlcv 表對齊。
  v5.5.7 (2026-05-09):
    - [文檔] 補齊「大規模並行調度」與「手動單點調試」執行範例。

【執行範例說明】
1. 手動抓取擴展期權資料 (例如 台指選擇權)：
   $ python scripts/ingestion/fetch_extended_derivative_data.py
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

def fetch_ext_deriv():
    """
    抓取台股選擇權日資料。
    
    執行範例：
    $ python scripts/ingestion/fetch_extended_derivative_data.py
    """
    t0 = time.monotonic()
    api = FinMindClient()
    
    # 🔍 資料表對齊：options_ohlcv
    last_date = get_latest_date("options_ohlcv") or "2010-01-01"
    
    logger.info(f"🌀 正在同步擴展期權資料 (Since: {last_date})...")
    data = api.get_data("TaiwanOptionDaily", start_date=last_date)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    # 🔴 混合日誌紀錄 (Category: ingestion)
    write_pipeline_log(
        task_name="fetch_extended_derivative",
        stock_id="MARKET_WIDE",
        status="success" if data is not None else "failed",
        category="ingestion",
        duration_ms=elapsed_ms,
        rows=len(data)
    )
    return len(data)

if __name__ == "__main__":
    fetch_ext_deriv()