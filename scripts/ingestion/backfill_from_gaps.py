"""
backfill_from_gaps.py v6.0 (Trinity Core Final)
================================================================================
數據斷層修復指揮官 — 混合模式日誌標準版
負責自動檢測資料庫中的交易日斷層，並針對核心標的啟動精準補齊。

修訂歷程：
  v6.0 (2026-05-10):
    - [核心] 升級至 Trinity Core v6.0 標準，優化混合日誌紀錄。
  v5.5.7 (2026-05-09):
    - [核心] 重構對接 Hybrid Logging 分類體系。

【執行範例說明】
1. 啟動全系統數據空隙補齊任務 (自動掃描所有標的之時間缺失)：
   $ python scripts/ingestion/backfill_from_gaps.py
================================================================================
"""

import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── 系統路徑修復 (v3.1) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "pipeline", "ingestion"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, get_db_stock_ids
    # 引用單元抓取函數
    from ingestion.parallel_fetch import fetch_stock_data_unit
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_backfill_process():
    """
    執行全系統斷層檢查與補齊。
    
    執行範例：
    $ python scripts/ingestion/backfill_from_gaps.py
    """
    t_start = time.monotonic()
    logger.info("🚀 [Trinity] 啟動全核心標的斷層檢查 (v6.0)...")
    
    active_stocks = get_db_stock_ids()
    if not active_stocks:
        logger.warning("⚠️ 查無核心標的，請先執行 migrate_stocks_config.py")
        return

    # 使用並行執行緒加速補齊
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(fetch_stock_data_unit, active_stocks)
    
    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    
    # 🔴 混合日誌紀錄 (Category: maintenance)
    write_pipeline_log(
        task_name="backfill_gap_master",
        stock_id="SYSTEM",
        status="success",
        category="maintenance",
        duration_ms=elapsed_ms,
        rows=len(active_stocks)
    )
    
    logger.info(f"🏆 [Master] 斷層檢查與補齊完畢！耗時: {elapsed_ms}ms")

if __name__ == "__main__":
    run_backfill_process()