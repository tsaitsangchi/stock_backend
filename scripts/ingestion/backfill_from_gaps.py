"""
backfill_from_gaps.py v5.5.26 (Trinity Core Final)
================================================================================
數據斷層修復指揮官 — 混合模式日誌實作版
負責自動檢測資料庫中的交易日斷層，並針對 150 檔標的啟動精準補齊。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.15 (2026-05-09):
    - [核心] 實作全自動斷層偵測與並行補齊邏輯。

【執行範例說明】

1. 直接從命令行執行（啟動全自動斷層掃描與補齊）：
   $ python scripts/ingestion/backfill_from_gaps.py

2. 日誌查閱 (追蹤哪些標的補齊了多少天)：
   SELECT task_name, stock_id, status, rows_processed as gap_days, duration_ms 
   FROM pipeline_execution_log 
   WHERE task_name = 'backfill_gap_unit' 
   ORDER BY created_at DESC LIMIT 20;

3. 統計今日修復的數據總筆數：
   SELECT SUM(rows_processed) FROM pipeline_execution_log 
   WHERE task_name = 'backfill_gap_unit' AND status = 'success' AND created_at > CURRENT_DATE;
"""

import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── 系統路徑修復 (v3.0) ──
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
    # 引用原有的抓取函數
    from ingestion.parallel_fetch import fetch_stock_data_unit
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_backfill_process():
    t_start = time.monotonic()
    logger.info("🚀 [Trinity] 啟動全核心標的斷層檢查 (v5.5.26)...")
    active_stocks = get_db_stock_ids()
    if not active_stocks: return

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(fetch_stock_data_unit, active_stocks)
    
    # 3. 🔴 自動產出/更新數據戰情網頁 (Dashboard)
    try:
        from monitor.data_audit_engine import audit_completeness
        audit_completeness()
    except ImportError:
        logger.warning("⚠️ 無法載入稽核引擎，跳過網頁更新。")

    elapsed_sec = round(time.monotonic() - t_start, 2)
    write_pipeline_log("backfill_gap_master", "SYSTEM", "success", "sys", int(elapsed_sec * 1000))
    logger.info(f"🏆 [Master] 斷層檢查與補齊完畢！耗時: {elapsed_sec}s")
    logger.info(f"🌐 戰情網頁已同步更新: monitor/dashboard.html")

if __name__ == "__main__":
    run_backfill_process()