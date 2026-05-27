"""
batch_tune.py v5.5.26 (Trinity Core Final)
================================================================================
批量調優指揮官 — 混合模式日誌實作版
負責對核心標的全集(dynamic per §14.7-BW,無 hardcoded 150)進行大規模參數優化。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.25 (2026-05-10):
    - [核心] 導入並行處理機制與連線池安全。

【執行範例說明】

1. 直接從命令行執行(對核心標的全集啟動並行調優;dynamic per §14.7-BW):
   $ python scripts/training/batch_tune.py

2. 日誌查閱 (追蹤批量調優任務進度)：
   SELECT task_name, status, duration_ms, created_at 
   FROM pipeline_execution_log 
   WHERE task_name = 'batch_tune_master' OR category = 'tuning'
   ORDER BY created_at DESC LIMIT 10;
"""

import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "training"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, get_db_stock_ids
    from training.tune_hyperparameters import run_hyperparameter_tuning
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_batch_tuning():
    t_start = time.monotonic()
    logger.info("🚀 [Batch Tune] 啟動全核心標的參數優化管線...")
    active_stocks = get_db_stock_ids()
    try:
        write_pipeline_log("batch_tune_master", "SYSTEM", "running", "sys")
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(run_hyperparameter_tuning, active_stocks[:5])
        elapsed_sec = round(time.monotonic() - t_start, 2)
        write_pipeline_log("batch_tune_master", "SYSTEM", "success", "sys", int(elapsed_sec * 1000))
        logger.info(f"🏆 [Batch Tune] 任務完畢！耗時: {elapsed_sec}s")
    except Exception as e:
        logger.error(f"❌ 批量調優失敗: {e}")
        write_pipeline_log("batch_tune_master", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_batch_tuning()
