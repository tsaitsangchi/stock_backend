"""
auto_train_manager.py v5.5.26 (Trinity Core Final)
================================================================================
訓練自動化總管 — 混合模式日誌實作版
負責排程與監控核心標的全集(dynamic per §14.7-BW,無 hardcoded 150)之模型迭代週期。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明（含 CLI、Python 引用與 SQL 查閱）。
  v5.5.25 (2026-05-10):
    - [核心] 對接 db_utils (v4.13) 與 path_setup (v3.0)。
    - [規範] 實作混合模式日誌：統一管線紀錄 + 專項模型紀錄。

【執行範例說明】

1. 直接從命令行執行(啟動核心標的全集自動訓練;dynamic per §14.7-BW):
   $ python scripts/training/auto_train_manager.py

2. 在其他系統中作為定時任務 (Crontab) 呼叫：
   0 15 * * 1-5 /usr/bin/python3 /path/to/scripts/training/auto_train_manager.py

3. 日誌查閱 (驗證管線執行狀態)：
   SELECT * FROM pipeline_execution_log 
   WHERE task_name LIKE 'auto_train_manager%' 
   ORDER BY created_at DESC LIMIT 5;
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "models", "pipeline"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
    from models.parallel_train import run_parallel_training
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_daily_training_orchestration():
    t_start = time.monotonic()
    logger.info("🚀 [Training] 啟動全核心標的模型迭代管線...")
    try:
        write_pipeline_log("auto_train_manager_start", "SYSTEM", "running", "sys")
        success_count = run_parallel_training()
        elapsed_sec = round(time.monotonic() - t_start, 2)
        write_pipeline_log("auto_train_manager_finish", "SYSTEM", "success", "sys", int(elapsed_sec * 1000), success_count)
        logger.info(f"🏆 [Manager] 任務完畢！耗時: {elapsed_sec}s")
    except Exception as e:
        logger.error(f"❌ 訓練管線中斷: {e}")
        write_pipeline_log("auto_train_manager_failed", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_daily_training_orchestration()
