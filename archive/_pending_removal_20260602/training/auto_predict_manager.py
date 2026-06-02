"""
auto_predict_manager.py v5.5.26 (Trinity Core Final)
================================================================================
推論自動化總管 — 混合模式日誌實作版
負責排程每日核心標的全集(dynamic per §14.7-BW,無 hardcoded 150)之預測訊號生成與結果驗證。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.25 (2026-05-10):
    - [核心] 對接 db_utils (v4.13) 與 path_setup (v3.0)。

【執行範例說明】

1. 直接從命令行執行(產出明日核心標的全集預測訊號;dynamic per §14.7-BW):
   $ python scripts/training/auto_predict_manager.py

2. 日誌查閱 (追蹤推論成功率)：
   SELECT task_name, status, rows_processed, duration_ms 
   FROM pipeline_execution_log 
   WHERE category = 'inference' OR task_name = 'auto_predict_manager_run'
   ORDER BY created_at DESC LIMIT 10;

3. 預測結果查閱 (查看明天的買賣建議)：
   SELECT stock_id, signal, confidence, pred_price 
   FROM predictions 
   ORDER BY confidence DESC LIMIT 10;
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
    from models.parallel_inference import run_full_inference
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_daily_inference_orchestration():
    t_start = time.monotonic()
    logger.info("🚀 [Inference] 啟動全核心標的預測訊號生成...")
    try:
        write_pipeline_log("auto_predict_manager_run", "SYSTEM", "running", "sys")
        run_full_inference()
        elapsed_sec = round(time.monotonic() - t_start, 2)
        write_pipeline_log("auto_predict_manager_run", "SYSTEM", "success", "sys", int(elapsed_sec * 1000))
        logger.info(f"🏆 [Manager] 預測管線執行成功！耗時: {elapsed_sec}s")
    except Exception as e:
        logger.error(f"❌ 預測管線故障: {e}")
        write_pipeline_log("auto_predict_manager_run", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_daily_inference_orchestration()
