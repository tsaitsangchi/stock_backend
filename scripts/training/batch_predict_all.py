"""
batch_predict_all.py v5.5.25 (Trinity Core Final)
================================================================================
批量預測指揮官 (Training 目錄版)
負責一次性產出核心標的全集(dynamic per §14.7-BW,無 hardcoded 150)之預測訊號。

修訂歷程：
  v5.5.25 (2026-05-10):
    - [對接] 引用 models.parallel_inference 核心邏輯。
    - [規範] 完全對接混合日誌 (Category: sys)。
"""

import sys
import logging
from pathlib import Path

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "models"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    # 引用 models 核心
    from models.parallel_inference import run_full_inference
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 [Predict] 從 Training 目錄啟動全核心批量預測...")
    run_full_inference()
    logger.info("🏆 [Predict] 預測任務產出完畢。")
