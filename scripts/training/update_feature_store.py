"""
update_feature_store.py v5.5.26 (Trinity Core Final)
================================================================================
特徵庫更新器 — 混合模式日誌實作版
負責在訓練前更新 150 檔標的的特徵矩陣與時序資料。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.25 (2026-05-10):
    - [核心] 對接 db_utils (v4.13) 與 path_setup (v3.0)。

【執行範例說明】

1. 直接從命令行執行（更新全核心特徵）：
   $ python scripts/training/update_feature_store.py

2. 日誌查閱 (確認特徵同步是否成功)：
   SELECT task_name, stock_id, status, duration_ms 
   FROM pipeline_execution_log 
   WHERE task_name = 'feature_store_sync' OR category = 'feature'
   ORDER BY created_at DESC LIMIT 10;
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "features"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
    from features.run_feature_engineering import run_feature_engineering_pipeline
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def update_all_features():
    t_start = time.monotonic()
    logger.info("🧬 [Feature] 啟動全核心特徵庫同步任務...")
    try:
        write_pipeline_log("feature_store_sync", "SYSTEM", "running", "sys")
        run_feature_engineering_pipeline()
        elapsed_sec = round(time.monotonic() - t_start, 2)
        write_pipeline_log("feature_store_sync", "SYSTEM", "success", "sys", int(elapsed_sec * 1000))
        logger.info(f"🏆 [Feature] 特徵庫同步完畢！耗時: {elapsed_sec}s")
    except Exception as e:
        logger.error(f"❌ 特徵同步失敗: {e}")
        write_pipeline_log("feature_store_sync", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    update_all_features()
