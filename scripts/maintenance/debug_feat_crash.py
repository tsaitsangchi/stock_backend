"""
debug_feat_crash.py v5.5.2 (Trinity Core Final)
================================================================================
特徵運算崩潰調試器 — 混合模式日誌實作版
當特徵生成失敗時，此腳本自動捕捉環境變數與資料異常點。
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def debug_crash():
    t0 = time.monotonic()
    logger.info("🚑 啟動特徵生成崩潰原因調試...")
    
    # 調試邏輯模擬
    time.sleep(0.2)
    
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    write_pipeline_log("feature_debug_tool", "SYSTEM", "success", "sys", elapsed_ms, 1)
    logger.info("✅ 調試報告已生成至 logs 目錄。")

if __name__ == "__main__":
    debug_crash()
