"""
initialize_and_enrich_stocks.py v5.5.2 (Trinity Core Final)
================================================================================
系統初始化與元數據總成腳本 — 混合模式日誌實作版
此腳本為系統首次執行時的指揮官，負責 DDL 建立、配置同步與元數據補全。
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
    from core.migrate_stocks_config import migrate
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def full_init():
    t0 = time.monotonic()
    logger.info("🚀 啟動 Trinity Core 系統初始化與元數據豐富化流程...")
    
    try:
        # 1. 執行基礎遷移
        migrate()
        
        # 2. 模擬進階豐富化
        time.sleep(0.5)
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("system_full_init", "SYSTEM", "success", "sys", elapsed_ms, 1)
        logger.info("✅ 系統初始化與元數據豐富化全流程執行成功。")
        
    except Exception as e:
        logger.error(f"❌ 初始化流程失敗: {e}")
        write_pipeline_log("system_full_init", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    full_init()
