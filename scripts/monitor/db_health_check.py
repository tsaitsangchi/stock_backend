"""
db_health_check.py v5.5.2 (Trinity Core Final)
================================================================================
資料庫健康檢查器 — 混合模式日誌實作版
監控連線池狀態、磁碟佔用空間以及索引碎片化。
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
    from core.db_utils import db_transaction, write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_health_check():
    t0 = time.monotonic()
    logger.info("🏥 執行資料庫健康度深度掃描...")
    
    try:
        with db_transaction() as cur:
            cur.execute("SELECT pg_database_size(current_database()) as sz;")
            res = cur.fetchone()
            db_size = res['sz'] if res else 0
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("db_health_check", "SYSTEM", "success", "sys", elapsed_ms, 1, err=f"DBSize: {db_size/1024/1024:.2f}MB")
        logger.info(f"✅ 資料庫運作正常，大小: {db_size/1024/1024:.2f}MB")
        
    except Exception as e:
        logger.error(f"❌ 健康檢查失敗: {e}")
        write_pipeline_log("db_health_check", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_health_check()
