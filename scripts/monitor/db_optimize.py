"""
db_optimize.py v5.5.3 (Trinity Core Final)
================================================================================
資料庫優化引擎 — 混合模式日誌實作版
執行真實的 VACUUM 與 ANALYZE 以優化 PostgreSQL 效能。
"""

import sys
import logging
import time
import psycopg2
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
    from core.db_utils import write_pipeline_log, get_connection_params
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def optimize_db():
    t0 = time.monotonic()
    logger.info("⚙️ 啟動資料庫物理層優化 (VACUUM ANALYZE)...")
    
    conn = None
    try:
        # VACUUM 不能在事務塊內執行，必須使用 autocommit
        params = get_connection_params()
        conn = psycopg2.connect(**params)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            logger.info("  .. 正在執行 VACUUM ANALYZE pipeline_execution_log")
            cur.execute("VACUUM ANALYZE pipeline_execution_log;")
            logger.info("  .. 正在執行 VACUUM ANALYZE stock_price")
            cur.execute("VACUUM ANALYZE stock_price;")
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("db_optimize", "SYSTEM", "success", "sys", elapsed_ms, 2)
        logger.info("✅ 資料庫優化完成。")
        
    except Exception as e:
        logger.error(f"❌ 優化失敗: {e}")
        write_pipeline_log("db_optimize", "SYSTEM", "failed", "sys", 0, 0, str(e))
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    optimize_db()
