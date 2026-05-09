"""
pipeline_monitor.py v5.5.2 (Trinity Core Final)
================================================================================
管線生命週期監測器 — 混合模式日誌實作版
即時偵測 pipeline_execution_log 中的 failed 狀態並發出告警。
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

def monitor_pipeline():
    t0 = time.monotonic()
    logger.info("📡 正在掃描近一小時內的異常任務...")
    
    try:
        with db_transaction() as cur:
            cur.execute("""
                SELECT count(*) as cnt 
                FROM pipeline_execution_log 
                WHERE status = 'failed' AND created_at > NOW() - INTERVAL '1 hour';
            """)
            fail_count = cur.fetchone()['cnt']
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("pipeline_failure_scan", "SYSTEM", "success", "sys", elapsed_ms, fail_count)
        
        if fail_count > 0:
            logger.warning(f"⚠️ 偵測到 {fail_count} 筆失敗任務，請檢查日誌！")
        else:
            logger.info("✅ 管線運作完美，無失敗紀錄。")
            
    except Exception as e:
        logger.error(f"❌ 監測失敗: {e}")
        write_pipeline_log("pipeline_failure_scan", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    monitor_pipeline()
