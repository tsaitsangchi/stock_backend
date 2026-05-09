"""
pipeline_monitor.py v5.5.3 (Trinity Core Final)
================================================================================
管線生命週期監測器 — 混合模式日誌實作版
偵測失敗任務並具備警報擴展介面。
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

def send_alert(task_name: str, err: str):
    """
    發送警報介面 (可擴展至 LINE/Telegram/Email)。
    """
    logger.critical(f"🚨 [ALERT] Task '{task_name}' failed: {err}")

def monitor_pipeline():
    t0 = time.monotonic()
    logger.info("📡 正在掃描全系統任務狀態...")
    
    try:
        with db_transaction() as cur:
            cur.execute("""
                SELECT task_name, error_msg 
                FROM pipeline_execution_log 
                WHERE status = 'failed' AND created_at > NOW() - INTERVAL '10 minutes';
            """)
            failures = cur.fetchall()
            
        for f in failures:
            send_alert(f['task_name'], f['error_msg'])
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("pipeline_failure_scan", "SYSTEM", "success", "sys", elapsed_ms, len(failures))
        
    except Exception as e:
        logger.error(f"❌ 監測失敗: {e}")
        write_pipeline_log("pipeline_failure_scan", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    monitor_pipeline()
