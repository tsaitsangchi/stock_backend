"""
dashboard.py v5.5.2 (Trinity Core Final)
================================================================================
系統狀況儀表板 — 混合模式日誌實作版
從 pipeline_execution_log 提取摘要，產出每日運行健康報告。
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

def generate_report():
    t0 = time.monotonic()
    logger.info("📊 正在產出資料管線運行報告 (Daily Insights)...")
    
    try:
        with db_transaction() as cur:
            cur.execute("""
                SELECT status, count(*) as cnt 
                FROM pipeline_execution_log 
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY status;
            """)
            summary = {r['status']: r['cnt'] for r in cur.fetchall()}
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("daily_dashboard", "SYSTEM", "success", "sys", elapsed_ms, sum(summary.values()))
        logger.info(f"✅ 報告產出完成: {summary}")
        
    except Exception as e:
        logger.error(f"❌ 報告產出失敗: {e}")
        write_pipeline_log("daily_dashboard", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    generate_report()
