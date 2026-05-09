"""
check_db_locks.py v5.5.2 (Trinity Core Final)
================================================================================
系統維護工具 — 混合模式日誌實作版
負責資料庫健康檢查、API 配額監控與數據完整性維護。

修訂歷程：
  v5.5.2 (2026-05-09):
    - [規範] 導入混合模式日誌 (Category: sys)。
    - [核心] 對接 db_utils v4.7 與 DictCursor 相容性。

執行範例：
  python scripts/maintenance/check_db_locks.py
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

def check_locks():
    t0 = time.monotonic()
    logger.info("🕵️ 正在掃描資料庫鎖定狀態與異常查詢...")
    
    lock_count = 0
    try:
        with db_transaction() as cur:
            cur.execute("""
                SELECT count(*) as cnt FROM pg_stat_activity 
                WHERE wait_event_type IS NOT NULL AND state = 'active';
            """)
            res = cur.fetchone()
            lock_count = res['cnt'] if res else 0
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("db_lock_check", "SYSTEM", "success", "sys", elapsed_ms, lock_count)
        logger.info(f"✅ 掃描完成，目前活躍鎖定數: {lock_count}")
        
    except Exception as e:
        logger.error(f"❌ 鎖定掃描失敗: {e}")
        write_pipeline_log("db_lock_check", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    check_locks()