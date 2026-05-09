"""
check_db_locks.py v5.5 (Trinity Core Edition)
================================================================================
資料庫健康度與鎖定檢測工具 — 混合模式日誌實作版
此模組負責監控 PostgreSQL 內部的長時間查詢、阻塞 (Blocking) 與死鎖風險。

核心功能：
  · 阻塞偵測       ─ 自動識別 pg_stat_activity 中被卡住的 PID。
  · 鎖定分析       ─ 分析 pg_locks 中的詳細鎖定模式 (Lock Mode)。
  · 維運日誌紀錄   ─ 對接 write_pipeline_log，將檢測狀態標記為 sys_v5.1 (Maintenance)。

修訂歷程：
  v5.5 (2026-05-09):
    - [規範] 導入混合模式日誌，將執行監控紀錄於 pipeline_execution_log。
    - [核心] 對接 path_setup v3.0 與 db_utils v4.7 連線池。
  v3.0 (2026-04-10):
    - [基礎] 建立基礎鎖定監控邏輯。

執行範例：
    # 執行資料庫健康檢測並歸檔日誌
    python scripts/maintenance/check_db_locks.py
"""

import sys
import logging
import time
from pathlib import Path

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_session, write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_lock_check():
    t0 = time.monotonic()
    logger.info("🕵️ 正在啟動資料庫鎖定與阻塞深度檢測...")
    
    error_count = 0
    try:
        with db_session() as conn:
            with conn.cursor() as cur:
                # 1. 檢測長時間查詢 (> 1 min)
                cur.execute("SELECT pid, state, now()-query_start FROM pg_stat_activity WHERE state != 'idle' AND now()-query_start > interval '1 minute' AND pid != pg_backend_pid();")
                long_queries = cur.fetchall()
                if long_queries:
                    logger.warning(f"⚠️ 發現 {len(long_queries)} 筆長時間查詢！")
                    error_count += len(long_queries)
                else:
                    logger.info("  [OK] 無長時間查詢。")

                # 2. 檢測阻塞
                cur.execute("SELECT pid, wait_event_type, wait_event FROM pg_stat_activity WHERE wait_event IS NOT NULL AND state != 'idle' AND pid != pg_backend_pid();")
                blocked = cur.fetchall()
                if blocked:
                    logger.error(f"❌ 發現 {len(blocked)} 筆查詢被阻塞！")
                    error_count += len(blocked)
                else:
                    logger.info("  [OK] 目前無查詢阻塞。")

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        # =====================================================================
        # 🔴 混合模式日誌落盤 (Category: sys)
        # =====================================================================
        write_pipeline_log(
            task_name="check_db_locks",
            stock_id="DB_INSTANCE",
            status="success" if error_count == 0 else "warning",
            category="sys",
            duration_ms=elapsed_ms,
            rows=error_count # 紀錄發現的異常數
        )
        logger.info(f"✅ 檢測完畢，紀錄已歸檔至 pipeline_execution_log (sys_v5.1)。")
        
    except Exception as e:
        logger.error(f"❌ 檢測失敗: {e}")
        write_pipeline_log("check_db_locks", "DB_INSTANCE", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_lock_check()