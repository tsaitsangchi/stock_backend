"""
scripts/fetchers/check_db_locks.py v2.0 (Integrity Edition)
=========================================================
負責檢測 PostgreSQL 資料庫中的長時間查詢與阻塞鎖定。
採用專案標準 core.path_setup 與 core.db_utils 進行整合。
"""

import sys
import logging
from pathlib import Path

# ── 專案環境初始化 ──
try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
except ImportError:
    # 意外情況下的手動路徑修復
    _THIS_DIR = Path(__file__).resolve().parent
    _SCRIPTS_DIR = _THIS_DIR.parent
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))

from core.db_utils import get_db_conn

# ── 設定日誌 ──
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def check_db_locks():
    """檢測資料庫中的長時間查詢與鎖定狀態。"""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # 1. 檢測長時間查詢 (Long Running Queries)
            logger.info("--- 執行中的長時間查詢 (超過 1 分鐘) ---")
            cur.execute("""
                SELECT pid, now() - query_start AS duration, query, state
                FROM pg_stat_activity
                WHERE state != 'idle' 
                  AND now() - query_start > interval '1 minute'
                  AND pid != pg_backend_pid()
                ORDER BY duration DESC;
            """)
            rows = cur.fetchall()
            if not rows:
                logger.info("  [OK] 未發現長時間執行的查詢。")
            else:
                for r in rows:
                    logger.warning(f"  PID: {r[0]} | 耗時: {r[1]} | 狀態: {r[3]}")
                    logger.warning(f"  查詢內容: {r[2][:250]}...")

            # 2. 檢測等待鎖定的查詢 (Blocked Queries)
            logger.info("\n--- 正在等待鎖定的查詢 (Blocked Queries) ---")
            cur.execute("""
                SELECT pid, wait_event_type, wait_event, query 
                FROM pg_stat_activity 
                WHERE wait_event IS NOT NULL 
                  AND state != 'idle'
                  AND pid != pg_backend_pid();
            """)
            rows = cur.fetchall()
            if not rows:
                logger.info("  [OK] 目前沒有查詢因等待鎖定而阻塞。")
            else:
                for r in rows:
                    logger.error(f"  PID: {r[0]} | 等待事件: {r[1]}/{r[2]}")
                    logger.error(f"  阻塞查詢: {r[3][:250]}...")

            # 3. 檢測詳細鎖定資訊 (Lock Details)
            logger.info("\n--- 詳細鎖定資訊 (pg_locks) ---")
            cur.execute("""
                SELECT 
                    a.datname, l.relation::regclass, l.transactionid, l.mode, l.granted,
                    a.query, a.query_start, age(now(), a.query_start) AS age, a.pid
                FROM pg_stat_activity a
                JOIN pg_locks l ON l.pid = a.pid
                WHERE a.pid != pg_backend_pid()
                  AND l.mode != 'AccessShareLock' -- 過濾一般的讀取鎖
                ORDER BY a.query_start;
            """)
            rows = cur.fetchall()
            if not rows:
                logger.info("  [OK] 未發現異常鎖定模式。")
            else:
                for r in rows:
                    logger.info(f"  DB: {r[0]} | Table: {r[1]} | Mode: {r[3]} | Granted: {r[4]}")
                    logger.info(f"  PID: {r[8]} | Age: {r[7]} | Query: {r[5][:150]}...")

    except Exception as e:
        logger.error(f"❌ 檢測資料庫鎖定時發生錯誤: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_db_locks()
