"""
check_db_locks.py v3.0 (Infrastructure Monitoring Edition)
量化系統核心：資料庫鎖定與效能瓶頸檢測工具
================================================================================
v3.0 重大升級：
  · 完美對接 db_utils v4.0：全面採用 @contextmanager (db_session) 進行安全的唯讀查詢，
    確保受惠於連線池機制，並避免監控腳本本身成為消耗連線或產生「髒連線」的元凶。
  · 整合 path_setup v3.0：標準化路徑載入機制，支援環境變數 TRINITY_ROOT 的生產環境部署。

執行範例：
    # 手動檢測當前資料庫的鎖定與阻塞狀況 (若一切正常，應全數顯示 [OK])
    python scripts/fetchers/check_db_locks.py
"""

import sys
import logging
from pathlib import Path

# ── 系統路徑修復與標準化 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for sub in ("", "core"):
    p = (_SCRIPTS_DIR / sub) if sub else _SCRIPTS_DIR
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

try:
    # 引入 v4.0 的連線池 Session 管理器
    from core.db_utils import db_session
except ImportError as e:
    print(f"無法匯入核心模組: {e}", file=sys.stderr)
    sys.exit(1)

# ── 設定日誌 ──
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def check_db_locks():
    """檢測資料庫中的長時間查詢與鎖定狀態。"""
    logger.info("開始執行資料庫健康度與鎖定檢測...")
    
    try:
        # 使用 db_session() 獲取連線，退出區塊時會自動歸還至連線池
        with db_session() as conn:
            with conn.cursor() as cur:
                
                # =========================================================
                # 1. 檢測長時間查詢 (Long Running Queries)
                # =========================================================
                logger.info("--- 執行中的長時間查詢 (超過 1 分鐘) ---")
                cur.execute("""
                    SELECT pid, now() - query_start AS duration, query, state
                    FROM pg_stat_activity
                    WHERE state != 'idle' 
                      AND now() - query_start > interval '1 minute'
                      AND pid != pg_backend_pid();
                """)
                rows = cur.fetchall()
                if not rows:
                    logger.info("  [OK] 無執行超過 1 分鐘的查詢。")
                else:
                    for r in rows:
                        logger.warning(f"  PID: {r[0]} | 耗時: {r[1]} | 狀態: {r[3]}")
                        logger.warning(f"  查詢: {r[2][:250]}...")

                # =========================================================
                # 2. 檢測阻塞的查詢 (Blocked Queries)
                # =========================================================
                logger.info("\n--- 阻塞的查詢 (被其他 Lock 卡住) ---")
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

                # =========================================================
                # 3. 檢測詳細鎖定資訊 (Lock Details)
                # =========================================================
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
                        logger.info(f"  DB: {r[0]} | Table: {r[1]} | Mode: {r[3]} | Granted: {r[4]} | PID: {r[8]}")
                        
        logger.info("✅ 檢測完畢。")

    except Exception as e:
        logger.error(f"❌ 檢測資料庫鎖定狀態時發生錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_db_locks()