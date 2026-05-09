"""
db_utils.py v4.6 (Trinity Core Edition)
================================================================================
資料庫核心工具組：連線池管理、自動事務控制、DML 筆數監控、以及標準化抓取日誌。
此模組為整個系統的數據持久化基石，確保數據寫入的原子性 (Atomicity) 與可觀測性。

核心功能：
  · ThreadedConnectionPool ─ 高併發連線複用。
  · db_transaction         ─ 自動化事務開關與異常回滾 (Rollback)。
  · write_fetch_log        ─ 完美記錄每一筆抓取的筆數 (rows_inserted) 與狀態。
  · ensure_ddl             ─ 冪等建表與索引維護。

修訂歷程：
  v4.6 (2026-05-09):
    - [監控] write_fetch_log 補齊 rows_inserted 欄位，實現精準數據監控。
    - [核心] get_latest_date 支援動態 id_column，解決匯入總經 ID 差異問題。
    - [事務] 強化 commit_per_stock_per_day，在批次失敗時自動降級為原子重試。
  v4.5 (2026-05-01):
    - [架構] 統一 ThreadedConnectionPool 管理，移除 manual conn 傳遞。
    - [核心] ensure_ddl 簽章重構，簡化為 ensure_ddl(ddl)。
  v4.0 (2026-04-15):
    - [穩定] 引入 db_transaction 解決多執行緒併發寫入時的死鎖問題。

執行範例：
    # 範例 1：使用事務確保原子寫入
    from core.db_utils import db_transaction
    with db_transaction() as cur:
        cur.execute("INSERT INTO test (id) VALUES (%s)", (1,))
    
    # 範例 2：紀錄抓取日誌 (帶筆數與狀態)
    from core.db_utils import write_fetch_log
    write_fetch_log(table_name="price", stock_id="2330", status="success", 
                    fetch_mode="incremental", fetch_date_from="2021-01-01", 
                    fetch_date_to="2021-01-02", duration_ms=150, rows_inserted=24)
"""

import os
import logging
import time
import psycopg2
import psycopg2.extras
import psycopg2.pool
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定日誌
logger = logging.getLogger(__name__)

# =====================================================================
# 1. 連線池管理 (Connection Pool)
# =====================================================================

class DatabaseManager:
    _pool = None

    @classmethod
    def get_pool(cls):
        if cls._pool is None:
            try:
                cls._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=20,
                    dbname=os.getenv("DB_NAME", "stock"),
                    user=os.getenv("DB_USER", "stock"),
                    password=os.getenv("DB_PASSWORD", "stock"),
                    host=os.getenv("DB_HOST", "localhost"),
                    port=os.getenv("DB_PORT", "5432")
                )
                logger.info("✅ 成功初始化 ThreadedConnectionPool (v4.5 Config Aligned)")
            except Exception as e:
                logger.error(f"❌ 資料庫連線池初始化失敗: {e}")
                raise
        return cls._pool

@contextmanager
def db_session():
    """連線工作階段管理器"""
    pool = DatabaseManager.get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

@contextmanager
def db_transaction():
    """事務管理器：提供 Cursor 並處理自動提交與回滾"""
    with db_session() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur

# =====================================================================
# 2. DDL 與維護工具 (Maintenance)
# =====================================================================

def ensure_ddl(ddl_query: str):
    """執行 DDL 語句 (Trinity v4.5 簽章：僅接收 DDL 字串)"""
    with db_transaction() as cur:
        try:
            # 使用 Savepoint 保護 DDL 執行
            cur.execute("SAVEPOINT sp_ddl;")
            cur.execute(ddl_query)
            cur.execute("RELEASE SAVEPOINT sp_ddl;")
        except Exception as e:
            cur.execute("ROLLBACK TO SAVEPOINT sp_ddl;")
            logger.debug(f"DDL 執行略過或已存在: {e}")

def write_fetch_log(table_name: str, stock_id: Optional[str], status: str, 
                    fetch_mode: str, fetch_date_from: str, fetch_date_to: str, 
                    duration_ms: int, rows_inserted: int = 0, error_message: Optional[str] = None):
    """
    標準化抓取日誌寫入 (v4.6 升級版)
    新增 rows_inserted 欄位追蹤抓取筆數。
    """
    sql = """
    INSERT INTO fetch_log (
        table_name, stock_id, status, fetch_mode, 
        fetch_date_from, fetch_date_to, duration_ms, rows_inserted, error_message
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with db_transaction() as cur:
        cur.execute(sql, (
            table_name, stock_id, status, fetch_mode, 
            fetch_date_from, fetch_date_to, duration_ms, rows_inserted, error_message
        ))

# =====================================================================
# 3. 數據偵測與查詢 (Detection)
# =====================================================================

def get_latest_date(table_name: str, stock_id: Optional[str] = None, id_column: str = "stock_id") -> Optional[str]:
    """
    獲取資料庫中該表的最新日期 (v4.6 升級版)
    支援自定義 id_column 以適應 futures_id, option_id 等欄位。
    """
    sql = f"SELECT MAX(date) FROM {table_name}"
    params = []
    if stock_id:
        sql += f" WHERE {id_column} = %s"
        params.append(stock_id)
    
    with db_transaction() as cur:
        cur.execute(sql, params)
        res = cur.fetchone()
        if res and res['max']:
            # 處理不同類型的返回值
            mdate = res['max']
            return mdate.strftime("%Y-%m-%d") if hasattr(mdate, 'strftime') else str(mdate)
        return None

def get_db_stock_ids(types: tuple = ("twse", "otc")) -> List[str]:
    """獲取資料庫中的股票 ID 清單"""
    sql = "SELECT stock_id FROM stocks WHERE type = ANY(%s) AND is_active = TRUE ORDER BY stock_id"
    with db_transaction() as cur:
        cur.execute(sql, (list(types),))
        return [r['stock_id'] for r in cur.fetchall()]

# =====================================================================
# 4. 高性能寫入 (Writing)
# =====================================================================

def commit_per_stock_per_day(table_name: str, records: List[Dict[str, Any]], 
                             upsert_query: str, stock_id: str) -> Tuple[int, int]:
    """
    混合式安全寫入：優先執行大批次寫入，若失敗則自動降級為逐筆保護寫入。
    回傳: (成功筆數, 失敗筆數)
    """
    if not records:
        return 0, 0

    # 1. 嘗試大批次寫入
    try:
        with db_transaction() as cur:
            psycopg2.extras.execute_batch(cur, upsert_query, records)
            return len(records), 0
    except Exception as e:
        logger.warning(f"  ⚠️ [批次失敗] {stock_id} @ {table_name}，嘗試逐筆重試：{e}")
    
    # 2. 降級為逐筆保護寫入
    success_count = 0
    error_count = 0
    for rec in records:
        try:
            with db_transaction() as cur:
                cur.execute(upsert_query, rec)
                success_count += 1
        except Exception:
            error_count += 1
            
    return success_count, error_count

# =====================================================================
# 5. 安全轉換工具 (Safety Utils)
# =====================================================================

def safe_float(val: Any) -> Optional[float]:
    if val is None or val == "": return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def safe_int(val: Any) -> Optional[int]:
    f = safe_float(val)
    return int(f) if f is not None else None

# =====================================================================
# 6. 維運日誌工具 (Maintenance Log)
# =====================================================================

class FailureLogger:
    """用於紀錄抓取過程中的具體錯誤組合"""
    def __init__(self, label: str):
        self.label = label
        self.failures = []

    def log_failure(self, table: str, stock_id: str, start: str, end: str, error: str):
        self.failures.append({
            "table": table, "stock_id": stock_id, 
            "start": start, "end": end, "error": error
        })

    def summary(self):
        if not self.failures:
            return
        logger.error(f"━━━ {self.label} 失敗摘要 (共 {len(self.failures)} 筆) ━━━")
        for f in self.failures[:10]:
            logger.error(f"  ❌ {f['stock_id']} [{f['start']}~{f['end']}]: {f['error'][:100]}")
        if len(self.failures) > 10:
            logger.error(f"  ...等其餘 {len(self.failures)-10} 筆錯誤")

DDL_FETCH_LOG = """
CREATE TABLE IF NOT EXISTS fetch_log (
    run_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    table_name VARCHAR(100),
    stock_id VARCHAR(50),
    status VARCHAR(50),
    fetch_mode VARCHAR(50),
    fetch_date_from DATE,
    fetch_date_to DATE,
    duration_ms INTEGER,
    rows_inserted INTEGER,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_table_stock ON fetch_log (table_name, stock_id);
CREATE INDEX IF NOT EXISTS idx_fetch_log_run_ts ON fetch_log (run_ts DESC);
"""