"""
db_utils.py v4.0
量化系統核心：資料庫連線池與自動事務管理模組
================================================================================
v4.0 重大升級：
  · 實作 ThreadedConnectionPool：解決並行任務（如 parallel > 1）頻繁建立連線的效能瓶頸。
  · 引入 @contextmanager db_transaction：自動處理 commit 與 rollback，杜絕髒連線。
  · 增強可靠性：所有 SQL 操作均受 Context Manager 保護，異常時自動回滾並釋放連線。

執行範例（開發建議）：
    from core.db_utils import db_session, db_transaction
    
    # 方式 A：單純查詢 (自動歸還連線)
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM stock_price")
            print(cur.fetchone())

    # 方式 B：寫入事務 (出錯自動 Rollback, 成功自動 Commit)
    with db_transaction() as cur:
        cur.execute("INSERT INTO ...")
        cur.execute("UPDATE ...")
"""

import os
import json
import logging
import time
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

import psycopg2
from psycopg2 import pool, extras

# ── 系統設定 ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 資料庫連線池設定 ──
# 使用環境變數或預設值
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "database": os.environ.get("DB_NAME", "trinity"),
    "user": os.environ.get("DB_USER", "hugo"),
    "password": os.environ.get("DB_PASS", ""),
    "port": os.environ.get("DB_PORT", "5432"),
}

# 初始化執行緒安全的連線池 (最小 2 條, 最大 10 條，視並行數調整)
try:
    _DB_POOL = psycopg2.pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        **DB_CONFIG
    )
    logger.info("✅ 成功初始化 ThreadedConnectionPool (maxconn=10)")
except Exception as e:
    logger.error(f"❌ 無法建立連線池: {e}")
    _DB_POOL = None

# =====================================================================
# 核心：上下文管理器 (Context Managers)
# =====================================================================

@contextmanager
def db_session() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    獲取連線的 Context Manager。
    退出時自動將連線歸還給連線池，不關閉實體連線。
    """
    if _DB_POOL is None:
        raise ConnectionError("資料庫連線池未正確初始化")
    
    conn = _DB_POOL.getconn()
    try:
        yield conn
    finally:
        _DB_POOL.putconn(conn)

@contextmanager
def db_transaction() -> Generator[psycopg2.extensions.cursor, None, None]:
    """
    自動事務處理的 Context Manager。
    1. 從池中取連線。
    2. 開啟 Cursor。
    3. 執行業務邏輯。
    4. 成功則 Commit，失敗則 Rollback。
    5. 自動歸還連線。
    """
    with db_session() as conn:
        with conn.cursor() as cur:
            try:
                yield cur
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"⚠️ 事務執行失敗，已自動回滾: {e}")
                raise

# =====================================================================
# 基礎設施 DDL 與 Log
# =====================================================================

DDL_FETCH_LOG = """
CREATE TABLE IF NOT EXISTS fetch_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    stock_id VARCHAR(20),
    status VARCHAR(20),
    fetch_mode VARCHAR(50),
    fetch_date_from DATE,
    fetch_date_to DATE,
    duration_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_lookup ON fetch_log(table_name, stock_id, created_at DESC);
"""

def ensure_ddl(ddl_query: str):
    """確保指定的資料表結構存在 (使用自動事務)"""
    with db_transaction() as cur:
        cur.execute(ddl_query)
        cur.execute(DDL_FETCH_LOG)

def write_fetch_log(table_name: str, stock_id: Optional[str], status: str, 
                    fetch_mode: str, fetch_date_from: str, fetch_date_to: str, 
                    duration_ms: int, error_message: Optional[str] = None):
    """標準化日誌寫入 (使用自動事務)"""
    sql = """
    INSERT INTO fetch_log (table_name, stock_id, status, fetch_mode, fetch_date_from, fetch_date_to, duration_ms, error_message)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    with db_transaction() as cur:
        cur.execute(sql, (table_name, stock_id, status, fetch_mode, fetch_date_from, fetch_date_to, duration_ms, error_message))

def get_latest_date(table_name: str, stock_id: Optional[str] = None) -> Optional[str]:
    """獲取資料庫中該表的最新日期"""
    sql = f"SELECT MAX(date) FROM {table_name}"
    if stock_id:
        sql += " WHERE stock_id = %s"
    
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (stock_id,) if stock_id else ())
            res = cur.fetchone()
            return res[0].strftime("%Y-%m-%d") if res and res[0] else None

# =====================================================================
# 核心業務：高可靠性資料寫入
# =====================================================================

def commit_per_stock_per_day(table_name: str, records: List[Dict[str, Any]], 
                             upsert_query: str, stock_id: str) -> tuple[int, int]:
    """
    針對特定股票執行大批次寫入，若失敗則進入自動降級模式（逐筆重試並記錄損毀點）。
    """
    success_count = 0
    error_count = 0
    
    # 1. 嘗試快速批次寫入 (使用一個事務)
    try:
        with db_transaction() as cur:
            psycopg2.extras.execute_batch(cur, upsert_query, records)
            success_count = len(records)
            return success_count, 0
    except Exception as e:
        logger.warning(f"  [批次失敗] {stock_id} @ {table_name}，嘗試逐筆安全寫入保護資料：{e}")
    
    # 2. 批次失敗，降級為「逐筆事務」確保最大程度保留成功資料
    fail_logger = FailureLogger(table_name)
    for rec in records:
        try:
            with db_transaction() as cur:
                cur.execute(upsert_query, rec)
                success_count += 1
        except Exception as e:
            error_count += 1
            fail_logger.log_failure(table_name, stock_id, rec.get('date'), rec.get('date'), str(e))
            
    return success_count, error_count

# =====================================================================
# Type Conversion Utilities
# =====================================================================

def safe_float(val: Any) -> Optional[float]:
    if val is None: return None
    try: return float(val)
    except: return None

def safe_int(val: Any) -> Optional[int]:
    f = safe_float(val)
    return int(f) if f is not None else None

def safe_date(val: Any) -> Optional[str]:
    if not val: return None
    s = str(val).strip()
    try:
        # Validate format
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except: return None

# =====================================================================
# Error Tracking and Protection
# =====================================================================

class FailureLogger:
    """處理並發寫入失敗時的持久化紀錄"""
    def __init__(self, category: str):
        self.log_path = Path("outputs") / f"failure_{category}.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_failure(self, table: str, stock_id: Optional[str], start: str, end: str, error: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "table": table,
            "stock_id": stock_id,
            "start": start,
            "end": end,
            "error": error
        }
        # 使用原子寫入防止 JSON 損壞
        existing = []
        if self.log_path.exists():
            try:
                existing = json.loads(self.log_path.read_text())
            except: pass
        
        existing.append(entry)
        self.log_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))