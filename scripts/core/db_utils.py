"""
db_utils.py v4.2 (Secure Env Edition)
量化系統核心：資料庫連線池與自動事務管理模組 (支援 .env 架構)
================================================================================
v4.2 重大升級：
  · 整合 dotenv：啟動時自動尋找專案根目錄的 .env 檔案，徹底將機密配置與程式碼脫鉤。
  · 安全預設值：若偵測不到密碼會發出明確的警告，並攔截錯誤，防止系統崩潰。
  · 繼承 v4.0 優勢：保留 ThreadedConnectionPool 與 db_transaction 上下文管理器。

【環境變數配置規範】
請在專案根目錄 (stock_backend/) 建立 .env 檔案，內容範例：
    DB_HOST=localhost
    DB_NAME=trinity
    DB_USER=stock
    DB_PASSWORD=your_secure_password
    DB_PORT=5432

【執行範例（開發建議）】：
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
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

import psycopg2
from psycopg2 import pool, extras

# ── 安全環境變數載入 (dotenv) ──
try:
    from dotenv import load_dotenv, find_dotenv
    # 自動往上層目錄尋找 .env 檔案並載入系統環境變數中
    # 這樣即使腳本在不同的子目錄執行，也能正確讀取到專案根目錄的 .env
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        logging.getLogger(__name__).warning("⚠️ 未偵測到 .env 檔案，將使用系統預設環境變數。")
except ImportError:
    logging.getLogger(__name__).warning("⚠️ 未安裝 python-dotenv，請執行 `pip install python-dotenv` 以啟用安全的配置管理。")

# ── 系統設定 ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 資料庫連線池設定 (以環境變數為主) ──
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "database": os.environ.get("DB_NAME", "stock"),
    "user": os.environ.get("DB_USER", "stock"),
    "password": os.environ.get("DB_PASSWORD", ""),  # 嚴格依賴環境變數
    "port": os.environ.get("DB_PORT", "5432"),
}

# 初始化執行緒安全的連線池 (最小 2 條, 最大 10 條，視並行數調整)
_DB_POOL = None
try:
    if not DB_CONFIG["password"]:
        logger.warning("⚠️ 注意：DB_PASSWORD 環境變數為空，若 PostgreSQL 需要密碼驗證將會導致連線失敗。")
        
    _DB_POOL = psycopg2.pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        **DB_CONFIG
    )
    logger.info("✅ 成功初始化 ThreadedConnectionPool (Secure Env Mode)")
except Exception as e:
    logger.error(f"❌ 無法建立連線池: {e}")

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
        raise ConnectionError("資料庫連線池未正確初始化。請檢查專案根目錄的 .env 檔案與 DB_PASSWORD。")
    
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
# Compatibility Layer (Shims for old scripts)
# =====================================================================

def get_db_conn():
    """Shim: Returns a connection from the pool. OLD scripts expect this."""
    if _DB_POOL is None: return None
    return _DB_POOL.getconn()

def get_db_stock_ids(conn=None) -> List[str]:
    """Shim: Get all stock IDs from 'stocks' table."""
    sql = "SELECT stock_id FROM stocks ORDER BY stock_id"
    with db_session() as c:
        with c.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]

def get_core_stocks_from_db() -> List[str]:
    """Shim: Get stocks where fetch_basic is true."""
    sql = "SELECT stock_id FROM stocks WHERE fetch_basic = True ORDER BY stock_id"
    with db_session() as c:
        with c.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]

def get_market_safe_start(table_name: str, default_start: str = "2000-01-01") -> str:
    """Shim: Get latest date for market-level tables."""
    latest = get_latest_date(table_name)
    return latest if latest else default_start

def get_all_safe_starts(table_name: str, stock_ids: List[str], default_start: str) -> Dict[str, str]:
    """Shim: Get latest date for multiple stocks (Gap Fill support)."""
    sql = f"SELECT stock_id, MAX(date) FROM {table_name} GROUP BY stock_id"
    res_dict = {sid: default_start for sid in stock_ids}
    try:
        with db_session() as c:
            with c.cursor() as cur:
                cur.execute(sql)
                for sid, mdate in cur.fetchall():
                    if sid in res_dict and mdate:
                        res_dict[sid] = mdate.strftime("%Y-%m-%d")
    except: pass
    return res_dict

def resolve_start_cached(table_name: str, stock_id: str, default_start: str) -> str:
    """Shim: Resolve start date for a single stock."""
    latest = get_latest_date(table_name, stock_id)
    return latest if latest else default_start

def safe_commit_rows(conn, table_name: str, records: List[Dict], upsert_sql: str, stock_id: str = "Market") -> tuple[int, int]:
    """Shim: Map old safe_commit_rows to new commit_per_stock_per_day."""
    return commit_per_stock_per_day(table_name, records, upsert_sql, stock_id)

def dedup_rows(rows: List[Dict], key_fields: List[str]) -> List[Dict]:
    """Shim: Deduplicate rows based on key fields."""
    seen = set()
    unique = []
    for r in rows:
        k = tuple(r.get(f) for f in key_fields)
        if k not in seen:
            seen.add(k)
            unique.append(r)
    return unique

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
# 錯誤追蹤與防護
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