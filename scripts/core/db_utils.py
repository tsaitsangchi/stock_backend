"""
db_utils.py v4.13 (Trinity Core Final)
================================================================================
資料庫維運核心 — 極致穩定版
具備完善的 DDL 自癒機制，確保 created_at 與 extra_metadata 欄位對齊。

修訂歷程：
  v4.13 (2026-05-09):
    - [修復] 強制補齊 evaluation_log 的 created_at 欄位。
"""

import os
import sys
import json
import logging
import threading
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool, extras
from pathlib import Path

# ── 系統路徑與 Dotenv ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

_POOL = None
_POOL_LOCK = threading.Lock()
_OWNER_PID = None

def get_pool():
    global _POOL, _OWNER_PID
    current_pid = os.getpid()
    with _POOL_LOCK:
        if _POOL is None or _OWNER_PID != current_pid:
            user = os.getenv("DB_USER", "stock")
            pw = os.getenv("DB_PASSWORD", "stock")
            host = os.getenv("DB_HOST", "127.0.0.1")
            port = os.getenv("DB_PORT", "5432")
            dbname = os.getenv("DB_NAME", "stock")
            dsn = f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"
            if _POOL is not None:
                try: _POOL.closeall()
                except: pass
            _POOL = pool.ThreadedConnectionPool(1, 30, dsn=dsn, cursor_factory=extras.RealDictCursor)
            _OWNER_PID = current_pid
            logging.info(f"✅ 連線池就緒 (PID: {current_pid})")
    return _POOL

@contextmanager
def db_session():
    p = get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        p.putconn(conn)

@contextmanager
def db_transaction():
    with db_session() as conn:
        with conn.cursor() as cur:
            yield cur

def ensure_ddl():
    """
    自癒 DDL。確保欄位完整性。
    """
    ddl_queries = [
        "CREATE TABLE IF NOT EXISTS pipeline_execution_log (id SERIAL PRIMARY KEY, task_name TEXT NOT NULL, stock_id TEXT, status TEXT, category TEXT, duration_ms INTEGER, rows_processed INTEGER, error_message TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);",
        "CREATE TABLE IF NOT EXISTS evaluation_log (id SERIAL PRIMARY KEY, stock_id TEXT NOT NULL, model_name TEXT, sharpe_ratio NUMERIC, max_drawdown NUMERIC, total_return NUMERIC, win_rate NUMERIC, start_date DATE, end_date DATE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);",
        "ALTER TABLE evaluation_log ADD COLUMN IF NOT EXISTS extra_metadata JSONB;",
        "ALTER TABLE evaluation_log ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
    ]
    try:
        with db_transaction() as cur:
            for q in ddl_queries:
                cur.execute(q)
    except Exception as e:
        print(f"[ERROR] Migration 失敗: {e}")

def write_pipeline_log(task_name, stock_id, status, category, duration_ms=0, rows=0, err=None):
    sql = "INSERT INTO pipeline_execution_log (task_name, stock_id, status, category, duration_ms, rows_processed, error_message) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    try:
        with db_transaction() as cur: cur.execute(sql, (task_name, stock_id, status, category, duration_ms, rows, str(err) if err else None))
    except Exception as e: print(f"[ERROR] log 失敗: {e}")

def write_evaluation_log(stock_id, model_name, sharpe, mdd, ret, win_rate, start, end, extra=None):
    sql = "INSERT INTO evaluation_log (stock_id, model_name, sharpe_ratio, max_drawdown, total_return, win_rate, start_date, end_date, extra_metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    try:
        with db_transaction() as cur: cur.execute(sql, (stock_id, model_name, sharpe, mdd, ret, win_rate, start, end, json.dumps(extra) if extra else None))
    except Exception as e: print(f"[ERROR] evaluation_log 失敗: {e}")

def get_db_stock_ids(fetch_type=None, industry=None, market_type=None):
    sql = "SELECT stock_id FROM stocks WHERE is_active = TRUE"
    params = []
    if fetch_type: sql += f" AND fetch_{fetch_type} = TRUE"
    if industry: sql += " AND industry = %s"; params.append(industry)
    if market_type: sql += " AND market_type = %s"; params.append(market_type)
    with db_transaction() as cur:
        cur.execute(sql, tuple(params))
        return [r['stock_id'] for r in cur.fetchall()]