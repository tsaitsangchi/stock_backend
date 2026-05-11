"""
db_utils.py v2.27 (Quantum Finance Edition)
================================================================================
資料庫核心引擎 — 極致範例與自癒加強版 (Quantum v5.2 標準)
負責管理資產元數據、結構自癒與高效能批量寫入。

修訂歷程：
  v2.27 (2026-05-11): [修復] 將 ensure_infrastructure 整合進 record_lifecycle，確保引導不崩潰。
  v2.26 (2026-05-11): [標準] 對齊 .env v4.0。

【執行範例矩陣 (Database Operation Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [基礎設施手動自癒]        │ $ python scripts/core/db_utils.py                      │
│ 2. [單一個股：獲取元數據]    │ info = get_stock_metadata("2330")                      │
│ 3. [核心標的：強制批量更新]  │ bulk_upsert("stocks", core_list, ["stock_id"])         │
│ 4. [全量標的：所有Table稽核] │ check_all_tables_integrity()                           │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, logging, time, platform
from pathlib import Path
from typing import List, Dict
from contextlib import contextmanager
from psycopg2 import pool, extras
from psycopg2.extras import execute_values

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core.path_setup import get_log_dir

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME = os.getenv("DB_NAME", "stock")
DB_USER = os.getenv("DB_USER", "stock")
DB_PASSWORD = os.getenv("DB_PASSWORD", "stock")
DB_PORT = os.getenv("DB_PORT", "5432")

_pool = None
_infra_ready = False

def get_pool():
    global _pool
    if _pool is None:
        _pool = pool.ThreadedConnectionPool(1, 20, host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    return _pool

@contextmanager
def db_connection():
    p = get_pool(); conn = p.getconn()
    try: yield conn; conn.commit()
    except Exception as e: conn.rollback(); raise e
    finally: p.putconn(conn)

@contextmanager
def db_transaction():
    with db_connection() as conn:
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            yield cur

def ensure_infrastructure():
    global _infra_ready
    with db_transaction() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS stocks (stock_id VARCHAR(20) PRIMARY KEY);")
        cur.execute("CREATE TABLE IF NOT EXISTS pipeline_execution_log (id SERIAL PRIMARY KEY, task_name VARCHAR(100), stock_id VARCHAR(20), status VARCHAR(20), category VARCHAR(50), duration_ms INTEGER, rows_affected INTEGER, error_msg TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
        cur.execute("CREATE TABLE IF NOT EXISTS data_audit_log (id SERIAL PRIMARY KEY, table_name VARCHAR(100), stock_id VARCHAR(20), data_date DATE, action_type VARCHAR(50), rows_affected INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
    _infra_ready = True
    return 0

def get_db_stock_ids(is_core=None, active_only=True):
    sql = "SELECT stock_id FROM stocks WHERE 1=1"
    if is_core is not None: sql += f" AND is_core = {is_core}"
    if active_only: sql += " AND is_active = TRUE"
    with db_transaction() as cur:
        cur.execute(sql); return [r['stock_id'] for r in cur.fetchall()]

def bulk_upsert(table_name: str, data: List[Dict], unique_cols: List[str]):
    if not data: return 0
    cols = data[0].keys()
    query = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s ON CONFLICT ({', '.join(unique_cols)}) DO UPDATE SET {', '.join([f'{c} = EXCLUDED.{c}' for c in cols if c not in unique_cols])}"
    with db_transaction() as cur:
        execute_values(cur, query, [[d.get(c) for c in cols] for d in data])
        return cur.rowcount

def get_latest_date(table_name: str, stock_id: str):
    with db_transaction() as cur:
        cur.execute(f"SELECT MAX(date) as last_date FROM {table_name} WHERE stock_id = %s", (stock_id,))
        res = cur.fetchone(); return res['last_date'] if res else None

def write_pipeline_log(task_name, stock_id, status, category, duration=0, rows=0, error=None):
    if not _infra_ready: ensure_infrastructure()
    with db_transaction() as cur:
        cur.execute("INSERT INTO pipeline_execution_log (task_name, stock_id, status, category, duration_ms, rows_affected, error_msg) VALUES (%s, %s, %s, %s, %s, %s, %s)", (task_name, stock_id, status, category, duration, rows, str(error) if error else None))

@contextmanager
def record_lifecycle(task_name, category="general", stock_id="SYSTEM"):
    if not _infra_ready: ensure_infrastructure()
    t0 = time.monotonic()
    try:
        yield
        duration = int((time.monotonic() - t0) * 1000)
        write_pipeline_log(task_name, stock_id, "success", category, duration)
    except Exception as e:
        duration = int((time.monotonic() - t0) * 1000)
        write_pipeline_log(task_name, stock_id, "failed", category, duration, error=str(e))
        raise e

def write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected):
    if not _infra_ready: ensure_infrastructure()
    with db_transaction() as cur:
        cur.execute("INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected) VALUES (%s, %s, %s, %s, %s)", (table_name, stock_id, data_date, action_type, rows_affected))

if __name__ == "__main__":
    ensure_infrastructure()