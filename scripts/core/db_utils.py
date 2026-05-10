"""
db_utils.py v4.20 (Trinity Core Final)
================================================================================
修訂歷程：
  v4.20 (2026-05-10): [修正] 強化路徑自癒 Bootstrap，解決 No module named 'core'。
"""
import os, sys, logging, threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from pathlib import Path
import psycopg2
from psycopg2 import pool, extras

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS = None
for p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent]:
    if p.name == "scripts" or (p / "scripts").exists():
        _SCRIPTS = p if p.name == "scripts" else (p / "scripts")
        break
if _SCRIPTS:
    if str(_SCRIPTS) not in sys.path: sys.path.insert(0, str(_SCRIPTS))
    if str(_SCRIPTS.parent) not in sys.path: sys.path.insert(0, str(_SCRIPTS.parent))

try: from dotenv import load_dotenv; load_dotenv(Path(_SCRIPTS.parent if _SCRIPTS else "") / ".env")
except: pass

_POOL = None
_POOL_LOCK = threading.Lock()

def get_pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            dsn = f"postgresql://{os.getenv('DB_USER','stock')}:{os.getenv('DB_PASSWORD','stock')}@{os.getenv('DB_HOST','127.0.0.1')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME','stock')}"
            _POOL = pool.ThreadedConnectionPool(1, 30, dsn=dsn, cursor_factory=extras.RealDictCursor)
    return _POOL

@contextmanager
def db_session():
    p = get_pool(); conn = p.getconn()
    try: yield conn; conn.commit()
    except Exception as e: conn.rollback(); raise e
    finally: p.putconn(conn)

@contextmanager
def db_transaction():
    with db_session() as conn:
        with conn.cursor() as cur: yield cur

def bulk_upsert(table_name: str, data: List[Dict], unique_cols: List[str] = ["date", "stock_id"]):
    if not data: return 0
    cols = [k.lower() for k in data[0].keys()]
    update_cols = [c for c in cols if c not in unique_cols]
    update_str = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s ON CONFLICT ({', '.join(unique_cols)}) DO UPDATE SET {update_str}"
    values = [[(v if v != "" else None) for v in d.values()] for d in data]
    try:
        with db_transaction() as cur: extras.execute_values(cur, sql, values)
        return len(data)
    except Exception as e:
        logging.error(f"❌ [DB] {table_name} Bulk Upsert 失敗: {e}"); return 0

def write_pipeline_log(task_name, stock_id, status, category, duration_ms=0, rows=0, err=None):
    sql = "INSERT INTO pipeline_execution_log (task_name, stock_id, status, category, duration_ms, rows_processed, error_message) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    try:
        with db_transaction() as cur: 
            cur.execute(sql, (task_name, stock_id, status, category, duration_ms, rows, str(err) if err else None))
    except Exception as e: print(f"[ERROR] Pipeline Log 失敗: {e}")

def get_db_stock_ids():
    with db_transaction() as cur:
        cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE")
        return [r['stock_id'] for r in cur.fetchall()]

def get_latest_date(table_name: str, stock_id: str = None) -> str:
    sql = f"SELECT MAX(date) as last_date FROM {table_name}"
    if stock_id: sql += f" WHERE stock_id = '{stock_id}'"
    try:
        with db_transaction() as cur:
            cur.execute(sql); res = cur.fetchone()
            return res['last_date'].strftime("%Y-%m-%d") if res and res['last_date'] else None
    except: return None