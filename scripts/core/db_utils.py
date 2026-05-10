"""
db_utils.py v2.5 (Quantum Finance Edition)
================================================================================
資料庫公用工具函式庫 — 高吞吐二進位版 (Quantum v5.1 標準)
提供連線池管理、PID 自癒、批量寫入 (UPSERT) 以及混合模式日誌介面。

修訂歷程：
  v2.5 (2026-05-10): [修復] 補回 get_db_connection 確保維護工具與治理腳本相容性。
  v2.4 (2026-05-10): [核心] 新增 write_data_audit_log 實作混合模式審計紀錄。
  v2.2-2.3: 實作連線池與日期偵測。
================================================================================
"""
import os, sys, logging, threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
import psycopg2
from psycopg2 import pool, extras

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path, get_scripts_dir
    ensure_scripts_on_path(__file__)
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from path_setup import get_scripts_dir

try:
    from dotenv import load_dotenv
    _ENV_PATH = get_scripts_dir().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    else:
        load_dotenv()
except: pass

_POOL = None
_POOL_PID = None
_POOL_LOCK = threading.Lock()

def get_pool():
    global _POOL, _POOL_PID
    with _POOL_LOCK:
        curr_pid = os.getpid()
        if _POOL is None or _POOL_PID != curr_pid:
            dsn = f"postgresql://{os.getenv('DB_USER','stock')}:{os.getenv('DB_PASSWORD','stock')}@{os.getenv('DB_HOST','127.0.0.1')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME','stock')}"
            _POOL = pool.ThreadedConnectionPool(1, 30, dsn=dsn, cursor_factory=extras.RealDictCursor)
            _POOL_PID = curr_pid
    return _POOL

def get_db_connection():
    """獲取單次連線 (支援舊版治理工具)。"""
    dsn = f"postgresql://{os.getenv('DB_USER','stock')}:{os.getenv('DB_PASSWORD','stock')}@{os.getenv('DB_HOST','127.0.0.1')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME','stock')}"
    return psycopg2.connect(dsn)

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

def write_data_audit_log(table_name, stock_id, start_date, end_date, rows_affected):
    sql = "INSERT INTO data_audit_log (table_name, stock_id, start_date, end_date, rows_affected) VALUES (%s, %s, %s, %s, %s)"
    try:
        with db_transaction() as cur:
            cur.execute(sql, (table_name, stock_id, start_date, end_date, rows_affected))
    except Exception as e:
        print(f"[WARNING] 正在嘗試自動修復 Audit Log 表: {e}")
        try:
            with db_transaction() as cur:
                cur.execute("DROP TABLE IF EXISTS data_audit_log CASCADE;")
                cur.execute("""
                    CREATE TABLE data_audit_log (
                        id SERIAL PRIMARY KEY,
                        table_name VARCHAR(100),
                        stock_id VARCHAR(50),
                        start_date DATE,
                        end_date DATE,
                        rows_affected INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute(sql, (table_name, stock_id, start_date, end_date, rows_affected))
        except: print(f"[ERROR] Data Audit Log 最終失敗")

def get_db_stock_ids(active_only=True):
    sql = "SELECT stock_id FROM stocks"
    if active_only: sql += " WHERE is_active = TRUE"
    with db_transaction() as cur:
        cur.execute(sql)
        return [r['stock_id'] for r in cur.fetchall()]

def get_latest_date(table_name: str, stock_id: str = None, id_column: str = "stock_id") -> str:
    sql = f"SELECT MAX(date) as last_date FROM {table_name}"
    if stock_id: 
        sql += f" WHERE {id_column} = '{stock_id}'"
    try:
        with db_transaction() as cur:
            cur.execute(sql); res = cur.fetchone()
            return res['last_date'].strftime("%Y-%m-%d") if res and res['last_date'] else None
    except: return None