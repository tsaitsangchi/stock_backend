"""
db_utils.py v2.10 (Quantum Finance Edition)
================================================================================
資料庫公用工具函式庫 — 高吞吐二進位版 (Quantum v5.1 標準)
提供連線池管理、PID 自癒、以及包含 model_metadata 的基礎設施自癒功能。

修訂歷程：
  v2.10(2026-05-11): [自癒] 新增 model_metadata 表格自動建立功能。
  v2.9 (2026-05-11): [標準化] 擴充全場景執行範例，涵蓋強制更新與核心股遍歷。
================================================================================
"""
import os, sys, logging, threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
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

_ENV_PATH = "N/A"
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
    dsn = f"postgresql://{os.getenv('DB_USER','stock')}:{os.getenv('DB_PASSWORD','stock')}@{os.getenv('DB_HOST','127.0.0.1')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME','stock')}"
    return psycopg2.connect(dsn)

def ensure_infrastructure():
    """確保所有基礎日誌與元數據表格存在。"""
    sql_statements = [
        # 1. 生命週期日誌
        """CREATE TABLE IF NOT EXISTS pipeline_execution_log (
            id SERIAL PRIMARY KEY,
            task_name VARCHAR(100),
            stock_id VARCHAR(50),
            status VARCHAR(20),
            category VARCHAR(50),
            duration_ms INTEGER,
            rows_processed INTEGER,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
        # 2. 數據審計日誌
        """CREATE TABLE IF NOT EXISTS data_audit_log (
            id SERIAL PRIMARY KEY,
            table_name VARCHAR(100),
            stock_id VARCHAR(50),
            start_date DATE,
            end_date DATE,
            rows_affected INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
        # 3. 模型元數據註冊中心
        """CREATE TABLE IF NOT EXISTS model_metadata (
            stock_id VARCHAR(50),
            model_name VARCHAR(100),
            model_path TEXT,
            accuracy FLOAT,
            oof_da FLOAT,
            oof_sharpe FLOAT,
            feature_count INTEGER,
            params JSONB,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (stock_id, model_name)
        );"""
    ]
    try:
        with db_transaction() as cur:
            for sql in sql_statements:
                cur.execute(sql)
        return True
    except Exception as e:
        print(f"❌ 初始化基礎設施失敗: {e}"); return False

def check_db_health():
    print("\n" + "="*60)
    print("🚀 Quantum Finance: 系統核心與日誌儀表板 (v2.10)")
    print("="*60)
    try:
        ensure_infrastructure()
        with db_transaction() as cur:
            cur.execute("SELECT version(), now();")
            res = cur.fetchone()
            print(f"📡 環境配置  : {str(_ENV_PATH)[:50]}...")
            print(f"⏰ 資料庫時間: {res['now'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            p = get_pool()
            used = len(getattr(p, '_used', [])) if isinstance(getattr(p, '_used', []), (list, dict)) else 0
            print(f"🏊 連線池狀態: {used} / {getattr(p, 'maxconn', '30')}")
            print("-" * 30)
            
            yesterday = datetime.now() - timedelta(days=1)
            cur.execute("SELECT count(*) as total, count(*) FILTER (WHERE status = 'success') as success, count(*) FILTER (WHERE status = 'failed') as failed FROM pipeline_execution_log WHERE created_at > %s", (yesterday,))
            stats = cur.fetchone()
            print(f"📊 最近 24H 生命周期 (Lifecycle): {stats['total']} 筆 ({stats['success']} ✅ / {stats['failed']} ❌)")
            
            cur.execute("SELECT count(*) as total FROM data_audit_log;")
            print(f"📋 分類審計紀錄 (Audit Log): {cur.fetchone()['total']} 筆紀錄")
            
            cur.execute("SELECT count(*) as total FROM model_metadata;")
            print(f"🤖 模型註冊中心 (Models): {cur.fetchone()['total']} 個已註冊模型")
        return True
    except Exception as e:
        print(f"❌ 儀表板回報失敗: {e}"); return False

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
        with db_transaction() as cur: cur.execute(sql, (table_name, stock_id, start_date, end_date, rows_affected))
    except:
        ensure_infrastructure()
        try:
            with db_transaction() as cur: cur.execute(sql, (table_name, stock_id, start_date, end_date, rows_affected))
        except: pass

@contextmanager
def record_lifecycle(task_name: str, category: str, stock_id: str = "GLOBAL"):
    t0 = datetime.now()
    try:
        yield
        duration = int((datetime.now() - t0).total_seconds() * 1000)
        write_pipeline_log(task_name, stock_id, "success", category, duration)
    except Exception as e:
        duration = int((datetime.now() - t0).total_seconds() * 1000)
        write_pipeline_log(task_name, stock_id, "failed", category, duration, err=e)
        raise e

def get_db_stock_ids(active_only=True):
    sql = "SELECT stock_id FROM stocks"
    if active_only: sql += " WHERE is_active = TRUE"
    with db_transaction() as cur:
        cur.execute(sql); return [r['stock_id'] for r in cur.fetchall()]

def get_latest_date(table_name: str, stock_id: str = None, id_column: str = "stock_id") -> str:
    sql = f"SELECT MAX(date) as last_date FROM {table_name}"
    if stock_id: sql += f" WHERE {id_column} = '{stock_id}'"
    try:
        with db_transaction() as cur:
            cur.execute(sql); res = cur.fetchone()
            return res['last_date'].strftime("%Y-%m-%d") if res and res['last_date'] else None
    except: return None

if __name__ == "__main__":
    if check_db_health():
        print("\n🚀 Quantum 基礎架構運行良好。已確認 model_metadata 表格可用。")
        print("="*60 + "\n")
    else: sys.exit(1)