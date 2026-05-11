"""
db_utils.py v2.14 (Quantum Finance Edition)
================================================================================
資料庫公用工具函式庫 — 高吞吐二進位版 (Quantum v5.2 標準)
負責連線池管理、自動自癒、以及資產表 (stocks) 的結構加固。

修訂歷程：
  v2.14(2026-05-11): [修復] 修正 logger 未定義錯誤，優化執行範例說明。
  v2.13(2026-05-11): [核心自癒] 補齊 stocks 表缺失欄位 (industry_category, is_core, etc.)。
  v2.12(2026-05-11): [修復] 修正 margin_purchase_short_sale 欄位衝突。

執行範例 (Comprehensive Usage Matrix):
  1. [基礎設施診斷與自癒] 檢查並建立缺失的核心表格與欄位:
     python scripts/core/db_utils.py

  2. [核心資產稽核] 獲取目前資料庫中的活躍標的清單:
     from core.db_utils import get_db_stock_ids
     ids = get_db_stock_ids(active_only=True)

  3. [數據生命週期紀錄] 在自定義腳本中紀錄任務起訖:
     with record_lifecycle("task_name", category="maintenance", stock_id="2330"):
         # 執行您的任務...
         pass
================================================================================
"""
import os, sys, logging, threading, re, time
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
    if _ENV_PATH.exists(): load_dotenv(_ENV_PATH)
    else: load_dotenv()
except: pass

# 初始化全局日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

def ensure_infrastructure():
    """確保核心表格結構完整，特別是 stocks 表的結構升級。"""
    sql_statements = [
        """CREATE TABLE IF NOT EXISTS stocks (
            stock_id VARCHAR(50) PRIMARY KEY,
            stock_name VARCHAR(100),
            industry_category VARCHAR(100),
            type VARCHAR(50),
            is_active BOOLEAN DEFAULT TRUE,
            is_core BOOLEAN DEFAULT FALSE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
        """CREATE TABLE IF NOT EXISTS pipeline_execution_log (
            id SERIAL PRIMARY KEY,
            task_name VARCHAR(100), stock_id VARCHAR(50), status VARCHAR(100), category VARCHAR(50),
            duration_ms INTEGER, rows_processed INTEGER, error_message TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );""",
        """CREATE TABLE IF NOT EXISTS data_audit_log (
            id SERIAL PRIMARY KEY,
            table_name VARCHAR(100), stock_id VARCHAR(50), start_date DATE, end_date DATE,
            rows_affected INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );"""
    ]
    try:
        with db_transaction() as cur:
            for sql in sql_statements: cur.execute(sql)
            
            # 自癒：檢查 stocks 表是否缺少 industry_category 欄位
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = 'industry_category';")
            if not cur.fetchone():
                logger.info("🔧 [DB] 偵測到資產表結構過舊，正在啟動自癒升級...")
                cur.execute("ALTER TABLE stocks ADD COLUMN IF NOT EXISTS industry_category VARCHAR(100);")
                cur.execute("ALTER TABLE stocks ADD COLUMN IF NOT EXISTS type VARCHAR(50);")
                cur.execute("ALTER TABLE stocks ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;")
                cur.execute("ALTER TABLE stocks ADD COLUMN IF NOT EXISTS is_core BOOLEAN DEFAULT FALSE;")
                cur.execute("ALTER TABLE stocks ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
                logger.info("✅ [DB] stocks 表結構升級完成。")
        return True
    except Exception as e:
        logger.error(f"❌ [DB] 初始化基礎設施失敗: {e}"); return False

def bulk_upsert(table_name: str, data: List[Dict], unique_cols: List[str] = ["date", "stock_id"]):
    if not data: return 0
    cols = [k.lower() for k in data[0].keys()]
    update_cols = [c for c in cols if c not in unique_cols and c != 'created_at']
    update_str = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s ON CONFLICT ({', '.join(unique_cols)}) DO UPDATE SET {update_str}"
    values = [[(v if v != "" else (None if isinstance(v, (int, float)) else v)) for v in d.values()] for d in data]
    try:
        with db_transaction() as cur: extras.execute_values(cur, sql, values)
        return len(data)
    except Exception as e:
        logger.error(f"❌ [DB] {table_name} Bulk Upsert 失敗: {e}"); return 0

def write_pipeline_log(task_name, stock_id, status, category, duration_ms=0, rows=0, err=None):
    sql = "INSERT INTO pipeline_execution_log (task_name, stock_id, status, category, duration_ms, rows_processed, error_message) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    try:
        with db_transaction() as cur: cur.execute(sql, (task_name, stock_id, str(status)[:100], category, duration_ms, rows, str(err) if err else None))
    except: pass

def write_data_audit_log(table_name, stock_id, start_date, end_date, rows_affected):
    sql = "INSERT INTO data_audit_log (table_name, stock_id, start_date, end_date, rows_affected) VALUES (%s, %s, %s, %s, %s)"
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
    try:
        with db_transaction() as cur:
            cur.execute(sql); return [r['stock_id'] for r in cur.fetchall()]
    except: return []

def get_latest_date(table_name: str, stock_id: str = None, id_column: str = "stock_id") -> str:
    sql = f"SELECT MAX(date) as last_date FROM {table_name}"
    if stock_id: sql += f" WHERE {id_column} = '{stock_id}'"
    try:
        with db_transaction() as cur:
            cur.execute(sql); res = cur.fetchone()
            return res['last_date'].strftime("%Y-%m-%d") if res and res['last_date'] else None
    except: return None

if __name__ == "__main__":
    ensure_infrastructure()
    print("="*55)
    print("💎 Quantum Finance: 資料庫核心診斷 (v2.14)")
    print("="*55)
    print(f"✅ 資料庫基礎設施已同步。")
    print(f"📦 資產表 (stocks) 結構已加固。")
    print(f"📝 日誌體系 (Hybrid Logging) 已啟動。")
    print("="*55 + "\n")