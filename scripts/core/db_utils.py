"""
db_utils.py v2.17 (Quantum Finance Edition)
================================================================================
資料庫核心工具箱 — 生產級自癒與混合日誌引擎 (Quantum v5.2 標準)
負責處理連線池、批量入庫 (Bulk Upsert)、以及基礎設施的自動診斷與修復。

修訂歷程：
  v2.17 (2026-05-11): [修復] 自動為 data_audit_log 加固 data_date, action_type 欄位，解決審計寫入失敗問題。
  v2.16 (2026-05-11): [修復] 修正連線參數讀取，對齊 .env 檔案 (DB_PASSWORD)。
  v2.15 (2026-05-11): [修復] 加固 pipeline_execution_log 的 error_msg 欄位。

【執行範例矩陣】
  1. [強制執行基礎設施修復與加固] $ python scripts/core/db_utils.py
================================================================================
"""
import os, sys, logging, time, re
from datetime import datetime, date
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv
from pathlib import Path

# ── 系統路徑自癒 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

load_dotenv(_SCRIPTS_DIR / ".env")

logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            database=os.getenv("DB_NAME", "stock"),
            user=os.getenv("DB_USER", "stock"),
            password=os.getenv("DB_PASSWORD", "stock"),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"❌ 資料庫連線失敗: {e}")
        raise e

@contextmanager
def db_transaction():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def ensure_infrastructure():
    """終極自癒引擎：確保所有核心資料表與欄位皆符合 Quantum 生產標準。"""
    try:
        with db_transaction() as cur:
            # 1. 加固 pipeline_execution_log
            cur.execute("CREATE TABLE IF NOT EXISTS pipeline_execution_log (id SERIAL PRIMARY KEY);")
            pipeline_cols = {
                "task_name": "VARCHAR(100)",
                "stock_id": "VARCHAR(50)",
                "status": "VARCHAR(50)",
                "category": "VARCHAR(50)",
                "duration_ms": "INTEGER",
                "rows_processed": "INTEGER",
                "error_msg": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
            for col, col_type in pipeline_cols.items():
                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'pipeline_execution_log' AND column_name = '{col}';")
                if not cur.fetchone():
                    logger.info(f"🔧 [DB] 偵測到日誌表缺失 {col}，正在執行自癒升級...")
                    cur.execute(f"ALTER TABLE pipeline_execution_log ADD COLUMN {col} {col_type};")

            # 2. 加固 data_audit_log (修復 data_date 缺失問題)
            cur.execute("CREATE TABLE IF NOT EXISTS data_audit_log (id SERIAL PRIMARY KEY);")
            audit_cols = {
                "table_name": "VARCHAR(100)",
                "stock_id": "VARCHAR(50)",
                "data_date": "DATE",
                "action_type": "VARCHAR(50)",
                "rows_affected": "INTEGER",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
            for col, col_type in audit_cols.items():
                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'data_audit_log' AND column_name = '{col}';")
                if not cur.fetchone():
                    logger.info(f"🔧 [DB] 偵測到審計表缺失 {col}，正在執行自癒升級...")
                    cur.execute(f"ALTER TABLE data_audit_log ADD COLUMN {col} {col_type};")

            # 3. 加固資產主表 (stocks)
            cur.execute("CREATE TABLE IF NOT EXISTS stocks (stock_id VARCHAR(50) PRIMARY KEY);")
            stock_cols = {
                "stock_name": "VARCHAR(100)",
                "industry_category": "VARCHAR(100)",
                "type": "VARCHAR(50)",
                "is_active": "BOOLEAN DEFAULT TRUE",
                "is_core": "BOOLEAN DEFAULT FALSE",
                "updated_at": "TIMESTAMP"
            }
            for col, col_type in stock_cols.items():
                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = '{col}';")
                if not cur.fetchone():
                    logger.info(f"🔧 [DB] 偵測到資產表缺失 {col}，正在執行自癒升級...")
                    cur.execute(f"ALTER TABLE stocks ADD COLUMN {col} {col_type};")

        return True
    except Exception as e:
        print(f"❌ 初始化基礎設施失敗: {e}")
        return False

# 其他函數保持穩定...
def bulk_upsert(table_name, data_list, unique_cols=["stock_id", "date"]):
    if not data_list: return 0
    with db_transaction() as cur:
        cols = data_list[0].keys()
        query = f"""
            INSERT INTO {table_name} ({', '.join(cols)})
            VALUES %s
            ON CONFLICT ({', '.join(unique_cols)}) DO UPDATE SET
            {', '.join([f"{col} = EXCLUDED.{col}" for col in cols if col not in unique_cols])}
        """
        vals = [tuple(d.values()) for d in data_list]
        execute_values(cur, query, vals)
        return len(data_list)

@contextmanager
def record_lifecycle(task_name, category="ingestion", stock_id="ALL"):
    t0 = time.monotonic()
    status = "success"
    rows = 0
    error = None
    try:
        yield
    except Exception as e:
        status = "failed"
        error = str(e)
        raise e
    finally:
        duration = int((time.monotonic() - t0) * 1000)
        write_pipeline_log(task_name, stock_id, status, category, duration, rows, error)

def write_pipeline_log(task_name, stock_id, status, category, duration, rows, error=None):
    try:
        with db_transaction() as cur:
            cur.execute("""
                INSERT INTO pipeline_execution_log 
                (task_name, stock_id, status, category, duration_ms, rows_processed, error_msg)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (task_name, stock_id, status, category, duration, rows, error))
    except Exception as e:
        print(f"❌ 寫入 Pipeline Log 失敗: {e}")

def write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected):
    try:
        with db_transaction() as cur:
            cur.execute("""
                INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected)
                VALUES (%s, %s, %s, %s, %s)
            """, (table_name, stock_id, data_date, action_type, rows_affected))
    except Exception as e:
        print(f"❌ 寫入 Audit Log 失敗: {e}")

def get_latest_date(table_name, stock_id):
    try:
        with db_transaction() as cur:
            cur.execute(f"SELECT MAX(date) as last_date FROM {table_name} WHERE stock_id = %s", (stock_id,))
            res = cur.fetchone()
            return res['last_date'] if res else None
    except: return None

def get_db_stock_ids(active_only=True):
    try:
        with db_transaction() as cur:
            query = "SELECT stock_id FROM stocks"
            if active_only: query += " WHERE is_active = TRUE"
            cur.execute(query)
            return [r['stock_id'] for r in cur.fetchall()]
    except: return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if ensure_infrastructure():
        print("="*55)
        print("💎 Quantum Finance: 資料庫核心診斷 (v2.17)")
        print("="*55)
        print("✅ 資料庫基礎設施已全量同步。")
        print("📦 審計表 (data_audit_log) 已加固核心欄位。")
        print("📝 日誌體系 (Hybrid Logging) 已啟動。")
        print("="*55 + "\n")