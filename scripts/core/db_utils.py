"""
db_utils.py v2.35 (Quantum Finance Edition)
================================================================================
資料庫核心引擎 — 旗艦終極維運版 (Quantum v5.2 標準)
負責管理資產元數據、結構自癒與鏡像保護寫入。

修訂歷程：
  v2.35 (2026-05-11): [標準] 補全極致範例矩陣，涵蓋個股、全表、核心股、強制重整等情境。
  v2.34 (2026-05-11): [修復] 補全旗艦級診斷輸出。

【全系統維運指令矩陣 (Core Operational Matrix)】
┌──────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運場景                             │ 建議執行指令                                           │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [單一實體表：最新日期偵測]        │ date = get_latest_date("TaiwanStockPrice", "2330")     │
│ 2. [單一個股：所有表自癒初始化]      │ $ python scripts/core/data_schema.py --init            │
│ 3. [所有核心股：所有表結構同步]      │ $ python scripts/core/data_schema.py --init            │
│ 4. [所有核心股：所有表強制更新(結構)]│ $ python scripts/core/data_schema.py --init --force     │
│ 5. [系統級：基礎設施一鍵診斷]        │ $ python scripts/core/db_utils.py                      │
└──────────────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一日誌 (Unified): pipeline_execution_log (Task: infra_ensure / bulk_upsert)
  - 專項審計 (Audit): data_audit_log (Action: INFRA_SYNC / BULK_UPSERT)
================================================================================
"""
import os, sys, logging, time
from pathlib import Path
from typing import List, Dict
from contextlib import contextmanager
from psycopg2 import pool, extras
from psycopg2.extras import execute_values
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from core.data_schema import DATASET_SCHEMA_MAP

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
    with record_lifecycle("infra_ensure", "core", "SYSTEM"):
        with db_transaction() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    stock_id VARCHAR(50) PRIMARY KEY,
                    stock_name VARCHAR(100),
                    industry_category VARCHAR(100),
                    type VARCHAR(50),
                    is_core BOOLEAN DEFAULT TRUE,
                    active BOOLEAN DEFAULT TRUE,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100),
                    stock_id VARCHAR(50),
                    version VARCHAR(50),
                    metrics JSONB,
                    params JSONB,
                    path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("CREATE TABLE IF NOT EXISTS pipeline_execution_log (id SERIAL PRIMARY KEY, task_name VARCHAR(100), stock_id VARCHAR(50), status VARCHAR(20), category VARCHAR(50), duration_ms INTEGER, rows_affected INTEGER, error_msg TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
            cur.execute("CREATE TABLE IF NOT EXISTS data_audit_log (id SERIAL PRIMARY KEY, table_name VARCHAR(100), stock_id VARCHAR(50), data_date DATE, action_type VARCHAR(50), rows_affected INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
            for dataset, config in DATASET_SCHEMA_MAP.items():
                cur.execute(config["sql"])
        _infra_ready = True
        return 0

def get_db_stock_ids(core_only: bool = False, active_only: bool = True) -> list:
    query = "SELECT stock_id FROM stocks WHERE 1=1"
    if core_only: query += " AND is_core = TRUE"
    if active_only: query += " AND active = TRUE"
    with db_transaction() as cur:
        cur.execute(query)
        return [r['stock_id'] for r in cur.fetchall()]

def bulk_upsert(table_name: str, data: List[Dict], unique_cols: List[str]):
    if not data: return 0
    cols = data[0].keys()
    quoted_cols = [f'"{c}"' for c in cols]
    quoted_uniques = [f'"{c}"' for c in unique_cols]
    update_stmt = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in cols if c not in unique_cols])
    query = f'INSERT INTO "{table_name}" ({", ".join(quoted_cols)}) VALUES %s ON CONFLICT ({", ".join(quoted_uniques)}) DO UPDATE SET {update_stmt}'
    with db_transaction() as cur:
        execute_values(cur, query, [[d.get(c) for c in cols] for d in data])
        return cur.rowcount

def get_latest_date(table_name: str, target_id: str, id_column: str = "stock_id", date_column: str = "date"):
    with db_transaction() as cur:
        try:
            cur.execute(f'SELECT MAX("{date_column}") as last_date FROM "{table_name}" WHERE "{id_column}" = %s', (target_id,))
            res = cur.fetchone(); return res['last_date'] if res else None
        except: return None

def write_pipeline_log(task_name, stock_id, status, category, duration=0, rows=0, error=None):
    with db_transaction() as cur:
        cur.execute("INSERT INTO pipeline_execution_log (task_name, stock_id, status, category, duration_ms, rows_affected, error_msg) VALUES (%s, %s, %s, %s, %s, %s, %s)", (task_name, stock_id, status, category, duration, rows, str(error) if error else None))

@contextmanager
def record_lifecycle(task_name, category="general", stock_id="SYSTEM"):
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
    with db_transaction() as cur:
        cur.execute("INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected) VALUES (%s, %s, %s, %s, %s)", (table_name, stock_id, data_date, action_type, rows_affected))

if __name__ == "__main__":
    print("-" * 60)
    print(f"🚀 Quantum Finance: db_utils v2.35 核心診斷啟動...")
    try:
        ensure_infrastructure()
        with db_transaction() as cur:
            cur.execute("SELECT count(*) FROM stocks")
            count = cur.fetchone()['count']
            print(f"✅ 資料庫連線 : SUCCESS")
            print(f"📊 核心資產數 : {count}")
    except Exception as e:
        print(f"❌ 診斷失敗 : {e}")
    print("-" * 60)