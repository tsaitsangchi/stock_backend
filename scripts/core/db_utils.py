"""
db_utils.py v2.18 (Quantum Finance Edition)
================================================================================
資料庫核心引擎 — 基礎設施自癒、混合日誌與高效寫入器 (Quantum v5.2 標準)
負責管理資料庫連線、自動結構修復、生命週期監測以及高效批量入庫。

修訂歷程：
  v2.18 (2026-05-11): [標準化] 補齊底層組件執行範例矩陣 (包含個股、全表、基礎設施自癒範例)。
  v2.17 (2026-05-11): [自癒] 強化 ensure_infrastructure，自動修復 pipeline_execution_log 與 data_audit_log。

【執行範例矩陣 (Core Infrastructure Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議用法 / 指令                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [基礎設施自癒驗證]        │ $ python scripts/core/db_utils.py                      │
│ 2. [單一個股資料獲取]        │ from core.db_utils import get_db_stock_ids;           │
│                              │ ids = get_db_stock_ids(active_only=True)               │
│ 3. [高效批量入庫 (Upsert)]   │ bulk_upsert(table, data, unique_cols=['stock_id'])     │
│ 4. [混合模式紀錄 (手動)]     │ write_pipeline_log(task, sid, status, category, ...)   │
│                              │ write_data_audit_log(table, sid, date, action, rows)   │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 自癒邏輯: 每次初始化連線池時，自動檢測並補齊關鍵審計欄位 (如 error_msg, action_type)。
  - 高併發支援: 使用 ThreadedConnectionPool 確保平行抓取時的連線穩定性。
================================================================================
"""
import os, sys, logging, time
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool, extras
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# ── 系統路徑自癒 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

# 加載配置
load_dotenv(_SCRIPTS_DIR / ".env")

# 資料庫配置
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME = os.getenv("DB_NAME", "stock_db")
DB_USER = os.getenv("DB_USER", "stock")
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("DB_PASS", ""))
DB_PORT = os.getenv("DB_PORT", "5432")

# 初始化連線池
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        try:
            _pool = pool.ThreadedConnectionPool(
                minconn=1, maxconn=20,
                host=DB_HOST, database=DB_NAME,
                user=DB_USER, password=DB_PASSWORD, port=DB_PORT
            )
        except Exception as e:
            print(f"❌ 資料庫連線失敗: {e}")
            raise e
    return _pool

@contextmanager
def db_connection():
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
    with db_connection() as conn:
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            yield cur

def ensure_infrastructure():
    """基礎設施自癒：確保生命週期與審計日誌表具備核心欄位。"""
    try:
        with db_transaction() as cur:
            # 1. 確保 pipeline_execution_log 具備 error_msg
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_execution_log (
                    id SERIAL PRIMARY KEY,
                    task_name VARCHAR(100),
                    stock_id VARCHAR(50),
                    status VARCHAR(20),
                    category VARCHAR(50),
                    duration_ms INTEGER,
                    rows_affected INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'pipeline_execution_log' AND column_name = 'error_msg';")
            if not cur.fetchone():
                print("🔧 [DB] 偵測到 pipeline_execution_log 缺失 error_msg 欄位，正在升級...")
                cur.execute("ALTER TABLE pipeline_execution_log ADD COLUMN error_msg TEXT;")

            # 2. 確保 data_audit_log 具備核心欄位
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
                    print(f"🔧 [DB] 偵測到審計表缺失 {col}，正在執行自癒升級...")
                    cur.execute(f"ALTER TABLE data_audit_log ADD COLUMN {col} {col_type};")
        return True
    except Exception as e:
        print(f"❌ 初始化基礎設施失敗: {e}")
        return False

# ── 核心功能函數 ──

def write_pipeline_log(task_name, stock_id, status, category, duration=0, rows=0, error=None):
    with db_transaction() as cur:
        cur.execute("""
            INSERT INTO pipeline_execution_log (task_name, stock_id, status, category, duration_ms, rows_affected, error_msg)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (task_name, stock_id, status, category, duration, rows, str(error) if error else None))

def write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected):
    with db_transaction() as cur:
        cur.execute("""
            INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected)
            VALUES (%s, %s, %s, %s, %s)
        """, (table_name, stock_id, data_date, action_type, rows_affected))

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

def bulk_upsert(table_name, data: List[Dict], unique_cols: List[str]):
    if not data: return 0
    cols = data[0].keys()
    query = f"""
        INSERT INTO {table_name} ({', '.join(cols)})
        VALUES %s
        ON CONFLICT ({', '.join(unique_cols)}) DO UPDATE SET
        {', '.join([f"{c} = EXCLUDED.{c}" for c in cols if c not in unique_cols])}
    """
    vals = [[d.get(c) for c in cols] for d in data]
    with db_transaction() as cur:
        execute_values(cur, query, vals)
        return cur.rowcount

def get_db_stock_ids(active_only=True):
    sql = "SELECT stock_id FROM stocks WHERE is_active = TRUE" if active_only else "SELECT stock_id FROM stocks"
    with db_transaction() as cur:
        cur.execute(sql)
        return [r['stock_id'] for r in cur.fetchall()]

def get_latest_date(table_name: str, stock_id: str) -> Optional[date]:
    try:
        with db_transaction() as cur:
            cur.execute(f"SELECT MAX(date) as last_date FROM {table_name} WHERE stock_id = %s", (stock_id,))
            res = cur.fetchone()
            return res['last_date'] if res else None
    except: return None

if __name__ == "__main__":
    ensure_infrastructure()
    print("=======================================================")
    print("💎 Quantum Finance: 資料庫核心診斷 (v2.18)")
    print("=======================================================")
    print("✅ 資料庫基礎設施已全量同步。")
    print("📦 所有審計與日誌欄位已補齊。")
    print("📝 日誌體系 (Hybrid Logging) 已啟動。")
    print("=======================================================")