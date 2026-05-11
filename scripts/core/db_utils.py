"""
db_utils.py v2.25 (Quantum Finance Edition)
================================================================================
資料庫核心引擎 — 全功能恢復版 (Quantum v5.2 標準)
負責管理資產元數據、結構自癒與高效能批量寫入。

修訂歷程：
  v2.25 (2026-05-11): [修復] 補齊遺漏的 bulk_upsert, get_latest_date 等核心函式。
  v2.24 (2026-05-11): [環境] 修正 .env 加載路徑。
================================================================================
"""
import os, sys, logging, time, platform, json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool, extras
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# ── 智能根目錄偵測 ──
_THIS_DIR = Path(__file__).resolve().parent
def find_project_root(current_path: Path) -> Path:
    for parent in [current_path] + list(current_path.parents):
        if (parent / ".env").exists(): return parent
    return current_path.parent
_PROJECT_ROOT = find_project_root(_THIS_DIR)
load_dotenv(_PROJECT_ROOT / ".env")

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME = os.getenv("DB_NAME", "stock_db")
DB_USER = os.getenv("DB_USER", "stock")
DB_PASSWORD = os.getenv("DB_PASSWORD", "stock")
DB_PORT = os.getenv("DB_PORT", "5432")

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        try:
            _pool = pool.ThreadedConnectionPool(1, 20, host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        except Exception as e:
            logging.error(f"資料庫連線池初始化失敗: {e}")
            raise e
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
    """基礎設施自癒：擴充 stocks 與日誌表格。"""
    healed_count = 0
    try:
        with db_transaction() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS stocks (stock_id VARCHAR(20) PRIMARY KEY);")
            stock_cols = {
                "name": "VARCHAR(100)", "industry": "VARCHAR(100)", 
                "us_chain_tickers": "TEXT", "use_adr_premium": "BOOLEAN DEFAULT FALSE",
                "is_core": "BOOLEAN DEFAULT FALSE", "is_active": "BOOLEAN DEFAULT TRUE"
            }
            for col, col_type in stock_cols.items():
                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = '{col}';")
                if not cur.fetchone(): 
                    cur.execute(f"ALTER TABLE stocks ADD COLUMN {col} {col_type};"); healed_count += 1
            
            cur.execute("CREATE TABLE IF NOT EXISTS pipeline_execution_log (id SERIAL PRIMARY KEY, task_name VARCHAR(100), stock_id VARCHAR(20), status VARCHAR(20), category VARCHAR(50), duration_ms INTEGER, rows_affected INTEGER, error_msg TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
            cur.execute("CREATE TABLE IF NOT EXISTS data_audit_log (id SERIAL PRIMARY KEY, table_name VARCHAR(100), stock_id VARCHAR(20), data_date DATE, action_type VARCHAR(50), rows_affected INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
        return healed_count
    except Exception as e:
        logging.error(f"基礎設施修復失敗: {e}"); return -1

def get_db_stock_ids(is_core=None, active_only=True):
    sql = "SELECT stock_id FROM stocks WHERE 1=1"
    params = []
    if is_core is not None:
        sql += " AND is_core = %s"; params.append(is_core)
    if active_only:
        sql += " AND is_active = TRUE"
    with db_transaction() as cur:
        cur.execute(sql, params); return [r['stock_id'] for r in cur.fetchall()]

def bulk_upsert(table_name: str, data: List[Dict], unique_cols: List[str]):
    """核心高效能批量更新函式。"""
    if not data: return 0
    cols = data[0].keys()
    query = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s ON CONFLICT ({', '.join(unique_cols)}) DO UPDATE SET {', '.join([f'{c} = EXCLUDED.{c}' for c in cols if c not in unique_cols])}"
    with db_transaction() as cur:
        execute_values(cur, query, [[d.get(c) for c in cols] for d in data])
        return cur.rowcount

def get_latest_date(table_name: str, stock_id: str):
    """獲取特定標的之最新數據日期。"""
    with db_transaction() as cur:
        cur.execute(f"SELECT MAX(date) as last_date FROM {table_name} WHERE stock_id = %s", (stock_id,))
        res = cur.fetchone(); return res['last_date'] if res else None

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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    h_count = ensure_infrastructure()
    print("\n" + "💎"*40)
    print(f"🚀 Quantum Finance: 資料庫核心診斷報告 (v2.25)")
    print(f"✅ 執行結果  : {'SUCCESS' if h_count >= 0 else 'FAILED'}")
    print(f"🖥️  操作系統  : {platform.system()} {platform.release()}")
    print(f"🔧 結構自癒  : {max(0, h_count)} 處欄位已同步")
    print("-" * 80)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (DB_HEAL)")
    print("💎"*40 + "\n")