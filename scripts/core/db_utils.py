"""
db_utils.py v4.14 (Trinity Core Final)
================================================================================
資料庫維運核心 — 混合日誌與架構自癒版
負責提供 ThreadedConnectionPool、混合日誌寫入、以及資料表架構自動修復。

修訂歷程：
  v4.14 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明與 SQL 稽核指引。
  v4.13 (2026-05-09):
    - [核心] 補齊 created_at 與 extra_metadata 欄位自癒能力。

【執行範例說明】

1. 初始化資料庫架構（建立所有日誌表）：
   ------------------------------------------------------------
   from core.db_utils import ensure_ddl
   ensure_ddl()
   ------------------------------------------------------------

2. 寫入管線日誌 (Python 引用範例)：
   ------------------------------------------------------------
   from core.db_utils import write_pipeline_log
   write_pipeline_log("my_task", "2330", "success", "sys", 100)
   ------------------------------------------------------------

3. SQL 維運查閱 (快速診斷管線健康度)：
   -- 查看最近 20 筆管線執行紀錄
   SELECT * FROM pipeline_execution_log ORDER BY created_at DESC LIMIT 20;

   -- 統計各分類任務的成功率
   SELECT category, status, COUNT(*) FROM pipeline_execution_log GROUP BY category, status;
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
        # 1. 偵測 PID 變化或池已關閉
        if _POOL is None or _OWNER_PID != current_pid:
            _reset_pool(current_pid)
        else:
            # 2. [自癒強化] 檢查池中連線是否依然存活
            try:
                conn = _POOL.getconn()
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                _POOL.putconn(conn)
            except Exception:
                logging.warning("⚠️ 偵測到失效連線，正在重啟連線池...")
                _reset_pool(current_pid)
    return _POOL

def _reset_pool(current_pid):
    global _POOL, _OWNER_PID
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
    logging.info(f"✅ 連線池已重置與就緒 (PID: {current_pid})")

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
        print(f"[ERROR] DDL 失敗: {e}")

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