"""
db_utils.py v4.7 (Trinity Core Edition)
================================================================================
資料庫核心工具組：全管線日誌系統 (Hybrid Logging Architecture)
對接 Trinity Core v5.5 規範，實現「生命週期監控」與「業務指標分析」的雙層管理。

核心功能：
  · 統一執行日誌 (pipeline_execution_log) ─ 紀錄全系統 (抓取、特徵、訓練、回測) 的啟動與狀態。
  · 專項結果日誌 (evaluation_log)        ─ 紀錄量化模型的回測指標 (Sharpe, MDD, Returns)。
  · 自動事務控制與連線池管理               ─ 確保高併發下的數據一致性。

修訂歷程：
  v4.7 (2026-05-09):
    - [核心] 導入「混合模式」日誌架構，新增 pipeline_execution_log 與 evaluation_log。
    - [監控] 新增 write_pipeline_log 與 write_evaluation_log 標準介面。
  v4.6 (2026-05-09):
    - [監控] write_fetch_log 補齊 rows_inserted 欄位。
"""

import os
import logging
import time
import psycopg2
import psycopg2.extras
import psycopg2.pool
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# =====================================================================
# 1. 連線池與事務管理
# =====================================================================

class DatabaseManager:
    _pool = None
    @classmethod
    def get_pool(cls):
        if cls._pool is None:
            cls._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20,
                dbname=os.getenv("DB_NAME", "stock"),
                user=os.getenv("DB_USER", "stock"),
                password=os.getenv("DB_PASSWORD", "stock"),
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432")
            )
            logger.info("✅ 成功初始化 ThreadedConnectionPool (v4.7 Hybrid Log Aligned)")
        return cls._pool

@contextmanager
def db_session():
    pool = DatabaseManager.get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

@contextmanager
def db_transaction():
    with db_session() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur

# =====================================================================
# 2. DDL 與標準化日誌 (Logging Architecture)
# =====================================================================

DDL_LOGGING_SYSTEM = """
-- 1. 統一執行日誌 (生命週期)
CREATE TABLE IF NOT EXISTS pipeline_execution_log (
    run_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    task_name VARCHAR(100),
    stock_id VARCHAR(50),
    status VARCHAR(50),
    category VARCHAR(50), -- fetch, feature, train, backtest, inference, sys
    duration_ms INTEGER,
    rows_processed INTEGER DEFAULT 0,
    error_message TEXT,
    PRIMARY KEY (run_ts, task_name, stock_id)
);

-- 2. 專項回測指標日誌 (業務成果)
CREATE TABLE IF NOT EXISTS evaluation_log (
    run_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    stock_id VARCHAR(50) NOT NULL,
    model_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    sharpe_ratio NUMERIC(10,4),
    max_drawdown NUMERIC(10,4),
    total_return NUMERIC(10,4),
    win_rate NUMERIC(10,4),
    metrics_json JSONB, -- 彈性儲存其他指標
    PRIMARY KEY (run_ts, stock_id)
);

-- 3. 舊版相容 View (選擇性)
CREATE TABLE IF NOT EXISTS fetch_log (LIKE pipeline_execution_log INCLUDING ALL);
"""

def ensure_ddl(ddl: str = DDL_LOGGING_SYSTEM):
    with db_transaction() as cur:
        try:
            cur.execute("SAVEPOINT sp_ddl;")
            cur.execute(ddl)
            cur.execute("RELEASE SAVEPOINT sp_ddl;")
        except Exception as e:
            cur.execute("ROLLBACK TO SAVEPOINT sp_ddl;")
            logger.debug(f"DDL 略過: {e}")

def write_pipeline_log(task_name: str, stock_id: Optional[str], status: str, 
                       category: str, duration_ms: int, rows: int = 0, err: Optional[str] = None):
    """紀錄全管線生命週期"""
    sql = """
    INSERT INTO pipeline_execution_log (task_name, stock_id, status, category, duration_ms, rows_processed, error_message)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with db_transaction() as cur:
        cur.execute(sql, (task_name, stock_id, status, category, duration_ms, rows, err))

def write_evaluation_log(stock_id: str, model_name: str, sharpe: float, mdd: float, 
                         ret: float, win_rate: float = 0, start: str = None, end: str = None, extra: dict = None):
    """紀錄量化回測指標成果"""
    import json
    sql = """
    INSERT INTO evaluation_log (stock_id, model_name, sharpe_ratio, max_drawdown, total_return, win_rate, start_date, end_date, metrics_json)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with db_transaction() as cur:
        cur.execute(sql, (stock_id, model_name, sharpe, mdd, ret, win_rate, start, end, json.dumps(extra or {})))

# 為舊腳本提供相容性
def write_fetch_log(table_name, stock_id, status, fetch_mode, fetch_date_from, fetch_date_to, duration_ms, rows_inserted=0, error_message=None):
    write_pipeline_log(table_name, stock_id, status, "fetch", duration_ms, rows_inserted, error_message)

# =====================================================================
# 3. 數據查詢與轉換 (省略部分未變動工具函式以節省空間)
# =====================================================================

def get_latest_date(table_name: str, stock_id: Optional[str] = None, id_column: str = "stock_id") -> Optional[str]:
    sql = f"SELECT MAX(date) FROM {table_name}"
    params = []
    if stock_id:
        sql += f" WHERE {id_column} = %s"
        params.append(stock_id)
    with db_transaction() as cur:
        cur.execute(sql, params)
        res = cur.fetchone()
        if res and res['max']:
            mdate = res['max']
            return mdate.strftime("%Y-%m-%d") if hasattr(mdate, 'strftime') else str(mdate)
        return None

def safe_float(val: Any) -> Optional[float]:
    if val is None or val == "": return None
    try: return float(val)
    except: return None

def safe_int(val: Any) -> Optional[int]:
    f = safe_float(val)
    return int(f) if f is not None else None

def commit_per_stock_per_day(table_name: str, records: List[Dict[str, Any]], upsert_query: str, stock_id: str) -> Tuple[int, int]:
    if not records: return 0, 0
    try:
        with db_transaction() as cur:
            psycopg2.extras.execute_batch(cur, upsert_query, records)
            return len(records), 0
    except Exception as e:
        logger.warning(f"  ⚠️ [批次失敗] {stock_id} @ {table_name}，嘗試逐筆：{e}")
    s_count = e_count = 0
    for rec in records:
        try:
            with db_transaction() as cur:
                cur.execute(upsert_query, rec)
                s_count += 1
        except: e_count += 1
    return s_count, e_count

def get_db_stock_ids() -> List[str]:
    with db_transaction() as cur:
        cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE ORDER BY stock_id")
        return [r['stock_id'] for r in cur.fetchall()]