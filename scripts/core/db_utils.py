"""
core/db_utils.py — 統一的 PostgreSQL 工具函式 v2.0
================================================
v2.0 新增（優化報告建議）：
  - asyncpg 非同步驅動支援：
    · get_asyncpg_pool()：建立 asyncpg 連線池
    · async_bulk_upsert()：asyncpg 協程版批次寫入（直接 UPSERT）
    · async_bulk_copy_upsert()：二進位暫存表高速複製 + UPSERT
      繞過 SQL 字串解析開銷，大型資料集（衍生品明細）寫入速度提升 70%+
  - 向後相容：所有 psycopg2 同步函式介面不變

原有函式：
  - get_db_conn()、ensure_ddl()、bulk_upsert()
  - safe_float() / safe_int() / safe_date() / safe_timestamp()
  - get_all_latest_dates()、get_all_safe_starts()
  - resolve_start_cached()、resolve_start()
  - dedup_rows()、get_db_stock_ids()
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

import psycopg2
import psycopg2.extras

from config import DB_CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 同步連線（psycopg2）
# ─────────────────────────────────────────────
def get_db_conn() -> psycopg2.extensions.connection:
    """建立並回傳 PostgreSQL 同步連線（psycopg2）。"""
    return psycopg2.connect(**DB_CONFIG)


# ─────────────────────────────────────────────
# 非同步連線池（asyncpg）
# ─────────────────────────────────────────────
_asyncpg_pool = None
_asyncpg_pool_lock = None


async def get_asyncpg_pool(min_size: int = 2, max_size: int = 10):
    """
    取得（或建立）全域 asyncpg 連線池。
    首次呼叫時初始化，後續重用同一池。

    asyncpg 直接實作 PostgreSQL 二進位協定，
    比 psycopg2 字串型協定速度快 3-5 倍。

    Parameters
    ----------
    min_size : 連線池最小連線數（預設 2）
    max_size : 連線池最大連線數（預設 10）

    Returns
    -------
    asyncpg.Pool
    """
    import asyncio
    import asyncpg

    global _asyncpg_pool, _asyncpg_pool_lock

    if _asyncpg_pool_lock is None:
        _asyncpg_pool_lock = asyncio.Lock()

    async with _asyncpg_pool_lock:
        if _asyncpg_pool is None or _asyncpg_pool._closed:
            _asyncpg_pool = await asyncpg.create_pool(
                host=DB_CONFIG.get("host", "localhost"),
                port=DB_CONFIG.get("port", 5432),
                database=DB_CONFIG.get("dbname"),
                user=DB_CONFIG.get("user"),
                password=DB_CONFIG.get("password"),
                min_size=min_size,
                max_size=max_size,
                command_timeout=300,
            )
            logger.info(
                f"asyncpg 連線池建立完成（min={min_size}, max={max_size}）"
            )
    return _asyncpg_pool


async def close_asyncpg_pool() -> None:
    """關閉全域 asyncpg 連線池（程式結束前呼叫）。"""
    global _asyncpg_pool
    if _asyncpg_pool and not _asyncpg_pool._closed:
        await _asyncpg_pool.close()
        _asyncpg_pool = None
        logger.info("asyncpg 連線池已關閉")


# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
DDL_FETCH_LOG = """
CREATE TABLE IF NOT EXISTS fetch_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    table_name VARCHAR(50),
    stock_id VARCHAR(50),
    start_date DATE,
    end_date DATE,
    rows_count INTEGER,
    status VARCHAR(20),
    error_msg TEXT
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_table ON fetch_log (table_name, timestamp DESC);
"""


def ensure_ddl(conn, *ddls: str) -> None:
    """執行一或多個 DDL 語句（冪等，IF NOT EXISTS）。"""
    with conn.cursor() as cur:
        cur.execute(DDL_FETCH_LOG)
        for ddl in ddls:
            cur.execute(ddl)
    conn.commit()


def log_fetch_result(
    conn,
    table_name: str,
    stock_id: str,
    start_date: str,
    end_date: str,
    rows_count: int,
    status: str,
    error_msg: str = None,
) -> None:
    """將抓取結果記錄至 fetch_log 資料表。"""
    sql = """
    INSERT INTO fetch_log (table_name, stock_id, start_date, end_date, rows_count, status, error_msg)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table_name, stock_id, start_date, end_date, rows_count, status, error_msg))
    conn.commit()


# ─────────────────────────────────────────────
# 同步批次 Upsert（psycopg2）
# ─────────────────────────────────────────────
def bulk_upsert(
    conn,
    sql: str,
    rows: list,
    template: str,
    page_size: int = 2000,
) -> int:
    """
    使用 psycopg2 execute_values 批次寫入（同步）。

    Parameters
    ----------
    conn      : psycopg2 連線
    sql       : INSERT ... ON CONFLICT ... 語句
    rows      : tuple list
    template  : 欄位佔位符（如 "(%s::date, %s, %s::numeric)"）
    page_size : 每批寫入筆數（預設 2000）

    Returns
    -------
    int  實際寫入筆數
    """
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, sql, rows, template=template, page_size=page_size
        )
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────
# 非同步批次 Upsert（asyncpg）
# ─────────────────────────────────────────────
async def async_bulk_upsert(
    pool,
    sql: str,
    rows: list[tuple],
) -> int:
    """
    asyncpg 非同步批次 UPSERT（適用中小型資料集）。

    Parameters
    ----------
    pool : asyncpg.Pool（由 get_asyncpg_pool() 取得）
    sql  : $1, $2, ... 佔位符的 INSERT ... ON CONFLICT 語句
    rows : tuple list

    Returns
    -------
    int  寫入筆數

    Example
    -------
    sql = '''
        INSERT INTO stock_price (date, stock_id, close)
        VALUES ($1, $2, $3::numeric)
        ON CONFLICT (date, stock_id) DO UPDATE SET close = EXCLUDED.close
    '''
    await async_bulk_upsert(pool, sql, [(date, sid, price), ...])
    """
    if not rows:
        return 0
    async with pool.acquire() as conn:
        await conn.executemany(sql, rows)
    return len(rows)


async def async_bulk_copy_upsert(
    pool,
    staging_table: str,
    target_table: str,
    columns: list[str],
    rows: list[tuple],
    conflict_columns: list[str],
    update_columns: list[str],
) -> int:
    """
    asyncpg 二進位暫存表高速複製 + UPSERT（適用大型資料集）。

    流程：
      1. 建立無約束暫存表（TEMP TABLE LIKE target_table）
      2. 利用 copy_records_to_table 以二進位串流高速寫入暫存表
         （繞過 SQL 解析，速度比 execute_values 快 3-5 倍）
      3. 單一 INSERT ... ON CONFLICT DO UPDATE 將暫存表資料 UPSERT 至目標表
      4. 自動清理暫存表（SESSION 結束自動刪除）

    Parameters
    ----------
    pool             : asyncpg.Pool
    staging_table    : 暫存表名稱（如 "_tmp_stock_price"）
    target_table     : 目標表名稱（如 "stock_price"）
    columns          : 欄位名稱列表（如 ["date", "stock_id", "close"]）
    rows             : 資料 tuple list
    conflict_columns : 衝突判斷欄位（如 ["date", "stock_id"]）
    update_columns   : 衝突時更新的欄位（如 ["close"]）

    Returns
    -------
    int  寫入筆數
    """
    if not rows:
        return 0

    cols_csv = ", ".join(columns)
    conflict_csv = ", ".join(conflict_columns)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in update_columns
    )

    async with pool.acquire() as conn:
        async with conn.transaction():
            # 建立無約束暫存表
            await conn.execute(
                f"""
                CREATE TEMP TABLE {staging_table}
                (LIKE {target_table} INCLUDING DEFAULTS)
                ON COMMIT DROP
                """
            )
            # 二進位串流高速複製
            await conn.copy_records_to_table(
                staging_table,
                records=rows,
                columns=columns,
            )
            # UPSERT 至目標表
            result = await conn.execute(
                f"""
                INSERT INTO {target_table} ({cols_csv})
                SELECT {cols_csv} FROM {staging_table}
                ON CONFLICT ({conflict_csv}) DO UPDATE SET {update_set}
                """
            )
            # result 格式：'INSERT 0 N'
            try:
                count = int(result.split()[-1])
            except (IndexError, ValueError):
                count = len(rows)

    return count


# ─────────────────────────────────────────────
# 型別轉換工具
# ─────────────────────────────────────────────
def safe_float(val) -> float | None:
    """安全轉換為 float，無效值回傳 None。"""
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NONE", "NAN", ""):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def safe_int(val) -> int | None:
    """安全轉換為 int，無效值回傳 None。"""
    f = safe_float(val)
    return int(f) if f is not None else None


def safe_bigint(val) -> int | None:
    """同 safe_int，明確用於 BIGINT 欄位。"""
    return safe_int(val)


def safe_date(val) -> str | None:
    """安全解析日期字串，回傳 "YYYY-MM-DD" 或 None。"""
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NONE", "NAN", ""):
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except (TypeError, ValueError):
        return None


def safe_timestamp(val) -> datetime | None:
    """
    安全解析時間戳字串，回傳 datetime 或 None。
    支援格式："%Y-%m-%d %H:%M:%S" / "%H:%M:%S"
    """
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NONE", "NAN", ""):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except (TypeError, ValueError):
            continue
    return None


# ─────────────────────────────────────────────
# 增量更新輔助
# ─────────────────────────────────────────────
def get_all_latest_dates(conn, table: str) -> dict[str, str]:
    """查出所有股票的最新日期。回傳 { stock_id: "YYYY-MM-DD" }"""
    with conn.cursor() as cur:
        cur.execute(f"SELECT stock_id, MAX(date) FROM {table} GROUP BY stock_id")
        return {
            row[0]: row[1].strftime("%Y-%m-%d")
            for row in cur.fetchall()
            if row[1] is not None
        }


def get_all_safe_starts(
    conn, table: str, window_days: int = 60, gap_interval: str = "1 day"
) -> dict[str, str]:
    """
    [v3 Trinity] 智能偵測起始點，支援不同頻率（1 day, 1 month, 3 months）。
    偵測最近 window_days 內的「第一個斷層（Gap）」，而非單純取 MAX(date)+1。
    回傳 dict: { stock_id: "YYYY-MM-DD" }
    """
    dow_filter = (
        "AND extract(dow from t1.date + interval '1 day') NOT IN (0, 6)"
        if gap_interval == "1 day" else ""
    )

    sql = f"""
    WITH gaps AS (
        SELECT
            t1.stock_id,
            MIN(t1.date + interval '{gap_interval}') as gap_start
        FROM {table} t1
        WHERE t1.date >= CURRENT_DATE - interval '{window_days} days'
          AND t1.date < CURRENT_DATE
          AND NOT EXISTS (
              SELECT 1 FROM {table} t2
              WHERE t2.stock_id = t1.stock_id
                AND t2.date = t1.date + interval '{gap_interval}'
          )
          {dow_filter}
        GROUP BY t1.stock_id
    ),
    max_dates AS (
        SELECT stock_id, MAX(date) as last_date FROM {table} GROUP BY stock_id
    )
    SELECT
        m.stock_id,
        COALESCE(g.gap_start, m.last_date + interval '{gap_interval}') as safe_start
    FROM max_dates m
    LEFT JOIN gaps g ON m.stock_id = g.stock_id
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return {
            row[0]: row[1].strftime("%Y-%m-%d")
            for row in cur.fetchall()
            if row[1] is not None
        }


def get_latest_date(conn, table: str, stock_id: str) -> str | None:
    """查詢單支股票的最新日期，無資料回傳 None。"""
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(date) FROM {table} WHERE stock_id = %s", (stock_id,))
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
    return None


def get_latest_date_by_col(conn, table: str, id_col: str, data_id: str) -> str | None:
    """查詢非 stock_id 主鍵欄位（如 country、currency）的最新日期。"""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MAX(date) FROM {table} WHERE {id_col} = %s",
            (data_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
    return None


def get_market_safe_start(conn, table: str, window_days: int = 60) -> str | None:
    """
    [v3 Trinity] 查詢市場層資料表（無 stock_id）的「安全起始日」。
    """
    sql = f"""
    WITH gap AS (
        SELECT MIN(t1.date + interval '1 day') as gap_start
        FROM {table} t1
        WHERE t1.date >= CURRENT_DATE - interval '{window_days} days'
          AND t1.date < CURRENT_DATE
          AND NOT EXISTS (
              SELECT 1 FROM {table} t2
              WHERE t2.date = t1.date + interval '1 day'
          )
          AND extract(dow from t1.date + interval '1 day') NOT IN (0, 6)
    )
    SELECT COALESCE(
        (SELECT gap_start FROM gap),
        (SELECT MAX(date) + interval '1 day' FROM {table})
    )
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
    return None


def get_safe_start(
    conn, table: str, stock_id: str, window_days: int = 60, gap_interval: str = "1 day"
) -> str | None:
    """
    [v3 Trinity] 查詢單支股票的「安全起始日」，支援多頻率。
    """
    dow_filter = (
        "AND extract(dow from t1.date + interval '1 day') NOT IN (0, 6)"
        if gap_interval == "1 day" else ""
    )

    sql = f"""
    WITH gap AS (
        SELECT MIN(t1.date + interval '{gap_interval}') as gap_start
        FROM {table} t1
        WHERE t1.stock_id = %s
          AND t1.date >= CURRENT_DATE - interval '{window_days} days'
          AND t1.date < CURRENT_DATE
          AND NOT EXISTS (
              SELECT 1 FROM {table} t2
              WHERE t2.stock_id = t1.stock_id
                AND t2.date = t1.date + interval '{gap_interval}'
          )
          {dow_filter}
    )
    SELECT COALESCE(
        (SELECT gap_start FROM gap),
        (SELECT MAX(date) + interval '{gap_interval}' FROM {table} WHERE stock_id = %s)
    )
    """
    with conn.cursor() as cur:
        cur.execute(sql, (stock_id, stock_id))
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
    return None


def resolve_start_cached(
    stock_id: str,
    latest_dates: dict,
    global_start: str,
    dataset_earliest: str,
    force: bool,
    today: str | None = None,
) -> str | None:
    """
    從預載快取（latest_dates）決定起始日，不再每支逐筆查 DB。

    Returns
    -------
    str   需抓取的起始日 "YYYY-MM-DD"
    None  表示此股票已是最新，無需抓取
    """
    _today = today or date.today().strftime("%Y-%m-%d")
    effective_start = max(global_start, dataset_earliest)

    if force:
        return effective_start

    safe_start = latest_dates.get(str(stock_id))

    if safe_start is None:
        return effective_start

    if safe_start > _today:
        return None

    return max(safe_start, dataset_earliest)


def resolve_start(
    conn,
    table: str,
    id_col: str,
    data_id: str,
    global_start: str,
    force: bool,
    today: str | None = None,
) -> str | None:
    """
    直接查 DB 決定起始日（適用於非逐股票的資料集，如宏觀資料）。

    Returns
    -------
    str   需抓取的起始日 "YYYY-MM-DD"
    None  表示已是最新，無需抓取
    """
    _today = today or date.today().strftime("%Y-%m-%d")

    if force:
        return global_start

    latest = get_latest_date_by_col(conn, table, id_col, data_id)
    if latest is None:
        return global_start

    next_day = (
        datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if next_day > _today:
        return None

    return max(next_day, global_start)


# ─────────────────────────────────────────────
# 資料庫輔助
# ─────────────────────────────────────────────
def dedup_rows(rows: list, key_indices: tuple) -> list:
    """
    依指定欄位索引去除重複列（後出現的覆蓋先出現的）。
    解決 FinMind 偶爾回傳同批資料含重複 PK 的問題。
    """
    seen = {}
    for row in rows:
        key = tuple(row[i] for i in key_indices)
        seen[key] = row
    return list(seen.values())


def get_db_stock_ids(conn, types: tuple = ("twse", "otc")) -> list[str]:
    """從 stock_info 取得所有指定類型的股票代號清單。"""
    placeholders = ", ".join(["%s"] * len(types))
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT stock_id FROM stock_info WHERE type IN ({placeholders}) ORDER BY stock_id",
            types,
        )
        return [row[0] for row in cur.fetchall()]
