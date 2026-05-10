"""
core/db_utils.py — 統一的 PostgreSQL 工具函式
================================================
[P3 修正] 取代散落在 10 個 fetch_*.py 中重複的 DB 工具函式：
  - get_db_conn()
  - ensure_ddl()
  - bulk_upsert()
  - safe_float() / safe_int() / safe_date() / safe_timestamp()
  - get_all_latest_dates()
  - resolve_start_cached()
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import psycopg2
import psycopg2.extras

from config import DB_CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 連線
# ─────────────────────────────────────────────
def get_db_conn() -> psycopg2.extensions.connection:
    """建立並回傳 PostgreSQL 連線。"""
    return psycopg2.connect(**DB_CONFIG)


# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
def ensure_ddl(conn, *ddls: str) -> None:
    """執行一或多個 DDL 語句（冪等，IF NOT EXISTS）。"""
    with conn.cursor() as cur:
        for ddl in ddls:
            cur.execute(ddl)
    conn.commit()


# ─────────────────────────────────────────────
# 批次 Upsert
# ─────────────────────────────────────────────
def bulk_upsert(
    conn,
    sql: str,
    rows: list,
    template: str,
    page_size: int = 2000,
) -> int:
    """
    使用 execute_values 批次寫入。

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
    """
    安全解析日期字串，回傳 "YYYY-MM-DD" 或 None。
    """
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
    """
    一次查出指定資料表所有股票的最新日期（GROUP BY 優化）。
    回傳 dict: { stock_id: "YYYY-MM-DD" }
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT stock_id, MAX(date) FROM {table} GROUP BY stock_id")
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

    Parameters
    ----------
    stock_id        : 目標股票代號
    latest_dates    : get_all_latest_dates() 回傳的快取 dict
    global_start    : CLI 傳入的 --start 參數
    dataset_earliest: 該資料集 API 最早可用日期
    force           : True → 強制從 global_start 重抓
    today           : 今日日期字串（預設自動取得）

    Returns
    -------
    str   需抓取的起始日 "YYYY-MM-DD"
    None  表示此股票已是最新，無需抓取
    """
    _today = today or date.today().strftime("%Y-%m-%d")
    effective_start = max(global_start, dataset_earliest)

    if force:
        return effective_start

    latest = latest_dates.get(str(stock_id))
    if latest is None:
        return effective_start

    next_day = (
        datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if next_day > _today:
        return None  # 已是最新，跳過

    return max(next_day, dataset_earliest)


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
