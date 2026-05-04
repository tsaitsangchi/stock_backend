"""
core/db_utils.py — 統一的 PostgreSQL 工具函式 v3.0（逐支逐日 commit 完整性版）
==============================================================================
v3.0 重大改進（資料寫入完整性）：
  ★ safe_commit_rows()       逐筆/逐組 commit 主力函式，寫入後立即 commit、失敗 rollback。
  ★ commit_per_stock()       依 stock_id 分組逐支 commit，回傳 { stock_id: rows_committed }。
  ★ commit_per_day()         依日期分組逐日 commit，回傳 { date: rows_committed }。
  ★ commit_per_stock_per_day() 雙層粒度（每支股票×每日）皆獨立 commit，最細粒度的完整性。
  ★ commit_per_group()       通用分組 commit：呼叫端自訂 group_key_fn。
  ★ safe_bulk_upsert()       bulk_upsert 的 transaction-safe 版本，失敗自動 rollback。
  ★ with_savepoint()         savepoint context manager，巢狀交易安全。
  ★ FailureLogger            統一的失敗清單 append 寫檔器（即時落盤，崩潰不丟失）。
  ★ async_safe_commit_rows() asyncpg 版逐支 commit（每呼叫一次自動短交易）。
  ★ async_commit_per_stock_per_day() asyncpg 版雙層粒度。
  ★ get_failure_log_path()   失敗清單檔名統一規則（outputs/{table}_failed_YYYYMMDD.json）。
  ★ is_conn_healthy()        檢測連線是否處於可用狀態，失敗交易自動 rollback。
  ★ bulk_upsert / 所有讀取函式 補上 rollback 容錯，避免毒交易擴散。

v2.0 既有：
  - asyncpg 非同步驅動（get_asyncpg_pool / async_bulk_upsert / async_bulk_copy_upsert）
  - psycopg2 同步介面（get_db_conn / ensure_ddl / bulk_upsert）

向後相容：所有 v2.0 介面完全不變，舊 fetcher 不需修改即可使用。
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# ── 自我修復 sys.path（讓本檔可從任何位置直接執行 / import）──
_THIS_DIR = Path(__file__).resolve().parent      # scripts/core
_SCRIPTS_DIR = _THIS_DIR.parent                  # scripts
for _p in (_SCRIPTS_DIR, _THIS_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

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
    """執行一或多個 DDL 語句（冪等，IF NOT EXISTS）。失敗時 rollback。"""
    try:
        with conn.cursor() as cur:
            cur.execute(DDL_FETCH_LOG)
            for ddl in ddls:
                cur.execute(ddl)
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


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
    """將抓取結果記錄至 fetch_log（含 rollback 容錯）。"""
    sql = """
    INSERT INTO fetch_log (table_name, stock_id, start_date, end_date, rows_count, status, error_msg)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (table_name, stock_id, start_date, end_date, rows_count, status, error_msg))
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.debug(f"fetch_log 寫入失敗（不影響主流程）：{e}")


# ─────────────────────────────────────────────
# 同步批次 Upsert（psycopg2）— v2.0 介面（強化 rollback）
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
    成功時 commit；失敗時 rollback（v3.0 新增 rollback 容錯）。

    註：呼叫端若希望「失敗時也回傳 0 不拋例外」，請改用 safe_bulk_upsert()。
    若希望逐支 / 逐日 / 雙層粒度 commit，請用 commit_per_stock / commit_per_day /
    commit_per_stock_per_day。
    """
    if not rows:
        return 0
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, sql, rows, template=template, page_size=page_size
            )
        conn.commit()
        return len(rows)
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


# ─────────────────────────────────────────────
# 逐筆 / 逐組 commit 主力函式（v3.0 新增）
# ─────────────────────────────────────────────
def safe_bulk_upsert(
    conn,
    sql: str,
    rows: list,
    template: str,
    page_size: int = 2000,
) -> int:
    """
    bulk_upsert 的「失敗回 0」版本（不拋例外）。
    成功 → 寫入並 commit，回傳寫入筆數。
    失敗 → 自動 rollback，回傳 0，不拋例外。
    """
    if not rows:
        return 0
    try:
        return bulk_upsert(conn, sql, rows, template, page_size)
    except Exception as e:
        logger.error(f"safe_bulk_upsert 失敗（已 rollback）：{e}")
        return 0


def safe_commit_rows(
    conn,
    sql: str,
    rows: list,
    template: str,
    label: str = "",
    page_size: int = 2000,
) -> int:
    """
    ⭐ 逐組 commit 主力函式（推薦給所有 fetcher 使用） ⭐

    寫入單支 / 單日 / 單組資料後立即 commit；失敗時 rollback，回傳 0 不拋例外。

    保證：
      1. 每組成功寫入後立即落地（崩潰前的資料不會回滾）。
      2. 單組失敗不影響後續呼叫（rollback 後 conn 可繼續使用）。
      3. label 會印在錯誤日誌中，方便定位（建議格式 "{table}/{stock}" 或
         "{table}/{stock}/{date}"）。

    Parameters
    ----------
    conn      : psycopg2 連線
    sql       : INSERT ... ON CONFLICT ... 語句
    rows      : tuple list
    template  : 欄位佔位符（如 "(%s::date, %s, %s::numeric)"）
    label     : 錯誤日誌標籤
    page_size : 每批寫入筆數（預設 2000）

    Returns
    -------
    int  實際寫入筆數（失敗回 0）
    """
    if not rows:
        return 0
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, sql, rows, template=template, page_size=page_size
            )
        conn.commit()
        return len(rows)
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        prefix = f"[{label}] " if label else ""
        logger.error(f"  {prefix}寫入失敗，已 rollback：{e}")
        return 0


def commit_per_group(
    conn,
    sql: str,
    rows: list,
    template: str,
    group_key_fn: Callable[[tuple], Any],
    label_prefix: str = "",
    failure_logger: "FailureLogger | None" = None,
    page_size: int = 2000,
) -> dict:
    """
    ⭐ 通用「逐組 commit」函式 ⭐

    依 group_key_fn(row) 將 rows 分組，每組獨立寫入 + commit；單組失敗不影響其他組。

    Parameters
    ----------
    conn         : psycopg2 連線
    sql          : INSERT ... ON CONFLICT ... 語句
    rows         : tuple list
    template     : 欄位佔位符
    group_key_fn : row → group key 的函式（如 lambda r: r[1] 取第 2 欄當 stock_id）
    label_prefix : 錯誤標籤前綴（如 "month_revenue"）
    failure_logger: 失敗時自動 record（可選）
    page_size    : 每批寫入筆數

    Returns
    -------
    dict  { group_key: rows_committed }（失敗組會是 0）
    """
    if not rows:
        return {}

    groups: dict[Any, list] = defaultdict(list)
    for r in rows:
        try:
            groups[group_key_fn(r)].append(r)
        except Exception as e:
            logger.warning(f"  [{label_prefix}] group_key_fn 異常筆，歸入 _UNKNOWN：{e}")
            groups["_UNKNOWN"].append(r)

    results: dict[Any, int] = {}
    for key, sub_rows in groups.items():
        sub_label = f"{label_prefix}/{key}" if label_prefix else str(key)
        n = safe_commit_rows(conn, sql, sub_rows, template, label=sub_label, page_size=page_size)
        results[key] = n
        if n == 0 and sub_rows and failure_logger is not None:
            failure_logger.record(
                stock_id=str(key),
                rows=len(sub_rows),
                error="bulk_upsert returned 0 (rolled back)",
            )
    return results


def commit_per_stock(
    conn,
    sql: str,
    rows: list,
    template: str,
    stock_index: int = 1,
    label_prefix: str = "",
    failure_logger: "FailureLogger | None" = None,
    page_size: int = 2000,
) -> dict:
    """
    ⭐ 依 stock_id 分組「逐支 commit」 ⭐

    rows 中每筆的第 stock_index 欄為 stock_id（一般 PK 排序為 (date, stock_id, ...)，
    所以預設 stock_index=1）。

    Returns
    -------
    dict  { stock_id: rows_committed }
    """
    return commit_per_group(
        conn, sql, rows, template,
        group_key_fn=lambda r: r[stock_index],
        label_prefix=label_prefix,
        failure_logger=failure_logger,
        page_size=page_size,
    )


def commit_per_day(
    conn,
    sql: str,
    rows: list,
    template: str,
    date_index: int = 0,
    label_prefix: str = "",
    failure_logger: "FailureLogger | None" = None,
    page_size: int = 2000,
) -> dict:
    """
    ⭐ 依日期分組「逐日 commit」 ⭐

    rows 中每筆的第 date_index 欄為日期（PK 第一欄，預設 date_index=0）。
    適用於市場層 / 衍生品 / 全市場 dump 等以日期為主軸的資料。

    Returns
    -------
    dict  { date_str: rows_committed }
    """
    def _key(r):
        v = r[date_index]
        if isinstance(v, (date, datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v)

    return commit_per_group(
        conn, sql, rows, template,
        group_key_fn=_key,
        label_prefix=label_prefix,
        failure_logger=failure_logger,
        page_size=page_size,
    )


def commit_per_stock_per_day(
    conn,
    sql: str,
    rows: list,
    template: str,
    date_index: int = 0,
    stock_index: int = 1,
    label_prefix: str = "",
    failure_logger: "FailureLogger | None" = None,
    page_size: int = 2000,
) -> dict:
    """
    ⭐⭐ 「逐支 × 逐日」雙層粒度 commit（最細粒度的完整性） ⭐⭐

    把 rows 依 (stock_id, date) 分組，每對 (sid, day) 獨立寫入 + commit。
    單對失敗不影響其他對，崩潰時已寫入的 (sid, day) 對都已落地。

    使用時機：
      · 對「資料完整性」要求極高（如金融交易訊號、公告法定資料）。
      · 單支股票一天的資料量不大（每組通常 1~50 筆，commit 開銷可接受）。

    Parameters
    ----------
    conn        : psycopg2 連線
    sql / rows / template : 同 bulk_upsert
    date_index  : rows 中日期欄位的 index（預設 0）
    stock_index : rows 中 stock_id 欄位的 index（預設 1）
    label_prefix: 錯誤標籤前綴
    failure_logger: 失敗時自動 record
    page_size   : 每組批次大小

    Returns
    -------
    dict  { (stock_id, date): rows_committed }
    """
    def _key(r):
        v = r[date_index]
        if isinstance(v, (date, datetime)):
            d_str = v.strftime("%Y-%m-%d")
        else:
            d_str = str(v)
        return (r[stock_index], d_str)

    if not rows:
        return {}

    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        try:
            groups[_key(r)].append(r)
        except Exception as e:
            logger.warning(f"  [{label_prefix}] key 計算異常筆，跳過：{e}")

    results: dict[tuple, int] = {}
    for (sid, dstr), sub_rows in groups.items():
        sub_label = f"{label_prefix}/{sid}/{dstr}" if label_prefix else f"{sid}/{dstr}"
        n = safe_commit_rows(conn, sql, sub_rows, template, label=sub_label, page_size=page_size)
        results[(sid, dstr)] = n
        if n == 0 and sub_rows and failure_logger is not None:
            failure_logger.record(
                stock_id=str(sid), date=dstr,
                rows=len(sub_rows),
                error="bulk_upsert returned 0 (rolled back)",
            )
    return results


# ─────────────────────────────────────────────
# Savepoint context manager（巢狀交易支援）
# ─────────────────────────────────────────────
@contextmanager
def with_savepoint(conn, name: str = "sp"):
    """
    建立 savepoint 進入巢狀交易。例外時 rollback 至 savepoint。

    用法：
        with conn.cursor() as cur:
            cur.execute("INSERT INTO ... VALUES (1)")
            with with_savepoint(conn, "sp_inner"):
                cur.execute("INSERT INTO ... VALUES (2)")  # 此處失敗只會回到 SAVEPOINT
        conn.commit()  # 外層仍可提交（INSERT 1 保留）
    """
    safe_name = "".join(c for c in name if c.isalnum() or c == "_") or "sp"
    with conn.cursor() as cur:
        cur.execute(f"SAVEPOINT {safe_name}")
    try:
        yield
        with conn.cursor() as cur:
            cur.execute(f"RELEASE SAVEPOINT {safe_name}")
    except Exception:
        try:
            with conn.cursor() as cur:
                cur.execute(f"ROLLBACK TO SAVEPOINT {safe_name}")
                cur.execute(f"RELEASE SAVEPOINT {safe_name}")
        except Exception:
            pass
        raise


# ─────────────────────────────────────────────
# 失敗清單即時 append（v3.0 新增）
# ─────────────────────────────────────────────
def get_failure_log_path(table: str, base_dir: str | Path | None = None) -> Path:
    """
    取得「失敗清單」檔案的標準路徑。
    格式：{base_dir or scripts/outputs}/{table}_failed_YYYYMMDD.json
    """
    if base_dir is None:
        base = Path(__file__).resolve().parent.parent / "outputs"
    else:
        base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{table}_failed_{date.today().strftime('%Y%m%d')}.json"


_failure_lock = threading.Lock()


def append_failure_json(path: str | Path, item: dict, max_items: int = 50000) -> None:
    """
    將單筆失敗記錄即時 append 至 JSON 檔（atomic write）。

    特性：
      · 檔案不存在會自動建立。
      · 寫入採 tmp + rename 原子寫入，崩潰時不會留下半份檔案。
      · 同一程序中以 lock 防止多執行緒競爭寫入。
      · max_items 上限，超過時 trim 最舊的記錄。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with _failure_lock:
        existing: list = []
        if p.exists():
            try:
                existing = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []

        existing.append(item)
        if len(existing) > max_items:
            existing = existing[-max_items:]

        try:
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp.replace(p)
        except Exception as e:
            logger.warning(f"append_failure_json 失敗：{e}")


class FailureLogger:
    """
    統一的失敗紀錄器：
      · append 至 JSON 失敗清單（每筆即時落盤）。
      · 同步寫入 fetch_log 資料表（可選）。
      · 控制台 logger.error。

    用法：
        flog = FailureLogger("month_revenue")
        for sid in stock_ids:
            try:
                ...
            except Exception as e:
                flog.record(stock_id=sid, error=str(e))
        flog.summary()
    """

    def __init__(
        self,
        table: str,
        base_dir: str | Path | None = None,
        log_to_db: bool = False,
        db_conn=None,
    ):
        self.table = table
        self.path = get_failure_log_path(table, base_dir)
        self.log_to_db = log_to_db
        self.db_conn = db_conn
        self.failures: list[dict] = []

    def record(self, **kwargs) -> None:
        """記錄一筆失敗。kwargs 至少應含 stock_id 與 error。"""
        kwargs.setdefault("timestamp", datetime.now().isoformat())
        kwargs.setdefault("table", self.table)
        self.failures.append(kwargs)
        append_failure_json(self.path, kwargs)

        prefix = f"[{self.table}"
        sid = kwargs.get("stock_id")
        if sid:
            prefix += f"/{sid}"
        d = kwargs.get("date")
        if d:
            prefix += f"/{d}"
        prefix += "]"
        logger.error(f"  {prefix} 失敗：{kwargs.get('error', '')}")

        if self.log_to_db and self.db_conn is not None:
            try:
                log_fetch_result(
                    self.db_conn,
                    table_name=self.table,
                    stock_id=str(sid or "MARKET"),
                    start_date=kwargs.get("start_date") or date.today().strftime("%Y-%m-%d"),
                    end_date=kwargs.get("end_date") or date.today().strftime("%Y-%m-%d"),
                    rows_count=0,
                    status="FAILED",
                    error_msg=kwargs.get("error"),
                )
            except Exception as e:
                logger.debug(f"FailureLogger 寫 fetch_log 失敗（忽略）：{e}")

    def summary(self) -> None:
        if self.failures:
            logger.info(
                f"  [{self.table}] 失敗清單已寫入：{self.path}（{len(self.failures)} 筆）"
            )

    def __len__(self) -> int:
        return len(self.failures)


# ─────────────────────────────────────────────
# 非同步批次 Upsert（asyncpg）— v3.0 強化失敗處理
# ─────────────────────────────────────────────
async def async_bulk_upsert(
    pool,
    sql: str,
    rows: list[tuple],
) -> int:
    """asyncpg 非同步批次 UPSERT（每呼叫一次自動 transaction + commit）。"""
    if not rows:
        return 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(sql, rows)
    return len(rows)


async def async_safe_commit_rows(
    pool,
    sql: str,
    rows: list[tuple],
    label: str = "",
) -> int:
    """
    ⭐ asyncpg 版的逐組 commit ⭐
    每呼叫一次自動建立短交易 + commit；失敗時 transaction 自動 rollback，回傳 0。
    """
    if not rows:
        return 0
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(sql, rows)
        return len(rows)
    except Exception as e:
        prefix = f"[{label}] " if label else ""
        logger.error(f"  {prefix}async 寫入失敗，已 rollback：{e}")
        return 0


async def async_commit_per_stock_per_day(
    pool,
    sql: str,
    rows: list[tuple],
    date_index: int = 0,
    stock_index: int = 1,
    label_prefix: str = "",
    concurrency: int = 8,
) -> dict:
    """
    ⭐⭐ asyncpg 版「逐支 × 逐日」雙層粒度 commit ⭐⭐

    每對 (stock_id, date) 在 asyncpg 連線池中以獨立短交易並行寫入。

    Parameters
    ----------
    concurrency : 同時並行的 (sid, day) 數（預設 8，避免連線池耗盡）

    Returns
    -------
    dict  { (stock_id, date): rows_committed }
    """
    import asyncio

    if not rows:
        return {}

    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        v = r[date_index]
        d_str = v.strftime("%Y-%m-%d") if isinstance(v, (date, datetime)) else str(v)
        groups[(r[stock_index], d_str)].append(r)

    sem = asyncio.Semaphore(concurrency)
    results: dict[tuple, int] = {}

    async def _commit_one(key, sub_rows):
        sid, dstr = key
        async with sem:
            sub_label = f"{label_prefix}/{sid}/{dstr}" if label_prefix else f"{sid}/{dstr}"
            n = await async_safe_commit_rows(pool, sql, sub_rows, label=sub_label)
            results[key] = n

    await asyncio.gather(*[_commit_one(k, v) for k, v in groups.items()])
    return results


async def async_bulk_copy_upsert(
    pool,
    staging_table: str,
    target_table: str,
    columns: list[str],
    rows: list[tuple],
    conflict_columns: list[str],
    update_columns: list[str],
) -> int:
    """asyncpg 二進位暫存表高速複製 + UPSERT（適用大型資料集）。"""
    if not rows:
        return 0

    cols_csv = ", ".join(columns)
    conflict_csv = ", ".join(conflict_columns)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in update_columns
    )

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                f"""
                CREATE TEMP TABLE {staging_table}
                (LIKE {target_table} INCLUDING DEFAULTS)
                ON COMMIT DROP
                """
            )
            await conn.copy_records_to_table(
                staging_table,
                records=rows,
                columns=columns,
            )
            result = await conn.execute(
                f"""
                INSERT INTO {target_table} ({cols_csv})
                SELECT {cols_csv} FROM {staging_table}
                ON CONFLICT ({conflict_csv}) DO UPDATE SET {update_set}
                """
            )
            try:
                count = int(result.split()[-1])
            except (IndexError, ValueError):
                count = len(rows)

    return count


# ─────────────────────────────────────────────
# 型別轉換工具
# ─────────────────────────────────────────────
def safe_float(val) -> float | None:
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
    f = safe_float(val)
    return int(f) if f is not None else None


def safe_bigint(val) -> int | None:
    return safe_int(val)


def safe_date(val) -> str | None:
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


def safe_mapper(mapper, raw: dict, label: str = "") -> Optional[tuple]:
    """包裝 mapper(raw)，異常筆回傳 None 並記錄 warning。"""
    try:
        return mapper(raw)
    except Exception as e:
        prefix = f"[{label}] " if label else ""
        logger.warning(f"  {prefix}mapper 異常筆，跳過：{e}")
        return None


def map_rows_safe(mapper, data: Iterable[dict], label: str = "") -> list[tuple]:
    """批次 safe_mapper：自動過濾掉 mapper 異常的筆。"""
    out: list[tuple] = []
    for r in data:
        m = safe_mapper(mapper, r, label)
        if m is not None:
            out.append(m)
    return out


# ─────────────────────────────────────────────
# 增量更新輔助（皆補上 rollback 容錯）
# ─────────────────────────────────────────────
def get_all_latest_dates(conn, table: str, key_col: str = "stock_id") -> dict[str, str]:
    """查出所有標的的最新日期。回傳 { id: "YYYY-MM-DD" }"""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {key_col}, MAX(date) FROM {table} GROUP BY {key_col}")
            return {
                row[0]: row[1].strftime("%Y-%m-%d")
                for row in cur.fetchall()
                if row[1] is not None
            }
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def get_all_safe_starts(
    conn, table: str, window_days: int = 60, gap_interval: str = "1 day", key_col: str = "stock_id"
) -> dict[str, str]:
    """智能偵測起始點，支援不同頻率。"""
    dow_filter = (
        "AND extract(dow from t1.date + interval '1 day') NOT IN (0, 6)"
        if gap_interval == "1 day" else ""
    )

    sql = f"""
    WITH gaps AS (
        SELECT
            t1.{key_col},
            MIN(t1.date + interval '{gap_interval}') as gap_start
        FROM {table} t1
        WHERE t1.date >= CURRENT_DATE - interval '{window_days} days'
          AND t1.date < CURRENT_DATE
          AND NOT EXISTS (
              SELECT 1 FROM {table} t2
              WHERE t2.{key_col} = t1.{key_col}
                AND t2.date = t1.date + interval '{gap_interval}'
          )
          {dow_filter}
        GROUP BY t1.{key_col}
    ),
    max_dates AS (
        SELECT {key_col}, MAX(date) as last_date FROM {table} GROUP BY {key_col}
    )
    SELECT
        m.{key_col},
        COALESCE(g.gap_start, m.last_date + interval '{gap_interval}') as safe_start
    FROM max_dates m
    LEFT JOIN gaps g ON m.{key_col} = g.{key_col}
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return {
                row[0]: row[1].strftime("%Y-%m-%d")
                for row in cur.fetchall()
                if row[1] is not None
            }
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def get_latest_date(conn, table: str, stock_id: str) -> str | None:
    """查詢單支股票的最新日期。"""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(date) FROM {table} WHERE stock_id = %s", (stock_id,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0].strftime("%Y-%m-%d")
        return None
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def get_latest_date_by_col(conn, table: str, id_col: str, data_id: str) -> str | None:
    """查詢非 stock_id 主鍵欄位的最新日期。"""
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT MAX(date) FROM {table} WHERE {id_col} = %s",
                (data_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0].strftime("%Y-%m-%d")
        return None
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def get_market_safe_start(conn, table: str, window_days: int = 60) -> str | None:
    """市場層資料表的「安全起始日」。"""
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
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if row and row[0]:
                return row[0].strftime("%Y-%m-%d")
        return None
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def get_safe_start(
    conn, table: str, stock_id: str, window_days: int = 60, gap_interval: str = "1 day"
) -> str | None:
    """單支股票的「安全起始日」。"""
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
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (stock_id, stock_id))
            row = cur.fetchone()
            if row and row[0]:
                return row[0].strftime("%Y-%m-%d")
        return None
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def resolve_start_cached(
    stock_id: str,
    latest_dates: dict,
    global_start: str,
    dataset_earliest: str,
    force: bool,
    today: str | None = None,
) -> str | None:
    """從預載快取（latest_dates）決定起始日。"""
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
    """直接查 DB 決定起始日。"""
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
    """依指定欄位索引去除重複列。"""
    seen = {}
    for row in rows:
        key = tuple(row[i] for i in key_indices)
        seen[key] = row
    return list(seen.values())


def get_db_stock_ids(conn, types: tuple = ("twse", "otc")) -> list[str]:
    """從 stock_info 取得所有指定類型的股票代號清單。"""
    placeholders = ", ".join(["%s"] * len(types))
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT stock_id FROM stock_info WHERE type IN ({placeholders}) ORDER BY stock_id",
                types,
            )
            return [row[0] for row in cur.fetchall()]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


# ─────────────────────────────────────────────
# 健康檢查（v3.0 新增）
# ─────────────────────────────────────────────
def is_conn_healthy(conn) -> bool:
    """
    檢查連線是否處於可用狀態（非 InFailedSqlTransaction）。
    若不健康會嘗試自動 rollback 修復。
    """
    if conn is None or conn.closed:
        return False
    try:
        status = conn.get_transaction_status()
    except Exception:
        return False
    if status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
        try:
            conn.rollback()
            logger.info("檢測到失敗交易，已自動 rollback")
            return True
        except Exception:
            return False
    return True


__all__ = [
    # 連線
    "get_db_conn", "get_asyncpg_pool", "close_asyncpg_pool",
    # DDL / log
    "ensure_ddl", "log_fetch_result", "DDL_FETCH_LOG",
    # 同步寫入
    "bulk_upsert", "safe_bulk_upsert", "safe_commit_rows",
    "commit_per_group", "commit_per_stock", "commit_per_day",
    "commit_per_stock_per_day",
    "with_savepoint", "is_conn_healthy",
    # 失敗清單
    "FailureLogger", "append_failure_json", "get_failure_log_path",
    # 非同步寫入
    "async_bulk_upsert", "async_safe_commit_rows",
    "async_commit_per_stock_per_day", "async_bulk_copy_upsert",
    # 型別轉換
    "safe_float", "safe_int", "safe_bigint", "safe_date", "safe_timestamp",
    "safe_mapper", "map_rows_safe",
    # 增量更新
    "get_all_latest_dates", "get_all_safe_starts",
    "get_latest_date", "get_latest_date_by_col",
    "get_market_safe_start", "get_safe_start",
    "resolve_start_cached", "resolve_start",
    # 其他
    "dedup_rows", "get_db_stock_ids",
]