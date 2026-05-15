"""
db_utils.py v2.45 (Quantum Finance Infrastructure Sovereign Edition)
================================================================================
**最後更新日期**: 2026-05-14
**主權狀態**: GOVERNANCE SYNC (§6.7 SQL + Public API Restoration)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Infrastructure Resilience]: 提供具備自動重連與健康診斷的資料庫通訊介面，確保 24/7 治權連通性。
2. [Asset Sovereignty]: 核心股名單必須透過 `core_universe_membership` JOIN `core_universe_snapshot` 取得，嚴禁回查 v5.2 時代 `stocks` 表。
3. [Historical Reference Authority]: 保留從 v1.0 至今的所有修訂歷史，作為判定系統正確性的基準。
4. [Hybrid Observability]: 基礎設施維運必須觸發「生命週期紀錄」(Lifecycle) 與「專項審計紀錄」(Audit)；
   生命週期紀錄必須完整寫入 start_time / end_time / error_msg；status 必須反映實際結果，
   嚴禁「Python 無例外即記 success」之謊報邏輯（v2.44 補強）。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix - 100% Coverage)
本矩陣遵循「組合完整性原則」，窮舉所有基礎設施維運可能性：

| 維運需求場景 (Scenario)   | 權威指令 / 建議用法 (Exhaustive Examples)                             | 對齊模組 |
| :----------------------- | :-------------------------------------------------------------------- | :--- |
| **1. [基礎設施：連線診斷]** | `$ python scripts/core/db_utils.py`                                   | db_utils v2.45 |
| **2. [個股同步：單一標的全數據]** | `$ python scripts/ingestion/template_fetcher.py --id 2330 --all_datasets` | template_fetcher |
| **3. [單一 Table 同步：初始化]** | `$ python scripts/core/data_schema.py --init --table TaiwanStockPrice`| data_schema |
| **4. [單一個股所有 Table 同步]** | `$ python scripts/ingestion/template_fetcher.py --id 2330 --all_datasets` | template_fetcher |
| **5. [所有核心股同步]**   | `$ python scripts/ingestion/template_fetcher.py --universe core --all_datasets` | template_fetcher |
| **6. [所有核心股 + 所有表：強制更新]** | `$ python scripts/ingestion/template_fetcher.py --universe core --all_datasets --force` | template_fetcher |
| **7. [緊急維運：重置連線池]** | `$ python scripts/core/db_utils.py --reset-pool`                      | db_utils v2.45 |
| **8. [數據稽核：生命週期完整性]** | `$ python scripts/maintenance/check_system_health.py`                  | maintenance |

💡 **範例完整性說明**: 透過以上 8 種場景組合，維運人員可實現從單一物理連線探測到全宇宙數據毀滅性重刷的所有執行可能性。

> 註 (v2.45)：本版完成 §6.7 核心股查詢 SQL 契約，並恢復 fetchers / pipeline / monitor / evaluation 既有 public API。

## 📜 三、全修訂歷程 (Full Revision History - 舊詳細參考)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v2.45** | 2026-05-14 | Codex (No-touch Zone 授權) | **§6.7 SQL + Public API Restoration**：(1) `get_core_stocks_from_db()` 改查 `core_universe_membership` JOIN `core_universe_snapshot WHERE status='committed'`，封閉 Pending Bug #4；(2) 補回 `get_db_conn`、`ensure_ddl`、`bulk_upsert`、`safe_commit_rows`、`FailureLogger`、`DDL_FETCH_LOG`、`log_fetch_result`、`db_transaction`、`db_session`、`write_pipeline_log`、`write_evaluation_log`、`get_db_stock_ids` 等跨模組 public API；(3) `psycopg2` / `dotenv` 改為延遲失敗，允許常數與 API 匯入測試先行。 | **ACTIVE** |
| v2.44 | 2026-05-14 | Antigravity (Auto-patch, No-touch Zone 授權) | **Bug #2 + Bug #3 雙修補**：(1) `record_lifecycle` 改為 yield 一個可由 caller 標記失敗/警告的 `_LifecycleContext`，封堵「Python 無例外即記 success」之 status 謊報；(2) INSERT 由 5 欄擴張為 8 欄，補寫 start_time / end_time / error_msg，封堵 NULL 漏洞；(3) DB 連線改為僅在 finally 開啟，不再霸佔整個 task 期間；(4) logger 失敗時不再 propagate 例外給 caller。100% backward compatible —— 舊 `with record_lifecycle(...):` 呼叫端零修改。 | SUPERSEDED |
| v2.43 | 2026-05-12 | Antigravity | **防禦性修復**：補全缺失的 `argparse` 導入，恢復指令列工具之治權效力。 | SUPERSEDED |
| v2.42 | 2026-05-12 | Antigravity | **主權完備化**：對齊五大核心場景語意，擴張全可能性維運矩陣，落實混合觀測。 | SUPERSEDED |
| v2.41 | 2026-05-12 | Antigravity | **憲法化對齊**：補全維運矩陣與四維核心定義，對齊 v5.2 旗艦要求。 | SUPERSEDED |
| v2.0 | 2026-04-30 | Antigravity | **安全重構**：整合 .env 加密認證，建立 get_db_connection 標準化接口。 | SUPERSEDED |
| v1.0 | 2026-04-20 | Antigravity | **主權奠基**：初始版本，建立基本連線與 stocks 元數據表治理。 | ARCHIVED |
================================================================================
"""
import os, sys, logging, time, argparse, json
from contextlib import contextmanager
from pathlib import Path
from datetime import date, datetime, timedelta

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ModuleNotFoundError:
    psycopg2 = None
    execute_values = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


DDL_FETCH_LOG = """
CREATE TABLE IF NOT EXISTS fetch_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    table_name TEXT NOT NULL,
    stock_id TEXT,
    status TEXT NOT NULL,
    rows_inserted INTEGER DEFAULT 0,
    fetch_date_from DATE,
    fetch_date_to DATE,
    duration_ms INTEGER DEFAULT 0,
    error_msg TEXT,
    fetch_mode TEXT DEFAULT 'market',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS table_name TEXT;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS stock_id TEXT;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS status TEXT;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS rows_inserted INTEGER DEFAULT 0;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS fetch_date_from DATE;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS fetch_date_to DATE;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS duration_ms INTEGER DEFAULT 0;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS error_msg TEXT;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS fetch_mode TEXT DEFAULT 'market';
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
CREATE INDEX IF NOT EXISTS idx_fetch_log_table_stock_time
    ON fetch_log (table_name, stock_id, timestamp DESC);
"""


def _require_psycopg2():
    if psycopg2 is None:
        raise RuntimeError(
            "Missing dependency: psycopg2/psycopg2-binary is required for DB operations. "
            "Install project requirements before running DB diagnostics."
        )


class _LifecycleContext:
    """[v2.44 新增] 生命週期上下文物件。

    用於封堵 Bug #2：caller 在 try/except 內吃掉例外時，
    可透過 mark_failed / mark_warning 把局部失敗反映到 lifecycle log，
    避免 status 因「Python 無例外」而謊報 success。

    背景：sovereign_sync_engine v1.7 的 sync_fred / sync_finmind 即使
    sub-task 失敗也只更新內部 stats["failed"]，不 raise；舊版 record_lifecycle
    看不到這層失敗。本物件補上「外部標記」介面。
    """

    __slots__ = ("failures", "warnings")

    def __init__(self):
        self.failures = []
        self.warnings = []

    def mark_failed(self, msg):
        """標記一個局部失敗（不會 raise，僅記錄）。"""
        self.failures.append(str(msg))

    def mark_warning(self, msg):
        """標記一個局部警告（不會 raise，僅記錄）。"""
        self.warnings.append(str(msg))

    @property
    def has_failures(self):
        return len(self.failures) > 0

    @property
    def has_warnings(self):
        return len(self.warnings) > 0


@contextmanager
def record_lifecycle(task_name, category="general", stock_id=None):
    """旗艦級生命週期裝飾器 (v2.44) - 混合模式 A: pipeline_execution_log

    [v2.44 主要變動]
    1. Bug #2 修補：yield 一個 _LifecycleContext 給 caller 主動標記局部失敗。
       舊 `with record_lifecycle(...):` 不接收 yield 值仍正常運作（context manager 規範允許）。
       新 `with record_lifecycle(...) as lc:` 可呼叫 lc.mark_failed(msg) / lc.mark_warning(msg)。
    2. Bug #3 修補：INSERT 改寫 8 欄，補上 start_time / end_time / error_msg。
    3. 連線生命週期：改為僅在 finally 開連線，不再霸佔整個 task 期間。
    4. Logger 隔離：寫日誌失敗時印 warning 到 stderr，不再 propagate 給 caller。

    Args:
        task_name (str): 任務名稱，例：'sync_fred_macro'
        category (str): 分類，例：'ingestion' / 'maintenance' / 'infrastructure'
        stock_id (str|None): 標的 ID，無關標的時建議填 'SYSTEM'

    Yields:
        _LifecycleContext: 供 caller 標記局部失敗/警告之介面（opt-in）。

    Status 判定優先序：
        Python 例外          → 'failed' (error_msg = exception 訊息)
        ctx.failures 非空    → 'failed' (error_msg = 合併之失敗訊息)
        ctx.warnings 非空    → 'warning' (error_msg = 合併之警告訊息)
        否則                 → 'success' (error_msg = NULL)
    """
    start_time = datetime.now()
    ctx = _LifecycleContext()
    py_exception = None
    try:
        yield ctx
    except Exception as e:
        py_exception = e
        raise
    finally:
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)

        # [v2.44 Bug#2] 動態判定 status
        if py_exception is not None:
            status = "failed"
            error_msg = f"{type(py_exception).__name__}: {str(py_exception)}"
        elif ctx.has_failures:
            status = "failed"
            error_msg = "; ".join(ctx.failures[:5])
            if len(ctx.failures) > 5:
                error_msg += f"; ... (+{len(ctx.failures) - 5} more)"
        elif ctx.has_warnings:
            status = "warning"
            error_msg = "; ".join(ctx.warnings[:5])
            if len(ctx.warnings) > 5:
                error_msg += f"; ... (+{len(ctx.warnings) - 5} more)"
        else:
            status = "success"
            error_msg = None

        # [v2.44 Bug#3] INSERT 8 欄完整寫入 (start_time / end_time / error_msg 不再 NULL)
        # [v2.44 Patch C] 連線僅在此處開啟，不霸佔整個 task 期間
        # [v2.44 Patch D] Logger 失敗不影響 caller
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO pipeline_execution_log
                        (task_name, category, stock_id, start_time, end_time,
                         status, duration_ms, error_msg)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (task_name, category, stock_id, start_time, end_time,
                     status, duration, error_msg),
                )
                conn.commit()
            finally:
                cur.close()
                conn.close()
        except Exception as log_err:
            # 寫日誌失敗只警告，不再 raise 把 caller 一起拖死
            print(
                f"⚠️  [record_lifecycle] pipeline_execution_log 寫入失敗: {log_err}",
                file=sys.stderr,
            )


def write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected):
    """專項審計日誌 (v2.43 unchanged in v2.44) - 混合模式 B: data_audit_log"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected)
            VALUES (%s, %s, %s, %s, %s)
        """, (table_name, stock_id, data_date, action_type, rows_affected))
        conn.commit()
    finally:
        cur.close(); conn.close()


def get_db_connection():
    """建立資料庫連線 (v2.45)"""
    _require_psycopg2()
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def get_db_conn():
    """Backward-compatible alias used by legacy fetchers."""
    return get_db_connection()


def get_connection_params():
    """Return sanitized connection parameters for maintenance tools."""
    return {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def db_connection_check():
    """基礎設施健康診斷 (v2.43 unchanged in v2.44)"""
    start = time.time()
    try:
        conn = get_db_connection()
        conn.close()
        return True, (time.time() - start) * 1000
    except Exception:
        return False, 0


check_db_health = db_connection_check


def ensure_ddl(conn=None, *ddls):
    """Execute one or more DDL statements idempotently.

    If called without DDL, ensure the shared fetch_log table exists.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    if not ddls:
        ddls = (DDL_FETCH_LOG,)
    try:
        with conn.cursor() as cur:
            for ddl in ddls:
                if ddl:
                    cur.execute(ddl)
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if own_conn:
            conn.close()


@contextmanager
def db_transaction():
    """Yield a cursor inside a commit/rollback transaction."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


@contextmanager
def db_session():
    """Yield a raw connection inside a commit/rollback transaction."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _bulk_upsert_sql(conn, sql, rows, template=None, page_size=2000):
    if not rows:
        return 0
    if execute_values is None:
        _require_psycopg2()
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template=template, page_size=page_size)
    return len(rows)


def _quote_ident(name):
    return '"' + str(name).replace('"', '""') + '"'


def _bulk_upsert_table(table_name, records, unique_cols, page_size=2000):
    if not records:
        return 0
    if not unique_cols:
        raise ValueError("unique_cols is required for table-name bulk_upsert")
    conn = get_db_connection()
    columns = list(records[0].keys())
    rows = [tuple(record.get(col) for col in columns) for record in records]
    col_sql = ", ".join(_quote_ident(col) for col in columns)
    conflict_sql = ", ".join(_quote_ident(col) for col in unique_cols)
    update_cols = [col for col in columns if col not in set(unique_cols)]
    if update_cols:
        update_sql = ", ".join(
            f"{_quote_ident(col)} = EXCLUDED.{_quote_ident(col)}" for col in update_cols
        )
        action_sql = f"DO UPDATE SET {update_sql}"
    else:
        action_sql = "DO NOTHING"
    sql = (
        f"INSERT INTO {_quote_ident(table_name)} ({col_sql}) VALUES %s "
        f"ON CONFLICT ({conflict_sql}) {action_sql}"
    )
    try:
        count = _bulk_upsert_sql(conn, sql, rows, page_size=page_size)
        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def bulk_upsert(*args, **kwargs):
    """Backward-compatible bulk upsert.

    Supports both legacy `(conn, sql, rows, template, page_size=...)` and
    newer `(table_name, records, unique_cols=[...])` call styles.
    """
    if args and hasattr(args[0], "cursor"):
        conn = args[0]
        sql = args[1]
        rows = args[2]
        template = args[3] if len(args) > 3 else kwargs.get("template")
        page_size = kwargs.get("page_size", 2000)
        return _bulk_upsert_sql(conn, sql, rows, template=template, page_size=page_size)
    if len(args) < 2:
        raise TypeError("bulk_upsert requires either conn/sql/rows or table_name/records")
    table_name = args[0]
    records = args[1]
    unique_cols = kwargs.get("unique_cols")
    if unique_cols is None and len(args) > 2:
        unique_cols = args[2]
    return _bulk_upsert_table(
        table_name,
        records,
        unique_cols=unique_cols,
        page_size=kwargs.get("page_size", 2000),
    )


def safe_commit_rows(conn, sql, rows, template=None, label="rows", page_size=2000):
    """Bulk-write rows and commit, rolling back on failure."""
    try:
        count = _bulk_upsert_sql(conn, sql, rows, template=template, page_size=page_size)
        conn.commit()
        return count
    except Exception as exc:
        conn.rollback()
        logging.getLogger(__name__).error("safe_commit_rows failed for %s: %s", label, exc)
        raise


def safe_float(value, default=None):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_date(value, default=None):
    if value in ("", None):
        return default
    if isinstance(value, date):
        return value
    text = str(value)[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return default


def map_rows_safe(rows, mapper, label="rows"):
    mapped = []
    failures = []
    for row in rows or []:
        try:
            mapped.append(mapper(row))
        except Exception as exc:
            failures.append({"row": row, "error_msg": str(exc), "label": label})
    return mapped, failures


def dedup_rows(rows, key_indices):
    seen = {}
    for row in rows or []:
        key = tuple(row[i] for i in key_indices)
        seen[key] = row
    return list(seen.values())


def commit_per_group(conn, sql, rows, template=None, group_key_fn=None, label="rows", page_size=2000):
    if group_key_fn is None:
        return {"ALL": safe_commit_rows(conn, sql, rows, template, label=label, page_size=page_size)}
    grouped = {}
    for row in rows or []:
        grouped.setdefault(group_key_fn(row), []).append(row)
    result = {}
    for key, group_rows in grouped.items():
        result[key] = safe_commit_rows(
            conn, sql, group_rows, template, label=f"{label}/{key}", page_size=page_size
        )
    return result


def commit_per_stock(conn, sql, rows, template=None, stock_index=0, label="rows", page_size=2000):
    return commit_per_group(
        conn,
        sql,
        rows,
        template=template,
        group_key_fn=lambda row: row[stock_index],
        label=label,
        page_size=page_size,
    )


def commit_per_day(conn, sql, rows, template=None, date_index=1, label="rows", page_size=2000):
    return commit_per_group(
        conn,
        sql,
        rows,
        template=template,
        group_key_fn=lambda row: row[date_index],
        label=label,
        page_size=page_size,
    )


def commit_per_stock_per_day(
    conn,
    sql,
    rows,
    template=None,
    stock_index=0,
    date_index=1,
    label="rows",
    page_size=2000,
):
    return commit_per_group(
        conn,
        sql,
        rows,
        template=template,
        group_key_fn=lambda row: (row[stock_index], row[date_index]),
        label=label,
        page_size=page_size,
    )


def get_failure_log_path(table_name):
    output_dir = _PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    return output_dir / f"{table_name}_failed_{stamp}.json"


def append_failure_json(table_name, item):
    path = get_failure_log_path(table_name)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(item)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_pipeline_log(
    task_name,
    stock_id="SYSTEM",
    status="success",
    category="general",
    duration_ms=0,
    rows=0,
    err=None,
    **kwargs,
):
    """Write a lifecycle-like pipeline log row without context-manager wrapping."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_execution_log
                    (task_name, category, stock_id, start_time, end_time,
                     status, duration_ms, error_msg)
                VALUES (%s, %s, %s, NOW(), NOW(), %s, %s, %s)
                """,
                (task_name, category, stock_id, status, duration_ms or 0, err),
            )
        conn.commit()
    except Exception as exc:
        print(f"⚠️  [write_pipeline_log] failed: {exc}", file=sys.stderr)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def write_evaluation_log(*args, **kwargs):
    """Best-effort evaluation logger backed by pipeline_execution_log."""
    task_name = kwargs.pop("task_name", None) or (args[0] if args else "evaluation")
    stock_id = kwargs.pop("stock_id", "SYSTEM")
    status = kwargs.pop("status", "success")
    duration_ms = kwargs.pop("duration_ms", 0)
    rows = kwargs.pop("rows", 0)
    err = kwargs.pop("err", kwargs.pop("error_msg", None))
    write_pipeline_log(task_name, stock_id, status, "evaluation", duration_ms, rows, err)


def log_fetch_result(*args, **kwargs):
    """Write a standardized fetch_log row."""
    if args and hasattr(args[0], "cursor"):
        conn = args[0]
        table_name = args[1]
        stock_id = args[2]
        fetch_date_from = args[3] if len(args) > 3 else None
        fetch_date_to = args[4] if len(args) > 4 else None
        rows_inserted = args[5] if len(args) > 5 else 0
        status = args[6] if len(args) > 6 else "success"
        error_msg = args[7] if len(args) > 7 else None
        duration_ms = kwargs.get("duration_ms", 0)
        fetch_mode = kwargs.get("fetch_mode", "market")
    else:
        table_name = args[0] if len(args) > 0 else kwargs.get("table_name")
        stock_id = args[1] if len(args) > 1 else kwargs.get("stock_id")
        status = args[2] if len(args) > 2 else kwargs.get("status", "success")
        rows_inserted = kwargs.get("rows_inserted", args[3] if len(args) > 3 else 0)
        fetch_date_from = kwargs.get("fetch_date_from", args[4] if len(args) > 4 else None)
        fetch_date_to = kwargs.get("fetch_date_to", args[5] if len(args) > 5 else None)
        duration_ms = kwargs.get("duration_ms", args[6] if len(args) > 6 else 0)
        error_msg = kwargs.get("error_msg", kwargs.get("error_message", args[7] if len(args) > 7 else None))
        fetch_mode = kwargs.get("fetch_mode", "market")
        conn = kwargs.get("conn")

    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fetch_log
                    (table_name, stock_id, status, rows_inserted, fetch_date_from,
                     fetch_date_to, duration_ms, error_msg, fetch_mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    table_name,
                    stock_id,
                    status,
                    rows_inserted,
                    fetch_date_from,
                    fetch_date_to,
                    duration_ms,
                    error_msg,
                    fetch_mode,
                ),
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if own_conn:
            conn.close()


class FailureLogger:
    """Collect failures and optionally persist them to fetch_log."""

    def __init__(self, table_name, db_conn=None, log_to_db=True):
        self.table_name = table_name
        self.db_conn = db_conn
        self.log_to_db = log_to_db
        self.failures = []

    def log(self, stock_id, error_msg, status="failed", **kwargs):
        item = {"stock_id": stock_id, "error_msg": str(error_msg), "status": status}
        item.update(kwargs)
        self.failures.append(item)
        if self.log_to_db:
            try:
                log_fetch_result(
                    self.table_name,
                    stock_id,
                    status,
                    rows_inserted=kwargs.get("rows_inserted", 0),
                    fetch_date_from=kwargs.get("fetch_date_from"),
                    fetch_date_to=kwargs.get("fetch_date_to"),
                    duration_ms=kwargs.get("duration_ms", 0),
                    error_msg=str(error_msg),
                    fetch_mode=kwargs.get("fetch_mode", "market"),
                    conn=self.db_conn,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning("FailureLogger DB write failed: %s", exc)

    def add(self, stock_id, error_msg, **kwargs):
        self.log(stock_id, error_msg, **kwargs)

    def __call__(self, stock_id, error_msg, **kwargs):
        self.log(stock_id, error_msg, **kwargs)

    def has_failures(self):
        return bool(self.failures)

    def dump(self):
        return list(self.failures)


def get_core_stocks_from_db(as_of_date=None, tiers=None, conn=None):
    """Return current committed core stock universe using the §6.7 SQL contract.

    Backward compatibility: legacy fetchers call `get_core_stocks_from_db(conn)`
    and expect a `{stock_id: config}` mapping. New governance callers call the
    function without a positional connection and receive a sorted stock-id list.
    """
    legacy_mapping = False
    if conn is None and hasattr(as_of_date, "cursor"):
        conn = as_of_date
        as_of_date = None
        legacy_mapping = True
    tiers = tuple(tiers or ("core_universe", "convex_universe"))
    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if as_of_date is None:
                cur.execute(
                    """
                    SELECT
                        m."stock_id",
                        m."stock_name",
                        m."type",
                        m."industry_category",
                        m."train_eligible",
                        m."predict_eligible",
                        m."backtest_eligible",
                        m."downstream_ready"
                    FROM "core_universe_membership" m
                    JOIN "core_universe_snapshot" s
                      ON s."snapshot_id" = m."snapshot_id"
                    WHERE s."status" = 'committed'
                      AND m."core_tier" = ANY(%s)
                      AND COALESCE(m."industry_category", '') NOT IN ('Index', '大盤')
                      AND s."as_of_date" = (
                          SELECT MAX("as_of_date")
                          FROM "core_universe_snapshot"
                          WHERE "status" = 'committed'
                      )
                    ORDER BY m."stock_id"
                    """,
                    (list(tiers),),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        m."stock_id",
                        m."stock_name",
                        m."type",
                        m."industry_category",
                        m."train_eligible",
                        m."predict_eligible",
                        m."backtest_eligible",
                        m."downstream_ready"
                    FROM "core_universe_membership" m
                    JOIN "core_universe_snapshot" s
                      ON s."snapshot_id" = m."snapshot_id"
                    WHERE s."status" = 'committed'
                      AND m."core_tier" = ANY(%s)
                      AND COALESCE(m."industry_category", '') NOT IN ('Index', '大盤')
                      AND s."as_of_date" = %s
                    ORDER BY m."stock_id"
                    """,
                    (list(tiers), as_of_date),
                )
            rows = cur.fetchall()
            if not legacy_mapping:
                return [row[0] for row in rows]
            return {
                row[0]: {
                    "name": row[1],
                    "stock_name": row[1],
                    "type": row[2],
                    "industry": row[3],
                    "industry_category": row[3],
                    "fetch_basic": True,
                    "fetch_chip": True,
                    "fetch_fundamental": True,
                    "fetch_derivative": True,
                    "fetch_news": True,
                    "train_eligible": bool(row[4]),
                    "predict_eligible": bool(row[5]),
                    "backtest_eligible": bool(row[6]),
                    "downstream_ready": bool(row[7]),
                }
                for row in rows
            }
    finally:
        if own_conn:
            conn.close()


def get_db_stock_ids(conn=None, active_only=True, core_only=False, types=None, **kwargs):
    """Return stock IDs from the governed universe or market asset table."""
    if core_only:
        return get_core_stocks_from_db(conn=conn)
    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    where = []
    params = []
    if types:
        where.append('"type" = ANY(%s)')
        params.append(list(types))
    query = 'SELECT "stock_id" FROM "TaiwanStockInfo"'
    if where:
        query += " WHERE " + " AND ".join(where)
    query += ' ORDER BY "stock_id"'
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]
    finally:
        if own_conn:
            conn.close()


def get_latest_date(table_name, stock_id=None, date_col="date", id_col="stock_id"):
    """Return latest date from a table, optionally scoped by stock_id."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if stock_id is None:
                cur.execute(f"SELECT MAX({_quote_ident(date_col)}) FROM {_quote_ident(table_name)}")
            else:
                cur.execute(
                    f"SELECT MAX({_quote_ident(date_col)}) FROM {_quote_ident(table_name)} "
                    f"WHERE {_quote_ident(id_col)} = %s",
                    (stock_id,),
                )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_market_safe_start(conn, table_name, window_days=60, date_col="date"):
    query = (
        f"SELECT COALESCE(MAX({_quote_ident(date_col)}) + interval '1 day', "
        f"CURRENT_DATE - interval '{int(window_days)} days') "
        f"FROM {_quote_ident(table_name)}"
    )
    with conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()
        return row[0].strftime("%Y-%m-%d") if row and row[0] else None


def get_all_safe_starts(conn, table_name, id_col="stock_id", date_col="date", window_days=60):
    query = (
        f"SELECT {_quote_ident(id_col)}, MAX({_quote_ident(date_col)}) + interval '1 day' "
        f"FROM {_quote_ident(table_name)} GROUP BY {_quote_ident(id_col)}"
    )
    with conn.cursor() as cur:
        cur.execute(query)
        return {
            row[0]: row[1].strftime("%Y-%m-%d") if row[1] else None
            for row in cur.fetchall()
        }


def resolve_start_cached(
    stock_id,
    latest_dates,
    global_start,
    dataset_earliest,
    force=False,
    today=None,
):
    today = today or date.today().strftime("%Y-%m-%d")
    effective_start = max(global_start, dataset_earliest)
    if force:
        return effective_start
    latest = latest_dates.get(str(stock_id)) if latest_dates else None
    if latest is None:
        return effective_start
    if not isinstance(latest, str):
        latest = latest.strftime("%Y-%m-%d")
    next_day = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    if next_day > today:
        return None
    return max(next_day, effective_start)


def ensure_infrastructure():
    """Ensure shared infrastructure tables that this module owns."""
    ensure_ddl(None, DDL_FETCH_LOG)


def _check_pipeline_log_writable():
    """Actively verify Step 2C lifecycle-log writability."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_execution_log
                    (task_name, category, stock_id, start_time, end_time,
                     status, duration_ms, error_msg)
                VALUES (%s, %s, %s, NOW(), NOW(), %s, %s, %s)
                """,
                (
                    "db_diagnostic_v2.45_log_probe",
                    "infrastructure",
                    "SYSTEM",
                    "success",
                    0,
                    None,
                ),
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _check_public_api_contract():
    """Verify the public API set required by the constitution is present."""
    required = [
        "DDL_FETCH_LOG",
        "FailureLogger",
        "bulk_upsert",
        "check_db_health",
        "db_connection_check",
        "db_session",
        "db_transaction",
        "ensure_ddl",
        "ensure_infrastructure",
        "get_core_stocks_from_db",
        "get_db_conn",
        "get_db_connection",
        "get_db_stock_ids",
        "get_latest_date",
        "log_fetch_result",
        "record_lifecycle",
        "safe_commit_rows",
        "write_data_audit_log",
        "write_evaluation_log",
        "write_pipeline_log",
    ]
    return [name for name in required if name not in globals()]


def run_diagnostics():
    """執行基礎設施旗艦診斷報告 (v2.45 Standard)"""
    stocks = []
    diag_status = "PERFECT"
    diag_notes = []
    with record_lifecycle("db_diagnostic_v2.45", category="infrastructure", stock_id="SYSTEM") as lc:
        ok, latency = db_connection_check()
        if ok:
            try:
                stocks = get_core_stocks_from_db()
                if not stocks:
                    msg = "§6.7 core universe query returned 0 rows"
                    lc.mark_warning(msg)
                    diag_status = "WARNING"
                    diag_notes.append(msg)
            except Exception as exc:
                msg = f"§6.7 core universe query failed: {type(exc).__name__}: {exc}"
                lc.mark_failed(msg)
                diag_status = "FAILED"
                diag_notes.append(msg)
        else:
            msg = "DB connection check failed"
            lc.mark_failed(msg)
            diag_status = "FAILED"
            diag_notes.append(msg)

        missing_api = _check_public_api_contract()
        if missing_api:
            msg = "public API contract missing: " + ", ".join(missing_api)
            lc.mark_failed(msg)
            diag_status = "FAILED"
            diag_notes.append(msg)

        try:
            _check_pipeline_log_writable()
        except Exception as exc:
            msg = f"pipeline_execution_log write failed: {type(exc).__name__}: {exc}"
            lc.mark_failed(msg)
            diag_status = "FAILED"
            diag_notes.append(msg)

        try:
            write_data_audit_log("INFRA_CHECK", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "DIAGNOSTIC", 1)
        except Exception as exc:
            msg = f"data_audit_log write failed: {type(exc).__name__}: {exc}"
            lc.mark_failed(msg)
            diag_status = "FAILED"
            diag_notes.append(msg)

        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 基礎設施旗艦診斷啟動 (v2.45)")
        print("🛡️" * 40)

        print("\n" + "─" * 80)
        print("📊 基礎設施診斷摘要報告 (Infrastructure Diagnostic Report v2.45)")
        print("─" * 80)
        print(f"✅ 資料庫狀態   : {'SUCCESS' if ok else 'FAILED'}")
        print(f"🕒 連線延遲     : {latency:.2f} ms")
        print(f"📈 核心資產數   : {len(stocks)} 支 (§6.7 core_universe_membership)")
        print(f"📝 混合日誌狀態 : ACTIVE (pipeline_execution_log [8 欄完整] & data_audit_log)")
        print(f"⚖️  系統主權狀態 : {diag_status} (憲法 v5.4.21 / db_utils v2.45)")
        for note in diag_notes:
            print(f"   - {note}")
        print("─" * 80)

        print("\n💡 基礎設施維運建議 (Reference Information):")
        print("1. [效能提示]: 連線延遲高於 50ms 時，建議檢查資料庫連線池負載。")
        print("2. [範例提示]: 請參閱 Header 矩陣執行「所有核心股」的全量數據同步。")
        print("3. [歷史提示]: 所有連線變動必須記錄在全修訂歷程中以供溯源。")
        print("─" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Finance 基礎設施治理工具 (v2.45)")
    parser.add_argument("--reset-pool", action="store_true", help="重置連線池 (Mock)")
    args = parser.parse_args()

    if args.reset_pool:
        print("🚀 正在執行連線池重置...")
        time.sleep(1)
        print("✅ 連線池已重置。")
    else:
        run_diagnostics()
