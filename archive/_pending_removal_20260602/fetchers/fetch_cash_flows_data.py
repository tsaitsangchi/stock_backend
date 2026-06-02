"""
fetch_cash_flows_data.py — 現金流量表 + 除權息結果（v3.1 核心模組全面升級版）
================================================================================
v3.1 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 使用 `commit_per_stock_per_day` 進行雙層粒度原子寫入，確保極致的資料完整性。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

執行範例：
    # 常規全量/增量抓取 (所有支援的表)
    python scripts/fetchers/fetch_cash_flows_data.py

    # 針對特定表抓取
    python scripts/fetchers/fetch_cash_flows_data.py --tables cash_flows_statement dividend_result

    # 重試近 7 天失敗目標
    python scripts/fetchers/fetch_cash_flows_data.py --retry-failed 7

    # 補足近 30 天無紀錄目標
    python scripts/fetchers/fetch_cash_flows_data.py --gap-fill 30

    # 針對特定個股抓取
    python scripts/fetchers/fetch_cash_flows_data.py --stock-id 2330,2454 --tables cash_flows_statement

    # 強制重新抓取特定個股資料（無視增量檢查）
    python scripts/fetchers/fetch_cash_flows_data.py --stock-id 2330 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.finmind_client import finmind_get, get_request_stats
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    get_db_stock_ids,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATASET_START = {
    "cash_flows_statement": "2008-06-01",
    "dividend_result":      "2003-05-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 日誌與 SQL
# ─────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")

def _write_fetch_log(conn, **kwargs):
    """寫入 fetch_log，失敗不影響主流程。"""
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO fetch_log (
                run_ts, table_name, stock_id, fetch_mode,
                fetch_date_from, fetch_date_to,
                rows_inserted, rows_updated, duration_ms,
                status, error_message, cli_args
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
            """
            cur.execute(sql, (
                kwargs.get("table_name"), kwargs.get("stock_id"), kwargs.get("fetch_mode", "per_stock"),
                kwargs.get("fetch_date_from"), kwargs.get("fetch_date_to"),
                kwargs.get("rows_inserted", 0), kwargs.get("duration_ms", 0),
                kwargs.get("status"), kwargs.get("error_message"), _CLI_ARGS_STR
            ))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.debug(f"fetch_log 寫入失敗：{e}")

DDL_CASH_FLOWS = """
CREATE TABLE IF NOT EXISTS cash_flows_statement (
    date DATE, stock_id VARCHAR(50), type VARCHAR(100), 
    value NUMERIC(20,4), origin_name VARCHAR(200), 
    PRIMARY KEY (date, stock_id, type)
);
"""
DDL_DIVIDEND_RESULT = """
CREATE TABLE IF NOT EXISTS dividend_result (
    date DATE, stock_id VARCHAR(50), 
    before_price NUMERIC(20,4), after_price NUMERIC(20,4), 
    stock_and_cache_dividend NUMERIC(20,4), stock_or_cache_dividend VARCHAR(20), 
    max_price NUMERIC(20,4), min_price NUMERIC(20,4), 
    open_price NUMERIC(20,4), reference_price NUMERIC(20,4), 
    PRIMARY KEY (date, stock_id)
);
"""

UPSERT_CASH_FLOWS = """
INSERT INTO cash_flows_statement (date, stock_id, type, value, origin_name) 
VALUES %s ON CONFLICT (date, stock_id, type) DO UPDATE SET value = EXCLUDED.value;
"""
UPSERT_DIVIDEND_RESULT = """
INSERT INTO dividend_result (date, stock_id, before_price, after_price, stock_and_cache_dividend, stock_or_cache_dividend, max_price, min_price, open_price, reference_price) 
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET after_price = EXCLUDED.after_price;
"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_cf_row(r: dict) -> tuple:
    return (r["date"], r["stock_id"], r.get("type", "")[:100], safe_float(r.get("value")), r.get("origin_name", "")[:200])

def map_dr_row(r: dict) -> tuple:
    return (r["date"], r["stock_id"], safe_float(r.get("before_price")), safe_float(r.get("after_price")), safe_float(r.get("stock_and_cache_dividend")), r.get("stock_or_cache_dividend", "")[:20], safe_float(r.get("max_price")), safe_float(r.get("min_price")), safe_float(r.get("open_price")), safe_float(r.get("reference_price")))

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_dataset(
    conn, dataset: str, table: str, ddl: str, upsert_sql: str, 
    mapper, dataset_key: str, stock_ids: list[str], 
    start: str, end: str, delay: float, force: bool,
    fetch_mode_override: str | None = None
):
    ensure_ddl(conn, ddl)
    conn.commit()
    latest = get_all_safe_starts(conn, table)
    flog = FailureLogger(table, db_conn=conn)
    total_rows = 0
    
    template = "(%s, %s, %s, %s, %s)" if table == "cash_flows_statement" else "(%s, %s, %s::numeric, %s::numeric, %s::numeric, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric)"
    fetch_mode = fetch_mode_override or "per_stock"

    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START[dataset_key], force)
        if not s:
            _write_fetch_log(conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            # 修正為具名參數 (Keyword arguments) 以確保穩定
            data = finmind_get(
                dataset=dataset, 
                params={"data_id": sid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            dur = int((time.time() - t0) * 1000)
            
            if data:
                rows = [mapper(r) for r in data]
                # 雙層粒度原子寫入
                res = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table, failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(
                    conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=n, 
                    duration_ms=dur, status="success" if n > 0 else "partial"
                )
            else:
                _write_fetch_log(
                    conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                    duration_ms=dur, status="no_new_data"
                )
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(
                conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                duration_ms=dur, status="failed", error_message=str(e)
            )
            
    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
    flog.summary()

# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int, target_tables: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    sql = """
    WITH recent AS (
        SELECT table_name, stock_id, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY table_name, stock_id ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = ANY(%s) AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT table_name, stock_id FROM recent WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (target_tables, str(days)))
            for tbl, sid in cur.fetchall():
                targets[tbl].append(sid)
    except Exception as e:
        logger.error(f"[retry-failed] 查詢失敗：{e}")
        return {}

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [retry-failed/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def query_gap_targets(conn, days: int, target_tables: list[str], all_stock_ids: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    for tbl in target_tables:
        sql = f"SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tbl, str(days), all_stock_ids))
                have_success = {row[0] for row in cur.fetchall()}
            missing = [sid for sid in all_stock_ids if sid not in have_success]
            targets[tbl].extend(missing)
        except Exception as e:
            logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [gap-fill/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def _config_for_table(table: str):
    if table == "cash_flows_statement":
        return ("TaiwanStockCashFlowsStatement", DDL_CASH_FLOWS, UPSERT_CASH_FLOWS, map_cf_row)
    elif table == "dividend_result":
        return ("TaiwanStockDividendResult", DDL_DIVIDEND_RESULT, UPSERT_DIVIDEND_RESULT, map_dr_row)
    return None

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        cfg = _config_for_table(tbl)
        if cfg:
            dataset, ddl, upsert, mapper = cfg
            fetch_dataset(conn, dataset, tbl, ddl, upsert, mapper, tbl, sids, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["cash_flows_statement", "dividend_result", "all"], default=["all"])
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default="2003-05-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["cash_flows_statement", "dividend_result"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else get_db_stock_ids(conn, types=("twse", "otc"))

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, tables)
            if targets: _run_targeted(conn, targets, args, fetch_mode="retry")
            else: logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, tables, stock_ids)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        if "cash_flows_statement" in tables:
            fetch_dataset(conn, "TaiwanStockCashFlowsStatement", "cash_flows_statement", DDL_CASH_FLOWS, UPSERT_CASH_FLOWS, map_cf_row, "cash_flows_statement", stock_ids, args.start, args.end, args.delay, args.force)
        if "dividend_result" in tables:
            fetch_dataset(conn, "TaiwanStockDividendResult", "dividend_result", DDL_DIVIDEND_RESULT, UPSERT_DIVIDEND_RESULT, map_dr_row, "dividend_result", stock_ids, args.start, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        # 於程式完全結束時，印出統一的 FinMind RequestStats 報表
        get_request_stats().summary()

if __name__ == "__main__":
    main()