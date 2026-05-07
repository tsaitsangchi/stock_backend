import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse
import time

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_cash_flows_data.py — 現金流量表 + 除權息結果（v3.1 fetch_log 整合版）
================================================================================
v3.1 改進：
  · 整合 fetch_log：每次抓取操作（不論成功、失敗、跳過）皆會記錄，
    作為 monitor.update_daily_status / 燈號判斷的依據。
  · fetch_log 寫入採 best-effort 模式，失敗僅印 warning，不中斷主流程。
  · 完整支援 v3.1 核心架構：FailureLogger + commit_per_stock_per_day。
  · 支援 --retry-failed N 與 --gap-fill N 模式，實現智慧補抓。

v3.0 既有：
  · 導入 commit_per_stock_per_day：每一對 (sid, date) 獨立原子寫入。
  · 全面整合 FailureLogger：精準追蹤健康度。

執行（常規）：
    python fetch_cash_flows_data.py
    python fetch_cash_flows_data.py --tables cash_flows_statement dividend_result
    python fetch_cash_flows_data.py --stock-id 2330 --force
    python fetch_cash_flows_data.py --stock-id 2330 --tables cash_flows_statement dividend_result --force
    python fetch_cash_flows_data.py --stock-id 2330,2454 --tables cash_flows_statement --force

執行（模式切換）：
    # 重試最近 7 天失敗的任務
    python fetch_cash_flows_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的任務
    python fetch_cash_flows_data.py --gap-fill 30
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DATASET_START = {
    "cash_flows_statement": "2008-06-01",
    "dividend_result":      "2003-05-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

_CLI_ARGS_STR = " ".join(sys.argv)

def _write_fetch_log(conn, **kwargs):
    """寫入 fetch_log，失敗不影響主流程。"""
    from core.db_utils import DDL_FETCH_LOG
    try:
        with conn.cursor() as cur:
            # 確保欄位名稱與新版 db_utils / monitoring 一致
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

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_CASH_FLOWS = """CREATE TABLE IF NOT EXISTS cash_flows_statement (date DATE, stock_id VARCHAR(50), type VARCHAR(100), value NUMERIC(20,4), origin_name VARCHAR(200), PRIMARY KEY (date, stock_id, type));"""
DDL_DIVIDEND_RESULT = """CREATE TABLE IF NOT EXISTS dividend_result (date DATE, stock_id VARCHAR(50), before_price NUMERIC(20,4), after_price NUMERIC(20,4), stock_and_cache_dividend NUMERIC(20,4), stock_or_cache_dividend VARCHAR(20), max_price NUMERIC(20,4), min_price NUMERIC(20,4), open_price NUMERIC(20,4), reference_price NUMERIC(20,4), PRIMARY KEY (date, stock_id));"""

UPSERT_CASH_FLOWS = """INSERT INTO cash_flows_statement (date, stock_id, type, value, origin_name) VALUES %s ON CONFLICT (date, stock_id, type) DO UPDATE SET value = EXCLUDED.value;"""
UPSERT_DIVIDEND_RESULT = """INSERT INTO dividend_result (date, stock_id, before_price, after_price, stock_and_cache_dividend, stock_or_cache_dividend, max_price, min_price, open_price, reference_price) VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET after_price = EXCLUDED.after_price;"""

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
def fetch_dataset(conn, dataset, table, ddl, upsert_sql, mapper, dataset_key, stock_ids, start, end, delay, force):
    ensure_ddl(conn, ddl)
    latest = get_all_safe_starts(conn, table)
    flog = FailureLogger(table, db_conn=conn)
    total_rows = 0
    template = "(%s, %s, %s, %s, %s)" if table == "cash_flows_statement" else "(%s, %s, %s::numeric, %s::numeric, %s::numeric, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric)"

    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START[dataset_key], force)
        if not s:
            _write_fetch_log(conn, table_name=table, stock_id=sid, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            data = finmind_get(dataset, {"data_id": sid, "start_date": s, "end_date": end}, delay)
            dur = int((time.time() - t0) * 1000)
            if data:
                rows = [mapper(r) for r in data]
                res = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table, failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, table_name=table, stock_id=sid, fetch_date_from=s, fetch_date_to=end, 
                                 rows_inserted=n, duration_ms=dur, status="success")
            else:
                _write_fetch_log(conn, table_name=table, stock_id=sid, fetch_date_from=s, fetch_date_to=end, 
                                 rows_inserted=0, duration_ms=dur, status="no_new_data")
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e))
            _write_fetch_log(conn, table_name=table, stock_id=sid, fetch_date_from=s, fetch_date_to=end, 
                             rows_inserted=0, duration_ms=dur, status="failed", error_message=str(e))
    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
    flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["cash_flows_statement", "dividend_result", "all"], default=["all"])
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default="2003-05-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["cash_flows_statement", "dividend_result"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        from core.db_utils import get_db_stock_ids
        stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else get_db_stock_ids(conn)
        if "cash_flows_statement" in tables: fetch_dataset(conn, "TaiwanStockCashFlowsStatement", "cash_flows_statement", DDL_CASH_FLOWS, UPSERT_CASH_FLOWS, map_cf_row, "cash_flows_statement", stock_ids, args.start, args.end, args.delay, args.force)
        if "dividend_result" in tables: fetch_dataset(conn, "TaiwanStockDividendResult", "dividend_result", DDL_DIVIDEND_RESULT, UPSERT_DIVIDEND_RESULT, map_dr_row, "dividend_result", stock_ids, args.start, args.end, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()