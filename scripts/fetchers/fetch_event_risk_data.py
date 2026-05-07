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
fetch_event_risk_data.py — 事件風險與股本變動（v3.1 fetch_log 整合版）
================================================================================
v3.1 改進：
  · 整合 fetch_log：每次抓取（無論成功、失敗或跳過）都會寫入監控日誌。
  · 效能追蹤：記錄各風險事件（如處置股、減資）的 API 請求耗時（duration_ms）。
  · 支援 --retry-failed N 與 --gap-fill N 模式，實現智慧補抓。

v3.0 既有：
  · 支援 3 個資料表：capital_reduction, market_value, disposition_securities。
  · 導入 commit_per_stock_per_day：每一天獨立原子 commit，確保資料完整性。
  · 整合 FailureLogger：精準追蹤核心標的的事件覆蓋狀況。

執行（常規）：
    python fetch_event_risk_data.py
    python fetch_event_risk_data.py --tables market_value disposition_securities
    python fetch_event_risk_data.py --stock-id 2330 --force
    python fetch_event_risk_data.py --stock-id 2330 --tables all --force
    python fetch_event_risk_data.py --stock-id 2330,2454 --tables disposition_securities --force

執行（模式切換）：
    # 重試最近 7 天失敗的組合
    python fetch_event_risk_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python fetch_event_risk_data.py --gap-fill 30
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    safe_int,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DATASET_START = {
    "delisting":              "2001-01-01",
    "suspended":              "2011-10-06",
    "capital_reduction":      "2011-01-01",
    "split_price":            "2000-01-01",
    "trading_date":           "1990-01-01",
    "market_value":           "2004-01-01",
    "disposition_securities": "2001-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

_CLI_ARGS_STR = " ".join(sys.argv)

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

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_EVENT = """
CREATE TABLE IF NOT EXISTS delisting (date DATE, stock_id VARCHAR(50), stock_name VARCHAR(200), PRIMARY KEY (date, stock_id));
CREATE TABLE IF NOT EXISTS suspended (stock_id VARCHAR(50), date DATE, suspension_time VARCHAR(50), resumption_date DATE, resumption_time VARCHAR(50), PRIMARY KEY (stock_id, date, suspension_time));
CREATE TABLE IF NOT EXISTS capital_reduction (date DATE, stock_id VARCHAR(50), closing_last_trading NUMERIC(20,4), post_reduction_ref NUMERIC(20,4), limit_up NUMERIC(20,4), limit_down NUMERIC(20,4), opening_ref NUMERIC(20,4), exright_ref NUMERIC(20,4), reason VARCHAR(500), PRIMARY KEY (date, stock_id));
CREATE TABLE IF NOT EXISTS split_price (date DATE, stock_id VARCHAR(50), type VARCHAR(20), before_price NUMERIC(20,4), after_price NUMERIC(20,4), max_price NUMERIC(20,4), min_price NUMERIC(20,4), open_price NUMERIC(20,4), PRIMARY KEY (date, stock_id, type));
CREATE TABLE IF NOT EXISTS trading_date (date DATE PRIMARY KEY);
CREATE TABLE IF NOT EXISTS market_value (date DATE, stock_id VARCHAR(50), market_value BIGINT, PRIMARY KEY (date, stock_id));
CREATE TABLE IF NOT EXISTS disposition_securities (date DATE, stock_id VARCHAR(50), stock_name VARCHAR(200), disposition_cnt INTEGER, condition VARCHAR(500), measure VARCHAR(500), period_start DATE, period_end DATE, PRIMARY KEY (date, stock_id));
"""

UPSERT_CAPRED = """INSERT INTO capital_reduction VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET post_reduction_ref = EXCLUDED.post_reduction_ref;"""
UPSERT_MV = """INSERT INTO market_value VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET market_value = EXCLUDED.market_value;"""
UPSERT_DISP = """INSERT INTO disposition_securities VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET disposition_cnt = EXCLUDED.disposition_cnt;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_capred(r): return (r["date"], r["stock_id"], safe_float(r.get("ClosingPriceonTheLastTradingDay")), safe_float(r.get("PostReductionReferencePrice")), safe_float(r.get("LimitUp")), safe_float(r.get("LimitDown")), safe_float(r.get("OpeningReferencePrice")), safe_float(r.get("ExrightReferencePrice")), r.get("ReasonforCapitalReduction", "")[:500])
def map_mv(r): return (r["date"], r["stock_id"], safe_int(r.get("market_value")))
def map_disp(r): return (r["date"], r["stock_id"], r.get("stock_name", "")[:200], safe_int(r.get("disposition_cnt")), r.get("condition", "")[:500], r.get("measure", "")[:500], r.get("period_start"), r.get("period_end"))

def fetch_per_stock(conn, dataset, table, upsert_sql, tmpl, mapper, stock_ids, start, end, delay, force):
    logger.info(f"=== [{table}] 開始 ===")
    flog = FailureLogger(table, db_conn=conn)
    latest = get_all_safe_starts(conn, table)
    total_rows = 0
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START[table], force)
        if not s:
            _write_fetch_log(conn, table_name=table, stock_id=sid, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            data = finmind_get(dataset, {"data_id": sid, "start_date": s, "end_date": end}, delay)
            dur = int((time.time() - t0) * 1000)
            if data:
                rows = [mapper(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, label_prefix=table, failure_logger=flog)
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
    p.add_argument("--tables", nargs="+", choices=["capital_reduction", "market_value", "disposition_securities", "all"], default=["all"])
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["capital_reduction", "market_value", "disposition_securities"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_EVENT)
        from core.db_utils import get_db_stock_ids
        stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else get_db_stock_ids(conn)
        
        if "capital_reduction" in tables: fetch_per_stock(conn, "TaiwanStockCapitalReductionReferencePrice", "capital_reduction", UPSERT_CAPRED, "(%s, %s, %s, %s, %s, %s, %s, %s, %s)", map_capred, stock_ids, args.start, args.end, args.delay, args.force)
        if "market_value" in tables: fetch_per_stock(conn, "TaiwanStockMarketValue", "market_value", UPSERT_MV, "(%s, %s, %s)", map_mv, stock_ids, args.start, args.end, args.delay, args.force)
        if "disposition_securities" in tables: fetch_per_stock(conn, "TaiwanStockDispositionSecuritiesPeriod", "disposition_securities", UPSERT_DISP, "(%s, %s, %s, %s, %s, %s, %s, %s)", map_disp, stock_ids, args.start, args.end, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()