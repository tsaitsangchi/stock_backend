import sys
import logging
import time
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_fundamental_data.py v3.1 — 基本面資料（監控整合標準版）
================================================================================
v3.1 重大改進：
  · 整合 fetch_log v3.1：每一支股票、每一資料表（財報、營收、股利）均記錄抓取狀態。
  · 效能監控：精準追蹤每一請求的 API 耗時（duration_ms）。
  · 狀態追蹤：支援 success, failed, no_new_data, skipped 等標準化狀態。
  · 完整註解：提供多樣化執行範例，便於維運。

支援資料表：
  · financial_statements (財報)
  · balance_sheet        (資產負債表)
  · month_revenue        (月營收)
  · dividend             (股利政策)

執行範例（常規）：
    python fetch_fundamental_data.py                # 抓取 150 支股票的所有基本面資料
    python fetch_fundamental_data.py --stock-id 2330 # 僅抓取 TSMC
    python fetch_fundamental_data.py --tables month_revenue # 僅抓取營收

執行範例（強制重抓）：
    python fetch_fundamental_data.py --stock-id 2330 --force --tables all
    python fetch_fundamental_data.py --start 2024-01-01 --force
    python fetch_fundamental_data.py --tables all --force

執行範例（模式切換）：
    python fetch_fundamental_data.py --retry-failed 7
    python fetch_fundamental_data.py --gap-fill 30
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    safe_int,
    safe_date,
    safe_timestamp,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_START_DATES = {
    "financial_statements": "1990-03-01",
    "balance_sheet":        "2011-12-01",
    "month_revenue":        "2002-02-01",
    "dividend":             "2005-05-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1990-03-01"

_CLI_ARGS_STR = " ".join(sys.argv)

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None):
    """v3.1 標準化日誌寫入"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (
                    run_ts, table_name, stock_id, status, rows_inserted, 
                    fetch_date_from, fetch_date_to, duration_ms, error_message, cli_args
                ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (table_name, stock_id, status, rows_inserted, fetch_date_from, fetch_date_to, duration_ms, error_message, _CLI_ARGS_STR))
        conn.commit()
    except Exception as e:
        logger.warning(f"無法寫入 fetch_log: {e}")

# ──────────────────────────────────────────────
# DDL & SQL
# ──────────────────────────────────────────────
DDL_FINANCIAL_STATEMENTS = """
CREATE TABLE IF NOT EXISTS financial_statements (date DATE, stock_id VARCHAR(50), type VARCHAR(50), value NUMERIC(20,4), origin_name VARCHAR(100), PRIMARY KEY (date, stock_id, type));
CREATE INDEX IF NOT EXISTS idx_fin_stock ON financial_statements (stock_id);
"""
DDL_BALANCE_SHEET = """
CREATE TABLE IF NOT EXISTS balance_sheet (date DATE, stock_id VARCHAR(50), type VARCHAR(50), value NUMERIC(20,4), origin_name VARCHAR(100), PRIMARY KEY (date, stock_id, type));
CREATE INDEX IF NOT EXISTS idx_bs_stock ON balance_sheet (stock_id);
"""
DDL_MONTH_REVENUE = """
CREATE TABLE IF NOT EXISTS month_revenue (date DATE, stock_id VARCHAR(50), country VARCHAR(50), revenue BIGINT, revenue_month INTEGER, revenue_year INTEGER, PRIMARY KEY (date, stock_id));
CREATE INDEX IF NOT EXISTS idx_rev_stock ON month_revenue (stock_id);
"""
DDL_DIVIDEND = """
CREATE TABLE IF NOT EXISTS dividend (
    date DATE, stock_id VARCHAR(50), year VARCHAR(4), stock_earnings_distribution NUMERIC(20,4), stock_statutory_surplus NUMERIC(20,4), stock_ex_dividend_trading_date DATE,
    total_employee_stock_dividend NUMERIC(20,4), total_employee_stock_dividend_amount NUMERIC(20,4), ratio_of_employee_stock_dividend_of_total NUMERIC(20,4), ratio_of_employee_stock_dividend NUMERIC(20,4),
    cash_earnings_distribution NUMERIC(20,4), cash_statutory_surplus NUMERIC(20,4), cash_ex_dividend_trading_date DATE, cash_dividend_payment_date DATE, total_employee_cash_dividend NUMERIC(20,4),
    total_number_of_cash_capital_increase NUMERIC(20,4), cash_increase_subscription_rate NUMERIC(20,4), cash_increase_subscription_price NUMERIC(20,4), remuneration_of_directors_and_supervisors NUMERIC(20,4),
    participate_distribution_of_total_shares NUMERIC(20,4), announcement_date DATE, announcement_time TIMESTAMP, PRIMARY KEY (date, stock_id)
);
"""

UPSERT_FINANCIAL_STATEMENTS = """INSERT INTO financial_statements (date, stock_id, type, value, origin_name) VALUES %s ON CONFLICT (date, stock_id, type) DO UPDATE SET value = EXCLUDED.value;"""
UPSERT_BALANCE_SHEET = """INSERT INTO balance_sheet (date, stock_id, type, value, origin_name) VALUES %s ON CONFLICT (date, stock_id, type) DO UPDATE SET value = EXCLUDED.value;"""
UPSERT_MONTH_REVENUE = """INSERT INTO month_revenue (date, stock_id, country, revenue, revenue_month, revenue_year) VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET revenue = EXCLUDED.revenue;"""
UPSERT_DIVIDEND = """
INSERT INTO dividend (date, stock_id, year, stock_earnings_distribution, stock_statutory_surplus, stock_ex_dividend_trading_date, total_employee_stock_dividend, total_employee_stock_dividend_amount, ratio_of_employee_stock_dividend_of_total, ratio_of_employee_stock_dividend, cash_earnings_distribution, cash_statutory_surplus, cash_ex_dividend_trading_date, cash_dividend_payment_date, total_employee_cash_dividend, total_number_of_cash_capital_increase, cash_increase_subscription_rate, cash_increase_subscription_price, remuneration_of_directors_and_supervisors, participate_distribution_of_total_shares, announcement_date, announcement_time)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET year = EXCLUDED.year;
"""

# ──────────────────────────────────────────────
# Mappers
# ──────────────────────────────────────────────
def map_fin_row(r: dict) -> tuple:
    return (r["date"], r["stock_id"], str(r.get("type", ""))[:50], safe_float(r.get("value")), str(r.get("origin_name", ""))[:100])

def map_rev_row(r: dict) -> tuple:
    return (r["date"], r["stock_id"], str(r.get("country", ""))[:50], safe_int(r.get("revenue")), safe_int(r.get("revenue_month")), safe_int(r.get("revenue_year")))

def map_div_row(r: dict) -> tuple:
    ann_time_raw = r.get("AnnouncementTime", "")
    ann_date_raw = r.get("AnnouncementDate", "")
    ann_time = safe_timestamp(f"{ann_date_raw} {ann_time_raw}") if (ann_time_raw and ann_date_raw) else None
    return (r["date"], r["stock_id"], str(r.get("year", ""))[:4], safe_float(r.get("StockEarningsDistribution")), safe_float(r.get("StockStatutorySurplus")), safe_date(r.get("StockExDividendTradingDate")), safe_float(r.get("TotalEmployeeStockDividend")), safe_float(r.get("TotalEmployeeStockDividendAmount")), safe_float(r.get("RatioOfEmployeeStockDividendOfTotal")), safe_float(r.get("RatioOfEmployeeStockDividend")), safe_float(r.get("CashEarningsDistribution")), safe_float(r.get("CashStatutorySurplus")), safe_date(r.get("CashExDividendTradingDate")), safe_date(r.get("CashDividendPaymentDate")), safe_float(r.get("TotalEmployeeCashDividend")), safe_float(r.get("TotalNumberOfCashCapitalIncrease")), safe_float(r.get("CashIncreaseSubscriptionRate")), safe_float(r.get("CashIncreaseSubscriptionPrice") or r.get("CashIncreaseSubscriptionpRrice")), safe_float(r.get("RemunerationOfDirectorsAndSupervisors")), safe_float(r.get("ParticipateDistributionOfTotalShares")), safe_date(ann_date_raw), ann_time)

# ──────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────
def fetch_quarterly(conn, tables, start, end, delay, force, stock_ids):
    for table in [t for t in ["financial_statements", "balance_sheet"] if t in tables]:
        logger.info(f"=== [{table}] 開始 ===")
        ensure_ddl(conn, DDL_FINANCIAL_STATEMENTS if table=="financial_statements" else DDL_BALANCE_SHEET)
        latest = get_all_safe_starts(conn, table)
        flog = FailureLogger(table, db_conn=conn)
        upsert_sql = UPSERT_FINANCIAL_STATEMENTS if table=="financial_statements" else UPSERT_BALANCE_SHEET
        dataset = "TaiwanStockFinancialStatements" if table=="financial_statements" else "TaiwanStockBalanceSheet"
        total_rows = 0
        for sid in stock_ids:
            s = resolve_start_cached(sid, latest, start, DATASET_START_DATES[table], force)
            if not s:
                _write_fetch_log(conn, table, sid, "skipped")
                continue
            
            start_ts = time.time()
            try:
                data = finmind_get(dataset, {"data_id": sid, "start_date": s, "end_date": end}, delay)
                duration = int((time.time() - start_ts) * 1000)
                if data:
                    rows = [map_fin_row(r) for r in data]
                    rows = dedup_rows(rows, (0, 1, 2))
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, "(%s::date, %s, %s, %s::numeric, %s)", label_prefix=table, failure_logger=flog)
                    count = sum(res.values())
                    total_rows += count
                    _write_fetch_log(conn, table, sid, "success", rows_inserted=count, fetch_date_from=s, fetch_date_to=end, duration_ms=duration)
                else:
                    _write_fetch_log(conn, table, sid, "no_new_data", duration_ms=duration)
            except Exception as e:
                duration = int((time.time() - start_ts) * 1000)
                flog.record(stock_id=sid, error=str(e))
                _write_fetch_log(conn, table, sid, "failed", duration_ms=duration, error_message=str(e))
        logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
        flog.summary()

def fetch_revenue(conn, start, end, delay, force, stock_ids):
    logger.info("=== [month_revenue] 開始 ===")
    ensure_ddl(conn, DDL_MONTH_REVENUE)
    latest = get_all_safe_starts(conn, "month_revenue")
    flog = FailureLogger("month_revenue", db_conn=conn)
    total_rows = 0
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START_DATES["month_revenue"], force)
        if not s:
            _write_fetch_log(conn, "month_revenue", sid, "skipped")
            continue
        
        start_ts = time.time()
        try:
            data = finmind_get("TaiwanStockMonthRevenue", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            duration = int((time.time() - start_ts) * 1000)
            if data:
                rows = [map_rev_row(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_MONTH_REVENUE, rows, "(%s::date, %s, %s, %s, %s, %s)", label_prefix="month_revenue", failure_logger=flog)
                count = sum(res.values())
                total_rows += count
                _write_fetch_log(conn, "month_revenue", sid, "success", rows_inserted=count, fetch_date_from=s, fetch_date_to=end, duration_ms=duration)
            else:
                _write_fetch_log(conn, "month_revenue", sid, "no_new_data", duration_ms=duration)
        except Exception as e:
            duration = int((time.time() - start_ts) * 1000)
            flog.record(stock_id=sid, error=str(e))
            _write_fetch_log(conn, "month_revenue", sid, "failed", duration_ms=duration, error_message=str(e))
    logger.info(f"  [month_revenue] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_dividend(conn, start, end, delay, force, stock_ids):
    logger.info("=== [dividend] 開始 ===")
    ensure_ddl(conn, DDL_DIVIDEND)
    latest = get_all_safe_starts(conn, "dividend")
    flog = FailureLogger("dividend", db_conn=conn)
    total_rows = 0
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START_DATES["dividend"], force)
        if not s:
            _write_fetch_log(conn, "dividend", sid, "skipped")
            continue
        
        start_ts = time.time()
        try:
            data = finmind_get("TaiwanStockDividend", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            duration = int((time.time() - start_ts) * 1000)
            if data:
                rows = [map_div_row(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_DIVIDEND, rows, "(%s::date, %s, %s, %s::numeric, %s::numeric, %s::date, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::date, %s::date, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::date, %s::timestamp)", label_prefix="dividend", failure_logger=flog)
                count = sum(res.values())
                total_rows += count
                _write_fetch_log(conn, "dividend", sid, "success", rows_inserted=count, fetch_date_from=s, fetch_date_to=end, duration_ms=duration)
            else:
                _write_fetch_log(conn, "dividend", sid, "no_new_data", duration_ms=duration)
        except Exception as e:
            duration = int((time.time() - start_ts) * 1000)
            flog.record(stock_id=sid, error=str(e))
            _write_fetch_log(conn, "dividend", sid, "failed", duration_ms=duration, error_message=str(e))
    logger.info(f"  [dividend] 總共寫入 {total_rows} 筆")
    flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["financial_statements", "balance_sheet", "month_revenue", "dividend", "all"], default=["all"])
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stock-id", default=None)
    p.add_argument("--retry-failed", type=int, help="不適用於此腳本，僅為一致性保留")
    p.add_argument("--gap-fill", type=int, help="不適用於此腳本，僅為一致性保留")
    args = p.parse_args()

    tables = ["financial_statements", "balance_sheet", "month_revenue", "dividend"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        from core.db_utils import get_db_stock_ids
        stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else get_db_stock_ids(conn)
        if set(["financial_statements", "balance_sheet"]) & set(tables): fetch_quarterly(conn, tables, args.start, args.end, args.delay, args.force, stock_ids)
        if "month_revenue" in tables: fetch_revenue(conn, args.start, args.end, args.delay, args.force, stock_ids)
        if "dividend" in tables: fetch_dividend(conn, args.start, args.end, args.delay, args.force, stock_ids)
    finally:
        conn.close()

if __name__ == "__main__":
    main()