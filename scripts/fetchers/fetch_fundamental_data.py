import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_fundamental_data.py v3.0 — 基本面資料（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：財報、營收、股利每一天資料均獨立原子 commit。
  ★ 全面整合 FailureLogger：精準追蹤 150 支股票在不同基本面資料集間的抓取狀況。
  ★ 結構一致化：與技術面、國際面腳本維持相同寫入規範，確保生產矩陣穩定。
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
            if not s: continue
            try:
                data = finmind_get(dataset, {"data_id": sid, "start_date": s, "end_date": end}, delay)
                if data:
                    rows = [map_fin_row(r) for r in data]
                    rows = dedup_rows(rows, (0, 1, 2))
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, "(%s::date, %s, %s, %s::numeric, %s)", label_prefix=table, failure_logger=flog)
                    total_rows += sum(res.values())
            except Exception as e: flog.record(stock_id=sid, error=str(e))
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
        if not s: continue
        try:
            data = finmind_get("TaiwanStockMonthRevenue", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [map_rev_row(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_MONTH_REVENUE, rows, "(%s::date, %s, %s, %s, %s, %s)", label_prefix="month_revenue", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=sid, error=str(e))
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
        if not s: continue
        try:
            data = finmind_get("TaiwanStockDividend", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [map_div_row(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_DIVIDEND, rows, "(%s::date, %s, %s, %s::numeric, %s::numeric, %s::date, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::date, %s::date, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::date, %s::timestamp)", label_prefix="dividend", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=sid, error=str(e))
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