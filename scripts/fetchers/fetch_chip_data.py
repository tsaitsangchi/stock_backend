"""
fetch_chip_data.py v4.0
Integrated fetcher for institutional investors, margin, and shareholding.
================================================================================
v4.0 Fixes:
  - Fully compatible with core/db_utils.py v4.0 (ThreadedConnectionPool).
  - Uses db_transaction() for automatic commit/rollback.
  - Supports FinMindClient v4.0 SQLite caching.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# --- Path Repair ---
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for sub in ("", "core"):
    p = (_SCRIPTS_DIR / sub) if sub else _SCRIPTS_DIR
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from core.db_utils import (
        db_session, db_transaction, ensure_ddl, commit_per_stock_per_day,
        get_latest_date, write_fetch_log, FailureLogger,
        safe_int, safe_float
    )
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"Import Error: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# DDL Definitions
# =====================================================================
DDL_INSTITUTIONAL = """
CREATE TABLE IF NOT EXISTS institutional_investors_buy_sell (
    date DATE NOT NULL,
    stock_id VARCHAR(20) NOT NULL,
    buy BIGINT,
    sell BIGINT,
    name VARCHAR(50) NOT NULL,
    PRIMARY KEY (date, stock_id, name)
);
"""

DDL_MARGIN = """
CREATE TABLE IF NOT EXISTS margin_purchase_short_sale (
    date DATE NOT NULL,
    stock_id VARCHAR(20) NOT NULL,
    MarginPurchaseBuy BIGINT,
    MarginPurchaseSell BIGINT,
    MarginPurchaseCashRepayment BIGINT,
    MarginPurchaseYesterdayBalance BIGINT,
    MarginPurchaseTodayBalance BIGINT,
    MarginPurchaseLimit BIGINT,
    ShortSaleBuy BIGINT,
    ShortSaleSell BIGINT,
    ShortSaleCashRepayment BIGINT,
    ShortSaleYesterdayBalance BIGINT,
    ShortSaleTodayBalance BIGINT,
    ShortSaleLimit BIGINT,
    OffsetLoanAndShort BIGINT,
    Note VARCHAR(100),
    PRIMARY KEY (date, stock_id)
);
"""

DDL_SHAREHOLDING = """
CREATE TABLE IF NOT EXISTS shareholding (
    date DATE NOT NULL,
    stock_id VARCHAR(20) NOT NULL,
    Shareholding BIGINT,
    percent NUMERIC(20,6),
    HoldClass VARCHAR(50) NOT NULL,
    PRIMARY KEY (date, stock_id, HoldClass)
);
"""

# =====================================================================
# SQL Templates
# =====================================================================
UPSERT_INSTITUTIONAL = """
INSERT INTO institutional_investors_buy_sell (date, stock_id, buy, sell, name)
VALUES (%(date)s, %(stock_id)s, %(buy)s, %(sell)s, %(name)s)
ON CONFLICT (date, stock_id, name) DO UPDATE SET
    buy = EXCLUDED.buy,
    sell = EXCLUDED.sell;
"""

UPSERT_MARGIN = """
INSERT INTO margin_purchase_short_sale (
    date, stock_id, MarginPurchaseBuy, MarginPurchaseSell, MarginPurchaseCashRepayment,
    MarginPurchaseYesterdayBalance, MarginPurchaseTodayBalance, MarginPurchaseLimit,
    ShortSaleBuy, ShortSaleSell, ShortSaleCashRepayment, ShortSaleYesterdayBalance,
    ShortSaleTodayBalance, ShortSaleLimit, OffsetLoanAndShort, Note
) VALUES (
    %(date)s, %(stock_id)s, %(MarginPurchaseBuy)s, %(MarginPurchaseSell)s, %(MarginPurchaseCashRepayment)s,
    %(MarginPurchaseYesterdayBalance)s, %(MarginPurchaseTodayBalance)s, %(MarginPurchaseLimit)s,
    %(ShortSaleBuy)s, %(ShortSaleSell)s, %(ShortSaleCashRepayment)s, %(ShortSaleYesterdayBalance)s,
    %(ShortSaleTodayBalance)s, %(ShortSaleLimit)s, %(OffsetLoanAndShort)s, %(Note)s
) ON CONFLICT (date, stock_id) DO UPDATE SET
    MarginPurchaseBuy = EXCLUDED.MarginPurchaseBuy,
    MarginPurchaseSell = EXCLUDED.MarginPurchaseSell,
    MarginPurchaseCashRepayment = EXCLUDED.MarginPurchaseCashRepayment,
    MarginPurchaseYesterdayBalance = EXCLUDED.MarginPurchaseYesterdayBalance,
    MarginPurchaseTodayBalance = EXCLUDED.MarginPurchaseTodayBalance,
    MarginPurchaseLimit = EXCLUDED.MarginPurchaseLimit,
    ShortSaleBuy = EXCLUDED.ShortSaleBuy,
    ShortSaleSell = EXCLUDED.ShortSaleSell,
    ShortSaleCashRepayment = EXCLUDED.ShortSaleCashRepayment,
    ShortSaleYesterdayBalance = EXCLUDED.ShortSaleYesterdayBalance,
    ShortSaleTodayBalance = EXCLUDED.ShortSaleTodayBalance,
    ShortSaleLimit = EXCLUDED.ShortSaleLimit,
    OffsetLoanAndShort = EXCLUDED.OffsetLoanAndShort,
    Note = EXCLUDED.Note;
"""

UPSERT_SHAREHOLDING = """
INSERT INTO shareholding (date, stock_id, Shareholding, percent, HoldClass)
VALUES (%(date)s, %(stock_id)s, %(Shareholding)s, %(percent)s, %(HoldClass)s)
ON CONFLICT (date, stock_id, HoldClass) DO UPDATE SET
    Shareholding = EXCLUDED.Shareholding,
    percent = EXCLUDED.percent;
"""

# =====================================================================
# Mapper Functions
# =====================================================================
def map_institutional(row: dict) -> dict:
    return {
        "date": row["date"],
        "stock_id": row["stock_id"],
        "buy": safe_int(row.get("buy", 0)),
        "sell": safe_int(row.get("sell", 0)),
        "name": row["name"]
    }

def map_margin(row: dict) -> dict:
    return {
        "date": row["date"],
        "stock_id": row["stock_id"],
        "MarginPurchaseBuy": safe_int(row.get("MarginPurchaseBuy", 0)),
        "MarginPurchaseSell": safe_int(row.get("MarginPurchaseSell", 0)),
        "MarginPurchaseCashRepayment": safe_int(row.get("MarginPurchaseCashRepayment", 0)),
        "MarginPurchaseYesterdayBalance": safe_int(row.get("MarginPurchaseYesterdayBalance", 0)),
        "MarginPurchaseTodayBalance": safe_int(row.get("MarginPurchaseTodayBalance", 0)),
        "MarginPurchaseLimit": safe_int(row.get("MarginPurchaseLimit", 0)),
        "ShortSaleBuy": safe_int(row.get("ShortSaleBuy", 0)),
        "ShortSaleSell": safe_int(row.get("ShortSaleSell", 0)),
        "ShortSaleCashRepayment": safe_int(row.get("ShortSaleCashRepayment", 0)),
        "ShortSaleYesterdayBalance": safe_int(row.get("ShortSaleYesterdayBalance", 0)),
        "ShortSaleTodayBalance": safe_int(row.get("ShortSaleTodayBalance", 0)),
        "ShortSaleLimit": safe_int(row.get("ShortSaleLimit", 0)),
        "OffsetLoanAndShort": safe_int(row.get("OffsetLoanAndShort", 0)),
        "Note": row.get("Note", "")
    }

def map_shareholding(row: dict) -> dict:
    hc = str(row.get("HoldClass", ""))
    if hc.isdigit():
        hc = f"Class_{hc}"
    elif hc == "":
        hc = "Unknown"
        
    return {
        "date": row["date"],
        "stock_id": row["stock_id"],
        "Shareholding": safe_int(row.get("Shareholding", 0)),
        "percent": safe_float(row.get("percent", 0.0)),
        "HoldClass": hc
    }

# =====================================================================
# Main Fetch Logic (Per stock, Per table)
# =====================================================================
def fetch_per_stock_task(table: str, ddl: str, upsert_sql: str, ds_template: str, 
                         mapper_func, stock_ids: list, start_date: str, end_date: str, 
                         delay: float, force: bool, chunk_days: int = 90):
    """Fetch data from FinMind in chunks and commit to DB."""
    api = FinMindClient()
    fail_logger = FailureLogger(f"chip_{table}")

    ensure_ddl(ddl)

    logger.info(f"=== [{table}] Start ({len(stock_ids)} stocks) ===")
    
    for sid in stock_ids:
        current_start = start_date
        if not force and current_start == "2020-01-01":
            db_latest = get_latest_date(table, stock_id=sid)
            if db_latest:
                current_start = db_latest
        
        logger.info(f"  Fetching {sid} [{current_start} ~ {end_date}]")
        
        seg_start_dt = datetime.strptime(current_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        while seg_start_dt <= end_dt:
            seg_start = seg_start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(seg_start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            try:
                data = api.get_data(
                    dataset=ds_template,
                    data_id=sid,
                    start_date=seg_start,
                    end_date=seg_end
                )
                duration_ms = int((time.monotonic() - t0) * 1000)

                if not data:
                    seg_start_dt = seg_end_dt + timedelta(days=1)
                    continue

                records = [mapper_func(row) for row in data if 'date' in row]

                if records:
                    success_count, error_count = commit_per_stock_per_day(
                        table_name=table,
                        records=records,
                        upsert_query=upsert_sql,
                        stock_id=sid
                    )
                    
                    status = "success" if error_count == 0 else "partial"
                    write_fetch_log(table, sid, status, "chip_script", seg_start, seg_end, duration_ms)
                
                seg_start_dt = seg_end_dt + timedelta(days=1)
                time.sleep(delay)

            except Exception as e:
                duration_ms = int((time.monotonic() - t0) * 1000)
                logger.error(f"  [Error] {sid} {seg_start}: {e}")
                write_fetch_log(table, sid, "failed", "chip_script", seg_start, seg_end, duration_ms, str(e))
                fail_logger.log_failure(table, sid, seg_start, seg_end, str(e))
                seg_start_dt = seg_end_dt + timedelta(days=1)
        
        time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description="Chip Data Fetcher v4.0")
    parser.add_argument("--stock-id", required=True, help="Stock IDs separated by comma")
    parser.add_argument("--tables", default="all", help="Tables: all, institutional, margin, shareholding")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.today().strftime('%Y-%m-%d'), help="End date (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing data")
    args = parser.parse_args()

    stock_ids = [s.strip() for s in args.stock_id.split(",")]
    tables = [t.strip().lower() for t in args.tables.split(",")]
    do_all = "all" in tables

    try:
        if do_all or "institutional" in tables or "institutional_investors_buy_sell" in tables:
            fetch_per_stock_task(
                table="institutional_investors_buy_sell",
                ddl=DDL_INSTITUTIONAL, upsert_sql=UPSERT_INSTITUTIONAL,
                ds_template="TaiwanStockInstitutionalInvestorsBuySell",
                mapper_func=map_institutional, stock_ids=stock_ids,
                start_date=args.start, end_date=args.end, delay=args.delay, force=args.force
            )
        
        if do_all or "margin" in tables or "margin_purchase_short_sale" in tables:
            fetch_per_stock_task(
                table="margin_purchase_short_sale",
                ddl=DDL_MARGIN, upsert_sql=UPSERT_MARGIN,
                ds_template="TaiwanStockMarginPurchaseShortSale",
                mapper_func=map_margin, stock_ids=stock_ids,
                start_date=args.start, end_date=args.end, delay=args.delay, force=args.force
            )

        if do_all or "shareholding" in tables:
            fetch_per_stock_task(
                table="shareholding",
                ddl=DDL_SHAREHOLDING, upsert_sql=UPSERT_SHAREHOLDING,
                ds_template="TaiwanStockShareholding",
                mapper_func=map_shareholding, stock_ids=stock_ids,
                start_date=args.start, end_date=args.end, delay=args.delay, force=args.force
            )

    finally:
        logger.info("All chip data processing completed.")

if __name__ == "__main__":
    main()