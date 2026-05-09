"""
fetch_advanced_chip_data.py v4.0
Advanced chip data fetcher (compatible with core v4.0 architecture)
================================================================================
v4.0 Fixes:
  - Compatible with db_transaction() and ThreadedConnectionPool.
  - Replaced finmind_get with FinMindClient singleton.
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
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
# DDL & SQL
# =====================================================================
DDL_MAP = {
    "securities_lending": """
        CREATE TABLE IF NOT EXISTS securities_lending (
            date DATE NOT NULL,
            stock_id VARCHAR(20) NOT NULL,
            ShortSaleSettlement BIGINT,
            ShortSaleTodayBalance BIGINT,
            ShortSaleYesterdayBalance BIGINT,
            ShortSaleLimit BIGINT,
            PRIMARY KEY (date, stock_id)
        );
    """,
    "daily_short_balance": """
        CREATE TABLE IF NOT EXISTS daily_short_balance (
            date DATE NOT NULL,
            stock_id VARCHAR(20) NOT NULL,
            ShortSaleTodayBalance BIGINT,
            ShortSaleYesterdayBalance BIGINT,
            ShortSaleTodaySell BIGINT,
            ShortSaleTodayBuy BIGINT,
            PRIMARY KEY (date, stock_id)
        );
    """
}

UPSERT_MAP = {
    "securities_lending": """
        INSERT INTO securities_lending (date, stock_id, ShortSaleSettlement, ShortSaleTodayBalance, ShortSaleYesterdayBalance, ShortSaleLimit)
        VALUES (%(date)s, %(stock_id)s, %(ShortSaleSettlement)s, %(ShortSaleTodayBalance)s, %(ShortSaleYesterdayBalance)s, %(ShortSaleLimit)s)
        ON CONFLICT (date, stock_id) DO UPDATE SET
            ShortSaleSettlement = EXCLUDED.ShortSaleSettlement,
            ShortSaleTodayBalance = EXCLUDED.ShortSaleTodayBalance,
            ShortSaleYesterdayBalance = EXCLUDED.ShortSaleYesterdayBalance,
            ShortSaleLimit = EXCLUDED.ShortSaleLimit;
    """,
    "daily_short_balance": """
        INSERT INTO daily_short_balance (date, stock_id, ShortSaleTodayBalance, ShortSaleYesterdayBalance, ShortSaleTodaySell, ShortSaleTodayBuy)
        VALUES (%(date)s, %(stock_id)s, %(ShortSaleTodayBalance)s, %(ShortSaleYesterdayBalance)s, %(ShortSaleTodaySell)s, %(ShortSaleTodayBuy)s)
        ON CONFLICT (date, stock_id) DO UPDATE SET
            ShortSaleTodayBalance = EXCLUDED.ShortSaleTodayBalance,
            ShortSaleYesterdayBalance = EXCLUDED.ShortSaleYesterdayBalance,
            ShortSaleTodaySell = EXCLUDED.ShortSaleTodaySell,
            ShortSaleTodayBuy = EXCLUDED.ShortSaleTodayBuy;
    """
}

# =====================================================================
# Mappers
# =====================================================================
def map_lending(row: dict) -> dict:
    return {
        "date": row["date"],
        "stock_id": row["stock_id"],
        "ShortSaleSettlement": safe_int(row.get("ShortSaleSettlement", 0)),
        "ShortSaleTodayBalance": safe_int(row.get("ShortSaleTodayBalance", 0)),
        "ShortSaleYesterdayBalance": safe_int(row.get("ShortSaleYesterdayBalance", 0)),
        "ShortSaleLimit": safe_int(row.get("ShortSaleLimit", 0))
    }

# =====================================================================
# Core Logic
# =====================================================================
def fetch_table(table: str, ds_name: str, mapper, stock_ids: list, start: str, end: str, force: bool):
    api = FinMindClient()
    ensure_ddl(DDL_MAP[table])
    logger.info(f"--- Fetching {table} ---")
    
    for sid in stock_ids:
        cur_start = start
        if not force:
            latest = get_latest_date(table, sid)
            if latest: cur_start = latest
            
        try:
            t0 = time.monotonic()
            data = api.get_data(ds_name, sid, cur_start, end)
            duration = int((time.monotonic() - t0) * 1000)
            
            if not data: continue
            
            records = [mapper(row) for row in data if 'date' in row]
            if records:
                success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], sid)
                write_fetch_log(table, sid, "success" if error == 0 else "partial", "advanced", cur_start, end, duration)
                logger.info(f"  {sid}: committed {success} rows")
        except Exception as e:
            logger.error(f"  {sid} failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Chip Fetcher v4.0")
    parser.add_argument("--stock-id", required=True)
    parser.add_argument("--tables", default="securities_lending")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    stock_ids = [s.strip() for s in args.stock_id.split(",")]
    tables = [t.strip() for t in args.tables.split(",")]
    end = date.today().strftime("%Y-%m-%d")
    
    if "securities_lending" in tables:
        fetch_table("securities_lending", "TaiwanStockSecuritiesLending", map_lending, stock_ids, "2020-01-01", end, args.force)

if __name__ == "__main__":
    main()