"""
fetch_technical_data.py v4.0
Technical data fetcher (compatible with core v4.0 architecture)
================================================================================
v4.0 Fixes:
  - Fully compatible with ThreadedConnectionPool and db_transaction().
  - Uses FinMindClient singleton with SQLite caching.
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
DDL_STOCK_PRICE = """
CREATE TABLE IF NOT EXISTS stock_price (
    date DATE NOT NULL,
    stock_id VARCHAR(20) NOT NULL,
    open NUMERIC(20,6),
    high NUMERIC(20,6),
    low NUMERIC(20,6),
    close NUMERIC(20,6),
    volume BIGINT,
    PRIMARY KEY (date, stock_id)
);
"""

UPSERT_STOCK_PRICE = """
INSERT INTO stock_price (date, stock_id, open, high, low, close, volume)
VALUES (%(date)s, %(stock_id)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
ON CONFLICT (date, stock_id) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume;
"""

# =====================================================================
# Mapper
# =====================================================================
def map_price(row: dict) -> dict:
    return {
        "date": row["date"],
        "stock_id": row["stock_id"],
        "open": safe_float(row.get("open")),
        "high": safe_float(row.get("high")),
        "low": safe_float(row.get("low")),
        "close": safe_float(row.get("close")),
        "volume": safe_int(row.get("Trading_Volume", 0))
    }

# =====================================================================
# Core Logic
# =====================================================================
def fetch_prices(stock_ids: list, start: str, end: str, force: bool):
    api = FinMindClient()
    ensure_ddl(DDL_STOCK_PRICE)
    
    for sid in stock_ids:
        cur_start = start
        if not force:
            latest = get_latest_date("stock_price", sid)
            if latest: cur_start = latest
            
        try:
            t0 = time.monotonic()
            data = api.get_data("TaiwanStockPrice", sid, cur_start, end)
            duration = int((time.monotonic() - t0) * 1000)
            
            if not data: continue
            
            records = [map_price(row) for row in data if 'date' in row]
            if records:
                success, error = commit_per_stock_per_day("stock_price", records, UPSERT_STOCK_PRICE, sid)
                write_fetch_log("stock_price", sid, "success" if error == 0 else "partial", "technical", cur_start, end, duration)
                logger.info(f"  {sid}: committed {success} rows")
        except Exception as e:
            logger.error(f"  {sid} failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Technical Fetcher v4.0")
    parser.add_argument("--stock-id", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    stock_ids = [s.strip() for s in args.stock_id.split(",")]
    end = date.today().strftime("%Y-%m-%d")
    fetch_prices(stock_ids, "2020-01-01", end, args.force)

if __name__ == "__main__":
    main()