from __future__ import annotations
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_international_data.py — 國際市場資料（v3.1 fetch_log 整合版）
================================================================================
v3.1 改進：
  ★ 整合 fetch_log：將美股 (US Stocks)、原油 (Crude Oil) 與黃金 (Gold) 的抓取狀態寫入系統日誌表。
  ★ 標準化註解：提供完整執行範例，便於補抓特定標的或強制重抓。
  ★ 效能追蹤：記錄每一標的的抓取筆數與 API 耗時 (ms)。

支援資料表：
  · us_stock_price      (美股價量：AAPL, NVDA, TSLA, ...)
  · crude_oil_prices    (原油價格：WTI, Brent)
  · gold_price          (國際金價)

執行範例（常規）：
    python fetch_international_data.py                    # 抓取所有國際資料
    python fetch_international_data.py --ids AAPL,NVDA    # 僅抓取指定美股
    python fetch_international_data.py --tables gold_price # 僅抓取金價

執行範例（強制重抓）：
    python fetch_international_data.py --ids NVDA --force
    python fetch_international_data.py --tables all --force --start 2020-01-01

執行範例（補漏）：
    python fetch_international_data.py --start 2024-05-01
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    safe_int,
    get_all_safe_starts,
    get_market_safe_start,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    commit_per_day,
    dedup_rows,
)
from config import INTERNATIONAL_WATCHLIST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_START_DATES = {
    "us_stock_price":   "1962-01-02",
    "crude_oil_prices": "1987-05-20",
    "gold_price":       "1968-04-01",
}
CRUDE_OIL_IDS = ["WTI", "Brent"]
DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1962-01-02"

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

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_US_STOCK_PRICE = """
CREATE TABLE IF NOT EXISTS us_stock_price (
    date DATE, stock_id VARCHAR(50), adj_close NUMERIC(20,4), close NUMERIC(20,4),
    high NUMERIC(20,4), low NUMERIC(20,4), open NUMERIC(20,4), volume BIGINT,
    PRIMARY KEY (date, stock_id)
);
"""
DDL_CRUDE_OIL_PRICES = """
CREATE TABLE IF NOT EXISTS crude_oil_prices (
    date DATE, name VARCHAR(50), price NUMERIC(20,4),
    PRIMARY KEY (date, name)
);
"""
DDL_GOLD_PRICE = """
CREATE TABLE IF NOT EXISTS gold_price (
    date DATE, price NUMERIC(20,4),
    PRIMARY KEY (date)
);
"""

UPSERT_US_STOCK_PRICE = """
INSERT INTO us_stock_price (date, stock_id, adj_close, close, high, low, open, volume)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET close = EXCLUDED.close, volume = EXCLUDED.volume;
"""
UPSERT_CRUDE_OIL_PRICES = """
INSERT INTO crude_oil_prices (date, name, price)
VALUES %s ON CONFLICT (date, name) DO UPDATE SET price = EXCLUDED.price;
"""
UPSERT_GOLD_PRICE = """
INSERT INTO gold_price (date, price)
VALUES %s ON CONFLICT (date) DO UPDATE SET price = EXCLUDED.price;
"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def _fmt_date(d):
    s = str(d)
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S" if " " in s else "%Y-%m-%d").strftime("%Y-%m-%d")

def map_us_stock(r: dict) -> tuple:
    return (_fmt_date(r["date"]), r["stock_id"], safe_float(r.get("Adj_Close")), safe_float(r.get("Close")), safe_float(r.get("High")), safe_float(r.get("Low")), safe_float(r.get("Open")), safe_int(r.get("Volume")))

def map_crude_oil(r: dict) -> tuple:
    return (_fmt_date(r["date"]), r["name"], safe_float(r.get("price")))

def map_gold_price(r: dict) -> tuple:
    return (_fmt_date(r["date"]), "GOLD", safe_float(r.get("Price")))

# ─────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────
def fetch_us_stock_price(conn, start, end, delay, force, target_ids):
    logger.info("=== [us_stock_price] 開始 ===")
    ensure_ddl(conn, DDL_US_STOCK_PRICE)
    stock_ids = target_ids if target_ids else INTERNATIONAL_WATCHLIST
    latest = get_all_safe_starts(conn, "us_stock_price")
    flog = FailureLogger("us_stock_price", db_conn=conn)
    total_rows = 0

    for i, sid in enumerate(stock_ids, 1):
        s = resolve_start_cached(sid, latest, start, DATASET_START_DATES["us_stock_price"], force)
        if not s: 
            _write_fetch_log(conn, "us_stock_price", sid, "skipped")
            continue
        
        start_ts = time.time()
        try:
            data = finmind_get("USStockPrice", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            duration = int((time.time() - start_ts) * 1000)
            
            if data:
                rows = [map_us_stock(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_US_STOCK_PRICE, rows, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)", label_prefix="us_stock", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, "us_stock_price", sid, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration)
            else:
                _write_fetch_log(conn, "us_stock_price", sid, "no_new_data", duration_ms=duration)
        except Exception as e: 
            duration = int((time.time() - start_ts) * 1000)
            flog.record(stock_id=sid, error=str(e))
            _write_fetch_log(conn, "us_stock_price", sid, "failed", duration_ms=duration, error_message=str(e))
            
        if i % 20 == 0: logger.info(f"  進度：{i}/{len(stock_ids)}")

    flog.summary()
    logger.info(f"=== [us_stock_price] 完成：{total_rows} 筆 ===\n")

def fetch_crude_oil_prices(conn, start, end, delay, force):
    logger.info("=== [crude_oil_prices] 開始 ===")
    ensure_ddl(conn, DDL_CRUDE_OIL_PRICES)
    latest = get_all_safe_starts(conn, "crude_oil_prices", key_col="name")
    flog = FailureLogger("crude_oil", db_conn=conn)
    total_rows = 0

    for oid in CRUDE_OIL_IDS:
        s = resolve_start_cached(oid, latest, start, DATASET_START_DATES["crude_oil_prices"], force)
        if not s:
            _write_fetch_log(conn, "crude_oil_prices", oid, "skipped")
            continue
            
        start_ts = time.time()
        try:
            data = finmind_get("CrudeOilPrices", {"data_id": oid, "start_date": s, "end_date": end}, delay)
            duration = int((time.time() - start_ts) * 1000)
            
            if data:
                rows = [map_crude_oil(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_CRUDE_OIL_PRICES, rows, "(%s::date,%s,%s::numeric)", label_prefix="crude_oil", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, "crude_oil_prices", oid, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration)
            else:
                _write_fetch_log(conn, "crude_oil_prices", oid, "no_new_data", duration_ms=duration)
        except Exception as e: 
            duration = int((time.time() - start_ts) * 1000)
            flog.record(stock_id=oid, error=str(e))
            _write_fetch_log(conn, "crude_oil_prices", oid, "failed", duration_ms=duration, error_message=str(e))

    flog.summary()
    logger.info(f"=== [crude_oil_prices] 完成：{total_rows} 筆 ===\n")

def fetch_gold_price(conn, start, end, delay, force):
    logger.info("=== [gold_price] 開始 ===")
    ensure_ddl(conn, DDL_GOLD_PRICE)
    m_start = get_market_safe_start(conn, "gold_price")
    latest = {"GOLD": m_start} if m_start else {}
    flog = FailureLogger("gold_price", db_conn=conn)
    
    s = resolve_start_cached("GOLD", latest, start, DATASET_START_DATES["gold_price"], force)
    if not s: 
        logger.info("  [gold_price] 已是最新。")
        _write_fetch_log(conn, "gold_price", "GOLD", "skipped")
        return

    start_ts = time.time()
    try:
        data = finmind_get("GoldPrice", {"start_date": s, "end_date": end}, delay)
        duration = int((time.time() - start_ts) * 1000)
        
        if data:
            rows_full = [map_gold_price(r) for r in data]
            rows_full = dedup_rows(rows_full, (0, 1))
            rows_for_commit = [(r[0], r[2]) for r in rows_full]

            UPSERT_GOLD = (
                "INSERT INTO gold_price (date, price) VALUES %s "
                "ON CONFLICT (date) DO UPDATE SET price = EXCLUDED.price"
            )
            res = commit_per_day(
                conn, UPSERT_GOLD, rows_for_commit,
                "(%s::date, %s::numeric)",
                date_index=0, label_prefix="gold_price", failure_logger=flog,
            )
            n = sum(res.values())
            logger.info(f"  [gold_price] 寫入 {n} 筆")
            _write_fetch_log(conn, "gold_price", "GOLD", "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration)
        else:
            _write_fetch_log(conn, "gold_price", "GOLD", "no_new_data", duration_ms=duration)
    except Exception as e:
        duration = int((time.time() - start_ts) * 1000)
        flog.record(stock_id="GOLD", error=str(e))
        _write_fetch_log(conn, "gold_price", "GOLD", "failed", duration_ms=duration, error_message=str(e))
        
    flog.summary()
    logger.info("=== [gold_price] 完成 ===\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["us_stock_price", "crude_oil_prices", "gold_price", "all"], default=["all"])
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["us_stock_price", "crude_oil_prices", "gold_price"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        if "us_stock_price" in tables: fetch_us_stock_price(conn, args.start, args.end, args.delay, args.force, args.ids)
        if "crude_oil_prices" in tables: fetch_crude_oil_prices(conn, args.start, args.end, args.delay, args.force)
        if "gold_price" in tables: fetch_gold_price(conn, args.start, args.end, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
