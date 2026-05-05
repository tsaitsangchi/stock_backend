from __future__ import annotations
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
fetch_international_data.py v3.0 — 國際影響資料（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：確保美股、原油、黃金每一天資料均獨立 commit。
  ★ 全面整合 FailureLogger：精準捕捉並彙整所有抓取/寫入失敗。
  ★ 標準化處理：對接 core v3.0，提升在全球宏觀數據抓取時的系統韌性。
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
    # 為了套用 commit_per_stock_per_day，我們在 mapper 中補一個 dummy id "GOLD"
    # 但寫入 SQL 時只取 (date, price)，所以返回 (date, "GOLD", price)
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
        if not s: continue
        try:
            data = finmind_get("USStockPrice", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [map_us_stock(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_US_STOCK_PRICE, rows, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)", label_prefix="us_stock", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=sid, error=str(e))
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
        if not s: continue
        try:
            data = finmind_get("CrudeOilPrices", {"data_id": oid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [map_crude_oil(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_CRUDE_OIL_PRICES, rows, "(%s::date,%s,%s::numeric)", label_prefix="crude_oil", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=oid, error=str(e))

    flog.summary()
    logger.info(f"=== [crude_oil_prices] 完成：{total_rows} 筆 ===\n")

def fetch_gold_price(conn, start, end, delay, force):
    logger.info("=== [gold_price] 開始 ===")
    ensure_ddl(conn, DDL_GOLD_PRICE)
    # gold_price 無 key，使用市場層級偵測
    m_start = get_market_safe_start(conn, "gold_price")
    latest = {"GOLD": m_start} if m_start else {}
    flog = FailureLogger("gold_price", db_conn=conn)
    
    s = resolve_start_cached("GOLD", latest, start, DATASET_START_DATES["gold_price"], force)
    if not s: 
        logger.info("  [gold_price] 已是最新。")
        return

    try:
        data = finmind_get("GoldPrice", {"start_date": s, "end_date": end}, delay)
        if data:
            # map_gold_price 回傳 (date, "GOLD", price)；gold_price 表本身只有 (date, price)
            # 因此這裡剝掉中間的 "GOLD" key，並改用 commit_per_day（市場層單值資料的正確語意）
            rows_full = [map_gold_price(r) for r in data]
            rows_full = dedup_rows(rows_full, (0, 1))
            rows_for_commit = [(r[0], r[2]) for r in rows_full]  # (date, price)

            UPSERT_GOLD = (
                "INSERT INTO gold_price (date, price) VALUES %s "
                "ON CONFLICT (date) DO UPDATE SET price = EXCLUDED.price"
            )
            res = commit_per_day(
                conn, UPSERT_GOLD, rows_for_commit,
                "(%s::date, %s::numeric)",
                date_index=0, label_prefix="gold_price", failure_logger=flog,
            )
            logger.info(f"  [gold_price] 寫入 {sum(res.values())} 筆")
    except Exception as e:
        flog.record(stock_id="GOLD", error=str(e))
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
