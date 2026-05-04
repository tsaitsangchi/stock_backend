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
fetch_macro_data.py v3.0 — 總經與產業特徵（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：利率、匯率、債券收益率每一天資料均獨立原子 commit。
  ★ 全面整合 FailureLogger：精準追蹤全球利率與匯率波動數據的抓取狀況。
  ★ 結構規範化：移除本地冗餘工具，全面對接 core v3.0。
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
    dedup_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# DDL & SQL
# ──────────────────────────────────────────────
DDL_MACRO = """
CREATE TABLE IF NOT EXISTS interest_rate (date DATE, country VARCHAR(50), full_country_name VARCHAR(100), interest_rate NUMERIC(20,2), PRIMARY KEY (date, country));
CREATE TABLE IF NOT EXISTS exchange_rate (date DATE, currency VARCHAR(50), cash_buy NUMERIC(20,4), cash_sell NUMERIC(20,4), spot_buy NUMERIC(20,4), spot_sell NUMERIC(20,4), PRIMARY KEY (date, currency));
CREATE TABLE IF NOT EXISTS bond_yield (date DATE, bond_id VARCHAR(50), value NUMERIC(10,4), PRIMARY KEY (date, bond_id));
"""

UPSERT_INTEREST = """INSERT INTO interest_rate (date, country, full_country_name, interest_rate) VALUES %s ON CONFLICT (date, country) DO UPDATE SET interest_rate = EXCLUDED.interest_rate;"""
UPSERT_EXCHANGE = """INSERT INTO exchange_rate (date, currency, cash_buy, cash_sell, spot_buy, spot_sell) VALUES %s ON CONFLICT (date, currency) DO UPDATE SET spot_buy = EXCLUDED.spot_buy;"""
UPSERT_BOND = """INSERT INTO bond_yield (date, bond_id, value) VALUES %s ON CONFLICT (date, bond_id) DO UPDATE SET value = EXCLUDED.value;"""

# ──────────────────────────────────────────────
# Fetchers
# ──────────────────────────────────────────────
def fetch_interest_rate(conn, start, end, delay, force):
    logger.info("=== [interest_rate] 開始 ===")
    latest = get_all_safe_starts(conn, "interest_rate", key_col="country")
    flog = FailureLogger("interest_rate", db_conn=conn)
    total_rows = 0
    for country in ["FED", "BOJ", "ECB", "PBOC"]:
        s = resolve_start_cached(country, latest, start, "2000-01-01", force)
        if not s: continue
        try:
            data = finmind_get("InterestRate", {"data_id": country, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [(r["date"], r["country"], r.get("full_country_name"), r["interest_rate"]) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_INTEREST, rows, "(%s, %s, %s, %s::numeric)", label_prefix="interest", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=country, error=str(e))
    logger.info(f"  [interest_rate] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_exchange_rate(conn, start, end, delay, force):
    logger.info("=== [exchange_rate] 開始 ===")
    latest = get_all_safe_starts(conn, "exchange_rate", key_col="currency")
    flog = FailureLogger("exchange_rate", db_conn=conn)
    total_rows = 0
    for curr in ["USD", "JPY", "EUR"]:
        s = resolve_start_cached(curr, latest, start, "2000-01-01", force)
        if not s: continue
        try:
            data = finmind_get("TaiwanExchangeRate", {"data_id": curr, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [(r["date"], r["currency"], r.get("cash_buy"), r.get("cash_sell"), r.get("spot_buy"), r.get("spot_sell")) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_EXCHANGE, rows, "(%s, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric)", label_prefix="exchange", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=curr, error=str(e))
    logger.info(f"  [exchange_rate] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_bond_yield(conn, start, end, delay, force):
    logger.info("=== [bond_yield] 開始 ===")
    latest = get_all_safe_starts(conn, "bond_yield", key_col="bond_id")
    flog = FailureLogger("bond_yield", db_conn=conn)
    total_rows = 0
    for bid, sid in [("United States 10-Year", "US10Y"), ("United States 2-Year", "US2Y")]:
        s = resolve_start_cached(sid, latest, start, "2000-01-01", force)
        if not s: continue
        try:
            data = finmind_get("GovernmentBondsYield", {"data_id": bid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [(r["date"], sid, r["value"]) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_BOND, rows, "(%s, %s, %s::numeric)", label_prefix="bond", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=sid, error=str(e))
    logger.info(f"  [bond_yield] 總共寫入 {total_rows} 筆")
    flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_MACRO)
        fetch_interest_rate(conn, args.start, args.end, args.delay, args.force)
        fetch_exchange_rate(conn, args.start, args.end, args.delay, args.force)
        fetch_bond_yield(conn, args.start, args.end, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
