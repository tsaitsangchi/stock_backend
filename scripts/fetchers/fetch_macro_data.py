import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_macro_data.py  v2.2
抓取宏觀經濟與產業特徵資料：
  - interest_rate  ← InterestRate (央行利率)
  - exchange_rate  ← ExchangeRate (匯率)
  - bond_yield     ← GovernmentBondsYield (公債殖利率)

v2.2 改進：
  · 導入 safe_commit_rows() 與 dump_failures()。
  · 強化 atomicity：按國家、幣別、債券 ID 立即 commit。
  · 失敗清單寫入 outputs/{table}_failed_{date}.json。
  · 確保 DDL 執行後立即 commit。
"""

import argparse
import json
import logging
import time
from datetime import date, timedelta, datetime

import psycopg2
import psycopg2.extras

from core.finmind_client import finmind_get, wait_until_next_hour  # noqa: F401
from core.db_utils import (
    get_db_conn,
    ensure_ddl as base_ensure_ddl,
    bulk_upsert,
    safe_float,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_ddl(conn):
    with conn.cursor() as cur:
        # bond_yield
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bond_yield (
                date DATE,
                bond_id VARCHAR(50),
                value NUMERIC(10,4),
                PRIMARY KEY (date, bond_id)
            );
            CREATE INDEX IF NOT EXISTS idx_bond_yield_id ON bond_yield (bond_id);
        """)
        # interest_rate (ensure exists)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interest_rate (
                date DATE,
                country VARCHAR(50),
                full_country_name VARCHAR(100),
                interest_rate NUMERIC(20,2),
                PRIMARY KEY (date, country)
            );
        """)
        # exchange_rate (ensure exists)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exchange_rate (
                date DATE,
                currency VARCHAR(50),
                cash_buy NUMERIC(20,4),
                cash_sell NUMERIC(20,4),
                spot_buy NUMERIC(20,4),
                spot_sell NUMERIC(20,4),
                PRIMARY KEY (date, currency)
            );
        """)
    conn.commit()


# ──────────────────────────────────────────────
# 逐支 commit 工具函式
# ──────────────────────────────────────────────
def safe_commit_rows(conn, upsert_sql: str, rows: list, template: str,
                      label: str = "") -> int:
    if not rows:
        return 0
    try:
        n = bulk_upsert(conn, upsert_sql, rows, template)
        conn.commit()
        return n
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f"  [{label}] 寫入失敗，已 rollback：{e}")
        return 0


def dump_failures(table: str, failures: list) -> None:
    if not failures:
        return
    out = OUTPUT_DIR / f"{table}_failed_{date.today().strftime('%Y%m%d')}.json"
    try:
        out.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"  失敗清單已寫入：{out}（{len(failures)} 筆）")
    except Exception as e:
        logger.warning(f"  寫入失敗清單時發生錯誤：{e}")

def get_latest_date(conn, table: str, id_col: str, data_id: str):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MAX(date) FROM {table} WHERE {id_col} = %s",
            (data_id,)
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
        return None

def resolve_start(conn, table: str, id_col: str, data_id: str, global_start: str, force: bool):
    if force:
        return global_start

    latest = get_latest_date(conn, table, id_col, data_id)
    if latest is None:
        return global_start

    next_day = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    if next_day > date.today().strftime("%Y-%m-%d"):
        return None
    return max(next_day, global_start)

def fetch_interest_rate(conn, start_date, end_date, force=False):
    logger.info("=== 抓取 InterestRate ===")
    countries = ["FED", "BOJ", "ECB", "PBOC"]
    total = 0
    failures = []
    upsert_sql = """
        INSERT INTO interest_rate (date, country, full_country_name, interest_rate)
        VALUES %s ON CONFLICT (date, country) DO UPDATE SET interest_rate = EXCLUDED.interest_rate
    """
    for country in countries:
        try:
            actual_start = resolve_start(conn, "interest_rate", "country", country, start_date, force)
            if not actual_start:
                logger.info(f"  [{country}] 已是最新，跳過")
                continue

            data = finmind_get("InterestRate", {"data_id": country, "start_date": actual_start, "end_date": end_date}, raise_on_error=True)
            if data:
                rows = [(r["date"], r["country"], r.get("full_country_name"), r["interest_rate"]) for r in data]
                n = safe_commit_rows(conn, upsert_sql, rows, "(%s, %s, %s, %s::numeric)", label=f"interest_rate/{country}")
                total += n
                logger.info(f"  [{country}] {actual_start} ~ {end_date} 寫入 {n} 筆")
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            failures.append({"country": country, "error": str(e)})
            logger.error(f"  [interest_rate/{country}] 失敗：{e}")

    dump_failures("interest_rate", failures)
    return total

def fetch_exchange_rate(conn, start_date, end_date, force=False):
    logger.info("=== 抓取 ExchangeRate ===")
    currencies = ["USD", "JPY", "EUR"]
    total = 0
    failures = []
    upsert_sql = """
        INSERT INTO exchange_rate (date, currency, cash_buy, cash_sell, spot_buy, spot_sell)
        VALUES %s ON CONFLICT (date, currency) DO UPDATE SET
            spot_buy = EXCLUDED.spot_buy,
            spot_sell = EXCLUDED.spot_sell
    """
    for curr in currencies:
        try:
            actual_start = resolve_start(conn, "exchange_rate", "currency", curr, start_date, force)
            if not actual_start:
                logger.info(f"  [{curr}] 已是最新，跳過")
                continue

            data = finmind_get("TaiwanExchangeRate", {"data_id": curr, "start_date": actual_start, "end_date": end_date}, raise_on_error=True)
            if data:
                rows = [(r["date"], r["currency"], r.get("cash_buy"), r.get("cash_sell"), r.get("spot_buy"), r.get("spot_sell")) for r in data]
                n = safe_commit_rows(conn, upsert_sql, rows, "(%s, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric)", label=f"exchange_rate/{curr}")
                total += n
                logger.info(f"  [{curr}] {actual_start} ~ {end_date} 寫入 {n} 筆")
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            failures.append({"currency": curr, "error": str(e)})
            logger.error(f"  [exchange_rate/{curr}] 失敗：{e}")

    dump_failures("exchange_rate", failures)
    return total

def fetch_bond_yield(conn, start_date, end_date, force=False):
    logger.info("=== 抓取 GovernmentBondsYield ===")
    # US 10Y, US 2Y
    bonds = ["United States 10-Year", "United States 2-Year"]
    total = 0
    failures = []
    upsert_sql = """
        INSERT INTO bond_yield (date, bond_id, value)
        VALUES %s ON CONFLICT (date, bond_id) DO UPDATE SET value = EXCLUDED.value
    """
    for bid in bonds:
        try:
            short_id = "US10Y" if "10-Year" in bid else "US2Y"
            actual_start = resolve_start(conn, "bond_yield", "bond_id", short_id, start_date, force)
            if not actual_start:
                logger.info(f"  [{short_id}] 已是最新，跳過")
                continue

            data = finmind_get("GovernmentBondsYield", {"data_id": bid, "start_date": actual_start, "end_date": end_date}, raise_on_error=True)
            if data:
                rows = [(r["date"], short_id, r["value"]) for r in data]
                n = safe_commit_rows(conn, upsert_sql, rows, "(%s, %s, %s::numeric)", label=f"bond_yield/{short_id}")
                total += n
                logger.info(f"  [{short_id}] {actual_start} ~ {end_date} 寫入 {n} 筆")
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            failures.append({"bond_id": bid, "error": str(e)})
            logger.error(f"  [bond_yield/{bid}] 失敗：{e}")

    dump_failures("bond_yield", failures)
    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--force", action="store_true", help="強制重抓")
    args = parser.parse_args()

    try:
        conn = get_db_conn()
        ensure_ddl(conn)

        fetch_interest_rate(conn, args.start, args.end, args.force)
        fetch_exchange_rate(conn, args.start, args.end, args.force)
        fetch_bond_yield(conn, args.start, args.end, args.force)
    except Exception as e:
        logger.error(f"主程序錯誤：{e}")
    finally:
        conn.close()
    logger.info("Macro data 抓取完成")

if __name__ == "__main__":
    main()
