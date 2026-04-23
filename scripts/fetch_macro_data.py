"""
fetch_macro_data.py
抓取宏觀經濟與產業特徵資料：
  - interest_rate  ← InterestRate (央行利率)
  - exchange_rate  ← ExchangeRate (匯率)
  - bond_yield     ← GovernmentBondsYield (公債殖利率)
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime
import psycopg2
import psycopg2.extras
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ"
    ".muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"
)

DB_CONFIG = {
    "dbname": "stock",
    "user": "stock",
    "password": "stock",
    "host": "172.31.122.166",
    "port": "5432",
}

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def wait_until_next_hour():
    """
    當遇到 402 (Payment Required) 錯誤時，代表 API 用量達上限。
    通常 FinMind 會在整點重置配額，因此等待至下一整點。
    """
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_sec = (next_hour - now).total_seconds() + 65
    logger.warning(
        f"API 用量達上限（402），等待至下一整點重置。"
        f"目前時間：{now.strftime('%H:%M:%S')}，"
        f"預計恢復：{next_hour.strftime('%H:%M:%S')}，"
        f"等待 {wait_sec:.0f} 秒…"
    )
    time.sleep(wait_sec)
    logger.info("等待結束，恢復請求。")

def finmind_get(dataset: str, params: dict, delay: float = 1.2) -> list:
    """
    通用 FinMind API 請求，包含自動處理 402 錯誤的重試機制。
    """
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}

    while True:
        try:
            resp = requests.get(FINMIND_API_URL, headers=headers, params=req_params, timeout=60)

            # 檢查 HTTP 狀態碼
            if resp.status_code == 402:
                wait_until_next_hour()
                continue

            resp.raise_for_status()
            payload = resp.json()

            # 檢查 API 業務邏輯狀態碼
            status = payload.get("status")
            if status == 402:
                wait_until_next_hour()
                continue

            if status != 200:
                logger.warning(f"[{dataset}] status={status}, msg={payload.get('msg')}")
                return []

            time.sleep(delay)
            return payload.get("data", [])

        except requests.exceptions.RequestException as e:
            # 處理可能被 raise_for_status 拋出的 402
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 402:
                wait_until_next_hour()
                continue
            logger.error(f"[{dataset}] 請求失敗: {e}")
            return []

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

def fetch_interest_rate(conn, start_date, end_date):
    logger.info("=== 抓取 InterestRate ===")
    countries = ["FED", "BOJ", "ECB", "PBOC"]
    total = 0
    for country in countries:
        data = finmind_get("InterestRate", {"data_id": country, "start_date": start_date, "end_date": end_date})
        if data:
            rows = [(r["date"], r["country"], r.get("full_country_name"), r["interest_rate"]) for r in data]
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO interest_rate (date, country, full_country_name, interest_rate)
                    VALUES %s ON CONFLICT (date, country) DO UPDATE SET interest_rate = EXCLUDED.interest_rate
                """, rows)
            conn.commit()
            total += len(rows)
            logger.info(f"  [{country}] 寫入 {len(rows)} 筆")
    return total

def fetch_exchange_rate(conn, start_date, end_date):
    logger.info("=== 抓取 ExchangeRate ===")
    currencies = ["USD", "JPY", "EUR"]
    total = 0
    for curr in currencies:
        data = finmind_get("ExchangeRate", {"data_id": curr, "start_date": start_date, "end_date": end_date})
        if data:
            rows = [(r["date"], r["currency"], r.get("cash_buy"), r.get("cash_sell"), r.get("spot_buy"), r.get("spot_sell")) for r in data]
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO exchange_rate (date, currency, cash_buy, cash_sell, spot_buy, spot_sell)
                    VALUES %s ON CONFLICT (date, currency) DO UPDATE SET 
                        spot_buy = EXCLUDED.spot_buy,
                        spot_sell = EXCLUDED.spot_sell
                """, rows)
            conn.commit()
            total += len(rows)
            logger.info(f"  [{curr}] 寫入 {len(rows)} 筆")
    return total

def fetch_bond_yield(conn, start_date, end_date):
    logger.info("=== 抓取 GovernmentBondsYield ===")
    # US 10Y, US 2Y
    bonds = ["United States 10-Year", "United States 2-Year"]
    total = 0
    for bid in bonds:
        data = finmind_get("GovernmentBondsYield", {"data_id": bid, "start_date": start_date, "end_date": end_date})
        if data:
            # Map "United States 10-Year" -> "US10Y" for easier usage
            short_id = "US10Y" if "10-Year" in bid else "US2Y"
            rows = [(r["date"], short_id, r["value"]) for r in data]
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO bond_yield (date, bond_id, value)
                    VALUES %s ON CONFLICT (date, bond_id) DO UPDATE SET value = EXCLUDED.value
                """, rows)
            conn.commit()
            total += len(rows)
            logger.info(f"  [{short_id}] 寫入 {len(rows)} 筆")
    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    conn = get_db_conn()
    ensure_ddl(conn)
    
    fetch_interest_rate(conn, args.start, args.end)
    fetch_exchange_rate(conn, args.start, args.end)
    fetch_bond_yield(conn, args.start, args.end)
    
    conn.close()
    logger.info("Macro data 抓取完成")

if __name__ == "__main__":
    main()
