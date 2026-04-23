"""
fetch_international_data.py  v1.0
從 FinMind API 抓取國際影響資料並寫入 PostgreSQL：
  - us_stock_price  ← USStockPrice   (Free, 逐支股票)
  - crude_oil_prices← CrudeOilPrices (Free, WTI + Brent 兩次請求)
  - gold_price      ← GoldPrice      (Free, 不需 data_id)

執行範例：
    # 增量更新（全部三張表）
    python fetch_international_data.py

    # 只抓美股
    python fetch_international_data.py --tables us_stock_price

    # 指定日期區間
    python fetch_international_data.py --start 2023-01-01 --end 2024-12-31

    # 強制重抓（忽略 DB 已有資料）
    python fetch_international_data.py --force

    # 只抓原油 + 黃金（不含美股，較快）
    python fetch_international_data.py --tables crude_oil_prices gold_price

注意事項：
  - us_stock_price 採逐支請求模式，需先有 USStockInfo 資料（程式會自動抓取）。
  - crude_oil_prices 固定抓取 WTI、Brent 兩種，共 2 次 API。
  - gold_price 不需 data_id，單次請求即可取得全部資料。
  - 預設請求間隔 1.2 秒，可用 --delay 調整。
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime

import psycopg2
import psycopg2.extras
import requests
import pandas as pd

# ======================
# 設定 logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ======================
# FinMind API 設定
# ======================
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ"
    ".muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"
)

# ======================
# PostgreSQL 連線設定
# ======================
DB_CONFIG = {
    "dbname": "stock",
    "user": "stock",
    "password": "stock",
    "host": "172.31.122.166",
    "port": "5432",
}

# ======================
# 各資料集最早可用日期
# ======================
DATASET_START_DATES = {
    "us_stock_price":   "1962-01-02",   # USStockPrice 最早可追溯
    "crude_oil_prices": "1987-05-20",   # CrudeOilPrices
    "gold_price":       "1968-04-01",   # GoldPrice
}

# 原油固定抓取的品種
CRUDE_OIL_IDS = ["WTI", "Brent"]

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1962-01-02"

# ======================
# DDL
# ======================
DDL_US_STOCK_PRICE = """
CREATE TABLE IF NOT EXISTS us_stock_price (
    date       DATE,
    stock_id   VARCHAR(50),
    adj_close  NUMERIC(20,4),
    close      NUMERIC(20,4),
    high       NUMERIC(20,4),
    low        NUMERIC(20,4),
    open       NUMERIC(20,4),
    volume     BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_us_stock_price_stock_id ON us_stock_price (stock_id);
"""

DDL_CRUDE_OIL_PRICES = """
CREATE TABLE IF NOT EXISTS crude_oil_prices (
    date   DATE,
    name   VARCHAR(50),
    price  NUMERIC(20,4),
    PRIMARY KEY (date, name)
);
CREATE INDEX IF NOT EXISTS idx_crude_oil_prices_name ON crude_oil_prices (name);
"""

DDL_GOLD_PRICE = """
CREATE TABLE IF NOT EXISTS gold_price (
    date   DATE,
    price  NUMERIC(20,4),
    PRIMARY KEY (date)
);
"""

# ======================
# Upsert SQL
# ======================
UPSERT_US_STOCK_PRICE = """
INSERT INTO us_stock_price
    (date, stock_id, adj_close, close, high, low, open, volume)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    adj_close = EXCLUDED.adj_close,
    close     = EXCLUDED.close,
    high      = EXCLUDED.high,
    low       = EXCLUDED.low,
    open      = EXCLUDED.open,
    volume    = EXCLUDED.volume;
"""

UPSERT_CRUDE_OIL_PRICES = """
INSERT INTO crude_oil_prices (date, name, price)
VALUES %s
ON CONFLICT (date, name) DO UPDATE SET
    price = EXCLUDED.price;
"""

UPSERT_GOLD_PRICE = """
INSERT INTO gold_price (date, price)
VALUES %s
ON CONFLICT (date) DO UPDATE SET
    price = EXCLUDED.price;
"""


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────
def safe_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NONE", "NAN", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def safe_int(val):
    f = safe_float(val)
    return int(f) if f is not None else None


def wait_until_next_hour():
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


def finmind_get(dataset: str, params: dict, delay: float) -> list:
    """
    通用 FinMind API 請求（含重試與 402 限流處理）。
    """
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    base_params = {"dataset": dataset, **params}

    while True:
        for attempt in range(1, 4):
            try:
                resp = requests.get(
                    FINMIND_API_URL, headers=headers, params=base_params, timeout=120
                )
                if resp.status_code == 402:
                    wait_until_next_hour()
                    break
                resp.raise_for_status()
                payload = resp.json()
                status = payload.get("status")
                if status == 402:
                    wait_until_next_hour()
                    break
                if status != 200:
                    msg = payload.get("msg", "")
                    logger.warning(
                        f"[{dataset}] 非預期 status={status}, msg={msg}，跳過"
                    )
                    return []
                time.sleep(delay)
                return payload.get("data", [])

            except requests.HTTPError as http_err:
                code = http_err.response.status_code if http_err.response is not None else 0
                if code == 402:
                    wait_until_next_hour()
                    break
                else:
                    logger.warning(f"[{dataset}] HTTP {code} 錯誤：{http_err}")
                    if attempt < 3:
                        time.sleep(delay * 3)
                    else:
                        logger.error(f"[{dataset}] 重試 3 次均失敗，跳過此次請求")
                        return []
            except Exception as exc:
                logger.warning(f"[{dataset}] 第 {attempt} 次請求失敗：{exc}")
                if attempt < 3:
                    time.sleep(delay * 3)
                else:
                    logger.error(f"[{dataset}] 重試 3 次均失敗，跳過此次請求")
                    return []
        else:
            break


# ──────────────────────────────────────────────
# DB 工具
# ──────────────────────────────────────────────
def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_ddl(conn, *ddls):
    with conn.cursor() as cur:
        for ddl in ddls:
            cur.execute(ddl)
    conn.commit()


def bulk_upsert(conn, upsert_sql: str, rows: list, template: str, page_size: int = 2000):
    if not rows:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, upsert_sql, rows, template=template, page_size=page_size
        )
    conn.commit()


def get_latest_date(conn, table: str, date_col: str = "date", key_col: str = None) -> dict | str | None:
    """
    取得資料表最新日期。
    - key_col 為 None → 回傳單一字串（gold_price 無 key）
    - key_col 有值    → 回傳 dict { key_value: "YYYY-MM-DD" }（us_stock_price, crude_oil_prices）
    """
    with conn.cursor() as cur:
        if key_col:
            cur.execute(
                f"SELECT {key_col}, MAX({date_col}) FROM {table} GROUP BY {key_col}"
            )
            return {
                row[0]: row[1].strftime("%Y-%m-%d")
                for row in cur.fetchall()
                if row[1] is not None
            }
        else:
            cur.execute(f"SELECT MAX({date_col}) FROM {table}")
            row = cur.fetchone()
            return row[0].strftime("%Y-%m-%d") if row and row[0] else None


def resolve_start(latest: str | None, global_start: str, dataset_key: str, force: bool) -> str:
    """
    依 DB 最新日期決定起始日（單一時間序列用，如 gold_price）。
    """
    earliest = DATASET_START_DATES[dataset_key]
    effective_start = max(global_start, earliest)
    if force or latest is None:
        return effective_start
    next_day = (
        datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    return max(next_day, earliest)


def resolve_start_by_key(latest_dict: dict, key: str, global_start: str, dataset_key: str, force: bool) -> str | None:
    """
    依 DB 最新日期決定起始日（有 key 的序列，如 stock_id 或 crude oil 品種）。
    回傳 None 表示已是最新，不需抓取。
    """
    earliest = DATASET_START_DATES[dataset_key]
    effective_start = max(global_start, earliest)
    if force:
        return effective_start
    latest = latest_dict.get(key)
    if latest is None:
        return effective_start
    next_day = (
        datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    if next_day > DEFAULT_END:
        return None  # 已是最新
    return max(next_day, earliest)


# ──────────────────────────────────────────────
# Row mappers
# ──────────────────────────────────────────────
def map_us_stock_row(r: dict) -> tuple:
    """
    USStockPrice API 欄位（大寫）→ DB 欄位（小寫）
    API: date, stock_id, Adj_Close, Close, High, Low, Open, Volume
    """
    return (
        r["date"],
        r["stock_id"],
        safe_float(r.get("Adj_Close")),
        safe_float(r.get("Close")),
        safe_float(r.get("High")),
        safe_float(r.get("Low")),
        safe_float(r.get("Open")),
        safe_int(r.get("Volume")),
    )


def map_crude_oil_row(r: dict) -> tuple:
    """
    CrudeOilPrices API 欄位 → DB
    API: date, name, price
    """
    return (
        r["date"],
        r["name"],
        safe_float(r.get("price")),
    )


def map_gold_price_row(r: dict) -> tuple:
    """
    GoldPrice API 欄位 → DB
    API: date, Price（注意大寫 P）
    """
    return (
        r["date"],
        safe_float(r.get("Price")),
    )


# ──────────────────────────────────────────────
# 取得所有美股 stock_id（USStockInfo）
# ──────────────────────────────────────────────
def get_us_stock_ids(delay: float) -> list:
    """
    從 FinMind USStockInfo 取得所有美股 stock_id 清單。
    不需起始/結束日期，一次回傳全部。
    """
    logger.info("[USStockInfo] 正在取得美股清單…")
    data = finmind_get("USStockInfo", {}, delay)
    if not data:
        logger.error("[USStockInfo] 無法取得美股清單，請確認 Token 與網路狀態")
        return []
    ids = sorted({r["stock_id"] for r in data if r.get("stock_id")})
    logger.info(f"[USStockInfo] 共取得 {len(ids)} 支美股")
    return ids


# ──────────────────────────────────────────────
# 主要抓取函式
# ──────────────────────────────────────────────
def fetch_us_stock_price(
    conn, start_date: str, end_date: str, delay: float, force: bool, tickers: list = None
):
    """
    抓取美股股價（USStockPrice）。若 tickers 為 None 則抓取全市場。
    """
    logger.info("=== [us_stock_price] 開始抓取 ===")

    if tickers:
        stock_ids = tickers
        logger.info(f"  指定抓取 {len(stock_ids)} 支股票：{stock_ids}")
    else:
        stock_ids = get_us_stock_ids(delay)
        if not stock_ids:
            return

    # 批次載入 DB 最新日期
    latest_dates = get_latest_date(conn, "us_stock_price", key_col="stock_id")

    total_rows = 0
    skipped = 0
    template = "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)"

    for i, sid in enumerate(stock_ids, 1):
        s = resolve_start_by_key(latest_dates, sid, start_date, "us_stock_price", force)
        if s is None:
            skipped += 1
            continue

        data = finmind_get(
            "USStockPrice",
            {"data_id": sid, "start_date": s, "end_date": end_date},
            delay,
        )
        if data:
            rows = [map_us_stock_row(r) for r in data]
            # 依 (date, stock_id) 去重
            df_dedup = pd.DataFrame(rows).drop_duplicates(subset=[0, 1], keep="last")
            rows = [tuple(x) for x in df_dedup.values]
            
            bulk_upsert(conn, UPSERT_US_STOCK_PRICE, rows, template)
            total_rows += len(rows)

        if i % 100 == 0:
            logger.info(f"  進度：{i}/{len(stock_ids)}  已略過（最新）：{skipped}")

    logger.info(
        f"=== [us_stock_price] 完成  寫入：{total_rows} 筆  略過：{skipped} 支 ==="
    )


def fetch_crude_oil_prices(
    conn, start_date: str, end_date: str, delay: float, force: bool
):
    """
    抓取原油價格（CrudeOilPrices）：WTI + Brent，共 2 次 API。
    """
    logger.info("=== [crude_oil_prices] 開始抓取 ===")

    # 批次載入 DB 最新日期（以 name 為 key）
    latest_dates = get_latest_date(conn, "crude_oil_prices", key_col="name")

    total_rows = 0
    template = "(%s::date,%s,%s::numeric)"

    for oil_id in CRUDE_OIL_IDS:
        s = resolve_start_by_key(latest_dates, oil_id, start_date, "crude_oil_prices", force)
        if s is None:
            logger.info(f"  [{oil_id}] 已是最新，略過")
            continue

        logger.info(f"  [{oil_id}] 抓取 {s} ~ {end_date}")
        data = finmind_get(
            "CrudeOilPrices",
            {"data_id": oil_id, "start_date": s, "end_date": end_date},
            delay,
        )
        if data:
            rows = [map_crude_oil_row(r) for r in data]
            bulk_upsert(conn, UPSERT_CRUDE_OIL_PRICES, rows, template)
            total_rows += len(rows)
            logger.info(f"  [{oil_id}] 寫入 {len(rows)} 筆")

    logger.info(f"=== [crude_oil_prices] 完成  寫入：{total_rows} 筆 ===")


def fetch_gold_price(
    conn, start_date: str, end_date: str, delay: float, force: bool
):
    """
    抓取黃金價格（GoldPrice），無 data_id，單次請求。
    """
    logger.info("=== [gold_price] 開始抓取 ===")

    latest = get_latest_date(conn, "gold_price")
    s = resolve_start(latest, start_date, "gold_price", force)

    if not force and latest and s > DEFAULT_END:
        logger.info("  [gold_price] 已是最新，略過")
        return

    logger.info(f"  [gold_price] 抓取 {s} ~ {end_date}")
    data = finmind_get(
        "GoldPrice",
        {"start_date": s, "end_date": end_date},
        delay,
    )
    if data:
        rows = [map_gold_price_row(r) for r in data]
        # 依 date 去重
        df_dedup = pd.DataFrame(rows).drop_duplicates(subset=[0], keep="last")
        rows = [tuple(x) for x in df_dedup.values]
        
        template = "(%s::date,%s::numeric)"
        bulk_upsert(conn, UPSERT_GOLD_PRICE, rows, template)
        logger.info(f"  [gold_price] 寫入 {len(rows)} 筆")

    logger.info("=== [gold_price] 完成 ===")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
ALL_TABLES = ["us_stock_price", "crude_oil_prices", "gold_price"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FinMind 國際影響資料抓取工具 v1.0"
    )
    parser.add_argument(
        "--tables", nargs="+",
        choices=ALL_TABLES + ["all"],
        default=["all"],
        help="要抓取的資料表（預設 all）",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help="開始日期 YYYY-MM-DD（預設依各資料集最早日期）",
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help="結束日期 YYYY-MM-DD（預設今天）",
    )
    parser.add_argument(
        "--delay", type=float, default=1.2,
        help="每次 API 請求後等待秒數（預設 1.2）",
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="指定要抓取的美股代號（如 AAPL MSFT），若未指定則抓取全市場",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重抓：忽略 DB 已有資料，從 --start 重新覆蓋",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tables = ALL_TABLES if "all" in args.tables else args.tables

    logger.info(f"抓取資料表：{tables}")
    logger.info(f"日期區間：{args.start} ~ {args.end}")
    logger.info(f"請求間隔：{args.delay} 秒")
    logger.info(f"執行模式：{'強制重抓' if args.force else '增量模式'}")

    try:
        conn = get_db_conn()
    except psycopg2.OperationalError as e:
        logger.error(f"PostgreSQL 連線失敗：{e}")
        sys.exit(1)

    try:
        # 建立所有需要的資料表（DDL）
        ddls_needed = []
        if "us_stock_price"   in tables: ddls_needed.append(DDL_US_STOCK_PRICE)
        if "crude_oil_prices" in tables: ddls_needed.append(DDL_CRUDE_OIL_PRICES)
        if "gold_price"       in tables: ddls_needed.append(DDL_GOLD_PRICE)
        ensure_ddl(conn, *ddls_needed)

        # 依序執行各資料集
        if "us_stock_price" in tables:
            fetch_us_stock_price(conn, args.start, args.end, args.delay, args.force, tickers=args.tickers)

        if "crude_oil_prices" in tables:
            fetch_crude_oil_prices(conn, args.start, args.end, args.delay, args.force)

        if "gold_price" in tables:
            fetch_gold_price(conn, args.start, args.end, args.delay, args.force)

    except Exception as e:
        logger.error(f"未預期錯誤：{e}")
        raise
    finally:
        conn.close()

    logger.info("全部完成！")


if __name__ == "__main__":
    main()
