import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_international_data.py  v2.2
從 FinMind API 抓取國際影響資料並寫入 PostgreSQL：
  - us_stock_price  ← USStockPrice
  - crude_oil_prices← CrudeOilPrices
  - gold_price      ← GoldPrice

v2.2 改進：
  · 導入 safe_commit_rows() 與 dump_failures()。
  · 強化 atomicity：美股按 ticker commit，原油按品種 commit。
  · 失敗清單寫入 outputs/{table}_failed_{date}.json。
  · 確保 DDL 執行後立即 commit。

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
import time
import json
from datetime import date, timedelta, datetime

import psycopg2
import pandas as pd
from config import INTERNATIONAL_WATCHLIST

from core.finmind_client import finmind_get, wait_until_next_hour  # noqa: F401
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    safe_float,
    safe_int,
)

# ======================
# 設定 logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    next_day_dt = datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    next_day = next_day_dt.strftime("%Y-%m-%d")

    if next_day > DEFAULT_END:
        return None  # 已是最新

    # 週末防護：如果 latest 是週五 (weekday 4)，且今天是週六或週日，則視為已最新
    latest_dt = datetime.strptime(latest, "%Y-%m-%d")
    today_dt = datetime.today()
    if latest_dt.weekday() == 4: # 週五
        if (today_dt - latest_dt).days <= 2:
            return None

    return max(next_day, earliest)


# ──────────────────────────────────────────────
# Row mappers
# ──────────────────────────────────────────────
def map_us_stock_row(r: dict) -> tuple:
    """
    USStockPrice API 欄位（大寫）→ DB 欄位（小寫）
    API: date, stock_id, Adj_Close, Close, High, Low, Open, Volume
    """
    # 標準化日期為 YYYY-MM-DD
    std_date = datetime.strptime(str(r["date"]), "%Y-%m-%d %H:%M:%S" if " " in str(r["date"]) else "%Y-%m-%d").strftime("%Y-%m-%d")
    return (
        std_date,
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
    # 標準化日期為 YYYY-MM-DD
    std_date = datetime.strptime(str(r["date"]), "%Y-%m-%d %H:%M:%S" if " " in str(r["date"]) else "%Y-%m-%d").strftime("%Y-%m-%d")
    return (
        std_date,
        r["name"],
        safe_float(r.get("price")),
    )


def map_gold_price_row(r: dict) -> tuple:
    """
    GoldPrice API 欄位 → DB
    API: date, Price（注意大寫 P）
    """
    # 標準化日期為 YYYY-MM-DD
    std_date = datetime.strptime(str(r["date"]), "%Y-%m-%d %H:%M:%S" if " " in str(r["date"]) else "%Y-%m-%d").strftime("%Y-%m-%d")
    return (
        std_date,
        safe_float(r.get("Price")),
    )
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


# ──────────────────────────────────────────────
# 取得所有美股 stock_id（USStockInfo）
# ──────────────────────────────────────────────
def get_us_stock_ids(delay: float) -> list:
    """
    從 FinMind USStockInfo 取得所有美股 stock_id 清單。
    不需起始/結束日期，一次回傳全部。
    """
    logger.info("[USStockInfo] 正在取得美股清單…")
    data = finmind_get("USStockInfo", {}, delay, raise_on_error=True)
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
        # 預設改用 config.py 的白名單，避免全市場抓取耗盡額度
        stock_ids = INTERNATIONAL_WATCHLIST
        logger.info(f"  使用預設白名單抓取 {len(stock_ids)} 支關鍵標的")

    # 批次載入 DB 最新日期
    latest_dates = get_latest_date(conn, "us_stock_price", key_col="stock_id")

    total_rows = 0
    skipped = 0
    failures = []
    template = "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)"

    for i, sid in enumerate(stock_ids, 1):
        try:
            s = resolve_start_by_key(latest_dates, sid, start_date, "us_stock_price", force)
            if s is None:
                skipped += 1
                continue

            data = finmind_get(
                "USStockPrice",
                {"data_id": sid, "start_date": s, "end_date": end_date},
                delay,
                raise_on_error=True
            )
            if data:
                # 依 (date, stock_id) 去重
                seen = {}
                for r in data:
                    try:
                        row = map_us_stock_row(r)
                        key = (row[0], row[1])
                        seen[key] = row
                    except Exception:
                        continue
                rows = list(seen.values())

                n = safe_commit_rows(conn, UPSERT_US_STOCK_PRICE, rows, template, label=f"us_stock/{sid}")
                total_rows += n
            
            if i % 50 == 0:
                logger.info(f"  進度：{i}/{len(stock_ids)}  已略過（最新）：{skipped}")
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            failures.append({"stock_id": sid, "error": str(e)})
            logger.error(f"  [us_stock/{sid}] 失敗：{e}")

    dump_failures("us_stock_price", failures)
    logger.info(
        f"=== [us_stock_price] 完成  寫入：{total_rows} 筆  略過：{skipped} 支  失敗：{len(failures)} ==="
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
    failures = []
    template = "(%s::date,%s,%s::numeric)"

    for oil_id in CRUDE_OIL_IDS:
        try:
            s = resolve_start_by_key(latest_dates, oil_id, start_date, "crude_oil_prices", force)
            if s is None:
                logger.info(f"  [{oil_id}] 已是最新，略過")
                continue

            logger.info(f"  [{oil_id}] 抓取 {s} ~ {end_date}")
            data = finmind_get(
                "CrudeOilPrices",
                {"data_id": oil_id, "start_date": s, "end_date": end_date},
                delay,
                raise_on_error=True
            )
            if data:
                # 依 (date, name) 去重
                seen = {}
                for r in data:
                    try:
                        row = map_crude_oil_row(r)
                        key = (row[0], row[1])
                        seen[key] = row
                    except Exception:
                        continue
                rows = list(seen.values())

                n = safe_commit_rows(conn, UPSERT_CRUDE_OIL_PRICES, rows, template, label=f"crude_oil/{oil_id}")
                total_rows += n
                logger.info(f"  [{oil_id}] 寫入 {n} 筆")
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            failures.append({"oil_id": oil_id, "error": str(e)})
            logger.error(f"  [crude_oil/{oil_id}] 失敗：{e}")

    dump_failures("crude_oil_prices", failures)
    logger.info(f"=== [crude_oil_prices] 完成  寫入：{total_rows} 筆  失敗：{len(failures)} ===")


def fetch_gold_price(
    conn, start_date: str, end_date: str, delay: float, force: bool
):
    """
    抓取黃金價格（GoldPrice），無 data_id，單次請求。
    """
    try:
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
            raise_on_error=True
        )
        if data:
            # 依 date 去重
            seen = {}
            for r in data:
                try:
                    row = map_gold_price_row(r)
                    key = row[0]
                    seen[key] = row
                except Exception:
                    continue
            rows = list(seen.values())

            template = "(%s::date,%s::numeric)"
            n = safe_commit_rows(conn, UPSERT_GOLD_PRICE, rows, template, label="gold_price")
            logger.info(f"  [gold_price] 寫入 {n} 筆")

        logger.info("=== [gold_price] 完成 ===")
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        logger.error(f"  [gold_price] 失敗：{e}")
        dump_failures("gold_price", [{"error": str(e)}])


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
        conn.commit()

        # 依序執行各資料集
        if "us_stock_price" in tables:
            fetch_us_stock_price(conn, args.start, args.end, args.delay, args.force, tickers=args.tickers)

        if "crude_oil_prices" in tables:
            fetch_crude_oil_prices(conn, args.start, args.end, args.delay, args.force)

        if "gold_price" in tables:
            fetch_gold_price(conn, args.start, args.end, args.delay, args.force)

    except Exception as e:
        logger.error(f"主程序發生錯誤：{e}")
        # 不再強制 raise，讓 parallel_fetch 捕捉狀態
    finally:
        conn.close()

    logger.info("全部完成！")


if __name__ == "__main__":
    main()
