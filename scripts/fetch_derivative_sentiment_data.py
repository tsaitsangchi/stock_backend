"""
fetch_derivative_sentiment_data.py — 衍生品與情緒指標資料抓取
============================================================
Backer/Sponsor 方案資料集：

  1. options_large_oi    ← TaiwanOptionOpenInterestLargeTraders (選擇權大額交易人)
     Signal: Put/Call 大戶持倉 → 聰明錢偏好
     Columns: date, option_id, put_call, contract_type, market_open_interest, buy_top5_trader_open_interest, ...

  2. fear_greed_index    ← CnnFearGreedIndex (恐懼與貪婪指數)
     Signal: 市場情緒溫度計 → 極端值作為反向指標
     Columns: date, fear_greed, fear_greed_emotion

  3. block_trading       ← TaiwanStockBlockTradingDailyReport (鉅額交易買賣)
     Signal: 大宗交易 → 機構換手信號
     Columns: date, stock_id, securities_trader_id, price, buy, sell, trade_type

執行：
    python fetch_derivative_sentiment_data.py
    python fetch_derivative_sentiment_data.py --tables fear_greed_index block_trading
    python fetch_derivative_sentiment_data.py --force
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import date, datetime, timedelta

import psycopg2

from config import FINMIND_TOKEN, DB_CONFIG
import psycopg2.extras
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
DATASET_START = {
    "options_large_oi":  "2018-01-01",
    "fear_greed_index":  "2011-01-03",
    "block_trading":     "2021-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────

DDL_OPTIONS_LARGE_OI = """
CREATE TABLE IF NOT EXISTS options_oi_large_holders (
    date               DATE,
    option_id          VARCHAR(50),
    put_call           VARCHAR(10),
    contract_type      VARCHAR(50),
    name               VARCHAR(100),
    market_open_interest                  NUMERIC,
    buy_top5_trader_open_interest         NUMERIC,
    buy_top5_trader_open_interest_per     NUMERIC,
    buy_top10_trader_open_interest        NUMERIC,
    buy_top10_trader_open_interest_per    NUMERIC,
    sell_top5_trader_open_interest        NUMERIC,
    sell_top5_trader_open_interest_per    NUMERIC,
    sell_top10_trader_open_interest       NUMERIC,
    sell_top10_trader_open_interest_per   NUMERIC,
    buy_top5_specific_open_interest       NUMERIC,
    buy_top5_specific_open_interest_per   NUMERIC,
    buy_top10_specific_open_interest      NUMERIC,
    buy_top10_specific_open_interest_per  NUMERIC,
    sell_top5_specific_open_interest      NUMERIC,
    sell_top5_specific_open_interest_per  NUMERIC,
    sell_top10_specific_open_interest     NUMERIC,
    sell_top10_specific_open_interest_per NUMERIC,
    PRIMARY KEY (date, option_id, put_call, contract_type)
);
CREATE INDEX IF NOT EXISTS idx_ooi_date ON options_oi_large_holders (date);
"""

DDL_FEAR_GREED_INDEX = """
CREATE TABLE IF NOT EXISTS fear_greed_index (
    date               DATE PRIMARY KEY,
    fear_greed         NUMERIC,
    fear_greed_emotion VARCHAR(50)
);
"""

DDL_BLOCK_TRADING = """
CREATE TABLE IF NOT EXISTS block_trading (
    date                 DATE,
    stock_id             VARCHAR(50),
    securities_trader_id VARCHAR(50),
    securities_trader    VARCHAR(100),
    price                NUMERIC(10,2),
    buy                  NUMERIC,
    sell                 NUMERIC,
    trade_type           VARCHAR(50),
    PRIMARY KEY (date, stock_id, securities_trader_id, price, trade_type)
);
CREATE INDEX IF NOT EXISTS idx_bt_stock_date ON block_trading (stock_id, date);
CREATE INDEX IF NOT EXISTS idx_bt_date ON block_trading (date);
"""

# ─────────────────────────────────────────────
# Upsert SQL
# ─────────────────────────────────────────────

UPSERT_OPTIONS_LARGE_OI = """
INSERT INTO options_oi_large_holders (
    date, option_id, put_call, contract_type, name, market_open_interest,
    buy_top5_trader_open_interest, buy_top5_trader_open_interest_per,
    buy_top10_trader_open_interest, buy_top10_trader_open_interest_per,
    sell_top5_trader_open_interest, sell_top5_trader_open_interest_per,
    sell_top10_trader_open_interest, sell_top10_trader_open_interest_per,
    buy_top5_specific_open_interest, buy_top5_specific_open_interest_per,
    buy_top10_specific_open_interest, buy_top10_specific_open_interest_per,
    sell_top5_specific_open_interest, sell_top5_specific_open_interest_per,
    sell_top10_specific_open_interest, sell_top10_specific_open_interest_per
) VALUES %s
ON CONFLICT (date, option_id, put_call, contract_type) DO UPDATE SET
    name = EXCLUDED.name,
    market_open_interest = EXCLUDED.market_open_interest,
    buy_top5_trader_open_interest = EXCLUDED.buy_top5_trader_open_interest,
    buy_top5_trader_open_interest_per = EXCLUDED.buy_top5_trader_open_interest_per,
    buy_top10_trader_open_interest = EXCLUDED.buy_top10_trader_open_interest,
    buy_top10_trader_open_interest_per = EXCLUDED.buy_top10_trader_open_interest_per,
    sell_top5_trader_open_interest = EXCLUDED.sell_top5_trader_open_interest,
    sell_top5_trader_open_interest_per = EXCLUDED.sell_top5_trader_open_interest_per,
    sell_top10_trader_open_interest = EXCLUDED.sell_top10_trader_open_interest,
    sell_top10_trader_open_interest_per = EXCLUDED.sell_top10_trader_open_interest_per,
    buy_top5_specific_open_interest = EXCLUDED.buy_top5_specific_open_interest,
    buy_top5_specific_open_interest_per = EXCLUDED.buy_top5_specific_open_interest_per,
    buy_top10_specific_open_interest = EXCLUDED.buy_top10_specific_open_interest,
    buy_top10_specific_open_interest_per = EXCLUDED.buy_top10_specific_open_interest_per,
    sell_top5_specific_open_interest = EXCLUDED.sell_top5_specific_open_interest,
    sell_top5_specific_open_interest_per = EXCLUDED.sell_top5_specific_open_interest_per,
    sell_top10_specific_open_interest = EXCLUDED.sell_top10_specific_open_interest,
    sell_top10_specific_open_interest_per = EXCLUDED.sell_top10_specific_open_interest_per;
"""

UPSERT_FEAR_GREED_INDEX = """
INSERT INTO fear_greed_index (date, fear_greed, fear_greed_emotion)
VALUES %s
ON CONFLICT (date) DO UPDATE SET
    fear_greed = EXCLUDED.fear_greed,
    fear_greed_emotion = EXCLUDED.fear_greed_emotion;
"""

UPSERT_BLOCK_TRADING = """
INSERT INTO block_trading (date, stock_id, securities_trader_id, securities_trader, price, buy, sell, trade_type)
VALUES %s
ON CONFLICT (date, stock_id, securities_trader_id, price, trade_type) DO UPDATE SET
    securities_trader = EXCLUDED.securities_trader,
    buy = EXCLUDED.buy,
    sell = EXCLUDED.sell;
"""

# ─────────────────────────────────────────────
# [P0 重構] 工具改用共享模組
# ─────────────────────────────────────────────
from core.finmind_client import (  # noqa: E402,F401
    finmind_get,
    wait_until_next_hour as wait_next_hour,  # 沿用本檔原命名
)
from core.db_utils import (  # noqa: E402,F401
    get_db_conn as get_conn,  # 沿用本檔原命名
    ensure_ddl,
    safe_float,
    safe_int,
)


# ─────────────────────────────────────────────
# ① 選擇權大額交易人未沖銷部位
# ─────────────────────────────────────────────

def fetch_options_large_oi(conn, start: str, end: str, delay: float, force: bool):
    logger.info("\n=== [options_large_oi] 開始 ===")
    ensure_ddl(conn, DDL_OPTIONS_LARGE_OI)

    s = start
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM options_oi_large_holders")
            last = cur.fetchone()[0]
        if last:
            s = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    if s > end:
        logger.info("[options_large_oi] 已是最新，跳過")
        return

    # 按月分批抓
    start_d = datetime.strptime(s, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    total   = 0

    while start_d <= end_d:
        chunk_end = min(start_d + timedelta(days=30), end_d)
        rows = finmind_get(
            "TaiwanOptionOpenInterestLargeTraders",
            {"start_date": start_d.strftime("%Y-%m-%d"),
             "end_date":   chunk_end.strftime("%Y-%m-%d")},
            delay,
        )
        if rows:
            unique_records = {}
            for r in rows:
                key = (
                    r.get("date"),
                    str(r.get("option_id", "")),
                    r.get("put_call", ""),
                    str(r.get("contract_type", ""))
                )
                if None not in key and "" not in key:
                    unique_records[key] = (
                        key[0], key[1], key[2], key[3],
                        r.get("name", ""),
                        safe_float(r.get("market_open_interest")),
                        safe_float(r.get("buy_top5_trader_open_interest")),
                        safe_float(r.get("buy_top5_trader_open_interest_per")),
                        safe_float(r.get("buy_top10_trader_open_interest")),
                        safe_float(r.get("buy_top10_trader_open_interest_per")),
                        safe_float(r.get("sell_top5_trader_open_interest")),
                        safe_float(r.get("sell_top5_trader_open_interest_per")),
                        safe_float(r.get("sell_top10_trader_open_interest")),
                        safe_float(r.get("sell_top10_trader_open_interest_per")),
                        safe_float(r.get("buy_top5_specific_open_interest")),
                        safe_float(r.get("buy_top5_specific_open_interest_per")),
                        safe_float(r.get("buy_top10_specific_open_interest")),
                        safe_float(r.get("buy_top10_specific_open_interest_per")),
                        safe_float(r.get("sell_top5_specific_open_interest")),
                        safe_float(r.get("sell_top5_specific_open_interest_per")),
                        safe_float(r.get("sell_top10_specific_open_interest")),
                        safe_float(r.get("sell_top10_specific_open_interest_per")),
                    )

            records = list(unique_records.values())
            if records:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, UPSERT_OPTIONS_LARGE_OI, records)
                conn.commit()
                total += len(records)
                logger.info(f"  [options_large_oi] {start_d}~{chunk_end}: {len(records)} 筆")

        start_d = chunk_end + timedelta(days=1)

    logger.info(f"=== [options_large_oi] 完成，{total} 筆 ===")


# ─────────────────────────────────────────────
# ② 恐懼與貪婪指數
# ─────────────────────────────────────────────

def fetch_fear_greed_index(conn, start: str, end: str, delay: float, force: bool):
    logger.info("\n=== [fear_greed_index] 開始 ===")
    ensure_ddl(conn, DDL_FEAR_GREED_INDEX)

    s = start
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM fear_greed_index")
            last = cur.fetchone()[0]
        if last:
            s = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    if s > end:
        logger.info("[fear_greed_index] 已是最新，跳過")
        return

    # 分批抓
    start_d = datetime.strptime(s, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    total   = 0

    while start_d <= end_d:
        chunk_end = min(start_d + timedelta(days=180), end_d)
        rows = finmind_get(
            "CnnFearGreedIndex",
            {"start_date": start_d.strftime("%Y-%m-%d"),
             "end_date":   chunk_end.strftime("%Y-%m-%d")},
            delay,
        )
        if rows:
            records = [
                (
                    r.get("date"),
                    safe_float(r.get("fear_greed")),
                    r.get("fear_greed_emotion", "")
                )
                for r in rows if r.get("date")
            ]
            if records:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, UPSERT_FEAR_GREED_INDEX, records)
                conn.commit()
                total += len(records)
                logger.info(f"  [fear_greed_index] {start_d}~{chunk_end}: {len(records)} 筆")

        start_d = chunk_end + timedelta(days=1)

    logger.info(f"=== [fear_greed_index] 完成，{total} 筆 ===")


# ─────────────────────────────────────────────
# ③ 鉅額交易買賣日報表
# ─────────────────────────────────────────────

def fetch_block_trading(conn, start: str, end: str, delay: float, force: bool):
    logger.info("\n=== [block_trading] 開始 ===")
    ensure_ddl(conn, DDL_BLOCK_TRADING)

    s = start
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM block_trading")
            last = cur.fetchone()[0]
        if last:
            s = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    if s > end:
        logger.info("[block_trading] 已是最新，跳過")
        return

    # 按月分批抓
    start_d = datetime.strptime(s, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    total   = 0

    while start_d <= end_d:
        chunk_end = min(start_d + timedelta(days=30), end_d)
        rows = finmind_get(
            "TaiwanStockBlockTradingDailyReport",
            {"start_date": start_d.strftime("%Y-%m-%d"),
             "end_date":   chunk_end.strftime("%Y-%m-%d")},
            delay,
        )
        if rows:
            unique_records = {}
            for r in rows:
                key = (
                    r.get("date"),
                    str(r.get("stock_id", "")),
                    str(r.get("securities_trader_id", "")),
                    safe_float(r.get("price")),
                    str(r.get("trade_type", ""))
                )
                if None not in key and "" not in key:
                    unique_records[key] = (
                        key[0], key[1], key[2],
                        r.get("securities_trader", ""),
                        key[3],
                        safe_float(r.get("buy")),
                        safe_float(r.get("sell")),
                        key[4]
                    )

            records = list(unique_records.values())
            if records:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, UPSERT_BLOCK_TRADING, records)
                conn.commit()
                total += len(records)
                logger.info(f"  [block_trading] {start_d}~{chunk_end}: {len(records)} 筆")

        start_d = chunk_end + timedelta(days=1)

    logger.info(f"=== [block_trading] 完成，{total} 筆 ===")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="衍生品與情緒指標資料抓取")
    p.add_argument("--tables", nargs="+",
                   choices=["options_large_oi", "fear_greed_index", "block_trading"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tables = args.tables or ["options_large_oi", "fear_greed_index", "block_trading"]

    logger.info("=" * 60)
    logger.info("  衍生品與情緒指標資料抓取")
    logger.info(f"  資料集：{tables}")
    logger.info("=" * 60)

    try:
        conn = get_conn()
    except Exception as e:
        logger.error(f"DB 連線失敗：{e}"); sys.exit(1)

    for tbl in tables:
        s = args.start or DATASET_START.get(tbl) or "2024-01-01"

        if tbl == "options_large_oi":
            fetch_options_large_oi(conn, s, args.end, args.delay, args.force)
        elif tbl == "fear_greed_index":
            fetch_fear_greed_index(conn, s, args.end, args.delay, args.force)
        elif tbl == "block_trading":
            fetch_block_trading(conn, s, args.end, args.delay, args.force)

    conn.close()
    logger.info("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()
