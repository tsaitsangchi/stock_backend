"""
fetch_sponsor_chip_data.py — Sponsor 方案進階籌碼資料抓取
==========================================================
資料集（已驗證 API 欄位）：
  1. holding_shares_per  ← TaiwanStockHoldingSharesPer
     欄位：date, stock_id, HoldingSharesLevel, people, percent, unit
  2. broker_trades       ← TaiwanStockShareholdingByBroker
     欄位：date, stock_id, broker_id, broker_name, buy, sell
  3. eight_banks         ← TaiwanStockEightKindOfState
     欄位：date, stock_id, buy, sell
  4. futures_large_oi    ← TaiwanFuturesOpenInterestLargeTraders
     欄位：date, contract_code, name, long_position, short_position...

執行：
    python fetch_sponsor_chip_data.py                    # 全部
    python fetch_sponsor_chip_data.py --tables holding_shares_per eight_banks
    python fetch_sponsor_chip_data.py --stock-id 2330   # 指定股票
    python fetch_sponsor_chip_data.py --force           # 強制重抓
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import date, datetime, timedelta

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
    "dbname": "stock", "user": "stock",
    "password": "stock", "host": "localhost", "port": "5432",
}
DATASET_START = {
    "holding_shares_per": "2014-01-01",
    "broker_trades":      "2016-01-01",
    "eight_banks":        "2021-01-01",
    "futures_large_oi":   "2010-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL（使用驗證後的真實欄位名稱）
# ─────────────────────────────────────────────

DDL_HOLDING = """
CREATE TABLE IF NOT EXISTS holding_shares_per (
    date         DATE,
    stock_id     VARCHAR(50),
    level        VARCHAR(50),   -- HoldingSharesLevel (級距代碼)
    people       BIGINT,        -- 持股人數
    percent      NUMERIC(10,4), -- 持股比例 %
    unit         VARCHAR(100),  -- 級距文字說明
    PRIMARY KEY (date, stock_id, level)
);
CREATE INDEX IF NOT EXISTS idx_hsp_stock ON holding_shares_per (stock_id, date);
"""

DDL_BROKER = """
CREATE TABLE IF NOT EXISTS broker_trades (
    date         DATE,
    stock_id     VARCHAR(50),
    broker_id    VARCHAR(50),
    broker_name  VARCHAR(100),
    buy          BIGINT,
    sell         BIGINT,
    PRIMARY KEY (date, stock_id, broker_id)
);
CREATE INDEX IF NOT EXISTS idx_bt_stock  ON broker_trades (stock_id, date);
CREATE INDEX IF NOT EXISTS idx_bt_broker ON broker_trades (broker_id, date);
"""

DDL_EIGHT_BANKS = """
CREATE TABLE IF NOT EXISTS eight_banks (
    date         DATE,
    stock_id     VARCHAR(50),
    buy          BIGINT,
    sell         BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_eb_stock ON eight_banks (stock_id, date);
"""

DDL_FUTURES_LARGE_OI = """
CREATE TABLE IF NOT EXISTS futures_large_oi (
    date                  DATE,
    contract_code         VARCHAR(20),
    name                  VARCHAR(50),
    long_position         BIGINT,
    long_position_over50  BIGINT,
    short_position        BIGINT,
    short_position_over50 BIGINT,
    net_position          BIGINT,
    market_total_oi       BIGINT,
    PRIMARY KEY (date, contract_code, name)
);
CREATE INDEX IF NOT EXISTS idx_floi_date ON futures_large_oi (date, contract_code);
"""

# ─────────────────────────────────────────────
# Upsert SQL
# ─────────────────────────────────────────────

UPSERT_HOLDING = """
INSERT INTO holding_shares_per (date, stock_id, level, people, percent, unit)
VALUES %s
ON CONFLICT (date, stock_id, level) DO UPDATE SET
    people  = EXCLUDED.people,
    percent = EXCLUDED.percent,
    unit    = EXCLUDED.unit;
"""

UPSERT_BROKER = """
INSERT INTO broker_trades (date, stock_id, broker_id, broker_name, buy, sell)
VALUES %s
ON CONFLICT (date, stock_id, broker_id) DO UPDATE SET
    broker_name = EXCLUDED.broker_name,
    buy  = EXCLUDED.buy,
    sell = EXCLUDED.sell;
"""

UPSERT_EIGHT_BANKS = """
INSERT INTO eight_banks (date, stock_id, buy, sell)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    buy  = EXCLUDED.buy,
    sell = EXCLUDED.sell;
"""

UPSERT_FUTURES_LARGE_OI = """
INSERT INTO futures_large_oi (
    date, contract_code, name,
    long_position, long_position_over50,
    short_position, short_position_over50,
    net_position, market_total_oi
) VALUES %s
ON CONFLICT (date, contract_code, name) DO UPDATE SET
    long_position         = EXCLUDED.long_position,
    long_position_over50  = EXCLUDED.long_position_over50,
    short_position        = EXCLUDED.short_position,
    short_position_over50 = EXCLUDED.short_position_over50,
    net_position          = EXCLUDED.net_position,
    market_total_oi       = EXCLUDED.market_total_oi;
"""

# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────

def safe_int(v):
    if v is None: return None
    try: return int(float(str(v).strip()))
    except: return None

def safe_float(v):
    if v is None: return None
    try: return float(str(v).strip())
    except: return None

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def ensure_ddl(conn, *ddls):
    with conn.cursor() as cur:
        for d in ddls:
            cur.execute(d)
    conn.commit()

def wait_next_hour():
    now = datetime.now()
    nxt = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    sec = (nxt - now).total_seconds() + 65
    logger.warning(f"402 用量上限，等待至 {nxt.strftime('%H:%M:%S')}（{sec:.0f}s）")
    time.sleep(sec)
    logger.info("恢復請求。")

def finmind_get(dataset: str, params: dict, delay: float = 1.2) -> list:
    """帶指數退避的 FinMind 請求（ConnectTimeout 最多 5 次重試）"""
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    rp = {"dataset": dataset, **params}
    MAX, BASE = 5, 5.0

    while True:
        for attempt in range(1, MAX + 1):
            try:
                resp = requests.get(FINMIND_API_URL, headers=headers,
                                    params=rp, timeout=(15, 120), verify=False)
                if resp.status_code == 402:
                    wait_next_hour(); break
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("status") == 402:
                    wait_next_hour(); break
                if payload.get("status") != 200:
                    logger.warning(f"[{dataset}] status={payload.get('status')}, 跳過")
                    return []
                time.sleep(delay)
                return payload.get("data", [])

            except requests.exceptions.ConnectTimeout:
                w = BASE * (3 ** (attempt - 1)) + random.uniform(0, 3)
                if attempt < MAX:
                    logger.warning(f"[{dataset}] 連線逾時({attempt}/{MAX})，{w:.0f}s 後重試")
                    time.sleep(w)
                else:
                    logger.error(f"[{dataset}] 連線逾時，已重試 {MAX} 次，跳過"); return []

            except requests.exceptions.ReadTimeout:
                w = BASE * (2 ** (attempt - 1)) + random.uniform(0, 2)
                if attempt < MAX:
                    logger.warning(f"[{dataset}] 讀取逾時({attempt}/{MAX})，{w:.0f}s 後重試")
                    time.sleep(w)
                else:
                    logger.error(f"[{dataset}] 讀取逾時，已重試 {MAX} 次，跳過"); return []

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 400:
                    logger.debug(f"[{dataset}] HTTP 400: {e.response.text if e.response else 'No response body'}")
                    return []
                if code == 402: wait_next_hour(); break
                w = BASE * (2 ** (attempt - 1))
                logger.warning(f"[{dataset}] HTTP {code} ({attempt}/{MAX})，{w:.0f}s 後重試")
                time.sleep(w)

            except Exception as exc:
                w = BASE * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.warning(f"[{dataset}] 失敗({attempt}/{MAX})：{type(exc).__name__}: {exc}，{w:.0f}s 後重試")
                time.sleep(w)
        else:
            logger.error(f"[{dataset}] 已重試 {MAX} 次，跳過")
            return []


def get_max_dates(conn, table: str, pk_col: str = "stock_id") -> dict:
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT {pk_col}, MAX(date) FROM {table} GROUP BY {pk_col}")
            return {r[0]: r[1] for r in cur.fetchall()}
        except Exception:
            conn.rollback()
            return {}


# ─────────────────────────────────────────────
# ① 股權持股分級表
# ─────────────────────────────────────────────

def fetch_holding_shares_per(conn, stock_ids, start, end, delay, force):
    logger.info(f"\n=== [holding_shares_per] 開始（{len(stock_ids)} 支）===")
    ensure_ddl(conn, DDL_HOLDING)
    max_dates = {} if force else get_max_dates(conn, "holding_shares_per")
    total = 0

    for i, sid in enumerate(stock_ids, 1):
        last = max_dates.get(sid)
        s = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else start
        if s > end:
            continue

        rows = finmind_get("TaiwanStockHoldingSharesPer",
                           {"data_id": sid, "start_date": s, "end_date": end}, delay)
        if not rows:
            continue

        records = []
        for r in rows:
            # 真實欄位：HoldingSharesLevel, people, percent, unit
            lv = str(r.get("HoldingSharesLevel", ""))
            records.append((
                r.get("date"), sid, lv,
                safe_int(r.get("people")),
                safe_float(r.get("percent")),
                r.get("unit", lv),
            ))

        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, UPSERT_HOLDING, records)
        conn.commit()
        total += len(records)

        if i % 100 == 0:
            logger.info(f"  [holding_shares_per] {i}/{len(stock_ids)}，累計 {total:,} 筆")

    logger.info(f"=== [holding_shares_per] 完成，{total:,} 筆 ===")


# ─────────────────────────────────────────────
# ② 台股分點資料表（預設只抓重點股票）
# ─────────────────────────────────────────────

def fetch_broker_trades(conn, stock_ids, start, end, delay, force):
    logger.info(f"\n=== [broker_trades] 開始（{len(stock_ids)} 支）===")
    ensure_ddl(conn, DDL_BROKER)
    max_dates = {} if force else get_max_dates(conn, "broker_trades")
    total = 0
    CHUNK = 60  # 分點資料量大，分 60 天一批

    for sid in stock_ids:
        last = max_dates.get(sid)
        chunk_s = (last + timedelta(days=1)) if last else \
                  datetime.strptime(start, "%Y-%m-%d").date()
        end_d = datetime.strptime(end, "%Y-%m-%d").date()

        while chunk_s <= end_d:
            chunk_e = min(chunk_s + timedelta(days=CHUNK - 1), end_d)
            rows = finmind_get(
                "TaiwanStockShareholdingByBroker",
                {"data_id": sid,
                 "start_date": chunk_s.strftime("%Y-%m-%d"),
                 "end_date":   chunk_e.strftime("%Y-%m-%d")},
                delay,
            )
            if rows:
                records = [
                    (r.get("date"), sid, str(r.get("broker_id", "")),
                     r.get("broker_name", ""),
                     safe_int(r.get("buy")), safe_int(r.get("sell")))
                    for r in rows
                ]
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, UPSERT_BROKER, records)
                conn.commit()
                total += len(records)
                logger.info(f"  [broker_trades] {sid} {chunk_s}~{chunk_e}: {len(records)} 筆")

            chunk_s = chunk_e + timedelta(days=1)

    logger.info(f"=== [broker_trades] 完成，{total:,} 筆 ===")


# ─────────────────────────────────────────────
# ③ 八大行庫買賣表
# ─────────────────────────────────────────────

def fetch_eight_banks(conn, stock_ids, start, end, delay, force):
    """
    八大行庫買賣超 (TaiwanStockGovernmentBankBuySell)
    注意：此 Dataset 不支援 data_id，必須一次抓取全市場。
    """
    logger.info(f"\n=== [eight_banks] 開始抓取全市場資料 ({start} ~ {end}) ===")
    ensure_ddl(conn, DDL_EIGHT_BANKS)
    
    # 這裡我們採取按月抓取的策略，避免單次請求過大
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    
    current_dt = start_dt
    total = 0
    
    # 為了加速，我們先找出 DB 裡最大的日期
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM eight_banks")
            last_date = cur.fetchone()[0]
            if last_date:
                current_dt = max(current_dt, datetime.combine(last_date + timedelta(days=1), datetime.min.time()))

    # 策略：優先抓取最近一年的資料，讓訓練能盡早開始
    # 然後再補齊剩下的歷史資料
    today_dt = datetime.combine(date.today(), datetime.min.time())
    one_year_ago = today_dt - timedelta(days=365)
    
    # 建立日期清單並排序（最近的優先）
    all_dates = []
    temp_dt = start_dt
    while temp_dt <= end_dt:
        all_dates.append(temp_dt)
        temp_dt += timedelta(days=1)
        
    # 優先序：最近一年的日期 (降序) -> 剩下的日期 (降序)
    recent_dates = sorted([d for d in all_dates if d >= one_year_ago], reverse=True)
    older_dates = sorted([d for d in all_dates if d < one_year_ago], reverse=True)
    ordered_dates = recent_dates + older_dates

    # 找出已存在的日期避免重複
    existing_dates = set()
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT date FROM eight_banks")
            existing_dates = {row[0] for row in cur.fetchall()}

    for current_dt in ordered_dates:
        s_str = current_dt.strftime("%Y-%m-%d")
        if current_dt.date() in existing_dates:
            continue
            
        logger.info(f"  抓取 {s_str}...")
        rows = finmind_get("TaiwanStockGovernmentBankBuySell",
                           {"start_date": s_str, "end_date": s_str}, delay)
        
        if rows:
            # 由於此 Dataset 回傳各行庫明細，我們需按 (date, stock_id) 進行彙總
            agg = {}
            for r in rows:
                key = (r.get("date"), r.get("stock_id"))
                buy = safe_int(r.get("buy"))
                sell = safe_int(r.get("sell"))
                if key not in agg:
                    agg[key] = [0, 0]
                agg[key][0] += buy
                agg[key][1] += sell
            
            records = [
                (date_str, sid, vols[0], vols[1])
                for (date_str, sid), vols in agg.items()
            ]
            
            # Debug: log info
            keys = [(r[0], r[1]) for r in records]
            unique_keys = set(keys)
            logger.info(f"Aggregated: {len(records)} rows. Unique keys: {len(unique_keys)}")
            if len(records) > 0:
                logger.info(f"Sample key: {keys[0]} (Types: {type(keys[0][0])}, {type(keys[0][1])})")
            
            if len(keys) != len(unique_keys):
                logger.error(f"CRITICAL: Duplicate keys found in records!")
                # Find the duplicates
                seen = set()
                for k in keys:
                    if k in seen:
                        logger.error(f"Duplicate key: {k}")
                    seen.add(k)
            
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, UPSERT_EIGHT_BANKS, records)
            conn.commit()
            total += len(records)
            logger.info(f"    → 寫入 {len(records):,} 筆（累計 {total:,} 筆）")

    logger.info(f"=== [eight_banks] 完成，累計 {total:,} 筆 ===")


# ─────────────────────────────────────────────
# ④ 期貨大額交易人未沖銷部位
# ─────────────────────────────────────────────

def fetch_futures_large_oi(conn, start, end, delay, force):
    logger.info("\n=== [futures_large_oi] 開始 ===")
    ensure_ddl(conn, DDL_FUTURES_LARGE_OI)

    s = start
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM futures_large_oi WHERE contract_code='TX'")
            last = cur.fetchone()[0]
        if last:
            s = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    if s > end:
        logger.info("[futures_large_oi] 已是最新，跳過")
        return

    rows = finmind_get(
        "TaiwanFuturesOpenInterestLargeTraders",
        {"start_date": s, "end_date": end},
        delay,
    )
    if not rows:
        logger.info("[futures_large_oi] 無資料（請確認 Sponsor 方案已啟用此資料集）")
        return

    records = [
        (r.get("date"), r.get("contract_code", ""), r.get("name", ""),
         safe_int(r.get("long_position")),
         safe_int(r.get("long_position_over50")),
         safe_int(r.get("short_position")),
         safe_int(r.get("short_position_over50")),
         safe_int(r.get("net_position")),
         safe_int(r.get("market_total_oi")))
        for r in rows
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, UPSERT_FUTURES_LARGE_OI, records)
    conn.commit()
    logger.info(f"=== [futures_large_oi] 完成，{len(records):,} 筆 ===")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def get_stock_ids(conn, stock_id_arg):
    if stock_id_arg:
        return [s.strip() for s in stock_id_arg.split(",")]
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT stock_id FROM stock_info ORDER BY stock_id")
        return [r[0] for r in cur.fetchall()]


def parse_args():
    p = argparse.ArgumentParser(description="Sponsor 進階籌碼資料抓取")
    p.add_argument("--tables", nargs="+",
                   choices=["holding_shares_per", "broker_trades",
                            "eight_banks", "futures_large_oi"])
    p.add_argument("--stock-id", default=None, help="逗號分隔股票代碼")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tables = args.tables or ["holding_shares_per", "broker_trades",
                              "eight_banks", "futures_large_oi"]

    logger.info("=" * 60)
    logger.info("  Sponsor 進階籌碼資料抓取管線啟動")
    logger.info(f"  資料集：{tables}  結束日：{args.end}")
    logger.info("=" * 60)

    try:
        conn = get_conn()
    except Exception as e:
        logger.error(f"DB 連線失敗：{e}"); sys.exit(1)

    stock_ids = get_stock_ids(conn, args.stock_id)
    logger.info(f"共 {len(stock_ids)} 支股票")

    for tbl in tables:
        s = args.start or DATASET_START.get(tbl, "2010-01-01")

        if tbl == "holding_shares_per":
            fetch_holding_shares_per(conn, stock_ids, s, args.end, args.delay, args.force)
        elif tbl == "broker_trades":
            # 分點資料量極大，未指定 stock_id 時只抓 2330
            targets = [args.stock_id] if args.stock_id else ["2330"]
            fetch_broker_trades(conn, targets, s, args.end, args.delay, args.force)
        elif tbl == "eight_banks":
            fetch_eight_banks(conn, stock_ids, s, args.end, args.delay, args.force)
        elif tbl == "futures_large_oi":
            fetch_futures_large_oi(conn, s, args.end, args.delay, args.force)

    conn.close()
    logger.info("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()
