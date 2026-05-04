from __future__ import annotations
import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_sponsor_chip_data.py — Sponsor 方案進階籌碼資料抓取
==========================================================
資料集（已驗證 API 欄位）：
  1. holding_shares_per  ← TaiwanStockHoldingSharesPer
     欄位：date, stock_id, HoldingSharesLevel, people, percent, unit
  2. broker_trades       ← TaiwanStockShareholdingByBroker
     欄位：date, stock_id, broker_id, broker_name, buy, sell
  3. eight_banks         ← TaiwanStockGovernmentBankBuySell
     欄位：date, stock_id, buy, sell
  4. futures_large_oi    ← TaiwanFuturesOpenInterestLargeTraders
     欄位：date, contract_code, name, long_position, short_position...

修改摘要（第四輪審查）：
  [P0-SEC]  移除硬編碼 FINMIND_TOKEN / DB_CONFIG，改由 config.py 統一管理
  [P0]      使用 core.finmind_client.finmind_get（統一重試邏輯）
  [P3]      使用 core.db_utils 工具函式（ensure_ddl, bulk_upsert, safe_int 等）
  [P0 合併] 合併 fetch_sponsor_chip_data_py.eight_banks 的改進版 fetch_eight_banks()
            → 改為按月分塊請求（避免單次資料過大）
            → 修正防死循環邏輯（s_str == e_str 且無資料時推進一天）
            → 直接儲存全市場八大行庫資料（原始個別行庫明細彙總至 stock 層）

執行：
    python fetch_sponsor_chip_data.py                    # 全部
    python fetch_sponsor_chip_data.py --tables holding_shares_per eight_banks
    python fetch_sponsor_chip_data.py --stock-id 2330   # 指定股票
    python fetch_sponsor_chip_data.py --force           # 強制重抓
"""

import argparse
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta

import psycopg2
import urllib3

# 隱藏 InsecureRequestWarning（當 verify=False 時）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import DB_CONFIG  # noqa: F401（db_utils 內部使用）
from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    safe_int,
    safe_float,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

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
CREATE TABLE IF NOT EXISTS eight_banks_buy_sell (
    date     DATE,
    stock_id VARCHAR(50),
    buy      BIGINT,
    sell     BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_eight_banks_date ON eight_banks_buy_sell (date);
CREATE INDEX IF NOT EXISTS idx_eb_stock ON eight_banks_buy_sell (stock_id, date);
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
INSERT INTO eight_banks_buy_sell (date, stock_id, buy, sell)
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


def get_max_dates(conn, table: str, pk_col: str = "stock_id") -> dict:
    """取得各 pk_col 值在 table 中的最大 date，回傳 {pk_val: date} 字典。"""
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

        rows = finmind_get(
            "TaiwanStockHoldingSharesPer",
            {"data_id": sid, "start_date": s, "end_date": end},
            delay,
            raise_on_error=True
        )
        if not rows:
            continue

        # 去重
        unique = {}
        for r in rows:
            lv = str(r.get("HoldingSharesLevel", ""))
            row = (
                r.get("date"), sid, lv,
                safe_int(r.get("people")),
                safe_float(r.get("percent")),
                r.get("unit", lv),
            )
            unique[(row[0], row[1], row[2])] = row
        records = list(unique.values())

        bulk_upsert(conn, UPSERT_HOLDING, records, "(%s, %s, %s, %s, %s, %s)")
        total += len(records)

        if i % 100 == 0:
            logger.info(f"  [holding_shares_per] {i}/{len(stock_ids)}，累計 {total:,} 筆")

    logger.info(f"=== [holding_shares_per] 完成，{total:,} 筆 ===")


# ─────────────────────────────────────────────
# ② 台股分點資料表
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
                "TaiwanStockTradingDailyReport",
                {
                    "data_id":    sid,
                    "start_date": chunk_s.strftime("%Y-%m-%d"),
                    "end_date":   chunk_e.strftime("%Y-%m-%d"),
                },
                delay,
                raise_on_error=True
            )
            if rows:
                # 彙總：同日、同股票、同券商的資料合併（去除價格分維度）
                agg: dict = {}
                for r in rows:
                    key = (r.get("date"), sid, str(r.get("securities_trader_id", "")))
                    buy  = safe_int(r.get("buy") or 0) or 0
                    sell = safe_int(r.get("sell") or 0) or 0
                    name = r.get("securities_trader", "")
                    if key not in agg:
                        agg[key] = {"buy": 0, "sell": 0, "name": name}
                    agg[key]["buy"]  += buy
                    agg[key]["sell"] += sell

                records = [
                    (k[0], k[1], k[2], v["name"], v["buy"], v["sell"])
                    for k, v in agg.items()
                ]
                bulk_upsert(conn, UPSERT_BROKER, records, "(%s, %s, %s, %s, %s, %s)")
                total += len(records)
                logger.info(
                    f"  [broker_trades] {sid} {chunk_s}~{chunk_e}: "
                    f"{len(records)} 筆（彙總自 {len(rows)} 筆）"
                )

            chunk_s = chunk_e + timedelta(days=1)

    logger.info(f"=== [broker_trades] 完成，{total:,} 筆 ===")


# ─────────────────────────────────────────────
# ③ 八大行庫買賣表
#   [P0 合併] fetch_sponsor_chip_data_py.eight_banks
#   改進：按月分塊請求 + 防死循環
# ─────────────────────────────────────────────

def fetch_eight_banks(conn, stock_ids, start, end, delay, force):
    """
    八大行庫買賣超 (TaiwanStockGovernmentBankBuySell)
    注意：此 Dataset 不支援 data_id，必須一次抓取全市場。

    [改進] 按月分塊請求（原版逐日請求 API 次數過多）。
    [改進] 正確防死循環：若當月分塊無資料且起迄相同日，推進至下月。
    [改進] 直接儲存全市場資料（不過濾個股），完整保留八大行庫資訊。
    """
    logger.info(f"\n=== [eight_banks] 開始抓取全市場資料 ({start} ~ {end}) ===")
    ensure_ddl(conn, DDL_EIGHT_BANKS)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    # 若 DB 已有資料，從最新日期的隔天開始
    current_dt = start_dt
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM eight_banks_buy_sell")
            last_date = cur.fetchone()[0]
        if last_date:
            current_dt = max(
                current_dt,
                datetime.combine(last_date + timedelta(days=1), datetime.min.time()),
            )

    if current_dt > end_dt:
        logger.info("  [eight_banks] 已是最新，跳過")
        return

    total = 0
    while current_dt <= end_dt:
        # [P0-FIX] 八大行庫資料量極大，FinMind 強制要求每次只能抓一天 (Status 400 if multiple days)
        d_str = current_dt.strftime("%Y-%m-%d")
        logger.info(f"  抓取 {d_str} (單日模式)…")
        
        rows = finmind_get(
            "TaiwanStockGovernmentBankBuySell",
            {"start_date": d_str, "end_date": d_str},
            delay,
            raise_on_error=True
        )

        if rows:
            # 去重
            unique = {}
            for r in rows:
                row = (
                    r.get("date"),
                    r.get("stock_id"),
                    safe_int(r.get("buy")),
                    safe_int(r.get("sell")),
                )
                unique[(row[0], row[1])] = row
            # ── 逐支模式進行 commit ──
            rows_by_stock = defaultdict(list)
            for rec in unique.values():
                sid = rec[1]  # stock_id
                rows_by_stock[sid].append(rec)

            for sid, s_rows in rows_by_stock.items():
                bulk_upsert(conn, UPSERT_EIGHT_BANKS, s_rows, "(%s, %s, %s, %s)")
            
            total += len(unique)
            logger.info(
                f"    → 寫入完成（含 {len(rows_by_stock)} 支股票，"
                f"共 {len(unique):,} 筆；累計 {total:,} 筆）"
            )

        # 推進至隔日
        current_dt += timedelta(days=1)

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
            cur.execute(
                "SELECT MAX(date) FROM futures_large_oi WHERE contract_code='TX'"
            )
            last = cur.fetchone()[0]
        if last:
            next_day = (last + timedelta(days=1)).strftime("%Y-%m-%d")
            if next_day > end:
                logger.info("  [futures_large_oi] 已是最新，跳過")
                return
            s = next_day

    rows = finmind_get(
        "TaiwanFuturesOpenInterestLargeTraders",
        {"start_date": s, "end_date": end},
        delay,
        raise_on_error=True
    )
    if not rows:
        logger.info(
            "  [futures_large_oi] 無資料（請確認 Sponsor 方案已啟用此資料集）"
        )
        return

    # 去重
    unique = {}
    for r in rows:
        row = (
            r.get("date"),
            r.get("contract_code", ""),
            r.get("name", ""),
            safe_int(r.get("long_position")),
            safe_int(r.get("long_position_over50")),
            safe_int(r.get("short_position")),
            safe_int(r.get("short_position_over50")),
            safe_int(r.get("net_position")),
            safe_int(r.get("market_total_oi")),
        )
        unique[(row[0], row[1], row[2])] = row
    records = list(unique.values())
    if records:
        # ── 逐項模式進行 commit ──
        rows_by_contract = defaultdict(list)
        for rec in records:
            cc = rec[1]  # contract_code
            rows_by_contract[cc].append(rec)

        for cc, s_rows in rows_by_contract.items():
            bulk_upsert(conn, UPSERT_FUTURES_LARGE_OI, s_rows, "(%s, %s, %s, %s, %s, %s, %s, %s, %s)")
        
        logger.info(f"=== [futures_large_oi] 完成，寫入 {len(rows_by_contract)} 項商品（共 {len(records):,} 筆） ===")


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
    p.add_argument(
        "--tables", nargs="+",
        choices=["holding_shares_per", "broker_trades", "eight_banks", "futures_large_oi"],
    )
    p.add_argument("--stock-id", default=None, help="逗號分隔股票代碼")
    p.add_argument("--start",  default=None)
    p.add_argument("--end",    default=DEFAULT_END)
    p.add_argument("--delay",  type=float, default=1.2)
    p.add_argument("--force",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tables = args.tables or [
        "eight_banks", "futures_large_oi"
    ]

    logger.info("=" * 60)
    logger.info("  Sponsor 進階籌碼資料抓取管線啟動")
    logger.info(f"  資料集：{tables}  結束日：{args.end}")
    logger.info("=" * 60)

    try:
        conn = get_db_conn()
    except Exception as e:
        logger.error(f"DB 連線失敗：{e}")
        sys.exit(1)

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
