from __future__ import annotations
import sys
import json
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_derivative_sentiment_data.py — 衍生品與情緒指標（逐支 commit 完整性版）
============================================================================
v2.2 改進：
  · 已有的逐支 commit 補上 try/except + rollback，避免單支失敗造成卡住的 transaction。
  · fear_greed_index 改為「整個 chunk 一次寫入 + 一次 commit」，避免逐筆 commit 的效能浪費。
  · 失敗清單寫入 outputs/{table}_failed_{date}.json。
"""

import argparse
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timedelta

import psycopg2.extras

from core.finmind_client import (
    finmind_get,
    wait_until_next_hour as wait_next_hour,
)
from core.db_utils import (
    get_db_conn as get_conn,
    ensure_ddl,
    safe_float,
    safe_int,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# 逐支 commit 工具函式（直接以 execute_values 寫入並 commit）
# ─────────────────────────────────────────────
def safe_execute_commit(conn, upsert_sql: str, rows: list, label: str = "") -> int:
    if not rows:
        return 0
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, upsert_sql, rows)
        conn.commit()
        return len(rows)
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


# ─────────────────────────────────────────────
# ① 選擇權大額交易人未沖銷部位
# ─────────────────────────────────────────────
def fetch_options_large_oi(conn, start: str, end: str, delay: float, force: bool):
    logger.info("\n=== [options_large_oi] 開始 ===")
    ensure_ddl(conn, DDL_OPTIONS_LARGE_OI)
    conn.commit()

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

    start_d = datetime.strptime(s, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    total   = 0
    failures: list[dict] = []

    while start_d <= end_d:
        chunk_end = min(start_d + timedelta(days=30), end_d)
        try:
            rows = finmind_get(
                "TaiwanOptionOpenInterestLargeTraders",
                {"start_date": start_d.strftime("%Y-%m-%d"),
                 "end_date":   chunk_end.strftime("%Y-%m-%d")},
                delay,
            )
        except Exception as e:
            logger.error(f"  [options_large_oi] {start_d}~{chunk_end} API 失敗：{e}")
            failures.append({"chunk": f"{start_d}~{chunk_end}", "error": str(e)})
            start_d = chunk_end + timedelta(days=1)
            continue

        if rows:
            unique_records = {}
            for r in rows:
                try:
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
                except Exception as e:
                    logger.warning(f"  [options_large_oi] mapper 異常筆，跳過：{e}")

            if unique_records:
                # ── 逐 option_id 進行 commit ──
                rows_by_id: dict[str, list] = defaultdict(list)
                for rec in unique_records.values():
                    oid = rec[1]
                    rows_by_id[oid].append(rec)

                ok = 0
                for oid, s_rows in rows_by_id.items():
                    n = safe_execute_commit(
                        conn, UPSERT_OPTIONS_LARGE_OI, s_rows,
                        label=f"options_large_oi/{oid}"
                    )
                    if n == 0 and s_rows:
                        failures.append({"option_id": oid, "chunk": f"{start_d}~{chunk_end}",
                                         "rows": len(s_rows)})
                    ok += n

                total += ok
                logger.info(
                    f"  [options_large_oi] {start_d}~{chunk_end}: "
                    f"寫入 {len(rows_by_id)} 支商品（共 {ok} 筆）"
                )

        start_d = chunk_end + timedelta(days=1)

    dump_failures("options_large_oi", failures)
    logger.info(f"=== [options_large_oi] 完成，{total} 筆 ===")


# ─────────────────────────────────────────────
# ② 恐懼與貪婪指數（市場層 — chunk 一次 commit）
# ─────────────────────────────────────────────
def fetch_fear_greed_index(conn, start: str, end: str, delay: float, force: bool):
    logger.info("\n=== [fear_greed_index] 開始 ===")
    ensure_ddl(conn, DDL_FEAR_GREED_INDEX)
    conn.commit()

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

    start_d = datetime.strptime(s, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    total   = 0
    failures: list[dict] = []

    while start_d <= end_d:
        chunk_end = min(start_d + timedelta(days=180), end_d)
        try:
            rows = finmind_get(
                "CnnFearGreedIndex",
                {"start_date": start_d.strftime("%Y-%m-%d"),
                 "end_date":   chunk_end.strftime("%Y-%m-%d")},
                delay,
            )
        except Exception as e:
            logger.error(f"  [fear_greed_index] {start_d}~{chunk_end} API 失敗：{e}")
            failures.append({"chunk": f"{start_d}~{chunk_end}", "error": str(e)})
            start_d = chunk_end + timedelta(days=1)
            continue

        if rows:
            records = []
            for r in rows:
                try:
                    if r.get("date"):
                        records.append((
                            r.get("date"),
                            safe_float(r.get("fear_greed")),
                            r.get("fear_greed_emotion", "")
                        ))
                except Exception as e:
                    logger.warning(f"  [fear_greed_index] mapper 異常筆，跳過：{e}")

            if records:
                # 整個 chunk 一次 commit（市場層資料 — 單筆 commit 浪費）
                n = safe_execute_commit(
                    conn, UPSERT_FEAR_GREED_INDEX, records,
                    label=f"fear_greed_index/{start_d}~{chunk_end}"
                )
                if n == 0 and records:
                    failures.append({"chunk": f"{start_d}~{chunk_end}",
                                     "rows": len(records)})
                total += n
                logger.info(f"  [fear_greed_index] {start_d}~{chunk_end}: 寫入 {n} 筆")

        start_d = chunk_end + timedelta(days=1)

    dump_failures("fear_greed_index", failures)
    logger.info(f"=== [fear_greed_index] 完成，{total} 筆 ===")


# ─────────────────────────────────────────────
# ③ 鉅額交易買賣日報表（逐支 commit）
# ─────────────────────────────────────────────
def fetch_block_trading(conn, start: str, end: str, delay: float, force: bool):
    logger.info("\n=== [block_trading] 開始 ===")
    ensure_ddl(conn, DDL_BLOCK_TRADING)
    conn.commit()

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

    start_d = datetime.strptime(s, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    total   = 0
    failures: list[dict] = []

    while start_d <= end_d:
        chunk_end = min(start_d + timedelta(days=30), end_d)
        try:
            rows = finmind_get(
                "TaiwanStockBlockTradingDailyReport",
                {"start_date": start_d.strftime("%Y-%m-%d"),
                 "end_date":   chunk_end.strftime("%Y-%m-%d")},
                delay,
            )
        except Exception as e:
            logger.error(f"  [block_trading] {start_d}~{chunk_end} API 失敗：{e}")
            failures.append({"chunk": f"{start_d}~{chunk_end}", "error": str(e)})
            start_d = chunk_end + timedelta(days=1)
            continue

        if rows:
            unique_records = {}
            for r in rows:
                try:
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
                except Exception as e:
                    logger.warning(f"  [block_trading] mapper 異常筆，跳過：{e}")

            if unique_records:
                # 逐 stock_id 進行 commit
                rows_by_stock: dict[str, list] = defaultdict(list)
                for rec in unique_records.values():
                    sid = rec[1]
                    rows_by_stock[sid].append(rec)

                ok = 0
                for sid, s_rows in rows_by_stock.items():
                    n = safe_execute_commit(
                        conn, UPSERT_BLOCK_TRADING, s_rows,
                        label=f"block_trading/{sid}"
                    )
                    if n == 0 and s_rows:
                        failures.append({"stock_id": sid, "chunk": f"{start_d}~{chunk_end}",
                                         "rows": len(s_rows)})
                    ok += n

                total += ok
                logger.info(
                    f"  [block_trading] {start_d}~{chunk_end}: "
                    f"寫入 {len(rows_by_stock)} 支股票（共 {ok} 筆）"
                )

        start_d = chunk_end + timedelta(days=1)

    dump_failures("block_trading", failures)
    logger.info(f"=== [block_trading] 完成，{total} 筆 ===")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="衍生品與情緒指標資料抓取 v2.2")
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

    try:
        for tbl in tables:
            s = args.start or DATASET_START.get(tbl) or "2024-01-01"
            try:
                if tbl == "options_large_oi":
                    fetch_options_large_oi(conn, s, args.end, args.delay, args.force)
                elif tbl == "fear_greed_index":
                    fetch_fear_greed_index(conn, s, args.end, args.delay, args.force)
                elif tbl == "block_trading":
                    fetch_block_trading(conn, s, args.end, args.delay, args.force)
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                logger.error(f"[{tbl}] 未預期錯誤：{e}")
                continue
    finally:
        conn.close()
    logger.info("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()