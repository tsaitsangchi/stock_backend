"""
fetch_balance_sheet_data.py — TaiwanStockBalanceSheet fetcher(§14.7-CA Phase F-1 / §0.1 100% closure)
================================================================================
最後更新日期: 2026-05-27
主權狀態: IMPLEMENTED (§14.7-CA Phase F-1 / §0.1 Investment family 100% closure)
最高原則: Balance Sheet Raw Schema Authority

## 一、核心定義說明
- [BalanceSheet Raw Schema]: 從 FinMind `TaiwanStockBalanceSheet` API 取 long-format
  (date / stock_id / type / value / origin_name)。Cooper-Gulen-Schill 2008 之
  asset_growth anomaly + §14.7-BI ROE 解鎖之 prerequisite。
- [Publication-date Discipline]: quarter-end 遞延 45 天(per §8.5-9 hardcoded_conservative)。
- [Universe Lock]: core_universe ∪ convex_universe(N dynamic per §14.7-BW)。

## 二、CLI 範例
    python scripts/fetchers/fetch_balance_sheet_data.py --start 2018-01-01
    python scripts/fetchers/fetch_balance_sheet_data.py --stock-id 2330 --force
"""
from __future__ import annotations

import sys
import os
import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from psycopg2.extras import execute_values

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

_project_root = _base_dir.parent
load_dotenv(_project_root / ".env")

from core.db_utils import get_db_conn, get_core_stocks_from_db, ensure_ddl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN")

DEFAULT_START = "2018-01-01"
DEFAULT_END = date.today().strftime("%Y-%m-%d")

DDL_BALANCE_SHEET = """
CREATE TABLE IF NOT EXISTS "TaiwanStockBalanceSheet" (
    date         DATE,
    stock_id     VARCHAR(20),
    type         VARCHAR(255),
    value        NUMERIC(24,4),
    origin_name  VARCHAR(255)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_balance_sheet
    ON "TaiwanStockBalanceSheet" (stock_id, date, type, origin_name);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_date
    ON "TaiwanStockBalanceSheet" (date);
"""

UPSERT_BALANCE_SHEET = """
INSERT INTO "TaiwanStockBalanceSheet" (stock_id, date, type, value, origin_name)
VALUES %s
ON CONFLICT (stock_id, date, type, origin_name)
DO UPDATE SET value = EXCLUDED.value;
"""


def fetch_balance_sheet_for_stock(sid: str, start: str, end: str, retries: int = 3, delay: float = 0.4):
    """Pull TaiwanStockBalanceSheet for one stock;retry on transient error。"""
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"} if FINMIND_TOKEN else {}
    params = {
        "dataset": "TaiwanStockBalanceSheet",
        "data_id": sid,
        "start_date": start,
        "end_date": end,
    }
    for attempt in range(retries):
        try:
            r = requests.get(FINMIND_API_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                d = r.json()
                if d.get("msg") == "success":
                    return d.get("data", [])
                else:
                    logger.warning(f"[{sid}] FinMind msg: {d.get('msg')}")
                    return []
            elif r.status_code == 402:  # rate limited / quota
                logger.warning(f"[{sid}] HTTP 402 (quota / rate limit); sleeping 60s")
                time.sleep(60)
            else:
                logger.warning(f"[{sid}] HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            logger.warning(f"[{sid}] request error: {e}")
        time.sleep(delay * (attempt + 1))
    return None  # 全失敗


def map_row(r: dict):
    """Quarter-end publication 遞延 45 天 / 截斷 VARCHAR(255)。"""
    try:
        original_date = datetime.strptime(r["date"], "%Y-%m-%d")
        publish_date = original_date + timedelta(days=45)
        val = r.get("value")
        if val is None:
            return None
        return (
            str(r["stock_id"]),
            publish_date.date(),
            str(r.get("type", ""))[:255],
            float(val),
            str(r.get("origin_name", r.get("type", "")))[:255],
        )
    except Exception as e:
        logger.debug(f"map error: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=8, help="concurrent worker threads")
    p.add_argument("--force", action="store_true", help="re-fetch existing dates")
    p.add_argument("--limit", type=int, default=0, help="limit number of stocks(testing)")
    args = p.parse_args()

    if not FINMIND_TOKEN:
        logger.error("❌ FINMIND_TOKEN missing from .env")
        sys.exit(1)

    conn = get_db_conn()
    try:
        # DDL
        ensure_ddl(conn, DDL_BALANCE_SHEET)
        conn.commit()

        # 決定 universe
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            stock_configs = get_core_stocks_from_db(conn)
            stock_ids = sorted([sid for sid, cfg in stock_configs.items() if cfg.get("is_active", True)])

        if args.limit > 0:
            stock_ids = stock_ids[:args.limit]

        logger.info(f"=== TaiwanStockBalanceSheet sync / N={len(stock_ids)} / start={args.start} ===")

        # 取每股 latest date in DB to skip already-synced(unless --force)
        existing_latest = {}
        if not args.force:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT stock_id, MAX(date) FROM "TaiwanStockBalanceSheet" '
                    'WHERE stock_id = ANY(%s) GROUP BY stock_id',
                    (stock_ids,),
                )
                for sid, d in cur.fetchall():
                    existing_latest[sid] = d

        total_rows = 0
        success_count = 0
        skip_count = 0
        fail_count = 0
        to_fetch = []

        for sid in stock_ids:
            eff_start = args.start
            if sid in existing_latest and existing_latest[sid] is not None:
                latest_d = existing_latest[sid]
                if (date.today() - latest_d).days < 90:
                    skip_count += 1
                    continue
                eff_start = (latest_d + timedelta(days=1)).strftime("%Y-%m-%d")
            to_fetch.append((sid, eff_start))

        logger.info(f"  to fetch: {len(to_fetch)} / pre-skipped(up-to-date): {skip_count}")

        # 並行抓取(workers=8;FinMind Sponsor 6000/hour 可承受)
        def _worker(args_tuple):
            sid, eff_start = args_tuple
            data = fetch_balance_sheet_for_stock(sid, eff_start, args.end, retries=3, delay=args.delay)
            return sid, data

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, t): t[0] for t in to_fetch}
            for idx, fut in enumerate(as_completed(futures), 1):
                sid = futures[fut]
                try:
                    _, data = fut.result()
                except Exception as e:
                    fail_count += 1
                    logger.warning(f"  [{idx}/{len(to_fetch)}] {sid} EXCEPTION: {e}")
                    continue

                if data is None:
                    fail_count += 1
                elif data:
                    rows = [map_row(r) for r in data]
                    rows = [r for r in rows if r]
                    if rows:
                        with conn.cursor() as cur:
                            execute_values(cur, UPSERT_BALANCE_SHEET, rows, page_size=500)
                        conn.commit()
                        total_rows += len(rows)
                        success_count += 1

                if idx % 100 == 0:
                    logger.info(f"  progress {idx}/{len(to_fetch)} / rows={total_rows} / success={success_count} / fail={fail_count}")

        logger.info(f"✅ DONE: rows={total_rows} / success={success_count} / skip={skip_count} / fail={fail_count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
