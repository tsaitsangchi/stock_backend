from __future__ import annotations
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
fetch_price_adj_data.py v3.0 — 還原股價 + 當沖交易 + 漲跌停價（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：每一對 (sid, date) 獨立原子寫入。
  ★ 全面整合 FailureLogger：精準捕捉並彙整所有抓取/寫入失敗。
  ★ 結構優化：移除冗餘實作，全面對接 core v3.0 標準。
"""

from core.finmind_client import finmind_get, BatchNotSupportedError
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    safe_int,
    get_all_safe_starts,
    get_db_stock_ids,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
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
    "price_adj":   "1994-10-01",
    "day_trading": "2014-01-01",
    "price_limit": "2000-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

DEFAULT_CHUNK_DAYS              = 90
DEFAULT_BATCH_THRESHOLD         = 20
BATCH_RETURN_WARNING_THRESHOLD  = 5000

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_PRICE_ADJ = """
CREATE TABLE IF NOT EXISTS price_adj (
    date             DATE,
    stock_id         VARCHAR(50),
    trading_volume   BIGINT,
    trading_money    BIGINT,
    open             NUMERIC(20,4),
    max              NUMERIC(20,4),
    min              NUMERIC(20,4),
    close            NUMERIC(20,4),
    spread           NUMERIC(20,4),
    trading_turnover NUMERIC(20,4),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_price_adj_stock ON price_adj (stock_id, date);
"""

DDL_DAY_TRADING = """
CREATE TABLE IF NOT EXISTS day_trading (
    date            DATE,
    stock_id        VARCHAR(50),
    buy_after_sale  VARCHAR(20),
    volume          BIGINT,
    buy_amount      BIGINT,
    sell_amount     BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_day_trading_stock ON day_trading (stock_id, date);
"""

DDL_PRICE_LIMIT = """
CREATE TABLE IF NOT EXISTS price_limit (
    date            DATE,
    stock_id        VARCHAR(50),
    reference_price NUMERIC(20,4),
    limit_up        NUMERIC(20,4),
    limit_down      NUMERIC(20,4),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_price_limit_stock ON price_limit (stock_id, date);
"""

UPSERT_PRICE_ADJ = """
INSERT INTO price_adj
    (date, stock_id, trading_volume, trading_money, open, max, min, close, spread, trading_turnover)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET
    trading_volume = EXCLUDED.trading_volume, close = EXCLUDED.close;
"""

UPSERT_DAY_TRADING = """
INSERT INTO day_trading (date, stock_id, buy_after_sale, volume, buy_amount, sell_amount)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET
    volume = EXCLUDED.volume, buy_amount = EXCLUDED.buy_amount;
"""

UPSERT_PRICE_LIMIT = """
INSERT INTO price_limit (date, stock_id, reference_price, limit_up, limit_down)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET
    reference_price = EXCLUDED.reference_price, limit_up = EXCLUDED.limit_up;
"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_price_adj(r: dict) -> tuple:
    return (r["date"], r["stock_id"], safe_int(r.get("Trading_Volume")), safe_int(r.get("Trading_money")), safe_float(r.get("open")), safe_float(r.get("max")), safe_float(r.get("min")), safe_float(r.get("close")), safe_float(r.get("spread")), safe_float(r.get("Trading_turnover")))

def map_day_trading(r: dict) -> tuple:
    return (r["date"], r["stock_id"], str(r.get("BuyAfterSale", "")), safe_int(r.get("Volume")), safe_int(r.get("BuyAmount")), safe_int(r.get("SellAmount")))

def map_price_limit(r: dict) -> tuple:
    return (r["date"], r["stock_id"], safe_float(r.get("reference_price")), safe_float(r.get("limit_up")), safe_float(r.get("limit_down")))

# ─────────────────────────────────────────────
# Core Loop
# ─────────────────────────────────────────────
def fetch_dataset_unified(conn, dataset: str, table: str, ddl: str, upsert_sql: str, template: str, mapper, dataset_key: str, stock_ids: list[str], start: str, end: str, delay: float, force: bool, use_batch: bool, batch_threshold: int, chunk_days: int):
    ensure_ddl(conn, ddl)
    latest_dates = get_all_safe_starts(conn, table)
    flog = FailureLogger(table, db_conn=conn)
    
    # 起始日分組
    groups = defaultdict(list)
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest_dates, start, DATASET_START[dataset_key], force)
        if s: groups[s].append(sid)

    if not groups:
        logger.info(f"[{table}] 資料已是最新。")
        return

    total_api = total_rows = 0
    batch_disabled = False

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        if use_batch and len(sids) >= batch_threshold and not batch_disabled:
            # ── 批次模式 ──
            seg_start = group_start
            seg_end_limit = datetime.strptime(end, "%Y-%m-%d")
            chunk_data = []
            try:
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_limit: break
                    seg_end = min((seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"), end)
                    logger.info(f"  [{table}] 批次 {seg_start}~{seg_end}（{len(sids)} 支）")
                    data = finmind_get(dataset, {"start_date": seg_start, "end_date": seg_end}, delay, raise_on_batch_400=True)
                    total_api += 1
                    for r in data:
                        if r.get("stock_id") in sids_set:
                            try: chunk_data.append(mapper(r))
                            except Exception: pass
                    seg_start = (datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            except BatchNotSupportedError:
                logger.warning(f"  [{table}] Fallback 為逐支模式。")
                batch_disabled = True
                chunk_data = []

            if chunk_data:
                chunk_data = dedup_rows(chunk_data, (0, 1))
                res = commit_per_stock_per_day(conn, upsert_sql, chunk_data, template, label_prefix=table, failure_logger=flog)
                total_rows += sum(res.values())

        if (not use_batch) or len(sids) < batch_threshold or batch_disabled:
            # ── 逐支模式 ──
            for sid in sids:
                try:
                    data = finmind_get(dataset, {"data_id": sid, "start_date": group_start, "end_date": end}, delay)
                    total_api += 1
                    if not data: continue
                    rows = [mapper(r) for r in data]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table, failure_logger=flog)
                    total_rows += sum(res.values())
                except Exception as e:
                    flog.record(stock_id=sid, error=str(e))

    flog.summary()
    logger.info(f"[{table}] 完成  API：{total_api}  寫入：{total_rows}  失敗：{len(flog)}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["price_adj", "day_trading", "price_limit", "all"], default=["all"])
    p.add_argument("--stock-id", type=str, default=None)
    p.add_argument("--start", type=str, default="1994-10-01")
    p.add_argument("--end",   type=str, default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--per-stock", action="store_true")
    p.add_argument("--batch-threshold", type=int, default=DEFAULT_BATCH_THRESHOLD)
    p.add_argument("--chunk-days",      type=int, default=DEFAULT_CHUNK_DAYS)
    args = p.parse_args()

    tables = ["price_adj", "day_trading", "price_limit"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            stock_ids = get_db_stock_ids(conn, types=("twse", "otc"))

        logger.info(f"目標股票：{len(stock_ids)} 支 | 模式：{'逐支' if args.per_stock else '批次'}")

        configs = {
            "price_adj":   ("TaiwanStockPriceAdj", DDL_PRICE_ADJ, UPSERT_PRICE_ADJ, "(%s::date,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric)", map_price_adj),
            "day_trading": ("TaiwanStockDayTrading", DDL_DAY_TRADING, UPSERT_DAY_TRADING, "(%s::date,%s,%s,%s,%s,%s)", map_day_trading),
            "price_limit": ("TaiwanStockPriceLimit", DDL_PRICE_LIMIT, UPSERT_PRICE_LIMIT, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric)", map_price_limit),
        }
        for key in tables:
            dataset, ddl, upsert, tmpl, mapper = configs[key]
            fetch_dataset_unified(conn, dataset, key, ddl, upsert, tmpl, mapper, key, stock_ids, args.start, args.end, args.delay, args.force, not args.per_stock, args.batch_threshold, args.chunk_days)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
