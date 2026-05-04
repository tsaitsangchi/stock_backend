import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_technical_data.py  v3.0（逐支逐日 commit 完整性版）
================================================
從 FinMind API 抓取技術面資料並寫入 PostgreSQL：
  - stock_price ← TaiwanStockPrice
  - stock_per   ← TaiwanStockPER

v3.0 改進（極致韌性）：
  · 導入 commit_per_stock_per_day()：每一支股票、每一天的資料獨立 commit。
  · 全面整合 FailureLogger：精準追蹤每一筆失敗原因。
  · 優化批次與逐支模式：無論哪種模式，寫入皆維持最細粒度原子性。
"""

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date, timedelta, datetime

from core.finmind_client import (
    finmind_get as _core_finmind_get,
    BatchNotSupportedError,
)
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    safe_int,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
)

def finmind_get(dataset: str, params: dict, delay: float, **kwargs) -> list:
    return _core_finmind_get(dataset, params, delay=delay, raise_on_batch_400=True, **kwargs)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_START_DATES = {
    "stock_price": "1994-10-01",
    "stock_per":   "2005-10-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1994-10-01"
DEFAULT_CHUNK_DAYS    = 90
DEFAULT_BATCH_THRESHOLD = 20
BATCH_RETURN_WARNING_THRESHOLD = 5000

# ──────────────────────────────────────────────
# DDL & SQL
# ──────────────────────────────────────────────
DDL_STOCK_PRICE = """
CREATE TABLE IF NOT EXISTS stock_price (
    date             DATE,
    stock_id         VARCHAR(50),
    trading_volume   BIGINT,
    trading_money    BIGINT,
    open             NUMERIC(10,4),
    max              NUMERIC(10,4),
    min              NUMERIC(10,4),
    close            NUMERIC(10,4),
    spread           NUMERIC(10,4),
    trading_turnover INTEGER,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_stock_price_stock_id ON stock_price (stock_id);
"""

DDL_STOCK_PER = """
CREATE TABLE IF NOT EXISTS stock_per (
    date           DATE,
    stock_id       VARCHAR(50),
    dividend_yield NUMERIC(10,4),
    per            NUMERIC(10,4),
    pbr            NUMERIC(10,4),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_stock_per_stock_id ON stock_per (stock_id);
"""

UPSERT_STOCK_PRICE = """
INSERT INTO stock_price
    (date, stock_id, trading_volume, trading_money, open, max, min, close, spread, trading_turnover)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    trading_volume   = EXCLUDED.trading_volume,
    trading_money    = EXCLUDED.trading_money,
    open             = EXCLUDED.open,
    max              = EXCLUDED.max,
    min              = EXCLUDED.min,
    close            = EXCLUDED.close,
    spread           = EXCLUDED.spread,
    trading_turnover = EXCLUDED.trading_turnover;
"""

UPSERT_STOCK_PER = """
INSERT INTO stock_per (date, stock_id, dividend_yield, per, pbr)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    dividend_yield = EXCLUDED.dividend_yield,
    per            = EXCLUDED.per,
    pbr            = EXCLUDED.pbr;
"""

# ──────────────────────────────────────────────
# Row Mappers
# ──────────────────────────────────────────────
def map_price_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_int(r.get("Trading_Volume")),
        safe_int(r.get("Trading_money")),
        safe_float(r.get("open")),
        safe_float(r.get("max")),
        safe_float(r.get("min")),
        safe_float(r.get("close")),
        safe_float(r.get("spread")),
        safe_int(r.get("Trading_turnover")),
    )

def map_per_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_float(r.get("dividend_yield")),
        safe_float(r.get("PER")),
        safe_float(r.get("PBR")),
    )

# ──────────────────────────────────────────────
# 資料抓取與寫入核心
# ──────────────────────────────────────────────
def fetch_dataset_batch(
    conn, dataset: str, table: str, upsert_sql: str, template: str, row_mapper,
    start_date: str, end_date: str, delay: float, force: bool,
    chunk_days: int, batch_threshold: int, valid_stock_ids: set, latest_dates: dict
):
    groups = defaultdict(list)
    for sid in valid_stock_ids:
        s = resolve_start_cached(sid, latest_dates, start_date, DATASET_START_DATES[table], force)
        if s: groups[s].append(sid)

    if not groups:
        logger.info(f"  [{table}] 資料皆已是最新，跳過。")
        return

    total_api = total_rows = 0
    flog = FailureLogger(table, db_conn=conn)
    batch_disabled = False

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        if len(sids) >= batch_threshold and not batch_disabled:
            # ── 批次模式 ──
            seg_start = group_start
            seg_end_limit = datetime.strptime(end_date, "%Y-%m-%d")
            
            chunk_rows = []
            try:
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_limit: break
                    seg_end = min((seg_start_dt + timedelta(days=chunk_days-1)).strftime("%Y-%m-%d"), end_date)

                    logger.info(f"  [{table}] 批次請求 {seg_start}~{seg_end}（{len(sids)} 支）")
                    data = finmind_get(dataset, {"start_date": seg_start, "end_date": seg_end}, delay)
                    total_api += 1

                    for r in data:
                        if r.get("stock_id") in sids_set:
                            try: chunk_rows.append(row_mapper(r))
                            except Exception: pass

                    seg_start = (datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            except BatchNotSupportedError:
                logger.warning(f"  [{table}] Fallback 為逐支模式。")
                batch_disabled = True
                chunk_rows = []

            if chunk_rows:
                chunk_rows = dedup_rows(chunk_rows, (0, 1))
                res = commit_per_stock_per_day(conn, upsert_sql, chunk_rows, template, label_prefix=table, failure_logger=flog)
                total_rows += sum(res.values())

        if len(sids) < batch_threshold or batch_disabled:
            # ── 逐支模式 ──
            for sid in sids:
                try:
                    data = finmind_get(dataset, {"data_id": sid, "start_date": group_start, "end_date": end_date}, delay)
                    total_api += 1
                    if not data: continue
                    rows = [row_mapper(r) for r in data]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table, failure_logger=flog)
                    total_rows += sum(res.values())
                except Exception as e:
                    flog.record(stock_id=sid, error=str(e))

    flog.summary()
    logger.info(f"[{table}] 完成  API：{total_api}  寫入：{total_rows}  失敗：{len(flog)}")

def fetch_both_per_stock(start_date, end_date, delay, force, tables, stock_id=None):
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_STOCK_PRICE, DDL_STOCK_PER)
        stock_ids = [s.strip() for s in stock_id.split(",")] if stock_id else list(get_all_safe_starts(conn, "stock_price").keys()) or ["2330"]
        
        latest_price = get_all_safe_starts(conn, "stock_price") if "stock_price" in tables else {}
        latest_per = get_all_safe_starts(conn, "stock_per") if "stock_per" in tables else {}
        
        flog_price = FailureLogger("stock_price", db_conn=conn)
        flog_per = FailureLogger("stock_per", db_conn=conn)

        for sid in stock_ids:
            if "stock_price" in tables:
                s = resolve_start_cached(sid, latest_price, start_date, DATASET_START_DATES["stock_price"], force)
                if s:
                    try:
                        data = finmind_get("TaiwanStockPrice", {"data_id": sid, "start_date": s, "end_date": end_date}, delay)
                        if data:
                            rows = [map_price_row(r) for r in data]
                            commit_per_stock_per_day(conn, UPSERT_STOCK_PRICE, rows, "(%s::date,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)", label_prefix="stock_price", failure_logger=flog_price)
                    except Exception as e: flog_price.record(stock_id=sid, error=str(e))

            if "stock_per" in tables:
                s = resolve_start_cached(sid, latest_per, start_date, DATASET_START_DATES["stock_per"], force)
                if s:
                    try:
                        data = finmind_get("TaiwanStockPER", {"data_id": sid, "start_date": s, "end_date": end_date}, delay)
                        if data:
                            rows = [map_per_row(r) for r in data]
                            commit_per_stock_per_day(conn, UPSERT_STOCK_PER, rows, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric)", label_prefix="stock_per", failure_logger=flog_per)
                    except Exception as e: flog_per.record(stock_id=sid, error=str(e))
        
        flog_price.summary()
        flog_per.summary()
        logger.info(f"逐支抓取結束。")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="FinMind 技術面資料抓取工具 v3.0")
    parser.add_argument("--tables", nargs="+", choices=["stock_price", "stock_per", "all"], default=["all"])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--per-stock", action="store_true")
    parser.add_argument("--batch-threshold", type=int, default=DEFAULT_BATCH_THRESHOLD)
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--stock-id", default=None)
    args = parser.parse_args()

    tables = ["stock_price", "stock_per"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        if args.per_stock:
            fetch_both_per_stock(args.start, args.end, args.delay, args.force, tables, args.stock_id)
        else:
            ensure_ddl(conn, DDL_STOCK_PRICE, DDL_STOCK_PER)
            stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else None
            if not stock_ids:
                from core.db_utils import get_db_stock_ids
                stock_ids = get_db_stock_ids(conn)
            
            valid_set = set(stock_ids)
            if "stock_price" in tables:
                fetch_dataset_batch(conn, "TaiwanStockPrice", "stock_price", UPSERT_STOCK_PRICE, "(%s::date,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)", map_price_row, args.start, args.end, args.delay, args.force, args.chunk_days, args.batch_threshold, valid_set, get_all_safe_starts(conn, "stock_price"))
            if "stock_per" in tables:
                fetch_dataset_batch(conn, "TaiwanStockPER", "stock_per", UPSERT_STOCK_PER, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric)", map_per_row, args.start, args.end, args.delay, args.force, args.chunk_days, args.batch_threshold, valid_set, get_all_safe_starts(conn, "stock_per"))
    finally:
        conn.close()

if __name__ == "__main__":
    main()
