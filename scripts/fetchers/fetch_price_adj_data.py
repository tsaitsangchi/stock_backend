import argparse
import time
import json
import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta

# ── sys.path 自我修復 ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_price_adj_data.py v3.1 — 還原股價 + 當沖交易 + 漲跌停價（監控整合標準版）
================================================================================
v3.1 重大改進：
  · 整合 fetch_log v3.1：支援「批次」與「逐支」模式日誌，精準記錄 160 支核心標的抓取狀況。
  · 效能監控：記錄批次 chunk 或單一請求的 API 耗時（duration_ms）。
  · 狀態追蹤：支援 success, failed, no_new_data, skipped 等標準化狀態。
  · 彈性架構：維持原有的 Batch/Per-stock 自動 fallback 機制，並強化錯誤回報。

執行範例（常規）：
    python scripts/fetchers/fetch_price_adj_data.py                # 抓取 160 支核心標的所有資料（批次模式）
    python scripts/fetchers/fetch_price_adj_data.py --per-stock    # 強制使用逐支抓取模式
    python scripts/fetchers/fetch_price_adj_data.py --stock-id 2330 # 僅抓取台積電

執行範例（強制重抓）：
    python scripts/fetchers/fetch_price_adj_data.py --force --tables price_adj
    python scripts/fetchers/fetch_price_adj_data.py --start 2024-01-01 --force
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

DATASET_START = {
    "price_adj":   "1994-10-01",
    "day_trading": "2014-01-01",
    "price_limit": "2000-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

DEFAULT_CHUNK_DAYS              = 90
DEFAULT_BATCH_THRESHOLD         = 20
_CLI_ARGS_STR = " ".join(sys.argv)

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None):
    """v3.1 標準化日誌寫入"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (
                    run_ts, table_name, stock_id, status, rows_inserted, 
                    fetch_date_from, fetch_date_to, duration_ms, error_message, cli_args
                ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (table_name, stock_id, status, rows_inserted, fetch_date_from, fetch_date_to, duration_ms, error_message, _CLI_ARGS_STR))
        conn.commit()
    except Exception as e:
        logger.warning(f"無法寫入 fetch_log: {e}")

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
    logger.info(f"=== [{table}] 開始 ===")
    ensure_ddl(conn, ddl)
    latest_dates = get_all_safe_starts(conn, table)
    flog = FailureLogger(table, db_conn=conn)
    
    # 起始日分組
    groups = defaultdict(list)
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest_dates, start, DATASET_START[dataset_key], force)
        if s: 
            groups[s].append(sid)
        else:
            _write_fetch_log(conn, table, sid, "skipped", error_message="up_to_date")

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
            
            try:
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_limit: break
                    seg_end = min((seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"), end)
                    
                    logger.info(f"  [{table}] 批次 {seg_start}~{seg_end}（{len(sids)} 支）")
                    start_time = time.time()
                    data = finmind_get(dataset, {"start_date": seg_start, "end_date": seg_end}, delay, raise_on_batch_400=True)
                    duration_ms = int((time.time() - start_time) * 1000)
                    total_api += 1
                    
                    chunk_data = []
                    actual_sids_in_chunk = set()
                    if data:
                        for r in data:
                            if r.get("stock_id") in sids_set:
                                try: 
                                    chunk_data.append(mapper(r))
                                    actual_sids_in_chunk.add(r.get("stock_id"))
                                except Exception: pass
                    
                    if chunk_data:
                        chunk_data = dedup_rows(chunk_data, (0, 1))
                        res = commit_per_stock_per_day(conn, upsert_sql, chunk_data, template, label_prefix=table, failure_logger=flog)
                        n = sum(res.values())
                        total_rows += n
                        # 批次成功，為參與標的記錄日誌（此處採簡化：記錄整批成功標的）
                        for sid in actual_sids_in_chunk:
                            _write_fetch_log(conn, table, sid, "success", rows_inserted=0, fetch_date_from=seg_start, fetch_date_to=seg_end, duration_ms=duration_ms)
                    else:
                        # 整批無資料
                        for sid in sids:
                            _write_fetch_log(conn, table, sid, "no_new_data", fetch_date_from=seg_start, fetch_date_to=seg_end, duration_ms=duration_ms)

                    seg_start = (datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            except BatchNotSupportedError:
                logger.warning(f"  [{table}] Fallback 為逐支模式。")
                batch_disabled = True

        if (not use_batch) or len(sids) < batch_threshold or batch_disabled:
            # ── 逐支模式 ──
            for sid in sids:
                start_time = time.time()
                try:
                    data = finmind_get(dataset, {"data_id": sid, "start_date": group_start, "end_date": end}, delay)
                    duration_ms = int((time.time() - start_time) * 1000)
                    total_api += 1
                    
                    if not data:
                        _write_fetch_log(conn, table, sid, "no_new_data", fetch_date_from=group_start, fetch_date_to=end, duration_ms=duration_ms)
                        continue
                    
                    rows = [mapper(r) for r in data]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table, failure_logger=flog)
                    n = sum(res.values())
                    total_rows += n
                    _write_fetch_log(conn, table, sid, "success", rows_inserted=n, fetch_date_from=group_start, fetch_date_to=end, duration_ms=duration_ms)
                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    flog.record(stock_id=sid, error=str(e))
                    _write_fetch_log(conn, table, sid, "failed", fetch_date_from=group_start, fetch_date_to=end, duration_ms=duration_ms, error_message=str(e))

    flog.summary()
    logger.info(f"[{table}] 完成  API：{total_api}  寫入：{total_rows}  失敗：{len(flog)}")

def main():
    p = argparse.ArgumentParser(description="還原股價與交易補強資料 (v3.1 — 監控整合標準版)")
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
