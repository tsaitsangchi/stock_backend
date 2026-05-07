import sys
import logging
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import date, timedelta, datetime
import argparse

# ── sys.path 自我修復 ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_technical_data.py v3.1 — 技術面資料（監控整合標準版）
================================================================================
v3.1 重大改進：
  · 整合 fetch_log v3.1：記錄股價（stock_price）與本益比（stock_per）任務狀態。
  · 效能監控：精準追蹤 API 請求與資料處理的耗時（duration_ms）。
  · 狀態追蹤：支援 success, failed, no_new_data, skipped 等標準化狀態。
  · 彈性模式：支援全市場批次抓取與特定標的逐支抓取。

執行範例（常規）：
    python scripts/fetchers/fetch_technical_data.py                # 抓取所有標的技術面資料（批次模式）
    python scripts/fetchers/fetch_technical_data.py --tables stock_per # 僅抓取 PER/PBR

執行範例（強制重抓）：
    # 強制重抓特定個股的所有技術面資料
    python scripts/fetchers/fetch_technical_data.py --stock-id 2330 --force --tables all
"""

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

DATASET_START_DATES = {
    "stock_price": "1994-10-01",
    "stock_per":   "2005-10-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1994-10-01"
DEFAULT_CHUNK_DAYS    = 90
DEFAULT_BATCH_THRESHOLD = 20
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
        if s: 
            groups[s].append(sid)
        else:
            _write_fetch_log(conn, table, sid, "skipped", error_message="up_to_date")

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
            
            try:
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_limit: break
                    seg_end = min((seg_start_dt + timedelta(days=chunk_days-1)).strftime("%Y-%m-%d"), end_date)

                    logger.info(f"  [{table}] 批次請求 {seg_start}~{seg_end}（{len(sids)} 支）")
                    start_time = time.time()
                    data = finmind_get(dataset, {"start_date": seg_start, "end_date": seg_end}, delay)
                    duration_ms = int((time.time() - start_time) * 1000)
                    total_api += 1

                    chunk_rows = []
                    actual_sids_in_chunk = set()
                    for r in data:
                        sid = r.get("stock_id")
                        if sid in sids_set:
                            try: 
                                chunk_rows.append(row_mapper(r))
                                actual_sids_in_chunk.add(sid)
                            except Exception: pass

                    if chunk_rows:
                        chunk_rows = dedup_rows(chunk_rows, (0, 1))
                        res = commit_per_stock_per_day(conn, upsert_sql, chunk_rows, template, label_prefix=table, failure_logger=flog)
                        n = sum(res.values())
                        total_rows += n
                        for sid in actual_sids_in_chunk:
                            # 注意：批次模式下，無法精確區分每支股票的寫入筆數，統一標註任務成功
                            _write_fetch_log(conn, table, sid, "success", fetch_date_from=seg_start, fetch_date_to=seg_end, duration_ms=duration_ms)
                    else:
                        for sid in sids:
                            _write_fetch_log(conn, table, sid, "no_new_data", fetch_date_from=seg_start, fetch_date_to=seg_end, duration_ms=duration_ms)

                    seg_start = (datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            except BatchNotSupportedError:
                logger.warning(f"  [{table}] Fallback 為逐支模式。")
                batch_disabled = True
            except Exception as e:
                logger.error(f"  [{table}] 批次抓取發生錯誤: {e}")
                for sid in sids:
                    _write_fetch_log(conn, table, sid, "failed", error_message=str(e))

        if len(sids) < batch_threshold or batch_disabled:
            # ── 逐支模式 ──
            for sid in sids:
                start_time = time.time()
                try:
                    data = finmind_get(dataset, {"data_id": sid, "start_date": group_start, "end_date": end_date}, delay)
                    duration_ms = int((time.time() - start_time) * 1000)
                    total_api += 1
                    if not data:
                        _write_fetch_log(conn, table, sid, "no_new_data", fetch_date_from=group_start, fetch_date_to=end_date, duration_ms=duration_ms)
                        continue
                    rows = [row_mapper(r) for r in data]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table, failure_logger=flog)
                    n = sum(res.values())
                    total_rows += n
                    _write_fetch_log(conn, table, sid, "success", rows_inserted=n, fetch_date_from=group_start, fetch_date_to=end_date, duration_ms=duration_ms)
                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    flog.record(stock_id=sid, error=str(e))
                    _write_fetch_log(conn, table, sid, "failed", fetch_date_from=group_start, fetch_date_to=end_date, duration_ms=duration_ms, error_message=str(e))

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
                if not s:
                    _write_fetch_log(conn, "stock_price", sid, "skipped", error_message="up_to_date")
                else:
                    start_time = time.time()
                    try:
                        data = finmind_get("TaiwanStockPrice", {"data_id": sid, "start_date": s, "end_date": end_date}, delay)
                        duration_ms = int((time.time() - start_time) * 1000)
                        if data:
                            rows = [map_price_row(r) for r in data]
                            res = commit_per_stock_per_day(conn, UPSERT_STOCK_PRICE, rows, "(%s::date,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)", label_prefix="stock_price", failure_logger=flog_price)
                            _write_fetch_log(conn, "stock_price", sid, "success", rows_inserted=sum(res.values()), fetch_date_from=s, fetch_date_to=end_date, duration_ms=duration_ms)
                        else:
                            _write_fetch_log(conn, "stock_price", sid, "no_new_data", fetch_date_from=s, fetch_date_to=end_date, duration_ms=duration_ms)
                    except Exception as e:
                        duration_ms = int((time.time() - start_time) * 1000)
                        flog_price.record(stock_id=sid, error=str(e))
                        _write_fetch_log(conn, "stock_price", sid, "failed", fetch_date_from=s, fetch_date_to=end_date, duration_ms=duration_ms, error_message=str(e))

            if "stock_per" in tables:
                s = resolve_start_cached(sid, latest_per, start_date, DATASET_START_DATES["stock_per"], force)
                if not s:
                    _write_fetch_log(conn, "stock_per", sid, "skipped", error_message="up_to_date")
                else:
                    start_time = time.time()
                    try:
                        data = finmind_get("TaiwanStockPER", {"data_id": sid, "start_date": s, "end_date": end_date}, delay)
                        duration_ms = int((time.time() - start_time) * 1000)
                        if data:
                            rows = [map_per_row(r) for r in data]
                            res = commit_per_stock_per_day(conn, UPSERT_STOCK_PER, rows, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric)", label_prefix="stock_per", failure_logger=flog_per)
                            _write_fetch_log(conn, "stock_per", sid, "success", rows_inserted=sum(res.values()), fetch_date_from=s, fetch_date_to=end_date, duration_ms=duration_ms)
                        else:
                            _write_fetch_log(conn, "stock_per", sid, "no_new_data", fetch_date_from=s, fetch_date_to=end_date, duration_ms=duration_ms)
                    except Exception as e:
                        duration_ms = int((time.time() - start_time) * 1000)
                        flog_per.record(stock_id=sid, error=str(e))
                        _write_fetch_log(conn, "stock_per", sid, "failed", fetch_date_from=s, fetch_date_to=end_date, duration_ms=duration_ms, error_message=str(e))
        
        flog_price.summary()
        flog_per.summary()
        logger.info(f"逐支抓取結束。")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="技術面資料抓取 (v3.1 — 監控整合標準版)")
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
