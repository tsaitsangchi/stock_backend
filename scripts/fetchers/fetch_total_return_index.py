import sys
import logging
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

# ── sys.path 自我修復 ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_total_return_index.py v3.1 — 台股報酬指數（監控整合標準版）
================================================================================
v3.1 重大改進：
  · 整合 fetch_log v3.1：記錄 TAIEX (大盤) 與 TPEx (櫃買) 報酬指數抓取狀態。
  · 效能監控：精準追蹤 API 請求與資料處理的耗時（duration_ms）。
  · 狀態追蹤：支援 success, failed, no_new_data, skipped 等標準化狀態。
  · CLI 標準化：導入 argparse，支援 --force, --start, --end 等標準參數。

執行範例（常規）：
    python scripts/fetchers/fetch_total_return_index.py                # 增量抓取 TAIEX/TPEx 報酬指數

執行範例（強制重抓）：
    python scripts/fetchers/fetch_total_return_index.py --force        # 強制從起始日重刷
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATASET_START = "2010-01-01"
DEFAULT_END = date.today().strftime("%Y-%m-%d")
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
DDL_TOTAL_RETURN = """CREATE TABLE IF NOT EXISTS total_return_index (date DATE, stock_id VARCHAR(50), price NUMERIC(20,4), PRIMARY KEY (date, stock_id));"""
UPSERT_TOTAL_RETURN = """INSERT INTO total_return_index (date, stock_id, price) VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET price = EXCLUDED.price;"""

def map_tr(r): 
    return (r["date"], r["stock_id"], safe_float(r.get("price")))

def main():
    p = argparse.ArgumentParser(description="台股報酬指數抓取 (v3.1 — 監控整合標準版)")
    p.add_argument("--start", default=DATASET_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_TOTAL_RETURN)
        latest = get_all_safe_starts(conn, "total_return_index")
        flog = FailureLogger("total_return_index", db_conn=conn)
        total_rows = 0
        
        for data_id in ["TAIEX", "TPEx"]:
            s = resolve_start_cached(data_id, latest, args.start, DATASET_START, args.force)
            if not s:
                _write_fetch_log(conn, "total_return_index", data_id, "skipped", error_message="up_to_date")
                continue
            
            start_time = time.time()
            try:
                data = finmind_get("TaiwanStockTotalReturnIndex", {"data_id": data_id, "start_date": s, "end_date": args.end}, args.delay)
                duration_ms = int((time.time() - start_time) * 1000)
                if data:
                    rows = [map_tr(r) for r in data]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, UPSERT_TOTAL_RETURN, rows, "(%s, %s, %s::numeric)", label_prefix="total_return_index", failure_logger=flog)
                    n = sum(res.values())
                    total_rows += n
                    _write_fetch_log(conn, "total_return_index", data_id, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=args.end, duration_ms=duration_ms)
                else:
                    _write_fetch_log(conn, "total_return_index", data_id, "no_new_data", fetch_date_from=s, fetch_date_to=args.end, duration_ms=duration_ms)
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                flog.record(stock_id=data_id, error=str(e))
                _write_fetch_log(conn, "total_return_index", data_id, "failed", fetch_date_from=s, fetch_date_to=args.end, duration_ms=duration_ms, error_message=str(e))
                
        logger.info(f"  [total_return_index] 總共寫入 {total_rows} 筆")
        flog.summary()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
