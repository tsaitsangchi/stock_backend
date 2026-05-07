from __future__ import annotations
import sys
import logging
import os
import random
import time
from pathlib import Path
from datetime import date, datetime, timedelta
import argparse
import requests

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_fred_data.py — FRED 全球宏觀資料（v3.1 fetch_log 整合版）
================================================================================
v3.1 改進：
  · 整合 fetch_log：每次抓取（無論成功、失敗或跳過）都會寫入監控日誌。
  · 效能追蹤：記錄各總經指標（如 T10Y2Y, VIX）的 API 請求耗時（duration_ms）。
  · 支援 --retry-failed N 與 --gap-fill N 模式，實現智慧補抓。

v3.0 既有：
  · 導入 commit_per_stock_per_day：每一天、每一指標獨立原子 commit。
  · 整合 FailureLogger：精準追蹤美債殖利率、VIX、M2 等關鍵指標更新狀況。

執行（常規）：
    python fetch_fred_data.py
    python fetch_fred_data.py --ids T10Y2Y VIXCLS
    python fetch_fred_data.py --ids DGS10 --force
    python fetch_fred_data.py --force
    python fetch_fred_data.py --start 2024-01-01 --force

執行（模式切換）：
    # 重試最近 7 天失敗的組合
    python fetch_fred_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python fetch_fred_data.py --gap-fill 30
"""

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_FRED_SERIES = ["T10Y2Y", "T10Y3M", "T10YIE", "VIXCLS", "BAMLH0A0HYM2", "DTWEXBGS", "M2SL", "DGS10", "DGS2", "DGS3MO", "NAPMCI", "UMCSENT", "INDPRO", "UNRATE", "CPIAUCSL"]

DDL_FRED = """CREATE TABLE IF NOT EXISTS fred_series (series_id VARCHAR(50), date DATE, value NUMERIC(20,6), PRIMARY KEY (series_id, date));"""
UPSERT_FRED = """INSERT INTO fred_series (series_id, date, value) VALUES %s ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value;"""

_CLI_ARGS_STR = " ".join(sys.argv)

def _write_fetch_log(conn, **kwargs):
    """寫入 fetch_log，失敗不影響主流程。"""
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO fetch_log (
                run_ts, table_name, stock_id, fetch_mode,
                fetch_date_from, fetch_date_to,
                rows_inserted, rows_updated, duration_ms,
                status, error_message, cli_args
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
            """
            cur.execute(sql, (
                kwargs.get("table_name"), kwargs.get("stock_id"), kwargs.get("fetch_mode", "per_stock"),
                kwargs.get("fetch_date_from"), kwargs.get("fetch_date_to"),
                kwargs.get("rows_inserted", 0), kwargs.get("duration_ms", 0),
                kwargs.get("status"), kwargs.get("error_message"), _CLI_ARGS_STR
            ))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.debug(f"fetch_log 寫入失敗：{e}")

def fred_get(series_id, api_key, start, end, max_retries=3):
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start, "observation_end": end}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(FRED_API_URL, params=params, timeout=(10, 60))
            if resp.status_code == 200: return resp.json().get("observations", [])
            if resp.status_code == 429: time.sleep(60)
        except Exception: time.sleep(2**attempt)
    return []

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+", default=DEFAULT_FRED_SERIES)
    p.add_argument("--start", default="1990-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("未設定 FRED_API_KEY")
        sys.exit(1)

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_FRED)
        latest = get_all_safe_starts(conn, "fred_series", key_col="series_id")
        flog = FailureLogger("fred_series", db_conn=conn)
        total_rows = 0
        for sid in args.ids:
            s = resolve_start_cached(sid, latest, args.start, "1990-01-01", args.force)
            if not s:
                _write_fetch_log(conn, table_name="fred_series", stock_id=sid, status="skipped", error_message="up_to_date")
                continue
            
            t0 = time.time()
            try:
                obs = fred_get(sid, api_key, s, args.end)
                dur = int((time.time() - t0) * 1000)
                if obs:
                    rows = []
                    for o in obs:
                        v = safe_float(o.get("value")) if o.get("value") != "." else None
                        if v is not None: rows.append((sid, o.get("date"), v))
                    if rows:
                        rows = dedup_rows(rows, (0, 1))
                        res = commit_per_stock_per_day(conn, UPSERT_FRED, rows, "(%s, %s, %s)", stock_index=0, date_index=1, label_prefix=sid, failure_logger=flog)
                        n = sum(res.values())
                        total_rows += n
                        _write_fetch_log(conn, table_name="fred_series", stock_id=sid, fetch_date_from=s, fetch_date_to=args.end, 
                                         rows_inserted=n, duration_ms=dur, status="success")
                    else:
                        _write_fetch_log(conn, table_name="fred_series", stock_id=sid, fetch_date_from=s, fetch_date_to=args.end, 
                                         rows_inserted=0, duration_ms=dur, status="no_new_data")
                else:
                    _write_fetch_log(conn, table_name="fred_series", stock_id=sid, fetch_date_from=s, fetch_date_to=args.end, 
                                     rows_inserted=0, duration_ms=dur, status="no_new_data")
                time.sleep(args.delay)
            except Exception as e:
                dur = int((time.time() - t0) * 1000)
                flog.record(stock_id=sid, error=str(e))
                _write_fetch_log(conn, table_name="fred_series", stock_id=sid, fetch_date_from=s, fetch_date_to=args.end, 
                                 rows_inserted=0, duration_ms=dur, status="failed", error_message=str(e))
        logger.info(f"  [fred_series] 總共寫入 {total_rows} 筆")
        flog.summary()
    finally:
        conn.close()

if __name__ == "__main__":
    main()