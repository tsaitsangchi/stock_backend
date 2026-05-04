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
fetch_fred_data.py v3.0 — FRED 全球宏觀資料（逐 series 逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：將 series_id 視為股票代號，實現每一天資料獨立 commit。
  ★ 全面整合 FailureLogger：精準追蹤美債殖利率、VIX、M2 等關鍵總經指標的更新狀況。
  ★ 韌性強化：在 API 回傳 NA 或點號時正確過濾，確保資料庫數值精度。
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
    p.add_argument("--series", nargs="+", default=DEFAULT_FRED_SERIES)
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
        for sid in args.series:
            s = resolve_start_cached(sid, latest, args.start, "1990-01-01", args.force)
            if not s: continue
            try:
                obs = fred_get(sid, api_key, s, args.end)
                if obs:
                    rows = []
                    for o in obs:
                        v = safe_float(o.get("value")) if o.get("value") != "." else None
                        if v is not None: rows.append((sid, o.get("date"), v))
                    if rows:
                        rows = dedup_rows(rows, (0, 1))
                        res = commit_per_stock_per_day(conn, UPSERT_FRED, rows, "(%s, %s, %s)", stock_index=0, date_index=1, label_prefix=sid, failure_logger=flog)
                        total_rows += sum(res.values())
                time.sleep(args.delay)
            except Exception as e: flog.record(stock_id=sid, error=str(e))
        logger.info(f"  [fred_series] 總共寫入 {total_rows} 筆")
        flog.summary()
    finally:
        conn.close()

if __name__ == "__main__":
    main()