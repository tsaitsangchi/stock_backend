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
fetch_derivative_data.py v3.0 — 期貨/選擇權日成交（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：期貨與選擇權每一天成交資料獨立原子 commit。
  ★ 全面整合 FailureLogger：精準追蹤 TX (台指期)、TXO (台指選) 等商品的更新狀況。
  ★ 韌性強化：在 API 回傳為空時改用 fallback 商品清單，確保關鍵行情不漏抓。
"""

from core.finmind_client import finmind_get
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DATASET_START = {"futures_daily": "1998-07-01", "option_daily": "2001-12-01"}
DEFAULT_END = date.today().strftime("%Y-%m-%d")
FALLBACK_FUTURES_IDS = ["TX", "MTX", "TE", "TF"]
FALLBACK_OPTIONS_IDS = ["TXO", "TEO", "TFO"]

DDL_FUTURES = """CREATE TABLE IF NOT EXISTS futures_ohlcv (date DATE, futures_id VARCHAR(50), contract_date VARCHAR(6), open NUMERIC(10,4), max NUMERIC(10,4), min NUMERIC(10,4), close NUMERIC(10,4), spread NUMERIC(10,4), spread_per NUMERIC(5,2), volume BIGINT, settlement_price NUMERIC(10,4), open_interest BIGINT, trading_session VARCHAR(20), PRIMARY KEY (date, futures_id, contract_date, trading_session));"""
DDL_OPTIONS = """CREATE TABLE IF NOT EXISTS options_ohlcv (date DATE, option_id VARCHAR(50), contract_date VARCHAR(6), strike_price NUMERIC(10,4), call_put VARCHAR(4), open NUMERIC(10,4), max NUMERIC(10,4), min NUMERIC(10,4), close NUMERIC(10,4), volume BIGINT, settlement_price NUMERIC(10,4), open_interest BIGINT, trading_session VARCHAR(20), PRIMARY KEY (date, option_id, contract_date, strike_price, call_put, trading_session));"""

UPSERT_FUTURES = """INSERT INTO futures_ohlcv (date, futures_id, contract_date, open, max, min, close, spread, spread_per, volume, settlement_price, open_interest, trading_session) VALUES %s ON CONFLICT (date, futures_id, contract_date, trading_session) DO UPDATE SET close = EXCLUDED.close;"""
UPSERT_OPTIONS = """INSERT INTO options_ohlcv (date, option_id, contract_date, strike_price, call_put, open, max, min, close, volume, settlement_price, open_interest, trading_session) VALUES %s ON CONFLICT (date, option_id, contract_date, strike_price, call_put, trading_session) DO UPDATE SET close = EXCLUDED.close;"""

def map_fut(r): return (r["date"], r.get("futures_id"), str(r.get("contract_date", ""))[:6], safe_float(r.get("open")), safe_float(r.get("max")), safe_float(r.get("min")), safe_float(r.get("close")), safe_float(r.get("spread")), safe_float(r.get("spread_per")), safe_int(r.get("volume")), safe_float(r.get("settlement_price")), safe_int(r.get("open_interest")), str(r.get("trading_session", "") or "")[:20])
def map_opt(r): return (r["date"], r.get("option_id"), str(r.get("contract_date", ""))[:6], safe_float(r.get("strike_price")), str(r.get("call_put", ""))[:4], safe_float(r.get("open")), safe_float(r.get("max")), safe_float(r.get("min")), safe_float(r.get("close")), safe_int(r.get("volume")), safe_float(r.get("settlement_price")), safe_int(r.get("open_interest")), str(r.get("trading_session", "") or "")[:20])

def fetch_derivative(conn, dataset, table, ddl, upsert_sql, mapper, ids, start, end, delay, force):
    ensure_ddl(conn, ddl)
    # 確保使用正確的 ID 欄位名
    id_col = "futures_id" if "futures" in table else "option_id"
    latest = get_all_safe_starts(conn, table, key_col=id_col)
    flog = FailureLogger(table, db_conn=conn)
    total_rows = 0
    tmpl = "(%s::date, %s, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s, %s::numeric, %s, %s)" if "futures" in table else "(%s::date, %s, %s, %s::numeric, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s, %s::numeric, %s, %s)"
    
    for iid in ids:
        s = resolve_start_cached(iid, latest, start, DATASET_START.get(dataset, "2000-01-01"), force)
        if not s: continue
        try:
            data = finmind_get(dataset, {"data_id": iid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [mapper(r) for r in data]
                if "futures" in table:
                    rows = dedup_rows(rows, (0, 1, 2, 12))
                else:
                    rows = dedup_rows(rows, (0, 1, 2, 3, 4, 12))
                
                res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, label_prefix=table, failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=iid, error=str(e))
    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
    flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["futures_ohlcv", "options_ohlcv", "all"], default=["all"])
    p.add_argument("--ids", nargs="+")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["futures_ohlcv", "options_ohlcv"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        if "futures_ohlcv" in tables: 
            fetch_derivative(conn, "TaiwanFuturesDaily", "futures_ohlcv", DDL_FUTURES, UPSERT_FUTURES, map_fut, args.ids or FALLBACK_FUTURES_IDS, args.start, args.end, args.delay, args.force)
        if "options_ohlcv" in tables: 
            fetch_derivative(conn, "TaiwanOptionDaily", "options_ohlcv", DDL_OPTIONS, UPSERT_OPTIONS, map_opt, args.ids or FALLBACK_OPTIONS_IDS, args.start, args.end, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()