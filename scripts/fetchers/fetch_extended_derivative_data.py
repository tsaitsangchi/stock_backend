import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse
import time

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_extended_derivative_data.py — 期貨/選擇權 II（v3.1 fetch_log 整合版）
================================================================================
v3.1 改進：
  · 整合 fetch_log：每次抓取（無論成功、失敗或跳過）都會寫入監控日誌。
  · 效能追蹤：記錄三大法人期權部位的 API 請求與處理耗時（duration_ms）。
  · 支援 --retry-failed N 與 --gap-fill N 模式，實現智慧補抓。

v3.0 既有：
  · 支援 2 個資料表：futures_inst_investors, options_inst_investors。
  · 導入 commit_per_stock_per_day：每一天、每一法人部位獨立原子 commit。
  · 整合 FailureLogger：精準追蹤衍生品進階籌碼資料的抓取狀況。

執行（常規）：
    python fetch_extended_derivative_data.py
    python fetch_extended_derivative_data.py --tables futures_inst_investors
    python fetch_extended_derivative_data.py --ids TX,MTX,TXO --force
    python fetch_extended_derivative_data.py --ids TX,MTX,TXO --tables all --force
    python fetch_extended_derivative_data.py --tables all --force

執行（模式切換）：
    # 重試最近 7 天失敗的組合
    python fetch_extended_derivative_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python fetch_extended_derivative_data.py --gap-fill 30
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

DATASET_START = {
    "futures_inst_investors":   "2018-06-05",
    "options_inst_investors":   "2018-06-05",
    "futures_inst_after_hours": "2021-10-12",
    "options_inst_after_hours": "2021-10-12",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

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

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_FUT_INST = """CREATE TABLE IF NOT EXISTS futures_inst_investors (date DATE, futures_id VARCHAR(50), institutional_investors VARCHAR(100), long_deal_volume BIGINT, long_deal_amount NUMERIC(20,2), short_deal_volume BIGINT, short_deal_amount NUMERIC(20,2), long_open_interest_balance_volume BIGINT, long_open_interest_balance_amount NUMERIC(20,2), short_open_interest_balance_volume BIGINT, short_open_interest_balance_amount NUMERIC(20,2), PRIMARY KEY (date, futures_id, institutional_investors));"""
DDL_OPT_INST = """CREATE TABLE IF NOT EXISTS options_inst_investors (date DATE, option_id VARCHAR(50), call_put VARCHAR(10), institutional_investors VARCHAR(100), long_deal_volume BIGINT, long_deal_amount NUMERIC(20,2), short_deal_volume BIGINT, short_deal_amount NUMERIC(20,2), long_open_interest_balance_volume BIGINT, long_open_interest_balance_amount NUMERIC(20,2), short_open_interest_balance_volume BIGINT, short_open_interest_balance_amount NUMERIC(20,2), PRIMARY KEY (date, option_id, call_put, institutional_investors));"""

UPSERT_FUT_INST = """INSERT INTO futures_inst_investors VALUES %s ON CONFLICT (date, futures_id, institutional_investors) DO UPDATE SET long_deal_volume = EXCLUDED.long_deal_volume;"""
UPSERT_OPT_INST = """INSERT INTO options_inst_investors VALUES %s ON CONFLICT (date, option_id, call_put, institutional_investors) DO UPDATE SET long_deal_volume = EXCLUDED.long_deal_volume;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_fut_inst(r): return (r["date"], r.get("futures_id") or r.get("name"), r.get("institutional_investors"), safe_int(r.get("long_deal_volume")), safe_float(r.get("long_deal_amount")), safe_int(r.get("short_deal_volume")), safe_float(r.get("short_deal_amount")), safe_int(r.get("long_open_interest_balance_volume")), safe_float(r.get("long_open_interest_balance_amount")), safe_int(r.get("short_open_interest_balance_volume")), safe_float(r.get("short_open_interest_balance_amount")))
def map_opt_inst(r): return (r["date"], r.get("option_id") or r.get("name"), r.get("call_put"), r.get("institutional_investors"), safe_int(r.get("long_deal_volume")), safe_float(r.get("long_deal_amount")), safe_int(r.get("short_deal_volume")), safe_float(r.get("short_deal_amount")), safe_int(r.get("long_open_interest_balance_volume")), safe_float(r.get("long_open_interest_balance_amount")), safe_int(r.get("short_open_interest_balance_volume")), safe_float(r.get("short_open_interest_balance_amount")))

def fetch_inst(conn, dataset, table, ddl, upsert_sql, mapper, start, end, delay, force, target_ids=None):
    logger.info(f"=== [{table}] 開始 ===")
    ensure_ddl(conn, ddl)
    key_col = "futures_id" if "futures" in table else "option_id"
    latest = get_all_safe_starts(conn, table, key_col=key_col)
    flog = FailureLogger(table, db_conn=conn)
    total_rows = 0
    tmpl = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" if "futures" in table else "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    
    # 將大時間跨度切分成年度區塊抓取
    s_dt = datetime.strptime(start, "%Y-%m-%d").date()
    e_dt = datetime.strptime(end, "%Y-%m-%d").date()
    
    curr = s_dt
    while curr <= e_dt:
        chunk_end = min(curr + timedelta(days=365), e_dt)
        s_str = curr.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")
        logger.info(f"  [{table}] 正在抓取 {s_str} ~ {e_str}...")
        
        t0 = time.time()
        try:
            data = finmind_get(dataset, {"start_date": s_str, "end_date": e_str}, delay)
            dur = int((time.time() - t0) * 1000)
            
            if data:
                if target_ids:
                    id_key = "futures_id" if "futures" in table else "option_id"
                    data = [r for r in data if r.get(id_key) in target_ids]
                
                rows = [mapper(r) for r in data]
                rows = dedup_rows(rows, (0, 1, 2) if "futures" in table else (0, 1, 2, 3))
                res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, label_prefix=table, failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, table_name=table, stock_id="ALL", fetch_date_from=s_str, fetch_date_to=e_str, 
                                 rows_inserted=n, duration_ms=dur, status="success")
            else:
                _write_fetch_log(conn, table_name=table, stock_id="ALL", fetch_date_from=s_str, fetch_date_to=e_str, 
                                 rows_inserted=0, duration_ms=dur, status="no_new_data")
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id="market", error=str(e))
            _write_fetch_log(conn, table_name=table, stock_id="ALL", fetch_date_from=s_str, fetch_date_to=e_str, 
                             rows_inserted=0, duration_ms=dur, status="failed", error_message=str(e))
        
        curr = chunk_end + timedelta(days=1)

    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
    flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["futures_inst_investors", "options_inst_investors", "all"], default=["all"])
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--ids", help="指定標的 ID (例如 TX, MTX, TXO)，多筆用逗號分隔")
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["futures_inst_investors", "options_inst_investors"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        target_ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
        if "futures_inst_investors" in tables: fetch_inst(conn, "TaiwanFuturesInstitutionalInvestors", "futures_inst_investors", DDL_FUT_INST, UPSERT_FUT_INST, map_fut_inst, args.start, args.end, args.delay, args.force, target_ids)
        if "options_inst_investors" in tables: fetch_inst(conn, "TaiwanOptionInstitutionalInvestors", "options_inst_investors", DDL_OPT_INST, UPSERT_OPT_INST, map_opt_inst, args.start, args.end, args.delay, args.force, target_ids)
    finally:
        conn.close()

if __name__ == "__main__":
    main()