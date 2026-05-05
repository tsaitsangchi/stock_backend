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
fetch_extended_derivative_data.py v3.0 — 期貨/選擇權 II（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：三大法人期權部位、盤後交易、自營商買賣每一天獨立原子 commit。
  ★ 全面整合 FailureLogger：精準追蹤衍生品進階籌碼資料的抓取狀況。
  ★ 結構一致化：與技術面、籌碼面腳本維持相同寫入規範，確保生產管線高可用。
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
    
    try:
        # 衍生品籌碼 API (TaiwanFuturesInstitutionalInvestors / TaiwanOptionInstitutionalInvestors)
        # 通常不支援 data_id 參數，或支援方式不同。建議一次抓整天，再於本地過濾。
        data = finmind_get(dataset, {"start_date": start, "end_date": end}, delay)
        
        if data and target_ids:
            # 依據 target_ids 過濾
            id_key = "futures_id" if "futures" in table else "option_id"
            # 注意：API 回傳欄位名可能是 futures_id / option_id
            data = [r for r in data if r.get(id_key) in target_ids]
        
        if data:
            rows = [mapper(r) for r in data]
            # 去重：(date, id, investor)
            rows = dedup_rows(rows, (0, 1, 2) if "futures" in table else (0, 1, 2, 3))
            res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, label_prefix=table, failure_logger=flog)
            total_rows += sum(res.values())
    except Exception as e: flog.record(stock_id="market", error=str(e))
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