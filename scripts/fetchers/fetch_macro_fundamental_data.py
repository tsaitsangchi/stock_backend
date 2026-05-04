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
fetch_macro_fundamental_data.py v3.0 — 總經與基本面補強資料抓取（逐支逐日 commit 完整性版）
====================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：景氣信號、市值權重、產業鏈分類每一天獨立原子 commit。
  ★ 全面整合 FailureLogger：精準追蹤總經補強資料在全市場 150 支核心標的之覆蓋狀況。
  ★ 結構規範化：移除本地冗餘工具，確保生產管線高可用。
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
    commit_per_day,
    dedup_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DATASET_START = {
    "business_indicator":  "1982-01-01",
    "market_value_weight": "2024-10-30",
    "industry_chain":      None,
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_MACRO_FUND = """
CREATE TABLE IF NOT EXISTS business_indicator (date DATE PRIMARY KEY, "leading" NUMERIC(10,2), leading_notrend NUMERIC(10,2), coincident NUMERIC(10,2), coincident_notrend NUMERIC(10,2), lagging NUMERIC(10,2), lagging_notrend NUMERIC(10,2), monitoring NUMERIC(10,2), monitoring_color VARCHAR(20));
CREATE TABLE IF NOT EXISTS market_value_weight (date DATE, stock_id VARCHAR(50), stock_name VARCHAR(100), rank INTEGER, weight_per NUMERIC(8,4), type VARCHAR(10), PRIMARY KEY (date, stock_id));
CREATE TABLE IF NOT EXISTS industry_chain (stock_id VARCHAR(50), industry VARCHAR(100), sub_industry VARCHAR(100), date DATE, PRIMARY KEY (stock_id));
"""

UPSERT_BI = """INSERT INTO business_indicator VALUES %s ON CONFLICT (date) DO UPDATE SET monitoring = EXCLUDED.monitoring;"""
UPSERT_MVW = """INSERT INTO market_value_weight VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET weight_per = EXCLUDED.weight_per;"""
UPSERT_IC = """INSERT INTO industry_chain VALUES %s ON CONFLICT (stock_id) DO UPDATE SET industry = EXCLUDED.industry;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_bi(r): return (r["date"], safe_float(r.get("leading")), safe_float(r.get("leading_notrend")), safe_float(r.get("coincident")), safe_float(r.get("coincident_notrend")), safe_float(r.get("lagging")), safe_float(r.get("lagging_notrend")), safe_float(r.get("monitoring")), r.get("monitoring_color", "")[:20])
def map_mvw(r): return (r["date"], str(r.get("stock_id", "")), r.get("stock_name", "")[:100], safe_int(r.get("rank")), safe_float(r.get("weight_per")), r.get("type", "")[:10])
def map_ic(r): return (str(r.get("stock_id", "")), r.get("industry", "")[:100], r.get("sub_industry", "")[:100], r.get("date"))

def fetch_macro_fund(conn, dataset, table, upsert_sql, mapper, start, end, delay, force):
    logger.info(f"=== [{table}] 開始 ===")
    flog = FailureLogger(table, db_conn=conn)
    try:
        params = {"start_date": start, "end_date": end} if start else {}
        data = finmind_get(dataset, params, delay)
        if data:
            rows = [mapper(r) for r in data]
            # ⭐ 主動去重 ⭐
            if table == "business_indicator":
                rows = dedup_rows(rows, (0,))
            elif table == "market_value_weight":
                rows = dedup_rows(rows, (0, 1))
            elif table == "industry_chain":
                rows = dedup_rows(rows, (0,))

            tmpl = "(%s::date,%s,%s,%s,%s,%s,%s,%s,%s)" if table == "business_indicator" else ("(%s::date,%s,%s,%s,%s,%s)" if table == "market_value_weight" else "(%s,%s,%s,%s::date)")
            if table == "business_indicator":
                # 景氣指標只有日期，使用 commit_per_day
                res = commit_per_day(conn, upsert_sql, rows, tmpl, date_index=0, label_prefix=table, failure_logger=flog)
            else:
                # 市值權重與產業鏈有 stock_id
                stock_idx = 1 if table == "market_value_weight" else 0
                date_idx = 0 if table == "market_value_weight" else (3 if table == "industry_chain" else None)
                res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, stock_index=stock_idx, date_index=date_idx, label_prefix=table, failure_logger=flog)
            
            logger.info(f"  [{table}] 寫入 {sum(res.values())} 筆")
    except Exception as e: flog.record(stock_id="market", error=str(e))
    flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["business_indicator", "market_value_weight", "industry_chain", "all"], default=["all"])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["business_indicator", "market_value_weight", "industry_chain"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_MACRO_FUND)
        if "business_indicator" in tables: fetch_macro_fund(conn, "TaiwanBusinessIndicator", "business_indicator", UPSERT_BI, map_bi, args.start, args.end, args.delay, args.force)
        if "market_value_weight" in tables: fetch_macro_fund(conn, "TaiwanStockMarketValueWeight", "market_value_weight", UPSERT_MVW, map_mvw, args.start, args.end, args.delay, args.force)
        if "industry_chain" in tables: fetch_macro_fund(conn, "TaiwanStockIndustryChain", "industry_chain", UPSERT_IC, map_ic, None, None, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
