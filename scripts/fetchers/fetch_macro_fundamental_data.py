import sys
import logging
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_macro_fundamental_data.py v3.1 — 總經與基本面補強資料（可觀察性監控版）
====================================================================================
v3.1 重大改進：
  ★ 導入 fetch_log：整合任務監控，記錄狀態、耗時、資料行數與 CLI 執行參數。
  ★ 原子性 Commit：景氣信號、市值權重與產業鏈分類採逐日或逐支 commit，確保斷點續傳。
  ★ 健壯性增強：強化 FailureLogger，精準追蹤總經補強資料在核心標的之覆蓋狀況。

使用範例：
    # 執行所有任務（預設起始 2021-01-01）
    ./venv/bin/python scripts/fetchers/fetch_macro_fundamental_data.py

    # 僅更新景氣指標
    ./venv/bin/python scripts/fetchers/fetch_macro_fundamental_data.py --tables business_indicator

    # 強制重抓市值權重歷史資料
    ./venv/bin/python scripts/fetchers/fetch_macro_fundamental_data.py --tables market_value_weight --start 2024-01-01 --force

    # 強制更新所有資料表（從預設起始日開始全量重抓）
    ./venv/bin/python scripts/fetchers/fetch_macro_fundamental_data.py --tables all --force
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
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
CREATE TABLE IF NOT EXISTS business_indicator (
    date DATE PRIMARY KEY, 
    "leading" NUMERIC(10,2), 
    leading_notrend NUMERIC(10,2), 
    coincident NUMERIC(10,2), 
    coincident_notrend NUMERIC(10,2), 
    lagging NUMERIC(10,2), 
    lagging_notrend NUMERIC(10,2), 
    monitoring NUMERIC(10,2), 
    monitoring_color VARCHAR(20)
);
CREATE TABLE IF NOT EXISTS market_value_weight (
    date DATE, 
    stock_id VARCHAR(50), 
    stock_name VARCHAR(100), 
    rank INTEGER, 
    weight_per NUMERIC(8,4), 
    type VARCHAR(10), 
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS industry_chain (
    stock_id VARCHAR(50), 
    industry VARCHAR(100), 
    sub_industry VARCHAR(100), 
    date DATE, 
    PRIMARY KEY (stock_id)
);
"""

UPSERT_BI = """
INSERT INTO business_indicator 
VALUES %s 
ON CONFLICT (date) DO UPDATE SET monitoring = EXCLUDED.monitoring;
"""

UPSERT_MVW = """
INSERT INTO market_value_weight 
VALUES %s 
ON CONFLICT (date, stock_id) DO UPDATE SET weight_per = EXCLUDED.weight_per;
"""

UPSERT_IC = """
INSERT INTO industry_chain 
VALUES %s 
ON CONFLICT (stock_id) DO UPDATE SET industry = EXCLUDED.industry;
"""

def _write_fetch_log(conn, table_name, status, rows_inserted, duration_ms, error_msg, cli_args):
    """v3.1 標準可觀察性日誌記錄器"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (run_ts, table_name, status, rows_inserted, duration_ms, error_message, cli_args)
                VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s)
            """, (table_name, status, rows_inserted, int(duration_ms), error_msg, cli_args))
            conn.commit()
    except Exception as e:
        logger.warning(f"無法寫入 fetch_log: {e}")

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_bi(r): 
    return (
        r["date"], 
        safe_float(r.get("leading")), 
        safe_float(r.get("leading_notrend")), 
        safe_float(r.get("coincident")), 
        safe_float(r.get("coincident_notrend")), 
        safe_float(r.get("lagging")), 
        safe_float(r.get("lagging_notrend")), 
        safe_float(r.get("monitoring")), 
        r.get("monitoring_color", "")[:20]
    )

def map_mvw(r): 
    return (
        r["date"], 
        str(r.get("stock_id", "")), 
        r.get("stock_name", "")[:100], 
        safe_int(r.get("rank")), 
        safe_float(r.get("weight_per")), 
        r.get("type", "")[:10]
    )

def map_ic(r): 
    return (
        str(r.get("stock_id", "")), 
        r.get("industry", "")[:100], 
        r.get("sub_industry", "")[:100], 
        r.get("date")
    )

def fetch_macro_fund(conn, dataset, table, upsert_sql, mapper, start, end, delay, force, cli_args):
    start_time = time.time()
    logger.info(f"=== [{table}] 開始 ===")
    
    flog = FailureLogger(table, db_conn=conn)
    total_rows = 0
    status = "success"
    error_msg = None
    
    try:
        params = {"start_date": start, "end_date": end} if start else {}
        data = finmind_get(dataset, params, delay)
        if data:
            rows = [mapper(r) for r in data]
            
            # ⭐ 去重與 DDL 模板處理 ⭐
            if table == "business_indicator":
                rows = dedup_rows(rows, (0,))
                tmpl = "(%s::date,%s,%s,%s,%s,%s,%s,%s,%s)"
                res = commit_per_day(conn, upsert_sql, rows, tmpl, date_index=0, label_prefix=table, failure_logger=flog)
            elif table == "market_value_weight":
                rows = dedup_rows(rows, (0, 1))
                tmpl = "(%s::date,%s,%s,%s,%s,%s)"
                res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, stock_index=1, date_index=0, label_prefix=table, failure_logger=flog)
            elif table == "industry_chain":
                rows = dedup_rows(rows, (0,))
                tmpl = "(%s,%s,%s,%s::date)"
                res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, stock_index=0, date_index=3, label_prefix=table, failure_logger=flog)
            
            total_rows = sum(res.values())
        else:
            status = "no_data"
            
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        logger.error(f"抓取 {table} 發生錯誤: {e}")
    finally:
        duration_ms = (time.time() - start_time) * 1000
        _write_fetch_log(conn, table, status, total_rows, duration_ms, error_msg, cli_args)
        logger.info(f"  [{table}] 寫入 {total_rows} 筆 (耗時 {duration_ms:.0f}ms)")
        flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["business_indicator", "market_value_weight", "industry_chain", "all"], default=["all"])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cli_args = json.dumps(vars(args))
    tables = ["business_indicator", "market_value_weight", "industry_chain"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_MACRO_FUND)
        
        if "business_indicator" in tables:
            fetch_macro_fund(conn, "TaiwanBusinessIndicator", "business_indicator", UPSERT_BI, map_bi, args.start, args.end, args.delay, args.force, cli_args)
        
        if "market_value_weight" in tables:
            fetch_macro_fund(conn, "TaiwanStockMarketValueWeight", "market_value_weight", UPSERT_MVW, map_mvw, args.start, args.end, args.delay, args.force, cli_args)
        
        if "industry_chain" in tables:
            # 產業鏈不支援日期範圍，直接抓取
            fetch_macro_fund(conn, "TaiwanStockIndustryChain", "industry_chain", UPSERT_IC, map_ic, None, None, args.delay, args.force, cli_args)
            
    finally:
        conn.close()

if __name__ == "__main__":
    main()
