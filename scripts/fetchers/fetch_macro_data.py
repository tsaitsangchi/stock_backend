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
fetch_macro_data.py v3.1 — 總經與產業特徵（可觀察性監控版）
================================================================================
v3.1 重大改進：
  ★ 導入 fetch_log：將任務狀態 (success/failed)、耗時 (ms)、寫入行數及 CLI 參數存入資料庫。
  ★ 結構規範化：整合 FailureLogger 與原子性 commit 機制，確保數據完整性。
  ★ CLI 增強：支援指定標的、強制重抓與時間範圍過濾。

使用範例：
    # 抓取所有總經數據（預設起始 2010-01-01）
    ./venv/bin/python scripts/fetchers/fetch_macro_data.py

    # 強制重抓指定貨幣的匯率
    ./venv/bin/python scripts/fetchers/fetch_macro_data.py --ids USD,JPY --force

    # 抓取特定時間段的債券殖利率
    ./venv/bin/python scripts/fetchers/fetch_macro_data.py --start 2024-01-01 --end 2024-05-01
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

# ──────────────────────────────────────────────
# DDL & SQL
# ──────────────────────────────────────────────
DDL_MACRO = """
CREATE TABLE IF NOT EXISTS interest_rate (
    date DATE, 
    country VARCHAR(50), 
    full_country_name VARCHAR(100), 
    interest_rate NUMERIC(20,2), 
    PRIMARY KEY (date, country)
);
CREATE TABLE IF NOT EXISTS exchange_rate (
    date DATE, 
    currency VARCHAR(50), 
    cash_buy NUMERIC(20,4), 
    cash_sell NUMERIC(20,4), 
    spot_buy NUMERIC(20,4), 
    spot_sell NUMERIC(20,4), 
    PRIMARY KEY (date, currency)
);
CREATE TABLE IF NOT EXISTS bond_yield (
    date DATE, 
    bond_id VARCHAR(50), 
    value NUMERIC(10,4), 
    PRIMARY KEY (date, bond_id)
);
"""

UPSERT_INTEREST = """
INSERT INTO interest_rate (date, country, full_country_name, interest_rate) 
VALUES %s 
ON CONFLICT (date, country) DO UPDATE SET interest_rate = EXCLUDED.interest_rate;
"""

UPSERT_EXCHANGE = """
INSERT INTO exchange_rate (date, currency, cash_buy, cash_sell, spot_buy, spot_sell) 
VALUES %s 
ON CONFLICT (date, currency) DO UPDATE SET 
    cash_buy = EXCLUDED.cash_buy,
    cash_sell = EXCLUDED.cash_sell,
    spot_buy = EXCLUDED.spot_buy,
    spot_sell = EXCLUDED.spot_sell;
"""

UPSERT_BOND = """
INSERT INTO bond_yield (date, bond_id, value) 
VALUES %s 
ON CONFLICT (date, bond_id) DO UPDATE SET value = EXCLUDED.value;
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

# ──────────────────────────────────────────────
# Fetchers
# ──────────────────────────────────────────────
def fetch_interest_rate(conn, start, end, delay, force, target_ids, cli_args):
    start_time = time.time()
    logger.info("=== [interest_rate] 開始 ===")
    
    latest = get_all_safe_starts(conn, "interest_rate", key_col="country")
    flog = FailureLogger("interest_rate", db_conn=conn)
    total_rows = 0
    status = "success"
    error_msg = None
    
    try:
        countries = target_ids if target_ids else ["FED", "BOJ", "ECB", "PBOC"]
        for country in countries:
            if country not in ["FED", "BOJ", "ECB", "PBOC"]: continue
            s = resolve_start_cached(country, latest, start, "2000-01-01", force)
            if not s: continue
            
            data = finmind_get("InterestRate", {"data_id": country, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [(r["date"], r["country"], r.get("full_country_name"), r["interest_rate"]) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_INTEREST, rows, "(%s, %s, %s, %s::numeric)", label_prefix="interest", failure_logger=flog)
                total_rows += sum(res.values())
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        logger.error(f"抓取 interest_rate 發生錯誤: {e}")
    finally:
        duration_ms = (time.time() - start_time) * 1000
        _write_fetch_log(conn, "interest_rate", status, total_rows, duration_ms, error_msg, cli_args)
        logger.info(f"  [interest_rate] 總共寫入 {total_rows} 筆 (耗時 {duration_ms:.0f}ms)")
        flog.summary()

def fetch_exchange_rate(conn, start, end, delay, force, target_ids, cli_args):
    start_time = time.time()
    logger.info("=== [exchange_rate] 開始 ===")
    
    latest = get_all_safe_starts(conn, "exchange_rate", key_col="currency")
    flog = FailureLogger("exchange_rate", db_conn=conn)
    total_rows = 0
    status = "success"
    error_msg = None
    
    try:
        currencies = target_ids if target_ids else ["USD", "JPY", "EUR"]
        for curr in currencies:
            if curr not in ["USD", "JPY", "EUR"]: continue
            s = resolve_start_cached(curr, latest, start, "2000-01-01", force)
            if not s: continue
            
            data = finmind_get("TaiwanExchangeRate", {"data_id": curr, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [(r["date"], r["currency"], r.get("cash_buy"), r.get("cash_sell"), r.get("spot_buy"), r.get("spot_sell")) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_EXCHANGE, rows, "(%s, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric)", label_prefix="exchange", failure_logger=flog)
                total_rows += sum(res.values())
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        logger.error(f"抓取 exchange_rate 發生錯誤: {e}")
    finally:
        duration_ms = (time.time() - start_time) * 1000
        _write_fetch_log(conn, "exchange_rate", status, total_rows, duration_ms, error_msg, cli_args)
        logger.info(f"  [exchange_rate] 總共寫入 {total_rows} 筆 (耗時 {duration_ms:.0f}ms)")
        flog.summary()

def fetch_bond_yield(conn, start, end, delay, force, target_ids, cli_args):
    start_time = time.time()
    logger.info("=== [bond_yield] 開始 ===")
    
    latest = get_all_safe_starts(conn, "bond_yield", key_col="bond_id")
    flog = FailureLogger("bond_yield", db_conn=conn)
    total_rows = 0
    status = "success"
    error_msg = None
    
    try:
        all_bonds = [("United States 10-Year", "US10Y"), ("United States 2-Year", "US2Y")]
        if target_ids:
            all_bonds = [b for b in all_bonds if b[1] in target_ids]
        
        for bid, sid in all_bonds:
            s = resolve_start_cached(sid, latest, start, "2000-01-01", force)
            if not s: continue
            
            data = finmind_get("GovernmentBondsYield", {"data_id": bid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [(r["date"], sid, r["value"]) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_BOND, rows, "(%s, %s, %s::numeric)", label_prefix="bond", failure_logger=flog)
                total_rows += sum(res.values())
    except Exception as e:
        status = "failed"
        error_msg = str(e)
        logger.error(f"抓取 bond_yield 發生錯誤: {e}")
    finally:
        duration_ms = (time.time() - start_time) * 1000
        _write_fetch_log(conn, "bond_yield", status, total_rows, duration_ms, error_msg, cli_args)
        logger.info(f"  [bond_yield] 總共寫入 {total_rows} 筆 (耗時 {duration_ms:.0f}ms)")
        flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--ids")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cli_args = json.dumps(vars(args))
    target_ids = args.ids.split(",") if args.ids else None
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_MACRO)
        fetch_interest_rate(conn, args.start, args.end, args.delay, args.force, target_ids, cli_args)
        fetch_exchange_rate(conn, args.start, args.end, args.delay, args.force, target_ids, cli_args)
        fetch_bond_yield(conn, args.start, args.end, args.delay, args.force, target_ids, cli_args)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
