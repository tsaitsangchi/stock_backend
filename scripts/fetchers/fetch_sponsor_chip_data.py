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
fetch_sponsor_chip_data.py v3.1 — 進階籌碼資料（監控整合標準版）
================================================================================
v3.1 重大改進：
  · 整合 fetch_log v3.1：針對大戶持股、券商分點、八大行庫等任務記錄完整監控日誌。
  · 效能監控：精準追蹤 API 請求與資料處理的耗時（duration_ms）。
  · 狀態追蹤：支援 success, failed, no_new_data, skipped 等標準化狀態。
  · 速度優化：維持八大行庫的月批次抓取機制，顯著提升同步效率。

執行範例（常規）：
    python scripts/fetchers/fetch_sponsor_chip_data.py                # 抓取所有進階籌碼資料
    # 八大行庫：FinMind 限制每次只回傳單日資料，程式自動逐日抓取
    python scripts/fetchers/fetch_sponsor_chip_data.py --tables eight_banks

執行範例（指定標的）：
    python scripts/fetchers/fetch_sponsor_chip_data.py --stock-id 2330 --tables holding_shares_per
    python scripts/fetchers/fetch_sponsor_chip_data.py --stock-id 2330,2317 --force --tables holding_shares_per
    # 單一標的強制重抓（不含八大行庫，因其為全市場資料集，不支援個股篩選）
    python scripts/fetchers/fetch_sponsor_chip_data.py --stock-id 2330 --force --tables holding_shares_per futures_large_oi

執行範例（全市場資料）：
    # 八大行庫必須使用全市場模式抓取（本地再過濾）
    python scripts/fetchers/fetch_sponsor_chip_data.py --tables eight_banks
    python scripts/fetchers/fetch_sponsor_chip_data.py --tables eight_banks --start 2024-01-01 --force

執行範例（期貨）：
    python scripts/fetchers/fetch_sponsor_chip_data.py --tables futures_large_oi
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_int,
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

DATASET_START = {
    "holding_shares_per": "2014-01-01",
    "eight_banks":        "2021-01-01",
    "futures_large_oi":   "2010-01-01",
}
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

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_HOLDING = """CREATE TABLE IF NOT EXISTS holding_shares_per (date DATE, stock_id VARCHAR(50), level VARCHAR(50), people BIGINT, percent NUMERIC(10,4), unit VARCHAR(100), PRIMARY KEY (date, stock_id, level));"""
DDL_EIGHT_BANKS = """CREATE TABLE IF NOT EXISTS eight_banks_buy_sell (date DATE, stock_id VARCHAR(50), buy BIGINT, sell BIGINT, PRIMARY KEY (date, stock_id));"""
DDL_FUTURES_LARGE_OI = """CREATE TABLE IF NOT EXISTS futures_large_oi (date DATE, contract_code VARCHAR(20), name VARCHAR(50), long_position BIGINT, long_position_over50 BIGINT, short_position BIGINT, short_position_over50 BIGINT, net_position BIGINT, market_total_oi BIGINT, PRIMARY KEY (date, contract_code, name));"""

UPSERT_HOLDING = """INSERT INTO holding_shares_per (date, stock_id, level, people, percent, unit) VALUES %s ON CONFLICT (date, stock_id, level) DO UPDATE SET people = EXCLUDED.people, percent = EXCLUDED.percent;"""
UPSERT_EIGHT_BANKS = """INSERT INTO eight_banks_buy_sell (date, stock_id, buy, sell) VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET buy = EXCLUDED.buy, sell = EXCLUDED.sell;"""
UPSERT_FUTURES_LARGE_OI = """INSERT INTO futures_large_oi (date, contract_code, name, long_position, long_position_over50, short_position, short_position_over50, net_position, market_total_oi) VALUES %s ON CONFLICT (date, contract_code, name) DO UPDATE SET net_position = EXCLUDED.net_position;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_holding(r: dict) -> tuple:
    lv = str(r.get("HoldingSharesLevel", ""))
    return (r["date"], r["stock_id"], lv, safe_int(r.get("people")), safe_float(r.get("percent")), r.get("unit", lv))

def map_eight_banks(r: dict) -> tuple:
    return (r["date"], r["stock_id"], safe_int(r.get("buy")), safe_int(r.get("sell")))

def map_futures_oi(r: dict) -> tuple:
    return (r["date"], r.get("contract_code", ""), r.get("name", ""), safe_int(r.get("long_position")), safe_int(r.get("long_position_over50")), safe_int(r.get("short_position")), safe_int(r.get("short_position_over50")), safe_int(r.get("net_position")), safe_int(r.get("market_total_oi")))

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_holding(conn, stock_ids, start, end, delay, force):
    logger.info("=== [holding_shares_per] 開始 ===")
    ensure_ddl(conn, DDL_HOLDING)
    latest = get_all_safe_starts(conn, "holding_shares_per")
    flog = FailureLogger("holding_shares_per", db_conn=conn)
    total_rows = 0
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START["holding_shares_per"], force)
        if not s:
            _write_fetch_log(conn, "holding_shares_per", sid, "skipped", error_message="up_to_date")
            continue
        
        start_time = time.time()
        try:
            data = finmind_get("TaiwanStockHoldingSharesPer", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            duration_ms = int((time.time() - start_time) * 1000)
            if data:
                rows = [map_holding(r) for r in data]
                rows = dedup_rows(rows, (0, 1, 2))
                res = commit_per_stock_per_day(conn, UPSERT_HOLDING, rows, "(%s, %s, %s, %s, %s, %s)", label_prefix="holding", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, "holding_shares_per", sid, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration_ms)
            else:
                _write_fetch_log(conn, "holding_shares_per", sid, "no_new_data", fetch_date_from=s, fetch_date_to=end, duration_ms=duration_ms)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            flog.record(stock_id=sid, error=str(e))
            _write_fetch_log(conn, "holding_shares_per", sid, "failed", fetch_date_from=s, fetch_date_to=end, duration_ms=duration_ms, error_message=str(e))
            
    logger.info(f"  [holding_shares_per] 總共寫入 {total_rows} 筆")
    flog.summary()

EIGHT_BANKS_CHUNK_DAYS = 30 

def fetch_eight_banks(conn, stock_ids, start, end, delay, force):
    logger.info(f"=== [eight_banks] 開始 (過濾 {len(stock_ids)} 支) ===")
    ensure_ddl(conn, DDL_EIGHT_BANKS)
    flog = FailureLogger("eight_banks", db_conn=conn)

    if not start and not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM eight_banks_buy_sell")
            max_d = cur.fetchone()[0]
            if max_d:
                start = (max_d + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start = DATASET_START["eight_banks"]

    s_dt = datetime.strptime(start or DATASET_START["eight_banks"], "%Y-%m-%d").date()
    e_dt = datetime.strptime(end, "%Y-%m-%d").date()
    if s_dt > e_dt:
        logger.info(f"  [eight_banks] 資料已是最新的。")
        _write_fetch_log(conn, "eight_banks_buy_sell", "ALL", "skipped", error_message="up_to_date")
        return

    # ⚠️ FinMind 限制：TaiwanStockGovernmentBankBuySell 不支援 end_date
    # 每次只能抓單日全市場資料，以 start_date 逐日請求
    s_set = set(stock_ids) if stock_ids else None
    total_rows = 0
    curr = s_dt
    while curr <= e_dt:
        d_str = curr.strftime("%Y-%m-%d")
        start_time = time.time()
        try:
            # 僅傳 start_date，不傳 end_date
            data = finmind_get("TaiwanStockGovernmentBankBuySell", {"start_date": d_str}, delay)
            duration_ms = int((time.time() - start_time) * 1000)
            if data:
                agg = defaultdict(lambda: [0, 0])
                for r in data:
                    sid = r.get("stock_id")
                    if s_set and sid not in s_set: continue
                    k = (r["date"], sid)
                    agg[k][0] += safe_int(r.get("buy", 0))
                    agg[k][1] += safe_int(r.get("sell", 0))

                if agg:
                    rows = [(k[0], k[1], v[0], v[1]) for k, v in agg.items()]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, UPSERT_EIGHT_BANKS, rows, "(%s, %s, %s, %s)", label_prefix="eight_banks", failure_logger=flog)
                    n = sum(res.values())
                    total_rows += n
                    if n > 0:
                        logger.info(f"  [eight_banks] {d_str} 寫入 {n} 筆")
                    _write_fetch_log(conn, "eight_banks_buy_sell", "ALL", "success", rows_inserted=n, fetch_date_from=d_str, fetch_date_to=d_str, duration_ms=duration_ms)
                else:
                    _write_fetch_log(conn, "eight_banks_buy_sell", "ALL", "no_new_data", fetch_date_from=d_str, fetch_date_to=d_str, duration_ms=duration_ms)
            else:
                _write_fetch_log(conn, "eight_banks_buy_sell", "ALL", "no_new_data", fetch_date_from=d_str, fetch_date_to=d_str, duration_ms=duration_ms)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            flog.record(date=d_str, error=str(e))
            _write_fetch_log(conn, "eight_banks_buy_sell", "ALL", "failed", fetch_date_from=d_str, fetch_date_to=d_str, duration_ms=duration_ms, error_message=str(e))
        curr += timedelta(days=1)

    logger.info(f"  [eight_banks] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_futures_oi(conn, start, end, delay, force):
    logger.info("=== [futures_large_oi] 開始 ===")
    ensure_ddl(conn, DDL_FUTURES_LARGE_OI)
    flog = FailureLogger("futures_large_oi", db_conn=conn)
    start_time = time.time()
    try:
        data = finmind_get("TaiwanFuturesOpenInterestLargeTraders", {"start_date": start, "end_date": end}, delay)
        duration_ms = int((time.time() - start_time) * 1000)
        if data:
            rows = [map_futures_oi(r) for r in data]
            rows = dedup_rows(rows, (0, 1, 2))
            res = commit_per_stock_per_day(conn, UPSERT_FUTURES_LARGE_OI, rows, "(%s, %s, %s, %s, %s, %s, %s, %s, %s)", stock_index=1, label_prefix="futures", failure_logger=flog)
            n = sum(res.values())
            logger.info(f"  [futures_large_oi] 寫入 {n} 筆")
            _write_fetch_log(conn, "futures_large_oi", "FUTURES", "success", rows_inserted=n, fetch_date_from=start, fetch_date_to=end, duration_ms=duration_ms)
        else:
            _write_fetch_log(conn, "futures_large_oi", "FUTURES", "no_new_data", fetch_date_from=start, fetch_date_to=end, duration_ms=duration_ms)
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        flog.record(stock_id="futures", error=str(e))
        _write_fetch_log(conn, "futures_large_oi", "FUTURES", "failed", fetch_date_from=start, fetch_date_to=end, duration_ms=duration_ms, error_message=str(e))
    flog.summary()

def main():
    p = argparse.ArgumentParser(description="進階籌碼資料抓取 (v3.1 — 監控整合標準版)")
    p.add_argument("--tables", nargs="+", choices=["holding_shares_per", "eight_banks", "futures_large_oi", "all"], default=["all"])
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["holding_shares_per", "eight_banks", "futures_large_oi"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        from core.db_utils import get_db_stock_ids
        stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else get_db_stock_ids(conn)
        if "holding_shares_per" in tables: fetch_holding(conn, stock_ids, args.start or DATASET_START["holding_shares_per"], args.end, args.delay, args.force)
        if "eight_banks" in tables: fetch_eight_banks(conn, stock_ids, args.start, args.end, args.delay, args.force)
        if "futures_large_oi" in tables: fetch_futures_oi(conn, args.start or DATASET_START["futures_large_oi"], args.end, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
