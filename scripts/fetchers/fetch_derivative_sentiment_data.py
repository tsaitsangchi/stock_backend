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
fetch_derivative_sentiment_data.py v3.0 — 衍生品與情緒指標（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：選擇權大額交易、鉅額交易、恐懼與貪婪指數每一天獨立原子 commit。
  ★ 全面整合 FailureLogger：精準追蹤情緒指標在 150 支股票與市場層級的更新狀況。
  ★ 結構規範化：移除本地冗餘工具，確保生產管線高可用。
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    get_db_stock_ids,
    ensure_ddl,
    safe_float,
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
    "options_large_oi":  "2018-01-01",
    "fear_greed_index":  "2011-01-03",
    "block_trading":     "2021-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_SENTIMENT = """
CREATE TABLE IF NOT EXISTS options_oi_large_holders (date DATE, option_id VARCHAR(50), put_call VARCHAR(10), contract_type VARCHAR(50), name VARCHAR(100), market_open_interest NUMERIC, buy_top5_trader_open_interest NUMERIC, buy_top5_trader_open_interest_per NUMERIC, buy_top10_trader_open_interest NUMERIC, buy_top10_trader_open_interest_per NUMERIC, sell_top5_trader_open_interest NUMERIC, sell_top5_trader_open_interest_per NUMERIC, sell_top10_trader_open_interest NUMERIC, sell_top10_trader_open_interest_per NUMERIC, buy_top5_specific_open_interest NUMERIC, buy_top5_specific_open_interest_per NUMERIC, buy_top10_specific_open_interest NUMERIC, buy_top10_specific_open_interest_per NUMERIC, sell_top5_specific_open_interest NUMERIC, sell_top5_specific_open_interest_per NUMERIC, sell_top10_specific_open_interest NUMERIC, sell_top10_specific_open_interest_per NUMERIC, PRIMARY KEY (date, option_id, put_call, contract_type));
CREATE TABLE IF NOT EXISTS fear_greed_index (date DATE PRIMARY KEY, fear_greed NUMERIC, fear_greed_emotion VARCHAR(50));
CREATE TABLE IF NOT EXISTS block_trading (date DATE, stock_id VARCHAR(50), securities_trader_id VARCHAR(50), securities_trader VARCHAR(100), price NUMERIC(10,2), buy NUMERIC, sell NUMERIC, trade_type VARCHAR(50), PRIMARY KEY (date, stock_id, securities_trader_id, price, trade_type));
"""

UPSERT_OPTIONS_LARGE_OI = """INSERT INTO options_oi_large_holders VALUES %s ON CONFLICT (date, option_id, put_call, contract_type) DO UPDATE SET market_open_interest = EXCLUDED.market_open_interest;"""
UPSERT_FEAR_GREED = """INSERT INTO fear_greed_index (date, fear_greed, fear_greed_emotion) VALUES %s ON CONFLICT (date) DO UPDATE SET fear_greed = EXCLUDED.fear_greed;"""
UPSERT_BLOCK_TRADING = """INSERT INTO block_trading (date, stock_id, securities_trader_id, securities_trader, price, buy, sell, trade_type) VALUES %s ON CONFLICT (date, stock_id, securities_trader_id, price, trade_type) DO UPDATE SET buy = EXCLUDED.buy;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_opt_large(r): return (r["date"], str(r.get("option_id", "")), r.get("put_call", ""), str(r.get("contract_type", "")), r.get("name", ""), safe_float(r.get("market_open_interest")), safe_float(r.get("buy_top5_trader_open_interest")), safe_float(r.get("buy_top5_trader_open_interest_per")), safe_float(r.get("buy_top10_trader_open_interest")), safe_float(r.get("buy_top10_trader_open_interest_per")), safe_float(r.get("sell_top5_trader_open_interest")), safe_float(r.get("sell_top5_trader_open_interest_per")), safe_float(r.get("sell_top10_trader_open_interest")), safe_float(r.get("sell_top10_trader_open_interest_per")), safe_float(r.get("buy_top5_specific_open_interest")), safe_float(r.get("buy_top5_specific_open_interest_per")), safe_float(r.get("buy_top10_specific_open_interest")), safe_float(r.get("buy_top10_specific_open_interest_per")), safe_float(r.get("sell_top5_specific_open_interest")), safe_float(r.get("sell_top5_specific_open_interest_per")), safe_float(r.get("sell_top10_specific_open_interest")), safe_float(r.get("sell_top10_specific_open_interest_per")))
def map_fg(r): return (r["date"], safe_float(r.get("fear_greed")), r.get("fear_greed_emotion", "")[:50])
def map_block(r): return (r["date"], r["stock_id"], str(r.get("securities_trader_id", "")), r.get("securities_trader", "")[:100], safe_float(r.get("price")), safe_float(r.get("buy")), safe_float(r.get("sell")), str(r.get("trade_type", ""))[:50])

def fetch_block_trading(conn, stock_ids, start, end, delay, force):
    logger.info("=== [block_trading] 開始 ===")
    flog = FailureLogger("block_trading", db_conn=conn)
    latest = get_all_safe_starts(conn, "block_trading")
    total_rows = 0
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START["block_trading"], force)
        if not s: continue
        try:
            data = finmind_get("TaiwanStockBlockTrade", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [map_block(r) for r in data]
                rows = dedup_rows(rows, (0, 1, 2, 4, 7))
                res = commit_per_stock_per_day(conn, UPSERT_BLOCK_TRADING, rows, None, label_prefix=f"block/{sid}", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=sid, error=str(e))
    logger.info(f"  [block_trading] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_sentiment(conn, dataset, table, upsert_sql, mapper, start, end, delay, force):
    logger.info(f"=== [{table}] 開始 ===")
    flog = FailureLogger(table, db_conn=conn)
    
    # ── 取得交易日清單 ──
    with conn.cursor() as cur:
        cur.execute("SELECT date FROM trading_date")
        trading_days = {r[0] for r in cur.fetchall()}

    # ⭐ 自動尋找起始日 ⭐
    if not start and not force:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(date) FROM {table}")
            max_d = cur.fetchone()[0]
            if max_d:
                start = (max_d + timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"  [{table}] 自動從資料庫最新日期續傳：{start}")
            else:
                start = DATASET_START.get(table, "2021-01-01")
    
    s_dt = datetime.strptime(start or DATASET_START.get(table, "2021-01-01"), "%Y-%m-%d").date()
    e_dt = datetime.strptime(end, "%Y-%m-%d").date()
    
    total_rows = 0
    curr = s_dt
    while curr <= e_dt:
        if table == "options_oi_large_holders" and curr not in trading_days:
            curr += timedelta(days=1)
            continue
            
        d_str = curr.strftime("%Y-%m-%d")
        logger.info(f"  [{table}] 正在抓取 {d_str}...")
        try:
            # 對於 options_large_oi，強制單日抓取
            if table == "options_oi_large_holders":
                data = finmind_get(dataset, {"start_date": d_str, "end_date": d_str}, delay)
            else:
                # 其他資料集 (如 fear_greed) 可能支援範圍，但為了保險也可逐日
                data = finmind_get(dataset, {"start_date": d_str, "end_date": d_str}, delay)
                
            if data:
                rows = [mapper(r) for r in data]
                if table == "fear_greed_index":
                    rows = dedup_rows(rows, (0,))
                    res = commit_per_day(conn, upsert_sql, rows, "(%s::date, %s::numeric, %s)", date_index=0, label_prefix=table, failure_logger=flog)
                else:
                    rows = dedup_rows(rows, (0, 1, 2, 3))
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, None, label_prefix=table, failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e:
            flog.record(date=d_str, error=str(e))
        
        # 如果是 fear_greed 且我們發現它其實支援範圍抓取，可以優化。
        # 但目前為了穩定性，先統一逐日或小波段。
        curr += timedelta(days=1)

    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
    flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["options_large_oi", "fear_greed_index", "block_trading", "all"], default=["all"])
    p.add_argument("--ids")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["options_large_oi", "fear_greed_index", "block_trading"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_SENTIMENT)
        stock_ids = [s.strip() for s in args.ids.split(",")] if args.ids else get_db_stock_ids(conn)
        
        if "options_large_oi" in tables: fetch_sentiment(conn, "TaiwanOptionOpenInterestLargeTraders", "options_oi_large_holders", UPSERT_OPTIONS_LARGE_OI, map_opt_large, args.start, args.end, args.delay, args.force)
        if "fear_greed_index" in tables: fetch_sentiment(conn, "CnnFearGreedIndex", "fear_greed_index", UPSERT_FEAR_GREED, map_fg, args.start, args.end, args.delay, args.force)
        if "block_trading" in tables: fetch_block_trading(conn, stock_ids, args.start, args.end, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()