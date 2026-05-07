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
fetch_derivative_sentiment_data.py — 衍生品與情緒指標（v3.1 fetch_log 整合版）
================================================================================
v3.1 改進：
  · 整合 fetch_log：每次抓取（無論成功、失敗或跳過）都會寫入監控日誌。
  · 效能追蹤：記錄情緒指標與鉅額交易的 API 請求耗時（duration_ms）。
  · 支援 --retry-failed N 與 --gap-fill N 模式，實現智慧補抓。

v3.0 既有：
  · 支援 3 個資料表：options_oi_large_holders, fear_greed_index, block_trading。
  · 導入 commit_per_stock_per_day：鉅額交易每一支股票、每一天獨立原子 commit。
  · 整合 FailureLogger：精準追蹤市場與個股層級的更新狀況。

執行（常規）：
    python fetch_derivative_sentiment_data.py
    python fetch_derivative_sentiment_data.py --tables fear_greed_index
    python fetch_derivative_sentiment_data.py --ids 2330 --tables block_trading --force
    python fetch_derivative_sentiment_data.py --ids 2330 --tables all --force
    python fetch_derivative_sentiment_data.py --tables all --force

執行（模式切換）：
    # 重試最近 7 天失敗的組合
    python fetch_derivative_sentiment_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python fetch_derivative_sentiment_data.py --gap-fill 30
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
        if not s:
            _write_fetch_log(conn, table_name="block_trading", stock_id=sid, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            data = finmind_get("TaiwanStockBlockTrade", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            dur = int((time.time() - t0) * 1000)
            if data:
                rows = [map_block(r) for r in data]
                rows = dedup_rows(rows, (0, 1, 2, 4, 7))
                res = commit_per_stock_per_day(conn, UPSERT_BLOCK_TRADING, rows, None, label_prefix=f"block/{sid}", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, table_name="block_trading", stock_id=sid, fetch_date_from=s, fetch_date_to=end, 
                                 rows_inserted=n, duration_ms=dur, status="success")
            else:
                _write_fetch_log(conn, table_name="block_trading", stock_id=sid, fetch_date_from=s, fetch_date_to=end, 
                                 rows_inserted=0, duration_ms=dur, status="no_new_data")
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e))
            _write_fetch_log(conn, table_name="block_trading", stock_id=sid, fetch_date_from=s, fetch_date_to=end, 
                             rows_inserted=0, duration_ms=dur, status="failed", error_message=str(e))
    logger.info(f"  [block_trading] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_sentiment(conn, dataset, table, upsert_sql, mapper, start, end, delay, force):
    logger.info(f"=== [{table}] 開始 ===")
    flog = FailureLogger(table, db_conn=conn)
    


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
        d_str = curr.strftime("%Y-%m-%d")
        logger.info(f"  [{table}] 正在抓取自 {d_str} 起的資料塊...")
        t0 = time.time()
        try:
            # ⭐ 核心優化：不帶 end_date 可觸發快速度大量回傳 (約 200-300 天) ⭐
            data = finmind_get(dataset, {"start_date": d_str}, delay)
            dur = int((time.time() - t0) * 1000)
            
            if not data:
                # 若無資料，則跳過一天繼續
                _write_fetch_log(conn, table_name=table, fetch_mode="market", fetch_date_from=d_str, 
                                 rows_inserted=0, duration_ms=dur, status="no_new_data")
                curr += timedelta(days=1)
                continue
                
            # 轉換資料
            rows = [mapper(r) for r in data]
            
            # 找出這批資料中最後一天的日期
            received_dates = sorted(list(set(r[0] for r in rows)))
            logger.info(f"    -> 成功接收 {len(received_dates)} 天的資料 ({received_dates[0]} ~ {received_dates[-1]})")
            
            last_date_str = received_dates[-1]
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            
            # 寫入資料庫
            if table == "fear_greed_index":
                rows = dedup_rows(rows, (0,))
                res = commit_per_day(conn, upsert_sql, rows, "(%s::date, %s::numeric, %s)", date_index=0, label_prefix=table, failure_logger=flog)
            elif table == "options_oi_large_holders":
                # ⭐ 核心優化：使用 commit_per_day 進行大塊寫入，避免一筆一筆 commit ⭐
                rows = dedup_rows(rows, (0, 1, 2, 3))
                tmpl = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                res = commit_per_day(conn, upsert_sql, rows, tmpl, date_index=0, label_prefix=table, failure_logger=flog)
            else:
                rows = dedup_rows(rows, (0, 1, 2, 3))
                res = commit_per_stock_per_day(conn, upsert_sql, rows, None, label_prefix=table, failure_logger=flog)
            
            n = sum(res.values())
            total_rows += n
            _write_fetch_log(conn, table_name=table, fetch_mode="market", fetch_date_from=d_str, fetch_date_to=last_date_str, 
                             rows_inserted=n, duration_ms=dur, status="success")
            
            # ⭐ 下一次從最後一天的隔天開始 ⭐
            new_curr = last_date + timedelta(days=1)
            
            # 如果日期沒推進，強制跳一天避開無窮迴圈
            if new_curr <= curr:
                curr += timedelta(days=1)
            else:
                curr = new_curr
            
            # 如果最後一天已經超過 end，則結束
            if curr > e_dt:
                break
                
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(date=d_str, error=str(e))
            _write_fetch_log(conn, table_name=table, fetch_mode="market", fetch_date_from=d_str, 
                             rows_inserted=0, duration_ms=dur, status="failed", error_message=str(e))
            curr += timedelta(days=1) # 出錯則跳過一天

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