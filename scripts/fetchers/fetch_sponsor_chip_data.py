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
fetch_sponsor_chip_data.py v3.0 — Sponsor 進階籌碼（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：分點買賣、持股分級、八大行庫、期貨部位每一天獨立 commit。
  ★ 針對 broker_trades（海量數據）優化：確保在長時抓取中每一天的進度都即時落盤。
  ★ 全面整合 FailureLogger：精準捕捉 Sponsor 方案各資料集的抓取異常。
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DATASET_START = {
    "holding_shares_per": "2014-01-01",
    "broker_trades":      "2016-01-01",
    "eight_banks":        "2021-01-01",
    "futures_large_oi":   "2010-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
DDL_HOLDING = """CREATE TABLE IF NOT EXISTS holding_shares_per (date DATE, stock_id VARCHAR(50), level VARCHAR(50), people BIGINT, percent NUMERIC(10,4), unit VARCHAR(100), PRIMARY KEY (date, stock_id, level));"""
DDL_BROKER = """CREATE TABLE IF NOT EXISTS broker_trades (date DATE, stock_id VARCHAR(50), broker_id VARCHAR(50), broker_name VARCHAR(100), buy BIGINT, sell BIGINT, PRIMARY KEY (date, stock_id, broker_id));"""
DDL_EIGHT_BANKS = """CREATE TABLE IF NOT EXISTS eight_banks_buy_sell (date DATE, stock_id VARCHAR(50), buy BIGINT, sell BIGINT, PRIMARY KEY (date, stock_id));"""
DDL_FUTURES_LARGE_OI = """CREATE TABLE IF NOT EXISTS futures_large_oi (date DATE, contract_code VARCHAR(20), name VARCHAR(50), long_position BIGINT, long_position_over50 BIGINT, short_position BIGINT, short_position_over50 BIGINT, net_position BIGINT, market_total_oi BIGINT, PRIMARY KEY (date, contract_code, name));"""

UPSERT_HOLDING = """INSERT INTO holding_shares_per (date, stock_id, level, people, percent, unit) VALUES %s ON CONFLICT (date, stock_id, level) DO UPDATE SET people = EXCLUDED.people, percent = EXCLUDED.percent;"""
UPSERT_BROKER = """INSERT INTO broker_trades (date, stock_id, broker_id, broker_name, buy, sell) VALUES %s ON CONFLICT (date, stock_id, broker_id) DO UPDATE SET buy = EXCLUDED.buy, sell = EXCLUDED.sell;"""
UPSERT_EIGHT_BANKS = """INSERT INTO eight_banks_buy_sell (date, stock_id, buy, sell) VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET buy = EXCLUDED.buy, sell = EXCLUDED.sell;"""
UPSERT_FUTURES_LARGE_OI = """INSERT INTO futures_large_oi (date, contract_code, name, long_position, long_position_over50, short_position, short_position_over50, net_position, market_total_oi) VALUES %s ON CONFLICT (date, contract_code, name) DO UPDATE SET net_position = EXCLUDED.net_position;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_holding(r: dict) -> tuple:
    lv = str(r.get("HoldingSharesLevel", ""))
    return (r["date"], r["stock_id"], lv, safe_int(r.get("people")), safe_float(r.get("percent")), r.get("unit", lv))

def map_broker(r: dict) -> tuple:
    # 注意：BrokerTrades 需要按 (date, stock_id, broker_id) 彙總，因為原始 API 可能按價格拆分
    return (r["date"], r["stock_id"], str(r.get("securities_trader_id", "")), r.get("securities_trader", ""), safe_int(r.get("buy")), safe_int(r.get("sell")))

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
        if not s: continue
        try:
            data = finmind_get("TaiwanStockHoldingSharesPer", {"data_id": sid, "start_date": s, "end_date": end}, delay)
            if data:
                rows = [map_holding(r) for r in data]
                rows = dedup_rows(rows, (0, 1, 2))
                res = commit_per_stock_per_day(conn, UPSERT_HOLDING, rows, "(%s, %s, %s, %s, %s, %s)", label_prefix="holding", failure_logger=flog)
                total_rows += sum(res.values())
        except Exception as e: flog.record(stock_id=sid, error=str(e))
    logger.info(f"  [holding_shares_per] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_broker(conn, stock_ids, start, end, delay, force):
    logger.info("=== [broker_trades] 開始 ===")
    ensure_ddl(conn, DDL_BROKER)
    latest = get_all_safe_starts(conn, "broker_trades")
    flog = FailureLogger("broker_trades", db_conn=conn)
    total_rows = 0
    for sid in stock_ids:
        s_dt = datetime.strptime(resolve_start_cached(sid, latest, start, DATASET_START["broker_trades"], force) or end, "%Y-%m-%d").date()
        e_dt = datetime.strptime(end, "%Y-%m-%d").date()
        while s_dt <= e_dt:
            # 分點資料一次抓 30 天，避免資料過大
            c_e = min(s_dt + timedelta(days=29), e_dt)
            try:
                data = finmind_get("TaiwanStockTradingDailyReport", {"data_id": sid, "start_date": s_dt.strftime("%Y-%m-%d"), "end_date": c_e.strftime("%Y-%m-%d")}, delay)
                if data:
                    # 彙總同日同券商
                    agg = defaultdict(lambda: [0, 0, ""])
                    for r in data:
                        k = (r["date"], sid, str(r.get("securities_trader_id", "")))
                        agg[k][0] += safe_int(r.get("buy"))
                        agg[k][1] += safe_int(r.get("sell"))
                        agg[k][2] = r.get("securities_trader", "")
                    rows = [(k[0], k[1], k[2], v[2], v[0], v[1]) for k, v in agg.items()]
                    rows = dedup_rows(rows, (0, 1, 2))
                    res = commit_per_stock_per_day(conn, UPSERT_BROKER, rows, "(%s, %s, %s, %s, %s, %s)", label_prefix="broker", failure_logger=flog)
                    total_rows += sum(res.values())
            except Exception as e: flog.record(stock_id=sid, error=str(e))
            s_dt = c_e + timedelta(days=1)
    logger.info(f"  [broker_trades] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_eight_banks(conn, stock_ids, start, end, delay, force):
    logger.info(f"=== [eight_banks] 開始 (過濾 {len(stock_ids)} 支) ===")
    ensure_ddl(conn, DDL_EIGHT_BANKS)
    flog = FailureLogger("eight_banks", db_conn=conn)
    
    # ⭐ 自動尋找起始日 ⭐
    if not start and not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM eight_banks_buy_sell")
            max_d = cur.fetchone()[0]
            if max_d:
                start = (max_d + timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"  [eight_banks] 自動從資料庫最新日期續傳：{start}")
            else:
                start = DATASET_START["eight_banks"]
    
    # ❗ 此 Dataset 強制「全市場」+「單日」抓取 ❗
    s_dt = datetime.strptime(start or DATASET_START["eight_banks"], "%Y-%m-%d").date()
    e_dt = datetime.strptime(end, "%Y-%m-%d").date()
    
    total_rows = 0
    curr = s_dt
    while curr <= e_dt:
        d_str = curr.strftime("%Y-%m-%d")
        try:
            # ❗ 單日抓取，且不帶 end_date 以免 400 錯誤 ❗
            data = finmind_get("TaiwanStockGovernmentBankBuySell", {"start_date": d_str}, delay)
            if data:
                # ⭐ 本地彙總：同一天同一支股票可能有各家行庫資料，加總起來 ⭐
                agg = defaultdict(lambda: [0, 0])
                s_set = set(stock_ids) if stock_ids else None
                
                for r in data:
                    sid = r.get("stock_id")
                    if s_set and sid not in s_set:
                        continue
                    k = (d_str, sid)
                    agg[k][0] += safe_int(r.get("buy", 0))
                    agg[k][1] += safe_int(r.get("sell", 0))
                
                if agg:
                    rows = [(k[0], k[1], v[0], v[1]) for k, v in agg.items()]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, UPSERT_EIGHT_BANKS, rows, "(%s, %s, %s, %s)", label_prefix="eight_banks", failure_logger=flog)
                    total_rows += sum(res.values())
        except Exception as e:
            flog.record(date=d_str, error=f"Day {d_str} failed: {e}")
        
        curr += timedelta(days=1)

    logger.info(f"  [eight_banks] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_futures_oi(conn, start, end, delay, force):
    logger.info("=== [futures_large_oi] 開始 ===")
    ensure_ddl(conn, DDL_FUTURES_LARGE_OI)
    flog = FailureLogger("futures_large_oi", db_conn=conn)
    try:
        data = finmind_get("TaiwanFuturesOpenInterestLargeTraders", {"start_date": start, "end_date": end}, delay)
        if data:
            rows = [map_futures_oi(r) for r in data]
            rows = dedup_rows(rows, (0, 1, 2))
            res = commit_per_stock_per_day(conn, UPSERT_FUTURES_LARGE_OI, rows, "(%s, %s, %s, %s, %s, %s, %s, %s, %s)", stock_index=1, label_prefix="futures", failure_logger=flog)
            logger.info(f"  [futures_large_oi] 寫入 {sum(res.values())} 筆")
    except Exception as e: flog.record(stock_id="futures", error=str(e))
    flog.summary()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["holding_shares_per", "broker_trades", "eight_banks", "futures_large_oi", "all"], default=["all"])
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tables = ["holding_shares_per", "broker_trades", "eight_banks", "futures_large_oi"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    try:
        from core.db_utils import get_db_stock_ids
        stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else get_db_stock_ids(conn)
        if "holding_shares_per" in tables: fetch_holding(conn, stock_ids, args.start or DATASET_START["holding_shares_per"], args.end, args.delay, args.force)
        if "broker_trades" in tables: fetch_broker(conn, [args.stock_id] if args.stock_id else ["2330"], args.start or DATASET_START["broker_trades"], args.end, args.delay, args.force)
        if "eight_banks" in tables: fetch_eight_banks(conn, stock_ids, args.start or DATASET_START["eight_banks"], args.end, args.delay, args.force)
        if "futures_large_oi" in tables: fetch_futures_oi(conn, args.start or DATASET_START["futures_large_oi"], args.end, args.delay, args.force)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
