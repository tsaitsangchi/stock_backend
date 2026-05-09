"""
fetch_sponsor_chip_data.py v5.1 (Trinity Core Edition)
================================================================================
進階籌碼資料抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取大戶持股比例、八大行庫買賣超及期貨大額交易人未平倉量。

涵蓋資料表：
  · holding_shares_per      ─ 大戶持股比例 (TaiwanStockHoldingSharesPer)
  · eight_banks_buy_sell    ─ 八大行庫買賣超 (TaiwanStockGovernmentBankBuySell)
  · futures_large_oi        ─ 期貨大額未平倉 (TaiwanFuturesOpenInterestLargeTraders)

核心特色：
  · 智慧逐日輪詢     ─ 針對「八大行庫」API 限制，自動實作穩定的單日抓取與聚合。
  · 斷路器保護       ─ 對接 v5.1 核心，排除業務錯誤對系統熔斷的干擾。
  · 非法 ID 過濾     ─ 自動過濾非個股 ID，維護 API 請求健康度。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ 402 自動休眠 + 智慧斷路器
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 修正 ImportError，換裝 FinMindClient().get_data()。
    - [核心] 移除 get_db_conn，全面改用 db_session 與 db_transaction。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，實現精準監控。
    - [穩定] 導入 is_valid_stock_id 過濾非數字 ID，防止 Automobile 等代碼觸發熔斷。
  v3.2 (2024-05-01):
    - [基礎] 建立基礎進階籌碼抓取架構。

執行範例：
    # 範例 1：抓取所有進階籌碼資料 (增量更新)
    python scripts/fetchers/fetch_sponsor_chip_data.py
    
    # 範例 2：僅抓取八大行庫資料
    python scripts/fetchers/fetch_sponsor_chip_data.py --tables eight_banks
    
    # 範例 3：針對台積電強制重抓最近一年的大戶持股
    python scripts/fetchers/fetch_sponsor_chip_data.py --stock-id 2330 --tables holding_shares_per --days 365 --force
"""

import sys
import argparse
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import (
        db_session, db_transaction, ensure_ddl, commit_per_stock_per_day,
        get_latest_date, write_fetch_log, FailureLogger,
        safe_int, safe_float, get_db_stock_ids
    )
    from core.finmind_client import FinMindClient, get_request_stats
    from core.migrate_stocks_config import migrate as sync_stocks_table
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. DDL 與 SQL 配置
# =====================================================================

DDL_HOLDING = """CREATE TABLE IF NOT EXISTS holding_shares_per (date DATE, stock_id VARCHAR(50), level VARCHAR(50), people BIGINT, percent NUMERIC(20,6), unit VARCHAR(100), PRIMARY KEY (date, stock_id, level));"""
DDL_EIGHT_BANKS = """CREATE TABLE IF NOT EXISTS eight_banks_buy_sell (date DATE, stock_id VARCHAR(50), buy BIGINT, sell BIGINT, PRIMARY KEY (date, stock_id));"""
DDL_FUTURES_LARGE_OI = """CREATE TABLE IF NOT EXISTS futures_large_oi (date DATE, contract_code VARCHAR(20), name VARCHAR(50), long_position BIGINT, long_position_over50 BIGINT, short_position BIGINT, short_position_over50 BIGINT, net_position BIGINT, market_total_oi BIGINT, PRIMARY KEY (date, contract_code, name));"""

UPSERT_HOLDING = """INSERT INTO holding_shares_per (date, stock_id, level, people, percent, unit) VALUES (%(date)s, %(stock_id)s, %(level)s, %(people)s, %(percent)s, %(unit)s) ON CONFLICT (date, stock_id, level) DO UPDATE SET people = EXCLUDED.people, percent = EXCLUDED.percent;"""
UPSERT_EIGHT_BANKS = """INSERT INTO eight_banks_buy_sell (date, stock_id, buy, sell) VALUES (%(date)s, %(stock_id)s, %(buy)s, %(sell)s) ON CONFLICT (date, stock_id) DO UPDATE SET buy = EXCLUDED.buy, sell = EXCLUDED.sell;"""
UPSERT_FUTURES_LARGE_OI = """INSERT INTO futures_large_oi (date, contract_code, name, long_position, long_position_over50, short_position, short_position_over50, net_position, market_total_oi) VALUES (%(date)s, %(contract_code)s, %(name)s, %(long_position)s, %(long_position_over50)s, %(short_position)s, %(short_position_over50)s, %(net_position)s, %(market_total_oi)s) ON CONFLICT (date, contract_code, name) DO UPDATE SET net_position = EXCLUDED.net_position;"""

# =====================================================================
# 2. 工具函式
# =====================================================================

def is_valid_stock_id(sid: str) -> bool:
    return sid.isdigit() and len(sid) >= 4

def taipei_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    except:
        return date.today().strftime("%Y-%m-%d")

def next_day(date_str: str) -> str:
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (d + timedelta(days=1)).strftime("%Y-%m-%d")
    except:
        return str(date_str)

def resolve_stock_ids(args) -> List[str]:
    if args.stock_id:
        return [s.strip() for s in args.stock_id.split(",") if s.strip()]
    try:
        with db_session() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT stock_id FROM stocks WHERE fetch_chip = TRUE AND is_active = TRUE ORDER BY stock_id")
                return [r[0] for r in cur.fetchall()]
    except: return []

# =====================================================================
# 3. 核心抓取邏輯 (v5.1 精準監控)
# =====================================================================

def fetch_holding(stock_ids: list[str], start: str, end: str, force: bool) -> None:
    api = FinMindClient()
    table = "holding_shares_per"
    ensure_ddl(DDL_HOLDING)
    
    logger.info(f"━━━ 抓取資料表 {table} ({len(stock_ids)} stocks) ━━━")
    for sid in stock_ids:
        if not is_valid_stock_id(sid): continue
        
        cur_start = start
        if not force:
            latest = get_latest_date(table, sid)
            if latest:
                cur_start = next_day(latest)
                if cur_start > end:
                    write_fetch_log(table, sid, "skipped", "sponsor_v5.1", str(latest), end, 0, 0, "up_to_date")
                    continue

        try:
            t0 = time.monotonic()
            data = api.get_data("TaiwanStockHoldingSharesPer", sid, cur_start, end)
            duration_ms = int((time.monotonic() - t0) * 1000)
            
            if data:
                records = []
                for r in data:
                    lv = str(r.get("HoldingSharesLevel", ""))
                    records.append({
                        "date": r["date"], "stock_id": sid, "level": lv,
                        "people": safe_int(r.get("people")), "percent": safe_float(r.get("percent")), "unit": r.get("unit", lv)
                    })
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_HOLDING, sid)
                    write_fetch_log(table, sid, "success" if error == 0 else "partial", "sponsor_v5.1", cur_start, end, duration_ms, success, None)
                    logger.info(f"  ✓ {sid}: {success} rows")
        except Exception as e:
            logger.error(f"  ❌ {sid} @ {table}: {e}")
            write_fetch_log(table, sid, "failed", "sponsor_v5.1", cur_start, end, 0, 0, str(e))

def fetch_eight_banks(stock_ids: list[str], start: str, end: str, force: bool) -> None:
    api = FinMindClient()
    table = "eight_banks_buy_sell"
    ensure_ddl(DDL_EIGHT_BANKS)
    
    cur_start = start
    if not start and not force:
        latest = get_latest_date(table, "ALL")
        cur_start = next_day(latest) if latest else "2021-01-01"
    
    s_dt = datetime.strptime(cur_start, "%Y-%m-%d").date()
    e_dt = datetime.strptime(end, "%Y-%m-%d").date()
    
    logger.info(f"━━━ 抓取資料表 {table} (逐日輪詢) ━━━")
    curr = s_dt
    sid_set = set(stock_ids) if stock_ids else None

    while curr <= e_dt:
        if curr.weekday() >= 5: # 跳過週末
            curr += timedelta(days=1); continue
            
        d_str = curr.strftime("%Y-%m-%d")
        t0 = time.monotonic()
        try:
            data = api.get_data("TaiwanStockGovernmentBankBuySell", start_date=d_str)
            duration_ms = int((time.monotonic() - t0) * 1000)
            
            if data:
                agg = defaultdict(lambda: {"buy": 0, "sell": 0})
                for r in data:
                    sid = r.get("stock_id")
                    if sid_set and sid not in sid_set: continue
                    agg[sid]["buy"] += safe_int(r.get("buy", 0))
                    agg[sid]["sell"] += safe_int(r.get("sell", 0))

                records = [{"date": d_str, "stock_id": sid, "buy": v["buy"], "sell": v["sell"]} for sid, v in agg.items()]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_EIGHT_BANKS, "ALL")
                    write_fetch_log(table, "ALL", "success", "sponsor_v5.1", d_str, d_str, duration_ms, success, None)
                    logger.info(f"  ✓ {d_str}: {success} stocks updated")
            else:
                write_fetch_log(table, "ALL", "no_new_data", "sponsor_v5.1", d_str, d_str, duration_ms, 0, None)
        except Exception as e:
            logger.error(f"  ❌ {d_str} @ {table}: {e}")
            write_fetch_log(table, "ALL", "failed", "sponsor_v5.1", d_str, d_str, 0, 0, str(e))
        
        curr += timedelta(days=1)
        time.sleep(0.5)

def fetch_futures_oi(start: str, end: str, force: bool) -> None:
    api = FinMindClient()
    table = "futures_large_oi"
    ensure_ddl(DDL_FUTURES_LARGE_OI)
    
    logger.info(f"━━━ 抓取資料表 {table} ━━━")
    t0 = time.monotonic()
    try:
        data = api.get_data("TaiwanFuturesOpenInterestLargeTraders", start_date=start, end_date=end)
        duration_ms = int((time.monotonic() - t0) * 1000)
        if data:
            records = [{
                "date": r["date"], "contract_code": r.get("contract_code", ""), "name": r.get("name", ""),
                "long_position": safe_int(r.get("long_position")), "long_position_over50": safe_int(r.get("long_position_over50")),
                "short_position": safe_int(r.get("short_position")), "short_position_over50": safe_int(r.get("short_position_over50")),
                "net_position": safe_int(r.get("net_position")), "market_total_oi": safe_int(r.get("market_total_oi"))
            } for r in data]
            success, error = commit_per_stock_per_day(table, records, UPSERT_FUTURES_LARGE_OI, "FUTURES")
            write_fetch_log(table, "FUTURES", "success", "sponsor_v5.1", start, end, duration_ms, success, None)
            logger.info(f"  ✓ {table}: {success} rows")
        else:
            write_fetch_log(table, "FUTURES", "no_new_data", "sponsor_v5.1", start, end, duration_ms, 0, None)
    except Exception as e:
        logger.error(f"  ❌ {table} 抓取崩潰: {e}")
        write_fetch_log(table, "FUTURES", "failed", "sponsor_v5.1", start, end, 0, 0, str(e))

def main():
    parser = argparse.ArgumentParser(description="Sponsor Chip Fetcher v5.1 (Trinity Core Edition)")
    parser.add_argument("--tables", nargs="+", choices=["holding_shares_per", "eight_banks", "futures_large_oi", "all"], default=["all"])
    parser.add_argument("--stock-id", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tables = ["holding_shares_per", "eight_banks", "futures_large_oi"] if "all" in args.tables else args.tables
    stock_ids = resolve_stock_ids(args)
    end_date = args.end or taipei_today()
    start_date = args.start
    if not start_date:
        e_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (e_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"  Sponsor Chip Fetcher v5.1  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)

    if "holding_shares_per" in tables:
        fetch_holding(stock_ids, start_date, end_date, args.force)
    
    if "eight_banks" in tables:
        fetch_eight_banks(stock_ids, start_date, end_date, args.force)
    
    if "futures_large_oi" in tables:
        fetch_futures_oi(start_date, end_date, args.force)

    logger.info("🎉 全部完成")
    try: get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()