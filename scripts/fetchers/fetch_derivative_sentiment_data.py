"""
fetch_derivative_sentiment_data.py v5.0 (Trinity Core Edition)
================================================================================
情緒指標與鉅額交易抓取器 — 完美對接 core/ 五大核心模組
支援並發抓取、自動事務管理與市場/個股路由切換。

涵蓋資料表：
  · options_oi_large_holders ─ 期權大額未平倉 (FinMind: TaiwanOptionOpenInterestLargeTraders)
  · fear_greed_index         ─ 貪婪恐懼指數     (FinMind: CnnFearGreedIndex)
  · block_trading            ─ 鉅額交易         (FinMind: TaiwanStockBlockTrade)

對接核心模組 (scripts/core/)：
  · db_utils v4.5            ─ 連線池 + 自動事務 + 批次寫入降級
  · finmind_client v4.1      ─ Singleton + SQLite 快取 + 斷路器保護
  · migrate_stocks_config v3.0 ─ 自動確保 stocks 表配置與 config.py 同步
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.0 (2026-05-09):
    - [核心] 修正 ImportError，將 commit_per_day 統一升級為 commit_per_stock_per_day。
    - [性能] 引入 ThreadPoolExecutor (支援 --workers)，支援多執行緒並行抓取。
    - [穩定] 實作 90 天智慧分段抓取 (Chunking)，解決 V4 API 長區間 422 錯誤。
    - [整合] 完美對接 core v4.5 核心組件。
  v3.2 (2026-04-20):
    - [基礎] 建立情緒指標與鉅額交易原型。

執行範例：
    # 範例 1：抓取所有情緒指標 (增量更新)
    python scripts/fetchers/fetch_derivative_sentiment_data.py --tables options_large_oi,fear_greed
    
    # 範例 2：抓取核心個股的鉅額交易並使用 5 執行緒
    python scripts/fetchers/fetch_derivative_sentiment_data.py --tables block_trading --all --workers 5
    
    # 範例 3：全量更新所有情緒與指標
    python scripts/fetchers/fetch_derivative_sentiment_data.py --all --workers 5
"""

import sys
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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
# 1. DDL 與 SQL 對照表
# =====================================================================

DDL_MAP = {
    "options_oi_large_holders": """
        CREATE TABLE IF NOT EXISTS options_oi_large_holders (
            date DATE NOT NULL,
            option_id VARCHAR(50) NOT NULL,
            put_call VARCHAR(10) NOT NULL,
            contract_type VARCHAR(50) NOT NULL,
            name VARCHAR(100),
            market_open_interest NUMERIC(20,6),
            buy_top5_trader_open_interest NUMERIC(20,6),
            buy_top5_trader_open_interest_per NUMERIC(20,6),
            buy_top10_trader_open_interest NUMERIC(20,6),
            buy_top10_trader_open_interest_per NUMERIC(20,6),
            sell_top5_trader_open_interest NUMERIC(20,6),
            sell_top5_trader_open_interest_per NUMERIC(20,6),
            sell_top10_trader_open_interest NUMERIC(20,6),
            sell_top10_trader_open_interest_per NUMERIC(20,6),
            buy_top5_specific_open_interest NUMERIC(20,6),
            buy_top5_specific_open_interest_per NUMERIC(20,6),
            buy_top10_specific_open_interest NUMERIC(20,6),
            buy_top10_specific_open_interest_per NUMERIC(20,6),
            sell_top5_specific_open_interest NUMERIC(20,6),
            sell_top5_specific_open_interest_per NUMERIC(20,6),
            sell_top10_specific_open_interest NUMERIC(20,6),
            sell_top10_specific_open_interest_per NUMERIC(20,6),
            PRIMARY KEY (date, option_id, put_call, contract_type)
        );
    """,
    "fear_greed_index": """
        CREATE TABLE IF NOT EXISTS fear_greed_index (
            date DATE PRIMARY KEY,
            fear_greed NUMERIC(20,6),
            fear_greed_emotion VARCHAR(50)
        );
    """,
    "block_trading": """
        CREATE TABLE IF NOT EXISTS block_trading (
            date DATE NOT NULL,
            stock_id VARCHAR(50) NOT NULL,
            securities_trader_id VARCHAR(50) NOT NULL,
            securities_trader VARCHAR(100),
            price NUMERIC(20,6) NOT NULL,
            buy NUMERIC(20,6),
            sell NUMERIC(20,6),
            trade_type VARCHAR(50) NOT NULL,
            PRIMARY KEY (date, stock_id, securities_trader_id, price, trade_type)
        );
    """
}

UPSERT_MAP = {
    "options_oi_large_holders": """
        INSERT INTO options_oi_large_holders VALUES (
            %(date)s, %(option_id)s, %(put_call)s, %(contract_type)s, %(name)s,
            %(market_open_interest)s, %(buy_top5_trader_open_interest)s, %(buy_top5_trader_open_interest_per)s,
            %(buy_top10_trader_open_interest)s, %(buy_top10_trader_open_interest_per)s,
            %(sell_top5_trader_open_interest)s, %(sell_top5_trader_open_interest_per)s,
            %(sell_top10_trader_open_interest)s, %(sell_top10_trader_open_interest_per)s,
            %(buy_top5_specific_open_interest)s, %(buy_top5_specific_open_interest_per)s,
            %(buy_top10_specific_open_interest)s, %(buy_top10_specific_open_interest_per)s,
            %(sell_top5_specific_open_interest)s, %(sell_top5_specific_open_interest_per)s,
            %(sell_top10_specific_open_interest)s, %(sell_top10_specific_open_interest_per)s
        ) ON CONFLICT (date, option_id, put_call, contract_type) DO UPDATE SET market_open_interest = EXCLUDED.market_open_interest;
    """,
    "fear_greed_index": """
        INSERT INTO fear_greed_index (date, fear_greed, fear_greed_emotion)
        VALUES (%(date)s, %(fear_greed)s, %(fear_greed_emotion)s)
        ON CONFLICT (date) DO UPDATE SET fear_greed = EXCLUDED.fear_greed;
    """,
    "block_trading": """
        INSERT INTO block_trading (date, stock_id, securities_trader_id, securities_trader, price, buy, sell, trade_type)
        VALUES (%(date)s, %(stock_id)s, %(securities_trader_id)s, %(securities_trader)s, %(price)s, %(buy)s, %(sell)s, %(trade_type)s)
        ON CONFLICT (date, stock_id, securities_trader_id, price, trade_type) DO UPDATE SET buy = EXCLUDED.buy;
    """
}

DATASET_MAP = {
    "options_oi_large_holders": "TaiwanOptionOpenInterestLargeTraders",
    "fear_greed_index":         "CnnFearGreedIndex",
    "block_trading":            "TaiwanStockBlockTrade",
}

# =====================================================================
# 2. Mappers
# =====================================================================

def map_opt_large(r: dict) -> dict:
    return {
        "date": r["date"], "option_id": str(r.get("option_id", "")), "put_call": r.get("put_call", ""), "contract_type": str(r.get("contract_type", "")),
        "name": r.get("name", ""), "market_open_interest": safe_float(r.get("market_open_interest")),
        "buy_top5_trader_open_interest": safe_float(r.get("buy_top5_trader_open_interest")), "buy_top5_trader_open_interest_per": safe_float(r.get("buy_top5_trader_open_interest_per")),
        "buy_top10_trader_open_interest": safe_float(r.get("buy_top10_trader_open_interest")), "buy_top10_trader_open_interest_per": safe_float(r.get("buy_top10_trader_open_interest_per")),
        "sell_top5_trader_open_interest": safe_float(r.get("sell_top5_trader_open_interest")), "sell_top5_trader_open_interest_per": safe_float(r.get("sell_top5_trader_open_interest_per")),
        "sell_top10_trader_open_interest": safe_float(r.get("sell_top10_trader_open_interest")), "sell_top10_trader_open_interest_per": safe_float(r.get("sell_top10_trader_open_interest_per")),
        "buy_top5_specific_open_interest": safe_float(r.get("buy_top5_specific_open_interest")), "buy_top5_specific_open_interest_per": safe_float(r.get("buy_top5_specific_open_interest_per")),
        "buy_top10_specific_open_interest": safe_float(r.get("buy_top10_specific_open_interest")), "buy_top10_specific_open_interest_per": safe_float(r.get("buy_top10_specific_open_interest_per")),
        "sell_top5_specific_open_interest": safe_float(r.get("sell_top5_specific_open_interest")), "sell_top5_specific_open_interest_per": safe_float(r.get("sell_top5_specific_open_interest_per")),
        "sell_top10_specific_open_interest": safe_float(r.get("sell_top10_specific_open_interest")), "sell_top10_specific_open_interest_per": safe_float(r.get("sell_top10_specific_open_interest_per"))
    }

def map_fg(r: dict) -> dict:
    return {"date": r["date"], "fear_greed": safe_float(r.get("fear_greed")), "fear_greed_emotion": r.get("fear_greed_emotion", "")[:50]}

def map_block(r: dict) -> dict:
    return {
        "date": r["date"], "stock_id": r["stock_id"], "securities_trader_id": str(r.get("securities_trader_id", "")),
        "securities_trader": r.get("securities_trader", "")[:100], "price": safe_float(r.get("price")),
        "buy": safe_float(r.get("volume", 0.0)), "sell": 0.0, "trade_type": str(r.get("trade_type", "Unknown"))[:50]
    }

MAPPER_MAP = {
    "options_oi_large_holders": map_opt_large,
    "fear_greed_index":         map_fg,
    "block_trading":            map_block,
}

# =====================================================================
# 3. 工具函式
# =====================================================================

def taipei_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    except:
        return date.today().strftime("%Y-%m-%d")

def next_day(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (d + timedelta(days=1)).strftime("%Y-%m-%d")
    except:
        return date_str

def resolve_stock_ids(args) -> List[str]:
    if args.all:
        try:
            with db_session() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE ORDER BY stock_id")
                    sids = [r[0] for r in cur.fetchall()]
            if sids: return sids
        except: pass
        return get_db_stock_ids() or []
    if not args.ids: return []
    return [s.strip() for s in args.ids.split(",") if s.strip()]

# =====================================================================
# 4. 核心抓取邏輯
# =====================================================================

def fetch_one_unit(table: str, unit_id: Optional[str], start: str, end: str, force: bool) -> Tuple[str, int, int]:
    """unit_id 為 None 時代表市場級資料 (如 Fear Greed)"""
    api = FinMindClient()
    fail_log = FailureLogger(f"sentiment_{table}")
    cur_start = start
    
    if not force:
        latest = get_latest_date(table, unit_id)
        if latest:
            cur_start = next_day(str(latest))
            if cur_start > end: return (unit_id or "market"), 0, 0

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 90
        
        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            data = api.get_data(DATASET_MAP[table], unit_id, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [MAPPER_MAP[table](row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], unit_id)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, unit_id, "success" if error == 0 else "partial", "sentiment_v5", seg_start, seg_end, duration_ms)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.3)

        return (unit_id or "market"), total_success, total_error
    except Exception as e:
        logger.error(f"  ❌ {(unit_id or 'market')} @ {table}: {e}")
        fail_log.log_failure(table, unit_id, cur_start, end, str(e))
        return (unit_id or "market"), 0, 0

def main():
    parser = argparse.ArgumentParser(description="Sentiment Data Fetcher v5.0 (Trinity Core Edition)")
    parser.add_argument("--tables", type=str, default="all")
    parser.add_argument("--ids", type=str, default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Step 0: 資產同步
    try:
        sync_stocks_table()
    except: pass

    # 決定目標表
    table_inputs = [t.strip().lower() for t in args.tables.split(",") if t.strip()]
    target_tables = []
    if "all" in table_inputs:
        target_tables = list(MAPPER_MAP.keys())
    else:
        mapping = {"options_large_oi": "options_oi_large_holders", "fear_greed": "fear_greed_index", "block_trading": "block_trading"}
        for t in table_inputs:
            if t in mapping: target_tables.append(mapping[t])
            elif t in DDL_MAP: target_tables.append(t)

    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Sentiment Data Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    logger.info(f"  目標資料表  : {target_tables}")
    logger.info(f"  日期區間    : {args.start} ~ {end_date}")
    logger.info(f"  Workers     : {args.workers}")
    logger.info("=" * 70)

    for table in target_tables:
        ensure_ddl(DDL_MAP[table])
        
        if table == "block_trading":
            # 個股層級
            sids = resolve_stock_ids(args)
            logger.info(f"━━━ 抓取資料表 {table} ({len(sids)} stocks) ━━━")
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(fetch_one_unit, table, sid, args.start, end_date, args.force): sid for sid in sids}
                for fut in as_completed(futures):
                    sid, ok, err = fut.result()
                    if ok: logger.info(f"  ✓ {sid}: {ok} rows")
        else:
            # 市場層級
            logger.info(f"━━━ 抓取資料表 {table} (Market Level) ━━━")
            _, ok, err = fetch_one_unit(table, None, args.start, end_date, args.force)
            if ok: logger.info(f"  ✓ market: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()