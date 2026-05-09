"""
fetch_price_adj_data.py v5.1 (Trinity Core Edition)
================================================================================
還原股價與交易補強資料抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取還原股價、當沖交易量及每日漲跌停價。

涵蓋資料表：
  · price_adj        ─ 還原股價 (TaiwanStockPriceAdj)
  · day_trading      ─ 當沖交易 (TaiwanStockDayTrading)
  · price_limit      ─ 漲跌停價 (TaiwanStockPriceLimit)

核心功能：
  · 智慧模式切換     ─ 自動偵測標的數量，在「全市場批次」與「單一標的」模式間智慧切換。
  · 斷路器保護       ─ 對接 v5.1 核心，排除業務錯誤 (422) 對系統熔斷的干擾。
  · 非法 ID 過濾     ─ 自動過濾非個股 ID (如 Automobile)，維護連線池健康。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ 402 自動休眠 + 智慧斷路器
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 修正 ImportError，移除 finmind_get，全面換裝 FinMindClient()。
    - [核心] 移除 get_db_conn，改用 db_session 與 db_transaction 確保資料一致性。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，實現精準監控。
    - [穩定] 導入 is_valid_stock_id 過濾非數字 ID，防止 Automobile 等代碼觸發熔斷。
  v3.2 (2024-05-01):
    - [基礎] 建立基礎還原股價抓取邏輯。

執行範例：
    # 範例 1：全市場智慧抓取 (優先嘗試批次模式)
    python scripts/fetchers/fetch_price_adj_data.py --all
    
    # 範例 2：僅針對特定個股進行強制抓取
    python scripts/fetchers/fetch_price_adj_data.py --stock-id 2330,2454 --force
    
    # 範例 3：針對當沖交易資料表進行補漏
    python scripts/fetchers/fetch_price_adj_data.py --tables day_trading --days 30
"""

import sys
import argparse
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

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
# 1. 配置與 DDL
# =====================================================================

DDL_PRICE_ADJ = """
CREATE TABLE IF NOT EXISTS price_adj (
    date DATE NOT NULL, stock_id VARCHAR(50) NOT NULL,
    trading_volume BIGINT, trading_money BIGINT,
    open NUMERIC(20,6), max NUMERIC(20,6), min NUMERIC(20,6), close NUMERIC(20,6),
    spread NUMERIC(20,6), trading_turnover NUMERIC(20,6),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_price_adj_stock ON price_adj (stock_id, date DESC);
"""

DDL_DAY_TRADING = """
CREATE TABLE IF NOT EXISTS day_trading (
    date DATE NOT NULL, stock_id VARCHAR(50) NOT NULL,
    buy_after_sale VARCHAR(20), volume BIGINT,
    buy_amount BIGINT, sell_amount BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_day_trading_stock ON day_trading (stock_id, date DESC);
"""

DDL_PRICE_LIMIT = """
CREATE TABLE IF NOT EXISTS price_limit (
    date DATE NOT NULL, stock_id VARCHAR(50) NOT NULL,
    reference_price NUMERIC(20,6), limit_up NUMERIC(20,6), limit_down NUMERIC(20,6),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_price_limit_stock ON price_limit (stock_id, date DESC);
"""

UPSERT_MAP = {
    "price_adj": """INSERT INTO price_adj (date, stock_id, trading_volume, trading_money, open, max, min, close, spread, trading_turnover) VALUES (%(date)s, %(stock_id)s, %(trading_volume)s, %(trading_money)s, %(open)s, %(max)s, %(min)s, %(close)s, %(spread)s, %(trading_turnover)s) ON CONFLICT (date, stock_id) DO UPDATE SET trading_volume = EXCLUDED.trading_volume, close = EXCLUDED.close;""",
    "day_trading": """INSERT INTO day_trading (date, stock_id, buy_after_sale, volume, buy_amount, sell_amount) VALUES (%(date)s, %(stock_id)s, %(buy_after_sale)s, %(volume)s, %(buy_amount)s, %(sell_amount)s) ON CONFLICT (date, stock_id) DO UPDATE SET volume = EXCLUDED.volume, buy_amount = EXCLUDED.buy_amount;""",
    "price_limit": """INSERT INTO price_limit (date, stock_id, reference_price, limit_up, limit_down) VALUES (%(date)s, %(stock_id)s, %(reference_price)s, %(limit_up)s, %(limit_down)s) ON CONFLICT (date, stock_id) DO UPDATE SET reference_price = EXCLUDED.reference_price, limit_up = EXCLUDED.limit_up;"""
}

DATASET_MAP = {
    "price_adj":   "TaiwanStockPriceAdj",
    "day_trading": "TaiwanStockDayTrading",
    "price_limit": "TaiwanStockPriceLimit",
}

# =====================================================================
# 2. 工具函式
# =====================================================================

def is_valid_stock_id(sid: str) -> bool:
    return sid.isdigit() and len(sid) >= 4

def taipei_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    except: return date.today().strftime("%Y-%m-%d")

def next_day(date_str: str) -> str:
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (d + timedelta(days=1)).strftime("%Y-%m-%d")
    except: return str(date_str)

def resolve_stock_ids(args) -> List[str]:
    if args.stock_id: return [s.strip() for s in args.stock_id.split(",") if s.strip()]
    try:
        with db_session() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT stock_id FROM stocks WHERE fetch_basic = TRUE AND is_active = TRUE ORDER BY stock_id")
                return [r[0] for r in cur.fetchall()]
    except: return []

# =====================================================================
# 3. Mappers
# =====================================================================

def map_price_adj(r: dict) -> dict:
    return {"date": r["date"], "stock_id": str(r["stock_id"]), "trading_volume": safe_int(r.get("Trading_Volume")), "trading_money": safe_int(r.get("Trading_money")), "open": safe_float(r.get("open")), "max": safe_float(r.get("max")), "min": safe_float(r.get("min")), "close": safe_float(r.get("close")), "spread": safe_float(r.get("spread")), "trading_turnover": safe_float(r.get("Trading_turnover"))}

def map_day_trading(r: dict) -> dict:
    return {"date": r["date"], "stock_id": str(r["stock_id"]), "buy_after_sale": str(r.get("BuyAfterSale", "")), "volume": safe_int(r.get("Volume")), "buy_amount": safe_int(r.get("BuyAmount")), "sell_amount": safe_int(r.get("SellAmount"))}

def map_price_limit(r: dict) -> dict:
    return {"date": r["date"], "stock_id": str(r["stock_id"]), "reference_price": safe_float(r.get("reference_price")), "limit_up": safe_float(r.get("limit_up")), "limit_down": safe_float(r.get("limit_down"))}

MAPPER_MAP = {"price_adj": map_price_adj, "day_trading": map_day_trading, "price_limit": map_price_limit}

# =====================================================================
# 4. 核心抓取邏輯
# =====================================================================

def fetch_one_stock(table: str, stock_id: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    if not is_valid_stock_id(stock_id):
        write_fetch_log(table, stock_id, "skipped", "adj_v5.1", start, end, 0, 0, "invalid_stock_id")
        return stock_id, 0, 0

    cur_start = start
    if not force:
        latest = get_latest_date(table, stock_id)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, stock_id, "skipped", "adj_v5.1", str(latest), end, 0, 0, "up_to_date")
                return stock_id, 0, 0

    try:
        t0 = time.monotonic()
        data = api.get_data(DATASET_MAP[table], stock_id, cur_start, end)
        duration_ms = int((time.monotonic() - t0) * 1000)
        
        if data:
            records = [MAPPER_MAP[table](row) for row in data if "date" in row]
            if records:
                success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], stock_id)
                write_fetch_log(table, stock_id, "success" if error == 0 else "partial", "adj_v5.1", cur_start, end, duration_ms, success, None)
                return stock_id, success, error
        write_fetch_log(table, stock_id, "no_new_data", "adj_v5.1", cur_start, end, duration_ms, 0, None)
        return stock_id, 0, 0
    except Exception as e:
        logger.error(f"  ❌ {stock_id} @ {table}: {e}")
        write_fetch_log(table, stock_id, "failed", "adj_v5.1", cur_start, end, 0, 0, str(e))
        return stock_id, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Price Adj Fetcher v5.1 (Trinity Core Edition)")
    parser.add_argument("--tables", nargs="+", choices=["price_adj", "day_trading", "price_limit", "all"], default=["all"])
    parser.add_argument("--stock-id", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    try: sync_stocks_table()
    except: pass

    target_tables = ["price_adj", "day_trading", "price_limit"] if "all" in args.tables else args.tables
    stock_ids = resolve_stock_ids(args)
    end_date = args.end or taipei_today()
    start_date = args.start
    if not start_date:
        e_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (e_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"  Price Adj Fetcher v5.1  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)

    for table in target_tables:
        ensure_ddl(globals()[f"DDL_{table.upper()}"])
        logger.info(f"━━━ 抓取資料表 {table} ({len(stock_ids)} stocks) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_stock, table, sid, start_date, end_date, args.force): sid for sid in stock_ids}
            for fut in as_completed(futures):
                sid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 全部完成")
    try: get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()