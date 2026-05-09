"""
fetch_international_data.py v5.1 (Trinity Core Edition)
================================================================================
國際市場資料抓取器 — 完美對接 core/ 五大核心模組
支援智慧分段抓取、多執行緒並發、自動事務管理與筆數監控。

涵蓋資料表：
  · us_stock_price      (美股價量：AAPL, NVDA, TSLA, ...)
  · crude_oil_prices    (原油價格：WTI, Brent)
  · gold_price          (國際金價)

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 自動事務 + 筆數紀錄 (rows_inserted)
  · finmind_client v5.1      ─ Singleton + SQLite 快取 + 智慧斷路器 (排除業務錯誤)
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 對接 FinMindClient v5.1 智慧斷路器，業務錯誤不再導致熔斷。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數。
  v5.0 (2026-05-09):
    - [核心] 修正 ImportError，移除 get_db_conn，全面換裝 db_transaction。

執行範例：
    # 範例 1：抓取所有國際市場資料 (增量更新)
    python scripts/fetchers/fetch_international_data.py
    
    # 範例 2：並發抓取特定美股並使用 5 執行緒
    python scripts/fetchers/fetch_international_data.py --ids AAPL,NVDA,TSLA,MSFT,GOOGL --workers 5
    
    # 範例 3：全量更新金價資料
    python scripts/fetchers/fetch_international_data.py --tables gold_price --force
"""

import sys
import argparse
import logging
import time
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
        safe_float, safe_int
    )
    from core.finmind_client import FinMindClient, get_request_stats
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. 常數與 DDL
# =====================================================================

DEFAULT_US_WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "QQQ", "SPY"]
DEFAULT_CRUDE_IDS = ["WTI", "Brent"]

DDL_MAP = {
    "us_stock_price": """
        CREATE TABLE IF NOT EXISTS us_stock_price (
            date DATE NOT NULL,
            stock_id VARCHAR(50) NOT NULL,
            adj_close NUMERIC(20,6),
            close NUMERIC(20,6),
            high NUMERIC(20,6),
            low NUMERIC(20,6),
            open NUMERIC(20,6),
            volume BIGINT,
            PRIMARY KEY (date, stock_id)
        );
        CREATE INDEX IF NOT EXISTS idx_us_stock_id_date ON us_stock_price (stock_id, date DESC);
    """,
    "crude_oil_prices": """
        CREATE TABLE IF NOT EXISTS crude_oil_prices (
            date DATE NOT NULL,
            name VARCHAR(50) NOT NULL,
            price NUMERIC(20,6),
            PRIMARY KEY (date, name)
        );
    """,
    "gold_price": """
        CREATE TABLE IF NOT EXISTS gold_price (
            date DATE NOT NULL,
            price NUMERIC(20,6),
            PRIMARY KEY (date)
        );
    """
}

UPSERT_MAP = {
    "us_stock_price": """
        INSERT INTO us_stock_price (date, stock_id, adj_close, close, high, low, open, volume)
        VALUES (%(date)s, %(stock_id)s, %(adj_close)s, %(close)s, %(high)s, %(low)s, %(open)s, %(volume)s)
        ON CONFLICT (date, stock_id) DO UPDATE SET adj_close = EXCLUDED.adj_close;
    """,
    "crude_oil_prices": """
        INSERT INTO crude_oil_prices (date, name, price)
        VALUES (%(date)s, %(name)s, %(price)s)
        ON CONFLICT (date, name) DO UPDATE SET price = EXCLUDED.price;
    """,
    "gold_price": """
        INSERT INTO gold_price (date, price)
        VALUES (%(date)s, %(price)s)
        ON CONFLICT (date) DO UPDATE SET price = EXCLUDED.price;
    """
}

DATASET_MAP = {
    "us_stock_price": "USStockPrice",
    "crude_oil_prices": "CrudeOilPrices",
    "gold_price": "GoldPrice",
}

# =====================================================================
# 2. Mappers
# =====================================================================

def map_us_stock(r: dict) -> dict:
    return {
        "date": r["date"], "stock_id": r["stock_id"],
        "adj_close": safe_float(r.get("Adj_Close")), "close": safe_float(r.get("Close")),
        "high": safe_float(r.get("High")), "low": safe_float(r.get("Low")),
        "open": safe_float(r.get("Open")), "volume": safe_int(r.get("Volume"))
    }

def map_crude_oil(r: dict) -> dict:
    return {"date": r["date"], "name": r["name"], "price": safe_float(r.get("price"))}

def map_gold_price(r: dict) -> dict:
    return {"date": r["date"], "price": safe_float(r.get("Price"))}

MAPPER_MAP = {
    "us_stock_price": map_us_stock,
    "crude_oil_prices": map_crude_oil,
    "gold_price": map_gold_price,
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

# =====================================================================
# 4. 核心抓取邏輯
# =====================================================================

def fetch_one_id(table: str, data_id: Optional[str], start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    fail_log = FailureLogger(f"intl_{table}")
    cur_start = start
    
    # 決定 ID 欄位
    id_col = "stock_id" if table == "us_stock_price" else "name"
    if table == "gold_price": id_col = None # 金價表無 ID 欄位
    
    if not force:
        latest = get_latest_date(table, data_id, id_column=id_col) if id_col else get_latest_date(table)
        if latest:
            cur_start = next_day(str(latest))
            if cur_start > end:
                write_fetch_log(table, data_id, "skipped", "intl_v5", str(latest), end, 0, 0, "up_to_date")
                return data_id or "GOLD", 0, 0

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 90
        
        upsert_query = UPSERT_MAP[table]
        mapper = MAPPER_MAP[table]

        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            # 修正參數傳遞邏輯
            data = api.get_data(DATASET_MAP[table], data_id, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [mapper(row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, upsert_query, data_id or "GOLD")
                    total_success += success
                    total_error += error
                    write_fetch_log(table, data_id, "success" if error == 0 else "partial", "intl_v5", seg_start, seg_end, duration_ms, success, None)
                else:
                    write_fetch_log(table, data_id, "no_new_data", "intl_v5", seg_start, seg_end, duration_ms, 0, "empty_records")
            else:
                write_fetch_log(table, data_id, "no_new_data", "intl_v5", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.3)

        return data_id or "GOLD", total_success, total_error
    except Exception as e:
        logger.error(f"  ❌ {data_id or 'GOLD'} @ {table}: {e}")
        fail_log.log_failure(table, data_id or "GOLD", cur_start, end, str(e))
        return data_id or "GOLD", 0, 0

def main():
    parser = argparse.ArgumentParser(description="International Data Fetcher v5.0 (Trinity Core Edition)")
    parser.add_argument("--tables", type=str, default="all")
    parser.add_argument("--ids", type=str, default="")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    table_inputs = [t.strip().lower() for t in args.tables.split(",") if t.strip()]
    target_tables = list(MAPPER_MAP.keys()) if "all" in table_inputs else [t for t in table_inputs if t in MAPPER_MAP]
    
    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  International Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    logger.info(f"  目標資料表  : {target_tables}")
    logger.info(f"  日期區間    : {args.start} ~ {end_date}")
    logger.info(f"  Workers     : {args.workers}")
    logger.info("=" * 70)

    for table in target_tables:
        ensure_ddl(DDL_MAP[table])
        
        # 決定要抓取的 ID
        if table == "us_stock_price":
            ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else DEFAULT_US_WATCHLIST
        elif table == "crude_oil_prices":
            ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else DEFAULT_CRUDE_IDS
        else: # gold_price
            ids = [None] # 金價表無 ID 欄位
            
        logger.info(f"━━━ 抓取資料表 {table} ({len(ids)} IDs) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_id, table, tid, args.start, end_date, args.force): tid for tid in ids}
            for fut in as_completed(futures):
                label, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {label}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()