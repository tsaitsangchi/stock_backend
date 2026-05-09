"""
fetch_technical_data.py v5.1 (Trinity Core Edition)
================================================================================
技術面價量資料抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取台股日 K 線數據 (開高低收量)，為所有技術指標計算的基礎。

主要功能：
  · 智慧增量抓取   ─ 自動偵測 stock_price 表中各標的的最後日期，實現零重複補抓。
  · 欄位自動對齊   ─ 採用標準名稱 (max, min, trading_volume) 以相容核心資料結構。
  · 非法 ID 過濾   ─ 自動跳過非個股 ID (如 Automobile)，防止觸發斷路器熔斷。
  · 多執行緒並發   ─ 支援 --workers 參數，大幅提升全市場價量同步速度。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ 402 自動休眠 + 智慧斷路器
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 修正欄位名稱 (high -> max, low -> min, volume -> trading_volume) 以對齊現有 Schema。
    - [核心] 升級連線邏輯，支援 FinMindClient v5.1 智慧斷路器。
    - [參數] 新增 --all, --start, --end, --workers 參數。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數。
    - [穩定] 導入 is_valid_stock_id 過濾機制。
  v4.0 (2026-04-15):
    - [基礎] 建立基礎技術面抓取架構。

執行範例：
    # 範例 1：抓取核心標的最近 30 天技術面資料 (增量更新)
    python scripts/ingestion/fetch_technical_data.py --all --days 30
    
    # 範例 2：並發抓取全市場最近 7 天資料
    python scripts/ingestion/fetch_technical_data.py --all --days 7 --workers 5
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
        get_latest_date, write_fetch_log, safe_int, safe_float
    )
    from core.finmind_client import FinMindClient, get_request_stats
    from core.migrate_stocks_config import migrate as sync_stocks_table
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. DDL 與 SQL (對齊現有 Schema)
# =====================================================================

DDL_STOCK_PRICE = """
CREATE TABLE IF NOT EXISTS stock_price (
    date DATE NOT NULL,
    stock_id VARCHAR(50) NOT NULL,
    open NUMERIC(20,6),
    max NUMERIC(20,6),
    min NUMERIC(20,6),
    close NUMERIC(20,6),
    trading_volume BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_stock_price_id_date ON stock_price (stock_id, date DESC);
"""

UPSERT_STOCK_PRICE = """
INSERT INTO stock_price (date, stock_id, open, max, min, close, trading_volume)
VALUES (%(date)s, %(stock_id)s, %(open)s, %(max)s, %(min)s, %(close)s, %(trading_volume)s)
ON CONFLICT (date, stock_id) DO UPDATE SET
    open = EXCLUDED.open, max = EXCLUDED.max, min = EXCLUDED.min, 
    close = EXCLUDED.close, trading_volume = EXCLUDED.trading_volume;
"""

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
                cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE ORDER BY stock_id")
                return [r[0] for r in cur.fetchall()]
    except: return []

# =====================================================================
# 3. 核心抓取邏輯
# =====================================================================

def map_price(row: dict) -> dict:
    return {
        "date": row["date"], "stock_id": row["stock_id"],
        "open": safe_float(row.get("open")), "max": safe_float(row.get("max")),
        "min": safe_float(row.get("min")), "close": safe_float(row.get("close")),
        "trading_volume": safe_int(row.get("Trading_Volume", 0))
    }

def fetch_one_stock(stock_id: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    table = "stock_price"
    
    if not is_valid_stock_id(stock_id):
        write_fetch_log(table, stock_id, "skipped", "tech_v5.1", start, end, 0, 0, "invalid_stock_id")
        return stock_id, 0, 0

    cur_start = start
    if not force:
        latest = get_latest_date(table, stock_id)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, stock_id, "skipped", "tech_v5.1", str(latest), end, 0, 0, "up_to_date")
                return stock_id, 0, 0

    try:
        t0 = time.monotonic()
        data = api.get_data("TaiwanStockPrice", stock_id, cur_start, end)
        duration_ms = int((time.monotonic() - t0) * 1000)
        
        if data:
            records = [map_price(row) for row in data if "date" in row]
            if records:
                success, error = commit_per_stock_per_day(table, records, UPSERT_STOCK_PRICE, stock_id)
                write_fetch_log(table, stock_id, "success" if error == 0 else "partial", "tech_v5.1", cur_start, end, duration_ms, success, None)
                return stock_id, success, error
        
        write_fetch_log(table, stock_id, "no_new_data", "tech_v5.1", cur_start, end, duration_ms, 0, None)
        return stock_id, 0, 0
    except Exception as e:
        logger.error(f"  ❌ {stock_id} @ {table}: {e}")
        write_fetch_log(table, stock_id, "failed", "tech_v5.1", cur_start, end, 0, 0, str(e))
        return stock_id, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Technical Data Fetcher v5.1 (Trinity Core Edition)")
    parser.add_argument("--stock-id", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if not args.stock_id and not args.all:
        parser.print_help()
        sys.exit(1)

    try: sync_stocks_table()
    except: pass

    ensure_ddl(DDL_STOCK_PRICE)
    stock_ids = resolve_stock_ids(args)
    end_date = args.end or taipei_today()
    start_date = args.start
    if not start_date:
        e_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (e_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"  Technical Data Fetcher v5.1  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one_stock, sid, start_date, end_date, args.force): sid for sid in stock_ids}
        for fut in as_completed(futures):
            sid, ok, err = fut.result()
            if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 全部完成")
    try: get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()