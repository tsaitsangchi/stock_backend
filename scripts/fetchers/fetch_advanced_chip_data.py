"""
fetch_advanced_chip_data.py v5.2 (Trinity Core Edition)
================================================================================
進階籌碼資料抓取器 — 完美對接 core/ 五大核心模組

涵蓋資料表：
  · securities_lending      ─ 借券/融券餘額 (FinMind: TaiwanStockMarginPurchaseShortSale)
  · daily_short_balance     ─ 當日借券回補   (FinMind: TaiwanStockMarginPurchaseShortSale)

對接核心模組 (scripts/core/)：
  · db_utils v4.5            ─ 連線池 + 自動事務 + 批次寫入降級
  · finmind_client v4.1      ─ Singleton + SQLite 快取 + 斷路器保護
  · migrate_stocks_config v3.0 ─ 自動確保 stocks 表配置與 config.py 同步
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.2 (2026-05-09):
    - [修復] 將 daily_short_balance 資料集更正為 TaiwanStockMarginPurchaseShortSale。
    - [修復] 修正 FinMind V4 欄位名稱 (ShortSaleBuy, ShortSaleSell 等)。
    - [穩定] 將分段抓取 (Chunking) 下修至 30 天，徹底根除 422 錯誤。
  v5.1 (2026-05-09):
    - [核心] 實作 60 天分段抓取，處理資料表結構衝突。
    - [整合] 引入 migrate_stocks_config.migrate()。
  v5.0 (2026-05-08):
    - [升級] 完美對接 Core v4.5 並發架構。

執行範例：
    python scripts/fetchers/fetch_advanced_chip_data.py --stock-id 2330 --tables daily_short_balance
    python scripts/fetchers/fetch_advanced_chip_data.py --all --workers 5
"""

import sys
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# ── 系統路徑修復 ──
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
# 1. 資料表 DDL 與 UPSERT SQL
# =====================================================================

DDL_MAP = {
    "securities_lending": """
        CREATE TABLE IF NOT EXISTS securities_lending (
            date                       DATE        NOT NULL,
            stock_id                   VARCHAR(20) NOT NULL,
            ShortSaleTodayBalance      BIGINT,
            ShortSaleYesterdayBalance  BIGINT,
            ShortSaleLimit             BIGINT,
            PRIMARY KEY (date, stock_id)
        );
        CREATE INDEX IF NOT EXISTS idx_sec_lend_stock_date ON securities_lending(stock_id, date DESC);
    """,
    "daily_short_balance": """
        CREATE TABLE IF NOT EXISTS daily_short_balance (
            date                       DATE        NOT NULL,
            stock_id                   VARCHAR(20) NOT NULL,
            ShortSaleTodayBalance      BIGINT,
            ShortSaleYesterdayBalance  BIGINT,
            ShortSaleTodaySell         BIGINT,
            ShortSaleTodayBuy          BIGINT,
            PRIMARY KEY (date, stock_id)
        );
        CREATE INDEX IF NOT EXISTS idx_short_bal_stock_date ON daily_short_balance(stock_id, date DESC);
    """,
}

UPSERT_MAP = {
    "securities_lending": """
        INSERT INTO securities_lending
            (date, stock_id, ShortSaleTodayBalance, ShortSaleYesterdayBalance, ShortSaleLimit)
        VALUES
            (%(date)s, %(stock_id)s, %(ShortSaleTodayBalance)s, %(ShortSaleYesterdayBalance)s, %(ShortSaleLimit)s)
        ON CONFLICT (date, stock_id) DO UPDATE SET
            ShortSaleTodayBalance     = EXCLUDED.ShortSaleTodayBalance,
            ShortSaleYesterdayBalance = EXCLUDED.ShortSaleYesterdayBalance,
            ShortSaleLimit            = EXCLUDED.ShortSaleLimit;
    """,
    "daily_short_balance": """
        INSERT INTO daily_short_balance
            (date, stock_id, ShortSaleTodayBalance, ShortSaleYesterdayBalance,
             ShortSaleTodaySell, ShortSaleTodayBuy)
        VALUES
            (%(date)s, %(stock_id)s, %(ShortSaleTodayBalance)s, %(ShortSaleYesterdayBalance)s,
             %(ShortSaleTodaySell)s, %(ShortSaleTodayBuy)s)
        ON CONFLICT (date, stock_id) DO UPDATE SET
            ShortSaleTodayBalance     = EXCLUDED.ShortSaleTodayBalance,
            ShortSaleYesterdayBalance = EXCLUDED.ShortSaleYesterdayBalance,
            ShortSaleTodaySell        = EXCLUDED.ShortSaleTodaySell,
            ShortSaleTodayBuy         = EXCLUDED.ShortSaleTodayBuy;
    """,
}

DATASET_MAP = {
    "securities_lending":  "TaiwanStockMarginPurchaseShortSale",
    "daily_short_balance": "TaiwanStockMarginPurchaseShortSale",
}

# =====================================================================
# 2. 資料對映器 (Mappers)
# =====================================================================

def map_lending(row: dict) -> dict:
    return {
        "date": row["date"],
        "stock_id": row["stock_id"],
        "ShortSaleTodayBalance": safe_int(row.get("ShortSaleTodayBalance", 0)),
        "ShortSaleYesterdayBalance": safe_int(row.get("ShortSaleYesterdayBalance", 0)),
        "ShortSaleLimit": safe_int(row.get("ShortSaleLimit", 0)),
    }

def map_short_balance(row: dict) -> dict:
    return {
        "date":                       row["date"],
        "stock_id":                   row["stock_id"],
        "ShortSaleTodayBalance":      safe_int(row.get("ShortSaleTodayBalance", 0)),
        "ShortSaleYesterdayBalance":  safe_int(row.get("ShortSaleYesterdayBalance", 0)),
        "ShortSaleTodaySell":         safe_int(row.get("ShortSaleSell", 0)),
        "ShortSaleTodayBuy":          safe_int(row.get("ShortSaleBuy", 0)),
    }

MAPPER_MAP: dict[str, Callable[[dict], dict]] = {
    "securities_lending":  map_lending,
    "daily_short_balance": map_short_balance,
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
                    cur.execute("SELECT stock_id FROM stocks WHERE fetch_chip = TRUE AND is_active = TRUE ORDER BY stock_id")
                    sids = [r[0] for r in cur.fetchall()]
            if sids: return sids
        except: pass
        return get_db_stock_ids() or []
    if not args.stock_id: return []
    return [s.strip() for s in args.stock_id.split(",") if s.strip()]

# =====================================================================
# 4. 核心抓取邏輯
# =====================================================================

def fetch_one_stock(table: str, stock_id: str, start: str, end: str, force: bool, dry_run: bool) -> Tuple[str, int, int, str]:
    api = FinMindClient()
    fail_log = FailureLogger("advanced_chip")
    cur_start = start
    if not force:
        latest = get_latest_date(table, stock_id)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end: return stock_id, 0, 0, "skip"

    if dry_run: return stock_id, 0, 0, "ok"

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 30 # v5.2 下修至 30 天確保穩定
        
        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            data = api.get_data(DATASET_MAP[table], stock_id, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [MAPPER_MAP[table](row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], stock_id)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, stock_id, "success" if error == 0 else "partial", "advanced_v5", seg_start, seg_end, duration_ms)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.5)

        return stock_id, total_success, total_error, "ok"
    except Exception as e:
        logger.error(f"  ❌ {stock_id} @ {table}: {e}")
        fail_log.log_failure(table, stock_id, cur_start, end, str(e))
        return stock_id, 0, 0, "error"

def main():
    parser = argparse.ArgumentParser(description="Advanced Chip Fetcher v5.2 (Trinity Core Edition)")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--stock-id", type=str, default="")
    grp.add_argument("--all", action="store_true")
    parser.add_argument("--tables", type=str, default="daily_short_balance")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("正在執行資產清單與 DDL 預檢...")
    try:
        sync_stocks_table()
    except Exception as e:
        logger.warning(f"資產同步略過: {e}")

    stock_ids = resolve_stock_ids(args)
    if not stock_ids:
        logger.error("找不到個股清單。")
        sys.exit(1)

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    end_date = args.end or taipei_today()

    for table in tables:
        if table not in DATASET_MAP: continue
        ensure_ddl(DDL_MAP[table])
        
        logger.info(f"━━━ 抓取資料表 {table} ({len(stock_ids)} stocks) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_stock, table, sid, args.start, end_date, args.force, args.dry_run): sid for sid in stock_ids}
            for fut in as_completed(futures):
                sid, ok, err, status = fut.result()
                if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()