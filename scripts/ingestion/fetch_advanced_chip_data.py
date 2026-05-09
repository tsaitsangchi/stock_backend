"""
fetch_advanced_chip_data.py v5.3 (Trinity Core Edition)
================================================================================
進階籌碼資料抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取個股的借券/融券餘額與當日借券回補資訊。

主要功能：
  · 非法 ID 過濾     ─ 自動跳過非個股 ID (如 Automobile)，根治斷路器誤觸。
  · 智慧分段抓取     ─ 實作 30 天切塊邏輯 (v5.3)，解決 V4 API 長區間超時問題。
  · 筆數追蹤紀錄     ─ 完美對接 write_fetch_log，實現 rows_inserted 精準監控。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ Singleton + 智慧斷路器 (排除業務級錯誤)
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.3 (2026-05-09):
    - [穩定] 導入 is_valid_stock_id 過濾機制，徹底解決無效 ID 導致的熔斷問題。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數。
  v5.2 (2026-05-09):
    - [修復] 更正 daily_short_balance 資料集與欄位名稱。
    - [穩定] 下修分段抓取至 30 天。

執行範例：
    # 範例 1：並發抓取全市場借券回補資料
    python scripts/fetchers/fetch_advanced_chip_data.py --all --workers 5
    
    # 範例 2：針對特定標的強制重補
    python scripts/fetchers/fetch_advanced_chip_data.py --stock-id 2330 --tables daily_short_balance --force
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
        get_latest_date, write_fetch_log, safe_int, get_db_stock_ids
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
    "securities_lending": """CREATE TABLE IF NOT EXISTS securities_lending (date DATE NOT NULL, stock_id VARCHAR(20) NOT NULL, ShortSaleTodayBalance BIGINT, ShortSaleYesterdayBalance BIGINT, ShortSaleLimit BIGINT, PRIMARY KEY (date, stock_id));""",
    "daily_short_balance": """CREATE TABLE IF NOT EXISTS daily_short_balance (date DATE NOT NULL, stock_id VARCHAR(20) NOT NULL, ShortSaleTodayBalance BIGINT, ShortSaleYesterdayBalance BIGINT, ShortSaleTodaySell BIGINT, ShortSaleTodayBuy BIGINT, PRIMARY KEY (date, stock_id));""",
}

UPSERT_MAP = {
    "securities_lending": """INSERT INTO securities_lending (date, stock_id, ShortSaleTodayBalance, ShortSaleYesterdayBalance, ShortSaleLimit) VALUES (%(date)s, %(stock_id)s, %(ShortSaleTodayBalance)s, %(ShortSaleYesterdayBalance)s, %(ShortSaleLimit)s) ON CONFLICT (date, stock_id) DO UPDATE SET ShortSaleTodayBalance = EXCLUDED.ShortSaleTodayBalance;""",
    "daily_short_balance": """INSERT INTO daily_short_balance (date, stock_id, ShortSaleTodayBalance, ShortSaleYesterdayBalance, ShortSaleTodaySell, ShortSaleTodayBuy) VALUES (%(date)s, %(stock_id)s, %(ShortSaleTodayBalance)s, %(ShortSaleYesterdayBalance)s, %(ShortSaleTodaySell)s, %(ShortSaleTodayBuy)s) ON CONFLICT (date, stock_id) DO UPDATE SET ShortSaleTodayBalance = EXCLUDED.ShortSaleTodayBalance;""",
}

DATASET_MAP = {"securities_lending": "TaiwanStockMarginPurchaseShortSale", "daily_short_balance": "TaiwanStockMarginPurchaseShortSale"}

# =====================================================================
# 2. 工具與 Mapper
# =====================================================================

def is_valid_stock_id(sid: str) -> bool:
    return sid.isdigit() and len(sid) >= 4

def map_lending(row: dict) -> dict:
    return {"date": row["date"], "stock_id": row["stock_id"], "ShortSaleTodayBalance": safe_int(row.get("ShortSaleTodayBalance", 0)), "ShortSaleYesterdayBalance": safe_int(row.get("ShortSaleYesterdayBalance", 0)), "ShortSaleLimit": safe_int(row.get("ShortSaleLimit", 0))}

def map_short_balance(row: dict) -> dict:
    return {"date": row["date"], "stock_id": row["stock_id"], "ShortSaleTodayBalance": safe_int(row.get("ShortSaleTodayBalance", 0)), "ShortSaleYesterdayBalance": safe_int(row.get("ShortSaleYesterdayBalance", 0)), "ShortSaleTodaySell": safe_int(row.get("ShortSaleSell", 0)), "ShortSaleTodayBuy": safe_int(row.get("ShortSaleBuy", 0))}

MAPPER_MAP = {"securities_lending": map_lending, "daily_short_balance": map_short_balance}

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
    if args.all:
        try:
            with db_session() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE ORDER BY stock_id")
                    return [r[0] for r in cur.fetchall()]
        except: pass
        return get_db_stock_ids() or []
    return [s.strip() for s in args.stock_id.split(",") if s.strip()]

# =====================================================================
# 3. 核心抓取邏輯
# =====================================================================

def fetch_one_stock(table: str, sid: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    if not is_valid_stock_id(sid):
        write_fetch_log(table, sid, "skipped", "advanced_v5.3", start, end, 0, 0, "invalid_stock_id")
        return sid, 0, 0

    cur_start = start
    if not force:
        latest = get_latest_date(table, sid)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, sid, "skipped", "advanced_v5.3", str(latest), end, 0, 0, "up_to_date")
                return sid, 0, 0

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 30
        
        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            data = api.get_data(DATASET_MAP[table], sid, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [MAPPER_MAP[table](row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], sid)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, sid, "success" if error == 0 else "partial", "advanced_v5.3", seg_start, seg_end, duration_ms, success, None)
                else: write_fetch_log(table, sid, "no_new_data", "advanced_v5.3", seg_start, seg_end, duration_ms, 0, None)
            else: write_fetch_log(table, sid, "no_new_data", "advanced_v5.3", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.5)

        return sid, total_success, total_error
    except Exception as e:
        logger.error(f"  ❌ {sid} @ {table}: {e}")
        write_fetch_log(table, sid, "failed", "advanced_v5.3", cur_start, end, 0, 0, str(e))
        return sid, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Advanced Chip Fetcher v5.3 (Trinity Core Edition)")
    parser.add_argument("--stock-id", type=str, default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tables", type=str, default="daily_short_balance")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if not args.stock_id and not args.all:
        parser.print_help()
        sys.exit(1)

    try: sync_stocks_table()
    except: pass

    stock_ids = resolve_stock_ids(args)
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Advanced Chip Fetcher v5.3  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)

    for table in tables:
        if table not in DATASET_MAP: continue
        ensure_ddl(DDL_MAP[table])
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_stock, table, sid, args.start, end_date, args.force): sid for sid in stock_ids}
            for fut in as_completed(futures):
                sid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 全部完成")
    try: get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()