"""
fetch_cash_flows_data.py v5.1 (Trinity Core Edition)
================================================================================
基本面數據抓取器：現金流量表 & 除權息結果 — 完美對接 core/ 五大核心模組
此模組負責抓取企業的現金流狀態與股利發放數據，為基本面模型提供關鍵特徵。

核心功能：
  · 智慧模式切換     ─ 自動過濾無效 ID (如 Automobile)，防止 422 錯誤觸發熔斷。
  · 斷路器相容       ─ 對接 v5.1 智慧斷路器，業務級權限或配額錯誤不會導致全域阻斷。
  · 筆數精準紀錄     ─ 完美對接 write_fetch_log，實現 rows_inserted 精準監控與版本識別。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ Singleton + 智慧斷路器 (排除業務級錯誤)
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [穩定] 導入 is_valid_stock_id 過濾機制，徹底解決非個股 ID 導致的熔斷問題。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，將模式標記為 cf_v5.1。
  v5.0 (2026-05-09):
    - [核心] 修正 ImportError，換裝 db_transaction，對接 Core v4.6。

執行範例：
    # 範例 1：並發抓取所有個股的現金流量與除權息
    python scripts/fetchers/fetch_cash_flows_data.py --all --workers 5
    
    # 範例 2：針對特定標的強制重補
    python scripts/fetchers/fetch_cash_flows_data.py --stock-id 2330 --force
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
        get_latest_date, write_fetch_log, safe_float, get_db_stock_ids
    )
    from core.finmind_client import FinMindClient, get_request_stats
    from core.migrate_stocks_config import migrate as sync_stocks_table
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. 資料表 DDL
# =====================================================================

DDL_CASH_FLOWS = """CREATE TABLE IF NOT EXISTS cash_flows_statement (date DATE NOT NULL, stock_id VARCHAR(20) NOT NULL, type VARCHAR(100) NOT NULL, value NUMERIC(24,4), origin_name VARCHAR(200), PRIMARY KEY (date, stock_id, type));"""
DDL_DIVIDEND_RESULT = """CREATE TABLE IF NOT EXISTS dividend_result (date DATE NOT NULL, stock_id VARCHAR(20) NOT NULL, before_price NUMERIC(20,6), after_price NUMERIC(20,6), stock_and_cache_dividend NUMERIC(20,6), stock_or_cache_dividend VARCHAR(20), max_price NUMERIC(20,6), min_price NUMERIC(20,6), open_price NUMERIC(20,6), reference_price NUMERIC(20,6), PRIMARY KEY (date, stock_id));"""

UPSERT_MAP = {
    "cash_flows_statement": """INSERT INTO cash_flows_statement (date, stock_id, type, value, origin_name) VALUES (%(date)s, %(stock_id)s, %(type)s, %(value)s, %(origin_name)s) ON CONFLICT (date, stock_id, type) DO UPDATE SET value = EXCLUDED.value;""",
    "dividend_result": """INSERT INTO dividend_result (date, stock_id, before_price, after_price, stock_and_cache_dividend, stock_or_cache_dividend, max_price, min_price, open_price, reference_price) VALUES (%(date)s, %(stock_id)s, %(before_price)s, %(after_price)s, %(stock_and_cache_dividend)s, %(stock_or_cache_dividend)s, %(max_price)s, %(min_price)s, %(open_price)s, %(reference_price)s) ON CONFLICT (date, stock_id) DO UPDATE SET after_price = EXCLUDED.after_price;""",
}

DATASET_MAP = {"cash_flows_statement": "TaiwanStockCashFlowsStatement", "dividend_result": "TaiwanStockDividendResult"}

# =====================================================================
# 2. 工具與 Mapper
# =====================================================================

def is_valid_stock_id(sid: str) -> bool:
    return sid.isdigit() and len(sid) >= 4

def map_cf_row(r: dict) -> dict:
    return {"date": r["date"], "stock_id": r["stock_id"], "type": str(r.get("type", "Unknown"))[:100], "value": safe_float(r.get("value", 0.0)), "origin_name": str(r.get("origin_name", ""))[:200]}

def map_dr_row(r: dict) -> dict:
    return {"date": r["date"], "stock_id": r["stock_id"], "before_price": safe_float(r.get("before_price", 0.0)), "after_price": safe_float(r.get("after_price", 0.0)), "stock_and_cache_dividend": safe_float(r.get("stock_and_cache_dividend", 0.0)), "stock_or_cache_dividend": str(r.get("stock_or_cache_dividend", ""))[:20], "max_price": safe_float(r.get("max_price", 0.0)), "min_price": safe_float(r.get("min_price", 0.0)), "open_price": safe_float(r.get("open_price", 0.0)), "reference_price": safe_float(r.get("reference_price", 0.0))}

MAPPER_MAP = {"cash_flows_statement": map_cf_row, "dividend_result": map_dr_row}

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
# 3. 核心邏輯
# =====================================================================

def fetch_one_stock(table: str, sid: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    if not is_valid_stock_id(sid):
        write_fetch_log(table, sid, "skipped", "cf_v5.1", start, end, 0, 0, "invalid_stock_id")
        return sid, 0, 0

    cur_start = start
    if not force:
        latest = get_latest_date(table, sid)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, sid, "skipped", "cf_v5.1", str(latest), end, 0, 0, "up_to_date")
                return sid, 0, 0

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 365
        
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
                    write_fetch_log(table, sid, "success" if error == 0 else "partial", "cf_v5.1", seg_start, seg_end, duration_ms, success, None)
                else: write_fetch_log(table, sid, "no_new_data", "cf_v5.1", seg_start, seg_end, duration_ms, 0, None)
            else: write_fetch_log(table, sid, "no_new_data", "cf_v5.1", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.3)

        return sid, total_success, total_error
    except Exception as e:
        logger.error(f"  ❌ {sid} @ {table}: {e}")
        write_fetch_log(table, sid, "failed", "cf_v5.1", cur_start, end, 0, 0, str(e))
        return sid, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Cash Flows Fetcher v5.1 (Trinity Core Edition)")
    parser.add_argument("--stock-id", type=str, default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tables", type=str, default="cash_flows_statement,dividend_result")
    parser.add_argument("--start", type=str, default="2021-01-01")
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
    tables = [t.strip().lower() for t in args.tables.split(",") if t.strip()]
    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Cash Flows Fetcher v5.1  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    for table in tables:
        if table not in DATASET_MAP: continue
        ensure_ddl(DDL_CASH_FLOWS if "cash_flow" in table else DDL_DIVIDEND_RESULT)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_stock, table, sid, args.start, end_date, args.force): sid for sid in stock_ids}
            for fut in as_completed(futures):
                sid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try: get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()