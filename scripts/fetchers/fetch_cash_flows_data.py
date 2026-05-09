"""
fetch_cash_flows_data.py v5.0 (Trinity Core Edition)
================================================================================
基本面數據抓取器：現金流量表 & 除權息結果
此模組負責抓取企業的現金流狀態與股利發放數據，為基本面模型提供關鍵特徵。

涵蓋資料表：
  · cash_flows_statement    ─ 現金流量表 (TaiwanStockCashFlowsStatement)
  · dividend_result         ─ 除權息結果 (TaiwanStockDividendResult)

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.0      ─ 402 自動休眠 1000s + Singleton 連線
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.0 (2026-05-09):
    - [核心] 修正 ImportError，換裝 db_transaction，對接 Core v4.6。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數。
    - [並發] 維持 ThreadPoolExecutor 支援，優化分段抓取為 365 天。
  v4.0 (2026-04-15):
    - [基礎] 建立基礎抓取原型。

執行範例：
    # 範例 1：抓取台積電現金流量與除權息 (增量更新)
    python scripts/fetchers/fetch_cash_flows_data.py --stock-id 2330
    
    # 範例 2：並發抓取所有個股 (依 stocks 表設定) 並使用 5 執行緒
    python scripts/fetchers/fetch_cash_flows_data.py --all --workers 5
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
        safe_float, get_db_stock_ids
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

DDL_CASH_FLOWS = """
CREATE TABLE IF NOT EXISTS cash_flows_statement (
    date DATE NOT NULL,
    stock_id VARCHAR(20) NOT NULL,
    type VARCHAR(100) NOT NULL,
    value NUMERIC(24,4),
    origin_name VARCHAR(200),
    PRIMARY KEY (date, stock_id, type)
);
CREATE INDEX IF NOT EXISTS idx_cf_stock_date ON cash_flows_statement(stock_id, date DESC);
"""

DDL_DIVIDEND_RESULT = """
CREATE TABLE IF NOT EXISTS dividend_result (
    date DATE NOT NULL,
    stock_id VARCHAR(20) NOT NULL,
    before_price NUMERIC(20,6),
    after_price NUMERIC(20,6),
    stock_and_cache_dividend NUMERIC(20,6),
    stock_or_cache_dividend VARCHAR(20),
    max_price NUMERIC(20,6),
    min_price NUMERIC(20,6),
    open_price NUMERIC(20,6),
    reference_price NUMERIC(20,6),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_dr_stock_date ON dividend_result(stock_id, date DESC);
"""

UPSERT_MAP = {
    "cash_flows_statement": """
        INSERT INTO cash_flows_statement (date, stock_id, type, value, origin_name)
        VALUES (%(date)s, %(stock_id)s, %(type)s, %(value)s, %(origin_name)s)
        ON CONFLICT (date, stock_id, type) DO UPDATE SET value = EXCLUDED.value;
    """,
    "dividend_result": """
        INSERT INTO dividend_result (date, stock_id, before_price, after_price, stock_and_cache_dividend, stock_or_cache_dividend, max_price, min_price, open_price, reference_price)
        VALUES (%(date)s, %(stock_id)s, %(before_price)s, %(after_price)s, %(stock_and_cache_dividend)s, %(stock_or_cache_dividend)s, %(max_price)s, %(min_price)s, %(open_price)s, %(reference_price)s)
        ON CONFLICT (date, stock_id) DO UPDATE SET after_price = EXCLUDED.after_price;
    """,
}

DATASET_MAP = {
    "cash_flows_statement": "TaiwanStockCashFlowsStatement",
    "dividend_result": "TaiwanStockDividendResult",
}

# =====================================================================
# 2. Mappers
# =====================================================================

def map_cf_row(r: dict) -> dict:
    return {
        "date": r["date"], "stock_id": r["stock_id"],
        "type": r.get("type", "Unknown")[:100],
        "value": safe_float(r.get("value", 0.0)),
        "origin_name": r.get("origin_name", "")[:200]
    }

def map_dr_row(r: dict) -> dict:
    return {
        "date": r["date"], "stock_id": r["stock_id"],
        "before_price": safe_float(r.get("before_price", 0.0)),
        "after_price": safe_float(r.get("after_price", 0.0)),
        "stock_and_cache_dividend": safe_float(r.get("stock_and_cache_dividend", 0.0)),
        "stock_or_cache_dividend": r.get("stock_or_cache_dividend", "")[:20],
        "max_price": safe_float(r.get("max_price", 0.0)),
        "min_price": safe_float(r.get("min_price", 0.0)),
        "open_price": safe_float(r.get("open_price", 0.0)),
        "reference_price": safe_float(r.get("reference_price", 0.0))
    }

MAPPER_MAP = {
    "cash_flows_statement": map_cf_row,
    "dividend_result": map_dr_row,
}

# =====================================================================
# 3. 工具
# =====================================================================

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
    if args.all:
        try:
            with db_session() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT stock_id FROM stocks WHERE fetch_fundamental = TRUE AND is_active = TRUE ORDER BY stock_id")
                    sids = [r[0] for r in cur.fetchall()]
            if sids: return sids
        except: pass
        return get_db_stock_ids() or []
    if not args.stock_id: return []
    return [s.strip() for s in args.stock_id.split(",") if s.strip()]

# =====================================================================
# 4. 核心邏輯
# =====================================================================

def fetch_one_stock(table: str, sid: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    fail_log = FailureLogger(f"cf_{table}")
    cur_start = start
    
    if not force:
        latest = get_latest_date(table, sid)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, sid, "skipped", "cf_v5", str(latest), end, 0, 0, "up_to_date")
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
                    write_fetch_log(table, sid, "success" if error == 0 else "partial", "cf_v5", seg_start, seg_end, duration_ms, success, None)
                else:
                    write_fetch_log(table, sid, "no_new_data", "cf_v5", seg_start, seg_end, duration_ms, 0, "empty_records")
            else:
                write_fetch_log(table, sid, "no_new_data", "cf_v5", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.3)

        return sid, total_success, total_error
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000) if 't0' in locals() else 0
        logger.error(f"  ❌ {sid} @ {table}: {e}")
        fail_log.log_failure(table, sid, cur_start, end, str(e))
        write_fetch_log(table, sid, "failed", "cf_v5", cur_start, end, duration_ms, 0, str(e))
        return sid, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Cash Flows Fetcher v5.0 (Trinity Core Edition)")
    parser.add_argument("--stock-id", type=str, default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tables", type=str, default="cash_flows_statement,dividend_result")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    try:
        sync_stocks_table()
    except Exception as e:
        logger.warning(f"資產同步略過: {e}")

    stock_ids = resolve_stock_ids(args)
    if not stock_ids:
        logger.error("找不到個股清單。")
        sys.exit(1)

    tables = [t.strip().lower() for t in args.tables.split(",") if t.strip()]
    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Cash Flows Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    for table in tables:
        if table == "cash_flows_statement": ensure_ddl(DDL_CASH_FLOWS)
        elif table == "dividend_result": ensure_ddl(DDL_DIVIDEND_RESULT)
        else: continue
        
        logger.info(f"━━━ 抓取資料表 {table} ({len(stock_ids)} stocks) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_stock, table, sid, args.start, end_date, args.force): sid for sid in stock_ids}
            for fut in as_completed(futures):
                sid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()