"""
fetch_fundamental_data.py v5.0 (Trinity Core Edition)
================================================================================
基本面財報抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取企業的綜合損益表與資產負債表，並自動處理財報發布的遞延邏輯。

涵蓋資料表：
  · financial_statements    ─ 綜合損益與資產負債表 (TaiwanStockFinancialStatements)

核心特色：
  · 前視偏誤處理：自動將財報日期遞延 45 天作為發布日 (Publish Date)，確保回測真實性。
  · Schema 自動升級：自動處理舊版欄位長度不足或 Primary Key 不一致的問題。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.0      ─ 402 自動休眠 1000s + Singleton 連線
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.0 (2026-05-09):
    - [核心] 修正 ImportError，移除 get_db_conn，全面換裝 db_transaction。
    - [監控] 對接 Core v4.6，補齊 write_fetch_log 的 rows_inserted 參數。
    - [並發] 引入 ThreadPoolExecutor (支援 --workers)，支援多個指標並行抓取。
    - [穩定] 實作 365 天智慧分段抓取 (Chunking)，解決 V4 API 長區間超時問題。
  v4.0 (2026-04-10):
    - [基礎] 建立基礎抓取原型與 Schema 自動升級機制。

執行範例：
    # 範例 1：抓取台積電財報 (增量更新)
    python scripts/fetchers/fetch_fundamental_data.py --stock-id 2330
    
    # 範例 2：並發抓取全市場財報並使用 5 執行緒
    python scripts/fetchers/fetch_fundamental_data.py --all --workers 5
    
    # 範例 3：全量更新特定日期後的財報 (強制重補)
    python scripts/fetchers/fetch_fundamental_data.py --start 2020-01-01 --force
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
# 1. 常數與 DDL
# =====================================================================

DDL_FINANCIAL_STATEMENT = """
    CREATE TABLE IF NOT EXISTS financial_statements (
        stock_id    VARCHAR(50) NOT NULL,
        date        DATE NOT NULL,
        type        VARCHAR(255) NOT NULL,
        value       NUMERIC(24,4),
        origin_name VARCHAR(255) NOT NULL,
        PRIMARY KEY (stock_id, date, type, origin_name)
    );
    CREATE INDEX IF NOT EXISTS idx_fin_stock_date ON financial_statements (stock_id, date DESC);
"""

UPSERT_FINANCIAL_STATEMENT = """
    INSERT INTO financial_statements (stock_id, date, type, value, origin_name)
    VALUES (%(stock_id)s, %(date)s, %(type)s, %(value)s, %(origin_name)s)
    ON CONFLICT (stock_id, date, type, origin_name) 
    DO UPDATE SET value = EXCLUDED.value;
"""

def upgrade_schema_v5():
    """自動化 Schema 升級：確保欄位長度與 Primary Key 正確"""
    with db_transaction() as cur:
        try:
            cur.execute("ALTER TABLE financial_statements ALTER COLUMN type TYPE VARCHAR(255);")
            cur.execute("ALTER TABLE financial_statements ALTER COLUMN origin_name TYPE VARCHAR(255);")
            cur.execute("ALTER TABLE financial_statements ALTER COLUMN value TYPE NUMERIC(24,4);")
            
            cur.execute("""
                SELECT a.attname
                FROM   pg_index i
                JOIN   pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = 'financial_statements'::regclass AND i.indisprimary;
            """)
            pk_cols = {r['attname'] for r in cur.fetchall()}
            if pk_cols and "origin_name" not in pk_cols:
                logger.warning("🛠️ 偵測到舊版 Primary Key，正在自動升級為 (stock_id, date, type, origin_name)...")
                cur.execute("ALTER TABLE financial_statements DROP CONSTRAINT IF EXISTS financial_statements_pkey;")
                cur.execute("ALTER TABLE financial_statements ADD PRIMARY KEY (stock_id, date, type, origin_name);")
        except Exception as e:
            logger.debug(f"Schema 升級略過: {e}")

# =====================================================================
# 2. Mappers
# =====================================================================

def map_financial_statement(r: dict) -> dict:
    """處理財報資料，並包含遞延 45 天的前視偏誤消除邏輯。"""
    original_date = datetime.strptime(r["date"], "%Y-%m-%d")
    publish_date = original_date + timedelta(days=45)
    return {
        "stock_id": str(r["stock_id"]), 
        "date": publish_date.strftime("%Y-%m-%d"), 
        "type": str(r.get("type", ""))[:255],
        "value": safe_float(r.get("value")), 
        "origin_name": str(r.get("origin_name", r.get("type", "")))[:255]
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
    if not args.stock_id: return []
    return [s.strip() for s in args.stock_id.split(",") if s.strip()]

# =====================================================================
# 4. 核心抓取邏輯
# =====================================================================

def fetch_one_stock(sid: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    fail_log = FailureLogger(f"fundamental_{sid}")
    cur_start = start
    table = "financial_statements"

    if not force:
        latest = get_latest_date(table, sid)
        if latest:
            cur_start = next_day(str(latest))
            if cur_start > end:
                write_fetch_log(table, sid, "skipped", "fund_v5", str(latest), end, 0, 0, "up_to_date")
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
            data = api.get_data("TaiwanStockFinancialStatements", sid, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [map_financial_statement(row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_FINANCIAL_STATEMENT, sid)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, sid, "success" if error == 0 else "partial", "fund_v5", seg_start, seg_end, duration_ms, success, None)
                else:
                    write_fetch_log(table, sid, "no_new_data", "fund_v5", seg_start, seg_end, duration_ms, 0, "empty_records")
            else:
                write_fetch_log(table, sid, "no_new_data", "fund_v5", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.3)

        return sid, total_success, total_error
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000) if 't0' in locals() else 0
        logger.error(f"  ❌ {sid} 抓取崩潰: {e}")
        fail_log.log_failure(table, sid, cur_start, end, str(e))
        write_fetch_log(table, sid, "failed", "fund_v5", cur_start, end, duration_ms, 0, str(e))
        return sid, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Fundamental Data Fetcher v5.0 (Trinity Core Edition)")
    parser.add_argument("--stock-id", type=str, default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    try:
        sync_stocks_table()
        ensure_ddl(DDL_FINANCIAL_STATEMENT)
        upgrade_schema_v5()
    except: pass

    stock_ids = resolve_stock_ids(args)
    if not stock_ids:
        logger.error("找不到個股清單。")
        sys.exit(1)

    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Fundamental Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one_stock, sid, args.start, end_date, args.force): sid for sid in stock_ids}
        for fut in as_completed(futures):
            sid, ok, err = fut.result()
            if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()