"""
fetch_extended_derivative_data.py v5.0 (Trinity Core Edition)
================================================================================
期貨與選擇權三大法人抓取器 — 完美對接 core/ 五大核心模組
支援智慧分段抓取、多執行緒並發與自動事務管理。

涵蓋資料表：
  · futures_inst_investors ─ 期貨三大法人 (FinMind: TaiwanFuturesInstitutionalInvestors)
  · options_inst_investors ─ 選擇權三大法人 (FinMind: TaiwanOptionInstitutionalInvestors)

對接核心模組 (scripts/core/)：
  · db_utils v4.5            ─ 連線池 + 自動事務 + 批次寫入降級 + 筆數紀錄
  · finmind_client v4.1      ─ Singleton + SQLite 快取 + 斷路器保護
  · migrate_stocks_config v3.0 ─ 自動確保 stocks 表配置與 config.py 同步
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.0 (2026-05-09):
    - [核心] 修正 ensure_ddl 參數報錯，全面對接 Core v4.5 並發架構。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，實現精準監控。
    - [性能] 實作 90 天智慧分段抓取 (Chunking)，取代慢速的「逐日抓取」模式。
    - [並發] 引入 ThreadPoolExecutor (支援 --workers)。

執行範例：
    # 範例 1：抓取台指期三大法人 (增量更新)
    python scripts/fetchers/fetch_extended_derivative_data.py --tables futures_inst_investors --ids TX
    
    # 範例 2：並發抓取多個熱門期權商品並使用 3 執行緒
    python scripts/fetchers/fetch_extended_derivative_data.py --ids TX,MTX,TXO --workers 3
    
    # 範例 3：全量更新所有法人衍生品資料
    python scripts/fetchers/fetch_extended_derivative_data.py --all --workers 5
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
    "futures_inst_investors": """
        CREATE TABLE IF NOT EXISTS futures_inst_investors (
            date DATE NOT NULL,
            futures_id VARCHAR(50) NOT NULL,
            institutional_investors VARCHAR(100) NOT NULL,
            long_deal_volume BIGINT,
            long_deal_amount NUMERIC(24,4),
            short_deal_volume BIGINT,
            short_deal_amount NUMERIC(24,4),
            long_open_interest_balance_volume BIGINT,
            long_open_interest_balance_amount NUMERIC(24,4),
            short_open_interest_balance_volume BIGINT,
            short_open_interest_balance_amount NUMERIC(24,4),
            PRIMARY KEY (date, futures_id, institutional_investors)
        );
    """,
    "options_inst_investors": """
        CREATE TABLE IF NOT EXISTS options_inst_investors (
            date DATE NOT NULL,
            option_id VARCHAR(50) NOT NULL,
            call_put VARCHAR(10) NOT NULL,
            institutional_investors VARCHAR(100) NOT NULL,
            long_deal_volume BIGINT,
            long_deal_amount NUMERIC(24,4),
            short_deal_volume BIGINT,
            short_deal_amount NUMERIC(24,4),
            long_open_interest_balance_volume BIGINT,
            long_open_interest_balance_amount NUMERIC(24,4),
            short_open_interest_balance_volume BIGINT,
            short_open_interest_balance_amount NUMERIC(24,4),
            PRIMARY KEY (date, option_id, call_put, institutional_investors)
        );
    """
}

UPSERT_FUT = """
    INSERT INTO futures_inst_investors (
        date, futures_id, institutional_investors, long_deal_volume, long_deal_amount,
        short_deal_volume, short_deal_amount, long_open_interest_balance_volume,
        long_open_interest_balance_amount, short_open_interest_balance_volume,
        short_open_interest_balance_amount
    ) VALUES (
        %(date)s, %(futures_id)s, %(institutional_investors)s, %(long_deal_volume)s, %(long_deal_amount)s,
        %(short_deal_volume)s, %(short_deal_amount)s, %(long_open_interest_balance_volume)s,
        %(long_open_interest_balance_amount)s, %(short_open_interest_balance_volume)s,
        %(short_open_interest_balance_amount)s
    ) ON CONFLICT (date, futures_id, institutional_investors) DO UPDATE SET long_deal_volume = EXCLUDED.long_deal_volume;
"""

UPSERT_OPT = """
    INSERT INTO options_inst_investors (
        date, option_id, call_put, institutional_investors, long_deal_volume, long_deal_amount,
        short_deal_volume, short_deal_amount, long_open_interest_balance_volume,
        long_open_interest_balance_amount, short_open_interest_balance_volume,
        short_open_interest_balance_amount
    ) VALUES (
        %(date)s, %(option_id)s, %(call_put)s, %(institutional_investors)s, %(long_deal_volume)s, %(long_deal_amount)s,
        %(short_deal_volume)s, %(short_deal_amount)s, %(long_open_interest_balance_volume)s,
        %(long_open_interest_balance_amount)s, %(short_open_interest_balance_volume)s,
        %(short_open_interest_balance_amount)s
    ) ON CONFLICT (date, option_id, call_put, institutional_investors) DO UPDATE SET long_deal_volume = EXCLUDED.long_deal_volume;
"""

DATASET_MAP = {
    "futures_inst_investors": "TaiwanFuturesInstitutionalInvestors",
    "options_inst_investors": "TaiwanOptionInstitutionalInvestors",
}

# =====================================================================
# 2. Mappers
# =====================================================================

def map_fut(r: dict) -> dict:
    return {
        "date": r["date"], "futures_id": r.get("futures_id") or r.get("name"),
        "institutional_investors": r.get("institutional_investors"),
        "long_deal_volume": safe_int(r.get("long_deal_volume")), "long_deal_amount": safe_float(r.get("long_deal_amount")),
        "short_deal_volume": safe_int(r.get("short_deal_volume")), "short_deal_amount": safe_float(r.get("short_deal_amount")),
        "long_open_interest_balance_volume": safe_int(r.get("long_open_interest_balance_volume")),
        "long_open_interest_balance_amount": safe_float(r.get("long_open_interest_balance_amount")),
        "short_open_interest_balance_volume": safe_int(r.get("short_open_interest_balance_volume")),
        "short_open_interest_balance_amount": safe_float(r.get("short_open_interest_balance_amount"))
    }

def map_opt(r: dict) -> dict:
    return {
        "date": r["date"], "option_id": r.get("option_id") or r.get("name"),
        "call_put": r.get("call_put"), "institutional_investors": r.get("institutional_investors"),
        "long_deal_volume": safe_int(r.get("long_deal_volume")), "long_deal_amount": safe_float(r.get("long_deal_amount")),
        "short_deal_volume": safe_int(r.get("short_deal_volume")), "short_deal_amount": safe_float(r.get("short_deal_amount")),
        "long_open_interest_balance_volume": safe_int(r.get("long_open_interest_balance_volume")),
        "long_open_interest_balance_amount": safe_float(r.get("long_open_interest_balance_amount")),
        "short_open_interest_balance_volume": safe_int(r.get("short_open_interest_balance_volume")),
        "short_open_interest_balance_amount": safe_float(r.get("short_open_interest_balance_amount"))
    }

MAPPER_MAP = {
    "futures_inst_investors": map_fut,
    "options_inst_investors": map_opt,
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

def fetch_one_id(table: str, data_id: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    fail_log = FailureLogger(f"derivative_ext_{table}")
    cur_start = start
    
    # 決定 ID 欄位 (DDL 裡 futures_id 或 option_id)
    id_col = "futures_id" if "futures" in table else "option_id"
    
    if not force:
        latest = get_latest_date(table, data_id, id_column=id_col)
        if latest:
            cur_start = next_day(str(latest))
            if cur_start > end:
                write_fetch_log(table, data_id, "skipped", "ext_der_v5", str(latest), end, 0, 0, "up_to_date")
                return data_id, 0, 0

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 90
        
        upsert_query = UPSERT_FUT if "futures" in table else UPSERT_OPT
        mapper = MAPPER_MAP[table]

        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            data = api.get_data(DATASET_MAP[table], data_id, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [mapper(row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, upsert_query, data_id)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, data_id, "success" if error == 0 else "partial", "ext_der_v5", seg_start, seg_end, duration_ms, success, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.3)

        return data_id, total_success, total_error
    except Exception as e:
        logger.error(f"  ❌ {data_id} @ {table}: {e}")
        fail_log.log_failure(table, data_id, cur_start, end, str(e))
        return data_id, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Extended Derivative Fetcher v5.0 (Trinity Core Edition)")
    parser.add_argument("--tables", type=str, default="all")
    parser.add_argument("--ids", type=str, default="TX,MTX,TXO")
    parser.add_argument("--all", action="store_true", help="自動抓取常用熱門期權商品")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # 決定商品清單
    if args.all:
        target_ids = ["TX", "MTX", "TE", "TF", "TXO", "TEO", "TFO"]
    else:
        target_ids = [s.strip() for s in args.ids.split(",") if s.strip()]

    # 決定目標表
    table_inputs = [t.strip().lower() for t in args.tables.split(",") if t.strip()]
    target_tables = list(MAPPER_MAP.keys()) if "all" in table_inputs else [t for t in table_inputs if t in MAPPER_MAP]

    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Extended Derivative Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    logger.info(f"  目標商品    : {target_ids}")
    logger.info(f"  目標資料表  : {target_tables}")
    logger.info(f"  日期區間    : {args.start} ~ {end_date}")
    logger.info(f"  Workers     : {args.workers}")
    logger.info("=" * 70)

    for table in target_tables:
        ensure_ddl(DDL_MAP[table])
        logger.info(f"━━━ 抓取資料表 {table} ({len(target_ids)} IDs) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_id, table, tid, args.start, end_date, args.force): tid for tid in target_ids}
            for fut in as_completed(futures):
                tid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {tid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()