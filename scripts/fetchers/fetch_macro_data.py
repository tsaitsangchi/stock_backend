"""
fetch_macro_data.py v5.0 (Trinity Core Edition)
================================================================================
國內外宏觀經濟數據抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取全球利率、台灣匯率與公債殖利率，為總經模型提供核心輸入指標。

涵蓋資料表：
  · interest_rate  ─ 各國利率 (FinMind: InterestRate, 如 FED, BOJ, ECB)
  · exchange_rate  ─ 台灣匯率 (FinMind: TaiwanExchangeRate, 如 USD, JPY)
  · bond_yield     ─ 公債殖利率 (FinMind: GovernmentBondsYield, 如 US10Y)

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.0      ─ 402 自動休眠 1000s + Singleton 連線
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.0 (2026-05-09):
    - [核心] 修正 ImportError，移除 get_db_conn，全面換裝 db_transaction。
    - [監控] 對接 Core v4.6，補齊 write_fetch_log 的 rows_inserted 參數。
    - [並發] 引入 ThreadPoolExecutor (支援 --workers)，支援多國指標並行抓取。
    - [靈活] 自動識別各表 ID 欄位 (country, currency, bond_id)，確保增量更新邏輯正確。
  v4.0 (2026-04-20):
    - [基礎] 建立基礎抓取原型。

執行範例：
    # 範例 1：抓取所有預設總經數據 (增量更新)
    python scripts/fetchers/fetch_macro_data.py
    
    # 範例 2：並發抓取多國利率並使用 4 執行緒
    python scripts/fetchers/fetch_macro_data.py --tables interest_rate --workers 4
    
    # 範例 3：全量更新最近一年的匯率資料
    python scripts/fetchers/fetch_macro_data.py --tables exchange_rate --start 2025-01-01 --force
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
        safe_float
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

DEFAULT_INTEREST = ["FED", "BOJ", "ECB", "PBOC"]
DEFAULT_EXCHANGE = ["USD", "JPY", "EUR"]
DEFAULT_BOND_MAP = {"US10Y": "United States 10-Year", "US2Y": "United States 2-Year"}

DDL_MAP = {
    "interest_rate": """
        CREATE TABLE IF NOT EXISTS interest_rate (
            date DATE NOT NULL,
            country VARCHAR(50) NOT NULL,
            full_country_name VARCHAR(100),
            interest_rate NUMERIC(20,6),
            PRIMARY KEY (date, country)
        );
    """,
    "exchange_rate": """
        CREATE TABLE IF NOT EXISTS exchange_rate (
            date DATE NOT NULL,
            currency VARCHAR(50) NOT NULL,
            cash_buy NUMERIC(20,6),
            cash_sell NUMERIC(20,6),
            spot_buy NUMERIC(20,6),
            spot_sell NUMERIC(20,6),
            PRIMARY KEY (date, currency)
        );
    """,
    "bond_yield": """
        CREATE TABLE IF NOT EXISTS bond_yield (
            date DATE NOT NULL,
            bond_id VARCHAR(50) NOT NULL,
            value NUMERIC(20,6),
            PRIMARY KEY (date, bond_id)
        );
    """
}

UPSERT_MAP = {
    "interest_rate": """
        INSERT INTO interest_rate (date, country, full_country_name, interest_rate)
        VALUES (%(date)s, %(country)s, %(full_country_name)s, %(interest_rate)s)
        ON CONFLICT (date, country) DO UPDATE SET interest_rate = EXCLUDED.interest_rate;
    """,
    "exchange_rate": """
        INSERT INTO exchange_rate (date, currency, cash_buy, cash_sell, spot_buy, spot_sell)
        VALUES (%(date)s, %(currency)s, %(cash_buy)s, %(cash_sell)s, %(spot_buy)s, %(spot_sell)s)
        ON CONFLICT (date, currency) DO UPDATE SET spot_buy = EXCLUDED.spot_buy;
    """,
    "bond_yield": """
        INSERT INTO bond_yield (date, bond_id, value)
        VALUES (%(date)s, %(bond_id)s, %(value)s)
        ON CONFLICT (date, bond_id) DO UPDATE SET value = EXCLUDED.value;
    """
}

DATASET_MAP = {
    "interest_rate": "InterestRate",
    "exchange_rate": "TaiwanExchangeRate",
    "bond_yield": "GovernmentBondsYield",
}

ID_COL_MAP = {
    "interest_rate": "country",
    "exchange_rate": "currency",
    "bond_yield": "bond_id"
}

# =====================================================================
# 2. Mappers
# =====================================================================

def map_interest(r: dict) -> dict:
    return {
        "date": r["date"], "country": r["country"],
        "full_country_name": r.get("full_country_name"),
        "interest_rate": safe_float(r.get("interest_rate"))
    }

def map_exchange(r: dict) -> dict:
    return {
        "date": r["date"], "currency": r["currency"],
        "cash_buy": safe_float(r.get("cash_buy")), "cash_sell": safe_float(r.get("cash_sell")),
        "spot_buy": safe_float(r.get("spot_buy")), "spot_sell": safe_float(r.get("spot_sell"))
    }

def map_bond(r: dict, bond_id: str) -> dict:
    return {"date": r["date"], "bond_id": bond_id, "value": safe_float(r.get("value"))}

MAPPER_MAP = {
    "interest_rate": map_interest,
    "exchange_rate": map_exchange,
    "bond_yield": map_bond,
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
    fail_log = FailureLogger(f"macro_{table}")
    cur_start = start
    id_col = ID_COL_MAP[table]

    api_id = DEFAULT_BOND_MAP.get(data_id) if table == "bond_yield" else data_id
    
    if not force:
        latest = get_latest_date(table, data_id, id_column=id_col)
        if latest:
            cur_start = next_day(str(latest))
            if cur_start > end:
                write_fetch_log(table, data_id, "skipped", "macro_v5", str(latest), end, 0, 0, "up_to_date")
                return data_id, 0, 0

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
            data = api.get_data(DATASET_MAP[table], api_id, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                if table == "bond_yield":
                    records = [MAPPER_MAP[table](row, data_id) for row in data if "date" in row]
                else:
                    records = [MAPPER_MAP[table](row) for row in data if "date" in row]
                
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], data_id)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, data_id, "success" if error == 0 else "partial", "macro_v5", seg_start, seg_end, duration_ms, success, None)
                else:
                    write_fetch_log(table, data_id, "no_new_data", "macro_v5", seg_start, seg_end, duration_ms, 0, "empty_records")
            else:
                write_fetch_log(table, data_id, "no_new_data", "macro_v5", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.3)

        return data_id, total_success, total_error
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000) if 't0' in locals() else 0
        logger.error(f"  ❌ {data_id} @ {table}: {e}")
        fail_log.log_failure(table, data_id, cur_start, end, str(e))
        write_fetch_log(table, data_id, "failed", "macro_v5", cur_start, end, duration_ms, 0, str(e))
        return data_id, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Macro Data Fetcher v5.0 (Trinity Core Edition)")
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
    logger.info(f"  Macro Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    for table in target_tables:
        ensure_ddl(DDL_MAP[table])
        if table == "interest_rate":
            ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else DEFAULT_INTEREST
        elif table == "exchange_rate":
            ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else DEFAULT_EXCHANGE
        else:
            ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else list(DEFAULT_BOND_MAP.keys())
            
        logger.info(f"━━━ 抓取資料表 {table} ({len(ids)} IDs) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_id, table, tid, args.start, end_date, args.force): tid for tid in ids}
            for fut in as_completed(futures):
                tid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {tid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()