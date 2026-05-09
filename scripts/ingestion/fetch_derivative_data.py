"""
fetch_derivative_data.py v5.0 (Trinity Core Edition)
================================================================================
衍生性商品（期貨/選擇權）抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取期貨與選擇權的日成交數據 (OHLCV)，為衍生性商品模型提供核心特徵。

涵蓋資料表：
  · futures_ohlcv    ─ 期貨日成交 (TaiwanFuturesDaily, 如 TX, MTX)
  · options_ohlcv    ─ 選擇權日成交 (TaiwanOptionDaily, 如 TXO)

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.0      ─ 402 自動休眠 1000s + Singleton 連線
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.0 (2026-05-09):
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，實現精準筆數追蹤。
    - [核心] 修正異常紀錄邏輯，確保 402 錯誤會同步寫入 fetch_log。
    - [性能] 引入 ThreadPoolExecutor 並發支援，優化選擇權 90 天智慧分段。
  v4.0 (2026-04-15):
    - [基礎] 建立基礎抓取原型。

執行範例：
    # 範例 1：抓取主要期貨商品 (TX, MTX)
    python scripts/fetchers/fetch_derivative_data.py --tables futures_ohlcv
    
    # 範例 2：並發抓取所有選擇權商品並使用 4 執行緒
    python scripts/fetchers/fetch_derivative_data.py --tables options_ohlcv --workers 4
    
    # 範例 3：針對台指期進行全量更新 (增量更新為預設)
    python scripts/fetchers/fetch_derivative_data.py --ids TX --force
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
        safe_int, safe_float
    )
    from core.finmind_client import FinMindClient, get_request_stats
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. DDL 與 SQL 配置
# =====================================================================

DDL_MAP = {
    "futures_ohlcv": """
        CREATE TABLE IF NOT EXISTS futures_ohlcv (
            date DATE NOT NULL,
            futures_id VARCHAR(50) NOT NULL,
            contract_date VARCHAR(6) NOT NULL,
            open NUMERIC(20,6),
            max NUMERIC(20,6),
            min NUMERIC(20,6),
            close NUMERIC(20,6),
            spread NUMERIC(20,6),
            spread_per NUMERIC(20,6),
            volume BIGINT,
            settlement_price NUMERIC(20,6),
            open_interest BIGINT,
            trading_session VARCHAR(20) NOT NULL,
            PRIMARY KEY (date, futures_id, contract_date, trading_session)
        );
        CREATE INDEX IF NOT EXISTS idx_fut_id_date ON futures_ohlcv(futures_id, date DESC);
    """,
    "options_ohlcv": """
        CREATE TABLE IF NOT EXISTS options_ohlcv (
            date DATE NOT NULL,
            option_id VARCHAR(50) NOT NULL,
            contract_date VARCHAR(6) NOT NULL,
            strike_price NUMERIC(20,6) NOT NULL,
            call_put VARCHAR(4) NOT NULL,
            open NUMERIC(20,6),
            max NUMERIC(20,6),
            min NUMERIC(20,6),
            close NUMERIC(20,6),
            volume BIGINT,
            settlement_price NUMERIC(20,6),
            open_interest BIGINT,
            trading_session VARCHAR(20) NOT NULL,
            PRIMARY KEY (date, option_id, contract_date, strike_price, call_put, trading_session)
        );
        CREATE INDEX IF NOT EXISTS idx_opt_id_date ON options_ohlcv(option_id, date DESC);
    """
}

UPSERT_MAP = {
    "futures_ohlcv": """
        INSERT INTO futures_ohlcv (date, futures_id, contract_date, open, max, min, close, volume, settlement_price, open_interest, trading_session)
        VALUES (%(date)s, %(futures_id)s, %(contract_date)s, %(open)s, %(max)s, %(min)s, %(close)s, %(volume)s, %(settlement_price)s, %(open_interest)s, %(trading_session)s)
        ON CONFLICT (date, futures_id, contract_date, trading_session) DO UPDATE SET close = EXCLUDED.close;
    """,
    "options_ohlcv": """
        INSERT INTO options_ohlcv (date, option_id, contract_date, strike_price, call_put, open, max, min, close, volume, settlement_price, open_interest, trading_session)
        VALUES (%(date)s, %(option_id)s, %(contract_date)s, %(strike_price)s, %(call_put)s, %(open)s, %(max)s, %(min)s, %(close)s, %(volume)s, %(settlement_price)s, %(open_interest)s, %(trading_session)s)
        ON CONFLICT (date, option_id, contract_date, strike_price, call_put, trading_session) DO UPDATE SET close = EXCLUDED.close;
    """
}

DATASET_MAP = {
    "futures_ohlcv": "TaiwanFuturesDaily",
    "options_ohlcv": "TaiwanOptionDaily",
}

# =====================================================================
# 2. Mappers
# =====================================================================

def map_fut(r: dict) -> dict:
    return {
        "date": r["date"], "futures_id": r.get("futures_id"), "contract_date": str(r.get("contract_date", ""))[:6],
        "open": safe_float(r.get("open")), "max": safe_float(r.get("max")), "min": safe_float(r.get("min")),
        "close": safe_float(r.get("close")), "volume": safe_int(r.get("volume")), 
        "settlement_price": safe_float(r.get("settlement_price")), "open_interest": safe_int(r.get("open_interest")), 
        "trading_session": str(r.get("trading_session", "position"))[:20]
    }

def map_opt(r: dict) -> dict:
    return {
        "date": r["date"], "option_id": r.get("option_id"), "contract_date": str(r.get("contract_date", ""))[:6],
        "strike_price": safe_float(r.get("strike_price")), "call_put": str(r.get("call_put", ""))[:4],
        "open": safe_float(r.get("open")), "max": safe_float(r.get("max")), "min": safe_float(r.get("min")),
        "close": safe_float(r.get("close")), "volume": safe_int(r.get("volume")), 
        "settlement_price": safe_float(r.get("settlement_price")), "open_interest": safe_int(r.get("open_interest")), 
        "trading_session": str(r.get("trading_session", "position"))[:20]
    }

MAPPER_MAP = {"futures_ohlcv": map_fut, "options_ohlcv": map_opt}

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
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (d + timedelta(days=1)).strftime("%Y-%m-%d")
    except:
        return str(date_str)

# =====================================================================
# 4. 核心邏輯
# =====================================================================

def fetch_one_id(table: str, iid: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    fail_log = FailureLogger(f"derivative_{table}")
    cur_start = start
    id_col = "futures_id" if "futures" in table else "option_id"
    
    if not force:
        latest = get_latest_date(table, iid, id_column=id_col)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, iid, "skipped", "derivative_v5", str(latest), end, 0, 0, "up_to_date")
                return iid, 0, 0

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 90 if "options" in table else 365
        
        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            data = api.get_data(DATASET_MAP[table], iid, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [MAPPER_MAP[table](row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], iid)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, iid, "success" if error == 0 else "partial", "derivative_v5", seg_start, seg_end, duration_ms, success, None)
                else:
                    write_fetch_log(table, iid, "no_new_data", "derivative_v5", seg_start, seg_end, duration_ms, 0, "empty")
            else:
                write_fetch_log(table, iid, "no_new_data", "derivative_v5", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.3)

        return iid, total_success, total_error
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000) if 't0' in locals() else 0
        logger.error(f"  ❌ {iid} @ {table}: {e}")
        fail_log.log_failure(table, iid, cur_start, end, str(e))
        write_fetch_log(table, iid, "failed", "derivative_v5", cur_start, end, duration_ms, 0, str(e))
        return iid, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Derivative Fetcher v5.0 (Trinity Core Edition)")
    parser.add_argument("--ids", type=str, default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tables", type=str, default="all")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    tables = [t.strip().lower() for t in args.tables.split(",") if t.strip()]
    if "all" in tables: tables = ["futures_ohlcv", "options_ohlcv"]

    target_ids = {
        "futures_ohlcv": [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else ["TX", "MTX", "TE", "TF"],
        "options_ohlcv": [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else ["TXO", "TEO", "TFO"]
    }

    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Derivative Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    for table in tables:
        ensure_ddl(DDL_MAP[table])
        ids = target_ids[table]
        logger.info(f"━━━ 抓取資料表 {table} ({len(ids)} IDs) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_id, table, iid, args.start, end_date, args.force): iid for iid in ids}
            for fut in as_completed(futures):
                iid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {iid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()