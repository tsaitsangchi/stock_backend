"""
fetch_fred_data.py v5.0 (Trinity Core Edition)
================================================================================
FRED 全球宏觀資料抓取器 — 完美對接 core/ 五大核心模組
此模組負責從 St. Louis Fed (FRED) API 抓取全球宏觀指標，如美債利差、VIX、M2 與 CPI。

主要指標：
  · T10Y2Y   ─ 10 年減 2 年美債利差 (衰退預警指標)
  · VIXCLS   ─ CBOE 波動率指數 (市場情緒)
  · M2SL     ─ 美國 M2 貨幣供給 (流動性)
  · CPIAUCSL ─ 美國消費者物價指數 (通膨)

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.0 (2026-05-09):
    - [核心] 修正 ImportError，移除 get_db_conn，全面換裝 db_transaction。
    - [監控] 對接 Core v4.6，補齊 write_fetch_log 的 rows_inserted 參數。
    - [靈活] 調用 get_latest_date 並指定 id_column="series_id"。
    - [並發] 引入 ThreadPoolExecutor (支援 --workers)，支援多指標並行抓取。
  v4.0 (2026-04-10):
    - [基礎] 建立基礎抓取原型，整合 FRED API 請求。

執行範例：
    # 範例 1：抓取所有預設 FRED 指標 (增量更新)
    python scripts/fetchers/fetch_fred_data.py
    
    # 範例 2：並發抓取特定指標並使用 3 執行緒
    python scripts/fetchers/fetch_fred_data.py --ids T10Y2Y,VIXCLS,DGS10 --workers 3
    
    # 範例 3：全量更新最近一年的總經數據 (強制重補)
    python scripts/fetchers/fetch_fred_data.py --start 2025-01-01 --force
"""

import sys
import os
import argparse
import logging
import time
import random
import requests
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
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. 常數與 DDL
# =====================================================================

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_FRED_SERIES = [
    "T10Y2Y", "T10Y3M", "T10YIE", "VIXCLS", "BAMLH0A0HYM2", 
    "DTWEXBGS", "M2SL", "DGS10", "DGS2", "DGS3MO", 
    "UMCSENT", "INDPRO", "UNRATE", "CPIAUCSL"
]

DDL_FRED = """
    CREATE TABLE IF NOT EXISTS fred_series (
        series_id VARCHAR(50) NOT NULL,
        date DATE NOT NULL,
        value NUMERIC(20,6),
        PRIMARY KEY (series_id, date)
    );
    CREATE INDEX IF NOT EXISTS idx_fred_series_id_date ON fred_series(series_id, date DESC);
"""

UPSERT_FRED = """
    INSERT INTO fred_series (series_id, date, value)
    VALUES (%(series_id)s, %(date)s, %(value)s)
    ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value;
"""

# =====================================================================
# 2. API 客戶端
# =====================================================================

def fred_get(series_id: str, api_key: str, start: str, end: str, max_retries: int = 3) -> list:
    """從 FRED API 獲取資料，包含指數退避重試機制。"""
    params = {
        "series_id": series_id, 
        "api_key": api_key, 
        "file_type": "json", 
        "observation_start": start, 
        "observation_end": end
    }
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(FRED_API_URL, params=params, timeout=(10, 60))
            if resp.status_code == 200:
                return resp.json().get("observations", [])
            elif resp.status_code == 429:
                logger.warning(f"  ⚠️ FRED API (429) 限速，等待 60s... ({attempt}/{max_retries})")
                time.sleep(60)
            else:
                resp.raise_for_status()
        except Exception as e:
            last_error = e
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"  ⚠️ FRED API 失敗 ({e})，{sleep_time:.1f}s 後重試... ({attempt}/{max_retries})")
            time.sleep(sleep_time)
            
    logger.error(f"  ❌ {series_id} 請求失敗達上限: {last_error}")
    return []

# =====================================================================
# 3. 核心抓取邏輯
# =====================================================================

def fetch_one_series(sid: str, api_key: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    fail_log = FailureLogger(f"fred_{sid}")
    cur_start = start
    
    if not force:
        latest = get_latest_date("fred_series", sid, id_column="series_id")
        if latest:
            d = datetime.strptime(str(latest), "%Y-%m-%d").date()
            cur_start = (d + timedelta(days=1)).strftime("%Y-%m-%d")
            if cur_start > end:
                write_fetch_log("fred_series", sid, "skipped", "fred_v5", str(latest), end, 0, 0, "up_to_date")
                return sid, 0, 0

    try:
        t0 = time.monotonic()
        obs = fred_get(sid, api_key, cur_start, end)
        duration_ms = int((time.monotonic() - t0) * 1000)

        if obs:
            records = []
            for o in obs:
                val_str = o.get("value", ".")
                val = safe_float(val_str) if val_str != "." else None
                if val is not None:
                    records.append({"series_id": sid, "date": o.get("date"), "value": val})
            
            if records:
                success, error = commit_per_stock_per_day("fred_series", records, UPSERT_FRED, sid)
                write_fetch_log("fred_series", sid, "success" if error == 0 else "partial", "fred_v5", cur_start, end, duration_ms, success, None)
                return sid, success, error
        
        write_fetch_log("fred_series", sid, "no_new_data", "fred_v5", cur_start, end, duration_ms, 0, None)
        return sid, 0, 0
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000) if 't0' in locals() else 0
        logger.error(f"  ❌ {sid} 抓取崩潰: {e}")
        fail_log.log_failure("fred_series", sid, cur_start, end, str(e))
        write_fetch_log("fred_series", sid, "failed", "fred_v5", cur_start, end, duration_ms, 0, str(e))
        return sid, 0, 0

def main():
    parser = argparse.ArgumentParser(description="FRED Macro Fetcher v5.0 (Trinity Core Edition)")
    parser.add_argument("--ids", type=str, default=",".join(DEFAULT_FRED_SERIES))
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("❌ 嚴重錯誤：未設定環境變數 FRED_API_KEY")
        sys.exit(1)

    target_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    end_date = args.end or date.today().strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"  FRED Macro Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    
    ensure_ddl(DDL_FRED)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one_series, sid, api_key, args.start, end_date, args.force): sid for sid in target_ids}
        for fut in as_completed(futures):
            sid, ok, err = fut.result()
            if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")

if __name__ == "__main__":
    main()