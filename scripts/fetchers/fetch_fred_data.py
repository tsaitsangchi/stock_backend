"""
fetch_fred_data.py — FRED 全球宏觀資料（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 模組化抓取邏輯，與其他核心腳本架構對齊。
  ★ 強化 `fred_get` 錯誤處理與退避重試機制。

v3.1 既有：
  · 整合 fetch_log：每次抓取（無論成功、失敗或跳過）都會寫入監控日誌。
  · 效能追蹤：記錄各總經指標的 API 請求耗時。
  · 導入 commit_per_stock_per_day：每一天、每一指標獨立原子 commit。

執行範例（常規）：
    # 抓取預設的所有 FRED 總經指標
    python scripts/fetchers/fetch_fred_data.py
    
    # 針對特定指標抓取
    python scripts/fetchers/fetch_fred_data.py --ids T10Y2Y VIXCLS
    
    # 強制重抓特定指標
    python scripts/fetchers/fetch_fred_data.py --ids DGS10 --force
    
    # 強制重抓指定日期後的所有指標
    python scripts/fetchers/fetch_fred_data.py --start 2024-01-01 --force

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的指標
    python scripts/fetchers/fetch_fred_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的指標
    python scripts/fetchers/fetch_fred_data.py --gap-fill 30
"""

from __future__ import annotations

import sys
import logging
import os
import random
import time
from pathlib import Path
from datetime import date, datetime, timedelta
import argparse
import requests
from collections import defaultdict

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_FRED_SERIES = [
    "T10Y2Y", "T10Y3M", "T10YIE", "VIXCLS", "BAMLH0A0HYM2", 
    "DTWEXBGS", "M2SL", "DGS10", "DGS2", "DGS3MO", 
    "UMCSENT", "INDPRO", "UNRATE", "CPIAUCSL"
]

DDL_FRED = """CREATE TABLE IF NOT EXISTS fred_series (series_id VARCHAR(50), date DATE, value NUMERIC(20,6), PRIMARY KEY (series_id, date));"""
UPSERT_FRED = """INSERT INTO fred_series (series_id, date, value) VALUES %s ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value;"""

_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 日誌與 API 客戶端
# ─────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")

def _write_fetch_log(conn, **kwargs):
    """寫入 fetch_log，失敗不影響主流程。"""
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO fetch_log (
                run_ts, table_name, stock_id, fetch_mode,
                fetch_date_from, fetch_date_to,
                rows_inserted, rows_updated, duration_ms,
                status, error_message, cli_args
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
            """
            cur.execute(sql, (
                kwargs.get("table_name"), kwargs.get("stock_id"), kwargs.get("fetch_mode", "per_stock"),
                kwargs.get("fetch_date_from"), kwargs.get("fetch_date_to"),
                kwargs.get("rows_inserted", 0), kwargs.get("duration_ms", 0),
                kwargs.get("status"), kwargs.get("error_message"), _CLI_ARGS_STR
            ))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.debug(f"fetch_log 寫入失敗：{e}")

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
                logger.warning(f"FRED API 配額受限 (429)，等待 60 秒後重試... (Attempt {attempt}/{max_retries})")
                time.sleep(60)
            else:
                resp.raise_for_status()
                
        except Exception as e:
            last_error = e
            jitter = random.uniform(0, 1)
            sleep_time = (2 ** attempt) + jitter
            logger.warning(f"FRED API 請求失敗 ({e})，{sleep_time:.1f} 秒後重試... (Attempt {attempt}/{max_retries})")
            time.sleep(sleep_time)
            
    raise RuntimeError(f"FRED API 請求失敗已達最大重試次數 ({max_retries}): {last_error}")

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_fred_series(
    conn, series_ids: list[str], api_key: str, 
    start: str, end: str, delay: float, force: bool,
    fetch_mode_override: str | None = None
):
    logger.info("=== [fred_series] 開始 ===")
    ensure_ddl(conn, DDL_FRED)
    conn.commit()
    
    latest = get_all_safe_starts(conn, "fred_series", key_col="series_id")
    flog = FailureLogger("fred_series", db_conn=conn)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"

    for sid in series_ids:
        s = resolve_start_cached(sid, latest, start, "1990-01-01", force)
        if not s:
            _write_fetch_log(conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            obs = fred_get(sid, api_key, s, end)
            dur = int((time.time() - t0) * 1000)
            
            if obs:
                rows = []
                for o in obs:
                    v = safe_float(o.get("value")) if o.get("value") != "." else None
                    if v is not None: 
                        rows.append((sid, o.get("date"), v))
                        
                if rows:
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(
                        conn, UPSERT_FRED, rows, "(%s, %s, %s)", 
                        stock_index=0, date_index=1, label_prefix=sid, failure_logger=flog
                    )
                    n = sum(res.values())
                    total_rows += n
                    _write_fetch_log(
                        conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode,
                        fetch_date_from=s, fetch_date_to=end, rows_inserted=n, 
                        duration_ms=dur, status="success" if n > 0 else "partial"
                    )
                else:
                    _write_fetch_log(
                        conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode,
                        fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                        duration_ms=dur, status="no_new_data"
                    )
            else:
                _write_fetch_log(
                    conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode,
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                    duration_ms=dur, status="no_new_data"
                )
            time.sleep(delay)
            
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(
                conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode,
                fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                duration_ms=dur, status="failed", error_message=str(e)
            )
            
    logger.info(f"  [fred_series] 總共寫入 {total_rows} 筆")
    flog.summary()


# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int) -> list[str]:
    targets = []
    sql = """
    WITH recent AS (
        SELECT stock_id, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = 'fred_series' AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT stock_id FROM recent WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (str(days),))
            targets = [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"[retry-failed] 查詢失敗：{e}")

    if targets:
        logger.info(f"  [retry-failed/fred_series] 發現 {len(targets)} 個目標 (例：{targets[:5]})")
    return targets

def query_gap_targets(conn, days: int, all_series_ids: list[str]) -> list[str]:
    targets = []
    sql = """
    SELECT DISTINCT stock_id FROM fetch_log 
    WHERE table_name = 'fred_series' AND status = 'success' 
      AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (str(days), all_series_ids))
            have_success = {row[0] for row in cur.fetchall()}
        targets = [sid for sid in all_series_ids if sid not in have_success]
    except Exception as e:
        logger.error(f"[gap-fill/fred_series] 查詢失敗：{e}")

    if targets:
        logger.info(f"  [gap-fill/fred_series] 發現 {len(targets)} 個目標 (例：{targets[:5]})")
    return targets


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+", default=DEFAULT_FRED_SERIES, help="指定要抓取的 FRED 指標代號")
    p.add_argument("--start", default="1990-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("❌ 嚴重錯誤：未設定環境變數 FRED_API_KEY")
        sys.exit(1)

    conn = get_db_conn()
    try:
        _ensure_fetch_log_table(conn)

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed)
            if targets: 
                fetch_fred_series(conn, targets, api_key, args.start, args.end, args.delay, force=True, fetch_mode_override="retry")
            else: 
                logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, args.ids)
            if targets: 
                fetch_fred_series(conn, targets, api_key, args.start, args.end, args.delay, force=True, fetch_mode_override="gap_fill")
            else: 
                logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        fetch_fred_series(conn, args.ids, api_key, args.start, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")

if __name__ == "__main__":
    main()