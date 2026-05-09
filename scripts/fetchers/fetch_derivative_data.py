"""
fetch_derivative_data.py — 期貨/選擇權日成交（v3.3 巨量資料防超時版）
================================================================================
v3.3 改進：
  ★ 導入「智慧切塊抓取 (Chunking)」機制：針對 TXO 等巨量資料，自動將日期區間
    切割為每 90 天一塊分批抓取並即時寫入，徹底解決 API Read timed out 的問題。

v3.2 既有（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 使用 `commit_per_stock_per_day` 進行雙層粒度原子寫入，確保極致的資料完整性。

執行範例（常規）：
    python scripts/fetchers/fetch_derivative_data.py
    python scripts/fetchers/fetch_derivative_data.py --tables futures_ohlcv options_ohlcv
    python scripts/fetchers/fetch_derivative_data.py --ids TX MTX --force
    python scripts/fetchers/fetch_derivative_data.py --ids TXO --tables options_ohlcv --force

執行範例（維運與模式切換）：
    python scripts/fetchers/fetch_derivative_data.py --retry-failed 7
    python scripts/fetchers/fetch_derivative_data.py --gap-fill 30
"""

from __future__ import annotations

import sys
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse
import time
from pathlib import Path

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.finmind_client import finmind_get, get_request_stats
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    safe_int,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
    write_fetch_log,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 使用 FinMind Dataset 名稱作為 Key 對應最舊起始日
DATASET_START = {
    "TaiwanFuturesDaily": "1998-07-01", 
    "TaiwanOptionDaily": "2001-12-01"
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")
FALLBACK_FUTURES_IDS = ["TX", "MTX", "TE", "TF"]
FALLBACK_OPTIONS_IDS = ["TXO", "TEO", "TFO"]

_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 日誌與 SQL
# ─────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")

DDL_FUTURES = """
CREATE TABLE IF NOT EXISTS futures_ohlcv (
    date DATE, futures_id VARCHAR(50), contract_date VARCHAR(6), 
    open NUMERIC(20,6), max NUMERIC(20,6), min NUMERIC(20,6), 
    close NUMERIC(20,6), spread NUMERIC(20,6), spread_per NUMERIC(20,6), 
    volume BIGINT, settlement_price NUMERIC(20,6), open_interest BIGINT, 
    trading_session VARCHAR(20), 
    PRIMARY KEY (date, futures_id, contract_date, trading_session)
);
"""
DDL_OPTIONS = """
CREATE TABLE IF NOT EXISTS options_ohlcv (
    date DATE, option_id VARCHAR(50), contract_date VARCHAR(6), 
    strike_price NUMERIC(20,6), call_put VARCHAR(4), 
    open NUMERIC(20,6), max NUMERIC(20,6), min NUMERIC(20,6), 
    close NUMERIC(20,6), volume BIGINT, settlement_price NUMERIC(20,6), 
    open_interest BIGINT, trading_session VARCHAR(20), 
    PRIMARY KEY (date, option_id, contract_date, strike_price, call_put, trading_session)
);
"""

UPSERT_FUTURES = """
INSERT INTO futures_ohlcv (
    date, futures_id, contract_date, open, max, min, close, spread, 
    spread_per, volume, settlement_price, open_interest, trading_session
) VALUES %s 
ON CONFLICT (date, futures_id, contract_date, trading_session) DO UPDATE SET close = EXCLUDED.close;
"""
UPSERT_OPTIONS = """
INSERT INTO options_ohlcv (
    date, option_id, contract_date, strike_price, call_put, 
    open, max, min, close, volume, settlement_price, open_interest, trading_session
) VALUES %s 
ON CONFLICT (date, option_id, contract_date, strike_price, call_put, trading_session) DO UPDATE SET close = EXCLUDED.close;
"""

def map_fut(r): return (r["date"], r.get("futures_id"), str(r.get("contract_date", ""))[:6], safe_float(r.get("open")), safe_float(r.get("max")), safe_float(r.get("min")), safe_float(r.get("close")), safe_float(r.get("spread")), safe_float(r.get("spread_per")), safe_int(r.get("volume")), safe_float(r.get("settlement_price")), safe_int(r.get("open_interest")), str(r.get("trading_session", "") or "")[:20])
def map_opt(r): return (r["date"], r.get("option_id"), str(r.get("contract_date", ""))[:6], safe_float(r.get("strike_price")), str(r.get("call_put", ""))[:4], safe_float(r.get("open")), safe_float(r.get("max")), safe_float(r.get("min")), safe_float(r.get("close")), safe_int(r.get("volume")), safe_float(r.get("settlement_price")), safe_int(r.get("open_interest")), str(r.get("trading_session", "") or "")[:20])

# ─────────────────────────────────────────────
# Fetcher Logic (支援防 Timeout 的 Chunking 切塊抓取)
# ─────────────────────────────────────────────
def fetch_derivative(
    conn, dataset: str, table: str, ddl: str, upsert_sql: str, 
    mapper, ids: list[str], start: str, end: str, delay: float, force: bool,
    fetch_mode_override: str | None = None
):
    ensure_ddl(conn, ddl)
    conn.commit()
    
    # 確保使用正確的 ID 欄位名 (期貨為 futures_id，選擇權為 option_id)
    id_col = "futures_id" if "futures" in table else "option_id"
    latest = get_all_safe_starts(conn, table, key_col=id_col)
    
    flog = FailureLogger(table, db_conn=conn)
    total_rows = 0
    tmpl = "(%s::date, %s, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s, %s::numeric, %s, %s)" if "futures" in table else "(%s::date, %s, %s, %s::numeric, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric, %s, %s::numeric, %s, %s)"
    fetch_mode = fetch_mode_override or "per_stock"
    
    # 設定 Chunk 大小：選擇權資料龐大（每 90 天切一塊），期貨相對較小（每 365 天切一塊）
    chunk_days = 90 if "options" in table else 365
    
    for iid in ids:
        s = resolve_start_cached(iid, latest, start, DATASET_START.get(dataset, "2000-01-01"), force)
        if not s:
            write_fetch_log(conn, table_name=table, stock_id=iid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        stock_total_rows = 0
        seg_start = s
        seg_end_dt = datetime.strptime(end, "%Y-%m-%d")
        
        try:
            # 依區段分批抓取並直接寫入
            while True:
                seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                if seg_start_dt > seg_end_dt:
                    break
                seg_end = min((seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"), end)
                
                logger.info(f"  [{table}/{iid}] 區段抓取：{seg_start} ~ {seg_end}")
                data = finmind_get(
                    dataset=dataset, 
                    params={"data_id": iid, "start_date": seg_start, "end_date": seg_end}, 
                    delay=delay,
                    raise_on_error=True
                )
                
                if data:
                    rows = [mapper(r) for r in data]
                    if "futures" in table:
                        rows = dedup_rows(rows, (0, 1, 2, 12))
                    else:
                        rows = dedup_rows(rows, (0, 1, 2, 3, 4, 12))
                    
                    # 雙層粒度原子寫入確保寫入中斷也不會損毀資料庫
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, label_prefix=table, failure_logger=flog)
                    stock_total_rows += sum(res.values())

                # 下一區段
                seg_start = (datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

            dur = int((time.time() - t0) * 1000)
            total_rows += stock_total_rows
            
            if stock_total_rows > 0:
                write_fetch_log(
                    conn, table_name=table, stock_id=iid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=stock_total_rows, 
                    duration_ms=dur, status="success"
                )
            else:
                write_fetch_log(
                    conn, table_name=table, stock_id=iid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                    duration_ms=dur, status="no_new_data"
                )
                
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            error_msg = str(e)
            flog.record(stock_id=iid, error=error_msg, start_date=s, end_date=end)
            write_fetch_log(
                conn, table_name=table, stock_id=iid, fetch_mode=fetch_mode, 
                fetch_date_from=s, fetch_date_to=end, rows_inserted=stock_total_rows, 
                duration_ms=dur, status="failed", error_message=error_msg
            )
            
    logger.info(f"  [{table}] 本次共寫入 {total_rows} 筆")
    flog.summary()


# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int, target_tables: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    sql = """
    WITH recent AS (
        SELECT table_name, stock_id, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY table_name, stock_id ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = ANY(%s) AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT table_name, stock_id FROM recent WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (target_tables, str(days)))
            for tbl, sid in cur.fetchall():
                targets[tbl].append(sid)
    except Exception as e:
        logger.error(f"[retry-failed] 查詢失敗：{e}")
        return {}

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [retry-failed/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def query_gap_targets(conn, days: int, target_tables: list[str], target_ids_map: dict[str, list[str]]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    for tbl in target_tables:
        all_ids = target_ids_map.get(tbl, [])
        if not all_ids:
            continue
            
        sql = f"SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tbl, str(days), all_ids))
                have_success = {row[0] for row in cur.fetchall()}
            missing = [sid for sid in all_ids if sid not in have_success]
            targets[tbl].extend(missing)
        except Exception as e:
            logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [gap-fill/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def _config_for_table(table: str):
    if table == "futures_ohlcv":
        return ("TaiwanFuturesDaily", DDL_FUTURES, UPSERT_FUTURES, map_fut)
    elif table == "options_ohlcv":
        return ("TaiwanOptionDaily", DDL_OPTIONS, UPSERT_OPTIONS, map_opt)
    return None

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        cfg = _config_for_table(tbl)
        if cfg:
            dataset, ddl, upsert, mapper = cfg
            fetch_derivative(
                conn, dataset, tbl, ddl, upsert, mapper, sids, 
                args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode
            )

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["futures_ohlcv", "options_ohlcv", "all"], default=["all"])
    p.add_argument("--ids", nargs="+", help="指定要抓取的商品代號 (例如 TX TXO)，預設為核心期貨與選擇權")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["futures_ohlcv", "options_ohlcv"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        
        # 決定全域的目標 ID 清單
        target_ids_map = {
            "futures_ohlcv": args.ids if args.ids else FALLBACK_FUTURES_IDS,
            "options_ohlcv": args.ids if args.ids else FALLBACK_OPTIONS_IDS
        }

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, tables)
            if targets: _run_targeted(conn, targets, args, fetch_mode="retry")
            else: logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, tables, target_ids_map)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        if "futures_ohlcv" in tables: 
            fetch_derivative(
                conn, "TaiwanFuturesDaily", "futures_ohlcv", DDL_FUTURES, UPSERT_FUTURES, 
                map_fut, target_ids_map["futures_ohlcv"], args.start, args.end, args.delay, args.force
            )
        if "options_ohlcv" in tables: 
            fetch_derivative(
                conn, "TaiwanOptionDaily", "options_ohlcv", DDL_OPTIONS, UPSERT_OPTIONS, 
                map_opt, target_ids_map["options_ohlcv"], args.start, args.end, args.delay, args.force
            )
            
    finally:
        conn.close()
        logger.info("全部完成")
        # 於程式完全結束時，印出統一的 FinMind RequestStats 報表
        get_request_stats().summary()

if __name__ == "__main__":
    main()