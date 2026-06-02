"""
fetch_event_risk_data.py — 事件風險與股本變動（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 預設改為讀取 `stocks` 表中 `is_active=True` 的活躍核心標的。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

執行範例（常規）：
    # 抓取所有事件風險與股本資料
    python scripts/fetchers/fetch_event_risk_data.py
    
    # 針對特定資料表抓取
    python scripts/fetchers/fetch_event_risk_data.py --tables market_value disposition_securities
    
    # 針對特定個股抓取特定資料 (強制更新)
    python scripts/fetchers/fetch_event_risk_data.py --stock-id 2330,2454 --tables disposition_securities --force
    python scripts/fetchers/fetch_event_risk_data.py --stock-id 2330 --tables all --force

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的組合
    python scripts/fetchers/fetch_event_risk_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python scripts/fetchers/fetch_event_risk_data.py --gap-fill 30
    
    # 針對處置股特定補抓
    python scripts/fetchers/fetch_event_risk_data.py --gap-fill 14 --tables disposition_securities
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
    get_db_stock_ids,
    get_core_stocks_from_db,
    ensure_ddl,
    safe_float,
    safe_int,
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

DATASET_START = {
    "delisting":              "2001-01-01",
    "suspended":              "2011-10-06",
    "capital_reduction":      "2011-01-01",
    "split_price":            "2000-01-01",
    "trading_date":           "1990-01-01",
    "market_value":           "2004-01-01",
    "disposition_securities": "2001-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

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

DDL_EVENT = """
CREATE TABLE IF NOT EXISTS delisting (date DATE, stock_id VARCHAR(50), stock_name VARCHAR(200), PRIMARY KEY (date, stock_id));
CREATE TABLE IF NOT EXISTS suspended (stock_id VARCHAR(50), date DATE, suspension_time VARCHAR(50), resumption_date DATE, resumption_time VARCHAR(50), PRIMARY KEY (stock_id, date, suspension_time));
CREATE TABLE IF NOT EXISTS capital_reduction (date DATE, stock_id VARCHAR(50), closing_last_trading NUMERIC(20,4), post_reduction_ref NUMERIC(20,4), limit_up NUMERIC(20,4), limit_down NUMERIC(20,4), opening_ref NUMERIC(20,4), exright_ref NUMERIC(20,4), reason VARCHAR(500), PRIMARY KEY (date, stock_id));
CREATE TABLE IF NOT EXISTS split_price (date DATE, stock_id VARCHAR(50), type VARCHAR(20), before_price NUMERIC(20,4), after_price NUMERIC(20,4), max_price NUMERIC(20,4), min_price NUMERIC(20,4), open_price NUMERIC(20,4), PRIMARY KEY (date, stock_id, type));
CREATE TABLE IF NOT EXISTS trading_date (date DATE PRIMARY KEY);
CREATE TABLE IF NOT EXISTS market_value (date DATE, stock_id VARCHAR(50), market_value BIGINT, PRIMARY KEY (date, stock_id));
CREATE TABLE IF NOT EXISTS disposition_securities (date DATE, stock_id VARCHAR(50), stock_name VARCHAR(200), disposition_cnt INTEGER, condition VARCHAR(500), measure VARCHAR(500), period_start DATE, period_end DATE, PRIMARY KEY (date, stock_id));
"""

UPSERT_CAPRED = """INSERT INTO capital_reduction VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET post_reduction_ref = EXCLUDED.post_reduction_ref;"""
UPSERT_MV = """INSERT INTO market_value VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET market_value = EXCLUDED.market_value;"""
UPSERT_DISP = """INSERT INTO disposition_securities VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET disposition_cnt = EXCLUDED.disposition_cnt;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_capred(r): return (r["date"], r["stock_id"], safe_float(r.get("ClosingPriceonTheLastTradingDay")), safe_float(r.get("PostReductionReferencePrice")), safe_float(r.get("LimitUp")), safe_float(r.get("LimitDown")), safe_float(r.get("OpeningReferencePrice")), safe_float(r.get("ExrightReferencePrice")), str(r.get("ReasonforCapitalReduction", ""))[:500])
def map_mv(r): return (r["date"], r["stock_id"], safe_int(r.get("market_value")))
def map_disp(r): return (r["date"], r["stock_id"], str(r.get("stock_name", ""))[:200], safe_int(r.get("disposition_cnt")), str(r.get("condition", ""))[:500], str(r.get("measure", ""))[:500], r.get("period_start"), r.get("period_end"))

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_per_stock(
    conn, dataset: str, table: str, upsert_sql: str, tmpl: str, 
    mapper, stock_ids: list[str], start: str, end: str, 
    delay: float, force: bool, fetch_mode_override: str | None = None
):
    logger.info(f"=== [{table}] 開始 ===")
    ensure_ddl(conn, DDL_EVENT)
    conn.commit()
    
    flog = FailureLogger(table, db_conn=conn)
    latest = get_all_safe_starts(conn, table)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START[table], force)
        if not s:
            _write_fetch_log(conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset=dataset, 
                params={"data_id": sid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            dur = int((time.time() - t0) * 1000)
            
            if data:
                rows = [mapper(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, label_prefix=table, failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(
                    conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=n, 
                    duration_ms=dur, status="success" if n > 0 else "partial"
                )
            else:
                _write_fetch_log(
                    conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                    duration_ms=dur, status="no_new_data"
                )
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(
                conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                duration_ms=dur, status="failed", error_message=str(e)
            )
            
    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
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

def query_gap_targets(conn, days: int, target_tables: list[str], all_stock_ids: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    for tbl in target_tables:
        sql = f"SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tbl, str(days), all_stock_ids))
                have_success = {row[0] for row in cur.fetchall()}
            missing = [sid for sid in all_stock_ids if sid not in have_success]
            targets[tbl].extend(missing)
        except Exception as e:
            logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [gap-fill/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        
        if tbl == "capital_reduction":
            fetch_per_stock(conn, "TaiwanStockCapitalReductionReferencePrice", "capital_reduction", UPSERT_CAPRED, "(%s, %s, %s, %s, %s, %s, %s, %s, %s)", map_capred, sids, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)
        elif tbl == "market_value":
            fetch_per_stock(conn, "TaiwanStockMarketValue", "market_value", UPSERT_MV, "(%s, %s, %s)", map_mv, sids, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)
        elif tbl == "disposition_securities":
            fetch_per_stock(conn, "TaiwanStockDispositionSecuritiesPeriod", "disposition_securities", UPSERT_DISP, "(%s, %s, %s, %s, %s, %s, %s, %s)", map_disp, sids, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["capital_reduction", "market_value", "disposition_securities", "all"], default=["all"])
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["capital_reduction", "market_value", "disposition_securities"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        ensure_ddl(conn, DDL_EVENT)
        
        # 決定要抓取的標的
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            # 優先使用動態核心配置
            stock_configs = get_core_stocks_from_db(conn)
            if stock_configs:
                stock_ids = [sid for sid, cfg in stock_configs.items() if cfg.get("is_active", True)]
            else:
                stock_ids = get_db_stock_ids(conn)
        
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
            targets = query_gap_targets(conn, args.gap_fill, tables, stock_ids)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        if "capital_reduction" in tables: 
            fetch_per_stock(conn, "TaiwanStockCapitalReductionReferencePrice", "capital_reduction", UPSERT_CAPRED, "(%s, %s, %s, %s, %s, %s, %s, %s, %s)", map_capred, stock_ids, args.start, args.end, args.delay, args.force)
        if "market_value" in tables: 
            fetch_per_stock(conn, "TaiwanStockMarketValue", "market_value", UPSERT_MV, "(%s, %s, %s)", map_mv, stock_ids, args.start, args.end, args.delay, args.force)
        if "disposition_securities" in tables: 
            fetch_per_stock(conn, "TaiwanStockDispositionSecuritiesPeriod", "disposition_securities", UPSERT_DISP, "(%s, %s, %s, %s, %s, %s, %s, %s)", map_disp, stock_ids, args.start, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        # 於程式完全結束時，印出統一的 FinMind RequestStats 報表
        get_request_stats().summary()

if __name__ == "__main__":
    main()