"""
fetch_technical_data.py — 技術面資料（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 整合並簡化「批次全市場」與「逐支抓取」邏輯為單一通用函式。
  ★ 自動讀取 `stocks` 表中 `fetch_basic=True` 的核心標的。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

執行範例（常規）：
    # 抓取核心標的所有技術面資料（優先嘗試批次模式，若不支援自動降級為逐支）
    python scripts/fetchers/fetch_technical_data.py
    
    # 僅抓取 PER/PBR
    python scripts/fetchers/fetch_technical_data.py --tables stock_per
    
    # 強制使用逐支抓取模式
    python scripts/fetchers/fetch_technical_data.py --per-stock
    
    # 僅抓取特定標的 (強制更新)
    python scripts/fetchers/fetch_technical_data.py --stock-id 2330,2454 --force

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的任務
    python scripts/fetchers/fetch_technical_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的個股
    python scripts/fetchers/fetch_technical_data.py --gap-fill 30
"""

from __future__ import annotations

import sys
import logging
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.finmind_client import finmind_get, get_request_stats, BatchNotSupportedError
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    safe_int,
    get_all_safe_starts,
    get_db_stock_ids,
    get_core_stocks_from_db,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATASET_START_DATES = {
    "stock_price": "1994-10-01",
    "stock_per":   "2005-10-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1994-10-01"
DEFAULT_CHUNK_DAYS      = 90
DEFAULT_BATCH_THRESHOLD = 20
_CLI_ARGS_STR = " ".join(sys.argv)

# ──────────────────────────────────────────────
# 日誌與 SQL
# ──────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None, fetch_mode="per_stock"):
    """v3.2 標準化日誌寫入，失敗不影響主流程"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (
                    run_ts, table_name, stock_id, fetch_mode, status, rows_inserted, 
                    fetch_date_from, fetch_date_to, duration_ms, error_message, cli_args
                ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (table_name, stock_id, fetch_mode, status, rows_inserted, fetch_date_from, fetch_date_to, duration_ms, error_message, _CLI_ARGS_STR))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"無法寫入 fetch_log: {e}")

DDL_STOCK_PRICE = """
CREATE TABLE IF NOT EXISTS stock_price (
    date             DATE,
    stock_id         VARCHAR(50),
    trading_volume   BIGINT,
    trading_money    BIGINT,
    open             NUMERIC(10,4),
    max              NUMERIC(10,4),
    min              NUMERIC(10,4),
    close            NUMERIC(10,4),
    spread           NUMERIC(10,4),
    trading_turnover INTEGER,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_stock_price_stock_id ON stock_price (stock_id);
"""

DDL_STOCK_PER = """
CREATE TABLE IF NOT EXISTS stock_per (
    date           DATE,
    stock_id       VARCHAR(50),
    dividend_yield NUMERIC(10,4),
    per            NUMERIC(10,4),
    pbr            NUMERIC(10,4),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_stock_per_stock_id ON stock_per (stock_id);
"""

UPSERT_STOCK_PRICE = """
INSERT INTO stock_price
    (date, stock_id, trading_volume, trading_money, open, max, min, close, spread, trading_turnover)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    trading_volume   = EXCLUDED.trading_volume,
    trading_money    = EXCLUDED.trading_money,
    open             = EXCLUDED.open,
    max              = EXCLUDED.max,
    min              = EXCLUDED.min,
    close            = EXCLUDED.close,
    spread           = EXCLUDED.spread,
    trading_turnover = EXCLUDED.trading_turnover;
"""

UPSERT_STOCK_PER = """
INSERT INTO stock_per (date, stock_id, dividend_yield, per, pbr)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    dividend_yield = EXCLUDED.dividend_yield,
    per            = EXCLUDED.per,
    pbr            = EXCLUDED.pbr;
"""

# ──────────────────────────────────────────────
# Row Mappers
# ──────────────────────────────────────────────
def map_price_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_int(r.get("Trading_Volume")),
        safe_int(r.get("Trading_money")),
        safe_float(r.get("open")),
        safe_float(r.get("max")),
        safe_float(r.get("min")),
        safe_float(r.get("close")),
        safe_float(r.get("spread")),
        safe_int(r.get("Trading_turnover")),
    )

def map_per_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_float(r.get("dividend_yield")),
        safe_float(r.get("PER")),
        safe_float(r.get("PBR")),
    )

# ──────────────────────────────────────────────
# 資料抓取與寫入核心
# ──────────────────────────────────────────────
def fetch_dataset_unified(
    conn, dataset: str, table: str, ddl: str, upsert_sql: str, template: str, 
    mapper, dataset_key: str, stock_ids: list[str], start: str, end: str, 
    delay: float, force: bool, use_batch: bool, batch_threshold: int, 
    chunk_days: int, fetch_mode_override: str | None = None
):
    logger.info(f"=== [{table}] 開始 ===")
    ensure_ddl(conn, ddl)
    conn.commit()
    
    latest_dates = get_all_safe_starts(conn, table)
    flog = FailureLogger(table, db_conn=conn)
    fetch_mode = fetch_mode_override or ("batch" if use_batch else "per_stock")
    
    # 決定每個股票的起始日期並分組
    groups = defaultdict(list)
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest_dates, start, DATASET_START_DATES[dataset_key], force)
        if s: 
            groups[s].append(sid)
        else:
            _write_fetch_log(conn, table, sid, "skipped", error_message="up_to_date", fetch_mode=fetch_mode)

    if not groups:
        logger.info(f"  [{table}] 資料皆已是最新，跳過。")
        return

    total_api = total_rows = 0
    batch_disabled = False

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        # ── 批次模式 ──
        used_batch = False
        if use_batch and len(sids) >= batch_threshold and not batch_disabled:
            used_batch = True
            seg_start = group_start
            seg_end_limit = datetime.strptime(end, "%Y-%m-%d")
            
            try:
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_limit: break
                    seg_end = min((seg_start_dt + timedelta(days=chunk_days-1)).strftime("%Y-%m-%d"), end)

                    logger.info(f"  [{table}] 批次請求 {seg_start}~{seg_end}（{len(sids)} 支）")
                    start_time = time.time()
                    
                    # 修正：全面改為具名參數 (Keyword arguments)
                    data = finmind_get(
                        dataset=dataset, 
                        params={"start_date": seg_start, "end_date": seg_end}, 
                        delay=delay,
                        raise_on_batch_400=True,
                        raise_on_error=True
                    )
                    duration_ms = int((time.time() - start_time) * 1000)
                    total_api += 1

                    chunk_rows = []
                    actual_sids_in_chunk = set()
                    if data:
                        for r in data:
                            sid = str(r.get("stock_id", ""))
                            if sid in sids_set:
                                try: 
                                    chunk_rows.append(mapper(r))
                                    actual_sids_in_chunk.add(sid)
                                except Exception: pass

                    if chunk_rows:
                        chunk_rows = dedup_rows(chunk_rows, (0, 1))
                        res = commit_per_stock_per_day(conn, upsert_sql, chunk_rows, template, label_prefix=table, failure_logger=flog)
                        n = sum(res.values())
                        total_rows += n
                        
                        # 批次成功，為參與標的記錄日誌
                        for sid in actual_sids_in_chunk:
                            _write_fetch_log(conn, table, sid, "success", rows_inserted=0, fetch_date_from=seg_start, fetch_date_to=seg_end, duration_ms=duration_ms, fetch_mode=fetch_mode)
                    else:
                        # 整批無資料
                        for sid in sids:
                            _write_fetch_log(conn, table, sid, "no_new_data", fetch_date_from=seg_start, fetch_date_to=seg_end, duration_ms=duration_ms, fetch_mode=fetch_mode)

                    seg_start = (datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            except BatchNotSupportedError:
                logger.warning(f"  [{table}] Fallback 為逐支模式。")
                batch_disabled = True
                used_batch = False
            except Exception as e:
                logger.error(f"  [{table}] 批次抓取發生錯誤: {e}")
                for sid in sids:
                    _write_fetch_log(conn, table, sid, "failed", fetch_date_from=group_start, fetch_date_to=end, error_message=str(e), fetch_mode=fetch_mode)

        # ── 逐支模式 ──
        if (not use_batch) or len(sids) < batch_threshold or batch_disabled:
            if used_batch:
                continue
            
            fetch_mode_single = fetch_mode_override or "per_stock"
            for sid in sids:
                start_time = time.time()
                try:
                    data = finmind_get(
                        dataset=dataset, 
                        params={"data_id": sid, "start_date": group_start, "end_date": end}, 
                        delay=delay,
                        raise_on_error=True
                    )
                    duration_ms = int((time.time() - start_time) * 1000)
                    total_api += 1
                    
                    if not data:
                        _write_fetch_log(conn, table, sid, "no_new_data", fetch_date_from=group_start, fetch_date_to=end, duration_ms=duration_ms, fetch_mode=fetch_mode_single)
                        continue
                        
                    rows = [mapper(r) for r in data]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=f"{table}/{sid}", failure_logger=flog)
                    n = sum(res.values())
                    total_rows += n
                    _write_fetch_log(conn, table, sid, "success", rows_inserted=n, fetch_date_from=group_start, fetch_date_to=end, duration_ms=duration_ms, fetch_mode=fetch_mode_single)
                    
                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    flog.record(stock_id=sid, error=str(e), start_date=group_start, end_date=end)
                    _write_fetch_log(conn, table, sid, "failed", fetch_date_from=group_start, fetch_date_to=end, duration_ms=duration_ms, error_message=str(e), fetch_mode=fetch_mode_single)

    flog.summary()
    logger.info(f"[{table}] 完成  API：{total_api}  寫入：{total_rows}  失敗：{len(flog)}")


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

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str, configs: dict):
    use_batch = not args.per_stock
    for tbl, sids in targets.items():
        if not sids or tbl not in configs: continue
        dataset, ddl, upsert, tmpl, mapper = configs[tbl]
        fetch_dataset_unified(
            conn, dataset, tbl, ddl, upsert, tmpl, mapper, tbl, sids, 
            args.start, args.end, args.delay, force=True, 
            use_batch=use_batch, batch_threshold=args.batch_threshold, 
            chunk_days=args.chunk_days, fetch_mode_override=fetch_mode
        )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="技術面資料抓取 (v3.2 — 核心模組升級版)")
    parser.add_argument("--tables", nargs="+", choices=["stock_price", "stock_per", "all"], default=["all"])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--per-stock", action="store_true")
    parser.add_argument("--batch-threshold", type=int, default=DEFAULT_BATCH_THRESHOLD)
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--stock-id", default=None)
    parser.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    parser.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = parser.parse_args()

    tables = ["stock_price", "stock_per"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    configs = {
        "stock_price": ("TaiwanStockPrice", DDL_STOCK_PRICE, UPSERT_STOCK_PRICE, "(%s::date,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)", map_price_row),
        "stock_per":   ("TaiwanStockPER", DDL_STOCK_PER, UPSERT_STOCK_PER, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric)", map_per_row),
    }
    
    try:
        _ensure_fetch_log_table(conn)
        
        # 決定要抓取的標的
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            # 優先使用動態核心配置，過濾 fetch_basic = TRUE
            stock_configs = get_core_stocks_from_db(conn)
            if stock_configs:
                stock_ids = [sid for sid, cfg in stock_configs.items() if cfg.get("is_active", True) and cfg.get("fetch_basic", True)]
            else:
                stock_ids = get_db_stock_ids(conn, types=("twse", "otc"))
                
        if not stock_ids:
            logger.warning("未找到需要抓取的標的（請確認 stocks 表中 fetch_basic 是否有設為 TRUE）。")
            return

        logger.info(f"目標股票：{len(stock_ids)} 支 | 模式：{'逐支' if args.per_stock else '優先批次'}")

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, tables)
            if targets: _run_targeted(conn, targets, args, fetch_mode="retry", configs=configs)
            else: logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, tables, stock_ids)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill", configs=configs)
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        for key in tables:
            dataset, ddl, upsert, tmpl, mapper = configs[key]
            fetch_dataset_unified(
                conn, dataset, key, ddl, upsert, tmpl, mapper, key, 
                stock_ids, args.start, args.end, args.delay, args.force, 
                not args.per_stock, args.batch_threshold, args.chunk_days
            )
            
    finally:
        conn.close()
        logger.info("全部完成")
        get_request_stats().summary()

if __name__ == "__main__":
    main()