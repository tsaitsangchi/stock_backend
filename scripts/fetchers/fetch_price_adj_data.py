"""
fetch_price_adj_data.py — 還原股價 + 當沖交易 + 漲跌停價（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 結合 `FailureLogger` 與 `commit_per_stock_per_day` 進行雙層粒度原子寫入。
  ★ 預設動態撈取資料庫定義的 `fetch_basic=True` 核心清單。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

執行範例（常規）：
    # 抓取核心標的的所有資料（優先嘗試批次模式，若不支援自動降級為逐支）
    python scripts/fetchers/fetch_price_adj_data.py
    
    # 強制使用逐支抓取模式
    python scripts/fetchers/fetch_price_adj_data.py --per-stock
    
    # 僅抓取台積電與聯發科 (強制更新)
    python scripts/fetchers/fetch_price_adj_data.py --stock-id 2330,2454 --force

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的任務
    python scripts/fetchers/fetch_price_adj_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的個股
    python scripts/fetchers/fetch_price_adj_data.py --gap-fill 30
    
    # 針對特定資料表進行補漏
    python scripts/fetchers/fetch_price_adj_data.py --gap-fill 14 --tables price_adj
"""

from __future__ import annotations

import argparse
import time
import sys
import logging
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

DATASET_START = {
    "price_adj":   "1994-10-01",
    "day_trading": "2014-01-01",
    "price_limit": "2000-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

DEFAULT_CHUNK_DAYS      = 90
DEFAULT_BATCH_THRESHOLD = 20
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

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None, fetch_mode="per_stock"):
    """v3.2 標準化日誌寫入"""
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

DDL_PRICE_ADJ = """
CREATE TABLE IF NOT EXISTS price_adj (
    date             DATE,
    stock_id         VARCHAR(50),
    trading_volume   BIGINT,
    trading_money    BIGINT,
    open             NUMERIC(20,4),
    max              NUMERIC(20,4),
    min              NUMERIC(20,4),
    close            NUMERIC(20,4),
    spread           NUMERIC(20,4),
    trading_turnover NUMERIC(20,4),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_price_adj_stock ON price_adj (stock_id, date);
"""

DDL_DAY_TRADING = """
CREATE TABLE IF NOT EXISTS day_trading (
    date            DATE,
    stock_id        VARCHAR(50),
    buy_after_sale  VARCHAR(20),
    volume          BIGINT,
    buy_amount      BIGINT,
    sell_amount     BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_day_trading_stock ON day_trading (stock_id, date);
"""

DDL_PRICE_LIMIT = """
CREATE TABLE IF NOT EXISTS price_limit (
    date            DATE,
    stock_id        VARCHAR(50),
    reference_price NUMERIC(20,4),
    limit_up        NUMERIC(20,4),
    limit_down      NUMERIC(20,4),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_price_limit_stock ON price_limit (stock_id, date);
"""

UPSERT_PRICE_ADJ = """
INSERT INTO price_adj
    (date, stock_id, trading_volume, trading_money, open, max, min, close, spread, trading_turnover)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET
    trading_volume = EXCLUDED.trading_volume, close = EXCLUDED.close;
"""

UPSERT_DAY_TRADING = """
INSERT INTO day_trading (date, stock_id, buy_after_sale, volume, buy_amount, sell_amount)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET
    volume = EXCLUDED.volume, buy_amount = EXCLUDED.buy_amount;
"""

UPSERT_PRICE_LIMIT = """
INSERT INTO price_limit (date, stock_id, reference_price, limit_up, limit_down)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET
    reference_price = EXCLUDED.reference_price, limit_up = EXCLUDED.limit_up;
"""


# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_price_adj(r: dict) -> tuple:
    return (r["date"], str(r["stock_id"]), safe_int(r.get("Trading_Volume")), safe_int(r.get("Trading_money")), safe_float(r.get("open")), safe_float(r.get("max")), safe_float(r.get("min")), safe_float(r.get("close")), safe_float(r.get("spread")), safe_float(r.get("Trading_turnover")))

def map_day_trading(r: dict) -> tuple:
    return (r["date"], str(r["stock_id"]), str(r.get("BuyAfterSale", "")), safe_int(r.get("Volume")), safe_int(r.get("BuyAmount")), safe_int(r.get("SellAmount")))

def map_price_limit(r: dict) -> tuple:
    return (r["date"], str(r["stock_id"]), safe_float(r.get("reference_price")), safe_float(r.get("limit_up")), safe_float(r.get("limit_down")))


# ─────────────────────────────────────────────
# Core Loop
# ─────────────────────────────────────────────
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
    
    # 起始日分組
    groups = defaultdict(list)
    fetch_mode = fetch_mode_override or ("batch" if use_batch else "per_stock")
    
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest_dates, start, DATASET_START[dataset_key], force)
        if s: 
            groups[s].append(sid)
        else:
            _write_fetch_log(conn, table, sid, "skipped", error_message="up_to_date", fetch_mode=fetch_mode)

    if not groups:
        logger.info(f"[{table}] 資料已是最新。")
        return

    total_api = total_rows = 0
    batch_disabled = False

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        if use_batch and len(sids) >= batch_threshold and not batch_disabled:
            # ── 批次模式 ──
            seg_start = group_start
            seg_end_limit = datetime.strptime(end, "%Y-%m-%d")
            
            try:
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_limit: break
                    seg_end = min((seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"), end)
                    
                    logger.info(f"  [{table}] 批次 {seg_start}~{seg_end}（{len(sids)} 支）")
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
                    
                    chunk_data = []
                    actual_sids_in_chunk = set()
                    if data:
                        for r in data:
                            if str(r.get("stock_id")) in sids_set:
                                try: 
                                    chunk_data.append(mapper(r))
                                    actual_sids_in_chunk.add(str(r.get("stock_id")))
                                except Exception: pass
                    
                    if chunk_data:
                        chunk_data = dedup_rows(chunk_data, (0, 1))
                        res = commit_per_stock_per_day(conn, upsert_sql, chunk_data, template, label_prefix=table, failure_logger=flog)
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
                logger.warning(f"  [{table}] 該資料集不支援批次查詢，Fallback 為逐支模式。")
                batch_disabled = True
            except Exception as e:
                logger.error(f"  [{table}] 批次抓取失敗：{e}")
                for sid in sids:
                    _write_fetch_log(conn, table, sid, "failed", fetch_date_from=seg_start, fetch_date_to=end, error_message=str(e), fetch_mode=fetch_mode)

        if (not use_batch) or len(sids) < batch_threshold or batch_disabled:
            # ── 逐支模式 ──
            fetch_mode_single = fetch_mode_override or "per_stock"
            for sid in sids:
                start_time = time.time()
                try:
                    # 修正：全面改為具名參數 (Keyword arguments)
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
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table, failure_logger=flog)
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
    p = argparse.ArgumentParser(description="還原股價與交易補強資料 (v3.2 — 核心模組升級版)")
    p.add_argument("--tables", nargs="+", choices=["price_adj", "day_trading", "price_limit", "all"], default=["all"])
    p.add_argument("--stock-id", type=str, default=None)
    p.add_argument("--start", type=str, default="1994-10-01")
    p.add_argument("--end",   type=str, default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--per-stock", action="store_true", help="強制使用逐支抓取模式")
    p.add_argument("--batch-threshold", type=int, default=DEFAULT_BATCH_THRESHOLD)
    p.add_argument("--chunk-days",      type=int, default=DEFAULT_CHUNK_DAYS)
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["price_adj", "day_trading", "price_limit"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    configs = {
        "price_adj":   ("TaiwanStockPriceAdj", DDL_PRICE_ADJ, UPSERT_PRICE_ADJ, "(%s::date,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric)", map_price_adj),
        "day_trading": ("TaiwanStockDayTrading", DDL_DAY_TRADING, UPSERT_DAY_TRADING, "(%s::date,%s,%s,%s,%s,%s)", map_day_trading),
        "price_limit": ("TaiwanStockPriceLimit", DDL_PRICE_LIMIT, UPSERT_PRICE_LIMIT, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric)", map_price_limit),
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