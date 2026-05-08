import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta

# ── 啟動引導 (Bootstrap)：確保能找到 scripts/core ──
_scripts_dir = Path(__file__).resolve().parent.parent  # 指向 scripts/
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from core.path_setup import ensure_scripts_on_path, get_outputs_dir
ensure_scripts_on_path(__file__)

from config import DB_CONFIG  # noqa: F401
from core.finmind_client import finmind_get, BatchNotSupportedError
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    safe_float,
    safe_int,
    get_core_stocks_from_db,
    get_db_stock_ids,
    get_all_safe_starts,
    get_market_safe_start,
    resolve_start_cached,
    log_fetch_result,
    safe_commit_rows,
    FailureLogger
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = get_outputs_dir()

DATASET_START = {
    "total_margin_short":      "2001-01-01",
    "total_inst_investors":    "2004-04-01",
    "securities_lending":      "2001-05-01",
    "daily_short_balance":     "2005-07-01",
    "margin_short_suspension": "2015-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# 哪些表是「市場層級」（無 stock_id）
MARKET_LEVEL_TABLES = {"total_margin_short", "total_inst_investors"}
PER_STOCK_TABLES = set(DATASET_START.keys()) - MARKET_LEVEL_TABLES

# CLI 命令字串（給 fetch_log 紀錄）
_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 業務 DDL（沿用原版）
# ─────────────────────────────────────────────
DDL_TOTAL_MARGIN_SHORT = """
CREATE TABLE IF NOT EXISTS total_margin_short (
    date          DATE,
    name          VARCHAR(50),
    today_balance BIGINT,
    yes_balance   BIGINT,
    buy           BIGINT,
    sell          BIGINT,
    return_qty    BIGINT,
    PRIMARY KEY (date, name)
);
CREATE INDEX IF NOT EXISTS idx_tms_date ON total_margin_short (date);
"""

DDL_TOTAL_INST = """
CREATE TABLE IF NOT EXISTS total_inst_investors (
    date DATE,
    name VARCHAR(100),
    buy  BIGINT,
    sell BIGINT,
    PRIMARY KEY (date, name)
);
CREATE INDEX IF NOT EXISTS idx_tii_date ON total_inst_investors (date);
"""

DDL_SBL = """
CREATE TABLE IF NOT EXISTS securities_lending (
    date                  DATE,
    stock_id              VARCHAR(50),
    transaction_type      VARCHAR(50),
    volume                BIGINT,
    fee_rate              NUMERIC(20,4),
    close                 NUMERIC(20,4),
    original_return_date  DATE,
    original_lending_period INTEGER,
    PRIMARY KEY (date, stock_id, transaction_type)
);
CREATE INDEX IF NOT EXISTS idx_sbl_stock ON securities_lending (stock_id, date);
"""

DDL_DAILY_SHORT = """
CREATE TABLE IF NOT EXISTS daily_short_balance (
    date                                       DATE,
    stock_id                                   VARCHAR(50),
    margin_short_prev_balance                  BIGINT,
    margin_short_short_sales                   BIGINT,
    margin_short_short_covering                BIGINT,
    margin_short_stock_redemption              BIGINT,
    margin_short_current_balance               BIGINT,
    margin_short_quota                         BIGINT,
    sbl_short_prev_balance                     BIGINT,
    sbl_short_short_sales                      BIGINT,
    sbl_short_returns                          BIGINT,
    sbl_short_adjustments                      BIGINT,
    sbl_short_current_balance                  BIGINT,
    sbl_short_quota                            BIGINT,
    sbl_short_short_covering                   BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_dsb_stock ON daily_short_balance (stock_id, date);
"""

DDL_MARGIN_SHORT_SUSPENSION = """
CREATE TABLE IF NOT EXISTS margin_short_suspension (
    date     DATE,
    stock_id VARCHAR(50),
    end_date DATE,
    reason   VARCHAR(500),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_mss_stock ON margin_short_suspension (stock_id, date);
"""

# ─────────────────────────────────────────────
# Upsert SQL（沿用原版）
# ─────────────────────────────────────────────
UPSERT_TOTAL_MARGIN_SHORT = """
INSERT INTO total_margin_short (date, name, today_balance, yes_balance, buy, sell, return_qty)
VALUES %s
ON CONFLICT (date, name) DO UPDATE SET
    today_balance = EXCLUDED.today_balance,
    yes_balance   = EXCLUDED.yes_balance,
    buy           = EXCLUDED.buy,
    sell          = EXCLUDED.sell,
    return_qty    = EXCLUDED.return_qty;
"""

UPSERT_TOTAL_INST = """
INSERT INTO total_inst_investors (date, name, buy, sell)
VALUES %s
ON CONFLICT (date, name) DO UPDATE SET
    buy  = EXCLUDED.buy,
    sell = EXCLUDED.sell;
"""

UPSERT_SBL = """
INSERT INTO securities_lending (
    date, stock_id, transaction_type, volume, fee_rate, close,
    original_return_date, original_lending_period
) VALUES %s
ON CONFLICT (date, stock_id, transaction_type) DO UPDATE SET
    volume                  = EXCLUDED.volume,
    fee_rate                = EXCLUDED.fee_rate,
    close                   = EXCLUDED.close,
    original_return_date    = EXCLUDED.original_return_date,
    original_lending_period = EXCLUDED.original_lending_period;
"""

UPSERT_DAILY_SHORT = """
INSERT INTO daily_short_balance (
    date, stock_id,
    margin_short_prev_balance, margin_short_short_sales,
    margin_short_short_covering, margin_short_stock_redemption,
    margin_short_current_balance, margin_short_quota,
    sbl_short_prev_balance, sbl_short_short_sales,
    sbl_short_returns, sbl_short_adjustments,
    sbl_short_current_balance, sbl_short_quota, sbl_short_short_covering
) VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    margin_short_prev_balance     = EXCLUDED.margin_short_prev_balance,
    margin_short_short_sales      = EXCLUDED.margin_short_short_sales,
    margin_short_short_covering   = EXCLUDED.margin_short_short_covering,
    margin_short_stock_redemption = EXCLUDED.margin_short_stock_redemption,
    margin_short_current_balance  = EXCLUDED.margin_short_current_balance,
    margin_short_quota            = EXCLUDED.margin_short_quota,
    sbl_short_prev_balance        = EXCLUDED.sbl_short_prev_balance,
    sbl_short_short_sales         = EXCLUDED.sbl_short_short_sales,
    sbl_short_returns             = EXCLUDED.sbl_short_returns,
    sbl_short_adjustments         = EXCLUDED.sbl_short_adjustments,
    sbl_short_current_balance     = EXCLUDED.sbl_short_current_balance,
    sbl_short_quota               = EXCLUDED.sbl_short_quota,
    sbl_short_short_covering      = EXCLUDED.sbl_short_short_covering;
"""

UPSERT_MARGIN_SHORT_SUSPENSION = """
INSERT INTO margin_short_suspension (date, stock_id, end_date, reason)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    end_date = EXCLUDED.end_date,
    reason   = EXCLUDED.reason;
"""


# ─────────────────────────────────────────────
# 抓取邏輯（整合 db_utils 標準日誌與 FailureLogger）
# ─────────────────────────────────────────────
def fetch_market_dataset(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    start: str, end: str, delay: float, force: bool,
):
    ensure_ddl(conn, ddl)
    s = start
    if not force:
        safe_s = get_market_safe_start(conn, table)
        if safe_s:
            if safe_s > end:
                logger.info(f"[{table}] 已最新，跳過")
                log_fetch_result(conn, table, "MARKET", None, None, 0, "skipped")
                return
            s = max(safe_s, DATASET_START[dataset_key])
    
    s = max(s, DATASET_START[dataset_key])
    logger.info(f"[{table}] 抓取 {s} ~ {end}")

    t0 = time.time()
    try:
        data = finmind_get(dataset, {"start_date": s, "end_date": end}, delay, raise_on_error=True)
    except Exception as e:
        logger.error(f"[{table}] API 失敗：{e}")
        log_fetch_result(conn, table, "MARKET", s, end, 0, "failed", str(e))
        return

    if not data:
        logger.info(f"[{table}] 無新資料")
        log_fetch_result(conn, table, "MARKET", s, end, 0, "no_new_data")
        return

    # 去重
    rows_map = {}
    for r in data:
        try:
            mapped = mapper(r)
            pk = mapped[:2] # (date, name)
            rows_map[pk] = mapped
        except Exception as e:
            logger.warning(f"[{table}] mapper 異常筆，跳過：{e}")

    rows = list(rows_map.values())
    n = safe_commit_rows(conn, upsert_sql, rows, template, label=table)
    status = "success" if n > 0 else "failed"
    log_fetch_result(conn, table, "MARKET", s, end, n, status)
    logger.info(f"[{table}] 寫入 {n} 筆")


def fetch_per_stock_dataset(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    stock_ids: list[str], start: str, end: str,
    delay: float, force: bool, use_batch: bool,
    batch_threshold: int = 20, chunk_days: int = 90,
    fetch_mode_override: str | None = None,
):
    """個股層資料：支援批次或逐支。每支股票寫入後立即 commit。"""
    ensure_ddl(conn, ddl)
    latest_dates = get_all_safe_starts(conn, table)
    failure_logger = FailureLogger(table)

    stock_starts: dict[str, str] = {}
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest_dates, start, DATASET_START[dataset_key], force)
        if s is None:
            log_fetch_result(conn, table, sid, None, None, 0, "skipped")
        else:
            stock_starts[sid] = s

    if not stock_starts:
        logger.info(f"[{table}] 全部標的皆已最新")
        return

    # 按起始日分組，優化批次請求
    groups: dict[str, list] = defaultdict(list)
    for sid, s in stock_starts.items():
        groups[s].append(sid)

    total_rows = 0
    batch_disabled = False
    pk_len = 3 if table == "securities_lending" else 2

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        # ── 批次模式 ──
        if use_batch and len(sids) >= batch_threshold and not batch_disabled:
            logger.info(f"[{table}] 啟動批次模式：{group_start} ~ {end} ({len(sids)} 支)")
            try:
                # 簡單分段抓取防止超長
                seg_start = group_start
                seg_end_dt = datetime.strptime(end, "%Y-%m-%d")
                all_data = []
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_dt: break
                    seg_end = min((seg_start_dt + timedelta(days=chunk_days-1)).strftime("%Y-%m-%d"), end)
                    
                    data = finmind_get(
                        dataset, {"start_date": seg_start, "end_date": seg_end},
                        delay, raise_on_error=True, raise_on_batch_400=True
                    )
                    all_data.extend([r for r in data if r.get("stock_id") in sids_set])
                    
                    seg_start = (datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                
                # 分組寫入
                rows_by_stock = defaultdict(list)
                for r in all_data:
                    try: rows_by_stock[r["stock_id"]].append(mapper(r))
                    except: pass
                
                for sid in sids:
                    s_rows = rows_by_stock.get(sid, [])
                    if not s_rows:
                        log_fetch_result(conn, table, sid, group_start, end, 0, "no_new_data")
                        continue
                    
                    # 去重
                    rows_map = {r[:pk_len]: r for r in s_rows}
                    n = safe_commit_rows(conn, upsert_sql, list(rows_map.values()), template, label=f"{table}/{sid}")
                    total_rows += n
                    log_fetch_result(conn, table, sid, group_start, end, n, "success" if n > 0 else "failed")
                continue

            except BatchNotSupportedError:
                logger.warning(f"[{table}] 不支援批次，切換至逐支模式")
                batch_disabled = True
            except Exception as e:
                logger.error(f"[{table}] 批次抓取嚴重錯誤：{e}")
                for sid in sids: log_fetch_result(conn, table, sid, group_start, end, 0, "failed", str(e))
                continue

        # ── 逐支模式 ──
        for sid in sids:
            try:
                data = finmind_get(dataset, {"data_id": sid, "start_date": group_start, "end_date": end}, delay, raise_on_error=True)
                if not data:
                    log_fetch_result(conn, table, sid, group_start, end, 0, "no_new_data")
                    continue
                
                rows_map = {mapper(r)[:pk_len]: mapper(r) for r in data if r}
                n = safe_commit_rows(conn, upsert_sql, list(rows_map.values()), template, label=f"{table}/{sid}")
                total_rows += n
                log_fetch_result(conn, table, sid, group_start, end, n, "success" if n > 0 else "failed")
            except Exception as e:
                logger.error(f"[{table}/{sid}] 失敗：{e}")
                failure_logger.record(sid, 0, str(e))
                log_fetch_result(conn, table, sid, group_start, end, 0, "failed", str(e))

    failure_logger.summary()
    logger.info(f"[{table}] 總計寫入 {total_rows} 筆")


# ─────────────────────────────────────────────
# Mapper（沿用原版）
# ─────────────────────────────────────────────
def map_total_margin(r: dict) -> tuple:
    return (
        r["date"], r.get("name"),
        safe_int(r.get("TodayBalance")),
        safe_int(r.get("YesBalance")),
        safe_int(r.get("buy")),
        safe_int(r.get("sell")),
        safe_int(r.get("Return")),
    )

def map_total_inst(r: dict) -> tuple:
    return (
        r["date"], r.get("name"),
        safe_int(r.get("buy")),
        safe_int(r.get("sell")),
    )

def map_sbl(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        r.get("transaction_type"),
        safe_int(r.get("volume")),
        safe_float(r.get("fee_rate")),
        safe_float(r.get("close")),
        r.get("original_return_date") or None,
        safe_int(r.get("original_lending_period")),
    )

def map_daily_short(r: dict) -> tuple:
    f = lambda k: safe_int(r.get(k))
    return (
        r["date"], r["stock_id"],
        f("MarginShortSalesPreviousDayBalance"), f("MarginShortSalesShortSales"),
        f("MarginShortSalesShortCovering"), f("MarginShortSalesStockRedemption"),
        f("MarginShortSalesCurrentDayBalance"), f("MarginShortSalesQuota"),
        f("SBLShortSalesPreviousDayBalance"), f("SBLShortSalesShortSales"),
        f("SBLShortSalesReturns"), f("SBLShortSalesAdjustments"),
        f("SBLShortSalesCurrentDayBalance"), f("SBLShortSalesQuota"),
        f("SBLShortSalesShortCovering"),
    )

def map_margin_susp(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        r.get("end_date") or None,
        r.get("reason"),
    )

# ═════════════════════════════════════════════
# 抓取邏輯（含 fetch_log 整合）
# ═════════════════════════════════════════════
def fetch_market_dataset(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    start: str, end: str, delay: float, force: bool,
):
    ensure_ddl(conn, ddl)
    conn.commit()
    s = start
    if not force:
        safe_s = get_market_safe_start(conn, table)
        if safe_s:
            if safe_s > end:
                logger.info(f"[{table}] 已最新，跳過")
                _write_fetch_log(
                    conn, table_name=table, stock_id=None, fetch_mode="market",
                    fetch_date_from=None, fetch_date_to=None,
                    rows_inserted=0, duration_ms=0,
                    status="skipped", error_message="up_to_date",
                )
                return
            s = max(safe_s, DATASET_START[dataset_key])
    s = max(s, DATASET_START[dataset_key])
    logger.info(f"[{table}] 抓取 {s} ~ {end}")

    t0 = time.time()
    try:
        data = finmind_get(dataset, {"start_date": s, "end_date": end}, delay, raise_on_error=True)
    except Exception as e:
        duration_ms = int((time.time() - t0) * 1000)
        logger.error(f"[{table}] API 失敗：{e}")
        _write_fetch_log(
            conn, table_name=table, stock_id=None, fetch_mode="market",
            fetch_date_from=s, fetch_date_to=end,
            rows_inserted=0, duration_ms=duration_ms,
            status="failed", error_message=str(e),
        )
        return

    if not data:
        duration_ms = int((time.time() - t0) * 1000)
        logger.info(f"[{table}] 無新資料")
        _write_fetch_log(
            conn, table_name=table, stock_id=None, fetch_mode="market",
            fetch_date_from=s, fetch_date_to=end,
            rows_inserted=0, duration_ms=duration_ms,
            status="no_new_data",
        )
        return

    rows_map = {}
    for r in data:
        try:
            mapped = mapper(r)
            pk = mapped[:2]
            rows_map[pk] = mapped
        except Exception as e:
            logger.warning(f"[{table}] mapper 異常筆，跳過：{e}")

    rows = list(rows_map.values())
    n = safe_commit_rows(conn, upsert_sql, rows, template, label=table)
    duration_ms = int((time.time() - t0) * 1000)
    status = "success" if n > 0 else ("partial" if rows else "no_new_data")
    logger.info(f"[{table}] 寫入 {n} 筆（{duration_ms} ms）")
    _write_fetch_log(
        conn, table_name=table, stock_id=None, fetch_mode="market",
        fetch_date_from=s, fetch_date_to=end,
        rows_inserted=n, duration_ms=duration_ms,
        status=status,
    )


def fetch_per_stock_dataset(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    stock_ids: list[str], start: str, end: str,
    delay: float, force: bool, use_batch: bool,
    batch_threshold: int = 20, chunk_days: int = 90,
    fetch_mode_override: str | None = None,
):
    """個股層資料：Sponsor 可不帶 data_id 批次抓全市場，否則逐支。
    無論批次或逐支，**每支股票寫入後立即 commit**，並於 fetch_log 記錄。"""
    ensure_ddl(conn, ddl)
    conn.commit()
    valid_set = set(stock_ids)
    latest_dates = get_all_safe_starts(conn, table)

    stock_starts: dict[str, str] = {}
    skipped = 0
    for sid in stock_ids:
        s = resolve_start_cached(
            sid, latest_dates, start, DATASET_START[dataset_key], force,
        )
        if s is None:
            skipped += 1
            # skipped 也記一筆，作為「今天 audit 看到此股已是最新」的證據
            _write_fetch_log(
                conn, table_name=table, stock_id=sid,
                fetch_mode=fetch_mode_override or "per_stock",
                fetch_date_from=None, fetch_date_to=None,
                rows_inserted=0, duration_ms=0,
                status="skipped", error_message="up_to_date",
            )
        else:
            stock_starts[sid] = s

    logger.info(f"[{table}] 需抓取：{len(stock_starts)} 支，已最新略過：{skipped} 支")
    if not stock_starts:
        return

    groups: dict[str, list] = defaultdict(list)
    for sid, s in stock_starts.items():
        groups[s].append(sid)

    total_api, total_rows = 0, 0
    batch_disabled = False
    failures: list[dict] = []
    pk_len = 3 if table == "securities_lending" else 2

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        # ── 批次模式 ──
        used_batch = False
        if use_batch and len(sids) >= batch_threshold and not batch_disabled:
            used_batch = True
            seg_start = group_start
            seg_end_dt = datetime.strptime(end, "%Y-%m-%d")
            chunk_rows = []
            t_batch_0 = time.time()
            batch_failed_msg: str | None = None
            try:
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_dt:
                        break
                    seg_end = min(
                        (seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"),
                        end,
                    )
                    logger.info(f"  [{table}] 批次 {seg_start}~{seg_end}（{len(sids)} 支）")
                    data = finmind_get(
                        dataset, {"start_date": seg_start, "end_date": seg_end},
                        delay, raise_on_batch_400=True,
                        raise_on_error=True
                    )
                    total_api += 1
                    chunk_rows.extend([r for r in data if r.get("stock_id") in sids_set])
                    seg_start = (
                        datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
                    ).strftime("%Y-%m-%d")
            except BatchNotSupportedError as e:
                logger.warning(f"  {e}；改逐支")
                batch_disabled = True
                used_batch = False
                chunk_rows = []
            except Exception as e:
                batch_failed_msg = str(e)
                logger.error(f"  [{table}] 批次抓取失敗：{e}")
                chunk_rows = []

            batch_duration_ms = int((time.time() - t_batch_0) * 1000)
            avg_dur = batch_duration_ms // max(len(sids), 1)

            if chunk_rows:
                # ── 逐支 commit + 逐支寫 fetch_log ──
                rows_by_stock: dict[str, list] = defaultdict(list)
                for r in chunk_rows:
                    sid = r.get("stock_id")
                    if sid not in sids_set:
                        continue
                    try:
                        rows_by_stock[sid].append(mapper(r))
                    except Exception as e:
                        logger.warning(f"  [{table}/{sid}] mapper 異常筆，跳過：{e}")

                # 每一支 batch 內被請求的股票都產出 1 筆 fetch_log
                for sid in sids:
                    s_rows = rows_by_stock.get(sid, [])
                    if not s_rows:
                        # 在 batch 範圍內但 API 沒回該股 → 視為 no_new_data
                        _write_fetch_log(
                            conn, table_name=table, stock_id=sid,
                            fetch_mode=fetch_mode_override or "batch",
                            fetch_date_from=group_start, fetch_date_to=end,
                            rows_inserted=0, duration_ms=avg_dur,
                            status="no_new_data",
                        )
                        continue
                    try:
                        rows_map = {}
                        for row in s_rows:
                            rows_map[row[:pk_len]] = row
                        final_rows = list(rows_map.values())
                        n = safe_commit_rows(
                            conn, upsert_sql, final_rows, template,
                            label=f"{table}/{sid}",
                        )
                        total_rows += n
                        _write_fetch_log(
                            conn, table_name=table, stock_id=sid,
                            fetch_mode=fetch_mode_override or "batch",
                            fetch_date_from=group_start, fetch_date_to=end,
                            rows_inserted=n, duration_ms=avg_dur,
                            status="success" if n > 0 else "partial",
                        )
                    except Exception as e:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        failures.append({"stock_id": sid, "error": str(e)})
                        logger.error(f"  [{table}/{sid}] 寫入失敗：{e}")
                        _write_fetch_log(
                            conn, table_name=table, stock_id=sid,
                            fetch_mode=fetch_mode_override or "batch",
                            fetch_date_from=group_start, fetch_date_to=end,
                            rows_inserted=0, duration_ms=avg_dur,
                            status="failed", error_message=str(e),
                        )

                logger.info(
                    f"  [{table}] 批次寫入完成（含 {len(rows_by_stock)} 支股票）"
                )
            elif batch_failed_msg:
                # 批次整段失敗 → 該 group 全部標 failed
                for sid in sids:
                    _write_fetch_log(
                        conn, table_name=table, stock_id=sid,
                        fetch_mode=fetch_mode_override or "batch",
                        fetch_date_from=group_start, fetch_date_to=end,
                        rows_inserted=0, duration_ms=avg_dur,
                        status="failed", error_message=batch_failed_msg,
                    )

        # ── 逐支模式 ──
        if (not use_batch) or len(sids) < batch_threshold or batch_disabled:
            if used_batch:
                # 已在 batch 區段處理過，避免重複
                continue
            for sid in sids:
                t_stock_0 = time.time()
                try:
                    data = finmind_get(
                        dataset, {"data_id": sid, "start_date": group_start, "end_date": end},
                        delay,
                        raise_on_error=True
                    )
                    total_api += 1
                    if not data:
                        dur = int((time.time() - t_stock_0) * 1000)
                        _write_fetch_log(
                            conn, table_name=table, stock_id=sid,
                            fetch_mode=fetch_mode_override or "per_stock",
                            fetch_date_from=group_start, fetch_date_to=end,
                            rows_inserted=0, duration_ms=dur,
                            status="no_new_data",
                        )
                        continue
                    rows_map = {}
                    for r in data:
                        try:
                            mapped = mapper(r)
                            rows_map[mapped[:pk_len]] = mapped
                        except Exception as e:
                            logger.warning(f"  [{table}/{sid}] mapper 異常筆，跳過：{e}")

                    rows = list(rows_map.values())
                    n = safe_commit_rows(
                        conn, upsert_sql, rows, template,
                        label=f"{table}/{sid}",
                    )
                    total_rows += n
                    dur = int((time.time() - t_stock_0) * 1000)
                    _write_fetch_log(
                        conn, table_name=table, stock_id=sid,
                        fetch_mode=fetch_mode_override or "per_stock",
                        fetch_date_from=group_start, fetch_date_to=end,
                        rows_inserted=n, duration_ms=dur,
                        status="success" if n > 0 else "partial",
                    )
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    failures.append({"stock_id": sid, "error": str(e)})
                    logger.error(f"  [{table}/{sid}] 失敗：{e}")
                    dur = int((time.time() - t_stock_0) * 1000)
                    _write_fetch_log(
                        conn, table_name=table, stock_id=sid,
                        fetch_mode=fetch_mode_override or "per_stock",
                        fetch_date_from=group_start, fetch_date_to=end,
                        rows_inserted=0, duration_ms=dur,
                        status="failed", error_message=str(e),
                    )

    dump_failures(table, failures)
    logger.info(f"[{table}] 完成 API:{total_api} 寫入:{total_rows} 失敗:{len(failures)}")


# ═════════════════════════════════════════════
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ═════════════════════════════════════════════
def query_failed_targets(
    conn, days: int, target_tables: list[str],
) -> dict[str, list[str | None]]:
    """
    找出近 N 天 status='failed' 的 (table_name, stock_id) 對，且**最近一筆**
    狀態仍為 failed（即還沒被後續成功覆蓋）。

    回傳：{table_name: [stock_id, ...]}，市場層級表的 stock_id 為 None。
    """
    targets: dict[str, list[str | None]] = defaultdict(list)
    sql = """
    WITH recent AS (
        SELECT table_name, stock_id, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY table_name, stock_id ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = ANY(%s)
          AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT table_name, stock_id
    FROM recent
    WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (target_tables, str(days)))
            for tbl, sid in cur.fetchall():
                targets[tbl].append(sid)
    except Exception as e:
        logger.error(f"[retry-failed] 查詢 fetch_log 失敗：{e}")
        return {}

    for tbl in targets:
        n = len(targets[tbl])
        sample = [s for s in targets[tbl] if s is not None][:5]
        logger.info(f"  [retry-failed/{tbl}] {n} 個目標"
                    + (f"，例：{sample}" if sample else "（市場層級）"))
    return targets


def query_gap_targets(
    conn, days: int, target_tables: list[str], all_stock_ids: list[str],
) -> dict[str, list[str | None]]:
    """
    找出近 N 天「沒有 success 紀錄」的 (table_name, stock_id) 對。

    對市場層級表：若整段時間內沒有任何 success → 整張表回 stock_id=None
    對個股層級表：對 all_stock_ids 中每支股，若無 success → 加入清單
    """
    targets: dict[str, list[str | None]] = defaultdict(list)

    for tbl in target_tables:
        if tbl in MARKET_LEVEL_TABLES:
            sql = """
            SELECT 1 FROM fetch_log
            WHERE table_name = %s AND status = 'success'
              AND run_ts > NOW() - (%s || ' days')::interval
            LIMIT 1;
            """
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (tbl, str(days)))
                    if cur.fetchone() is None:
                        targets[tbl].append(None)
            except Exception as e:
                logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")
        else:
            # 個股層：找出 all_stock_ids 中近 N 天無 success 紀錄者
            sql = """
            SELECT DISTINCT stock_id FROM fetch_log
            WHERE table_name = %s AND status = 'success'
              AND run_ts > NOW() - (%s || ' days')::interval
              AND stock_id = ANY(%s);
            """
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (tbl, str(days), all_stock_ids))
                    have_success = {row[0] for row in cur.fetchall()}
                missing = [sid for sid in all_stock_ids if sid not in have_success]
                targets[tbl].extend(missing)
            except Exception as e:
                logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, lst in targets.items():
        sample = [s for s in lst if s is not None][:5]
        logger.info(f"  [gap-fill/{tbl}] {len(lst)} 個目標"
                    + (f"，例：{sample}" if sample else "（市場層級）"))
    return targets


# ═════════════════════════════════════════════
# 表名映射回 (dataset, ddl, upsert, template, mapper)，給 retry/gap-fill 用
# ═════════════════════════════════════════════
def _config_for_table(table: str):
    table_configs = {
        "total_margin_short": (
            "TaiwanStockTotalMarginPurchaseShortSale", DDL_TOTAL_MARGIN_SHORT,
            UPSERT_TOTAL_MARGIN_SHORT, "(%s, %s, %s, %s, %s, %s, %s)",
            map_total_margin,
        ),
        "total_inst_investors": (
            "TaiwanStockTotalInstitutionalInvestors", DDL_TOTAL_INST,
            UPSERT_TOTAL_INST, "(%s, %s, %s, %s)", map_total_inst,
        ),
        "securities_lending": (
            "TaiwanStockSecuritiesLending", DDL_SBL,
            UPSERT_SBL, "(%s, %s, %s, %s, %s, %s, %s, %s)", map_sbl,
        ),
        "daily_short_balance": (
            "TaiwanDailyShortSaleBalances", DDL_DAILY_SHORT,
            UPSERT_DAILY_SHORT,
            "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            map_daily_short,
        ),
        "margin_short_suspension": (
            "TaiwanStockMarginShortSaleSuspension", DDL_MARGIN_SHORT_SUSPENSION,
            UPSERT_MARGIN_SHORT_SUSPENSION, "(%s, %s, %s, %s)", map_margin_susp,
        ),
    }
    return table_configs[table]


def _run_targeted(
    conn, targets: dict[str, list[str | None]], args, fetch_mode: str,
) -> None:
    """retry-failed / gap-fill 共用執行器：強制 force=True。"""
    use_batch = not args.per_stock
    for tbl, sids in targets.items():
        if not sids:
            continue
        dataset, ddl, upsert, tmpl, mapper = _config_for_table(tbl)

        if tbl in MARKET_LEVEL_TABLES:
            # 市場層級：sids = [None] 即代表此表需要重抓
            fetch_market_dataset(
                conn, dataset, tbl, ddl, upsert, tmpl, mapper, tbl,
                args.start, args.end, args.delay, force=True,
            )
        else:
            # 個股層級：只跑指定的股票清單
            real_sids = [s for s in sids if s is not None]
            if not real_sids:
                continue
            fetch_per_stock_dataset(
                conn, dataset, tbl, ddl, upsert, tmpl, mapper, tbl,
                real_sids, args.start, args.end, args.delay,
                force=True, use_batch=use_batch,
                fetch_mode_override=fetch_mode,
            )


# ═════════════════════════════════════════════
# main
# ═════════════════════════════════════════════
# ═════════════════════════════════════════════
# main
# ═════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+",
                   choices=list(DATASET_START.keys()),
                   default=list(DATASET_START.keys()))
    p.add_argument("--stock-id", type=str, default=None)
    p.add_argument("--start", type=str, default="2001-01-01")
    p.add_argument("--end",   type=str, default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--per-stock", action="store_true")
    # v2.3 新增
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS",
                   help="重試 fetch_log 中近 N 天最後狀態為 failed 的 (table, stock) 對")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS",
                   help="補抓 fetch_log 中近 N 天無 success 紀錄的 (table, stock) 對")
    args = p.parse_args()

    use_batch = not args.per_stock

    conn = get_db_conn()
    try:
        # 模式 A/B/C 判斷
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            # ⭐ DB 驅動：優先從 stocks 資料表抓取開啟 fetch_chip 的標的
            core_stocks = get_core_stocks_from_db(conn, require_active=True)
            if core_stocks:
                stock_ids = [sid for sid, cfg in core_stocks.items() if cfg.get("fetch_chip")]
                logger.info(f"從資料庫載入 {len(stock_ids)} 支核心籌碼標的")
            else:
                stock_ids = get_db_stock_ids(conn, types=("twse", "otc"))
                logger.warning(f"stocks 表無資料，Fallback 使用全市場標的 ({len(stock_ids)} 支)")

        # ─────────────────────────────────────────
        # 模式 A：retry-failed
        # ─────────────────────────────────────────
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, args.tables)
            if not targets:
                logger.info("沒有找到需要重試的目標，結束。")
                return
            _run_targeted(conn, targets, args, fetch_mode="retry")
            return

        # ─────────────────────────────────────────
        # 模式 B：gap-fill
        # ─────────────────────────────────────────
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, args.tables, stock_ids)
            if not targets:
                logger.info("沒有找到需要補抓的目標，結束。")
                return
            _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            return

        # ─────────────────────────────────────────
        # 模式 C：常規邏輯
        # ─────────────────────────────────────────
        # 1. 市場層資料
        if "total_margin_short" in args.tables:
            fetch_market_dataset(
                conn, "TaiwanStockTotalMarginPurchaseShortSale", "total_margin_short",
                DDL_TOTAL_MARGIN_SHORT, UPSERT_TOTAL_MARGIN_SHORT,
                "(%s, %s, %s, %s, %s, %s, %s)", map_total_margin,
                "total_margin_short", args.start, args.end, args.delay, args.force,
            )
        if "total_inst_investors" in args.tables:
            fetch_market_dataset(
                conn, "TaiwanStockTotalInstitutionalInvestors", "total_inst_investors",
                DDL_TOTAL_INST, UPSERT_TOTAL_INST,
                "(%s, %s, %s, %s)", map_total_inst,
                "total_inst_investors", args.start, args.end, args.delay, args.force,
            )

        # 2. 個股層資料
        per_stock_configs = [
            ("securities_lending", "TaiwanStockSecuritiesLending", DDL_SBL,
             UPSERT_SBL, "(%s, %s, %s, %s, %s, %s, %s, %s)", map_sbl),
            ("daily_short_balance", "TaiwanDailyShortSaleBalances", DDL_DAILY_SHORT,
             UPSERT_DAILY_SHORT,
             "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", map_daily_short),
            ("margin_short_suspension", "TaiwanStockMarginShortSaleSuspension",
             DDL_MARGIN_SHORT_SUSPENSION, UPSERT_MARGIN_SHORT_SUSPENSION,
             "(%s, %s, %s, %s)", map_margin_susp),
        ]
        for key, dataset, ddl, upsert, tmpl, mapper in per_stock_configs:
            if key not in args.tables:
                continue
            fetch_per_stock_dataset(
                conn, dataset, key, ddl, upsert, tmpl, mapper, key,
                stock_ids, args.start, args.end, args.delay, args.force, use_batch,
            )
    finally:
        conn.close()
    logger.info("全部完成")


if __name__ == "__main__":
    main()
