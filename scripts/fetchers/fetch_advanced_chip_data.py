from __future__ import annotations
import sys
import logging
from pathlib import Path

# ── sys.path 自我修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for sub in ("", "core", "fetchers"):
    p = (_SCRIPTS_DIR / sub) if sub else _SCRIPTS_DIR
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

"""
fetch_advanced_chip_data.py v3.0 — 進階籌碼與融資融券（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 整合 core v3.0：全面使用 FailureLogger、safe_commit_rows。
  ★ 逐支逐日 commit：個股資料（SBL, DailyShort）採用最細粒度的 commit 策略。
  ★ 斷路器與統計：整合 finmind_client v3.0 統計報表。
  ★ 原子寫入：失敗清單採原子寫入，確保日誌完整。
"""

import argparse
import time
from collections import defaultdict
from datetime import date, datetime, timedelta

from core.path_setup import ensure_scripts_on_path, get_outputs_dir, ensure_dirs_exist
ensure_scripts_on_path(__file__)

from core.finmind_client import finmind_get, BatchNotSupportedError, get_request_stats
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    safe_int,
    get_db_stock_ids,
    get_all_safe_starts,
    get_market_safe_start,
    resolve_start_cached,
    FailureLogger,
    map_rows_safe,
    commit_per_day,
    commit_per_stock_per_day,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 初始化目錄
ensure_dirs_exist()
OUTPUT_DIR = get_outputs_dir()

DATASET_START = {
    "total_margin_short":      "2001-01-01",
    "total_inst_investors":    "2004-04-01",
    "securities_lending":      "2001-05-01",
    "daily_short_balance":     "2005-07-01",
    "margin_short_suspension": "2015-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL & SQL
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
"""

DDL_TOTAL_INST = """
CREATE TABLE IF NOT EXISTS total_inst_investors (
    date DATE,
    name VARCHAR(100),
    buy  BIGINT,
    sell BIGINT,
    PRIMARY KEY (date, name)
);
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
"""

DDL_MARGIN_SHORT_SUSPENSION = """
CREATE TABLE IF NOT EXISTS margin_short_suspension (
    date     DATE,
    stock_id VARCHAR(50),
    end_date DATE,
    reason   VARCHAR(500),
    PRIMARY KEY (date, stock_id)
);
"""

UPSERT_TOTAL_MARGIN_SHORT = """
INSERT INTO total_margin_short (date, name, today_balance, yes_balance, buy, sell, return_qty)
VALUES %s ON CONFLICT (date, name) DO UPDATE SET
    today_balance = EXCLUDED.today_balance, yes_balance = EXCLUDED.yes_balance,
    buy = EXCLUDED.buy, sell = EXCLUDED.sell, return_qty = EXCLUDED.return_qty;
"""

UPSERT_TOTAL_INST = """
INSERT INTO total_inst_investors (date, name, buy, sell)
VALUES %s ON CONFLICT (date, name) DO UPDATE SET
    buy = EXCLUDED.buy, sell = EXCLUDED.sell;
"""

UPSERT_SBL = """
INSERT INTO securities_lending (
    date, stock_id, transaction_type, volume, fee_rate, close,
    original_return_date, original_lending_period
) VALUES %s ON CONFLICT (date, stock_id, transaction_type) DO UPDATE SET
    volume = EXCLUDED.volume, fee_rate = EXCLUDED.fee_rate, close = EXCLUDED.close,
    original_return_date = EXCLUDED.original_return_date,
    original_lending_period = EXCLUDED.original_lending_period;
"""

UPSERT_DAILY_SHORT = """
INSERT INTO daily_short_balance (
    date, stock_id, margin_short_prev_balance, margin_short_short_sales,
    margin_short_short_covering, margin_short_stock_redemption,
    margin_short_current_balance, margin_short_quota,
    sbl_short_prev_balance, sbl_short_short_sales,
    sbl_short_returns, sbl_short_adjustments,
    sbl_short_current_balance, sbl_short_quota, sbl_short_short_covering
) VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET
    margin_short_prev_balance = EXCLUDED.margin_short_prev_balance,
    margin_short_short_sales = EXCLUDED.margin_short_short_sales,
    margin_short_current_balance = EXCLUDED.margin_short_current_balance,
    sbl_short_current_balance = EXCLUDED.sbl_short_current_balance;
"""

UPSERT_MARGIN_SHORT_SUSPENSION = """
INSERT INTO margin_short_suspension (date, stock_id, end_date, reason)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET
    end_date = EXCLUDED.end_date, reason = EXCLUDED.reason;
"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_total_margin(r: dict) -> tuple:
    return (
        r["date"], r.get("name"),
        safe_int(r.get("TodayBalance")), safe_int(r.get("YesBalance")),
        safe_int(r.get("buy")), safe_int(r.get("sell")), safe_int(r.get("Return")),
    )

def map_total_inst(r: dict) -> tuple:
    return (r["date"], r.get("name"), safe_int(r.get("buy")), safe_int(r.get("sell")))

def map_sbl(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"], r.get("transaction_type"),
        safe_int(r.get("volume")), safe_float(r.get("fee_rate")),
        safe_float(r.get("close")), r.get("original_return_date") or None,
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
    return (r["date"], r["stock_id"], r.get("end_date") or None, r.get("reason"))

# ─────────────────────────────────────────────
# 抓取邏輯
# ─────────────────────────────────────────────
def fetch_market_dataset(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    start: str, end: str, force: bool,
):
    ensure_ddl(conn, ddl)
    flog = FailureLogger(table, db_conn=conn)

    s = start
    if not force:
        safe_s = get_market_safe_start(conn, table)
        if safe_s and safe_s > end:
            logger.info(f"[{table}] 已最新")
            return
        if safe_s: s = max(safe_s, DATASET_START[dataset_key])

    logger.info(f"[{table}] 抓取 {s} ~ {end}")
    data = finmind_get(dataset, {"start_date": s, "end_date": end})
    if not data:
        logger.info(f"[{table}] 無新資料")
        return

    rows = map_rows_safe(mapper, data, label=table)
    # 市場層資料：逐日 commit
    results = commit_per_day(conn, upsert_sql, rows, template, label_prefix=table, failure_logger=flog)
    logger.info(f"[{table}] 完成，共寫入 {sum(results.values())} 筆，橫跨 {len(results)} 天")
    flog.summary()


def fetch_per_stock_dataset(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    stock_ids: list[str], start: str, end: str,
    force: bool, use_batch: bool,
    batch_threshold: int = 20, chunk_days: int = 60,
):
    ensure_ddl(conn, ddl)
    flog = FailureLogger(table, db_conn=conn)
    latest_dates = get_all_safe_starts(conn, table)

    stock_starts: dict[str, str] = {}
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest_dates, start, DATASET_START[dataset_key], force)
        if s: stock_starts[sid] = s

    if not stock_starts:
        logger.info(f"[{table}] 全數已最新")
        return

    logger.info(f"[{table}] 待補抓：{len(stock_starts)} 支")
    groups = defaultdict(list)
    for sid, s in stock_starts.items(): groups[s].append(sid)

    batch_disabled = False
    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        # ── 批次模式 ──
        if use_batch and len(sids) >= batch_threshold and not batch_disabled:
            seg_start = group_start
            while True:
                seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                if seg_start_dt > datetime.strptime(end, "%Y-%m-%d"): break
                seg_end = min((seg_start_dt + timedelta(days=chunk_days-1)).strftime("%Y-%m-%d"), end)
                
                try:
                    logger.info(f"  [{table}] 批次 {seg_start}~{seg_end} ({len(sids)} 支)")
                    data = finmind_get(dataset, {"start_date": seg_start, "end_date": seg_end}, raise_on_batch_400=True)
                    chunk_rows = map_rows_safe(mapper, [r for r in data if r.get("stock_id") in sids_set], label=table)
                    
                    # ⭐ 逐支逐日 Commit (v3.0 最強規格) ⭐
                    commit_per_stock_per_day(conn, upsert_sql, chunk_rows, template, label_prefix=table, failure_logger=flog)
                except BatchNotSupportedError:
                    logger.warning(f"  [{table}] 不支援批次，轉為逐支模式")
                    batch_disabled = True; break
                except Exception as e:
                    flog.record(stock_id="BATCH", error=f"{seg_start} 失敗: {e}")

                seg_start = (datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        # ── 逐支模式 ──
        if not use_batch or len(sids) < batch_threshold or batch_disabled:
            for sid in sids:
                try:
                    data = finmind_get(dataset, {"data_id": sid, "start_date": group_start, "end_date": end})
                    if not data: continue
                    s_rows = map_rows_safe(mapper, data, label=f"{table}/{sid}")
                    # 逐日 commit (對該支股票而言)
                    commit_per_day(conn, upsert_sql, s_rows, template, label_prefix=f"{table}/{sid}", failure_logger=flog)
                except Exception as e:
                    flog.record(stock_id=sid, error=str(e))

    flog.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", choices=list(DATASET_START.keys()), default=list(DATASET_START.keys()))
    parser.add_argument("--stock-id", type=str)
    parser.add_argument("--start", type=str, default="2001-01-01")
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--per-stock", action="store_true")
    args = parser.parse_args()

    conn = get_db_conn()
    try:
        stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else get_db_stock_ids(conn)
        
        # 1. 市場層
        if "total_margin_short" in args.tables:
            fetch_market_dataset(conn, "TaiwanStockTotalMarginPurchaseShortSale", "total_margin_short", 
                                 DDL_TOTAL_MARGIN_SHORT, UPSERT_TOTAL_MARGIN_SHORT, "(%s,%s,%s,%s,%s,%s,%s)", 
                                 map_total_margin, "total_margin_short", args.start, args.end, args.force)
        
        if "total_inst_investors" in args.tables:
            fetch_market_dataset(conn, "TaiwanStockTotalInstitutionalInvestors", "total_inst_investors",
                                 DDL_TOTAL_INST, UPSERT_TOTAL_INST, "(%s,%s,%s,%s)", 
                                 map_total_inst, "total_inst_investors", args.start, args.end, args.force)

        # 2. 個股層
        configs = [
            ("securities_lending", "TaiwanStockSecuritiesLending", DDL_SBL, UPSERT_SBL, "(%s,%s,%s,%s,%s,%s,%s,%s)", map_sbl),
            ("daily_short_balance", "TaiwanDailyShortSaleBalances", DDL_DAILY_SHORT, UPSERT_DAILY_SHORT, 
             "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", map_daily_short),
            ("margin_short_suspension", "TaiwanStockMarginShortSaleSuspension", DDL_MARGIN_SHORT_SUSPENSION, 
             UPSERT_MARGIN_SHORT_SUSPENSION, "(%s,%s,%s,%s)", map_margin_susp),
        ]
        for key, ds, ddl, upsert, tmpl, mapper in configs:
            if key in args.tables:
                fetch_per_stock_dataset(conn, ds, key, ddl, upsert, tmpl, mapper, key, stock_ids, 
                                        args.start, args.end, args.force, not args.per_stock)
    finally:
        conn.close()
        get_request_stats().summary()

if __name__ == "__main__":
    main()