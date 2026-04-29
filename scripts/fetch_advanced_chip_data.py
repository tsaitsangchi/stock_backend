"""
fetch_advanced_chip_data.py — 進階籌碼與融資融券資料
======================================================
[新增] 補齊原 fetch_chip_data.py 沒抓的「市場層」與「放空」資料：

  1. total_margin_short ← TaiwanStockTotalMarginPurchaseShortSale
     資料範圍：2001-01-01 ~ now（Free）
     用途：整體市場融資融券餘額。散戶恐慌指標、信用過熱指標。
     非個股資料 → 以 name (融資/融券) 為 PK 的一部分。

  2. total_inst_investors ← TaiwanStockTotalInstitutionalInvestors
     資料範圍：2004-04-01 ~ now（Free）
     用途：整體市場三大法人買賣金額（外資/投信/自營商）。
           大盤主力資金方向，補強 institutional_investors_buy_sell（個股層）。

  3. securities_lending ← TaiwanStockSecuritiesLending
     資料範圍：2001-05-01 ~ now（Free per stock，Sponsor 可批次）
     用途：借券成交明細。機構放空的最直接證據（學術上的 short interest）。
           借券量激增 = 大量機構押注下跌，是強烈反向 smart money 訊號。

  4. daily_short_balance ← TaiwanDailyShortSaleBalances
     資料範圍：2005-07-01 ~ now（Free per stock，Sponsor 可批次）
     用途：信用額度總量管制餘額表。融券 + SBL（借券放空）的完整放空圖像。

  5. margin_short_suspension ← TaiwanStockMarginShortSaleSuspension
     資料範圍：2015-01-01 ~ now（Free per stock，Sponsor 可批次）
     用途：暫停融券賣出表。被列入暫停常是軋空前兆，事件後 5-10 日報酬顯著為正。

衍生因子（建議在 feature_engineering.py 補上）：
  - retail_panic_index = total_margin_balance.pct_change(5)（負值越大越恐慌）
  - sbl_short_intensity = sbl_short_sales / shares_outstanding
  - short_squeeze_signal = (margin_short_suspension flag) AND (foreign_buy > 0)
  - institutional_breadth = sum(三大法人買超天數，過去 5 日)

執行：
    python fetch_advanced_chip_data.py
    python fetch_advanced_chip_data.py --tables securities_lending daily_short_balance
    python fetch_advanced_chip_data.py --stock-id 2330 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

from config import DB_CONFIG  # noqa: F401
from core.finmind_client import finmind_get, BatchNotSupportedError
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    safe_float,
    safe_int,
    get_db_stock_ids,
    get_all_safe_starts,
    get_market_safe_start,
    resolve_start_cached,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATASET_START = {
    "total_margin_short":      "2001-01-01",
    "total_inst_investors":    "2004-04-01",
    "securities_lending":      "2001-05-01",
    "daily_short_balance":     "2005-07-01",
    "margin_short_suspension": "2015-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
DDL_TOTAL_MARGIN_SHORT = """
CREATE TABLE IF NOT EXISTS total_margin_short (
    date          DATE,
    name          VARCHAR(50),    -- 融資 / 融券
    today_balance BIGINT,
    yes_balance   BIGINT,
    buy           BIGINT,
    sell          BIGINT,
    return_qty    BIGINT,         -- API 欄位 Return（避開 SQL 保留字）
    PRIMARY KEY (date, name)
);
CREATE INDEX IF NOT EXISTS idx_tms_date ON total_margin_short (date);
"""

DDL_TOTAL_INST = """
CREATE TABLE IF NOT EXISTS total_inst_investors (
    date DATE,
    name VARCHAR(100),  -- 外資、投信、自營商等
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
# Upsert SQL
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
# Mapper
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

# ─────────────────────────────────────────────
# 抓取邏輯
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
                return
            s = max(safe_s, DATASET_START[dataset_key])
    s = max(s, DATASET_START[dataset_key])
    logger.info(f"[{table}] 抓取 {s} ~ {end}")
    data = finmind_get(dataset, {"start_date": s, "end_date": end}, delay)
    if not data:
        logger.info(f"[{table}] 無新資料")
        return
    rows = [mapper(r) for r in data]
    n = bulk_upsert(conn, upsert_sql, rows, template)
    logger.info(f"[{table}] 寫入 {n} 筆")

def fetch_per_stock_dataset(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    stock_ids: list[str], start: str, end: str,
    delay: float, force: bool, use_batch: bool,
    batch_threshold: int = 20, chunk_days: int = 90,
):
    """個股層資料：Sponsor 可不帶 data_id 批次抓全市場，否則逐支。"""
    ensure_ddl(conn, ddl)
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

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        if use_batch and len(sids) >= batch_threshold and not batch_disabled:
            seg_start = group_start
            seg_end_dt = datetime.strptime(end, "%Y-%m-%d")
            chunk_rows = []
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
                    )
                    total_api += 1
                    chunk_rows.extend([r for r in data if r.get("stock_id") in sids_set])
                    seg_start = (
                        datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
                    ).strftime("%Y-%m-%d")
            except BatchNotSupportedError as e:
                logger.warning(f"  {e}；改逐支")
                batch_disabled = True
                chunk_rows = []
            if chunk_rows:
                rows = [mapper(r) for r in chunk_rows]
                bulk_upsert(conn, upsert_sql, rows, template)
                total_rows += len(rows)

        if (not use_batch) or len(sids) < batch_threshold or batch_disabled:
            for sid in sids:
                data = finmind_get(
                    dataset, {"data_id": sid, "start_date": group_start, "end_date": end},
                    delay,
                )
                total_api += 1
                if not data:
                    continue
                rows = [mapper(r) for r in data]
                bulk_upsert(conn, upsert_sql, rows, template)
                total_rows += len(rows)

    logger.info(f"[{table}] 完成 API:{total_api} 寫入:{total_rows}")

# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
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
    args = p.parse_args()

    use_batch = not args.per_stock

    conn = get_db_conn()
    try:
        # 取得個股 list
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            stock_ids = get_db_stock_ids(conn, types=("twse", "otc"))

        # 1. 市場層資料（不需 stock_id）
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

        # 2. 個股層資料（Sponsor 批次最佳）
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
    logger.info("✅ 全部完成")

if __name__ == "__main__":
    main()
