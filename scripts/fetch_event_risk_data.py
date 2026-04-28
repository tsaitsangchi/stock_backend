"""
fetch_event_risk_data.py — 事件風險與股本變動資料
====================================================
[新增] 黑天鵝防護網與特徵跳變保護。
       這些資料雖然事件稀疏，但對訓練資料的「乾淨度」與持倉前的風險檢查至關重要。

  1. delisting ← TaiwanStockDelisting（Free，全市場一次抓）
     用途：下市櫃名單。signal_filter 應在持倉前 hard block 已下市標的。

  2. suspended ← TaiwanStockSuspended（Sponsor，全市場一次抓）
     用途：暫停交易公告（含恢復時間）。同上，避免對暫停期間的資料做特徵運算。

  3. capital_reduction ← TaiwanStockCapitalReductionReferencePrice（Free per-stock）
     用途：減資恢復買賣參考價。沒抓的話，減資日的 OHLCV 會出現巨大跳動被
           特徵工程誤識為信號。

  4. split_price ← TaiwanStockSplitPrice（Free，全表一次抓）
     用途：分割/反分割後參考價。同 capital_reduction 的特徵跳變保護。

  5. trading_date ← TaiwanStockTradingDate（Free，全表一次抓）
     用途：台股交易日表。feature_engineering 中所有 "shift(N)" 應該是「N 個交易日」
           而非「N 個自然日」，這張表是日曆對齊的基礎。

  6. market_value ← TaiwanStockMarketValue（Sponsor 批次最佳）
     資料範圍：2004-01-01 ~ now
     用途：個股每日市值。size factor（小市值 vs 大市值）— Fama-French 5-factor 之一。

  7. disposition_securities ← TaiwanStockDispositionSecuritiesPeriod（Backer/Sponsor）
     用途：處置股票公告。處置期間流動性受限，應作為持倉黑名單。

衍生因子（建議在 feature_engineering.py 補上）：
  - is_delisted        = stock_id 是否在 delisting 表內（hard block）
  - is_suspended_today = 今天是否有暫停交易公告（hard block）
  - in_disposition     = 處置期間內（限制持倉量）
  - log_market_cap     = log(market_value)（size factor）
  - market_cap_chg_30d = market_cap.pct_change(30)
  - days_since_capital_reduction = 距上次減資的天數（事件衝擊衰減特徵）

執行：
    python fetch_event_risk_data.py
    python fetch_event_risk_data.py --tables delisting suspended trading_date
    python fetch_event_risk_data.py --force
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
    get_all_latest_dates,
    resolve_start_cached,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
DDL_DELISTING = """
CREATE TABLE IF NOT EXISTS delisting (
    date       DATE,
    stock_id   VARCHAR(50),
    stock_name VARCHAR(200),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_delisting_stock ON delisting (stock_id);
"""

DDL_SUSPENDED = """
CREATE TABLE IF NOT EXISTS suspended (
    stock_id         VARCHAR(50),
    date             DATE,           -- 暫停日期
    suspension_time  VARCHAR(50),
    resumption_date  DATE,
    resumption_time  VARCHAR(50),
    PRIMARY KEY (stock_id, date, suspension_time)
);
CREATE INDEX IF NOT EXISTS idx_suspended_stock ON suspended (stock_id, date);
"""

DDL_CAPITAL_REDUCTION = """
CREATE TABLE IF NOT EXISTS capital_reduction (
    date                       DATE,
    stock_id                   VARCHAR(50),
    closing_last_trading       NUMERIC(20,4),  -- ClosingPriceonTheLastTradingDay
    post_reduction_ref         NUMERIC(20,4),  -- PostReductionReferencePrice
    limit_up                   NUMERIC(20,4),
    limit_down                 NUMERIC(20,4),
    opening_ref                NUMERIC(20,4),  -- OpeningReferencePrice
    exright_ref                NUMERIC(20,4),  -- ExrightReferencePrice
    reason                     VARCHAR(500),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_capred_stock ON capital_reduction (stock_id, date);
"""

DDL_SPLIT_PRICE = """
CREATE TABLE IF NOT EXISTS split_price (
    date          DATE,
    stock_id      VARCHAR(50),
    type          VARCHAR(20),    -- 分割 / 反分割
    before_price  NUMERIC(20,4),
    after_price   NUMERIC(20,4),
    max_price     NUMERIC(20,4),
    min_price     NUMERIC(20,4),
    open_price    NUMERIC(20,4),
    PRIMARY KEY (date, stock_id, type)
);
CREATE INDEX IF NOT EXISTS idx_split_stock ON split_price (stock_id, date);
"""

DDL_TRADING_DATE = """
CREATE TABLE IF NOT EXISTS trading_date (
    date DATE PRIMARY KEY
);
"""

DDL_MARKET_VALUE = """
CREATE TABLE IF NOT EXISTS market_value (
    date         DATE,
    stock_id     VARCHAR(50),
    market_value BIGINT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_mv_stock ON market_value (stock_id, date);
"""

DDL_DISPOSITION = """
CREATE TABLE IF NOT EXISTS disposition_securities (
    date            DATE,
    stock_id        VARCHAR(50),
    stock_name      VARCHAR(200),
    disposition_cnt INTEGER,
    condition       VARCHAR(500),
    measure         VARCHAR(500),
    period_start    DATE,
    period_end      DATE,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_disp_stock ON disposition_securities (stock_id, date);
"""

# ─────────────────────────────────────────────
# Upsert SQL
# ─────────────────────────────────────────────
UPSERT_DELISTING = """
INSERT INTO delisting (date, stock_id, stock_name)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET stock_name = EXCLUDED.stock_name;
"""

UPSERT_SUSPENDED = """
INSERT INTO suspended (stock_id, date, suspension_time, resumption_date, resumption_time)
VALUES %s
ON CONFLICT (stock_id, date, suspension_time) DO UPDATE SET
    resumption_date = EXCLUDED.resumption_date,
    resumption_time = EXCLUDED.resumption_time;
"""

UPSERT_CAPRED = """
INSERT INTO capital_reduction (
    date, stock_id, closing_last_trading, post_reduction_ref,
    limit_up, limit_down, opening_ref, exright_ref, reason
) VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    closing_last_trading = EXCLUDED.closing_last_trading,
    post_reduction_ref   = EXCLUDED.post_reduction_ref,
    limit_up             = EXCLUDED.limit_up,
    limit_down           = EXCLUDED.limit_down,
    opening_ref          = EXCLUDED.opening_ref,
    exright_ref          = EXCLUDED.exright_ref,
    reason               = EXCLUDED.reason;
"""

UPSERT_SPLIT = """
INSERT INTO split_price (date, stock_id, type, before_price, after_price, max_price, min_price, open_price)
VALUES %s
ON CONFLICT (date, stock_id, type) DO UPDATE SET
    before_price = EXCLUDED.before_price,
    after_price  = EXCLUDED.after_price,
    max_price    = EXCLUDED.max_price,
    min_price    = EXCLUDED.min_price,
    open_price   = EXCLUDED.open_price;
"""

UPSERT_TRADING_DATE = """
INSERT INTO trading_date (date) VALUES %s
ON CONFLICT (date) DO NOTHING;
"""

UPSERT_MARKET_VALUE = """
INSERT INTO market_value (date, stock_id, market_value)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET market_value = EXCLUDED.market_value;
"""

UPSERT_DISPOSITION = """
INSERT INTO disposition_securities (
    date, stock_id, stock_name, disposition_cnt,
    condition, measure, period_start, period_end
) VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    stock_name      = EXCLUDED.stock_name,
    disposition_cnt = EXCLUDED.disposition_cnt,
    condition       = EXCLUDED.condition,
    measure         = EXCLUDED.measure,
    period_start    = EXCLUDED.period_start,
    period_end      = EXCLUDED.period_end;
"""

# ─────────────────────────────────────────────
# Mapper
# ─────────────────────────────────────────────
def map_delisting(r): return (r["date"], r["stock_id"], r.get("stock_name"))

def map_suspended(r): return (
    r["stock_id"], r["date"], r.get("suspension_time"),
    r.get("resumption_date") or None, r.get("resumption_time"),
)

def map_capred(r): return (
    r["date"], r["stock_id"],
    safe_float(r.get("ClosingPriceonTheLastTradingDay")),
    safe_float(r.get("PostReductionReferencePrice")),
    safe_float(r.get("LimitUp")),
    safe_float(r.get("LimitDown")),
    safe_float(r.get("OpeningReferencePrice")),
    safe_float(r.get("ExrightReferencePrice")),
    r.get("ReasonforCapitalReduction"),
)

def map_split(r): return (
    r["date"], r["stock_id"], r.get("type"),
    safe_float(r.get("before_price")),
    safe_float(r.get("after_price")),
    safe_float(r.get("max_price")),
    safe_float(r.get("min_price")),
    safe_float(r.get("open_price")),
)

def map_trading_date(r): return (r["date"],)

def map_market_value(r): return (
    r["date"], r["stock_id"], safe_int(r.get("market_value")),
)

def map_disposition(r): return (
    r["date"], r["stock_id"], r.get("stock_name"),
    safe_int(r.get("disposition_cnt")),
    r.get("condition"), r.get("measure"),
    r.get("period_start") or None,
    r.get("period_end")   or None,
)

# ─────────────────────────────────────────────
# 抓取邏輯
# ─────────────────────────────────────────────
def latest_market_date(conn, table: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(date) FROM {table}")
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
    return None

def fetch_market_full_dump(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    start: str, end: str, delay: float, force: bool,
    pass_dates: bool = True,
):
    """
    全市場單次抓（不需 data_id）。delisting/suspended/split_price/trading_date/disposition 屬此類。
    pass_dates=False 表示 API 不接受 start/end_date（如 TaiwanStockDelisting）。
    """
    ensure_ddl(conn, ddl)
    s = start
    if not force:
        last = latest_market_date(conn, table)
        if last:
            next_d = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if next_d > end and pass_dates:
                logger.info(f"[{table}] 已最新（{last}），跳過")
                return
            s = max(next_d, DATASET_START[dataset_key])
    s = max(s, DATASET_START[dataset_key])

    params = {}
    if pass_dates:
        params = {"start_date": s, "end_date": end}
    logger.info(f"[{table}] 抓取 {s if pass_dates else 'ALL'} ~ {end if pass_dates else ''}")

    data = finmind_get(dataset, params, delay)
    if not data:
        logger.info(f"[{table}] 無新資料")
        return
    rows = [mapper(r) for r in data]
    n = bulk_upsert(conn, upsert_sql, rows, template)
    logger.info(f"[{table}] 寫入 {n} 筆")

def fetch_per_stock_full(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    stock_ids: list[str], start: str, end: str, delay: float,
    force: bool, use_batch: bool,
    batch_threshold: int = 20, chunk_days: int = 90,
):
    """逐股或批次抓（market_value, capital_reduction）。"""
    ensure_ddl(conn, ddl)
    valid_set = set(stock_ids)
    latest_dates = get_all_latest_dates(conn, table)

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

    groups = defaultdict(list)
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
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            stock_ids = get_db_stock_ids(conn, types=("twse", "otc"))

        # 全市場單次抓（無 stock_id）
        full_dump_configs = [
            ("delisting", "TaiwanStockDelisting", DDL_DELISTING,
             UPSERT_DELISTING, "(%s, %s, %s)", map_delisting, False),
            ("suspended", "TaiwanStockSuspended", DDL_SUSPENDED,
             UPSERT_SUSPENDED, "(%s, %s, %s, %s, %s)", map_suspended, True),
            ("split_price", "TaiwanStockSplitPrice", DDL_SPLIT_PRICE,
             UPSERT_SPLIT, "(%s, %s, %s, %s, %s, %s, %s, %s)", map_split, False),
            ("trading_date", "TaiwanStockTradingDate", DDL_TRADING_DATE,
             UPSERT_TRADING_DATE, "(%s,)", map_trading_date, False),
            ("disposition_securities", "TaiwanStockDispositionSecuritiesPeriod",
             DDL_DISPOSITION, UPSERT_DISPOSITION,
             "(%s, %s, %s, %s, %s, %s, %s, %s)", map_disposition, True),
        ]
        for key, dataset, ddl, upsert, tmpl, mapper, pass_dates in full_dump_configs:
            if key not in args.tables:
                continue
            fetch_market_full_dump(
                conn, dataset, key, ddl, upsert, tmpl, mapper, key,
                args.start, args.end, args.delay, args.force, pass_dates,
            )

        # 個股層批次抓
        per_stock_configs = [
            ("capital_reduction", "TaiwanStockCapitalReductionReferencePrice",
             DDL_CAPITAL_REDUCTION, UPSERT_CAPRED,
             "(%s, %s, %s, %s, %s, %s, %s, %s, %s)", map_capred),
            ("market_value", "TaiwanStockMarketValue", DDL_MARKET_VALUE,
             UPSERT_MARKET_VALUE, "(%s, %s, %s)", map_market_value),
        ]
        for key, dataset, ddl, upsert, tmpl, mapper in per_stock_configs:
            if key not in args.tables:
                continue
            fetch_per_stock_full(
                conn, dataset, key, ddl, upsert, tmpl, mapper, key,
                stock_ids, args.start, args.end, args.delay, args.force, use_batch,
            )
    finally:
        conn.close()
    logger.info("✅ 全部完成")

if __name__ == "__main__":
    main()
