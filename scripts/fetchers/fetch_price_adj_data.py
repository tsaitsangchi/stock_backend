from __future__ import annotations
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ['fetchers', 'pipeline', 'training', 'monitor']: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
"""
fetch_price_adj_data.py — 還原股價 + 當沖交易 + 漲跌停價
==========================================================
[新增] 補齊三個技術面缺口：

  1. price_adj ← TaiwanStockPriceAdj（**P0 關鍵**）
     資料範圍：1994-10-01 ~ now
     用途：除權息調整後股價，消除個股 log return 在除權息日的人為跳空。
           目前 stock_price 是未調整價，特徵工程算 log return 會把除權息誤識為下跌。
     批次模式：Sponsor 可不帶 data_id 一次抓全市場（極大節省 API 配額）

  2. day_trading ← TaiwanStockDayTrading
     資料範圍：2014-01-01 ~ now
     用途：當沖比 = (BuyAmount + SellAmount) / 兩倍成交額。台股獨有的高頻情緒指標。
           當沖比 > 30% 通常代表投機度過熱，可作為 regime 訊號。
     批次模式：同上

  3. price_limit ← TaiwanStockPriceLimit
     資料範圍：2000-01-01 ~ now
     用途：漲跌停價。可衍生「漲停強度」、「是否觸及漲停」等微觀結構特徵。
           limit_up/limit_down 為 0 表示無漲跌幅限制（槓桿 ETF / 興櫃股），
           可作為股票類型分類依據。

衍生因子（建議在 feature_engineering.py 補上）：
  - log_return_adj   = log(close_adj_t / close_adj_{t-1})  ← 真正的乾淨報酬率
  - day_trading_pct  = (BuyAfterSale + SellAfterBuy) / total_volume
  - touched_limit_up = 1 if max >= limit_up * 0.998 else 0
  - limit_close_pct  = close / limit_up（漲幅佔上限的比例，動量強度）

執行：
    python fetch_price_adj_data.py                       # 全部，批次模式
    python fetch_price_adj_data.py --tables price_adj    # 只抓還原價
    python fetch_price_adj_data.py --per-stock           # 退回逐支模式
    python fetch_price_adj_data.py --force --start 2010-01-01
"""

import argparse
import logging
from datetime import date, datetime, timedelta

from config import DB_CONFIG  # noqa: F401
from core.finmind_client import finmind_get, BatchNotSupportedError
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    safe_float,
    safe_int,
    get_all_latest_dates,
    get_db_stock_ids,
    resolve_start_cached,
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

# 批次抓取設定（同 fetch_technical_data.py）
DEFAULT_CHUNK_DAYS              = 90
DEFAULT_BATCH_THRESHOLD         = 20
BATCH_RETURN_WARNING_THRESHOLD  = 5000

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Upsert SQL
# ─────────────────────────────────────────────
UPSERT_PRICE_ADJ = """
INSERT INTO price_adj
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

UPSERT_DAY_TRADING = """
INSERT INTO day_trading (date, stock_id, buy_after_sale, volume, buy_amount, sell_amount)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    buy_after_sale = EXCLUDED.buy_after_sale,
    volume         = EXCLUDED.volume,
    buy_amount     = EXCLUDED.buy_amount,
    sell_amount    = EXCLUDED.sell_amount;
"""

UPSERT_PRICE_LIMIT = """
INSERT INTO price_limit (date, stock_id, reference_price, limit_up, limit_down)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    reference_price = EXCLUDED.reference_price,
    limit_up        = EXCLUDED.limit_up,
    limit_down      = EXCLUDED.limit_down;
"""

# ─────────────────────────────────────────────
# Mapper
# ─────────────────────────────────────────────
def map_price_adj(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_int(r.get("Trading_Volume")),
        safe_int(r.get("Trading_money")),
        safe_float(r.get("open")),
        safe_float(r.get("max")),
        safe_float(r.get("min")),
        safe_float(r.get("close")),
        safe_float(r.get("spread")),
        safe_float(r.get("Trading_turnover")),
    )

def map_day_trading(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        r.get("BuyAfterSale"),
        safe_int(r.get("Volume")),
        safe_int(r.get("BuyAmount")),
        safe_int(r.get("SellAmount")),
    )

def map_price_limit(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_float(r.get("reference_price")),
        safe_float(r.get("limit_up")),
        safe_float(r.get("limit_down")),
    )

# ─────────────────────────────────────────────
# 通用批次抓取（與 fetch_technical_data.py 同邏輯）
# ─────────────────────────────────────────────
def fetch_dataset_unified(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper,
    dataset_key: str, stock_ids: list[str],
    start: str, end: str, delay: float, force: bool,
    use_batch: bool, batch_threshold: int, chunk_days: int,
):
    """
    若 use_batch=True 且某起始日股票數 >= batch_threshold，採批次模式
    （不帶 data_id，全市場一次抓），否則退回逐支。
    """
    ensure_ddl(conn, ddl)
    valid_set = set(stock_ids)
    latest_dates = get_all_latest_dates(conn, table)

    # 計算每支股票的 actual_start
    stock_starts: dict[str, str] = {}
    skipped = 0
    for sid in stock_ids:
        s = resolve_start_cached(
            sid, latest_dates, start,
            DATASET_START[dataset_key], force,
        )
        if s is None:
            skipped += 1
        else:
            stock_starts[sid] = s

    logger.info(f"[{table}] 需抓取：{len(stock_starts)} 支，已最新略過：{skipped} 支")
    if not stock_starts:
        return

    # 分組：相同 start_date 的股票歸為一組
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for sid, s in stock_starts.items():
        groups[s].append(sid)

    total_api = 0
    total_rows = 0
    batch_disabled = False

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        if use_batch and len(sids) >= batch_threshold and not batch_disabled:
            # 批次模式：按 chunk_days 切段
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
                    if len(data) >= BATCH_RETURN_WARNING_THRESHOLD:
                        logger.warning(
                            f"  [{table}] 批次回傳 {len(data)} 筆，可能逼近 API 上限"
                        )
                    chunk_rows.extend([r for r in data if r.get("stock_id") in sids_set])
                    seg_start = (
                        datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
                    ).strftime("%Y-%m-%d")
            except BatchNotSupportedError as e:
                logger.warning(f"  {e}；改逐支模式")
                batch_disabled = True
                chunk_rows = []

            if chunk_rows:
                rows = [mapper(r) for r in chunk_rows]
                bulk_upsert(conn, upsert_sql, rows, template)
                total_rows += len(rows)
                logger.info(f"  [{table}] 批次寫入 {len(rows)} 筆")

        if (not use_batch) or len(sids) < batch_threshold or batch_disabled:
            # 逐支模式
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

    logger.info(f"[{table}] 完成 API:{total_api} 寫入:{total_rows} 筆")

# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+",
                   choices=["price_adj", "day_trading", "price_limit"],
                   default=["price_adj", "day_trading", "price_limit"])
    p.add_argument("--stock-id", type=str, default=None,
                   help="指定股票代號（逗號分隔），未指定則抓 stock_info 內所有 twse/otc")
    p.add_argument("--start", type=str, default="1994-10-01")
    p.add_argument("--end",   type=str, default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--per-stock", action="store_true",
                   help="停用批次模式，退回逐支請求（相容/除錯用）")
    p.add_argument("--batch-threshold", type=int, default=DEFAULT_BATCH_THRESHOLD)
    p.add_argument("--chunk-days",      type=int, default=DEFAULT_CHUNK_DAYS)
    args = p.parse_args()

    use_batch = not args.per_stock

    conn = get_db_conn()
    try:
        # 取得目標股票
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            stock_ids = get_db_stock_ids(conn, types=("twse", "otc"))

        logger.info(f"目標股票：{len(stock_ids)} 支")
        logger.info(f"執行模式：{'批次模式' if use_batch else '逐支模式'} | 增量：{not args.force}")

        configs = [
            ("price_adj",   "TaiwanStockPriceAdj",      DDL_PRICE_ADJ,
             UPSERT_PRICE_ADJ, "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", map_price_adj),
            ("day_trading", "TaiwanStockDayTrading",    DDL_DAY_TRADING,
             UPSERT_DAY_TRADING, "(%s, %s, %s, %s, %s, %s)", map_day_trading),
            ("price_limit", "TaiwanStockPriceLimit",    DDL_PRICE_LIMIT,
             UPSERT_PRICE_LIMIT, "(%s, %s, %s, %s, %s)", map_price_limit),
        ]
        for key, dataset, ddl, upsert, tmpl, mapper in configs:
            if key not in args.tables:
                continue
            fetch_dataset_unified(
                conn, dataset, key, ddl, upsert, tmpl, mapper, key,
                stock_ids, args.start, args.end, args.delay, args.force,
                use_batch, args.batch_threshold, args.chunk_days,
            )
    finally:
        conn.close()
    logger.info("✅ 全部完成")

if __name__ == "__main__":
    main()
