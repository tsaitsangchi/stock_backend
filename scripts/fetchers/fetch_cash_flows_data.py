from __future__ import annotations
import sys
import json
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_cash_flows_data.py — 現金流量表 + 除權息結果（逐支 commit 完整性版）
======================================================================
v2.2 改進：
  · safe_commit_rows()：每支股票寫入後立即 commit，失敗 rollback。
  · 主迴圈以 try/except 包單支，單支失敗不影響其他股票。
  · 失敗清單寫入 outputs/{table}_failed_{date}.json。

執行：
    python fetch_cash_flows_data.py
    python fetch_cash_flows_data.py --tables cash_flows_statement
    python fetch_cash_flows_data.py --stock-id 2330
    python fetch_cash_flows_data.py --force --start 2008-06-01
"""

import argparse
import logging
from datetime import date, datetime, timedelta

from config import DB_CONFIG  # noqa: F401（db_utils 內部使用）
from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    safe_float,
    safe_date,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_START = {
    "cash_flows_statement": "2008-06-01",
    "dividend_result":      "2003-05-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
DDL_CASH_FLOWS = """
CREATE TABLE IF NOT EXISTS cash_flows_statement (
    date        DATE,
    stock_id    VARCHAR(50),
    type        VARCHAR(100),
    value       NUMERIC(20,4),
    origin_name VARCHAR(200),
    PRIMARY KEY (date, stock_id, type)
);
CREATE INDEX IF NOT EXISTS idx_cf_stock ON cash_flows_statement (stock_id, date);
"""

DDL_DIVIDEND_RESULT = """
CREATE TABLE IF NOT EXISTS dividend_result (
    date                       DATE,
    stock_id                   VARCHAR(50),
    before_price               NUMERIC(20,4),
    after_price                NUMERIC(20,4),
    stock_and_cache_dividend   NUMERIC(20,4),
    stock_or_cache_dividend    VARCHAR(20),
    max_price                  NUMERIC(20,4),
    min_price                  NUMERIC(20,4),
    open_price                 NUMERIC(20,4),
    reference_price            NUMERIC(20,4),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_div_result_stock ON dividend_result (stock_id, date);
"""

# ─────────────────────────────────────────────
# Upsert SQL
# ─────────────────────────────────────────────
UPSERT_CASH_FLOWS = """
INSERT INTO cash_flows_statement (date, stock_id, type, value, origin_name)
VALUES %s
ON CONFLICT (date, stock_id, type) DO UPDATE SET
    value       = EXCLUDED.value,
    origin_name = EXCLUDED.origin_name;
"""

UPSERT_DIVIDEND_RESULT = """
INSERT INTO dividend_result (
    date, stock_id, before_price, after_price,
    stock_and_cache_dividend, stock_or_cache_dividend,
    max_price, min_price, open_price, reference_price
) VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    before_price             = EXCLUDED.before_price,
    after_price              = EXCLUDED.after_price,
    stock_and_cache_dividend = EXCLUDED.stock_and_cache_dividend,
    stock_or_cache_dividend  = EXCLUDED.stock_or_cache_dividend,
    max_price                = EXCLUDED.max_price,
    min_price                = EXCLUDED.min_price,
    open_price               = EXCLUDED.open_price,
    reference_price          = EXCLUDED.reference_price;
"""


# ─────────────────────────────────────────────
# 逐支 commit 工具函式
# ─────────────────────────────────────────────
def safe_commit_rows(conn, upsert_sql: str, rows: list, template: str,
                      label: str = "") -> int:
    if not rows:
        return 0
    try:
        n = bulk_upsert(conn, upsert_sql, rows, template)
        conn.commit()
        return n
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f"  [{label}] 寫入失敗，已 rollback：{e}")
        return 0


def dump_failures(table: str, failures: list) -> None:
    if not failures:
        return
    out = OUTPUT_DIR / f"{table}_failed_{date.today().strftime('%Y%m%d')}.json"
    try:
        out.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"  失敗清單已寫入：{out}（{len(failures)} 支）")
    except Exception as e:
        logger.warning(f"  寫入失敗清單時發生錯誤：{e}")


# ─────────────────────────────────────────────
# Mapper
# ─────────────────────────────────────────────
def map_cf_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"], r.get("type"),
        safe_float(r.get("value")),
        r.get("origin_name"),
    )

def map_dr_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_float(r.get("before_price")),
        safe_float(r.get("after_price")),
        safe_float(r.get("stock_and_cache_dividend")),
        r.get("stock_or_cache_dividend"),
        safe_float(r.get("max_price")),
        safe_float(r.get("min_price")),
        safe_float(r.get("open_price")),
        safe_float(r.get("reference_price")),
    )

# ─────────────────────────────────────────────
# 增量起始日：以 DB 最新日期 +1 天
# ─────────────────────────────────────────────
def latest_date(conn, table: str, stock_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MAX(date) FROM {table} WHERE stock_id = %s", (stock_id,)
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
    return None

def resolve_start(conn, table: str, stock_id: str, dataset_key: str,
                  global_start: str, force: bool) -> str | None:
    earliest = DATASET_START[dataset_key]
    if force:
        return max(global_start, earliest)
    last = latest_date(conn, table, stock_id)
    if not last:
        return max(global_start, earliest)
    next_day = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    if next_day > today:
        return None  # 已最新
    return max(next_day, earliest)

# ─────────────────────────────────────────────
# 抓取主邏輯（逐支股票，每支寫入後立即 commit）
# ─────────────────────────────────────────────
def fetch_dataset_per_stock(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, mapper, dataset_key: str,
    stock_ids: list[str], start: str, end: str,
    delay: float, force: bool,
):
    ensure_ddl(conn, ddl)
    conn.commit()
    logger.info(f"=== [{table}] 開始 ===")
    logger.info(f"  目標股票：{len(stock_ids)} 支")

    template = (
        "(%s, %s, %s, %s, %s)"
        if table == "cash_flows_statement"
        else "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )

    total_rows = 0
    skipped = 0
    failures = []

    for i, sid in enumerate(stock_ids, 1):
        try:
            s = resolve_start(conn, table, sid, dataset_key, start, force)
            if s is None:
                skipped += 1
                continue
            data = finmind_get(
                dataset, {"data_id": sid, "start_date": s, "end_date": end}, delay,
                raise_on_error=True
            )
            if not data:
                continue
            rows = [mapper(r) for r in data]
            n = safe_commit_rows(conn, upsert_sql, rows, template,
                                 label=f"{table}/{sid}")
            total_rows += n
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            failures.append({"stock_id": sid, "error": str(e)})
            logger.error(f"  [{table}/{sid}] 失敗：{e}")

        if i % 100 == 0:
            logger.info(f"  [{table}] 進度：{i}/{len(stock_ids)}，"
                        f"累計 {total_rows} 筆（略過：{skipped}，失敗：{len(failures)}）")

    dump_failures(table, failures)
    logger.info(
        f"  [{table}] 完成，共寫入 {total_rows} 筆"
        f"（略過已最新：{skipped} 支，失敗：{len(failures)} 支）"
    )

# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def get_target_stock_ids(stock_id_arg: str | None) -> list[str]:
    if stock_id_arg:
        return [s.strip() for s in stock_id_arg.split(",")]
    from config import STOCK_CONFIGS
    return list(STOCK_CONFIGS.keys())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+",
                   choices=["cash_flows_statement", "dividend_result"],
                   default=["cash_flows_statement", "dividend_result"])
    p.add_argument("--stock-id", type=str, default=None,
                   help="指定股票代號，逗號分隔；未指定則抓 STOCK_CONFIGS 全部")
    p.add_argument("--start", type=str, default="2003-05-01")
    p.add_argument("--end",   type=str, default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    stock_ids = get_target_stock_ids(args.stock_id)
    logger.info(f"抓取資料表：{args.tables}")
    logger.info(f"日期區間：{args.start} ~ {args.end}")
    logger.info(f"請求間隔：{args.delay} 秒  |  增量模式：{not args.force}")

    conn = get_db_conn()
    try:
        if "cash_flows_statement" in args.tables:
            fetch_dataset_per_stock(
                conn, "TaiwanStockCashFlowsStatement", "cash_flows_statement",
                DDL_CASH_FLOWS, UPSERT_CASH_FLOWS, map_cf_row,
                "cash_flows_statement", stock_ids,
                args.start, args.end, args.delay, args.force,
            )
        if "dividend_result" in args.tables:
            fetch_dataset_per_stock(
                conn, "TaiwanStockDividendResult", "dividend_result",
                DDL_DIVIDEND_RESULT, UPSERT_DIVIDEND_RESULT, map_dr_row,
                "dividend_result", stock_ids,
                args.start, args.end, args.delay, args.force,
            )
    finally:
        conn.close()
    logger.info("全部完成")

if __name__ == "__main__":
    main()