"""
fetch_chip_data.py  v2.1（去重複化 + core 模組整合）
從 FinMind API 抓取籌碼面資料並寫入 PostgreSQL：
  - institutional_investors_buy_sell ← TaiwanStockInstitutionalInvestorsBuySell
  - margin_purchase_short_sale       ← TaiwanStockMarginPurchaseShortSale
  - shareholding                     ← TaiwanStockShareholding

v2.1 改進：
  · 修正頂部三重 sys.path 重複插入
  · 移除本地 get_db_conn 重複定義 → 改用 core.db_utils
  · 移除冗餘 FINMIND_API_URL 常數（已在 core.finmind_client 定義）
  · 修正 main() 中重複建立 conn + 重複取得 target_stocks 的問題
"""

import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import argparse
import logging
from datetime import date

import psycopg2

from config import DB_CONFIG
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    get_all_safe_starts,
    resolve_start_cached,
    safe_int,
    safe_float,
    safe_date,
)
from core.finmind_client import finmind_get

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 各資料集最早可用日期
# ──────────────────────────────────────────────
DATASET_START_DATES = {
    "institutional_investors_buy_sell": "2005-01-01",
    "margin_purchase_short_sale":       "2001-01-01",
    "shareholding":                     "2004-02-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "2001-01-01"

# ──────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────
DDL_INSTITUTIONAL_INVESTORS = """
CREATE TABLE IF NOT EXISTS institutional_investors_buy_sell (
    date     DATE,
    stock_id VARCHAR(50),
    buy      BIGINT,
    name     VARCHAR(50),
    sell     BIGINT,
    PRIMARY KEY (date, stock_id, name)
);
CREATE INDEX IF NOT EXISTS idx_institutional_investors_buy_sell_stock_id
    ON institutional_investors_buy_sell (stock_id);
"""

DDL_MARGIN_PURCHASE = """
CREATE TABLE IF NOT EXISTS margin_purchase_short_sale (
    date                              DATE,
    stock_id                          VARCHAR(50),
    margin_purchase_buy               INTEGER,
    margin_purchase_cash_repayment    INTEGER,
    margin_purchase_limit             BIGINT,
    margin_purchase_sell              INTEGER,
    margin_purchase_today_balance     INTEGER,
    margin_purchase_yesterday_balance INTEGER,
    note                              TEXT,
    offset_loan_and_short             INTEGER,
    short_sale_buy                    INTEGER,
    short_sale_cash_repayment         INTEGER,
    short_sale_limit                  BIGINT,
    short_sale_sell                   INTEGER,
    short_sale_today_balance          INTEGER,
    short_sale_yesterday_balance      INTEGER,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_margin_purchase_short_sale_stock_id
    ON margin_purchase_short_sale (stock_id);
"""

DDL_SHAREHOLDING = """
CREATE TABLE IF NOT EXISTS shareholding (
    date                                DATE,
    stock_id                            VARCHAR(50),
    stock_name                          VARCHAR(50),
    international_code                  VARCHAR(20),
    foreign_investment_remaining_shares BIGINT,
    foreign_investment_shares           BIGINT,
    foreign_investment_remain_ratio     NUMERIC(10,4),
    foreign_investment_shares_ratio     NUMERIC(10,4),
    foreign_investment_upper_limit_ratio NUMERIC(10,4),
    chinese_investment_upper_limit_ratio NUMERIC(10,4),
    number_of_shares_issued             BIGINT,
    recently_declare_date               DATE,
    note                                TEXT,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_shareholding_stock_id ON shareholding (stock_id);
"""

# ──────────────────────────────────────────────
# Upsert SQL
# ──────────────────────────────────────────────
UPSERT_INSTITUTIONAL_INVESTORS = """
INSERT INTO institutional_investors_buy_sell (date, stock_id, buy, name, sell)
VALUES %s
ON CONFLICT (date, stock_id, name) DO UPDATE SET
    buy  = EXCLUDED.buy,
    sell = EXCLUDED.sell;
"""

UPSERT_MARGIN_PURCHASE = """
INSERT INTO margin_purchase_short_sale (
    date, stock_id,
    margin_purchase_buy, margin_purchase_cash_repayment, margin_purchase_limit,
    margin_purchase_sell, margin_purchase_today_balance, margin_purchase_yesterday_balance,
    note, offset_loan_and_short,
    short_sale_buy, short_sale_cash_repayment, short_sale_limit,
    short_sale_sell, short_sale_today_balance, short_sale_yesterday_balance
) VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    margin_purchase_buy               = EXCLUDED.margin_purchase_buy,
    margin_purchase_cash_repayment    = EXCLUDED.margin_purchase_cash_repayment,
    margin_purchase_limit             = EXCLUDED.margin_purchase_limit,
    margin_purchase_sell              = EXCLUDED.margin_purchase_sell,
    margin_purchase_today_balance     = EXCLUDED.margin_purchase_today_balance,
    margin_purchase_yesterday_balance = EXCLUDED.margin_purchase_yesterday_balance,
    note                              = EXCLUDED.note,
    offset_loan_and_short             = EXCLUDED.offset_loan_and_short,
    short_sale_buy                    = EXCLUDED.short_sale_buy,
    short_sale_cash_repayment         = EXCLUDED.short_sale_cash_repayment,
    short_sale_limit                  = EXCLUDED.short_sale_limit,
    short_sale_sell                   = EXCLUDED.short_sale_sell,
    short_sale_today_balance          = EXCLUDED.short_sale_today_balance,
    short_sale_yesterday_balance      = EXCLUDED.short_sale_yesterday_balance;
"""

UPSERT_SHAREHOLDING = """
INSERT INTO shareholding (
    date, stock_id, stock_name, international_code,
    foreign_investment_remaining_shares, foreign_investment_shares,
    foreign_investment_remain_ratio, foreign_investment_shares_ratio,
    foreign_investment_upper_limit_ratio, chinese_investment_upper_limit_ratio,
    number_of_shares_issued, recently_declare_date, note
) VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    stock_name                           = EXCLUDED.stock_name,
    international_code                   = EXCLUDED.international_code,
    foreign_investment_remaining_shares  = EXCLUDED.foreign_investment_remaining_shares,
    foreign_investment_shares            = EXCLUDED.foreign_investment_shares,
    foreign_investment_remain_ratio      = EXCLUDED.foreign_investment_remain_ratio,
    foreign_investment_shares_ratio      = EXCLUDED.foreign_investment_shares_ratio,
    foreign_investment_upper_limit_ratio = EXCLUDED.foreign_investment_upper_limit_ratio,
    chinese_investment_upper_limit_ratio = EXCLUDED.chinese_investment_upper_limit_ratio,
    number_of_shares_issued              = EXCLUDED.number_of_shares_issued,
    recently_declare_date                = EXCLUDED.recently_declare_date,
    note                                 = EXCLUDED.note;
"""


# ──────────────────────────────────────────────
# 輔助函式
# ──────────────────────────────────────────────
def get_target_stock_ids(conn, stock_id_arg=None) -> list:
    if stock_id_arg:
        return [s.strip() for s in stock_id_arg.split(",")]
    from config import STOCK_CONFIGS
    return list(STOCK_CONFIGS.keys())


# ──────────────────────────────────────────────
# institutional_investors_buy_sell（三大法人）
# ──────────────────────────────────────────────
def fetch_institutional_investors_buy_sell(
    start_date: str, end_date: str, delay: float, force: bool, stock_ids: list
):
    logger.info(f"\n=== [institutional_investors_buy_sell] 開始（{len(stock_ids)} 支）===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_INSTITUTIONAL_INVESTORS)
        latest_dates = get_all_safe_starts(conn, "institutional_investors_buy_sell")
        total_rows = skipped = 0

        for i, sid in enumerate(stock_ids, 1):
            actual_start = resolve_start_cached(
                sid, latest_dates, start_date,
                DATASET_START_DATES["institutional_investors_buy_sell"], force
            )
            if actual_start is None:
                skipped += 1
                continue

            data = finmind_get(
                "TaiwanStockInstitutionalInvestorsBuySell",
                {"data_id": sid, "start_date": actual_start, "end_date": end_date},
                delay,
            )
            if not data:
                continue

            rows = [
                (
                    r["date"],
                    r["stock_id"],
                    safe_int(r.get("buy")),
                    str(r.get("name", ""))[:50],
                    safe_int(r.get("sell")),
                )
                for r in data
            ]
            total_rows += bulk_upsert(
                conn, UPSERT_INSTITUTIONAL_INVESTORS, rows,
                "(%s::date, %s, %s, %s, %s)",
            )
            if i % 100 == 0:
                logger.info(f"  進度：{i}/{len(stock_ids)}，累計 {total_rows} 筆（略過：{skipped} 支）")

    finally:
        conn.close()
    logger.info(
        f"=== [institutional_investors_buy_sell] 完成，"
        f"共寫入 {total_rows} 筆（略過：{skipped} 支）==="
    )


# ──────────────────────────────────────────────
# margin_purchase_short_sale（融資融券）
# ──────────────────────────────────────────────
def fetch_margin_purchase_short_sale(
    start_date: str, end_date: str, delay: float, force: bool, stock_ids: list
):
    logger.info(f"\n=== [margin_purchase_short_sale] 開始（{len(stock_ids)} 支）===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_MARGIN_PURCHASE)
        latest_dates = get_all_safe_starts(conn, "margin_purchase_short_sale")
        total_rows = skipped = 0

        for i, sid in enumerate(stock_ids, 1):
            actual_start = resolve_start_cached(
                sid, latest_dates, start_date,
                DATASET_START_DATES["margin_purchase_short_sale"], force
            )
            if actual_start is None:
                skipped += 1
                continue

            data = finmind_get(
                "TaiwanStockMarginPurchaseShortSale",
                {"data_id": sid, "start_date": actual_start, "end_date": end_date},
                delay,
            )
            if not data:
                continue

            rows = [
                (
                    r["date"],
                    r["stock_id"],
                    safe_int(r.get("MarginPurchaseBuy")),
                    safe_int(r.get("MarginPurchaseCashRepayment")),
                    safe_int(r.get("MarginPurchaseLimit")),
                    safe_int(r.get("MarginPurchaseSell")),
                    safe_int(r.get("MarginPurchaseTodayBalance")),
                    safe_int(r.get("MarginPurchaseYesterdayBalance")),
                    str(r.get("Note", "") or ""),
                    safe_int(r.get("OffsetLoanAndShort")),
                    safe_int(r.get("ShortSaleBuy")),
                    safe_int(r.get("ShortSaleCashRepayment")),
                    safe_int(r.get("ShortSaleLimit")),
                    safe_int(r.get("ShortSaleSell")),
                    safe_int(r.get("ShortSaleTodayBalance")),
                    safe_int(r.get("ShortSaleYesterdayBalance")),
                )
                for r in data
            ]
            total_rows += bulk_upsert(
                conn, UPSERT_MARGIN_PURCHASE, rows,
                "(%s::date, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            )
            if i % 100 == 0:
                logger.info(f"  進度：{i}/{len(stock_ids)}，累計 {total_rows} 筆（略過：{skipped} 支）")

    finally:
        conn.close()
    logger.info(
        f"=== [margin_purchase_short_sale] 完成，"
        f"共寫入 {total_rows} 筆（略過：{skipped} 支）==="
    )


# ──────────────────────────────────────────────
# shareholding（外資持股）
# ──────────────────────────────────────────────
def fetch_shareholding(
    start_date: str, end_date: str, delay: float, force: bool, stock_ids: list
):
    logger.info(f"\n=== [shareholding] 開始（{len(stock_ids)} 支）===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_SHAREHOLDING)
        latest_dates = get_all_safe_starts(conn, "shareholding")
        total_rows = skipped = 0

        for i, sid in enumerate(stock_ids, 1):
            actual_start = resolve_start_cached(
                sid, latest_dates, start_date,
                DATASET_START_DATES["shareholding"], force
            )
            if actual_start is None:
                skipped += 1
                continue

            data = finmind_get(
                "TaiwanStockShareholding",
                {"data_id": sid, "start_date": actual_start, "end_date": end_date},
                delay,
            )
            if not data:
                continue

            rows = [
                (
                    r["date"],
                    r["stock_id"],
                    str(r.get("stock_name", "") or "")[:50],
                    str(r.get("InternationalCode", "") or "")[:20],
                    safe_int(r.get("ForeignInvestmentRemainingShares")),
                    safe_int(r.get("ForeignInvestmentShares")),
                    safe_float(r.get("ForeignInvestmentRemainRatio")),
                    safe_float(r.get("ForeignInvestmentSharesRatio")),
                    safe_float(r.get("ForeignInvestmentUpperLimitRatio")),
                    safe_float(r.get("ChineseInvestmentUpperLimitRatio")),
                    safe_int(r.get("NumberOfSharesIssued")),
                    safe_date(r.get("RecentlyDeclareDate")),
                    str(r.get("note", "") or ""),
                )
                for r in data
            ]
            total_rows += bulk_upsert(
                conn, UPSERT_SHAREHOLDING, rows,
                (
                    "(%s::date, %s, %s, %s,"
                    " %s, %s,"
                    " %s::numeric, %s::numeric, %s::numeric, %s::numeric,"
                    " %s, %s::date, %s)"
                ),
            )
            if i % 100 == 0:
                logger.info(f"  進度：{i}/{len(stock_ids)}，累計 {total_rows} 筆（略過：{skipped} 支）")

    finally:
        conn.close()
    logger.info(
        f"=== [shareholding] 完成，"
        f"共寫入 {total_rows} 筆（略過：{skipped} 支）==="
    )


# ──────────────────────────────────────────────
# CLI 主程式
# ──────────────────────────────────────────────
TABLE_FUNCS = {
    "institutional_investors_buy_sell": fetch_institutional_investors_buy_sell,
    "margin_purchase_short_sale":       fetch_margin_purchase_short_sale,
    "shareholding":                     fetch_shareholding,
}


def parse_args():
    parser = argparse.ArgumentParser(description="FinMind 籌碼面資料抓取工具 v2.1")
    parser.add_argument(
        "--tables", nargs="+",
        choices=list(TABLE_FUNCS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end",   default=DEFAULT_END)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stock-id", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    tables = list(TABLE_FUNCS.keys()) if "all" in args.tables else args.tables

    conn = get_db_conn()
    target_stocks = get_target_stock_ids(conn, args.stock_id)
    conn.close()

    mode = "強制重抓" if args.force else "增量模式（自動跳過已最新資料）"
    logger.info(f"抓取資料表：{tables}")
    logger.info(f"模式：{mode}，目標股票數：{len(target_stocks)}")
    logger.info(f"日期區間：{args.start} ~ {args.end}")

    for table in tables:
        try:
            TABLE_FUNCS[table](args.start, args.end, args.delay, args.force, target_stocks)
        except psycopg2.OperationalError as e:
            logger.error(f"PostgreSQL 連線失敗：{e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"[{table}] 未預期錯誤：{e}")
            raise


if __name__ == "__main__":
    main()
