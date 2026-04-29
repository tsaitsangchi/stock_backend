"""
fetch_chip_data.py
從 FinMind API 抓取籌碼面資料並寫入 PostgreSQL：
  - institutional_investors_buy_sell ← TaiwanStockInstitutionalInvestorsBuySell (三大法人, Free)
  - margin_purchase_short_sale       ← TaiwanStockMarginPurchaseShortSale       (融資融券, Free)
  - shareholding                     ← TaiwanStockShareholding                  (外資持股, Free)

需求套件：
    pip install requests psycopg2-binary

執行範例：
    # 首次全量抓取（自動從各資料集最早日期開始）
    python fetch_chip_data.py

    # 只抓三大法人
    python fetch_chip_data.py --tables institutional_investors_buy_sell

    # 強制重抓（忽略 DB 已有資料）
    python fetch_chip_data.py --force

注意事項：
  - 預設啟用「增量模式」：每支股票自動從 DB 最新日期的隔天開始抓，避免重複請求。
  - institutional_investors_buy_sell 的 PK 含 name（外資/投信/自營），
    增量判斷以 MAX(date) 為準。
  - 預設請求間隔 1.2 秒，可用 --delay 調整。
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime

import psycopg2
import psycopg2.extras
import requests

from config import DB_CONFIG

# ======================
# 設定 logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ======================
# FinMind API 設定
# ======================
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

# ======================
# PostgreSQL 連線設定
# ======================
# ======================
# 各資料集最早可用日期
# ======================
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
# [P0 重構] 工具函式統一改用 core 模組
# 移除本檔重複的 safe_*、ensure_ddl、bulk_upsert、finmind_get、wait_until_next_hour、get_db_conn
# ──────────────────────────────────────────────
from core.finmind_client import finmind_get, wait_until_next_hour  # noqa: E402,F401
from core.db_utils import (  # noqa: E402,F401
    safe_date,
    get_all_safe_starts,
    resolve_start_cached,
)


def get_target_stock_ids(conn, stock_id_arg=None):
    if stock_id_arg:
        return [s.strip() for s in stock_id_arg.split(",")]
    
    from config import STOCK_CONFIGS, FINMIND_TOKEN, DB_CONFIG
    return list(STOCK_CONFIGS.keys())

def get_db_stock_info(conn):
    """取得所有 WSE/OTC 股票的基本資訊（供非 87 支時使用）"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT stock_id FROM stock_info WHERE type IN ('twse', 'otc') ORDER BY stock_id"
        )
        return [row[0] for row in cur.fetchall()]




# ──────────────────────────────────────────────
# institutional_investors_buy_sell（三大法人）
# ──────────────────────────────────────────────
def fetch_institutional_investors_buy_sell(start_date: str, end_date: str, delay: float, force: bool, stock_ids: list = None):
    logger.info(f"\n=== [institutional_investors_buy_sell] 開始（{len(stock_ids) if stock_ids else '全市場'}）===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_INSTITUTIONAL_INVESTORS)
        targets = stock_ids if stock_ids else get_db_stock_info(conn)
        
        # [v3] 批次預載安全起始日，取代原本迴圈內的逐筆查詢
        latest_dates = get_all_safe_starts(conn, "institutional_investors_buy_sell")
        
        total_rows = 0
        skipped = 0

        for i, sid in enumerate(targets, 1):
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
                    safe_bigint(r.get("buy")),
                    str(r.get("name", ""))[:50],
                    safe_bigint(r.get("sell")),
                )
                for r in data
            ]
            total_rows += bulk_upsert(
                conn, UPSERT_INSTITUTIONAL_INVESTORS, rows,
                "(%s::date, %s, %s, %s, %s)",
            )
            if i % 100 == 0:
                logger.info(f"  進度：{i}/{len(targets)}，累計 {total_rows} 筆（略過已最新：{skipped} 支）")

    finally:
        conn.close()
    logger.info(f"=== [institutional_investors_buy_sell] 完成，共寫入 {total_rows} 筆（略過：{skipped} 支）===")


# ──────────────────────────────────────────────
# margin_purchase_short_sale（融資融券）
# ──────────────────────────────────────────────
def fetch_margin_purchase_short_sale(start_date: str, end_date: str, delay: float, force: bool, stock_ids: list = None):
    logger.info(f"\n=== [margin_purchase_short_sale] 開始（{len(stock_ids) if stock_ids else '全市場'}）===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_MARGIN_PURCHASE)
        targets = stock_ids if stock_ids else get_db_stock_info(conn)
        
        # [v3] 批次預載安全起始日
        latest_dates = get_all_safe_starts(conn, "margin_purchase_short_sale")
        
        total_rows = 0
        skipped = 0

        for i, sid in enumerate(targets, 1):
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
                    safe_bigint(r.get("MarginPurchaseLimit")),
                    safe_int(r.get("MarginPurchaseSell")),
                    safe_int(r.get("MarginPurchaseTodayBalance")),
                    safe_int(r.get("MarginPurchaseYesterdayBalance")),
                    str(r.get("Note", "") or ""),
                    safe_int(r.get("OffsetLoanAndShort")),
                    safe_int(r.get("ShortSaleBuy")),
                    safe_int(r.get("ShortSaleCashRepayment")),
                    safe_bigint(r.get("ShortSaleLimit")),
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
                logger.info(f"  進度：{i}/{len(targets)}，累計 {total_rows} 筆（略過已最新：{skipped} 支）")

    finally:
        conn.close()
    logger.info(f"=== [margin_purchase_short_sale] 完成，共寫入 {total_rows} 筆（略過：{skipped} 支）===")


# ──────────────────────────────────────────────
# shareholding（外資持股）
# ──────────────────────────────────────────────
def fetch_shareholding(start_date: str, end_date: str, delay: float, force: bool, stock_ids: list = None):
    logger.info(f"\n=== [shareholding] 開始（{len(stock_ids) if stock_ids else '全市場'}）===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_SHAREHOLDING)
        targets = stock_ids if stock_ids else get_db_stock_info(conn)
        
        # [v3] 批次預載安全起始日
        latest_dates = get_all_safe_starts(conn, "shareholding")
        
        total_rows = 0
        skipped = 0

        for i, sid in enumerate(targets, 1):
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
                    safe_bigint(r.get("ForeignInvestmentRemainingShares")),
                    safe_bigint(r.get("ForeignInvestmentShares")),
                    safe_float(r.get("ForeignInvestmentRemainRatio")),
                    safe_float(r.get("ForeignInvestmentSharesRatio")),
                    safe_float(r.get("ForeignInvestmentUpperLimitRatio")),
                    safe_float(r.get("ChineseInvestmentUpperLimitRatio")),
                    safe_bigint(r.get("NumberOfSharesIssued")),
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
                logger.info(f"  進度：{i}/{len(stock_ids)}，累計 {total_rows} 筆（略過已最新：{skipped} 支）")

    finally:
        conn.close()
    logger.info(f"=== [shareholding] 完成，共寫入 {total_rows} 筆（略過：{skipped} 支）===")


# ──────────────────────────────────────────────
# CLI 主程式
# ──────────────────────────────────────────────
TABLE_FUNCS = {
    "institutional_investors_buy_sell": fetch_institutional_investors_buy_sell,
    "margin_purchase_short_sale":       fetch_margin_purchase_short_sale,
    "shareholding":                     fetch_shareholding,
}


def parse_args():
    parser = argparse.ArgumentParser(description="FinMind 籌碼面資料抓取工具")
    parser.add_argument(
        "--tables", nargs="+",
        choices=list(TABLE_FUNCS.keys()) + ["all"],
        default=["all"],
        help="要抓取的資料表（預設 all）",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help="開始日期 YYYY-MM-DD（增量模式下為首次抓取的起始日，預設 2001-01-01）",
    )
    parser.add_argument("--end", default=DEFAULT_END, help="結束日期 YYYY-MM-DD（預設今天）")
    parser.add_argument(
        "--delay", type=float, default=1.2,
        help="每次 API 請求後的等待秒數（預設 1.2）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重抓：忽略 DB 已有資料，從 --start 開始重新覆蓋",
    )
    parser.add_argument("--stock-id", default=None, help="指定股票代號（多個請用逗號隔開）")
    return parser.parse_args()


def main():
    args = parse_args()
    tables = list(TABLE_FUNCS.keys()) if "all" in args.tables else args.tables

    conn = get_db_conn()
    
    # 決定目標股票
    if args.stock_id:
        target_stocks = [s.strip() for s in args.stock_id.split(",")]
    else:
        # 如果沒指定，預設先抓 87 支重點股
        from config import STOCK_CONFIGS, FINMIND_TOKEN, DB_CONFIG
        target_stocks = list(STOCK_CONFIGS.keys())
        
    mode = "強制重抓" if args.force else "增量模式（自動跳過已最新資料）"
    logger.info(f"抓取資料表：{tables}")
    logger.info(f"模式：{mode}，目標股票數：{len(target_stocks)}")
    logger.info(f"日期區間：{args.start} ~ {args.end}")
    logger.info(f"請求間隔：{args.delay} 秒")
    logger.info(f"執行模式：{mode}")

    conn = get_db_conn()
    target_stocks = get_target_stock_ids(conn, args.stock_id)
    
    for table in tables:
        try:
            TABLE_FUNCS[table](args.start, args.end, args.delay, args.force, target_stocks)
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
        except psycopg2.OperationalError as e:
            logger.error(f"PostgreSQL 連線失敗：{e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"[{table}] 未預期錯誤：{e}")
            raise


if __name__ == "__main__":
    main()
