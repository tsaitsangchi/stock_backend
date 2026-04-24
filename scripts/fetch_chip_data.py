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
FINMIND_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ"
    ".muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"
)

# ======================
# PostgreSQL 連線設定
# ======================
DB_CONFIG = {
    "dbname": "stock",
    "user": "stock",
    "password": "stock",
    "host": "localhost",
    "port": "5432",
}

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
# 工具函式
# ──────────────────────────────────────────────
def safe_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NONE", "NAN", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def safe_int(val):
    f = safe_float(val)
    return int(f) if f is not None else None


def safe_bigint(val):
    f = safe_float(val)
    return int(f) if f is not None else None


def safe_date(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NONE", "NAN", ""):
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        return None


def wait_until_next_hour():
    """
    當遇到 402 (Payment Required) 錯誤時，代表 API 用量達上限。
    通常 FinMind 會在整點重備配額，因此等待至下一整點。
    """
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_sec = (next_hour - now).total_seconds() + 65
    logger.warning(
        f"API 用量達上限（402），等待至下一整點重置。"
        f"目前時間：{now.strftime('%H:%M:%S')}，"
        f"預計恢復：{next_hour.strftime('%H:%M:%S')}，"
        f"等待 {wait_sec:.0f} 秒…"
    )
    time.sleep(wait_sec)
    logger.info("等待結束，恢復請求。")


def finmind_get(dataset: str, params: dict, delay: float) -> list:
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}

    while True:
        for attempt in range(1, 4):
            try:
                resp = requests.get(
                    FINMIND_API_URL, headers=headers, params=req_params, timeout=(15, 120)
                )
                if resp.status_code == 402:
                    wait_until_next_hour()
                    break
                resp.raise_for_status()
                payload = resp.json()
                status = payload.get("status")
                if status == 402:
                    wait_until_next_hour()
                    break
                if status != 200:
                    logger.warning(
                        f"[{dataset}] status={status}, msg={payload.get('msg')}，跳過"
                    )
                    return []
                time.sleep(delay)
                return payload.get("data", [])

            except requests.HTTPError as http_err:
                code = http_err.response.status_code if http_err.response is not None else 0
                if code == 400:
                    logger.debug(
                        f"[{dataset}] 400 Bad Request，跳過 "
                        f"(data_id={params.get('data_id')}, start={params.get('start_date')})"
                    )
                    return []
                elif code == 402:
                    wait_until_next_hour()
                    break
                else:
                    logger.warning(f"[{dataset}] HTTP {code} 錯誤：{http_err}")
                    if attempt < 3:
                        time.sleep(delay * 3)
                    else:
                        logger.error(f"[{dataset}] 重試 3 次均失敗，跳過")
                        return []
            except Exception as exc:
                logger.warning(f"[{dataset}] 第 {attempt} 次請求失敗：{exc}")
                if attempt < 3:
                    time.sleep(delay * 3)
                else:
                    logger.error(f"[{dataset}] 重試 3 次均失敗，跳過")
                    return []
        else:
            break


def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_ddl(conn, ddl: str):
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def bulk_upsert(conn, sql: str, rows: list, template: str, page_size: int = 2000):
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, sql, rows, template=template, page_size=page_size
        )
    conn.commit()
    return len(rows)


def get_all_stock_ids(conn) -> list:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT stock_id FROM stock_info WHERE type IN ('twse', 'otc') ORDER BY stock_id"
        )
        return [row[0] for row in cur.fetchall()]


def get_latest_date(conn, table: str, stock_id: str):
    """查詢指定資料表中某支股票已有的最新日期，無資料回傳 None。"""
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(date) FROM {table} WHERE stock_id = %s", (stock_id,))
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
        return None


def resolve_start(conn, table: str, stock_id: str, global_start: str, dataset_key: str, force: bool):
    """
    決定實際抓取起始日：
      force=True  → 使用 global_start（強制重抓）
      force=False → DB 有資料則從最新日期+1天起抓；無資料則從 global_start 起抓
    回傳 None 表示此股票資料已是最新，不需抓取。
    """
    earliest = DATASET_START_DATES[dataset_key]
    effective_start = max(global_start, earliest)

    if force:
        return effective_start

    latest = get_latest_date(conn, table, stock_id)
    if latest is None:
        return effective_start

    next_day = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    if next_day > DEFAULT_END:
        return None  # 已是最新

    return max(next_day, earliest)


# ──────────────────────────────────────────────
# institutional_investors_buy_sell（三大法人）
# ──────────────────────────────────────────────
def fetch_institutional_investors_buy_sell(start_date: str, end_date: str, delay: float, force: bool):
    logger.info("=== [institutional_investors_buy_sell] 開始抓取 ===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_INSTITUTIONAL_INVESTORS)
        stock_ids = get_all_stock_ids(conn)
        logger.info(f"共 {len(stock_ids)} 支股票待處理")
        total_rows = 0
        skipped = 0

        for i, sid in enumerate(stock_ids, 1):
            actual_start = resolve_start(
                conn, "institutional_investors_buy_sell", sid,
                start_date, "institutional_investors_buy_sell", force
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
                logger.info(f"  進度：{i}/{len(stock_ids)}，累計 {total_rows} 筆（略過已最新：{skipped} 支）")

    finally:
        conn.close()
    logger.info(f"=== [institutional_investors_buy_sell] 完成，共寫入 {total_rows} 筆（略過：{skipped} 支）===")


# ──────────────────────────────────────────────
# margin_purchase_short_sale（融資融券）
# ──────────────────────────────────────────────
def fetch_margin_purchase_short_sale(start_date: str, end_date: str, delay: float, force: bool):
    logger.info("=== [margin_purchase_short_sale] 開始抓取 ===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_MARGIN_PURCHASE)
        stock_ids = get_all_stock_ids(conn)
        logger.info(f"共 {len(stock_ids)} 支股票待處理")
        total_rows = 0
        skipped = 0

        for i, sid in enumerate(stock_ids, 1):
            actual_start = resolve_start(
                conn, "margin_purchase_short_sale", sid,
                start_date, "margin_purchase_short_sale", force
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
                logger.info(f"  進度：{i}/{len(stock_ids)}，累計 {total_rows} 筆（略過已最新：{skipped} 支）")

    finally:
        conn.close()
    logger.info(f"=== [margin_purchase_short_sale] 完成，共寫入 {total_rows} 筆（略過：{skipped} 支）===")


# ──────────────────────────────────────────────
# shareholding（外資持股）
# ──────────────────────────────────────────────
def fetch_shareholding(start_date: str, end_date: str, delay: float, force: bool):
    logger.info("=== [shareholding] 開始抓取 ===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_SHAREHOLDING)
        stock_ids = get_all_stock_ids(conn)
        logger.info(f"共 {len(stock_ids)} 支股票待處理")
        total_rows = 0
        skipped = 0

        for i, sid in enumerate(stock_ids, 1):
            actual_start = resolve_start(
                conn, "shareholding", sid,
                start_date, "shareholding", force
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
    return parser.parse_args()


def main():
    args = parse_args()
    tables = list(TABLE_FUNCS.keys()) if "all" in args.tables else args.tables

    mode = "強制重抓" if args.force else "增量模式（自動跳過已最新資料）"
    logger.info(f"抓取資料表：{tables}")
    logger.info(f"日期區間：{args.start} ~ {args.end}")
    logger.info(f"請求間隔：{args.delay} 秒")
    logger.info(f"執行模式：{mode}")

    for table in tables:
        try:
            TABLE_FUNCS[table](args.start, args.end, args.delay, args.force)
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
