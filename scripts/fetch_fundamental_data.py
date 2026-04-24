"""
fetch_fundamental_data.py  v2.0（API 用量優化版）
從 FinMind API 抓取基本面資料並寫入 PostgreSQL：
  - financial_statements ← TaiwanStockFinancialStatements (綜合損益表, Free)
  - balance_sheet        ← TaiwanStockBalanceSheet        (資產負債表, Free)
  - month_revenue        ← TaiwanStockMonthRevenue        (月營收,     Free)
  - dividend             ← TaiwanStockDividend            (股利政策,   Free)

v2.0 優化重點（三層節省 API 用量）：

  ① DB 最新日期批次預載（全表共用，效能優化）
       原：每支股票 SELECT MAX(date)（N 次 SQL → 約 2,000 次/資料集）
       後：一條 GROUP BY SQL 預載所有股票最新日期 → dict 查快取

  ② financial_statements + balance_sheet 合併迴圈
       兩者同為季報、相同股票清單、相同 API 模式
       原：兩個獨立迴圈各跑 ~2,000 次 API = 4,000 次
       後：同一迴圈內依序抓取，省去重複遍歷與 DB 連線開銷
       --per-table 旗標可還原為分別抓取

  ③ month_revenue 批次模式（最大節省）
       增量更新時大多數股票 start_date 相同（上個月月初）
       → 不帶 data_id 的全市場請求，依 --chunk-days 分段
       原：~2,000 次 API   後：少數幾次批次請求
       --per-stock 旗標可還原為逐支請求

  ④ dividend 智慧跳過（原有邏輯強化）
       保留逐支請求（股利資料量少、更新不規律）
       + 快取最新日期（省掉逐筆 SQL）
       + get_expected_latest_date() 判斷是否需要抓取

執行範例：
    # 增量更新（預設，API 用量最少）
    python fetch_fundamental_data.py

    # 只抓月營收 + 財報
    python fetch_fundamental_data.py --tables month_revenue financial_statements balance_sheet

    # 退回逐支模式（相容舊行為）
    python fetch_fundamental_data.py --per-stock --per-table

    # 強制重抓（建議搭配 --tables 限縮範圍）
    python fetch_fundamental_data.py --force --tables month_revenue

注意事項：
  - 財報公告截止日慣例（台灣法規）：
      Q1 (3/31) → 5/15  │ Q2 (6/30) → 8/14
      Q3 (9/30) → 11/14  │ Q4 (12/31) → 隔年 3/31
  - 月營收：次月 10 日前公告，最新可用月為「上個月」
  - 股利：股東會集中 5~6 月，保守取「去年底」為最新穩定基準
  - dedup_rows() 在寫入前去除 API 偶爾回傳的重複列，
    避免 PostgreSQL ON CONFLICT DO UPDATE CardinalityViolation 錯誤。
  - 批次模式回傳全市場資料，程式自動過濾僅保留 stock_info 內的股票。
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
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
    "financial_statements": "1990-03-01",
    "balance_sheet":        "2011-12-01",
    "month_revenue":        "2002-02-01",
    "dividend":             "2005-05-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1990-03-01"

# 批次模式：每次請求最多涵蓋幾天（避免單次回傳筆數過多）
DEFAULT_CHUNK_DAYS      = 60    # 月營收用，60 天 = 約 2 個月的全市場資料
DEFAULT_BATCH_THRESHOLD = 20    # 同起始日超過此支數才啟用批次

# ──────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────
DDL_FINANCIAL_STATEMENTS = """
CREATE TABLE IF NOT EXISTS financial_statements (
    date        DATE,
    stock_id    VARCHAR(50),
    type        VARCHAR(50),
    value       NUMERIC(20,4),
    origin_name VARCHAR(100),
    PRIMARY KEY (date, stock_id, type)
);
CREATE INDEX IF NOT EXISTS idx_financial_statements_stock_id ON financial_statements (stock_id);
"""

DDL_BALANCE_SHEET = """
CREATE TABLE IF NOT EXISTS balance_sheet (
    date        DATE,
    stock_id    VARCHAR(50),
    type        VARCHAR(50),
    value       NUMERIC(20,4),
    origin_name VARCHAR(100),
    PRIMARY KEY (date, stock_id, type)
);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_stock_id ON balance_sheet (stock_id);
"""

DDL_MONTH_REVENUE = """
CREATE TABLE IF NOT EXISTS month_revenue (
    date          DATE,
    stock_id      VARCHAR(50),
    country       VARCHAR(50),
    revenue       BIGINT,
    revenue_month INTEGER,
    revenue_year  INTEGER,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_month_revenue_stock_id ON month_revenue (stock_id);
"""

DDL_DIVIDEND = """
CREATE TABLE IF NOT EXISTS dividend (
    date                                       DATE,
    stock_id                                   VARCHAR(50),
    year                                       VARCHAR(4),
    stock_earnings_distribution                NUMERIC(20,4),
    stock_statutory_surplus                    NUMERIC(20,4),
    stock_ex_dividend_trading_date             DATE,
    total_employee_stock_dividend              NUMERIC(20,4),
    total_employee_stock_dividend_amount       NUMERIC(20,4),
    ratio_of_employee_stock_dividend_of_total  NUMERIC(20,4),
    ratio_of_employee_stock_dividend           NUMERIC(20,4),
    cash_earnings_distribution                 NUMERIC(20,4),
    cash_statutory_surplus                     NUMERIC(20,4),
    cash_ex_dividend_trading_date              DATE,
    cash_dividend_payment_date                 DATE,
    total_employee_cash_dividend               NUMERIC(20,4),
    total_number_of_cash_capital_increase      NUMERIC(20,4),
    cash_increase_subscription_rate            NUMERIC(20,4),
    cash_increase_subscription_price           NUMERIC(20,4),
    remuneration_of_directors_and_supervisors  NUMERIC(20,4),
    participate_distribution_of_total_shares   NUMERIC(20,4),
    announcement_date                          DATE,
    announcement_time                          TIMESTAMP,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_dividend_stock_id ON dividend (stock_id);
"""

MIGRATE_DIVIDEND_COLUMNS = """
DO $$
DECLARE
    col TEXT;
    cols TEXT[] := ARRAY[
        'stock_earnings_distribution',
        'stock_statutory_surplus',
        'total_employee_stock_dividend',
        'total_employee_stock_dividend_amount',
        'ratio_of_employee_stock_dividend_of_total',
        'ratio_of_employee_stock_dividend',
        'cash_earnings_distribution',
        'cash_statutory_surplus',
        'total_employee_cash_dividend',
        'total_number_of_cash_capital_increase',
        'cash_increase_subscription_rate',
        'cash_increase_subscription_price',
        'remuneration_of_directors_and_supervisors',
        'participate_distribution_of_total_shares'
    ];
BEGIN
    FOREACH col IN ARRAY cols LOOP
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'dividend'
              AND column_name = col
              AND numeric_precision < 20
        ) THEN
            EXECUTE format(
                'ALTER TABLE dividend ALTER COLUMN %I TYPE NUMERIC(20,4)', col
            );
            RAISE NOTICE 'Migrated dividend.% to NUMERIC(20,4)', col;
        END IF;
    END LOOP;
END
$$;
"""

# ──────────────────────────────────────────────
# Upsert SQL
# ──────────────────────────────────────────────
UPSERT_FINANCIAL_STATEMENTS = """
INSERT INTO financial_statements (date, stock_id, type, value, origin_name)
VALUES %s
ON CONFLICT (date, stock_id, type) DO UPDATE SET
    value       = EXCLUDED.value,
    origin_name = EXCLUDED.origin_name;
"""

UPSERT_BALANCE_SHEET = """
INSERT INTO balance_sheet (date, stock_id, type, value, origin_name)
VALUES %s
ON CONFLICT (date, stock_id, type) DO UPDATE SET
    value       = EXCLUDED.value,
    origin_name = EXCLUDED.origin_name;
"""

UPSERT_MONTH_REVENUE = """
INSERT INTO month_revenue (date, stock_id, country, revenue, revenue_month, revenue_year)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    country       = EXCLUDED.country,
    revenue       = EXCLUDED.revenue,
    revenue_month = EXCLUDED.revenue_month,
    revenue_year  = EXCLUDED.revenue_year;
"""

UPSERT_DIVIDEND = """
INSERT INTO dividend (
    date, stock_id, year,
    stock_earnings_distribution, stock_statutory_surplus,
    stock_ex_dividend_trading_date,
    total_employee_stock_dividend, total_employee_stock_dividend_amount,
    ratio_of_employee_stock_dividend_of_total, ratio_of_employee_stock_dividend,
    cash_earnings_distribution, cash_statutory_surplus,
    cash_ex_dividend_trading_date, cash_dividend_payment_date,
    total_employee_cash_dividend, total_number_of_cash_capital_increase,
    cash_increase_subscription_rate, cash_increase_subscription_price,
    remuneration_of_directors_and_supervisors,
    participate_distribution_of_total_shares,
    announcement_date, announcement_time
) VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    year                                      = EXCLUDED.year,
    stock_earnings_distribution               = EXCLUDED.stock_earnings_distribution,
    stock_statutory_surplus                   = EXCLUDED.stock_statutory_surplus,
    stock_ex_dividend_trading_date            = EXCLUDED.stock_ex_dividend_trading_date,
    total_employee_stock_dividend             = EXCLUDED.total_employee_stock_dividend,
    total_employee_stock_dividend_amount      = EXCLUDED.total_employee_stock_dividend_amount,
    ratio_of_employee_stock_dividend_of_total = EXCLUDED.ratio_of_employee_stock_dividend_of_total,
    ratio_of_employee_stock_dividend          = EXCLUDED.ratio_of_employee_stock_dividend,
    cash_earnings_distribution                = EXCLUDED.cash_earnings_distribution,
    cash_statutory_surplus                    = EXCLUDED.cash_statutory_surplus,
    cash_ex_dividend_trading_date             = EXCLUDED.cash_ex_dividend_trading_date,
    cash_dividend_payment_date                = EXCLUDED.cash_dividend_payment_date,
    total_employee_cash_dividend              = EXCLUDED.total_employee_cash_dividend,
    total_number_of_cash_capital_increase     = EXCLUDED.total_number_of_cash_capital_increase,
    cash_increase_subscription_rate           = EXCLUDED.cash_increase_subscription_rate,
    cash_increase_subscription_price          = EXCLUDED.cash_increase_subscription_price,
    remuneration_of_directors_and_supervisors = EXCLUDED.remuneration_of_directors_and_supervisors,
    participate_distribution_of_total_shares  = EXCLUDED.participate_distribution_of_total_shares,
    announcement_date                         = EXCLUDED.announcement_date,
    announcement_time                         = EXCLUDED.announcement_time;
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


def safe_timestamp(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NONE", "NAN", ""):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def dedup_rows(rows: list, key_indices: tuple) -> list:
    """
    依指定欄位索引去除重複列（後出現的覆蓋先出現的）。
    解決 FinMind 偶爾回傳同批資料含重複 PK 導致 CardinalityViolation 的問題。
    """
    seen = {}
    for row in rows:
        key = tuple(row[i] for i in key_indices)
        seen[key] = row
    return list(seen.values())


def get_expected_latest_date(dataset_key: str) -> str:
    """
    依各資料集的公告週期，推算目前「合理可取得的最新資料日期」。
    DB 只要已有此日期，就無需再發 API 請求。
    """
    today = date.today()

    if dataset_key in ("financial_statements", "balance_sheet"):
        quarters = [
            (5,  15, 3,  31, False),
            (8,  14, 6,  30, False),
            (11, 14, 9,  30, False),
            (3,  31, 12, 31, True),
        ]
        latest_qend = None
        for dm, dd, qm, qd, cross_year in quarters:
            deadline = date(today.year, dm, dd)
            qend = date(today.year - 1 if cross_year else today.year, qm, qd)
            if today >= deadline:
                if latest_qend is None or qend > latest_qend:
                    latest_qend = qend
        if latest_qend is None:
            latest_qend = date(today.year - 1, 9, 30)
        return latest_qend.strftime("%Y-%m-%d")

    elif dataset_key == "month_revenue":
        first_of_this_month = today.replace(day=1)
        last_month_first = (first_of_this_month - timedelta(days=1)).replace(day=1)
        return last_month_first.strftime("%Y-%m-%d")

    elif dataset_key == "dividend":
        if today >= date(today.year, 7, 1):
            return date(today.year, 12, 31).strftime("%Y-%m-%d")
        else:
            return date(today.year - 1, 12, 31).strftime("%Y-%m-%d")

    return today.strftime("%Y-%m-%d")


def wait_until_next_hour():
    """
    當遇到 402 (Payment Required) 錯誤時，代表 API 用量達上限。
    通常 FinMind 會在整點重置配額，因此等待至下一整點。
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
    """
    通用 FinMind API 請求（v2 — 強化超時與重試）。

    修正項目：
      ① timeout 改為 (connect=15, read=120)：connect 過慢直接中斷，不等滿 120 秒
      ② 指數退避（5s → 20s → 60s）：避免連線風暴，給 API 服務器恢復時間
      ③ 最多重試 5 次（逾時類錯誤）/ 3 次（其他 HTTP 錯誤）
      ④ ConnectTimeout 與 ReadTimeout 分開處理，記錄更清晰
    """
    import random
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}

    MAX_RETRIES   = 5       # 連線逾時最多重試次數
    BASE_WAIT     = 5.0     # 指數退避基礎秒數

    while True:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    FINMIND_API_URL,
                    headers=headers,
                    params=req_params,
                    timeout=(15, 120),   # (connect_timeout, read_timeout)
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

            except requests.exceptions.ConnectTimeout:
                # 連線階段超時（DNS 慢或 API 服務器暫時不回應）
                wait_sec = BASE_WAIT * (3 ** (attempt - 1)) + random.uniform(0, 3)
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"[{dataset}] 連線逾時（第 {attempt}/{MAX_RETRIES} 次），"
                        f"{wait_sec:.0f} 秒後重試…"
                    )
                    time.sleep(wait_sec)
                else:
                    logger.error(f"[{dataset}] 連線逾時，已重試 {MAX_RETRIES} 次，跳過")
                    return []

            except requests.exceptions.ReadTimeout:
                # 讀取階段超時（資料量過大或 API 服務器處理慢）
                wait_sec = BASE_WAIT * (2 ** (attempt - 1)) + random.uniform(0, 2)
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"[{dataset}] 讀取逾時（第 {attempt}/{MAX_RETRIES} 次），"
                        f"{wait_sec:.0f} 秒後重試…"
                    )
                    time.sleep(wait_sec)
                else:
                    logger.error(f"[{dataset}] 讀取逾時，已重試 {MAX_RETRIES} 次，跳過")
                    return []

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
                    wait_sec = BASE_WAIT * (2 ** (attempt - 1))
                    logger.warning(f"[{dataset}] HTTP {code} 錯誤，{wait_sec:.0f} 秒後重試：{http_err}")
                    if attempt < 3:
                        time.sleep(wait_sec)
                    else:
                        logger.error(f"[{dataset}] HTTP 錯誤，已重試 3 次，跳過")
                        return []

            except Exception as exc:
                wait_sec = BASE_WAIT * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.warning(
                    f"[{dataset}] 第 {attempt}/{MAX_RETRIES} 次請求失敗：{exc}，"
                    f"{wait_sec:.0f} 秒後重試"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait_sec)
                else:
                    logger.error(f"[{dataset}] 已重試 {MAX_RETRIES} 次，跳過")
                    return []
        else:
            break


# ──────────────────────────────────────────────
# DB 工具
# ──────────────────────────────────────────────
def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_ddl(conn, *ddls):
    with conn.cursor() as cur:
        for ddl in ddls:
            cur.execute(ddl)
    conn.commit()


def migrate_dividend_columns(conn):
    with conn.cursor() as cur:
        cur.execute(MIGRATE_DIVIDEND_COLUMNS)
    conn.commit()
    logger.info("[dividend] 欄位精度確認完畢（已確保 NUMERIC(20,4)）")


def bulk_upsert(conn, sql: str, rows: list, template: str, page_size: int = 2000) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, sql, rows, template=template, page_size=page_size
        )
    conn.commit()
    return len(rows)


def get_all_stock_ids(conn, config_only: bool = False) -> list:
    if config_only:
        from config import STOCK_CONFIGS
        return list(STOCK_CONFIGS.keys())
    from config import STOCK_CONFIGS
    return list(STOCK_CONFIGS.keys())


# ① DB 最新日期批次預載（一條 SQL 取代 N 次逐筆查詢）
def get_all_latest_dates(conn, table: str) -> dict:
    """
    一次查出指定資料表所有股票的最新日期。
    回傳 { stock_id: "YYYY-MM-DD" }

    原本每支股票 SELECT MAX(date) WHERE stock_id=?（N 次 SQL）
    現改為一條 GROUP BY SQL，預載後全程用 dict 查快取。
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT stock_id, MAX(date) FROM {table} GROUP BY stock_id")
        return {
            row[0]: row[1].strftime("%Y-%m-%d")
            for row in cur.fetchall()
            if row[1] is not None
        }


def resolve_start_cached(
    stock_id: str, latest_dates: dict,
    global_start: str, dataset_key: str, force: bool
):
    """
    從預載快取（latest_dates）決定起始日，不再逐筆查 DB。
    回傳 None 表示此股票已是最新，不需抓取。
    """
    earliest = DATASET_START_DATES[dataset_key]
    effective_start = max(global_start, earliest)

    if force:
        return effective_start

    latest = latest_dates.get(str(stock_id))

    if latest is None:
        return effective_start

    expected_latest = get_expected_latest_date(dataset_key)
    if latest >= expected_latest:
        return None  # 已達最新預期，跳過

    next_day = (
        datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    return max(next_day, earliest)


# ──────────────────────────────────────────────
# Row mappers
# ──────────────────────────────────────────────
def map_fin_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        str(r.get("type", ""))[:50],
        safe_float(r.get("value")),
        str(r.get("origin_name", ""))[:100],
    )


def map_rev_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        str(r.get("country", ""))[:50],
        safe_int(r.get("revenue")),
        safe_int(r.get("revenue_month")),
        safe_int(r.get("revenue_year")),
    )


def map_div_row(r: dict) -> tuple:
    ann_time_raw = r.get("AnnouncementTime", "")
    ann_date_raw = r.get("AnnouncementDate", "")
    if ann_time_raw and ann_date_raw:
        s = str(ann_time_raw).strip()
        ann_time = safe_timestamp(
            f"{ann_date_raw} {s}" if len(s) <= 8 else s
        )
    else:
        ann_time = None
    return (
        r["date"], r["stock_id"],
        str(r.get("year", ""))[:4],
        safe_float(r.get("StockEarningsDistribution")),
        safe_float(r.get("StockStatutorySurplus")),
        safe_date(r.get("StockExDividendTradingDate")),
        safe_float(r.get("TotalEmployeeStockDividend")),
        safe_float(r.get("TotalEmployeeStockDividendAmount")),
        safe_float(r.get("RatioOfEmployeeStockDividendOfTotal")),
        safe_float(r.get("RatioOfEmployeeStockDividend")),
        safe_float(r.get("CashEarningsDistribution")),
        safe_float(r.get("CashStatutorySurplus")),
        safe_date(r.get("CashExDividendTradingDate")),
        safe_date(r.get("CashDividendPaymentDate")),
        safe_float(r.get("TotalEmployeeCashDividend")),
        safe_float(r.get("TotalNumberOfCashCapitalIncrease")),
        safe_float(r.get("CashIncreaseSubscriptionRate")),
        safe_float(
            r.get("CashIncreaseSubscriptionPrice")
            or r.get("CashIncreaseSubscriptionpRrice")
        ),
        safe_float(r.get("RemunerationOfDirectorsAndSupervisors")),
        safe_float(r.get("ParticipateDistributionOfTotalShares")),
        safe_date(ann_date_raw),
        ann_time,
    )


# ──────────────────────────────────────────────
# ② financial_statements + balance_sheet 合併迴圈
# ──────────────────────────────────────────────
def fetch_quarterly_combined(
    start_date: str, end_date: str, delay: float, force: bool,
    tables: list, stock_ids: list,
):
    """
    財報合併迴圈：financial_statements 與 balance_sheet 共享同一支股票迴圈。
    兩者同為季報，更新週期完全一致，合併後省掉重複遍歷與 DB 連線開銷。

    tables 可為：
      ["financial_statements", "balance_sheet"]  兩者皆抓
      ["financial_statements"]                   只抓損益表
      ["balance_sheet"]                          只抓資負表
    """
    do_fin = "financial_statements" in tables
    do_bs  = "balance_sheet"        in tables

    label = " + ".join(
        t for t, flag in [("financial_statements", do_fin), ("balance_sheet", do_bs)] if flag
    )
    logger.info(f"=== [{label}] 合併迴圈開始 ===")

    expected = get_expected_latest_date("financial_statements")
    logger.info(f"  預期最新季報日期：{expected}")

    conn = get_db_conn()
    try:
        if do_fin: ensure_ddl(conn, DDL_FINANCIAL_STATEMENTS)
        if do_bs:  ensure_ddl(conn, DDL_BALANCE_SHEET)

        logger.info(f"共 {len(stock_ids)} 支股票待處理")

        # ① 批次預載兩張表最新日期
        latest_fin = get_all_latest_dates(conn, "financial_statements") if do_fin else {}
        latest_bs  = get_all_latest_dates(conn, "balance_sheet")        if do_bs  else {}

        total_fin = total_bs = 0
        skipped_fin = skipped_bs = 0

        for i, sid in enumerate(stock_ids, 1):

            # ── financial_statements ──
            if do_fin:
                s = resolve_start_cached(
                    sid, latest_fin, start_date, "financial_statements", force
                )
                if s is None:
                    skipped_fin += 1
                else:
                    data = finmind_get(
                        "TaiwanStockFinancialStatements",
                        {"data_id": sid, "start_date": s, "end_date": end_date},
                        delay,
                    )
                    if data:
                        rows = dedup_rows([map_fin_row(r) for r in data], (0, 1, 2))
                        total_fin += bulk_upsert(
                            conn, UPSERT_FINANCIAL_STATEMENTS, rows,
                            "(%s::date, %s, %s, %s::numeric, %s)",
                        )

            # ── balance_sheet ──
            if do_bs:
                s = resolve_start_cached(
                    sid, latest_bs, start_date, "balance_sheet", force
                )
                if s is None:
                    skipped_bs += 1
                else:
                    data = finmind_get(
                        "TaiwanStockBalanceSheet",
                        {"data_id": sid, "start_date": s, "end_date": end_date},
                        delay,
                    )
                    if data:
                        rows = dedup_rows([map_fin_row(r) for r in data], (0, 1, 2))
                        total_bs += bulk_upsert(
                            conn, UPSERT_BALANCE_SHEET, rows,
                            "(%s::date, %s, %s, %s::numeric, %s)",
                        )

            if i % 100 == 0:
                logger.info(
                    f"  進度：{i}/{len(stock_ids)}"
                    + (f"  fin 略過：{skipped_fin}" if do_fin else "")
                    + (f"  bs 略過：{skipped_bs}"  if do_bs  else "")
                )

    finally:
        conn.close()

    if do_fin:
        logger.info(
            f"=== [financial_statements] 完成，"
            f"共寫入 {total_fin} 筆（略過已最新：{skipped_fin} 支）==="
        )
    if do_bs:
        logger.info(
            f"=== [balance_sheet] 完成，"
            f"共寫入 {total_bs} 筆（略過已最新：{skipped_bs} 支）==="
        )


# ──────────────────────────────────────────────
# ③ month_revenue 批次模式
# ──────────────────────────────────────────────
def fetch_month_revenue_batch(
    start_date: str, end_date: str, delay: float, force: bool,
    chunk_days: int, batch_threshold: int, valid_stock_ids: set,
    conn,
):
    """
    批次模式：同一起始日的股票合併為不帶 data_id 的全市場請求。
    增量更新時絕大多數股票 start_date 相同（上個月月初），
    可將 ~2,000 次請求壓縮到少數幾次批次請求。
    """
    latest_dates = get_all_latest_dates(conn, "month_revenue")

    stock_starts: dict[str, str] = {}
    skipped = 0
    for sid in sorted(valid_stock_ids):
        s = resolve_start_cached(sid, latest_dates, start_date, "month_revenue", force)
        if s is None:
            skipped += 1
        else:
            stock_starts[sid] = s

    logger.info(
        f"[month_revenue] 需抓取：{len(stock_starts)} 支，已最新略過：{skipped} 支"
    )
    if not stock_starts:
        return 0

    # 按 actual_start 分組
    groups: dict[str, list] = defaultdict(list)
    for sid, s in stock_starts.items():
        groups[s].append(sid)

    logger.info(
        f"[month_revenue] 共 {len(groups)} 個不同起始日"
        f"（批次閾值：>= {batch_threshold} 支才合併）"
    )

    total_api = 0
    total_rows = 0

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        if len(sids) >= batch_threshold:
            # 批次模式：不帶 data_id，按 chunk_days 分段
            seg_start = group_start
            seg_end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            chunk_rows_all = []

            while True:
                seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                if seg_start_dt > seg_end_dt:
                    break
                seg_end = min(
                    (seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"),
                    end_date,
                )
                logger.info(
                    f"  [month_revenue] 批次請求 {seg_start}~{seg_end}"
                    f"（{len(sids)} 支，不帶 data_id）"
                )
                data = finmind_get(
                    "TaiwanStockMonthRevenue",
                    {"start_date": seg_start, "end_date": seg_end},
                    delay,
                )
                total_api += 1
                filtered = [r for r in data if r.get("stock_id") in sids_set]
                chunk_rows_all.extend(filtered)
                seg_start = (
                    datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")

            if chunk_rows_all:
                rows = dedup_rows([map_rev_row(r) for r in chunk_rows_all], (0, 1))
                total_rows += bulk_upsert(
                    conn, UPSERT_MONTH_REVENUE, rows,
                    "(%s::date, %s, %s, %s, %s, %s)",
                )
                logger.info(f"  [month_revenue] 批次寫入 {len(rows)} 筆")
        else:
            # 小尾巴：逐支請求
            for sid in sids:
                data = finmind_get(
                    "TaiwanStockMonthRevenue",
                    {"data_id": sid, "start_date": group_start, "end_date": end_date},
                    delay,
                )
                total_api += 1
                if not data:
                    continue
                rows = dedup_rows([map_rev_row(r) for r in data], (0, 1))
                total_rows += bulk_upsert(
                    conn, UPSERT_MONTH_REVENUE, rows,
                    "(%s::date, %s, %s, %s, %s, %s)",
                )

    logger.info(
        f"=== [month_revenue] 完成  "
        f"API 請求：{total_api} 次  寫入：{total_rows} 筆 ==="
    )
    return total_rows


def fetch_month_revenue_per_stock(
    start_date: str, end_date: str, delay: float, force: bool,
    valid_stock_ids: list, conn,
):
    """逐支模式（--per-stock 時使用）"""
    latest_dates = get_all_latest_dates(conn, "month_revenue")
    total_rows = skipped = 0

    for i, sid in enumerate(sorted(valid_stock_ids), 1):
        s = resolve_start_cached(sid, latest_dates, start_date, "month_revenue", force)
        if s is None:
            skipped += 1
            continue
        data = finmind_get(
            "TaiwanStockMonthRevenue",
            {"data_id": sid, "start_date": s, "end_date": end_date},
            delay,
        )
        if data:
            rows = dedup_rows([map_rev_row(r) for r in data], (0, 1))
            total_rows += bulk_upsert(
                conn, UPSERT_MONTH_REVENUE, rows,
                "(%s::date, %s, %s, %s, %s, %s)",
            )
        if i % 100 == 0:
            logger.info(
                f"  [month_revenue] 進度：{i}/{len(valid_stock_ids)}"
                f"  略過：{skipped}"
            )

    logger.info(
        f"=== [month_revenue] 完成，"
        f"共寫入 {total_rows} 筆（略過已最新：{skipped} 支）==="
    )


def fetch_month_revenue(
    start_date: str, end_date: str, delay: float, force: bool,
    per_stock: bool, chunk_days: int, batch_threshold: int,
    stock_ids: list,
):
    expected = get_expected_latest_date("month_revenue")
    logger.info(f"=== [month_revenue] 開始抓取（預期最新：{expected}）===")

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_MONTH_REVENUE)
        logger.info(f"共 {len(stock_ids)} 支股票待處理")

        if per_stock:
            fetch_month_revenue_per_stock(
                start_date, end_date, delay, force, stock_ids, conn
            )
        else:
            fetch_month_revenue_batch(
                start_date, end_date, delay, force,
                chunk_days, batch_threshold, set(stock_ids), conn,
            )
    finally:
        conn.close()


# ──────────────────────────────────────────────
# ④ dividend（逐支 + 快取最新日期）
# ──────────────────────────────────────────────
def fetch_dividend(start_date: str, end_date: str, delay: float, force: bool, stock_ids: list):
    expected = get_expected_latest_date("dividend")
    logger.info(f"=== [dividend] 開始抓取（預期最新：{expected}）===")

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_DIVIDEND)
        migrate_dividend_columns(conn)
        logger.info(f"共 {len(stock_ids)} 支股票待處理")

        # ① 批次預載最新日期
        latest_dates = get_all_latest_dates(conn, "dividend")

        total_rows = skipped = 0

        for i, sid in enumerate(stock_ids, 1):
            s = resolve_start_cached(sid, latest_dates, start_date, "dividend", force)
            if s is None:
                skipped += 1
                continue

            data = finmind_get(
                "TaiwanStockDividend",
                {"data_id": sid, "start_date": s, "end_date": end_date},
                delay,
            )
            if not data:
                continue

            rows = dedup_rows([map_div_row(r) for r in data], (0, 1))
            total_rows += bulk_upsert(
                conn, UPSERT_DIVIDEND, rows,
                (
                    "(%s::date, %s, %s,"
                    " %s::numeric, %s::numeric, %s::date,"
                    " %s::numeric, %s::numeric, %s::numeric, %s::numeric,"
                    " %s::numeric, %s::numeric, %s::date, %s::date,"
                    " %s::numeric, %s::numeric, %s::numeric, %s::numeric,"
                    " %s::numeric, %s::numeric,"
                    " %s::date, %s::timestamp)"
                ),
            )
            if i % 100 == 0:
                logger.info(
                    f"  [dividend] 進度：{i}/{len(stock_ids)}"
                    f"  累計 {total_rows} 筆（略過已最新：{skipped} 支）"
                )

    finally:
        conn.close()
    logger.info(
        f"=== [dividend] 完成，"
        f"共寫入 {total_rows} 筆（略過已最新：{skipped} 支）==="
    )


# ──────────────────────────────────────────────
# CLI 主程式
# ──────────────────────────────────────────────
ALL_TABLES = ["financial_statements", "balance_sheet", "month_revenue", "dividend"]
# 季報合併群組（可以一起處理）
QUARTERLY_TABLES = {"financial_statements", "balance_sheet"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FinMind 基本面資料抓取工具 v2.0（API 用量優化版）"
    )
    parser.add_argument(
        "--tables", nargs="+",
        choices=ALL_TABLES + ["all"],
        default=["all"],
        help="要抓取的資料表（預設 all）",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help="開始日期 YYYY-MM-DD（預設 1990-03-01）",
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help="結束日期 YYYY-MM-DD（預設今天）",
    )
    parser.add_argument(
        "--delay", type=float, default=1.2,
        help="每次 API 請求後等待秒數（預設 1.2）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重抓：忽略 DB 已有資料，從 --start 開始重新覆蓋",
    )
    # ② 合併迴圈控制
    parser.add_argument(
        "--per-table", action="store_true",
        help="停用 financial_statements + balance_sheet 合併迴圈，改為分別抓取",
    )
    # ③ 批次模式控制
    parser.add_argument(
        "--per-stock", action="store_true",
        help="month_revenue 退回逐支請求（相容舊行為）",
    )
    parser.add_argument(
        "--batch-threshold", type=int, default=DEFAULT_BATCH_THRESHOLD,
        help=f"批次閾值：同起始日超過此支數才合併（預設 {DEFAULT_BATCH_THRESHOLD}）",
    )
    parser.add_argument(
        "--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS,
        help=f"批次每段天數（預設 {DEFAULT_CHUNK_DAYS}）",
    )
    parser.add_argument("--stock-id", default=None, help="指定股票代號（多個請用逗號隔開）")
    return parser.parse_args()


def main():
    args = parse_args()
    tables = ALL_TABLES if "all" in args.tables else args.tables

    conn = get_db_conn()
    target_stocks = get_target_stock_ids(conn, args.stock_id)
    conn.close()

    mode_str = "強制重抓" if args.force else "增量模式（依公告週期智慧跳過）"
    logger.info(f"目標股票數：{len(target_stocks)}")
    logger.info(f"抓取資料表：{tables}")
    logger.info(f"日期區間：{args.start} ~ {args.end}")
    logger.info(f"請求間隔：{args.delay} 秒")
    logger.info(f"執行模式：{mode_str}")
    logger.info(
        f"財報合併迴圈：{'關閉 (--per-table)' if args.per_table else '啟用'}  "
        f"月營收批次：{'關閉 (--per-stock)' if args.per_stock else '啟用'}"
    )

    for key in tables:
        logger.info(f"  [{key}] 預期最新日期 = {get_expected_latest_date(key)}")

    try:
        # ── 季報：財報 + 資負表（合併或分別）──────────────
        quarterly = [t for t in tables if t in QUARTERLY_TABLES]
        if quarterly:
            fetch_quarterly_combined(
                args.start, args.end, args.delay, args.force, quarterly, target_stocks
            )

        # ── 月營收（批次或逐支）────────────────────────────
        if "month_revenue" in tables:
            fetch_month_revenue(
                args.start, args.end, args.delay, args.force,
                args.per_stock, args.chunk_days, args.batch_threshold,
                target_stocks
            )

        # ── 股利（逐支 + 快取）─────────────────────────────
        if "dividend" in tables:
            fetch_dividend(args.start, args.end, args.delay, args.force, target_stocks)

    except psycopg2.OperationalError as e:
        logger.error(f"PostgreSQL 連線失敗：{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"未預期錯誤：{e}")
        raise


if __name__ == "__main__":
    main()
