"""
fetch_fundamental_data.py  v2.2（逐支 commit 完整性版）
=========================================================
從 FinMind API 抓取基本面資料並寫入 PostgreSQL：
  - financial_statements ← TaiwanStockFinancialStatements
  - balance_sheet        ← TaiwanStockBalanceSheet
  - month_revenue        ← TaiwanStockMonthRevenue
  - dividend             ← TaiwanStockDividend

v2.2 改進：
  · safe_commit_rows()：每支股票 / 每組寫入後立即 commit，失敗 rollback。
  · 主迴圈以 try/except 包單支，單支失敗不影響其他股票。
  · fetch_quarterly_combined：fin + bs 一支股票結束才一起 commit（保證跨表一致）。
  · 失敗清單寫入 outputs/{table}_failed_{date}.json。
"""

import sys
import json
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import argparse
import logging
from collections import defaultdict
from datetime import date, timedelta, datetime

import psycopg2

from config import DB_CONFIG
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    get_all_latest_dates,
    safe_float,
    safe_int,
    safe_date,
    safe_timestamp,
    dedup_rows,
)
from core.finmind_client import finmind_get

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 各資料集最早可用日期
# ──────────────────────────────────────────────
DATASET_START_DATES = {
    "financial_statements": "1990-03-01",
    "balance_sheet":        "2011-12-01",
    "month_revenue":        "2002-02-01",
    "dividend":             "2005-05-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1990-03-01"

DEFAULT_CHUNK_DAYS      = 60
DEFAULT_BATCH_THRESHOLD = 20

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
# 逐支 commit 工具函式
# ──────────────────────────────────────────────
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
        logger.info(f"  失敗清單已寫入：{out}（{len(failures)} 筆）")
    except Exception as e:
        logger.warning(f"  寫入失敗清單時發生錯誤：{e}")


# ──────────────────────────────────────────────
# 輔助函式
# ──────────────────────────────────────────────
def get_expected_latest_date(dataset_key: str) -> str:
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


def migrate_dividend_columns(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(MIGRATE_DIVIDEND_COLUMNS)
    conn.commit()
    logger.info("[dividend] 欄位精度確認完畢（已確保 NUMERIC(20,4)）")


def get_target_stock_ids(conn, stock_id_arg: str = None) -> list:
    if stock_id_arg:
        return [s.strip() for s in stock_id_arg.split(",")]
    from config import STOCK_CONFIGS
    return list(STOCK_CONFIGS.keys())


def resolve_start_fundamental(
    stock_id: str,
    latest_dates: dict,
    global_start: str,
    dataset_key: str,
    force: bool,
) -> str | None:
    earliest = DATASET_START_DATES[dataset_key]
    effective_start = max(global_start, earliest)

    if force:
        return effective_start

    latest = latest_dates.get(str(stock_id))
    if latest is None:
        return effective_start

    expected_latest = get_expected_latest_date(dataset_key)
    if latest >= expected_latest:
        return None

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
# financial_statements + balance_sheet 合併迴圈
# 每支股票 fin + bs 都成功才一起 commit，確保跨表一致性
# ──────────────────────────────────────────────
def fetch_quarterly_combined(
    start_date: str, end_date: str, delay: float, force: bool,
    tables: list, stock_ids: list,
):
    do_fin = "financial_statements" in tables
    do_bs  = "balance_sheet"        in tables

    label = " + ".join(
        t for t, flag in [("financial_statements", do_fin), ("balance_sheet", do_bs)] if flag
    )
    logger.info(f"=== [{label}] 合併迴圈開始 ===")

    expected = get_expected_latest_date("financial_statements")
    logger.info(f"  預期最新季報日期：{expected}")

    conn = get_db_conn()
    failures_fin: list[dict] = []
    failures_bs:  list[dict] = []
    try:
        if do_fin:
            ensure_ddl(conn, DDL_FINANCIAL_STATEMENTS); conn.commit()
        if do_bs:
            ensure_ddl(conn, DDL_BALANCE_SHEET);        conn.commit()

        logger.info(f"共 {len(stock_ids)} 支股票待處理")

        latest_fin = get_all_latest_dates(conn, "financial_statements") if do_fin else {}
        latest_bs  = get_all_latest_dates(conn, "balance_sheet")        if do_bs  else {}

        total_fin = total_bs = 0
        skipped_fin = skipped_bs = 0

        for i, sid in enumerate(stock_ids, 1):
            # fin
            if do_fin:
                try:
                    s = resolve_start_fundamental(sid, latest_fin, start_date, "financial_statements", force)
                    if s is None:
                        skipped_fin += 1
                    else:
                        data = finmind_get(
                            "TaiwanStockFinancialStatements",
                            {"data_id": sid, "start_date": s, "end_date": end_date},
                            delay,
                            raise_on_error=True
                        )
                        if data:
                            rows = dedup_rows([map_fin_row(r) for r in data], (0, 1, 2))
                            n = safe_commit_rows(
                                conn, UPSERT_FINANCIAL_STATEMENTS, rows,
                                "(%s::date, %s, %s, %s::numeric, %s)",
                                label=f"financial_statements/{sid}"
                            )
                            total_fin += n
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    failures_fin.append({"stock_id": sid, "error": str(e)})
                    logger.error(f"  [financial_statements/{sid}] 失敗：{e}")

            # bs
            if do_bs:
                try:
                    s = resolve_start_fundamental(sid, latest_bs, start_date, "balance_sheet", force)
                    if s is None:
                        skipped_bs += 1
                    else:
                        data = finmind_get(
                            "TaiwanStockBalanceSheet",
                            {"data_id": sid, "start_date": s, "end_date": end_date},
                            delay,
                            raise_on_error=True
                        )
                        if data:
                            rows = dedup_rows([map_fin_row(r) for r in data], (0, 1, 2))
                            n = safe_commit_rows(
                                conn, UPSERT_BALANCE_SHEET, rows,
                                "(%s::date, %s, %s, %s::numeric, %s)",
                                label=f"balance_sheet/{sid}"
                            )
                            total_bs += n
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    failures_bs.append({"stock_id": sid, "error": str(e)})
                    logger.error(f"  [balance_sheet/{sid}] 失敗：{e}")

            if i % 100 == 0:
                logger.info(
                    f"  進度：{i}/{len(stock_ids)}"
                    + (f"  fin 略過：{skipped_fin}  失敗：{len(failures_fin)}" if do_fin else "")
                    + (f"  bs 略過：{skipped_bs}  失敗：{len(failures_bs)}"  if do_bs  else "")
                )

    finally:
        conn.close()

    if do_fin:
        dump_failures("financial_statements", failures_fin)
        logger.info(
            f"=== [financial_statements] 完成，"
            f"共寫入 {total_fin} 筆（略過：{skipped_fin}，失敗：{len(failures_fin)}）==="
        )
    if do_bs:
        dump_failures("balance_sheet", failures_bs)
        logger.info(
            f"=== [balance_sheet] 完成，"
            f"共寫入 {total_bs} 筆（略過：{skipped_bs}，失敗：{len(failures_bs)}）==="
        )


# ──────────────────────────────────────────────
# month_revenue 批次模式（每支股票 commit 一次）
# ──────────────────────────────────────────────
def fetch_month_revenue_batch(
    start_date: str, end_date: str, delay: float, force: bool,
    chunk_days: int, batch_threshold: int, valid_stock_ids: set,
    conn,
):
    latest_dates = get_all_latest_dates(conn, "month_revenue")

    stock_starts: dict[str, str] = {}
    skipped = 0
    for sid in sorted(valid_stock_ids):
        s = resolve_start_fundamental(sid, latest_dates, start_date, "month_revenue", force)
        if s is None:
            skipped += 1
        else:
            stock_starts[sid] = s

    logger.info(
        f"[month_revenue] 需抓取：{len(stock_starts)} 支，已最新略過：{skipped} 支"
    )
    if not stock_starts:
        return 0

    groups: dict[str, list] = defaultdict(list)
    for sid, s in stock_starts.items():
        groups[s].append(sid)

    logger.info(
        f"[month_revenue] 共 {len(groups)} 個不同起始日"
        f"（批次閾值：>= {batch_threshold} 支才合併）"
    )

    total_api = 0
    total_rows = 0
    failures: list[dict] = []

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        # ── 批次模式 ──
        if len(sids) >= batch_threshold:
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
                try:
                    data = finmind_get(
                        "TaiwanStockMonthRevenue",
                        {"start_date": seg_start, "end_date": seg_end},
                        delay,
                    )
                except Exception as e:
                    logger.error(f"  [month_revenue] {seg_start}~{seg_end} API 失敗：{e}")
                    failures.append({"chunk": f"{seg_start}~{seg_end}", "error": str(e)})
                    seg_start = (
                        datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
                    ).strftime("%Y-%m-%d")
                    continue

                total_api += 1
                filtered = [r for r in data if r.get("stock_id") in sids_set]
                chunk_rows_all.extend(filtered)
                seg_start = (
                    datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")

            if chunk_rows_all:
                # ── 逐支模式進行 commit ──
                rows_by_stock: dict[str, list] = defaultdict(list)
                for r in chunk_rows_all:
                    try:
                        rows_by_stock[r["stock_id"]].append(map_rev_row(r))
                    except Exception as e:
                        logger.warning(f"  [month_revenue/{r.get('stock_id')}] mapper 異常：{e}")

                for sid, s_rows in rows_by_stock.items():
                    try:
                        final_rows = dedup_rows(s_rows, (0, 1))
                        n = safe_commit_rows(
                            conn, UPSERT_MONTH_REVENUE, final_rows,
                            "(%s::date, %s, %s, %s, %s, %s)",
                            label=f"month_revenue/{sid}"
                        )
                        total_rows += n
                    except Exception as e:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        failures.append({"stock_id": sid, "error": str(e)})
                        logger.error(f"  [month_revenue/{sid}] 寫入失敗：{e}")

                logger.info(
                    f"  [month_revenue] 批次寫入完成（含 {len(rows_by_stock)} 支股票）"
                )
        # ── 逐支模式 ──
        else:
            for sid in sids:
                try:
                    data = finmind_get(
                        "TaiwanStockMonthRevenue",
                        {"data_id": sid, "start_date": group_start, "end_date": end_date},
                        delay,
                    )
                    total_api += 1
                    if not data:
                        continue
                    rows = dedup_rows([map_rev_row(r) for r in data], (0, 1))
                    n = safe_commit_rows(
                        conn, UPSERT_MONTH_REVENUE, rows,
                        "(%s::date, %s, %s, %s, %s, %s)",
                        label=f"month_revenue/{sid}"
                    )
                    total_rows += n
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    failures.append({"stock_id": sid, "error": str(e)})
                    logger.error(f"  [month_revenue/{sid}] 失敗：{e}")

    dump_failures("month_revenue", failures)
    logger.info(
        f"=== [month_revenue] 完成  "
        f"API 請求：{total_api} 次  寫入：{total_rows} 筆  失敗：{len(failures)} ==="
    )
    return total_rows


def fetch_month_revenue_per_stock(
    start_date: str, end_date: str, delay: float, force: bool,
    valid_stock_ids: list, conn,
):
    latest_dates = get_all_latest_dates(conn, "month_revenue")
    total_rows = skipped = 0
    failures: list[dict] = []

    for i, sid in enumerate(sorted(valid_stock_ids), 1):
        try:
            s = resolve_start_fundamental(sid, latest_dates, start_date, "month_revenue", force)
            if s is None:
                skipped += 1
                continue
            data = finmind_get(
                "TaiwanStockMonthRevenue",
                {"data_id": sid, "start_date": s, "end_date": end_date},
                delay,
                raise_on_error=True
            )
            if data:
                rows = dedup_rows([map_rev_row(r) for r in data], (0, 1))
                n = safe_commit_rows(
                    conn, UPSERT_MONTH_REVENUE, rows,
                    "(%s::date, %s, %s, %s, %s, %s)",
                    label=f"month_revenue/{sid}"
                )
                total_rows += n
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            failures.append({"stock_id": sid, "error": str(e)})
            logger.error(f"  [month_revenue/{sid}] 失敗：{e}")

        if i % 100 == 0:
            logger.info(
                f"  [month_revenue] 進度：{i}/{len(valid_stock_ids)}"
                f"  略過：{skipped}  失敗：{len(failures)}"
            )

    dump_failures("month_revenue", failures)
    logger.info(
        f"=== [month_revenue] 完成，"
        f"共寫入 {total_rows} 筆（略過：{skipped}，失敗：{len(failures)}）==="
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
        conn.commit()
        logger.info(f"共 {len(stock_ids)} 支股票待處理")

        if per_stock:
            fetch_month_revenue_per_stock(start_date, end_date, delay, force, stock_ids, conn)
        else:
            fetch_month_revenue_batch(
                start_date, end_date, delay, force,
                chunk_days, batch_threshold, set(stock_ids), conn,
            )
    finally:
        conn.close()


# ──────────────────────────────────────────────
# dividend（逐支 + 快取最新日期）
# ──────────────────────────────────────────────
def fetch_dividend(start_date: str, end_date: str, delay: float, force: bool, stock_ids: list):
    expected = get_expected_latest_date("dividend")
    logger.info(f"=== [dividend] 開始抓取（預期最新：{expected}）===")

    conn = get_db_conn()
    failures: list[dict] = []
    try:
        ensure_ddl(conn, DDL_DIVIDEND)
        migrate_dividend_columns(conn)
        logger.info(f"共 {len(stock_ids)} 支股票待處理")

        latest_dates = get_all_latest_dates(conn, "dividend")
        total_rows = skipped = 0

        for i, sid in enumerate(stock_ids, 1):
            try:
                s = resolve_start_fundamental(sid, latest_dates, start_date, "dividend", force)
                if s is None:
                    skipped += 1
                    continue

                data = finmind_get(
                    "TaiwanStockDividend",
                    {"data_id": sid, "start_date": s, "end_date": end_date},
                    delay,
                    raise_on_error=True
                )
                if not data:
                    continue

                rows = dedup_rows([map_div_row(r) for r in data], (0, 1))
                n = safe_commit_rows(
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
                    label=f"dividend/{sid}"
                )
                total_rows += n
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                failures.append({"stock_id": sid, "error": str(e)})
                logger.error(f"  [dividend/{sid}] 失敗：{e}")

            if i % 100 == 0:
                logger.info(
                    f"  [dividend] 進度：{i}/{len(stock_ids)}"
                    f"  累計 {total_rows} 筆（略過：{skipped}，失敗：{len(failures)}）"
                )

    finally:
        conn.close()
    dump_failures("dividend", failures)
    logger.info(
        f"=== [dividend] 完成，"
        f"共寫入 {total_rows} 筆（略過：{skipped}，失敗：{len(failures)}）==="
    )


# ──────────────────────────────────────────────
# CLI 主程式
# ──────────────────────────────────────────────
ALL_TABLES = ["financial_statements", "balance_sheet", "month_revenue", "dividend"]
QUARTERLY_TABLES = {"financial_statements", "balance_sheet"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FinMind 基本面資料抓取工具 v2.2（逐支 commit 完整性版）"
    )
    parser.add_argument(
        "--tables", nargs="+",
        choices=ALL_TABLES + ["all"],
        default=["all"],
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end",   default=DEFAULT_END)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--per-table", action="store_true",
                        help="停用 financial_statements + balance_sheet 合併迴圈")
    parser.add_argument("--per-stock", action="store_true",
                        help="month_revenue 退回逐支請求")
    parser.add_argument("--batch-threshold", type=int, default=DEFAULT_BATCH_THRESHOLD)
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--stock-id", default=None)
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
    logger.info(f"請求間隔：{args.delay} 秒（Token Bucket 速率限制）")
    logger.info(f"執行模式：{mode_str}")

    for key in tables:
        logger.info(f"  [{key}] 預期最新日期 = {get_expected_latest_date(key)}")

    try:
        quarterly = [t for t in tables if t in QUARTERLY_TABLES]
        if quarterly:
            try:
                fetch_quarterly_combined(
                    args.start, args.end, args.delay, args.force, quarterly, target_stocks
                )
            except Exception as e:
                logger.error(f"[quarterly_combined] 未預期錯誤：{e}")

        if "month_revenue" in tables:
            try:
                fetch_month_revenue(
                    args.start, args.end, args.delay, args.force,
                    args.per_stock, args.chunk_days, args.batch_threshold,
                    target_stocks
                )
            except Exception as e:
                logger.error(f"[month_revenue] 未預期錯誤：{e}")

        if "dividend" in tables:
            try:
                fetch_dividend(args.start, args.end, args.delay, args.force, target_stocks)
            except Exception as e:
                logger.error(f"[dividend] 未預期錯誤：{e}")

    except psycopg2.OperationalError as e:
        logger.error(f"PostgreSQL 連線失敗：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()