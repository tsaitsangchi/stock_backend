import argparse
import time
import json
import sys
import logging
from pathlib import Path
from datetime import date, datetime

# ── sys.path 自我修復 ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_chip_data.py v3.1 — 籌碼面核心資料（可觀察性監控版）
================================================================================
v3.1 重大改進：
  ★ 整合 fetch_log v3.1：每一支股票、每一資料表（三大法人、融資融券、借券）均記錄抓取狀態。
  ★ 效能監控：精準追蹤每一請求的 API 耗時（duration_ms）。
  ★ 狀態追蹤：支援 success, failed, no_new_data, skipped 等標準化狀態。
  ★ 結構規範化：整合 FailureLogger 與原子性 commit 機制，確保數據完整性。

執行範例（常規）：
    python scripts/fetchers/fetch_chip_data.py                # 抓取 160 支核心標的的所有籌碼資料
    python scripts/fetchers/fetch_chip_data.py --stock-id 2330 # 僅抓取台積電
    python scripts/fetchers/fetch_chip_data.py --tables shareholding # 僅抓取借券資料

執行範例（強制重抓）：
    python scripts/fetchers/fetch_chip_data.py --stock-id 2330 --force --tables all
    python scripts/fetchers/fetch_chip_data.py --tables all --force
"""

from core.finmind_client import finmind_get, get_request_stats
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    get_all_safe_starts,
    resolve_start_cached,
    safe_int,
    safe_float,
    safe_date,
    FailureLogger,
    map_rows_safe,
    commit_per_stock_per_day,
    dedup_rows,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATASET_START_DATES = {
    "institutional_investors_buy_sell": "2005-01-01",
    "margin_purchase_short_sale":       "2001-01-01",
    "shareholding":                     "2004-02-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")
_CLI_ARGS_STR = " ".join(sys.argv)

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None):
    """v3.1 標準化日誌寫入"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (
                    run_ts, table_name, stock_id, status, rows_inserted, 
                    fetch_date_from, fetch_date_to, duration_ms, error_message, cli_args
                ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (table_name, stock_id, status, rows_inserted, fetch_date_from, fetch_date_to, duration_ms, error_message, _CLI_ARGS_STR))
        conn.commit()
    except Exception as e:
        logger.warning(f"無法寫入 fetch_log: {e}")

# ──────────────────────────────────────────────
# DDL & SQL
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
"""

UPSERT_INSTITUTIONAL_INVESTORS = """
INSERT INTO institutional_investors_buy_sell (date, stock_id, buy, name, sell)
VALUES %s 
ON CONFLICT (date, stock_id, name) DO UPDATE SET
    buy = EXCLUDED.buy, 
    sell = EXCLUDED.sell;
"""

UPSERT_MARGIN_PURCHASE = """
INSERT INTO margin_purchase_short_sale (
    date, stock_id, margin_purchase_buy, margin_purchase_cash_repayment, 
    margin_purchase_limit, margin_purchase_sell, margin_purchase_today_balance, 
    margin_purchase_yesterday_balance, note, offset_loan_and_short,
    short_sale_buy, short_sale_cash_repayment, short_sale_limit,
    short_sale_sell, short_sale_today_balance, short_sale_yesterday_balance
) VALUES %s 
ON CONFLICT (date, stock_id) DO UPDATE SET
    margin_purchase_today_balance = EXCLUDED.margin_purchase_today_balance,
    short_sale_today_balance = EXCLUDED.short_sale_today_balance;
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
    foreign_investment_shares_ratio = EXCLUDED.foreign_investment_shares_ratio,
    number_of_shares_issued = EXCLUDED.number_of_shares_issued;
"""

# ──────────────────────────────────────────────
# Mappers
# ──────────────────────────────────────────────
def map_inst(r: dict) -> tuple:
    return (r["date"], r["stock_id"], safe_int(r.get("buy")), str(r.get("name", ""))[:50], safe_int(r.get("sell")))

def map_margin(r: dict) -> tuple:
    f = lambda k: safe_int(r.get(k))
    return (
        r["date"], r["stock_id"], f("MarginPurchaseBuy"), f("MarginPurchaseCashRepayment"),
        f("MarginPurchaseLimit"), f("MarginPurchaseSell"), f("MarginPurchaseTodayBalance"),
        f("MarginPurchaseYesterdayBalance"), str(r.get("Note", "") or ""), f("OffsetLoanAndShort"),
        f("ShortSaleBuy"), f("ShortSaleCashRepayment"), f("ShortSaleLimit"),
        f("ShortSaleSell"), f("ShortSaleTodayBalance"), f("ShortSaleYesterdayBalance")
    )

def map_share(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"], str(r.get("stock_name", "") or "")[:50],
        str(r.get("InternationalCode", "") or "")[:20],
        safe_int(r.get("ForeignInvestmentRemainingShares")), safe_int(r.get("ForeignInvestmentShares")),
        safe_float(r.get("ForeignInvestmentRemainRatio")), safe_float(r.get("ForeignInvestmentSharesRatio")),
        safe_float(r.get("ForeignInvestmentUpperLimitRatio")), safe_float(r.get("ChineseInvestmentUpperLimitRatio")),
        safe_int(r.get("NumberOfSharesIssued")), safe_date(r.get("RecentlyDeclareDate")),
        str(r.get("note", "") or "")
    )

def fetch_per_stock_task(
    conn, dataset_name: str, table_name: str, ddl: str, upsert_sql: str,
    template: str, mapper, stock_ids: list, start_date: str, end_date: str, force: bool
):
    logger.info(f"=== [{table_name}] 開始（{len(stock_ids)} 支）===")
    ensure_ddl(conn, ddl)
    flog = FailureLogger(table_name, db_conn=conn)
    latest_dates = get_all_safe_starts(conn, table_name)
    
    total_rows = 0
    for sid in stock_ids:
        actual_start = resolve_start_cached(sid, latest_dates, start_date, DATASET_START_DATES[table_name], force)
        if not actual_start:
            _write_fetch_log(conn, table_name, sid, "skipped", error_message="up_to_date")
            continue

        start_time = time.time()
        try:
            data = finmind_get(dataset_name, {"data_id": sid, "start_date": actual_start, "end_date": end_date})
            duration_ms = int((time.time() - start_time) * 1000)
            
            if not data:
                _write_fetch_log(conn, table_name, sid, "no_new_data", fetch_date_from=actual_start, fetch_date_to=end_date, duration_ms=duration_ms)
                continue
            
            rows = map_rows_safe(mapper, data, label=f"{table_name}/{sid}")
            
            if table_name == "institutional_investors_buy_sell":
                rows = dedup_rows(rows, (0, 1, 3))
            else:
                rows = dedup_rows(rows, (0, 1))

            results = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table_name, failure_logger=flog)
            n = sum(results.values())
            total_rows += n
            _write_fetch_log(conn, table_name, sid, "success", rows_inserted=n, fetch_date_from=actual_start, fetch_date_to=end_date, duration_ms=duration_ms)
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            flog.record(stock_id=sid, error=str(e))
            _write_fetch_log(conn, table_name, sid, "failed", fetch_date_from=actual_start, fetch_date_to=end_date, duration_ms=duration_ms, error_message=str(e))

    flog.summary()
    logger.info(f"=== [{table_name}] 完成，共寫入 {total_rows} 筆 ===\n")


def main():
    parser = argparse.ArgumentParser(description="FinMind 籌碼面資料抓取工具 v3.1")
    parser.add_argument("--tables", nargs="+", choices=["institutional_investors_buy_sell", "margin_purchase_short_sale", "shareholding", "all"], default=["all"])
    parser.add_argument("--start", default="2001-01-01")
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stock-id", default=None)
    args = parser.parse_args()

    tables = ["institutional_investors_buy_sell", "margin_purchase_short_sale", "shareholding"] if "all" in args.tables else args.tables
    
    conn = get_db_conn()
    try:
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            from core.db_utils import get_db_stock_ids
            stock_ids = get_db_stock_ids(conn)

        configs = {
            "institutional_investors_buy_sell": ("TaiwanStockInstitutionalInvestorsBuySell", DDL_INSTITUTIONAL_INVESTORS, UPSERT_INSTITUTIONAL_INVESTORS, "(%s::date,%s,%s,%s,%s)", map_inst),
            "margin_purchase_short_sale":       ("TaiwanStockMarginPurchaseShortSale", DDL_MARGIN_PURCHASE, UPSERT_MARGIN_PURCHASE, "(%s::date,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", map_margin),
            "shareholding":                     ("TaiwanStockShareholding", DDL_SHAREHOLDING, UPSERT_SHAREHOLDING, "(%s::date,%s,%s,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s,%s::date,%s)", map_share),
        }

        for table in tables:
            ds, ddl, upsert, tmpl, mapper = configs[table]
            fetch_per_stock_task(conn, ds, table, ddl, upsert, tmpl, mapper, stock_ids, args.start, args.end, args.force)
            
    finally:
        conn.close()
        get_request_stats().summary()

if __name__ == "__main__":
    main()