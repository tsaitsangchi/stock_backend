"""
fetch_chip_data.py v3.2 — 籌碼面核心資料（核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 使用 `commit_per_stock_per_day` 進行雙層粒度原子寫入，確保極致的資料完整性。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

執行範例：
    # 常規全量/增量抓取 (根據資料庫 stocks 表中 fetch_chip=True 的核心標的)
    python scripts/fetchers/fetch_chip_data.py
    
    # 針對特定資料表抓取
    python scripts/fetchers/fetch_chip_data.py --tables shareholding
    
    # 重試近 7 天失敗目標
    python scripts/fetchers/fetch_chip_data.py --retry-failed 7
    
    # 補足近 30 天無紀錄目標
    python scripts/fetchers/fetch_chip_data.py --gap-fill 30
    
    # 針對特定個股抓取特定資料
    python scripts/fetchers/fetch_chip_data.py --stock-id 2330,2454 --tables margin_purchase_short_sale
    
    # 強制重新抓取特定個股資料（無視增量檢查）
    python scripts/fetchers/fetch_chip_data.py --stock-id 2330 --force --tables all
"""

from __future__ import annotations

import argparse
import time
import sys
import logging
from collections import defaultdict
from pathlib import Path
from datetime import date, datetime

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
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
    get_core_stocks_from_db,
    DDL_FETCH_LOG
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

# ──────────────────────────────────────────────
# 日誌與 SQL
# ──────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")

def _write_fetch_log(conn, table_name, stock_id, status, fetch_mode="per_stock", rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None):
    """v3.2 標準化日誌寫入，失敗不影響主流程"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (
                    run_ts, table_name, stock_id, fetch_mode, status, rows_inserted, 
                    fetch_date_from, fetch_date_to, duration_ms, error_message, cli_args
                ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (table_name, stock_id, fetch_mode, status, rows_inserted, fetch_date_from, fetch_date_to, duration_ms, error_message, _CLI_ARGS_STR))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"無法寫入 fetch_log: {e}")

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

# ──────────────────────────────────────────────
# Fetcher Logic
# ──────────────────────────────────────────────
def fetch_per_stock_task(
    conn, dataset_name: str, table_name: str, ddl: str, upsert_sql: str,
    template: str, mapper, stock_ids: list, start_date: str, end_date: str, 
    delay: float, force: bool, fetch_mode_override: str | None = None
):
    logger.info(f"=== [{table_name}] 開始（{len(stock_ids)} 支）===")
    ensure_ddl(conn, ddl)
    conn.commit()
    
    flog = FailureLogger(table_name, db_conn=conn)
    latest_dates = get_all_safe_starts(conn, table_name)
    fetch_mode = fetch_mode_override or "per_stock"
    total_rows = 0
    
    for sid in stock_ids:
        actual_start = resolve_start_cached(sid, latest_dates, start_date, DATASET_START_DATES[table_name], force)
        if not actual_start:
            _write_fetch_log(conn, table_name, sid, "skipped", fetch_mode=fetch_mode, error_message="up_to_date")
            continue

        start_time = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset=dataset_name, 
                params={"data_id": sid, "start_date": actual_start, "end_date": end_date},
                delay=delay,
                raise_on_error=True
            )
            duration_ms = int((time.time() - start_time) * 1000)
            
            if not data:
                _write_fetch_log(conn, table_name, sid, "no_new_data", fetch_mode=fetch_mode, fetch_date_from=actual_start, fetch_date_to=end_date, duration_ms=duration_ms)
                continue
            
            rows = map_rows_safe(mapper, data, label=f"{table_name}/{sid}")
            
            if table_name == "institutional_investors_buy_sell":
                rows = dedup_rows(rows, (0, 1, 3))
            else:
                rows = dedup_rows(rows, (0, 1))

            results = commit_per_stock_per_day(conn, upsert_sql, rows, template, label_prefix=table_name, failure_logger=flog)
            n = sum(results.values())
            total_rows += n
            _write_fetch_log(conn, table_name, sid, "success" if n > 0 else "partial", fetch_mode=fetch_mode, rows_inserted=n, fetch_date_from=actual_start, fetch_date_to=end_date, duration_ms=duration_ms)
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=actual_start, end_date=end_date)
            _write_fetch_log(conn, table_name, sid, "failed", fetch_mode=fetch_mode, fetch_date_from=actual_start, fetch_date_to=end_date, duration_ms=duration_ms, error_message=str(e))

    flog.summary()
    logger.info(f"=== [{table_name}] 完成，共寫入 {total_rows} 筆 ===\n")


# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int, target_tables: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    sql = """
    WITH recent AS (
        SELECT table_name, stock_id, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY table_name, stock_id ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = ANY(%s) AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT table_name, stock_id FROM recent WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (target_tables, str(days)))
            for tbl, sid in cur.fetchall():
                targets[tbl].append(sid)
    except Exception as e:
        logger.error(f"[retry-failed] 查詢失敗：{e}")
        return {}

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [retry-failed/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def query_gap_targets(conn, days: int, target_tables: list[str], all_stock_ids: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    for tbl in target_tables:
        sql = f"SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tbl, str(days), all_stock_ids))
                have_success = {row[0] for row in cur.fetchall()}
            missing = [sid for sid in all_stock_ids if sid not in have_success]
            targets[tbl].extend(missing)
        except Exception as e:
            logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [gap-fill/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def _config_for_table(table: str):
    configs = {
        "institutional_investors_buy_sell": ("TaiwanStockInstitutionalInvestorsBuySell", DDL_INSTITUTIONAL_INVESTORS, UPSERT_INSTITUTIONAL_INVESTORS, "(%s::date,%s,%s,%s,%s)", map_inst),
        "margin_purchase_short_sale":       ("TaiwanStockMarginPurchaseShortSale", DDL_MARGIN_PURCHASE, UPSERT_MARGIN_PURCHASE, "(%s::date,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", map_margin),
        "shareholding":                     ("TaiwanStockShareholding", DDL_SHAREHOLDING, UPSERT_SHAREHOLDING, "(%s::date,%s,%s,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s,%s::date,%s)", map_share),
    }
    return configs.get(table)

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        cfg = _config_for_table(tbl)
        if cfg:
            ds, ddl, upsert, tmpl, mapper = cfg
            fetch_per_stock_task(conn, ds, tbl, ddl, upsert, tmpl, mapper, sids, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FinMind 籌碼面資料抓取工具 v3.2")
    parser.add_argument("--tables", nargs="+", choices=list(DATASET_START_DATES.keys()) + ["all"], default=["all"])
    parser.add_argument("--start", default="2001-01-01")
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stock-id", default=None)
    parser.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    parser.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = parser.parse_args()

    tables = list(DATASET_START_DATES.keys()) if "all" in args.tables else args.tables
    
    conn = get_db_conn()
    try:
        _ensure_fetch_log_table(conn)

        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            # v3.1 改用 DB 動態配置與開關
            stock_configs = get_core_stocks_from_db(conn)
            stock_ids = [sid for sid, cfg in stock_configs.items() if cfg.get("fetch_chip", True)]
            logger.info(f"從資料庫讀取到 {len(stock_ids)} 支核心股票 (已過濾 fetch_chip=True)")

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, tables)
            if targets: _run_targeted(conn, targets, args, fetch_mode="retry")
            else: logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, tables, stock_ids)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        for table in tables:
            cfg = _config_for_table(table)
            if cfg:
                ds, ddl, upsert, tmpl, mapper = cfg
                fetch_per_stock_task(conn, ds, table, ddl, upsert, tmpl, mapper, stock_ids, args.start, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        get_request_stats().summary()

if __name__ == "__main__":
    main()