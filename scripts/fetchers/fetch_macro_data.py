"""
fetch_macro_data.py — 總經與產業特徵（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 使用 `commit_per_stock_per_day` 進行雙層粒度原子寫入，確保極致的資料完整性。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

支援資料表：
  · interest_rate  (各國利率：FED, BOJ, ECB, PBOC)
  · exchange_rate  (台灣匯率：USD, JPY, EUR)
  · bond_yield     (公債殖利率：US10Y, US2Y)

執行範例（常規）：
    # 抓取所有總經數據（預設起始 2010-01-01）
    python scripts/fetchers/fetch_macro_data.py

    # 僅抓取匯率與債券殖利率
    python scripts/fetchers/fetch_macro_data.py --tables exchange_rate bond_yield

    # 強制重抓指定國家貨幣的匯率
    python scripts/fetchers/fetch_macro_data.py --tables exchange_rate --ids USD,JPY --force

    # 抓取特定時間段的債券殖利率
    python scripts/fetchers/fetch_macro_data.py --tables bond_yield --start 2024-01-01 --end 2024-05-01

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的目標
    python scripts/fetchers/fetch_macro_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python scripts/fetchers/fetch_macro_data.py --gap-fill 30
"""

from __future__ import annotations

import sys
import logging
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta

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
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 預設標的設定
DEFAULT_INTEREST = ["FED", "BOJ", "ECB", "PBOC"]
DEFAULT_EXCHANGE = ["USD", "JPY", "EUR"]
DEFAULT_BOND_MAP = {
    "US10Y": "United States 10-Year",
    "US2Y": "United States 2-Year"
}

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

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None, fetch_mode="per_stock"):
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

DDL_MACRO = """
CREATE TABLE IF NOT EXISTS interest_rate (
    date DATE, 
    country VARCHAR(50), 
    full_country_name VARCHAR(100), 
    interest_rate NUMERIC(20,2), 
    PRIMARY KEY (date, country)
);
CREATE TABLE IF NOT EXISTS exchange_rate (
    date DATE, 
    currency VARCHAR(50), 
    cash_buy NUMERIC(20,4), 
    cash_sell NUMERIC(20,4), 
    spot_buy NUMERIC(20,4), 
    spot_sell NUMERIC(20,4), 
    PRIMARY KEY (date, currency)
);
CREATE TABLE IF NOT EXISTS bond_yield (
    date DATE, 
    bond_id VARCHAR(50), 
    value NUMERIC(10,4), 
    PRIMARY KEY (date, bond_id)
);
"""

UPSERT_INTEREST = """
INSERT INTO interest_rate (date, country, full_country_name, interest_rate) 
VALUES %s 
ON CONFLICT (date, country) DO UPDATE SET interest_rate = EXCLUDED.interest_rate;
"""

UPSERT_EXCHANGE = """
INSERT INTO exchange_rate (date, currency, cash_buy, cash_sell, spot_buy, spot_sell) 
VALUES %s 
ON CONFLICT (date, currency) DO UPDATE SET 
    cash_buy = EXCLUDED.cash_buy,
    cash_sell = EXCLUDED.cash_sell,
    spot_buy = EXCLUDED.spot_buy,
    spot_sell = EXCLUDED.spot_sell;
"""

UPSERT_BOND = """
INSERT INTO bond_yield (date, bond_id, value) 
VALUES %s 
ON CONFLICT (date, bond_id) DO UPDATE SET value = EXCLUDED.value;
"""

# ──────────────────────────────────────────────
# Fetchers
# ──────────────────────────────────────────────
def fetch_interest_rate(conn, start, end, delay, force, target_ids=None, fetch_mode_override=None):
    logger.info("=== [interest_rate] 開始 ===")
    ensure_ddl(conn, DDL_MACRO)
    latest = get_all_safe_starts(conn, "interest_rate", key_col="country")
    flog = FailureLogger("interest_rate", db_conn=conn)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    
    countries = target_ids if target_ids else DEFAULT_INTEREST
    for country in countries:
        if country not in DEFAULT_INTEREST: continue
        s = resolve_start_cached(country, latest, start, "2000-01-01", force)
        if not s:
            _write_fetch_log(conn, table_name="interest_rate", stock_id=country, status="skipped", error_message="up_to_date", fetch_mode=fetch_mode)
            continue
            
        t0 = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset="InterestRate", 
                params={"data_id": country, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            duration_ms = int((time.time() - t0) * 1000)
            
            if data:
                rows = [(r["date"], r["country"], r.get("full_country_name", "")[:100], safe_float(r.get("interest_rate"))) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_INTEREST, rows, "(%s, %s, %s, %s::numeric)", stock_index=1, date_index=0, label_prefix="interest", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, "interest_rate", country, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration_ms, fetch_mode=fetch_mode)
            else:
                _write_fetch_log(conn, "interest_rate", country, "no_new_data", duration_ms=duration_ms, fetch_mode=fetch_mode)
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            flog.record(stock_id=country, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(conn, "interest_rate", country, "failed", duration_ms=duration_ms, error_message=str(e), fetch_mode=fetch_mode)

    logger.info(f"  [interest_rate] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_exchange_rate(conn, start, end, delay, force, target_ids=None, fetch_mode_override=None):
    logger.info("=== [exchange_rate] 開始 ===")
    ensure_ddl(conn, DDL_MACRO)
    latest = get_all_safe_starts(conn, "exchange_rate", key_col="currency")
    flog = FailureLogger("exchange_rate", db_conn=conn)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    
    currencies = target_ids if target_ids else DEFAULT_EXCHANGE
    for curr in currencies:
        if curr not in DEFAULT_EXCHANGE: continue
        s = resolve_start_cached(curr, latest, start, "2000-01-01", force)
        if not s:
            _write_fetch_log(conn, table_name="exchange_rate", stock_id=curr, status="skipped", error_message="up_to_date", fetch_mode=fetch_mode)
            continue
            
        t0 = time.time()
        try:
            data = finmind_get(
                dataset="TaiwanExchangeRate", 
                params={"data_id": curr, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            duration_ms = int((time.time() - t0) * 1000)
            
            if data:
                rows = [(r["date"], r["currency"], safe_float(r.get("cash_buy")), safe_float(r.get("cash_sell")), safe_float(r.get("spot_buy")), safe_float(r.get("spot_sell"))) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_EXCHANGE, rows, "(%s, %s, %s::numeric, %s::numeric, %s::numeric, %s::numeric)", stock_index=1, date_index=0, label_prefix="exchange", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, "exchange_rate", curr, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration_ms, fetch_mode=fetch_mode)
            else:
                _write_fetch_log(conn, "exchange_rate", curr, "no_new_data", duration_ms=duration_ms, fetch_mode=fetch_mode)
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            flog.record(stock_id=curr, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(conn, "exchange_rate", curr, "failed", duration_ms=duration_ms, error_message=str(e), fetch_mode=fetch_mode)

    logger.info(f"  [exchange_rate] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_bond_yield(conn, start, end, delay, force, target_ids=None, fetch_mode_override=None):
    logger.info("=== [bond_yield] 開始 ===")
    ensure_ddl(conn, DDL_MACRO)
    latest = get_all_safe_starts(conn, "bond_yield", key_col="bond_id")
    flog = FailureLogger("bond_yield", db_conn=conn)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    
    ids_to_fetch = target_ids if target_ids else list(DEFAULT_BOND_MAP.keys())
    
    for sid in ids_to_fetch:
        bid = DEFAULT_BOND_MAP.get(sid)
        if not bid: continue
        
        s = resolve_start_cached(sid, latest, start, "2000-01-01", force)
        if not s:
            _write_fetch_log(conn, table_name="bond_yield", stock_id=sid, status="skipped", error_message="up_to_date", fetch_mode=fetch_mode)
            continue
            
        t0 = time.time()
        try:
            data = finmind_get(
                dataset="GovernmentBondsYield", 
                params={"data_id": bid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            duration_ms = int((time.time() - t0) * 1000)
            
            if data:
                rows = [(r["date"], sid, safe_float(r["value"])) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_BOND, rows, "(%s, %s, %s::numeric)", stock_index=1, date_index=0, label_prefix="bond", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, "bond_yield", sid, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration_ms, fetch_mode=fetch_mode)
            else:
                _write_fetch_log(conn, "bond_yield", sid, "no_new_data", duration_ms=duration_ms, fetch_mode=fetch_mode)
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(conn, "bond_yield", sid, "failed", duration_ms=duration_ms, error_message=str(e), fetch_mode=fetch_mode)

    logger.info(f"  [bond_yield] 總共寫入 {total_rows} 筆")
    flog.summary()


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

def query_gap_targets(conn, days: int, target_tables: list[str], target_ids_map: dict[str, list[str]]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    for tbl in target_tables:
        all_ids = target_ids_map.get(tbl, [])
        if not all_ids:
            continue
            
        sql = f"SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tbl, str(days), all_ids))
                have_success = {row[0] for row in cur.fetchall()}
            missing = [sid for sid in all_ids if sid not in have_success]
            targets[tbl].extend(missing)
        except Exception as e:
            logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [gap-fill/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        
        if tbl == "interest_rate":
            fetch_interest_rate(conn, args.start, args.end, args.delay, force=True, target_ids=sids, fetch_mode_override=fetch_mode)
        elif tbl == "exchange_rate":
            fetch_exchange_rate(conn, args.start, args.end, args.delay, force=True, target_ids=sids, fetch_mode_override=fetch_mode)
        elif tbl == "bond_yield":
            fetch_bond_yield(conn, args.start, args.end, args.delay, force=True, target_ids=sids, fetch_mode_override=fetch_mode)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["interest_rate", "exchange_rate", "bond_yield", "all"], default=["all"])
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--ids", help="指定要抓取的代號 (例如 USD,JPY,US10Y)，多筆用逗號分隔")
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["interest_rate", "exchange_rate", "bond_yield"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        
        # 決定全域的目標 ID 清單
        target_ids_list = [s.strip() for s in args.ids.split(",")] if args.ids else None
        target_ids_map = {
            "interest_rate": target_ids_list if target_ids_list else DEFAULT_INTEREST,
            "exchange_rate": target_ids_list if target_ids_list else DEFAULT_EXCHANGE,
            "bond_yield": target_ids_list if target_ids_list else list(DEFAULT_BOND_MAP.keys())
        }

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
            targets = query_gap_targets(conn, args.gap_fill, tables, target_ids_map)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        if "interest_rate" in tables:
            fetch_interest_rate(conn, args.start, args.end, args.delay, args.force, target_ids_list)
        if "exchange_rate" in tables:
            fetch_exchange_rate(conn, args.start, args.end, args.delay, args.force, target_ids_list)
        if "bond_yield" in tables:
            fetch_bond_yield(conn, args.start, args.end, args.delay, args.force, target_ids_list)
            
    finally:
        conn.close()
        logger.info("全部完成")
        # 於程式完全結束時，印出統一的 FinMind RequestStats 報表
        get_request_stats().summary()

if __name__ == "__main__":
    main()