"""
fetch_sponsor_chip_data.py — 進階籌碼資料（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 自動讀取 `stocks` 表中 `fetch_chip=True` 的核心標的。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

執行範例（常規）：
    # 抓取所有進階籌碼資料 (大戶持股、八大行庫、期貨大額)
    python scripts/fetchers/fetch_sponsor_chip_data.py
    
    # 僅抓取八大行庫 (FinMind 限制每次只回傳單日資料，程式會自動逐日抓取)
    python scripts/fetchers/fetch_sponsor_chip_data.py --tables eight_banks

執行範例（強制重抓與指定標的）：
    # 僅抓取台積電與鴻海的大戶持股
    python scripts/fetchers/fetch_sponsor_chip_data.py --stock-id 2330,2317 --force --tables holding_shares_per
    
    # 強制重抓特定日期的八大行庫
    python scripts/fetchers/fetch_sponsor_chip_data.py --tables eight_banks --start 2024-01-01 --force

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的任務
    python scripts/fetchers/fetch_sponsor_chip_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料表
    python scripts/fetchers/fetch_sponsor_chip_data.py --gap-fill 30
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
    safe_int,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    write_fetch_log,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
    get_db_stock_ids,
    get_core_stocks_from_db,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATASET_START = {
    "holding_shares_per": "2014-01-01",
    "eight_banks":        "2021-01-01",
    "futures_large_oi":   "2010-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")
_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 日誌與 SQL
# ─────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")


DDL_HOLDING = """CREATE TABLE IF NOT EXISTS holding_shares_per (date DATE, stock_id VARCHAR(50), level VARCHAR(50), people BIGINT, percent NUMERIC(20,6), unit VARCHAR(100), PRIMARY KEY (date, stock_id, level));"""
DDL_EIGHT_BANKS = """CREATE TABLE IF NOT EXISTS eight_banks_buy_sell (date DATE, stock_id VARCHAR(50), buy BIGINT, sell BIGINT, PRIMARY KEY (date, stock_id));"""
DDL_FUTURES_LARGE_OI = """CREATE TABLE IF NOT EXISTS futures_large_oi (date DATE, contract_code VARCHAR(20), name VARCHAR(50), long_position BIGINT, long_position_over50 BIGINT, short_position BIGINT, short_position_over50 BIGINT, net_position BIGINT, market_total_oi BIGINT, PRIMARY KEY (date, contract_code, name));"""

UPSERT_HOLDING = """INSERT INTO holding_shares_per (date, stock_id, level, people, percent, unit) VALUES %s ON CONFLICT (date, stock_id, level) DO UPDATE SET people = EXCLUDED.people, percent = EXCLUDED.percent;"""
UPSERT_EIGHT_BANKS = """INSERT INTO eight_banks_buy_sell (date, stock_id, buy, sell) VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET buy = EXCLUDED.buy, sell = EXCLUDED.sell;"""
UPSERT_FUTURES_LARGE_OI = """INSERT INTO futures_large_oi (date, contract_code, name, long_position, long_position_over50, short_position, short_position_over50, net_position, market_total_oi) VALUES %s ON CONFLICT (date, contract_code, name) DO UPDATE SET net_position = EXCLUDED.net_position;"""


# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_holding(r: dict) -> tuple:
    lv = str(r.get("HoldingSharesLevel", ""))
    return (r["date"], r["stock_id"], lv, safe_int(r.get("people")), safe_float(r.get("percent")), r.get("unit", lv))

def map_eight_banks(r: dict) -> tuple:
    return (r["date"], r["stock_id"], safe_int(r.get("buy")), safe_int(r.get("sell")))

def map_futures_oi(r: dict) -> tuple:
    return (r["date"], r.get("contract_code", ""), r.get("name", ""), safe_int(r.get("long_position")), safe_int(r.get("long_position_over50")), safe_int(r.get("short_position")), safe_int(r.get("short_position_over50")), safe_int(r.get("net_position")), safe_int(r.get("market_total_oi")))


# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_holding(conn, stock_ids: list[str], start: str, end: str, delay: float, force: bool, fetch_mode_override: str | None = None):
    logger.info("=== [holding_shares_per] 開始 ===")
    ensure_ddl(conn, DDL_HOLDING)
    conn.commit()
    
    latest = get_all_safe_starts(conn, "holding_shares_per")
    flog = FailureLogger("holding_shares_per", db_conn=conn)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START["holding_shares_per"], force)
        if not s:
            write_fetch_log(conn, table_name="holding_shares_per", stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        start_time = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset="TaiwanStockHoldingSharesPer", 
                params={"data_id": sid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            duration_ms = int((time.time() - start_time) * 1000)
            
            if data:
                rows = [map_holding(r) for r in data]
                rows = dedup_rows(rows, (0, 1, 2))
                res = commit_per_stock_per_day(conn, UPSERT_HOLDING, rows, "(%s, %s, %s, %s, %s, %s)", label_prefix="holding", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                write_fetch_log(conn, table_name="holding_shares_per", stock_id=sid, fetch_mode=fetch_mode, fetch_date_from=s, fetch_date_to=end, rows_inserted=n, duration_ms=duration_ms, status="success")
            else:
                write_fetch_log(conn, table_name="holding_shares_per", stock_id=sid, fetch_mode=fetch_mode, fetch_date_from=s, fetch_date_to=end, rows_inserted=0, duration_ms=duration_ms, status="no_new_data")
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            write_fetch_log(conn, table_name="holding_shares_per", stock_id=sid, fetch_mode=fetch_mode, fetch_date_from=s, fetch_date_to=end, rows_inserted=0, duration_ms=duration_ms, status="failed", error_message=str(e))
            
    logger.info(f"  [holding_shares_per] 總共寫入 {total_rows} 筆")
    flog.summary()


def fetch_eight_banks(conn, stock_ids: list[str], start: str, end: str, delay: float, force: bool, fetch_mode_override: str | None = None):
    logger.info(f"=== [eight_banks] 開始 (目標全市場，若有指定 ID 則本地過濾) ===")
    ensure_ddl(conn, DDL_EIGHT_BANKS)
    conn.commit()
    
    flog = FailureLogger("eight_banks", db_conn=conn)
    fetch_mode = fetch_mode_override or "market"

    if not start and not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM eight_banks_buy_sell")
            max_d = cur.fetchone()[0]
            if max_d:
                start = (max_d + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start = DATASET_START["eight_banks"]

    s_dt = datetime.strptime(start or DATASET_START["eight_banks"], "%Y-%m-%d").date()
    e_dt = datetime.strptime(end, "%Y-%m-%d").date()
    
    if s_dt > e_dt:
        logger.info(f"  [eight_banks] 資料已是最新的。")
        write_fetch_log(conn, table_name="eight_banks_buy_sell", stock_id="ALL", fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
        return

    # ⚠️ FinMind 限制：TaiwanStockGovernmentBankBuySell 不支援 end_date，只能逐日抓取
    s_set = set(stock_ids) if stock_ids else None
    total_rows = 0
    curr = s_dt
    
    while curr <= e_dt:
        if curr.weekday() >= 5: # 跳過週末
            curr += timedelta(days=1)
            continue
            
        d_str = curr.strftime("%Y-%m-%d")
        start_time = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)，僅傳 start_date
            data = finmind_get(
                dataset="TaiwanStockGovernmentBankBuySell", 
                params={"start_date": d_str}, 
                delay=delay,
                raise_on_error=True
            )
            duration_ms = int((time.time() - start_time) * 1000)
            
            if data:
                agg = defaultdict(lambda: [0, 0])
                for r in data:
                    sid = r.get("stock_id")
                    if s_set and sid not in s_set: continue
                    k = (r["date"], sid)
                    agg[k][0] += safe_int(r.get("buy", 0))
                    agg[k][1] += safe_int(r.get("sell", 0))

                if agg:
                    rows = [(k[0], k[1], v[0], v[1]) for k, v in agg.items()]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, UPSERT_EIGHT_BANKS, rows, "(%s, %s, %s, %s)", label_prefix="eight_banks", failure_logger=flog)
                    n = sum(res.values())
                    total_rows += n
                    write_fetch_log(conn, table_name="eight_banks_buy_sell", stock_id="ALL", fetch_mode=fetch_mode, status="success", rows_inserted=n, fetch_date_from=d_str, fetch_date_to=d_str, duration_ms=duration_ms)
                else:
                    write_fetch_log(conn, table_name="eight_banks_buy_sell", stock_id="ALL", fetch_mode=fetch_mode, status="no_new_data", fetch_date_from=d_str, fetch_date_to=d_str, duration_ms=duration_ms)
            else:
                write_fetch_log(conn, table_name="eight_banks_buy_sell", stock_id="ALL", fetch_mode=fetch_mode, status="no_new_data", fetch_date_from=d_str, fetch_date_to=d_str, duration_ms=duration_ms)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            flog.record(date=d_str, error=str(e))
            write_fetch_log(conn, table_name="eight_banks_buy_sell", stock_id="ALL", fetch_mode=fetch_mode, status="failed", fetch_date_from=d_str, fetch_date_to=d_str, duration_ms=duration_ms, error_message=str(e))
        
        curr += timedelta(days=1)

    logger.info(f"  [eight_banks] 總共寫入 {total_rows} 筆")
    flog.summary()


def fetch_futures_oi(conn, start: str, end: str, delay: float, force: bool, fetch_mode_override: str | None = None):
    logger.info("=== [futures_large_oi] 開始 ===")
    ensure_ddl(conn, DDL_FUTURES_LARGE_OI)
    conn.commit()
    
    flog = FailureLogger("futures_large_oi", db_conn=conn)
    fetch_mode = fetch_mode_override or "market"
    start_time = time.time()
    
    try:
        # 修正：全面改為具名參數 (Keyword arguments)
        data = finmind_get(
            dataset="TaiwanFuturesOpenInterestLargeTraders", 
            params={"start_date": start, "end_date": end}, 
            delay=delay,
            raise_on_error=True
        )
        duration_ms = int((time.time() - start_time) * 1000)
        
        if data:
            rows = [map_futures_oi(r) for r in data]
            rows = dedup_rows(rows, (0, 1, 2))
            res = commit_per_stock_per_day(conn, UPSERT_FUTURES_LARGE_OI, rows, "(%s, %s, %s, %s, %s, %s, %s, %s, %s)", stock_index=1, label_prefix="futures", failure_logger=flog)
            n = sum(res.values())
            logger.info(f"  [futures_large_oi] 寫入 {n} 筆")
            write_fetch_log(conn, table_name="futures_large_oi", stock_id="FUTURES", fetch_mode=fetch_mode, status="success", rows_inserted=n, fetch_date_from=start, fetch_date_to=end, duration_ms=duration_ms)
        else:
            write_fetch_log(conn, table_name="futures_large_oi", stock_id="FUTURES", fetch_mode=fetch_mode, status="no_new_data", fetch_date_from=start, fetch_date_to=end, duration_ms=duration_ms)
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        flog.record(stock_id="futures", error=str(e), start_date=start, end_date=end)
        write_fetch_log(conn, table_name="futures_large_oi", stock_id="FUTURES", fetch_mode=fetch_mode, status="failed", fetch_date_from=start, fetch_date_to=end, duration_ms=duration_ms, error_message=str(e))
        
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

def query_gap_targets(conn, days: int, target_tables: list[str], all_stock_ids: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    for tbl in target_tables:
        if tbl == "eight_banks":
            sql = "SELECT 1 FROM fetch_log WHERE table_name = 'eight_banks_buy_sell' AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval LIMIT 1;"
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (str(days),))
                    if cur.fetchone() is None: targets[tbl].append("ALL")
            except Exception as e: logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")
            
        elif tbl == "futures_large_oi":
            sql = "SELECT 1 FROM fetch_log WHERE table_name = 'futures_large_oi' AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval LIMIT 1;"
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (str(days),))
                    if cur.fetchone() is None: targets[tbl].append("FUTURES")
            except Exception as e: logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")
            
        elif tbl == "holding_shares_per":
            sql = "SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = 'holding_shares_per' AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (str(days), all_stock_ids))
                    have_success = {row[0] for row in cur.fetchall()}
                missing = [sid for sid in all_stock_ids if sid not in have_success]
                targets[tbl].extend(missing)
            except Exception as e: logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [gap-fill/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        
        if tbl == "holding_shares_per":
            fetch_holding(conn, sids, args.start or DATASET_START["holding_shares_per"], args.end, args.delay, force=True, fetch_mode_override=fetch_mode)
        elif tbl == "eight_banks":
            fetch_eight_banks(conn, None, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)
        elif tbl == "futures_large_oi":
            fetch_futures_oi(conn, args.start or DATASET_START["futures_large_oi"], args.end, args.delay, force=True, fetch_mode_override=fetch_mode)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="進階籌碼資料抓取 (v3.2 — 核心模組全面升級版)")
    p.add_argument("--tables", nargs="+", choices=["holding_shares_per", "eight_banks", "futures_large_oi", "all"], default=["all"])
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["holding_shares_per", "eight_banks", "futures_large_oi"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        
        # 決定要抓取的標的
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            # 優先使用動態核心配置，並過濾 fetch_chip = TRUE
            stock_configs = get_core_stocks_from_db(conn)
            if stock_configs:
                stock_ids = [sid for sid, cfg in stock_configs.items() if cfg.get("is_active", True) and cfg.get("fetch_chip", True)]
            else:
                stock_ids = get_db_stock_ids(conn)

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
        if "holding_shares_per" in tables:
            fetch_holding(conn, stock_ids, args.start or DATASET_START["holding_shares_per"], args.end, args.delay, args.force)
        
        if "eight_banks" in tables:
            fetch_eight_banks(conn, stock_ids, args.start, args.end, args.delay, args.force)
        
        if "futures_large_oi" in tables:
            fetch_futures_oi(conn, args.start or DATASET_START["futures_large_oi"], args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        # 於程式完全結束時，印出統一的 FinMind RequestStats 報表
        get_request_stats().summary()

if __name__ == "__main__":
    main()