"""
fetch_extended_derivative_data.py — 期貨/選擇權 II（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 修復日誌摘要重複印出的 Bug，並在結束時自動印出 `finmind_client` 統計報表。

執行範例（常規）：
    # 抓取所有三大法人衍生品資料
    python scripts/fetchers/fetch_extended_derivative_data.py
    
    # 針對特定表抓取
    python scripts/fetchers/fetch_extended_derivative_data.py --tables futures_inst_investors
    
    # 針對特定商品抓取 (過濾)
    python scripts/fetchers/fetch_extended_derivative_data.py --ids TX,MTX,TXO --force
    python scripts/fetchers/fetch_extended_derivative_data.py --ids TX,MTX,TXO --tables all --force

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的組合
    python scripts/fetchers/fetch_extended_derivative_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python scripts/fetchers/fetch_extended_derivative_data.py --gap-fill 30
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse
import time

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
    safe_int,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
    write_fetch_log,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATASET_START = {
    "futures_inst_investors":   "2018-06-05",
    "options_inst_investors":   "2018-06-05",
    "futures_inst_after_hours": "2021-10-12",
    "options_inst_after_hours": "2021-10-12",
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

DDL_FUT_INST = """CREATE TABLE IF NOT EXISTS futures_inst_investors (date DATE, futures_id VARCHAR(50), institutional_investors VARCHAR(100), long_deal_volume BIGINT, long_deal_amount NUMERIC(20,6), short_deal_volume BIGINT, short_deal_amount NUMERIC(20,6), long_open_interest_balance_volume BIGINT, long_open_interest_balance_amount NUMERIC(20,6), short_open_interest_balance_volume BIGINT, short_open_interest_balance_amount NUMERIC(20,6), PRIMARY KEY (date, futures_id, institutional_investors));"""
DDL_OPT_INST = """CREATE TABLE IF NOT EXISTS options_inst_investors (date DATE, option_id VARCHAR(50), call_put VARCHAR(10), institutional_investors VARCHAR(100), long_deal_volume BIGINT, long_deal_amount NUMERIC(20,6), short_deal_volume BIGINT, short_deal_amount NUMERIC(20,6), long_open_interest_balance_volume BIGINT, long_open_interest_balance_amount NUMERIC(20,6), short_open_interest_balance_volume BIGINT, short_open_interest_balance_amount NUMERIC(20,6), PRIMARY KEY (date, option_id, call_put, institutional_investors));"""

UPSERT_FUT_INST = """INSERT INTO futures_inst_investors VALUES %s ON CONFLICT (date, futures_id, institutional_investors) DO UPDATE SET long_deal_volume = EXCLUDED.long_deal_volume;"""
UPSERT_OPT_INST = """INSERT INTO options_inst_investors VALUES %s ON CONFLICT (date, option_id, call_put, institutional_investors) DO UPDATE SET long_deal_volume = EXCLUDED.long_deal_volume;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_fut_inst(r): return (r["date"], r.get("futures_id") or r.get("name"), r.get("institutional_investors"), safe_int(r.get("long_deal_volume")), safe_float(r.get("long_deal_amount")), safe_int(r.get("short_deal_volume")), safe_float(r.get("short_deal_amount")), safe_int(r.get("long_open_interest_balance_volume")), safe_float(r.get("long_open_interest_balance_amount")), safe_int(r.get("short_open_interest_balance_volume")), safe_float(r.get("short_open_interest_balance_amount")))
def map_opt_inst(r): return (r["date"], r.get("option_id") or r.get("name"), r.get("call_put"), r.get("institutional_investors"), safe_int(r.get("long_deal_volume")), safe_float(r.get("long_deal_amount")), safe_int(r.get("short_deal_volume")), safe_float(r.get("short_deal_amount")), safe_int(r.get("long_open_interest_balance_volume")), safe_float(r.get("long_open_interest_balance_amount")), safe_int(r.get("short_open_interest_balance_volume")), safe_float(r.get("short_open_interest_balance_amount")))

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_inst(conn, dataset, table, ddl, upsert_sql, mapper, start, end, delay, force, target_ids=None, fetch_mode_override=None):
    logger.info(f"=== [{table}] 開始 ===")
    ensure_ddl(conn, ddl)
    conn.commit()
    
    key_col = "futures_id" if "futures" in table else "option_id"
    # 使用 "ALL" 取得此表最後更新時間（市場級）
    latest = get_all_safe_starts(conn, table, key_col=key_col)
    # 取全部商品中最新的日期當作安全起點
    safe_start = max(latest.values()) if latest else DATASET_START.get(table, "2018-06-05")
    
    if not force and safe_start > end:
        logger.info(f"  [{table}] 已最新，跳過。")
        return
        
    s_dt = datetime.strptime(start if force else safe_start, "%Y-%m-%d").date()
    e_dt = datetime.strptime(end, "%Y-%m-%d").date()
    
    flog = FailureLogger(table, db_conn=conn)
    total_rows = 0
    tmpl = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" if "futures" in table else "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    fetch_mode = fetch_mode_override or "market"
    
    curr = s_dt
    while curr <= e_dt:
        if curr.weekday() >= 5: # 跳過週末
            curr += timedelta(days=1)
            continue
            
        d_str = curr.strftime("%Y-%m-%d")
        logger.info(f"  [{table}] 正在抓取 {d_str}...")
        
        t0 = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset=dataset, 
                params={"start_date": d_str, "end_date": d_str}, 
                delay=delay,
                raise_on_error=True
            )
            dur = int((time.time() - t0) * 1000)
            
            if data:
                if target_ids:
                    id_key = "futures_id" if "futures" in table else "option_id"
                    data = [r for r in data if r.get(id_key) in target_ids]
                
                rows = [mapper(r) for r in data]
                rows = dedup_rows(rows, (0, 1, 2) if "futures" in table else (0, 1, 2, 3))
                res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, label_prefix=table, failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                write_fetch_log(
                    conn, table_name=table, stock_id="ALL", fetch_mode=fetch_mode, 
                    fetch_date_from=d_str, fetch_date_to=d_str, 
                    rows_inserted=n, duration_ms=dur, status="success" if n > 0 else "partial"
                )
            else:
                write_fetch_log(
                    conn, table_name=table, stock_id="ALL", fetch_mode=fetch_mode, 
                    fetch_date_from=d_str, fetch_date_to=d_str, 
                    rows_inserted=0, duration_ms=dur, status="no_new_data"
                )
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id="market", error=str(e), date=d_str)
            write_fetch_log(
                conn, table_name=table, stock_id="ALL", fetch_mode=fetch_mode, 
                fetch_date_from=d_str, fetch_date_to=d_str, 
                rows_inserted=0, duration_ms=dur, status="failed", error_message=str(e)
            )
        
        curr += timedelta(days=1)

    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
    flog.summary()


# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int, target_tables: list[str]) -> list[str]:
    """回傳需 retry 的資料表清單 (此處寫入 fetch_log 的 stock_id 為 'ALL')"""
    targets = []
    sql = """
    WITH recent AS (
        SELECT table_name, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY table_name ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = ANY(%s) AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT table_name FROM recent WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (target_tables, str(days)))
            targets = [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"[retry-failed] 查詢失敗：{e}")
    
    if targets:
        logger.info(f"  [retry-failed] 發現 {len(targets)} 個需重試的資料表：{targets}")
    return targets

def query_gap_targets(conn, days: int, target_tables: list[str]) -> list[str]:
    """回傳需補抓的資料表清單 (近 N 天無 success 紀錄)"""
    targets = []
    for tbl in target_tables:
        sql = """
        SELECT 1 FROM fetch_log 
        WHERE table_name = %s AND status = 'success' 
          AND run_ts > NOW() - (%s || ' days')::interval LIMIT 1;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tbl, str(days)))
                if cur.fetchone() is None:
                    targets.append(tbl)
        except Exception as e:
            logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    if targets:
        logger.info(f"  [gap-fill] 發現 {len(targets)} 個需補抓的資料表：{targets}")
    return targets


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["futures_inst_investors", "options_inst_investors", "all"], default=["all"])
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--ids", help="指定標的 ID (例如 TX, MTX, TXO)，多筆用逗號分隔")
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["futures_inst_investors", "options_inst_investors"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        target_ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
        
        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            run_tables = query_failed_targets(conn, args.retry_failed, tables)
            if not run_tables:
                logger.info("沒有找到需要重試的目標，結束。")
                return
            for tbl in run_tables:
                if tbl == "futures_inst_investors": fetch_inst(conn, "TaiwanFuturesInstitutionalInvestors", "futures_inst_investors", DDL_FUT_INST, UPSERT_FUT_INST, map_fut_inst, args.start, args.end, args.delay, force=True, target_ids=target_ids, fetch_mode_override="retry")
                if tbl == "options_inst_investors": fetch_inst(conn, "TaiwanOptionInstitutionalInvestors", "options_inst_investors", DDL_OPT_INST, UPSERT_OPT_INST, map_opt_inst, args.start, args.end, args.delay, force=True, target_ids=target_ids, fetch_mode_override="retry")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            run_tables = query_gap_targets(conn, args.gap_fill, tables)
            if not run_tables:
                logger.info("沒有找到需要補抓的目標，結束。")
                return
            for tbl in run_tables:
                if tbl == "futures_inst_investors": fetch_inst(conn, "TaiwanFuturesInstitutionalInvestors", "futures_inst_investors", DDL_FUT_INST, UPSERT_FUT_INST, map_fut_inst, args.start, args.end, args.delay, force=True, target_ids=target_ids, fetch_mode_override="gap_fill")
                if tbl == "options_inst_investors": fetch_inst(conn, "TaiwanOptionInstitutionalInvestors", "options_inst_investors", DDL_OPT_INST, UPSERT_OPT_INST, map_opt_inst, args.start, args.end, args.delay, force=True, target_ids=target_ids, fetch_mode_override="gap_fill")
            return

        # 模式 C：常規抓取
        if "futures_inst_investors" in tables: 
            fetch_inst(conn, "TaiwanFuturesInstitutionalInvestors", "futures_inst_investors", DDL_FUT_INST, UPSERT_FUT_INST, map_fut_inst, args.start, args.end, args.delay, args.force, target_ids)
        if "options_inst_investors" in tables: 
            fetch_inst(conn, "TaiwanOptionInstitutionalInvestors", "options_inst_investors", DDL_OPT_INST, UPSERT_OPT_INST, map_opt_inst, args.start, args.end, args.delay, args.force, target_ids)
            
    finally:
        conn.close()
        logger.info("全部完成")
        # 於程式完全結束時，印出統一的 FinMind RequestStats 報表
        get_request_stats().summary()

if __name__ == "__main__":
    main()