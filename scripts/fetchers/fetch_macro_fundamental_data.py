"""
fetch_macro_fundamental_data.py — 總經與基本面補強資料（v3.3 巨量資料防空版）
====================================================================================
v3.3 改進：
  ★ 實作「時間切塊 (Chunking)」：針對市值權重 (market_value_weight) 這種動輒百萬筆
    的巨量全市場資料，自動將請求切分為每 30 天一塊，突破 FinMind API 的單次索取上限，
    解決範圍過大導致伺服器直接回傳空陣列 (寫入 0 筆) 的問題。

v3.2 既有（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 根據資料特性，精細化調用不同的原子寫入函式（commit_per_day / commit_per_stock）。

使用範例（常規）：
    python scripts/fetchers/fetch_macro_fundamental_data.py
    python scripts/fetchers/fetch_macro_fundamental_data.py --tables market_value_weight --start 2024-01-01 --force
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
    safe_int,
    get_market_safe_start,
    FailureLogger,
    commit_per_stock_per_day,
    commit_per_day,
    commit_per_stock,
    dedup_rows,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATASET_START = {
    "business_indicator":  "1982-01-01",
    "market_value_weight": "2024-10-30",
    "industry_chain":      None,
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

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None, fetch_mode="market"):
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

DDL_MACRO_FUND = """
CREATE TABLE IF NOT EXISTS business_indicator (
    date DATE PRIMARY KEY, 
    "leading" NUMERIC(10,2), 
    leading_notrend NUMERIC(10,2), 
    coincident NUMERIC(10,2), 
    coincident_notrend NUMERIC(10,2), 
    lagging NUMERIC(10,2), 
    lagging_notrend NUMERIC(10,2), 
    monitoring NUMERIC(10,2), 
    monitoring_color VARCHAR(20)
);
CREATE TABLE IF NOT EXISTS market_value_weight (
    date DATE, 
    stock_id VARCHAR(50), 
    stock_name VARCHAR(100), 
    rank INTEGER, 
    weight_per NUMERIC(8,4), 
    type VARCHAR(10), 
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS industry_chain (
    stock_id VARCHAR(50), 
    industry VARCHAR(100), 
    sub_industry VARCHAR(100), 
    date DATE, 
    PRIMARY KEY (stock_id)
);
"""

UPSERT_BI = """
INSERT INTO business_indicator 
VALUES %s 
ON CONFLICT (date) DO UPDATE SET monitoring = EXCLUDED.monitoring;
"""

UPSERT_MVW = """
INSERT INTO market_value_weight 
VALUES %s 
ON CONFLICT (date, stock_id) DO UPDATE SET weight_per = EXCLUDED.weight_per;
"""

UPSERT_IC = """
INSERT INTO industry_chain 
VALUES %s 
ON CONFLICT (stock_id) DO UPDATE SET industry = EXCLUDED.industry;
"""


# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_bi(r): 
    return (
        r["date"], 
        safe_float(r.get("leading")), 
        safe_float(r.get("leading_notrend")), 
        safe_float(r.get("coincident")), 
        safe_float(r.get("coincident_notrend")), 
        safe_float(r.get("lagging")), 
        safe_float(r.get("lagging_notrend")), 
        safe_float(r.get("monitoring")), 
        r.get("monitoring_color", "")[:20]
    )

def map_mvw(r): 
    return (
        r["date"], 
        str(r.get("stock_id", "")), 
        r.get("stock_name", "")[:100], 
        safe_int(r.get("rank")), 
        safe_float(r.get("weight_per")), 
        r.get("type", "")[:10]
    )

def map_ic(r): 
    return (
        str(r.get("stock_id", "")), 
        r.get("industry", "")[:100], 
        r.get("sub_industry", "")[:100], 
        r.get("date")
    )


# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_macro_fund(conn, dataset, table, upsert_sql, mapper, start, end, delay, force, fetch_mode_override=None):
    logger.info(f"=== [{table}] 開始 ===")
    ensure_ddl(conn, DDL_MACRO_FUND)
    conn.commit()
    
    flog = FailureLogger(table, db_conn=conn)
    total_rows = 0
    fetch_mode = fetch_mode_override or "market"
    
    # 決定實際抓取的起始日期 (industry_chain 沒有日期概念，全量抓取)
    s = None
    if table != "industry_chain":
        safe_s = get_market_safe_start(conn, table) if not force else None
        s = safe_s if safe_s else (start or DATASET_START[table])
        if s and end and s > end:
            logger.info(f"  [{table}] 已最新，跳過")
            _write_fetch_log(conn, table, "ALL", "skipped", fetch_date_from=s, fetch_date_to=end, fetch_mode=fetch_mode)
            return
            
    t0 = time.time()
    try:
        # ⭐ v3.3 針對巨量全市場資料 (market_value_weight) 實作切塊抓取 ⭐
        if table == "market_value_weight":
            s_dt = datetime.strptime(s, "%Y-%m-%d").date()
            e_dt = datetime.strptime(end, "%Y-%m-%d").date()
            curr_start = s_dt
            
            while curr_start <= e_dt:
                # 每次只抓 30 天 (約 6 萬筆資料)，確保在 API 限制之內
                curr_end = min(curr_start + timedelta(days=29), e_dt)
                s_str = curr_start.strftime("%Y-%m-%d")
                e_str = curr_end.strftime("%Y-%m-%d")
                
                logger.info(f"  [{table}] 批次抓取區間：{s_str} ~ {e_str}...")
                
                data = finmind_get(
                    dataset=dataset, 
                    params={"start_date": s_str, "end_date": e_str}, 
                    delay=delay,
                    raise_on_error=True
                )
                
                if data:
                    rows = [mapper(r) for r in data]
                    rows = dedup_rows(rows, (0, 1))
                    tmpl = "(%s::date,%s,%s,%s,%s,%s)"
                    res = commit_per_stock_per_day(conn, upsert_sql, rows, tmpl, stock_index=1, date_index=0, label_prefix=table, failure_logger=flog)
                    n = sum(res.values())
                    total_rows += n
                
                # 推進至下一個 30 天
                curr_start = curr_end + timedelta(days=1)
                
            duration_ms = int((time.time() - t0) * 1000)
            status = "success" if total_rows > 0 else "no_new_data"
            _write_fetch_log(conn, table, "ALL", status, total_rows, s, end, duration_ms, fetch_mode=fetch_mode)

        else:
            # 景氣對策信號、產業鏈等資料量較小，維持單次抓取
            params = {}
            if s: params["start_date"] = s
            if end and table != "industry_chain": params["end_date"] = end
                
            data = finmind_get(
                dataset=dataset, 
                params=params, 
                delay=delay,
                raise_on_error=True
            )
            duration_ms = int((time.time() - t0) * 1000)
            
            if data:
                rows = [mapper(r) for r in data]
                if table == "business_indicator":
                    rows = dedup_rows(rows, (0,))
                    tmpl = "(%s::date,%s,%s,%s,%s,%s,%s,%s,%s)"
                    res = commit_per_day(conn, upsert_sql, rows, tmpl, date_index=0, label_prefix=table, failure_logger=flog)
                elif table == "industry_chain":
                    rows = dedup_rows(rows, (0,))
                    tmpl = "(%s,%s,%s,%s::date)"
                    res = commit_per_stock(conn, upsert_sql, rows, tmpl, stock_index=0, label_prefix=table, failure_logger=flog)
                
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, table, "ALL", "success", total_rows, s, end, duration_ms, fetch_mode=fetch_mode)
            else:
                _write_fetch_log(conn, table, "ALL", "no_new_data", 0, s, end, duration_ms, fetch_mode=fetch_mode)
            
    except Exception as e:
        duration_ms = int((time.time() - t0) * 1000)
        flog.record(stock_id="ALL", error=str(e), start_date=s, end_date=end)
        _write_fetch_log(conn, table, "ALL", "failed", 0, s, end, duration_ms, error_message=str(e), fetch_mode=fetch_mode)
        
    logger.info(f"  [{table}] 寫入 {total_rows} 筆")
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
    p.add_argument("--tables", nargs="+", choices=["business_indicator", "market_value_weight", "industry_chain", "all"], default=["all"])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["business_indicator", "market_value_weight", "industry_chain"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        ensure_ddl(conn, DDL_MACRO_FUND)
        
        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            run_tables = query_failed_targets(conn, args.retry_failed, tables)
            if not run_tables:
                logger.info("沒有找到需要重試的目標，結束。")
                return
            for tbl in run_tables:
                if tbl == "business_indicator": fetch_macro_fund(conn, "TaiwanBusinessIndicator", "business_indicator", UPSERT_BI, map_bi, args.start, args.end, args.delay, force=True, fetch_mode_override="retry")
                if tbl == "market_value_weight": fetch_macro_fund(conn, "TaiwanStockMarketValueWeight", "market_value_weight", UPSERT_MVW, map_mvw, args.start, args.end, args.delay, force=True, fetch_mode_override="retry")
                if tbl == "industry_chain": fetch_macro_fund(conn, "TaiwanStockIndustryChain", "industry_chain", UPSERT_IC, map_ic, None, None, args.delay, force=True, fetch_mode_override="retry")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            run_tables = query_gap_targets(conn, args.gap_fill, tables)
            if not run_tables:
                logger.info("沒有找到需要補抓的目標，結束。")
                return
            for tbl in run_tables:
                if tbl == "business_indicator": fetch_macro_fund(conn, "TaiwanBusinessIndicator", "business_indicator", UPSERT_BI, map_bi, args.start, args.end, args.delay, force=True, fetch_mode_override="gap_fill")
                if tbl == "market_value_weight": fetch_macro_fund(conn, "TaiwanStockMarketValueWeight", "market_value_weight", UPSERT_MVW, map_mvw, args.start, args.end, args.delay, force=True, fetch_mode_override="gap_fill")
                if tbl == "industry_chain": fetch_macro_fund(conn, "TaiwanStockIndustryChain", "industry_chain", UPSERT_IC, map_ic, None, None, args.delay, force=True, fetch_mode_override="gap_fill")
            return

        # 模式 C：常規抓取
        if "business_indicator" in tables:
            fetch_macro_fund(conn, "TaiwanBusinessIndicator", "business_indicator", UPSERT_BI, map_bi, args.start, args.end, args.delay, args.force)
        
        if "market_value_weight" in tables:
            fetch_macro_fund(conn, "TaiwanStockMarketValueWeight", "market_value_weight", UPSERT_MVW, map_mvw, args.start, args.end, args.delay, args.force)
        
        if "industry_chain" in tables:
            fetch_macro_fund(conn, "TaiwanStockIndustryChain", "industry_chain", UPSERT_IC, map_ic, None, None, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        get_request_stats().summary()

if __name__ == "__main__":
    main()