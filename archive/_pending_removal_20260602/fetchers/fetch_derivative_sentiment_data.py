"""
fetch_derivative_sentiment_data.py — 衍生品與情緒指標（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 鉅額交易預設改為讀取 `stocks` 表中 `fetch_derivative=True` 或活躍的核心標的。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

執行範例（常規）：
    # 抓取所有情緒與衍生品表
    python scripts/fetchers/fetch_derivative_sentiment_data.py
    
    # 針對特定市場層級指標抓取
    python scripts/fetchers/fetch_derivative_sentiment_data.py --tables fear_greed_index
    
    # 針對特定個股抓取鉅額交易 (強制更新)
    python scripts/fetchers/fetch_derivative_sentiment_data.py --ids 2330 --tables block_trading --force

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的組合
    python scripts/fetchers/fetch_derivative_sentiment_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python scripts/fetchers/fetch_derivative_sentiment_data.py --gap-fill 30
    
    # 針對鉅額交易特定補抓
    python scripts/fetchers/fetch_derivative_sentiment_data.py --gap-fill 14 --tables block_trading
"""

from __future__ import annotations

import sys
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse
import time
from pathlib import Path

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
    get_db_stock_ids,
    get_core_stocks_from_db,
    ensure_ddl,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    commit_per_day,
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
    "options_large_oi":  "2018-01-01",
    "fear_greed_index":  "2011-01-03",
    "block_trading":     "2021-01-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

MARKET_LEVEL_TABLES = {"options_oi_large_holders", "fear_greed_index"}

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

def _write_fetch_log(conn, **kwargs):
    """寫入 fetch_log，失敗不影響主流程。"""
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO fetch_log (
                run_ts, table_name, stock_id, fetch_mode,
                fetch_date_from, fetch_date_to,
                rows_inserted, rows_updated, duration_ms,
                status, error_message, cli_args
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
            """
            cur.execute(sql, (
                kwargs.get("table_name"), kwargs.get("stock_id"), kwargs.get("fetch_mode", "per_stock"),
                kwargs.get("fetch_date_from"), kwargs.get("fetch_date_to"),
                kwargs.get("rows_inserted", 0), kwargs.get("duration_ms", 0),
                kwargs.get("status"), kwargs.get("error_message"), _CLI_ARGS_STR
            ))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.debug(f"fetch_log 寫入失敗：{e}")

DDL_SENTIMENT = """
CREATE TABLE IF NOT EXISTS options_oi_large_holders (date DATE, option_id VARCHAR(50), put_call VARCHAR(10), contract_type VARCHAR(50), name VARCHAR(100), market_open_interest NUMERIC, buy_top5_trader_open_interest NUMERIC, buy_top5_trader_open_interest_per NUMERIC, buy_top10_trader_open_interest NUMERIC, buy_top10_trader_open_interest_per NUMERIC, sell_top5_trader_open_interest NUMERIC, sell_top5_trader_open_interest_per NUMERIC, sell_top10_trader_open_interest NUMERIC, sell_top10_trader_open_interest_per NUMERIC, buy_top5_specific_open_interest NUMERIC, buy_top5_specific_open_interest_per NUMERIC, buy_top10_specific_open_interest NUMERIC, buy_top10_specific_open_interest_per NUMERIC, sell_top5_specific_open_interest NUMERIC, sell_top5_specific_open_interest_per NUMERIC, sell_top10_specific_open_interest NUMERIC, sell_top10_specific_open_interest_per NUMERIC, PRIMARY KEY (date, option_id, put_call, contract_type));
CREATE TABLE IF NOT EXISTS fear_greed_index (date DATE PRIMARY KEY, fear_greed NUMERIC, fear_greed_emotion VARCHAR(50));
CREATE TABLE IF NOT EXISTS block_trading (date DATE, stock_id VARCHAR(50), securities_trader_id VARCHAR(50), securities_trader VARCHAR(100), price NUMERIC(10,2), buy NUMERIC, sell NUMERIC, trade_type VARCHAR(50), PRIMARY KEY (date, stock_id, securities_trader_id, price, trade_type));
"""

UPSERT_OPTIONS_LARGE_OI = """INSERT INTO options_oi_large_holders VALUES %s ON CONFLICT (date, option_id, put_call, contract_type) DO UPDATE SET market_open_interest = EXCLUDED.market_open_interest;"""
UPSERT_FEAR_GREED = """INSERT INTO fear_greed_index (date, fear_greed, fear_greed_emotion) VALUES %s ON CONFLICT (date) DO UPDATE SET fear_greed = EXCLUDED.fear_greed;"""
UPSERT_BLOCK_TRADING = """INSERT INTO block_trading (date, stock_id, securities_trader_id, securities_trader, price, buy, sell, trade_type) VALUES %s ON CONFLICT (date, stock_id, securities_trader_id, price, trade_type) DO UPDATE SET buy = EXCLUDED.buy, sell = EXCLUDED.sell;"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_opt_large(r): return (r["date"], str(r.get("option_id", "")), r.get("put_call", ""), str(r.get("contract_type", "")), r.get("name", ""), safe_float(r.get("market_open_interest")), safe_float(r.get("buy_top5_trader_open_interest")), safe_float(r.get("buy_top5_trader_open_interest_per")), safe_float(r.get("buy_top10_trader_open_interest")), safe_float(r.get("buy_top10_trader_open_interest_per")), safe_float(r.get("sell_top5_trader_open_interest")), safe_float(r.get("sell_top5_trader_open_interest_per")), safe_float(r.get("sell_top10_trader_open_interest")), safe_float(r.get("sell_top10_trader_open_interest_per")), safe_float(r.get("buy_top5_specific_open_interest")), safe_float(r.get("buy_top5_specific_open_interest_per")), safe_float(r.get("buy_top10_specific_open_interest")), safe_float(r.get("buy_top10_specific_open_interest_per")), safe_float(r.get("sell_top5_specific_open_interest")), safe_float(r.get("sell_top5_specific_open_interest_per")), safe_float(r.get("sell_top10_specific_open_interest")), safe_float(r.get("sell_top10_specific_open_interest_per")))
def map_fg(r): return (r["date"], safe_float(r.get("fear_greed")), r.get("fear_greed_emotion", "")[:50])
def map_block(r): return (r["date"], r["stock_id"], str(r.get("securities_trader_id", "")), r.get("securities_trader", "")[:100], safe_float(r.get("price")), safe_float(r.get("volume")), 0.0, str(r.get("trade_type", ""))[:50])

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_block_trading(conn, stock_ids, start, end, delay, force, fetch_mode_override=None):
    logger.info("=== [block_trading] 開始 ===")
    ensure_ddl(conn, DDL_SENTIMENT)
    conn.commit()
    
    flog = FailureLogger("block_trading", db_conn=conn)
    latest = get_all_safe_starts(conn, "block_trading")
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START["block_trading"], force)
        if not s:
            _write_fetch_log(conn, table_name="block_trading", stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset="TaiwanStockBlockTrade", 
                params={"data_id": sid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            dur = int((time.time() - t0) * 1000)
            
            if data:
                rows = [map_block(r) for r in data]
                rows = dedup_rows(rows, (0, 1, 2, 4, 7))
                res = commit_per_stock_per_day(conn, UPSERT_BLOCK_TRADING, rows, None, label_prefix=f"block/{sid}", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(
                    conn, table_name="block_trading", stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=n, 
                    duration_ms=dur, status="success" if n > 0 else "partial"
                )
            else:
                _write_fetch_log(
                    conn, table_name="block_trading", stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                    duration_ms=dur, status="no_new_data"
                )
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(
                conn, table_name="block_trading", stock_id=sid, fetch_mode=fetch_mode, 
                fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                duration_ms=dur, status="failed", error_message=str(e)
            )
            
    logger.info(f"  [block_trading] 總共寫入 {total_rows} 筆")
    flog.summary()

def fetch_sentiment(conn, dataset, table, upsert_sql, mapper, start, end, delay, force, fetch_mode_override=None):
    logger.info(f"=== [{table}] 開始 ===")
    ensure_ddl(conn, DDL_SENTIMENT)
    conn.commit()
    
    flog = FailureLogger(table, db_conn=conn)
    fetch_mode = fetch_mode_override or "market"

    # ⭐ 自動尋找起始日 ⭐
    if not start and not force:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(date) FROM {table}")
            max_d = cur.fetchone()[0]
            if max_d:
                start = (max_d + timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"  [{table}] 自動從資料庫最新日期續傳：{start}")
            else:
                start = DATASET_START.get(table.replace("options_oi_large_holders", "options_large_oi"), "2021-01-01")
    
    s_dt = datetime.strptime(start or DATASET_START.get(table.replace("options_oi_large_holders", "options_large_oi"), "2021-01-01"), "%Y-%m-%d").date()
    e_dt = datetime.strptime(end, "%Y-%m-%d").date()
    
    if s_dt > e_dt:
        logger.info(f"  [{table}] 已最新，跳過")
        _write_fetch_log(conn, table_name=table, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
        return
    
    total_rows = 0
    curr = s_dt
    while curr <= e_dt:
        d_str = curr.strftime("%Y-%m-%d")
        logger.info(f"  [{table}] 正在抓取自 {d_str} 起的資料塊...")
        t0 = time.time()
        try:
            # ⭐ 核心優化：不帶 end_date 可觸發快速度大量回傳 (約 200-300 天) 
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset=dataset, 
                params={"start_date": d_str}, 
                delay=delay,
                raise_on_error=True
            )
            dur = int((time.time() - t0) * 1000)
            
            if not data:
                # 若無資料，則跳過一天繼續
                _write_fetch_log(
                    conn, table_name=table, fetch_mode=fetch_mode, fetch_date_from=d_str, 
                    rows_inserted=0, duration_ms=dur, status="no_new_data"
                )
                curr += timedelta(days=1)
                continue
                
            # 轉換資料
            rows = [mapper(r) for r in data]
            
            # 找出這批資料中最後一天的日期
            received_dates = sorted(list(set(r[0] for r in rows)))
            logger.info(f"    -> 成功接收 {len(received_dates)} 天的資料 ({received_dates[0]} ~ {received_dates[-1]})")
            
            last_date_str = received_dates[-1]
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            
            # 寫入資料庫
            if table == "fear_greed_index":
                rows = dedup_rows(rows, (0,))
                res = commit_per_day(conn, upsert_sql, rows, "(%s::date, %s::numeric, %s)", date_index=0, label_prefix=table, failure_logger=flog)
            elif table == "options_oi_large_holders":
                rows = dedup_rows(rows, (0, 1, 2, 3))
                tmpl = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                res = commit_per_day(conn, upsert_sql, rows, tmpl, date_index=0, label_prefix=table, failure_logger=flog)
            else:
                rows = dedup_rows(rows, (0, 1, 2, 3))
                res = commit_per_stock_per_day(conn, upsert_sql, rows, None, label_prefix=table, failure_logger=flog)
            
            n = sum(res.values())
            total_rows += n
            _write_fetch_log(
                conn, table_name=table, fetch_mode=fetch_mode, 
                fetch_date_from=d_str, fetch_date_to=last_date_str, 
                rows_inserted=n, duration_ms=dur, status="success" if n > 0 else "partial"
            )
            
            # ⭐ 下一次從最後一天的隔天開始 ⭐
            new_curr = last_date + timedelta(days=1)
            
            # 如果日期沒推進，強制跳一天避開無窮迴圈
            if new_curr <= curr:
                curr += timedelta(days=1)
            else:
                curr = new_curr
            
            if curr > e_dt:
                break
                
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(date=d_str, error=str(e))
            _write_fetch_log(
                conn, table_name=table, fetch_mode=fetch_mode, fetch_date_from=d_str, 
                rows_inserted=0, duration_ms=dur, status="failed", error_message=str(e)
            )
            curr += timedelta(days=1)

    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
    flog.summary()

# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int, target_tables: list[str]) -> dict[str, list[str | None]]:
    targets: dict[str, list[str | None]] = defaultdict(list)
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
        sample = [s for s in sids if s is not None][:5]
        logger.info(f"  [retry-failed/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def query_gap_targets(conn, days: int, target_tables: list[str], all_stock_ids: list[str]) -> dict[str, list[str | None]]:
    targets: dict[str, list[str | None]] = defaultdict(list)
    for tbl in target_tables:
        if tbl in MARKET_LEVEL_TABLES:
            sql = f"SELECT 1 FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval LIMIT 1;"
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (tbl, str(days)))
                    if cur.fetchone() is None: targets[tbl].append(None)
            except Exception as e: logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")
        else:
            sql = f"SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (tbl, str(days), all_stock_ids))
                    have_success = {row[0] for row in cur.fetchall()}
                missing = [sid for sid in all_stock_ids if sid not in have_success]
                targets[tbl].extend(missing)
            except Exception as e: logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, lst in targets.items():
        sample = [s for s in lst if s is not None][:5]
        logger.info(f"  [gap-fill/{tbl}] {len(lst)} 個目標 (例：{sample})")
    return targets

def _run_targeted(conn, targets: dict[str, list[str | None]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        
        if tbl == "options_oi_large_holders":
            fetch_sentiment(conn, "TaiwanOptionOpenInterestLargeTraders", "options_oi_large_holders", UPSERT_OPTIONS_LARGE_OI, map_opt_large, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)
        elif tbl == "fear_greed_index":
            fetch_sentiment(conn, "CnnFearGreedIndex", "fear_greed_index", UPSERT_FEAR_GREED, map_fg, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)
        elif tbl == "block_trading":
            real_sids = [s for s in sids if s is not None]
            if real_sids:
                fetch_block_trading(conn, real_sids, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["options_large_oi", "fear_greed_index", "block_trading", "all"], default=["all"])
    p.add_argument("--ids")
    p.add_argument("--start", default="")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["options_large_oi", "fear_greed_index", "block_trading"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        
        # 決定鉅額交易要抓取的標的
        if args.ids:
            stock_ids = [s.strip() for s in args.ids.split(",")]
        else:
            # 優先使用動態核心配置，否則退回舊版
            stock_configs = get_core_stocks_from_db(conn)
            if stock_configs:
                stock_ids = [sid for sid, cfg in stock_configs.items() if cfg.get("is_active", True)]
            else:
                stock_ids = get_db_stock_ids(conn)
        
        db_table_map = {
            "options_large_oi": "options_oi_large_holders",
            "fear_greed_index": "fear_greed_index",
            "block_trading": "block_trading"
        }
        db_tables = [db_table_map[t] for t in tables]

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, db_tables)
            if targets: _run_targeted(conn, targets, args, fetch_mode="retry")
            else: logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, db_tables, stock_ids)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        if "options_large_oi" in tables: 
            fetch_sentiment(conn, "TaiwanOptionOpenInterestLargeTraders", "options_oi_large_holders", UPSERT_OPTIONS_LARGE_OI, map_opt_large, args.start, args.end, args.delay, args.force)
        if "fear_greed_index" in tables: 
            fetch_sentiment(conn, "CnnFearGreedIndex", "fear_greed_index", UPSERT_FEAR_GREED, map_fg, args.start, args.end, args.delay, args.force)
        if "block_trading" in tables: 
            fetch_block_trading(conn, stock_ids, args.start or DATASET_START["block_trading"], args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        # 於程式完全結束時，印出統一的 FinMind RequestStats 報表
        get_request_stats().summary()

if __name__ == "__main__":
    main()