"""
fetch_news_data.py — 個股相關新聞（v3.4 核心模組全面升級版）
================================================================================
v3.4 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 捨棄舊版逐日呼叫，導入「時間切塊 (Chunking)」機制：以 30 天為一塊批次請求，保護配額。
  ★ 預設改為讀取 `stocks` 表中 `fetch_news=True` 的核心標的。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

執行範例（常規）：
    # 抓取核心標的最近 30 天新聞
    python scripts/fetchers/fetch_news_data.py
    
    # 僅抓取特定台積電新聞
    python scripts/fetchers/fetch_news_data.py --stock-id 2330
    
    # 抓取特定日期區間 (強制更新)
    python scripts/fetchers/fetch_news_data.py --start 2024-01-01 --end 2024-05-01 --force
    
    # 抓取全市場新聞（不建議，極消耗配額）
    python scripts/fetchers/fetch_news_data.py --all-market

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的任務
    python scripts/fetchers/fetch_news_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的個股
    python scripts/fetchers/fetch_news_data.py --gap-fill 30
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
    get_db_stock_ids,
    get_core_stocks_from_db,
    FailureLogger,
    commit_per_stock_per_day,
    resolve_start_cached,
    get_all_safe_starts,
    dedup_rows,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None, fetch_mode="per_stock"):
    """v3.4 標準化日誌寫入，失敗不影響主流程"""
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

DDL_NEWS = """
CREATE TABLE IF NOT EXISTS stock_news (
    date DATE, 
    stock_id VARCHAR(50), 
    title TEXT, 
    description TEXT, 
    source VARCHAR(200), 
    link TEXT, 
    PRIMARY KEY (date, stock_id, title)
);
CREATE INDEX IF NOT EXISTS idx_news_stock_date ON stock_news (stock_id, date);
"""

UPSERT_NEWS = """
INSERT INTO stock_news (date, stock_id, title, description, source, link) 
VALUES %s 
ON CONFLICT (date, stock_id, title) DO UPDATE SET 
    source = EXCLUDED.source,
    description = EXCLUDED.description;
"""


# ──────────────────────────────────────────────
# Mappers
# ──────────────────────────────────────────────
def map_news(r: dict) -> tuple:
    return (
        r["date"], 
        str(r["stock_id"]), 
        (r.get("title") or "")[:1000], 
        r.get("description"), 
        str(r.get("source") or "")[:200], 
        r.get("link")
    )


# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_stock_news(conn, stock_ids: list[str], start: str, end: str, delay: float, force: bool, fetch_mode_override: str | None = None):
    table = "stock_news"
    logger.info(f"=== [{table}] 開始（目標 {len(stock_ids)} 支股票） ===")
    
    ensure_ddl(conn, DDL_NEWS)
    conn.commit()
    
    flog = FailureLogger(table, db_conn=conn)
    latest = get_all_safe_starts(conn, table)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    tmpl = "(%s::date, %s, %s, %s, %s, %s)"

    for i, sid in enumerate(stock_ids, 1):
        s = resolve_start_cached(sid, latest, start, "2000-01-01", force)
        if not s:
            _write_fetch_log(conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        stock_total_rows = 0
        s_dt = datetime.strptime(s, "%Y-%m-%d").date()
        e_dt = datetime.strptime(end, "%Y-%m-%d").date()
        curr_start = s_dt
        
        try:
            # ⭐ 導入切塊抓取 (Chunking)：每 30 天為一個請求，保護配額同時加速
            while curr_start <= e_dt:
                curr_end = min(curr_start + timedelta(days=29), e_dt)
                s_str = curr_start.strftime("%Y-%m-%d")
                e_str = curr_end.strftime("%Y-%m-%d")

                # 修正：全面改為具名參數 (Keyword arguments)
                data = finmind_get(
                    dataset="TaiwanStockNews", 
                    params={"data_id": sid, "start_date": s_str, "end_date": e_str}, 
                    delay=delay,
                    raise_on_error=True
                )
                
                if data:
                    rows = [map_news(r) for r in data]
                    rows = dedup_rows(rows, (0, 1, 2)) # PK: date, stock_id, title
                    
                    res = commit_per_stock_per_day(conn, UPSERT_NEWS, rows, tmpl, stock_index=1, date_index=0, label_prefix=f"news/{sid}", failure_logger=flog)
                    stock_total_rows += sum(res.values())
                
                curr_start = curr_end + timedelta(days=1)

            duration_ms = int((time.time() - t0) * 1000)
            total_rows += stock_total_rows
            
            status = "success" if stock_total_rows > 0 else "no_new_data"
            _write_fetch_log(
                conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                fetch_date_from=s, fetch_date_to=end, rows_inserted=stock_total_rows, 
                duration_ms=duration_ms, status=status
            )
            
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(
                conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                fetch_date_from=s, fetch_date_to=end, rows_inserted=stock_total_rows, 
                duration_ms=duration_ms, status="failed", error_message=str(e)
            )
            
        if i % 10 == 0:
            logger.info(f"  進度：{i}/{len(stock_ids)}（累計寫入 {total_rows} 筆）")

    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
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


def resolve_stock_ids(args, conn) -> list[str]:
    """根據 CLI 參數與核心資料庫表解析要抓取的股票代號"""
    if args.stock_id:
        return [s.strip() for s in args.stock_id.split(",") if s.strip()]
    if args.all_market:
        return get_db_stock_ids(conn)
        
    # 預設：使用 DB 中的核心清單，並要求 fetch_news = TRUE
    stock_configs = get_core_stocks_from_db(conn)
    if stock_configs:
        return [sid for sid, cfg in stock_configs.items() if cfg.get("is_active", True) and cfg.get("fetch_news", True)]
    
    # 若資料庫無設定，退回預設全市場
    return get_db_stock_ids(conn)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="個股新聞抓取 (v3.4 — 核心模組升級版)")
    p.add_argument("--stock-id", default=None)
    p.add_argument("--all-market", action="store_true", help="抓取全市場（不建議，極消耗配額）")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=0.8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    conn = get_db_conn()
    try:
        _ensure_fetch_log_table(conn)
        stock_ids = resolve_stock_ids(args, conn)
        
        if not stock_ids:
            logger.warning("未找到需要抓取新聞的標的（請確認 stocks 表中 fetch_news 是否有設為 TRUE）。")
            return

        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        start_str = args.start if args.start else (end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, ["stock_news"])
            if targets and "stock_news" in targets: 
                fetch_stock_news(conn, targets["stock_news"], start_str, args.end, args.delay, force=True, fetch_mode_override="retry")
            else: 
                logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, ["stock_news"], stock_ids)
            if targets and "stock_news" in targets: 
                fetch_stock_news(conn, targets["stock_news"], start_str, args.end, args.delay, force=True, fetch_mode_override="gap_fill")
            else: 
                logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        fetch_stock_news(conn, stock_ids, start_str, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        get_request_stats().summary()

if __name__ == "__main__":
    main()