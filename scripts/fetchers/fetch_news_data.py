import sys
import logging
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_news_data.py v3.2 — 個股相關新聞（可觀察性監控版）
================================================================================
v3.2 重大改進：
  ★ 整合 fetch_log v3.1：每一支股票的抓取狀態、耗時與筆數均記錄至監控日誌。
  ★ 核心優先：預設抓取 STOCK_CONFIGS 的核心股票，避開全市場雜訊與 400 錯誤。
  ★ 效能監控：精準追蹤每一請求的 API 耗時（duration_ms）。
  ★ 原子性 Commit：每一天、每一支股票獨立 commit，確保斷點續傳。

執行範例（預設抓取核心標的）：
    python scripts/fetchers/fetch_news_data.py

    # 抓取特定標的最近 7 天新聞
    python scripts/fetchers/fetch_news_data.py --stock-id 2330,2317 --days 7

    # 強制重抓最近 30 天資料
    python scripts/fetchers/fetch_news_data.py --days 30 --force

    # 抓取特定時間範圍
    python scripts/fetchers/fetch_news_data.py --start 2024-01-01 --end 2024-05-01
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    get_db_stock_ids,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
)
from config import STOCK_CONFIGS

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

def map_news(r: dict) -> tuple:
    return (r["date"], r["stock_id"], (r.get("title") or "")[:1000], r.get("description"), r.get("source"), r.get("link"))

def get_existing_dates(conn, stock_id: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT date FROM stock_news WHERE stock_id = %s", (stock_id,))
        return {row[0].strftime("%Y-%m-%d") for row in cur.fetchall()}

def resolve_stock_ids(args, conn) -> list[str]:
    if args.stock_id:
        return [s.strip() for s in args.stock_id.split(",") if s.strip()]
    if args.all_market:
        return get_db_stock_ids(conn)
    return list(STOCK_CONFIGS.keys())

def main():
    p = argparse.ArgumentParser(description="個股新聞抓取 (v3.2 — 可觀察性監控版)")
    p.add_argument("--stock-id", default=None)
    p.add_argument("--all-market", action="store_true")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=0.8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_NEWS)
        stock_ids = resolve_stock_ids(args, conn)

        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        start_dt = datetime.strptime(args.start, "%Y-%m-%d") if args.start else (end_dt - timedelta(days=args.days))
        all_dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_dt - start_dt).days + 1)]
        
        f_from = start_dt.strftime("%Y-%m-%d")
        f_to = end_dt.strftime("%Y-%m-%d")

        logger.info(f"日期範圍：{f_from} ~ {f_to}（目標股票：{len(stock_ids)} 支）")

        flog = FailureLogger("stock_news", db_conn=conn)
        total_rows = 0

        for i, sid in enumerate(stock_ids, 1):
            start_time = time.time()
            try:
                existing = set() if args.force else get_existing_dates(conn, sid)
                todo = [d for d in all_dates if d not in existing]
                
                if not todo:
                    _write_fetch_log(conn, "stock_news", sid, "skipped", fetch_date_from=f_from, fetch_date_to=f_to, error_message="up_to_date")
                    continue

                stock_rows = 0
                for d in todo:
                    data = finmind_get("TaiwanStockNews", {"data_id": sid, "start_date": d}, args.delay)
                    if data:
                        rows = [map_news(r) for r in data]
                        rows = dedup_rows(rows, (0, 1, 2))
                        res = commit_per_stock_per_day(conn, UPSERT_NEWS, rows, "(%s, %s, %s, %s, %s, %s)", label_prefix=f"news/{sid}", failure_logger=flog)
                        stock_rows += sum(res.values())
                
                duration_ms = int((time.time() - start_time) * 1000)
                total_rows += stock_rows
                
                status = "success" if stock_rows > 0 else "no_new_data"
                _write_fetch_log(conn, "stock_news", sid, status, rows_inserted=stock_rows, fetch_date_from=f_from, fetch_date_to=f_to, duration_ms=duration_ms)

                if i % 10 == 0:
                    logger.info(f"進度：{i}/{len(stock_ids)}（累計寫入 {total_rows} 筆）")
                    
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                flog.record(stock_id=sid, error=str(e))
                _write_fetch_log(conn, "stock_news", sid, "failed", fetch_date_from=f_from, fetch_date_to=f_to, duration_ms=duration_ms, error_message=str(e))

        flog.summary()
        logger.info(f"=== [stock_news] 完成，共寫入 {total_rows} 筆 ===")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
