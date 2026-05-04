import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_news_data.py v3.0 — 個股相關新聞（逐支逐日 commit 完整性版）
================================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：針對新聞這種高頻且易逾時的資料，每一天獨立落盤。
  ★ 增量優化：保留 get_existing_dates 機制，僅抓取資料庫缺失的日期，最小化 API 消耗。
  ★ 全面整合 FailureLogger：精準追蹤 150 支股票在漫長抓取過程中的失敗點，支援斷點續傳。
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DDL_NEWS = """
CREATE TABLE IF NOT EXISTS stock_news (date DATE, stock_id VARCHAR(50), title TEXT, description TEXT, source VARCHAR(200), link TEXT, PRIMARY KEY (date, stock_id, title));
CREATE INDEX IF NOT EXISTS idx_news_stock_date ON stock_news (stock_id, date);
"""
UPSERT_NEWS = """INSERT INTO stock_news (date, stock_id, title, description, source, link) VALUES %s ON CONFLICT (date, stock_id, title) DO UPDATE SET source = EXCLUDED.source;"""

def map_news(r: dict) -> tuple:
    return (r["date"], r["stock_id"], (r.get("title") or "")[:1000], r.get("description"), r.get("source"), r.get("link"))

def get_existing_dates(conn, stock_id: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT date FROM stock_news WHERE stock_id = %s", (stock_id,))
        return {row[0].strftime("%Y-%m-%d") for row in cur.fetchall()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock-id", default=None)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=0.8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_NEWS)
        stock_ids = [s.strip() for s in args.stock_id.split(",")] if args.stock_id else get_db_stock_ids(conn)
        
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        start_dt = datetime.strptime(args.start, "%Y-%m-%d") if args.start else (end_dt - timedelta(days=args.days))
        all_dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_dt - start_dt).days + 1)]
        
        flog = FailureLogger("stock_news", db_conn=conn)
        total_rows = 0
        
        for i, sid in enumerate(stock_ids, 1):
            try:
                existing = set() if args.force else get_existing_dates(conn, sid)
                todo = [d for d in all_dates if d not in existing]
                if not todo: continue
                
                logger.info(f"[{sid}] 抓取 {len(todo)} 天（已跳過 {len(existing)} 天）")
                for d in todo:
                    try:
                        data = finmind_get("TaiwanStockNews", {"data_id": sid, "start_date": d}, args.delay)
                        if data:
                            rows = [map_news(r) for r in data]
                            rows = dedup_rows(rows, (0, 1, 2)) # date, stock_id, title
                            res = commit_per_stock_per_day(conn, UPSERT_NEWS, rows, "(%s, %s, %s, %s, %s, %s)", label_prefix=f"news/{sid}", failure_logger=flog)
                            total_rows += sum(res.values())
                    except Exception as e: flog.record(stock_id=sid, error=f"{d}: {str(e)}")
                
                if i % 10 == 0: logger.info(f"進度：{i}/{len(stock_ids)}")
            except Exception as e: flog.record(stock_id=sid, error=str(e))
        flog.summary()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
