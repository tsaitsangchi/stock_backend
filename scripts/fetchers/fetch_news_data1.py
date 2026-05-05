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
fetch_news_data.py v3.1 — 個股相關新聞（範圍批次抓取版）
================================================================================
v3.1 速度優化：
  ★ 由「逐股票 × 逐單日」改為「逐股票 × 逐連續區段」批次抓取。
    例如 30 天若都缺，原本要打 30 次 API；現在只打 1 次（FinMind 支援 end_date）。
  ★ 仍保留 get_existing_dates 增量機制；缺失日期會分群成連續區段，
    區段間 gap > MAX_GAP_DAYS 時才切斷，避免重抓已存在的大段日期。
  ★ DB 寫入仍走 commit_per_stock_per_day，逐日原子落盤不變。

v3.0 沿用：
  · commit_per_stock_per_day 確保每一天獨立落盤
  · FailureLogger 追蹤逐支失敗點
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

# ─────────────────────────────────────────────
# 將缺失日期切成連續區段（v3.1）
# ─────────────────────────────────────────────
MAX_GAP_DAYS = 7  # gap 超過 7 天就斷成兩段（避免重抓很大一段已存在資料）

def group_consecutive_dates(dates: list[str], max_gap_days: int = MAX_GAP_DAYS) -> list[tuple[str, str]]:
    """把排序後的日期字串清單切成 [(start, end), ...] 連續區段。"""
    if not dates:
        return []
    sorted_dates = sorted(dates)
    ranges: list[tuple[str, str]] = []
    range_start = sorted_dates[0]
    prev = sorted_dates[0]
    prev_dt = datetime.strptime(prev, "%Y-%m-%d")
    for d in sorted_dates[1:]:
        d_dt = datetime.strptime(d, "%Y-%m-%d")
        if (d_dt - prev_dt).days > max_gap_days:
            ranges.append((range_start, prev))
            range_start = d
        prev = d
        prev_dt = d_dt
    ranges.append((range_start, prev))
    return ranges

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

                # ⭐ v3.1：把缺失日期切成連續區段，每段一次 API ⭐
                ranges = group_consecutive_dates(todo)
                logger.info(
                    f"[{sid}] 抓取 {len(todo)} 天 / 分為 {len(ranges)} 段"
                    f"（已跳過 {len(existing)} 天）"
                )
                for r_start, r_end in ranges:
                    try:
                        data = finmind_get(
                            "TaiwanStockNews",
                            {"data_id": sid, "start_date": r_start, "end_date": r_end},
                            args.delay,
                        )
                        if data:
                            rows = [map_news(r) for r in data]
                            rows = dedup_rows(rows, (0, 1, 2))  # date, stock_id, title
                            res = commit_per_stock_per_day(
                                conn, UPSERT_NEWS, rows,
                                "(%s, %s, %s, %s, %s, %s)",
                                label_prefix=f"news/{sid}", failure_logger=flog,
                            )
                            total_rows += sum(res.values())
                    except Exception as e:
                        flog.record(stock_id=sid, error=f"{r_start}~{r_end}: {str(e)}")

                if i % 10 == 0: logger.info(f"進度：{i}/{len(stock_ids)}  累計寫入 {total_rows} 筆")
            except Exception as e: flog.record(stock_id=sid, error=str(e))
        logger.info(f"[stock_news] 總共寫入 {total_rows} 筆")
        flog.summary()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
