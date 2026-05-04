from __future__ import annotations
import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_news_data.py — 個股相關新聞 v2.2
====================================
[v2.2 標準化]：
  · 導入 safe_commit_rows() 與 dump_failures()。
  · 強化 atomicity：逐日逐股 commit 新聞內容。
  · 失敗清單寫入 outputs/stock_news_failed_{date}.json。
  · 確保 DDL 執行後立即 commit。
"""

import json
import argparse
import logging
from datetime import date, datetime, timedelta

from config import DB_CONFIG  # noqa: F401
from core.finmind_client import finmind_get
from core.db_utils import get_db_conn, ensure_ddl, bulk_upsert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_START = "2018-01-01"

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
DDL_NEWS = """
CREATE TABLE IF NOT EXISTS stock_news (
    date        DATE,
    stock_id    VARCHAR(50),
    title       TEXT,
    description TEXT,
    source      VARCHAR(200),
    link        TEXT,
    PRIMARY KEY (date, stock_id, title)
);
CREATE INDEX IF NOT EXISTS idx_news_stock_date ON stock_news (stock_id, date);
CREATE INDEX IF NOT EXISTS idx_news_date       ON stock_news (date);
"""

UPSERT_NEWS = """
INSERT INTO stock_news (date, stock_id, title, description, source, link)
VALUES %s
ON CONFLICT (date, stock_id, title) DO UPDATE SET
    description = EXCLUDED.description,
    source      = EXCLUDED.source,
    link        = EXCLUDED.link;
"""

def map_news(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        (r.get("title") or "")[:1000],   # 防超長
        r.get("description"),
        r.get("source"),
        r.get("link"),
    )

# ─────────────────────────────────────────────
# 抓取邏輯：API 限單日單股，逐日逐股 loop
# ──────────────────────────────────────────────
# 逐支 commit 工具函式
# ──────────────────────────────────────────────
def safe_commit_rows(conn, upsert_sql: str, rows: list, template: str,
                      label: str = "") -> int:
    if not rows:
        return 0
    try:
        n = bulk_upsert(conn, upsert_sql, rows, template)
        conn.commit()
        return n
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f"  [{label}] 寫入失敗，已 rollback：{e}")
        return 0


def dump_failures(table: str, failures: list) -> None:
    if not failures:
        return
    out = OUTPUT_DIR / f"{table}_failed_{date.today().strftime('%Y%m%d')}.json"
    try:
        out.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"  失敗清單已寫入：{out}（{len(failures)} 筆）")
    except Exception as e:
        logger.warning(f"  寫入失敗清單時發生錯誤：{e}")


# ─────────────────────────────────────────────
# 抓取邏輯：API 限單日單股，逐日逐股 loop
# ─────────────────────────────────────────────
def fetch_news_for_stock(conn, stock_id: str, dates: list[str], delay: float):
    """對單一股票，依 dates 列表逐日抓新聞。"""
    total_rows = 0
    for d in dates:
        try:
            data = finmind_get("TaiwanStockNews",
                               {"data_id": stock_id, "start_date": d}, delay,
                               raise_on_error=True)
            if not data:
                continue
            # 依 PK (date, stock_id, title) 去重
            seen = {}
            for r in data:
                try:
                    row = map_news(r)
                    d_str = str(row[0])[:10]
                    pk = (d_str, row[1], row[2])
                    seen[pk] = row
                except Exception:
                    continue
            rows = list(seen.values())
            if rows:
                n = safe_commit_rows(conn, UPSERT_NEWS, rows, "(%s, %s, %s, %s, %s, %s)", label=f"news/{stock_id}/{d}")
                total_rows += n
        except Exception as e:
            logger.error(f"  [{stock_id}/{d}] 抓取或寫入失敗：{e}")
            # 繼續下一日
    return total_rows

def get_existing_dates(conn, stock_id: str) -> set[str]:
    """已抓過的日期集合（用於增量跳過）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date FROM stock_news WHERE stock_id = %s",
            (stock_id,),
        )
        return {row[0].strftime("%Y-%m-%d") for row in cur.fetchall()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock-id", type=str, default=None,
                   help="指定股票（逗號分隔），未指定則抓 STOCK_CONFIGS 全部")
    p.add_argument("--days", type=int, default=30,
                   help="抓取最近 N 天（預設 30）")
    p.add_argument("--start", type=str, default=None,
                   help="若指定，覆蓋 --days 改為從此日期起算")
    p.add_argument("--end",   type=str, default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=0.8)
    p.add_argument("--force", action="store_true",
                   help="強制重抓，不跳過 DB 內已存在日期")
    args = p.parse_args()

    # 決定股票清單
    if args.stock_id:
        stock_ids = [s.strip() for s in args.stock_id.split(",")]
    else:
        from config import STOCK_CONFIGS
        stock_ids = list(STOCK_CONFIGS.keys())

    # 決定日期範圍
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=args.days)
    start_dt = max(start_dt, datetime.strptime(DATASET_START, "%Y-%m-%d"))

    # 產生日期列表（每日一個 API 呼叫）
    n_days = (end_dt - start_dt).days + 1
    all_dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
                 for i in range(n_days)]

    logger.info(f"目標股票：{len(stock_ids)} 支")
    logger.info(f"日期範圍：{all_dates[0]} ~ {all_dates[-1]}（{n_days} 天）")
    logger.info(f"預估 API 次數：{len(stock_ids) * n_days}（單支 × 單日）")

    conn = get_db_conn()
    failures = []
    try:
        ensure_ddl(conn, DDL_NEWS)
        conn.commit()
        for sid in stock_ids:
            try:
                existing = set() if args.force else get_existing_dates(conn, sid)
                todo = [d for d in all_dates if d not in existing]
                if not todo:
                    logger.info(f"[{sid}] 全部日期已存在，跳過")
                    continue
                logger.info(f"[{sid}] 抓取 {len(todo)} 天（已存在 {len(existing)} 天）")
                n = fetch_news_for_stock(conn, sid, todo, args.delay)
                logger.info(f"[{sid}] 累計寫入 {n} 則新聞")
            except Exception as e:
                try: conn.rollback()
                except Exception: pass
                failures.append({"stock_id": sid, "error": str(e)})
                logger.error(f"  [{sid}] 處理中發生錯誤：{e}")

    finally:
        conn.close()
    
    dump_failures("stock_news", failures)
    logger.info("全部完成")

if __name__ == "__main__":
    main()
