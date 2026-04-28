"""
fetch_news_data.py — 個股相關新聞
====================================
[新增] TaiwanStockNews（Free tier，但 API 限制：每次只能抓單日單股）

  資料集：TaiwanStockNews
  範圍：依 FinMind 各源實際提供日期，多數可回溯至 2018 左右
  欄位：date, stock_id, description, link, source, title

  特殊限制：
    - 一次 API 呼叫只能取「單一日期 + 單一 data_id」的新聞
    - 若標的數 × 天數很大，配額會快速消耗
    - Sponsor 配額 6000/hr，建議每日只抓「STOCK_CONFIGS 內的核心標的」
      （約 10-30 支），歷史補抓限制在 30 天內

衍生因子：
  - news_intensity_5d  = 過去 5 日新聞數量（注意力因子）
  - news_intensity_zscore = 對個股自身歷史的 z-score
  - news_source_diversity = 不同新聞源數量（散布度）
  - 進階：本地 NLP 模型對 title + description 做 sentiment scoring
          → news_sentiment_score（需另建 NLP 推論服務）

執行：
    python fetch_news_data.py                              # 預設抓最近 30 天
    python fetch_news_data.py --days 7                     # 最近 7 天
    python fetch_news_data.py --stock-id 2330 --days 90    # 單股回溯 90 天
    python fetch_news_data.py --force --start 2024-01-01   # 強制全量
"""
from __future__ import annotations

import argparse
import logging
import sys
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
# ─────────────────────────────────────────────
def fetch_news_for_stock(conn, stock_id: str, dates: list[str], delay: float):
    """對單一股票，依 dates 列表逐日抓新聞。"""
    total_rows = 0
    for d in dates:
        data = finmind_get("TaiwanStockNews",
                           {"data_id": stock_id, "start_date": d}, delay)
        if not data:
            continue
        # 去重：同日同股同 title 視為同一則
        seen = set()
        rows = []
        for r in data:
            key = (r.get("date"), r.get("stock_id"), r.get("title", "")[:1000])
            if key in seen:
                continue
            seen.add(key)
            rows.append(map_news(r))
        if rows:
            bulk_upsert(conn, UPSERT_NEWS, rows, "(%s, %s, %s, %s, %s, %s)")
            total_rows += len(rows)
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
    try:
        ensure_ddl(conn, DDL_NEWS)
        for sid in stock_ids:
            existing = set() if args.force else get_existing_dates(conn, sid)
            todo = [d for d in all_dates if d not in existing]
            if not todo:
                logger.info(f"[{sid}] 全部日期已存在，跳過")
                continue
            logger.info(f"[{sid}] 抓取 {len(todo)} 天（已存在 {len(existing)} 天）")
            n = fetch_news_for_stock(conn, sid, todo, args.delay)
            logger.info(f"[{sid}] 寫入 {n} 則新聞")
    finally:
        conn.close()
    logger.info("✅ 全部完成")

if __name__ == "__main__":
    main()
