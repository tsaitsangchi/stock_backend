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
fetch_news_data.py v3.2 — 個股相關新聞（核心股票優先版）
================================================================================
v3.2 改進（避免 ETF / 權證 400 雜訊）：
  ★ 預設抓 config.STOCK_CONFIGS 內的 ~150 支核心股票
    （之前是抓 stock_info 全市場 5000+ 標的，包括無新聞資料的 ETF 與權證，
     大量 400 Bad Request 雜訊浪費 API 配額）
  ★ 新增 --all-market 旗標：若需抓全市場（含 ETF / 權證）才走舊行為
  ★ 新增 --from-config 為預設值（保留向後相容）
  ★ --stock-id 可逗號分隔多支：例如 --stock-id 2330,2317,2454

v3.0 沿用：
  · commit_per_stock_per_day：每一天獨立落盤
  · get_existing_dates：增量抓取，跳過已存日期
  · FailureLogger：精準追蹤失敗點，支援斷點續傳

執行：
    python fetch_news_data.py                       # 抓 STOCK_CONFIGS 全部（預設）
    python fetch_news_data.py --stock-id 2330        # 只抓 2330
    python fetch_news_data.py --stock-id 2330,2317   # 抓多支
    python fetch_news_data.py --all-market           # 全市場（不建議，吃配額）
    python fetch_news_data.py --days 60 --force      # 強制重抓最近 60 天
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


def resolve_stock_ids(args, conn) -> list[str]:
    """
    [v3.2] 決定要抓的 stock_ids 清單，優先序：
      1. --stock-id 明確指定（逗號分隔）
      2. --all-market 旗標 → stock_info 全市場（含 ETF / 權證）
      3. 預設 → config.STOCK_CONFIGS 的核心股票
    """
    if args.stock_id:
        sids = [s.strip() for s in args.stock_id.split(",") if s.strip()]
        logger.info(f"[v3.2] 使用 --stock-id 指定的 {len(sids)} 支股票")
        return sids

    if args.all_market:
        sids = get_db_stock_ids(conn)
        logger.warning(
            f"[v3.2] --all-market 模式啟用：將抓 stock_info 全市場 {len(sids)} 支標的 "
            f"（含大量 ETF / 權證，FinMind 會回 400，請確認 API 配額充足）"
        )
        return sids

    sids = list(STOCK_CONFIGS.keys())
    logger.info(
        f"[v3.2] 預設模式：使用 config.STOCK_CONFIGS 的 {len(sids)} 支核心股票"
        f"（避開 ETF / 權證 400 雜訊；如需全市場請加 --all-market）"
    )
    return sids


def main():
    p = argparse.ArgumentParser(description="個股新聞抓取 (v3.2 — 核心股票優先版)")
    p.add_argument("--stock-id", default=None,
                   help="指定股票（逗號分隔多支）；不傳則用 config.STOCK_CONFIGS")
    p.add_argument("--all-market", action="store_true",
                   help="抓 stock_info 全市場（含 ETF / 權證，會有大量 400）")
    p.add_argument("--days", type=int, default=30, help="抓取最近 N 天（預設 30）")
    p.add_argument("--start", default=None, help="起始日 YYYY-MM-DD（覆蓋 --days）")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"), help="結束日（預設今天）")
    p.add_argument("--delay", type=float, default=0.8, help="API 呼叫間隔秒數（預設 0.8）")
    p.add_argument("--force", action="store_true", help="忽略 get_existing_dates，強制重抓")
    args = p.parse_args()

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_NEWS)
        stock_ids = resolve_stock_ids(args, conn)

        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        start_dt = datetime.strptime(args.start, "%Y-%m-%d") if args.start else (end_dt - timedelta(days=args.days))
        all_dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_dt - start_dt).days + 1)]

        logger.info(f"日期範圍：{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}（共 {len(all_dates)} 天）")
        logger.info(f"目標股票數：{len(stock_ids)} 支")

        flog = FailureLogger("stock_news", db_conn=conn)
        total_rows = 0
        skipped_stocks = 0   # 已是最新、無需抓取的股票數

        for i, sid in enumerate(stock_ids, 1):
            try:
                existing = set() if args.force else get_existing_dates(conn, sid)
                todo = [d for d in all_dates if d not in existing]
                if not todo:
                    skipped_stocks += 1
                    continue

                logger.info(f"[{sid}] 抓取 {len(todo)} 天（已跳過 {len(existing)} 天）")
                for d in todo:
                    try:
                        data = finmind_get("TaiwanStockNews", {"data_id": sid, "start_date": d}, args.delay)
                        if data:
                            rows = [map_news(r) for r in data]
                            rows = dedup_rows(rows, (0, 1, 2))  # date, stock_id, title
                            res = commit_per_stock_per_day(conn, UPSERT_NEWS, rows, "(%s, %s, %s, %s, %s, %s)", label_prefix=f"news/{sid}", failure_logger=flog)
                            total_rows += sum(res.values())
                    except Exception as e:
                        flog.record(stock_id=sid, error=f"{d}: {str(e)}")

                if i % 10 == 0:
                    logger.info(f"進度：{i}/{len(stock_ids)}（已寫入 {total_rows} 筆，已最新略過 {skipped_stocks} 支）")
            except Exception as e:
                flog.record(stock_id=sid, error=str(e))

        # ── 收尾摘要 ──
        logger.info("=" * 60)
        logger.info(f"  完成！處理 {len(stock_ids)} 支股票")
        logger.info(f"  寫入 {total_rows:,} 筆新聞")
        logger.info(f"  已最新略過 {skipped_stocks} 支")
        logger.info(f"  失敗 {len(flog)} 筆（詳見 {flog.path}）")
        logger.info("=" * 60)
        flog.summary()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
