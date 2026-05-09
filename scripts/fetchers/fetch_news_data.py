"""
fetch_news_data.py v5.1 (Trinity Core Edition)
================================================================================
個股相關新聞抓取器 — 完美對接 core/ 五大核心模組
此模組負責從 FinMind 抓取個股新聞標題、摘要與來源連結，為情緒分析提供資料源。

主要功能：
  · 逐日抓取邏輯   ─ 針對 FinMind 新聞 API 限制，實作穩定且高效的單日輪詢。
  · 非法 ID 過濾   ─ 自動跳過產業代碼 (如 Automobile)，防止觸發斷路器熔斷。
  · 智慧增量補抓   ─ 自動從 fetch_log 推算缺失區間，避免重複抓取消耗配額。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ 402 自動休眠 + 智慧斷路器 (排除業務錯誤)
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 修正 ImportError，移除 finmind_get，全面換裝 FinMindClient().get_data()。
    - [核心] 移除 get_db_conn，改用 db_session 與 db_transaction 確保資料一致性。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，實現精準筆數追蹤。
    - [穩定] 導入 is_valid_stock_id 過濾非數字 ID，避免觸發 422 導致斷路器熔斷。
  v3.4 (2024-05-01):
    - [基礎] 建立基礎新聞抓取邏輯。

執行範例：
    # 範例 1：抓取核心標的最近 30 天新聞 (增量更新)
    python scripts/fetchers/fetch_news_data.py --days 30
    
    # 範例 2：抓取特定台積電新聞
    python scripts/fetchers/fetch_news_data.py --stock-id 2330
    
    # 範例 3：針對全市場個股補抓最近 7 天新聞
    python scripts/fetchers/fetch_news_data.py --all-market --days 7 --workers 3
"""

import sys
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import (
        db_session, db_transaction, ensure_ddl, commit_per_stock_per_day,
        get_latest_date, write_fetch_log, FailureLogger,
        get_db_stock_ids
    )
    from core.finmind_client import FinMindClient, get_request_stats
    from core.migrate_stocks_config import migrate as sync_stocks_table
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. DDL 與 SQL 配置
# =====================================================================

DDL_NEWS = """
CREATE TABLE IF NOT EXISTS stock_news (
    date DATE NOT NULL, 
    stock_id VARCHAR(50) NOT NULL, 
    title TEXT NOT NULL, 
    description TEXT, 
    source VARCHAR(200), 
    link TEXT, 
    PRIMARY KEY (date, stock_id, title)
);
CREATE INDEX IF NOT EXISTS idx_news_stock_date ON stock_news (stock_id, date DESC);
"""

UPSERT_NEWS = """
INSERT INTO stock_news (date, stock_id, title, description, source, link) 
VALUES (%(date)s, %(stock_id)s, %(title)s, %(description)s, %(source)s, %(link)s) 
ON CONFLICT (date, stock_id, title) DO UPDATE SET 
    source = EXCLUDED.source,
    description = EXCLUDED.description;
"""

# =====================================================================
# 2. 工具函式
# =====================================================================

def is_valid_stock_id(sid: str) -> bool:
    """過濾產業指數名稱或無效 ID (如 Automobile, 9955)"""
    return sid.isdigit() and len(sid) >= 4

def taipei_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    except:
        return date.today().strftime("%Y-%m-%d")

def next_day(date_str: str) -> str:
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (d + timedelta(days=1)).strftime("%Y-%m-%d")
    except:
        return str(date_str)

def resolve_stock_ids(args) -> List[str]:
    if args.stock_id:
        return [s.strip() for s in args.stock_id.split(",") if s.strip()]
    
    try:
        with db_session() as conn:
            with conn.cursor() as cur:
                if args.all_market:
                    cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE ORDER BY stock_id")
                else:
                    cur.execute("SELECT stock_id FROM stocks WHERE fetch_news = TRUE AND is_active = TRUE ORDER BY stock_id")
                return [r[0] for r in cur.fetchall()]
    except:
        return []

# =====================================================================
# 3. 核心抓取邏輯 (v5.1 精準監控)
# =====================================================================

def map_news(r: dict) -> dict:
    return {
        "date": r["date"], 
        "stock_id": str(r["stock_id"]), 
        "title": (r.get("title") or "")[:1000], 
        "description": r.get("description"), 
        "source": str(r.get("source") or "")[:200], 
        "link": r.get("link")
    }

def fetch_one_stock(stock_id: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    table = "stock_news"
    
    # 1. 預檢 ID 合法性
    if not is_valid_stock_id(stock_id):
        logger.debug(f"  ⏭️ {stock_id} 為無效代碼，自動跳過。")
        write_fetch_log(table, stock_id, "skipped", "news_v5.1", start, end, 0, 0, "invalid_stock_id")
        return stock_id, 0, 0

    cur_start = start
    if not force:
        latest = get_latest_date(table, stock_id)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, stock_id, "skipped", "news_v5.1", str(latest), end, 0, 0, "up_to_date")
                return stock_id, 0, 0

    try:
        s_dt = datetime.strptime(cur_start, "%Y-%m-%d").date()
        e_dt = datetime.strptime(end, "%Y-%m-%d").date()
        curr_dt = s_dt
        total_success = total_error = 0
        
        while curr_dt <= e_dt:
            s_str = curr_dt.strftime("%Y-%m-%d")
            t0 = time.monotonic()
            
            # ⭐ 逐日抓取：新聞 API 規則為單日查詢
            data = api.get_data("TaiwanStockNews", stock_id, s_str)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [map_news(row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_NEWS, stock_id)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, stock_id, "success" if error == 0 else "partial", "news_v5.1", s_str, s_str, duration_ms, success, None)
            
            curr_dt += timedelta(days=1)
            time.sleep(0.3)

        return stock_id, total_success, total_error

    except Exception as e:
        msg = str(e)
        logger.error(f"  ❌ {stock_id} @ {table}: {msg}")
        status = "circuit_open" if "Circuit Breaker" in msg else "failed"
        write_fetch_log(table, stock_id, status, "news_v5.1", cur_start, end, 0, 0, msg)
        return stock_id, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Stock News Fetcher v5.1 (Trinity Core Edition)")
    parser.add_argument("--stock-id", type=str, default=None)
    parser.add_argument("--all-market", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    try:
        sync_stocks_table()
        ensure_ddl(DDL_NEWS)
    except: pass

    stock_ids = resolve_stock_ids(args)
    if not stock_ids:
        logger.warning("未找到需要抓取新聞的標的。")
        return

    end_date = args.end or taipei_today()
    start_date = args.start
    if not start_date:
        e_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (e_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"  Stock News Fetcher v5.1  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    logger.info(f"  目標個股    : {len(stock_ids)}")
    logger.info(f"  日期區間    : {start_date} ~ {end_date}")
    logger.info(f"  Workers     : {args.workers}")
    logger.info("=" * 70)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one_stock, sid, start_date, end_date, args.force): sid for sid in stock_ids}
        for fut in as_completed(futures):
            sid, ok, err = fut.result()
            if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")
    try:
        get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()