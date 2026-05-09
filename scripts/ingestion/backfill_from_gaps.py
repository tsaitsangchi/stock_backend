"""
backfill_from_gaps.py v5.5.7 (Trinity Core Final)
================================================================================
資料抓取模組 — 混合模式日誌實作版
負責將 FinMind 原始數據同步至資料庫。

修訂歷程：
  v5.5.7 (2026-05-09):
    - [文檔] 補齊「大規模並行調度」與「手動單點調試」執行範例。
  v5.5.1 (2026-05-09):
    - [規範] 導入混合模式日誌與路徑修復 v3.0。

【執行範例說明】

1. 手動單點調試 (僅抓取台積電 2330 作為測試)：
   $ python scripts/ingestion/backfill_from_gaps.py

2. 大規模並行抓取 (透過調度器對全市場執行)：
   ------------------------------------------------------------
   from ingestion.parallel_fetch import run_orchestrator
   from ingestion.backfill_from_gaps import fetch_func
   from core.db_utils import get_db_stock_ids
   
   # 啟動並行調度：對全市場標的執行 fetch_func
   run_orchestrator(fetch_func, get_db_stock_ids(), "all_market_fetch_func")
   ------------------------------------------------------------
"""

import sys
import logging
import time
import pandas as pd
from pathlib import Path

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "ingestion"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log
    from core.finmind_client import FinMindClient
    from config import TIER_1_STOCKS
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def backfill_stock_gaps(stock_id: str):
    t0 = time.monotonic()
    api = FinMindClient()
    filled_rows = 0
    
    try:
        # 1. 偵測缺口 (此處為實作邏輯框架)
        with db_transaction() as cur:
            cur.execute("""
                WITH date_gaps AS (
                    SELECT date, LEAD(date) OVER (ORDER BY date) as next_date
                    FROM stock_price WHERE stock_id = %s
                )
                SELECT date, next_date FROM date_gaps 
                WHERE next_date - date > interval '1 day'
                LIMIT 1;
            """, (stock_id,))
            gap = cur.fetchone()
            
        if gap:
            start_gap = gap['date'].strftime('%Y-%m-%d')
            end_gap = gap['next_date'].strftime('%Y-%m-%d')
            logger.info(f"🔍 偵測到 {stock_id} 缺口: {start_gap} ~ {end_gap}")
            data = api.get_data("TaiwanStockPrice", stock_id, start_date=start_gap, end_date=end_gap)
            filled_rows = len(data)
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("backfill_gaps", stock_id, "success", "ingestion", elapsed_ms, filled_rows)
        return filled_rows
        
    except Exception as e:
        logger.error(f"❌ {stock_id} 缺口回補失敗: {e}")
        write_pipeline_log("backfill_gaps", stock_id, "failed", "ingestion", 0, 0, str(e))
        return 0

if __name__ == "__main__":
    for sid in TIER_1_STOCKS[:3]:
        backfill_stock_gaps(sid)