"""
backfill_from_gaps.py v5.5.13 (Trinity Core Final)
================================================================================
資料斷層自癒引擎 — 混合模式日誌實作版 (修復 DictCursor 兼容性)

修訂歷程：
  v5.5.13 (2026-05-09):
    - [修復] 修正 KeyError: 0 異常 (對接 DictCursor)。
    - [優化] 強化 SQL 欄位別名，確保資料讀取穩定。
"""

import sys
import logging
import time
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
    from core.db_utils import db_transaction, write_pipeline_log, get_db_stock_ids
    from core.finmind_client import FinMindClient
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fetch_func(stock_id: str):
    t0 = time.monotonic()
    api = FinMindClient()
    total_filled = 0
    
    try:
        # 1. 偵測資料缺口
        with db_transaction() as cur:
            cur.execute("""
                WITH date_range AS (
                    SELECT MIN(date) as start_date, MAX(date) as end_date
                    FROM stock_price WHERE stock_id = %s
                ),
                expected_dates AS (
                    SELECT generate_series(start_date, end_date, '1 day'::interval)::date as missing_date
                    FROM date_range
                )
                SELECT e.missing_date 
                FROM expected_dates e
                LEFT JOIN stock_price s ON e.missing_date = s.date AND s.stock_id = %s
                WHERE s.date IS NULL
                ORDER BY e.missing_date
            """, (stock_id, stock_id))
            
            # 使用 dict 方式讀取欄位，避免 KeyError: 0
            rows = cur.fetchall()
            missing_dates = [r['missing_date'] for r in rows] if rows else []
            
        if missing_dates:
            start_sync = missing_dates[0].strftime('%Y-%m-%d')
            logger.info(f"🔍 偵測到 {stock_id} 存在 {len(missing_dates)} 天斷層，啟動回補 (Since: {start_sync})...")
            
            data = api.get_data("TaiwanStockPrice", stock_id, start_date=start_sync)
            total_filled = len(data)
            
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        write_pipeline_log("backfill_gap_unit", stock_id, "success", "ingestion", elapsed_ms, total_filled)
        return total_filled
        
    except Exception as e:
        logger.error(f"❌ {stock_id} 斷層修復失敗: {e}")
        write_pipeline_log("backfill_gap_unit", stock_id, "failed", "ingestion", 0, 0, str(e))
        return 0

if __name__ == "__main__":
    from ingestion.parallel_fetch import run_orchestrator
    active_stocks = get_db_stock_ids()
    if active_stocks:
        logger.info(f"🚀 [Trinity] 啟動全核心斷層掃描與自癒 (Targets: {len(active_stocks)})")
        run_orchestrator(task_func=fetch_func, stock_ids=active_stocks, task_label="core_gap_backfill", workers=4)