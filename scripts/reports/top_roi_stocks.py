"""
top_roi_stocks.py v5.5.26 (Trinity Core Final)
================================================================================
投資潛力戰報工具 — 核心金股篩選版
負責比對最新預測價格與市場實時收盤價，精算出預期投報率最高的前三名標的。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [核心] 整合 stock_price 與 predictions 資料表，實作自動 ROI 排名。
    - [文檔] 補齊極致詳細的執行範例說明。

【執行範例說明】

1. 直接從命令行執行（產出前三名潛力股戰報）：
   $ python scripts/reports/top_roi_stocks.py

2. SQL 查閱 (手動驗證資料庫中的預測與現價對齊狀況)：
   WITH LatestPrices AS (
       SELECT DISTINCT ON (stock_id) stock_id, close FROM stock_price ORDER BY stock_id, date DESC
   ),
   LatestPreds AS (
       SELECT DISTINCT ON (stock_id) stock_id, pred_price, confidence FROM predictions ORDER BY stock_id, created_at DESC
   )
   SELECT p.stock_id, lp.close as current_price, p.pred_price, 
          ((p.pred_price - lp.close) / lp.close) as expected_return
   FROM LatestPreds p JOIN LatestPrices lp ON p.stock_id = lp.stock_id
   ORDER BY expected_return DESC LIMIT 3;
"""

import sys
import logging
from pathlib import Path

# ── 系統路徑修復 (v3.1) ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
for _sub in ("scripts", "scripts/core"):
    _p = _PROJECT_ROOT / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.db_utils import db_session, write_pipeline_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_top_roi_report():
    logger.info("📊 [Report] 正在精算全核心標的預期投報率排行...")
    
    sql = """
        WITH LatestPrices AS (
            SELECT DISTINCT ON (stock_id) stock_id, close 
            FROM stock_price 
            ORDER BY stock_id, date DESC
        ),
        LatestPredictions AS (
            SELECT DISTINCT ON (stock_id) stock_id, pred_price, confidence, signal
            FROM predictions 
            WHERE created_at > (NOW() - INTERVAL '24 hours')
            ORDER BY stock_id, created_at DESC
        )
        SELECT 
            p.stock_id, 
            lp.close as current_price, 
            p.pred_price, 
            p.confidence,
            ((p.pred_price - lp.close) / lp.close) as expected_return
        FROM LatestPredictions p
        JOIN LatestPrices lp ON p.stock_id = lp.stock_id
        WHERE p.signal = 'BUY'
        ORDER BY expected_return DESC
        LIMIT 3
    """
    
    try:
        with db_session() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                
                print('\n💎 【Trinity 2026 Core - 投資價值潛力前三名】')
                print('=' * 75)
                print(f'排名 | 標代 | 目前價格 | 預測價格 | 預期漲幅 | 信心度')
                print('-' * 75)
                for i, r in enumerate(rows, 1):
                    print(f'{i:02d}   | {r["stock_id"]} | {float(r["current_price"]):>8.2f} | {float(r["pred_price"]):>8.2f} | {r["expected_return"]:>8.2%} | {r["confidence"]:>6.2%}')
                print('=' * 75)
                
        write_pipeline_log("generate_top_roi_report", "SYSTEM", "success", "analysis")
        
    except Exception as e:
        logger.error(f"❌ 報表產出失敗: {e}")
        write_pipeline_log("generate_top_roi_report", "SYSTEM", "failed", "analysis", 0, 0, str(e))

if __name__ == "__main__":
    generate_top_roi_report()
