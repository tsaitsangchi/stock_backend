"""
run_feature_engineering.py v5.5.16 (Trinity Core Final)
================================================================================
特徵工程執行指揮官 — 混合模式日誌實作版
負責對 150 檔核心標的進行大規模特徵矩陣精煉，支援並行運算與生命週期追蹤。

修訂歷程：
  v5.5.16 (2026-05-09):
    - [修復] 修正 SyntaxError 轉義字元異常。
    - [核心] 實作 150 檔標的全量並行特徵生成邏輯。
    - [規範] 完全對接 db_utils v4.8 與混合日誌系統 (Category: feature)。

【執行範例說明】

1. 直接啟動全核心標的特徵生成 (預設 150 檔)：
   $ python scripts/features/run_feature_engineering.py

2. 查詢特徵生成日誌：
   SELECT * FROM pipeline_execution_log WHERE category = 'feature' ORDER BY created_at DESC;
"""

import sys
import logging
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "features"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, write_pipeline_log, get_db_stock_ids
    from features.feature_engineering import build_features
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def process_single_stock(stock_id: str):
    """
    對單一標的執行特徵精煉的核心邏輯。
    """
    t0 = time.monotonic()
    try:
        # 1. 從資料庫讀取最新量價資料 (僅讀取必要之最近 500 筆)
        with db_transaction() as cur:
            sql = "SELECT date, open, max, min, close, trading_volume FROM stock_price WHERE stock_id = %s ORDER BY date DESC LIMIT 500"
            cur.execute(sql, (stock_id,))
            rows = cur.fetchall()
            if not rows:
                return 0
            
            # 轉換為 DataFrame 並按日期正序排列
            df = pd.DataFrame(rows).sort_values("date")
        
        # 2. 調用核心特徵構建函式
        logger.info(f"🧬 正在精煉 {stock_id} 特徵矩陣...")
        feat_df = build_features(df, stock_id, for_inference=True)
        
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        
        # 🔴 混合日誌紀錄 (單一任務紀錄)
        write_pipeline_log(
            task_name="feature_generation_unit",
            stock_id=stock_id,
            status="success",
            category="feature",
            duration_ms=elapsed_ms,
            rows=len(feat_df)
        )
        return len(feat_df)
        
    except Exception as e:
        logger.error(f"❌ {stock_id} 特徵精煉失敗: {e}")
        write_pipeline_log("feature_generation_unit", stock_id, "failed", "feature", 0, 0, str(e))
        return 0

def run_feature_pipeline():
    """
    主控流程：並行處理 150 檔標的。
    """
    t_start = time.monotonic()
    logger.info("🚀 [Trinity] 啟動全核心標的特徵精煉矩陣 (v5.5.16)...")
    
    # 🎯 自動獲取當前資料庫中啟用的 150 檔核心標的
    active_stocks = get_db_stock_ids()
    
    if not active_stocks:
        logger.warning("⚠️ 找不到任何啟用的核心標的，請確認 stocks 表狀態。")
        return

    logger.info(f"📊 準備計算標的數: {len(active_stocks)}")
    
    total_rows = 0
    # 使用 ThreadPoolExecutor 進行並行運算 (依 CPU 核心調整 max_workers)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_stock, active_stocks))
        total_rows = sum(results)
    
    elapsed_sec = round(time.monotonic() - t_start, 2)
    
    # 🔴 混合日誌紀錄 (總管線紀錄)
    write_pipeline_log(
        task_name="full_feature_sync_master",
        stock_id="SYSTEM",
        status="success",
        category="sys",
        duration_ms=int(elapsed_sec * 1000),
        rows=total_rows
    )
    
    logger.info(f"🏆 [Master] 全核心特徵精煉完成！處理標的: {len(active_stocks)}, 總處理特徵筆數: {total_rows}, 耗時: {elapsed_sec}s")

if __name__ == "__main__":
    run_feature_pipeline()
