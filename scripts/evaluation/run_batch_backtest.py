"""
run_batch_backtest.py v5.5.23 (Trinity Core Final)
================================================================================
全核心批量回測指揮官 — 混合模式日誌實作版 (架構對接版)
負責對 150 檔核心標的進行並行回測驗證。

修訂歷程：
  v5.5.23 (2026-05-09):
    - [修復] 對接 BacktestEngine 類別架構。
    - [核心] 實作 150 檔標的全量並行回測。
    - [規範] 完全對接雙層日誌 (pipeline_log + evaluation_log)。
"""

import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "evaluation"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, get_db_stock_ids
    from evaluation.backtest_engine import BacktestEngine
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_single_stock(stock_id: str):
    """
    執行單一標的的回測模擬。
    """
    try:
        engine = BacktestEngine(stock_id=stock_id)
        success = engine.run_simulation()
        return success
    except Exception as e:
        logger.error(f"❌ {stock_id} 回測異常: {e}")
        return False

def run_batch_evaluation():
    t_start = time.monotonic()
    logger.info("🚀 [Trinity] 啟動全核心批量回測驗證 (v5.5.23)...")
    
    # 🎯 獲取 150 檔核心標的
    active_stocks = get_db_stock_ids()
    
    if not active_stocks:
        logger.warning("⚠️ 找不到啟用的核心標的。")
        return

    logger.info(f"📊 準備驗證標的總數: {len(active_stocks)}")
    
    # 使用 ThreadPool 進行並行回測 (I/O 密集型)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(evaluate_single_stock, active_stocks))
    
    success_count = sum(1 for r in results if r)
    elapsed_sec = round(time.monotonic() - t_start, 2)
    
    # 🔴 總體日誌紀錄 (生命週期總結)
    write_pipeline_log(
        task_name="full_batch_backtest_master",
        stock_id="SYSTEM",
        status="success",
        category="sys",
        duration_ms=int(elapsed_sec * 1000),
        rows=success_count
    )
    
    logger.info(f"🏆 [Master] 全核心批量回測完畢！")
    logger.info(f"📈 成功率: {success_count}/{len(active_stocks)} | 總耗時: {elapsed_sec}s")

if __name__ == "__main__":
    run_batch_evaluation()
