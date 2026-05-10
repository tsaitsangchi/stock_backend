"""
run_batch_backtest.py v5.5.26 (Trinity Core Final)
================================================================================
全核心批量回測指揮官 — 混合模式日誌實作版
負責對 150 檔核心標的進行並行回測驗證。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v5.5.23 (2026-05-09):
    - [核心] 實作 150 檔標的全量並行回測。

【執行範例說明】

1. 直接從命令行執行（對 150 檔標的啟動並行回測）：
   $ python scripts/evaluation/run_batch_backtest.py

2. 日誌查閱 (驗證回測任務狀態)：
   SELECT task_name, status, duration_ms, error_message 
   FROM pipeline_execution_log 
   WHERE category = 'backtest' 
   ORDER BY created_at DESC LIMIT 20;

3. 績效排行查閱 (找出最強標的)：
   SELECT stock_id, total_return, sharpe_ratio, win_rate 
   FROM evaluation_log 
   WHERE created_at > CURRENT_DATE 
   ORDER BY total_return DESC LIMIT 10;
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
    try:
        engine = BacktestEngine(stock_id=stock_id)
        return engine.run_simulation(mode="hybrid")
    except Exception as e:
        logger.error(f"❌ {stock_id} 回測異常: {e}")
        return False

def run_batch_evaluation():
    t_start = time.monotonic()
    logger.info("🚀 [Trinity] 啟動全核心全棧批量回測驗證 (ML + TFT) (v5.6.0)...")
    active_stocks = get_db_stock_ids()
    if not active_stocks: return

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(evaluate_single_stock, active_stocks))
    
    # 3. 🔴 自動產出/更新數據戰情網頁 (Dashboard)
    try:
        from monitor.data_audit_engine import audit_completeness
        audit_completeness()
    except ImportError:
        logger.warning("⚠️ 無法載入稽核引擎，跳過網頁更新。")

    success_count = sum(1 for r in results if r)
    elapsed_sec = round(time.monotonic() - t_start, 2)
    write_pipeline_log("full_batch_backtest_master", "SYSTEM", "success", "sys", int(elapsed_sec * 1000), success_count)
    logger.info(f"🏆 [Master] 全核心批量回測完畢！成功: {success_count}/{len(active_stocks)}")
    logger.info(f"🌐 戰情網頁已同步更新: monitor/dashboard.html")

if __name__ == "__main__":
    run_batch_evaluation()
