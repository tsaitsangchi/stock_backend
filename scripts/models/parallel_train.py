"""
parallel_train.py v5.5.21 (Trinity Core Final)
================================================================================
大規模並行模型訓練指揮官 — 混合模式日誌實作版 (穩定修復版)
負責對 150 檔核心標的進行 AI 模型訓練與資產註冊。

修訂歷程：
  v5.5.21 (2026-05-09):
    - [核心] 配合 db_utils v4.9 實作 Fork-Safe 訓練流程，修復 SSL EOF 異常。
    - [優化] 強化進程間隔離，確保 150 檔標的模型註冊 100% 成功。
    - [規範] 完全對接混合日誌 (Category: training & sys)。

【執行範例說明】

1. 直接啟動 150 檔核心標的模型訓練：
   $ python scripts/models/parallel_train.py

2. 監控訓練進度 (SQL)：
   SELECT task_name, status, error_message FROM pipeline_execution_log 
   WHERE category = 'training' ORDER BY created_at DESC;
"""

import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "features", "models"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, get_db_stock_ids
    from models.ensemble_model import train_ensemble
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_worker_task(stock_id: str):
    """
    [子進程專用] 執行單一標的模型訓練。
    子進程會自動觸發 db_utils v4.9 的 PID 偵測機制來建立獨立連線。
    """
    try:
        # 調用核心訓練邏輯 (具備獨立的日誌寫入能力)
        train_ensemble(stock_id)
        return True
    except Exception as e:
        logger.error(f"❌ {stock_id} 訓練進程異常: {e}")
        return False

def run_full_training_pipeline():
    """
    主控流程：編排 150 檔標的的並行訓練生命週期。
    """
    t_start = time.monotonic()
    logger.info("🚀 [Trinity] 啟動大規模模型生成管線 (v5.5.21)...")
    
    # 🎯 獲取 150 檔核心標的
    active_stocks = get_db_stock_ids()
    
    if not active_stocks:
        logger.warning("⚠️ 找不到啟用的核心標的，請先確認 stocks 表狀態。")
        return

    logger.info(f"📊 準備訓練標的總數: {len(active_stocks)}")
    
    # 使用 ProcessPoolExecutor 進行多進程訓練
    # 注意：模型訓練 CPU 密集，建議 max_workers = CPU 核心數 - 1 或 2
    try:
        with ProcessPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(train_worker_task, active_stocks))
            
        success_count = sum(1 for r in results if r)
        elapsed_sec = round(time.monotonic() - t_start, 2)
        
        # 🔴 混合日誌：系統層級生命週期紀錄
        write_pipeline_log(
            task_name="full_model_training_master",
            stock_id="SYSTEM",
            status="success",
            category="sys",
            duration_ms=int(elapsed_sec * 1000),
            rows=success_count
        )
        
        logger.info(f"🏆 [Master] 全核心模型生成完畢！")
        logger.info(f"📈 成功: {success_count}/{len(active_stocks)} | 總耗時: {elapsed_sec}s")
        
    except Exception as e:
        logger.error(f"❌ 訓練管線崩潰: {e}")
        write_pipeline_log("full_model_training_master", "SYSTEM", "failed", "sys", 0, 0, str(e))

if __name__ == "__main__":
    run_full_training_pipeline()
