"""
parallel_inference.py v5.5.24 (Trinity Core Final)
================================================================================
全核心並行預測指揮官 — 混合模式日誌實作版
負責對 150 檔核心標的進行 AI 推論，產出明天的買賣訊號與信心度。

修訂歷程：
  v5.5.24 (2026-05-09):
    - [核心] 實作 150 檔標的全量並行推論。
    - [數據] 自動建立 predictions 資料表，儲存每日預測結果。
    - [規範] 完全對接混合日誌 (Category: inference & sys)。
"""

import sys
import logging
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── 系統路徑修復 (v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "models"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import write_pipeline_log, get_db_stock_ids, db_transaction
    from models.ensemble_model import predict_ensemble
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def ensure_prediction_table():
    """
    確保預測結果表存在。
    """
    sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            stock_id TEXT NOT NULL,
            pred_price NUMERIC,
            confidence NUMERIC,
            signal TEXT,
            pred_date DATE DEFAULT CURRENT_DATE + INTERVAL '1 day',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    with db_transaction() as cur:
        cur.execute(sql)

def save_prediction(res):
    """
    將單一預測結果存入資料庫。
    """
    if not res: return
    sql = """
        INSERT INTO predictions (stock_id, pred_price, confidence, signal)
        VALUES (%s, %s, %s, %s)
    """
    with db_transaction() as cur:
        cur.execute(sql, (res['stock_id'], res['pred_price'], res['confidence'], res['signal']))

def run_full_inference():
    t_start = time.monotonic()
    logger.info("🚀 [Trinity] 啟動大規模預測訊號產出任務 (v5.5.24)...")
    
    # 🎯 初始化環境與標的
    ensure_prediction_table()
    active_stocks = get_db_stock_ids()
    
    if not active_stocks:
        logger.warning("⚠️ 找不到啟用的核心標的。")
        return

    logger.info(f"📊 準備推論標的總數: {len(active_stocks)}")
    
    # 使用 ThreadPool 進行並行推論
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 1. 執行推論
        results = list(executor.map(predict_ensemble, active_stocks))
        
        # 2. 批量落盤
        success_count = 0
        for res in results:
            if res:
                save_prediction(res)
                success_count += 1
    
    elapsed_sec = round(time.monotonic() - t_start, 2)
    
    # 🔴 總體日誌紀錄 (Category: sys)
    write_pipeline_log(
        task_name="full_inference_master",
        stock_id="SYSTEM",
        status="success",
        category="sys",
        duration_ms=int(elapsed_sec * 1000),
        rows=success_count
    )
    
    logger.info(f"🏆 [Master] 全核心預測產出完畢！成功: {success_count}/{len(active_stocks)}, 耗時: {elapsed_sec}s")

if __name__ == "__main__":
    run_full_inference()
