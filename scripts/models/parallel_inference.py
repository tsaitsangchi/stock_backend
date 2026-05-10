"""
parallel_inference.py v5.5.26 (Trinity Core Final)
================================================================================
全核心並行預測指揮官 — 混合模式日誌實作版
負責對 150 檔核心標的進行 AI 推論，產出明天的買賣訊號與信心度。

修訂歷程：
  v5.6.1 (2026-05-10):
    - [核心] 整合 ML + TFT 混合預測訊號。
    - [視覺] 自動觸發戰情網頁更新。
  v5.5.26 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。

【執行範例說明】

1. 直接從命令行執行（啟動並行推論）：
   $ python scripts/models/parallel_inference.py

2. 日誌查閱 (追蹤推論成功率與耗時)：
   SELECT task_name, status, rows_processed, duration_ms 
   FROM pipeline_execution_log 
   WHERE category = 'inference' 
   ORDER BY created_at DESC LIMIT 20;

3. 預測結果查閱 (提取明日潛力金股)：
   SELECT stock_id, signal, confidence, pred_price 
   FROM predictions 
   WHERE signal = 'BUY' AND created_at > CURRENT_DATE 
   ORDER BY confidence DESC LIMIT 10;
"""

import sys
import logging
import time
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
    from models.ensemble_model import predict_ensemble, predict_tft
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def ensure_prediction_table():
    sql = "CREATE TABLE IF NOT EXISTS predictions (id SERIAL PRIMARY KEY, stock_id TEXT NOT NULL, pred_price NUMERIC, confidence NUMERIC, signal TEXT, pred_date DATE DEFAULT CURRENT_DATE + INTERVAL '1 day', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
    with db_transaction() as cur: cur.execute(sql)

def save_prediction(res):
    if not res: return
    sql = "INSERT INTO predictions (stock_id, pred_price, confidence, signal) VALUES (%s, %s, %s, %s)"
    with db_transaction() as cur: cur.execute(sql, (res['stock_id'], res['pred_price'], res['confidence'], res['signal']))

def run_full_inference():
    t_start = time.monotonic()
    logger.info("🚀 [Trinity] 啟動大規模混合預測訊號產出 (ML + TFT) (v5.6.1)...")
    ensure_prediction_table()
    active_stocks = get_db_stock_ids()
    if not active_stocks: return
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 同步取得兩套模型的預測
        res_ml = list(executor.map(predict_ensemble, active_stocks))
        res_dl = list(executor.map(predict_tft, active_stocks))
        
        success_count = 0
        for m, d in zip(res_ml, res_dl):
            if m and d:
                # 混合策略：加權平均或信心度加乘
                hybrid_res = {
                    'stock_id': m['stock_id'],
                    'pred_price': round((m['pred_price'] + d['pred_price']) / 2, 2),
                    'confidence': round((m['confidence'] + d['confidence']) / 2, 4),
                    'signal': "BUY" if (m['signal'] == "BUY" or d['signal'] == "BUY") else "HOLD"
                }
                save_prediction(hybrid_res)
                success_count += 1
    
    # 3. 🔴 自動產出/更新數據戰情網頁 (Dashboard)
    try:
        from monitor.data_audit_engine import audit_completeness
        audit_completeness()
    except ImportError:
        logger.warning("⚠️ 無法載入稽核引擎，跳過網頁更新。")

    elapsed_sec = round(time.monotonic() - t_start, 2)
    write_pipeline_log("full_inference_master", "SYSTEM", "success", "sys", int(elapsed_sec * 1000), success_count)
    logger.info(f"🏆 [Master] 全核心混合預測完畢！耗時: {elapsed_sec}s")
    logger.info(f"🌐 戰情網頁已同步更新: monitor/dashboard.html")

if __name__ == "__main__":
    run_full_inference()
