"""
scripts/automate_daily.py
全自動化生產管線整合器 v1.0

市場邏輯：
量化交易的生命力在於「週期性」與「一致性」。
本腳本將推論、監控與優化三大核心環節串聯，確保每日決策皆基於：
1. 完整的最新數據。
2. 經過驗證的模型健康度。
3. 全市場寬度感知的配置邏輯。
"""

import subprocess
import sys
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import STOCK_CONFIGS, SYSTEM_STABILITY_CONFIG

# 環境路徑設定
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
MAX_WORKERS = 4  # 推論並行數

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_predict(stock_id: str) -> dict:
    """
    執行單一標的推論，包含異常處理與降級機制。
    回傳：{'success': bool, 'stock_id': str, 'error': str}
    """
    timeout = SYSTEM_STABILITY_CONFIG.get("inference_timeout", 45)
    cmd = [VENV_PYTHON, "scripts/predict.py", "--stock-id", stock_id]
    
    try:
        # 使用 subprocess 執行，並設定超時限制
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        return {"success": True, "stock_id": stock_id}
    except subprocess.TimeoutExpired:
        logger.error(f"⌛ {stock_id} 推論超時 (>{timeout}s)，啟動降級機制...")
        # 降級方案：嘗試不帶 TFT 執行快速推論
        try:
            cmd_fast = cmd + ["--no-tft"]
            subprocess.run(cmd_fast, capture_output=True, text=True, check=True, timeout=15)
            return {"success": True, "stock_id": stock_id, "warning": "TFT_TIMEOUT_FALLBACK"}
        except Exception as e:
            logger.error(f"❌ {stock_id} 快速降級推論亦失敗: {e}")
            return {"success": False, "stock_id": stock_id, "error": "Inference Timeout"}
    except Exception as e:
        logger.error(f"❌ {stock_id} 推論發生未知錯誤: {e}")
        return {"success": False, "stock_id": stock_id, "error": str(e)}

def main():
    start_time = time.time()
    logger.info("=== 🚀 啟動 Antigravity 每日全自動管線 (穩定性強化版) ===")
    
    # 1. 並行批次推論
    stock_ids = list(STOCK_CONFIGS.keys())
    logger.info(f"[Step 1] 執行批次推論 (個股數: {len(stock_ids)}, 並行數: {MAX_WORKERS})")
    
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_sid = {executor.submit(run_predict, sid): sid for sid in stock_ids}
        for future in as_completed(future_to_sid):
            results.append(future.result())
            success_so_far = sum(1 for r in results if r["success"])
            if len(results) % 10 == 0:
                logger.info(f"進度：{len(results)}/{len(stock_ids)} (成功: {success_so_far})")
    
    success_count = sum(1 for r in results if r["success"])
    failed_stocks = [r["stock_id"] for r in results if not r["success"]]
    
    logger.info(f"✅ 批次推論完成。成功: {success_count}, 失敗: {len(stock_ids) - success_count}")
    
    if failed_stocks:
        logger.warning(f"⚠️ 以下標的推論失敗，將在組合優化階段標記為 neutral (0.5)：{failed_stocks}")
    
    # 2. 健康檢查
    logger.info("[Step 2] 執行核心模型健康診斷")
    try:
        subprocess.run([VENV_PYTHON, "scripts/model_health_check.py"], check=True)
    except Exception as e:
        logger.error(f"❌ 健康檢查執行失敗（可能資料庫連線異常）: {e}")
        
    # 3. 投資組合優化
    logger.info("[Step 3] 執行投資組合優化與生成決策報告")
    try:
        # 優化器會讀取 predict_date 當日預測，缺失標的將自動降級處理
        subprocess.run([VENV_PYTHON, "scripts/portfolio_optimizer.py", "--budget", "100000"], check=True)
    except Exception as e:
        logger.error(f"❌ 投資組合優化失敗: {e}")

    # 4. 刷新資料庫物化視圖 (Materialized Views)
    # [P1 修復 2.8] 確保每日抓取與推論後，視圖能反映最新數據
    logger.info("[Step 4] 刷新資料庫物化視圖 (PostgreSQL Optimization)")
    try:
        subprocess.run([VENV_PYTHON, "scripts/db_optimize.py", "--refresh-only"], check=True)
        logger.info("✅ 物化視圖刷新完成。")
    except Exception as e:
        logger.warning(f"⚠️ 物化視圖刷新失敗（不影響主要管線）: {e}")
        
    duration = time.time() - start_time
    logger.info(f"=== ✅ 管線執行完畢 (總耗時: {duration/60:.1f} 分鐘) ===")

if __name__ == "__main__":
    main()
