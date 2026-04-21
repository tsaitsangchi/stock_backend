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
from config import STOCK_CONFIGS

# 環境路徑設定
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
MAX_WORKERS = 4  # 推論並行數

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_predict(stock_id: str) -> bool:
    """執行單一標的推論"""
    cmd = [VENV_PYTHON, "scripts/predict.py", "--stock-id", stock_id]
    try:
        # 使用 subprocess 執行，並捕獲輸出
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except Exception as e:
        logger.error(f"❌ {stock_id} 推論失敗: {e}")
        return False

def main():
    start_time = time.time()
    logger.info("=== 🚀 啟動 Antigravity 每日全自動管線 ===")
    
    # 1. 並行批次推論
    stock_ids = list(STOCK_CONFIGS.keys())
    logger.info(f"[Step 1] 執行批次推論 (個股數: {len(stock_ids)}, 並行數: {MAX_WORKERS})")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_sid = {executor.submit(run_predict, sid): sid for sid in stock_ids}
        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            if future.result():
                success_count += 1
                if success_count % 5 == 0:
                    logger.info(f"已完成 {success_count}/{len(stock_ids)} 支個股推論")
    
    logger.info(f"✅ 批次推論完成。成功: {success_count}, 失敗: {len(stock_ids) - success_count}")
    
    # 2. 健康檢查
    logger.info("[Step 2] 執行核心模型健康診斷")
    try:
        subprocess.run([VENV_PYTHON, "scripts/model_health_check.py"], check=True)
    except Exception as e:
        logger.error(f"❌ 健康檢查執行失敗: {e}")
        
    # 3. 投資組合優化
    logger.info("[Step 3] 執行投資組合優化與生成決策報告")
    try:
        subprocess.run([VENV_PYTHON, "scripts/portfolio_optimizer.py", "--budget", "100000"], check=True)
    except Exception as e:
        logger.error(f"❌ 投資組合優化失敗: {e}")
        
    duration = time.time() - start_time
    logger.info(f"=== ✅ 管線執行完畢 (總耗時: {duration/60:.1f} 分鐘) ===")

if __name__ == "__main__":
    main()
