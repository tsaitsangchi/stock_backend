import sys
import subprocess
import logging
from pathlib import Path

# ── 注入路徑 ──────────────────────────────────────
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: 
    sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))

from config import STOCK_CONFIGS

# 路徑設定
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
PREDICT_SCRIPT = SCRIPTS_DIR / "training" / "predict.py"
MODEL_DIR = SCRIPTS_DIR / "outputs" / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BatchPredict")

def main():
    logger.info("=== 啟動全系統 80 檔標的預測補齊任務 ===")
    
    stock_ids = list(STOCK_CONFIGS.keys())
    total = len(stock_ids)
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, sid in enumerate(stock_ids, 1):
        model_path = MODEL_DIR / f"ensemble_{sid}.pkl"
        
        if not model_path.exists():
            logger.warning(f"[{i}/{total}] {sid} 找不到模型，跳過。")
            skip_count += 1
            continue
            
        logger.info(f"[{i}/{total}] 正在執行 {sid} 預測...")
        
        try:
            # 執行 predict.py
            cmd = [VENV_PYTHON, str(PREDICT_SCRIPT), "--stock-id", sid]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"✅ {sid} 預測完成。")
                success_count += 1
            else:
                logger.error(f"❌ {sid} 預測失敗！ Error: {result.stderr.strip()}")
                fail_count += 1
                
        except Exception as e:
            logger.error(f"💥 {sid} 執行時發生異常: {e}")
            fail_count += 1

    logger.info("============================================================")
    logger.info(f"任務結束。成功: {success_count} | 失敗: {fail_count} | 跳過: {skip_count}")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
