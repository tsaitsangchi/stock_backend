import subprocess
import sys
import logging
import time
from pathlib import Path

# Setup paths
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from config import STOCK_CONFIGS, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

VENV_PYTHON = str(base_dir.parent / "venv" / "bin" / "python3")
TUNE_SCRIPT = str(base_dir / "training" / "tune_hyperparameters.py")

def main():
    logger.info("=== 啟動全系統超參數批次調優 (Batch Tuning) ===")
    
    # 獲取所有標的 (也可以根據優先級排序)
    all_sids = list(STOCK_CONFIGS.keys())
    total = len(all_sids)
    
    success_count = 0
    fail_count = 0
    
    for i, sid in enumerate(all_sids):
        name = STOCK_CONFIGS[sid].get("name", "Unknown")
        logger.info(f"[{i+1}/{total}] 正在為 {sid} ({name}) 進行調優...")
        
        log_file = OUTPUT_DIR / f"tune_{sid}.log"
        
        start_time = time.time()
        try:
            # 循序執行以防止 GPU OOM
            with open(log_file, "w") as f:
                subprocess.run(
                    [VENV_PYTHON, TUNE_SCRIPT, "--stock-id", sid],
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            duration = time.time() - start_time
            logger.info(f"✅ [{sid}] 調優完成！耗時: {duration/60:.1f} 分鐘")
            success_count += 1
        except subprocess.CalledProcessError:
            logger.error(f"❌ [{sid}] 調優失敗，請查看日誌: {log_file}")
            fail_count += 1
        except Exception as e:
            logger.error(f"❌ [{sid}] 發生意外錯誤: {e}")
            fail_count += 1
            
        # 緩衝一下，釋放顯存
        time.sleep(5)
        
    logger.info(f"=== 調優任務結束 | 成功: {success_count} | 失敗: {fail_count} ===")

if __name__ == "__main__":
    main()
