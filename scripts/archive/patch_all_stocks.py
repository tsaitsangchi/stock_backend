import sys
import os
import logging
# 修正路徑以確保能讀取到 config
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.config import STOCK_CONFIGS

# 設定路徑與環境
VENV_PYTHON = "/home/hugo/project/stock_backend/venv/bin/python3"
BASE_DIR = "/home/hugo/project/stock_backend"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_command(cmd):
    logger.info(f"執行指令: {cmd}")
    res = os.system(f"cd {BASE_DIR} && {cmd}")
    return res == 0

def main():
    # 取得所有待補齊的股票 ID（排除台積電 2330、鴻海 2317、聯發科 2454，因為已經處理過）
    skip_ids = ["2330", "2317", "2454"]
    target_ids = [sid for sid in STOCK_CONFIGS.keys() if sid not in skip_ids]
    
    logger.info(f"=== 開始集體補丁作業，共 {len(target_ids)} 支股票 ===")
    
    for i, sid in enumerate(target_ids, 1):
        logger.info(f"[{i}/{len(target_ids)}] 處理中: {sid} ({STOCK_CONFIGS[sid]['name']})")
        
        # 1. 補齊技術面與 PER (自 2005)
        cmd_tech = f"{VENV_PYTHON} scripts/fetch_technical_data.py --stock-id {sid} --start 2005-01-01 --force"
        if not run_command(cmd_tech):
            logger.error(f"{sid} 技術面補丁失敗，跳過下一步")
            continue
            
        # 2. 補齊基本面與營收 (自 2002)
        cmd_fund = f"{VENV_PYTHON} scripts/fetch_fundamental_data.py --stock-id {sid} --start 2002-01-01 --force"
        if not run_command(cmd_fund):
            logger.error(f"{sid} 基本面補丁失敗")
            continue
            
        logger.info(f"{sid} 資料修復完成。")

    logger.info("=== 集體補丁作業結束 ===")

if __name__ == "__main__":
    main()
