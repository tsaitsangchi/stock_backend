import time
import subprocess
import logging
import sys
import psycopg2
from datetime import datetime, timedelta
from config import STOCK_CONFIGS, OUTPUT_DIR

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "logs" / "auto_train.log"),
    ],
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "dbname": "stock",
    "user": "stock",
    "password": "stock",
    "host": "localhost",
    "port": "5432",
}

# Tables to check for completion
CHECK_TABLES = [
    "stock_price",
    "stock_per",
    "institutional_investors_buy_sell",
    "financial_statements",
    "eight_banks"
]

# We expect data to be up to at least 1 day ago (or today if after 16:00)
def get_target_date():
    now = datetime.now()
    # If today is Monday-Friday and after 16:00, we might expect today's data.
    # For safety, let's say we want data up to yesterday (or last Friday if today is Monday).
    if now.hour >= 17:
        return now.date()
    else:
        return (now - timedelta(days=1)).date()

def is_data_complete(stock_id, target_date):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            for table in CHECK_TABLES:
                cur.execute(f"SELECT MAX(date) FROM {table} WHERE stock_id = %s", (stock_id,))
                max_date = cur.fetchone()[0]
                if not max_date or max_date < target_date:
                    # Special handling for financial_statements which are quarterly
                    if table == "financial_statements":
                        if not max_date or (datetime.now().date() - max_date).days > 120:
                            return False, f"{table} outdated ({max_date})"
                        continue
                    return False, f"{table} outdated ({max_date})"
        conn.close()
        return True, "Complete"
    except Exception as e:
        logger.error(f"Error checking {stock_id}: {e}")
        return False, str(e)

def run_task(cmd):
    logger.info(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode

def main():
    stock_ids = list(STOCK_CONFIGS.keys())
    trained_today = set()
    
    logger.info(f"Auto Train Manager started for {len(stock_ids)} stocks.")
    
    while True:
        target_date = get_target_date()
        logger.info(f"Looping through stocks (Target Date: {target_date})...")
        
        for stock_id in stock_ids:
            if stock_id in trained_today:
                continue
                
            complete, reason = is_data_complete(stock_id, target_date)
            if complete:
                logger.info(f"🚀 Stock {stock_id} data is READY. Starting pipeline...")
                
                # 1. Update Feature Store
                ret = run_task(["./venv/bin/python", "scripts/update_feature_store.py", "--stock-id", stock_id])
                if ret != 0:
                    logger.error(f"Feature Store update failed for {stock_id}")
                    continue
                
                # 2. Train Model
                ret = run_task(["./venv/bin/python", "scripts/train_evaluate.py", "--stock-id", stock_id])
                if ret == 0:
                    logger.info(f"✅ Stock {stock_id} training COMPLETED.")
                    trained_today.add(stock_id)
                    # Broadcast status
                    subprocess.run(["wall", f"✅ [系統通知] 個股 {stock_id} 模型訓練完成！"])
                else:
                    logger.error(f"Training failed for {stock_id}")
            else:
                # logger.debug(f"Stock {stock_id} not ready: {reason}")
                pass
        
        if len(trained_today) == len(stock_ids):
            logger.info("All stocks trained for today. Manager exiting Phase 1.")
            break
            
        logger.info(f"Progress: {len(trained_today)}/{len(stock_ids)}. Sleeping 60s...")
        time.sleep(60)

if __name__ == "__main__":
    main()
