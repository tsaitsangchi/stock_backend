import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
import sys
from pathlib import Path
import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime

import psycopg2

from config import FINMIND_TOKEN, DB_CONFIG
import psycopg2.extras
import requests
import pandas as pd

# ======================
# 設定 logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ======================
# FinMind API 設定
# ======================
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

# ======================
# PostgreSQL 連線設定
# ======================
# ======================
# DDL
# ======================
DDL_TOTAL_RETURN_INDEX = """
CREATE TABLE IF NOT EXISTS total_return_index (
    date       DATE,
    stock_id   VARCHAR(50),
    price      NUMERIC(20,4),
    PRIMARY KEY (date, stock_id)
);
"""

# [P0 重構] 改用共享 finmind_get（指數退避、402 等待）並轉成 DataFrame
from core.finmind_client import finmind_get as _core_finmind_get


def finmind_get(dataset: str, data_id: str = None, start: str = None, end: str = None):
    """本檔包裝器：保留原 (data_id, start, end) 介面與 DataFrame 回傳值，
    內部請求改走 core.finmind_client.finmind_get（具完整重試與配額保護）。"""
    params = {}
    if data_id:
        params["data_id"] = data_id
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end
    data = _core_finmind_get(dataset, params)
    return pd.DataFrame(data)

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(DDL_TOTAL_RETURN_INDEX)
        conn.commit()

        # 抓取 TAIEX 與 TPEx 報酬指數
        for data_id in ["TAIEX", "TPEx"]:
            logger.info(f"正在抓取 {data_id} 報酬指數...")
            
            # 取得 DB 現有最新日期
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(date) FROM total_return_index WHERE stock_id = %s", (data_id,))
                latest = cur.fetchone()[0]
                
            start = (latest + timedelta(days=1)).strftime("%Y-%m-%d") if latest else "2010-01-01"
            
            df = finmind_get("TaiwanStockTotalReturnIndex", data_id=data_id, start=start)
            
            if df.empty:
                logger.info(f"{data_id} 無新資料。")
                continue
                
            # 寫入 DB
            records = []
            for _, row in df.iterrows():
                records.append((row["date"], row["stock_id"], row["price"]))
                
            sql = """
                INSERT INTO total_return_index (date, stock_id, price)
                VALUES %s
                ON CONFLICT (date, stock_id) DO UPDATE SET price = EXCLUDED.price;
            """
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, sql, records)
            conn.commit()
            logger.info(f"已更新 {data_id} 共 {len(records)} 筆資料。")
            time.sleep(1.2)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
