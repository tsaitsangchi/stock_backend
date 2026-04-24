import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime

import psycopg2
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
FINMIND_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ"
    ".muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"
)

# ======================
# PostgreSQL 連線設定
# ======================
DB_CONFIG = {
    "dbname": "stock",
    "user": "stock",
    "password": "stock",
    "host": "localhost",
    "port": "5432",
}

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

def finmind_get(dataset: str, data_id: str = None, start: str = None, end: str = None):
    params = {
        "dataset": dataset,
        "token": FINMIND_TOKEN,
    }
    if data_id:
        params["data_id"] = data_id
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end

    try:
        res = requests.get(FINMIND_API_URL, params=params, timeout=30)
        res.raise_for_status()
        data = res.json().get("data", [])
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"API 請求失敗 ({dataset}): {e}")
        return pd.DataFrame()

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
