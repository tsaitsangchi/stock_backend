import logging
import argparse
import sys
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
import psycopg2.extras
import json

from config import STOCK_CONFIGS
from data_pipeline import build_daily_frame, DB_CONFIG
from feature_engineering import build_features_with_medium_term

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DDL_FEATURE_STORE = """
CREATE TABLE IF NOT EXISTS daily_features (
    date DATE,
    stock_id VARCHAR(50),
    features JSONB,
    PRIMARY KEY (date, stock_id)
);
"""

UPSERT_FEATURE_STORE = """
INSERT INTO daily_features (date, stock_id, features)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    features = EXCLUDED.features;
"""

def get_latest_feature_date(conn, stock_id: str):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM daily_features WHERE stock_id = %s", (stock_id,))
        row = cur.fetchone()
        if row and row[0]:
            return row[0]
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-id", type=str, help="指定更新單一股票代號")
    args = parser.parse_args()

    logger.info("=== Feature Store Update Started ===")
    conn = psycopg2.connect(**DB_CONFIG)
    
    try:
        with conn.cursor() as cur:
            cur.execute(DDL_FEATURE_STORE)
        conn.commit()

        if args.stock_id:
            stock_ids = [args.stock_id]
        else:
            stock_ids = list(STOCK_CONFIGS.keys())
            
        logger.info(f"Updating feature store for {len(stock_ids)} stocks...")

        for i, stock_id in enumerate(stock_ids, 1):
            latest_date_obj = get_latest_feature_date(conn, stock_id)
            
            if latest_date_obj:
                fetch_start_date = (latest_date_obj - timedelta(days=250)).strftime("%Y-%m-%d")
                latest_date_str = latest_date_obj.strftime("%Y-%m-%d")
            else:
                fetch_start_date = "2010-01-01"
                latest_date_str = "1900-01-01"
            
            logger.info(f"[{i}/{len(stock_ids)}] {stock_id} - Fetching from {fetch_start_date}")
            
            try:
                df = build_daily_frame(stock_id, start_date=fetch_start_date)
            except Exception as e:
                logger.error(f"Failed to build daily frame for {stock_id}: {e}")
                continue
                
            if df.empty:
                logger.info(f"No data found for {stock_id}.")
                continue

            try:
                df = build_features_with_medium_term(df, stock_id=stock_id, for_inference=True)
            except Exception as e:
                logger.error(f"Failed to calculate features for {stock_id}: {e}")
                continue
            
            new_df = df[df.index > latest_date_str].copy()
            
            if new_df.empty:
                logger.info(f"{stock_id} is already up to date.")
                continue
                
            # Prepare for JSON serialization
            # Replace NaNs with None for JSON compliance
            clean_df = new_df.replace({pd.NA: None, float('inf'): None, float('-inf'): None})
            # Convert to JSON and back to list of dicts to handle Timestamp and other non-serializable types
            json_str = clean_df.to_json(orient='records', date_format='iso')
            records_data = json.loads(json_str)
            
            records = []
            for idx, row_data in enumerate(records_data):
                date_str = new_df.index[idx].strftime("%Y-%m-%d")
                records.append((date_str, stock_id, json.dumps(row_data)))
            
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, 
                    UPSERT_FEATURE_STORE, 
                    records,
                    page_size=1000
                )
            conn.commit()
            logger.info(f"{stock_id} updated {len(records)} records.")

    except Exception as e:
        logger.error(f"Feature Store update failed: {e}")
        conn.rollback()
    finally:
        conn.close()
        logger.info("=== Feature Store Update Completed ===")

if __name__ == "__main__":
    main()
