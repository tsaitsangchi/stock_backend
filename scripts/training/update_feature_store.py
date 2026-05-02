"""
update_feature_store.py — 每日特徵存儲更新 v2
==============================================
增量更新 PostgreSQL `daily_features` 表，每筆 row 一個 (date, stock_id) JSONB。

v2 改進：
  · 修正頂部三重 sys.path 重複插入
  · 改用 core.db_utils.get_db_conn / ensure_ddl / bulk_upsert（去除直接呼叫 psycopg2.extras）
  · 業務邏輯（feature build、JSON 序列化、增量切片）全部保留
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401

import argparse
import json
import logging
from datetime import timedelta

import pandas as pd

from config import STOCK_CONFIGS
from core.db_utils import get_db_conn, ensure_ddl, bulk_upsert
from data_pipeline import build_daily_frame
from feature_engineering import build_features_with_medium_term

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DDL_FEATURE_STORE = """
CREATE TABLE IF NOT EXISTS daily_features (
    date     DATE,
    stock_id VARCHAR(50),
    features JSONB,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_daily_features_stock_id ON daily_features (stock_id);
"""

UPSERT_FEATURE_STORE = """
INSERT INTO daily_features (date, stock_id, features)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    features = EXCLUDED.features;
"""


def get_latest_feature_date(conn, stock_id: str):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(date) FROM daily_features WHERE stock_id = %s",
            (stock_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0]
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-id", type=str, help="指定更新單一股票代號")
    args = parser.parse_args()

    logger.info("=== Feature Store Update Started ===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_FEATURE_STORE)

        stock_ids = [args.stock_id] if args.stock_id else list(STOCK_CONFIGS.keys())
        logger.info(f"Updating feature store for {len(stock_ids)} stocks...")

        for i, stock_id in enumerate(stock_ids, 1):
            latest_date_obj = get_latest_feature_date(conn, stock_id)

            if latest_date_obj:
                fetch_start_date = (latest_date_obj - timedelta(days=250)).strftime("%Y-%m-%d")
                latest_date_str  = latest_date_obj.strftime("%Y-%m-%d")
            else:
                fetch_start_date = "2010-01-01"
                latest_date_str  = "1900-01-01"

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

            # 序列化：將 NaN/inf 換為 None 後轉 JSON
            clean_df = new_df.replace({pd.NA: None, float("inf"): None, float("-inf"): None})
            json_str = clean_df.to_json(orient="records", date_format="iso")
            records_data = json.loads(json_str)

            records = [
                (new_df.index[idx].strftime("%Y-%m-%d"), stock_id, json.dumps(row_data))
                for idx, row_data in enumerate(records_data)
            ]

            written = bulk_upsert(
                conn, UPSERT_FEATURE_STORE, records,
                template="(%s::date, %s, %s::jsonb)",
                page_size=1000,
            )
            logger.info(f"{stock_id} updated {written} records.")

    except Exception as e:
        logger.error(f"Feature Store update failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info("=== Feature Store Update Completed ===")


if __name__ == "__main__":
    main()
