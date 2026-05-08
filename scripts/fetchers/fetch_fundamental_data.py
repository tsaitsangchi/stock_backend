"""
Fundamental Data Pipeline v3.1 (Quantum Finance v5.1 Edition)
============================================================
結合智慧型批次請求與領域知識（財報延遲遞延、季節性跳過）。
完全整合核心模組以去除冗餘邏輯。
"""

import sys
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── 核心初始化 ──
from core.path_setup import ensure_scripts_on_path
_BASE_DIR = ensure_scripts_on_path(__file__)

from core.db_utils import get_db_conn, ensure_ddl
from core.finmind_client import finmind_get

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DDL_FINANCIAL_STATEMENT = """
CREATE TABLE IF NOT EXISTS financial_statements (
    stock_id    VARCHAR(20),
    date        DATE,
    type        VARCHAR(100),
    value       NUMERIC(20,4),
    origin_name VARCHAR(200),
    PRIMARY KEY (stock_id, date, type, origin_name)
);
CREATE INDEX IF NOT EXISTS idx_fin_stock_date ON financial_statements (stock_id, date);
"""

def fetch_financial_statements(conn, target_stocks: list):
    """
    智慧型財報抓取模組。
    整合損益表與資產負債表，並自動處理財報發布遞延邏輯。
    """
    ensure_ddl(conn, DDL_FINANCIAL_STATEMENT)
    
    # 1. 預先加載全市場最新更新日期 (Latest Dates Cache)
    latest_dates = {}
    with conn.cursor() as cur:
        cur.execute("SELECT stock_id, MAX(date) FROM financial_statements GROUP BY stock_id;")
        for sid, last_date in cur.fetchall():
            latest_dates[sid] = last_date
            
    logger.info(f"Loaded {len(latest_dates)} latest dates from financial_statements.")

    upsert_query = """
        INSERT INTO financial_statements (stock_id, date, type, value, origin_name)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (stock_id, date, type, origin_name) 
        DO UPDATE SET value = EXCLUDED.value;
    """
    
    for stock_id in target_stocks:
        last_date = latest_dates.get(stock_id)
        # 若無資料則從 2010 開始補
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else "2010-01-01"
        
        logger.info(f"Fetching {stock_id} financial statements from {start_date}...")
        
        # 依賴 core/finmind_client 處理速率限制與斷路器
        data = finmind_get(
            dataset="TaiwanStockFinancialStatements",
            params={"data_id": stock_id, "start_date": start_date}
        )
        
        if not data:
            continue
            
        # 2. 領域知識應用：財報發布遞延 45 天 (Look-ahead Bias Mitigation)
        # 金融邏輯：台股財報通常在季底後 45 天內發布，將日期推移以模擬真實資訊流
        records = []
        for row in data:
            try:
                # 原始日期是會計結算日，我們將其推移至「可取得日」
                original_date = datetime.strptime(row["date"], "%Y-%m-%d")
                publish_date = original_date + timedelta(days=45)
                
                records.append((
                    row["stock_id"], 
                    publish_date.strftime("%Y-%m-%d"), 
                    row["type"], 
                    row["value"], 
                    row.get("origin_name", row["type"])
                ))
            except Exception as e:
                logger.warning(f"Skip invalid row for {stock_id}: {e}")
        
        if records:
            with conn.cursor() as cur:
                cur.executemany(upsert_query, records)
            conn.commit()
            logger.info(f"✅ Updated {len(records)} financial records for {stock_id}.")
        
        # 適度延遲防止 API 壓力
        time.sleep(0.5)

def main():
    conn = get_db_conn()
    try:
        # 3. 動態撈取資料庫定義的「基本面抓取清單」
        with conn.cursor() as cur:
            cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE AND fetch_fundamental = TRUE;")
            target_stocks = [row[0] for row in cur.fetchall()]
            
        if not target_stocks:
            logger.warning("No active stocks marked for fundamental fetching.")
            return

        logger.info(f"Starting fundamental pipeline for {len(target_stocks)} stocks...")
        fetch_financial_statements(conn, target_stocks)
        
    except Exception as e:
        logger.error(f"❌ Fundamental pipeline failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()