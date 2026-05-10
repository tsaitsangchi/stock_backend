"""
db_optimize.py — PostgreSQL 17 效能優化工具
==========================================
針對 80+ 標的與大規模特徵工程進行索引優化、物化視圖建立與分區建議。
"""

import logging
import sys
import psycopg2
from pathlib import Path

# 注入路徑
sys.path.append(str(Path(__file__).resolve().parent))
from data_pipeline import DB_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_sql(conn, sql, msg=None):
    if msg: logger.info(msg)
    with conn.cursor() as cur:
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"執行失敗: {e}")

def optimize_indices(conn):
    """
    建立複合索引 (stock_id, date DESC)。
    這能極大化加速 build_daily_frame() 中的 JOIN 與排序效能。
    """
    tables = [
        "stock_price", "stock_per", "institutional_investors_buy_sell",
        "margin_purchase_short_sale", "shareholding", "month_revenue",
        "financial_statements", "balance_sheet"
    ]
    
    for table in tables:
        index_name = f"idx_{table}_id_date"
        sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} (stock_id, date DESC);"
        run_sql(conn, sql, f"建立索引: {index_name} on {table}")

def setup_materialized_views(conn):
    """
    建立物化視圖 (Materialized View) 預處理常用特徵。
    """
    # 建立一個基礎的每日特徵快照（以 2330 為範例或全市場）
    # 這裡提供 DDL 範例，實際可根據需求擴展
    view_sql = """
    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_market_sync AS
    SELECT 
        p.date, p.stock_id, p.close, p.volume,
        i.foreign_investor_net, i.investment_trust_net,
        m.margin_purchase_today_balance as margin_bal
    FROM stock_price p
    LEFT JOIN institutional_investors_buy_sell i ON p.date = i.date AND p.stock_id = i.stock_id
    LEFT JOIN margin_purchase_short_sale m ON p.date = m.date AND p.stock_id = m.stock_id
    WITH NO DATA;
    """
    run_sql(conn, view_sql, "建立物化視圖: mv_daily_market_sync")
    
    # 建立唯一索引以便支援 CONCURRENTLY 刷新
    run_sql(conn, "CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_sync_date_id ON mv_daily_market_sync (date, stock_id);", "建立物化視圖索引")

def refresh_views(conn):
    """
    刷新物化視圖。
    """
    run_sql(conn, "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_market_sync;", "刷新物化視圖 (CONCURRENTLY)")

def main():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("=== 開始資料庫效能優化 (PostgreSQL 17) ===")
        
        # 1. 索引優化
        optimize_indices(conn)
        
        # 2. 物化視圖
        setup_materialized_views(conn)
        
        # 3. 執行第一次刷新
        # run_sql(conn, "REFRESH MATERIALIZED VIEW mv_daily_market_sync;", "初始刷新物化視圖")
        
        logger.info("=== 優化完成 ===")
        conn.close()
    except Exception as e:
        logger.error(f"連線失敗: {e}")

if __name__ == "__main__":
    main()
