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
"""
db_optimize.py — PostgreSQL 17 效能優化工具
==========================================
針對 80+ 標的與大規模特徵工程進行索引優化、物化視圖建立與分區建議。

修改摘要（第三輪審查修復）：
  [P1 2.8] 加入 --refresh-only CLI 參數，供 automate_daily.py 每日呼叫
  [P1 2.8] 解除初始刷新的註解，確保視圖建立後立刻填充資料
"""

import argparse
import logging
import sys
import psycopg2
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from data_pipeline import DB_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_sql(conn, sql, msg=None):
    if msg:
        logger.info(msg)
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
    能極大化加速 build_daily_frame() 中的 JOIN 與排序效能。
    """
    tables = [
        "stock_price", "stock_per", "institutional_investors_buy_sell",
        "margin_purchase_short_sale", "shareholding", "month_revenue",
        "financial_statements", "balance_sheet",
    ]
    for table in tables:
        index_name = f"idx_{table}_id_date"
        sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} (stock_id, date DESC);"
        run_sql(conn, sql, f"建立索引: {index_name} on {table}")


def setup_materialized_views(conn):
    """建立物化視圖 (Materialized View) 預處理常用特徵。"""
    # [修正] 欄位名稱必須與資料庫實際 schema 一致
    # stock_price: trading_volume
    # institutional_investors_buy_sell: 需透過聚合轉為寬格式
    view_sql = """
    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_market_sync AS
    SELECT
        p.date, p.stock_id, p.close, 
        p.trading_volume AS volume,
        SUM(CASE WHEN i.name = 'Foreign_Investor' THEN (i.buy - i.sell) ELSE 0 END) AS foreign_net,
        SUM(CASE WHEN i.name = 'Investment_Trust' THEN (i.buy - i.sell) ELSE 0 END) AS trust_net,
        m.margin_purchase_today_balance AS margin_bal
    FROM stock_price p
    LEFT JOIN institutional_investors_buy_sell i
           ON p.date = i.date AND p.stock_id = i.stock_id
    LEFT JOIN margin_purchase_short_sale m
           ON p.date = m.date AND p.stock_id = m.stock_id
    GROUP BY p.date, p.stock_id, p.close, p.trading_volume, m.margin_purchase_today_balance
    WITH NO DATA;
    """
    run_sql(conn, view_sql, "建立物化視圖: mv_daily_market_sync")
    run_sql(
        conn,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_sync_date_id ON mv_daily_market_sync (date, stock_id);",
        "建立物化視圖唯一索引（CONCURRENTLY 刷新必要）",
    )


def ensure_columns(conn):
    """確保必要的資料表與欄位存在。"""
    # [P1] 建立預測紀錄表
    forecast_sql = """
    CREATE TABLE IF NOT EXISTS public.stock_forecast_daily (
        predict_date DATE,
        forecast_date DATE,
        stock_id VARCHAR(20),
        current_close FLOAT,
        prob_up FLOAT,
        day_offset INT,
        is_backfill BOOLEAN DEFAULT FALSE,
        warning_flag TEXT DEFAULT '',
        PRIMARY KEY (predict_date, forecast_date, stock_id, day_offset)
    );
    """
    run_sql(conn, forecast_sql, "確保 stock_forecast_daily 資料表存在")

    # [P0 修復 2.2] 加入 is_backfill 欄位（以防表已存在但版本較舊）
    run_sql(
        conn,
        "ALTER TABLE stock_forecast_daily ADD COLUMN IF NOT EXISTS is_backfill BOOLEAN DEFAULT FALSE;",
        "確保 is_backfill 欄位存在於 stock_forecast_daily",
    )
    run_sql(
        conn,
        "ALTER TABLE stock_forecast_daily ADD COLUMN IF NOT EXISTS warning_flag TEXT DEFAULT '';",
        "確保 warning_flag 欄位存在於 stock_forecast_daily",
    )
    
    # [P3 修復 3.1] 確保 stock_dynamics_registry 包含 signal_filter.py 所需的所有欄位
    dynamics_sql = """
    CREATE TABLE IF NOT EXISTS stock_dynamics_registry (
        stock_id            VARCHAR(20) PRIMARY KEY,
        avg_mass            FLOAT DEFAULT 12.0,
        gravity_elasticity  FLOAT DEFAULT 0.05,
        innovation_velocity FLOAT DEFAULT 1.0,
        info_sensitivity    FLOAT DEFAULT 0.5,
        fat_tail_index      FLOAT DEFAULT 3.0,
        convexity_score     FLOAT DEFAULT 0.0,
        tail_risk_score     FLOAT DEFAULT 0.0,
        wave_track          VARCHAR(50) DEFAULT 'LEGACY_IT',
        updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    run_sql(conn, dynamics_sql, "確保 stock_dynamics_registry 資料表結構完整")
    
    # 若表已存在，補齊缺失欄位
    extra_cols = [
        ("info_sensitivity", "FLOAT DEFAULT 0.5"),
        ("fat_tail_index", "FLOAT DEFAULT 3.0"),
        ("convexity_score", "FLOAT DEFAULT 0.0"),
        ("tail_risk_score", "FLOAT DEFAULT 0.0"),
        ("wave_track", "VARCHAR(50) DEFAULT 'LEGACY_IT'"),
    ]
    for col, c_type in extra_cols:
        run_sql(conn, f"ALTER TABLE stock_dynamics_registry ADD COLUMN IF NOT EXISTS {col} {c_type};")


def refresh_views(conn, concurrently=True):
    """刷新物化視圖（[P1 2.8] 供每日管線呼叫）。"""
    method = "CONCURRENTLY" if concurrently else ""
    run_sql(
        conn,
        f"REFRESH MATERIALIZED VIEW {method} mv_daily_market_sync;",
        f"刷新物化視圖 ({method})",
    )


def main():
    # [P1 2.8] 加入 --refresh-only 參數，供 automate_daily.py 每日呼叫
    parser = argparse.ArgumentParser(description="PostgreSQL 效能優化工具")
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="只刷新物化視圖，不重建索引（每日管線使用）",
    )
    args = parser.parse_args()

    try:
        conn = psycopg2.connect(**DB_CONFIG)

        if args.refresh_only:
            # 每日管線快速路徑：只刷新視圖
            logger.info("=== [--refresh-only] 刷新物化視圖 ===")
            refresh_views(conn, concurrently=True)
            logger.info("=== 刷新完成 ===")
        else:
            # 完整優化路徑（初次安裝或維護時執行）
            logger.info("=== 開始資料庫效能優化 (PostgreSQL 17) ===")
            ensure_columns(conn)
            optimize_indices(conn)
            setup_materialized_views(conn)
            # [P1 2.8] 第一次刷新不能用 CONCURRENTLY，因為視圖尚無資料
            refresh_views(conn, concurrently=False)
            logger.info("=== 優化完成 ===")

        conn.close()
    except Exception as e:
        logger.error(f"連線失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
