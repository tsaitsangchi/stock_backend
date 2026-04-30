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
db_health_check.py — PostgreSQL 資料庫效能與健康檢查工具
=========================================================
功能：
  1. 資料表與索引大小分析 (Total/Data/Index size)
  2. 索引使用率分析 (找出未被使用的索引)
  3. 快取命中率 (Cache Hit Ratio)
  4. 慢查詢與鎖定 (Long-running Queries / Locks)

執行：
    python scripts/db_health_check.py
"""
import logging
import sys
from core.db_utils import get_db_conn
from config import DB_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def print_separator(title):
    logger.info("\n" + "="*80)
    logger.info(f"  {title}")
    logger.info("="*80)

def check_table_sizes(conn):
    print_separator("1. 資料表大小分析 (Top 10)")
    query = """
    SELECT 
        relname AS table_name,
        pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
        pg_size_pretty(pg_relation_size(relid)) AS data_size,
        pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size,
        n_live_tup AS row_estimate
    FROM pg_stat_user_tables
    ORDER BY pg_total_relation_size(relid) DESC
    LIMIT 10;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        logger.info(f"{'Table Name':<30} | {'Total Size':<12} | {'Data Size':<12} | {'Index Size':<12} | {'Rows'}")
        logger.info("-" * 85)
        for r in rows:
            logger.info(f"{r[0]:<30} | {r[1]:<12} | {r[2]:<12} | {r[3]:<12} | {r[4]:,}")

def check_index_usage(conn):
    print_separator("2. 索引使用率 (找出無效索引)")
    query = """
    SELECT 
        relname AS table_name,
        indexrelname AS index_name,
        idx_scan,
        pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
    FROM pg_stat_user_indexes
    WHERE idx_scan = 0 AND schemaname = 'public'
    ORDER BY pg_relation_size(indexrelid) DESC;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        if not rows:
            logger.info("✅ 讚！所有索引皆有被使用過。")
        else:
            logger.info(f"{'Table Name':<30} | {'Unused Index Name':<30} | {'Size'}")
            logger.info("-" * 75)
            for r in rows:
                logger.info(f"{r[0]:<30} | {r[1]:<30} | {r[3]}")

def check_cache_hit_ratio(conn):
    print_separator("3. 快取命中率 (Buffer Cache Hit Ratio)")
    query = """
    SELECT 
        datname,
        100 * blks_hit / (blks_hit + blks_read + 1) AS cache_hit_ratio
    FROM pg_stat_database
    WHERE datname = %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (DB_CONFIG['dbname'],))
        row = cur.fetchone()
        if row:
            ratio = float(row[1])
            status = "✅ 優秀" if ratio > 95 else "⚠️ 警告（可能需要增加 RAM）"
            logger.info(f"資料庫 [{row[0]}] 快取命中率: {ratio:.2f}%  -> {status}")

def check_slow_queries(conn):
    print_separator("4. 當前執行中查詢 (與可能鎖定)")
    query = """
    SELECT 
        pid,
        now() - query_start AS duration,
        state,
        query
    FROM pg_stat_activity
    WHERE state != 'idle' AND now() - query_start > interval '5 seconds'
    ORDER BY duration DESC;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        if not rows:
            logger.info("✅ 目前無執行超過 5 秒的查詢。")
        else:
            for r in rows:
                logger.info(f"PID: {r[0]} | 耗時: {r[1]} | 狀態: {r[2]}")
                logger.info(f"查詢: {r[3][:200]}...")
                logger.info("-" * 40)

def main():
    logger.info("正在連線資料庫進行效能診斷...")
    try:
        conn = get_db_conn()
        check_table_sizes(conn)
        check_index_usage(conn)
        check_cache_hit_ratio(conn)
        check_slow_queries(conn)
        conn.close()
    except Exception as e:
        logger.error(f"❌ 診斷失敗：{e}")

if __name__ == "__main__":
    main()
