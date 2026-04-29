"""
db_auto_fix.py — 資料庫自動修復與清理工具 (第四輪優化配套)
=========================================================
功能：
  1. 刪除僵屍連線 (Idle in transaction > 10m)
  2. 資料表名稱遷移 (將舊表 option_daily/futures_daily/eight_banks 改名為新表名)
  3. 索引重建 (解決 daily_features 膨脹問題)
  4. 全域清理 (VACUUM ANALYZE)

執行：
    python scripts/db_auto_fix.py
"""
import logging
import sys
from core.db_utils import get_db_conn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def run_fix():
    conn = get_db_conn()
    conn.autocommit = True
    cur = conn.cursor()

    # 1. 殺掉僵屍連線
    logger.info("Step 1: 清理超過 10 分鐘的僵屍連線...")
    cur.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE state = 'idle in transaction'
          AND now() - query_start > interval '10 minutes';
    """)
    logger.info(f"   已清理 {cur.rowcount} 個僵屍連線")

    # 2. 資料表遷移 (檢查舊表是否存在且新表是否為空)
    migrations = [
        ("option_daily", "options_ohlcv"),
        ("futures_daily", "futures_ohlcv"),
        ("eight_banks", "eight_banks_buy_sell"),
        ("options_large_oi", "options_oi_large_holders")
    ]

    logger.info("Step 2: 執行資料表名稱遷移 (Align with TABLE_REGISTRY)...")
    for old, new in migrations:
        try:
            # 檢查舊表是否存在
            cur.execute(f"SELECT to_regclass('{old}')")
            if not cur.fetchone()[0]:
                continue
            
            # 檢查新表是否存在，若存在且為空則刪除新表，準備改名
            cur.execute(f"SELECT to_regclass('{new}')")
            if cur.fetchone()[0]:
                cur.execute(f"SELECT COUNT(*) FROM {new}")
                count = cur.fetchone()[0]
                if count == 0:
                    logger.info(f"   刪除空的新表 {new}...")
                    cur.execute(f"DROP TABLE {new} CASCADE")
                else:
                    logger.warning(f"   警告：{new} 已有資料，略過改名")
                    continue
            
            logger.info(f"   將舊表 {old} 改名為 {new}...")
            cur.execute(f"ALTER TABLE {old} RENAME TO {new}")
        except Exception as e:
            logger.error(f"   遷移 {old} 失敗：{e}")

    # 3. 解決索引膨脹
    logger.info("Step 3: 重建 daily_features 索引以解決膨脹...")
    try:
        cur.execute("REINDEX TABLE daily_features")
        logger.info("   REINDEX 完成")
    except Exception as e:
        logger.error(f"   REINDEX 失敗：{e}")

    # 4. 全域清理
    logger.info("Step 4: 執行 VACUUM ANALYZE (釋放空間與更新統計資訊)...")
    try:
        cur.execute("VACUUM ANALYZE")
        logger.info("   VACUUM ANALYZE 完成")
    except Exception as e:
        logger.error(f"   VACUUM 失敗：{e}")

    conn.close()
    logger.info("✅ 資料庫修復與清理完成")

if __name__ == "__main__":
    run_fix()
