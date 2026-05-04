import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_stock_info.py  v2.2
=========================
從 FinMind API 抓取 TaiwanStockInfo（台股總覽）並寫入 PostgreSQL stock_info 資料表。

v2.2 改進：
  · 導入 safe_commit_rows() 與 dump_failures()。
  · 強化 atomicity：逐支 commit 標的資訊。
  · 失敗清單寫入 outputs/stock_info_failed_{date}.json。
"""

import json
import logging
import time
from datetime import date, datetime, timedelta

import psycopg2

from core.finmind_client import finmind_get, wait_until_next_hour
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    FailureLogger,
    commit_per_group,
    dedup_rows,
)

# ======================
# 設定 logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# DDL（若資料表尚未存在則建立）
# ======================
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_info (
    stock_id          VARCHAR(50) PRIMARY KEY,
    stock_name        VARCHAR(100),
    industry_category VARCHAR(100),
    type              VARCHAR(50),
    date              DATE
);
CREATE INDEX IF NOT EXISTS idx_stock_info_date ON stock_info (date);
"""

# ======================
# Upsert SQL
# ======================
UPSERT_SQL = """
INSERT INTO stock_info (stock_id, stock_name, industry_category, type, date)
VALUES %s
ON CONFLICT (stock_id)
DO UPDATE SET
    stock_name        = EXCLUDED.stock_name,
    industry_category = EXCLUDED.industry_category,
    type              = EXCLUDED.type,
    date              = EXCLUDED.date;
"""


# ──────────────────────────────────────────────
# 1. 抓取與轉換
# ──────────────────────────────────────────────
def fetch_and_transform(delay: float = 1.0) -> list[tuple]:
    """
    抓取台股總覽並去重轉換。
    """
    data = finmind_get("TaiwanStockInfo", {}, delay)
    if not data:
        return []

    seen = {}
    for r in data:
        sid = r.get("stock_id")
        if not sid:
            continue
        _date = r.get("date", "")
        date_val = _date if (_date and _date.upper() != "NONE") else None

        # 同一 stock_id 保留日期最新的那筆
        if sid not in seen or (date_val and date_val > (seen[sid][4] or "")):
            seen[sid] = (
                sid,
                r.get("stock_name"),
                r.get("industry_category"),
                r.get("type"),
                date_val,
            )

    rows = list(seen.values())
    logger.info(f"去重後剩 {len(rows)} 筆（原始 {len(data)} 筆）")
    return rows


# ──────────────────────────────────────────────
# 2. 執行更新
# ──────────────────────────────────────────────
def run_update(delay: float = 1.0) -> None:
    rows = fetch_and_transform(delay)
    if not rows:
        logger.warning("沒有資料可更新")
        return

    conn = get_db_conn()
    flog = FailureLogger("stock_info", db_conn=conn)
    try:
        ensure_ddl(conn, CREATE_TABLE_SQL)
        
        # 去重：以 stock_id (0) 為 key
        rows = dedup_rows(rows, (0,))
        
        res = commit_per_group(
            conn, UPSERT_SQL, rows, 
            template="(%s, %s, %s, %s, %s::date)",
            group_key_fn=lambda r: r[0],
            label_prefix="stock_info",
            failure_logger=flog
        )

        logger.info(f"完成，逐支寫入 {sum(res.values())} 筆標的資訊")
    finally:
        conn.close()
    
    flog.summary()


def main():
    try:
        run_update()
    except Exception as e:
        logger.error(f"執行失敗：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
