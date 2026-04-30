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
fetch_stock_info.py
從 FinMind API 抓取 TaiwanStockInfo（台股總覽）並寫入 PostgreSQL stock_info 資料表。

需求套件：
    pip install requests psycopg2-binary
"""

import logging
import sys

import time
from datetime import datetime, timedelta

import psycopg2

from config import FINMIND_TOKEN, DB_CONFIG
import psycopg2.extras
import requests

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
# [P0 重構] 工具函式統一改用 core 模組
# ──────────────────────────────────────────────
from core.finmind_client import finmind_get, wait_until_next_hour
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
)


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
    try:
        ensure_ddl(conn, CREATE_TABLE_SQL)
        n = bulk_upsert(
            conn, UPSERT_SQL, rows, 
            template="(%s, %s, %s, %s, %s::date)"
        )
        logger.info(f"✅ 完成，upsert {n} 筆")
    finally:
        conn.close()


def main():
    try:
        run_update()
    except Exception as e:
        logger.error(f"執行失敗：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
