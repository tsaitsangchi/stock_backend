import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_stock_info.py  v3.0 (Unified Stocks Edition)
=================================================
從 FinMind API 抓取 TaiwanStockInfo 並寫入核心 stocks 資料表。
與 Quantum Finance v5.1 監控架構對齊。
"""

import json
import logging
import time
import os
from datetime import date, datetime, timedelta

import psycopg2

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
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
# DDL (使用統一的 stocks 資料表)
# ======================
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stocks (
    stock_id          VARCHAR(50) PRIMARY KEY,
    stock_name        VARCHAR(100),
    industry          VARCHAR(100),
    market_type       VARCHAR(50),
    last_update_date  DATE,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_stocks_market_type ON stocks (market_type);
"""

# ======================
# Upsert SQL
# ======================
UPSERT_SQL = """
INSERT INTO stocks (stock_id, stock_name, industry, market_type, last_update_date)
VALUES %s
ON CONFLICT (stock_id)
DO UPDATE SET
    stock_name        = EXCLUDED.stock_name,
    industry          = EXCLUDED.industry,
    market_type       = EXCLUDED.market_type,
    last_update_date  = EXCLUDED.last_update_date,
    updated_at        = CURRENT_TIMESTAMP;
"""

def safe_commit_rows(conn, upsert_sql: str, rows: list, template: str,
                      label: str = "") -> int:
    if not rows:
        return 0
    try:
        n = bulk_upsert(conn, upsert_sql, rows, template)
        conn.commit()
        return n
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f"  [{label}] 寫入失敗，已 rollback：{e}")
        return 0

def fetch_and_transform() -> list[tuple]:
    data = finmind_get("TaiwanStockInfo", {}, raise_on_error=True)
    if not data:
        return []

    seen = {}
    for r in data:
        sid = r.get("stock_id")
        if not sid:
            continue
        _date = r.get("date", "")
        date_val = _date if (_date and _date.upper() != "NONE") else None

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

def run_update() -> None:
    rows = fetch_and_transform()
    if not rows:
        logger.warning("沒有資料可更新")
        return

    conn = get_db_conn()
    try:
        ensure_ddl(conn, CREATE_TABLE_SQL)
        conn.commit()

        success_count = 0
        for row in rows:
            sid = row[0]
            n = safe_commit_rows(
                conn, UPSERT_SQL, [row],
                template="(%s, %s, %s, %s, %s::date)",
                label=f"stocks/{sid}"
            )
            if n > 0:
                success_count += n

        logger.info(f"完成，逐支寫入 {success_count} 筆標的資訊至 stocks 資料表")
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
