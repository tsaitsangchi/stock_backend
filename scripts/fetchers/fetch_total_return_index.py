fetch_total_return_index.py  v2.2
=========================
從 FinMind API 抓取 TaiwanStockTotalReturnIndex（台股報酬指數）並寫入 PostgreSQL。

v2.2 改進：
  · 導入 safe_commit_rows() 與 dump_failures()。
  · 強化 atomicity：按指數 ID（TAIEX, TPEx）立即 commit。
  · 失敗清單寫入 outputs/total_return_index_failed_{date}.json。
  · 確保 DDL 執行後立即 commit。
import json
import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime

import psycopg2
import psycopg2.extras
import pandas as pd

from config import DB_CONFIG
from core.finmind_client import finmind_get as _core_finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# DDL
# ======================
DDL_TOTAL_RETURN_INDEX = """
CREATE TABLE IF NOT EXISTS total_return_index (
    date       DATE,
    stock_id   VARCHAR(50),
    price      NUMERIC(20,4),
    PRIMARY KEY (date, stock_id)
);
"""

# ──────────────────────────────────────────────
# 逐支 commit 工具函式
# ──────────────────────────────────────────────
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


def dump_failures(table: str, failures: list) -> None:
    if not failures:
        return
    out = OUTPUT_DIR / f"{table}_failed_{date.today().strftime('%Y%m%d')}.json"
    try:
        out.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"  失敗清單已寫入：{out}（{len(failures)} 筆）")
    except Exception as e:
        logger.warning(f"  寫入失敗清單時發生錯誤：{e}")


def finmind_get(dataset: str, data_id: str = None, start: str = None, end: str = None):
    params = {}
    if data_id: params["data_id"] = data_id
    if start:   params["start_date"] = start
    if end:     params["end_date"] = end
    data = _core_finmind_get(dataset, params)
    return pd.DataFrame(data)

def main():
    conn = get_db_conn()
    failures = []
    try:
        ensure_ddl(conn, DDL_TOTAL_RETURN_INDEX)
        conn.commit()

        # 抓取 TAIEX 與 TPEx 報酬指數
        for data_id in ["TAIEX", "TPEx"]:
            try:
                logger.info(f"正在抓取 {data_id} 報酬指數...")
                
                with conn.cursor() as cur:
                    cur.execute("SELECT MAX(date) FROM total_return_index WHERE stock_id = %s", (data_id,))
                    latest = cur.fetchone()[0]
                    
                start = (latest + timedelta(days=1)).strftime("%Y-%m-%d") if latest else "2010-01-01"
                
                df = finmind_get("TaiwanStockTotalReturnIndex", data_id=data_id, start=start)
                
                if df.empty:
                    logger.info(f"{data_id} 無新資料。")
                    continue
                    
                records = []
                for _, row in df.iterrows():
                    records.append((row["date"], row["stock_id"], row["price"]))
                    
                sql = """
                    INSERT INTO total_return_index (date, stock_id, price)
                    VALUES %s
                    ON CONFLICT (date, stock_id) DO UPDATE SET price = EXCLUDED.price;
                """
                n = safe_commit_rows(conn, sql, records, "(%s, %s, %s::numeric)", label=f"total_return_index/{data_id}")
                logger.info(f"已更新 {data_id} 共 {n} 筆資料。")
                time.sleep(1.2)
            except Exception as e:
                try: conn.rollback()
                except Exception: pass
                failures.append({"data_id": data_id, "error": str(e)})
                logger.error(f"  [{data_id}] 失敗：{e}")

    finally:
        conn.close()

    dump_failures("total_return_index", failures)

if __name__ == "__main__":
    main()
