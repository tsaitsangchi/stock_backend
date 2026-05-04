import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

"""
fetch_total_return_index.py v3.0 — 台股報酬指數（逐支逐日 commit 完整性版）
========================================================================
v3.0 重大改進：
  ★ 導入 commit_per_stock_per_day：TAIEX (大盤) 與 TPEx (櫃買) 報酬指數每一天獨立原子 commit。
  ★ 全面整合 FailureLogger：精準追蹤基準指數的更新完整性。
  ★ 結構規範化：移除本地冗餘工具，確保生產管線高可用。
"""

from core.finmind_client import finmind_get
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DDL_TOTAL_RETURN = """CREATE TABLE IF NOT EXISTS total_return_index (date DATE, stock_id VARCHAR(50), price NUMERIC(20,4), PRIMARY KEY (date, stock_id));"""
UPSERT_TOTAL_RETURN = """INSERT INTO total_return_index (date, stock_id, price) VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET price = EXCLUDED.price;"""

def map_tr(r): return (r["date"], r["stock_id"], safe_float(r.get("price")))

def main():
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_TOTAL_RETURN)
        latest = get_all_safe_starts(conn, "total_return_index")
        flog = FailureLogger("total_return_index", db_conn=conn)
        total_rows = 0
        
        for data_id in ["TAIEX", "TPEx"]:
            s = resolve_start_cached(data_id, latest, "2010-01-01", "2010-01-01", False)
            if not s: continue
            try:
                data = finmind_get("TaiwanStockTotalReturnIndex", {"data_id": data_id, "start_date": s}, 1.2)
                if data:
                    rows = [map_tr(r) for r in data]
                    rows = dedup_rows(rows, (0, 1))
                    res = commit_per_stock_per_day(conn, UPSERT_TOTAL_RETURN, rows, "(%s, %s, %s::numeric)", label_prefix="total_return_index", failure_logger=flog)
                    total_rows += sum(res.values())
            except Exception as e: flog.record(stock_id=data_id, error=str(e))
        logger.info(f"  [total_return_index] 總共寫入 {total_rows} 筆")
        flog.summary()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
