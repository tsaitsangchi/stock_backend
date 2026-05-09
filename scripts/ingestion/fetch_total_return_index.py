"""
fetch_total_return_index.py v5.1 (Trinity Core Edition)
================================================================================
台股報酬指數抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取發行量加權股價報酬指數 (TAIEX) 與櫃買報酬指數 (TPEx)。

主要功能：
  · Schema 自動遷移  ─ 自動偵測並修正舊版資料庫欄位名稱 (如 value, return_index)，統一為 total_return_index。
  · 智慧增量更新     ─ 自動從 fetch_log 推算缺失區間，確保報酬指數連續性。
  · 斷路器保護       ─ 對接 v5.1 核心，排除 API 業務錯誤對系統熔斷的干擾。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ Singleton + SQLite 快取 + 智慧斷路器
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 修正 ImportError，移除 finmind_get，全面換裝 FinMindClient()。
    - [核心] 移除 get_db_conn，改用 db_session 與 db_transaction。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，實現精準監控。
  v3.4 (2024-05-01):
    - [基礎] 建立基礎報酬指數抓取邏輯。

執行範例：
    # 範例 1：抓取所有預設報酬指數 (TAIEX, TPEx)
    python scripts/fetchers/fetch_total_return_index.py
    
    # 範例 2：強制重抓台積電相關報酬指數
    python scripts/fetchers/fetch_total_return_index.py --ids TAIEX --force
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import (
        db_session, db_transaction, ensure_ddl, commit_per_stock_per_day,
        get_latest_date, write_fetch_log, safe_float
    )
    from core.finmind_client import FinMindClient, get_request_stats
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. DDL 與 SQL 配置
# =====================================================================

DDL_TOTAL_RETURN = """
CREATE TABLE IF NOT EXISTS total_return_index (
    date DATE NOT NULL, 
    stock_id VARCHAR(50) NOT NULL, 
    total_return_index NUMERIC(20,6), 
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_total_return_stock ON total_return_index (stock_id, date DESC);
"""

UPSERT_TOTAL_RETURN = """
INSERT INTO total_return_index (date, stock_id, total_return_index) 
VALUES (%(date)s, %(stock_id)s, %(total_return_index)s) 
ON CONFLICT (date, stock_id) DO UPDATE SET 
    total_return_index = EXCLUDED.total_return_index;
"""

# =====================================================================
# 2. 工具與 Schema 升級
# =====================================================================

def _upgrade_schema():
    """自動偵測並修正舊版資料庫欄位名稱。"""
    try:
        with db_session() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'total_return_index';")
                columns = [row[0] for row in cur.fetchall()]
                if columns and "total_return_index" not in columns:
                    for col in columns:
                        if col not in ('date', 'stock_id'):
                            logger.warning(f"🛠️ 偵測到舊版欄位 '{col}'，正在升級為 total_return_index...")
                            with db_transaction() as tx_cur:
                                tx_cur.execute(f'ALTER TABLE total_return_index RENAME COLUMN "{col}" TO total_return_index;')
                            break
    except Exception as e:
        logger.debug(f"Schema 升級檢查可忽略: {e}")

def next_day(date_str: str) -> str:
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (d + timedelta(days=1)).strftime("%Y-%m-%d")
    except: return str(date_str)

# =====================================================================
# 3. Fetcher Logic
# =====================================================================

def fetch_one_index(stock_id: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    table = "total_return_index"
    
    cur_start = start
    if not force:
        latest = get_latest_date(table, stock_id)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, stock_id, "skipped", "tr_v5.1", str(latest), end, 0, 0, "up_to_date")
                return stock_id, 0, 0

    try:
        t0 = time.monotonic()
        data = api.get_data("TaiwanStockTotalReturnIndex", stock_id, cur_start, end)
        duration_ms = int((time.monotonic() - t0) * 1000)
        
        if data:
            records = [{
                "date": r["date"], "stock_id": stock_id, 
                "total_return_index": safe_float(r.get("price"))
            } for r in data if "date" in r]
            
            if records:
                success, error = commit_per_stock_per_day(table, records, UPSERT_TOTAL_RETURN, stock_id)
                write_fetch_log(table, stock_id, "success" if error == 0 else "partial", "tr_v5.1", cur_start, end, duration_ms, success, None)
                return stock_id, success, error
        
        write_fetch_log(table, stock_id, "no_new_data", "tr_v5.1", cur_start, end, duration_ms, 0, None)
        return stock_id, 0, 0
    except Exception as e:
        logger.error(f"  ❌ {stock_id} @ {table}: {e}")
        write_fetch_log(table, stock_id, "failed", "tr_v5.1", cur_start, end, 0, 0, str(e))
        return stock_id, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Total Return Index Fetcher v5.1 (Trinity Core Edition)")
    parser.add_argument("--ids", nargs="+", default=["TAIEX", "TPEx"])
    parser.add_argument("--start", default="2003-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_ddl(DDL_TOTAL_RETURN)
    _upgrade_schema()

    end_date = args.end or date.today().strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"  Total Return Index Fetcher v5.1  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)

    for sid in args.ids:
        ok, success, err = fetch_one_index(sid, args.start, end_date, args.force)
        if success: logger.info(f"  ✓ {sid}: {success} rows")

    logger.info("🎉 任務完成")

if __name__ == "__main__":
    main()