"""
fetch_stock_info.py — 股票基本資訊與全市場代號 (v3.2 核心模組全面升級版)
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 整合 `FailureLogger` 與 `fetch_log`，提供完整的資料觀測。
  ★ 採用整批寫入 (safe_commit_rows) 以提升萬筆台股代號的更新效率。
  ★ 程式結束時自動印出 `finmind_client` 的 RequestStats 統計報表。

執行範例：
    # 抓取全市場所有股票、ETF、權證等基本資訊
    python scripts/fetchers/fetch_stock_info.py
"""

from __future__ import annotations

import sys
import logging
import time
import argparse
from pathlib import Path
from datetime import date, datetime

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.finmind_client import finmind_get, get_request_stats
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_commit_rows,
    FailureLogger,
    write_fetch_log,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_CLI_ARGS_STR = " ".join(sys.argv)


# ─────────────────────────────────────────────
# DDL & SQL
# ─────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")


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


# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_and_transform(delay: float) -> tuple[list[tuple], int]:
    """抓取全市場股票代號與名稱資訊。"""
    start_ts = time.time()
    
    # 修正：全面改為具名參數 (Keyword arguments)
    data = finmind_get(
        dataset="TaiwanStockInfo", 
        params={}, 
        delay=delay,
        raise_on_error=True
    )
    duration_ms = int((time.time() - start_ts) * 1000)
    
    if not data:
        return [], duration_ms

    seen = {}
    for r in data:
        sid = r.get("stock_id")
        if not sid:
            continue
            
        _date = r.get("date", "")
        date_val = _date if (_date and str(_date).upper() != "NONE") else None

        # 如果遇到同樣的 stock_id，保留日期最新的一筆
        if sid not in seen or (date_val and date_val > (seen[sid][4] or "")):
            seen[sid] = (
                str(sid)[:50],
                str(r.get("stock_name", ""))[:100],
                str(r.get("industry_category", ""))[:100],
                str(r.get("type", ""))[:50],
                date_val,
            )

    rows = list(seen.values())
    logger.info(f"去重後剩 {len(rows)} 筆（原始 {len(data)} 筆）")
    return rows, duration_ms


def run_update(delay: float):
    logger.info("=== [stocks info] 開始 ===")
    conn = get_db_conn()
    flog = FailureLogger("stocks", db_conn=conn)
    
    try:
        _ensure_fetch_log_table(conn)
        ensure_ddl(conn, CREATE_TABLE_SQL)
        conn.commit()

        try:
            rows, duration_ms = fetch_and_transform(delay)
            if not rows:
                logger.warning("沒有資料可更新")
                _write_fetch_log(conn, "stocks", "ALL", "no_new_data", duration_ms=duration_ms)
                return

            # 使用 safe_commit_rows 進行整批更新 (以 page_size=2000 寫入，效率大幅超越舊版逐筆寫入)
            n = safe_commit_rows(
                conn, UPSERT_SQL, rows,
                template="(%s, %s, %s, %s, %s::date)",
                label="stocks_info_update"
            )
            
            if n > 0:
                logger.info(f"✅ 成功寫入 {n} 筆標的資訊至 stocks 資料表")
                write_fetch_log(conn, table_name="stocks", stock_id="ALL", status="success", rows_inserted=n, duration_ms=duration_ms)
            else:
                write_fetch_log(conn, table_name="stocks", stock_id="ALL", status="failed", rows_inserted=0, duration_ms=duration_ms, error_message="safe_commit_rows 回傳 0")
                
        except Exception as e:
            logger.error(f"執行失敗：{e}")
            flog.record(stock_id="ALL", error=str(e))
            write_fetch_log(conn, table_name="stocks", stock_id="ALL", status="failed", rows_inserted=0, duration_ms=0, error_message=str(e))
            
    finally:
        conn.close()
        flog.summary()


def main():
    p = argparse.ArgumentParser(description="台股全市場代號更新 (v3.2 — 核心模組升級版)")
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args()

    try:
        run_update(args.delay)
    finally:
        get_request_stats().summary()

if __name__ == "__main__":
    main()