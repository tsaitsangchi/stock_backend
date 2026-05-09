"""
fetch_stock_info.py v5.1 (Trinity Core Edition)
================================================================================
股票基本資訊與全市場代號抓取器 — 完美對接 core/ 五大核心模組
此模組負責同步 FinMind 的 TaiwanStockInfo，更新全市場股票、ETF 與權證的基礎屬性。

核心功能：
  · 全市場同步       ─ 抓取超過 10,000 筆標的，包含證券代號、名稱、產業與市場別。
  · 智慧去重         ─ 自動識別重複 ID 並保留最新發布日期 (last_update_date) 的資料。
  · 高效整批寫入     ─ 採用 db_utils 的高效寫入機制，秒級完成萬筆資料同步。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ Singleton + SQLite 快取 + 智慧斷路器
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 修正 ImportError，移除 finmind_get，全面換裝 FinMindClient()。
    - [核心] 移除 get_db_conn，改用 db_session 與 db_transaction。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，實現精準監控。
  v3.2 (2024-05-01):
    - [基礎] 建立基礎代號同步架構。

執行範例：
    # 抓取全市場所有股票、ETF、權證等基本資訊
    python scripts/fetchers/fetch_stock_info.py
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import date, datetime
from typing import List, Optional, Tuple, Dict

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
        write_fetch_log, FailureLogger
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
VALUES (%(stock_id)s, %(stock_name)s, %(industry)s, %(market_type)s, %(last_update_date)s)
ON CONFLICT (stock_id)
DO UPDATE SET
    stock_name        = EXCLUDED.stock_name,
    industry          = EXCLUDED.industry,
    market_type       = EXCLUDED.market_type,
    last_update_date  = EXCLUDED.last_update_date,
    updated_at        = CURRENT_TIMESTAMP;
"""

# =====================================================================
# 2. Fetcher Logic
# =====================================================================

def run_update():
    logger.info("=== [stocks info] 開始 ===")
    api = FinMindClient()
    table = "stocks"
    
    ensure_ddl(CREATE_TABLE_SQL)

    try:
        t0 = time.monotonic()
        data = api.get_data(dataset="TaiwanStockInfo")
        duration_ms = int((time.monotonic() - t0) * 1000)
        
        if not data:
            logger.warning("沒有資料可更新")
            write_fetch_log(table, "ALL", "no_new_data", "info_v5.1", None, None, duration_ms, 0, None)
            return

        seen = {}
        for r in data:
            sid = r.get("stock_id")
            if not sid: continue
                
            _date = r.get("date", "")
            date_val = _date if (_date and str(_date).upper() != "NONE") else None

            # 如果遇到同樣的 stock_id，保留日期最新的一筆
            if sid not in seen or (date_val and date_val > (seen[sid]["last_update_date"] or "")):
                seen[sid] = {
                    "stock_id": str(sid)[:50],
                    "stock_name": str(r.get("stock_name", ""))[:100],
                    "industry": str(r.get("industry_category", ""))[:100],
                    "market_type": str(r.get("type", ""))[:50],
                    "last_update_date": date_val,
                }

        records = list(seen.values())
        logger.info(f"去重後剩 {len(records)} 筆（原始 {len(data)} 筆）")

        if records:
            # 由於筆數眾多 (10,000+)，使用事務批次處理
            success, error = commit_per_stock_per_day(table, records, UPSERT_SQL, "ALL")
            write_fetch_log(table, "ALL", "success" if error == 0 else "partial", "info_v5.1", None, None, duration_ms, success, None)
            logger.info(f"✅ 成功寫入 {success} 筆標的資訊")
            
    except Exception as e:
        logger.error(f"執行失敗：{e}")
        write_fetch_log(table, "ALL", "failed", "info_v5.1", None, None, 0, 0, str(e))

def main():
    parser = argparse.ArgumentParser(description="台股全市場代號更新 v5.1 (Trinity Core Edition)")
    args = parser.parse_args()

    try:
        run_update()
    finally:
        try: get_request_stats().summary()
        except: pass

if __name__ == "__main__":
    main()