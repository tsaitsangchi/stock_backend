"""
fetch_macro_fundamental_data.py v5.1 (Trinity Core Edition)
================================================================================
巨量宏觀基本面抓取器：景氣信號、市值權重 (巨量)、產業鏈
完美對接 core/ 五大核心模組，具備 30 天智慧分段與 402 自動等待機制。

涵蓋資料表：
  · business_indicator     ─ 景氣對策信號 (TaiwanBusinessIndicator)
  · market_value_weight    ─ 台股個股權重 (TaiwanStockMarketValueWeight) - 每日約 2000 筆，巨量
  · industry_chain         ─ 產業鏈結構 (TaiwanStockIndustryChain)

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池管理 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ 智慧斷路器 (排除業務錯誤) + Singleton 單例連線
  · path_setup v3.0          ─ 自動修復系統路徑

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 對接 FinMindClient v5.1 智慧斷路器。
    - [監控] 確保 rows_inserted 精準追蹤。
  v5.0 (2026-05-09):
    - [核心] 修正 ImportError，移除 finmind_get，全面對接 FinMindClient().get_data()。
  v3.3 (2026-04-20):
    - [基礎] 初步引入時間切塊 (Chunking) 概念以處理市值權重數據。
  v3.0 (2026-04-10):
    - [結構] 整合 path_setup 與 ensure_ddl，具備自動建表能力。

執行範例：
    # 範例 1：常規增量抓取 (所有資料表)
    python scripts/fetchers/fetch_macro_fundamental_data.py
    
    # 範例 2：僅抓取市值權重並指定起始日期與強制模式 (歷史資料補齊)
    python scripts/fetchers/fetch_macro_fundamental_data.py --tables market_value_weight --start 2024-01-01 --force
    
    # 範例 3：指定特定日期區間，並觀察 30 天切塊運作情況
    python scripts/fetchers/fetch_macro_fundamental_data.py --start 2023-01-01 --end 2023-12-31
"""

import sys
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
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
        get_latest_date, write_fetch_log, FailureLogger,
        safe_int, safe_float
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

DDL_MACRO_FUND = """
CREATE TABLE IF NOT EXISTS business_indicator (
    date DATE PRIMARY KEY, 
    "leading" NUMERIC(20,6), 
    leading_notrend NUMERIC(20,6), 
    coincident NUMERIC(20,6), 
    coincident_notrend NUMERIC(20,6), 
    lagging NUMERIC(20,6), 
    lagging_notrend NUMERIC(20,6), 
    monitoring NUMERIC(20,6), 
    monitoring_color VARCHAR(20)
);
CREATE TABLE IF NOT EXISTS market_value_weight (
    date DATE, 
    stock_id VARCHAR(50), 
    stock_name VARCHAR(100), 
    rank INTEGER, 
    weight_per NUMERIC(20,6), 
    type VARCHAR(10), 
    PRIMARY KEY (date, stock_id)
);
CREATE TABLE IF NOT EXISTS industry_chain (
    stock_id VARCHAR(50) PRIMARY KEY, 
    industry VARCHAR(100), 
    sub_industry VARCHAR(100), 
    date DATE
);
"""

UPSERT_MAP = {
    "business_indicator": """
        INSERT INTO business_indicator (date, "leading", coincident, lagging, monitoring, monitoring_color)
        VALUES (%(date)s, %(leading)s, %(coincident)s, %(lagging)s, %(monitoring)s, %(monitoring_color)s)
        ON CONFLICT (date) DO UPDATE SET monitoring = EXCLUDED.monitoring;
    """,
    "market_value_weight": """
        INSERT INTO market_value_weight (date, stock_id, stock_name, rank, weight_per, type)
        VALUES (%(date)s, %(stock_id)s, %(stock_name)s, %(rank)s, %(weight_per)s, %(type)s)
        ON CONFLICT (date, stock_id) DO UPDATE SET weight_per = EXCLUDED.weight_per;
    """,
    "industry_chain": """
        INSERT INTO industry_chain (stock_id, industry, sub_industry, date)
        VALUES (%(stock_id)s, %(industry)s, %(sub_industry)s, %(date)s)
        ON CONFLICT (stock_id) DO UPDATE SET industry = EXCLUDED.industry;
    """
}

DATASET_MAP = {
    "business_indicator": "TaiwanBusinessIndicator",
    "market_value_weight": "TaiwanStockMarketValueWeight",
    "industry_chain": "TaiwanStockIndustryChain"
}

# =====================================================================
# 2. Mappers
# =====================================================================

def map_bi(r):
    return {"date": r["date"], "leading": safe_float(r.get("leading")), "coincident": safe_float(r.get("coincident")), "lagging": safe_float(r.get("lagging")), "monitoring": safe_float(r.get("monitoring")), "monitoring_color": r.get("monitoring_color", "")[:20]}

def map_mvw(r):
    return {"date": r["date"], "stock_id": str(r.get("stock_id", "")), "stock_name": r.get("stock_name", "")[:100], "rank": safe_int(r.get("rank")), "weight_per": safe_float(r.get("weight_per")), "type": r.get("type", "")[:10]}

def map_ic(r):
    return {"stock_id": str(r.get("stock_id", "")), "industry": r.get("industry", "")[:100], "sub_industry": r.get("sub_industry", "")[:100], "date": r.get("date")}

MAPPER_MAP = {"business_indicator": map_bi, "market_value_weight": map_mvw, "industry_chain": map_ic}

# =====================================================================
# 3. 核心邏輯
# =====================================================================

def fetch_table(table: str, start: str, end: str, force: bool):
    api = FinMindClient()
    cur_start = start
    
    if table != "industry_chain" and not force:
        latest = get_latest_date(table)
        if latest:
            d = datetime.strptime(str(latest), "%Y-%m-%d").date()
            cur_start = (d + timedelta(days=1)).strftime("%Y-%m-%d")
            if cur_start > end:
                write_fetch_log(table, "ALL", "skipped", "macro_fund_v5", str(latest), end, 0, 0, "up_to_date")
                return

    try:
        t0 = time.monotonic()
        total_success = 0
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        chunk_days = 30 if table == "market_value_weight" else 365
        
        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            data = api.get_data(DATASET_MAP[table], "", seg_start, seg_end)
            if data:
                records = [MAPPER_MAP[table](row) for row in data if "date" in row or "stock_id" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], "ALL")
                    total_success += success
            
            start_dt = seg_end_dt + timedelta(days=1)
            if table == "market_value_weight": time.sleep(0.5)

        duration_ms = int((time.monotonic() - t0) * 1000)
        status = "success" if total_success > 0 else "no_new_data"
        write_fetch_log(table, "ALL", status, "macro_fund_v5", cur_start, end, duration_ms, total_success, None)
        logger.info(f"  ✓ {table}: {total_success} rows")
        
    except Exception as e:
        logger.error(f"  ❌ {table}: {e}")
        write_fetch_log(table, "ALL", "failed", "macro_fund_v5", cur_start, end, 0, 0, str(e))

def main():
    parser = argparse.ArgumentParser(description="Macro Fundamental Fetcher v5.0")
    parser.add_argument("--tables", type=str, default="all")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_ddl(DDL_MACRO_FUND)
    end_date = args.end or date.today().strftime("%Y-%m-%d")
    table_list = ["business_indicator", "market_value_weight", "industry_chain"] if args.tables == "all" else args.tables.split(",")

    logger.info("=" * 70)
    logger.info(f"  Macro Fundamental Fetcher v5.0  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)

    for tbl in table_list:
        fetch_table(tbl.strip(), args.start, end_date, args.force)

    logger.info("🎉 任務完成。")
    get_request_stats().summary()

if __name__ == "__main__":
    main()