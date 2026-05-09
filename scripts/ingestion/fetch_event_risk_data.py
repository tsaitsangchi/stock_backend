"""
fetch_event_risk_data.py v5.1 (Trinity Core Edition)
================================================================================
事件風險與市值抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取個股市值、處置股資訊以及減資參考價。

涵蓋資料表：
  · market_value           ─ 個股市值 (FinMind: TaiwanStockMarketValue)
  · disposition_securities ─ 處置股資訊 (FinMind: TaiwanStockDispositionSecuritiesPeriod)
  · capital_reduction      ─ 減資參考價 (FinMind: TaiwanStockCapitalReductionReferencePrice)

核心功能：
  · 非法 ID 過濾     ─ 自動跳過非個股 ID (如 Automobile, TAIEX)，根治斷路器誤觸。
  · 智慧分段抓取     ─ 實作 90 天切塊邏輯，解決 V4 API 長區間請求導致的 422 錯誤。
  · 並發處理機制     ─ 支援 --workers，大幅提升全市場市值資料的同步效率。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤 (rows_inserted)
  · finmind_client v5.1      ─ Singleton + 智慧斷路器 (排除業務錯誤)
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [穩定] 導入 is_valid_stock_id 過濾機制，徹底解決 Automobile 等非個股 ID 觸發熔斷的問題。
    - [核心] 對接 FinMindClient v5.1 智慧斷路器，業務級 404/422 不再計入錯誤閾值。
    - [監控] 補齊 write_fetch_log 的 rows_inserted 參數，實現精準監控。
  v5.0 (2026-05-09):
    - [核心] 修正 ensure_ddl 參數報錯，對接 Core v4.5 並發架構。

執行範例：
    # 範例 1：抓取全市場市值 (增量更新)
    python scripts/fetchers/fetch_event_risk_data.py --tables market_value --all --workers 5
    
    # 範例 2：抓取特定個股的事件風險 (處置股、減資)
    python scripts/fetchers/fetch_event_risk_data.py --stock-id 2330,2317 --force
"""

import sys
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
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
        get_latest_date, write_fetch_log, FailureLogger,
        safe_int, safe_float, get_db_stock_ids
    )
    from core.finmind_client import FinMindClient, get_request_stats
    from core.migrate_stocks_config import migrate as sync_stocks_table
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. DDL 與 SQL 對照表
# =====================================================================

DDL_MAP = {
    "market_value": """CREATE TABLE IF NOT EXISTS market_value (date DATE NOT NULL, stock_id VARCHAR(50) NOT NULL, market_value BIGINT, PRIMARY KEY (date, stock_id)); CREATE INDEX IF NOT EXISTS idx_mv_stock_date ON market_value(stock_id, date DESC);""",
    "disposition_securities": """CREATE TABLE IF NOT EXISTS disposition_securities (date DATE NOT NULL, stock_id VARCHAR(50) NOT NULL, stock_name VARCHAR(200), disposition_cnt INTEGER, condition VARCHAR(500), measure VARCHAR(500), period_start DATE, period_end DATE, PRIMARY KEY (date, stock_id)); CREATE INDEX IF NOT EXISTS idx_disp_stock_date ON disposition_securities(stock_id, date DESC);""",
    "capital_reduction": """CREATE TABLE IF NOT EXISTS capital_reduction (date DATE NOT NULL, stock_id VARCHAR(50) NOT NULL, closing_last_trading NUMERIC(20,6), post_reduction_ref NUMERIC(20,6), limit_up NUMERIC(20,6), limit_down NUMERIC(20,6), opening_ref NUMERIC(20,6), exright_ref NUMERIC(20,6), reason VARCHAR(500), PRIMARY KEY (date, stock_id));"""
}

UPSERT_MAP = {
    "market_value": """INSERT INTO market_value (date, stock_id, market_value) VALUES (%(date)s, %(stock_id)s, %(market_value)s) ON CONFLICT (date, stock_id) DO UPDATE SET market_value = EXCLUDED.market_value;""",
    "disposition_securities": """INSERT INTO disposition_securities (date, stock_id, stock_name, disposition_cnt, condition, measure, period_start, period_end) VALUES (%(date)s, %(stock_id)s, %(stock_name)s, %(disposition_cnt)s, %(condition)s, %(measure)s, %(period_start)s, %(period_end)s) ON CONFLICT (date, stock_id) DO UPDATE SET disposition_cnt = EXCLUDED.disposition_cnt;""",
    "capital_reduction": """INSERT INTO capital_reduction (date, stock_id, closing_last_trading, post_reduction_ref, limit_up, limit_down, opening_ref, exright_ref, reason) VALUES (%(date)s, %(stock_id)s, %(closing_last_trading)s, %(post_reduction_ref)s, %(limit_up)s, %(limit_down)s, %(opening_ref)s, %(exright_ref)s, %(reason)s) ON CONFLICT (date, stock_id) DO UPDATE SET post_reduction_ref = EXCLUDED.post_reduction_ref;"""
}

DATASET_MAP = {
    "market_value":           "TaiwanStockMarketValue",
    "disposition_securities": "TaiwanStockDispositionSecuritiesPeriod",
    "capital_reduction":      "TaiwanStockCapitalReductionReferencePrice",
}

# =====================================================================
# 2. Mappers 與工具
# =====================================================================

def map_mv(r: dict) -> dict:
    return {"date": r["date"], "stock_id": r["stock_id"], "market_value": safe_int(r.get("market_value", 0))}

def map_disp(r: dict) -> dict:
    return {"date": r["date"], "stock_id": r["stock_id"], "stock_name": str(r.get("stock_name", ""))[:200], "disposition_cnt": safe_int(r.get("disposition_cnt", 0)), "condition": str(r.get("condition", ""))[:500], "measure": str(r.get("measure", ""))[:500], "period_start": r.get("period_start"), "period_end": r.get("period_end")}

def map_capred(r: dict) -> dict:
    return {"date": r["date"], "stock_id": r["stock_id"], "closing_last_trading": safe_float(r.get("ClosingPriceonTheLastTradingDay")), "post_reduction_ref": safe_float(r.get("PostReductionReferencePrice")), "limit_up": safe_float(r.get("LimitUp")), "limit_down": safe_float(r.get("LimitDown")), "opening_ref": safe_float(r.get("OpeningReferencePrice")), "exright_ref": safe_float(r.get("ExrightReferencePrice")), "reason": str(r.get("ReasonforCapitalReduction", ""))[:500]}

MAPPER_MAP = {"market_value": map_mv, "disposition_securities": map_disp, "capital_reduction": map_capred}

def is_valid_stock_id(sid: str) -> bool:
    """過濾產業指數名稱 (如 Automobile, TAIEX)"""
    return sid.isdigit() and len(sid) >= 4

def taipei_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    except: return date.today().strftime("%Y-%m-%d")

def next_day(date_str: str) -> str:
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (d + timedelta(days=1)).strftime("%Y-%m-%d")
    except: return str(date_str)

def resolve_stock_ids(args) -> List[str]:
    if args.stock_id: return [s.strip() for s in args.stock_id.split(",") if s.strip()]
    try:
        with db_session() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE ORDER BY stock_id")
                return [r[0] for r in cur.fetchall()]
    except: return []

# =====================================================================
# 3. 核心抓取邏輯
# =====================================================================

def fetch_one_stock(table: str, sid: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    if not is_valid_stock_id(sid):
        write_fetch_log(table, sid, "skipped", "event_v5.1", start, end, 0, 0, "invalid_stock_id")
        return sid, 0, 0

    cur_start = start
    if not force:
        latest = get_latest_date(table, sid)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, sid, "skipped", "event_v5.1", str(latest), end, 0, 0, "up_to_date")
                return sid, 0, 0

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 90
        
        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            data = api.get_data(DATASET_MAP[table], sid, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [MAPPER_MAP[table](row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], sid)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, sid, "success" if error == 0 else "partial", "event_v5.1", seg_start, seg_end, duration_ms, success, None)
            else:
                write_fetch_log(table, sid, "no_new_data", "event_v5.1", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.1)

        return sid, total_success, total_error
    except Exception as e:
        logger.error(f"  ❌ {sid} @ {table}: {e}")
        write_fetch_log(table, sid, "failed", "event_v5.1", cur_start, end, 0, 0, str(e))
        return sid, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Event Risk Fetcher v5.1 (Trinity Core Edition)")
    parser.add_argument("--tables", type=str, default="all")
    parser.add_argument("--stock-id", type=str, default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    try: sync_stocks_table()
    except: pass

    table_inputs = [t.strip().lower() for t in args.tables.split(",") if t.strip()]
    target_tables = list(MAPPER_MAP.keys()) if "all" in table_inputs else [t for t in table_inputs if t in MAPPER_MAP]
    stock_ids = resolve_stock_ids(args)
    if not stock_ids:
        logger.error("找不到個股清單。")
        sys.exit(1)

    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Event Risk Fetcher v5.1  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)
    logger.info(f"  目標個股    : {len(stock_ids)}")
    logger.info(f"  目標資料表  : {target_tables}")
    logger.info(f"  日期區間    : {args.start} ~ {end_date}")
    logger.info(f"  Workers     : {args.workers}")
    logger.info("=" * 70)

    for table in target_tables:
        ensure_ddl(DDL_MAP[table])
        logger.info(f"━━━ 抓取資料表 {table} ({len(stock_ids)} stocks) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_stock, table, sid, args.start, end_date, args.force): sid for sid in stock_ids}
            for fut in as_completed(futures):
                sid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 全部完成")
    try: get_request_stats().summary()
    except: pass

if __name__ == "__main__":
    main()