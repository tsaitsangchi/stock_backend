"""
fetch_chip_data.py v5.1 (Trinity Core Edition)
================================================================================
三大法人籌碼面抓取器 — 完美對接 core/ 五大核心模組
此模組負責抓取法人買賣超、融資融券與外資持股等籌碼核心數據。

涵蓋資料表：
  · institutional_investors_buy_sell ─ 三大法人買賣超
  · margin_purchase_short_sale      ─ 融資融券餘額
  · shareholding                    ─ 外資持股比例

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 引入 is_valid_stock_id 過濾非數字 ID，防止產業名稱觸發 422 導致斷路器熔斷。
    - [監控] 強化 write_fetch_log，明確標記 "circuit_open" 狀態，精準呈現熔斷現場。
    - [效能] 對接 Core v5.1 智慧斷路器，業務錯誤不再導致全線停擺。
  v5.0 (2026-05-09):
    - [監控] 對接 Core v4.6，補齊 write_fetch_log 的 rows_inserted 參數。

執行範例：
    # 範例 1：全市場掃描並使用 5 執行緒 (依 stocks 表設定)
    python scripts/fetchers/fetch_chip_data.py --all --workers 5
    
    # 範例 2：針對特定個股進行除錯與強制抓取
    python scripts/fetchers/fetch_chip_data.py --stock-id 2330 --force
"""

import sys
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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
# 1. 配置與 DDL (同 v5.0)
# =====================================================================

DDL_MAP = {
    "institutional_investors_buy_sell": """
        CREATE TABLE IF NOT EXISTS institutional_investors_buy_sell (
            date DATE NOT NULL, stock_id VARCHAR(20) NOT NULL,
            buy BIGINT, sell BIGINT, name VARCHAR(50) NOT NULL,
            PRIMARY KEY (date, stock_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_inst_stock_date ON institutional_investors_buy_sell(stock_id, date DESC);
    """,
    "margin_purchase_short_sale": """
        CREATE TABLE IF NOT EXISTS margin_purchase_short_sale (
            date DATE NOT NULL, stock_id VARCHAR(20) NOT NULL,
            MarginPurchaseBuy BIGINT, MarginPurchaseSell BIGINT,
            MarginPurchaseCashRepayment BIGINT, MarginPurchaseYesterdayBalance BIGINT,
            MarginPurchaseTodayBalance BIGINT, MarginPurchaseLimit BIGINT,
            ShortSaleBuy BIGINT, ShortSaleSell BIGINT,
            ShortSaleCashRepayment BIGINT, ShortSaleYesterdayBalance BIGINT,
            ShortSaleTodayBalance BIGINT, ShortSaleLimit BIGINT,
            OffsetLoanAndShort BIGINT, Note VARCHAR(100),
            PRIMARY KEY (date, stock_id)
        );
        CREATE INDEX IF NOT EXISTS idx_margin_stock_date ON margin_purchase_short_sale(stock_id, date DESC);
    """,
    "shareholding": """
        CREATE TABLE IF NOT EXISTS shareholding (
            date DATE NOT NULL, stock_id VARCHAR(20) NOT NULL,
            Shareholding BIGINT, percent NUMERIC(20,6),
            HoldClass VARCHAR(50) NOT NULL,
            PRIMARY KEY (date, stock_id, HoldClass)
        );
        CREATE INDEX IF NOT EXISTS idx_share_stock_date ON shareholding(stock_id, date DESC);
    """
}

UPSERT_MAP = {
    "institutional_investors_buy_sell": """
        INSERT INTO institutional_investors_buy_sell (date, stock_id, buy, sell, name)
        VALUES (%(date)s, %(stock_id)s, %(buy)s, %(sell)s, %(name)s)
        ON CONFLICT (date, stock_id, name) DO UPDATE SET buy = EXCLUDED.buy, sell = EXCLUDED.sell;
    """,
    "margin_purchase_short_sale": """
        INSERT INTO margin_purchase_short_sale (
            date, stock_id, MarginPurchaseBuy, MarginPurchaseSell, MarginPurchaseCashRepayment,
            MarginPurchaseYesterdayBalance, MarginPurchaseTodayBalance, MarginPurchaseLimit,
            ShortSaleBuy, ShortSaleSell, ShortSaleCashRepayment, ShortSaleYesterdayBalance,
            ShortSaleTodayBalance, ShortSaleLimit, OffsetLoanAndShort, Note
        ) VALUES (
            %(date)s, %(stock_id)s, %(MarginPurchaseBuy)s, %(MarginPurchaseSell)s, %(MarginPurchaseCashRepayment)s,
            %(MarginPurchaseYesterdayBalance)s, %(MarginPurchaseTodayBalance)s, %(MarginPurchaseLimit)s,
            %(ShortSaleBuy)s, %(ShortSaleSell)s, %(ShortSaleCashRepayment)s, %(ShortSaleYesterdayBalance)s,
            %(ShortSaleTodayBalance)s, %(ShortSaleLimit)s, %(OffsetLoanAndShort)s, %(Note)s
        ) ON CONFLICT (date, stock_id) DO UPDATE SET
            MarginPurchaseBuy = EXCLUDED.MarginPurchaseBuy, MarginPurchaseSell = EXCLUDED.MarginPurchaseSell,
            MarginPurchaseTodayBalance = EXCLUDED.MarginPurchaseTodayBalance, ShortSaleTodayBalance = EXCLUDED.ShortSaleTodayBalance;
    """,
    "shareholding": """
        INSERT INTO shareholding (date, stock_id, Shareholding, percent, HoldClass)
        VALUES (%(date)s, %(stock_id)s, %(Shareholding)s, %(percent)s, %(HoldClass)s)
        ON CONFLICT (date, stock_id, HoldClass) DO UPDATE SET Shareholding = EXCLUDED.Shareholding, percent = EXCLUDED.percent;
    """
}

DATASET_MAP = {
    "institutional": "TaiwanStockInstitutionalInvestorsBuySell",
    "margin":        "TaiwanStockMarginPurchaseShortSale",
    "shareholding":  "TaiwanStockShareholding",
}

TABLE_TO_DATASET = {
    "institutional": "institutional_investors_buy_sell",
    "margin":        "margin_purchase_short_sale",
    "shareholding":  "shareholding",
}

# =====================================================================
# 2. 工具函式
# =====================================================================

def is_valid_stock_id(sid: str) -> bool:
    """過濾產業指數名稱或無效 ID (如 Automobile, 9955)"""
    return sid.isdigit() and len(sid) >= 4

def taipei_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    except:
        return date.today().strftime("%Y-%m-%d")

def next_day(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (d + timedelta(days=1)).strftime("%Y-%m-%d")
    except:
        return date_str

def resolve_stock_ids(args) -> List[str]:
    if args.all:
        try:
            with db_session() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT stock_id FROM stocks WHERE fetch_chip = TRUE AND is_active = TRUE ORDER BY stock_id")
                    sids = [r[0] for r in cur.fetchall()]
            if sids: return sids
        except: pass
        return get_db_stock_ids() or []
    if not args.stock_id: return []
    return [s.strip() for s in args.stock_id.split(",") if s.strip()]

# =====================================================================
# 3. 核心抓取邏輯 (v5.1 精準監控)
# =====================================================================

def map_row(table: str, row: dict) -> dict:
    if "institutional" in table:
        return {"date": row["date"], "stock_id": row["stock_id"], "buy": safe_int(row.get("buy", 0)), "sell": safe_int(row.get("sell", 0)), "name": row["name"]}
    if "margin" in table:
        return {
            "date": row["date"], "stock_id": row["stock_id"],
            "MarginPurchaseBuy": safe_int(row.get("MarginPurchaseBuy", 0)), "MarginPurchaseSell": safe_int(row.get("MarginPurchaseSell", 0)),
            "MarginPurchaseCashRepayment": safe_int(row.get("MarginPurchaseCashRepayment", 0)), "MarginPurchaseYesterdayBalance": safe_int(row.get("MarginPurchaseYesterdayBalance", 0)),
            "MarginPurchaseTodayBalance": safe_int(row.get("MarginPurchaseTodayBalance", 0)), "MarginPurchaseLimit": safe_int(row.get("MarginPurchaseLimit", 0)),
            "ShortSaleBuy": safe_int(row.get("ShortSaleBuy", 0)), "ShortSaleSell": safe_int(row.get("ShortSaleSell", 0)),
            "ShortSaleCashRepayment": safe_int(row.get("ShortSaleCashRepayment", 0)), "ShortSaleYesterdayBalance": safe_int(row.get("ShortSaleYesterdayBalance", 0)),
            "ShortSaleTodayBalance": safe_int(row.get("ShortSaleTodayBalance", 0)), "ShortSaleLimit": safe_int(row.get("ShortSaleLimit", 0)),
            "OffsetLoanAndShort": safe_int(row.get("OffsetLoanAndShort", 0)), "Note": row.get("Note", "")
        }
    if "shareholding" in table:
        hc = str(row.get("HoldClass", "Unknown"))
        hc = f"Class_{hc}" if hc.isdigit() else hc
        return {"date": row["date"], "stock_id": row["stock_id"], "Shareholding": safe_int(row.get("Shareholding", 0)), "percent": safe_float(row.get("percent", 0.0)), "HoldClass": hc}
    return {}

def fetch_one_stock(table: str, stock_id: str, start: str, end: str, force: bool) -> Tuple[str, int, int]:
    api = FinMindClient()
    
    # 1. 預檢 ID 合法性
    if not is_valid_stock_id(stock_id):
        logger.debug(f"  ⏭️ {stock_id} 為無效代碼，自動跳過。")
        write_fetch_log(table, stock_id, "skipped", "chip_v5.1", start, end, 0, 0, "invalid_stock_id")
        return stock_id, 0, 0

    cur_start = start
    if not force:
        latest = get_latest_date(table, stock_id)
        if latest:
            cur_start = next_day(latest)
            if cur_start > end:
                write_fetch_log(table, stock_id, "skipped", "chip_v5.1", str(latest), end, 0, 0, "up_to_date")
                return stock_id, 0, 0

    try:
        start_dt = datetime.strptime(cur_start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        total_success = total_error = 0
        chunk_days = 30 if "margin" in table else 90
        
        while start_dt <= end_dt:
            seg_start = start_dt.strftime("%Y-%m-%d")
            seg_end_dt = min(start_dt + timedelta(days=chunk_days - 1), end_dt)
            seg_end = seg_end_dt.strftime("%Y-%m-%d")
            
            t0 = time.monotonic()
            data = api.get_data(DATASET_MAP[table.split("_")[0]], stock_id, seg_start, seg_end)
            duration_ms = int((time.monotonic() - t0) * 1000)

            if data:
                records = [map_row(table, row) for row in data if "date" in row]
                if records:
                    success, error = commit_per_stock_per_day(table, records, UPSERT_MAP[table], stock_id)
                    total_success += success
                    total_error += error
                    write_fetch_log(table, stock_id, "success" if error == 0 else "partial", "chip_v5.1", seg_start, seg_end, duration_ms, success, None)
                else:
                    write_fetch_log(table, stock_id, "no_new_data", "chip_v5.1", seg_start, seg_end, duration_ms, 0, "empty_response")
            else:
                write_fetch_log(table, stock_id, "no_new_data", "chip_v5.1", seg_start, seg_end, duration_ms, 0, None)
            
            start_dt = seg_end_dt + timedelta(days=1)
            time.sleep(0.2)

        return stock_id, total_success, total_error

    except Exception as e:
        msg = str(e)
        logger.error(f"  ❌ {stock_id} @ {table}: {msg}")
        status = "circuit_open" if "Circuit Breaker" in msg else "failed"
        write_fetch_log(table, stock_id, status, "chip_v5.1", cur_start, end, 0, 0, msg)
        return stock_id, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Chip Data Fetcher v5.1 (Trinity Core Edition)")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--stock-id", type=str, default="")
    grp.add_argument("--all", action="store_true")
    parser.add_argument("--tables", type=str, default="all")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    try:
        sync_stocks_table()
    except: pass

    stock_ids = resolve_stock_ids(args)
    table_inputs = [t.strip().lower() for t in args.tables.split(",") if t.strip()]
    target_tables = list(UPSERT_MAP.keys()) if "all" in table_inputs else [TABLE_TO_DATASET[t] for t in table_inputs if t in TABLE_TO_DATASET]
    end_date = args.end or taipei_today()

    logger.info("=" * 70)
    logger.info(f"  Chip Data Fetcher v5.1  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 70)

    for table in target_tables:
        ensure_ddl(DDL_MAP[table])
        logger.info(f"━━━ 抓取資料表 {table} ({len(stock_ids)} stocks) ━━━")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one_stock, table, sid, args.start, end_date, args.force): sid for sid in stock_ids}
            for fut in as_completed(futures):
                sid, ok, err = fut.result()
                if ok: logger.info(f"  ✓ {sid}: {ok} rows")

    logger.info("🎉 抓取任務完成。")

if __name__ == "__main__":
    main()