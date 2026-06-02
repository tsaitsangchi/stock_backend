"""
fetch_international_data.py — 國際市場資料（v3.2 核心模組全面升級版）
================================================================================
v3.2 改進（配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 修正 `finmind_get` 參數傳遞方式為具名參數 (Keyword Arguments) 避免型別崩潰。
  ★ 程式結束時自動印出 `finmind_client` 的 `RequestStats` 統計報表。

支援資料表：
  · us_stock_price      (美股價量：AAPL, NVDA, TSLA, ...)
  · crude_oil_prices    (原油價格：WTI, Brent)
  · gold_price          (國際金價)

執行範例（常規）：
    # 抓取所有國際資料
    python scripts/fetchers/fetch_international_data.py
    
    # 僅抓取指定美股
    python scripts/fetchers/fetch_international_data.py --ids AAPL,NVDA
    
    # 僅抓取金價
    python scripts/fetchers/fetch_international_data.py --tables gold_price

執行範例（強制重抓）：
    python scripts/fetchers/fetch_international_data.py --ids NVDA --force
    python scripts/fetchers/fetch_international_data.py --tables all --force --start 2020-01-01

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的目標
    python scripts/fetchers/fetch_international_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的資料
    python scripts/fetchers/fetch_international_data.py --gap-fill 30
    
    # 針對美股特定補抓
    python scripts/fetchers/fetch_international_data.py --gap-fill 14 --tables us_stock_price
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse

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
    safe_float,
    safe_int,
    get_all_safe_starts,
    get_market_safe_start,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    commit_per_day,
    dedup_rows,
    DDL_FETCH_LOG
)

# 依賴專案內定義好的國際關注清單
try:
    from config import INTERNATIONAL_WATCHLIST
except ImportError:
    INTERNATIONAL_WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATASET_START_DATES = {
    "us_stock_price":   "1962-01-02",
    "crude_oil_prices": "1987-05-20",
    "gold_price":       "1968-04-01",
}
CRUDE_OIL_IDS = ["WTI", "Brent"]
DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1962-01-02"

_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 日誌與 SQL
# ─────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None, fetch_mode="per_stock"):
    """v3.2 標準化日誌寫入，失敗不影響主流程"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (
                    run_ts, table_name, stock_id, fetch_mode, status, rows_inserted, 
                    fetch_date_from, fetch_date_to, duration_ms, error_message, cli_args
                ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (table_name, stock_id, fetch_mode, status, rows_inserted, fetch_date_from, fetch_date_to, duration_ms, error_message, _CLI_ARGS_STR))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"無法寫入 fetch_log: {e}")

DDL_US_STOCK_PRICE = """
CREATE TABLE IF NOT EXISTS us_stock_price (
    date DATE, stock_id VARCHAR(50), adj_close NUMERIC(20,4), close NUMERIC(20,4),
    high NUMERIC(20,4), low NUMERIC(20,4), open NUMERIC(20,4), volume BIGINT,
    PRIMARY KEY (date, stock_id)
);
"""
DDL_CRUDE_OIL_PRICES = """
CREATE TABLE IF NOT EXISTS crude_oil_prices (
    date DATE, name VARCHAR(50), price NUMERIC(20,4),
    PRIMARY KEY (date, name)
);
"""
DDL_GOLD_PRICE = """
CREATE TABLE IF NOT EXISTS gold_price (
    date DATE, price NUMERIC(20,4),
    PRIMARY KEY (date)
);
"""

UPSERT_US_STOCK_PRICE = """
INSERT INTO us_stock_price (date, stock_id, adj_close, close, high, low, open, volume)
VALUES %s ON CONFLICT (date, stock_id) DO UPDATE SET close = EXCLUDED.close, volume = EXCLUDED.volume;
"""
UPSERT_CRUDE_OIL_PRICES = """
INSERT INTO crude_oil_prices (date, name, price)
VALUES %s ON CONFLICT (date, name) DO UPDATE SET price = EXCLUDED.price;
"""
UPSERT_GOLD_PRICE = """
INSERT INTO gold_price (date, price)
VALUES %s ON CONFLICT (date) DO UPDATE SET price = EXCLUDED.price;
"""

# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def _fmt_date(d):
    s = str(d)
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S" if " " in s else "%Y-%m-%d").strftime("%Y-%m-%d")

def map_us_stock(r: dict) -> tuple:
    return (_fmt_date(r["date"]), r["stock_id"], safe_float(r.get("Adj_Close")), safe_float(r.get("Close")), safe_float(r.get("High")), safe_float(r.get("Low")), safe_float(r.get("Open")), safe_int(r.get("Volume")))

def map_crude_oil(r: dict) -> tuple:
    return (_fmt_date(r["date"]), r["name"], safe_float(r.get("price")))

def map_gold_price(r: dict) -> tuple:
    return (_fmt_date(r["date"]), "GOLD", safe_float(r.get("Price")))

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_us_stock_price(conn, start, end, delay, force, target_ids, fetch_mode_override=None):
    logger.info("=== [us_stock_price] 開始 ===")
    ensure_ddl(conn, DDL_US_STOCK_PRICE)
    stock_ids = target_ids if target_ids else INTERNATIONAL_WATCHLIST
    latest = get_all_safe_starts(conn, "us_stock_price")
    flog = FailureLogger("us_stock_price", db_conn=conn)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"

    for i, sid in enumerate(stock_ids, 1):
        s = resolve_start_cached(sid, latest, start, DATASET_START_DATES["us_stock_price"], force)
        if not s: 
            _write_fetch_log(conn, "us_stock_price", sid, "skipped", fetch_mode=fetch_mode)
            continue
        
        start_ts = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset="USStockPrice", 
                params={"data_id": sid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            duration = int((time.time() - start_ts) * 1000)
            
            if data:
                rows = [map_us_stock(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_US_STOCK_PRICE, rows, "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)", label_prefix="us_stock", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, "us_stock_price", sid, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration, fetch_mode=fetch_mode)
            else:
                _write_fetch_log(conn, "us_stock_price", sid, "no_new_data", duration_ms=duration, fetch_mode=fetch_mode)
        except Exception as e: 
            duration = int((time.time() - start_ts) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(conn, "us_stock_price", sid, "failed", duration_ms=duration, error_message=str(e), fetch_mode=fetch_mode)
            
        if i % 20 == 0: logger.info(f"  進度：{i}/{len(stock_ids)}")

    flog.summary()
    logger.info(f"=== [us_stock_price] 完成：{total_rows} 筆 ===\n")

def fetch_crude_oil_prices(conn, start, end, delay, force, target_ids=None, fetch_mode_override=None):
    logger.info("=== [crude_oil_prices] 開始 ===")
    ensure_ddl(conn, DDL_CRUDE_OIL_PRICES)
    latest = get_all_safe_starts(conn, "crude_oil_prices", key_col="name")
    flog = FailureLogger("crude_oil", db_conn=conn)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    
    ids_to_fetch = target_ids if target_ids else CRUDE_OIL_IDS

    for oid in ids_to_fetch:
        s = resolve_start_cached(oid, latest, start, DATASET_START_DATES["crude_oil_prices"], force)
        if not s:
            _write_fetch_log(conn, "crude_oil_prices", oid, "skipped", fetch_mode=fetch_mode)
            continue
            
        start_ts = time.time()
        try:
            # 修正：全面改為具名參數 (Keyword arguments)
            data = finmind_get(
                dataset="CrudeOilPrices", 
                params={"data_id": oid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            duration = int((time.time() - start_ts) * 1000)
            
            if data:
                rows = [map_crude_oil(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                res = commit_per_stock_per_day(conn, UPSERT_CRUDE_OIL_PRICES, rows, "(%s::date,%s,%s::numeric)", label_prefix="crude_oil", failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(conn, "crude_oil_prices", oid, "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration, fetch_mode=fetch_mode)
            else:
                _write_fetch_log(conn, "crude_oil_prices", oid, "no_new_data", duration_ms=duration, fetch_mode=fetch_mode)
        except Exception as e: 
            duration = int((time.time() - start_ts) * 1000)
            flog.record(stock_id=oid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(conn, "crude_oil_prices", oid, "failed", duration_ms=duration, error_message=str(e), fetch_mode=fetch_mode)

    flog.summary()
    logger.info(f"=== [crude_oil_prices] 完成：{total_rows} 筆 ===\n")

def fetch_gold_price(conn, start, end, delay, force, fetch_mode_override=None):
    logger.info("=== [gold_price] 開始 ===")
    ensure_ddl(conn, DDL_GOLD_PRICE)
    m_start = get_market_safe_start(conn, "gold_price")
    latest = {"GOLD": m_start} if m_start else {}
    flog = FailureLogger("gold_price", db_conn=conn)
    fetch_mode = fetch_mode_override or "market"
    
    s = resolve_start_cached("GOLD", latest, start, DATASET_START_DATES["gold_price"], force)
    if not s: 
        logger.info("  [gold_price] 已是最新。")
        _write_fetch_log(conn, "gold_price", "GOLD", "skipped", fetch_mode=fetch_mode)
        return

    start_ts = time.time()
    try:
        # 修正：全面改為具名參數 (Keyword arguments)
        data = finmind_get(
            dataset="GoldPrice", 
            params={"start_date": s, "end_date": end}, 
            delay=delay,
            raise_on_error=True
        )
        duration = int((time.time() - start_ts) * 1000)
        
        if data:
            rows_full = [map_gold_price(r) for r in data]
            rows_full = dedup_rows(rows_full, (0, 1))
            
            # Gold price table schema expects (date, price) without the "GOLD" string
            rows_for_commit = [(r[0], r[2]) for r in rows_full]

            UPSERT_GOLD = (
                "INSERT INTO gold_price (date, price) VALUES %s "
                "ON CONFLICT (date) DO UPDATE SET price = EXCLUDED.price"
            )
            res = commit_per_day(
                conn, UPSERT_GOLD, rows_for_commit,
                "(%s::date, %s::numeric)",
                date_index=0, label_prefix="gold_price", failure_logger=flog,
            )
            n = sum(res.values())
            logger.info(f"  [gold_price] 寫入 {n} 筆")
            _write_fetch_log(conn, "gold_price", "GOLD", "success", rows_inserted=n, fetch_date_from=s, fetch_date_to=end, duration_ms=duration, fetch_mode=fetch_mode)
        else:
            _write_fetch_log(conn, "gold_price", "GOLD", "no_new_data", duration_ms=duration, fetch_mode=fetch_mode)
    except Exception as e:
        duration = int((time.time() - start_ts) * 1000)
        flog.record(stock_id="GOLD", error=str(e), start_date=s, end_date=end)
        _write_fetch_log(conn, "gold_price", "GOLD", "failed", duration_ms=duration, error_message=str(e), fetch_mode=fetch_mode)
        
    flog.summary()
    logger.info("=== [gold_price] 完成 ===\n")


# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int, target_tables: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    sql = """
    WITH recent AS (
        SELECT table_name, stock_id, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY table_name, stock_id ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = ANY(%s) AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT table_name, stock_id FROM recent WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (target_tables, str(days)))
            for tbl, sid in cur.fetchall():
                targets[tbl].append(sid)
    except Exception as e:
        logger.error(f"[retry-failed] 查詢失敗：{e}")
        return {}

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [retry-failed/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def query_gap_targets(conn, days: int, target_tables: list[str], target_ids_map: dict[str, list[str]]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    for tbl in target_tables:
        all_ids = target_ids_map.get(tbl, [])
        if not all_ids:
            continue
            
        sql = f"SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tbl, str(days), all_ids))
                have_success = {row[0] for row in cur.fetchall()}
            missing = [sid for sid in all_ids if sid not in have_success]
            targets[tbl].extend(missing)
        except Exception as e:
            logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [gap-fill/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        
        if tbl == "us_stock_price":
            fetch_us_stock_price(conn, args.start, args.end, args.delay, force=True, target_ids=sids, fetch_mode_override=fetch_mode)
        elif tbl == "crude_oil_prices":
            fetch_crude_oil_prices(conn, args.start, args.end, args.delay, force=True, target_ids=sids, fetch_mode_override=fetch_mode)
        elif tbl == "gold_price":
            # Gold is a market-level table, sids will contain "GOLD"
            fetch_gold_price(conn, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["us_stock_price", "crude_oil_prices", "gold_price", "all"], default=["all"])
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--ids", nargs="+", help="針對特定美股或原油代號抓取")
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["us_stock_price", "crude_oil_prices", "gold_price"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        
        # 定義各資料表的全域對應 ID 清單
        target_ids_map = {
            "us_stock_price": args.ids if args.ids else INTERNATIONAL_WATCHLIST,
            "crude_oil_prices": args.ids if args.ids else CRUDE_OIL_IDS,
            "gold_price": ["GOLD"]
        }

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, tables)
            if targets: _run_targeted(conn, targets, args, fetch_mode="retry")
            else: logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, tables, target_ids_map)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        if "us_stock_price" in tables: 
            fetch_us_stock_price(conn, args.start, args.end, args.delay, args.force, args.ids)
        if "crude_oil_prices" in tables: 
            fetch_crude_oil_prices(conn, args.start, args.end, args.delay, args.force, args.ids)
        if "gold_price" in tables: 
            fetch_gold_price(conn, args.start, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        get_request_stats().summary()

if __name__ == "__main__":
    main()