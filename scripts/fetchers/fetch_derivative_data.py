import sys
import json
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_derivative_data.py — 期貨/選擇權日成交（逐支 commit 完整性版）
====================================================================
v2.2 改進：
  · safe_commit_rows()：每支商品寫入後立即 commit，失敗 rollback。
  · 主迴圈以 try/except 包單一商品，單個商品失敗不影響其他商品。
  · 失敗清單寫入 outputs/{table}_failed_{date}.json。

執行範例：
    python fetch_derivative_data.py
    python fetch_derivative_data.py --tables futures_daily
    python fetch_derivative_data.py --force
    python fetch_derivative_data.py --start 2024-01-01 --end 2026-03-19
"""

import argparse
import logging
import time
from datetime import date, timedelta, datetime

import psycopg2

from core.finmind_client import finmind_get, wait_until_next_hour  # noqa: F401
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    bulk_upsert,
    safe_float,
    safe_int,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 各資料集最早可用日期
DATASET_START_DATES = {
    "futures_daily": "1998-07-01",
    "option_daily":  "2001-12-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1998-07-01"

# 常用商品代碼降級備案（TaiwanFutOptDailyInfo 回傳空清單時使用）
FALLBACK_FUTURES_IDS = [
    "TX", "MTX", "TXO", "TE", "TF", "XIF", "G2F", "GDF", "BRF", "SPF",
    "UDF", "NDF", "RHF", "RTF", "SJF", "EXF", "TGF", "GTF", "FTX", "E4F",
]
FALLBACK_OPTIONS_IDS = [
    "TXO", "TEO", "TFO", "XIO", "GTF", "TGO",
]

TYPE_MAP = {
    "futures": ["TaiwanFuturesDaily", "期貨", "futures", "future"],
    "options": ["TaiwanOptionDaily", "選擇權", "options", "option"],
}

# ──────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────
DDL_FUTURES_DAILY = """
CREATE TABLE IF NOT EXISTS futures_ohlcv (
    date             DATE,
    futures_id       VARCHAR(50),
    contract_date    VARCHAR(6),
    open             NUMERIC(10,4),
    max              NUMERIC(10,4),
    min              NUMERIC(10,4),
    close            NUMERIC(10,4),
    spread           NUMERIC(10,4),
    spread_per       NUMERIC(5,2),
    volume           BIGINT,
    settlement_price NUMERIC(10,4),
    open_interest    BIGINT,
    trading_session  VARCHAR(20),
    PRIMARY KEY (date, futures_id, contract_date)
);
CREATE INDEX IF NOT EXISTS idx_futures_ohlcv_futures_id ON futures_ohlcv (futures_id);
"""

DDL_OPTION_DAILY = """
CREATE TABLE IF NOT EXISTS options_ohlcv (
    date             DATE,
    option_id        VARCHAR(50),
    contract_date    VARCHAR(6),
    strike_price     NUMERIC(10,4),
    call_put         VARCHAR(4),
    open             NUMERIC(10,4),
    max              NUMERIC(10,4),
    min              NUMERIC(10,4),
    close            NUMERIC(10,4),
    volume           BIGINT,
    settlement_price NUMERIC(10,4),
    open_interest    BIGINT,
    trading_session  VARCHAR(20),
    PRIMARY KEY (date, option_id, contract_date, strike_price, call_put)
);
CREATE INDEX IF NOT EXISTS idx_options_ohlcv_option_id ON options_ohlcv (option_id);
"""

# ──────────────────────────────────────────────
# Upsert SQL
# ──────────────────────────────────────────────
UPSERT_FUTURES_DAILY = """
INSERT INTO futures_ohlcv (
    date, futures_id, contract_date,
    open, max, min, close, spread, spread_per,
    volume, settlement_price, open_interest, trading_session
) VALUES %s
ON CONFLICT (date, futures_id, contract_date) DO UPDATE SET
    open             = EXCLUDED.open,
    max              = EXCLUDED.max,
    min              = EXCLUDED.min,
    close            = EXCLUDED.close,
    spread           = EXCLUDED.spread,
    spread_per       = EXCLUDED.spread_per,
    volume           = EXCLUDED.volume,
    settlement_price = EXCLUDED.settlement_price,
    open_interest    = EXCLUDED.open_interest,
    trading_session  = EXCLUDED.trading_session;
"""

UPSERT_OPTION_DAILY = """
INSERT INTO options_ohlcv (
    date, option_id, contract_date, strike_price, call_put,
    open, max, min, close,
    volume, settlement_price, open_interest, trading_session
) VALUES %s
ON CONFLICT (date, option_id, contract_date, strike_price, call_put) DO UPDATE SET
    open             = EXCLUDED.open,
    max              = EXCLUDED.max,
    min              = EXCLUDED.min,
    close            = EXCLUDED.close,
    volume           = EXCLUDED.volume,
    settlement_price = EXCLUDED.settlement_price,
    open_interest    = EXCLUDED.open_interest,
    trading_session  = EXCLUDED.trading_session;
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
        logger.info(f"  失敗清單已寫入：{out}（{len(failures)} 個商品）")
    except Exception as e:
        logger.warning(f"  寫入失敗清單時發生錯誤：{e}")


def dedup_rows(rows: list, key_indices: tuple) -> list:
    seen = {}
    for row in rows:
        key = tuple(row[i] for i in key_indices)
        seen[key] = row
    deduped = list(seen.values())
    if len(deduped) < len(rows):
        removed = len(rows) - len(deduped)
        logger.debug(f"dedup_rows：去除 {removed} 筆重複 PK 列")
    return deduped


def get_latest_date(conn, table: str, id_col: str, data_id: str):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MAX(date) FROM {table} WHERE {id_col} = %s",
            (data_id,)
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
        return None


def resolve_start(conn, table: str, id_col: str, data_id: str,
                  global_start: str, dataset_key: str, force: bool):
    earliest = DATASET_START_DATES[dataset_key]
    effective_start = max(global_start, earliest)

    if force:
        return effective_start

    latest = get_latest_date(conn, table, id_col, data_id)
    if latest is None:
        return effective_start

    next_day = (
        datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if next_day > DEFAULT_END:
        return None

    return max(next_day, earliest)


def _extract_instrument_id(r: dict) -> str:
    for field in ("futures_id", "option_id", "code", "id", "symbol"):
        val = r.get(field)
        if val and str(val).strip():
            return str(val).strip()
    return ""


def get_instrument_ids(delay: float, instrument_type: str) -> list:
    fallback = (
        FALLBACK_FUTURES_IDS if instrument_type == "futures"
        else FALLBACK_OPTIONS_IDS
    )

    try:
        data = finmind_get("TaiwanFutOptDailyInfo", {}, delay)
    except Exception as e:
        logger.warning(f"TaiwanFutOptDailyInfo 取得失敗：{e}，改用 fallback")
        return list(fallback)

    if not data:
        logger.warning(
            f"TaiwanFutOptDailyInfo 回傳空資料，"
            f"改用預設 {instrument_type} 商品代碼（共 {len(fallback)} 個）"
        )
        return list(fallback)

    logger.info(
        f"TaiwanFutOptDailyInfo 回傳 {len(data)} 筆，"
        f"欄位名稱：{list(data[0].keys())}"
    )
    unique_types = set(str(r.get("type", "")).strip() for r in data)
    logger.info(f"  type 值分布：{unique_types}")

    keywords = TYPE_MAP.get(instrument_type, [])
    ids = [
        _extract_instrument_id(r)
        for r in data
        if any(
            str(r.get("type", "")).strip().lower() == kw.lower()
            for kw in keywords
        )
    ]
    ids = [i for i in ids if i]

    if ids:
        logger.info(f"取得 {instrument_type} 商品共 {len(ids)} 個：{ids}")
        return ids

    logger.warning(
        f"比對 type={keywords} 後無結果（實際 type：{unique_types}），"
        f"改用預設 {instrument_type} 商品代碼（共 {len(fallback)} 個）"
    )
    return list(fallback)


# ──────────────────────────────────────────────
# futures_daily（期貨日成交）
# ──────────────────────────────────────────────
def fetch_futures_daily(start_date: str, end_date: str, delay: float, force: bool, target_ids: list = None):
    logger.info("=== [futures_daily] 開始抓取 ===")
    conn = get_db_conn()
    failures = []
    try:
        ensure_ddl(conn, DDL_FUTURES_DAILY)
        conn.commit()

        if target_ids:
            futures_ids = target_ids
        else:
            futures_ids = get_instrument_ids(delay, "futures")

        if not futures_ids:
            logger.error("[futures_daily] 商品代碼清單為空，無法繼續")
            return

        total_rows = 0
        skipped    = 0

        for i, fid in enumerate(futures_ids, 1):
            try:
                actual_start = resolve_start(
                    conn, "futures_ohlcv", "futures_id", fid,
                    start_date, "futures_daily", force
                )
                if actual_start is None:
                    skipped += 1
                    continue

                logger.info(
                    f"  [{i}/{len(futures_ids)}] {fid}  {actual_start} ~ {end_date}"
                )
                data = finmind_get(
                    "TaiwanFuturesDaily",
                    {"data_id": fid, "start_date": actual_start, "end_date": end_date},
                    delay,
                )
                if not data:
                    continue

                rows = []
                for r in data:
                    try:
                        rows.append((
                            r["date"],
                            r.get("futures_id", fid),
                            str(r.get("contract_date", ""))[:6],
                            safe_float(r.get("open")),
                            safe_float(r.get("max")),
                            safe_float(r.get("min")),
                            safe_float(r.get("close")),
                            safe_float(r.get("spread")),
                            safe_float(r.get("spread_per")),
                            safe_int(r.get("volume")),
                            safe_float(r.get("settlement_price")),
                            safe_int(r.get("open_interest")),
                            str(r.get("trading_session", "") or "")[:20],
                        ))
                    except Exception as e:
                        logger.warning(f"    [{fid}] mapper 異常筆，跳過：{e}")
                rows = dedup_rows(rows, key_indices=(0, 1, 2))
                n = safe_commit_rows(
                    conn, UPSERT_FUTURES_DAILY, rows,
                    (
                        "(%s::date, %s, %s,"
                        " %s::numeric, %s::numeric, %s::numeric, %s::numeric,"
                        " %s::numeric, %s::numeric,"
                        " %s, %s::numeric, %s, %s)"
                    ),
                    label=f"futures_daily/{fid}",
                )
                total_rows += n
                logger.info(f"    → 寫入 {n} 筆（累計 {total_rows}）")
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                failures.append({"futures_id": fid, "error": str(e)})
                logger.error(f"  [futures_daily/{fid}] 失敗：{e}")

    finally:
        conn.close()
    dump_failures("futures_daily", failures)
    logger.info(
        f"=== [futures_daily] 完成，共寫入 {total_rows} 筆"
        f"（略過已最新：{skipped} 個，失敗：{len(failures)} 個）==="
    )


# ──────────────────────────────────────────────
# option_daily（選擇權日成交）
# ──────────────────────────────────────────────
def fetch_option_daily(start_date: str, end_date: str, delay: float, force: bool, target_ids: list = None):
    logger.info("=== [option_daily] 開始抓取 ===")
    conn = get_db_conn()
    failures = []
    try:
        ensure_ddl(conn, DDL_OPTION_DAILY)
        conn.commit()

        if target_ids:
            option_ids = target_ids
        else:
            option_ids = get_instrument_ids(delay, "options")

        if not option_ids:
            logger.error("[option_daily] 商品代碼清單為空，無法繼續")
            return

        total_rows = 0
        skipped    = 0

        for i, oid in enumerate(option_ids, 1):
            try:
                actual_start = resolve_start(
                    conn, "options_ohlcv", "option_id", oid,
                    start_date, "option_daily", force
                )
                if actual_start is None:
                    skipped += 1
                    continue

                logger.info(
                    f"  [{i}/{len(option_ids)}] {oid}  {actual_start} ~ {end_date}"
                )
                data = finmind_get(
                    "TaiwanOptionDaily",
                    {"data_id": oid, "start_date": actual_start, "end_date": end_date},
                    delay,
                )
                if not data:
                    continue

                rows = []
                for r in data:
                    try:
                        rows.append((
                            r["date"],
                            r.get("option_id", oid),
                            str(r.get("contract_date", ""))[:6],
                            safe_float(r.get("strike_price")),
                            str(r.get("call_put", "") or "")[:4],
                            safe_float(r.get("open")),
                            safe_float(r.get("max")),
                            safe_float(r.get("min")),
                            safe_float(r.get("close")),
                            safe_int(r.get("volume")),
                            safe_float(r.get("settlement_price")),
                            safe_int(r.get("open_interest")),
                            str(r.get("trading_session", "") or "")[:20],
                        ))
                    except Exception as e:
                        logger.warning(f"    [{oid}] mapper 異常筆，跳過：{e}")
                rows = dedup_rows(rows, key_indices=(0, 1, 2, 3, 4))
                n = safe_commit_rows(
                    conn, UPSERT_OPTION_DAILY, rows,
                    (
                        "(%s::date, %s, %s, %s::numeric, %s,"
                        " %s::numeric, %s::numeric, %s::numeric, %s::numeric,"
                        " %s, %s::numeric, %s, %s)"
                    ),
                    label=f"option_daily/{oid}",
                )
                total_rows += n
                logger.info(f"    → 寫入 {n} 筆（累計 {total_rows}）")
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                failures.append({"option_id": oid, "error": str(e)})
                logger.error(f"  [option_daily/{oid}] 失敗：{e}")

    finally:
        conn.close()
    dump_failures("option_daily", failures)
    logger.info(
        f"=== [option_daily] 完成，共寫入 {total_rows} 筆"
        f"（略過已最新：{skipped} 個，失敗：{len(failures)} 個）==="
    )


# ──────────────────────────────────────────────
# CLI 主程式
# ──────────────────────────────────────────────
TABLE_FUNCS = {
    "futures_daily": fetch_futures_daily,
    "option_daily":  fetch_option_daily,
}


def parse_args():
    parser = argparse.ArgumentParser(description="FinMind 衍生品資料抓取工具 v2.2")
    parser.add_argument(
        "--tables", nargs="+",
        choices=list(TABLE_FUNCS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ids", nargs="+",
                        help="指定要抓取的商品代碼（例如 TX TFO CDF）")
    return parser.parse_args()


def main():
    args = parse_args()
    tables = list(TABLE_FUNCS.keys()) if "all" in args.tables else args.tables

    mode = "強制重抓" if args.force else "增量模式（自動跳過已最新資料）"
    logger.info(f"抓取資料表：{tables}")
    if args.ids:
        logger.info(f"指定商品代碼：{args.ids}")
    logger.info(f"日期區間：{args.start} ~ {args.end}")
    logger.info(f"請求間隔：{args.delay} 秒")
    logger.info(f"執行模式：{mode}")

    for table in tables:
        try:
            TABLE_FUNCS[table](args.start, args.end, args.delay, args.force, args.ids)
        except psycopg2.OperationalError as e:
            logger.error(f"PostgreSQL 連線失敗：{e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"[{table}] 未預期錯誤：{e}")
            continue


if __name__ == "__main__":
    main()