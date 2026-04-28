"""
fetch_technical_data.py  v2.0（API 用量優化版）
從 FinMind API 抓取技術面資料並寫入 PostgreSQL：
  - stock_price ← TaiwanStockPrice (Free, 逐支股票)
  - stock_per   ← TaiwanStockPER   (Free, 逐支股票)

v2.0 優化重點（三層節省 API 用量）：
  ① 批次模式（最大節省）：
       增量更新時，同一起始日的股票合併成「一次不帶 data_id 的全市場請求」
       原：2000 支 × 2 資料集 = 4000 次 API
       後：少數幾次批次請求（依起始日分組）
       --batch-threshold N  超過 N 支同起始日才啟用批次（預設 20）
       --per-stock          停用批次模式，退回逐支請求（相容舊行為）

  ② 合併迴圈（中等節省）：
       price + PER 在同一個股票迴圈內同時抓取，省掉重複遍歷開銷
       逐支模式下：每次迴圈只需對同一支股票發出 2 次請求即可完成

  ③ DB 最新日期批次預載（效能優化）：
       原：每支股票查一次 SELECT MAX(date)（N 次 SQL）
       後：兩條 GROUP BY SQL 一次載入所有股票的最新日期

執行範例：
    # 增量更新（批次模式，API 用量最少）
    python fetch_technical_data.py

    # 指定區間（批次模式）
    python fetch_technical_data.py --start 2025-01-01

    # 退回逐支請求（相容舊行為）
    python fetch_technical_data.py --per-stock

    # 強制重抓全部（逐支模式，資料量大時建議搭配 --tables）
    python fetch_technical_data.py --force --per-stock --tables stock_price

注意事項：
  - 批次模式回傳的是全市場所有股票資料，僅保留 stock_info 內的股票。
  - 批次請求單次回傳筆數可能受 FinMind 限制，程式會自動分段（--chunk-days）。
  - 預設請求間隔 1.2 秒，可用 --delay 調整。
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from datetime import date, timedelta, datetime

import psycopg2
import psycopg2.extras
import requests

# ======================
# 自訂例外
# ======================
class BatchNotSupportedError(Exception):
    """批次請求（不帶 data_id）被 FinMind 拒絕（帳號等級不足），需 fallback 為逐支模式。"""


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
FINMIND_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ"
    ".muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"
)

# ======================
# PostgreSQL 連線設定
# ======================
DB_CONFIG = {
    "dbname": "stock",
    "user": "stock",
    "password": "stock",
    "host": "localhost",
    "port": "5432",
}

# ======================
# 各資料集最早可用日期
# ======================
DATASET_START_DATES = {
    "stock_price": "1994-10-01",
    "stock_per":   "2005-10-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1994-10-01"

# 批次模式預設：一次最多抓幾天（避免單次回傳筆數超過 FinMind 限制）
DEFAULT_CHUNK_DAYS    = 90
# 超過此數量的同起始日股票才啟用批次請求（低於則逐支）
DEFAULT_BATCH_THRESHOLD = 20

# ======================
# DDL
# ======================
DDL_STOCK_PRICE = """
CREATE TABLE IF NOT EXISTS stock_price (
    date             DATE,
    stock_id         VARCHAR(50),
    trading_volume   BIGINT,
    trading_money    BIGINT,
    open             NUMERIC(10,4),
    max              NUMERIC(10,4),
    min              NUMERIC(10,4),
    close            NUMERIC(10,4),
    spread           NUMERIC(10,4),
    trading_turnover INTEGER,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_stock_price_stock_id ON stock_price (stock_id);
"""

DDL_STOCK_PER = """
CREATE TABLE IF NOT EXISTS stock_per (
    date           DATE,
    stock_id       VARCHAR(50),
    dividend_yield NUMERIC(10,4),
    per            NUMERIC(10,4),
    pbr            NUMERIC(10,4),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_stock_per_stock_id ON stock_per (stock_id);
"""

# ======================
# Upsert SQL
# ======================
UPSERT_STOCK_PRICE = """
INSERT INTO stock_price
    (date, stock_id, trading_volume, trading_money, open, max, min, close, spread, trading_turnover)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    trading_volume   = EXCLUDED.trading_volume,
    trading_money    = EXCLUDED.trading_money,
    open             = EXCLUDED.open,
    max              = EXCLUDED.max,
    min              = EXCLUDED.min,
    close            = EXCLUDED.close,
    spread           = EXCLUDED.spread,
    trading_turnover = EXCLUDED.trading_turnover;
"""

UPSERT_STOCK_PER = """
INSERT INTO stock_per (date, stock_id, dividend_yield, per, pbr)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    dividend_yield = EXCLUDED.dividend_yield,
    per            = EXCLUDED.per,
    pbr            = EXCLUDED.pbr;
"""


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────
def safe_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NONE", "NAN", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def safe_int(val):
    f = safe_float(val)
    return int(f) if f is not None else None


def wait_until_next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_sec = (next_hour - now).total_seconds() + 65
    logger.warning(
        f"API 用量達上限（402），等待至下一整點重置。"
        f"目前時間：{now.strftime('%H:%M:%S')}，"
        f"預計恢復：{next_hour.strftime('%H:%M:%S')}，"
        f"等待 {wait_sec:.0f} 秒…"
    )
    time.sleep(wait_sec)
    logger.info("等待結束，恢復請求。")


def finmind_get(dataset: str, params: dict, delay: float) -> list:
    """
    通用 FinMind API 請求。
    params 帶 data_id → 單支股票
    params 不帶 data_id → 全市場（批次模式用）
    """
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    base_params = {"dataset": dataset, **params}

    while True:
        for attempt in range(1, 4):
            try:
                resp = requests.get(
                    FINMIND_API_URL, headers=headers, params=base_params, timeout=(15, 120)
                )
                if resp.status_code == 402:
                    wait_until_next_hour()
                    break
                resp.raise_for_status()
                payload = resp.json()
                status = payload.get("status")
                if status == 402:
                    wait_until_next_hour()
                    break
                if status != 200:
                    msg = payload.get("msg", "")
                    # 批次請求（無 data_id）收到 400 → 帳號等級不足，需 fallback
                    if status == 400 and "data_id" not in params:
                        raise BatchNotSupportedError(
                            f"[{dataset}] 批次請求被拒絕（帳號等級不足），msg={msg}"
                        )
                    logger.warning(
                        f"[{dataset}] 非預期 status={status}, msg={msg}，跳過"
                    )
                    return []
                time.sleep(delay)
                return payload.get("data", [])

            except requests.HTTPError as http_err:
                code = http_err.response.status_code if http_err.response is not None else 0
                if code == 400:
                    # 批次請求（無 data_id）收到 HTTP 400 → 帳號等級不足
                    if "data_id" not in params:
                        raise BatchNotSupportedError(
                            f"[{dataset}] 批次請求被拒絕（HTTP 400，帳號等級不足）"
                        )
                    logger.debug(
                        f"[{dataset}] 400 Bad Request，跳過 "
                        f"(data_id={params.get('data_id')}, date={params.get('start_date')})"
                    )
                    return []
                elif code == 402:
                    wait_until_next_hour()
                    break
                else:
                    logger.warning(f"[{dataset}] HTTP {code} 錯誤：{http_err}")
                    if attempt < 3:
                        time.sleep(delay * 3)
                    else:
                        logger.error(f"[{dataset}] 重試 3 次均失敗，跳過此次請求")
                        return []
            except Exception as exc:
                logger.warning(f"[{dataset}] 第 {attempt} 次請求失敗：{exc}")
                if attempt < 3:
                    time.sleep(delay * 3)
                else:
                    logger.error(f"[{dataset}] 重試 3 次均失敗，跳過此次請求")
                    return []
        else:
            break


# ──────────────────────────────────────────────
# DB 工具
# ──────────────────────────────────────────────
def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_ddl(conn, *ddls):
    with conn.cursor() as cur:
        for ddl in ddls:
            cur.execute(ddl)
    conn.commit()


def bulk_upsert(conn, upsert_sql: str, rows: list, template: str, page_size=2000):
    if not rows:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, upsert_sql, rows, template=template, page_size=page_size
        )
    conn.commit()


def get_all_stock_ids(conn, stock_id_arg=None):
    if stock_id_arg:
        return [s.strip() for s in stock_id_arg.split(",")]
    from config import STOCK_CONFIGS
    return list(STOCK_CONFIGS.keys())


# ③ DB 最新日期批次預載（一次 SQL 取代 N 次逐筆查詢）
def get_all_latest_dates(conn, table: str) -> dict:
    """
    一次查出指定資料表所有股票的最新日期。
    回傳 dict: { stock_id: "YYYY-MM-DD" }
    原本：每支股票 SELECT MAX(date)（N 次 SQL）
    現在：一條 GROUP BY SQL 搞定
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT stock_id, MAX(date) FROM {table} GROUP BY stock_id")
        return {
            row[0]: row[1].strftime("%Y-%m-%d")
            for row in cur.fetchall()
            if row[1] is not None
        }


def resolve_start_cached(stock_id: str, latest_dates: dict,
                         global_start: str, dataset_key: str, force: bool):
    """
    從預載快取（latest_dates）決定起始日，不再每支逐筆查 DB。
    回傳 None 表示此股票不需抓取（已是最新）。
    """
    earliest = DATASET_START_DATES[dataset_key]
    effective_start = max(global_start, earliest)

    if force:
        return effective_start

    latest = latest_dates.get(str(stock_id))
    if latest is None:
        return effective_start  # DB 無此股票，從頭抓

    next_day = (
        datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if next_day > DEFAULT_END:
        return None  # 已是最新，不需抓取

    return max(next_day, earliest)


# ──────────────────────────────────────────────
# Row mapper（API response → DB tuple）
# ──────────────────────────────────────────────
def map_price_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_int(r.get("Trading_Volume")),
        safe_int(r.get("Trading_money")),
        safe_float(r.get("open")),
        safe_float(r.get("max")),
        safe_float(r.get("min")),
        safe_float(r.get("close")),
        safe_float(r.get("spread")),
        safe_int(r.get("Trading_turnover")),
    )


def map_per_row(r: dict) -> tuple:
    return (
        r["date"], r["stock_id"],
        safe_float(r.get("dividend_yield")),
        safe_float(r.get("PER")),
        safe_float(r.get("PBR")),
    )


# ──────────────────────────────────────────────
# ① 批次模式：按起始日分組，同日股票合併一次請求
# ──────────────────────────────────────────────
def fetch_dataset_batch(
    dataset: str, table: str,
    upsert_sql: str, template: str, row_mapper,
    dataset_key: str, start_date: str, end_date: str,
    delay: float, force: bool, chunk_days: int, batch_threshold: int,
    valid_stock_ids: set, latest_dates: dict,
):
    """
    批次模式核心：
      1. 依 stock_id 計算各自的 actual_start
      2. 按 actual_start 分組
      3. 若同一 start 的股票數 >= batch_threshold → 不帶 data_id，一次抓全市場
         否則 → 逐支請求（小尾巴）
      4. 大型批次按 chunk_days 分段，避免單次回傳筆數過多
    """
    stock_ids = sorted(valid_stock_ids)

    # 計算各股票的 actual_start
    stock_starts: dict[str, str] = {}
    skipped = 0
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest_dates, start_date, dataset_key, force)
        if s is None:
            skipped += 1
        else:
            stock_starts[sid] = s

    logger.info(
        f"[{table}] 需抓取：{len(stock_starts)} 支，已最新略過：{skipped} 支"
    )

    if not stock_starts:
        return

    # 按 actual_start 分組
    groups: dict[str, list] = defaultdict(list)
    for sid, s in stock_starts.items():
        groups[s].append(sid)

    logger.info(
        f"[{table}] 共 {len(groups)} 個不同起始日"
        f"（批次閾值：>= {batch_threshold} 支才合併）"
    )

    total_api_calls = 0
    total_rows = 0
    # 一旦偵測到批次不支援，後續所有組別都改用逐支模式
    batch_disabled = False

    for group_start in sorted(groups.keys()):
        sids = groups[group_start]
        sids_set = set(sids)

        if len(sids) >= batch_threshold and not batch_disabled:
            # ── 批次模式：不帶 data_id，全市場一次抓 ──
            # 按 chunk_days 切段
            seg_start = group_start
            seg_end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            chunk_rows = []
            try:
                while True:
                    seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
                    if seg_start_dt > seg_end_dt:
                        break
                    seg_end = min(
                        (seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"),
                        end_date,
                    )

                    logger.info(
                        f"  [{table}] 批次請求 {seg_start}~{seg_end}"
                        f"（{len(sids)} 支，不帶 data_id）"
                    )
                    data = finmind_get(
                        dataset,
                        {"start_date": seg_start, "end_date": seg_end},
                        delay,
                    )
                    total_api_calls += 1

                    # 只保留有效股票（過濾 ETF、索引等非目標資料）
                    filtered = [r for r in data if r.get("stock_id") in sids_set]
                    chunk_rows.extend(filtered)

                    seg_start = (
                        datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
                    ).strftime("%Y-%m-%d")

            except BatchNotSupportedError as e:
                logger.warning(str(e))
                logger.warning(
                    f"  [{table}] 自動 fallback：改為逐支請求模式"
                    f"（本帳號等級不支援批次查詢，後續所有組別均改用逐支）"
                )
                batch_disabled = True
                chunk_rows = []  # 捨棄批次已抓的不完整資料

            if chunk_rows:
                rows = [row_mapper(r) for r in chunk_rows]
                bulk_upsert(conn_ref[0], upsert_sql, rows, template)
                total_rows += len(rows)
                logger.info(f"  [{table}] 批次寫入 {len(rows)} 筆")

        if len(sids) < batch_threshold or batch_disabled:
            # ── 小尾巴 或 fallback 逐支請求 ──
            if batch_disabled:
                logger.info(
                    f"  [{table}] fallback 逐支：處理 {len(sids)} 支"
                    f"（起始日 {group_start}）"
                )
            for sid in sids:
                data = finmind_get(
                    dataset,
                    {"data_id": sid, "start_date": group_start, "end_date": end_date},
                    delay,
                )
                total_api_calls += 1
                if not data:
                    continue
                rows = [row_mapper(r) for r in data]
                bulk_upsert(conn_ref[0], upsert_sql, rows, template)
                total_rows += len(rows)

    logger.info(
        f"[{table}] 完成  API 請求：{total_api_calls} 次  寫入：{total_rows} 筆"
    )


# conn_ref 供 fetch_dataset_batch 內 bulk_upsert 使用
conn_ref = [None]


# ──────────────────────────────────────────────
# ② 合併迴圈：price + PER 同時抓（逐支模式）
# ──────────────────────────────────────────────
def fetch_both_per_stock(
    start_date: str, end_date: str, delay: float, force: bool,
    tables: list, stock_id: str = None,
):
    """
    逐支股票模式（--per-stock）。
    price 與 PER 在同一個迴圈內依序抓取，避免重複遍歷股票清單。
    """
    logger.info("=== [逐支模式] price + PER 合併迴圈 ===")
    conn = get_db_conn()
    conn_ref[0] = conn

    try:
        ensure_ddl(conn, DDL_STOCK_PRICE, DDL_STOCK_PER)
        
        if stock_id:
            stock_ids = [s.strip() for s in stock_id.split(",")]
        else:
            stock_ids = get_all_stock_ids(conn)
            
        logger.info(f"共 {len(stock_ids)} 支股票待處理")

        # ③ 預載兩張表的最新日期（各一條 SQL）
        latest_price = get_all_latest_dates(conn, "stock_price") if "stock_price" in tables else {}
        latest_per   = get_all_latest_dates(conn, "stock_per")   if "stock_per"   in tables else {}

        skipped_price = skipped_per = 0

        for i, sid in enumerate(stock_ids, 1):
            # ── stock_price ──
            if "stock_price" in tables:
                s = resolve_start_cached(sid, latest_price, start_date, "stock_price", force)
                if s is None:
                    skipped_price += 1
                else:
                    data = finmind_get(
                        "TaiwanStockPrice",
                        {"data_id": sid, "start_date": s, "end_date": end_date},
                        delay,
                    )
                    if data:
                        rows = [map_price_row(r) for r in data]
                        bulk_upsert(
                            conn, UPSERT_STOCK_PRICE, rows,
                            "(%s::date,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)",
                        )

            # ── stock_per ──
            if "stock_per" in tables:
                s = resolve_start_cached(sid, latest_per, start_date, "stock_per", force)
                if s is None:
                    skipped_per += 1
                else:
                    data = finmind_get(
                        "TaiwanStockPER",
                        {"data_id": sid, "start_date": s, "end_date": end_date},
                        delay,
                    )
                    if data:
                        rows = [map_per_row(r) for r in data]
                        bulk_upsert(
                            conn, UPSERT_STOCK_PER, rows,
                            "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric)",
                        )

            if i % 100 == 0:
                logger.info(
                    f"  進度：{i}/{len(stock_ids)}"
                    f"  price 略過：{skipped_price}  per 略過：{skipped_per}"
                )

    finally:
        conn.close()

    logger.info(
        f"=== [逐支模式] 完成"
        f"  price 略過：{skipped_price}  per 略過：{skipped_per} ==="
    )


# ──────────────────────────────────────────────
# 批次模式入口
# ──────────────────────────────────────────────
def fetch_both_batch(
    start_date: str, end_date: str, delay: float, force: bool,
    tables: list, chunk_days: int, batch_threshold: int, stock_id: str = None,
):
    """
    批次模式（預設）。
    依資料集分別執行，每個資料集按起始日分組後批次請求。
    """
    conn = get_db_conn()
    conn_ref[0] = conn

    try:
        ensure_ddl(conn, DDL_STOCK_PRICE, DDL_STOCK_PER)
        
        if stock_id:
            stock_ids = [s.strip() for s in stock_id.split(",")]
        else:
            stock_ids = get_all_stock_ids(conn)
            
        valid_set = set(stock_ids)
        logger.info(f"目標清單共 {len(valid_set)} 支有效股票")

        if "stock_price" in tables:
            logger.info("=== [stock_price] 批次模式 ===")
            latest = get_all_latest_dates(conn, "stock_price")
            fetch_dataset_batch(
                "TaiwanStockPrice", "stock_price",
                UPSERT_STOCK_PRICE,
                "(%s::date,%s,%s,%s,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s::numeric,%s)",
                map_price_row,
                "stock_price", start_date, end_date,
                delay, force, chunk_days, batch_threshold,
                valid_set, latest,
            )

        if "stock_per" in tables:
            logger.info("=== [stock_per] 批次模式 ===")
            latest = get_all_latest_dates(conn, "stock_per")
            fetch_dataset_batch(
                "TaiwanStockPER", "stock_per",
                UPSERT_STOCK_PER,
                "(%s::date,%s,%s::numeric,%s::numeric,%s::numeric)",
                map_per_row,
                "stock_per", start_date, end_date,
                delay, force, chunk_days, batch_threshold,
                valid_set, latest,
            )
    finally:
        conn.close()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="FinMind 技術面資料抓取工具 v2.0（API 用量優化版）"
    )
    parser.add_argument(
        "--tables", nargs="+",
        choices=["stock_price", "stock_per", "all"],
        default=["all"],
        help="要抓取的資料表（預設 all）",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help="開始日期 YYYY-MM-DD（預設 1994-10-01）",
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help="結束日期 YYYY-MM-DD（預設今天）",
    )
    parser.add_argument(
        "--delay", type=float, default=1.2,
        help="每次 API 請求後等待秒數（預設 1.2）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重抓：忽略 DB 已有資料，從 --start 重新覆蓋",
    )
    # ① 批次模式控制
    parser.add_argument(
        "--per-stock", action="store_true",
        help="停用批次模式，退回逐支請求（相容舊行為）",
    )
    parser.add_argument(
        "--batch-threshold", type=int, default=DEFAULT_BATCH_THRESHOLD,
        help=f"批次閾值：同一起始日超過此支數才合併請求（預設 {DEFAULT_BATCH_THRESHOLD}）",
    )
    parser.add_argument(
        "--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS,
        help=f"批次請求每段天數（預設 {DEFAULT_CHUNK_DAYS}，避免單次回傳筆數過多）",
    )
    parser.add_argument("--stock-id", default=None, help="指定股票代號（多個請用逗號隔開）")
    return parser.parse_args()


def main():
    args = parse_args()
    tables = ["stock_price", "stock_per"] if "all" in args.tables else args.tables

    mode_str = "逐支模式（--per-stock）" if args.per_stock else "批次模式（API 用量最少）"
    logger.info(f"抓取資料表：{tables}")
    logger.info(f"日期區間：{args.start} ~ {args.end}")
    logger.info(f"請求間隔：{args.delay} 秒")
    logger.info(f"執行模式：{'強制重抓' if args.force else '增量模式'}  |  {mode_str}")
    if not args.per_stock:
        logger.info(
            f"批次閾值：>= {args.batch_threshold} 支  "
            f"每段天數：{args.chunk_days} 天"
        )

    try:
        if args.per_stock:
            # ② 合併迴圈逐支模式
            fetch_both_per_stock(args.start, args.end, args.delay, args.force, tables, args.stock_id)
        else:
            # ① 批次模式
            fetch_both_batch(
                args.start, args.end, args.delay, args.force,
                tables, args.chunk_days, args.batch_threshold, args.stock_id,
            )
    except psycopg2.OperationalError as e:
        logger.error(f"PostgreSQL 連線失敗：{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"未預期錯誤：{e}")
        raise


if __name__ == "__main__":
    main()
