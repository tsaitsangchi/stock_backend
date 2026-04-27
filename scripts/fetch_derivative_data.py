"""
fetch_derivative_data.py
從 FinMind API 抓取衍生品資料並寫入 PostgreSQL：
  - futures_daily ← TaiwanFuturesDaily (期貨日成交, Free)
  - option_daily  ← TaiwanOptionDaily  (選擇權日成交, Free)

需求套件：
    pip install requests psycopg2-binary

執行範例：
    # 首次全量抓取（自動從各資料集最早日期開始）
    python fetch_derivative_data.py

    # 只抓期貨
    python fetch_derivative_data.py --tables futures_daily

    # 強制重抓（忽略 DB 已有資料）
    python fetch_derivative_data.py --force

    # 指定區間
    python fetch_derivative_data.py --start 2024-01-01 --end 2026-03-19

修正記錄：
  v2 2026-03-19
    【重大修正】get_instrument_ids() 比對邏輯：
      TaiwanFutOptDailyInfo 回傳的 type 欄位為中文
      （'期貨' / '選擇權'），不是英文（'futures' / 'options'）。
      原版比對永遠為 0 筆，導致程式直接略過所有商品。
    【新增】常用商品代碼降級備案：
      若 TaiwanFutOptDailyInfo 回傳空清單（API 異常或 token 問題），
      自動改用預設的常用商品代碼，確保程式可繼續執行。
    【新增】type 欄位診斷日誌：
      第一次取得商品清單時印出所有 type 值，方便未來排查。
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta, datetime

import psycopg2

from config import FINMIND_TOKEN, DB_CONFIG
import psycopg2.extras
import requests

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

# ======================
# PostgreSQL 連線設定
# ======================
# ======================
# 各資料集最早可用日期
# ======================
DATASET_START_DATES = {
    "futures_daily": "1998-07-01",
    "option_daily":  "2001-12-01",
}

DEFAULT_END   = date.today().strftime("%Y-%m-%d")
DEFAULT_START = "1998-07-01"

# ======================
# ★ 常用商品代碼降級備案
# ======================
# 當 TaiwanFutOptDailyInfo 回傳空清單時使用
# 來源：台灣期交所官網 + FinMind 文件
FALLBACK_FUTURES_IDS = [
    "TX",    # 臺股期貨
    "MTX",   # 小型臺指期貨
    "TXO",   # 臺指選擇權（期貨化）
    "TE",    # 電子期貨
    "TF",    # 金融期貨
    "XIF",   # 非金電期貨
    "G2F",   # 美元兌人民幣期貨
    "GDF",   # 黃金期貨
    "BRF",   # 布蘭特原油期貨
    "SPF",   # S&P 500 期貨
    "UDF",   # 美元指數期貨
    "NDF",   # 英鎊兌美元期貨
    "RHF",   # 臺幣兌美元匯率期貨
    "RTF",   # 小型臺幣兌美元匯率期貨
    "SJF",   # 日圓兌美元期貨
    "EXF",   # 歐元兌美元期貨
    "TGF",   # 臺灣 50 期貨
    "GTF",   # 富台期貨
    "FTX",   # 臺灣永續期貨
    "E4F",   # 臺灣 ESG 永續期貨
]

FALLBACK_OPTIONS_IDS = [
    "TXO",   # 臺指選擇權
    "TEO",   # 電子選擇權
    "TFO",   # 金融選擇權
    "XIO",   # 非金電選擇權
    "GTF",   # 富台選擇權
    "TGO",   # 台灣 50 選擇權
]

# ======================
# TaiwanFutOptDailyInfo type 欄位對應表
# FinMind 實際回傳中文值，兼容英文關鍵字搜尋
# ======================
TYPE_MAP = {
    # FinMind 實際回傳值（2026年驗證）：TaiwanFuturesDaily / TaiwanOptionDaily
    # 同時保留中文與英文關鍵字作為兼容備案
    "futures": ["TaiwanFuturesDaily", "期貨", "futures", "future"],
    "options": ["TaiwanOptionDaily", "選擇權", "options", "option"],
}

# ──────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────
DDL_FUTURES_DAILY = """
CREATE TABLE IF NOT EXISTS futures_daily (
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
CREATE INDEX IF NOT EXISTS idx_futures_daily_futures_id ON futures_daily (futures_id);
"""

DDL_OPTION_DAILY = """
CREATE TABLE IF NOT EXISTS option_daily (
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
CREATE INDEX IF NOT EXISTS idx_option_daily_option_id ON option_daily (option_id);
"""

# ──────────────────────────────────────────────
# Upsert SQL
# ──────────────────────────────────────────────
UPSERT_FUTURES_DAILY = """
INSERT INTO futures_daily (
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
INSERT INTO option_daily (
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
    """
    當遇到 402 (Payment Required) 錯誤時，代表 API 用量達上限。
    通常 FinMind 會在整點重置配額，因此等待至下一整點。
    """
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
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}

    while True:
        for attempt in range(1, 4):
            try:
                resp = requests.get(
                    FINMIND_API_URL, headers=headers, params=req_params, timeout=(15, 120)
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
                    logger.warning(
                        f"[{dataset}] status={status}, msg={payload.get('msg')}，跳過"
                    )
                    return []
                time.sleep(delay)
                return payload.get("data", [])

            except requests.HTTPError as http_err:
                code = http_err.response.status_code if http_err.response is not None else 0
                if code == 400:
                    logger.debug(
                        f"[{dataset}] 400 Bad Request，跳過 "
                        f"(data_id={params.get('data_id')}, start={params.get('start_date')})"
                    )
                    return []
                elif code == 402:
                    wait_until_next_hour()
                    break
                else:
                    logger.warning(f"[{dataset}] HTTP {code}：{http_err}")
                    if attempt < 3:
                        time.sleep(delay * 3)
                    else:
                        logger.error(f"[{dataset}] 重試 3 次均失敗，跳過")
                        return []
            except Exception as exc:
                logger.warning(f"[{dataset}] 第 {attempt} 次請求失敗：{exc}")
                if attempt < 3:
                    time.sleep(delay * 3)
                else:
                    logger.error(f"[{dataset}] 重試 3 次均失敗，跳過")
                    return []
        else:
            break


def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_ddl(conn, ddl: str):
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def bulk_upsert(conn, sql: str, rows: list, template: str, page_size: int = 2000):
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, sql, rows, template=template, page_size=page_size
        )
    conn.commit()
    return len(rows)


def dedup_rows(rows: list, key_indices: tuple) -> list:
    """
    在寫入 DB 前去除 API 回傳的重複 PK 列。
    ON CONFLICT DO UPDATE 在同一批次中出現重複 PK 時會拋出 CardinalityViolation。

    key_indices: 組成 PRIMARY KEY 的欄位索引（tuple）
      - futures_daily PK = (date, futures_id, contract_date) → (0, 1, 2)
      - option_daily  PK = (date, option_id, contract_date, strike_price, call_put) → (0, 1, 2, 3, 4)
    """
    seen = {}
    for row in rows:
        key = tuple(row[i] for i in key_indices)
        seen[key] = row   # 後出現的覆蓋先出現的（保留最新）
    deduped = list(seen.values())
    if len(deduped) < len(rows):
        removed = len(rows) - len(deduped)
        import logging
        logging.getLogger(__name__).debug(f"dedup_rows：去除 {removed} 筆重複 PK 列")
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
    """
    從 TaiwanFutOptDailyInfo 單筆記錄中安全取出商品代碼。

    FinMind 回傳的識別欄位名稱歷史上有過變動：
      - 舊版：code
      - 現版（2024+ 驗證）：futures_id
    依序嘗試所有已知欄位名稱，確保相容。
    """
    for field in ("futures_id", "option_id", "code", "id", "symbol"):
        val = r.get(field)
        if val and str(val).strip():
            return str(val).strip()
    return ""


def get_instrument_ids(delay: float, instrument_type: str) -> list:
    """
    從 TaiwanFutOptDailyInfo 取得商品代碼清單。

    Bug 修正（v3）：
      ① 欄位名稱：原版使用 r["code"]，但 FinMind 現版回傳欄位為 futures_id。
                   r["code"] 不存在時 KeyError 被 list comprehension 靜默吞掉，
                   導致 ids=[] 但程式不報錯，fallback 也沒正確觸發。
                   修正：使用 _extract_instrument_id() 依序嘗試所有已知欄位名稱。
      ② fallback：原版 fallback 在某些程式碼版本下未正確觸發。
                   修正：明確判斷 ids 是否為空，無論原因一律走 fallback。
      ③ 診斷：增加第一筆記錄的完整欄位列印，方便排查未來欄位變動。
    """
    fallback = (
        FALLBACK_FUTURES_IDS if instrument_type == "futures"
        else FALLBACK_OPTIONS_IDS
    )

    data = finmind_get("TaiwanFutOptDailyInfo", {}, delay)

    if not data:
        logger.warning(
            f"TaiwanFutOptDailyInfo 回傳空資料，"
            f"改用預設 {instrument_type} 商品代碼（共 {len(fallback)} 個）"
        )
        return list(fallback)

    # ★ 診斷：印出第一筆的所有欄位名稱 + 實際 type 分布，方便排查欄位變動
    logger.info(
        f"TaiwanFutOptDailyInfo 回傳 {len(data)} 筆，"
        f"欄位名稱：{list(data[0].keys())}"
    )
    unique_types = set(str(r.get("type", "")).strip() for r in data)
    logger.info(f"  type 值分布：{unique_types}")

    # ① 修正：使用 _extract_instrument_id() 取代 r["code"]
    keywords = TYPE_MAP.get(instrument_type, [])
    ids = [
        _extract_instrument_id(r)
        for r in data
        if any(
            str(r.get("type", "")).strip().lower() == kw.lower()
            for kw in keywords
        )
    ]
    # 去除空字串（欄位不存在時 _extract_instrument_id 回傳 ""）
    ids = [i for i in ids if i]

    if ids:
        logger.info(f"取得 {instrument_type} 商品共 {len(ids)} 個：{ids}")
        return ids

    # ② 修正：型別比對後為空（type 值與 TYPE_MAP 不符）→ 走 fallback
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
    try:
        ensure_ddl(conn, DDL_FUTURES_DAILY)

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
            actual_start = resolve_start(
                conn, "futures_daily", "futures_id", fid,
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

            rows = [
                (
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
                )
                for r in data
            ]
            rows = dedup_rows(rows, key_indices=(0, 1, 2))
            n = bulk_upsert(
                conn, UPSERT_FUTURES_DAILY, rows,
                (
                    "(%s::date, %s, %s,"
                    " %s::numeric, %s::numeric, %s::numeric, %s::numeric,"
                    " %s::numeric, %s::numeric,"
                    " %s, %s::numeric, %s, %s)"
                ),
            )
            total_rows += n
            logger.info(f"    → 寫入 {n} 筆（累計 {total_rows}）")

    finally:
        conn.close()
    logger.info(
        f"=== [futures_daily] 完成，共寫入 {total_rows} 筆"
        f"（略過已最新：{skipped} 個）==="
    )


# ──────────────────────────────────────────────
# option_daily（選擇權日成交）
# ──────────────────────────────────────────────
def fetch_option_daily(start_date: str, end_date: str, delay: float, force: bool, target_ids: list = None):
    logger.info("=== [option_daily] 開始抓取 ===")
    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_OPTION_DAILY)

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
            actual_start = resolve_start(
                conn, "option_daily", "option_id", oid,
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

            rows = [
                (
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
                )
                for r in data
            ]
            rows = dedup_rows(rows, key_indices=(0, 1, 2, 3, 4))
            n = bulk_upsert(
                conn, UPSERT_OPTION_DAILY, rows,
                (
                    "(%s::date, %s, %s, %s::numeric, %s,"
                    " %s::numeric, %s::numeric, %s::numeric, %s::numeric,"
                    " %s, %s::numeric, %s, %s)"
                ),
            )
            total_rows += n
            logger.info(f"    → 寫入 {n} 筆（累計 {total_rows}）")

    finally:
        conn.close()
    logger.info(
        f"=== [option_daily] 完成，共寫入 {total_rows} 筆"
        f"（略過已最新：{skipped} 個）==="
    )


# ──────────────────────────────────────────────
# CLI 主程式
# ──────────────────────────────────────────────
TABLE_FUNCS = {
    "futures_daily": fetch_futures_daily,
    "option_daily":  fetch_option_daily,
}


def parse_args():
    parser = argparse.ArgumentParser(description="FinMind 衍生品資料抓取工具 v2")
    parser.add_argument(
        "--tables", nargs="+",
        choices=list(TABLE_FUNCS.keys()) + ["all"],
        default=["all"],
        help="要抓取的資料表（預設 all）",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help="開始日期 YYYY-MM-DD（預設 1998-07-01）",
    )
    parser.add_argument("--end", default=DEFAULT_END,
                        help="結束日期 YYYY-MM-DD（預設今天）")
    parser.add_argument(
        "--delay", type=float, default=1.2,
        help="每次 API 請求後的等待秒數（預設 1.2）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重抓：忽略 DB 已有資料",
    )
    parser.add_argument(
        "--ids", nargs="+",
        help="指定要抓取的商品代碼（例如 TX TFO CDF），若不指定則抓取全部",
    )
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
            # 修改：傳遞 args.ids 給抓取函式
            TABLE_FUNCS[table](args.start, args.end, args.delay, args.force, args.ids)
        except psycopg2.OperationalError as e:
            logger.error(f"PostgreSQL 連線失敗：{e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"[{table}] 未預期錯誤：{e}")
            raise


if __name__ == "__main__":
    main()
