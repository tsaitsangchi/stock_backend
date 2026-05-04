from __future__ import annotations
import sys
import json
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_fred_data.py — FRED API 全球宏觀資料（逐 series commit 完整性版）
=========================================================================
v2.2 改進：
  · safe_commit_rows()：每個 series 寫入後立即 commit，失敗 rollback。
  · 主迴圈 try/except 後補上 conn.rollback()，避免 transaction 卡住。
  · 失敗清單寫入 outputs/fred_failed_{date}.json。

執行：
    python fetch_fred_data.py
    python fetch_fred_data.py --series VIXCLS T10Y2Y
    python fetch_fred_data.py --force --start 2000-01-01
"""

import argparse
import logging
import os
import random
import time
from datetime import date, datetime, timedelta

import requests

from core.db_utils import get_db_conn, ensure_ddl, bulk_upsert, safe_float

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

DEFAULT_FRED_SERIES = [
    # 殖利率曲線
    "T10Y2Y", "T10Y3M", "T10YIE",
    # 風險指標
    "VIXCLS", "BAMLH0A0HYM2",
    # 美元 / 流動性
    "DTWEXBGS", "M2SL",
    "DGS10", "DGS2", "DGS3MO",
    # 美國景氣
    "NAPMCI", "UMCSENT", "INDPRO", "UNRATE", "CPIAUCSL",
]

DDL_FRED = """
CREATE TABLE IF NOT EXISTS fred_series (
    series_id VARCHAR(50),
    date      DATE,
    value     NUMERIC(20,6),
    PRIMARY KEY (series_id, date)
);
CREATE INDEX IF NOT EXISTS idx_fred_date ON fred_series (date);
"""

UPSERT_FRED = """
INSERT INTO fred_series (series_id, date, value)
VALUES %s
ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value;
"""


# ─────────────────────────────────────────────
# 逐 series commit 工具函式
# ─────────────────────────────────────────────
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


def dump_failures(failures: list) -> None:
    if not failures:
        return
    out = OUTPUT_DIR / f"fred_failed_{date.today().strftime('%Y%m%d')}.json"
    try:
        out.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"  失敗清單已寫入：{out}（{len(failures)} 個 series）")
    except Exception as e:
        logger.warning(f"  寫入失敗清單時發生錯誤：{e}")


# ─────────────────────────────────────────────
# 抓取邏輯
# ─────────────────────────────────────────────
def fred_get(series_id: str, api_key: str,
             start: str, end: str,
             max_retries: int = 3) -> list[dict]:
    params = {
        "series_id":         series_id,
        "api_key":           api_key,
        "file_type":         "json",
        "observation_start": start,
        "observation_end":   end,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(FRED_API_URL, params=params, timeout=(10, 60))
            if resp.status_code != 200:
                logger.warning(
                    f"[{series_id}] HTTP {resp.status_code}：{resp.text[:200]}"
                )
                if resp.status_code == 429:
                    time.sleep(60)
                    continue
                return []
            data = resp.json()
            return data.get("observations", [])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = 2 ** (attempt - 1) + random.uniform(0, 1.0)
            logger.warning(f"[{series_id}] 第 {attempt}/{max_retries} 次失敗：{e}，{wait:.2f}s 後重試")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** (attempt - 1) + random.uniform(0, 1.0)
            logger.warning(f"[{series_id}] 第 {attempt}/{max_retries} 次異常：{e}")
            time.sleep(wait)
    logger.error(f"[{series_id}] 已重試 {max_retries} 次，放棄")
    return []


def latest_date_for_series(conn, series_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(date) FROM fred_series WHERE series_id = %s",
            (series_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
    return None


def fetch_series(conn, series_id: str, api_key: str,
                 start: str, end: str, force: bool, delay: float) -> int:
    s = start
    if not force:
        last = latest_date_for_series(conn, series_id)
        if last:
            next_d = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if next_d > end:
                logger.info(f"[{series_id}] 已最新（{last}），跳過")
                return 0
            s = next_d
    logger.info(f"[{series_id}] 抓取 {s} ~ {end}")

    obs = fred_get(series_id, api_key, s, end)
    if not obs:
        logger.info(f"[{series_id}] 無新資料")
        return 0

    rows = []
    for o in obs:
        try:
            v = safe_float(o.get("value")) if o.get("value") != "." else None
            if v is None:
                continue
            rows.append((series_id, o.get("date"), v))
        except Exception as e:
            logger.warning(f"  [{series_id}] mapper 異常筆，跳過：{e}")

    if rows:
        n = safe_commit_rows(conn, UPSERT_FRED, rows, "(%s, %s, %s)", label=series_id)
        logger.info(f"[{series_id}] 寫入 {n} 筆")
        time.sleep(delay)
        return n
    else:
        logger.info(f"[{series_id}] 全部為 NA，未寫入")
        time.sleep(delay)
        return 0


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--series", nargs="+", default=DEFAULT_FRED_SERIES,
                   help="要抓取的 FRED series_id 清單")
    p.add_argument("--start", type=str, default="1990-01-01")
    p.add_argument("--end",   type=str, default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=0.5,
                   help="series 之間的休眠秒數（FRED 限制 120 req/min）")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error(
            "未設定 FRED_API_KEY。請至 https://fredaccount.stlouisfed.org/ 申請後，"
            "填入 scripts/.env 的 FRED_API_KEY=...。"
        )
        sys.exit(1)

    logger.info(f"目標 series：{len(args.series)} 個")
    logger.info(f"區間：{args.start} ~ {args.end}")

    conn = get_db_conn()
    failures = []
    try:
        ensure_ddl(conn, DDL_FRED)
        conn.commit()
        for sid in args.series:
            try:
                fetch_series(conn, sid, api_key, args.start, args.end, args.force, args.delay)
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                failures.append({"series_id": sid, "error": str(e)})
                logger.error(f"[{sid}] 失敗：{e}")
                continue
    finally:
        conn.close()

    dump_failures(failures)
    logger.info("全部完成")

if __name__ == "__main__":
    main()