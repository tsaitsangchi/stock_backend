from __future__ import annotations
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ['fetchers', 'pipeline', 'training', 'monitor']: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
"""
fetch_fred_data.py — FRED API 全球宏觀資料（FinMind 範圍外）
================================================================
[新增] 使用 St. Louis Fed FRED API 補齊 FinMind 沒有的關鍵宏觀指標。

  資料源：FRED (Federal Reserve Economic Data)
  https://fred.stlouisfed.org/docs/api/fred/
  Endpoint: https://api.stlouisfed.org/fred/series/observations
  認證：免費申請 API key（https://fredaccount.stlouisfed.org/）
        填入 .env 的 FRED_API_KEY

  抓取的 series（可在 config.FRED_SERIES_IDS 調整）：

  ── 美國公債殖利率曲線（衰退預警）──
    T10Y2Y  美國 10Y - 2Y 利差（最常用衰退指標，倒掛後 6-18 個月衰退）
    T10Y3M  美國 10Y - 3M 利差（聯準會偏好的衰退指標）
    T10YIE  美國 10 年期 BEI（市場通膨預期）

  ── 風險偏好/恐慌指標 ──
    VIXCLS  CBOE 波動率指數（VIX，全球恐慌指數）
    BAMLH0A0HYM2  美國高收益債信用利差（信用緊縮預警）

  ── 美元 / 流動性 ──
    DTWEXBGS  美元指數（廣義 DXY，反映全球美元強弱）
    M2SL      美國 M2 貨幣供給（流動性的真正指標，比 USD/TWD 強）
    DGS10     10 年期國債殖利率（與台美利差計算 carry trade）
    DGS2      2 年期國債殖利率
    DGS3MO    3 個月國庫券殖利率

  ── 美國景氣 ──
    NAPMCI    ISM 製造業 PMI（與 SOX 高度相關，半導體景氣領先指標）
    UMCSENT   密大消費者信心指數
    INDPRO    工業生產指數
    UNRATE    失業率
    CPIAUCSL  美國 CPI（用於計算 real yield）

衍生因子（建議在 feature_engineering.py 補上）：
  - yield_curve_inverted   = (T10Y2Y < 0) flag
  - vix_zscore_252         = VIX 對 252 日均值的 z-score
  - real_yield_10y         = DGS10 - 10年通膨預期（CPIAUCSL YoY）
  - dxy_momentum_60d       = DTWEXBGS.pct_change(60)
  - m2_growth_yoy          = M2SL.pct_change(252)
  - tw_us_carry_trade      = (TW 利率 - DGS2) × USD/TWD 動量
  - pmi_cross_50           = NAPMCI 上穿/下穿 50 的事件特徵

執行：
    python fetch_fred_data.py                   # 全部增量
    python fetch_fred_data.py --series VIXCLS T10Y2Y
    python fetch_fred_data.py --force --start 2000-01-01
"""

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta

import requests

from config import DB_CONFIG  # noqa: F401
from core.db_utils import get_db_conn, ensure_ddl, bulk_upsert, safe_float

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# 預設抓取的 series
DEFAULT_FRED_SERIES = [
    # 殖利率曲線
    "T10Y2Y", "T10Y3M", "T10YIE",
    # 風險指標
    "VIXCLS", "BAMLH0A0HYM2",
    # 美元/流動性
    "DTWEXBGS", "M2SL",
    "DGS10", "DGS2", "DGS3MO",
    # 美國景氣
    "NAPMCI", "UMCSENT", "INDPRO", "UNRATE", "CPIAUCSL",
]

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
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
# 抓取邏輯
# ─────────────────────────────────────────────
def fred_get(series_id: str, api_key: str,
             start: str, end: str,
             max_retries: int = 3) -> list[dict]:
    """
    向 FRED API 請求單一 series 的觀測值。
    失敗時指數退避重試（1s -> 2s -> 4s）。
    """
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
                    time.sleep(60)  # rate limited
                    continue
                return []
            data = resp.json()
            return data.get("observations", [])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = 2 ** (attempt - 1)
            logger.warning(f"[{series_id}] 第 {attempt}/{max_retries} 次失敗：{e}，{wait}s 後重試")
            time.sleep(wait)
        except Exception as e:
            logger.warning(f"[{series_id}] 第 {attempt}/{max_retries} 次異常：{e}")
            time.sleep(2 ** (attempt - 1))
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
                 start: str, end: str, force: bool, delay: float):
    s = start
    if not force:
        last = latest_date_for_series(conn, series_id)
        if last:
            next_d = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if next_d > end:
                logger.info(f"[{series_id}] 已最新（{last}），跳過")
                return
            s = next_d
    logger.info(f"[{series_id}] 抓取 {s} ~ {end}")

    obs = fred_get(series_id, api_key, s, end)
    if not obs:
        logger.info(f"[{series_id}] 無新資料")
        return

    rows = []
    for o in obs:
        v = safe_float(o.get("value")) if o.get("value") != "." else None
        if v is None:
            continue  # FRED 用 "." 表示 NA
        rows.append((series_id, o.get("date"), v))

    if rows:
        bulk_upsert(conn, UPSERT_FRED, rows, "(%s, %s, %s)")
        logger.info(f"[{series_id}] 寫入 {len(rows)} 筆")
    else:
        logger.info(f"[{series_id}] 全部為 NA，未寫入")

    time.sleep(delay)

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
            "❌ 未設定 FRED_API_KEY。請至 https://fredaccount.stlouisfed.org/ 申請後，"
            "填入 scripts/.env 的 FRED_API_KEY=...。"
        )
        sys.exit(1)

    logger.info(f"目標 series：{len(args.series)} 個")
    logger.info(f"區間：{args.start} ~ {args.end}")

    conn = get_db_conn()
    try:
        ensure_ddl(conn, DDL_FRED)
        for sid in args.series:
            try:
                fetch_series(conn, sid, api_key, args.start, args.end, args.force, args.delay)
            except Exception as e:
                logger.error(f"[{sid}] 失敗：{e}")
                continue
    finally:
        conn.close()
    logger.info("✅ 全部完成")

if __name__ == "__main__":
    main()
