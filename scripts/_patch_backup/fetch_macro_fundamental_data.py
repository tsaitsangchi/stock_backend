"""
fetch_macro_fundamental_data.py — 總經與基本面補強資料抓取
============================================================
Backer/Sponsor 方案資料集：

  1. business_indicator   ← TaiwanBusinessIndicator  (景氣對策信號)
     Signal: 燈號轉換作為宏觀 Regime Hard Block
     Columns: date, leading, coincident, lagging, monitoring, monitoring_color

  2. market_value_weight  ← TaiwanStockMarketValueWeight (台股市值比重表)
     Signal: 台積電佔指數比重變化 → RS Line / 資金集中度
     Columns: date, stock_id, stock_name, rank, weight_per, type

  3. industry_chain       ← TaiwanStockIndustryChain (個體公司所屬產業鏈)
     Signal: 供應鏈關聯 → 上下游連動信號
     Columns: stock_id, industry, sub_industry, date

執行：
    python fetch_macro_fundamental_data.py               # 全部
    python fetch_macro_fundamental_data.py --tables business_indicator market_value_weight
    python fetch_macro_fundamental_data.py --force
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import date, datetime, timedelta

import psycopg2
import psycopg2.extras
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ"
    ".muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"
)
DB_CONFIG = {
    "dbname": "stock", "user": "stock",
    "password": "stock", "host": "localhost", "port": "5432",
}
DATASET_START = {
    "business_indicator":  "1982-01-01",
    "market_value_weight": "2024-10-30",  # API 資料起始日
    "industry_chain":      None,          # 無日期參數，全量抓取
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────

DDL_BUSINESS_INDICATOR = """
CREATE TABLE IF NOT EXISTS business_indicator (
    date               DATE  PRIMARY KEY,
    "leading"            NUMERIC(10,2),      -- 領先指標
    leading_notrend    NUMERIC(10,2),      -- 領先指標去趨勢
    coincident         NUMERIC(10,2),      -- 同期指標
    coincident_notrend NUMERIC(10,2),
    lagging            NUMERIC(10,2),      -- 落後指標
    lagging_notrend    NUMERIC(10,2),
    monitoring         NUMERIC(10,2),      -- 景氣對策信號綜合分數
    monitoring_color   VARCHAR(20)         -- 燈號：red/yellow-red/green/yellow-blue/blue
);
CREATE INDEX IF NOT EXISTS idx_bi_date ON business_indicator (date);
"""

DDL_MARKET_VALUE_WEIGHT = """
CREATE TABLE IF NOT EXISTS market_value_weight (
    date         DATE,
    stock_id     VARCHAR(50),
    stock_name   VARCHAR(100),
    rank         INTEGER,        -- 市值排名
    weight_per   NUMERIC(8,4),   -- 佔指數比重 %
    type         VARCHAR(10),    -- twse / tpex
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_mvw_stock ON market_value_weight (stock_id, date);
CREATE INDEX IF NOT EXISTS idx_mvw_date  ON market_value_weight (date);
"""

DDL_INDUSTRY_CHAIN = """
CREATE TABLE IF NOT EXISTS industry_chain (
    stock_id     VARCHAR(50),
    industry     VARCHAR(100),   -- 所屬產業
    sub_industry VARCHAR(100),   -- 所屬子產業
    date         DATE,           -- 資料更新日
    PRIMARY KEY (stock_id)
);
CREATE INDEX IF NOT EXISTS idx_ic_industry ON industry_chain (industry);
"""

# ─────────────────────────────────────────────
# Upsert SQL
# ─────────────────────────────────────────────

UPSERT_BUSINESS_INDICATOR = """
INSERT INTO business_indicator (
    date, "leading", leading_notrend, coincident, coincident_notrend,
    lagging, lagging_notrend, monitoring, monitoring_color
) VALUES %s
ON CONFLICT (date) DO UPDATE SET
    "leading"            = EXCLUDED."leading",
    leading_notrend    = EXCLUDED.leading_notrend,
    coincident         = EXCLUDED.coincident,
    coincident_notrend = EXCLUDED.coincident_notrend,
    lagging            = EXCLUDED.lagging,
    lagging_notrend    = EXCLUDED.lagging_notrend,
    monitoring         = EXCLUDED.monitoring,
    monitoring_color   = EXCLUDED.monitoring_color;
"""

UPSERT_MARKET_VALUE_WEIGHT = """
INSERT INTO market_value_weight (date, stock_id, stock_name, rank, weight_per, type)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    stock_name = EXCLUDED.stock_name,
    rank       = EXCLUDED.rank,
    weight_per = EXCLUDED.weight_per,
    type       = EXCLUDED.type;
"""

UPSERT_INDUSTRY_CHAIN = """
INSERT INTO industry_chain (stock_id, industry, sub_industry, date)
VALUES %s
ON CONFLICT (stock_id) DO UPDATE SET
    industry     = EXCLUDED.industry,
    sub_industry = EXCLUDED.sub_industry,
    date         = EXCLUDED.date;
"""

# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────

def safe_float(v):
    if v is None: return None
    try: return float(str(v).strip())
    except: return None

def safe_int(v):
    if v is None: return None
    try: return int(float(str(v).strip()))
    except: return None

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def ensure_ddl(conn, *ddls):
    with conn.cursor() as cur:
        for d in ddls:
            cur.execute(d)
    conn.commit()

def wait_next_hour():
    now = datetime.now()
    nxt = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    sec = (nxt - now).total_seconds() + 65
    logger.warning(f"402 用量上限，等待至 {nxt.strftime('%H:%M:%S')}（{sec:.0f}s）")
    time.sleep(sec)
    logger.info("恢復請求。")

def finmind_get(dataset: str, params: dict, delay: float = 1.2) -> list:
    """帶指數退避的 FinMind 請求（ConnectTimeout 最多 5 次重試）"""
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    rp = {"dataset": dataset, **params}
    MAX, BASE = 5, 5.0

    while True:
        for attempt in range(1, MAX + 1):
            try:
                resp = requests.get(FINMIND_API_URL, headers=headers,
                                    params=rp, timeout=(15, 120))
                if resp.status_code == 402:
                    wait_next_hour(); break
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("status") == 402:
                    wait_next_hour(); break
                if payload.get("status") != 200:
                    logger.warning(f"[{dataset}] status={payload.get('status')}, 跳過")
                    return []
                time.sleep(delay)
                return payload.get("data", [])

            except requests.exceptions.ConnectTimeout:
                w = BASE * (3 ** (attempt - 1)) + random.uniform(0, 3)
                if attempt < MAX:
                    logger.warning(f"[{dataset}] 連線逾時({attempt}/{MAX})，{w:.0f}s 後重試")
                    time.sleep(w)
                else:
                    logger.error(f"[{dataset}] 連線逾時，已重試 {MAX} 次，跳過"); return []

            except requests.exceptions.ReadTimeout:
                w = BASE * (2 ** (attempt - 1)) + random.uniform(0, 2)
                if attempt < MAX:
                    logger.warning(f"[{dataset}] 讀取逾時({attempt}/{MAX})，{w:.0f}s 後重試")
                    time.sleep(w)
                else:
                    logger.error(f"[{dataset}] 讀取逾時，已重試 {MAX} 次，跳過"); return []

            except requests.HTTPError as e:
                code = e.response.status_code if e.response else 0
                if code == 400: return []
                if code == 402: wait_next_hour(); break
                w = BASE * (2 ** (attempt - 1))
                if attempt < 3:
                    logger.warning(f"[{dataset}] HTTP {code}，{w:.0f}s 後重試"); time.sleep(w)
                else:
                    logger.error(f"[{dataset}] HTTP {code}，已重試 3 次，跳過"); return []

            except Exception as exc:
                w = BASE * (2 ** (attempt - 1)) + random.uniform(0, 2)
                if attempt < MAX:
                    logger.warning(f"[{dataset}] 失敗({attempt}/{MAX})：{exc}，{w:.0f}s 後重試")
                    time.sleep(w)
                else:
                    logger.error(f"[{dataset}] 已重試 {MAX} 次，跳過"); return []
        else:
            break


# ─────────────────────────────────────────────
# ① 台灣每月景氣對策信號表
# ─────────────────────────────────────────────

def fetch_business_indicator(conn, start: str, end: str, delay: float, force: bool):
    logger.info("\n=== [business_indicator] 開始 ===")
    ensure_ddl(conn, DDL_BUSINESS_INDICATOR)

    s = start
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM business_indicator")
            last = cur.fetchone()[0]
        if last:
            s = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    if s > end:
        logger.info("[business_indicator] 已是最新，跳過")
        return

    rows = finmind_get("TaiwanBusinessIndicator",
                       {"start_date": s, "end_date": end}, delay)
    if not rows:
        logger.info("[business_indicator] 無資料（確認 Backer 方案是否啟用）")
        return

    records = [
        (
            r.get("date"),
            safe_float(r.get("leading")),
            safe_float(r.get("leading_notrend")),
            safe_float(r.get("coincident")),
            safe_float(r.get("coincident_notrend")),
            safe_float(r.get("lagging")),
            safe_float(r.get("lagging_notrend")),
            safe_float(r.get("monitoring")),
            r.get("monitoring_color"),
        )
        for r in rows
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, UPSERT_BUSINESS_INDICATOR, records)
    conn.commit()
    logger.info(f"=== [business_indicator] 完成，{len(records)} 筆 ===")

    # 顯示最近幾筆燈號（快速確認資料正確性）
    with conn.cursor() as cur:
        cur.execute("""
            SELECT date, monitoring, monitoring_color
            FROM business_indicator
            ORDER BY date DESC LIMIT 5
        """)
        logger.info("  最近燈號：")
        for row in cur.fetchall():
            logger.info(f"    {row[0]}  分數={row[1]}  燈號={row[2]}")


# ─────────────────────────────────────────────
# ② 台股市值比重表
# ─────────────────────────────────────────────

def fetch_market_value_weight(conn, start: str, end: str, delay: float, force: bool):
    logger.info("\n=== [market_value_weight] 開始 ===")
    ensure_ddl(conn, DDL_MARKET_VALUE_WEIGHT)

    s = start
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM market_value_weight")
            last = cur.fetchone()[0]
        if last:
            s = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    if s > end:
        logger.info("[market_value_weight] 已是最新，跳過")
        return

    # 按月分批抓（避免單次資料過大）
    start_d = datetime.strptime(s, "%Y-%m-%d").date()
    end_d   = datetime.strptime(end, "%Y-%m-%d").date()
    total   = 0

    while start_d <= end_d:
        # 每次抓 30 天
        chunk_end = min(start_d + timedelta(days=30), end_d)
        rows = finmind_get(
            "TaiwanStockMarketValueWeight",
            {"start_date": start_d.strftime("%Y-%m-%d"),
             "end_date":   chunk_end.strftime("%Y-%m-%d")},
            delay,
        )
        if rows:
            records = [
                (
                    r.get("date"),
                    str(r.get("stock_id", "")),
                    r.get("stock_name", ""),
                    safe_int(r.get("rank")),
                    safe_float(r.get("weight_per")),
                    r.get("type", ""),
                )
                for r in rows
            ]
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, UPSERT_MARKET_VALUE_WEIGHT, records)
            conn.commit()
            total += len(records)
            logger.info(f"  [market_value_weight] {start_d}~{chunk_end}: {len(records)} 筆")

        start_d = chunk_end + timedelta(days=1)

    logger.info(f"=== [market_value_weight] 完成，{total} 筆 ===")


# ─────────────────────────────────────────────
# ③ 個體公司所屬產業鏈
# ─────────────────────────────────────────────

def fetch_industry_chain(conn, delay: float, force: bool):
    logger.info("\n=== [industry_chain] 開始 ===")
    ensure_ddl(conn, DDL_INDUSTRY_CHAIN)

    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM industry_chain")
            cnt = cur.fetchone()[0]
        if cnt > 0:
            logger.info(f"[industry_chain] 已有 {cnt} 筆，跳過（使用 --force 強制重抓）")
            return

    rows = finmind_get("TaiwanStockIndustryChain", {}, delay)
    if not rows:
        logger.info("[industry_chain] 無資料（確認 Backer 方案是否啟用）")
        return

    # API 回傳可能會有重複的 stock_id（多個產業分類），為避免 DB Upsert 報錯，進行去重（保留第一筆）
    unique_records = {}
    for r in rows:
        sid = str(r.get("stock_id", ""))
        if sid and sid not in unique_records:
            unique_records[sid] = (
                sid,
                r.get("industry", ""),
                r.get("sub_industry", ""),
                r.get("date"),
            )
    records = list(unique_records.values())

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, UPSERT_INDUSTRY_CHAIN, records)
    conn.commit()
    logger.info(f"=== [industry_chain] 完成，{len(records)} 筆 ===")

    # 顯示產業分布
    with conn.cursor() as cur:
        cur.execute("""
            SELECT industry, COUNT(*) as cnt
            FROM industry_chain
            GROUP BY industry ORDER BY cnt DESC LIMIT 10
        """)
        logger.info("  前10大產業：")
        for row in cur.fetchall():
            logger.info(f"    {row[0]}: {row[1]} 家")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="總經與基本面補強資料抓取")
    p.add_argument("--tables", nargs="+",
                   choices=["business_indicator", "market_value_weight", "industry_chain"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tables = args.tables or ["business_indicator", "market_value_weight", "industry_chain"]

    logger.info("=" * 60)
    logger.info("  總經與基本面補強資料抓取")
    logger.info(f"  資料集：{tables}")
    logger.info("=" * 60)

    try:
        conn = get_conn()
    except Exception as e:
        logger.error(f"DB 連線失敗：{e}"); sys.exit(1)

    for tbl in tables:
        s = args.start or DATASET_START.get(tbl) or "2024-01-01"

        if tbl == "business_indicator":
            fetch_business_indicator(conn, s, args.end, args.delay, args.force)
        elif tbl == "market_value_weight":
            fetch_market_value_weight(conn, s, args.end, args.delay, args.force)
        elif tbl == "industry_chain":
            fetch_industry_chain(conn, args.delay, args.force)

    conn.close()
    logger.info("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()
