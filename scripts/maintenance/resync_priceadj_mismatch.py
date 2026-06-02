"""
resync_priceadj_mismatch.py v0.2 (priceadj 重同步)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: ACTIVE (one-off / maintenance)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:**一次性/維運** 工具:重新同步 TaiwanStockPriceAdj 之 value_mismatch(除權息回溯)。

**輸入 → 輸出**:既有 DB → 一次性處理結果

**為什麼需要它**:記述性保留(已執行);非常態流程。

## 📜 一、核心定義說明 (Core Definitions)

1. **[One-off Script]**:一次性/維運腳本
2. **[Sovereignty Declaration]**:本程式為**非 charter-core 子系統**工具(charter 可達/引用),不涉 §3.1/§3.2 序列治權主軸、不持五套禁令、不處理 §8.5 anti-leakage。 本檔為**一次性**腳本,非常態 pipeline;保留作 audit trail(亦為 C-隔離候選)。
3. **[Historical Reference Authority]**:本檔標頭版本為記述性快照,非權威來源(權威為憲章 + 程式現行碼)。

## 📊 二、全量功能群矩陣 (Functional Group Matrix)

| 功能 / 指令 | 說明 |
| :--- | :--- |
| python <此檔> | 執行一次性處理 |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.2 | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話補正;標示一次性。原邏輯不變。 | **ACTIVE** |

## 原始說明
P0 re-sync — find all mismatch stocks via live API + upsert DB to match latest FinMind adjustment
"""
import os, sys, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from psycopg2.extras import execute_values

sys.path.insert(0, "/home/hugo/project/stock_backend/scripts")
load_dotenv(Path("/home/hugo/project/stock_backend/.env"))

from core.db_utils import get_db_conn

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN")
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# Wider date window covers all data that's been retroactively adjusted
START = "2024-01-01"
END = date.today().strftime("%Y-%m-%d")


def get_active_universe(cur):
    cur.execute("""
        SELECT m.stock_id FROM core_universe_membership m
        JOIN core_universe_snapshot s ON s.snapshot_id = m.snapshot_id
        WHERE s.status='committed' AND m.core_tier='core_universe'
          AND s.as_of_date = (SELECT MAX(as_of_date) FROM core_universe_snapshot WHERE status='committed')
        ORDER BY m.stock_id
    """)
    return [r[0] for r in cur.fetchall()]


def fetch_finmind_priceadj(sid: str, start: str, end: str):
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    for attempt in range(3):
        try:
            r = requests.get(FINMIND_URL, params={
                "dataset": "TaiwanStockPriceAdj",
                "data_id": sid, "start_date": start, "end_date": end,
            }, headers=headers, timeout=15)
            if r.status_code == 200:
                d = r.json()
                if d.get("msg") == "success":
                    return d.get("data", [])
            elif r.status_code == 402:
                time.sleep(30)
        except requests.RequestException:
            time.sleep(1 * (attempt + 1))
    return None


def identify_mismatch_stocks(cur, universe, sample_start="2026-05-14", sample_end="2026-05-20"):
    """Identify mismatch stocks via 5-day sample close comparison."""
    # Pre-load DB
    cur.execute(
        '''SELECT stock_id, date, "close"::numeric
           FROM "TaiwanStockPriceAdj"
           WHERE stock_id = ANY(%s) AND date BETWEEN %s AND %s''',
        (universe, sample_start, sample_end),
    )
    db_close = {}
    for sid, d, c in cur.fetchall():
        db_close.setdefault(sid, {})[d.strftime("%Y-%m-%d")] = float(c) if c else 0.0

    mismatches = []
    print(f"Identifying mismatch stocks via 5-day close comparison...")

    def _worker(sid):
        data = fetch_finmind_priceadj(sid, sample_start, sample_end)
        if data is None:
            return sid, None
        for row in data:
            api_c = float(row.get("close", 0))
            db_c = db_close.get(sid, {}).get(row["date"], 0)
            if db_c > 0 and abs(db_c - api_c) > 0.001:
                return sid, "mismatch"
        return sid, "match"

    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_worker, sid): sid for sid in universe}
        for idx, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                _, status = fut.result()
            except Exception:
                continue
            if status == "mismatch":
                mismatches.append(sid)
            if idx % 200 == 0:
                print(f"  progress {idx}/{len(universe)} / mismatches found: {len(mismatches)}")
    return mismatches


def resync_stock(sid, conn, start, end):
    """Pull FinMind API + upsert DB."""
    data = fetch_finmind_priceadj(sid, start, end)
    if not data:
        return sid, 0, "api_fail"
    # FinMind PriceAdj columns: date, stock_id, Trading_Volume, Trading_money, open, max, min, close, spread, Trading_turnover
    rows = []
    for r in data:
        try:
            rows.append((
                r["date"], r["stock_id"],
                float(r.get("Trading_Volume", 0)),
                float(r.get("Trading_money", 0)),
                float(r.get("open", 0)),
                float(r.get("max", 0)),
                float(r.get("min", 0)),
                float(r.get("close", 0)),
                float(r.get("spread", 0)),
                float(r.get("Trading_turnover", 0)),
            ))
        except (TypeError, ValueError):
            continue
    if not rows:
        return sid, 0, "no_data"
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO "TaiwanStockPriceAdj"
                (date, stock_id, "Trading_Volume", "Trading_money", "open", "max", "min", "close", "spread", "Trading_turnover")
            VALUES %s
            ON CONFLICT (stock_id, date) DO UPDATE SET
                "Trading_Volume" = EXCLUDED."Trading_Volume",
                "Trading_money" = EXCLUDED."Trading_money",
                "open" = EXCLUDED."open",
                "max" = EXCLUDED."max",
                "min" = EXCLUDED."min",
                "close" = EXCLUDED."close",
                "spread" = EXCLUDED."spread",
                "Trading_turnover" = EXCLUDED."Trading_turnover"
        """, rows, page_size=500)
    conn.commit()
    return sid, len(rows), "ok"


def main():
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        universe = get_active_universe(cur)
        print(f"Active universe: {len(universe)} stocks")

        # Step 1: identify mismatch stocks
        mismatch_stocks = identify_mismatch_stocks(cur, universe)
        print(f"\n📊 Mismatch stocks identified: {len(mismatch_stocks)}")
        print(f"   Stock IDs: {mismatch_stocks}")

        # Step 2: re-sync each(wider date range to cover all retroactive history)
        print(f"\n📡 Re-syncing {len(mismatch_stocks)} stocks(date range {START} ~ {END})...")
        total_rows = 0
        for sid in mismatch_stocks:
            _, n, status = resync_stock(sid, conn, START, END)
            print(f"  {sid}: {status} / {n} rows")
            total_rows += n
            time.sleep(0.15)
        print(f"\n✅ Re-sync done. Total rows upserted: {total_rows}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
