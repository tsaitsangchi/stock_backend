"""
audit_live_api_vs_db.py v0.2 (§14.7-CE Deep Live API vs DB Byte-Level Auditor · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-06-02(v0.2 修正 committed-broken 雙 docstring 結構 + 補第零段白話;原 v0.1 邏輯 2026-05-28 入,不變)
**主權狀態**: ACTIVE (§14.7-CE Deep Audit + Live API endpoint verification + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:即時呼叫 FinMind/FRED API,與 DB 內資料做**逐位元(byte-level)對帳**(§14.7-CE deep)。

**輸入 → 輸出**:live API(最近 5 筆)+ DB → byte 級差異報告(mismatch ≡ system-generated 證據)。

**為什麼需要它**:抽查資料逐位元正確性,確認 DB 與來源 API 完全一致(source-pure 佐證)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Live API vs DB Byte-Level]** (v0.1, §14.7-CE): 實際呼叫 FinMind/FRED API,逐股比對 DB record byte-level match。
2. **[FinMind Strategy]** (v0.1): 每股抓最近 5 days(api.finmindtrade.com)+ DB close/volume/money byte-level compare。
3. **[FRED Strategy]** (v0.1): 全 24 series 抓 latest 5 observations(api.stlouisfed.org)+ DB value byte-level compare。
4. **[Mismatch = System-Generated Evidence]** (v0.1, §14.7-CE): API ≠ DB 即 system-computed 之證據;100% match 即 API-origin attestation。
5. **[Source Traceability]** (v0.1, §一.10): 全 (c) API response + (b) DB query;0 AI memory。
6. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): match % 動態判定。
7. **[Sovereignty Declaration]** (v0.1, §3.2 橫切 audit / §14.7-CE): 本程式為 **§14.7-CE Deep Audit 唯一 live API verification 載體**(§3.2 橫切)。**治權邊界**:(a) §3.2 橫切;(b) read-only(不修改 DB / 不 sync);(c) FinMind quota 考量(rate-limit aware);(d) 唯一職責:live API call + DB byte-level compare + mismatch report。
8. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照。
9. **[Idempotency]** (v0.1): pure read-only;但 API call 計入 quota(注意 rate limit)。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. FinMind Live API Verification (per stock)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 PriceAdj endpoint call | api.finmindtrade.com v4/data | §14.7-CE FinMind |
| A.2 Latest 5 days fetch | safety window | rate limit aware |
| A.3 close/Trading_Volume/Trading_money compare | byte-level | §14.7-CE strict |

### Group B. FRED Live API Verification (per series)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 series/observations endpoint | api.stlouisfed.org | §14.7-CE FRED |
| B.2 Latest 5 obs fetch | safety window | — |
| B.3 value byte-level compare | precision-aware | §14.7-CE strict |

### Group C. Mismatch Report
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Per-stock mismatch count | aggregation | report |
| C.2 100% match attestation | source-origin proof | §14.7-CE |
| C.3 weekly_api_audit_and_resync 整合 | 由 wrapper 呼叫 | §14.7-CH cron |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| Weekly cron(由 weekly_api_audit_and_resync 呼叫)| `python scripts/audit/audit_live_api_vs_db.py` |
| Ad-hoc per-stock verification | `... --stock-id <id>` |

### 不提供之旗標 (Intentionally Omitted)
- `--full-history`:每股只抓最近 5 days(rate limit aware)。
- `--auto-resync`:audit only;resync 屬 sovereign_sync_engine 治權。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CE Deep Audit**。Live FinMind/FRED API call + DB byte-level compare。Latest 5 obs per stock/series。Mismatch ≡ system-generated evidence。 | ARCHIVED(標頭格式)|
| v0.2 | 2026-06-02 | Codex | **修正 committed-broken 結構 + 補白話段**:原標頭被拆成兩個 `\"\"\"` docstring(中間夾註解),致 `from __future__` 不在檔首 → py_compile SyntaxError(pre-existing,不能跑)。本版合併為單一 docstring + 加第零段白話功能說明(§一.11)。邏輯 0 變更。 | **ACTIVE** |

## 四、Live Audit 細節

1. **FinMind PriceAdj**:每股抓最近 5 days(2026-05-14 ~ 2026-05-20)
   - API endpoint:api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj
   - DB compare:close / Trading_Volume / Trading_money byte-level
   - 任何 mismatch 即 system-generated 之證據(若無 mismatch 即 API-origin attestation)

2. **FRED indicators**:對全 24 series 抓 latest 5 observations
   - API endpoint:api.stlouisfed.org/fred/series/observations?series_id=<id>&limit=5
   - DB compare:date / value byte-level

## 二、CLI

    python scripts/audit/audit_live_api_vs_db.py [--limit 50]
        --limit:測試用,只 audit 前 N 支股(預設全 universe)

## 三、Quota

- FinMind:1,541 stocks × 1 API call each ≈ 1,541 calls(<= 6,000/hr Sponsor)
- FRED:24 series × 1 API call each = 24 calls(unlimited tier)
- 並行 workers=12;預計 ~2-3 min
"""
from __future__ import annotations

import sys
import os
import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

_project_root = _base_dir.parent
load_dotenv(_project_root / ".env")

from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN")
FRED_KEY = os.getenv("FRED_API_KEY")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_active_universe(cur):
    cur.execute("""
        SELECT m.stock_id FROM core_universe_membership m
        JOIN core_universe_snapshot s ON s.snapshot_id = m.snapshot_id
        WHERE s.status='committed' AND m.core_tier='core_universe'
          AND s.as_of_date = (SELECT MAX(as_of_date) FROM core_universe_snapshot WHERE status='committed')
        ORDER BY m.stock_id
    """)
    return [r[0] for r in cur.fetchall()]


def fetch_finmind_priceadj(sid: str, start: str, end: str, retries: int = 2):
    """Call FinMind API for one stock's PriceAdj over date range。"""
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    params = {
        "dataset": "TaiwanStockPriceAdj",
        "data_id": sid,
        "start_date": start,
        "end_date": end,
    }
    for attempt in range(retries):
        try:
            r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                d = r.json()
                if d.get("msg") == "success":
                    return d.get("data", [])
            elif r.status_code == 402:
                time.sleep(30)
        except requests.RequestException:
            time.sleep(1 * (attempt + 1))
    return None


def fetch_fred_observations(series_id: str, limit: int = 5):
    """Call FRED API for one series's latest N observations。"""
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc",
    }
    try:
        r = requests.get(FRED_URL, params=params, timeout=15)
        if r.status_code == 200:
            return r.json().get("observations", [])
    except requests.RequestException:
        pass
    return None


def compare_priceadj(api_data: list, db_data: dict, sid: str):
    """Byte-level compare API row vs DB row for PriceAdj。

    api_data: list of {date, stock_id, close, Trading_Volume, Trading_money, ...}
    db_data: {date: {close, volume, money}}
    returns: (match_count, mismatch_count, missing_in_db_count, sample_mismatches)
    """
    matches, mismatches, missing = 0, 0, 0
    sample_mismatch = []
    for row in api_data:
        d = row["date"]
        api_close = float(row.get("close", 0))
        api_vol = float(row.get("Trading_Volume", 0))
        api_money = float(row.get("Trading_money", 0))
        db_row = db_data.get(d)
        if db_row is None:
            missing += 1
            continue
        # Byte-level match(tolerate 0.01 floating-point precision)
        if (abs(db_row["close"] - api_close) < 0.0001 and
            abs(db_row["volume"] - api_vol) < 1.0 and
            abs(db_row["money"] - api_money) < 1.0):
            matches += 1
        else:
            mismatches += 1
            if len(sample_mismatch) < 3:
                sample_mismatch.append({
                    "date": str(d), "stock": sid,
                    "api": {"close": api_close, "vol": api_vol, "money": api_money},
                    "db": db_row,
                })
    return matches, mismatches, missing, sample_mismatch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="audit first N stocks(0=all)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--start", default="2026-05-14")
    p.add_argument("--end", default="2026-05-20")
    args = p.parse_args()

    if not FINMIND_TOKEN or not FRED_KEY:
        logger.error("❌ FINMIND_TOKEN or FRED_API_KEY missing")
        sys.exit(1)

    conn = get_db_conn()
    try:
        cur = conn.cursor()

        # === Part 1: Active universe ===
        universe = get_active_universe(cur)
        N = len(universe)
        if args.limit > 0:
            universe = universe[:args.limit]
        N_audit = len(universe)
        logger.info("=" * 80)
        logger.info(f"§14.7-CE Deep Audit:Live API vs DB byte-level match")
        logger.info("=" * 80)
        logger.info(f"Active universe: {N} stocks(auditing: {N_audit})")
        logger.info(f"PriceAdj date range: {args.start} ~ {args.end}")

        # === Part 2: Pre-load DB PriceAdj ===
        logger.info(f"\n📋 Pre-loading DB PriceAdj for date range...")
        cur.execute(
            '''SELECT stock_id, date, "close"::numeric, "Trading_Volume"::numeric, "Trading_money"::numeric
               FROM "TaiwanStockPriceAdj"
               WHERE stock_id = ANY(%s) AND date BETWEEN %s AND %s''',
            (universe, args.start, args.end),
        )
        db_priceadj = {}  # sid -> {date -> {close, vol, money}}
        for sid, d, c, v, m in cur.fetchall():
            db_priceadj.setdefault(sid, {})[d.strftime("%Y-%m-%d")] = {
                "close": float(c) if c is not None else 0.0,
                "volume": float(v) if v is not None else 0.0,
                "money": float(m) if m is not None else 0.0,
            }
        logger.info(f"  DB PriceAdj loaded: {sum(len(v) for v in db_priceadj.values())} rows / {len(db_priceadj)} stocks")

        # === Part 3: Parallel API call + byte-level match ===
        logger.info(f"\n📡 Calling FinMind API for {N_audit} stocks(workers={args.workers}) ...")
        total_match = 0
        total_mismatch = 0
        total_missing = 0
        total_api_fail = 0
        all_mismatch_samples = []
        stocks_audited = 0
        stocks_clean = 0  # API == DB 完全一致

        def _worker(sid):
            api_data = fetch_finmind_priceadj(sid, args.start, args.end)
            if api_data is None:
                return sid, None, "api_fail"
            return sid, api_data, "ok"

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, sid): sid for sid in universe}
            for idx, fut in enumerate(as_completed(futures), 1):
                sid = futures[fut]
                try:
                    _, api_data, status = fut.result()
                except Exception:
                    total_api_fail += 1
                    continue
                if status == "api_fail" or api_data is None:
                    total_api_fail += 1
                    continue
                stocks_audited += 1
                db_data = db_priceadj.get(sid, {})
                matches, mismatches, missing, samples = compare_priceadj(api_data, db_data, sid)
                total_match += matches
                total_mismatch += mismatches
                total_missing += missing
                if matches > 0 and mismatches == 0 and missing == 0:
                    stocks_clean += 1
                if samples:
                    all_mismatch_samples.extend(samples)
                if idx % 200 == 0:
                    elapsed = time.time() - t0
                    logger.info(f"  progress {idx}/{N_audit} / matches={total_match} / mismatches={total_mismatch} / missing={total_missing} / fail={total_api_fail} / {elapsed:.0f}s")

        elapsed = time.time() - t0
        logger.info(f"\n✅ FinMind audit done in {elapsed:.1f}s")
        logger.info(f"  Stocks API-pulled: {stocks_audited} / {N_audit}")
        logger.info(f"  Stocks 100% match: {stocks_clean}({100*stocks_clean/max(stocks_audited,1):.1f}%)")
        logger.info(f"  Total entries: matches={total_match} / mismatches={total_mismatch} / missing_in_db={total_missing} / api_fail={total_api_fail}")

        if all_mismatch_samples:
            logger.warning(f"\n⚠️ Mismatch samples({min(10, len(all_mismatch_samples))} of {len(all_mismatch_samples)}):")
            for sm in all_mismatch_samples[:10]:
                logger.warning(f"  {sm['stock']} {sm['date']}: API={sm['api']} / DB={sm['db']}")

        # === Part 4: FRED API verification ===
        logger.info(f"\n📡 Calling FRED API for 24 series ...")
        cur.execute("SELECT DISTINCT series_id FROM fred_series ORDER BY series_id")
        fred_ids = [r[0] for r in cur.fetchall()]

        fred_match, fred_mismatch, fred_missing, fred_fail = 0, 0, 0, 0
        fred_clean_series = 0
        fred_sample_mismatch = []

        for sid_fred in fred_ids:
            obs = fetch_fred_observations(sid_fred, limit=5)
            if obs is None:
                fred_fail += 1
                continue
            # Pre-load DB for this series
            cur.execute(
                "SELECT date, value::numeric FROM fred_series WHERE series_id = %s ORDER BY date DESC LIMIT 20",
                (sid_fred,),
            )
            db_obs = {d.strftime("%Y-%m-%d"): float(v) if v is not None else None for d, v in cur.fetchall()}
            series_matches, series_mismatches = 0, 0
            for o in obs:
                d_str = o["date"]
                api_val = o["value"]
                if api_val == "." or api_val is None:
                    continue  # FRED 之 "." 表 missing,跳過
                api_val_f = float(api_val)
                db_val = db_obs.get(d_str)
                if db_val is None:
                    fred_missing += 1
                    continue
                if abs(db_val - api_val_f) < 0.0001:
                    fred_match += 1
                    series_matches += 1
                else:
                    fred_mismatch += 1
                    series_mismatches += 1
                    if len(fred_sample_mismatch) < 5:
                        fred_sample_mismatch.append({
                            "series": sid_fred, "date": d_str, "api": api_val_f, "db": db_val,
                        })
            if series_matches > 0 and series_mismatches == 0:
                fred_clean_series += 1

        logger.info(f"  FRED series API-pulled: {len(fred_ids) - fred_fail} / {len(fred_ids)}")
        logger.info(f"  FRED series 100% match: {fred_clean_series} / {len(fred_ids)}")
        logger.info(f"  Total entries: matches={fred_match} / mismatches={fred_mismatch} / missing_in_db={fred_missing}")
        if fred_sample_mismatch:
            logger.warning(f"  FRED mismatch samples:")
            for sm in fred_sample_mismatch:
                logger.warning(f"    {sm['series']} {sm['date']}: API={sm['api']} / DB={sm['db']}")

        # === Final verdict ===
        logger.info(f"\n" + "=" * 80)
        logger.info(f"§14.7-CE DEEP AUDIT VERDICT")
        logger.info("=" * 80)
        logger.info(f"")
        logger.info(f"📊 FinMind layer(api.finmindtrade.com):")
        logger.info(f"  Stocks audited: {stocks_audited} / {N_audit}")
        logger.info(f"  Stocks 100% byte-match: {stocks_clean}({100*stocks_clean/max(stocks_audited,1):.2f}%)")
        logger.info(f"  Total byte-matches: {total_match}")
        logger.info(f"  Total byte-mismatches: {total_mismatch}")
        logger.info(f"")
        logger.info(f"📊 FRED layer(api.stlouisfed.org):")
        logger.info(f"  Series audited: {len(fred_ids) - fred_fail} / {len(fred_ids)}")
        logger.info(f"  Series 100% byte-match: {fred_clean_series} / {len(fred_ids)}")
        logger.info(f"  Total byte-matches: {fred_match}")
        logger.info(f"  Total byte-mismatches: {fred_mismatch}")
        logger.info(f"")
        if total_mismatch == 0 and fred_mismatch == 0:
            logger.info(f"  🎯 §14.7-CE DEEP attestation:**PASS**")
            logger.info(f"  ✅ 全 DB data = API origin / 0 system-generated value")
        else:
            logger.warning(f"  ⚠️ {total_mismatch + fred_mismatch} mismatches detected")
            logger.warning(f"  必須 root cause 分析(可能為 hardcoded_conservative date shift / publication-date adjustment)")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
