"""
audit_full_db_vs_api_reconcile.py v0.1 (全股 × 全史 × 全表 DB↔API 完整對帳器 · §14.7-CE family · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-30
**主權狀態**: ACTIVE (§14.7-CE Family Complete Reconciliation + Live FinMind/FRED API 全史逐筆 byte-level 比對 + §一.10 三類來源 traceability + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:全股×全史×全表 **DB↔API 對帳**(§14.7-CE):逐筆比對 DB 與 FinMind/FRED API。

**輸入 → 輸出**:DB + API → 差異報告(value/count/missing)

**為什麼需要它**:確認 raw 資料真實完整(§14.7-DD PHASE 5)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Full Reconciliation Scope]** (v0.1, §14.7-CE): 對 scope 內**每一支個股 × 每一張 FinMind raw 表 × 全部歷史**逐筆呼叫 live API,與 DB row 做 byte-level 一致性對帳;補足 `audit_live_api_vs_db.py`(僅最近 5 days × PriceAdj)之全史缺口。
2. **[Schema SSOT Alignment]** (v0.2, §0.0-I / §14.7-DJ): 表 columns / unique-key 經 `core.data_schema.get_dataset_columns/get_dataset_keys` 取得(infra+FRED 用宣告;FinMind 表查 DB information_schema = generic auto-schema 所建),**不自行 hardcode 任何欄位名**(避免 AI 幻像 schema);DATASET_REGISTRY FinMind 部分已 §14.7-DJ 退役。
3. **[Per-Row Key]** (v0.1): 每筆 row 之比對 key = 該表 `unique_constraints`(如 PriceAdj=[date,stock_id] / FinStmt=[date,stock_id,type,origin_name]);API row 與 DB row 以同 key 對齊後逐欄比對。
4. **[Byte-Level Compare]** (v0.1): 數值欄 tol `abs(a-b) <= max(1e-4, 1e-6*max(|a|,|b|))`(對齊 NUMERIC(20,6));日期欄 ISO(YYYY-MM-DD)字串比對;字串欄 strip 後 exact。
5. **[Four Outcome Taxonomy]** (v0.1): 每筆對帳結果歸入 4 類 —— `matched`(key 雙方都有且全欄一致)/ `value_mismatch`(key 雙方都有但某欄不符)/ `missing_in_db`(API 有、DB 無)/ `extra_in_db`(DB 有、API 無)。
6. **[Ingest-Metadata Skip — Transparent]** (v0.1, §14.7-BA): `TaiwanStockMonthRevenue.create_time` 經 §14.7-BA 實證為 DB 寫入時間(非 API publication 值)、`TaiwanStockInfo.date` 為 registry_snapshot_date(API=今日 / DB=上次 sync,恆異)→ 顯式 skip 比對並於報告揭露;`FredData.realtime_start/end` 為 ingest vintage(§14.7-BB)→ FRED 僅比對 `value`。此為**透明揭露之 ingest-side metadata**,非隱藏 discrepancy。
7. **[Mismatch = System-Generated Evidence]** (v0.1, §14.7-CE): API ≠ DB 即 system-computed / 非 API-origin 之證據;`value_mismatch == 0` 即全 overlapping rows = API-origin attestation(per §一.10 #1 三類唯一允許來源 (c) API)。
8. **[Source Traceability]** (v0.1, §一.10): 全 (c) API response + (b) DB query;**0 AI memory**;報告每個數字 trace 回 actual API call / DB row。
9. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): PASS/FAIL 由 `value_mismatch` 動態計算,不硬編。
10. **[Read-Only DB]** (v0.1): 全 connection `set_session(readonly=True)`;純對帳器,**不 sync / 不修改 DB**(resync 屬 `sovereign_sync_engine` / `resync_priceadj_mismatch` 治權)。
11. **[Rate-Limit Aware]** (v0.1, §C): FinMind 402 → backoff sleep 30s 重試;`--max-calls` 預設 6000(Sponsor 單一 quota window)→ 估算超過即 abort,須 explicit 提高才跑全市場(§二.2 / §C 授權邊界)。
12. **[Sovereignty Declaration]** (v0.1, §3.2 橫切 audit / §14.7-CE): 本程式為 **§14.7-CE family 之全史完整對帳載體**(§3.2 橫切)。**治權邊界**:(a) §3.2 橫切,不涉五套禁令 / T1-T3 分層 / §8.5 anti-leakage;(b) read-only;(c) 唯一職責 = live API 全史 call + DB byte-level 逐筆對帳 + mismatch 報告;(d) 不做 universe 挑選 / feature 計算 / model 訓練。
13. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照,非權威來源(權威為 charter §14.7-CE + data_schema SSOT)。
14. **[Idempotency]** (v0.1): pure read-only 可重跑;但每次 run 計入 FinMind quota(注意 hourly cap)。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Scope 決議 (要對帳哪些股)
| 子項 | 對應方法/CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 core(委任快照核心股)| `--scope core` → committed snapshot core_tier='core_universe' | §6.7 Universe SSOT |
| A.2 candidates(委任快照全 tier)| `--scope candidates` → committed snapshot 全 membership | §6.7 |
| A.3 all(全市場有價股)| `--scope all` → DISTINCT stock_id FROM TaiwanStockPriceAdj | full-market |
| A.4 smoke 限制 | `--limit N` 取前 N 股 | §C rate-limit aware |

### Group B. FinMind 全史逐筆對帳 (per stock × 10 raw tables)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 全史 API fetch | `fetch_finmind()` start=`--start`(預設 1990-01-01 全史)| §14.7-CE |
| B.2 DB 全史 load | `load_db_rows()` 依 `get_dataset_columns()`(DB information_schema) | §0.0-I / §14.7-DJ |
| B.3 unique-key 對齊 + 逐欄 byte compare | `reconcile_table()` | §14.7-CE strict |
| B.4 4-taxonomy 計數 | matched / value_mismatch / missing_in_db / extra_in_db | core 定義 #5 |

### Group C. TaiwanStockInfo 市場註冊對帳 (單次 call 全股)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 data_id="" 單次 fetch | `reconcile_info()` | §14.7-CE |
| C.2 key=stock_id 比對 scope 內股 | stock_name/industry_category/type/date | strict |

### Group D. FRED 宏觀全史對帳 (per series_id)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1 動態表發現(fred_series / FredData)| `_discover_fred_tables()` | 防 hallucinated table |
| D.2 全 observations fetch | `fetch_fred_full()` sort=asc 無 limit | §14.7-CE FRED |
| D.3 value by date 比對 | `reconcile_fred()` | §一.10 (c) |

### Group E. 報告 + Verdict
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| E.1 per-table summary + sample mismatch | `write_report()` → reports/ | §二.4 |
| E.2 動態 verdict(value_mismatch==0 → PASS)| §5.6.3 | Zero Hardcoded Verdict |
| E.3 JSON dump(可選 `--json-out`)| traceability | §一.10 #2 |

### 對齊憲章 §二 維運矩陣 (標準場景索引)
| 場景 | 命令 |
| :--- | :--- |
| Smoke test(前 3 核心股 / 全表 / 全史)| `python scripts/audit/audit_full_db_vs_api_reconcile.py --scope core --limit 3` |
| 核心股全史完整對帳(~398 股 × 10 表 ≈ 3,980 calls / ~40 min)| `... --scope core` |
| 全市場全史完整對帳(~2,800 股 × 10 表 ≈ 28,000 calls / ~4.7 hr)| `... --scope all --max-calls 30000` |
| 單表對帳 | `... --scope core --tables TaiwanStockPriceAdj` |
| 僅 FRED 宏觀對帳 | `... --scope core --limit 1 --tables _none_`(個股 0 表 + FRED)|

### 不提供之旗標 (Intentionally Omitted)
- `--auto-resync`:本程式 audit-only;resync 屬 `sovereign_sync_engine` / `resync_priceadj_mismatch` 治權(避免越權修改 DB)。
- `--write-db`:read-only 強制,無寫入路徑。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-30 | Claude | **首版**:§14.7-CE family 全股 × 全史 × 全表 DB↔API 完整對帳器。10 FinMind per-stock 表 + TaiwanStockInfo + FRED(fred_series/FredData)逐筆 byte-level 對帳;4-taxonomy(matched/value_mismatch/missing_in_db/extra_in_db);schema 全 import 自 data_schema SSOT(0 hardcoded 欄位);thread-local read-only conn;FinMind 402 backoff + `--max-calls` 6000 quota guard;markdown 報告 + JSON dump。補足 `audit_live_api_vs_db.py`(僅 5-day × PriceAdj)之全史缺口。 | SUPERSEDED |
| **v0.2** | 2026-06-08 | Claude | **§14.7-DJ Pure-Generic 對齊**:schema 來源由退役之 `DATASET_REGISTRY[table]` 改為 `get_dataset_columns()`/`get_dataset_keys()`(infra+FRED 宣告;FinMind 表查 DB information_schema = generic auto-schema 所建)。新增 `_table_spec()` 包裝(下游 `spec["columns"]`/`["unique_constraints"]` 介面零改);`PERSTOCK_TABLES` 改 `FINMIND_PIPELINE_DATASETS`;import 移除 DATASET_REGISTRY/FINMIND_API_TABLES。**無幻像保證強化**:expected-schema = DB 實體 + API 回應雙權威(非手維 registry,杜絕 registry 漂移);過窄 PK 之列丟失仍由 api_rows(raw)vs matched 落差浮現。對帳邏輯/4-taxonomy/quota guard 全不變。 | **ACTIVE** |
"""
from __future__ import annotations

import sys
import os
import json
import logging
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

_project_root = _base_dir.parent
load_dotenv(_project_root / ".env")

from core.db_utils import get_db_conn
# §14.7-DJ (pure-generic):FinMind 表 schema 改由 DB information_schema 推導(get_dataset_columns/keys),
# 不再 import DATASET_REGISTRY/FINMIND_API_TABLES。expected-schema = DB 實體(generic auto-schema 建)+ API 回應雙權威。
from core.data_schema import (
    FINMIND_PIPELINE_DATASETS,
    LOCAL_DERIVED_COLUMNS,
    get_dataset_columns,
    get_dataset_keys,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN")
FRED_KEY = os.getenv("FRED_API_KEY")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

TOOL_VER = "v0.2"
HISTORY_FLOOR = "1990-01-01"  # API+DB 全史 floor;API 只回傳實際存在區間

# data_id="" 之市場註冊表單次 call,與 FRED 一樣不進 per-stock 迴圈
# §14.7-DJ:FINMIND_PIPELINE_DATASETS 已不含 TaiwanStockInfo(roster 單獨 reconcile_info 處理)
PERSTOCK_TABLES = list(FINMIND_PIPELINE_DATASETS)


def _table_spec(table: str) -> dict:
    """§14.7-DJ:以 DB information_schema 推導之 schema 取代退役之 DATASET_REGISTRY[table]。
    回傳 {"columns": {col:type}, "unique_constraints": [keys]}(沿用既有 spec 介面,下游零改)。"""
    return {"columns": get_dataset_columns(table), "unique_constraints": get_dataset_keys(table)}

# §14.7-BA / §14.7-BB:ingest-side metadata 欄位,非 API publication 值 → 透明 skip
SKIP_COLUMNS = {
    "TaiwanStockMonthRevenue": {"create_time"},  # DB 寫入時間(§14.7-BA 實證 median lag ~5y)
    "TaiwanStockInfo": {"date"},  # registry_snapshot_date:API=今日 snapshot / DB=上次 sync 日,本質 ingest-side,恆異
}

# ── 全域 API call 計數(thread-safe,quota 透明)──
_calls_lock = threading.Lock()
_api_calls = 0


def _count_call(n: int = 1) -> None:
    global _api_calls
    with _calls_lock:
        _api_calls += n


# ── 全域 FinMind 發話速率限制器(token-bucket / slot 序列化)──
# 全市場 ~27.7k calls 若無限流會在數分鐘內衝破 6000/hr → 大量 402 → retries 耗盡 → api_fail 資料遺失。
# 故將全域發話速率壓在安全線下(預設 5500/hr),使 402 幾乎不發生。僅作用於 FinMind(FRED 為 unlimited tier)。
_rate_lock = threading.Lock()
_next_slot = [0.0]       # monotonic 下一可發話時點
_min_interval = [0.0]    # 秒/call;0 = 不限(由 main 依 --rate-per-hour 設定)


def _rate_gate() -> None:
    if _min_interval[0] <= 0:
        return
    with _rate_lock:
        slot = max(time.monotonic(), _next_slot[0])
        _next_slot[0] = slot + _min_interval[0]
    delay = slot - time.monotonic()
    if delay > 0:
        time.sleep(delay)


# ── thread-local read-only DB connection(psycopg2 非 thread-safe → 每 worker 一條)──
_tls = threading.local()


def _conn():
    c = getattr(_tls, "conn", None)
    if c is None:
        c = get_db_conn()
        c.set_session(readonly=True)
        _tls.conn = c
    return c


# ══════════════════════════════════════════════════════════════════════════════
# 值正規化 + 比對 helpers
# ══════════════════════════════════════════════════════════════════════════════
def _col_kinds(columns: dict) -> dict:
    """DDL type → {'num','date','str'} 比對類別。"""
    kinds = {}
    for col, ddl in columns.items():
        t = ddl.upper()
        if "NUMERIC" in t or "INT" in t or "SERIAL" in t or "DECIMAL" in t or "DOUBLE" in t or "REAL" in t:
            kinds[col] = "num"
        elif "DATE" in t or "TIMESTAMP" in t:
            kinds[col] = "date"
        else:
            kinds[col] = "str"
    return kinds


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v in ("", ".", "None", "null", "NaN", "nan"):
            return None
        try:
            return float(v)
        except ValueError:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in ("", ".", "None", "null"))


def _num_eq(av, dv) -> bool:
    a, b = _to_float(av), _to_float(dv)
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1e-4, 1e-6 * max(abs(a), abs(b)))


def _date_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip()
    # 日期欄全零值('0' / '0000-00-00' 等)為 null-sentinel:FinMind 對無申報日之 ETF
    # 於 RecentlyDeclareDate 回傳 '0',DB ingest 正規化為 NULL。任何真實日期(1900-2100)
    # 必含 1-9 數字,故「無 1-9 數字」即 sentinel → 正規化為空(與 NULL 比對相等)。
    if s and not any(ch in "123456789" for ch in s):
        return ""
    return s[:10] if len(s) >= 10 else s


def _str_norm(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _key(row: dict, unique_cols: list, kinds: dict) -> tuple:
    """以 unique_constraints 構造 hashable 對齊 key(date 正規化為 ISO)。"""
    parts = []
    for c in unique_cols:
        if kinds.get(c) == "date":
            parts.append(_date_str(row.get(c)))
        else:
            parts.append(_str_norm(row.get(c)))
    return tuple(parts)


def _compare_row(api_row: dict, db_row: dict, kinds: dict, unique_cols: set, skip: set) -> list:
    """逐欄 byte-level 比對;回傳 [(col, api_val, db_val)] 之 diff list。"""
    diffs = []
    for col, kind in kinds.items():
        if col in unique_cols or col in skip:
            continue
        if col not in api_row:  # API 未回傳該欄 → 無法比對,跳過
            continue
        av = api_row.get(col)
        dv = db_row.get(col)
        if _is_blank(av) and _is_blank(dv):
            continue
        if kind == "num":
            if not _num_eq(av, dv):
                diffs.append((col, av, dv))
        elif kind == "date":
            if _date_str(av) != _date_str(dv):
                diffs.append((col, av, dv))
        else:
            if _str_norm(av) != _str_norm(dv):
                diffs.append((col, av, dv))
    return diffs


# ══════════════════════════════════════════════════════════════════════════════
# API fetch
# ══════════════════════════════════════════════════════════════════════════════
def fetch_finmind(dataset: str, data_id: str, start: str, end: str, retries: int = 3):
    """呼叫 FinMind v4/data;回傳 list[dict] 或 None(失敗)。每次嘗試計入 quota。"""
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    params = {"dataset": dataset, "data_id": data_id, "start_date": start, "end_date": end}
    for attempt in range(retries):
        _rate_gate()
        _count_call()
        try:
            r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                d = r.json()
                msg = (d.get("msg") or "").lower()
                if d.get("msg") == "success":
                    return d.get("data", [])
                if "limit" in msg or "quota" in msg:
                    time.sleep(30)
                    continue
                return d.get("data", [])  # 其他 200 變體仍回 data(可能空)
            if r.status_code == 402:
                time.sleep(30)
                continue
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def fetch_fred_full(series_id: str, retries: int = 3):
    """呼叫 FRED observations(全史,sort asc);回傳 list[dict] 或 None。"""
    params = {"series_id": series_id, "api_key": FRED_KEY, "file_type": "json", "sort_order": "asc"}
    for attempt in range(retries):
        _count_call()
        try:
            r = requests.get(FRED_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("observations", [])
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


# ══════════════════════════════════════════════════════════════════════════════
# DB load
# ══════════════════════════════════════════════════════════════════════════════
def load_db_rows(cur, table: str, stock_id: str, start: str) -> dict:
    spec = _table_spec(table)
    cols = list(spec["columns"].keys())
    kinds = _col_kinds(spec["columns"])
    uniq = spec["unique_constraints"]
    quoted = ", ".join(f'"{c}"' for c in cols)
    sql = f'SELECT {quoted} FROM "{table}" WHERE stock_id = %s'
    params = [stock_id]
    if "date" in spec["columns"] and start:
        sql += ' AND "date" >= %s'
        params.append(start)
    cur.execute(sql, params)
    out = {}
    for row in cur.fetchall():
        rd = dict(zip(cols, row))
        out[_key(rd, uniq, kinds)] = rd
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Reconcile
# ══════════════════════════════════════════════════════════════════════════════
def _blank_result() -> dict:
    return {
        "api_fail": False, "api_rows": 0, "db_rows": 0,
        "matched": 0, "value_mismatch": 0, "missing_in_db": 0, "extra_in_db": 0,
        "samples": [],
    }


def reconcile_table(table: str, api_rows: list, db_rows: dict, sample_cap: int, sid: str) -> dict:
    spec = _table_spec(table)
    kinds = _col_kinds(spec["columns"])
    uniq = spec["unique_constraints"]
    uniq_set = set(uniq)
    skip = set(SKIP_COLUMNS.get(table, set())) | set(LOCAL_DERIVED_COLUMNS.get(table, set()))

    api_map = {}
    for ar in api_rows:
        api_map[_key(ar, uniq, kinds)] = ar

    res = _blank_result()
    res["api_rows"] = len(api_rows)
    res["db_rows"] = len(db_rows)
    samples = res["samples"]

    for k, ar in api_map.items():
        dr = db_rows.get(k)
        if dr is None:
            res["missing_in_db"] += 1
            if len(samples) < sample_cap:
                samples.append({"stock": sid, "key": list(k), "type": "missing_in_db"})
            continue
        diffs = _compare_row(ar, dr, kinds, uniq_set, skip)
        if diffs:
            res["value_mismatch"] += 1
            if len(samples) < sample_cap:
                samples.append({
                    "stock": sid, "key": list(k), "type": "value_mismatch",
                    "diffs": [(c, _str_norm(a) or _date_str(a), _str_norm(d) or _date_str(d)) for c, a, d in diffs[:6]],
                })
        else:
            res["matched"] += 1

    for k in db_rows.keys() - api_map.keys():
        res["extra_in_db"] += 1
        if len(samples) < sample_cap:
            samples.append({"stock": sid, "key": list(k), "type": "extra_in_db"})

    return res


def reconcile_info(cur, scope_ids: list, start: str, end: str, sample_cap: int) -> dict:
    """TaiwanStockInfo:單次 data_id='' 全股 call,比對 scope 內個股之 metadata。"""
    table = "TaiwanStockInfo"
    api_rows = fetch_finmind(table, "", start, end)
    if api_rows is None:
        r = _blank_result()
        r["api_fail"] = True
        return r

    spec = _table_spec(table)
    kinds = _col_kinds(spec["columns"])
    uniq = spec["unique_constraints"]
    cols = list(spec["columns"].keys())
    scope_set = set(scope_ids)

    # DB rows for scope
    quoted = ", ".join(f'"{c}"' for c in cols)
    cur.execute(f'SELECT {quoted} FROM "{table}" WHERE stock_id = ANY(%s)', (scope_ids,))
    db_rows = {}
    for row in cur.fetchall():
        rd = dict(zip(cols, row))
        db_rows[_key(rd, uniq, kinds)] = rd

    # API rows filtered to scope
    api_rows = [ar for ar in api_rows if _str_norm(ar.get("stock_id")) in scope_set]
    return reconcile_table(table, api_rows, db_rows, sample_cap, sid="(info)")


def _discover_fred_tables(cur) -> list:
    """回傳 DB 中實際存在之 FRED 表(防 hallucinated table)。"""
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public' AND table_name IN ('fred_series','FredData')
    """)
    return [r[0] for r in cur.fetchall()]


def reconcile_fred(cur, sample_cap: int) -> dict:
    """FRED 全史 value-by-date 對帳(realtime_* vintage 欄不比 — §14.7-BB)。"""
    out = {
        "series_total": 0, "series_clean": 0, "series_fail": 0,
        "matched": 0, "value_mismatch": 0, "missing_in_db": 0, "extra_in_db": 0,
        "samples": [], "per_table": {},
    }
    samples = out["samples"]
    for tbl in _discover_fred_tables(cur):
        tbl_sql = "fred_series" if tbl == "fred_series" else f'"{tbl}"'
        cur.execute(f"SELECT DISTINCT series_id FROM {tbl_sql} ORDER BY series_id")
        series_ids = [r[0] for r in cur.fetchall()]
        tinfo = {"series": len(series_ids), "matched": 0, "value_mismatch": 0,
                 "missing_in_db": 0, "extra_in_db": 0, "fail": 0}
        for s in series_ids:
            out["series_total"] += 1
            obs = fetch_fred_full(s)
            if obs is None:
                out["series_fail"] += 1
                tinfo["fail"] += 1
                continue
            cur.execute(f"SELECT date, value FROM {tbl_sql} WHERE series_id = %s", (s,))
            db = {_date_str(d): v for d, v in cur.fetchall()}
            api = {}
            for o in obs:
                val = o.get("value")
                if val in (".", None, ""):
                    continue
                api[_date_str(o.get("date"))] = val
            s_mismatch = 0
            for d, av in api.items():
                if d not in db:
                    out["missing_in_db"] += 1
                    tinfo["missing_in_db"] += 1
                    continue
                if _num_eq(av, db[d]):
                    out["matched"] += 1
                    tinfo["matched"] += 1
                else:
                    out["value_mismatch"] += 1
                    tinfo["value_mismatch"] += 1
                    s_mismatch += 1
                    if len(samples) < sample_cap:
                        samples.append({"table": tbl, "series": s, "date": d,
                                        "api": _str_norm(av), "db": _str_norm(db[d])})
            for d in db.keys() - api.keys():
                out["extra_in_db"] += 1
                tinfo["extra_in_db"] += 1
            if s_mismatch == 0 and tinfo["fail"] == 0:
                out["series_clean"] += 1
        out["per_table"][tbl] = tinfo
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Scope
# ══════════════════════════════════════════════════════════════════════════════
def get_scope_stocks(cur, scope: str) -> list:
    if scope == "core":
        cur.execute("""
            SELECT m.stock_id FROM core_universe_membership m
            JOIN core_universe_snapshot s ON s.snapshot_id = m.snapshot_id
            WHERE s.status='committed' AND m.core_tier='core_universe'
              AND s.as_of_date = (SELECT MAX(as_of_date) FROM core_universe_snapshot WHERE status='committed')
            ORDER BY m.stock_id
        """)
    elif scope == "candidates":
        cur.execute("""
            SELECT m.stock_id FROM core_universe_membership m
            JOIN core_universe_snapshot s ON s.snapshot_id = m.snapshot_id
            WHERE s.status='committed'
              AND s.as_of_date = (SELECT MAX(as_of_date) FROM core_universe_snapshot WHERE status='committed')
            ORDER BY m.stock_id
        """)
    else:  # all
        cur.execute('SELECT DISTINCT stock_id FROM "TaiwanStockPriceAdj" ORDER BY stock_id')
    return [r[0] for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════
def write_report(path: str, meta: dict, perstock_agg: dict, info_res, fred_res) -> None:
    L = []
    L.append(f"# 全股 × 全史 × 全表 DB↔API 完整對帳報告")
    L.append("")
    L.append(f"- 產生時間:{meta['generated']}")
    L.append(f"- 工具:`audit_full_db_vs_api_reconcile.py` {TOOL_VER}(§14.7-CE family)")
    L.append(f"- Scope:`{meta['scope']}`(對帳 {meta['n_stocks']} 股)")
    L.append(f"- 歷史區間:{meta['start']} ~ {meta['end']}")
    L.append(f"- FinMind 表:{', '.join(meta['tables']) if meta['tables'] else '(無 per-stock 表)'}")
    L.append(f"- 實際 API calls:{meta['api_calls']} / 耗時:{meta['elapsed']:.1f}s")
    L.append(f"- 數值容差:abs(a-b) ≤ max(1e-4, 1e-6·max(|a|,|b|));日期 ISO 比對;字串 strip exact")
    L.append(f"- Skip(ingest-metadata 透明揭露):{json.dumps({k: sorted(v) for k, v in SKIP_COLUMNS.items()}, ensure_ascii=False)};FRED 僅比對 value(realtime_* vintage 不比)")
    L.append("")

    # verdict
    tot_vm = sum(a["value_mismatch"] for a in perstock_agg.values())
    tot_vm += (info_res["value_mismatch"] if info_res else 0)
    tot_vm += (fred_res["value_mismatch"] if fred_res else 0)
    L.append("## 裁決 (Verdict)")
    L.append("")
    if tot_vm == 0:
        L.append("🎯 **PASS** — 所有 key 同時存在於 API 與 DB 之 rows **100% byte-level 一致**(value_mismatch=0)。")
        L.append("→ 全 DB 值 = FinMind/FRED API origin;**0 system-generated / 0 AI 幻像值**(per §一.10 #1)。")
    else:
        L.append(f"⚠️ **{tot_vm} 筆 value_mismatch** — 偵測到 API ≠ DB 之 row,須 root-cause(可能 publication-date 調整 / 待 resync / 計算欄)。")
    L.append("")
    L.append("> `missing_in_db`(API 有 DB 無)/ `extra_in_db`(DB 有 API 無)為覆蓋率差異:")
    L.append("> 區間邊緣少量屬正常;大量則代表 DB 落後 → 須 `sovereign_sync_engine` resync(本程式不自動修)。")
    L.append("")

    # per-stock tables
    L.append("## FinMind per-stock 表(全史逐筆)")
    L.append("")
    L.append("| 表 | 已對帳股數 | API rows | DB rows | matched | value_mismatch | missing_in_db | extra_in_db | api_fail |")
    L.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for t in meta["tables"]:
        a = perstock_agg.get(t, {})
        L.append(f"| {t} | {a.get('stocks_done', 0)} | {a.get('api_rows', 0)} | {a.get('db_rows', 0)} "
                 f"| {a.get('matched', 0)} | {a.get('value_mismatch', 0)} | {a.get('missing_in_db', 0)} "
                 f"| {a.get('extra_in_db', 0)} | {a.get('api_fail', 0)} |")
    L.append("")

    # info
    if info_res:
        L.append("## TaiwanStockInfo(市場註冊;單次全股 call)")
        L.append("")
        if info_res.get("api_fail"):
            L.append("- API 失敗,未對帳。")
        else:
            L.append(f"- API rows(scope 內):{info_res['api_rows']} / DB rows:{info_res['db_rows']}")
            L.append(f"- matched={info_res['matched']} / value_mismatch={info_res['value_mismatch']} "
                     f"/ missing_in_db={info_res['missing_in_db']} / extra_in_db={info_res['extra_in_db']}")
        L.append("")

    # fred
    if fred_res:
        L.append("## FRED 宏觀(全史 value-by-date)")
        L.append("")
        L.append(f"- series 總數:{fred_res['series_total']} / 100% match:{fred_res['series_clean']} / fail:{fred_res['series_fail']}")
        L.append(f"- matched={fred_res['matched']} / value_mismatch={fred_res['value_mismatch']} "
                 f"/ missing_in_db={fred_res['missing_in_db']} / extra_in_db={fred_res['extra_in_db']}")
        for tbl, ti in fred_res.get("per_table", {}).items():
            L.append(f"  - `{tbl}`:{ti['series']} series / matched={ti['matched']} / value_mismatch={ti['value_mismatch']} "
                     f"/ missing={ti['missing_in_db']} / extra={ti['extra_in_db']} / fail={ti['fail']}")
        L.append("")

    # samples
    L.append("## Mismatch 樣本(每類上限)")
    L.append("")
    any_sample = False
    for t in meta["tables"]:
        smp = perstock_agg.get(t, {}).get("samples", [])
        if smp:
            any_sample = True
            L.append(f"### {t}")
            for s in smp[:20]:
                if s["type"] == "value_mismatch":
                    L.append(f"- `{s['stock']}` key={s['key']} diffs={s['diffs']}")
                else:
                    L.append(f"- `{s['stock']}` key={s['key']} [{s['type']}]")
            L.append("")
    if info_res and info_res.get("samples"):
        any_sample = True
        L.append("### TaiwanStockInfo")
        for s in info_res["samples"][:20]:
            L.append(f"- `{s['stock']}` key={s['key']} [{s['type']}]"
                     + (f" diffs={s['diffs']}" if s["type"] == "value_mismatch" else ""))
        L.append("")
    if fred_res and fred_res.get("samples"):
        any_sample = True
        L.append("### FRED")
        for s in fred_res["samples"][:20]:
            L.append(f"- `{s['table']}` {s['series']} {s['date']}: API={s['api']} / DB={s['db']}")
        L.append("")
    if not any_sample:
        L.append("（無 mismatch 樣本）")
        L.append("")

    Path(path).write_text("\n".join(L), encoding="utf-8")
    logger.info(f"📝 報告已寫入:{path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="全股 × 全史 × 全表 DB↔API 完整對帳(§14.7-CE family)")
    ap.add_argument("--scope", choices=["core", "candidates", "all"], default="core",
                    help="core=委任核心股 / candidates=委任全 tier / all=全市場有價股(預設 core)")
    ap.add_argument("--tables", default="",
                    help="逗號分隔 FinMind per-stock 表子集;預設全 10 表;'_none_'=不對帳個股表(只跑 Info/FRED)")
    ap.add_argument("--start", default=HISTORY_FLOOR, help=f"歷史 floor(預設 {HISTORY_FLOOR} 全史)")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--limit", type=int, default=0, help="只對帳前 N 股(smoke;0=scope 全部)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=6000,
                    help="FinMind call 硬上限(0=不限);估算超過即 abort(§C quota guard)")
    ap.add_argument("--rate-per-hour", type=int, default=5500,
                    help="FinMind 全域發話速率上限/hr(0=不限);壓在 6000/hr 安全線下避免 402 → api_fail")
    ap.add_argument("--no-fred", action="store_true")
    ap.add_argument("--no-info", action="store_true")
    ap.add_argument("--sample-cap", type=int, default=20, help="每表 mismatch 樣本上限")
    ap.add_argument("--out", default="", help="報告路徑(預設 reports/full_db_vs_api_reconciliation_<日期>.md)")
    ap.add_argument("--json-out", default="", help="結果 JSON dump 路徑(可選)")
    args = ap.parse_args()

    if not FINMIND_TOKEN:
        logger.error("❌ FINMIND_TOKEN missing(.env)")
        sys.exit(1)
    if not args.no_fred and not FRED_KEY:
        logger.error("❌ FRED_API_KEY missing(.env);如僅對帳 FinMind 可加 --no-fred")
        sys.exit(1)

    # 全域 FinMind 發話速率限制(token-bucket;0=不限)
    _min_interval[0] = (3600.0 / args.rate_per_hour) if args.rate_per_hour > 0 else 0.0

    # 表子集
    if args.tables.strip() == "_none_":
        tables = []
    elif args.tables.strip():
        req = [t.strip() for t in args.tables.split(",") if t.strip()]
        bad = [t for t in req if t not in PERSTOCK_TABLES]
        if bad:
            logger.error(f"❌ 未知/非 per-stock 表:{bad};可選:{PERSTOCK_TABLES}")
            sys.exit(1)
        tables = req
    else:
        tables = list(PERSTOCK_TABLES)

    conn = get_db_conn()
    conn.set_session(readonly=True)
    cur = conn.cursor()

    stocks = get_scope_stocks(cur, args.scope)
    n_total = len(stocks)
    if args.limit > 0:
        stocks = stocks[:args.limit]
    n_stocks = len(stocks)

    # FRED series 數(估算用)
    n_fred = 0
    if not args.no_fred:
        for tbl in _discover_fred_tables(cur):
            tbl_sql = "fred_series" if tbl == "fred_series" else f'"{tbl}"'
            cur.execute(f"SELECT COUNT(DISTINCT series_id) FROM {tbl_sql}")
            n_fred += cur.fetchone()[0]

    est = n_stocks * len(tables) + (0 if args.no_info else 1) + n_fred

    logger.info("=" * 80)
    logger.info(f"§14.7-CE Family 全股 × 全史 × 全表 DB↔API 完整對帳 {TOOL_VER}")
    logger.info("=" * 80)
    logger.info(f"Scope={args.scope}:{n_total} 股(對帳 {n_stocks})")
    logger.info(f"per-stock 表={len(tables)}:{tables}")
    logger.info(f"Info={'off' if args.no_info else 'on'} / FRED={'off' if args.no_fred else f'on({n_fred} series)'}")
    logger.info(f"歷史={args.start}~{args.end} / workers={args.workers}")
    logger.info(f"估算 API calls ≈ {est}(FinMind {n_stocks}×{len(tables)} + Info + FRED {n_fred})")
    _eff_rate = args.rate_per_hour if args.rate_per_hour > 0 else 6000
    logger.info(f"發話速率上限={'不限' if args.rate_per_hour <= 0 else f'{args.rate_per_hour}/hr'}"
                f"(間隔 {_min_interval[0]:.3f}s/call)→ 估 ~{est / _eff_rate:.2f} hr")

    if args.max_calls > 0 and est > args.max_calls:
        logger.error("=" * 80)
        logger.error(f"⛔ 估算 {est} calls > --max-calls {args.max_calls} → ABORT(未發任何 API call)")
        logger.error(f"   §C/§二.2 quota 授權邊界:全市場請 explicit 加 `--max-calls {((est // 1000) + 1) * 1000}`")
        logger.error(f"   或先 smoke:`--limit 3`(=~{3 * len(tables) + 1 + n_fred} calls)")
        logger.error("=" * 80)
        conn.close()
        sys.exit(2)

    # per-stock 表 accumulator
    perstock_agg = {t: {"stocks_done": 0, "api_rows": 0, "db_rows": 0, "matched": 0,
                        "value_mismatch": 0, "missing_in_db": 0, "extra_in_db": 0,
                        "api_fail": 0, "samples": []} for t in tables}

    t0 = time.time()

    def _worker(sid: str):
        c = _conn()
        cur_w = c.cursor()
        out = {}
        for table in tables:
            api_data = fetch_finmind(table, sid, args.start, args.end)
            if api_data is None:
                out[table] = {"api_fail": True}
                continue
            db_rows = load_db_rows(cur_w, table, sid, args.start)
            out[table] = reconcile_table(table, api_data, db_rows, args.sample_cap, sid)
        cur_w.close()
        return sid, out

    if tables:
        logger.info(f"\n📡 FinMind 對帳啟動(workers={args.workers})...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, sid): sid for sid in stocks}
            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    _, out = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"  worker 例外:{e}")
                    continue
                for table, r in out.items():
                    agg = perstock_agg[table]
                    if r.get("api_fail"):
                        agg["api_fail"] += 1
                        continue
                    agg["stocks_done"] += 1
                    agg["api_rows"] += r["api_rows"]
                    agg["db_rows"] += r["db_rows"]
                    agg["matched"] += r["matched"]
                    agg["value_mismatch"] += r["value_mismatch"]
                    agg["missing_in_db"] += r["missing_in_db"]
                    agg["extra_in_db"] += r["extra_in_db"]
                    for s in r["samples"]:
                        if len(agg["samples"]) < args.sample_cap:
                            agg["samples"].append(s)
                if idx % 50 == 0:
                    el = time.time() - t0
                    vm = sum(a["value_mismatch"] for a in perstock_agg.values())
                    logger.info(f"  progress {idx}/{n_stocks} stocks / calls={_api_calls} "
                                f"/ value_mismatch={vm} / {el:.0f}s")
        logger.info(f"✅ FinMind per-stock 對帳完成({time.time() - t0:.1f}s / calls={_api_calls})")

    # Info
    info_res = None
    if not args.no_info:
        logger.info("\n📡 TaiwanStockInfo 對帳(單次全股 call)...")
        info_res = reconcile_info(cur, stocks, args.start, args.end, args.sample_cap)
        if info_res.get("api_fail"):
            logger.warning("  TaiwanStockInfo API 失敗")
        else:
            logger.info(f"  Info matched={info_res['matched']} / value_mismatch={info_res['value_mismatch']} "
                        f"/ missing={info_res['missing_in_db']} / extra={info_res['extra_in_db']}")

    # FRED
    fred_res = None
    if not args.no_fred:
        logger.info("\n📡 FRED 宏觀對帳(全史 value-by-date)...")
        fred_res = reconcile_fred(cur, args.sample_cap)
        logger.info(f"  FRED series clean={fred_res['series_clean']}/{fred_res['series_total']} "
                    f"/ value_mismatch={fred_res['value_mismatch']} / matched={fred_res['matched']}")

    elapsed = time.time() - t0
    conn.close()

    # verdict log
    tot_vm = sum(a["value_mismatch"] for a in perstock_agg.values())
    tot_vm += (info_res["value_mismatch"] if info_res else 0)
    tot_vm += (fred_res["value_mismatch"] if fred_res else 0)
    tot_missing = sum(a["missing_in_db"] for a in perstock_agg.values())
    tot_extra = sum(a["extra_in_db"] for a in perstock_agg.values())
    logger.info("\n" + "=" * 80)
    logger.info("§14.7-CE FAMILY 完整對帳 VERDICT")
    logger.info("=" * 80)
    logger.info(f"  總 value_mismatch={tot_vm} / per-stock missing_in_db={tot_missing} / extra_in_db={tot_extra}")
    logger.info(f"  實際 API calls={_api_calls} / 耗時={elapsed:.1f}s")
    if tot_vm == 0:
        logger.info("  🎯 PASS:全 overlapping rows byte-match → DB = API origin / 0 system-generated")
    else:
        logger.warning(f"  ⚠️ {tot_vm} value_mismatch → 須 root-cause")

    # reports
    out_path = args.out or str(_project_root / "reports" /
                               f"full_db_vs_api_reconciliation_{datetime.now():%Y%m%d}.md")
    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": args.scope, "n_stocks": n_stocks, "tables": tables,
        "start": args.start, "end": args.end, "api_calls": _api_calls, "elapsed": elapsed,
    }
    write_report(out_path, meta, perstock_agg, info_res, fred_res)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "meta": meta, "perstock": perstock_agg, "info": info_res, "fred": fred_res,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logger.info(f"📝 JSON dump:{args.json_out}")


if __name__ == "__main__":
    main()
