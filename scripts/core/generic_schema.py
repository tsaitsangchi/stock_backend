"""
generic_schema.py v1.2 (Generic Auto-Schema Provisioning · 任意 API dataset → 自動推導型別/建表/upsert 單一引用源;v1.2 §14.7-DJ 字串型別下限 VARCHAR 100→255[用戶 2026-06-08 directive];v1.1 FRED-generic:KEY_CANDIDATES series_id 前置於 date 使逐 series FRED 樣本正確推 (series_id,date) 複合 PK)
================================================================================
**最後更新日期**: 2026-06-08
**主權狀態**: GENERIC AUTO-SCHEMA SSOT (§0.0-I 單一引用源) — 從 API 回應動態推導欄位型別 + 自動建表 + 自動偵測/重用主鍵 + 冪等 upsert;退役 DATASET_REGISTRY schema 白名單後之全 FinMind 原始資料表唯一建表機制;§一.10 source-traceable(全部資料來自 API,零 synthetic/零 impute)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:它是一組「看 API 回來的資料長什麼樣,就自動決定每個欄位該用什麼型別、自動建出一張表、把資料安全寫進去」的共用工具函式——讓系統不需要任何人事先在 registry 把 schema 寫死,任何 API dataset 都能落地。

**它怎麼做(步驟)**:
1. **看資料推型別**(`infer_schema`):純數字→`NUMERIC`(至少 20,6,值很大自動加大);`YYYY-MM-DD` 樣貌→`DATE`(讓發布日比較等日期運算正確);`stock_id`/券商碼/`year`/`Time`→強制 `VARCHAR`(避免被當數字掉前導零或誤判);其餘→`VARCHAR`(至少 255,超長改 `TEXT`)。
2. **偵測主鍵**(`detect_keys`):從 stock_id/date/type/name/year… 等候選欄,貪婪挑出能唯一識別列的最小組合;挑不出就退回「全部非空欄」(寧可主鍵寬一點也不要掉資料)。
3. **建表 + 補欄**(`ensure_table`):`CREATE TABLE IF NOT EXISTS`;**若表已存在,改讀 DB 的實際主鍵並沿用**(主鍵首建後就固定,避免之後用單股小樣本重新推導出更窄的主鍵而把資料覆蓋掉);缺欄則 `ALTER TABLE ADD COLUMN` 補上。
4. **冪等寫入**(`upsert`):`ON CONFLICT (主鍵) DO UPDATE`;數字保留 API 原始字串讓 PostgreSQL 精確轉型(不經 Python float,無精度假象);`none/null/nan` 字串視為 NULL。

**輸入 / 輸出**:輸入=一批 API 回來的 dict 列(`rows`)+ 一個 psycopg2 cursor;輸出=DB 內一張對應表 + 寫入的資料 + 回報推導出的 schema/主鍵/列數。

**它不做的事(治權邊界)**:**不抓 API**(呼叫端負責 fetch/throttle/quota/斷點續傳——本模組只處理「資料 → schema → 表 → upsert」);**不算特徵、不選股**(自動建的表要當特徵仍須過 §14.7-DC source-pure gate);**不管 infra log 表 DDL**(`pipeline_execution_log`/`data_audit_log` 由 `data_schema.py` 持明確 DDL——系統內部寫入非 API dataset);⚠️ **FRED 自 2026-06-08(cont) 已改 generic**(FredData/fred_series 由本模組建,非 data_schema DDL);**不產生任何 synthetic/impute 值**(NULL 即 NULL,§一.10)。

**為什麼需要它**:退役 `DATASET_REGISTRY`「只認 11 張預定義表」白名單後(用戶 2026-06-08 directive「不應該存在 sync 引擎只認 11 註冊表,全部的表都應是通用 ingester 建的」),需要一個**單一引用源**(§0.0-I)承載「自動建表」邏輯,供 `sovereign_sync_engine`(全市場全史 sync)與 `finmind_generic_ingest`(單 dataset 探索)共用,避免兩份拷貝漂移。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. **[Generic Auto-Schema SSOT]** (v1.0, §0.0-I): 全系統「從 API 回應推導 schema + 自動建表」之**唯一引用源**;呼叫端不得自行複製本邏輯。字串 ≥ VARCHAR(255)、數字 ≥ NUMERIC(20,6)(用戶 directive 下限;2026-06-08 字串下限 100→255),值超界自動加大。
2. **[Sovereignty Declaration]** (v1.0, 憲法 §3.2 橫切 library): 本模組為 §3.2 橫切基礎設施 library(schema-provisioning helper),非 §3.1 序列模組。**治權邊界**:(a) §3.2 橫切 utility;(b) 五套禁令(§0.1-A/§0.2-A/§0.3-A/§0.0-E.4/§6.8)不涉;(c) T1-T3 不分層;(d) §8.5 anti-leakage 不處理(raw schema provisioning;發布日治權仍由 `data_schema.build_publication_date_gate` 持有);(e) **不選股不評分**;(f) **不算特徵**;(g) **不抓 API**(呼叫端持有);(h) **不管 infra log 表 DDL**(`pipeline_execution_log`/`data_audit_log` 由 `data_schema.py` 持有;FRED 自 2026-06-08(cont) 改 generic,FredData/fred_series 由本模組建);(i) 唯一職責:rows → schema 推導 → 建表/補欄 → 主鍵偵測/重用 → 冪等 upsert。
3. **[Source Authority / No Synthetic]** (v1.0, §14.7-CC/§一.10): 本模組**不生成任何值**;只忠實落地呼叫端傳入之 API rows;NUMERIC 保留原始字串由 PG 精確 cast(無 Python float 中介),`none/null/nan/nat`/空字串 → NULL。
4. **[Key Stability]** (v1.0): 主鍵首次建表時由 `detect_keys` 推導;**表已存在則一律讀 DB 實際 PK 沿用**,杜絕「單股小樣本推導出更窄主鍵 → ON CONFLICT 覆蓋 → 靜默掉資料」之風險。誠實揭露(§一.8):首建若樣本非代表性,理論上仍可能推導出非最小完整鍵;多列/股之 dataset 由全市場 sync 之首股(通常含完整 type/name 維度)觸發,實務風險低;任何 key-induced dup 由 §14.7-CE DB-vs-API 對帳鏈作為安全網捕捉。
5. **[Idempotency]** (v1.0): `CREATE TABLE IF NOT EXISTS` + `ADD COLUMN`(僅補缺)+ `ON CONFLICT (key) DO UPDATE`;同範圍重跑安全,不重複建表、不丟欄。
6. **[Zero Hardcoded Verdict]** (v1.0, §5.6.3): 本模組為 library 無 verdict;呼叫端依實際寫入列數動態判定成敗,不硬編。
7. **[Historical Reference Authority]** (v1.0): 本模組推導之 schema 為記述性快照,非權威 SSOT;權威為 DB 實體 schema(information_schema)+ API 回應本身(§14.7-CE DB-vs-API 對帳之雙權威)。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)
本模組為 library(無 CLI);依函式職責分群:

| 函式 | 職責 | 治權契約 |
| :--- | :--- | :--- |
| `infer_schema(rows)` | rows → {欄:SQL型別}(數字/日期/字串/識別碼動態判定) | [Generic Auto-Schema SSOT] |
| `detect_keys(rows, schema)` | 從候選欄貪婪推導唯一鍵;退回全非空欄 | [Key Stability] |
| `existing_pk(cur, table)` | 讀 DB 既有表之實際 PRIMARY KEY 欄(無則 []) | [Key Stability] |
| `ensure_table(cur, table, schema, keys)` | 建表/補欄;**回傳 effective_keys(既有 PK 優先)** | [Idempotency]+[Key Stability] |
| `upsert(cur, table, rows, schema, keys)` | `ON CONFLICT (key) DO UPDATE` 冪等寫入;回傳列數 | [Idempotency]+[Source Authority] |
| `provision_and_upsert(cur, table, rows)` | 一站式:infer→ensure→upsert;回傳 (列數, schema, keys) | 上述全部 |

### 不提供之能力 (Intentionally Omitted)
- **不抓 API**:fetch/throttle/quota/斷點續傳由呼叫端(`sovereign_sync_engine`/`finmind_generic_ingest`)持有。
- **不管 infra/FRED DDL**:那些由 `data_schema.py` 持明確 DDL。
- **不開/不關 DB 連線**:接受呼叫端傳入之 cursor(交易邊界由呼叫端控制)。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v1.0 | 2026-06-08 | Claude | 首版:從 `finmind_generic_ingest.py v0.1` 提升 infer_schema/detect_keys/ensure_table/upsert 為 §0.0-I 共用 SSOT;新增 (a) DATE 自動偵測(YYYY-MM-DD)以保 §8.5 發布日 gate;(b) `existing_pk` + ensure_table 重用既有 PK 之 [Key Stability];(c) NUMERIC 保留原始字串精確 cast(無 float 假象);(d) `year` 等期別欄強制字串。對映用戶 directive「全部的表都應是通用 ingester 建的」+ 退役 DATASET_REGISTRY schema 白名單。 | SUPERSEDED |
| v1.1 | 2026-06-08 | Claude | §14.7-DJ FRED-generic 全退役:`KEY_CANDIDATES` 將 `series_id` 前置於 `date`,使 FRED 逐 series 樣本(series_id 恆定)正確推出 `(series_id, date)` 複合 PK(推翻舊「FRED local-derived key generic 不可推導」前提);FRED 兩 writer(`sync_fred`→FredData、`fetch_fred_data`→fred_series)改全 generic。 | SUPERSEDED |
| v1.2 | 2026-06-08 | Claude | **字串型別下限 `MIN_VARCHAR` 100→255**(用戶 directive「所有欄位字串型態最少要大於 varchar(255) 以上」):所有 generic-built 表之字串欄 ≥ `VARCHAR(255)`(值超界仍自動加大;maxlen > `TEXT_THRESHOLD`=4000 改 `TEXT`);docstring [Generic Auto-Schema SSOT] + 白話段同步;`finmind_generic_ingest.py` 因 `import core.generic_schema` 自動繼承新下限(無獨立常數)。**附帶**:修正 v1.1 漏改之 header stale「不管 FRED / FredData 由 data_schema DDL」(FRED 已 generic,FredData/fred_series 由本模組建)。 | **ACTIVE** |
"""

from __future__ import annotations

import re

MIN_VARCHAR = 255          # 用戶 directive(2026-06-08 升下限 100→255):字串至少 VARCHAR(255)
NUM_PRECISION_MIN = 20     # 用戶 directive:數字至少 NUMERIC(20,6)
NUM_SCALE_MIN = 6
TEXT_THRESHOLD = 4000      # 觀測字串超過此長度改用 TEXT
# 主鍵候選欄(依優先序;挑出能唯一識別列之最小組合)。皆為結構性鍵提示(非資料值)。
KEY_CANDIDATES = ["stock_id", "securities_trader_id", "series_id", "date", "Time", "time",
                  "year", "type", "name", "industry_category", "origin_name", "item"]
# series_id 排在 date 之前:FRED 逐 series 同步時單一樣本 series_id 恆定 → [series_id] 非唯一 →
# 續加 date → (series_id, date) 唯一,正確推出複合 PK(避免誤判為單 date PK 致跨 series ON CONFLICT 誤刪)
# 強制字串欄(數值樣貌但屬識別碼/期別標籤,須保留原樣不被當數字)
FORCE_STR = {"stock_id", "securities_trader_id", "year"}
# 一律視為 DATE 之欄名(語意明確之純日期欄)
FORCE_DATE = {"date"}

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# 統一 NULL 語意:None / 空字串 / 佔位字串(none/null/nan/nat)。detect_keys(nonnull 判定)、
# _conv(寫入轉換)、infer_schema(型別取樣)必須用同一判定,否則「detect_keys 視為有值 → 進 PK,
# 但 _conv 轉 None → 插入 NOT NULL 違反」(2026-06-08 TaiwanStockInfo 指數列 date='None' 實證 bug)。
_NULL_STRINGS = ("", "none", "null", "nan", "nat")


def _is_null(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in _NULL_STRINGS)


def _is_num(v):
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str) and _NUM_RE.match(v.strip()):
        return True
    return False


def _is_date(v):
    return isinstance(v, str) and bool(_DATE_RE.match(v.strip()))


def _num_digits(v):
    """回傳 (整數位數, 小數位數)。"""
    s = repr(float(v)) if not isinstance(v, str) else v.strip()
    if "e" in s.lower():
        s = format(float(s), "f")
    s = s.lstrip("-")
    intp, _, decp = s.partition(".")
    return len(intp.lstrip("0") or "0"), len(decp.rstrip("0"))


def infer_schema(rows):
    """rows(list[dict]) → {col: sql_type_str}。型別由觀測值動態推導。"""
    cols = []
    for r in rows:
        for c in r.keys():
            if c not in cols:
                cols.append(c)
    schema = {}
    for c in cols:
        vals = [r.get(c) for r in rows if not _is_null(r.get(c))]
        if c in FORCE_DATE:
            schema[c] = "DATE"
            continue
        if c in ("Time", "time") or c in FORCE_STR:
            maxlen = max((len(str(v)) for v in vals), default=0)
            schema[c] = f"VARCHAR({max(MIN_VARCHAR, maxlen + 20)})"
            continue
        if vals and all(_is_date(v) for v in vals):
            schema[c] = "DATE"
        elif vals and all(_is_num(v) for v in vals):
            max_int = max_dec = 0
            for v in vals:
                i, d = _num_digits(v)
                max_int = max(max_int, i)
                max_dec = max(max_dec, d)
            scale = max(NUM_SCALE_MIN, min(max_dec, 12))
            precision = max(NUM_PRECISION_MIN, max_int + scale + 4)
            schema[c] = f"NUMERIC({precision},{scale})"
        else:
            maxlen = max((len(str(v)) for v in vals), default=0)
            schema[c] = "TEXT" if maxlen > TEXT_THRESHOLD else f"VARCHAR({max(MIN_VARCHAR, maxlen + 20)})"
    return schema


def detect_keys(rows, schema):
    """從候選欄貪婪挑出能唯一識別 sample 列之最小組合;不行則退回非空非 TEXT 欄。
    PK 欄不可為 NULL → 只用「sample 中全部非空」之欄(nonnull)。"""
    cols = list(schema.keys())
    nonnull = {c for c in cols if all(not _is_null(r.get(c)) for r in rows)}
    cands = [c for c in KEY_CANDIDATES if c in cols and c in nonnull]
    chosen = []
    for c in cands:
        chosen.append(c)
        seen = set()
        uniq = True
        for r in rows:
            k = tuple(r.get(x) for x in chosen)
            if k in seen:
                uniq = False
                break
            seen.add(k)
        if uniq:
            return chosen
    # fallback:非空 + 非 TEXT 欄(PK 不可空、不對 TEXT 建索引)
    fb = [c for c in cols if c in nonnull and schema[c] != "TEXT"]
    return fb or [c for c in cols if c in nonnull] or cols


def existing_columns(cur, table):
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name=%s", (table,))
    return {r[0] for r in cur.fetchall()}


def existing_pk(cur, table):
    """回傳既有表之 PRIMARY KEY 欄(依 key 順序);表不存在或無 PK → []。"""
    cur.execute(
        """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = to_regclass(%s) AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """,
        (f'"{table}"',),
    )
    return [r[0] for r in cur.fetchall()]


def ensure_table(cur, table, schema, keys):
    """建表(若不存在)或補缺欄(若存在);回傳 effective_keys。
    **表已存在且有 PK → 一律沿用既有 PK**(Key Stability),否則用傳入 keys。"""
    have = existing_columns(cur, table)
    if not have:
        coldefs = ", ".join(f'"{c}" {t}' for c, t in schema.items())
        pk = ", ".join(f'"{c}"' for c in keys)
        cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({coldefs}, '
                    f'CONSTRAINT "{table}_pk" PRIMARY KEY ({pk}))')
        return list(keys)
    # 表已存在:補缺欄
    for c, t in schema.items():
        if c not in have:
            cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" {t}')
    pk = existing_pk(cur, table)
    return pk or list(keys)


def _conv(v):
    """忠實轉換:字串原樣交 PG 精確 cast(DATE/NUMERIC/VARCHAR 皆接受字串);
    none/null/nan/nat/空字串 → NULL(與 _is_null 一致);數字原樣。"""
    if _is_null(v):
        return None
    if isinstance(v, str):
        return v.strip()
    return v


def upsert(cur, table, rows, schema, keys):
    """ON CONFLICT (keys) DO UPDATE 冪等寫入;回傳寫入列數。"""
    from psycopg2.extras import execute_values
    cols = list(schema.keys())
    data = [tuple(_conv(r.get(c)) for c in cols) for r in rows]
    collist = ", ".join(f'"{c}"' for c in cols)
    updates = ", ".join(f'"{c}"=EXCLUDED."{c}"' for c in cols if c not in keys)
    conflict = ", ".join(f'"{c}"' for c in keys)
    on_conflict = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
    execute_values(
        cur,
        f'INSERT INTO "{table}" ({collist}) VALUES %s ON CONFLICT ({conflict}) {on_conflict}',
        data, page_size=1000)
    return len(data)


def provision_and_upsert(cur, table, rows):
    """一站式:推導 schema → 建表/補欄(重用既有 PK)→ 冪等 upsert。
    回傳 (寫入列數, schema, effective_keys)。呼叫端負責 commit + audit log。"""
    schema = infer_schema(rows)
    keys = detect_keys(rows, schema)
    eff_keys = ensure_table(cur, table, schema, keys)
    n = upsert(cur, table, rows, schema, eff_keys)
    return n, schema, eff_keys
