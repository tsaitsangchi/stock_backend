"""
audit_api_schema_compliance.py v0.3 (Quantum Finance 9-Layer Schema + Data Integrity Audit)
================================================================================
**最後更新日期**: 2026-05-21
**主權狀態**: ACTIVE (憲法 v6.0.0 對齊 + 維運矩陣場景齊全（含 --report-out 自訂路徑）；100% 合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Nine-Layer Audit]: 9 層動態檢驗（A DDL↔DB Physical / B API↔Type / C Length-Precision /
   D NULL Ratio / E PK-Unique / F Duplicate Row / G Date Continuity / H Referential /
   I Value Range）。各層獨立統計，任一 FAILED 即整體 FAILED。
2. [Strict Exit Mode]: 任何 Layer FAILED → `sys.exit(1)`，阻斷下游 maintenance / ingestion。
   無 `--strict` flag（預設嚴格）。
3. [Schema SSOT]: DATASET_REGISTRY 為唯一 schema ground truth；本工具僅讀取不修改。
4. [API Sample Contract]: Layer B/C/D 採樣 N 筆（預設 100）做 type/length/range 驗證；
   --skip-api-probe 可只跑 DB-side Layer A/E/F/G/H/I 六層。
5. [Zero Hardcoded Verdict]: 主權判定 PERFECT/FAILED 必須依執行結果動態計算，嚴禁硬編碼。
   對齊憲章 §5.6.3 與 §3.2 Step 接受標準。
6. [Sovereignty Declaration]: 本模組為憲章 §3.2A 橫切基礎設施稽核工具（憲章 L2483）；
   落實 §1.4 [Defensive Architecture] + L2388「SQL 型別寬度不得更窄」+ §6.7 SQL 契約
   referential integrity；不涉及 §0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §0.0-F.3 五套禁令；
   不在 §0.1.1 T1/T2/T3 分層內；不處理 §8.5 anti-leakage（本工具是 audit，非時間序列建模）；
   不調度 universe；DATASET_REGISTRY 為唯一 schema SSOT，本工具不另立 schema 定義。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [標準執行：9 層全跑 + FRED]** | `$ python scripts/maintenance/audit_api_schema_compliance.py --include-fred` | audit_api_schema_compliance v0.3 |
| **2. [單一表]** | `$ python scripts/maintenance/audit_api_schema_compliance.py --table TaiwanStockPrice` | audit_api_schema_compliance v0.3 |
| **3. [離線：只跑 DB-side 6 層]** | `$ python scripts/maintenance/audit_api_schema_compliance.py --skip-api-probe` | audit_api_schema_compliance v0.3 |
| **4. [自訂 layer]** | `$ python scripts/maintenance/audit_api_schema_compliance.py --layers A,E,H` | audit_api_schema_compliance v0.3 |
| **5. [自訂取樣大小]** | `$ python scripts/maintenance/audit_api_schema_compliance.py --sample-size 500` | audit_api_schema_compliance v0.3 |
| **6. [自訂報告路徑]** | `$ python scripts/maintenance/audit_api_schema_compliance.py --include-fred --report-out reports/custom_audit_path.md` | audit_api_schema_compliance v0.3 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.3** | 2026-05-21 | Codex | **維運矩陣補入 --report-out 場景（CLAUDE.md §四 #4 連鎖實證；達真 100% 合規）**：依 CLAUDE.md §四 #4「8 項標頭強制檢驗」第 5 項「全量維運指令總矩陣場景齊全」之檢驗，揭露 v0.2 維運矩陣只列 5 場景，但 CLI 實際支援 6 個 flag（含 `--report-out` 之自訂報告路徑）— 矩陣未對齊治權現況。**補正內容**：(I) 維運矩陣新增第 6 場景「[自訂報告路徑]」對應 CLI `--include-fred --report-out reports/custom_audit_path.md`；(II) 主權狀態行升至「(憲法 v6.0.0 對齊 + 維運矩陣場景齊全（含 --report-out 自訂路徑）；100% 合規)」；(III) TOOL_VER v0.2 → v0.3；(IV) 維運矩陣 6 場景之 cosmetic v0.2 → v0.3。**9 層 audit 邏輯、CLI 介面（6 flag 不變）、`--report-out` 處理邏輯（v0.1 既有）、DATASET_REGISTRY 引用、verdict 計算、record_lifecycle + write_data_audit_log 接線、所有公開行為皆無變更**；本補正純為標頭維運矩陣完整化（對齊 CLAUDE.md §四 #4 第 5 項；繼 `data_schema.py v2.16`（commit `2a4c0f2`）後第 2 例 §四 #4 連鎖實證）。合規度：v0.2 ≈98%（缺 --report-out 場景）→ v0.3 100%。 | **ACTIVE** |
| v0.2 | 2026-05-21 | Codex | **L19 cross-ref 精確化 + 標頭 100% 合規補強（per_program_audit 跟進）**：依逐元件治權合規審計揭露之 1 處 minor 漂移：L19 [Sovereignty Declaration] 條之「憲章 L24XX 第 6 行」為入憲時遺留之未填 placeholder，違反 §0.0-I 單一引用源精神之 cross-ref 精確性要求。**補正內容**：(I) L19 「憲章 L24XX 第 6 行」→「憲章 L2483」（§3.2A 子表 audit_api_schema_compliance 之實際行號）；對齊 `data_schema.py v2.15`「憲章 L2440 / §6.0A L2709」/ `core/__init__.py v1.17`「憲章 L2457 / L2488 / L5589」之具體行號 cross-ref 治權慣例；(II) 主權狀態行升至「(憲法 v6.0.0 對齊 + §3.2A 橫切稽核工具入憲落地 + cross-ref 精確化；100% 合規)」；(III) TOOL_VER v0.1 → v0.2；(IV) 標頭版本 / 維運矩陣 5 場景 / 修訂歷程之 cosmetic v0.1 → v0.2。**9 層 audit 邏輯、CLI 介面（--include-fred / --table / --skip-api-probe / --sample-size / --layers / --report-out）、DATASET_REGISTRY 引用、verdict 計算、record_lifecycle + write_data_audit_log 接線、所有公開行為皆無變更**；本補正純為標頭 cross-ref 精確化。合規度：v0.1 ≈97%（L24XX placeholder）→ v0.2 100%。 | SUPERSEDED |
| v0.1 | 2026-05-20 | Codex | **首版入憲落地（§0.0-G Step 3 完成；對應憲章 §14.7-AJ）**：依使用者要求補齊憲章 L2388「SQL 型別寬度不得更窄」+ §6.7 SQL referential integrity 之 audit 缺口。**9 層動態檢驗**：(A) DDL ↔ DB Physical Consistency（data_type / character_maximum_length / numeric_precision / numeric_scale）；(B) API Sample ↔ DDL Type Compatibility（Decimal cast / date parse）；(C) API Sample Length / Precision Range（max length / max abs vs DDL 範圍）；(D) NULL Ratio Sanity（unique 欄位 NULL > 50% 為 FAILED）；(E) PK / Unique Constraint Uniqueness（COUNT vs COUNT DISTINCT）；(F) Duplicate Row Detection（全欄位重複）；(G) Date Series Continuity（工作日連續性）；(H) Referential Integrity（stock_id ∈ TaiwanStockInfo，依 §6.7）；(I) Value Range Sanity（負值 / 物理常識）。**嚴格模式**：任何 FAILED → exit 1（無 --strict flag）。**取樣大小**：100（預設可調）。**CLI**：--include-fred / --table / --skip-api-probe / --sample-size / --layers / --report-out。**輸出**：reports/api_schema_compliance_audit_<YYYYMMDD_HHMM>.md 含 9 層細粒度結果。**自我合規**：標頭 6 條核心定義（含 [Zero Hardcoded Verdict] + [Sovereignty Declaration]）對齊 per_program_audit §7.5 八項檢查面；verdict 動態計算對齊 §5.6.3；exit code 對齊 §3.2 接受標準；record_lifecycle + write_data_audit_log 接線對齊 §0.4 可觀察性；DATASET_REGISTRY 為唯一 schema SSOT 對齊 §0.0-I。 | SUPERSEDED |
================================================================================
"""
import os
import sys
import time
import re
import argparse
import requests
from pathlib import Path
from datetime import datetime
from decimal import Decimal, InvalidOperation

# ──────────────────────────────────────────────────────────────────────────────
# 治權常數 (Constitution Constants)
# ──────────────────────────────────────────────────────────────────────────────
CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v0.3"
DEFAULT_SAMPLE_SIZE = 100

LAYERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
LAYER_NAMES = {
    "A": "DDL ↔ DB Physical Consistency",
    "B": "API Sample ↔ DDL Type Compatibility",
    "C": "API Sample Length / Precision Range",
    "D": "NULL Ratio Sanity",
    "E": "PK / Unique Constraint Uniqueness",
    "F": "Duplicate Row Detection",
    "G": "Date Series Continuity",
    "H": "Referential Integrity",
    "I": "Value Range Sanity",
}

# ──────────────────────────────────────────────────────────────────────────────
# 系統級架構引導
# ──────────────────────────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS_FILE.parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.data_schema import (
        DATASET_REGISTRY,
        FINMIND_API_TABLES,
        FRED_CONTRACT_SERIES,
        LOCAL_DERIVED_COLUMNS,
        INFRA_TABLES,
    )
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
    from core.finmind_client import FinMindClient
    from core.path_setup import get_report_dir
except ImportError as exc:
    print("❌ 核心組件導入失敗: {}".format(exc))
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# DDL 字串解析
# ──────────────────────────────────────────────────────────────────────────────
DDL_TYPE_PATTERN = re.compile(r'^([A-Z]+)(?:\(([\d,\s]+)\))?', re.IGNORECASE)


def parse_ddl(ddl_str):
    """Parse DDL string like 'VARCHAR(255)' / 'NUMERIC(20, 6)' / 'DATE' / 'INTEGER' / 'TIMESTAMP DEFAULT ...' / 'SERIAL PRIMARY KEY'

    Returns dict: {base_type, length, precision, scale, is_nullable, ddl_raw}
    """
    raw = ddl_str
    s = ddl_str.strip().upper()
    if "SERIAL" in s:
        return {"base_type": "INTEGER", "length": None, "precision": None, "scale": None,
                "is_nullable": False, "ddl_raw": raw}
    if "DEFAULT" in s:
        s = s.split("DEFAULT")[0].strip()
    if "PRIMARY KEY" in s:
        s = s.replace("PRIMARY KEY", "").strip()

    if s in ("TIMESTAMP", "DATE", "BIGINT", "INTEGER", "TEXT"):
        return {"base_type": s, "length": None, "precision": None, "scale": None,
                "is_nullable": True, "ddl_raw": raw}

    m = DDL_TYPE_PATTERN.match(s)
    if not m:
        return {"base_type": "UNKNOWN", "length": None, "precision": None, "scale": None,
                "is_nullable": True, "ddl_raw": raw}
    base = m.group(1)
    args = m.group(2)

    if base == "VARCHAR":
        length = int(args) if args else None
        return {"base_type": "VARCHAR", "length": length, "precision": None, "scale": None,
                "is_nullable": True, "ddl_raw": raw}
    if base == "NUMERIC":
        if args:
            parts = [p.strip() for p in args.split(",")]
            prec = int(parts[0])
            scale = int(parts[1]) if len(parts) > 1 else 0
            return {"base_type": "NUMERIC", "length": None, "precision": prec, "scale": scale,
                    "is_nullable": True, "ddl_raw": raw}
        return {"base_type": "NUMERIC", "length": None, "precision": None, "scale": None,
                "is_nullable": True, "ddl_raw": raw}

    return {"base_type": base, "length": None, "precision": None, "scale": None,
            "is_nullable": True, "ddl_raw": raw}


# 對齊 PostgreSQL information_schema.columns 之 data_type 字串
DDL_TO_PG_TYPE = {
    "VARCHAR": "character varying",
    "NUMERIC": "numeric",
    "DATE": "date",
    "TIMESTAMP": "timestamp without time zone",
    "BIGINT": "bigint",
    "INTEGER": "integer",
    "TEXT": "text",
}


# ──────────────────────────────────────────────────────────────────────────────
# 主稽核類
# ──────────────────────────────────────────────────────────────────────────────
class ApiSchemaComplianceAuditor:
    """9 層 schema + 資料完整性 audit。對齊憲章 §3.2A 治權邊界。"""

    def __init__(self, sample_size=DEFAULT_SAMPLE_SIZE, include_fred=False, table=None,
                 skip_api_probe=False, layers=None, report_out=None):
        self.sample_size = sample_size
        self.include_fred = include_fred
        self.target_table = table
        self.skip_api_probe = skip_api_probe
        self.active_layers = layers or LAYERS
        self.report_out = report_out
        self.constitution_ver = CONSTITUTION_VER
        self.tool_ver = TOOL_VER
        self.start_time = time.time()
        self.stats = {layer: {"pass": 0, "fail": 0, "details": []} for layer in LAYERS}
        self.layer_skipped = {layer: False for layer in LAYERS}
        self._api_samples = {}

    def _record(self, layer, status, table, column, detail):
        icon = {"pass": "✅", "fail": "❌"}[status]
        self.stats[layer][status] += 1
        self.stats[layer]["details"].append(
            "{} [{}] {}.{}: {}".format(icon, layer, table, column, detail)
        )

    def _target_tables(self):
        if self.target_table:
            return [self.target_table]
        return list(DATASET_REGISTRY.keys())

    def _fetch_api_sample(self, table_name):
        """Fetch sample rows from FinMind or FRED API. Returns list or None."""
        if table_name in self._api_samples:
            return self._api_samples[table_name]
        if table_name in INFRA_TABLES:
            self._api_samples[table_name] = None
            return None
        try:
            if table_name == "FredData":
                if not self.include_fred:
                    self._api_samples[table_name] = None
                    return None
                api_key = os.getenv("FRED_API_KEY")
                if not api_key:
                    self._api_samples[table_name] = None
                    return None
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": FRED_CONTRACT_SERIES,
                    "api_key": api_key,
                    "file_type": "json",
                    "limit": self.sample_size,
                }
                res = requests.get(url, params=params, timeout=30)
                res.raise_for_status()
                data = res.json().get("observations", [])
                for d in data:
                    d["series_id"] = FRED_CONTRACT_SERIES
                self._api_samples[table_name] = data
                return data

            if table_name in FINMIND_API_TABLES:
                probe = FINMIND_API_TABLES[table_name]
                client = FinMindClient()
                params = {
                    "dataset": probe["dataset"],
                    "start_date": probe["start_date"],
                }
                if probe.get("data_id") is not None:
                    params["data_id"] = probe.get("data_id", "")
                if client.token:
                    params["token"] = client.token
                res = requests.get(client.api_url, params=params, headers=client.headers, timeout=30)
                res.raise_for_status()
                payload = res.json()
                data = payload.get("data", [])[:self.sample_size]
                self._api_samples[table_name] = data
                return data
        except Exception as exc:
            self._api_samples[table_name] = None
            return None
        return None

    # ============== Layer A: DDL ↔ DB Physical ==============
    def audit_layer_a(self):
        """Layer A: DDL ↔ DB Physical Consistency"""
        if "A" not in self.active_layers:
            self.layer_skipped["A"] = True
            return
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name in self._target_tables():
                if table_name not in DATASET_REGISTRY:
                    continue
                config = DATASET_REGISTRY[table_name]
                cur.execute("""
                    SELECT column_name, data_type, character_maximum_length,
                           numeric_precision, numeric_scale, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                """, (table_name,))
                db_cols = {r[0]: {
                    "data_type": r[1],
                    "char_max_len": r[2],
                    "num_prec": r[3],
                    "num_scale": r[4],
                    "nullable": r[5],
                } for r in cur.fetchall()}

                if not db_cols:
                    for col_name in config["columns"]:
                        self._record("A", "fail", table_name, col_name, "DB 表不存在")
                    continue

                for col_name, ddl_str in config["columns"].items():
                    ddl_parsed = parse_ddl(ddl_str)
                    if col_name not in db_cols:
                        self._record("A", "fail", table_name, col_name,
                                     "DDL 要求 {} 但 DB 無此欄位".format(ddl_parsed["base_type"]))
                        continue
                    db_info = db_cols[col_name]
                    expected_pg = DDL_TO_PG_TYPE.get(ddl_parsed["base_type"])
                    if expected_pg is None:
                        self._record("A", "fail", table_name, col_name,
                                     "DDL type {} 無對應 PostgreSQL type".format(ddl_parsed["base_type"]))
                        continue
                    if db_info["data_type"] != expected_pg:
                        self._record("A", "fail", table_name, col_name,
                                     "type 不符：DDL {} → 預期 {}，實際 {}".format(
                                         ddl_parsed["base_type"], expected_pg, db_info["data_type"]))
                        continue
                    if ddl_parsed["base_type"] == "VARCHAR" and ddl_parsed["length"]:
                        if db_info["char_max_len"] != ddl_parsed["length"]:
                            self._record("A", "fail", table_name, col_name,
                                         "VARCHAR length 不符：DDL {} 實際 {}".format(
                                             ddl_parsed["length"], db_info["char_max_len"]))
                            continue
                    if ddl_parsed["base_type"] == "NUMERIC" and ddl_parsed["precision"]:
                        if db_info["num_prec"] != ddl_parsed["precision"]:
                            self._record("A", "fail", table_name, col_name,
                                         "NUMERIC precision 不符：DDL ({},{}) 實際 ({},{})".format(
                                             ddl_parsed["precision"], ddl_parsed["scale"],
                                             db_info["num_prec"], db_info["num_scale"]))
                            continue
                        if db_info["num_scale"] != ddl_parsed["scale"]:
                            self._record("A", "fail", table_name, col_name,
                                         "NUMERIC scale 不符：DDL ({},{}) 實際 ({},{})".format(
                                             ddl_parsed["precision"], ddl_parsed["scale"],
                                             db_info["num_prec"], db_info["num_scale"]))
                            continue
                    self._record("A", "pass", table_name, col_name,
                                 "{} 對齊".format(expected_pg))
        finally:
            cur.close()
            conn.close()

    # ============== Layer B: API Sample ↔ DDL Type Compatibility ==============
    def audit_layer_b(self):
        if "B" not in self.active_layers or self.skip_api_probe:
            self.layer_skipped["B"] = True
            return
        for table_name in self._target_tables():
            if table_name in INFRA_TABLES:
                continue
            sample = self._fetch_api_sample(table_name)
            if sample is None:
                if table_name == "FredData" and not self.include_fred:
                    continue
                self._record("B", "fail", table_name, "*",
                             "API 取樣失敗（請求 {} 筆）".format(self.sample_size))
                continue
            if not sample:
                continue
            config = DATASET_REGISTRY[table_name]
            derived = LOCAL_DERIVED_COLUMNS.get(table_name, set())
            for col_name, ddl_str in config["columns"].items():
                if col_name in derived:
                    continue
                ddl_parsed = parse_ddl(ddl_str)
                cast_total = 0
                cast_failures = 0
                for row in sample:
                    if col_name not in row:
                        continue
                    val = row[col_name]
                    if val is None or val == "" or val == ".":
                        continue
                    cast_total += 1
                    try:
                        if ddl_parsed["base_type"] == "NUMERIC":
                            Decimal(str(val))
                        elif ddl_parsed["base_type"] in ("VARCHAR", "TEXT"):
                            str(val)
                        elif ddl_parsed["base_type"] == "DATE":
                            datetime.strptime(str(val)[:10], "%Y-%m-%d")
                        elif ddl_parsed["base_type"] == "TIMESTAMP":
                            datetime.strptime(str(val)[:19], "%Y-%m-%d %H:%M:%S")
                        elif ddl_parsed["base_type"] in ("INTEGER", "BIGINT"):
                            int(val)
                    except (InvalidOperation, ValueError, TypeError):
                        cast_failures += 1
                if cast_total == 0:
                    self._record("B", "pass", table_name, col_name,
                                 "樣本中此欄位無非空值")
                elif cast_failures > 0:
                    self._record("B", "fail", table_name, col_name,
                                 "cast 失敗 {}/{}（DDL {}）".format(
                                     cast_failures, cast_total, ddl_parsed["base_type"]))
                else:
                    self._record("B", "pass", table_name, col_name,
                                 "cast 通過 {}/{}".format(cast_total, cast_total))

    # ============== Layer C: Length / Precision Range ==============
    def audit_layer_c(self):
        if "C" not in self.active_layers or self.skip_api_probe:
            self.layer_skipped["C"] = True
            return
        for table_name in self._target_tables():
            if table_name in INFRA_TABLES:
                continue
            sample = self._fetch_api_sample(table_name)
            if not sample:
                continue
            config = DATASET_REGISTRY[table_name]
            derived = LOCAL_DERIVED_COLUMNS.get(table_name, set())
            for col_name, ddl_str in config["columns"].items():
                if col_name in derived:
                    continue
                ddl_parsed = parse_ddl(ddl_str)
                if ddl_parsed["base_type"] == "VARCHAR" and ddl_parsed["length"]:
                    max_len = 0
                    for row in sample:
                        val = row.get(col_name)
                        if val:
                            max_len = max(max_len, len(str(val)))
                    if max_len > ddl_parsed["length"]:
                        self._record("C", "fail", table_name, col_name,
                                     "max length {} > VARCHAR({})".format(max_len, ddl_parsed["length"]))
                    else:
                        self._record("C", "pass", table_name, col_name,
                                     "max length {} ≤ {}".format(max_len, ddl_parsed["length"]))
                elif ddl_parsed["base_type"] == "NUMERIC" and ddl_parsed["precision"]:
                    max_abs = Decimal(0)
                    for row in sample:
                        val = row.get(col_name)
                        if val is None or val == "" or val == ".":
                            continue
                        try:
                            d = Decimal(str(val)).copy_abs()
                            if d > max_abs:
                                max_abs = d
                        except InvalidOperation:
                            continue
                    integer_digits = ddl_parsed["precision"] - ddl_parsed["scale"]
                    max_allowed = Decimal(10) ** integer_digits
                    if max_abs >= max_allowed:
                        self._record("C", "fail", table_name, col_name,
                                     "max abs {} ≥ NUMERIC({},{}) 上限 {}".format(
                                         max_abs, ddl_parsed["precision"], ddl_parsed["scale"], max_allowed))
                    else:
                        self._record("C", "pass", table_name, col_name,
                                     "max abs {} < {}".format(max_abs, max_allowed))

    # ============== Layer D: NULL Ratio Sanity ==============
    def audit_layer_d(self):
        if "D" not in self.active_layers or self.skip_api_probe:
            self.layer_skipped["D"] = True
            return
        for table_name in self._target_tables():
            if table_name in INFRA_TABLES:
                continue
            sample = self._fetch_api_sample(table_name)
            if not sample:
                continue
            config = DATASET_REGISTRY[table_name]
            unique_cols = set(config.get("unique_constraints", []) or [])
            for col_name in config["columns"]:
                null_count = sum(1 for row in sample if row.get(col_name) in (None, "", "."))
                ratio = null_count / len(sample) if sample else 0
                if col_name in unique_cols and null_count > 0:
                    self._record("D", "fail", table_name, col_name,
                                 "unique 欄位含 {} 個 NULL (ratio={:.0%})".format(null_count, ratio))
                else:
                    self._record("D", "pass", table_name, col_name,
                                 "NULL 比例 {:.0%}".format(ratio))

    # ============== Layer E: PK / Unique Uniqueness ==============
    def audit_layer_e(self):
        if "E" not in self.active_layers:
            self.layer_skipped["E"] = True
            return
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name in self._target_tables():
                if table_name not in DATASET_REGISTRY:
                    continue
                config = DATASET_REGISTRY[table_name]
                unique_cols = config.get("unique_constraints", []) or []
                if not unique_cols:
                    continue
                try:
                    cols_str = ", ".join(['"{}"'.format(c) for c in unique_cols])
                    cur.execute('SELECT COUNT(*), COUNT(DISTINCT ({})) FROM "{}"'.format(
                        cols_str, table_name))
                    row = cur.fetchone()
                    if row is None:
                        continue
                    total, distinct = row
                    if total > distinct:
                        self._record("E", "fail", table_name, ",".join(unique_cols),
                                     "unique 衝突：total={}, distinct={}, dup={}".format(
                                         total, distinct, total - distinct))
                    else:
                        self._record("E", "pass", table_name, ",".join(unique_cols),
                                     "unique 一致：{} rows".format(total))
                except Exception as exc:
                    self._record("E", "fail", table_name, ",".join(unique_cols),
                                 "查詢失敗：{}: {}".format(type(exc).__name__, exc))
                    conn.rollback()
        finally:
            cur.close()
            conn.close()

    # ============== Layer F: Duplicate Row Detection ==============
    def audit_layer_f(self):
        if "F" not in self.active_layers:
            self.layer_skipped["F"] = True
            return
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name in self._target_tables():
                if table_name not in DATASET_REGISTRY:
                    continue
                config = DATASET_REGISTRY[table_name]
                cols = [c for c, ddl in config["columns"].items()
                        if "SERIAL" not in ddl.upper()]
                if not cols:
                    continue
                try:
                    cols_str = ", ".join(['"{}"'.format(c) for c in cols])
                    cur.execute('SELECT COUNT(*), COUNT(DISTINCT ({})) FROM "{}"'.format(
                        cols_str, table_name))
                    row = cur.fetchone()
                    if row is None:
                        continue
                    total, distinct = row
                    if total > distinct:
                        self._record("F", "fail", table_name, "row",
                                     "duplicate row：total={}, distinct={}, dup={}".format(
                                         total, distinct, total - distinct))
                    else:
                        self._record("F", "pass", table_name, "row",
                                     "無重複：{} rows".format(total))
                except Exception as exc:
                    self._record("F", "fail", table_name, "row",
                                 "查詢失敗：{}: {}".format(type(exc).__name__, exc))
                    conn.rollback()
        finally:
            cur.close()
            conn.close()

    # ============== Layer G: Date Series Continuity ==============
    def audit_layer_g(self):
        if "G" not in self.active_layers:
            self.layer_skipped["G"] = True
            return
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name in self._target_tables():
                if table_name not in DATASET_REGISTRY:
                    continue
                config = DATASET_REGISTRY[table_name]
                if "date" not in config["columns"] or table_name in INFRA_TABLES:
                    continue
                try:
                    cur.execute('SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM "{}"'.format(
                        table_name))
                    row = cur.fetchone()
                    if row is None or row[0] is None:
                        self._record("G", "pass", table_name, "date", "空表，無需檢驗")
                        continue
                    min_d, max_d, distinct_dates = row
                    days_range = (max_d - min_d).days + 1
                    expected_workdays = days_range * 5 / 7
                    threshold = expected_workdays * 0.8
                    if distinct_dates < threshold and distinct_dates > 0:
                        self._record("G", "fail", table_name, "date",
                                     "date 連續性低：{}~{} 範圍 {} 天，預期工作日 ~{}，實際 {}".format(
                                         min_d, max_d, days_range,
                                         int(expected_workdays), distinct_dates))
                    else:
                        self._record("G", "pass", table_name, "date",
                                     "date 連續性正常：{}~{}, {} 個 distinct dates (~{} 工作日預期)".format(
                                         min_d, max_d, distinct_dates, int(expected_workdays)))
                except Exception as exc:
                    self._record("G", "fail", table_name, "date",
                                 "查詢失敗：{}: {}".format(type(exc).__name__, exc))
                    conn.rollback()
        finally:
            cur.close()
            conn.close()

    # ============== Layer H: Referential Integrity ==============
    def audit_layer_h(self):
        if "H" not in self.active_layers:
            self.layer_skipped["H"] = True
            return
        # 對所有含 stock_id 的非 TaiwanStockInfo 表，驗證 stock_id ∈ TaiwanStockInfo
        stock_level_tables = [
            t for t, cfg in DATASET_REGISTRY.items()
            if "stock_id" in cfg["columns"] and t != "TaiwanStockInfo" and t not in INFRA_TABLES
        ]
        if self.target_table:
            if self.target_table not in stock_level_tables:
                self.layer_skipped["H"] = True
                return
            stock_level_tables = [self.target_table]

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            try:
                cur.execute('SELECT COUNT(*) FROM "TaiwanStockInfo"')
                info_count = cur.fetchone()[0]
            except Exception as exc:
                for t in stock_level_tables:
                    self._record("H", "fail", t, "stock_id",
                                 "TaiwanStockInfo 查詢失敗：{}".format(exc))
                conn.rollback()
                return

            if info_count == 0:
                for t in stock_level_tables:
                    self._record("H", "pass", t, "stock_id",
                                 "TaiwanStockInfo 空表，referential check 跳過（bootstrap 階段）")
                return

            for table_name in stock_level_tables:
                try:
                    cur.execute('''
                        SELECT COUNT(DISTINCT t.stock_id)
                        FROM "{}" t
                        WHERE NOT EXISTS (
                            SELECT 1 FROM "TaiwanStockInfo" i WHERE i.stock_id = t.stock_id
                        )
                    '''.format(table_name))
                    orphan_count = cur.fetchone()[0]
                    if orphan_count > 0:
                        self._record("H", "fail", table_name, "stock_id",
                                     "{} 個 stock_id 不在 TaiwanStockInfo".format(orphan_count))
                    else:
                        self._record("H", "pass", table_name, "stock_id",
                                     "所有 stock_id ∈ TaiwanStockInfo")
                except Exception as exc:
                    self._record("H", "fail", table_name, "stock_id",
                                 "查詢失敗：{}: {}".format(type(exc).__name__, exc))
                    conn.rollback()
        finally:
            cur.close()
            conn.close()

    # ============== Layer I: Value Range Sanity ==============
    def audit_layer_i(self):
        if "I" not in self.active_layers:
            self.layer_skipped["I"] = True
            return
        sanity_rules = {
            "TaiwanStockPrice": [
                ("Trading_Volume", '"Trading_Volume" < 0', "Trading_Volume 為負"),
                ("close", '"close" < 0', "close 為負"),
                ("close", '"close" > 100000', "close > 100000（異常大）"),
            ],
            "TaiwanStockPriceAdj": [
                ("Trading_Volume", '"Trading_Volume" < 0', "Trading_Volume 為負"),
                ("close", '"close" < 0', "close 為負"),
            ],
            "TaiwanStockPER": [
                ("PBR", '"PBR" < 0', "PBR < 0"),
                ("dividend_yield", '"dividend_yield" < 0', "dividend_yield < 0"),
            ],
            "TaiwanStockMonthRevenue": [
                ("revenue", '"revenue" < 0', "revenue < 0"),
            ],
        }
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name, rules in sanity_rules.items():
                if self.target_table and self.target_table != table_name:
                    continue
                if table_name not in DATASET_REGISTRY:
                    continue
                for col, condition, desc in rules:
                    try:
                        cur.execute('SELECT COUNT(*) FROM "{}" WHERE {}'.format(table_name, condition))
                        bad_count = cur.fetchone()[0]
                        if bad_count > 0:
                            self._record("I", "fail", table_name, col,
                                         "{}：{} rows".format(desc, bad_count))
                        else:
                            self._record("I", "pass", table_name, col,
                                         "{}：0 rows".format(desc))
                    except Exception as exc:
                        self._record("I", "fail", table_name, col,
                                     "查詢失敗：{}: {}".format(type(exc).__name__, exc))
                        conn.rollback()
        finally:
            cur.close()
            conn.close()

    # ============== Verdict + Report ==============
    def _compute_verdict(self):
        """動態計算 verdict (§5.6.3 [Zero Hardcoded Verdict])"""
        total_fail = sum(s["fail"] for s in self.stats.values())
        if total_fail > 0:
            return "FAILED"
        return "PERFECT"

    def report(self, verdict):
        """寫入 reports/api_schema_compliance_audit_<timestamp>.md"""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        if self.report_out:
            report_path = Path(self.report_out)
        else:
            report_dir = Path(get_report_dir())
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "api_schema_compliance_audit_{}.md".format(ts)

        latency_ms = (time.time() - self.start_time) * 1000
        lines = []
        lines.append("# API Schema Compliance Audit Report ({})\n".format(self.tool_ver))
        lines.append("**執行日期**: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        lines.append("**對應憲章**: 系統架構大憲章_{}.md / §3.2A / §14.7-AJ\n".format(self.constitution_ver))
        lines.append("## 環境快照\n")
        lines.append("- sample_size: {}".format(self.sample_size))
        lines.append("- include_fred: {}".format(self.include_fred))
        lines.append("- target_table: {}".format(self.target_table or "ALL"))
        lines.append("- skip_api_probe: {}".format(self.skip_api_probe))
        lines.append("- active_layers: {}".format(",".join(self.active_layers)))
        lines.append("")
        lines.append("## 9 層結果摘要\n")
        lines.append("| Layer | 名稱 | PASS | FAIL | 狀態 |")
        lines.append("|---|---|---|---|---|")
        for layer in LAYERS:
            s = self.stats[layer]
            skipped = self.layer_skipped[layer]
            status = "SKIP" if skipped else ("FAIL" if s["fail"] > 0 else "PASS")
            lines.append("| {} | {} | {} | {} | {} |".format(
                layer, LAYER_NAMES[layer], s["pass"], s["fail"], status))
        lines.append("")
        lines.append("**verdict**: {}".format(verdict))
        lines.append("**latency**: {:.1f} ms".format(latency_ms))
        lines.append("")
        lines.append("## 9 層詳細紀錄\n")
        for layer in LAYERS:
            s = self.stats[layer]
            if self.layer_skipped[layer]:
                lines.append("### Layer {}: {} — SKIPPED\n".format(layer, LAYER_NAMES[layer]))
                continue
            lines.append("### Layer {}: {}\n".format(layer, LAYER_NAMES[layer]))
            lines.append("- PASS={}, FAIL={}".format(s["pass"], s["fail"]))
            for detail in s["details"]:
                lines.append("- {}".format(detail))
            lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def run(self):
        """執行 9 層 audit + 寫報告 + 終端摘要 + 動態 verdict"""
        verdict = "FAILED"
        try:
            with record_lifecycle(
                "api_schema_compliance_audit_{}".format(self.tool_ver),
                category="audit",
                stock_id="SYSTEM",
            ) as lc:
                for layer in LAYERS:
                    if layer in self.active_layers:
                        method = getattr(self, "audit_layer_{}".format(layer.lower()))
                        try:
                            method()
                        except Exception as exc:
                            self._record(layer, "fail", "SYSTEM", "*",
                                         "layer {} crashed: {}: {}".format(
                                             layer, type(exc).__name__, exc))

                verdict = self._compute_verdict()
                report_path = self.report(verdict)

                # 終端摘要
                print("\n" + "🛡️" * 40)
                print("🚀 Quantum Finance: API Schema Compliance Audit ({})".format(self.tool_ver))
                print("🛡️" * 40)
                print("治權基準 : 系統架構大憲章_{}.md / §3.2A / §14.7-AJ".format(self.constitution_ver))
                print("─" * 80)
                for layer in LAYERS:
                    s = self.stats[layer]
                    if self.layer_skipped[layer]:
                        print("⏸️  [Layer {}] {}: SKIPPED".format(layer, LAYER_NAMES[layer]))
                        continue
                    icon = "❌" if s["fail"] > 0 else "✅"
                    print("{} [Layer {}] {}: PASS={}, FAIL={}".format(
                        icon, layer, LAYER_NAMES[layer], s["pass"], s["fail"]))
                print("─" * 80)
                latency_ms = (time.time() - self.start_time) * 1000
                print("🕒 總計耗時 : {:.2f} ms".format(latency_ms))
                print("📄 報告路徑 : {}".format(report_path))
                print("⚖️  主權判定 : {}".format(verdict))
                print("🛡️" * 40 + "\n")

                # lifecycle 接線
                if verdict == "FAILED":
                    for layer, s in self.stats.items():
                        if s["fail"] > 0:
                            lc.mark_failed("Layer {} {} failures".format(layer, s["fail"]))

                # data_audit_log 旁系寫入（best-effort）
                try:
                    write_data_audit_log(
                        "API_SCHEMA_COMPLIANCE_AUDIT",
                        "SYSTEM",
                        datetime.now().strftime("%Y-%m-%d"),
                        "AUDIT_{}".format(self.tool_ver),
                        1 if verdict == "PERFECT" else 0,
                    )
                except Exception:
                    pass
        except Exception as exc:
            print("❌ audit 內部例外：{}: {}".format(type(exc).__name__, exc))
            return "FAILED"
        return verdict


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="audit_api_schema_compliance {} - 9 層 schema + 資料完整性深度驗收".format(TOOL_VER)
    )
    parser.add_argument("--include-fred", action="store_true",
                        help="含 FRED API probe（Layer B/C/D）")
    parser.add_argument("--table", type=str,
                        help="只跑單一表（13 張 DATASET_REGISTRY 之一）")
    parser.add_argument("--skip-api-probe", action="store_true",
                        help="略過 Layer B/C/D 之 API 探測；只跑 DB-side 6 層")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Layer B/C/D 取樣大小（預設 {}）".format(DEFAULT_SAMPLE_SIZE))
    parser.add_argument("--layers", type=str,
                        help="comma-separated layers, e.g. A,B,E（預設全 9 層）")
    parser.add_argument("--report-out", type=str,
                        help="自訂報告寫入路徑（預設 reports/api_schema_compliance_audit_<ts>.md）")
    args = parser.parse_args()

    active_layers = LAYERS
    if args.layers:
        active_layers = [l.strip().upper() for l in args.layers.split(",") if l.strip()]
        invalid = [l for l in active_layers if l not in LAYERS]
        if invalid:
            print("❌ 無效的 layer：{}（合法：{}）".format(invalid, LAYERS))
            sys.exit(1)

    auditor = ApiSchemaComplianceAuditor(
        sample_size=args.sample_size,
        include_fred=args.include_fred,
        table=args.table,
        skip_api_probe=args.skip_api_probe,
        layers=active_layers,
        report_out=args.report_out,
    )
    verdict = auditor.run()

    # 對齊 §3.2 接受標準：PERFECT → exit 0，FAILED → exit 1
    sys.exit(0 if verdict == "PERFECT" else 1)
