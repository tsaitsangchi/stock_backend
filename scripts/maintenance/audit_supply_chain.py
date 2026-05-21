"""
audit_supply_chain.py v1.19 (Post-Schema Compliance Audit · Functional Group Matrix Edition)
================================================================================
**最後更新日期**: 2026-05-21
**主權狀態**: POST-SCHEMA AUDIT (憲法 v6.0.0 對齊 + §3.2A 橫切稽核身分自我宣告 + 維運矩陣重組為 5 大功能群視角；8 項檢查面 100% 合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Post-Schema Audit]: 本工具定位於 `data_schema.py --init --force` 之後（憲章 §二 序列 L2403、Step 3 維運矩陣 L2422-2423），驗收 API + DB + logs 三軸對齊。
2. [API Contract Reuse]: API 欄位契約以 `data_schema.py` 當前版本之 API-first probe 為權威來源（透過 import `SovereignSchemaManager` / `DATASET_REGISTRY` / `FINMIND_API_TABLES`），不重新定義契約。
3. [Database State Verification]: 必須驗收 13 張實體表、欄位大小寫、row count、FRED series 完整性與 freshness 能力。
4. [Lifecycle Integrity]: 必須驗收 `pipeline_execution_log` / `data_audit_log`，並透過 `record_lifecycle(... ) as lc` 回寫 warning / failed。
5. [Zero Hardcoded Verdict]: 主權判定動態計算（`compute_verdict()`）：FAIL > 0 → FAILED；WARN > 0 → WARNING；皆 0 → PERFECT，對齊 §5.6.3。
6. [Sovereignty Declaration]: 本程式為 **§3.2A 橫切稽核工具**（cross-ref 憲章 L2470 §3.2A 子表 / L2478 本程式列項 / L2485 §3.2A 治權邊界），執行時點對應 **§3.1 Step 3**（§二 序列 L2403 / 維運矩陣 L2422-2423）。治權邊界：(a) 遵守 §3.2A 治權邊界 L2485（§5.6.3 零硬編 PERFECT、§0.4 可觀察性、§3.2 接受標準：PERFECT/WARNING exit 0；FAILED exit 1）；(b) **不屬 §3.1 序列模組**（不擅自進行 ingestion / decision / sizing）；(c) 五套禁令（§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §6.8）不涉；(d) **T1-T3 不分層**；(e) **§8.5 anti-leakage 不處理**（由 `audit_leakage.py` 負責）；(f) **不選股不評分**（由 `core_universe_builder.py` 負責）；(g) **不持有 Raw API Schema**（由 `data_schema.py` 持有）；(h) 唯一職責：API + DB + logs 三軸驗收。
7. [Historical Reference Authority]: 本程式之 `schema_ver` 屬於記述性快照（記載當下對齊之 `data_schema` 版本），非權威來源；API 契約權威來源永遠是 `data_schema.py` 當前版本之 API-first probe（§3.1 子表 L2452 之「對齊 data_schema.py」表述為治權記述，非硬鎖版本）。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

> 本程式作為 §3.2A 橫切稽核工具，依「驗收面向」拆分為 5 大功能群；每群對應憲章治權契約。
> 接受標準（§3.2A L2485 + §3.2 接受標準）：PERFECT/WARNING → exit 0；FAILED → exit 1。

### Group A. API 契約驗收 (API Contract Verification)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 FinMind 10 表 API 契約 probe（per `FINMIND_API_TABLES`）| `audit_api_contracts(source="finmind")` → `manager._probe_finmind_contract()` | §1.4 / §一 4. [Supply Chain Sovereignty] |
| A.2 FRED 1 表（`FredData`）API 契約 probe | `audit_api_contracts(source="fred")` → `manager._probe_fred_contract()` | §1.4 / §一 4. [Supply Chain Sovereignty] |
| A.3 PASS / FAILED / WARNING 三分類紀錄 | `_record("API-Contract", ...)` | §5.6.3 零硬編 |
| A.4 預設模式合計 11 probes（FinMind 10 + FRED 1）| `audit_api_contracts(source=None)` | §1.4 |
| 對應 CLI | `--api-only` 或 `--source [finmind|fred]` | — |

### Group B. DB Schema 驗收 (Database Physical Schema)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 `to_regclass` 13 表存在性檢查 | `audit_db_schema()._table_exists()` | §1.4 / §3.1 Step 2 後置 |
| B.2 欄位 missing / extra 比對 DATASET_REGISTRY | `audit_db_schema()` | §1.4 SSOT |
| B.3 row count 統計 | `audit_db_schema()` | §0.4 可觀察性 |
| B.4 欄位大小寫精確匹配 | `actual_cols vs expected_cols` | §1.4 API mirror |
| 對應 CLI | `--db-only` 或預設 | — |

### Group C. FRED Series & Freshness (FRED 數據新鮮度)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 FredData 表存在性 | `audit_fred_series_and_freshness()._table_exists()` | §1.4 |
| C.2 4 大 series 完整性（DFF/UNRATE/T10Y2Y/VIXCLS）| `FRED_MACRO_LIST` 比對 | FRED bootstrap |
| C.3 per-series freshness 閾值（DFF/T10Y2Y/VIXCLS 7d；UNRATE 60d）| `FRED_FRESHNESS_DAYS` | §1.4 |
| C.4 pre-ingestion DEFERRED 容忍 | row_count == 0 時不 FAIL | bootstrap-tolerant |
| 對應 CLI | 預設啟用（非 `--api-only`）| — |

### Group D. Lifecycle Logs 驗收 (Pipeline & Data Audit Logs)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1 `pipeline_execution_log` / `data_audit_log` 表存在 | `audit_logs()` | §0.4 / §1.6 Hybrid Observability |
| D.2 24h 視窗 task status 異常掃描 | `audit_logs(window_hours=24)` | §3.2A / Step 3 log-window L2491 |
| D.3 task `end_time IS NULL` ANOMALY 偵測 | `audit_logs()` | §0.4 |
| D.4 `record_lifecycle()` 自我寫入 + warning/failed 回寫 | `run()` 内的 `with record_lifecycle(...)` | §0.4 / §3.2A L2485 |
| D.5 pre-bootstrap clean window 容忍 | DEFERRED 不 FAIL | L2491 容許 |
| 對應 CLI | `--include-logs` | — |

### Group E. Verdict 動態判定 (Zero Hardcoded Verdict)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| E.1 PASS / WARN / FAIL 計數 | `_record()` 之 counter | §5.6.3 |
| E.2 動態 verdict 計算 | `compute_verdict()` | §5.6.3 |
| E.3 FAILED → `sys.exit(1)` | `run()` 末段 | §3.2 接受標準 / §3.2A L2485 |
| E.4 PERFECT/WARNING → exit 0 | 隱含預設 | §3.2 接受標準 |
| E.5 `record_lifecycle` warning/failed 回寫 | `lc.mark_warning()` / `lc.mark_failed()` | §0.4 |
| E.6 報告產生（`reports/compliance_audit_<ts>.md`）| `generate_report(verdict)` | §0.4 audit trail |
| 對應 CLI | 所有模式皆觸發 | — |

### 對齊憲章 §二 維運矩陣（標準場景）
| 場景 | 指令 | 對應功能群 |
| :--- | :--- | :--- |
| **3. [schema 後驗收：API+DB+logs]** | `python scripts/maintenance/audit_supply_chain.py --include-logs` | A + B + C + D + E |
| **3A. [離線驗收：僅 DB+logs]** | `python scripts/maintenance/audit_supply_chain.py --db-only --include-logs` | B + C + D + E |
| **3B. [API 契約驗收：FinMind/FRED]** | `python scripts/maintenance/audit_supply_chain.py --api-only` | A + E |
| **3C. [單一來源 API 驗收]** | `python scripts/maintenance/audit_supply_chain.py --api-only --source [finmind|fred]` | A + E |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.19** | 2026-05-21 | Codex | **8 項標頭強制檢驗 100% 合規 + 維運矩陣重組為 5 大功能群視角**：(a) 主權狀態行補入 v1.19 修補摘要；(b) 最後更新日期 2026-05-14 → 2026-05-21；(c) 核心定義新增 [Sovereignty Declaration] §3.2A 橫切稽核身分自我宣告 + [Historical Reference Authority]（[Truth-based Verdict] 重命名為 [Zero Hardcoded Verdict] 對齊 §5.6.3 與全系統治權慣例）；(d) cross-ref 精確行號補入（§3.2A L2470/L2478/L2485 + §二 L2403 + 維運矩陣 L2422-2423 + Step 3 詮釋 L2491）；(e) 維運矩陣重組為 5 大功能群（A. API 契約 / B. DB Schema / C. FRED Freshness / D. Lifecycle Logs / E. Verdict 動態判定），場景擴至 4 條（3/3A/3B/3C）；(f) cosmetic 對齊：`data_schema v2.11` → 動態 `v2.16`；對齊憲章 v5.4.22 → v6.0.0-FINAL；補入模組級 `CONSTITUTION_VER` + `TOOL_VER` 常數；`self.schema_ver` 更新至 v2.16。介面零變動：4 個 CLI flag (`--source` / `--db-only` / `--api-only` / `--include-logs`) 不變、5 大驗收方法不變、`compute_verdict()` 邏輯不變、`record_lifecycle()` 整合不變。對應 CLAUDE.md §四 #4 8 項標頭強制檢驗治權慣例。 | **ACTIVE** |
| v1.18 | 2026-05-14 | Codex | **schema 後驗收稽核**：對齊憲章 v5.4.22 與 data_schema v2.11；驗收 API contract、13 張 DB table、pipeline/data audit logs；接上 lifecycle context warning/failed 回寫。 | SUPERSEDED |
| v1.17 | 2026-05-13 | Auto-patch | DB-state aware 稽核；新增 FRED 完整性、lifecycle log 交叉比對、動態 verdict。 | SUPERSEDED |
| v1.16 | 2026-05-13 | Antigravity | 創世圓滿：對齊憲法 v5.4.18。 | ARCHIVED |
================================================================================
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v1.19"

_THIS_FILE = Path(__file__).resolve()
_MAINTENANCE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINTENANCE_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.path_setup import get_report_dir
    from core.db_utils import get_db_connection, record_lifecycle
    from core.data_schema import DATASET_REGISTRY, FINMIND_API_TABLES, SovereignSchemaManager
except ImportError as exc:
    print(f"❌ 核心組件導入失敗: {exc}")
    sys.exit(1)


class ComplianceAuditor:
    FRED_MACRO_LIST = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]
    FRED_FRESHNESS_DAYS = {"DFF": 7, "T10Y2Y": 7, "VIXCLS": 7, "UNRATE": 60}
    INFRA_TABLES = {"pipeline_execution_log", "data_audit_log"}

    def __init__(self):
        self.constitution_ver = CONSTITUTION_VER
        self.tool_ver = TOOL_VER
        self.schema_ver = "v2.16"
        self.report_path = get_report_dir() / f"compliance_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        self.audit_results = []
        self.pass_count = 0
        self.warn_count = 0
        self.fail_count = 0

    def _record(self, source, item, status, detail):
        self.audit_results.append([source, item, status, detail])
        upper = status.upper()
        if "❌" in status or "FAILED" in upper:
            self.fail_count += 1
        elif "⚠️" in status or "WARN" in upper or "STALE" in upper or "ANOMALY" in upper:
            self.warn_count += 1
        else:
            self.pass_count += 1

    def compute_verdict(self):
        if self.fail_count > 0:
            return "FAILED"
        if self.warn_count > 0:
            return "WARNING"
        return "PERFECT"

    def _table_exists(self, cur, table_name):
        cur.execute("SELECT to_regclass(%s);", (f'public."{table_name}"',))
        return cur.fetchone()[0] is not None

    def audit_api_contracts(self, source=None):
        if source:
            print(f"🔎 正在驗收 {source} API 契約 (憲法 {self.constitution_ver}, data_schema {self.schema_ver})...")
        else:
            print(f"🔎 正在驗收 FinMind/FRED API 契約 (憲法 {self.constitution_ver}, data_schema {self.schema_ver})...")

        manager = SovereignSchemaManager()
        targets = []
        if source == "finmind":
            targets = list(FINMIND_API_TABLES.keys())
        elif source == "fred":
            targets = ["FredData"]
        else:
            targets = list(FINMIND_API_TABLES.keys()) + ["FredData"]

        for table_name in targets:
            before = len(manager.contract_stats["details"])
            try:
                if table_name == "FredData":
                    manager._probe_fred_contract()
                else:
                    manager._probe_finmind_contract(table_name)
            except Exception as exc:
                manager._record_contract("failed", table_name, f"{type(exc).__name__}: {exc}")
            for line in manager.contract_stats["details"][before:]:
                if "[API-PASS]" in line:
                    self._record("API-Contract", table_name, "✅ PASS", line)
                elif "[API-FAILED]" in line:
                    self._record("API-Contract", table_name, "❌ FAILED", line)
                else:
                    self._record("API-Contract", table_name, "⚠️ WARNING", line)

    def audit_db_schema(self):
        print(f"🧱 正在驗收 DB 實體 schema (13 tables, data_schema {self.schema_ver})...")
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name, config in DATASET_REGISTRY.items():
                if not self._table_exists(cur, table_name):
                    self._record("DB-Schema", table_name, "❌ FAILED", "table missing")
                    continue

                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (table_name,),
                )
                actual_cols = [row[0] for row in cur.fetchall()]
                expected_cols = list(config["columns"].keys())
                missing = [col for col in expected_cols if col not in actual_cols]
                extra = [col for col in actual_cols if col not in expected_cols]

                cur.execute(f'SELECT COUNT(*) FROM "{table_name}";')
                row_count = cur.fetchone()[0]

                if missing or extra:
                    detail = []
                    if missing:
                        detail.append(f"missing={missing}")
                    if extra:
                        detail.append(f"extra={extra}")
                    self._record("DB-Schema", table_name, "❌ FAILED", "; ".join(detail))
                else:
                    self._record("DB-Schema", table_name, "✅ PASS", f"{len(actual_cols)} columns matched; rows={row_count}")
        finally:
            cur.close()
            conn.close()

    def audit_fred_series_and_freshness(self):
        print(f"📈 正在驗收 FRED series 與 freshness 能力...")
        conn = get_db_connection()
        cur = conn.cursor()
        today = datetime.now().date()
        try:
            if not self._table_exists(cur, "FredData"):
                self._record("Freshness", "FredData", "❌ FAILED", "FredData table missing")
                return

            cur.execute('SELECT COUNT(*) FROM "FredData";')
            total = cur.fetchone()[0]
            if total == 0:
                self._record("Freshness", "FredData", "✅ DEFERRED", "pre-ingestion: table exists but has no data yet")
                return

            cur.execute('SELECT series_id, COUNT(*) FROM "FredData" GROUP BY series_id;')
            present = {row[0]: row[1] for row in cur.fetchall()}
            missing = [sid for sid in self.FRED_MACRO_LIST if sid not in present]
            if missing:
                self._record("DB-FRED", "completeness", "❌ FAILED", f"missing series={missing}")
            else:
                self._record("DB-FRED", "completeness", "✅ PASS", f"series counts={present}")

            for sid in self.FRED_MACRO_LIST:
                cur.execute('SELECT MAX(date) FROM "FredData" WHERE series_id = %s;', (sid,))
                latest = cur.fetchone()[0]
                if latest is None:
                    self._record("Freshness", f"FRED/{sid}", "✅ DEFERRED", "pre-ingestion: no data yet")
                    continue
                threshold = self.FRED_FRESHNESS_DAYS.get(sid, 7)
                age = (today - latest).days
                if age > threshold:
                    self._record("Freshness", f"FRED/{sid}", "⚠️ STALE", f"latest={latest}, age={age}d, threshold={threshold}d")
                else:
                    self._record("Freshness", f"FRED/{sid}", "✅ PASS", f"latest={latest}, age={age}d")
        finally:
            cur.close()
            conn.close()

    def audit_logs(self, window_hours=24):
        print(f"📝 正在驗收 pipeline_execution_log / data_audit_log...")
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            for table_name in ["pipeline_execution_log", "data_audit_log"]:
                if not self._table_exists(cur, table_name):
                    self._record("Log-Schema", table_name, "❌ FAILED", "log table missing")
                    continue
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}";')
                count = cur.fetchone()[0]
                self._record("Log-Schema", table_name, "✅ PASS", f"exists; rows={count}")

            if not self._table_exists(cur, "pipeline_execution_log"):
                return
            cutoff = datetime.now() - timedelta(hours=window_hours)
            cur.execute(
                """
                SELECT task_name, status, start_time, end_time, error_msg
                FROM pipeline_execution_log
                WHERE start_time >= %s
                ORDER BY start_time DESC;
                """,
                (cutoff,),
            )
            rows = cur.fetchall()
            if not rows:
                self._record("Pipeline-Log", "recent_tasks", "✅ DEFERRED", f"no tasks in last {window_hours}h")
                return

            bad_status = [row for row in rows if row[1] and row[1].lower() not in {"success", "warning"}]
            if bad_status:
                sample = "; ".join(f"{r[0]}={r[1]}" for r in bad_status[:3])
                self._record("Pipeline-Log", "task_status", "❌ FAILED", sample)
            else:
                self._record("Pipeline-Log", "task_status", "✅ PASS", f"{len(rows)} recent tasks have acceptable status")

            missing_end = [row for row in rows if row[3] is None]
            if missing_end:
                sample = "; ".join(row[0] for row in missing_end[:3])
                self._record("Pipeline-Log", "end_time", "⚠️ ANOMALY", f"end_time NULL: {sample}")
            else:
                self._record("Pipeline-Log", "end_time", "✅ PASS", f"{len(rows)} recent tasks have end_time")
        finally:
            cur.close()
            conn.close()

    def generate_report(self, verdict):
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("# Quantum Finance schema 後驗收稽核報告\n\n")
            f.write(f"- **時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **憲章**: 系統架構大憲章_{self.constitution_ver}.md\n")
            f.write(f"- **稽核工具**: audit_supply_chain {self.tool_ver}\n")
            f.write(f"- **schema 基準**: data_schema {self.schema_ver}\n")
            f.write(f"- **判定結果**: **{verdict}** (PASS={self.pass_count}, WARN={self.warn_count}, FAIL={self.fail_count})\n\n")
            f.write("## 稽核明細\n\n")
            f.write("| 來源層 | 項目 | 狀態 | 詳細 |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for source, item, status, detail in self.audit_results:
                safe_detail = str(detail).replace("\n", "<br>")
                f.write(f"| {source} | {item} | {status} | {safe_detail} |\n")

    def run(self, source=None, db_only=False, api_only=False, include_logs=False):
        task_name = f"post_schema_audit_{TOOL_VER}"
        with record_lifecycle(task_name, category="maintenance", stock_id="SYSTEM") as lifecycle:
            if not db_only:
                self.audit_api_contracts(source=source)
            if not api_only:
                self.audit_db_schema()
                self.audit_fred_series_and_freshness()
                if include_logs or db_only:
                    self.audit_logs()

            verdict = self.compute_verdict()
            if verdict == "FAILED" and hasattr(lifecycle, "mark_failed"):
                lifecycle.mark_failed(f"audit_supply_chain {self.tool_ver} failed: FAIL={self.fail_count}")
            elif verdict == "WARNING" and hasattr(lifecycle, "mark_warning"):
                lifecycle.mark_warning(f"audit_supply_chain {self.tool_ver} warning: WARN={self.warn_count}")

            self.generate_report(verdict)

            print("\n" + "🛡️" * 40)
            print(f"🚀 Quantum Finance: schema 後驗收稽核 ({self.tool_ver})")
            print("🛡️" * 40)
            print(f"📊 治權對齊報告 : {self.report_path.name}")
            print(f"📊 稽核項目統計 : PASS={self.pass_count}, WARN={self.warn_count}, FAIL={self.fail_count}")
            print(f"🛡️ 對齊基準     : 憲法 {self.constitution_ver} / data_schema {self.schema_ver}")
            print(f"⚖️  主權判定     : {verdict}")
            print("🛡️" * 40 + "\n")

            if verdict == "FAILED":
                sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Quantum Finance schema 後驗收稽核 ({TOOL_VER})")
    parser.add_argument("--source", type=str, choices=["finmind", "fred"], help="只驗收指定 API 來源")
    parser.add_argument("--db-only", action="store_true", help="只跑 DB/logs 驗收，跳過 API 呼叫")
    parser.add_argument("--api-only", action="store_true", help="只跑 API contract 驗收，跳過 DB/logs")
    parser.add_argument("--include-logs", action="store_true", help="納入 pipeline_execution_log / data_audit_log 驗收")
    args = parser.parse_args()

    ComplianceAuditor().run(
        source=args.source,
        db_only=args.db_only,
        api_only=args.api_only,
        include_logs=args.include_logs,
    )
