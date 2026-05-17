"""
audit_supply_chain.py v1.18 (Post-Schema Compliance Audit Edition)
================================================================================
**最後更新日期**: 2026-05-14
**主權狀態**: POST-SCHEMA AUDIT (憲法 v5.4.22 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Post-Schema Audit]: 本工具定位於 `data_schema.py v2.11 --init --force` 之後，驗收 API + DB + logs。
2. [API Contract Reuse]: API 欄位契約以 `data_schema.py v2.11` 的 API-first probe 為權威來源。
3. [Database State Verification]: 必須驗收 13 張實體表、欄位大小寫、row count、FRED series 與 freshness 能力。
4. [Lifecycle Integrity]: 必須驗收 `pipeline_execution_log` / `data_audit_log`，並透過 `record_lifecycle(... ) as lc` 回寫 warning / failed。
5. [Truth-based Verdict]: 主權判定必須動態計算：FAIL > 0 -> FAILED；WARN > 0 -> WARNING；皆 0 -> PERFECT。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 (Exhaustive Examples) | 對齊模組 |
| :--- | :--- | :--- |
| **1. [schema 後驗收：API + DB + logs]** | `$ python scripts/maintenance/audit_supply_chain.py --include-logs` | audit_tool v1.18 |
| **2. [離線驗收：僅 DB + logs]** | `$ python scripts/maintenance/audit_supply_chain.py --db-only --include-logs` | audit_tool v1.18 |
| **3. [API 契約驗收：FinMind/FRED]** | `$ python scripts/maintenance/audit_supply_chain.py --api-only` | audit_tool v1.18 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.18** | 2026-05-14 | Codex | **schema 後驗收稽核**：對齊憲章 v5.4.22 與 data_schema v2.11；驗收 API contract、13 張 DB table、pipeline/data audit logs；接上 lifecycle context warning/failed 回寫。 | **ACTIVE** |
| v1.17 | 2026-05-13 | Auto-patch | DB-state aware 稽核；新增 FRED 完整性、lifecycle log 交叉比對、動態 verdict。 | SUPERSEDED |
| v1.16 | 2026-05-13 | Antigravity | 創世圓滿：對齊憲法 v5.4.18。 | ARCHIVED |
================================================================================
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

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
        self.constitution_ver = "v6.0.0"
        self.tool_ver = "v1.18"
        self.schema_ver = "v2.11"
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
        task_name = "post_schema_audit_v1.18"
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
    parser = argparse.ArgumentParser(description="Quantum Finance schema 後驗收稽核 (v1.18)")
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
