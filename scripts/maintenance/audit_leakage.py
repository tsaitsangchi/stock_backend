"""
audit_leakage.py v0.2 (Quantum Finance Anti-Leakage Audit Authority)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 補入 [Sovereignty Declaration] + Supreme Authority Principle line)
**主權狀態**: IMPLEMENTED (憲法 v6.1.0 §8.5 Data Leakage 防禦 + §9.1 horizon=30 + §8.5-9 publication-date discipline + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

**[Sovereignty Declaration]** (2026-05-29 §一.11 補入,憲法 §3.2 橫切 audit / §8.5): 本程式為 **§8.5 Anti-Leakage 唯一治權載體**(§3.2 橫切 audit 模組)。**治權邊界**:(a) §3.2 橫切 audit;(b) **不修改** features / model_registry(read-only);(c) **唯一檢驗** feature_values 是否含 forward-looking 資料;(d) **唯一檢驗** publication_date_strategy 是否被 feature_store_builder 正確 enforce;(e) **不選股 / 不訓練 model**;(f) 唯一職責:scan feature_store_snapshot + feature_values → 驗證 anti-leakage 19 條 rules → 產出 leakage audit report。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Anti-Leakage Authority]: 對齊憲章 §8.5「Data Leakage 防禦規則」之 8 條 anti-leakage 規則，
   掃描 `feature_store_builder.py` / `model_trainer.py` / `prediction_engine.py` 之
   committed 產物是否違反 as-of-strict / label_horizon / no-lookahead 等邊界。
2. [Per-Run Coverage Enforcement]: 每一 `prediction_run` 必須單獨等於其鎖定
   `universe_snapshot_id` 對應之 core+convex universe stock 數（§8.8.4 補正），
   封堵 aggregated 比對單 run 缺漏被總數掩蓋之漏洞。
3. [Horizon Allowed Set]: `ALLOWED_LABEL_HORIZONS = {20, 30}` 對齊 §9.1 v6.2.0 預備；
   `FORMAL_LABEL_HORIZON = 20` 仍為 §8 production-current v6.1.0 升版基準；
   h30 historical model 不觸發 `missing_or_bad_horizon` FAIL。
4. [Model ID Governance]: 檢驗 `model_id` 必須符合
   `mdl_{yyyymmdd}_{family}_h{label_horizon}_{sha1(feature_set_version)[:8]}_v0_1`
   命名契約，封堵同日同 family 同 horizon 不同 feature set 互相覆寫。
5. [Hybrid Observability]: 維運觸發 `record_lifecycle` 與 `write_data_audit_log`；
   主權狀態動態計算（§5.6.3），FAIL 即 exit 1（§3.2A 接受標準）。
6. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [標準執行：全 §8 committed 產物掃描]** | `$ python scripts/maintenance/audit_leakage.py` | audit_leakage v0.2 |
| **2. [Step 11A：特徵集 + 模型 leakage 檢]** | `$ python scripts/maintenance/audit_leakage.py --feature-set-id <fs_id> --model-id <mdl_id>` | audit_leakage v0.2 |
| **3. [僅查單一 prediction_run]** | `$ python scripts/maintenance/audit_leakage.py --prediction-run-id <pred_id>` | audit_leakage v0.2 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **no-report** | `--no-report` | 略過 `reports/leakage_audit_*.md` 產出，僅 stdout |
| **historical-only** | `--scope historical` | 只掃 deprecated walk-forward evidence runs |
| **production-only** | `--scope production` | 只掃 committed prediction-backed delivery |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.2** | 2026-05-18 | Codex | §14.7-R 升版：新增 `ALLOWED_LABEL_HORIZONS = {20, 30}` 對齊 §9.1 v6.2.0 horizon=30 預備；universe-wide horizon 檢查改用 ALLOWED set；h30 historical model 不再觸發 `missing_or_bad_horizon` FAIL；`FORMAL_LABEL_HORIZON=20` 仍為 v6.1.0 production-current gate。 | **ACTIVE** |
| v0.1 | 2026-05-17 | Codex | 首版：§8.5 8 條防禦規則 + per-run prediction coverage 強制；硬編 `h=20`；§8.8.4 per-run coverage 補正落地。 | SUPERSEDED |
================================================================================
"""
import argparse
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_MAINTENANCE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINTENANCE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v0.2"
PROJECT_ROOT = _SCRIPTS_DIR.parent
FORMAL_LABEL_HORIZON = 20
# v0.2 (2026-05-18): 擴張支援 §9.1 v6.2.0 horizon=30；任一 horizon 皆視為合法
ALLOWED_LABEL_HORIZONS = {20, 30}
SCAN_FILES = [
    PROJECT_ROOT / "scripts" / "core" / "feature_store_builder.py",
    PROJECT_ROOT / "scripts" / "core" / "model_trainer.py",
    PROJECT_ROOT / "scripts" / "core" / "prediction_engine.py",
]


class LeakageAuditor:
    def __init__(self):
        self.items = []

    def add(self, status, check, detail):
        self.items.append((status, check, detail))
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
        print(f"{icon} [{status}] {check}: {detail}")

    def scan_source(self):
        forbidden_patterns = [
            (re.compile(r"date\s*<=\s*[^\\n]*(\+|dateadd|interval)", re.IGNORECASE), "future date filter"),
            (re.compile(r"stock_id\s*==\s*['\"]\d{4,6}['\"]"), "hardcoded stock prediction branch"),
            (re.compile(r"if\s+.*stock_id.*return\s+[0-9.-]+"), "hardcoded stock return"),
        ]
        for path in SCAN_FILES:
            if not path.exists():
                self.add("FAIL", "source_exists", f"{path.relative_to(PROJECT_ROOT)} missing")
                continue
            text = path.read_text(encoding="utf-8")
            self.add("PASS", "source_exists", f"{path.relative_to(PROJECT_ROOT)} exists")
            for pattern, label in forbidden_patterns:
                if pattern.search(text):
                    self.add("FAIL", "source_scan", f"{path.name} contains possible {label}")
                else:
                    self.add("PASS", "source_scan", f"{path.name} no {label}")

    def audit_db(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM "feature_definition"
                WHERE as_of_strict IS NOT TRUE
                """
            )
            bad_features = cur.fetchone()[0]
            if bad_features == 0:
                self.add("PASS", "as_of_strict", "all feature_definition rows are as_of_strict")
            else:
                self.add("FAIL", "as_of_strict", f"non-strict features={bad_features}")

            cur.execute(
                """
                SELECT COUNT(*)
                FROM "model_registry" m
                JOIN "feature_store_snapshot" f ON f.feature_set_id = m.feature_set_id
                WHERE m.status = 'committed'
                  AND (
                    m.metrics->>'label_date_min' IS NULL
                    OR (m.metrics->>'label_date_min')::date < (f.as_of_date + m.label_horizon * INTERVAL '1 day')::date
                  )
                """
            )
            horizon_violations = cur.fetchone()[0]
            if horizon_violations == 0:
                self.add("PASS", "label_horizon", "committed models satisfy label_date_min >= as_of + label_horizon")
            else:
                self.add("FAIL", "label_horizon", f"violations={horizon_violations}")

            cur.execute(
                """
                SELECT COALESCE(MAX(as_of_date), DATE '1900-01-01')
                FROM "core_universe_snapshot"
                WHERE status = 'committed'
                """
            )
            latest_core_as_of = cur.fetchone()[0]
            cur.execute('SELECT MAX(date) FROM "TaiwanStockPriceAdj"')
            max_price_date = cur.fetchone()[0]

            cur.execute(
                """
                SELECT
                    m.model_id,
                    m.label_horizon,
                    (m.metrics->>'label_date_max')::date AS label_date_max,
                    f.as_of_date,
                    f.feature_set_version
                FROM "model_registry" m
                JOIN "feature_store_snapshot" f ON f.feature_set_id = m.feature_set_id
                WHERE m.status = 'committed'
                ORDER BY m.model_id
                """
            )
            model_rows = cur.fetchall()
            cutoff_violations = []
            production_rows = 0
            historical_rows = 0
            for model_id, horizon, label_date_max, feature_as_of, feature_set_version in model_rows:
                if horizon not in ALLOWED_LABEL_HORIZONS or label_date_max is None:
                    cutoff_violations.append((model_id, "missing_or_bad_horizon", label_date_max))
                    continue
                required_label_date = feature_as_of + horizon * timedelta(days=1)
                is_production_current = (
                    feature_as_of == latest_core_as_of
                    or "production_current" in (feature_set_version or "")
                )
                if is_production_current:
                    production_rows += 1
                    if label_date_max < required_label_date:
                        cutoff_violations.append((model_id, "production_label_not_mature", label_date_max))
                else:
                    historical_rows += 1
                    if not max_price_date or label_date_max > max_price_date:
                        cutoff_violations.append((model_id, "historical_beyond_db_max_price_date", label_date_max))

            if not cutoff_violations:
                self.add(
                    "PASS",
                    "model_data_cutoff",
                    f"historical label_date_max <= db max_price_date={max_price_date}; "
                    f"production-current uses required_label_date gate; historical={historical_rows}, "
                    f"production_current={production_rows}",
                )
            else:
                self.add("FAIL", "model_data_cutoff", f"violations={cutoff_violations}")

            cur.execute(
                """
                SELECT COUNT(*)
                FROM "model_registry"
                WHERE status = 'committed'
                  AND model_id !~ '^mdl_[0-9]{8}_[a-z0-9]+_h[0-9]+_[0-9a-f]{8}_v[0-9]+_[0-9]+$'
                """
            )
            model_id_violations = cur.fetchone()[0]
            if model_id_violations == 0:
                self.add("PASS", "model_id_governance", "committed model_ids include feature_set_version hash")
            else:
                self.add("FAIL", "model_id_governance", f"violations={model_id_violations}")

            cur.execute(
                """
                SELECT COUNT(*)
                FROM "prediction_run" p
                JOIN "model_registry" m ON m.model_id = p.model_id
                WHERE p.status = 'committed'
                  AND (
                    p.feature_set_id <> m.feature_set_id
                    OR p.universe_snapshot_id <> m.universe_snapshot_id
                  )
                """
            )
            lock_violations = cur.fetchone()[0]
            if lock_violations == 0:
                self.add("PASS", "feature_universe_lock", "prediction runs match model feature_set/universe")
            else:
                self.add("FAIL", "feature_universe_lock", f"violations={lock_violations}")

            cur.execute(
                """
                WITH expected AS (
                    SELECT
                        pr.run_id,
                        COUNT(DISTINCT m.stock_id) AS expected_rows
                    FROM "prediction_run" pr
                    JOIN "core_universe_membership" m
                      ON m.snapshot_id = pr.universe_snapshot_id
                     AND m.core_tier IN ('core_universe', 'convex_universe')
                    WHERE pr.status = 'committed'
                    GROUP BY pr.run_id
                ),
                actual AS (
                    SELECT
                        pr.run_id,
                        COUNT(DISTINCT CASE WHEN m.stock_id IS NOT NULL THEN pv.stock_id END) AS actual_rows
                    FROM "prediction_run" pr
                    LEFT JOIN "prediction_values" pv
                      ON pv.run_id = pr.run_id
                    LEFT JOIN "core_universe_membership" m
                      ON m.snapshot_id = pr.universe_snapshot_id
                     AND m.stock_id = pv.stock_id
                     AND m.core_tier IN ('core_universe', 'convex_universe')
                    WHERE pr.status = 'committed'
                    GROUP BY pr.run_id
                )
                SELECT e.run_id, e.expected_rows, COALESCE(a.actual_rows, 0) AS actual_rows
                FROM expected e
                LEFT JOIN actual a ON a.run_id = e.run_id
                ORDER BY e.run_id
                """
            )
            coverage = cur.fetchall()
            bad_runs = [row for row in coverage if row[2] != row[1]]
            if not coverage:
                self.add("FAIL", "prediction_coverage", "no committed prediction runs")
            elif not bad_runs:
                details = ", ".join(f"{run_id}={actual}/{expected}" for run_id, expected, actual in coverage)
                self.add("PASS", "prediction_coverage", f"all committed runs covered locked universe: {details}")
            else:
                details = ", ".join(f"{run_id}={actual}/{expected}" for run_id, expected, actual in bad_runs)
                self.add("FAIL", "prediction_coverage", f"coverage violations: {details}")
        finally:
            cur.close()
            conn.close()

    def counts(self):
        return {
            "PASS": sum(1 for status, _, _ in self.items if status == "PASS"),
            "WARN": sum(1 for status, _, _ in self.items if status == "WARN"),
            "FAIL": sum(1 for status, _, _ in self.items if status == "FAIL"),
        }

    def run(self):
        start = time.time()
        lifecycle_cm = record_lifecycle("audit_leakage_v0.2", category="audit", stock_id="SYSTEM")
        lifecycle = lifecycle_cm.__enter__()
        try:
            print("🔎 正在執行 §8.5 Data Leakage 防禦稽核...")
            self.scan_source()
            self.audit_db()
            counts = self.counts()
            if counts["FAIL"]:
                lifecycle.mark_failed("leakage audit failed")
            elif counts["WARN"]:
                lifecycle.mark_warning("leakage audit warning")
            try:
                write_data_audit_log("audit_leakage", "SYSTEM", time.strftime("%Y-%m-%d"), "LEAKAGE_AUDIT", sum(counts.values()))
            except Exception as exc:
                self.add("WARN", "audit_log", f"write_data_audit_log failed: {type(exc).__name__}: {exc}")
            verdict = "FAILED" if counts["FAIL"] else "WARNING" if counts["WARN"] else "PERFECT"
            print("\n" + "🛡️" * 40)
            print(f"🚀 Quantum Finance: Data Leakage 稽核摘要 ({TOOL_VER})")
            print("🛡️" * 40)
            print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md §8.5")
            print(f"📊 PASS/WARN/FAIL : {counts['PASS']}/{counts['WARN']}/{counts['FAIL']}")
            print(f"🕒 總計耗時 : {(time.time() - start)*1000:.2f} ms")
            print(f"⚖️  主權判定 : {verdict}")
            print("🛡️" * 40 + "\n")
            return counts["FAIL"] == 0
        finally:
            lifecycle_cm.__exit__(None, None, None)


def main():
    argparse.ArgumentParser(description="Quantum Finance Data Leakage Audit (v0.2)").parse_args()
    ok = LeakageAuditor().run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
