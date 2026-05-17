"""
audit_downstream_readiness.py v0.1
================================================================================
Quantum Finance §8 Promotion Readiness Audit Authority

Purpose:
1. Summarize whether §8 has enough historical clean-validation evidence.
2. Separately decide whether §8 is ready for v6.1.0 successor production-current promotion.
3. Write a promotion readiness report under reports/.
================================================================================
"""
import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_MAINTENANCE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINTENANCE_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_REPORTS_DIR = _PROJECT_ROOT / "reports"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v0.1"
HISTORICAL_MODEL_CUTOFF = "2025-05-15"
HISTORICAL_MODEL_CUTOFF_DATE = date.fromisoformat(HISTORICAL_MODEL_CUTOFF)
FORMAL_LABEL_HORIZON = 20
MODEL_ID_PATTERN = re.compile(r"^mdl_[0-9]{8}_[a-z0-9]+_h[0-9]+_[0-9a-f]{8}_v[0-9]+_[0-9]+$")

REQUIRED_TABLES = [
    "feature_store_snapshot",
    "feature_definition",
    "feature_values",
    "model_registry",
    "model_training_run",
    "prediction_run",
    "prediction_values",
    "core_universe_snapshot",
    "core_universe_membership",
    "TaiwanStockPriceAdj",
]

REQUIRED_FILES = [
    _PROJECT_ROOT / "scripts" / "core" / "feature_store_builder.py",
    _PROJECT_ROOT / "scripts" / "core" / "model_trainer.py",
    _PROJECT_ROOT / "scripts" / "core" / "prediction_engine.py",
    _PROJECT_ROOT / "scripts" / "maintenance" / "audit_leakage.py",
]


@dataclass
class AuditItem:
    status: str
    check: str
    detail: str


class DownstreamReadinessAuditor:
    def __init__(self, write_report=True):
        self.write_report = write_report
        self.items = []
        self.report_path = None
        self.context = {}

    def add(self, status, check, detail):
        self.items.append(AuditItem(status, check, detail))
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
        print(f"{icon} [{status}] {check}: {detail}")

    def pass_(self, check, detail):
        self.add("PASS", check, detail)

    def warn(self, check, detail):
        self.add("WARN", check, detail)

    def fail(self, check, detail):
        self.add("FAIL", check, detail)

    def counts(self):
        return {
            "PASS": sum(1 for item in self.items if item.status == "PASS"),
            "WARN": sum(1 for item in self.items if item.status == "WARN"),
            "FAIL": sum(1 for item in self.items if item.status == "FAIL"),
        }

    def verdict(self):
        counts = self.counts()
        if counts["FAIL"]:
            return "FAILED"
        if self.context.get("production_current_ready"):
            return "READY_FOR_V5_4_23"
        if self.context.get("historical_clean_ready"):
            return "READY_FOR_DRAFT_EVIDENCE"
        return "WARNING"

    def _table_exists(self, cur, table_name):
        cur.execute("SELECT to_regclass(%s);", (f'public."{table_name}"',))
        return cur.fetchone()[0] is not None

    def check_files(self):
        for path in REQUIRED_FILES:
            if path.exists():
                self.pass_("required_file", f"{path.relative_to(_PROJECT_ROOT)} exists")
            else:
                self.fail("required_file", f"{path.relative_to(_PROJECT_ROOT)} missing")

    def check_tables(self, cur):
        for table in REQUIRED_TABLES:
            if self._table_exists(cur, table):
                self.pass_("required_table", f"{table} exists")
            else:
                self.fail("required_table", f"{table} missing")

    def check_committed_model(self, cur):
        cur.execute(
            """
            SELECT mr.model_id, mr.feature_set_id, mr.universe_snapshot_id, mr.label_horizon,
                   mr.metrics->>'ic_mean', (mr.metrics->>'label_date_max')::date, mr.metrics->>'trainer',
                   mr.artifact_path,
                   EXISTS (
                       SELECT 1
                       FROM "prediction_run" pr
                       WHERE pr.model_id = mr.model_id
                         AND pr.status = 'committed'
                   ) AS has_committed_prediction,
                   f.as_of_date,
                   f.feature_set_version
            FROM "model_registry" mr
            JOIN "feature_store_snapshot" f ON f.feature_set_id = mr.feature_set_id
            WHERE mr.status = 'committed'
            ORDER BY mr.created_at DESC, mr.model_id DESC
            """
        )
        rows = cur.fetchall()
        if not rows:
            self.fail("committed_model_cardinality", "committed models=0, expected at least 1")
            return None

        prediction_backed = [row for row in rows if row[8]]
        if len(prediction_backed) != 1:
            self.fail(
                "committed_model_cardinality",
                f"prediction-backed committed models={len(prediction_backed)}, expected exactly 1; committed models={len(rows)}",
            )
            return None

        model_id, feature_set_id, universe_snapshot_id, horizon, ic_raw, label_max, trainer, artifact_path, _, feature_as_of, feature_version = prediction_backed[0]
        self.context["model_id"] = model_id
        self.context["feature_set_id"] = feature_set_id
        self.context["universe_snapshot_id"] = universe_snapshot_id
        self.context["artifact_path"] = artifact_path
        self.context["model_label_date_max"] = label_max
        self.context["model_feature_as_of"] = feature_as_of
        self.context["model_feature_set_version"] = feature_version
        self.context["committed_model_count"] = len(rows)
        self.pass_(
            "committed_model_cardinality",
            f"current prediction-backed model={model_id}; committed_model_count={len(rows)}; historical walk-forward models allowed",
        )

        bad_model_ids = [row[0] for row in rows if not MODEL_ID_PATTERN.match(row[0])]
        if not bad_model_ids:
            self.pass_("model_id_governance", f"all committed model_ids include feature_set_version hash: count={len(rows)}")
        else:
            self.fail("model_id_governance", f"model_ids without hash rule: {bad_model_ids}")

        bad_horizons = [(row[0], row[3]) for row in rows if row[3] != FORMAL_LABEL_HORIZON]
        if not bad_horizons:
            self.pass_("label_horizon", f"all committed models horizon={FORMAL_LABEL_HORIZON}: count={len(rows)}")
        else:
            self.fail("label_horizon", f"bad horizons={bad_horizons}, expected {FORMAL_LABEL_HORIZON}")

        cur.execute(
            """
            SELECT COALESCE(MAX(as_of_date), DATE '1900-01-01')
            FROM "core_universe_snapshot"
            WHERE status = 'committed'
            """
        )
        latest_core_as_of = cur.fetchone()[0]
        cutoff_violations = []
        historical_count = 0
        production_count = 0
        for row in rows:
            model_id, _, _, horizon, _, label_max, _, _, _, feature_as_of, feature_version = row
            if horizon != FORMAL_LABEL_HORIZON or not label_max:
                cutoff_violations.append((model_id, "missing_or_bad_horizon", label_max))
                continue
            required_label_date = feature_as_of + timedelta(days=horizon)
            is_production_current = (
                feature_as_of == latest_core_as_of
                or "production_current" in (feature_version or "")
            )
            if is_production_current:
                production_count += 1
                if label_max < required_label_date:
                    cutoff_violations.append((model_id, "production_label_not_mature", label_max))
            else:
                historical_count += 1
                if label_max > HISTORICAL_MODEL_CUTOFF_DATE:
                    cutoff_violations.append((model_id, "historical_cutoff", label_max))
        if not cutoff_violations:
            self.pass_(
                "model_data_cutoff",
                f"historical label_date_max <= {HISTORICAL_MODEL_CUTOFF}; production-current uses required_label_date gate: "
                f"historical={historical_count}, production_current={production_count}",
            )
        else:
            self.fail("model_data_cutoff", f"cutoff violations={cutoff_violations}")

        quality_failures = []
        for row in rows:
            try:
                ic_value = float(row[4])
            except (TypeError, ValueError):
                ic_value = None
            if ic_value is None or ic_value <= 0:
                quality_failures.append((row[0], row[4]))
        if not quality_failures:
            self.pass_("model_quality", f"all committed models have ic_mean > 0: count={len(rows)}; current_trainer={trainer}")
        else:
            self.fail("model_quality", f"ic_mean failures={quality_failures}")

        missing_artifacts = []
        for row in rows:
            artifact = _PROJECT_ROOT / row[7] / "model.json"
            if not artifact.exists():
                missing_artifacts.append(str(artifact.relative_to(_PROJECT_ROOT)))
        if not missing_artifacts:
            self.pass_("model_artifact", f"all committed model artifacts exist: count={len(rows)}")
        else:
            self.fail("model_artifact", f"missing artifacts={missing_artifacts}")

        return prediction_backed[0]

    def check_feature_set(self, cur):
        feature_set_id = self.context.get("feature_set_id")
        if not feature_set_id:
            return
        cur.execute(
            """
            SELECT feature_set_id, feature_set_version, as_of_date, source_data_cutoff,
                   universe_snapshot_id, label_horizon, status, total_stocks, feature_count
            FROM "feature_store_snapshot"
            WHERE feature_set_id = %s
            """,
            (feature_set_id,),
        )
        row = cur.fetchone()
        if not row:
            self.fail("feature_set", f"{feature_set_id} missing")
            return

        _, version, as_of_date, source_cutoff, universe_snapshot_id, horizon, status, total_stocks, feature_count = row
        self.context["feature_set_version"] = version
        self.context["as_of_date"] = as_of_date
        self.context["source_data_cutoff"] = source_cutoff
        if status == "committed":
            self.pass_("feature_set_status", f"{feature_set_id} committed")
        else:
            self.fail("feature_set_status", f"{feature_set_id} status={status}")
        if universe_snapshot_id == self.context.get("universe_snapshot_id"):
            self.pass_("feature_universe_lock", "model and feature_set universe match")
        else:
            self.fail("feature_universe_lock", "model and feature_set universe mismatch")
        if horizon == FORMAL_LABEL_HORIZON:
            self.pass_("feature_label_horizon", f"horizon={horizon}")
        else:
            self.fail("feature_label_horizon", f"horizon={horizon}, expected {FORMAL_LABEL_HORIZON}")
        if total_stocks == 150 and feature_count >= 20:
            self.pass_("feature_coverage", f"stocks={total_stocks}, feature_count={feature_count}")
        else:
            self.fail("feature_coverage", f"stocks={total_stocks}, feature_count={feature_count}")

        cur.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(CASE WHEN is_null_imputed THEN 1 ELSE 0 END), 0)
            FROM "feature_values"
            WHERE feature_set_id = %s
            """,
            (feature_set_id,),
        )
        value_rows, imputed_rows = cur.fetchone()
        if value_rows > 0:
            self.pass_("feature_values", f"rows={value_rows}, imputed={imputed_rows}")
        else:
            self.fail("feature_values", "no feature values")

    def check_prediction(self, cur):
        model_id = self.context.get("model_id")
        if not model_id:
            return
        cur.execute(
            """
            SELECT run_id, feature_set_id, universe_snapshot_id, rows_written, status
            FROM "prediction_run"
            WHERE model_id = %s AND status = 'committed'
            ORDER BY run_id DESC
            """,
            (model_id,),
        )
        rows = cur.fetchall()
        if len(rows) != 1:
            self.fail("committed_prediction_cardinality", f"committed prediction runs={len(rows)}, expected exactly 1")
            return
        run_id, feature_set_id, universe_snapshot_id, rows_written, _ = rows[0]
        self.context["prediction_run_id"] = run_id
        self.pass_("committed_prediction_cardinality", f"run={run_id}")
        if feature_set_id == self.context.get("feature_set_id") and universe_snapshot_id == self.context.get("universe_snapshot_id"):
            self.pass_("prediction_lock", "prediction run matches model feature_set/universe")
        else:
            self.fail("prediction_lock", "prediction run lock mismatch")

        cur.execute(
            """
            SELECT COUNT(DISTINCT stock_id)
            FROM "prediction_values"
            WHERE run_id = %s
            """,
            (run_id,),
        )
        prediction_rows = cur.fetchone()[0]
        if rows_written == 150 and prediction_rows == 150:
            self.pass_("prediction_coverage", f"rows_written={rows_written}, values={prediction_rows}")
        else:
            self.fail("prediction_coverage", f"rows_written={rows_written}, values={prediction_rows}, expected 150")

    def check_historical_readiness(self):
        counts = self.counts()
        ready = counts["FAIL"] == 0
        self.context["historical_clean_ready"] = ready
        if ready:
            self.pass_("historical_clean_validation", "Step 9->10->11->11A evidence is clean for draft acceptance")
        else:
            self.fail("historical_clean_validation", "one or more clean validation checks failed")

    def check_production_current(self, cur):
        cur.execute(
            """
            SELECT snapshot_id, as_of_date
            FROM "core_universe_snapshot"
            WHERE status = 'committed'
            ORDER BY as_of_date DESC, created_at DESC, snapshot_id DESC
            LIMIT 1
            """
        )
        snapshot = cur.fetchone()
        if not snapshot:
            self.fail("production_current_snapshot", "no committed core_universe_snapshot")
            self.context["production_current_ready"] = False
            return
        snapshot_id, as_of_date = snapshot
        required_label_date = as_of_date + timedelta(days=FORMAL_LABEL_HORIZON)
        cur.execute('SELECT MAX(date) FROM "TaiwanStockPriceAdj"')
        max_price_date = cur.fetchone()[0]
        self.context["production_snapshot_id"] = snapshot_id
        self.context["production_as_of_date"] = as_of_date
        self.context["production_required_label_date"] = required_label_date
        self.context["max_price_date"] = max_price_date
        model_feature_as_of = self.context.get("model_feature_as_of")
        model_label_date_max = self.context.get("model_label_date_max")

        if not max_price_date or max_price_date < required_label_date:
            self.context["production_current_ready"] = False
            self.warn(
                "production_current_label_window",
                f"blocked: max_price_date={max_price_date}, required_label_date={required_label_date}",
            )
        elif model_feature_as_of != as_of_date:
            self.context["production_current_ready"] = False
            self.warn(
                "production_current_delivery_model",
                f"blocked: prediction-backed model feature_as_of={model_feature_as_of}, production_as_of_date={as_of_date}",
            )
        elif not model_label_date_max or model_label_date_max < required_label_date:
            self.context["production_current_ready"] = False
            self.warn(
                "production_current_delivery_model",
                f"blocked: model_label_date_max={model_label_date_max}, required_label_date={required_label_date}",
            )
        else:
            self.context["production_current_ready"] = True
            self.pass_(
                "production_current_label_window",
                f"max_price_date={max_price_date} >= required_label_date={required_label_date}; "
                f"prediction-backed model feature_as_of={model_feature_as_of}, label_date_max={model_label_date_max}",
            )

    def audit_db(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            self.check_tables(cur)
            if any(item.status == "FAIL" and item.check == "required_table" for item in self.items):
                return
            self.check_committed_model(cur)
            self.check_feature_set(cur)
            self.check_prediction(cur)
            self.check_historical_readiness()
            self.check_production_current(cur)
        finally:
            cur.close()
            conn.close()

    def write_report_file(self):
        if not self.write_report:
            return None
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = _REPORTS_DIR / f"downstream_promotion_readiness_{timestamp}.md"
        counts = self.counts()
        verdict = self.verdict()
        lines = [
            "# Downstream Promotion Readiness Report",
            "",
            f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- constitution: 系統架構大憲章_{CONSTITUTION_VER}.md §8",
            f"- tool: audit_downstream_readiness.py {TOOL_VER}",
            f"- verdict: {verdict}",
            f"- PASS/WARN/FAIL: {counts['PASS']}/{counts['WARN']}/{counts['FAIL']}",
            "",
            "## Current Evidence",
            "",
            f"- model_id: `{self.context.get('model_id', 'N/A')}`",
            f"- feature_set_id: `{self.context.get('feature_set_id', 'N/A')}`",
            f"- feature_set_version: `{self.context.get('feature_set_version', 'N/A')}`",
            f"- prediction_run_id: `{self.context.get('prediction_run_id', 'N/A')}`",
            f"- as_of_date: `{self.context.get('as_of_date', 'N/A')}`",
            f"- historical model cutoff: `{HISTORICAL_MODEL_CUTOFF}`",
            "- production-current cutoff: `required_label_date` gate (not historical cutoff)",
            "",
            "## Production-Current Gate",
            "",
            f"- production_snapshot_id: `{self.context.get('production_snapshot_id', 'N/A')}`",
            f"- production_as_of_date: `{self.context.get('production_as_of_date', 'N/A')}`",
            f"- required_label_date: `{self.context.get('production_required_label_date', 'N/A')}`",
            f"- max_price_date: `{self.context.get('max_price_date', 'N/A')}`",
            "",
            "## Decision",
            "",
        ]
        if verdict == "READY_FOR_V5_4_23":
            lines.append("§8 is successor-ready for production-current promotion review; verdict name is retained for compatibility.")
        elif verdict == "READY_FOR_DRAFT_EVIDENCE":
            lines.append("§8 has clean historical h20 evidence, but successor promotion is blocked until production-current label data and delivery model are available.")
        else:
            lines.append("§8 is not ready; failed checks must be fixed before promotion review.")
        lines.extend(["", "## Checks", ""])
        for item in self.items:
            lines.append(f"- **{item.status}** `{item.check}`: {item.detail}")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.report_path

    def report(self, start, report_path):
        counts = self.counts()
        verdict = self.verdict()
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: Downstream Promotion Readiness ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md §8")
        print(f"📊 PASS/WARN/FAIL : {counts['PASS']}/{counts['WARN']}/{counts['FAIL']}")
        print(f"📄 報告 : {report_path.name if report_path else 'NO-REPORT'}")
        print(f"🕒 總計耗時 : {(time.time() - start)*1000:.2f} ms")
        print(f"⚖️  升版判定 : {verdict}")
        print("🛡️" * 40 + "\n")
        return counts["FAIL"] == 0

    def run(self):
        start = time.time()
        lifecycle_cm = record_lifecycle("audit_downstream_readiness_v0.1", category="audit", stock_id="SYSTEM")
        lifecycle = lifecycle_cm.__enter__()
        try:
            print("🔎 正在執行 §8 Promotion Readiness 稽核...")
            self.check_files()
            self.audit_db()
            counts = self.counts()
            if counts["FAIL"]:
                lifecycle.mark_failed("downstream readiness failed")
            elif self.verdict() != "READY_FOR_V5_4_23":
                lifecycle.mark_warning(self.verdict())
            try:
                write_data_audit_log(
                    "audit_downstream_readiness",
                    "SYSTEM",
                    datetime.now().strftime("%Y-%m-%d"),
                    self.verdict(),
                    sum(counts.values()),
                )
            except Exception as exc:
                self.warn("audit_log", f"write_data_audit_log failed: {type(exc).__name__}: {exc}")
            report_path = self.write_report_file()
            return self.report(start, report_path)
        finally:
            lifecycle_cm.__exit__(None, None, None)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Finance §8 Downstream Promotion Readiness Audit (v0.1)")
    parser.add_argument("--no-report", action="store_true", help="Do not write reports/downstream_promotion_readiness_*.md")
    return parser.parse_args()


def main():
    args = parse_args()
    ok = DownstreamReadinessAuditor(write_report=not args.no_report).run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
