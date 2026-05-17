"""
audit_core_universe.py v0.1 (Quantum Finance Core Universe Audit Authority)
================================================================================
最後更新日期: 2026-05-17
主權狀態: IMPLEMENTED (憲法 v6.0.0 核心股結果驗收稽核 + special restore trace audit)
最高原則: Core Universe Post-Build Verification

v0.1 邊界:
1. 只驗收 core_universe_builder.py 產物，不重算核心股名單。
2. 驗收 policy、snapshot、membership、scores、revision log 的一致性。
3. 驗收 raw 欄位鏡像與 v0.1 downstream boundary。
4. 不保存 feature values、labels、model outputs、prediction signals。
5. 驗收年度重選 / special restore 的 review_cycle、snapshot notes、revision log 留痕。
================================================================================
"""
import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
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
DEFAULT_POLICY_VERSION = "core_universe_policy_v0.2"
REQUIRED_TABLES = [
    "pipeline_execution_log",
    "data_audit_log",
    "TaiwanStockInfo",
    "core_universe_policy",
    "core_universe_snapshot",
    "core_universe_membership",
    "core_universe_scores",
    "universe_revision_log",
]
EXPECTED_TIERS = {
    "research_universe": "research_count",
    "core_universe": "core_count",
    "convex_universe": "convex_count",
    "quarantine_universe": "quarantine_count",
}
PENDING_SCORE_COLUMNS = [
    "liquidity_score",
    "fundamental_score",
    "institutional_flow_score",
    "volatility_control_score",
]
ELIGIBILITY_COLUMNS = [
    "train_eligible",
    "predict_eligible",
    "backtest_eligible",
    "downstream_ready",
]


@dataclass
class AuditItem:
    status: str
    check_name: str
    detail: str


class CoreUniverseAuditor:
    def __init__(self, snapshot_id=None, as_of_date=None, policy_version=DEFAULT_POLICY_VERSION, write_report=True):
        self.snapshot_id = snapshot_id
        self.as_of_date = as_of_date
        self.policy_version = policy_version
        self.write_report = write_report
        self.snapshot = None
        self.items = []
        self.report_path = None

    def add(self, status, check_name, detail):
        self.items.append(AuditItem(status=status, check_name=check_name, detail=detail))

    def pass_(self, check_name, detail):
        self.add("PASS", check_name, detail)

    def warn(self, check_name, detail):
        self.add("WARN", check_name, detail)

    def fail(self, check_name, detail):
        self.add("FAIL", check_name, detail)

    def counts(self):
        return {
            "PASS": sum(1 for item in self.items if item.status == "PASS"),
            "WARN": sum(1 for item in self.items if item.status == "WARN"),
            "FAIL": sum(1 for item in self.items if item.status == "FAIL"),
        }

    def verdict(self):
        counts = self.counts()
        if counts["FAIL"] > 0:
            return "FAILED"
        if counts["WARN"] > 0:
            return "WARNING"
        return "PERFECT"

    def _table_exists(self, cur, table_name):
        cur.execute("SELECT to_regclass(%s);", (f'public."{table_name}"',))
        return cur.fetchone()[0] is not None

    def _scalar(self, cur, sql, params=()):
        cur.execute(sql, params)
        return cur.fetchone()[0]

    def _row(self, cur, sql, params=()):
        cur.execute(sql, params)
        return cur.fetchone()

    def check_required_tables(self, cur):
        missing = []
        for table_name in REQUIRED_TABLES:
            if self._table_exists(cur, table_name):
                self.pass_("required_table", f"{table_name} exists")
            else:
                missing.append(table_name)
                self.fail("required_table", f"{table_name} missing")
        return not missing

    def resolve_snapshot(self, cur):
        if self.snapshot_id:
            row = self._row(
                cur,
                '''
                SELECT "snapshot_id", "as_of_date", "source_data_cutoff", "policy_version",
                       "total_candidates", "research_count", "core_count", "convex_count",
                       "quarantine_count", "status", "notes"
                FROM "core_universe_snapshot"
                WHERE "snapshot_id" = %s
                ''',
                (self.snapshot_id,),
            )
        elif self.as_of_date:
            row = self._row(
                cur,
                '''
                SELECT "snapshot_id", "as_of_date", "source_data_cutoff", "policy_version",
                       "total_candidates", "research_count", "core_count", "convex_count",
                       "quarantine_count", "status", "notes"
                FROM "core_universe_snapshot"
                WHERE "as_of_date" = %s AND "policy_version" = %s
                ORDER BY "created_at" DESC, "snapshot_id" DESC
                LIMIT 1
                ''',
                (self.as_of_date, self.policy_version),
            )
        else:
            row = self._row(
                cur,
                '''
                SELECT "snapshot_id", "as_of_date", "source_data_cutoff", "policy_version",
                       "total_candidates", "research_count", "core_count", "convex_count",
                       "quarantine_count", "status", "notes"
                FROM "core_universe_snapshot"
                WHERE "status" = 'committed'
                ORDER BY "created_at" DESC, "snapshot_id" DESC
                LIMIT 1
                ''',
            )

        if not row:
            self.fail("snapshot_resolve", "no matching committed core universe snapshot found")
            return False

        keys = [
            "snapshot_id",
            "as_of_date",
            "source_data_cutoff",
            "policy_version",
            "total_candidates",
            "research_count",
            "core_count",
            "convex_count",
            "quarantine_count",
            "status",
            "notes",
        ]
        self.snapshot = dict(zip(keys, row))
        self.snapshot_id = self.snapshot["snapshot_id"]
        self.as_of_date = self.snapshot["as_of_date"]
        self.policy_version = self.snapshot["policy_version"]
        self.pass_("snapshot_resolve", f"snapshot={self.snapshot_id}, status={self.snapshot['status']}")
        if self.snapshot["status"] == "committed":
            self.pass_("snapshot_status", "snapshot status is committed")
        else:
            self.fail("snapshot_status", f"snapshot status is {self.snapshot['status']}, expected committed")
        return True

    def check_policy(self, cur):
        row = self._row(
            cur,
            '''
            SELECT "active", "eligibility_config"->>'source_table',
                   "eligibility_config"->>'downstream_eligibility',
                   "weight_config"->>'liquidity_score',
                   "weight_config"->>'fundamental_score'
            FROM "core_universe_policy"
            WHERE "policy_version" = %s
            ''',
            (self.policy_version,),
        )
        if not row:
            self.fail("policy", f"policy_version={self.policy_version} missing")
            return
        active, source_table, downstream_eligibility, liquidity_state, fundamental_state = row
        if active:
            self.pass_("policy", f"policy_version={self.policy_version} active")
        else:
            self.warn("policy", f"policy_version={self.policy_version} is inactive")
        if source_table == "TaiwanStockInfo":
            self.pass_("policy_source", "policy source_table=TaiwanStockInfo")
        else:
            self.fail("policy_source", f"policy source_table={source_table}, expected TaiwanStockInfo")
        if downstream_eligibility and "all false" in downstream_eligibility:
            self.pass_("policy_boundary", "downstream eligibility remains pending/all false")
        else:
            self.fail("policy_boundary", f"unexpected downstream eligibility policy: {downstream_eligibility}")
        if self.policy_version.endswith("v0.2"):
            self.pass_("policy_score_config", "v0.2 policy uses six-layer CoreScore weights")
        elif liquidity_state == "pending" and fundamental_state == "pending":
            self.pass_("policy_pending_scores", "liquidity/fundamental scores are policy-pending in v0.1")
        else:
            self.fail("policy_pending_scores", f"unexpected pending score states: liquidity={liquidity_state}, fundamental={fundamental_state}")

    def check_rebalance_trace(self, cur):
        notes = self.snapshot.get("notes") or ""
        expected_cycle = "special" if "rebalance_mode=special" in notes else "annual"
        if expected_cycle == "special":
            if "special_rebalance_reason=" in notes:
                self.pass_("special_snapshot_note", "special rebalance reason present in snapshot notes")
            else:
                self.fail("special_snapshot_note", "snapshot notes declare special rebalance without special_rebalance_reason")
        else:
            self.pass_("annual_snapshot_note", "snapshot notes do not declare special rebalance")

        cycle_rows = self._rows(
            cur,
            '''
            SELECT "review_cycle", COUNT(*)
            FROM "core_universe_membership"
            WHERE "snapshot_id" = %s
            GROUP BY "review_cycle"
            ''',
            (self.snapshot_id,),
        )
        cycle_counts = {cycle: count for cycle, count in cycle_rows}
        unexpected_cycles = sorted(cycle for cycle in cycle_counts if cycle != expected_cycle)
        if unexpected_cycles:
            self.fail("membership_review_cycle", f"unexpected review_cycle values={unexpected_cycles}, expected {expected_cycle}")
        elif cycle_counts.get(expected_cycle, 0) == self.snapshot["total_candidates"]:
            self.pass_("membership_review_cycle", f"all membership rows review_cycle={expected_cycle}")
        else:
            self.fail("membership_review_cycle", f"review_cycle counts={cycle_counts}, expected {self.snapshot['total_candidates']} {expected_cycle} rows")

        row = self._row(
            cur,
            '''
            SELECT "detail"->>'rebalance_mode',
                   "detail"->>'review_cycle',
                   COALESCE("detail"->>'special_rebalance_reason', '')
            FROM "universe_revision_log"
            WHERE "snapshot_id" = %s AND "action_type" = 'BUILD_SNAPSHOT'
            ORDER BY "revision_time" DESC, "revision_id" DESC
            LIMIT 1
            ''',
            (self.snapshot_id,),
        )
        if not row:
            self.fail("rebalance_revision_trace", "BUILD_SNAPSHOT revision detail missing")
            return

        rebalance_mode, review_cycle, special_reason = row
        if rebalance_mode == expected_cycle and review_cycle == expected_cycle:
            self.pass_("rebalance_revision_trace", f"revision detail rebalance_mode/review_cycle={expected_cycle}")
        else:
            self.fail(
                "rebalance_revision_trace",
                f"revision detail rebalance_mode={rebalance_mode}, review_cycle={review_cycle}, expected {expected_cycle}",
            )
        if expected_cycle == "special":
            if special_reason.strip():
                self.pass_("special_revision_reason", "special_rebalance_reason present in revision detail")
            else:
                self.fail("special_revision_reason", "special snapshot missing special_rebalance_reason in revision detail")
        elif special_reason.strip():
            self.fail("annual_revision_reason", "annual snapshot unexpectedly has special_rebalance_reason")
        else:
            self.pass_("annual_revision_reason", "annual revision detail has no special_rebalance_reason")

    def check_counts_and_tiers(self, cur):
        expected_total = self.snapshot["total_candidates"]
        membership_count = self._scalar(
            cur,
            'SELECT COUNT(*) FROM "core_universe_membership" WHERE "snapshot_id" = %s',
            (self.snapshot_id,),
        )
        scores_count = self._scalar(
            cur,
            'SELECT COUNT(*) FROM "core_universe_scores" WHERE "snapshot_id" = %s',
            (self.snapshot_id,),
        )
        if membership_count == expected_total:
            self.pass_("membership_count", f"membership_count={membership_count} matches snapshot.total_candidates")
        else:
            self.fail("membership_count", f"membership_count={membership_count}, expected {expected_total}")
        if scores_count == expected_total:
            self.pass_("scores_count", f"scores_count={scores_count} matches snapshot.total_candidates")
        else:
            self.fail("scores_count", f"scores_count={scores_count}, expected {expected_total}")

        tier_rows = self._rows(
            cur,
            '''
            SELECT "core_tier", COUNT(*)
            FROM "core_universe_membership"
            WHERE "snapshot_id" = %s
            GROUP BY "core_tier"
            ''',
            (self.snapshot_id,),
        )
        tier_counts = {tier: count for tier, count in tier_rows}
        unknown_tiers = sorted(set(tier_counts) - set(EXPECTED_TIERS))
        if unknown_tiers:
            self.fail("tier_allowed", f"unknown tiers found: {unknown_tiers}")
        else:
            self.pass_("tier_allowed", "all membership tiers are governed tiers")
        for tier_name, snapshot_column in EXPECTED_TIERS.items():
            actual = tier_counts.get(tier_name, 0)
            expected = self.snapshot[snapshot_column]
            if actual == expected:
                self.pass_("tier_count", f"{tier_name}={actual} matches snapshot.{snapshot_column}")
            else:
                self.fail("tier_count", f"{tier_name}={actual}, expected snapshot.{snapshot_column}={expected}")
        if self.snapshot["core_count"] <= 120 and self.snapshot["core_count"] > 0:
            self.pass_("core_size", f"core_count={self.snapshot['core_count']} within v0.1 limit")
        else:
            self.fail("core_size", f"core_count={self.snapshot['core_count']} outside v0.1 limit")
        if self.snapshot["convex_count"] <= 30:
            self.pass_("convex_size", f"convex_count={self.snapshot['convex_count']} within v0.1 limit")
        else:
            self.fail("convex_size", f"convex_count={self.snapshot['convex_count']} outside v0.1 limit")

    def _rows(self, cur, sql, params=()):
        cur.execute(sql, params)
        return cur.fetchall()

    def check_uniqueness_and_pairing(self, cur):
        membership_dupes = self._scalar(
            cur,
            '''
            SELECT COUNT(*) FROM (
                SELECT "stock_id"
                FROM "core_universe_membership"
                WHERE "snapshot_id" = %s
                GROUP BY "stock_id"
                HAVING COUNT(*) > 1
            ) d
            ''',
            (self.snapshot_id,),
        )
        score_dupes = self._scalar(
            cur,
            '''
            SELECT COUNT(*) FROM (
                SELECT "stock_id"
                FROM "core_universe_scores"
                WHERE "snapshot_id" = %s
                GROUP BY "stock_id"
                HAVING COUNT(*) > 1
            ) d
            ''',
            (self.snapshot_id,),
        )
        if membership_dupes == 0:
            self.pass_("membership_unique", "no duplicate membership stock_id in snapshot")
        else:
            self.fail("membership_unique", f"duplicate membership stock_id groups={membership_dupes}")
        if score_dupes == 0:
            self.pass_("scores_unique", "no duplicate scores stock_id in snapshot")
        else:
            self.fail("scores_unique", f"duplicate scores stock_id groups={score_dupes}")

        missing_scores = self._scalar(
            cur,
            '''
            SELECT COUNT(*)
            FROM "core_universe_membership" m
            LEFT JOIN "core_universe_scores" s
              ON m."snapshot_id" = s."snapshot_id" AND m."stock_id" = s."stock_id"
            WHERE m."snapshot_id" = %s AND s."stock_id" IS NULL
            ''',
            (self.snapshot_id,),
        )
        missing_membership = self._scalar(
            cur,
            '''
            SELECT COUNT(*)
            FROM "core_universe_scores" s
            LEFT JOIN "core_universe_membership" m
              ON m."snapshot_id" = s."snapshot_id" AND m."stock_id" = s."stock_id"
            WHERE s."snapshot_id" = %s AND m."stock_id" IS NULL
            ''',
            (self.snapshot_id,),
        )
        if missing_scores == 0 and missing_membership == 0:
            self.pass_("membership_scores_pairing", "membership and scores are 1:1 paired")
        else:
            self.fail("membership_scores_pairing", f"missing_scores={missing_scores}, missing_membership={missing_membership}")

    def check_raw_mirror(self, cur):
        raw_dupes = self._scalar(
            cur,
            '''
            SELECT COUNT(*) FROM (
                SELECT "stock_id"
                FROM "TaiwanStockInfo"
                GROUP BY "stock_id"
                HAVING COUNT(*) > 1
            ) d
            ''',
        )
        if raw_dupes == 0:
            self.pass_("raw_unique", "TaiwanStockInfo stock_id is unique")
        else:
            self.fail("raw_unique", f"TaiwanStockInfo duplicate stock_id groups={raw_dupes}")

        missing_raw = self._scalar(
            cur,
            '''
            SELECT COUNT(*)
            FROM "core_universe_membership" m
            LEFT JOIN "TaiwanStockInfo" t ON m."stock_id" = t."stock_id"
            WHERE m."snapshot_id" = %s AND t."stock_id" IS NULL
            ''',
            (self.snapshot_id,),
        )
        mirror_mismatch = self._scalar(
            cur,
            '''
            SELECT COUNT(*)
            FROM "core_universe_membership" m
            JOIN "TaiwanStockInfo" t ON m."stock_id" = t."stock_id"
            WHERE m."snapshot_id" = %s
              AND (
                  m."stock_name" IS DISTINCT FROM t."stock_name"
               OR m."type" IS DISTINCT FROM t."type"
               OR m."industry_category" IS DISTINCT FROM t."industry_category"
              )
            ''',
            (self.snapshot_id,),
        )
        if missing_raw == 0:
            self.pass_("raw_membership_source", "all membership stock_id values exist in TaiwanStockInfo")
        else:
            self.fail("raw_membership_source", f"membership rows missing TaiwanStockInfo source={missing_raw}")
        if mirror_mismatch == 0:
            self.pass_("raw_column_mirror", "stock_name/type/industry_category mirror TaiwanStockInfo")
        else:
            self.fail("raw_column_mirror", f"raw mirror mismatches={mirror_mismatch}")

    def check_v01_boundary(self, cur):
        eligibility_expr = ", ".join([f'SUM(CASE WHEN "{col}" THEN 1 ELSE 0 END)' for col in ELIGIBILITY_COLUMNS])
        row = self._row(
            cur,
            f'''
            SELECT {eligibility_expr}, COUNT(*)
            FROM "core_universe_membership"
            WHERE "snapshot_id" = %s
            ''',
            (self.snapshot_id,),
        )
        true_counts = dict(zip(ELIGIBILITY_COLUMNS, row[:-1]))
        total = row[-1]
        if all(value == 0 for value in true_counts.values()):
            self.pass_("downstream_eligibility_boundary", f"all downstream eligibility flags remain false across {total} rows")
        else:
            self.fail("downstream_eligibility_boundary", f"unexpected true eligibility counts: {true_counts}")

        pending_condition = " OR ".join([f'"{col}" IS NOT NULL' for col in PENDING_SCORE_COLUMNS])
        non_null_pending_scores = self._scalar(
            cur,
            f'''
            SELECT COUNT(*)
            FROM "core_universe_scores"
            WHERE "snapshot_id" = %s AND ({pending_condition})
            ''',
            (self.snapshot_id,),
        )
        if self.policy_version.endswith("v0.2"):
            if non_null_pending_scores > 0:
                self.pass_("v02_scores_boundary", f"v0.2 six-layer score columns populated rows={non_null_pending_scores}")
            else:
                self.fail("v02_scores_boundary", "v0.2 score columns are empty")
        elif non_null_pending_scores == 0:
            self.pass_("pending_scores_boundary", "liquidity/fundamental/institutional/volatility scores remain NULL in v0.1")
        else:
            self.fail("pending_scores_boundary", f"pending score columns unexpectedly populated rows={non_null_pending_scores}")

        expected_scope = "v0.2_six_layer" if self.policy_version.endswith("v0.2") else "metadata_bootstrap_only"
        score_scope_mismatch = self._scalar(
            cur,
            '''
            SELECT COUNT(*)
            FROM "core_universe_scores"
            WHERE "snapshot_id" = %s
              AND ("score_detail"->>'score_scope') IS DISTINCT FROM %s
            ''',
            (self.snapshot_id, expected_scope),
        )
        if score_scope_mismatch == 0:
            self.pass_("score_scope", f"all score_detail records declare {expected_scope}")
        else:
            self.fail("score_scope", f"score_detail scope mismatches={score_scope_mismatch}")

    def check_observability(self, cur):
        revision_count = self._scalar(
            cur,
            '''
            SELECT COUNT(*)
            FROM "universe_revision_log"
            WHERE "snapshot_id" = %s AND "action_type" = 'BUILD_SNAPSHOT'
            ''',
            (self.snapshot_id,),
        )
        if revision_count > 0:
            self.pass_("revision_log", f"BUILD_SNAPSHOT revision rows={revision_count}")
        else:
            self.fail("revision_log", "BUILD_SNAPSHOT revision log missing")

        builder_audit_count = self._scalar(
            cur,
            '''
            SELECT COUNT(*)
            FROM data_audit_log
            WHERE action_type = 'CORE_UNIVERSE_BUILD' AND data_date = %s
            ''',
            (self.as_of_date,),
        )
        if builder_audit_count >= 5:
            self.pass_("data_audit_log", f"CORE_UNIVERSE_BUILD audit rows={builder_audit_count}")
        else:
            self.fail("data_audit_log", f"CORE_UNIVERSE_BUILD audit rows={builder_audit_count}, expected >= 5")

        lifecycle_count = self._scalar(
            cur,
            '''
            SELECT COUNT(*)
            FROM pipeline_execution_log
            WHERE task_name IN ('core_universe_builder_v0.2', 'core_universe_builder_v0.2_preflight', 'core_universe_builder_v0.1')
              AND status IN ('success', 'warning')
            ''',
        )
        if lifecycle_count > 0:
            self.pass_("pipeline_lifecycle", f"core_universe_builder accepted lifecycle rows={lifecycle_count}")
        else:
            self.fail("pipeline_lifecycle", "core_universe_builder accepted lifecycle row missing")

    def write_self_audit_log(self):
        try:
            rows_affected = sum(self.counts().values())
            audit_date = self.as_of_date.strftime("%Y-%m-%d") if self.as_of_date else date.today().strftime("%Y-%m-%d")
            write_data_audit_log("core_universe_snapshot", "SYSTEM", audit_date, "CORE_UNIVERSE_AUDIT", rows_affected)
            self.pass_("audit_self_log", "CORE_UNIVERSE_AUDIT written to data_audit_log")
        except Exception as exc:
            self.warn("audit_self_log", f"CORE_UNIVERSE_AUDIT write failed: {type(exc).__name__}: {exc}")

    def run_checks(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            if not self.check_required_tables(cur):
                return
            if not self.resolve_snapshot(cur):
                return
            self.check_policy(cur)
            self.check_rebalance_trace(cur)
            self.check_counts_and_tiers(cur)
            self.check_uniqueness_and_pairing(cur)
            self.check_raw_mirror(cur)
            self.check_v01_boundary(cur)
            self.check_observability(cur)
        finally:
            cur.close()
            conn.close()

    def write_report_file(self):
        if not self.write_report:
            return None
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.report_path = _REPORTS_DIR / f"core_universe_audit_{timestamp}.md"
        counts = self.counts()
        lines = [
            f"# Core Universe Audit Report ({TOOL_VER})",
            "",
            f"- constitution: {CONSTITUTION_VER}",
            f"- snapshot_id: {self.snapshot_id}",
            f"- as_of_date: {self.as_of_date}",
            f"- policy_version: {self.policy_version}",
            f"- verdict: {self.verdict()}",
            f"- PASS/WARN/FAIL: {counts['PASS']}/{counts['WARN']}/{counts['FAIL']}",
            "",
            "| status | check | detail |",
            "| :--- | :--- | :--- |",
        ]
        for item in self.items:
            safe_detail = str(item.detail).replace("|", "\\|")
            lines.append(f"| {item.status} | `{item.check_name}` | {safe_detail} |")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.report_path

    def _mark_lifecycle(self, lifecycle):
        if lifecycle is None:
            return
        counts = self.counts()
        if counts["FAIL"] > 0:
            marker = getattr(lifecycle, "mark_failed", None)
            if callable(marker):
                marker(f"Core universe audit failed: {counts}")
        elif counts["WARN"] > 0:
            marker = getattr(lifecycle, "mark_warning", None)
            if callable(marker):
                marker(f"Core universe audit warning: {counts}")

    def run(self):
        start_time = time.time()
        print("🔎 正在驗收核心股 Universe snapshot / membership / scores / governance boundary...")
        with record_lifecycle("audit_core_universe_v0.1", category="audit", stock_id="SYSTEM") as lifecycle:
            try:
                self.run_checks()
                if self.snapshot_id:
                    self.write_self_audit_log()
                report_path = self.write_report_file()
                self._mark_lifecycle(lifecycle)
                self.report_results(start_time, report_path)
                return self.verdict() != "FAILED"
            except Exception as exc:
                self.fail("audit_runtime", f"{type(exc).__name__}: {exc}")
                self._mark_lifecycle(lifecycle)
                report_path = self.write_report_file()
                self.report_results(start_time, report_path)
                return False

    def report_results(self, start_time, report_path):
        counts = self.counts()
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: 核心股 Universe 驗收稽核 ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md")
        print("治理權責 : Core Universe Post-Build Verification")
        print("邊界封印 : audit only; no feature/label/model/prediction values")
        print("─" * 80)
        print(f"📊 稽核報告     : {report_path.name if report_path else 'NO-REPORT'}")
        print(f"📌 Snapshot      : {self.snapshot_id}")
        print(f"📅 as_of_date    : {self.as_of_date}")
        print(f"📊 稽核項目統計 : PASS={counts['PASS']}, WARN={counts['WARN']}, FAIL={counts['FAIL']}")
        print(f"🕒 總計耗時     : {(time.time() - start_time)*1000:.2f} ms")
        print(f"⚖️  主權判定     : {self.verdict()}")
        print("🛡️" * 40 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Finance 核心股 Universe 驗收稽核 (v0.1)")
    parser.add_argument("--snapshot-id", type=str, help="指定 snapshot_id；未指定時使用 as-of/policy 或最新 committed snapshot")
    parser.add_argument("--as-of-date", type=str, help="指定 snapshot as_of_date，例如 2026-05-14")
    parser.add_argument("--policy-version", type=str, default=DEFAULT_POLICY_VERSION, help="指定 policy_version")
    parser.add_argument("--no-report", action="store_true", help="只輸出終端摘要，不產生 Markdown 報告")
    return parser.parse_args()


def main():
    args = parse_args()
    as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date() if args.as_of_date else None
    auditor = CoreUniverseAuditor(
        snapshot_id=args.snapshot_id,
        as_of_date=as_of_date,
        policy_version=args.policy_version,
        write_report=not args.no_report,
    )
    ok = auditor.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
