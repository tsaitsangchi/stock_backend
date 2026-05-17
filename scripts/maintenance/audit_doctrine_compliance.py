"""
audit_doctrine_compliance.py v0.1 (Quantum Finance §0 Supreme Doctrine Compliance Auditor)
================================================================================
最後更新日期: 2026-05-17
主權狀態: IMPLEMENTED (憲法 v6.0.0 §0 四大支柱 + §0.7 升版規則自動化執行)
最高原則: Doctrine Compliance Authority — §0 從文件 → 機器強制

依憲章 §0.7 升版規則：
  「v6.x.0 與 v7.0.0 之任何升版提案，皆必須附 §0 四大支柱之治理檢驗報告；
   若無法明示新條款對映至本章某一支柱（或同時對映至多支柱），該升版即不得進入正式 review。」

本工具實作上述規則之自動化執行，使 §0 從紙上原則升級為機器可強制之治權 gate。

v0.1 邊界:
1. 只讀憲章 §0 四大支柱定義 + DB committed snapshots + 程式碼 metadata；不寫入任何 governance table。
2. 對映檢驗為 PASS/WARN/FAIL；FAIL 即升版必須阻擋。
3. 可選 `--scan-module <path>` 對單一新模組做 §0 對映檢驗。
4. 可選 `--for-promotion <ver>` 啟用升版額外檢查（例如 v6.1.0 須含 §8 升強制契約之證據）。
5. 自動產生 `reports/doctrine_compliance_<timestamp>.md` 治理檢驗報告（除非 `--no-report`）。
================================================================================
"""
import argparse
import re
import sys
import time
from datetime import datetime
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
TOOL_VER = "v0.1"
PROJECT_ROOT = _SCRIPTS_DIR.parent
CHARTER_PATH = PROJECT_ROOT / "reports" / "系統架構大憲章_v6.0.0.md"

# §3.1/§3.2 charter-listed modules — used for "in scope" determination
CHARTER_MODULES = [
    "scripts/core/path_setup.py",
    "scripts/core/data_schema.py",
    "scripts/core/db_utils.py",
    "scripts/core/__init__.py",
    "scripts/core/finmind_client.py",
    "scripts/core/core_universe_schema.py",
    "scripts/core/core_universe_builder.py",
    "scripts/core/feature_store_schema.py",
    "scripts/core/feature_store_builder.py",
    "scripts/core/model_trainer.py",
    "scripts/core/prediction_engine.py",
    "scripts/ingestion/sovereign_sync_engine.py",
    "scripts/maintenance/audit_supply_chain.py",
    "scripts/maintenance/audit_core_universe.py",
    "scripts/maintenance/audit_leakage.py",
    "scripts/maintenance/audit_downstream_readiness.py",
]

# 四大支柱對映關鍵字（用於 --scan-module 之語意對映檢驗）
PILLAR_KEYWORDS = {
    "P1_first_principles": [
        "CoreScore", "LiquidityMass", "InstitutionalFlow", "VolatilityControl",
        "rank_ic", "robust_rank_ic", "feature_store", "feature_values",
        "log_return", "volatility", "ma_ratio", "max_drawdown",
        "foreign_net", "trust_net", "throttle",
    ],
    "P2_pareto_barbell": [
        "core_universe", "convex_universe", "research_universe", "quarantine_universe",
        "core_tier", "rebalance", "annual_rebalance", "special_rebalance",
        "core_universe_membership",
    ],
    "P3_kondratiev_2026": [
        "THEME_KEYWORDS", "theme_resonance", "macro_dff", "macro_vix",
        "macro_t10y2y", "macro_unrate", "industry_category", "MBNRIC",
        "半導體", "生技", "醫療", "綠能",
    ],
    "P4_observability_digital_twin": [
        "record_lifecycle", "write_data_audit_log", "pipeline_execution_log",
        "data_audit_log", "audit_leakage", "audit_downstream_readiness",
        "audit_supply_chain", "audit_core_universe", "as_of_strict",
        "feature_universe_lock", "model_id_governance", "no-hardcode", "dynamic verdict",
    ],
}

PILLAR_NAMES = {
    "P1_first_principles": "§0.1 第一性原則與市場物理學",
    "P2_pareto_barbell": "§0.2 八二法則與不對稱槓鈴",
    "P3_kondratiev_2026": "§0.3 康波週期與 2026 雙重共振",
    "P4_observability_digital_twin": "§0.4 可觀察性與數位孿生完整性",
}


class DoctrineAuditor:
    def __init__(self, scan_module=None, for_promotion=None, write_report=True):
        self.scan_module = Path(scan_module).resolve() if scan_module else None
        self.for_promotion = for_promotion
        self.write_report = write_report
        self.items = []  # list of (pillar, status, check, detail)
        self.charter_loaded = False
        self.charter_text = ""

    def add(self, pillar, status, check, detail):
        self.items.append((pillar, status, check, detail))
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
        pillar_short = pillar.split("_", 1)[0]
        print(f"{icon} [{status}] [{pillar_short}] {check}: {detail}")

    def load_charter(self):
        if not CHARTER_PATH.exists():
            self.add("P4_observability_digital_twin", "FAIL", "charter_present",
                     f"憲章不存在: {CHARTER_PATH}")
            return False
        self.charter_text = CHARTER_PATH.read_text(encoding="utf-8")
        if "### 0.1 第一支柱" not in self.charter_text:
            self.add("P4_observability_digital_twin", "FAIL", "charter_doctrine",
                     "憲章 §0.1 不存在；§0 系統核心思想未入憲")
            return False
        if "### 0.4 第四支柱" not in self.charter_text:
            self.add("P4_observability_digital_twin", "FAIL", "charter_doctrine",
                     "憲章 §0.4 不存在；四大支柱不完整")
            return False
        self.add("P4_observability_digital_twin", "PASS", "charter_doctrine",
                 "憲章 §0.1〜§0.4 四大支柱完整存在")
        self.charter_loaded = True
        return True

    # ── 四大支柱 DB / 程式碼對映檢驗 ──────────────────────────────────────────

    def audit_p1_first_principles(self, cur):
        """§0.1 物理量化（F=M×ΔlnP 重力井）對映檢驗"""
        # 1. CoreScore 六層存在於 policy weight_config
        cur.execute("""
            SELECT weight_config FROM core_universe_policy
            WHERE active = TRUE
            ORDER BY effective_from DESC NULLS LAST LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            self.add("P1_first_principles", "FAIL", "corescore_policy",
                     "無 active core_universe_policy；§0.1 物理量化未落地")
            return
        weights = row[0] or {}
        required_layers = ["data_quality_score", "liquidity_mass_score",
                           "fundamental_gravity_score", "theme_resonance_score",
                           "institutional_flow_score", "volatility_control_score"]
        missing = [k for k in required_layers if k not in weights]
        if missing:
            self.add("P1_first_principles", "FAIL", "corescore_six_layers",
                     f"policy weight_config 缺少六層 components: {missing}")
        else:
            self.add("P1_first_principles", "PASS", "corescore_six_layers",
                     "六層 CoreScore (DQ/LM/FG/TR/IF/VC) 完整存在於 active policy")

        # 2. feature_definition 含 liquidity / institutional / volatility 三群（重力井觀測載體）
        cur.execute("""
            SELECT feature_group, COUNT(*) FROM feature_definition
            WHERE feature_set_id IN (
                SELECT feature_set_id FROM feature_store_snapshot WHERE status='committed'
            )
            GROUP BY feature_group
        """)
        groups = {g: c for g, c in cur.fetchall()}
        required_groups = ["price", "liquidity", "institutional"]
        missing_g = [g for g in required_groups if g not in groups or groups[g] == 0]
        if missing_g:
            self.add("P1_first_principles", "WARN", "physics_features",
                     f"feature_definition 缺少物理量化群 (price/liquidity/institutional): {missing_g}")
        else:
            self.add("P1_first_principles", "PASS", "physics_features",
                     f"feature_definition 含完整物理量化群: {groups}")

    def audit_p2_pareto_barbell(self, cur):
        """§0.2 八二法則 + 不對稱槓鈴對映檢驗"""
        cur.execute("""
            SELECT m.core_tier, COUNT(*) FROM core_universe_membership m
            JOIN core_universe_snapshot s ON s.snapshot_id = m.snapshot_id
            WHERE s.status = 'committed'
              AND s.as_of_date = (SELECT MAX(as_of_date) FROM core_universe_snapshot WHERE status='committed')
            GROUP BY m.core_tier
        """)
        tiers = {t: c for t, c in cur.fetchall()}
        if not tiers:
            self.add("P2_pareto_barbell", "FAIL", "tier_structure",
                     "無 committed core_universe_snapshot；§0.2 槓鈴結構未實現")
            return
        core_n = tiers.get("core_universe", 0)
        convex_n = tiers.get("convex_universe", 0)
        research_n = tiers.get("research_universe", 0)
        quarantine_n = tiers.get("quarantine_universe", 0)
        right_tail = core_n + convex_n
        total = sum(tiers.values())
        if right_tail < 100 or right_tail > 200:
            self.add("P2_pareto_barbell", "WARN", "right_tail_size",
                     f"core+convex={right_tail} 偏離預期 (100-200)；§0.2 槓鈴攻擊端可能失衡")
        else:
            self.add("P2_pareto_barbell", "PASS", "right_tail_size",
                     f"core+convex={right_tail} (core {core_n} + convex {convex_n}) 符合槓鈴右尾規模")

        if quarantine_n == 0:
            self.add("P2_pareto_barbell", "WARN", "left_tail_isolation",
                     "quarantine_universe 為 0；§0.2 左尾結構性剔除未啟動")
        else:
            self.add("P2_pareto_barbell", "PASS", "left_tail_isolation",
                     f"quarantine_universe={quarantine_n} 已執行左尾剔除")

        if research_n < 1000:
            self.add("P2_pareto_barbell", "WARN", "middle_observation",
                     f"research_universe={research_n} 偏小；§0.2 中段觀測池可能不足")
        else:
            self.add("P2_pareto_barbell", "PASS", "middle_observation",
                     f"research_universe={research_n} 中段觀測池規模合理")

        # §6.8 annual rebalance guard 存在於 core_universe_builder.py
        builder = PROJECT_ROOT / "scripts/core/core_universe_builder.py"
        if builder.exists() and "_annual_rebalance_guard" in builder.read_text(encoding="utf-8"):
            self.add("P2_pareto_barbell", "PASS", "no_drift_guard",
                     "§6.8 annual rebalance guard 已實作於 core_universe_builder")
        else:
            self.add("P2_pareto_barbell", "FAIL", "no_drift_guard",
                     "core_universe_builder 缺少 _annual_rebalance_guard")

    def audit_p3_kondratiev_2026(self, cur):
        """§0.3 康波週期 + 6th wave MBNRIC 對映檢驗"""
        # 1. THEME_KEYWORDS 涵蓋第六波核心主題
        builder = PROJECT_ROOT / "scripts/core/core_universe_builder.py"
        if not builder.exists():
            self.add("P3_kondratiev_2026", "FAIL", "theme_keywords",
                     "core_universe_builder.py 不存在")
            return
        builder_text = builder.read_text(encoding="utf-8")
        required_themes = ["半導體", "生技", "醫療", "綠能"]
        missing_themes = [t for t in required_themes if t not in builder_text]
        if missing_themes:
            self.add("P3_kondratiev_2026", "FAIL", "theme_keywords",
                     f"THEME_KEYWORDS 缺少第六波 MBNRIC 必要主題: {missing_themes}")
        else:
            self.add("P3_kondratiev_2026", "PASS", "theme_keywords",
                     "THEME_KEYWORDS 涵蓋第六波 MBNRIC 核心主題（半導體/生技/醫療/綠能）")

        # 2. feature_definition 含 macro 群（DFF/VIX/T10Y2Y/UNRATE）
        cur.execute("""
            SELECT feature_name FROM feature_definition
            WHERE feature_group = 'macro'
              AND feature_set_id IN (
                  SELECT feature_set_id FROM feature_store_snapshot WHERE status='committed'
              )
        """)
        macro_features = {row[0] for row in cur.fetchall()}
        required_macro = ["macro_dff_level", "macro_vix_level", "macro_t10y2y_level", "macro_unrate_yoy"]
        missing_macro = [m for m in required_macro if m not in macro_features]
        if missing_macro:
            self.add("P3_kondratiev_2026", "WARN", "macro_features",
                     f"feature_definition macro 群缺少: {missing_macro}")
        else:
            self.add("P3_kondratiev_2026", "PASS", "macro_features",
                     f"feature_definition macro 群完整 ({sorted(macro_features)})")

        # 3. FredData 含四核心序列
        cur.execute("SELECT DISTINCT series_id FROM \"FredData\"")
        series = {row[0] for row in cur.fetchall()}
        required_series = {"DFF", "VIXCLS", "T10Y2Y", "UNRATE"}
        missing_series = required_series - series
        if missing_series:
            self.add("P3_kondratiev_2026", "FAIL", "fred_macro_data",
                     f"FredData 缺少宏觀序列: {missing_series}")
        else:
            self.add("P3_kondratiev_2026", "PASS", "fred_macro_data",
                     "FredData 完整含四核心序列 (DFF/VIXCLS/T10Y2Y/UNRATE)")

    def audit_p4_observability_digital_twin(self, cur):
        """§0.4 可觀察性 + 數位孿生對映檢驗"""
        # 1. pipeline_execution_log + data_audit_log 存在且活躍
        cur.execute("SELECT to_regclass('public.pipeline_execution_log')")
        if not cur.fetchone()[0]:
            self.add("P4_observability_digital_twin", "FAIL", "log_tables",
                     "pipeline_execution_log 表不存在")
            return
        cur.execute("SELECT to_regclass('public.data_audit_log')")
        if not cur.fetchone()[0]:
            self.add("P4_observability_digital_twin", "FAIL", "log_tables",
                     "data_audit_log 表不存在")
            return
        cur.execute("SELECT COUNT(*) FROM pipeline_execution_log")
        pipe_n = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM data_audit_log")
        audit_n = cur.fetchone()[0]
        if pipe_n == 0 or audit_n == 0:
            self.add("P4_observability_digital_twin", "WARN", "log_tables",
                     f"log 表存在但無紀錄: pipeline={pipe_n}, audit={audit_n}")
        else:
            self.add("P4_observability_digital_twin", "PASS", "log_tables",
                     f"混合日誌活躍: pipeline_execution_log={pipe_n}, data_audit_log={audit_n}")

        # 2. record_lifecycle 在所有 charter 模組中被使用
        missing_lifecycle = []
        for rel in CHARTER_MODULES:
            p = PROJECT_ROOT / rel
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            # path_setup / __init__ / db_utils / data_schema / finmind_client / feature_store_schema /
            # core_universe_schema / audit_supply_chain 為基礎/schema/audit 模組，不要求 record_lifecycle
            if rel in {"scripts/core/path_setup.py", "scripts/core/__init__.py",
                       "scripts/core/db_utils.py", "scripts/core/data_schema.py",
                       "scripts/core/finmind_client.py", "scripts/core/feature_store_schema.py",
                       "scripts/core/core_universe_schema.py",
                       "scripts/maintenance/audit_supply_chain.py",
                       "scripts/maintenance/audit_core_universe.py",
                       "scripts/maintenance/audit_downstream_readiness.py"}:
                continue
            if "record_lifecycle" not in text:
                missing_lifecycle.append(rel)
        if missing_lifecycle:
            self.add("P4_observability_digital_twin", "FAIL", "lifecycle_usage",
                     f"charter 模組未使用 record_lifecycle: {missing_lifecycle}")
        else:
            self.add("P4_observability_digital_twin", "PASS", "lifecycle_usage",
                     "所有運行型 charter 模組皆使用 record_lifecycle")

        # 3. §5.6.3 零硬編 PERFECT — 掃描所有 charter 模組
        bad_modules = []
        bad_pattern = re.compile(r'"PERFECT\s*\([^)]*v5\.', re.IGNORECASE)
        for rel in CHARTER_MODULES:
            p = PROJECT_ROOT / rel
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            if bad_pattern.search(text):
                bad_modules.append(rel)
        if bad_modules:
            self.add("P4_observability_digital_twin", "WARN", "no_hardcoded_perfect",
                     f"charter 模組仍有硬編 PERFECT (v5.x) 字串: {bad_modules}")
        else:
            self.add("P4_observability_digital_twin", "PASS", "no_hardcoded_perfect",
                     "charter 模組無硬編 PERFECT 違憲字串")

        # 4. §6.7 SQL 契約由 db_utils.get_core_stocks_from_db 集中提供（SSOT）
        db_utils = PROJECT_ROOT / "scripts/core/db_utils.py"
        if db_utils.exists() and "get_core_stocks_from_db" in db_utils.read_text(encoding="utf-8"):
            self.add("P4_observability_digital_twin", "PASS", "sql_ssot",
                     "§6.7 SQL 由 db_utils.get_core_stocks_from_db 集中提供")
        else:
            self.add("P4_observability_digital_twin", "FAIL", "sql_ssot",
                     "db_utils 缺少 get_core_stocks_from_db；§6.7 SSOT 漂移")

    # ── --scan-module 對映檢驗 ────────────────────────────────────────────────

    def scan_new_module(self):
        """對單一新模組做 §0 四大支柱語意對映檢驗"""
        if not self.scan_module.exists():
            self.add("P4_observability_digital_twin", "FAIL", "scan_module",
                     f"模組不存在: {self.scan_module}")
            return
        text = self.scan_module.read_text(encoding="utf-8")
        rel = str(self.scan_module.relative_to(PROJECT_ROOT))
        print(f"\n📄 [SCAN-MODULE] {rel}")
        matched_pillars = []
        for pillar, kws in PILLAR_KEYWORDS.items():
            hits = [kw for kw in kws if kw in text]
            if hits:
                matched_pillars.append(pillar)
                self.add(pillar, "PASS", "pillar_mapping",
                         f"{rel} 對映 {PILLAR_NAMES[pillar]} via keywords: {hits[:4]}{'...' if len(hits)>4 else ''}")
        if not matched_pillars:
            self.add("P4_observability_digital_twin", "FAIL", "no_pillar_mapping",
                     f"{rel} 無法對映 §0 任一支柱；違反 §0.7 升版規則「無法明示對映即不得進入正式 review」")

    # ── --for-promotion 升版額外檢查 ─────────────────────────────────────────

    def audit_for_promotion(self, cur):
        """v6.1.0 / v7.0.0 升版額外檢查"""
        target = self.for_promotion
        print(f"\n🚀 [FOR-PROMOTION] target={target}")
        if target.startswith("v6.1") or target.startswith("v7"):
            # 升 §8 強制契約：必須通過 audit_downstream_readiness=READY_FOR_V5_4_23 (or successor)
            cur.execute("""
                SELECT COUNT(*) FROM model_registry m
                WHERE m.status='committed'
                  AND (m.metrics->>'label_date_min')::date >=
                      ((SELECT as_of_date FROM core_universe_snapshot
                        WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1)
                       + m.label_horizon * INTERVAL '1 day')::date
            """)
            valid_models = cur.fetchone()[0]
            if valid_models == 0:
                self.add("P4_observability_digital_twin", "FAIL", "promotion_gate",
                         f"target={target} 升版 BLOCKED：無 production-current 模型符合 label_date >= as_of+horizon；命中 §8.8.9-D 條件 #1")
            else:
                self.add("P4_observability_digital_twin", "PASS", "promotion_gate",
                         f"target={target} 升版條件之 production-current 模型存在: {valid_models} 件")

    # ── verdict / report ────────────────────────────────────────────────────

    def counts(self):
        return {
            "PASS": sum(1 for _, s, _, _ in self.items if s == "PASS"),
            "WARN": sum(1 for _, s, _, _ in self.items if s == "WARN"),
            "FAIL": sum(1 for _, s, _, _ in self.items if s == "FAIL"),
        }

    def pillar_counts(self):
        result = {p: {"PASS": 0, "WARN": 0, "FAIL": 0} for p in PILLAR_NAMES}
        for p, s, _, _ in self.items:
            if p in result:
                result[p][s] += 1
        return result

    def write_markdown_report(self, counts, verdict, elapsed_ms):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = PROJECT_ROOT / "reports" / f"doctrine_compliance_{ts}.md"
        pillar_counts = self.pillar_counts()
        lines = [
            f"# §0 系統核心思想治權檢驗報告 (Doctrine Compliance Audit)\n",
            f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- tool: audit_doctrine_compliance.py {TOOL_VER}",
            f"- constitution: 系統架構大憲章_{CONSTITUTION_VER}.md §0",
            f"- verdict: **{verdict}**",
            f"- PASS/WARN/FAIL: {counts['PASS']}/{counts['WARN']}/{counts['FAIL']}",
            f"- elapsed: {elapsed_ms:.2f} ms",
            "",
        ]
        if self.scan_module:
            lines.append(f"- scan_module: `{self.scan_module.relative_to(PROJECT_ROOT)}`")
        if self.for_promotion:
            lines.append(f"- for_promotion: `{self.for_promotion}`")
        lines += ["", "## 四大支柱檢驗摘要\n", "| 支柱 | PASS | WARN | FAIL |", "|---|---:|---:|---:|"]
        for p, name in PILLAR_NAMES.items():
            c = pillar_counts[p]
            lines.append(f"| {name} | {c['PASS']} | {c['WARN']} | {c['FAIL']} |")
        lines += ["", "## 檢驗項目明細\n"]
        current_pillar = None
        for p, status, check, detail in self.items:
            if p != current_pillar:
                lines.append(f"\n### {PILLAR_NAMES.get(p, p)}\n")
                current_pillar = p
            lines.append(f"- **{status}** `{check}`: {detail}")
        lines += [
            "",
            "## §0.7 升版規則對照",
            "",
            "本工具實作憲章 §0.7「升版提案必須附 §0 四大支柱治理檢驗報告；無法明示對映即不得進入正式 review」。",
            "",
            "- `PERFECT` (FAIL=0, WARN=0)：升版可進入正式 review。",
            "- `WARNING` (FAIL=0, WARN>0)：升版可進入 review，但需明文解釋每項 WARN。",
            "- `FAILED` (FAIL>0)：升版必須阻擋；任一 FAIL 即違反 §0.7。",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def run(self):
        start = time.time()
        lifecycle_cm = record_lifecycle("audit_doctrine_compliance_v0.1",
                                       category="audit", stock_id="SYSTEM")
        lifecycle = lifecycle_cm.__enter__()
        try:
            print(f"🌌 §0 系統核心思想治權檢驗 (audit_doctrine_compliance {TOOL_VER} / 憲法 {CONSTITUTION_VER})")
            print("─" * 80)
            if not self.load_charter():
                return self._finalize(start, lifecycle)

            conn = get_db_connection()
            cur = conn.cursor()
            try:
                self.audit_p1_first_principles(cur)
                self.audit_p2_pareto_barbell(cur)
                self.audit_p3_kondratiev_2026(cur)
                self.audit_p4_observability_digital_twin(cur)
                if self.scan_module:
                    self.scan_new_module()
                if self.for_promotion:
                    self.audit_for_promotion(cur)
            finally:
                cur.close()
                conn.close()
            return self._finalize(start, lifecycle)
        finally:
            lifecycle_cm.__exit__(None, None, None)

    def _finalize(self, start, lifecycle):
        counts = self.counts()
        elapsed_ms = (time.time() - start) * 1000
        if counts["FAIL"] > 0:
            verdict = "FAILED"
            if lifecycle:
                lifecycle.mark_failed("doctrine compliance failed")
        elif counts["WARN"] > 0:
            verdict = "WARNING"
            if lifecycle:
                lifecycle.mark_warning("doctrine compliance warning")
        else:
            verdict = "PERFECT"

        report_path = None
        if self.write_report:
            report_path = self.write_markdown_report(counts, verdict, elapsed_ms)

        try:
            write_data_audit_log("audit_doctrine_compliance", "SYSTEM",
                                time.strftime("%Y-%m-%d"),
                                "DOCTRINE_COMPLIANCE_AUDIT", sum(counts.values()))
        except Exception as exc:
            print(f"⚠️  audit_log 寫入失敗: {type(exc).__name__}: {exc}")

        print("─" * 80)
        print(f"\n{'🛡️' * 40}")
        print(f"🌌 Quantum Finance §0 治權檢驗摘要 ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md §0 系統核心思想")
        print(f"📊 PASS/WARN/FAIL : {counts['PASS']}/{counts['WARN']}/{counts['FAIL']}")
        pc = self.pillar_counts()
        for p, name in PILLAR_NAMES.items():
            c = pc[p]
            print(f"   {name} : PASS={c['PASS']} WARN={c['WARN']} FAIL={c['FAIL']}")
        if report_path:
            print(f"📄 報告 : {report_path.relative_to(PROJECT_ROOT)}")
        else:
            print("📄 報告 : NO-REPORT")
        print(f"🕒 總計耗時 : {elapsed_ms:.2f} ms")
        print(f"⚖️  治權判定 : {verdict}")
        print("🛡️" * 40 + "\n")
        return counts["FAIL"] == 0


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Finance §0 Supreme Doctrine Compliance Auditor (v0.1)")
    parser.add_argument("--scan-module", type=str, default=None,
                        help="對單一新模組做 §0 四大支柱語意對映檢驗（路徑）")
    parser.add_argument("--for-promotion", type=str, default=None,
                        help="升版額外檢查；指定 target version 如 v6.1.0 / v7.0.0")
    parser.add_argument("--no-report", action="store_true",
                        help="不產生 markdown 報告檔案")
    args = parser.parse_args()

    auditor = DoctrineAuditor(
        scan_module=args.scan_module,
        for_promotion=args.for_promotion,
        write_report=(not args.no_report),
    )
    ok = auditor.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
