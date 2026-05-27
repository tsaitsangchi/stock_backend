"""
audit_core_universe.py v0.2 (Quantum Finance Core Universe Audit Authority)
================================================================================
最後更新日期: 2026-05-25
主權狀態: IMPLEMENTED (憲法 v6.1.0 對齊 + v0.7.1 builder 配套(§14.7-BH RMS 公式追溯)
        + v0.6 policy 識別擴張 + FG 11 sub-score / IF 12 sub-score / VC RMS 凸性對齊 score_detail 驗收)
最高原則: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則) — Core Universe Post-Build Verification

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Core Universe Audit Authority]: 對齊憲章 §6.7 / §6.8 / §8.8.6，只驗收
   `core_universe_builder.py` 產物，**不重算**核心股名單；本工具不是核心股
   選擇器，是核心股結果之 post-build 驗收。
2. [Zero Hardcoded Verdict] (§5.6.3 動態判定): 主權狀態 PASS/WARN/FAIL 由
   `self.verdict()` 動態計算（FAIL > 0 → FAILED；WARN > 0 → WARNING；else PERFECT），
   不硬寫；任何「score_scope 未識別」應 fail 而非 silent pass。
3. [Sovereignty Declaration] (§3.1/§3.2/§3.2A 治權位階): 本工具屬 §6.7 / §6.8
   核心股治權之 post-build 驗收層；**不**處理 §8.5 anti-leakage（feature_store 層責）；
   **不**涉 §0.1-A / §0.2-A / §0.3-A 五套禁令（不重算 CoreScore、不外推 K-wave、
   不寫 alpha 固定值）；**不**分層至 T1/T2/T3（資料驗收屬 T0 治權確認）；
   policy_version 識別擴張僅作 score_scope 字串一致性驗收，**不**重算 sub-score 數值。
4. [Consistency Coverage]: 驗收 policy、snapshot、membership、scores、
   revision log 之一致性；驗收 raw 欄位鏡像（§1 第 5 條 Derived Schema
   欄位繼承）與 v0.1 downstream boundary；v0.2 新增 score_detail 鍵存在性驗收
   （FG 11 鍵 / IF 12 鍵 / VC 4 鍵；per policy version）。
5. [Annual Rebalance Guard Verification]: 驗收年度重選 / special restore
   之 review_cycle、snapshot notes、revision log 留痕；§8.8.6 第 2 條
   same-day reason duplication INFO 偵測（v6.0.0 補登 41 項檢查）。
6. [Boundary Integrity]: 不保存 feature values、labels、model outputs、
   prediction signals；§8 下游治理不在本工具範圍。
7. [Hybrid Observability]: 維運觸發 `record_lifecycle` 與 `write_data_audit_log`；
   主權狀態動態計算（§5.6.3），FAIL 即 exit 1；lifecycle task_name 對齊
   `core_universe_builder_v0.2`（builder 各小版維持同一 task_name）。
8. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準；
   v0.1 條目（v6.0.0 patch line）保留為歷史記述，不更動（§0.0-I.7 / L26）。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [Step 4C：年度重選後驗收]** | `$ python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14` | audit_core_universe v0.2 |
| **2. [DB 全重建驗收]** | `$ python scripts/maintenance/audit_core_universe.py --as-of-date <YYYY-MM-DD>` | audit_core_universe v0.2 |
| **3. [special restore 後驗收]** | `$ python scripts/maintenance/audit_core_universe.py --as-of-date <restore-date>` | audit_core_universe v0.2 |
| **4. [v0.6 policy 新版驗收]** | `$ python scripts/maintenance/audit_core_universe.py --as-of-date <YYYY-MM-DD> --policy-version core_universe_policy_v0.6` | audit_core_universe v0.2 + builder v0.7 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **no-report** | `--no-report` | 略過 `reports/core_universe_audit_*.md` 產出 |
| **strict** | `--strict` | special restore notes / reason 字串嚴格檢驗，缺漏即 FAIL |
| **legacy-fallback** | `--allow-latest-registry-fallback` | bootstrap 期間允許 candidate fallback 為 latest registry |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.2** | 2026-05-25 | Codex | **v6.1.0-patch 第十二輪程式：audit 工具配套 v0.7 builder + v0.4/v0.5/v0.6 policy 識別擴張**。對映 §14.7-BC / §14.7-BE / §14.7-BF / §14.7-BG 五輪 builder 升版（v0.3 → v0.7）後 audit 工具長期未追上之治權缺口（builder v0.5 commit `c4dc523` 已標註「audit_core_universe 配套需求：audit 工具需加 `core_universe_policy_v0.4` 識別（另案升版）」），本版一次補完三輪。**補正內容**：(I) `CONSTITUTION_VER v6.0.0 → v6.1.0`、`TOOL_VER v0.1 → v0.2`、`DEFAULT_POLICY_VERSION v0.2 → v0.6`；(II) 新增 `POLICY_SCORE_SCOPE_MAP`（policy_version → 預期 score_scope 對映表，五項：v0.2/v0.3/v0.4/v0.5/v0.6 → "v0.2_six_layer"/"v0.3_six_layer_extended"/"v0.5_eleven_sub_score"/"v0.6_F_proxy_augmented"/"v0.7_VC_convexity_aligned"）；(III) 新增 `EXPECTED_SCORE_DETAIL_KEYS`（per policy 期望 score_detail 鍵集合：v0.4 +6 FG / v0.5 +11 FG +12 IF / v0.6 +11 FG +12 IF +4 VC）；(IV) `check_policy()` 加 v0.4/v0.5/v0.6 三條 score_config 分支；(V) `check_v01_boundary()` 之 `expected_scope` 改 dict lookup，不再 endswith hardcoded；(VI) 新增 `check_score_detail_keys()` 方法驗收 score_detail 鍵集合（FAIL 若缺鍵 > 20%）；(VII) `check_observability()` task_name 列表加 v0.3/v0.4/v0.5/v0.5.1/v0.6/v0.7 builder（雖 lifecycle name 維持 v0.2，但前向相容）；(VIII) `run()` lifecycle name `audit_core_universe_v0.1 → v0.2`；(IX) 標頭 8-項 docstring compliance 重寫（per CLAUDE.md §四 #4）含 [Zero Hardcoded Verdict] + [Sovereignty Declaration] 治權自我宣告。**對既有 snapshot 影響**：零；既有 v0.2 snapshot audit 結果完全 backward-compatible（仍 41/0/0 PERFECT）；新 v0.6 snapshot 經本工具驗收將回報新項目通過。**邏輯動量**：不改任何 CoreScore 重算、不改 §6.7 SSOT 150、不改 §6.4 公式、不改 raw DDL、不改 CLI 參數結構（只 default 值升）、不改 5 張治理表寫入順序、不改 annual_rebalance_guard。**Cross-Reference 精確行號**：constants 區塊 L80-82；POLICY_SCORE_SCOPE_MAP 定義 L114-120；EXPECTED_SCORE_DETAIL_KEYS 定義 L124-158；check_policy 政策版本分支 L329-333；check_v01_boundary 之 expected_scope dict lookup L643；check_score_detail_keys 新方法 L659；check_observability 之 task_name 列表 L733。同步配套：無需新 charter 子節（屬工具配套）；無需新設計研究（屬機械式擴張）。 | **ACTIVE** |
| v0.1 | 2026-05-14 | Codex | 首版：依憲章 §6.7 / §6.8 / §8.8.6 落地，驗收 policy/snapshot/membership/scores/revision log；2026-05-17 補入 `check_same_day_reason_duplication()`（§8.8.6 第 2 條）；2026-05-18 v6.0.0-patch 補入 §6.8 annual guard 衍生檢查，總檢驗項由 36 → 40 → 41。 | SUPERSEDED |
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


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.3"  # §14.7-BT Phase D-2 配套:audit 工具加 v0.8_dynamic policy 識別
DEFAULT_POLICY_VERSION = "core_universe_policy_v0.7"  # production-current default(legacy);v0.8_dynamic 為 opt-in
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

# v0.2: per policy_version → 預期 score_scope 字串對映表（builder 寫入 score_detail.score_scope；
# 對映 builder 各版本記述：v0.2/v0.3/v0.5/v0.6/v0.7 score_scope 字串）。
POLICY_SCORE_SCOPE_MAP = {
    "core_universe_policy_v0.2": "v0.2_six_layer",
    "core_universe_policy_v0.3": "v0.3_six_layer_extended",
    "core_universe_policy_v0.4": "v0.5_eleven_sub_score",
    "core_universe_policy_v0.5": "v0.6_F_proxy_augmented",
    "core_universe_policy_v0.6": "v0.7.1_VC_convexity_aligned_rms",
    "core_universe_policy_v0.7": "v0.8_roe_unlocked_via_balance_sheet",
    "core_universe_policy_v0.8_dynamic": "v0.10_dynamic_universe_via_top_pct_composite_corescore",  # §14.7-BT Phase C+D
    "core_universe_policy_v0.10_pure_doctrine": "v0.10_pure_doctrine_no_hardcode",  # §14.7-BW
    "core_universe_policy_v0.10_pure_doctrine_weekly": "v0.10_pure_doctrine_weekly",  # §14.7-BX
    "core_universe_policy_v0.11_feature_completeness_gate": "v0.11_feature_completeness_gate",  # §14.7-CB
    "core_universe_policy_v0.12_raw_data_completeness_gate": "v0.12_raw_data_completeness_gate",  # §14.7-CD
    "core_universe_policy_v0.13_doctrine_native_gate": "v0.13_doctrine_native_gate_§CG",  # §14.7-CG native integration
}

# v0.2: per policy_version → score_detail 期望鍵集合（驗收 builder 對應版本之 sub-score 透明寫入）。
# 不驗值（避免重算 CoreScore，違反 [Sovereignty Declaration]）；只驗鍵存在性。
EXPECTED_SCORE_DETAIL_KEYS = {
    "core_universe_policy_v0.2": set(),  # baseline，六層 CoreScore 直接寫入欄位，無 sub-score detail
    "core_universe_policy_v0.3": {
        "fg_gross_margin", "fg_roe",
    },
    "core_universe_policy_v0.4": {
        "fg_gross_margin", "fg_roe",
        "fg_per", "fg_pbr", "fg_div_yield", "fg_div_count_5y",
        "fg_op_margin", "fg_pretax_margin", "fg_continuing_op_ratio", "fg_attributable_ratio",
        "fg_part_dist_5y_avg",
    },
    "core_universe_policy_v0.5": {
        "fg_gross_margin", "fg_roe",
        "fg_per", "fg_pbr", "fg_div_yield", "fg_div_count_5y",
        "fg_op_margin", "fg_pretax_margin", "fg_continuing_op_ratio", "fg_attributable_ratio",
        "fg_part_dist_5y_avg",
        "if_dealer_self_net", "if_dealer_hedge_net",
        "if_margin_bal_60d", "if_short_bal_60d", "if_short_margin_ratio",
        "if_margin_trend_60d", "if_margin_repay_trend",
        "if_foreign_ratio", "if_foreign_remain_ratio", "if_foreign_upper_limit",
        "if_num_shares_issued", "if_foreign_ratio_60d_change",
    },
    "core_universe_policy_v0.6": {
        "fg_gross_margin", "fg_roe",
        "fg_per", "fg_pbr", "fg_div_yield", "fg_div_count_5y",
        "fg_op_margin", "fg_pretax_margin", "fg_continuing_op_ratio", "fg_attributable_ratio",
        "fg_part_dist_5y_avg",
        "if_dealer_self_net", "if_dealer_hedge_net",
        "if_margin_bal_60d", "if_short_bal_60d", "if_short_margin_ratio",
        "if_margin_trend_60d", "if_margin_repay_trend",
        "if_foreign_ratio", "if_foreign_remain_ratio", "if_foreign_upper_limit",
        "if_num_shares_issued", "if_foreign_ratio_60d_change",
        "vc_convexity_60d", "vc_upside_rms_60d", "vc_downside_rms_60d", "vc_cc_sigma_60d",
    },
    "core_universe_policy_v0.7": {
        # v0.6 全部 31 keys + v0.8 新增 2 keys(fg_equity / fg_ni_4q_sum)
        "fg_gross_margin", "fg_roe",
        "fg_per", "fg_pbr", "fg_div_yield", "fg_div_count_5y",
        "fg_op_margin", "fg_pretax_margin", "fg_continuing_op_ratio", "fg_attributable_ratio",
        "fg_part_dist_5y_avg",
        "if_dealer_self_net", "if_dealer_hedge_net",
        "if_margin_bal_60d", "if_short_bal_60d", "if_short_margin_ratio",
        "if_margin_trend_60d", "if_margin_repay_trend",
        "if_foreign_ratio", "if_foreign_remain_ratio", "if_foreign_upper_limit",
        "if_num_shares_issued", "if_foreign_ratio_60d_change",
        "vc_convexity_60d", "vc_upside_rms_60d", "vc_downside_rms_60d", "vc_cc_sigma_60d",
        # v0.8 §14.7-BI ROE 解鎖 transparency keys
        "fg_equity", "fg_ni_4q_sum",
    },
    # §14.7-BT Phase D-2:v0.8_dynamic policy(同 v0.7 score keys;只 selection 邏輯改 dynamic)
    "core_universe_policy_v0.8_dynamic": {
        # 同 v0.7 之 33 keys(score_detail keys 不變;dynamic 只改 selection;非 score 公式)
        "fg_gross_margin", "fg_roe",
        "fg_per", "fg_pbr", "fg_div_yield", "fg_div_count_5y",
        "fg_op_margin", "fg_pretax_margin", "fg_continuing_op_ratio", "fg_attributable_ratio",
        "fg_part_dist_5y_avg",
        "if_dealer_self_net", "if_dealer_hedge_net",
        "if_margin_bal_60d", "if_short_bal_60d", "if_short_margin_ratio",
        "if_margin_trend_60d", "if_margin_repay_trend",
        "if_foreign_ratio", "if_foreign_remain_ratio", "if_foreign_upper_limit",
        "if_num_shares_issued", "if_foreign_ratio_60d_change",
        "vc_convexity_60d", "vc_upside_rms_60d", "vc_downside_rms_60d", "vc_cc_sigma_60d",
        "fg_equity", "fg_ni_4q_sum",
    },
    # §14.7-BW v0.10 pure doctrine(廢棄 CoreScore;只 doctrine gate;無 score_detail keys 強制需求)
    "core_universe_policy_v0.10_pure_doctrine": set(),
    "core_universe_policy_v0.10_pure_doctrine_weekly": set(),
    # §14.7-CB v0.11 feature completeness gate(post-process layer;不寫 scores)
    "core_universe_policy_v0.11_feature_completeness_gate": set(),
    # §14.7-CD v0.12 raw data completeness gate(post-process layer;不寫 scores)
    "core_universe_policy_v0.12_raw_data_completeness_gate": set(),
    # §14.7-CG v0.13 native gate integration(integrated builder;§14.7-CF Invariant 1 pure;不寫 scores per §14.7-BW)
    "core_universe_policy_v0.13_doctrine_native_gate": set(),
}


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
        elif self.policy_version.endswith("v0.3"):
            self.pass_("policy_score_config", "v0.3 policy uses six-layer CoreScore weights with FG extended sub-scores (gross_margin + ROE)")
        elif self.policy_version.endswith("v0.4"):
            self.pass_("policy_score_config", "v0.4 policy uses six-layer CoreScore weights with FG 11 sub-scores (V augmentation Phase C/D + FinStmt + ParticipateDistribution SELECT-only animation)")
        elif self.policy_version.endswith("v0.5"):
            self.pass_("policy_score_config", "v0.5 policy uses six-layer CoreScore weights with FG 11 sub-scores + IF 12 sub-scores (F proxy Phase F.1-F.3: Dealer directional / Margin 4 / Shareholding 5)")
        elif self.policy_version.endswith("v0.6"):
            self.pass_("policy_score_config", "v0.6 policy uses six-layer CoreScore weights with FG 11 sub-scores + IF 12 sub-scores + VC convexity-aware RMS (upside_RMS − downside_RMS raw-first path;§9.10 正式條文 對齊 §9.9 強制契約)")
        elif self.policy_version.endswith("v0.7"):
            self.pass_("policy_score_config", "v0.7 policy uses six-layer CoreScore weights with FG 12 sub-scores (含 ROE 解鎖 from TaiwanStockBalanceSheet) + IF 12 sub-scores + VC convexity-aware RMS;§14.7-BI 資料現實裁決首例「解鎖成功」")
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

    def check_same_day_reason_duplication(self, cur):
        """§8.8.6 第 2 條：同一日同一 special override reason 重複觸發須回報 INFO（v6.0.0 不強制 reject）。"""
        rows = self._rows(
            cur,
            '''
            SELECT
                DATE(rl."revision_time") AS rev_date,
                COALESCE(rl."detail"->>'special_rebalance_reason', '') AS reason,
                COUNT(*) AS hit_count,
                ARRAY_AGG(rl."snapshot_id" ORDER BY rl."revision_time") AS snapshot_ids
            FROM "universe_revision_log" rl
            WHERE rl."action_type" = 'BUILD_SNAPSHOT'
              AND rl."detail"->>'rebalance_mode' = 'special'
              AND COALESCE(rl."detail"->>'special_rebalance_reason', '') <> ''
            GROUP BY rev_date, reason
            HAVING COUNT(*) > 1
            ORDER BY rev_date DESC, hit_count DESC
            '''
        )
        if not rows:
            self.pass_("same_day_reason_dedup", "§8.8.6 第 2 條：無同日重複 special override reason")
            return
        for rev_date, reason, hit_count, snapshot_ids in rows:
            short_reason = (reason[:40] + "...") if len(reason) > 40 else reason
            self.warn(
                "same_day_reason_dedup",
                f"§8.8.6 INFO：{rev_date} 日 reason='{short_reason}' 重複 {hit_count} 次；snapshot_ids={list(snapshot_ids)}；建議使用不同 <stage> 後綴避免 audit 噪訊",
            )

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
        # §14.7-BW pure doctrine + 2026-05-27 directive:取消 hardcoded 120/30 之 v0.1 limit
        # N 為 doctrine 結果,任何 N > 0 皆 PASS(was: core <= 120, convex <= 30)
        if self.snapshot["core_count"] > 0:
            self.pass_("core_size", f"core_count={self.snapshot['core_count']} — dynamic per §14.7-BW")
        else:
            self.fail("core_size", f"core_count=0;無 doctrine-pass stock")
        # convex_count = 0 為 v0.10 pure doctrine 之正常情境(tier 概念 v0.10 不適用)
        self.pass_("convex_size", f"convex_count={self.snapshot['convex_count']} — dynamic per §14.7-BW (0=v0.10 normal)")

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
        if self.policy_version.endswith(("v0.2", "v0.3", "v0.4", "v0.5", "v0.6", "v0.7")):
            if non_null_pending_scores > 0:
                self.pass_("v02_scores_boundary", f"six-layer score columns populated rows={non_null_pending_scores} (policy={self.policy_version})")
            else:
                self.fail("v02_scores_boundary", f"six-layer score columns are empty (policy={self.policy_version})")
        elif non_null_pending_scores == 0:
            self.pass_("pending_scores_boundary", "liquidity/fundamental/institutional/volatility scores remain NULL in v0.1")
        else:
            self.fail("pending_scores_boundary", f"pending score columns unexpectedly populated rows={non_null_pending_scores}")

        expected_scope = POLICY_SCORE_SCOPE_MAP.get(self.policy_version, "metadata_bootstrap_only")
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
            self.fail("score_scope", f"score_detail scope mismatches={score_scope_mismatch} (expected {expected_scope} for policy={self.policy_version})")

    def check_score_detail_keys(self, cur):
        """v0.2 新增：驗收 score_detail 中是否含 builder 對應版本之 sub-score 透明寫入鍵。

        - 不驗值（避免重算 CoreScore；違反 [Sovereignty Declaration]）。
        - 對 baseline policy（v0.2）跳過（無 sub-score detail）。
        - 對 v0.3+ policy 抽樣 1 row score_detail，比對 EXPECTED_SCORE_DETAIL_KEYS。
        - 缺鍵 ≤ 20% → WARN；> 20% → FAIL；全鍵齊備 → PASS。
        """
        expected_keys = EXPECTED_SCORE_DETAIL_KEYS.get(self.policy_version)
        if expected_keys is None:
            self.warn("score_detail_keys", f"policy={self.policy_version} not in EXPECTED_SCORE_DETAIL_KEYS map (skipped)")
            return
        if not expected_keys:
            self.pass_("score_detail_keys", f"policy={self.policy_version} has no sub-score detail expectation (baseline)")
            return
        row = self._row(
            cur,
            '''
            SELECT "score_detail"
            FROM "core_universe_scores"
            WHERE "snapshot_id" = %s
            LIMIT 1
            ''',
            (self.snapshot_id,),
        )
        if not row or row[0] is None:
            self.fail("score_detail_keys", f"no score_detail row found for snapshot={self.snapshot_id}")
            return
        actual_keys = set(row[0].keys()) if isinstance(row[0], dict) else set()
        missing = expected_keys - actual_keys
        if not missing:
            self.pass_("score_detail_keys", f"all {len(expected_keys)} expected score_detail sub-score keys present (policy={self.policy_version})")
            return
        miss_ratio = len(missing) / len(expected_keys)
        sample_missing = sorted(missing)[:5]
        if miss_ratio > 0.20:
            self.fail("score_detail_keys", f"missing {len(missing)}/{len(expected_keys)} ({miss_ratio:.0%}) sub-score keys; sample={sample_missing}")
        else:
            self.warn("score_detail_keys", f"missing {len(missing)}/{len(expected_keys)} ({miss_ratio:.0%}) sub-score keys; sample={sample_missing}")

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
            WHERE task_name IN (
                'core_universe_builder_v0.1',
                'core_universe_builder_v0.2',
                'core_universe_builder_v0.2_preflight',
                'core_universe_builder_v0.3',
                'core_universe_builder_v0.4',
                'core_universe_builder_v0.5',
                'core_universe_builder_v0.5.1',
                'core_universe_builder_v0.6',
                'core_universe_builder_v0.7'
            )
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
            self.check_same_day_reason_duplication(cur)
            self.check_counts_and_tiers(cur)
            self.check_uniqueness_and_pairing(cur)
            self.check_raw_mirror(cur)
            self.check_v01_boundary(cur)
            self.check_score_detail_keys(cur)
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
        with record_lifecycle("audit_core_universe_v0.2", category="audit", stock_id="SYSTEM") as lifecycle:
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
    parser = argparse.ArgumentParser(description="Quantum Finance 核心股 Universe 驗收稽核 (v0.2)")
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
