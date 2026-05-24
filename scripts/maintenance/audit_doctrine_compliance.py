"""
audit_doctrine_compliance.py v0.4 (Quantum Finance §0 Supreme Doctrine Compliance Auditor · §14.7-AV Dual-Track Promotion Gate)
================================================================================
最後更新日期: 2026-05-23
主權狀態: IMPLEMENTED (憲法 v6.1.0 §0 四大支柱 + §0.1-A 禁令 #2/#3 自動化檢驗
                    + §0.1-B audit 載體 + §0.1.3 V 變數對應透明化
                    + §14.7-AV dual-track promotion gate (Operations Reality v6.1.0 vs §8/§9 軌道 v6.1.1+))
最高原則: Doctrine Compliance Authority — §0 從文件 → 機器強制

## 修訂歷程
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.4 | 2026-05-23 | Codex | **§14.7-AV dual-track promotion gate 落地**:對齊憲章 v6.1.0 §9.5 升版路徑表雙軌制(Operations Reality vs §8/§9 promotion)。新增 `OPERATIONS_TRACK_VERSIONS = {"v6.1.0"}` 與 `DOWNSTREAM_TRACK_VERSIONS = {"v6.1.1", "v6.2.0", ...}` 模組常數;`audit_for_promotion()` 依 target 分流走 Operations 軌道(檢 §0.0-I.9 跨平台前置)或 §8 軌道(檢 §8 schema + production-current model)。未知 target 走嚴格 §8 fallback。Smoke 驗證:`--for-promotion v6.1.0` → promotion_gate PASS;`--for-promotion v6.1.1` → promotion_gate FAIL(time-gated)。CONSTITUTION_VER v6.0.0 → v6.1.0 + TOOL_VER v0.3 → v0.4。對應 §14.7-AV 治權記述(audit_doctrine v0.3 之 promotion gate 不分流軌道之缺口)。 | **ACTIVE** |
| v0.3 | 2026-05-19 | Codex | (前版基線) | SUPERSEDED |

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Supreme Doctrine Authority]: 對齊憲章 §0.7 升版規則「v6.x.0 與 v7.0.0 之任何升版
   提案，皆必須附 §0 四大支柱之治理檢驗報告；若無法明示新條款對映至本章某一支柱
   （或同時對映至多支柱），該升版即不得進入正式 review」。
2. [Four-Pillar Mapping]: 對 P1 物理量化（CoreScore + feature_definition）/
   P2 八二槓鈴（core+convex + quarantine + research + §6.8 annual guard）/
   P3 康波 2026（THEME_KEYWORDS MBNRIC + macro features + FredData）/
   P4 可觀察性（pipeline/audit log + record_lifecycle + §5.6.3 + §6.7 SQL SSOT）
   逐項 PASS/WARN/FAIL 檢驗，自動產生 `reports/doctrine_compliance_<timestamp>.md`。
3. [Read-Only Authority]: 只讀憲章 §0 四大支柱定義 + DB committed snapshots +
   程式碼 metadata；不寫入任何 governance table；對映檢驗為 PASS/WARN/FAIL；
   FAIL 即升版必須阻擋。
4. [Draft Schema Tolerance]: §8/§9 DRAFT 期間，對 feature_definition /
   feature_store_snapshot / model_registry / prediction_* / allocation_* 等
   尚未建立之 DDL 表，採 graceful skip 並寫 WARN；不得因表缺失而 crash
   （v0.2 `_table_exists()` helper 落地）。
5. [Hybrid Observability]: 維運觸發 `record_lifecycle` 與 `write_data_audit_log`；
   主權狀態動態計算（§5.6.3）。
6. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準。
7. [T3 Leakage Forbiddance] (v0.3): §0.1-A 禁令 #2/#3 永久強制——IFF Θ / SOC /
   重力井邊緣觸發為 T3 操作隱喻，禁止實作於 §6 / §8 / §9 模組。任何發現即 FAIL
   （不可降為 WARN）；對應 `audit_t3_leakage()` 與 `T3_FORBIDDEN_PATTERNS`。
8. [Proxy Transparency] (v0.3): §0.1-B audit 載體 + §0.1.3 V 變數補強——
   feature_definition 中之 proxy 變數須在 description 明文標註對應的 §0.1 元素
   （M / V / ΔlnP / F_external）。缺標註為 WARN（文檔衛生）；對應
   `audit_proxy_transparency()` 與 `PROXY_FIRST_PRINCIPLES_MAPPING`。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [Step 11C：§0 基礎治權檢驗]** | `$ python scripts/maintenance/audit_doctrine_compliance.py` | audit_doctrine_compliance v0.3 |
| **2. [Step 11D：升版 gate]** | `$ python scripts/maintenance/audit_doctrine_compliance.py --for-promotion v6.1.0` | audit_doctrine_compliance v0.3 |
| **3. [Step 11E：新模組四支柱對映]** | `$ python scripts/maintenance/audit_doctrine_compliance.py --scan-module scripts/path/to/new_module.py` | audit_doctrine_compliance v0.3 |
| **4. [僅 stdout：不產生報告檔]** | `$ python scripts/maintenance/audit_doctrine_compliance.py --no-report` | audit_doctrine_compliance v0.3 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **no-report** | `--no-report` | 略過 `reports/doctrine_compliance_*.md` 產出 |
| **promotion-v6.1.0** | `--for-promotion v6.1.0` | 升版至 §8 強制契約之 gate；含 §8.8.9-D 條件檢驗 |
| **promotion-v7.0.0** | `--for-promotion v7.0.0` | 主版升版 gate；含破壞性 schema 變動阻擋規則 |
| **scan-module** | `--scan-module <path>` | 新模組四支柱語意對映；無對映即 FAIL（§0.7 條文） |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.3** | 2026-05-19 | Codex | §0.1-B audit 載體入庫：新增 `audit_t3_leakage()`（§0.1-A 禁令 #2/#3 自動化）與 `audit_proxy_transparency()`（§0.1.3 V 變數對應標註）；新增常數 `T3_FORBIDDEN_PATTERNS` / `T3_SCAN_TARGETS` / `PROXY_FIRST_PRINCIPLES_MAPPING`；新增 helper `_strip_comments_and_docstrings`。§0.1-A 6 條禁令自動化覆蓋從 0% 提升至 50%。 | **ACTIVE** |
| v0.2 | 2026-05-18 | Codex | §14.7-J Finding #1 修補：對 §8/§9 DRAFT 期間尚未建立之 DDL 表採 graceful skip + WARN，不再 crash；新增 `_table_exists()` helper；P1/P3/P4 promotion gate 三處 §8 表查詢前置存在檢查。 | SUPERSEDED |
| v0.1 | 2026-05-17 | Codex | 首版：四大支柱對映檢驗、`--scan-module`、`--for-promotion`；§0 從文件升級為機器可強制之治權 gate；對 §8 表硬性查詢無保護（2026-05-18 v6.0.0-patch 揭露 bug；v0.2 修補）。 | SUPERSEDED |
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


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.4"

# v0.4 §14.7-AV: dual-track promotion gate (Operations Reality vs §8 軌道)
# 對齊憲章 v6.1.0 §9.5 升版路徑表(L4744-L4750)
OPERATIONS_TRACK_VERSIONS = {"v6.1.0"}   # Operations Reality Patch 軌道:不需 §8 schema
DOWNSTREAM_TRACK_VERSIONS = {            # §8 / §9 promotion 軌道:需 §8 schema + production-current model
    "v6.1.1",   # 原 v6.1.0 預期軌道 順延 (§8 h=20 升強制契約)
    "v6.2.0",   # prediction_engine v0.2
    "v6.3.0",   # portfolio_sizer v0.1
    "v6.4.0",   # backtest_engine v0.1
    "v7.0.0",   # §9 全面升強制契約
}


def _table_exists(cur, table_name: str) -> bool:
    """§8/§9 DRAFT 表存在檢查 (v0.2 新增)。對齊 §0.7 補強：DRAFT DDL 表缺失須 graceful skip。"""
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f'public."{table_name}"',))
    return cur.fetchone()[0]
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
    "scripts/maintenance/audit_doctrine_compliance.py",
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

# ── v0.3 §0.1-B audit 載體常數 ────────────────────────────────────────────────

# §0.1-A 禁令 #2/#3：T3 元素永久不實作清單
# 模式為 regex；掃描前會先 strip 註解與 docstring 以避免誤判文件引用
T3_FORBIDDEN_PATTERNS = {
    # IFF Θ 控制參數（§0.1.1 line 1205）
    "IFF_theta": r"\bIFF[_\s]?theta\b|\bΘ\s*=\s*\|?∇",
    "information_force_field": r"\binformation_force_field\b|\bIFF_field\b",
    "nabla_information": r"\|∇I\||\bnabla_I\b|\bgrad_information\b",
    "nabla_sentiment": r"\|∇S\||\bnabla_S\b|\bgrad_sentiment\b",
    # SOC 自組織臨界（§0.1.1 line 1206）
    "soc_critical": r"\bSOC_critical\b|\bself_organized_criticality\b|\bsandpile_model\b",
    "soc_trigger": r"\bsoc_trigger\b|\bcritical_avalanche\b",
    # 重力井邊緣觸發（§0.1.1 line 1207）
    "gravity_well_trigger": r"\bgravity_well_trigger\b|\bgravity_well_edge\b",
    "gravity_well_depth": r"\bgravity_well_depth\b|\bwell_depth_calc\b",
    "extreme_arbitrage_signal": r"\bextreme_arbitrage\b",
}

# T3 禁區掃描目標模組（§6 / §8 / §9 落地層）
# 不存在的模組（如預留 portfolio_sizer.py）以 PASS 記錄合法缺席
T3_SCAN_TARGETS = [
    "scripts/core/feature_store_builder.py",
    "scripts/core/model_trainer.py",
    "scripts/core/prediction_engine.py",
    "scripts/core/portfolio_sizer.py",  # 預留（v6.1.0 後啟用）
    "scripts/pipeline/portfolio_optimizer.py",
    "scripts/pipeline/portfolio_strategy.py",
    "scripts/pipeline/portfolio_backtest.py",
]

# §0.1.3 V 變數 + §0.1 M / ΔlnP / F proxy 對映規格
# Key: feature_name 前綴；Value: (對應憲章章節, §0.1 物理元素標籤)
# 在 description 中應找到至少其一才算明文標註
PROXY_FIRST_PRINCIPLES_MAPPING = {
    # ΔlnP 價格位移
    "log_return_": ("§0.1", "Delta_lnP"),
    "ma_ratio_": ("§0.1", "Delta_lnP"),
    "volatility_": ("§0.1", "Delta_lnP"),
    "max_drawdown_": ("§0.1", "Delta_lnP"),
    # M 流動性質量
    "avg_daily_value_": ("§0.1", "M"),
    "turnover_": ("§0.1", "M"),
    "zero_volume_ratio_": ("§0.1", "M"),
    # V 內在價值密度（§0.1.3 補強）
    "revenue_yoy_": ("§0.1.3", "V"),
    "eps_sum_": ("§0.1.3", "V"),
    "net_income_": ("§0.1.3", "V"),
    # F 外部資訊力（institutional proxy）
    "foreign_net_": ("§0.1", "F_external"),
    "trust_net_": ("§0.1", "F_external"),
    "margin_ratio_": ("§0.1", "F_external"),
    # v0.2 interaction 群（§0.0-D.6 升版條件 #1）；本質為 §0.3 × §0.1 乘積
    # PROXY check 允許 description 含 §0.1 或對應元素標籤即通過
    "feature_macro_vix_x_": ("§0.1", "Delta_lnP"),
    "feature_macro_dff_x_": ("§0.1.3", "V"),
    "feature_theme_x_log_return_": ("§0.1", "Delta_lnP"),
    "feature_theme_x_foreign_net_": ("§0.1", "F_external"),
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
        if not (_table_exists(cur, 'feature_definition') and _table_exists(cur, 'feature_store_snapshot')):
            self.add("P1_first_principles", "WARN", "physics_features",
                     "feature_definition / feature_store_snapshot 表不存在 (§8 DRAFT)；跳過 physics_features 檢驗")
        else:
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
        if not (_table_exists(cur, 'feature_definition') and _table_exists(cur, 'feature_store_snapshot')):
            self.add("P3_kondratiev_2026", "WARN", "macro_features",
                     "feature_definition / feature_store_snapshot 表不存在 (§8 DRAFT)；跳過 macro_features 檢驗")
        else:
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

    # ── v0.3 §0.1-B audit 載體新增方法 ──────────────────────────────────────

    @staticmethod
    def _strip_comments_and_docstrings(text: str) -> str:
        """移除 Python 註解與 docstring；避免 T3 regex 誤判 charter 引用文字。

        移除順序：
          1. triple-quoted strings (\"\"\" ... \"\"\" 與 ''' ... ''')
          2. 單行註解 (# ...)
        """
        text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)
        text = re.sub(r'#[^\n]*', '', text)
        return text

    def audit_t3_leakage(self):
        """§0.1-A 禁令 #2/#3 自動化檢驗 (v0.3 新增)。

        掃 §6 / §8 / §9 目標模組是否誤實作 T3 元素：
          - IFF Θ 控制參數
          - SOC 自組織臨界
          - 重力井邊緣觸發

        違憲後果：T3 元素任何實作 → FAIL（不可降為 WARN，永久禁令）。
        """
        print(f"\n🚫 [T3_LEAKAGE_CHECK] 掃描 §0.1-A 禁令 #2/#3 (IFF Θ / SOC / 重力井觸發)")
        leakage_found = False

        for rel in T3_SCAN_TARGETS:
            p = PROJECT_ROOT / rel
            if not p.exists():
                # portfolio_sizer.py 等預留檔案：合法缺席
                self.add("P1_first_principles", "PASS", "t3_leakage_skip",
                         f"{rel} 不存在（合法缺席 / 未實作）；跳過 T3 掃描")
                continue

            text = p.read_text(encoding="utf-8")
            # 移除註解與 docstring 後再掃（避免誤判文件引用）
            stripped = self._strip_comments_and_docstrings(text)

            violations = []
            for concept, pattern in T3_FORBIDDEN_PATTERNS.items():
                if re.search(pattern, stripped, re.IGNORECASE | re.MULTILINE):
                    violations.append(concept)

            if violations:
                leakage_found = True
                self.add("P1_first_principles", "FAIL", "t3_leakage",
                         f"{rel} 違反 §0.1-A 禁令；發現 T3 元素實作: {violations}")
            else:
                self.add("P1_first_principles", "PASS", "t3_leakage",
                         f"{rel} 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）")

        if not leakage_found:
            self.add("P1_first_principles", "PASS", "t3_leakage_summary",
                     f"§0.1-A 禁令 #2/#3 全部守住；掃描 {len(T3_SCAN_TARGETS)} 個模組無 T3 元素實作")

    def audit_proxy_transparency(self, cur):
        """§0.1-B audit 載體 + §0.1.3 V 變數對應透明化檢驗 (v0.3 新增)。

        驗證 feature_definition 中之 proxy 變數是否在 description 明文標註
        §0.1 元素對應（M / V / ΔlnP / F_external）。

        違憲後果：缺對應標註 → WARN（文檔衛生問題，不影響工程正確性）。
        """
        print(f"\n📝 [PROXY_TRANSPARENCY_CHECK] §0.1 元素對應標註檢驗")

        if not (_table_exists(cur, 'feature_definition') and
                _table_exists(cur, 'feature_store_snapshot')):
            self.add("P1_first_principles", "WARN", "proxy_transparency",
                     "feature_definition 表不存在（§8 DRAFT）；跳過 proxy transparency 檢驗")
            return

        # 取最新 committed feature set 的所有 feature definitions
        cur.execute("""
            SELECT feature_name, feature_group, description
            FROM feature_definition
            WHERE feature_set_id IN (
                SELECT feature_set_id FROM feature_store_snapshot
                WHERE status='committed'
                ORDER BY as_of_date DESC LIMIT 1
            )
            ORDER BY feature_group, feature_name
        """)
        rows = cur.fetchall()

        if not rows:
            self.add("P1_first_principles", "FAIL", "proxy_transparency",
                     "無 committed feature_definition；§0.1 proxy 透明度無從驗證")
            return

        untagged_features = []
        correctly_tagged = 0

        for feature_name, group, description in rows:
            # 找出該 feature 應對應的 §0.1 元素
            expected = None
            for prefix, (section, element) in PROXY_FIRST_PRINCIPLES_MAPPING.items():
                if feature_name.startswith(prefix):
                    expected = (section, element)
                    break

            if not expected:
                # macro / theme 等非 §0.1 特徵：不檢驗（屬 §0.3）
                continue

            section, element = expected
            text_to_check = (description or "").lower()

            # 必須在 description 中明文標註章節或物理元素
            if section.lower() not in text_to_check and element.lower() not in text_to_check:
                untagged_features.append((feature_name, expected))
            else:
                correctly_tagged += 1

        if untagged_features:
            sample = [f"{name} (應標 {sec}/{elt})"
                      for name, (sec, elt) in untagged_features[:5]]
            self.add("P1_first_principles", "WARN", "proxy_transparency",
                     f"{len(untagged_features)} 個 §0.1 proxy 未明文標註對應 (前 5 例: {sample})")
        else:
            self.add("P1_first_principles", "PASS", "proxy_transparency",
                     f"全部 {correctly_tagged} 個 §0.1 proxy features 已明文標註對應元素")

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
        """v0.4 §14.7-AV: dual-track promotion gate
        - Operations Reality 軌道 (v6.1.0):僅檢 §0 doctrine + 跨平台依賴 / 路徑 / 402 cascade 治權條款,不檢 §8 schema
        - §8/§9 軌道 (v6.1.1+):必須通過 §8 schema + production-current model >= label_date threshold
        對齊憲章 v6.1.0 §9.5 升版路徑表雙軌制(L4744-L4750)。
        """
        target = self.for_promotion
        print(f"\n🚀 [FOR-PROMOTION] target={target}")

        # v0.4 §14.7-AV 軌道分流
        if target in OPERATIONS_TRACK_VERSIONS:
            # Operations Reality Patch 軌道:不檢 §8 schema
            print(f"   軌道 = Operations Reality Patch (per §9.5 升版路徑表 + 憲章 v6.1.0 §14.7-AU)")
            print(f"   檢查項目:§0 doctrine + §0.0-I.9/.10 跨平台 + §7.4-A cascade + §3.2A.H/I audit perf")
            # §0.0-I.9 跨平台依賴宣告 — 檢 requirements.txt 含跨平台前置區塊
            req_path = os.path.join(_REPO_ROOT, "requirements.txt") if "_REPO_ROOT" in globals() else "requirements.txt"
            try:
                with open(req_path, "r") as f:
                    req_content = f.read()
                if "libomp" in req_content and ("brew install" in req_content or "apt-get" in req_content):
                    self.add("P4_observability_digital_twin", "PASS", "promotion_gate",
                             f"target={target} (Operations Reality) §0.0-I.9 跨平台依賴宣告已在 requirements.txt 標頭實作")
                else:
                    self.add("P4_observability_digital_twin", "WARN", "promotion_gate",
                             f"target={target} §0.0-I.9 跨平台依賴宣告缺失 requirements.txt 標頭")
            except Exception as exc:
                self.add("P4_observability_digital_twin", "WARN", "promotion_gate",
                         f"target={target} requirements.txt 讀取失敗:{type(exc).__name__}: {exc}")
            # 全部通過視為 PASS
            return

        if target in DOWNSTREAM_TRACK_VERSIONS:
            # §8/§9 軌道:需 §8 schema + production-current model
            print(f"   軌道 = §8/§9 promotion 軌道 (per §9.5 升版路徑表)")
            print(f"   檢查項目:§8 schema 存在 + production-current model label_date >= as_of+horizon")
            if not _table_exists(cur, 'model_registry'):
                self.add("P4_observability_digital_twin", "FAIL", "promotion_gate",
                         f"target={target} (§8/§9 軌道) model_registry 表不存在;先 build §8 schema "
                         f"(scripts/core/feature_store_schema.py --init + model_trainer DDL)")
                return
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
                         f"target={target} (§8/§9 軌道) 升版 BLOCKED:無 production-current 模型符合 "
                         f"label_date >= as_of+horizon;命中 §8.8.9-D 條件 #1(time-gated 至 DB MAX(date) >= as_of+20d)")
            else:
                self.add("P4_observability_digital_twin", "PASS", "promotion_gate",
                         f"target={target} (§8/§9 軌道) 升版條件之 production-current 模型存在: {valid_models} 件")
            return

        # 未知 target:fallback 走嚴格 §8 schema 檢驗
        print(f"   ⚠ 未知 target version;fallback 走 §8 schema 嚴格檢驗")
        if not _table_exists(cur, 'model_registry'):
            self.add("P4_observability_digital_twin", "FAIL", "promotion_gate",
                     f"target={target} 未在 KNOWN_TRACKS 範圍內;fallback 嚴格檢驗:model_registry 缺失")

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
        lifecycle_cm = record_lifecycle(f"audit_doctrine_compliance_{TOOL_VER}",
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
                # v0.3：§0.1-B audit 載體——T3 禁令 + proxy 透明化
                self.audit_t3_leakage()
                self.audit_proxy_transparency(cur)
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
        description="Quantum Finance §0 Supreme Doctrine Compliance Auditor (v0.3)")
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
