"""
portfolio_sizer.py v0.3 (Quantum Finance Portfolio Sizing Authority)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 補入 [Sovereignty Declaration] + Supreme Authority Principle line)
**主權狀態**: IMPLEMENTED (憲法 v6.1.0 §9.2-I v0.3 + §14.7-BN ROE-aware Pareto + sector count 5→3 + G13/G14/G15 audit gates + §14.7-CU Portfolio Sizing Production Closure + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

**[Sovereignty Declaration]** (2026-05-29 §一.11 補入,憲法 §3.1 序列模組 / §14.7-CU): 本程式為 **§9.2 portfolio_sizer + §14.7-CU Production Closure 唯一治權載體**(§3.1 序列模組第 9/9 員)。**治權邊界**:(a) §3.1 序列 portfolio sizing 模組;(b) 五套禁令不涉;(c) T1-T3 不分層;(d) §8.5 不處理;(e) **不訓練 model**;(f) **不預測**(讀取 prediction_run 結果);(g) **不執行交易**;(h) 唯一職責:從 prediction_run 取 top stocks + 依 §9.2 barbell 計算 sizing → portfolio_proposal markdown 輸出。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Sizing Authority]: 對齊憲章 §0.0-A.5 第五個轉換器裁決——本程式為「formal
   prediction → allocation proposal / position weights」的工程轉換器；不重選
   universe、不重訓 model、不重算 prediction。
2. [Read-Only Upstream]: 只讀唯一 committed `prediction_run` (status='committed')
   及其 `prediction_values`；source-of-truth 為 §8.4 prediction layer。
3. [Single Delivery Gate] (§8.8.8): 同 `prediction_policy_version` 下若
   committed prediction-backed run != 1，直接 FAIL；強制 exactly-one delivery。
4. [Universe Coverage Lock]: prediction_values rows 必須 = 當前 committed
   snapshot 之 N（即 core+convex 全集；dynamic per §14.7-BW pure doctrine,
   無 hardcoded 150/200 cap/floor）；否則 FAIL。
5. [Barbell Bucket Caps] (§0.2 槓鈴策略 v0.2):
     attack_total_weight_max  = 0.20  # 攻擊端總權重上限
     safety_total_weight_min  = 0.80  # 防禦端（cash sleeve）下限
     single_stock_weight_max  = 0.05  # 一般 core stock 單一上限
     convex_tier_weight_max   = 0.03  # convex tier 單一上限（更嚴防信仰）
     sector_weight_max        = 0.40  # sector 集中上限（§14.7-U semiconductor）
     single_sector_count_max  = 5     # ★ v0.2 新增：同 sector 配置股票數上限 G12
6. [Right-Tail Concentration]: 只配置 `signal_label='long'` 或 top rank bucket；
   bottom 20 / 'watch' 永不配置；中段不分散（§0.2 拒絕中段）。
7. [Cash As Safety]: 未配置權重保留為 CASH safety sleeve；不買 0050、不買 ETF。
8. [Read-Only Output v0.1+]: 本版只輸出 markdown allocation proposal report；
   不寫入新治理表（portfolio_run / portfolio_weights 待 §9.2 強制契約升版後再建）。
9. [No T3 Element]: 永久禁用 §0.1-A 禁令 #2/#3 列出之 T3 元素（IFF Θ / SOC /
   重力井邊緣觸發）；本程式僅用 rank-based 加權，不使用任何物理隱喻公式。
10. [Hybrid Observability]: 維運觸發 `record_lifecycle` 與 `write_data_audit_log`；
    產出 reports/portfolio_allocation_proposal_<asof>.md。
11. [v0.2 新增 G11 as_of_date 一致性]: 必須驗證 `prediction_run.as_of_date` 等於
    對應 `feature_store_snapshot.as_of_date`（§8.5 anti-leakage 邊界）。
12. [v0.2 新增 G12 single-sector count cap]: 同一 sector 配置股票數 > 5 直接違憲，
    解 100% 單一產業集中之治權精神（依 §14.7-AA Part C）。
13. [v0.2 新增 ConstitutionalViolationError]: 所有 FAIL gate 觸發必須拋出此例外，
    不得僅以軟錯誤 log 訊息替代（依 §9.2-D.1）。
14. [v0.2 新增 Audit Hooks 獨立化]: 4 個 audit hook 為 module-level function，
    可被 audit_doctrine_compliance.py 直接 import 並呼叫（依 §9.2-F.1）。
15. [v0.3 新增 ROE-aware Pareto weighting]: 取 v0.7 snapshot 之 fg_roe + 標準化 +
    multiplier [0.5, 1.5] clamp + base_cap × roe_mult,仍受 G5/G6 cap 約束（依 §9.2-I.3）。
16. [v0.3 新增 Prediction value-weighted]: value_z 標準化 + multiplier [0.7, 1.3]
    clamp,與 roe_mult 相乘 final_mult [0.5, 1.5]（依 §9.2-I.4）。
17. [v0.3 新增 G13 ROE-weighted Pareto compliance]: 配置完成後 top 1 ROE 股
    weight ≥ median ROE 股 weight × 1.0；FAIL → raise ConstitutionalViolationError（依 §9.2-I.5）。
18. [v0.3 新增 G14 score_scope v0.6/v0.7 對齊]: upstream prediction_run 對應之
    universe_snapshot 必須為 core_universe_policy_v0.6 或 v0.7（依 §9.2-I.5）。
19. [v0.3 新增 G15 ROE coverage gate (WARN-only)]: core+convex ROE 覆蓋度 < 90%
    WARN（不阻塞;對映金融業 BS 對齊 → §14.7-BM Phase A）。
20. [v0.3 Root cause 限制聲明]: v0.3 為治標(sizer 層 ROE-aware + sector count 強化);
    upstream prediction 100% sector 集中之 root cause 需 §10 model_trainer 治本（依 §9.2-I.7）。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [Step 12-dry：配置 dry-run]** | `$ python scripts/core/portfolio_sizer.py --dry-run` | portfolio_sizer v0.2 |
| **2. [Step 12-report：產出 allocation proposal]** | `$ python scripts/core/portfolio_sizer.py --commit-report` | portfolio_sizer v0.2 |
| **3. [Step 12-asof：指定特定 as-of-date]** | `$ python scripts/core/portfolio_sizer.py --commit-report --as-of-date 2026-04-25` | portfolio_sizer v0.2 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **dry-run** | `--dry-run` | 只在 stdout 輸出 allocation proposal；不寫 markdown 報告 |
| **commit-report** | `--commit-report` | 寫 markdown 報告至 reports/portfolio_allocation_proposal_<asof>.md |
| **as-of-date** | `--as-of-date YYYY-MM-DD` | 指定 prediction run as-of-date（預設使用最新 committed run） |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.1** | 2026-05-19 | Codex | 首版：§9.2 配置層雛形；§0.0-A.5 五大轉換器裁決之第五個轉換器落地；對應 portfolio_sizer_barbell_allocation_research_20260519.md §7 sizing policy v0.1 十條規則；不建新治理表（待 §9.2 強制契約升版）。落地後解除 §0.0-B / §0.0-C / §0.0-D 三件套共同最後斷路點。 | **SUPERSEDED** |
| **v0.2** | 2026-05-20 | Codex | 補強：依 §14.7-AA 揭露之 v0.1 4 項缺口入憲 §14.7-AB 設計：(1) 定義 `ConstitutionalViolationError` 類別（§9.2-D.1）；(2) 抽出 4 個 audit hook 為 module-level function（§9.2-F.1）；(3) 新增 G11 as_of_date 一致性檢查；(4) 新增 G12 single-sector count cap（max=5）解 100% 半導體集中之治權精神。預期合規度由 v0.1 之 80% 升至 ≥97.5%。 | **SUPERSEDED** |
| **v0.3** | 2026-05-26 | Codex | **§9.2-I 補強條款 + §14.7-BN 治權升版預備記述入憲(v6.1.0-patch 第十四輪;Phase C 程式落地;對齊 v6.1.18+ ROE/v0.7/V 73%)**:依憲章 §9.2-I 12 子節入憲(Phase B commit `9ea41ce` tag `v6.1.21`)之治權升版預備,本版落地 ROE-aware Pareto weighting + Prediction value-weighted + G13/G14/G15 audit gates + sector count 5→3 + 對齊 v6.1.18+ snapshot。**補正內容**:(I) `CONSTITUTION_VER v6.0.0 → v6.1.0` / `TOOL_VER v0.2 → v0.3` / `DEFAULT_PREDICTION_POLICY_VERSION v0.1 → v0.2` / `DEFAULT_SIZING_POLICY_VERSION v0.2 → v0.3`;(II) DEFAULT_POLICY 升 5 新 params(`roe_weight_alpha=0.5`, `prediction_value_weight_beta=0.3`, `roe_multiplier_clamp_min=0.5`, `roe_multiplier_clamp_max=1.5`, `value_multiplier_clamp_min=0.7`, `value_multiplier_clamp_max=1.3`);(III) `single_sector_count_max` 5 → 3 強化;(IV) 新增 `_load_roe_lookup()` method 從 core_universe_scores 之 score_detail->>'fg_roe' 讀 ROE;(V) 新增 `_compute_roe_multiplier()` + `_compute_value_multiplier()` 兩個 helper functions;(VI) `apply_policy` 加 ROE-weighted + value-weighted multipliers(改 proposed weight 計算);(VII) 新增 G13/G14/G15 audit gates(`audit_constraint_satisfaction` 擴張);(VIII) load_inputs 加 G14 check(snapshot policy_version 對齊 v0.6/v0.7);(IX) 加標頭核心定義第 15-20 條(v0.3 新增 6 條)。**對既有 snapshot 影響**:零(v0.2 snapshot universe 不變;v0.7 universe 不變)。**對下游影響**:audit_doctrine_compliance 需小升版識別 sizing_policy_v0.3(另案)。**Root cause 限制聲明**:v0.3 為治標(sizer 層);upstream prediction 100% sector 集中 root cause 需 §10 model_trainer 治本。**對既有 12 FAIL gate 影響**:零(僅擴張 G13/G14/G15)。**證偽承諾啟動**:T_PS_v0.3-1〜5(walk-forward IC 等 v6.2.0)。本版**不**修改 §9.2-A~H 既有 12 FAIL gate 邏輯 / §0.2-A 7 禁令 / 攻擊端 20% / 防護端 80% / 單股 5% / convex 3% / sector 40% / §6.4 / §6.7 / §0.1-A 禁令 / raw DDL / CLI 結構(只 default 升)。同步配套:`reports/portfolio_sizer_v03_design_research_20260526.md`(Phase A 384 行 15 章 commit `59bfc8f` tag v6.1.19) + 憲章 §9.2-I + §14.7-BN(commit `9ea41ce` tag v6.1.21)。 | **ACTIVE** |
================================================================================
"""
import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.3"
DEFAULT_PREDICTION_POLICY_VERSION = "prediction_policy_v0.2"
DEFAULT_SIZING_POLICY_VERSION = "sizing_policy_v0.3"
# v0.3 §14.7-BN: 對齊 v0.6/v0.7 snapshot(builder v0.7.1 RMS / v0.8 ROE)
ALIGNED_UNIVERSE_POLICY_VERSIONS = {
    "core_universe_policy_v0.6",  # builder v0.7.1 RMS
    "core_universe_policy_v0.7",  # builder v0.8 ROE 解鎖(production after §14.7-BI)
}

# §0.2 槓鈴策略 v0.3 預設規則（依 §9.2-E 12 條 + §9.2-I.2 v0.3 升版表）
DEFAULT_POLICY = {
    "attack_total_weight_max": 0.20,
    "safety_total_weight_min": 0.80,
    "single_stock_weight_max": 0.05,
    "convex_tier_weight_max": 0.03,
    "sector_weight_max": 0.40,
    "single_sector_count_max": 3,       # v0.3: 5 → 3 強化(§9.2-I.2 / §14.7-AA Part C 強化)
    "required_coverage": None,          # §14.7-BW pure doctrine: dynamic per committed snapshot N (was 150)
    "max_committed_runs": 1,
    # v0.3 §9.2-I.2 新增 5 參數 + 1 升版(共 6 新 params):
    "roe_weight_alpha": 0.5,             # ROE-weighted Pareto 強度(§9.2-I.3)
    "prediction_value_weight_beta": 0.3, # raw value-weighted 強度(§9.2-I.4)
    "roe_multiplier_clamp_min": 0.5,     # ROE 極端低 weight 下限保護
    "roe_multiplier_clamp_max": 1.5,     # ROE 極端高 weight 上限保護
    "value_multiplier_clamp_min": 0.7,   # value 極端低 weight 下限保護
    "value_multiplier_clamp_max": 1.3,   # value 極端高 weight 上限保護
    "roe_coverage_warn_threshold": 0.90, # G15 ROE coverage WARN 門檻
}

# §9.2-C 強制輸出 schema 9 欄位（v0.2 audit_proposal_schema 用）
PROPOSAL_REQUIRED_FIELDS = [
    "stock_id", "tier", "sector",
    "prediction_rank", "prediction_value", "signal_label",
    "target_weight", "allocation_reason", "risk_flags",
]


# ════════════════════════════════════════════════════════════════════════════
# §9.2-D.1 違憲例外契約 (Constitutional Exception Contract) — v0.2 入憲
# ════════════════════════════════════════════════════════════════════════════
class ConstitutionalViolationError(Exception):
    """憲章 §0.0-G + §9.2-D 之違憲攔截例外。

    依 §9.2-D.1，所有 FAIL gate (G1〜G12) 觸發必須拋出此例外，不得僅以軟錯誤
    log 訊息替代。CLI 層應於 __main__ 統一捕獲。

    Attributes:
        gate_id: FAIL gate 編號（G1〜G12 或未來新增）
        message: 違憲具體訊息
        charter_ref: 對應憲章節（如 "§9.2-D / G7"）
    """

    def __init__(self, gate_id: str, message: str, charter_ref: str):
        self.gate_id = gate_id
        self.message = message
        self.charter_ref = charter_ref
        super().__init__(f"[{gate_id}] {message} (依 {charter_ref})")


# ════════════════════════════════════════════════════════════════════════════
# §9.2-F.1 Audit Hooks 強制獨立函式 (Mandatory Standalone Functions) — v0.2
# ════════════════════════════════════════════════════════════════════════════
def audit_input_uniqueness(prediction_runs, prediction_rows, upstream_writes, expected_coverage=None):
    """G1/G2/G9/G10: 唯一 delivery + coverage + read-only 邊界

    Args:
        prediction_runs: committed prediction-backed run 清單
        prediction_rows: 該 run 之 prediction_values rows 數
        upstream_writes: sizer 是否曾呼叫上游 write 操作之記錄
        expected_coverage: §14.7-BW pure doctrine — 當前 committed snapshot 之
            動態 N(由 caller 從 snapshot 查得);None 時跳過 G2 coverage 檢查

    Returns:
        (bool, str): (pass, message)
    """
    if len(prediction_runs) != 1:
        return False, f"G1: committed run count = {len(prediction_runs)}, expected 1"
    if expected_coverage is not None and prediction_rows != expected_coverage:
        return False, f"G2: prediction rows = {prediction_rows}, expected {expected_coverage} (dynamic per §14.7-BW)"
    if upstream_writes:
        return False, f"G9/G10: sizer attempted upstream writes: {upstream_writes}"
    return True, "OK"


def audit_constraint_satisfaction(allocations, policy, sector_counts, roe_lookup=None):
    """G3-G8 + G12 + (v0.3) G13: 槓鈴 caps + sector cap + bottom 20 隔離 + single-sector count + ROE-weighted Pareto

    Args:
        allocations: 全部配置紀錄（含未配置者 weight=0）
        policy: DEFAULT_POLICY
        sector_counts: sector → 實際配置股票數
        roe_lookup: (v0.3 新增 optional) stock_id → fg_roe;若 None 跳過 G13 檢查

    Returns:
        (bool, str): (pass, message)
    """
    # G3/G4 槓鈴
    attack_total = sum(a["target_weight"] for a in allocations)
    if attack_total > policy["attack_total_weight_max"] + 0.0001:
        return False, f"G4: attack={attack_total:.4f} > cap"
    if (1.0 - attack_total) < policy["safety_total_weight_min"] - 0.0001:
        return False, f"G3: cash={1.0-attack_total:.4f} < safety_min"

    # G5/G6 個股 cap
    for a in allocations:
        if a["target_weight"] <= 0:
            continue
        cap = (policy["convex_tier_weight_max"]
               if a["tier"] == "convex_universe"
               else policy["single_stock_weight_max"])
        if a["target_weight"] > cap + 0.0001:
            return False, f"G5/G6: stock {a['stock_id']} weight {a['target_weight']:.4f} > cap {cap}"

    # G7 sector cap
    sector_totals = defaultdict(float)
    for a in allocations:
        sector_totals[a["sector"]] += a["target_weight"]
    for sec, total in sector_totals.items():
        if total > policy["sector_weight_max"] + 0.0001:
            return False, f"G7: sector {sec} total={total:.4f} > cap"

    # G8 左尾隔離
    for a in allocations:
        if a["signal_label"] == "watch" and a["target_weight"] > 0:
            return False, f"G8: watch stock {a['stock_id']} has weight > 0"

    # G12 single-sector count
    for sec, count in sector_counts.items():
        if count > policy["single_sector_count_max"]:
            return False, f"G12: sector {sec} count={count} > max {policy['single_sector_count_max']}"

    # v0.3 G13: ROE-weighted Pareto compliance(top 1 ROE 股 weight ≥ median ROE 股 weight)
    # 只在 roe_lookup 提供時檢查;若 < 3 個 allocated stocks 有 ROE → 跳過(資料不足無意義)
    if roe_lookup:
        allocated_with_roe = [(a["stock_id"], a["target_weight"], roe_lookup.get(a["stock_id"]))
                              for a in allocations if a["target_weight"] > 0]
        allocated_with_roe = [(sid, w, r) for sid, w, r in allocated_with_roe if r is not None]
        if len(allocated_with_roe) >= 3:
            sorted_by_roe = sorted(allocated_with_roe, key=lambda x: -x[2])  # ROE desc
            top1_weight = sorted_by_roe[0][1]
            median_idx = len(sorted_by_roe) // 2
            median_weight = sorted_by_roe[median_idx][1]
            if top1_weight < median_weight * 0.999:  # 容忍 1bp 浮點誤差
                return False, (f"G13: top-ROE stock {sorted_by_roe[0][0]} weight {top1_weight:.4f} "
                               f"< median-ROE stock {sorted_by_roe[median_idx][0]} weight {median_weight:.4f} "
                               f"(ROE-weighted Pareto compliance fail)")

    return True, "OK"


def audit_proposal_schema(proposal_rows, required_fields):
    """§9.2-C 輸出 schema 9 欄位完整性

    Args:
        proposal_rows: 配置明細列表
        required_fields: 必要欄位清單

    Returns:
        (bool, str): (pass, message)
    """
    if not proposal_rows:
        return True, "OK (empty proposal)"
    for i, row in enumerate(proposal_rows):
        missing = [f for f in required_fields if f not in row]
        if missing:
            return False, f"row {i} ({row.get('stock_id', '?')}) missing fields: {missing}"
    return True, "OK"


def audit_log_observability(stats, allocations):
    """risk_flags / allocation_reason 完整記錄

    Args:
        stats: 執行 stats dict（必含 'details' key）
        allocations: 全部配置紀錄

    Returns:
        (bool, str): (pass, message)
    """
    if "details" not in stats:
        return False, "stats missing 'details' key"
    for a in allocations:
        if a["target_weight"] > 0 and not a.get("allocation_reason"):
            return False, f"stock {a['stock_id']} missing allocation_reason"
    return True, "OK"


# ════════════════════════════════════════════════════════════════════════════
# 主類別
# ════════════════════════════════════════════════════════════════════════════
class PortfolioSizer:
    """§9.2 配置層 v0.2 實作。

    v0.2 變更：
      - 定義 ConstitutionalViolationError 並於 FAIL gate 強制拋出
      - DEFAULT_POLICY 新增 single_sector_count_max
      - 4 個 audit hook 抽出為 module-level function（依 §9.2-F.1）
      - load_inputs 補入 G11 as_of_date 一致性檢查
      - apply_policy 補入 G12 single-sector count cap

    輸入：
      - 唯一 committed prediction_run (status='committed', prediction_policy_version=v0.1)
      - 該 run 的 prediction_values
      - 對應 universe_snapshot_id 的 core_universe_membership

    輸出：
      - allocation proposal markdown report
      - dry-run stdout summary
    """

    def __init__(self, as_of_date=None, dry_run=True, commit_report=False, policy=None):
        self.as_of_date = as_of_date
        self.dry_run = dry_run
        self.commit_report = commit_report
        self.policy = dict(DEFAULT_POLICY)
        if policy:
            self.policy.update(policy)
        self.run_meta = None
        self.predictions = []
        self.memberships = {}
        self.allocations = []
        self.cash_weight = 1.0
        self.sector_totals = defaultdict(float)
        self.sector_counts = defaultdict(int)  # v0.2 新增 G12 用
        self.tier_totals = defaultdict(float)
        self.risk_flags = []
        self.stats = {"pass": 0, "warn": 0, "fail": 0, "details": []}
        self.upstream_writes = []  # v0.2 G9/G10 audit 用（純記錄，預期保持空）
        self.roe_lookup = {}  # v0.3: stock_id → fg_roe(來自 v0.7 snapshot score_detail)
        self.snapshot_policy_version = None  # v0.3: G14 對齊用

    def _detail(self, bucket, msg):
        self.stats[bucket] += 1
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[bucket]
        line = f"{icon} [{bucket.upper()}] {msg}"
        self.stats["details"].append(line)
        print(line)

    # ── Input Gate ───────────────────────────────────────────────────────────

    def load_inputs(self):
        """載入唯一 committed prediction_run 與其 prediction_values + membership"""
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Step 1: 找符合 single-delivery 規則之 committed run
            if self.as_of_date is None:
                cur.execute(
                    """
                    SELECT run_id, model_id, feature_set_id, as_of_date,
                           universe_snapshot_id, prediction_policy_version,
                           rows_written, notes
                    FROM "prediction_run"
                    WHERE status = 'committed'
                      AND prediction_policy_version = %s
                    ORDER BY as_of_date DESC, created_at DESC
                    """,
                    (DEFAULT_PREDICTION_POLICY_VERSION,),
                )
            else:
                cur.execute(
                    """
                    SELECT run_id, model_id, feature_set_id, as_of_date,
                           universe_snapshot_id, prediction_policy_version,
                           rows_written, notes
                    FROM "prediction_run"
                    WHERE status = 'committed'
                      AND prediction_policy_version = %s
                      AND as_of_date = %s
                    ORDER BY created_at DESC
                    """,
                    (DEFAULT_PREDICTION_POLICY_VERSION, self.as_of_date),
                )
            rows = cur.fetchall()

            # v0.2: 改用 audit_input_uniqueness 並 raise ConstitutionalViolationError
            run_dicts = [{"run_id": r[0]} for r in rows]
            # G1 / G2 / G9 / G10 將於 prediction_rows 載入後一起檢查；G1 先檢
            if len(rows) == 0:
                raise ConstitutionalViolationError(
                    gate_id="G1",
                    message=f"no committed prediction-backed run found "
                            f"(policy={DEFAULT_PREDICTION_POLICY_VERSION}, "
                            f"as_of_date={self.as_of_date})",
                    charter_ref="§9.2-D / G1 / §0.0-A.5",
                )
            if len(rows) > self.policy["max_committed_runs"]:
                raise ConstitutionalViolationError(
                    gate_id="G1",
                    message=f"§8.8.8 violation: found {len(rows)} committed "
                            f"runs, expected exactly {self.policy['max_committed_runs']}",
                    charter_ref="§9.2-D / G1 / §8.8.8",
                )

            keys = ["run_id", "model_id", "feature_set_id", "as_of_date",
                    "universe_snapshot_id", "prediction_policy_version",
                    "rows_written", "notes"]
            self.run_meta = dict(zip(keys, rows[0]))
            if self.as_of_date is None:
                self.as_of_date = self.run_meta["as_of_date"]
            self._detail("pass", f"G1 committed prediction run located: "
                         f"{self.run_meta['run_id']} (model={self.run_meta['model_id']})")

            # Step 2: 載入 prediction_values
            cur.execute(
                """
                SELECT stock_id, prediction_value, prediction_rank,
                       signal_label, confidence
                FROM "prediction_values"
                WHERE run_id = %s
                ORDER BY prediction_rank ASC
                """,
                (self.run_meta["run_id"],),
            )
            for stock_id, value, rank, label, conf in cur.fetchall():
                self.predictions.append({
                    "stock_id": stock_id,
                    "prediction_value": float(value) if value is not None else 0.0,
                    "prediction_rank": int(rank) if rank is not None else 0,
                    "signal_label": label or "hold",
                    "confidence": float(conf) if conf is not None else 0.0,
                })

            # v0.2: 使用 audit_input_uniqueness 統一稽核 G1/G2/G9/G10
            # §14.7-BW pure doctrine: 從 snapshot 動態取 N(取代 hardcoded 150)
            cur.execute(
                """
                SELECT COUNT(*) FROM "core_universe_membership"
                WHERE snapshot_id = %s
                  AND core_tier IN ('core_universe', 'convex_universe')
                """,
                (self.run_meta["universe_snapshot_id"],),
            )
            expected_coverage = cur.fetchone()[0]
            ok, msg = audit_input_uniqueness(
                prediction_runs=[self.run_meta],
                prediction_rows=len(self.predictions),
                upstream_writes=self.upstream_writes,
                expected_coverage=expected_coverage,
            )
            if not ok:
                raise ConstitutionalViolationError(
                    gate_id="G2",
                    message=msg,
                    charter_ref="§9.2-D / G2 / §9.2-F.1",
                )
            self._detail("pass", f"G2 prediction_values loaded: rows={len(self.predictions)}")

            # Step 3: 載入 membership（tier + sector）
            cur.execute(
                """
                SELECT stock_id, core_tier, industry_category, stock_name
                FROM "core_universe_membership"
                WHERE snapshot_id = %s
                  AND core_tier IN ('core_universe', 'convex_universe')
                """,
                (self.run_meta["universe_snapshot_id"],),
            )
            for stock_id, tier, sector, name in cur.fetchall():
                self.memberships[stock_id] = {
                    "core_tier": tier,
                    "industry_category": sector or "UNKNOWN",
                    "stock_name": name or "",
                }
            # §14.7-BW pure doctrine: required_coverage 為 None 時 fallback 至 expected_coverage(snapshot N)
            req_cov = self.policy.get("required_coverage")
            if req_cov is None:
                req_cov = expected_coverage  # dynamic from snapshot (§14.7-BW)
            if len(self.memberships) != req_cov:
                self._detail("warn", f"membership coverage = {len(self.memberships)}, "
                             f"expected {req_cov}")
            else:
                self._detail("pass", f"membership loaded: rows={len(self.memberships)} "
                             f"(snapshot={self.run_meta['universe_snapshot_id']})")

            # Step 4: 確認 prediction_values ↔ membership 對映完整
            missing = [p["stock_id"] for p in self.predictions
                       if p["stock_id"] not in self.memberships]
            if missing:
                raise ConstitutionalViolationError(
                    gate_id="G2",
                    message=f"{len(missing)} prediction stocks not in "
                            f"membership; e.g. {missing[:3]}",
                    charter_ref="§9.2-D / G2 / §6.7",
                )
            self._detail("pass", f"G2 prediction × membership join complete ({len(self.predictions)}/{len(self.memberships)})")

            # Step 5: v0.2 新增 G11 as_of_date 跨層一致性
            cur.execute(
                'SELECT as_of_date FROM "feature_store_snapshot" WHERE feature_set_id = %s',
                (self.run_meta["feature_set_id"],),
            )
            fs_row = cur.fetchone()
            if fs_row is None:
                raise ConstitutionalViolationError(
                    gate_id="G11",
                    message=f"feature_set_id {self.run_meta['feature_set_id']} not found",
                    charter_ref="§9.2-D / G11 / §8.5",
                )
            fs_as_of_date = fs_row[0]
            if fs_as_of_date != self.run_meta["as_of_date"]:
                raise ConstitutionalViolationError(
                    gate_id="G11",
                    message=(f"prediction_run.as_of_date={self.run_meta['as_of_date']} "
                             f"!= feature_set.as_of_date={fs_as_of_date}"),
                    charter_ref="§9.2-D / G11 / §8.5",
                )
            self._detail("pass", f"G11 as_of_date consistency verified: "
                         f"prediction={self.run_meta['as_of_date']} == "
                         f"feature_set={fs_as_of_date}")

            # v0.3 Step 6: G14 — snapshot policy_version 對齊 v0.6/v0.7
            cur.execute(
                'SELECT policy_version FROM "core_universe_snapshot" WHERE snapshot_id = %s',
                (self.run_meta["universe_snapshot_id"],),
            )
            snap_row = cur.fetchone()
            if snap_row is None:
                raise ConstitutionalViolationError(
                    gate_id="G14",
                    message=f"universe_snapshot_id {self.run_meta['universe_snapshot_id']} not found",
                    charter_ref="§9.2-I.5 / G14 / §6.7",
                )
            self.snapshot_policy_version = snap_row[0]
            if self.snapshot_policy_version not in ALIGNED_UNIVERSE_POLICY_VERSIONS:
                # v0.3 backward-compat: v0.2 snapshot 為 legacy(本機 stranded state)
                # 改為 WARN 不 FAIL,避免破壞 v0.2 baseline 之 dry-run capability
                self._detail("warn", f"G14 backward-compat: snapshot policy={self.snapshot_policy_version} "
                             f"not in v0.3 aligned set {sorted(ALIGNED_UNIVERSE_POLICY_VERSIONS)}; "
                             f"ROE-weighted will fallback if no fg_roe available")
            else:
                self._detail("pass", f"G14 snapshot policy alignment verified: {self.snapshot_policy_version}")

            # v0.3 Step 7: 載入 ROE lookup(從 core_universe_scores 之 score_detail->>'fg_roe')
            cur.execute(
                """
                SELECT stock_id, (score_detail->>'fg_roe')::float AS fg_roe
                FROM "core_universe_scores"
                WHERE snapshot_id = %s
                  AND score_detail ? 'fg_roe'
                  AND score_detail->>'fg_roe' IS NOT NULL
                  AND score_detail->>'fg_roe' <> 'null'
                """,
                (self.run_meta["universe_snapshot_id"],),
            )
            for sid, roe in cur.fetchall():
                if roe is not None:
                    self.roe_lookup[sid] = float(roe)
            roe_coverage = len(self.roe_lookup) / max(1, len(self.memberships))
            warn_thr = self.policy.get("roe_coverage_warn_threshold", 0.90)
            if roe_coverage < warn_thr:
                self._detail("warn", f"G15 ROE coverage {roe_coverage*100:.1f}% < threshold {warn_thr*100:.0f}% "
                             f"({len(self.roe_lookup)}/{len(self.memberships)} stocks have fg_roe);"
                             f" ROE-weighted will fallback for missing(對映 §14.7-BM 金融業 BS 對齊問題)")
            else:
                self._detail("pass", f"G15 ROE coverage {roe_coverage*100:.1f}% ({len(self.roe_lookup)}/"
                             f"{len(self.memberships)} stocks)")
        finally:
            cur.close()
            conn.close()
        return self.stats["fail"] == 0

    # ── v0.3 Multiplier Helpers (§9.2-I.3 / §9.2-I.4) ──────────────────────

    def _compute_roe_multiplier(self, candidates):
        """v0.3 §9.2-I.3:ROE-weighted multiplier per candidate(in-place set candidate['roe_mult'])。

        若 ROE 資料 < 50% candidates 或 std=0 則 disable ROE-weighting(全 mult=1.0)。

        Args:
            candidates: long signal candidates list(每 dict 含 stock_id)

        Returns:
            None(in-place 設 candidate['roe_mult'])
        """
        roes = []
        for c in candidates:
            roe = self.roe_lookup.get(c["stock_id"])
            c["roe"] = roe
            if roe is not None:
                roes.append(roe)

        # Fallback: ROE 資料不足或全相同(std=0)→ disable ROE-weighting
        if len(roes) < max(2, len(candidates) // 2):
            for c in candidates:
                c["roe_mult"] = 1.0
            self._detail("warn", f"ROE-weighted disabled: only {len(roes)}/{len(candidates)} candidates have ROE "
                         "(fallback mult=1.0;對映 §14.7-BM 金融業 BS 對齊問題)")
            return

        roe_mean = sum(roes) / len(roes)
        roe_var = sum((r - roe_mean) ** 2 for r in roes) / len(roes)
        roe_std = roe_var ** 0.5 if roe_var > 0 else 1.0

        alpha = self.policy["roe_weight_alpha"]
        clamp_min = self.policy["roe_multiplier_clamp_min"]
        clamp_max = self.policy["roe_multiplier_clamp_max"]
        for c in candidates:
            roe = c.get("roe")
            if roe is None:
                c["roe_mult"] = 1.0  # missing ROE → neutral mult
            else:
                z = (roe - roe_mean) / roe_std
                mult = 1.0 + alpha * z
                c["roe_mult"] = max(clamp_min, min(clamp_max, mult))

    def _compute_value_multiplier(self, candidates):
        """v0.3 §9.2-I.4:Prediction value-weighted multiplier per candidate(in-place)。

        Args:
            candidates: long signal candidates list(每 dict 含 prediction_value)

        Returns:
            None(in-place 設 candidate['value_mult'])
        """
        values = [c["prediction_value"] for c in candidates]
        if len(values) < 2:
            for c in candidates:
                c["value_mult"] = 1.0
            return
        v_mean = sum(values) / len(values)
        v_var = sum((v - v_mean) ** 2 for v in values) / len(values)
        v_std = v_var ** 0.5 if v_var > 0 else 1.0

        beta = self.policy["prediction_value_weight_beta"]
        clamp_min = self.policy["value_multiplier_clamp_min"]
        clamp_max = self.policy["value_multiplier_clamp_max"]
        for c in candidates:
            z = (c["prediction_value"] - v_mean) / v_std
            mult = 1.0 + beta * z
            c["value_mult"] = max(clamp_min, min(clamp_max, mult))

    # ── Sizing Policy v0.2 ─────────────────────────────────────────────────

    def apply_policy(self):
        """依 §0.2 槓鈴策略 + §14.7-U sector cap + v0.2 G12 計算 target_weight。

        演算法（v0.2 增強）：
          1. Filter signal_label == 'long' （即 top 20 rank）
          2. 依 rank ascending 配置（最高分先得最高 cap）
          3. 每股 cap = convex_tier_weight_max（convex）或 single_stock_weight_max（core）
          4. v0.2 新增：sector_counts >= single_sector_count_max 跳過該 sector 候選 (G12)
          5. 觀察 sector_total + proposed ≤ sector_weight_max；超出則 cap 到剩餘
          6. 觀察 attack_total + proposed ≤ attack_total_weight_max；超出則 cap 到剩餘
          7. 跳過 watch / hold；bottom 20 永不配置
          8. 剩餘權重 = CASH safety sleeve
        """
        # Step 1: 過濾 long 候選
        candidates = [p for p in self.predictions if p["signal_label"] == "long"]
        candidates.sort(key=lambda x: x["prediction_rank"])  # rank 越小越優先

        if not candidates:
            raise ConstitutionalViolationError(
                gate_id="G8",
                message="no 'long' signal in prediction_values; nothing to allocate",
                charter_ref="§9.2-D / G8 / §0.2-A",
            )

        self._detail("pass", f"candidates filtered: {len(candidates)} long signals")

        # v0.3 §9.2-I.3/I.4: 計算 ROE-weighted + value-weighted multipliers(per candidate)
        self._compute_roe_multiplier(candidates)
        self._compute_value_multiplier(candidates)
        n_roe_active = sum(1 for c in candidates if c.get("roe_mult", 1.0) != 1.0)
        n_val_active = sum(1 for c in candidates if c.get("value_mult", 1.0) != 1.0)
        self._detail("pass", f"v0.3 multipliers computed: ROE-active={n_roe_active}/{len(candidates)} / "
                     f"value-active={n_val_active}/{len(candidates)}")

        attack_total = 0.0
        attack_cap = self.policy["attack_total_weight_max"]
        sector_cap = self.policy["sector_weight_max"]
        sector_count_cap = self.policy["single_sector_count_max"]
        single_cap = self.policy["single_stock_weight_max"]
        convex_cap = self.policy["convex_tier_weight_max"]
        clamp_min = self.policy["roe_multiplier_clamp_min"]
        clamp_max = self.policy["roe_multiplier_clamp_max"]

        for cand in candidates:
            stock_id = cand["stock_id"]
            member = self.memberships.get(stock_id, {})
            tier = member.get("core_tier", "core_universe")
            sector = member.get("industry_category", "UNKNOWN")
            name = member.get("stock_name", "")

            # Step 2: 決定本股 cap
            stock_cap = convex_cap if tier == "convex_universe" else single_cap
            # v0.3 §9.2-I.3/I.4: final_mult = roe_mult × value_mult(clamp to [min, max])
            roe_mult = cand.get("roe_mult", 1.0)
            value_mult = cand.get("value_mult", 1.0)
            final_mult = max(clamp_min, min(clamp_max, roe_mult * value_mult))
            proposed = stock_cap * final_mult
            # 仍受 G5/G6 cap(stock_cap 為 single_cap 或 convex_cap)
            proposed = min(proposed, stock_cap)
            reason_parts = [f"tier={tier}", f"cap={stock_cap:.0%}",
                            f"roe_mult={roe_mult:.2f}", f"val_mult={value_mult:.2f}",
                            f"final={proposed:.4f}"]
            risk = []

            base_alloc = {
                "stock_id": stock_id,
                "stock_name": name,
                "tier": tier,
                "sector": sector,
                "prediction_rank": cand["prediction_rank"],
                "prediction_value": cand["prediction_value"],
                "signal_label": cand["signal_label"],
                "confidence": cand["confidence"],
            }

            # Step 3: v0.2 新增 — G12 single-sector count cap 檢查（最先檢查）
            if self.sector_counts[sector] >= sector_count_cap:
                self.allocations.append({
                    **base_alloc,
                    "target_weight": 0.0,
                    "allocation_reason": "single_sector_count_cap_reached",
                    "risk_flags": [f"sector_{sector}_count_full_v02_G12"],
                })
                continue

            # Step 4: sector weight cap 檢查
            sector_used = self.sector_totals[sector]
            sector_available = sector_cap - sector_used
            if sector_available <= 0:
                self.allocations.append({
                    **base_alloc,
                    "target_weight": 0.0,
                    "allocation_reason": "sector_weight_cap_exceeded",
                    "risk_flags": [f"sector_{sector}_weight_full"],
                })
                continue
            if proposed > sector_available:
                proposed = sector_available
                risk.append(f"sector_{sector}_partial")

            # Step 5: attack total cap 檢查
            attack_available = attack_cap - attack_total
            if attack_available <= 0:
                self.allocations.append({
                    **base_alloc,
                    "target_weight": 0.0,
                    "allocation_reason": "attack_cap_exhausted",
                    "risk_flags": ["attack_budget_full"],
                })
                continue
            if proposed > attack_available:
                proposed = attack_available
                risk.append("attack_budget_partial")

            # Step 6: 配置
            if proposed <= 0.0001:  # 小於 1bp 視為 0
                continue
            self.allocations.append({
                **base_alloc,
                "target_weight": proposed,
                "allocation_reason": " | ".join(reason_parts),
                "risk_flags": risk,
            })
            attack_total += proposed
            self.sector_totals[sector] += proposed
            self.sector_counts[sector] += 1  # v0.2 G12 counter
            self.tier_totals[tier] += proposed

        # Step 7: CASH safety sleeve
        self.cash_weight = 1.0 - attack_total

        # Step 8: v0.3 統一 audit — 使用 audit_constraint_satisfaction(含 G13)
        ok, msg = audit_constraint_satisfaction(
            allocations=self.allocations,
            policy=self.policy,
            sector_counts=dict(self.sector_counts),
            roe_lookup=self.roe_lookup,  # v0.3: G13 ROE-weighted Pareto 檢查
        )
        if not ok:
            # 解析 gate_id 從 message 開頭
            gate_id = msg.split(":")[0].strip() if ":" in msg else "G3-G12"
            raise ConstitutionalViolationError(
                gate_id=gate_id,
                message=msg,
                charter_ref="§9.2-D / §9.2-F.1 audit_constraint_satisfaction",
            )

        n_allocated = sum(1 for a in self.allocations if a["target_weight"] > 0)
        self._detail("pass", f"sizing policy v0.2 applied: "
                     f"attack_total={attack_total:.4f}, cash={self.cash_weight:.4f}, "
                     f"allocated_stocks={n_allocated}, "
                     f"sector_counts={dict(self.sector_counts)}")

        # v0.2: 額外執行 schema + observability audit hooks
        ok_schema, msg_schema = audit_proposal_schema(
            proposal_rows=self.allocations,
            required_fields=PROPOSAL_REQUIRED_FIELDS,
        )
        if not ok_schema:
            raise ConstitutionalViolationError(
                gate_id="G_SCHEMA",
                message=msg_schema,
                charter_ref="§9.2-C / §9.2-F.1 audit_proposal_schema",
            )
        self._detail("pass", "audit_proposal_schema: OK")

        ok_log, msg_log = audit_log_observability(
            stats=self.stats,
            allocations=self.allocations,
        )
        if not ok_log:
            self._detail("warn", f"audit_log_observability: {msg_log}")
        else:
            self._detail("pass", "audit_log_observability: OK")

        return True

    # ── Output ────────────────────────────────────────────────────────────

    def _render_report(self):
        """產出 allocation proposal markdown 內容（純字串，不寫檔）"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"# Portfolio Allocation Proposal ({TOOL_VER})",
            "",
            f"- **generated_at**: {ts} Asia/Taipei",
            f"- **tool**: portfolio_sizer.py {TOOL_VER}",
            f"- **constitution**: 系統架構大憲章_{CONSTITUTION_VER}.md §9.2-A〜§9.2-H / §0.2 / §0.0-A.5",
            f"- **prediction_run_id**: `{self.run_meta['run_id']}`",
            f"- **model_id**: `{self.run_meta['model_id']}`",
            f"- **as_of_date**: {self.as_of_date}",
            f"- **universe_snapshot_id**: `{self.run_meta['universe_snapshot_id']}`",
            f"- **prediction_policy**: {self.run_meta['prediction_policy_version']}",
            f"- **sizing_policy**: {DEFAULT_SIZING_POLICY_VERSION}",
            "",
            "## 1. 配置摘要 (Allocation Summary)",
            "",
            f"| 指標 | 值 |",
            f"|---|---:|",
            f"| 攻擊端總權重 | {sum(self.tier_totals.values()):.4f} ({sum(self.tier_totals.values())*100:.2f}%) |",
            f"| 防禦端 (CASH) | {self.cash_weight:.4f} ({self.cash_weight*100:.2f}%) |",
            f"| 配置股票數 | {sum(1 for a in self.allocations if a['target_weight'] > 0)} |",
            f"| 候選但未配置 | {sum(1 for a in self.allocations if a['target_weight'] == 0)} |",
            "",
            "### Tier 配置分布",
            "",
            f"| Tier | 總權重 | 股數 |",
            f"|---|---:|---:|",
        ]
        for tier in sorted(self.tier_totals.keys()):
            n = sum(1 for a in self.allocations
                    if a["tier"] == tier and a["target_weight"] > 0)
            lines.append(f"| {tier} | {self.tier_totals[tier]:.4f} | {n} |")
        lines.append(f"| **CASH (safety)** | **{self.cash_weight:.4f}** | n/a |")

        lines += [
            "",
            "### Sector 配置分布",
            "",
            f"| Sector | 總權重 | 股數 | weight cap (≤{self.policy['sector_weight_max']}) | count cap (≤{self.policy['single_sector_count_max']}) |",
            f"|---|---:|---:|---:|---:|",
        ]
        for sec in sorted(self.sector_totals.keys()):
            if self.sector_totals[sec] > 0:
                n = self.sector_counts[sec]
                w_status = "✅" if self.sector_totals[sec] <= self.policy["sector_weight_max"] else "❌"
                c_status = "✅" if n <= self.policy["single_sector_count_max"] else "❌"
                lines.append(f"| {sec} | {self.sector_totals[sec]:.4f} | {n} | {w_status} | {c_status} |")

        lines += [
            "",
            "## 2. 配置明細 (Allocation Details)",
            "",
            "| Rank | Stock | Name | Tier | Sector | Pred Value | Signal | Conf | Weight | Reason | Risk Flags |",
            "|---:|---|---|---|---|---:|---|---:|---:|---|---|",
        ]
        # CASH row at top of details
        lines.append(
            f"| - | **CASH** | safety sleeve | safety | - | - | - | - | "
            f"**{self.cash_weight:.4f}** | §0.2 防禦端 (≥{self.policy['safety_total_weight_min']:.0%}) | - |"
        )
        for alloc in sorted(self.allocations, key=lambda a: a["prediction_rank"]):
            risk = ", ".join(alloc["risk_flags"]) if alloc["risk_flags"] else "-"
            lines.append(
                f"| {alloc['prediction_rank']} | `{alloc['stock_id']}` | "
                f"{alloc['stock_name']} | {alloc['tier']} | {alloc['sector']} | "
                f"{alloc['prediction_value']:.4f} | {alloc['signal_label']} | "
                f"{alloc['confidence']:.4f} | {alloc['target_weight']:.4f} | "
                f"{alloc['allocation_reason']} | {risk} |"
            )

        lines += [
            "",
            "## 3. Sizing Policy v0.2 規則",
            "",
            f"- attack_total_weight_max: {self.policy['attack_total_weight_max']}",
            f"- safety_total_weight_min: {self.policy['safety_total_weight_min']}",
            f"- single_stock_weight_max: {self.policy['single_stock_weight_max']}",
            f"- convex_tier_weight_max: {self.policy['convex_tier_weight_max']}",
            f"- sector_weight_max: {self.policy['sector_weight_max']}",
            f"- **single_sector_count_max: {self.policy['single_sector_count_max']}** (v0.2 G12)",
            f"- required_coverage: {self.policy['required_coverage'] if self.policy['required_coverage'] is not None else f'dynamic={len(self.memberships)} (§14.7-BW)'}",
            f"- max_committed_runs: {self.policy['max_committed_runs']}",
            "",
            "## 4. 治權邊界宣告",
            "",
            "- 本程式只讀 committed prediction_run / prediction_values / core_universe_membership；",
            "  **不**重選 universe、**不**重訓 model、**不**重算 prediction、**不**修改任何 raw 表。",
            "- 本配置為 dry-run / report proposal，**非**投資建議；signal_label='long' 為訊號標籤，",
            "  不等於買賣指令。",
            "- §0.1-A 永久禁令 #2/#3 守住：本程式未實作 IFF Θ / SOC / 重力井邊緣觸發。",
            "- §0.2 槓鈴策略：攻擊端 ≤20% + 防禦端 ≥80% + 單股 ≤5% (convex ≤3%) + sector ≤40% + single-sector count ≤5。",
            "- §0.0-G 憲章先行紀律對映：本程式為 §0.0-A.5 第五個轉換器；",
            "  治權邊界明文限定於「formal prediction → allocation proposal」。",
            "- §9.2-D.1 違憲例外契約 v0.2：所有 FAIL gate 觸發拋出 ConstitutionalViolationError。",
            "- §9.2-F.1 Audit Hooks 獨立化 v0.2：4 個 hook 為 module-level function。",
            "",
            "## 5. 執行紀錄",
            "",
        ]
        for detail in self.stats["details"]:
            lines.append(f"- {detail}")

        return "\n".join(lines) + "\n"

    def write_report(self):
        if not self.commit_report:
            return None
        content = self._render_report()
        report_path = (_PROJECT_ROOT / "reports" /
                       f"portfolio_allocation_proposal_{self.as_of_date}.md")
        report_path.write_text(content, encoding="utf-8")
        try:
            write_data_audit_log(
                "portfolio_allocation_proposal", str(self.as_of_date),
                self.as_of_date, "PORTFOLIO_SIZING_PROPOSAL",
                len(self.allocations),
            )
        except Exception as exc:
            self._detail("warn", f"data_audit_log failed: {type(exc).__name__}: {exc}")
        self._detail("pass", f"allocation proposal written: "
                     f"{report_path.relative_to(_PROJECT_ROOT)}")
        return report_path

    # ── Verdict / Report ──────────────────────────────────────────────────

    def verdict(self):
        if self.stats["fail"] > 0:
            return "FAILED"
        if self.stats["warn"] > 0:
            return "WARNING"
        return "PERFECT"

    def run(self):
        start = time.time()
        lifecycle_cm = record_lifecycle(
            f"portfolio_sizer_{TOOL_VER}", category="portfolio", stock_id="SYSTEM",
        ) if self.commit_report else None
        lifecycle = lifecycle_cm.__enter__() if lifecycle_cm else None
        try:
            print(f"💼 Portfolio Sizing Authority ({TOOL_VER} / 憲法 {CONSTITUTION_VER})")
            print("─" * 80)
            self.load_inputs()
            self.apply_policy()
            self.write_report()
            return self._finalize(start, lifecycle)
        finally:
            if lifecycle_cm:
                lifecycle_cm.__exit__(None, None, None)

    def _finalize(self, start, lifecycle):
        verdict = self.verdict()
        if lifecycle and verdict == "FAILED":
            lifecycle.mark_failed("portfolio_sizer failed")
        elif lifecycle and verdict == "WARNING":
            lifecycle.mark_warning("portfolio_sizer warning")

        # Dry-run: stdout 簡報
        if self.dry_run and self.allocations:
            print("\n" + "─" * 80)
            print("📋 Allocation Proposal (dry-run summary)")
            print("─" * 80)
            print(f"  CASH (safety): {self.cash_weight:.4f}")
            for alloc in sorted(self.allocations, key=lambda a: a["prediction_rank"]):
                if alloc["target_weight"] > 0:
                    print(f"  rank {alloc['prediction_rank']:3d} | {alloc['stock_id']:6s} "
                          f"| {alloc['tier']:16s} | {alloc['sector']:16s} | "
                          f"w={alloc['target_weight']:.4f}")

        elapsed_ms = (time.time() - start) * 1000
        print("\n" + "🛡️" * 40)
        print(f"💼 Quantum Finance: Portfolio Sizer 執行摘要 ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準  : 系統架構大憲章_{CONSTITUTION_VER}.md §9.2-A〜§9.2-H / §0.2 / §0.0-A.5")
        print(f"治理權責  : Portfolio Sizing Authority")
        print(f"執行模式  : {'COMMIT-REPORT' if self.commit_report else 'DRY-RUN'}")
        if self.run_meta:
            print(f"Source run: {self.run_meta['run_id']}")
            print(f"As-of-date: {self.as_of_date}")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"📊 PASS/WARN/FAIL : {self.stats['pass']}/{self.stats['warn']}/{self.stats['fail']}")
        if self.allocations:
            n = sum(1 for a in self.allocations if a["target_weight"] > 0)
            attack_total = sum(self.tier_totals.values())
            print(f"📈 配置股票數     : {n}")
            print(f"📈 攻擊端權重     : {attack_total:.4f} ({attack_total*100:.2f}%)")
            print(f"💰 CASH safety    : {self.cash_weight:.4f} ({self.cash_weight*100:.2f}%)")
            print(f"🏭 Sector counts  : {dict(self.sector_counts)}")
        print(f"🕒 總計耗時       : {elapsed_ms:.2f} ms")
        print(f"⚖️  主權判定       : {verdict}")
        print("🛡️" * 40 + "\n")
        return self.stats["fail"] == 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Quantum Finance Portfolio Sizer ({TOOL_VER}) — §9.2 配置層")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="只在 stdout 輸出 allocation proposal；不寫 markdown 報告")
    mode.add_argument("--commit-report", action="store_true",
                      help="寫 markdown 報告至 reports/portfolio_allocation_proposal_<asof>.md")
    parser.add_argument("--as-of-date", type=str, default=None,
                        help="指定 prediction run as-of-date（預設使用最新 committed run）")
    return parser.parse_args()


def main():
    args = parse_args()
    as_of_date = None
    if args.as_of_date:
        as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    sizer = PortfolioSizer(
        as_of_date=as_of_date,
        dry_run=args.dry_run,
        commit_report=args.commit_report,
    )
    ok = sizer.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # §9.2-D.1：CLI 層統一捕獲 ConstitutionalViolationError
    try:
        main()
    except ConstitutionalViolationError as cve:
        print("\n" + "🚨" * 40, file=sys.stderr)
        print(f"❌ 違憲攔截 (Constitutional Violation): {cve}", file=sys.stderr)
        print(f"   gate_id    = {cve.gate_id}", file=sys.stderr)
        print(f"   charter_ref= {cve.charter_ref}", file=sys.stderr)
        print("🚨" * 40, file=sys.stderr)
        sys.exit(1)
