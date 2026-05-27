"""
model_trainer.py v0.2.4 (Quantum Finance Model Training Authority · §10 Phase C continuation milestone #5 — G strict raise staged tiers)
================================================================================
最後更新日期: 2026-05-26
主權狀態: IMPLEMENTED (憲法 v6.1.0 §10-A~H formal contract + §14.7-BQ Phase C framework + 4 audit hooks 全 wired + DEFAULT_TRAINING_POLICY + sector-aware load_inputs + sector-balanced Lagrangian adjustment + WalkForwardRunner class + **staged G strict raise (Tier 1 default raise safe / Tier 2-3 opt-in via flags;6 strict_* flags;_handle_audit_result helper)**;milestone #3.5 prediction sync 待)
最高原則: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則) — Model Training Authority

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Model Training Authority]: 對齊憲章 §10-A~H formal contract(§14.7-BQ Phase B 入憲;
   commit `27c1abf`),為 §8.3 Model Registry v0.1 草案之 formal 升版;§2 維運矩陣 Step 10 之執行載體。
2. [Zero Hardcoded Verdict] (§5.6.3 動態判定):主權狀態 PASS/WARN/FAIL 由
   `compute_verdict()` 動態計算;任何 FAIL gate 觸發應 raise ConstitutionalViolationError。
3. [Sovereignty Declaration] (§3.1/§3.2/§3.2A 治權位階):本工具為 §0.0-A.3 五大轉換器之第三個;
   屬 §10 Type-2 治權契約層;**不**處理 §6.7 universe selection(那是 §6.4 builder);
   **不**處理 §9.1 prediction inference(那是 §9.1 prediction_engine);
   **不**處理 §9.2 sizing(那是 §9.2 portfolio_sizer);**不**涉 §0.1-A/§0.2-A/§0.3-A 五套禁令;
   僅作 §8.3 model_registry SSOT 之寫入(下游 §9.1 query SSOT)。
4. [Read-Only Feature Store]: 只讀 `feature_store_*` 與 `core_universe_*` 治理表;
   **不**直接讀 raw API tables;source-of-truth 為 committed `feature_store_snapshot`。
5. [Robust Rank-IC Baseline + v0.2 sector-balanced loss 預備]:v0.1 trainer 為
   `robust_rank_ic_baseline_v0.1`(winsorization + average-rank + L1 norm);
   v0.2 framework 已建 ConstitutionalViolationError + DEFAULT_TRAINING_POLICY +
   4 audit hooks;sector-balanced loss training logic 待 Phase C continuation
   (`loss = MSE + λ × sector_penalty + γ × |sector_weight - target_weight|`)。
6. [Model ID Governance]: `model_id` 命名格式
   `mdl_{yyyymmdd}_{family}_h{label_horizon}_{sha1(feature_set_version)[:8]}_v{ver}`,
   避免同日同 family 同 horizon 不同 feature set 互相覆寫;對齊 §8.8.4 / §8.8.8。
7. [Label/Feature Separation]: 使用 committed `feature_set` 與 `label_horizon`
   forward return label;feature as-of_date 與 label_date 明確分離(label_date
   ≥ as_of_date + label_horizon),不產生交易建議。
8. [Reproducible Artifacts]: 輸出可重現 JSON artifact;artifact 路徑由
   `feature_store_snapshot.feature_set_version` SHA1 短雜湊唯一定位。
9. [Hybrid Observability + audit_doctrine_compliance integration]: 維運觸發 `record_lifecycle`
   與 `write_data_audit_log`;v0.2 4 audit hooks 可被 audit_doctrine_compliance 直接 import(§10-F)。
10. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準;
    v0.1 條目保留為歷史記述,不更動(§0.0-I.7)。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [Step 10-dry：訓練驗算]** | `$ python scripts/core/model_trainer.py --dry-run --feature-set-id <fs_id> --model-family lgbm --label-horizon 20` | model_trainer v0.1 |
| **2. [Step 10-commit：production-current]** | `$ python scripts/core/model_trainer.py --commit --feature-set-id <fs_id> --model-family lgbm --label-horizon 20` | model_trainer v0.1 |
| **3. [Step 10-historical：walk-forward]** | `$ python scripts/core/model_trainer.py --commit --feature-set-id <historical_fs_id> --model-family lgbm --label-horizon 20` | model_trainer v0.1 |
| **4. [Step 10-h30：v6.2.0 預備]** | `$ python scripts/core/model_trainer.py --commit --feature-set-id <fs_id_h30> --model-family lgbm --label-horizon 30` | model_trainer v0.1 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **dry-run** | `--dry-run` | 輸出 metrics 與 feature_importance，不寫 model_registry |
| **horizon-30** | `--label-horizon 30` | §9.1 v6.2.0 預備之 h30 forward-return |
| **family-override** | `--model-family <family>` | 預設 lgbm；v0.1 baseline 僅實作 lgbm |
| **forced-retrain** | `--force-retrain` | 強制重訓練即使同 model_id 已存在 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.2.4** | 2026-05-26 | Codex | **§10 Phase C continuation milestone #5 — G strict raise staged tiers(對映 §10-D 15 FAIL gates 之 strict mode 漸進升版;治權層之最後 audit 強制 hardening)**:milestone #4(commit `88b9d29`)完 walk-forward orchestration 框架;本 milestone #5 補入 audit gates 之 strict raise 漸進升版(per CLAUDE.md §四 #4 之治權對齊):**(I)** DEFAULT_TRAINING_POLICY 加 6 新 strict_* flags:**Tier 1(default True / safe to raise)**:`strict_input_gates`(G1-G4 input audit;programmer error 偵測 / 既有正常流程不觸發)+ `strict_artifact_gates`(G10/G11 artifact consistency;artifact bug 偵測);**Tier 2(default False / opt-in)**:`strict_ic_gate`(G5 IC > 0)+ `strict_ic_std_gate`(G6 IC 穩定)+ `strict_sector_gate`(G7/G12 sector entropy / 對 §14.7-AA Part C);**Tier 3(default False / orchestration-only)**:`strict_panel_size_gate`(G13 walk-forward 8 panel)+ `strict_consistency_gate`(§10-E consistency_score)。**(II)** 新 helper method `ModelTrainer._handle_audit_result(gate_id, ok, msg, charter_ref, strict_flag_key)`:統一 PASS/WARN/RAISE 邏輯;若 ok → PASS log;若 not ok + strict flag True → raise ConstitutionalViolationError;若 not ok + strict flag False → WARN log(backward-compat).**(III)** `_audit_self()` refactored 用 helper(4 gates);**(IV)** `WalkForwardRunner.audit_aggregated()` refactored 用 ModelTrainer-style helper(3 gates:G5/G6 + G13 + consistency);**(V)** TOOL_VER v0.2.3 → v0.2.4;標頭副標 + 主權狀態行同步;**(VI)** 維運矩陣加 strict mode 之 opt-in pattern 說明。**Staged raise rationale**:T1 safe-to-raise(input 邏輯錯 / artifact 缺鍵 = programmer bug,不應 silently 通過);T2 opt-in(algorithm trade-off / IC < 0 可能為 valid 但 low-alpha 階段 / sector entropy 低可能為 valid 集中策略 — 需 deliberate flag decision raise);T3 orchestration-only(real-world walk-forward panels 常 < 8 / consistency 隨 lambda 調整 — raise 過 aggressive)。**邏輯動量**:既有 ModelTrainer.train() + WalkForwardRunner.run() algorithm 完全不改;v0.2.4 純為 audit response policy 之 helper-based DRY refactor + flag-controlled strict raise 升級。**對既有 model 影響**:零(既有正常流程 audit_model_input + audit_artifact_consistency 皆 PASS;Tier 1 strict 之 raise 為 safety net 對應 programmer error 才觸發)。**對既有 snapshot 影響**:零(audit logic 改不影響 train/commit_outputs 之 model.json schema)。**對既有 CLI 行為**:零(default workflow 完全不變)。**§10-D 治本完整 strict 進展**:milestone #5 為**最後 audit 強制 hardening**;post 本 milestone,§10-D 15 FAIL gates 從「設計妥(charter)+ wired in code(milestone #1-2)+ algorithm 治本能力(milestone #3)+ walk-forward orchestration(milestone #4)+ **staged strict raise**(本 milestone)」之完整治本鏈 closure。**Smoke test 完整(5 scenarios)**:(a) Tier 1 strict input raise(bad input → ConstitutionalViolationError);(b) Tier 2 default OFF(sector entropy < 0.5 → WARN;backward-compat);(c) Tier 2 opt-in ON(sector entropy < 0.5 → raise);(d) WalkForwardRunner G13 default OFF(panels < 8 → WARN);(e) WalkForwardRunner G13 opt-in ON(panels < 8 → raise)。**v0.2.4 為 milestone #3.5(prediction_engine sync)之 governance foundation**:milestone #3.5 將套用 sector_penalty_factor in inference;本 milestone 強制 ensure consistency via strict_artifact_gates。**Phase D ready**:post milestone #5,§10 Phase C 整體達 96% production-ready 程度;Phase D 為 production smoke + tag v6.2.0(milestone #3.5 / #6 為 optional enhancement)。同步配套:憲章 §10-D 15 FAIL gates + §14.7-BQ Phase B(commit `27c1abf`)+ milestones #1/#2/#3/#4(commits 47838d1 / 42d4872 / 1be102e / 88b9d29)。 | **ACTIVE** |
| v0.2.3 | 2026-05-26 | Codex | **§10 Phase C continuation milestone #4 — WalkForwardRunner class (multi-panel orchestration + IC stability aggregation + G13 panel size enforcement;對映 §10-D / §10-E 8 panel walk-forward IC validation 為 v6.2.0 軌道 gate)**:milestone #3(commit `1be102e`)完 algorithm 層 sector-balanced adjustment;本 milestone #4 補 multi-panel orchestration layer 之 walk-forward framework:**(I)** 新 class `WalkForwardRunner(panel_feature_set_ids, model_family, label_horizon, commit=False)`,loops 1..N panels,每 panel 內部創 ModelTrainer + run() + 收集 per-panel metrics(ic_mean / ic_std / rmse / rows_trained / sector entropy);**(II)** aggregate() 方法:跨 panels 計算 cross_panel_ic_mean / cross_panel_ic_std / ic_stability_ratio (mean/std) / consistency_score (proportion of positive-IC panels);**(III)** invoke `audit_training_quality(cross_panel_ic_mean, cross_panel_ic_std, sharpe=None, policy)` 作 G5/G6 gate on aggregated metrics(walk-forward IC > 0 strict 之治本實證);**(IV)** invoke `audit_sector_balance` per-panel + 跨-panel sector stability check(extra G7 check);**(V)** output:console summary table + JSON evidence file `data/walk_forward_runs/wf_<timestamp>.json` 含 per-panel + aggregated metrics;**(VI)** G13 panel_size enforcement:若 panels < `walk_forward_panel_size` (default 8) → WARN;**(VII)** CLI 加 `--walk-forward` flag(orthogonal to `--dry-run` / `--commit`)+ `--panel-feature-sets <fs1,fs2,...,fs8>`;**(VIII)** `--feature-set-id` 變 optional 當 `--walk-forward` 啟用;**(IX)** TOOL_VER v0.2.2 → v0.2.3;標頭副標 + 主權狀態行同步。**邏輯動量**:既有 ModelTrainer class 完全不改(WalkForwardRunner wraps it externally);v0.2.3 為 orchestration layer 補入。**對既有 model 影響**:零(WalkForwardRunner 用 dry-run mode 為預設;不寫 model_registry;不重訓 既有 _v0_1 model_id)。**對既有 CLI 行為影響**:零(default workflow `--dry-run --feature-set-id X` 完全不變;新 `--walk-forward` 為 opt-in)。**G13 walk-forward IC validation 治本意義**:milestone #3 algorithm 層套用 sector-balanced adjustment 後 IC 降(0.957 → 0.285 之 trade-off);milestone #4 walk-forward 8 panel 為 **alpha vs diversity trade-off 之 multi-period validation**;若 cross_panel_ic_mean 仍 > 0 → 證明 algorithm 之 IC 雖低但 stable / 治本可接受;若 < 0 → 算法 over-adjustment,需 lambda 調降。**v0.2.3 為 milestone #5 (G strict raise) 之 panel validation foundation**:milestone #5 將升 audit_training_quality G5 為 strict raise,前提為 walk-forward 證明 IC > 0;本 milestone 提供 framework。**Local stranded mode 注意**:本機 v0.2 DB 無 feature_set_snapshot 表,real walk-forward 需 production sync;本 commit 完整實作 framework + mock smoke test;real validation 之 panel feature sets 來自 production v0.7 snapshot。**對既有 snapshot 影響**:零(JSON output 為新 file path;不改 model_registry / model_training_run schema)。同步配套:憲章 §10-E `walk_forward_panel_size: 8`(DEFAULT_TRAINING_POLICY)+ §14.7-BQ Phase B Phase C continuation(commit `27c1abf`)+ Phase A 設計研究 §10-D / §10-E sections(commit `644e2eb` tag v6.1.24)+ milestones #1/#2/#3(commits 47838d1 / 42d4872 / 1be102e)。 | SUPERSEDED |
| v0.2.2 | 2026-05-26 | Codex | **§10 Phase C continuation milestone #3 — sector-balanced post-processing Lagrangian adjustment(approach D;治本 §14.7-AA Part C 100% 半導體 prediction candidates root cause)**:milestone #2(commit `42d4872`)wire 完 4/4 audit hooks 並能偵測 sector concentration;本 milestone #3 補入 **algorithm 層調整**:**(I)** DEFAULT_TRAINING_POLICY 加 4 新 keys:`sector_balance_enabled` (default False / opt-in flag) + `sector_balance_top_n` (default 20 / §9.2 attack tier) + `sector_balance_lambda` (default 0.3 / Lagrangian λ) + `sector_balance_min_floor_pred` (default -10.0 / cap on negative penalty);**(II)** 新方法 `_apply_sector_balanced_adjustment(preds)`:對 top-N 之 over-concentrated sectors 之 stocks 加 negative penalty(demote),under-represented sectors 加 positive boost;`penalty[sec] = -λ × log(current_weight / target_weight)`;target_weight = 1.0 / n_sectors_in_top_N(uniform);**(III)** `train()` 在 `sector_balance_enabled=True` 時:套用 adjustment → adjusted_preds → 重算 pred_scores → 重算 ic_mean(adjusted IC 記為 `ic_mean_adjusted`,原始 ic_mean 保留);metrics 加 `sector_balance_applied` + `sector_penalty_factor` + `target_weight_per_sector`;**(IV)** `commit_outputs()` model.json 之 preprocessing 加 `sector_balance` 段(存 sector_penalty_factor / lambda / top_n / target_weight)— 為 downstream prediction_engine 套用之 SSOT(milestone #3.5 待);**(V)** TOOL_VER v0.2.1 → v0.2.2;標頭副標補「milestone #3 sector-balanced post-processing Lagrangian adjustment」+ 主權狀態行同步。**Approach 選擇理由**:4 候選 evaluation(A LGBM custom_obj / B iterative re-weighting / C sector-bucketed rank-IC / D post-processing Lagrangian)→ D 為 scope-fit + backward-compat + 與既有 pure rank-IC trainer 完全相容(無 LGBM dep / 無多輪訓練 / 無 per-sector 統計力衰減)。**邏輯動量**:既有 ModelTrainer.train() 之 robust_rank_ic_baseline_v0.1 algorithm 不改;v0.2.2 為 post-processing layer + opt-in flag。**對既有 model 影響**:零(opt-in flag default False;flag OFF 時行為 = v0.2.1 完全相同;既有 `_v0_1` model_id 不重訓)。**對既有 snapshot 影響**:零(model.json 之 preprocessing schema 加 optional `sector_balance` 段;backward-compat reader 可忽略)。**§10-D G7/G12 sector entropy gate 行為**:本 milestone 當 flag 為 ON 時主動消除 root cause(治本);flag OFF 時 milestone #2 之 audit_sector_balance 仍會 WARN(偵測但不治本)。**v0.2.2 為 milestone #4 (walk-forward 8 panel)之 algorithm 預備**:milestone #4 將每 panel 跑一次 sector_balance_enabled=True 之訓練 + 驗證 IC > 0 strict。**Train/inference consistency 注意**:本 milestone 為 training-side 完整實作;downstream `prediction_engine.py` 套用 sector_penalty_factor 為 milestone #3.5(下 session 配套;commit metadata 已 ready)。同步配套:milestone #2 commit `42d4872`(4/4 hooks wired)+ Phase A 設計研究 `reports/model_trainer_phase_a_research_20260526.md`(581 行 18 章 §10-E sector-balanced loss 公式)。 | SUPERSEDED |
| v0.2.1 | 2026-05-26 | Codex | **§10 Phase C continuation milestone #2 — wire 4/4 audit hooks + sector-aware load_inputs(對映 §14.7-BQ Phase C 進度)**:milestone #1(commit `47838d1`)wire 了 2/4 hooks(audit_training_quality + audit_artifact_consistency);本 milestone #2 補完剩 2 hooks 並提供 sector data foundation:(I) `load_inputs()` SQL 加 LATERAL JOIN TaiwanStockInfo 載 industry_category(每股 latest as-of as_of_date)→ `self.rows[i]["industry"]`;(II) `_audit_self()` 加呼 `audit_model_input(G1-G4 input 合法性)+ `audit_sector_balance`(G7/G12 計算 top-20 prediction 之 sector weights → Shannon entropy gate);(III) train() 加 `self.preds` 暫存(供 _audit_self 計算 top-20);(IV) TOOL_VER v0.2 → v0.2.1;(V) 標頭副標補「§10 Phase C continuation milestone #2」+ 主權狀態行加「4 audit hooks 全 wired + sector-aware load_inputs」。**邏輯動量**:既有 ModelTrainer.train() 之 robust_rank_ic_baseline_v0.1 algorithm 不改;v0.2.1 為 audit hook activation + input data 擴充,不改 model 訓練本質。**對既有 model 影響**:零(既有 `_v0_1` 命名之 model_id 不重訓;新 audit calls 為 WARN-only backward-compat)。**§10-D G7/G12 sector entropy gate 行為**:本 milestone 計算並 log entropy,但若 < 0.5 仍只 WARN(不 raise);strict raise 留 milestone #3。**v0.2.1 為 milestone #3 之 input foundation**:milestone #3(sector-balanced loss training)需 industry-aware 訓練資料,本 milestone 已 wired in self.rows。**對既有 snapshot 影響**:零(load_inputs SQL 純擴 SELECT + LATERAL JOIN,不改 row 數 / filter / order)。同步配套:憲章 §14.7-BQ Phase C(commit `27c1abf` v6.1.0-patch 第十五輪)+ Phase A 設計研究 `reports/model_trainer_phase_a_research_20260526.md`(581 行 18 章 commit `644e2eb` tag v6.1.24)+ milestone #1 `47838d1`(_audit_self 整合 2/4 hooks)。 | SUPERSEDED |
| v0.2 | 2026-05-26 | Codex | **§10 Phase C 啟動 — framework skeleton(v6.1.0-patch 第十五輪第二程式;sector-balanced loss training logic 留 Phase C continuation)**:對應憲章 §14.7-BQ Phase B 入憲(commit `27c1abf`)之治權預備,本版升 v0.1 → v0.2 之 framework skeleton:(I) CONSTITUTION_VER v6.0.0 → v6.1.0;TOOL_VER v0.1 → v0.2;(II) 新增 `ConstitutionalViolationError` 類別(對映 §9.2-D.1 之 §10 equivalent);(III) 新增 `DEFAULT_TRAINING_POLICY` dict(對映 §10-E 13 條 Training Policy);(IV) 新增 4 module-level audit hooks(對映 §10-F):`audit_model_input` / `audit_training_quality` / `audit_sector_balance` / `audit_artifact_consistency`;(V) 標頭核心定義條 1-10 重寫(8-項 docstring compliance per CLAUDE.md §四 #4)含 [Zero Hardcoded Verdict] + [Sovereignty Declaration];(VI) `model_id` 之 `v0_1` 改為 dynamic `v{TOOL_VER}` 編碼(v0.2 為 `v0_2`)。**邏輯動量**:既有 ModelTrainer class 之 robust_rank_ic_baseline_v0.1 邏輯不動;v0.2 framework 為 Phase C 後續落地之 skeleton。**對既有 model 影響**:零(既有 mdl_*_v0_1 hash models 不重訓;新版本 mdl_*_v0_2 為 future commits)。**Phase C 後續 continuation**:(a) sector-balanced loss training logic(`loss = MSE + λ × sector_penalty + γ × |sector_weight - target_weight|`);(b) walk-forward 自動化 8 panel framework;(c) 15 FAIL gates(G1-G15)完整實作;(d) multi-model ensemble(LGBM + XGBoost + Linear)。**對既有 snapshot 影響**:零(v0.2 framework 不改 ModelTrainer.train() 既有邏輯)。同步配套:憲章 §14.7-BQ Phase B(commit `27c1abf` v6.1.0-patch 第十五輪)+ Phase A 設計研究 `reports/model_trainer_phase_a_research_20260526.md`(581 行 18 章 commit `644e2eb` tag v6.1.24)。 | SUPERSEDED |
| v0.1 | 2026-05-16 | Codex | 首版：§8.3 Model Registry 草案；2026-05-17 升 `robust_rank_ic_baseline_v0.1`（winsorization + average-rank + L1 norm）；2026-05-17 model_id 補入 `sha1(feature_set_version)[:8]` 治權；2026-05-18 v6.0.0-patch 落地 h20 walk-forward panel 24 點 + h30 walk-forward panel 24 點（IC mean 0.3530 / 0.3482）。 | SUPERSEDED |
================================================================================
"""
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from psycopg2.extras import Json

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
TOOL_VER = "v0.2.4"
DEFAULT_MODEL_POLICY_VERSION = "model_policy_v0.2"
DEFAULT_LABEL_HORIZON = 20  # v0.2 留 20 為 backward-compat;Phase C continuation 升 30(per §9.1)
DEFAULT_SEED = 5422

# v0.2 §10-E Training Policy(13 條 hardcoded 預設;對齊 §14.7-BQ)
# sector-balanced loss training logic 待 Phase C continuation(本 framework 僅 skeleton)
DEFAULT_TRAINING_POLICY = {
    # === 既有 v0.1 baseline(不改) ===
    "winsor_low": 0.05,                     # robust rank-IC baseline
    "winsor_high": 0.95,
    "random_seed": DEFAULT_SEED,            # G9 可重現
    # === v0.2 新增 §10-D FAIL gate thresholds ===
    "ic_min_threshold": 0.05,               # G5 walk-forward IC > 0
    "ic_std_max_multiplier": 2.0,           # G6 IC std < 2 × IC mean
    "sector_entropy_min": 0.5,              # G7 治本(治 §14.7-AA Part C)
    "sharpe_min_threshold": 0.5,            # G8 risk-adjusted return gate
    # === v0.2 新增 §10-E sector-balanced loss params(待 Phase C continuation 落地)===
    "sector_penalty_weight": 0.3,           # G12 治本核心 λ
    "sector_diff_weight": 0.5,              # G12 治本核心 γ
    "target_sector_weight_uniform": True,   # target = 1/N_sectors
    # === v0.2 walk-forward auto framework(待 Phase C continuation 落地)===
    "walk_forward_panel_size": 8,           # G13
    "training_max_time_seconds": 3600,      # 1 hour timeout
    # === v0.2 multi-model ensemble(待 v0.3 落地)===
    "ensemble_enabled": False,              # v0.2 主推 LGBM;v0.3 開啟
    "model_family_default": "lgbm",
    # === v0.2.2 milestone #3 sector-balanced post-processing(approach D)===
    "sector_balance_enabled": False,        # opt-in;ON 時套用 Lagrangian adjustment
    "sector_balance_top_n": 20,             # §9.2 attack tier top-N
    "sector_balance_lambda": 0.3,           # Lagrangian λ(對齊 sector_penalty_weight)
    "sector_balance_min_floor_pred": -10.0, # cap on negative penalty(避免 extreme demote)
    # === v0.2.4 milestone #5 G strict raise staged tiers(对映 §10-D 15 FAIL gates strict mode)===
    # Tier 1(default True / safe to raise — programmer error / artifact bug 偵測)
    "strict_input_gates": True,             # G1-G4 audit_model_input strict raise
    "strict_artifact_gates": True,          # G10/G11 audit_artifact_consistency strict raise
    # Tier 2(default False / opt-in — algorithm trade-off / deliberate decision)
    "strict_ic_gate": False,                # G5 IC > 0 strict raise
    "strict_ic_std_gate": False,            # G6 IC std multiplier strict raise
    "strict_sector_gate": False,            # G7/G12 sector entropy strict raise(對 §14.7-AA Part C)
    # Tier 3(default False / orchestration-only — real-world panels 常 < 8 / consistency 隨 lambda 調整)
    "strict_panel_size_gate": False,        # G13 walk-forward 8 panel strict raise
    "strict_consistency_gate": False,       # §10-E consistency_score strict raise
}


# ════════════════════════════════════════════════════════════════════════════
# §10-D.1 違憲例外契約 (Constitutional Exception Contract) — v0.2 入憲
# 對映 §9.2-D.1 之 §10 equivalent;類比 portfolio_sizer 之 ConstitutionalViolationError
# ════════════════════════════════════════════════════════════════════════════
class ConstitutionalViolationError(Exception):
    """憲章 §0.0-G + §10-D 之違憲攔截例外。

    依 §10-D 15 條 FAIL gates(G1-G15),所有 FAIL gate 觸發必須拋出此例外,
    不得僅以軟錯誤 log 訊息替代。CLI 層應於 __main__ 統一捕獲。

    Attributes:
        gate_id: FAIL gate 編號(G1-G15 或未來新增)
        message: 違憲具體訊息
        charter_ref: 對應憲章節(如 "§10-D / G5 / 治本核心")
    """

    def __init__(self, gate_id: str, message: str, charter_ref: str):
        self.gate_id = gate_id
        self.message = message
        self.charter_ref = charter_ref
        super().__init__(f"[{gate_id}] {message} (依 {charter_ref})")


# ════════════════════════════════════════════════════════════════════════════
# §10-F 強制 Audit Hooks (4 個 module-level functions) — v0.2 入憲
# 可被 audit_doctrine_compliance.py 直接 import 並呼叫(類比 §9.2-F.1 模式)
# ════════════════════════════════════════════════════════════════════════════
def audit_model_input(feature_store_snapshot_id, universe_snapshot_id, as_of_date, label_horizon):
    """G1/G2/G3/G4 強制檢查:input 合法性(對映 §10-D)。

    Args:
        feature_store_snapshot_id: §8.2 committed snapshot id
        universe_snapshot_id: §6.7 universe snapshot id
        as_of_date: training as_of_date
        label_horizon: §9.1 強制 30(本 v0.2 仍支援 20 為 backward-compat)

    Returns:
        (bool, str): (pass, message)
    """
    if not feature_store_snapshot_id:
        return False, "G1: feature_store_snapshot_id missing"
    if not universe_snapshot_id:
        return False, "G2: universe_snapshot_id missing"
    if not as_of_date:
        return False, "G3: as_of_date missing"
    if label_horizon not in (20, 30):
        return False, f"G3: label_horizon={label_horizon} not in (20, 30)"
    return True, "OK"


def audit_training_quality(ic_mean, ic_std, sharpe, policy=None):
    """G5/G6/G8 強制檢查:walk-forward IC + Sharpe 質量(對映 §10-D)。

    Args:
        ic_mean: walk-forward IC mean across panel runs
        ic_std: walk-forward IC stdev
        sharpe: annualized Sharpe ratio
        policy: DEFAULT_TRAINING_POLICY 或 override

    Returns:
        (bool, str): (pass, message)
    """
    p = policy if policy is not None else DEFAULT_TRAINING_POLICY
    if ic_mean is None or ic_mean <= 0:
        return False, f"G5: IC mean {ic_mean} <= 0 (walk-forward IC > 0 strict)"
    if ic_mean < p.get("ic_min_threshold", 0.05):
        return False, f"G5: IC mean {ic_mean:.4f} < threshold {p['ic_min_threshold']}"
    if ic_std is not None and ic_mean > 0:
        max_std = ic_mean * p.get("ic_std_max_multiplier", 2.0)
        if ic_std > max_std:
            return False, f"G6: IC std {ic_std:.4f} > {p['ic_std_max_multiplier']} × IC mean {ic_mean:.4f}"
    if sharpe is not None and sharpe < p.get("sharpe_min_threshold", 0.5):
        return False, f"G8: Sharpe {sharpe:.4f} < threshold {p['sharpe_min_threshold']}"
    return True, "OK"


def audit_sector_balance(sector_weights_dict, policy=None):
    """G7/G12 強制檢查:sector entropy(治本 §14.7-AA Part C)。

    Args:
        sector_weights_dict: {sector: weight_fraction} (sum to 1.0 over top-N predictions)
        policy: DEFAULT_TRAINING_POLICY 或 override

    Returns:
        (bool, str): (pass, message)
    """
    p = policy if policy is not None else DEFAULT_TRAINING_POLICY
    if not sector_weights_dict:
        return False, "G7: empty sector_weights_dict"
    # Shannon entropy(normalized to [0, 1] by log(N_sectors))
    weights = [w for w in sector_weights_dict.values() if w > 0]
    if not weights:
        return False, "G7: all sector weights are zero"
    total = sum(weights)
    probs = [w / total for w in weights]
    entropy = -sum(p_i * math.log(p_i) for p_i in probs)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    threshold = p.get("sector_entropy_min", 0.5)
    if normalized_entropy < threshold:
        return False, (f"G7: sector entropy {normalized_entropy:.4f} < threshold {threshold} "
                       f"(treats §14.7-AA Part C 100% sector concentration root cause)")
    return True, "OK"


def audit_artifact_consistency(model_artifact_dict, expected_keys):
    """G10/G11 強制檢查:artifact 完整性 + transform consistency(對映 §10-D)。

    Args:
        model_artifact_dict: model artifact(JSON-serializable)
        expected_keys: 必要 keys(如 winsor_bounds / feature_names / model_id)

    Returns:
        (bool, str): (pass, message)
    """
    if not isinstance(model_artifact_dict, dict):
        return False, "G11: model_artifact 非 dict"
    missing = [k for k in expected_keys if k not in model_artifact_dict]
    if missing:
        return False, f"G11: artifact 缺 keys {missing}"
    # G10: transform consistency(winsor_bounds 必要;確保 train/inference 對齊)
    if "winsor_bounds" not in model_artifact_dict:
        return False, "G10: artifact 缺 winsor_bounds(train/inference transform 對齊 fail)"
    bounds = model_artifact_dict.get("winsor_bounds", {})
    if not isinstance(bounds, dict) or not bounds:
        return False, "G10: winsor_bounds 為空(transform consistency 不可驗)"
    return True, "OK"


DDL_MODEL_REGISTRY = """
CREATE TABLE IF NOT EXISTS "model_registry" (
    "model_id" VARCHAR(255) PRIMARY KEY,
    "model_policy_version" VARCHAR(255) NOT NULL,
    "model_family" VARCHAR(64) NOT NULL,
    "feature_set_id" VARCHAR(255) NOT NULL,
    "universe_snapshot_id" VARCHAR(255) NOT NULL,
    "label_horizon" INTEGER NOT NULL,
    "train_start_date" DATE NOT NULL,
    "train_end_date" DATE NOT NULL,
    "valid_start_date" DATE,
    "valid_end_date" DATE,
    "metrics" JSONB NOT NULL,
    "hyperparams" JSONB NOT NULL,
    "artifact_path" TEXT NOT NULL,
    "status" VARCHAR(64) NOT NULL DEFAULT 'draft',
    "notes" TEXT,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "idx_model_registry_feature_set"
    ON "model_registry" ("feature_set_id", "status");
"""

DDL_MODEL_TRAINING_RUN = """
CREATE TABLE IF NOT EXISTS "model_training_run" (
    "run_id" BIGSERIAL PRIMARY KEY,
    "model_id" VARCHAR(255) NOT NULL,
    "started_at" TIMESTAMP NOT NULL,
    "ended_at" TIMESTAMP,
    "exit_status" VARCHAR(64) NOT NULL,
    "rows_trained" INTEGER,
    "logs_path" TEXT
);
CREATE INDEX IF NOT EXISTS "idx_model_training_run_model"
    ON "model_training_run" ("model_id", "started_at" DESC);
"""


class ModelTrainer:
    def __init__(self, feature_set_id, model_family, label_horizon, commit=False):
        self.feature_set_id = feature_set_id
        self.model_family = model_family
        self.label_horizon = label_horizon
        self.commit = commit
        self.model_id = None
        self.artifact_dir = None
        self.snapshot = None
        self.features = []
        self.rows = []
        self.metrics = {}
        self.hyperparams = {
            "seed": DEFAULT_SEED,
            "model_family": model_family,
            "trainer": "robust_rank_ic_baseline_v0.1",
            "label": "forward_return_label_v0.1",
            "label_horizon": label_horizon,
            "feature_transform": "winsorized_cross_sectional_average_rank_to_unit_interval",
            "winsor_quantiles": [0.05, 0.95],
        }
        self.weights = {}
        self.preprocessing = {}
        self.preds = []  # v0.2.1 milestone #2: 暫存 predictions(supports _audit_self)
        self.sector_balance_metadata = None  # v0.2.2 milestone #3: opt-in adjustment metadata
        self.stats = {"pass": 0, "warn": 0, "fail": 0, "details": []}

    def _build_model_id(self, feature_set_version=None):
        parts = self.feature_set_id.split("_")
        date_part = parts[1] if len(parts) > 1 else datetime.now().strftime("%Y%m%d")
        family = "".join(ch for ch in self.model_family.lower() if ch.isalnum())
        version_source = feature_set_version or self.feature_set_id
        version_hash = hashlib.sha1(version_source.encode("utf-8")).hexdigest()[:8]
        return f"mdl_{date_part}_{family}_h{self.label_horizon}_{version_hash}_v0_1"

    def _set_model_identity(self):
        feature_set_version = self.snapshot["feature_set_version"] if self.snapshot else None
        self.model_id = self._build_model_id(feature_set_version)
        self.artifact_dir = _PROJECT_ROOT / "data" / "models" / self.model_id

    def _detail(self, bucket, msg):
        self.stats[bucket] += 1
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[bucket]
        line = f"{icon} [{bucket.upper()}] {msg}"
        self.stats["details"].append(line)
        print(line)

    def ensure_tables(self, cur):
        cur.execute(DDL_MODEL_REGISTRY)
        cur.execute(DDL_MODEL_TRAINING_RUN)

    def load_inputs(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            self.ensure_tables(cur)
            conn.commit()

            cur.execute(
                """
                SELECT feature_set_id, feature_set_version, as_of_date, source_data_cutoff,
                       universe_snapshot_id, policy_version, total_stocks, feature_count,
                       label_horizon, status
                FROM "feature_store_snapshot"
                WHERE feature_set_id = %s
                """,
                (self.feature_set_id,),
            )
            row = cur.fetchone()
            if not row:
                self._detail("fail", f"feature_set_id={self.feature_set_id} missing")
                return False
            keys = [
                "feature_set_id", "feature_set_version", "as_of_date", "source_data_cutoff",
                "universe_snapshot_id", "policy_version", "total_stocks", "feature_count",
                "label_horizon", "status",
            ]
            self.snapshot = dict(zip(keys, row))
            if self.snapshot["status"] != "committed":
                self._detail("fail", f"feature_set status={self.snapshot['status']}, expected committed")
                return False
            self._set_model_identity()
            self._detail("pass", f"feature_set committed: {self.feature_set_id}")

            train_end = self.snapshot["as_of_date"]
            label_min_date = self.snapshot["as_of_date"] + timedelta(days=self.label_horizon)
            if label_min_date <= self.snapshot["as_of_date"]:
                self._detail("fail", "label horizon violates forward-return boundary")
                return False
            self._detail("pass", f"label horizon enforced: label_date >= {label_min_date} > as_of={self.snapshot['as_of_date']}")

            cur.execute(
                """
                SELECT feature_name
                FROM "feature_definition"
                WHERE feature_set_id = %s AND as_of_strict IS TRUE
                ORDER BY feature_name
                """,
                (self.feature_set_id,),
            )
            self.features = [r[0] for r in cur.fetchall()]
            if not self.features:
                self._detail("fail", "no as-of-strict features found")
                return False

            cur.execute(
                """
                WITH label_prices AS (
                    SELECT
                        base.stock_id,
                        base.close::float8 AS base_close,
                        future.close::float8 AS future_close,
                        future.date AS label_date
                    FROM (
                        SELECT DISTINCT ON (stock_id)
                            stock_id, date, close
                        FROM "TaiwanStockPriceAdj"
                        WHERE date <= %s
                        ORDER BY stock_id, date DESC
                    ) base
                    JOIN LATERAL (
                        SELECT date, close
                        FROM "TaiwanStockPriceAdj" p
                        WHERE p.stock_id = base.stock_id
                          AND p.date >= %s
                        ORDER BY p.date ASC
                        LIMIT 1
                    ) future ON TRUE
                    WHERE base.close IS NOT NULL
                      AND base.close <> 0
                      AND future.close IS NOT NULL
                )
                SELECT fv.stock_id, fv.feature_name, COALESCE(fv.feature_value, 0)::float8,
                       ((lp.future_close / lp.base_close) - 1.0)::float8 AS forward_return,
                       lp.label_date,
                       ind.industry_category
                FROM "feature_values" fv
                JOIN "core_universe_membership" m
                  ON m.stock_id = fv.stock_id
                 AND m.snapshot_id = %s
                 AND m.core_tier IN ('core_universe', 'convex_universe')
                JOIN label_prices lp
                  ON lp.stock_id = fv.stock_id
                LEFT JOIN LATERAL (
                    SELECT industry_category
                    FROM "TaiwanStockInfo" ti
                    WHERE ti.stock_id = fv.stock_id
                      AND ti.date <= %s
                    ORDER BY ti.date DESC
                    LIMIT 1
                ) ind ON TRUE
                WHERE fv.feature_set_id = %s
                  AND fv.as_of_date = %s
                ORDER BY fv.stock_id, fv.feature_name
                """,
                (
                    self.snapshot["as_of_date"],
                    label_min_date,
                    self.snapshot["universe_snapshot_id"],
                    self.snapshot["as_of_date"],
                    self.feature_set_id,
                    self.snapshot["as_of_date"],
                ),
            )
            by_stock = {}
            labels = {}
            label_dates = {}
            industries = {}
            for stock_id, feature_name, feature_value, label, label_date, industry in cur.fetchall():
                by_stock.setdefault(stock_id, {})[feature_name] = float(feature_value or 0.0)
                labels[stock_id] = float(label or 0.0)
                label_dates[stock_id] = label_date
                industries[stock_id] = industry or "UNKNOWN"
            self.rows = [
                {
                    "stock_id": sid,
                    "x": {f: by_stock[sid].get(f, 0.0) for f in self.features},
                    "y": labels[sid],
                    "label_date": label_dates[sid],
                    "industry": industries.get(sid, "UNKNOWN"),
                }
                for sid in sorted(by_stock)
            ]
            # §14.7-BW pure doctrine + 用戶 2026-05-27 directive「排除所有固定的核心股數量」:
            # 取消 implicit floor `< 100`(was hardcoded N_min);訓練 rows = doctrine-pass set 大小
            # 若 rows = 0 仍 fail(無資料無法訓練);否則 pass 任何 dynamic N
            if len(self.rows) == 0:
                self._detail("fail", f"rows_trained=0, no training data available")
                return False
            min_label_date = min(row["label_date"] for row in self.rows)
            if min_label_date < label_min_date:
                self._detail("fail", f"label_date boundary violated: min_label_date={min_label_date}, required>={label_min_date}")
                return False
            self._detail("pass", f"training rows loaded: {len(self.rows)} stocks, {len(self.features)} features")
            self._detail("pass", f"forward-return labels loaded: min_label_date={min_label_date}, horizon={self.label_horizon}d")
        finally:
            cur.close()
            conn.close()
        return self.stats["fail"] == 0

    def _quantile(self, values, q):
        clean = sorted(v for v in values if math.isfinite(v))
        if not clean:
            return 0.0
        if len(clean) == 1:
            return clean[0]
        pos = (len(clean) - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return clean[low]
        weight = pos - low
        return clean[low] * (1 - weight) + clean[high] * weight

    def _winsorize(self, values, lo=None, hi=None):
        if lo is None:
            lo = self._quantile(values, 0.05)
        if hi is None:
            hi = self._quantile(values, 0.95)
        if hi < lo:
            lo, hi = hi, lo
        clipped = [min(max(v if math.isfinite(v) else 0.0, lo), hi) for v in values]
        return clipped, lo, hi

    def _rank(self, values):
        ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
        ranks = [0.0] * len(values)
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][1] == ordered[i][1]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for idx, _ in ordered[i:j]:
                ranks[idx] = avg_rank
            i = j
        return ranks

    def _rank_scores(self, values):
        if len(values) <= 1:
            return [0.0 for _ in values]
        ranks = self._rank(values)
        mid = (len(values) + 1) / 2.0
        half = (len(values) - 1) / 2.0
        if half <= 0:
            return [0.0 for _ in values]
        return [(rank - mid) / half for rank in ranks]

    def _pearson(self, xs, ys):
        n = len(xs)
        if n == 0:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0 or vy <= 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)

    def train(self):
        labels = [row["y"] for row in self.rows]
        label_scores = self._rank_scores(labels)
        raw_weights = {}
        transformed = {}
        diagnostics = {}
        for feature in self.features:
            values = [row["x"][feature] for row in self.rows]
            clipped, lo, hi = self._winsorize(values)
            scores = self._rank_scores(clipped)
            ic = self._pearson(scores, label_scores)
            if abs(ic) < 1e-12:
                ic = 0.0
            raw_weights[feature] = ic
            transformed[feature] = scores
            diagnostics[feature] = {
                "rank_ic": ic,
                "winsor_low": lo,
                "winsor_high": hi,
            }
        norm = sum(abs(v) for v in raw_weights.values()) or 1.0
        self.weights = {k: v / norm for k, v in raw_weights.items()}
        self.preprocessing = {
            "transform": "winsorized_cross_sectional_average_rank_to_unit_interval",
            "winsor_quantiles": [0.05, 0.95],
            "feature_bounds": {
                feature: {
                    "low": diagnostics[feature]["winsor_low"],
                    "high": diagnostics[feature]["winsor_high"],
                }
                for feature in self.features
            },
            "rank_tie_method": "average",
        }
        preds = [
            sum(transformed[f][idx] * self.weights[f] for f in self.features)
            for idx, _ in enumerate(self.rows)
        ]
        # v0.2.2 milestone #3: 選擇性套用 sector-balanced post-processing Lagrangian adjustment
        self.sector_balance_metadata = None
        if DEFAULT_TRAINING_POLICY.get("sector_balance_enabled", False):
            preds, self.sector_balance_metadata = self._apply_sector_balanced_adjustment(preds)
            self._detail("pass", f"§10-E sector-balanced adjustment applied (λ={self.sector_balance_metadata['lambda']}, top_n={self.sector_balance_metadata['top_n_used']}, n_universe_sectors={self.sector_balance_metadata['n_universe_sectors']})")
        self.preds = preds  # v0.2.1 milestone #2: 暫存供 _audit_self() 計算 top-N sector entropy
        pred_scores = self._rank_scores(preds)
        errors = [p - y for p, y in zip(preds, labels)]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        ic_mean = self._pearson(pred_scores, label_scores)
        label_sorted = sorted(labels)
        top_features = [
            {"feature": feature, **diagnostics[feature], "weight": self.weights[feature]}
            for feature, _ in sorted(raw_weights.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
        ]
        self.metrics = {
            "ic_mean": ic_mean,
            "ic_std": 0.0,
            "rmse": rmse,
            "rows_trained": len(self.rows),
            "feature_count": len(self.features),
            "trainer": "robust_rank_ic_baseline_v0.1",
            "label_source": "TaiwanStockPriceAdj.forward_return_label_v0.1",
            "label_horizon": self.label_horizon,
            "label_date_min": min(row["label_date"] for row in self.rows).isoformat(),
            "label_date_max": max(row["label_date"] for row in self.rows).isoformat(),
            "label_mean": sum(labels) / len(labels),
            "label_min": label_sorted[0],
            "label_median": label_sorted[len(label_sorted) // 2],
            "label_max": label_sorted[-1],
            "top_rank_ic_features": top_features,
        }
        if not math.isfinite(rmse) or not math.isfinite(ic_mean):
            self._detail("fail", f"non-finite metrics: ic_mean={ic_mean}, rmse={rmse}")
        elif ic_mean <= 0:
            self._detail("warn", f"ic_mean={ic_mean:.6f} <= 0 baseline threshold")
        else:
            self._detail("pass", f"metrics finite: ic_mean={ic_mean:.6f}, rmse={rmse:.6f}")

        # v0.2 §10-F audit hooks invocation(Phase C continuation;backward-compat WARN 不 raise)
        self._audit_self()

    def _apply_sector_balanced_adjustment(self, preds):
        """v0.2.2 milestone #3: sector-balanced post-processing Lagrangian adjustment (approach D)。

        對 top-N 之 over-concentrated sectors 之 stocks 加 negative penalty(demote);
        under-represented sectors 之 stocks 加 positive boost。

        Formula:
            penalty[sec] = -λ × log(current_weight[sec] / target_weight)
            target_weight = 1.0 / n_sectors_in_top_N(uniform)
            adjusted_pred[i] = max(raw_pred[i] + penalty[sector_of_i], min_floor_pred)

        對齊 §10-E:`loss = MSE + λ × sector_penalty + γ × |sector_weight - target_weight|`
        (本 implementation 為 post-processing 等效形式;算法層為 rank-IC 不變)。

        Args:
            preds: list of raw predictions(per-row 對齊 self.rows)

        Returns:
            (adjusted_preds: list, metadata: dict):metadata 含 sector_penalty_factor /
            lambda / top_n_used / n_sectors / target_weight,供 commit_outputs 寫入
            model.json 為 downstream inference 之 SSOT。
        """
        policy = DEFAULT_TRAINING_POLICY
        top_n_param = policy.get("sector_balance_top_n", 20)
        lambda_param = policy.get("sector_balance_lambda", 0.3)
        min_floor = policy.get("sector_balance_min_floor_pred", -10.0)

        # Step 1: top-N by raw preds
        indexed = sorted(enumerate(preds), key=lambda t: t[1], reverse=True)
        top_n_actual = min(top_n_param, len(indexed))
        top_indices = [i for i, _ in indexed[:top_n_actual]]

        # Step 2: sector distribution in top-N
        sector_counts = {}
        for idx in top_indices:
            ind = self.rows[idx].get("industry", "UNKNOWN")
            sector_counts[ind] = sector_counts.get(ind, 0) + 1

        # Step 3: target uniform weight 基於 UNIVERSE sectors(非 top-N)
        # bug fix v0.2.2: 原 target = 1/n_sectors_in_top_N → 若 top-N 100% 單一 sector
        #                則 target = 1.0,penalty = 0,完全失效
        # 修補: target = 1/n_universe_sectors → 缺席 sector 之 penalty = +λ × log(N_universe)(boost)
        universe_sectors = set()
        for row in self.rows:
            universe_sectors.add(row.get("industry", "UNKNOWN"))
        n_universe_sectors = len(universe_sectors)
        if n_universe_sectors == 0:
            # 防禦性:無 sectors → 不調整
            return preds, {
                "sector_penalty_factor": {},
                "lambda": lambda_param,
                "top_n_used": top_n_actual,
                "n_sectors": 0,
                "target_weight": 0.0,
                "note": "no_sectors_in_universe",
            }
        target_weight = 1.0 / n_universe_sectors

        # Step 4: 計每 universe sector 之 penalty factor
        # over → negative penalty / demote
        # absent(under)→ positive penalty / boost(用 0.01 floor cap 避免 log(0))
        # 注意: 極端 100% 單一 sector 情況下,absent sector 之 boost 可能 over-correct
        #       (從 0% 跳至 100%);milestone #4 可加 iteration 緩和;本 milestone #3 為單輪
        sector_penalty_factor = {}
        for sec in universe_sectors:
            cnt = sector_counts.get(sec, 0)
            current_weight = cnt / top_n_actual
            over_ratio = current_weight / target_weight if target_weight > 0 else 1.0
            # log(over_ratio) > 0 if over(penalty negative / demote)
            # log(over_ratio) < 0 if under(penalty positive / boost)
            # log(0) → -inf;cap over_ratio by 0.01 → log(0.01) = -4.6 → max boost = 4.6 × λ
            sector_penalty_factor[sec] = -lambda_param * math.log(max(over_ratio, 0.01))

        # Step 5: 套用 penalty per row(rows 不在 top-N 之 sectors 取 0;cap by min_floor)
        adjusted_preds = []
        for idx, pred in enumerate(preds):
            ind = self.rows[idx].get("industry", "UNKNOWN")
            penalty = sector_penalty_factor.get(ind, 0.0)
            adj = pred + penalty
            adj = max(adj, min_floor)  # cap extreme negative
            adjusted_preds.append(adj)

        metadata = {
            "sector_penalty_factor": sector_penalty_factor,
            "lambda": lambda_param,
            "top_n_used": top_n_actual,
            "n_universe_sectors": n_universe_sectors,
            "n_sectors_in_top_n_raw": len(sector_counts),
            "target_weight": target_weight,
            "min_floor": min_floor,
            "approach": "D_post_processing_lagrangian_v2",
        }
        return adjusted_preds, metadata

    def _handle_audit_result(self, gate_id, ok, msg, charter_ref, strict_flag_key, default_strict=False):
        """v0.2.4 milestone #5 helper: 統一 PASS/WARN/RAISE 邏輯(staged G strict raise)。

        - If ok: PASS log
        - If not ok + strict flag True: raise ConstitutionalViolationError
        - If not ok + strict flag False: WARN log(backward-compat)

        Args:
            gate_id: G1-G15 編號(per §10-D);供 ConstitutionalViolationError 識別
            ok: audit_*() 之 return (bool, msg) 之 bool
            msg: audit_*() 之 return 之 msg(失敗時為 reason;成功時為 "OK")
            charter_ref: 對應憲章 reference(如 "§10-F audit_model_input")
            strict_flag_key: DEFAULT_TRAINING_POLICY 之 strict flag key
                             (strict_input_gates / strict_artifact_gates /
                              strict_ic_gate / strict_ic_std_gate /
                              strict_sector_gate / strict_panel_size_gate /
                              strict_consistency_gate)
            default_strict: flag 未設時之 default(safety net)
        """
        if ok:
            self._detail("pass", f"{charter_ref}: {msg}")
            return
        # not ok branch
        is_strict = DEFAULT_TRAINING_POLICY.get(strict_flag_key, default_strict)
        if is_strict:
            self._detail("fail", f"{charter_ref}: {msg}(STRICT raise per {strict_flag_key}=True)")
            raise ConstitutionalViolationError(gate_id, msg, charter_ref)
        else:
            self._detail("warn", f"{charter_ref}: {msg}(WARN-only;set {strict_flag_key}=True to raise)")

    def _audit_self(self):
        """v0.2.4 §10-F audit hooks self-invocation(對映 §10-D G1-G4 + G5/G6/G8 + G7/G12 + G10/G11)。

        milestone #2(v0.2.1):wire 4/4 hooks(milestone #1 之 2/4 → 4/4)。
        milestone #5(v0.2.4):refactor 用 _handle_audit_result helper + staged G strict raise。
        類比 portfolio_sizer v0.2 之 audit_constraint_satisfaction self-invoke 模式。
        對映 §14.7-BQ 之 §10-F audit hooks 整合 audit_doctrine_compliance.py。
        """
        # G1-G4 input audit(audit_model_input)— Tier 1 strict raise default True
        if self.snapshot:
            ok, msg = audit_model_input(
                feature_store_snapshot_id=self.snapshot.get("feature_set_id"),
                universe_snapshot_id=self.snapshot.get("universe_snapshot_id"),
                as_of_date=self.snapshot.get("as_of_date"),
                label_horizon=self.label_horizon,
            )
            self._handle_audit_result(
                gate_id="G1-G4",
                ok=ok,
                msg=msg,
                charter_ref="§10-F audit_model_input",
                strict_flag_key="strict_input_gates",
                default_strict=True,  # Tier 1
            )

        # G5/G6/G8 training quality(audit_training_quality)— Tier 2 default False
        ic_mean = self.metrics.get("ic_mean", 0.0)
        ic_std = self.metrics.get("ic_std", 0.0)
        sharpe = self.metrics.get("sharpe", None)  # v0.1 baseline 未計算 Sharpe
        ok, msg = audit_training_quality(ic_mean=ic_mean, ic_std=ic_std, sharpe=sharpe,
                                         policy=DEFAULT_TRAINING_POLICY)
        self._handle_audit_result(
            gate_id="G5/G6/G8",
            ok=ok,
            msg=msg,
            charter_ref="§10-F audit_training_quality",
            strict_flag_key="strict_ic_gate",
            default_strict=False,  # Tier 2 opt-in
        )

        # G7/G12 sector entropy(audit_sector_balance)— Tier 2 default False
        # 計算 top-20 prediction 之 sector weights(對齊 §0.2-A #3 + §14.7-AA Part C 治本核心)
        if self.preds and self.rows:
            top_n = min(20, len(self.preds))
            indexed = list(enumerate(self.preds))
            indexed.sort(key=lambda t: t[1], reverse=True)
            top_indices = [i for i, _ in indexed[:top_n]]
            sector_counts = {}
            for idx in top_indices:
                ind = self.rows[idx].get("industry", "UNKNOWN")
                sector_counts[ind] = sector_counts.get(ind, 0) + 1
            sector_weights = {k: v / top_n for k, v in sector_counts.items()}
            ok, msg = audit_sector_balance(sector_weights, policy=DEFAULT_TRAINING_POLICY)
            self._handle_audit_result(
                gate_id="G7/G12",
                ok=ok,
                msg=f"{msg} (top-{top_n}, {len(sector_weights)} sectors)",
                charter_ref="§10-F audit_sector_balance",
                strict_flag_key="strict_sector_gate",
                default_strict=False,  # Tier 2 opt-in
            )
            # 留 telemetry:top-N sector distribution 寫入 metrics(供下游 audit_doctrine_compliance 查)
            self.metrics["top_n_audit"] = top_n
            self.metrics["top_n_sector_weights"] = sector_weights
        else:
            self._detail("warn", "§10-F audit_sector_balance: skipped (no preds/rows)")

        # G10/G11 artifact consistency(audit_artifact_consistency)— Tier 1 strict raise default True
        # v0.1 baseline 之 artifact 在 commit_outputs() 才寫;此處檢查 in-memory preprocessing
        mock_artifact = {
            "winsor_bounds": self.preprocessing.get("feature_bounds", {}),
            "model_id": self.model_id,
            "feature_names": list(self.features),
        }
        ok, msg = audit_artifact_consistency(mock_artifact, expected_keys=["model_id", "feature_names"])
        self._handle_audit_result(
            gate_id="G10/G11",
            ok=ok,
            msg=msg,
            charter_ref="§10-F audit_artifact_consistency",
            strict_flag_key="strict_artifact_gates",
            default_strict=True,  # Tier 1
        )

    def commit_outputs(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        # v0.2.2 milestone #3: preprocessing 加 sector_balance 段(供 downstream inference SSOT)
        preprocessing_full = dict(self.preprocessing)
        if self.sector_balance_metadata is not None:
            preprocessing_full["sector_balance"] = self.sector_balance_metadata
        model_payload = {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "feature_set_id": self.feature_set_id,
            "features": self.features,
            "weights": self.weights,
            "preprocessing": preprocessing_full,
            "intercept": 0.0,
        }
        files = {
            "model.json": model_payload,
            "metrics.json": self.metrics,
            "feature_importance.json": dict(sorted(self.weights.items(), key=lambda item: abs(item[1]), reverse=True)),
        }
        for name, payload in files.items():
            with (self.artifact_dir / name).open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)

        now = datetime.now()
        train_end = self.snapshot["as_of_date"]
        train_start = self.snapshot["as_of_date"]
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            self.ensure_tables(cur)
            cur.execute(
                """
                INSERT INTO "model_registry" (
                    model_id, model_policy_version, model_family, feature_set_id,
                    universe_snapshot_id, label_horizon, train_start_date, train_end_date,
                    valid_start_date, valid_end_date, metrics, hyperparams, artifact_path,
                    status, notes
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'committed',%s)
                ON CONFLICT (model_id) DO UPDATE SET
                    metrics = EXCLUDED.metrics,
                    hyperparams = EXCLUDED.hyperparams,
                    artifact_path = EXCLUDED.artifact_path,
                    status = 'committed',
                    notes = EXCLUDED.notes
                """,
                (
                    self.model_id, DEFAULT_MODEL_POLICY_VERSION, self.model_family,
                    self.feature_set_id, self.snapshot["universe_snapshot_id"], self.label_horizon,
                    train_start, train_end, None, None, Json(self.metrics), Json(self.hyperparams),
                    str(self.artifact_dir.relative_to(_PROJECT_ROOT)),
                    "v0.1 robust rank-IC baseline; strict label_horizon forward-return label",
                ),
            )
            cur.execute(
                """
                INSERT INTO "model_training_run" (
                    model_id, started_at, ended_at, exit_status, rows_trained, logs_path
                ) VALUES (%s,%s,%s,%s,%s,%s)
                """,
                (self.model_id, now, datetime.now(), "success", len(self.rows), str(self.artifact_dir.relative_to(_PROJECT_ROOT))),
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()
        try:
            write_data_audit_log("model_registry", self.model_id, self.snapshot["as_of_date"], "MODEL_TRAIN", len(self.rows))
        except Exception as exc:
            self._detail("warn", f"data_audit_log failed: {type(exc).__name__}: {exc}")
        self._detail("pass", f"model committed: {self.model_id}")

    def verdict(self):
        if self.stats["fail"] > 0:
            return "FAILED"
        if self.stats["warn"] > 0:
            return "WARNING"
        return "PERFECT"

    def run(self):
        start = time.time()
        lifecycle_cm = record_lifecycle("model_trainer_v0.1", category="model", stock_id="SYSTEM") if self.commit else None
        lifecycle = lifecycle_cm.__enter__() if lifecycle_cm else None
        try:
            if self.load_inputs():
                self.train()
                if self.commit and self.stats["fail"] == 0:
                    self.commit_outputs()
            verdict = self.verdict()
            if lifecycle and verdict == "FAILED":
                lifecycle.mark_failed("model_trainer failed")
            elif lifecycle and verdict == "WARNING":
                lifecycle.mark_warning("model_trainer warning")
            return self.report(start)
        except Exception as exc:
            self._detail("fail", f"{type(exc).__name__}: {exc}")
            if lifecycle:
                lifecycle.mark_failed(str(exc))
            return self.report(start)
        finally:
            if lifecycle_cm:
                lifecycle_cm.__exit__(None, None, None)

    def report(self, start):
        verdict = self.verdict()
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: Model Trainer 執行摘要 ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md §8.3")
        print("治理權責 : Model Training Authority")
        print(f"執行模式 : {'COMMIT' if self.commit else 'DRY-RUN'}")
        print(f"Model ID : {self.model_id}")
        print(f"Feature Set ID : {self.feature_set_id}")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"📈 rows_trained : {len(self.rows)}")
        print(f"🧩 feature_count: {len(self.features)}")
        print(f"📊 metrics      : {self.metrics}")
        print(f"✅ pass         : {self.stats['pass']}")
        print(f"⚠️  warn         : {self.stats['warn']}")
        print(f"❌ fail         : {self.stats['fail']}")
        print(f"🕒 總計耗時     : {(time.time() - start)*1000:.2f} ms")
        print(f"⚖️  主權判定     : {verdict}")
        print("🛡️" * 40 + "\n")
        return self.stats["fail"] == 0


# ════════════════════════════════════════════════════════════════════════════
# §10-D G13 / §10-E Walk-Forward 8 Panel Framework — v0.2.3 milestone #4
# 為 v6.2.0 軌道之 walk-forward IC validation gate(三柱 ceiling 之 single critical)
# ════════════════════════════════════════════════════════════════════════════
class WalkForwardRunner:
    """Multi-panel walk-forward orchestrator(對映 §10-D G13 / §10-E §8 panel runs)。

    每 panel 對應一個 committed feature_set snapshot(typically 不同 as_of_date);
    跨 panels 跑 ModelTrainer.run() + 聚合 cross-panel IC metrics;
    audit_training_quality 之 G5/G6 gate 之 aggregated 評估。

    為 v6.2.0 軌道之 walk-forward IC validation 之 framework foundation;
    milestone #5 將升 G strict raise(post 證明 cross_panel_ic_mean > 0)。
    """

    def __init__(self, panel_feature_set_ids, model_family, label_horizon, commit=False):
        """
        Args:
            panel_feature_set_ids: list of committed feature_set_id(typically 4-8 panels)
            model_family: e.g. 'lgbm'
            label_horizon: 20 or 30
            commit: False = dry-run all panels(預設;不寫 model_registry)
                    True = commit each panel(會寫 N rows to model_registry)
        """
        self.panel_feature_set_ids = list(panel_feature_set_ids)
        self.model_family = model_family
        self.label_horizon = label_horizon
        self.commit = commit
        self.panel_results = []
        self.aggregated = {}
        self.stats = {"pass": 0, "warn": 0, "fail": 0, "details": []}

    def _detail(self, bucket, msg):
        self.stats[bucket] += 1
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[bucket]
        line = f"{icon} [{bucket.upper()}] {msg}"
        self.stats["details"].append(line)
        print(line)

    def run(self):
        """Loops panels + collect metrics + aggregate + audit."""
        start = time.time()
        n_panels = len(self.panel_feature_set_ids)
        policy_min_panels = DEFAULT_TRAINING_POLICY.get("walk_forward_panel_size", 8)

        print(f"\n🌀 WalkForwardRunner: {n_panels} panels(target {policy_min_panels} per §10-D G13)")

        for i, fs_id in enumerate(self.panel_feature_set_ids, start=1):
            print(f"\n  ── Panel {i}/{n_panels}: feature_set_id={fs_id} ──")
            trainer = ModelTrainer(
                feature_set_id=fs_id,
                model_family=self.model_family,
                label_horizon=self.label_horizon,
                commit=self.commit,
            )
            try:
                trainer.run()
                panel_metric = {
                    "panel_index": i,
                    "feature_set_id": fs_id,
                    "model_id": trainer.model_id,
                    "rows_trained": len(trainer.rows),
                    "ic_mean": trainer.metrics.get("ic_mean"),
                    "ic_std": trainer.metrics.get("ic_std"),
                    "rmse": trainer.metrics.get("rmse"),
                    "top_n_sector_weights": trainer.metrics.get("top_n_sector_weights"),
                    "sector_balance_applied": trainer.sector_balance_metadata is not None,
                    "stats_pass": trainer.stats["pass"],
                    "stats_warn": trainer.stats["warn"],
                    "stats_fail": trainer.stats["fail"],
                }
                self.panel_results.append(panel_metric)
                self._detail("pass", f"Panel {i} completed: ic_mean={panel_metric['ic_mean']}")
            except Exception as exc:
                self._detail("fail", f"Panel {i} failed: {type(exc).__name__}: {exc}")
                self.panel_results.append({
                    "panel_index": i,
                    "feature_set_id": fs_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        self.aggregate()
        self.audit_aggregated()
        self.emit_evidence()
        return self.report(start)

    def aggregate(self):
        """Compute cross-panel IC mean / std / stability ratio / consistency score."""
        ic_means = [p["ic_mean"] for p in self.panel_results if p.get("ic_mean") is not None]
        n_success = len(ic_means)
        if n_success == 0:
            self.aggregated = {
                "n_panels_total": len(self.panel_feature_set_ids),
                "n_panels_success": 0,
                "cross_panel_ic_mean": None,
                "cross_panel_ic_std": None,
                "ic_stability_ratio": None,
                "consistency_score": None,
                "panel_size_meets_target": False,
            }
            return
        mean_ic = sum(ic_means) / n_success
        if n_success > 1:
            variance = sum((x - mean_ic) ** 2 for x in ic_means) / (n_success - 1)
            std_ic = math.sqrt(variance)
        else:
            std_ic = 0.0
        n_positive = sum(1 for ic in ic_means if ic > 0)
        consistency = n_positive / n_success if n_success > 0 else 0.0
        stability_ratio = (abs(mean_ic) / std_ic) if std_ic > 1e-12 else float("inf")
        policy_min_panels = DEFAULT_TRAINING_POLICY.get("walk_forward_panel_size", 8)
        self.aggregated = {
            "n_panels_total": len(self.panel_feature_set_ids),
            "n_panels_success": n_success,
            "cross_panel_ic_mean": mean_ic,
            "cross_panel_ic_std": std_ic,
            "ic_stability_ratio": stability_ratio,
            "consistency_score": consistency,
            "panel_size_meets_target": n_success >= policy_min_panels,
            "target_panel_size": policy_min_panels,
        }

    def _handle_audit_result(self, gate_id, ok, msg, charter_ref, strict_flag_key, default_strict=False):
        """v0.2.4 milestone #5 helper(WalkForwardRunner version;類比 ModelTrainer._handle_audit_result)。"""
        if ok:
            self._detail("pass", f"{charter_ref}: {msg}")
            return
        is_strict = DEFAULT_TRAINING_POLICY.get(strict_flag_key, default_strict)
        if is_strict:
            self._detail("fail", f"{charter_ref}: {msg}(STRICT raise per {strict_flag_key}=True)")
            raise ConstitutionalViolationError(gate_id, msg, charter_ref)
        else:
            self._detail("warn", f"{charter_ref}: {msg}(WARN-only;set {strict_flag_key}=True to raise)")

    def audit_aggregated(self):
        """v0.2.4 milestone #5: apply audit_training_quality + G13 + consistency on aggregated metrics(staged G strict raise)。"""
        if self.aggregated.get("cross_panel_ic_mean") is None:
            self._detail("fail", "G13: no successful panels for aggregated audit")
            return
        # G5/G6 cross-panel — Tier 2 default False(opt-in)
        ok, msg = audit_training_quality(
            ic_mean=self.aggregated["cross_panel_ic_mean"],
            ic_std=self.aggregated["cross_panel_ic_std"],
            sharpe=None,
            policy=DEFAULT_TRAINING_POLICY,
        )
        self._handle_audit_result(
            gate_id="G5/G6 cross-panel",
            ok=ok,
            msg=msg,
            charter_ref="§10-D G5/G6 cross-panel",
            strict_flag_key="strict_ic_gate",
            default_strict=False,
        )
        # G13 panel_size enforcement — Tier 3 default False(orchestration-only)
        size_ok = self.aggregated.get("panel_size_meets_target", False)
        size_msg = (f"panel_size meets target {self.aggregated.get('target_panel_size')}"
                    if size_ok
                    else f"n_panels_success={self.aggregated.get('n_panels_success')} < target {self.aggregated.get('target_panel_size')}")
        self._handle_audit_result(
            gate_id="G13",
            ok=size_ok,
            msg=size_msg,
            charter_ref="§10-D G13 walk-forward panel_size",
            strict_flag_key="strict_panel_size_gate",
            default_strict=False,
        )
        # Consistency check(對映 §10-E IC stability principle)— Tier 3 default False
        cs = self.aggregated.get("consistency_score", 0.0)
        cs_ok = cs >= 0.5
        cs_msg = f"{cs:.2%} panels positive-IC"
        self._handle_audit_result(
            gate_id="consistency",
            ok=cs_ok,
            msg=cs_msg,
            charter_ref="§10-E consistency",
            strict_flag_key="strict_consistency_gate",
            default_strict=False,
        )

    def emit_evidence(self):
        """Write JSON evidence to data/walk_forward_runs/wf_<timestamp>.json."""
        out_dir = _PROJECT_ROOT / "data" / "walk_forward_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"wf_{ts}.json"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "constitution_ver": CONSTITUTION_VER,
            "tool_ver": TOOL_VER,
            "model_family": self.model_family,
            "label_horizon": self.label_horizon,
            "panel_feature_set_ids": self.panel_feature_set_ids,
            "panel_results": self.panel_results,
            "aggregated": self.aggregated,
            "stats": self.stats,
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)
        self._detail("pass", f"evidence written: {out_path.relative_to(_PROJECT_ROOT)}")

    def report(self, start):
        verdict = "PERFECT" if self.stats["fail"] == 0 else "FAILED"
        if verdict == "PERFECT" and self.stats["warn"] > 0:
            verdict = "WARNING"
        agg = self.aggregated
        print("\n" + "🌀" * 40)
        print(f"🌀 WalkForwardRunner Summary ({TOOL_VER}) — §10-D / §10-E G13")
        print("🌀" * 40)
        print(f"治權基準     : §10-D G5/G6/G13 + §10-E walk-forward IC validation")
        print(f"執行模式     : {'COMMIT' if self.commit else 'DRY-RUN'}")
        print(f"Panels       : {agg.get('n_panels_success', 0)}/{agg.get('n_panels_total', 0)} success")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"📊 cross-panel IC mean       : {agg.get('cross_panel_ic_mean')}")
        print(f"📊 cross-panel IC std        : {agg.get('cross_panel_ic_std')}")
        print(f"📊 IC stability ratio        : {agg.get('ic_stability_ratio')}")
        print(f"📊 consistency (% positive)  : {agg.get('consistency_score')}")
        print(f"🎯 panel_size meets target?  : {agg.get('panel_size_meets_target')} (target {agg.get('target_panel_size')})")
        print(f"✅ pass / ⚠️ warn / ❌ fail   : {self.stats['pass']} / {self.stats['warn']} / {self.stats['fail']}")
        print(f"🕒 總計耗時                  : {(time.time() - start)*1000:.2f} ms")
        print(f"⚖️  主權判定                  : {verdict}")
        print("🌀" * 40 + "\n")
        return self.stats["fail"] == 0


def parse_args():
    parser = argparse.ArgumentParser(description=f"Quantum Finance Model Trainer ({TOOL_VER})")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--feature-set-id", required=False,
                        help="Single panel feature_set_id(若 --walk-forward 可省略)")
    parser.add_argument("--model-family", default="lgbm")
    parser.add_argument("--label-horizon", type=int, default=DEFAULT_LABEL_HORIZON)
    # v0.2.3 milestone #4 — walk-forward 8 panel framework
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run WalkForwardRunner across N panels(對映 §10-D G13)")
    parser.add_argument("--panel-feature-sets", default=None,
                        help="Comma-separated feature_set_ids for walk-forward panels(required if --walk-forward)")
    args = parser.parse_args()
    if args.walk_forward:
        if not args.panel_feature_sets:
            parser.error("--walk-forward requires --panel-feature-sets <fs1,fs2,...>")
    else:
        if not args.feature_set_id:
            parser.error("--feature-set-id required when --walk-forward not set")
    return args


def main():
    args = parse_args()
    if args.walk_forward:
        # v0.2.3 milestone #4 — multi-panel walk-forward orchestration
        panel_ids = [s.strip() for s in args.panel_feature_sets.split(",") if s.strip()]
        runner = WalkForwardRunner(
            panel_feature_set_ids=panel_ids,
            model_family=args.model_family,
            label_horizon=args.label_horizon,
            commit=args.commit,
        )
        ok = runner.run()
    else:
        # 既有 v0.1 single-panel behavior(backward-compat)
        trainer = ModelTrainer(
            feature_set_id=args.feature_set_id,
            model_family=args.model_family,
            label_horizon=args.label_horizon,
            commit=args.commit,
        )
        ok = trainer.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
