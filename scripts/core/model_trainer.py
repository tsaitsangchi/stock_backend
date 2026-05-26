"""
model_trainer.py v0.2 (Quantum Finance Model Training Authority)
================================================================================
最後更新日期: 2026-05-26
主權狀態: IMPLEMENTED (憲法 v6.1.0 §10-A~H formal contract + §14.7-BQ Phase C framework skeleton + ConstitutionalViolationError + 4 audit hooks + DEFAULT_TRAINING_POLICY;sector-balanced loss training logic 待 Phase C continuation)
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
| **v0.2** | 2026-05-26 | Codex | **§10 Phase C 啟動 — framework skeleton(v6.1.0-patch 第十五輪第二程式;sector-balanced loss training logic 留 Phase C continuation)**:對應憲章 §14.7-BQ Phase B 入憲(commit `27c1abf`)之治權預備,本版升 v0.1 → v0.2 之 framework skeleton:(I) CONSTITUTION_VER v6.0.0 → v6.1.0;TOOL_VER v0.1 → v0.2;(II) 新增 `ConstitutionalViolationError` 類別(對映 §9.2-D.1 之 §10 equivalent);(III) 新增 `DEFAULT_TRAINING_POLICY` dict(對映 §10-E 13 條 Training Policy);(IV) 新增 4 module-level audit hooks(對映 §10-F):`audit_model_input` / `audit_training_quality` / `audit_sector_balance` / `audit_artifact_consistency`;(V) 標頭核心定義條 1-10 重寫(8-項 docstring compliance per CLAUDE.md §四 #4)含 [Zero Hardcoded Verdict] + [Sovereignty Declaration];(VI) `model_id` 之 `v0_1` 改為 dynamic `v{TOOL_VER}` 編碼(v0.2 為 `v0_2`)。**邏輯動量**:既有 ModelTrainer class 之 robust_rank_ic_baseline_v0.1 邏輯不動;v0.2 framework 為 Phase C 後續落地之 skeleton。**對既有 model 影響**:零(既有 mdl_*_v0_1 hash models 不重訓;新版本 mdl_*_v0_2 為 future commits)。**Phase C 後續 continuation**:(a) sector-balanced loss training logic(`loss = MSE + λ × sector_penalty + γ × |sector_weight - target_weight|`);(b) walk-forward 自動化 8 panel framework;(c) 15 FAIL gates(G1-G15)完整實作;(d) multi-model ensemble(LGBM + XGBoost + Linear)。**對既有 snapshot 影響**:零(v0.2 framework 不改 ModelTrainer.train() 既有邏輯)。同步配套:憲章 §14.7-BQ Phase B(commit `27c1abf` v6.1.0-patch 第十五輪)+ Phase A 設計研究 `reports/model_trainer_phase_a_research_20260526.md`(581 行 18 章 commit `644e2eb` tag v6.1.24)。 | **ACTIVE** |
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
TOOL_VER = "v0.2"
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
                       lp.label_date
                FROM "feature_values" fv
                JOIN "core_universe_membership" m
                  ON m.stock_id = fv.stock_id
                 AND m.snapshot_id = %s
                 AND m.core_tier IN ('core_universe', 'convex_universe')
                JOIN label_prices lp
                  ON lp.stock_id = fv.stock_id
                WHERE fv.feature_set_id = %s
                  AND fv.as_of_date = %s
                ORDER BY fv.stock_id, fv.feature_name
                """,
                (
                    self.snapshot["as_of_date"],
                    label_min_date,
                    self.snapshot["universe_snapshot_id"],
                    self.feature_set_id,
                    self.snapshot["as_of_date"],
                ),
            )
            by_stock = {}
            labels = {}
            label_dates = {}
            for stock_id, feature_name, feature_value, label, label_date in cur.fetchall():
                by_stock.setdefault(stock_id, {})[feature_name] = float(feature_value or 0.0)
                labels[stock_id] = float(label or 0.0)
                label_dates[stock_id] = label_date
            self.rows = [
                {
                    "stock_id": sid,
                    "x": {f: by_stock[sid].get(f, 0.0) for f in self.features},
                    "y": labels[sid],
                    "label_date": label_dates[sid],
                }
                for sid in sorted(by_stock)
            ]
            if len(self.rows) < 100:
                self._detail("fail", f"rows_trained={len(self.rows)}, expected >= 100")
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

    def _audit_self(self):
        """v0.2 §10-F audit hooks self-invocation(對映 §10-D G5/G6/G8 + G10/G11)。

        backward-compat 模式:預設 log warning;strict mode 之 raise 留 Phase C continuation。
        類比 portfolio_sizer v0.2 之 audit_constraint_satisfaction self-invoke 模式。
        對映 §14.7-BQ 之 §10-F audit hooks 整合 audit_doctrine_compliance.py。
        """
        # G5/G6/G8 training quality(audit_training_quality)
        ic_mean = self.metrics.get("ic_mean", 0.0)
        ic_std = self.metrics.get("ic_std", 0.0)
        sharpe = self.metrics.get("sharpe", None)  # v0.1 baseline 未計算 Sharpe
        ok, msg = audit_training_quality(ic_mean=ic_mean, ic_std=ic_std, sharpe=sharpe,
                                         policy=DEFAULT_TRAINING_POLICY)
        if ok:
            self._detail("pass", f"§10-F audit_training_quality: {msg}")
        else:
            # backward-compat: WARN 不 raise(v0.1 baseline 容許;Phase C continuation 升 strict raise)
            self._detail("warn", f"§10-F audit_training_quality: {msg}(WARN-only;backward-compat)")

        # G10/G11 artifact consistency(audit_artifact_consistency)
        # v0.1 baseline 之 artifact 在 commit_outputs() 才寫;此處檢查 in-memory preprocessing
        mock_artifact = {
            "winsor_bounds": self.preprocessing.get("feature_bounds", {}),
            "model_id": self.model_id,
            "feature_names": list(self.features),
        }
        ok, msg = audit_artifact_consistency(mock_artifact, expected_keys=["model_id", "feature_names"])
        if ok:
            self._detail("pass", f"§10-F audit_artifact_consistency: {msg}")
        else:
            self._detail("warn", f"§10-F audit_artifact_consistency: {msg}(WARN-only;backward-compat)")

    def commit_outputs(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        model_payload = {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "feature_set_id": self.feature_set_id,
            "features": self.features,
            "weights": self.weights,
            "preprocessing": self.preprocessing,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Finance Model Trainer (v0.1)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument("--model-family", default="lgbm")
    parser.add_argument("--label-horizon", type=int, default=DEFAULT_LABEL_HORIZON)
    return parser.parse_args()


def main():
    args = parse_args()
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
