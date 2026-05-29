"""
prediction_engine.py v0.3 (Quantum Finance Prediction Authority · §10 Phase C milestone #3.5 — train/inference sector_balance consistency)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 補入 [Sovereignty Declaration] + Supreme Authority Principle line)
**主權狀態**: IMPLEMENTED (憲法 v6.1.0 §8.4 + §8.8.8 exactly-one prediction-backed + §10 milestone #3.5 sector_balance inference consistency + §14.7-CT Prediction Production Closure + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

**[Sovereignty Declaration]** (2026-05-29 §一.11 補入,憲法 §3.1 序列模組 / §14.7-CT): 本程式為 **§9.1 prediction_engine + §14.7-CT Production Closure 唯一治權載體**(§3.1 序列模組第 8/9 員)。**治權邊界**:(a) §3.1 序列 prediction 模組;(b) 五套禁令不涉;(c) T1-T3 不分層;(d) §8.5 已 handle by feature_store_builder;(e) **不訓練 model**(load committed model from model_registry);(f) **不算 features**(讀取 feature_values);(g) **不持有 portfolio sizing**(由 portfolio_sizer 負責);(h) 唯一職責:load committed model + 對 latest feature_values 預測 + 寫入 prediction_run governance table。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Prediction Authority]: 對齊憲章 §8.4 Prediction Table v0.1 草案，
   作為 §2 維運矩陣 Step 11 之執行載體（§8.7 矩陣延伸）；載入 model artifact +
   feature_values 推論並寫 `prediction_run` / `prediction_values`。
2. [Read-Only Upstream]: 只讀 committed `model_registry` artifact 與 committed
   `feature_store_snapshot`；**不**重新訓練、**不**修改 Feature Store、**不**修改
   Model Registry；source-of-truth 為 model_registry artifact。
3. [Transform Consistency]: 推論時使用 model artifact 中的 winsor bounds
   與 average-rank transform，確保 train/predict transform 完全一致；
   防止 `model_trainer.py` 訓練與推論 transform 漂移。
4. [Universe Coverage Lock]: 每一 `prediction_run` 必須單獨等於其鎖定
   `universe_snapshot_id` 對應之 core+convex universe stock 數（§8.8.4
   per-run coverage 強制）；coverage 不足即 verdict WARN。
5. [Single Delivery Invariant]: §8.8.8「exactly 1 prediction-backed」規則：
   多次 `--commit` 後僅最新 prediction_run 保留 `status='committed'`，
   其餘標記 `deprecated`（evidence-only）；對齊 §8.8.10 Final Delivery Index。
6. [Deterministic Output]: prediction_run_id 命名格式
   `pred_{yyyymmdd}_{model_id}`，提供可重現之 audit trail。
7. [Hybrid Observability]: 維運觸發 `record_lifecycle` 與 `write_data_audit_log`；
   主權判定動態計算（§5.6.3）。
8. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準。
9. [Single-Delivery Automation] (v0.2): §8.8.8 exactly-one prediction-backed 自
   動化——`--deprecate-previous` 在 commit 後將同 `prediction_policy_version`
   下其他 `committed` run 改為 `deprecated`，並於 notes 寫入
   `superseded_by={new_run_id}`；`--commit-as-evidence-only` 將新 run 直接寫
   為 `deprecated`，供 walk-forward / h30 historical evidence。兩 flag 互斥。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [Step 11-dry：推論驗算]** | `$ python scripts/core/prediction_engine.py --dry-run --model-id <mdl_id> --as-of-date 2026-05-15` | prediction_engine v0.2 |
| **2. [Step 11-commit：production-current]** | `$ python scripts/core/prediction_engine.py --commit --deprecate-previous --model-id <mdl_id> --as-of-date 2026-05-15` | prediction_engine v0.2 |
| **3. [Step 11-historical：walk-forward evidence]** | `$ python scripts/core/prediction_engine.py --commit --commit-as-evidence-only --model-id <historical_mdl_id> --as-of-date <historical_date>` | prediction_engine v0.2 |
| **4. [Step 11-h30：v6.2.0 預備]** | `$ python scripts/core/prediction_engine.py --commit --commit-as-evidence-only --model-id <mdl_id_h30> --as-of-date <date>` | prediction_engine v0.2 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **dry-run** | `--dry-run` | 輸出 predictions distribution 與 coverage，不寫 prediction_values |
| **deprecate-previous** | `--deprecate-previous`（v0.2 落地）| commit 同時標記同 policy 下既有 committed run 為 deprecated 並寫 `superseded_by` 標記；與 --commit-as-evidence-only 互斥 |
| **evidence-only** | `--commit-as-evidence-only`（v0.2 落地）| 寫 prediction_values 供 audit；prediction_run.status 直接設為 deprecated；不影響當前 production-current delivery |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.3** | 2026-05-26 | Codex | **§10 Phase C milestone #3.5 — train/inference sector_balance consistency(對映 §10-D G10/G11 transform consistency requirement;補修 milestone #3 之 inference-side gap)**:model_trainer milestone #3(commit `1be102e`)在 model.json `preprocessing.sector_balance` 寫入 sector_penalty_factor 之 SSOT;本 v0.3 prediction_engine 補 inference-side 套用,以確保 train/inference 之 prediction values 完全一致(若 train 套了 / inference 沒套 → 兩端 prediction divergence,違反 §10-D G10 transform consistency)。**4 處 edits**:**(I)** `load_inputs()` SQL 加 LATERAL JOIN TaiwanStockInfo 載 `industry_category` per stock(同 model_trainer milestone #2 pattern;LATERAL 取 latest as-of as_of_date);self.rows 加 `"industry"` field;**(II)** `predict()` 在 raw 計算後,讀 `model.preprocessing.sector_balance` 之 metadata;若存在(model 為 milestone #3+ 訓練)則套 Lagrangian:`adjusted_pred[i] = max(raw_pred[i] + sector_penalty_factor[stock_industry], min_floor)`;用 adjusted values 重新 ordering(取代 raw values);若不存在(legacy v0.2 model 或 sector_balance_enabled=False 訓練)則完全 backward-compat(行為 = v0.2);**(III)** TOOL_VER v0.2 → v0.3;CONSTITUTION_VER v6.0.0 → v6.1.0(對齊 model_trainer / 其他 v6.1.0 模組);**(IV)** 標頭 8 docstring + 修訂歷程 v0.3 ACTIVE / v0.2 SUPERSEDED。**邏輯動量**:既有 v0.2 之 deprecate-previous / commit-as-evidence-only flags 完全不變;rank → label → confidence 邏輯不變;唯獨在 ordering 之前加 sector_balance adjustment(opt-in via model.json 之 metadata existence)。**對既有 model 影響**:零(legacy v0.2 model.json 無 sector_balance section;inference 行為 = v0.2 完全相同;新 v0.3 model.json with sector_balance section → inference 套 Lagrangian 確保 train/predict 一致)。**對既有 prediction_run 影響**:零(本次純為 inference logic 升級;不改 prediction_run / prediction_values DDL;不重跑既有 predictions)。**§10-D G10/G11 transform consistency 對齊**:milestone #3 之 trainer.preprocessing.sector_balance 為 SSOT;v0.3 prediction_engine 透過 model.json reading 套用同一 transform;確保 train/inference 之 prediction 完全 deterministic 對齊。**對既有 CLI 行為**:零(default workflow `--dry-run --model-id X --as-of-date X` 完全不變;sector_balance 套用為 model-driven 自動偵測)。**§10 Phase C continuation 進度 cumulative(post milestone #3.5)**:milestone #5 G strict raise 完成(commit `583f268`)+ 本 milestone #3.5 → §10 治本 96% → **98%**(train/inference consistency closure);Phase D production v6.2.0 tag 為剩餘 final step。**Smoke test**:mock model.json with sector_balance section(approach D_post_processing_lagrangian_v2);verify adjusted_pred = max(raw + penalty, min_floor)邏輯 + ordering 正確。同步配套:憲章 §10-D G10/G11 + §14.7-BQ Phase B + model_trainer milestone #3(commit `1be102e` v0.2.2)+ milestones #1/#2/#3/#4/#5(commits 47838d1 / 42d4872 / 1be102e / 88b9d29 / 583f268)。 | **ACTIVE** |
| v0.2 | 2026-05-19 | Codex | §8.8.8 single-delivery 自動化：新增 `--deprecate-previous` 與 `--commit-as-evidence-only` 兩 CLI flag；commit_outputs() 支援新 status 路徑與 supersedes 標記；兩 flag 互斥且皆需 --commit。配合 prediction_engine_formal_prediction_research_20260519.md §9.1 / §9.2 建議落地。 | SUPERSEDED |
| v0.1 | 2026-05-16 | Codex | 首版：§8.4 Prediction Table 草案；2026-05-17 補入 winsor bounds + average-rank transform 一致性（與 trainer 對齊）；2026-05-18 v6.0.0-patch 落地 §8.8.8 exactly 1 prediction-backed 規則與 §8.8.10 Final Delivery Index；唯一 committed delivery 為 `pred_20260425_mdl_20260425_lgbm_h20_d969ffb1_v0_1`（IC=0.3716）。 | SUPERSEDED |
================================================================================
"""
import argparse
import json
import math
import sys
import time
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
DEFAULT_PREDICTION_POLICY_VERSION = "prediction_policy_v0.2"  # §14.7-CU 2026-05-28:對齊 portfolio_sizer v0.3 expectation(原 v0.1 drift fix)

DDL_PREDICTION_RUN = """
CREATE TABLE IF NOT EXISTS "prediction_run" (
    "run_id" VARCHAR(255) PRIMARY KEY,
    "model_id" VARCHAR(255) NOT NULL,
    "feature_set_id" VARCHAR(255) NOT NULL,
    "as_of_date" DATE NOT NULL,
    "universe_snapshot_id" VARCHAR(255) NOT NULL,
    "prediction_policy_version" VARCHAR(255) NOT NULL,
    "rows_written" INTEGER NOT NULL,
    "status" VARCHAR(64) NOT NULL DEFAULT 'draft',
    "notes" TEXT,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

DDL_PREDICTION_VALUES = """
CREATE TABLE IF NOT EXISTS "prediction_values" (
    "run_id" VARCHAR(255) NOT NULL,
    "stock_id" VARCHAR(255) NOT NULL,
    "as_of_date" DATE NOT NULL,
    "prediction_value" NUMERIC(24, 8) NOT NULL,
    "prediction_rank" INTEGER,
    "signal_label" VARCHAR(64),
    "confidence" NUMERIC(12, 8),
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY ("run_id", "stock_id", "as_of_date")
);
CREATE INDEX IF NOT EXISTS "idx_prediction_values_stock_date"
    ON "prediction_values" ("stock_id", "as_of_date");
"""


class PredictionEngine:
    def __init__(self, model_id, as_of_date, commit=False,
                 deprecate_previous=False, commit_as_evidence_only=False):
        self.model_id = model_id
        self.as_of_date = as_of_date
        self.commit = commit
        # v0.2 §8.8.8 single-delivery 自動化 flags（互斥）
        self.deprecate_previous = deprecate_previous
        self.commit_as_evidence_only = commit_as_evidence_only
        self.model = None
        self.registry = None
        self.rows = []
        self.predictions = []
        self.run_id = f"pred_{as_of_date.strftime('%Y%m%d')}_{model_id}"
        self.stats = {"pass": 0, "warn": 0, "fail": 0, "details": []}

    def _detail(self, bucket, msg):
        self.stats[bucket] += 1
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[bucket]
        line = f"{icon} [{bucket.upper()}] {msg}"
        self.stats["details"].append(line)
        print(line)

    def ensure_tables(self, cur):
        cur.execute(DDL_PREDICTION_RUN)
        cur.execute(DDL_PREDICTION_VALUES)

    def load_inputs(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            self.ensure_tables(cur)
            conn.commit()
            cur.execute(
                """
                SELECT model_id, model_family, feature_set_id, universe_snapshot_id,
                       metrics, hyperparams, artifact_path, status
                FROM "model_registry"
                WHERE model_id = %s
                """,
                (self.model_id,),
            )
            row = cur.fetchone()
            if not row:
                self._detail("fail", f"model_id={self.model_id} missing")
                return False
            keys = ["model_id", "model_family", "feature_set_id", "universe_snapshot_id", "metrics", "hyperparams", "artifact_path", "status"]
            self.registry = dict(zip(keys, row))
            if self.registry["status"] != "committed":
                self._detail("fail", f"model status={self.registry['status']}, expected committed")
                return False
            artifact_path = _PROJECT_ROOT / self.registry["artifact_path"] / "model.json"
            if not artifact_path.exists():
                self._detail("fail", f"model artifact missing: {artifact_path}")
                return False
            with artifact_path.open("r", encoding="utf-8") as fh:
                self.model = json.load(fh)
            self._detail("pass", f"committed model loaded: {self.model_id}")

            cur.execute(
                """
                SELECT status, as_of_date, universe_snapshot_id
                FROM "feature_store_snapshot"
                WHERE feature_set_id = %s
                """,
                (self.registry["feature_set_id"],),
            )
            fs = cur.fetchone()
            if not fs or fs[0] != "committed":
                self._detail("fail", "feature_set is not committed")
                return False
            if fs[1] != self.as_of_date:
                self._detail("fail", f"feature_set as_of_date={fs[1]} != requested {self.as_of_date}")
                return False
            if fs[2] != self.registry["universe_snapshot_id"]:
                self._detail("fail", "universe snapshot mismatch between model and feature_set")
                return False
            self._detail("pass", f"feature_set locked: {self.registry['feature_set_id']}")

            # v0.3 milestone #3.5: 加 LATERAL JOIN TaiwanStockInfo 載 industry_category
            #                       供 inference-side sector_balance adjustment
            cur.execute(
                """
                SELECT fv.stock_id, fv.feature_name, COALESCE(fv.feature_value, 0)::float8,
                       fv.is_null_imputed, ind.industry_category
                FROM "feature_values" fv
                JOIN "core_universe_membership" m
                  ON m.stock_id = fv.stock_id
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
                  AND m.snapshot_id = %s
                  AND m.core_tier IN ('core_universe', 'convex_universe')
                ORDER BY fv.stock_id, fv.feature_name
                """,
                (self.as_of_date, self.registry["feature_set_id"], self.as_of_date, self.registry["universe_snapshot_id"]),
            )
            by_stock = {}
            industries = {}
            imputed = 0
            total = 0
            for stock_id, feature_name, feature_value, is_imputed, industry in cur.fetchall():
                by_stock.setdefault(stock_id, {})[feature_name] = float(feature_value or 0.0)
                industries[stock_id] = industry or "UNKNOWN"
                total += 1
                imputed += 1 if is_imputed else 0
            features = self.model["features"]
            self.rows = [
                {
                    "stock_id": sid,
                    "x": {f: by_stock[sid].get(f, 0.0) for f in features},
                    "industry": industries.get(sid, "UNKNOWN"),
                }
                for sid in sorted(by_stock)
            ]
            # §14.7-BW pure doctrine: 從 snapshot 動態取 N(取代 hardcoded 150)
            cur.execute(
                """
                SELECT COUNT(*) FROM "core_universe_membership"
                WHERE snapshot_id = %s
                  AND core_tier IN ('core_universe', 'convex_universe')
                """,
                (self.registry["universe_snapshot_id"],),
            )
            expected_n = cur.fetchone()[0]
            if len(self.rows) != expected_n:
                self._detail("fail", f"prediction universe rows={len(self.rows)}, expected {expected_n} (dynamic per §14.7-BW)")
                return False
            null_ratio = imputed / total if total else 0
            if null_ratio > 0.05:
                self._detail("warn", f"imputed feature ratio={null_ratio:.4f} > 5%")
            else:
                self._detail("pass", f"imputed feature ratio={null_ratio:.4f}")
        finally:
            cur.close()
            conn.close()
        return self.stats["fail"] == 0

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

    def predict(self):
        weights = self.model["weights"]
        preprocessing = self.model.get("preprocessing", {})
        feature_bounds = preprocessing.get("feature_bounds", {})
        transformed = {}
        for feature in weights:
            values = []
            bounds = feature_bounds.get(feature, {})
            lo = bounds.get("low")
            hi = bounds.get("high")
            for row in self.rows:
                value = row["x"].get(feature, 0.0)
                value = value if math.isfinite(value) else 0.0
                if lo is not None and hi is not None:
                    value = min(max(value, float(lo)), float(hi))
                values.append(value)
            transformed[feature] = self._rank_scores(values)
        # v0.3 milestone #3.5: raw predictions(unadjusted)
        raw_values_per_row = []
        for idx, row in enumerate(self.rows):
            value = sum(transformed[name][idx] * float(weight) for name, weight in weights.items())
            raw_values_per_row.append((row["stock_id"], value, row.get("industry", "UNKNOWN")))

        # v0.3 milestone #3.5: 套 sector_balance Lagrangian adjustment if model has it
        # train/inference consistency:model.json 之 preprocessing.sector_balance 為 SSOT
        sb = preprocessing.get("sector_balance")
        if sb:
            sector_penalty_factor = sb.get("sector_penalty_factor", {})
            min_floor = float(sb.get("min_floor", -10.0))
            adjusted = []
            for stock_id, raw_val, industry in raw_values_per_row:
                penalty = float(sector_penalty_factor.get(industry, 0.0))
                adj_val = max(raw_val + penalty, min_floor)
                adjusted.append((stock_id, adj_val))
            self._detail("pass",
                         f"§10 milestone #3.5 sector_balance adjustment applied "
                         f"(λ={sb.get('lambda')}, n_sectors={len(sector_penalty_factor)}, "
                         f"approach={sb.get('approach', 'unknown')})")
            raw = adjusted
        else:
            # legacy v0.2 model or sector_balance_enabled=False training → no adjustment
            raw = [(sid, val) for sid, val, _ in raw_values_per_row]
            self._detail("pass", "§10 milestone #3.5: model has no sector_balance section(legacy / opt-out;backward-compat)")

        ordered = sorted(raw, key=lambda item: item[1], reverse=True)
        n = len(ordered)
        self.predictions = []
        for rank, (stock_id, value) in enumerate(ordered, start=1):
            if rank <= 20:
                label = "long"
            elif rank > n - 20:
                label = "watch"
            else:
                label = "hold"
            confidence = rank / n if n else 0.0
            confidence = abs(confidence - 0.5) * 2
            if not math.isfinite(value):
                self._detail("fail", f"non-finite prediction for {stock_id}")
                continue
            self.predictions.append((stock_id, value, rank, label, confidence))
        self._detail("pass", f"predictions computed rows={len(self.predictions)}")

    def commit_outputs(self):
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            self.ensure_tables(cur)
            cur.execute('SELECT 1 FROM "prediction_run" WHERE run_id = %s', (self.run_id,))
            if cur.fetchone():
                suffix = datetime.now().strftime("%H%M%S")
                self.run_id = f"{self.run_id}_{suffix}"
                self._detail("warn", f"existing prediction run found; new run_id={self.run_id}")

            # v0.2 §8.8.8 single-delivery 自動化：決定本 run 之 status 與 notes
            if self.commit_as_evidence_only:
                new_status = "deprecated"
                notes = (
                    "v0.2 evidence-only: prediction_values written for audit/walk-forward "
                    "evidence; status set to 'deprecated' on insert; not a production-current "
                    "delivery; signal labels are not investment advice"
                )
            else:
                new_status = "committed"
                notes = (
                    "v0.2 baseline inference; signal labels are not investment advice"
                )

            cur.execute(
                """
                INSERT INTO "prediction_run" (
                    run_id, model_id, feature_set_id, as_of_date, universe_snapshot_id,
                    prediction_policy_version, rows_written, status, notes
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    self.run_id, self.model_id, self.registry["feature_set_id"], self.as_of_date,
                    self.registry["universe_snapshot_id"], DEFAULT_PREDICTION_POLICY_VERSION,
                    len(self.predictions), new_status, notes,
                ),
            )
            cur.executemany(
                """
                INSERT INTO "prediction_values" (
                    run_id, stock_id, as_of_date, prediction_value,
                    prediction_rank, signal_label, confidence
                ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                [(self.run_id, sid, self.as_of_date, val, rank, label, conf) for sid, val, rank, label, conf in self.predictions],
            )

            # v0.2 --deprecate-previous：將同 policy 下其他 committed run 改為 deprecated
            # 注意：committing as evidence-only 時不執行此邏輯（本 run 即為 deprecated）
            if self.deprecate_previous and not self.commit_as_evidence_only:
                cur.execute(
                    """
                    SELECT run_id FROM "prediction_run"
                    WHERE prediction_policy_version = %s
                      AND status = 'committed'
                      AND run_id <> %s
                    """,
                    (DEFAULT_PREDICTION_POLICY_VERSION, self.run_id),
                )
                superseded = [row[0] for row in cur.fetchall()]
                if superseded:
                    supersede_note = (
                        f" | superseded_by={self.run_id} "
                        f"at {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
                    )
                    cur.execute(
                        """
                        UPDATE "prediction_run"
                        SET status = 'deprecated',
                            notes = COALESCE(notes, '') || %s
                        WHERE prediction_policy_version = %s
                          AND status = 'committed'
                          AND run_id <> %s
                        """,
                        (supersede_note, DEFAULT_PREDICTION_POLICY_VERSION, self.run_id),
                    )
                    self._detail(
                        "pass",
                        f"§8.8.8 single-delivery: deprecated {len(superseded)} previous "
                        f"committed run(s) under policy {DEFAULT_PREDICTION_POLICY_VERSION}",
                    )
                else:
                    self._detail(
                        "pass",
                        f"§8.8.8 single-delivery: no previous committed run to deprecate "
                        f"(policy={DEFAULT_PREDICTION_POLICY_VERSION})",
                    )

            conn.commit()
        finally:
            cur.close()
            conn.close()
        try:
            write_data_audit_log(
                "prediction_values", self.run_id, self.as_of_date,
                "PREDICTION_RUN_EVIDENCE" if self.commit_as_evidence_only else "PREDICTION_RUN",
                len(self.predictions),
            )
        except Exception as exc:
            self._detail("warn", f"data_audit_log failed: {type(exc).__name__}: {exc}")
        run_mode = "evidence-only (deprecated)" if self.commit_as_evidence_only else "committed"
        self._detail("pass", f"prediction run inserted as {run_mode}: {self.run_id}")

    def verdict(self):
        if self.stats["fail"] > 0:
            return "FAILED"
        if self.stats["warn"] > 0:
            return "WARNING"
        return "PERFECT"

    def run(self):
        start = time.time()
        lifecycle_cm = record_lifecycle("prediction_engine_v0.1", category="prediction", stock_id="SYSTEM") if self.commit else None
        lifecycle = lifecycle_cm.__enter__() if lifecycle_cm else None
        try:
            if self.load_inputs():
                self.predict()
                if self.commit and self.stats["fail"] == 0:
                    self.commit_outputs()
            verdict = self.verdict()
            if lifecycle and verdict == "FAILED":
                lifecycle.mark_failed("prediction_engine failed")
            elif lifecycle and verdict == "WARNING":
                lifecycle.mark_warning("prediction_engine warning")
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
        print(f"🚀 Quantum Finance: Prediction Engine 執行摘要 ({TOOL_VER})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{CONSTITUTION_VER}.md §8.4")
        print("治理權責 : Prediction Authority")
        print(f"執行模式 : {'COMMIT' if self.commit else 'DRY-RUN'}")
        print(f"Run ID   : {self.run_id}")
        print(f"Model ID : {self.model_id}")
        print("────────────────────────────────────────────────────────────────────────────────")
        print(f"📈 predictions : {len(self.predictions)}")
        print(f"✅ pass        : {self.stats['pass']}")
        print(f"⚠️  warn        : {self.stats['warn']}")
        print(f"❌ fail        : {self.stats['fail']}")
        print(f"🕒 總計耗時    : {(time.time() - start)*1000:.2f} ms")
        print(f"⚖️  主權判定    : {verdict}")
        print("🛡️" * 40 + "\n")
        return self.stats["fail"] == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Finance Prediction Engine (v0.2)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--as-of-date", required=True)
    # v0.2 §8.8.8 single-delivery 自動化（兩 flag 互斥；皆需配 --commit）
    delivery = parser.add_mutually_exclusive_group()
    delivery.add_argument(
        "--deprecate-previous", action="store_true",
        help="commit 同時將同 policy 下其他 committed run 改為 deprecated 並寫 superseded_by 標記",
    )
    delivery.add_argument(
        "--commit-as-evidence-only", action="store_true",
        help="寫 prediction_values 供 audit；prediction_run.status 直接設為 deprecated；不影響當前 production-current delivery",
    )
    args = parser.parse_args()
    if args.dry_run and (args.deprecate_previous or args.commit_as_evidence_only):
        parser.error("--deprecate-previous / --commit-as-evidence-only 需與 --commit 一起使用")
    return args


def main():
    args = parse_args()
    as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    engine = PredictionEngine(
        args.model_id, as_of_date, commit=args.commit,
        deprecate_previous=args.deprecate_previous,
        commit_as_evidence_only=args.commit_as_evidence_only,
    )
    ok = engine.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
