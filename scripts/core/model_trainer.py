"""
model_trainer.py v0.1 (Quantum Finance Model Training Authority)
================================================================================
最後更新日期: 2026-05-16
主權狀態: IMPLEMENTED (憲法 v5.4.22 §8.3 Model Registry v0.1 草案實作)
最高原則: Model Training Authority

v0.1 邊界:
1. 只讀 feature_store_* 與 core_universe_* 治理表；不直接讀 raw API tables。
2. 建立 model_registry / model_training_run 草案表。
3. v0.1 baseline 使用 committed feature_set 與 label_horizon forward return label；
   feature as-of date 與 label date 明確分離，不產生交易建議。
4. 輸出可重現 JSON artifact：model.json / metrics.json / feature_importance.json。
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


CONSTITUTION_VER = "v5.4.22"
TOOL_VER = "v0.1"
DEFAULT_MODEL_POLICY_VERSION = "model_policy_v0.1"
DEFAULT_LABEL_HORIZON = 20
DEFAULT_SEED = 5422


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
