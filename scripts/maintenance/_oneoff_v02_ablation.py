"""One-off ablation: v0.2 interaction group impact at as_of=2026-04-25.

不入正式工具鏈；驗證完即可刪除。對應 §0.0-D.6 升版條件 #1 ablation 驗證。
"""
import json
import math
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from core.db_utils import get_db_connection


MODEL_ID_V01 = "mdl_20260425_lgbm_h20_5c7f36c2_v0_1"
MODEL_ID_V02 = "mdl_20260425_lgbm_h20_f1102a9f_v0_1"


def _rank(values):
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


def _rank_score(values):
    if len(values) <= 1:
        return [0.0] * len(values)
    r = _rank(values)
    mid = (len(values) + 1) / 2.0
    half = (len(values) - 1) / 2.0
    if half <= 0:
        return [0.0] * len(values)
    return [(x - mid) / half for x in r]


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def load_model(cur, model_id):
    cur.execute(
        """
        SELECT artifact_path, feature_set_id FROM model_registry
        WHERE model_id = %s
        """,
        (model_id,),
    )
    artifact_path, feature_set_id = cur.fetchone()
    # PROJECT_ROOT
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    p = PROJECT_ROOT / artifact_path / "model.json"
    return json.loads(p.read_text(encoding="utf-8")), feature_set_id


def run_ablation(model_id, feature_set_id, as_of_date, label_date):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        model, fsid = load_model(cur, model_id)
        weights = model["weights"]
        bounds = model.get("preprocessing", {}).get("feature_bounds", {})

        # Load feature_values
        cur.execute(
            """
            SELECT stock_id, feature_name, COALESCE(feature_value, 0)::float8
            FROM feature_values
            WHERE feature_set_id = %s AND as_of_date = %s
            """,
            (fsid, as_of_date),
        )
        by_stock = {}
        for sid, fname, val in cur.fetchall():
            by_stock.setdefault(sid, {})[fname] = float(val or 0)

        # Load labels (forward return)
        # 對齊 model_trainer.py 規則：base_close = date <= as_of 的最近一筆；
        #                              future_close = date >= as_of + horizon 的第一筆
        cur.execute(
            """
            WITH base AS (
                SELECT DISTINCT ON (stock_id) stock_id, close::float8 AS base_close
                FROM "TaiwanStockPriceAdj"
                WHERE date <= %s
                ORDER BY stock_id, date DESC
            ),
            fwd AS (
                SELECT DISTINCT ON (stock_id) stock_id, close::float8 AS future_close
                FROM "TaiwanStockPriceAdj"
                WHERE date >= %s
                ORDER BY stock_id, date ASC
            )
            SELECT b.stock_id, (f.future_close / b.base_close - 1) AS forward_return
            FROM base b JOIN fwd f ON b.stock_id = f.stock_id
            WHERE b.base_close > 0
            """,
            (as_of_date, label_date),
        )
        labels = {sid: ret for sid, ret in cur.fetchall()}

        # Build feature_definition group lookup
        cur.execute(
            """
            SELECT feature_name, feature_group FROM feature_definition
            WHERE feature_set_id = %s
            """,
            (fsid,),
        )
        groups = {fname: grp for fname, grp in cur.fetchall()}

        # Intersect stocks
        stocks = sorted(set(by_stock) & set(labels))
        print(f"  stocks: {len(stocks)} (intersection of features × labels)")

        # Compute transformed feature ranks per feature (with winsor)
        feature_ranks = {}
        for fname in weights:
            vals = []
            b = bounds.get(fname, {})
            lo = b.get("low")
            hi = b.get("high")
            for sid in stocks:
                v = by_stock.get(sid, {}).get(fname, 0.0)
                if not math.isfinite(v):
                    v = 0.0
                if lo is not None and hi is not None:
                    v = min(max(v, float(lo)), float(hi))
                vals.append(v)
            feature_ranks[fname] = _rank_score(vals)

        label_vals = [labels[sid] for sid in stocks]
        label_ranks = _rank_score(label_vals)

        # Full IC
        scores = [0.0] * len(stocks)
        for fname, w in weights.items():
            for i, r in enumerate(feature_ranks[fname]):
                scores[i] += r * float(w)
        score_ranks = _rank_score(scores)
        full_ic = _pearson(score_ranks, label_ranks)

        # Group ablation
        all_groups = sorted(set(groups.values()))
        results = {"full_ic": full_ic, "groups": {}}
        for drop_grp in all_groups:
            scores = [0.0] * len(stocks)
            for fname, w in weights.items():
                if groups.get(fname) == drop_grp:
                    continue
                for i, r in enumerate(feature_ranks[fname]):
                    scores[i] += r * float(w)
            score_ranks = _rank_score(scores)
            grp_ic = _pearson(score_ranks, label_ranks)
            results["groups"][drop_grp] = {
                "ic": grp_ic,
                "drop_minus_full": grp_ic - full_ic,
                "n_features": sum(1 for f, g in groups.items() if g == drop_grp and f in weights),
            }
        return results
    finally:
        cur.close()
        conn.close()


def main():
    AS_OF = "2026-04-25"
    LABEL_DATE = "2026-05-15"

    print("=" * 70)
    print(f"v0.2 ablation @ as_of={AS_OF}, label_date={LABEL_DATE}")
    print(f"Model: {MODEL_ID_V02}")
    print(f"Feature set: fs_20260425_feature_set_v0_2 (31 features)")
    print("=" * 70)
    v02 = run_ablation(MODEL_ID_V02, "fs_20260425_feature_set_v0_2", AS_OF, LABEL_DATE)
    print(f"\n  full_ic = {v02['full_ic']:.4f}\n")
    print(f"  {'Group':20s} | {'n':>3s} | {'IC w/o':>8s} | {'Δ vs full':>10s} | Verdict")
    print(f"  {'-'*20} | {'-'*3} | {'-'*8} | {'-'*10} | -------")
    for g, r in sorted(v02["groups"].items(), key=lambda x: x[1]["drop_minus_full"]):
        delta = r["drop_minus_full"]
        if delta < -0.005:
            v = "✅ HELPFUL"
        elif delta > 0.005:
            v = "❌ HARMFUL"
        else:
            v = "➖ neutral"
        print(f"  {g:20s} | {r['n_features']:>3d} | {r['ic']:>8.4f} | {delta:>+10.4f} | {v}")

    print()
    print("=" * 70)
    print("§0.0-D.6 升版條件 #1 驗證裁決")
    print("=" * 70)
    if "interaction" in v02["groups"]:
        ig = v02["groups"]["interaction"]
        print(f"  interaction group: {ig['n_features']} features")
        print(f"  drop_minus_full = {ig['drop_minus_full']:+.4f}")
        if ig["drop_minus_full"] < -0.005:
            print(f"  verdict         : ✅ INTERACTION 群有助於 IC（>0.5% impact）")
            print(f"  §0.0-D.6 #1     : 達成 → §0.3 戰術層脫離 IC=0 結構性失效")
        elif ig["drop_minus_full"] < 0:
            print(f"  verdict         : ⚠️ INTERACTION 群有微小助益（<0.5%）")
            print(f"  §0.0-D.6 #1     : 部分達成；需多時點 walk-forward 確認穩定性")
        elif abs(ig["drop_minus_full"]) <= 0.0001:
            print(f"  verdict         : ➖ INTERACTION 群 IC 影響為 0（單時點 macro 為常數）")
            print(f"  §0.0-D.6 #1     : 未達成；需 nonlinear model 或多時點 regime 才能擷取")
        else:
            print(f"  verdict         : ❌ INTERACTION 群有害（drop_minus_full > 0）")
            print(f"  §0.0-D.6 #1     : 否決；需重新設計交互式")
    else:
        print(f"  ❌ interaction group not found in model")


if __name__ == "__main__":
    main()
