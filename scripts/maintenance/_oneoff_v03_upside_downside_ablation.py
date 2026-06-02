"""
_oneoff_v03_upside_downside_ablation.py v0.2 (v0.3 upside/downside ablation)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: ACTIVE (one-off / maintenance)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:**一次性** ablation 腳本:測 v0.3 上行/下行特徵之影響(已跑完)。

**輸入 → 輸出**:既有 DB → 一次性處理結果

**為什麼需要它**:記述性保留(已執行);非常態流程。

## 📜 一、核心定義說明 (Core Definitions)

1. **[One-off Script]**:一次性/維運腳本
2. **[Sovereignty Declaration]**:本程式為**非 charter-core 子系統**工具(charter 可達/引用),不涉 §3.1/§3.2 序列治權主軸、不持五套禁令、不處理 §8.5 anti-leakage。 本檔為**一次性**腳本,非常態 pipeline;保留作 audit trail(亦為 C-隔離候選)。
3. **[Historical Reference Authority]**:本檔標頭版本為記述性快照,非權威來源(權威為憲章 + 程式現行碼)。

## 📊 二、全量功能群矩陣 (Functional Group Matrix)

| 功能 / 指令 | 說明 |
| :--- | :--- |
| python <此檔> | 執行一次性處理 |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.2 | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話補正;標示一次性。原邏輯不變。 | **ACTIVE** |

## 原始說明
One-off ablation: v0.3 upside/downside features impact.

不入正式工具鏈；驗證完即可刪除。對應 §9.9 P1 v0.1 升版條件 ablation 驗證
（依 §14.7-AD / §14.7-AE）。

執行步驟（在主環境）：
  1. python scripts/core/feature_store_builder.py \\
       --commit --as-of-date 2026-04-25 \\
       --feature-set-version feature_set_v0.3 --label-horizon 20
  2. python scripts/core/model_trainer.py \\
       --commit --feature-set-id fs_20260425_feature_set_v0_3 \\
       --label-horizon 20
       # → 取得新 model_id（替換下方 MODEL_ID_V03）
  3. python scripts/maintenance/_oneoff_v03_upside_downside_ablation.py

裁決規則（§14.7-AD §6.2 預期）：
  - upside_volatility_60d / upside_capture_60d 之 drop_minus_full < 0 → HELPFUL
  - downside_volatility_60d / downside_capture_60d 之 drop_minus_full < 0 → HELPFUL
  - 整 price group 之 drop_minus_full 預期相對於 v0.1 更為負（凸性增益）
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


MODEL_ID_V01 = "mdl_20260425_lgbm_h20_5c7f36c2_v0_1"  # v0.1 base (27 features)
MODEL_ID_V02 = "mdl_20260425_lgbm_h20_f1102a9f_v0_1"  # v0.2 含 interaction (31 features)
MODEL_ID_V03 = None  # ← 主環境 trainer 跑完後填入

FEATURE_SET_V03 = "fs_20260425_feature_set_v0_3"
V03_NEW_FEATURES = {
    "upside_volatility_60d",
    "downside_volatility_60d",
    "upside_capture_60d",
    "downside_capture_60d",
}


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
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"model_id {model_id} not found in model_registry")
    artifact_path, feature_set_id = row
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    p = PROJECT_ROOT / artifact_path / "model.json"
    return json.loads(p.read_text(encoding="utf-8")), feature_set_id


def run_ablation(model_id, feature_set_id, as_of_date, label_date):
    """完整 ablation：full IC + group ablation + 4 個 v0.3 個別特徵 ablation"""
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

        # Feature group lookup
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
        results = {
            "model_id": model_id,
            "feature_set_id": fsid,
            "as_of_date": str(as_of_date),
            "label_date": str(label_date),
            "n_stocks": len(stocks),
            "full_ic": full_ic,
            "groups": {},
            "individual_v03_features": {},
        }
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

        # v0.3 個別 feature ablation（針對 4 個新特徵）
        for fname in V03_NEW_FEATURES:
            if fname not in weights:
                continue
            scores = [0.0] * len(stocks)
            for f, w in weights.items():
                if f == fname:
                    continue
                for i, r in enumerate(feature_ranks[f]):
                    scores[i] += r * float(w)
            score_ranks = _rank_score(scores)
            ic = _pearson(score_ranks, label_ranks)
            results["individual_v03_features"][fname] = {
                "ic": ic,
                "drop_minus_full": ic - full_ic,
            }

        return results
    finally:
        cur.close()
        conn.close()


def main():
    AS_OF = "2026-04-25"
    LABEL_DATE = "2026-05-15"

    if MODEL_ID_V03 is None:
        print("=" * 70)
        print("❌ MODEL_ID_V03 未設定")
        print("=" * 70)
        print()
        print("請先執行以下步驟，取得 v0.3 model_id 後填入本檔案頂端：")
        print()
        print("  Step 1 (commit v0.3 feature set)：")
        print("    python scripts/core/feature_store_builder.py \\")
        print(f"      --commit --as-of-date {AS_OF} \\")
        print("      --feature-set-version feature_set_v0.3 --label-horizon 20")
        print()
        print("  Step 2 (train v0.3 model)：")
        print("    python scripts/core/model_trainer.py \\")
        print(f"      --commit --feature-set-id {FEATURE_SET_V03} \\")
        print("      --label-horizon 20")
        print()
        print("  Step 3 (從 model_registry 取得 model_id 並填入 MODEL_ID_V03)：")
        print("    SELECT model_id FROM model_registry")
        print("    WHERE feature_set_id = 'fs_20260425_feature_set_v0_3'")
        print("    ORDER BY created_at DESC LIMIT 1;")
        print()
        print("  Step 4 (re-run this script)：")
        print("    python scripts/maintenance/_oneoff_v03_upside_downside_ablation.py")
        sys.exit(2)

    print("=" * 70)
    print(f"v0.3 ablation @ as_of={AS_OF}, label_date={LABEL_DATE}")
    print(f"Model: {MODEL_ID_V03}")
    print(f"Feature set: {FEATURE_SET_V03} (31 features: 27 base + 4 upside/downside)")
    print("=" * 70)
    v03 = run_ablation(MODEL_ID_V03, FEATURE_SET_V03, AS_OF, LABEL_DATE)
    print(f"\n  full_ic = {v03['full_ic']:.4f}")
    print(f"  n_stocks = {v03['n_stocks']}\n")

    # Group ablation
    print(f"  {'Group':20s} | {'n':>3s} | {'IC w/o':>8s} | {'Δ vs full':>10s} | Verdict")
    print(f"  {'-'*20} | {'-'*3} | {'-'*8} | {'-'*10} | -------")
    for g, r in sorted(v03["groups"].items(), key=lambda x: x[1]["drop_minus_full"]):
        delta = r["drop_minus_full"]
        if delta < -0.005:
            v = "✅ HELPFUL"
        elif delta > 0.005:
            v = "❌ HARMFUL"
        else:
            v = "➖ neutral"
        print(f"  {g:20s} | {r['n_features']:>3d} | {r['ic']:>8.4f} | {delta:>+10.4f} | {v}")

    # Individual v0.3 feature ablation
    print()
    print(f"  --- v0.3 個別新特徵 ablation ---")
    print(f"  {'Feature':30s} | {'IC w/o':>8s} | {'Δ vs full':>10s} | Verdict")
    print(f"  {'-'*30} | {'-'*8} | {'-'*10} | -------")
    for fname, r in sorted(v03["individual_v03_features"].items(),
                           key=lambda x: x[1]["drop_minus_full"]):
        delta = r["drop_minus_full"]
        if delta < -0.002:
            v = "✅ HELPFUL"
        elif delta > 0.002:
            v = "❌ HARMFUL"
        else:
            v = "➖ neutral"
        print(f"  {fname:30s} | {r['ic']:>8.4f} | {delta:>+10.4f} | {v}")

    # §9.9 升版條件驗證
    print()
    print("=" * 70)
    print("§9.9 P1 v0.1 升版條件驗證裁決")
    print("=" * 70)

    # 個別新特徵預期方向
    expected_directions = {
        "upside_volatility_60d": "HELPFUL（凸性右尾捕捉）",
        "upside_capture_60d": "HELPFUL（上行爆發力）",
        "downside_volatility_60d": "HELPFUL（風險懲罰）",
        "downside_capture_60d": "HELPFUL（下行衝擊）",
    }
    helpful_count = 0
    for fname, expected in expected_directions.items():
        r = v03["individual_v03_features"].get(fname, {})
        delta = r.get("drop_minus_full", 0)
        if delta < -0.002:
            status = "✅ HELPFUL"
            helpful_count += 1
        elif delta > 0.002:
            status = "❌ HARMFUL"
        else:
            status = "➖ neutral"
        print(f"  {fname:30s} | Δ={delta:>+.4f} | {status} | 預期: {expected}")

    print()
    if helpful_count >= 2:
        print(f"  ✅ v0.3 升版條件達成：{helpful_count}/4 新特徵 HELPFUL")
        print(f"  下一步：P1 v0.3 修改 core_universe_builder VolatilityControl 評分公式")
    elif helpful_count >= 1:
        print(f"  ⚠️ v0.3 升版條件部分達成：{helpful_count}/4 新特徵 HELPFUL")
        print(f"  建議：保留 v0.3 為實證證據，但暫不啟動 v0.3 VolatilityControl 修改")
    else:
        print(f"  ❌ v0.3 升版條件未達成：0/4 新特徵 HELPFUL")
        print(f"  與 P2 interaction features 一樣，需重新評估設計")
        print(f"  建議：保留 v0.3 為歷史證據，採非線性 trainer 才能擷取（同 §0.3-A 治權禁令邏輯）")

    # 保存 JSON 結果供後續入憲使用
    import json as _json
    out_path = Path(__file__).resolve().parent.parent.parent / "reports" / "v03_upside_downside_ablation_result.json"
    out_path.write_text(_json.dumps(v03, indent=2, default=str), encoding="utf-8")
    print(f"\n  📁 JSON 結果儲存於：{out_path.relative_to(out_path.parent.parent.parent)}")


if __name__ == "__main__":
    main()
