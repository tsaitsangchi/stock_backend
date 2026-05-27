"""Phase D Ablation — Feature IC analysis v0.1 vs v0.4 baseline.

§14.7-CA T_CA-1 證偽 gate:v0.3(現 v0.4) ensemble IC ≥ v0.1 baseline + 0.02
本 script 以單一 historical as_of_date 計算 per-feature Spearman IC,並彙整:
- v0.1 subset(27 base features)IC distribution
- v0.4 full(64 features)IC distribution
- per-pillar(§0.1 / §0.2 / §0.3)IC 平均
- top-N highest absolute IC features

Forward label = 20-day log return from as_of_t1 close to as_of_t1+~30d close。
"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db_utils import get_db_connection

V01_FEATURES = {
    # 27 base features (v0.1 baseline)
    "log_return_20d", "log_return_60d", "log_return_252d",
    "volatility_60d", "volatility_252d",
    "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d",
    "turnover_mean_60d", "zero_volume_ratio_252d",
    "revenue_yoy_12m", "revenue_yoy_3m",
    "eps_sum_4q", "net_income_positive_ratio_8q",
    "foreign_net_20d", "foreign_net_60d", "trust_net_20d", "trust_net_60d",
    "margin_ratio_60d",
    "theme_strength", "theme_is_semiconductor",
    "macro_dff_level", "macro_vix_level", "macro_t10y2y_level", "macro_unrate_yoy",
}

# Pillar mapping(per §0.1 / §0.2 / §0.3 doctrine)
PILLAR_MAP = {
    "§0.1": {"price", "liquidity", "fundamental", "institutional", "value", "quality", "investment"},
    "§0.2": {"pareto"},
    "§0.3": {"macro", "kwave", "multi_cycle", "microstructure"},
    "§0.cross": {"theme", "interaction"},
}


def spearman_ic(pairs):
    """Compute Spearman rank correlation."""
    n = len(pairs)
    if n < 10:
        return None
    # Rank both x and y
    x_sorted = sorted(range(n), key=lambda i: pairs[i][0])
    y_sorted = sorted(range(n), key=lambda i: pairs[i][1])
    rank_x = [0.0] * n
    rank_y = [0.0] * n
    # Average ranks for ties
    for r, i in enumerate(x_sorted):
        rank_x[i] = r + 1
    for r, i in enumerate(y_sorted):
        rank_y[i] = r + 1
    sum_d2 = sum((rank_x[i] - rank_y[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * sum_d2) / (n * (n * n - 1))


def main():
    import math

    as_of_t1 = date(2026, 4, 30)
    feature_set_id = "fs_20260430_feature_set_v0_4_ablation_20260430"
    horizon_days = 20

    print(f"📊 Phase D Ablation:as_of_t1={as_of_t1} / feature_set={feature_set_id} / horizon={horizon_days}d")

    conn = get_db_connection()
    cur = conn.cursor()

    # 1. 找 T1 之 trading day(close price reference)
    cur.execute(
        'SELECT MAX(date) FROM "TaiwanStockPriceAdj" WHERE date <= %s',
        (as_of_t1,),
    )
    t1_date = cur.fetchone()[0]

    # 2. 找 T1 + ~30 calendar days(>= 20 trading days)之 reference
    # 過濾 partial-sync dates(<1000 stocks)
    t2_target = t1_date + timedelta(days=35)
    cur.execute(
        """
        SELECT date FROM "TaiwanStockPriceAdj" WHERE date <= %s
        GROUP BY date HAVING COUNT(DISTINCT stock_id) >= 1000
        ORDER BY date DESC LIMIT 1
        """,
        (t2_target,),
    )
    t2_date = cur.fetchone()[0]

    print(f"   T1 trading day:{t1_date}")
    print(f"   T2 trading day:{t2_date}({(t2_date - t1_date).days} calendar days forward)")

    # 3. Load close at T1 and T2 per stock
    cur.execute(
        'SELECT stock_id, "close"::numeric FROM "TaiwanStockPriceAdj" WHERE date = %s',
        (t1_date,),
    )
    p_t1 = {sid: float(c) for sid, c in cur.fetchall()}
    cur.execute(
        'SELECT stock_id, "close"::numeric FROM "TaiwanStockPriceAdj" WHERE date = %s',
        (t2_date,),
    )
    p_t2 = {sid: float(c) for sid, c in cur.fetchall()}

    # 4. Compute forward log returns
    labels = {}
    for sid, c1 in p_t1.items():
        c2 = p_t2.get(sid)
        if c1 and c2 and c1 > 0 and c2 > 0:
            labels[sid] = math.log(c2 / c1)
    print(f"   Forward labels coverage:{len(labels)} stocks")

    # 5. Load all features at T1
    cur.execute(
        """
        SELECT fv.feature_name, fd.feature_group, fv.stock_id, fv.feature_value::numeric
        FROM feature_values fv
        JOIN feature_definition fd USING (feature_set_id, feature_name)
        WHERE fv.feature_set_id = %s
        """,
        (feature_set_id,),
    )
    by_feature = {}
    feature_group_map = {}
    for fname, fgroup, sid, fval in cur.fetchall():
        by_feature.setdefault(fname, []).append((sid, float(fval) if fval is not None else None))
        feature_group_map[fname] = fgroup

    cur.close()
    conn.close()

    # 6. Compute IC per feature
    feature_ic = {}
    for fname, sid_vals in by_feature.items():
        pairs = []
        for sid, v in sid_vals:
            if v is None:
                continue
            label = labels.get(sid)
            if label is None:
                continue
            pairs.append((v, label))
        if len(pairs) < 30:
            continue
        ic = spearman_ic(pairs)
        if ic is None:
            continue
        feature_ic[fname] = (ic, len(pairs), feature_group_map[fname])

    # 7. 分組統計
    def pillar_of(group):
        for p, groups in PILLAR_MAP.items():
            if group in groups:
                return p
        return "§0.unknown"

    print(f"\n📈 Per-feature IC(64 features 中 {len(feature_ic)} 個有效)")
    print("=" * 100)
    rows_sorted = sorted(feature_ic.items(), key=lambda x: -abs(x[1][0]))
    print(f"{'Rank':>4}  {'Feature':40}  {'Group':18}  {'Pillar':8}  {'IC':>8}  {'N':>6}")
    print("-" * 100)
    for rank, (fname, (ic, n, group)) in enumerate(rows_sorted, 1):
        print(f"{rank:>4}  {fname:40}  {group:18}  {pillar_of(group):8}  {ic:>+8.4f}  {n:>6}")

    # 8. v0.1 vs v0.4 aggregate
    print("\n" + "=" * 100)
    print("📊 v0.1 vs v0.4 IC aggregate(absolute mean)")
    print("=" * 100)
    v01_ics = [abs(ic) for fname, (ic, _, _) in feature_ic.items() if fname in V01_FEATURES]
    v04_ics = [abs(ic) for _, (ic, _, _) in feature_ic.items()]
    new_ics = [abs(ic) for fname, (ic, _, _) in feature_ic.items() if fname not in V01_FEATURES]

    def stats(arr):
        if not arr:
            return None
        return {
            "n": len(arr),
            "mean_abs_ic": sum(arr) / len(arr),
            "max_abs_ic": max(arr),
            "min_abs_ic": min(arr),
        }

    s_v01 = stats(v01_ics)
    s_v04 = stats(v04_ics)
    s_new = stats(new_ics)

    print(f"v0.1(27 base)        : n={s_v01['n']}  mean|IC|={s_v01['mean_abs_ic']:.4f}  max|IC|={s_v01['max_abs_ic']:.4f}")
    print(f"v0.4(64 全集)         : n={s_v04['n']}  mean|IC|={s_v04['mean_abs_ic']:.4f}  max|IC|={s_v04['max_abs_ic']:.4f}")
    print(f"v0.4 新增(37 features): n={s_new['n']}  mean|IC|={s_new['mean_abs_ic']:.4f}  max|IC|={s_new['max_abs_ic']:.4f}")

    # 9. Per-pillar aggregate
    print("\n" + "=" * 100)
    print("📊 Per-pillar IC aggregate")
    print("=" * 100)
    per_pillar = {}
    for fname, (ic, _, group) in feature_ic.items():
        p = pillar_of(group)
        per_pillar.setdefault(p, []).append(abs(ic))
    for p in sorted(per_pillar.keys()):
        arr = per_pillar[p]
        print(f"{p:12}: n={len(arr):>3}  mean|IC|={sum(arr)/len(arr):.4f}  max|IC|={max(arr):.4f}")

    # 10. T_CA-1 證偽 gate 對應實證
    print("\n" + "=" * 100)
    print("🎯 §14.7-CA T_CA-1 證偽 gate baseline(單 as_of_date 簡化版)")
    print("=" * 100)
    delta = s_v04["mean_abs_ic"] - s_v01["mean_abs_ic"]
    threshold = 0.02
    verdict = "✅ PASS" if delta >= threshold else "⚠️ NOT YET"
    print(f"v0.4 mean|IC| - v0.1 mean|IC| = {delta:+.4f}(threshold ≥ {threshold:+.4f})")
    print(f"Verdict:{verdict}")
    print(f"\n⚠️  注意:本 ablation 為 single-date IC scan(非 walk-forward IC distribution),")
    print(f"   作為 Phase D-lite 證據;完整 walk-forward 證偽需 8+ historical points cross-validation。")


if __name__ == "__main__":
    main()
