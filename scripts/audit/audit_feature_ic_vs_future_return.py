"""
audit_feature_ic_vs_future_return.py — 43 features × forward return Spearman IC empirical audit
================================================================================
最後更新日期: 2026-05-28
治權: §14.7-CM Empirical IC Doctrine — 模型訓練有效性量化證據

對 §14.7-CL 43 canonical SPEC features 計算 Spearman rank IC vs forward N-day return:
1. Picks historical fs snapshot for features(default: 至少 14 trading days 前)
2. Computes forward return from t to t+N(default N=14, configurable)
3. Spearman rank correlation per feature(no scipy needed)
4. Statistical significance test(|t| > 1.96 → p<.05, two-tailed)
5. Output by pillar + sorted ranking + summary stats

用途:
- 每週 cron 重跑(per §14.7-CM 治權)— track IC degradation
- 新 feature 入 §14.7-CL SPEC 前須通過 |IC| > 0.03 baseline 檢驗
- 治權判準十七純化軸:Feature-Empirical-IC

Usage:
  python audit_feature_ic_vs_future_return.py [--horizon 14]

CLI args(optional):
  --horizon N      forward window trading days(default 14)
  --as-of DATE     specific feature snapshot date(default auto-pick t-horizon)
"""
from __future__ import annotations
import sys, argparse, math, logging
from pathlib import Path
from collections import defaultdict

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# 43 SPEC features per §14.7-CL canonical scope(9 explicit subgroups)
SPEC_43 = [
    ("§0.1.A Momentum", ["log_return_20d", "log_return_60d", "log_return_252d",
                         "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d"]),
    ("§0.1.B Volatility", ["upside_volatility_60d", "downside_volatility_60d", "convexity_60d",
                           "volatility_60d", "volatility_252d",
                           "upside_capture_60d", "downside_capture_60d"]),
    ("§0.1.C Liquidity", ["avg_daily_value_log_60d", "avg_daily_value_log_252d",
                          "amihud_illiquidity_60d", "zero_volume_ratio_252d",
                          "turnover_mean_60d"]),
    ("§0.1.D Value", ["pe_ratio", "pb_ratio", "dividend_yield"]),
    ("§0.1.E Quality", ["roe_ttm", "operating_margin_ttm",
                        "eps_sum_4q", "net_income_positive_ratio_8q"]),
    ("§0.1.F Investment", ["revenue_yoy_3m_log", "asset_growth_yoy",
                           "revenue_yoy_3m", "revenue_yoy_12m"]),
    ("§0.2.A Pareto", ["right_tail_concentration_60d", "barbell_balance_60d",
                       "preferential_attachment_60d", "fitness_signal_60d",
                       "right_tail_returns_skew_252d",
                       "liquidity_rank_pct_sector_60d", "size_log_zscore_sector"]),
    ("§0.2.B Institutional", ["foreign_net_20d", "foreign_net_60d",
                              "trust_net_20d", "trust_net_60d",
                              "margin_ratio_60d"]),
    ("§0.2.C Theme", ["theme_strength", "theme_is_semiconductor"]),
]


def spearman_ic(x, y):
    """Spearman rank correlation;numpy 實作 — 無 scipy 依賴"""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def find_feature_snapshot(cur, horizon_days):
    """Pick fs snapshot that's ≥ horizon trading days old(or earliest available)"""
    cur.execute("""
        SELECT feature_set_id, as_of_date FROM feature_store_snapshot
        WHERE as_of_date <= (CURRENT_DATE - INTERVAL '%s days')
        ORDER BY as_of_date DESC LIMIT 1
    """, (horizon_days + 7,))  # +7 cushion for weekends
    row = cur.fetchone()
    if row:
        return row
    # Fallback: any historical snapshot
    cur.execute("""
        SELECT feature_set_id, as_of_date FROM feature_store_snapshot
        ORDER BY as_of_date ASC LIMIT 1
    """)
    return cur.fetchone()


def find_forward_end_date(cur, t0, horizon_days):
    """Pick forward end date — closest trading day to t0+horizon with sufficient data"""
    cur.execute("""
        SELECT date, COUNT(DISTINCT stock_id)
        FROM "TaiwanStockPriceAdj"
        WHERE date BETWEEN %s::date AND (%s::date + INTERVAL '%s days')
        GROUP BY date
        HAVING COUNT(DISTINCT stock_id) > 1000
        ORDER BY date DESC LIMIT 1
    """, (t0, t0, horizon_days + 14))
    row = cur.fetchone()
    return row[0] if row else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=14, help="Forward window trading days(default 14)")
    parser.add_argument("--as-of", type=str, default=None, help="Specific feature as_of date YYYY-MM-DD")
    args = parser.parse_args()

    conn = get_db_conn()
    try:
        cur = conn.cursor()

        # 1. Pick feature snapshot
        if args.as_of:
            cur.execute("""
                SELECT feature_set_id, as_of_date FROM feature_store_snapshot
                WHERE as_of_date = %s ORDER BY as_of_date DESC LIMIT 1
            """, (args.as_of,))
            snap = cur.fetchone()
        else:
            snap = find_feature_snapshot(cur, args.horizon)

        if not snap:
            logger.error("❌ No feature snapshot found")
            return
        fs_id, t0 = snap

        # 2. Pick forward end date
        t1 = find_forward_end_date(cur, t0, args.horizon)
        if not t1:
            logger.error(f"❌ No PriceAdj data for forward window from {t0} + {args.horizon} days")
            return

        actual_days = (t1 - t0).days
        logger.info("=" * 110)
        logger.info("§14.7-CM Empirical IC Audit — 43 features × forward return")
        logger.info("=" * 110)
        logger.info(f"Feature snapshot: {fs_id} as_of t={t0}")
        logger.info(f"Forward end date: t+={t1}(~{actual_days} calendar days)")
        logger.info(f"Configured horizon: {args.horizon} trading days")

        # 3. Current core universe
        cur.execute("""
            SELECT m.stock_id FROM core_universe_membership m
            JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
            WHERE s.status='committed' AND m.core_tier='core_universe'
        """)
        universe = {r[0] for r in cur.fetchall()}
        logger.info(f"Universe: {len(universe)} current core stocks(§14.7-CJ super-strict gate)")

        # 4. Forward returns
        cur.execute("""
            WITH t0p AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close > 0),
                 t1p AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close > 0)
            SELECT t0p.stock_id, LN(t1p.close::numeric / t0p.close::numeric)
            FROM t0p JOIN t1p ON t0p.stock_id = t1p.stock_id
        """, (t0, t1))
        fwd_ret = {sid: float(ret) for sid, ret in cur.fetchall() if sid in universe}
        logger.info(f"Forward return samples: {len(fwd_ret)} stocks")
        if len(fwd_ret) < 100:
            logger.error(f"❌ Insufficient forward samples (n={len(fwd_ret)} < 100)")
            return
        rets = sorted(fwd_ret.values())
        logger.info(f"  Return: min={rets[0]:.4f}, median={rets[len(rets)//2]:.4f}, max={rets[-1]:.4f}")

        # 5. Features at t
        cur.execute("""
            SELECT stock_id, feature_name, feature_value::numeric
            FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s)
        """, (fs_id, list(universe)))
        features = defaultdict(dict)
        for sid, fname, val in cur.fetchall():
            if val is not None:
                features[fname][sid] = float(val)

        # 6. Compute IC
        sig_threshold = 1.96 / math.sqrt(max(len(fwd_ret) - 2, 1))
        logger.info(f"  顯著閾值 |IC| > {sig_threshold:.4f}(p<.05, n={len(fwd_ret)})")

        results = []
        for group, feats in SPEC_43:
            for fname in feats:
                if fname not in features:
                    results.append((group, fname, None, None, None, "❌ missing"))
                    continue
                pairs = [(features[fname][sid], fwd_ret[sid])
                         for sid in features[fname] if sid in fwd_ret]
                if len(pairs) < 100:
                    results.append((group, fname, None, None, len(pairs), "❌ few"))
                    continue
                xvals, yvals = zip(*pairs)
                if np.std(xvals) < 1e-10:
                    results.append((group, fname, None, None, len(pairs), "❌ const"))
                    continue
                ic = spearman_ic(xvals, yvals)
                n = len(pairs)
                t_stat = ic * math.sqrt((n - 2) / max(1 - ic ** 2, 1e-12))
                sig = "✅ p<.05" if abs(t_stat) > 1.96 else ("△ p<.10" if abs(t_stat) > 1.645 else "—")
                results.append((group, fname, ic, t_stat, n, sig))

        # 7. Output by group
        logger.info("=" * 110)
        logger.info(f"{'Feature':38} {'IC':>9} {'|IC|':>8} {'t-stat':>9} {'N':>5} {'Sig':>10}")
        logger.info("=" * 110)
        for group, feats in SPEC_43:
            logger.info(f"\n▼ {group}")
            for fname in feats:
                r = next((x for x in results if x[1] == fname), None)
                _, _, ic, t_stat, n, sig = r
                ic_s = f"{ic:>+9.4f}" if ic is not None else f"{'NA':>9}"
                abs_s = f"{abs(ic):>8.4f}" if ic is not None else f"{'NA':>8}"
                t_s = f"{t_stat:>+9.2f}" if t_stat is not None else f"{'NA':>9}"
                n_s = f"{n:>5}" if n is not None else f"{'NA':>5}"
                logger.info(f"  {fname:38} {ic_s} {abs_s} {t_s} {n_s} {sig:>10}")

        # 8. Top by |IC|
        res_ic = [r for r in results if r[2] is not None]
        logger.info("\n" + "=" * 110)
        logger.info(f"🏆 Top by |IC|(predictive strength ranking)— {len(res_ic)} features evaluated")
        logger.info("=" * 110)
        logger.info(f"{'Rank':>4} {'Feature':38} {'Group':22} {'IC':>9} {'t-stat':>9} {'Sig':>10}")
        sorted_res = sorted(res_ic, key=lambda r: abs(r[2]), reverse=True)
        for rank, (group, fname, ic, t, n, sig) in enumerate(sorted_res, 1):
            logger.info(f"{rank:>4} {fname:38} {group[:22]:22} {ic:>+9.4f} {t:>+9.2f} {sig:>10}")

        # 9. Summary
        n_sig = sum(1 for r in res_ic if abs(r[3]) > 1.96)
        n_pos = sum(1 for r in res_ic if r[2] > 0)
        n_neg = sum(1 for r in res_ic if r[2] < 0)
        mean_abs_ic = float(np.mean([abs(r[2]) for r in res_ic]))
        max_abs_ic = max(abs(r[2]) for r in res_ic)
        logger.info("\n" + "=" * 110)
        logger.info("§14.7-CM EMPIRICAL IC SUMMARY")
        logger.info("=" * 110)
        logger.info(f"  Evaluated: {len(res_ic)}/43 features")
        logger.info(f"  統計顯著(|t|>1.96, p<.05): {n_sig}/{len(res_ic)} = {n_sig/len(res_ic)*100:.1f}%")
        logger.info(f"  Positive IC: {n_pos}({n_pos/len(res_ic)*100:.0f}%) / Negative IC: {n_neg}")
        logger.info(f"  Mean |IC|: {mean_abs_ic:.4f}(literature threshold ~0.05 / treaty baseline 0.03)")
        logger.info(f"  Max  |IC|: {max_abs_ic:.4f}({sorted_res[0][1]})")

        # 10. Doctrine gate verdict(per §14.7-CM)
        passes_treaty = (mean_abs_ic > 0.03) and (n_sig >= len(res_ic) * 0.3)
        if passes_treaty:
            logger.info(f"\n  🎯 §14.7-CM Empirical IC Doctrine Gate: **PASS**")
            logger.info(f"  ✅ Mean |IC| > 0.03 baseline + ≥30% features p<.05 → 43 SPEC 集體具備模型訓練有效性")
        else:
            logger.warning(f"\n  ⚠️ §14.7-CM Gate: VIOLATION DETECTED")
            logger.warning(f"     Mean |IC| ({mean_abs_ic:.4f}) ≤ 0.03 OR Sig rate ({n_sig/len(res_ic)*100:.0f}%) < 30%")
            logger.warning(f"     → 觸發 feature re-evaluation per T_CM-3")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
