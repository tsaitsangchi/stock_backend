"""
audit_per_stock_feature_validity.py — Per-stock × per-feature completeness + correctness audit
================================================================================
最後更新日期: 2026-05-28
治權: §14.7-CI strict feature validity gate enforce check

對 v0.14 strict active universe(N=1,541)× 37 unique spec features 做:
1. Completeness:每股每 feature 必有 non-null value(per §14.7-CI 治權)
2. Correctness:每 feature 之 statistical distribution + outlier check
3. Sample spot check:5 隨機 stocks 顯示 full feature 值
"""
from __future__ import annotations
import sys, logging
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

# 37 unique spec features(per §14.7-CA + §14.7-CI 治權)
SPEC_FEATURES = {
    # §0.1 第一性原理 (16)
    "§0.1 Momentum": ["log_return_20d", "log_return_60d", "log_return_252d"],
    "§0.1 Volatility": ["upside_volatility_60d", "downside_volatility_60d", "convexity_60d"],
    "§0.1 Liquidity": ["avg_daily_value_log_60d", "amihud_illiquidity_60d", "zero_volume_ratio_252d"],
    "§0.1 Value": ["pe_ratio", "pb_ratio", "dividend_yield"],
    "§0.1 Quality": ["roe_ttm", "operating_margin_ttm"],
    "§0.1 Investment": ["revenue_yoy_3m_log", "asset_growth_yoy"],
    # §0.2 八二法則 (7, amihud 共用)
    "§0.2 Pareto": ["right_tail_concentration_60d", "barbell_balance_60d", "preferential_attachment_60d",
                    "fitness_signal_60d", "right_tail_returns_skew_252d",
                    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector"],
    # §0.3 康波週期 (14, broadcast)
    "§0.3.1 K-wave": ["kwave_tech_paradigm_strength", "kwave_credit_cycle_phase",
                      "kwave_credit_to_gdp_gap", "kwave_demographics_trend",
                      "kwave_commodity_supercycle", "kwave_phase_indicator"],
    "§0.3.2 Multi-cycle": ["mc_monetary_regime", "mc_yield_curve_inversion",
                            "mc_oil_juglar_phase", "mc_semi_kitchin", "mc_shipping_juglar"],
    "§0.3.3 Microstructure": ["ms_volatility_regime", "ms_vix_term_structure", "ms_market_stress"],
}

# 合理 range(per feature semantic)— for correctness check
FEATURE_RANGES = {
    # Momentum: log returns 應 [-1, 5]
    "log_return_20d": (-1.5, 3.0), "log_return_60d": (-2.0, 5.0), "log_return_252d": (-3.0, 8.0),
    # Volatility: positive bounded
    "upside_volatility_60d": (0, 0.5), "downside_volatility_60d": (0, 0.5), "convexity_60d": (-0.5, 0.5),
    # Liquidity
    "avg_daily_value_log_60d": (3, 12), "amihud_illiquidity_60d": (0, 1e-3),
    "zero_volume_ratio_252d": (0, 1),
    # Value
    "pe_ratio": (0, 1000), "pb_ratio": (0, 50), "dividend_yield": (0, 30),
    # Quality
    "roe_ttm": (-2, 2), "operating_margin_ttm": (-5, 5),
    # Investment
    "revenue_yoy_3m_log": (-10, 10), "asset_growth_yoy": (-1, 10),
    # Pareto
    "right_tail_concentration_60d": (0, 1), "barbell_balance_60d": (0, 1),
    "preferential_attachment_60d": (3, 14), "fitness_signal_60d": (0, 1e10),
    "right_tail_returns_skew_252d": (-10, 10),
    "liquidity_rank_pct_sector_60d": (0, 1), "size_log_zscore_sector": (-5, 5),
    # K-wave: broadcast(同值 across all stocks)
    "kwave_tech_paradigm_strength": (-1, 1), "kwave_credit_cycle_phase": (-0.5, 0.5),
    "kwave_credit_to_gdp_gap": (50, 200), "kwave_demographics_trend": (-1, 1),
    "kwave_commodity_supercycle": (-0.5, 0.5), "kwave_phase_indicator": (-50, 50),
    # Multi-cycle
    "mc_monetary_regime": (-0.5, 0.5), "mc_yield_curve_inversion": (-3, 3),
    "mc_oil_juglar_phase": (-2, 2), "mc_semi_kitchin": (-2, 2), "mc_shipping_juglar": (-2, 2),
    # Microstructure
    "ms_volatility_regime": (5, 80), "ms_vix_term_structure": (-1, 3), "ms_market_stress": (0, 1),
}


def main():
    conn = get_db_conn()
    try:
        cur = conn.cursor()

        # 1. 取 active universe
        cur.execute("""
            SELECT m.stock_id FROM core_universe_membership m
            JOIN core_universe_snapshot s ON s.snapshot_id = m.snapshot_id
            WHERE s.status='committed' AND m.core_tier='core_universe'
            ORDER BY m.stock_id
        """)
        universe = [r[0] for r in cur.fetchall()]
        N = len(universe)

        cur.execute("SELECT feature_set_id FROM feature_store_snapshot WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1")
        fs_id = cur.fetchone()[0]

        logger.info("=" * 80)
        logger.info(f"§14.7-CI Per-stock × Per-feature Validity Audit(N={N})")
        logger.info(f"Active feature_set: {fs_id}")
        logger.info("=" * 80)

        # 2. Per-feature 統計 audit
        all_features = []
        for _, fs in SPEC_FEATURES.items():
            all_features.extend(fs)
        unique_features = sorted(set(all_features))

        logger.info(f"\n📊 Part 1:Per-feature aggregate stats(N={N} × {len(unique_features)} unique features)")
        logger.info("-" * 80)
        logger.info(f"{'Feature':40} {'N':>5} {'Min':>12} {'Median':>12} {'Max':>12} {'Range OK':>10}")
        logger.info("-" * 80)

        per_feature_stats = {}
        total_completeness_ok = 0
        total_correctness_ok = 0
        for feat in unique_features:
            cur.execute(f"""
                SELECT COUNT(*),
                  MIN(feature_value::numeric), AVG(feature_value::numeric),
                  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY feature_value::numeric),
                  MAX(feature_value::numeric)
                FROM feature_values
                WHERE feature_set_id=%s AND feature_name=%s
                  AND stock_id = ANY(%s)
            """, (fs_id, feat, universe))
            row = cur.fetchone()
            n, mn, avg, med, mx = row
            n = n or 0
            mn = float(mn) if mn is not None else None
            mx = float(mx) if mx is not None else None
            med = float(med) if med is not None else None

            # Completeness check
            complete_ok = (n == N)
            if complete_ok:
                total_completeness_ok += 1

            # Correctness check (range)
            range_ok = True
            if feat in FEATURE_RANGES and mn is not None and mx is not None:
                lo, hi = FEATURE_RANGES[feat]
                if mn < lo or mx > hi:
                    range_ok = False
            if range_ok:
                total_correctness_ok += 1

            per_feature_stats[feat] = (n, mn, med, mx, complete_ok, range_ok)
            mn_s = f"{mn:>12.4g}" if mn is not None else f"{'NULL':>12}"
            med_s = f"{med:>12.4g}" if med is not None else f"{'NULL':>12}"
            mx_s = f"{mx:>12.4g}" if mx is not None else f"{'NULL':>12}"
            status = "✅" if (complete_ok and range_ok) else ("⚠️" if complete_ok else "❌")
            logger.info(f"{feat:40} {n:>5} {mn_s} {med_s} {mx_s} {'✅' if range_ok else '❌':>10}")

        # 3. Per-feature summary
        logger.info(f"\n📊 Part 2:Audit summary")
        logger.info("-" * 80)
        logger.info(f"  Completeness(每 feature × {N} stocks 全有值):{total_completeness_ok}/{len(unique_features)}")
        logger.info(f"  Correctness(每 feature value 在合理 range):{total_correctness_ok}/{len(unique_features)}")

        # 4. Per-stock completeness check
        logger.info(f"\n📊 Part 3:Per-stock × all features completeness")
        cur.execute(f"""
            SELECT stock_id, COUNT(DISTINCT feature_name) AS n_features
            FROM feature_values
            WHERE feature_set_id=%s AND feature_name = ANY(%s) AND stock_id = ANY(%s)
            GROUP BY stock_id
        """, (fs_id, unique_features, universe))
        stock_coverage = {r[0]: r[1] for r in cur.fetchall()}
        n_stocks_complete = sum(1 for s in universe if stock_coverage.get(s, 0) == len(unique_features))
        logger.info(f"  Stocks 全 {len(unique_features)}/{len(unique_features)} features:{n_stocks_complete}/{N}")
        if n_stocks_complete == N:
            logger.info(f"  🎯 完整 100% × 100% × 100% per §14.7-CI strict gate")

        # 5. Sample 5 stocks spot check
        logger.info(f"\n📊 Part 4:Sample spot check(5 random stocks)")
        import random
        random.seed(42)
        samples = random.sample(universe, min(5, N))
        for sid in samples:
            cur.execute("""
                SELECT feature_name, feature_value::numeric FROM feature_values
                WHERE feature_set_id=%s AND stock_id=%s AND feature_name = ANY(%s)
                ORDER BY feature_name
            """, (fs_id, sid, unique_features))
            features = dict(cur.fetchall())
            logger.info(f"\n  Stock {sid}:{len(features)}/{len(unique_features)} features")
            for k in ["log_return_60d", "pe_ratio", "roe_ttm", "operating_margin_ttm",
                      "kwave_credit_to_gdp_gap", "mc_yield_curve_inversion"]:
                v = features.get(k)
                logger.info(f"    {k:40} = {float(v):>15.6f}" if v is not None else f"    {k:40} = NULL")

        # 6. Final verdict
        logger.info(f"\n" + "=" * 80)
        logger.info(f"§14.7-CI VALIDITY AUDIT FINAL VERDICT")
        logger.info("=" * 80)
        logger.info(f"  Universe N: {N}")
        logger.info(f"  Unique spec features: {len(unique_features)}")
        logger.info(f"  Completeness(per-feature × stocks):{total_completeness_ok}/{len(unique_features)} features 全 {N} stocks 有值")
        logger.info(f"  Correctness(per-feature value range):{total_correctness_ok}/{len(unique_features)} features in expected range")
        logger.info(f"  Per-stock completeness:{n_stocks_complete}/{N} stocks 全 {len(unique_features)} features")
        all_pass = (total_completeness_ok == len(unique_features) and
                    total_correctness_ok == len(unique_features) and
                    n_stocks_complete == N)
        if all_pass:
            logger.info(f"\n  🎯 §14.7-CI Strict Feature Validity Gate:**PASS**")
            logger.info(f"  ✅ {N} stocks × {len(unique_features)} features = {N*len(unique_features):,} entries 全 complete + correct")
        else:
            logger.warning(f"\n  ⚠️ Some checks failed — review per-feature table above")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
