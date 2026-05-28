"""
audit_backtest_walk_forward.py — Real 8-panel walk-forward backtest
================================================================================
最後更新日期: 2026-05-28
治權: §14.7-CV Backtest Production Closure(將 walk-forward IC 延伸為 portfolio P&L 實證)

對 8 個 historical snapshots(fs_20260105 → fs_20260415)+ 30d forward returns
跑 real portfolio backtest:
  1. Top-20 long candidates per panel(per model predictions)
  2. Equal-weight portfolio
  3. Forward 30d return per panel
  4. Cross-panel Sharpe / MDD / Win Rate
  5. Compare vs benchmark(equal-weight universe)

Treaty gates(per §14.7-CV):
  - Cross-panel Sharpe > 0(positive expected return)
  - Win rate ≥ 50%(超過 random walk)
  - MDD ≤ 30%(controlled drawdown)
  - Top-20 outperform equal-weight universe(positive alpha)
"""
from __future__ import annotations
import sys, logging, math
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

# 8 panels with (as_of, label_date)
PANELS = [
    ("fs_20260105_feature_set_v0_4", "2026-01-05", "2026-02-04"),
    ("fs_20260120_feature_set_v0_4", "2026-01-20", "2026-02-19"),
    ("fs_20260205_feature_set_v0_4", "2026-02-05", "2026-03-07"),
    ("fs_20260220_feature_set_v0_4", "2026-02-20", "2026-03-22"),
    ("fs_20260305_feature_set_v0_4", "2026-03-05", "2026-04-04"),
    ("fs_20260316_feature_set_v0_4", "2026-03-16", "2026-04-15"),
    ("fs_20260401_feature_set_v0_4", "2026-04-01", "2026-05-01"),
    ("fs_20260415_feature_set_v0_4", "2026-04-15", "2026-05-15"),
]


def main():
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id WHERE s.status='committed' AND m.core_tier='core_universe'")
        universe = list({r[0] for r in cur.fetchall()})

        logger.info("=" * 110)
        logger.info("§14.7-CV Walk-Forward Real Backtest(Top-20 Long Strategy)")
        logger.info("=" * 110)
        logger.info(f"  Universe: {len(universe)} stocks(§14.7-CJ)")
        logger.info(f"  Panels: {len(PANELS)}")
        logger.info(f"  Horizon: 30 calendar days")
        logger.info(f"  Strategy: Equal-weight top-20 predictions vs equal-weight universe")
        logger.info("")

        # Simple top-20 selection: use 60d log return as predictor (top feature per IC)
        # 為何 60d:per §14.7-CS walk-forward audit,log_return_60d 為 top 預測 feature
        # 對於每 panel:取 top 20 highest 60d return → 等權 portfolio → forward 30d return

        panel_results = []
        for fs_id, as_of, label_date in PANELS:
            # Find nearest trading day for as_of(可能落非交易日)
            cur.execute("""
                SELECT MIN(date) FROM "TaiwanStockPriceAdj"
                WHERE date >= %s AND stock_id ~ '^[0-9]'
                  AND date <= (%s::date + INTERVAL '7 days')
            """, (as_of, as_of))
            r = cur.fetchone()
            actual_t0 = r[0] if r and r[0] else as_of

            # Find nearest trading day for label_date(forward 30d)
            cur.execute("""
                SELECT MIN(date) FROM "TaiwanStockPriceAdj"
                WHERE date >= %s AND stock_id ~ '^[0-9]'
                  AND date <= (%s::date + INTERVAL '10 days')
            """, (label_date, label_date))
            r = cur.fetchone()
            actual_t1 = r[0] if r and r[0] else label_date

            # Compute predictions: rank by log_return_60d at as_of
            cur.execute("""
                SELECT stock_id, feature_value::numeric FROM feature_values
                WHERE feature_set_id=%s AND feature_name='log_return_60d'
                AND stock_id=ANY(%s)
            """, (fs_id, universe))
            scores = {r[0]: float(r[1]) for r in cur.fetchall() if r[1] is not None}

            # Top 20
            top20 = sorted(scores.items(), key=lambda x: -x[1])[:20]
            top20_ids = [s[0] for s in top20]

            # Forward return using actual trading days
            cur.execute("""
                WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
                     t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
                SELECT t0.stock_id, LN(t1.close::numeric / t0.close::numeric)
                FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id
            """, (actual_t0, actual_t1))
            returns = {r[0]: float(r[1]) for r in cur.fetchall()}

            # Top-20 portfolio return(equal-weight)
            top20_rets = [returns[s] for s in top20_ids if s in returns]
            top20_ret = float(np.mean(top20_rets)) if top20_rets else 0
            top20_std = float(np.std(top20_rets)) if top20_rets else 0

            # Universe equal-weight benchmark
            univ_rets = [returns[s] for s in universe if s in returns]
            univ_ret = float(np.mean(univ_rets)) if univ_rets else 0
            univ_std = float(np.std(univ_rets)) if univ_rets else 0

            alpha = top20_ret - univ_ret  # excess return over equal-weight benchmark
            panel_results.append({
                "panel": fs_id.replace("_feature_set_v0_4", "")[3:],
                "as_of": as_of,
                "label_date": label_date,
                "top20_ret": top20_ret,
                "top20_std": top20_std,
                "univ_ret": univ_ret,
                "univ_std": univ_std,
                "alpha": alpha,
                "n_top20_filled": len(top20_rets),
                "n_univ_filled": len(univ_rets),
            })

            logger.info(f"  Panel {as_of}: top20 ret={top20_ret:>+8.4f} | universe={univ_ret:>+8.4f} | alpha={alpha:>+8.4f} | N={len(top20_rets)}/{len(univ_rets)}")

        # Aggregate metrics
        logger.info("\n" + "=" * 110)
        logger.info("Cross-Panel Statistics")
        logger.info("=" * 110)
        top20_returns = [p["top20_ret"] for p in panel_results]
        univ_returns = [p["univ_ret"] for p in panel_results]
        alphas = [p["alpha"] for p in panel_results]

        # Strategy metrics
        mean_ret = float(np.mean(top20_returns))
        std_ret = float(np.std(top20_returns, ddof=1))
        sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0  # annualize: 12 panels/year
        win_rate = sum(1 for r in top20_returns if r > 0) / len(top20_returns)
        max_panel_loss = min(top20_returns)
        # Cumulative compound for MDD
        cum_returns = []
        cum = 0
        for r in top20_returns:
            cum += r  # using log returns,additive
            cum_returns.append(cum)
        peak = cum_returns[0]; mdd = 0
        for c in cum_returns:
            if c > peak: peak = c
            dd = peak - c
            if dd > mdd: mdd = dd

        # Benchmark metrics
        bench_mean = float(np.mean(univ_returns))
        bench_std = float(np.std(univ_returns, ddof=1))

        # Alpha statistics
        mean_alpha = float(np.mean(alphas))
        std_alpha = float(np.std(alphas, ddof=1))
        info_ratio = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0

        logger.info(f"\n  Top-20 Strategy:")
        logger.info(f"    Mean 30d return:        {mean_ret:>+8.4f}({mean_ret*100:>+6.2f}%)")
        logger.info(f"    Std 30d return:         {std_ret:>+8.4f}")
        logger.info(f"    Sharpe(annualized):    {sharpe:>+8.4f}")
        logger.info(f"    Win rate:               {win_rate*100:>5.1f}%")
        logger.info(f"    Max panel loss:         {max_panel_loss:>+8.4f}")
        logger.info(f"    Max drawdown:           {mdd:>+8.4f}({mdd*100:>5.2f}%)")
        logger.info(f"    Cumulative return:      {cum_returns[-1]:>+8.4f}")

        logger.info(f"\n  Equal-Weight Universe Benchmark:")
        logger.info(f"    Mean 30d return:        {bench_mean:>+8.4f}({bench_mean*100:>+6.2f}%)")
        logger.info(f"    Std 30d return:         {bench_std:>+8.4f}")

        logger.info(f"\n  Alpha(Top-20 - Universe):")
        logger.info(f"    Mean alpha:             {mean_alpha:>+8.4f}({mean_alpha*100:>+6.2f}%)")
        logger.info(f"    Std alpha:              {std_alpha:>+8.4f}")
        logger.info(f"    Information Ratio:      {info_ratio:>+8.4f}")

        # Treaty Gates
        logger.info("\n" + "=" * 110)
        logger.info("§14.7-CV Treaty Gates")
        logger.info("=" * 110)
        gate_1 = "✅ PASS" if sharpe > 0 else "❌ VIOLATION"
        logger.info(f"  Gate CV-1(Sharpe > 0):              {gate_1}({sharpe:.4f})")
        gate_2 = "✅ PASS" if win_rate >= 0.5 else "❌ VIOLATION"
        logger.info(f"  Gate CV-2(Win rate ≥ 50%):         {gate_2}({win_rate*100:.1f}%)")
        gate_3 = "✅ PASS" if mdd <= 0.30 else f"⚠️ ALERT({mdd*100:.1f}%)"
        logger.info(f"  Gate CV-3(MDD ≤ 30%):              {gate_3}({mdd*100:.2f}%)")
        gate_4 = "✅ PASS" if mean_alpha > 0 else "❌ VIOLATION"
        logger.info(f"  Gate CV-4(Mean alpha > 0):         {gate_4}({mean_alpha:.4f})")

        all_pass = (sharpe > 0 and win_rate >= 0.5 and mdd <= 0.30 and mean_alpha > 0)
        if all_pass:
            logger.info(f"\n  🎯 §14.7-CV Backtest Gate: **PASS**")
            logger.info(f"  ✅ Strategy 在 8 panels walk-forward 證明:")
            logger.info(f"     - Sharpe={sharpe:.2f}(annualized)")
            logger.info(f"     - Win rate={win_rate*100:.0f}%")
            logger.info(f"     - 平均超額報酬 {mean_alpha*100:+.2f}% / 30d horizon")
        else:
            logger.warning(f"\n  ⚠️ §14.7-CV partial PASS / VIOLATION")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
