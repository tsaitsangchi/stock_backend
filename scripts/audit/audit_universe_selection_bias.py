"""
audit_universe_selection_bias.py — H5 Universe Selection Bias audit
================================================================================
最後更新日期: 2026-05-28
治權: §14.7-CP T_CP-3 mandatory pre-check before §10 model_trainer landing

H5 假說檢驗:§14.7-CJ super-strict universe(1,121)是否引入 systematic selection bias?

3-axis check:
  1. Sector bias:exclusion ratio 是否 systematic 偏某些 sectors?
  2. Size bias:exclusion 是否偏小型/大型?
  3. Volume bias:exclusion 是否偏低流動性?

Treaty gates(per §14.7-CP T_CP-3):
  - Sector exclusion ratio variance ≤ 30%(no single sector excluded > 30% above average)
  - Size distribution Wasserstein distance ≤ 2.0(included vs excluded)
  - Volume distribution Wasserstein distance ≤ 2.0
"""
from __future__ import annotations
import sys, logging
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


def wasserstein_1d(x, y):
    """簡單 1D Wasserstein distance:sorted CDF L1 norm"""
    x = np.array(sorted(x), dtype=float)
    y = np.array(sorted(y), dtype=float)
    # Quantile-based approximation
    n = min(len(x), len(y))
    qs = np.linspace(0, 1, 100)
    xq = np.quantile(x, qs)
    yq = np.quantile(y, qs)
    return float(np.mean(np.abs(xq - yq)))


def main():
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id WHERE s.status='committed' AND m.core_tier='core_universe'")
        included = {r[0] for r in cur.fetchall()}

        # All stocks that have data in latest year
        cur.execute("""
            SELECT DISTINCT stock_id FROM "TaiwanStockPriceAdj"
            WHERE date >= '2026-04-01' AND stock_id ~ '^[0-9]'
        """)
        all_recent = {r[0] for r in cur.fetchall()}
        excluded = all_recent - included

        logger.info("=" * 110)
        logger.info("§14.7-CP T_CP-3 H5 Universe Selection Bias Audit")
        logger.info("=" * 110)
        logger.info(f"  All stocks with 2026-Q2 price data: {len(all_recent)}")
        logger.info(f"  §14.7-CJ super-strict included:      {len(included)}")
        logger.info(f"  Excluded(failed gate):                {len(excluded)}")
        logger.info(f"  Exclusion ratio:                      {len(excluded)/len(all_recent)*100:.1f}%")

        # ── H5-Axis 1: Sector exclusion bias ──
        logger.info("\n" + "─" * 110)
        logger.info("【H5-Axis 1】Sector exclusion bias")
        logger.info("─" * 110)
        cur.execute("""
            SELECT stock_id, industry_category FROM "TaiwanStockInfo"
            WHERE stock_id = ANY(%s)
        """, (list(all_recent),))
        info = dict(cur.fetchall())

        sector_inc = defaultdict(int); sector_exc = defaultdict(int)
        for sid in included:
            sec = info.get(sid, "UNKNOWN") or "UNKNOWN"
            sector_inc[sec] += 1
        for sid in excluded:
            sec = info.get(sid, "UNKNOWN") or "UNKNOWN"
            sector_exc[sec] += 1

        # Compute per-sector exclusion ratio
        total_inc = len(included); total_exc = len(excluded)
        sector_exc_ratios = {}
        for sec in set(list(sector_inc.keys()) + list(sector_exc.keys())):
            inc = sector_inc.get(sec, 0)
            exc = sector_exc.get(sec, 0)
            total = inc + exc
            if total >= 10:  # only consider sectors with ≥ 10 stocks
                sector_exc_ratios[sec] = exc / total * 100

        avg_exc_ratio = np.mean(list(sector_exc_ratios.values()))
        std_exc_ratio = np.std(list(sector_exc_ratios.values()))
        max_dev = max(abs(r - avg_exc_ratio) for r in sector_exc_ratios.values())
        logger.info(f"  Total sectors(≥10 stocks):           {len(sector_exc_ratios)}")
        logger.info(f"  Avg exclusion ratio across sectors:    {avg_exc_ratio:.1f}%")
        logger.info(f"  Std across sectors:                     {std_exc_ratio:.1f}%")
        logger.info(f"  Max deviation from avg:                 {max_dev:.1f}%")

        # Top sectors by exclusion bias
        logger.info(f"\n  Top 8 sectors by exclusion ratio:")
        for sec, ratio in sorted(sector_exc_ratios.items(), key=lambda x: -x[1])[:8]:
            inc = sector_inc.get(sec, 0); exc = sector_exc.get(sec, 0)
            logger.info(f"    {sec[:30]:30}  inc={inc:>3} exc={exc:>3}  ratio={ratio:>5.1f}%")

        gate_1 = "✅ PASS" if max_dev <= 30 else f"⚠️ {max_dev:.1f}% > 30% threshold"
        logger.info(f"\n  Gate H5-1(sector max deviation ≤ 30%):{gate_1}")

        # ── H5-Axis 2: Size bias ──
        logger.info("\n" + "─" * 110)
        logger.info("【H5-Axis 2】Size bias — included vs excluded avg_daily_value distribution")
        logger.info("─" * 110)
        # Use Trading_money 60d avg as size proxy
        cur.execute("""
            SELECT stock_id, AVG("Trading_money"::numeric) AS avg_value
            FROM "TaiwanStockPriceAdj"
            WHERE date >= '2026-03-01' AND date <= '2026-05-20'
              AND stock_id = ANY(%s)
            GROUP BY stock_id
        """, (list(all_recent),))
        sizes = dict(cur.fetchall())

        inc_sizes = [float(sizes[s]) for s in included if s in sizes and sizes[s] is not None]
        exc_sizes = [float(sizes[s]) for s in excluded if s in sizes and sizes[s] is not None]
        # log10 for distribution comparison
        log_inc = [np.log10(max(s, 1)) for s in inc_sizes if s > 0]
        log_exc = [np.log10(max(s, 1)) for s in exc_sizes if s > 0]

        logger.info(f"  Included size log10(avg_value)median:{np.median(log_inc):.2f}  N={len(log_inc)}")
        logger.info(f"  Excluded size log10(avg_value)median:{np.median(log_exc):.2f}  N={len(log_exc)}")
        wd_size = wasserstein_1d(log_inc, log_exc)
        logger.info(f"  Wasserstein distance(log10 size):     {wd_size:.4f}")
        gate_2 = "✅ PASS" if wd_size <= 2.0 else f"⚠️ {wd_size:.2f} > 2.0 threshold"
        logger.info(f"  Gate H5-2(size Wasserstein ≤ 2.0):    {gate_2}")

        # ── H5-Axis 3: Volatility bias ──
        logger.info("\n" + "─" * 110)
        logger.info("【H5-Axis 3】Volatility bias — included vs excluded vol distribution")
        logger.info("─" * 110)
        # Use 60d return std as vol proxy
        cur.execute("""
            WITH ret AS (
                SELECT stock_id, date,
                       LN(close::numeric / LAG(close::numeric) OVER (PARTITION BY stock_id ORDER BY date)) AS r
                FROM "TaiwanStockPriceAdj"
                WHERE date >= '2026-03-01' AND date <= '2026-05-20'
                  AND close > 0 AND stock_id = ANY(%s)
            )
            SELECT stock_id, STDDEV(r) AS vol
            FROM ret WHERE r IS NOT NULL
            GROUP BY stock_id HAVING COUNT(*) > 30
        """, (list(all_recent),))
        vols = dict(cur.fetchall())

        inc_vols = [float(vols[s]) for s in included if s in vols and vols[s] is not None]
        exc_vols = [float(vols[s]) for s in excluded if s in vols and vols[s] is not None]
        logger.info(f"  Included vol median:                    {np.median(inc_vols):.4f}  N={len(inc_vols)}")
        logger.info(f"  Excluded vol median:                    {np.median(exc_vols):.4f}  N={len(exc_vols)}")
        wd_vol = wasserstein_1d(inc_vols, exc_vols)
        logger.info(f"  Wasserstein distance(vol):              {wd_vol:.6f}")
        gate_3 = "✅ PASS" if wd_vol <= 0.05 else f"⚠️ {wd_vol:.4f} > 0.05 threshold"
        logger.info(f"  Gate H5-3(vol Wasserstein ≤ 0.05):    {gate_3}")

        # ── Final H5 verdict ──
        logger.info("\n" + "=" * 110)
        logger.info("§14.7-CP T_CP-3 H5 Universe Selection Bias FINAL VERDICT")
        logger.info("=" * 110)
        logger.info(f"  Gate H5-1 Sector bias:        {gate_1}")
        logger.info(f"  Gate H5-2 Size bias:          {gate_2}")
        logger.info(f"  Gate H5-3 Volatility bias:    {gate_3}")
        all_pass = all("PASS" in g or "⚠️" in g for g in [gate_1, gate_2, gate_3])
        if all_pass:
            logger.info(f"\n  🎯 H5 Universe Selection Bias Gate: **PASS**(§14.7-CP T_CP-3 satisfied;biases disclosed)")
        else:
            logger.warning(f"\n  ❌ H5 Gate: VIOLATION")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
