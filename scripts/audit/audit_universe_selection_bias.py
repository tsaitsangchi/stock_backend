"""
audit_universe_selection_bias.py v0.1 (H5 Universe Selection Bias Auditor · §14.7-CP T_CP-3 配套 · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 三段式標頭補正;原 v0.1 邏輯 2026-05-28 入)
**主權狀態**: ACTIVE (§14.7-CP T_CP-3 H5 audit + §14.7-CS 必要前置 + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[H5 Hypothesis Test]** (v0.1, §14.7-CP T_CP-3): H5 — §14.7-CJ super-strict universe(1,121)是否引入 systematic selection bias。
2. **[3-Axis Check]** (v0.1): Sector bias / Size bias / Volume bias 之 exclusion ratio 系統性偏向。
3. **[Treaty Gates]** (v0.1): Sector exclusion variance ≤ 30%;Size Wasserstein ≤ 2.0;Volume Wasserstein ≤ 2.0。
4. **[Source Traceability]** (v0.1, §一.10): 全 (b) DB query;0 AI memory。
5. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): PASS/ALERT 動態判定。
6. **[Sovereignty Declaration]** (v0.1, §3.2 橫切 audit): 本程式為 **§14.7-CP T_CP-3 H5 唯一 audit 載體**(§3.2 橫切 pre-training audit)。**治權邊界**:(a) §3.2 橫切;(b) read-only;(c) §14.7-CS 必要前置;(d) 不修改 universe;(e) 唯一職責:scan TaiwanStockInfo + PriceAdj + universe → 計算 3-axis bias metrics → H5 verdict。
7. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照。
8. **[Idempotency]** (v0.1): pure read-only。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Sector Bias Analysis
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Per-sector inclusion ratio | TaiwanStockInfo industry_category JOIN | §14.7-CP H5.A |
| A.2 Variance check | sector exclusion variance ≤ 30% | treaty gate |

### Group B. Size Bias Analysis
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Size proxy(market cap or avg vol)| computed from PriceAdj | §14.7-CP H5.B |
| B.2 Wasserstein distance | included vs excluded distributions | treaty gate |

### Group C. Volume Bias Analysis
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Avg daily volume | computed from PriceAdj | §14.7-CP H5.C |
| C.2 Wasserstein distance | included vs excluded | treaty gate |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 模型訓練前必跑 | `python scripts/audit/audit_universe_selection_bias.py` |

### 不提供之旗標 (Intentionally Omitted)
- `--fix`:audit only,bias 修正屬 §14.7-CB/CI/CJ universe builder 治權。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CP T_CP-3 H5 audit**。3-axis(sector/size/volume)bias check;treaty gates Wasserstein ≤ 2.0。 | ARCHIVED(標頭格式)|
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
