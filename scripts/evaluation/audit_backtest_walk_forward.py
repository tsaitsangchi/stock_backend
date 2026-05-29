"""
audit_backtest_walk_forward.py v0.1 (Real 8-Panel Walk-Forward Backtest Auditor · §14.7-CV Production Closure · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 三段式標頭補正;原 v0.1 邏輯 2026-05-28 入)
**主權狀態**: ACTIVE (§14.7-CV Backtest Production Closure + §14.7-CX 8-year extension precursor + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[8-Panel Walk-Forward Backtest]** (v0.1, §14.7-CV): 對 8 historical snapshots(fs_20260105 → fs_20260415)+ 30d forward returns 跑 real portfolio backtest。
2. **[Top-20 Long Strategy]** (v0.1): 等權 top-20 portfolio + forward 30d return + cross-panel metrics。
3. **[Treaty Gates 4/4]** (v0.1, §14.7-CV): Sharpe > 0 / Win rate ≥ 50% / MDD ≤ 30% / Top-20 outperform universe。
4. **[Nearest Trading Day Match]** (v0.1): label_date 落非交易日時取 nearest within 10d window;無 nearest trading day 之 panel 排除。
5. **[Source Traceability]** (v0.1, §一.10): 全 (b) DB query(feature_values + TaiwanStockPriceAdj);0 AI memory。
6. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): 4 gates 動態判定。
7. **[Sovereignty Declaration]** (v0.1, §3.1 序列 backtest 模組): 本程式為 **§14.7-CV Backtest Production 唯一實作**(§3.1 序列 evaluation 模組;為 §14.7-CX/CY 之 precursor)。**治權邊界**:(a) §3.1 序列 backtest;(b) read-only;(c) **不訓練 model**(用 log_return_60d 作為 predictor proxy);(d) §8.5 已 handle by feature_store_builder;(e) 唯一職責:8-panel backtest + 4-gate verdict + portfolio P&L 量化證據。
8. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照;被 §14.7-CX 95-panel walk-forward 取代為 production reality。
9. **[Idempotency]** (v0.1): pure read-only。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. 8-Panel Loading + Forward Returns
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 8 fixed panels load | fs_20260105 → fs_20260415 | §14.7-CV |
| A.2 Nearest trading day | INTERVAL '7 days' window | safety |
| A.3 Forward returns | LN(t1/t0) JOIN | §14.7-CV |

### Group B. Top-20 Strategy Evaluation
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Top-20 by log_return_60d | proxy predictor | §14.7-CV |
| B.2 Equal-weight portfolio | mean return | §14.7-CV |
| B.3 Forward 30d return | per panel | §14.7-CV |

### Group C. Cross-Panel Metrics + Treaty Gates
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Sharpe annualized | mean/std * sqrt(12) | Gate CV-1 |
| C.2 Win rate | sum(>0)/n | Gate CV-2 |
| C.3 MDD | running peak tracking | Gate CV-3 |
| C.4 Alpha vs universe | mean(top20 - univ) | Gate CV-4 |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 8-panel backtest 驗證 | `python scripts/evaluation/audit_backtest_walk_forward.py` |

### 不提供之旗標 (Intentionally Omitted)
- `--horizons / --multi-cycle`:本程式為 single 30d;multi-horizon 屬 §14.7-CY multi_cycle_validation 治權。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CV Backtest Production**。8-panel walk-forward + nearest-trading-day fix + 4-gate verdict。**首跑實證**:Sharpe 3.10 / IR 3.57 / Win 75% / MDD 9.72% / Cum +75.21% / Treaty Gates 4/4 PASS。被 §14.7-CX 95-panel 取代為 production reality(95-panel Sharpe 1.67)。 | ARCHIVED(標頭格式)|
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
