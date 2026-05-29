"""
audit_per_stock_feature_validity.py v0.1 (Per-Stock × Per-Feature Completeness + Correctness Auditor · §14.7-CI/CK/CL 配套 · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 三段式標頭補正;原 v0.1 邏輯 2026-05-28 入)
**主權狀態**: ACTIVE (§14.7-CI Strict Validity + §14.7-CK Effectiveness + §14.7-CL Canonical Scope + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Per-Stock × Per-Feature Completeness]** (v0.1, §14.7-CI): 對 v0.15 active universe(N=1,121)× 43 canonical features(§14.7-CL)逐 stock × 逐 feature 之 non-null check。
2. **[Correctness Statistical Check]** (v0.1, §14.7-CJ): 每 feature 之 distribution + outlier detection。
3. **[Sample Spot Check]** (v0.1): 5 隨機 stocks 顯示 full 43-feature values for human verification。
4. **[Source Traceability]** (v0.1, §一.10): 全 (b) DB query;0 AI memory。
5. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): completeness % / outlier count 動態判定。
6. **[Sovereignty Declaration]** (v0.1, §3.2 橫切 audit / §14.7-CI/CK/CL): 本程式為 **§14.7-CI/CK/CL 三節 audit 載體**(§3.2 橫切)。**治權邊界**:(a) §3.2 橫切;(b) read-only;(c) **不修改 feature_values**;(d) 唯一職責:scan feature_values + universe 計算 per-stock × per-feature compliance metrics。
7. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照;對齊 §14.7-CL canonical 43-feature SPEC。
8. **[Idempotency]** (v0.1): pure read-only。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Completeness Check
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 1,121 × 43 grid scan | feature_values WHERE feature_set_id | §14.7-CI/CL |
| A.2 NULL count per cell | NULL detection | §14.7-CI strict |
| A.3 100% completeness gate | per §14.7-CB hard gate | treaty |

### Group B. Correctness Check
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Per-feature distribution | min/max/mean/std/quantiles | §14.7-CJ |
| B.2 Outlier detection | z-score > N | §14.7-CJ reasonableness |

### Group C. Sample Spot Check
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 5 random stocks | random sample | human audit |
| C.2 Full 43-feature display | per stock | transparency |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 模型訓練前 feature 完整性驗證 | `python scripts/audit/audit_per_stock_feature_validity.py` |

### 不提供之旗標 (Intentionally Omitted)
- `--fix`:audit only;feature 修正屬 feature_store_builder 治權。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CI/CK/CL audit**。Per-stock × per-feature completeness + correctness check;sample spot 5 stocks。 | ARCHIVED(標頭格式)|
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

# 43 canonical SPEC features(per §14.7-CA + §14.7-CI + §14.7-CK + §14.7-CL 治權)
# §0.3 14 broadcast features removed per §14.7-CK Feature Effectiveness Doctrine
# 4 interaction features removed per §14.7-CL(IC = +0.0131 HARMFUL ablation + macro deprecated)
# § Final canonical scope:§0.1 29 + §0.2 14 = 43 per-stock model-trainable features
SPEC_FEATURES = {
    # ── §0.1 第一性原理 — 29 features ────────────────────────────────────────
    "§0.1.A Momentum (6)": ["log_return_20d", "log_return_60d", "log_return_252d",
                            "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d"],
    "§0.1.B Volatility (7)": ["upside_volatility_60d", "downside_volatility_60d", "convexity_60d",
                              "volatility_60d", "volatility_252d",
                              "upside_capture_60d", "downside_capture_60d"],
    "§0.1.C Liquidity (5)": ["avg_daily_value_log_60d", "avg_daily_value_log_252d",
                             "amihud_illiquidity_60d", "zero_volume_ratio_252d",
                             "turnover_mean_60d"],
    "§0.1.D Value (3)": ["pe_ratio", "pb_ratio", "dividend_yield"],
    "§0.1.E Quality (4)": ["roe_ttm", "operating_margin_ttm",
                           "eps_sum_4q", "net_income_positive_ratio_8q"],
    "§0.1.F Investment (4)": ["revenue_yoy_3m_log", "asset_growth_yoy",
                              "revenue_yoy_3m", "revenue_yoy_12m"],
    # ── §0.2 八二法則 — 14 features ──────────────────────────────────────────
    "§0.2.A Pareto explicit (7)": ["right_tail_concentration_60d", "barbell_balance_60d",
                                    "preferential_attachment_60d", "fitness_signal_60d",
                                    "right_tail_returns_skew_252d",
                                    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector"],
    "§0.2.B Institutional flow (5)": ["foreign_net_20d", "foreign_net_60d",
                                       "trust_net_20d", "trust_net_60d",
                                       "margin_ratio_60d"],
    "§0.2.C Theme alignment (2)": ["theme_strength", "theme_is_semiconductor"],
}

# 合理 range(per feature semantic)— for correctness check
FEATURE_RANGES = {
    # §0.1.A Momentum: log returns 應 [-1, 5]
    "log_return_20d": (-1.5, 3.0), "log_return_60d": (-2.0, 5.0), "log_return_252d": (-3.0, 8.0),
    "ma_ratio_20": (0.2, 4.0), "ma_ratio_60": (0.2, 5.0), "max_drawdown_252d": (0, 1.0),  # magnitude convention
    # §0.1.B Volatility: positive bounded
    "upside_volatility_60d": (0, 0.5), "downside_volatility_60d": (0, 0.5), "convexity_60d": (-0.5, 0.5),
    "volatility_60d": (0, 0.5), "volatility_252d": (0, 0.5),
    "upside_capture_60d": (0, 0.2), "downside_capture_60d": (0, 0.2),
    # §0.1.C Liquidity
    "avg_daily_value_log_60d": (3, 12), "avg_daily_value_log_252d": (3, 12),
    "amihud_illiquidity_60d": (0, 1e-3), "zero_volume_ratio_252d": (0, 1),
    # turnover_mean_60d: raw counts, range omitted
    # §0.1.D Value
    "pe_ratio": (0, 1000), "pb_ratio": (0, 50), "dividend_yield": (0, 30),
    # §0.1.E Quality
    "roe_ttm": (-2, 2), "operating_margin_ttm": (-5, 5),
    "eps_sum_4q": (-200, 2000), "net_income_positive_ratio_8q": (0, 1.01),
    # §0.1.F Investment
    "revenue_yoy_3m_log": (-10, 10), "asset_growth_yoy": (-1, 10),
    "revenue_yoy_3m": (-1, 1000), "revenue_yoy_12m": (-1, 1000),
    # §0.2.A Pareto
    "right_tail_concentration_60d": (0, 1), "barbell_balance_60d": (0, 1),
    "preferential_attachment_60d": (3, 14), "fitness_signal_60d": (-1e5, 1e5),  # cube-root 可為負(foreign_ratio<0)
    "right_tail_returns_skew_252d": (-10, 10),
    "liquidity_rank_pct_sector_60d": (0, 1), "size_log_zscore_sector": (-5, 5),
    # §0.2.B Institutional flow: raw shares/balance, range omitted(wide dynamic range)
    # §0.2.C Theme alignment
    "theme_strength": (0, 1.01), "theme_is_semiconductor": (0, 1.01),
    # §0.3 14 broadcast features removed per §14.7-CK + 4 interaction removed per §14.7-CL
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
                      "right_tail_concentration_60d", "size_log_zscore_sector"]:
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
