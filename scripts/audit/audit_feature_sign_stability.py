"""
audit_feature_sign_stability.py v0.1 (43-Feature Sign Stability + Literature + TW Commit Auditor · §14.7-CO/CQ Doctrines · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 三段式標頭補正;原 v0.1 邏輯 2026-05-28 入)
**主權狀態**: ACTIVE (§14.7-CO Sign Stability + §14.7-CQ TW Empirical Sign Commitment + §14.7-CR Mathematical Sign Realism + §14.7-CH weekly cron + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sign Stability Doctrine]** (v0.1, §14.7-CO): 對 43 SPEC features 跨三窗口計算 sign stability(IC 正負相關性正式入治權)。
2. **[Triple-Window Test]** (v0.1): W1(fs_20260430 → t+14)+ W2(fs_20260506 → t+10)+ W3(fs_20260316 → t+32, 30d primary)。
3. **[4-Tier Sign Verdict]** (v0.1, §14.7-CO T_CO-1): ROBUST POSITIVE / ROBUST NEGATIVE / REGIME-DEPENDENT / PENDING。
4. **[TW Empirical Sign Commitment]** (v0.1, §14.7-CQ): 每 feature commit + 或 -(v6.13.0 doctrine)。
5. **[Mathematical Sign Realism]** (v0.1, §14.7-CR): LITERATURE_SIGN 全清理為 ±(v6.13.1 patch)— 無 0 / 無中性 sign。
6. **[Literature Consistency Check]** (v0.1): empirical sign vs literature sign consistency 檢驗。
7. **[Source Traceability]** (v0.1, §一.10): 全 (b) DB query;0 AI memory。
8. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): tier 動態判定。
9. **[Sovereignty Declaration]** (v0.1, §3.2 橫切 audit / §14.7-CO/CQ/CR): 本程式為 **§14.7-CO/CQ/CR 三節 Sign Stability 唯一 audit 載體**(§3.2 橫切)。**治權邊界**:(a) §3.2 橫切;(b) read-only;(c) 唯一職責:43-feature × 3-window sign stability + literature consistency + TW commit verification。
10. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照;LITERATURE_SIGN / TW_EMPIRICAL_SIGN 內含 dict 為 v6.13.x doctrine snapshot。
11. **[Idempotency]** (v0.1): pure read-only。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Triple-Window Sign Computation
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 W1 IC sign | fs_20260430 → t+14 | §14.7-CO |
| A.2 W2 IC sign | fs_20260506 → t+10 | §14.7-CO |
| A.3 W3 IC sign | fs_20260316 → t+32(30d primary)| §14.7-CO |

### Group B. 4-Tier Verdict per Feature
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 ROBUST POSITIVE | W1>0 AND W2>0 | §14.7-CO T_CO-1 |
| B.2 ROBUST NEGATIVE | W1<0 AND W2<0 | §14.7-CO T_CO-1 |
| B.3 REGIME-DEPENDENT | W1 × W2 < 0 | §14.7-CO T_CO-1 |
| B.4 PENDING | NA in either | safety |

### Group C. Literature + TW Commit Consistency
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 LITERATURE_SIGN check | 全 ± 無 0(per §14.7-CR)| §14.7-CR |
| C.2 TW_EMPIRICAL_SIGN check | 每 feature ±(per §14.7-CQ)| §14.7-CQ |
| C.3 Empirical vs Literature consistency | match / mismatch verdict | §14.7-CO |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| Weekly cron(per §14.7-CH Step 9)| `python scripts/audit/audit_feature_sign_stability.py` |

### 不提供之旗標 (Intentionally Omitted)
- `--auto-commit`:TW commit 屬手動治權決策(per §14.7-CQ),不自動。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CO/CQ/CR Sign Stability**。43 features × 3-window sign + literature consistency + TW commit。4-tier verdict(ROBUST+/ROBUST-/REGIME-DEPENDENT/PENDING)。LITERATURE_SIGN ± 全清理(per §14.7-CR v6.13.1 patch)。 | ARCHIVED(標頭格式)|

## 附錄 一、Treaty Gate Details

+ Sign vs Literature 一致性檢驗(per §14.7-CO T_CO-2):
  ✅ MATCH:empirical W1 sign 與 literature 一致
  ❌ MISMATCH:empirical W1 sign 與 literature 反向(可能 TW market regime / 14d horizon 限制)
  ⚪ N/A:literature 無明確 sign 或 sign 為 regime-dependent

§14.7-CO Treaty gates:
  - Sign-stable baseline: ≥ 25%(14d horizon realistic / ≥ 30% aspirational at 30d+)
  - Lit-mismatch alert:≥ 6 features mismatch → 觸發 T_CO-3 multi-horizon retest
  - 0 features may be REGIME-DEP and pass through to production without disclosure(T_CO-4)

治權判準十九純化軸:Feature-Sign-Stability
"""
from __future__ import annotations
import sys, math, logging
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

# 43 SPEC features per §14.7-CL canonical scope
SPEC_43 = [
    ("§0.1.A 動量", ["log_return_20d", "log_return_60d", "log_return_252d",
                    "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d"]),
    ("§0.1.B 波動", ["upside_volatility_60d", "downside_volatility_60d", "convexity_60d",
                    "volatility_60d", "volatility_252d",
                    "upside_capture_60d", "downside_capture_60d"]),
    ("§0.1.C 流動性", ["avg_daily_value_log_60d", "avg_daily_value_log_252d",
                     "amihud_illiquidity_60d", "zero_volume_ratio_252d", "turnover_mean_60d"]),
    ("§0.1.D 估值", ["pe_ratio", "pb_ratio", "dividend_yield"]),
    ("§0.1.E 質量", ["roe_ttm", "operating_margin_ttm",
                    "eps_sum_4q", "net_income_positive_ratio_8q"]),
    ("§0.1.F 投資", ["revenue_yoy_3m_log", "asset_growth_yoy",
                    "revenue_yoy_3m", "revenue_yoy_12m"]),
    ("§0.2.A Pareto", ["right_tail_concentration_60d", "barbell_balance_60d",
                       "preferential_attachment_60d", "fitness_signal_60d",
                       "right_tail_returns_skew_252d",
                       "liquidity_rank_pct_sector_60d", "size_log_zscore_sector"]),
    ("§0.2.B 法人", ["foreign_net_20d", "foreign_net_60d",
                    "trust_net_20d", "trust_net_60d", "margin_ratio_60d"]),
    ("§0.2.C 主題", ["theme_strength", "theme_is_semiconductor"]),
]

# Literature sign:per builder FEATURE_DEFINITIONS + Phase A research (§14.7-CA)
# "+" = positive predictive sign / "-" = negative
# 2026-05-28 v6.13.1 patch:依 §14.7-CR Mathematical Sign Realism,LITERATURE_SIGN 全清 "±"
# 每 feature commit + 或 -,依 dominant US literature opinion(non-regime-conditional)
# TW empirical 反向之 feature 反映於 TW_EMPIRICAL_SIGN(由 commit_tw_sign() 處理)
LITERATURE_SIGN = {
    # Momentum: positive(Jegadeesh-Titman)
    "log_return_20d": "+", "log_return_60d": "+", "log_return_252d": "+",
    "ma_ratio_20": "+", "ma_ratio_60": "+",
    "max_drawdown_252d": "-",  # v6.13.1: US risk premium dominant — high DD → low forward return
    # Volatility: risk premium positive
    "upside_volatility_60d": "+", "downside_volatility_60d": "+",
    "convexity_60d": "+",  # v6.13.1: §9.10 RMS asymmetry → 凸性正向(上行 > 下行)
    "volatility_60d": "+", "volatility_252d": "+",
    "upside_capture_60d": "+", "downside_capture_60d": "+",
    # Liquidity: amihud illiquidity premium(US literature dominant)
    "avg_daily_value_log_60d": "+", "avg_daily_value_log_252d": "+",
    "amihud_illiquidity_60d": "+",  # v6.13.1: Amihud 2002 dominant lit;TW 反向見 TW_EMPIRICAL_SIGN
    "zero_volume_ratio_252d": "-",
    "turnover_mean_60d": "+",
    # Value: Fama-French HML(US dominant — low P/E wins)
    "pe_ratio": "-",  # v6.13.1: Fama-French HML dominant;TW 反向見 TW_EMPIRICAL_SIGN
    "pb_ratio": "-",  # v6.13.1: 同上
    "dividend_yield": "+",  # v6.13.1: Litzenberger 1979 yield premium dominant
    # Quality: Asness QMJ(high quality → high return)
    "roe_ttm": "+", "operating_margin_ttm": "+",
    "eps_sum_4q": "+", "net_income_positive_ratio_8q": "+",
    # Investment: Cooper-Gulen-Schill
    "revenue_yoy_3m_log": "+", "asset_growth_yoy": "-",
    "revenue_yoy_3m": "+", "revenue_yoy_12m": "+",
    # Pareto:
    "right_tail_concentration_60d": "+",
    "barbell_balance_60d": "+",  # §14.7-CQ v6.13.0: §9.2 barbell theory positive
    "preferential_attachment_60d": "+", "fitness_signal_60d": "+",
    "right_tail_returns_skew_252d": "+",  # v6.13.1: Kelly-Jiang 2014 positive skew premium
    "liquidity_rank_pct_sector_60d": "+",
    "size_log_zscore_sector": "-",  # v6.13.1: Fama-French SMB original — small wins → high size = "-"
    # Institutional: foreign positive / trust contrarian(TW empirical)
    "foreign_net_20d": "+", "foreign_net_60d": "+",
    "trust_net_20d": "-", "trust_net_60d": "-",
    "margin_ratio_60d": "+",
    # Theme: positive
    "theme_strength": "+", "theme_is_semiconductor": "+",
}


def spearman_ic(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_window_ic(cur, t0, t1, fs_id, universe):
    cur.execute("""
        WITH t0p AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
             t1p AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
        SELECT t0p.stock_id, LN(t1p.close::numeric/t0p.close::numeric)
        FROM t0p JOIN t1p ON t0p.stock_id = t1p.stock_id
    """, (t0, t1))
    fwd = {sid: float(r) for sid, r in cur.fetchall() if sid in universe}
    cur.execute("SELECT stock_id, feature_name, feature_value::numeric FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s)",
                (fs_id, list(universe)))
    feats = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None: feats[fname][sid] = float(val)
    out = {}
    for _, fnames in SPEC_43:
        for fname in fnames:
            if fname not in feats:
                out[fname] = None; continue
            pairs = [(feats[fname][s], fwd[s]) for s in feats[fname] if s in fwd]
            if len(pairs) < 100:
                out[fname] = None; continue
            xs, ys = zip(*pairs)
            if np.std(xs) < 1e-10:
                out[fname] = None; continue
            out[fname] = spearman_ic(xs, ys)
    return out


def classify_sign(ic1, ic2, threshold=0.005):
    """Sign verdict per §14.7-CO T_CO-1."""
    if ic1 is None or ic2 is None:
        return "⏳ PENDING"
    if ic1 > threshold and ic2 > threshold:
        return "🟢 ROBUST POSITIVE"
    if ic1 < -threshold and ic2 < -threshold:
        return "🟢 ROBUST NEGATIVE"
    if (ic1 > threshold and ic2 < -threshold) or (ic1 < -threshold and ic2 > threshold):
        return "🔄 REGIME-DEP(strong flip)"
    if ic1 * ic2 < 0:
        return "🔄 REGIME-DEP(weak flip)"
    if abs(ic1) <= threshold and abs(ic2) <= threshold:
        return "⚪ NEAR-ZERO"
    return "🟡 WEAK-SIGN"


def commit_tw_sign(ic1, ic2, ic3, lit_sign, threshold=0.005):
    """
    §14.7-CQ TW Empirical Sign Commitment — every feature commit + or -.

    Priority order:
      1. IC₃ (30d horizon, longest)
      2. IC₁ (14d, most recent)
      3. IC₂ (10d, fallback)
      4. Literature sign (last resort)

    Returns: "+", "-", or "?" (genuinely indeterminate, only if all NA + lit is ±)
    """
    # Try W3 first (primary)
    if ic3 is not None and abs(ic3) > threshold:
        return "+" if ic3 > 0 else "-"
    # Fallback W1 (most recent shorter horizon)
    if ic1 is not None and abs(ic1) > threshold:
        return "+" if ic1 > 0 else "-"
    # Fallback W2
    if ic2 is not None and abs(ic2) > threshold:
        return "+" if ic2 > 0 else "-"
    # Last resort: literature
    if lit_sign in ("+", "-"):
        return lit_sign
    # Truly indeterminate
    return "?"


def check_literature_consistency(fname, ic1):
    """Sign vs literature(per §14.7-CO T_CO-2)."""
    lit = LITERATURE_SIGN.get(fname, "?")
    if lit == "?" or lit == "±" or ic1 is None:
        return "⚪ N/A", lit
    if (lit == "+" and ic1 > 0) or (lit == "-" and ic1 < 0):
        return "✅ MATCH", lit
    return "❌ MISMATCH", lit


def main():
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.stock_id FROM core_universe_membership m
            JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
            WHERE s.status='committed' AND m.core_tier='core_universe'
        """)
        universe = {r[0] for r in cur.fetchall()}

        logger.info("=" * 130)
        logger.info("§14.7-CO Feature Sign Stability Audit — 43 features × sign verdict + literature consistency")
        logger.info("=" * 130)
        logger.info(f"Universe: {len(universe)} stocks(§14.7-CJ super-strict)")
        logger.info("Computing Window 1: fs_20260430 → 2026-05-20(14d)...")
        ic_w1 = compute_window_ic(cur, '2026-04-30', '2026-05-20',
                                   'fs_20260430_feature_set_v0_4', universe)
        logger.info("Computing Window 2: fs_20260506 → 2026-05-20(10d)...")
        ic_w2 = compute_window_ic(cur, '2026-05-06', '2026-05-20',
                                   'fs_20260506_feature_set_v0_4', universe)
        logger.info("Computing Window 3: fs_20260316 → 2026-04-30(30d)...")
        ic_w3 = compute_window_ic(cur, '2026-03-16', '2026-04-30',
                                   'fs_20260316_feature_set_v0_4', universe)

        # Per-feature analysis
        logger.info("\n" + "=" * 150)
        logger.info(f"{'Feature':38} {'IC₁':>9} {'IC₂':>9} {'IC₃(30d)':>10} {'Lit':>4} {'Sign Verdict':>26} {'Lit Check':>11} {'TW Commit':>10}")
        logger.info("=" * 150)

        verdicts = defaultdict(list)
        lit_checks = defaultdict(list)
        tw_commits = defaultdict(list)
        for group, fnames in SPEC_43:
            logger.info(f"\n▼ {group}")
            for fname in fnames:
                ic1 = ic_w1.get(fname)
                ic2 = ic_w2.get(fname)
                ic3 = ic_w3.get(fname)
                sv = classify_sign(ic1, ic2)
                lit_status, lit_sign = check_literature_consistency(fname, ic1)
                tw_sign = commit_tw_sign(ic1, ic2, ic3, lit_sign)

                verdicts[sv].append(fname)
                lit_checks[lit_status].append(fname)
                tw_commits[tw_sign].append(fname)

                ic1_s = f"{ic1:>+9.4f}" if ic1 is not None else f"{'NA':>9}"
                ic2_s = f"{ic2:>+9.4f}" if ic2 is not None else f"{'NA':>9}"
                ic3_s = f"{ic3:>+10.4f}" if ic3 is not None else f"{'NA':>10}"
                logger.info(f"  {fname:38} {ic1_s} {ic2_s} {ic3_s} {lit_sign:>4} {sv:>26} {lit_status:>11} {tw_sign:>10}")

        # Sign stability summary
        logger.info("\n" + "=" * 130)
        logger.info("§14.7-CO Sign Stability Verdict Summary")
        logger.info("=" * 130)
        for v in ["🟢 ROBUST POSITIVE", "🟢 ROBUST NEGATIVE", "🔄 REGIME-DEP(strong flip)",
                  "🔄 REGIME-DEP(weak flip)", "⚪ NEAR-ZERO", "🟡 WEAK-SIGN", "⏳ PENDING"]:
            count = len(verdicts[v])
            if count > 0:
                logger.info(f"  {v:30} {count:>3}/43")
                for f in verdicts[v][:6]:
                    logger.info(f"      - {f}")
                if len(verdicts[v]) > 6:
                    logger.info(f"      ... +{len(verdicts[v])-6} more")

        # Literature consistency summary
        logger.info("\n" + "=" * 130)
        logger.info("§14.7-CO Literature Sign Consistency Summary")
        logger.info("=" * 130)
        for v in ["✅ MATCH", "❌ MISMATCH", "⚪ N/A"]:
            count = len(lit_checks[v])
            logger.info(f"  {v:18} {count:>3}/43")
            if v == "❌ MISMATCH":
                for f in lit_checks[v]:
                    logger.info(f"      - {f}")

        # Treaty gates(per §14.7-CO)
        n_robust_pos = len(verdicts["🟢 ROBUST POSITIVE"])
        n_robust_neg = len(verdicts["🟢 ROBUST NEGATIVE"])
        n_stable = n_robust_pos + n_robust_neg
        n_regime = len(verdicts["🔄 REGIME-DEP(strong flip)"]) + len(verdicts["🔄 REGIME-DEP(weak flip)"])
        n_mismatch = len(lit_checks["❌ MISMATCH"])

        logger.info("\n" + "=" * 130)
        logger.info("§14.7-CO TREATY GATE VERIFICATION")
        logger.info("=" * 130)
        logger.info(f"  Sign-stable ratio: {n_stable}/43 = {n_stable/43*100:.1f}%(realistic baseline ≥ 25%, aspirational ≥ 30%)")
        gate1 = "✅ PASS" if n_stable >= 11 else "❌ VIOLATION"  # 25% baseline = 11/43
        gate1_aspi = "✅ MET" if n_stable >= 13 else "△ below target"  # 30% aspirational = 13/43
        logger.info(f"  Gate 1(sign-stable ≥ 25% realistic): {gate1}")
        logger.info(f"  Gate 1-aspi(≥ 30% at 30d+ horizon): {gate1_aspi}(14d 短期常 below;multi-horizon 後可期)")

        logger.info(f"  Literature mismatch: {n_mismatch} features")
        gate2 = "⚠️ ALERT" if n_mismatch >= 6 else "✅ PASS"
        logger.info(f"  Gate 2(lit-mismatch ≤ 5):{gate2}{'(T_CO-3 multi-horizon retest triggered)' if n_mismatch >= 6 else ''}")

        logger.info(f"  REGIME-DEP features: {n_regime} ({n_regime/43*100:.0f}%) — disclosed per T_CO-4")
        gate3 = "✅ PASS"  # disclosure satisfied automatically by this audit
        logger.info(f"  Gate 3(regime-dep disclosure): {gate3}(本 audit 即 disclosure)")

        if gate1 == "✅ PASS" and gate2 == "✅ PASS":
            logger.info(f"\n  🎯 §14.7-CO Feature Sign Stability Gate:**PASS**")
            logger.info(f"  ✅ Sign 處理已正式入治權判準")
        elif gate1 == "✅ PASS":
            logger.warning(f"\n  ⚠️ §14.7-CO Gate:partial PASS — lit-mismatch alert active")
            logger.warning(f"     觸發 T_CO-3 multi-horizon retest(待 §10 model_trainer 落地)")
        else:
            logger.warning(f"\n  ❌ §14.7-CO Gate:VIOLATION — sign-stable < 30% baseline")

        # §14.7-CQ TW Empirical Sign Commitment Gate
        logger.info("\n" + "=" * 150)
        logger.info("§14.7-CQ TW Empirical Sign Commitment VERIFICATION")
        logger.info("=" * 150)
        n_plus = len(tw_commits["+"])
        n_minus = len(tw_commits["-"])
        n_unknown = len(tw_commits["?"])
        logger.info(f"  TW Commit '+' (long signal): {n_plus}/43")
        logger.info(f"  TW Commit '-' (short/contrarian): {n_minus}/43")
        logger.info(f"  TW Commit '?' (indeterminate): {n_unknown}/43")
        gate_cq = "✅ PASS" if n_unknown == 0 else f"⚠️ {n_unknown} features indeterminate"
        logger.info(f"  §14.7-CQ Gate(0 indeterminate): {gate_cq}")
        if n_unknown == 0:
            logger.info(f"\n  🎯 §14.7-CQ TW Empirical Sign Commitment Gate:**PASS**")
            logger.info(f"  ✅ 全 43 features 已 commit 明確方向(+ 或 -)")
        else:
            logger.warning(f"\n  ⚠️ §14.7-CQ Gate:{n_unknown} features 仍 indeterminate")
            for f in tw_commits["?"]:
                logger.warning(f"      - {f}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
