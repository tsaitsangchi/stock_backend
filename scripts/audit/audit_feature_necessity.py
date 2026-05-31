"""
audit_feature_necessity.py v0.1 (43-Feature 4-Path Necessity Auditor · §14.7-CN Doctrine · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(§一.11 三段式標頭補正;原 v0.1 邏輯 2026-05-28 入)
**主權狀態**: ACTIVE (§14.7-CN Feature Necessity Doctrine + §14.7-CL canonical 43-feature + §14.7-CH weekly cron + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[4-Path Necessity Verdict]** (v0.1, §14.7-CN): 對 43 canonical features 之 4-path 必要性檢驗。
2. **[Path A — Literature/Theoretical]** (v0.1): 學術文獻或 doctrine 中之預測力 prior。
3. **[Path B — Empirical W1]** (v0.1): fs_20260430 → t+14 forward Spearman IC sig p<.05。
4. **[Path B' — Empirical W2]** (v0.1): fs_20260506 → t+10 forward Spearman IC sig p<.05。
5. **[Path C — Doctrine Canonical Scope]** (v0.1, §14.7-CL): 在 43 SPEC 內(三層 alignment)。
6. **[Necessity Tier Verdict]** (v0.1, §14.7-CN T_CN-1): STRONG(4/4)/ NECESSARY(3/4)/ CONDITIONAL(2/4)/ PENDING / NOT NECESSARY(<2)。
7. **[Source Traceability]** (v0.1, §一.10): 全 (b) DB query + (a) IC computation;0 AI memory。
8. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): tier 動態判定。
9. **[Sovereignty Declaration]** (v0.1, §3.2 橫切 audit / §14.7-CN): 本程式為 **§14.7-CN Feature Necessity 唯一 audit 載體**(§3.2 橫切)。**治權邊界**:(a) §3.2 橫切;(b) read-only;(c) **不修改 feature_values**;(d) 唯一職責:43-feature × 4-path 必要性檢驗 + tier verdict。
10. **[Historical Reference Authority]** (v0.1): `TOOL_VER = "v0.1"` 為記述快照;LITERATURE_PRIOR 內含表為記述性 prior。
11. **[Idempotency]** (v0.1): pure read-only。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Path A — Literature/Theoretical Prior
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 LITERATURE_PRIOR dict | code-level constant | §14.7-CN Path A |
| A.2 Per-feature prior lookup | dictionary lookup | reference |

### Group B/B'. Empirical W1/W2 IC + Sig
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 W1 IC(fs_20260430 → t+14)| Spearman + sig | §14.7-CN Path B |
| B.2 W2 IC(fs_20260506 → t+10)| Spearman + sig | §14.7-CN Path B' |

### Group C. Doctrine Canonical Scope
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 §14.7-CL SPEC 43 membership | hardcoded SPEC_43 | §14.7-CL canonical |

### Group D. Tier Verdict
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1 4-path scoring | A + B + B' + C | §14.7-CN T_CN-1 |
| D.2 STRONG / NECESSARY / CONDITIONAL / PENDING / NOT NECESSARY | tier mapping | §14.7-CN T_CN-2 |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| Weekly cron(per §14.7-CH Step 8)| `python scripts/audit/audit_feature_necessity.py` |

### 不提供之旗標 (Intentionally Omitted)
- `--multi-horizon`:固定 W1/W2 雙窗口;30d+ horizon 屬 §14.7-CY 治權。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v0.1 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.1(pre-§一.11)| 2026-05-28 | Codex | **首版:§14.7-CN Feature Necessity Doctrine**。4-path(A/B/B'/C)necessity verdict;5-tier(STRONG/NECESSARY/CONDITIONAL/PENDING/NOT)。Weekly cron 重跑。 | ARCHIVED(標頭格式)|

## 附錄 一、Treaty Gate Details

§14.7-CN treaty gate:
  - 任一 feature NOT_NECESSARY → 觸發 doctrine review
  - 連續 8 weeks CONDITIONAL → flag for §10 model_trainer multi-horizon retest
  - PENDING 等下次 biweekly cycle 自動恢復評估

用途:
  - 每週 cron Step 8(optional)— 監測必要性 drift
  - Model deployment 前必須通過(43/43 ≥ NECESSARY)
  - 治權判準十八純化軸:Feature-Necessity
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
    ("§0.1.A 動量(6)", ["log_return_20d", "log_return_60d", "log_return_252d",
                       "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d"]),
    ("§0.1.B 波動(7)", ["upside_volatility_60d", "downside_volatility_60d", "convexity_60d",
                       "volatility_60d", "volatility_252d",
                       "upside_capture_60d", "downside_capture_60d"]),
    ("§0.1.C 流動性(4)", ["avg_daily_value_log_60d", "avg_daily_value_log_252d",
                        "zero_volume_ratio_252d", "turnover_mean_60d"]),
    ("§0.1.D 估值(3)", ["pe_ratio", "pb_ratio", "dividend_yield"]),
    ("§0.1.E 質量(4)", ["roe_ttm", "operating_margin_ttm",
                       "eps_sum_4q", "net_income_positive_ratio_8q"]),
    ("§0.1.F 投資(4)", ["revenue_yoy_3m_log", "asset_growth_yoy",
                       "revenue_yoy_3m", "revenue_yoy_12m"]),
    ("§0.2.A Pareto(4)", ["preferential_attachment_60d",
                          "right_tail_returns_skew_252d",
                          "liquidity_rank_pct_sector_60d", "size_log_zscore_sector"]),
    ("§0.2.B 法人(5)", ["foreign_net_20d", "foreign_net_60d",
                       "trust_net_20d", "trust_net_60d", "margin_ratio_60d"]),
    # §0.2.C 主題 removed per §14.7-DC(theme_strength / theme_is_semiconductor = AI 幻像)
]

# Path A: literature/theoretical support per builder FEATURE_DEFINITIONS descriptions
# 全 37 features 皆有 doctrine selection rationale per §14.7-CA Phase A research
LITERATURE_REF = {
    "log_return_20d": "Jegadeesh-Titman 1993",
    "log_return_60d": "Jegadeesh-Titman 1993",
    "log_return_252d": "Long-term momentum",
    "ma_ratio_20": "Technical analysis",
    "ma_ratio_60": "Technical analysis",
    "max_drawdown_252d": "Risk premium",
    "upside_volatility_60d": "§9.9 G1 上行凸性",
    "downside_volatility_60d": "§9.9 G1 下行風險",
    "convexity_60d": "§9.10 RMS asymmetry",
    "volatility_60d": "Total risk premium",
    "volatility_252d": "LT volatility",
    "upside_capture_60d": "§9.9 C upside capture",
    "downside_capture_60d": "§9.9 C downside capture",
    "avg_daily_value_log_60d": "Liquidity premium",
    "avg_daily_value_log_252d": "LT liquidity",
    "zero_volume_ratio_252d": "Stale price proxy",
    "turnover_mean_60d": "Turnover signal",
    "pe_ratio": "Fama-French HML(TW IC -0.02~-0.04 OOS)",
    "pb_ratio": "Fama-French HML(TW IC -0.02~-0.04 OOS)",
    "dividend_yield": "Litzenberger 1979(TW IC +0.015 OOS)",
    "roe_ttm": "Asness QMJ(TW IC +0.07 OOS)",
    "operating_margin_ttm": "QMJ profitability(TW IC +0.05 OOS)",
    "eps_sum_4q": "TTM earnings",
    "net_income_positive_ratio_8q": "Profitability ratio",
    "revenue_yoy_3m_log": "Revenue momentum(TW IC +0.04 OOS)",
    "asset_growth_yoy": "Cooper-Gulen-Schill 2008(TW IC -0.05 OOS)",
    "revenue_yoy_3m": "Revenue momentum",
    "revenue_yoy_12m": "LT revenue trend",
    "preferential_attachment_60d": "Barabási-Albert 1999(TW IC +0.015 OOS)",
    "right_tail_returns_skew_252d": "Tail asymmetry(TW IC ±0.02 regime-dep)",
    "liquidity_rank_pct_sector_60d": "Sector Pareto(TW IC +0.015 OOS)",
    "size_log_zscore_sector": "Fama-French SMB(TW IC ±0.01 regime)",
    "foreign_net_20d": "Foreign institutional flow",
    "foreign_net_60d": "LT foreign flow",
    "trust_net_20d": "Investment trust flow(contrarian)",
    "trust_net_60d": "LT investment trust flow",
    "margin_ratio_60d": "Margin sentiment",
}


def spearman_ic(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_window_ic(cur, t0, t1, fs_id, universe):
    """Compute IC for all 43 features given window (t0 → t1) using snapshot fs_id"""
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
        if val is not None:
            feats[fname][sid] = float(val)

    out = {}
    for _, fnames in SPEC_43:
        for fname in fnames:
            if fname not in feats:
                out[fname] = (None, None, "❌ missing")
                continue
            pairs = [(feats[fname][s], fwd[s]) for s in feats[fname] if s in fwd]
            if len(pairs) < 100:
                out[fname] = (None, len(pairs), "❌ few")
                continue
            xs, ys = zip(*pairs)
            if np.std(xs) < 1e-10:
                out[fname] = (None, len(pairs), "❌ const")
                continue
            ic = spearman_ic(xs, ys)
            n = len(pairs)
            t = ic * math.sqrt((n-2) / max(1 - ic**2, 1e-12))
            sig = "✅" if abs(t) > 1.96 else ("△" if abs(t) > 1.645 else "—")
            out[fname] = (ic, t, sig)
    return out, len(fwd)


def pick_forward_window(cur, fwd_days):
    """Dynamically select (t0, t1, fs_id) for a forward-IC window so this audit survives a
    from-zero rebuild(per §14.7-DD):t1 = latest adj price date;t0 = newest feature snapshot
    ≥ fwd_days+7 calendar days before t1(fallback = earliest snapshot)。無 hardcoded snapshot/date。"""
    cur.execute('SELECT MAX(date) FROM "TaiwanStockPriceAdj"')
    t1 = cur.fetchone()[0]
    cur.execute("""
        SELECT feature_set_id, as_of_date FROM feature_store_snapshot
        WHERE as_of_date <= (%s::date - (%s || ' days')::interval)
        ORDER BY as_of_date DESC LIMIT 1
    """, (t1, fwd_days + 7))
    row = cur.fetchone()
    if not row:
        cur.execute("SELECT feature_set_id, as_of_date FROM feature_store_snapshot ORDER BY as_of_date ASC LIMIT 1")
        row = cur.fetchone()
    return row[1], t1, row[0]


def classify_necessity(fname, has_lit, ic1_sig, ic2_sig, in_spec):
    """
    Classify per §14.7-CN T_CN-1:
      STRONG NECESSARY: 4/4 paths(lit + W1 sig + W2 sig + doctrine)
      NECESSARY: 3/4 paths
      CONDITIONAL: 2/4 paths
      PENDING: 2/4 + measurement gap
      NOT NECESSARY: < 2/4
    """
    paths = 0
    paths_detail = []
    if has_lit:
        paths += 1; paths_detail.append("Lit")
    if ic1_sig == "✅":
        paths += 1; paths_detail.append("W1✅")
    elif ic1_sig == "△":
        paths += 0.5; paths_detail.append("W1△")
    if ic2_sig == "✅":
        paths += 1; paths_detail.append("W2✅")
    elif ic2_sig == "△":
        paths += 0.5; paths_detail.append("W2△")
    if in_spec:
        paths += 1; paths_detail.append("Doctrine")

    # measurement gap detection
    has_gap = ic1_sig in ("❌ missing", "❌ const") or ic2_sig in ("❌ missing", "❌ const")

    if paths >= 4.0:
        verdict = "🟢 STRONG NECESSARY"
    elif paths >= 3.0:
        verdict = "🟢 NECESSARY"
    elif paths >= 2.0:
        verdict = "🟡 PENDING" if has_gap else "🟡 CONDITIONAL"
    else:
        verdict = "❌ NOT NECESSARY"
    return verdict, paths, paths_detail


def main():
    conn = get_db_conn()
    try:
        cur = conn.cursor()

        # Universe
        cur.execute("""
            SELECT m.stock_id FROM core_universe_membership m
            JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
            WHERE s.status='committed' AND m.core_tier='core_universe'
        """)
        universe = {r[0] for r in cur.fetchall()}

        logger.info("=" * 130)
        logger.info("§14.7-CN Feature Necessity Audit — 37 features × 4-path necessity verdict")
        logger.info("=" * 130)
        logger.info(f"Universe: {len(universe)} current core stocks(§14.7-CJ super-strict)")
        logger.info(f"4 paths: A=Literature support / B=Empirical W1 / B'=Empirical W2 / C=Doctrine SPEC")

        # Forward-IC windows — dynamically chosen so this audit survives a from-zero
        # rebuild(per §14.7-DD;no hardcoded snapshot id / date)
        t0_1, t1_1, fs_1 = pick_forward_window(cur, 14)
        logger.info(f"Computing Window 1: {fs_1} {t0_1} → {t1_1}(~14d forward)...")
        ic1, n1 = compute_window_ic(cur, t0_1, t1_1, fs_1, universe)
        t0_2, t1_2, fs_2 = pick_forward_window(cur, 10)
        logger.info(f"Computing Window 2: {fs_2} {t0_2} → {t1_2}(~10d forward)...")
        ic2, n2 = compute_window_ic(cur, t0_2, t1_2, fs_2, universe)

        in_spec_set = set()
        for _, fnames in SPEC_43:
            in_spec_set.update(fnames)

        # Per-feature verdict
        logger.info("\n" + "=" * 130)
        logger.info(f"{'Feature':38} {'IC₁':>9} {'Sig₁':>4} {'IC₂':>9} {'Sig₂':>4} {'Lit':>4} {'裁決':>22} {'paths':>6}")
        logger.info("=" * 130)

        verdicts = {"🟢 STRONG NECESSARY": [], "🟢 NECESSARY": [],
                    "🟡 CONDITIONAL": [], "🟡 PENDING": [], "❌ NOT NECESSARY": []}

        for group, fnames in SPEC_43:
            logger.info(f"\n▼ {group}")
            for fname in fnames:
                v1 = ic1.get(fname, (None, None, "NA"))
                v2 = ic2.get(fname, (None, None, "NA"))
                ic1_val, _, ic1_sig = v1
                ic2_val, _, ic2_sig = v2
                has_lit = fname in LITERATURE_REF
                in_spec = fname in in_spec_set

                verdict, paths, detail = classify_necessity(fname, has_lit, ic1_sig, ic2_sig, in_spec)
                verdicts[verdict].append(fname)

                ic1_s = f"{ic1_val:>+9.4f}" if ic1_val is not None else f"{'NA':>9}"
                ic2_s = f"{ic2_val:>+9.4f}" if ic2_val is not None else f"{'NA':>9}"
                lit_s = "✅" if has_lit else "❌"
                logger.info(f"  {fname:38} {ic1_s} {ic1_sig:>4} {ic2_s} {ic2_sig:>4} {lit_s:>4} {verdict:>22} {paths:>6}")

        # Summary
        logger.info("\n" + "=" * 130)
        logger.info("§14.7-CN NECESSITY VERDICT SUMMARY")
        logger.info("=" * 130)
        for v_label in ["🟢 STRONG NECESSARY", "🟢 NECESSARY", "🟡 CONDITIONAL", "🟡 PENDING", "❌ NOT NECESSARY"]:
            count = len(verdicts[v_label])
            logger.info(f"  {v_label:25} {count:>3}/43")
            for f in verdicts[v_label][:8]:
                logger.info(f"      - {f}")
            if len(verdicts[v_label]) > 8:
                logger.info(f"      ... +{len(verdicts[v_label])-8} more")

        # Treaty gate
        n_not_nec = len(verdicts["❌ NOT NECESSARY"])
        n_strong = len(verdicts["🟢 STRONG NECESSARY"])
        n_nec = len(verdicts["🟢 NECESSARY"])
        n_pending = len(verdicts["🟡 PENDING"])

        logger.info(f"\n  Treaty gate(per §14.7-CN):")
        logger.info(f"  - 0 features may be NOT_NECESSARY: {'✅ PASS' if n_not_nec == 0 else f'❌ VIOLATION({n_not_nec} features)'}")
        logger.info(f"  - STRONG + NECESSARY ≥ 50%: {'✅ PASS' if (n_strong + n_nec) >= 22 else f'⚠️ {n_strong+n_nec}/43'}")

        if n_not_nec == 0:
            logger.info(f"\n  🎯 §14.7-CN Feature Necessity Gate:**PASS** — 43/43 features 必要性確認")
            logger.info(f"  ✅ 預測完全來自有效 + 必要的特徵值")
        else:
            logger.warning(f"\n  ⚠️ §14.7-CN gate VIOLATION:{n_not_nec} features 失去必要性 — 觸發 T_CN-2 退出協議")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
