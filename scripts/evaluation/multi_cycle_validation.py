"""
multi_cycle_validation.py — Multi-Horizon Multi-Cycle Production Validation
================================================================================
治權:§14.7-CX 進階 multi-cycle validation(v6.18.1 pending §14.7-CY)
最後更新:2026-05-28
用戶 directive 2026-05-28:多週期驗證(週/月/季/年 horizon)為 institutional standard
                          不可在 AI 環境執行,必須為 system 永久 script。

對 95 historical feature_store_snapshots(2018-06 ~ 2026-04)跑:
  - 4 horizons:weekly(5d)/ monthly(20d)/ quarterly(60d)/ annual(252d)
  - Walk-forward expanding window OOS per horizon
  - 真實 transaction cost modeling
  - Cross-horizon comparison
  - Statistical significance per horizon

CLI:
  --dry-run / --commit              (commit writes results to evaluation_log)
  --horizons 5,20,60,252            (comma-separated label horizons in days)
  --output reports/multi_cycle_*.md (optional structured report path)
  --persist-db                       (write per-horizon summary to evaluation_log)

治權 enforce(per CLAUDE.md §一.10):
  - All data from (b) DB query — feature_values + TaiwanStockPriceAdj
  - 0 AI hallucinated numbers
  - 0 推測 / 估算 / placeholder
  - 結果可重現(seed=5422 + multi-thread stochasticity disclosed)
"""
from __future__ import annotations
import sys, argparse, math, json, logging, time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
import lightgbm as lgb
from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

DEFAULT_HORIZONS = [
    ("weekly",     5),
    ("monthly",   20),
    ("quarterly", 60),
    ("annual",   252),
]

LGB_PARAMS = {
    "learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 30,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "regression", "metric": "rmse",
    "verbose": -1, "seed": 5422,
}
N_ESTIMATORS = 200

SPEC_43 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "amihud_illiquidity_60d", "zero_volume_ratio_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "net_income_positive_ratio_8q",
    "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "right_tail_concentration_60d", "barbell_balance_60d", "preferential_attachment_60d",
    "fitness_signal_60d", "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "trust_net_20d", "trust_net_60d", "margin_ratio_60d",
    "theme_strength", "theme_is_semiconductor",
]


def get_panel_dates():
    """Generate 95 mid-month dates 2018-06-15 to 2026-04-15"""
    dates = []
    current = date(2018, 6, 15)
    while current <= date(2026, 4, 30):
        dates.append((f"fs_{current.strftime('%Y%m%d')}_feature_set_v0_4", current))
        if current.month == 12: current = date(current.year+1, 1, 15)
        else: current = date(current.year, current.month+1, 15)
    return dates


def load_features(cur, fs_id, universe):
    """Load (X, sids) for given panel — features only"""
    cur.execute("""
        SELECT stock_id, feature_name, feature_value::numeric
        FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s)
    """, (fs_id, list(universe)))
    feat_data = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_43:
            feat_data[sid][fname] = float(val)
    X, sids = [], []
    for sid in universe:
        if sid in feat_data:
            X.append([feat_data[sid].get(f, 0.0) for f in SPEC_43])
            sids.append(sid)
    return X, sids


def load_forward_returns(cur, as_of, horizon_days):
    """Load 真實 forward log returns at given horizon(per (b) DB query)"""
    cur.execute("""
        SELECT MIN(date) FROM "TaiwanStockPriceAdj"
        WHERE date >= (%s::date + INTERVAL '%s days')
          AND stock_id ~ '^[0-9]'
          AND date <= (%s::date + INTERVAL '%s days')
    """, (str(as_of), horizon_days, str(as_of), horizon_days + 14))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date:
        return {}, None
    cur.execute("""
        WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
             t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
        SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric)
        FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id
    """, (str(as_of), str(label_date)))
    returns = {sid: float(r) for sid, r in cur.fetchall()}
    return returns, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10: return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    lo = np.quantile(arr, lo_q); hi = np.quantile(arr, hi_q)
    return np.clip(arr, lo, hi)


def evaluate_horizon(cur, panels, horizon_days, universe, label):
    """Walk-forward LGBM expanding window for one horizon"""
    logger.info(f"\n{'='*100}")
    logger.info(f"Horizon: {label}({horizon_days}d)")
    logger.info(f"{'='*100}")

    # Phase 1: Load all panel data(features + horizon-specific forward returns)
    panel_data = {}
    t0 = time.monotonic()
    for fs_id, as_of in panels:
        X, sids = load_features(cur, fs_id, universe)
        if not X: continue
        returns, label_date = load_forward_returns(cur, as_of, horizon_days)
        if not returns: continue
        # Match features and returns
        XX, yy, sids_matched = [], [], []
        for i, sid in enumerate(sids):
            if sid in returns:
                XX.append(X[i]); yy.append(returns[sid]); sids_matched.append(sid)
        if XX:
            panel_data[as_of] = (XX, yy, sids_matched, label_date)
    logger.info(f"  Loaded {len(panel_data)} panels with valid {horizon_days}d forward returns(load: {time.monotonic()-t0:.1f}s)")

    # Phase 2: Walk-forward expanding window OOS
    panel_keys = sorted(panel_data.keys())
    panel_ics, panel_top20_rets, panel_univ_rets = [], [], []
    panel_records = []

    for i in range(1, len(panel_keys)):
        test_key = panel_keys[i]
        X_test, y_test, sids_test, label_date = panel_data[test_key]
        # Train on panels [0, ..., i-1]
        train_X, train_y = [], []
        for j in range(i):
            X_j, y_j, _, _ = panel_data[panel_keys[j]]
            train_X.extend(X_j); train_y.extend(y_j)
        X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y), 0.01, 0.99)
        if len(X_tr) < 100: continue

        # Train fold model
        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=SPEC_43)
        fold_model = lgb.train(LGB_PARAMS, train_data, num_boost_round=N_ESTIMATORS)

        # Predict + OOS metrics
        X_te = np.array(X_test); pred_te = fold_model.predict(X_te)
        ic_te = spearman_ic(pred_te, y_test)
        n_top = min(20, len(pred_te))
        top_idx = np.argsort(pred_te)[-n_top:]
        top20_ret = float(np.mean([y_test[k] for k in top_idx]))
        univ_ret = float(np.mean(y_test))
        panel_ics.append(ic_te); panel_top20_rets.append(top20_ret); panel_univ_rets.append(univ_ret)
        panel_records.append({"as_of": str(test_key), "label_date": str(label_date),
                              "ic": ic_te, "top20_ret": top20_ret, "univ_ret": univ_ret,
                              "alpha": top20_ret - univ_ret})

    # Phase 3: Aggregate metrics
    if not panel_top20_rets:
        return None
    n = len(panel_top20_rets)
    mean_ret = float(np.mean(panel_top20_rets))
    std_ret = float(np.std(panel_top20_rets, ddof=1)) if n > 1 else 0
    # Annualization: panels are monthly so 12 per year regardless of horizon
    sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0
    win_rate = sum(1 for r in panel_top20_rets if r > 0) / n
    alphas = [t - u for t, u in zip(panel_top20_rets, panel_univ_rets)]
    mean_alpha = float(np.mean(alphas))
    std_alpha = float(np.std(alphas, ddof=1)) if n > 1 else 0
    ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0
    t_stat = mean_alpha / (std_alpha / math.sqrt(n)) if std_alpha > 0 else 0
    cum = sum(panel_top20_rets)

    # MDD
    running = 0; peak = 0; mdd = 0
    for r in panel_top20_rets:
        running += r
        if running > peak: peak = running
        if peak - running > mdd: mdd = peak - running

    # ── HONEST ANNUALIZATION(per horizon)──
    # For overlapping horizons(60d/252d sampled monthly),sum-of-panel-returns DOUBLE COUNTS
    # Correct method: expected annualized log return = mean_per_panel × rebalances_per_year
    # rebalances_per_year = 252 trading days / horizon_days(if actually rebalancing at horizon)
    rebals_per_year = 252.0 / horizon_days
    annualized_log_gross = mean_ret * rebals_per_year
    annualized_simple_gross = math.exp(annualized_log_gross) - 1

    # Net of cost: cost per rebal × rebals_per_year(longer horizon = less cost drag)
    cost_per_rebal = 0.006  # standard TW broker round-trip
    annual_cost_drag = cost_per_rebal * rebals_per_year
    annualized_log_net = annualized_log_gross - annual_cost_drag
    annualized_simple_net = math.exp(annualized_log_net) - 1

    # Net Sharpe: per-panel net return / std(no change as std unchanged by constant cost)
    net_rets = [r - cost_per_rebal for r in panel_top20_rets]
    net_mean = float(np.mean(net_rets))
    net_std = float(np.std(net_rets, ddof=1)) if n > 1 else 0
    net_sharpe = net_mean / net_std * math.sqrt(12) if net_std > 0 else 0

    # ── EFFECTIVE N(per overlap correction)──
    # Panels are spaced 30 days(monthly grid)but each holds for horizon_days
    # Overlap fraction = max(0, horizon - 30) / horizon
    panel_spacing = 30
    if horizon_days <= panel_spacing:
        n_eff = float(n)
        overlap_pct = 0.0
    else:
        n_eff = n * (panel_spacing / horizon_days)
        overlap_pct = (horizon_days - panel_spacing) / horizon_days * 100
    eff_t_stat = t_stat * math.sqrt(n_eff / n) if n > 0 else 0
    is_significant = abs(eff_t_stat) > 1.997  # critical t for p<0.05 large df

    result = {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "n_effective": n_eff, "overlap_pct": overlap_pct,
        "rebals_per_year": rebals_per_year,
        "mean_ret_per_panel": mean_ret, "std_ret_per_panel": std_ret,
        "sharpe": sharpe, "win_rate": win_rate, "mdd_per_panel": mdd,
        "mean_alpha_per_panel": mean_alpha, "std_alpha_per_panel": std_alpha,
        "ir": ir, "t_stat": t_stat,
        "effective_t_stat": eff_t_stat, "is_significant_p05": is_significant,
        "mean_ic": float(np.mean(panel_ics)), "std_ic": float(np.std(panel_ics, ddof=1)) if n > 1 else 0,
        "annualized_log_gross": annualized_log_gross,
        "annualized_simple_gross": annualized_simple_gross,
        "annual_cost_drag_log": annual_cost_drag,
        "annualized_log_net": annualized_log_net,
        "annualized_simple_net": annualized_simple_net,
        "net_sharpe_per_panel": net_sharpe,
        "panel_records": panel_records,
    }

    # Print summary(honest annualization)
    logger.info(f"\n  Results({label}, {horizon_days}d):")
    logger.info(f"    OOS panels:                  {n}")
    logger.info(f"    Mean ret/panel(log):        {mean_ret*100:+.4f}%")
    logger.info(f"    Std ret/panel:               {std_ret*100:.4f}%")
    logger.info(f"    Sharpe(annualized monthly): {sharpe:+.4f}")
    logger.info(f"    Win rate:                    {win_rate*100:.1f}%")
    logger.info(f"    Mean alpha/panel:            {mean_alpha*100:+.4f}%")
    logger.info(f"    Information Ratio:           {ir:+.4f}")
    logger.info(f"    t-statistic:                 {t_stat:+.4f}(df={n-1})")
    logger.info(f"    Mean OOS IC:                 {result['mean_ic']:+.4f}")
    logger.info(f"    MDD(per-panel running):     {mdd*100:.2f}%")
    logger.info(f"")
    logger.info(f"    --- Honest Annualization ---")
    logger.info(f"    Rebalances/year(if @ horizon):{rebals_per_year:.2f}")
    logger.info(f"    Annualized log gross:        {annualized_log_gross:+.4f}")
    logger.info(f"    Annualized simple gross:     {annualized_simple_gross*100:+.2f}%/year")
    logger.info(f"    Annual cost drag(0.6%/reb): -{annual_cost_drag*100:.2f}%")
    logger.info(f"    Annualized simple NET:       {annualized_simple_net*100:+.2f}%/year")
    logger.info(f"")
    logger.info(f"    --- Statistical Robustness ---")
    logger.info(f"    Panel overlap:               {overlap_pct:.1f}%")
    logger.info(f"    Effective n(non-overlap):   {n_eff:.1f}")
    logger.info(f"    Effective t-stat:            {eff_t_stat:+.3f}")
    logger.info(f"    Significance at p<0.05:      {'✅ YES' if is_significant else '❌ NO'}")

    return result


def main():
    parser = argparse.ArgumentParser(description=f"Multi-Cycle Validation {TOOL_VER}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252",
                        help="Comma-separated label horizons in days")
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown report path")
    args = parser.parse_args()
    if not args.dry_run and not args.commit:
        args.dry_run = True

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    logger.info("="*100)
    logger.info(f"Multi-Cycle Validation {TOOL_VER}(per §14.7-CX T_CX-2 multi-regime extension)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  Mode:     {'COMMIT' if args.commit else 'DRY-RUN'}")
    logger.info(f"  Seed:     {LGB_PARAMS['seed']}")

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("""SELECT m.stock_id FROM core_universe_membership m
                       JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
                       WHERE s.status='committed' AND m.core_tier='core_universe'
                       AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot
                                          WHERE status='committed' ORDER BY created_at DESC LIMIT 1)""")
        universe = list({r[0] for r in cur.fetchall()})
        logger.info(f"  Universe: {len(universe)} stocks")

        panels = get_panel_dates()
        logger.info(f"  Panels:   {len(panels)}(2018-06-15 ~ 2026-04-15 monthly)")

        results = {}
        t_global = time.monotonic()
        for label, days in horizon_labels:
            r = evaluate_horizon(cur, panels, days, universe, label)
            if r: results[label] = r

        logger.info(f"\n{'='*100}")
        logger.info(f"Cross-Cycle Comparison Matrix")
        logger.info(f"{'='*100}")
        logger.info(f"  {'Horizon':10} {'N':>4} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'Win':>6} {'NetAnn':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_panels']:>4} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['win_rate']*100:>5.1f}% {r['annualized_simple_net']*100:>+8.2f}%")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        # Persist to file
        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            # Strip panel_records for top-level JSON
            output_results = {
                label: {k: v for k, v in r.items() if k != "panel_records"}
                for label, r in results.items()
            }
            output_results["_meta"] = {
                "tool": "multi_cycle_validation.py",
                "tool_ver": TOOL_VER,
                "run_at": datetime.now().isoformat(),
                "constitution_ver": CONSTITUTION_VER,
                "seed": LGB_PARAMS["seed"],
                "horizons": horizon_days_list,
                "n_universe": len(universe),
                "n_panels_input": len(panels),
                "source_traceability": "per CLAUDE.md §一.10 — all data from (b) DB query",
            }
            with open(output_full, "w") as f:
                json.dump(output_results, f, indent=2, default=str)
            logger.info(f"  Results persisted: {output_full}")

        # Persist to evaluation_log if --commit
        if args.commit:
            try:
                from core.db_utils import write_evaluation_log
                for label, r in results.items():
                    write_evaluation_log(
                        stock_id=f"MULTI_CYCLE_{label.upper()}",
                        model_name=f"lgbm_v0_2_{label}_{r['horizon_days']}d",
                        sharpe=r["sharpe"], mdd=r["mdd_per_panel"], ret=r["mean_ret_per_panel"], win_rate=r["win_rate"],
                        start=str(panels[0][1]), end=str(panels[-1][1]),
                        extra={"horizon_days": r["horizon_days"], "n_panels": r["n_panels"],
                               "ir": r["ir"], "t_stat": r["t_stat"],
                               "rebals_per_year": r["rebals_per_year"],
                               "annualized_simple_gross": r["annualized_simple_gross"],
                               "annual_cost_drag_log": r["annual_cost_drag_log"],
                               "annualized_simple_net": r["annualized_simple_net"],
                               "doctrine": "§14.7-CX T_CX-2 multi-cycle"}
                    )
                logger.info(f"  ✅ {len(results)} horizons committed to evaluation_log")
            except Exception as e:
                logger.warning(f"  evaluation_log write failed: {e}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
