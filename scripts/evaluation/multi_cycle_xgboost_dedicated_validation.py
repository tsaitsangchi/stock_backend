"""
multi_cycle_xgboost_dedicated_validation.py v0.1 (Multi-Cycle XGBoost Validation + Precision/Reliability Analysis · §14.7-CY 第八實作 dedicated · per Canonical Comparison Framework · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29
**主權狀態**: MULTI-CYCLE 4-HORIZON XGBOOST VALIDATION + PRECISION/RELIABILITY + CANONICAL COMPARISON FRAMEWORK + §14.7-CY HORIZON-EXTENSION + §14.7-CX 8-YEAR OOS + §14.7-CW TREE-FAMILY 第八實作 dedicated + §一.10 SOURCE-TRACEABLE + §一.11 三段式合規
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Multi-Cycle Horizon Coverage]** (v0.1, §14.7-CY T_CY-2): 4 horizons(weekly 5d / monthly 20d / quarterly 60d / annual 252d)完全對齊其他 multi-cycle validators。
2. **[XGBoost Level-Wise GBT]** (v0.1): 200 trees / learning_rate=0.05 / max_depth=5 / min_child_weight=5 / subsample=0.8 / colsample_bytree=0.8 / reg_alpha=0.1 / reg_lambda=0.1 / seed=5422;XGBoost 之 level-wise(BFS)growth + Hessian-based 二階梯度。
3. **[Canonical Comparison Framework]** (v0.1, per RF 建立): metrics 與 LGBM v0.2 / LightGBM dedicated / 既存 XGBoost v0.1 / CatBoost / Ensemble / Random Forest / Extra Trees validators 完全 standardized,確保 8-tree comparison reliable。
4. **[Overlap-Corrected n_effective]** (v0.1, §14.7-CY T_CY-3): n_eff = n × (30/horizon),長 horizon 之 overlap penalty;effective t-stat = t × sqrt(n_eff/n)。
5. **[Honest Annualization]** (v0.1, §14.7-CY T_CY-4): mean × (252/horizon),非 √N 高估。
6. **[Cost-Drag Per Horizon]** (v0.1, §14.7-CY T_CY-5): 0.6%/rebal × rebals_per_year。
7. **[Precision Analysis]** (v0.1, new layer): Directional Hit Rate / Top-20 Actual Overlap / RMSE / MAE。
8. **[Reliability Analysis]** (v0.1, new layer): IC Stability CoV / Significance Robustness。
9. **[System Script Mandatory]** (v0.1, §14.7-CY T_CY-1): system 永久 script。
10. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query + (a) program output;0 AI memory reuse。
11. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): significance + precision tier 動態判定。
12. **[Sovereignty Declaration]** (v0.1, §3.1 序列模組): 本程式為 **§14.7-CY 第八 evaluation 實作 dedicated**(LGBM v0.2 / 既存 XGBoost / CatBoost / Ensemble / Random Forest / Extra Trees / LightGBM dedicated 為前七)。**治權邊界**:(a) §3.1 evaluation;(b) read-only;(c) 不訓練 production model;(d) 不修改 DB;(e) 唯一職責:XGBoost 4-horizon walk-forward + precision/reliability + JSON 持久化。
13. **[Historical Reference Authority]** (v0.1): 既存 XGBoost v0.1 multi-cycle 結果(quarterly Eff t=4.36 / Sharpe 2.63 / NetAnn +29.35%)為 §14.7-CY/CZ reference;本 dedicated v0.1 為 Canonical Comparison Framework 對齊版本,reproducibility-aware。
14. **[Idempotency]** (v0.1): pure read-only;JSON output 含 timestamp。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Horizon-Specific Walk-Forward — `--horizons <days_csv>`
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1-4 weekly/monthly/quarterly/annual | `evaluate_horizon()` 4 calls | §14.7-CY T_CY-2 |

### Group B. Walk-Forward XGBoost Training (per horizon)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window | train [0..i-1] → test i | §14.7-CW T_CW-2 |
| B.2 XGB params | XGB_PARAMS(200/0.05/5/5/0.8/0.8/0.1/0.1/5422)| §14.7-CW T_CW-4 |
| B.3 Level-wise growth | `tree_method='hist'` | XGBoost essence |
| B.4 Spearman IC | rank correlation | §14.7-CM |

### Group C. Overlap Correction + Honest Annualization
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 n_effective | n × (30/horizon) | §14.7-CY T_CY-3 |
| C.2 Effective t-stat | t × sqrt(n_eff/n) | §14.7-CY T_CY-3 |
| C.3 Annualization | mean × (252/horizon) | §14.7-CY T_CY-4 |
| C.4 Cost-drag | 0.006 × rebals_per_year | §14.7-CY T_CY-5 |

### Group D. Precision Analysis(新層)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1 Directional Hit Rate | sign(pred) == sign(actual) | precision |
| D.2 Top-20 Actual Overlap | predicted top-20 ∩ actual top-20 / 20 | precision |
| D.3 RMSE / MAE | magnitude error | regression |

### Group E. Reliability Analysis(新層)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| E.1 IC Stability CoV | std(IC) / |mean(IC)| | reliability |
| E.2 Significance Robust | abs(eff_t) > 1.997 | §14.7-CY T_CY-3 |

### Group F. JSON Persistence
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| F.1 Cross-cycle matrix stdout | per-horizon row | §14.7-CY T_CY-6 |
| F.2 JSON output | `reports/multi_cycle_xgboost_dedicated_<ts>.json` | §一.10 |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 日常 dry-run | `python scripts/evaluation/multi_cycle_xgboost_dedicated_validation.py --dry-run` |

### 不提供之旗標 (Intentionally Omitted)
- `--seed`:固定 5422 per §14.7-CW T_CW-4。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **首版:§14.7-CY 第八實作 dedicated(XGBoost)** under Canonical Comparison Framework。 (1) 4-horizon walk-forward,xgboost XGBRegressor;(2) 與既存 multi_cycle_xgboost_validation.py 並存;此 dedicated 版本為 Canonical Comparison Framework 對齊;(3) Precision/Reliability 新層延續其他 multi-cycle validators;(4) Hyperparameters 對齊既存 XGBoost v0.1。 | **ACTIVE** |
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
import xgboost as xgb
from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
SEED = 5422

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": SEED,
    "verbosity": 0,
    "n_jobs": -1,
}

SPEC_43 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "zero_volume_ratio_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "net_income_positive_ratio_8q",
    "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "preferential_attachment_60d",
    "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "trust_net_20d", "trust_net_60d", "margin_ratio_60d",
    # §14.7-DC v0.3 strict: theme_is_semiconductor + fitness_signal_60d + theme_strength all removed (hardcoded knowledge / transitively tainted = AI hallucination)
]


def get_panel_dates():
    dates = []
    current = date(2018, 6, 15)
    while current <= date(2026, 4, 30):
        dates.append((f"fs_{current.strftime('%Y%m%d')}_feature_set_v0_5", current))
        if current.month == 12: current = date(current.year+1, 1, 15)
        else: current = date(current.year, current.month+1, 15)
    return dates


def load_features(cur, fs_id, universe):
    cur.execute("SELECT stock_id, feature_name, feature_value::numeric FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s) AND is_null_imputed IS NOT TRUE", (fs_id, list(universe)))
    feat_data = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_43:
            feat_data[sid][fname] = float(val)
    X, sids = [], []
    for sid in universe:
        if sid in feat_data:
            X.append([feat_data[sid].get(f, 0.0) for f in SPEC_43]); sids.append(sid)
    return X, sids


def load_forward_returns(cur, as_of, horizon_days):
    cur.execute("SELECT MIN(date) FROM \"TaiwanStockPriceAdj\" WHERE date >= (%s::date + INTERVAL '%s days') AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')", (str(as_of), horizon_days, str(as_of), horizon_days + 14))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date: return {}, None
    cur.execute("WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id", (str(as_of), str(label_date)))
    return {sid: float(r) for sid, r in cur.fetchall()}, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10: return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    return np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q))


def evaluate_horizon(cur, panels, horizon_days, universe, label):
    logger.info(f"\n{'='*100}\nHorizon: {label}({horizon_days}d)\n{'='*100}")
    panel_data = {}
    t0 = time.monotonic()
    for fs_id, as_of in panels:
        X, sids = load_features(cur, fs_id, universe)
        if not X: continue
        returns, label_date = load_forward_returns(cur, as_of, horizon_days)
        if not returns: continue
        XX, yy, sids_matched = [], [], []
        for i, sid in enumerate(sids):
            if sid in returns:
                XX.append(X[i]); yy.append(returns[sid]); sids_matched.append(sid)
        if XX: panel_data[as_of] = (XX, yy, sids_matched, label_date)
    logger.info(f"  Loaded {len(panel_data)} panels(load: {time.monotonic()-t0:.1f}s)")

    panel_keys = sorted(panel_data.keys())
    panel_ics, panel_top20_rets, panel_univ_rets = [], [], []
    panel_hit_rates, panel_overlaps, panel_rmses, panel_maes = [], [], [], []

    for i in range(1, len(panel_keys)):
        test_key = panel_keys[i]
        X_test, y_test, _, _ = panel_data[test_key]
        train_X, train_y = [], []
        for j in range(i):
            X_j, y_j, _, _ = panel_data[panel_keys[j]]
            train_X.extend(X_j); train_y.extend(y_j)
        X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y))
        if len(X_tr) < 100: continue

        fold_model = xgb.XGBRegressor(**XGB_PARAMS)
        fold_model.fit(X_tr, y_tr)
        X_te = np.array(X_test)
        pred_te = fold_model.predict(X_te)
        ic = spearman_ic(pred_te, y_test)

        n_top = min(20, len(pred_te))
        top_idx = np.argsort(pred_te)[-n_top:]
        actual_top_idx = np.argsort(y_test)[-n_top:]
        top20_ret = float(np.mean([y_test[k] for k in top_idx]))
        univ_ret = float(np.mean(y_test))

        y_arr = np.array(y_test); pred_arr = np.array(pred_te)
        hit_rate = float(np.mean(np.sign(pred_arr) == np.sign(y_arr)))
        overlap = len(set(top_idx.tolist()) & set(actual_top_idx.tolist())) / n_top
        rmse = float(np.sqrt(np.mean((pred_arr - y_arr) ** 2)))
        mae = float(np.mean(np.abs(pred_arr - y_arr)))

        panel_ics.append(ic); panel_top20_rets.append(top20_ret); panel_univ_rets.append(univ_ret)
        panel_hit_rates.append(hit_rate); panel_overlaps.append(overlap); panel_rmses.append(rmse); panel_maes.append(mae)

    if not panel_top20_rets: return None

    n = len(panel_top20_rets)
    mean_ret = float(np.mean(panel_top20_rets))
    std_ret = float(np.std(panel_top20_rets, ddof=1)) if n > 1 else 0
    sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0
    win_rate = sum(1 for r in panel_top20_rets if r > 0) / n
    alphas = [t - u for t, u in zip(panel_top20_rets, panel_univ_rets)]
    mean_alpha = float(np.mean(alphas))
    std_alpha = float(np.std(alphas, ddof=1)) if n > 1 else 0
    ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0
    t_stat = mean_alpha / (std_alpha / math.sqrt(n)) if std_alpha > 0 else 0
    running = 0; peak = 0; mdd = 0
    for r in panel_top20_rets:
        running += r
        if running > peak: peak = running
        if peak - running > mdd: mdd = peak - running

    rebals_per_year = 252.0 / horizon_days
    annualized_log_gross = mean_ret * rebals_per_year
    annualized_simple_gross = math.exp(annualized_log_gross) - 1
    cost_per_rebal = 0.006
    annual_cost_drag = cost_per_rebal * rebals_per_year
    annualized_simple_net = math.exp(annualized_log_gross - annual_cost_drag) - 1

    panel_spacing = 30
    if horizon_days <= panel_spacing:
        n_eff = float(n); overlap_pct = 0.0
    else:
        n_eff = n * (panel_spacing / horizon_days)
        overlap_pct = (horizon_days - panel_spacing) / horizon_days * 100
    eff_t_stat = t_stat * math.sqrt(n_eff / n) if n > 0 else 0
    is_significant = abs(eff_t_stat) > 1.997

    mean_hit = float(np.mean(panel_hit_rates))
    mean_overlap = float(np.mean(panel_overlaps))
    mean_rmse = float(np.mean(panel_rmses))
    mean_mae = float(np.mean(panel_maes))
    ic_cov = float(np.std(panel_ics, ddof=1) / abs(np.mean(panel_ics))) if np.mean(panel_ics) != 0 else float('inf')

    result = {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "n_effective": n_eff, "overlap_pct": overlap_pct, "rebals_per_year": rebals_per_year,
        "mean_ret_per_panel": mean_ret, "sharpe": sharpe, "win_rate": win_rate, "mdd_per_panel": mdd,
        "mean_alpha_per_panel": mean_alpha, "ir": ir, "t_stat": t_stat,
        "effective_t_stat": eff_t_stat, "is_significant_p05": is_significant,
        "mean_ic": float(np.mean(panel_ics)),
        "annualized_simple_gross": annualized_simple_gross,
        "annual_cost_drag_log": annual_cost_drag,
        "annualized_simple_net": annualized_simple_net,
        "precision_directional_hit_rate": mean_hit,
        "precision_top20_actual_overlap": mean_overlap,
        "precision_rmse": mean_rmse,
        "precision_mae": mean_mae,
        "reliability_ic_stability_cov": ic_cov,
    }

    logger.info(f"\n  Results({label}, {horizon_days}d):")
    logger.info(f"    OOS panels: {n} | n_eff: {n_eff:.1f}")
    logger.info(f"    Sharpe: {sharpe:+.4f} | Win: {win_rate*100:.1f}% | α: {mean_alpha*100:+.4f}% | IR: {ir:+.4f}")
    logger.info(f"    Eff t-stat: {eff_t_stat:+.3f} | Sig p<0.05: {'✅' if is_significant else '❌'}")
    logger.info(f"    Annualized NET: {annualized_simple_net*100:+.2f}%/yr | Mean IC: {result['mean_ic']:+.4f}")
    logger.info(f"")
    logger.info(f"    --- Precision ---")
    logger.info(f"    Directional hit rate: {mean_hit*100:.1f}% | Top-20 actual overlap: {mean_overlap*100:.1f}%")
    logger.info(f"    RMSE: {mean_rmse:.4f} | MAE: {mean_mae:.4f}")
    logger.info(f"")
    logger.info(f"    --- Reliability ---")
    logger.info(f"    IC stability(CoV): {ic_cov:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=f"Multi-Cycle XGBoost Validation {TOOL_VER}(Canonical Comparison Framework)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="§一.10 #3 multi-run seed(canonical 5422;≥3 distinct seeds → min/median/max/mean)")
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True
    XGB_PARAMS["random_state"] = args.seed; globals()["SEED"] = args.seed  # §一.10 #3: inject run seed

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    logger.info("="*100)
    logger.info(f"Multi-Cycle XGBoost Validation {TOOL_VER}(per §14.7-CY / Canonical Comparison Framework)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  XGBoost version: {xgb.__version__}")
    logger.info(f"  Mode: {'COMMIT' if args.commit else 'DRY-RUN'}")

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id WHERE s.status='committed' AND m.core_tier='core_universe' AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)")
        universe = list({r[0] for r in cur.fetchall()})
        logger.info(f"  Universe: {len(universe)} stocks")

        panels = get_panel_dates()
        logger.info(f"  Panels: {len(panels)}(2018-06-15 ~ 2026-04-15 monthly)")

        results = {}
        t_global = time.monotonic()
        for label, days in horizon_labels:
            r = evaluate_horizon(cur, panels, days, universe, label)
            if r: results[label] = r

        logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix(XGBoost dedicated)+ Precision/Reliability\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'NetAnn':>9} {'HitRate':>9} {'Overlap':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['annualized_simple_net']*100:>+8.2f}% {r['precision_directional_hit_rate']*100:>8.1f}% {r['precision_top20_actual_overlap']*100:>8.1f}%")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_xgboost_dedicated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            output_results = {label: r for label, r in results.items()}
            output_results["_meta"] = {
                "tool": "multi_cycle_xgboost_dedicated_validation.py", "tool_ver": TOOL_VER,
                "model_family": "xgboost", "xgboost_version": xgb.__version__,
                "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER, "seed": SEED,
                "horizons": horizon_days_list, "n_universe": len(universe), "n_panels_input": len(panels),
                "source_traceability": "per CLAUDE.md §一.10 — all (b) DB query",
            }
            with open(output_full, "w") as f:
                json.dump(output_results, f, indent=2, default=str)
            logger.info(f"  Results persisted: {output_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
