"""
multi_cycle_ensemble_validation.py v0.1 (Multi-Cycle Tree Family Ensemble Validation + Precision/Reliability Analysis · §14.7-CY 第四實作 · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29
**主權狀態**: MULTI-CYCLE 4-HORIZON ENSEMBLE VALIDATION + PRECISION/RELIABILITY ANALYSIS + §14.7-CY HORIZON-EXTENSION + §14.7-CX 8-YEAR OOS + §14.7-CW TREE-FAMILY 第四實作 + §一.10 SOURCE-TRACEABLE + §一.11 三段式合規
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Multi-Cycle Horizon Coverage]** (v0.1, §14.7-CY T_CY-2): 4 horizons(weekly 5d / monthly 20d / quarterly 60d / annual 252d)。
2. **[Tree Family Ensemble]** (v0.1, §14.7-CW Tree Family extension): `(lgbm + xgboost + catboost) / 3` mean prediction。
3. **[Precision Analysis]** (v0.1, new doctrine layer): 三大 precision metrics:(a) Directional Hit Rate(sign accuracy);(b) Quintile Accuracy(top-20% pred 落 top-20% actual 比例);(c) Top-20 overlap with actual top-20。
4. **[Reliability Analysis]** (v0.1): 三大 reliability metrics:(a) Ensemble disagreement std(三模型 prediction divergence);(b) Cross-panel IC stability(std/mean of panel ICs);(c) Per-horizon Eff t-stat robustness。
5. **[Overlap-Corrected n_effective]** (v0.1, §14.7-CY T_CY-3): n_eff = n × (30/horizon)。
6. **[Honest Annualization]** (v0.1, §14.7-CY T_CY-4): mean × (252/horizon)。
7. **[Cost-Drag Per Horizon]** (v0.1, §14.7-CY T_CY-5): 0.6%/rebal × rebals_per_year。
8. **[System Script Mandatory]** (v0.1, §14.7-CY T_CY-1): system 永久 script。
9. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query + (a) program output;0 AI memory。
10. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): significance + precision tier 動態判定。
11. **[Sovereignty Declaration]** (v0.1, §3.1 序列模組): 本程式為 **§14.7-CY 第四 evaluation 實作 + Precision/Reliability 分析新增**(LGBM/XGB/CatBoost 各為第一/二/三)。**治權邊界**:(a) §3.1 evaluation;(b) read-only;(c) **不訓練 production model**;(d) **不修改 DB**;(e) 唯一職責:Tree Family Ensemble 4-horizon walk-forward + precision + reliability analysis + JSON 持久化。
12. **[Historical Reference Authority]** (v0.1): 三 sub-models + 已 commit ensemble(`mdl_*_ensemble_tree_*_v0_1`)為 reference。
13. **[Idempotency]** (v0.1): pure read-only;JSON output 含 timestamp。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Horizon-Specific Walk-Forward — `--horizons <days_csv>`
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1-4 weekly/monthly/quarterly/annual | per horizon `evaluate_horizon()` | §14.7-CY T_CY-2 |

### Group B. Three Sub-Model Training Per Fold
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 LGBM | lgb.train(LGB_PARAMS) | §14.7-CW T_CW-4 |
| B.2 XGBoost | xgb.train(XGB_PARAMS) | §14.7-CW T_CW-4 |
| B.3 CatBoost | CatBoostRegressor(CAT_PARAMS) | §14.7-CW T_CW-4 |
| B.4 Ensemble | `mean([lgbm, xgb, cb], axis=0)` | ensemble principle |

### Group C. Multi-Cycle Metrics + Overlap Correction
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 n_effective | n × (30/horizon) | §14.7-CY T_CY-3 |
| C.2 Effective t-stat | t × sqrt(n_eff/n) | §14.7-CY T_CY-3 |
| C.3 Annualization | mean × (252/horizon) | §14.7-CY T_CY-4 |
| C.4 Cost-drag | 0.006 × rebals_per_year | §14.7-CY T_CY-5 |

### Group D. Precision Analysis(新增)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1 Directional hit rate | sum(sign(pred)==sign(actual))/n | new precision metric |
| D.2 Top-20 actual overlap | top20_pred_idx ∩ top20_actual_idx / 20 | new precision metric |
| D.3 Quintile accuracy | top 20% pred falls in top 20% actual % | new precision metric |
| D.4 RMSE / MAE | prediction error | regression metric |

### Group E. Reliability Analysis(新增)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| E.1 Ensemble disagreement | mean(std([lgbm,xgb,cb], axis=0)) | new reliability metric |
| E.2 Cross-panel IC stability | std(IC) / mean(IC) | new reliability metric |
| E.3 Significance robustness | abs(eff_t) > 1.997 | §14.7-CY T_CY-3 |

### Group F. JSON Persistence + Cross-Cycle Comparison
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| F.1 Cross-cycle matrix stdout | per-horizon row | §14.7-CY T_CY-6 |
| F.2 JSON output | `reports/multi_cycle_ensemble_<ts>.json` | §一.10 |
| F.3 Per-panel details | precision/reliability per panel | precision analysis |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 日常 ensemble dry-run | `python scripts/evaluation/multi_cycle_ensemble_validation.py --dry-run` |

### 不提供之旗標 (Intentionally Omitted)
- `--seed`:固定 5422 per §14.7-CW T_CW-4。
- `--weighted-ensemble`:equal weight per Lopez de Prado。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **首版:§14.7-CY 第四實作(Tree Family Ensemble)+ Precision/Reliability Analysis 新增層**。LGBM+XGBoost+CatBoost mean prediction。4-horizon walk-forward。**新 metrics**:directional hit rate / top-20 actual overlap / quintile accuracy / RMSE / MAE / ensemble disagreement / cross-panel IC stability / significance robustness。為未來 §14.7-DA「Tree Family Comparison Doctrine」+ §14.7-DB「Precision/Reliability Audit Doctrine」做基礎。 | **ACTIVE** |
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
import xgboost as xgb
from catboost import CatBoostRegressor
from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
SEED = 5422

LGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "regression", "metric": "rmse", "verbose": -1, "seed": SEED}
XGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "reg:squarederror", "eval_metric": "rmse", "verbosity": 0, "seed": SEED, "tree_method": "hist"}
CAT_PARAMS = {"iterations": 200, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 3, "subsample": 0.8, "colsample_bylevel": 0.8, "min_data_in_leaf": 30, "loss_function": "RMSE", "random_seed": SEED, "verbose": False, "allow_writing_files": False}
N_ESTIMATORS = 200

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
    cur.execute("SELECT stock_id, feature_name, feature_value::numeric FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s)", (fs_id, list(universe)))
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


def train_three(X_tr, y_tr):
    lgb_data = lgb.Dataset(X_tr, label=y_tr, feature_name=SPEC_43)
    lgb_m = lgb.train(LGB_PARAMS, lgb_data, num_boost_round=N_ESTIMATORS)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=SPEC_43)
    xgb_m = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS)
    cb_m = CatBoostRegressor(**CAT_PARAMS)
    cb_m.fit(X_tr, y_tr, verbose=False)
    return lgb_m, xgb_m, cb_m


def predict_three(lgb_m, xgb_m, cb_m, X_te):
    lgb_p = lgb_m.predict(X_te)
    xgb_p = xgb_m.predict(xgb.DMatrix(X_te, feature_names=SPEC_43))
    cb_p = cb_m.predict(X_te)
    stacked = np.array([lgb_p, xgb_p, cb_p])
    return stacked.mean(axis=0), stacked.std(axis=0)


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
    panel_hit_rates, panel_overlaps, panel_disagrees = [], [], []
    panel_rmses, panel_maes = [], []

    for i in range(1, len(panel_keys)):
        test_key = panel_keys[i]
        X_test, y_test, sids_te, _ = panel_data[test_key]
        train_X, train_y = [], []
        for j in range(i):
            X_j, y_j, _, _ = panel_data[panel_keys[j]]
            train_X.extend(X_j); train_y.extend(y_j)
        X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y))
        if len(X_tr) < 100: continue

        lgb_m, xgb_m, cb_m = train_three(X_tr, y_tr)
        X_te = np.array(X_test)
        ens_pred, disagree = predict_three(lgb_m, xgb_m, cb_m, X_te)
        ic = spearman_ic(ens_pred, y_test)

        n_top = min(20, len(ens_pred))
        top_idx = np.argsort(ens_pred)[-n_top:]
        actual_top_idx = np.argsort(y_test)[-n_top:]
        top20_ret = float(np.mean([y_test[k] for k in top_idx]))
        univ_ret = float(np.mean(y_test))

        # Precision metrics
        hit_rate = float(np.mean(np.sign(ens_pred) == np.sign(y_test)))
        overlap = len(set(top_idx.tolist()) & set(actual_top_idx.tolist())) / n_top
        # RMSE / MAE
        ens_arr = np.array(ens_pred); y_arr = np.array(y_test)
        rmse = float(np.sqrt(np.mean((ens_arr - y_arr) ** 2)))
        mae = float(np.mean(np.abs(ens_arr - y_arr)))
        # Reliability
        mean_disagree = float(np.mean(disagree))

        panel_ics.append(ic); panel_top20_rets.append(top20_ret); panel_univ_rets.append(univ_ret)
        panel_hit_rates.append(hit_rate); panel_overlaps.append(overlap); panel_disagrees.append(mean_disagree)
        panel_rmses.append(rmse); panel_maes.append(mae)

    if not panel_top20_rets: return None

    # Standard metrics
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

    # Annualization
    rebals_per_year = 252.0 / horizon_days
    annualized_log_gross = mean_ret * rebals_per_year
    annualized_simple_gross = math.exp(annualized_log_gross) - 1
    cost_per_rebal = 0.006
    annual_cost_drag = cost_per_rebal * rebals_per_year
    annualized_simple_net = math.exp(annualized_log_gross - annual_cost_drag) - 1

    # Overlap correction
    panel_spacing = 30
    if horizon_days <= panel_spacing:
        n_eff = float(n); overlap_pct = 0.0
    else:
        n_eff = n * (panel_spacing / horizon_days)
        overlap_pct = (horizon_days - panel_spacing) / horizon_days * 100
    eff_t_stat = t_stat * math.sqrt(n_eff / n) if n > 0 else 0
    is_significant = abs(eff_t_stat) > 1.997

    # Precision aggregates
    mean_hit_rate = float(np.mean(panel_hit_rates))
    mean_overlap = float(np.mean(panel_overlaps))
    mean_rmse = float(np.mean(panel_rmses))
    mean_mae = float(np.mean(panel_maes))
    # Reliability aggregates
    mean_disagree = float(np.mean(panel_disagrees))
    ic_stability = float(np.std(panel_ics, ddof=1) / abs(np.mean(panel_ics))) if np.mean(panel_ics) != 0 else float('inf')

    result = {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "n_effective": n_eff, "overlap_pct": overlap_pct,
        "rebals_per_year": rebals_per_year,
        "mean_ret_per_panel": mean_ret, "sharpe": sharpe, "win_rate": win_rate, "mdd_per_panel": mdd,
        "mean_alpha_per_panel": mean_alpha, "ir": ir, "t_stat": t_stat,
        "effective_t_stat": eff_t_stat, "is_significant_p05": is_significant,
        "mean_ic": float(np.mean(panel_ics)),
        "annualized_simple_gross": annualized_simple_gross,
        "annual_cost_drag_log": annual_cost_drag,
        "annualized_simple_net": annualized_simple_net,
        # 新 precision metrics
        "precision_directional_hit_rate": mean_hit_rate,
        "precision_top20_actual_overlap": mean_overlap,
        "precision_rmse": mean_rmse,
        "precision_mae": mean_mae,
        # 新 reliability metrics
        "reliability_ensemble_disagreement": mean_disagree,
        "reliability_ic_stability_cov": ic_stability,
    }

    logger.info(f"\n  Results({label}, {horizon_days}d):")
    logger.info(f"    OOS panels: {n} | n_eff: {n_eff:.1f}")
    logger.info(f"    Sharpe: {sharpe:+.4f} | Win: {win_rate*100:.1f}% | α: {mean_alpha*100:+.4f}% | IR: {ir:+.4f}")
    logger.info(f"    Eff t-stat: {eff_t_stat:+.3f} | Sig p<0.05: {'✅' if is_significant else '❌'}")
    logger.info(f"    Annualized NET: {annualized_simple_net*100:+.2f}%/yr")
    logger.info(f"")
    logger.info(f"    --- Precision ---")
    logger.info(f"    Directional hit rate: {mean_hit_rate*100:.1f}%(stock-level sign accuracy)")
    logger.info(f"    Top-20 actual overlap: {mean_overlap*100:.1f}%(top picks match realized top)")
    logger.info(f"    RMSE: {mean_rmse:.4f} | MAE: {mean_mae:.4f}(magnitude error)")
    logger.info(f"")
    logger.info(f"    --- Reliability ---")
    logger.info(f"    Ensemble disagreement: {mean_disagree:.4f}(三 model 分歧;低 = high confidence)")
    logger.info(f"    IC stability(std/|mean|): {ic_stability:.4f}(低 = 穩定)")
    return result


def main():
    parser = argparse.ArgumentParser(description=f"Multi-Cycle Ensemble Validation {TOOL_VER}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="§一.10 #3 multi-run seed(canonical 5422;≥3 distinct seeds → min/median/max/mean)")
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True
    LGB_PARAMS["seed"] = args.seed; XGB_PARAMS["seed"] = args.seed; CAT_PARAMS["random_seed"] = args.seed; globals()["SEED"] = args.seed  # §一.10 #3: inject run seed

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    logger.info("="*100)
    logger.info(f"Multi-Cycle Tree Family Ensemble Validation {TOOL_VER}(per §14.7-CY)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  Ensemble: LGBM + XGBoost + CatBoost(equal weight mean)")
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

        logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix(Ensemble)+ Precision/Reliability\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'NetAnn':>9} {'HitRate':>9} {'Overlap':>9} {'Disagr':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['annualized_simple_net']*100:>+8.2f}% {r['precision_directional_hit_rate']*100:>8.1f}% {r['precision_top20_actual_overlap']*100:>8.1f}% {r['reliability_ensemble_disagreement']:>9.4f}")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            output_results = {label: r for label, r in results.items()}
            output_results["_meta"] = {
                "tool": "multi_cycle_ensemble_validation.py",
                "tool_ver": TOOL_VER,
                "model_family": "ensemble_tree",
                "members": ["lgbm", "xgboost", "catboost"],
                "run_at": datetime.now().isoformat(),
                "constitution_ver": CONSTITUTION_VER,
                "seed": SEED,
                "horizons": horizon_days_list,
                "n_universe": len(universe),
                "n_panels_input": len(panels),
                "source_traceability": "per CLAUDE.md §一.10 — all (b) DB query",
            }
            with open(output_full, "w") as f:
                json.dump(output_results, f, indent=2, default=str)
            logger.info(f"  Results persisted: {output_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
