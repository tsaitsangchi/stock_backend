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
from core.db_utils import get_db_conn, get_canonical_panel_dates, summarize_horizon_metrics

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


def load_features(cur, fs_id, universe):
    cur.execute("SELECT stock_id, feature_name, feature_value::numeric FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s) AND is_null_imputed IS NOT TRUE", (fs_id, list(universe)))
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
    panel_pa = []  # §14.7-DF: (pred, actual) per panel → 共用 helper(單一來源)

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
        panel_pa.append((ens_pred, y_test))
    result = summarize_horizon_metrics(label, horizon_days, panel_pa)  # §14.7-DF Canonical Metric SSOT(單一來源)
    if result is None:
        return None

    logger.info(f"  {label}({horizon_days}d): Sharpe {result['sharpe']:+.3f} | Eff t {result['effective_t_stat']:+.3f} | Win {result['win_rate']*100:.1f}% | IC {result['mean_ic']:+.4f} | NET {result['annualized_simple_net']*100:+.1f}%/yr")
    logger.info(f"    precision: hit {result['precision_directional_hit_rate']*100:.1f}% | top-20 overlap {result['precision_top20_actual_overlap']*100:.1f}% | RMSE {result['precision_rmse']:.4f} | reliability IC-CoV {result['reliability_ic_stability_cov']:.4f}")
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

        panels = get_canonical_panel_dates("feature_set_v0.5")  # §14.7-DE / §0.0-I 單一引用源
        logger.info(f"  Panels: {len(panels)} ({panels[0][1]} ~ {panels[-1][1]}, data-driven §14.7-DE)")

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
