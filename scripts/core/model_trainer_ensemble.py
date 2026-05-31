"""
model_trainer_ensemble.py v0.1 (Tree Family Ensemble Trainer · LGBM+XGBoost+CatBoost 三模型平均 · §14.7-CW Tree Family 第四實作 · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29
**主權狀態**: TREE FAMILY ENSEMBLE + §14.7-CW 第四實作(三 tree base 模型 mean prediction)+ §14.7-CS MODEL-TRAINING-LANDING + §14.7-CL 43-FEATURE CANONICAL + §一.10 SOURCE-TRACEABLE + §一.11 三段式合規
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Tree Family Ensemble Authority]** (v0.1, §14.7-CW Tree Family extension): 本程式為 LGBM v0.2 + XGBoost v0.1 + CatBoost v0.1 之 ensemble mean prediction;預期 sharper + 較單模型 robust(消 stochasticity)。
2. **[Three-Member Mean Prediction]** (v0.1): `ensemble_pred = (lgbm + xgb + catboost) / 3`;不採 weighted ensemble(避免 overfitting weights)。
3. **[Ensemble Disagreement = Confidence Proxy]** (v0.1): `std([lgbm, xgb, catboost])` 為 ensemble uncertainty;低 disagreement → 高 confidence。
4. **[Expanding Window Walk-Forward OOS]** (v0.1, 憲法 §14.7-CW T_CW-2): train [panel 0..i-1] → test panel i;三 sub-models 同 protocol。
5. **[Conservative Hyperparameters]** (v0.1, 憲法 §14.7-CW T_CW-4): 每 sub-model 用其原 v0.1/v0.2 conservative defaults;seed=5422 統一。
6. **[Treaty Gates 4/4]** (v0.1, §14.7-CW T_CW-5): Sharpe > 0 / Win ≥ 50% / MDD ≤ 30% / Mean α > 0。
7. **[43 Canonical Features]** (v0.1, §14.7-CL): 三 sub-models 共用同 SPEC_43。
8. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query + (a) program output;0 AI memory。
9. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): 4 Treaty Gates + 個別 sub-model 比較動態判定。
10. **[Sovereignty Declaration]** (v0.1, §3.1 序列模組): 本程式為 **§10 model_trainer 第四實作**(LGBM/XGB/CatBoost 為前三)。**治權邊界**:(a) §3.1 序列 training;(b) 五套禁令不涉;(c) T1-T3 不分層;(d) §8.5 features 已 anti-leakage;(e) 不選股 / 不算 feature;(f) **不評估 multi-cycle**(由 multi_cycle_ensemble_validation 負責);(g) 唯一職責:三 sub-models 各自訓練 + ensemble mean prediction + 統一 Treaty gates 評估 + 個別 + ensemble metrics 持久化。
11. **[Historical Reference Authority]** (v0.1): 三 sub-models v0.1/v0.2 為 prior implementation reference。
12. **[Idempotency]** (v0.1): model_registry INSERT ON CONFLICT;artifact_path 為 ensemble dir 含 3 sub-models + ensemble metrics。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Universe + Feature Loading
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Latest core_universe | DB query | §14.7-CF |
| A.2 Feature values | `load_panel_data()` | §14.7-CL |
| A.3 Forward returns | PriceAdj LN(t1/t0) | §14.7-CV |

### Group B. Three Sub-Model Training (per fold)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 LGBM sub-model | `lgb.train()` with seed 5422 | §14.7-CW T_CW-4 |
| B.2 XGBoost sub-model | `xgb.train()` with seed 5422 | §14.7-CW T_CW-4 |
| B.3 CatBoost sub-model | `CatBoostRegressor()` with seed 5422 | §14.7-CW T_CW-4 |

### Group C. Ensemble Mean Prediction + Disagreement
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Mean prediction | `(lgbm + xgb + catboost) / 3` | ensemble principle |
| C.2 Disagreement std | `std([lgbm, xgb, catboost], axis=0)` | confidence proxy |

### Group D. Treaty Gates Evaluation (Ensemble + per sub-model)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1-4 ensemble Sharpe/Win/MDD/α | per Gates | §14.7-CW Gates 1-4 |
| D.5 Per sub-model comparison | individual metrics | benchmark |

### Group E. Artifact Persistence (--commit only)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| E.1 model.txt (LGBM) + model.json (XGB) + model.cbm (CatBoost) | per sub-model | tri-format |
| E.2 ensemble_metrics.json | unified | §一.10 |
| E.3 model_registry INSERT | family='ensemble_tree' | §10 SSOT |

### Group F. CLI + Mode Control
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| F.1 `--dry-run` / `--commit` | mode control | safe default |
| F.2 `--panel-feature-sets <csv>` | walk-forward override | §14.7-CX |
| F.3 `--label-horizon N` | default 30d | §14.7-CW |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 日常 dry-run | `python scripts/core/model_trainer_ensemble.py --dry-run` |
| Commit ensemble | `python scripts/core/model_trainer_ensemble.py --commit` |

### 不提供之旗標 (Intentionally Omitted)
- `--weighted-ensemble`:採 equal weight 避免 overfitting(per Lopez de Prado 建議)。
- `--add-fourth-model`:tree family 固定 3 members。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **首版:§14.7-CW Tree Family 第四實作(Ensemble)**。LGBM + XGBoost + CatBoost mean prediction。Equal weight(per Lopez de Prado《Advances in Financial ML》Chapter 6 — 加權 ensemble 容易 overfit weights)。8-panel walk-forward 全 4 Treaty Gates 動態判定。三 sub-models 個別 metrics + ensemble metrics 比較。Model artifact 含三 sub-model files + ensemble metrics.json。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, hashlib, json, logging, math
from datetime import datetime, date
from pathlib import Path
from collections import defaultdict

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
MODEL_FAMILY = "ensemble_tree"
SEED = 5422

LGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "regression", "metric": "rmse", "verbose": -1, "seed": SEED}
XGB_PARAMS = {"learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "reg:squarederror", "eval_metric": "rmse", "verbosity": 0, "seed": SEED, "tree_method": "hist"}
CAT_PARAMS = {"iterations": 200, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 3, "subsample": 0.8, "colsample_bylevel": 0.8, "min_data_in_leaf": 30, "loss_function": "RMSE", "random_seed": SEED, "verbose": False, "allow_writing_files": False}
N_ESTIMATORS = 200

DEFAULT_PANELS = [
    ("fs_20250915_feature_set_v0_5", "2025-09-15"),
    ("fs_20251015_feature_set_v0_5", "2025-10-15"),
    ("fs_20251115_feature_set_v0_5", "2025-11-15"),
    ("fs_20251215_feature_set_v0_5", "2025-12-15"),
    ("fs_20260115_feature_set_v0_5", "2026-01-15"),
    ("fs_20260215_feature_set_v0_5", "2026-02-15"),
    ("fs_20260315_feature_set_v0_5", "2026-03-15"),
    ("fs_20260415_feature_set_v0_5", "2026-04-15"),
]

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


def load_panel_data(cur, fs_id, as_of, label_horizon, universe):
    cur.execute("SELECT stock_id, feature_name, feature_value::numeric FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s) AND is_null_imputed IS NOT TRUE", (fs_id, list(universe)))
    feat_data = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_43:
            feat_data[sid][fname] = float(val)
    cur.execute("SELECT MIN(date) FROM \"TaiwanStockPriceAdj\" WHERE date >= (%s::date + INTERVAL '%s days') AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')", (as_of, label_horizon, as_of, label_horizon + 10))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date: return [], [], [], None
    cur.execute("WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id", (as_of, label_date))
    returns = {sid: float(r) for sid, r in cur.fetchall() if sid in universe}
    X, y, sids = [], [], []
    for sid in universe:
        if sid in feat_data and sid in returns:
            X.append([feat_data[sid].get(f, 0.0) for f in SPEC_43])
            y.append(returns[sid]); sids.append(sid)
    return X, y, sids, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10: return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    return np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q))


def train_three_models(X_tr, y_tr):
    """Train LGBM + XGBoost + CatBoost on same data。"""
    # LGBM
    lgb_data = lgb.Dataset(X_tr, label=y_tr, feature_name=SPEC_43)
    lgb_model = lgb.train({k:v for k,v in LGB_PARAMS.items() if k != "n_estimators"}, lgb_data, num_boost_round=N_ESTIMATORS)
    # XGBoost
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=SPEC_43)
    xgb_model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS)
    # CatBoost
    cb_model = CatBoostRegressor(**CAT_PARAMS)
    cb_model.fit(X_tr, y_tr, verbose=False)
    return lgb_model, xgb_model, cb_model


def predict_three(lgb_m, xgb_m, cb_m, X_te):
    """Predict + return ensemble mean + per-model preds + disagreement"""
    lgb_pred = lgb_m.predict(X_te)
    xgb_pred = xgb_m.predict(xgb.DMatrix(X_te, feature_names=SPEC_43))
    cb_pred = cb_m.predict(X_te)
    stacked = np.array([lgb_pred, xgb_pred, cb_pred])
    ensemble_pred = stacked.mean(axis=0)
    disagreement = stacked.std(axis=0)
    return ensemble_pred, disagreement, lgb_pred, xgb_pred, cb_pred


def evaluate_predictions(pred, y, label):
    """Compute per-model Sharpe/Win/α from top-20 strategy"""
    ic = spearman_ic(pred, y)
    n_top = min(20, len(pred))
    top_idx = np.argsort(pred)[-n_top:]
    top20_ret = float(np.mean([y[k] for k in top_idx]))
    univ_ret = float(np.mean(y))
    return ic, top20_ret, univ_ret


def main():
    parser = argparse.ArgumentParser(description=f"Tree Family Ensemble Trainer ({TOOL_VER})")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--label-horizon", type=int, default=30)
    parser.add_argument("--panel-feature-sets", type=str, default=None)
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id WHERE s.status='committed' AND m.core_tier='core_universe' AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)")
        universe = list({r[0] for r in cur.fetchall()})

        if args.panel_feature_sets:
            panels = []
            for fs in args.panel_feature_sets.split(","):
                fs = fs.strip()
                cur.execute("SELECT as_of_date FROM feature_store_snapshot WHERE feature_set_id=%s", (fs,))
                r = cur.fetchone()
                if r: panels.append((fs, str(r[0])))
        else:
            panels = DEFAULT_PANELS

        logger.info("=" * 120)
        logger.info(f"§14.7-CW Tree Family Ensemble Trainer {TOOL_VER}(LGBM+XGBoost+CatBoost mean)")
        logger.info("=" * 120)
        logger.info(f"  Universe:              {len(universe)} stocks")
        logger.info(f"  Panels:                {len(panels)}")
        logger.info(f"  Label horizon:         {args.label_horizon}d")
        logger.info(f"  Mode:                  {'COMMIT' if args.commit else 'DRY-RUN'}")
        logger.info(f"  Features(§14.7-CL):  {len(SPEC_43)}")
        logger.info(f"  Seed:                  {SEED}")

        logger.info("\n──── Loading walk-forward training data ────")
        per_panel = {}
        for fs_id, as_of in panels:
            X, y, sids, label_date = load_panel_data(cur, fs_id, as_of, args.label_horizon, universe)
            if not X:
                logger.warning(f"  Panel {as_of}:no valid forward data,skipped")
                continue
            logger.info(f"  Panel {as_of} → label_date={label_date}:N={len(X)}")
            per_panel[as_of] = (X, y, sids, label_date)

        all_X = [x for k in per_panel.values() for x in k[0]]
        all_y = [y for k in per_panel.values() for y in k[1]]
        X_train = np.array(all_X); y_train = np.array(all_y)
        logger.info(f"\n  Total training rows: {len(X_train):,}")

        logger.info("\n──── Walk-Forward Expanding Window 三模型 + Ensemble OOS ────")

        panels_list = sorted(per_panel.items(), key=lambda x: x[0])
        # Storage
        panel_metrics = {model: {"ic": [], "top20": [], "univ": []} for model in ["lgbm", "xgboost", "catboost", "ensemble"]}
        # Per-panel detail(for precision/reliability analysis later)
        panel_details = []

        for i in range(1, len(panels_list)):
            test_as_of, (X_test, y_test, sids_test, label_date) = panels_list[i]
            train_X, train_y = [], []
            for j in range(i):
                X_j, y_j, _, _ = panels_list[j][1]
                train_X.extend(X_j); train_y.extend(y_j)
            X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y), 0.01, 0.99)

            lgb_m, xgb_m, cb_m = train_three_models(X_tr, y_tr)
            X_te = np.array(X_test)
            ens_pred, disagree, lgb_p, xgb_p, cb_p = predict_three(lgb_m, xgb_m, cb_m, X_te)

            # Per-model evaluation
            for name, pred in [("lgbm", lgb_p), ("xgboost", xgb_p), ("catboost", cb_p), ("ensemble", ens_pred)]:
                ic, top20_ret, univ_ret = evaluate_predictions(pred, y_test, name)
                panel_metrics[name]["ic"].append(ic)
                panel_metrics[name]["top20"].append(top20_ret)
                panel_metrics[name]["univ"].append(univ_ret)

            # Per-panel detail for ensemble(for precision analysis)
            ens_ic = panel_metrics["ensemble"]["ic"][-1]
            ens_top20 = panel_metrics["ensemble"]["top20"][-1]
            ens_univ = panel_metrics["ensemble"]["univ"][-1]
            n_top = min(20, len(ens_pred))
            top_idx = np.argsort(ens_pred)[-n_top:]
            actual_top_idx = np.argsort(y_test)[-n_top:]
            top20_overlap = len(set(top_idx.tolist()) & set(actual_top_idx.tolist())) / n_top
            mean_disagree = float(np.mean(disagree))

            panel_details.append({
                "panel": test_as_of, "n_test": len(X_te),
                "ic": ens_ic, "top20_ret": ens_top20, "univ_ret": ens_univ,
                "alpha": ens_top20 - ens_univ,
                "top20_overlap_actual": top20_overlap,  # precision metric
                "mean_disagreement": mean_disagree,  # reliability metric
                "top20_stock_ids": [sids_test[k] for k in top_idx],
                "actual_top20_stock_ids": [sids_test[k] for k in actual_top_idx],
            })

            logger.info(f"  Train[0..{i-1}] → Test {test_as_of}:")
            logger.info(f"    LGBM     IC={panel_metrics['lgbm']['ic'][-1]:+.4f} / Top20={panel_metrics['lgbm']['top20'][-1]:+.4f}")
            logger.info(f"    XGBoost  IC={panel_metrics['xgboost']['ic'][-1]:+.4f} / Top20={panel_metrics['xgboost']['top20'][-1]:+.4f}")
            logger.info(f"    CatBoost IC={panel_metrics['catboost']['ic'][-1]:+.4f} / Top20={panel_metrics['catboost']['top20'][-1]:+.4f}")
            logger.info(f"    ENSEMBLE IC={ens_ic:+.4f} / Top20={ens_top20:+.4f} / Top20-actual overlap={top20_overlap*100:.0f}% / disagreement std={mean_disagree:.4f}")

        # Compute final metrics per model
        logger.info("\n──── Per-Model Final Metrics ────")
        results = {}
        for name in ["lgbm", "xgboost", "catboost", "ensemble"]:
            m = panel_metrics[name]
            top20 = m["top20"]; univ = m["univ"]; ics = m["ic"]
            mean_ret = float(np.mean(top20)); std_ret = float(np.std(top20, ddof=1))
            sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0
            win = sum(1 for r in top20 if r > 0) / len(top20)
            alphas = [t - u for t, u in zip(top20, univ)]
            mean_alpha = float(np.mean(alphas)); std_alpha = float(np.std(alphas, ddof=1))
            ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0
            t_stat = mean_alpha / (std_alpha / math.sqrt(len(alphas))) if std_alpha > 0 else 0
            mdd = 0; peak = 0; running = 0
            for r in top20:
                running += r
                if running > peak: peak = running
                if peak - running > mdd: mdd = peak - running
            cum = sum(top20)
            mean_ic = float(np.mean(ics))
            results[name] = {
                "sharpe": sharpe, "win_rate": win, "mdd": mdd,
                "mean_alpha": mean_alpha, "ir": ir, "t_statistic": t_stat,
                "cumulative_return": cum, "mean_ic": mean_ic, "n_panels": len(top20),
            }
            logger.info(f"  {name:10}: Sharpe={sharpe:+.4f} / Win={win*100:.1f}% / α={mean_alpha*100:+.2f}% / IR={ir:+.4f} / MDD={mdd*100:.2f}% / IC={mean_ic:+.4f}")

        # Treaty Gates for ensemble
        ens = results["ensemble"]
        logger.info("\n──── §14.7-CW Treaty Gates(Ensemble)────")
        g1 = "✅ PASS" if ens["sharpe"] > 0 else "❌"
        g2 = "✅ PASS" if ens["win_rate"] >= 0.5 else "❌"
        g3 = "✅ PASS" if ens["mdd"] <= 0.30 else "⚠️"
        g4 = "✅ PASS" if ens["mean_alpha"] > 0 else "❌"
        logger.info(f"  Gate CW-1: {g1}({ens['sharpe']:.4f})")
        logger.info(f"  Gate CW-2: {g2}({ens['win_rate']*100:.1f}%)")
        logger.info(f"  Gate CW-3: {g3}({ens['mdd']*100:.2f}%)")
        logger.info(f"  Gate CW-4: {g4}({ens['mean_alpha']:.4f})")

        # Precision + reliability summary
        logger.info("\n──── Ensemble Precision + Reliability ────")
        avg_overlap = float(np.mean([d["top20_overlap_actual"] for d in panel_details]))
        avg_disagree = float(np.mean([d["mean_disagreement"] for d in panel_details]))
        logger.info(f"  Top-20 actual overlap mean: {avg_overlap*100:.1f}%(precision)")
        logger.info(f"  Mean disagreement std:      {avg_disagree:.4f}(reliability:lower = higher confidence)")

        if args.commit:
            logger.info("\n──── COMMIT mode ────")
            feature_set_hash = hashlib.sha1("feature_set_v0.4".encode()).hexdigest()[:8]
            train_date = max(p[1] for p in panels)
            model_id = f"mdl_{train_date.replace('-', '')}_ensemble_tree_h{args.label_horizon}_{feature_set_hash}_v0_1"
            artifact_dir = Path("data/models") / model_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Train final on all data + save sub-models
            y_train_w = winsorize(y_train, 0.01, 0.99)
            lgb_m, xgb_m, cb_m = train_three_models(X_train, y_train_w)
            lgb_m.save_model(str(artifact_dir / "model_lgbm.txt"))
            xgb_m.save_model(str(artifact_dir / "model_xgboost.json"))
            cb_m.save_model(str(artifact_dir / "model_catboost.cbm"))

            metrics = {
                "trainer": "ensemble_tree_v0_1",
                "model_family": MODEL_FAMILY,
                "members": ["lgbm", "xgboost", "catboost"],
                "label_horizon": args.label_horizon,
                "feature_count": len(SPEC_43),
                "rows_trained": len(X_train),
                "panels": len(panels),
                "per_model_metrics": results,
                "ensemble_metrics": results["ensemble"],
                "precision_top20_overlap_mean": avg_overlap,
                "reliability_disagreement_mean": avg_disagree,
                "panel_details": panel_details,
            }
            with open(artifact_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            params_dump = {"lgb": LGB_PARAMS, "xgb": XGB_PARAMS, "catboost": CAT_PARAMS, "n_estimators": N_ESTIMATORS, "seed": SEED}
            with open(artifact_dir / "hyperparams.json", "w") as f:
                json.dump(params_dump, f, indent=2)

            cur.execute("SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY snapshot_id DESC LIMIT 1")
            r = cur.fetchone()
            universe_snapshot_id = r[0] if r else None
            cur.execute("""
                INSERT INTO model_registry(model_id, model_policy_version, model_family, feature_set_id, universe_snapshot_id, label_horizon, train_start_date, train_end_date, metrics, hyperparams, artifact_path, status, notes)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)
                ON CONFLICT(model_id) DO UPDATE SET metrics = EXCLUDED.metrics, hyperparams = EXCLUDED.hyperparams, artifact_path = EXCLUDED.artifact_path, status = EXCLUDED.status, notes = EXCLUDED.notes
            """, (model_id, "model_policy_v0.1", MODEL_FAMILY, panels[-1][0], universe_snapshot_id, args.label_horizon, panels[0][1], train_date, json.dumps(metrics, default=str), json.dumps(params_dump), str(artifact_dir), "committed", "v0.1 Tree Family Ensemble(LGBM+XGB+CatBoost mean)"))
            conn.commit()
            logger.info(f"  ✅ Ensemble committed: {model_id}")
            logger.info(f"  ✅ Artifact: {artifact_dir}/(3 sub-models + metrics)")
            logger.info(f"  ✅ model_registry inserted")

        logger.info("\n" + "=" * 120)
        verdict = "PERFECT" if all("PASS" in g for g in [g1, g2, g3, g4]) else "WARNING"
        logger.info(f"§14.7-CW Tree Family Ensemble Trainer {TOOL_VER}: 主權判定 {verdict}")
        logger.info("=" * 120)
        if verdict == "PERFECT":
            logger.info(f"  🎯 Ensemble: Sharpe={ens['sharpe']:.2f} / IR={ens['ir']:.2f} / Win={ens['win_rate']*100:.0f}% / Top-20 overlap={avg_overlap*100:.0f}%")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
