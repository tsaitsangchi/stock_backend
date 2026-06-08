"""
feature_shap_purged_audit.py v0.1 (五鏡之第四+五鏡:Ensemble TreeSHAP + Purged-CV · §8.5 / §14.7-CM / §一.15)
================================================================================
**最後更新日期**: 2026-06-07
**主權狀態**: RESEARCH-MODE ENSEMBLE-SHAP(model-aware signed contribution)+ PURGED/EMBARGO WALK-FORWARD(§8.5 anti-leakage)+ §一.10 SOURCE-TRACEABLE + 不動 DB / 不 commit
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用「五鏡特徵分診法」的**第四鏡(SHAP)+ 第五鏡(purged-CV)**補洞 ——
SHAP 給每個特徵「**在 ensemble 模型中、考慮其他特徵後**的有號貢獻」(比單變量 IC 精準、抓得到條件價值);
purged-CV 把長 horizon「相鄰 panel 報酬窗重疊」造成的洩漏清掉(對齊憲章 §8.5),看真實 IC 比沒洩漏時掉多少。

**它怎麼做**:
1. 397 核心 + canonical panels。2. walk-forward 訓練 ensemble(LightGBM+XGBoost+CatBoost 平均)。
3. **第五鏡 purged**:訓練只用「label 窗已在 test 時點前完全實現」的 panel(t_j + horizon ≤ t_i − embargo),清除重疊洩漏;同時跑「非 purged」對照。
4. **第四鏡 SHAP**:每 test panel 算三樹原生 SHAP(pred_contrib)平均 = ensemble SHAP;聚合各特徵 mean|SHAP|(重要度)+ 有號方向(corr(特徵值, SHAP))。
5. 並排輸出 purged vs 非 purged 的 OOS IC(揭露洩漏膨脹)+ 各特徵 SHAP 有號貢獻排名。

**source-pure**:全 (b) DB query;SHAP 用三樹原生 pred_contrib(免裝 shap 套件);anti-leakage(§8.5)+ purged 強化;不寫 DB、不 commit、不 retrain production(§3.1 evaluation)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. **[Ensemble TreeSHAP — Lens 4]** (v0.1): ensemble=三樹平均 → SHAP 可加性 → ensemble SHAP = mean(三樹原生 SHAP);給 model-aware 有號貢獻 + 條件價值(補單變量 IC 之盲點)。
2. **[Purged/Embargo Walk-Forward — Lens 5]** (v0.1, §8.5 / López de Prado): 訓練 panel 之 label 窗 [t_j, t_j+H] 與 test 重疊則 purge(t_j + H + embargo > t_i 排除);清長 horizon 重疊洩漏。
3. **[Leakage-Inflation Disclosure]** (v0.1, §一.8): 並排 purged vs 非 purged IC,誠實揭露長 horizon 訊號有多少來自重疊洩漏。
4. **[Source-Traceable]** (v0.1, §一.10): 全 (b) DB query;0 scipy;SHAP 三樹原生。
5. **[Research-Mode]** (v0.1, §3.1): read-only;不寫 DB / 不 commit / 不改 SPEC。
6. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): 重要度/方向/IC 動態計算。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
### A. 主要運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 季+年 SHAP + purged 對照 | `python scripts/evaluation/feature_shap_purged_audit.py` | §8.5 / §14.7-CM |
| A.2 指定 horizon | `--horizons 60` | §14.7-CY |
| A.3 embargo 天數 | `--embargo-days 5` | §8.5 |
| A.4 threads(對並行客氣) | `--threads 3` | 維運 |
### 不提供之旗標 (Intentionally Omitted)
- `--commit`:research-mode,永不寫 DB。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-07 | Claude | 首版:五鏡法第四(ensemble TreeSHAP 有號貢獻)+ 第五(purged/embargo walk-forward,§8.5 重疊洩漏清除)鏡;並排 purged vs 非 purged IC 揭露長 horizon 洩漏膨脹。research-mode。 | **ACTIVE** |
"""

from __future__ import annotations
import sys, argparse, math, json, logging, time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from core.db_utils import get_db_conn, get_canonical_panel_dates

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"
SEED = 5422
FEATURE_SET_VERSION = "feature_set_v0.6"
N_ESTIMATORS = 200

SPEC_37 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "margin_ratio_60d",
]


def lgb_params(threads):
    return {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 30,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "reg_alpha": 0.1,
            "reg_lambda": 0.1, "objective": "regression", "metric": "rmse", "verbose": -1,
            "seed": SEED, "num_threads": threads}

def xgb_params(threads):
    return {"learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "objective": "reg:squarederror",
            "eval_metric": "rmse", "verbosity": 0, "seed": SEED, "tree_method": "hist", "nthread": threads}

def cat_params(threads):
    return {"iterations": N_ESTIMATORS, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 3, "subsample": 0.8,
            "colsample_bylevel": 0.8, "min_data_in_leaf": 30, "loss_function": "RMSE", "random_seed": SEED,
            "verbose": False, "allow_writing_files": False, "thread_count": threads}


def load_universe(cur):
    cur.execute(
        "SELECT m.stock_id FROM core_universe_membership m JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id "
        "WHERE s.status='committed' AND m.core_tier='core_universe' "
        "AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)"
    )
    return list({r[0] for r in cur.fetchall()})


def load_features(cur, fs_id, universe):
    cur.execute(
        "SELECT stock_id, feature_name, feature_value::numeric FROM feature_values "
        "WHERE feature_set_id=%s AND stock_id=ANY(%s) AND is_null_imputed IS NOT TRUE",
        (fs_id, list(universe)),
    )
    feat = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_37:
            feat[sid][fname] = float(val)
    return feat


def load_forward_returns(cur, as_of, horizon_days):
    cur.execute(
        "SELECT MIN(date) FROM \"TaiwanStockPriceAdj\" WHERE date >= (%s::date + INTERVAL '%s days') "
        "AND stock_id ~ '^[0-9]' AND date <= (%s::date + INTERVAL '%s days')",
        (str(as_of), horizon_days, str(as_of), horizon_days + 14),
    )
    r = cur.fetchone()
    ld = r[0] if r and r[0] else None
    if not ld:
        return {}
    cur.execute(
        "WITH t0 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0), "
        "t1 AS (SELECT stock_id, close FROM \"TaiwanStockPriceAdj\" WHERE date=%s AND close>0) "
        "SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric) FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id",
        (str(as_of), str(ld)),
    )
    return {sid: float(r) for sid, r in cur.fetchall()}


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    return np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q))


def ensemble_train_shap(Xtr, ytr, Xte, threads):
    """訓練三樹 + 預測 + ensemble SHAP(三樹原生 pred_contrib 平均,去 bias 欄)。"""
    names = SPEC_37
    m_l = lgb.train(lgb_params(threads), lgb.Dataset(Xtr, label=ytr, feature_name=names), num_boost_round=N_ESTIMATORS)
    m_x = xgb.train(xgb_params(threads), xgb.DMatrix(Xtr, label=ytr, feature_names=names), num_boost_round=N_ESTIMATORS)
    m_c = CatBoostRegressor(**cat_params(threads)); m_c.fit(Xtr, ytr, verbose=False)
    # predictions
    p = (m_l.predict(Xte) + m_x.predict(xgb.DMatrix(Xte, feature_names=names)) + m_c.predict(Xte)) / 3.0
    # SHAP(去最後 bias 欄)
    s_l = m_l.predict(Xte, pred_contrib=True)[:, :-1]
    s_x = m_x.predict(xgb.DMatrix(Xte, feature_names=names), pred_contribs=True)[:, :-1]
    s_c = np.array(m_c.get_feature_importance(Pool(Xte), type='ShapValues'))[:, :-1]
    shap_ens = (s_l + s_x + s_c) / 3.0
    return p, shap_ens


def evaluate(cur, panels, feat_cache, horizon_days, label, purge, embargo_days, threads):
    mode = f"purged(embargo={embargo_days}d)" if purge else "非purged"
    logger.info(f"\n{'='*100}\nHorizon {label}({horizon_days}d) — {mode}\n{'='*100}")
    panel_data = {}
    for fs_id, as_of in panels:
        if as_of not in feat_cache:
            continue
        sids, X = feat_cache[as_of]
        ret = load_forward_returns(cur, as_of, horizon_days)
        idx = [i for i, s in enumerate(sids) if s in ret]
        if not idx:
            continue
        panel_data[as_of] = (X[idx, :], np.array([ret[sids[i]] for i in idx]))

    keys = sorted(panel_data.keys())
    purge_cal = int(horizon_days * 1.45) + embargo_days  # trading→calendar 近似 + embargo
    ic_list = []
    shap_abs_sum = np.zeros(len(SPEC_37)); shap_n = 0
    feat_vals_acc = [[] for _ in SPEC_37]; shap_vals_acc = [[] for _ in SPEC_37]
    n_purged_total = 0
    for i in range(1, len(keys)):
        t_i = keys[i]
        Xte, yte = panel_data[t_i]
        tr_js = []
        for j in range(i):
            if purge and (t_i - keys[j]).days <= purge_cal:
                n_purged_total += 1
                continue
            tr_js.append(j)
        if not tr_js:
            continue
        Xtr = np.vstack([panel_data[keys[j]][0] for j in tr_js])
        ytr = winsorize(np.concatenate([panel_data[keys[j]][1] for j in tr_js]))
        if len(Xtr) < 100:
            continue
        pred, shap_ens = ensemble_train_shap(Xtr, ytr, Xte, threads)
        ic_list.append(spearman_ic(pred, yte))
        shap_abs_sum += np.abs(shap_ens).mean(axis=0); shap_n += 1
        for f in range(len(SPEC_37)):
            feat_vals_acc[f].append(Xte[:, f]); shap_vals_acc[f].append(shap_ens[:, f])

    if shap_n == 0:
        return None
    mean_abs_shap = shap_abs_sum / shap_n
    # 有號方向:corr(特徵值, 該特徵 SHAP)
    signed_dir = []
    for f in range(len(SPEC_37)):
        fv = np.concatenate(feat_vals_acc[f]); sv = np.concatenate(shap_vals_acc[f])
        if np.std(fv) < 1e-12 or np.std(sv) < 1e-12:
            signed_dir.append(0.0)
        else:
            signed_dir.append(float(np.corrcoef(fv, sv)[0, 1]))
    mean_ic = float(np.mean(ic_list))
    logger.info(f"  OOS IC = {mean_ic:+.4f}(n={len(ic_list)} panels;purged 共剔除 {n_purged_total} 訓練-panel 次)")
    feats = sorted(range(len(SPEC_37)), key=lambda k: mean_abs_shap[k], reverse=True)
    return {"horizon": label, "horizon_days": horizon_days, "mode": mode, "mean_ic": mean_ic,
            "n_panels": len(ic_list), "n_purged": n_purged_total,
            "shap_ranking": [{"feature": SPEC_37[k], "mean_abs_shap": float(mean_abs_shap[k]),
                              "signed_direction": signed_dir[k]} for k in feats]}


def main():
    parser = argparse.ArgumentParser(description=f"Feature SHAP + Purged-CV Audit {TOOL_VER}")
    parser.add_argument("--horizons", type=str, default="60,252")
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    labels = []
    for d in horizon_days_list:
        labels.append(("weekly" if d <= 7 else "monthly" if d <= 30 else "quarterly" if d <= 90 else "annual", d))

    logger.info("=" * 100)
    logger.info(f"五鏡 第四(Ensemble SHAP)+ 第五(Purged-CV)Audit {TOOL_VER}(§8.5 / §14.7-CM)")
    logger.info("=" * 100)
    logger.info(f"  ensemble: LGBM+XGBoost+CatBoost | threads={args.threads} | embargo={args.embargo_days}d")

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        universe = load_universe(cur)
        panels = get_canonical_panel_dates(FEATURE_SET_VERSION)
        logger.info(f"  Universe {len(universe)} | Panels {len(panels)} ({panels[0][1]}~{panels[-1][1]})")
        feat_cache = {}
        for fs_id, as_of in panels:
            feat = load_features(cur, fs_id, universe)
            sids = [s for s in universe if feat.get(s) and all(f in feat[s] for f in SPEC_37)]
            if sids:
                feat_cache[as_of] = (sids, np.array([[feat[s][f] for f in SPEC_37] for s in sids]))
        logger.info(f"  feature cache: {len(feat_cache)} panels\n")

        results = {}
        t0 = time.monotonic()
        for label, days in labels:
            for purge in (False, True):
                r = evaluate(cur, panels, feat_cache, days, label, purge, args.embargo_days, args.threads)
                if r:
                    results[f"{label}_{'purged' if purge else 'raw'}"] = r

        # 並排:purged vs raw IC + SHAP top10
        logger.info(f"\n{'='*100}\n第五鏡:purged vs 非purged OOS IC(揭露長 horizon 重疊洩漏膨脹)\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'非purged IC':>12} {'purged IC':>12} {'洩漏膨脹Δ':>12}")
        for label, _ in labels:
            raw = results.get(f"{label}_raw"); pur = results.get(f"{label}_purged")
            if raw and pur:
                infl = raw["mean_ic"] - pur["mean_ic"]
                logger.info(f"  {label:10} {raw['mean_ic']:>+12.4f} {pur['mean_ic']:>+12.4f} {infl:>+12.4f}")
        logger.info(f"\n{'='*100}\n第四鏡:Ensemble SHAP 有號貢獻 Top-12(purged 模式;季度)\n{'='*100}")
        q = results.get("quarterly_purged") or results.get("annual_purged")
        if q:
            logger.info(f"  {'#':>2} {'特徵':30} {'mean|SHAP|':>11} {'有號方向':>9}")
            for i, e in enumerate(q["shap_ranking"][:12], 1):
                arrow = "↑正" if e["signed_direction"] > 0.05 else "↓負" if e["signed_direction"] < -0.05 else "·中"
                logger.info(f"  {i:>2} {e['feature']:30} {e['mean_abs_shap']:>11.5f} {e['signed_direction']:>+7.2f}{arrow}")
        logger.info(f"\n  Total elapsed: {time.monotonic()-t0:.1f}s")

        out_path = args.output or f"reports/feature_shap_purged_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_full = Path(_base_dir).parent / out_path
        out_full.parent.mkdir(parents=True, exist_ok=True)
        results["_meta"] = {"tool": "feature_shap_purged_audit.py", "tool_ver": TOOL_VER,
                            "ensemble": ["lgbm", "xgboost", "catboost"], "embargo_days": args.embargo_days,
                            "run_at": datetime.now().isoformat(), "constitution_ver": CONSTITUTION_VER,
                            "seed": SEED, "horizons": horizon_days_list, "n_universe": len(universe),
                            "n_panels": len(panels), "source_traceability": "per §一.10 (b) DB query + native TreeSHAP",
                            "governance": "research-mode; no DB write; no commit (§3.1)"}
        with open(out_full, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\n  Results persisted: {out_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
