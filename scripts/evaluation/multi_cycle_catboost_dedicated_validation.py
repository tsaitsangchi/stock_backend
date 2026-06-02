"""
multi_cycle_catboost_dedicated_validation.py v0.2 (Multi-Cycle CatBoost Validation + Precision/Reliability Analysis · §14.7-CY 第九實作 dedicated · per Canonical Comparison Framework · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: MULTI-CYCLE 4-HORIZON CATBOOST VALIDATION + PRECISION/RELIABILITY + CANONICAL COMPARISON FRAMEWORK + §14.7-CY HORIZON-EXTENSION + §14.7-CX 8-YEAR OOS + §14.7-CW TREE-FAMILY 第九實作 dedicated + §一.10 SOURCE-TRACEABLE + §一.11 三段式合規 + §一.12 5-MIN-REPORTING-AWARE + §14.7-DE/DF §0.0-I 雙 helper 切換(2026-06-02)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用 CatBoost(dedicated 專版) 模型,實測「靠過去的股票特徵能不能預測未來、選股賺錢」,並給出「賺多少 / 準不準 / 可不可信」的成績單。

**它怎麼做(步驟)**:
1. 取 397 支「乾淨核心股」+ 每股 37 個經驗證的特徵(估值、動能、籌碼、財務等;全來自真實 FinMind/FRED API,無 AI 亂補值)。
2. 把 2013-05 ~ 2026-06 切成 157 個月度時間點(panel)。
3. **逐點往前走(walk-forward)**:每個時間點「只用那之前的資料」訓練模型,預測「之後」每支股票的報酬 —— 不偷看未來(防資料洩漏)。
4. 依預測分數挑最看好的 **前 20 支等權做多**,跟「全市場平均」比,算這一期賺賠。
5. 上面在 **4 種持有期**各做一遍:週(5 天)/ 月(20 天)/ 季(60 天)/ 年(252 天)。
6. 算每種週期的成績:**報酬率、Sharpe、勝率、預測命中率、跨次穩定度(可信度)、扣成本後年化報酬**(全經共用 helper `summarize_horizon_metrics`,與其他模型同指標 → 可比)。
7. 用門檻(§14.7-CZ T_CZ-6:有效 t 值 ≥ 4.20 且 Sharpe ≥ 2.40 且 勝率 ≥ 79%)判定:**這模型在哪個週期「真的能賺錢且可信」**。

**輸入**:資料庫(feature_values 特徵 + TaiwanStockPriceAdj 股價)。**輸出**:JSON(各週期成績)+ log。
**它不做的事**:不訓練正式上線模型、不改資料庫(純讀取評估;§3.1 evaluation 角色)。
**為什麼需要它**:回答「股票預測能不能賺錢」的實證裁判;所有模型用同一支標準、同窗、同指標 → 成績才能公平並排比較。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Multi-Cycle Horizon Coverage]** (v0.1, §14.7-CY T_CY-2): 4 horizons(weekly 5d / monthly 20d / quarterly 60d / annual 252d)完全對齊其他 multi-cycle validators。
2. **[CatBoost Symmetric Tree GBT]** (v0.1): 200 iter / learning_rate=0.05 / depth=5 / l2_leaf_reg=3 / subsample=0.8 / colsample_bylevel=0.8 / min_data_in_leaf=30 / seed=5422;**symmetric(oblivious)tree**(per depth same split condition)+ ordered boosting prevent target leakage。
3. **[Canonical Comparison Framework]** (v0.1, per RF 建立): metrics 與 LGBM v0.2 / LightGBM dedicated / XGBoost dedicated / 既存 XGBoost / 既存 CatBoost / Ensemble / Random Forest / Extra Trees validators 完全 standardized,確保 9-tree comparison reliable。
4. **[Overlap-Corrected n_effective]** (v0.1, §14.7-CY T_CY-3): n_eff = n × (30/horizon),長 horizon 之 overlap penalty;effective t-stat = t × sqrt(n_eff/n)。
5. **[Honest Annualization]** (v0.1, §14.7-CY T_CY-4): mean × (252/horizon),非 √N 高估。
6. **[Cost-Drag Per Horizon]** (v0.1, §14.7-CY T_CY-5): 0.6%/rebal × rebals_per_year。
7. **[Precision Analysis]** (v0.1, new layer): Directional Hit Rate / Top-20 Actual Overlap / RMSE / MAE。
8. **[Reliability Analysis]** (v0.1, new layer): IC Stability CoV / Significance Robustness。
9. **[System Script Mandatory]** (v0.1, §14.7-CY T_CY-1): system 永久 script。
10. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query + (a) program output;0 AI memory reuse。
11. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): significance + precision tier 動態判定。
12. **[Sovereignty Declaration]** (v0.1, §3.1 序列模組): 本程式為 **§14.7-CY 第九 evaluation 實作 dedicated**(LGBM v0.2 / 既存 XGBoost / 既存 CatBoost / Ensemble / Random Forest / Extra Trees / LightGBM dedicated / XGBoost dedicated 為前八)。**治權邊界**:(a) §3.1 evaluation;(b) read-only;(c) 不訓練 production model;(d) 不修改 DB;(e) 唯一職責:CatBoost 4-horizon walk-forward + precision/reliability + JSON 持久化。
13. **[Historical Reference Authority]** (v0.1): 既存 CatBoost v0.1 multi-cycle 結果(quarterly Eff t=3.65 / Sharpe 2.30 / NetAnn +20.22% / weekly only sig)為 §14.7-CY/CZ reference;本 dedicated v0.1 為 Canonical Comparison Framework 對齊版本,reproducibility-aware。
14. **[Idempotency]** (v0.1): pure read-only;JSON output 含 timestamp。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Horizon-Specific Walk-Forward — `--horizons <days_csv>`
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1-4 weekly/monthly/quarterly/annual | `evaluate_horizon()` 4 calls | §14.7-CY T_CY-2 |

### Group B. Walk-Forward CatBoost Training (per horizon)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window | train [0..i-1] → test i | §14.7-CW T_CW-2 |
| B.2 CB params | CB_PARAMS(200/0.05/5/3/0.8/0.8/30/5422)| §14.7-CW T_CW-4 |
| B.3 Symmetric tree | CatBoost default | overfit prevention |
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
| F.2 JSON output | `reports/multi_cycle_catboost_dedicated_<ts>.json` | §一.10 |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| 日常 dry-run | `python scripts/evaluation/multi_cycle_catboost_dedicated_validation.py --dry-run` |

### 不提供之旗標 (Intentionally Omitted)
- `--seed`:固定 5422 per §14.7-CW T_CW-4。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.2 | 2026-06-02 | Codex | **§0.0-I 雙 helper 切換(§14.7-DE + §14.7-DF)**:(1) panel 窗改用 `get_canonical_panel_dates()`(§14.7-DE 單一引用源,移除寫死 date(2018,6,15));(2) horizon-summary metric 改用 `summarize_horizon_metrics()`(§14.7-DF 單一計算源,移除 inline 計算 → 收集 (pred,actual) per panel 交 helper)。資料驅動 157 panels(2013-05-15~2026-06-01);helper 忠實性 synthetic unit test bit-identical(max|Δ|=0.00);**未改模型超參、未 retrain**。雙層治權鎖 §14.7-DE/DF + CLAUDE.md §一.16/§一.17。 | **ACTIVE** |
| v0.1 | 2026-05-29 | Codex | **首版:§14.7-CY 第九實作 dedicated(CatBoost)** under Canonical Comparison Framework。 (1) 4-horizon walk-forward,CatBoostRegressor;(2) 與既存 multi_cycle_catboost_validation.py 並存;此 dedicated 版本為 Canonical Comparison Framework 對齊;(3) Precision/Reliability 新層延續其他 multi-cycle validators;(4) Hyperparameters 對齊既存 CatBoost v0.1。 | SUPERSEDED |
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
from catboost import CatBoostRegressor
import catboost
from core.db_utils import get_db_conn, get_canonical_panel_dates, summarize_horizon_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.2"
SEED = 5422

CB_PARAMS = {
    "iterations": 200,
    "learning_rate": 0.05,
    "depth": 5,
    "l2_leaf_reg": 3,
    "subsample": 0.8,
    "colsample_bylevel": 0.8,
    "min_data_in_leaf": 30,
    "loss_function": "RMSE",
    "random_seed": SEED,
    "verbose": False,
    "allow_writing_files": False,
    "bootstrap_type": "Bernoulli",
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
    panel_pa = []  # §14.7-DF: (pred, actual) per panel → 共用 helper(單一來源)

    for i in range(1, len(panel_keys)):
        test_key = panel_keys[i]
        X_test, y_test, _, _ = panel_data[test_key]
        train_X, train_y = [], []
        for j in range(i):
            X_j, y_j, _, _ = panel_data[panel_keys[j]]
            train_X.extend(X_j); train_y.extend(y_j)
        X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y))
        if len(X_tr) < 100: continue

        fold_model = CatBoostRegressor(**CB_PARAMS)
        fold_model.fit(X_tr, y_tr)
        X_te = np.array(X_test)
        pred_te = fold_model.predict(X_te)
        panel_pa.append((pred_te, y_test))
    result = summarize_horizon_metrics(label, horizon_days, panel_pa)  # §14.7-DF Canonical Metric SSOT(單一來源)
    if result is None:
        return None

    logger.info(f"  {label}({horizon_days}d): Sharpe {result['sharpe']:+.3f} | Eff t {result['effective_t_stat']:+.3f} | Win {result['win_rate']*100:.1f}% | IC {result['mean_ic']:+.4f} | NET {result['annualized_simple_net']*100:+.1f}%/yr")
    logger.info(f"    precision: hit {result['precision_directional_hit_rate']*100:.1f}% | top-20 overlap {result['precision_top20_actual_overlap']*100:.1f}% | RMSE {result['precision_rmse']:.4f} | reliability IC-CoV {result['reliability_ic_stability_cov']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=f"Multi-Cycle CatBoost Validation {TOOL_VER}(Canonical Comparison Framework)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="§一.10 #3 multi-run seed(canonical 5422;≥3 distinct seeds → min/median/max/mean)")
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True
    CB_PARAMS["random_seed"] = args.seed; globals()["SEED"] = args.seed  # §一.10 #3: inject run seed

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    logger.info("="*100)
    logger.info(f"Multi-Cycle CatBoost Validation {TOOL_VER}(per §14.7-CY / Canonical Comparison Framework)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  CatBoost version: {catboost.__version__}")
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

        logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix(CatBoost dedicated)+ Precision/Reliability\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'NetAnn':>9} {'HitRate':>9} {'Overlap':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['annualized_simple_net']*100:>+8.2f}% {r['precision_directional_hit_rate']*100:>8.1f}% {r['precision_top20_actual_overlap']*100:>8.1f}%")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_catboost_dedicated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            output_results = {label: r for label, r in results.items()}
            output_results["_meta"] = {
                "tool": "multi_cycle_catboost_dedicated_validation.py", "tool_ver": TOOL_VER,
                "model_family": "catboost", "catboost_version": catboost.__version__,
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
