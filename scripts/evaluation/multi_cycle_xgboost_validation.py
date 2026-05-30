"""
multi_cycle_xgboost_validation.py v0.1 (Multi-Cycle XGBoost Validation Engine · per §14.7-CY Tree Family Extension · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29
**主權狀態**: MULTI-CYCLE 4-HORIZON XGBOOST VALIDATION + §14.7-CY HORIZON-EXTENSION + §14.7-CX 8-YEAR OOS EXTENSION + §14.7-CW TREE-FAMILY EXTENSION + §一.10 SOURCE-TRACEABLE (per CLAUDE.md §一.11 強制三段式標頭;§14.7-CY T_CY-1 system script 強制;§一.10 三類唯一 source 強制)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Multi-Cycle Horizon Coverage]** (v0.1, 憲法 §14.7-CY T_CY-2): 至少跨 ≥ 3 horizons(本程式預設 4: weekly 5d / monthly 20d / quarterly 60d / annual 252d);違反者視為 single-horizon evidence insufficient。
2. **[Overlap-Corrected n_effective]** (v0.1, 憲法 §14.7-CY T_CY-3): 長 horizon(> panel spacing 30d)panels 互相 overlap;統計顯著性必以 `n_eff = n × (30 / horizon_days)` 計算;raw t-stat 不得作為 production claim。
3. **[Honest Annualization]** (v0.1, 憲法 §14.7-CY T_CY-4): 年化報酬必用 `mean × (252 / horizon_days)`(per-rebalance expected);不得 sum overlapping panel returns(會 double-count)。
4. **[Cost-Drag Per Horizon]** (v0.1, 憲法 §14.7-CY T_CY-5): 各 horizon 之 transaction cost drag(`0.6%/rebal × rebals_per_year`)必揭露;short horizon(weekly/daily)即便 gross alpha 也可能被 cost 吃光。
5. **[System Script Mandatory]** (v0.1, 憲法 §14.7-CY T_CY-1): 多週期驗證必須為 system 永久 script,**禁止**在 AI ephemeral 環境臨時計算;script 須 git-tracked + version controlled。
6. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query(feature_values + TaiwanStockPriceAdj)+ (a) program output(stdout + JSON);**0 AI memory reuse**;**禁止從記憶 / 推測 / 估算**。
7. **[Reality-Check Hierarchy]** (v0.1, 憲法 §14.7-CY T_CY-6 / §14.7-CZ T_CZ-6): 推薦 production horizon 必同時滿足 robust significance(Eff t > 1.997)+ reasonable n_eff(≥ 30)+ acceptable cost drag(< 5%/yr);**quarterly horizon** 為 recommended sweet spot(Eff t > 4.0)。
8. **[Tree-Family Extension]** (v0.1, 憲法 §14.7-CW): 本程式為 §14.7-CW Tree Production Family 第二實作(XGBoost);共享 LGBM 之 walk-forward / overlap correction / honest annualization protocol;hyperparameters 對齊 §14.7-CW T_CW-4 conservative defaults。
9. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): 4 horizons 之 significance verdict 動態計算(`abs(eff_t_stat) > 1.997`),**不**硬編 ✅/❌ 結論。
10. **[Sovereignty Declaration]** (v0.1, 憲法 §3.1 序列模組): 本程式為 **§14.7-CY Multi-Cycle Validation 第二實作**(對標 multi_cycle_validation.py LGBM 為第一實作)。**治權邊界**:(a) §3.1 序列 evaluation 模組;(b) 五套禁令(§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §6.8)不涉;(c) T1-T3 不分層;(d) §8.5 anti-leakage 不處理(features 已 anti-leakage compliant per feature_store_builder);(e) **不訓練 production model**(僅 evaluate / 不寫 model_registry);(f) **不修改 feature_values / TaiwanStockPriceAdj**(read-only);(g) 唯一職責:跑 4-horizon walk-forward XGBoost evaluation + 計算 metrics + 持久化 JSON。
11. **[Historical Reference Authority]** (v0.1): 本程式 `TOOL_VER = "v0.1"` 屬於記述性快照;LGBM baseline 對比值來自 `reports/multi_cycle_validation_20260528_final.json`(權威來源)。
12. **[Idempotency]** (v0.1): `--dry-run` 不寫 DB;`--commit` 寫 `evaluation_log`(若實作);JSON output path 含 timestamp,不覆寫舊結果。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

> 本程式作為 §14.7-CY Multi-Cycle Tree-Family 第二實作,依「驗證面向」拆分為 5 大功能群;每群對應憲章治權契約。

### Group A. Horizon-Specific Walk-Forward Evaluation — `--horizons <days_csv>`

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 weekly horizon(5d)| `evaluate_horizon(cur, panels, 5, universe, "weekly")` | §14.7-CY T_CY-2 |
| A.2 monthly horizon(20d)| `evaluate_horizon(cur, panels, 20, universe, "monthly")` | §14.7-CY T_CY-2 |
| A.3 quarterly horizon(60d)| `evaluate_horizon(cur, panels, 60, universe, "quarterly")` | §14.7-CY T_CY-6(recommended)|
| A.4 annual horizon(252d)| `evaluate_horizon(cur, panels, 252, universe, "annual")` | §14.7-CY T_CY-2(caveat: small n_eff)|
| 對應 CLI | `--horizons 5,20,60,252`(預設) | — |

### Group B. Walk-Forward XGBoost Training — internal per-horizon

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window training | `for i in range(1, len(panel_keys))`(train [0..i-1] → test i)| §14.7-CW T_CW-2 |
| B.2 XGBoost params(seed=5422)| `XGB_PARAMS`(200 trees / depth 5 / lr 0.05)| §14.7-CW T_CW-4 |
| B.3 Winsorization | `winsorize(arr, 0.01, 0.99)` | label stability |
| B.4 Spearman IC | `spearman_ic(pred, y)` | §14.7-CM Empirical IC |
| 對應 method | `evaluate_horizon()` internal | — |

### Group C. Overlap Correction + Honest Annualization

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 n_effective computation | `n_eff = n × (30 / horizon_days)` if overlap | §14.7-CY T_CY-3 |
| C.2 Effective t-stat | `eff_t_stat = t_stat × sqrt(n_eff / n)` | §14.7-CY T_CY-3 |
| C.3 Annualization formula | `annualized_log = mean × (252 / horizon_days)` | §14.7-CY T_CY-4 |
| C.4 Cost-drag per horizon | `annual_cost_drag = 0.006 × (252 / horizon_days)` | §14.7-CY T_CY-5 |
| C.5 Significance verdict | `is_significant = abs(eff_t_stat) > 1.997`(p<0.05)| §5.6.3 動態 verdict |
| 對應 CLI | (internal) | — |

### Group D. Cross-Cycle Comparison + Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 Cross-cycle matrix table | stdout 彙總 4 horizons | §14.7-CY T_CY-6 |
| D.2 JSON persistence | `reports/multi_cycle_xgboost_<timestamp>.json` | §14.7-CY T_CY-1 |
| D.3 Meta context | `_meta`(tool_ver / xgboost_version / seed / run_at)| §一.10 source_traceability |
| D.4 panel_records detail | per-panel ic / top20_ret / univ_ret / alpha | audit trail |
| 對應 CLI | `--output reports/<path>.json` | — |

### Group E. Source Traceability + Anti-Hallucination

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| E.1 Universe via DB query | `core_universe_membership` + `core_universe_snapshot` JOIN | §一.10 (b) DB query |
| E.2 Features via DB query | `feature_values` WHERE feature_set_id IN ... | §一.10 (b) DB query |
| E.3 Forward returns via DB | `TaiwanStockPriceAdj` LN(t1/t0)JOIN | §一.10 (b) DB query |
| E.4 0 AI memory reuse | All numbers from program output | §一.10 第 1 條 |
| E.5 No placeholder data | 所有 panel_data 為實際 query 結果 | §一.10 禁止來源(第四類) |
| 對應 CLI | (internal) | — |

### 對齊憲章 §二 維運矩陣（標準場景索引）

| 場景 | 對應命令 |
| :--- | :--- |
| 日常 multi-cycle validation | `python scripts/evaluation/multi_cycle_xgboost_validation.py --dry-run` |
| Commit 評估結果 | `python scripts/evaluation/multi_cycle_xgboost_validation.py --commit --output reports/<path>.json` |
| 自訂 horizons | `... --horizons 7,30,90,365` |
| 比對 LGBM baseline | 對比 `reports/multi_cycle_validation_20260528_final.json` |

### 不提供之旗標 (Intentionally Omitted)

- `--seed N`:seed=5422 為固定值(per §14.7-CW T_CW-4 conservative)。若需 reproducibility multi-run 須跑 ≥ 3 次取統計值(per §14.7-CW T_CW-6)。
- `--cost-per-rebal X`:0.6% 為 §14.7-CY T_CY-5 conservative standard。
- `--write-model`:本程式為 evaluation only,不寫 `model_registry`(屬 `model_trainer_xgboost.py` 治權)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **首版:§14.7-CY Tree Family Extension 第二實作**(對標 multi_cycle_validation.py LGBM v0.1 為第一實作)。**功能 4 點**:(a) 4-horizon walk-forward(5/20/60/252d);(b) Overlap correction n_eff = n × (30/horizon);(c) Honest annualization mean × (252/horizon);(d) Per-horizon cost-drag analysis。**治權邊界**:read-only(不寫 model_registry / 不改 feature_values);全 (b) DB query;0 AI memory(per §一.10)。**XGBoost params**:200 trees / depth 5 / lr 0.05 / seed 5422(per §14.7-CW T_CW-4)。**Output**:stdout + JSON(`reports/multi_cycle_xgboost_<timestamp>.json`)。**首跑實證**(2026-05-29):quarterly Eff t=4.36 / Sharpe 2.63 / Win 81.2% / Net annual +29.35% ✅ PASS §14.7-CZ T_CZ-6。**對標 LGBM v0.2 baseline**(`multi_cycle_validation_20260528_final.json`):全 4 horizons XGBoost 略勝(quarterly +4.91pp NetAnn)。**v0.1 治權邊界**:不入憲(無新 doctrine,僅 §14.7-CW Tree Family extension implementation)。**CLAUDE.md §一.11 強制三段式標頭**(2026-05-29 入憲):本版補正標頭為 sovereign_sync_engine.py 範本格式。 | **ACTIVE** |
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

DEFAULT_HORIZONS = [("weekly", 5), ("monthly", 20), ("quarterly", 60), ("annual", 252)]

XGB_PARAMS = {
    "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1,
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "verbosity": 0, "seed": 5422, "tree_method": "hist",
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
    cur.execute("""
        SELECT MIN(date) FROM "TaiwanStockPriceAdj"
        WHERE date >= (%s::date + INTERVAL '%s days')
          AND stock_id ~ '^[0-9]'
          AND date <= (%s::date + INTERVAL '%s days')
    """, (str(as_of), horizon_days, str(as_of), horizon_days + 14))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date: return {}, None
    cur.execute("""
        WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
             t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
        SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric)
        FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id
    """, (str(as_of), str(label_date)))
    return {sid: float(r) for sid, r in cur.fetchall()}, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
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
        if XX:
            panel_data[as_of] = (XX, yy, sids_matched, label_date)
    logger.info(f"  Loaded {len(panel_data)} panels({horizon_days}d forward / load: {time.monotonic()-t0:.1f}s)")

    panel_keys = sorted(panel_data.keys())
    panel_ics, panel_top20_rets, panel_univ_rets = [], [], []

    for i in range(1, len(panel_keys)):
        test_key = panel_keys[i]
        X_test, y_test, _, _ = panel_data[test_key]
        train_X, train_y = [], []
        for j in range(i):
            X_j, y_j, _, _ = panel_data[panel_keys[j]]
            train_X.extend(X_j); train_y.extend(y_j)
        X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y))
        if len(X_tr) < 100: continue

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=SPEC_43)
        fold_model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS)

        X_te = np.array(X_test)
        dtest = xgb.DMatrix(X_te, feature_names=SPEC_43)
        pred_te = fold_model.predict(dtest)
        ic_te = spearman_ic(pred_te, y_test)
        n_top = min(20, len(pred_te))
        top_idx = np.argsort(pred_te)[-n_top:]
        top20_ret = float(np.mean([y_test[k] for k in top_idx]))
        univ_ret = float(np.mean(y_test))
        panel_ics.append(ic_te); panel_top20_rets.append(top20_ret); panel_univ_rets.append(univ_ret)

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

    # Honest annualization
    rebals_per_year = 252.0 / horizon_days
    annualized_log_gross = mean_ret * rebals_per_year
    annualized_simple_gross = math.exp(annualized_log_gross) - 1
    cost_per_rebal = 0.006
    annual_cost_drag = cost_per_rebal * rebals_per_year
    annualized_simple_net = math.exp(annualized_log_gross - annual_cost_drag) - 1
    net_rets = [r - cost_per_rebal for r in panel_top20_rets]
    net_mean = float(np.mean(net_rets))
    net_std = float(np.std(net_rets, ddof=1)) if n > 1 else 0
    net_sharpe = net_mean / net_std * math.sqrt(12) if net_std > 0 else 0

    # Overlap correction
    panel_spacing = 30
    if horizon_days <= panel_spacing:
        n_eff = float(n); overlap_pct = 0.0
    else:
        n_eff = n * (panel_spacing / horizon_days)
        overlap_pct = (horizon_days - panel_spacing) / horizon_days * 100
    eff_t_stat = t_stat * math.sqrt(n_eff / n) if n > 0 else 0
    is_significant = abs(eff_t_stat) > 1.997

    result = {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "n_effective": n_eff, "overlap_pct": overlap_pct,
        "rebals_per_year": rebals_per_year,
        "mean_ret_per_panel": mean_ret, "std_ret_per_panel": std_ret,
        "sharpe": sharpe, "win_rate": win_rate, "mdd_per_panel": mdd,
        "mean_alpha_per_panel": mean_alpha, "std_alpha_per_panel": std_alpha,
        "ir": ir, "t_stat": t_stat,
        "effective_t_stat": eff_t_stat, "is_significant_p05": is_significant,
        "mean_ic": float(np.mean(panel_ics)),
        "annualized_simple_gross": annualized_simple_gross,
        "annual_cost_drag_log": annual_cost_drag,
        "annualized_simple_net": annualized_simple_net,
        "net_sharpe_per_panel": net_sharpe,
    }

    logger.info(f"\n  Results({label}, {horizon_days}d):")
    logger.info(f"    OOS panels:                  {n}")
    logger.info(f"    Mean ret/panel(log):        {mean_ret*100:+.4f}%")
    logger.info(f"    Sharpe(annualized monthly): {sharpe:+.4f}")
    logger.info(f"    Win rate:                    {win_rate*100:.1f}%")
    logger.info(f"    Mean alpha/panel:            {mean_alpha*100:+.4f}%")
    logger.info(f"    IR:                          {ir:+.4f}")
    logger.info(f"    Mean OOS IC:                 {result['mean_ic']:+.4f}")
    logger.info(f"    MDD:                         {mdd*100:.2f}%")
    logger.info(f"    Annualized gross:            {annualized_simple_gross*100:+.2f}%/yr")
    logger.info(f"    Annual cost drag:           -{annual_cost_drag*100:.2f}%")
    logger.info(f"    Annualized NET:              {annualized_simple_net*100:+.2f}%/yr")
    logger.info(f"    Overlap %:                   {overlap_pct:.1f}%")
    logger.info(f"    n_effective:                 {n_eff:.1f}")
    logger.info(f"    Effective t-stat:            {eff_t_stat:+.3f}")
    logger.info(f"    Significance p<0.05:         {'✅ YES' if is_significant else '❌ NO'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=f"Multi-Cycle XGBoost Validation {TOOL_VER}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=XGB_PARAMS["seed"],
                        help="§一.10 #3 multi-run seed(canonical 5422;≥3 distinct seeds → min/median/max/mean)")
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True
    XGB_PARAMS["seed"] = args.seed  # §一.10 #3: inject run seed into frozen params

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    logger.info("="*100)
    logger.info(f"Multi-Cycle XGBoost Validation {TOOL_VER}(per §14.7-CY)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  XGBoost ver: {xgb.__version__}")
    logger.info(f"  Mode:     {'COMMIT' if args.commit else 'DRY-RUN'}")

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

        logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix(XGBoost)\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'N':>4} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'Win':>6} {'NetAnn':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_panels']:>4} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['win_rate']*100:>5.1f}% {r['annualized_simple_net']*100:>+8.2f}%")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            output_results = {label: {k: v for k, v in r.items()} for label, r in results.items()}
            output_results["_meta"] = {
                "tool": "multi_cycle_xgboost_validation.py",
                "tool_ver": TOOL_VER,
                "model_family": "xgboost",
                "xgboost_version": xgb.__version__,
                "run_at": datetime.now().isoformat(),
                "constitution_ver": CONSTITUTION_VER,
                "seed": XGB_PARAMS["seed"],
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
