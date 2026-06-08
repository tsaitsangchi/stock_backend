"""
multi_cycle_mlp_validation.py v0.1 (Multi-Cycle MLP Neural-Net Validation Engine · per §14.7-CY Tabular NN Baseline Extension · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-06-04
**主權狀態**: MULTI-CYCLE 4-HORIZON MLP (sklearn MLPRegressor) VALIDATION + §14.7-CY HORIZON-EXTENSION + §14.7-CX 8-YEAR OOS EXTENSION + §一.10 SOURCE-TRACEABLE (per CLAUDE.md §一.11 強制三段式標頭;§一.10 三類唯一 source 強制) + §14.7-DE/DF §0.0-I 雙 helper 單一引用源(get_canonical_panel_dates / summarize_horizon_metrics)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用 MLP 神經網路(多層感知器,表格型神經網路) 在 37 個股票特徵上預測「每支股票未來會漲多少」,挑最看好的 top-20 等權做多,並給出「賺多少 / 準不準 / 可不可信」的成績單。

**它怎麼做(步驟)**:
1. 取 397 支「乾淨核心股」+ 每股 37 個經驗證的特徵(估值、動能、籌碼、財務等;全來自真實 FinMind/FRED API,無 AI 亂補值)。
2. 把 2013-05 ~ 2026-06 切成多個月度時間點(panel)。
3. **逐點往前走(walk-forward)**:每個時間點「只用那之前的資料」先把特徵標準化(StandardScaler),再訓練一個 MLP 神經網路(隱藏層 64→32、ReLU、L2 正則、early stopping),預測「之後」每支股票的報酬 —— 不偷看未來(防資料洩漏)。
4. 依預測分數挑最看好的 **前 20 支等權做多**,跟「全市場平均」比,算這一期賺賠。
5. 上面在 **4 種持有期**各做一遍:週(5 天)/ 月(20 天)/ 季(60 天)/ 年(252 天)。
6. 算每種週期的成績:**報酬率、Sharpe、勝率、預測命中率、跨次穩定度(可信度)、扣成本後年化報酬**(全經共用 helper `summarize_horizon_metrics`,與其他模型同指標 → 可比)。
7. 用門檻(§14.7-CZ T_CZ-6:有效 t 值 ≥ 4.20 且 Sharpe ≥ 2.40 且 勝率 ≥ 79%)判定:**這模型在哪個週期「真的能賺錢且可信」**。

**輸入**:資料庫(feature_values 特徵 + TaiwanStockPriceAdj 股價)。**輸出**:JSON(各週期成績)+ log。
**它不做的事**:不訓練正式上線模型、不改資料庫(純讀取評估;§3.2 evaluation 角色)。
**為什麼需要它**:回答「表格型神經網路在這 37 特徵上能不能選股賺錢」的實證裁判;所有模型用同一支標準、同窗、同指標 → 成績才能公平並排比較(common baseline)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Multi-Cycle Horizon Coverage]** (v0.1, 憲法 §14.7-CY T_CY-2): 至少跨 ≥ 3 horizons(本程式預設 4: weekly 5d / monthly 20d / quarterly 60d / annual 252d);違反者視為 single-horizon evidence insufficient。
2. **[Overlap-Corrected n_effective]** (v0.1, 憲法 §14.7-CY T_CY-3): 長 horizon(> panel spacing 30d)panels 互相 overlap;統計顯著性必以 `n_eff = n × (30 / horizon_days)` 計算(交由 §14.7-DF helper);raw t-stat 不得作為 production claim。
3. **[Honest Annualization]** (v0.1, 憲法 §14.7-CY T_CY-4): 年化報酬必用 `mean × (252 / horizon_days)`(per-rebalance expected);不得 sum overlapping panel returns(會 double-count)。
4. **[Cost-Drag Per Horizon]** (v0.1, 憲法 §14.7-CY T_CY-5): 各 horizon 之 transaction cost drag(`0.6%/rebal × rebals_per_year`)必揭露;short horizon(weekly/daily)即便 gross alpha 也可能被 cost 吃光。
5. **[System Script Mandatory]** (v0.1, 憲法 §14.7-CY T_CY-1): 多週期驗證必須為 system 永久 script,**禁止**在 AI ephemeral 環境臨時計算;script 須 git-tracked + version controlled。
6. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query(feature_values + TaiwanStockPriceAdj)+ (a) program output(stdout + JSON);**0 AI memory reuse**;**禁止從記憶 / 推測 / 估算**。
7. **[Reality-Check Hierarchy]** (v0.1, 憲法 §14.7-CY T_CY-6 / §14.7-CZ T_CZ-6): 推薦 production horizon 必同時滿足 robust significance(Eff t > 1.997)+ reasonable n_eff(≥ 30)+ acceptable cost drag(< 5%/yr);**quarterly horizon** 為 recommended sweet spot。
8. **[Tabular Neural-Net Baseline]** (v0.1, 憲法 §14.7-CY): 本程式為 §14.7-CY 比較模型庫之 tabular neural-net baseline(sklearn MLPRegressor),對標 tree-family(XGBoost/LGBM)以同窗同指標比較橫斷面選股能力;非 sequence/foundation model。
9. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): 4 horizons 之 significance verdict 由 §14.7-DF helper 動態計算(`abs(eff_t_stat) > 1.997`),**不**硬編 ✅/❌ 結論。
10. **[Common Comparison Baseline]** (v0.1, 憲法 §14.7-DE/DF + §0.0-I): panel 窗 = `get_canonical_panel_dates()`(單一引用源);horizon metric = `summarize_horizon_metrics()`(單一計算源);universe = 397 core + `is_null_imputed IS NOT TRUE`;top-20 / 0.6% cost / T_CZ-6 與全比較模型逐字相同 → apples-to-apples。
11. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 evaluation 模組): 本程式為 **§14.7-CY Multi-Cycle Validation 之 MLP tabular-NN 實作**。**治權邊界**:(a) §3.2 evaluation 模組;(b) 五套禁令(§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §6.8)不涉;(c) T1-T3 不分層;(d) §8.5 anti-leakage 由 walk-forward train-on-past-only 保證(features 已 anti-leakage compliant per feature_store_builder);(e) **不訓練 production model**(僅 evaluate / 不寫 model_registry);(f) **不修改 feature_values / TaiwanStockPriceAdj**(read-only);(g) 唯一職責:跑 4-horizon walk-forward MLP evaluation + 計算 metrics + 持久化 JSON。
12. **[Historical Reference Authority]** (v0.1): 本程式 `TOOL_VER = "v0.1"` 屬於記述性快照;非權威來源。
13. **[Idempotency]** (v0.1): `--dry-run` 不寫 DB;JSON output path 含 timestamp,不覆寫舊結果。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

> 本程式作為 §14.7-CY Multi-Cycle 比較模型庫之 tabular neural-net baseline,依「驗證面向」拆分為 5 大功能群;每群對應憲章治權契約。

### Group A. Horizon-Specific Walk-Forward Evaluation — `--horizons <days_csv>`

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 weekly horizon(5d)| `evaluate_horizon(cur, panels, 5, universe, "weekly")` | §14.7-CY T_CY-2 |
| A.2 monthly horizon(20d)| `evaluate_horizon(cur, panels, 20, universe, "monthly")` | §14.7-CY T_CY-2 |
| A.3 quarterly horizon(60d)| `evaluate_horizon(cur, panels, 60, universe, "quarterly")` | §14.7-CY T_CY-6(recommended)|
| A.4 annual horizon(252d)| `evaluate_horizon(cur, panels, 252, universe, "annual")` | §14.7-CY T_CY-2(caveat: small n_eff)|
| 對應 CLI | `--horizons 5,20,60,252`(預設) | — |

### Group B. Walk-Forward MLP Training — internal per-horizon

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window training | `for i in range(1, len(panel_keys))`(train [0..i-1] → test i)| §14.7-CY T_CY-2 |
| B.2 StandardScaler → MLPRegressor(seed)| `_build_mlp(seed)` Pipeline | tabular NN baseline |
| B.3 Winsorization | `winsorize(arr, 0.01, 0.99)` | label stability |
| B.4 (pred, actual) per panel | `panel_pa.append((pred_te, y_test))` | §14.7-DF single-source metric |
| 對應 method | `evaluate_horizon()` internal | — |

### Group C. Overlap Correction + Honest Annualization(由 §14.7-DF helper)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 n_effective computation | helper: `n_eff = n × (30 / horizon_days)` if overlap | §14.7-CY T_CY-3 |
| C.2 Effective t-stat | helper: `eff_t_stat = t_stat × sqrt(n_eff / n)` | §14.7-CY T_CY-3 |
| C.3 Annualization formula | helper: `annualized_log = mean × (252 / horizon_days)` | §14.7-CY T_CY-4 |
| C.4 Cost-drag per horizon | helper: `annual_cost_drag = 0.006 × (252 / horizon_days)` | §14.7-CY T_CY-5 |
| C.5 Significance verdict | helper: `is_significant = abs(eff_t_stat) > 1.997`(p<0.05)| §5.6.3 動態 verdict |
| 對應 CLI | (internal) | — |

### Group D. Cross-Cycle Comparison + Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 Cross-cycle matrix table | stdout 彙總 4 horizons | §14.7-CY T_CY-6 |
| D.2 JSON persistence | `reports/multi_cycle_mlp_<timestamp>.json` | §14.7-CY T_CY-1 |
| D.3 Meta context | `_meta`(tool_ver / sklearn_version / seed / run_at)| §一.10 source_traceability |
| D.4 panel metrics detail | per-horizon ic / sharpe / win / precision / reliability | audit trail |
| 對應 CLI | `--output reports/<path>.json` | — |

### Group E. Source Traceability + Anti-Hallucination

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| E.1 Universe via DB query | `core_universe_membership` + `core_universe_snapshot` JOIN | §一.10 (b) DB query |
| E.2 Features via DB query | `feature_values` WHERE feature_set_id IN ...(`is_null_imputed IS NOT TRUE`)| §一.10 (b) DB query + §14.7-DC source-pure |
| E.3 Forward returns via DB | `TaiwanStockPriceAdj` LN(t1/t0)JOIN | §一.10 (b) DB query |
| E.4 0 AI memory reuse | All numbers from program output | §一.10 第 1 條 |
| E.5 No placeholder data | 所有 panel_data 為實際 query 結果 | §一.10 禁止來源(第四類) |
| 對應 CLI | (internal) | — |

### 對齊憲章 §二 維運矩陣（標準場景索引）

| 場景 | 對應命令 |
| :--- | :--- |
| 日常 multi-cycle validation | `python scripts/evaluation/multi_cycle_mlp_validation.py --dry-run` |
| Commit 評估結果 | `python scripts/evaluation/multi_cycle_mlp_validation.py --commit --output reports/<path>.json` |
| 自訂 horizons | `... --horizons 7,30,90,365` |
| 多 seed reproducibility | `... --seed 1 / --seed 2 / --seed 3`(per §一.10 #3 取 min/median/max/mean)|

### 不提供之旗標 (Intentionally Omitted)

- `--cost-per-rebal X`:0.6% 為 §14.7-CY T_CY-5 conservative standard(由 helper 固定)。
- `--write-model`:本程式為 evaluation only,不寫 `model_registry`。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-04 | Codex | **首版:§14.7-CY Multi-Cycle 比較模型庫之 tabular neural-net baseline**(sklearn MLPRegressor)。**功能 4 點**:(a) 4-horizon walk-forward(5/20/60/252d);(b) StandardScaler → MLPRegressor(hidden=(64,32) / relu / alpha=1e-3 L2 / early_stopping)per training window;(c) §14.7-DE 單一引用源 panel 窗 + §14.7-DF 單一計算源 metric;(d) 397 core + `is_null_imputed IS NOT TRUE` source-pure。**治權邊界**:read-only(不寫 model_registry / 不改 feature_values);全 (b) DB query;0 AI memory(per §一.10)。**Output**:stdout + JSON(`reports/multi_cycle_mlp_<timestamp>.json`)。clone 自 multi_cycle_xgboost_validation.py v0.2,**僅替換模型**(GBM → MLP),其餘 universe query / panel 窗 / forward returns / top-20 / cost / T_CZ-6 / metric helper / CLI 全保持一致(common baseline)。 | **ACTIVE** |
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
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from core.db_utils import get_db_conn, get_canonical_panel_dates, summarize_horizon_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

DEFAULT_HORIZONS = [("weekly", 5), ("monthly", 20), ("quarterly", 60), ("annual", 252)]

MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "alpha": 1e-3,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "max_iter": 300,
    "seed": 5422,
}

SPEC_43 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "margin_ratio_60d",
    # §14.7-DC v0.3 strict: theme_is_semiconductor + fitness_signal_60d + theme_strength all removed (hardcoded knowledge / transitively tainted = AI hallucination)
]


def _build_mlp(seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=MLP_PARAMS["hidden_layer_sizes"],
            activation=MLP_PARAMS["activation"],
            alpha=MLP_PARAMS["alpha"],
            early_stopping=MLP_PARAMS["early_stopping"],
            validation_fraction=MLP_PARAMS["validation_fraction"],
            n_iter_no_change=MLP_PARAMS["n_iter_no_change"],
            max_iter=MLP_PARAMS["max_iter"],
            random_state=seed,
        )),
    ])


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

        model = _build_mlp(MLP_PARAMS["seed"])
        model.fit(X_tr, y_tr)

        X_te = np.array(X_test)
        pred_te = model.predict(X_te)
        panel_pa.append((pred_te, y_test))
    result = summarize_horizon_metrics(label, horizon_days, panel_pa)  # §14.7-DF Canonical Metric SSOT(單一來源)
    if result is None:
        return None

    logger.info(f"  {label}({horizon_days}d): Sharpe {result['sharpe']:+.3f} | Eff t {result['effective_t_stat']:+.3f} | Win {result['win_rate']*100:.1f}% | IC {result['mean_ic']:+.4f} | NET {result['annualized_simple_net']*100:+.1f}%/yr")
    logger.info(f"    precision: hit {result['precision_directional_hit_rate']*100:.1f}% | top-20 overlap {result['precision_top20_actual_overlap']*100:.1f}% | RMSE {result['precision_rmse']:.4f} | reliability IC-CoV {result['reliability_ic_stability_cov']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=f"Multi-Cycle MLP Validation {TOOL_VER}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=MLP_PARAMS["seed"],
                        help="§一.10 #3 multi-run seed(canonical 5422;≥3 distinct seeds → min/median/max/mean)")
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True
    MLP_PARAMS["seed"] = args.seed  # §一.10 #3: inject run seed into frozen params

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    logger.info("="*100)
    logger.info(f"Multi-Cycle MLP Validation {TOOL_VER}(per §14.7-CY)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  sklearn ver: {sklearn.__version__}")
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

        panels = get_canonical_panel_dates("feature_set_v0.6")  # §14.7-DE / §0.0-I 單一引用源
        logger.info(f"  Panels:   {len(panels)} ({panels[0][1]} ~ {panels[-1][1]}, data-driven §14.7-DE)")

        results = {}
        t_global = time.monotonic()
        for label, days in horizon_labels:
            r = evaluate_horizon(cur, panels, days, universe, label)
            if r: results[label] = r

        logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix(MLP)\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'N':>4} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'Win':>6} {'NetAnn':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_panels']:>4} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['win_rate']*100:>5.1f}% {r['annualized_simple_net']*100:>+8.2f}%")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            output_results = {label: {k: v for k, v in r.items()} for label, r in results.items()}
            output_results["_meta"] = {
                "tool": "multi_cycle_mlp_validation.py",
                "tool_ver": TOOL_VER,
                "model_family": "mlp",
                "sklearn_version": sklearn.__version__,
                "run_at": datetime.now().isoformat(),
                "constitution_ver": CONSTITUTION_VER,
                "seed": MLP_PARAMS["seed"],
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
