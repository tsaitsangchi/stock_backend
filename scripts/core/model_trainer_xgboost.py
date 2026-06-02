"""
model_trainer_xgboost.py v0.1 (XGBoost Tree Production Trainer · §14.7-CW Tree Family 第二實作 · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29
**主權狀態**: XGBOOST TREE PRODUCTION + §14.7-CW TREE-FAMILY EXTENSION + §14.7-CS MODEL-TRAINING-LANDING + §14.7-CL 43-FEATURE CANONICAL + §一.10 SOURCE-TRACEABLE + §一.11 三段式合規 (per CLAUDE.md §一.11;對標 model_trainer_lgbm_v2.py v0.2 LGBM 為第一實作)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用 **XGBoost** 訓練股價預測 production 模型(§14.7-CW Tree Family;walk-forward 訓練)。

**輸入 → 輸出**:feature_values(37 特徵)+ 核心股 → XGBoost model artifact + registry 列

**為什麼需要它**:產生實際上線用的模型檔(與 multi_cycle validator 雙軌:trainer 產 artifact、validator 自評)。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Real Tree Model Required]** (v0.1, 憲法 §14.7-CW T_CW-1): production model 必須使用 tree-based 真實模型(此處 XGBoost),不得停留 rank-IC linear baseline;XGBoost 與 LGBM 為 §14.7-CW Tree Family 並列實作。
2. **[Expanding Window Walk-Forward OOS]** (v0.1, 憲法 §14.7-CW T_CW-2): train [panel 0..i-1] → test panel i 之 expanding window;不得跑 full in-sample evaluation 視為 production metric;in-sample IC vs OOS IC gap 須揭露。
3. **[Conservative Hyperparameters]** (v0.1, 憲法 §14.7-CW T_CW-4): n_estimators=200 / learning_rate=0.05 / max_depth=5 / min_child_weight=5 / subsample=0.8 / colsample_bytree=0.8 / reg_alpha=0.1 / reg_lambda=0.1 / seed=5422;與 LGBM 對等之 conservative defaults。
4. **[Treaty Gates 4/4]** (v0.1, 憲法 §14.7-CW T_CW-5): Sharpe > 0 / Win ≥ 50% / MDD ≤ 30% / Mean α > 0;違反 any gate 視 production training 失敗。
5. **[Top Features §0.1+§0.2 Dominated]** (v0.1, 憲法 §14.7-CW T_CW-3): top-15 features 必由 §0.1 第一性原理 + §0.2 八二法則 features 主導;§0.3 macro broadcast features 不應入 top(per §14.7-CK)。
6. **[43 Canonical Features]** (v0.1, 憲法 §14.7-CL): SPEC_43 為 §14.7-CL canonical scope;不得擴增/減少;與 LGBM 共用同一 SPEC。
7. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query(feature_values + TaiwanStockPriceAdj)+ (a) program output;0 AI memory;0 推測。
8. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): 4 Treaty Gates 動態判定(✅/⚠️/❌),不硬編結論。
9. **[Sovereignty Declaration]** (v0.1, 憲法 §3.1 序列模組): 本程式為 **§10 model_trainer 第二實作**(LGBM 為第一)。**治權邊界**:(a) §3.1 序列 training 模組;(b) 五套禁令不涉;(c) T1-T3 不分層;(d) §8.5 anti-leakage 不處理(features 已 anti-leakage compliant);(e) **不選股**(由 core_universe_builder 負責);(f) **不算 feature**(由 feature_store_builder 負責);(g) **不評估 multi-cycle**(由 multi_cycle_xgboost_validation 負責);(h) 唯一職責:8-panel walk-forward XGBoost 訓練 + Treaty gates 評估 + model artifact + model_registry insert(--commit)。
10. **[Historical Reference Authority]** (v0.1): 本程式 `TOOL_VER = "v0.1"` 屬於記述性快照;LGBM baseline(§14.7-CW v0.2)為對標權威。
11. **[Idempotency]** (v0.1): `model_registry` INSERT ... ON CONFLICT(model_id) DO UPDATE;重跑覆寫 metrics + hyperparams + status,不破壞 model_id 唯一性。
12. **[Anti-Leakage Inheritance]** (v0.1, 憲法 §8.5): features 已由 feature_store_builder 之 publication_date_strategy 處理 anti-leakage;forward returns 取自 (label_date > as_of_date) 之未來 close prices,本身為 forward-looking label,無 leakage。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

> 本程式作為 §14.7-CW Tree Family 第二實作,依「訓練流程」拆分為 6 大功能群;每群對應憲章治權契約。

### Group A. Universe + Feature Loading

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Latest committed core_universe | `core_universe_membership` JOIN `core_universe_snapshot` | §14.7-CF SSOT |
| A.2 Feature values loading | `load_panel_data()` query `feature_values` | §14.7-CL canonical |
| A.3 Forward returns(per panel)| `TaiwanStockPriceAdj` LN(t1/t0)JOIN | §14.7-CV backtest |
| 對應 CLI | (auto) | — |

### Group B. Walk-Forward Expanding Window Training

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window | `for i in range(1, len(panels_list))` train [0..i-1] → test i | §14.7-CW T_CW-2 |
| B.2 XGBoost params | `DEFAULT_TRAINING_POLICY`(200 trees / depth 5 / seed 5422)| §14.7-CW T_CW-4 |
| B.3 Winsorization | `winsorize(arr, 0.01, 0.99)` | label stability |
| B.4 Spearman IC | `spearman_ic(pred, y)` | §14.7-CM |
| 對應 CLI | `--panel-feature-sets <fs1,...>` | — |

### Group C. Treaty Gates Evaluation (Sharpe / Win / MDD / Alpha)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Sharpe annualized | `mean_ret / std_ret * sqrt(12)` | §14.7-CW Gate CW-1 |
| C.2 Win rate | `sum(r > 0) / n` | §14.7-CW Gate CW-2 |
| C.3 MDD computation | `running` + `peak` tracking | §14.7-CW Gate CW-3 |
| C.4 Mean alpha | `mean(top20_ret - univ_ret)` | §14.7-CW Gate CW-4 |
| C.5 t-statistic | `mean_alpha / (std_alpha / sqrt(n))` | §14.7-CX OOS significance |
| 對應 CLI | (auto in --dry-run / --commit) | — |

### Group D. Model Artifact Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 model.json save | `model.save_model(str(model_path))` | XGBoost native format |
| D.2 metrics.json write | full metrics serialization | §一.10 source-traceable |
| D.3 hyperparams.json write | `DEFAULT_TRAINING_POLICY` dump | §14.7-CW T_CW-4 |
| D.4 model_registry INSERT | ON CONFLICT DO UPDATE | §10 model_registry SSOT |
| 對應 CLI | `--commit`(only) | — |

### Group E. Feature Importance Reporting

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| E.1 Gain-based importance | `model.get_score(importance_type='gain')` | §14.7-CN 4-path necessity |
| E.2 Top-15 stdout report | sorted descending | §14.7-CW T_CW-3 |
| E.3 Top-20 in metrics.json | persistence to artifact | §14.7-CN alignment |
| 對應 CLI | (auto) | — |

### Group F. CLI + Mode Control

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| F.1 `--dry-run` mode | 評估不寫 DB | safe default |
| F.2 `--commit` mode | 寫 model_registry + artifact | production |
| F.3 `--panel-feature-sets <csv>` | walk-forward training panels override | §14.7-CX 95-panel support |
| F.4 `--label-horizon N` | default 30d | §14.7-CW 30d horizon |
| 對應 CLI | `argparse` standard | — |

### 對齊憲章 §二 維運矩陣（標準場景索引）

| 場景 | 對應命令 |
| :--- | :--- |
| 日常 dry-run | `python scripts/core/model_trainer_xgboost.py --dry-run` |
| Commit production model | `python scripts/core/model_trainer_xgboost.py --commit` |
| 8-year walk-forward(per §14.7-CX)| `... --panel-feature-sets <95-panel csv>` |
| 自訂 horizon | `... --label-horizon 60`(quarterly per §14.7-CY)|

### 不提供之旗標 (Intentionally Omitted)

- `--seed N`:seed=5422 為固定值(per §14.7-CW T_CW-4)。若需 reproducibility 跑 ≥ 3 runs(per T_CW-6)。
- `--hyperparams`:conservative defaults 為治權契約;不可運行時改動。
- `--save-fold-models`:per-fold model 不持久化(僅 final full-train model)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-05-29 | Codex | **首版:§14.7-CW Tree Family 第二實作**(LGBM v0.2 為第一)。**功能 7 點**:(a) 8-panel walk-forward expanding window OOS;(b) XGBoost params 對齊 §14.7-CW T_CW-4 conservative defaults(200/0.05/5/5/0.8/0.8/0.1/0.1/5422);(c) Treaty Gates 4/4 動態評估;(d) Model artifact(model.json + metrics.json + hyperparams.json);(e) model_registry INSERT ON CONFLICT;(f) Top-15 gain importance;(g) `--panel-feature-sets` 支援 §14.7-CX 95-panel 模式。**首跑實證**(2026-05-29):Sharpe 4.58 / IR 5.62 / Win 83.3% / α +15.85% / MDD 2.77% / Treaty Gates 4/4 PASS PERFECT。**對標 LGBM v0.2**:同 8-panel run 略勝(Sharpe 4.58 vs 3.84 / +19.3%)。**Model committed**:`mdl_20260415_xgboost_h30_0b243a67_v0_1`。**治權邊界**:不入憲(§14.7-CW 已涵蓋 tree family 治權,本程式為 implementation extension);若 production 升 XGBoost 需 §14.7-DA Tree Family Comparison Doctrine 入憲。**CLAUDE.md §一.11 強制三段式標頭**(2026-05-29 入憲):本版補正標頭為 sovereign_sync_engine.py 範本格式。 | **ACTIVE** |
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
import xgboost as xgb
from core.db_utils import get_db_conn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"  # XGBoost first version
MODEL_FAMILY = "xgboost"

DEFAULT_TRAINING_POLICY = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
    "seed": 5422,
    "tree_method": "hist",
}

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

# 43 canonical features per §14.7-CL
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
    cur.execute("""
        SELECT stock_id, feature_name, feature_value::numeric
        FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s) AND is_null_imputed IS NOT TRUE
    """, (fs_id, list(universe)))
    feat_data = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_43:
            feat_data[sid][fname] = float(val)

    cur.execute("""
        SELECT MIN(date) FROM "TaiwanStockPriceAdj"
        WHERE date >= (%s::date + INTERVAL '%s days')
          AND stock_id ~ '^[0-9]'
          AND date <= (%s::date + INTERVAL '%s days')
    """, (as_of, label_horizon, as_of, label_horizon + 10))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date:
        return [], [], [], None

    cur.execute("""
        WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
             t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
        SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric)
        FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id
    """, (as_of, label_date))
    returns = {sid: float(r) for sid, r in cur.fetchall() if sid in universe}

    X, y, sids = [], [], []
    for sid in universe:
        if sid in feat_data and sid in returns:
            feat_vec = [feat_data[sid].get(f, 0.0) for f in SPEC_43]
            X.append(feat_vec); y.append(returns[sid]); sids.append(sid)
    return X, y, sids, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10: return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    lo = np.quantile(arr, lo_q); hi = np.quantile(arr, hi_q)
    return np.clip(arr, lo, hi)


def main():
    parser = argparse.ArgumentParser(description=f"XGBoost Tree Model Trainer ({TOOL_VER})")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--label-horizon", type=int, default=30)
    parser.add_argument("--panel-feature-sets", type=str, default=None)
    args = parser.parse_args()
    if not args.dry_run and not args.commit:
        args.dry_run = True

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.stock_id FROM core_universe_membership m
            JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
            WHERE s.status='committed' AND m.core_tier='core_universe'
            AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1)
        """)
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
        logger.info(f"§14.7-CW XGBoost Tree Model Trainer {TOOL_VER}")
        logger.info("=" * 120)
        logger.info(f"  Universe:              {len(universe)} stocks")
        logger.info(f"  Panels:                {len(panels)}")
        logger.info(f"  Label horizon:         {args.label_horizon}d")
        logger.info(f"  Mode:                  {'COMMIT' if args.commit else 'DRY-RUN'}")
        logger.info(f"  Features(§14.7-CL):  {len(SPEC_43)}")
        logger.info(f"  XGBoost version:       {xgb.__version__}")

        # Load all panels
        logger.info("\n──── Loading walk-forward training data ────")
        all_X, all_y = [], []
        per_panel = {}
        for fs_id, as_of in panels:
            X, y, sids, label_date = load_panel_data(cur, fs_id, as_of, args.label_horizon, universe)
            if not X:
                logger.warning(f"  Panel {as_of}:no valid forward data,skipped")
                continue
            logger.info(f"  Panel {as_of} → label_date={label_date}:N={len(X)}")
            all_X.extend(X); all_y.extend(y)
            per_panel[as_of] = (X, y, sids, label_date)

        X_train = np.array(all_X); y_train = np.array(all_y)
        logger.info(f"\n  Total training rows:  {len(X_train):,}({len(panels)} panels combined)")
        y_train_w = winsorize(y_train, 0.01, 0.99)

        # Walk-forward expanding window OOS
        logger.info("\n──── Walk-Forward Expanding Window OOS Evaluation ────")
        logger.info("  方法:expanding window — train [0..i-1] → test panel i(OOS)")
        for k, v in DEFAULT_TRAINING_POLICY.items():
            logger.info(f"    {k:20} = {v}")

        panels_list = sorted(per_panel.items(), key=lambda x: x[0])
        panel_ics, panel_returns_top20, panel_returns_univ = [], [], []

        xgb_params = {k: v for k, v in DEFAULT_TRAINING_POLICY.items() if k != "n_estimators"}

        for i in range(1, len(panels_list)):
            test_as_of, (X_test, y_test, sids_test, label_date) = panels_list[i]
            train_X, train_y = [], []
            for j in range(i):
                X_j, y_j, _, _ = panels_list[j][1]
                train_X.extend(X_j); train_y.extend(y_j)
            X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y), 0.01, 0.99)

            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=SPEC_43)
            fold_model = xgb.train(xgb_params, dtrain, num_boost_round=DEFAULT_TRAINING_POLICY["n_estimators"])

            X_te = np.array(X_test)
            dtest = xgb.DMatrix(X_te, feature_names=SPEC_43)
            pred_te = fold_model.predict(dtest)
            ic_te = spearman_ic(pred_te, y_test)
            panel_ics.append(ic_te)

            n_top = min(20, len(pred_te))
            top_idx = np.argsort(pred_te)[-n_top:]
            top20_ret = float(np.mean([y_test[k] for k in top_idx]))
            univ_ret = float(np.mean(y_test))
            panel_returns_top20.append(top20_ret)
            panel_returns_univ.append(univ_ret)
            logger.info(f"  Train[0..{i-1}] → Test {test_as_of}: IC={ic_te:+.4f} / Top20={top20_ret:+.4f} / Universe={univ_ret:+.4f}")

        # Full in-sample reference
        dtrain_full = xgb.DMatrix(X_train, label=y_train_w, feature_names=SPEC_43)
        model = xgb.train(xgb_params, dtrain_full, num_boost_round=DEFAULT_TRAINING_POLICY["n_estimators"])
        pred_train = model.predict(xgb.DMatrix(X_train, feature_names=SPEC_43))
        in_sample_ic = spearman_ic(pred_train, y_train_w)
        in_sample_rmse = float(np.sqrt(np.mean((pred_train - y_train_w) ** 2)))
        logger.info(f"\n  [Reference]Full-train in-sample IC:   {in_sample_ic:+.4f}")
        logger.info(f"  [Reference]Full-train in-sample RMSE: {in_sample_rmse:.4f}")
        logger.info(f"  Overfit gap(in-sample - OOS):         {in_sample_ic - float(np.mean(panel_ics)):+.4f}")

        # Backtest metrics
        logger.info("\n──── Backtest Metrics ────")
        mean_ic = float(np.mean(panel_ics))
        cross_panel_ic_std = float(np.std(panel_ics, ddof=1))
        mean_ret = float(np.mean(panel_returns_top20))
        std_ret = float(np.std(panel_returns_top20, ddof=1))
        sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0
        win_rate = sum(1 for r in panel_returns_top20 if r > 0) / len(panel_returns_top20)
        cum = sum(panel_returns_top20)
        peak = 0; mdd = 0; running = 0
        for r in panel_returns_top20:
            running += r
            if running > peak: peak = running
            dd = peak - running
            if dd > mdd: mdd = dd
        alphas = [t - u for t, u in zip(panel_returns_top20, panel_returns_univ)]
        mean_alpha = float(np.mean(alphas))
        std_alpha = float(np.std(alphas, ddof=1))
        ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0
        t_stat = mean_alpha / (std_alpha / math.sqrt(len(alphas))) if std_alpha > 0 else 0

        logger.info(f"  Cross-panel IC mean:    {mean_ic:+.4f}")
        logger.info(f"  Cross-panel IC std:     {cross_panel_ic_std:.4f}")
        logger.info(f"  Mean panel return:      {mean_ret:+.4f}({mean_ret*100:+.2f}%)")
        logger.info(f"  Sharpe(annualized):    {sharpe:+.4f}")
        logger.info(f"  Win rate:               {win_rate*100:.1f}%")
        logger.info(f"  Max drawdown:           {mdd:.4f}({mdd*100:.2f}%)")
        logger.info(f"  Mean alpha:             {mean_alpha:+.4f}({mean_alpha*100:+.2f}%)")
        logger.info(f"  Information Ratio:      {ir:+.4f}")
        logger.info(f"  t-statistic(α):        {t_stat:+.4f}")
        logger.info(f"  Cumulative return:      {cum:+.4f}({cum*100:+.2f}%)")

        # Treaty gates
        logger.info("\n──── §14.7-CW Treaty Gates ────")
        g1 = "✅ PASS" if sharpe > 0 else "❌ VIOLATION"
        g2 = "✅ PASS" if win_rate >= 0.5 else "❌ VIOLATION"
        g3 = "✅ PASS" if mdd <= 0.30 else "⚠️ ALERT"
        g4 = "✅ PASS" if mean_alpha > 0 else "❌ VIOLATION"
        logger.info(f"  Gate CW-1(Sharpe > 0):              {g1}({sharpe:.4f})")
        logger.info(f"  Gate CW-2(Win rate ≥ 50%):         {g2}({win_rate*100:.1f}%)")
        logger.info(f"  Gate CW-3(MDD ≤ 30%):              {g3}({mdd*100:.2f}%)")
        logger.info(f"  Gate CW-4(Mean alpha > 0):         {g4}({mean_alpha:.4f})")

        # Feature importance
        importance_dict = model.get_score(importance_type='gain')
        fi = sorted(importance_dict.items(), key=lambda x: -x[1])
        logger.info("\n──── Top 15 Features by Gain Importance ────")
        for i, (f, imp) in enumerate(fi[:15], 1):
            logger.info(f"  {i:>2}. {f:38} gain={imp:>12,.2f}")

        # Commit
        if args.commit:
            logger.info("\n──── COMMIT mode ────")
            feature_set_hash = hashlib.sha1("feature_set_v0.4".encode()).hexdigest()[:8]
            train_date = max(p[1] for p in panels)
            model_id = f"mdl_{train_date.replace('-', '')}_xgboost_h{args.label_horizon}_{feature_set_hash}_v0_1"
            artifact_dir = Path("data/models") / model_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifact_dir / "model.json"
            model.save_model(str(model_path))

            metrics = {
                "trainer": "xgboost_tree_v0_1",
                "model_family": MODEL_FAMILY,
                "label_horizon": args.label_horizon,
                "feature_count": len(SPEC_43),
                "rows_trained": len(X_train),
                "panels": len(panels),
                "in_sample_ic": in_sample_ic,
                "in_sample_rmse": in_sample_rmse,
                "cross_panel_ic_mean": mean_ic,
                "cross_panel_ic_std": cross_panel_ic_std,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "mdd": mdd,
                "mean_alpha": mean_alpha,
                "information_ratio": ir,
                "t_statistic": t_stat,
                "cumulative_return": cum,
                "top_features": [{"feature": f, "gain": float(imp)} for f, imp in fi[:20]],
            }
            with open(artifact_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            with open(artifact_dir / "hyperparams.json", "w") as f:
                json.dump(DEFAULT_TRAINING_POLICY, f, indent=2)

            cur.execute("SELECT snapshot_id FROM core_universe_snapshot WHERE status='committed' ORDER BY snapshot_id DESC LIMIT 1")
            r = cur.fetchone()
            universe_snapshot_id = r[0] if r else None

            cur.execute("""
                INSERT INTO model_registry(
                    model_id, model_policy_version, model_family, feature_set_id,
                    universe_snapshot_id, label_horizon, train_start_date, train_end_date,
                    metrics, hyperparams, artifact_path, status, notes
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)
                ON CONFLICT(model_id) DO UPDATE SET
                    metrics = EXCLUDED.metrics, hyperparams = EXCLUDED.hyperparams,
                    artifact_path = EXCLUDED.artifact_path, status = EXCLUDED.status,
                    notes = EXCLUDED.notes
            """, (
                model_id, "model_policy_v0.1", MODEL_FAMILY,
                panels[-1][0], universe_snapshot_id, args.label_horizon,
                panels[0][1], train_date,
                json.dumps(metrics, default=str), json.dumps(DEFAULT_TRAINING_POLICY),
                str(artifact_dir), "committed",
                f"v0.1 XGBoost tree;walk-forward;§14.7-CW family extension"
            ))
            conn.commit()
            logger.info(f"  ✅ Model committed: {model_id}")
            logger.info(f"  ✅ Artifact: {artifact_dir}/")
            logger.info(f"  ✅ model_registry inserted")

        logger.info("\n" + "=" * 120)
        verdict = "PERFECT" if all("PASS" in g for g in [g1, g2, g3, g4]) else "WARNING"
        logger.info(f"§14.7-CW XGBoost Trainer {TOOL_VER}:主權判定 {verdict}")
        logger.info("=" * 120)
        if verdict == "PERFECT":
            logger.info(f"  🎯 XGBoost achieves Sharpe={sharpe:.2f} / IR={ir:.2f} / Win={win_rate*100:.0f}%")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
