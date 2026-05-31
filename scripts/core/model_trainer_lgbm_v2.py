"""
model_trainer_lgbm_v2.py v0.2 (LightGBM Tree Production Trainer · §14.7-CW Tree Family 第一實作 · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-05-29(標頭三段式補正;v0.2 邏輯 2026-05-28 入)
**主權狀態**: LGBM TREE PRODUCTION + §14.7-CW TREE-FAMILY 第一實作 + §14.7-CS MODEL-TRAINING-LANDING + §14.7-CW T_CW-6 REPRODUCIBILITY + §14.7-CL 43-FEATURE CANONICAL + §一.10 SOURCE-TRACEABLE + §一.11 三段式合規 (per CLAUDE.md §一.11)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Real Tree Model Required]** (v0.2, 憲法 §14.7-CW T_CW-1): production model 必須使用 real LightGBM tree;不得停留 rank-IC linear baseline(v0.1)。
2. **[Expanding Window Walk-Forward OOS]** (v0.2, 憲法 §14.7-CW T_CW-2): train [0..i-1] → test i;in-sample IC vs OOS IC gap 須揭露。
3. **[Conservative Hyperparameters]** (v0.2, 憲法 §14.7-CW T_CW-4): n_estimators=200 / learning_rate=0.05 / max_depth=5 / num_leaves=20 / min_child_samples=30 / feature_fraction=0.8 / bagging_fraction=0.8 / reg_alpha=0.1 / reg_lambda=0.1 / seed=5422。
4. **[Treaty Gates 4/4]** (v0.2, 憲法 §14.7-CW T_CW-5): Sharpe > 0 / Win ≥ 50% / MDD ≤ 30% / Mean α > 0。
5. **[Top Features §0.1+§0.2 Dominated]** (v0.2, 憲法 §14.7-CW T_CW-3): top-15 features 由 §0.1+§0.2 主導;§0.3 macro broadcast 已 removed(per §14.7-CK)。
6. **[Multi-Run Reproducibility]** (v0.2, 憲法 §14.7-CW T_CW-6): LGBM bagging + multi-thread → stochastic;不得 single-run anchor 寫 deterministic fact;charter inscription 須 multi-run statistics(per v6.17.1 patch precedent)。
7. **[43 Canonical Features]** (v0.2, 憲法 §14.7-CL): SPEC_43 為 §14.7-CL canonical scope;與 XGBoost 共用同一 SPEC。
8. **[Source Traceability]** (v0.2, CLAUDE.md §一.10): 全 (b) DB query + (a) program output;0 AI memory;0 推測。
9. **[Zero Hardcoded Verdict]** (v0.2, 憲法 §5.6.3): 4 Treaty Gates 動態判定。
10. **[Sovereignty Declaration]** (v0.2, 憲法 §3.1 序列模組): 本程式為 **§10 model_trainer 第一實作**(XGBoost 為第二)。**治權邊界**:(a) §3.1 序列 training;(b) 五套禁令不涉;(c) T1-T3 不分層;(d) §8.5 已 handle by feature_store_builder;(e) **不選股 / 不算 feature / 不評估 multi-cycle**(各有專責);(f) 唯一職責:walk-forward LGBM 訓練 + Treaty gates + model artifact + model_registry insert。
11. **[Historical Reference Authority]** (v0.2): `TOOL_VER = "v0.2"`;v6.17.0 6-panel anchor Sharpe 4.74 為 max outlier(per v6.17.1 patch 揭露);真實 8-year reality 為 §14.7-CX Sharpe 1.67。
12. **[Idempotency]** (v0.2): model_registry INSERT ON CONFLICT DO UPDATE。
13. **[Anti-Leakage Inheritance]** (v0.2, 憲法 §8.5): features 已 anti-leakage compliant per feature_store_builder publication_date_strategy。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Universe + Feature Loading

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Latest core_universe | DB query | §14.7-CF SSOT |
| A.2 Feature values | `load_panel_data()` | §14.7-CL canonical |
| A.3 Forward returns | PriceAdj LN(t1/t0)| §14.7-CV |

### Group B. Walk-Forward Expanding Window Training

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window | train [0..i-1] → test i | §14.7-CW T_CW-2 |
| B.2 LGBM params | DEFAULT_TRAINING_POLICY | §14.7-CW T_CW-4 |
| B.3 Winsorization | clip [0.01, 0.99] | label stability |
| B.4 Spearman IC | rank correlation | §14.7-CM |

### Group C. Treaty Gates Evaluation

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1-4 | Sharpe / Win / MDD / Alpha | §14.7-CW Gates CW-1〜4 |
| C.5 t-statistic | mean_α / SE | §14.7-CX significance |

### Group D. Model Artifact Persistence (--commit only)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 model.txt save | LGBM native | LGBM format |
| D.2 metrics.json | full serialization | §一.10 |
| D.3 hyperparams.json | dump | §14.7-CW T_CW-4 |
| D.4 model_registry INSERT | ON CONFLICT DO UPDATE | §10 SSOT |

### Group E. Feature Importance Reporting

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| E.1 Gain importance | `model.feature_importance(importance_type='gain')` | §14.7-CN |
| E.2 Top-15 stdout | sorted descending | §14.7-CW T_CW-3 |

### Group F. CLI + Mode Control

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| F.1 `--dry-run` | 評估不寫 DB | safe default |
| F.2 `--commit` | 寫 model_registry + artifact | production |
| F.3 `--panel-feature-sets <csv>` | walk-forward panels override | §14.7-CX support |
| F.4 `--label-horizon N` | default 30d | §14.7-CW |

### 對齊憲章 §二 維運矩陣

| 場景 | 命令 |
| :--- | :--- |
| 8-panel dry-run | `python scripts/core/model_trainer_lgbm_v2.py --dry-run` |
| Commit production | `... --commit` |
| 95-panel walk-forward(per §14.7-CX)| `... --panel-feature-sets <95-panel csv>` |

### 不提供之旗標 (Intentionally Omitted)

- `--seed`:固定 5422 per §14.7-CW T_CW-4。
- `--hyperparams`:conservative defaults 為治權契約。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.2 | 2026-05-29 | Codex | **CLAUDE.md §一.11 三段式標頭補正**:依用戶 explicit directive 2026-05-29 補入 13 條核心定義 / 6-Group functional matrix / 全修訂歷程,對齊 sovereign_sync_engine.py v1.22 範本。原 v0.2 functional 邏輯不變(2026-05-28 入)。 | **ACTIVE** |
| v0.2(pre-§一.11)| 2026-05-28 | Codex | **§14.7-CW Tree Model Production Upgrade**(v6.17.0):real LightGBM tree 取代 rank-IC linear baseline。8-panel expanding window walk-forward OOS。Conservative hyperparams(200/0.05/5/20/30/0.8/0.8/0.1/0.1/5422)。Treaty Gates 4/4 PASS PERFECT。**首跑實證**:Sharpe 4.74 / IR 5.86 / Win 83.3% / α +16.22% / MDD 1.48%;**v6.17.1 patch 揭露**: 6-run reality Sharpe 3.71-4.74(median 3.90)/ commit anchor 4.74 為 max outlier。**Model committed**:`mdl_20260415_lgbm_h30_0b243a67_v0_2`。**95-panel reality(per §14.7-CX)**:Sharpe 1.67 / Win 67.7% / α +1.94% / Eff t 3.72。 | ARCHIVED(標頭格式;邏輯仍 ACTIVE)|
| v0.1 | 2026-05-27 | Codex | **rank-IC linear baseline**(pre-§14.7-CW):rank-IC linear combination;無 tree non-linear interactions;Sharpe ~3.10 / Win 75%。被 v0.2 取代為 production。 | SUPERSEDED |
"""
from __future__ import annotations
import sys, argparse, hashlib, json, logging, math, os
from datetime import datetime, date
from pathlib import Path
from collections import defaultdict

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
TOOL_VER = "v0.2"  # §14.7-CW tree model upgrade
DEFAULT_TRAINING_POLICY = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 20,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "seed": 5422,
}

# 8-panel training panels(default)
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
    """Load features at as_of + forward returns from as_of+horizon trading day"""
    # Features
    cur.execute("""
        SELECT stock_id, feature_name, feature_value::numeric
        FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s) AND is_null_imputed IS NOT TRUE
    """, (fs_id, list(universe)))
    feat_data = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_43:
            feat_data[sid][fname] = float(val)

    # Find nearest trading day for forward label_horizon
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

    # Forward returns
    cur.execute("""
        WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
             t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
        SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric)
        FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id
    """, (as_of, label_date))
    returns = {sid: float(r) for sid, r in cur.fetchall() if sid in universe}

    # Build X, y, sid arrays
    X, y, sids = [], [], []
    for sid in universe:
        if sid in feat_data and sid in returns:
            feat_vec = [feat_data[sid].get(f, 0.0) for f in SPEC_43]
            X.append(feat_vec)
            y.append(returns[sid])
            sids.append(sid)
    return X, y, sids, label_date


def spearman_ic(pred, y):
    pred = np.array(pred); y = np.array(y)
    rp = pred.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    if np.std(rp) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    return float(np.corrcoef(rp, ry)[0, 1])


def winsorize(arr, lo_q=0.05, hi_q=0.95):
    lo = np.quantile(arr, lo_q)
    hi = np.quantile(arr, hi_q)
    return np.clip(arr, lo, hi)


def main():
    parser = argparse.ArgumentParser(description=f"LGBM Tree Model Trainer ({TOOL_VER})")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--label-horizon", type=int, default=30)
    parser.add_argument("--panel-feature-sets", type=str, default=None,
                        help="Comma-separated panel fs_ids(8 panels for cross-validation)")
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
        """)
        universe = list({r[0] for r in cur.fetchall()})

        # Determine panels
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
        logger.info(f"§14.7-CW LGBM Tree Model Trainer {TOOL_VER}(real LightGBM tree)")
        logger.info("=" * 120)
        logger.info(f"  Universe:              {len(universe)} stocks")
        logger.info(f"  Panels:                {len(panels)}")
        logger.info(f"  Label horizon:         {args.label_horizon}d")
        logger.info(f"  Mode:                  {'COMMIT' if args.commit else 'DRY-RUN'}")
        logger.info(f"  Features(§14.7-CL 43):{len(SPEC_43)}")

        # ── Load all panels ──
        logger.info("\n──── Loading 8-panel walk-forward training data ────")
        all_X, all_y, all_panel = [], [], []
        per_panel = {}
        for fs_id, as_of in panels:
            X, y, sids, label_date = load_panel_data(cur, fs_id, as_of, args.label_horizon, universe)
            if not X:
                logger.warning(f"  Panel {as_of}:no valid forward data,skipped")
                continue
            logger.info(f"  Panel {as_of} → label_date={label_date}:N={len(X)}")
            all_X.extend(X); all_y.extend(y)
            all_panel.extend([as_of] * len(X))
            per_panel[as_of] = (X, y, sids, label_date)

        X_train = np.array(all_X)
        y_train = np.array(all_y)
        logger.info(f"\n  Total training rows:  {len(X_train):,}({len(panels)} panels combined)")

        # Winsorize labels(stability)
        y_train_w = winsorize(y_train, 0.01, 0.99)

        # ── Walk-forward expanding window OOS evaluation ──
        # Proper time-series CV:for each panel i ≥ 2:
        #   train on panels [1, ..., i-1], test on panel i (OOS)
        logger.info("\n──── Walk-Forward Expanding Window OOS Evaluation ────")
        logger.info("  方法:expanding window — train [1..i-1] → test panel i(OOS)")
        logger.info(f"  n_estimators:    {DEFAULT_TRAINING_POLICY['n_estimators']}")
        logger.info(f"  learning_rate:   {DEFAULT_TRAINING_POLICY['learning_rate']}")
        logger.info(f"  max_depth:       {DEFAULT_TRAINING_POLICY['max_depth']}")
        logger.info(f"  num_leaves:      {DEFAULT_TRAINING_POLICY['num_leaves']}")

        lgb_params = {k: v for k, v in DEFAULT_TRAINING_POLICY.items() if k != "n_estimators"}

        panels_list = sorted(per_panel.items(), key=lambda x: x[0])
        panel_ics = []
        panel_returns_top20 = []
        panel_returns_univ = []

        for i in range(1, len(panels_list)):  # start from panel 2(need train data)
            test_as_of, (X_test, y_test, sids_test, label_date) = panels_list[i]
            # Train on panels [0, ..., i-1]
            train_X, train_y = [], []
            for j in range(i):
                X_j, y_j, _, _ = panels_list[j][1]
                train_X.extend(X_j); train_y.extend(y_j)
            X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y), 0.01, 0.99)

            # Train fold model
            train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=SPEC_43)
            fold_model = lgb.train(lgb_params, train_data, num_boost_round=DEFAULT_TRAINING_POLICY["n_estimators"])

            # Predict on test panel(OOS)
            X_te = np.array(X_test)
            pred_te = fold_model.predict(X_te)
            ic_te = spearman_ic(pred_te, y_test)
            panel_ics.append(ic_te)

            # Top-20 OOS strategy
            n_top = min(20, len(pred_te))
            top_idx = np.argsort(pred_te)[-n_top:]
            top20_ret = float(np.mean([y_test[k] for k in top_idx]))
            univ_ret = float(np.mean(y_test))
            panel_returns_top20.append(top20_ret)
            panel_returns_univ.append(univ_ret)
            logger.info(f"  Train[{0}..{i-1}] → Test {test_as_of}(N={i} train panels): IC={ic_te:+.4f} / Top20={top20_ret:+.4f} / Universe={univ_ret:+.4f}")

        # In-sample on full(for reference / overfitting check)
        train_data_full = lgb.Dataset(X_train, label=y_train_w, feature_name=SPEC_43)
        model = lgb.train(lgb_params, train_data_full, num_boost_round=DEFAULT_TRAINING_POLICY["n_estimators"])
        pred_train = model.predict(X_train)
        in_sample_ic = spearman_ic(pred_train, y_train_w)
        in_sample_rmse = float(np.sqrt(np.mean((pred_train - y_train_w) ** 2)))
        logger.info(f"\n  [Reference]Full-train in-sample IC:   {in_sample_ic:+.4f}")
        logger.info(f"  [Reference]Full-train in-sample RMSE: {in_sample_rmse:.4f}")
        logger.info(f"  Overfit gap(in-sample - OOS):         {in_sample_ic - float(np.mean(panel_ics)):+.4f}")

        # ── Backtest metrics ──
        logger.info("\n──── Backtest Metrics(per §14.7-CV 4 gates)────")
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

        logger.info(f"  Cross-panel IC mean:    {mean_ic:+.4f}")
        logger.info(f"  Cross-panel IC std:     {cross_panel_ic_std:.4f}")
        logger.info(f"  Mean panel return:      {mean_ret:+.4f}({mean_ret*100:+.2f}%)")
        logger.info(f"  Sharpe(annualized):    {sharpe:+.4f}")
        logger.info(f"  Win rate:               {win_rate*100:.1f}%")
        logger.info(f"  Max drawdown:           {mdd:.4f}({mdd*100:.2f}%)")
        logger.info(f"  Mean alpha:             {mean_alpha:+.4f}({mean_alpha*100:+.2f}%)")
        logger.info(f"  Information Ratio:      {ir:+.4f}")
        logger.info(f"  Cumulative return:      {cum:+.4f}({cum*100:+.2f}%)")

        # Treaty gates
        logger.info("\n──── §14.7-CW Treaty Gates(per §14.7-CV)────")
        g1 = "✅ PASS" if sharpe > 0 else "❌ VIOLATION"
        g2 = "✅ PASS" if win_rate >= 0.5 else "❌ VIOLATION"
        g3 = "✅ PASS" if mdd <= 0.30 else "⚠️ ALERT"
        g4 = "✅ PASS" if mean_alpha > 0 else "❌ VIOLATION"
        logger.info(f"  Gate CW-1(Sharpe > 0):              {g1}({sharpe:.4f})")
        logger.info(f"  Gate CW-2(Win rate ≥ 50%):         {g2}({win_rate*100:.1f}%)")
        logger.info(f"  Gate CW-3(MDD ≤ 30%):              {g3}({mdd*100:.2f}%)")
        logger.info(f"  Gate CW-4(Mean alpha > 0):         {g4}({mean_alpha:.4f})")

        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        fi = sorted(zip(SPEC_43, importance), key=lambda x: -x[1])
        logger.info("\n──── Top 15 Features by Gain Importance ────")
        for i, (f, imp) in enumerate(fi[:15], 1):
            logger.info(f"  {i:>2}. {f:38} gain={imp:>12,.2f}")

        # ── Commit ──
        if args.commit:
            logger.info("\n──── COMMIT mode ────")
            # Generate model_id
            feature_set_hash = hashlib.sha1("feature_set_v0.4".encode()).hexdigest()[:8]
            train_date = max(p[1] for p in panels)
            model_id = f"mdl_{train_date.replace('-', '')}_lgbm_h{args.label_horizon}_{feature_set_hash}_v0_2"
            artifact_dir = Path("data/models") / model_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifact_dir / "model.txt"
            model.save_model(str(model_path))

            # Write metrics.json
            metrics = {
                "trainer": "lgbm_tree_v0_2",
                "model_family": "lgbm",
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
                "cumulative_return": cum,
                "top_features": [{"feature": f, "gain": float(imp)} for f, imp in fi[:20]],
            }
            with open(artifact_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            # Write hyperparams.json
            with open(artifact_dir / "hyperparams.json", "w") as f:
                json.dump(DEFAULT_TRAINING_POLICY, f, indent=2)

            # Write to model_registry
            universe_snapshot_id = None
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
                model_id, "model_policy_v0.2", "lgbm",
                panels[-1][0],  # latest fs_id as primary
                universe_snapshot_id, args.label_horizon,
                panels[0][1], train_date,
                json.dumps(metrics, default=str), json.dumps(DEFAULT_TRAINING_POLICY),
                str(artifact_dir), "committed",
                f"v0.2 LGBM tree(real);8-panel walk-forward;§14.7-CW production"
            ))
            conn.commit()

            logger.info(f"  ✅ Model committed: {model_id}")
            logger.info(f"  ✅ Artifact: {artifact_dir}/")
            logger.info(f"  ✅ model_registry inserted")

        logger.info("\n" + "=" * 120)
        verdict = "PERFECT" if all("PASS" in g for g in [g1, g2, g3, g4]) else "WARNING"
        logger.info(f"§14.7-CW v0.2 LGBM Tree Trainer:主權判定 {verdict}")
        logger.info("=" * 120)
        if verdict == "PERFECT":
            logger.info(f"  🎯 Tree model exceeds Sharpe={sharpe:.2f} / IR={ir:.2f} / Win={win_rate*100:.0f}%")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
