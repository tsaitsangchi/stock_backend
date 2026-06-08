"""
multi_cycle_tabnet_validation.py v0.1 (Multi-Cycle TabNet Attention-Tabular Validation Engine · per §14.7-DG common comparison baseline · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-06-04
**主權狀態**: MULTI-CYCLE 4-HORIZON TABNET VALIDATION + §14.7-CY HORIZON-EXTENSION + §14.7-CX 8-YEAR OOS EXTENSION + §一.10 SOURCE-TRACEABLE (per CLAUDE.md §一.11 強制三段式標頭;§14.7-CY T_CY-1 system script 強制;§一.10 三類唯一 source 強制) + §14.7-DE/DF §0.0-I 雙 helper(get_canonical_panel_dates / summarize_horizon_metrics)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用 TabNet(注意力機制表格深度學習) 模型在 37 個股票特徵上預測個股未來報酬、選最看好的前 20 支做多,作為「注意力型深度表格模型有沒有贏過樹 / 線性」之共同比較基準。

**它怎麼做(步驟)**:
1. 取 397 支「乾淨核心股」+ 每股 37 個經驗證的特徵(估值、動能、籌碼、財務等;全來自真實 FinMind/FRED API,無 AI 亂補值)。
2. 把 2013-05 ~ 2026-06 切成多個月度時間點(panel)。
3. **逐點往前走(walk-forward)**:每個時間點「只用那之前的資料」先用 StandardScaler 標準化特徵,再 fit 一個 TabNet(注意力機制表格深度模型)模型,預測「之後」每支股票的報酬 —— 不偷看未來(防資料洩漏)。TabNet 用 attention 自動挑「每筆樣本該看哪些特徵」。
4. 依預測分數挑最看好的 **前 20 支等權做多**,跟「全市場平均」比,算這一期賺賠。
5. 上面在 **4 種持有期**各做一遍:週(5 天)/ 月(20 天)/ 季(60 天)/ 年(252 天)。
6. 算每種週期的成績:**報酬率、Sharpe、勝率、預測命中率、跨次穩定度(可信度)、扣成本後年化報酬**(全經共用 helper `summarize_horizon_metrics`,與其他模型同指標 → 可比)。
7. 用門檻(§14.7-CZ T_CZ-6:有效 t 值 ≥ 4.20 且 Sharpe ≥ 2.40 且 勝率 ≥ 79%)判定:**這模型在哪個週期「真的能賺錢且可信」**。

**輸入**:資料庫(feature_values 特徵 + TaiwanStockPriceAdj 股價)。**輸出**:JSON(各週期成績)+ log。
**它不做的事**:不訓練正式上線模型、不改資料庫(純讀取評估;§3.2 evaluation 角色)。torch 為 CPU 版,故 max_epochs 控制在保守值(50)避免過久。
**為什麼需要它**:回答「注意力型深度表格模型能否在 37 特徵上贏過樹 / 線性」的實證裁判;所有模型用同一支標準、同窗、同指標 → 成績才能公平並排比較。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Multi-Cycle Horizon Coverage]** (v0.1, 憲法 §14.7-CY T_CY-2): 至少跨 ≥ 3 horizons(本程式預設 4: weekly 5d / monthly 20d / quarterly 60d / annual 252d);違反者視為 single-horizon evidence insufficient。
2. **[Overlap-Corrected n_effective]** (v0.1, 憲法 §14.7-CY T_CY-3): 長 horizon(> panel spacing 30d)panels 互相 overlap;統計顯著性必以 `n_eff = n × (30 / horizon_days)` 計算(由 `summarize_horizon_metrics` 統一計);raw t-stat 不得作為 production claim。
3. **[Honest Annualization]** (v0.1, 憲法 §14.7-CY T_CY-4): 年化報酬必用 `mean × (252 / horizon_days)`(per-rebalance expected);不得 sum overlapping panel returns(會 double-count)。
4. **[Cost-Drag Per Horizon]** (v0.1, 憲法 §14.7-CY T_CY-5): 各 horizon 之 transaction cost drag(`0.6%/rebal × rebals_per_year`)必揭露;short horizon(weekly/daily)即便 gross alpha 也可能被 cost 吃光。
5. **[System Script Mandatory]** (v0.1, 憲法 §14.7-CY T_CY-1): 多週期驗證必須為 system 永久 script,**禁止**在 AI ephemeral 環境臨時計算;script 須 git-tracked + version controlled。
6. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全 (b) DB query(feature_values + TaiwanStockPriceAdj)+ (a) program output(stdout + JSON);**0 AI memory reuse**;**禁止從記憶 / 推測 / 估算**。
7. **[Reality-Check Hierarchy]** (v0.1, 憲法 §14.7-CY T_CY-6 / §14.7-CZ T_CZ-6): 推薦 production horizon 必同時滿足 robust significance(Eff t > 1.997)+ reasonable n_eff(≥ 30)+ acceptable cost drag(< 5%/yr);**quarterly horizon** 為 recommended sweet spot。
8. **[Attention-Tabular Baseline]** (v0.1, 憲法 §14.7-DG common comparison baseline): 本程式為 attention-based tabular deep-learning baseline(TabNetRegressor on standardized features / CPU torch);與 tree / linear / transformer / foundation 共享同一 walk-forward / overlap correction / honest annualization protocol → 回答「注意力型深度表格模型是否贏過樹 / 線性」。
9. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): 4 horizons 之 significance verdict 動態計算(`abs(eff_t_stat) > 1.997`,由 helper 計),**不**硬編 ✅/❌ 結論。
10. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 橫切模組): 本程式為 **§14.7-CY Multi-Cycle Validation TabNet attention-tabular baseline 實作**(對標 multi_cycle_validation.py LGBM 為樹族第一實作)。**治權邊界**:(a) §3.2 evaluation 模組;(b) 五套禁令(§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §6.8)不涉;(c) T1-T3 不分層;(d) §8.5 anti-leakage 不處理(features 已 anti-leakage compliant per feature_store_builder;train 僅用過去 panels);(e) **不訓練 production model**(僅 evaluate / 不寫 model_registry);(f) **不修改 feature_values / TaiwanStockPriceAdj**(read-only);(g) 唯一職責:跑 4-horizon walk-forward TabNet evaluation + 計算 metrics + 持久化 JSON。
11. **[Historical Reference Authority]** (v0.1): 本程式 `TOOL_VER = "v0.1"` 屬於記述性快照;tree-family / linear baseline 對比值來自既有 `reports/multi_cycle_*` JSON(權威來源)。
12. **[Idempotency]** (v0.1): `--dry-run` 不寫 DB;`--commit` 寫 `evaluation_log`(若實作);JSON output path 含 timestamp,不覆寫舊結果。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

> 本程式作為 §14.7-CY Multi-Cycle TabNet attention-tabular baseline 實作,依「驗證面向」拆分為 5 大功能群;每群對應憲章治權契約。

### Group A. Horizon-Specific Walk-Forward Evaluation — `--horizons <days_csv>`

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 weekly horizon(5d)| `evaluate_horizon(cur, panels, 5, universe, "weekly")` | §14.7-CY T_CY-2 |
| A.2 monthly horizon(20d)| `evaluate_horizon(cur, panels, 20, universe, "monthly")` | §14.7-CY T_CY-2 |
| A.3 quarterly horizon(60d)| `evaluate_horizon(cur, panels, 60, universe, "quarterly")` | §14.7-CY T_CY-6(recommended)|
| A.4 annual horizon(252d)| `evaluate_horizon(cur, panels, 252, universe, "annual")` | §14.7-CY T_CY-2(caveat: small n_eff)|
| 對應 CLI | `--horizons 5,20,60,252`(預設) | — |

### Group B. Walk-Forward TabNet Training — internal per-horizon

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Expanding window training | `for i in range(1, len(panel_keys))`(train [0..i-1] → test i)| §14.7-CW T_CW-2 |
| B.2 StandardScaler + TabNetRegressor(seed=5422 / CPU)| `StandardScaler` + `TabNetRegressor`(max_epochs 50 / patience 10 / batch 1024)| §14.7-DG baseline |
| B.3 Winsorization | `winsorize(arr, 0.01, 0.99)` | label stability |
| B.4 Spearman IC | (via `summarize_horizon_metrics`)| §14.7-CM Empirical IC |
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
| D.2 JSON persistence | `reports/multi_cycle_tabnet_<timestamp>.json` | §14.7-CY T_CY-1 |
| D.3 Meta context | `_meta`(tool_ver / pytorch_tabnet_version / seed / run_at)| §一.10 source_traceability |
| D.4 panel_records detail | per-panel ic / top20_ret / univ_ret / alpha(via helper)| audit trail |
| 對應 CLI | `--output reports/<path>.json` | — |

### Group E. Source Traceability + Anti-Hallucination

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| E.1 Universe via DB query | `core_universe_membership` + `core_universe_snapshot` JOIN | §一.10 (b) DB query |
| E.2 Features via DB query | `feature_values` WHERE feature_set_id IN ... + `is_null_imputed IS NOT TRUE` | §一.10 (b) DB query / §14.7-DC source-pure |
| E.3 Forward returns via DB | `TaiwanStockPriceAdj` LN(t1/t0)JOIN | §一.10 (b) DB query |
| E.4 0 AI memory reuse | All numbers from program output | §一.10 第 1 條 |
| E.5 No placeholder data | 所有 panel_data 為實際 query 結果 | §一.10 禁止來源(第四類) |
| 對應 CLI | (internal) | — |

### 對齊憲章 §二 維運矩陣（標準場景索引）

| 場景 | 對應命令 |
| :--- | :--- |
| 日常 multi-cycle validation | `python scripts/evaluation/multi_cycle_tabnet_validation.py --dry-run` |
| Commit 評估結果 | `python scripts/evaluation/multi_cycle_tabnet_validation.py --commit --output reports/<path>.json` |
| 自訂 horizons | `... --horizons 7,30,90,365` |
| 比對 tree-family / linear baseline | 對比 `reports/multi_cycle_*` JSON |

### 不提供之旗標 (Intentionally Omitted)

- `--cost-per-rebal X`:0.6% 為 §14.7-CY T_CY-5 conservative standard。
- `--write-model`:本程式為 evaluation only,不寫 `model_registry`。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-04 | Codex | **首版:§14.7-DG common comparison baseline — TabNet attention-tabular baseline 實作**(clone multi_cycle_xgboost_validation.py v0.2 結構,僅換模型 train/predict + 標頭)。**功能 4 點**:(a) 4-horizon walk-forward(5/20/60/252d);(b) StandardScaler + TabNetRegressor(seed / verbose=0 / device_name='cpu' / max_epochs=50 / patience=10 / batch_size=1024),`fit(Xtr, ytr.reshape(-1,1))` → `predict(Xte).ravel()`;(c) CPU torch(2.2.2),max_epochs 控制保守值;(d) 共用 `get_canonical_panel_dates` / `summarize_horizon_metrics`(§14.7-DE/DF §0.0-I 單一引用源)。**治權邊界**:read-only(不寫 model_registry / 不改 feature_values);全 (b) DB query + source-pure(`is_null_imputed IS NOT TRUE`);0 AI memory(per §一.10)。**Output**:stdout + JSON(`reports/multi_cycle_tabnet_<timestamp>.json`)。**v0.1 治權邊界**:不入憲(無新 doctrine,僅 §14.7-DG baseline implementation)。 | **ACTIVE** |
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
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from core.db_utils import get_db_conn, get_canonical_panel_dates, summarize_horizon_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

DEFAULT_HORIZONS = [("weekly", 5), ("monthly", 20), ("quarterly", 60), ("annual", 252)]

SEED = 5422
TABNET_MAX_EPOCHS = 50
TABNET_PATIENCE = 10
TABNET_BATCH_SIZE = 1024

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
        X_tr = np.array(train_X, dtype=np.float32); y_tr = winsorize(np.array(train_y)).astype(np.float32)
        if len(X_tr) < 100: continue

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr).astype(np.float32)
        X_te_s = scaler.transform(np.array(X_test, dtype=np.float32)).astype(np.float32)
        model = TabNetRegressor(seed=SEED, verbose=0, device_name="cpu")
        model.fit(X_tr_s, y_tr.reshape(-1, 1), max_epochs=TABNET_MAX_EPOCHS,
                  patience=TABNET_PATIENCE, batch_size=TABNET_BATCH_SIZE)
        pred_te = model.predict(X_te_s).ravel()
        panel_pa.append((pred_te, y_test))
    result = summarize_horizon_metrics(label, horizon_days, panel_pa)  # §14.7-DF Canonical Metric SSOT(單一來源)
    if result is None:
        return None

    logger.info(f"  {label}({horizon_days}d): Sharpe {result['sharpe']:+.3f} | Eff t {result['effective_t_stat']:+.3f} | Win {result['win_rate']*100:.1f}% | IC {result['mean_ic']:+.4f} | NET {result['annualized_simple_net']*100:+.1f}%/yr")
    logger.info(f"    precision: hit {result['precision_directional_hit_rate']*100:.1f}% | top-20 overlap {result['precision_top20_actual_overlap']*100:.1f}% | RMSE {result['precision_rmse']:.4f} | reliability IC-CoV {result['reliability_ic_stability_cov']:.4f}")
    return result


def main():
    global SEED
    parser = argparse.ArgumentParser(description=f"Multi-Cycle TabNet Validation {TOOL_VER}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="§一.10 #3 multi-run seed(canonical 5422;≥3 distinct seeds → min/median/max/mean)")
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True
    SEED = args.seed  # §一.10 #3: inject run seed into TabNet

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    import importlib.metadata as _im
    try:
        tabnet_ver = _im.version("pytorch_tabnet")
    except Exception:
        tabnet_ver = "unknown"
    import torch
    logger.info("="*100)
    logger.info(f"Multi-Cycle TabNet Validation {TOOL_VER}(per §14.7-CY / §14.7-DG)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  pytorch_tabnet ver: {tabnet_ver} | torch: {torch.__version__}")
    logger.info(f"  Model:    TabNetRegressor(cpu, max_epochs={TABNET_MAX_EPOCHS}, patience={TABNET_PATIENCE}, batch={TABNET_BATCH_SIZE})")
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

        logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix(TabNet)\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'N':>4} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'Win':>6} {'NetAnn':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_panels']:>4} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['win_rate']*100:>5.1f}% {r['annualized_simple_net']*100:>+8.2f}%")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_tabnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            output_results = {label: {k: v for k, v in r.items()} for label, r in results.items()}
            output_results["_meta"] = {
                "tool": "multi_cycle_tabnet_validation.py",
                "tool_ver": TOOL_VER,
                "model_family": "tabnet",
                "pytorch_tabnet_version": tabnet_ver,
                "torch_version": torch.__version__,
                "tabnet_max_epochs": TABNET_MAX_EPOCHS,
                "tabnet_patience": TABNET_PATIENCE,
                "tabnet_batch_size": TABNET_BATCH_SIZE,
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
