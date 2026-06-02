# Extra Trees 模型驗證 + 6-Tree Canonical Comparison Framework 報告(2026-05-29)

**Model**:Extra Trees(Extremely Randomized Trees / sklearn ExtraTreesRegressor)
**sklearn version**:1.8.0
**Trainer**:`scripts/core/model_trainer_extra_trees.py`(v0.1 / 343 行;§一.11 三段式合規)
**Multi-cycle Validator**:`scripts/evaluation/multi_cycle_extra_trees_validation.py`(v0.1 / 282 行;§一.11 三段式合規)
**治權對標**:§14.7-CW Tree Family 第六實作 / §14.7-CX 8-year OOS / §14.7-CY 4-horizon validation / §14.7-CZ T_CZ-6 reality check / **per Canonical Comparison Framework**(per RF v0.1 建立)
**Source compliance**:per CLAUDE.md §一.10 — 全 (b) DB query + (a) program output / 0 AI memory
**Pairwise ablation**:本報告為 **ET vs RF bagging 家族 pairwise ablation** 之核心揭露 — 揭露 random-split vs best-split-within-subset 之 architectural effect

---

## ⭐ 一、Canonical Comparison Framework(per 用戶 directive「相同的比較基準定義」)

本 framework **per Random Forest v0.1 report 首次建立**,作為 future model 對比 reliable 的標準基準。Extra Trees v0.1 為 framework 的第 6 個對齊實作。

### 1.1 Same Data Foundation(相同資料基礎)

| 元素 | Standardized 值 | 治權契約 |
|---|---|---|
| Features | **SPEC_43**(43 canonical features)| §14.7-CL |
| Universe | **Latest committed core_universe**(N=1,121)| §14.7-CF |
| Historical panels | **95 monthly fs_v0_4 snapshots**(2018-06 ~ 2026-04)| §14.7-CX |
| Forward returns | **LN(t1/t0)** from TaiwanStockPriceAdj | §14.7-CV |
| Anti-leakage | **§8.5 publication_date_strategy** | §8.5 / §14.7-CB |

### 1.2 Same Evaluation Protocol(相同評估流程)

| 元素 | Standardized 值 |
|---|---|
| Window scheme | **Expanding window walk-forward OOS** |
| 8-panel commit run | for Treaty Gates baseline |
| 95-panel multi-cycle | for production reality |
| 4 horizons | **weekly(5d)/ monthly(20d)/ quarterly(60d)/ annual(252d)** |
| Top-20 strategy | **equal-weight top-20** by prediction rank |
| Cost model | **0.6% per rebalance**(TW standard broker)|

### 1.3 Same Standard Metrics(相同標準指標)

| Category | Metrics |
|---|---|
| **Treaty Gates 4(§14.7-CW)** | Sharpe / Win rate / MDD / Mean α |
| **Multi-Cycle(§14.7-CY)** | Eff t-stat(n_eff corrected)/ Sharpe / NetAnn / IR |
| **NEW Precision** | Directional Hit Rate / Top-20 Actual Overlap / RMSE / MAE |
| **NEW Reliability** | IC Stability CoV / Significance Robustness |
| **Aggregate** | Cross-panel IC mean / Cumulative return |

### 1.4 Same Hyperparameter Philosophy(per §14.7-CW T_CW-4)

| Hyperparameter | Standardized 值(全 6 模型統一)|
|---|---|
| n_estimators | **200** |
| max_depth | **5** |
| seed / random_state | **5422** |
| Regularization | **Conservative**(min_samples 30 / l2_leaf_reg 3 / reg_alpha 0.1 / reg_lambda 0.1)|

### 1.5 Same Report Template

本報告完全對齊 RF v0.1 之 10-section 結構。

---

## 二、Extra Trees 模型做法

### 2.1 架構說明

**Extra Trees(Extremely Randomized Trees)** 為 Geurts et al. 2006 提出之 bagging 家族變體:
- **完全隨機 split threshold**:每 split 在隨機選 feature 上之 **隨機 threshold**(不搜索 best)
- **Feature subsampling**:同 RF 之 `sqrt(n_features)`
- **Bootstrap aggregating**:本程式 `bootstrap=True`(與 RF 對等;sklearn ET 預設 bootstrap=False)
- **Variance reduction++**:隨機 split 比 RF 之 best-split-within-subset 引入更多 randomness → 更低 variance / 略高 bias

### 2.2 ET vs RF 架構差異(bagging 家族 pairwise ablation 之核心)

| 維度 | Extra Trees | Random Forest |
|---|---|---|
| Split 方式 | **隨機 threshold**(no search)| Best split within random subset |
| Bias vs Variance | **更低 variance / 略高 bias** | 平衡 |
| Speed | **更快**(無 best-split search)| 較慢 |
| Diversity | **更高**(更隨機)| 中等 |
| Overfit tendency | **更低** | 低 |

### 2.3 ET vs GBT Family

| 維度 | Extra Trees | GBT Family(LGBM/XGB/CatBoost)|
|---|---|---|
| Strategy | **Bagging + random split** | **Boosting**(sequential)|
| Architectural orthogonality | 與 GBT 完全互補 | — |
| Tuning | Less sensitive | More sensitive |

### 2.4 Hyperparameters(per §14.7-CW T_CW-4)

```python
{
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_leaf": 30,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 5422,
    "n_jobs": -1,
}
```

### 2.5 訓練資料 source(per §一.10 全 (b) DB query)

| Layer | 真實 source | 行數 |
|---|---|---|
| Universe | `core_universe_membership` WHERE policy=v0.15 | 1,121 stocks |
| Features | `feature_values` WHERE feature_set_id=fs_v0_4 | 4.7M rows × 43 features |
| Forward returns | `TaiwanStockPriceAdj` LN(t1/t0)| 真實 close price ratios |
| Historical panels | 2018-06 ~ 2026-04 monthly | 95 panels |

---

## 三、8-Panel Walk-Forward(commit run)

**Trainer command**:`python scripts/core/model_trainer_extra_trees.py --commit`

**Source**:`data/models/mdl_20260415_extra_trees_h30_0b243a67_v0_1/metrics.json`(per §一.10 (a) program output)

| 指標 | 真實值 | Treaty Gate |
|---|---|---|
| Cross-panel IC mean | **+0.1894** | — |
| Cross-panel IC std | 0.2120 | — |
| In-sample IC | +0.2742 | — |
| **Overfit gap**(in - OOS)| **+0.0848** ⭐(最小 / **比 RF 0.175 還小**!)| acceptable |
| Sharpe(annualized)| **+3.486** | ✅ Gate CW-1 PASS |
| Win rate | **83.3%** | ✅ Gate CW-2 PASS |
| **MDD** | **2.17%** | ✅ Gate CW-3 PASS |
| Mean alpha / 30d | **+9.35%** | ✅ Gate CW-4 PASS |
| Information Ratio | **+4.395** | — |
| t-statistic(α)| +3.107 | — |
| Cumulative return | +65.72% | — |
| **Treaty Gates 4/4** | **PASS** | **主權判定 PERFECT** |

**Model artifact**:`data/models/mdl_20260415_extra_trees_h30_0b243a67_v0_1/`(model.pkl + metrics.json + hyperparams.json)

### 3.1 ET 8-panel 特性揭露

⭐ **Overfit gap 0.085 為 6 models 中最小** — random-split bagging 為極強 anti-overfit
⭐ **Sharpe 3.49 為 bagging 家族最佳**(對比 RF 3.25)
⚠️ **MDD 2.17% 較 RF 0.10% 高 22 倍** — random-split 雖更 variance-reducing,但 path-level robustness 略差
⚠️ **Multi-thread non-determinism**:dry-run Sharpe=3.36 vs commit Sharpe=3.49(差 0.13);per §一.10 #3 須 ≥3 runs 取得 distribution(本 single run 為 commit anchor,須 reproducibility patch 補完整 range)

---

## 四、Multi-Cycle 4-Horizon Walk-Forward(95 panels × 4 horizons)

**Validator command**:
```bash
python scripts/evaluation/multi_cycle_extra_trees_validation.py \
    --horizons 5,20,60,252 \
    --output reports/multi_cycle_extra_trees_20260529.json
```

**Total elapsed**:274.3s(4.6 min,**比 RF 470s 快 41%** — random-split 加速 fitting)

**Source**:`reports/multi_cycle_extra_trees_20260529.json`(per §一.10 (a) program output)

### 4.1 Cross-Cycle Comparison Matrix(per §14.7-CY)

| Horizon | Days | N | n_eff | Eff t | Sig p<0.05 | Sharpe | Net Annual | Hit Rate | Top-20 Overlap |
|---|---|---|---|---|---|---|---|---|---|
| weekly | 5 | 65 | 65.0 | +0.902 | ❌ | 0.657 | +6.29% | 50.7% | 6.4% |
| monthly | 20 | 65 | 65.0 | +1.428 | ❌ | 0.921 | +18.32% | 51.8% | 6.5% |
| **quarterly** | **60** | **64** | **32.0** | **+0.836** ❌ | **❌** | **1.237** | **+8.33%** | 51.0% | **3.1%** |
| annual | 252 | 61 | 7.3 | +2.306 | ✅ | **3.713** | **+17.05%** | **59.7%** | 5.8% |

### 4.2 §14.7-CZ T_CZ-6 Reality Check(quarterly)

| 指標 | Required | **Extra Trees** |
|---|---|---|
| Eff t-stat | ≥ 4.20 | **+0.836 ❌**(差 3.36 / **80% below threshold**)|
| Sharpe | ≥ 2.40 | **1.24 ❌**(差 1.16)|
| Win rate | ≥ 79% | **68.8% ❌** |

❌ **Extra Trees 嚴重不達 T_CZ-6 quarterly production threshold**(遠不及 RF 之 2.47,亦不及任何 GBT family)。

### 4.3 ET 多週期信度發現(per Canonical Reliability Analysis)

| Horizon | IC CoV(reliability)| 解讀 |
|---|---|---|
| weekly | **7.237** | 極不 stable |
| monthly | **29.603** | 極不 stable(monthly IC 接近 0)|
| quarterly | 4.878 | 不 stable |
| annual | 1.185 | 中等 stable |

---

## 五、Top-15 Feature Importance(Extra Trees MDI)

**Source**:`data/models/mdl_20260415_extra_trees_h30_0b243a67_v0_1/metrics.json`(per §一.10 (a))

| Rank | Feature | Importance | 三基柱歸屬 |
|---|---|---|---|
| 1 | **right_tail_concentration_60d** | **0.1736** | **§0.2 八二法則** |
| 2 | upside_capture_60d | 0.0951 | §0.1 |
| 3 | **barbell_balance_60d** | 0.0831 | **§0.2** |
| 4 | volatility_60d | 0.0729 | §0.1 |
| 5 | downside_volatility_60d | 0.0651 | §0.1 |
| 6 | upside_volatility_60d | 0.0634 | §0.1 |
| 7 | liquidity_rank_pct_sector_60d | 0.0562 | §0.1 microstructure |
| 8 | downside_capture_60d | 0.0537 | §0.1 |
| 9 | avg_daily_value_log_60d | 0.0454 | §0.1 microstructure |
| 10 | size_log_zscore_sector | 0.0385 | §0.1 |
| 11 | **preferential_attachment_60d** | 0.0370 | **§0.2** |
| 12 | volatility_252d | 0.0328 | §0.1 |
| 13 | **fitness_signal_60d** | 0.0280 | **§0.2** |
| 14 | avg_daily_value_log_252d | 0.0203 | §0.1 |
| 15 | log_return_60d | 0.0186 | §0.1 |

**§14.7-CN 對齊**:Top-15 中 §0.1 = 11 / §0.2 = 4 / §0.3 = 0 ✅ — 與 RF 完全相同分布

**ET vs RF Top-15 對比**:極相近(均由 right_tail_concentration_60d 主導),確認 bagging 家族對 feature 之共識較高(vs GBT 之 horizon-sensitive top features)

---

## 六、🎯 Precision Analysis(per Canonical Framework)

### 6.1 Three Precision Metrics(per multi_cycle_extra_trees JSON)

| Horizon | Hit Rate(方向)| Top-20 Overlap(精準)| RMSE | MAE |
|---|---|---|---|---|
| weekly | 50.7% | 6.4% | 0.043 | 0.028 |
| monthly | 51.8% | 6.5% | 0.082 | 0.056 |
| **quarterly** | 51.0% | **3.1%** ⚠️ | 0.141 | 0.099 |
| annual | **59.7%** ⭐ | 5.8% | 0.293 | 0.211 |

### 6.2 ET vs RF Precision 對比(bagging 家族)

| 指標 | Extra Trees | Random Forest | Verdict |
|---|---|---|---|
| Quarterly Hit Rate | 51.0% | 50.1% | **ET +0.9pp** |
| Quarterly Top-20 Overlap | 3.1% | 2.5% | **ET +0.6pp**(微優)|
| Annual Hit Rate | 59.7% | 60.4% | RF +0.7pp |
| Annual Top-20 Overlap | 5.8% | 6.0% | RF +0.2pp |

⚠️ **重大 honest insight**:**ET quarterly top-20 overlap 僅 3.1%** — 20 picks 中**僅 0.6 支真在 actual top-20**(quarterly random expected = 1.78%,ET 僅 1.7× over random)。略優於 RF 之 1.4× over random,但仍**遠遜於 production threshold**。

### 6.3 ET 路徑型 metrics

| Horizon | mean panel MDD |
|---|---|
| weekly | 17.3% |
| monthly | — |
| quarterly | — |
| annual | — |

⚠️ ET multi-cycle MDD per panel(weekly 17.3%)較高,反映完全隨機 split 在短期 horizon 下 path 不穩定。

---

## 七、🎯 Reliability Analysis(per Canonical Framework)

### 7.1 IC Stability(CoV)

| Horizon | ET IC CoV | RF IC CoV | 解讀 |
|---|---|---|---|
| weekly | 7.237 | — | ET 極不 stable |
| monthly | 29.603 | — | ET monthly IC 近 0 → CoV 極大 |
| quarterly | 4.878 | — | ET 不 stable |
| annual | 1.185 | 0.572 | RF annual 較 stable |

### 7.2 Significance Robustness 對比(bagging 家族)

| Horizon | ET Eff t | RF Eff t | Comparison |
|---|---|---|---|
| weekly | +0.902 | +1.760 | **RF +0.86 stronger** |
| monthly | +1.428 | +1.132 | ET +0.30 |
| **quarterly** | **+0.836** | **+2.471** | **RF +1.64 stronger** ⭐ |
| annual | +2.306 | +2.881 | RF +0.58 stronger |

### 7.3 ET vs RF 信度結論

⭐ **Random Forest 在 multi-cycle 顯著優於 Extra Trees**:
- RF best-split-within-subset 保留必要 signal
- ET 完全隨機 split 過度消除 signal
- 此為 bagging 家族 pairwise ablation 之 **核心 empirical finding**

❌ ET 之「進一步 randomness」**未轉化為更佳 generalization** — 反之,在 multi-cycle 完全劣於 RF

---

## 八、🏆 6-Tree Model Final Comparison(per Canonical Framework)

### 8.1 Quarterly Horizon Comparison(production 主軸)

| Model | Eff t | Sharpe | NetAnn | Hit Rate | Top-20 Overlap | T_CZ-6 | Architecture |
|---|---|---|---|---|---|---|---|
| **LGBM v0.2** | 4.20 | 2.55 | +24.44% | — | — | **✅** | **Boosting(GBT)** |
| **XGBoost v0.1** | **4.36** ⭐ | 2.63 | **+29.35%** ⭐ | — | — | **✅** ⭐ | **Boosting(GBT)** |
| **CatBoost v0.1** | 3.65 | 2.30 | +20.22% | 52.0% | 5.0% | ❌ | **Boosting(GBT)** |
| **Ensemble v0.1** | 4.14 | **2.68** | +23.46% | 52.0% | 5.0% | ⚠️ | **Equal-weight 3 GBT** |
| **Random Forest v0.1** | 2.47 ❌ | 1.81 | +14.05% | 50.1% | 2.5% | ❌ | **Bagging(best-split)** |
| **Extra Trees v0.1** | **0.836** ❌❌ | 1.24 | +8.33% | 51.0% | **3.1%** | ❌ | **Bagging(random-split)** |

### 8.2 8-Panel Sharpe + MDD + Overfit Gap Comparison

| Model | Sharpe | MDD | Overfit Gap |
|---|---|---|---|
| LGBM v0.2 | 3.84 | 2.52% | 0.366 |
| XGBoost v0.1 | **4.58** ⭐ | 2.77% | 0.426 |
| CatBoost v0.1 | 4.29 | 3.07% | 0.246 |
| Ensemble v0.1 | 3.98 | 3.60% | — |
| Random Forest v0.1 | 3.25 | **0.10%** ⭐⭐ | 0.175 |
| **Extra Trees v0.1** | **3.49** | 2.17% | **0.085** ⭐⭐(最小)|

### 8.3 6-Tree Ranking 總結

| Rank | Best at | Model |
|---|---|---|
| 🥇 | **Quarterly production**(T_CZ-6 pass)| **XGBoost** |
| 🥈 | Weekly high-frequency(only sig)| CatBoost |
| 🥉 | 30d baseline(LGBM v0.2 = §14.7-CW production)| LGBM |
| 4 | Annual + high reliability | Ensemble |
| 5 | **Best 8-panel MDD**(0.10%) | Random Forest |
| 6 | **Lowest overfit gap**(0.085) | **Extra Trees** |

### 8.4 Bagging vs Boosting 家族 pairwise 對比

| 維度 | **Bagging(RF + ET)** | **Boosting(GBT 4 個)** |
|---|---|---|
| Quarterly Eff t 平均 | 1.65 | 4.09 ⭐ |
| Quarterly Sharpe 平均 | 1.53 | 2.54 ⭐ |
| Best Overfit Gap | 0.085(ET)⭐ | 0.246(CatBoost) |
| Best MDD | 0.10%(RF)⭐ | 2.52%(LGBM) |
| **Verdict** | **保守 + 防 overfit** | **production 生產主軸** |

### 8.5 Bagging 家族 pairwise ablation(ET vs RF)— 核心 empirical 揭露

| 維度 | Extra Trees | Random Forest | Architectural Insight |
|---|---|---|---|
| Split 方式 | **完全隨機** | Best-split-within-subset | RF 保留必要 signal |
| Quarterly Eff t | **0.836** | **2.47** | **RF 顯著 +1.64** |
| Quarterly Sharpe | 1.24 | 1.81 | RF +0.57 |
| 8-panel Sharpe | 3.49 | 3.25 | ET +0.24 |
| 8-panel MDD | 2.17% | 0.10% | **RF 顯著更佳** |
| 8-panel Overfit | 0.085 | 0.175 | ET 更佳 |
| Multi-cycle 速度 | 274s ⭐ | 470s | ET +41% 更快 |

⭐ **核心 empirical conclusion**:在本系統 TW 數據 +43 features + 95 panels 之 multi-cycle setting 下:
- **「best-split-within-subset(RF)優於完全隨機 split(ET)」**(quarterly Eff t +1.64)
- **隨機性過頭反失去 signal**,ET 之 8-panel 看似強(Sharpe 3.49)為 small-sample illusion
- ET 在 8-panel 之低 overfit gap(0.085)為 mean-prediction 收斂幅度差導致,non-informative

---

## 九、賺錢能力裁決 — Extra Trees

### 9.1 三層裁決(per CLAUDE.md §一.10 honest)

| 層 | ET 答案 |
|---|---|
| 1. 統計上 ET 有 alpha?(8-panel) | ✅ **YES**(commit t=3.11)|
| 2. 統計上 ET 有 alpha?(multi-cycle quarterly) | ❌ **NO**(Eff t=0.836)|
| 3. Walk-forward 會賺?(95-panel) | ✅(annual +17.05%/yr)but weak elsewhere |
| 4. 達 §14.7-CZ T_CZ-6 production? | ❌❌ **嚴重不達**(Eff t 0.836 << 4.20)|
| 5. 比 RF 好? | ❌ **顯著不及**(quarterly Eff t -1.64)|
| 6. ET 獨特優勢? | ⭐ **Overfit gap 0.085(最小)+ 速度最快**|

### 9.2 ET 適用場景

| 場景 | 推薦 ET? |
|---|---|
| Production prediction(quarterly)| ❌❌(嚴重不及)|
| **Anti-overfit baseline**(極 small datasets)| ⭐ **YES** |
| **快速 prototyping**(速度優勢)| ⭐ **YES** |
| Risk-averse portfolio(超低 MDD)| ❌(RF 更佳)|
| Multi-model variance source | ✅(diverse architecture)|

### 9.3 Honest caveats(per §一.10)

1. **Multi-thread non-determinism**:dry-run Sharpe 3.36 vs commit Sharpe 3.49 / 差 0.13(per §一.10 #3 須 ≥3 runs;本 commit anchor 為 single run / 須 reproducibility patch 補完整 range)
2. **ET 8-panel sharpe 看似比 RF 高,但 multi-cycle 顯著劣於 RF**(8-panel 為 small-sample illusion)
3. **Quarterly Eff t 0.836 < 1.0** = 接近 noise threshold(< 1 standard error)
4. **Top-20 overlap 3.1% quarterly** = 接近 random(20/1121 = 1.78%,僅 1.7× over random)
5. **Bagging 家族總體不及 GBT 家族**(per §14.7-CW production doctrine)

---

## 十、Charter Compliance + Source Traceability

### 10.1 Treaty compliance

| Treaty | 狀態 |
|---|---|
| §14.7-CW T_CW-1 Real tree | ✅(sklearn ExtraTreesRegressor)|
| T_CW-2 Expanding window | ✅ |
| T_CW-3 §0.1+§0.2 features dominated | ✅(11 §0.1 + 4 §0.2)|
| T_CW-4 Conservative params | ✅(200/5/sqrt/5422)|
| T_CW-5 Gates 4/4 PASS | ✅(8-panel) |
| T_CW-6 Multi-run | ⚠️ single run(dry-run/commit 差 0.13 揭露 non-determinism)|
| §14.7-CY T_CY-1 System script | ✅ |
| §14.7-CY T_CY-2-5 Multi-cycle | ✅ |
| §14.7-CY T_CY-6 Recommended | ❌ quarterly 嚴重不及 |
| **§14.7-CZ T_CZ-6 Reality Check** | **❌❌ 嚴重不達** |
| §一.10 Source-traceable | ✅ |
| §一.11 三段式合規 | ✅ Both scripts |
| **Canonical Comparison Framework** | ✅ **完全對齊 RF v0.1** |

### 10.2 Source Traceability(per §一.10)

| 數字 | Source |
|---|---|
| 8-panel commit metrics | `data/models/mdl_20260415_extra_trees_h30_0b243a67_v0_1/metrics.json` |
| Multi-cycle log | `/tmp/et_multicycle.log` |
| Multi-cycle JSON | `reports/multi_cycle_extra_trees_20260529.json` |
| Model artifact | `data/models/mdl_20260415_extra_trees_h30_0b243a67_v0_1/` |
| DB model_registry | `mdl_20260415_extra_trees_h30_0b243a67_v0_1` status=committed |
| 比對 5 models 數字 | 各 model 之 `data/models/<id>/metrics.json` + `reports/multi_cycle_*_20260529.json` |

### 10.3 §一.11 三段式合規驗證

| Script | 三段式 |
|---|---|
| `scripts/core/model_trainer_extra_trees.py` | ✅ 標頭 14 Core Definitions(含 [Sovereignty Declaration]、[Canonical Comparison Framework]、[ET vs RF Pairwise Ablation])+ 全量功能群矩陣 A-F + 全修訂歷程 v0.1 |
| `scripts/evaluation/multi_cycle_extra_trees_validation.py` | ✅ 標頭 14 Core Definitions(同上)+ 全量功能群矩陣 A-F + 全修訂歷程 v0.1 |

---

## 十一、結論(6-Tree Canonical Comparison Framework 階段性)

### 11.1 ET v0.1 production 判定

❌ **Extra Trees 不適合本系統 production prediction**:
- Quarterly Eff t=0.836 < 4.20(嚴重不達)
- Multi-cycle 顯著劣於 RF(同為 bagging 家族,RF 更優)
- 完全隨機 split 過度消除 signal

⭐ **Extra Trees 之獨特價值**:
- **Overfit gap 最小**(0.085,6 models 中)
- **速度最快**(multi-cycle 274s,RF 470s 之 58%)
- **Bagging 家族 pairwise ablation 之重要對照**

### 11.2 6-Tree Canonical Comparison Framework 成熟度

✅ **Framework 已驗證 6 個 architecturally distinct models**:
- 4 個 Boosting(LGBM / XGBoost / CatBoost / Ensemble)
- 2 個 Bagging(RF / ET)

✅ **Future model 對比 reliable** — 同 data foundation / 同 protocol / 同 metrics / 同 hyperparams / 同 template,per Lopez de Prado backtest comparison standards

### 11.3 Production 推薦

| 用途 | 推薦 Model |
|---|---|
| **Quarterly production**(最優)| **XGBoost v0.1** |
| 30d baseline(已 production)| LGBM v0.2(per §14.7-CW)|
| Weekly high-frequency | CatBoost v0.1 |
| Risk-averse(超低 MDD)| Random Forest v0.1 |
| Multi-model ensemble diversifier | Extra Trees v0.1(architecturally distinct)|

---

**Report 完成時間**:2026-05-29 11:11
**Model ID**:`mdl_20260415_extra_trees_h30_0b243a67_v0_1`
**Author**:Codex(AI)/ 治權對標:CLAUDE.md §一.10 + §一.11
