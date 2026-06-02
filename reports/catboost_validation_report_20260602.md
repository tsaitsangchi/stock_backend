# CatBoost Multi-Cycle 驗證報告(模型 #3)

**模型**：CatBoost(CatBoostRegressor / 200 iterations / depth 5 / lr 0.05)| **類別**：Tree
**最後更新**：2026-06-02
**比較基準**：依 `common_model_comparison_baseline_20260602.md`(同一 universe / 窗 / horizon / metric / gate)
**Metric 來源**：`core.db_utils.summarize_horizon_metrics()`(§14.7-DF 單一計算源 — 與 LightGBM/XGBoost **同一 helper**,完全可比)
**資料來源**：全 (b) DB query(§一.10)| **Seeds**：5422 / 1009 / 7331(median + range，§一.10 #3)

---

## 一、運行配置(= 共同比較基準)
397 source-pure 核心股 × `feature_set_v0.5` 37 特徵 × 157 panels(2013-05-15~2026-06-01,`get_canonical_panel_dates()` §14.7-DE)× 4 horizon × top-20 等權多頭 × 成本 0.006/rebal。

## 二、4-Horizon × 3-Seed 聚合(median，[min–max])

| Horizon | net 年化 | Sharpe | Eff t ⭐ | Win | IC | IR | MDD/panel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| weekly 5d | +11.7% [8.8–14.6] | 0.965 [0.86–0.98] | 1.948 | 67.9% | 0.010 | 0.655 | 0.153 |
| monthly 20d | +27.7% [26.8–31.5] | 1.294 [1.26–1.43] | 3.948 | 63.2% | 0.037 | 1.328 | 0.291 |
| quarterly 60d | +21.3% [20.6–21.4] | 2.097 [2.07–2.14] | 6.127 | 73.3% | 0.103 | 2.929 | 0.413 |
| **annual 252d** | **+26.3% [25.8–26.6]** | **4.215 [4.13–4.25]** | **5.054** | **89.2%** | **0.238** | **5.024** | 0.539 |

## 三、精準度 + 信任度

| Horizon | 方向命中 | top-20 重疊 | IC-CoV | |
| :--- | :--- | :--- | :--- | :--- |
| weekly | 50.4% | 9.6% | 12.45 | 無 edge |
| monthly | 52.5% | 10.8% | 3.09 | 微弱 |
| quarterly | 52.4% | 10.5% | 0.83 | 中等 |
| **annual** | **61.6%** | **14.8%** | **0.43** | 明顯 edge + 穩定 |

跨 seed 穩定:annual net 僅 25.8–26.6%、Sharpe 4.13–4.25 → 隨機性低,可信。

## 四、T_CZ-6 Gate + 賺錢裁決

| Horizon | Eff t | Sharpe | Win | Gate | 裁決 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| weekly | 1.948 ❌ | 0.965 ❌ | 67.9% ❌ | 🔴 | 🟠 微弱 |
| monthly | 3.948 ❌ | 1.294 ❌ | 63.2% ❌ | 🔴 | 🟡 獲利未達門檻(net +27.7%、eff_t 顯著) |
| quarterly | 6.127 ✅ | 2.097 ❌ | 73.3% ❌ | 🔴 | 🟡 獲利未達門檻(net +21.3%) |
| **annual** | **5.054 ✅** | **4.215 ✅** | **89.2% ✅** | 🟢 **PASS** | 🟢 **Production-grade**(net +26.3%/yr) |

### 總裁決
> **能賺錢 — annual 達 production-grade**(net +26.3%、Sharpe 4.22、Win 89.2%、過 T_CZ-6)。Monthly 顯著獲利(net +27.7%、eff_t 3.95 為三樹最高)。Weekly 無 edge。

---

## 五、三模型 Head-to-Head(#1 #2 #3,同基準,annual median)

| metric | LightGBM | XGBoost | CatBoost |
| :--- | :--- | :--- | :--- |
| **net 年化** | +30.2% | **+32.7%** | +26.3% |
| **Sharpe** | 4.62 | **4.63** | 4.22 |
| **Eff t** | **6.02** | 5.77 | 5.05 |
| **Win** | 89.2% | **90.2%** | 89.2% |
| **IC** | 0.276 | **0.297** | 0.238 |
| **T_CZ-6** | 🟢 | 🟢 | 🟢 |

**monthly 比較**(CatBoost 此處最強):Eff t — LGBM 3.48 / XGB 3.02 / **CatBoost 3.95**;Sharpe — 1.256 / 1.178 / **1.294**。

> **三樹模型 annual 皆過 T_CZ-6,排名 XGBoost ≳ LightGBM > CatBoost**(net/Sharpe)。差距不大,且趨勢一致(annual 最強、weekly 無 edge),再次印證**訊號來自資料 + 中長週期排序**,非單一模型。CatBoost 在 monthly 略勝。

---

## 六、來源 + 重現

- 數字 trace 至 `/tmp/cb_ref_s{5422,1009,7331}.json`(全 (b) DB query;metric 由 §14.7-DF 共用 helper)。
- ⚠️ 非決定性(§一.10 #3 / T_DF-4):3 seed 取 range/median。

```bash
for s in 5422 1009 7331; do
  ./venv/bin/python scripts/evaluation/multi_cycle_catboost_validation.py --commit --seed $s --output /tmp/cb_ref_s${s}.json
done
```

> 進度:tree-family 已完成 3/9 報告(LightGBM/XGBoost/CatBoost);下一步可續推 Ensemble / Random Forest / Extra Trees → 再 transformer/foundation → master comparison。
