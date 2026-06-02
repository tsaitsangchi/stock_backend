# Random Forest Multi-Cycle 驗證報告(模型 #5)

**模型**：Random Forest（200 trees / depth 5 / max_features=sqrt / bootstrap）| **類別**：Tree（bagging）
**最後更新**：2026-06-02
**比較基準**：依 `common_model_comparison_baseline_20260602.md`（同 universe / 窗 / horizon / metric / gate）
**Metric 來源**：`core.db_utils.summarize_horizon_metrics()`（§14.7-DF 單一源，與其他模型同 helper）
**資料來源**：全 (b) DB query（§一.10）| **Seeds**：5422 / 1009 / 7331（median + range）

---

## 一、4-Horizon × 3-Seed 聚合（median）

| Horizon | net 年化 | Sharpe | Eff t ⭐ | Win | IC | IR | MDD | 命中 | top-20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| weekly 5d | +15.1% | 0.919 | 2.023 | 66.0% | **−0.000** | 0.681 | 0.165 | 51.4% | 9.8% |
| monthly 20d | +25.7% | 1.203 | 3.423 | 66.0% | 0.017 | 1.152 | 0.345 | 54.0% | 11.2% |
| quarterly 60d | +20.8% | 2.126 | 5.789 | 76.2% | 0.080 | 2.768 | 0.392 | 51.9% | 11.0% |
| **annual 252d** | **+23.2% [22.6–23.3]** | **3.927 [3.79–4.00]** | **4.566** | **88.2%** | **0.195** | **4.539** | 0.722 | 59.9% | 12.6% |

> ⚠️ **weekly IC ≈ 0**（−0.0001）→ RF 在短週期**幾乎無排序訊號**（IC-CoV 因此爆大、無意義）。IC 隨 horizon 上升較其他樹**慢**（annual 0.195,為樹族最低）。

## 二、T_CZ-6 + 賺錢裁決

| Horizon | Eff t | Sharpe | Win | Gate | 裁決 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| weekly | 2.023 ❌ | 0.919 ❌ | 66.0% ❌ | 🔴 | 🟠 微弱（IC≈0 無 edge）|
| monthly | 3.423 ❌ | 1.203 ❌ | 66.0% ❌ | 🔴 | 🟡 獲利未達門檻 |
| quarterly | 5.789 ✅ | 2.126 ❌ | 76.2% ❌ | 🔴 | 🟡 獲利未達門檻 |
| **annual** | **4.566 ✅** | **3.927 ✅** | **88.2% ✅** | 🟢 **PASS** | 🟢 **Production-grade**（net +23.2%/yr）|

> **能賺錢 — annual production-grade**（剛過 gate:Sharpe 3.93、eff_t 4.57）,但為樹族最弱。

---

## 三、五模型 Head-to-Head（annual median，同基準）

| metric | LightGBM | XGBoost | CatBoost | Ensemble | **RandomForest** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **net 年化** | +30.2% | **+32.7%** | +26.3% | +30.3% | +23.2% |
| **Sharpe** | 4.62 | **4.63** | 4.22 | 4.59 | 3.93 |
| **Eff t** | **6.02** | 5.77 | 5.05 | 5.94 | 4.57 |
| **IC** | 0.276 | **0.297** | 0.238 | 0.282 | 0.195 |
| **T_CZ-6** | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |

### 關鍵發現（誠實）
> **Boosting > Bagging**:RF（bagging,annual Sharpe 3.93 / IC 0.195）為**五模型最弱**,明顯遜於 boosting 樹（XGB/LGBM ~4.6 / IC ~0.28）。意義:在這組 37 source-pure 特徵上,**梯度提升的逐步殘差擬合比隨機森林的平均 bagging 更能抽出排序訊號**。RF 仍過 annual T_CZ-6,但若要選一個,boosting 明顯較佳。**五模型 annual 全過 gate** → 「年線可賺錢」結論穩健(非單一模型僥倖)。

---

## 四、來源 + 重現
- 數字 trace 至 `/tmp/rf_ref_s{5422,1009,7331}.json`（全 (b) DB query；§14.7-DF helper）。

```bash
for s in 5422 1009 7331; do
  ./venv/bin/python scripts/evaluation/multi_cycle_random_forest_validation.py --commit --seed $s --output /tmp/rf_ref_s${s}.json
done
```

> 進度:tree-family **5/6 報告完成**（LGBM/XGB/CatBoost/Ensemble/RF）；ET #6 跑中,transformer_dedicated #7 待續。
