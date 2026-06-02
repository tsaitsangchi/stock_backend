# Ensemble Multi-Cycle 驗證報告(模型 #4)

**模型**：三樹集成(LightGBM + XGBoost + CatBoost 預測平均)| **類別**：Tree-ensemble
**最後更新**：2026-06-02
**比較基準**：依 `common_model_comparison_baseline_20260602.md`（同 universe / 窗 / horizon / metric / gate）
**Metric 來源**：`core.db_utils.summarize_horizon_metrics()`（§14.7-DF 單一源，與其他模型同 helper）+ ensemble 獨有 `reliability_ensemble_disagreement`（三模型分歧度，common keys 之外保留）
**資料來源**：全 (b) DB query（§一.10）| **Seeds**：5422 / 1009 / 7331（median + range）

> ⚠️ **bug-fix 揭露（§一.8）**：本模型 validator 於 v6.26.3 metric-helper 重構時引入 2 個 regression（KeyError `reliability_ensemble_disagreement` + per-stock 陣列 ValueError），已修正（收集 disagree → `float(np.mean())` 加回 result）+ weekly smoke 驗證通過後重跑。**修正 uncommitted**，下個 checkpoint 一併 commit。

---

## 一、4-Horizon × 3-Seed 聚合（median，[min–max]）

| Horizon | net 年化 | Sharpe | Eff t ⭐ | Win | IC | 三模型分歧度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| weekly 5d | +8.0% | 0.859 | 1.375 | 66.0% | 0.009 | 0.0021 |
| monthly 20d | +26.1% | 1.244 | 3.450 | 66.0% | 0.041 | 0.0044 |
| quarterly 60d | +23.9% | 2.213 | 6.945 | 75.2% | 0.121 | 0.0082 |
| **annual 252d** | **+30.3% [30.0–30.7]** | **4.589 [4.54–4.64]** | **5.939** | **90.2%** | **0.282** | 0.0192 |

- **分歧度遞增**:三模型在長 horizon 預測差異變大(weekly 0.2% → annual 1.9%)→ 長期預測不確定性較高。
- annual 跨 seed 極穩（net 30.0–30.7%）；annual precision: 命中 62.7% / top-20 重疊 17.1% / IC-CoV 0.389。

## 二、T_CZ-6 + 賺錢裁決

| Horizon | Eff t | Sharpe | Win | Gate | 裁決 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| weekly | 1.375 ❌ | 0.859 ❌ | 66.0% ❌ | 🔴 | 🟠 微弱 |
| monthly | 3.450 ❌ | 1.244 ❌ | 66.0% ❌ | 🔴 | 🟡 獲利未達門檻 |
| quarterly | 6.945 ✅ | 2.213 ❌ | 75.2% ❌ | 🔴 | 🟡 獲利未達門檻(eff_t 最高) |
| **annual** | **5.939 ✅** | **4.589 ✅** | **90.2% ✅** | 🟢 **PASS** | 🟢 **Production-grade**(net +30.3%/yr) |

> **能賺錢 — annual production-grade**(net +30.3%、Sharpe 4.59、Win 90.2%)。quarterly eff_t 6.945 為四模型最高。

---

## 三、四模型 Head-to-Head（annual median，同基準）

| metric | LightGBM | XGBoost | CatBoost | **Ensemble** |
| :--- | :--- | :--- | :--- | :--- |
| **net 年化** | +30.2% | **+32.7%** | +26.3% | +30.3% |
| **Sharpe** | 4.62 | **4.63** | 4.22 | 4.59 |
| **Eff t** | **6.02** | 5.77 | 5.05 | 5.94 |
| **Win** | 89.2% | **90.2%** | 89.2% | **90.2%** |
| **IC** | 0.276 | **0.297** | 0.238 | 0.282 |
| **T_CZ-6** | 🟢 | 🟢 | 🟢 | 🟢 |

### 關鍵發現（誠實）
> **集成沒有打敗最佳單一模型**:Ensemble annual（Sharpe 4.59 / net +30.3%）≈ LightGBM、**略遜於 XGBoost**（4.63 / +32.7%）—— 因它是三者平均,被較弱的 CatBoost（4.22）拉低。這是**真實且重要的結論**:在這組高度相關的樹模型上,簡單平均集成**不優於選最強單一模型**。Ensemble 的價值在 quarterly（eff_t 6.945 最高）與穩定性,非 annual 報酬。

---

## 四、來源 + 重現
- 數字 trace 至 `/tmp/ens_ref_s{5422,1009,7331}.json`（全 (b) DB query）。
- ⚠️ 非決定性 + bug-fix 後重跑（§一.10 #3 / T_DF-4）。

```bash
for s in 5422 1009 7331; do
  ./venv/bin/python scripts/evaluation/multi_cycle_ensemble_validation.py --commit --seed $s --output /tmp/ens_ref_s${s}.json
done
```

> 進度:tree-family 4/6 報告完成（LGBM/XGB/CatBoost/Ensemble）；RF #5 / ET #6 跑中,transformer_dedicated #7 待續。
