# Extra Trees Multi-Cycle 驗證報告（模型 #6）

**模型**：Extra Trees（極端隨機樹,200 trees / depth 5 / max_features=sqrt）| **類別**：Tree（extremely-randomized bagging）
**最後更新**：2026-06-02
**比較基準**：依 `common_model_comparison_baseline_20260602.md`（同基準）| **Metric**：`summarize_horizon_metrics()`（§14.7-DF 共用 helper）
**資料來源**：全 (b) DB query（§一.10）| **Seeds**：5422 / 1009 / 7331

---

## 一、4-Horizon × 3-Seed 聚合（median）

| Horizon | net 年化 | Sharpe | Eff t ⭐ | Win | IC | IR | 命中 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| weekly 5d | +12.1% | 0.846 | 1.571 | 66.0% | **−0.004** | 0.529 | 51.9% |
| monthly 20d | +27.9% | 1.201 | 3.200 | 66.0% | 0.007 | 1.077 | 54.9% |
| quarterly 60d | +17.3% | 1.909 | 4.567 | 73.3% | 0.042 | 2.183 | 52.5% |
| **annual 252d** | **+22.8% [22.7–23.1]** | **3.887 [3.78–4.02]** | **4.445** | **87.3%** | **0.143** | **4.419** | 59.3% |

> weekly/monthly IC ≈ 0（甚至負）→ 短週期無訊號;annual IC 0.143 為**六模型最低**。

## 二、T_CZ-6 + 裁決

| Horizon | Eff t | Sharpe | Win | Gate |
| :--- | :--- | :--- | :--- | :--- |
| weekly / monthly / quarterly | — | <2.40 | <79% | 🔴 |
| **annual** | **4.445 ✅** | **3.887 ✅** | **87.3% ✅** | 🟢 **PASS**（net +22.8%/yr,**剛過**）|

> **能賺錢 — annual production-grade**,但為樹族最弱(eff_t 4.445 僅略高於 4.20 門檻)。

---

## 三、六模型 Head-to-Head（annual median,同基準）

| metric | LightGBM | XGBoost | CatBoost | Ensemble | RandomForest | **ExtraTrees** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **net 年化** | +30.2% | **+32.7%** | +26.3% | +30.3% | +23.2% | +22.8% |
| **Sharpe** | 4.62 | **4.63** | 4.22 | 4.59 | 3.93 | 3.89 |
| **Eff t** | **6.02** | 5.77 | 5.05 | 5.94 | 4.57 | 4.45 |
| **IC** | 0.276 | **0.297** | 0.238 | 0.282 | 0.195 | 0.143 |
| **T_CZ-6** | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |

### 關鍵結論（六樹模型完成）
> **明確排序**(annual):**XGBoost ≈ LightGBM > Ensemble > CatBoost > RandomForest > ExtraTrees**。
> - **Boosting（XGB/LGBM/CatBoost,IC 0.24-0.30）> Bagging（RF/ET,IC 0.14-0.20)** —— 梯度提升逐步擬合殘差,抽排序訊號明顯優於隨機森林平均;**Extra Trees 的極端隨機化進一步弱化訊號**(IC 最低)。
> - **Ensemble 平均 ≈ 最佳單一,未超越**(被弱者拉低)。
> - ⭐ **六樹模型 annual 全過 T_CZ-6** → 「**年線可賺錢**」結論**高度穩健**(跨 6 模型 × 3 seed 一致,非僥倖);最佳實務 = **選 XGBoost 或 LightGBM,annual horizon,net +30~33%/yr**。

---

## 四、來源 + 重現
```bash
for s in 5422 1009 7331; do
  ./venv/bin/python scripts/evaluation/multi_cycle_extra_trees_validation.py --commit --seed $s --output /tmp/et_ref_s${s}.json
done
```
> 進度:**6 樹模型報告全完成**(LGBM/XGB/CatBoost/Ensemble/RF/ET);transformer_dedicated #7（nn.Transformer on 37 特徵)跑中 → 將是首個 non-tree 機制對照。
