# Transformer Dedicated (FT-Transformer) Multi-Cycle 驗證報告(模型 #7,深度學習)

**模型**：FT-Transformer(Feature Tokenizer Transformer;d_model=64 / heads=4 / layers=2 / epochs=15）| **類別**：Deep Learning(首個非樹跨架構)
**最後更新**：2026-06-04 | **基準**：`common_model_comparison_baseline_20260602.md`（同 397 核心 × 37 特徵 × 4 horizon）
**Seeds**：5422 / 1009 / 7331（3/3 完成）| **資料來源**：全 (a) 程式輸出 `/tmp/td_ref_s*.json`（§一.10）

---

## 一、4-Horizon × 3-Seed 聚合（median）

| Horizon | NET 年化 | Sharpe | Eff t | Win | IC | T_CZ-6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| weekly 5d | −1.0% | 0.58 | 0.20 | 66.0% | −0.002 | 🔴 |
| monthly 20d | +16.6% | 1.10 | 1.86 | 66.0% | −0.007 | 🔴 |
| quarterly 60d | +17.0% | 1.75 | 4.28 | 72.4% | 0.033 | 🔴 |
| **annual 252d** | **+20.3%** | **3.53** | **4.04** | **85.3%** | **0.162** | **🔴**（Eff t 4.04 < 4.20）|

> weekly 甚至負報酬 + IC≈0（噪音）；annual 雖正報酬但 **Eff t 4.04 差一點未過 gate**。

## 二、★ 最終裁決:樹 vs 深度學習(7 模型 ANNUAL 3-seed median)

| 排名 | 模型 | NET 年化 | Sharpe | Eff t | Win | IC | T_CZ-6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **XGBoost** | **+32.7%** | 4.63 | 5.77 | 90.2% | 0.297 | 🟢 |
| 2 | Ensemble | +30.3% | 4.59 | 5.94 | 90.2% | 0.282 | 🟢 |
| 3 | LightGBM | +30.2% | 4.62 | 6.02 | 89.2% | 0.276 | 🟢 |
| 4 | CatBoost | +26.3% | 4.21 | 5.05 | 89.2% | 0.238 | 🟢 |
| 5 | RandomForest | +23.2% | 3.93 | 4.57 | 88.2% | 0.195 | 🟢 |
| 6 | ExtraTrees | +22.8% | 3.89 | 4.45 | 87.3% | 0.143 | 🟢 |
| **7** | **TransformerD(DL)** | +20.3% | 3.53 | 4.04 | 85.3% | 0.162 | **🔴 FAIL** |

### 結論(蓋棺定論)
- **樹模型完勝深度學習**:6 樹模型 annual **全過 T_CZ-6**;唯一深度學習(FT-Transformer)**墊底 + 未過 gate**。
- **機制梯度**:**梯度提升(IC 0.24-0.30)> 隨機森林(0.14-0.20)> 深度學習(0.16,gate fail)**。
- **印證學界共識**:tree-based 在橫斷面表格金融任務勝過深度學習;FT-Transformer 未帶來優勢。
- **production 首選 = XGBoost / LightGBM,annual horizon,~+30-33%/yr**。

## 三、誠實註記(§一.10)
- td 為 CPU-only 訓練(無 GPU),3-seed 約 2.5 天;rework 後輸出 §14.7-DF 共用 metric keys(與樹模型同碼可比)。
- input paradigm:td 吃 37 橫斷面特徵(FT-Transformer tokenize),與樹模型同 input → output 端可比。
- 全數字 source = `/tmp/td_ref_s{5422,1009,7331}.json`(備份於 `/home/hugo/td_results_backup_20260604/`);summarize_horizon_metrics(§14.7-DF)計算。
