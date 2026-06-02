# 共同模型比較基準定義 (Common Model Comparison Baseline) — SSOT

**文件性質**：本檔為「多模型標準化評估框架」之**比較基準單一事實來源 (SSOT)**。全部待評估模型(tree / transformer / foundation / 股票專用)之 multi-cycle 驗證**必須**遵循本檔定義之**同一**universe、窗、horizon、metric、gate、seed protocol，否則跨模型「精準度 / 信任度」比較不可比、不可採信。
**最後更新**：2026-06-02
**治權依據**：§14.7-DE(Canonical Panel Source 單一引用源)+ §0.0-I(單一引用源)+ §一.13(反硬編 source-pure)+ §14.7-CY/CZ/T_CZ-6(multi-cycle validation gate)+ §14.7-DC(source-pure universe)+ §一.10(資料真實性)
**動因**：用戶 directive「建標準化多模型評估框架，回答『股票預測能否真的賺錢?』，僅用真實 DB 資料、multi-cycle walk-forward、每股最長歷史、每模型獨立程式、**相同比較基準定義**以利可靠精準度/信任度比較」。

---

## 一、評估目的

回答單一問題:**「以本系統 source-pure 特徵 + 各模型，股票預測能否真的賺錢?」**

- 不是看單一漂亮數字，而是看**跨 horizon × 跨 seed** 的穩定獲利能力與顯著性。
- 每個模型產出**同結構**報告 → 並排比較 → 哪個模型在哪個 horizon 真正可信可獲利。

---

## 二、固定比較基準(全模型共用，不得各自更改)

| 維度 | 固定值 | 來源 / 治權 |
| :--- | :--- | :--- |
| **Universe** | 397 source-pure 核心股 | §14.7-DC PAN-HISTORICAL gate(任一 panel 含 imputed → 排除);`status='committed' ∧ core_tier='core_universe'` |
| **特徵集** | `feature_set_v0.5`，**37 個** source-pure 特徵 | §14.7-DC;K-wave 7 死重已移除(§一.15) |
| **驗證窗** | **資料驅動**:157 panels / **2013-05-15 ~ 2026-06-01** | `core.db_utils.get_canonical_panel_dates("feature_set_v0.5")`(§14.7-DE 單一引用源，**禁寫死**) |
| **Panel 定義** | 每月中(15 號)snapshot | feature_store_builder |
| **Horizons** | weekly **5d** / monthly **20d** / quarterly **60d** / annual **252d** | §14.7-CY 4-horizon |
| **驗證法** | walk-forward expanding window，每股最長歷史 | §14.7-CY;train < as_of < label(§8.5 anti-leakage) |
| **組合構建** | 預測分數 top-20 等權多頭 vs universe 等權基準 | 標準化(全模型同) |
| **交易成本** | log cost drag per rebalance(`cost_per_rebal=0.006`) | Tier-3 透明揭露(§一.13 v0.6;TW broker fee estimate) |
| **Seeds** | **≥ 3**(median + range，異常值註明) | §一.10 #3 stochastic ≥3 runs |
| **資料來源** | 全 (b) DB query(無估算、無 AI 補值) | §一.10;`source_traceability` 入 JSON `_meta` |

---

## 三、標準化 metric 定義(每模型報告之同一欄位)

### A. 獲利能力 (Can it make money?)
- **`annualized_simple_net`** — 扣成本後年化簡單報酬(top-20 組合)。
- **`mean_ret_per_panel`** / **`mean_alpha_per_panel`** — 每 panel 平均報酬 / 超額(vs 基準)。

### B. 風險調整後績效
- **`sharpe`** — 月-panel 報酬年化 Sharpe(`mean/std × √12`)。
- **`ir`** — Information Ratio(alpha 年化)。
- **`mdd_per_panel`** — 每 panel 最大回撤。

### C. 統計顯著性(核心可信度)
- **`effective_t_stat`** ⭐ — **overlap-corrected** t 統計量(`t × √(n_eff/n)`，§14.7-CY T_CY-3)。**這是 T_CZ-6 gate 採用的顯著性指標**(非 raw `t_stat`，因長 horizon panels 重疊會虛增 n)。
- **`is_significant_p05`** — `|eff_t| > 1.997`。

### D. 精準度 (Precision)
- **`precision_directional_hit_rate`** — 方向命中率(預測漲跌對的比例;0.5=無方向 edge)。
- **`precision_top20_actual_overlap`** — 預測 top-20 與實際 top-20 的重疊率。
- **`precision_rmse`** / **`precision_mae`** — 預測 vs 實際報酬之 RMSE / MAE。

### E. 信任度 / 可靠度 (Trust / Reliability)
- **跨 seed spread**(min/max range) — 越窄越可信(隨機性低)。
- **`reliability_ic_stability_cov`** — IC 跨 panel 的變異係數(CoV，越低越穩定)。
- **`mean_ic`** — Spearman 排序 IC(預測力強度;§14.7-CM)。

---

## 四、T_CZ-6 Production Gate(§14.7-CZ)

某 horizon 視為「production-grade 可獲利且可信」須**同時**滿足(取跨 seed **median**):

> **Effective t ≥ 4.20  ∧  Sharpe ≥ 2.40  ∧  Win rate ≥ 79%**

- 三者皆 ✅ → 🟢 **PASS**(該 horizon production-grade)。
- 任一 ❌ → 🔴 **FAIL**(該 horizon 不足 production，但可能仍獲利，須在「賺錢裁決」分級說明)。

## 五、「能否賺錢」裁決分級(每模型報告須給)

| 等級 | 條件 |
| :--- | :--- |
| 🟢 **Production-grade 獲利** | 過 T_CZ-6 gate(該 horizon) |
| 🟡 **獲利但未達 production 嚴格門檻** | `annualized_simple_net > 0` ∧ `eff_t` 顯著(>1.997)，但未過 T_CZ-6 |
| 🟠 **微弱 / 邊際** | net 正但小、或方向 hit rate ≈ 0.5(無 edge) |
| 🔴 **不獲利 / 不可信** | net ≤ 0 或不顯著 |

---

## 六、程式架構(每模型獨立程式 + 共用單一來源)

- 每模型一支 `scripts/evaluation/multi_cycle_<model>_validation.py`(獨立可重跑)。
- **全部** import `core.db_utils.get_canonical_panel_dates()` 取窗(§14.7-DE 單一引用源，**禁各自寫死 panel 日期/特徵數**;違反 = T_DE-1~4)。
- 全部在**系統環境**跑(`venv/bin/python`)，**非** AI 環境;結果寫 JSON(`_meta.source_traceability` 標記)。
- **Metric 計算單一引用源**(§14.7-DF / §一.17):horizon-summary metric(sharpe/eff_t/IC/win/precision/reliability)唯一計算來源 = `core.db_utils.summarize_horizon_metrics()`;validator 只產每 panel `(pred, actual)`,全部 metric 由 helper 算 → 全模型 metric 碼 100% 一致、可比(**禁各自 inline 計算**;違反 = T_DF-1~4)。
  - ✅ **9 helper-using**(tree-like):lightgbm / xgboost / catboost / random_forest / extra_trees / ensemble / xgboost_dedicated / catboost_dedicated / transformer_dedicated。
  - ⏳ **4 torch/foundation**(chronos / itransformer / patchtst / tft):自有 calibration 導向 `aggregate_horizon`;各模型 rework 時額外輸出共同 keys(非強制改用 tree helper,以保校準資訊)。

---

## 七、待評估模型清單(11+ 模型)

| 類別 | 模型 | 狀態 |
| :--- | :--- | :--- |
| Tree | **LightGBM** | ✅ 參考基準模型(本框架首個;見 `lightgbm_validation_report_20260602.md`) |
| Tree | XGBoost | ⏳ 次個 |
| Tree | CatBoost | ⏳ |
| Tree | Multi-tree Ensemble | ⏳ |
| Transformer | TFT / iTransformer / PatchTST | ⏳(部分舊 validator) |
| Foundation | TimesFM / Chronos | ⏳(Chronos 部分舊 validator;TimesFM 待寫) |
| 股票專用 | Stockformer / HIST | ⏳(待寫) |

每模型完成後填入**同結構**報告，最後彙整 master comparison。

---

## 八、治權 cross-reference

- 比較基準窗單一來源:`core/db_utils.py v2.49` `get_canonical_panel_dates()`
- 入憲:憲章 §14.7-DE(T_DE-1~4)+ CLAUDE.md §一.16(雙層治權鎖)
- Gate:§14.7-CZ T_CZ-6 / §14.7-CY T_CY-3(eff_t)
- Universe:§14.7-DC source-pure / §一.13 反硬編
- 資料真實性:§一.10(全數字 trace 至 (b) DB query)
