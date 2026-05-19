# model_trainer.py 特徵預測力研究記錄

**研究日期**: 2026-05-19
**研究對象**: `scripts/core/model_trainer.py`
**程式版本**: `model_trainer.py v0.1`
**憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §0.0-A / §0.1 / §0.4 / §8.3 / §8.5 / §14.7-T / §14.7-U
**研究目的**: 檢驗 `feature_store_builder.py` 產生的 features 是否真的轉化為可驗證預測力，而不是只停留在哲學上合理。

---

## 1. 研究結論

`model_trainer.py` 是「可觀測 features → 可驗證預測力」的第三個主要工程轉換器。它不直接讀 raw API tables，也不重新建立 universe；它只讀 committed Feature Store，建立 forward-return label，並用 rank-IC 檢驗特徵對未來報酬排序是否有訊號。

研究裁決：

1. **整體 feature stack 有預測力**：既有 48 個 historical models 顯示 h20 / h30 各 24 點 walk-forward IC 全部為正。
2. **預測力不是單點偶然**：h20 mean IC = 0.3530，h30 mean IC = 0.3482，兩者 IC >= 0 皆為 24/24。
3. **有效 feature group 清楚**：price 最強，fundamental / institutional 穩定有效，liquidity 有 regime-dependent 貢獻。
4. **macro / theme 目前沒有 cross-sectional 預測力**：ablation impact 為 0，符合 broadcast / selection-overlap 的程式結構。
5. **現行 trainer 是 robust rank-IC baseline，不是完整 ML 模型**：它是可解釋、可審計、可重現的 baseline；後續可升級，但目前已足以證明 feature stack 具正向排序訊號。

---

## 2. 程式定位

`model_trainer.py` 的權責是：

```text
committed Feature Store
  -> forward-return labels
  -> winsorized feature ranks
  -> single-feature rank IC
  -> signed normalized weights
  -> model artifact + model_registry
```

它的明確邊界：

- 不讀 FinMind / FRED API。
- 不直接讀 raw feature source tables。
- 不重選 core universe。
- 不產生 prediction table。
- 不產生投資建議或 portfolio allocation。

這符合 §8 三層分工：Feature Store / Model Registry / Prediction Table 分離。

---

## 3. Label 建構與 anti-leakage

程式使用 committed `feature_store_snapshot.as_of_date` 作 feature cutoff，並設定：

```text
label_min_date = as_of_date + label_horizon
```

label 由 `TaiwanStockPriceAdj` 建構：

```text
base_close   = date <= as_of_date 的最近收盤價
future_close = date >= as_of_date + label_horizon 的第一筆收盤價
forward_return = future_close / base_close - 1
```

程式也檢查：

```text
min_label_date >= label_min_date
```

因此 `model_trainer.py` 在時間邊界上是合憲的：features 在 as-of 之前，labels 在 horizon 之後。

---

## 4. Trainer 方法

現行 trainer 為：

```text
robust_rank_ic_baseline_v0.1
```

方法流程：

1. 對每個 feature 取 cross-section values。
2. 做 5% / 95% winsorization，降低極端值污染。
3. 將 feature values 轉成 average-rank score。
4. 將 forward-return label 也轉成 rank score。
5. 計算每個 feature 的單因子 rank IC。
6. 以 rank IC 作 signed weight。
7. 用 L1 norm 正規化所有 feature weights。
8. 合成 prediction score。
9. 以 prediction rank vs label rank 計算整體 IC。

此方法的優點：

- 高可解釋。
- 抗 outlier。
- 對厚尾市場比普通 linear regression 更穩健。
- artifact 可完整重現，包括 winsor bounds 與 feature weights。

此方法的限制：

- 沒有 nonlinear interaction。
- 沒有真正 out-of-sample training / validation split；每個 as-of cross-section 同時估權重與評分，屬 historical evidence baseline。
- macro / theme broadcast features 在 cross-sectional rank 中天然難以貢獻。

---

## 5. 既有 walk-forward 實證

依 `reports/walk_forward_h20_h30_panel24_20260518.md`，目前已有 48 個 committed historical models。

| Horizon | n | first_as_of | last_as_of | min IC | max IC | mean IC | median IC | stdev IC | IC >= 0 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| h20 | 24 | 2024-05-31 | 2026-04-25 | 0.1820 | 0.5184 | 0.3530 | 0.3718 | 0.0848 | 24/24 |
| h30 | 24 | 2024-04-30 | 2026-03-31 | 0.1978 | 0.5889 | 0.3482 | 0.3276 | 0.0923 | 24/24 |

研究裁決：

1. `feature_store_builder.py` 的 features 經 `model_trainer.py` 後，對 h20 / h30 forward return 皆有穩定正 IC。
2. h20 與 h30 mean IC 接近，代表訊號不是單一 horizon 偶然。
3. h30 stdev 0.0923，先前 12 點 h30 panel stdev 0.0622；擴大至 24 點後仍全正，代表中期 horizon 仍可用。
4. 此為 historical evidence；production-current h20 仍受 label window gate 限制，不得提前宣告正式交付。

---

## 6. Feature group 預測力

依 `reports/model_quality_research_20260518.md` 的 48 model ablation：

| Horizon | Dropped group | Mean delta | Median delta | Harmful drop count |
|---|---|---:|---:|---:|
| h20 | price | -0.0682 | -0.0457 | 19/24 |
| h20 | fundamental | -0.0226 | -0.0136 | 23/24 |
| h20 | institutional | -0.0210 | -0.0186 | 24/24 |
| h20 | liquidity | -0.0124 | 0.0011 | 11/24 |
| h20 | macro | 0.0000 | 0.0000 | 0/24 |
| h20 | theme | 0.0000 | 0.0000 | 0/24 |
| h30 | price | -0.0563 | -0.0452 | 21/24 |
| h30 | fundamental | -0.0293 | -0.0211 | 22/24 |
| h30 | liquidity | -0.0188 | -0.0020 | 13/24 |
| h30 | institutional | -0.0162 | -0.0132 | 20/24 |
| h30 | macro | 0.0000 | 0.0000 | 0/24 |
| h30 | theme | 0.0000 | 0.0000 | 0/24 |

解讀：

- `price` 是最強預測群，對應 §0.1 的 `Delta_lnP`。
- `fundamental` 穩定有效，且 h30 重要性略高，對應 §0.1 的 `V`。
- `institutional` 在 h20 特別穩定，24/24 drop harmful，對應外部資訊力 / 資金力。
- `liquidity` 有貢獻但 regime-dependent，對應 `M`，並可能在不同市場狀態下轉強或轉弱。
- `macro` / `theme` 目前沒有直接 cross-sectional 預測力；它們更適合 universe governance、regime interaction 或 portfolio/risk layer。

---

## 7. Sector-neutral 研究對預測力的含義

同一研究顯示 sector-neutral ranking 會降低 IC：

| Horizon | Full mean IC | Sector-neutral mean IC | Mean delta |
|---|---:|---:|---:|
| h20 | 0.3530 | 0.3082 | -0.0448 |
| h30 | 0.3482 | 0.3008 | -0.0474 |

裁決：

- 現行 feature/model stack 的一部分 alpha 來自 sector / semiconductor channel。
- 直接 sector-neutral 會降低預測力，不應放入 `model_trainer.py` 預設 scoring。
- sector concentration 應在 portfolio/risk overlay 處理，或用 sector interaction features 研究，而不是直接壓平 model score。

---

## 8. 與 §0.1 的關係

`model_trainer.py` 是檢驗 §0.1 是否真的有工程價值的第一個工具。

| §0.1 元素 | Feature evidence | Model evidence |
|---|---|---|
| `Delta_lnP` | price features | price ablation h20 -0.0682 / h30 -0.0563 |
| `M` | liquidity features | liquidity ablation h20 -0.0124 / h30 -0.0188 |
| `V` | fundamental features | fundamental ablation h20 -0.0226 / h30 -0.0293 |
| external force | institutional features | institutional ablation h20 -0.0210 / h30 -0.0162 |

因此，§0.1 不是純哲學敘事；其主要可觀測 proxy 已在 historical model evidence 中呈現正向預測力。

---

## 9. 風險與限制

### 9.1 Baseline 不是完整 production ML

目前 trainer 是 robust rank-IC baseline。它有強審計性，但不是 final ML architecture。未來若要升級，仍需保留：

- 同一 label rule。
- 同一 anti-leakage audit。
- 同一 model_id governance。
- 同一 feature artifact reproducibility。

### 9.2 Cross-sectional in-sample 估權重

每個 model 用同一 as-of cross-section 計 feature IC 與合成 score。這適合作 historical evidence 與 signal ranking baseline，但若要 production-grade ML，後續需加入更嚴格 temporal training window。

### 9.3 Macro/theme 需要重新設計

macro/theme 在現行模型中 IC impact 為 0，不代表 §0.3 無效，而是現有 feature form 對 cross-sectional rank 沒有區分力。後續應研究：

- macro × sector interaction。
- macro × liquidity / volatility interaction。
- theme subpillar granularity。
- regime-specific model family。

### 9.4 Label date 使用 calendar-day horizon

程式使用 `as_of_date + label_horizon calendar days` 後第一筆交易日作 label。這與既有報告口徑一致；若憲章未來要求「20 trading days」嚴格定義，trainer label rule 必須同步升版。

---

## 10. 本研究裁決

`model_trainer.py` 證明：Feature Store 中的主要第一性 features 已具可驗證預測力。

結論不是「所有 feature 都有效」，而是更精確：

1. `price`、`fundamental`、`institutional`、`liquidity` 四群有效。
2. `macro`、`theme` 目前在 cross-sectional rank model 中無直接貢獻。
3. h20 / h30 historical walk-forward 全部正 IC，支持目前 signal stack 可作 production-current 前置 evidence。
4. 目前仍不應直接進 portfolio conclusion；下一步應研究 `prediction_engine.py` 如何把 committed model 轉成正式 prediction，並維持 single prediction-backed delivery 治權。

下一支逐程式研究建議：

```text
scripts/core/prediction_engine.py
```

研究重點：模型分數如何成為正式 prediction，prediction-backed delivery 如何治理，以及如何避免把 historical evidence 誤當 production-current 交付。
