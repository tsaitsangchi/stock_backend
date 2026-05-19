# prediction_engine.py 正式 Prediction 產生研究記錄

**研究日期**: 2026-05-19
**研究對象**: `scripts/core/prediction_engine.py`
**程式版本**: `prediction_engine.py v0.1`
**憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §0.0-A / §8.4 / §8.8.8 / §8.8.10 / §9.1
**研究目的**: 說明 committed model 如何被轉成正式 prediction-backed delivery，以及 prediction layer 的治理邊界。

---

## 1. 研究結論

`prediction_engine.py` 是「committed model → formal prediction」的第四個主要工程轉換器。它不訓練模型、不改 Feature Store、不改 Model Registry；它只讀 committed `model_registry` 與 committed `feature_store_snapshot`，使用 model artifact 中保存的 weights / preprocessing bounds 產生 prediction ranks，並寫入：

```text
prediction_run
prediction_values
```

研究裁決：

1. **正式 prediction 的來源是 model artifact，不是重新訓練。**
2. **推論 transform 與訓練 transform 對齊**：使用 model artifact 的 winsor bounds 與 average-rank transform。
3. **prediction 是排序型輸出**：輸出 `prediction_value`、`prediction_rank`、`signal_label`、`confidence`，但不是買賣建議。
4. **coverage 是硬邊界**：現版要求 prediction universe rows = 150；不足即 FAIL。
5. **single delivery 治權主要由憲章與 audit 維持**：程式 docstring 宣告 exactly-one prediction-backed，但現行 CLI 尚未實作 `--deprecate-previous` / `--commit-as-evidence-only`，歷史 evidence 仍曾靠後處理 SQL 標記 deprecated。

---

## 2. 程式定位

`prediction_engine.py` 的資料流：

```text
committed model_registry
  -> model artifact model.json
  -> committed feature_store_snapshot
  -> feature_values for core+convex universe
  -> train-consistent transform
  -> prediction rank / signal label
  -> prediction_run + prediction_values
```

它的邊界：

- 不讀 raw API tables。
- 不建立 label。
- 不重新估 weights。
- 不修改 upstream Feature Store / Model Registry。
- 不產生 portfolio weights。

這符合 §8.4 Prediction Table 治權：prediction layer 只負責把已審核模型轉成可追蹤 prediction run。

---

## 3. Input Gate

`load_inputs()` 依序檢查：

1. `model_registry.model_id` 必須存在。
2. model status 必須是 `committed`。
3. artifact path 下的 `model.json` 必須存在。
4. `feature_store_snapshot` 必須是 `committed`。
5. requested `as_of_date` 必須等於 feature set 的 `as_of_date`。
6. feature set 的 `universe_snapshot_id` 必須等於 model_registry 的 `universe_snapshot_id`。
7. `feature_values` 必須可 join 到該 snapshot 的 `core_universe` / `convex_universe`。
8. prediction rows 必須等於 150。

此處最重要的是：prediction 不允許拿錯日期、錯 universe、錯 feature set 或未 committed model 進行推論。

---

## 4. Transform Consistency

模型訓練時 `model_trainer.py` 會輸出：

```json
{
  "weights": {...},
  "preprocessing": {
    "feature_bounds": {
      "<feature>": {"low": ..., "high": ...}
    },
    "rank_tie_method": "average"
  }
}
```

`prediction_engine.py` 推論時：

1. 讀取每個 feature 的 value。
2. 依 artifact 中的 `low` / `high` 做 winsor clipping。
3. 對每個 feature 做 cross-sectional average-rank score。
4. 用 artifact 中的 signed weights 加權。

因此推論與訓練共享同一類 transform，避免 train/predict preprocessing drift。

---

## 5. Prediction Output

`predict()` 產生：

| 欄位 | 來源 | 語意 |
|---|---|---|
| `prediction_value` | weighted rank score | 模型排序分數，不是價格目標。 |
| `prediction_rank` | score descending rank | 1 為最高分。 |
| `signal_label` | rank bucket | top 20 = `long`，bottom 20 = `watch`，其他 = `hold`。 |
| `confidence` | rank distance from median | 排名離中位數越遠 confidence 越高。 |

注意：`signal_label='long'` 在此程式中是訊號標籤，不是交易指令。它仍需經 §9.2 portfolio/risk layer 才能成為配置建議。

---

## 6. Formal Delivery Governance

程式 docstring 宣告：

```text
Single Delivery Invariant:
exactly 1 prediction-backed
```

現行實作中，`commit_outputs()` 會寫入新的 `prediction_run.status='committed'`，但沒有在程式碼中自動 deprecate 其他 committed runs。若 run_id 已存在，程式會加 timestamp suffix，這能避免 primary key 衝突，但不會自動維持 exactly-one。

因此現行治理真相是：

| 項目 | 狀態 |
|---|---|
| prediction run / values 寫入 | 已實作 |
| transform consistency | 已實作 |
| 150-row coverage gate | 已實作 |
| exactly-one prediction-backed 自動化 | **未完整實作** |
| `--deprecate-previous` CLI | docstring 提及，但 parse_args 未實作 |
| `--commit-as-evidence-only` CLI | 規劃中，尚未實作 |

研究裁決：`prediction_engine.py` 可以產生正式 prediction，但「唯一正式 delivery」目前仍需 audit / 後處理 SQL / 下游 readiness 規則共同維持。這是後續小型程式修補的明確候選。

---

## 7. 與 h20/h30 evidence 的關係

目前唯一 committed prediction-backed delivery 仍為：

```text
pred_20260425_mdl_20260425_lgbm_h20_d969ffb1_v0_1
```

依 §14.7-T：

- h20 / h30 historical models 可保留為 committed model evidence。
- historical prediction runs 應標記 deprecated。
- production-current delivery 只能有一個 committed prediction-backed run。

因此 prediction layer 的治權重點不是「能否多產生 prediction」，而是「哪些 prediction 具有正式 delivery 地位」。

---

## 8. 已符合憲章的地方

1. **只讀 committed model 與 committed feature set**
   符合 read-only upstream。

2. **不重新訓練**
   權重完全來自 artifact。

3. **Train/predict transform 一致**
   使用 artifact winsor bounds 與 rank transform。

4. **Universe coverage 強制**
   150 rows 不足直接 FAIL。

5. **Prediction output 可追蹤**
   `run_id = pred_<yyyymmdd>_<model_id>`，可連回 model / feature set / universe。

6. **不產生 portfolio allocation**
   prediction layer 未越權進 §9.2。

---

## 9. 值得後續研究或改善的地方

### 9.1 實作 `--deprecate-previous`

目前 docstring 有此模式，但 CLI 與 `commit_outputs()` 未落地。建議：

- 新增 `--deprecate-previous` flag。
- commit 新 run 前或後，將同一 policy 範圍內其他 `prediction_run.status='committed'` 改為 `deprecated`。
- 留下 notes，例如 `superseded_by=<new_run_id>`。

### 9.2 實作 `--commit-as-evidence-only`

walk-forward / h30 historical evidence 常需要 commit prediction 後立刻 deprecated。建議：

- 新增 `--commit-as-evidence-only`。
- 寫入 prediction_values 供 audit。
- prediction_run.status 直接設為 `deprecated`。
- 避免 audit_downstream_readiness 過渡期出現 prediction-backed count > 1。

### 9.3 signal_label 命名風險

`long` / `hold` / `watch` 容易被誤讀為交易建議。雖然 notes 已寫明不是 investment advice，但後續可研究：

- 改名為 `rank_top20` / `rank_mid` / `rank_bottom20`。
- 或保留 label，但在 prediction_policy 中明確定義為 rank bucket。

### 9.4 confidence 定義偏簡單

目前 confidence 只由 rank distance from median 決定，不使用 feature uncertainty、imputation ratio 或 model IC stability。後續可研究：

- 加入 model historical IC。
- 加入 feature imputation ratio。
- 加入 sector concentration / liquidity risk。

### 9.5 prediction coverage hardcode 150

現行 `len(self.rows) != 150` 直接 FAIL。這符合目前 core+convex 150，但若未來年度治理變更 core/convex 數量，應改成從 universe snapshot / membership 計算 expected count。

---

## 10. 本研究裁決

`prediction_engine.py` 已能把 committed model 轉成正式 prediction run，並維持大部分 §8.4 治權邊界。它是模型進入正式交付層的門。

但它仍有一個重要缺口：**exactly-one prediction-backed delivery 尚未完全自動化**。這不影響既有 evidence 的合法性，因 §14.7-T 已透過後處理與 readiness audit 維持單一 delivery；但若要讓 pipeline 更乾淨，應優先補 `--deprecate-previous` 與 `--commit-as-evidence-only`。

下一步若繼續逐程式研究，應進入 portfolio / sizing 層：

```text
scripts/pipeline/portfolio_strategy.py
scripts/pipeline/portfolio_optimizer.py
scripts/pipeline/portfolio_backtest.py
```

或先做小型程式修補：

```text
scripts/core/prediction_engine.py
  --deprecate-previous
  --commit-as-evidence-only
```
