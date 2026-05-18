# Downstream Forward-Return Revalidation - 2026-05-16 21:47:35 CST

依使用者要求：「應先補刷 core+convex 150 檔歷史資料，再把 trainer 改成嚴格 label-horizon forward return。」

## 本輪程式修改

| 檔案 | 修改 |
| :--- | :--- |
| `scripts/core/model_trainer.py` | 將 label 來源由 `core_universe_scores.core_score` governance proxy 改為 `TaiwanStockPriceAdj` forward-return label；強制 `label_date >= as_of_date + label_horizon`；model_id 加入 horizon，例如 `mdl_20260514_lgbm_h1_v0_1`。 |
| `scripts/maintenance/audit_leakage.py` | label horizon 稽核改為檢查 committed model 的 `metrics.label_date_min >= feature_set.as_of_date + label_horizon`；prediction coverage 僅計 committed prediction runs。 |

## 歷史資料補刷

指令：

```bash
python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730
```

結果：

```text
成功同步項目 : 1308
警告同步項目 : 46
失敗同步項目 : 0
跳過同步項目 : 0
402-recovered : 0
總計寫入筆數 : 690194
總計耗時     : 607.83 s
主權判定     : WARNING
```

警告來源主要為部分股票 / dataset API 回傳 0 筆，未出現 API 額度失敗或 402/403/429 阻斷。

補刷後核心 raw rows：

```text
TaiwanStockPriceAdj                         70027
TaiwanStockMonthRevenue                      3369
TaiwanStockFinancialStatements              17450
TaiwanStockInstitutionalInvestorsBuySell   326780
TaiwanStockMarginPurchaseShortSale          63277
```

## Step 9 重跑

指令：

```bash
python scripts/core/feature_store_builder.py --commit --as-of-date 2026-05-14
```

結果：

```text
price series loaded for 147 stocks
feature rows: 3980
null imputed: 47
verdict: WARNING
```

補刷前對照：price series 0 stocks、feature rows 1950、null imputed 1050。補刷後 Feature Store 品質明顯改善。

另建立 horizon=1 驗證用 feature set：

```bash
python scripts/core/feature_store_builder.py --commit --as-of-date 2026-05-14 --feature-set-version feature_set_v0.1_h1 --label-horizon 1
```

結果：

```text
feature_set_id: fs_20260514_feature_set_v0_1_h1
price series loaded for 147 stocks
feature rows: 3980
null imputed: 47
verdict: PERFECT
```

## Step 10 Forward-Return Trainer

### horizon=20 正式預設測試

指令：

```bash
python scripts/core/model_trainer.py --dry-run --feature-set-id fs_20260514_feature_set_v0_1 --model-family lgbm
```

結果：

```text
label horizon enforced: label_date >= 2026-06-03 > as_of=2026-05-14
rows_trained=0
verdict=FAILED
```

裁決：這是正確的防洩漏阻擋。`as_of_date=2026-05-14` 加 `label_horizon=20` 需要 2026-06-03 之後的 label price；目前資料時間尚未到達，不能訓練 20 日 forward-return 模型。

### horizon=1 端到端驗證

指令：

```bash
python scripts/core/model_trainer.py --dry-run --feature-set-id fs_20260514_feature_set_v0_1_h1 --model-family lgbm --label-horizon 1
python scripts/core/model_trainer.py --commit --feature-set-id fs_20260514_feature_set_v0_1_h1 --model-family lgbm --label-horizon 1
```

結果：

```text
model_id: mdl_20260514_lgbm_h1_v0_1
rows_trained: 147
feature_count: 27
label_source: TaiwanStockPriceAdj.forward_return_label_v0.1
label_horizon: 1
label_date_min: 2026-05-15
label_date_max: 2026-05-15
ic_mean: -0.04453377056116782
rmse: 1231584.7850061648
verdict: WARNING
```

裁決：forward-return label 已落地且通過日期邊界；但 baseline IC <= 0，模型品質為 WARNING，不可升為正式模型契約。

## Step 11 Prediction

指令：

```bash
python scripts/core/prediction_engine.py --commit --model-id mdl_20260514_lgbm_h1_v0_1 --as-of-date 2026-05-14
```

結果：

```text
run_id: pred_20260514_mdl_20260514_lgbm_h1_v0_1
predictions: 150
imputed feature ratio: 0.0118
verdict: PERFECT
```

舊 proxy-label model 與 prediction 已標為 deprecated：

```text
mdl_20260514_lgbm_v0_1: deprecated
pred_20260514_mdl_20260514_lgbm_v0_1: deprecated
```

## Step 11A Leakage Audit

指令：

```bash
python scripts/maintenance/audit_leakage.py
```

結果：

```text
PASS/WARN/FAIL: 16/0/0
prediction coverage: 150 committed rows
verdict: PERFECT
```

## 最終 DB 狀態

```text
feature_values      7960
model_registry      2
prediction_run      2
prediction_values   300
```

其中 committed 現行物件：

```text
model_id: mdl_20260514_lgbm_h1_v0_1
prediction_run: pred_20260514_mdl_20260514_lgbm_h1_v0_1
prediction rows: 150
```

## 是否可升 v5.4.23

裁決：**仍不建議升 v5.4.23**。

理由：

1. 20 日 forward-return label 被正確阻擋，表示防洩漏機制有效，但也表示正式預設 horizon 尚未具備 label 資料。
2. horizon=1 已跑通 Step 9→10→11→11A，但 trainer verdict 是 WARNING，IC=-0.0445，不符合「metrics 合理且 IC_mean > 0」的 PERFECT 接受標準。
3. Feature Store 品質大幅改善，prediction imputed ratio 已降至 1.18%，但仍需改善 trainer baseline 與等待/取得足夠 forward label。

建議下一步：

1. 等待或補入 2026-06-03 之後價格資料，再以 `label_horizon=20` 重跑 trainer。
2. 改善 baseline 訓練邏輯，例如加入 train/validation split、標準化、winsorization、rank target 或更穩定的線性/ridge baseline。
3. 重新驗證 `ic_mean > 0` 且 Step 10/11/11A 全部無 FAIL；再評估 §8 升 v5.4.23。
