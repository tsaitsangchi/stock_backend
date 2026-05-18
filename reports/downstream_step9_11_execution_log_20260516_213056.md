# Downstream Step 9-11 Execution Log - 2026-05-16 21:30:56 CST

依使用者要求：「先把三個模組補起來，跑通 Step 9→10→11→11A，再決定是否把 §8 從草案升到 v5.4.23。」

## 本輪新增模組

| 模組 | 版本 | 狀態 | 說明 |
| :--- | :--- | :--- | :--- |
| `scripts/core/model_trainer.py` | v0.1 | IMPLEMENTED | 建立 `model_registry` / `model_training_run`；讀 committed `feature_values` 與 `core_universe_scores.core_score` governance proxy label；輸出 JSON artifact。 |
| `scripts/core/prediction_engine.py` | v0.1 | IMPLEMENTED | 建立 `prediction_run` / `prediction_values`；讀 committed `model_registry` artifact 與 committed feature set；寫入 150 檔 core+convex 推論。 |
| `scripts/maintenance/audit_leakage.py` | v0.1 | IMPLEMENTED | 執行 §8.5 防洩漏稽核：source scan、as-of-strict、label horizon、feature/universe lock、prediction coverage。 |

## 執行結果

| Step | 指令 | 結果 | 摘要 |
| :--- | :--- | :--- | :--- |
| 9 dry-run | `python scripts/core/feature_store_builder.py --dry-run --as-of-date 2026-05-14` | WARNING | 已存在 committed `feature_set_id=fs_20260514_feature_set_v0_1`，重跑將覆寫；27 features、150 stocks、1950 feature rows、null_imputed=1050。 |
| 9 commit | `python scripts/core/feature_store_builder.py --commit --as-of-date 2026-05-14` | WARNING | 同上，成功重寫 committed feature set。 |
| 10 dry-run | `python scripts/core/model_trainer.py --dry-run --feature-set-id fs_20260514_feature_set_v0_1 --model-family lgbm` | PERFECT | rows_trained=150、feature_count=27、IC=0.9938415、RMSE=22.3609347。 |
| 10 commit | `python scripts/core/model_trainer.py --commit --feature-set-id fs_20260514_feature_set_v0_1 --model-family lgbm` | PERFECT | committed `model_id=mdl_20260514_lgbm_v0_1`；artifact path=`data/models/mdl_20260514_lgbm_v0_1/`。 |
| 11 dry-run | `python scripts/core/prediction_engine.py --dry-run --model-id mdl_20260514_lgbm_v0_1 --as-of-date 2026-05-14` | WARNING | 150 predictions computed；WARNING 來自 imputed feature ratio=0.5385 > 5%。 |
| 11 commit | `python scripts/core/prediction_engine.py --commit --model-id mdl_20260514_lgbm_v0_1 --as-of-date 2026-05-14` | WARNING | committed `run_id=pred_20260514_mdl_20260514_lgbm_v0_1`；prediction rows=150。 |
| 11A | `python scripts/maintenance/audit_leakage.py` | PERFECT | PASS/WARN/FAIL=16/0/0；as-of-strict、label horizon、feature/universe lock、prediction coverage 均通過。 |

## 實際 DB 寫入

```text
feature_store_snapshot    1
feature_definition        27
feature_values            1950
model_registry            1
model_training_run        1
prediction_run            1
prediction_values         150
```

Committed model:

```text
model_id: mdl_20260514_lgbm_v0_1
status: committed
metrics:
  ic_mean: 0.9938415040668475
  ic_std: 0.0
  rmse: 22.36093472578758
  rows_trained: 150
  feature_count: 27
  label_source: core_universe_scores.core_score governance_proxy_label_v0.1
```

Committed prediction:

```text
run_id: pred_20260514_mdl_20260514_lgbm_v0_1
status: committed
rows_written: 150
```

## 問題與裁決

1. **Sandbox DB 連線限制**：`model_trainer.py`、`prediction_engine.py`、`audit_leakage.py` 首次在 sandbox 內執行時無法連 `127.0.0.1:5432`；非 sandbox 重跑通過。裁決為執行環境限制。
2. **Step 9 WARNING**：原因是 feature set 已存在並被重寫。這是重跑情境的 idempotency warning，不是資料契約失敗。
3. **Step 11 WARNING**：`imputed feature ratio=0.5385`，來自本輪上游 price/revenue/financial/institutional raw tables 仍為 0 rows，Feature Store 以 v0.1 null strategy 補值。推論覆蓋 150 檔完成，但品質尚未達可升正式強制契約的標準。
4. **Trainer label 邊界**：v0.1 baseline 使用 `core_universe_scores.core_score` 作為 governance proxy label，未使用未來 return。這可用於端到端管線驗收，但不等於正式可交易模型。

## 是否升 §8 至 v5.4.23

裁決：**暫不升 v5.4.23**。

理由：

1. §8 接受標準雖已達成「三模組可執行、Step 9→10→11→11A 可跑通」的最低端到端條件，但 Step 11 仍為 WARNING，且缺值補值比例高達 53.85%。
2. `model_trainer.py v0.1` 使用 governance proxy label，而非正式 forward return label；因此目前只能證明下游治理容器與流程可運作，尚不能宣稱正式模型訓練契約成熟。
3. 上游歷史資料 coverage 尚不足，Step 9 的 price/revenue/financial/institutional 特徵大量補值；應先補刷 raw historical data，再重跑 Step 9→10→11→11A，直到 Step 11 至少降到低補值 WARNING 或 PERFECT。

建議下一步：

1. 補刷 core+convex 150 檔歷史資料：price、monthly revenue、PER、institutional、margin、financial statements。
2. 重跑 Step 9→10→11→11A，目標是 `prediction_engine.py` imputed feature ratio ≤ 5%。
3. 將 trainer label 從 governance proxy label 改為嚴格 `label_horizon` forward-return label，並由 `audit_leakage.py` 驗證 label date boundary。
4. 上述完成後，再把 §8 從草案升為 v5.4.23 強制契約。
