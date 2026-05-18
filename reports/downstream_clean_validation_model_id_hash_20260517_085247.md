# Downstream Clean Validation With Model ID Hash

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` §8 draft
Purpose: 修正 `model_id` 生成規則後，以全新 feature set 執行 clean validation。

## 1. Issue

舊 `model_id` 只由 date、model family、label horizon 組成：

```text
mdl_{yyyymmdd}_{family}_h{label_horizon}_v0_1
```

風險：

- 同一 `as_of_date`、同一 family、同一 horizon 下，不同 `feature_set_version` 會產生相同 `model_id`。
- `model_registry` 會被 `ON CONFLICT(model_id)` 覆寫。
- `data/models/{model_id}` artifact path 也會被覆寫。
- 這違反 §8 Model Registry 的可重現 artifact 治權。

## 2. Fix

Updated:

- `scripts/core/model_trainer.py`

New rule:

```text
mdl_{yyyymmdd}_{family}_h{label_horizon}_{sha1(feature_set_version)[:8]}_v0_1
```

Implementation notes:

- `feature_set_version` is read from committed `feature_store_snapshot`.
- model identity is assigned after `load_inputs()` loads the snapshot.
- artifact path follows the new `model_id`.

## 3. Clean Validation Inputs

Step 9 command:

```bash
python scripts/core/feature_store_builder.py --commit --as-of-date 2025-04-25 --feature-set-version feature_set_v0.1_h20_20250515_cutoff_rankic_validation --label-horizon 20
```

Result:

- feature_set_id: `fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation`
- feature_set_version hash: `5c7f36c2`
- status: PERFECT
- stocks: 150
- features: 27
- feature rows: 2965
- null imputed: 83

## 4. Step 10

Dry-run:

```bash
python scripts/core/model_trainer.py --dry-run --feature-set-id fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation --model-family lgbm --label-horizon 20
```

Commit:

```bash
python scripts/core/model_trainer.py --commit --feature-set-id fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation --model-family lgbm --label-horizon 20
```

Result:

- model_id: `mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- status: PERFECT
- trainer: `robust_rank_ic_baseline_v0.1`
- rows_trained: 144
- feature_count: 27
- label_date_min: `2025-05-15`
- label_date_max: `2025-05-15`
- ic_mean: `0.4911100393859014`
- rmse: `0.31412841802110064`

## 5. Step 11

Command:

```bash
python scripts/core/prediction_engine.py --commit --model-id mdl_20250425_lgbm_h20_5c7f36c2_v0_1 --as-of-date 2025-04-25
```

Result:

- run_id: `pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- status: PERFECT
- predictions: 150
- warnings: 0
- failures: 0

## 6. State Cleanup

Deprecated:

- `pred_20250425_mdl_20250425_lgbm_h20_v0_1_084558`

Reason:

- superseded by clean validation model-id-hash run.

## 7. Step 11A

Command:

```bash
python scripts/maintenance/audit_leakage.py
```

Result:

- status: PERFECT
- PASS/WARN/FAIL: 18/0/0
- model_id_governance: PASS
- committed prediction coverage: `pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1=150/150`

## 8. Decision

The model ID collision risk is resolved.

Clean validation evidence now uses a fresh feature set, a unique model ID tied to `feature_set_version`, and a fresh prediction run:

- Step 9: PERFECT
- Step 10: PERFECT
- Step 11: PERFECT
- Step 11A: PERFECT

§8 remains ACTIVE (DRAFT) until production-current validation is available.
