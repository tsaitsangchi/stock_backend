# Downstream ML 2025-05-15 Cutoff Validation Log

Date: 2026-05-17
Context: 2026-05-17 is Sunday; no new trading data is expected.
Constitution: `系統架構大憲章_v5.4.22.md` §8 draft

## 1. Governance Decision

使用者指定：所有模型交易資料以 `2025-05-15` 以前資料執行，且必須符合憲章治權。

Strict `label_horizon=20` 下，若使用先前驗證點 `as_of_date=2025-05-14`，label date 會落在 `2025-06-03`，超過 `2025-05-15` cutoff。因此本次將有效歷史驗證點前移：

- as_of_date: `2025-04-25`
- label_horizon: `20`
- required label date: `2025-05-15`
- model trading data cutoff: `2025-05-15`

此設定同時滿足：

- feature as-of strict: feature source date `<= 2025-04-25`
- label horizon strict: label date `>= 2025-04-25 + 20`
- cutoff strict: label date `<= 2025-05-15`

## 2. Step 9 Feature Store

Command:

```bash
python scripts/core/feature_store_builder.py --commit --as-of-date 2025-04-25 --feature-set-version feature_set_v0.1_h20_20250515_cutoff --label-horizon 20
```

Result:

- feature_set_id: `fs_20250425_feature_set_v0_1_h20_20250515_cutoff`
- status: PERFECT
- stocks: 150
- feature_count: 27
- feature rows: 2965
- null imputed: 83
- warnings: 0
- failures: 0

Note: feature_store_builder summary printed `source_cutoff=2026-05-16` because raw DB contains later rows; feature SQL still uses as-of-strict filters `date <= as_of_date`.

## 3. Step 10 Model Trainer

Dry-run:

```bash
python scripts/core/model_trainer.py --dry-run --feature-set-id fs_20250425_feature_set_v0_1_h20_20250515_cutoff --model-family lgbm --label-horizon 20
```

Commit:

```bash
python scripts/core/model_trainer.py --commit --feature-set-id fs_20250425_feature_set_v0_1_h20_20250515_cutoff --model-family lgbm --label-horizon 20
```

Result:

- model_id: `mdl_20250425_lgbm_h20_v0_1`
- status: WARNING
- rows_trained: 144
- feature_count: 27
- label_source: `TaiwanStockPriceAdj.forward_return_label_v0.1`
- label_horizon: 20
- label_date_min: `2025-05-15`
- label_date_max: `2025-05-15`
- ic_mean: `-0.03824451410658307`
- rmse: `877834.309090824`

Decision: WARNING is acceptable for pipeline governance execution, but the negative IC means this model is not quality-approved for promotion to v5.4.23.

## 4. Step 11 Prediction

Command:

```bash
python scripts/core/prediction_engine.py --commit --model-id mdl_20250425_lgbm_h20_v0_1 --as-of-date 2025-04-25
```

Result:

- run_id: `pred_20250425_mdl_20250425_lgbm_h20_v0_1`
- status: PERFECT
- predictions: 150
- imputed feature ratio: 0.0280
- warnings: 0
- failures: 0

## 5. Governance State Cleanup

Deprecated models:

- `mdl_20250514_lgbm_h20_v0_1`: label_date_max=`2025-06-03`, violates cutoff `<= 2025-05-15`
- `mdl_20260514_lgbm_h1_v0_1`: non-formal horizon h1 and label_date_max=`2026-05-15`

Deprecated prediction runs:

- `pred_20250514_mdl_20250514_lgbm_h20_v0_1`
- `pred_20260514_mdl_20260514_lgbm_h1_v0_1`

Current committed model:

- `mdl_20250425_lgbm_h20_v0_1`

Current committed prediction run:

- `pred_20250425_mdl_20250425_lgbm_h20_v0_1`

## 6. Step 11A Leakage Audit

Code update:

- `scripts/maintenance/audit_leakage.py`
- added `model_data_cutoff` check:
  - committed model must use `label_horizon=20`
  - committed model must have `metrics.label_date_max <= 2025-05-15`

Verification:

```bash
python -m py_compile scripts/maintenance/audit_leakage.py
python scripts/maintenance/audit_leakage.py
```

Audit result:

- status: PERFECT
- PASS/WARN/FAIL: 17/0/0
- model_data_cutoff: PASS
- prediction_coverage: `pred_20250425_mdl_20250425_lgbm_h20_v0_1=150/150`

## 7. Constitution Update

Updated:

- `reports/系統架構大憲章_v5.4.22.md`

Changes:

- Added 2026-05-17 non-trading-day cutoff ruling.
- Updated §8.8.3 recorded historical validation evidence to the cutoff-compliant `2025-04-25 / h20 / label_date=2025-05-15` run.
- Added §8.8.5 cutoff ruling.
- Marked prior `2025-05-14 / h20` validation as pipeline reference only, not cutoff-valid evidence.

## 8. Decision

The system now complies with the requested rule that current model trading data is based on data through `2025-05-15`.

§8 remains ACTIVE (DRAFT), because:

- Step 10 is WARNING due to negative IC.
- Production-current validation for `as_of_date=2026-05-14` still requires future label data after the formal horizon.

