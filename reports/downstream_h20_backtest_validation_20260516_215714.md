# Downstream ML Step 9-11A Historical 20-Day Validation Log

Date: 2026-05-16
Constitution: `系統架構大憲章_v5.4.22.md` §8 draft
Purpose: 使用前一年歷史資料驗證正式 `label_horizon=20` forward-return contract。

## 1. Validation Rationale

原先 `as_of_date=2026-05-14` 無法正式驗收 20-day forward return，因為需要至少 `2026-06-03` 之後的價格資料。

本次改用歷史驗證點：

- `as_of_date=2025-05-14`
- `label_horizon=20`
- required label date: `2025-06-03`
- DB price coverage: `2024-05-16` through `2026-05-15`

因此可在不使用未來不可得資料的前提下，驗證 Step 9 -> Step 10 -> Step 11 -> Step 11A 的正式 20-day pipeline。

## 2. Step 9 Feature Store Build

Command:

```bash
python scripts/core/feature_store_builder.py --commit --as-of-date 2025-05-14 --feature-set-version feature_set_v0.1_h20_2025_validation --label-horizon 20
```

Result:

- feature_set_id: `fs_20250514_feature_set_v0_1_h20_2025_validation`
- status: PERFECT
- universe: 150 stocks
- price series loaded: 146 stocks
- feature rows: 2965
- feature count: 27
- null imputed: 83
- warnings: 0
- failures: 0

## 3. Step 10 Strict Forward-Return Training

Dry-run command:

```bash
python scripts/core/model_trainer.py --dry-run --feature-set-id fs_20250514_feature_set_v0_1_h20_2025_validation --model-family lgbm --label-horizon 20
```

Commit command:

```bash
python scripts/core/model_trainer.py --commit --feature-set-id fs_20250514_feature_set_v0_1_h20_2025_validation --model-family lgbm --label-horizon 20
```

Result:

- model_id: `mdl_20250514_lgbm_h20_v0_1`
- status: PERFECT
- rows_trained: 145
- feature_count: 27
- label_source: `TaiwanStockPriceAdj.forward_return_label_v0.1`
- label_horizon: 20
- label_date_min: `2025-06-03`
- label_date_max: `2025-06-03`
- ic_mean: `0.19672886159659897`
- rmse: `901496.7979223129`
- warnings: 0
- failures: 0

Validation note: `label_date_min >= as_of_date + label_horizon` 成立，符合 strict forward-return anti-leakage contract。

## 4. Step 11 Prediction

Command:

```bash
python scripts/core/prediction_engine.py --commit --model-id mdl_20250514_lgbm_h20_v0_1 --as-of-date 2025-05-14
```

Result:

- run_id: `pred_20250514_mdl_20250514_lgbm_h20_v0_1`
- status: PERFECT
- predictions: 150
- imputed feature ratio: 0.0280
- warnings: 0
- failures: 0

## 5. Step 11A Leakage Audit

Command:

```bash
python scripts/maintenance/audit_leakage.py
```

Result:

- status: PERFECT
- PASS/WARN/FAIL: 16/0/0
- label horizon audit: PASS
- feature/universe lock audit: PASS
- prediction coverage audit: PASS

Coverage audit was tightened during this validation to verify each committed prediction run independently:

- `pred_20250514_mdl_20250514_lgbm_h20_v0_1=150/150`
- `pred_20260514_mdl_20260514_lgbm_h1_v0_1=150/150`

## 6. Code Change

Updated:

- `scripts/maintenance/audit_leakage.py`

Change:

- prediction coverage check now validates every committed `prediction_run` against its locked core+convex universe count.
- previous aggregate coverage count could pass while hiding a deficient individual run.

Verification:

```bash
python -m py_compile scripts/maintenance/audit_leakage.py
python scripts/maintenance/audit_leakage.py
```

Both passed.

## 7. Decision

The formal `label_horizon=20` downstream pipeline is validated on historical data:

- Step 9: PASS/PERFECT
- Step 10: PASS/PERFECT
- Step 11: PASS/PERFECT
- Step 11A: PASS/PERFECT

This resolves the earlier limitation that only `horizon=1` could be run on `as_of_date=2026-05-14`.

Promotion consideration:

- From a pipeline and anti-leakage perspective, §8 now has a successful 20-day historical validation.
- For production-current validation at `as_of_date=2026-05-14`, the system still needs prices on or after `2026-06-03`.
- Recommended next decision: update `系統架構大憲章_v5.4.22.md` §8 acceptance note to allow historical 20-day validation as draft acceptance evidence, while reserving production-current validation for the first run after `2026-06-03`.
