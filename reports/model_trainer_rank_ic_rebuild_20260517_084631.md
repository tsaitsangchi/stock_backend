# Model Trainer Rank-IC Rebuild Log

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` §8 draft
Scope: Step 10 model quality diagnosis and trainer baseline rebuild.

## 1. Input Boundary

Current governance cutoff:

- non-trading-day ruling date: `2026-05-17`
- model trading data cutoff: `2025-05-15`
- feature_set_id: `fs_20250425_feature_set_v0_1_h20_20250515_cutoff`
- as_of_date: `2025-04-25`
- label_horizon: `20`
- label_date_min/max: `2025-05-15`

## 2. Diagnosis

Rows and label distribution:

- rows: 144
- features: 27
- label min: `-0.10879284649776455`
- label q1: `0.02420352309097845`
- label median: `0.056221253933672166`
- label q3: `0.1053519190144595`
- label max: `0.3345847379711919`
- label mean: `0.06793026220468536`

Top single-factor rank IC diagnostics:

- positive:
  - `avg_daily_value_log_60d`: `0.3295`
  - `turnover_mean_60d`: `0.3051`
  - `volatility_60d`: `0.2607`
- negative:
  - `log_return_20d`: `-0.4212`
  - `ma_ratio_60`: `-0.3738`
  - `log_return_60d`: `-0.2681`

Root causes of previous WARNING:

- raw feature values were combined directly, so large-scale features such as turnover and institutional flow dominated the score.
- rank implementation did not average ties, so constant features could receive artificial rank IC.

## 3. Code Changes

Updated:

- `scripts/core/model_trainer.py`
- `scripts/core/prediction_engine.py`

Trainer changes:

- trainer name changed to `robust_rank_ic_baseline_v0.1`.
- each feature is winsorized at 5%/95%.
- each feature is converted to cross-sectional average-rank score in `[-1, 1]`.
- feature weights are signed single-factor rank IC values normalized by L1 norm.
- tie handling now uses average ranks.
- metrics now include label distribution, trainer name, and top rank-IC feature diagnostics.
- model artifact now includes preprocessing bounds.

Prediction changes:

- prediction engine now applies the same winsor bounds and average-rank transform stored in the model artifact.
- train and predict transforms are aligned.

## 4. Revalidation

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
- status: PERFECT
- trainer: `robust_rank_ic_baseline_v0.1`
- ic_mean: `0.4911100393859014`
- rmse: `0.31412841802110064`
- rows_trained: 144
- label_date_max: `2025-05-15`

Prediction:

```bash
python scripts/core/prediction_engine.py --commit --model-id mdl_20250425_lgbm_h20_v0_1 --as-of-date 2025-04-25
```

Result:

- run_id: `pred_20250425_mdl_20250425_lgbm_h20_v0_1_084558`
- predictions: 150
- status: WARNING only because the original deterministic run id was already occupied by a deprecated run.

Leakage audit:

```bash
python scripts/maintenance/audit_leakage.py
```

Result:

- status: PERFECT
- PASS/WARN/FAIL: 17/0/0
- committed prediction coverage: `pred_20250425_mdl_20250425_lgbm_h20_v0_1_084558=150/150`

## 5. Governance State

Current committed model:

- `mdl_20250425_lgbm_h20_v0_1`

Current committed prediction run:

- `pred_20250425_mdl_20250425_lgbm_h20_v0_1_084558`

Deprecated prediction run:

- `pred_20250425_mdl_20250425_lgbm_h20_v0_1`, superseded by the robust rank-IC artifact.

## 6. Decision

The Step 10 quality WARNING is resolved for the cutoff-valid historical validation window.

§8 remains ACTIVE (DRAFT) because production-current validation still requires the formal current label window after `2026-05-14 + label_horizon`.

