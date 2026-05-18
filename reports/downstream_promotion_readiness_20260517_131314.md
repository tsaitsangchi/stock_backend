# Downstream Promotion Readiness Report

- generated_at: 2026-05-17 13:13:14
- constitution: 系統架構大憲章_v5.4.22.md §8
- tool: audit_downstream_readiness.py v0.1
- verdict: READY_FOR_DRAFT_EVIDENCE
- PASS/WARN/FAIL: 29/1/0

## Current Evidence

- model_id: `mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- feature_set_id: `fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation`
- feature_set_version: `feature_set_v0.1_h20_20250515_cutoff_rankic_validation`
- prediction_run_id: `pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- as_of_date: `2025-04-25`
- model trading data cutoff: `2025-05-15`

## Production-Current Gate

- production_snapshot_id: `core_universe_20260514_core_universe_policy_v0_2`
- production_as_of_date: `2026-05-14`
- required_label_date: `2026-06-03`
- max_price_date: `2026-05-15`

## Decision

§8 has clean historical h20 evidence, but v5.4.23 promotion is blocked until production-current label data is available.

## Checks

- **PASS** `required_file`: scripts/core/feature_store_builder.py exists
- **PASS** `required_file`: scripts/core/model_trainer.py exists
- **PASS** `required_file`: scripts/core/prediction_engine.py exists
- **PASS** `required_file`: scripts/maintenance/audit_leakage.py exists
- **PASS** `required_table`: feature_store_snapshot exists
- **PASS** `required_table`: feature_definition exists
- **PASS** `required_table`: feature_values exists
- **PASS** `required_table`: model_registry exists
- **PASS** `required_table`: model_training_run exists
- **PASS** `required_table`: prediction_run exists
- **PASS** `required_table`: prediction_values exists
- **PASS** `required_table`: core_universe_snapshot exists
- **PASS** `required_table`: core_universe_membership exists
- **PASS** `required_table`: TaiwanStockPriceAdj exists
- **PASS** `committed_model_cardinality`: current prediction-backed model=mdl_20250425_lgbm_h20_5c7f36c2_v0_1; committed_model_count=9; historical walk-forward models allowed
- **PASS** `model_id_governance`: all committed model_ids include feature_set_version hash: count=9
- **PASS** `label_horizon`: all committed models horizon=20: count=9
- **PASS** `model_data_cutoff`: all committed label_date_max <= 2025-05-15: count=9
- **PASS** `model_quality`: all committed models have ic_mean > 0: count=9; current_trainer=robust_rank_ic_baseline_v0.1
- **PASS** `model_artifact`: all committed model artifacts exist: count=9
- **PASS** `feature_set_status`: fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation committed
- **PASS** `feature_universe_lock`: model and feature_set universe match
- **PASS** `feature_label_horizon`: horizon=20
- **PASS** `feature_coverage`: stocks=150, feature_count=27
- **PASS** `feature_values`: rows=2965, imputed=83
- **PASS** `committed_prediction_cardinality`: run=pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1
- **PASS** `prediction_lock`: prediction run matches model feature_set/universe
- **PASS** `prediction_coverage`: rows_written=150, values=150
- **PASS** `historical_clean_validation`: Step 9->10->11->11A evidence is clean for draft acceptance
- **WARN** `production_current_label_window`: blocked: max_price_date=2026-05-15, required_label_date=2026-06-03
