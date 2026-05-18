# Downstream Promotion Readiness Report

- generated_at: 2026-05-18 13:12:23
- constitution: 系統架構大憲章_v6.0.0.md §8
- tool: audit_downstream_readiness.py v0.1
- verdict: READY_FOR_DRAFT_EVIDENCE
- PASS/WARN/FAIL: 29/1/0

## Current Evidence

- model_id: `mdl_20260425_lgbm_h20_d969ffb1_v0_1`
- feature_set_id: `fs_20260425_feature_set_v0_1_h20_historical_20260425_strict_source`
- feature_set_version: `feature_set_v0.1_h20_historical_20260425_strict_source`
- prediction_run_id: `pred_20260425_mdl_20260425_lgbm_h20_d969ffb1_v0_1`
- as_of_date: `2026-04-25`
- historical model cutoff: `db max_price_date=2026-05-15`
- production-current cutoff: `required_label_date` gate (not historical cutoff)

## Production-Current Gate

- production_snapshot_id: `core_universe_20260515_core_universe_policy_v0_2`
- production_as_of_date: `2026-05-15`
- required_label_date: `2026-06-04`
- max_price_date: `2026-05-15`

## Decision

§8 has clean historical h20 evidence, but successor promotion is blocked until production-current label data and delivery model are available.

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
- **PASS** `committed_model_cardinality`: current prediction-backed model=mdl_20260425_lgbm_h20_d969ffb1_v0_1; committed_model_count=1; historical walk-forward models allowed
- **PASS** `model_id_governance`: all committed model_ids include feature_set_version hash: count=1
- **PASS** `label_horizon`: all committed models horizon=20: count=1
- **PASS** `model_data_cutoff`: historical label_date_max <= db max_price_date=2026-05-15; production-current uses required_label_date gate: historical=1, production_current=0
- **PASS** `model_quality`: all committed models have ic_mean > 0: count=1; current_trainer=robust_rank_ic_baseline_v0.1
- **PASS** `model_artifact`: all committed model artifacts exist: count=1
- **PASS** `feature_set_status`: fs_20260425_feature_set_v0_1_h20_historical_20260425_strict_source committed
- **PASS** `feature_universe_lock`: model and feature_set universe match
- **PASS** `feature_label_horizon`: horizon=20
- **PASS** `feature_coverage`: stocks=150, feature_count=27
- **PASS** `feature_values`: rows=3975, imputed=51
- **PASS** `committed_prediction_cardinality`: run=pred_20260425_mdl_20260425_lgbm_h20_d969ffb1_v0_1
- **PASS** `prediction_lock`: prediction run matches model feature_set/universe
- **PASS** `prediction_coverage`: rows_written=150, values=150
- **PASS** `historical_clean_validation`: Step 9->10->11->11A evidence is clean for draft acceptance
- **WARN** `production_current_label_window`: blocked: max_price_date=2026-05-15, required_label_date=2026-06-04
