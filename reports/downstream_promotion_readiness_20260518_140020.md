# Downstream Promotion Readiness Report

- generated_at: 2026-05-18 14:00:20
- constitution: 系統架構大憲章_v6.0.0.md §8
- tool: audit_downstream_readiness.py v0.1
- verdict: FAILED
- PASS/WARN/FAIL: 14/1/2

## Current Evidence

- model_id: `N/A`
- feature_set_id: `N/A`
- feature_set_version: `N/A`
- prediction_run_id: `N/A`
- as_of_date: `N/A`
- historical model cutoff: `db max_price_date=2026-05-15`
- production-current cutoff: `required_label_date` gate (not historical cutoff)

## Production-Current Gate

- production_snapshot_id: `core_universe_20260515_core_universe_policy_v0_2`
- production_as_of_date: `2026-05-15`
- required_label_date: `2026-06-04`
- max_price_date: `2026-05-15`

## Decision

§8 is not ready; failed checks must be fixed before promotion review.

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
- **FAIL** `committed_model_cardinality`: prediction-backed committed models=2, expected exactly 1; committed models=13
- **FAIL** `historical_clean_validation`: one or more clean validation checks failed
- **WARN** `production_current_label_window`: blocked: max_price_date=2026-05-15, required_label_date=2026-06-04
