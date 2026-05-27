# Downstream Promotion Readiness Report

- generated_at: 2026-05-27 11:14:18
- constitution: 系統架構大憲章_v6.0.0.md §8
- tool: audit_downstream_readiness.py v0.2
- verdict: FAILED
- PASS/WARN/FAIL: 13/1/1

## Current Evidence

- model_id: `N/A`
- feature_set_id: `N/A`
- feature_set_version: `N/A`
- prediction_run_id: `N/A`
- as_of_date: `N/A`
- historical model cutoff: `db max_price_date=N/A`
- production-current cutoff: `required_label_date` gate (not historical cutoff)

## Production-Current Gate

- production_snapshot_id: `N/A`
- production_as_of_date: `N/A`
- required_label_date: `N/A`
- max_price_date: `N/A`

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
- **FAIL** `required_table`: prediction_values missing
- **PASS** `required_table`: core_universe_snapshot exists
- **PASS** `required_table`: core_universe_membership exists
- **PASS** `required_table`: TaiwanStockPriceAdj exists
- **WARN** `audit_log`: write_data_audit_log failed: InvalidColumnReference: there is no unique or exclusion constraint matching the ON CONFLICT specification

