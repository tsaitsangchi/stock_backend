# Downstream Final Delivery Index

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` §8 draft
Purpose: 最終一致性掃描與交付索引。

## 1. Final Scan Result

Commands:

```bash
python scripts/maintenance/audit_leakage.py
python scripts/maintenance/audit_downstream_readiness.py --no-report
```

Results:

- Leakage audit: PERFECT, PASS/WARN/FAIL = 18/0/0
- Downstream readiness: READY_FOR_DRAFT_EVIDENCE, PASS/WARN/FAIL = 29/1/0
- 唯一 readiness WARN: production-current label window not mature
  - production as_of_date: `2026-05-14`
  - required label date: `2026-06-03`
  - DB max price date: `2026-05-15`

## 2. Current Committed Evidence

### Feature Store

Current feature set used by the committed model:

- feature_set_id: `fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation`
- feature_set_version: `feature_set_v0.1_h20_20250515_cutoff_rankic_validation`
- as_of_date: `2025-04-25`
- label_horizon: `20`
- status: `committed`
- total_stocks: `150`
- feature_count: `27`

### Model Registry

Current committed model:

- model_id: `mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- feature_set_id: `fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation`
- trainer: `robust_rank_ic_baseline_v0.1`
- ic_mean: `0.4911100393859014`
- label_date_max: `2025-05-15`
- model_id governance: includes `feature_set_version` hash `5c7f36c2`

### Prediction Run

Current committed prediction run:

- run_id: `pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- model_id: `mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- feature_set_id: `fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation`
- rows_written: `150`
- status: `committed`

## 3. Superseded / Reference Evidence

The following artifacts are retained for audit history but are not current promotion evidence:

- `mdl_20250425_lgbm_h20_v0_1`: deprecated; superseded by feature_set_version-hash model id governance.
- `pred_20250425_mdl_20250425_lgbm_h20_v0_1`: deprecated; old deterministic run.
- `pred_20250425_mdl_20250425_lgbm_h20_v0_1_084558`: deprecated; superseded by clean validation hash run.
- `mdl_20250514_lgbm_h20_v0_1`: deprecated; label date `2025-06-03` exceeds `2025-05-15` cutoff.
- `pred_20250514_mdl_20250514_lgbm_h20_v0_1`: deprecated; model deprecated by cutoff ruling.
- `mdl_20260514_lgbm_h1_v0_1`: deprecated; non-formal horizon h1.
- `pred_20260514_mdl_20260514_lgbm_h1_v0_1`: deprecated; model deprecated by horizon/cutoff ruling.
- `mdl_20260514_lgbm_v0_1`: deprecated; proxy-label legacy model.
- `pred_20260514_mdl_20260514_lgbm_v0_1`: deprecated; proxy-label legacy prediction.

Committed feature sets not locked by the current committed model are historical snapshots or reference runs. They remain in DB for auditability, but current §8 evidence is the rankic validation feature set listed above.

## 4. Current Implementation Files

Current §8 modules:

- `scripts/core/feature_store_schema.py`
- `scripts/core/feature_store_builder.py`
- `scripts/core/model_trainer.py`
- `scripts/core/prediction_engine.py`
- `scripts/maintenance/audit_leakage.py`
- `scripts/maintenance/audit_downstream_readiness.py`

Key implementation state:

- Feature builder: as-of strict, no labels, no predictions.
- Trainer: strict forward-return label, `label_horizon=20`, `robust_rank_ic_baseline_v0.1`.
- Model ID: `feature_set_version` SHA1 short hash included.
- Prediction engine: artifact-aligned winsor bounds and rank transform.
- Leakage audit: as-of strict, h20/cutoff, model_id governance, per-run coverage.
- Readiness audit: separates historical draft evidence from production-current promotion gate.

## 5. Current Reports

Current evidence reports:

- `reports/core_universe_v02_execution_20260517_091045.md`
- `reports/feature_store_after_corescore_v02_dryrun_20260517_091511.md`
- `reports/walk_forward_h20_training_execution_20260517_094442.md`
- `reports/downstream_clean_validation_model_id_hash_20260517_085247.md`
- `reports/model_trainer_rank_ic_rebuild_20260517_084631.md`
- `reports/downstream_20250515_cutoff_validation_20260517_055828.md`
- `reports/downstream_promotion_readiness_20260517_090108.md`
- `reports/downstream_production_current_restart_plan_20260517_090446.md`

Reference reports:

- `reports/downstream_h20_backtest_validation_20260516_215714.md`
- `reports/downstream_forward_return_revalidation_20260516_214735.md`
- `reports/downstream_step9_11_execution_log_20260516_213056.md`

## 6. Final Decision

§8 state:

- Historical clean h20 draft evidence: COMPLETE
- Leakage audit: PERFECT
- Readiness audit: READY_FOR_DRAFT_EVIDENCE
- v5.4.23 promotion: BLOCKED

Blocking condition:

- production-current label window is not mature.
- Current production as_of_date is `2026-05-14`.
- Formal h20 required label date is `2026-06-03`.
- DB currently has TaiwanStockPriceAdj only through `2026-05-15`.

Required future action:

After `2026-06-03` price data is available in DB, rerun production-current Step 9 -> Step 10 -> Step 11 -> Step 11A -> Step 11B. Only then should §8 promotion to v5.4.23 be reconsidered.
