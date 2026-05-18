# Downstream Production-Current Restart Plan

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` §8 draft
Purpose: 固定未來 production-current 驗收重啟條件、指令序列與升版判定。

## 1. Current State

Current §8 state:

- Historical clean h20 draft evidence: COMPLETE
- Leakage audit: PERFECT
- Readiness audit: READY_FOR_DRAFT_EVIDENCE
- v5.4.23 promotion: BLOCKED

Current committed evidence:

- feature_set_id: `fs_20250425_feature_set_v0_1_h20_20250515_cutoff_rankic_validation`
- model_id: `mdl_20250425_lgbm_h20_5c7f36c2_v0_1`
- prediction_run_id: `pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1`

Current production gate:

- latest committed production snapshot: `core_universe_20260514_core_universe_policy_v0_2`
- production as_of_date: `2026-05-14`
- formal label_horizon: `20`
- required label date: `2026-06-03`
- current DB max price date at last audit: `2026-05-15`

## 2. Restart Preconditions

Before running production-current validation, verify:

1. `TaiwanStockPriceAdj` has price data through at least `2026-06-03`.
2. The latest committed `core_universe_snapshot` is still `core_universe_20260514_core_universe_policy_v0_2`.
3. If the latest committed snapshot has changed, recompute:
   - `production_as_of_date`
   - `required_label_date = production_as_of_date + label_horizon`
   - all Step 9 feature_set ids and model ids from the new date.
4. `audit_leakage.py` must remain PERFECT before promotion review.
5. `audit_downstream_readiness.py` must return `READY_FOR_V5_4_23` before §8 can be promoted.

Precheck command:

```bash
python -c 'import sys; sys.path.insert(0,"scripts"); from core.db_utils import get_db_connection; conn=get_db_connection(); cur=conn.cursor(); cur.execute("SELECT MAX(date) FROM \"TaiwanStockPriceAdj\""); print("max_price_date", cur.fetchone()[0]); cur.execute("SELECT snapshot_id, as_of_date FROM core_universe_snapshot WHERE status='\''committed'\'' ORDER BY as_of_date DESC, created_at DESC, snapshot_id DESC LIMIT 1"); print("latest_snapshot", cur.fetchone()); cur.close(); conn.close()'
```

Expected for the current production gate:

```text
max_price_date >= 2026-06-03
latest_snapshot = ('core_universe_20260514_core_universe_policy_v0_2', 2026-05-14)
```

## 3. Production-Current Command Sequence

The following commands assume the latest committed production snapshot remains `as_of_date=2026-05-14`.

### Step 9: Feature Store Build

Dry-run:

```bash
python scripts/core/feature_store_builder.py --dry-run --as-of-date 2026-05-14 --feature-set-version feature_set_v0.1_h20_production_current --label-horizon 20
```

Commit:

```bash
python scripts/core/feature_store_builder.py --commit --as-of-date 2026-05-14 --feature-set-version feature_set_v0.1_h20_production_current --label-horizon 20
```

Expected feature_set_id:

```text
fs_20260514_feature_set_v0_1_h20_production_current
```

### Step 10: Strict h20 Model Training

Dry-run:

```bash
python scripts/core/model_trainer.py --dry-run --feature-set-id fs_20260514_feature_set_v0_1_h20_production_current --model-family lgbm --label-horizon 20
```

Commit:

```bash
python scripts/core/model_trainer.py --commit --feature-set-id fs_20260514_feature_set_v0_1_h20_production_current --model-family lgbm --label-horizon 20
```

Expected properties:

- `label_date_min >= 2026-06-03`
- `label_date_max >= 2026-06-03`
- `model_id` includes the `feature_set_version` hash.
- `ic_mean > 0`
- status: PERFECT

If Step 10 is WARNING or FAILED, do not promote §8.

### Step 11: Prediction

Use the committed model id emitted by Step 10.

Template:

```bash
python scripts/core/prediction_engine.py --commit --model-id <MODEL_ID_FROM_STEP_10> --as-of-date 2026-05-14
```

Expected:

- 150 predictions
- model feature_set and universe locked
- status: PERFECT

### Step 11A: Leakage Audit

```bash
python scripts/maintenance/audit_leakage.py
```

Expected:

- status: PERFECT
- no model_id governance violation
- no label horizon violation
- no prediction coverage violation

### Step 11B: Promotion Readiness Audit

```bash
python scripts/maintenance/audit_downstream_readiness.py
```

Expected:

```text
READY_FOR_V5_4_23
```

If output is:

- `READY_FOR_DRAFT_EVIDENCE`: production-current label window or another final gate is still not satisfied.
- `FAILED`: fix failed checks before any promotion discussion.

## 4. Promotion Rule

§8 may only be promoted from ACTIVE (DRAFT) to v5.4.23 if all are true:

1. Step 9 production-current feature build is PERFECT.
2. Step 10 production-current h20 training is PERFECT.
3. Step 11 production-current prediction is PERFECT.
4. Step 11A leakage audit is PERFECT.
5. Step 11B readiness audit returns `READY_FOR_V5_4_23`.
6. `系統架構大憲章_v5.4.22.md` is updated into a new `v5.4.23` document with §8 changed from draft to mandatory contract.

## 5. Non-Promotion Conditions

Do not promote §8 if any of the following occurs:

- `TaiwanStockPriceAdj.MAX(date) < required_label_date`
- latest committed core universe snapshot changes and the command sequence is not recomputed
- Step 10 `ic_mean <= 0`
- committed model id does not include the `feature_set_version` hash
- any committed prediction run has coverage other than 150/150
- `audit_leakage.py` is not PERFECT
- `audit_downstream_readiness.py` is not `READY_FOR_V5_4_23`

## 6. Current Final Decision

As of 2026-05-17:

- No further §8 implementation work is required for draft evidence.
- No v5.4.23 promotion is allowed yet.
- Next operational trigger is DB price availability through `2026-06-03`.

