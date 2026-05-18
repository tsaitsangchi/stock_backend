# Production-Current H20 Runbook (2026-06-04+)

- created_at: 2026-05-18 Asia/Taipei
- constitution: `reports/系統架構大憲章_v6.0.0.md` §8 / §14.7
- purpose: execute production-current h20 model promotion only after label data is available
- gate date: 2026-06-04 or later

## Preconditions

Do not start this runbook until all conditions are true:

1. `TaiwanStockPriceAdj.MAX(date) >= DATE '2026-06-04'`.
2. Daily sync for the latest available trading data has completed.
3. Strict source audit returns `PERFECT`.
4. `audit_leakage.py` and `audit_downstream_readiness.py` remain v0.2 or newer.
5. No new production-current model is committed before the label gate passes.

## Step 0: Check Label Availability

```bash
psql "$DATABASE_URL" -c 'SELECT MAX(date) AS max_price_date FROM "TaiwanStockPriceAdj";'
```

Expected:

```text
max_price_date >= 2026-06-04
```

## Step 1: Strict Source Re-Audit

```bash
.venv/bin/python scripts/maintenance/audit_source_availability.py \
  --universe core \
  --all \
  --include-fred \
  --strict \
  --snapshot-out /tmp/api_source_alignment_production_current_20260604.json
```

Required result:

```text
verdict=PERFECT
FinMind mismatch=0
FRED mismatch=0
```

If any mismatch appears, stop and backfill only the targeted `stock_id + dataset` pairs using `--strict-source-history`.

## Step 2: Build Production-Current Feature Store

```bash
.venv/bin/python scripts/core/feature_store_builder.py \
  --commit \
  --as-of-date 2026-05-15 \
  --feature-set-version feature_set_v0.1_h20_production_current_20260604 \
  --label-horizon 20
```

Required result:

```text
verdict=PERFECT
stocks=150
features>=27
```

## Step 3: Train Production-Current H20 Model

Use the `feature_set_id` printed by Step 2.

```bash
.venv/bin/python scripts/core/model_trainer.py \
  --commit \
  --feature-set-id <FEATURE_SET_ID_FROM_STEP_2> \
  --model-family lgbm \
  --label-horizon 20
```

Required result:

```text
verdict=PERFECT
label_date_max >= 2026-06-04
ic_mean > 0
```

## Step 4: Commit Prediction

Use the `model_id` printed by Step 3.

```bash
.venv/bin/python scripts/core/prediction_engine.py \
  --commit \
  --model-id <MODEL_ID_FROM_STEP_3> \
  --as-of-date 2026-05-15
```

Required result:

```text
verdict=PERFECT
predictions=150
```

If the prior historical h20 delivery remains committed, deprecate the old prediction run only after the new production-current prediction commits successfully.

## Step 5: Audits

```bash
.venv/bin/python scripts/maintenance/audit_leakage.py
.venv/bin/python scripts/maintenance/audit_downstream_readiness.py
```

Required result:

```text
audit_leakage: PERFECT
audit_downstream_readiness: READY_FOR_V5_4_23
```

`READY_FOR_V5_4_23` is the legacy-compatible verdict name for production-current readiness. If the tool has been renamed before this run, use the successor verdict documented in the charter.

## Step 6: Promotion Doctrine Gate

```bash
.venv/bin/python scripts/maintenance/audit_doctrine_compliance.py --for-promotion v6.1.0
```

Required result:

```text
PERFECT or promotion PASS
```

Do not propose v6.1.0 if this gate fails.

## Step 7: Archive

Create and commit:

- production-current feature/model/prediction report
- downstream readiness report
- doctrine promotion report
- charter patch recording the production-current evidence and v6.1.0 promotion decision

## Stop Conditions

Stop immediately if any of these occurs:

- strict source audit mismatch
- `label_date_max < 2026-06-04`
- committed prediction coverage is not 150/150
- more than one prediction-backed committed model remains after final cleanup
- readiness remains `READY_FOR_DRAFT_EVIDENCE` instead of production-current ready
