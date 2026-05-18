# Feature Store After CoreScore v0.2 Dry-Run Log

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` §8.2
Purpose: 確認 Feature Store Step 9 正確鎖定 CoreScore v0.2 committed universe。

## 1. Command

```bash
python scripts/core/feature_store_builder.py --dry-run --as-of-date 2026-05-14 --feature-set-version feature_set_v0.1_h20_after_corescore_v02 --label-horizon 20
```

## 2. Result

- verdict: PERFECT
- feature_set_id: `fs_20260514_feature_set_v0_1_h20_after_corescore_v02`
- feature_set_version: `feature_set_v0.1_h20_after_corescore_v02`
- universe_snapshot_id: `core_universe_20260514_core_universe_policy_v0_2`
- policy_version: `core_universe_policy_v0.2`
- as_of_date: `2026-05-14`
- label_horizon: `20`
- stocks scored: `150`
- features defined: `27`
- feature rows: `3980`
- null imputed: `47`
- preflight PASS/WARN/FAIL: `14/0/0`
- warnings: `0`
- failures: `0`

## 3. DB Write Verification

Dry-run did not write governance tables:

- `feature_store_snapshot` rows for this feature_set_id: `0`
- `feature_values` rows for this feature_set_id: `0`

## 4. Decision

Feature Store Step 9 preflight is ready after CoreScore v0.2.

The builder correctly locks:

- latest committed CoreScore v0.2 universe snapshot
- core+convex 150 stocks
- 27 as-of-strict features

No commit was performed in this step. This feature set can be committed later when production-current validation is ready to proceed, but Step 10 production-current h20 training still requires label data on or after `2026-06-03`.

