# Core Universe Audit Report (v0.3)

- constitution: v6.1.0
- snapshot_id: core_universe_20260527_core_universe_policy_v0_13_doctrine_native_gate
- as_of_date: 2026-05-27
- policy_version: core_universe_policy_v0.13_doctrine_native_gate
- verdict: FAILED
- PASS/WARN/FAIL: 29/1/11

| status | check | detail |
| :--- | :--- | :--- |
| PASS | `required_table` | pipeline_execution_log exists |
| PASS | `required_table` | data_audit_log exists |
| PASS | `required_table` | TaiwanStockInfo exists |
| PASS | `required_table` | core_universe_policy exists |
| PASS | `required_table` | core_universe_snapshot exists |
| PASS | `required_table` | core_universe_membership exists |
| PASS | `required_table` | core_universe_scores exists |
| PASS | `required_table` | universe_revision_log exists |
| PASS | `snapshot_resolve` | snapshot=core_universe_20260527_core_universe_policy_v0_13_doctrine_native_gate, status=committed |
| PASS | `snapshot_status` | snapshot status is committed |
| PASS | `policy` | policy_version=core_universe_policy_v0.13_doctrine_native_gate active |
| FAIL | `policy_source` | policy source_table=None, expected TaiwanStockInfo |
| FAIL | `policy_boundary` | unexpected downstream eligibility policy: None |
| FAIL | `policy_pending_scores` | unexpected pending score states: liquidity=None, fundamental=None |
| PASS | `annual_snapshot_note` | snapshot notes do not declare special rebalance |
| FAIL | `membership_review_cycle` | unexpected review_cycle values=[None], expected annual |
| FAIL | `rebalance_revision_trace` | BUILD_SNAPSHOT revision detail missing |
| PASS | `same_day_reason_dedup` | §8.8.6 第 2 條：無同日重複 special override reason |
| FAIL | `membership_count` | membership_count=1583, expected 2803 |
| FAIL | `scores_count` | scores_count=0, expected 2803 |
| PASS | `tier_allowed` | all membership tiers are governed tiers |
| PASS | `tier_count` | research_universe=0 matches snapshot.research_count |
| PASS | `tier_count` | core_universe=1583 matches snapshot.core_count |
| PASS | `tier_count` | convex_universe=0 matches snapshot.convex_count |
| PASS | `tier_count` | quarantine_universe=0 matches snapshot.quarantine_count |
| PASS | `core_size` | core_count=1583 — dynamic per §14.7-BW |
| PASS | `convex_size` | convex_count=0 — dynamic per §14.7-BW (0=v0.10 normal) |
| PASS | `membership_unique` | no duplicate membership stock_id in snapshot |
| PASS | `scores_unique` | no duplicate scores stock_id in snapshot |
| FAIL | `membership_scores_pairing` | missing_scores=1583, missing_membership=0 |
| PASS | `raw_unique` | TaiwanStockInfo stock_id is unique |
| PASS | `raw_membership_source` | all membership stock_id values exist in TaiwanStockInfo |
| FAIL | `raw_column_mirror` | raw mirror mismatches=1583 |
| PASS | `downstream_eligibility_boundary` | all downstream eligibility flags remain false across 1583 rows |
| PASS | `pending_scores_boundary` | liquidity/fundamental/institutional/volatility scores remain NULL in v0.1 |
| PASS | `score_scope` | all score_detail records declare v0.13_doctrine_native_gate_§CG |
| PASS | `score_detail_keys` | policy=core_universe_policy_v0.13_doctrine_native_gate has no sub-score detail expectation (baseline) |
| FAIL | `revision_log` | BUILD_SNAPSHOT revision log missing |
| FAIL | `data_audit_log` | CORE_UNIVERSE_BUILD audit rows=0, expected >= 5 |
| PASS | `pipeline_lifecycle` | core_universe_builder accepted lifecycle rows=8 |
| WARN | `audit_self_log` | CORE_UNIVERSE_AUDIT write failed: InvalidColumnReference: there is no unique or exclusion constraint matching the ON CONFLICT specification
 |
