# Core Universe Audit Report (v0.1)

- constitution: v5.4.22
- snapshot_id: core_universe_20260514_core_universe_policy_v0_1
- as_of_date: 2026-05-14
- policy_version: core_universe_policy_v0.1
- verdict: PERFECT
- PASS/WARN/FAIL: 36/0/0

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
| PASS | `snapshot_resolve` | snapshot=core_universe_20260514_core_universe_policy_v0_1, status=committed |
| PASS | `snapshot_status` | snapshot status is committed |
| PASS | `policy` | policy_version=core_universe_policy_v0.1 active |
| PASS | `policy_source` | policy source_table=TaiwanStockInfo |
| PASS | `policy_boundary` | downstream eligibility remains pending/all false |
| PASS | `policy_pending_scores` | liquidity/fundamental scores are policy-pending in v0.1 |
| PASS | `membership_count` | membership_count=2799 matches snapshot.total_candidates |
| PASS | `scores_count` | scores_count=2799 matches snapshot.total_candidates |
| PASS | `tier_allowed` | all membership tiers are governed tiers |
| PASS | `tier_count` | research_universe=2271 matches snapshot.research_count |
| PASS | `tier_count` | core_universe=120 matches snapshot.core_count |
| PASS | `tier_count` | convex_universe=30 matches snapshot.convex_count |
| PASS | `tier_count` | quarantine_universe=378 matches snapshot.quarantine_count |
| PASS | `core_size` | core_count=120 within v0.1 limit |
| PASS | `convex_size` | convex_count=30 within v0.1 limit |
| PASS | `membership_unique` | no duplicate membership stock_id in snapshot |
| PASS | `scores_unique` | no duplicate scores stock_id in snapshot |
| PASS | `membership_scores_pairing` | membership and scores are 1:1 paired |
| PASS | `raw_unique` | TaiwanStockInfo stock_id is unique |
| PASS | `raw_membership_source` | all membership stock_id values exist in TaiwanStockInfo |
| PASS | `raw_column_mirror` | stock_name/type/industry_category mirror TaiwanStockInfo |
| PASS | `downstream_eligibility_boundary` | all downstream eligibility flags remain false across 2799 rows |
| PASS | `pending_scores_boundary` | liquidity/fundamental/institutional/volatility scores remain NULL in v0.1 |
| PASS | `score_scope` | all score_detail records declare metadata_bootstrap_only |
| PASS | `revision_log` | BUILD_SNAPSHOT revision rows=2 |
| PASS | `data_audit_log` | CORE_UNIVERSE_BUILD audit rows=10 |
| PASS | `pipeline_lifecycle` | core_universe_builder success lifecycle rows=2 |
| PASS | `audit_self_log` | CORE_UNIVERSE_AUDIT written to data_audit_log |
