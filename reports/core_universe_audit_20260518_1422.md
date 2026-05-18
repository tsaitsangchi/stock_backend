# Core Universe Audit Report (v0.1)

- constitution: v6.0.0
- snapshot_id: core_universe_20260515_core_universe_policy_v0_2
- as_of_date: 2026-05-15
- policy_version: core_universe_policy_v0.2
- verdict: PERFECT
- PASS/WARN/FAIL: 41/0/0

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
| PASS | `snapshot_resolve` | snapshot=core_universe_20260515_core_universe_policy_v0_2, status=committed |
| PASS | `snapshot_status` | snapshot status is committed |
| PASS | `policy` | policy_version=core_universe_policy_v0.2 active |
| PASS | `policy_source` | policy source_table=TaiwanStockInfo |
| PASS | `policy_boundary` | downstream eligibility remains pending/all false |
| PASS | `policy_score_config` | v0.2 policy uses six-layer CoreScore weights |
| PASS | `special_snapshot_note` | special rebalance reason present in snapshot notes |
| PASS | `membership_review_cycle` | all membership rows review_cycle=special |
| PASS | `rebalance_revision_trace` | revision detail rebalance_mode/review_cycle=special |
| PASS | `special_revision_reason` | special_rebalance_reason present in revision detail |
| PASS | `same_day_reason_dedup` | §8.8.6 第 2 條：無同日重複 special override reason |
| PASS | `membership_count` | membership_count=2798 matches snapshot.total_candidates |
| PASS | `scores_count` | scores_count=2798 matches snapshot.total_candidates |
| PASS | `tier_allowed` | all membership tiers are governed tiers |
| PASS | `tier_count` | research_universe=2270 matches snapshot.research_count |
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
| PASS | `downstream_eligibility_boundary` | all downstream eligibility flags remain false across 2798 rows |
| PASS | `v02_scores_boundary` | v0.2 six-layer score columns populated rows=2798 |
| PASS | `score_scope` | all score_detail records declare v0.2_six_layer |
| PASS | `revision_log` | BUILD_SNAPSHOT revision rows=2 |
| PASS | `data_audit_log` | CORE_UNIVERSE_BUILD audit rows=10 |
| PASS | `pipeline_lifecycle` | core_universe_builder accepted lifecycle rows=2 |
| PASS | `audit_self_log` | CORE_UNIVERSE_AUDIT written to data_audit_log |
