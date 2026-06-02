# Core Universe Audit Report (v0.3)

- constitution: v6.1.0
- snapshot_id: core_universe_20260527_core_universe_policy_v0_10_pure_doctrine_weekly
- as_of_date: 2026-05-27
- policy_version: core_universe_policy_v0.10_pure_doctrine_weekly
- verdict: FAILED
- PASS/WARN/FAIL: 32/2/7

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
| PASS | `snapshot_resolve` | snapshot=core_universe_20260527_core_universe_policy_v0_10_pure_doctrine_weekly, status=committed |
| PASS | `snapshot_status` | snapshot status is committed |
| PASS | `policy` | policy_version=core_universe_policy_v0.10_pure_doctrine_weekly active |
| PASS | `policy_source` | policy source_table=TaiwanStockInfo |
| PASS | `policy_boundary` | downstream eligibility remains pending/all false |
| FAIL | `policy_pending_scores` | unexpected pending score states: liquidity=None, fundamental=None |
| PASS | `annual_snapshot_note` | snapshot notes do not declare special rebalance |
| PASS | `membership_review_cycle` | all membership rows review_cycle=annual |
| FAIL | `rebalance_revision_trace` | BUILD_SNAPSHOT revision detail missing |
| PASS | `same_day_reason_dedup` | §8.8.6 第 2 條：無同日重複 special override reason |
| PASS | `membership_count` | membership_count=2799 matches snapshot.total_candidates |
| FAIL | `scores_count` | scores_count=0, expected 2799 |
| PASS | `tier_allowed` | all membership tiers are governed tiers |
| PASS | `tier_count` | research_universe=942 matches snapshot.research_count |
| PASS | `tier_count` | core_universe=1857 matches snapshot.core_count |
| PASS | `tier_count` | convex_universe=0 matches snapshot.convex_count |
| PASS | `tier_count` | quarantine_universe=0 matches snapshot.quarantine_count |
| PASS | `core_size` | core_count=1857 — dynamic per §14.7-BW |
| PASS | `convex_size` | convex_count=0 — dynamic per §14.7-BW (0=v0.10 normal) |
| PASS | `membership_unique` | no duplicate membership stock_id in snapshot |
| PASS | `scores_unique` | no duplicate scores stock_id in snapshot |
| FAIL | `membership_scores_pairing` | missing_scores=2799, missing_membership=0 |
| PASS | `raw_unique` | TaiwanStockInfo stock_id is unique |
| PASS | `raw_membership_source` | all membership stock_id values exist in TaiwanStockInfo |
| PASS | `raw_column_mirror` | stock_name/type/industry_category mirror TaiwanStockInfo |
| FAIL | `downstream_eligibility_boundary` | unexpected true eligibility counts: {'train_eligible': 1857, 'predict_eligible': 1857, 'backtest_eligible': 1857, 'downstream_ready': 1857} |
| PASS | `pending_scores_boundary` | liquidity/fundamental/institutional/volatility scores remain NULL in v0.1 |
| PASS | `score_scope` | all score_detail records declare metadata_bootstrap_only |
| WARN | `score_detail_keys` | policy=core_universe_policy_v0.10_pure_doctrine_weekly not in EXPECTED_SCORE_DETAIL_KEYS map (skipped) |
| FAIL | `revision_log` | BUILD_SNAPSHOT revision log missing |
| FAIL | `data_audit_log` | CORE_UNIVERSE_BUILD audit rows=0, expected >= 5 |
| PASS | `pipeline_lifecycle` | core_universe_builder accepted lifecycle rows=3 |
| WARN | `audit_self_log` | CORE_UNIVERSE_AUDIT write failed: InvalidColumnReference: there is no unique or exclusion constraint matching the ON CONFLICT specification
 |
