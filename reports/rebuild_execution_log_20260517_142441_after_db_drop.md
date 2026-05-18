# Rebuild Execution Log After DB Full Drop

- date: 2026-05-17
- timezone: Asia/Taipei
- constitution: `reports/系統架構大憲章_v5.4.22.md`
- scope: database all tables deleted by user; rebuild from zero to accepted CoreScore v0.2 snapshot and downstream smoke
- code baseline: modified `core_universe_builder.py` + modified `audit_core_universe.py` with special restore trace audit
- final verdict: ACCEPTED WITH GOVERNED WARNINGS

## 1. Execution Sequence

| Step | Command | Verdict | Result |
|---|---|---|---|
| 1 | `python scripts/core/path_setup.py` | PERFECT | 25-path sovereignty aligned; `.env` anchor matched; DB logging hook reported `BOOTSTRAP-DEFERRED`, expected before schema exists. |
| 2 | `python scripts/core/data_schema.py --init --force` | PERFECT ALIGNMENT | API contract PASS/WARN/FAIL=`11/0/0`; 13 raw/log tables created. |
| 2B | `python scripts/core/core_universe_schema.py --init` | PERFECT | Core universe governance 7 tables created; preflight=`9/0/0`. |
| 2C | `python scripts/core/db_utils.py` | WARNING | DB connected and log tables active; §6.7 core assets=`0`, legal bootstrap warning before snapshot restore. |
| 3 | `python scripts/maintenance/audit_supply_chain.py --include-logs` | PERFECT | Report `compliance_audit_20260517_1357.md`; PASS/WARN/FAIL=`29/0/0`. |
| 4 | `python scripts/ingestion/sovereign_sync_engine.py --seed` | PERFECT | `TaiwanStockInfo` fetched 3402 rows; FRED 4 series synced; total writes=`7287`. |
| 7A-1 | `python scripts/core/core_universe_builder.py --dry-run --as-of-date 2026-05-14` | WARNING | Empty DB individual history stage; v0.2 contract=`10/10/0`; total candidates=2798, core=120, convex=30, research=2270, quarantine=378. |
| 7B-1 | `python scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-14 --special-rebalance-reason "database full rebuild restore accepted CoreScore v0.2 snapshot bootstrap before historical refill"` | WARNING | Bootstrap restore committed to allow core+convex sync; written rows=`5599`; rebalance mode=`special`. |
| 7C | `python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730` | WARNING | Locked current committed core+convex universe; success=`1308`, warning=`46`, failed=`0`, rows=`688867`, elapsed=`706.50s`. |
| 7A-2 | `python scripts/core/core_universe_builder.py --dry-run --as-of-date 2026-05-14` | WARNING | Post-history v0.2 contract=`16/4/0`; core-sync coverage: price `146/150`, revenue `140/150`, financial `147/150`. |
| 7B-2 | `python scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-14 --special-rebalance-reason "database full rebuild restore accepted CoreScore v0.2 snapshot after historical refill"` | WARNING | Final accepted restore commit; snapshot=`core_universe_20260514_core_universe_policy_v0_2`; written rows=`5599`; rebalance mode=`special`. |
| 8 | `python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14` | PERFECT | Report `core_universe_audit_20260517_1410.md`; PASS/WARN/FAIL=`40/0/0`; includes special restore trace audit. |
| Final infra | `python scripts/core/db_utils.py` | PERFECT | §6.7 core assets=`150`; pipeline/data audit logs active. |
| §8 schema | `python scripts/core/feature_store_schema.py --init` | PERFECT | Feature Store 3 tables created; preflight=`6/0/0`. |
| §8 dry-run | `python scripts/core/feature_store_builder.py --dry-run --as-of-date 2026-05-14 --feature-set-version feature_set_v0.1_h20_production_current --label-horizon 20` | PERFECT | Locked v0.2 snapshot; 150 stocks, 27 features, 3980 rows, 47 imputed; dry-run only. |
| Final audit | `python scripts/maintenance/audit_supply_chain.py --include-logs` | PERFECT | Report `compliance_audit_20260517_1411.md`; PASS/WARN/FAIL=`33/0/0`. |

## 2. Final DB State

```text
TaiwanStockInfo             2798
FredData                    3885
core_universe_snapshot      1
core_universe_membership    2798
core_universe_scores        2798
feature_store_snapshot      0
feature_definition          0
feature_values              0
TaiwanStockPriceAdj         69885 rows / 148 stocks / 2024-05-17..2026-05-15
```

Latest snapshot:

```text
snapshot_id       core_universe_20260514_core_universe_policy_v0_2
status            committed
total_candidates  2798
core              120
convex            30
research          2270
quarantine        378
notes             rebalance_mode=special; special_rebalance_reason=database full rebuild restore accepted CoreScore v0.2 snapshot after historical refill
```

Tier counts:

```text
convex_universe      30
core_universe        120
quarantine_universe  378
research_universe    2270
```

Special restore trace:

```text
membership.review_cycle = special for 2798 rows
latest_revision.rebalance_mode = special
latest_revision.review_cycle = special
latest_revision.special_rebalance_reason = database full rebuild restore accepted CoreScore v0.2 snapshot after historical refill
```

## 3. Issues And Rulings

1. `db_utils.py` returned §6.7 core assets=`0` before core snapshot restore.
   - Ruling: legal bootstrap warning. It converged to PERFECT after final restore, with §6.7 core assets=`150`.

2. Initial CoreScore dry-run and bootstrap commit returned v0.2 contract warnings.
   - Ruling: expected after full table deletion because individual historical tables were empty. The first special commit was only a temporary restore anchor for core+convex data sync, not a new daily stock selection.

3. `core_universe_builder.py --commit --as-of-date 2026-05-14` required `--special-rebalance-reason`.
   - Ruling: correct. `2026-05-14` is not an annual year-end rebalance date. This rebuild is a DB full rebuild restore of an accepted snapshot, not a regular annual reselection.

4. `sovereign_sync_engine.py --universe core --all --days 730` returned WARNING with 46 API 0-row warnings.
   - Ruling: non-blocking. Failed count was 0 and downstream coverage returned to accepted baseline.

5. Final CoreScore restore commit remained WARNING.
   - Ruling: accepted governed warning. Causes were special restore trace warning, market-wide zero coverage for non-core names, and `TaiwanStockInfo` as-of fallback. Core-sync coverage was restored to the accepted baseline: price `146/150`, revenue `140/150`, financial `147/150`.

6. `feature_store_snapshot`, `feature_definition`, and `feature_values` remain 0 rows.
   - Ruling: expected. This run only performed Feature Store dry-run smoke; it did not commit feature values and did not run production h20 training.

7. A direct sandbox DB summary query failed with `OperationalError`; the same query succeeded under authorized execution.
   - Ruling: execution environment boundary, not schema/data failure.

## 4. Final Ruling

The system was rebuilt from zero after all database tables were deleted. The accepted CoreScore v0.2 snapshot was restored with special restore governance trace, not regular daily reselection. Core universe audit, infrastructure diagnostic, Feature Store dry-run, and final supply-chain audit all passed. The system is ready for the next governed step: daily sync / downstream dry-run or, after the production-current h20 label window matures, Step 9 to Step 11A validation under §8.
