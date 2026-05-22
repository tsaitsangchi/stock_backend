# Strict source availability audit

- **time**: 2026-05-22 15:21:48
- **constitution**: 系統架構大憲章_v6.0.0.md §14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ
- **tool**: audit_source_availability v0.5
- **start_date**: 1990-01-01
- **drift_tolerance**: 3 day(s) (§6.8.8-C; 0 = strict)
- **scope**: stocks=1 (universe=core), datasets=1
- **verdict**: **WARNING**
- **summary**: checked=1, source_empty_ok=0, time_drift_ok=0, mismatch=1, api_errors=0

## Mismatches

| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max | error |
|---|---|---|---:|---|---|---:|---|---|---|
| 2330 | TaiwanStockPrice | MISMATCH | 8001 | 1994-09-13 | 2026-05-22 | 8000 | 1994-09-13 | 2026-05-21 |  |

## Targeted Backfill Commands

```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPrice --strict-source-history
```
