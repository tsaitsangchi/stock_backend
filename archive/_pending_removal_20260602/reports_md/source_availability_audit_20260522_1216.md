# Core 150 strict source availability audit

- **time**: 2026-05-22 12:16:31
- **constitution**: 系統架構大憲章_v6.0.0.md §14.7-L + §6.8.8-C + §14.7-AP
- **tool**: audit_source_availability v0.2
- **start_date**: 1990-01-01
- **drift_tolerance**: 3 day(s) (§6.8.8-C; 0 = strict)
- **scope**: stocks=150, datasets=9
- **verdict**: **PERFECT**
- **summary**: checked=1350, source_empty_ok=0, time_drift_ok=0, mismatch=0, api_errors=0

- **fred_summary**: checked=4, time_drift_ok=0, mismatch=0, api_errors=0

## Mismatches

None.


## FRED Valid Observation Alignment

| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |
|---|---|---:|---|---|---:|---|---|
| DFF | OK | 26257 | 1954-07-01 | 2026-05-20 | 26257 | 1954-07-01 | 2026-05-20 |
| UNRATE | OK | 939 | 1948-01-01 | 2026-04-01 | 939 | 1948-01-01 | 2026-04-01 |
| T10Y2Y | OK | 12490 | 1976-06-01 | 2026-05-21 | 12490 | 1976-06-01 | 2026-05-21 |
| VIXCLS | OK | 9190 | 1990-01-02 | 2026-05-20 | 9190 | 1990-01-02 | 2026-05-20 |
