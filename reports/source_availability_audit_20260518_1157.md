# Core 150 strict source availability audit

- **time**: 2026-05-18 11:57:27
- **constitution**: 系統架構大憲章_v6.0.0.md §14.7-L
- **tool**: audit_source_availability v0.1
- **start_date**: 1990-01-01
- **scope**: stocks=150, datasets=9
- **verdict**: **PERFECT**
- **summary**: checked=1350, source_empty_ok=13, mismatch=0, api_errors=0

- **fred_summary**: checked=4, mismatch=0, api_errors=0

## Mismatches

None.


## FRED Valid Observation Alignment

| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |
|---|---|---:|---|---|---:|---|---|
| DFF | OK | 26251 | 1954-07-01 | 2026-05-14 | 26251 | 1954-07-01 | 2026-05-14 |
| UNRATE | OK | 939 | 1948-01-01 | 2026-04-01 | 939 | 1948-01-01 | 2026-04-01 |
| T10Y2Y | OK | 12486 | 1976-06-01 | 2026-05-15 | 12486 | 1976-06-01 | 2026-05-15 |
| VIXCLS | OK | 9186 | 1990-01-02 | 2026-05-14 | 9186 | 1990-01-02 | 2026-05-14 |
