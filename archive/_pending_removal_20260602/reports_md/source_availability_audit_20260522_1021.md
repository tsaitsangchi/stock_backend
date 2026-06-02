# Core 150 strict source availability audit

- **time**: 2026-05-22 10:21:28
- **constitution**: 系統架構大憲章_v6.0.0.md §14.7-L
- **tool**: audit_source_availability v0.1
- **start_date**: 1990-01-01
- **scope**: stocks=1, datasets=3
- **verdict**: **FAILED**
- **summary**: checked=3, source_empty_ok=0, mismatch=0, api_errors=0

- **fred_summary**: checked=4, mismatch=3, api_errors=0

## Mismatches

None.


## FRED Valid Observation Alignment

| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |
|---|---|---:|---|---|---:|---|---|
| DFF | MISMATCH | 26257 | 1954-07-01 | 2026-05-20 | 26256 | 1954-07-01 | 2026-05-19 |
| UNRATE | OK | 939 | 1948-01-01 | 2026-04-01 | 939 | 1948-01-01 | 2026-04-01 |
| T10Y2Y | MISMATCH | 12490 | 1976-06-01 | 2026-05-21 | 12489 | 1976-06-01 | 2026-05-20 |
| VIXCLS | MISMATCH | 9190 | 1990-01-02 | 2026-05-20 | 9189 | 1990-01-02 | 2026-05-19 |
