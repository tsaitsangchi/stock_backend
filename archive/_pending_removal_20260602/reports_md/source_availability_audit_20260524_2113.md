# Strict source availability audit

- **time**: 2026-05-24 21:13:38
- **constitution**: 系統架構大憲章_v6.1.0.md §14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ
- **tool**: audit_source_availability v0.7
- **start_date**: 1990-01-01
- **drift_tolerance**: 3 day(s) (§6.8.8-C; 0 = strict)
- **scope**: stocks=2771 (universe=full), datasets=9
- **special_full_market_reason**: v6.1.0 recursive item3 phase7 audit3 immediate launch after quota recovery (§6.8.7 第 (4) 條 / §6.8.8-D)
- **verdict**: **PERFECT**
- **summary**: checked=24939, source_empty_ok=2946, time_drift_ok=0, mismatch=0, api_errors=0

- **fred_summary**: checked=4, time_drift_ok=0, mismatch=0, api_errors=0

## Mismatches

None.


## FRED Valid Observation Alignment

| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |
|---|---|---:|---|---|---:|---|---|
| DFF | OK | 26258 | 1954-07-01 | 2026-05-21 | 26258 | 1954-07-01 | 2026-05-21 |
| UNRATE | OK | 939 | 1948-01-01 | 2026-04-01 | 939 | 1948-01-01 | 2026-04-01 |
| T10Y2Y | OK | 12491 | 1976-06-01 | 2026-05-22 | 12491 | 1976-06-01 | 2026-05-22 |
| VIXCLS | OK | 9191 | 1990-01-02 | 2026-05-21 | 9191 | 1990-01-02 | 2026-05-21 |
