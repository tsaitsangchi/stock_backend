# Strict source availability audit

- **time**: 2026-05-24 01:09:24
- **constitution**: 系統架構大憲章_v6.1.0.md §14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ
- **tool**: audit_source_availability v0.7
- **start_date**: 1990-01-01
- **drift_tolerance**: 3 day(s) (§6.8.8-C; 0 = strict)
- **scope**: stocks=2771 (universe=full), datasets=9
- **special_full_market_reason**: v6.1.0 §3.2A.I parallel 2-worker audit verification rerun (clean isolation) (§6.8.7 第 (4) 條 / §6.8.8-D)
- **verdict**: **FAILED**
- **summary**: checked=24939, source_empty_ok=2946, time_drift_ok=0, mismatch=1, api_errors=2

- **fred_summary**: checked=4, time_drift_ok=0, mismatch=0, api_errors=0

## Mismatches

| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max | error |
|---|---|---|---:|---|---|---:|---|---|---|
| 00875 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | ConnectionError: ConnectionError after 1 retries [30]: HTTPSConnectionPool(host='api.finmindtrade.com', port=443): Max retries exceeded with url: /api/v4/data?dataset=TaiwanStockShareholding&data_id=00875&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA (Caused by NameResolutionError("HTTPSConnection(host='api.finmindtrade.com', port=443): Failed to resolve 'api.finmindtrade.com' ([Errno 8] nodename nor servname provided, or not known)")) |
| 00876 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | ConnectionError: ConnectionError after 1 retries [30]: HTTPSConnectionPool(host='api.finmindtrade.com', port=443): Max retries exceeded with url: /api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=00876&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA (Caused by NameResolutionError("HTTPSConnection(host='api.finmindtrade.com', port=443): Failed to resolve 'api.finmindtrade.com' ([Errno 8] nodename nor servname provided, or not known)")) |
| 6983 | TaiwanStockShareholding | MISMATCH | 1 | 2026-05-22 | 2026-05-22 | 0 | None | None |  |

## Targeted Backfill Commands

```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6983 --dataset TaiwanStockShareholding --strict-source-history
```

## FRED Valid Observation Alignment

| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |
|---|---|---:|---|---|---:|---|---|
| DFF | OK | 26258 | 1954-07-01 | 2026-05-21 | 26258 | 1954-07-01 | 2026-05-21 |
| UNRATE | OK | 939 | 1948-01-01 | 2026-04-01 | 939 | 1948-01-01 | 2026-04-01 |
| T10Y2Y | OK | 12491 | 1976-06-01 | 2026-05-22 | 12491 | 1976-06-01 | 2026-05-22 |
| VIXCLS | OK | 9191 | 1990-01-02 | 2026-05-21 | 9191 | 1990-01-02 | 2026-05-21 |
