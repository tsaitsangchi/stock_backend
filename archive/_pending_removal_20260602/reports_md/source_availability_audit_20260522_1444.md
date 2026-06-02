# Strict source availability audit

- **time**: 2026-05-22 14:44:53
- **constitution**: 系統架構大憲章_v6.0.0.md §14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ
- **tool**: audit_source_availability v0.4
- **start_date**: 1990-01-01
- **drift_tolerance**: 3 day(s) (§6.8.8-C; 0 = strict)
- **scope**: stocks=150 (universe=core), datasets=1
- **verdict**: **WARNING**
- **summary**: checked=150, source_empty_ok=0, time_drift_ok=0, mismatch=104, api_errors=0

## Mismatches

| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max |
|---|---|---|---:|---|---|---:|---|---|
| 1503 | TaiwanStockPrice | MISMATCH | 8768 | 1992-01-04 | 2026-05-22 | 8767 | 1992-01-04 | 2026-05-21 |
| 1504 | TaiwanStockPrice | MISMATCH | 8768 | 1992-01-04 | 2026-05-22 | 8767 | 1992-01-04 | 2026-05-21 |
| 1513 | TaiwanStockPrice | MISMATCH | 8117 | 1994-03-08 | 2026-05-22 | 8116 | 1994-03-08 | 2026-05-21 |
| 1514 | TaiwanStockPrice | MISMATCH | 8132 | 1994-03-26 | 2026-05-22 | 8131 | 1994-03-26 | 2026-05-21 |
| 1519 | TaiwanStockPrice | MISMATCH | 7263 | 1997-04-16 | 2026-05-22 | 7262 | 1997-04-16 | 2026-05-21 |
| 1560 | TaiwanStockPrice | MISMATCH | 5236 | 2005-01-31 | 2026-05-22 | 5235 | 2005-01-31 | 2026-05-21 |
| 1590 | TaiwanStockPrice | MISMATCH | 3778 | 2010-12-13 | 2026-05-22 | 3777 | 2010-12-13 | 2026-05-21 |
| 1795 | TaiwanStockPrice | MISMATCH | 4750 | 2006-12-13 | 2026-05-22 | 4749 | 2006-12-13 | 2026-05-21 |
| 2049 | TaiwanStockPrice | MISMATCH | 4646 | 2007-06-20 | 2026-05-22 | 4645 | 2007-06-20 | 2026-05-21 |
| 2059 | TaiwanStockPrice | MISMATCH | 5110 | 2005-04-28 | 2026-05-22 | 5109 | 2005-04-28 | 2026-05-21 |
| 2301 | TaiwanStockPrice | MISMATCH | 8737 | 1992-01-04 | 2026-05-22 | 8736 | 1992-01-04 | 2026-05-21 |
| 2303 | TaiwanStockPrice | MISMATCH | 8757 | 1992-01-04 | 2026-05-22 | 8756 | 1992-01-04 | 2026-05-21 |
| 2308 | TaiwanStockPrice | MISMATCH | 8768 | 1992-01-04 | 2026-05-22 | 8767 | 1992-01-04 | 2026-05-21 |
| 2313 | TaiwanStockPrice | MISMATCH | 8768 | 1992-01-04 | 2026-05-22 | 8767 | 1992-01-04 | 2026-05-21 |
| 2316 | TaiwanStockPrice | MISMATCH | 8750 | 1992-01-04 | 2026-05-22 | 8749 | 1992-01-04 | 2026-05-21 |
| 2317 | TaiwanStockPrice | MISMATCH | 8762 | 1992-01-04 | 2026-05-22 | 8761 | 1992-01-04 | 2026-05-21 |
| 2327 | TaiwanStockPrice | MISMATCH | 8200 | 1993-10-22 | 2026-05-22 | 8199 | 1993-10-22 | 2026-05-21 |
| 2329 | TaiwanStockPrice | MISMATCH | 8071 | 1994-04-20 | 2026-05-22 | 8070 | 1994-04-20 | 2026-05-21 |
| 2330 | TaiwanStockPrice | MISMATCH | 8001 | 1994-09-13 | 2026-05-22 | 8000 | 1994-09-13 | 2026-05-21 |
| 2337 | TaiwanStockPrice | MISMATCH | 7841 | 1995-03-15 | 2026-05-22 | 7840 | 1995-03-15 | 2026-05-21 |
| 2344 | TaiwanStockPrice | MISMATCH | 7688 | 1995-10-18 | 2026-05-22 | 7687 | 1995-10-18 | 2026-05-21 |
| 2345 | TaiwanStockPrice | MISMATCH | 7667 | 1995-11-15 | 2026-05-22 | 7666 | 1995-11-15 | 2026-05-21 |
| 2354 | TaiwanStockPrice | MISMATCH | 7408 | 1996-10-08 | 2026-05-22 | 7407 | 1996-10-08 | 2026-05-21 |
| 2356 | TaiwanStockPrice | MISMATCH | 7382 | 1996-11-13 | 2026-05-22 | 7381 | 1996-11-13 | 2026-05-21 |
| 2357 | TaiwanStockPrice | MISMATCH | 7355 | 1996-11-14 | 2026-05-22 | 7354 | 1996-11-14 | 2026-05-21 |
| 2360 | TaiwanStockPrice | MISMATCH | 7349 | 1996-12-21 | 2026-05-22 | 7348 | 1996-12-21 | 2026-05-21 |
| 2363 | TaiwanStockPrice | MISMATCH | 7147 | 1997-08-01 | 2026-05-22 | 7146 | 1997-08-01 | 2026-05-21 |
| 2368 | TaiwanStockPrice | MISMATCH | 7004 | 1998-03-09 | 2026-05-22 | 7003 | 1998-03-09 | 2026-05-21 |
| 2374 | TaiwanStockPrice | MISMATCH | 6776 | 1999-01-05 | 2026-05-22 | 6775 | 1999-01-05 | 2026-05-21 |
| 2376 | TaiwanStockPrice | MISMATCH | 6857 | 1998-09-24 | 2026-05-22 | 6856 | 1998-09-24 | 2026-05-21 |
| 2379 | TaiwanStockPrice | MISMATCH | 6827 | 1998-10-26 | 2026-05-22 | 6826 | 1998-10-26 | 2026-05-21 |
| 2382 | TaiwanStockPrice | MISMATCH | 6781 | 1999-01-08 | 2026-05-22 | 6780 | 1999-01-08 | 2026-05-21 |
| 2383 | TaiwanStockPrice | MISMATCH | 6809 | 1998-11-27 | 2026-05-22 | 6808 | 1998-11-27 | 2026-05-21 |
| 2395 | TaiwanStockPrice | MISMATCH | 6531 | 1999-12-13 | 2026-05-22 | 6530 | 1999-12-13 | 2026-05-21 |
| 2402 | TaiwanStockPrice | MISMATCH | 6510 | 2000-01-14 | 2026-05-22 | 6509 | 2000-01-14 | 2026-05-21 |
| 2404 | TaiwanStockPrice | MISMATCH | 6463 | 2000-03-14 | 2026-05-22 | 6462 | 2000-03-14 | 2026-05-21 |
| 2408 | TaiwanStockPrice | MISMATCH | 6334 | 2000-08-17 | 2026-05-22 | 6333 | 2000-08-17 | 2026-05-21 |
| 2412 | TaiwanStockPrice | MISMATCH | 6247 | 2000-10-27 | 2026-05-22 | 6246 | 2000-10-27 | 2026-05-21 |
| 2421 | TaiwanStockPrice | MISMATCH | 6333 | 2000-09-11 | 2026-05-22 | 6332 | 2000-09-11 | 2026-05-21 |
| 2441 | TaiwanStockPrice | MISMATCH | 6298 | 2000-10-26 | 2026-05-22 | 6297 | 2000-10-26 | 2026-05-21 |
| 2449 | TaiwanStockPrice | MISMATCH | 6165 | 2001-05-09 | 2026-05-22 | 6164 | 2001-05-09 | 2026-05-21 |
| 2451 | TaiwanStockPrice | MISMATCH | 6169 | 2001-05-03 | 2026-05-22 | 6168 | 2001-05-03 | 2026-05-21 |
| 2454 | TaiwanStockPrice | MISMATCH | 6113 | 2001-07-23 | 2026-05-22 | 6112 | 2001-07-23 | 2026-05-21 |
| 2455 | TaiwanStockPrice | MISMATCH | 5981 | 2002-01-24 | 2026-05-22 | 5980 | 2002-01-24 | 2026-05-21 |
| 2467 | TaiwanStockPrice | MISMATCH | 6075 | 2001-09-19 | 2026-05-22 | 6074 | 2001-09-19 | 2026-05-21 |
| 2481 | TaiwanStockPrice | MISMATCH | 6070 | 2001-09-19 | 2026-05-22 | 6069 | 2001-09-19 | 2026-05-21 |
| 2486 | TaiwanStockPrice | MISMATCH | 6075 | 2001-09-19 | 2026-05-22 | 6074 | 2001-09-19 | 2026-05-21 |
| 2492 | TaiwanStockPrice | MISMATCH | 6053 | 2001-09-19 | 2026-05-22 | 6052 | 2001-09-19 | 2026-05-21 |
| 3006 | TaiwanStockPrice | MISMATCH | 5968 | 2002-03-04 | 2026-05-22 | 5967 | 2002-03-04 | 2026-05-21 |
| 3013 | TaiwanStockPrice | MISMATCH | 5831 | 2002-09-16 | 2026-05-22 | 5830 | 2002-09-16 | 2026-05-21 |
| 3017 | TaiwanStockPrice | MISMATCH | 5822 | 2002-09-27 | 2026-05-22 | 5821 | 2002-09-27 | 2026-05-21 |
| 3019 | TaiwanStockPrice | MISMATCH | 5845 | 2002-08-26 | 2026-05-22 | 5844 | 2002-08-26 | 2026-05-21 |
| 3030 | TaiwanStockPrice | MISMATCH | 5801 | 2002-10-29 | 2026-05-22 | 5800 | 2002-10-29 | 2026-05-21 |
| 3036 | TaiwanStockPrice | MISMATCH | 5845 | 2002-08-26 | 2026-05-22 | 5844 | 2002-08-26 | 2026-05-21 |
| 3037 | TaiwanStockPrice | MISMATCH | 5845 | 2002-08-26 | 2026-05-22 | 5844 | 2002-08-26 | 2026-05-21 |
| 3044 | TaiwanStockPrice | MISMATCH | 5845 | 2002-08-26 | 2026-05-22 | 5844 | 2002-08-26 | 2026-05-21 |
| 3090 | TaiwanStockPrice | MISMATCH | 5473 | 2003-11-10 | 2026-05-22 | 5472 | 2003-11-10 | 2026-05-21 |
| 3167 | TaiwanStockPrice | MISMATCH | 4751 | 2006-12-13 | 2026-05-22 | 4750 | 2006-12-13 | 2026-05-21 |
| 3189 | TaiwanStockPrice | MISMATCH | 5301 | 2004-11-01 | 2026-05-22 | 5300 | 2004-11-01 | 2026-05-21 |
| 3231 | TaiwanStockPrice | MISMATCH | 5601 | 2003-08-19 | 2026-05-22 | 5600 | 2003-08-19 | 2026-05-21 |
| 3305 | TaiwanStockPrice | MISMATCH | 5066 | 2005-06-30 | 2026-05-22 | 5065 | 2005-06-30 | 2026-05-21 |
| 3406 | TaiwanStockPrice | MISMATCH | 5018 | 2005-12-20 | 2026-05-22 | 5017 | 2005-12-20 | 2026-05-21 |
| 3443 | TaiwanStockPrice | MISMATCH | 4802 | 2006-11-03 | 2026-05-22 | 4801 | 2006-11-03 | 2026-05-21 |
| 3481 | TaiwanStockPrice | MISMATCH | 4789 | 2006-10-24 | 2026-05-22 | 4788 | 2006-10-24 | 2026-05-21 |
| 3515 | TaiwanStockPrice | MISMATCH | 4748 | 2006-12-13 | 2026-05-22 | 4747 | 2006-12-13 | 2026-05-21 |
| 3533 | TaiwanStockPrice | MISMATCH | 4756 | 2006-12-13 | 2026-05-22 | 4755 | 2006-12-13 | 2026-05-21 |
| 3563 | TaiwanStockPrice | MISMATCH | 4640 | 2007-06-27 | 2026-05-22 | 4639 | 2007-06-27 | 2026-05-21 |
| 3653 | TaiwanStockPrice | MISMATCH | 4292 | 2008-11-20 | 2026-05-22 | 4291 | 2008-11-20 | 2026-05-21 |
| 3661 | TaiwanStockPrice | MISMATCH | 3764 | 2010-12-23 | 2026-05-22 | 3763 | 2010-12-23 | 2026-05-21 |
| 3665 | TaiwanStockPrice | MISMATCH | 3695 | 2011-04-14 | 2026-05-22 | 3694 | 2011-04-14 | 2026-05-21 |
| 3702 | TaiwanStockPrice | MISMATCH | 5041 | 2005-11-09 | 2026-05-22 | 5040 | 2005-11-09 | 2026-05-21 |
| 3706 | TaiwanStockPrice | MISMATCH | 3097 | 2013-09-12 | 2026-05-22 | 3096 | 2013-09-12 | 2026-05-21 |
| 3711 | TaiwanStockPrice | MISMATCH | 1963 | 2018-04-30 | 2026-05-22 | 1962 | 2018-04-30 | 2026-05-21 |
| 3715 | TaiwanStockPrice | MISMATCH | 904 | 2022-08-25 | 2026-05-22 | 903 | 2022-08-25 | 2026-05-21 |
| 4739 | TaiwanStockPrice | MISMATCH | 3778 | 2010-12-13 | 2026-05-22 | 3777 | 2010-12-13 | 2026-05-21 |
| 4904 | TaiwanStockPrice | MISMATCH | 5601 | 2003-08-01 | 2026-05-22 | 5600 | 2003-08-01 | 2026-05-21 |
| 4916 | TaiwanStockPrice | MISMATCH | 4012 | 2009-12-25 | 2026-05-22 | 4011 | 2009-12-25 | 2026-05-21 |
| 4958 | TaiwanStockPrice | MISMATCH | 3521 | 2011-12-26 | 2026-05-22 | 3520 | 2011-12-26 | 2026-05-21 |
| 4967 | TaiwanStockPrice | MISMATCH | 3800 | 2010-10-13 | 2026-05-22 | 3799 | 2010-10-13 | 2026-05-21 |
| 5269 | TaiwanStockPrice | MISMATCH | 3506 | 2012-01-12 | 2026-05-22 | 3505 | 2012-01-12 | 2026-05-21 |
| 5469 | TaiwanStockPrice | MISMATCH | 5606 | 2003-08-01 | 2026-05-22 | 5605 | 2003-08-01 | 2026-05-21 |
| 6139 | TaiwanStockPrice | MISMATCH | 5613 | 2003-08-01 | 2026-05-22 | 5612 | 2003-08-01 | 2026-05-21 |
| 6191 | TaiwanStockPrice | MISMATCH | 5537 | 2003-08-01 | 2026-05-22 | 5536 | 2003-08-01 | 2026-05-21 |
| 6213 | TaiwanStockPrice | MISMATCH | 5542 | 2003-08-01 | 2026-05-22 | 5541 | 2003-08-01 | 2026-05-21 |
| 6215 | TaiwanStockPrice | MISMATCH | 5542 | 2003-08-01 | 2026-05-22 | 5541 | 2003-08-01 | 2026-05-21 |
| 6239 | TaiwanStockPrice | MISMATCH | 5613 | 2003-08-01 | 2026-05-22 | 5612 | 2003-08-01 | 2026-05-21 |
| 6282 | TaiwanStockPrice | MISMATCH | 5587 | 2003-09-08 | 2026-05-22 | 5586 | 2003-09-08 | 2026-05-21 |
| 6442 | TaiwanStockPrice | MISMATCH | 2974 | 2014-03-17 | 2026-05-22 | 2973 | 2014-03-17 | 2026-05-21 |
| 6446 | TaiwanStockPrice | MISMATCH | 2976 | 2014-03-11 | 2026-05-22 | 2975 | 2014-03-11 | 2026-05-21 |
| 6515 | TaiwanStockPrice | MISMATCH | 1580 | 2019-11-19 | 2026-05-22 | 1579 | 2019-11-19 | 2026-05-21 |
| 6531 | TaiwanStockPrice | MISMATCH | 2661 | 2015-06-12 | 2026-05-22 | 2660 | 2015-06-12 | 2026-05-21 |
| 6669 | TaiwanStockPrice | MISMATCH | 2072 | 2017-11-13 | 2026-05-22 | 2071 | 2017-11-13 | 2026-05-21 |
| 6691 | TaiwanStockPrice | MISMATCH | 1892 | 2018-08-08 | 2026-05-22 | 1891 | 2018-08-08 | 2026-05-21 |
| 6770 | TaiwanStockPrice | MISMATCH | 1321 | 2020-12-09 | 2026-05-22 | 1320 | 2020-12-09 | 2026-05-21 |
| 6781 | TaiwanStockPrice | MISMATCH | 1257 | 2021-03-22 | 2026-05-22 | 1256 | 2021-03-22 | 2026-05-21 |
| 6805 | TaiwanStockPrice | MISMATCH | 1334 | 2020-11-20 | 2026-05-22 | 1333 | 2020-11-20 | 2026-05-21 |
| 8021 | TaiwanStockPrice | MISMATCH | 5187 | 2004-12-21 | 2026-05-22 | 5186 | 2004-12-21 | 2026-05-21 |
| 8028 | TaiwanStockPrice | MISMATCH | 3085 | 2006-12-13 | 2026-05-22 | 3084 | 2006-12-13 | 2026-05-21 |
| 8046 | TaiwanStockPrice | MISMATCH | 4948 | 2006-04-07 | 2026-05-22 | 4947 | 2006-04-07 | 2026-05-21 |
| 8112 | TaiwanStockPrice | MISMATCH | 5415 | 2004-02-09 | 2026-05-22 | 5414 | 2004-02-09 | 2026-05-21 |
| 8150 | TaiwanStockPrice | MISMATCH | 3191 | 2013-04-19 | 2026-05-22 | 3190 | 2013-04-19 | 2026-05-21 |
| 8210 | TaiwanStockPrice | MISMATCH | 5110 | 2005-04-28 | 2026-05-22 | 5109 | 2005-04-28 | 2026-05-21 |
| 8271 | TaiwanStockPrice | MISMATCH | 4748 | 2006-12-13 | 2026-05-22 | 4747 | 2006-12-13 | 2026-05-21 |
| 8996 | TaiwanStockPrice | MISMATCH | 4819 | 2006-06-30 | 2026-05-22 | 4818 | 2006-06-30 | 2026-05-21 |

## Targeted Backfill Commands

```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1503 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1504 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1513 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1514 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1519 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1560 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1590 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1795 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2049 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2059 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2301 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2303 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2308 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2313 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2316 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2317 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2327 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2329 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2337 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2344 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2345 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2354 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2356 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2357 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2360 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2363 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2368 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2374 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2376 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2379 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2382 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2383 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2395 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2402 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2404 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2408 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2412 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2421 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2441 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2449 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2451 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2454 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2455 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2467 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2481 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2486 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2492 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3006 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3013 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3017 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3019 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3030 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3036 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3037 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3044 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3167 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3189 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3231 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3305 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3406 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3443 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3481 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3515 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3533 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3563 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3653 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3661 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3665 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3702 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3706 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3711 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3715 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4739 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4904 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4916 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4958 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4967 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5269 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5469 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6139 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6191 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6213 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6215 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6239 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6282 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6442 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6446 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6515 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6531 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6669 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6691 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6770 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6781 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6805 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8021 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8028 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8046 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8112 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8150 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8210 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8271 --dataset TaiwanStockPrice --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8996 --dataset TaiwanStockPrice --strict-source-history
```
