# Strict source availability audit

- **time**: 2026-05-22 20:12:46
- **constitution**: 系統架構大憲章_v6.0.0.md §14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ
- **tool**: audit_source_availability v0.6
- **start_date**: 1990-01-01
- **drift_tolerance**: 3 day(s) (§6.8.8-C; 0 = strict)
- **scope**: stocks=2767 (universe=full), datasets=9
- **special_full_market_reason**: §6.8.8-D / §14.7-AQ landing initial full-market baseline 2026-05-22 (v0.6 with §14.7-AS hotfix + progress heartbeat) (§6.8.7 第 (4) 條 / §6.8.8-D)
- **verdict**: **FAILED**
- **summary**: checked=24903, source_empty_ok=2872, time_drift_ok=11206, mismatch=1887, api_errors=724

- **fred_summary**: checked=4, time_drift_ok=0, mismatch=0, api_errors=0

## Mismatches

| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max | error |
|---|---|---|---:|---|---|---:|---|---|---|
| 006201 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15661 | 2011-02-08 | 2026-05-22 | 15656 | 2011-02-08 | 2026-05-21 |  |
| 00625K | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5683 | 2016-02-05 | 2026-05-21 | 5678 | 2016-02-05 | 2026-05-15 |  |
| 00636K | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10186 | 2016-02-05 | 2026-05-21 | 10181 | 2016-02-05 | 2026-05-20 |  |
| 00657K | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10160 | 2016-02-05 | 2026-05-21 | 10155 | 2016-02-05 | 2026-05-20 |  |
| 00668K | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10096 | 2016-02-05 | 2026-05-21 | 10091 | 2016-02-05 | 2026-05-20 |  |
| 00679B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11135 | 2017-01-17 | 2026-05-22 | 11130 | 2017-01-17 | 2026-05-21 |  |
| 00687B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10908 | 2017-04-13 | 2026-05-22 | 10903 | 2017-04-13 | 2026-05-21 |  |
| 00687C | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1590 | 2025-01-10 | 2026-05-22 | 1585 | 2025-01-10 | 2026-05-21 |  |
| 00694B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10770 | 2017-06-08 | 2026-05-22 | 10765 | 2017-06-08 | 2026-05-21 |  |
| 00695B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10770 | 2017-06-08 | 2026-05-22 | 10765 | 2017-06-08 | 2026-05-21 |  |
| 00696B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10775 | 2017-06-08 | 2026-05-22 | 10770 | 2017-06-08 | 2026-05-21 |  |
| 00697B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10717 | 2017-06-23 | 2026-05-22 | 10712 | 2017-06-23 | 2026-05-21 |  |
| 00719B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10070 | 2018-02-01 | 2026-05-22 | 10065 | 2018-02-01 | 2026-05-21 |  |
| 00720B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10070 | 2018-02-01 | 2026-05-22 | 10065 | 2018-02-01 | 2026-05-21 |  |
| 00722B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10239 | 2017-12-15 | 2026-05-22 | 10234 | 2017-12-15 | 2026-05-21 |  |
| 00723B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10220 | 2017-12-15 | 2026-05-22 | 10215 | 2017-12-15 | 2026-05-21 |  |
| 00724B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10234 | 2017-12-15 | 2026-05-22 | 10229 | 2017-12-15 | 2026-05-21 |  |
| 00725B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10040 | 2018-02-07 | 2026-05-22 | 10035 | 2018-02-07 | 2026-05-21 |  |
| 00726B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10040 | 2018-02-07 | 2026-05-22 | 10035 | 2018-02-07 | 2026-05-21 |  |
| 00727B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10005 | 2018-02-07 | 2026-05-22 | 10000 | 2018-02-07 | 2026-05-21 |  |
| 00734B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9685 | 2018-05-24 | 2026-05-22 | 9680 | 2018-05-24 | 2026-05-21 |  |
| 00740B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9670 | 2018-06-08 | 2026-05-22 | 9665 | 2018-06-08 | 2026-05-21 |  |
| 00741B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9670 | 2018-06-08 | 2026-05-22 | 9665 | 2018-06-08 | 2026-05-21 |  |
| 00746B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9430 | 2018-08-15 | 2026-05-22 | 9425 | 2018-08-15 | 2026-05-21 |  |
| 00749B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9275 | 2018-09-20 | 2026-05-22 | 9270 | 2018-09-20 | 2026-05-21 |  |
| 00750B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9280 | 2018-09-20 | 2026-05-22 | 9275 | 2018-09-20 | 2026-05-21 |  |
| 00751B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9265 | 2018-10-03 | 2026-05-22 | 9260 | 2018-10-03 | 2026-05-21 |  |
| 00754B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9205 | 2018-10-22 | 2026-05-22 | 9200 | 2018-10-22 | 2026-05-21 |  |
| 00755B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9205 | 2018-10-22 | 2026-05-22 | 9200 | 2018-10-22 | 2026-05-21 |  |
| 00756B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9205 | 2018-10-22 | 2026-05-22 | 9200 | 2018-10-22 | 2026-05-21 |  |
| 00758B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9080 | 2018-11-14 | 2026-05-22 | 9075 | 2018-11-14 | 2026-05-21 |  |
| 00759B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9065 | 2018-11-14 | 2026-05-22 | 9060 | 2018-11-14 | 2026-05-21 |  |
| 00760B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9085 | 2018-11-14 | 2026-05-22 | 9080 | 2018-11-14 | 2026-05-21 |  |
| 00761B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9060 | 2018-11-30 | 2026-05-22 | 9055 | 2018-11-30 | 2026-05-21 |  |
| 00764B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8975 | 2018-12-24 | 2026-05-22 | 8970 | 2018-12-24 | 2026-05-21 |  |
| 00768B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8855 | 2019-01-28 | 2026-05-22 | 8850 | 2019-01-28 | 2026-05-21 |  |
| 00772B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8855 | 2019-01-29 | 2026-05-22 | 8850 | 2019-01-29 | 2026-05-21 |  |
| 00773B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8855 | 2019-01-29 | 2026-05-22 | 8850 | 2019-01-29 | 2026-05-21 |  |
| 00777B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8820 | 2019-02-14 | 2026-05-22 | 8815 | 2019-02-14 | 2026-05-21 |  |
| 00778B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8725 | 2019-02-14 | 2026-05-22 | 8720 | 2019-02-14 | 2026-05-21 |  |
| 00779B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8810 | 2019-02-14 | 2026-05-22 | 8805 | 2019-02-14 | 2026-05-21 |  |
| 00780B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8745 | 2019-03-07 | 2026-05-22 | 8740 | 2019-03-07 | 2026-05-21 |  |
| 00781B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8755 | 2019-03-07 | 2026-05-22 | 8750 | 2019-03-07 | 2026-05-21 |  |
| 00782B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8755 | 2019-03-07 | 2026-05-22 | 8750 | 2019-03-07 | 2026-05-21 |  |
| 00785B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8670 | 2019-03-27 | 2026-05-22 | 8665 | 2019-03-27 | 2026-05-21 |  |
| 00786B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8560 | 2019-04-02 | 2026-05-22 | 8555 | 2019-04-02 | 2026-05-21 |  |
| 00787B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8600 | 2019-04-02 | 2026-05-22 | 8595 | 2019-04-02 | 2026-05-21 |  |
| 00788B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8575 | 2019-04-02 | 2026-05-22 | 8570 | 2019-04-02 | 2026-05-21 |  |
| 00789B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8705 | 2019-03-20 | 2026-05-22 | 8700 | 2019-03-20 | 2026-05-21 |  |
| 00792B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8670 | 2019-04-03 | 2026-05-22 | 8665 | 2019-04-03 | 2026-05-21 |  |
| 00793B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8660 | 2019-04-03 | 2026-05-22 | 8655 | 2019-04-03 | 2026-05-21 |  |
| 00795B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8650 | 2019-04-11 | 2026-05-22 | 8645 | 2019-04-11 | 2026-05-21 |  |
| 00799B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8570 | 2019-05-03 | 2026-05-22 | 8565 | 2019-05-03 | 2026-05-21 |  |
| 00836B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8510 | 2019-05-22 | 2026-05-22 | 8505 | 2019-05-22 | 2026-05-21 |  |
| 00842B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8390 | 2019-06-14 | 2026-05-22 | 8385 | 2019-06-14 | 2026-05-21 |  |
| 00844B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8025 | 2019-07-18 | 2026-05-22 | 8020 | 2019-07-18 | 2026-05-21 |  |
| 00845B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8310 | 2019-07-16 | 2026-05-22 | 8305 | 2019-07-16 | 2026-05-21 |  |
| 00846B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8315 | 2019-07-16 | 2026-05-22 | 8310 | 2019-07-16 | 2026-05-21 |  |
| 00847B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8270 | 2019-07-30 | 2026-05-22 | 8265 | 2019-07-30 | 2026-05-21 |  |
| 00848B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8270 | 2019-07-30 | 2026-05-22 | 8265 | 2019-07-30 | 2026-05-21 |  |
| 00849B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8270 | 2019-07-30 | 2026-05-22 | 8265 | 2019-07-30 | 2026-05-21 |  |
| 00853B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7145 | 2019-09-04 | 2026-05-22 | 7140 | 2019-09-04 | 2026-05-21 |  |
| 00856B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8045 | 2019-10-01 | 2026-05-22 | 8040 | 2019-10-01 | 2026-05-21 |  |
| 00857B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8060 | 2019-10-01 | 2026-05-22 | 8055 | 2019-10-01 | 2026-05-21 |  |
| 00858 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8060 | 2019-10-01 | 2026-05-22 | 8055 | 2019-10-01 | 2026-05-21 |  |
| 00859B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8000 | 2019-10-18 | 2026-05-22 | 7995 | 2019-10-18 | 2026-05-21 |  |
| 00860B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8000 | 2019-10-18 | 2026-05-22 | 7995 | 2019-10-18 | 2026-05-21 |  |
| 00862B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8005 | 2019-10-18 | 2026-05-22 | 8000 | 2019-10-18 | 2026-05-21 |  |
| 00863B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8005 | 2019-10-18 | 2026-05-22 | 8000 | 2019-10-18 | 2026-05-21 |  |
| 00864B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8005 | 2019-10-18 | 2026-05-22 | 8000 | 2019-10-18 | 2026-05-21 |  |
| 00867B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7805 | 2019-11-15 | 2026-05-22 | 7800 | 2019-11-15 | 2026-05-21 |  |
| 00870B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7265 | 2020-01-09 | 2026-05-22 | 7260 | 2020-01-09 | 2026-05-21 |  |
| 00877 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7085 | 2020-07-22 | 2026-05-22 | 7080 | 2020-07-22 | 2026-05-21 |  |
| 00884B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6400 | 2021-02-04 | 2026-05-22 | 6395 | 2021-02-04 | 2026-05-21 |  |
| 00886 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6190 | 2021-03-31 | 2026-05-22 | 6185 | 2021-03-31 | 2026-05-21 |  |
| 00887 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6240 | 2021-03-31 | 2026-05-22 | 6235 | 2021-03-31 | 2026-05-21 |  |
| 00888 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6245 | 2021-03-31 | 2026-05-22 | 6240 | 2021-03-31 | 2026-05-21 |  |
| 00890B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6115 | 2021-05-10 | 2026-05-22 | 6110 | 2021-05-10 | 2026-05-21 |  |
| 00928 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3635 | 2023-05-24 | 2026-05-22 | 3630 | 2023-05-24 | 2026-05-21 |  |
| 00931B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3385 | 2023-08-07 | 2026-05-22 | 3380 | 2023-08-07 | 2026-05-21 |  |
| 00933B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3135 | 2023-10-19 | 2026-05-22 | 3130 | 2023-10-19 | 2026-05-21 |  |
| 00937B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2970 | 2023-12-05 | 2026-05-22 | 2965 | 2023-12-05 | 2026-05-21 |  |
| 00942B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2540 | 2024-04-18 | 2026-05-22 | 2535 | 2024-04-18 | 2026-05-21 |  |
| 00948B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2355 | 2024-06-12 | 2026-05-22 | 2350 | 2024-06-12 | 2026-05-21 |  |
| 00950B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2260 | 2024-07-09 | 2026-05-22 | 2255 | 2024-07-09 | 2026-05-21 |  |
| 00955 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2120 | 2024-08-20 | 2026-05-22 | 2115 | 2024-08-20 | 2026-05-21 |  |
| 00957B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2045 | 2024-09-10 | 2026-05-22 | 2040 | 2024-09-10 | 2026-05-21 |  |
| 00958B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2020 | 2024-09-18 | 2026-05-22 | 2015 | 2024-09-18 | 2026-05-21 |  |
| 00959B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2090 | 2024-08-28 | 2026-05-22 | 2085 | 2024-08-28 | 2026-05-21 |  |
| 00966B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1725 | 2024-12-16 | 2026-05-22 | 1720 | 2024-12-16 | 2026-05-21 |  |
| 00967B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1750 | 2024-12-09 | 2026-05-22 | 1745 | 2024-12-09 | 2026-05-21 |  |
| 00968B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1750 | 2024-12-09 | 2026-05-22 | 1745 | 2024-12-09 | 2026-05-21 |  |
| 00969B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1750 | 2024-12-09 | 2026-05-22 | 1745 | 2024-12-09 | 2026-05-21 |  |
| 00970B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1720 | 2024-12-17 | 2026-05-22 | 1715 | 2024-12-17 | 2026-05-21 |  |
| 009806 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1200 | 2025-05-28 | 2026-05-22 | 1195 | 2025-05-28 | 2026-05-21 |  |
| 009807 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1200 | 2025-05-28 | 2026-05-22 | 1195 | 2025-05-28 | 2026-05-21 |  |
| 00980B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1595 | 2025-01-22 | 2026-05-22 | 1590 | 2025-01-22 | 2026-05-21 |  |
| 00980D | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 965 | 2025-08-04 | 2026-05-22 | 960 | 2025-08-04 | 2026-05-21 |  |
| 00980T | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 920 | 2025-08-15 | 2026-05-22 | 915 | 2025-08-15 | 2026-05-21 |  |
| 009814 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 490 | 2025-12-19 | 2026-05-22 | 485 | 2025-12-19 | 2026-05-21 |  |
| 009815 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 370 | 2026-01-26 | 2026-05-22 | 365 | 2026-01-26 | 2026-05-21 |  |
| 00981B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1470 | 2025-03-10 | 2026-05-22 | 1465 | 2025-03-10 | 2026-05-21 |  |
| 00981D | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 810 | 2025-09-16 | 2026-05-22 | 805 | 2025-09-16 | 2026-05-21 |  |
| 00982B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1300 | 2025-04-28 | 2026-05-22 | 1295 | 2025-04-28 | 2026-05-21 |  |
| 00983B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1255 | 2025-05-13 | 2026-05-22 | 1250 | 2025-05-13 | 2026-05-21 |  |
| 00984B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1255 | 2025-05-13 | 2026-05-22 | 1250 | 2025-05-13 | 2026-05-21 |  |
| 00985D | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 185 | 2026-03-30 | 2026-05-22 | 180 | 2026-03-30 | 2026-05-21 |  |
| 00986B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 610 | 2025-11-12 | 2026-05-22 | 605 | 2025-11-12 | 2026-05-21 |  |
| 00986D | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 135 | 2026-04-15 | 2026-05-22 | 130 | 2026-04-15 | 2026-05-21 |  |
| 00987B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 380 | 2026-01-22 | 2026-05-22 | 375 | 2026-01-22 | 2026-05-21 |  |
| 00988B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 200 | 2026-03-25 | 2026-05-22 | 195 | 2026-03-25 | 2026-05-21 |  |
| 00989B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 180 | 2026-03-31 | 2026-05-22 | 175 | 2026-03-31 | 2026-05-21 |  |
| 00998A | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 135 | 2026-04-15 | 2026-05-22 | 130 | 2026-04-15 | 2026-05-21 |  |
| 020001 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4970 | 2019-04-30 | 2026-05-22 | 4965 | 2019-04-30 | 2026-05-18 |  |
| 020025 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5515 | 2021-07-22 | 2026-05-22 | 5510 | 2021-07-22 | 2026-05-21 |  |
| 020027 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5760 | 2021-07-19 | 2026-05-22 | 5755 | 2021-07-19 | 2026-05-21 |  |
| 020033 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4565 | 2022-05-25 | 2026-05-22 | 4560 | 2022-05-25 | 2026-05-19 |  |
| 020040 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2245 | 2023-12-18 | 2026-05-22 | 2240 | 2023-12-18 | 2026-05-21 |  |
| 020041 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1175 | 2023-12-25 | 2026-05-22 | 1170 | 2023-12-25 | 2026-05-21 |  |
| 1240 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7300 | 2018-08-15 | 2026-05-22 | 7295 | 2018-08-15 | 2026-05-21 |  |
| 1259 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6175 | 2011-01-26 | 2026-05-22 | 6170 | 2011-01-26 | 2026-05-21 |  |
| 1264 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9732 | 2014-05-21 | 2026-05-22 | 9727 | 2014-05-21 | 2026-05-21 |  |
| 1268 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7143 | 2017-03-21 | 2026-05-22 | 7138 | 2017-03-21 | 2026-05-21 |  |
| 1294 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2026 | 2023-04-13 | 2026-05-22 | 2021 | 2023-04-13 | 2026-05-21 |  |
| 1295 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1406 | 2024-03-19 | 2026-05-22 | 1401 | 2024-03-19 | 2026-05-21 |  |
| 1336 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11780 | 2007-06-27 | 2026-05-22 | 11775 | 2007-06-27 | 2026-05-21 |  |
| 1341 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5150 | 2018-12-28 | 2026-05-22 | 5145 | 2018-12-28 | 2026-05-21 |  |
| 1402 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 1413 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12319 | 2012-05-02 | 2026-05-22 | 12314 | 2012-05-02 | 2026-05-21 |  |
| 1416 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14444 | 2012-05-02 | 2026-05-22 | 14439 | 2012-05-02 | 2026-05-21 |  |
| 1423 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13005 | 2012-05-02 | 2026-05-22 | 13000 | 2012-05-02 | 2026-05-21 |  |
| 1434 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15163 | 2012-05-02 | 2026-05-22 | 15158 | 2012-05-02 | 2026-05-21 |  |
| 1439 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13542 | 2012-05-02 | 2026-05-22 | 13537 | 2012-05-02 | 2026-05-21 |  |
| 1444 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15153 | 2012-05-02 | 2026-05-22 | 15148 | 2012-05-02 | 2026-05-21 |  |
| 1445 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12921 | 2012-05-02 | 2026-05-22 | 12916 | 2012-05-02 | 2026-05-21 |  |
| 1446 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13954 | 2012-05-02 | 2026-05-22 | 13949 | 2012-05-02 | 2026-05-21 |  |
| 1447 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15151 | 2012-05-02 | 2026-05-22 | 15146 | 2012-05-02 | 2026-05-21 |  |
| 1449 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10581 | 2013-07-03 | 2026-05-22 | 10576 | 2013-07-03 | 2026-05-21 |  |
| 1454 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12746 | 2012-05-02 | 2026-05-22 | 12741 | 2012-05-02 | 2026-05-21 |  |
| 1455 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15155 | 2012-05-02 | 2026-05-22 | 15150 | 2012-05-02 | 2026-05-21 |  |
| 1457 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15076 | 2012-05-02 | 2026-05-22 | 15071 | 2012-05-02 | 2026-05-21 |  |
| 1459 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14119 | 2012-05-02 | 2026-05-22 | 14114 | 2012-05-02 | 2026-05-21 |  |
| 1463 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15015 | 2012-05-02 | 2026-05-22 | 15010 | 2012-05-02 | 2026-05-21 |  |
| 1464 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14854 | 2012-05-02 | 2026-05-22 | 14849 | 2012-05-02 | 2026-05-21 |  |
| 1465 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11601 | 2012-05-02 | 2026-05-22 | 11596 | 2012-05-02 | 2026-05-21 |  |
| 1466 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12978 | 2012-05-02 | 2026-05-22 | 12973 | 2012-05-02 | 2026-05-21 |  |
| 1470 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7566 | 2012-05-02 | 2026-05-22 | 7561 | 2012-05-02 | 2026-05-21 |  |
| 1473 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14960 | 2012-05-02 | 2026-05-22 | 14955 | 2012-05-02 | 2026-05-21 |  |
| 1477 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 1516 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9705 | 2013-07-03 | 2026-05-22 | 9700 | 2013-07-03 | 2026-05-21 |  |
| 1521 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14476 | 2012-05-02 | 2026-05-22 | 14471 | 2012-05-02 | 2026-05-21 |  |
| 1529 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8427 | 2013-07-03 | 2026-05-22 | 8422 | 2013-07-03 | 2026-05-21 |  |
| 1530 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12711 | 2012-05-02 | 2026-05-22 | 12706 | 2012-05-02 | 2026-05-21 |  |
| 1531 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14766 | 2012-05-02 | 2026-05-22 | 14761 | 2012-05-02 | 2026-05-21 |  |
| 1532 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15161 | 2012-05-02 | 2026-05-22 | 15156 | 2012-05-02 | 2026-05-21 |  |
| 1535 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13845 | 2012-05-02 | 2026-05-22 | 13840 | 2012-05-02 | 2026-05-21 |  |
| 1536 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15160 | 2012-05-02 | 2026-05-22 | 15155 | 2012-05-02 | 2026-05-21 |  |
| 1539 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13834 | 2012-05-02 | 2026-05-22 | 13829 | 2012-05-02 | 2026-05-21 |  |
| 1541 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12869 | 2012-05-02 | 2026-05-22 | 12864 | 2012-05-02 | 2026-05-21 |  |
| 1560 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15158 | 2012-05-02 | 2026-05-22 | 15153 | 2012-05-02 | 2026-05-21 |  |
| 1563 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7212 | 2007-07-06 | 2026-05-22 | 7207 | 2007-07-06 | 2026-05-21 |  |
| 1565 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20289 | 2005-01-03 | 2026-05-22 | 20284 | 2005-01-03 | 2026-05-21 |  |
| 1568 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13532 | 2007-07-11 | 2026-05-22 | 13527 | 2007-07-11 | 2026-05-21 |  |
| 1569 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16362 | 2005-08-10 | 2026-05-22 | 16357 | 2005-08-10 | 2026-05-21 |  |
| 1570 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9787 | 2005-06-30 | 2026-05-22 | 9782 | 2005-06-30 | 2026-05-21 |  |
| 1580 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17023 | 2007-11-26 | 2026-05-22 | 17018 | 2007-11-26 | 2026-05-21 |  |
| 1583 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11385 | 2007-06-25 | 2026-05-22 | 11380 | 2007-06-25 | 2026-05-21 |  |
| 1584 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13152 | 2007-06-21 | 2026-05-22 | 13147 | 2007-06-21 | 2026-05-21 |  |
| 1586 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13432 | 2009-12-23 | 2026-05-22 | 13427 | 2009-12-23 | 2026-05-21 |  |
| 1587 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8570 | 2014-05-29 | 2026-05-22 | 8565 | 2014-05-29 | 2026-05-21 |  |
| 1591 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9242 | 2010-04-09 | 2026-05-22 | 9237 | 2010-04-09 | 2026-05-21 |  |
| 1593 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9911 | 2012-06-13 | 2026-05-22 | 9906 | 2012-06-13 | 2026-05-21 |  |
| 1595 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10984 | 2011-02-16 | 2026-05-22 | 10979 | 2011-02-16 | 2026-05-21 |  |
| 1597 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13554 | 2011-06-17 | 2026-05-22 | 13549 | 2011-06-17 | 2026-05-21 |  |
| 1598 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11414 | 2012-03-01 | 2026-05-22 | 11409 | 2012-03-01 | 2026-05-21 |  |
| 1599 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11802 | 2012-02-03 | 2026-05-22 | 11797 | 2012-02-03 | 2026-05-21 |  |
| 1603 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13371 | 2012-05-02 | 2026-05-22 | 13366 | 2012-05-02 | 2026-05-21 |  |
| 1604 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15118 | 2012-05-02 | 2026-05-22 | 15113 | 2012-05-02 | 2026-05-21 |  |
| 1605 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 1608 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15107 | 2012-05-02 | 2026-05-22 | 15102 | 2012-05-02 | 2026-05-21 |  |
| 1609 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15107 | 2012-05-02 | 2026-05-22 | 15102 | 2012-05-02 | 2026-05-21 |  |
| 1611 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15033 | 2012-05-02 | 2026-05-22 | 15028 | 2012-05-02 | 2026-05-21 |  |
| 1612 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14759 | 2012-05-02 | 2026-05-22 | 14754 | 2012-05-02 | 2026-05-21 |  |
| 1614 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14072 | 2012-05-02 | 2026-05-22 | 14067 | 2012-05-02 | 2026-05-21 |  |
| 1615 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12045 | 2012-05-02 | 2026-05-22 | 12040 | 2012-05-02 | 2026-05-21 |  |
| 1616 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12756 | 2012-05-02 | 2026-05-22 | 12751 | 2012-05-02 | 2026-05-21 |  |
| 1617 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11780 | 2012-05-02 | 2026-05-22 | 11775 | 2012-05-02 | 2026-05-21 |  |
| 1618 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14360 | 2012-05-02 | 2026-05-22 | 14355 | 2012-05-02 | 2026-05-21 |  |
| 1626 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14072 | 2013-01-21 | 2026-05-22 | 14067 | 2013-01-21 | 2026-05-21 |  |
| 1702 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-22 | 15157 | 2012-05-02 | 2026-05-21 |  |
| 1707 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 1708 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15121 | 2012-05-02 | 2026-05-22 | 15116 | 2012-05-02 | 2026-05-21 |  |
| 1709 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14995 | 2012-05-02 | 2026-05-22 | 14990 | 2012-05-02 | 2026-05-21 |  |
| 1710 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 1711 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15155 | 2012-05-02 | 2026-05-22 | 15150 | 2012-05-02 | 2026-05-21 |  |
| 1712 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15128 | 2012-05-02 | 2026-05-22 | 15123 | 2012-05-02 | 2026-05-21 |  |
| 1713 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11913 | 2012-05-02 | 2026-05-22 | 11908 | 2012-05-02 | 2026-05-21 |  |
| 1714 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15159 | 2012-05-02 | 2026-05-22 | 15154 | 2012-05-02 | 2026-05-21 |  |
| 1717 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15117 | 2012-05-02 | 2026-05-22 | 15112 | 2012-05-02 | 2026-05-21 |  |
| 1718 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 1720 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15096 | 2012-05-02 | 2026-05-22 | 15091 | 2012-05-02 | 2026-05-21 |  |
| 1721 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13716 | 2012-05-03 | 2026-05-22 | 13711 | 2012-05-03 | 2026-05-21 |  |
| 1722 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 1723 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 1725 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11241 | 2012-05-02 | 2026-05-22 | 11236 | 2012-05-02 | 2026-05-21 |  |
| 1726 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14371 | 2012-05-02 | 2026-05-22 | 14366 | 2012-05-02 | 2026-05-21 |  |
| 1727 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12585 | 2012-05-02 | 2026-05-22 | 12580 | 2012-05-02 | 2026-05-21 |  |
| 1730 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14011 | 2012-05-03 | 2026-05-22 | 14006 | 2012-05-03 | 2026-05-21 |  |
| 1731 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13704 | 2012-05-02 | 2026-05-22 | 13699 | 2012-05-02 | 2026-05-21 |  |
| 1732 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11860 | 2012-05-02 | 2026-05-22 | 11855 | 2012-05-02 | 2026-05-21 |  |
| 1733 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15151 | 2012-05-02 | 2026-05-22 | 15146 | 2012-05-02 | 2026-05-21 |  |
| 1734 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14976 | 2012-05-02 | 2026-05-22 | 14971 | 2012-05-02 | 2026-05-21 |  |
| 1735 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9679 | 2012-05-02 | 2026-05-22 | 9674 | 2012-05-02 | 2026-05-21 |  |
| 1736 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15161 | 2012-05-02 | 2026-05-22 | 15156 | 2012-05-02 | 2026-05-21 |  |
| 1737 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15094 | 2012-05-02 | 2026-05-22 | 15089 | 2012-05-02 | 2026-05-21 |  |
| 1752 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15331 | 2008-05-26 | 2026-05-22 | 15326 | 2008-05-26 | 2026-05-21 |  |
| 1760 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11563 | 2007-07-25 | 2026-05-22 | 11558 | 2007-07-25 | 2026-05-21 |  |
| 1762 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15154 | 2009-12-10 | 2026-05-22 | 15149 | 2009-12-10 | 2026-05-21 |  |
| 1773 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15141 | 2008-03-11 | 2026-05-22 | 15136 | 2008-03-11 | 2026-05-21 |  |
| 1776 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8268 | 2007-12-21 | 2026-05-22 | 8263 | 2007-12-21 | 2026-05-21 |  |
| 1781 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11477 | 2005-01-05 | 2026-05-22 | 11472 | 2005-01-05 | 2026-05-21 |  |
| 1783 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13348 | 2013-04-16 | 2026-05-22 | 13343 | 2013-04-16 | 2026-05-21 |  |
| 1784 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15948 | 2007-07-05 | 2026-05-22 | 15943 | 2007-07-05 | 2026-05-21 |  |
| 1785 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19600 | 2005-01-31 | 2026-05-22 | 19595 | 2005-01-31 | 2026-05-21 |  |
| 1786 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13720 | 2011-01-19 | 2026-05-22 | 13715 | 2011-01-19 | 2026-05-21 |  |
| 1788 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12148 | 2007-12-11 | 2026-05-22 | 12143 | 2007-12-11 | 2026-05-21 |  |
| 1789 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15540 | 2010-10-29 | 2026-05-22 | 15535 | 2010-10-29 | 2026-05-21 |  |
| 1795 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15991 | 2007-06-25 | 2026-05-22 | 15986 | 2007-06-25 | 2026-05-21 |  |
| 1796 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6261 | 2016-09-02 | 2026-05-22 | 6256 | 2016-09-02 | 2026-05-21 |  |
| 1799 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10882 | 2007-07-04 | 2026-05-22 | 10877 | 2007-07-04 | 2026-05-21 |  |
| 1802 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 1805 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10854 | 2012-05-09 | 2026-05-22 | 10849 | 2012-05-09 | 2026-05-21 |  |
| 1806 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14931 | 2012-05-02 | 2026-05-22 | 14926 | 2012-05-02 | 2026-05-21 |  |
| 1808 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15068 | 2012-05-02 | 2026-05-22 | 15063 | 2012-05-02 | 2026-05-21 |  |
| 1809 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14566 | 2012-05-02 | 2026-05-22 | 14561 | 2012-05-02 | 2026-05-21 |  |
| 1810 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14703 | 2012-05-02 | 2026-05-22 | 14698 | 2012-05-02 | 2026-05-21 |  |
| 1813 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8593 | 2008-06-24 | 2026-05-22 | 8588 | 2008-06-24 | 2026-05-20 |  |
| 1815 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19035 | 2006-01-23 | 2026-05-22 | 19030 | 2006-01-23 | 2026-05-21 |  |
| 1817 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13576 | 2011-08-24 | 2026-05-22 | 13571 | 2011-08-24 | 2026-05-21 |  |
| 1903 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13937 | 2012-05-02 | 2026-05-22 | 13932 | 2012-05-02 | 2026-05-21 |  |
| 1904 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15161 | 2012-05-02 | 2026-05-22 | 15156 | 2012-05-02 | 2026-05-21 |  |
| 1905 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15102 | 2012-05-02 | 2026-05-22 | 15097 | 2012-05-02 | 2026-05-21 |  |
| 1906 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14227 | 2012-05-02 | 2026-05-22 | 14222 | 2012-05-02 | 2026-05-21 |  |
| 1907 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15160 | 2012-05-02 | 2026-05-22 | 15155 | 2012-05-02 | 2026-05-21 |  |
| 1909 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-22 | 15157 | 2012-05-02 | 2026-05-21 |  |
| 2002 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2006 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15125 | 2012-05-02 | 2026-05-22 | 15120 | 2012-05-02 | 2026-05-21 |  |
| 2007 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12891 | 2012-05-02 | 2026-05-22 | 12886 | 2012-05-02 | 2026-05-21 |  |
| 2008 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11611 | 2012-06-06 | 2026-05-22 | 11606 | 2012-06-06 | 2026-05-21 |  |
| 2009 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14553 | 2012-05-02 | 2026-05-22 | 14548 | 2012-05-02 | 2026-05-21 |  |
| 2010 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14987 | 2012-05-02 | 2026-05-22 | 14982 | 2012-05-02 | 2026-05-21 |  |
| 2012 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13711 | 2012-05-02 | 2026-05-22 | 13706 | 2012-05-02 | 2026-05-21 |  |
| 2013 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14709 | 2012-05-02 | 2026-05-22 | 14704 | 2012-05-02 | 2026-05-21 |  |
| 2014 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15149 | 2012-05-02 | 2026-05-22 | 15144 | 2012-05-02 | 2026-05-21 |  |
| 2015 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2017 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14812 | 2012-05-02 | 2026-05-22 | 14807 | 2012-05-02 | 2026-05-21 |  |
| 2020 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14810 | 2012-05-02 | 2026-05-22 | 14805 | 2012-05-02 | 2026-05-21 |  |
| 2022 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14867 | 2012-05-02 | 2026-05-22 | 14862 | 2012-05-02 | 2026-05-21 |  |
| 2023 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 2025 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8018 | 2012-05-02 | 2026-05-22 | 8013 | 2012-05-02 | 2026-05-21 |  |
| 2027 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15150 | 2012-05-02 | 2026-05-22 | 15145 | 2012-05-02 | 2026-05-21 |  |
| 2028 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11952 | 2012-05-02 | 2026-05-22 | 11947 | 2012-05-02 | 2026-05-21 |  |
| 2029 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14926 | 2012-05-02 | 2026-05-22 | 14921 | 2012-05-02 | 2026-05-21 |  |
| 2030 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14980 | 2012-05-02 | 2026-05-22 | 14975 | 2012-05-02 | 2026-05-21 |  |
| 2031 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15136 | 2012-05-02 | 2026-05-22 | 15131 | 2012-05-02 | 2026-05-21 |  |
| 2032 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13475 | 2012-05-02 | 2026-05-22 | 13470 | 2012-05-02 | 2026-05-21 |  |
| 2033 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12366 | 2012-05-02 | 2026-05-22 | 12361 | 2012-05-02 | 2026-05-21 |  |
| 2034 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15154 | 2012-05-02 | 2026-05-22 | 15149 | 2012-05-02 | 2026-05-21 |  |
| 2035 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8625 | 2006-07-07 | 2026-05-22 | 8620 | 2006-07-07 | 2026-05-21 |  |
| 2038 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13734 | 2008-05-16 | 2026-05-22 | 13729 | 2008-05-16 | 2026-05-21 |  |
| 2049 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15479 | 2007-06-20 | 2026-05-22 | 15474 | 2007-06-20 | 2026-05-21 |  |
| 2059 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17283 | 2005-04-28 | 2026-05-22 | 17278 | 2005-04-28 | 2026-05-21 |  |
| 2061 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10014 | 2008-03-05 | 2026-05-22 | 10009 | 2008-03-05 | 2026-05-21 |  |
| 2062 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15104 | 2007-06-21 | 2026-05-22 | 15099 | 2007-06-21 | 2026-05-21 |  |
| 2063 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8726 | 2009-09-22 | 2026-05-22 | 8721 | 2009-09-22 | 2026-05-21 |  |
| 2064 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5481 | 2014-12-26 | 2026-05-22 | 5476 | 2014-12-26 | 2026-05-21 |  |
| 2065 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8878 | 2017-03-31 | 2026-05-22 | 8873 | 2017-03-31 | 2026-05-21 |  |
| 2066 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10211 | 2013-05-08 | 2026-05-22 | 10206 | 2013-05-08 | 2026-05-21 |  |
| 2067 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4282 | 2013-07-17 | 2026-05-22 | 4277 | 2013-07-17 | 2026-05-21 |  |
| 2069 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11436 | 2016-02-05 | 2026-05-22 | 11431 | 2016-02-05 | 2026-05-21 |  |
| 2070 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6167 | 2016-06-21 | 2026-05-22 | 6162 | 2016-06-21 | 2026-05-21 |  |
| 2072 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2067 | 2020-11-30 | 2026-05-22 | 2062 | 2020-11-30 | 2026-05-21 |  |
| 2101 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15128 | 2012-05-02 | 2026-05-22 | 15123 | 2012-05-02 | 2026-05-21 |  |
| 2102 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15133 | 2012-05-02 | 2026-05-22 | 15128 | 2012-05-02 | 2026-05-21 |  |
| 2103 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2104 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15154 | 2012-05-02 | 2026-05-22 | 15149 | 2012-05-02 | 2026-05-21 |  |
| 2105 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2106 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15155 | 2012-05-02 | 2026-05-22 | 15150 | 2012-05-02 | 2026-05-21 |  |
| 2107 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15037 | 2012-05-02 | 2026-05-22 | 15032 | 2012-05-02 | 2026-05-21 |  |
| 2108 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15133 | 2012-05-02 | 2026-05-22 | 15128 | 2012-05-02 | 2026-05-21 |  |
| 2109 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14425 | 2012-05-02 | 2026-05-22 | 14420 | 2012-05-02 | 2026-05-21 |  |
| 2114 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14448 | 2006-09-29 | 2026-05-22 | 14443 | 2006-09-29 | 2026-05-21 |  |
| 2115 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12391 | 2013-12-25 | 2026-05-22 | 12386 | 2013-12-25 | 2026-05-21 |  |
| 2201 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15128 | 2012-05-02 | 2026-05-22 | 15123 | 2012-05-02 | 2026-05-21 |  |
| 2204 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15129 | 2012-05-02 | 2026-05-22 | 15124 | 2012-05-02 | 2026-05-21 |  |
| 2206 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15154 | 2012-05-02 | 2026-05-22 | 15149 | 2012-05-02 | 2026-05-21 |  |
| 2207 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15159 | 2012-05-02 | 2026-05-22 | 15154 | 2012-05-02 | 2026-05-21 |  |
| 2208 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15116 | 2012-05-02 | 2026-05-22 | 15111 | 2012-05-02 | 2026-05-21 |  |
| 2211 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6343 | 2020-01-20 | 2026-05-22 | 6338 | 2020-01-20 | 2026-05-21 |  |
| 2221 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10869 | 2007-06-25 | 2026-05-22 | 10864 | 2007-06-25 | 2026-05-21 |  |
| 2227 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15130 | 2012-05-02 | 2026-05-22 | 15125 | 2012-05-02 | 2026-05-21 |  |
| 2228 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14122 | 2012-10-16 | 2026-05-22 | 14117 | 2012-10-16 | 2026-05-21 |  |
| 2230 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8361 | 2009-03-02 | 2026-05-22 | 8356 | 2009-03-02 | 2026-05-21 |  |
| 2231 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16136 | 2007-11-21 | 2026-05-22 | 16131 | 2007-11-21 | 2026-05-21 |  |
| 2233 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15486 | 2010-10-25 | 2026-05-22 | 15481 | 2010-10-25 | 2026-05-21 |  |
| 2236 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11217 | 2015-06-03 | 2026-05-22 | 11212 | 2015-06-03 | 2026-05-21 |  |
| 2239 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12067 | 2016-01-27 | 2026-05-22 | 12062 | 2016-01-27 | 2026-05-21 |  |
| 2241 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8277 | 2016-03-10 | 2026-05-22 | 8272 | 2016-03-10 | 2026-05-21 |  |
| 2243 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8907 | 2016-02-05 | 2026-05-22 | 8902 | 2016-02-05 | 2026-05-21 |  |
| 2247 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6826 | 2018-07-30 | 2026-05-22 | 6821 | 2018-07-30 | 2026-05-21 |  |
| 2248 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1622 | 2020-12-30 | 2026-05-22 | 1617 | 2020-12-30 | 2026-05-21 |  |
| 2250 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5895 | 2021-05-31 | 2026-05-22 | 5890 | 2021-05-31 | 2026-05-21 |  |
| 2254 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2948 | 2023-09-18 | 2026-05-22 | 2943 | 2023-09-18 | 2026-05-21 |  |
| 2258 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3015 | 2023-11-20 | 2026-05-22 | 3010 | 2023-11-20 | 2026-05-21 |  |
| 2301 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2302 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13651 | 2012-05-02 | 2026-05-22 | 13646 | 2012-05-02 | 2026-05-21 |  |
| 2303 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 2305 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14431 | 2012-05-02 | 2026-05-22 | 14426 | 2012-05-02 | 2026-05-21 |  |
| 2308 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2312 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15163 | 2012-05-02 | 2026-05-22 | 15158 | 2012-05-02 | 2026-05-21 |  |
| 2313 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2314 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14841 | 2012-05-02 | 2026-05-22 | 14836 | 2012-05-02 | 2026-05-21 |  |
| 2316 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15054 | 2012-05-02 | 2026-05-22 | 15049 | 2012-05-02 | 2026-05-21 |  |
| 2317 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15133 | 2012-05-02 | 2026-05-22 | 15128 | 2012-05-02 | 2026-05-21 |  |
| 2323 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15063 | 2012-05-02 | 2026-05-22 | 15058 | 2012-05-02 | 2026-05-21 |  |
| 2324 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 2327 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14960 | 2012-05-02 | 2026-05-22 | 14955 | 2012-05-02 | 2026-05-21 |  |
| 2328 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 2329 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15052 | 2012-05-02 | 2026-05-22 | 15047 | 2012-05-02 | 2026-05-21 |  |
| 2330 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2331 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15117 | 2012-05-02 | 2026-05-22 | 15112 | 2012-05-02 | 2026-05-21 |  |
| 2332 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15125 | 2012-05-02 | 2026-05-22 | 15120 | 2012-05-02 | 2026-05-21 |  |
| 2337 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15128 | 2012-05-02 | 2026-05-22 | 15123 | 2012-05-02 | 2026-05-21 |  |
| 2338 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14982 | 2012-05-02 | 2026-05-22 | 14977 | 2012-05-02 | 2026-05-21 |  |
| 2340 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15089 | 2012-05-02 | 2026-05-22 | 15084 | 2012-05-02 | 2026-05-21 |  |
| 2342 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14301 | 2012-05-02 | 2026-05-22 | 14296 | 2012-05-02 | 2026-05-21 |  |
| 2344 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2345 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2347 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2348 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10362 | 2014-12-22 | 2026-05-22 | 10357 | 2014-12-22 | 2026-05-21 |  |
| 2349 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15063 | 2012-05-02 | 2026-05-22 | 15058 | 2012-05-02 | 2026-05-21 |  |
| 2351 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 2352 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15142 | 2012-05-02 | 2026-05-22 | 15137 | 2012-05-02 | 2026-05-21 |  |
| 2353 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2354 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2355 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2356 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2357 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2359 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14818 | 2012-05-02 | 2026-05-22 | 14813 | 2012-05-02 | 2026-05-21 |  |
| 2360 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2362 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15156 | 2012-05-02 | 2026-05-22 | 15151 | 2012-05-02 | 2026-05-21 |  |
| 2363 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15076 | 2012-05-02 | 2026-05-22 | 15071 | 2012-05-02 | 2026-05-21 |  |
| 2364 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6583 | 2012-05-02 | 2026-05-22 | 6578 | 2012-05-02 | 2026-05-21 |  |
| 2365 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15031 | 2012-05-02 | 2026-05-22 | 15026 | 2012-05-02 | 2026-05-21 |  |
| 2367 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15166 | 2012-05-02 | 2026-05-22 | 15161 | 2012-05-02 | 2026-05-21 |  |
| 2368 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15118 | 2012-05-02 | 2026-05-22 | 15113 | 2012-05-02 | 2026-05-21 |  |
| 2369 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15161 | 2012-05-02 | 2026-05-22 | 15156 | 2012-05-02 | 2026-05-21 |  |
| 2371 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15128 | 2012-05-02 | 2026-05-22 | 15123 | 2012-05-02 | 2026-05-21 |  |
| 2373 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14957 | 2012-05-02 | 2026-05-22 | 14952 | 2012-05-02 | 2026-05-21 |  |
| 2374 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15111 | 2012-05-02 | 2026-05-22 | 15106 | 2012-05-02 | 2026-05-21 |  |
| 2375 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14523 | 2012-05-02 | 2026-05-22 | 14518 | 2012-05-02 | 2026-05-21 |  |
| 2376 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2377 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2379 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2380 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12594 | 2012-05-02 | 2026-05-22 | 12589 | 2012-05-02 | 2026-05-21 |  |
| 2382 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2383 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2385 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2387 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15016 | 2012-05-02 | 2026-05-22 | 15011 | 2012-05-02 | 2026-05-21 |  |
| 2388 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14903 | 2012-05-02 | 2026-05-22 | 14898 | 2012-05-02 | 2026-05-21 |  |
| 2390 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15062 | 2012-05-02 | 2026-05-22 | 15057 | 2012-05-02 | 2026-05-21 |  |
| 2392 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-22 | 15157 | 2012-05-02 | 2026-05-21 |  |
| 2393 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2395 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15163 | 2012-05-02 | 2026-05-22 | 15158 | 2012-05-02 | 2026-05-21 |  |
| 2397 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14811 | 2012-05-02 | 2026-05-22 | 14806 | 2012-05-02 | 2026-05-21 |  |
| 2399 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14964 | 2012-05-02 | 2026-05-22 | 14959 | 2012-05-02 | 2026-05-21 |  |
| 2401 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15159 | 2012-05-02 | 2026-05-22 | 15154 | 2012-05-02 | 2026-05-21 |  |
| 2402 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15155 | 2012-05-02 | 2026-05-22 | 15150 | 2012-05-02 | 2026-05-21 |  |
| 2404 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15123 | 2012-05-02 | 2026-05-22 | 15118 | 2012-05-02 | 2026-05-21 |  |
| 2405 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15150 | 2012-05-02 | 2026-05-22 | 15145 | 2012-05-02 | 2026-05-21 |  |
| 2406 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15144 | 2012-05-02 | 2026-05-22 | 15139 | 2012-05-02 | 2026-05-21 |  |
| 2408 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14974 | 2012-05-02 | 2026-05-22 | 14969 | 2012-05-02 | 2026-05-21 |  |
| 2409 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15137 | 2012-05-02 | 2026-05-22 | 15132 | 2012-05-02 | 2026-05-21 |  |
| 2412 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2413 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14825 | 2012-05-02 | 2026-05-22 | 14820 | 2012-05-02 | 2026-05-21 |  |
| 2414 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13996 | 2012-05-02 | 2026-05-22 | 13991 | 2012-05-02 | 2026-05-21 |  |
| 2415 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14652 | 2012-05-02 | 2026-05-22 | 14647 | 2012-05-02 | 2026-05-21 |  |
| 2417 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15042 | 2012-05-02 | 2026-05-22 | 15037 | 2012-05-02 | 2026-05-21 |  |
| 2419 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15122 | 2012-05-02 | 2026-05-22 | 15117 | 2012-05-02 | 2026-05-21 |  |
| 2420 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15128 | 2012-05-02 | 2026-05-22 | 15123 | 2012-05-02 | 2026-05-21 |  |
| 2421 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15103 | 2012-05-02 | 2026-05-22 | 15098 | 2012-05-02 | 2026-05-21 |  |
| 2423 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13633 | 2012-05-02 | 2026-05-22 | 13628 | 2012-05-02 | 2026-05-21 |  |
| 2424 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6596 | 2012-05-17 | 2026-05-22 | 6591 | 2012-05-17 | 2026-05-21 |  |
| 2425 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15040 | 2012-05-02 | 2026-05-22 | 15035 | 2012-05-02 | 2026-05-21 |  |
| 2426 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15113 | 2012-05-02 | 2026-05-22 | 15108 | 2012-05-02 | 2026-05-21 |  |
| 2427 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14648 | 2012-05-02 | 2026-05-22 | 14643 | 2012-05-02 | 2026-05-21 |  |
| 2428 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15159 | 2012-05-02 | 2026-05-22 | 15154 | 2012-05-02 | 2026-05-21 |  |
| 2429 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7623 | 2013-05-21 | 2026-05-22 | 7618 | 2013-05-21 | 2026-05-21 |  |
| 2430 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14765 | 2012-05-02 | 2026-05-22 | 14760 | 2012-05-02 | 2026-05-21 |  |
| 2431 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14900 | 2012-05-02 | 2026-05-22 | 14895 | 2012-05-02 | 2026-05-21 |  |
| 2432 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3210 | 2023-05-31 | 2026-05-22 | 3205 | 2023-05-31 | 2026-05-21 |  |
| 2433 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14181 | 2012-05-02 | 2026-05-22 | 14176 | 2012-05-02 | 2026-05-21 |  |
| 2434 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10386 | 2012-05-02 | 2026-05-22 | 10381 | 2012-05-02 | 2026-05-21 |  |
| 2436 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15078 | 2012-05-02 | 2026-05-22 | 15073 | 2012-05-02 | 2026-05-21 |  |
| 2438 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9704 | 2012-11-30 | 2026-05-22 | 9699 | 2012-11-30 | 2026-05-21 |  |
| 2439 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 2440 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12463 | 2012-05-02 | 2026-05-22 | 12458 | 2012-05-02 | 2026-05-21 |  |
| 2441 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15161 | 2012-05-02 | 2026-05-22 | 15156 | 2012-05-02 | 2026-05-21 |  |
| 2442 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14741 | 2012-05-02 | 2026-05-22 | 14736 | 2012-05-02 | 2026-05-21 |  |
| 2444 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14824 | 2012-05-02 | 2026-05-22 | 14819 | 2012-05-02 | 2026-05-21 |  |
| 2449 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 2450 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15133 | 2012-05-02 | 2026-05-22 | 15128 | 2012-05-02 | 2026-05-21 |  |
| 2451 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 2453 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14384 | 2012-05-02 | 2026-05-22 | 14379 | 2012-05-02 | 2026-05-21 |  |
| 2454 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2455 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15145 | 2012-05-02 | 2026-05-22 | 15140 | 2012-05-02 | 2026-05-21 |  |
| 2457 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15161 | 2012-05-02 | 2026-05-22 | 15156 | 2012-05-02 | 2026-05-21 |  |
| 2458 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15137 | 2012-05-02 | 2026-05-22 | 15132 | 2012-05-02 | 2026-05-21 |  |
| 2459 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15017 | 2012-05-02 | 2026-05-22 | 15012 | 2012-05-02 | 2026-05-21 |  |
| 2460 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13898 | 2012-05-02 | 2026-05-22 | 13893 | 2012-05-02 | 2026-05-21 |  |
| 2461 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15067 | 2012-05-02 | 2026-05-22 | 15062 | 2012-05-02 | 2026-05-21 |  |
| 2462 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14757 | 2012-05-02 | 2026-05-22 | 14752 | 2012-05-02 | 2026-05-21 |  |
| 2464 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15134 | 2012-05-02 | 2026-05-22 | 15129 | 2012-05-02 | 2026-05-21 |  |
| 2465 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14007 | 2012-05-02 | 2026-05-22 | 14002 | 2012-05-02 | 2026-05-21 |  |
| 2466 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14343 | 2012-05-02 | 2026-05-22 | 14338 | 2012-05-02 | 2026-05-21 |  |
| 2467 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15105 | 2012-05-02 | 2026-05-22 | 15100 | 2012-05-02 | 2026-05-21 |  |
| 2468 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12111 | 2012-05-02 | 2026-05-22 | 12106 | 2012-05-02 | 2026-05-21 |  |
| 2471 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12492 | 2012-05-02 | 2026-05-22 | 12487 | 2012-05-02 | 2026-05-21 |  |
| 2472 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15080 | 2012-05-02 | 2026-05-22 | 15075 | 2012-05-02 | 2026-05-21 |  |
| 2474 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 2476 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15057 | 2012-05-02 | 2026-05-22 | 15052 | 2012-05-02 | 2026-05-21 |  |
| 2477 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14694 | 2012-05-02 | 2026-05-22 | 14689 | 2012-05-02 | 2026-05-21 |  |
| 2478 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14891 | 2012-05-02 | 2026-05-22 | 14886 | 2012-05-02 | 2026-05-21 |  |
| 2480 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15054 | 2012-05-02 | 2026-05-22 | 15049 | 2012-05-02 | 2026-05-21 |  |
| 2481 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15140 | 2012-05-02 | 2026-05-22 | 15135 | 2012-05-02 | 2026-05-21 |  |
| 2482 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13361 | 2012-05-02 | 2026-05-22 | 13356 | 2012-05-02 | 2026-05-21 |  |
| 2483 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13882 | 2012-05-02 | 2026-05-22 | 13877 | 2012-05-02 | 2026-05-21 |  |
| 2484 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15118 | 2012-05-02 | 2026-05-22 | 15113 | 2012-05-02 | 2026-05-21 |  |
| 2485 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 2486 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15087 | 2012-05-02 | 2026-05-22 | 15082 | 2012-05-02 | 2026-05-21 |  |
| 2488 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14915 | 2012-05-02 | 2026-05-22 | 14910 | 2012-05-02 | 2026-05-21 |  |
| 2489 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15030 | 2012-05-02 | 2026-05-22 | 15025 | 2012-05-02 | 2026-05-21 |  |
| 2491 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10160 | 2012-05-02 | 2026-05-22 | 10155 | 2012-05-02 | 2026-05-21 |  |
| 2492 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15080 | 2012-05-02 | 2026-05-22 | 15075 | 2012-05-02 | 2026-05-21 |  |
| 2493 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14967 | 2012-05-02 | 2026-05-22 | 14962 | 2012-05-02 | 2026-05-21 |  |
| 2495 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15086 | 2012-05-02 | 2026-05-22 | 15081 | 2012-05-02 | 2026-05-21 |  |
| 2496 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9457 | 2012-08-30 | 2026-05-22 | 9452 | 2012-08-30 | 2026-05-21 |  |
| 2497 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14889 | 2012-05-02 | 2026-05-22 | 14884 | 2012-05-02 | 2026-05-21 |  |
| 2498 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2501 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15119 | 2012-05-02 | 2026-05-22 | 15114 | 2012-05-02 | 2026-05-21 |  |
| 2504 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15118 | 2012-05-02 | 2026-05-22 | 15113 | 2012-05-02 | 2026-05-21 |  |
| 2505 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15112 | 2012-05-02 | 2026-05-22 | 15107 | 2012-05-02 | 2026-05-21 |  |
| 2506 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14096 | 2012-05-02 | 2026-05-22 | 14091 | 2012-05-02 | 2026-05-21 |  |
| 2509 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14210 | 2012-05-02 | 2026-05-22 | 14205 | 2012-05-02 | 2026-05-21 |  |
| 2511 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15144 | 2012-05-02 | 2026-05-22 | 15139 | 2012-05-02 | 2026-05-21 |  |
| 2514 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15093 | 2012-05-02 | 2026-05-22 | 15088 | 2012-05-02 | 2026-05-21 |  |
| 2515 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2516 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13921 | 2012-05-02 | 2026-05-22 | 13916 | 2012-05-02 | 2026-05-21 |  |
| 2520 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2524 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14735 | 2012-05-02 | 2026-05-22 | 14730 | 2012-05-02 | 2026-05-21 |  |
| 2527 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14917 | 2012-05-02 | 2026-05-22 | 14912 | 2012-05-02 | 2026-05-21 |  |
| 2528 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12494 | 2012-05-07 | 2026-05-22 | 12489 | 2012-05-07 | 2026-05-21 |  |
| 2530 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14557 | 2012-05-02 | 2026-05-22 | 14552 | 2012-05-02 | 2026-05-21 |  |
| 2534 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15084 | 2012-05-02 | 2026-05-22 | 15079 | 2012-05-02 | 2026-05-21 |  |
| 2535 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15014 | 2012-05-02 | 2026-05-22 | 15009 | 2012-05-02 | 2026-05-21 |  |
| 2536 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15144 | 2012-05-02 | 2026-05-22 | 15139 | 2012-05-02 | 2026-05-21 |  |
| 2537 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14639 | 2012-05-04 | 2026-05-22 | 14634 | 2012-05-04 | 2026-05-21 |  |
| 2538 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15151 | 2012-05-02 | 2026-05-22 | 15146 | 2012-05-02 | 2026-05-21 |  |
| 2539 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13052 | 2012-05-02 | 2026-05-22 | 13047 | 2012-05-02 | 2026-05-21 |  |
| 2540 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10340 | 2014-11-19 | 2026-05-22 | 10335 | 2014-11-19 | 2026-05-21 |  |
| 2542 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15145 | 2012-05-02 | 2026-05-22 | 15140 | 2012-05-02 | 2026-05-21 |  |
| 2543 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11230 | 2012-05-02 | 2026-05-22 | 11225 | 2012-05-02 | 2026-05-21 |  |
| 2545 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15156 | 2012-05-02 | 2026-05-22 | 15151 | 2012-05-02 | 2026-05-21 |  |
| 2546 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13648 | 2012-05-02 | 2026-05-22 | 13643 | 2012-05-02 | 2026-05-21 |  |
| 2547 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15159 | 2012-05-02 | 2026-05-22 | 15154 | 2012-05-02 | 2026-05-21 |  |
| 2548 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2596 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9343 | 2008-06-25 | 2026-05-22 | 9338 | 2008-06-25 | 2026-05-21 |  |
| 2597 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14236 | 2008-01-29 | 2026-05-22 | 14231 | 2008-01-29 | 2026-05-21 |  |
| 2601 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-22 | 15157 | 2012-05-02 | 2026-05-21 |  |
| 2603 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15137 | 2012-05-02 | 2026-05-22 | 15132 | 2012-05-02 | 2026-05-21 |  |
| 2605 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2606 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 2607 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15124 | 2012-05-02 | 2026-05-22 | 15119 | 2012-05-02 | 2026-05-21 |  |
| 2608 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15139 | 2012-05-02 | 2026-05-22 | 15134 | 2012-05-02 | 2026-05-21 |  |
| 2609 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15123 | 2012-05-02 | 2026-05-22 | 15118 | 2012-05-02 | 2026-05-21 |  |
| 2610 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2611 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13383 | 2012-05-02 | 2026-05-22 | 13378 | 2012-05-02 | 2026-05-21 |  |
| 2612 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15090 | 2012-05-02 | 2026-05-22 | 15085 | 2012-05-02 | 2026-05-21 |  |
| 2613 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13086 | 2012-05-02 | 2026-05-22 | 13081 | 2012-05-02 | 2026-05-21 |  |
| 2614 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14975 | 2012-05-02 | 2026-05-22 | 14970 | 2012-05-02 | 2026-05-21 |  |
| 2615 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2616 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14173 | 2012-05-02 | 2026-05-22 | 14168 | 2012-05-02 | 2026-05-21 |  |
| 2617 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15122 | 2012-05-02 | 2026-05-22 | 15117 | 2012-05-02 | 2026-05-21 |  |
| 2618 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2630 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8860 | 2017-09-18 | 2026-05-22 | 8855 | 2017-09-18 | 2026-05-21 |  |
| 2633 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13798 | 2007-06-20 | 2026-05-22 | 13793 | 2007-06-20 | 2026-05-21 |  |
| 2634 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13446 | 2014-08-25 | 2026-05-22 | 13441 | 2014-08-25 | 2026-05-21 |  |
| 2636 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15559 | 2007-08-02 | 2026-05-22 | 15554 | 2007-08-02 | 2026-05-21 |  |
| 2637 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15244 | 2009-09-08 | 2026-05-22 | 15239 | 2009-09-08 | 2026-05-21 |  |
| 2640 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13556 | 2011-07-26 | 2026-05-22 | 13551 | 2011-07-26 | 2026-05-21 |  |
| 2641 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11190 | 2012-11-06 | 2026-05-22 | 11185 | 2012-11-06 | 2026-05-21 |  |
| 2642 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12728 | 2012-12-26 | 2026-05-22 | 12723 | 2012-12-26 | 2026-05-21 |  |
| 2643 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9147 | 2015-08-25 | 2026-05-22 | 9142 | 2015-08-25 | 2026-05-21 |  |
| 2645 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4326 | 2022-02-18 | 2026-05-22 | 4321 | 2022-02-18 | 2026-05-21 |  |
| 2646 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3244 | 2022-09-30 | 2026-05-22 | 3239 | 2022-09-30 | 2026-05-21 |  |
| 2701 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12265 | 2012-05-02 | 2026-05-22 | 12260 | 2012-05-02 | 2026-05-21 |  |
| 2702 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12121 | 2012-05-02 | 2026-05-22 | 12116 | 2012-05-02 | 2026-05-21 |  |
| 2704 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15034 | 2012-05-02 | 2026-05-22 | 15029 | 2012-05-02 | 2026-05-21 |  |
| 2705 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14681 | 2012-05-02 | 2026-05-22 | 14676 | 2012-05-02 | 2026-05-21 |  |
| 2706 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14079 | 2012-05-02 | 2026-05-22 | 14074 | 2012-05-02 | 2026-05-21 |  |
| 2707 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15146 | 2012-05-02 | 2026-05-22 | 15141 | 2012-05-02 | 2026-05-21 |  |
| 2712 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8756 | 2013-07-31 | 2026-05-22 | 8751 | 2013-07-31 | 2026-05-21 |  |
| 2718 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3967 | 2009-07-07 | 2026-05-22 | 3962 | 2009-07-07 | 2026-05-21 |  |
| 2722 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10838 | 2012-05-02 | 2026-05-22 | 10833 | 2012-05-02 | 2026-05-21 |  |
| 2723 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15160 | 2012-05-02 | 2026-05-22 | 15155 | 2012-05-02 | 2026-05-21 |  |
| 2726 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11399 | 2014-12-24 | 2026-05-22 | 11394 | 2014-12-24 | 2026-05-21 |  |
| 2727 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15353 | 2011-04-29 | 2026-05-22 | 15348 | 2011-04-29 | 2026-05-21 |  |
| 2729 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12164 | 2011-09-16 | 2026-05-22 | 12159 | 2011-09-16 | 2026-05-21 |  |
| 2731 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14089 | 2012-10-23 | 2026-05-22 | 14084 | 2012-10-23 | 2026-05-21 |  |
| 2732 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11930 | 2013-01-14 | 2026-05-22 | 11925 | 2013-01-14 | 2026-05-21 |  |
| 2734 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8372 | 2013-07-05 | 2026-05-22 | 8367 | 2013-07-05 | 2026-05-21 |  |
| 2736 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8072 | 2015-12-14 | 2026-05-22 | 8067 | 2015-12-14 | 2026-05-21 |  |
| 2739 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10659 | 2015-06-03 | 2026-05-22 | 10654 | 2015-06-03 | 2026-05-21 |  |
| 2743 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6810 | 2015-04-02 | 2026-05-22 | 6805 | 2015-04-02 | 2026-05-21 |  |
| 2745 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6965 | 2015-09-23 | 2026-05-22 | 6960 | 2015-09-23 | 2026-05-21 |  |
| 2748 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11019 | 2015-11-02 | 2026-05-22 | 11014 | 2015-11-02 | 2026-05-21 |  |
| 2752 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4139 | 2018-08-21 | 2026-05-22 | 4134 | 2018-08-21 | 2026-05-21 |  |
| 2753 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5957 | 2020-09-29 | 2026-05-22 | 5952 | 2020-09-29 | 2026-05-21 |  |
| 2754 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6651 | 2020-03-03 | 2026-05-22 | 6646 | 2020-03-03 | 2026-05-21 |  |
| 2755 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4308 | 2019-08-23 | 2026-05-22 | 4303 | 2019-08-23 | 2026-05-21 |  |
| 2756 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3645 | 2022-01-19 | 2026-05-22 | 3640 | 2022-01-19 | 2026-05-21 |  |
| 2762 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2785 | 2024-01-24 | 2026-05-22 | 2780 | 2024-01-24 | 2026-05-21 |  |
| 2801 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2812 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 2816 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14872 | 2012-05-02 | 2026-05-22 | 14867 | 2012-05-02 | 2026-05-21 |  |
| 2820 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15156 | 2012-05-02 | 2026-05-22 | 15151 | 2012-05-02 | 2026-05-21 |  |
| 2832 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14965 | 2012-05-02 | 2026-05-22 | 14960 | 2012-05-02 | 2026-05-21 |  |
| 2834 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2836 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15120 | 2012-05-02 | 2026-05-22 | 15115 | 2012-05-02 | 2026-05-21 |  |
| 2838 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15145 | 2012-05-02 | 2026-05-22 | 15140 | 2012-05-02 | 2026-05-21 |  |
| 2845 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15156 | 2012-05-02 | 2026-05-22 | 15151 | 2012-05-02 | 2026-05-21 |  |
| 2849 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14562 | 2012-05-02 | 2026-05-22 | 14557 | 2012-05-02 | 2026-05-21 |  |
| 2850 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15122 | 2012-05-02 | 2026-05-22 | 15117 | 2012-05-02 | 2026-05-21 |  |
| 2851 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14551 | 2012-05-02 | 2026-05-22 | 14546 | 2012-05-02 | 2026-05-21 |  |
| 2852 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14871 | 2012-05-02 | 2026-05-22 | 14866 | 2012-05-02 | 2026-05-21 |  |
| 2855 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15140 | 2012-05-02 | 2026-05-22 | 15135 | 2012-05-02 | 2026-05-21 |  |
| 2867 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14767 | 2012-02-09 | 2026-05-22 | 14762 | 2012-02-09 | 2026-05-21 |  |
| 2880 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2881 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2881B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9265 | 2018-04-23 | 2026-05-22 | 9260 | 2018-04-23 | 2026-05-21 |  |
| 2881C | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2080 | 2021-10-29 | 2026-05-22 | 2075 | 2021-10-29 | 2026-05-19 |  |
| 2882 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2882A | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8756 | 2016-02-05 | 2026-05-22 | 8751 | 2016-02-05 | 2026-05-21 |  |
| 2882B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8875 | 2018-08-08 | 2026-05-22 | 8870 | 2018-08-08 | 2026-05-21 |  |
| 2883 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2883B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5295 | 2021-12-30 | 2026-05-22 | 5290 | 2021-12-30 | 2026-05-21 |  |
| 2884 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 2885 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2886 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2887 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15163 | 2012-05-02 | 2026-05-22 | 15158 | 2012-05-02 | 2026-05-21 |  |
| 2887E | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5040 | 2017-02-10 | 2026-05-22 | 5035 | 2017-02-10 | 2026-05-21 |  |
| 2887G | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 200 | 2025-07-25 | 2026-05-22 | 195 | 2025-07-25 | 2026-04-23 |  |
| 2887H | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 310 | 2025-07-25 | 2026-05-22 | 305 | 2025-07-25 | 2026-05-20 |  |
| 2887I | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1000 | 2025-07-24 | 2026-05-22 | 995 | 2025-07-24 | 2026-05-21 |  |
| 2889 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15148 | 2012-05-02 | 2026-05-22 | 15143 | 2012-05-02 | 2026-05-21 |  |
| 2890 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2891 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2891B | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8470 | 2018-01-26 | 2026-05-22 | 8465 | 2018-01-26 | 2026-05-21 |  |
| 2891C | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7180 | 2019-05-06 | 2026-05-22 | 7175 | 2019-05-06 | 2026-05-21 |  |
| 2892 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2897 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10982 | 2007-06-25 | 2026-05-22 | 10977 | 2007-06-25 | 2026-05-21 |  |
| 2901 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9463 | 2012-05-02 | 2026-05-22 | 9458 | 2012-05-02 | 2026-05-21 |  |
| 2903 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 2904 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12033 | 2012-05-02 | 2026-05-22 | 12028 | 2012-05-02 | 2026-05-21 |  |
| 2905 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15139 | 2012-05-02 | 2026-05-22 | 15134 | 2012-05-02 | 2026-05-21 |  |
| 2906 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13702 | 2012-05-02 | 2026-05-22 | 13697 | 2012-05-02 | 2026-05-21 |  |
| 2908 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15143 | 2012-05-02 | 2026-05-22 | 15138 | 2012-05-02 | 2026-05-21 |  |
| 2910 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9361 | 2012-05-02 | 2026-05-22 | 9356 | 2012-05-02 | 2026-05-21 |  |
| 2911 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14171 | 2012-05-02 | 2026-05-22 | 14166 | 2012-05-02 | 2026-05-21 |  |
| 2912 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 2913 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 2915 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15137 | 2012-05-02 | 2026-05-22 | 15132 | 2012-05-02 | 2026-05-21 |  |
| 2916 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7345 | 2005-03-16 | 2026-05-22 | 7340 | 2005-03-16 | 2026-05-21 |  |
| 2923 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13030 | 2012-12-07 | 2026-05-22 | 13025 | 2012-12-07 | 2026-05-21 |  |
| 2929 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13769 | 2013-12-30 | 2026-05-22 | 13764 | 2013-12-30 | 2026-05-21 |  |
| 2937 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3106 | 2016-06-15 | 2026-05-22 | 3101 | 2016-06-15 | 2026-05-21 |  |
| 2939 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9264 | 2017-08-08 | 2026-05-22 | 9259 | 2017-08-08 | 2026-05-21 |  |
| 2945 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3694 | 2019-04-24 | 2026-05-22 | 3689 | 2019-04-24 | 2026-05-21 |  |
| 2949 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 831 | 2023-09-18 | 2026-05-22 | 826 | 2023-09-18 | 2026-05-21 |  |
| 3002 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12667 | 2012-05-02 | 2026-05-22 | 12662 | 2012-05-02 | 2026-05-21 |  |
| 3003 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15121 | 2012-05-02 | 2026-05-22 | 15116 | 2012-05-02 | 2026-05-21 |  |
| 3004 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14513 | 2012-05-02 | 2026-05-22 | 14508 | 2012-05-02 | 2026-05-21 |  |
| 3005 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3006 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15169 | 2012-05-02 | 2026-05-22 | 15164 | 2012-05-02 | 2026-05-21 |  |
| 3008 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3010 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15135 | 2012-05-02 | 2026-05-22 | 15130 | 2012-05-02 | 2026-05-21 |  |
| 3011 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14131 | 2012-05-02 | 2026-05-22 | 14126 | 2012-05-02 | 2026-05-21 |  |
| 3013 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15106 | 2012-05-02 | 2026-05-22 | 15101 | 2012-05-02 | 2026-05-21 |  |
| 3014 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15125 | 2012-05-02 | 2026-05-22 | 15120 | 2012-05-02 | 2026-05-21 |  |
| 3015 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14847 | 2012-05-02 | 2026-05-22 | 14842 | 2012-05-02 | 2026-05-21 |  |
| 3016 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15046 | 2012-05-02 | 2026-05-22 | 15041 | 2012-05-02 | 2026-05-21 |  |
| 3017 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15165 | 2012-05-02 | 2026-05-22 | 15160 | 2012-05-02 | 2026-05-21 |  |
| 3019 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 3021 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12184 | 2012-05-04 | 2026-05-22 | 12179 | 2012-05-04 | 2026-05-21 |  |
| 3022 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15093 | 2012-05-02 | 2026-05-22 | 15088 | 2012-05-02 | 2026-05-21 |  |
| 3023 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3024 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14119 | 2012-05-02 | 2026-05-22 | 14114 | 2012-05-02 | 2026-05-21 |  |
| 3025 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14076 | 2012-05-02 | 2026-05-22 | 14071 | 2012-05-02 | 2026-05-21 |  |
| 3026 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15080 | 2012-05-02 | 2026-05-22 | 15075 | 2012-05-02 | 2026-05-21 |  |
| 3027 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14198 | 2012-05-02 | 2026-05-22 | 14193 | 2012-05-02 | 2026-05-21 |  |
| 3028 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15053 | 2012-05-02 | 2026-05-22 | 15048 | 2012-05-02 | 2026-05-21 |  |
| 3029 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14987 | 2012-05-02 | 2026-05-22 | 14982 | 2012-05-02 | 2026-05-21 |  |
| 3030 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15155 | 2012-05-02 | 2026-05-22 | 15150 | 2012-05-02 | 2026-05-21 |  |
| 3031 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15078 | 2012-05-02 | 2026-05-22 | 15073 | 2012-05-02 | 2026-05-21 |  |
| 3032 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14698 | 2012-05-02 | 2026-05-22 | 14693 | 2012-05-02 | 2026-05-21 |  |
| 3033 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15117 | 2012-05-02 | 2026-05-22 | 15112 | 2012-05-02 | 2026-05-21 |  |
| 3034 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3035 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15132 | 2012-05-02 | 2026-05-22 | 15127 | 2012-05-02 | 2026-05-21 |  |
| 3036 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15145 | 2012-05-02 | 2026-05-22 | 15140 | 2012-05-02 | 2026-05-21 |  |
| 3037 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 3038 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14903 | 2012-05-02 | 2026-05-22 | 14898 | 2012-05-02 | 2026-05-21 |  |
| 3040 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13318 | 2012-05-02 | 2026-05-22 | 13313 | 2012-05-02 | 2026-05-21 |  |
| 3041 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15092 | 2012-05-02 | 2026-05-22 | 15087 | 2012-05-02 | 2026-05-21 |  |
| 3042 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15163 | 2012-05-02 | 2026-05-22 | 15158 | 2012-05-02 | 2026-05-21 |  |
| 3043 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6749 | 2012-05-02 | 2026-05-22 | 6744 | 2012-05-02 | 2026-05-21 |  |
| 3044 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3045 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 3046 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11407 | 2012-05-02 | 2026-05-22 | 11402 | 2012-05-02 | 2026-05-21 |  |
| 3047 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15091 | 2012-05-02 | 2026-05-22 | 15086 | 2012-05-02 | 2026-05-21 |  |
| 3048 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14849 | 2012-05-02 | 2026-05-22 | 14844 | 2012-05-02 | 2026-05-21 |  |
| 3049 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15118 | 2012-05-02 | 2026-05-22 | 15113 | 2012-05-02 | 2026-05-21 |  |
| 3050 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14856 | 2012-05-02 | 2026-05-22 | 14851 | 2012-05-02 | 2026-05-21 |  |
| 3051 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10810 | 2012-07-11 | 2026-05-22 | 10805 | 2012-07-11 | 2026-05-21 |  |
| 3052 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14556 | 2012-05-02 | 2026-05-22 | 14551 | 2012-05-02 | 2026-05-21 |  |
| 3054 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12230 | 2012-05-02 | 2026-05-22 | 12225 | 2012-05-02 | 2026-05-21 |  |
| 3055 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14689 | 2012-05-02 | 2026-05-22 | 14684 | 2012-05-02 | 2026-05-21 |  |
| 3056 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15107 | 2012-05-02 | 2026-05-22 | 15102 | 2012-05-02 | 2026-05-21 |  |
| 3057 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14395 | 2012-05-02 | 2026-05-22 | 14390 | 2012-05-02 | 2026-05-21 |  |
| 3058 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14294 | 2012-05-02 | 2026-05-22 | 14289 | 2012-05-02 | 2026-05-21 |  |
| 3059 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15139 | 2012-05-02 | 2026-05-22 | 15134 | 2012-05-02 | 2026-05-21 |  |
| 3060 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15103 | 2012-05-02 | 2026-05-22 | 15098 | 2012-05-02 | 2026-05-21 |  |
| 3062 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15160 | 2012-05-02 | 2026-05-22 | 15155 | 2012-05-02 | 2026-05-21 |  |
| 3066 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8257 | 2005-01-03 | 2026-05-22 | 8252 | 2005-01-03 | 2026-05-21 |  |
| 3071 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14001 | 2005-01-03 | 2026-05-22 | 13996 | 2005-01-03 | 2026-05-21 |  |
| 3073 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7123 | 2007-06-27 | 2026-05-22 | 7118 | 2007-06-27 | 2026-05-21 |  |
| 3078 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18616 | 2005-02-14 | 2026-05-22 | 18611 | 2005-02-14 | 2026-05-21 |  |
| 3081 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13040 | 2014-07-15 | 2026-05-22 | 13035 | 2014-07-15 | 2026-05-21 |  |
| 3083 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19281 | 2005-01-03 | 2026-05-22 | 19276 | 2005-01-03 | 2026-05-21 |  |
| 3086 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11962 | 2005-03-08 | 2026-05-22 | 11957 | 2005-03-08 | 2026-05-21 |  |
| 3088 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17209 | 2005-04-29 | 2026-05-22 | 17204 | 2005-04-29 | 2026-05-21 |  |
| 3090 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16655 | 2005-02-16 | 2026-05-22 | 16650 | 2005-02-16 | 2026-05-21 |  |
| 3092 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13735 | 2007-08-03 | 2026-05-22 | 13730 | 2007-08-03 | 2026-05-21 |  |
| 3093 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8605 | 2005-06-17 | 2026-05-22 | 8600 | 2005-06-17 | 2026-05-21 |  |
| 3094 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14865 | 2007-06-20 | 2026-05-22 | 14860 | 2007-06-20 | 2026-05-21 |  |
| 3095 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9878 | 2005-05-26 | 2026-05-22 | 9873 | 2005-05-26 | 2026-05-21 |  |
| 3105 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15886 | 2009-10-20 | 2026-05-22 | 15881 | 2009-10-20 | 2026-05-21 |  |
| 3114 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11874 | 2005-02-03 | 2026-05-22 | 11869 | 2005-02-03 | 2026-05-21 |  |
| 3115 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2861 | 2005-04-08 | 2026-05-22 | 2856 | 2005-04-08 | 2026-05-21 |  |
| 3118 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9806 | 2008-04-16 | 2026-05-22 | 9801 | 2008-04-16 | 2026-05-20 |  |
| 3122 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9992 | 2009-04-20 | 2026-05-22 | 9987 | 2009-04-20 | 2026-05-21 |  |
| 3128 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11600 | 2006-09-11 | 2026-05-22 | 11595 | 2006-09-11 | 2026-05-21 |  |
| 3130 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12544 | 2012-05-02 | 2026-05-22 | 12539 | 2012-05-02 | 2026-05-21 |  |
| 3131 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15017 | 2009-07-21 | 2026-05-22 | 15012 | 2009-07-21 | 2026-05-21 |  |
| 3135 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1721 | 2022-02-18 | 2026-05-22 | 1716 | 2022-02-18 | 2026-05-21 |  |
| 3138 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7009 | 2008-05-14 | 2026-05-22 | 7004 | 2008-05-14 | 2026-05-21 |  |
| 3141 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14105 | 2007-07-09 | 2026-05-22 | 14100 | 2007-07-09 | 2026-05-21 |  |
| 3147 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8423 | 2017-05-23 | 2026-05-22 | 8418 | 2017-05-23 | 2026-05-21 |  |
| 3149 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16679 | 2007-06-25 | 2026-05-22 | 16674 | 2007-06-25 | 2026-05-21 |  |
| 3150 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2634 | 2010-12-06 | 2026-05-22 | 2629 | 2010-12-06 | 2026-05-21 |  |
| 3152 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18017 | 2007-06-26 | 2026-05-22 | 18012 | 2007-06-26 | 2026-05-21 |  |
| 3158 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 772 | 2019-03-08 | 2026-05-22 | 767 | 2019-03-08 | 2026-05-14 |  |
| 3162 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9393 | 2005-10-03 | 2026-05-22 | 9388 | 2005-10-03 | 2026-05-21 |  |
| 3163 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14720 | 2012-01-05 | 2026-05-22 | 14715 | 2012-01-05 | 2026-05-21 |  |
| 3164 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14666 | 2008-05-20 | 2026-05-22 | 14661 | 2008-05-20 | 2026-05-21 |  |
| 3167 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14118 | 2007-06-25 | 2026-05-22 | 14113 | 2007-06-25 | 2026-05-21 |  |
| 3168 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2739 | 2007-08-14 | 2026-05-22 | 2734 | 2007-08-14 | 2026-05-21 |  |
| 3169 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15199 | 2007-07-02 | 2026-05-22 | 15194 | 2007-07-02 | 2026-05-21 |  |
| 3171 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8069 | 2005-02-14 | 2026-05-22 | 8064 | 2005-02-14 | 2026-05-21 |  |
| 3176 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15235 | 2007-07-11 | 2026-05-22 | 15230 | 2007-07-11 | 2026-05-21 |  |
| 3178 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8572 | 2008-02-19 | 2026-05-22 | 8567 | 2008-02-19 | 2026-05-21 |  |
| 3188 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11253 | 2005-01-04 | 2026-05-22 | 11248 | 2005-01-04 | 2026-05-21 |  |
| 3189 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3191 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9744 | 2005-04-12 | 2026-05-22 | 9739 | 2005-04-12 | 2026-05-21 |  |
| 3205 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13378 | 2006-04-03 | 2026-05-22 | 13373 | 2006-04-03 | 2026-05-21 |  |
| 3206 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13967 | 2007-02-08 | 2026-05-22 | 13962 | 2007-02-08 | 2026-05-21 |  |
| 3207 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9033 | 2005-02-21 | 2026-05-22 | 9028 | 2005-02-21 | 2026-05-21 |  |
| 3209 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15289 | 2005-01-04 | 2026-05-22 | 15284 | 2005-01-04 | 2026-05-21 |  |
| 3211 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20330 | 2005-01-03 | 2026-05-22 | 20325 | 2005-01-03 | 2026-05-21 |  |
| 3213 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15667 | 2005-01-12 | 2026-05-22 | 15662 | 2005-01-12 | 2026-05-21 |  |
| 3217 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13832 | 2005-01-03 | 2026-05-22 | 13827 | 2005-01-03 | 2026-05-21 |  |
| 3218 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16071 | 2005-01-12 | 2026-05-22 | 16066 | 2005-01-12 | 2026-05-21 |  |
| 3219 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7275 | 2005-01-26 | 2026-05-22 | 7270 | 2005-01-26 | 2026-05-21 |  |
| 3221 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15969 | 2006-02-22 | 2026-05-22 | 15964 | 2006-02-22 | 2026-05-21 |  |
| 3224 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13620 | 2005-05-13 | 2026-05-22 | 13615 | 2005-05-13 | 2026-05-21 |  |
| 3226 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4623 | 2005-01-12 | 2026-05-22 | 4618 | 2005-01-12 | 2026-05-20 |  |
| 3227 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19623 | 2006-05-04 | 2026-05-22 | 19618 | 2006-05-04 | 2026-05-21 |  |
| 3228 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15004 | 2005-03-07 | 2026-05-22 | 14999 | 2005-03-07 | 2026-05-21 |  |
| 3229 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11846 | 2005-04-13 | 2026-05-22 | 11841 | 2005-04-13 | 2026-05-21 |  |
| 3230 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10441 | 2006-09-25 | 2026-05-22 | 10436 | 2006-09-25 | 2026-05-21 |  |
| 3231 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3232 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12427 | 2005-04-01 | 2026-05-22 | 12422 | 2005-04-01 | 2026-05-21 |  |
| 3234 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15947 | 2007-06-21 | 2026-05-22 | 15942 | 2007-06-21 | 2026-05-21 |  |
| 3236 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14619 | 2005-01-11 | 2026-05-22 | 14614 | 2005-01-11 | 2026-05-21 |  |
| 3252 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8728 | 2005-01-03 | 2026-05-22 | 8723 | 2005-01-03 | 2026-05-21 |  |
| 3257 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15102 | 2010-04-28 | 2026-05-22 | 15097 | 2010-04-28 | 2026-05-21 |  |
| 3259 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11221 | 2007-06-25 | 2026-05-22 | 11216 | 2007-06-25 | 2026-05-21 |  |
| 3259 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | ConnectionError: HTTPSConnectionPool(host='api.finmindtrade.com', port=443): Max retries exceeded with url: /api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=3259&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA (Caused by NameResolutionError("HTTPSConnection(host='api.finmindtrade.com', port=443): Failed to resolve 'api.finmindtrade.com' ([Errno -3] Temporary failure in name resolution)")) |
| 3259 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | ConnectionError: HTTPSConnectionPool(host='api.finmindtrade.com', port=443): Max retries exceeded with url: /api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=3259&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA (Caused by NameResolutionError("HTTPSConnection(host='api.finmindtrade.com', port=443): Failed to resolve 'api.finmindtrade.com' ([Errno -3] Temporary failure in name resolution)")) |
| 3260 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20535 | 2005-01-03 | 2026-05-22 | 20530 | 2005-01-03 | 2026-05-21 |  |
| 3264 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20543 | 2005-01-05 | 2026-05-22 | 20538 | 2005-01-05 | 2026-05-21 |  |
| 3265 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19182 | 2005-08-08 | 2026-05-22 | 19177 | 2005-08-08 | 2026-05-21 |  |
| 3266 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14668 | 2005-01-25 | 2026-05-22 | 14663 | 2005-01-25 | 2026-05-21 |  |
| 3268 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11063 | 2005-01-27 | 2026-05-22 | 11058 | 2005-01-27 | 2026-05-21 |  |
| 3272 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13566 | 2007-08-13 | 2026-05-22 | 13561 | 2007-08-13 | 2026-05-21 |  |
| 3276 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11333 | 2005-01-12 | 2026-05-22 | 11328 | 2005-01-12 | 2026-05-21 |  |
| 3284 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11336 | 2005-03-28 | 2026-05-22 | 11331 | 2005-03-28 | 2026-05-21 |  |
| 3285 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10860 | 2007-12-04 | 2026-05-22 | 10855 | 2007-12-04 | 2026-05-21 |  |
| 3287 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10046 | 2008-01-04 | 2026-05-22 | 10041 | 2008-01-04 | 2026-05-21 |  |
| 3288 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5933 | 2005-04-15 | 2026-05-22 | 5928 | 2005-04-15 | 2026-05-19 |  |
| 3289 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18243 | 2005-01-11 | 2026-05-22 | 18238 | 2005-01-11 | 2026-05-21 |  |
| 3290 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15125 | 2005-05-19 | 2026-05-22 | 15120 | 2005-05-19 | 2026-05-21 |  |
| 3293 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19344 | 2006-07-12 | 2026-05-22 | 19339 | 2006-07-12 | 2026-05-21 |  |
| 3294 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15602 | 2007-04-27 | 2026-05-22 | 15597 | 2007-04-27 | 2026-05-21 |  |
| 3296 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15287 | 2005-03-16 | 2026-05-22 | 15282 | 2005-03-16 | 2026-05-21 |  |
| 3297 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9632 | 2005-06-08 | 2026-05-22 | 9627 | 2005-06-08 | 2026-05-21 |  |
| 3303 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15218 | 2007-05-28 | 2026-05-22 | 15213 | 2007-05-28 | 2026-05-21 |  |
| 3305 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16198 | 2005-07-05 | 2026-05-22 | 16193 | 2005-07-05 | 2026-05-21 |  |
| 3306 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13013 | 2005-07-05 | 2026-05-22 | 13008 | 2005-07-05 | 2026-05-21 |  |
| 3308 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11827 | 2007-09-06 | 2026-05-22 | 11822 | 2007-09-06 | 2026-05-21 |  |
| 3310 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9783 | 2008-03-24 | 2026-05-22 | 9778 | 2008-03-24 | 2026-05-21 |  |
| 3311 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15002 | 2012-05-02 | 2026-05-22 | 14997 | 2012-05-02 | 2026-05-21 |  |
| 3312 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15227 | 2005-11-07 | 2026-05-22 | 15222 | 2005-11-07 | 2026-05-21 |  |
| 3313 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13209 | 2006-05-29 | 2026-05-22 | 13204 | 2006-05-29 | 2026-05-21 |  |
| 3317 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15746 | 2007-06-27 | 2026-05-22 | 15741 | 2007-06-27 | 2026-05-21 |  |
| 3321 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10618 | 2007-07-20 | 2026-05-22 | 10613 | 2007-07-20 | 2026-05-21 |  |
| 3322 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12369 | 2007-06-20 | 2026-05-22 | 12364 | 2007-06-20 | 2026-05-21 |  |
| 3323 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17491 | 2007-06-20 | 2026-05-22 | 17486 | 2007-06-20 | 2026-05-21 |  |
| 3324 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16958 | 2005-05-13 | 2026-05-22 | 16953 | 2005-05-13 | 2026-05-21 |  |
| 3325 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15918 | 2005-08-15 | 2026-05-22 | 15913 | 2005-08-15 | 2026-05-21 |  |
| 3332 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4916 | 2007-06-27 | 2026-05-22 | 4911 | 2007-06-27 | 2026-05-21 |  |
| 3338 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13759 | 2007-08-23 | 2026-05-22 | 13754 | 2007-08-23 | 2026-05-21 |  |
| 3339 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12339 | 2006-06-27 | 2026-05-22 | 12334 | 2006-06-27 | 2026-05-21 |  |
| 3346 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11999 | 2012-12-18 | 2026-05-22 | 11994 | 2012-12-18 | 2026-05-21 |  |
| 3349 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2145 | 2012-01-10 | 2026-05-22 | 2140 | 2012-01-10 | 2026-05-21 |  |
| 3354 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11548 | 2005-12-14 | 2026-05-22 | 11543 | 2005-12-14 | 2026-05-21 |  |
| 3356 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15092 | 2012-05-02 | 2026-05-22 | 15087 | 2012-05-02 | 2026-05-21 |  |
| 3357 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6471 | 2013-11-04 | 2026-05-22 | 6466 | 2013-11-04 | 2026-05-21 |  |
| 3360 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14111 | 2006-06-06 | 2026-05-22 | 14106 | 2006-06-06 | 2026-05-21 |  |
| 3362 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14312 | 2005-09-07 | 2026-05-22 | 14307 | 2005-09-07 | 2026-05-21 |  |
| 3363 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14336 | 2008-11-14 | 2026-05-22 | 14331 | 2008-11-14 | 2026-05-21 |  |
| 3372 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15037 | 2005-12-16 | 2026-05-22 | 15032 | 2005-12-16 | 2026-05-21 |  |
| 3373 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11865 | 2007-07-31 | 2026-05-22 | 11860 | 2007-07-31 | 2026-05-21 |  |
| 3374 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15100 | 2007-06-20 | 2026-05-22 | 15095 | 2007-06-20 | 2026-05-21 |  |
| 3376 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16657 | 2005-12-26 | 2026-05-22 | 16652 | 2005-12-26 | 2026-05-21 |  |
| 3379 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10433 | 2009-12-25 | 2026-05-22 | 10428 | 2009-12-25 | 2026-05-21 |  |
| 3380 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15133 | 2012-05-02 | 2026-05-22 | 15128 | 2012-05-02 | 2026-05-21 |  |
| 3388 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16003 | 2006-02-22 | 2026-05-22 | 15998 | 2006-02-22 | 2026-05-21 |  |
| 3390 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15067 | 2006-02-22 | 2026-05-22 | 15062 | 2006-02-22 | 2026-05-21 |  |
| 3402 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14272 | 2007-01-11 | 2026-05-22 | 14267 | 2007-01-11 | 2026-05-21 |  |
| 3406 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 3406 | TaiwanStockDividend | MISMATCH | 18 | 2005-06-01 | 2026-06-14 | 17 | 2005-06-01 | 2025-07-22 |  |
| 3413 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13480 | 2007-06-20 | 2026-05-22 | 13475 | 2007-06-20 | 2026-05-21 |  |
| 3416 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14548 | 2007-07-03 | 2026-05-22 | 14543 | 2007-07-03 | 2026-05-21 |  |
| 3419 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15912 | 2006-05-12 | 2026-05-22 | 15907 | 2006-05-12 | 2026-05-21 |  |
| 3430 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4839 | 2016-08-25 | 2026-05-22 | 4834 | 2016-08-25 | 2026-05-21 |  |
| 3432 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13115 | 2010-04-13 | 2026-05-22 | 13110 | 2010-04-13 | 2026-05-21 |  |
| 3434 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7677 | 2007-07-03 | 2026-05-22 | 7672 | 2007-07-03 | 2026-05-21 |  |
| 3437 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14446 | 2007-06-20 | 2026-05-22 | 14441 | 2007-06-20 | 2026-05-21 |  |
| 3438 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15772 | 2006-07-19 | 2026-05-22 | 15767 | 2006-07-19 | 2026-05-21 |  |
| 3441 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14369 | 2007-06-29 | 2026-05-22 | 14364 | 2007-06-29 | 2026-05-21 |  |
| 3443 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3444 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10946 | 2007-06-20 | 2026-05-22 | 10941 | 2007-06-20 | 2026-05-21 |  |
| 3447 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4251 | 2021-11-17 | 2026-05-22 | 4246 | 2021-11-17 | 2026-05-21 |  |
| 3450 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 3455 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16643 | 2007-06-28 | 2026-05-22 | 16638 | 2007-06-28 | 2026-05-21 |  |
| 3465 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6413 | 2008-01-21 | 2026-05-22 | 6408 | 2008-01-21 | 2026-05-21 |  |
| 3466 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9782 | 2006-05-17 | 2026-05-22 | 9777 | 2006-05-17 | 2026-05-21 |  |
| 3467 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1399 | 2023-12-22 | 2026-05-22 | 1394 | 2023-12-22 | 2026-05-21 |  |
| 3479 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15850 | 2007-10-08 | 2026-05-22 | 15845 | 2007-10-08 | 2026-05-21 |  |
| 3481 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15067 | 2012-05-02 | 2026-05-22 | 15062 | 2012-05-02 | 2026-05-21 |  |
| 3483 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16472 | 2006-12-18 | 2026-05-22 | 16467 | 2006-12-18 | 2026-05-21 |  |
| 3484 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15029 | 2007-09-27 | 2026-05-22 | 15024 | 2007-09-27 | 2026-05-21 |  |
| 3485 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 833 | 2015-02-04 | 2026-05-22 | 828 | 2015-02-04 | 2026-05-21 |  |
| 3489 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10376 | 2007-06-28 | 2026-05-22 | 10371 | 2007-06-28 | 2026-05-21 |  |
| 3490 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15179 | 2007-07-26 | 2026-05-22 | 15174 | 2007-07-26 | 2026-05-21 |  |
| 3491 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17700 | 2007-07-26 | 2026-05-22 | 17695 | 2007-07-26 | 2026-05-21 |  |
| 3492 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10149 | 2007-06-26 | 2026-05-22 | 10144 | 2007-06-26 | 2026-05-21 |  |
| 3494 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12764 | 2007-06-21 | 2026-05-22 | 12759 | 2007-06-21 | 2026-05-21 |  |
| 3498 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15459 | 2007-08-07 | 2026-05-22 | 15454 | 2007-08-07 | 2026-05-21 |  |
| 3499 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12712 | 2006-10-19 | 2026-05-22 | 12707 | 2006-10-19 | 2026-05-21 |  |
| 3501 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15134 | 2007-09-07 | 2026-05-22 | 15129 | 2007-09-07 | 2026-05-21 |  |
| 3504 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15153 | 2012-05-02 | 2026-05-22 | 15148 | 2012-05-02 | 2026-05-21 |  |
| 3508 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18108 | 2007-07-13 | 2026-05-22 | 18103 | 2007-07-13 | 2026-05-21 |  |
| 3511 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13007 | 2007-11-22 | 2026-05-22 | 13002 | 2007-11-22 | 2026-05-21 |  |
| 3512 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10611 | 2007-07-11 | 2026-05-22 | 10606 | 2007-07-11 | 2026-05-21 |  |
| 3515 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15031 | 2007-07-30 | 2026-05-22 | 15026 | 2007-07-30 | 2026-05-21 |  |
| 3516 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9763 | 2007-06-25 | 2026-05-22 | 9758 | 2007-06-25 | 2026-05-21 |  |
| 3518 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14872 | 2007-06-20 | 2026-05-22 | 14867 | 2007-06-20 | 2026-05-21 |  |
| 3520 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13194 | 2007-09-17 | 2026-05-22 | 13189 | 2007-09-17 | 2026-05-21 |  |
| 3521 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10041 | 2007-07-18 | 2026-05-22 | 10036 | 2007-07-18 | 2026-05-21 |  |
| 3522 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12853 | 2007-09-20 | 2026-05-22 | 12848 | 2007-09-20 | 2026-05-21 |  |
| 3523 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10075 | 2007-07-12 | 2026-05-22 | 10070 | 2007-07-12 | 2026-05-21 |  |
| 3526 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16693 | 2007-11-02 | 2026-05-22 | 16688 | 2007-11-02 | 2026-05-21 |  |
| 3527 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17987 | 2007-06-26 | 2026-05-22 | 17982 | 2007-06-26 | 2026-05-21 |  |
| 3528 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11571 | 2008-06-24 | 2026-05-22 | 11566 | 2008-06-24 | 2026-05-21 |  |
| 3529 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16208 | 2007-09-28 | 2026-05-22 | 16203 | 2007-09-28 | 2026-05-21 |  |
| 3530 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10245 | 2007-08-31 | 2026-05-22 | 10240 | 2007-08-31 | 2026-05-21 |  |
| 3532 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15193 | 2007-06-20 | 2026-05-22 | 15188 | 2007-06-20 | 2026-05-21 |  |
| 3533 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15171 | 2007-08-16 | 2026-05-22 | 15166 | 2007-08-16 | 2026-05-21 |  |
| 3535 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15056 | 2007-06-26 | 2026-05-22 | 15051 | 2007-06-26 | 2026-05-21 |  |
| 3537 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14232 | 2007-11-23 | 2026-05-22 | 14227 | 2007-11-23 | 2026-05-21 |  |
| 3540 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2007-12-12 | 2026-05-22 | 15167 | 2007-12-12 | 2026-05-21 |  |
| 3541 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14660 | 2007-06-20 | 2026-05-22 | 14655 | 2007-06-20 | 2026-05-21 |  |
| 3543 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7970 | 2007-06-25 | 2026-05-22 | 7965 | 2007-06-25 | 2026-05-21 |  |
| 3545 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15097 | 2012-05-02 | 2026-05-22 | 15092 | 2012-05-02 | 2026-05-21 |  |
| 3546 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16924 | 2007-07-19 | 2026-05-22 | 16919 | 2007-07-19 | 2026-05-21 |  |
| 3548 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15591 | 2007-07-26 | 2026-05-22 | 15586 | 2007-07-26 | 2026-05-21 |  |
| 3550 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13846 | 2008-03-21 | 2026-05-22 | 13841 | 2008-03-21 | 2026-05-21 |  |
| 3551 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15287 | 2007-06-21 | 2026-05-22 | 15282 | 2007-06-21 | 2026-05-21 |  |
| 3552 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17231 | 2007-11-27 | 2026-05-22 | 17226 | 2007-11-27 | 2026-05-21 |  |
| 3555 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7450 | 2007-06-20 | 2026-05-22 | 7445 | 2007-06-20 | 2026-05-21 |  |
| 3556 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17532 | 2007-09-20 | 2026-05-22 | 17527 | 2007-09-20 | 2026-05-21 |  |
| 3557 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14054 | 2007-07-02 | 2026-05-22 | 14049 | 2007-07-02 | 2026-05-21 |  |
| 3558 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14354 | 2007-07-09 | 2026-05-22 | 14349 | 2007-07-09 | 2026-05-21 |  |
| 3563 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14762 | 2007-06-28 | 2026-05-22 | 14757 | 2007-06-28 | 2026-05-21 |  |
| 3564 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9759 | 2007-08-08 | 2026-05-22 | 9754 | 2007-08-08 | 2026-05-21 |  |
| 3567 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12036 | 2008-02-20 | 2026-05-22 | 12031 | 2008-02-20 | 2026-05-21 |  |
| 3570 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8226 | 2008-08-15 | 2026-05-22 | 8221 | 2008-08-15 | 2026-05-21 |  |
| 3576 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15508 | 2007-10-24 | 2026-05-22 | 15503 | 2007-10-24 | 2026-05-21 |  |
| 3577 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10232 | 2008-06-18 | 2026-05-22 | 10227 | 2008-06-18 | 2026-05-21 |  |
| 3580 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13268 | 2007-12-06 | 2026-05-22 | 13263 | 2007-12-06 | 2026-05-21 |  |
| 3581 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9720 | 2007-11-27 | 2026-05-22 | 9715 | 2007-11-27 | 2026-05-21 |  |
| 3583 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14495 | 2013-01-14 | 2026-05-22 | 14490 | 2013-01-14 | 2026-05-21 |  |
| 3587 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2009-04-02 | 2026-05-22 | 15159 | 2009-04-02 | 2026-05-21 |  |
| 3588 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15201 | 2007-10-29 | 2026-05-22 | 15196 | 2007-10-29 | 2026-05-21 |  |
| 3591 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15476 | 2008-04-18 | 2026-05-22 | 15471 | 2008-04-18 | 2026-05-21 |  |
| 3592 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10455 | 2007-11-06 | 2026-05-22 | 10450 | 2007-11-06 | 2026-05-21 |  |
| 3594 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12707 | 2008-10-16 | 2026-05-22 | 12702 | 2008-10-16 | 2026-05-21 |  |
| 3596 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15528 | 2007-11-01 | 2026-05-22 | 15523 | 2007-11-01 | 2026-05-21 |  |
| 3597 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6281 | 2009-06-29 | 2026-05-22 | 6276 | 2009-06-29 | 2026-05-21 |  |
| 3605 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15242 | 2008-01-10 | 2026-05-22 | 15237 | 2008-01-10 | 2026-05-21 |  |
| 3607 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15181 | 2008-01-29 | 2026-05-22 | 15176 | 2008-01-29 | 2026-05-21 |  |
| 3609 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7945 | 2009-01-06 | 2026-05-22 | 7940 | 2009-01-06 | 2026-05-21 |  |
| 3611 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16524 | 2008-12-08 | 2026-05-22 | 16519 | 2008-12-08 | 2026-05-21 |  |
| 3615 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15281 | 2008-12-30 | 2026-05-22 | 15276 | 2008-12-30 | 2026-05-21 |  |
| 3617 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15170 | 2008-04-24 | 2026-05-22 | 15165 | 2008-04-24 | 2026-05-21 |  |
| 3622 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15401 | 2008-04-07 | 2026-05-22 | 15396 | 2008-04-07 | 2026-05-21 |  |
| 3623 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12693 | 2008-04-14 | 2026-05-22 | 12688 | 2008-04-14 | 2026-05-21 |  |
| 3624 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15671 | 2008-05-15 | 2026-05-22 | 15666 | 2008-05-15 | 2026-05-21 |  |
| 3625 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13465 | 2008-05-05 | 2026-05-22 | 13460 | 2008-05-05 | 2026-05-21 |  |
| 3628 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11908 | 2009-10-19 | 2026-05-22 | 11903 | 2009-10-19 | 2026-05-21 |  |
| 3630 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16699 | 2008-08-19 | 2026-05-22 | 16694 | 2008-08-19 | 2026-05-21 |  |
| 3631 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12018 | 2008-09-30 | 2026-05-22 | 12013 | 2008-09-30 | 2026-05-21 |  |
| 3645 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15276 | 2010-09-16 | 2026-05-22 | 15271 | 2010-09-16 | 2026-05-21 |  |
| 3652 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9222 | 2009-08-27 | 2026-05-22 | 9217 | 2009-08-27 | 2026-05-21 |  |
| 3653 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15261 | 2008-11-21 | 2026-05-22 | 15256 | 2008-11-21 | 2026-05-21 |  |
| 3661 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14054 | 2011-01-04 | 2026-05-22 | 14049 | 2011-01-04 | 2026-05-21 |  |
| 3663 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10576 | 2012-01-11 | 2026-05-22 | 10571 | 2012-01-11 | 2026-05-21 |  |
| 3665 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15125 | 2012-05-02 | 2026-05-22 | 15120 | 2012-05-02 | 2026-05-21 |  |
| 3666 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12083 | 2009-06-19 | 2026-05-22 | 12078 | 2009-06-19 | 2026-05-21 |  |
| 3669 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13206 | 2009-09-02 | 2026-05-22 | 13201 | 2009-09-02 | 2026-05-21 |  |
| 3672 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7420 | 2014-01-13 | 2026-05-22 | 7415 | 2014-01-13 | 2026-05-21 |  |
| 3673 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 3675 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11691 | 2009-09-09 | 2026-05-22 | 11686 | 2009-09-09 | 2026-05-21 |  |
| 3679 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14386 | 2009-12-30 | 2026-05-22 | 14381 | 2009-12-30 | 2026-05-21 |  |
| 3680 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14519 | 2009-10-16 | 2026-05-22 | 14514 | 2009-10-16 | 2026-05-21 |  |
| 3684 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5261 | 2009-12-31 | 2026-05-22 | 5256 | 2009-12-31 | 2026-05-21 |  |
| 3685 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11203 | 2009-11-27 | 2026-05-22 | 11198 | 2009-11-27 | 2026-05-21 |  |
| 3686 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14081 | 2009-11-23 | 2026-05-22 | 14076 | 2009-11-23 | 2026-05-21 |  |
| 3687 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14547 | 2010-09-06 | 2026-05-22 | 14542 | 2010-09-06 | 2026-05-21 |  |
| 3689 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14648 | 2010-12-16 | 2026-05-22 | 14643 | 2010-12-16 | 2026-05-21 |  |
| 3691 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16454 | 2009-10-21 | 2026-05-22 | 16449 | 2009-10-21 | 2026-05-21 |  |
| 3693 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12153 | 2010-12-30 | 2026-05-22 | 12148 | 2010-12-30 | 2026-05-21 |  |
| 3694 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15017 | 2009-12-10 | 2026-05-22 | 15012 | 2009-12-10 | 2026-05-21 |  |
| 3701 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8705 | 2012-05-02 | 2026-05-22 | 8700 | 2012-05-02 | 2026-05-21 |  |
| 3702 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15137 | 2012-05-02 | 2026-05-22 | 15132 | 2012-05-02 | 2026-05-21 |  |
| 3703 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15126 | 2012-05-02 | 2026-05-22 | 15121 | 2012-05-02 | 2026-05-21 |  |
| 3704 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15129 | 2012-05-02 | 2026-05-22 | 15124 | 2012-05-02 | 2026-05-21 |  |
| 3705 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15114 | 2012-05-02 | 2026-05-22 | 15109 | 2012-05-02 | 2026-05-21 |  |
| 3706 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14149 | 2013-05-16 | 2026-05-22 | 14144 | 2013-05-16 | 2026-05-21 |  |
| 3707 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12792 | 2014-10-01 | 2026-05-22 | 12787 | 2014-10-01 | 2026-05-21 |  |
| 3708 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11525 | 2016-02-05 | 2026-05-22 | 11520 | 2016-02-05 | 2026-05-21 |  |
| 3709 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7338 | 2017-09-01 | 2026-05-22 | 7333 | 2017-09-01 | 2026-05-21 |  |
| 3710 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9200 | 2018-01-03 | 2026-05-22 | 9195 | 2018-01-03 | 2026-05-21 |  |
| 3711 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9815 | 2018-04-30 | 2026-05-22 | 9810 | 2018-04-30 | 2026-05-21 |  |
| 3712 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9260 | 2018-10-01 | 2026-05-22 | 9255 | 2018-10-01 | 2026-05-21 |  |
| 3713 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6115 | 2020-03-02 | 2026-05-22 | 6110 | 2020-03-02 | 2026-05-21 |  |
| 3714 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6510 | 2021-01-06 | 2026-05-22 | 6505 | 2021-01-06 | 2026-05-21 |  |
| 3715 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4520 | 2022-08-25 | 2026-05-22 | 4515 | 2022-08-25 | 2026-05-21 |  |
| 3716 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2075 | 2024-09-02 | 2026-05-22 | 2070 | 2024-09-02 | 2026-05-21 |  |
| 3717 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 920 | 2025-08-15 | 2026-05-22 | 915 | 2025-08-15 | 2026-05-21 |  |
| 4102 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12411 | 2005-11-07 | 2026-05-22 | 12406 | 2005-11-07 | 2026-05-21 |  |
| 4104 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16091 | 2005-02-03 | 2026-05-22 | 16086 | 2005-02-03 | 2026-05-21 |  |
| 4105 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18881 | 2005-03-24 | 2026-05-22 | 18876 | 2005-03-24 | 2026-05-21 |  |
| 4106 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14716 | 2012-05-02 | 2026-05-22 | 14711 | 2012-05-02 | 2026-05-21 |  |
| 4107 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17894 | 2005-04-13 | 2026-05-22 | 17889 | 2005-04-13 | 2026-05-21 |  |
| 4108 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15985 | 2005-01-07 | 2026-05-22 | 15980 | 2005-01-07 | 2026-05-21 |  |
| 4109 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13963 | 2006-01-25 | 2026-05-22 | 13958 | 2006-01-25 | 2026-05-21 |  |
| 4111 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13017 | 2005-01-10 | 2026-05-22 | 13012 | 2005-01-10 | 2026-05-21 |  |
| 4113 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14200 | 2006-12-08 | 2026-05-22 | 14195 | 2006-12-08 | 2026-05-21 |  |
| 4114 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17576 | 2005-09-29 | 2026-05-22 | 17571 | 2005-09-29 | 2026-05-21 |  |
| 4116 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8964 | 2008-06-26 | 2026-05-22 | 8959 | 2008-06-26 | 2026-05-21 |  |
| 4119 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15099 | 2012-05-02 | 2026-05-22 | 15094 | 2012-05-02 | 2026-05-21 |  |
| 4120 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17123 | 2005-06-17 | 2026-05-22 | 17118 | 2005-06-17 | 2026-05-21 |  |
| 4121 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13074 | 2005-06-03 | 2026-05-22 | 13069 | 2005-06-03 | 2026-05-21 |  |
| 4123 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17268 | 2005-07-11 | 2026-05-22 | 17263 | 2005-07-11 | 2026-05-21 |  |
| 4126 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18589 | 2005-01-10 | 2026-05-22 | 18584 | 2005-01-10 | 2026-05-21 |  |
| 4127 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12656 | 2005-08-01 | 2026-05-22 | 12651 | 2005-08-01 | 2026-05-20 |  |
| 4128 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17922 | 2006-06-09 | 2026-05-22 | 17917 | 2006-06-09 | 2026-05-21 |  |
| 4129 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17406 | 2005-01-04 | 2026-05-22 | 17401 | 2005-01-04 | 2026-05-21 |  |
| 4130 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15273 | 2007-06-20 | 2026-05-22 | 15268 | 2007-06-20 | 2026-05-21 |  |
| 4131 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4368 | 2007-06-28 | 2026-05-22 | 4363 | 2007-06-28 | 2026-05-21 |  |
| 4133 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14042 | 2008-05-29 | 2026-05-22 | 14037 | 2008-05-29 | 2026-05-21 |  |
| 4137 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13996 | 2013-05-16 | 2026-05-22 | 13991 | 2013-05-16 | 2026-05-21 |  |
| 4138 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11756 | 2010-01-20 | 2026-05-22 | 11751 | 2010-01-20 | 2026-05-21 |  |
| 4139 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7970 | 2009-11-23 | 2026-05-22 | 7965 | 2009-11-23 | 2026-05-21 |  |
| 4142 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15298 | 2010-09-30 | 2026-05-22 | 15293 | 2010-09-30 | 2026-05-21 |  |
| 4147 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14613 | 2010-06-28 | 2026-05-22 | 14608 | 2010-06-28 | 2026-05-21 |  |
| 4148 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10467 | 2016-02-05 | 2026-05-22 | 10462 | 2016-02-05 | 2026-05-21 |  |
| 4153 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12053 | 2010-12-30 | 2026-05-22 | 12048 | 2010-12-30 | 2026-05-21 |  |
| 4154 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5837 | 2011-12-28 | 2026-05-22 | 5832 | 2011-12-28 | 2026-05-21 |  |
| 4155 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9759 | 2011-05-23 | 2026-05-22 | 9754 | 2011-05-23 | 2026-05-21 |  |
| 4157 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14052 | 2013-08-30 | 2026-05-22 | 14047 | 2013-08-30 | 2026-05-21 |  |
| 4160 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7911 | 2011-11-03 | 2026-05-22 | 7906 | 2011-11-03 | 2026-05-21 |  |
| 4161 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11919 | 2013-04-08 | 2026-05-22 | 11914 | 2013-04-08 | 2026-05-21 |  |
| 4162 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15089 | 2011-10-05 | 2026-05-22 | 15084 | 2011-10-05 | 2026-05-21 |  |
| 4163 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14164 | 2012-02-29 | 2026-05-22 | 14159 | 2012-02-29 | 2026-05-21 |  |
| 4164 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15152 | 2011-10-28 | 2026-05-22 | 15147 | 2011-10-28 | 2026-05-21 |  |
| 4166 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2805 | 2012-11-01 | 2026-05-22 | 2800 | 2012-11-01 | 2026-05-21 |  |
| 4167 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12937 | 2012-12-21 | 2026-05-22 | 12932 | 2012-12-21 | 2026-05-21 |  |
| 4168 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13126 | 2012-02-23 | 2026-05-22 | 13121 | 2012-02-23 | 2026-05-21 |  |
| 4169 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1372 | 2012-05-11 | 2026-05-22 | 1367 | 2012-05-11 | 2026-05-21 |  |
| 4171 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10790 | 2013-08-14 | 2026-05-22 | 10785 | 2013-08-14 | 2026-05-21 |  |
| 4173 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11651 | 2013-10-31 | 2026-05-22 | 11646 | 2013-10-31 | 2026-05-21 |  |
| 4174 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14057 | 2012-12-12 | 2026-05-22 | 14052 | 2012-12-12 | 2026-05-21 |  |
| 4175 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8453 | 2013-01-31 | 2026-05-22 | 8448 | 2013-01-31 | 2026-05-21 |  |
| 4178 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 171 | 2024-12-05 | 2026-05-22 | 166 | 2024-12-05 | 2026-05-21 |  |
| 4188 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7742 | 2013-12-31 | 2026-05-22 | 7737 | 2013-12-31 | 2026-05-21 |  |
| 4188 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=4188&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4188 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=4188&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4188 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=4188&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4190 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=4190&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4192 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=4192&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4194 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=4194&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4195 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=4195&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4197 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=4197&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4198 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=4198&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4198 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=4198&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4198 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=4198&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4198 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=4198&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4198 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=4198&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 4205 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7201 | 2007-07-12 | 2026-05-22 | 7196 | 2007-07-12 | 2026-05-20 |  |
| 4207 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12251 | 2006-10-12 | 2026-05-22 | 12246 | 2006-10-12 | 2026-05-21 |  |
| 4303 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9963 | 2005-03-22 | 2026-05-22 | 9958 | 2005-03-22 | 2026-05-21 |  |
| 4304 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3032 | 2008-08-01 | 2026-05-22 | 3027 | 2008-08-01 | 2026-05-20 |  |
| 4305 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3135 | 2006-08-29 | 2026-05-22 | 3130 | 2006-08-29 | 2026-05-19 |  |
| 4306 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15496 | 2005-01-03 | 2026-05-22 | 15491 | 2005-01-03 | 2026-05-21 |  |
| 4401 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16200 | 2006-09-21 | 2026-05-22 | 16195 | 2006-09-21 | 2026-05-21 |  |
| 4402 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8716 | 2006-12-27 | 2026-05-22 | 8711 | 2006-12-27 | 2026-05-21 |  |
| 4413 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4417 | 2005-02-24 | 2026-05-22 | 4412 | 2005-02-24 | 2026-05-06 |  |
| 4414 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13812 | 2012-05-02 | 2026-05-22 | 13807 | 2012-05-02 | 2026-05-21 |  |
| 4416 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4093 | 2012-09-05 | 2026-05-22 | 4088 | 2012-09-05 | 2026-05-21 |  |
| 4417 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14073 | 2005-01-12 | 2026-05-22 | 14068 | 2005-01-12 | 2026-05-21 |  |
| 4419 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1648 | 2007-01-23 | 2026-05-22 | 1643 | 2007-01-23 | 2026-05-21 |  |
| 4420 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8159 | 2007-05-28 | 2026-05-22 | 8154 | 2007-05-28 | 2026-05-21 |  |
| 4426 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15151 | 2006-11-13 | 2026-05-22 | 15146 | 2006-11-13 | 2026-05-21 |  |
| 4430 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7930 | 2009-11-02 | 2026-05-22 | 7925 | 2009-11-02 | 2026-05-19 |  |
| 4432 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12106 | 2012-03-01 | 2026-05-22 | 12101 | 2012-03-01 | 2026-05-21 |  |
| 4438 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11694 | 2015-08-03 | 2026-05-22 | 11689 | 2015-08-03 | 2026-05-21 |  |
| 4439 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6885 | 2019-12-05 | 2026-05-22 | 6880 | 2019-12-05 | 2026-05-21 |  |
| 4440 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5499 | 2021-08-10 | 2026-05-22 | 5494 | 2021-08-10 | 2026-05-21 |  |
| 4441 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1870 | 2022-02-16 | 2026-05-22 | 1865 | 2022-02-16 | 2026-05-21 |  |
| 4442 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2130 | 2023-12-04 | 2026-05-22 | 2125 | 2023-12-04 | 2026-05-21 |  |
| 4502 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8856 | 2005-01-07 | 2026-05-22 | 8851 | 2005-01-07 | 2026-05-21 |  |
| 4503 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9844 | 2005-01-13 | 2026-05-22 | 9839 | 2005-01-13 | 2026-05-21 |  |
| 4506 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14160 | 2005-08-30 | 2026-05-22 | 14155 | 2005-08-30 | 2026-05-21 |  |
| 4510 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12328 | 2005-01-25 | 2026-05-22 | 12323 | 2005-01-25 | 2026-05-21 |  |
| 4513 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11310 | 2005-01-28 | 2026-05-22 | 11305 | 2005-01-28 | 2026-05-21 |  |
| 4523 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12477 | 2005-01-07 | 2026-05-22 | 12472 | 2005-01-07 | 2026-05-21 |  |
| 4526 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15063 | 2012-05-02 | 2026-05-22 | 15058 | 2012-05-02 | 2026-05-21 |  |
| 4527 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8936 | 2005-02-17 | 2026-05-22 | 8931 | 2005-02-17 | 2026-05-20 |  |
| 4528 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14762 | 2005-05-03 | 2026-05-22 | 14757 | 2005-05-03 | 2026-05-21 |  |
| 4530 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5494 | 2005-01-13 | 2026-05-22 | 5489 | 2005-01-13 | 2026-05-21 |  |
| 4532 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 4533 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14307 | 2005-04-22 | 2026-05-22 | 14302 | 2005-04-22 | 2026-05-21 |  |
| 4534 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11600 | 2005-01-04 | 2026-05-22 | 11595 | 2005-01-04 | 2026-05-21 |  |
| 4536 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14024 | 2012-03-01 | 2026-05-22 | 14019 | 2012-03-01 | 2026-05-21 |  |
| 4538 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4738 | 2016-05-04 | 2026-05-22 | 4733 | 2016-05-04 | 2026-05-20 |  |
| 4540 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9623 | 2014-12-30 | 2026-05-22 | 9618 | 2014-12-30 | 2026-05-21 |  |
| 4541 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11388 | 2013-07-31 | 2026-05-22 | 11383 | 2013-07-31 | 2026-05-21 |  |
| 4542 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6055 | 2014-01-20 | 2026-05-22 | 6050 | 2014-01-20 | 2026-05-21 |  |
| 4543 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7188 | 2016-03-15 | 2026-05-22 | 7183 | 2016-03-15 | 2026-05-21 |  |
| 4545 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11447 | 2013-07-22 | 2026-05-22 | 11442 | 2013-07-22 | 2026-05-21 |  |
| 4549 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12575 | 2014-01-02 | 2026-05-22 | 12570 | 2014-01-02 | 2026-05-21 |  |
| 4550 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7606 | 2014-04-18 | 2026-05-22 | 7601 | 2014-04-18 | 2026-05-21 |  |
| 4551 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12612 | 2014-03-04 | 2026-05-22 | 12607 | 2014-03-04 | 2026-05-21 |  |
| 4552 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11584 | 2016-02-05 | 2026-05-22 | 11579 | 2016-02-05 | 2026-05-21 |  |
| 4554 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6167 | 2014-09-09 | 2026-05-22 | 6162 | 2014-09-09 | 2026-05-21 |  |
| 4555 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10445 | 2014-09-23 | 2026-05-22 | 10440 | 2014-09-23 | 2026-05-21 |  |
| 4556 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4002 | 2015-12-22 | 2026-05-22 | 3997 | 2015-12-22 | 2026-05-21 |  |
| 4557 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11203 | 2015-10-12 | 2026-05-22 | 11198 | 2015-10-12 | 2026-05-21 |  |
| 4558 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4061 | 2022-02-11 | 2026-05-22 | 4056 | 2022-02-11 | 2026-05-21 |  |
| 4560 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10686 | 2016-02-05 | 2026-05-22 | 10681 | 2016-02-05 | 2026-05-21 |  |
| 4561 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7240 | 2017-10-18 | 2026-05-22 | 7235 | 2017-10-18 | 2026-05-21 |  |
| 4562 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8560 | 2016-01-25 | 2026-05-22 | 8555 | 2016-01-25 | 2026-05-21 |  |
| 4563 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7423 | 2016-09-29 | 2026-05-22 | 7418 | 2016-09-29 | 2026-05-21 |  |
| 4564 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9094 | 2017-02-06 | 2026-05-22 | 9089 | 2017-02-06 | 2026-05-21 |  |
| 4566 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9560 | 2017-03-15 | 2026-05-22 | 9555 | 2017-03-15 | 2026-05-21 |  |
| 4569 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2995 | 2023-08-01 | 2026-05-22 | 2990 | 2023-08-01 | 2026-05-21 |  |
| 4571 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7490 | 2019-09-17 | 2026-05-22 | 7485 | 2019-09-17 | 2026-05-21 |  |
| 4572 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8034 | 2018-08-03 | 2026-05-22 | 8029 | 2018-08-03 | 2026-05-21 |  |
| 4576 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8056 | 2019-03-15 | 2026-05-22 | 8051 | 2019-03-15 | 2026-05-21 |  |
| 4577 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4525 | 2020-11-18 | 2026-05-22 | 4520 | 2020-11-18 | 2026-05-21 |  |
| 4580 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7280 | 2019-03-18 | 2026-05-22 | 7275 | 2019-03-18 | 2026-05-21 |  |
| 4581 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5820 | 2020-03-10 | 2026-05-22 | 5815 | 2020-03-10 | 2026-05-21 |  |
| 4582 | TaiwanStockPER | MISMATCH | 1 | 2026-05-22 | 2026-05-22 | 0 | None | None |  |
| 4582 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1130 | 2020-01-17 | 2026-05-22 | 1125 | 2020-01-17 | 2026-05-20 |  |
| 4582 | TaiwanStockShareholding | MISMATCH | 1 | 2026-05-21 | 2026-05-21 | 0 | None | None |  |
| 4583 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4706 | 2021-01-25 | 2026-05-22 | 4701 | 2021-01-25 | 2026-05-21 |  |
| 4585 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 929 | 2024-09-26 | 2026-05-22 | 924 | 2024-09-26 | 2026-05-21 |  |
| 4588 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3138 | 2022-12-30 | 2026-05-22 | 3133 | 2022-12-30 | 2026-05-21 |  |
| 4590 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 562 | 2024-11-12 | 2026-05-22 | 557 | 2024-11-12 | 2026-05-21 |  |
| 4702 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11399 | 2005-04-06 | 2026-05-22 | 11394 | 2005-04-06 | 2026-05-21 |  |
| 4706 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5749 | 2005-10-17 | 2026-05-22 | 5744 | 2005-10-17 | 2026-05-21 |  |
| 4707 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13980 | 2006-03-29 | 2026-05-22 | 13975 | 2006-03-29 | 2026-05-21 |  |
| 4711 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7373 | 2007-04-26 | 2026-05-22 | 7368 | 2007-04-26 | 2026-05-21 |  |
| 4714 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13392 | 2006-07-20 | 2026-05-22 | 13387 | 2006-07-20 | 2026-05-21 |  |
| 4716 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13243 | 2005-08-15 | 2026-05-22 | 13238 | 2005-08-15 | 2026-05-21 |  |
| 4720 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11279 | 2007-06-21 | 2026-05-22 | 11274 | 2007-06-21 | 2026-05-21 |  |
| 4721 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15766 | 2005-01-03 | 2026-05-22 | 15761 | 2005-01-03 | 2026-05-21 |  |
| 4722 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15871 | 2006-12-18 | 2026-05-22 | 15866 | 2006-12-18 | 2026-05-21 |  |
| 4726 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13107 | 2008-11-04 | 2026-05-22 | 13102 | 2008-11-04 | 2026-05-21 |  |
| 4729 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13051 | 2007-10-02 | 2026-05-22 | 13046 | 2007-10-02 | 2026-05-21 |  |
| 4735 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13162 | 2010-04-27 | 2026-05-22 | 13157 | 2010-04-27 | 2026-05-21 |  |
| 4736 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16118 | 2008-09-26 | 2026-05-22 | 16113 | 2008-09-26 | 2026-05-21 |  |
| 4737 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13186 | 2009-09-16 | 2026-05-22 | 13181 | 2009-09-16 | 2026-05-21 |  |
| 4739 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13749 | 2011-01-03 | 2026-05-22 | 13744 | 2011-01-03 | 2026-05-21 |  |
| 4741 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5850 | 2010-06-02 | 2026-05-22 | 5845 | 2010-06-02 | 2026-05-21 |  |
| 4743 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15244 | 2009-10-02 | 2026-05-22 | 15239 | 2009-10-02 | 2026-05-21 |  |
| 4744 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8642 | 2014-03-06 | 2026-05-22 | 8637 | 2014-03-06 | 2026-05-21 |  |
| 4745 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13367 | 2009-05-25 | 2026-05-22 | 13362 | 2009-05-25 | 2026-05-21 |  |
| 4746 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15243 | 2009-12-03 | 2026-05-22 | 15238 | 2009-12-03 | 2026-05-21 |  |
| 4747 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6874 | 2012-07-27 | 2026-05-22 | 6869 | 2012-07-27 | 2026-05-21 |  |
| 4749 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3113 | 2010-09-14 | 2026-05-22 | 3108 | 2010-09-14 | 2026-05-21 |  |
| 4754 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4574 | 2016-09-29 | 2026-05-22 | 4569 | 2016-09-29 | 2026-05-21 |  |
| 4755 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12070 | 2013-07-23 | 2026-05-22 | 12065 | 2013-07-23 | 2026-05-21 |  |
| 4760 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8431 | 2015-11-09 | 2026-05-22 | 8426 | 2015-11-09 | 2026-05-21 |  |
| 4763 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12264 | 2015-11-09 | 2026-05-22 | 12259 | 2015-11-09 | 2026-05-21 |  |
| 4764 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8801 | 2017-04-06 | 2026-05-22 | 8796 | 2017-04-06 | 2026-05-21 |  |
| 4766 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9425 | 2017-07-21 | 2026-05-22 | 9420 | 2017-07-21 | 2026-05-21 |  |
| 4767 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2415 | 2018-10-29 | 2026-05-22 | 2410 | 2018-10-29 | 2026-05-19 |  |
| 4768 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5147 | 2019-11-28 | 2026-05-22 | 5142 | 2019-11-28 | 2026-05-21 |  |
| 4770 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5566 | 2021-02-22 | 2026-05-22 | 5561 | 2021-02-22 | 2026-05-21 |  |
| 4771 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3257 | 2021-11-26 | 2026-05-22 | 3252 | 2021-11-26 | 2026-05-21 |  |
| 4772 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2557 | 2022-11-23 | 2026-05-22 | 2552 | 2022-11-23 | 2026-05-21 |  |
| 4806 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3588 | 2014-12-08 | 2026-05-22 | 3583 | 2014-12-08 | 2026-05-21 |  |
| 4807 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8175 | 2016-02-05 | 2026-05-22 | 8170 | 2016-02-05 | 2026-05-21 |  |
| 4903 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12407 | 2006-03-16 | 2026-05-22 | 12402 | 2006-03-16 | 2026-05-21 |  |
| 4904 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15636 | 2005-01-03 | 2026-05-22 | 15631 | 2005-01-03 | 2026-05-21 |  |
| 4905 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8070 | 2005-03-30 | 2026-05-22 | 8065 | 2005-03-30 | 2026-05-21 |  |
| 4906 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 4907 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8009 | 2005-03-08 | 2026-05-22 | 8004 | 2005-03-08 | 2026-05-21 |  |
| 4908 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16200 | 2005-12-21 | 2026-05-22 | 16195 | 2005-12-21 | 2026-05-21 |  |
| 4909 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18022 | 2005-01-18 | 2026-05-22 | 18017 | 2005-01-18 | 2026-05-21 |  |
| 4911 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9536 | 2008-05-09 | 2026-05-22 | 9531 | 2008-05-09 | 2026-05-21 |  |
| 4912 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15004 | 2011-04-29 | 2026-05-22 | 14999 | 2011-04-29 | 2026-05-21 |  |
| 4915 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15361 | 2009-12-31 | 2026-05-22 | 15356 | 2009-12-31 | 2026-05-21 |  |
| 4916 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14292 | 2010-01-06 | 2026-05-22 | 14287 | 2010-01-06 | 2026-05-21 |  |
| 4919 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15185 | 2010-05-18 | 2026-05-22 | 15180 | 2010-05-18 | 2026-05-21 |  |
| 4923 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5407 | 2010-06-30 | 2026-05-22 | 5402 | 2010-06-30 | 2026-05-21 |  |
| 4924 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10868 | 2010-03-12 | 2026-05-22 | 10863 | 2010-03-12 | 2026-05-21 |  |
| 4927 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15484 | 2010-03-05 | 2026-05-22 | 15479 | 2010-03-05 | 2026-05-21 |  |
| 4930 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14161 | 2012-05-02 | 2026-05-22 | 14156 | 2012-05-02 | 2026-05-21 |  |
| 4931 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8594 | 2010-04-02 | 2026-05-22 | 8589 | 2010-04-02 | 2026-05-21 |  |
| 4933 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14499 | 2010-05-10 | 2026-05-22 | 14494 | 2010-05-10 | 2026-05-21 |  |
| 4934 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15155 | 2010-04-23 | 2026-05-22 | 15150 | 2010-04-23 | 2026-05-21 |  |
| 4935 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 4938 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 4939 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14512 | 2010-07-01 | 2026-05-22 | 14507 | 2010-07-01 | 2026-05-21 |  |
| 4942 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14785 | 2010-05-06 | 2026-05-22 | 14780 | 2010-05-06 | 2026-05-21 |  |
| 4946 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10323 | 2010-06-29 | 2026-05-22 | 10318 | 2010-06-29 | 2026-05-21 |  |
| 4949 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | ReadTimeout: HTTPSConnectionPool(host='api.finmindtrade.com', port=443): Read timed out. (read timeout=30) |
| 4949 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3122 | 2010-07-22 | 2026-05-22 | 3117 | 2010-07-22 | 2026-05-21 |  |
| 4951 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4569 | 2010-09-28 | 2026-05-22 | 4564 | 2010-09-28 | 2026-05-21 |  |
| 4952 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15153 | 2010-08-17 | 2026-05-22 | 15148 | 2010-08-17 | 2026-05-21 |  |
| 4953 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12141 | 2010-11-19 | 2026-05-22 | 12136 | 2010-11-19 | 2026-05-21 |  |
| 4956 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14822 | 2010-12-01 | 2026-05-22 | 14817 | 2010-12-01 | 2026-05-21 |  |
| 4958 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-22 | 15162 | 2012-05-02 | 2026-05-21 |  |
| 4960 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15228 | 2010-09-01 | 2026-05-22 | 15223 | 2010-09-01 | 2026-05-21 |  |
| 4961 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9812 | 2010-11-30 | 2026-05-22 | 9807 | 2010-11-30 | 2026-05-21 |  |
| 4966 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15504 | 2011-09-13 | 2026-05-22 | 15499 | 2011-09-13 | 2026-05-21 |  |
| 4967 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8861 | 2010-10-13 | 2026-05-22 | 8856 | 2010-10-13 | 2026-05-21 |  |
| 4968 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12489 | 2010-10-01 | 2026-05-22 | 12484 | 2010-10-01 | 2026-05-21 |  |
| 4971 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13980 | 2013-07-24 | 2026-05-22 | 13975 | 2013-07-24 | 2026-05-21 |  |
| 4973 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12332 | 2012-06-19 | 2026-05-22 | 12327 | 2012-06-19 | 2026-05-21 |  |
| 4974 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14773 | 2010-12-08 | 2026-05-22 | 14768 | 2010-12-08 | 2026-05-21 |  |
| 4976 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14241 | 2011-03-02 | 2026-05-22 | 14236 | 2011-03-02 | 2026-05-21 |  |
| 4977 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14301 | 2012-12-26 | 2026-05-22 | 14296 | 2012-12-26 | 2026-05-21 |  |
| 4979 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15491 | 2010-11-19 | 2026-05-22 | 15486 | 2010-11-19 | 2026-05-21 |  |
| 4989 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9925 | 2011-01-28 | 2026-05-22 | 9920 | 2011-01-28 | 2026-05-21 |  |
| 4991 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13205 | 2014-02-25 | 2026-05-22 | 13200 | 2014-02-25 | 2026-05-21 |  |
| 4994 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14715 | 2011-12-13 | 2026-05-22 | 14710 | 2011-12-13 | 2026-05-21 |  |
| 4995 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8283 | 2011-01-05 | 2026-05-22 | 8278 | 2011-01-05 | 2026-05-21 |  |
| 4999 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14397 | 2012-11-19 | 2026-05-22 | 14392 | 2012-11-19 | 2026-05-21 |  |
| 5007 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18225 | 2005-01-03 | 2026-05-22 | 18220 | 2005-01-03 | 2026-05-21 |  |
| 5009 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20443 | 2005-01-03 | 2026-05-22 | 20438 | 2005-01-03 | 2026-05-21 |  |
| 5011 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7723 | 2012-11-27 | 2026-05-22 | 7718 | 2012-11-27 | 2026-05-21 |  |
| 5013 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9770 | 2005-06-20 | 2026-05-22 | 9765 | 2005-06-20 | 2026-05-21 |  |
| 5014 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12741 | 2005-01-10 | 2026-05-22 | 12736 | 2005-01-10 | 2026-05-21 |  |
| 5015 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11516 | 2005-01-19 | 2026-05-22 | 11511 | 2005-01-19 | 2026-05-21 |  |
| 5016 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5620 | 2005-01-27 | 2026-05-22 | 5615 | 2005-01-27 | 2026-05-21 |  |
| 5201 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10300 | 2005-04-22 | 2026-05-22 | 10295 | 2005-04-22 | 2026-05-21 |  |
| 5202 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8893 | 2006-08-02 | 2026-05-22 | 8888 | 2006-08-02 | 2026-05-21 |  |
| 5203 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15117 | 2012-05-02 | 2026-05-22 | 15112 | 2012-05-02 | 2026-05-21 |  |
| 5206 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9810 | 2006-01-02 | 2026-05-22 | 9805 | 2006-01-02 | 2026-05-21 |  |
| 5209 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6908 | 2006-04-18 | 2026-05-22 | 6903 | 2006-04-18 | 2026-05-21 |  |
| 5210 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9929 | 2005-01-28 | 2026-05-22 | 9924 | 2005-01-28 | 2026-05-21 |  |
| 5211 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12002 | 2005-01-04 | 2026-05-22 | 11997 | 2005-01-04 | 2026-05-21 |  |
| 5212 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8783 | 2005-03-30 | 2026-05-22 | 8778 | 2005-03-30 | 2026-05-21 |  |
| 5213 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15898 | 2006-04-17 | 2026-05-22 | 15893 | 2006-04-17 | 2026-05-21 |  |
| 5213 | TaiwanStockDividend | MISMATCH | 30 | 2008-08-03 | 2026-06-14 | 29 | 2008-08-03 | 2025-09-30 |  |
| 5215 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13881 | 2012-05-02 | 2026-05-22 | 13876 | 2012-05-02 | 2026-05-21 |  |
| 5220 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8675 | 2014-10-16 | 2026-05-22 | 8670 | 2014-10-16 | 2026-05-21 |  |
| 5222 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6079 | 2016-11-09 | 2026-05-22 | 6074 | 2016-11-09 | 2026-05-21 |  |
| 5223 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8160 | 2018-07-02 | 2026-05-22 | 8155 | 2018-07-02 | 2026-05-21 |  |
| 5225 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13517 | 2012-11-05 | 2026-05-22 | 13512 | 2012-11-05 | 2026-05-21 |  |
| 5227 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13408 | 2013-12-09 | 2026-05-22 | 13403 | 2013-12-09 | 2026-05-21 |  |
| 5228 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3594 | 2018-04-18 | 2026-05-22 | 3589 | 2018-04-18 | 2026-05-21 |  |
| 5230 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13201 | 2011-05-03 | 2026-05-22 | 13196 | 2011-05-03 | 2026-05-21 |  |
| 5234 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14869 | 2011-05-18 | 2026-05-22 | 14864 | 2011-05-18 | 2026-05-21 |  |
| 5236 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6164 | 2020-10-06 | 2026-05-22 | 6159 | 2020-10-06 | 2026-05-21 |  |
| 5240 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 516 | 2012-01-18 | 2026-05-22 | 513 | 2012-01-18 | 2026-05-14 |  |
| 5243 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13979 | 2013-05-16 | 2026-05-22 | 13974 | 2013-05-16 | 2026-05-21 |  |
| 5244 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5681 | 2012-02-07 | 2026-05-22 | 5676 | 2012-02-07 | 2026-05-21 |  |
| 5245 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11640 | 2014-08-14 | 2026-05-22 | 11635 | 2014-08-14 | 2026-05-21 |  |
| 5251 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9720 | 2012-11-26 | 2026-05-22 | 9715 | 2012-11-26 | 2026-05-21 |  |
| 5258 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10767 | 2014-03-07 | 2026-05-22 | 10762 | 2014-03-07 | 2026-05-21 |  |
| 5263 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14203 | 2012-10-09 | 2026-05-22 | 14198 | 2012-10-09 | 2026-05-21 |  |
| 5269 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14607 | 2012-01-17 | 2026-05-22 | 14602 | 2012-01-17 | 2026-05-21 |  |
| 5272 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13585 | 2012-06-27 | 2026-05-22 | 13580 | 2012-06-27 | 2026-05-21 |  |
| 5274 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14470 | 2012-06-19 | 2026-05-22 | 14465 | 2012-06-19 | 2026-05-21 |  |
| 5276 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2743 | 2014-06-12 | 2026-05-22 | 2738 | 2014-06-12 | 2026-05-21 |  |
| 5278 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8561 | 2012-07-06 | 2026-05-22 | 8556 | 2012-07-06 | 2026-05-21 |  |
| 5283 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8617 | 2012-10-12 | 2026-05-22 | 8612 | 2012-10-12 | 2026-05-21 |  |
| 5284 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10146 | 2014-10-07 | 2026-05-22 | 10141 | 2014-10-07 | 2026-05-21 |  |
| 5285 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12863 | 2013-06-04 | 2026-05-22 | 12858 | 2013-06-04 | 2026-05-21 |  |
| 5287 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13642 | 2012-11-28 | 2026-05-22 | 13637 | 2012-11-28 | 2026-05-21 |  |
| 5288 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13208 | 2014-09-25 | 2026-05-22 | 13203 | 2014-09-25 | 2026-05-21 |  |
| 5289 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14135 | 2012-10-17 | 2026-05-22 | 14130 | 2012-10-17 | 2026-05-21 |  |
| 5291 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9851 | 2014-07-22 | 2026-05-22 | 9846 | 2014-07-22 | 2026-05-21 |  |
| 5292 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3131 | 2020-12-25 | 2026-05-22 | 3126 | 2020-12-25 | 2026-05-21 |  |
| 5299 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10182 | 2013-08-29 | 2026-05-22 | 10177 | 2013-08-29 | 2026-05-21 |  |
| 5301 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7041 | 2005-06-28 | 2026-05-22 | 7036 | 2005-06-28 | 2026-05-18 |  |
| 5302 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11349 | 2005-02-22 | 2026-05-22 | 11344 | 2005-02-22 | 2026-05-21 |  |
| 5306 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15458 | 2005-03-10 | 2026-05-22 | 15453 | 2005-03-10 | 2026-05-21 |  |
| 5309 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15858 | 2005-01-04 | 2026-05-22 | 15853 | 2005-01-04 | 2026-05-21 |  |
| 5312 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14891 | 2005-01-25 | 2026-05-22 | 14886 | 2005-01-25 | 2026-05-21 |  |
| 5314 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6156 | 2005-01-13 | 2026-05-22 | 6151 | 2005-01-13 | 2026-05-21 |  |
| 5315 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14676 | 2005-01-10 | 2026-05-22 | 14671 | 2005-01-10 | 2026-05-21 |  |
| 5321 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8658 | 2005-03-11 | 2026-05-22 | 8653 | 2005-03-11 | 2026-05-21 |  |
| 5324 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9080 | 2009-04-15 | 2026-05-22 | 9075 | 2009-04-15 | 2026-05-21 |  |
| 5328 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13609 | 2005-01-12 | 2026-05-22 | 13604 | 2005-01-12 | 2026-05-21 |  |
| 5340 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12410 | 2005-01-05 | 2026-05-22 | 12405 | 2005-01-05 | 2026-05-21 |  |
| 5344 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8140 | 2006-04-25 | 2026-05-22 | 8135 | 2006-04-25 | 2026-05-21 |  |
| 5345 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3761 | 2005-01-03 | 2026-05-22 | 3756 | 2005-01-03 | 2026-04-17 |  |
| 5347 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20587 | 2005-01-03 | 2026-05-22 | 20582 | 2005-01-03 | 2026-05-21 |  |
| 5351 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20064 | 2005-01-03 | 2026-05-22 | 20059 | 2005-01-03 | 2026-05-21 |  |
| 5353 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9868 | 2005-03-10 | 2026-05-22 | 9863 | 2005-03-10 | 2026-05-21 |  |
| 5355 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12671 | 2006-03-10 | 2026-05-22 | 12666 | 2006-03-10 | 2026-05-21 |  |
| 5356 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16918 | 2005-02-17 | 2026-05-22 | 16913 | 2005-02-17 | 2026-05-21 |  |
| 5364 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6594 | 2005-01-17 | 2026-05-22 | 6589 | 2005-01-17 | 2026-05-21 |  |
| 5371 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20520 | 2005-01-03 | 2026-05-22 | 20515 | 2005-01-03 | 2026-05-21 |  |
| 5381 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6834 | 2005-01-05 | 2026-05-22 | 6829 | 2005-01-05 | 2026-05-21 |  |
| 5386 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10887 | 2005-03-25 | 2026-05-22 | 10882 | 2005-03-25 | 2026-05-21 |  |
| 5388 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16717 | 2005-01-03 | 2026-05-22 | 16712 | 2005-01-03 | 2026-05-21 |  |
| 5392 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19323 | 2005-07-20 | 2026-05-22 | 19318 | 2005-07-20 | 2026-05-21 |  |
| 5398 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9681 | 2009-12-28 | 2026-05-22 | 9676 | 2009-12-28 | 2026-05-21 |  |
| 5403 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12838 | 2005-04-22 | 2026-05-22 | 12833 | 2005-04-22 | 2026-05-21 |  |
| 5410 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14390 | 2005-06-07 | 2026-05-22 | 14385 | 2005-06-07 | 2026-05-21 |  |
| 5425 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19797 | 2005-03-04 | 2026-05-22 | 19792 | 2005-03-04 | 2026-05-21 |  |
| 5426 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12729 | 2005-03-21 | 2026-05-22 | 12724 | 2005-03-21 | 2026-05-21 |  |
| 5432 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7501 | 2005-03-24 | 2026-05-22 | 7496 | 2005-03-24 | 2026-05-21 |  |
| 5434 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15165 | 2012-05-02 | 2026-05-22 | 15160 | 2012-05-02 | 2026-05-21 |  |
| 5438 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13699 | 2005-01-04 | 2026-05-22 | 13694 | 2005-01-04 | 2026-05-21 |  |
| 5439 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14682 | 2005-08-15 | 2026-05-22 | 14677 | 2005-08-15 | 2026-05-21 |  |
| 5443 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18467 | 2005-03-03 | 2026-05-22 | 18462 | 2005-03-03 | 2026-05-21 |  |
| 5450 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10130 | 2005-01-10 | 2026-05-22 | 10125 | 2005-01-10 | 2026-05-21 |  |
| 5452 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13615 | 2005-03-28 | 2026-05-22 | 13610 | 2005-03-28 | 2026-05-21 |  |
| 5455 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3019 | 2005-01-05 | 2026-05-22 | 3014 | 2005-01-05 | 2026-05-21 |  |
| 5457 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16742 | 2005-03-10 | 2026-05-22 | 16737 | 2005-03-10 | 2026-05-21 |  |
| 5460 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8772 | 2005-01-31 | 2026-05-22 | 8767 | 2005-01-31 | 2026-05-21 |  |
| 5464 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9558 | 2005-03-01 | 2026-05-22 | 9553 | 2005-03-01 | 2026-05-21 |  |
| 5465 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11144 | 2005-03-11 | 2026-05-22 | 11139 | 2005-03-11 | 2026-05-21 |  |
| 5468 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5181 | 2005-01-05 | 2026-05-22 | 5176 | 2005-01-05 | 2026-05-21 |  |
| 5469 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15110 | 2012-05-02 | 2026-05-22 | 15105 | 2012-05-02 | 2026-05-21 |  |
| 5471 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15143 | 2012-05-02 | 2026-05-22 | 15138 | 2012-05-02 | 2026-05-21 |  |
| 5474 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14730 | 2006-01-02 | 2026-05-22 | 14725 | 2006-01-02 | 2026-05-21 |  |
| 5475 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14668 | 2005-01-03 | 2026-05-22 | 14663 | 2005-01-03 | 2026-05-21 |  |
| 5478 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20225 | 2005-01-03 | 2026-05-22 | 20220 | 2005-01-03 | 2026-05-21 |  |
| 5481 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13065 | 2005-01-06 | 2026-05-22 | 13060 | 2005-01-06 | 2026-05-21 |  |
| 5483 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20527 | 2005-01-03 | 2026-05-22 | 20522 | 2005-01-03 | 2026-05-21 |  |
| 5484 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12092 | 2012-05-02 | 2026-05-22 | 12087 | 2012-05-02 | 2026-05-21 |  |
| 5487 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8127 | 2005-02-16 | 2026-05-22 | 8122 | 2005-02-16 | 2026-05-21 |  |
| 5488 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11179 | 2005-01-07 | 2026-05-22 | 11174 | 2005-01-07 | 2026-05-21 |  |
| 5489 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12649 | 2005-04-04 | 2026-05-22 | 12644 | 2005-04-04 | 2026-05-21 |  |
| 5490 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18641 | 2005-01-04 | 2026-05-22 | 18636 | 2005-01-04 | 2026-05-21 |  |
| 5493 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9583 | 2006-04-28 | 2026-05-22 | 9578 | 2006-04-28 | 2026-05-21 |  |
| 5498 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14528 | 2005-01-04 | 2026-05-22 | 14523 | 2005-01-04 | 2026-05-21 |  |
| 5498 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5498 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5498 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5498 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5498 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5508 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5508&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5511 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5511&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5512 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5512&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5514 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5514&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5515 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5515&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5516 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5516&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5519 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5519&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5520 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5520&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5521 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5521&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5522 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5522&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5523 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5523&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5523 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5523&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5523 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5523&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5523 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5523&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5523 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5523&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5525 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13990 | 2012-05-02 | 2026-05-22 | 13985 | 2012-05-02 | 2026-05-21 |  |
| 5529 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9820 | 2005-09-09 | 2026-05-22 | 9815 | 2005-09-09 | 2026-05-21 |  |
| 5530 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17279 | 2006-01-06 | 2026-05-22 | 17274 | 2006-01-06 | 2026-05-21 |  |
| 5531 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15122 | 2005-01-07 | 2026-05-22 | 15117 | 2005-01-07 | 2026-05-21 |  |
| 5533 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15682 | 2005-01-03 | 2026-05-22 | 15677 | 2005-01-03 | 2026-05-21 |  |
| 5534 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 5536 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15953 | 2009-11-17 | 2026-05-22 | 15948 | 2009-11-17 | 2026-05-21 |  |
| 5538 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10972 | 2013-05-16 | 2026-05-22 | 10967 | 2013-05-16 | 2026-05-21 |  |
| 5543 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5543 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5543 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5543 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5543 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5543 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5543 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5543 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5543 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5543&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5546 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5546&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5547 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5547&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5548 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5548&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5601 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=5601&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5603 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=5603&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5603 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=5603&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5603 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=5603&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5603 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=5603&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5603 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=5603&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5603 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=5603&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 5604 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8641 | 2006-05-12 | 2026-05-22 | 8636 | 2006-05-12 | 2026-05-21 |  |
| 5607 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14451 | 2012-05-02 | 2026-05-22 | 14446 | 2012-05-02 | 2026-05-21 |  |
| 5608 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14990 | 2012-05-02 | 2026-05-22 | 14985 | 2012-05-02 | 2026-05-21 |  |
| 5609 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17683 | 2005-01-10 | 2026-05-22 | 17678 | 2005-01-10 | 2026-05-21 |  |
| 5701 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8711 | 2005-01-04 | 2026-05-22 | 8706 | 2005-01-04 | 2026-05-21 |  |
| 5704 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6821 | 2005-08-22 | 2026-05-22 | 6816 | 2005-08-22 | 2026-05-21 |  |
| 5706 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15936 | 2005-12-06 | 2026-05-22 | 15931 | 2005-12-06 | 2026-05-21 |  |
| 5859 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 942 | 2015-04-17 | 2026-05-22 | 939 | 2015-04-17 | 2026-05-04 |  |
| 5864 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8025 | 2009-04-20 | 2026-05-22 | 8020 | 2009-04-20 | 2026-05-21 |  |
| 5871 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15472 | 2011-07-11 | 2026-05-22 | 15467 | 2011-07-11 | 2026-05-21 |  |
| 5876 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11291 | 2014-10-01 | 2026-05-22 | 11286 | 2014-10-01 | 2026-05-21 |  |
| 5880 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 5902 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7342 | 2005-06-17 | 2026-05-22 | 7337 | 2005-06-17 | 2026-05-20 |  |
| 5903 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17237 | 2005-02-01 | 2026-05-22 | 17232 | 2005-02-01 | 2026-05-21 |  |
| 5904 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16914 | 2005-01-06 | 2026-05-22 | 16909 | 2005-01-06 | 2026-05-21 |  |
| 5905 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16623 | 2005-01-03 | 2026-05-22 | 16618 | 2005-01-03 | 2026-05-21 |  |
| 5906 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8579 | 2012-05-02 | 2026-05-22 | 8574 | 2012-05-02 | 2026-05-21 |  |
| 5907 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15043 | 2012-06-06 | 2026-05-22 | 15038 | 2012-06-06 | 2026-05-21 |  |
| 6005 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15719 | 2005-01-04 | 2026-05-22 | 15714 | 2005-01-04 | 2026-05-21 |  |
| 6015 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15976 | 2005-03-01 | 2026-05-22 | 15971 | 2005-03-01 | 2026-05-21 |  |
| 6016 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17440 | 2005-02-14 | 2026-05-22 | 17435 | 2005-02-14 | 2026-05-21 |  |
| 6020 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3881 | 2005-03-01 | 2026-05-22 | 3876 | 2005-03-01 | 2026-05-21 |  |
| 6021 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11493 | 2005-03-01 | 2026-05-22 | 11488 | 2005-03-01 | 2026-05-21 |  |
| 6023 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14105 | 2007-06-25 | 2026-05-22 | 14100 | 2007-06-25 | 2026-05-21 |  |
| 6024 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14385 | 2008-05-15 | 2026-05-22 | 14380 | 2008-05-15 | 2026-05-21 |  |
| 6026 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11024 | 2015-08-17 | 2026-05-22 | 11019 | 2015-08-17 | 2026-05-21 |  |
| 6028 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 476 | 2023-09-19 | 2026-05-22 | 471 | 2023-09-19 | 2026-05-14 |  |
| 6101 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10032 | 2005-03-09 | 2026-05-22 | 10027 | 2005-03-09 | 2026-05-21 |  |
| 6104 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19147 | 2005-01-06 | 2026-05-22 | 19142 | 2005-01-06 | 2026-05-21 |  |
| 6108 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16395 | 2005-01-03 | 2026-05-22 | 16390 | 2005-01-03 | 2026-05-21 |  |
| 6109 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9784 | 2005-07-21 | 2026-05-22 | 9779 | 2005-07-21 | 2026-05-21 |  |
| 6111 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13219 | 2005-01-11 | 2026-05-22 | 13214 | 2005-01-11 | 2026-05-21 |  |
| 6112 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14700 | 2012-05-02 | 2026-05-22 | 14695 | 2012-05-02 | 2026-05-21 |  |
| 6113 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11044 | 2005-03-02 | 2026-05-22 | 11039 | 2005-03-02 | 2026-05-21 |  |
| 6114 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10404 | 2007-04-02 | 2026-05-22 | 10399 | 2007-04-02 | 2026-05-21 |  |
| 6115 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14944 | 2012-05-02 | 2026-05-22 | 14939 | 2012-05-02 | 2026-05-21 |  |
| 6116 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15124 | 2012-05-02 | 2026-05-22 | 15119 | 2012-05-02 | 2026-05-21 |  |
| 6117 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10862 | 2012-05-02 | 2026-05-22 | 10857 | 2012-05-02 | 2026-05-21 |  |
| 6118 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12976 | 2005-04-29 | 2026-05-22 | 12971 | 2005-04-29 | 2026-05-21 |  |
| 6120 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16336 | 2005-01-03 | 2026-05-22 | 16331 | 2005-01-03 | 2026-05-21 |  |
| 6121 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20561 | 2005-01-03 | 2026-05-22 | 20556 | 2005-01-03 | 2026-05-21 |  |
| 6122 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10865 | 2006-01-06 | 2026-05-22 | 10860 | 2006-01-06 | 2026-05-21 |  |
| 6123 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15668 | 2005-03-08 | 2026-05-22 | 15663 | 2005-03-08 | 2026-05-21 |  |
| 6124 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12211 | 2005-01-18 | 2026-05-22 | 12206 | 2005-01-18 | 2026-05-21 |  |
| 6125 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19062 | 2005-01-25 | 2026-05-22 | 19057 | 2005-01-25 | 2026-05-21 |  |
| 6126 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16624 | 2005-06-16 | 2026-05-22 | 16619 | 2005-06-16 | 2026-05-21 |  |
| 6127 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15584 | 2005-02-18 | 2026-05-22 | 15579 | 2005-02-18 | 2026-05-21 |  |
| 6128 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13932 | 2012-05-02 | 2026-05-22 | 13927 | 2012-05-02 | 2026-05-21 |  |
| 6129 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18594 | 2005-01-26 | 2026-05-22 | 18589 | 2005-01-26 | 2026-05-21 |  |
| 6130 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7875 | 2005-05-26 | 2026-05-22 | 7870 | 2005-05-26 | 2026-05-21 |  |
| 6133 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12341 | 2012-05-02 | 2026-05-22 | 12336 | 2012-05-02 | 2026-05-21 |  |
| 6134 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12484 | 2005-06-06 | 2026-05-22 | 12479 | 2005-06-06 | 2026-05-21 |  |
| 6136 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14183 | 2012-05-02 | 2026-05-22 | 14178 | 2012-05-02 | 2026-05-21 |  |
| 6138 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19772 | 2005-01-03 | 2026-05-22 | 19767 | 2005-01-03 | 2026-05-21 |  |
| 6139 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15078 | 2012-05-02 | 2026-05-22 | 15073 | 2012-05-02 | 2026-05-21 |  |
| 6140 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8402 | 2005-07-22 | 2026-05-22 | 8397 | 2005-07-22 | 2026-05-21 |  |
| 6141 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14250 | 2012-05-02 | 2026-05-22 | 14245 | 2012-05-02 | 2026-05-21 |  |
| 6142 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15041 | 2012-05-02 | 2026-05-22 | 15036 | 2012-05-02 | 2026-05-21 |  |
| 6143 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6143&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6143 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6143&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6143 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6143&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6143 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6143&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6143 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6143&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6143 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6143&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6143 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6143&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6143 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6143&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6144 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6144&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6145 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6145&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6146 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6146&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6147 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6147&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6148 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6148&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6150 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6150&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6151 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6151&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6152 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6152&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6153 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6153&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6154 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6154&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6155 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6155&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6156 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6156&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6158 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6158&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6160 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6160&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6161 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6161&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6163 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6163&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6164 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6164&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6164 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15378 | 2005-06-29 | 2026-05-22 | 15373 | 2005-06-29 | 2026-05-21 |  |
| 6165 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13398 | 2012-05-02 | 2026-05-22 | 13393 | 2012-05-02 | 2026-05-21 |  |
| 6166 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15138 | 2012-05-02 | 2026-05-22 | 15133 | 2012-05-02 | 2026-05-21 |  |
| 6167 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12862 | 2005-01-25 | 2026-05-22 | 12857 | 2005-01-25 | 2026-05-21 |  |
| 6168 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15121 | 2012-05-02 | 2026-05-22 | 15116 | 2012-05-02 | 2026-05-21 |  |
| 6169 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12002 | 2006-01-18 | 2026-05-22 | 11997 | 2006-01-18 | 2026-05-21 |  |
| 6170 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16049 | 2005-03-08 | 2026-05-22 | 16044 | 2005-03-08 | 2026-05-21 |  |
| 6171 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8840 | 2006-05-02 | 2026-05-22 | 8835 | 2006-05-02 | 2026-05-21 |  |
| 6173 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17141 | 2005-01-21 | 2026-05-22 | 17136 | 2005-01-21 | 2026-05-21 |  |
| 6174 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10076 | 2005-03-10 | 2026-05-22 | 10071 | 2005-03-10 | 2026-05-21 |  |
| 6175 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14882 | 2005-01-04 | 2026-05-22 | 14877 | 2005-01-04 | 2026-05-21 |  |
| 6176 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16906 | 2005-01-03 | 2026-05-22 | 16901 | 2005-01-03 | 2026-05-21 |  |
| 6177 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15600 | 2005-04-11 | 2026-05-22 | 15595 | 2005-04-11 | 2026-05-21 |  |
| 6179 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11694 | 2005-02-22 | 2026-05-22 | 11689 | 2005-02-22 | 2026-05-21 |  |
| 6180 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19543 | 2005-01-03 | 2026-05-22 | 19538 | 2005-01-03 | 2026-05-21 |  |
| 6182 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20389 | 2005-02-23 | 2026-05-22 | 20384 | 2005-02-23 | 2026-05-21 |  |
| 6183 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14733 | 2005-01-31 | 2026-05-22 | 14728 | 2005-01-31 | 2026-05-21 |  |
| 6184 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14045 | 2005-01-05 | 2026-05-22 | 14040 | 2005-01-05 | 2026-05-21 |  |
| 6185 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15165 | 2005-01-24 | 2026-05-22 | 15160 | 2005-01-24 | 2026-05-21 |  |
| 6186 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13584 | 2005-01-14 | 2026-05-22 | 13579 | 2005-01-14 | 2026-05-21 |  |
| 6187 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17241 | 2005-01-03 | 2026-05-22 | 17236 | 2005-01-03 | 2026-05-21 |  |
| 6188 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20336 | 2005-01-03 | 2026-05-22 | 20331 | 2005-01-03 | 2026-05-21 |  |
| 6189 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15080 | 2012-05-02 | 2026-05-22 | 15075 | 2012-05-02 | 2026-05-21 |  |
| 6190 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13970 | 2006-04-25 | 2026-05-22 | 13965 | 2006-04-25 | 2026-05-21 |  |
| 6191 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16821 | 2005-02-17 | 2026-05-22 | 16816 | 2005-02-17 | 2026-05-21 |  |
| 6192 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15075 | 2012-05-02 | 2026-05-22 | 15070 | 2012-05-02 | 2026-05-21 |  |
| 6194 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13111 | 2005-01-04 | 2026-05-22 | 13106 | 2005-01-04 | 2026-05-21 |  |
| 6195 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12832 | 2007-09-10 | 2026-05-22 | 12827 | 2007-09-10 | 2026-05-21 |  |
| 6196 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15131 | 2012-05-02 | 2026-05-22 | 15126 | 2012-05-02 | 2026-05-21 |  |
| 6197 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15069 | 2012-05-02 | 2026-05-22 | 15064 | 2012-05-02 | 2026-05-21 |  |
| 6198 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4395 | 2005-03-16 | 2026-05-22 | 4390 | 2005-03-16 | 2026-05-21 |  |
| 6199 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4144 | 2006-12-14 | 2026-05-22 | 4139 | 2006-12-14 | 2026-05-21 |  |
| 6201 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11300 | 2012-05-02 | 2026-05-22 | 11295 | 2012-05-02 | 2026-05-21 |  |
| 6202 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15147 | 2012-05-02 | 2026-05-22 | 15142 | 2012-05-02 | 2026-05-21 |  |
| 6203 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14289 | 2005-08-10 | 2026-05-22 | 14284 | 2005-08-10 | 2026-05-21 |  |
| 6204 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8707 | 2005-01-20 | 2026-05-22 | 8702 | 2005-01-20 | 2026-05-21 |  |
| 6205 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15441 | 2005-01-03 | 2026-05-22 | 15436 | 2005-01-03 | 2026-05-21 |  |
| 6206 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15152 | 2012-05-02 | 2026-05-22 | 15147 | 2012-05-02 | 2026-05-21 |  |
| 6207 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17013 | 2005-01-03 | 2026-05-22 | 17008 | 2005-01-03 | 2026-05-21 |  |
| 6208 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16412 | 2005-04-12 | 2026-05-22 | 16407 | 2005-04-12 | 2026-05-21 |  |
| 6209 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15145 | 2012-05-02 | 2026-05-22 | 15140 | 2012-05-02 | 2026-05-21 |  |
| 6210 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10616 | 2007-05-16 | 2026-05-22 | 10611 | 2007-05-16 | 2026-05-21 |  |
| 6212 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4107 | 2006-11-09 | 2026-05-22 | 4102 | 2006-11-09 | 2026-05-21 |  |
| 6213 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17059 | 2005-01-03 | 2026-05-22 | 17054 | 2005-01-03 | 2026-05-21 |  |
| 6214 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17772 | 2005-01-21 | 2026-05-22 | 17767 | 2005-01-21 | 2026-05-21 |  |
| 6215 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14870 | 2005-01-26 | 2026-05-22 | 14865 | 2005-01-26 | 2026-05-21 |  |
| 6216 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14082 | 2012-05-02 | 2026-05-22 | 14077 | 2012-05-02 | 2026-05-21 |  |
| 6217 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16108 | 2005-04-20 | 2026-05-22 | 16103 | 2005-04-20 | 2026-05-21 |  |
| 6218 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9818 | 2005-01-14 | 2026-05-22 | 9813 | 2005-01-14 | 2026-05-21 |  |
| 6219 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11584 | 2005-09-08 | 2026-05-22 | 11579 | 2005-09-08 | 2026-05-21 |  |
| 6220 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15779 | 2005-07-19 | 2026-05-22 | 15774 | 2005-07-19 | 2026-05-21 |  |
| 6221 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10517 | 2005-06-15 | 2026-05-22 | 10512 | 2005-06-15 | 2026-05-21 |  |
| 6222 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4737 | 2005-01-04 | 2026-05-22 | 4732 | 2005-01-04 | 2026-05-21 |  |
| 6223 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20190 | 2005-01-03 | 2026-05-22 | 20185 | 2005-01-03 | 2026-05-21 |  |
| 6224 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16039 | 2005-02-23 | 2026-05-22 | 16034 | 2005-02-23 | 2026-05-21 |  |
| 6225 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1144 | 2012-05-29 | 2026-05-22 | 1139 | 2012-05-29 | 2026-05-21 |  |
| 6226 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14801 | 2005-04-04 | 2026-05-22 | 14796 | 2005-04-04 | 2026-05-21 |  |
| 6227 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13047 | 2005-01-20 | 2026-05-22 | 13042 | 2005-01-20 | 2026-05-21 |  |
| 6229 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11244 | 2005-05-26 | 2026-05-22 | 11239 | 2005-05-26 | 2026-05-21 |  |
| 6230 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18264 | 2005-01-03 | 2026-05-22 | 18259 | 2005-01-03 | 2026-05-21 |  |
| 6231 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17470 | 2005-07-08 | 2026-05-22 | 17465 | 2005-07-08 | 2026-05-21 |  |
| 6233 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16591 | 2005-01-03 | 2026-05-22 | 16586 | 2005-01-03 | 2026-05-21 |  |
| 6234 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15046 | 2005-01-04 | 2026-05-22 | 15041 | 2005-01-04 | 2026-05-21 |  |
| 6235 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14901 | 2012-05-02 | 2026-05-22 | 14896 | 2012-05-02 | 2026-05-21 |  |
| 6237 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18186 | 2005-01-03 | 2026-05-22 | 18181 | 2005-01-03 | 2026-05-21 |  |
| 6239 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 6240 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7071 | 2010-05-10 | 2026-05-22 | 7066 | 2010-05-10 | 2026-05-21 |  |
| 6241 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6816 | 2006-12-18 | 2026-05-22 | 6811 | 2006-12-18 | 2026-05-21 |  |
| 6243 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17963 | 2005-01-05 | 2026-05-22 | 17958 | 2005-01-05 | 2026-05-21 |  |
| 6244 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20563 | 2005-01-03 | 2026-05-22 | 20558 | 2005-01-03 | 2026-05-21 |  |
| 6245 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17454 | 2005-01-04 | 2026-05-22 | 17449 | 2005-01-04 | 2026-05-21 |  |
| 6246 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10336 | 2006-11-21 | 2026-05-22 | 10331 | 2006-11-21 | 2026-05-21 |  |
| 6248 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8545 | 2005-01-25 | 2026-05-22 | 8540 | 2005-01-25 | 2026-05-21 |  |
| 6257 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 6259 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5110 | 2005-01-11 | 2026-05-22 | 5105 | 2005-01-11 | 2026-05-21 |  |
| 6261 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20283 | 2005-01-03 | 2026-05-22 | 20278 | 2005-01-03 | 2026-05-21 |  |
| 6263 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13822 | 2005-01-03 | 2026-05-22 | 13817 | 2005-01-03 | 2026-05-21 |  |
| 6264 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7479 | 2005-01-03 | 2026-05-22 | 7474 | 2005-01-03 | 2026-05-20 |  |
| 6265 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12002 | 2005-04-20 | 2026-05-22 | 11997 | 2005-04-20 | 2026-05-21 |  |
| 6266 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13556 | 2005-01-10 | 2026-05-22 | 13551 | 2005-01-10 | 2026-05-21 |  |
| 6269 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 6270 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11850 | 2005-01-03 | 2026-05-22 | 11845 | 2005-01-03 | 2026-05-21 |  |
| 6271 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15214 | 2007-06-20 | 2026-05-22 | 15209 | 2007-06-20 | 2026-05-21 |  |
| 6272 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2060 | 2007-09-21 | 2026-05-22 | 2055 | 2007-09-21 | 2026-05-21 |  |
| 6274 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18507 | 2005-01-03 | 2026-05-22 | 18502 | 2005-01-03 | 2026-05-21 |  |
| 6275 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13313 | 2005-02-22 | 2026-05-22 | 13308 | 2005-02-22 | 2026-05-21 |  |
| 6276 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11617 | 2005-02-18 | 2026-05-22 | 11612 | 2005-02-18 | 2026-05-21 |  |
| 6277 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15136 | 2012-05-02 | 2026-05-22 | 15131 | 2012-05-02 | 2026-05-21 |  |
| 6278 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18967 | 2005-03-17 | 2026-05-22 | 18962 | 2005-03-17 | 2026-05-21 |  |
| 6279 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18033 | 2005-01-07 | 2026-05-22 | 18028 | 2005-01-07 | 2026-05-21 |  |
| 6281 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14447 | 2012-05-02 | 2026-05-22 | 14442 | 2012-05-02 | 2026-05-21 |  |
| 6282 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-22 | 15163 | 2012-05-02 | 2026-05-21 |  |
| 6283 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15593 | 2005-01-07 | 2026-05-22 | 15588 | 2005-01-07 | 2026-05-21 |  |
| 6284 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18249 | 2005-01-05 | 2026-05-22 | 18244 | 2005-01-05 | 2026-05-21 |  |
| 6285 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 6290 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18405 | 2005-06-06 | 2026-05-22 | 18400 | 2005-06-06 | 2026-05-21 |  |
| 6291 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6382 | 2005-02-14 | 2026-05-22 | 6377 | 2005-02-14 | 2026-05-21 |  |
| 6292 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10071 | 2005-02-01 | 2026-05-22 | 10066 | 2005-02-01 | 2026-05-21 |  |
| 6294 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10206 | 2005-02-16 | 2026-05-22 | 10201 | 2005-02-16 | 2026-05-21 |  |
| 6405 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12065 | 2013-01-02 | 2026-05-22 | 12060 | 2013-01-02 | 2026-05-21 |  |
| 6409 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13957 | 2013-08-01 | 2026-05-22 | 13952 | 2013-08-01 | 2026-05-21 |  |
| 6411 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13636 | 2014-03-11 | 2026-05-22 | 13631 | 2014-03-11 | 2026-05-21 |  |
| 6412 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14064 | 2013-01-29 | 2026-05-22 | 14059 | 2013-01-29 | 2026-05-21 |  |
| 6414 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14029 | 2013-01-04 | 2026-05-22 | 14024 | 2013-01-04 | 2026-05-21 |  |
| 6415 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13933 | 2013-05-16 | 2026-05-22 | 13928 | 2013-05-16 | 2026-05-21 |  |
| 6416 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10928 | 2013-05-31 | 2026-05-22 | 10923 | 2013-05-31 | 2026-05-21 |  |
| 6417 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6383 | 2013-03-14 | 2026-05-22 | 6378 | 2013-03-14 | 2026-05-21 |  |
| 6418 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6958 | 2015-04-10 | 2026-05-22 | 6953 | 2015-04-10 | 2026-05-21 |  |
| 6419 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7001 | 2013-07-22 | 2026-05-22 | 6996 | 2013-07-22 | 2026-05-21 |  |
| 6423 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3513 | 2014-10-23 | 2026-05-22 | 3508 | 2014-10-23 | 2026-05-21 |  |
| 6425 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7537 | 2013-09-09 | 2026-05-22 | 7532 | 2013-09-09 | 2026-05-21 |  |
| 6426 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12387 | 2013-10-31 | 2026-05-22 | 12382 | 2013-10-31 | 2026-05-21 |  |
| 6431 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10484 | 2014-12-04 | 2026-05-22 | 10479 | 2014-12-04 | 2026-05-21 |  |
| 6432 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10400 | 2015-04-23 | 2026-05-22 | 10395 | 2015-04-23 | 2026-05-21 |  |
| 6435 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11936 | 2013-12-25 | 2026-05-22 | 11931 | 2013-12-25 | 2026-05-21 |  |
| 6438 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11398 | 2013-11-25 | 2026-05-22 | 11393 | 2013-11-25 | 2026-05-21 |  |
| 6441 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8836 | 2014-01-14 | 2026-05-22 | 8831 | 2014-01-14 | 2026-05-21 |  |
| 6442 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11834 | 2014-03-17 | 2026-05-22 | 11829 | 2014-03-17 | 2026-05-21 |  |
| 6443 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12346 | 2015-09-18 | 2026-05-22 | 12341 | 2015-09-18 | 2026-05-21 |  |
| 6446 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12499 | 2014-03-11 | 2026-05-22 | 12494 | 2014-03-11 | 2026-05-21 |  |
| 6449 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13066 | 2014-03-03 | 2026-05-22 | 13061 | 2014-03-03 | 2026-05-21 |  |
| 6451 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13090 | 2015-01-26 | 2026-05-22 | 13085 | 2015-01-26 | 2026-05-21 |  |
| 6456 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12734 | 2015-06-12 | 2026-05-22 | 12729 | 2015-06-12 | 2026-05-21 |  |
| 6461 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9281 | 2014-07-15 | 2026-05-22 | 9276 | 2014-07-15 | 2026-05-21 |  |
| 6462 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13111 | 2014-06-26 | 2026-05-22 | 13106 | 2014-06-26 | 2026-05-21 |  |
| 6464 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9402 | 2014-08-13 | 2026-05-22 | 9397 | 2014-08-13 | 2026-05-21 |  |
| 6465 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6214 | 2014-07-17 | 2026-05-22 | 6209 | 2014-07-17 | 2026-05-21 |  |
| 6469 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10236 | 2015-05-28 | 2026-05-22 | 10231 | 2015-05-28 | 2026-05-21 |  |
| 6470 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11671 | 2015-02-10 | 2026-05-22 | 11666 | 2015-02-10 | 2026-05-21 |  |
| 6472 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10439 | 2015-02-26 | 2026-05-22 | 10434 | 2015-02-26 | 2026-05-21 |  |
| 6473 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 921 | 2014-12-10 | 2026-05-22 | 918 | 2014-12-10 | 2026-05-15 |  |
| 6477 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11236 | 2016-02-05 | 2026-05-22 | 11231 | 2016-02-05 | 2026-05-21 |  |
| 6482 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4375 | 2016-06-13 | 2026-05-22 | 4370 | 2016-06-13 | 2026-05-21 |  |
| 6485 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11681 | 2014-11-17 | 2026-05-22 | 11676 | 2014-11-17 | 2026-05-21 |  |
| 6486 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10822 | 2014-11-13 | 2026-05-22 | 10817 | 2014-11-13 | 2026-05-21 |  |
| 6488 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12922 | 2014-11-04 | 2026-05-22 | 12917 | 2014-11-04 | 2026-05-21 |  |
| 6491 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8773 | 2016-08-25 | 2026-05-22 | 8768 | 2016-08-25 | 2026-05-21 |  |
| 6492 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6492&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6492 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6492&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6492 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6492&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6492 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6492&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6492 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6492&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6492 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6492&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6493 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6493&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6494 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6494&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6496 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6496&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6498 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6498 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6498 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6498 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6498 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6498&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6498 | TaiwanStockDividend | MISMATCH | 8 | 2015-08-30 | 2026-07-05 | 7 | 2015-08-30 | 2025-06-25 |  |
| 6499 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10315 | 2014-12-17 | 2026-05-22 | 10310 | 2014-12-17 | 2026-05-21 |  |
| 6504 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14358 | 2007-08-10 | 2026-05-22 | 14353 | 2007-08-10 | 2026-05-21 |  |
| 6505 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 6506 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9419 | 2007-08-07 | 2026-05-22 | 9414 | 2007-08-07 | 2026-05-21 |  |
| 6508 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12746 | 2005-02-21 | 2026-05-22 | 12741 | 2005-02-21 | 2026-05-21 |  |
| 6509 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17924 | 2005-02-18 | 2026-05-22 | 17919 | 2005-02-18 | 2026-05-21 |  |
| 6510 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12266 | 2015-03-03 | 2026-05-22 | 12261 | 2015-03-03 | 2026-05-21 |  |
| 6515 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6854 | 2019-11-19 | 2026-05-22 | 6849 | 2019-11-19 | 2026-05-21 |  |
| 6516 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7126 | 2015-01-26 | 2026-05-22 | 7121 | 2015-01-26 | 2026-05-21 |  |
| 6517 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4207 | 2021-11-29 | 2026-05-22 | 4202 | 2021-11-29 | 2026-05-21 |  |
| 6523 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11043 | 2015-04-30 | 2026-05-22 | 11038 | 2015-04-30 | 2026-05-21 |  |
| 6525 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11918 | 2016-02-05 | 2026-05-22 | 11913 | 2016-02-05 | 2026-05-21 |  |
| 6526 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4407 | 2015-07-08 | 2026-05-22 | 4402 | 2015-07-08 | 2026-05-21 |  |
| 6527 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4737 | 2015-05-22 | 2026-05-22 | 4732 | 2015-05-22 | 2026-05-21 |  |
| 6530 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7560 | 2016-12-27 | 2026-05-22 | 7555 | 2016-12-27 | 2026-05-21 |  |
| 6531 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11694 | 2015-06-18 | 2026-05-22 | 11689 | 2015-06-18 | 2026-05-21 |  |
| 6532 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10942 | 2015-06-09 | 2026-05-22 | 10937 | 2015-06-09 | 2026-05-21 |  |
| 6533 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11339 | 2015-08-10 | 2026-05-22 | 11334 | 2015-08-10 | 2026-05-21 |  |
| 6534 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3766 | 2015-07-31 | 2026-05-22 | 3761 | 2015-07-31 | 2026-05-21 |  |
| 6535 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10100 | 2015-12-16 | 2026-05-22 | 10095 | 2015-12-16 | 2026-05-21 |  |
| 6538 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9296 | 2016-03-25 | 2026-05-22 | 9291 | 2016-03-25 | 2026-05-21 |  |
| 6541 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10904 | 2015-08-17 | 2026-05-22 | 10899 | 2015-08-17 | 2026-05-21 |  |
| 6542 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6695 | 2016-05-24 | 2026-05-22 | 6690 | 2016-05-24 | 2026-05-21 |  |
| 6542 | TaiwanStockDividend | MISMATCH | 7 | 2015-10-20 | 2026-07-07 | 6 | 2015-10-20 | 2025-07-08 |  |
| 6546 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5295 | 2015-07-31 | 2026-05-22 | 5290 | 2015-07-31 | 2026-05-21 |  |
| 6547 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9420 | 2015-12-08 | 2026-05-22 | 9415 | 2015-12-08 | 2026-05-21 |  |
| 6548 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11245 | 2016-06-23 | 2026-05-22 | 11240 | 2016-06-23 | 2026-05-21 |  |
| 6550 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6809 | 2016-02-15 | 2026-05-22 | 6804 | 2016-02-15 | 2026-05-21 |  |
| 6552 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11134 | 2015-09-08 | 2026-05-22 | 11129 | 2015-09-08 | 2026-05-21 |  |
| 6556 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3960 | 2017-11-01 | 2026-05-22 | 3955 | 2017-11-01 | 2026-05-11 |  |
| 6558 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10025 | 2016-06-15 | 2026-05-22 | 10020 | 2016-06-15 | 2026-05-21 |  |
| 6560 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8600 | 2017-01-24 | 2026-05-22 | 8595 | 2017-01-24 | 2026-05-21 |  |
| 6561 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9663 | 2017-07-06 | 2026-05-22 | 9658 | 2017-07-06 | 2026-05-21 |  |
| 6568 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11483 | 2016-02-19 | 2026-05-22 | 11478 | 2016-02-19 | 2026-05-21 |  |
| 6569 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7197 | 2016-01-25 | 2026-05-22 | 7192 | 2016-01-25 | 2026-05-21 |  |
| 6570 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6791 | 2017-12-20 | 2026-05-22 | 6786 | 2017-12-20 | 2026-05-21 |  |
| 6573 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9907 | 2017-09-26 | 2026-05-22 | 9902 | 2017-09-26 | 2026-05-21 |  |
| 6574 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1730 | 2016-08-25 | 2026-05-22 | 1725 | 2016-08-25 | 2026-05-21 |  |
| 6576 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9704 | 2016-04-12 | 2026-05-22 | 9699 | 2016-04-12 | 2026-05-21 |  |
| 6577 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7915 | 2017-03-17 | 2026-05-22 | 7910 | 2017-03-17 | 2026-05-21 |  |
| 6578 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7447 | 2016-08-10 | 2026-05-22 | 7442 | 2016-08-10 | 2026-05-21 |  |
| 6579 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9970 | 2016-02-05 | 2026-05-22 | 9965 | 2016-02-05 | 2026-05-21 |  |
| 6581 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9685 | 2017-12-20 | 2026-05-22 | 9680 | 2017-12-20 | 2026-05-21 |  |
| 6582 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9726 | 2016-02-05 | 2026-05-22 | 9721 | 2016-02-05 | 2026-05-21 |  |
| 6584 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3960 | 2022-03-08 | 2026-05-22 | 3955 | 2022-03-08 | 2026-05-21 |  |
| 6585 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4885 | 2018-01-18 | 2026-05-22 | 4880 | 2018-01-18 | 2026-05-21 |  |
| 6588 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4813 | 2016-09-02 | 2026-05-22 | 4808 | 2016-09-02 | 2026-05-21 |  |
| 6589 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8339 | 2016-12-26 | 2026-05-22 | 8334 | 2016-12-26 | 2026-05-21 |  |
| 6591 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10247 | 2016-12-26 | 2026-05-22 | 10242 | 2016-12-26 | 2026-05-21 |  |
| 6592 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7872 | 2019-04-02 | 2026-05-22 | 7867 | 2019-04-02 | 2026-05-21 |  |
| 6593 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5161 | 2017-11-10 | 2026-05-22 | 5156 | 2017-11-10 | 2026-05-21 |  |
| 6596 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7816 | 2017-06-27 | 2026-05-22 | 7811 | 2017-06-27 | 2026-05-21 |  |
| 6597 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 755 | 2021-09-07 | 2026-05-22 | 750 | 2021-09-07 | 2026-05-21 |  |
| 6598 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7803 | 2017-02-09 | 2026-05-22 | 7798 | 2017-02-09 | 2026-05-21 |  |
| 6603 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12759 | 2005-01-28 | 2026-05-22 | 12754 | 2005-01-28 | 2026-05-21 |  |
| 6605 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15139 | 2012-05-02 | 2026-05-22 | 15134 | 2012-05-02 | 2026-05-21 |  |
| 6606 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3784 | 2007-06-27 | 2026-05-22 | 3779 | 2007-06-27 | 2026-05-21 |  |
| 6609 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13944 | 2005-01-03 | 2026-05-22 | 13939 | 2005-01-03 | 2026-05-21 |  |
| 6612 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8322 | 2018-07-03 | 2026-05-22 | 8317 | 2018-07-03 | 2026-05-21 |  |
| 6613 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9644 | 2016-12-29 | 2026-05-22 | 9639 | 2016-12-29 | 2026-05-21 |  |
| 6614 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 638 | 2024-09-03 | 2026-05-22 | 633 | 2024-09-03 | 2026-05-21 |  |
| 6615 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3849 | 2017-01-20 | 2026-05-22 | 3844 | 2017-01-20 | 2026-05-21 |  |
| 6616 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6555 | 2018-01-10 | 2026-05-22 | 6550 | 2018-01-10 | 2026-05-20 |  |
| 6617 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3842 | 2017-02-15 | 2026-05-22 | 3837 | 2017-02-15 | 2026-05-21 |  |
| 6620 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2821 | 2017-04-19 | 2026-05-22 | 2816 | 2017-04-19 | 2026-05-21 |  |
| 6624 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2787 | 2017-03-20 | 2026-05-22 | 2782 | 2017-03-20 | 2026-05-21 |  |
| 6625 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8125 | 2017-11-24 | 2026-05-22 | 8120 | 2017-11-24 | 2026-05-21 |  |
| 6629 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4190 | 2019-06-10 | 2026-05-22 | 4185 | 2019-06-10 | 2026-05-21 |  |
| 6637 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2258 | 2018-09-13 | 2026-05-22 | 2253 | 2018-09-13 | 2026-05-21 |  |
| 6639 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 489 | 2021-10-06 | 2026-05-22 | 486 | 2021-10-06 | 2026-05-18 |  |
| 6640 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5582 | 2017-09-05 | 2026-05-22 | 5577 | 2017-09-05 | 2026-05-21 |  |
| 6641 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6875 | 2018-10-01 | 2026-05-22 | 6870 | 2018-10-01 | 2026-05-21 |  |
| 6641 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6641&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6641 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6641&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6641 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6641&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6641 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6641&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6641 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6641&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6642 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6642&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6642 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6642&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6642 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6642&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6642 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6642&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6642 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6642&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6642 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6642&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6642 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6642&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6642 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6642&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6643 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8920 | 2018-01-24 | 2026-05-22 | 8915 | 2018-01-24 | 2026-05-21 |  |
| 6645 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3700 | 2018-06-14 | 2026-05-22 | 3695 | 2018-06-14 | 2026-05-21 |  |
| 6649 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5917 | 2019-01-18 | 2026-05-22 | 5912 | 2019-01-18 | 2026-05-21 |  |
| 6651 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6579 | 2017-10-06 | 2026-05-22 | 6574 | 2017-10-06 | 2026-05-21 |  |
| 6654 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6805 | 2018-06-12 | 2026-05-22 | 6800 | 2018-06-12 | 2026-05-21 |  |
| 6655 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6971 | 2018-01-18 | 2026-05-22 | 6966 | 2018-01-18 | 2026-05-21 |  |
| 6657 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3656 | 2018-08-08 | 2026-05-22 | 3651 | 2018-08-08 | 2026-05-21 |  |
| 6658 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3167 | 2021-11-12 | 2026-05-22 | 3162 | 2021-11-12 | 2026-05-21 |  |
| 6658 | TaiwanStockDividend | MISMATCH | 6 | 2022-09-03 | 2026-06-14 | 5 | 2022-09-03 | 2025-05-28 |  |
| 6664 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9055 | 2017-11-08 | 2026-05-22 | 9050 | 2017-11-08 | 2026-05-21 |  |
| 6665 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 735 | 2017-11-23 | 2026-05-22 | 732 | 2017-11-23 | 2026-05-18 |  |
| 6666 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8430 | 2018-11-19 | 2026-05-22 | 8425 | 2018-11-19 | 2026-05-21 |  |
| 6667 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8560 | 2018-11-13 | 2026-05-22 | 8555 | 2018-11-13 | 2026-05-21 |  |
| 6668 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9203 | 2017-11-30 | 2026-05-22 | 9198 | 2017-11-30 | 2026-05-21 |  |
| 6669 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9573 | 2017-11-13 | 2026-05-22 | 9568 | 2017-11-13 | 2026-05-21 |  |
| 6670 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9044 | 2018-03-15 | 2026-05-22 | 9039 | 2018-03-15 | 2026-05-21 |  |
| 6671 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6915 | 2018-12-17 | 2026-05-22 | 6910 | 2018-12-17 | 2026-05-21 |  |
| 6672 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8739 | 2018-01-26 | 2026-05-22 | 8734 | 2018-01-26 | 2026-05-21 |  |
| 6674 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7440 | 2018-01-15 | 2026-05-22 | 7435 | 2018-01-15 | 2026-05-21 |  |
| 6679 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8772 | 2018-05-30 | 2026-05-22 | 8767 | 2018-05-30 | 2026-05-21 |  |
| 6683 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8573 | 2018-08-03 | 2026-05-22 | 8568 | 2018-08-03 | 2026-05-21 |  |
| 6684 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5928 | 2018-06-11 | 2026-05-22 | 5923 | 2018-06-11 | 2026-05-21 |  |
| 6689 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5645 | 2018-08-22 | 2026-05-22 | 5640 | 2018-08-22 | 2026-05-21 |  |
| 6690 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6097 | 2019-03-08 | 2026-05-22 | 6092 | 2019-03-08 | 2026-05-21 |  |
| 6691 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5635 | 2019-04-18 | 2026-05-22 | 5630 | 2019-04-18 | 2026-05-21 |  |
| 6693 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5205 | 2019-03-07 | 2026-05-22 | 5200 | 2019-03-07 | 2026-05-21 |  |
| 6695 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4452 | 2019-09-23 | 2026-05-22 | 4447 | 2019-09-23 | 2026-05-21 |  |
| 6697 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3915 | 2019-12-30 | 2026-05-22 | 3910 | 2019-12-30 | 2026-05-21 |  |
| 6698 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7986 | 2018-12-12 | 2026-05-22 | 7981 | 2018-12-12 | 2026-05-21 |  |
| 6703 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5725 | 2018-10-25 | 2026-05-22 | 5720 | 2018-10-25 | 2026-05-21 |  |
| 6706 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8045 | 2019-01-14 | 2026-05-22 | 8040 | 2019-01-14 | 2026-05-21 |  |
| 6712 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6649 | 2019-05-02 | 2026-05-22 | 6644 | 2019-05-02 | 2026-05-21 |  |
| 6715 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7935 | 2018-12-10 | 2026-05-22 | 7930 | 2018-12-10 | 2026-05-21 |  |
| 6716 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5970 | 2020-03-30 | 2026-05-22 | 5965 | 2020-03-30 | 2026-05-21 |  |
| 6719 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5780 | 2021-03-12 | 2026-05-22 | 5775 | 2021-03-12 | 2026-05-21 |  |
| 6720 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2350 | 2021-04-09 | 2026-05-22 | 2345 | 2021-04-09 | 2026-05-21 |  |
| 6721 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 714 | 2019-05-21 | 2026-05-22 | 709 | 2019-05-21 | 2026-05-20 |  |
| 6722 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 839 | 2023-05-15 | 2026-05-22 | 834 | 2023-05-15 | 2026-05-21 |  |
| 6725 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1630 | 2023-05-05 | 2026-05-22 | 1625 | 2023-05-05 | 2026-05-21 |  |
| 6727 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6480 | 2019-06-10 | 2026-05-22 | 6475 | 2019-06-10 | 2026-05-21 |  |
| 6727 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6727&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6727 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6727&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6727 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6727&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6728 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=6728&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6729 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=6729&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6729 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=6729&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 6730 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 595 | 2020-12-02 | 2026-05-22 | 590 | 2020-12-02 | 2026-05-20 |  |
| 6732 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7721 | 2019-06-18 | 2026-05-22 | 7716 | 2019-06-18 | 2026-05-21 |  |
| 6733 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5788 | 2019-06-18 | 2026-05-22 | 5783 | 2019-06-18 | 2026-05-21 |  |
| 6735 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3551 | 2021-05-28 | 2026-05-22 | 3546 | 2021-05-28 | 2026-05-21 |  |
| 6738 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 714 | 2020-04-07 | 2026-05-22 | 711 | 2020-04-07 | 2026-05-15 |  |
| 6739 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2241 | 2021-09-06 | 2026-05-22 | 2236 | 2021-09-06 | 2026-05-21 |  |
| 6741 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6065 | 2021-05-25 | 2026-05-22 | 6060 | 2021-05-25 | 2026-05-21 |  |
| 6742 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3403 | 2021-11-08 | 2026-05-22 | 3398 | 2021-11-08 | 2026-05-21 |  |
| 6743 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6777 | 2019-12-06 | 2026-05-22 | 6772 | 2019-12-06 | 2026-05-21 |  |
| 6751 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3535 | 2021-01-05 | 2026-05-22 | 3530 | 2021-01-05 | 2026-05-21 |  |
| 6752 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4396 | 2020-01-10 | 2026-05-22 | 4391 | 2020-01-10 | 2026-05-21 |  |
| 6753 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4168 | 2021-02-18 | 2026-05-22 | 4163 | 2021-02-18 | 2026-05-21 |  |
| 6754 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6853 | 2020-08-20 | 2026-05-22 | 6848 | 2020-08-20 | 2026-05-21 |  |
| 6756 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6626 | 2019-12-20 | 2026-05-22 | 6621 | 2019-12-20 | 2026-05-21 |  |
| 6757 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4129 | 2019-12-16 | 2026-05-22 | 4124 | 2019-12-16 | 2026-05-21 |  |
| 6761 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5158 | 2021-08-09 | 2026-05-22 | 5153 | 2021-08-09 | 2026-05-21 |  |
| 6763 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5451 | 2020-05-04 | 2026-05-22 | 5446 | 2020-05-04 | 2026-05-21 |  |
| 6767 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3242 | 2020-05-28 | 2026-05-22 | 3237 | 2020-05-28 | 2026-05-21 |  |
| 6768 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6170 | 2021-04-22 | 2026-05-22 | 6165 | 2021-04-22 | 2026-05-21 |  |
| 6770 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6121 | 2020-12-09 | 2026-05-22 | 6116 | 2020-12-09 | 2026-05-21 |  |
| 6771 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2281 | 2020-06-19 | 2026-05-22 | 2276 | 2020-06-19 | 2026-05-21 |  |
| 6776 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6152 | 2020-05-18 | 2026-05-22 | 6147 | 2020-05-18 | 2026-05-21 |  |
| 6781 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6285 | 2021-03-22 | 2026-05-22 | 6280 | 2021-03-22 | 2026-05-21 |  |
| 6782 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4669 | 2020-09-07 | 2026-05-22 | 4664 | 2020-09-07 | 2026-05-21 |  |
| 6785 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3197 | 2022-01-12 | 2026-05-22 | 3192 | 2022-01-12 | 2026-05-21 |  |
| 6788 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5922 | 2020-10-15 | 2026-05-22 | 5917 | 2020-10-15 | 2026-05-21 |  |
| 6789 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5419 | 2021-04-06 | 2026-05-22 | 5414 | 2021-04-06 | 2026-05-21 |  |
| 6790 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5742 | 2020-10-27 | 2026-05-22 | 5737 | 2020-10-27 | 2026-05-21 |  |
| 6791 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3470 | 2022-11-16 | 2026-05-22 | 3465 | 2022-11-16 | 2026-05-21 |  |
| 6792 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5559 | 2020-11-20 | 2026-05-22 | 5554 | 2020-11-20 | 2026-05-21 |  |
| 6794 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2125 | 2023-05-15 | 2026-05-22 | 2120 | 2023-05-15 | 2026-05-21 |  |
| 6796 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4759 | 2021-05-28 | 2026-05-22 | 4754 | 2021-05-28 | 2026-05-21 |  |
| 6799 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5388 | 2020-11-27 | 2026-05-22 | 5383 | 2020-11-27 | 2026-05-21 |  |
| 6803 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15247 | 2008-09-17 | 2026-05-22 | 15242 | 2008-09-17 | 2026-05-21 |  |
| 6804 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4050 | 2022-01-13 | 2026-05-22 | 4045 | 2022-01-13 | 2026-05-21 |  |
| 6805 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4626 | 2021-01-07 | 2026-05-22 | 4621 | 2021-01-07 | 2026-05-21 |  |
| 6806 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5686 | 2020-12-07 | 2026-05-22 | 5681 | 2020-12-07 | 2026-05-21 |  |
| 6807 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4235 | 2022-04-26 | 2026-05-22 | 4230 | 2022-04-26 | 2026-05-21 |  |
| 6811 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4567 | 2021-01-19 | 2026-05-22 | 4562 | 2021-01-19 | 2026-05-21 |  |
| 6821 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3526 | 2023-04-13 | 2026-05-22 | 3521 | 2023-04-13 | 2026-05-21 |  |
| 6823 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3711 | 2021-06-17 | 2026-05-22 | 3706 | 2021-06-17 | 2026-05-21 |  |
| 6829 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5083 | 2021-11-18 | 2026-05-22 | 5078 | 2021-11-18 | 2026-05-21 |  |
| 6830 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4794 | 2021-07-26 | 2026-05-22 | 4789 | 2021-07-26 | 2026-05-21 |  |
| 6831 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1732 | 2021-12-14 | 2026-05-22 | 1727 | 2021-12-14 | 2026-05-21 |  |
| 6833 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 663 | 2021-06-09 | 2026-05-22 | 660 | 2021-06-09 | 2026-05-06 |  |
| 6834 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4262 | 2021-08-31 | 2026-05-22 | 4257 | 2021-08-31 | 2026-05-21 |  |
| 6835 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4208 | 2021-09-27 | 2026-05-22 | 4203 | 2021-09-27 | 2026-05-21 |  |
| 6838 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2821 | 2022-02-16 | 2026-05-22 | 2816 | 2022-02-16 | 2026-05-21 |  |
| 6840 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2778 | 2022-01-04 | 2026-05-22 | 2773 | 2022-01-04 | 2026-05-21 |  |
| 6841 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4060 | 2021-09-22 | 2026-05-22 | 4055 | 2021-09-22 | 2026-05-21 |  |
| 6843 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3400 | 2022-11-22 | 2026-05-22 | 3395 | 2022-11-22 | 2026-05-20 |  |
| 6846 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2150 | 2023-05-16 | 2026-05-22 | 2145 | 2023-05-16 | 2026-05-20 |  |
| 6854 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4325 | 2022-08-18 | 2026-05-22 | 4320 | 2022-08-18 | 2026-05-21 |  |
| 6855 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2425 | 2022-10-07 | 2026-05-22 | 2420 | 2022-10-07 | 2026-05-21 |  |
| 6858 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 165 | 2022-02-22 | 2026-05-22 | 162 | 2022-02-22 | 2026-04-24 |  |
| 6859 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4027 | 2021-12-10 | 2026-05-22 | 4022 | 2021-12-10 | 2026-05-21 |  |
| 6861 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3842 | 2021-11-30 | 2026-05-22 | 3837 | 2021-11-30 | 2026-05-21 |  |
| 6862 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1905 | 2024-10-23 | 2026-05-22 | 1900 | 2024-10-23 | 2026-05-21 |  |
| 6863 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3845 | 2023-03-21 | 2026-05-22 | 3840 | 2023-03-21 | 2026-05-21 |  |
| 6865 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2835 | 2022-04-20 | 2026-05-22 | 2830 | 2022-04-20 | 2026-05-21 |  |
| 6869 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3981 | 2022-01-20 | 2026-05-22 | 3976 | 2022-01-20 | 2026-05-21 |  |
| 6870 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3270 | 2022-02-17 | 2026-05-22 | 3265 | 2022-02-17 | 2026-05-21 |  |
| 6873 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4029 | 2022-01-24 | 2026-05-22 | 4024 | 2022-01-24 | 2026-05-21 |  |
| 6874 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3870 | 2022-12-14 | 2026-05-22 | 3865 | 2022-12-14 | 2026-05-21 |  |
| 6875 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3045 | 2022-11-07 | 2026-05-22 | 3040 | 2022-11-07 | 2026-05-21 |  |
| 6877 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3332 | 2022-03-04 | 2026-05-22 | 3327 | 2022-03-04 | 2026-05-21 |  |
| 6884 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 265 | 2023-09-25 | 2026-05-22 | 260 | 2023-09-25 | 2026-05-18 |  |
| 6885 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2953 | 2022-08-04 | 2026-05-22 | 2948 | 2022-08-04 | 2026-05-21 |  |
| 6887 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1355 | 2025-03-10 | 2026-05-22 | 1350 | 2025-03-10 | 2026-05-21 |  |
| 6890 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2355 | 2024-06-12 | 2026-05-22 | 2350 | 2024-06-12 | 2026-05-21 |  |
| 6894 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2423 | 2023-09-26 | 2026-05-22 | 2418 | 2023-09-26 | 2026-05-21 |  |
| 6895 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3333 | 2023-01-31 | 2026-05-22 | 3328 | 2023-01-31 | 2026-05-21 |  |
| 6899 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1997 | 2022-07-06 | 2026-05-22 | 1992 | 2022-07-06 | 2026-05-21 |  |
| 6901 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3725 | 2022-07-25 | 2026-05-22 | 3720 | 2022-07-25 | 2026-05-21 |  |
| 6902 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2465 | 2023-07-13 | 2026-05-22 | 2460 | 2023-07-13 | 2026-05-21 |  |
| 6903 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2971 | 2022-07-27 | 2026-05-22 | 2966 | 2022-07-27 | 2026-05-21 |  |
| 6906 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3066 | 2022-12-15 | 2026-05-22 | 3061 | 2022-12-15 | 2026-05-21 |  |
| 6907 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 355 | 2026-01-29 | 2026-05-22 | 350 | 2026-01-29 | 2026-05-21 |  |
| 6908 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 348 | 2023-09-25 | 2026-05-22 | 343 | 2023-09-25 | 2026-05-21 |  |
| 6909 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2032 | 2023-03-30 | 2026-05-22 | 2027 | 2023-03-30 | 2026-05-21 |  |
| 6910 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1186 | 2023-05-25 | 2026-05-22 | 1181 | 2023-05-25 | 2026-05-21 |  |
| 6911 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 387 | 2023-03-29 | 2026-05-22 | 384 | 2023-03-29 | 2026-05-11 |  |
| 6913 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2280 | 2023-01-09 | 2026-05-22 | 2275 | 2023-01-09 | 2026-05-21 |  |
| 6913 | TaiwanStockDividend | MISMATCH | 4 | 2023-06-05 | 2026-06-16 | 3 | 2023-06-05 | 2025-07-07 |  |
| 6914 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2821 | 2023-02-24 | 2026-05-22 | 2816 | 2023-02-24 | 2026-05-21 |  |
| 6916 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2898 | 2023-09-19 | 2026-05-22 | 2893 | 2023-09-19 | 2026-05-21 |  |
| 6918 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1663 | 2023-01-06 | 2026-05-22 | 1658 | 2023-01-06 | 2026-05-21 |  |
| 6919 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2836 | 2023-01-03 | 2026-05-22 | 2831 | 2023-01-03 | 2026-05-21 |  |
| 6921 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 475 | 2025-12-24 | 2026-05-22 | 470 | 2025-12-24 | 2026-05-21 |  |
| 6922 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3036 | 2023-03-07 | 2026-05-22 | 3031 | 2023-03-07 | 2026-05-21 |  |
| 6923 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2477 | 2023-03-30 | 2026-05-22 | 2472 | 2023-03-30 | 2026-05-21 |  |
| 6924 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1360 | 2024-12-18 | 2026-05-22 | 1355 | 2024-12-18 | 2026-05-21 |  |
| 6925 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1638 | 2023-01-10 | 2026-05-22 | 1633 | 2023-01-10 | 2026-05-21 |  |
| 6928 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2579 | 2023-05-03 | 2026-05-22 | 2574 | 2023-05-03 | 2026-05-21 |  |
| 6929 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1227 | 2024-01-04 | 2026-05-22 | 1222 | 2024-01-04 | 2026-05-21 |  |
| 6931 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1987 | 2023-09-21 | 2026-05-22 | 1982 | 2023-09-21 | 2026-05-21 |  |
| 6932 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1350 | 2023-02-03 | 2026-05-22 | 1347 | 2023-02-03 | 2026-04-29 |  |
| 6933 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3065 | 2023-11-08 | 2026-05-22 | 3060 | 2023-11-08 | 2026-05-21 |  |
| 6934 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 444 | 2024-06-26 | 2026-05-22 | 439 | 2024-06-26 | 2026-05-21 |  |
| 6936 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1332 | 2023-03-24 | 2026-05-22 | 1327 | 2023-03-24 | 2026-05-21 |  |
| 6937 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2969 | 2023-11-15 | 2026-05-22 | 2964 | 2023-11-15 | 2026-05-21 |  |
| 6944 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2346 | 2023-04-27 | 2026-05-22 | 2341 | 2023-04-27 | 2026-05-21 |  |
| 6949 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2510 | 2024-03-08 | 2026-05-22 | 2505 | 2024-03-08 | 2026-05-21 |  |
| 6951 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2386 | 2023-05-25 | 2026-05-22 | 2381 | 2023-05-25 | 2026-05-21 |  |
| 6951 | TaiwanStockDividend | MISMATCH | 3 | 2024-06-10 | 2026-06-18 | 2 | 2024-06-10 | 2025-06-15 |  |
| 6952 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2265 | 2023-11-03 | 2026-05-22 | 2260 | 2023-11-03 | 2026-05-21 |  |
| 6953 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2572 | 2023-06-27 | 2026-05-22 | 2567 | 2023-06-27 | 2026-05-21 |  |
| 6957 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2305 | 2024-06-25 | 2026-05-22 | 2300 | 2024-06-25 | 2026-05-21 |  |
| 6958 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2196 | 2023-12-26 | 2026-05-22 | 2191 | 2023-12-26 | 2026-05-21 |  |
| 6962 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1795 | 2024-11-26 | 2026-05-22 | 1790 | 2024-11-26 | 2026-05-21 |  |
| 6965 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1475 | 2025-03-07 | 2026-05-22 | 1470 | 2025-03-07 | 2026-05-21 |  |
| 6967 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1638 | 2023-11-10 | 2026-05-22 | 1633 | 2023-11-10 | 2026-05-21 |  |
| 6968 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2113 | 2023-09-01 | 2026-05-22 | 2108 | 2023-09-01 | 2026-05-21 |  |
| 6969 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1905 | 2024-09-09 | 2026-05-22 | 1900 | 2024-09-09 | 2026-05-21 |  |
| 6971 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 858 | 2024-03-04 | 2026-05-22 | 853 | 2024-03-04 | 2026-05-21 |  |
| 6982 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1885 | 2023-09-28 | 2026-05-22 | 1880 | 2023-09-28 | 2026-05-21 |  |
| 6988 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1605 | 2024-11-04 | 2026-05-22 | 1600 | 2024-11-04 | 2026-05-21 |  |
| 6994 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1769 | 2024-01-31 | 2026-05-22 | 1764 | 2024-01-31 | 2026-05-21 |  |
| 6996 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1853 | 2023-12-20 | 2026-05-22 | 1848 | 2023-12-20 | 2026-05-21 |  |
| 6997 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 758 | 2024-01-11 | 2026-05-22 | 753 | 2024-01-11 | 2026-05-20 |  |
| 7402 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5843 | 2015-12-02 | 2026-05-22 | 5838 | 2015-12-02 | 2026-05-21 |  |
| 7547 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1484 | 2021-02-22 | 2026-05-22 | 1479 | 2021-02-22 | 2026-05-21 |  |
| 7556 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3956 | 2020-06-29 | 2026-05-22 | 3951 | 2020-06-29 | 2026-05-21 |  |
| 7561 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 246 | 2021-09-14 | 2026-05-22 | 243 | 2021-09-14 | 2026-05-12 |  |
| 7566 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 951 | 2020-07-01 | 2026-05-22 | 948 | 2020-07-01 | 2026-05-13 |  |
| 7584 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1819 | 2021-11-02 | 2026-05-22 | 1814 | 2021-11-02 | 2026-05-21 |  |
| 7607 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 363 | 2024-02-26 | 2026-05-22 | 360 | 2024-02-26 | 2026-05-15 |  |
| 7610 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1345 | 2023-10-04 | 2026-05-22 | 1340 | 2023-10-04 | 2026-05-21 |  |
| 7631 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1440 | 2024-01-23 | 2026-05-22 | 1435 | 2024-01-23 | 2026-05-21 |  |
| 7642 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1254 | 2024-03-20 | 2026-05-22 | 1249 | 2024-03-20 | 2026-05-21 |  |
| 7703 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1839 | 2024-02-16 | 2026-05-22 | 1834 | 2024-02-16 | 2026-05-21 |  |
| 7704 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1501 | 2024-02-02 | 2026-05-22 | 1496 | 2024-02-02 | 2026-05-21 |  |
| 7705 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1795 | 2024-01-10 | 2026-05-22 | 1790 | 2024-01-10 | 2026-05-21 |  |
| 7708 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 535 | 2024-11-15 | 2026-05-22 | 530 | 2024-11-15 | 2026-05-21 |  |
| 7709 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 899 | 2024-01-10 | 2026-05-22 | 894 | 2024-01-10 | 2026-05-21 |  |
| 7711 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 717 | 2025-02-11 | 2026-05-22 | 712 | 2025-02-11 | 2026-05-21 |  |
| 7712 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1973 | 2024-01-04 | 2026-05-22 | 1968 | 2024-01-04 | 2026-05-21 |  |
| 7713 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 930 | 2025-02-25 | 2026-05-22 | 925 | 2025-02-25 | 2026-05-21 |  |
| 7714 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1603 | 2024-01-08 | 2026-05-22 | 1598 | 2024-01-08 | 2026-05-21 |  |
| 7715 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1625 | 2024-01-26 | 2026-05-22 | 1620 | 2024-01-26 | 2026-05-21 |  |
| 7716 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 130 | 2024-04-16 | 2026-05-22 | 125 | 2024-04-16 | 2026-05-14 |  |
| 7717 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 600 | 2025-11-19 | 2026-05-22 | 595 | 2025-11-19 | 2026-05-21 |  |
| 7717 | TaiwanStockMarginPurchaseShortSale | MISMATCH | 1 | 2026-05-21 | 2026-05-21 | 0 | None | None |  |
| 7717 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7717&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7717 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7717&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7717 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7717&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7717 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7717&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7718 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7718&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7719 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7719&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7721 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7721&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7722 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7722&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7723 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7723&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7723 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7723&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7723 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1779 | 2024-01-25 | 2026-05-22 | 1774 | 2024-01-25 | 2026-05-21 |  |
| 7728 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1649 | 2024-03-25 | 2026-05-22 | 1644 | 2024-03-25 | 2026-05-21 |  |
| 7730 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 675 | 2024-04-08 | 2026-05-22 | 670 | 2024-04-08 | 2026-05-21 |  |
| 7732 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1443 | 2024-07-04 | 2026-05-22 | 1438 | 2024-07-04 | 2026-05-21 |  |
| 7734 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1864 | 2024-03-14 | 2026-05-22 | 1859 | 2024-03-14 | 2026-05-21 |  |
| 7736 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1504 | 2024-04-02 | 2026-05-22 | 1499 | 2024-04-02 | 2026-05-21 |  |
| 7737 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 99 | 2025-02-07 | 2026-05-22 | 96 | 2025-02-07 | 2026-05-14 |  |
| 7738 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 925 | 2024-06-13 | 2026-05-22 | 920 | 2024-06-13 | 2026-05-21 |  |
| 7740 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1212 | 2024-11-15 | 2026-05-22 | 1207 | 2024-11-15 | 2026-05-21 |  |
| 7740 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7740&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7740 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7740&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7740 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7740&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7740 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7740&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7742 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7742&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7743 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7743&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7744 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7744&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7744 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7744&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7744 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7744&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7744 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7744&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7744 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7744&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7744 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7744&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7744 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7744&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7749 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1145 | 2025-06-13 | 2026-05-22 | 1140 | 2025-06-13 | 2026-05-21 |  |
| 7750 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1402 | 2024-06-18 | 2026-05-22 | 1397 | 2024-06-18 | 2026-05-21 |  |
| 7751 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1263 | 2024-07-04 | 2026-05-22 | 1258 | 2024-07-04 | 2026-05-21 |  |
| 7753 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 772 | 2024-08-29 | 2026-05-22 | 767 | 2024-08-29 | 2026-05-21 |  |
| 7757 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 641 | 2024-08-15 | 2026-05-22 | 636 | 2024-08-15 | 2026-05-21 |  |
| 7757 | TaiwanStockDividend | MISMATCH | 2 | 2025-07-01 | 2026-06-15 | 1 | 2025-07-01 | 2025-07-01 |  |
| 7759 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 111 | 2025-03-10 | 2026-05-22 | 108 | 2025-03-10 | 2026-04-29 |  |
| 7760 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 112 | 2025-02-21 | 2026-05-22 | 107 | 2025-02-21 | 2026-05-21 |  |
| 7762 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7762&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7762 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7762&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7762 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7762&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7762 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7762&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7762 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7762&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7762 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7762&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7762 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7762&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7763 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7763&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7764 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7764&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7765 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7765&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7767 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7767&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7768 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7768&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7769 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7769&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7770 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7770&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7771 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7771&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7772 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7772&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7773 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7773&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7776 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7776&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7777 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7777&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7779 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7779&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7780 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7780&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7781 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=7781&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7782 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=7782&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7782 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=7782&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7782 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=7782&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7782 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=7782&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7782 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=7782&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 7786 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1088 | 2024-10-30 | 2026-05-22 | 1083 | 2024-10-30 | 2026-05-21 |  |
| 7788 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 763 | 2024-11-06 | 2026-05-22 | 758 | 2024-11-06 | 2026-05-21 |  |
| 7791 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 755 | 2024-11-26 | 2026-05-22 | 750 | 2024-11-26 | 2026-05-21 |  |
| 7792 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 453 | 2024-12-19 | 2026-05-22 | 448 | 2024-12-19 | 2026-05-21 |  |
| 7794 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 100 | 2026-04-21 | 2026-05-22 | 95 | 2026-04-21 | 2026-05-21 |  |
| 7795 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 613 | 2024-12-02 | 2026-05-22 | 608 | 2024-12-02 | 2026-05-21 |  |
| 7799 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1148 | 2024-11-27 | 2026-05-22 | 1143 | 2024-11-27 | 2026-05-21 |  |
| 7803 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 37 | 2024-11-28 | 2026-05-22 | 32 | 2024-11-28 | 2026-05-21 |  |
| 7805 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 674 | 2025-02-10 | 2026-05-22 | 669 | 2025-02-10 | 2026-05-21 |  |
| 7810 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 573 | 2025-07-15 | 2026-05-22 | 568 | 2025-07-15 | 2026-05-21 |  |
| 7811 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 221 | 2025-01-09 | 2026-05-22 | 216 | 2025-01-09 | 2026-05-21 |  |
| 7818 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 110 | 2025-02-07 | 2026-05-22 | 105 | 2025-02-07 | 2026-05-21 |  |
| 7820 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 227 | 2025-06-13 | 2026-05-22 | 222 | 2025-06-13 | 2026-05-21 |  |
| 7821 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 216 | 2025-05-08 | 2026-05-22 | 211 | 2025-05-08 | 2026-05-21 |  |
| 7822 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 581 | 2025-03-27 | 2026-05-22 | 576 | 2025-03-27 | 2026-05-21 |  |
| 7823 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 310 | 2026-02-05 | 2026-05-22 | 305 | 2026-02-05 | 2026-05-21 |  |
| 7828 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 701 | 2025-05-12 | 2026-05-22 | 696 | 2025-05-12 | 2026-05-21 |  |
| 7842 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 107 | 2025-06-10 | 2026-05-22 | 102 | 2025-06-10 | 2026-05-21 |  |
| 7858 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 192 | 2025-10-15 | 2026-05-22 | 189 | 2025-10-15 | 2026-05-14 |  |
| 7901 | TaiwanStockDividend | MISMATCH | 1 | 2026-06-29 | 2026-06-29 | 0 | None | None |  |
| 7907 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18 | 2026-04-22 | 2026-05-22 | 15 | 2026-04-22 | 2026-05-11 |  |
| 7913 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6 | 2026-05-15 | 2026-05-22 | 3 | 2026-05-15 | 2026-05-15 |  |
| 7918 | TaiwanStockFinancialStatements | MISMATCH | 14 | 2025-12-31 | 2025-12-31 | 1 | 2025-12-31 | 2025-12-31 |  |
| 7920 | TaiwanStockPriceAdj | MISMATCH | 2 | 2026-05-21 | 2026-05-22 | 0 | None | None |  |
| 8011 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15405 | 2010-03-15 | 2026-05-22 | 15400 | 2010-03-15 | 2026-05-21 |  |
| 8016 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15166 | 2012-05-02 | 2026-05-22 | 15161 | 2012-05-02 | 2026-05-21 |  |
| 8021 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17093 | 2005-01-03 | 2026-05-22 | 17088 | 2005-01-03 | 2026-05-21 |  |
| 8024 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9605 | 2006-09-25 | 2026-05-22 | 9600 | 2006-09-25 | 2026-05-21 |  |
| 8027 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12542 | 2008-11-13 | 2026-05-22 | 12537 | 2008-11-13 | 2026-05-21 |  |
| 8028 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10575 | 2007-06-21 | 2026-05-22 | 10570 | 2007-06-21 | 2026-05-21 |  |
| 8032 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7364 | 2005-02-24 | 2026-05-22 | 7359 | 2005-02-24 | 2026-05-21 |  |
| 8033 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15421 | 2005-01-19 | 2026-05-22 | 15416 | 2005-01-19 | 2026-05-21 |  |
| 8034 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12566 | 2005-01-05 | 2026-05-22 | 12561 | 2005-01-05 | 2026-05-21 |  |
| 8038 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13560 | 2007-10-01 | 2026-05-22 | 13555 | 2007-10-01 | 2026-05-21 |  |
| 8039 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18423 | 2005-01-03 | 2026-05-22 | 18418 | 2005-01-03 | 2026-05-21 |  |
| 8040 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13940 | 2007-04-30 | 2026-05-22 | 13935 | 2007-04-30 | 2026-05-21 |  |
| 8042 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17384 | 2005-03-07 | 2026-05-22 | 17379 | 2005-03-07 | 2026-05-21 |  |
| 8043 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13138 | 2005-01-03 | 2026-05-22 | 13133 | 2005-01-03 | 2026-05-21 |  |
| 8044 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18769 | 2005-01-28 | 2026-05-22 | 18764 | 2005-01-28 | 2026-05-21 |  |
| 8045 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2507 | 2008-09-18 | 2026-05-22 | 2502 | 2008-09-18 | 2026-05-21 |  |
| 8046 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15163 | 2012-05-02 | 2026-05-22 | 15158 | 2012-05-02 | 2026-05-21 |  |
| 8047 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5743 | 2005-01-12 | 2026-05-22 | 5738 | 2005-01-12 | 2026-05-21 |  |
| 8048 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12186 | 2007-06-22 | 2026-05-22 | 12181 | 2007-06-22 | 2026-05-21 |  |
| 8049 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16183 | 2005-01-13 | 2026-05-22 | 16178 | 2005-01-13 | 2026-05-21 |  |
| 8050 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18623 | 2005-01-03 | 2026-05-22 | 18618 | 2005-01-03 | 2026-05-21 |  |
| 8054 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18421 | 2005-01-11 | 2026-05-22 | 18416 | 2005-01-11 | 2026-05-21 |  |
| 8058 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 261 | 2025-08-12 | 2026-05-22 | 258 | 2025-08-12 | 2026-05-15 |  |
| 8059 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12050 | 2008-11-17 | 2026-05-22 | 12045 | 2008-11-17 | 2026-05-21 |  |
| 8064 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17170 | 2006-09-25 | 2026-05-22 | 17165 | 2006-09-25 | 2026-05-21 |  |
| 8066 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12035 | 2005-01-05 | 2026-05-22 | 12030 | 2005-01-05 | 2026-05-21 |  |
| 8068 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8828 | 2005-01-04 | 2026-05-22 | 8823 | 2005-01-04 | 2026-05-21 |  |
| 8069 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19955 | 2005-01-06 | 2026-05-22 | 19950 | 2005-01-06 | 2026-05-21 |  |
| 8070 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16859 | 2005-01-03 | 2026-05-22 | 16854 | 2005-01-03 | 2026-05-21 |  |
| 8071 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10701 | 2005-12-27 | 2026-05-22 | 10696 | 2005-12-27 | 2026-05-21 |  |
| 8072 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12876 | 2005-01-12 | 2026-05-22 | 12871 | 2005-01-12 | 2026-05-21 |  |
| 8074 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13024 | 2005-03-15 | 2026-05-22 | 13019 | 2005-03-15 | 2026-05-21 |  |
| 8076 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19844 | 2005-01-10 | 2026-05-22 | 19839 | 2005-01-10 | 2026-05-21 |  |
| 8080 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6916 | 2005-01-03 | 2026-05-22 | 6911 | 2005-01-03 | 2026-05-21 |  |
| 8081 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18077 | 2005-01-04 | 2026-05-22 | 18072 | 2005-01-04 | 2026-05-21 |  |
| 8083 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12349 | 2010-04-06 | 2026-05-22 | 12344 | 2010-04-06 | 2026-05-21 |  |
| 8084 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9342 | 2005-01-25 | 2026-05-22 | 9337 | 2005-01-25 | 2026-05-21 |  |
| 8085 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13621 | 2005-01-03 | 2026-05-22 | 13616 | 2005-01-03 | 2026-05-21 |  |
| 8086 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17193 | 2007-07-09 | 2026-05-22 | 17188 | 2007-07-09 | 2026-05-21 |  |
| 8087 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6351 | 2005-03-22 | 2026-05-22 | 6346 | 2005-03-22 | 2026-05-21 |  |
| 8088 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15018 | 2005-01-04 | 2026-05-22 | 15013 | 2005-01-04 | 2026-05-21 |  |
| 8089 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6801 | 2018-09-12 | 2026-05-22 | 6796 | 2018-09-12 | 2026-05-21 |  |
| 8091 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16465 | 2005-01-17 | 2026-05-22 | 16460 | 2005-01-17 | 2026-05-21 |  |
| 8092 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9534 | 2005-01-24 | 2026-05-22 | 9529 | 2005-01-24 | 2026-05-21 |  |
| 8093 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8698 | 2005-01-03 | 2026-05-22 | 8693 | 2005-01-03 | 2026-05-15 |  |
| 8096 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18144 | 2005-01-04 | 2026-05-22 | 18139 | 2005-01-04 | 2026-05-21 |  |
| 8097 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9697 | 2005-01-12 | 2026-05-22 | 9692 | 2005-01-12 | 2026-05-21 |  |
| 8099 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11339 | 2005-03-03 | 2026-05-22 | 11334 | 2005-03-03 | 2026-05-21 |  |
| 8102 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 315 | 2024-02-27 | 2026-05-22 | 310 | 2024-02-27 | 2026-05-21 |  |
| 8103 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17657 | 2005-03-10 | 2026-05-22 | 17652 | 2005-03-10 | 2026-05-21 |  |
| 8104 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9137 | 2016-07-26 | 2026-05-22 | 9132 | 2016-07-26 | 2026-05-21 |  |
| 8105 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16263 | 2005-01-03 | 2026-05-22 | 16258 | 2005-01-03 | 2026-05-21 |  |
| 8107 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15890 | 2005-11-15 | 2026-05-22 | 15885 | 2005-11-15 | 2026-05-21 |  |
| 8109 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14985 | 2005-03-22 | 2026-05-22 | 14980 | 2005-03-22 | 2026-05-21 |  |
| 8110 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15841 | 2006-08-23 | 2026-05-22 | 15836 | 2006-08-23 | 2026-05-21 |  |
| 8111 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15662 | 2005-01-03 | 2026-05-22 | 15657 | 2005-01-03 | 2026-05-21 |  |
| 8112 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16367 | 2005-01-28 | 2026-05-22 | 16362 | 2005-01-28 | 2026-05-21 |  |
| 8114 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17396 | 2005-07-21 | 2026-05-22 | 17391 | 2005-07-21 | 2026-05-21 |  |
| 8121 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17876 | 2005-02-17 | 2026-05-22 | 17871 | 2005-02-17 | 2026-05-21 |  |
| 8131 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15330 | 2007-06-20 | 2026-05-22 | 15325 | 2007-06-20 | 2026-05-21 |  |
| 8147 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9879 | 2013-01-16 | 2026-05-22 | 9874 | 2013-01-16 | 2026-05-21 |  |
| 8150 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13981 | 2013-05-15 | 2026-05-22 | 13976 | 2013-05-15 | 2026-05-21 |  |
| 8155 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13440 | 2012-12-24 | 2026-05-22 | 13435 | 2012-12-24 | 2026-05-21 |  |
| 8162 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2639 | 2022-09-29 | 2026-05-22 | 2634 | 2022-09-29 | 2026-05-21 |  |
| 8163 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15368 | 2007-06-20 | 2026-05-22 | 15363 | 2007-06-20 | 2026-05-21 |  |
| 8171 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11946 | 2010-10-12 | 2026-05-22 | 11941 | 2010-10-12 | 2026-05-21 |  |
| 8176 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10919 | 2007-06-20 | 2026-05-22 | 10914 | 2007-06-20 | 2026-05-21 |  |
| 8182 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13656 | 2006-05-25 | 2026-05-22 | 13651 | 2006-05-25 | 2026-05-21 |  |
| 8183 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14231 | 2005-04-01 | 2026-05-22 | 14226 | 2005-04-01 | 2026-05-21 |  |
| 8201 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13198 | 2007-10-18 | 2026-05-22 | 13193 | 2007-10-18 | 2026-05-21 |  |
| 8210 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17085 | 2005-05-04 | 2026-05-22 | 17080 | 2005-05-04 | 2026-05-21 |  |
| 8213 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15665 | 2007-06-25 | 2026-05-22 | 15660 | 2007-06-25 | 2026-05-21 |  |
| 8215 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15856 | 2007-06-26 | 2026-05-22 | 15851 | 2007-06-26 | 2026-05-21 |  |
| 8222 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12403 | 2013-12-18 | 2026-05-22 | 12398 | 2013-12-18 | 2026-05-21 |  |
| 8227 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4733 | 2007-10-03 | 2026-05-22 | 4728 | 2007-10-03 | 2026-05-21 |  |
| 8234 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15806 | 2007-06-07 | 2026-05-22 | 15801 | 2007-06-07 | 2026-05-21 |  |
| 8240 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16720 | 2005-06-28 | 2026-05-22 | 16715 | 2005-06-28 | 2026-05-21 |  |
| 8249 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14930 | 2012-05-02 | 2026-05-22 | 14925 | 2012-05-02 | 2026-05-21 |  |
| 8255 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18663 | 2006-04-25 | 2026-05-22 | 18658 | 2006-04-25 | 2026-05-21 |  |
| 8261 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17813 | 2005-01-03 | 2026-05-22 | 17808 | 2005-01-03 | 2026-05-21 |  |
| 8271 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15205 | 2007-10-09 | 2026-05-22 | 15200 | 2007-10-09 | 2026-05-21 |  |
| 8272 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 1306 | 2023-12-14 | 2026-05-22 | 1301 | 2023-12-14 | 2026-05-21 |  |
| 8277 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16030 | 2005-01-03 | 2026-05-22 | 16025 | 2005-01-03 | 2026-05-21 |  |
| 8279 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8629 | 2015-08-18 | 2026-05-22 | 8624 | 2015-08-18 | 2026-05-21 |  |
| 8284 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5539 | 2014-03-07 | 2026-05-22 | 5534 | 2014-03-07 | 2026-05-21 |  |
| 8289 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11119 | 2007-07-06 | 2026-05-22 | 11114 | 2007-07-06 | 2026-05-21 |  |
| 8289 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8289&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8291 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8291&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8298 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8298&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8299 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8299&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8329 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8329&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8341 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8341&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8342 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8342&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8342 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8342&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8342 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8342&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8349 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13124 | 2006-03-21 | 2026-05-22 | 13119 | 2006-03-21 | 2026-05-21 |  |
| 8354 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6712 | 2007-01-26 | 2026-05-22 | 6707 | 2007-01-26 | 2026-05-21 |  |
| 8358 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13201 | 2007-11-12 | 2026-05-22 | 13196 | 2007-11-12 | 2026-05-21 |  |
| 8367 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8752 | 2017-04-07 | 2026-05-22 | 8747 | 2017-04-07 | 2026-05-21 |  |
| 8374 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15155 | 2005-02-21 | 2026-05-22 | 15150 | 2005-02-21 | 2026-05-21 |  |
| 8383 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17203 | 2005-02-18 | 2026-05-22 | 17198 | 2005-02-18 | 2026-05-21 |  |
| 8390 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13267 | 2007-06-27 | 2026-05-22 | 13262 | 2007-06-27 | 2026-05-21 |  |
| 8401 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9607 | 2008-12-10 | 2026-05-22 | 9602 | 2008-12-10 | 2026-05-21 |  |
| 8403 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14752 | 2008-12-24 | 2026-05-22 | 14747 | 2008-12-24 | 2026-05-21 |  |
| 8404 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14935 | 2012-05-02 | 2026-05-22 | 14930 | 2012-05-02 | 2026-05-21 |  |
| 8409 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6994 | 2012-11-21 | 2026-05-22 | 6989 | 2012-11-21 | 2026-05-21 |  |
| 8410 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11433 | 2009-11-03 | 2026-05-22 | 11428 | 2009-11-03 | 2026-05-21 |  |
| 8411 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14058 | 2009-12-17 | 2026-05-22 | 14053 | 2009-12-17 | 2026-05-21 |  |
| 8415 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11167 | 2011-02-21 | 2026-05-22 | 11162 | 2011-02-21 | 2026-05-21 |  |
| 8422 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15178 | 2010-10-26 | 2026-05-22 | 15173 | 2010-10-26 | 2026-05-21 |  |
| 8423 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3322 | 2012-03-22 | 2026-05-22 | 3317 | 2012-03-22 | 2026-05-21 |  |
| 8424 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6122 | 2010-12-02 | 2026-05-22 | 6117 | 2010-12-02 | 2026-05-20 |  |
| 8426 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10087 | 2011-07-07 | 2026-05-22 | 10082 | 2011-07-07 | 2026-05-21 |  |
| 8429 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14624 | 2012-12-20 | 2026-05-22 | 14619 | 2012-12-20 | 2026-05-21 |  |
| 8431 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12927 | 2013-02-05 | 2026-05-22 | 12922 | 2013-02-05 | 2026-05-21 |  |
| 8432 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12027 | 2011-06-16 | 2026-05-22 | 12022 | 2011-06-16 | 2026-05-21 |  |
| 8433 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11020 | 2012-12-19 | 2026-05-22 | 11015 | 2012-12-19 | 2026-05-21 |  |
| 8435 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7562 | 2013-01-14 | 2026-05-22 | 7557 | 2013-01-14 | 2026-05-21 |  |
| 8436 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14346 | 2012-03-06 | 2026-05-22 | 14341 | 2012-03-06 | 2026-05-21 |  |
| 8437 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12503 | 2015-05-06 | 2026-05-22 | 12498 | 2015-05-06 | 2026-05-21 |  |
| 8438 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4765 | 2016-07-25 | 2026-05-22 | 4760 | 2016-07-25 | 2026-05-21 |  |
| 8440 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7842 | 2018-01-09 | 2026-05-22 | 7837 | 2018-01-09 | 2026-05-21 |  |
| 8442 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8978 | 2016-11-08 | 2026-05-22 | 8973 | 2016-11-08 | 2026-05-21 |  |
| 8444 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9244 | 2013-11-29 | 2026-05-22 | 9239 | 2013-11-29 | 2026-05-21 |  |
| 8446 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13422 | 2013-08-06 | 2026-05-22 | 13417 | 2013-08-06 | 2026-05-21 |  |
| 8450 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10126 | 2013-10-02 | 2026-05-22 | 10121 | 2013-10-02 | 2026-05-21 |  |
| 8454 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13402 | 2014-02-27 | 2026-05-22 | 13397 | 2014-02-27 | 2026-05-21 |  |
| 8455 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2565 | 2016-01-08 | 2026-05-22 | 2560 | 2016-01-08 | 2026-05-21 |  |
| 8458 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 522 | 2016-06-21 | 2026-05-22 | 519 | 2016-06-21 | 2026-04-23 |  |
| 8462 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11836 | 2015-12-10 | 2026-05-22 | 11831 | 2015-12-10 | 2026-05-21 |  |
| 8463 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11493 | 2015-05-26 | 2026-05-22 | 11488 | 2015-05-26 | 2026-05-21 |  |
| 8464 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12386 | 2015-02-03 | 2026-05-22 | 12381 | 2015-02-03 | 2026-05-21 |  |
| 8466 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10225 | 2016-02-05 | 2026-05-22 | 10220 | 2016-02-05 | 2026-05-21 |  |
| 8467 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8618 | 2015-12-15 | 2026-05-22 | 8613 | 2015-12-15 | 2026-05-21 |  |
| 8472 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3638 | 2015-02-11 | 2026-05-22 | 3633 | 2015-02-11 | 2026-05-21 |  |
| 8473 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11479 | 2015-04-09 | 2026-05-22 | 11474 | 2015-04-09 | 2026-05-21 |  |
| 8476 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8149 | 2015-08-28 | 2026-05-22 | 8144 | 2015-08-28 | 2026-05-21 |  |
| 8477 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6569 | 2016-06-23 | 2026-05-22 | 6564 | 2016-06-23 | 2026-05-21 |  |
| 8478 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9882 | 2016-07-20 | 2026-05-22 | 9877 | 2016-07-20 | 2026-05-21 |  |
| 8481 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8223 | 2016-10-04 | 2026-05-22 | 8218 | 2016-10-04 | 2026-05-21 |  |
| 8482 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8300 | 2018-08-15 | 2026-05-22 | 8295 | 2018-08-15 | 2026-05-20 |  |
| 8487 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 2409 | 2015-12-23 | 2026-05-22 | 2404 | 2015-12-23 | 2026-05-20 |  |
| 8488 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7089 | 2016-12-13 | 2026-05-22 | 7084 | 2016-12-13 | 2026-05-21 |  |
| 8489 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10610 | 2016-05-23 | 2026-05-22 | 10605 | 2016-05-23 | 2026-05-21 |  |
| 8499 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9709 | 2017-11-24 | 2026-05-22 | 9704 | 2017-11-24 | 2026-05-21 |  |
| 8905 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10600 | 2011-09-20 | 2026-05-22 | 10595 | 2011-09-20 | 2026-05-20 |  |
| 8908 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 6311 | 2007-07-23 | 2026-05-22 | 6306 | 2007-07-23 | 2026-05-21 |  |
| 8916 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14672 | 2006-06-28 | 2026-05-22 | 14667 | 2006-06-28 | 2026-05-21 |  |
| 8917 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 3467 | 2007-06-05 | 2026-05-22 | 3462 | 2007-06-05 | 2026-05-21 |  |
| 8924 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16764 | 2005-01-07 | 2026-05-22 | 16759 | 2005-01-07 | 2026-05-21 |  |
| 8926 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15151 | 2012-05-02 | 2026-05-22 | 15146 | 2012-05-02 | 2026-05-21 |  |
| 8927 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9558 | 2006-12-14 | 2026-05-22 | 9553 | 2006-12-14 | 2026-05-21 |  |
| 8928 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11869 | 2005-07-12 | 2026-05-22 | 11864 | 2005-07-12 | 2026-05-21 |  |
| 8929 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9794 | 2009-09-01 | 2026-05-22 | 9789 | 2009-09-01 | 2026-05-21 |  |
| 8930 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10159 | 2005-04-11 | 2026-05-22 | 10154 | 2005-04-11 | 2026-05-21 |  |
| 8931 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8701 | 2005-01-14 | 2026-05-22 | 8696 | 2005-01-14 | 2026-05-21 |  |
| 8932 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10134 | 2005-08-04 | 2026-05-22 | 10129 | 2005-08-04 | 2026-05-21 |  |
| 8933 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8933 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8933 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8933 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8933 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8933 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8933 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8933 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8933 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8933&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8935 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8935&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockShareholding | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockShareholding&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockFinancialStatements | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockFinancialStatements&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockMonthRevenue | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8936 | TaiwanStockDividend | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDividend&data_id=8936&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8937 | TaiwanStockPrice | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=8937&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8937 | TaiwanStockPriceAdj | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&data_id=8937&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8937 | TaiwanStockPER | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id=8937&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8937 | TaiwanStockInstitutionalInvestorsBuySell | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id=8937&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8937 | TaiwanStockMarginPurchaseShortSale | API_ERROR | None | None | None | None | None | None | HTTPError: 402 Client Error: Payment Required for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id=8937&start_date=1990-01-01&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| 8938 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18356 | 2005-01-10 | 2026-05-22 | 18351 | 2005-01-10 | 2026-05-21 |  |
| 8940 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11219 | 2005-01-13 | 2026-05-22 | 11214 | 2005-01-13 | 2026-05-21 |  |
| 8942 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17228 | 2005-01-19 | 2026-05-22 | 17223 | 2005-01-19 | 2026-05-21 |  |
| 8996 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17017 | 2006-07-05 | 2026-05-22 | 17012 | 2006-07-05 | 2026-05-21 |  |
| 9802 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14781 | 2012-10-18 | 2026-05-22 | 14776 | 2012-10-18 | 2026-05-21 |  |
| 9902 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13359 | 2012-05-02 | 2026-05-22 | 13354 | 2012-05-02 | 2026-05-21 |  |
| 9904 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 9905 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15005 | 2012-05-02 | 2026-05-22 | 15000 | 2012-05-02 | 2026-05-21 |  |
| 9906 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13739 | 2012-05-02 | 2026-05-22 | 13734 | 2012-05-02 | 2026-05-21 |  |
| 9907 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15155 | 2012-05-02 | 2026-05-22 | 15150 | 2012-05-02 | 2026-05-21 |  |
| 9908 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14545 | 2012-05-02 | 2026-05-22 | 14540 | 2012-05-02 | 2026-05-21 |  |
| 9910 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 9911 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15115 | 2012-05-02 | 2026-05-22 | 15110 | 2012-05-02 | 2026-05-21 |  |
| 9912 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11852 | 2012-05-02 | 2026-05-22 | 11847 | 2012-05-02 | 2026-05-21 |  |
| 9914 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 9917 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15157 | 2012-05-02 | 2026-05-22 | 15152 | 2012-05-02 | 2026-05-21 |  |
| 9918 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9645 | 2012-05-02 | 2026-05-22 | 9640 | 2012-05-02 | 2026-05-21 |  |
| 9919 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14237 | 2012-05-02 | 2026-05-22 | 14232 | 2012-05-02 | 2026-05-21 |  |
| 9921 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15159 | 2012-05-02 | 2026-05-22 | 15154 | 2012-05-02 | 2026-05-21 |  |
| 9924 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15078 | 2012-05-02 | 2026-05-22 | 15073 | 2012-05-02 | 2026-05-21 |  |
| 9924 | TaiwanStockDividend | MISMATCH | 22 | 2005-07-25 | 2026-06-15 | 21 | 2005-07-25 | 2025-06-17 |  |
| 9925 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15113 | 2012-05-02 | 2026-05-22 | 15108 | 2012-05-02 | 2026-05-21 |  |
| 9926 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9899 | 2012-05-02 | 2026-05-22 | 9894 | 2012-05-02 | 2026-05-21 |  |
| 9927 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15041 | 2012-05-02 | 2026-05-22 | 15036 | 2012-05-02 | 2026-05-21 |  |
| 9929 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9637 | 2012-05-02 | 2026-05-22 | 9632 | 2012-05-02 | 2026-05-21 |  |
| 9930 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14686 | 2012-05-02 | 2026-05-22 | 14681 | 2012-05-02 | 2026-05-21 |  |
| 9931 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8584 | 2012-05-07 | 2026-05-22 | 8579 | 2012-05-07 | 2026-05-21 |  |
| 9933 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-22 | 15157 | 2012-05-02 | 2026-05-21 |  |
| 9934 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15152 | 2012-05-02 | 2026-05-22 | 15147 | 2012-05-02 | 2026-05-21 |  |
| 9935 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14443 | 2012-05-02 | 2026-05-22 | 14438 | 2012-05-02 | 2026-05-21 |  |
| 9937 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13942 | 2012-05-02 | 2026-05-22 | 13937 | 2012-05-02 | 2026-05-21 |  |
| 9938 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-22 | 15159 | 2012-05-02 | 2026-05-21 |  |
| 9939 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-22 | 15167 | 2012-05-02 | 2026-05-21 |  |
| 9940 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15152 | 2012-05-02 | 2026-05-22 | 15147 | 2012-05-02 | 2026-05-21 |  |
| 9941 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15156 | 2012-05-02 | 2026-05-22 | 15151 | 2012-05-02 | 2026-05-21 |  |
| 9942 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14862 | 2012-05-02 | 2026-05-22 | 14857 | 2012-05-02 | 2026-05-21 |  |
| 9943 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15035 | 2012-05-02 | 2026-05-22 | 15030 | 2012-05-02 | 2026-05-21 |  |
| 9944 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12678 | 2012-05-02 | 2026-05-22 | 12673 | 2012-05-02 | 2026-05-21 |  |
| 9945 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15102 | 2012-05-02 | 2026-05-22 | 15097 | 2012-05-02 | 2026-05-21 |  |
| 9946 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14669 | 2005-03-23 | 2026-05-22 | 14664 | 2005-03-23 | 2026-05-21 |  |
| 9949 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 5063 | 2005-02-25 | 2026-05-22 | 5058 | 2005-02-25 | 2026-05-21 |  |
| 9950 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 4628 | 2005-01-03 | 2026-05-22 | 4623 | 2005-01-03 | 2026-05-21 |  |
| 9951 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16276 | 2005-01-03 | 2026-05-22 | 16271 | 2005-01-03 | 2026-05-21 |  |
| 9955 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14680 | 2005-01-04 | 2026-05-22 | 14675 | 2005-01-04 | 2026-05-21 |  |
| 9958 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13943 | 2007-12-25 | 2026-05-22 | 13938 | 2007-12-25 | 2026-05-21 |  |
| 9960 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 7427 | 2005-02-17 | 2026-05-22 | 7422 | 2005-02-17 | 2026-05-21 |  |
| 9962 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 8844 | 2006-07-12 | 2026-05-22 | 8839 | 2006-07-12 | 2026-05-21 |  |

## Targeted Backfill Commands

```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 006201 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00625K --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00636K --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00657K --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00668K --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00679B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00687B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00687C --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00694B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00695B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00696B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00697B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00719B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00720B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00722B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00723B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00724B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00725B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00726B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00727B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00734B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00740B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00741B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00746B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00749B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00750B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00751B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00754B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00755B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00756B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00758B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00759B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00760B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00761B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00764B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00768B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00772B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00773B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00777B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00778B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00779B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00780B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00781B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00782B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00785B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00786B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00787B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00788B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00789B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00792B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00793B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00795B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00799B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00836B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00842B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00844B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00845B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00846B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00847B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00848B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00849B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00853B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00856B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00857B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00858 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00859B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00860B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00862B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00863B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00864B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00867B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00870B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00877 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00884B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00886 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00887 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00888 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00890B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00928 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00931B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00933B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00937B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00942B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00948B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00950B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00955 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00957B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00958B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00959B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00966B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00967B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00968B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00969B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00970B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 009806 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 009807 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00980B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00980D --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00980T --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 009814 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 009815 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00981B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00981D --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00982B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00983B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00984B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00985D --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00986B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00986D --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00987B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00988B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00989B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 00998A --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 020001 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 020025 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 020027 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 020033 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 020040 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 020041 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1240 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1259 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1264 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1268 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1294 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1295 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1336 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1341 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1402 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1413 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1416 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1423 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1434 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1439 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1444 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1445 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1446 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1447 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1449 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1454 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1455 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1457 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1459 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1463 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1464 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1465 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1466 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1470 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1473 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1477 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1516 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1521 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1529 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1530 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1531 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1532 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1535 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1536 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1539 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1541 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1560 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1563 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1565 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1568 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1569 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1570 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1580 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1583 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1584 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1586 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1587 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1591 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1593 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1595 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1597 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1598 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1599 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1603 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1604 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1605 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1608 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1609 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1611 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1612 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1614 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1615 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1616 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1617 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1618 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1626 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1702 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1707 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1708 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1709 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1710 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1711 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1712 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1713 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1714 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1717 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1718 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1720 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1721 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1722 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1723 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1725 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1726 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1727 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1730 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1731 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1732 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1733 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1734 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1735 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1736 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1737 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1752 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1760 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1762 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1773 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1776 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1781 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1783 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1784 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1785 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1786 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1788 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1789 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1795 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1796 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1799 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1802 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1805 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1806 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1808 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1809 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1810 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1813 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1815 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1817 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1903 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1904 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1905 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1906 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1907 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1909 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2002 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2006 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2007 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2008 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2009 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2010 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2012 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2013 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2014 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2015 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2017 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2020 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2022 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2023 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2025 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2027 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2028 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2029 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2030 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2031 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2032 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2033 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2034 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2035 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2038 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2049 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2059 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2061 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2062 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2063 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2064 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2065 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2066 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2067 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2069 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2070 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2072 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2101 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2102 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2103 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2104 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2105 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2106 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2107 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2108 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2109 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2114 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2115 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2201 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2204 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2206 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2207 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2208 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2211 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2221 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2227 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2228 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2230 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2231 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2233 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2236 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2239 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2241 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2243 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2247 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2248 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2250 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2254 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2258 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2301 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2302 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2303 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2305 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2308 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2312 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2313 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2314 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2316 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2317 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2323 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2324 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2327 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2328 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2329 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2331 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2332 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2337 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2338 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2340 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2342 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2344 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2345 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2347 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2348 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2349 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2351 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2352 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2353 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2354 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2355 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2356 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2357 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2359 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2360 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2362 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2363 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2364 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2365 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2367 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2368 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2369 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2371 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2373 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2374 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2375 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2376 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2377 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2379 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2380 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2382 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2383 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2385 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2387 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2388 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2390 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2392 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2393 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2395 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2397 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2399 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2401 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2402 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2404 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2405 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2406 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2408 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2409 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2412 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2413 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2414 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2415 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2417 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2419 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2420 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2421 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2423 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2424 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2425 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2426 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2427 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2428 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2429 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2430 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2431 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2432 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2433 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2434 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2436 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2438 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2439 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2440 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2441 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2442 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2444 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2449 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2450 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2451 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2453 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2454 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2455 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2457 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2458 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2459 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2460 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2461 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2462 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2464 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2465 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2466 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2467 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2468 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2471 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2472 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2474 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2476 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2477 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2478 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2480 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2481 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2482 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2483 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2484 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2485 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2486 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2488 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2489 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2491 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2492 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2493 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2495 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2496 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2497 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2498 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2501 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2504 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2505 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2506 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2509 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2511 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2514 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2515 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2516 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2520 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2524 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2527 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2528 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2530 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2534 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2535 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2536 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2537 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2538 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2539 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2540 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2542 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2543 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2545 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2546 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2547 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2548 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2596 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2597 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2601 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2603 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2605 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2606 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2607 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2608 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2609 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2610 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2611 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2612 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2613 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2614 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2615 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2616 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2617 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2618 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2630 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2633 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2634 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2636 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2637 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2640 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2641 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2642 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2643 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2645 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2646 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2701 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2702 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2704 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2705 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2706 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2707 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2712 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2718 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2722 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2723 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2726 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2727 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2729 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2731 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2732 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2734 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2736 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2739 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2743 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2745 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2748 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2752 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2753 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2754 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2755 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2756 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2762 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2801 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2812 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2816 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2820 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2832 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2834 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2836 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2838 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2845 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2849 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2850 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2851 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2852 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2855 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2867 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2880 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2881 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2881B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2881C --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2882 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2882A --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2882B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2883 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2883B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2884 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2885 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2886 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2887 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2887E --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2887G --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2887H --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2887I --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2889 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2890 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2891 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2891B --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2891C --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2892 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2897 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2901 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2903 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2904 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2905 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2906 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2908 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2910 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2911 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2912 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2913 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2915 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2916 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2923 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2929 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2937 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2939 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2945 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2949 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3002 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3003 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3004 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3005 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3006 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3008 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3010 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3011 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3013 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3014 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3015 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3016 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3017 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3019 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3021 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3022 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3023 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3024 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3025 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3026 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3027 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3028 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3029 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3030 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3031 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3032 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3033 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3034 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3035 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3036 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3037 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3038 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3040 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3041 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3042 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3043 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3044 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3045 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3046 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3047 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3048 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3049 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3050 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3051 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3052 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3054 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3055 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3056 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3057 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3058 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3059 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3060 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3062 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3066 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3071 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3073 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3078 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3081 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3083 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3086 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3088 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3092 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3093 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3094 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3095 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3105 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3114 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3115 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3118 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3122 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3128 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3130 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3131 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3135 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3138 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3141 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3147 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3149 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3150 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3152 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3158 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3162 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3163 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3164 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3167 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3168 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3169 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3171 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3176 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3178 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3188 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3189 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3191 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3205 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3206 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3207 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3209 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3211 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3213 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3217 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3218 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3219 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3221 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3224 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3226 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3227 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3228 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3229 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3230 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3231 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3232 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3234 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3236 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3252 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3257 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3259 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3260 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3264 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3265 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3266 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3268 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3272 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3276 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3284 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3285 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3287 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3288 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3289 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3290 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3293 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3294 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3296 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3297 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3303 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3305 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3306 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3308 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3310 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3311 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3312 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3313 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3317 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3321 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3322 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3323 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3324 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3325 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3332 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3338 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3339 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3346 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3349 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3354 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3356 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3357 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3360 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3362 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3363 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3372 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3373 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3374 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3376 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3379 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3380 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3388 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3390 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3402 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3406 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3406 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3413 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3416 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3419 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3430 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3432 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3434 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3437 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3438 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3441 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3443 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3444 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3447 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3450 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3455 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3465 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3466 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3467 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3479 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3481 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3483 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3484 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3485 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3489 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3490 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3491 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3492 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3494 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3498 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3499 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3501 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3504 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3508 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3511 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3512 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3515 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3516 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3518 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3520 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3521 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3522 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3523 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3526 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3527 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3528 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3529 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3530 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3532 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3533 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3535 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3537 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3540 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3541 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3543 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3545 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3546 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3548 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3550 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3551 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3552 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3555 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3556 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3557 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3558 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3563 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3564 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3567 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3570 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3576 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3577 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3580 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3581 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3583 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3587 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3588 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3591 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3592 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3594 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3596 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3597 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3605 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3607 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3609 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3611 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3615 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3617 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3622 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3623 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3624 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3625 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3628 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3630 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3631 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3645 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3652 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3653 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3661 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3663 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3665 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3666 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3669 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3672 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3673 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3675 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3679 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3680 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3684 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3685 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3686 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3687 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3689 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3691 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3693 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3694 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3701 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3702 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3703 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3704 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3705 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3706 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3707 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3708 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3709 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3710 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3711 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3712 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3713 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3714 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3715 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3716 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3717 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4102 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4104 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4105 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4106 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4107 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4108 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4109 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4111 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4113 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4114 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4116 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4119 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4120 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4121 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4123 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4126 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4127 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4128 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4129 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4130 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4131 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4133 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4137 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4138 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4139 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4142 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4147 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4148 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4153 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4154 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4155 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4157 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4160 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4161 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4162 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4163 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4164 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4166 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4167 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4168 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4169 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4171 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4173 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4174 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4175 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4178 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4188 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4205 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4207 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4303 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4304 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4305 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4306 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4401 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4402 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4413 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4414 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4416 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4417 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4419 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4420 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4426 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4430 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4432 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4438 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4439 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4440 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4441 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4442 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4502 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4503 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4506 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4510 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4513 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4523 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4526 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4527 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4528 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4530 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4532 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4533 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4534 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4536 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4538 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4540 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4541 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4542 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4543 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4545 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4549 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4550 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4551 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4552 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4554 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4555 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4556 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4557 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4558 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4560 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4561 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4562 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4563 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4564 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4566 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4569 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4571 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4572 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4576 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4577 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4580 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4581 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4582 --dataset TaiwanStockPER --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4582 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4582 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4583 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4585 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4588 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4590 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4702 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4706 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4707 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4711 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4714 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4716 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4720 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4721 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4722 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4726 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4729 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4735 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4736 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4737 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4739 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4741 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4743 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4744 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4745 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4746 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4747 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4749 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4754 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4755 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4760 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4763 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4764 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4766 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4767 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4768 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4770 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4771 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4772 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4806 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4807 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4903 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4904 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4905 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4906 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4907 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4908 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4909 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4911 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4912 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4915 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4916 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4919 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4923 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4924 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4927 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4930 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4931 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4933 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4934 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4935 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4938 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4939 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4942 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4946 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4949 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4951 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4952 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4953 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4956 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4958 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4960 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4961 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4966 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4967 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4968 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4971 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4973 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4974 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4976 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4977 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4979 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4989 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4991 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4994 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4995 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4999 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5007 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5009 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5011 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5013 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5014 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5015 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5016 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5201 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5202 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5203 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5206 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5209 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5210 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5211 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5212 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5213 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5213 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5215 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5220 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5222 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5223 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5225 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5227 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5228 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5230 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5234 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5236 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5240 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5243 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5244 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5245 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5251 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5258 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5263 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5269 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5272 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5274 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5276 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5278 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5283 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5284 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5285 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5287 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5288 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5289 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5291 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5292 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5299 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5301 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5302 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5306 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5309 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5312 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5314 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5315 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5321 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5324 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5328 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5340 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5344 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5345 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5347 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5351 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5353 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5355 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5356 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5364 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5371 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5381 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5386 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5388 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5392 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5398 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5403 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5410 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5425 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5426 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5432 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5434 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5438 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5439 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5443 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5450 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5452 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5455 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5457 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5460 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5464 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5465 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5468 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5469 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5471 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5474 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5475 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5478 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5481 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5483 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5484 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5487 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5488 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5489 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5490 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5493 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5498 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5525 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5529 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5530 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5531 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5533 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5534 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5536 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5538 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5604 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5607 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5608 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5609 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5701 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5704 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5706 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5859 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5864 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5871 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5876 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5880 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5902 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5903 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5904 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5905 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5906 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5907 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6005 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6015 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6016 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6020 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6021 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6023 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6024 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6026 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6028 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6101 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6104 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6108 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6109 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6111 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6112 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6113 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6114 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6115 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6116 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6117 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6118 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6120 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6121 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6122 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6123 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6124 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6125 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6126 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6127 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6128 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6129 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6130 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6133 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6134 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6136 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6138 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6139 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6140 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6141 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6142 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6164 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6165 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6166 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6167 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6168 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6169 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6170 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6171 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6173 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6174 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6175 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6176 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6177 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6179 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6180 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6182 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6183 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6184 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6185 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6186 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6187 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6188 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6189 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6190 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6191 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6192 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6194 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6195 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6196 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6197 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6198 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6199 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6201 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6202 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6203 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6204 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6205 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6206 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6207 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6208 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6209 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6210 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6212 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6213 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6214 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6215 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6216 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6217 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6218 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6219 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6220 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6221 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6222 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6223 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6224 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6225 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6226 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6227 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6229 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6230 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6231 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6233 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6234 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6235 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6237 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6239 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6240 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6241 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6243 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6244 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6245 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6246 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6248 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6257 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6259 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6261 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6263 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6264 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6265 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6266 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6269 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6270 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6271 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6272 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6274 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6275 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6276 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6277 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6278 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6279 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6281 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6282 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6283 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6284 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6285 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6290 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6291 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6292 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6294 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6405 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6409 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6411 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6412 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6414 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6415 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6416 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6417 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6418 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6419 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6423 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6425 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6426 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6431 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6432 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6435 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6438 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6441 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6442 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6443 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6446 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6449 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6451 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6456 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6461 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6462 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6464 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6465 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6469 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6470 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6472 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6473 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6477 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6482 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6485 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6486 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6488 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6491 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6498 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6499 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6504 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6505 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6506 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6508 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6509 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6510 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6515 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6516 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6517 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6523 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6525 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6526 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6527 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6530 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6531 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6532 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6533 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6534 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6535 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6538 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6541 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6542 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6542 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6546 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6547 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6548 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6550 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6552 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6556 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6558 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6560 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6561 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6568 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6569 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6570 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6573 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6574 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6576 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6577 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6578 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6579 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6581 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6582 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6584 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6585 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6588 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6589 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6591 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6592 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6593 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6596 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6597 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6598 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6603 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6605 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6606 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6609 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6612 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6613 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6614 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6615 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6616 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6617 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6620 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6624 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6625 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6629 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6637 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6639 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6640 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6641 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6643 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6645 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6649 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6651 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6654 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6655 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6657 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6658 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6658 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6664 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6665 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6666 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6667 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6668 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6669 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6670 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6671 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6672 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6674 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6679 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6683 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6684 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6689 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6690 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6691 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6693 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6695 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6697 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6698 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6703 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6706 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6712 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6715 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6716 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6719 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6720 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6721 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6722 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6725 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6727 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6730 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6732 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6733 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6735 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6738 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6739 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6741 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6742 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6743 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6751 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6752 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6753 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6754 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6756 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6757 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6761 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6763 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6767 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6768 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6770 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6771 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6776 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6781 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6782 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6785 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6788 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6789 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6790 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6791 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6792 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6794 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6796 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6799 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6803 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6804 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6805 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6806 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6807 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6811 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6821 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6823 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6829 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6830 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6831 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6833 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6834 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6835 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6838 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6840 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6841 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6843 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6846 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6854 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6855 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6858 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6859 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6861 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6862 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6863 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6865 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6869 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6870 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6873 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6874 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6875 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6877 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6884 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6885 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6887 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6890 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6894 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6895 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6899 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6901 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6902 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6903 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6906 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6907 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6908 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6909 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6910 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6911 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6913 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6913 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6914 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6916 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6918 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6919 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6921 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6922 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6923 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6924 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6925 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6928 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6929 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6931 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6932 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6933 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6934 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6936 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6937 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6944 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6949 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6951 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6951 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6952 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6953 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6957 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6958 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6962 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6965 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6967 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6968 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6969 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6971 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6982 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6988 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6994 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6996 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6997 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7402 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7547 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7556 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7561 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7566 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7584 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7607 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7610 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7631 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7642 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7703 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7704 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7705 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7708 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7709 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7711 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7712 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7713 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7714 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7715 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7716 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7717 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7717 --dataset TaiwanStockMarginPurchaseShortSale --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7723 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7728 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7730 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7732 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7734 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7736 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7737 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7738 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7740 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7749 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7750 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7751 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7753 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7757 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7757 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7759 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7760 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7786 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7788 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7791 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7792 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7794 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7795 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7799 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7803 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7805 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7810 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7811 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7818 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7820 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7821 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7822 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7823 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7828 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7842 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7858 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7901 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7907 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7913 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7918 --dataset TaiwanStockFinancialStatements --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 7920 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8011 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8016 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8021 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8024 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8027 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8028 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8032 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8033 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8034 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8038 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8039 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8040 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8042 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8043 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8044 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8045 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8046 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8047 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8048 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8049 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8050 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8054 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8058 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8059 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8064 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8066 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8068 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8069 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8070 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8071 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8072 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8074 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8076 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8080 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8081 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8083 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8084 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8085 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8086 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8087 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8088 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8089 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8091 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8092 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8093 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8096 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8097 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8099 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8102 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8103 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8104 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8105 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8107 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8109 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8110 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8111 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8112 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8114 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8121 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8131 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8147 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8150 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8155 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8162 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8163 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8171 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8176 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8182 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8183 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8201 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8210 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8213 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8215 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8222 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8227 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8234 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8240 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8249 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8255 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8261 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8271 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8272 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8277 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8279 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8284 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8289 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8349 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8354 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8358 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8367 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8374 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8383 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8390 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8401 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8403 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8404 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8409 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8410 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8411 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8415 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8422 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8423 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8424 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8426 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8429 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8431 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8432 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8433 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8435 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8436 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8437 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8438 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8440 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8442 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8444 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8446 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8450 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8454 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8455 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8458 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8462 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8463 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8464 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8466 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8467 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8472 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8473 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8476 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8477 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8478 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8481 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8482 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8487 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8488 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8489 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8499 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8905 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8908 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8916 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8917 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8924 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8926 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8927 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8928 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8929 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8930 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8931 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8932 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8938 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8940 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8942 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8996 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9802 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9902 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9904 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9905 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9906 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9907 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9908 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9910 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9911 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9912 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9914 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9917 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9918 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9919 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9921 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9924 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9924 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9925 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9926 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9927 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9929 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9930 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9931 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9933 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9934 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9935 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9937 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9938 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9939 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9940 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9941 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9942 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9943 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9944 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9945 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9946 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9949 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9950 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9951 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9955 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9958 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9960 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9962 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```

## FRED Valid Observation Alignment

| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |
|---|---|---:|---|---|---:|---|---|
| DFF | OK | 26257 | 1954-07-01 | 2026-05-20 | 26257 | 1954-07-01 | 2026-05-20 |
| UNRATE | OK | 939 | 1948-01-01 | 2026-04-01 | 939 | 1948-01-01 | 2026-04-01 |
| T10Y2Y | OK | 12490 | 1976-06-01 | 2026-05-21 | 12490 | 1976-06-01 | 2026-05-21 |
| VIXCLS | OK | 9190 | 1990-01-02 | 2026-05-20 | 9190 | 1990-01-02 | 2026-05-20 |
