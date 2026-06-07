# Strict source availability audit

- **time**: 2026-06-07 03:56:58
- **constitution**: 系統架構大憲章_v6.1.0.md §14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ
- **tool**: audit_source_availability v0.7
- **start_date**: 1990-01-01
- **drift_tolerance**: 3 day(s) (§6.8.8-C; 0 = strict)
- **scope**: stocks=397 (universe=full), datasets=10
- **special_full_market_reason**: weekly automated audit 3 via launchd cron (§6.8.7 第 (4) 條 / §6.8.8-D)
- **verdict**: **FAILED**
- **summary**: checked=3970, source_empty_ok=1, time_drift_ok=397, mismatch=942, api_errors=0

- **fred_summary**: checked=4, time_drift_ok=0, mismatch=0, api_errors=0

## Mismatches

| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max | error |
|---|---|---|---:|---|---|---:|---|---|---|
| 1210 | TaiwanStockPriceAdj | MISMATCH | 8776 | 1992-01-06 | 2026-06-05 | 8771 | 1992-01-06 | 2026-05-29 |  |
| 1210 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1102 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1102 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1215 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1215 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1216 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1216 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1227 | TaiwanStockPriceAdj | MISMATCH | 8132 | 1994-04-11 | 2026-06-05 | 8127 | 1994-04-11 | 2026-05-29 |  |
| 1227 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1229 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1229 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1303 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1303 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1319 | TaiwanStockPriceAdj | MISMATCH | 7941 | 1994-12-13 | 2026-06-05 | 7936 | 1994-12-13 | 2026-05-29 |  |
| 1319 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1402 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1402 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1326 | TaiwanStockPriceAdj | MISMATCH | 5882 | 1992-01-06 | 2026-06-05 | 5877 | 1992-01-06 | 2026-05-29 |  |
| 1326 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1434 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1434 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1476 | TaiwanStockPriceAdj | MISMATCH | 6186 | 2001-04-19 | 2026-06-05 | 6181 | 2001-04-19 | 2026-05-29 |  |
| 1476 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1476 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 1476 | TaiwanStockDividend | MISMATCH | 32 | 2005-07-06 | 2026-06-25 | 31 | 2005-07-06 | 2025-07-18 |  |
| 1477 | TaiwanStockPriceAdj | MISMATCH | 5751 | 2003-01-22 | 2026-06-05 | 5746 | 2003-01-22 | 2026-05-29 |  |
| 1477 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1477 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 1504 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1504 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1504 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 1513 | TaiwanStockPriceAdj | MISMATCH | 8126 | 1994-03-09 | 2026-06-05 | 8121 | 1994-03-09 | 2026-05-29 |  |
| 1513 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1514 | TaiwanStockPriceAdj | MISMATCH | 8141 | 1994-03-28 | 2026-06-05 | 8136 | 1994-03-28 | 2026-05-29 |  |
| 1514 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1515 | TaiwanStockPriceAdj | MISMATCH | 7895 | 1995-02-06 | 2026-06-05 | 7890 | 1995-02-06 | 2026-05-29 |  |
| 1515 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1522 | TaiwanStockPriceAdj | MISMATCH | 7131 | 1997-10-07 | 2026-06-05 | 7126 | 1997-10-07 | 2026-05-29 |  |
| 1522 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1533 | TaiwanStockPriceAdj | MISMATCH | 6196 | 2001-04-09 | 2026-06-05 | 6191 | 2001-04-09 | 2026-05-29 |  |
| 1533 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1560 | TaiwanStockPriceAdj | MISMATCH | 5245 | 2005-02-01 | 2026-06-05 | 5240 | 2005-02-01 | 2026-05-29 |  |
| 1560 | TaiwanStockShareholding | MISMATCH | 5291 | 2005-01-28 | 2026-06-05 | 5286 | 2005-01-28 | 2026-05-29 |  |
| 1560 | TaiwanStockMonthRevenue | MISMATCH | 278 | 2003-05-01 | 2026-06-01 | 277 | 2003-05-01 | 2026-05-01 |  |
| 1565 | TaiwanStockPriceAdj | MISMATCH | 5390 | 2004-03-31 | 2026-06-05 | 5385 | 2004-03-31 | 2026-05-29 |  |
| 1565 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 1568 | TaiwanStockPriceAdj | MISMATCH | 4759 | 2006-12-19 | 2026-06-05 | 4754 | 2006-12-19 | 2026-05-29 |  |
| 1568 | TaiwanStockShareholding | MISMATCH | 2971 | 2012-03-29 | 2026-06-05 | 2966 | 2012-03-29 | 2026-05-29 |  |
| 1582 | TaiwanStockPriceAdj | MISMATCH | 4703 | 2006-12-19 | 2026-06-05 | 4698 | 2006-12-19 | 2026-05-29 |  |
| 1582 | TaiwanStockShareholding | MISMATCH | 4070 | 2009-12-16 | 2026-06-05 | 4065 | 2009-12-16 | 2026-05-29 |  |
| 1582 | TaiwanStockDividend | MISMATCH | 21 | 2006-10-30 | 2026-06-22 | 20 | 2006-10-30 | 2025-06-22 |  |
| 1590 | TaiwanStockPriceAdj | MISMATCH | 3787 | 2010-12-14 | 2026-06-05 | 3782 | 2010-12-14 | 2026-05-29 |  |
| 1590 | TaiwanStockShareholding | MISMATCH | 3821 | 2010-12-10 | 2026-06-05 | 3816 | 2010-12-10 | 2026-05-29 |  |
| 1590 | TaiwanStockMonthRevenue | MISMATCH | 184 | 2011-03-01 | 2026-06-01 | 183 | 2011-03-01 | 2026-05-01 |  |
| 1597 | TaiwanStockPriceAdj | MISMATCH | 3728 | 2011-03-11 | 2026-06-05 | 3723 | 2011-03-11 | 2026-05-29 |  |
| 1597 | TaiwanStockShareholding | MISMATCH | 3294 | 2012-12-27 | 2026-06-05 | 3289 | 2012-12-27 | 2026-05-29 |  |
| 1605 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1605 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1605 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 1608 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1608 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1609 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1609 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1609 | TaiwanStockDividend | MISMATCH | 19 | 2005-08-10 | 2026-06-29 | 18 | 2005-08-10 | 2025-07-20 |  |
| 1611 | TaiwanStockPriceAdj | MISMATCH | 8764 | 1992-01-06 | 2026-06-05 | 8759 | 1992-01-06 | 2026-05-29 |  |
| 1611 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1611 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 1612 | TaiwanStockPriceAdj | MISMATCH | 8396 | 1993-05-10 | 2026-06-05 | 8391 | 1993-05-10 | 2026-05-29 |  |
| 1612 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1612 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 1618 | TaiwanStockPriceAdj | MISMATCH | 6336 | 2000-09-13 | 2026-06-05 | 6331 | 2000-09-13 | 2026-05-29 |  |
| 1618 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1702 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1702 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1712 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1712 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1714 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1714 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1714 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 1717 | TaiwanStockPriceAdj | MISMATCH | 8132 | 1994-03-31 | 2026-06-05 | 8127 | 1994-03-31 | 2026-05-29 |  |
| 1717 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1722 | TaiwanStockPriceAdj | MISMATCH | 7008 | 1998-03-25 | 2026-06-05 | 7003 | 1998-03-25 | 2026-05-29 |  |
| 1722 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1736 | TaiwanStockPriceAdj | MISMATCH | 5759 | 2003-01-10 | 2026-06-05 | 5754 | 2003-01-10 | 2026-05-29 |  |
| 1736 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1752 | TaiwanStockPriceAdj | MISMATCH | 4753 | 2006-12-13 | 2026-06-05 | 4748 | 2006-12-13 | 2026-05-29 |  |
| 1752 | TaiwanStockShareholding | MISMATCH | 4162 | 2009-08-10 | 2026-06-05 | 4157 | 2009-08-10 | 2026-05-29 |  |
| 1773 | TaiwanStockPriceAdj | MISMATCH | 4503 | 2008-01-28 | 2026-06-05 | 4498 | 2008-01-28 | 2026-05-29 |  |
| 1773 | TaiwanStockShareholding | MISMATCH | 4276 | 2009-02-26 | 2026-06-05 | 4271 | 2009-02-26 | 2026-05-29 |  |
| 1773 | TaiwanStockMonthRevenue | MISMATCH | 233 | 2007-02-01 | 2026-06-01 | 232 | 2007-02-01 | 2026-05-01 |  |
| 1784 | TaiwanStockPriceAdj | MISMATCH | 4763 | 2006-12-19 | 2026-06-05 | 4758 | 2006-12-19 | 2026-05-29 |  |
| 1784 | TaiwanStockShareholding | MISMATCH | 4671 | 2007-07-25 | 2026-06-05 | 4666 | 2007-07-25 | 2026-05-29 |  |
| 1785 | TaiwanStockPriceAdj | MISMATCH | 5008 | 2005-02-01 | 2026-06-05 | 5003 | 2005-02-01 | 2026-05-29 |  |
| 1785 | TaiwanStockShareholding | MISMATCH | 5274 | 2005-01-28 | 2026-06-05 | 5269 | 2005-01-28 | 2026-05-29 |  |
| 1795 | TaiwanStockPriceAdj | MISMATCH | 4759 | 2006-12-13 | 2026-06-05 | 4754 | 2006-12-13 | 2026-05-29 |  |
| 1795 | TaiwanStockShareholding | MISMATCH | 4030 | 2010-01-28 | 2026-06-05 | 4025 | 2010-01-28 | 2026-05-29 |  |
| 1789 | TaiwanStockPriceAdj | MISMATCH | 3818 | 2010-11-01 | 2026-06-05 | 3813 | 2010-11-01 | 2026-05-29 |  |
| 1789 | TaiwanStockShareholding | MISMATCH | 3623 | 2011-09-28 | 2026-06-05 | 3618 | 2011-09-28 | 2026-05-29 |  |
| 1789 | TaiwanStockDividend | MISMATCH | 16 | 2011-08-01 | 2026-07-10 | 15 | 2011-08-01 | 2025-07-11 |  |
| 1806 | TaiwanStockPriceAdj | MISMATCH | 8528 | 1992-09-30 | 2026-06-05 | 8523 | 1992-09-30 | 2026-05-29 |  |
| 1806 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1809 | TaiwanStockPriceAdj | MISMATCH | 7540 | 1996-05-01 | 2026-06-05 | 7535 | 1996-05-01 | 2026-05-29 |  |
| 1809 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1810 | TaiwanStockPriceAdj | MISMATCH | 7450 | 1992-01-06 | 2026-06-05 | 7445 | 1992-01-06 | 2026-05-29 |  |
| 1810 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1815 | TaiwanStockPriceAdj | MISMATCH | 4934 | 2006-01-24 | 2026-06-05 | 4929 | 2006-01-24 | 2026-05-29 |  |
| 1815 | TaiwanStockShareholding | MISMATCH | 5039 | 2006-01-20 | 2026-06-05 | 5034 | 2006-01-20 | 2026-05-29 |  |
| 1904 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1904 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 1904 | TaiwanStockDividend | MISMATCH | 21 | 2005-06-25 | 2026-06-28 | 20 | 2005-06-25 | 2025-06-27 |  |
| 1907 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 1907 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2006 | TaiwanStockPriceAdj | MISMATCH | 8770 | 1992-01-06 | 2026-06-05 | 8765 | 1992-01-06 | 2026-05-29 |  |
| 2006 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2006 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2020 | TaiwanStockPriceAdj | MISMATCH | 8404 | 1993-04-28 | 2026-06-05 | 8399 | 1993-04-28 | 2026-05-29 |  |
| 2020 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2027 | TaiwanStockPriceAdj | MISMATCH | 7404 | 1996-10-28 | 2026-06-05 | 7399 | 1996-10-28 | 2026-05-29 |  |
| 2027 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2031 | TaiwanStockPriceAdj | MISMATCH | 6342 | 2000-09-13 | 2026-06-05 | 6337 | 2000-09-13 | 2026-05-29 |  |
| 2031 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2049 | TaiwanStockPriceAdj | MISMATCH | 4655 | 2007-06-21 | 2026-06-05 | 4650 | 2007-06-21 | 2026-05-29 |  |
| 2049 | TaiwanStockShareholding | MISMATCH | 4193 | 2009-06-25 | 2026-06-05 | 4188 | 2009-06-25 | 2026-05-29 |  |
| 2049 | TaiwanStockMonthRevenue | MISMATCH | 240 | 2006-07-01 | 2026-06-01 | 239 | 2006-07-01 | 2026-05-01 |  |
| 2059 | TaiwanStockPriceAdj | MISMATCH | 5119 | 2005-04-29 | 2026-06-05 | 5114 | 2005-04-29 | 2026-05-29 |  |
| 2059 | TaiwanStockShareholding | MISMATCH | 4445 | 2008-06-24 | 2026-06-05 | 4440 | 2008-06-24 | 2026-05-29 |  |
| 2059 | TaiwanStockMonthRevenue | MISMATCH | 276 | 2003-07-01 | 2026-06-01 | 275 | 2003-07-01 | 2026-05-01 |  |
| 2101 | TaiwanStockPriceAdj | MISMATCH | 8615 | 1992-01-06 | 2026-06-05 | 8610 | 1992-01-06 | 2026-05-29 |  |
| 2101 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2101 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2103 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2103 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2103 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2105 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2105 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2106 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2106 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2106 | TaiwanStockDividend | MISMATCH | 22 | 2005-08-07 | 2026-06-28 | 21 | 2005-08-07 | 2025-06-23 |  |
| 2108 | TaiwanStockPriceAdj | MISMATCH | 8543 | 1992-10-28 | 2026-06-05 | 8538 | 1992-10-28 | 2026-05-29 |  |
| 2108 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2108 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2201 | TaiwanStockPriceAdj | MISMATCH | 8770 | 1992-01-06 | 2026-06-05 | 8765 | 1992-01-06 | 2026-05-29 |  |
| 2201 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2204 | TaiwanStockPriceAdj | MISMATCH | 8771 | 1992-01-06 | 2026-06-05 | 8766 | 1992-01-06 | 2026-05-29 |  |
| 2204 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2206 | TaiwanStockPriceAdj | MISMATCH | 7474 | 1996-07-30 | 2026-06-05 | 7469 | 1996-07-30 | 2026-05-29 |  |
| 2206 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2233 | TaiwanStockPriceAdj | MISMATCH | 3822 | 2010-10-26 | 2026-06-05 | 3817 | 2010-10-26 | 2026-05-29 |  |
| 2233 | TaiwanStockShareholding | MISMATCH | 3586 | 2011-11-11 | 2026-06-05 | 3581 | 2011-11-11 | 2026-05-29 |  |
| 2301 | TaiwanStockPriceAdj | MISMATCH | 8746 | 1992-01-06 | 2026-06-05 | 8741 | 1992-01-06 | 2026-05-29 |  |
| 2301 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2303 | TaiwanStockPriceAdj | MISMATCH | 8766 | 1992-01-06 | 2026-06-05 | 8761 | 1992-01-06 | 2026-05-29 |  |
| 2303 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2303 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2305 | TaiwanStockPriceAdj | MISMATCH | 8754 | 1992-01-04 | 2026-06-05 | 8749 | 1992-01-04 | 2026-05-29 |  |
| 2305 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2312 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2312 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2312 | TaiwanStockDividend | MISMATCH | 18 | 2005-07-15 | 2026-06-29 | 17 | 2005-07-15 | 2025-07-07 |  |
| 2308 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2308 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2313 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2313 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2313 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2313 | TaiwanStockDividend | MISMATCH | 18 | 2007-07-30 | 2026-06-28 | 17 | 2007-07-30 | 2026-06-08 |  |
| 2316 | TaiwanStockPriceAdj | MISMATCH | 8759 | 1992-01-06 | 2026-06-05 | 8754 | 1992-01-06 | 2026-05-29 |  |
| 2316 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2317 | TaiwanStockPriceAdj | MISMATCH | 8771 | 1992-01-06 | 2026-06-05 | 8766 | 1992-01-06 | 2026-05-29 |  |
| 2317 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2317 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2323 | TaiwanStockPriceAdj | MISMATCH | 8723 | 1992-02-18 | 2026-06-05 | 8718 | 1992-02-18 | 2026-05-29 |  |
| 2323 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2324 | TaiwanStockPriceAdj | MISMATCH | 8745 | 1992-02-19 | 2026-06-05 | 8740 | 1992-02-19 | 2026-05-29 |  |
| 2324 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2327 | TaiwanStockPriceAdj | MISMATCH | 8209 | 1993-10-23 | 2026-06-05 | 8204 | 1993-10-23 | 2026-05-29 |  |
| 2327 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2328 | TaiwanStockPriceAdj | MISMATCH | 8245 | 1993-11-10 | 2026-06-05 | 8240 | 1993-11-10 | 2026-05-29 |  |
| 2328 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2330 | TaiwanStockPriceAdj | MISMATCH | 8010 | 1994-09-14 | 2026-06-05 | 8005 | 1994-09-14 | 2026-05-29 |  |
| 2330 | TaiwanStockShareholding | MISMATCH | 5534 | 2004-02-12 | 2026-06-05 | 5529 | 2004-02-12 | 2026-05-29 |  |
| 2344 | TaiwanStockPriceAdj | MISMATCH | 7697 | 1995-10-19 | 2026-06-05 | 7692 | 1995-10-19 | 2026-05-29 |  |
| 2344 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2344 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2331 | TaiwanStockPriceAdj | MISMATCH | 7988 | 1994-09-22 | 2026-06-05 | 7983 | 1994-09-22 | 2026-05-29 |  |
| 2331 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2345 | TaiwanStockPriceAdj | MISMATCH | 7676 | 1995-11-16 | 2026-06-05 | 7671 | 1995-11-16 | 2026-05-29 |  |
| 2345 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2345 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2347 | TaiwanStockPriceAdj | MISMATCH | 7653 | 1995-12-14 | 2026-06-05 | 7648 | 1995-12-14 | 2026-05-29 |  |
| 2347 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2348 | TaiwanStockPriceAdj | MISMATCH | 7577 | 1996-01-06 | 2026-06-05 | 7572 | 1996-01-06 | 2026-05-29 |  |
| 2348 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2348 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2351 | TaiwanStockPriceAdj | MISMATCH | 7553 | 1996-04-26 | 2026-06-05 | 7548 | 1996-04-26 | 2026-05-29 |  |
| 2351 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2352 | TaiwanStockPriceAdj | MISMATCH | 7467 | 1996-07-23 | 2026-06-05 | 7462 | 1996-07-23 | 2026-05-29 |  |
| 2352 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2352 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2353 | TaiwanStockPriceAdj | MISMATCH | 7432 | 1996-09-19 | 2026-06-05 | 7427 | 1996-09-19 | 2026-05-29 |  |
| 2353 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2353 | TaiwanStockDividend | MISMATCH | 17 | 2005-08-12 | 2026-06-26 | 16 | 2005-08-12 | 2025-06-26 |  |
| 2354 | TaiwanStockPriceAdj | MISMATCH | 7417 | 1996-10-09 | 2026-06-05 | 7412 | 1996-10-09 | 2026-05-29 |  |
| 2354 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2354 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2355 | TaiwanStockPriceAdj | MISMATCH | 7413 | 1996-10-15 | 2026-06-05 | 7408 | 1996-10-15 | 2026-05-29 |  |
| 2355 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2356 | TaiwanStockPriceAdj | MISMATCH | 7391 | 1996-11-14 | 2026-06-05 | 7386 | 1996-11-14 | 2026-05-29 |  |
| 2356 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2357 | TaiwanStockPriceAdj | MISMATCH | 7364 | 1996-11-15 | 2026-06-05 | 7359 | 1996-11-15 | 2026-05-29 |  |
| 2357 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2359 | TaiwanStockPriceAdj | MISMATCH | 7325 | 1996-12-20 | 2026-06-05 | 7320 | 1996-12-20 | 2026-05-29 |  |
| 2359 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2363 | TaiwanStockPriceAdj | MISMATCH | 7156 | 1997-08-02 | 2026-06-05 | 7151 | 1997-08-02 | 2026-05-29 |  |
| 2363 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2368 | TaiwanStockPriceAdj | MISMATCH | 7013 | 1998-03-10 | 2026-06-05 | 7008 | 1998-03-10 | 2026-05-29 |  |
| 2368 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2368 | TaiwanStockDividend | MISMATCH | 11 | 2006-07-08 | 2026-06-29 | 10 | 2006-07-08 | 2025-06-30 |  |
| 2374 | TaiwanStockPriceAdj | MISMATCH | 6785 | 1999-01-06 | 2026-06-05 | 6780 | 1999-01-06 | 2026-05-29 |  |
| 2374 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2375 | TaiwanStockPriceAdj | MISMATCH | 6833 | 1998-08-31 | 2026-06-05 | 6828 | 1998-08-31 | 2026-05-29 |  |
| 2375 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2376 | TaiwanStockPriceAdj | MISMATCH | 6866 | 1998-09-25 | 2026-06-05 | 6861 | 1998-09-25 | 2026-05-29 |  |
| 2376 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2376 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2377 | TaiwanStockPriceAdj | MISMATCH | 6839 | 1998-11-02 | 2026-06-05 | 6834 | 1998-11-02 | 2026-05-29 |  |
| 2377 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2379 | TaiwanStockPriceAdj | MISMATCH | 6836 | 1998-10-27 | 2026-06-05 | 6831 | 1998-10-27 | 2026-05-29 |  |
| 2379 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2379 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2382 | TaiwanStockPriceAdj | MISMATCH | 6790 | 1999-01-11 | 2026-06-05 | 6785 | 1999-01-11 | 2026-05-29 |  |
| 2382 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2385 | TaiwanStockPriceAdj | MISMATCH | 6793 | 1999-01-06 | 2026-06-05 | 6788 | 1999-01-06 | 2026-05-29 |  |
| 2385 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2385 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2393 | TaiwanStockPriceAdj | MISMATCH | 6569 | 1999-11-05 | 2026-06-05 | 6564 | 1999-11-05 | 2026-05-29 |  |
| 2393 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2393 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2395 | TaiwanStockPriceAdj | MISMATCH | 6540 | 1999-12-14 | 2026-06-05 | 6535 | 1999-12-14 | 2026-05-29 |  |
| 2395 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2395 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2399 | TaiwanStockPriceAdj | MISMATCH | 6537 | 1999-12-17 | 2026-06-05 | 6532 | 1999-12-17 | 2026-05-29 |  |
| 2399 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2402 | TaiwanStockPriceAdj | MISMATCH | 6519 | 2000-01-15 | 2026-06-05 | 6514 | 2000-01-15 | 2026-05-29 |  |
| 2402 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2402 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2404 | TaiwanStockPriceAdj | MISMATCH | 6472 | 2000-03-15 | 2026-06-05 | 6467 | 2000-03-15 | 2026-05-29 |  |
| 2404 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2408 | TaiwanStockPriceAdj | MISMATCH | 6343 | 2000-08-18 | 2026-06-05 | 6338 | 2000-08-18 | 2026-05-29 |  |
| 2408 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2408 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2409 | TaiwanStockPriceAdj | MISMATCH | 6336 | 2000-09-11 | 2026-06-05 | 6331 | 2000-09-11 | 2026-05-29 |  |
| 2409 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2412 | TaiwanStockPriceAdj | MISMATCH | 6256 | 2000-10-30 | 2026-06-05 | 6251 | 2000-10-30 | 2026-05-29 |  |
| 2412 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2413 | TaiwanStockPriceAdj | MISMATCH | 6342 | 2000-09-13 | 2026-06-05 | 6337 | 2000-09-13 | 2026-05-29 |  |
| 2413 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2417 | TaiwanStockPriceAdj | MISMATCH | 6335 | 2000-09-13 | 2026-06-05 | 6330 | 2000-09-13 | 2026-05-29 |  |
| 2417 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2417 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2421 | TaiwanStockPriceAdj | MISMATCH | 6342 | 2000-09-13 | 2026-06-05 | 6337 | 2000-09-13 | 2026-05-29 |  |
| 2421 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2427 | TaiwanStockPriceAdj | MISMATCH | 6342 | 2000-09-13 | 2026-06-05 | 6337 | 2000-09-13 | 2026-05-29 |  |
| 2427 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2428 | TaiwanStockPriceAdj | MISMATCH | 6342 | 2000-09-13 | 2026-06-05 | 6337 | 2000-09-13 | 2026-05-29 |  |
| 2428 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2436 | TaiwanStockPriceAdj | MISMATCH | 6327 | 2000-09-13 | 2026-06-05 | 6322 | 2000-09-13 | 2026-05-29 |  |
| 2436 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2436 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2439 | TaiwanStockPriceAdj | MISMATCH | 6342 | 2000-09-13 | 2026-06-05 | 6337 | 2000-09-13 | 2026-05-29 |  |
| 2439 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2439 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2441 | TaiwanStockPriceAdj | MISMATCH | 6307 | 2000-10-27 | 2026-06-05 | 6302 | 2000-10-27 | 2026-05-29 |  |
| 2441 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2441 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2442 | TaiwanStockPriceAdj | MISMATCH | 6254 | 2000-11-23 | 2026-06-05 | 6249 | 2000-11-23 | 2026-05-29 |  |
| 2442 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2449 | TaiwanStockPriceAdj | MISMATCH | 6174 | 2001-05-10 | 2026-06-05 | 6169 | 2001-05-10 | 2026-05-29 |  |
| 2449 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2451 | TaiwanStockPriceAdj | MISMATCH | 6178 | 2001-05-04 | 2026-06-05 | 6173 | 2001-05-04 | 2026-05-29 |  |
| 2451 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2451 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2453 | TaiwanStockPriceAdj | MISMATCH | 6165 | 2001-05-23 | 2026-06-05 | 6160 | 2001-05-23 | 2026-05-29 |  |
| 2453 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2453 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2454 | TaiwanStockPriceAdj | MISMATCH | 6122 | 2001-07-24 | 2026-06-05 | 6117 | 2001-07-24 | 2026-05-29 |  |
| 2454 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2458 | TaiwanStockPriceAdj | MISMATCH | 6077 | 2001-09-20 | 2026-06-05 | 6072 | 2001-09-20 | 2026-05-29 |  |
| 2458 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2458 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2467 | TaiwanStockPriceAdj | MISMATCH | 6084 | 2001-09-20 | 2026-06-05 | 6079 | 2001-09-20 | 2026-05-29 |  |
| 2467 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2467 | TaiwanStockDividend | MISMATCH | 21 | 2005-10-10 | 2026-06-27 | 20 | 2005-10-10 | 2025-06-03 |  |
| 2472 | TaiwanStockPriceAdj | MISMATCH | 6074 | 2001-09-20 | 2026-06-05 | 6069 | 2001-09-20 | 2026-05-29 |  |
| 2472 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2474 | TaiwanStockPriceAdj | MISMATCH | 6084 | 2001-09-20 | 2026-06-05 | 6079 | 2001-09-20 | 2026-05-29 |  |
| 2474 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2481 | TaiwanStockPriceAdj | MISMATCH | 6079 | 2001-09-20 | 2026-06-05 | 6074 | 2001-09-20 | 2026-05-29 |  |
| 2481 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2484 | TaiwanStockPriceAdj | MISMATCH | 6084 | 2001-09-20 | 2026-06-05 | 6079 | 2001-09-20 | 2026-05-29 |  |
| 2484 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2489 | TaiwanStockPriceAdj | MISMATCH | 6058 | 2001-09-20 | 2026-06-05 | 6053 | 2001-09-20 | 2026-05-29 |  |
| 2489 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2492 | TaiwanStockPriceAdj | MISMATCH | 6062 | 2001-09-20 | 2026-06-05 | 6057 | 2001-09-20 | 2026-05-29 |  |
| 2492 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2497 | TaiwanStockPriceAdj | MISMATCH | 5989 | 2002-02-05 | 2026-06-05 | 5984 | 2002-02-05 | 2026-05-29 |  |
| 2497 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2497 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2498 | TaiwanStockPriceAdj | MISMATCH | 5961 | 2002-03-27 | 2026-06-05 | 5956 | 2002-03-27 | 2026-05-29 |  |
| 2498 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2498 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 2498 | TaiwanStockDividend | MISMATCH | 12 | 2005-08-12 | 2026-06-28 | 11 | 2005-08-12 | 2019-08-25 |  |
| 2504 | TaiwanStockPriceAdj | MISMATCH | 8771 | 1992-01-06 | 2026-06-05 | 8766 | 1992-01-06 | 2026-05-29 |  |
| 2504 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2504 | TaiwanStockDividend | MISMATCH | 21 | 2005-08-13 | 2026-06-25 | 20 | 2005-08-13 | 2025-06-30 |  |
| 2515 | TaiwanStockPriceAdj | MISMATCH | 8451 | 1993-03-03 | 2026-06-05 | 8446 | 1993-03-03 | 2026-05-29 |  |
| 2515 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2520 | TaiwanStockPriceAdj | MISMATCH | 8255 | 1993-10-28 | 2026-06-05 | 8250 | 1993-10-28 | 2026-05-29 |  |
| 2520 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2535 | TaiwanStockPriceAdj | MISMATCH | 7580 | 1996-03-12 | 2026-06-05 | 7575 | 1996-03-12 | 2026-05-29 |  |
| 2535 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2542 | TaiwanStockPriceAdj | MISMATCH | 6700 | 1999-05-04 | 2026-06-05 | 6695 | 1999-05-04 | 2026-05-29 |  |
| 2542 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2545 | TaiwanStockPriceAdj | MISMATCH | 6341 | 2000-09-13 | 2026-06-05 | 6336 | 2000-09-13 | 2026-05-29 |  |
| 2545 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2603 | TaiwanStockPriceAdj | MISMATCH | 8770 | 1992-01-06 | 2026-06-05 | 8765 | 1992-01-06 | 2026-05-29 |  |
| 2603 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 2603 | TaiwanStockDividend | MISMATCH | 16 | 2005-08-26 | 2026-06-26 | 15 | 2005-08-26 | 2025-06-25 |  |
| 2605 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2605 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2606 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2606 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2610 | TaiwanStockPriceAdj | MISMATCH | 8442 | 1993-02-27 | 2026-06-05 | 8437 | 1993-02-27 | 2026-05-29 |  |
| 2610 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2612 | TaiwanStockPriceAdj | MISMATCH | 7976 | 1994-10-21 | 2026-06-05 | 7971 | 1994-10-21 | 2026-05-29 |  |
| 2612 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2615 | TaiwanStockPriceAdj | MISMATCH | 7535 | 1996-05-17 | 2026-06-05 | 7530 | 1996-05-17 | 2026-05-29 |  |
| 2615 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2617 | TaiwanStockPriceAdj | MISMATCH | 6938 | 1998-06-25 | 2026-06-05 | 6933 | 1998-06-25 | 2026-05-29 |  |
| 2617 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2618 | TaiwanStockPriceAdj | MISMATCH | 6072 | 2001-09-20 | 2026-06-05 | 6067 | 2001-09-20 | 2026-05-29 |  |
| 2618 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2634 | TaiwanStockPriceAdj | MISMATCH | 2873 | 2014-08-26 | 2026-06-05 | 2868 | 2014-08-26 | 2026-05-29 |  |
| 2634 | TaiwanStockShareholding | MISMATCH | 2900 | 2012-03-29 | 2026-06-05 | 2895 | 2012-03-29 | 2026-05-29 |  |
| 2634 | TaiwanStockDividend | MISMATCH | 13 | 2015-07-20 | 2026-07-14 | 12 | 2015-07-20 | 2025-07-15 |  |
| 2636 | TaiwanStockPriceAdj | MISMATCH | 4695 | 2007-04-24 | 2026-06-05 | 4690 | 2007-04-24 | 2026-05-29 |  |
| 2636 | TaiwanStockShareholding | MISMATCH | 2323 | 2012-03-29 | 2026-06-05 | 2318 | 2012-03-29 | 2026-05-29 |  |
| 2637 | TaiwanStockPriceAdj | MISMATCH | 4111 | 2009-09-01 | 2026-06-05 | 4106 | 2009-09-01 | 2026-05-29 |  |
| 2637 | TaiwanStockShareholding | MISMATCH | 3829 | 2010-11-30 | 2026-06-05 | 3824 | 2010-11-30 | 2026-05-29 |  |
| 2637 | TaiwanStockMonthRevenue | MISMATCH | 191 | 2009-09-01 | 2026-06-01 | 190 | 2009-09-01 | 2026-05-01 |  |
| 2727 | TaiwanStockPriceAdj | MISMATCH | 3695 | 2011-05-03 | 2026-06-05 | 3690 | 2011-05-03 | 2026-05-29 |  |
| 2727 | TaiwanStockShareholding | MISMATCH | 3516 | 2012-03-05 | 2026-06-05 | 3511 | 2012-03-05 | 2026-05-29 |  |
| 2731 | TaiwanStockPriceAdj | MISMATCH | 3322 | 2012-10-24 | 2026-06-05 | 3317 | 2012-10-24 | 2026-05-29 |  |
| 2731 | TaiwanStockShareholding | MISMATCH | 3128 | 2012-03-29 | 2026-06-05 | 3123 | 2012-03-29 | 2026-05-29 |  |
| 2732 | TaiwanStockPriceAdj | MISMATCH | 3281 | 2012-12-19 | 2026-06-05 | 3276 | 2012-12-19 | 2026-05-29 |  |
| 2732 | TaiwanStockShareholding | MISMATCH | 2796 | 2015-01-19 | 2026-06-05 | 2791 | 2015-01-19 | 2026-05-29 |  |
| 2903 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 2903 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 2915 | TaiwanStockPriceAdj | MISMATCH | 5875 | 1992-01-06 | 2026-06-05 | 5870 | 1992-01-06 | 2026-05-29 |  |
| 2915 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3003 | TaiwanStockPriceAdj | MISMATCH | 5985 | 2002-02-20 | 2026-06-05 | 5980 | 2002-02-20 | 2026-05-29 |  |
| 3003 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3004 | TaiwanStockPriceAdj | MISMATCH | 5920 | 2002-02-26 | 2026-06-05 | 5915 | 2002-02-26 | 2026-05-29 |  |
| 3004 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3004 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3005 | TaiwanStockPriceAdj | MISMATCH | 5981 | 2002-02-26 | 2026-06-05 | 5976 | 2002-02-26 | 2026-05-29 |  |
| 3005 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3006 | TaiwanStockPriceAdj | MISMATCH | 5977 | 2002-03-05 | 2026-06-05 | 5972 | 2002-03-05 | 2026-05-29 |  |
| 3006 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3006 | TaiwanStockMonthRevenue | MISMATCH | 292 | 2002-03-01 | 2026-06-01 | 291 | 2002-03-01 | 2026-05-01 |  |
| 3008 | TaiwanStockPriceAdj | MISMATCH | 5972 | 2002-03-12 | 2026-06-05 | 5967 | 2002-03-12 | 2026-05-29 |  |
| 3008 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3008 | TaiwanStockMonthRevenue | MISMATCH | 291 | 2002-04-01 | 2026-06-01 | 290 | 2002-04-01 | 2026-05-01 |  |
| 3013 | TaiwanStockPriceAdj | MISMATCH | 5840 | 2002-09-17 | 2026-06-05 | 5835 | 2002-09-17 | 2026-05-29 |  |
| 3013 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3014 | TaiwanStockPriceAdj | MISMATCH | 5801 | 2002-10-30 | 2026-06-05 | 5796 | 2002-10-30 | 2026-05-29 |  |
| 3014 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3016 | TaiwanStockPriceAdj | MISMATCH | 5767 | 2002-12-25 | 2026-06-05 | 5762 | 2002-12-25 | 2026-05-29 |  |
| 3016 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3017 | TaiwanStockPriceAdj | MISMATCH | 5831 | 2002-09-30 | 2026-06-05 | 5826 | 2002-09-30 | 2026-05-29 |  |
| 3017 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3019 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3019 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3019 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3019 | TaiwanStockDividend | MISMATCH | 15 | 2005-07-29 | 2026-06-30 | 14 | 2005-07-29 | 2025-06-30 |  |
| 3022 | TaiwanStockPriceAdj | MISMATCH | 5841 | 2002-08-27 | 2026-06-05 | 5836 | 2002-08-27 | 2026-05-29 |  |
| 3022 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3022 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3023 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3023 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3023 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3025 | TaiwanStockPriceAdj | MISMATCH | 5847 | 2002-08-27 | 2026-06-05 | 5842 | 2002-08-27 | 2026-05-29 |  |
| 3025 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3025 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3029 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3029 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3029 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3030 | TaiwanStockPriceAdj | MISMATCH | 5810 | 2002-10-30 | 2026-06-05 | 5805 | 2002-10-30 | 2026-05-29 |  |
| 3030 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3030 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3031 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3031 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3033 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3033 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3034 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3034 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3034 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3036 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3036 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3037 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3037 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3038 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3038 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3042 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3042 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3042 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 3044 | TaiwanStockPriceAdj | MISMATCH | 5854 | 2002-08-27 | 2026-06-05 | 5849 | 2002-08-27 | 2026-05-29 |  |
| 3044 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3048 | TaiwanStockPriceAdj | MISMATCH | 5829 | 2002-10-02 | 2026-06-05 | 5824 | 2002-10-02 | 2026-05-29 |  |
| 3048 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3048 | TaiwanStockMonthRevenue | MISMATCH | 287 | 2002-08-01 | 2026-06-01 | 286 | 2002-08-01 | 2026-05-01 |  |
| 3059 | TaiwanStockPriceAdj | MISMATCH | 5763 | 2002-12-25 | 2026-06-05 | 5758 | 2002-12-25 | 2026-05-29 |  |
| 3059 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3071 | TaiwanStockPriceAdj | MISMATCH | 5157 | 2004-12-31 | 2026-06-05 | 5152 | 2004-12-31 | 2026-05-29 |  |
| 3071 | TaiwanStockShareholding | MISMATCH | 5296 | 2004-12-29 | 2026-06-05 | 5291 | 2004-12-29 | 2026-05-29 |  |
| 3078 | TaiwanStockPriceAdj | MISMATCH | 5176 | 2005-02-01 | 2026-06-05 | 5171 | 2005-02-01 | 2026-05-29 |  |
| 3078 | TaiwanStockShareholding | MISMATCH | 5274 | 2005-01-28 | 2026-06-05 | 5269 | 2005-01-28 | 2026-05-29 |  |
| 3083 | TaiwanStockPriceAdj | MISMATCH | 5468 | 2003-12-03 | 2026-06-05 | 5463 | 2003-12-03 | 2026-05-29 |  |
| 3083 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 3088 | TaiwanStockPriceAdj | MISMATCH | 5121 | 2005-04-29 | 2026-06-05 | 5116 | 2005-04-29 | 2026-05-29 |  |
| 3088 | TaiwanStockShareholding | MISMATCH | 5223 | 2005-04-27 | 2026-06-05 | 5218 | 2005-04-27 | 2026-05-29 |  |
| 3090 | TaiwanStockPriceAdj | MISMATCH | 5482 | 2003-11-11 | 2026-06-05 | 5477 | 2003-11-11 | 2026-05-29 |  |
| 3090 | TaiwanStockShareholding | MISMATCH | 4564 | 2007-12-28 | 2026-06-05 | 4559 | 2007-12-28 | 2026-05-29 |  |
| 3090 | TaiwanStockMonthRevenue | MISMATCH | 288 | 2002-07-01 | 2026-06-01 | 287 | 2002-07-01 | 2026-05-01 |  |
| 3094 | TaiwanStockPriceAdj | MISMATCH | 4771 | 2006-12-19 | 2026-06-05 | 4766 | 2006-12-19 | 2026-05-29 |  |
| 3094 | TaiwanStockShareholding | MISMATCH | 4666 | 2007-08-03 | 2026-06-05 | 4661 | 2007-08-03 | 2026-05-29 |  |
| 3094 | TaiwanStockMonthRevenue | MISMATCH | 288 | 2002-07-01 | 2026-06-01 | 287 | 2002-07-01 | 2026-05-01 |  |
| 3105 | TaiwanStockPriceAdj | MISMATCH | 4069 | 2009-10-07 | 2026-06-05 | 4064 | 2009-10-07 | 2026-05-29 |  |
| 3105 | TaiwanStockShareholding | MISMATCH | 3571 | 2011-12-12 | 2026-06-05 | 3566 | 2011-12-12 | 2026-05-29 |  |
| 3105 | TaiwanStockMonthRevenue | MISMATCH | 201 | 2009-10-01 | 2026-06-01 | 200 | 2009-10-01 | 2026-05-01 |  |
| 3141 | TaiwanStockPriceAdj | MISMATCH | 4730 | 2006-12-19 | 2026-06-05 | 4725 | 2006-12-19 | 2026-05-29 |  |
| 3141 | TaiwanStockShareholding | MISMATCH | 3011 | 2014-03-13 | 2026-06-05 | 3006 | 2014-03-13 | 2026-05-29 |  |
| 3141 | TaiwanStockMonthRevenue | MISMATCH | 281 | 2003-02-01 | 2026-06-01 | 280 | 2003-02-01 | 2026-05-01 |  |
| 3163 | TaiwanStockPriceAdj | MISMATCH | 3565 | 2011-11-04 | 2026-06-05 | 3560 | 2011-11-04 | 2026-05-29 |  |
| 3163 | TaiwanStockShareholding | MISMATCH | 3326 | 2012-11-30 | 2026-06-05 | 3321 | 2012-11-30 | 2026-05-29 |  |
| 3169 | TaiwanStockPriceAdj | MISMATCH | 4759 | 2006-12-19 | 2026-06-05 | 4754 | 2006-12-19 | 2026-05-29 |  |
| 3169 | TaiwanStockShareholding | MISMATCH | 4087 | 2009-11-23 | 2026-06-05 | 4082 | 2009-11-23 | 2026-05-29 |  |
| 3169 | TaiwanStockMonthRevenue | MISMATCH | 278 | 2003-05-01 | 2026-06-01 | 277 | 2003-05-01 | 2026-05-01 |  |
| 3189 | TaiwanStockPriceAdj | MISMATCH | 5310 | 2004-11-02 | 2026-06-05 | 5305 | 2004-11-02 | 2026-05-29 |  |
| 3189 | TaiwanStockShareholding | MISMATCH | 5356 | 2004-10-29 | 2026-06-05 | 5351 | 2004-10-29 | 2026-05-29 |  |
| 3189 | TaiwanStockMonthRevenue | MISMATCH | 281 | 2003-02-01 | 2026-06-01 | 280 | 2003-02-01 | 2026-05-01 |  |
| 3209 | TaiwanStockPriceAdj | MISMATCH | 5389 | 2004-03-30 | 2026-06-05 | 5384 | 2004-03-30 | 2026-05-29 |  |
| 3209 | TaiwanStockShareholding | MISMATCH | 4338 | 2008-11-25 | 2026-06-05 | 4333 | 2008-11-25 | 2026-05-29 |  |
| 3211 | TaiwanStockPriceAdj | MISMATCH | 5236 | 2004-11-09 | 2026-06-05 | 5231 | 2004-11-09 | 2026-05-29 |  |
| 3211 | TaiwanStockShareholding | MISMATCH | 5334 | 2004-11-05 | 2026-06-05 | 5329 | 2004-11-05 | 2026-05-29 |  |
| 3211 | TaiwanStockMonthRevenue | MISMATCH | 281 | 2003-02-01 | 2026-06-01 | 280 | 2003-02-01 | 2026-05-01 |  |
| 3217 | TaiwanStockPriceAdj | MISMATCH | 5367 | 2004-05-03 | 2026-06-05 | 5362 | 2004-05-03 | 2026-05-29 |  |
| 3217 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 3218 | TaiwanStockPriceAdj | MISMATCH | 5221 | 2004-11-30 | 2026-06-05 | 5216 | 2004-11-30 | 2026-05-29 |  |
| 3218 | TaiwanStockShareholding | MISMATCH | 5319 | 2004-11-26 | 2026-06-05 | 5314 | 2004-11-26 | 2026-05-29 |  |
| 3227 | TaiwanStockPriceAdj | MISMATCH | 4870 | 2006-05-05 | 2026-06-05 | 4865 | 2006-05-05 | 2026-05-29 |  |
| 3227 | TaiwanStockShareholding | MISMATCH | 4976 | 2006-05-03 | 2026-06-05 | 4971 | 2006-05-03 | 2026-05-29 |  |
| 3227 | TaiwanStockMonthRevenue | MISMATCH | 281 | 2003-02-01 | 2026-06-01 | 280 | 2003-02-01 | 2026-05-01 |  |
| 3231 | TaiwanStockPriceAdj | MISMATCH | 5610 | 2003-08-20 | 2026-06-05 | 5605 | 2003-08-20 | 2026-05-29 |  |
| 3231 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 3257 | TaiwanStockPriceAdj | MISMATCH | 3952 | 2010-04-23 | 2026-06-05 | 3947 | 2010-04-23 | 2026-05-29 |  |
| 3257 | TaiwanStockShareholding | MISMATCH | 3756 | 2011-03-18 | 2026-06-05 | 3751 | 2011-03-18 | 2026-05-29 |  |
| 3260 | TaiwanStockPriceAdj | MISMATCH | 5256 | 2004-10-11 | 2026-06-05 | 5251 | 2004-10-11 | 2026-05-29 |  |
| 3260 | TaiwanStockShareholding | MISMATCH | 5354 | 2004-10-07 | 2026-06-05 | 5349 | 2004-10-07 | 2026-05-29 |  |
| 3264 | TaiwanStockPriceAdj | MISMATCH | 5194 | 2005-01-06 | 2026-06-05 | 5189 | 2005-01-06 | 2026-05-29 |  |
| 3264 | TaiwanStockShareholding | MISMATCH | 5292 | 2005-01-04 | 2026-06-05 | 5287 | 2005-01-04 | 2026-05-29 |  |
| 3264 | TaiwanStockMonthRevenue | MISMATCH | 278 | 2003-05-01 | 2026-06-01 | 277 | 2003-05-01 | 2026-05-01 |  |
| 3289 | TaiwanStockPriceAdj | MISMATCH | 5190 | 2004-12-29 | 2026-06-05 | 5185 | 2004-12-29 | 2026-05-29 |  |
| 3289 | TaiwanStockShareholding | MISMATCH | 5298 | 2004-12-27 | 2026-06-05 | 5293 | 2004-12-27 | 2026-05-29 |  |
| 3290 | TaiwanStockPriceAdj | MISMATCH | 5100 | 2005-05-20 | 2026-06-05 | 5095 | 2005-05-20 | 2026-05-29 |  |
| 3290 | TaiwanStockShareholding | MISMATCH | 5209 | 2005-05-18 | 2026-06-05 | 5204 | 2005-05-18 | 2026-05-29 |  |
| 3293 | TaiwanStockPriceAdj | MISMATCH | 4822 | 2006-07-13 | 2026-06-05 | 4817 | 2006-07-13 | 2026-05-29 |  |
| 3293 | TaiwanStockShareholding | MISMATCH | 4928 | 2006-07-11 | 2026-06-05 | 4923 | 2006-07-11 | 2026-05-29 |  |
| 3293 | TaiwanStockMonthRevenue | MISMATCH | 276 | 2003-07-01 | 2026-06-01 | 275 | 2003-07-01 | 2026-05-01 |  |
| 3312 | TaiwanStockPriceAdj | MISMATCH | 4954 | 2005-11-08 | 2026-06-05 | 4949 | 2005-11-08 | 2026-05-29 |  |
| 3312 | TaiwanStockShareholding | MISMATCH | 3809 | 2010-12-28 | 2026-06-05 | 3804 | 2010-12-28 | 2026-05-29 |  |
| 3312 | TaiwanStockMonthRevenue | MISMATCH | 274 | 2003-09-01 | 2026-06-01 | 273 | 2003-09-01 | 2026-05-01 |  |
| 3324 | TaiwanStockPriceAdj | MISMATCH | 5111 | 2005-05-16 | 2026-06-05 | 5106 | 2005-05-16 | 2026-05-29 |  |
| 3324 | TaiwanStockShareholding | MISMATCH | 5213 | 2005-05-12 | 2026-06-05 | 5208 | 2005-05-12 | 2026-05-29 |  |
| 3324 | TaiwanStockDividend | MISMATCH | 21 | 2005-08-17 | 2026-06-26 | 20 | 2005-08-17 | 2025-06-22 |  |
| 3317 | TaiwanStockPriceAdj | MISMATCH | 4771 | 2006-12-19 | 2026-06-05 | 4766 | 2006-12-19 | 2026-05-29 |  |
| 3317 | TaiwanStockShareholding | MISMATCH | 4661 | 2007-08-08 | 2026-06-05 | 4656 | 2007-08-08 | 2026-05-29 |  |
| 3338 | TaiwanStockPriceAdj | MISMATCH | 4756 | 2006-12-19 | 2026-06-05 | 4751 | 2006-12-19 | 2026-05-29 |  |
| 3338 | TaiwanStockShareholding | MISMATCH | 3071 | 2012-03-29 | 2026-06-05 | 3066 | 2012-03-29 | 2026-05-29 |  |
| 3356 | TaiwanStockPriceAdj | MISMATCH | 5206 | 2005-03-29 | 2026-06-05 | 5201 | 2005-03-29 | 2026-05-29 |  |
| 3356 | TaiwanStockShareholding | MISMATCH | 5256 | 2005-03-25 | 2026-06-05 | 5251 | 2005-03-25 | 2026-05-29 |  |
| 3376 | TaiwanStockPriceAdj | MISMATCH | 4949 | 2005-12-30 | 2026-06-05 | 4944 | 2005-12-30 | 2026-05-29 |  |
| 3376 | TaiwanStockShareholding | MISMATCH | 4564 | 2007-12-28 | 2026-06-05 | 4559 | 2007-12-28 | 2026-05-29 |  |
| 3402 | TaiwanStockPriceAdj | MISMATCH | 4710 | 2006-12-19 | 2026-06-05 | 4705 | 2006-12-19 | 2026-05-29 |  |
| 3402 | TaiwanStockShareholding | MISMATCH | 4803 | 2007-01-08 | 2026-06-05 | 4798 | 2007-01-08 | 2026-05-29 |  |
| 3406 | TaiwanStockPriceAdj | MISMATCH | 5027 | 2005-12-21 | 2026-06-05 | 5022 | 2005-12-21 | 2026-05-29 |  |
| 3406 | TaiwanStockShareholding | MISMATCH | 5071 | 2005-12-19 | 2026-06-05 | 5066 | 2005-12-19 | 2026-05-29 |  |
| 3413 | TaiwanStockPriceAdj | MISMATCH | 4738 | 2006-12-19 | 2026-06-05 | 4733 | 2006-12-19 | 2026-05-29 |  |
| 3413 | TaiwanStockShareholding | MISMATCH | 2672 | 2012-03-29 | 2026-06-05 | 2667 | 2012-03-29 | 2026-05-29 |  |
| 3413 | TaiwanStockMonthRevenue | MISMATCH | 264 | 2004-07-01 | 2026-06-01 | 263 | 2004-07-01 | 2026-05-01 |  |
| 3441 | TaiwanStockPriceAdj | MISMATCH | 4755 | 2006-12-19 | 2026-06-05 | 4750 | 2006-12-19 | 2026-05-29 |  |
| 3441 | TaiwanStockShareholding | MISMATCH | 3871 | 2010-10-01 | 2026-06-05 | 3866 | 2010-10-01 | 2026-05-29 |  |
| 3444 | TaiwanStockPriceAdj | MISMATCH | 4699 | 2007-04-19 | 2026-06-05 | 4694 | 2007-04-19 | 2026-05-29 |  |
| 3444 | TaiwanStockShareholding | MISMATCH | 4379 | 2008-09-25 | 2026-06-05 | 4374 | 2008-09-25 | 2026-05-29 |  |
| 3450 | TaiwanStockPriceAdj | MISMATCH | 4949 | 2006-04-13 | 2026-06-05 | 4944 | 2006-04-13 | 2026-05-29 |  |
| 3450 | TaiwanStockShareholding | MISMATCH | 4996 | 2006-04-11 | 2026-06-05 | 4991 | 2006-04-11 | 2026-05-29 |  |
| 3450 | TaiwanStockDividend | MISMATCH | 21 | 2005-06-23 | 2026-06-26 | 20 | 2005-06-23 | 2025-06-29 |  |
| 3455 | TaiwanStockPriceAdj | MISMATCH | 4768 | 2006-12-19 | 2026-06-05 | 4763 | 2006-12-19 | 2026-05-29 |  |
| 3455 | TaiwanStockShareholding | MISMATCH | 4564 | 2007-12-26 | 2026-06-05 | 4559 | 2007-12-26 | 2026-05-29 |  |
| 3481 | TaiwanStockPriceAdj | MISMATCH | 4798 | 2006-10-25 | 2026-06-05 | 4793 | 2006-10-25 | 2026-05-29 |  |
| 3481 | TaiwanStockShareholding | MISMATCH | 4861 | 2006-10-23 | 2026-06-05 | 4856 | 2006-10-23 | 2026-05-29 |  |
| 3479 | TaiwanStockPriceAdj | MISMATCH | 4581 | 2007-10-04 | 2026-06-05 | 4576 | 2007-10-04 | 2026-05-29 |  |
| 3479 | TaiwanStockShareholding | MISMATCH | 3953 | 2010-06-07 | 2026-06-05 | 3948 | 2010-06-07 | 2026-05-29 |  |
| 3483 | TaiwanStockPriceAdj | MISMATCH | 4712 | 2006-12-18 | 2026-06-05 | 4707 | 2006-12-18 | 2026-05-29 |  |
| 3483 | TaiwanStockShareholding | MISMATCH | 4818 | 2006-12-15 | 2026-06-05 | 4813 | 2006-12-15 | 2026-05-29 |  |
| 3484 | TaiwanStockPriceAdj | MISMATCH | 4759 | 2006-12-13 | 2026-06-05 | 4754 | 2006-12-13 | 2026-05-29 |  |
| 3484 | TaiwanStockShareholding | MISMATCH | 4598 | 2007-11-08 | 2026-06-05 | 4593 | 2007-11-08 | 2026-05-29 |  |
| 3498 | TaiwanStockPriceAdj | MISMATCH | 4763 | 2006-12-19 | 2026-06-05 | 4758 | 2006-12-19 | 2026-05-29 |  |
| 3498 | TaiwanStockShareholding | MISMATCH | 4638 | 2007-09-10 | 2026-06-05 | 4633 | 2007-09-10 | 2026-05-29 |  |
| 3515 | TaiwanStockPriceAdj | MISMATCH | 4757 | 2006-12-19 | 2026-06-05 | 4752 | 2006-12-19 | 2026-05-29 |  |
| 3515 | TaiwanStockShareholding | MISMATCH | 4601 | 2007-11-07 | 2026-06-05 | 4596 | 2007-11-07 | 2026-05-29 |  |
| 3526 | TaiwanStockPriceAdj | MISMATCH | 4759 | 2006-12-13 | 2026-06-05 | 4754 | 2006-12-13 | 2026-05-29 |  |
| 3526 | TaiwanStockShareholding | MISMATCH | 4603 | 2007-11-01 | 2026-06-05 | 4598 | 2007-11-01 | 2026-05-29 |  |
| 3533 | TaiwanStockPriceAdj | MISMATCH | 4765 | 2006-12-13 | 2026-06-05 | 4760 | 2006-12-13 | 2026-05-29 |  |
| 3533 | TaiwanStockShareholding | MISMATCH | 4579 | 2007-12-07 | 2026-06-05 | 4574 | 2007-12-07 | 2026-05-29 |  |
| 3540 | TaiwanStockPriceAdj | MISMATCH | 4755 | 2006-12-19 | 2026-06-05 | 4750 | 2006-12-19 | 2026-05-29 |  |
| 3540 | TaiwanStockShareholding | MISMATCH | 4575 | 2007-12-11 | 2026-06-05 | 4570 | 2007-12-11 | 2026-05-29 |  |
| 3535 | TaiwanStockPriceAdj | MISMATCH | 4761 | 2006-12-19 | 2026-06-05 | 4756 | 2006-12-19 | 2026-05-29 |  |
| 3535 | TaiwanStockShareholding | MISMATCH | 4542 | 2008-01-30 | 2026-06-05 | 4537 | 2008-01-30 | 2026-05-29 |  |
| 3552 | TaiwanStockPriceAdj | MISMATCH | 4653 | 2007-06-28 | 2026-06-05 | 4648 | 2007-06-28 | 2026-05-29 |  |
| 3552 | TaiwanStockShareholding | MISMATCH | 4310 | 2009-01-06 | 2026-06-05 | 4305 | 2009-01-06 | 2026-05-29 |  |
| 3563 | TaiwanStockPriceAdj | MISMATCH | 4649 | 2007-06-28 | 2026-06-05 | 4644 | 2007-06-28 | 2026-05-29 |  |
| 3563 | TaiwanStockShareholding | MISMATCH | 1761 | 2019-04-01 | 2026-06-05 | 1756 | 2019-04-01 | 2026-05-29 |  |
| 3563 | TaiwanStockMonthRevenue | MISMATCH | 229 | 2007-06-01 | 2026-06-01 | 228 | 2007-06-01 | 2026-05-01 |  |
| 3580 | TaiwanStockPriceAdj | MISMATCH | 4559 | 2007-10-26 | 2026-06-05 | 4554 | 2007-10-26 | 2026-05-29 |  |
| 3580 | TaiwanStockShareholding | MISMATCH | 3931 | 2010-07-08 | 2026-06-05 | 3926 | 2010-07-08 | 2026-05-29 |  |
| 3583 | TaiwanStockPriceAdj | MISMATCH | 3451 | 2012-04-24 | 2026-06-05 | 3446 | 2012-04-24 | 2026-05-29 |  |
| 3583 | TaiwanStockShareholding | MISMATCH | 3262 | 2012-03-29 | 2026-06-05 | 3257 | 2012-03-29 | 2026-05-29 |  |
| 3587 | TaiwanStockPriceAdj | MISMATCH | 4505 | 2008-01-22 | 2026-06-05 | 4500 | 2008-01-22 | 2026-05-29 |  |
| 3587 | TaiwanStockShareholding | MISMATCH | 4157 | 2009-08-17 | 2026-06-05 | 4152 | 2009-08-17 | 2026-05-29 |  |
| 3594 | TaiwanStockPriceAdj | MISMATCH | 4529 | 2007-12-11 | 2026-06-05 | 4524 | 2007-12-11 | 2026-05-29 |  |
| 3594 | TaiwanStockShareholding | MISMATCH | 3224 | 2013-05-06 | 2026-06-05 | 3219 | 2013-05-06 | 2026-05-29 |  |
| 3596 | TaiwanStockPriceAdj | MISMATCH | 4566 | 2007-10-31 | 2026-06-05 | 4561 | 2007-10-31 | 2026-05-29 |  |
| 3596 | TaiwanStockShareholding | MISMATCH | 4268 | 2009-03-10 | 2026-06-05 | 4263 | 2009-03-10 | 2026-05-29 |  |
| 3596 | TaiwanStockMonthRevenue | MISMATCH | 224 | 2007-11-01 | 2026-06-01 | 223 | 2007-11-01 | 2026-05-01 |  |
| 3605 | TaiwanStockPriceAdj | MISMATCH | 4523 | 2007-12-31 | 2026-06-05 | 4518 | 2007-12-31 | 2026-05-29 |  |
| 3605 | TaiwanStockShareholding | MISMATCH | 4257 | 2009-03-25 | 2026-06-05 | 4252 | 2009-03-25 | 2026-05-29 |  |
| 3605 | TaiwanStockMonthRevenue | MISMATCH | 223 | 2007-12-01 | 2026-06-01 | 222 | 2007-12-01 | 2026-05-01 |  |
| 3607 | TaiwanStockPriceAdj | MISMATCH | 4502 | 2008-01-29 | 2026-06-05 | 4497 | 2008-01-29 | 2026-05-29 |  |
| 3607 | TaiwanStockShareholding | MISMATCH | 4106 | 2009-10-27 | 2026-06-05 | 4101 | 2009-10-27 | 2026-05-29 |  |
| 3615 | TaiwanStockPriceAdj | MISMATCH | 4374 | 2008-07-25 | 2026-06-05 | 4369 | 2008-07-25 | 2026-05-29 |  |
| 3615 | TaiwanStockShareholding | MISMATCH | 4243 | 2009-04-14 | 2026-06-05 | 4238 | 2009-04-14 | 2026-05-29 |  |
| 3622 | TaiwanStockPriceAdj | MISMATCH | 4466 | 2008-03-31 | 2026-06-05 | 4461 | 2008-03-31 | 2026-05-29 |  |
| 3622 | TaiwanStockShareholding | MISMATCH | 4258 | 2009-03-24 | 2026-06-05 | 4253 | 2009-03-24 | 2026-05-29 |  |
| 3624 | TaiwanStockPriceAdj | MISMATCH | 4431 | 2008-05-16 | 2026-06-05 | 4426 | 2008-05-16 | 2026-05-29 |  |
| 3624 | TaiwanStockShareholding | MISMATCH | 3759 | 2011-03-15 | 2026-06-05 | 3754 | 2011-03-15 | 2026-05-29 |  |
| 3630 | TaiwanStockPriceAdj | MISMATCH | 4384 | 2008-07-22 | 2026-06-05 | 4379 | 2008-07-22 | 2026-05-29 |  |
| 3630 | TaiwanStockShareholding | MISMATCH | 3984 | 2010-04-23 | 2026-06-05 | 3979 | 2010-04-23 | 2026-05-29 |  |
| 3630 | TaiwanStockMonthRevenue | MISMATCH | 216 | 2008-07-01 | 2026-06-01 | 215 | 2008-07-01 | 2026-05-01 |  |
| 3645 | TaiwanStockPriceAdj | MISMATCH | 3848 | 2010-09-17 | 2026-06-05 | 3843 | 2010-09-17 | 2026-05-29 |  |
| 3645 | TaiwanStockShareholding | MISMATCH | 3619 | 2011-10-04 | 2026-06-05 | 3614 | 2011-10-04 | 2026-05-29 |  |
| 3653 | TaiwanStockPriceAdj | MISMATCH | 4301 | 2008-11-21 | 2026-06-05 | 4296 | 2008-11-21 | 2026-05-29 |  |
| 3653 | TaiwanStockShareholding | MISMATCH | 4091 | 2009-11-17 | 2026-06-05 | 4086 | 2009-11-17 | 2026-05-29 |  |
| 3653 | TaiwanStockMonthRevenue | MISMATCH | 212 | 2008-11-01 | 2026-06-01 | 211 | 2008-11-01 | 2026-05-01 |  |
| 3665 | TaiwanStockPriceAdj | MISMATCH | 3704 | 2011-04-21 | 2026-06-05 | 3699 | 2011-04-21 | 2026-05-29 |  |
| 3665 | TaiwanStockShareholding | MISMATCH | 3735 | 2011-04-20 | 2026-06-05 | 3730 | 2011-04-20 | 2026-05-29 |  |
| 3665 | TaiwanStockMonthRevenue | MISMATCH | 183 | 2011-04-01 | 2026-06-01 | 182 | 2011-04-01 | 2026-05-01 |  |
| 3673 | TaiwanStockPriceAdj | MISMATCH | 3818 | 2010-11-01 | 2026-06-05 | 3813 | 2010-11-01 | 2026-05-29 |  |
| 3673 | TaiwanStockShareholding | MISMATCH | 3852 | 2010-10-28 | 2026-06-05 | 3847 | 2010-10-28 | 2026-05-29 |  |
| 3680 | TaiwanStockPriceAdj | MISMATCH | 4077 | 2009-10-19 | 2026-06-05 | 4072 | 2009-10-19 | 2026-05-29 |  |
| 3680 | TaiwanStockShareholding | MISMATCH | 3643 | 2011-08-30 | 2026-06-05 | 3638 | 2011-08-30 | 2026-05-29 |  |
| 3687 | TaiwanStockPriceAdj | MISMATCH | 4021 | 2010-01-06 | 2026-06-05 | 4016 | 2010-01-06 | 2026-05-29 |  |
| 3687 | TaiwanStockShareholding | MISMATCH | 3800 | 2011-01-10 | 2026-06-05 | 3795 | 2011-01-10 | 2026-05-29 |  |
| 3689 | TaiwanStockPriceAdj | MISMATCH | 3782 | 2010-12-17 | 2026-06-05 | 3777 | 2010-12-17 | 2026-05-29 |  |
| 3689 | TaiwanStockShareholding | MISMATCH | 3358 | 2012-10-17 | 2026-06-05 | 3353 | 2012-10-17 | 2026-05-29 |  |
| 3689 | TaiwanStockMonthRevenue | MISMATCH | 198 | 2010-01-01 | 2026-06-01 | 197 | 2010-01-01 | 2026-05-01 |  |
| 3693 | TaiwanStockPriceAdj | MISMATCH | 3935 | 2010-05-11 | 2026-06-05 | 3930 | 2010-05-11 | 2026-05-29 |  |
| 3693 | TaiwanStockShareholding | MISMATCH | 3082 | 2013-11-26 | 2026-06-05 | 3077 | 2013-11-26 | 2026-05-29 |  |
| 3694 | TaiwanStockPriceAdj | MISMATCH | 4067 | 2009-11-02 | 2026-06-05 | 4062 | 2009-11-02 | 2026-05-29 |  |
| 3694 | TaiwanStockShareholding | MISMATCH | 3728 | 2011-04-29 | 2026-06-05 | 3723 | 2011-04-29 | 2026-05-29 |  |
| 3702 | TaiwanStockPriceAdj | MISMATCH | 5050 | 2005-11-10 | 2026-06-05 | 5045 | 2005-11-10 | 2026-05-29 |  |
| 3702 | TaiwanStockShareholding | MISMATCH | 5100 | 2005-11-08 | 2026-06-05 | 5095 | 2005-11-08 | 2026-05-29 |  |
| 3703 | TaiwanStockPriceAdj | MISMATCH | 3963 | 2010-04-07 | 2026-06-05 | 3958 | 2010-04-07 | 2026-05-29 |  |
| 3703 | TaiwanStockShareholding | MISMATCH | 3996 | 2010-04-07 | 2026-06-05 | 3991 | 2010-04-07 | 2026-05-29 |  |
| 3704 | TaiwanStockPriceAdj | MISMATCH | 3865 | 2010-08-13 | 2026-06-05 | 3860 | 2010-08-13 | 2026-05-29 |  |
| 3704 | TaiwanStockShareholding | MISMATCH | 3905 | 2010-08-13 | 2026-06-05 | 3900 | 2010-08-13 | 2026-05-29 |  |
| 3706 | TaiwanStockPriceAdj | MISMATCH | 3106 | 2013-09-13 | 2026-06-05 | 3101 | 2013-09-13 | 2026-05-29 |  |
| 3706 | TaiwanStockShareholding | MISMATCH | 3135 | 2012-03-29 | 2026-06-05 | 3130 | 2012-03-29 | 2026-05-29 |  |
| 4105 | TaiwanStockPriceAdj | MISMATCH | 6009 | 2001-09-28 | 2026-06-05 | 6004 | 2001-09-28 | 2026-05-29 |  |
| 4105 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 4105 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 4114 | TaiwanStockPriceAdj | MISMATCH | 5611 | 2003-05-13 | 2026-06-05 | 5606 | 2003-05-13 | 2026-05-29 |  |
| 4114 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 4119 | TaiwanStockPriceAdj | MISMATCH | 5505 | 2004-01-08 | 2026-06-05 | 5500 | 2004-01-08 | 2026-05-29 |  |
| 4119 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 4306 | TaiwanStockPriceAdj | MISMATCH | 5543 | 2003-08-04 | 2026-06-05 | 5538 | 2003-08-04 | 2026-05-29 |  |
| 4306 | TaiwanStockShareholding | MISMATCH | 4550 | 2008-01-18 | 2026-06-05 | 4545 | 2008-01-18 | 2026-05-29 |  |
| 4551 | TaiwanStockPriceAdj | MISMATCH | 3024 | 2014-01-09 | 2026-06-05 | 3019 | 2014-01-09 | 2026-05-29 |  |
| 4551 | TaiwanStockShareholding | MISMATCH | 2663 | 2012-03-29 | 2026-06-05 | 2658 | 2012-03-29 | 2026-05-29 |  |
| 4721 | TaiwanStockPriceAdj | MISMATCH | 6146 | 2001-03-12 | 2026-06-05 | 6141 | 2001-03-12 | 2026-05-29 |  |
| 4721 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 4739 | TaiwanStockPriceAdj | MISMATCH | 3787 | 2010-12-14 | 2026-06-05 | 3782 | 2010-12-14 | 2026-05-29 |  |
| 4739 | TaiwanStockShareholding | MISMATCH | 2147 | 2012-03-29 | 2026-06-05 | 2142 | 2012-03-29 | 2026-05-29 |  |
| 4739 | TaiwanStockDividend | MISMATCH | 21 | 2011-09-07 | 2026-06-28 | 20 | 2011-09-07 | 2025-12-24 |  |
| 4746 | TaiwanStockPriceAdj | MISMATCH | 4047 | 2009-11-30 | 2026-06-05 | 4042 | 2009-11-30 | 2026-05-29 |  |
| 4746 | TaiwanStockShareholding | MISMATCH | 3770 | 2011-02-25 | 2026-06-05 | 3765 | 2011-02-25 | 2026-05-29 |  |
| 4746 | TaiwanStockMonthRevenue | MISMATCH | 202 | 2009-09-01 | 2026-06-01 | 201 | 2009-09-01 | 2026-05-01 |  |
| 4763 | TaiwanStockPriceAdj | MISMATCH | 2569 | 2015-11-10 | 2026-06-05 | 2564 | 2015-11-10 | 2026-05-29 |  |
| 4763 | TaiwanStockShareholding | MISMATCH | 2601 | 2012-03-29 | 2026-06-05 | 2596 | 2012-03-29 | 2026-05-29 |  |
| 4912 | TaiwanStockPriceAdj | MISMATCH | 4033 | 2009-12-18 | 2026-06-05 | 4028 | 2009-12-18 | 2026-05-29 |  |
| 4912 | TaiwanStockShareholding | MISMATCH | 2718 | 2012-03-29 | 2026-06-05 | 2713 | 2012-03-29 | 2026-05-29 |  |
| 4915 | TaiwanStockPriceAdj | MISMATCH | 4021 | 2010-01-04 | 2026-06-05 | 4016 | 2010-01-04 | 2026-05-29 |  |
| 4915 | TaiwanStockShareholding | MISMATCH | 3367 | 2012-03-29 | 2026-06-05 | 3362 | 2012-03-29 | 2026-05-29 |  |
| 4915 | TaiwanStockMonthRevenue | MISMATCH | 199 | 2009-12-01 | 2026-06-01 | 198 | 2009-12-01 | 2026-05-01 |  |
| 4916 | TaiwanStockPriceAdj | MISMATCH | 4021 | 2009-12-28 | 2026-06-05 | 4016 | 2009-12-28 | 2026-05-29 |  |
| 4916 | TaiwanStockShareholding | MISMATCH | 3087 | 2012-03-29 | 2026-06-05 | 3082 | 2012-03-29 | 2026-05-29 |  |
| 4938 | TaiwanStockPriceAdj | MISMATCH | 3908 | 2010-06-25 | 2026-06-05 | 3903 | 2010-06-25 | 2026-05-29 |  |
| 4938 | TaiwanStockShareholding | MISMATCH | 3942 | 2010-06-23 | 2026-06-05 | 3937 | 2010-06-23 | 2026-05-29 |  |
| 4938 | TaiwanStockDividend | MISMATCH | 15 | 2011-09-18 | 2026-06-21 | 14 | 2011-09-18 | 2025-06-23 |  |
| 4952 | TaiwanStockPriceAdj | MISMATCH | 3876 | 2010-08-10 | 2026-06-05 | 3871 | 2010-08-10 | 2026-05-29 |  |
| 4952 | TaiwanStockShareholding | MISMATCH | 3601 | 2011-10-31 | 2026-06-05 | 3596 | 2011-10-31 | 2026-05-29 |  |
| 4958 | TaiwanStockPriceAdj | MISMATCH | 3530 | 2011-12-27 | 2026-06-05 | 3525 | 2011-12-27 | 2026-05-29 |  |
| 4958 | TaiwanStockShareholding | MISMATCH | 3562 | 2011-12-23 | 2026-06-05 | 3557 | 2011-12-23 | 2026-05-29 |  |
| 4958 | TaiwanStockMonthRevenue | MISMATCH | 175 | 2011-12-01 | 2026-06-01 | 174 | 2011-12-01 | 2026-05-01 |  |
| 4966 | TaiwanStockPriceAdj | MISMATCH | 3603 | 2011-09-14 | 2026-06-05 | 3598 | 2011-09-14 | 2026-05-29 |  |
| 4966 | TaiwanStockShareholding | MISMATCH | 3635 | 2011-09-09 | 2026-06-05 | 3630 | 2011-09-09 | 2026-05-29 |  |
| 4966 | TaiwanStockMonthRevenue | MISMATCH | 179 | 2011-08-01 | 2026-06-01 | 178 | 2011-08-01 | 2026-05-01 |  |
| 4968 | TaiwanStockPriceAdj | MISMATCH | 3832 | 2010-10-04 | 2026-06-05 | 3827 | 2010-10-04 | 2026-05-29 |  |
| 4968 | TaiwanStockShareholding | MISMATCH | 2597 | 2012-03-29 | 2026-06-05 | 2592 | 2012-03-29 | 2026-05-29 |  |
| 4968 | TaiwanStockMonthRevenue | MISMATCH | 190 | 2010-09-01 | 2026-06-01 | 189 | 2010-09-01 | 2026-05-01 |  |
| 4973 | TaiwanStockPriceAdj | MISMATCH | 3763 | 2011-01-12 | 2026-06-05 | 3758 | 2011-01-12 | 2026-05-29 |  |
| 4973 | TaiwanStockShareholding | MISMATCH | 3443 | 2012-06-18 | 2026-06-05 | 3438 | 2012-06-18 | 2026-05-29 |  |
| 4977 | TaiwanStockPriceAdj | MISMATCH | 3276 | 2012-12-27 | 2026-06-05 | 3271 | 2012-12-27 | 2026-05-29 |  |
| 4977 | TaiwanStockShareholding | MISMATCH | 3088 | 2012-03-29 | 2026-06-05 | 3083 | 2012-03-29 | 2026-05-29 |  |
| 4979 | TaiwanStockPriceAdj | MISMATCH | 3806 | 2010-11-17 | 2026-06-05 | 3801 | 2010-11-17 | 2026-05-29 |  |
| 4979 | TaiwanStockShareholding | MISMATCH | 3572 | 2011-12-09 | 2026-06-05 | 3567 | 2011-12-09 | 2026-05-29 |  |
| 5009 | TaiwanStockPriceAdj | MISMATCH | 6775 | 1998-10-19 | 2026-06-05 | 6770 | 1998-10-19 | 2026-05-29 |  |
| 5009 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5203 | TaiwanStockPriceAdj | MISMATCH | 5610 | 2003-08-04 | 2026-06-05 | 5605 | 2003-08-04 | 2026-05-29 |  |
| 5203 | TaiwanStockShareholding | MISMATCH | 5378 | 2004-09-27 | 2026-06-05 | 5373 | 2004-09-27 | 2026-05-29 |  |
| 5203 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 5243 | TaiwanStockPriceAdj | MISMATCH | 3056 | 2013-11-26 | 2026-06-05 | 3051 | 2013-11-26 | 2026-05-29 |  |
| 5243 | TaiwanStockShareholding | MISMATCH | 3085 | 2012-03-29 | 2026-06-05 | 3080 | 2012-03-29 | 2026-05-29 |  |
| 5243 | TaiwanStockMonthRevenue | MISMATCH | 153 | 2013-10-01 | 2026-06-01 | 152 | 2013-10-01 | 2026-05-01 |  |
| 5258 | TaiwanStockPriceAdj | MISMATCH | 3551 | 2011-11-15 | 2026-06-05 | 3546 | 2011-11-15 | 2026-05-29 |  |
| 5258 | TaiwanStockShareholding | MISMATCH | 2317 | 2012-03-29 | 2026-06-05 | 2312 | 2012-03-29 | 2026-05-29 |  |
| 5269 | TaiwanStockPriceAdj | MISMATCH | 3515 | 2012-01-13 | 2026-06-05 | 3510 | 2012-01-13 | 2026-05-29 |  |
| 5269 | TaiwanStockShareholding | MISMATCH | 3320 | 2012-03-29 | 2026-06-05 | 3315 | 2012-03-29 | 2026-05-29 |  |
| 5285 | TaiwanStockPriceAdj | MISMATCH | 3350 | 2012-09-13 | 2026-06-05 | 3345 | 2012-09-13 | 2026-05-29 |  |
| 5285 | TaiwanStockShareholding | MISMATCH | 3024 | 2012-03-29 | 2026-06-05 | 3019 | 2012-03-29 | 2026-05-29 |  |
| 5285 | TaiwanStockDividend | MISMATCH | 14 | 2013-08-21 | 2026-07-01 | 13 | 2013-08-21 | 2025-07-07 |  |
| 5289 | TaiwanStockPriceAdj | MISMATCH | 3333 | 2012-10-08 | 2026-06-05 | 3328 | 2012-10-08 | 2026-05-29 |  |
| 5289 | TaiwanStockShareholding | MISMATCH | 3082 | 2013-11-26 | 2026-06-05 | 3077 | 2013-11-26 | 2026-05-29 |  |
| 5289 | TaiwanStockMonthRevenue | MISMATCH | 166 | 2012-09-01 | 2026-06-01 | 165 | 2012-09-01 | 2026-05-01 |  |
| 5347 | TaiwanStockPriceAdj | MISMATCH | 6925 | 1998-03-26 | 2026-06-05 | 6920 | 1998-03-26 | 2026-05-29 |  |
| 5347 | TaiwanStockShareholding | MISMATCH | 5382 | 2004-08-04 | 2026-06-05 | 5377 | 2004-08-04 | 2026-05-29 |  |
| 5371 | TaiwanStockPriceAdj | MISMATCH | 6692 | 1999-01-21 | 2026-06-05 | 6687 | 1999-01-21 | 2026-05-29 |  |
| 5371 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5371 | TaiwanStockDividend | MISMATCH | 21 | 2005-08-09 | 2026-06-26 | 20 | 2005-08-09 | 2025-07-19 |  |
| 5388 | TaiwanStockPriceAdj | MISMATCH | 5551 | 2003-08-04 | 2026-06-05 | 5546 | 2003-08-04 | 2026-05-29 |  |
| 5388 | TaiwanStockShareholding | MISMATCH | 4584 | 2007-11-30 | 2026-06-05 | 4579 | 2007-11-30 | 2026-05-29 |  |
| 5388 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 5425 | TaiwanStockPriceAdj | MISMATCH | 6426 | 2000-02-22 | 2026-06-05 | 6421 | 2000-02-22 | 2026-05-29 |  |
| 5425 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5434 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 5434 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 5439 | TaiwanStockPriceAdj | MISMATCH | 6330 | 2000-06-27 | 2026-06-05 | 6325 | 2000-06-27 | 2026-05-29 |  |
| 5439 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5443 | TaiwanStockPriceAdj | MISMATCH | 6959 | 1998-02-10 | 2026-06-05 | 6954 | 1998-02-10 | 2026-05-29 |  |
| 5443 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5469 | TaiwanStockPriceAdj | MISMATCH | 5615 | 2003-08-04 | 2026-06-05 | 5610 | 2003-08-04 | 2026-05-29 |  |
| 5469 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 5457 | TaiwanStockPriceAdj | MISMATCH | 6251 | 2000-09-30 | 2026-06-05 | 6246 | 2000-09-30 | 2026-05-29 |  |
| 5457 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5471 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 5471 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 5471 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 5478 | TaiwanStockPriceAdj | MISMATCH | 6132 | 2001-03-30 | 2026-06-05 | 6127 | 2001-03-30 | 2026-05-29 |  |
| 5478 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5483 | TaiwanStockPriceAdj | MISMATCH | 6151 | 2001-03-05 | 2026-06-05 | 6146 | 2001-03-05 | 2026-05-29 |  |
| 5483 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5483 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 5512 | TaiwanStockPriceAdj | MISMATCH | 6744 | 1998-12-07 | 2026-06-05 | 6739 | 1998-12-07 | 2026-05-29 |  |
| 5512 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 5607 | TaiwanStockPriceAdj | MISMATCH | 5616 | 2003-08-04 | 2026-06-05 | 5611 | 2003-08-04 | 2026-05-29 |  |
| 5607 | TaiwanStockShareholding | MISMATCH | 5332 | 2004-12-02 | 2026-06-05 | 5327 | 2004-12-02 | 2026-05-29 |  |
| 5608 | TaiwanStockPriceAdj | MISMATCH | 5616 | 2003-08-04 | 2026-06-05 | 5611 | 2003-08-04 | 2026-05-29 |  |
| 5608 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 5608 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 5706 | TaiwanStockPriceAdj | MISMATCH | 5551 | 2003-08-04 | 2026-06-05 | 5546 | 2003-08-04 | 2026-05-29 |  |
| 5706 | TaiwanStockShareholding | MISMATCH | 3608 | 2011-10-20 | 2026-06-05 | 3603 | 2011-10-20 | 2026-05-29 |  |
| 5871 | TaiwanStockPriceAdj | MISMATCH | 3649 | 2011-07-11 | 2026-06-05 | 3644 | 2011-07-11 | 2026-05-29 |  |
| 5871 | TaiwanStockShareholding | MISMATCH | 3571 | 2011-12-12 | 2026-06-05 | 3566 | 2011-12-12 | 2026-05-29 |  |
| 6104 | TaiwanStockPriceAdj | MISMATCH | 6096 | 2001-05-23 | 2026-06-05 | 6091 | 2001-05-23 | 2026-05-29 |  |
| 6104 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6104 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 6118 | TaiwanStockPriceAdj | MISMATCH | 5991 | 2001-10-26 | 2026-06-05 | 5986 | 2001-10-26 | 2026-05-29 |  |
| 6118 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6121 | TaiwanStockPriceAdj | MISMATCH | 5958 | 2001-11-28 | 2026-06-05 | 5953 | 2001-11-28 | 2026-05-29 |  |
| 6121 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6121 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 6125 | TaiwanStockPriceAdj | MISMATCH | 5928 | 2002-01-24 | 2026-06-05 | 5923 | 2002-01-24 | 2026-05-29 |  |
| 6125 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6126 | TaiwanStockPriceAdj | MISMATCH | 5931 | 2002-01-21 | 2026-06-05 | 5926 | 2002-01-21 | 2026-05-29 |  |
| 6126 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6138 | TaiwanStockPriceAdj | MISMATCH | 5917 | 2002-01-21 | 2026-06-05 | 5912 | 2002-01-21 | 2026-05-29 |  |
| 6138 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6138 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 6139 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 6139 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 6143 | TaiwanStockPriceAdj | MISMATCH | 5925 | 2002-01-29 | 2026-06-05 | 5920 | 2002-01-29 | 2026-05-29 |  |
| 6143 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6147 | TaiwanStockPriceAdj | MISMATCH | 5922 | 2002-02-01 | 2026-06-05 | 5917 | 2002-02-01 | 2026-05-29 |  |
| 6147 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6150 | TaiwanStockPriceAdj | MISMATCH | 5885 | 2002-03-12 | 2026-06-05 | 5880 | 2002-03-12 | 2026-05-29 |  |
| 6150 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6161 | TaiwanStockPriceAdj | MISMATCH | 5896 | 2002-03-04 | 2026-06-05 | 5891 | 2002-03-04 | 2026-05-29 |  |
| 6161 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6163 | TaiwanStockPriceAdj | MISMATCH | 5897 | 2002-03-20 | 2026-06-05 | 5892 | 2002-03-20 | 2026-05-29 |  |
| 6163 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6166 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 6166 | TaiwanStockShareholding | MISMATCH | 5350 | 2004-11-08 | 2026-06-05 | 5345 | 2004-11-08 | 2026-05-29 |  |
| 6168 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 6168 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 6173 | TaiwanStockPriceAdj | MISMATCH | 5867 | 2002-04-22 | 2026-06-05 | 5862 | 2002-04-22 | 2026-05-29 |  |
| 6173 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6170 | TaiwanStockPriceAdj | MISMATCH | 5878 | 2002-04-17 | 2026-06-05 | 5873 | 2002-04-17 | 2026-05-29 |  |
| 6170 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6175 | TaiwanStockPriceAdj | MISMATCH | 5840 | 2002-06-11 | 2026-06-05 | 5835 | 2002-06-11 | 2026-05-29 |  |
| 6175 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6176 | TaiwanStockPriceAdj | MISMATCH | 5551 | 2003-08-04 | 2026-06-05 | 5546 | 2003-08-04 | 2026-05-29 |  |
| 6176 | TaiwanStockShareholding | MISMATCH | 4724 | 2007-05-14 | 2026-06-05 | 4719 | 2007-05-14 | 2026-05-29 |  |
| 6187 | TaiwanStockPriceAdj | MISMATCH | 5762 | 2002-09-30 | 2026-06-05 | 5757 | 2002-09-30 | 2026-05-29 |  |
| 6187 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6177 | TaiwanStockPriceAdj | MISMATCH | 5551 | 2003-08-04 | 2026-06-05 | 5546 | 2003-08-04 | 2026-05-29 |  |
| 6177 | TaiwanStockShareholding | MISMATCH | 3164 | 2012-03-29 | 2026-06-05 | 3159 | 2012-03-29 | 2026-05-29 |  |
| 6188 | TaiwanStockPriceAdj | MISMATCH | 5784 | 2002-08-28 | 2026-06-05 | 5779 | 2002-08-28 | 2026-05-29 |  |
| 6188 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6188 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 6190 | TaiwanStockPriceAdj | MISMATCH | 5803 | 2002-08-01 | 2026-06-05 | 5798 | 2002-08-01 | 2026-05-29 |  |
| 6190 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6191 | TaiwanStockPriceAdj | MISMATCH | 5546 | 2003-08-04 | 2026-06-05 | 5541 | 2003-08-04 | 2026-05-29 |  |
| 6191 | TaiwanStockShareholding | MISMATCH | 4615 | 2007-10-18 | 2026-06-05 | 4610 | 2007-10-18 | 2026-05-29 |  |
| 6196 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 6196 | TaiwanStockShareholding | MISMATCH | 5466 | 2004-05-21 | 2026-06-05 | 5461 | 2004-05-21 | 2026-05-29 |  |
| 6196 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 6197 | TaiwanStockPriceAdj | MISMATCH | 5612 | 2003-08-04 | 2026-06-05 | 5607 | 2003-08-04 | 2026-05-29 |  |
| 6197 | TaiwanStockShareholding | MISMATCH | 5350 | 2004-11-08 | 2026-06-05 | 5345 | 2004-11-08 | 2026-05-29 |  |
| 6197 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 6202 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 6202 | TaiwanStockShareholding | MISMATCH | 5378 | 2004-09-27 | 2026-06-05 | 5373 | 2004-09-27 | 2026-05-29 |  |
| 6209 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 6209 | TaiwanStockShareholding | MISMATCH | 5350 | 2004-11-08 | 2026-06-05 | 5345 | 2004-11-08 | 2026-05-29 |  |
| 6207 | TaiwanStockPriceAdj | MISMATCH | 5695 | 2002-12-24 | 2026-06-05 | 5690 | 2002-12-24 | 2026-05-29 |  |
| 6207 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6213 | TaiwanStockPriceAdj | MISMATCH | 5551 | 2003-08-04 | 2026-06-05 | 5546 | 2003-08-04 | 2026-05-29 |  |
| 6213 | TaiwanStockShareholding | MISMATCH | 4550 | 2008-01-18 | 2026-06-05 | 4545 | 2008-01-18 | 2026-05-29 |  |
| 6213 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 6231 | TaiwanStockPriceAdj | MISMATCH | 5677 | 2003-01-24 | 2026-06-05 | 5672 | 2003-01-24 | 2026-05-29 |  |
| 6231 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6235 | TaiwanStockPriceAdj | MISMATCH | 5609 | 2003-08-04 | 2026-06-05 | 5604 | 2003-08-04 | 2026-05-29 |  |
| 6235 | TaiwanStockShareholding | MISMATCH | 5401 | 2004-08-23 | 2026-06-05 | 5396 | 2004-08-23 | 2026-05-29 |  |
| 6235 | TaiwanStockMonthRevenue | MISMATCH | 291 | 2002-03-01 | 2026-06-01 | 290 | 2002-03-01 | 2026-05-01 |  |
| 6239 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 6239 | TaiwanStockShareholding | MISMATCH | 5350 | 2004-11-08 | 2026-06-05 | 5345 | 2004-11-08 | 2026-05-29 |  |
| 6239 | TaiwanStockMonthRevenue | MISMATCH | 291 | 2002-04-01 | 2026-06-01 | 290 | 2002-04-01 | 2026-05-01 |  |
| 6245 | TaiwanStockPriceAdj | MISMATCH | 5606 | 2003-05-20 | 2026-06-05 | 5601 | 2003-05-20 | 2026-05-29 |  |
| 6245 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6257 | TaiwanStockPriceAdj | MISMATCH | 5606 | 2003-08-26 | 2026-06-05 | 5601 | 2003-08-26 | 2026-05-29 |  |
| 6257 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 6257 | TaiwanStockMonthRevenue | MISMATCH | 283 | 2002-12-01 | 2026-06-01 | 282 | 2002-12-01 | 2026-05-01 |  |
| 6265 | TaiwanStockPriceAdj | MISMATCH | 5495 | 2003-10-27 | 2026-06-05 | 5490 | 2003-10-27 | 2026-05-29 |  |
| 6265 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6271 | TaiwanStockPriceAdj | MISMATCH | 4756 | 2006-12-19 | 2026-06-05 | 4751 | 2006-12-19 | 2026-05-29 |  |
| 6271 | TaiwanStockShareholding | MISMATCH | 4595 | 2007-11-15 | 2026-06-05 | 4590 | 2007-11-15 | 2026-05-29 |  |
| 6274 | TaiwanStockPriceAdj | MISMATCH | 5456 | 2003-12-19 | 2026-06-05 | 5451 | 2003-12-19 | 2026-05-29 |  |
| 6274 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6274 | TaiwanStockMonthRevenue | MISMATCH | 283 | 2002-12-01 | 2026-06-01 | 282 | 2002-12-01 | 2026-05-01 |  |
| 6275 | TaiwanStockPriceAdj | MISMATCH | 5203 | 2004-12-24 | 2026-06-05 | 5198 | 2004-12-24 | 2026-05-29 |  |
| 6275 | TaiwanStockShareholding | MISMATCH | 5301 | 2004-12-22 | 2026-06-05 | 5296 | 2004-12-22 | 2026-05-29 |  |
| 6279 | TaiwanStockPriceAdj | MISMATCH | 5472 | 2003-11-27 | 2026-06-05 | 5467 | 2003-11-27 | 2026-05-29 |  |
| 6279 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6279 | TaiwanStockMonthRevenue | MISMATCH | 282 | 2003-01-01 | 2026-06-01 | 281 | 2003-01-01 | 2026-05-01 |  |
| 6278 | TaiwanStockPriceAdj | MISMATCH | 5400 | 2004-03-15 | 2026-06-05 | 5395 | 2004-03-15 | 2026-05-29 |  |
| 6278 | TaiwanStockShareholding | MISMATCH | 3899 | 2010-08-23 | 2026-06-05 | 3894 | 2010-08-23 | 2026-05-29 |  |
| 6278 | TaiwanStockMonthRevenue | MISMATCH | 282 | 2003-01-01 | 2026-06-01 | 281 | 2003-01-01 | 2026-05-01 |  |
| 6284 | TaiwanStockPriceAdj | MISMATCH | 5319 | 2004-06-30 | 2026-06-05 | 5314 | 2004-06-30 | 2026-05-29 |  |
| 6284 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6282 | TaiwanStockPriceAdj | MISMATCH | 5596 | 2003-09-09 | 2026-06-05 | 5591 | 2003-09-09 | 2026-05-29 |  |
| 6282 | TaiwanStockShareholding | MISMATCH | 5536 | 2004-02-12 | 2026-06-05 | 5531 | 2004-02-12 | 2026-05-29 |  |
| 6285 | TaiwanStockPriceAdj | MISMATCH | 5587 | 2003-09-23 | 2026-06-05 | 5582 | 2003-09-23 | 2026-05-29 |  |
| 6285 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 6290 | TaiwanStockPriceAdj | MISMATCH | 5096 | 2005-06-06 | 2026-06-05 | 5091 | 2005-06-06 | 2026-05-29 |  |
| 6290 | TaiwanStockShareholding | MISMATCH | 5198 | 2005-06-02 | 2026-06-05 | 5193 | 2005-06-02 | 2026-05-29 |  |
| 6411 | TaiwanStockPriceAdj | MISMATCH | 3277 | 2012-12-26 | 2026-06-05 | 3272 | 2012-12-26 | 2026-05-29 |  |
| 6411 | TaiwanStockShareholding | MISMATCH | 3014 | 2014-03-10 | 2026-06-05 | 3009 | 2014-03-10 | 2026-05-29 |  |
| 6412 | TaiwanStockPriceAdj | MISMATCH | 3272 | 2013-01-04 | 2026-06-05 | 3267 | 2013-01-04 | 2026-05-29 |  |
| 6412 | TaiwanStockShareholding | MISMATCH | 3096 | 2012-03-29 | 2026-06-05 | 3091 | 2012-03-29 | 2026-05-29 |  |
| 6412 | TaiwanStockMonthRevenue | MISMATCH | 164 | 2012-11-01 | 2026-06-01 | 163 | 2012-11-01 | 2026-05-01 |  |
| 6414 | TaiwanStockPriceAdj | MISMATCH | 3282 | 2012-12-19 | 2026-06-05 | 3277 | 2012-12-19 | 2026-05-29 |  |
| 6414 | TaiwanStockShareholding | MISMATCH | 3002 | 2012-03-29 | 2026-06-05 | 2997 | 2012-03-29 | 2026-05-29 |  |
| 6414 | TaiwanStockMonthRevenue | MISMATCH | 163 | 2012-12-01 | 2026-06-01 | 162 | 2012-12-01 | 2026-05-01 |  |
| 6414 | TaiwanStockDividend | MISMATCH | 16 | 2013-07-09 | 2026-06-22 | 15 | 2013-07-09 | 2025-07-05 |  |
| 6435 | TaiwanStockPriceAdj | MISMATCH | 3042 | 2013-12-13 | 2026-06-05 | 3037 | 2013-12-13 | 2026-05-29 |  |
| 6435 | TaiwanStockShareholding | MISMATCH | 2558 | 2016-01-06 | 2026-06-05 | 2553 | 2016-01-06 | 2026-05-29 |  |
| 6435 | TaiwanStockMonthRevenue | MISMATCH | 154 | 2013-09-01 | 2026-06-01 | 153 | 2013-09-01 | 2026-05-01 |  |
| 6438 | TaiwanStockPriceAdj | MISMATCH | 3051 | 2013-11-26 | 2026-06-05 | 3046 | 2013-11-26 | 2026-05-29 |  |
| 6438 | TaiwanStockShareholding | MISMATCH | 2233 | 2017-04-24 | 2026-06-05 | 2228 | 2017-04-24 | 2026-05-29 |  |
| 6438 | TaiwanStockMonthRevenue | MISMATCH | 153 | 2013-10-01 | 2026-06-01 | 152 | 2013-10-01 | 2026-05-01 |  |
| 6438 | TaiwanStockDividend | MISMATCH | 20 | 2014-07-19 | 2026-06-26 | 19 | 2014-07-19 | 2025-12-18 |  |
| 6477 | TaiwanStockPriceAdj | MISMATCH | 2826 | 2014-10-29 | 2026-06-05 | 2821 | 2014-10-29 | 2026-05-29 |  |
| 6477 | TaiwanStockShareholding | MISMATCH | 2450 | 2012-03-29 | 2026-06-05 | 2445 | 2012-03-29 | 2026-05-29 |  |
| 6488 | TaiwanStockPriceAdj | MISMATCH | 2828 | 2014-10-29 | 2026-06-05 | 2823 | 2014-10-29 | 2026-05-29 |  |
| 6488 | TaiwanStockShareholding | MISMATCH | 2628 | 2015-09-24 | 2026-06-05 | 2623 | 2015-09-24 | 2026-05-29 |  |
| 6488 | TaiwanStockMonthRevenue | MISMATCH | 141 | 2014-10-01 | 2026-06-01 | 140 | 2014-10-01 | 2026-05-01 |  |
| 6505 | TaiwanStockPriceAdj | MISMATCH | 5519 | 2003-12-29 | 2026-06-05 | 5514 | 2003-12-29 | 2026-05-29 |  |
| 6505 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 6505 | TaiwanStockDividend | MISMATCH | 22 | 2005-07-25 | 2026-06-29 | 21 | 2005-07-25 | 2025-07-01 |  |
| 6509 | TaiwanStockPriceAdj | MISMATCH | 5381 | 2004-04-13 | 2026-06-05 | 5376 | 2004-04-13 | 2026-05-29 |  |
| 6509 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 6510 | TaiwanStockPriceAdj | MISMATCH | 2769 | 2015-01-21 | 2026-06-05 | 2764 | 2015-01-21 | 2026-05-29 |  |
| 6510 | TaiwanStockShareholding | MISMATCH | 2508 | 2016-03-23 | 2026-06-05 | 2503 | 2016-03-23 | 2026-05-29 |  |
| 6510 | TaiwanStockMonthRevenue | MISMATCH | 137 | 2015-02-01 | 2026-06-01 | 136 | 2015-02-01 | 2026-05-01 |  |
| 6532 | TaiwanStockPriceAdj | MISMATCH | 2688 | 2015-05-26 | 2026-06-05 | 2683 | 2015-05-26 | 2026-05-29 |  |
| 6532 | TaiwanStockShareholding | MISMATCH | 2382 | 2016-09-23 | 2026-06-05 | 2377 | 2016-09-23 | 2026-05-29 |  |
| 6532 | TaiwanStockMonthRevenue | MISMATCH | 134 | 2015-05-01 | 2026-06-01 | 133 | 2015-05-01 | 2026-05-01 |  |
| 6532 | TaiwanStockDividend | MISMATCH | 12 | 2015-07-28 | 2026-06-21 | 11 | 2015-07-28 | 2025-06-20 |  |
| 6531 | TaiwanStockPriceAdj | MISMATCH | 2670 | 2015-06-15 | 2026-06-05 | 2665 | 2015-06-15 | 2026-05-29 |  |
| 6531 | TaiwanStockShareholding | MISMATCH | 2464 | 2012-03-29 | 2026-06-05 | 2459 | 2012-03-29 | 2026-05-29 |  |
| 6531 | TaiwanStockMonthRevenue | MISMATCH | 138 | 2015-01-01 | 2026-06-01 | 137 | 2015-01-01 | 2026-05-01 |  |
| 6531 | TaiwanStockDividend | MISMATCH | 12 | 2015-09-07 | 2026-06-28 | 11 | 2015-09-07 | 2025-06-24 |  |
| 8011 | TaiwanStockPriceAdj | MISMATCH | 4258 | 2003-08-04 | 2026-06-05 | 4253 | 2003-08-04 | 2026-05-29 |  |
| 8011 | TaiwanStockShareholding | MISMATCH | 3631 | 2011-09-16 | 2026-06-05 | 3626 | 2011-09-16 | 2026-05-29 |  |
| 8016 | TaiwanStockPriceAdj | MISMATCH | 5520 | 2003-12-26 | 2026-06-05 | 5515 | 2003-12-26 | 2026-05-29 |  |
| 8016 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 8021 | TaiwanStockPriceAdj | MISMATCH | 5196 | 2004-12-22 | 2026-06-05 | 5191 | 2004-12-22 | 2026-05-29 |  |
| 8021 | TaiwanStockShareholding | MISMATCH | 4550 | 2008-01-18 | 2026-06-05 | 4545 | 2008-01-18 | 2026-05-29 |  |
| 8021 | TaiwanStockMonthRevenue | MISMATCH | 282 | 2003-01-01 | 2026-06-01 | 281 | 2003-01-01 | 2026-05-01 |  |
| 8039 | TaiwanStockPriceAdj | MISMATCH | 5453 | 2003-12-22 | 2026-06-05 | 5448 | 2003-12-22 | 2026-05-29 |  |
| 8039 | TaiwanStockShareholding | MISMATCH | 4070 | 2009-12-16 | 2026-06-05 | 4065 | 2009-12-16 | 2026-05-29 |  |
| 8039 | TaiwanStockMonthRevenue | MISMATCH | 283 | 2002-12-01 | 2026-06-01 | 282 | 2002-12-01 | 2026-05-01 |  |
| 8043 | TaiwanStockPriceAdj | MISMATCH | 5485 | 2003-10-27 | 2026-06-05 | 5480 | 2003-10-27 | 2026-05-29 |  |
| 8043 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 8046 | TaiwanStockPriceAdj | MISMATCH | 4957 | 2006-04-10 | 2026-06-05 | 4952 | 2006-04-10 | 2026-05-29 |  |
| 8046 | TaiwanStockShareholding | MISMATCH | 4999 | 2006-04-06 | 2026-06-05 | 4994 | 2006-04-06 | 2026-05-29 |  |
| 8046 | TaiwanStockMonthRevenue | MISMATCH | 282 | 2003-01-01 | 2026-06-01 | 281 | 2003-01-01 | 2026-05-01 |  |
| 8064 | TaiwanStockPriceAdj | MISMATCH | 4769 | 2006-09-26 | 2026-06-05 | 4764 | 2006-09-26 | 2026-05-29 |  |
| 8064 | TaiwanStockShareholding | MISMATCH | 4875 | 2006-09-22 | 2026-06-05 | 4870 | 2006-09-22 | 2026-05-29 |  |
| 8064 | TaiwanStockDividend | MISMATCH | 19 | 2005-07-09 | 2026-07-05 | 18 | 2005-07-09 | 2025-07-09 |  |
| 8069 | TaiwanStockPriceAdj | MISMATCH | 5390 | 2004-03-31 | 2026-06-05 | 5385 | 2004-03-31 | 2026-05-29 |  |
| 8069 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 8076 | TaiwanStockPriceAdj | MISMATCH | 5459 | 2003-12-16 | 2026-06-05 | 5454 | 2003-12-16 | 2026-05-29 |  |
| 8076 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 8081 | TaiwanStockPriceAdj | MISMATCH | 5282 | 2004-08-31 | 2026-06-05 | 5277 | 2004-08-31 | 2026-05-29 |  |
| 8081 | TaiwanStockShareholding | MISMATCH | 4314 | 2008-12-29 | 2026-06-05 | 4309 | 2008-12-29 | 2026-05-29 |  |
| 8091 | TaiwanStockPriceAdj | MISMATCH | 5323 | 2004-06-25 | 2026-06-05 | 5318 | 2004-06-25 | 2026-05-29 |  |
| 8091 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 8096 | TaiwanStockPriceAdj | MISMATCH | 5314 | 2004-07-16 | 2026-06-05 | 5309 | 2004-07-16 | 2026-05-29 |  |
| 8096 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 8110 | TaiwanStockPriceAdj | MISMATCH | 4790 | 2006-08-24 | 2026-06-05 | 4785 | 2006-08-24 | 2026-05-29 |  |
| 8110 | TaiwanStockShareholding | MISMATCH | 4607 | 2007-10-30 | 2026-06-05 | 4602 | 2007-10-30 | 2026-05-29 |  |
| 8111 | TaiwanStockPriceAdj | MISMATCH | 5418 | 2004-02-10 | 2026-06-05 | 5413 | 2004-02-10 | 2026-05-29 |  |
| 8111 | TaiwanStockShareholding | MISMATCH | 5394 | 2004-08-04 | 2026-06-05 | 5389 | 2004-08-04 | 2026-05-29 |  |
| 8111 | TaiwanStockMonthRevenue | MISMATCH | 281 | 2003-02-01 | 2026-06-01 | 280 | 2003-02-01 | 2026-05-01 |  |
| 8112 | TaiwanStockPriceAdj | MISMATCH | 5424 | 2004-02-10 | 2026-06-05 | 5419 | 2004-02-10 | 2026-05-29 |  |
| 8112 | TaiwanStockShareholding | MISMATCH | 4564 | 2007-12-28 | 2026-06-05 | 4559 | 2007-12-28 | 2026-05-29 |  |
| 8150 | TaiwanStockPriceAdj | MISMATCH | 3200 | 2013-04-22 | 2026-06-05 | 3195 | 2013-04-22 | 2026-05-29 |  |
| 8150 | TaiwanStockShareholding | MISMATCH | 2992 | 2014-04-10 | 2026-06-05 | 2987 | 2014-04-10 | 2026-05-29 |  |
| 8155 | TaiwanStockPriceAdj | MISMATCH | 3681 | 2011-05-23 | 2026-06-05 | 3676 | 2011-05-23 | 2026-05-29 |  |
| 8155 | TaiwanStockShareholding | MISMATCH | 3315 | 2012-12-17 | 2026-06-05 | 3310 | 2012-12-17 | 2026-05-29 |  |
| 8163 | TaiwanStockPriceAdj | MISMATCH | 4770 | 2006-12-19 | 2026-06-05 | 4765 | 2006-12-19 | 2026-05-29 |  |
| 8163 | TaiwanStockShareholding | MISMATCH | 4587 | 2007-11-27 | 2026-06-05 | 4582 | 2007-11-27 | 2026-05-29 |  |
| 8163 | TaiwanStockMonthRevenue | MISMATCH | 258 | 2005-01-01 | 2026-06-01 | 257 | 2005-01-01 | 2026-05-01 |  |
| 8183 | TaiwanStockPriceAdj | MISMATCH | 5141 | 2005-03-31 | 2026-06-05 | 5136 | 2005-03-31 | 2026-05-29 |  |
| 8183 | TaiwanStockShareholding | MISMATCH | 5240 | 2005-03-29 | 2026-06-05 | 5235 | 2005-03-29 | 2026-05-29 |  |
| 8210 | TaiwanStockPriceAdj | MISMATCH | 5119 | 2005-04-29 | 2026-06-05 | 5114 | 2005-04-29 | 2026-05-29 |  |
| 8210 | TaiwanStockShareholding | MISMATCH | 3579 | 2011-11-30 | 2026-06-05 | 3574 | 2011-11-30 | 2026-05-29 |  |
| 8213 | TaiwanStockPriceAdj | MISMATCH | 4767 | 2006-12-19 | 2026-06-05 | 4762 | 2006-12-19 | 2026-05-29 |  |
| 8213 | TaiwanStockShareholding | MISMATCH | 4064 | 2009-12-24 | 2026-06-05 | 4059 | 2009-12-24 | 2026-05-29 |  |
| 8255 | TaiwanStockPriceAdj | MISMATCH | 4876 | 2006-04-26 | 2026-06-05 | 4871 | 2006-04-26 | 2026-05-29 |  |
| 8255 | TaiwanStockShareholding | MISMATCH | 4982 | 2006-04-24 | 2026-06-05 | 4977 | 2006-04-24 | 2026-05-29 |  |
| 8271 | TaiwanStockPriceAdj | MISMATCH | 4757 | 2006-12-19 | 2026-06-05 | 4752 | 2006-12-19 | 2026-05-29 |  |
| 8271 | TaiwanStockShareholding | MISMATCH | 3809 | 2010-12-28 | 2026-06-05 | 3804 | 2010-12-28 | 2026-05-29 |  |
| 8271 | TaiwanStockMonthRevenue | MISMATCH | 281 | 2003-02-01 | 2026-06-01 | 280 | 2003-02-01 | 2026-05-01 |  |
| 8261 | TaiwanStockPriceAdj | MISMATCH | 5366 | 2004-04-16 | 2026-06-05 | 5361 | 2004-04-16 | 2026-05-29 |  |
| 8261 | TaiwanStockShareholding | MISMATCH | 4074 | 2009-12-10 | 2026-06-05 | 4069 | 2009-12-10 | 2026-05-29 |  |
| 8261 | TaiwanStockMonthRevenue | MISMATCH | 281 | 2003-02-01 | 2026-06-01 | 280 | 2003-02-01 | 2026-05-01 |  |
| 8299 | TaiwanStockPriceAdj | MISMATCH | 5216 | 2004-12-07 | 2026-06-05 | 5211 | 2004-12-07 | 2026-05-29 |  |
| 8299 | TaiwanStockShareholding | MISMATCH | 5314 | 2004-12-03 | 2026-06-05 | 5309 | 2004-12-03 | 2026-05-29 |  |
| 8299 | TaiwanStockMonthRevenue | MISMATCH | 277 | 2003-06-01 | 2026-06-01 | 276 | 2003-06-01 | 2026-05-01 |  |
| 8358 | TaiwanStockPriceAdj | MISMATCH | 4557 | 2007-11-08 | 2026-06-05 | 4552 | 2007-11-08 | 2026-05-29 |  |
| 8358 | TaiwanStockShareholding | MISMATCH | 3876 | 2010-09-24 | 2026-06-05 | 3871 | 2010-09-24 | 2026-05-29 |  |
| 8383 | TaiwanStockPriceAdj | MISMATCH | 5275 | 2004-09-13 | 2026-06-05 | 5270 | 2004-09-13 | 2026-05-29 |  |
| 8383 | TaiwanStockShareholding | MISMATCH | 5371 | 2004-09-09 | 2026-06-05 | 5366 | 2004-09-09 | 2026-05-29 |  |
| 8436 | TaiwanStockPriceAdj | MISMATCH | 3496 | 2012-02-16 | 2026-06-05 | 3491 | 2012-02-16 | 2026-05-29 |  |
| 8436 | TaiwanStockShareholding | MISMATCH | 3134 | 2013-09-11 | 2026-06-05 | 3129 | 2013-09-11 | 2026-05-29 |  |
| 8436 | TaiwanStockMonthRevenue | MISMATCH | 178 | 2011-09-01 | 2026-06-01 | 177 | 2011-09-01 | 2026-05-01 |  |
| 8462 | TaiwanStockPriceAdj | MISMATCH | 2867 | 2014-09-01 | 2026-06-05 | 2862 | 2014-09-01 | 2026-05-29 |  |
| 8462 | TaiwanStockShareholding | MISMATCH | 1773 | 2019-03-14 | 2026-06-05 | 1768 | 2019-03-14 | 2026-05-29 |  |
| 8462 | TaiwanStockDividend | MISMATCH | 13 | 2014-12-22 | 2026-06-27 | 12 | 2014-12-22 | 2025-07-06 |  |
| 8464 | TaiwanStockPriceAdj | MISMATCH | 2780 | 2015-01-07 | 2026-06-05 | 2775 | 2015-01-07 | 2026-05-29 |  |
| 8464 | TaiwanStockShareholding | MISMATCH | 2570 | 2012-03-29 | 2026-06-05 | 2565 | 2012-03-29 | 2026-05-29 |  |
| 8926 | TaiwanStockPriceAdj | MISMATCH | 5622 | 2003-08-04 | 2026-06-05 | 5617 | 2003-08-04 | 2026-05-29 |  |
| 8926 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 8927 | TaiwanStockPriceAdj | MISMATCH | 6231 | 2000-11-03 | 2026-06-05 | 6226 | 2000-11-03 | 2026-05-29 |  |
| 8927 | TaiwanStockShareholding | MISMATCH | 5393 | 2004-08-04 | 2026-06-05 | 5388 | 2004-08-04 | 2026-05-29 |  |
| 8936 | TaiwanStockPriceAdj | MISMATCH | 5776 | 2002-09-10 | 2026-06-05 | 5771 | 2002-09-10 | 2026-05-29 |  |
| 8936 | TaiwanStockShareholding | MISMATCH | 5393 | 2004-08-04 | 2026-06-05 | 5388 | 2004-08-04 | 2026-05-29 |  |
| 8938 | TaiwanStockPriceAdj | MISMATCH | 5717 | 2002-12-03 | 2026-06-05 | 5712 | 2002-12-03 | 2026-05-29 |  |
| 8938 | TaiwanStockShareholding | MISMATCH | 5393 | 2004-08-04 | 2026-06-05 | 5388 | 2004-08-04 | 2026-05-29 |  |
| 8996 | TaiwanStockPriceAdj | MISMATCH | 4828 | 2006-07-03 | 2026-06-05 | 4823 | 2006-07-03 | 2026-05-29 |  |
| 8996 | TaiwanStockShareholding | MISMATCH | 3031 | 2012-03-29 | 2026-06-05 | 3026 | 2012-03-29 | 2026-05-29 |  |
| 8996 | TaiwanStockMonthRevenue | MISMATCH | 274 | 2003-09-01 | 2026-06-01 | 273 | 2003-09-01 | 2026-05-01 |  |
| 9802 | TaiwanStockPriceAdj | MISMATCH | 3328 | 2012-10-19 | 2026-06-05 | 3323 | 2012-10-19 | 2026-05-29 |  |
| 9802 | TaiwanStockShareholding | MISMATCH | 3359 | 2012-03-29 | 2026-06-05 | 3354 | 2012-03-29 | 2026-05-29 |  |
| 9907 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 9907 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9907 | TaiwanStockDividend | MISMATCH | 21 | 2005-07-13 | 2026-06-28 | 20 | 2005-07-13 | 2025-07-26 |  |
| 9904 | TaiwanStockPriceAdj | MISMATCH | 8777 | 1992-01-06 | 2026-06-05 | 8772 | 1992-01-06 | 2026-05-29 |  |
| 9904 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9910 | TaiwanStockPriceAdj | MISMATCH | 8745 | 1992-02-19 | 2026-06-05 | 8740 | 1992-02-19 | 2026-05-29 |  |
| 9910 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9914 | TaiwanStockPriceAdj | MISMATCH | 8564 | 1992-10-01 | 2026-06-05 | 8559 | 1992-10-01 | 2026-05-29 |  |
| 9914 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9921 | TaiwanStockPriceAdj | MISMATCH | 7927 | 1994-12-30 | 2026-06-05 | 7922 | 1994-12-30 | 2026-05-29 |  |
| 9921 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9933 | TaiwanStockPriceAdj | MISMATCH | 6277 | 1999-01-06 | 2026-06-05 | 6272 | 1999-01-06 | 2026-05-29 |  |
| 9933 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9939 | TaiwanStockPriceAdj | MISMATCH | 6220 | 2001-03-05 | 2026-06-05 | 6215 | 2001-03-05 | 2026-05-29 |  |
| 9939 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9938 | TaiwanStockPriceAdj | MISMATCH | 6248 | 2001-01-15 | 2026-06-05 | 6243 | 2001-01-15 | 2026-05-29 |  |
| 9938 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9938 | TaiwanStockMonthRevenue | MISMATCH | 293 | 2002-02-01 | 2026-06-01 | 292 | 2002-02-01 | 2026-05-01 |  |
| 9941 | TaiwanStockPriceAdj | MISMATCH | 6084 | 2001-09-20 | 2026-06-05 | 6079 | 2001-09-20 | 2026-05-29 |  |
| 9941 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9945 | TaiwanStockPriceAdj | MISMATCH | 5777 | 1992-05-01 | 2026-06-05 | 5772 | 1992-05-01 | 2026-05-29 |  |
| 9945 | TaiwanStockShareholding | MISMATCH | 5537 | 2004-02-12 | 2026-06-05 | 5532 | 2004-02-12 | 2026-05-29 |  |
| 9955 | TaiwanStockPriceAdj | MISMATCH | 5446 | 2003-12-31 | 2026-06-05 | 5441 | 2003-12-31 | 2026-05-29 |  |
| 9955 | TaiwanStockShareholding | MISMATCH | 4550 | 2008-01-18 | 2026-06-05 | 4545 | 2008-01-18 | 2026-05-29 |  |
| 9958 | TaiwanStockPriceAdj | MISMATCH | 4763 | 2006-12-13 | 2026-06-05 | 4758 | 2006-12-13 | 2026-05-29 |  |
| 9958 | TaiwanStockShareholding | MISMATCH | 4518 | 2008-03-11 | 2026-06-05 | 4513 | 2008-03-11 | 2026-05-29 |  |
| 9958 | TaiwanStockMonthRevenue | MISMATCH | 281 | 2003-02-01 | 2026-06-01 | 280 | 2003-02-01 | 2026-05-01 |  |

## Targeted Backfill Commands

```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1210 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1210 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1102 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1102 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1215 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1215 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1216 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1216 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1227 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1227 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1229 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1229 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1303 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1303 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1319 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1319 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1402 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1402 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1326 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1326 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1434 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1434 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1476 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1476 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1476 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1476 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1477 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1477 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1477 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1504 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1504 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1504 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1513 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1513 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1514 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1514 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1515 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1515 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1522 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1522 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1533 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1533 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1560 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1560 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1560 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1565 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1565 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1568 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1568 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1582 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1582 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1582 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1590 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1590 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1590 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1597 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1597 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1605 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1605 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1605 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1608 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1608 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1609 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1609 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1609 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1611 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1611 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1611 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1612 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1612 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1612 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1618 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1618 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1702 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1702 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1712 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1712 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1714 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1714 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1714 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1717 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1717 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1722 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1722 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1736 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1736 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1752 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1752 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1773 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1773 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1773 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1784 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1784 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1785 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1785 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1795 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1795 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1789 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1789 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1789 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1806 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1806 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1809 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1809 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1810 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1810 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1815 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1815 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1904 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1904 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1904 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1907 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1907 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2006 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2006 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2006 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2020 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2020 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2027 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2027 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2031 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2031 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2049 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2049 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2049 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2059 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2059 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2059 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2101 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2101 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2101 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2103 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2103 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2103 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2105 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2105 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2106 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2106 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2106 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2108 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2108 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2108 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2201 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2201 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2204 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2204 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2206 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2206 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2233 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2233 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2301 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2301 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2303 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2303 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2303 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2305 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2305 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2312 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2312 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2312 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2308 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2308 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2313 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2313 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2313 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2313 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2316 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2316 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2317 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2317 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2317 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2323 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2323 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2324 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2324 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2327 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2327 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2328 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2328 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2344 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2344 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2344 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2331 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2331 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2345 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2345 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2345 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2347 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2347 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2348 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2348 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2348 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2351 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2351 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2352 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2352 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2352 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2353 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2353 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2353 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2354 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2354 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2354 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2355 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2355 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2356 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2356 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2357 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2357 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2359 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2359 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2363 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2363 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2368 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2368 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2368 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2374 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2374 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2375 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2375 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2376 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2376 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2376 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2377 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2377 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2379 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2379 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2379 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2382 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2382 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2385 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2385 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2385 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2393 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2393 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2393 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2395 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2395 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2395 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2399 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2399 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2402 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2402 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2402 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2404 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2404 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2408 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2408 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2408 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2409 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2409 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2412 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2412 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2413 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2413 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2417 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2417 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2417 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2421 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2421 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2427 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2427 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2428 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2428 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2436 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2436 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2436 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2439 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2439 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2439 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2441 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2441 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2441 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2442 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2442 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2449 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2449 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2451 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2451 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2451 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2453 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2453 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2453 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2454 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2454 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2458 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2458 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2458 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2467 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2467 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2467 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2472 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2472 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2474 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2474 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2481 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2481 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2484 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2484 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2489 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2489 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2492 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2492 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2497 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2497 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2497 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2498 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2498 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2498 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2498 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2504 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2504 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2504 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2515 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2515 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2520 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2520 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2535 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2535 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2542 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2542 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2545 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2545 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2603 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2603 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2603 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2605 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2605 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2606 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2606 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2610 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2610 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2612 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2612 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2615 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2615 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2617 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2617 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2618 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2618 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2634 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2634 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2634 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2636 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2636 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2637 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2637 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2637 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2727 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2727 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2731 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2731 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2732 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2732 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2903 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2903 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2915 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2915 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3003 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3003 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3004 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3004 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3004 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3005 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3005 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3006 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3006 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3006 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3008 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3008 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3008 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3013 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3013 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3014 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3014 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3016 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3016 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3017 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3017 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3019 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3019 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3019 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3019 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3022 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3022 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3022 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3023 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3023 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3023 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3025 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3025 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3025 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3029 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3029 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3029 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3030 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3030 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3030 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3031 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3031 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3033 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3033 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3034 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3034 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3034 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3036 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3036 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3037 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3037 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3038 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3038 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3042 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3042 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3042 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3044 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3044 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3048 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3048 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3048 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3059 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3059 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3071 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3071 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3078 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3078 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3083 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3083 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3088 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3088 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3094 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3094 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3094 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3105 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3105 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3105 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3141 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3141 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3141 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3163 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3163 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3169 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3169 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3169 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3189 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3189 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3189 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3209 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3209 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3211 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3211 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3211 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3217 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3217 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3218 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3218 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3227 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3227 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3227 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3231 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3231 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3257 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3257 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3260 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3260 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3264 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3264 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3264 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3289 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3289 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3290 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3290 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3293 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3293 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3293 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3312 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3312 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3312 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3324 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3324 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3324 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3317 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3317 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3338 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3338 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3356 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3356 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3376 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3376 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3402 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3402 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3406 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3406 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3413 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3413 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3413 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3441 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3441 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3444 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3444 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3450 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3450 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3450 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3455 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3455 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3481 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3481 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3479 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3479 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3483 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3483 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3484 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3484 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3498 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3498 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3515 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3515 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3526 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3526 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3533 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3533 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3540 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3540 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3535 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3535 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3552 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3552 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3563 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3563 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3563 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3580 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3580 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3583 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3583 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3587 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3587 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3594 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3594 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3596 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3596 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3596 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3605 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3605 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3605 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3607 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3607 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3615 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3615 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3622 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3622 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3624 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3624 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3630 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3630 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3630 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3645 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3645 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3653 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3653 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3653 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3665 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3665 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3665 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3673 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3673 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3680 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3680 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3687 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3687 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3689 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3689 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3689 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3693 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3693 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3694 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3694 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3702 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3702 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3703 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3703 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3704 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3704 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3706 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3706 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4105 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4105 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4105 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4114 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4114 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4119 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4119 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4306 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4306 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4551 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4551 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4721 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4721 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4739 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4739 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4739 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4746 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4746 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4746 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4763 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4763 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4912 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4912 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4915 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4915 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4915 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4916 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4916 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4938 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4938 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4938 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4952 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4952 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4958 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4958 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4958 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4966 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4966 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4966 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4968 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4968 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4968 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4973 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4973 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4977 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4977 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4979 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4979 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5009 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5009 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5203 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5203 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5203 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5243 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5243 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5243 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5258 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5258 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5269 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5269 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5285 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5285 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5285 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5289 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5289 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5289 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5347 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5347 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5371 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5371 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5371 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5388 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5388 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5388 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5425 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5425 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5434 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5434 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5439 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5439 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5443 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5443 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5469 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5469 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5457 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5457 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5471 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5471 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5471 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5478 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5478 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5483 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5483 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5483 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5512 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5512 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5607 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5607 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5608 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5608 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5608 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5706 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5706 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5871 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5871 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6104 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6104 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6104 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6118 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6118 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6121 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6121 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6121 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6125 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6125 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6126 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6126 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6138 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6138 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6138 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6139 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6139 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6143 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6143 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6147 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6147 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6150 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6150 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6161 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6161 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6163 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6163 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6166 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6166 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6168 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6168 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6173 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6173 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6170 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6170 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6175 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6175 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6176 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6176 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6187 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6187 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6177 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6177 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6188 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6188 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6188 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6190 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6190 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6191 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6191 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6196 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6196 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6196 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6197 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6197 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6197 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6202 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6202 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6209 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6209 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6207 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6207 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6213 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6213 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6213 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6231 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6231 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6235 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6235 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6235 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6239 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6239 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6239 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6245 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6245 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6257 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6257 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6257 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6265 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6265 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6271 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6271 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6274 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6274 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6274 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6275 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6275 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6279 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6279 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6279 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6278 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6278 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6278 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6284 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6284 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6282 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6282 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6285 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6285 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6290 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6290 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6411 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6411 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6412 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6412 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6412 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6414 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6414 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6414 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6414 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6435 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6435 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6435 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6438 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6438 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6438 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6438 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6477 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6477 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6488 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6488 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6488 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6505 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6505 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6505 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6509 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6509 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6510 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6510 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6510 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6532 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6532 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6532 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6532 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6531 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6531 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6531 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6531 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8011 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8011 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8016 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8016 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8021 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8021 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8021 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8039 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8039 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8039 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8043 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8043 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8046 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8046 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8046 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8064 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8064 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8064 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8069 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8069 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8076 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8076 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8081 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8081 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8091 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8091 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8096 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8096 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8110 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8110 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8111 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8111 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8111 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8112 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8112 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8150 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8150 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8155 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8155 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8163 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8163 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8163 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8183 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8183 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8210 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8210 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8213 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8213 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8255 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8255 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8271 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8271 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8271 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8261 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8261 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8261 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8299 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8299 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8299 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8358 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8358 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8383 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8383 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8436 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8436 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8436 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8462 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8462 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8462 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8464 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8464 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8926 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8926 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8927 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8927 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8936 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8936 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8938 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8938 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8996 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8996 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8996 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9802 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9802 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9907 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9907 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9907 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9904 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9904 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9910 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9910 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9914 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9914 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9921 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9921 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9933 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9933 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9939 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9939 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9938 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9938 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9938 --dataset TaiwanStockMonthRevenue --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9941 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9941 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9945 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9945 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9955 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9955 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9958 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9958 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9958 --dataset TaiwanStockMonthRevenue --strict-source-history
```

## FRED Valid Observation Alignment

| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |
|---|---|---:|---|---|---:|---|---|
| DFF | OK | 26272 | 1954-07-01 | 2026-06-04 | 26272 | 1954-07-01 | 2026-06-04 |
| UNRATE | OK | 940 | 1948-01-01 | 2026-05-01 | 940 | 1948-01-01 | 2026-05-01 |
| T10Y2Y | OK | 12500 | 1976-06-01 | 2026-06-05 | 12500 | 1976-06-01 | 2026-06-05 |
| VIXCLS | OK | 9201 | 1990-01-02 | 2026-06-04 | 9201 | 1990-01-02 | 2026-06-04 |
