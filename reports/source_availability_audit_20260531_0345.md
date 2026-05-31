# Strict source availability audit

- **time**: 2026-05-31 03:45:17
- **constitution**: 系統架構大憲章_v6.1.0.md §14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ
- **tool**: audit_source_availability v0.7
- **start_date**: 1990-01-01
- **drift_tolerance**: 3 day(s) (§6.8.8-C; 0 = strict)
- **scope**: stocks=398 (universe=full), datasets=10
- **special_full_market_reason**: weekly automated audit 3 via launchd cron (§6.8.7 第 (4) 條 / §6.8.8-D)
- **verdict**: **FAILED**
- **summary**: checked=3980, source_empty_ok=1, time_drift_ok=733, mismatch=1130, api_errors=0

- **fred_summary**: checked=4, time_drift_ok=1, mismatch=0, api_errors=0

## Mismatches

| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max | error |
|---|---|---|---:|---|---|---:|---|---|---|
| 1102 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1102 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15192 | 2012-05-02 | 2026-05-29 | 15182 | 2012-05-02 | 2026-05-27 |  |
| 1102 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1210 | TaiwanStockPriceAdj | MISMATCH | 8771 | 1992-01-06 | 2026-05-29 | 8766 | 1992-01-06 | 2026-05-22 |  |
| 1210 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15194 | 2012-05-02 | 2026-05-29 | 15184 | 2012-05-02 | 2026-05-27 |  |
| 1210 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1215 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1215 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15190 | 2012-05-02 | 2026-05-29 | 15180 | 2012-05-02 | 2026-05-27 |  |
| 1215 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1216 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1216 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15193 | 2012-05-02 | 2026-05-29 | 15183 | 2012-05-02 | 2026-05-27 |  |
| 1216 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1227 | TaiwanStockPriceAdj | MISMATCH | 8127 | 1994-04-11 | 2026-05-29 | 8122 | 1994-04-11 | 2026-05-22 |  |
| 1227 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15184 | 2012-05-02 | 2026-05-29 | 15174 | 2012-05-02 | 2026-05-27 |  |
| 1227 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1229 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1229 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15174 | 2012-05-02 | 2026-05-29 | 15164 | 2012-05-02 | 2026-05-27 |  |
| 1229 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1303 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1303 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15194 | 2012-05-02 | 2026-05-29 | 15184 | 2012-05-02 | 2026-05-27 |  |
| 1303 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1319 | TaiwanStockPriceAdj | MISMATCH | 7936 | 1994-12-13 | 2026-05-29 | 7931 | 1994-12-13 | 2026-05-22 |  |
| 1319 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 1319 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1326 | TaiwanStockPriceAdj | MISMATCH | 5877 | 1992-01-06 | 2026-05-29 | 5872 | 1992-01-06 | 2026-05-22 |  |
| 1326 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 1326 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1402 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1402 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 1402 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1434 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1434 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15188 | 2012-05-02 | 2026-05-29 | 15178 | 2012-05-02 | 2026-05-27 |  |
| 1434 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1476 | TaiwanStockPriceAdj | MISMATCH | 6181 | 2001-04-19 | 2026-05-29 | 6176 | 2001-04-19 | 2026-05-22 |  |
| 1476 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15192 | 2012-05-02 | 2026-05-29 | 15182 | 2012-05-02 | 2026-05-27 |  |
| 1476 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1477 | TaiwanStockPriceAdj | MISMATCH | 5746 | 2003-01-22 | 2026-05-29 | 5741 | 2003-01-22 | 2026-05-22 |  |
| 1477 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 1477 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1504 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1504 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1504 | TaiwanStockDividend | MISMATCH | 21 | 2006-07-10 | 2026-06-23 | 20 | 2006-07-10 | 2025-06-25 |  |
| 1513 | TaiwanStockPriceAdj | MISMATCH | 8121 | 1994-03-09 | 2026-05-29 | 8116 | 1994-03-09 | 2026-05-22 |  |
| 1513 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1513 | TaiwanStockDividend | MISMATCH | 22 | 2005-08-29 | 2026-07-15 | 21 | 2005-08-29 | 2025-09-05 |  |
| 1514 | TaiwanStockPriceAdj | MISMATCH | 8136 | 1994-03-28 | 2026-05-29 | 8131 | 1994-03-28 | 2026-05-22 |  |
| 1514 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1515 | TaiwanStockPriceAdj | MISMATCH | 7890 | 1995-02-06 | 2026-05-29 | 7885 | 1995-02-06 | 2026-05-22 |  |
| 1515 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14832 | 2012-05-02 | 2026-05-29 | 14822 | 2012-05-02 | 2026-05-27 |  |
| 1515 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1522 | TaiwanStockPriceAdj | MISMATCH | 7126 | 1997-10-07 | 2026-05-29 | 7121 | 1997-10-07 | 2026-05-22 |  |
| 1522 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15156 | 2012-05-02 | 2026-05-29 | 15146 | 2012-05-02 | 2026-05-27 |  |
| 1522 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1532 | TaiwanStockPriceAdj | MISMATCH | 6337 | 2000-09-13 | 2026-05-29 | 6332 | 2000-09-13 | 2026-05-22 |  |
| 1532 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15186 | 2012-05-02 | 2026-05-29 | 15176 | 2012-05-02 | 2026-05-27 |  |
| 1532 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1533 | TaiwanStockPriceAdj | MISMATCH | 6191 | 2001-04-09 | 2026-05-29 | 6186 | 2001-04-09 | 2026-05-22 |  |
| 1533 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14667 | 2012-05-02 | 2026-05-29 | 14657 | 2012-05-02 | 2026-05-27 |  |
| 1533 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1560 | TaiwanStockPriceAdj | MISMATCH | 5240 | 2005-02-01 | 2026-05-29 | 5235 | 2005-02-01 | 2026-05-22 |  |
| 1560 | TaiwanStockShareholding | MISMATCH | 5286 | 2005-01-28 | 2026-05-29 | 5281 | 2005-01-28 | 2026-05-22 |  |
| 1565 | TaiwanStockPriceAdj | MISMATCH | 5385 | 2004-03-31 | 2026-05-29 | 5380 | 2004-03-31 | 2026-05-22 |  |
| 1565 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20314 | 2005-01-03 | 2026-05-29 | 20304 | 2005-01-03 | 2026-05-27 |  |
| 1565 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 1568 | TaiwanStockPriceAdj | MISMATCH | 4754 | 2006-12-19 | 2026-05-29 | 4749 | 2006-12-19 | 2026-05-22 |  |
| 1568 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13557 | 2007-07-11 | 2026-05-29 | 13547 | 2007-07-11 | 2026-05-27 |  |
| 1568 | TaiwanStockShareholding | MISMATCH | 2966 | 2012-03-29 | 2026-05-29 | 2961 | 2012-03-29 | 2026-05-22 |  |
| 1582 | TaiwanStockPriceAdj | MISMATCH | 4698 | 2006-12-19 | 2026-05-29 | 4693 | 2006-12-19 | 2026-05-22 |  |
| 1582 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16952 | 2007-01-11 | 2026-05-29 | 16942 | 2007-01-11 | 2026-05-27 |  |
| 1582 | TaiwanStockShareholding | MISMATCH | 4065 | 2009-12-16 | 2026-05-29 | 4060 | 2009-12-16 | 2026-05-22 |  |
| 1590 | TaiwanStockPriceAdj | MISMATCH | 3782 | 2010-12-14 | 2026-05-29 | 3777 | 2010-12-14 | 2026-05-22 |  |
| 1590 | TaiwanStockShareholding | MISMATCH | 3816 | 2010-12-10 | 2026-05-29 | 3811 | 2010-12-10 | 2026-05-22 |  |
| 1597 | TaiwanStockPriceAdj | MISMATCH | 3723 | 2011-03-11 | 2026-05-29 | 3718 | 2011-03-11 | 2026-05-22 |  |
| 1597 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13579 | 2011-06-17 | 2026-05-29 | 13569 | 2011-06-17 | 2026-05-27 |  |
| 1597 | TaiwanStockShareholding | MISMATCH | 3289 | 2012-12-27 | 2026-05-29 | 3284 | 2012-12-27 | 2026-05-22 |  |
| 1605 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1605 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 1605 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1605 | TaiwanStockDividend | MISMATCH | 15 | 2005-07-19 | 2026-06-19 | 14 | 2005-07-19 | 2025-06-10 |  |
| 1608 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1608 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15132 | 2012-05-02 | 2026-05-29 | 15122 | 2012-05-02 | 2026-05-27 |  |
| 1608 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1609 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1609 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15132 | 2012-05-02 | 2026-05-29 | 15122 | 2012-05-02 | 2026-05-27 |  |
| 1609 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1611 | TaiwanStockPriceAdj | MISMATCH | 8759 | 1992-01-06 | 2026-05-29 | 8754 | 1992-01-06 | 2026-05-22 |  |
| 1611 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15058 | 2012-05-02 | 2026-05-29 | 15048 | 2012-05-02 | 2026-05-27 |  |
| 1611 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1612 | TaiwanStockPriceAdj | MISMATCH | 8391 | 1993-05-10 | 2026-05-29 | 8386 | 1993-05-10 | 2026-05-22 |  |
| 1612 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14784 | 2012-05-02 | 2026-05-29 | 14774 | 2012-05-02 | 2026-05-27 |  |
| 1612 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1618 | TaiwanStockPriceAdj | MISMATCH | 6331 | 2000-09-13 | 2026-05-29 | 6326 | 2000-09-13 | 2026-05-22 |  |
| 1618 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14385 | 2012-05-02 | 2026-05-29 | 14375 | 2012-05-02 | 2026-05-27 |  |
| 1618 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1702 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1702 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15187 | 2012-05-02 | 2026-05-29 | 15177 | 2012-05-02 | 2026-05-27 |  |
| 1702 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1712 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1712 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15153 | 2012-05-02 | 2026-05-29 | 15143 | 2012-05-02 | 2026-05-27 |  |
| 1712 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1714 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1714 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15184 | 2012-05-02 | 2026-05-29 | 15174 | 2012-05-02 | 2026-05-27 |  |
| 1714 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1717 | TaiwanStockPriceAdj | MISMATCH | 8127 | 1994-03-31 | 2026-05-29 | 8122 | 1994-03-31 | 2026-05-22 |  |
| 1717 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15142 | 2012-05-02 | 2026-05-29 | 15132 | 2012-05-02 | 2026-05-27 |  |
| 1717 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1717 | TaiwanStockDividend | MISMATCH | 28 | 2005-08-05 | 2026-07-05 | 27 | 2005-08-05 | 2026-05-02 |  |
| 1722 | TaiwanStockPriceAdj | MISMATCH | 7003 | 1998-03-25 | 2026-05-29 | 6998 | 1998-03-25 | 2026-05-22 |  |
| 1722 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15193 | 2012-05-02 | 2026-05-29 | 15183 | 2012-05-02 | 2026-05-27 |  |
| 1722 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1736 | TaiwanStockPriceAdj | MISMATCH | 5754 | 2003-01-10 | 2026-05-29 | 5749 | 2003-01-10 | 2026-05-22 |  |
| 1736 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15186 | 2012-05-02 | 2026-05-29 | 15176 | 2012-05-02 | 2026-05-27 |  |
| 1736 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1752 | TaiwanStockPriceAdj | MISMATCH | 4748 | 2006-12-13 | 2026-05-29 | 4743 | 2006-12-13 | 2026-05-22 |  |
| 1752 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15356 | 2008-05-26 | 2026-05-29 | 15346 | 2008-05-26 | 2026-05-27 |  |
| 1752 | TaiwanStockShareholding | MISMATCH | 4157 | 2009-08-10 | 2026-05-29 | 4152 | 2009-08-10 | 2026-05-22 |  |
| 1773 | TaiwanStockPriceAdj | MISMATCH | 4498 | 2008-01-28 | 2026-05-29 | 4493 | 2008-01-28 | 2026-05-22 |  |
| 1773 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15166 | 2008-03-11 | 2026-05-29 | 15156 | 2008-03-11 | 2026-05-27 |  |
| 1773 | TaiwanStockShareholding | MISMATCH | 4271 | 2009-02-26 | 2026-05-29 | 4266 | 2009-02-26 | 2026-05-22 |  |
| 1773 | TaiwanStockDividend | MISMATCH | 19 | 2008-07-21 | 2026-06-25 | 18 | 2008-07-21 | 2025-06-21 |  |
| 1784 | TaiwanStockPriceAdj | MISMATCH | 4758 | 2006-12-19 | 2026-05-29 | 4753 | 2006-12-19 | 2026-05-22 |  |
| 1784 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15973 | 2007-07-05 | 2026-05-29 | 15963 | 2007-07-05 | 2026-05-27 |  |
| 1784 | TaiwanStockShareholding | MISMATCH | 4666 | 2007-07-25 | 2026-05-29 | 4661 | 2007-07-25 | 2026-05-22 |  |
| 1785 | TaiwanStockPriceAdj | MISMATCH | 5003 | 2005-02-01 | 2026-05-29 | 4998 | 2005-02-01 | 2026-05-22 |  |
| 1785 | TaiwanStockShareholding | MISMATCH | 5269 | 2005-01-28 | 2026-05-29 | 5264 | 2005-01-28 | 2026-05-22 |  |
| 1789 | TaiwanStockPriceAdj | MISMATCH | 3813 | 2010-11-01 | 2026-05-29 | 3808 | 2010-11-01 | 2026-05-22 |  |
| 1789 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15565 | 2010-10-29 | 2026-05-29 | 15555 | 2010-10-29 | 2026-05-27 |  |
| 1789 | TaiwanStockShareholding | MISMATCH | 3618 | 2011-09-28 | 2026-05-29 | 3613 | 2011-09-28 | 2026-05-22 |  |
| 1795 | TaiwanStockPriceAdj | MISMATCH | 4754 | 2006-12-13 | 2026-05-29 | 4749 | 2006-12-13 | 2026-05-22 |  |
| 1795 | TaiwanStockShareholding | MISMATCH | 4025 | 2010-01-28 | 2026-05-29 | 4020 | 2010-01-28 | 2026-05-22 |  |
| 1806 | TaiwanStockPriceAdj | MISMATCH | 8523 | 1992-09-30 | 2026-05-29 | 8518 | 1992-09-30 | 2026-05-22 |  |
| 1806 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14956 | 2012-05-02 | 2026-05-29 | 14946 | 2012-05-02 | 2026-05-27 |  |
| 1806 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1809 | TaiwanStockPriceAdj | MISMATCH | 7535 | 1996-05-01 | 2026-05-29 | 7530 | 1996-05-01 | 2026-05-22 |  |
| 1809 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14591 | 2012-05-02 | 2026-05-29 | 14581 | 2012-05-02 | 2026-05-27 |  |
| 1809 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1810 | TaiwanStockPriceAdj | MISMATCH | 7445 | 1992-01-06 | 2026-05-29 | 7440 | 1992-01-06 | 2026-05-22 |  |
| 1810 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14728 | 2012-05-02 | 2026-05-29 | 14718 | 2012-05-02 | 2026-05-27 |  |
| 1810 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1810 | TaiwanStockDividend | MISMATCH | 12 | 2012-07-29 | 2026-06-20 | 11 | 2012-07-29 | 2025-07-08 |  |
| 1815 | TaiwanStockPriceAdj | MISMATCH | 4929 | 2006-01-24 | 2026-05-29 | 4924 | 2006-01-24 | 2026-05-22 |  |
| 1815 | TaiwanStockShareholding | MISMATCH | 5034 | 2006-01-20 | 2026-05-29 | 5029 | 2006-01-20 | 2026-05-22 |  |
| 1904 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1904 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15186 | 2012-05-02 | 2026-05-29 | 15176 | 2012-05-02 | 2026-05-27 |  |
| 1904 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 1907 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 1907 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15185 | 2012-05-02 | 2026-05-29 | 15175 | 2012-05-02 | 2026-05-27 |  |
| 1907 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2006 | TaiwanStockPriceAdj | MISMATCH | 8765 | 1992-01-06 | 2026-05-29 | 8760 | 1992-01-06 | 2026-05-22 |  |
| 2006 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15150 | 2012-05-02 | 2026-05-29 | 15140 | 2012-05-02 | 2026-05-27 |  |
| 2006 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2020 | TaiwanStockPriceAdj | MISMATCH | 8399 | 1993-04-28 | 2026-05-29 | 8394 | 1993-04-28 | 2026-05-22 |  |
| 2020 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14835 | 2012-05-02 | 2026-05-29 | 14825 | 2012-05-02 | 2026-05-27 |  |
| 2020 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2027 | TaiwanStockPriceAdj | MISMATCH | 7399 | 1996-10-28 | 2026-05-29 | 7394 | 1996-10-28 | 2026-05-22 |  |
| 2027 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15175 | 2012-05-02 | 2026-05-29 | 15165 | 2012-05-02 | 2026-05-27 |  |
| 2027 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2031 | TaiwanStockPriceAdj | MISMATCH | 6337 | 2000-09-13 | 2026-05-29 | 6332 | 2000-09-13 | 2026-05-22 |  |
| 2031 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15161 | 2012-05-02 | 2026-05-29 | 15151 | 2012-05-02 | 2026-05-27 |  |
| 2031 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2049 | TaiwanStockPriceAdj | MISMATCH | 4650 | 2007-06-21 | 2026-05-29 | 4645 | 2007-06-21 | 2026-05-22 |  |
| 2049 | TaiwanStockShareholding | MISMATCH | 4188 | 2009-06-25 | 2026-05-29 | 4183 | 2009-06-25 | 2026-05-22 |  |
| 2059 | TaiwanStockPriceAdj | MISMATCH | 5114 | 2005-04-29 | 2026-05-29 | 5109 | 2005-04-29 | 2026-05-22 |  |
| 2059 | TaiwanStockShareholding | MISMATCH | 4440 | 2008-06-24 | 2026-05-29 | 4435 | 2008-06-24 | 2026-05-22 |  |
| 2101 | TaiwanStockPriceAdj | MISMATCH | 8610 | 1992-01-06 | 2026-05-29 | 8605 | 1992-01-06 | 2026-05-22 |  |
| 2101 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15153 | 2012-05-02 | 2026-05-29 | 15143 | 2012-05-02 | 2026-05-27 |  |
| 2101 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2103 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2103 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15193 | 2012-05-02 | 2026-05-29 | 15183 | 2012-05-02 | 2026-05-27 |  |
| 2103 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2105 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2105 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2105 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2106 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2106 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15180 | 2012-05-02 | 2026-05-29 | 15170 | 2012-05-02 | 2026-05-27 |  |
| 2106 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2108 | TaiwanStockPriceAdj | MISMATCH | 8538 | 1992-10-28 | 2026-05-29 | 8533 | 1992-10-28 | 2026-05-22 |  |
| 2108 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15158 | 2012-05-02 | 2026-05-29 | 15148 | 2012-05-02 | 2026-05-27 |  |
| 2108 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2201 | TaiwanStockPriceAdj | MISMATCH | 8765 | 1992-01-06 | 2026-05-29 | 8760 | 1992-01-06 | 2026-05-22 |  |
| 2201 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15153 | 2012-05-02 | 2026-05-29 | 15143 | 2012-05-02 | 2026-05-27 |  |
| 2201 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2204 | TaiwanStockPriceAdj | MISMATCH | 8766 | 1992-01-06 | 2026-05-29 | 8761 | 1992-01-06 | 2026-05-22 |  |
| 2204 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15154 | 2012-05-02 | 2026-05-29 | 15144 | 2012-05-02 | 2026-05-27 |  |
| 2204 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2206 | TaiwanStockPriceAdj | MISMATCH | 7469 | 1996-07-30 | 2026-05-29 | 7464 | 1996-07-30 | 2026-05-22 |  |
| 2206 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15179 | 2012-05-02 | 2026-05-29 | 15169 | 2012-05-02 | 2026-05-27 |  |
| 2206 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2233 | TaiwanStockPriceAdj | MISMATCH | 3817 | 2010-10-26 | 2026-05-29 | 3812 | 2010-10-26 | 2026-05-22 |  |
| 2233 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15511 | 2010-10-25 | 2026-05-29 | 15501 | 2010-10-25 | 2026-05-27 |  |
| 2233 | TaiwanStockShareholding | MISMATCH | 3581 | 2011-11-11 | 2026-05-29 | 3576 | 2011-11-11 | 2026-05-22 |  |
| 2301 | TaiwanStockPriceAdj | MISMATCH | 8741 | 1992-01-06 | 2026-05-29 | 8736 | 1992-01-06 | 2026-05-22 |  |
| 2301 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2303 | TaiwanStockPriceAdj | MISMATCH | 8761 | 1992-01-06 | 2026-05-29 | 8756 | 1992-01-06 | 2026-05-22 |  |
| 2303 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2305 | TaiwanStockPriceAdj | MISMATCH | 8749 | 1992-01-04 | 2026-05-29 | 8744 | 1992-01-04 | 2026-05-22 |  |
| 2305 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14456 | 2012-05-02 | 2026-05-29 | 14446 | 2012-05-02 | 2026-05-27 |  |
| 2305 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2308 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2308 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2308 | TaiwanStockDividend | MISMATCH | 23 | 2005-06-30 | 2026-06-23 | 22 | 2005-06-30 | 2025-06-23 |  |
| 2312 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2312 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15188 | 2012-05-02 | 2026-05-29 | 15178 | 2012-05-02 | 2026-05-27 |  |
| 2312 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2313 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2313 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2316 | TaiwanStockPriceAdj | MISMATCH | 8754 | 1992-01-06 | 2026-05-29 | 8749 | 1992-01-06 | 2026-05-22 |  |
| 2316 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2317 | TaiwanStockPriceAdj | MISMATCH | 8766 | 1992-01-06 | 2026-05-29 | 8761 | 1992-01-06 | 2026-05-22 |  |
| 2317 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2323 | TaiwanStockPriceAdj | MISMATCH | 8718 | 1992-02-18 | 2026-05-29 | 8713 | 1992-02-18 | 2026-05-22 |  |
| 2323 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15088 | 2012-05-02 | 2026-05-29 | 15078 | 2012-05-02 | 2026-05-27 |  |
| 2323 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2324 | TaiwanStockPriceAdj | MISMATCH | 8740 | 1992-02-19 | 2026-05-29 | 8735 | 1992-02-19 | 2026-05-22 |  |
| 2324 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15192 | 2012-05-02 | 2026-05-29 | 15182 | 2012-05-02 | 2026-05-27 |  |
| 2324 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2327 | TaiwanStockPriceAdj | MISMATCH | 8204 | 1993-10-23 | 2026-05-29 | 8199 | 1993-10-23 | 2026-05-22 |  |
| 2327 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2327 | TaiwanStockDividend | MISMATCH | 19 | 2007-07-22 | 2026-06-20 | 18 | 2007-07-22 | 2025-06-21 |  |
| 2328 | TaiwanStockPriceAdj | MISMATCH | 8240 | 1993-11-10 | 2026-05-29 | 8235 | 1993-11-10 | 2026-05-22 |  |
| 2328 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15189 | 2012-05-02 | 2026-05-29 | 15179 | 2012-05-02 | 2026-05-27 |  |
| 2328 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2330 | TaiwanStockPriceAdj | MISMATCH | 8005 | 1994-09-14 | 2026-05-29 | 8000 | 1994-09-14 | 2026-05-22 |  |
| 2330 | TaiwanStockShareholding | MISMATCH | 5529 | 2004-02-12 | 2026-05-29 | 5524 | 2004-02-12 | 2026-05-22 |  |
| 2330 | TaiwanStockDividend | MISMATCH | 43 | 2005-06-19 | 2026-06-17 | 42 | 2005-06-19 | 2026-03-23 |  |
| 2331 | TaiwanStockPriceAdj | MISMATCH | 7983 | 1994-09-22 | 2026-05-29 | 7978 | 1994-09-22 | 2026-05-22 |  |
| 2331 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15142 | 2012-05-02 | 2026-05-29 | 15132 | 2012-05-02 | 2026-05-27 |  |
| 2331 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2344 | TaiwanStockPriceAdj | MISMATCH | 7692 | 1995-10-19 | 2026-05-29 | 7687 | 1995-10-19 | 2026-05-22 |  |
| 2344 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2347 | TaiwanStockPriceAdj | MISMATCH | 7648 | 1995-12-14 | 2026-05-29 | 7643 | 1995-12-14 | 2026-05-22 |  |
| 2347 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2347 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2345 | TaiwanStockPriceAdj | MISMATCH | 7671 | 1995-11-16 | 2026-05-29 | 7666 | 1995-11-16 | 2026-05-22 |  |
| 2345 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2348 | TaiwanStockPriceAdj | MISMATCH | 7572 | 1996-01-06 | 2026-05-29 | 7567 | 1996-01-06 | 2026-05-22 |  |
| 2348 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10387 | 2014-12-22 | 2026-05-29 | 10377 | 2014-12-22 | 2026-05-27 |  |
| 2348 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2351 | TaiwanStockPriceAdj | MISMATCH | 7548 | 1996-04-26 | 2026-05-29 | 7543 | 1996-04-26 | 2026-05-22 |  |
| 2351 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15189 | 2012-05-02 | 2026-05-29 | 15179 | 2012-05-02 | 2026-05-27 |  |
| 2351 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2352 | TaiwanStockPriceAdj | MISMATCH | 7462 | 1996-07-23 | 2026-05-29 | 7457 | 1996-07-23 | 2026-05-22 |  |
| 2352 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15167 | 2012-05-02 | 2026-05-29 | 15157 | 2012-05-02 | 2026-05-27 |  |
| 2352 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2353 | TaiwanStockPriceAdj | MISMATCH | 7427 | 1996-09-19 | 2026-05-29 | 7422 | 1996-09-19 | 2026-05-22 |  |
| 2353 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2353 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2354 | TaiwanStockPriceAdj | MISMATCH | 7412 | 1996-10-09 | 2026-05-29 | 7407 | 1996-10-09 | 2026-05-22 |  |
| 2354 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2355 | TaiwanStockPriceAdj | MISMATCH | 7408 | 1996-10-15 | 2026-05-29 | 7403 | 1996-10-15 | 2026-05-22 |  |
| 2355 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2355 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2356 | TaiwanStockPriceAdj | MISMATCH | 7386 | 1996-11-14 | 2026-05-29 | 7381 | 1996-11-14 | 2026-05-22 |  |
| 2356 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2357 | TaiwanStockPriceAdj | MISMATCH | 7359 | 1996-11-15 | 2026-05-29 | 7354 | 1996-11-15 | 2026-05-22 |  |
| 2357 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2363 | TaiwanStockPriceAdj | MISMATCH | 7151 | 1997-08-02 | 2026-05-29 | 7146 | 1997-08-02 | 2026-05-22 |  |
| 2363 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2359 | TaiwanStockPriceAdj | MISMATCH | 7320 | 1996-12-20 | 2026-05-29 | 7315 | 1996-12-20 | 2026-05-22 |  |
| 2359 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14843 | 2012-05-02 | 2026-05-29 | 14833 | 2012-05-02 | 2026-05-27 |  |
| 2359 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2368 | TaiwanStockPriceAdj | MISMATCH | 7008 | 1998-03-10 | 2026-05-29 | 7003 | 1998-03-10 | 2026-05-22 |  |
| 2368 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2374 | TaiwanStockPriceAdj | MISMATCH | 6780 | 1999-01-06 | 2026-05-29 | 6775 | 1999-01-06 | 2026-05-22 |  |
| 2374 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2375 | TaiwanStockPriceAdj | MISMATCH | 6828 | 1998-08-31 | 2026-05-29 | 6823 | 1998-08-31 | 2026-05-22 |  |
| 2375 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14548 | 2012-05-02 | 2026-05-29 | 14538 | 2012-05-02 | 2026-05-27 |  |
| 2375 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2376 | TaiwanStockPriceAdj | MISMATCH | 6861 | 1998-09-25 | 2026-05-29 | 6856 | 1998-09-25 | 2026-05-22 |  |
| 2376 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2377 | TaiwanStockPriceAdj | MISMATCH | 6834 | 1998-11-02 | 2026-05-29 | 6829 | 1998-11-02 | 2026-05-22 |  |
| 2377 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2377 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2379 | TaiwanStockPriceAdj | MISMATCH | 6831 | 1998-10-27 | 2026-05-29 | 6826 | 1998-10-27 | 2026-05-22 |  |
| 2379 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2382 | TaiwanStockPriceAdj | MISMATCH | 6785 | 1999-01-11 | 2026-05-29 | 6780 | 1999-01-11 | 2026-05-22 |  |
| 2382 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2385 | TaiwanStockPriceAdj | MISMATCH | 6788 | 1999-01-06 | 2026-05-29 | 6783 | 1999-01-06 | 2026-05-22 |  |
| 2385 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2385 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2393 | TaiwanStockPriceAdj | MISMATCH | 6564 | 1999-11-05 | 2026-05-29 | 6559 | 1999-11-05 | 2026-05-22 |  |
| 2393 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2393 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2395 | TaiwanStockPriceAdj | MISMATCH | 6535 | 1999-12-14 | 2026-05-29 | 6530 | 1999-12-14 | 2026-05-22 |  |
| 2395 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2399 | TaiwanStockPriceAdj | MISMATCH | 6532 | 1999-12-17 | 2026-05-29 | 6527 | 1999-12-17 | 2026-05-22 |  |
| 2399 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14989 | 2012-05-02 | 2026-05-29 | 14979 | 2012-05-02 | 2026-05-27 |  |
| 2399 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2402 | TaiwanStockPriceAdj | MISMATCH | 6514 | 2000-01-15 | 2026-05-29 | 6509 | 2000-01-15 | 2026-05-22 |  |
| 2402 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2404 | TaiwanStockPriceAdj | MISMATCH | 6467 | 2000-03-15 | 2026-05-29 | 6462 | 2000-03-15 | 2026-05-22 |  |
| 2404 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2408 | TaiwanStockPriceAdj | MISMATCH | 6338 | 2000-08-18 | 2026-05-29 | 6333 | 2000-08-18 | 2026-05-22 |  |
| 2408 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2409 | TaiwanStockPriceAdj | MISMATCH | 6331 | 2000-09-11 | 2026-05-29 | 6326 | 2000-09-11 | 2026-05-22 |  |
| 2409 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-29 | 15152 | 2012-05-02 | 2026-05-27 |  |
| 2409 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2412 | TaiwanStockPriceAdj | MISMATCH | 6251 | 2000-10-30 | 2026-05-29 | 6246 | 2000-10-30 | 2026-05-22 |  |
| 2412 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2413 | TaiwanStockPriceAdj | MISMATCH | 6337 | 2000-09-13 | 2026-05-29 | 6332 | 2000-09-13 | 2026-05-22 |  |
| 2413 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14850 | 2012-05-02 | 2026-05-29 | 14840 | 2012-05-02 | 2026-05-27 |  |
| 2413 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2417 | TaiwanStockPriceAdj | MISMATCH | 6330 | 2000-09-13 | 2026-05-29 | 6325 | 2000-09-13 | 2026-05-22 |  |
| 2417 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15067 | 2012-05-02 | 2026-05-29 | 15057 | 2012-05-02 | 2026-05-27 |  |
| 2417 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2421 | TaiwanStockPriceAdj | MISMATCH | 6337 | 2000-09-13 | 2026-05-29 | 6332 | 2000-09-13 | 2026-05-22 |  |
| 2421 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2421 | TaiwanStockDividend | MISMATCH | 23 | 2005-10-01 | 2026-06-16 | 22 | 2005-10-01 | 2026-03-02 |  |
| 2427 | TaiwanStockPriceAdj | MISMATCH | 6337 | 2000-09-13 | 2026-05-29 | 6332 | 2000-09-13 | 2026-05-22 |  |
| 2427 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14673 | 2012-05-02 | 2026-05-29 | 14663 | 2012-05-02 | 2026-05-27 |  |
| 2427 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2428 | TaiwanStockPriceAdj | MISMATCH | 6337 | 2000-09-13 | 2026-05-29 | 6332 | 2000-09-13 | 2026-05-22 |  |
| 2428 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15184 | 2012-05-02 | 2026-05-29 | 15174 | 2012-05-02 | 2026-05-27 |  |
| 2428 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2436 | TaiwanStockPriceAdj | MISMATCH | 6322 | 2000-09-13 | 2026-05-29 | 6317 | 2000-09-13 | 2026-05-22 |  |
| 2436 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15103 | 2012-05-02 | 2026-05-29 | 15093 | 2012-05-02 | 2026-05-27 |  |
| 2436 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2439 | TaiwanStockPriceAdj | MISMATCH | 6337 | 2000-09-13 | 2026-05-29 | 6332 | 2000-09-13 | 2026-05-22 |  |
| 2439 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15192 | 2012-05-02 | 2026-05-29 | 15182 | 2012-05-02 | 2026-05-27 |  |
| 2439 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2441 | TaiwanStockPriceAdj | MISMATCH | 6302 | 2000-10-27 | 2026-05-29 | 6297 | 2000-10-27 | 2026-05-22 |  |
| 2441 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2442 | TaiwanStockPriceAdj | MISMATCH | 6249 | 2000-11-23 | 2026-05-29 | 6244 | 2000-11-23 | 2026-05-22 |  |
| 2442 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14766 | 2012-05-02 | 2026-05-29 | 14756 | 2012-05-02 | 2026-05-27 |  |
| 2442 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2449 | TaiwanStockPriceAdj | MISMATCH | 6169 | 2001-05-10 | 2026-05-29 | 6164 | 2001-05-10 | 2026-05-22 |  |
| 2449 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2451 | TaiwanStockPriceAdj | MISMATCH | 6173 | 2001-05-04 | 2026-05-29 | 6168 | 2001-05-04 | 2026-05-22 |  |
| 2451 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2453 | TaiwanStockPriceAdj | MISMATCH | 6160 | 2001-05-23 | 2026-05-29 | 6155 | 2001-05-23 | 2026-05-22 |  |
| 2453 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14409 | 2012-05-02 | 2026-05-29 | 14399 | 2012-05-02 | 2026-05-27 |  |
| 2453 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2454 | TaiwanStockPriceAdj | MISMATCH | 6117 | 2001-07-24 | 2026-05-29 | 6112 | 2001-07-24 | 2026-05-22 |  |
| 2454 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2458 | TaiwanStockPriceAdj | MISMATCH | 6072 | 2001-09-20 | 2026-05-29 | 6067 | 2001-09-20 | 2026-05-22 |  |
| 2458 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-29 | 15152 | 2012-05-02 | 2026-05-27 |  |
| 2458 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2467 | TaiwanStockPriceAdj | MISMATCH | 6079 | 2001-09-20 | 2026-05-29 | 6074 | 2001-09-20 | 2026-05-22 |  |
| 2467 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2472 | TaiwanStockPriceAdj | MISMATCH | 6069 | 2001-09-20 | 2026-05-29 | 6064 | 2001-09-20 | 2026-05-22 |  |
| 2472 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15105 | 2012-05-02 | 2026-05-29 | 15095 | 2012-05-02 | 2026-05-27 |  |
| 2472 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2474 | TaiwanStockPriceAdj | MISMATCH | 6079 | 2001-09-20 | 2026-05-29 | 6074 | 2001-09-20 | 2026-05-22 |  |
| 2474 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15192 | 2012-05-02 | 2026-05-29 | 15182 | 2012-05-02 | 2026-05-27 |  |
| 2474 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2481 | TaiwanStockPriceAdj | MISMATCH | 6074 | 2001-09-20 | 2026-05-29 | 6069 | 2001-09-20 | 2026-05-22 |  |
| 2481 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2484 | TaiwanStockPriceAdj | MISMATCH | 6079 | 2001-09-20 | 2026-05-29 | 6074 | 2001-09-20 | 2026-05-22 |  |
| 2484 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15143 | 2012-05-02 | 2026-05-29 | 15133 | 2012-05-02 | 2026-05-27 |  |
| 2484 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2489 | TaiwanStockPriceAdj | MISMATCH | 6053 | 2001-09-20 | 2026-05-29 | 6048 | 2001-09-20 | 2026-05-22 |  |
| 2489 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15055 | 2012-05-02 | 2026-05-29 | 15045 | 2012-05-02 | 2026-05-27 |  |
| 2489 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2492 | TaiwanStockPriceAdj | MISMATCH | 6057 | 2001-09-20 | 2026-05-29 | 6052 | 2001-09-20 | 2026-05-22 |  |
| 2492 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2497 | TaiwanStockPriceAdj | MISMATCH | 5984 | 2002-02-05 | 2026-05-29 | 5979 | 2002-02-05 | 2026-05-22 |  |
| 2497 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14914 | 2012-05-02 | 2026-05-29 | 14904 | 2012-05-02 | 2026-05-27 |  |
| 2497 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2498 | TaiwanStockPriceAdj | MISMATCH | 5956 | 2002-03-27 | 2026-05-29 | 5951 | 2002-03-27 | 2026-05-22 |  |
| 2498 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15193 | 2012-05-02 | 2026-05-29 | 15183 | 2012-05-02 | 2026-05-27 |  |
| 2498 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2504 | TaiwanStockPriceAdj | MISMATCH | 8766 | 1992-01-06 | 2026-05-29 | 8761 | 1992-01-06 | 2026-05-22 |  |
| 2504 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15143 | 2012-05-02 | 2026-05-29 | 15133 | 2012-05-02 | 2026-05-27 |  |
| 2504 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2515 | TaiwanStockPriceAdj | MISMATCH | 8446 | 1993-03-03 | 2026-05-29 | 8441 | 1993-03-03 | 2026-05-22 |  |
| 2515 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15193 | 2012-05-02 | 2026-05-29 | 15183 | 2012-05-02 | 2026-05-27 |  |
| 2515 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2520 | TaiwanStockPriceAdj | MISMATCH | 8250 | 1993-10-28 | 2026-05-29 | 8245 | 1993-10-28 | 2026-05-22 |  |
| 2520 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2520 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2535 | TaiwanStockPriceAdj | MISMATCH | 7575 | 1996-03-12 | 2026-05-29 | 7570 | 1996-03-12 | 2026-05-22 |  |
| 2535 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15039 | 2012-05-02 | 2026-05-29 | 15029 | 2012-05-02 | 2026-05-27 |  |
| 2535 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2542 | TaiwanStockPriceAdj | MISMATCH | 6695 | 1999-05-04 | 2026-05-29 | 6690 | 1999-05-04 | 2026-05-22 |  |
| 2542 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15170 | 2012-05-02 | 2026-05-29 | 15160 | 2012-05-02 | 2026-05-27 |  |
| 2542 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2545 | TaiwanStockPriceAdj | MISMATCH | 6336 | 2000-09-13 | 2026-05-29 | 6331 | 2000-09-13 | 2026-05-22 |  |
| 2545 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15181 | 2012-05-02 | 2026-05-29 | 15171 | 2012-05-02 | 2026-05-27 |  |
| 2545 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2603 | TaiwanStockPriceAdj | MISMATCH | 8765 | 1992-01-06 | 2026-05-29 | 8760 | 1992-01-06 | 2026-05-22 |  |
| 2603 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-29 | 15152 | 2012-05-02 | 2026-05-27 |  |
| 2603 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 2605 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2605 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15193 | 2012-05-02 | 2026-05-29 | 15183 | 2012-05-02 | 2026-05-27 |  |
| 2605 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2606 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2606 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15189 | 2012-05-02 | 2026-05-29 | 15179 | 2012-05-02 | 2026-05-27 |  |
| 2606 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2606 | TaiwanStockDividend | MISMATCH | 22 | 2005-08-02 | 2026-06-22 | 21 | 2005-08-02 | 2025-06-24 |  |
| 2610 | TaiwanStockPriceAdj | MISMATCH | 8437 | 1993-02-27 | 2026-05-29 | 8432 | 1993-02-27 | 2026-05-22 |  |
| 2610 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2610 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2612 | TaiwanStockPriceAdj | MISMATCH | 7971 | 1994-10-21 | 2026-05-29 | 7966 | 1994-10-21 | 2026-05-22 |  |
| 2612 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15115 | 2012-05-02 | 2026-05-29 | 15105 | 2012-05-02 | 2026-05-27 |  |
| 2612 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2615 | TaiwanStockPriceAdj | MISMATCH | 7530 | 1996-05-17 | 2026-05-29 | 7525 | 1996-05-17 | 2026-05-22 |  |
| 2615 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2615 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2617 | TaiwanStockPriceAdj | MISMATCH | 6933 | 1998-06-25 | 2026-05-29 | 6928 | 1998-06-25 | 2026-05-22 |  |
| 2617 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15147 | 2012-05-02 | 2026-05-29 | 15137 | 2012-05-02 | 2026-05-27 |  |
| 2617 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2618 | TaiwanStockPriceAdj | MISMATCH | 6067 | 2001-09-20 | 2026-05-29 | 6062 | 2001-09-20 | 2026-05-22 |  |
| 2618 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 2618 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2634 | TaiwanStockPriceAdj | MISMATCH | 2868 | 2014-08-26 | 2026-05-29 | 2863 | 2014-08-26 | 2026-05-22 |  |
| 2634 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13471 | 2014-08-25 | 2026-05-29 | 13461 | 2014-08-25 | 2026-05-27 |  |
| 2634 | TaiwanStockShareholding | MISMATCH | 2895 | 2012-03-29 | 2026-05-29 | 2890 | 2012-03-29 | 2026-05-22 |  |
| 2636 | TaiwanStockPriceAdj | MISMATCH | 4690 | 2007-04-24 | 2026-05-29 | 4685 | 2007-04-24 | 2026-05-22 |  |
| 2636 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15584 | 2007-08-02 | 2026-05-29 | 15574 | 2007-08-02 | 2026-05-27 |  |
| 2636 | TaiwanStockShareholding | MISMATCH | 2318 | 2012-03-29 | 2026-05-29 | 2313 | 2012-03-29 | 2026-05-22 |  |
| 2637 | TaiwanStockPriceAdj | MISMATCH | 4106 | 2009-09-01 | 2026-05-29 | 4101 | 2009-09-01 | 2026-05-22 |  |
| 2637 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15269 | 2009-09-08 | 2026-05-29 | 15259 | 2009-09-08 | 2026-05-27 |  |
| 2637 | TaiwanStockShareholding | MISMATCH | 3824 | 2010-11-30 | 2026-05-29 | 3819 | 2010-11-30 | 2026-05-22 |  |
| 2637 | TaiwanStockDividend | MISMATCH | 21 | 2010-03-01 | 2026-06-16 | 20 | 2010-03-01 | 2025-06-21 |  |
| 2727 | TaiwanStockPriceAdj | MISMATCH | 3690 | 2011-05-03 | 2026-05-29 | 3685 | 2011-05-03 | 2026-05-22 |  |
| 2727 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15378 | 2011-04-29 | 2026-05-29 | 15368 | 2011-04-29 | 2026-05-27 |  |
| 2727 | TaiwanStockShareholding | MISMATCH | 3511 | 2012-03-05 | 2026-05-29 | 3506 | 2012-03-05 | 2026-05-22 |  |
| 2731 | TaiwanStockPriceAdj | MISMATCH | 3317 | 2012-10-24 | 2026-05-29 | 3312 | 2012-10-24 | 2026-05-22 |  |
| 2731 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14114 | 2012-10-23 | 2026-05-29 | 14104 | 2012-10-23 | 2026-05-27 |  |
| 2731 | TaiwanStockShareholding | MISMATCH | 3123 | 2012-03-29 | 2026-05-29 | 3118 | 2012-03-29 | 2026-05-22 |  |
| 2731 | TaiwanStockDividend | MISMATCH | 12 | 2013-07-21 | 2026-06-15 | 11 | 2013-07-21 | 2025-07-19 |  |
| 2732 | TaiwanStockPriceAdj | MISMATCH | 3276 | 2012-12-19 | 2026-05-29 | 3271 | 2012-12-19 | 2026-05-22 |  |
| 2732 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11955 | 2013-01-14 | 2026-05-29 | 11945 | 2013-01-14 | 2026-05-27 |  |
| 2732 | TaiwanStockShareholding | MISMATCH | 2791 | 2015-01-19 | 2026-05-29 | 2786 | 2015-01-19 | 2026-05-22 |  |
| 2903 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 2903 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15189 | 2012-05-02 | 2026-05-29 | 15179 | 2012-05-02 | 2026-05-27 |  |
| 2903 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 2915 | TaiwanStockPriceAdj | MISMATCH | 5870 | 1992-01-06 | 2026-05-29 | 5865 | 1992-01-06 | 2026-05-22 |  |
| 2915 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15162 | 2012-05-02 | 2026-05-29 | 15152 | 2012-05-02 | 2026-05-27 |  |
| 2915 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3003 | TaiwanStockPriceAdj | MISMATCH | 5980 | 2002-02-20 | 2026-05-29 | 5975 | 2002-02-20 | 2026-05-22 |  |
| 3003 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15146 | 2012-05-02 | 2026-05-29 | 15136 | 2012-05-02 | 2026-05-27 |  |
| 3003 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3004 | TaiwanStockPriceAdj | MISMATCH | 5915 | 2002-02-26 | 2026-05-29 | 5910 | 2002-02-26 | 2026-05-22 |  |
| 3004 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14538 | 2012-05-02 | 2026-05-29 | 14528 | 2012-05-02 | 2026-05-27 |  |
| 3004 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3005 | TaiwanStockPriceAdj | MISMATCH | 5976 | 2002-02-26 | 2026-05-29 | 5971 | 2002-02-26 | 2026-05-22 |  |
| 3005 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 3005 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3006 | TaiwanStockPriceAdj | MISMATCH | 5972 | 2002-03-05 | 2026-05-29 | 5967 | 2002-03-05 | 2026-05-22 |  |
| 3006 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3006 | TaiwanStockDividend | MISMATCH | 22 | 2005-08-30 | 2026-07-08 | 21 | 2005-08-30 | 2025-07-15 |  |
| 3008 | TaiwanStockPriceAdj | MISMATCH | 5967 | 2002-03-12 | 2026-05-29 | 5962 | 2002-03-12 | 2026-05-22 |  |
| 3008 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3013 | TaiwanStockPriceAdj | MISMATCH | 5835 | 2002-09-17 | 2026-05-29 | 5830 | 2002-09-17 | 2026-05-22 |  |
| 3013 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15131 | 2012-05-02 | 2026-05-29 | 15121 | 2012-05-02 | 2026-05-27 |  |
| 3013 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3014 | TaiwanStockPriceAdj | MISMATCH | 5796 | 2002-10-30 | 2026-05-29 | 5791 | 2002-10-30 | 2026-05-22 |  |
| 3014 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15150 | 2012-05-02 | 2026-05-29 | 15140 | 2012-05-02 | 2026-05-27 |  |
| 3014 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3016 | TaiwanStockPriceAdj | MISMATCH | 5762 | 2002-12-25 | 2026-05-29 | 5757 | 2002-12-25 | 2026-05-22 |  |
| 3016 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15071 | 2012-05-02 | 2026-05-29 | 15061 | 2012-05-02 | 2026-05-27 |  |
| 3016 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3017 | TaiwanStockPriceAdj | MISMATCH | 5826 | 2002-09-30 | 2026-05-29 | 5821 | 2002-09-30 | 2026-05-22 |  |
| 3017 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3019 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3019 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3022 | TaiwanStockPriceAdj | MISMATCH | 5836 | 2002-08-27 | 2026-05-29 | 5831 | 2002-08-27 | 2026-05-22 |  |
| 3022 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15118 | 2012-05-02 | 2026-05-29 | 15108 | 2012-05-02 | 2026-05-27 |  |
| 3022 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3023 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3023 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 3023 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3025 | TaiwanStockPriceAdj | MISMATCH | 5842 | 2002-08-27 | 2026-05-29 | 5837 | 2002-08-27 | 2026-05-22 |  |
| 3025 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14101 | 2012-05-02 | 2026-05-29 | 14091 | 2012-05-02 | 2026-05-27 |  |
| 3025 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3029 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3029 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15012 | 2012-05-02 | 2026-05-29 | 15002 | 2012-05-02 | 2026-05-27 |  |
| 3029 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3030 | TaiwanStockPriceAdj | MISMATCH | 5805 | 2002-10-30 | 2026-05-29 | 5800 | 2002-10-30 | 2026-05-22 |  |
| 3030 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3030 | TaiwanStockDividend | MISMATCH | 22 | 2005-09-05 | 2026-06-19 | 21 | 2005-09-05 | 2025-06-21 |  |
| 3031 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3031 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15103 | 2012-05-02 | 2026-05-29 | 15093 | 2012-05-02 | 2026-05-27 |  |
| 3031 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3033 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3033 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15142 | 2012-05-02 | 2026-05-29 | 15132 | 2012-05-02 | 2026-05-27 |  |
| 3033 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3034 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3034 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 3034 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3036 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3036 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3036 | TaiwanStockDividend | MISMATCH | 26 | 2005-08-24 | 2026-06-14 | 25 | 2005-08-24 | 2025-07-06 |  |
| 3037 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3037 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3038 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3038 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14928 | 2012-05-02 | 2026-05-29 | 14918 | 2012-05-02 | 2026-05-27 |  |
| 3038 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3042 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3042 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15188 | 2012-05-02 | 2026-05-29 | 15178 | 2012-05-02 | 2026-05-27 |  |
| 3042 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3042 | TaiwanStockDividend | MISMATCH | 22 | 2005-08-31 | 2026-06-22 | 21 | 2005-08-31 | 2025-06-22 |  |
| 3044 | TaiwanStockPriceAdj | MISMATCH | 5849 | 2002-08-27 | 2026-05-29 | 5844 | 2002-08-27 | 2026-05-22 |  |
| 3044 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3048 | TaiwanStockPriceAdj | MISMATCH | 5824 | 2002-10-02 | 2026-05-29 | 5819 | 2002-10-02 | 2026-05-22 |  |
| 3048 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14874 | 2012-05-02 | 2026-05-29 | 14864 | 2012-05-02 | 2026-05-27 |  |
| 3048 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3059 | TaiwanStockPriceAdj | MISMATCH | 5758 | 2002-12-25 | 2026-05-29 | 5753 | 2002-12-25 | 2026-05-22 |  |
| 3059 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15164 | 2012-05-02 | 2026-05-29 | 15154 | 2012-05-02 | 2026-05-27 |  |
| 3059 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3071 | TaiwanStockPriceAdj | MISMATCH | 5152 | 2004-12-31 | 2026-05-29 | 5147 | 2004-12-31 | 2026-05-22 |  |
| 3071 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14026 | 2005-01-03 | 2026-05-29 | 14016 | 2005-01-03 | 2026-05-27 |  |
| 3071 | TaiwanStockShareholding | MISMATCH | 5291 | 2004-12-29 | 2026-05-29 | 5286 | 2004-12-29 | 2026-05-22 |  |
| 3078 | TaiwanStockPriceAdj | MISMATCH | 5171 | 2005-02-01 | 2026-05-29 | 5166 | 2005-02-01 | 2026-05-22 |  |
| 3078 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18641 | 2005-02-14 | 2026-05-29 | 18631 | 2005-02-14 | 2026-05-27 |  |
| 3078 | TaiwanStockShareholding | MISMATCH | 5269 | 2005-01-28 | 2026-05-29 | 5264 | 2005-01-28 | 2026-05-22 |  |
| 3083 | TaiwanStockPriceAdj | MISMATCH | 5463 | 2003-12-03 | 2026-05-29 | 5458 | 2003-12-03 | 2026-05-22 |  |
| 3083 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19306 | 2005-01-03 | 2026-05-29 | 19296 | 2005-01-03 | 2026-05-27 |  |
| 3083 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 3088 | TaiwanStockPriceAdj | MISMATCH | 5116 | 2005-04-29 | 2026-05-29 | 5111 | 2005-04-29 | 2026-05-22 |  |
| 3088 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17234 | 2005-04-29 | 2026-05-29 | 17224 | 2005-04-29 | 2026-05-27 |  |
| 3088 | TaiwanStockShareholding | MISMATCH | 5218 | 2005-04-27 | 2026-05-29 | 5213 | 2005-04-27 | 2026-05-22 |  |
| 3090 | TaiwanStockPriceAdj | MISMATCH | 5477 | 2003-11-11 | 2026-05-29 | 5472 | 2003-11-11 | 2026-05-22 |  |
| 3090 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16680 | 2005-02-16 | 2026-05-29 | 16670 | 2005-02-16 | 2026-05-27 |  |
| 3090 | TaiwanStockShareholding | MISMATCH | 4559 | 2007-12-28 | 2026-05-29 | 4554 | 2007-12-28 | 2026-05-22 |  |
| 3090 | TaiwanStockDividend | MISMATCH | 22 | 2005-08-08 | 2026-07-09 | 21 | 2005-08-08 | 2025-07-09 |  |
| 3094 | TaiwanStockPriceAdj | MISMATCH | 4766 | 2006-12-19 | 2026-05-29 | 4761 | 2006-12-19 | 2026-05-22 |  |
| 3094 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14890 | 2007-06-20 | 2026-05-29 | 14880 | 2007-06-20 | 2026-05-27 |  |
| 3094 | TaiwanStockShareholding | MISMATCH | 4661 | 2007-08-03 | 2026-05-29 | 4656 | 2007-08-03 | 2026-05-22 |  |
| 3094 | TaiwanStockDividend | MISMATCH | 21 | 2007-06-01 | 2026-06-28 | 20 | 2007-06-01 | 2025-06-25 |  |
| 3105 | TaiwanStockPriceAdj | MISMATCH | 4064 | 2009-10-07 | 2026-05-29 | 4059 | 2009-10-07 | 2026-05-22 |  |
| 3105 | TaiwanStockShareholding | MISMATCH | 3566 | 2011-12-12 | 2026-05-29 | 3561 | 2011-12-12 | 2026-05-22 |  |
| 3105 | TaiwanStockDividend | MISMATCH | 15 | 2011-07-17 | 2026-06-19 | 14 | 2011-07-17 | 2025-06-21 |  |
| 3141 | TaiwanStockPriceAdj | MISMATCH | 4725 | 2006-12-19 | 2026-05-29 | 4720 | 2006-12-19 | 2026-05-22 |  |
| 3141 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14130 | 2007-07-09 | 2026-05-29 | 14120 | 2007-07-09 | 2026-05-27 |  |
| 3141 | TaiwanStockShareholding | MISMATCH | 3006 | 2014-03-13 | 2026-05-29 | 3001 | 2014-03-13 | 2026-05-22 |  |
| 3169 | TaiwanStockPriceAdj | MISMATCH | 4754 | 2006-12-19 | 2026-05-29 | 4749 | 2006-12-19 | 2026-05-22 |  |
| 3169 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15224 | 2007-07-02 | 2026-05-29 | 15214 | 2007-07-02 | 2026-05-27 |  |
| 3169 | TaiwanStockShareholding | MISMATCH | 4082 | 2009-11-23 | 2026-05-29 | 4077 | 2009-11-23 | 2026-05-22 |  |
| 3169 | TaiwanStockDividend | MISMATCH | 22 | 2005-08-01 | 2026-06-30 | 21 | 2005-08-01 | 2025-06-30 |  |
| 3189 | TaiwanStockPriceAdj | MISMATCH | 5305 | 2004-11-02 | 2026-05-29 | 5300 | 2004-11-02 | 2026-05-22 |  |
| 3189 | TaiwanStockShareholding | MISMATCH | 5351 | 2004-10-29 | 2026-05-29 | 5346 | 2004-10-29 | 2026-05-22 |  |
| 3209 | TaiwanStockPriceAdj | MISMATCH | 5384 | 2004-03-30 | 2026-05-29 | 5379 | 2004-03-30 | 2026-05-22 |  |
| 3209 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15314 | 2005-01-04 | 2026-05-29 | 15304 | 2005-01-04 | 2026-05-27 |  |
| 3209 | TaiwanStockShareholding | MISMATCH | 4333 | 2008-11-25 | 2026-05-29 | 4328 | 2008-11-25 | 2026-05-22 |  |
| 3211 | TaiwanStockPriceAdj | MISMATCH | 5231 | 2004-11-09 | 2026-05-29 | 5226 | 2004-11-09 | 2026-05-22 |  |
| 3211 | TaiwanStockShareholding | MISMATCH | 5329 | 2004-11-05 | 2026-05-29 | 5324 | 2004-11-05 | 2026-05-22 |  |
| 3217 | TaiwanStockPriceAdj | MISMATCH | 5362 | 2004-05-03 | 2026-05-29 | 5357 | 2004-05-03 | 2026-05-22 |  |
| 3217 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13857 | 2005-01-03 | 2026-05-29 | 13847 | 2005-01-03 | 2026-05-27 |  |
| 3217 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 3218 | TaiwanStockPriceAdj | MISMATCH | 5216 | 2004-11-30 | 2026-05-29 | 5211 | 2004-11-30 | 2026-05-22 |  |
| 3218 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16096 | 2005-01-12 | 2026-05-29 | 16086 | 2005-01-12 | 2026-05-27 |  |
| 3218 | TaiwanStockShareholding | MISMATCH | 5314 | 2004-11-26 | 2026-05-29 | 5309 | 2004-11-26 | 2026-05-22 |  |
| 3227 | TaiwanStockPriceAdj | MISMATCH | 4865 | 2006-05-05 | 2026-05-29 | 4860 | 2006-05-05 | 2026-05-22 |  |
| 3227 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19648 | 2006-05-04 | 2026-05-29 | 19638 | 2006-05-04 | 2026-05-27 |  |
| 3227 | TaiwanStockShareholding | MISMATCH | 4971 | 2006-05-03 | 2026-05-29 | 4966 | 2006-05-03 | 2026-05-22 |  |
| 3231 | TaiwanStockPriceAdj | MISMATCH | 5605 | 2003-08-20 | 2026-05-29 | 5600 | 2003-08-20 | 2026-05-22 |  |
| 3231 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 3257 | TaiwanStockPriceAdj | MISMATCH | 3947 | 2010-04-23 | 2026-05-29 | 3942 | 2010-04-23 | 2026-05-22 |  |
| 3257 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15127 | 2010-04-28 | 2026-05-29 | 15117 | 2010-04-28 | 2026-05-27 |  |
| 3257 | TaiwanStockShareholding | MISMATCH | 3751 | 2011-03-18 | 2026-05-29 | 3746 | 2011-03-18 | 2026-05-22 |  |
| 3260 | TaiwanStockPriceAdj | MISMATCH | 5251 | 2004-10-11 | 2026-05-29 | 5246 | 2004-10-11 | 2026-05-22 |  |
| 3260 | TaiwanStockShareholding | MISMATCH | 5349 | 2004-10-07 | 2026-05-29 | 5344 | 2004-10-07 | 2026-05-22 |  |
| 3264 | TaiwanStockPriceAdj | MISMATCH | 5189 | 2005-01-06 | 2026-05-29 | 5184 | 2005-01-06 | 2026-05-22 |  |
| 3264 | TaiwanStockShareholding | MISMATCH | 5287 | 2005-01-04 | 2026-05-29 | 5282 | 2005-01-04 | 2026-05-22 |  |
| 3289 | TaiwanStockPriceAdj | MISMATCH | 5185 | 2004-12-29 | 2026-05-29 | 5180 | 2004-12-29 | 2026-05-22 |  |
| 3289 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18268 | 2005-01-11 | 2026-05-29 | 18258 | 2005-01-11 | 2026-05-27 |  |
| 3289 | TaiwanStockShareholding | MISMATCH | 5293 | 2004-12-27 | 2026-05-29 | 5288 | 2004-12-27 | 2026-05-22 |  |
| 3290 | TaiwanStockPriceAdj | MISMATCH | 5095 | 2005-05-20 | 2026-05-29 | 5090 | 2005-05-20 | 2026-05-22 |  |
| 3290 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15150 | 2005-05-19 | 2026-05-29 | 15140 | 2005-05-19 | 2026-05-27 |  |
| 3290 | TaiwanStockShareholding | MISMATCH | 5204 | 2005-05-18 | 2026-05-29 | 5199 | 2005-05-18 | 2026-05-22 |  |
| 3293 | TaiwanStockPriceAdj | MISMATCH | 4817 | 2006-07-13 | 2026-05-29 | 4812 | 2006-07-13 | 2026-05-22 |  |
| 3293 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19369 | 2006-07-12 | 2026-05-29 | 19359 | 2006-07-12 | 2026-05-27 |  |
| 3293 | TaiwanStockShareholding | MISMATCH | 4923 | 2006-07-11 | 2026-05-29 | 4918 | 2006-07-11 | 2026-05-22 |  |
| 3312 | TaiwanStockPriceAdj | MISMATCH | 4949 | 2005-11-08 | 2026-05-29 | 4944 | 2005-11-08 | 2026-05-22 |  |
| 3312 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15252 | 2005-11-07 | 2026-05-29 | 15242 | 2005-11-07 | 2026-05-27 |  |
| 3312 | TaiwanStockShareholding | MISMATCH | 3804 | 2010-12-28 | 2026-05-29 | 3799 | 2010-12-28 | 2026-05-22 |  |
| 3317 | TaiwanStockPriceAdj | MISMATCH | 4766 | 2006-12-19 | 2026-05-29 | 4761 | 2006-12-19 | 2026-05-22 |  |
| 3317 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15771 | 2007-06-27 | 2026-05-29 | 15761 | 2007-06-27 | 2026-05-27 |  |
| 3317 | TaiwanStockShareholding | MISMATCH | 4656 | 2007-08-08 | 2026-05-29 | 4651 | 2007-08-08 | 2026-05-22 |  |
| 3324 | TaiwanStockPriceAdj | MISMATCH | 5106 | 2005-05-16 | 2026-05-29 | 5101 | 2005-05-16 | 2026-05-22 |  |
| 3324 | TaiwanStockShareholding | MISMATCH | 5208 | 2005-05-12 | 2026-05-29 | 5203 | 2005-05-12 | 2026-05-22 |  |
| 3338 | TaiwanStockPriceAdj | MISMATCH | 4751 | 2006-12-19 | 2026-05-29 | 4746 | 2006-12-19 | 2026-05-22 |  |
| 3338 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13784 | 2007-08-23 | 2026-05-29 | 13774 | 2007-08-23 | 2026-05-27 |  |
| 3338 | TaiwanStockShareholding | MISMATCH | 3066 | 2012-03-29 | 2026-05-29 | 3061 | 2012-03-29 | 2026-05-22 |  |
| 3356 | TaiwanStockPriceAdj | MISMATCH | 5201 | 2005-03-29 | 2026-05-29 | 5196 | 2005-03-29 | 2026-05-22 |  |
| 3356 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15117 | 2012-05-02 | 2026-05-29 | 15107 | 2012-05-02 | 2026-05-27 |  |
| 3356 | TaiwanStockShareholding | MISMATCH | 5251 | 2005-03-25 | 2026-05-29 | 5246 | 2005-03-25 | 2026-05-22 |  |
| 3376 | TaiwanStockPriceAdj | MISMATCH | 4944 | 2005-12-30 | 2026-05-29 | 4939 | 2005-12-30 | 2026-05-22 |  |
| 3376 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16682 | 2005-12-26 | 2026-05-29 | 16672 | 2005-12-26 | 2026-05-27 |  |
| 3376 | TaiwanStockShareholding | MISMATCH | 4559 | 2007-12-28 | 2026-05-29 | 4554 | 2007-12-28 | 2026-05-22 |  |
| 3402 | TaiwanStockPriceAdj | MISMATCH | 4705 | 2006-12-19 | 2026-05-29 | 4700 | 2006-12-19 | 2026-05-22 |  |
| 3402 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14297 | 2007-01-11 | 2026-05-29 | 14287 | 2007-01-11 | 2026-05-27 |  |
| 3402 | TaiwanStockShareholding | MISMATCH | 4798 | 2007-01-08 | 2026-05-29 | 4793 | 2007-01-08 | 2026-05-22 |  |
| 3406 | TaiwanStockPriceAdj | MISMATCH | 5022 | 2005-12-21 | 2026-05-29 | 5017 | 2005-12-21 | 2026-05-22 |  |
| 3406 | TaiwanStockShareholding | MISMATCH | 5066 | 2005-12-19 | 2026-05-29 | 5061 | 2005-12-19 | 2026-05-22 |  |
| 3413 | TaiwanStockPriceAdj | MISMATCH | 4733 | 2006-12-19 | 2026-05-29 | 4728 | 2006-12-19 | 2026-05-22 |  |
| 3413 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13505 | 2007-06-20 | 2026-05-29 | 13495 | 2007-06-20 | 2026-05-27 |  |
| 3413 | TaiwanStockShareholding | MISMATCH | 2667 | 2012-03-29 | 2026-05-29 | 2662 | 2012-03-29 | 2026-05-22 |  |
| 3441 | TaiwanStockPriceAdj | MISMATCH | 4750 | 2006-12-19 | 2026-05-29 | 4745 | 2006-12-19 | 2026-05-22 |  |
| 3441 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14394 | 2007-06-29 | 2026-05-29 | 14384 | 2007-06-29 | 2026-05-27 |  |
| 3441 | TaiwanStockShareholding | MISMATCH | 3866 | 2010-10-01 | 2026-05-29 | 3861 | 2010-10-01 | 2026-05-22 |  |
| 3441 | TaiwanStockDividend | MISMATCH | 22 | 2005-08-08 | 2026-06-15 | 21 | 2005-08-08 | 2025-06-15 |  |
| 3444 | TaiwanStockPriceAdj | MISMATCH | 4694 | 2007-04-19 | 2026-05-29 | 4689 | 2007-04-19 | 2026-05-22 |  |
| 3444 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10971 | 2007-06-20 | 2026-05-29 | 10961 | 2007-06-20 | 2026-05-27 |  |
| 3444 | TaiwanStockShareholding | MISMATCH | 4374 | 2008-09-25 | 2026-05-29 | 4369 | 2008-09-25 | 2026-05-22 |  |
| 3450 | TaiwanStockPriceAdj | MISMATCH | 4944 | 2006-04-13 | 2026-05-29 | 4939 | 2006-04-13 | 2026-05-22 |  |
| 3450 | TaiwanStockShareholding | MISMATCH | 4991 | 2006-04-11 | 2026-05-29 | 4986 | 2006-04-11 | 2026-05-22 |  |
| 3455 | TaiwanStockPriceAdj | MISMATCH | 4763 | 2006-12-19 | 2026-05-29 | 4758 | 2006-12-19 | 2026-05-22 |  |
| 3455 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16668 | 2007-06-28 | 2026-05-29 | 16658 | 2007-06-28 | 2026-05-27 |  |
| 3455 | TaiwanStockShareholding | MISMATCH | 4559 | 2007-12-26 | 2026-05-29 | 4554 | 2007-12-26 | 2026-05-22 |  |
| 3479 | TaiwanStockPriceAdj | MISMATCH | 4576 | 2007-10-04 | 2026-05-29 | 4571 | 2007-10-04 | 2026-05-22 |  |
| 3479 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15875 | 2007-10-08 | 2026-05-29 | 15865 | 2007-10-08 | 2026-05-27 |  |
| 3479 | TaiwanStockShareholding | MISMATCH | 3948 | 2010-06-07 | 2026-05-29 | 3943 | 2010-06-07 | 2026-05-22 |  |
| 3481 | TaiwanStockPriceAdj | MISMATCH | 4793 | 2006-10-25 | 2026-05-29 | 4788 | 2006-10-25 | 2026-05-22 |  |
| 3481 | TaiwanStockShareholding | MISMATCH | 4856 | 2006-10-23 | 2026-05-29 | 4851 | 2006-10-23 | 2026-05-22 |  |
| 3481 | TaiwanStockDividend | MISMATCH | 15 | 2007-09-02 | 2026-06-17 | 14 | 2007-09-02 | 2025-05-28 |  |
| 3483 | TaiwanStockPriceAdj | MISMATCH | 4707 | 2006-12-18 | 2026-05-29 | 4702 | 2006-12-18 | 2026-05-22 |  |
| 3483 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16497 | 2006-12-18 | 2026-05-29 | 16487 | 2006-12-18 | 2026-05-27 |  |
| 3483 | TaiwanStockShareholding | MISMATCH | 4813 | 2006-12-15 | 2026-05-29 | 4808 | 2006-12-15 | 2026-05-22 |  |
| 3484 | TaiwanStockPriceAdj | MISMATCH | 4754 | 2006-12-13 | 2026-05-29 | 4749 | 2006-12-13 | 2026-05-22 |  |
| 3484 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15054 | 2007-09-27 | 2026-05-29 | 15044 | 2007-09-27 | 2026-05-27 |  |
| 3484 | TaiwanStockShareholding | MISMATCH | 4593 | 2007-11-08 | 2026-05-29 | 4588 | 2007-11-08 | 2026-05-22 |  |
| 3498 | TaiwanStockPriceAdj | MISMATCH | 4758 | 2006-12-19 | 2026-05-29 | 4753 | 2006-12-19 | 2026-05-22 |  |
| 3498 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15484 | 2007-08-07 | 2026-05-29 | 15474 | 2007-08-07 | 2026-05-27 |  |
| 3498 | TaiwanStockShareholding | MISMATCH | 4633 | 2007-09-10 | 2026-05-29 | 4628 | 2007-09-10 | 2026-05-22 |  |
| 3515 | TaiwanStockPriceAdj | MISMATCH | 4752 | 2006-12-19 | 2026-05-29 | 4747 | 2006-12-19 | 2026-05-22 |  |
| 3515 | TaiwanStockShareholding | MISMATCH | 4596 | 2007-11-07 | 2026-05-29 | 4591 | 2007-11-07 | 2026-05-22 |  |
| 3526 | TaiwanStockPriceAdj | MISMATCH | 4754 | 2006-12-13 | 2026-05-29 | 4749 | 2006-12-13 | 2026-05-22 |  |
| 3526 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16718 | 2007-11-02 | 2026-05-29 | 16708 | 2007-11-02 | 2026-05-27 |  |
| 3526 | TaiwanStockShareholding | MISMATCH | 4598 | 2007-11-01 | 2026-05-29 | 4593 | 2007-11-01 | 2026-05-22 |  |
| 3533 | TaiwanStockPriceAdj | MISMATCH | 4760 | 2006-12-13 | 2026-05-29 | 4755 | 2006-12-13 | 2026-05-22 |  |
| 3533 | TaiwanStockShareholding | MISMATCH | 4574 | 2007-12-07 | 2026-05-29 | 4569 | 2007-12-07 | 2026-05-22 |  |
| 3535 | TaiwanStockPriceAdj | MISMATCH | 4756 | 2006-12-19 | 2026-05-29 | 4751 | 2006-12-19 | 2026-05-22 |  |
| 3535 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15081 | 2007-06-26 | 2026-05-29 | 15071 | 2007-06-26 | 2026-05-27 |  |
| 3535 | TaiwanStockShareholding | MISMATCH | 4537 | 2008-01-30 | 2026-05-29 | 4532 | 2008-01-30 | 2026-05-22 |  |
| 3540 | TaiwanStockPriceAdj | MISMATCH | 4750 | 2006-12-19 | 2026-05-29 | 4745 | 2006-12-19 | 2026-05-22 |  |
| 3540 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2007-12-12 | 2026-05-29 | 15187 | 2007-12-12 | 2026-05-27 |  |
| 3540 | TaiwanStockShareholding | MISMATCH | 4570 | 2007-12-11 | 2026-05-29 | 4565 | 2007-12-11 | 2026-05-22 |  |
| 3540 | TaiwanStockDividend | MISMATCH | 20 | 2007-07-09 | 2026-06-16 | 19 | 2007-07-09 | 2024-07-10 |  |
| 3552 | TaiwanStockPriceAdj | MISMATCH | 4648 | 2007-06-28 | 2026-05-29 | 4643 | 2007-06-28 | 2026-05-22 |  |
| 3552 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17256 | 2007-11-27 | 2026-05-29 | 17246 | 2007-11-27 | 2026-05-27 |  |
| 3552 | TaiwanStockShareholding | MISMATCH | 4305 | 2009-01-06 | 2026-05-29 | 4300 | 2009-01-06 | 2026-05-22 |  |
| 3563 | TaiwanStockPriceAdj | MISMATCH | 4644 | 2007-06-28 | 2026-05-29 | 4639 | 2007-06-28 | 2026-05-22 |  |
| 3563 | TaiwanStockShareholding | MISMATCH | 1756 | 2019-04-01 | 2026-05-29 | 1751 | 2019-04-01 | 2026-05-22 |  |
| 3563 | TaiwanStockDividend | MISMATCH | 29 | 2007-08-01 | 2026-07-05 | 28 | 2007-08-01 | 2025-11-10 |  |
| 3580 | TaiwanStockPriceAdj | MISMATCH | 4554 | 2007-10-26 | 2026-05-29 | 4549 | 2007-10-26 | 2026-05-22 |  |
| 3580 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13293 | 2007-12-06 | 2026-05-29 | 13283 | 2007-12-06 | 2026-05-27 |  |
| 3580 | TaiwanStockShareholding | MISMATCH | 3926 | 2010-07-08 | 2026-05-29 | 3921 | 2010-07-08 | 2026-05-22 |  |
| 3583 | TaiwanStockPriceAdj | MISMATCH | 3446 | 2012-04-24 | 2026-05-29 | 3441 | 2012-04-24 | 2026-05-22 |  |
| 3583 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14520 | 2013-01-14 | 2026-05-29 | 14510 | 2013-01-14 | 2026-05-27 |  |
| 3583 | TaiwanStockShareholding | MISMATCH | 3257 | 2012-03-29 | 2026-05-29 | 3252 | 2012-03-29 | 2026-05-22 |  |
| 3587 | TaiwanStockPriceAdj | MISMATCH | 4500 | 2008-01-22 | 2026-05-29 | 4495 | 2008-01-22 | 2026-05-22 |  |
| 3587 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15189 | 2009-04-02 | 2026-05-29 | 15179 | 2009-04-02 | 2026-05-27 |  |
| 3587 | TaiwanStockShareholding | MISMATCH | 4152 | 2009-08-17 | 2026-05-29 | 4147 | 2009-08-17 | 2026-05-22 |  |
| 3594 | TaiwanStockPriceAdj | MISMATCH | 4524 | 2007-12-11 | 2026-05-29 | 4519 | 2007-12-11 | 2026-05-22 |  |
| 3594 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12732 | 2008-10-16 | 2026-05-29 | 12722 | 2008-10-16 | 2026-05-27 |  |
| 3594 | TaiwanStockShareholding | MISMATCH | 3219 | 2013-05-06 | 2026-05-29 | 3214 | 2013-05-06 | 2026-05-22 |  |
| 3596 | TaiwanStockPriceAdj | MISMATCH | 4561 | 2007-10-31 | 2026-05-29 | 4556 | 2007-10-31 | 2026-05-22 |  |
| 3596 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15553 | 2007-11-01 | 2026-05-29 | 15543 | 2007-11-01 | 2026-05-27 |  |
| 3596 | TaiwanStockShareholding | MISMATCH | 4263 | 2009-03-10 | 2026-05-29 | 4258 | 2009-03-10 | 2026-05-22 |  |
| 3605 | TaiwanStockPriceAdj | MISMATCH | 4518 | 2007-12-31 | 2026-05-29 | 4513 | 2007-12-31 | 2026-05-22 |  |
| 3605 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15267 | 2008-01-10 | 2026-05-29 | 15257 | 2008-01-10 | 2026-05-27 |  |
| 3605 | TaiwanStockShareholding | MISMATCH | 4252 | 2009-03-25 | 2026-05-29 | 4247 | 2009-03-25 | 2026-05-22 |  |
| 3607 | TaiwanStockPriceAdj | MISMATCH | 4497 | 2008-01-29 | 2026-05-29 | 4492 | 2008-01-29 | 2026-05-22 |  |
| 3607 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15206 | 2008-01-29 | 2026-05-29 | 15196 | 2008-01-29 | 2026-05-27 |  |
| 3607 | TaiwanStockShareholding | MISMATCH | 4101 | 2009-10-27 | 2026-05-29 | 4096 | 2009-10-27 | 2026-05-22 |  |
| 3615 | TaiwanStockPriceAdj | MISMATCH | 4369 | 2008-07-25 | 2026-05-29 | 4364 | 2008-07-25 | 2026-05-22 |  |
| 3615 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15306 | 2008-12-30 | 2026-05-29 | 15296 | 2008-12-30 | 2026-05-27 |  |
| 3615 | TaiwanStockShareholding | MISMATCH | 4238 | 2009-04-14 | 2026-05-29 | 4233 | 2009-04-14 | 2026-05-22 |  |
| 3622 | TaiwanStockPriceAdj | MISMATCH | 4461 | 2008-03-31 | 2026-05-29 | 4456 | 2008-03-31 | 2026-05-22 |  |
| 3622 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15426 | 2008-04-07 | 2026-05-29 | 15416 | 2008-04-07 | 2026-05-27 |  |
| 3622 | TaiwanStockShareholding | MISMATCH | 4253 | 2009-03-24 | 2026-05-29 | 4248 | 2009-03-24 | 2026-05-22 |  |
| 3624 | TaiwanStockPriceAdj | MISMATCH | 4426 | 2008-05-16 | 2026-05-29 | 4421 | 2008-05-16 | 2026-05-22 |  |
| 3624 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15696 | 2008-05-15 | 2026-05-29 | 15686 | 2008-05-15 | 2026-05-27 |  |
| 3624 | TaiwanStockShareholding | MISMATCH | 3754 | 2011-03-15 | 2026-05-29 | 3749 | 2011-03-15 | 2026-05-22 |  |
| 3630 | TaiwanStockPriceAdj | MISMATCH | 4379 | 2008-07-22 | 2026-05-29 | 4374 | 2008-07-22 | 2026-05-22 |  |
| 3630 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16724 | 2008-08-19 | 2026-05-29 | 16714 | 2008-08-19 | 2026-05-27 |  |
| 3630 | TaiwanStockShareholding | MISMATCH | 3979 | 2010-04-23 | 2026-05-29 | 3974 | 2010-04-23 | 2026-05-22 |  |
| 3645 | TaiwanStockPriceAdj | MISMATCH | 3843 | 2010-09-17 | 2026-05-29 | 3838 | 2010-09-17 | 2026-05-22 |  |
| 3645 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15301 | 2010-09-16 | 2026-05-29 | 15291 | 2010-09-16 | 2026-05-27 |  |
| 3645 | TaiwanStockShareholding | MISMATCH | 3614 | 2011-10-04 | 2026-05-29 | 3609 | 2011-10-04 | 2026-05-22 |  |
| 3653 | TaiwanStockPriceAdj | MISMATCH | 4296 | 2008-11-21 | 2026-05-29 | 4291 | 2008-11-21 | 2026-05-22 |  |
| 3653 | TaiwanStockShareholding | MISMATCH | 4086 | 2009-11-17 | 2026-05-29 | 4081 | 2009-11-17 | 2026-05-22 |  |
| 3665 | TaiwanStockPriceAdj | MISMATCH | 3699 | 2011-04-21 | 2026-05-29 | 3694 | 2011-04-21 | 2026-05-22 |  |
| 3665 | TaiwanStockShareholding | MISMATCH | 3730 | 2011-04-20 | 2026-05-29 | 3725 | 2011-04-20 | 2026-05-22 |  |
| 3673 | TaiwanStockPriceAdj | MISMATCH | 3813 | 2010-11-01 | 2026-05-29 | 3808 | 2010-11-01 | 2026-05-22 |  |
| 3673 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 3673 | TaiwanStockShareholding | MISMATCH | 3847 | 2010-10-28 | 2026-05-29 | 3842 | 2010-10-28 | 2026-05-22 |  |
| 3680 | TaiwanStockPriceAdj | MISMATCH | 4072 | 2009-10-19 | 2026-05-29 | 4067 | 2009-10-19 | 2026-05-22 |  |
| 3680 | TaiwanStockShareholding | MISMATCH | 3638 | 2011-08-30 | 2026-05-29 | 3633 | 2011-08-30 | 2026-05-22 |  |
| 3687 | TaiwanStockPriceAdj | MISMATCH | 4016 | 2010-01-06 | 2026-05-29 | 4011 | 2010-01-06 | 2026-05-22 |  |
| 3687 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14572 | 2010-09-06 | 2026-05-29 | 14562 | 2010-09-06 | 2026-05-27 |  |
| 3687 | TaiwanStockShareholding | MISMATCH | 3795 | 2011-01-10 | 2026-05-29 | 3790 | 2011-01-10 | 2026-05-22 |  |
| 3689 | TaiwanStockPriceAdj | MISMATCH | 3777 | 2010-12-17 | 2026-05-29 | 3772 | 2010-12-17 | 2026-05-22 |  |
| 3689 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14673 | 2010-12-16 | 2026-05-29 | 14663 | 2010-12-16 | 2026-05-27 |  |
| 3689 | TaiwanStockShareholding | MISMATCH | 3353 | 2012-10-17 | 2026-05-29 | 3348 | 2012-10-17 | 2026-05-22 |  |
| 3693 | TaiwanStockPriceAdj | MISMATCH | 3930 | 2010-05-11 | 2026-05-29 | 3925 | 2010-05-11 | 2026-05-22 |  |
| 3693 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12178 | 2010-12-30 | 2026-05-29 | 12168 | 2010-12-30 | 2026-05-27 |  |
| 3693 | TaiwanStockShareholding | MISMATCH | 3077 | 2013-11-26 | 2026-05-29 | 3072 | 2013-11-26 | 2026-05-22 |  |
| 3694 | TaiwanStockPriceAdj | MISMATCH | 4062 | 2009-11-02 | 2026-05-29 | 4057 | 2009-11-02 | 2026-05-22 |  |
| 3694 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15042 | 2009-12-10 | 2026-05-29 | 15032 | 2009-12-10 | 2026-05-27 |  |
| 3694 | TaiwanStockShareholding | MISMATCH | 3723 | 2011-04-29 | 2026-05-29 | 3718 | 2011-04-29 | 2026-05-22 |  |
| 3702 | TaiwanStockPriceAdj | MISMATCH | 5045 | 2005-11-10 | 2026-05-29 | 5040 | 2005-11-10 | 2026-05-22 |  |
| 3702 | TaiwanStockShareholding | MISMATCH | 5095 | 2005-11-08 | 2026-05-29 | 5090 | 2005-11-08 | 2026-05-22 |  |
| 3703 | TaiwanStockPriceAdj | MISMATCH | 3958 | 2010-04-07 | 2026-05-29 | 3953 | 2010-04-07 | 2026-05-22 |  |
| 3703 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15151 | 2012-05-02 | 2026-05-29 | 15141 | 2012-05-02 | 2026-05-27 |  |
| 3703 | TaiwanStockShareholding | MISMATCH | 3991 | 2010-04-07 | 2026-05-29 | 3986 | 2010-04-07 | 2026-05-22 |  |
| 3704 | TaiwanStockPriceAdj | MISMATCH | 3860 | 2010-08-13 | 2026-05-29 | 3855 | 2010-08-13 | 2026-05-22 |  |
| 3704 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15154 | 2012-05-02 | 2026-05-29 | 15144 | 2012-05-02 | 2026-05-27 |  |
| 3704 | TaiwanStockShareholding | MISMATCH | 3900 | 2010-08-13 | 2026-05-29 | 3895 | 2010-08-13 | 2026-05-22 |  |
| 3706 | TaiwanStockPriceAdj | MISMATCH | 3101 | 2013-09-13 | 2026-05-29 | 3096 | 2013-09-13 | 2026-05-22 |  |
| 3706 | TaiwanStockShareholding | MISMATCH | 3130 | 2012-03-29 | 2026-05-29 | 3125 | 2012-03-29 | 2026-05-22 |  |
| 4105 | TaiwanStockPriceAdj | MISMATCH | 6004 | 2001-09-28 | 2026-05-29 | 5999 | 2001-09-28 | 2026-05-22 |  |
| 4105 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18906 | 2005-03-24 | 2026-05-29 | 18896 | 2005-03-24 | 2026-05-27 |  |
| 4105 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 4114 | TaiwanStockPriceAdj | MISMATCH | 5606 | 2003-05-13 | 2026-05-29 | 5601 | 2003-05-13 | 2026-05-22 |  |
| 4114 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17601 | 2005-09-29 | 2026-05-29 | 17591 | 2005-09-29 | 2026-05-27 |  |
| 4114 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 4119 | TaiwanStockPriceAdj | MISMATCH | 5500 | 2004-01-08 | 2026-05-29 | 5495 | 2004-01-08 | 2026-05-22 |  |
| 4119 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15124 | 2012-05-02 | 2026-05-29 | 15114 | 2012-05-02 | 2026-05-27 |  |
| 4119 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 4306 | TaiwanStockPriceAdj | MISMATCH | 5538 | 2003-08-04 | 2026-05-29 | 5533 | 2003-08-04 | 2026-05-22 |  |
| 4306 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15521 | 2005-01-03 | 2026-05-29 | 15511 | 2005-01-03 | 2026-05-27 |  |
| 4306 | TaiwanStockShareholding | MISMATCH | 4545 | 2008-01-18 | 2026-05-29 | 4540 | 2008-01-18 | 2026-05-22 |  |
| 4551 | TaiwanStockPriceAdj | MISMATCH | 3019 | 2014-01-09 | 2026-05-29 | 3014 | 2014-01-09 | 2026-05-22 |  |
| 4551 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12637 | 2014-03-04 | 2026-05-29 | 12627 | 2014-03-04 | 2026-05-27 |  |
| 4551 | TaiwanStockShareholding | MISMATCH | 2658 | 2012-03-29 | 2026-05-29 | 2653 | 2012-03-29 | 2026-05-22 |  |
| 4721 | TaiwanStockPriceAdj | MISMATCH | 6141 | 2001-03-12 | 2026-05-29 | 6136 | 2001-03-12 | 2026-05-22 |  |
| 4721 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15791 | 2005-01-03 | 2026-05-29 | 15781 | 2005-01-03 | 2026-05-27 |  |
| 4721 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 4739 | TaiwanStockPriceAdj | MISMATCH | 3782 | 2010-12-14 | 2026-05-29 | 3777 | 2010-12-14 | 2026-05-22 |  |
| 4739 | TaiwanStockShareholding | MISMATCH | 2142 | 2012-03-29 | 2026-05-29 | 2137 | 2012-03-29 | 2026-05-22 |  |
| 4746 | TaiwanStockPriceAdj | MISMATCH | 4042 | 2009-11-30 | 2026-05-29 | 4037 | 2009-11-30 | 2026-05-22 |  |
| 4746 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15268 | 2009-12-03 | 2026-05-29 | 15258 | 2009-12-03 | 2026-05-27 |  |
| 4746 | TaiwanStockShareholding | MISMATCH | 3765 | 2011-02-25 | 2026-05-29 | 3760 | 2011-02-25 | 2026-05-22 |  |
| 4763 | TaiwanStockPriceAdj | MISMATCH | 2564 | 2015-11-10 | 2026-05-29 | 2559 | 2015-11-10 | 2026-05-22 |  |
| 4763 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12289 | 2015-11-09 | 2026-05-29 | 12279 | 2015-11-09 | 2026-05-27 |  |
| 4763 | TaiwanStockShareholding | MISMATCH | 2596 | 2012-03-29 | 2026-05-29 | 2591 | 2012-03-29 | 2026-05-22 |  |
| 4912 | TaiwanStockPriceAdj | MISMATCH | 4028 | 2009-12-18 | 2026-05-29 | 4023 | 2009-12-18 | 2026-05-22 |  |
| 4912 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15029 | 2011-04-29 | 2026-05-29 | 15019 | 2011-04-29 | 2026-05-27 |  |
| 4912 | TaiwanStockShareholding | MISMATCH | 2713 | 2012-03-29 | 2026-05-29 | 2708 | 2012-03-29 | 2026-05-22 |  |
| 4915 | TaiwanStockPriceAdj | MISMATCH | 4016 | 2010-01-04 | 2026-05-29 | 4011 | 2010-01-04 | 2026-05-22 |  |
| 4915 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15386 | 2009-12-31 | 2026-05-29 | 15376 | 2009-12-31 | 2026-05-27 |  |
| 4915 | TaiwanStockShareholding | MISMATCH | 3362 | 2012-03-29 | 2026-05-29 | 3357 | 2012-03-29 | 2026-05-22 |  |
| 4916 | TaiwanStockPriceAdj | MISMATCH | 4016 | 2009-12-28 | 2026-05-29 | 4011 | 2009-12-28 | 2026-05-22 |  |
| 4916 | TaiwanStockShareholding | MISMATCH | 3082 | 2012-03-29 | 2026-05-29 | 3077 | 2012-03-29 | 2026-05-22 |  |
| 4938 | TaiwanStockPriceAdj | MISMATCH | 3903 | 2010-06-25 | 2026-05-29 | 3898 | 2010-06-25 | 2026-05-22 |  |
| 4938 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15192 | 2012-05-02 | 2026-05-29 | 15182 | 2012-05-02 | 2026-05-27 |  |
| 4938 | TaiwanStockShareholding | MISMATCH | 3937 | 2010-06-23 | 2026-05-29 | 3932 | 2010-06-23 | 2026-05-22 |  |
| 4952 | TaiwanStockPriceAdj | MISMATCH | 3871 | 2010-08-10 | 2026-05-29 | 3866 | 2010-08-10 | 2026-05-22 |  |
| 4952 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15178 | 2010-08-17 | 2026-05-29 | 15168 | 2010-08-17 | 2026-05-27 |  |
| 4952 | TaiwanStockShareholding | MISMATCH | 3596 | 2011-10-31 | 2026-05-29 | 3591 | 2011-10-31 | 2026-05-22 |  |
| 4952 | TaiwanStockDividend | MISMATCH | 16 | 2011-06-15 | 2026-06-30 | 15 | 2011-06-15 | 2025-07-01 |  |
| 4953 | TaiwanStockPriceAdj | MISMATCH | 3897 | 2010-06-28 | 2026-05-29 | 3892 | 2010-06-28 | 2026-05-22 |  |
| 4953 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12166 | 2010-11-19 | 2026-05-29 | 12156 | 2010-11-19 | 2026-05-27 |  |
| 4953 | TaiwanStockShareholding | MISMATCH | 3048 | 2014-01-07 | 2026-05-29 | 3043 | 2014-01-07 | 2026-05-22 |  |
| 4958 | TaiwanStockPriceAdj | MISMATCH | 3525 | 2011-12-27 | 2026-05-29 | 3520 | 2011-12-27 | 2026-05-22 |  |
| 4958 | TaiwanStockShareholding | MISMATCH | 3557 | 2011-12-23 | 2026-05-29 | 3552 | 2011-12-23 | 2026-05-22 |  |
| 4958 | TaiwanStockDividend | MISMATCH | 15 | 2012-08-19 | 2026-06-14 | 14 | 2012-08-19 | 2025-06-19 |  |
| 4966 | TaiwanStockPriceAdj | MISMATCH | 3598 | 2011-09-14 | 2026-05-29 | 3593 | 2011-09-14 | 2026-05-22 |  |
| 4966 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15529 | 2011-09-13 | 2026-05-29 | 15519 | 2011-09-13 | 2026-05-27 |  |
| 4966 | TaiwanStockShareholding | MISMATCH | 3630 | 2011-09-09 | 2026-05-29 | 3625 | 2011-09-09 | 2026-05-22 |  |
| 4968 | TaiwanStockPriceAdj | MISMATCH | 3827 | 2010-10-04 | 2026-05-29 | 3822 | 2010-10-04 | 2026-05-22 |  |
| 4968 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12514 | 2010-10-01 | 2026-05-29 | 12504 | 2010-10-01 | 2026-05-27 |  |
| 4968 | TaiwanStockShareholding | MISMATCH | 2592 | 2012-03-29 | 2026-05-29 | 2587 | 2012-03-29 | 2026-05-22 |  |
| 4973 | TaiwanStockPriceAdj | MISMATCH | 3758 | 2011-01-12 | 2026-05-29 | 3753 | 2011-01-12 | 2026-05-22 |  |
| 4973 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12357 | 2012-06-19 | 2026-05-29 | 12347 | 2012-06-19 | 2026-05-27 |  |
| 4973 | TaiwanStockShareholding | MISMATCH | 3438 | 2012-06-18 | 2026-05-29 | 3433 | 2012-06-18 | 2026-05-22 |  |
| 4977 | TaiwanStockPriceAdj | MISMATCH | 3271 | 2012-12-27 | 2026-05-29 | 3266 | 2012-12-27 | 2026-05-22 |  |
| 4977 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14326 | 2012-12-26 | 2026-05-29 | 14316 | 2012-12-26 | 2026-05-27 |  |
| 4977 | TaiwanStockShareholding | MISMATCH | 3083 | 2012-03-29 | 2026-05-29 | 3078 | 2012-03-29 | 2026-05-22 |  |
| 4979 | TaiwanStockPriceAdj | MISMATCH | 3801 | 2010-11-17 | 2026-05-29 | 3796 | 2010-11-17 | 2026-05-22 |  |
| 4979 | TaiwanStockShareholding | MISMATCH | 3567 | 2011-12-09 | 2026-05-29 | 3562 | 2011-12-09 | 2026-05-22 |  |
| 5009 | TaiwanStockPriceAdj | MISMATCH | 6770 | 1998-10-19 | 2026-05-29 | 6765 | 1998-10-19 | 2026-05-22 |  |
| 5009 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20468 | 2005-01-03 | 2026-05-29 | 20458 | 2005-01-03 | 2026-05-27 |  |
| 5009 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5203 | TaiwanStockPriceAdj | MISMATCH | 5605 | 2003-08-04 | 2026-05-29 | 5600 | 2003-08-04 | 2026-05-22 |  |
| 5203 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15142 | 2012-05-02 | 2026-05-29 | 15132 | 2012-05-02 | 2026-05-27 |  |
| 5203 | TaiwanStockShareholding | MISMATCH | 5373 | 2004-09-27 | 2026-05-29 | 5368 | 2004-09-27 | 2026-05-22 |  |
| 5243 | TaiwanStockPriceAdj | MISMATCH | 3051 | 2013-11-26 | 2026-05-29 | 3046 | 2013-11-26 | 2026-05-22 |  |
| 5243 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14004 | 2013-05-16 | 2026-05-29 | 13994 | 2013-05-16 | 2026-05-27 |  |
| 5243 | TaiwanStockShareholding | MISMATCH | 3080 | 2012-03-29 | 2026-05-29 | 3075 | 2012-03-29 | 2026-05-22 |  |
| 5258 | TaiwanStockPriceAdj | MISMATCH | 3546 | 2011-11-15 | 2026-05-29 | 3541 | 2011-11-15 | 2026-05-22 |  |
| 5258 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10792 | 2014-03-07 | 2026-05-29 | 10782 | 2014-03-07 | 2026-05-27 |  |
| 5258 | TaiwanStockShareholding | MISMATCH | 2312 | 2012-03-29 | 2026-05-29 | 2307 | 2012-03-29 | 2026-05-22 |  |
| 5285 | TaiwanStockPriceAdj | MISMATCH | 3345 | 2012-09-13 | 2026-05-29 | 3340 | 2012-09-13 | 2026-05-22 |  |
| 5285 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12888 | 2013-06-04 | 2026-05-29 | 12878 | 2013-06-04 | 2026-05-27 |  |
| 5285 | TaiwanStockShareholding | MISMATCH | 3019 | 2012-03-29 | 2026-05-29 | 3014 | 2012-03-29 | 2026-05-22 |  |
| 5269 | TaiwanStockPriceAdj | MISMATCH | 3510 | 2012-01-13 | 2026-05-29 | 3505 | 2012-01-13 | 2026-05-22 |  |
| 5269 | TaiwanStockShareholding | MISMATCH | 3315 | 2012-03-29 | 2026-05-29 | 3310 | 2012-03-29 | 2026-05-22 |  |
| 5289 | TaiwanStockPriceAdj | MISMATCH | 3328 | 2012-10-08 | 2026-05-29 | 3323 | 2012-10-08 | 2026-05-22 |  |
| 5289 | TaiwanStockShareholding | MISMATCH | 3077 | 2013-11-26 | 2026-05-29 | 3072 | 2013-11-26 | 2026-05-22 |  |
| 5347 | TaiwanStockPriceAdj | MISMATCH | 6920 | 1998-03-26 | 2026-05-29 | 6915 | 1998-03-26 | 2026-05-22 |  |
| 5347 | TaiwanStockShareholding | MISMATCH | 5377 | 2004-08-04 | 2026-05-29 | 5372 | 2004-08-04 | 2026-05-22 |  |
| 5388 | TaiwanStockPriceAdj | MISMATCH | 5546 | 2003-08-04 | 2026-05-29 | 5541 | 2003-08-04 | 2026-05-22 |  |
| 5388 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16742 | 2005-01-03 | 2026-05-29 | 16732 | 2005-01-03 | 2026-05-27 |  |
| 5388 | TaiwanStockShareholding | MISMATCH | 4579 | 2007-11-30 | 2026-05-29 | 4574 | 2007-11-30 | 2026-05-22 |  |
| 5371 | TaiwanStockPriceAdj | MISMATCH | 6687 | 1999-01-21 | 2026-05-29 | 6682 | 1999-01-21 | 2026-05-22 |  |
| 5371 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20545 | 2005-01-03 | 2026-05-29 | 20535 | 2005-01-03 | 2026-05-27 |  |
| 5371 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5425 | TaiwanStockPriceAdj | MISMATCH | 6421 | 2000-02-22 | 2026-05-29 | 6416 | 2000-02-22 | 2026-05-22 |  |
| 5425 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5434 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 5434 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15190 | 2012-05-02 | 2026-05-29 | 15180 | 2012-05-02 | 2026-05-27 |  |
| 5434 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 5439 | TaiwanStockPriceAdj | MISMATCH | 6325 | 2000-06-27 | 2026-05-29 | 6320 | 2000-06-27 | 2026-05-22 |  |
| 5439 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5443 | TaiwanStockPriceAdj | MISMATCH | 6954 | 1998-02-10 | 2026-05-29 | 6949 | 1998-02-10 | 2026-05-22 |  |
| 5443 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5469 | TaiwanStockPriceAdj | MISMATCH | 5610 | 2003-08-04 | 2026-05-29 | 5605 | 2003-08-04 | 2026-05-22 |  |
| 5469 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 5457 | TaiwanStockPriceAdj | MISMATCH | 6246 | 2000-09-30 | 2026-05-29 | 6241 | 2000-09-30 | 2026-05-22 |  |
| 5457 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16767 | 2005-03-10 | 2026-05-29 | 16757 | 2005-03-10 | 2026-05-27 |  |
| 5457 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5471 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 5471 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15168 | 2012-05-02 | 2026-05-29 | 15158 | 2012-05-02 | 2026-05-27 |  |
| 5471 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 5478 | TaiwanStockPriceAdj | MISMATCH | 6127 | 2001-03-30 | 2026-05-29 | 6122 | 2001-03-30 | 2026-05-22 |  |
| 5478 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20250 | 2005-01-03 | 2026-05-29 | 20240 | 2005-01-03 | 2026-05-27 |  |
| 5478 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5483 | TaiwanStockPriceAdj | MISMATCH | 6146 | 2001-03-05 | 2026-05-29 | 6141 | 2001-03-05 | 2026-05-22 |  |
| 5483 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5512 | TaiwanStockPriceAdj | MISMATCH | 6739 | 1998-12-07 | 2026-05-29 | 6734 | 1998-12-07 | 2026-05-22 |  |
| 5512 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19507 | 2005-01-03 | 2026-05-29 | 19497 | 2005-01-03 | 2026-05-27 |  |
| 5512 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 5607 | TaiwanStockPriceAdj | MISMATCH | 5611 | 2003-08-04 | 2026-05-29 | 5606 | 2003-08-04 | 2026-05-22 |  |
| 5607 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14476 | 2012-05-02 | 2026-05-29 | 14466 | 2012-05-02 | 2026-05-27 |  |
| 5607 | TaiwanStockShareholding | MISMATCH | 5327 | 2004-12-02 | 2026-05-29 | 5322 | 2004-12-02 | 2026-05-22 |  |
| 5608 | TaiwanStockPriceAdj | MISMATCH | 5611 | 2003-08-04 | 2026-05-29 | 5606 | 2003-08-04 | 2026-05-22 |  |
| 5608 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15015 | 2012-05-02 | 2026-05-29 | 15005 | 2012-05-02 | 2026-05-27 |  |
| 5608 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 5706 | TaiwanStockPriceAdj | MISMATCH | 5546 | 2003-08-04 | 2026-05-29 | 5541 | 2003-08-04 | 2026-05-22 |  |
| 5706 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15961 | 2005-12-06 | 2026-05-29 | 15951 | 2005-12-06 | 2026-05-27 |  |
| 5706 | TaiwanStockShareholding | MISMATCH | 3603 | 2011-10-20 | 2026-05-29 | 3598 | 2011-10-20 | 2026-05-22 |  |
| 5871 | TaiwanStockPriceAdj | MISMATCH | 3644 | 2011-07-11 | 2026-05-29 | 3639 | 2011-07-11 | 2026-05-22 |  |
| 5871 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15497 | 2011-07-11 | 2026-05-29 | 15487 | 2011-07-11 | 2026-05-27 |  |
| 5871 | TaiwanStockShareholding | MISMATCH | 3566 | 2011-12-12 | 2026-05-29 | 3561 | 2011-12-12 | 2026-05-22 |  |
| 6118 | TaiwanStockPriceAdj | MISMATCH | 5986 | 2001-10-26 | 2026-05-29 | 5981 | 2001-10-26 | 2026-05-22 |  |
| 6118 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13001 | 2005-04-29 | 2026-05-29 | 12991 | 2005-04-29 | 2026-05-27 |  |
| 6118 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6104 | TaiwanStockPriceAdj | MISMATCH | 6091 | 2001-05-23 | 2026-05-29 | 6086 | 2001-05-23 | 2026-05-22 |  |
| 6104 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19172 | 2005-01-06 | 2026-05-29 | 19162 | 2005-01-06 | 2026-05-27 |  |
| 6104 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6121 | TaiwanStockPriceAdj | MISMATCH | 5953 | 2001-11-28 | 2026-05-29 | 5948 | 2001-11-28 | 2026-05-22 |  |
| 6121 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20586 | 2005-01-03 | 2026-05-29 | 20576 | 2005-01-03 | 2026-05-27 |  |
| 6121 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6125 | TaiwanStockPriceAdj | MISMATCH | 5923 | 2002-01-24 | 2026-05-29 | 5918 | 2002-01-24 | 2026-05-22 |  |
| 6125 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19087 | 2005-01-25 | 2026-05-29 | 19077 | 2005-01-25 | 2026-05-27 |  |
| 6125 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6126 | TaiwanStockPriceAdj | MISMATCH | 5926 | 2002-01-21 | 2026-05-29 | 5921 | 2002-01-21 | 2026-05-22 |  |
| 6126 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16649 | 2005-06-16 | 2026-05-29 | 16639 | 2005-06-16 | 2026-05-27 |  |
| 6126 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6138 | TaiwanStockPriceAdj | MISMATCH | 5912 | 2002-01-21 | 2026-05-29 | 5907 | 2002-01-21 | 2026-05-22 |  |
| 6138 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6139 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 6139 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 6143 | TaiwanStockPriceAdj | MISMATCH | 5920 | 2002-01-29 | 2026-05-29 | 5915 | 2002-01-29 | 2026-05-22 |  |
| 6143 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18970 | 2005-12-22 | 2026-05-29 | 18960 | 2005-12-22 | 2026-05-27 |  |
| 6143 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6147 | TaiwanStockPriceAdj | MISMATCH | 5917 | 2002-02-01 | 2026-05-29 | 5912 | 2002-02-01 | 2026-05-22 |  |
| 6147 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20563 | 2005-01-03 | 2026-05-29 | 20553 | 2005-01-03 | 2026-05-27 |  |
| 6147 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6147 | TaiwanStockDividend | MISMATCH | 20 | 2005-06-30 | 2026-06-23 | 19 | 2005-06-30 | 2025-06-23 |  |
| 6150 | TaiwanStockPriceAdj | MISMATCH | 5880 | 2002-03-12 | 2026-05-29 | 5875 | 2002-03-12 | 2026-05-22 |  |
| 6150 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15565 | 2005-01-03 | 2026-05-29 | 15555 | 2005-01-03 | 2026-05-27 |  |
| 6150 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6161 | TaiwanStockPriceAdj | MISMATCH | 5891 | 2002-03-04 | 2026-05-29 | 5886 | 2002-03-04 | 2026-05-22 |  |
| 6161 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14718 | 2005-03-25 | 2026-05-29 | 14708 | 2005-03-25 | 2026-05-27 |  |
| 6161 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6163 | TaiwanStockPriceAdj | MISMATCH | 5892 | 2002-03-20 | 2026-05-29 | 5887 | 2002-03-20 | 2026-05-22 |  |
| 6163 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15712 | 2005-01-11 | 2026-05-29 | 15702 | 2005-01-11 | 2026-05-27 |  |
| 6163 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6166 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 6166 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15163 | 2012-05-02 | 2026-05-29 | 15153 | 2012-05-02 | 2026-05-27 |  |
| 6166 | TaiwanStockShareholding | MISMATCH | 5345 | 2004-11-08 | 2026-05-29 | 5340 | 2004-11-08 | 2026-05-22 |  |
| 6168 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 6168 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15146 | 2012-05-02 | 2026-05-29 | 15136 | 2012-05-02 | 2026-05-27 |  |
| 6168 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 6170 | TaiwanStockPriceAdj | MISMATCH | 5873 | 2002-04-17 | 2026-05-29 | 5868 | 2002-04-17 | 2026-05-22 |  |
| 6170 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16074 | 2005-03-08 | 2026-05-29 | 16064 | 2005-03-08 | 2026-05-27 |  |
| 6170 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6173 | TaiwanStockPriceAdj | MISMATCH | 5862 | 2002-04-22 | 2026-05-29 | 5857 | 2002-04-22 | 2026-05-22 |  |
| 6173 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17166 | 2005-01-21 | 2026-05-29 | 17156 | 2005-01-21 | 2026-05-27 |  |
| 6173 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6175 | TaiwanStockPriceAdj | MISMATCH | 5835 | 2002-06-11 | 2026-05-29 | 5830 | 2002-06-11 | 2026-05-22 |  |
| 6175 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14907 | 2005-01-04 | 2026-05-29 | 14897 | 2005-01-04 | 2026-05-27 |  |
| 6175 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6176 | TaiwanStockPriceAdj | MISMATCH | 5546 | 2003-08-04 | 2026-05-29 | 5541 | 2003-08-04 | 2026-05-22 |  |
| 6176 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16931 | 2005-01-03 | 2026-05-29 | 16921 | 2005-01-03 | 2026-05-27 |  |
| 6176 | TaiwanStockShareholding | MISMATCH | 4719 | 2007-05-14 | 2026-05-29 | 4714 | 2007-05-14 | 2026-05-22 |  |
| 6177 | TaiwanStockPriceAdj | MISMATCH | 5546 | 2003-08-04 | 2026-05-29 | 5541 | 2003-08-04 | 2026-05-22 |  |
| 6177 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15625 | 2005-04-11 | 2026-05-29 | 15615 | 2005-04-11 | 2026-05-27 |  |
| 6177 | TaiwanStockShareholding | MISMATCH | 3159 | 2012-03-29 | 2026-05-29 | 3154 | 2012-03-29 | 2026-05-22 |  |
| 6187 | TaiwanStockPriceAdj | MISMATCH | 5757 | 2002-09-30 | 2026-05-29 | 5752 | 2002-09-30 | 2026-05-22 |  |
| 6187 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17266 | 2005-01-03 | 2026-05-29 | 17256 | 2005-01-03 | 2026-05-27 |  |
| 6187 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6188 | TaiwanStockPriceAdj | MISMATCH | 5779 | 2002-08-28 | 2026-05-29 | 5774 | 2002-08-28 | 2026-05-22 |  |
| 6188 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 20361 | 2005-01-03 | 2026-05-29 | 20351 | 2005-01-03 | 2026-05-27 |  |
| 6188 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6190 | TaiwanStockPriceAdj | MISMATCH | 5798 | 2002-08-01 | 2026-05-29 | 5793 | 2002-08-01 | 2026-05-22 |  |
| 6190 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13995 | 2006-04-25 | 2026-05-29 | 13985 | 2006-04-25 | 2026-05-27 |  |
| 6190 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6191 | TaiwanStockPriceAdj | MISMATCH | 5541 | 2003-08-04 | 2026-05-29 | 5536 | 2003-08-04 | 2026-05-22 |  |
| 6191 | TaiwanStockShareholding | MISMATCH | 4610 | 2007-10-18 | 2026-05-29 | 4605 | 2007-10-18 | 2026-05-22 |  |
| 6196 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 6196 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15156 | 2012-05-02 | 2026-05-29 | 15146 | 2012-05-02 | 2026-05-27 |  |
| 6196 | TaiwanStockShareholding | MISMATCH | 5461 | 2004-05-21 | 2026-05-29 | 5456 | 2004-05-21 | 2026-05-22 |  |
| 6197 | TaiwanStockPriceAdj | MISMATCH | 5607 | 2003-08-04 | 2026-05-29 | 5602 | 2003-08-04 | 2026-05-22 |  |
| 6197 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15094 | 2012-05-02 | 2026-05-29 | 15084 | 2012-05-02 | 2026-05-27 |  |
| 6197 | TaiwanStockShareholding | MISMATCH | 5345 | 2004-11-08 | 2026-05-29 | 5340 | 2004-11-08 | 2026-05-22 |  |
| 6202 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 6202 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15172 | 2012-05-02 | 2026-05-29 | 15162 | 2012-05-02 | 2026-05-27 |  |
| 6202 | TaiwanStockShareholding | MISMATCH | 5373 | 2004-09-27 | 2026-05-29 | 5368 | 2004-09-27 | 2026-05-22 |  |
| 6207 | TaiwanStockPriceAdj | MISMATCH | 5690 | 2002-12-24 | 2026-05-29 | 5685 | 2002-12-24 | 2026-05-22 |  |
| 6207 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17038 | 2005-01-03 | 2026-05-29 | 17028 | 2005-01-03 | 2026-05-27 |  |
| 6207 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6209 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 6209 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15170 | 2012-05-02 | 2026-05-29 | 15160 | 2012-05-02 | 2026-05-27 |  |
| 6209 | TaiwanStockShareholding | MISMATCH | 5345 | 2004-11-08 | 2026-05-29 | 5340 | 2004-11-08 | 2026-05-22 |  |
| 6213 | TaiwanStockPriceAdj | MISMATCH | 5546 | 2003-08-04 | 2026-05-29 | 5541 | 2003-08-04 | 2026-05-22 |  |
| 6213 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17084 | 2005-01-03 | 2026-05-29 | 17074 | 2005-01-03 | 2026-05-27 |  |
| 6213 | TaiwanStockShareholding | MISMATCH | 4545 | 2008-01-18 | 2026-05-29 | 4540 | 2008-01-18 | 2026-05-22 |  |
| 6231 | TaiwanStockPriceAdj | MISMATCH | 5672 | 2003-01-24 | 2026-05-29 | 5667 | 2003-01-24 | 2026-05-22 |  |
| 6231 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17495 | 2005-07-08 | 2026-05-29 | 17485 | 2005-07-08 | 2026-05-27 |  |
| 6231 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6231 | TaiwanStockDividend | MISMATCH | 19 | 2008-08-20 | 2026-07-28 | 18 | 2008-08-20 | 2025-07-28 |  |
| 6235 | TaiwanStockPriceAdj | MISMATCH | 5604 | 2003-08-04 | 2026-05-29 | 5599 | 2003-08-04 | 2026-05-22 |  |
| 6235 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14926 | 2012-05-02 | 2026-05-29 | 14916 | 2012-05-02 | 2026-05-27 |  |
| 6235 | TaiwanStockShareholding | MISMATCH | 5396 | 2004-08-23 | 2026-05-29 | 5391 | 2004-08-23 | 2026-05-22 |  |
| 6239 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 6239 | TaiwanStockShareholding | MISMATCH | 5345 | 2004-11-08 | 2026-05-29 | 5340 | 2004-11-08 | 2026-05-22 |  |
| 6245 | TaiwanStockPriceAdj | MISMATCH | 5601 | 2003-05-20 | 2026-05-29 | 5596 | 2003-05-20 | 2026-05-22 |  |
| 6245 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17479 | 2005-01-04 | 2026-05-29 | 17469 | 2005-01-04 | 2026-05-27 |  |
| 6245 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6257 | TaiwanStockPriceAdj | MISMATCH | 5601 | 2003-08-26 | 2026-05-29 | 5596 | 2003-08-26 | 2026-05-22 |  |
| 6257 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 6265 | TaiwanStockPriceAdj | MISMATCH | 5490 | 2003-10-27 | 2026-05-29 | 5485 | 2003-10-27 | 2026-05-22 |  |
| 6265 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12027 | 2005-04-20 | 2026-05-29 | 12017 | 2005-04-20 | 2026-05-27 |  |
| 6265 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6271 | TaiwanStockPriceAdj | MISMATCH | 4751 | 2006-12-19 | 2026-05-29 | 4746 | 2006-12-19 | 2026-05-22 |  |
| 6271 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15239 | 2007-06-20 | 2026-05-29 | 15229 | 2007-06-20 | 2026-05-27 |  |
| 6271 | TaiwanStockShareholding | MISMATCH | 4590 | 2007-11-15 | 2026-05-29 | 4585 | 2007-11-15 | 2026-05-22 |  |
| 6271 | TaiwanStockDividend | MISMATCH | 24 | 2005-07-31 | 2026-06-16 | 23 | 2005-07-31 | 2025-06-21 |  |
| 6274 | TaiwanStockPriceAdj | MISMATCH | 5451 | 2003-12-19 | 2026-05-29 | 5446 | 2003-12-19 | 2026-05-22 |  |
| 6274 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6275 | TaiwanStockPriceAdj | MISMATCH | 5198 | 2004-12-24 | 2026-05-29 | 5193 | 2004-12-24 | 2026-05-22 |  |
| 6275 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13338 | 2005-02-22 | 2026-05-29 | 13328 | 2005-02-22 | 2026-05-27 |  |
| 6275 | TaiwanStockShareholding | MISMATCH | 5296 | 2004-12-22 | 2026-05-29 | 5291 | 2004-12-22 | 2026-05-22 |  |
| 6278 | TaiwanStockPriceAdj | MISMATCH | 5395 | 2004-03-15 | 2026-05-29 | 5390 | 2004-03-15 | 2026-05-22 |  |
| 6278 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18992 | 2005-03-17 | 2026-05-29 | 18982 | 2005-03-17 | 2026-05-27 |  |
| 6278 | TaiwanStockShareholding | MISMATCH | 3894 | 2010-08-23 | 2026-05-29 | 3889 | 2010-08-23 | 2026-05-22 |  |
| 6279 | TaiwanStockPriceAdj | MISMATCH | 5467 | 2003-11-27 | 2026-05-29 | 5462 | 2003-11-27 | 2026-05-22 |  |
| 6279 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18058 | 2005-01-07 | 2026-05-29 | 18048 | 2005-01-07 | 2026-05-27 |  |
| 6279 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6282 | TaiwanStockPriceAdj | MISMATCH | 5591 | 2003-09-09 | 2026-05-29 | 5586 | 2003-09-09 | 2026-05-22 |  |
| 6282 | TaiwanStockShareholding | MISMATCH | 5531 | 2004-02-12 | 2026-05-29 | 5526 | 2004-02-12 | 2026-05-22 |  |
| 6284 | TaiwanStockPriceAdj | MISMATCH | 5314 | 2004-06-30 | 2026-05-29 | 5309 | 2004-06-30 | 2026-05-22 |  |
| 6284 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18274 | 2005-01-05 | 2026-05-29 | 18264 | 2005-01-05 | 2026-05-27 |  |
| 6284 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6285 | TaiwanStockPriceAdj | MISMATCH | 5582 | 2003-09-23 | 2026-05-29 | 5577 | 2003-09-23 | 2026-05-22 |  |
| 6285 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 6285 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 6290 | TaiwanStockPriceAdj | MISMATCH | 5091 | 2005-06-06 | 2026-05-29 | 5086 | 2005-06-06 | 2026-05-22 |  |
| 6290 | TaiwanStockShareholding | MISMATCH | 5193 | 2005-06-02 | 2026-05-29 | 5188 | 2005-06-02 | 2026-05-22 |  |
| 6411 | TaiwanStockPriceAdj | MISMATCH | 3272 | 2012-12-26 | 2026-05-29 | 3267 | 2012-12-26 | 2026-05-22 |  |
| 6411 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13661 | 2014-03-11 | 2026-05-29 | 13651 | 2014-03-11 | 2026-05-27 |  |
| 6411 | TaiwanStockShareholding | MISMATCH | 3009 | 2014-03-10 | 2026-05-29 | 3004 | 2014-03-10 | 2026-05-22 |  |
| 6412 | TaiwanStockPriceAdj | MISMATCH | 3267 | 2013-01-04 | 2026-05-29 | 3262 | 2013-01-04 | 2026-05-22 |  |
| 6412 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14089 | 2013-01-29 | 2026-05-29 | 14079 | 2013-01-29 | 2026-05-27 |  |
| 6412 | TaiwanStockShareholding | MISMATCH | 3091 | 2012-03-29 | 2026-05-29 | 3086 | 2012-03-29 | 2026-05-22 |  |
| 6414 | TaiwanStockPriceAdj | MISMATCH | 3277 | 2012-12-19 | 2026-05-29 | 3272 | 2012-12-19 | 2026-05-22 |  |
| 6414 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14054 | 2013-01-04 | 2026-05-29 | 14044 | 2013-01-04 | 2026-05-27 |  |
| 6414 | TaiwanStockShareholding | MISMATCH | 2997 | 2012-03-29 | 2026-05-29 | 2992 | 2012-03-29 | 2026-05-22 |  |
| 6435 | TaiwanStockPriceAdj | MISMATCH | 3037 | 2013-12-13 | 2026-05-29 | 3032 | 2013-12-13 | 2026-05-22 |  |
| 6435 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11961 | 2013-12-25 | 2026-05-29 | 11951 | 2013-12-25 | 2026-05-27 |  |
| 6435 | TaiwanStockShareholding | MISMATCH | 2553 | 2016-01-06 | 2026-05-29 | 2548 | 2016-01-06 | 2026-05-22 |  |
| 6438 | TaiwanStockPriceAdj | MISMATCH | 3046 | 2013-11-26 | 2026-05-29 | 3041 | 2013-11-26 | 2026-05-22 |  |
| 6438 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11423 | 2013-11-25 | 2026-05-29 | 11413 | 2013-11-25 | 2026-05-27 |  |
| 6438 | TaiwanStockShareholding | MISMATCH | 2228 | 2017-04-24 | 2026-05-29 | 2223 | 2017-04-24 | 2026-05-22 |  |
| 6477 | TaiwanStockPriceAdj | MISMATCH | 2821 | 2014-10-29 | 2026-05-29 | 2816 | 2014-10-29 | 2026-05-22 |  |
| 6477 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11261 | 2016-02-05 | 2026-05-29 | 11251 | 2016-02-05 | 2026-05-27 |  |
| 6477 | TaiwanStockShareholding | MISMATCH | 2445 | 2012-03-29 | 2026-05-29 | 2440 | 2012-03-29 | 2026-05-22 |  |
| 6488 | TaiwanStockPriceAdj | MISMATCH | 2823 | 2014-10-29 | 2026-05-29 | 2818 | 2014-10-29 | 2026-05-22 |  |
| 6488 | TaiwanStockShareholding | MISMATCH | 2623 | 2015-09-24 | 2026-05-29 | 2618 | 2015-09-24 | 2026-05-22 |  |
| 6505 | TaiwanStockPriceAdj | MISMATCH | 5514 | 2003-12-29 | 2026-05-29 | 5509 | 2003-12-29 | 2026-05-22 |  |
| 6505 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15187 | 2012-05-02 | 2026-05-27 |  |
| 6505 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 6509 | TaiwanStockPriceAdj | MISMATCH | 5376 | 2004-04-13 | 2026-05-29 | 5371 | 2004-04-13 | 2026-05-22 |  |
| 6509 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17949 | 2005-02-18 | 2026-05-29 | 17939 | 2005-02-18 | 2026-05-27 |  |
| 6509 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 6510 | TaiwanStockPriceAdj | MISMATCH | 2764 | 2015-01-21 | 2026-05-29 | 2759 | 2015-01-21 | 2026-05-22 |  |
| 6510 | TaiwanStockShareholding | MISMATCH | 2503 | 2016-03-23 | 2026-05-29 | 2498 | 2016-03-23 | 2026-05-22 |  |
| 6531 | TaiwanStockPriceAdj | MISMATCH | 2665 | 2015-06-15 | 2026-05-29 | 2660 | 2015-06-15 | 2026-05-22 |  |
| 6531 | TaiwanStockShareholding | MISMATCH | 2459 | 2012-03-29 | 2026-05-29 | 2454 | 2012-03-29 | 2026-05-22 |  |
| 6532 | TaiwanStockPriceAdj | MISMATCH | 2683 | 2015-05-26 | 2026-05-29 | 2678 | 2015-05-26 | 2026-05-22 |  |
| 6532 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 10967 | 2015-06-09 | 2026-05-29 | 10957 | 2015-06-09 | 2026-05-27 |  |
| 6532 | TaiwanStockShareholding | MISMATCH | 2377 | 2016-09-23 | 2026-05-29 | 2372 | 2016-09-23 | 2026-05-22 |  |
| 8011 | TaiwanStockPriceAdj | MISMATCH | 4253 | 2003-08-04 | 2026-05-29 | 4248 | 2003-08-04 | 2026-05-22 |  |
| 8011 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15430 | 2010-03-15 | 2026-05-29 | 15425 | 2010-03-15 | 2026-05-28 |  |
| 8011 | TaiwanStockShareholding | MISMATCH | 3626 | 2011-09-16 | 2026-05-29 | 3621 | 2011-09-16 | 2026-05-22 |  |
| 8011 | TaiwanStockDividend | MISMATCH | 14 | 2010-09-06 | 2026-06-23 | 13 | 2010-09-06 | 2025-06-23 |  |
| 8016 | TaiwanStockPriceAdj | MISMATCH | 5515 | 2003-12-26 | 2026-05-29 | 5510 | 2003-12-26 | 2026-05-22 |  |
| 8016 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15191 | 2012-05-02 | 2026-05-29 | 15186 | 2012-05-02 | 2026-05-28 |  |
| 8016 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 8016 | TaiwanStockDividend | MISMATCH | 22 | 2005-06-08 | 2026-06-21 | 21 | 2005-06-08 | 2025-06-22 |  |
| 8021 | TaiwanStockPriceAdj | MISMATCH | 5191 | 2004-12-22 | 2026-05-29 | 5186 | 2004-12-22 | 2026-05-22 |  |
| 8021 | TaiwanStockShareholding | MISMATCH | 4545 | 2008-01-18 | 2026-05-29 | 4540 | 2008-01-18 | 2026-05-22 |  |
| 8039 | TaiwanStockPriceAdj | MISMATCH | 5448 | 2003-12-22 | 2026-05-29 | 5443 | 2003-12-22 | 2026-05-22 |  |
| 8039 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18448 | 2005-01-03 | 2026-05-29 | 18443 | 2005-01-03 | 2026-05-28 |  |
| 8039 | TaiwanStockShareholding | MISMATCH | 4065 | 2009-12-16 | 2026-05-29 | 4060 | 2009-12-16 | 2026-05-22 |  |
| 8043 | TaiwanStockPriceAdj | MISMATCH | 5480 | 2003-10-27 | 2026-05-29 | 5475 | 2003-10-27 | 2026-05-22 |  |
| 8043 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13163 | 2005-01-03 | 2026-05-29 | 13158 | 2005-01-03 | 2026-05-28 |  |
| 8043 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 8046 | TaiwanStockPriceAdj | MISMATCH | 4952 | 2006-04-10 | 2026-05-29 | 4947 | 2006-04-10 | 2026-05-22 |  |
| 8046 | TaiwanStockShareholding | MISMATCH | 4994 | 2006-04-06 | 2026-05-29 | 4989 | 2006-04-06 | 2026-05-22 |  |
| 8064 | TaiwanStockPriceAdj | MISMATCH | 4764 | 2006-09-26 | 2026-05-29 | 4759 | 2006-09-26 | 2026-05-22 |  |
| 8064 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17195 | 2006-09-25 | 2026-05-29 | 17190 | 2006-09-25 | 2026-05-28 |  |
| 8064 | TaiwanStockShareholding | MISMATCH | 4870 | 2006-09-22 | 2026-05-29 | 4865 | 2006-09-22 | 2026-05-22 |  |
| 8069 | TaiwanStockPriceAdj | MISMATCH | 5385 | 2004-03-31 | 2026-05-29 | 5380 | 2004-03-31 | 2026-05-22 |  |
| 8069 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19980 | 2005-01-06 | 2026-05-29 | 19975 | 2005-01-06 | 2026-05-28 |  |
| 8069 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 8076 | TaiwanStockPriceAdj | MISMATCH | 5454 | 2003-12-16 | 2026-05-29 | 5449 | 2003-12-16 | 2026-05-22 |  |
| 8076 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 19869 | 2005-01-10 | 2026-05-29 | 19864 | 2005-01-10 | 2026-05-28 |  |
| 8076 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 8081 | TaiwanStockPriceAdj | MISMATCH | 5277 | 2004-08-31 | 2026-05-29 | 5272 | 2004-08-31 | 2026-05-22 |  |
| 8081 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18102 | 2005-01-04 | 2026-05-29 | 18097 | 2005-01-04 | 2026-05-28 |  |
| 8081 | TaiwanStockShareholding | MISMATCH | 4309 | 2008-12-29 | 2026-05-29 | 4304 | 2008-12-29 | 2026-05-22 |  |
| 8091 | TaiwanStockPriceAdj | MISMATCH | 5318 | 2004-06-25 | 2026-05-29 | 5313 | 2004-06-25 | 2026-05-22 |  |
| 8091 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 16490 | 2005-01-17 | 2026-05-29 | 16485 | 2005-01-17 | 2026-05-28 |  |
| 8091 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 8091 | TaiwanStockDividend | MISMATCH | 22 | 2005-06-29 | 2026-06-22 | 21 | 2005-06-29 | 2025-04-22 |  |
| 8096 | TaiwanStockPriceAdj | MISMATCH | 5309 | 2004-07-16 | 2026-05-29 | 5304 | 2004-07-16 | 2026-05-22 |  |
| 8096 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18169 | 2005-01-04 | 2026-05-29 | 18164 | 2005-01-04 | 2026-05-28 |  |
| 8096 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 8110 | TaiwanStockPriceAdj | MISMATCH | 4785 | 2006-08-24 | 2026-05-29 | 4780 | 2006-08-24 | 2026-05-22 |  |
| 8110 | TaiwanStockShareholding | MISMATCH | 4602 | 2007-10-30 | 2026-05-29 | 4597 | 2007-10-30 | 2026-05-22 |  |
| 8111 | TaiwanStockPriceAdj | MISMATCH | 5413 | 2004-02-10 | 2026-05-29 | 5408 | 2004-02-10 | 2026-05-22 |  |
| 8111 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15687 | 2005-01-03 | 2026-05-29 | 15682 | 2005-01-03 | 2026-05-28 |  |
| 8111 | TaiwanStockShareholding | MISMATCH | 5389 | 2004-08-04 | 2026-05-29 | 5384 | 2004-08-04 | 2026-05-22 |  |
| 8112 | TaiwanStockPriceAdj | MISMATCH | 5419 | 2004-02-10 | 2026-05-29 | 5414 | 2004-02-10 | 2026-05-22 |  |
| 8112 | TaiwanStockShareholding | MISMATCH | 4559 | 2007-12-28 | 2026-05-29 | 4554 | 2007-12-28 | 2026-05-22 |  |
| 8150 | TaiwanStockPriceAdj | MISMATCH | 3195 | 2013-04-22 | 2026-05-29 | 3190 | 2013-04-22 | 2026-05-22 |  |
| 8150 | TaiwanStockShareholding | MISMATCH | 2987 | 2014-04-10 | 2026-05-29 | 2982 | 2014-04-10 | 2026-05-22 |  |
| 8150 | TaiwanStockDividend | MISMATCH | 14 | 2013-07-09 | 2026-07-05 | 13 | 2013-07-09 | 2025-07-05 |  |
| 8155 | TaiwanStockPriceAdj | MISMATCH | 3676 | 2011-05-23 | 2026-05-29 | 3671 | 2011-05-23 | 2026-05-22 |  |
| 8155 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13465 | 2012-12-24 | 2026-05-29 | 13460 | 2012-12-24 | 2026-05-28 |  |
| 8155 | TaiwanStockShareholding | MISMATCH | 3310 | 2012-12-17 | 2026-05-29 | 3305 | 2012-12-17 | 2026-05-22 |  |
| 8163 | TaiwanStockPriceAdj | MISMATCH | 4765 | 2006-12-19 | 2026-05-29 | 4760 | 2006-12-19 | 2026-05-22 |  |
| 8163 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15393 | 2007-06-20 | 2026-05-29 | 15388 | 2007-06-20 | 2026-05-28 |  |
| 8163 | TaiwanStockShareholding | MISMATCH | 4582 | 2007-11-27 | 2026-05-29 | 4577 | 2007-11-27 | 2026-05-22 |  |
| 8163 | TaiwanStockDividend | MISMATCH | 21 | 2006-08-27 | 2026-06-25 | 20 | 2006-08-27 | 2025-06-28 |  |
| 8183 | TaiwanStockPriceAdj | MISMATCH | 5136 | 2005-03-31 | 2026-05-29 | 5131 | 2005-03-31 | 2026-05-22 |  |
| 8183 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14256 | 2005-04-01 | 2026-05-29 | 14251 | 2005-04-01 | 2026-05-28 |  |
| 8183 | TaiwanStockShareholding | MISMATCH | 5235 | 2005-03-29 | 2026-05-29 | 5230 | 2005-03-29 | 2026-05-22 |  |
| 8210 | TaiwanStockPriceAdj | MISMATCH | 5114 | 2005-04-29 | 2026-05-29 | 5109 | 2005-04-29 | 2026-05-22 |  |
| 8210 | TaiwanStockShareholding | MISMATCH | 3574 | 2011-11-30 | 2026-05-29 | 3569 | 2011-11-30 | 2026-05-22 |  |
| 8213 | TaiwanStockPriceAdj | MISMATCH | 4762 | 2006-12-19 | 2026-05-29 | 4757 | 2006-12-19 | 2026-05-22 |  |
| 8213 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15690 | 2007-06-25 | 2026-05-29 | 15685 | 2007-06-25 | 2026-05-28 |  |
| 8213 | TaiwanStockShareholding | MISMATCH | 4059 | 2009-12-24 | 2026-05-29 | 4054 | 2009-12-24 | 2026-05-22 |  |
| 8213 | TaiwanStockDividend | MISMATCH | 27 | 2005-08-05 | 2026-07-26 | 26 | 2005-08-05 | 2025-08-08 |  |
| 8255 | TaiwanStockPriceAdj | MISMATCH | 4871 | 2006-04-26 | 2026-05-29 | 4866 | 2006-04-26 | 2026-05-22 |  |
| 8255 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18688 | 2006-04-25 | 2026-05-29 | 18683 | 2006-04-25 | 2026-05-28 |  |
| 8255 | TaiwanStockShareholding | MISMATCH | 4977 | 2006-04-24 | 2026-05-29 | 4972 | 2006-04-24 | 2026-05-22 |  |
| 8261 | TaiwanStockPriceAdj | MISMATCH | 5361 | 2004-04-16 | 2026-05-29 | 5356 | 2004-04-16 | 2026-05-22 |  |
| 8261 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17838 | 2005-01-03 | 2026-05-29 | 17833 | 2005-01-03 | 2026-05-28 |  |
| 8261 | TaiwanStockShareholding | MISMATCH | 4069 | 2009-12-10 | 2026-05-29 | 4064 | 2009-12-10 | 2026-05-22 |  |
| 8271 | TaiwanStockPriceAdj | MISMATCH | 4752 | 2006-12-19 | 2026-05-29 | 4747 | 2006-12-19 | 2026-05-22 |  |
| 8271 | TaiwanStockShareholding | MISMATCH | 3804 | 2010-12-28 | 2026-05-29 | 3799 | 2010-12-28 | 2026-05-22 |  |
| 8299 | TaiwanStockPriceAdj | MISMATCH | 5211 | 2004-12-07 | 2026-05-29 | 5206 | 2004-12-07 | 2026-05-22 |  |
| 8299 | TaiwanStockShareholding | MISMATCH | 5309 | 2004-12-03 | 2026-05-29 | 5304 | 2004-12-03 | 2026-05-22 |  |
| 8358 | TaiwanStockPriceAdj | MISMATCH | 4552 | 2007-11-08 | 2026-05-29 | 4547 | 2007-11-08 | 2026-05-22 |  |
| 8358 | TaiwanStockShareholding | MISMATCH | 3871 | 2010-09-24 | 2026-05-29 | 3866 | 2010-09-24 | 2026-05-22 |  |
| 8383 | TaiwanStockPriceAdj | MISMATCH | 5270 | 2004-09-13 | 2026-05-29 | 5265 | 2004-09-13 | 2026-05-22 |  |
| 8383 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17228 | 2005-02-18 | 2026-05-29 | 17223 | 2005-02-18 | 2026-05-28 |  |
| 8383 | TaiwanStockShareholding | MISMATCH | 5366 | 2004-09-09 | 2026-05-29 | 5361 | 2004-09-09 | 2026-05-22 |  |
| 8436 | TaiwanStockPriceAdj | MISMATCH | 3491 | 2012-02-16 | 2026-05-29 | 3486 | 2012-02-16 | 2026-05-22 |  |
| 8436 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14371 | 2012-03-06 | 2026-05-29 | 14366 | 2012-03-06 | 2026-05-28 |  |
| 8436 | TaiwanStockShareholding | MISMATCH | 3129 | 2013-09-11 | 2026-05-29 | 3124 | 2013-09-11 | 2026-05-22 |  |
| 8462 | TaiwanStockPriceAdj | MISMATCH | 2862 | 2014-09-01 | 2026-05-29 | 2857 | 2014-09-01 | 2026-05-22 |  |
| 8462 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 11861 | 2015-12-10 | 2026-05-29 | 11856 | 2015-12-10 | 2026-05-28 |  |
| 8462 | TaiwanStockShareholding | MISMATCH | 1768 | 2019-03-14 | 2026-05-29 | 1763 | 2019-03-14 | 2026-05-22 |  |
| 8464 | TaiwanStockPriceAdj | MISMATCH | 2775 | 2015-01-07 | 2026-05-29 | 2770 | 2015-01-07 | 2026-05-22 |  |
| 8464 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 12411 | 2015-02-03 | 2026-05-29 | 12406 | 2015-02-03 | 2026-05-28 |  |
| 8464 | TaiwanStockShareholding | MISMATCH | 2565 | 2012-03-29 | 2026-05-29 | 2560 | 2012-03-29 | 2026-05-22 |  |
| 8926 | TaiwanStockPriceAdj | MISMATCH | 5617 | 2003-08-04 | 2026-05-29 | 5612 | 2003-08-04 | 2026-05-22 |  |
| 8926 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15176 | 2012-05-02 | 2026-05-29 | 15171 | 2012-05-02 | 2026-05-28 |  |
| 8926 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 8927 | TaiwanStockPriceAdj | MISMATCH | 6226 | 2000-11-03 | 2026-05-29 | 6221 | 2000-11-03 | 2026-05-22 |  |
| 8927 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 9583 | 2006-12-14 | 2026-05-29 | 9578 | 2006-12-14 | 2026-05-28 |  |
| 8927 | TaiwanStockShareholding | MISMATCH | 5388 | 2004-08-04 | 2026-05-29 | 5383 | 2004-08-04 | 2026-05-22 |  |
| 8936 | TaiwanStockPriceAdj | MISMATCH | 5771 | 2002-09-10 | 2026-05-29 | 5766 | 2002-09-10 | 2026-05-22 |  |
| 8936 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 17564 | 2005-01-03 | 2026-05-29 | 17559 | 2005-01-03 | 2026-05-28 |  |
| 8936 | TaiwanStockShareholding | MISMATCH | 5388 | 2004-08-04 | 2026-05-29 | 5383 | 2004-08-04 | 2026-05-22 |  |
| 8938 | TaiwanStockPriceAdj | MISMATCH | 5712 | 2002-12-03 | 2026-05-29 | 5707 | 2002-12-03 | 2026-05-22 |  |
| 8938 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 18381 | 2005-01-10 | 2026-05-29 | 18376 | 2005-01-10 | 2026-05-28 |  |
| 8938 | TaiwanStockShareholding | MISMATCH | 5388 | 2004-08-04 | 2026-05-29 | 5383 | 2004-08-04 | 2026-05-22 |  |
| 8996 | TaiwanStockPriceAdj | MISMATCH | 4823 | 2006-07-03 | 2026-05-29 | 4818 | 2006-07-03 | 2026-05-22 |  |
| 8996 | TaiwanStockShareholding | MISMATCH | 3026 | 2012-03-29 | 2026-05-29 | 3021 | 2012-03-29 | 2026-05-22 |  |
| 9802 | TaiwanStockPriceAdj | MISMATCH | 3323 | 2012-10-19 | 2026-05-29 | 3318 | 2012-10-19 | 2026-05-22 |  |
| 9802 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14806 | 2012-10-18 | 2026-05-29 | 14801 | 2012-10-18 | 2026-05-28 |  |
| 9802 | TaiwanStockShareholding | MISMATCH | 3354 | 2012-03-29 | 2026-05-29 | 3349 | 2012-03-29 | 2026-05-22 |  |
| 9904 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 9904 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15192 | 2012-05-02 | 2026-05-28 |  |
| 9904 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9907 | TaiwanStockPriceAdj | MISMATCH | 8772 | 1992-01-06 | 2026-05-29 | 8767 | 1992-01-06 | 2026-05-22 |  |
| 9907 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15180 | 2012-05-02 | 2026-05-29 | 15175 | 2012-05-02 | 2026-05-28 |  |
| 9907 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9910 | TaiwanStockPriceAdj | MISMATCH | 8740 | 1992-02-19 | 2026-05-29 | 8735 | 1992-02-19 | 2026-05-22 |  |
| 9910 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15192 | 2012-05-02 | 2026-05-28 |  |
| 9910 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9910 | TaiwanStockDividend | MISMATCH | 22 | 2005-07-12 | 2026-06-21 | 21 | 2005-07-12 | 2025-06-22 |  |
| 9914 | TaiwanStockPriceAdj | MISMATCH | 8559 | 1992-10-01 | 2026-05-29 | 8554 | 1992-10-01 | 2026-05-22 |  |
| 9914 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15192 | 2012-05-02 | 2026-05-28 |  |
| 9914 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9921 | TaiwanStockPriceAdj | MISMATCH | 7922 | 1994-12-30 | 2026-05-29 | 7917 | 1994-12-30 | 2026-05-22 |  |
| 9921 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15184 | 2012-05-02 | 2026-05-29 | 15179 | 2012-05-02 | 2026-05-28 |  |
| 9921 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9933 | TaiwanStockPriceAdj | MISMATCH | 6272 | 1999-01-06 | 2026-05-29 | 6267 | 1999-01-06 | 2026-05-22 |  |
| 9933 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15187 | 2012-05-02 | 2026-05-29 | 15182 | 2012-05-02 | 2026-05-28 |  |
| 9933 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9938 | TaiwanStockPriceAdj | MISMATCH | 6243 | 2001-01-15 | 2026-05-29 | 6238 | 2001-01-15 | 2026-05-22 |  |
| 9938 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15189 | 2012-05-02 | 2026-05-29 | 15184 | 2012-05-02 | 2026-05-28 |  |
| 9938 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9939 | TaiwanStockPriceAdj | MISMATCH | 6215 | 2001-03-05 | 2026-05-29 | 6210 | 2001-03-05 | 2026-05-22 |  |
| 9939 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15197 | 2012-05-02 | 2026-05-29 | 15192 | 2012-05-02 | 2026-05-28 |  |
| 9939 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9941 | TaiwanStockPriceAdj | MISMATCH | 6079 | 2001-09-20 | 2026-05-29 | 6074 | 2001-09-20 | 2026-05-22 |  |
| 9941 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15181 | 2012-05-02 | 2026-05-29 | 15176 | 2012-05-02 | 2026-05-28 |  |
| 9941 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9945 | TaiwanStockPriceAdj | MISMATCH | 5772 | 1992-05-01 | 2026-05-29 | 5767 | 1992-05-01 | 2026-05-22 |  |
| 9945 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 15127 | 2012-05-02 | 2026-05-29 | 15122 | 2012-05-02 | 2026-05-28 |  |
| 9945 | TaiwanStockShareholding | MISMATCH | 5532 | 2004-02-12 | 2026-05-29 | 5527 | 2004-02-12 | 2026-05-22 |  |
| 9955 | TaiwanStockPriceAdj | MISMATCH | 5441 | 2003-12-31 | 2026-05-29 | 5436 | 2003-12-31 | 2026-05-22 |  |
| 9955 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 14705 | 2005-01-04 | 2026-05-29 | 14700 | 2005-01-04 | 2026-05-28 |  |
| 9955 | TaiwanStockShareholding | MISMATCH | 4545 | 2008-01-18 | 2026-05-29 | 4540 | 2008-01-18 | 2026-05-22 |  |
| 9958 | TaiwanStockPriceAdj | MISMATCH | 4758 | 2006-12-13 | 2026-05-29 | 4753 | 2006-12-13 | 2026-05-22 |  |
| 9958 | TaiwanStockInstitutionalInvestorsBuySell | MISMATCH | 13968 | 2007-12-25 | 2026-05-29 | 13963 | 2007-12-25 | 2026-05-28 |  |
| 9958 | TaiwanStockShareholding | MISMATCH | 4513 | 2008-03-11 | 2026-05-29 | 4508 | 2008-03-11 | 2026-05-22 |  |

## Targeted Backfill Commands

```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1102 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1102 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1102 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1210 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1210 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1210 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1215 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1215 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1215 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1216 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1216 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1216 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1227 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1227 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1227 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1229 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1229 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1229 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1303 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1303 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1303 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1319 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1319 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1319 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1326 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1326 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1326 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1402 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1402 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1402 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1434 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1434 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1434 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1476 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1476 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1476 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1477 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1477 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1477 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1504 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1504 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1504 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1513 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1513 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1513 --dataset TaiwanStockDividend --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1515 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1515 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1522 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1522 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1522 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1532 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1532 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1532 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1533 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1533 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1565 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1565 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1565 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1568 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1568 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1568 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1582 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1582 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1582 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1590 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1590 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1597 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1597 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1597 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1605 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1605 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1605 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1605 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1608 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1608 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1608 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1609 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1609 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1609 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1611 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1611 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1611 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1612 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1612 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1612 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1618 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1618 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1618 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1702 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1702 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1702 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1712 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1712 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1712 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1714 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1714 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1714 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1717 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1717 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1717 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1717 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1722 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1722 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1722 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1736 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1736 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1736 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1752 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1752 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1752 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1773 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1773 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1773 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1773 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1784 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1784 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1789 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1789 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1789 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1795 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1795 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1806 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1806 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1806 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1809 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1809 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1809 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1810 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1810 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1810 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1810 --dataset TaiwanStockDividend --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1904 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1904 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1907 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1907 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 1907 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2006 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2006 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2006 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2020 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2020 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2020 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2027 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2027 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2027 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2031 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2031 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2059 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2059 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2101 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2101 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2101 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2103 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2103 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2103 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2105 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2105 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2105 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2106 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2106 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2106 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2108 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2108 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2108 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2201 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2201 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2201 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2204 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2204 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2204 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2206 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2206 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2206 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2233 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2233 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2305 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2305 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2305 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2308 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2308 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2308 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2312 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2312 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2312 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2313 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2313 --dataset TaiwanStockShareholding --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2323 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2323 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2323 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2324 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2324 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2327 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2328 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2328 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2331 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2331 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2331 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2344 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2344 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2347 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2347 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2347 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2345 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2345 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2348 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2348 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2348 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2351 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2351 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2351 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2352 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2352 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2352 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2353 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2353 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2353 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2354 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2354 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2355 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2355 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2363 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2363 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2359 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2359 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2359 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2368 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2368 --dataset TaiwanStockShareholding --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2375 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2377 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2377 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2382 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2382 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2385 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2385 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2385 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2393 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2393 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2393 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2395 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2395 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2399 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2399 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2409 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2409 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2413 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2413 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2417 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2417 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2417 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2421 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2421 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2421 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2427 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2427 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2427 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2428 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2428 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2428 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2436 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2436 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2436 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2439 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2439 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2439 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2441 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2441 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2442 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2442 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2453 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2453 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2453 --dataset TaiwanStockShareholding --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2458 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2458 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2467 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2467 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2472 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2472 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2472 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2474 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2474 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2484 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2484 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2489 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2489 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2497 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2497 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2498 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2498 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2498 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2504 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2504 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2504 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2515 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2515 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2515 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2520 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2520 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2520 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2535 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2535 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2535 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2542 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2542 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2542 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2545 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2545 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2545 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2603 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2603 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2603 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2605 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2605 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2605 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2606 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2606 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2606 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2606 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2610 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2610 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2610 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2612 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2612 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2612 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2615 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2615 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2615 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2617 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2617 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2617 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2618 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2618 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2618 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2634 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2634 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2634 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2636 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2636 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2636 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2637 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2637 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2637 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2637 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2727 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2727 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2727 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2731 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2731 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2731 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2731 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2732 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2732 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2732 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2903 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2903 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2903 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2915 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2915 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 2915 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3003 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3003 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3003 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3004 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3004 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3004 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3005 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3005 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3006 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3008 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3008 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3013 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3013 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3013 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3014 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3014 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3014 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3016 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3016 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3022 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3022 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3022 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3023 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3023 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3023 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3025 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3025 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3025 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3029 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3029 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3029 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3030 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3030 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3030 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3031 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3031 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3031 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3033 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3033 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3033 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3034 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3034 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3034 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3036 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3036 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3036 --dataset TaiwanStockDividend --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3038 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3038 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3042 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3042 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3042 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3042 --dataset TaiwanStockDividend --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3048 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3048 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3059 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3059 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3059 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3071 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3071 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3071 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3078 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3078 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3078 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3083 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3083 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3083 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3088 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3088 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3088 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3090 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3094 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3094 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3094 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3094 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3105 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3105 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3105 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3141 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3141 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3141 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3169 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3169 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3169 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3169 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3189 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3189 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3209 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3209 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3217 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3217 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3217 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3218 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3218 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3218 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3227 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3227 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3227 --dataset TaiwanStockShareholding --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3257 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3289 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3289 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3289 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3290 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3290 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3290 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3293 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3293 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3293 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3312 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3312 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3312 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3317 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3317 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3317 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3324 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3324 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3338 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3338 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3338 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3356 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3356 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3356 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3376 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3376 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3376 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3402 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3402 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3413 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3413 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3441 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3441 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3441 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3441 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3444 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3444 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3455 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3455 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3455 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3479 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3479 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3479 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3481 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3481 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3481 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3483 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3483 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3483 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3484 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3484 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3484 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3498 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3498 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3526 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3535 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3535 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3535 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3540 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3540 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3540 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3540 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3552 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3552 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3563 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3580 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3580 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3580 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3583 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3583 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3583 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3587 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3587 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3587 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3594 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3594 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3594 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3596 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3596 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3596 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3605 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3605 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3605 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3607 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3607 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3607 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3615 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3615 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3615 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3622 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3622 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3622 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3624 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3624 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3624 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3630 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3630 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3630 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3645 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3645 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3665 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3665 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3673 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3673 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3687 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3687 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3689 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3689 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3689 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3693 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3693 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3693 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3694 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3694 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3703 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3703 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3704 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 3704 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4105 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4105 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4114 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4114 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4114 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4119 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4119 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4119 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4306 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4306 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4306 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4551 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4551 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4551 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4721 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4721 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4746 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4746 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4746 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4763 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4763 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4763 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4912 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4912 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4912 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4915 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4915 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4915 --dataset TaiwanStockShareholding --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4938 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4938 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4952 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4952 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4952 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4952 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4953 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4953 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4953 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4958 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4958 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4958 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4966 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4966 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4966 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4968 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4968 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4968 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4973 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4973 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4973 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4977 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 4977 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5009 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5009 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5203 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5203 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5203 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5243 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5243 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5243 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5258 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5258 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5258 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5285 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5285 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5285 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5269 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5269 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5289 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5289 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5347 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5347 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5388 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5388 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5388 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5371 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5371 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5371 --dataset TaiwanStockShareholding --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5434 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5457 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5457 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5471 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5471 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5471 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5478 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5478 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5512 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5512 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5512 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5607 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5607 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5607 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5608 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5608 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5608 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5706 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5706 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5706 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5871 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5871 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 5871 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6118 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6118 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6118 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6104 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6104 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6104 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6121 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6121 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6121 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6125 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6125 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6125 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6126 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6126 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6139 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6139 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6143 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6143 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6143 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6147 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6147 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6147 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6147 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6150 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6150 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6150 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6161 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6161 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6161 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6163 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6163 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6163 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6166 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6166 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6166 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6168 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6168 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6168 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6170 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6170 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6170 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6173 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6173 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6173 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6175 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6175 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6175 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6176 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6176 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6176 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6177 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6177 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6177 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6187 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6187 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6187 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6188 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6188 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6188 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6190 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6190 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6196 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6196 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6197 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6197 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6197 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6202 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6202 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6202 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6207 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6207 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6207 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6209 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6209 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6209 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6213 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6213 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6213 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6231 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6231 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6231 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6231 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6235 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6235 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6235 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6239 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6239 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6245 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6245 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6265 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6265 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6265 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6271 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6271 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6271 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6271 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6274 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6274 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6275 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6275 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6275 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6278 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6278 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6278 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6279 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6279 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6279 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6282 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6282 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6284 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6284 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6284 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6285 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6285 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6411 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6411 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6412 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6412 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6412 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6414 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6414 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6414 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6435 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6435 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6435 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6438 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6438 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6438 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6477 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6477 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6505 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6505 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6505 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6509 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6509 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6531 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6531 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6532 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6532 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 6532 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8011 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8011 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8011 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8011 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8016 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8016 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8016 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8016 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8021 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8021 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8039 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8039 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8039 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8043 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8043 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8064 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8064 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8064 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8069 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8069 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8069 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8076 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8076 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8076 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8081 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8081 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8081 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8091 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8091 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8091 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8091 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8096 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8096 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8111 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8111 --dataset TaiwanStockShareholding --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8150 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8155 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8155 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8155 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8163 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8163 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8163 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8163 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8183 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8183 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8213 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8213 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8213 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8255 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8255 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8255 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8261 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8261 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8261 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8271 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8271 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8299 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8299 --dataset TaiwanStockShareholding --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8383 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8383 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8436 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8436 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8436 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8462 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8462 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8462 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8464 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8464 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8464 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8926 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8926 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8926 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8927 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8927 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8927 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8936 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8936 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8936 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8938 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 8938 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
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
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9802 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9802 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9802 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9904 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9904 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9904 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9907 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9907 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9907 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9910 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9910 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9910 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9910 --dataset TaiwanStockDividend --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9914 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9914 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9914 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9921 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9921 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9921 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9933 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9933 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9933 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9938 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9938 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9938 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9939 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9939 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9939 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9941 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9941 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9941 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9945 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9945 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9945 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9955 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9955 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9955 --dataset TaiwanStockShareholding --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9958 --dataset TaiwanStockPriceAdj --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9958 --dataset TaiwanStockInstitutionalInvestorsBuySell --strict-source-history
```
```bash
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id 9958 --dataset TaiwanStockShareholding --strict-source-history
```

## FRED Valid Observation Alignment

| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |
|---|---|---:|---|---|---:|---|---|
| DFF | OK | 26265 | 1954-07-01 | 2026-05-28 | 26265 | 1954-07-01 | 2026-05-28 |
| UNRATE | OK | 939 | 1948-01-01 | 2026-04-01 | 939 | 1948-01-01 | 2026-04-01 |
| T10Y2Y | TIME_DRIFT_OK | 12495 | 1976-06-01 | 2026-05-29 | 12494 | 1976-06-01 | 2026-05-28 |
| VIXCLS | OK | 9196 | 1990-01-02 | 2026-05-28 | 9196 | 1990-01-02 | 2026-05-28 |
