# Quantum Finance schema 後驗收稽核報告

- **時間**: 2026-05-25 11:08:49
- **憲章**: 系統架構大憲章_v6.0.0.md
- **稽核工具**: audit_supply_chain v1.19
- **schema 基準**: data_schema v2.16
- **判定結果**: **FAILED** (PASS=28, WARN=0, FAIL=1)

## 稽核明細

| 來源層 | 項目 | 狀態 | 詳細 |
| :--- | :--- | :--- | :--- |
| API-Contract | TaiwanStockPrice | ✅ PASS | ✅ [API-PASS] TaiwanStockPrice - 10 columns matched |
| API-Contract | TaiwanStockPriceAdj | ❌ FAILED | ❌ [API-FAILED] TaiwanStockPriceAdj - HTTPError: 400 Client Error: Bad Request for url: https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPriceAdj&start_date=2024-05-01&data_id=2330&token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidHNhaXRzYW5nY2hpIiwiZW1haWwiOiJ0c2FpdHNhbmdjaGlAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6NX0.c7sOZ2QTupouC_X1SRgTUueBvI6uDMsptimIZl3EGVA |
| API-Contract | TaiwanStockPER | ✅ PASS | ✅ [API-PASS] TaiwanStockPER - 5 columns matched |
| API-Contract | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | ✅ [API-PASS] TaiwanStockInstitutionalInvestorsBuySell - 5 columns matched |
| API-Contract | TaiwanStockMarginPurchaseShortSale | ✅ PASS | ✅ [API-PASS] TaiwanStockMarginPurchaseShortSale - 16 columns matched |
| API-Contract | TaiwanStockShareholding | ✅ PASS | ✅ [API-PASS] TaiwanStockShareholding - 13 columns matched |
| API-Contract | TaiwanStockFinancialStatements | ✅ PASS | ✅ [API-PASS] TaiwanStockFinancialStatements - 5 columns matched |
| API-Contract | TaiwanStockMonthRevenue | ✅ PASS | ✅ [API-PASS] TaiwanStockMonthRevenue - 7 columns matched |
| API-Contract | TaiwanStockDividend | ✅ PASS | ✅ [API-PASS] TaiwanStockDividend - 22 columns matched |
| API-Contract | TaiwanStockInfo | ✅ PASS | ✅ [API-PASS] TaiwanStockInfo - 5 columns matched |
| API-Contract | FredData | ✅ PASS | ✅ [API-PASS] FredData - 4+1 derived columns matched |
| DB-Schema | pipeline_execution_log | ✅ PASS | 9 columns matched; rows=36 |
| DB-Schema | data_audit_log | ✅ PASS | 7 columns matched; rows=25647 |
| DB-Schema | TaiwanStockPrice | ✅ PASS | 10 columns matched; rows=10485300 |
| DB-Schema | TaiwanStockPriceAdj | ✅ PASS | 10 columns matched; rows=10481069 |
| DB-Schema | TaiwanStockPER | ✅ PASS | 5 columns matched; rows=7328884 |
| DB-Schema | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | 5 columns matched; rows=24963205 |
| DB-Schema | TaiwanStockMarginPurchaseShortSale | ✅ PASS | 16 columns matched; rows=7696032 |
| DB-Schema | TaiwanStockShareholding | ✅ PASS | 13 columns matched; rows=8353006 |
| DB-Schema | TaiwanStockFinancialStatements | ✅ PASS | 5 columns matched; rows=2656263 |
| DB-Schema | TaiwanStockMonthRevenue | ✅ PASS | 7 columns matched; rows=459383 |
| DB-Schema | TaiwanStockDividend | ✅ PASS | 22 columns matched; rows=29262 |
| DB-Schema | FredData | ✅ PASS | 5 columns matched; rows=48876 |
| DB-Schema | TaiwanStockInfo | ✅ PASS | 5 columns matched; rows=2799 |
| DB-FRED | completeness | ✅ PASS | series counts={'DFF': 26257, 'T10Y2Y': 12490, 'UNRATE': 939, 'VIXCLS': 9190} |
| Freshness | FRED/DFF | ✅ PASS | latest=2026-05-20, age=5d |
| Freshness | FRED/UNRATE | ✅ PASS | latest=2026-04-01, age=54d |
| Freshness | FRED/T10Y2Y | ✅ PASS | latest=2026-05-21, age=4d |
| Freshness | FRED/VIXCLS | ✅ PASS | latest=2026-05-20, age=5d |
