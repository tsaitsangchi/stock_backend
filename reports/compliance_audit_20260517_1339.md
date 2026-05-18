# Quantum Finance schema 後驗收稽核報告

- **時間**: 2026-05-17 13:39:44
- **憲章**: 系統架構大憲章_v5.4.22.md
- **稽核工具**: audit_supply_chain v1.18
- **schema 基準**: data_schema v2.11
- **判定結果**: **PERFECT** (PASS=33, WARN=0, FAIL=0)

## 稽核明細

| 來源層 | 項目 | 狀態 | 詳細 |
| :--- | :--- | :--- | :--- |
| API-Contract | TaiwanStockPrice | ✅ PASS | ✅ [API-PASS] TaiwanStockPrice - 10 columns matched |
| API-Contract | TaiwanStockPriceAdj | ✅ PASS | ✅ [API-PASS] TaiwanStockPriceAdj - 10 columns matched |
| API-Contract | TaiwanStockPER | ✅ PASS | ✅ [API-PASS] TaiwanStockPER - 5 columns matched |
| API-Contract | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | ✅ [API-PASS] TaiwanStockInstitutionalInvestorsBuySell - 5 columns matched |
| API-Contract | TaiwanStockMarginPurchaseShortSale | ✅ PASS | ✅ [API-PASS] TaiwanStockMarginPurchaseShortSale - 16 columns matched |
| API-Contract | TaiwanStockShareholding | ✅ PASS | ✅ [API-PASS] TaiwanStockShareholding - 13 columns matched |
| API-Contract | TaiwanStockFinancialStatements | ✅ PASS | ✅ [API-PASS] TaiwanStockFinancialStatements - 5 columns matched |
| API-Contract | TaiwanStockMonthRevenue | ✅ PASS | ✅ [API-PASS] TaiwanStockMonthRevenue - 7 columns matched |
| API-Contract | TaiwanStockDividend | ✅ PASS | ✅ [API-PASS] TaiwanStockDividend - 22 columns matched |
| API-Contract | TaiwanStockInfo | ✅ PASS | ✅ [API-PASS] TaiwanStockInfo - 5 columns matched |
| API-Contract | FredData | ✅ PASS | ✅ [API-PASS] FredData - 4+1 derived columns matched |
| DB-Schema | pipeline_execution_log | ✅ PASS | 9 columns matched; rows=12 |
| DB-Schema | data_audit_log | ✅ PASS | 7 columns matched; rows=1357 |
| DB-Schema | TaiwanStockPrice | ✅ PASS | 10 columns matched; rows=69885 |
| DB-Schema | TaiwanStockPriceAdj | ✅ PASS | 10 columns matched; rows=69885 |
| DB-Schema | TaiwanStockPER | ✅ PASS | 5 columns matched; rows=67135 |
| DB-Schema | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | 5 columns matched; rows=326165 |
| DB-Schema | TaiwanStockMarginPurchaseShortSale | ✅ PASS | 16 columns matched; rows=63150 |
| DB-Schema | TaiwanStockShareholding | ✅ PASS | 13 columns matched; rows=67691 |
| DB-Schema | TaiwanStockFinancialStatements | ✅ PASS | 5 columns matched; rows=17423 |
| DB-Schema | TaiwanStockMonthRevenue | ✅ PASS | 7 columns matched; rows=3369 |
| DB-Schema | TaiwanStockDividend | ✅ PASS | 22 columns matched; rows=279 |
| DB-Schema | FredData | ✅ PASS | 5 columns matched; rows=3885 |
| DB-Schema | TaiwanStockInfo | ✅ PASS | 5 columns matched; rows=2798 |
| DB-FRED | completeness | ✅ PASS | series counts={'DFF': 1000, 'T10Y2Y': 958, 'UNRATE': 939, 'VIXCLS': 988} |
| Freshness | FRED/DFF | ✅ PASS | latest=2026-05-14, age=3d |
| Freshness | FRED/UNRATE | ✅ PASS | latest=2026-04-01, age=46d |
| Freshness | FRED/T10Y2Y | ✅ PASS | latest=2026-05-15, age=2d |
| Freshness | FRED/VIXCLS | ✅ PASS | latest=2026-05-14, age=3d |
| Log-Schema | pipeline_execution_log | ✅ PASS | exists; rows=12 |
| Log-Schema | data_audit_log | ✅ PASS | exists; rows=1357 |
| Pipeline-Log | task_status | ✅ PASS | 12 recent tasks have acceptable status |
| Pipeline-Log | end_time | ✅ PASS | 12 recent tasks have end_time |
