# Quantum Finance schema 後驗收稽核報告

- **時間**: 2026-05-16 16:44:38
- **憲章**: 系統架構大憲章_v5.4.22.md
- **稽核工具**: audit_supply_chain v1.18
- **schema 基準**: data_schema v2.11
- **判定結果**: **PERFECT** (PASS=29, WARN=0, FAIL=0)

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
| DB-Schema | pipeline_execution_log | ✅ PASS | 9 columns matched; rows=3 |
| DB-Schema | data_audit_log | ✅ PASS | 7 columns matched; rows=19 |
| DB-Schema | TaiwanStockPrice | ✅ PASS | 10 columns matched; rows=0 |
| DB-Schema | TaiwanStockPriceAdj | ✅ PASS | 10 columns matched; rows=0 |
| DB-Schema | TaiwanStockPER | ✅ PASS | 5 columns matched; rows=0 |
| DB-Schema | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | 5 columns matched; rows=0 |
| DB-Schema | TaiwanStockMarginPurchaseShortSale | ✅ PASS | 16 columns matched; rows=0 |
| DB-Schema | TaiwanStockShareholding | ✅ PASS | 13 columns matched; rows=0 |
| DB-Schema | TaiwanStockFinancialStatements | ✅ PASS | 5 columns matched; rows=0 |
| DB-Schema | TaiwanStockMonthRevenue | ✅ PASS | 7 columns matched; rows=0 |
| DB-Schema | TaiwanStockDividend | ✅ PASS | 22 columns matched; rows=0 |
| DB-Schema | FredData | ✅ PASS | 5 columns matched; rows=0 |
| DB-Schema | TaiwanStockInfo | ✅ PASS | 5 columns matched; rows=0 |
| Freshness | FredData | ✅ DEFERRED | pre-ingestion: table exists but has no data yet |
| Log-Schema | pipeline_execution_log | ✅ PASS | exists; rows=3 |
| Log-Schema | data_audit_log | ✅ PASS | exists; rows=19 |
| Pipeline-Log | task_status | ✅ PASS | 3 recent tasks have acceptable status |
| Pipeline-Log | end_time | ✅ PASS | 3 recent tasks have end_time |
