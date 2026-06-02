# Quantum Finance schema 後驗收稽核報告

- **時間**: 2026-06-01 18:31:23
- **憲章**: 系統架構大憲章_v6.0.0.md
- **稽核工具**: audit_supply_chain v1.19
- **schema 基準**: data_schema v2.16
- **判定結果**: **FAILED** (PASS=33, WARN=1, FAIL=1)

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
| API-Contract | TaiwanStockBalanceSheet | ✅ PASS | ✅ [API-PASS] TaiwanStockBalanceSheet - 5 columns matched |
| API-Contract | TaiwanStockMonthRevenue | ✅ PASS | ✅ [API-PASS] TaiwanStockMonthRevenue - 7 columns matched |
| API-Contract | TaiwanStockDividend | ✅ PASS | ✅ [API-PASS] TaiwanStockDividend - 22 columns matched |
| API-Contract | TaiwanStockInfo | ✅ PASS | ✅ [API-PASS] TaiwanStockInfo - 5 columns matched |
| API-Contract | FredData | ✅ PASS | ✅ [API-PASS] FredData - 4+1 derived columns matched |
| DB-Schema | pipeline_execution_log | ✅ PASS | 9 columns matched; rows=123 |
| DB-Schema | data_audit_log | ✅ PASS | 7 columns matched; rows=28189 |
| DB-Schema | TaiwanStockPrice | ✅ PASS | 10 columns matched; rows=10519984 |
| DB-Schema | TaiwanStockPriceAdj | ✅ PASS | 10 columns matched; rows=10517934 |
| DB-Schema | TaiwanStockPER | ✅ PASS | 5 columns matched; rows=7353665 |
| DB-Schema | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | 5 columns matched; rows=25068187 |
| DB-Schema | TaiwanStockMarginPurchaseShortSale | ✅ PASS | 16 columns matched; rows=7723999 |
| DB-Schema | TaiwanStockShareholding | ✅ PASS | 13 columns matched; rows=8381096 |
| DB-Schema | TaiwanStockFinancialStatements | ✅ PASS | 5 columns matched; rows=2663362 |
| DB-Schema | TaiwanStockBalanceSheet | ✅ PASS | 5 columns matched; rows=8249000 |
| DB-Schema | TaiwanStockMonthRevenue | ✅ PASS | 7 columns matched; rows=460057 |
| DB-Schema | TaiwanStockDividend | ✅ PASS | 22 columns matched; rows=29421 |
| DB-Schema | FredData | ✅ PASS | 5 columns matched; rows=48895 |
| DB-Schema | TaiwanStockInfo | ✅ PASS | 5 columns matched; rows=2806 |
| DB-FRED | completeness | ✅ PASS | series counts={'DFF': 26265, 'T10Y2Y': 12495, 'UNRATE': 939, 'VIXCLS': 9196} |
| Freshness | FRED/DFF | ✅ PASS | latest=2026-05-28, age=4d |
| Freshness | FRED/UNRATE | ⚠️ STALE | latest=2026-04-01, age=61d, threshold=60d |
| Freshness | FRED/T10Y2Y | ✅ PASS | latest=2026-05-29, age=3d |
| Freshness | FRED/VIXCLS | ✅ PASS | latest=2026-05-28, age=4d |
| Log-Schema | pipeline_execution_log | ✅ PASS | exists; rows=123 |
| Log-Schema | data_audit_log | ✅ PASS | exists; rows=28189 |
| Pipeline-Log | task_status | ❌ FAILED | universe_completeness_schema_init_v0.1=failed; feature_store_builder_v0.1=failed; feature_store_builder_v0.1=failed |
| Pipeline-Log | end_time | ✅ PASS | 123 recent tasks have end_time |
