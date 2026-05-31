# Quantum Finance schema 後驗收稽核報告

- **時間**: 2026-05-31 19:02:19
- **憲章**: 系統架構大憲章_v6.0.0.md
- **稽核工具**: audit_supply_chain v1.19
- **schema 基準**: data_schema v2.16
- **判定結果**: **FAILED** (PASS=34, WARN=0, FAIL=1)

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
| DB-Schema | pipeline_execution_log | ✅ PASS | 9 columns matched; rows=278 |
| DB-Schema | data_audit_log | ✅ PASS | 7 columns matched; rows=23701 |
| DB-Schema | TaiwanStockPrice | ✅ PASS | 10 columns matched; rows=10516024 |
| DB-Schema | TaiwanStockPriceAdj | ✅ PASS | 10 columns matched; rows=10507832 |
| DB-Schema | TaiwanStockPER | ✅ PASS | 5 columns matched; rows=7352995 |
| DB-Schema | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | 5 columns matched; rows=25044796 |
| DB-Schema | TaiwanStockMarginPurchaseShortSale | ✅ PASS | 16 columns matched; rows=7722170 |
| DB-Schema | TaiwanStockShareholding | ✅ PASS | 13 columns matched; rows=8372857 |
| DB-Schema | TaiwanStockFinancialStatements | ✅ PASS | 5 columns matched; rows=2663205 |
| DB-Schema | TaiwanStockBalanceSheet | ✅ PASS | 5 columns matched; rows=8248086 |
| DB-Schema | TaiwanStockMonthRevenue | ✅ PASS | 7 columns matched; rows=460187 |
| DB-Schema | TaiwanStockDividend | ✅ PASS | 22 columns matched; rows=29312 |
| DB-Schema | FredData | ✅ PASS | 5 columns matched; rows=48895 |
| DB-Schema | TaiwanStockInfo | ✅ PASS | 5 columns matched; rows=2803 |
| DB-FRED | completeness | ✅ PASS | series counts={'DFF': 26265, 'T10Y2Y': 12495, 'UNRATE': 939, 'VIXCLS': 9196} |
| Freshness | FRED/DFF | ✅ PASS | latest=2026-05-28, age=3d |
| Freshness | FRED/UNRATE | ✅ PASS | latest=2026-04-01, age=60d |
| Freshness | FRED/T10Y2Y | ✅ PASS | latest=2026-05-29, age=2d |
| Freshness | FRED/VIXCLS | ✅ PASS | latest=2026-05-28, age=3d |
| Log-Schema | pipeline_execution_log | ✅ PASS | exists; rows=278 |
| Log-Schema | data_audit_log | ✅ PASS | exists; rows=23701 |
| Pipeline-Log | task_status | ❌ FAILED | audit_source_availability_full=failed |
| Pipeline-Log | end_time | ✅ PASS | 5 recent tasks have end_time |
