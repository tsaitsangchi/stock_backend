# Quantum Finance schema 後驗收稽核報告

- **時間**: 2026-06-09 07:25:49
- **憲章**: 系統架構大憲章_v6.0.0.md
- **稽核工具**: audit_supply_chain v1.20
- **schema 基準**: data_schema v2.16
- **判定結果**: **FAILED** (PASS=32, WARN=0, FAIL=1)

## 稽核明細

| 來源層 | 項目 | 狀態 | 詳細 |
| :--- | :--- | :--- | :--- |
| API-Contract | TaiwanStockPrice | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockPriceAdj | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockPER | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockMarginPurchaseShortSale | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockShareholding | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockFinancialStatements | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockBalanceSheet | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockMonthRevenue | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | TaiwanStockDividend | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| API-Contract | FredData | ✅ PASS | §14.7-DJ generic auto-schema:無宣告 DDL 契約 probe;由 §14.7-CE 逐股 DB-vs-API 對帳驗證 |
| DB-Schema | TaiwanStockPrice | ✅ PASS | 10 columns matched; rows=10702393 |
| DB-Schema | TaiwanStockPriceAdj | ✅ PASS | 10 columns matched; rows=10700339 |
| DB-Schema | TaiwanStockPER | ✅ PASS | 5 columns matched; rows=7363556 |
| DB-Schema | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | 5 columns matched; rows=25159261 |
| DB-Schema | TaiwanStockMarginPurchaseShortSale | ✅ PASS | 16 columns matched; rows=7741478 |
| DB-Schema | TaiwanStockShareholding | ✅ PASS | 13 columns matched; rows=8399781 |
| DB-Schema | TaiwanStockFinancialStatements | ✅ PASS | 5 columns matched; rows=2663415 |
| DB-Schema | TaiwanStockBalanceSheet | ✅ PASS | 5 columns matched; rows=8249356 |
| DB-Schema | TaiwanStockMonthRevenue | ✅ PASS | 7 columns matched; rows=460992 |
| DB-Schema | TaiwanStockDividend | ✅ PASS | 22 columns matched; rows=29650 |
| DB-Schema | FredData | ✅ PASS | 5 columns matched; rows=48916 |
| DB-Schema | data_audit_log | ✅ PASS | 7 columns matched; rows=24515 |
| DB-Schema | pipeline_execution_log | ✅ PASS | 9 columns matched; rows=11 |
| DB-FRED | completeness | ✅ PASS | series counts={'DFF': 26273, 'T10Y2Y': 12501, 'UNRATE': 940, 'VIXCLS': 9202} |
| Freshness | FRED/DFF | ✅ PASS | latest=2026-06-05, age=4d |
| Freshness | FRED/UNRATE | ✅ PASS | latest=2026-05-01, age=39d |
| Freshness | FRED/T10Y2Y | ✅ PASS | latest=2026-06-08, age=1d |
| Freshness | FRED/VIXCLS | ✅ PASS | latest=2026-06-05, age=4d |
| Log-Schema | pipeline_execution_log | ✅ PASS | exists; rows=11 |
| Log-Schema | data_audit_log | ✅ PASS | exists; rows=24515 |
| Pipeline-Log | task_status | ❌ FAILED | sync_all_full=failed |
| Pipeline-Log | end_time | ✅ PASS | 11 recent tasks have end_time |
