# Quantum Finance schema 後驗收稽核報告

- **時間**: 2026-05-16 16:29:12
- **憲章**: 系統架構大憲章_v5.4.22.md
- **稽核工具**: audit_supply_chain v1.18
- **schema 基準**: data_schema v2.11
- **判定結果**: **PERFECT** (PASS=22, WARN=0, FAIL=0)

## 稽核明細

| 來源層 | 項目 | 狀態 | 詳細 |
| :--- | :--- | :--- | :--- |
| DB-Schema | pipeline_execution_log | ✅ PASS | 9 columns matched; rows=23 |
| DB-Schema | data_audit_log | ✅ PASS | 7 columns matched; rows=40 |
| DB-Schema | TaiwanStockPrice | ✅ PASS | 10 columns matched; rows=0 |
| DB-Schema | TaiwanStockPriceAdj | ✅ PASS | 10 columns matched; rows=0 |
| DB-Schema | TaiwanStockPER | ✅ PASS | 5 columns matched; rows=0 |
| DB-Schema | TaiwanStockInstitutionalInvestorsBuySell | ✅ PASS | 5 columns matched; rows=0 |
| DB-Schema | TaiwanStockMarginPurchaseShortSale | ✅ PASS | 16 columns matched; rows=0 |
| DB-Schema | TaiwanStockShareholding | ✅ PASS | 13 columns matched; rows=0 |
| DB-Schema | TaiwanStockFinancialStatements | ✅ PASS | 5 columns matched; rows=0 |
| DB-Schema | TaiwanStockMonthRevenue | ✅ PASS | 7 columns matched; rows=0 |
| DB-Schema | TaiwanStockDividend | ✅ PASS | 22 columns matched; rows=0 |
| DB-Schema | FredData | ✅ PASS | 5 columns matched; rows=3885 |
| DB-Schema | TaiwanStockInfo | ✅ PASS | 5 columns matched; rows=2799 |
| DB-FRED | completeness | ✅ PASS | series counts={'DFF': 1000, 'T10Y2Y': 958, 'UNRATE': 939, 'VIXCLS': 988} |
| Freshness | FRED/DFF | ✅ PASS | latest=2026-05-14, age=2d |
| Freshness | FRED/UNRATE | ✅ PASS | latest=2026-04-01, age=45d |
| Freshness | FRED/T10Y2Y | ✅ PASS | latest=2026-05-15, age=1d |
| Freshness | FRED/VIXCLS | ✅ PASS | latest=2026-05-14, age=2d |
| Log-Schema | pipeline_execution_log | ✅ PASS | exists; rows=23 |
| Log-Schema | data_audit_log | ✅ PASS | exists; rows=40 |
| Pipeline-Log | task_status | ✅ PASS | 23 recent tasks have acceptable status |
| Pipeline-Log | end_time | ✅ PASS | 23 recent tasks have end_time |
