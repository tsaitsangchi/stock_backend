# API Schema Compliance Audit Report (v0.1)

**執行日期**: 2026-05-21 07:55:49
**對應憲章**: 系統架構大憲章_v6.0.0.md / §3.2A / §14.7-AJ

## 環境快照

- sample_size: 100
- include_fred: True
- target_table: ALL
- skip_api_probe: False
- active_layers: A,B,C,D,E,F,G,H,I

## 9 層結果摘要

| Layer | 名稱 | PASS | FAIL | 狀態 |
|---|---|---|---|---|
| A | DDL ↔ DB Physical Consistency | 0 | 119 | FAIL |
| B | API Sample ↔ DDL Type Compatibility | 102 | 0 | PASS |
| C | API Sample Length / Precision Range | 83 | 0 | PASS |
| D | NULL Ratio Sanity | 103 | 0 | PASS |
| E | PK / Unique Constraint Uniqueness | 0 | 11 | FAIL |
| F | Duplicate Row Detection | 0 | 13 | FAIL |
| G | Date Series Continuity | 0 | 11 | FAIL |
| H | Referential Integrity | 0 | 9 | FAIL |
| I | Value Range Sanity | 0 | 8 | FAIL |

**verdict**: FAILED
**latency**: 3430.3 ms

## 9 層詳細紀錄

### Layer A: DDL ↔ DB Physical Consistency

- PASS=0, FAIL=119
- ❌ [A] pipeline_execution_log.id: DB 表不存在
- ❌ [A] pipeline_execution_log.task_name: DB 表不存在
- ❌ [A] pipeline_execution_log.category: DB 表不存在
- ❌ [A] pipeline_execution_log.stock_id: DB 表不存在
- ❌ [A] pipeline_execution_log.start_time: DB 表不存在
- ❌ [A] pipeline_execution_log.end_time: DB 表不存在
- ❌ [A] pipeline_execution_log.status: DB 表不存在
- ❌ [A] pipeline_execution_log.duration_ms: DB 表不存在
- ❌ [A] pipeline_execution_log.error_msg: DB 表不存在
- ❌ [A] data_audit_log.id: DB 表不存在
- ❌ [A] data_audit_log.table_name: DB 表不存在
- ❌ [A] data_audit_log.stock_id: DB 表不存在
- ❌ [A] data_audit_log.data_date: DB 表不存在
- ❌ [A] data_audit_log.action_type: DB 表不存在
- ❌ [A] data_audit_log.rows_affected: DB 表不存在
- ❌ [A] data_audit_log.timestamp: DB 表不存在
- ❌ [A] TaiwanStockPrice.date: DB 表不存在
- ❌ [A] TaiwanStockPrice.stock_id: DB 表不存在
- ❌ [A] TaiwanStockPrice.Trading_Volume: DB 表不存在
- ❌ [A] TaiwanStockPrice.Trading_money: DB 表不存在
- ❌ [A] TaiwanStockPrice.open: DB 表不存在
- ❌ [A] TaiwanStockPrice.max: DB 表不存在
- ❌ [A] TaiwanStockPrice.min: DB 表不存在
- ❌ [A] TaiwanStockPrice.close: DB 表不存在
- ❌ [A] TaiwanStockPrice.spread: DB 表不存在
- ❌ [A] TaiwanStockPrice.Trading_turnover: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.date: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.stock_id: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.Trading_Volume: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.Trading_money: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.open: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.max: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.min: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.close: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.spread: DB 表不存在
- ❌ [A] TaiwanStockPriceAdj.Trading_turnover: DB 表不存在
- ❌ [A] TaiwanStockPER.date: DB 表不存在
- ❌ [A] TaiwanStockPER.stock_id: DB 表不存在
- ❌ [A] TaiwanStockPER.dividend_yield: DB 表不存在
- ❌ [A] TaiwanStockPER.PER: DB 表不存在
- ❌ [A] TaiwanStockPER.PBR: DB 表不存在
- ❌ [A] TaiwanStockInstitutionalInvestorsBuySell.date: DB 表不存在
- ❌ [A] TaiwanStockInstitutionalInvestorsBuySell.stock_id: DB 表不存在
- ❌ [A] TaiwanStockInstitutionalInvestorsBuySell.buy: DB 表不存在
- ❌ [A] TaiwanStockInstitutionalInvestorsBuySell.name: DB 表不存在
- ❌ [A] TaiwanStockInstitutionalInvestorsBuySell.sell: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.date: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.stock_id: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseBuy: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseSell: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseCashRepayment: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseLimit: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseTodayBalance: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseYesterdayBalance: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleBuy: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleSell: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleCashRepayment: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleLimit: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleTodayBalance: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleYesterdayBalance: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.OffsetLoanAndShort: DB 表不存在
- ❌ [A] TaiwanStockMarginPurchaseShortSale.Note: DB 表不存在
- ❌ [A] TaiwanStockShareholding.date: DB 表不存在
- ❌ [A] TaiwanStockShareholding.stock_id: DB 表不存在
- ❌ [A] TaiwanStockShareholding.stock_name: DB 表不存在
- ❌ [A] TaiwanStockShareholding.InternationalCode: DB 表不存在
- ❌ [A] TaiwanStockShareholding.ForeignInvestmentRemainingShares: DB 表不存在
- ❌ [A] TaiwanStockShareholding.ForeignInvestmentShares: DB 表不存在
- ❌ [A] TaiwanStockShareholding.ForeignInvestmentRemainRatio: DB 表不存在
- ❌ [A] TaiwanStockShareholding.ForeignInvestmentSharesRatio: DB 表不存在
- ❌ [A] TaiwanStockShareholding.NumberOfSharesIssued: DB 表不存在
- ❌ [A] TaiwanStockShareholding.ForeignInvestmentUpperLimitRatio: DB 表不存在
- ❌ [A] TaiwanStockShareholding.ChineseInvestmentUpperLimitRatio: DB 表不存在
- ❌ [A] TaiwanStockShareholding.RecentlyDeclareDate: DB 表不存在
- ❌ [A] TaiwanStockShareholding.note: DB 表不存在
- ❌ [A] TaiwanStockFinancialStatements.date: DB 表不存在
- ❌ [A] TaiwanStockFinancialStatements.stock_id: DB 表不存在
- ❌ [A] TaiwanStockFinancialStatements.type: DB 表不存在
- ❌ [A] TaiwanStockFinancialStatements.value: DB 表不存在
- ❌ [A] TaiwanStockFinancialStatements.origin_name: DB 表不存在
- ❌ [A] TaiwanStockMonthRevenue.date: DB 表不存在
- ❌ [A] TaiwanStockMonthRevenue.stock_id: DB 表不存在
- ❌ [A] TaiwanStockMonthRevenue.country: DB 表不存在
- ❌ [A] TaiwanStockMonthRevenue.revenue: DB 表不存在
- ❌ [A] TaiwanStockMonthRevenue.revenue_month: DB 表不存在
- ❌ [A] TaiwanStockMonthRevenue.revenue_year: DB 表不存在
- ❌ [A] TaiwanStockMonthRevenue.create_time: DB 表不存在
- ❌ [A] TaiwanStockDividend.date: DB 表不存在
- ❌ [A] TaiwanStockDividend.stock_id: DB 表不存在
- ❌ [A] TaiwanStockDividend.year: DB 表不存在
- ❌ [A] TaiwanStockDividend.StockEarningsDistribution: DB 表不存在
- ❌ [A] TaiwanStockDividend.StockStatutorySurplus: DB 表不存在
- ❌ [A] TaiwanStockDividend.CashEarningsDistribution: DB 表不存在
- ❌ [A] TaiwanStockDividend.CashStatutorySurplus: DB 表不存在
- ❌ [A] TaiwanStockDividend.AnnouncementDate: DB 表不存在
- ❌ [A] TaiwanStockDividend.AnnouncementTime: DB 表不存在
- ❌ [A] TaiwanStockDividend.CashDividendPaymentDate: DB 表不存在
- ❌ [A] TaiwanStockDividend.CashExDividendTradingDate: DB 表不存在
- ❌ [A] TaiwanStockDividend.CashIncreaseSubscriptionRate: DB 表不存在
- ❌ [A] TaiwanStockDividend.CashIncreaseSubscriptionpRrice: DB 表不存在
- ❌ [A] TaiwanStockDividend.ParticipateDistributionOfTotalShares: DB 表不存在
- ❌ [A] TaiwanStockDividend.RatioOfEmployeeStockDividend: DB 表不存在
- ❌ [A] TaiwanStockDividend.RatioOfEmployeeStockDividendOfTotal: DB 表不存在
- ❌ [A] TaiwanStockDividend.RemunerationOfDirectorsAndSupervisors: DB 表不存在
- ❌ [A] TaiwanStockDividend.StockExDividendTradingDate: DB 表不存在
- ❌ [A] TaiwanStockDividend.TotalEmployeeCashDividend: DB 表不存在
- ❌ [A] TaiwanStockDividend.TotalEmployeeStockDividend: DB 表不存在
- ❌ [A] TaiwanStockDividend.TotalEmployeeStockDividendAmount: DB 表不存在
- ❌ [A] TaiwanStockDividend.TotalNumberOfCashCapitalIncrease: DB 表不存在
- ❌ [A] FredData.date: DB 表不存在
- ❌ [A] FredData.series_id: DB 表不存在
- ❌ [A] FredData.value: DB 表不存在
- ❌ [A] FredData.realtime_start: DB 表不存在
- ❌ [A] FredData.realtime_end: DB 表不存在
- ❌ [A] TaiwanStockInfo.stock_id: DB 表不存在
- ❌ [A] TaiwanStockInfo.stock_name: DB 表不存在
- ❌ [A] TaiwanStockInfo.industry_category: DB 表不存在
- ❌ [A] TaiwanStockInfo.type: DB 表不存在
- ❌ [A] TaiwanStockInfo.date: DB 表不存在

### Layer B: API Sample ↔ DDL Type Compatibility

- PASS=102, FAIL=0
- ✅ [B] TaiwanStockPrice.date: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.stock_id: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.Trading_Volume: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.Trading_money: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.open: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.max: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.min: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.close: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.spread: cast 通過 100/100
- ✅ [B] TaiwanStockPrice.Trading_turnover: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.date: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.stock_id: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.Trading_Volume: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.Trading_money: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.open: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.max: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.min: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.close: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.spread: cast 通過 100/100
- ✅ [B] TaiwanStockPriceAdj.Trading_turnover: cast 通過 100/100
- ✅ [B] TaiwanStockPER.date: cast 通過 100/100
- ✅ [B] TaiwanStockPER.stock_id: cast 通過 100/100
- ✅ [B] TaiwanStockPER.dividend_yield: cast 通過 100/100
- ✅ [B] TaiwanStockPER.PER: cast 通過 100/100
- ✅ [B] TaiwanStockPER.PBR: cast 通過 100/100
- ✅ [B] TaiwanStockInstitutionalInvestorsBuySell.date: cast 通過 100/100
- ✅ [B] TaiwanStockInstitutionalInvestorsBuySell.stock_id: cast 通過 100/100
- ✅ [B] TaiwanStockInstitutionalInvestorsBuySell.buy: cast 通過 100/100
- ✅ [B] TaiwanStockInstitutionalInvestorsBuySell.name: cast 通過 100/100
- ✅ [B] TaiwanStockInstitutionalInvestorsBuySell.sell: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.date: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.stock_id: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.MarginPurchaseBuy: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.MarginPurchaseSell: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.MarginPurchaseCashRepayment: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.MarginPurchaseLimit: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.MarginPurchaseTodayBalance: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.MarginPurchaseYesterdayBalance: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.ShortSaleBuy: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.ShortSaleSell: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.ShortSaleCashRepayment: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.ShortSaleLimit: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.ShortSaleTodayBalance: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.ShortSaleYesterdayBalance: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.OffsetLoanAndShort: cast 通過 100/100
- ✅ [B] TaiwanStockMarginPurchaseShortSale.Note: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.date: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.stock_id: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.stock_name: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.InternationalCode: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.ForeignInvestmentRemainingShares: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.ForeignInvestmentShares: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.ForeignInvestmentRemainRatio: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.ForeignInvestmentSharesRatio: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.NumberOfSharesIssued: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.ForeignInvestmentUpperLimitRatio: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.ChineseInvestmentUpperLimitRatio: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.RecentlyDeclareDate: cast 通過 100/100
- ✅ [B] TaiwanStockShareholding.note: cast 通過 8/8
- ✅ [B] TaiwanStockFinancialStatements.date: cast 通過 100/100
- ✅ [B] TaiwanStockFinancialStatements.stock_id: cast 通過 100/100
- ✅ [B] TaiwanStockFinancialStatements.type: cast 通過 100/100
- ✅ [B] TaiwanStockFinancialStatements.value: cast 通過 100/100
- ✅ [B] TaiwanStockFinancialStatements.origin_name: cast 通過 100/100
- ✅ [B] TaiwanStockMonthRevenue.date: cast 通過 29/29
- ✅ [B] TaiwanStockMonthRevenue.stock_id: cast 通過 29/29
- ✅ [B] TaiwanStockMonthRevenue.country: cast 通過 29/29
- ✅ [B] TaiwanStockMonthRevenue.revenue: cast 通過 29/29
- ✅ [B] TaiwanStockMonthRevenue.revenue_month: cast 通過 29/29
- ✅ [B] TaiwanStockMonthRevenue.revenue_year: cast 通過 29/29
- ✅ [B] TaiwanStockMonthRevenue.create_time: cast 通過 3/3
- ✅ [B] TaiwanStockDividend.date: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.stock_id: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.year: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.StockEarningsDistribution: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.StockStatutorySurplus: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.CashEarningsDistribution: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.CashStatutorySurplus: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.AnnouncementDate: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.AnnouncementTime: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.CashDividendPaymentDate: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.CashExDividendTradingDate: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.CashIncreaseSubscriptionRate: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.CashIncreaseSubscriptionpRrice: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.ParticipateDistributionOfTotalShares: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.RatioOfEmployeeStockDividend: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.RatioOfEmployeeStockDividendOfTotal: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.RemunerationOfDirectorsAndSupervisors: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.StockExDividendTradingDate: 樣本中此欄位無非空值
- ✅ [B] TaiwanStockDividend.TotalEmployeeCashDividend: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.TotalEmployeeStockDividend: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.TotalEmployeeStockDividendAmount: cast 通過 25/25
- ✅ [B] TaiwanStockDividend.TotalNumberOfCashCapitalIncrease: cast 通過 25/25
- ✅ [B] FredData.date: cast 通過 100/100
- ✅ [B] FredData.value: cast 通過 100/100
- ✅ [B] FredData.realtime_start: cast 通過 100/100
- ✅ [B] FredData.realtime_end: cast 通過 100/100
- ✅ [B] TaiwanStockInfo.stock_id: cast 通過 100/100
- ✅ [B] TaiwanStockInfo.stock_name: cast 通過 100/100
- ✅ [B] TaiwanStockInfo.industry_category: cast 通過 100/100
- ✅ [B] TaiwanStockInfo.type: cast 通過 100/100
- ✅ [B] TaiwanStockInfo.date: cast 通過 100/100

### Layer C: API Sample Length / Precision Range

- PASS=83, FAIL=0
- ✅ [C] TaiwanStockPrice.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockPrice.Trading_Volume: max abs 159662795 < 100000000000000
- ✅ [C] TaiwanStockPrice.Trading_money: max abs 133482646539 < 100000000000000
- ✅ [C] TaiwanStockPrice.open: max abs 1065.0 < 100000000000000
- ✅ [C] TaiwanStockPrice.max: max abs 1080.0 < 100000000000000
- ✅ [C] TaiwanStockPrice.min: max abs 1055.0 < 100000000000000
- ✅ [C] TaiwanStockPrice.close: max abs 1080.0 < 100000000000000
- ✅ [C] TaiwanStockPrice.spread: max abs 88.0 < 100000000000000
- ✅ [C] TaiwanStockPrice.Trading_turnover: max abs 588373 < 100000000000000
- ✅ [C] TaiwanStockPriceAdj.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockPriceAdj.Trading_Volume: max abs 159662795 < 100000000000000
- ✅ [C] TaiwanStockPriceAdj.Trading_money: max abs 133482646539 < 100000000000000
- ✅ [C] TaiwanStockPriceAdj.open: max abs 1035.85678 < 100000000000000
- ✅ [C] TaiwanStockPriceAdj.max: max abs 1050.446312 < 100000000000000
- ✅ [C] TaiwanStockPriceAdj.min: max abs 1026.130425 < 100000000000000
- ✅ [C] TaiwanStockPriceAdj.close: max abs 1050.446312 < 100000000000000
- ✅ [C] TaiwanStockPriceAdj.spread: max abs 85.591922 < 100000000000000
- ✅ [C] TaiwanStockPriceAdj.Trading_turnover: max abs 588373 < 100000000000000
- ✅ [C] TaiwanStockPER.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockPER.dividend_yield: max abs 1.68 < 100000000000000
- ✅ [C] TaiwanStockPER.PER: max abs 32.69 < 100000000000000
- ✅ [C] TaiwanStockPER.PBR: max abs 7.7 < 100000000000000
- ✅ [C] TaiwanStockInstitutionalInvestorsBuySell.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockInstitutionalInvestorsBuySell.buy: max abs 36036339 < 100000000000000
- ✅ [C] TaiwanStockInstitutionalInvestorsBuySell.name: max length 19 ≤ 255
- ✅ [C] TaiwanStockInstitutionalInvestorsBuySell.sell: max abs 37455029 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockMarginPurchaseShortSale.MarginPurchaseBuy: max abs 6701 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.MarginPurchaseSell: max abs 5839 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.MarginPurchaseCashRepayment: max abs 146 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.MarginPurchaseLimit: max abs 6483757 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.MarginPurchaseTodayBalance: max abs 35648 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.MarginPurchaseYesterdayBalance: max abs 35648 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.ShortSaleBuy: max abs 367 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.ShortSaleSell: max abs 500 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.ShortSaleCashRepayment: max abs 64 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.ShortSaleLimit: max abs 6483757 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.ShortSaleTodayBalance: max abs 606 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.ShortSaleYesterdayBalance: max abs 606 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.OffsetLoanAndShort: max abs 469 < 100000000000000
- ✅ [C] TaiwanStockMarginPurchaseShortSale.Note: max length 2 ≤ 255
- ✅ [C] TaiwanStockShareholding.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockShareholding.stock_name: max length 3 ≤ 255
- ✅ [C] TaiwanStockShareholding.InternationalCode: max length 12 ≤ 255
- ✅ [C] TaiwanStockShareholding.ForeignInvestmentRemainingShares: max abs 6920119954 < 100000000000000
- ✅ [C] TaiwanStockShareholding.ForeignInvestmentShares: max abs 19408239899 < 100000000000000
- ✅ [C] TaiwanStockShareholding.ForeignInvestmentRemainRatio: max abs 26.68 < 100000000000000
- ✅ [C] TaiwanStockShareholding.ForeignInvestmentSharesRatio: max abs 74.83 < 100000000000000
- ✅ [C] TaiwanStockShareholding.NumberOfSharesIssued: max abs 25935030992 < 100000000000000
- ✅ [C] TaiwanStockShareholding.ForeignInvestmentUpperLimitRatio: max abs 100.0 < 100000000000000
- ✅ [C] TaiwanStockShareholding.ChineseInvestmentUpperLimitRatio: max abs 100.0 < 100000000000000
- ✅ [C] TaiwanStockShareholding.note: max length 133 ≤ 255
- ✅ [C] TaiwanStockFinancialStatements.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockFinancialStatements.type: max length 72 ≤ 255
- ✅ [C] TaiwanStockFinancialStatements.value: max abs 933791869000.0 < 100000000000000
- ✅ [C] TaiwanStockFinancialStatements.origin_name: max length 14 ≤ 255
- ✅ [C] TaiwanStockMonthRevenue.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockMonthRevenue.country: max length 6 ≤ 255
- ✅ [C] TaiwanStockMonthRevenue.revenue: max abs 415191699000 < 100000000000000
- ✅ [C] TaiwanStockMonthRevenue.revenue_month: max abs 12 < 100000000000000
- ✅ [C] TaiwanStockMonthRevenue.revenue_year: max abs 2026 < 100000000000000
- ✅ [C] TaiwanStockDividend.stock_id: max length 4 ≤ 255
- ✅ [C] TaiwanStockDividend.year: max length 7 ≤ 255
- ✅ [C] TaiwanStockDividend.StockEarningsDistribution: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.StockStatutorySurplus: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.CashEarningsDistribution: max abs 6.00003573 < 100000000000000
- ✅ [C] TaiwanStockDividend.CashStatutorySurplus: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.AnnouncementTime: max length 8 ≤ 255
- ✅ [C] TaiwanStockDividend.CashIncreaseSubscriptionRate: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.CashIncreaseSubscriptionpRrice: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.ParticipateDistributionOfTotalShares: max abs 25933629242.0 < 100000000000000
- ✅ [C] TaiwanStockDividend.RatioOfEmployeeStockDividend: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.RatioOfEmployeeStockDividendOfTotal: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.RemunerationOfDirectorsAndSupervisors: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.TotalEmployeeCashDividend: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.TotalEmployeeStockDividend: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.TotalEmployeeStockDividendAmount: max abs 0 < 100000000000000
- ✅ [C] TaiwanStockDividend.TotalNumberOfCashCapitalIncrease: max abs 0 < 100000000000000
- ✅ [C] FredData.value: max abs 1.44 < 100000000000000
- ✅ [C] TaiwanStockInfo.stock_id: max length 6 ≤ 255
- ✅ [C] TaiwanStockInfo.stock_name: max length 10 ≤ 255
- ✅ [C] TaiwanStockInfo.industry_category: max length 14 ≤ 255
- ✅ [C] TaiwanStockInfo.type: max length 8 ≤ 255

### Layer D: NULL Ratio Sanity

- PASS=103, FAIL=0
- ✅ [D] TaiwanStockPrice.date: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.Trading_Volume: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.Trading_money: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.open: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.max: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.min: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.close: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.spread: NULL 比例 0%
- ✅ [D] TaiwanStockPrice.Trading_turnover: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.date: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.Trading_Volume: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.Trading_money: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.open: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.max: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.min: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.close: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.spread: NULL 比例 0%
- ✅ [D] TaiwanStockPriceAdj.Trading_turnover: NULL 比例 0%
- ✅ [D] TaiwanStockPER.date: NULL 比例 0%
- ✅ [D] TaiwanStockPER.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockPER.dividend_yield: NULL 比例 0%
- ✅ [D] TaiwanStockPER.PER: NULL 比例 0%
- ✅ [D] TaiwanStockPER.PBR: NULL 比例 0%
- ✅ [D] TaiwanStockInstitutionalInvestorsBuySell.date: NULL 比例 0%
- ✅ [D] TaiwanStockInstitutionalInvestorsBuySell.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockInstitutionalInvestorsBuySell.buy: NULL 比例 0%
- ✅ [D] TaiwanStockInstitutionalInvestorsBuySell.name: NULL 比例 0%
- ✅ [D] TaiwanStockInstitutionalInvestorsBuySell.sell: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.date: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.MarginPurchaseBuy: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.MarginPurchaseSell: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.MarginPurchaseCashRepayment: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.MarginPurchaseLimit: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.MarginPurchaseTodayBalance: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.MarginPurchaseYesterdayBalance: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.ShortSaleBuy: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.ShortSaleSell: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.ShortSaleCashRepayment: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.ShortSaleLimit: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.ShortSaleTodayBalance: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.ShortSaleYesterdayBalance: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.OffsetLoanAndShort: NULL 比例 0%
- ✅ [D] TaiwanStockMarginPurchaseShortSale.Note: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.date: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.stock_name: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.InternationalCode: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.ForeignInvestmentRemainingShares: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.ForeignInvestmentShares: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.ForeignInvestmentRemainRatio: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.ForeignInvestmentSharesRatio: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.NumberOfSharesIssued: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.ForeignInvestmentUpperLimitRatio: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.ChineseInvestmentUpperLimitRatio: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.RecentlyDeclareDate: NULL 比例 0%
- ✅ [D] TaiwanStockShareholding.note: NULL 比例 92%
- ✅ [D] TaiwanStockFinancialStatements.date: NULL 比例 0%
- ✅ [D] TaiwanStockFinancialStatements.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockFinancialStatements.type: NULL 比例 0%
- ✅ [D] TaiwanStockFinancialStatements.value: NULL 比例 0%
- ✅ [D] TaiwanStockFinancialStatements.origin_name: NULL 比例 0%
- ✅ [D] TaiwanStockMonthRevenue.date: NULL 比例 0%
- ✅ [D] TaiwanStockMonthRevenue.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockMonthRevenue.country: NULL 比例 0%
- ✅ [D] TaiwanStockMonthRevenue.revenue: NULL 比例 0%
- ✅ [D] TaiwanStockMonthRevenue.revenue_month: NULL 比例 0%
- ✅ [D] TaiwanStockMonthRevenue.revenue_year: NULL 比例 0%
- ✅ [D] TaiwanStockMonthRevenue.create_time: NULL 比例 90%
- ✅ [D] TaiwanStockDividend.date: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.year: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.StockEarningsDistribution: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.StockStatutorySurplus: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.CashEarningsDistribution: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.CashStatutorySurplus: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.AnnouncementDate: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.AnnouncementTime: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.CashDividendPaymentDate: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.CashExDividendTradingDate: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.CashIncreaseSubscriptionRate: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.CashIncreaseSubscriptionpRrice: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.ParticipateDistributionOfTotalShares: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.RatioOfEmployeeStockDividend: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.RatioOfEmployeeStockDividendOfTotal: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.RemunerationOfDirectorsAndSupervisors: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.StockExDividendTradingDate: NULL 比例 100%
- ✅ [D] TaiwanStockDividend.TotalEmployeeCashDividend: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.TotalEmployeeStockDividend: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.TotalEmployeeStockDividendAmount: NULL 比例 0%
- ✅ [D] TaiwanStockDividend.TotalNumberOfCashCapitalIncrease: NULL 比例 0%
- ✅ [D] FredData.date: NULL 比例 0%
- ✅ [D] FredData.series_id: NULL 比例 0%
- ✅ [D] FredData.value: NULL 比例 0%
- ✅ [D] FredData.realtime_start: NULL 比例 0%
- ✅ [D] FredData.realtime_end: NULL 比例 0%
- ✅ [D] TaiwanStockInfo.stock_id: NULL 比例 0%
- ✅ [D] TaiwanStockInfo.stock_name: NULL 比例 0%
- ✅ [D] TaiwanStockInfo.industry_category: NULL 比例 0%
- ✅ [D] TaiwanStockInfo.type: NULL 比例 0%
- ✅ [D] TaiwanStockInfo.date: NULL 比例 0%

### Layer E: PK / Unique Constraint Uniqueness

- PASS=0, FAIL=11
- ❌ [E] TaiwanStockPrice.date,stock_id: 查詢失敗：UndefinedTable: relation "TaiwanStockPrice" does not exist
LINE 1: ...UNT(*), COUNT(DISTINCT ("date", "stock_id")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] TaiwanStockPriceAdj.date,stock_id: 查詢失敗：UndefinedTable: relation "TaiwanStockPriceAdj" does not exist
LINE 1: ...UNT(*), COUNT(DISTINCT ("date", "stock_id")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] TaiwanStockPER.date,stock_id: 查詢失敗：UndefinedTable: relation "TaiwanStockPER" does not exist
LINE 1: ...UNT(*), COUNT(DISTINCT ("date", "stock_id")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] TaiwanStockInstitutionalInvestorsBuySell.date,stock_id,name: 查詢失敗：UndefinedTable: relation "TaiwanStockInstitutionalInvestorsBuySell" does not exist
LINE 1: ...COUNT(DISTINCT ("date", "stock_id", "name")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] TaiwanStockMarginPurchaseShortSale.date,stock_id: 查詢失敗：UndefinedTable: relation "TaiwanStockMarginPurchaseShortSale" does not exist
LINE 1: ...UNT(*), COUNT(DISTINCT ("date", "stock_id")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] TaiwanStockShareholding.date,stock_id: 查詢失敗：UndefinedTable: relation "TaiwanStockShareholding" does not exist
LINE 1: ...UNT(*), COUNT(DISTINCT ("date", "stock_id")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] TaiwanStockFinancialStatements.date,stock_id,type,origin_name: 查詢失敗：UndefinedTable: relation "TaiwanStockFinancialStatements" does not exist
LINE 1: ...("date", "stock_id", "type", "origin_name")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] TaiwanStockMonthRevenue.date,stock_id: 查詢失敗：UndefinedTable: relation "TaiwanStockMonthRevenue" does not exist
LINE 1: ...UNT(*), COUNT(DISTINCT ("date", "stock_id")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] TaiwanStockDividend.date,stock_id,year: 查詢失敗：UndefinedTable: relation "TaiwanStockDividend" does not exist
LINE 1: ...COUNT(DISTINCT ("date", "stock_id", "year")) FROM "TaiwanSto...
                                                             ^

- ❌ [E] FredData.date,series_id: 查詢失敗：UndefinedTable: relation "FredData" does not exist
LINE 1: ...NT(*), COUNT(DISTINCT ("date", "series_id")) FROM "FredData"
                                                             ^

- ❌ [E] TaiwanStockInfo.stock_id: 查詢失敗：UndefinedTable: relation "TaiwanStockInfo" does not exist
LINE 1: ...ELECT COUNT(*), COUNT(DISTINCT ("stock_id")) FROM "TaiwanSto...
                                                             ^


### Layer F: Duplicate Row Detection

- PASS=0, FAIL=13
- ❌ [F] pipeline_execution_log.row: 查詢失敗：UndefinedTable: relation "pipeline_execution_log" does not exist
LINE 1: ...ime", "status", "duration_ms", "error_msg")) FROM "pipeline_...
                                                             ^

- ❌ [F] data_audit_log.row: 查詢失敗：UndefinedTable: relation "data_audit_log" does not exist
LINE 1: ...action_type", "rows_affected", "timestamp")) FROM "data_audi...
                                                             ^

- ❌ [F] TaiwanStockPrice.row: 查詢失敗：UndefinedTable: relation "TaiwanStockPrice" does not exist
LINE 1: ...in", "close", "spread", "Trading_turnover")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] TaiwanStockPriceAdj.row: 查詢失敗：UndefinedTable: relation "TaiwanStockPriceAdj" does not exist
LINE 1: ...in", "close", "spread", "Trading_turnover")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] TaiwanStockPER.row: 查詢失敗：UndefinedTable: relation "TaiwanStockPER" does not exist
LINE 1: ..."stock_id", "dividend_yield", "PER", "PBR")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] TaiwanStockInstitutionalInvestorsBuySell.row: 查詢失敗：UndefinedTable: relation "TaiwanStockInstitutionalInvestorsBuySell" does not exist
LINE 1: ...("date", "stock_id", "buy", "name", "sell")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] TaiwanStockMarginPurchaseShortSale.row: 查詢失敗：UndefinedTable: relation "TaiwanStockMarginPurchaseShortSale" does not exist
LINE 1: ...rdayBalance", "OffsetLoanAndShort", "Note")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] TaiwanStockShareholding.row: 查詢失敗：UndefinedTable: relation "TaiwanStockShareholding" does not exist
LINE 1: ...LimitRatio", "RecentlyDeclareDate", "note")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] TaiwanStockFinancialStatements.row: 查詢失敗：UndefinedTable: relation "TaiwanStockFinancialStatements" does not exist
LINE 1: ..."stock_id", "type", "value", "origin_name")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] TaiwanStockMonthRevenue.row: 查詢失敗：UndefinedTable: relation "TaiwanStockMonthRevenue" does not exist
LINE 1: ...enue_month", "revenue_year", "create_time")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] TaiwanStockDividend.row: 查詢失敗：UndefinedTable: relation "TaiwanStockDividend" does not exist
LINE 1: ...mount", "TotalNumberOfCashCapitalIncrease")) FROM "TaiwanSto...
                                                             ^

- ❌ [F] FredData.row: 查詢失敗：UndefinedTable: relation "FredData" does not exist
LINE 1: ... "value", "realtime_start", "realtime_end")) FROM "FredData"
                                                             ^

- ❌ [F] TaiwanStockInfo.row: 查詢失敗：UndefinedTable: relation "TaiwanStockInfo" does not exist
LINE 1: ...name", "industry_category", "type", "date")) FROM "TaiwanSto...
                                                             ^


### Layer G: Date Series Continuity

- PASS=0, FAIL=11
- ❌ [G] TaiwanStockPrice.date: 查詢失敗：UndefinedTable: relation "TaiwanStockPrice" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] TaiwanStockPriceAdj.date: 查詢失敗：UndefinedTable: relation "TaiwanStockPriceAdj" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] TaiwanStockPER.date: 查詢失敗：UndefinedTable: relation "TaiwanStockPER" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] TaiwanStockInstitutionalInvestorsBuySell.date: 查詢失敗：UndefinedTable: relation "TaiwanStockInstitutionalInvestorsBuySell" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] TaiwanStockMarginPurchaseShortSale.date: 查詢失敗：UndefinedTable: relation "TaiwanStockMarginPurchaseShortSale" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] TaiwanStockShareholding.date: 查詢失敗：UndefinedTable: relation "TaiwanStockShareholding" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] TaiwanStockFinancialStatements.date: 查詢失敗：UndefinedTable: relation "TaiwanStockFinancialStatements" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] TaiwanStockMonthRevenue.date: 查詢失敗：UndefinedTable: relation "TaiwanStockMonthRevenue" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] TaiwanStockDividend.date: 查詢失敗：UndefinedTable: relation "TaiwanStockDividend" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^

- ❌ [G] FredData.date: 查詢失敗：UndefinedTable: relation "FredData" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "FredData"
                                                             ^

- ❌ [G] TaiwanStockInfo.date: 查詢失敗：UndefinedTable: relation "TaiwanStockInfo" does not exist
LINE 1: ...T MIN(date), MAX(date), COUNT(DISTINCT date) FROM "TaiwanSto...
                                                             ^


### Layer H: Referential Integrity

- PASS=0, FAIL=9
- ❌ [H] TaiwanStockPrice.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^

- ❌ [H] TaiwanStockPriceAdj.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^

- ❌ [H] TaiwanStockPER.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^

- ❌ [H] TaiwanStockInstitutionalInvestorsBuySell.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^

- ❌ [H] TaiwanStockMarginPurchaseShortSale.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^

- ❌ [H] TaiwanStockShareholding.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^

- ❌ [H] TaiwanStockFinancialStatements.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^

- ❌ [H] TaiwanStockMonthRevenue.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^

- ❌ [H] TaiwanStockDividend.stock_id: TaiwanStockInfo 查詢失敗：relation "TaiwanStockInfo" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockInfo"
                             ^


### Layer I: Value Range Sanity

- PASS=0, FAIL=8
- ❌ [I] TaiwanStockPrice.Trading_Volume: 查詢失敗：UndefinedTable: relation "TaiwanStockPrice" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockPrice" WHERE "Trading_Volum...
                             ^

- ❌ [I] TaiwanStockPrice.close: 查詢失敗：UndefinedTable: relation "TaiwanStockPrice" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockPrice" WHERE "close" < 0
                             ^

- ❌ [I] TaiwanStockPrice.close: 查詢失敗：UndefinedTable: relation "TaiwanStockPrice" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockPrice" WHERE "close" > 1000...
                             ^

- ❌ [I] TaiwanStockPriceAdj.Trading_Volume: 查詢失敗：UndefinedTable: relation "TaiwanStockPriceAdj" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockPriceAdj" WHERE "Trading_Vo...
                             ^

- ❌ [I] TaiwanStockPriceAdj.close: 查詢失敗：UndefinedTable: relation "TaiwanStockPriceAdj" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockPriceAdj" WHERE "close" < 0
                             ^

- ❌ [I] TaiwanStockPER.PBR: 查詢失敗：UndefinedTable: relation "TaiwanStockPER" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockPER" WHERE "PBR" < 0
                             ^

- ❌ [I] TaiwanStockPER.dividend_yield: 查詢失敗：UndefinedTable: relation "TaiwanStockPER" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockPER" WHERE "dividend_yield"...
                             ^

- ❌ [I] TaiwanStockMonthRevenue.revenue: 查詢失敗：UndefinedTable: relation "TaiwanStockMonthRevenue" does not exist
LINE 1: SELECT COUNT(*) FROM "TaiwanStockMonthRevenue" WHERE "revenu...
                             ^

