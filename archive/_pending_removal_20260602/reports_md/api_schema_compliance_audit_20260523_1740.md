# API Schema Compliance Audit Report (v0.6)

**執行日期**: 2026-05-23 17:40:38
**對應憲章**: 系統架構大憲章_v6.1.0.md / §3.2A / §14.7-AJ

## 環境快照

- sample_size: 100
- include_fred: True
- target_table: ALL
- skip_api_probe: False
- active_layers: A,B,C,D,E,F,G,H,I

## 9 層結果摘要

| Layer | 名稱 | PASS | FAIL | 狀態 |
|---|---|---|---|---|
| A | DDL ↔ DB Physical Consistency | 119 | 0 | PASS |
| B | API Sample ↔ DDL Type Compatibility | 102 | 0 | PASS |
| C | API Sample Length / Precision Range | 83 | 0 | PASS |
| D | NULL Ratio Sanity | 103 | 0 | PASS |
| E | PK / Unique Constraint Uniqueness | 11 | 0 | PASS |
| F | Duplicate Row Detection | 13 | 0 | PASS |
| G | Date Series Continuity | 14 | 0 | PASS |
| H | Referential Integrity | 9 | 0 | PASS |
| I | Value Range Sanity | 7 | 0 | PASS |

**verdict**: PERFECT
**latency**: 121812.9 ms

## 9 層詳細紀錄

### Layer A: DDL ↔ DB Physical Consistency

- PASS=119, FAIL=0
- ✅ [A] pipeline_execution_log.id: integer 對齊
- ✅ [A] pipeline_execution_log.task_name: character varying 對齊
- ✅ [A] pipeline_execution_log.category: character varying 對齊
- ✅ [A] pipeline_execution_log.stock_id: character varying 對齊
- ✅ [A] pipeline_execution_log.start_time: timestamp without time zone 對齊
- ✅ [A] pipeline_execution_log.end_time: timestamp without time zone 對齊
- ✅ [A] pipeline_execution_log.status: character varying 對齊
- ✅ [A] pipeline_execution_log.duration_ms: bigint 對齊
- ✅ [A] pipeline_execution_log.error_msg: text 對齊
- ✅ [A] data_audit_log.id: integer 對齊
- ✅ [A] data_audit_log.table_name: character varying 對齊
- ✅ [A] data_audit_log.stock_id: character varying 對齊
- ✅ [A] data_audit_log.data_date: date 對齊
- ✅ [A] data_audit_log.action_type: character varying 對齊
- ✅ [A] data_audit_log.rows_affected: integer 對齊
- ✅ [A] data_audit_log.timestamp: timestamp without time zone 對齊
- ✅ [A] TaiwanStockPrice.date: date 對齊
- ✅ [A] TaiwanStockPrice.stock_id: character varying 對齊
- ✅ [A] TaiwanStockPrice.Trading_Volume: numeric 對齊
- ✅ [A] TaiwanStockPrice.Trading_money: numeric 對齊
- ✅ [A] TaiwanStockPrice.open: numeric 對齊
- ✅ [A] TaiwanStockPrice.max: numeric 對齊
- ✅ [A] TaiwanStockPrice.min: numeric 對齊
- ✅ [A] TaiwanStockPrice.close: numeric 對齊
- ✅ [A] TaiwanStockPrice.spread: numeric 對齊
- ✅ [A] TaiwanStockPrice.Trading_turnover: numeric 對齊
- ✅ [A] TaiwanStockPriceAdj.date: date 對齊
- ✅ [A] TaiwanStockPriceAdj.stock_id: character varying 對齊
- ✅ [A] TaiwanStockPriceAdj.Trading_Volume: numeric 對齊
- ✅ [A] TaiwanStockPriceAdj.Trading_money: numeric 對齊
- ✅ [A] TaiwanStockPriceAdj.open: numeric 對齊
- ✅ [A] TaiwanStockPriceAdj.max: numeric 對齊
- ✅ [A] TaiwanStockPriceAdj.min: numeric 對齊
- ✅ [A] TaiwanStockPriceAdj.close: numeric 對齊
- ✅ [A] TaiwanStockPriceAdj.spread: numeric 對齊
- ✅ [A] TaiwanStockPriceAdj.Trading_turnover: numeric 對齊
- ✅ [A] TaiwanStockPER.date: date 對齊
- ✅ [A] TaiwanStockPER.stock_id: character varying 對齊
- ✅ [A] TaiwanStockPER.dividend_yield: numeric 對齊
- ✅ [A] TaiwanStockPER.PER: numeric 對齊
- ✅ [A] TaiwanStockPER.PBR: numeric 對齊
- ✅ [A] TaiwanStockInstitutionalInvestorsBuySell.date: date 對齊
- ✅ [A] TaiwanStockInstitutionalInvestorsBuySell.stock_id: character varying 對齊
- ✅ [A] TaiwanStockInstitutionalInvestorsBuySell.buy: numeric 對齊
- ✅ [A] TaiwanStockInstitutionalInvestorsBuySell.name: character varying 對齊
- ✅ [A] TaiwanStockInstitutionalInvestorsBuySell.sell: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.date: date 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.stock_id: character varying 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseBuy: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseSell: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseCashRepayment: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseLimit: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseTodayBalance: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.MarginPurchaseYesterdayBalance: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleBuy: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleSell: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleCashRepayment: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleLimit: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleTodayBalance: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.ShortSaleYesterdayBalance: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.OffsetLoanAndShort: numeric 對齊
- ✅ [A] TaiwanStockMarginPurchaseShortSale.Note: character varying 對齊
- ✅ [A] TaiwanStockShareholding.date: date 對齊
- ✅ [A] TaiwanStockShareholding.stock_id: character varying 對齊
- ✅ [A] TaiwanStockShareholding.stock_name: character varying 對齊
- ✅ [A] TaiwanStockShareholding.InternationalCode: character varying 對齊
- ✅ [A] TaiwanStockShareholding.ForeignInvestmentRemainingShares: numeric 對齊
- ✅ [A] TaiwanStockShareholding.ForeignInvestmentShares: numeric 對齊
- ✅ [A] TaiwanStockShareholding.ForeignInvestmentRemainRatio: numeric 對齊
- ✅ [A] TaiwanStockShareholding.ForeignInvestmentSharesRatio: numeric 對齊
- ✅ [A] TaiwanStockShareholding.NumberOfSharesIssued: numeric 對齊
- ✅ [A] TaiwanStockShareholding.ForeignInvestmentUpperLimitRatio: numeric 對齊
- ✅ [A] TaiwanStockShareholding.ChineseInvestmentUpperLimitRatio: numeric 對齊
- ✅ [A] TaiwanStockShareholding.RecentlyDeclareDate: date 對齊
- ✅ [A] TaiwanStockShareholding.note: character varying 對齊
- ✅ [A] TaiwanStockFinancialStatements.date: date 對齊
- ✅ [A] TaiwanStockFinancialStatements.stock_id: character varying 對齊
- ✅ [A] TaiwanStockFinancialStatements.type: character varying 對齊
- ✅ [A] TaiwanStockFinancialStatements.value: numeric 對齊
- ✅ [A] TaiwanStockFinancialStatements.origin_name: character varying 對齊
- ✅ [A] TaiwanStockMonthRevenue.date: date 對齊
- ✅ [A] TaiwanStockMonthRevenue.stock_id: character varying 對齊
- ✅ [A] TaiwanStockMonthRevenue.country: character varying 對齊
- ✅ [A] TaiwanStockMonthRevenue.revenue: numeric 對齊
- ✅ [A] TaiwanStockMonthRevenue.revenue_month: numeric 對齊
- ✅ [A] TaiwanStockMonthRevenue.revenue_year: numeric 對齊
- ✅ [A] TaiwanStockMonthRevenue.create_time: date 對齊
- ✅ [A] TaiwanStockDividend.date: date 對齊
- ✅ [A] TaiwanStockDividend.stock_id: character varying 對齊
- ✅ [A] TaiwanStockDividend.year: character varying 對齊
- ✅ [A] TaiwanStockDividend.StockEarningsDistribution: numeric 對齊
- ✅ [A] TaiwanStockDividend.StockStatutorySurplus: numeric 對齊
- ✅ [A] TaiwanStockDividend.CashEarningsDistribution: numeric 對齊
- ✅ [A] TaiwanStockDividend.CashStatutorySurplus: numeric 對齊
- ✅ [A] TaiwanStockDividend.AnnouncementDate: date 對齊
- ✅ [A] TaiwanStockDividend.AnnouncementTime: character varying 對齊
- ✅ [A] TaiwanStockDividend.CashDividendPaymentDate: date 對齊
- ✅ [A] TaiwanStockDividend.CashExDividendTradingDate: date 對齊
- ✅ [A] TaiwanStockDividend.CashIncreaseSubscriptionRate: numeric 對齊
- ✅ [A] TaiwanStockDividend.CashIncreaseSubscriptionpRrice: numeric 對齊
- ✅ [A] TaiwanStockDividend.ParticipateDistributionOfTotalShares: numeric 對齊
- ✅ [A] TaiwanStockDividend.RatioOfEmployeeStockDividend: numeric 對齊
- ✅ [A] TaiwanStockDividend.RatioOfEmployeeStockDividendOfTotal: numeric 對齊
- ✅ [A] TaiwanStockDividend.RemunerationOfDirectorsAndSupervisors: numeric 對齊
- ✅ [A] TaiwanStockDividend.StockExDividendTradingDate: date 對齊
- ✅ [A] TaiwanStockDividend.TotalEmployeeCashDividend: numeric 對齊
- ✅ [A] TaiwanStockDividend.TotalEmployeeStockDividend: numeric 對齊
- ✅ [A] TaiwanStockDividend.TotalEmployeeStockDividendAmount: numeric 對齊
- ✅ [A] TaiwanStockDividend.TotalNumberOfCashCapitalIncrease: numeric 對齊
- ✅ [A] FredData.date: date 對齊
- ✅ [A] FredData.series_id: character varying 對齊
- ✅ [A] FredData.value: numeric 對齊
- ✅ [A] FredData.realtime_start: date 對齊
- ✅ [A] FredData.realtime_end: date 對齊
- ✅ [A] TaiwanStockInfo.stock_id: character varying 對齊
- ✅ [A] TaiwanStockInfo.stock_name: character varying 對齊
- ✅ [A] TaiwanStockInfo.industry_category: character varying 對齊
- ✅ [A] TaiwanStockInfo.type: character varying 對齊
- ✅ [A] TaiwanStockInfo.date: date 對齊

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

- PASS=11, FAIL=0
- ✅ [E] TaiwanStockPrice.date,stock_id: unique 一致：100507 (sampled(100507/10509879, 0.951%))
- ✅ [E] TaiwanStockPriceAdj.date,stock_id: unique 一致：99845 (sampled(99845/10507832, 0.952%))
- ✅ [E] TaiwanStockPER.date,stock_id: unique 一致：100042 (sampled(100042/7346851, 1.361%))
- ✅ [E] TaiwanStockInstitutionalInvestorsBuySell.date,stock_id,name: unique 一致：99979 (sampled(99979/25014641, 0.400%))
- ✅ [E] TaiwanStockMarginPurchaseShortSale.date,stock_id: unique 一致：99747 (sampled(99747/7717343, 1.296%))
- ✅ [E] TaiwanStockShareholding.date,stock_id: unique 一致：100190 (sampled(100190/8372856, 1.194%))
- ✅ [E] TaiwanStockFinancialStatements.date,stock_id,type,origin_name: unique 一致：99975 (sampled(99975/2663205, 3.755%))
- ✅ [E] TaiwanStockMonthRevenue.date,stock_id: unique 一致：460187 (full(460187))
- ✅ [E] TaiwanStockDividend.date,stock_id,year: unique 一致：29312 (full(29312))
- ✅ [E] FredData.date,series_id: unique 一致：48879 (full(48879))
- ✅ [E] TaiwanStockInfo.stock_id: unique 一致：2803 (full(2803))

### Layer F: Duplicate Row Detection

- PASS=13, FAIL=0
- ✅ [F] pipeline_execution_log.row: 無重複：24 (full(24))
- ✅ [F] data_audit_log.row: 無重複：22083 (full(22083))
- ✅ [F] TaiwanStockPrice.row: 無重複：99241 (sampled(99241/10509879, 0.951%))
- ✅ [F] TaiwanStockPriceAdj.row: 無重複：100032 (sampled(100032/10507832, 0.952%))
- ✅ [F] TaiwanStockPER.row: 無重複：100570 (sampled(100570/7346851, 1.361%))
- ✅ [F] TaiwanStockInstitutionalInvestorsBuySell.row: 無重複：99600 (sampled(99600/25014641, 0.400%))
- ✅ [F] TaiwanStockMarginPurchaseShortSale.row: 無重複：99671 (sampled(99671/7717343, 1.296%))
- ✅ [F] TaiwanStockShareholding.row: 無重複：100172 (sampled(100172/8372856, 1.194%))
- ✅ [F] TaiwanStockFinancialStatements.row: 無重複：100532 (sampled(100532/2663205, 3.755%))
- ✅ [F] TaiwanStockMonthRevenue.row: 無重複：460187 (full(460187))
- ✅ [F] TaiwanStockDividend.row: 無重複：29312 (full(29312))
- ✅ [F] FredData.row: 無重複：48879 (full(48879))
- ✅ [F] TaiwanStockInfo.row: 無重複：2803 (full(2803))

### Layer G: Date Series Continuity

- PASS=14, FAIL=0
- ✅ [G] TaiwanStockPrice.date: Daily 連續性正常：1992-01-04~2026-05-22, 8768 distinct dates (~8970 預期工作日)
- ✅ [G] TaiwanStockPriceAdj.date: Daily 連續性正常：1992-01-04~2026-05-22, 8768 distinct dates (~8970 預期工作日)
- ✅ [G] TaiwanStockPER.date: Daily 連續性正常：2005-09-02~2026-05-22, 5097 distinct dates (~5405 預期工作日)
- ✅ [G] TaiwanStockInstitutionalInvestorsBuySell.date: Daily 連續性正常：2005-01-03~2026-05-22, 5259 distinct dates (~5578 預期工作日)
- ✅ [G] TaiwanStockMarginPurchaseShortSale.date: Daily 連續性正常：2001-01-05~2026-05-22, 6250 distinct dates (~6620 預期工作日)
- ✅ [G] TaiwanStockShareholding.date: Daily 連續性正常：2004-02-12~2026-05-22, 5527 distinct dates (~5811 預期工作日)
- ✅ [G] TaiwanStockFinancialStatements.date: Quarterly 連續性正常：1990-03-31~2026-03-31, 145 季 (36.0 年；預期 ~144 ± 36)
- ✅ [G] TaiwanStockMonthRevenue.date: Monthly 連續性正常：2002-02-01~2026-05-01, 292 月 (24.2 年；預期 ~290 ± 24)
- ✅ [G] TaiwanStockDividend.date: Event-Driven 表（§3.2A.G 略過 continuity check；不定期事件型）
- ✅ [G] FredData/DFF.date: FRED Daily 連續性正常：1954-07-01~2026-05-21, 26258 dates
- ✅ [G] FredData/T10Y2Y.date: FRED Daily 連續性正常：1976-06-01~2026-05-22, 12491 dates
- ✅ [G] FredData/UNRATE.date: FRED Monthly 連續性正常：1948-01-01~2026-04-01, 939 月 (78.3 年；預期 ~939 ± 78)
- ✅ [G] FredData/VIXCLS.date: FRED Daily 連續性正常：1990-01-02~2026-05-21, 9191 dates
- ✅ [G] TaiwanStockInfo.date: Metadata Snapshot 表（§3.2A.G 略過 continuity check；無時序連續性）

### Layer H: Referential Integrity

- PASS=9, FAIL=0
- ✅ [H] TaiwanStockPrice.stock_id: 所有 stock_id ∈ TaiwanStockInfo
- ✅ [H] TaiwanStockPriceAdj.stock_id: 所有 stock_id ∈ TaiwanStockInfo
- ✅ [H] TaiwanStockPER.stock_id: 所有 stock_id ∈ TaiwanStockInfo
- ✅ [H] TaiwanStockInstitutionalInvestorsBuySell.stock_id: 所有 stock_id ∈ TaiwanStockInfo
- ✅ [H] TaiwanStockMarginPurchaseShortSale.stock_id: 所有 stock_id ∈ TaiwanStockInfo
- ✅ [H] TaiwanStockShareholding.stock_id: 所有 stock_id ∈ TaiwanStockInfo
- ✅ [H] TaiwanStockFinancialStatements.stock_id: 所有 stock_id ∈ TaiwanStockInfo
- ✅ [H] TaiwanStockMonthRevenue.stock_id: 所有 stock_id ∈ TaiwanStockInfo
- ✅ [H] TaiwanStockDividend.stock_id: 所有 stock_id ∈ TaiwanStockInfo

### Layer I: Value Range Sanity

- PASS=7, FAIL=0
- ✅ [I] TaiwanStockPrice.Trading_Volume: Trading_Volume 為負：0 rows
- ✅ [I] TaiwanStockPrice.close: close 為負：0 rows
- ✅ [I] TaiwanStockPrice.close: close > 100000（異常大）：0 rows
- ✅ [I] TaiwanStockPriceAdj.Trading_Volume: Trading_Volume 為負：0 rows
- ✅ [I] TaiwanStockPriceAdj.close: close 為負：0 rows
- ✅ [I] TaiwanStockPER.PBR: PBR < 0：0 rows
- ✅ [I] TaiwanStockPER.dividend_yield: dividend_yield < 0：0 rows
