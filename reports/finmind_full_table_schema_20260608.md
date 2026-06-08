# FinMind + FRED 通用 Ingester 完整 Table Schema Catalog（80 表）

**日期**:2026-06-08 | **治權**:主憲章 §14.7-DJ §二 之 companion 完整檔（per-column schema SSOT）| **source**:§一.10 實際 FinMind/FRED API 回應 build 結果
**大小寫**:表名/欄位名與 FinMind/FRED API 逐字一致（§14.7-CC,generic ingester 雙引號封裝保留大小寫）

> 長表 `TaiwanStockFinancialStatements`(17 科目)/`TaiwanStockBalanceSheet`(101 科目)/`TaiwanStockCashFlowsStatement` 之 `type` 科目枚舉(origin_name 中文)見主憲章 §14.7-DJ §二。

---

**完整 catalog:80 表有完整 schema**（FinMind Taiwan 66 + 非Taiwan 13 + FRED 1）。表名/欄位名 = FinMind/FRED API 確切大小寫逐字鏡像(§14.7-CC);型別由 generic auto-schema 依實際 API 回應值推導(§一.10);intraday(日以下)排除 9、待解 4。

## FinMind — Taiwan
#### `TaiwanBusinessIndicator` (9 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `leading` | NUMERIC(20,6) |
| `leading_notrend` | NUMERIC(20,6) |
| `coincident` | NUMERIC(20,6) |
| `coincident_notrend` | NUMERIC(20,6) |
| `lagging` | NUMERIC(20,6) |
| `lagging_notrend` | NUMERIC(20,6) |
| `monitoring` | NUMERIC(20,6) |
| `monitoring_color` | VARCHAR(100) |

#### `TaiwanDailyShortSaleBalances` (15 欄)
| 欄位 | 型態及大小 |
|---|---|
| `stock_id` | VARCHAR(100) |
| `MarginShortSalesPreviousDayBalance` | NUMERIC(20,6) |
| `MarginShortSalesShortSales` | NUMERIC(20,6) |
| `MarginShortSalesShortCovering` | NUMERIC(20,6) |
| `MarginShortSalesStockRedemption` | NUMERIC(20,6) |
| `MarginShortSalesCurrentDayBalance` | NUMERIC(20,6) |
| `MarginShortSalesQuota` | NUMERIC(20,6) |
| `SBLShortSalesPreviousDayBalance` | NUMERIC(20,6) |
| `SBLShortSalesShortSales` | NUMERIC(20,6) |
| `SBLShortSalesReturns` | NUMERIC(20,6) |
| `SBLShortSalesAdjustments` | NUMERIC(20,6) |
| `SBLShortSalesCurrentDayBalance` | NUMERIC(20,6) |
| `SBLShortSalesQuota` | NUMERIC(20,6) |
| `SBLShortSalesShortCovering` | NUMERIC(20,6) |
| `date` | DATE |

#### `TaiwanExchangeRate` (6 欄 · 最早資料 2006-01-02)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `currency` | VARCHAR(100) |
| `cash_buy` | NUMERIC(20,6) |
| `cash_sell` | NUMERIC(20,6) |
| `spot_buy` | NUMERIC(20,6) |
| `spot_sell` | NUMERIC(20,6) |

#### `TaiwanFutOptDailyInfo` (3 欄)
| 欄位 | 型態及大小 |
|---|---|
| `code` | VARCHAR(100) |
| `type` | VARCHAR(100) |
| `name` | VARCHAR(100) |

#### `TaiwanFutOptInstitutionalInvestors` (11 欄)
| 欄位 | 型態及大小 |
|---|---|
| `name` | VARCHAR(100) |
| `date` | DATE |
| `institutional_investors` | VARCHAR(100) |
| `long_deal_volume` | NUMERIC(20,6) |
| `long_deal_amount` | NUMERIC(20,6) |
| `short_deal_volume` | NUMERIC(20,6) |
| `short_deal_amount` | NUMERIC(20,6) |
| `long_open_interest_balance_volume` | NUMERIC(20,6) |
| `long_open_interest_balance_amount` | NUMERIC(20,6) |
| `short_open_interest_balance_volume` | NUMERIC(20,6) |
| `short_open_interest_balance_amount` | NUMERIC(20,6) |

#### `TaiwanFuturesDaily` (13 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `futures_id` | VARCHAR(100) |
| `contract_date` | VARCHAR(100) |
| `open` | NUMERIC(20,6) |
| `max` | NUMERIC(20,6) |
| `min` | NUMERIC(20,6) |
| `close` | NUMERIC(20,6) |
| `spread` | NUMERIC(20,6) |
| `spread_per` | NUMERIC(20,6) |
| `volume` | NUMERIC(20,6) |
| `settlement_price` | NUMERIC(20,6) |
| `open_interest` | NUMERIC(20,6) |
| `trading_session` | VARCHAR(100) |

#### `TaiwanFuturesDealerTradingVolumeDaily` (6 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `dealer_code` | VARCHAR(100) |
| `dealer_name` | VARCHAR(100) |
| `futures_id` | VARCHAR(100) |
| `volume` | NUMERIC(20,6) |
| `is_after_hour` | VARCHAR(100) |

#### `TaiwanFuturesFinalSettlementPrice` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `contract_month` | VARCHAR(100) |
| `futures_type` | VARCHAR(100) |
| `futures_id` | VARCHAR(100) |
| `futures_name` | VARCHAR(100) |
| `settlement_price` | NUMERIC(20,6) |
| `underlying_code` | VARCHAR(100) |
| `notional_value` | NUMERIC(20,6) |

#### `TaiwanFuturesInstitutionalInvestors` (11 欄)
| 欄位 | 型態及大小 |
|---|---|
| `futures_id` | VARCHAR(100) |
| `date` | DATE |
| `institutional_investors` | VARCHAR(100) |
| `long_deal_volume` | NUMERIC(20,6) |
| `long_deal_amount` | NUMERIC(20,6) |
| `short_deal_volume` | NUMERIC(20,6) |
| `short_deal_amount` | NUMERIC(20,6) |
| `long_open_interest_balance_volume` | NUMERIC(20,6) |
| `long_open_interest_balance_amount` | NUMERIC(20,6) |
| `short_open_interest_balance_volume` | NUMERIC(20,6) |
| `short_open_interest_balance_amount` | NUMERIC(20,6) |

#### `TaiwanFuturesInstitutionalInvestorsAfterHours` (7 欄)
| 欄位 | 型態及大小 |
|---|---|
| `futures_id` | VARCHAR(100) |
| `date` | DATE |
| `institutional_investors` | VARCHAR(100) |
| `long_deal_volume` | NUMERIC(20,6) |
| `long_deal_amount` | NUMERIC(20,6) |
| `short_deal_volume` | NUMERIC(20,6) |
| `short_deal_amount` | NUMERIC(20,6) |

#### `TaiwanFuturesOpenInterestLargeTraders` (21 欄)
| 欄位 | 型態及大小 |
|---|---|
| `name` | VARCHAR(100) |
| `contract_type` | VARCHAR(100) |
| `buy_top5_trader_open_interest` | NUMERIC(20,6) |
| `buy_top5_trader_open_interest_per` | NUMERIC(20,6) |
| `buy_top10_trader_open_interest` | NUMERIC(20,6) |
| `buy_top10_trader_open_interest_per` | NUMERIC(20,6) |
| `sell_top5_trader_open_interest` | NUMERIC(20,6) |
| `sell_top5_trader_open_interest_per` | NUMERIC(20,6) |
| `sell_top10_trader_open_interest` | NUMERIC(20,6) |
| `sell_top10_trader_open_interest_per` | NUMERIC(20,6) |
| `market_open_interest` | NUMERIC(20,6) |
| `buy_top5_specific_open_interest` | NUMERIC(20,6) |
| `buy_top5_specific_open_interest_per` | NUMERIC(20,6) |
| `buy_top10_specific_open_interest` | NUMERIC(20,6) |
| `buy_top10_specific_open_interest_per` | NUMERIC(20,6) |
| `sell_top5_specific_open_interest` | NUMERIC(20,6) |
| `sell_top5_specific_open_interest_per` | NUMERIC(20,6) |
| `sell_top10_specific_open_interest` | NUMERIC(20,6) |
| `sell_top10_specific_open_interest_per` | NUMERIC(20,6) |
| `date` | DATE |
| `futures_id` | VARCHAR(100) |

#### `TaiwanFuturesSpreadTrading` (14 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `futures_id` | VARCHAR(100) |
| `contract_date` | VARCHAR(100) |
| `open` | NUMERIC(20,6) |
| `max` | NUMERIC(20,6) |
| `min` | NUMERIC(20,6) |
| `close` | NUMERIC(20,6) |
| `best_bid` | NUMERIC(20,6) |
| `best_ask` | NUMERIC(20,6) |
| `historical_max` | NUMERIC(20,6) |
| `historical_min` | NUMERIC(20,6) |
| `spread_to_spread_volume` | NUMERIC(20,6) |
| `spread_to_single_volume` | NUMERIC(20,6) |
| `trading_session` | VARCHAR(100) |

#### `TaiwanOptionDaily` (13 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `option_id` | VARCHAR(100) |
| `contract_date` | VARCHAR(100) |
| `strike_price` | NUMERIC(20,6) |
| `call_put` | VARCHAR(100) |
| `open` | NUMERIC(20,6) |
| `max` | NUMERIC(20,6) |
| `min` | NUMERIC(20,6) |
| `close` | NUMERIC(20,6) |
| `volume` | NUMERIC(20,6) |
| `settlement_price` | NUMERIC(20,6) |
| `open_interest` | NUMERIC(20,6) |
| `trading_session` | VARCHAR(100) |

#### `TaiwanOptionDealerTradingVolumeDaily` (6 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `dealer_code` | VARCHAR(100) |
| `dealer_name` | VARCHAR(100) |
| `option_id` | VARCHAR(100) |
| `volume` | NUMERIC(20,6) |
| `is_after_hour` | VARCHAR(100) |

#### `TaiwanOptionFinalSettlementPrice` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `contract_month` | VARCHAR(100) |
| `option_type` | VARCHAR(100) |
| `option_id` | VARCHAR(100) |
| `option_name` | VARCHAR(100) |
| `settlement_price` | NUMERIC(20,6) |
| `underlying_code` | VARCHAR(100) |
| `notional_value` | NUMERIC(20,6) |

#### `TaiwanOptionInstitutionalInvestors` (12 欄)
| 欄位 | 型態及大小 |
|---|---|
| `option_id` | VARCHAR(100) |
| `date` | DATE |
| `call_put` | VARCHAR(100) |
| `institutional_investors` | VARCHAR(100) |
| `long_deal_volume` | NUMERIC(20,6) |
| `long_deal_amount` | NUMERIC(20,6) |
| `short_deal_volume` | NUMERIC(20,6) |
| `short_deal_amount` | NUMERIC(20,6) |
| `long_open_interest_balance_volume` | NUMERIC(20,6) |
| `long_open_interest_balance_amount` | NUMERIC(20,6) |
| `short_open_interest_balance_volume` | NUMERIC(20,6) |
| `short_open_interest_balance_amount` | NUMERIC(20,6) |

#### `TaiwanOptionInstitutionalInvestorsAfterHours` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `option_id` | VARCHAR(100) |
| `date` | DATE |
| `call_put` | VARCHAR(100) |
| `institutional_investors` | VARCHAR(100) |
| `long_deal_volume` | NUMERIC(20,6) |
| `long_deal_amount` | NUMERIC(20,6) |
| `short_deal_volume` | NUMERIC(20,6) |
| `short_deal_amount` | NUMERIC(20,6) |

#### `TaiwanOptionOpenInterestLargeTraders` (22 欄)
| 欄位 | 型態及大小 |
|---|---|
| `contract_type` | VARCHAR(100) |
| `buy_top5_trader_open_interest` | NUMERIC(20,6) |
| `buy_top5_trader_open_interest_per` | NUMERIC(20,6) |
| `buy_top10_trader_open_interest` | NUMERIC(20,6) |
| `buy_top10_trader_open_interest_per` | NUMERIC(20,6) |
| `sell_top5_trader_open_interest` | NUMERIC(20,6) |
| `sell_top5_trader_open_interest_per` | NUMERIC(20,6) |
| `sell_top10_trader_open_interest` | NUMERIC(20,6) |
| `sell_top10_trader_open_interest_per` | NUMERIC(20,6) |
| `market_open_interest` | NUMERIC(20,6) |
| `buy_top5_specific_open_interest` | NUMERIC(20,6) |
| `buy_top5_specific_open_interest_per` | NUMERIC(20,6) |
| `buy_top10_specific_open_interest` | NUMERIC(20,6) |
| `buy_top10_specific_open_interest_per` | NUMERIC(20,6) |
| `sell_top5_specific_open_interest` | NUMERIC(20,6) |
| `sell_top5_specific_open_interest_per` | NUMERIC(20,6) |
| `sell_top10_specific_open_interest` | NUMERIC(20,6) |
| `sell_top10_specific_open_interest_per` | NUMERIC(20,6) |
| `date` | DATE |
| `put_call` | VARCHAR(100) |
| `name` | VARCHAR(100) |
| `option_id` | VARCHAR(100) |

#### `TaiwanSecuritiesTraderInfo` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `securities_trader_id` | VARCHAR(100) |
| `securities_trader` | VARCHAR(100) |
| `date` | DATE |
| `address` | VARCHAR(100) |
| `phone` | VARCHAR(100) |

#### `TaiwanStock10Year` (3 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `close` | NUMERIC(20,6) |

#### `TaiwanStockBalanceSheet` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `type` | VARCHAR(110) |
| `value` | NUMERIC(23,6) |
| `origin_name` | VARCHAR(100) |

#### `TaiwanStockBlockTrade` (6 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `trade_type` | VARCHAR(100) |
| `price` | NUMERIC(20,6) |
| `volume` | NUMERIC(20,6) |
| `trading_money` | NUMERIC(21,6) |

#### `TaiwanStockCashFlowsStatement` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `type` | VARCHAR(100) |
| `value` | NUMERIC(23,6) |
| `origin_name` | VARCHAR(100) |

#### `TaiwanStockConvertibleBondDaily` (16 欄 · 最早資料 2020-01-16)
| 欄位 | 型態及大小 |
|---|---|
| `cb_id` | NUMERIC(20,6) |
| `cb_name` | VARCHAR(100) |
| `transaction_type` | VARCHAR(100) |
| `close` | NUMERIC(20,6) |
| `change` | NUMERIC(20,6) |
| `open` | NUMERIC(20,6) |
| `max` | NUMERIC(20,6) |
| `min` | NUMERIC(20,6) |
| `no_of_transactions` | NUMERIC(20,6) |
| `unit` | NUMERIC(20,6) |
| `trading_value` | NUMERIC(20,6) |
| `avg_price` | NUMERIC(20,6) |
| `next_ref_price` | NUMERIC(20,6) |
| `next_max_limit` | NUMERIC(20,6) |
| `next_min_limit` | NUMERIC(20,6) |
| `date` | DATE |

#### `TaiwanStockConvertibleBondDailyOverview` (23 欄 · 最早資料 2020-01-16)
| 欄位 | 型態及大小 |
|---|---|
| `cb_id` | NUMERIC(20,6) |
| `cb_name` | VARCHAR(100) |
| `date` | DATE |
| `InitialDateOfConversion` | DATE |
| `DueDateOfConversion` | DATE |
| `InitialDateOfStopConversion` | VARCHAR(100) |
| `DueDateOfStopConversion` | VARCHAR(100) |
| `ConversionPrice` | NUMERIC(20,6) |
| `NextEffectiveDateOfConversionPrice` | DATE |
| `LatestInitialDateOfPut` | VARCHAR(100) |
| `LatestDueDateOfPut` | VARCHAR(100) |
| `LatestPutPrice` | NUMERIC(20,6) |
| `InitialDateOfEarlyRedemption` | DATE |
| `DueDateOfEarlyRedemption` | DATE |
| `EarlyRedemptionPrice` | NUMERIC(20,6) |
| `DateOfDelisted` | DATE |
| `IssuanceAmount` | NUMERIC(20,6) |
| `OutstandingAmount` | NUMERIC(20,6) |
| `ReferencePrice` | NUMERIC(20,6) |
| `PriceOfUnderlyingStock` | NUMERIC(20,6) |
| `InitialDateOfSuspension` | VARCHAR(100) |
| `DueDateOfSuspension` | VARCHAR(100) |
| `CouponRate` | NUMERIC(20,6) |

#### `TaiwanStockConvertibleBondInfo` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `cb_id` | NUMERIC(20,6) |
| `cb_name` | VARCHAR(100) |
| `InitialDateOfConversion` | DATE |
| `DueDateOfConversion` | DATE |
| `IssuanceAmount` | NUMERIC(20,6) |

#### `TaiwanStockConvertibleBondInstitutionalInvestors` (13 欄 · 最早資料 2020-01-16)
| 欄位 | 型態及大小 |
|---|---|
| `Foreign_Investor_Buy` | NUMERIC(20,6) |
| `Foreign_Investor_Sell` | NUMERIC(20,6) |
| `Foreign_Investor_Overbuy` | NUMERIC(20,6) |
| `Investment_Trust_Buy` | NUMERIC(20,6) |
| `Investment_Trust_Sell` | NUMERIC(20,6) |
| `Investment_Trust_Overbuy` | NUMERIC(20,6) |
| `Dealer_self_Buy` | NUMERIC(20,6) |
| `Dealer_self_Sell` | NUMERIC(20,6) |
| `Dealer_self_Overbuy` | NUMERIC(20,6) |
| `Total_Overbuy` | NUMERIC(20,6) |
| `cb_id` | NUMERIC(20,6) |
| `cb_name` | VARCHAR(100) |
| `date` | DATE |

#### `TaiwanStockDayTrading` (6 欄)
| 欄位 | 型態及大小 |
|---|---|
| `stock_id` | VARCHAR(100) |
| `date` | DATE |
| `BuyAfterSale` | VARCHAR(100) |
| `Volume` | NUMERIC(20,6) |
| `BuyAmount` | NUMERIC(21,6) |
| `SellAmount` | NUMERIC(21,6) |

#### `TaiwanStockDayTradingBorrowingFeeRate` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |
| `InvestorBorrowedShares` | NUMERIC(20,6) |
| `InvestorBorrowingFeeRate` | NUMERIC(20,6) |

#### `TaiwanStockDayTradingSuspension` (4 欄)
| 欄位 | 型態及大小 |
|---|---|
| `stock_id` | VARCHAR(100) |
| `date` | DATE |
| `end_date` | DATE |
| `reason` | VARCHAR(100) |

#### `TaiwanStockDelisting` (3 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |

#### `TaiwanStockDispositionSecuritiesPeriod` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |
| `disposition_cnt` | NUMERIC(20,6) |
| `condition` | VARCHAR(100) |
| `measure` | VARCHAR(493) |
| `period_start` | DATE |
| `period_end` | DATE |

#### `TaiwanStockDividend` (22 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `year` | VARCHAR(100) |
| `StockEarningsDistribution` | NUMERIC(20,6) |
| `StockStatutorySurplus` | NUMERIC(20,6) |
| `StockExDividendTradingDate` | VARCHAR(100) |
| `TotalEmployeeStockDividend` | NUMERIC(20,6) |
| `TotalEmployeeStockDividendAmount` | NUMERIC(20,6) |
| `RatioOfEmployeeStockDividendOfTotal` | NUMERIC(20,6) |
| `RatioOfEmployeeStockDividend` | NUMERIC(20,6) |
| `CashEarningsDistribution` | NUMERIC(20,8) |
| `CashStatutorySurplus` | NUMERIC(20,6) |
| `CashExDividendTradingDate` | DATE |
| `CashDividendPaymentDate` | DATE |
| `TotalEmployeeCashDividend` | NUMERIC(20,6) |
| `TotalNumberOfCashCapitalIncrease` | NUMERIC(20,6) |
| `CashIncreaseSubscriptionRate` | NUMERIC(20,6) |
| `CashIncreaseSubscriptionpRrice` | NUMERIC(20,6) |
| `RemunerationOfDirectorsAndSupervisors` | NUMERIC(20,6) |
| `ParticipateDistributionOfTotalShares` | NUMERIC(21,6) |
| `AnnouncementDate` | DATE |
| `AnnouncementTime` | VARCHAR(100) |

#### `TaiwanStockDividendResult` (10 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `before_price` | NUMERIC(20,6) |
| `after_price` | NUMERIC(20,6) |
| `stock_and_cache_dividend` | NUMERIC(20,6) |
| `stock_or_cache_dividend` | VARCHAR(100) |
| `max_price` | NUMERIC(20,6) |
| `min_price` | NUMERIC(20,6) |
| `open_price` | NUMERIC(20,6) |
| `reference_price` | NUMERIC(20,6) |

#### `TaiwanStockFinancialStatements` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `type` | VARCHAR(100) |
| `value` | NUMERIC(23,6) |
| `origin_name` | VARCHAR(100) |

#### `TaiwanStockGovernmentBankBuySell` (7 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `bank_name` | VARCHAR(100) |
| `buy` | NUMERIC(20,6) |
| `sell` | NUMERIC(20,6) |
| `buy_amount` | NUMERIC(20,6) |
| `sell_amount` | NUMERIC(20,6) |

#### `TaiwanStockHoldingSharesPer` (6 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `HoldingSharesLevel` | VARCHAR(100) |
| `people` | NUMERIC(20,6) |
| `percent` | NUMERIC(20,6) |
| `unit` | NUMERIC(21,6) |

#### `TaiwanStockIndustryChain` (4 欄)
| 欄位 | 型態及大小 |
|---|---|
| `stock_id` | VARCHAR(100) |
| `industry` | VARCHAR(100) |
| `sub_industry` | VARCHAR(100) |
| `date` | DATE |

#### `TaiwanStockInfo` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `industry_category` | VARCHAR(100) |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |
| `type` | VARCHAR(100) |
| `date` | DATE |

#### `TaiwanStockInfoWithWarrant` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `industry_category` | VARCHAR(100) |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |
| `type` | VARCHAR(100) |
| `date` | DATE |

#### `TaiwanStockInfoWithWarrantSummary` (12 欄)
| 欄位 | 型態及大小 |
|---|---|
| `stock_id` | VARCHAR(100) |
| `date` | DATE |
| `close` | NUMERIC(20,6) |
| `target_stock_id` | NUMERIC(20,6) |
| `target_close` | NUMERIC(20,6) |
| `type` | VARCHAR(100) |
| `fulfillment_method` | VARCHAR(100) |
| `end_date` | DATE |
| `fulfillment_start_date` | DATE |
| `fulfillment_end_date` | DATE |
| `exercise_ratio` | NUMERIC(20,6) |
| `fulfillment_price` | NUMERIC(20,6) |

#### `TaiwanStockInstitutionalInvestorsBuySell` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `buy` | NUMERIC(20,6) |
| `name` | VARCHAR(100) |
| `sell` | NUMERIC(20,6) |

#### `TaiwanStockLoanCollateralBalance` (37 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `market` | VARCHAR(100) |
| `MarginPreviousDayBalance` | NUMERIC(20,6) |
| `MarginBuy` | NUMERIC(20,6) |
| `MarginSell` | NUMERIC(20,6) |
| `MarginCashRedemption` | NUMERIC(20,6) |
| `MarginCurrentDayBalance` | NUMERIC(20,6) |
| `MarginNextDayQuota` | NUMERIC(20,6) |
| `SecuritiesFirmLoanPreviousDayBalance` | NUMERIC(20,6) |
| `SecuritiesFirmLoanBuy` | NUMERIC(20,6) |
| `SecuritiesFirmLoanSell` | NUMERIC(20,6) |
| `SecuritiesFirmLoanCashRedemption` | NUMERIC(20,6) |
| `SecuritiesFirmLoanReplacement` | NUMERIC(20,6) |
| `SecuritiesFirmLoanCurrentDayBalance` | NUMERIC(20,6) |
| `SecuritiesFirmLoanNextDayQuota` | NUMERIC(20,6) |
| `UnrestrictedLoanPreviousDayBalance` | NUMERIC(20,6) |
| `UnrestrictedLoanBuy` | NUMERIC(20,6) |
| `UnrestrictedLoanSell` | NUMERIC(20,6) |
| `UnrestrictedLoanCashRedemption` | NUMERIC(20,6) |
| `UnrestrictedLoanReplacement` | NUMERIC(20,6) |
| `UnrestrictedLoanCurrentDayBalance` | NUMERIC(20,6) |
| `UnrestrictedLoanNextDayQuota` | NUMERIC(20,6) |
| `SecuritiesFinanceSecuredLoanPreviousDayBalance` | NUMERIC(20,6) |
| `SecuritiesFinanceSecuredLoanBuy` | NUMERIC(20,6) |
| `SecuritiesFinanceSecuredLoanSell` | NUMERIC(20,6) |
| `SecuritiesFinanceSecuredLoanCashRedemption` | NUMERIC(20,6) |
| `SecuritiesFinanceSecuredLoanReplacement` | NUMERIC(20,6) |
| `SecuritiesFinanceSecuredLoanCurrentDayBalance` | NUMERIC(20,6) |
| `SecuritiesFinanceSecuredLoanNextDayQuota` | NUMERIC(20,6) |
| `SettlementMarginPreviousDayBalance` | NUMERIC(20,6) |
| `SettlementMarginBuy` | NUMERIC(20,6) |
| `SettlementMarginSell` | NUMERIC(20,6) |
| `SettlementMarginCashRedemption` | NUMERIC(20,6) |
| `SettlementMarginReplacement` | NUMERIC(20,6) |
| `SettlementMarginCurrentDayBalance` | NUMERIC(20,6) |
| `SettlementMarginNextDayQuota` | NUMERIC(20,6) |

#### `TaiwanStockMarginPurchaseShortSale` (16 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `MarginPurchaseBuy` | NUMERIC(20,6) |
| `MarginPurchaseCashRepayment` | NUMERIC(20,6) |
| `MarginPurchaseLimit` | NUMERIC(20,6) |
| `MarginPurchaseSell` | NUMERIC(20,6) |
| `MarginPurchaseTodayBalance` | NUMERIC(20,6) |
| `MarginPurchaseYesterdayBalance` | NUMERIC(20,6) |
| `Note` | VARCHAR(100) |
| `OffsetLoanAndShort` | NUMERIC(20,6) |
| `ShortSaleBuy` | NUMERIC(20,6) |
| `ShortSaleCashRepayment` | NUMERIC(20,6) |
| `ShortSaleLimit` | NUMERIC(20,6) |
| `ShortSaleSell` | NUMERIC(20,6) |
| `ShortSaleTodayBalance` | NUMERIC(20,6) |
| `ShortSaleYesterdayBalance` | NUMERIC(20,6) |

#### `TaiwanStockMarginShortSaleSuspension` (4 欄)
| 欄位 | 型態及大小 |
|---|---|
| `stock_id` | VARCHAR(100) |
| `date` | DATE |
| `end_date` | DATE |
| `reason` | VARCHAR(100) |

#### `TaiwanStockMarketValue` (3 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `market_value` | NUMERIC(24,6) |

#### `TaiwanStockMarketValueWeight` (6 欄)
| 欄位 | 型態及大小 |
|---|---|
| `rank` | NUMERIC(20,6) |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |
| `weight_per` | NUMERIC(20,6) |
| `date` | DATE |
| `type` | VARCHAR(100) |

#### `TaiwanStockMonthPrice` (11 欄)
| 欄位 | 型態及大小 |
|---|---|
| `stock_id` | VARCHAR(100) |
| `ymonth` | VARCHAR(100) |
| `max` | NUMERIC(20,6) |
| `min` | NUMERIC(20,6) |
| `trading_volume` | NUMERIC(20,6) |
| `trading_money` | NUMERIC(23,6) |
| `trading_turnover` | NUMERIC(20,6) |
| `date` | DATE |
| `close` | NUMERIC(20,6) |
| `open` | NUMERIC(20,6) |
| `spread` | NUMERIC(20,6) |

#### `TaiwanStockMonthRevenue` (7 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `country` | VARCHAR(100) |
| `revenue` | NUMERIC(22,6) |
| `revenue_month` | NUMERIC(20,6) |
| `revenue_year` | NUMERIC(20,6) |
| `create_time` | DATE |

#### `TaiwanStockNews` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `link` | VARCHAR(100) |
| `source` | VARCHAR(100) |
| `title` | VARCHAR(100) |

#### `TaiwanStockPER` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `dividend_yield` | NUMERIC(20,6) |
| `PER` | NUMERIC(20,6) |
| `PBR` | NUMERIC(20,6) |

#### `TaiwanStockParValueChange` (8 欄 · 最早資料 2019-09-09)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |
| `before_close` | NUMERIC(20,6) |
| `after_ref_close` | NUMERIC(20,6) |
| `after_ref_max` | NUMERIC(20,6) |
| `after_ref_min` | NUMERIC(20,6) |
| `after_ref_open` | NUMERIC(20,6) |

#### `TaiwanStockPrice` (10 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Trading_Volume` | NUMERIC(20,6) |
| `Trading_money` | NUMERIC(22,6) |
| `open` | NUMERIC(20,6) |
| `max` | NUMERIC(20,6) |
| `min` | NUMERIC(20,6) |
| `close` | NUMERIC(20,6) |
| `spread` | NUMERIC(20,6) |
| `Trading_turnover` | NUMERIC(20,6) |

#### `TaiwanStockPriceAdj` (10 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Trading_Volume` | NUMERIC(20,6) |
| `Trading_money` | NUMERIC(22,6) |
| `open` | NUMERIC(20,6) |
| `max` | NUMERIC(20,6) |
| `min` | NUMERIC(20,6) |
| `close` | NUMERIC(20,6) |
| `spread` | NUMERIC(20,6) |
| `Trading_turnover` | NUMERIC(20,6) |

#### `TaiwanStockPriceLimit` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `reference_price` | NUMERIC(20,6) |
| `limit_up` | NUMERIC(20,6) |
| `limit_down` | NUMERIC(20,6) |

#### `TaiwanStockSecuritiesLending` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `transaction_type` | VARCHAR(100) |
| `volume` | NUMERIC(20,6) |
| `fee_rate` | NUMERIC(20,6) |
| `close` | NUMERIC(20,6) |
| `original_return_date` | DATE |
| `original_lending_period` | NUMERIC(20,6) |

#### `TaiwanStockShareholding` (13 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |
| `InternationalCode` | VARCHAR(100) |
| `ForeignInvestmentRemainingShares` | NUMERIC(20,6) |
| `ForeignInvestmentShares` | NUMERIC(21,6) |
| `ForeignInvestmentRemainRatio` | NUMERIC(20,6) |
| `ForeignInvestmentSharesRatio` | NUMERIC(20,6) |
| `ForeignInvestmentUpperLimitRatio` | NUMERIC(20,6) |
| `ChineseInvestmentUpperLimitRatio` | NUMERIC(20,6) |
| `NumberOfSharesIssued` | NUMERIC(21,6) |
| `RecentlyDeclareDate` | DATE |
| `note` | VARCHAR(100) |

#### `TaiwanStockSplitPrice` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `type` | VARCHAR(100) |
| `before_price` | NUMERIC(20,6) |
| `after_price` | NUMERIC(20,6) |
| `max_price` | NUMERIC(20,6) |
| `min_price` | NUMERIC(20,6) |
| `open_price` | NUMERIC(20,6) |

#### `TaiwanStockSuspended` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `suspension_time` | VARCHAR(100) |
| `resumption_date` | DATE |
| `resumption_time` | VARCHAR(100) |

#### `TaiwanStockTotalInstitutionalInvestors` (4 欄)
| 欄位 | 型態及大小 |
|---|---|
| `buy` | NUMERIC(23,6) |
| `date` | DATE |
| `name` | VARCHAR(100) |
| `sell` | NUMERIC(23,6) |

#### `TaiwanStockTotalMarginPurchaseShortSale` (7 欄)
| 欄位 | 型態及大小 |
|---|---|
| `TodayBalance` | NUMERIC(22,6) |
| `YesBalance` | NUMERIC(22,6) |
| `buy` | NUMERIC(21,6) |
| `date` | DATE |
| `name` | VARCHAR(100) |
| `Return` | NUMERIC(20,6) |
| `sell` | NUMERIC(21,6) |

#### `TaiwanStockTotalReturnIndex` (3 欄)
| 欄位 | 型態及大小 |
|---|---|
| `price` | NUMERIC(20,6) |
| `stock_id` | VARCHAR(100) |
| `date` | DATE |

#### `TaiwanStockTradingDailyReport` (7 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `securities_trader_id` | VARCHAR(100) |
| `securities_trader` | VARCHAR(100) |
| `price` | NUMERIC(20,6) |
| `buy` | NUMERIC(20,6) |
| `sell` | NUMERIC(20,6) |

#### `TaiwanStockTradingDate` (1 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |

#### `TaiwanStockWeekPrice` (11 欄)
| 欄位 | 型態及大小 |
|---|---|
| `stock_id` | VARCHAR(100) |
| `yweek` | VARCHAR(100) |
| `max` | NUMERIC(20,6) |
| `min` | NUMERIC(20,6) |
| `trading_volume` | NUMERIC(20,6) |
| `trading_money` | NUMERIC(22,6) |
| `trading_turnover` | NUMERIC(20,6) |
| `date` | DATE |
| `close` | NUMERIC(20,6) |
| `open` | NUMERIC(20,6) |
| `spread` | NUMERIC(20,6) |

#### `TaiwanTotalExchangeMarginMaintenance` (2 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `TotalExchangeMarginMaintenance` | NUMERIC(20,6) |

## FinMind — 非 Taiwan(美/歐/日/英股 + 指數/商品/匯率/利率)
#### `CnnFearGreedIndex` (3 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `fear_greed` | NUMERIC(20,6) |
| `fear_greed_emotion` | VARCHAR(100) |

#### `CrudeOilPrices` (3 欄 · 最早資料 1986-01-02)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `name` | VARCHAR(100) |
| `price` | NUMERIC(20,6) |

#### `EuropeStockInfo` (4 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Market` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |

#### `EuropeStockPrice` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Open` | NUMERIC(20,6) |
| `High` | NUMERIC(20,6) |
| `Low` | NUMERIC(20,6) |
| `Close` | NUMERIC(20,6) |
| `Adj_Close` | NUMERIC(20,6) |
| `Volume` | NUMERIC(20,6) |

#### `ExchangeRate` (4 欄)
| 欄位 | 型態及大小 |
|---|---|
| `InterbankRate` | NUMERIC(20,6) |
| `InverseInterbankRate` | NUMERIC(20,6) |
| `country` | VARCHAR(100) |
| `date` | DATE |

#### `GoldPrice` (2 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `Price` | NUMERIC(20,6) |

#### `InterestRate` (4 欄 · 最早資料 2024-01-31)
| 欄位 | 型態及大小 |
|---|---|
| `country` | VARCHAR(100) |
| `date` | DATE |
| `full_country_name` | VARCHAR(100) |
| `interest_rate` | NUMERIC(20,6) |

#### `JapanStockInfo` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Exchange` | VARCHAR(100) |
| `Sector` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |

#### `JapanStockPrice` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Adj_Close` | NUMERIC(20,6) |
| `Close` | NUMERIC(20,6) |
| `High` | NUMERIC(20,6) |
| `Low` | NUMERIC(20,6) |
| `Open` | NUMERIC(20,6) |
| `Volume` | NUMERIC(20,6) |

#### `UKStockInfo` (4 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Country` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |

#### `UKStockPrice` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Open` | NUMERIC(20,6) |
| `High` | NUMERIC(20,6) |
| `Low` | NUMERIC(20,6) |
| `Close` | NUMERIC(20,6) |
| `Adj_Close` | NUMERIC(20,6) |
| `Volume` | NUMERIC(20,6) |

#### `USStockInfo` (7 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `stock_name` | VARCHAR(100) |
| `Country` | VARCHAR(100) |
| `IPOYear` | NUMERIC(20,6) |
| `MarketCap` | VARCHAR(100) |
| `Subsector` | VARCHAR(100) |

#### `USStockPrice` (8 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `stock_id` | VARCHAR(100) |
| `Adj_Close` | NUMERIC(20,6) |
| `Close` | NUMERIC(20,6) |
| `High` | NUMERIC(20,6) |
| `Low` | NUMERIC(20,6) |
| `Open` | NUMERIC(20,6) |
| `Volume` | NUMERIC(20,6) |

## FRED
#### `FredData` (5 欄)
| 欄位 | 型態及大小 |
|---|---|
| `date` | DATE |
| `series_id` | VARCHAR(255) |
| `value` | NUMERIC(20,6) |
| `realtime_start` | DATE |
| `realtime_end` | DATE |

**`series_id` 24 series**:T10Y2Y/T10Y3M/T10YIE/VIXCLS/BAMLH0A0HYM2/DTWEXBGS/M2SL/DGS10/DGS2/DGS3MO/UMCSENT/INDPRO/UNRATE/CPIAUCSL/PATENTUSALLTOTAL/B985RC1Q027SBEA/TCMDO/LFWA64TTUSA647N/SPPOPDPNDOLUSA/PALLFNFINDEXQ/QUSPAM770A/WTISPLC/IPG3344S/PCU4831114831115(定義見 §14.7-DJ 前述)。

## 待解 4(需升 tier / 事件稀少 / 未知 country 參數,非不存在)
- `TaiwanStockCapitalReductionReferencePrice`
- `TaiwanStockBlockTradingDailyReport`
- `TaiwanStockWarrantTradingDailyReport`
- `GovernmentBondsYield`

## intraday 排除 9(日為最小單位,§14.7-DI T_DI-7)
- `TaiwanStockPriceTick` / `TaiwanStockKBar` / `TaiwanStockStatisticsOfOrderBookAndTrade` / `TaiwanVariousIndicators5Seconds` / `TaiwanStockEvery5SecondsIndex` / `TaiwanFuturesTick` / `TaiwanOptionTick` / `TaiwanFutOptTickInfo` / `USStockPriceMinute`