# FinMind + FRED 通用 Ingester 完整 Table Schema Catalog（82 表 · 逐欄中文定義）

**日期**:2026-06-08 | **治權**:主憲章 §14.7-DJ §二 之 companion 完整檔（per-column schema SSOT）| **source**:§一.10 實際 FinMind/FRED API 回應 build 結果
**大小寫**:表名/欄位名與 FinMind/FRED API 逐字一致（§14.7-CC,generic ingester 雙引號封裝保留大小寫）
**中文定義來源**:long-format(財報/資產負債/現金流量)之 `type` 科目 + 13 feature-input 表 = API `origin_name` / 資料字典（**authoritative**）；其餘探索表 = 標準財經術語**直譯**（欄名自描述,非 API 賦予之中文）。

> 長表 `TaiwanStockFinancialStatements`(17 科目)/`TaiwanStockBalanceSheet`(101 科目)/`TaiwanStockCashFlowsStatement` 之 `type` 科目枚舉(origin_name 中文)見主憲章 §14.7-DJ §二。

---

**完整 catalog:82 表有完整 schema**（FinMind Taiwan 67 + 非Taiwan 13 + FRED 2:`FredData`+`fred_series`）。表名/欄位名 = FinMind/FRED API 確切大小寫逐字鏡像(§14.7-CC);型別由 generic auto-schema 依實際 API 回應值推導(§一.10;**字串下限 VARCHAR(255)、數字下限 NUMERIC(20,6),值超界自動加大** — 2026-06-08 用戶 directive 字串下限 100→255;`VARCHAR(493)` 等 >255 者為實際觀測值超界自動加大);intraday(日以下)排除 9、待解 3。

> **🔬 2026-06-08 live API 驗證 note**:本字典 schema 經實際 FinMind/FRED API 逐一打驗證(**72/80 FinMind + 2 FRED 真打 ✓**)。**generic 型別為 sample-dependent**:欄位名 / 大小寫 / `VARCHAR(255)` / `NUMERIC(20,6)` floor / `date`→DATE 為**不變量**;**>255 VARCHAR 長度 / >(20,6) NUMERIC 精度 / 稀疏欄 VARCHAR↔NUMERIC↔DATE 隨觀測值 auto-widen**,最終以全市場全史 sync 觀測 max 定案。
> - **4 處 live 修正(本次已套用真值)**:`TaiwanStockNews.link` 255→**592**;`TaiwanStockDispositionSecuritiesPeriod.measure` 493→**510**;`USStockInfo.MarketCap` VARCHAR(255)→**NUMERIC(24,6)**(稀疏欄此樣本全數字);`TaiwanStockConvertibleBondDailyOverview` 之 `LatestInitialDateOfPut`/`LatestDueDateOfPut` VARCHAR→**DATE** + `IssuanceAmount` →**NUMERIC(21,6)**。
> - **1 表標 ᵈ**(2026-06-08 API 全參數〔USD/EUR/JPY/CNY/country/無-data_id〕皆回 0 rows〔HTTP 200 success〕→ 疑 deprecated/空集;schema 以**建檔時實打值**為準,**非缺表**):`ExchangeRate`。
> - **2026-06-08(cont) 再攻克 4 個 → 改 live-confirmed**(找對 data_id,皆 live 確認與建檔值一致):`TaiwanExchangeRate`(data_id=USD)、`CrudeOilPrices`(data_id=WTI)、`TaiwanStockMarginShortSaleSuspension`(data_id=2330)、`TaiwanStockCapitalReductionReferencePrice`(data_id=2603)。→ live 驗證覆蓋 **81/82**(79 FinMind + 2 FRED;2026-06-08 再攻克 `TaiwanStockGovernmentBankBuySell`〔無 data_id+單一 start〕/`TaiwanStockMarketValueWeight`〔data_id=2330〕/`InterestRate`〔無 data_id,start≥2024-02-01〕);僅餘 `ExchangeRate` 1 ᵈ。

## FinMind — Taiwan
#### `TaiwanBusinessIndicator` (9 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `leading` | NUMERIC(20,6) | 領先指標 |
| `leading_notrend` | NUMERIC(20,6) | 領先指標(不含趨勢) |
| `coincident` | NUMERIC(20,6) | 同時指標 |
| `coincident_notrend` | NUMERIC(20,6) | 同時指標(不含趨勢) |
| `lagging` | NUMERIC(20,6) | 落後指標 |
| `lagging_notrend` | NUMERIC(20,6) | 落後指標(不含趨勢) |
| `monitoring` | NUMERIC(20,6) | 景氣對策信號分數 |
| `monitoring_color` | VARCHAR(255) | 景氣對策信號燈號 |

#### `TaiwanDailyShortSaleBalances` (15 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `stock_id` | VARCHAR(255) | 證券代號 |
| `MarginShortSalesPreviousDayBalance` | NUMERIC(20,6) | 融券前日餘額 |
| `MarginShortSalesShortSales` | NUMERIC(20,6) | 融券賣出 |
| `MarginShortSalesShortCovering` | NUMERIC(20,6) | 融券買進回補 |
| `MarginShortSalesStockRedemption` | NUMERIC(20,6) | 融券現券償還 |
| `MarginShortSalesCurrentDayBalance` | NUMERIC(20,6) | 融券當日餘額 |
| `MarginShortSalesQuota` | NUMERIC(20,6) | 融券限額 |
| `SBLShortSalesPreviousDayBalance` | NUMERIC(20,6) | 借券賣出前日餘額 |
| `SBLShortSalesShortSales` | NUMERIC(20,6) | 借券賣出 |
| `SBLShortSalesReturns` | NUMERIC(20,6) | 借券賣出返還 |
| `SBLShortSalesAdjustments` | NUMERIC(20,6) | 借券賣出調整 |
| `SBLShortSalesCurrentDayBalance` | NUMERIC(20,6) | 借券賣出當日餘額 |
| `SBLShortSalesQuota` | NUMERIC(20,6) | 借券賣出限額 |
| `SBLShortSalesShortCovering` | NUMERIC(20,6) | 借券賣出回補 |
| `date` | DATE | 日期 |

#### `TaiwanExchangeRate` (6 欄 · 最早資料 2006-01-02) — ✓ 2026-06-08 live 驗證(data_id=USD)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `currency` | VARCHAR(255) | 幣別 |
| `cash_buy` | NUMERIC(20,6) | 現金買入匯率 |
| `cash_sell` | NUMERIC(20,6) | 現金賣出匯率 |
| `spot_buy` | NUMERIC(20,6) | 即期買入匯率 |
| `spot_sell` | NUMERIC(20,6) | 即期賣出匯率 |

#### `TaiwanFutOptDailyInfo` (3 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `code` | VARCHAR(255) | 商品代號 |
| `type` | VARCHAR(255) | 商品類別(期貨/選擇權) |
| `name` | VARCHAR(255) | 商品名稱 |

#### `TaiwanFutOptInstitutionalInvestors` (11 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `name` | VARCHAR(255) | 商品名稱 |
| `date` | DATE | 日期 |
| `institutional_investors` | VARCHAR(255) | 法人別 |
| `long_deal_volume` | NUMERIC(20,6) | 多方交易口數 |
| `long_deal_amount` | NUMERIC(20,6) | 多方交易金額 |
| `short_deal_volume` | NUMERIC(20,6) | 空方交易口數 |
| `short_deal_amount` | NUMERIC(20,6) | 空方交易金額 |
| `long_open_interest_balance_volume` | NUMERIC(20,6) | 多方未平倉口數 |
| `long_open_interest_balance_amount` | NUMERIC(20,6) | 多方未平倉金額 |
| `short_open_interest_balance_volume` | NUMERIC(20,6) | 空方未平倉口數 |
| `short_open_interest_balance_amount` | NUMERIC(20,6) | 空方未平倉金額 |

#### `TaiwanFuturesDaily` (13 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `futures_id` | VARCHAR(255) | 期貨商品代號 |
| `contract_date` | VARCHAR(255) | 契約月份 |
| `open` | NUMERIC(20,6) | 開盤價 |
| `max` | NUMERIC(20,6) | 最高價 |
| `min` | NUMERIC(20,6) | 最低價 |
| `close` | NUMERIC(20,6) | 收盤價 |
| `spread` | NUMERIC(20,6) | 漲跌價差 |
| `spread_per` | NUMERIC(20,6) | 漲跌幅(%) |
| `volume` | NUMERIC(20,6) | 成交量/成交口數 |
| `settlement_price` | NUMERIC(20,6) | 結算價 |
| `open_interest` | NUMERIC(20,6) | 未平倉量 |
| `trading_session` | VARCHAR(255) | 交易時段(日盤/夜盤) |

#### `TaiwanFuturesDealerTradingVolumeDaily` (6 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `dealer_code` | VARCHAR(255) | 自營商代號 |
| `dealer_name` | VARCHAR(255) | 自營商名稱 |
| `futures_id` | VARCHAR(255) | 期貨商品代號 |
| `volume` | NUMERIC(20,6) | 成交量/成交口數 |
| `is_after_hour` | VARCHAR(255) | 是否盤後 |

#### `TaiwanFuturesFinalSettlementPrice` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `contract_month` | VARCHAR(255) | 契約月份 |
| `futures_type` | VARCHAR(255) | 期貨類別 |
| `futures_id` | VARCHAR(255) | 期貨商品代號 |
| `futures_name` | VARCHAR(255) | 期貨名稱 |
| `settlement_price` | NUMERIC(20,6) | 結算價 |
| `underlying_code` | VARCHAR(255) | 標的代號 |
| `notional_value` | NUMERIC(20,6) | 契約價值 |

#### `TaiwanFuturesInstitutionalInvestors` (11 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `futures_id` | VARCHAR(255) | 期貨商品代號 |
| `date` | DATE | 日期 |
| `institutional_investors` | VARCHAR(255) | 法人別 |
| `long_deal_volume` | NUMERIC(20,6) | 多方交易口數 |
| `long_deal_amount` | NUMERIC(20,6) | 多方交易金額 |
| `short_deal_volume` | NUMERIC(20,6) | 空方交易口數 |
| `short_deal_amount` | NUMERIC(20,6) | 空方交易金額 |
| `long_open_interest_balance_volume` | NUMERIC(20,6) | 多方未平倉口數 |
| `long_open_interest_balance_amount` | NUMERIC(20,6) | 多方未平倉金額 |
| `short_open_interest_balance_volume` | NUMERIC(20,6) | 空方未平倉口數 |
| `short_open_interest_balance_amount` | NUMERIC(20,6) | 空方未平倉金額 |

#### `TaiwanFuturesInstitutionalInvestorsAfterHours` (7 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `futures_id` | VARCHAR(255) | 期貨商品代號 |
| `date` | DATE | 日期 |
| `institutional_investors` | VARCHAR(255) | 法人別 |
| `long_deal_volume` | NUMERIC(20,6) | 多方交易口數 |
| `long_deal_amount` | NUMERIC(20,6) | 多方交易金額 |
| `short_deal_volume` | NUMERIC(20,6) | 空方交易口數 |
| `short_deal_amount` | NUMERIC(20,6) | 空方交易金額 |

#### `TaiwanFuturesOpenInterestLargeTraders` (21 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `name` | VARCHAR(255) | 商品名稱 |
| `contract_type` | VARCHAR(255) | 契約類別 |
| `buy_top5_trader_open_interest` | NUMERIC(20,6) | 買方前5大交易人未平倉 |
| `buy_top5_trader_open_interest_per` | NUMERIC(20,6) | 買方前5大交易人未平倉% |
| `buy_top10_trader_open_interest` | NUMERIC(20,6) | 買方前10大交易人未平倉 |
| `buy_top10_trader_open_interest_per` | NUMERIC(20,6) | 買方前10大交易人未平倉% |
| `sell_top5_trader_open_interest` | NUMERIC(20,6) | 賣方前5大交易人未平倉 |
| `sell_top5_trader_open_interest_per` | NUMERIC(20,6) | 賣方前5大交易人未平倉% |
| `sell_top10_trader_open_interest` | NUMERIC(20,6) | 賣方前10大交易人未平倉 |
| `sell_top10_trader_open_interest_per` | NUMERIC(20,6) | 賣方前10大交易人未平倉% |
| `market_open_interest` | NUMERIC(20,6) | 全市場未平倉量 |
| `buy_top5_specific_open_interest` | NUMERIC(20,6) | 買方前5大特定法人未平倉 |
| `buy_top5_specific_open_interest_per` | NUMERIC(20,6) | 買方前5大特定法人未平倉% |
| `buy_top10_specific_open_interest` | NUMERIC(20,6) | 買方前10大特定法人未平倉 |
| `buy_top10_specific_open_interest_per` | NUMERIC(20,6) | 買方前10大特定法人未平倉% |
| `sell_top5_specific_open_interest` | NUMERIC(20,6) | 賣方前5大特定法人未平倉 |
| `sell_top5_specific_open_interest_per` | NUMERIC(20,6) | 賣方前5大特定法人未平倉% |
| `sell_top10_specific_open_interest` | NUMERIC(20,6) | 賣方前10大特定法人未平倉 |
| `sell_top10_specific_open_interest_per` | NUMERIC(20,6) | 賣方前10大特定法人未平倉% |
| `date` | DATE | 日期 |
| `futures_id` | VARCHAR(255) | 期貨商品代號 |

#### `TaiwanFuturesSpreadTrading` (14 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `futures_id` | VARCHAR(255) | 期貨商品代號 |
| `contract_date` | VARCHAR(255) | 契約月份 |
| `open` | NUMERIC(20,6) | 開盤價 |
| `max` | NUMERIC(20,6) | 最高價 |
| `min` | NUMERIC(20,6) | 最低價 |
| `close` | NUMERIC(20,6) | 收盤價 |
| `best_bid` | NUMERIC(20,6) | 最佳買價 |
| `best_ask` | NUMERIC(20,6) | 最佳賣價 |
| `historical_max` | NUMERIC(20,6) | 歷史最高 |
| `historical_min` | NUMERIC(20,6) | 歷史最低 |
| `spread_to_spread_volume` | NUMERIC(20,6) | 價差對價差成交量 |
| `spread_to_single_volume` | NUMERIC(20,6) | 價差對單式成交量 |
| `trading_session` | VARCHAR(255) | 交易時段(日盤/夜盤) |

#### `TaiwanOptionDaily` (13 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `option_id` | VARCHAR(255) | 選擇權商品代號 |
| `contract_date` | VARCHAR(255) | 契約月份 |
| `strike_price` | NUMERIC(20,6) | 履約價 |
| `call_put` | VARCHAR(255) | 買權/賣權(call/put) |
| `open` | NUMERIC(20,6) | 開盤價 |
| `max` | NUMERIC(20,6) | 最高價 |
| `min` | NUMERIC(20,6) | 最低價 |
| `close` | NUMERIC(20,6) | 收盤價 |
| `volume` | NUMERIC(20,6) | 成交量/成交口數 |
| `settlement_price` | NUMERIC(20,6) | 結算價 |
| `open_interest` | NUMERIC(20,6) | 未平倉量 |
| `trading_session` | VARCHAR(255) | 交易時段(日盤/夜盤) |

#### `TaiwanOptionDealerTradingVolumeDaily` (6 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `dealer_code` | VARCHAR(255) | 自營商代號 |
| `dealer_name` | VARCHAR(255) | 自營商名稱 |
| `option_id` | VARCHAR(255) | 選擇權商品代號 |
| `volume` | NUMERIC(20,6) | 成交量/成交口數 |
| `is_after_hour` | VARCHAR(255) | 是否盤後 |

#### `TaiwanOptionFinalSettlementPrice` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `contract_month` | VARCHAR(255) | 契約月份 |
| `option_type` | VARCHAR(255) | 選擇權類別 |
| `option_id` | VARCHAR(255) | 選擇權商品代號 |
| `option_name` | VARCHAR(255) | 選擇權名稱 |
| `settlement_price` | NUMERIC(20,6) | 結算價 |
| `underlying_code` | VARCHAR(255) | 標的代號 |
| `notional_value` | NUMERIC(20,6) | 契約價值 |

#### `TaiwanOptionInstitutionalInvestors` (12 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `option_id` | VARCHAR(255) | 選擇權商品代號 |
| `date` | DATE | 日期 |
| `call_put` | VARCHAR(255) | 買權/賣權(call/put) |
| `institutional_investors` | VARCHAR(255) | 法人別 |
| `long_deal_volume` | NUMERIC(20,6) | 多方交易口數 |
| `long_deal_amount` | NUMERIC(20,6) | 多方交易金額 |
| `short_deal_volume` | NUMERIC(20,6) | 空方交易口數 |
| `short_deal_amount` | NUMERIC(20,6) | 空方交易金額 |
| `long_open_interest_balance_volume` | NUMERIC(20,6) | 多方未平倉口數 |
| `long_open_interest_balance_amount` | NUMERIC(20,6) | 多方未平倉金額 |
| `short_open_interest_balance_volume` | NUMERIC(20,6) | 空方未平倉口數 |
| `short_open_interest_balance_amount` | NUMERIC(20,6) | 空方未平倉金額 |

#### `TaiwanOptionInstitutionalInvestorsAfterHours` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `option_id` | VARCHAR(255) | 選擇權商品代號 |
| `date` | DATE | 日期 |
| `call_put` | VARCHAR(255) | 買權/賣權(call/put) |
| `institutional_investors` | VARCHAR(255) | 法人別 |
| `long_deal_volume` | NUMERIC(20,6) | 多方交易口數 |
| `long_deal_amount` | NUMERIC(20,6) | 多方交易金額 |
| `short_deal_volume` | NUMERIC(20,6) | 空方交易口數 |
| `short_deal_amount` | NUMERIC(20,6) | 空方交易金額 |

#### `TaiwanOptionOpenInterestLargeTraders` (22 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `contract_type` | VARCHAR(255) | 契約類別 |
| `buy_top5_trader_open_interest` | NUMERIC(20,6) | 買方前5大交易人未平倉 |
| `buy_top5_trader_open_interest_per` | NUMERIC(20,6) | 買方前5大交易人未平倉% |
| `buy_top10_trader_open_interest` | NUMERIC(20,6) | 買方前10大交易人未平倉 |
| `buy_top10_trader_open_interest_per` | NUMERIC(20,6) | 買方前10大交易人未平倉% |
| `sell_top5_trader_open_interest` | NUMERIC(20,6) | 賣方前5大交易人未平倉 |
| `sell_top5_trader_open_interest_per` | NUMERIC(20,6) | 賣方前5大交易人未平倉% |
| `sell_top10_trader_open_interest` | NUMERIC(20,6) | 賣方前10大交易人未平倉 |
| `sell_top10_trader_open_interest_per` | NUMERIC(20,6) | 賣方前10大交易人未平倉% |
| `market_open_interest` | NUMERIC(20,6) | 全市場未平倉量 |
| `buy_top5_specific_open_interest` | NUMERIC(20,6) | 買方前5大特定法人未平倉 |
| `buy_top5_specific_open_interest_per` | NUMERIC(20,6) | 買方前5大特定法人未平倉% |
| `buy_top10_specific_open_interest` | NUMERIC(20,6) | 買方前10大特定法人未平倉 |
| `buy_top10_specific_open_interest_per` | NUMERIC(20,6) | 買方前10大特定法人未平倉% |
| `sell_top5_specific_open_interest` | NUMERIC(20,6) | 賣方前5大特定法人未平倉 |
| `sell_top5_specific_open_interest_per` | NUMERIC(20,6) | 賣方前5大特定法人未平倉% |
| `sell_top10_specific_open_interest` | NUMERIC(20,6) | 賣方前10大特定法人未平倉 |
| `sell_top10_specific_open_interest_per` | NUMERIC(20,6) | 賣方前10大特定法人未平倉% |
| `date` | DATE | 日期 |
| `put_call` | VARCHAR(255) | 買權/賣權 |
| `name` | VARCHAR(255) | 商品名稱 |
| `option_id` | VARCHAR(255) | 選擇權商品代號 |

#### `TaiwanSecuritiesTraderInfo` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `securities_trader_id` | VARCHAR(255) | 券商代號 |
| `securities_trader` | VARCHAR(255) | 券商名稱 |
| `date` | DATE | 日期 |
| `address` | VARCHAR(255) | 地址 |
| `phone` | VARCHAR(255) | 電話 |

#### `TaiwanStock10Year` (3 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `close` | NUMERIC(20,6) | 收盤價 |

#### `TaiwanStockBalanceSheet` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `type` | VARCHAR(255) | 會計科目代號(資產負債表;_per=占比變體;101 型見憲章§14.7-DJ§二) |
| `value` | NUMERIC(23,6) | 金額(元)或百分比(_per 列) |
| `origin_name` | VARCHAR(255) | 科目中文名(API 權威) |

#### `TaiwanStockBlockTrade` (6 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `trade_type` | VARCHAR(255) | 交易類別 |
| `price` | NUMERIC(20,6) | 價格 |
| `volume` | NUMERIC(20,6) | 成交量/成交口數 |
| `trading_money` | NUMERIC(21,6) | 成交金額(元) |

#### `TaiwanStockCapitalReductionReferencePrice` (9 欄 · 最早資料 2012-11-12 · per-stock 減資事件) — ✓ 2026-06-08 live 驗證(data_id=2603)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `ClosingPriceonTheLastTradingDay` | NUMERIC(20,6) | 最後交易日收盤價 |
| `PostReductionReferencePrice` | NUMERIC(20,6) | 減資後參考價 |
| `LimitUp` | NUMERIC(20,6) | 漲停價 |
| `LimitDown` | NUMERIC(20,6) | 跌停價 |
| `OpeningReferencePrice` | NUMERIC(20,6) | 開盤參考價 |
| `ExrightReferencePrice` | NUMERIC(20,6) | 除權參考價 |
| `ReasonforCapitalReduction` | VARCHAR(255) | 減資原因 |

#### `TaiwanStockCashFlowsStatement` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `type` | VARCHAR(255) | 會計科目代號(現金流量表) |
| `value` | NUMERIC(23,6) | 金額(元) |
| `origin_name` | VARCHAR(255) | 科目中文名(API 權威) |

#### `TaiwanStockConvertibleBondDaily` (16 欄 · 最早資料 2020-01-16)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `cb_id` | NUMERIC(20,6) | 可轉債代號 |
| `cb_name` | VARCHAR(255) | 可轉債名稱 |
| `transaction_type` | VARCHAR(255) | 交易類別 |
| `close` | NUMERIC(20,6) | 收盤價 |
| `change` | NUMERIC(20,6) | 漲跌 |
| `open` | NUMERIC(20,6) | 開盤價 |
| `max` | NUMERIC(20,6) | 最高價 |
| `min` | NUMERIC(20,6) | 最低價 |
| `no_of_transactions` | NUMERIC(20,6) | 成交筆數 |
| `unit` | NUMERIC(20,6) | 單位數 |
| `trading_value` | NUMERIC(20,6) | 成交值 |
| `avg_price` | NUMERIC(20,6) | 均價 |
| `next_ref_price` | NUMERIC(20,6) | 次日參考價 |
| `next_max_limit` | NUMERIC(20,6) | 次日漲停 |
| `next_min_limit` | NUMERIC(20,6) | 次日跌停 |
| `date` | DATE | 日期 |

#### `TaiwanStockConvertibleBondDailyOverview` (23 欄 · 最早資料 2020-01-16)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `cb_id` | NUMERIC(20,6) | 可轉債代號 |
| `cb_name` | VARCHAR(255) | 可轉債名稱 |
| `date` | DATE | 日期 |
| `InitialDateOfConversion` | DATE | 轉換起始日 |
| `DueDateOfConversion` | DATE | 轉換到期日 |
| `InitialDateOfStopConversion` | VARCHAR(255) | 停止轉換起日 |
| `DueDateOfStopConversion` | VARCHAR(255) | 停止轉換迄日 |
| `ConversionPrice` | NUMERIC(20,6) | 轉換價格 |
| `NextEffectiveDateOfConversionPrice` | DATE | 次一轉換價生效日 |
| `LatestInitialDateOfPut` | DATE | 最近賣回權起日 |
| `LatestDueDateOfPut` | DATE | 最近賣回權迄日 |
| `LatestPutPrice` | NUMERIC(20,6) | 最近賣回價 |
| `InitialDateOfEarlyRedemption` | DATE | 提前贖回起日 |
| `DueDateOfEarlyRedemption` | DATE | 提前贖回迄日 |
| `EarlyRedemptionPrice` | NUMERIC(20,6) | 提前贖回價 |
| `DateOfDelisted` | DATE | 下市日 |
| `IssuanceAmount` | NUMERIC(21,6) | 發行總額 |
| `OutstandingAmount` | NUMERIC(20,6) | 流通在外餘額 |
| `ReferencePrice` | NUMERIC(20,6) | 參考價 |
| `PriceOfUnderlyingStock` | NUMERIC(20,6) | 標的股價 |
| `InitialDateOfSuspension` | VARCHAR(255) | 暫停起日 |
| `DueDateOfSuspension` | VARCHAR(255) | 暫停迄日 |
| `CouponRate` | NUMERIC(20,6) | 票面利率(%) |

#### `TaiwanStockConvertibleBondInfo` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `cb_id` | NUMERIC(20,6) | 可轉債代號 |
| `cb_name` | VARCHAR(255) | 可轉債名稱 |
| `InitialDateOfConversion` | DATE | 轉換起始日 |
| `DueDateOfConversion` | DATE | 轉換到期日 |
| `IssuanceAmount` | NUMERIC(20,6) | 發行總額 |

#### `TaiwanStockConvertibleBondInstitutionalInvestors` (13 欄 · 最早資料 2020-01-16)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `Foreign_Investor_Buy` | NUMERIC(20,6) | 外資買進 |
| `Foreign_Investor_Sell` | NUMERIC(20,6) | 外資賣出 |
| `Foreign_Investor_Overbuy` | NUMERIC(20,6) | 外資買超 |
| `Investment_Trust_Buy` | NUMERIC(20,6) | 投信買進 |
| `Investment_Trust_Sell` | NUMERIC(20,6) | 投信賣出 |
| `Investment_Trust_Overbuy` | NUMERIC(20,6) | 投信買超 |
| `Dealer_self_Buy` | NUMERIC(20,6) | 自營商買進 |
| `Dealer_self_Sell` | NUMERIC(20,6) | 自營商賣出 |
| `Dealer_self_Overbuy` | NUMERIC(20,6) | 自營商買超 |
| `Total_Overbuy` | NUMERIC(20,6) | 合計買超 |
| `cb_id` | NUMERIC(20,6) | 可轉債代號 |
| `cb_name` | VARCHAR(255) | 可轉債名稱 |
| `date` | DATE | 日期 |

#### `TaiwanStockDayTrading` (6 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `stock_id` | VARCHAR(255) | 證券代號 |
| `date` | DATE | 日期 |
| `BuyAfterSale` | VARCHAR(255) | 先買後賣可否 |
| `Volume` | NUMERIC(20,6) | 成交量(股) |
| `BuyAmount` | NUMERIC(21,6) | 買進金額(元) |
| `SellAmount` | NUMERIC(21,6) | 賣出金額(元) |

#### `TaiwanStockDayTradingBorrowingFeeRate` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |
| `InvestorBorrowedShares` | NUMERIC(20,6) | 投資人借券股數 |
| `InvestorBorrowingFeeRate` | NUMERIC(20,6) | 投資人借券費率(%) |

#### `TaiwanStockDayTradingSuspension` (4 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `stock_id` | VARCHAR(255) | 證券代號 |
| `date` | DATE | 日期 |
| `end_date` | DATE | 結束日 |
| `reason` | VARCHAR(255) | 事由 |

#### `TaiwanStockDelisting` (3 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |

#### `TaiwanStockDispositionSecuritiesPeriod` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |
| `disposition_cnt` | NUMERIC(20,6) | 處置次數 |
| `condition` | VARCHAR(255) | 處置條件 |
| `measure` | VARCHAR(510) | 處置措施 |
| `period_start` | DATE | 處置起日 |
| `period_end` | DATE | 處置迄日 |

#### `TaiwanStockDividend` (22 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `year` | VARCHAR(255) | 所屬年度 |
| `StockEarningsDistribution` | NUMERIC(20,6) | 盈餘配股(元/股) |
| `StockStatutorySurplus` | NUMERIC(20,6) | 法定盈餘公積配股(元/股) |
| `StockExDividendTradingDate` | VARCHAR(255) | 除權交易日 |
| `TotalEmployeeStockDividend` | NUMERIC(20,6) | 員工配股總數 |
| `TotalEmployeeStockDividendAmount` | NUMERIC(20,6) | 員工配股總金額 |
| `RatioOfEmployeeStockDividendOfTotal` | NUMERIC(20,6) | 員工配股占總額比率 |
| `RatioOfEmployeeStockDividend` | NUMERIC(20,6) | 員工配股比率 |
| `CashEarningsDistribution` | NUMERIC(20,8) | 盈餘配息(元/股) |
| `CashStatutorySurplus` | NUMERIC(20,6) | 法定盈餘公積配息(元/股) |
| `CashExDividendTradingDate` | DATE | 除息交易日 |
| `CashDividendPaymentDate` | DATE | 現金股利發放日 |
| `TotalEmployeeCashDividend` | NUMERIC(20,6) | 員工現金紅利總額 |
| `TotalNumberOfCashCapitalIncrease` | NUMERIC(20,6) | 現金增資總股數 |
| `CashIncreaseSubscriptionRate` | NUMERIC(20,6) | 現金增資認購比率 |
| `CashIncreaseSubscriptionpRrice` | NUMERIC(20,6) | 現金增資認購價(API 原拼字) |
| `RemunerationOfDirectorsAndSupervisors` | NUMERIC(20,6) | 董監酬勞 |
| `ParticipateDistributionOfTotalShares` | NUMERIC(21,6) | 參與分配總股數 |
| `AnnouncementDate` | DATE | 公告日 |
| `AnnouncementTime` | VARCHAR(255) | 公告時間 |

#### `TaiwanStockDividendResult` (10 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `before_price` | NUMERIC(20,6) | 除權息前價 |
| `after_price` | NUMERIC(20,6) | 除權息後價 |
| `stock_and_cache_dividend` | NUMERIC(20,6) | 股票及現金股利 |
| `stock_or_cache_dividend` | VARCHAR(255) | 配股或配息別 |
| `max_price` | NUMERIC(20,6) | 最高價 |
| `min_price` | NUMERIC(20,6) | 最低價 |
| `open_price` | NUMERIC(20,6) | 開盤價 |
| `reference_price` | NUMERIC(20,6) | 參考價 |

#### `TaiwanStockFinancialStatements` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `type` | VARCHAR(255) | 會計科目代號(綜合損益表;17 型枚舉見憲章§14.7-DJ§二) |
| `value` | NUMERIC(23,6) | 金額(元;EPS=元/股) |
| `origin_name` | VARCHAR(255) | 科目中文名(API 權威) |

#### `TaiwanStockGovernmentBankBuySell` (7 欄) — ✓ 2026-06-08 live 驗證(不帶 data_id + 單一 start_date;帶 end_date 觸發 size-too-large)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `bank_name` | VARCHAR(255) | 行庫名稱 |
| `buy` | NUMERIC(20,6) | 買進 |
| `sell` | NUMERIC(20,6) | 賣出 |
| `buy_amount` | NUMERIC(20,6) | 買進金額(元) |
| `sell_amount` | NUMERIC(20,6) | 賣出金額(元) |

#### `TaiwanStockHoldingSharesPer` (6 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `HoldingSharesLevel` | VARCHAR(255) | 持股級距 |
| `people` | NUMERIC(20,6) | 人數 |
| `percent` | NUMERIC(20,6) | 占比(%) |
| `unit` | NUMERIC(21,6) | 單位數 |

#### `TaiwanStockIndustryChain` (4 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `stock_id` | VARCHAR(255) | 證券代號 |
| `industry` | VARCHAR(255) | 產業 |
| `sub_industry` | VARCHAR(255) | 次產業 |
| `date` | DATE | 日期 |

#### `TaiwanStockInfo` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `industry_category` | VARCHAR(255) | 產業類別 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |
| `type` | VARCHAR(255) | 市場別(twse上市/tpex上櫃/emerging興櫃) |
| `date` | DATE | 日期 |

#### `TaiwanStockInfoWithWarrant` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `industry_category` | VARCHAR(255) | 產業類別 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |
| `type` | VARCHAR(255) | 市場別 |
| `date` | DATE | 日期 |

#### `TaiwanStockInfoWithWarrantSummary` (12 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `stock_id` | VARCHAR(255) | 證券代號 |
| `date` | DATE | 日期 |
| `close` | NUMERIC(20,6) | 權證收盤價 |
| `target_stock_id` | NUMERIC(20,6) | 標的證券代號 |
| `target_close` | NUMERIC(20,6) | 標的收盤價 |
| `type` | VARCHAR(255) | 權證類別 |
| `fulfillment_method` | VARCHAR(255) | 履約方式 |
| `end_date` | DATE | 結束日 |
| `fulfillment_start_date` | DATE | 履約起日 |
| `fulfillment_end_date` | DATE | 履約迄日 |
| `exercise_ratio` | NUMERIC(20,6) | 行使比例 |
| `fulfillment_price` | NUMERIC(20,6) | 履約價 |

#### `TaiwanStockInstitutionalInvestorsBuySell` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `buy` | NUMERIC(20,6) | 買進(股/金額) |
| `name` | VARCHAR(255) | 法人別(Foreign_Investor/Investment_Trust/Dealer_self/Dealer_Hedging/Foreign_Dealer_Self) |
| `sell` | NUMERIC(20,6) | 賣出(股/金額) |

#### `TaiwanStockLoanCollateralBalance` (37 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `market` | VARCHAR(255) | 市場別 |
| `MarginPreviousDayBalance` | NUMERIC(20,6) | 融資前日餘額 |
| `MarginBuy` | NUMERIC(20,6) | 融資買進 |
| `MarginSell` | NUMERIC(20,6) | 融資賣出 |
| `MarginCashRedemption` | NUMERIC(20,6) | 融資現金償還 |
| `MarginCurrentDayBalance` | NUMERIC(20,6) | 融資當日餘額 |
| `MarginNextDayQuota` | NUMERIC(20,6) | 融資次日限額 |
| `SecuritiesFirmLoanPreviousDayBalance` | NUMERIC(20,6) | 券商借貸前日餘額 |
| `SecuritiesFirmLoanBuy` | NUMERIC(20,6) | 券商借貸買進 |
| `SecuritiesFirmLoanSell` | NUMERIC(20,6) | 券商借貸賣出 |
| `SecuritiesFirmLoanCashRedemption` | NUMERIC(20,6) | 券商借貸現金償還 |
| `SecuritiesFirmLoanReplacement` | NUMERIC(20,6) | 券商借貸補回 |
| `SecuritiesFirmLoanCurrentDayBalance` | NUMERIC(20,6) | 券商借貸當日餘額 |
| `SecuritiesFirmLoanNextDayQuota` | NUMERIC(20,6) | 券商借貸次日限額 |
| `UnrestrictedLoanPreviousDayBalance` | NUMERIC(20,6) | 無擔保借貸前日餘額 |
| `UnrestrictedLoanBuy` | NUMERIC(20,6) | 無擔保借貸買進 |
| `UnrestrictedLoanSell` | NUMERIC(20,6) | 無擔保借貸賣出 |
| `UnrestrictedLoanCashRedemption` | NUMERIC(20,6) | 無擔保借貸現金償還 |
| `UnrestrictedLoanReplacement` | NUMERIC(20,6) | 無擔保借貸補回 |
| `UnrestrictedLoanCurrentDayBalance` | NUMERIC(20,6) | 無擔保借貸當日餘額 |
| `UnrestrictedLoanNextDayQuota` | NUMERIC(20,6) | 無擔保借貸次日限額 |
| `SecuritiesFinanceSecuredLoanPreviousDayBalance` | NUMERIC(20,6) | 證金擔保借貸前日餘額 |
| `SecuritiesFinanceSecuredLoanBuy` | NUMERIC(20,6) | 證金擔保借貸買進 |
| `SecuritiesFinanceSecuredLoanSell` | NUMERIC(20,6) | 證金擔保借貸賣出 |
| `SecuritiesFinanceSecuredLoanCashRedemption` | NUMERIC(20,6) | 證金擔保借貸現金償還 |
| `SecuritiesFinanceSecuredLoanReplacement` | NUMERIC(20,6) | 證金擔保借貸補回 |
| `SecuritiesFinanceSecuredLoanCurrentDayBalance` | NUMERIC(20,6) | 證金擔保借貸當日餘額 |
| `SecuritiesFinanceSecuredLoanNextDayQuota` | NUMERIC(20,6) | 證金擔保借貸次日限額 |
| `SettlementMarginPreviousDayBalance` | NUMERIC(20,6) | 結算保證金前日餘額 |
| `SettlementMarginBuy` | NUMERIC(20,6) | 結算保證金買進 |
| `SettlementMarginSell` | NUMERIC(20,6) | 結算保證金賣出 |
| `SettlementMarginCashRedemption` | NUMERIC(20,6) | 結算保證金現金償還 |
| `SettlementMarginReplacement` | NUMERIC(20,6) | 結算保證金補回 |
| `SettlementMarginCurrentDayBalance` | NUMERIC(20,6) | 結算保證金當日餘額 |
| `SettlementMarginNextDayQuota` | NUMERIC(20,6) | 結算保證金次日限額 |

#### `TaiwanStockMarginPurchaseShortSale` (16 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `MarginPurchaseBuy` | NUMERIC(20,6) | 融資買進 |
| `MarginPurchaseCashRepayment` | NUMERIC(20,6) | 融資現金償還 |
| `MarginPurchaseLimit` | NUMERIC(20,6) | 融資限額 |
| `MarginPurchaseSell` | NUMERIC(20,6) | 融資賣出 |
| `MarginPurchaseTodayBalance` | NUMERIC(20,6) | 融資今日餘額 |
| `MarginPurchaseYesterdayBalance` | NUMERIC(20,6) | 融資昨日餘額 |
| `Note` | VARCHAR(255) | 註記 |
| `OffsetLoanAndShort` | NUMERIC(20,6) | 資券互抵 |
| `ShortSaleBuy` | NUMERIC(20,6) | 融券買進 |
| `ShortSaleCashRepayment` | NUMERIC(20,6) | 融券現券償還 |
| `ShortSaleLimit` | NUMERIC(20,6) | 融券限額 |
| `ShortSaleSell` | NUMERIC(20,6) | 融券賣出 |
| `ShortSaleTodayBalance` | NUMERIC(20,6) | 融券今日餘額 |
| `ShortSaleYesterdayBalance` | NUMERIC(20,6) | 融券昨日餘額 |

#### `TaiwanStockMarginShortSaleSuspension` (4 欄) — ✓ 2026-06-08 live 驗證(data_id=2330)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `stock_id` | VARCHAR(255) | 證券代號 |
| `date` | DATE | 日期 |
| `end_date` | DATE | 結束日 |
| `reason` | VARCHAR(255) | 事由 |

#### `TaiwanStockMarketValue` (3 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `market_value` | NUMERIC(24,6) | 市值(元) |

#### `TaiwanStockMarketValueWeight` (6 欄) — ✓ 2026-06-08 live 驗證(data_id=2330,per-stock 指數權重)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `rank` | NUMERIC(20,6) | 排名 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |
| `weight_per` | NUMERIC(20,6) | 權重(%) |
| `date` | DATE | 日期 |
| `type` | VARCHAR(255) | 市場別(twse/tpex) |

#### `TaiwanStockMonthPrice` (11 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `stock_id` | VARCHAR(255) | 證券代號 |
| `ymonth` | VARCHAR(255) | 年月 |
| `max` | NUMERIC(20,6) | 最高價 |
| `min` | NUMERIC(20,6) | 最低價 |
| `trading_volume` | NUMERIC(20,6) | 成交量(股) |
| `trading_money` | NUMERIC(23,6) | 成交金額(元) |
| `trading_turnover` | NUMERIC(20,6) | 成交筆數 |
| `date` | DATE | 日期 |
| `close` | NUMERIC(20,6) | 收盤價 |
| `open` | NUMERIC(20,6) | 開盤價 |
| `spread` | NUMERIC(20,6) | 漲跌價差 |

#### `TaiwanStockMonthRevenue` (7 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `country` | VARCHAR(255) | 地區/國別 |
| `revenue` | NUMERIC(22,6) | 當月營收(元) |
| `revenue_month` | NUMERIC(20,6) | 營收所屬月份 |
| `revenue_year` | NUMERIC(20,6) | 營收所屬年度 |
| `create_time` | DATE | 資料公佈日 |

#### `TaiwanStockNews` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `link` | VARCHAR(592) | 新聞連結 |
| `source` | VARCHAR(255) | 來源 |
| `title` | VARCHAR(255) | 標題 |

#### `TaiwanStockPER` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `dividend_yield` | NUMERIC(20,6) | 殖利率(%) |
| `PER` | NUMERIC(20,6) | 本益比 |
| `PBR` | NUMERIC(20,6) | 股價淨值比 |

#### `TaiwanStockParValueChange` (8 欄 · 最早資料 2019-09-09)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |
| `before_close` | NUMERIC(20,6) | 變更前收盤 |
| `after_ref_close` | NUMERIC(20,6) | 變更後參考收盤 |
| `after_ref_max` | NUMERIC(20,6) | 變更後參考漲停 |
| `after_ref_min` | NUMERIC(20,6) | 變更後參考跌停 |
| `after_ref_open` | NUMERIC(20,6) | 變更後參考開盤 |

#### `TaiwanStockPrice` (10 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Trading_Volume` | NUMERIC(20,6) | 成交股數 |
| `Trading_money` | NUMERIC(22,6) | 成交金額(元) |
| `open` | NUMERIC(20,6) | 開盤價 |
| `max` | NUMERIC(20,6) | 最高價 |
| `min` | NUMERIC(20,6) | 最低價 |
| `close` | NUMERIC(20,6) | 收盤價 |
| `spread` | NUMERIC(20,6) | 漲跌價差 |
| `Trading_turnover` | NUMERIC(20,6) | 成交筆數 |

#### `TaiwanStockPriceAdj` (10 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Trading_Volume` | NUMERIC(20,6) | 成交股數 |
| `Trading_money` | NUMERIC(22,6) | 成交金額(元) |
| `open` | NUMERIC(20,6) | 開盤價 |
| `max` | NUMERIC(20,6) | 最高價 |
| `min` | NUMERIC(20,6) | 最低價 |
| `close` | NUMERIC(20,6) | 收盤價 |
| `spread` | NUMERIC(20,6) | 漲跌價差 |
| `Trading_turnover` | NUMERIC(20,6) | 成交筆數 |

#### `TaiwanStockPriceLimit` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `reference_price` | NUMERIC(20,6) | 參考價 |
| `limit_up` | NUMERIC(20,6) | 漲停價 |
| `limit_down` | NUMERIC(20,6) | 跌停價 |

#### `TaiwanStockSecuritiesLending` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `transaction_type` | VARCHAR(255) | 交易類別 |
| `volume` | NUMERIC(20,6) | 成交量/成交口數 |
| `fee_rate` | NUMERIC(20,6) | 費率(%) |
| `close` | NUMERIC(20,6) | 收盤價 |
| `original_return_date` | DATE | 原訂返還日 |
| `original_lending_period` | NUMERIC(20,6) | 原訂借券期間 |

#### `TaiwanStockShareholding` (13 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |
| `InternationalCode` | VARCHAR(255) | ISIN 國際證券識別碼 |
| `ForeignInvestmentRemainingShares` | NUMERIC(20,6) | 外資尚可投資股數 |
| `ForeignInvestmentShares` | NUMERIC(21,6) | 外資已持有股數 |
| `ForeignInvestmentRemainRatio` | NUMERIC(20,6) | 外資尚可投資比率(%) |
| `ForeignInvestmentSharesRatio` | NUMERIC(20,6) | 外資持股比率(%) |
| `ForeignInvestmentUpperLimitRatio` | NUMERIC(20,6) | 外資投資上限比率(%) |
| `ChineseInvestmentUpperLimitRatio` | NUMERIC(20,6) | 陸資投資上限比率(%) |
| `NumberOfSharesIssued` | NUMERIC(21,6) | 已發行股數 |
| `RecentlyDeclareDate` | DATE | 最近申報日 |
| `note` | VARCHAR(255) | 註記 |

#### `TaiwanStockSplitPrice` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `type` | VARCHAR(255) | 分割類別 |
| `before_price` | NUMERIC(20,6) | 除權息前價 |
| `after_price` | NUMERIC(20,6) | 除權息後價 |
| `max_price` | NUMERIC(20,6) | 最高價 |
| `min_price` | NUMERIC(20,6) | 最低價 |
| `open_price` | NUMERIC(20,6) | 開盤價 |

#### `TaiwanStockSuspended` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `suspension_time` | VARCHAR(255) | 暫停交易時間 |
| `resumption_date` | DATE | 恢復交易日 |
| `resumption_time` | VARCHAR(255) | 恢復交易時間 |

#### `TaiwanStockTotalInstitutionalInvestors` (4 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `buy` | NUMERIC(23,6) | 買進(元) |
| `date` | DATE | 日期 |
| `name` | VARCHAR(255) | 法人別 |
| `sell` | NUMERIC(23,6) | 賣出(元) |

#### `TaiwanStockTotalMarginPurchaseShortSale` (7 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `TodayBalance` | NUMERIC(22,6) | 今日餘額 |
| `YesBalance` | NUMERIC(22,6) | 昨日餘額 |
| `buy` | NUMERIC(21,6) | 買進 |
| `date` | DATE | 日期 |
| `name` | VARCHAR(255) | 類別(融資/融券) |
| `Return` | NUMERIC(20,6) | 償還 |
| `sell` | NUMERIC(21,6) | 賣出 |

#### `TaiwanStockTotalReturnIndex` (3 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `price` | NUMERIC(20,6) | 價格 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `date` | DATE | 日期 |

#### `TaiwanStockTradingDailyReport` (7 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `securities_trader_id` | VARCHAR(255) | 券商代號 |
| `securities_trader` | VARCHAR(255) | 券商名稱 |
| `price` | NUMERIC(20,6) | 價格 |
| `buy` | NUMERIC(20,6) | 買進 |
| `sell` | NUMERIC(20,6) | 賣出 |

#### `TaiwanStockTradingDate` (1 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |

#### `TaiwanStockWeekPrice` (11 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `stock_id` | VARCHAR(255) | 證券代號 |
| `yweek` | VARCHAR(255) | 年週 |
| `max` | NUMERIC(20,6) | 最高價 |
| `min` | NUMERIC(20,6) | 最低價 |
| `trading_volume` | NUMERIC(20,6) | 成交量(股) |
| `trading_money` | NUMERIC(22,6) | 成交金額(元) |
| `trading_turnover` | NUMERIC(20,6) | 成交筆數 |
| `date` | DATE | 日期 |
| `close` | NUMERIC(20,6) | 收盤價 |
| `open` | NUMERIC(20,6) | 開盤價 |
| `spread` | NUMERIC(20,6) | 漲跌價差 |

#### `TaiwanTotalExchangeMarginMaintenance` (2 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `TotalExchangeMarginMaintenance` | NUMERIC(20,6) | 整體市場維持率(%) |

## FinMind — 非 Taiwan(美/歐/日/英股 + 指數/商品/匯率/利率)
#### `CnnFearGreedIndex` (3 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `fear_greed` | NUMERIC(20,6) | 恐懼貪婪指數 |
| `fear_greed_emotion` | VARCHAR(255) | 情緒分類 |

#### `CrudeOilPrices` (3 欄 · 最早資料 1986-01-02) — ✓ 2026-06-08 live 驗證(data_id=WTI)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `name` | VARCHAR(255) | 油品名稱(WTI/Brent) |
| `price` | NUMERIC(20,6) | 價格 |

#### `EuropeStockInfo` (4 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Market` | VARCHAR(255) | 市場 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |

#### `EuropeStockPrice` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Open` | NUMERIC(20,6) | 開盤價 |
| `High` | NUMERIC(20,6) | 最高價 |
| `Low` | NUMERIC(20,6) | 最低價 |
| `Close` | NUMERIC(20,6) | 收盤價 |
| `Adj_Close` | NUMERIC(20,6) | 還原收盤價 |
| `Volume` | NUMERIC(20,6) | 成交量(股) |

#### `ExchangeRate` (4 欄) — ᵈ(2026-06-08 API 試 USD/EUR/JPY/CNY/country/無-data_id 皆回 0 rows〔HTTP 200 success〕→ 疑 deprecated/空集;schema=建檔實打)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `InterbankRate` | NUMERIC(20,6) | 銀行間匯率 |
| `InverseInterbankRate` | NUMERIC(20,6) | 反向銀行間匯率 |
| `country` | VARCHAR(255) | 國別/幣別 |
| `date` | DATE | 日期 |

#### `GoldPrice` (2 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `Price` | NUMERIC(20,6) | 價格 |

#### `InterestRate` (4 欄 · 最早資料 2024-01-31) — ✓ 2026-06-08 live 驗證(不帶 data_id,start≥2024-02-01)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `country` | VARCHAR(255) | 地區/國別 |
| `date` | DATE | 日期 |
| `full_country_name` | VARCHAR(255) | 國家全名 |
| `interest_rate` | NUMERIC(20,6) | 利率(%) |

#### `JapanStockInfo` (5 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Exchange` | VARCHAR(255) | 交易所 |
| `Sector` | VARCHAR(255) | 產業別 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |

#### `JapanStockPrice` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Adj_Close` | NUMERIC(20,6) | 還原收盤價 |
| `Close` | NUMERIC(20,6) | 收盤價 |
| `High` | NUMERIC(20,6) | 最高價 |
| `Low` | NUMERIC(20,6) | 最低價 |
| `Open` | NUMERIC(20,6) | 開盤價 |
| `Volume` | NUMERIC(20,6) | 成交量(股) |

#### `UKStockInfo` (4 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Country` | VARCHAR(255) | 國別 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |

#### `UKStockPrice` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Open` | NUMERIC(20,6) | 開盤價 |
| `High` | NUMERIC(20,6) | 最高價 |
| `Low` | NUMERIC(20,6) | 最低價 |
| `Close` | NUMERIC(20,6) | 收盤價 |
| `Adj_Close` | NUMERIC(20,6) | 還原收盤價 |
| `Volume` | NUMERIC(20,6) | 成交量(股) |

#### `USStockInfo` (7 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `stock_name` | VARCHAR(255) | 證券簡稱 |
| `Country` | VARCHAR(255) | 國別 |
| `IPOYear` | NUMERIC(20,6) | 上市年 |
| `MarketCap` | NUMERIC(24,6) | 市值 |
| `Subsector` | VARCHAR(255) | 次產業 |

#### `USStockPrice` (8 欄)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `stock_id` | VARCHAR(255) | 證券代號 |
| `Adj_Close` | NUMERIC(20,6) | 還原收盤價 |
| `Close` | NUMERIC(20,6) | 收盤價 |
| `High` | NUMERIC(20,6) | 最高價 |
| `Low` | NUMERIC(20,6) | 最低價 |
| `Open` | NUMERIC(20,6) | 開盤價 |
| `Volume` | NUMERIC(20,6) | 成交量(股) |

## FRED（2 表,均 generic auto-schema;PK `(series_id, date)`)
#### `FredData` (5 欄) — `sovereign_sync_engine.sync_fred` 路徑,4 series
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `date` | DATE | 日期 |
| `series_id` | VARCHAR(255) | FRED 指標代號 |
| `value` | NUMERIC(20,6) | 觀測值 |
| `realtime_start` | DATE | ALFRED vintage 起(資料版本可得起日) |
| `realtime_end` | DATE | ALFRED vintage 迄 |

**`FredData` 4 series**(`FRED_LIST`):DFF/UNRATE/T10Y2Y/VIXCLS。

#### `fred_series` (3 欄) — `fetch_fred_data.py` 路徑,24 series(`feature_store_builder` K-wave 唯一來源)
| 欄位 | 型態及大小 | 中文定義 |
|---|---|---|
| `series_id` | VARCHAR(255) | FRED 指標代號 |
| `date` | DATE | 日期 |
| `value` | NUMERIC(20,6) | 觀測值 |

**`fred_series` 24 series**(`DEFAULT_FRED_SERIES`):T10Y2Y/T10Y3M/T10YIE/VIXCLS/BAMLH0A0HYM2/DTWEXBGS/M2SL/DGS10/DGS2/DGS3MO/UMCSENT/INDPRO/UNRATE/CPIAUCSL/PATENTUSALLTOTAL/B985RC1Q027SBEA/TCMDO/LFWA64TTUSA647N/SPPOPDPNDOLUSA/PALLFNFINDEXQ/QUSPAM770A/WTISPLC/IPG3344S/PCU4831114831115(定義見 §14.7-DJ 前述)。⚠️ 僅 3 欄(payload `{series_id,date,value}`,**無 realtime_***),與 `FredData`(5 欄)結構不同。

## 待解 3(需升 tier / 事件稀少 / 未知 country 參數,非不存在)
- `TaiwanStockBlockTradingDailyReport`
- `TaiwanStockWarrantTradingDailyReport`
- `GovernmentBondsYield`

## intraday 排除 9(日為最小單位,§14.7-DI T_DI-7)
- `TaiwanStockPriceTick` / `TaiwanStockKBar` / `TaiwanStockStatisticsOfOrderBookAndTrade` / `TaiwanVariousIndicators5Seconds` / `TaiwanStockEvery5SecondsIndex` / `TaiwanFuturesTick` / `TaiwanOptionTick` / `TaiwanFutOptTickInfo` / `USStockPriceMinute`