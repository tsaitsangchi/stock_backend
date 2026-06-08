# 通用 Ingester 資料表完整字典 (Generic-Ingested Table Data Dictionary)

**日期**:2026-06-08 | **治權**:主憲章 §14.7-DJ 之 companion 完整檔 | **source**:§一.10 source-traceable(FinMind/FRED API + DB information_schema,2330 樣本 sync 實證)

本檔為「經 FinMind API + FRED API 逐一建立、餵特徵庫之 12 feature-input 表」之完整資料字典(table 名 + 欄位名 + 型別 + 定義 + 長表 type 枚舉)。權威同主憲章 §14.7-DJ §二。

---

**通則**:全部資料 source = FinMind API(`api.finmindtrade.com/api/v4/data`)/ FRED API(`api.stlouisfed.org`),DB-verified 2026-06-08(以 2330 台積電樣本實際 sync 取得,§一.10 (b)(c));型別由 generic auto-schema **依實際觀測值動態推導**(字串≥VARCHAR(100)、數字≥NUMERIC(20,6),值超界自動加大——故 `Trading_money` NUMERIC(22,6)、`value` NUMERIC(23,6)等比舊硬編更貼合真實值、防截斷);per-stock 表 PK 由 detect_keys 自 API 偵測;**無 synthetic/impute 值**(§14.7-DC source-pure)。

**計數**:generic-ingester-built FinMind feature-input 表 = **11**(10 per-stock + TaiwanStockInfo 名冊);FRED-API 表 = **1**(FredData,explicit-DDL 路徑);= **12 feature-input 表**。另 **54 探索性 FinMind extras**(§14.7-DI T_DI-8 通用 ingester 建,deferred,**非特徵輸入**,本字典不展開)。

#### 1. `TaiwanStockInfo` — 市場個股基本資料 / 全市場名冊(roster)
- **來源**:FinMind TaiwanStockInfo(市場級,data_id 空) / **頻率**:快照(每次 sync 覆寫最新) / **PK**:['stock_id', 'type', 'industry_category'] / **用途**:roster 解析(get_db_stock_ids)+ 產業別/市場別過濾

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `industry_category` | character varying(100) | 產業類別(半導體業/金融保險業/…;Index 為大盤指數列) |
| `stock_id` | character varying(100) | 證券代號(e.g. 2330) |
| `stock_name` | character varying(100) | 證券簡稱(台積電) |
| `type` | character varying(100) | 市場別(twse=上市/tpex=上櫃/rotc/psl…) |
| `date` | date | 資料快照日 |

#### 2. `TaiwanStockPrice` — 未還原日線行情
- **來源**:FinMind TaiwanStockPrice / **頻率**:日(交易日) / **PK**:['stock_id', 'date'] / **用途**:成交量/流動性;原始未還原價(報酬計算主用 PriceAdj)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 交易日 |
| `stock_id` | character varying(100) | 證券代號 |
| `Trading_Volume` | numeric(20,6) | 成交股數(股) |
| `Trading_money` | numeric(22,6) | 成交金額(元) |
| `open` | numeric(20,6) | 開盤價(元) |
| `max` | numeric(20,6) | 最高價(元) |
| `min` | numeric(20,6) | 最低價(元) |
| `close` | numeric(20,6) | 收盤價(元) |
| `spread` | numeric(20,6) | 漲跌價差(收盤−前一交易日收盤) |
| `Trading_turnover` | numeric(20,6) | 成交筆數 |

#### 3. `TaiwanStockPriceAdj` — 還原權息日線行情
- **來源**:FinMind TaiwanStockPriceAdj / **頻率**:日(交易日) / **PK**:['stock_id', 'date'] / **用途**:連續報酬/波動/動能等價格特徵之權威來源(避免除權息跳空假訊號)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 交易日 |
| `stock_id` | character varying(100) | 證券代號 |
| `Trading_Volume` | numeric(20,6) | 成交股數(股) |
| `Trading_money` | numeric(22,6) | 成交金額(元) |
| `open` | numeric(20,6) | 還原開盤價 |
| `max` | numeric(20,6) | 還原最高價 |
| `min` | numeric(20,6) | 還原最低價 |
| `close` | numeric(20,6) | 還原收盤價 |
| `spread` | numeric(20,6) | 還原漲跌價差 |
| `Trading_turnover` | numeric(20,6) | 成交筆數 |

#### 4. `TaiwanStockPER` — 本益比/淨值比/殖利率
- **來源**:FinMind TaiwanStockPER / **頻率**:日(交易日) / **PK**:['stock_id', 'date'] / **用途**:估值特徵(PE/PB/殖利率)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 交易日 |
| `stock_id` | character varying(100) | 證券代號 |
| `dividend_yield` | numeric(20,6) | 殖利率(%)=近12月現金股利/股價 |
| `PER` | numeric(20,6) | 本益比=股價/近4季EPS |
| `PBR` | numeric(20,6) | 股價淨值比=股價/每股淨值 |

#### 5. `TaiwanStockInstitutionalInvestorsBuySell` — 三大法人買賣超
- **來源**:FinMind TaiwanStockInstitutionalInvestorsBuySell / **頻率**:日(交易日) / **PK**:['stock_id', 'date', 'name'] / **用途**:法人資金流特徵(外資/投信買賣超);net=buy−sell

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 交易日 |
| `stock_id` | character varying(100) | 證券代號 |
| `buy` | numeric(20,6) | 買進(股或金額) |
| `name` | character varying(100) | 法人別(見枚舉) |
| `sell` | numeric(20,6) | 賣出(股或金額) |

**`name` 法人別枚舉(DB-verified 5)**:
- `Dealer_Hedging` = 自營商(避險)
- `Dealer_self` = 自營商(自行買賣)
- `Foreign_Dealer_Self` = 外資自營商
- `Foreign_Investor` = 外資及陸資(不含外資自營商)
- `Investment_Trust` = 投信(證券投資信託)

#### 6. `TaiwanStockMarginPurchaseShortSale` — 融資融券
- **來源**:FinMind TaiwanStockMarginPurchaseShortSale / **頻率**:日(交易日) / **PK**:['stock_id', 'date'] / **用途**:信用交易特徵(融資/融券餘額變化)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 交易日 |
| `stock_id` | character varying(100) | 證券代號 |
| `MarginPurchaseBuy` | numeric(20,6) | 融資買進(張) |
| `MarginPurchaseCashRepayment` | numeric(20,6) | 融資現金償還(張) |
| `MarginPurchaseLimit` | numeric(20,6) | 融資限額(張) |
| `MarginPurchaseSell` | numeric(20,6) | 融資賣出(張) |
| `MarginPurchaseTodayBalance` | numeric(20,6) | 融資今日餘額(張) |
| `MarginPurchaseYesterdayBalance` | numeric(20,6) | 融資昨日餘額(張) |
| `Note` | character varying(100) | 註記(X=處置/異常標記等) |
| `OffsetLoanAndShort` | numeric(20,6) | 資券互抵(張) |
| `ShortSaleBuy` | numeric(20,6) | 融券買進/回補(張) |
| `ShortSaleCashRepayment` | numeric(20,6) | 融券現券償還(張) |
| `ShortSaleLimit` | numeric(20,6) | 融券限額(張) |
| `ShortSaleSell` | numeric(20,6) | 融券賣出(張) |
| `ShortSaleTodayBalance` | numeric(20,6) | 融券今日餘額(張) |
| `ShortSaleYesterdayBalance` | numeric(20,6) | 融券昨日餘額(張) |

#### 7. `TaiwanStockShareholding` — 外資及陸資持股
- **來源**:FinMind TaiwanStockShareholding / **頻率**:週(申報) / **PK**:['stock_id', 'date'] / **用途**:外資持股比率/趨勢特徵

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 申報日 |
| `stock_id` | character varying(100) | 證券代號 |
| `stock_name` | character varying(100) | 證券簡稱 |
| `InternationalCode` | character varying(100) | 國際證券識別碼 ISIN(TW0002330008) |
| `ForeignInvestmentRemainingShares` | numeric(20,6) | 外資尚可投資股數 |
| `ForeignInvestmentShares` | numeric(21,6) | 外資已持有股數 |
| `ForeignInvestmentRemainRatio` | numeric(20,6) | 外資尚可投資比率(%) |
| `ForeignInvestmentSharesRatio` | numeric(20,6) | 外資持股比率(%) |
| `ForeignInvestmentUpperLimitRatio` | numeric(20,6) | 外資投資上限比率(%) |
| `ChineseInvestmentUpperLimitRatio` | numeric(20,6) | 陸資投資上限比率(%) |
| `NumberOfSharesIssued` | numeric(21,6) | 已發行股數 |
| `RecentlyDeclareDate` | date | 最近申報日 |
| `note` | character varying(155) | 註記 |

#### 8. `TaiwanStockFinancialStatements` — 綜合損益表(long format,科目逐列)
- **來源**:FinMind TaiwanStockFinancialStatements / **頻率**:季(quarterly) / **PK**:['stock_id', 'date', 'type'] / **用途**:基本面特徵(EPS/營收/毛利/營益/稅後淨利/母公司權益…)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 財報日(季底) |
| `stock_id` | character varying(100) | 證券代號 |
| `type` | character varying(100) | 科目英文代號(見枚舉) |
| `value` | numeric(23,6) | 金額(元;EPS 為元/股) |
| `origin_name` | character varying(100) | 科目中文名(權威定義,源自 API) |

**`type` 科目枚舉(2330 樣本 DB-verified,17 型;`origin_name` 即權威中文定義)**:
```
ComprehensiveIncomeConsolidatedNetIncomeAttributedNonControllingInterest = 綜合損益總額歸屬於非控制權益
CostOfGoodsSold = 營業成本
EPS = 基本每股盈餘
EquityAttributableToOwnersOfParent = 淨利（淨損）歸屬於母公司業主
GrossProfit = 營業毛利（毛損）
IncomeAfterTaxes = 本期淨利（淨損）
IncomeFromContinuingOperations = 繼續營業單位本期淨利（淨損）
NoncontrollingInterests = 淨利（淨損）歸屬於非控制權益
OTHNOE = 其他收益及費損淨額
OperatingExpenses = 營業費用
OperatingIncome = 營業利益（損失）
OtherComprehensiveIncome = 其他綜合損益（淨額）
PreTaxIncome = 稅前淨利（淨損）
Revenue = 營業收入
TAX = 所得稅費用（利益）
TotalConsolidatedProfitForThePeriod = 本期綜合損益總額
TotalNonoperatingIncomeAndExpense = 營業外收入及支出
```

#### 9. `TaiwanStockBalanceSheet` — 資產負債表(long format,科目逐列;含 _per=占總額百分比變體)
- **來源**:FinMind TaiwanStockBalanceSheet / **頻率**:季(quarterly) / **PK**:['stock_id', 'date', 'type'] / **用途**:權益/槓桿/資產結構特徵(權益總額/資產總額/流動資產/負債/保留盈餘…;ROE 等)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 財報日(季底) |
| `stock_id` | character varying(100) | 證券代號 |
| `type` | character varying(110) | 科目英文代號(見枚舉;_per 後綴=占總資產/總額百分比) |
| `value` | numeric(23,6) | 金額(元)或百分比(_per 列) |
| `origin_name` | character varying(100) | 科目中文名(權威定義,源自 API) |

**`type` 科目枚舉(2330 樣本 DB-verified,101 型;`origin_name` 即權威中文定義)**:
```
AccountsPayable = 應付帳款
AccountsPayableToRelatedParties = 應付帳款－關係人
AccountsPayableToRelatedParties_per = 應付帳款－關係人
AccountsPayable_per = 應付帳款
AccountsReceivableDuefromRelatedPartiesNet = 應收帳款－關係人淨額
AccountsReceivableDuefromRelatedPartiesNet_per = 應收帳款－關係人淨額
AccountsReceivableNet = 應收帳款淨額
AccountsReceivableNet_per = 應收帳款淨額
BondsPayable = 應付公司債
BondsPayable_per = 應付公司債
CapitalStock = 股本合計
CapitalStock_per = 股本合計
CapitalSurplus = 資本公積合計
CapitalSurplusAdditionalPaidInCapital = 資本公積－發行溢價
CapitalSurplusAdditionalPaidInCapital_per = 資本公積－發行溢價
CapitalSurplusChangesInEquityOfAssociatesAndJointVenturesAccountedForUsingEquityMethod = 資本公積－採用權益法認列關聯企業及合資股權淨值之變動數
CapitalSurplusChangesInEquityOfAssociatesAndJointVenturesAccountedForUsingEquityMethod_per = 資本公積－採用權益法認列關聯企業及合資股權淨值之變動數
CapitalSurplusDonatedAssetsReceived = 資本公積－受贈資產
CapitalSurplusDonatedAssetsReceived_per = 資本公積－受贈資產
CapitalSurplusNetAssetsFromMerger = 資本公積－合併溢額
CapitalSurplusNetAssetsFromMerger_per = 資本公積－合併溢額
CapitalSurplus_per = 資本公積合計
CashAndCashEquivalents = 現金及約當現金
CashAndCashEquivalents_per = 現金及約當現金
CurrentAssets = 流動資產合計
CurrentAssets_per = 流動資產合計
CurrentDerivativeFinancialLiabilitiesForHedging = 避險之金融負債－流動
CurrentDerivativeFinancialLiabilitiesForHedging_per = 避險之金融負債－流動
CurrentFinancialAssetsAtFairvalueThroughProfitOrLoss = 透過損益按公允價值衡量之金融資產－流動
CurrentFinancialAssetsAtFairvalueThroughProfitOrLoss_per = 透過損益按公允價值衡量之金融資產－流動
CurrentFinancialLiabilitiesAtFairValueThroughProfitOrLoss = 透過損益按公允價值衡量之金融負債－流動
CurrentFinancialLiabilitiesAtFairValueThroughProfitOrLoss_per = 透過損益按公允價值衡量之金融負債－流動
CurrentLiabilities = 流動負債合計
CurrentLiabilities_per = 流動負債合計
CurrentTaxLiabilities = 本期所得稅負債
CurrentTaxLiabilities_per = 本期所得稅負債
DeferredTaxAssets = 遞延所得稅資產
DeferredTaxAssets_per = 遞延所得稅資產
Equity = 權益總額
EquityAttributableToOwnersOfParent = 歸屬於母公司業主之權益合計
EquityAttributableToOwnersOfParent_per = 歸屬於母公司業主之權益合計
Equity_per = 權益總額
FinancialAssetsAtAmortizedCost = 按攤銷後成本衡量之金融資產－流動
FinancialAssetsAtAmortizedCostNonCurrent = 按攤銷後成本衡量之金融資產－非流動
FinancialAssetsAtAmortizedCostNonCurrent_per = 按攤銷後成本衡量之金融資產－非流動
FinancialAssetsAtAmortizedCost_per = 按攤銷後成本衡量之金融資產－流動
FinancialAssetsAtFairvalueThroughOtherComprehensiveIncome = 透過其他綜合損益按公允價值衡量之金融資產－流動
FinancialAssetsAtFairvalueThroughOtherComprehensiveIncomeNonCurrent = 透過其他綜合損益按公允價值衡量之金融資產－非流動
FinancialAssetsAtFairvalueThroughOtherComprehensiveIncomeNonCurrent_per = 透過其他綜合損益按公允價值衡量之金融資產－非流動
FinancialAssetsAtFairvalueThroughOtherComprehensiveIncome_per = 透過其他綜合損益按公允價值衡量之金融資產－流動
HedgingAinancialAssets = 避險之金融資產－流動
HedgingAinancialAssets_per = 避險之金融資產－流動
IntangibleAssets = 無形資產
IntangibleAssets_per = 無形資產
Inventories = 存貨
Inventories_per = 存貨
InvestmentAccountedForUsingEquityMethod = 採用權益法之投資
InvestmentAccountedForUsingEquityMethod_per = 採用權益法之投資
LegalReserve = 法定盈餘公積
LegalReserve_per = 法定盈餘公積
Liabilities = 負債總額
Liabilities_per = 負債總額
LongtermBorrowings = 長期借款
LongtermBorrowings_per = 長期借款
NonCurrentFinancialAssetsAtFairvalueThroughProfitOrLoss = 透過損益按公允價值衡量之金融資產－非流動
NonCurrentFinancialAssetsAtFairvalueThroughProfitOrLoss_per = 透過損益按公允價值衡量之金融資產－非流動
NoncontrollingInterests = 非控制權益
NoncontrollingInterests_per = 非控制權益
NoncurrentAssets = 非流動資產合計
NoncurrentAssets_per = 非流動資產合計
NoncurrentLiabilities = 非流動負債合計
NoncurrentLiabilities_per = 非流動負債合計
NumberOfSharesInEntityHeldByEntityAndByItsSubsidiaries = 母公司暨子公司所持有之母公司庫藏股股數
OrdinaryShare = 普通股股本
OrdinaryShare_per = 普通股股本
OtherCurrentAssets = 其他流動資產
OtherCurrentAssets_per = 其他流動資產
OtherCurrentLiabilities = 其他流動負債
OtherCurrentLiabilities_per = 其他流動負債
OtherEquityInterest = 其他權益合計
OtherEquityInterest_per = 其他權益合計
OtherNoncurrentAssets = 其他非流動資產
OtherNoncurrentAssets_per = 其他非流動資產
OtherNoncurrentLiabilities = 其他非流動負債
OtherNoncurrentLiabilities_per = 其他非流動負債
OtherPayables = 其他應付款
OtherPayables_per = 其他應付款
OtherReceivablesDueFromRelatedParties = 其他應收款－關係人淨額
OtherReceivablesDueFromRelatedParties_per = 其他應收款－關係人淨額
PropertyPlantAndEquipment = 不動產、廠房及設備
PropertyPlantAndEquipment_per = 不動產、廠房及設備
RetainedEarnings = 保留盈餘合計
RetainedEarnings_per = 保留盈餘合計
RightOfUseAsset = 使用權資產
RightOfUseAsset_per = 使用權資產
TotalAssets = 資產總額
TotalAssets_per = 資產總額
TotalLiabilitiesEquity = 負債及權益總計
TotalLiabilitiesEquity_per = 負債及權益總計
UnappropriatedRetainedEarningsAaccumulatedDeficit = 未分配盈餘
UnappropriatedRetainedEarningsAaccumulatedDeficit_per = 未分配盈餘
```

#### 10. `TaiwanStockMonthRevenue` — 月營收
- **來源**:FinMind TaiwanStockMonthRevenue / **頻率**:月(monthly) / **PK**:['stock_id', 'date'] / **用途**:營收成長特徵(YoY/MoM/動能)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 資料月份(月初) |
| `stock_id` | character varying(100) | 證券代號 |
| `country` | character varying(100) | 地區(Taiwan) |
| `revenue` | numeric(22,6) | 當月營收(元) |
| `revenue_month` | numeric(20,6) | 營收所屬月份(1-12) |
| `revenue_year` | numeric(20,6) | 營收所屬年度 |
| `create_time` | date | 資料公佈日(§14.7-BA publication-date) |

#### 11. `TaiwanStockDividend` — 股利政策
- **來源**:FinMind TaiwanStockDividend / **頻率**:年(annual,申報) / **PK**:['stock_id', 'date'] / **用途**:股利政策特徵(配息率/配股)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 申報基準日 |
| `stock_id` | character varying(100) | 證券代號 |
| `year` | character varying(100) | 股利所屬年度 |
| `StockEarningsDistribution` | numeric(20,6) | 盈餘配股(元/股) |
| `StockStatutorySurplus` | numeric(20,6) | 法定盈餘公積配股(元/股) |
| `StockExDividendTradingDate` | character varying(100) | 除權交易日 |
| `TotalEmployeeStockDividend` | numeric(20,6) | 員工配股總數 |
| `TotalEmployeeStockDividendAmount` | numeric(20,6) | 員工配股總金額 |
| `RatioOfEmployeeStockDividendOfTotal` | numeric(20,6) | 員工配股占總額比率 |
| `RatioOfEmployeeStockDividend` | numeric(20,6) | 員工配股比率 |
| `CashEarningsDistribution` | numeric(20,8) | 盈餘配息(元/股) |
| `CashStatutorySurplus` | numeric(20,6) | 法定盈餘公積配息(元/股) |
| `CashExDividendTradingDate` | date | 除息交易日 |
| `CashDividendPaymentDate` | date | 現金股利發放日 |
| `TotalEmployeeCashDividend` | numeric(20,6) | 員工現金紅利總額 |
| `TotalNumberOfCashCapitalIncrease` | numeric(20,6) | 現金增資總股數 |
| `CashIncreaseSubscriptionRate` | numeric(20,6) | 現金增資認購比率 |
| `CashIncreaseSubscriptionpRrice` | numeric(20,6) | 現金增資認購價 |
| `RemunerationOfDirectorsAndSupervisors` | numeric(20,6) | 董監酬勞 |
| `ParticipateDistributionOfTotalShares` | numeric(21,6) | 參與分配總股數 |
| `AnnouncementDate` | date | 公告日(§8.5 publication-date strict) |
| `AnnouncementTime` | character varying(100) | 公告時間 |

#### 12. `FredData` — FRED 美國/全球宏觀指標(multi-series 單表)
- **來源**:FRED API api.stlouisfed.org / **頻率**:隨 series(日/月/季/年) / **PK**:['date', 'series_id'] / **用途**:宏觀/regime 特徵(殖利率曲線/波動/通膨/景氣;§0.3 K-wave 5 驅動 proxy)。⚠️ 建表路徑=explicit-DDL(非 generic;series_id 為 local-derived 不在 API 回應,generic 無法推導 key)

| 欄位 | 型別(DB-verified) | 定義 |
|---|---|---|
| `date` | date | 觀測日 |
| `series_id` | character varying(255) | FRED 指標代號(見枚舉) |
| `value` | numeric(20,6) | 觀測值 |
| `realtime_start` | date | ALFRED vintage 起(資料版本可得起日) |
| `realtime_end` | date | ALFRED vintage 迄 |

**`series_id` 指標枚舉(`fetch_fred_data.DEFAULT_FRED_SERIES` 24 series;K-wave §0.3 5 大驅動 proxy)**:

| series_id | 定義 |
|---|---|
| `T10Y2Y` | 10年−2年期美國公債殖利率差(殖利率曲線斜率;倒掛為衰退前兆) |
| `T10Y3M` | 10年−3月期美國公債殖利率差 |
| `T10YIE` | 10年期損益兩平通膨率(市場隱含通膨預期) |
| `VIXCLS` | CBOE VIX 波動率指數(恐慌指數) |
| `BAMLH0A0HYM2` | ICE BofA 美國高收益債信用利差 OAS |
| `DTWEXBGS` | 美元廣義貿易加權匯率指數 |
| `M2SL` | M2 貨幣供給(季調) |
| `DGS10` | 10年期美國公債殖利率 |
| `DGS2` | 2年期美國公債殖利率 |
| `DGS3MO` | 3月期美國公債殖利率 |
| `UMCSENT` | 密西根大學消費者信心指數 |
| `INDPRO` | 美國工業生產指數 |
| `UNRATE` | 美國失業率(%) |
| `CPIAUCSL` | 美國消費者物價指數 CPI(季調) |
| `PATENTUSALLTOTAL` | 美國核准專利總數(Schumpeter 創新/K-wave 科技軸,年) |
| `B985RC1Q027SBEA` | 美國民間智財產品投資(R&D+軟體+娛樂,季) |
| `TCMDO` | 美國信用市場未償債務總額(季) |
| `LFWA64TTUSA647N` | 美國工作年齡人口 15-64(年) |
| `SPPOPDPNDOLUSA` | 美國老年扶養比(年) |
| `PALLFNFINDEXQ` | IMF 全球大宗商品價格指數(季) |
| `QUSPAM770A` | BIS 美國民間非金融部門信用占GDP%(信用缺口 proxy,季) |
| `WTISPLC` | WTI 原油現貨價(月) |
| `IPG3344S` | 美國半導體工業生產指數(Kitchin 半導體循環,月,2017=100) |
| `PCU4831114831115` | 美國遠洋貨運 PPI(Juglar 航運循環,月,Jun1988=100) |

