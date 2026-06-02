# 全股 × 全史 × 全表 DB↔API 完整對帳報告

- 產生時間:2026-06-02 04:04:29
- 工具:`audit_full_db_vs_api_reconcile.py` v0.1(§14.7-CE family)
- Scope:`all`(對帳 2772 股)
- 歷史區間:1990-01-01 ~ 2026-06-01
- FinMind 表:TaiwanStockPrice, TaiwanStockPriceAdj, TaiwanStockPER, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockMarginPurchaseShortSale, TaiwanStockShareholding, TaiwanStockFinancialStatements, TaiwanStockBalanceSheet, TaiwanStockMonthRevenue, TaiwanStockDividend
- 實際 API calls:27760 / 耗時:18232.1s
- 數值容差:abs(a-b) ≤ max(1e-4, 1e-6·max(|a|,|b|));日期 ISO 比對;字串 strip exact
- Skip(ingest-metadata 透明揭露):{"TaiwanStockMonthRevenue": ["create_time"], "TaiwanStockInfo": ["date"]};FRED 僅比對 value(realtime_* vintage 不比)

## 裁決 (Verdict)

⚠️ **30647 筆 value_mismatch** — 偵測到 API ≠ DB 之 row,須 root-cause(可能 publication-date 調整 / 待 resync / 計算欄)。

> `missing_in_db`(API 有 DB 無)/ `extra_in_db`(DB 有 API 無)為覆蓋率差異:
> 區間邊緣少量屬正常;大量則代表 DB 落後 → 須 `sovereign_sync_engine` resync(本程式不自動修)。

## FinMind per-stock 表(全史逐筆)

| 表 | 已對帳股數 | API rows | DB rows | matched | value_mismatch | missing_in_db | extra_in_db | api_fail |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TaiwanStockPrice | 2772 | 10522682 | 10519982 | 10519982 | 0 | 2700 | 0 | 0 |
| TaiwanStockPriceAdj | 2772 | 10520634 | 10517934 | 10488886 | 29048 | 2700 | 0 | 0 |
| TaiwanStockPER | 2772 | 7355630 | 7353665 | 7353665 | 0 | 1965 | 0 | 0 |
| TaiwanStockInstitutionalInvestorsBuySell | 2772 | 25079889 | 25068181 | 25068181 | 0 | 11708 | 0 | 0 |
| TaiwanStockMarginPurchaseShortSale | 2772 | 7726162 | 7723999 | 7723999 | 0 | 2163 | 0 | 0 |
| TaiwanStockShareholding | 2772 | 8383429 | 8381096 | 8381096 | 0 | 2333 | 0 | 0 |
| TaiwanStockFinancialStatements | 2772 | 2663391 | 2663350 | 2663343 | 7 | 41 | 0 | 0 |
| TaiwanStockBalanceSheet | 2772 | 8249043 | 8248878 | 8247723 | 1141 | 179 | 14 | 0 |
| TaiwanStockMonthRevenue | 2772 | 460065 | 460055 | 460055 | 0 | 10 | 0 | 0 |
| TaiwanStockDividend | 2772 | 29205 | 29422 | 29204 | 1 | 0 | 217 | 0 |

## TaiwanStockInfo(市場註冊;單次全股 call)

- API rows(scope 內):588 / DB rows:2772
- matched=72 / value_mismatch=450 / missing_in_db=0 / extra_in_db=2250

## FRED 宏觀(全史 value-by-date)

- series 總數:28 / 100% match:28 / fail:0
- matched=119533 / value_mismatch=0 / missing_in_db=21590 / extra_in_db=3
  - `FredData`:4 series / matched=48895 / value_mismatch=0 / missing=1 / extra=0 / fail=0
  - `fred_series`:24 series / matched=70638 / value_mismatch=0 / missing=21589 / extra=3 / fail=0

## Mismatch 樣本(每類上限)

### TaiwanStockPrice
- `0050` key=['2026-06-01', '0050'] [missing_in_db]
- `0053` key=['2026-06-01', '0053'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A'] [missing_in_db]
- `0051` key=['2026-06-01', '0051'] [missing_in_db]
- `00401A` key=['2026-06-01', '00401A'] [missing_in_db]
- `00403A` key=['2026-06-01', '00403A'] [missing_in_db]
- `0052` key=['2026-06-01', '0052'] [missing_in_db]
- `0056` key=['2026-06-01', '0056'] [missing_in_db]
- `0055` key=['2026-06-01', '0055'] [missing_in_db]
- `0061` key=['2026-06-01', '0061'] [missing_in_db]
- `0057` key=['2026-06-01', '0057'] [missing_in_db]
- `006201` key=['2026-06-01', '006201'] [missing_in_db]
- `006203` key=['2026-06-01', '006203'] [missing_in_db]
- `006205` key=['2026-06-01', '006205'] [missing_in_db]
- `006204` key=['2026-06-01', '006204'] [missing_in_db]
- `006206` key=['2026-06-01', '006206'] [missing_in_db]
- `006207` key=['2026-06-01', '006207'] [missing_in_db]
- `006208` key=['2026-06-01', '006208'] [missing_in_db]
- `00625K` key=['2026-06-01', '00625K'] [missing_in_db]
- `00631L` key=['2026-06-01', '00631L'] [missing_in_db]

### TaiwanStockPriceAdj
- `0050` key=['2026-06-01', '0050'] [missing_in_db]
- `0053` key=['2026-06-01', '0053'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A'] [missing_in_db]
- `0051` key=['2026-06-01', '0051'] [missing_in_db]
- `00401A` key=['2026-06-01', '00401A'] [missing_in_db]
- `00403A` key=['2026-06-01', '00403A'] [missing_in_db]
- `0052` key=['2026-06-01', '0052'] [missing_in_db]
- `0056` key=['2026-06-01', '0056'] [missing_in_db]
- `0055` key=['2026-06-01', '0055'] [missing_in_db]
- `0061` key=['2026-06-01', '0061'] [missing_in_db]
- `0057` key=['2026-06-01', '0057'] [missing_in_db]
- `006201` key=['2026-06-01', '006201'] [missing_in_db]
- `006203` key=['2026-06-01', '006203'] [missing_in_db]
- `006205` key=['2026-06-01', '006205'] [missing_in_db]
- `006204` key=['2026-06-01', '006204'] [missing_in_db]
- `006206` key=['2026-06-01', '006206'] [missing_in_db]
- `006207` key=['2026-06-01', '006207'] [missing_in_db]
- `006208` key=['2026-06-01', '006208'] [missing_in_db]
- `00625K` key=['2026-06-01', '00625K'] [missing_in_db]
- `00631L` key=['2026-06-01', '00631L'] [missing_in_db]

### TaiwanStockPER
- `1101` key=['2026-06-01', '1101'] [missing_in_db]
- `1102` key=['2026-06-01', '1102'] [missing_in_db]
- `1104` key=['2026-06-01', '1104'] [missing_in_db]
- `1103` key=['2026-06-01', '1103'] [missing_in_db]
- `1108` key=['2026-06-01', '1108'] [missing_in_db]
- `1109` key=['2026-06-01', '1109'] [missing_in_db]
- `1201` key=['2026-06-01', '1201'] [missing_in_db]
- `1110` key=['2026-06-01', '1110'] [missing_in_db]
- `1203` key=['2026-06-01', '1203'] [missing_in_db]
- `1210` key=['2026-06-01', '1210'] [missing_in_db]
- `1213` key=['2026-06-01', '1213'] [missing_in_db]
- `1215` key=['2026-06-01', '1215'] [missing_in_db]
- `1216` key=['2026-06-01', '1216'] [missing_in_db]
- `1217` key=['2026-06-01', '1217'] [missing_in_db]
- `1218` key=['2026-06-01', '1218'] [missing_in_db]
- `1219` key=['2026-06-01', '1219'] [missing_in_db]
- `1220` key=['2026-06-01', '1220'] [missing_in_db]
- `1225` key=['2026-06-01', '1225'] [missing_in_db]
- `1227` key=['2026-06-01', '1227'] [missing_in_db]
- `1229` key=['2026-06-01', '1229'] [missing_in_db]

### TaiwanStockInstitutionalInvestorsBuySell
- `0050` key=['2026-06-01', '0050', 'Foreign_Investor'] [missing_in_db]
- `0050` key=['2026-06-01', '0050', 'Foreign_Dealer_Self'] [missing_in_db]
- `0050` key=['2026-06-01', '0050', 'Investment_Trust'] [missing_in_db]
- `0050` key=['2026-06-01', '0050', 'Dealer_self'] [missing_in_db]
- `0050` key=['2026-06-01', '0050', 'Dealer_Hedging'] [missing_in_db]
- `0053` key=['2026-06-01', '0053', 'Foreign_Investor'] [missing_in_db]
- `0053` key=['2026-06-01', '0053', 'Foreign_Dealer_Self'] [missing_in_db]
- `0053` key=['2026-06-01', '0053', 'Investment_Trust'] [missing_in_db]
- `0053` key=['2026-06-01', '0053', 'Dealer_self'] [missing_in_db]
- `0053` key=['2026-06-01', '0053', 'Dealer_Hedging'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Foreign_Investor'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Foreign_Dealer_Self'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Investment_Trust'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Dealer_self'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Dealer_Hedging'] [missing_in_db]
- `0051` key=['2026-06-01', '0051', 'Foreign_Investor'] [missing_in_db]
- `0051` key=['2026-06-01', '0051', 'Foreign_Dealer_Self'] [missing_in_db]
- `0051` key=['2026-06-01', '0051', 'Investment_Trust'] [missing_in_db]
- `0051` key=['2026-06-01', '0051', 'Dealer_self'] [missing_in_db]
- `0051` key=['2026-06-01', '0051', 'Dealer_Hedging'] [missing_in_db]

### TaiwanStockMarginPurchaseShortSale
- `0050` key=['2026-06-01', '0050'] [missing_in_db]
- `0053` key=['2026-06-01', '0053'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A'] [missing_in_db]
- `0051` key=['2026-06-01', '0051'] [missing_in_db]
- `00401A` key=['2026-06-01', '00401A'] [missing_in_db]
- `00403A` key=['2026-06-01', '00403A'] [missing_in_db]
- `0052` key=['2026-06-01', '0052'] [missing_in_db]
- `0056` key=['2026-06-01', '0056'] [missing_in_db]
- `0055` key=['2026-06-01', '0055'] [missing_in_db]
- `0061` key=['2026-06-01', '0061'] [missing_in_db]
- `0057` key=['2026-06-01', '0057'] [missing_in_db]
- `006201` key=['2026-06-01', '006201'] [missing_in_db]
- `006203` key=['2026-06-01', '006203'] [missing_in_db]
- `006205` key=['2026-06-01', '006205'] [missing_in_db]
- `006204` key=['2026-06-01', '006204'] [missing_in_db]
- `006206` key=['2026-06-01', '006206'] [missing_in_db]
- `006207` key=['2026-06-01', '006207'] [missing_in_db]
- `006208` key=['2026-06-01', '006208'] [missing_in_db]
- `00631L` key=['2026-06-01', '00631L'] [missing_in_db]
- `00632R` key=['2026-06-01', '00632R'] [missing_in_db]

### TaiwanStockShareholding
- `0050` key=['2026-06-01', '0050'] [missing_in_db]
- `0053` key=['2026-06-01', '0053'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A'] [missing_in_db]
- `0051` key=['2026-06-01', '0051'] [missing_in_db]
- `00401A` key=['2026-06-01', '00401A'] [missing_in_db]
- `00403A` key=['2026-06-01', '00403A'] [missing_in_db]
- `0052` key=['2026-06-01', '0052'] [missing_in_db]
- `0056` key=['2026-06-01', '0056'] [missing_in_db]
- `0055` key=['2026-06-01', '0055'] [missing_in_db]
- `0061` key=['2026-06-01', '0061'] [missing_in_db]
- `0057` key=['2026-06-01', '0057'] [missing_in_db]
- `006201` key=['2026-06-01', '006201'] [missing_in_db]
- `006203` key=['2026-06-01', '006203'] [missing_in_db]
- `006205` key=['2026-06-01', '006205'] [missing_in_db]
- `006204` key=['2026-06-01', '006204'] [missing_in_db]
- `006206` key=['2026-06-01', '006206'] [missing_in_db]
- `006207` key=['2026-06-01', '006207'] [missing_in_db]
- `006208` key=['2026-06-01', '006208'] [missing_in_db]
- `00625K` key=['2026-06-01', '00625K'] [missing_in_db]
- `00631L` key=['2026-06-01', '00631L'] [missing_in_db]

### TaiwanStockFinancialStatements
- `2379` key=['2025-12-31', '2379', 'EPS', '基本每股盈餘'] diffs=[('value', '5.18', '11.860000')]
- `2433` key=['2025-12-31', '2433', 'EPS', '基本每股盈餘'] diffs=[('value', '2.54', '0.650000')]
- `2484` key=['2025-12-31', '2484', 'EPS', '基本每股盈餘'] diffs=[('value', '0.52', '0.390000')]
- `2882` key=['2026-03-31', '2882', '保險其他營業成本', '保險其他營業成本'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'NetNonInterestIncome', '利息以外淨收益'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'NetInterestIncome', '利息淨收益'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'BadDebts', '呆帳費用、承諾及保證責任準備提存'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'TAX', '所得稅（費用）利益'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'OtherComprehensiveIncomeAfterTaxThePeriod', '本期其他綜合損益（稅後淨額）'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'IncomeAfterTax', '本期稅後淨利（淨損）'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'TotalConsolidatedProfitForThePeriod', '本期綜合損益總額'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'EquityAttributableToOwnersOfParent', '淨利（淨損）歸屬於母公司業主'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'NoncontrollingInterests', '淨利（淨損）歸屬於非控制權益'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'Revenue', '淨收益'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'OperatingExpenses', '營業費用'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'ComprehensiveIncomeConsolidatedNetIncomeAttributedNonControllingInterest', '綜合損益總額歸屬於非控制權益'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'IncomeFromContinuingOperations', '繼續營業單位本期淨利（淨損）'] [missing_in_db]
- `2882` key=['2026-03-31', '2882', 'PreTaxIncome', '繼續營業單位稅前損益'] [missing_in_db]
- `2887` key=['2026-03-31', '2887', '保險其他營業成本', '保險其他營業成本'] [missing_in_db]
- `2887` key=['2026-03-31', '2887', 'NetNonInterestIncome', '利息以外淨收益'] [missing_in_db]

### TaiwanStockBalanceSheet
- `1336` key=['2025-12-31', '1336', 'Prepayments', '預付款項'] diffs=[('value', '92020000.0', '80468000.000000')]
- `1336` key=['2025-12-31', '1336', 'Prepayments_per', '預付款項'] diffs=[('value', '2.93', '2.560000')]
- `1336` key=['2025-12-31', '1336', 'CurrentAssets', '流動資產合計'] diffs=[('value', '1209555000.0', '1198003000.000000')]
- `1336` key=['2025-12-31', '1336', 'CurrentAssets_per', '流動資產合計'] diffs=[('value', '38.49', '38.120000')]
- `1336` key=['2025-12-31', '1336', 'OtherNoncurrentAssets', '其他非流動資產'] diffs=[('value', '19564000.0', '31116000.000000')]
- `1336` key=['2025-12-31', '1336', 'OtherNoncurrentAssets_per', '其他非流動資產'] diffs=[('value', '0.62', '0.990000')]
- `1336` key=['2025-12-31', '1336', 'NoncurrentAssets', '非流動資產合計'] diffs=[('value', '1932818000.0', '1944370000.000000')]
- `1336` key=['2025-12-31', '1336', 'NoncurrentAssets_per', '非流動資產合計'] diffs=[('value', '61.51', '61.880000')]
- `1533` key=['2025-12-31', '1533', 'CashAndCashEquivalents_per', '現金及約當現金'] diffs=[('value', '6.76', '6.770000')]
- `1533` key=['2025-12-31', '1533', 'BillsReceivableNet_per', '應收票據淨額'] diffs=[('value', '3.98', '3.990000')]
- `1533` key=['2025-12-31', '1533', 'AccountsReceivableNet_per', '應收帳款淨額'] diffs=[('value', '9.69', '9.700000')]
- `1533` key=['2025-12-31', '1533', 'Inventories_per', '存貨'] diffs=[('value', '16.3', '16.310000')]
- `1533` key=['2025-12-31', '1533', 'OtherCurrentAssets', '其他流動資產'] diffs=[('value', '185768000.0', '178471000.000000')]
- `1533` key=['2025-12-31', '1533', 'OtherCurrentAssets_per', '其他流動資產'] diffs=[('value', '2.02', '1.940000')]
- `1533` key=['2025-12-31', '1533', 'CurrentAssets', '流動資產合計'] diffs=[('value', '5684080000.0', '5676783000.000000')]
- `1533` key=['2025-12-31', '1533', 'CurrentAssets_per', '流動資產合計'] diffs=[('value', '61.74', '61.710000')]
- `1533` key=['2025-12-31', '1533', 'PropertyPlantAndEquipment_per', '不動產、廠房及設備'] diffs=[('value', '23.82', '23.840000')]
- `1533` key=['2025-12-31', '1533', 'NoncurrentAssets_per', '非流動資產合計'] diffs=[('value', '38.26', '38.290000')]
- `1533` key=['2025-12-31', '1533', 'TotalAssets', '資產總額'] diffs=[('value', '9206983000.0', '9199686000.000000')]
- `1533` key=['2025-12-31', '1533', 'ShorttermBorrowings_per', '短期借款'] diffs=[('value', '12.68', '12.690000')]

### TaiwanStockMonthRevenue
- `3228` key=['2026-06-01', '3228'] [missing_in_db]
- `5315` key=['2026-06-01', '5315'] [missing_in_db]
- `5508` key=['2026-06-01', '5508'] [missing_in_db]
- `6158` key=['2026-06-01', '6158'] [missing_in_db]
- `6438` key=['2026-06-01', '6438'] [missing_in_db]
- `6518` key=['2026-06-01', '6518'] [missing_in_db]
- `6712` key=['2026-06-01', '6712'] [missing_in_db]
- `6785` key=['2026-06-01', '6785'] [missing_in_db]
- `6910` key=['2026-06-01', '6910'] [missing_in_db]
- `7427` key=['2026-06-01', '7427'] [missing_in_db]

### TaiwanStockDividend
- `00850` key=['2023-11-20', '00850', '112'] diffs=[('CashDividendPaymentDate', '1023-12-12', '')]
- `00939` key=['2026-06-08', '00939', '115'] [extra_in_db]
- `00940` key=['2026-06-15', '00940', '115'] [extra_in_db]
- `00946` key=['2026-06-08', '00946', '115'] [extra_in_db]
- `00953B` key=['2026-06-08', '00953B', '115'] [extra_in_db]
- `00984D` key=['2026-06-08', '00984D', '115'] [extra_in_db]
- `00985B` key=['2026-06-08', '00985B', '115'] [extra_in_db]
- `1215` key=['2026-06-02', '1215', '114年'] [extra_in_db]
- `1218` key=['2026-06-20', '1218', '114年'] [extra_in_db]
- `1219` key=['2026-06-14', '1219', '114年'] [extra_in_db]
- `1232` key=['2026-06-21', '1232', '114年'] [extra_in_db]
- `1233` key=['2026-07-27', '1233', '114年'] [extra_in_db]
- `1234` key=['2026-07-05', '1234', '114年'] [extra_in_db]
- `1312` key=['2026-06-19', '1312', '不適用'] [extra_in_db]
- `1312A` key=['2026-06-19', '1312A', '不適用'] [extra_in_db]
- `1319` key=['2026-06-28', '1319', '114年'] [extra_in_db]
- `1323` key=['2026-06-08', '1323', '114年'] [extra_in_db]
- `1416` key=['2026-06-15', '1416', '114年'] [extra_in_db]
- `1459` key=['2026-06-03', '1459', '114年'] [extra_in_db]
- `1473` key=['2026-06-16', '1473', '114年'] [extra_in_db]

### TaiwanStockInfo
- `(info)` key=['5481'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '其他')]
- `(info)` key=['5450'] [value_mismatch] diffs=[('stock_name', '寶聯通', '南良'), ('industry_category', '電腦及週邊設備業', '其他')]
- `(info)` key=['3629'] [value_mismatch] diffs=[('industry_category', '光電業', '文化創意業')]
- `(info)` key=['3687'] [value_mismatch] diffs=[('industry_category', '電子商務業', '數位雲端類')]
- `(info)` key=['6438'] [value_mismatch] diffs=[('industry_category', '其他電子類', '電子工業'), ('type', 'tpex', 'twse')]
- `(info)` key=['6426'] [value_mismatch] diffs=[('industry_category', '通信網路業', '電子工業'), ('type', 'tpex', 'twse')]
- `(info)` key=['3092'] [value_mismatch] diffs=[('type', 'tpex', 'twse')]
- `(info)` key=['8499'] [value_mismatch] diffs=[('industry_category', '其他', '電子工業')]
- `(info)` key=['2241'] [value_mismatch] diffs=[('industry_category', '電機機械', '汽車工業')]
- `(info)` key=['1443'] [value_mismatch] diffs=[('stock_name', '立益', '立益物流'), ('industry_category', '紡織纖維', '其他')]
- `(info)` key=['6165'] [value_mismatch] diffs=[('industry_category', '其他', '數位雲端')]
- `(info)` key=['2614'] [value_mismatch] diffs=[('industry_category', '貿易百貨', '其他')]
- `(info)` key=['2459'] [value_mismatch] diffs=[('industry_category', '電子通路業', '電子工業')]
- `(info)` key=['3669'] [value_mismatch] diffs=[('industry_category', '光電業', '通信網路業')]
- `(info)` key=['3450'] [value_mismatch] diffs=[('industry_category', '其他電子業', '電子工業')]
- `(info)` key=['1456'] [value_mismatch] diffs=[('industry_category', '紡織纖維', '建材營造')]
- `(info)` key=['1453'] [value_mismatch] diffs=[('industry_category', '紡織纖維', '建材營造')]
- `(info)` key=['2429'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '電子工業')]
- `(info)` key=['8489'] [value_mismatch] diffs=[('industry_category', '文化創意業', '其他')]
- `(info)` key=['3313'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '其他')]
