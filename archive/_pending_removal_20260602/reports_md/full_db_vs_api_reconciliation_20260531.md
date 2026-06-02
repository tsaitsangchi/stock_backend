# 全股 × 全史 × 全表 DB↔API 完整對帳報告

- 產生時間:2026-05-31 20:45:26
- 工具:`audit_full_db_vs_api_reconcile.py` v0.1(§14.7-CE family)
- Scope:`all`(對帳 2770 股)
- 歷史區間:1990-01-01 ~ 2026-05-31
- FinMind 表:TaiwanStockPrice, TaiwanStockPriceAdj, TaiwanStockPER, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockMarginPurchaseShortSale, TaiwanStockShareholding, TaiwanStockFinancialStatements, TaiwanStockBalanceSheet, TaiwanStockMonthRevenue, TaiwanStockDividend
- 實際 API calls:27743 / 耗時:18247.8s
- 數值容差:abs(a-b) ≤ max(1e-4, 1e-6·max(|a|,|b|));日期 ISO 比對;字串 strip exact
- Skip(ingest-metadata 透明揭露):{"TaiwanStockMonthRevenue": ["create_time"], "TaiwanStockInfo": ["date"]};FRED 僅比對 value(realtime_* vintage 不比)

## 裁決 (Verdict)

⚠️ **91950 筆 value_mismatch** — 偵測到 API ≠ DB 之 row,須 root-cause(可能 publication-date 調整 / 待 resync / 計算欄)。

> `missing_in_db`(API 有 DB 無)/ `extra_in_db`(DB 有 API 無)為覆蓋率差異:
> 區間邊緣少量屬正常;大量則代表 DB 落後 → 須 `sovereign_sync_engine` resync(本程式不自動修)。

## FinMind per-stock 表(全史逐筆)

| 表 | 已對帳股數 | API rows | DB rows | matched | value_mismatch | missing_in_db | extra_in_db | api_fail |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TaiwanStockPrice | 2770 | 10523364 | 10515667 | 10515667 | 0 | 7697 | 0 | 0 |
| TaiwanStockPriceAdj | 2770 | 10521318 | 10507832 | 10416981 | 90851 | 13486 | 0 | 0 |
| TaiwanStockPER | 2770 | 7356670 | 7352641 | 7352641 | 0 | 4029 | 0 | 0 |
| TaiwanStockInstitutionalInvestorsBuySell | 2770 | 25072814 | 25043013 | 25042965 | 48 | 29801 | 0 | 0 |
| TaiwanStockMarginPurchaseShortSale | 2770 | 7728152 | 7722170 | 7722170 | 0 | 5982 | 0 | 0 |
| TaiwanStockShareholding | 2770 | 8384522 | 8372857 | 8372857 | 0 | 11665 | 0 | 0 |
| TaiwanStockFinancialStatements | 2770 | 2663321 | 2663205 | 2663187 | 18 | 116 | 0 | 0 |
| TaiwanStockBalanceSheet | 2770 | 8248590 | 8248086 | 8247204 | 876 | 510 | 6 | 0 |
| TaiwanStockMonthRevenue | 2770 | 460187 | 460187 | 460186 | 1 | 0 | 0 | 0 |
| TaiwanStockDividend | 2770 | 29204 | 29312 | 29203 | 1 | 0 | 108 | 0 |

## TaiwanStockInfo(市場註冊;單次全股 call)

- API rows(scope 內):3742 / DB rows:2770
- matched=2625 / value_mismatch=145 / missing_in_db=0 / extra_in_db=0

## FRED 宏觀(全史 value-by-date)

- series 總數:28 / 100% match:26 / fail:0
- matched=119501 / value_mismatch=10 / missing_in_db=21611 / extra_in_db=1
  - `FredData`:4 series / matched=48895 / value_mismatch=0 / missing=0 / extra=0 / fail=0
  - `fred_series`:24 series / matched=70606 / value_mismatch=10 / missing=21611 / extra=1 / fail=0

## Mismatch 樣本(每類上限)

### TaiwanStockPrice
- `0053` key=['2026-05-25', '0053'] [missing_in_db]
- `0053` key=['2026-05-26', '0053'] [missing_in_db]
- `0053` key=['2026-05-27', '0053'] [missing_in_db]
- `0053` key=['2026-05-28', '0053'] [missing_in_db]
- `0053` key=['2026-05-29', '0053'] [missing_in_db]
- `00401A` key=['2026-05-25', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-26', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-27', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-28', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-29', '00401A'] [missing_in_db]
- `00400A` key=['2026-05-25', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-26', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-27', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-28', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-29', '00400A'] [missing_in_db]
- `0052` key=['2026-05-25', '0052'] [missing_in_db]
- `0052` key=['2026-05-26', '0052'] [missing_in_db]
- `0052` key=['2026-05-27', '0052'] [missing_in_db]
- `0052` key=['2026-05-28', '0052'] [missing_in_db]
- `0052` key=['2026-05-29', '0052'] [missing_in_db]

### TaiwanStockPriceAdj
- `0053` key=['2026-05-25', '0053'] [missing_in_db]
- `0053` key=['2026-05-26', '0053'] [missing_in_db]
- `0053` key=['2026-05-27', '0053'] [missing_in_db]
- `0053` key=['2026-05-28', '0053'] [missing_in_db]
- `0053` key=['2026-05-29', '0053'] [missing_in_db]
- `00401A` key=['2026-05-25', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-26', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-27', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-28', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-29', '00401A'] [missing_in_db]
- `00400A` key=['2026-05-25', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-26', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-27', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-28', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-29', '00400A'] [missing_in_db]
- `0052` key=['2026-05-25', '0052'] [missing_in_db]
- `0052` key=['2026-05-26', '0052'] [missing_in_db]
- `0052` key=['2026-05-27', '0052'] [missing_in_db]
- `0052` key=['2026-05-28', '0052'] [missing_in_db]
- `0052` key=['2026-05-29', '0052'] [missing_in_db]

### TaiwanStockPER
- `1101` key=['2026-05-28', '1101'] [missing_in_db]
- `1101` key=['2026-05-29', '1101'] [missing_in_db]
- `1102` key=['2026-05-28', '1102'] [missing_in_db]
- `1102` key=['2026-05-29', '1102'] [missing_in_db]
- `1103` key=['2026-05-28', '1103'] [missing_in_db]
- `1103` key=['2026-05-29', '1103'] [missing_in_db]
- `1104` key=['2026-05-28', '1104'] [missing_in_db]
- `1104` key=['2026-05-29', '1104'] [missing_in_db]
- `1108` key=['2026-05-28', '1108'] [missing_in_db]
- `1108` key=['2026-05-29', '1108'] [missing_in_db]
- `1109` key=['2026-05-28', '1109'] [missing_in_db]
- `1109` key=['2026-05-29', '1109'] [missing_in_db]
- `1110` key=['2026-05-28', '1110'] [missing_in_db]
- `1110` key=['2026-05-29', '1110'] [missing_in_db]
- `1201` key=['2026-05-28', '1201'] [missing_in_db]
- `1201` key=['2026-05-29', '1201'] [missing_in_db]
- `1203` key=['2026-05-28', '1203'] [missing_in_db]
- `1203` key=['2026-05-29', '1203'] [missing_in_db]
- `1210` key=['2026-05-28', '1210'] [missing_in_db]
- `1210` key=['2026-05-29', '1210'] [missing_in_db]

### TaiwanStockInstitutionalInvestorsBuySell
- `0053` key=['2026-05-25', '0053', 'Foreign_Investor'] [missing_in_db]
- `0053` key=['2026-05-25', '0053', 'Foreign_Dealer_Self'] [missing_in_db]
- `0053` key=['2026-05-25', '0053', 'Investment_Trust'] [missing_in_db]
- `0053` key=['2026-05-25', '0053', 'Dealer_self'] [missing_in_db]
- `0053` key=['2026-05-25', '0053', 'Dealer_Hedging'] [missing_in_db]
- `0053` key=['2026-05-26', '0053', 'Foreign_Investor'] [missing_in_db]
- `0053` key=['2026-05-26', '0053', 'Foreign_Dealer_Self'] [missing_in_db]
- `0053` key=['2026-05-26', '0053', 'Investment_Trust'] [missing_in_db]
- `0053` key=['2026-05-26', '0053', 'Dealer_self'] [missing_in_db]
- `0053` key=['2026-05-26', '0053', 'Dealer_Hedging'] [missing_in_db]
- `0053` key=['2026-05-27', '0053', 'Foreign_Investor'] [missing_in_db]
- `0053` key=['2026-05-27', '0053', 'Foreign_Dealer_Self'] [missing_in_db]
- `0053` key=['2026-05-27', '0053', 'Investment_Trust'] [missing_in_db]
- `0053` key=['2026-05-27', '0053', 'Dealer_self'] [missing_in_db]
- `0053` key=['2026-05-27', '0053', 'Dealer_Hedging'] [missing_in_db]
- `0053` key=['2026-05-28', '0053', 'Foreign_Investor'] [missing_in_db]
- `0053` key=['2026-05-28', '0053', 'Foreign_Dealer_Self'] [missing_in_db]
- `0053` key=['2026-05-28', '0053', 'Investment_Trust'] [missing_in_db]
- `0053` key=['2026-05-28', '0053', 'Dealer_self'] [missing_in_db]
- `0053` key=['2026-05-28', '0053', 'Dealer_Hedging'] [missing_in_db]

### TaiwanStockMarginPurchaseShortSale
- `0053` key=['2026-05-25', '0053'] [missing_in_db]
- `0053` key=['2026-05-26', '0053'] [missing_in_db]
- `0053` key=['2026-05-27', '0053'] [missing_in_db]
- `0053` key=['2026-05-28', '0053'] [missing_in_db]
- `0053` key=['2026-05-29', '0053'] [missing_in_db]
- `00401A` key=['2026-05-25', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-26', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-27', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-28', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-29', '00401A'] [missing_in_db]
- `00400A` key=['2026-05-25', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-26', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-27', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-28', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-29', '00400A'] [missing_in_db]
- `0052` key=['2026-05-25', '0052'] [missing_in_db]
- `0052` key=['2026-05-26', '0052'] [missing_in_db]
- `0052` key=['2026-05-27', '0052'] [missing_in_db]
- `0052` key=['2026-05-28', '0052'] [missing_in_db]
- `0052` key=['2026-05-29', '0052'] [missing_in_db]

### TaiwanStockShareholding
- `0053` key=['2026-05-25', '0053'] [missing_in_db]
- `0053` key=['2026-05-26', '0053'] [missing_in_db]
- `0053` key=['2026-05-27', '0053'] [missing_in_db]
- `0053` key=['2026-05-28', '0053'] [missing_in_db]
- `0053` key=['2026-05-29', '0053'] [missing_in_db]
- `00401A` key=['2026-05-25', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-26', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-27', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-28', '00401A'] [missing_in_db]
- `00401A` key=['2026-05-29', '00401A'] [missing_in_db]
- `00400A` key=['2026-05-25', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-26', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-27', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-28', '00400A'] [missing_in_db]
- `00400A` key=['2026-05-29', '00400A'] [missing_in_db]
- `0052` key=['2026-05-25', '0052'] [missing_in_db]
- `0052` key=['2026-05-26', '0052'] [missing_in_db]
- `0052` key=['2026-05-27', '0052'] [missing_in_db]
- `0052` key=['2026-05-28', '0052'] [missing_in_db]
- `0052` key=['2026-05-29', '0052'] [missing_in_db]

### TaiwanStockFinancialStatements
- `1526` key=['2025-12-31', '1526', 'EPS', '基本每股盈餘'] diffs=[('value', '-0.07', '0.010000')]
- `1786` key=['2025-12-31', '1786', 'EPS', '基本每股盈餘'] diffs=[('value', '1.67', '0.860000')]
- `2014` key=['2025-12-31', '2014', 'EPS', '基本每股盈餘'] diffs=[('value', '-0.19', '-0.560000')]
- `2062` key=['2025-12-31', '2062', 'EPS', '基本每股盈餘'] diffs=[('value', '0.2', '0.070000')]
- `2736` key=['2025-12-31', '2736', 'EPS', '基本每股盈餘'] diffs=[('value', '-0.16', '-1.020000')]
- `2880` key=['2026-03-31', '2880', '保險其他營業成本', '保險其他營業成本'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'NetNonInterestIncome', '利息以外淨收益'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'NetInterestIncome', '利息淨收益'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'BadDebts', '呆帳費用、承諾及保證責任準備提存'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'TAX', '所得稅（費用）利益'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'OtherComprehensiveIncomeAfterTaxThePeriod', '本期其他綜合損益（稅後淨額）'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'IncomeAfterTax', '本期稅後淨利（淨損）'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'TotalConsolidatedProfitForThePeriod', '本期綜合損益總額'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'EquityAttributableToOwnersOfParent', '淨利（淨損）歸屬於母公司業主'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'NoncontrollingInterests', '淨利（淨損）歸屬於非控制權益'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'Revenue', '淨收益'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'OperatingExpenses', '營業費用'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'ComprehensiveIncomeConsolidatedNetIncomeAttributedNonControllingInterest', '綜合損益總額歸屬於非控制權益'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'IncomeFromContinuingOperations', '繼續營業單位本期淨利（淨損）'] [missing_in_db]
- `2880` key=['2026-03-31', '2880', 'PreTaxIncome', '繼續營業單位稅前損益'] [missing_in_db]

### TaiwanStockBalanceSheet
- `1110` key=['2025-12-31', '1110', 'BillsReceivableNet', '應收票據淨額'] diffs=[('value', '291557000.0', '275321000.000000')]
- `1110` key=['2025-12-31', '1110', 'BillsReceivableNet_per', '應收票據淨額'] diffs=[('value', '2.23', '2.110000')]
- `1110` key=['2025-12-31', '1110', 'AccountsReceivableNet', '應收帳款淨額'] diffs=[('value', '610523000.0', '573845000.000000')]
- `1110` key=['2025-12-31', '1110', 'AccountsReceivableNet_per', '應收帳款淨額'] diffs=[('value', '4.67', '4.390000')]
- `1336` key=['2025-12-31', '1336', 'Prepayments', '預付款項'] diffs=[('value', '80468000.0', '92020000.000000')]
- `1336` key=['2025-12-31', '1336', 'Prepayments_per', '預付款項'] diffs=[('value', '2.56', '2.930000')]
- `1336` key=['2025-12-31', '1336', 'CurrentAssets', '流動資產合計'] diffs=[('value', '1198003000.0', '1209555000.000000')]
- `1336` key=['2025-12-31', '1336', 'CurrentAssets_per', '流動資產合計'] diffs=[('value', '38.12', '38.490000')]
- `1336` key=['2025-12-31', '1336', 'OtherNoncurrentAssets', '其他非流動資產'] diffs=[('value', '31116000.0', '19564000.000000')]
- `1336` key=['2025-12-31', '1336', 'OtherNoncurrentAssets_per', '其他非流動資產'] diffs=[('value', '0.99', '0.620000')]
- `1336` key=['2025-12-31', '1336', 'NoncurrentAssets', '非流動資產合計'] diffs=[('value', '1944370000.0', '1932818000.000000')]
- `1336` key=['2025-12-31', '1336', 'NoncurrentAssets_per', '非流動資產合計'] diffs=[('value', '61.88', '61.510000')]
- `1464` key=['2025-12-31', '1464', 'CashAndCashEquivalents_per', '現金及約當現金'] diffs=[('value', '8.99', '8.870000')]
- `1464` key=['2025-12-31', '1464', 'CurrentFinancialAssetsAtFairvalueThroughProfitOrLoss_per', '透過損益按公允價值衡量之金融資產－流動'] diffs=[('value', '0.38', '0.370000')]
- `1464` key=['2025-12-31', '1464', 'FinancialAssetsAtFairvalueThroughOtherComprehensiveIncome_per', '透過其他綜合損益按公允價值衡量之金融資產－流動'] diffs=[('value', '0.38', '0.370000')]
- `1464` key=['2025-12-31', '1464', 'FinancialAssetsAtAmortizedCost_per', '按攤銷後成本衡量之金融資產－流動'] diffs=[('value', '1.96', '1.930000')]
- `1464` key=['2025-12-31', '1464', 'FinancialAssetsAtAmortizedCostNonCurrent_per', '按攤銷後成本衡量之金融資產－非流動'] diffs=[('value', '1.33', '1.310000')]
- `1464` key=['2025-12-31', '1464', 'Prepayments_per', '預付款項'] diffs=[('value', '1.21', '1.190000')]
- `1464` key=['2025-12-31', '1464', 'AccountsReceivableNet_per', '應收帳款淨額'] diffs=[('value', '10.32', '10.170000')]
- `1464` key=['2025-12-31', '1464', 'Inventories_per', '存貨'] diffs=[('value', '23.19', '22.870000')]

### TaiwanStockMonthRevenue
- `4530` key=['2026-04-01', '4530'] diffs=[('revenue', '21585000', '44242000.000000')]

### TaiwanStockDividend
- `00850` key=['2023-11-20', '00850', '112'] diffs=[('CashDividendPaymentDate', '1023-12-12', '')]
- `00930` key=['2026-06-01', '00930', '115'] [extra_in_db]
- `00939` key=['2026-06-08', '00939', '115'] [extra_in_db]
- `00946` key=['2026-06-08', '00946', '115'] [extra_in_db]
- `00953B` key=['2026-06-08', '00953B', '115'] [extra_in_db]
- `00984D` key=['2026-06-08', '00984D', '115'] [extra_in_db]
- `00985B` key=['2026-06-08', '00985B', '115'] [extra_in_db]
- `1215` key=['2026-06-02', '1215', '114年'] [extra_in_db]
- `1219` key=['2026-06-14', '1219', '114年'] [extra_in_db]
- `1233` key=['2026-07-27', '1233', '114年'] [extra_in_db]
- `1234` key=['2026-07-05', '1234', '114年'] [extra_in_db]
- `1319` key=['2026-06-28', '1319', '114年'] [extra_in_db]
- `1323` key=['2026-06-08', '1323', '114年'] [extra_in_db]
- `1459` key=['2026-06-03', '1459', '114年'] [extra_in_db]
- `1515` key=['2026-06-16', '1515', '114年'] [extra_in_db]
- `1568` key=['2026-06-02', '1568', '114年'] [extra_in_db]
- `1590` key=['2026-07-29', '1590', '114年'] [extra_in_db]
- `1702` key=['2026-06-03', '1702', '114年'] [extra_in_db]
- `1708` key=['2026-06-05', '1708', '114年'] [extra_in_db]
- `1733` key=['2026-06-15', '1733', '114年'] [extra_in_db]

### TaiwanStockInfo
- `(info)` key=['3450'] [value_mismatch] diffs=[('industry_category', '電子工業', '半導體業')]
- `(info)` key=['6431'] [value_mismatch] diffs=[('industry_category', '化學生技醫療', '生技醫療業')]
- `(info)` key=['6924'] [value_mismatch] diffs=[('industry_category', '電子工業', '創新板股票')]
- `(info)` key=['6951'] [value_mismatch] diffs=[('industry_category', '創新板股票', '綠能環保')]
- `(info)` key=['6949'] [value_mismatch] diffs=[('industry_category', '創新板股票', '生技醫療業')]
- `(info)` key=['8487'] [value_mismatch] diffs=[('industry_category', '數位雲端', '創新板股票')]
- `(info)` key=['6936'] [value_mismatch] diffs=[('industry_category', '化學生技醫療', '生技醫療業')]
- `(info)` key=['6422'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '電子工業')]
- `(info)` key=['2456'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '電子工業')]
- `(info)` key=['6251'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '電子工業')]
- `(info)` key=['6983'] [value_mismatch] diffs=[('industry_category', '其他電子類', '其他電子業'), ('type', 'tpex', 'emerging')]
- `(info)` key=['7907'] [value_mismatch] diffs=[('industry_category', '其他電子業', '其他')]
- `(info)` key=['2601'] [value_mismatch] diffs=[('industry_category', '航運業', '貿易百貨')]
- `(info)` key=['6285'] [value_mismatch] diffs=[('industry_category', '通信網路業', '電子工業')]
- `(info)` key=['6271'] [value_mismatch] diffs=[('industry_category', '半導體業', '電子工業')]
- `(info)` key=['6552'] [value_mismatch] diffs=[('industry_category', '半導體業', '電子工業')]
- `(info)` key=['6269'] [value_mismatch] diffs=[('industry_category', '電子工業', '電子零組件業')]
- `(info)` key=['6550'] [value_mismatch] diffs=[('industry_category', '化學生技醫療', '生技醫療業')]
- `(info)` key=['6416'] [value_mismatch] diffs=[('industry_category', '通信網路業', '電子工業')]
- `(info)` key=['6415'] [value_mismatch] diffs=[('industry_category', '半導體業', '電子工業')]

### FRED
- `fred_series` B985RC1Q027SBEA 2026-01-01: API=795.681 / DB=797.188000
- `fred_series` M2SL 2023-05-01: API=20829.7 / DB=20829.600000
- `fred_series` M2SL 2024-05-01: API=21020.7 / DB=21020.600000
- `fred_series` M2SL 2025-03-01: API=21693.7 / DB=21693.600000
- `fred_series` M2SL 2025-04-01: API=21775.7 / DB=21775.600000
- `fred_series` M2SL 2025-05-01: API=21834.0 / DB=21833.900000
- `fred_series` M2SL 2025-08-01: API=22086.9 / DB=22087.000000
- `fred_series` M2SL 2025-11-01: API=22277.4 / DB=22277.500000
- `fred_series` M2SL 2025-12-01: API=22353.5 / DB=22353.600000
- `fred_series` M2SL 2026-02-01: API=22627.0 / DB=22627.300000
