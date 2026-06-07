# 全股 × 全史 × 全表 DB↔API 完整對帳報告

- 產生時間:2026-06-06 06:02:17
- 工具:`audit_full_db_vs_api_reconcile.py` v0.1(§14.7-CE family)
- Scope:`all`(對帳 2772 股)
- 歷史區間:1990-01-01 ~ 2026-06-05
- FinMind 表:TaiwanStockPrice, TaiwanStockPriceAdj, TaiwanStockPER, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockMarginPurchaseShortSale, TaiwanStockShareholding, TaiwanStockFinancialStatements, TaiwanStockBalanceSheet, TaiwanStockMonthRevenue, TaiwanStockDividend
- 實際 API calls:27750 / 耗時:25333.2s
- 數值容差:abs(a-b) ≤ max(1e-4, 1e-6·max(|a|,|b|));日期 ISO 比對;字串 strip exact
- Skip(ingest-metadata 透明揭露):{"TaiwanStockMonthRevenue": ["create_time"], "TaiwanStockInfo": ["date"]};FRED 僅比對 value(realtime_* vintage 不比)

## 裁決 (Verdict)

⚠️ **104087 筆 value_mismatch** — 偵測到 API ≠ DB 之 row,須 root-cause(可能 publication-date 調整 / 待 resync / 計算欄)。

> `missing_in_db`(API 有 DB 無)/ `extra_in_db`(DB 有 API 無)為覆蓋率差異:
> 區間邊緣少量屬正常;大量則代表 DB 落後 → 須 `sovereign_sync_engine` resync(本程式不自動修)。

## FinMind per-stock 表(全史逐筆)

| 表 | 已對帳股數 | API rows | DB rows | matched | value_mismatch | missing_in_db | extra_in_db | api_fail |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TaiwanStockPrice | 2772 | 10533478 | 10521810 | 10521810 | 0 | 11668 | 0 | 0 |
| TaiwanStockPriceAdj | 2772 | 10531430 | 10517934 | 10414968 | 102966 | 13496 | 0 | 0 |
| TaiwanStockPER | 2772 | 7363536 | 7355493 | 7355493 | 0 | 8043 | 0 | 0 |
| TaiwanStockInstitutionalInvestorsBuySell | 2772 | 25126461 | 25077296 | 25077222 | 74 | 49165 | 0 | 0 |
| TaiwanStockMarginPurchaseShortSale | 2772 | 7734822 | 7726104 | 7726104 | 0 | 8718 | 0 | 0 |
| TaiwanStockShareholding | 2772 | 8392763 | 8381096 | 8381096 | 0 | 11667 | 0 | 0 |
| TaiwanStockFinancialStatements | 2772 | 2663403 | 2663350 | 2663339 | 11 | 53 | 0 | 0 |
| TaiwanStockBalanceSheet | 2772 | 8249080 | 8248878 | 8248294 | 584 | 202 | 0 | 0 |
| TaiwanStockMonthRevenue | 2772 | 460525 | 460055 | 460055 | 0 | 470 | 0 | 0 |
| TaiwanStockDividend | 2772 | 29223 | 29421 | 29222 | 1 | 0 | 198 | 0 |

## TaiwanStockInfo(市場註冊;單次全股 call)

- API rows(scope 內):602 / DB rows:2772
- matched=87 / value_mismatch=448 / missing_in_db=0 / extra_in_db=2237

## FRED 宏觀(全史 value-by-date)

- series 總數:28 / 100% match:27 / fail:0
- matched=119542 / value_mismatch=3 / missing_in_db=21638 / extra_in_db=5
  - `FredData`:4 series / matched=48909 / value_mismatch=0 / missing=4 / extra=0 / fail=0
  - `fred_series`:24 series / matched=70633 / value_mismatch=3 / missing=21634 / extra=5 / fail=0

## Mismatch 樣本(每類上限)

### TaiwanStockPrice
- `00400A` key=['2026-06-01', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-05', '00400A'] [missing_in_db]
- `00403A` key=['2026-06-01', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-02', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-03', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-04', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-05', '00403A'] [missing_in_db]
- `00401A` key=['2026-06-01', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-02', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-03', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-04', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-05', '00401A'] [missing_in_db]
- `0050` key=['2026-06-01', '0050'] [missing_in_db]
- `0050` key=['2026-06-02', '0050'] [missing_in_db]
- `0050` key=['2026-06-03', '0050'] [missing_in_db]
- `0050` key=['2026-06-04', '0050'] [missing_in_db]
- `0050` key=['2026-06-05', '0050'] [missing_in_db]

### TaiwanStockPriceAdj
- `00400A` key=['2026-06-01', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-05', '00400A'] [missing_in_db]
- `00403A` key=['2026-06-01', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-02', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-03', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-04', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-05', '00403A'] [missing_in_db]
- `00401A` key=['2026-06-01', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-02', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-03', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-04', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-05', '00401A'] [missing_in_db]
- `0050` key=['2026-06-01', '0050'] [missing_in_db]
- `0050` key=['2026-06-02', '0050'] [missing_in_db]
- `0050` key=['2026-06-03', '0050'] [missing_in_db]
- `0050` key=['2026-06-04', '0050'] [missing_in_db]
- `0050` key=['2026-06-05', '0050'] [missing_in_db]

### TaiwanStockPER
- `1101` key=['2026-06-01', '1101'] [missing_in_db]
- `1101` key=['2026-06-02', '1101'] [missing_in_db]
- `1101` key=['2026-06-03', '1101'] [missing_in_db]
- `1101` key=['2026-06-04', '1101'] [missing_in_db]
- `1101` key=['2026-06-05', '1101'] [missing_in_db]
- `1102` key=['2026-06-03', '1102'] [missing_in_db]
- `1102` key=['2026-06-04', '1102'] [missing_in_db]
- `1102` key=['2026-06-05', '1102'] [missing_in_db]
- `1103` key=['2026-06-03', '1103'] [missing_in_db]
- `1103` key=['2026-06-04', '1103'] [missing_in_db]
- `1103` key=['2026-06-05', '1103'] [missing_in_db]
- `1104` key=['2026-06-03', '1104'] [missing_in_db]
- `1104` key=['2026-06-04', '1104'] [missing_in_db]
- `1104` key=['2026-06-05', '1104'] [missing_in_db]
- `1108` key=['2026-06-03', '1108'] [missing_in_db]
- `1108` key=['2026-06-04', '1108'] [missing_in_db]
- `1108` key=['2026-06-05', '1108'] [missing_in_db]
- `1109` key=['2026-06-03', '1109'] [missing_in_db]
- `1109` key=['2026-06-04', '1109'] [missing_in_db]
- `1109` key=['2026-06-05', '1109'] [missing_in_db]

### TaiwanStockInstitutionalInvestorsBuySell
- `00400A` key=['2026-06-01', '00400A', 'Foreign_Investor'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Foreign_Dealer_Self'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Investment_Trust'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Dealer_self'] [missing_in_db]
- `00400A` key=['2026-06-01', '00400A', 'Dealer_Hedging'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A', 'Foreign_Investor'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A', 'Foreign_Dealer_Self'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A', 'Investment_Trust'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A', 'Dealer_self'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A', 'Dealer_Hedging'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A', 'Foreign_Investor'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A', 'Foreign_Dealer_Self'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A', 'Investment_Trust'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A', 'Dealer_self'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A', 'Dealer_Hedging'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A', 'Foreign_Investor'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A', 'Foreign_Dealer_Self'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A', 'Investment_Trust'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A', 'Dealer_self'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A', 'Dealer_Hedging'] [missing_in_db]

### TaiwanStockMarginPurchaseShortSale
- `00400A` key=['2026-06-01', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-05', '00400A'] [missing_in_db]
- `00403A` key=['2026-06-01', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-02', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-03', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-04', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-05', '00403A'] [missing_in_db]
- `00401A` key=['2026-06-01', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-02', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-03', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-04', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-05', '00401A'] [missing_in_db]
- `0050` key=['2026-06-01', '0050'] [missing_in_db]
- `0050` key=['2026-06-02', '0050'] [missing_in_db]
- `0050` key=['2026-06-03', '0050'] [missing_in_db]
- `0050` key=['2026-06-04', '0050'] [missing_in_db]
- `0050` key=['2026-06-05', '0050'] [missing_in_db]

### TaiwanStockShareholding
- `00400A` key=['2026-06-01', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-02', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-03', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-04', '00400A'] [missing_in_db]
- `00400A` key=['2026-06-05', '00400A'] [missing_in_db]
- `00403A` key=['2026-06-01', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-02', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-03', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-04', '00403A'] [missing_in_db]
- `00403A` key=['2026-06-05', '00403A'] [missing_in_db]
- `00401A` key=['2026-06-01', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-02', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-03', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-04', '00401A'] [missing_in_db]
- `00401A` key=['2026-06-05', '00401A'] [missing_in_db]
- `0050` key=['2026-06-01', '0050'] [missing_in_db]
- `0050` key=['2026-06-02', '0050'] [missing_in_db]
- `0050` key=['2026-06-03', '0050'] [missing_in_db]
- `0050` key=['2026-06-04', '0050'] [missing_in_db]
- `0050` key=['2026-06-05', '0050'] [missing_in_db]

### TaiwanStockFinancialStatements
- `2453` key=['2025-12-31', '2453', 'EPS', '基本每股盈餘'] diffs=[('value', '1.64', '0.920000')]
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
- `2887` key=['2026-03-31', '2887', 'NetInterestIncome', '利息淨收益'] [missing_in_db]
- `2887` key=['2026-03-31', '2887', 'BadDebts', '呆帳費用、承諾及保證責任準備提存'] [missing_in_db]

### TaiwanStockBalanceSheet
- `1235` key=['2025-12-31', '1235', 'OtherPayables', '其他應付款'] diffs=[('value', '34784000.0', '34712000.000000')]
- `1235` key=['2025-12-31', '1235', 'OtherPayables_per', '其他應付款'] diffs=[('value', '0.63', '0.620000')]
- `1464` key=['2025-12-31', '1464', 'CashAndCashEquivalents_per', '現金及約當現金'] diffs=[('value', '8.99', '8.870000')]
- `1464` key=['2025-12-31', '1464', 'CurrentFinancialAssetsAtFairvalueThroughProfitOrLoss_per', '透過損益按公允價值衡量之金融資產－流動'] diffs=[('value', '0.38', '0.370000')]
- `1464` key=['2025-12-31', '1464', 'FinancialAssetsAtFairvalueThroughOtherComprehensiveIncome_per', '透過其他綜合損益按公允價值衡量之金融資產－流動'] diffs=[('value', '0.38', '0.370000')]
- `1464` key=['2025-12-31', '1464', 'FinancialAssetsAtAmortizedCost_per', '按攤銷後成本衡量之金融資產－流動'] diffs=[('value', '1.96', '1.930000')]
- `1464` key=['2025-12-31', '1464', 'FinancialAssetsAtAmortizedCostNonCurrent_per', '按攤銷後成本衡量之金融資產－非流動'] diffs=[('value', '1.33', '1.310000')]
- `1464` key=['2025-12-31', '1464', 'Prepayments_per', '預付款項'] diffs=[('value', '1.21', '1.190000')]
- `1464` key=['2025-12-31', '1464', 'AccountsReceivableNet_per', '應收帳款淨額'] diffs=[('value', '10.32', '10.170000')]
- `1464` key=['2025-12-31', '1464', 'Inventories_per', '存貨'] diffs=[('value', '23.19', '22.870000')]
- `1464` key=['2025-12-31', '1464', 'OtherCurrentAssets', '其他流動資產'] diffs=[('value', '463555000.0', '463573000.000000')]
- `1464` key=['2025-12-31', '1464', 'OtherCurrentAssets_per', '其他流動資產'] diffs=[('value', '2.73', '2.690000')]
- `1464` key=['2025-12-31', '1464', 'CurrentAssets', '流動資產合計'] diffs=[('value', '8473805000.0', '8473823000.000000')]
- `1464` key=['2025-12-31', '1464', 'CurrentAssets_per', '流動資產合計'] diffs=[('value', '49.88', '49.190000')]
- `1464` key=['2025-12-31', '1464', 'InvestmentAccountedForUsingEquityMethod', '採用權益法之投資'] diffs=[('value', '416113000.0', '413156000.000000')]
- `1464` key=['2025-12-31', '1464', 'InvestmentAccountedForUsingEquityMethod_per', '採用權益法之投資'] diffs=[('value', '2.45', '2.400000')]
- `1464` key=['2025-12-31', '1464', 'PropertyPlantAndEquipment', '不動產、廠房及設備'] diffs=[('value', '6507626000.0', '6967058000.000000')]
- `1464` key=['2025-12-31', '1464', 'PropertyPlantAndEquipment_per', '不動產、廠房及設備'] diffs=[('value', '38.31', '40.440000')]
- `1464` key=['2025-12-31', '1464', 'RightOfUseAsset', '使用權資產'] diffs=[('value', '551069000.0', '440220000.000000')]
- `1464` key=['2025-12-31', '1464', 'RightOfUseAsset_per', '使用權資產'] diffs=[('value', '3.24', '2.560000')]

### TaiwanStockMonthRevenue
- `1213` key=['2026-06-01', '1213'] [missing_in_db]
- `1256` key=['2026-06-01', '1256'] [missing_in_db]
- `1308` key=['2026-06-01', '1308'] [missing_in_db]
- `1342` key=['2026-06-01', '1342'] [missing_in_db]
- `1414` key=['2026-06-01', '1414'] [missing_in_db]
- `1435` key=['2026-06-01', '1435'] [missing_in_db]
- `1436` key=['2026-06-01', '1436'] [missing_in_db]
- `1437` key=['2026-06-01', '1437'] [missing_in_db]
- `1452` key=['2026-06-01', '1452'] [missing_in_db]
- `1454` key=['2026-06-01', '1454'] [missing_in_db]
- `1464` key=['2026-06-01', '1464'] [missing_in_db]
- `1476` key=['2026-06-01', '1476'] [missing_in_db]
- `1480` key=['2026-06-01', '1480'] [missing_in_db]
- `1477` key=['2026-06-01', '1477'] [missing_in_db]
- `1504` key=['2026-06-01', '1504'] [missing_in_db]
- `1535` key=['2026-06-01', '1535'] [missing_in_db]
- `1539` key=['2026-06-01', '1539'] [missing_in_db]
- `1560` key=['2026-06-01', '1560'] [missing_in_db]
- `1590` key=['2026-06-01', '1590'] [missing_in_db]
- `1611` key=['2026-06-01', '1611'] [missing_in_db]

### TaiwanStockDividend
- `00850` key=['2023-11-20', '00850', '112'] diffs=[('CashDividendPaymentDate', '1023-12-12', '')]
- `00939` key=['2026-06-08', '00939', '115'] [extra_in_db]
- `00940` key=['2026-06-15', '00940', '115'] [extra_in_db]
- `00946` key=['2026-06-08', '00946', '115'] [extra_in_db]
- `00953B` key=['2026-06-08', '00953B', '115'] [extra_in_db]
- `00984D` key=['2026-06-08', '00984D', '115'] [extra_in_db]
- `00985B` key=['2026-06-08', '00985B', '115'] [extra_in_db]
- `1219` key=['2026-06-14', '1219', '114年'] [extra_in_db]
- `1218` key=['2026-06-20', '1218', '114年'] [extra_in_db]
- `1233` key=['2026-07-27', '1233', '114年'] [extra_in_db]
- `1232` key=['2026-06-21', '1232', '114年'] [extra_in_db]
- `1234` key=['2026-07-05', '1234', '114年'] [extra_in_db]
- `1312A` key=['2026-06-19', '1312A', '不適用'] [extra_in_db]
- `1312` key=['2026-06-19', '1312', '不適用'] [extra_in_db]
- `1319` key=['2026-06-28', '1319', '114年'] [extra_in_db]
- `1323` key=['2026-06-08', '1323', '114年'] [extra_in_db]
- `1416` key=['2026-06-15', '1416', '114年'] [extra_in_db]
- `1473` key=['2026-06-16', '1473', '114年'] [extra_in_db]
- `1504` key=['2026-06-23', '1504', '114年'] [extra_in_db]
- `1513` key=['2026-07-15', '1513', '114年'] [extra_in_db]

### TaiwanStockInfo
- `(info)` key=['3687'] [value_mismatch] diffs=[('industry_category', '電子商務業', '數位雲端類')]
- `(info)` key=['3629'] [value_mismatch] diffs=[('industry_category', '光電業', '文化創意業')]
- `(info)` key=['5450'] [value_mismatch] diffs=[('stock_name', '寶聯通', '南良'), ('industry_category', '電腦及週邊設備業', '其他')]
- `(info)` key=['5481'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '其他')]
- `(info)` key=['6438'] [value_mismatch] diffs=[('industry_category', '其他電子類', '電子工業'), ('type', 'tpex', 'twse')]
- `(info)` key=['6426'] [value_mismatch] diffs=[('industry_category', '通信網路業', '電子工業'), ('type', 'tpex', 'twse')]
- `(info)` key=['3092'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '電子工業'), ('type', 'tpex', 'twse')]
- `(info)` key=['3669'] [value_mismatch] diffs=[('industry_category', '光電業', '電子工業')]
- `(info)` key=['1456'] [value_mismatch] diffs=[('industry_category', '紡織纖維', '建材營造')]
- `(info)` key=['1453'] [value_mismatch] diffs=[('industry_category', '紡織纖維', '建材營造')]
- `(info)` key=['2459'] [value_mismatch] diffs=[('industry_category', '電子通路業', '電子工業')]
- `(info)` key=['2614'] [value_mismatch] diffs=[('industry_category', '貿易百貨', '其他')]
- `(info)` key=['1443'] [value_mismatch] diffs=[('stock_name', '立益', '立益物流'), ('industry_category', '紡織纖維', '其他')]
- `(info)` key=['2241'] [value_mismatch] diffs=[('industry_category', '電機機械', '汽車工業')]
- `(info)` key=['3450'] [value_mismatch] diffs=[('industry_category', '其他電子業', '半導體業')]
- `(info)` key=['6165'] [value_mismatch] diffs=[('industry_category', '其他', '數位雲端')]
- `(info)` key=['8499'] [value_mismatch] diffs=[('industry_category', '其他', '電子工業')]
- `(info)` key=['2429'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '電子工業')]
- `(info)` key=['8354'] [value_mismatch] diffs=[('industry_category', '塑膠工業', '其他')]
- `(info)` key=['6275'] [value_mismatch] diffs=[('industry_category', '其他電子類', '電子零組件業')]

### FRED
- `fred_series` PALLFNFINDEXQ 2025-04-01: API=163.3754926931325 / DB=163.378298
- `fred_series` PALLFNFINDEXQ 2025-07-01: API=165.5241087552422 / DB=165.506674
- `fred_series` PALLFNFINDEXQ 2026-01-01: API=194.4702800391316 / DB=194.514936
