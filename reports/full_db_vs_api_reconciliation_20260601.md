# 全股 × 全史 × 全表 DB↔API 完整對帳報告

- 產生時間:2026-06-01 17:16:51
- 工具:`audit_full_db_vs_api_reconcile.py` v0.1(§14.7-CE family)
- Scope:`core`(對帳 397 股)
- 歷史區間:1990-01-01 ~ 2026-06-01
- FinMind 表:TaiwanStockPrice, TaiwanStockPriceAdj, TaiwanStockPER, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockMarginPurchaseShortSale, TaiwanStockShareholding, TaiwanStockFinancialStatements, TaiwanStockBalanceSheet, TaiwanStockMonthRevenue, TaiwanStockDividend
- 實際 API calls:4011 / 耗時:2654.8s
- 數值容差:abs(a-b) ≤ max(1e-4, 1e-6·max(|a|,|b|));日期 ISO 比對;字串 strip exact
- Skip(ingest-metadata 透明揭露):{"TaiwanStockMonthRevenue": ["create_time"], "TaiwanStockInfo": ["date"]};FRED 僅比對 value(realtime_* vintage 不比)

## 裁決 (Verdict)

⚠️ **5593 筆 value_mismatch** — 偵測到 API ≠ DB 之 row,須 root-cause(可能 publication-date 調整 / 待 resync / 計算欄)。

> `missing_in_db`(API 有 DB 無)/ `extra_in_db`(DB 有 API 無)為覆蓋率差異:
> 區間邊緣少量屬正常;大量則代表 DB 落後 → 須 `sovereign_sync_engine` resync(本程式不自動修)。

## FinMind per-stock 表(全史逐筆)

| 表 | 已對帳股數 | API rows | DB rows | matched | value_mismatch | missing_in_db | extra_in_db | api_fail |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TaiwanStockPrice | 397 | 2316606 | 2316209 | 2316209 | 0 | 397 | 0 | 0 |
| TaiwanStockPriceAdj | 397 | 2315902 | 2315813 | 2310267 | 5546 | 89 | 0 | 0 |
| TaiwanStockPER | 397 | 1827727 | 1827693 | 1827693 | 0 | 34 | 0 | 0 |
| TaiwanStockInstitutionalInvestorsBuySell | 397 | 6140678 | 6139463 | 6139463 | 0 | 1215 | 0 | 0 |
| TaiwanStockMarginPurchaseShortSale | 397 | 1965032 | 1965032 | 1965032 | 0 | 0 | 0 | 0 |
| TaiwanStockShareholding | 397 | 1934905 | 1934905 | 1934905 | 0 | 0 | 0 | 0 |
| TaiwanStockFinancialStatements | 397 | 650322 | 650322 | 650320 | 2 | 0 | 0 | 0 |
| TaiwanStockBalanceSheet | 397 | 1881825 | 1881825 | 1881825 | 0 | 0 | 0 | 0 |
| TaiwanStockMonthRevenue | 397 | 105747 | 105747 | 105747 | 0 | 0 | 0 | 0 |
| TaiwanStockDividend | 397 | 7450 | 7511 | 7450 | 0 | 0 | 61 | 0 |

## TaiwanStockInfo(市場註冊;單次全股 call)

- API rows(scope 內):609 / DB rows:397
- matched=352 / value_mismatch=45 / missing_in_db=0 / extra_in_db=0

## FRED 宏觀(全史 value-by-date)

- series 總數:28 / 100% match:28 / fail:0
- matched=119536 / value_mismatch=0 / missing_in_db=21586 / extra_in_db=0
  - `FredData`:4 series / matched=48895 / value_mismatch=0 / missing=0 / extra=0 / fail=0
  - `fred_series`:24 series / matched=70641 / value_mismatch=0 / missing=21586 / extra=0 / fail=0

## Mismatch 樣本(每類上限)

### TaiwanStockPrice
- `1102` key=['2026-06-01', '1102'] [missing_in_db]
- `1210` key=['2026-06-01', '1210'] [missing_in_db]
- `1215` key=['2026-06-01', '1215'] [missing_in_db]
- `1216` key=['2026-06-01', '1216'] [missing_in_db]
- `1227` key=['2026-06-01', '1227'] [missing_in_db]
- `1303` key=['2026-06-01', '1303'] [missing_in_db]
- `1319` key=['2026-06-01', '1319'] [missing_in_db]
- `1229` key=['2026-06-01', '1229'] [missing_in_db]
- `1402` key=['2026-06-01', '1402'] [missing_in_db]
- `1326` key=['2026-06-01', '1326'] [missing_in_db]
- `1434` key=['2026-06-01', '1434'] [missing_in_db]
- `1513` key=['2026-06-01', '1513'] [missing_in_db]
- `1476` key=['2026-06-01', '1476'] [missing_in_db]
- `1477` key=['2026-06-01', '1477'] [missing_in_db]
- `1514` key=['2026-06-01', '1514'] [missing_in_db]
- `1504` key=['2026-06-01', '1504'] [missing_in_db]
- `1515` key=['2026-06-01', '1515'] [missing_in_db]
- `1533` key=['2026-06-01', '1533'] [missing_in_db]
- `1522` key=['2026-06-01', '1522'] [missing_in_db]
- `1560` key=['2026-06-01', '1560'] [missing_in_db]

### TaiwanStockPriceAdj
- `2363` key=['2026-06-01', '2363'] [missing_in_db]
- `2399` key=['2026-06-01', '2399'] [missing_in_db]
- `2442` key=['2026-06-01', '2442'] [missing_in_db]
- `2492` key=['2026-06-01', '2492'] [missing_in_db]
- `2497` key=['2026-06-01', '2497'] [missing_in_db]
- `2542` key=['2026-06-01', '2542'] [missing_in_db]
- `3014` key=['2026-06-01', '3014'] [missing_in_db]
- `3022` key=['2026-06-01', '3022'] [missing_in_db]
- `3030` key=['2026-06-01', '3030'] [missing_in_db]
- `3033` key=['2026-06-01', '3033'] [missing_in_db]
- `3044` key=['2026-06-01', '3044'] [missing_in_db]
- `3105` key=['2026-06-01', '3105'] [missing_in_db]
- `3059` key=['2026-06-01', '3059'] [missing_in_db]
- `3088` key=['2026-06-01', '3088'] [missing_in_db]
- `3169` key=['2026-06-01', '3169'] [missing_in_db]
- `3189` key=['2026-06-01', '3189'] [missing_in_db]
- `3211` key=['2026-06-01', '3211'] [missing_in_db]
- `3257` key=['2026-06-01', '3257'] [missing_in_db]
- `3338` key=['2026-06-01', '3338'] [missing_in_db]
- `3402` key=['2026-06-01', '3402'] [missing_in_db]

### TaiwanStockPER
- `6196` key=['2026-06-01', '6196'] [missing_in_db]
- `6197` key=['2026-06-01', '6197'] [missing_in_db]
- `6202` key=['2026-06-01', '6202'] [missing_in_db]
- `6235` key=['2026-06-01', '6235'] [missing_in_db]
- `6239` key=['2026-06-01', '6239'] [missing_in_db]
- `6271` key=['2026-06-01', '6271'] [missing_in_db]
- `6278` key=['2026-06-01', '6278'] [missing_in_db]
- `6412` key=['2026-06-01', '6412'] [missing_in_db]
- `6438` key=['2026-06-01', '6438'] [missing_in_db]
- `6477` key=['2026-06-01', '6477'] [missing_in_db]
- `8011` key=['2026-06-01', '8011'] [missing_in_db]
- `8039` key=['2026-06-01', '8039'] [missing_in_db]
- `8046` key=['2026-06-01', '8046'] [missing_in_db]
- `8081` key=['2026-06-01', '8081'] [missing_in_db]
- `8110` key=['2026-06-01', '8110'] [missing_in_db]
- `8150` key=['2026-06-01', '8150'] [missing_in_db]
- `8213` key=['2026-06-01', '8213'] [missing_in_db]
- `8261` key=['2026-06-01', '8261'] [missing_in_db]
- `8462` key=['2026-06-01', '8462'] [missing_in_db]
- `8464` key=['2026-06-01', '8464'] [missing_in_db]

### TaiwanStockInstitutionalInvestorsBuySell
- `1565` key=['2026-06-01', '1565', 'Foreign_Investor'] [missing_in_db]
- `1565` key=['2026-06-01', '1565', 'Foreign_Dealer_Self'] [missing_in_db]
- `1565` key=['2026-06-01', '1565', 'Investment_Trust'] [missing_in_db]
- `1565` key=['2026-06-01', '1565', 'Dealer_self'] [missing_in_db]
- `1565` key=['2026-06-01', '1565', 'Dealer_Hedging'] [missing_in_db]
- `1784` key=['2026-06-01', '1784', 'Foreign_Investor'] [missing_in_db]
- `1784` key=['2026-06-01', '1784', 'Foreign_Dealer_Self'] [missing_in_db]
- `1784` key=['2026-06-01', '1784', 'Investment_Trust'] [missing_in_db]
- `1784` key=['2026-06-01', '1784', 'Dealer_self'] [missing_in_db]
- `1784` key=['2026-06-01', '1784', 'Dealer_Hedging'] [missing_in_db]
- `1785` key=['2026-06-01', '1785', 'Foreign_Investor'] [missing_in_db]
- `1785` key=['2026-06-01', '1785', 'Foreign_Dealer_Self'] [missing_in_db]
- `1785` key=['2026-06-01', '1785', 'Investment_Trust'] [missing_in_db]
- `1785` key=['2026-06-01', '1785', 'Dealer_self'] [missing_in_db]
- `1785` key=['2026-06-01', '1785', 'Dealer_Hedging'] [missing_in_db]
- `1815` key=['2026-06-01', '1815', 'Foreign_Investor'] [missing_in_db]
- `1815` key=['2026-06-01', '1815', 'Foreign_Dealer_Self'] [missing_in_db]
- `1815` key=['2026-06-01', '1815', 'Investment_Trust'] [missing_in_db]
- `1815` key=['2026-06-01', '1815', 'Dealer_self'] [missing_in_db]
- `1815` key=['2026-06-01', '1815', 'Dealer_Hedging'] [missing_in_db]

### TaiwanStockFinancialStatements
- `2484` key=['2025-12-31', '2484', 'EPS', '基本每股盈餘'] diffs=[('value', '0.52', '0.390000')]
- `4114` key=['2025-12-31', '4114', 'EPS', '基本每股盈餘'] diffs=[('value', '0.79', '0.380000')]

### TaiwanStockDividend
- `1215` key=['2026-06-02', '1215', '114年'] [extra_in_db]
- `1319` key=['2026-06-28', '1319', '114年'] [extra_in_db]
- `1513` key=['2026-07-15', '1513', '114年'] [extra_in_db]
- `1504` key=['2026-06-23', '1504', '114年'] [extra_in_db]
- `1515` key=['2026-06-16', '1515', '114年'] [extra_in_db]
- `1568` key=['2026-06-02', '1568', '114年'] [extra_in_db]
- `1590` key=['2026-07-29', '1590', '114年'] [extra_in_db]
- `1605` key=['2026-06-19', '1605', '114年'] [extra_in_db]
- `1702` key=['2026-06-03', '1702', '114年'] [extra_in_db]
- `1717` key=['2026-07-05', '1717', '不適用'] [extra_in_db]
- `1773` key=['2026-06-25', '1773', '114年'] [extra_in_db]
- `1810` key=['2026-06-20', '1810', '114年'] [extra_in_db]
- `2031` key=['2026-06-21', '2031', '114年'] [extra_in_db]
- `2105` key=['2026-06-25', '2105', '114年'] [extra_in_db]
- `2308` key=['2026-06-23', '2308', '114年'] [extra_in_db]
- `2313` key=['2026-06-08', '2313', '114年'] [extra_in_db]
- `2327` key=['2026-06-20', '2327', '114年後半年度'] [extra_in_db]
- `2330` key=['2026-06-17', '2330', '114年第4季'] [extra_in_db]
- `2385` key=['2026-06-21', '2385', '114年'] [extra_in_db]
- `2421` key=['2026-06-16', '2421', '114年'] [extra_in_db]

### TaiwanStockInfo
- `(info)` key=['6438'] [value_mismatch] diffs=[('industry_category', '其他電子業', '電子工業')]
- `(info)` key=['5243'] [value_mismatch] diffs=[('industry_category', '光電業', '電子工業')]
- `(info)` key=['5285'] [value_mismatch] diffs=[('industry_category', '電子工業', '半導體業')]
- `(info)` key=['5258'] [value_mismatch] diffs=[('industry_category', '電腦及週邊設備業', '電子工業')]
- `(info)` key=['5269'] [value_mismatch] diffs=[('industry_category', '半導體業', '電子工業')]
- `(info)` key=['4968'] [value_mismatch] diffs=[('industry_category', '半導體業', '電子工業')]
- `(info)` key=['6531'] [value_mismatch] diffs=[('industry_category', '半導體業', '電子工業')]
- `(info)` key=['4952'] [value_mismatch] diffs=[('industry_category', '電子工業', '半導體業')]
- `(info)` key=['5388'] [value_mismatch] diffs=[('industry_category', '通信網路業', '電子工業')]
- `(info)` key=['6196'] [value_mismatch] diffs=[('industry_category', '其他電子業', '電子工業')]
- `(info)` key=['6202'] [value_mismatch] diffs=[('industry_category', '半導體業', '電子工業')]
- `(info)` key=['6197'] [value_mismatch] diffs=[('industry_category', '電子零組件業', '電子工業')]
- `(info)` key=['6176'] [value_mismatch] diffs=[('industry_category', '電子工業', '光電業')]
- `(info)` key=['6209'] [value_mismatch] diffs=[('industry_category', '電子工業', '光電業')]
- `(info)` key=['5434'] [value_mismatch] diffs=[('industry_category', '電子工業', '電子通路業')]
- `(info)` key=['5471'] [value_mismatch] diffs=[('industry_category', '半導體業', '電子工業')]
- `(info)` key=['5469'] [value_mismatch] diffs=[('industry_category', '電子工業', '電子零組件業')]
- `(info)` key=['1773'] [value_mismatch] diffs=[('industry_category', '化學工業', '化學生技醫療')]
- `(info)` key=['1752'] [value_mismatch] diffs=[('industry_category', '生技醫療業', '化學生技醫療')]
- `(info)` key=['2393'] [value_mismatch] diffs=[('industry_category', '光電業', '電子工業')]
