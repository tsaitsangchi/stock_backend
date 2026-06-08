# 全股 × 全史 × 全表 DB↔API 完整對帳報告

- 產生時間:2026-06-08 23:10:25
- 工具:`audit_full_db_vs_api_reconcile.py` v0.2(§14.7-CE family)
- Scope:`all`(對帳 60 股)
- 歷史區間:1990-01-01 ~ 2026-06-08
- FinMind 表:TaiwanStockPrice, TaiwanStockPriceAdj, TaiwanStockPER, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockMarginPurchaseShortSale, TaiwanStockShareholding, TaiwanStockFinancialStatements, TaiwanStockBalanceSheet, TaiwanStockMonthRevenue, TaiwanStockDividend
- 實際 API calls:637 / 耗時:518.3s
- 數值容差:abs(a-b) ≤ max(1e-4, 1e-6·max(|a|,|b|));日期 ISO 比對;字串 strip exact
- Skip(ingest-metadata 透明揭露):{"TaiwanStockMonthRevenue": ["create_time"], "TaiwanStockInfo": ["date"]};FRED 僅比對 value(realtime_* vintage 不比)

## 裁決 (Verdict)

⚠️ **2 筆 value_mismatch** — 偵測到 API ≠ DB 之 row,須 root-cause(可能 publication-date 調整 / 待 resync / 計算欄)。

> `missing_in_db`(API 有 DB 無)/ `extra_in_db`(DB 有 API 無)為覆蓋率差異:
> 區間邊緣少量屬正常;大量則代表 DB 落後 → 須 `sovereign_sync_engine` resync(本程式不自動修)。

## FinMind per-stock 表(全史逐筆)

| 表 | 已對帳股數 | API rows | DB rows | matched | value_mismatch | missing_in_db | extra_in_db | api_fail |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TaiwanStockPrice | 60 | 166132 | 166132 | 166131 | 1 | 0 | 0 | 0 |
| TaiwanStockPriceAdj | 60 | 166114 | 166069 | 166069 | 0 | 45 | 0 | 0 |
| TaiwanStockPER | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TaiwanStockInstitutionalInvestorsBuySell | 60 | 673822 | 673787 | 673786 | 1 | 35 | 0 | 0 |
| TaiwanStockMarginPurchaseShortSale | 60 | 157458 | 157408 | 157408 | 0 | 50 | 0 | 0 |
| TaiwanStockShareholding | 60 | 167674 | 167620 | 167620 | 0 | 54 | 0 | 0 |
| TaiwanStockFinancialStatements | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TaiwanStockBalanceSheet | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TaiwanStockMonthRevenue | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TaiwanStockDividend | 60 | 221 | 221 | 221 | 0 | 0 | 0 | 0 |

## TaiwanStockInfo(市場註冊;單次全股 call)

- API rows(scope 內):61 / DB rows:60
- matched=60 / value_mismatch=0 / missing_in_db=1 / extra_in_db=0

## FRED 宏觀(全史 value-by-date)

- series 總數:28 / 100% match:28 / fail:0
- matched=119595 / value_mismatch=0 / missing_in_db=21588 / extra_in_db=3
  - `FredData`:4 series / matched=48914 / value_mismatch=0 / missing=0 / extra=0 / fail=0
  - `fred_series`:24 series / matched=70681 / value_mismatch=0 / missing=21588 / extra=3 / fail=0

## Mismatch 樣本(每類上限)

### TaiwanStockPrice
- `0050` key=['0050', '2026-06-08'] diffs=[('Trading_Volume', '524952235', '359952235.000000'), ('Trading_money', '52211365308', '36033115308.000000'), ('Trading_turnover', '599172', '599169.000000')]

### TaiwanStockPriceAdj
- `0050` key=['0050', '2026-06-08'] [missing_in_db]
- `0051` key=['0051', '2026-06-08'] [missing_in_db]
- `0052` key=['0052', '2026-06-08'] [missing_in_db]
- `00401A` key=['00401A', '2026-06-08'] [missing_in_db]
- `00403A` key=['00403A', '2026-06-08'] [missing_in_db]
- `0053` key=['0053', '2026-06-08'] [missing_in_db]
- `0055` key=['0055', '2026-06-08'] [missing_in_db]
- `0056` key=['0056', '2026-06-08'] [missing_in_db]
- `0061` key=['0061', '2026-06-08'] [missing_in_db]
- `006201` key=['006201', '2026-06-08'] [missing_in_db]
- `006204` key=['006204', '2026-06-08'] [missing_in_db]
- `006205` key=['006205', '2026-06-08'] [missing_in_db]
- `006206` key=['006206', '2026-06-08'] [missing_in_db]
- `006207` key=['006207', '2026-06-08'] [missing_in_db]
- `00625K` key=['00625K', '2026-06-08'] [missing_in_db]
- `00631L` key=['00631L', '2026-06-08'] [missing_in_db]
- `006208` key=['006208', '2026-06-08'] [missing_in_db]
- `00632R` key=['00632R', '2026-06-08'] [missing_in_db]
- `00633L` key=['00633L', '2026-06-08'] [missing_in_db]
- `00636` key=['00636', '2026-06-08'] [missing_in_db]

### TaiwanStockInstitutionalInvestorsBuySell
- `0050` key=['0050', '2026-06-08', 'Foreign_Investor'] diffs=[('buy', '193072340', '28072340.000000'), ('sell', '306397359', '141397359.000000')]
- `00625K` key=['00625K', '2026-06-08', 'Foreign_Investor'] [missing_in_db]
- `00625K` key=['00625K', '2026-06-08', 'Foreign_Dealer_Self'] [missing_in_db]
- `00625K` key=['00625K', '2026-06-08', 'Investment_Trust'] [missing_in_db]
- `00625K` key=['00625K', '2026-06-08', 'Dealer_self'] [missing_in_db]
- `00625K` key=['00625K', '2026-06-08', 'Dealer_Hedging'] [missing_in_db]
- `00631L` key=['00631L', '2026-06-08', 'Foreign_Investor'] [missing_in_db]
- `00631L` key=['00631L', '2026-06-08', 'Foreign_Dealer_Self'] [missing_in_db]
- `00631L` key=['00631L', '2026-06-08', 'Investment_Trust'] [missing_in_db]
- `00631L` key=['00631L', '2026-06-08', 'Dealer_self'] [missing_in_db]
- `00631L` key=['00631L', '2026-06-08', 'Dealer_Hedging'] [missing_in_db]
- `006208` key=['006208', '2026-06-08', 'Foreign_Investor'] [missing_in_db]
- `006208` key=['006208', '2026-06-08', 'Foreign_Dealer_Self'] [missing_in_db]
- `006208` key=['006208', '2026-06-08', 'Investment_Trust'] [missing_in_db]
- `006208` key=['006208', '2026-06-08', 'Dealer_self'] [missing_in_db]
- `006208` key=['006208', '2026-06-08', 'Dealer_Hedging'] [missing_in_db]
- `00636K` key=['00636K', '2026-06-08', 'Foreign_Investor'] [missing_in_db]
- `00636K` key=['00636K', '2026-06-08', 'Foreign_Dealer_Self'] [missing_in_db]
- `00636K` key=['00636K', '2026-06-08', 'Investment_Trust'] [missing_in_db]
- `00636K` key=['00636K', '2026-06-08', 'Dealer_self'] [missing_in_db]

### TaiwanStockMarginPurchaseShortSale
- `0050` key=['0050', '2026-06-08'] [missing_in_db]
- `0051` key=['0051', '2026-06-08'] [missing_in_db]
- `0052` key=['0052', '2026-06-08'] [missing_in_db]
- `00401A` key=['00401A', '2026-06-08'] [missing_in_db]
- `00403A` key=['00403A', '2026-06-08'] [missing_in_db]
- `00400A` key=['00400A', '2026-06-08'] [missing_in_db]
- `0053` key=['0053', '2026-06-08'] [missing_in_db]
- `0055` key=['0055', '2026-06-08'] [missing_in_db]
- `0056` key=['0056', '2026-06-08'] [missing_in_db]
- `0057` key=['0057', '2026-06-08'] [missing_in_db]
- `0061` key=['0061', '2026-06-08'] [missing_in_db]
- `006201` key=['006201', '2026-06-08'] [missing_in_db]
- `006203` key=['006203', '2026-06-08'] [missing_in_db]
- `006204` key=['006204', '2026-06-08'] [missing_in_db]
- `006205` key=['006205', '2026-06-08'] [missing_in_db]
- `006206` key=['006206', '2026-06-08'] [missing_in_db]
- `006207` key=['006207', '2026-06-08'] [missing_in_db]
- `00631L` key=['00631L', '2026-06-08'] [missing_in_db]
- `006208` key=['006208', '2026-06-08'] [missing_in_db]
- `00635U` key=['00635U', '2026-06-08'] [missing_in_db]

### TaiwanStockShareholding
- `0050` key=['0050', '2026-06-08'] [missing_in_db]
- `0051` key=['0051', '2026-06-08'] [missing_in_db]
- `0052` key=['0052', '2026-06-08'] [missing_in_db]
- `00401A` key=['00401A', '2026-06-08'] [missing_in_db]
- `00403A` key=['00403A', '2026-06-08'] [missing_in_db]
- `00400A` key=['00400A', '2026-06-08'] [missing_in_db]
- `0053` key=['0053', '2026-06-08'] [missing_in_db]
- `0055` key=['0055', '2026-06-08'] [missing_in_db]
- `0056` key=['0056', '2026-06-08'] [missing_in_db]
- `0057` key=['0057', '2026-06-08'] [missing_in_db]
- `0061` key=['0061', '2026-06-08'] [missing_in_db]
- `006201` key=['006201', '2026-06-08'] [missing_in_db]
- `006203` key=['006203', '2026-06-08'] [missing_in_db]
- `006204` key=['006204', '2026-06-08'] [missing_in_db]
- `006205` key=['006205', '2026-06-08'] [missing_in_db]
- `006206` key=['006206', '2026-06-08'] [missing_in_db]
- `006207` key=['006207', '2026-06-08'] [missing_in_db]
- `00625K` key=['00625K', '2026-06-08'] [missing_in_db]
- `00631L` key=['00631L', '2026-06-08'] [missing_in_db]
- `006208` key=['006208', '2026-06-08'] [missing_in_db]

### TaiwanStockInfo
- `(info)` key=['006201', 'tpex', '上櫃指數股票型基金(ETF)'] [missing_in_db]
