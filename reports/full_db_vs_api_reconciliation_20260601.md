# 全股 × 全史 × 全表 DB↔API 完整對帳報告

- 產生時間:2026-06-01 03:52:53
- 工具:`audit_full_db_vs_api_reconcile.py` v0.1(§14.7-CE family)
- Scope:`all`(對帳 30 股)
- 歷史區間:1990-01-01 ~ 2026-06-01
- FinMind 表:TaiwanStockPrice, TaiwanStockPriceAdj, TaiwanStockPER, TaiwanStockInstitutionalInvestorsBuySell, TaiwanStockMarginPurchaseShortSale, TaiwanStockShareholding, TaiwanStockFinancialStatements, TaiwanStockBalanceSheet, TaiwanStockMonthRevenue, TaiwanStockDividend
- 實際 API calls:338 / 耗時:231.5s
- 數值容差:abs(a-b) ≤ max(1e-4, 1e-6·max(|a|,|b|));日期 ISO 比對;字串 strip exact
- Skip(ingest-metadata 透明揭露):{"TaiwanStockMonthRevenue": ["create_time"], "TaiwanStockInfo": ["date"]};FRED 僅比對 value(realtime_* vintage 不比)

## 裁決 (Verdict)

🎯 **PASS** — 所有 key 同時存在於 API 與 DB 之 rows **100% byte-level 一致**(value_mismatch=0)。
→ 全 DB 值 = FinMind/FRED API origin;**0 system-generated / 0 AI 幻像值**(per §一.10 #1)。

> `missing_in_db`(API 有 DB 無)/ `extra_in_db`(DB 有 API 無)為覆蓋率差異:
> 區間邊緣少量屬正常;大量則代表 DB 落後 → 須 `sovereign_sync_engine` resync(本程式不自動修)。

## FinMind per-stock 表(全史逐筆)

| 表 | 已對帳股數 | API rows | DB rows | matched | value_mismatch | missing_in_db | extra_in_db | api_fail |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TaiwanStockPrice | 30 | 95912 | 95912 | 95912 | 0 | 0 | 0 | 0 |
| TaiwanStockPriceAdj | 30 | 95895 | 95895 | 95895 | 0 | 0 | 0 | 0 |
| TaiwanStockPER | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TaiwanStockInstitutionalInvestorsBuySell | 30 | 350110 | 350110 | 350110 | 0 | 0 | 0 | 0 |
| TaiwanStockMarginPurchaseShortSale | 30 | 91592 | 91592 | 91592 | 0 | 0 | 0 | 0 |
| TaiwanStockShareholding | 30 | 96705 | 96705 | 96705 | 0 | 0 | 0 | 0 |
| TaiwanStockFinancialStatements | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TaiwanStockBalanceSheet | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TaiwanStockMonthRevenue | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TaiwanStockDividend | 30 | 219 | 219 | 219 | 0 | 0 | 0 | 0 |

## TaiwanStockInfo(市場註冊;單次全股 call)

- API rows(scope 內):31 / DB rows:30
- matched=30 / value_mismatch=0 / missing_in_db=0 / extra_in_db=0

## FRED 宏觀(全史 value-by-date)

- series 總數:28 / 100% match:28 / fail:0
- matched=119536 / value_mismatch=0 / missing_in_db=21586 / extra_in_db=0
  - `FredData`:4 series / matched=48895 / value_mismatch=0 / missing=0 / extra=0 / fail=0
  - `fred_series`:24 series / matched=70641 / value_mismatch=0 / missing=21586 / extra=0 / fail=0

## Mismatch 樣本(每類上限)

（無 mismatch 樣本）
