# 全市場全歷史同步證據報告 — §14.7-DD PHASE 4

**產生日期**：2026-06-01
**對應 workflow**：§14.7-DD 12-PHASE Tree-Family 從零重建 — PHASE 4(全市場全歷史 FinMind + FRED 灌注)
**對應 runbook**：`reports/tree_based_from_zero_build_runbook_20260531.md` L194-216
**資料真實性**：本報告全部數字 trace 回 (a) 程式 stdout(`/tmp/phase4_sync.log`)或 (b) DB query(isolated venv,READ-ONLY)per §一.10。

---

## 一、執行指令(source a)

```bash
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py \
    --universe full --all --dataset-batched --workers 4 --dynamic-quota \
    --special-full-market-reason "§14.7-DD 12-PHASE 從零重建 PHASE4：空 DB 全市場全歷史灌注 (DB rebuild bootstrap)"
```

- 模式：dataset-batched(逐 dataset 跑完全 ~2774 名冊再換下一 dataset)+ 4 threads 單 PID。
- `--universe full` 自動啟用 `--strict-source-history`(start=1990-01-01,resume off)。
- Dataset 順序(confirmed)：Price → Adj → PER → Inst → Margin → Hold → Fin → Bal → Rev → Div → FRED(4 序列,`--universe full` 自動觸發 Step 4.2)。

## 二、引擎主權判定(source a — 引擎收尾 summary block)

| 指標 | 值 |
|---|---|
| 📈 成功同步項目 | 24373 |
| ⚠️ 警告同步項目 | 3372 |
| ❌ 失敗同步項目 | **0** |
| ⏭ 跳過同步項目 | 0 |
| ♻️ 402-recovered | 0 |
| 📝 總計寫入筆數 | 81,015,600 |
| 🕒 總計耗時 | 19061.77 s(≈ 5.29 hr) |
| ⚖️ 主權判定 | WARNING |

**WARNING 成因(非資料問題)**：3372 warnings 經 log grep 證實**幾乎全為 §7.6 A5 動態配額 80% headroom 預警**（`⚠️ §7.6 A5 預警：window=4800/4800 (80%)`）+ 1 條 §6.8.7 第(4)條全市場例外觸發。屬 throttle 管理噪音，**非 data error**。`失敗=0` 為資料完整性訊號。程序以 clean summary block 退出（原 PID 85277 已 EXITED；殘留兩個 monitor shell 因 `pgrep` self-match 死鎖，已 kill 清理）。

## 三、DB 落地驗證(source b — `COUNT(DISTINCT stock_id)` / `COUNT(*)`)

### 3.1 FinMind 10 raw tables

| 資料集 | 表 | distinct stock_id | rows | 自然上限對照 |
|---|---|---:|---:|---|
| Price | TaiwanStockPrice | 2774 | 10,519,984 | 全名冊 ✓ |
| Adj | TaiwanStockPriceAdj | 2772 | 10,517,934 | ✓ |
| PER | TaiwanStockPER | 2018 | 7,353,665 | 上限 ~2018 ✓ |
| Inst | TaiwanStockInstitutionalInvestorsBuySell | 2761 | 25,068,187 | ✓ |
| Margin | TaiwanStockMarginPurchaseShortSale | 2230 | 7,723,999 | ✓ |
| Hold | TaiwanStockShareholding | 2420 | 8,381,096 | ✓ |
| Fin | TaiwanStockFinancialStatements | 2350 | 2,663,362 | ✓ |
| Bal | TaiwanStockBalanceSheet | 2356 | 8,249,000 | ✓ |
| Rev | TaiwanStockMonthRevenue | 2360 | 460,057 | ✓ |
| Div | TaiwanStockDividend | 2328 | 29,421 | ✓ |

> distinct < 2774 為正常（非每股皆具每種資料型別；delisted/年輕股缺特定 dataset）。全部落在既知自然上限，無異常缺口。

### 3.2 FRED

| 表 | rows | distinct series | 備註 |
|---|---:|---:|---|
| FredData | 48,895 | 4 | DFF / UNRATE / T10Y2Y / VIXCLS（PHASE 4 Step 4.2 自動灌注）|
| fred_series | 70,641 | 24 | 含 macro_beta（§一.15）所需 **IPG3344S / T10Y2Y / UNRATE** ✓ |

`fred_series` 24 series_id：B985RC1Q027SBEA, BAMLH0A0HYM2, CPIAUCSL, DGS10, DGS2, DGS3MO, DTWEXBGS, INDPRO, IPG3344S, LFWA64TTUSA647N, M2SL, PALLFNFINDEXQ, PATENTUSALLTOTAL, PCU4831114831115, QUSPAM770A, SPPOPDPNDOLUSA, T10Y2Y, T10Y3M, T10YIE, TCMDO, UMCSENT, UNRATE, VIXCLS, WTISPLC。

### 3.3 歷史涵蓋範圍(source b — MIN/MAX date)

| 表 | 最早 | 最新 |
|---|---|---|
| Price | 1992-01-04 | 2026-05-29 |
| Adj | 1992-01-04 | 2026-05-29 |
| Inst | 2005-01-03 | 2026-05-29 |
| Rev | 2002-02-01 | 2026-05-01 |
| FredData | 1948-01-01 | 2026-05-29 |
| fred_series | 1990-01-01 | 2026-05-29 |

各表最早日期 = FinMind/FRED API 各 dataset 之真實最早可得日期（Inst 2005 / Rev 2002 為 API 起始，非缺漏）。

## 四、PHASE 4 驗收結論

✅ **PHASE 4 完成**：FinMind 10 datasets + FRED FredData(4 序列)全市場全歷史灌注完成，0 失敗，全 distinct 落於自然上限，全歷史涵蓋確認。`fred_series`（24 序列）已存在並含後續 PHASE 7 macro_beta 所需 3 序列，解除前序 PHASE 3「fred_series 不存在」疑慮。

➡️ **下一步 PHASE 5**：`audit_full_db_vs_api_reconcile.py --scope core`（DB↔API byte-level 對帳，PASS 判準 `value_mismatch == 0`）。
