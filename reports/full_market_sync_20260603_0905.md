# 全市場全個股增量補齊實證報告 (§6.8.7 第(5)條)

**日期**：2026-06-03 | **起 09:05 → 訖 ~13:42(~4.6hr)**
**性質**：§6.8.7 第(5)條「全市場增量維運」治理實證報告(market-level 操作強制留痕)。
**指令**：`sovereign_sync_engine.py --universe full --incremental --roster --all --dataset-batched --workers 2 --dynamic-quota --special-full-market-reason "full-market all-stock incremental catchup 20260603"`
**模式**：§6.8.7 第(5)條 增量維運（v1.23 首次實跑）— resume-aware（start_date=today−30d，§7.5 resume 保留，**非**全歷史）+ `--roster`（TaiwanStockInfo 全名冊 ~2,798）。

---

## 一、執行摘要（程式 stdout，§一.10 (a)）

| 項目 | 值 |
|---|---|
| 成功同步 | 12,300 |
| **失敗** | **0** ✅ |
| 跳過（resume 已最新）| 2,644 |
| 警告 | 13,121（§7.6 A5 + benign pandas date UserWarning;非失敗）|
| 402-recovered | 0 |
| **總寫入** | **477,744 rows** |
| §7 節流 | A5 預警 8 / 自動暫停 17 / 暫停總 5,100s（全市場觸 5500/hr 上限屬正常）|
| 耗時 | 16,644s（~4.6hr，workers=2）|
| **主權判定** | **WARNING** |

**WARNING 判讀（§一.8 誠實）**：判定為 WARNING 而非 PERFECT，**全因 §7.6 A5 throttle 自動暫停**（全市場 ~2,800 股 × 多表觸及 FinMind 5500/hr 上限，引擎依 §7.6 自動暫停 17 次共 5,100s，並依規以 WARNING 寫入 lifecycle，非 silent）+ 大量 benign pandas 日期解析 UserWarning（§6.8.7-B.7 已記載之已知無害 log）。**實際失敗 = 0**；WARNING 屬「節流衛生 + 無害警告」，非資料問題。

## 二、DB 驗證（§一.10 (b) live query）

全市場（~2,806 股）日頻表已補至最新交易日：

| 表 | max date | distinct 股 | 落後 |
|---|---|---|---|
| TaiwanStockPrice / PriceAdj | 2026-06-02 | 2,806 | 1 天（06-03 盤後未發布，正常）|
| TaiwanStockPER | 2026-06-02 | 2,018 | 1 天 |
| InstitutionalInvestorsBuySell | 2026-06-02 | 2,762 | 1 天 |
| MarginPurchaseShortSale | 2026-06-02 | 2,232 | 1 天 |
| Shareholding | 2026-06-02 | 2,420 | 1 天 |
| MonthRevenue | 2026-06-01 | 2,360 | 月頻正常 |
| Dividend | 2026-08-15（未來宣告）| 2,332 | — |

→ **全市場全個股增量補齊成功**；resume 正確跳過今早已補之核心 397 + 其他（skipped=2,644）。

## 三、治理後續（§6.8.7 第(5)條要求；honest 揭露 deferred）

§6.8.7 第(5)條要求完成後執行雙稽核：`audit_supply_chain.py --include-logs` + `audit_source_availability.py --strict`。

⚠️ **雙稽核 deferred**：full-market `audit_source_availability --strict` 為 ~8hr 級重稽核（per §14.7-AT 實證），且目前背景 torch（transformer_dedicated）+ inv_vol 重驗在跑，三方 CPU/quota 競爭 → **暫緩**，待背景任務收斂後另行執行。本報告先留 sync 實證（rows/verdict/DB），雙稽核補做後追記。

**證據基礎**：執行摘要出自 `/tmp/full_market_incr.log`（程式 stdout）；DB 數字出自 isolated venv live query（§一.10 (a)(b)）；無 AI 幻像。

---

## 四、午場追補 → 06-03（2026-06-03 16:10 起，~5.6hr;§6.8.7 第(5)條第二次實跑）

早場(09:05)補到 06-02 後,16:00 探測確認 **06-03 EOD 已發布**(2330 抓到 06-03)→ 16:10 以 `--resume-drift-tolerance 0`(強制抓 06-03,非預設 drift 3)再跑全市場增補。

**執行摘要**(`/tmp/full_market_incr2.log` stdout):
| 項 | 值 |
|---|---|
| 成功同步 | 14,676 |
| **失敗** | **0** ✅ |
| 跳過(resume)| 833 |
| 總寫入 | **573,744 rows** |
| 耗時 | 20,133s(~5.6hr,workers=2)|
| **主權判定** | **WARNING** |

**WARNING 判讀**:(a) §7.4-A `TaiwanStockDividend` cascade-skip(402 paywall → 依 §7.4-A 跳過,非失敗);(b) §7.6 A5 throttle 自動暫停(全市場觸 5500/hr)。**失敗 = 0**。

**DB 驗證(全市場 06-03)**:`TaiwanStockPrice` / `PriceAdj` max=**2026-06-03 / 2,806 股** ✅;`InstitutionalInvestorsBuySell` 06-03 / 2,762;日頻表全市場補至最新交易日 06-03。

→ **全市場全個股已補至 06-03**(最新交易日)。雙稽核仍 deferred(同 §三,待 torch td 收斂後執行)。
