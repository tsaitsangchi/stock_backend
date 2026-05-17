# Core Universe Sync 完整性驗證紀錄 — 2026-05-17 20:46

> 依憲章 v6.0.0-patch §6.8.7（日頻 core sync）+ §6.8.8 / §6.8.8-A（完整性驗證）執行；本紀錄為後續修改 `系統架構大憲章_v6.0.0.md` 之依據。

---

## 0. 執行概要

- **核心股 snapshot**：`core_universe_20260514_core_universe_policy_v0_2`（committed, as_of_date=2026-05-14）
- **核心股範圍**：120 core_universe + 30 convex_universe = **150 支**
- **執行命令**：`venv/bin/python scripts/ingestion/sovereign_sync_engine.py --universe core --days 7`
- **執行時點**：2026-05-17 20:45（星期日，盤後）
- **最近台股交易日**：2026-05-15 (Fri)
- **同步引擎版本**：`sovereign_sync_engine.py v1.11a`

## 1. Phase 1 — Pre-sync DB-only probe（§6.8.8-A 六步法）

### 1.1 時間錨點分類

| 錨點 | 值 |
|---|---|
| 今日 | 2026-05-17 (Sun, 非交易日) |
| 最近台股交易日 | 2026-05-15 |
| FRED `DFF` / `VIXCLS` 預期最新 | 2026-05-14（T+1 公布） |
| FRED `T10Y2Y` 預期最新 | 2026-05-15（同日可得） |
| FRED `UNRATE` 預期最新 | 2026-04-01（月頻，5 月首週公布 4 月值） |
| 月營收預期最新 | 2026-04（5/10 前公告） |
| 季報預期最新 | 2026-03-31（Q1 法定 5/15 前） |

### 1.2 Pre-sync 異常清單（150 → 4 類異常 11 stocks）

| 類型 | stock_id | stock_name | 性質 |
|---|---|---|---|
| A. 殭屍股 | 1701 | 中化 | 已下市/長期停止交易（最後價 2024-08-20）|
| A. 殭屍股 | 1729 | 必翔 | 已下市/長期停止交易（最後價 2017-05-17）|
| A. 殭屍股 | 3559 | 全智科 | 已下市/長期停止交易（最後價 2017-08-23）|
| B. tpex margin 結構缺漏 | 6708 | 天擎 | tpex 半導體業，FinMind margin 資料集不涵蓋或未進入信用交易 |
| B. tpex margin 結構缺漏 | 6907 | 雅特力-KY | 同上；另股利資料亦無（KY 新上市公司） |
| B. tpex margin 結構缺漏 | 7751 | 竑騰 | 同上 |
| B. tpex margin 結構缺漏 | 7770 | 君曜 | 同上 |
| B. tpex margin 結構缺漏 | 7772 | 耀穎 | 同上 |
| B. tpex margin 結構缺漏 | 7810 | 捷創科技 | 同上 |
| B. tpex margin 結構缺漏 | 7828 | 創新服務 | 同上 |
| B. tpex margin 結構缺漏 | 8102 | 傑霖科技 | 同上 |

## 2. Phase 2 — sovereign_sync_engine 執行結果

```
📈 成功同步項目 : 4         (FRED 4 序列全歷史分頁)
⚠️ 警告同步項目 : 20        (= 3 殭屍 × 4 個股表 + 8 tpex × Margin)
❌ 失敗同步項目 : 0
⏭ 跳過同步項目 : 580        (§7.5 resume — 已有 ≥ 2026-05-10 資料)
♻️ 402-recovered: 0
📝 總計寫入筆數 : 48,862
🕒 總計耗時     : 29.47 s
⚖️ 主權判定     : WARNING
```

### 2.1 20 條警告分類

- **Cat A — 殭屍股（12 筆）**：1701 / 1729 / 3559 × (Price + InstitutionalInvestors + MarginPurchaseShortSale + PER) → 每股 4 個日頻表全部 API 回 0 筆，是已下市的真實實況
- **Cat B — tpex 半導體 margin 結構缺漏（8 筆）**：6708 / 6907 / 7751 / 7770 / 7772 / 7810 / 7828 / 8102 × MarginPurchaseShortSale → FinMind 該資料集不含這些股票，是來源契約缺漏

### 2.2 Pipeline lifecycle

```
task_name='sync_all_core', status='warning', last_end=2026-05-17 20:45:55
```

## 3. Phase 3 — Post-sync DB-only probe

| 維度 | 結果 |
|---|---|
| Healthy universe（剔除 3 殭屍） | 147 支 |
| 5 個日頻個股表 = 最新交易日 2026-05-15 | **147 / 147** OK |
| Margin = 最新交易日（再剔除 8 tpex） | **139 / 139** OK |
| FRED 4 序列依各自頻率分層 | OK |

**裁決：`DB_COVERAGE_OK=True`（healthy 147 支）**

## 4. 過程問題清單（憲章修改依據）

### 問題 #1：核心股 snapshot 含已下市殭屍標的（CRITICAL）

- **現象**：150 支 core+convex 中存在 3 支已下市股票（1701 / 1729 / 3559），最長 9 年無交易資料
- **根因推測**：CoreScore v0.2 commit 時未過濾 `TaiwanStockInfo.is_delisted` / 最新交易日 < cutoff 之標的；或快照產生後標的相繼下市但無自動下架機制
- **對下游影響**：
  - §8 Feature Store 對殭屍股全部 imputed → 污染特徵分佈
  - §9 Portfolio Allocation 可能對下市股 sizing > 0 → 治權邊界違反（第 5 條：不對非上市標的）
  - sync 引擎每日報 12 條 warning → 噪音掩蓋真實警報
- **建議入憲項目**（v6.0.0-patch 候選）：
  1. §6.8 應強制 `core_universe_builder` 在 commit 前以 `TaiwanStockPrice.MAX(date) >= as_of_date - 30 days` 過濾活躍度
  2. 既有 committed snapshot 若發現殭屍股，須以 §6.8.6 bootstrap_intermediate 機制重簽 + audit trail，不得熱修改 membership
  3. 日常 sync 對殭屍股應 short-circuit（不打 API、不報 warning），改入 `quarantine_universe` 補位
  4. §9.4 治權邊界第 5 條（不對非上市標的）需有 audit hook 在 portfolio_sizer commit 前阻擋

### 問題 #2：tpex 部分標的結構性無信用交易資料（HIGH）

- **現象**：8 支 tpex 半導體股（6708 / 6907 / 7751 / 7770 / 7772 / 7810 / 7828 / 8102）`TaiwanStockMarginPurchaseShortSale` API 永遠回 0 筆
- **根因推測**：
  - (i) 這些標的可能屬「未開放信用交易」（新上市未滿 6 個月 / 全額交割 / 公司治理不符融資條件）
  - (ii) 或 FinMind 該 dataset 對 tpex 涵蓋不完整
- **對下游影響**：
  - 信用交易特徵（融資餘額 / 融券餘額 / 券資比）在這 8 支永遠為 NULL → §8 Feature Store imputation 必須處理
  - 每日 sync 報 8 條 warning → 噪音
- **建議入憲項目**：
  1. §6.8.7 sync 應引入「來源能力字典」(`SOURCE_CAPABILITY_MAP`)，明示哪些 `stock_id × dataset` 組合為結構性 NA，sync 對這些組合 short-circuit
  2. §6.8.8-A 完整性驗收應將「結構性 NA」與「同步遺漏」明確分離；DB_COVERAGE_OK 計算應排除已聲明的結構性 NA
  3. §8 Feature Store 對結構性 NA 之 imputation 策略須登錄（如：信用交易特徵對 8 支 tpex 半導體股一律 imputation=0 而非中位數）
  4. 須與 FinMind 端確認：tpex margin 是否屬於來源側限制？建議定期執行 `check_finmind_datalist.py` 驗證 dataset coverage

### 問題 #3：6907 雅特力-KY 股利資料缺漏（MEDIUM）

- **現象**：6907 健康日頻表全部 = 2026-05-15，但 `TaiwanStockDividend` 為 NULL
- **根因推測**：KY 公司（境外公司在台第二上市）可能尚未派息或派息週期與本國公司不同
- **對下游影響**：
  - 股利率特徵在 6907 為 NULL
  - 不應影響 §9 sizing，但須在 Feature Store imputation 表中登錄
- **建議入憲項目**：與問題 #2 一併納入「結構性 NA 登錄表」；KY 公司股利資料屬合理結構缺漏

### 問題 #4：DB-only probe 將「結構性 NA」誤判為「同步缺漏」（憲章漏洞，HIGH）

- **現象**：§6.8.8-A 現行六步法第 5 步「終端判定」未區分「該股 × 該表結構性無資料」與「同步漏抓」，導致 pre-sync probe 把 11 支結構性缺漏標的全列為 stale，sync 後仍列為「異常」
- **根因**：§6.8.8-A 寫作時假設 healthy universe，未涵蓋殭屍股 / 結構性 NA 情境
- **建議入憲項目**（v6.0.0-patch 必修）：
  1. §6.8.8-A 應新增第 7 步「結構性 NA 排除」：在終端判定前以「來源能力字典」與「殭屍股名單」減集
  2. 報告抬頭應明示 `healthy_universe_size` 與 `effective_denominator`，不得僅以 150/2798 為分母
  3. `DB_COVERAGE_OK` 應改為三層判定：
     - L1: 全 universe（含殭屍 + NA） — 僅供 audit
     - L2: healthy universe — 用於下游 gate
     - L3: effective universe（healthy 且該表非結構性 NA） — 用於該表單一驗收

### 問題 #5：sync 引擎 WARNING 主權判定噪音過多（MEDIUM）

- **現象**：sovereign_sync_engine 對 (殭屍 × 4 表) + (tpex × Margin) 累計 20 條 warning，整體判定 WARNING；但真實營運上這 20 條皆為結構性已知，非須警報事項
- **根因**：§7 三層防禦未引入「已知結構缺漏 allowlist」概念，所有 API 0-row 皆升為 warning
- **建議入憲項目**：
  1. §7.6 應新增 A6 條款「結構性 0-row allowlist」，允許在 `(stock_id, dataset)` 維度標記 `expected_empty=True`，命中 allowlist 之 0-row 不計入 warning 計數，僅寫入 INFO
  2. 主權判定應以「非 allowlist 之 warning 計數 > 0」為 WARNING 條件
  3. allowlist 變更需 audit log（who / when / reason）

### 問題 #6：當前 §6.8.8 / §6.8.8-A 缺少「核心股全 universe completeness」之執行 SOP（MEDIUM）

- **現象**：§6.8.8 僅定義單一股票 (2330) probe；§6.8.8-A 提供方法論但未明示「核心股 150 支全 universe completeness」之強制執行 SOP（如：每日 sync 後是否強制執行、報告留存路徑、Pass / Fail 升 §8 gate 之條件）
- **建議入憲項目**：
  1. 新增 §6.8.8-B「核心股 universe completeness daily protocol」：每日 `--universe core --days 7` 後強制執行 universe-wide probe，並產出 `reports/core_sync_completeness_<YYYYMMDD>_<HHMM>.md`
  2. 與 §8 升版 gate 連動：若 healthy universe DB_COVERAGE_OK = False，§8 production-current promotion 應 BLOCKED
  3. 與 cron 排程（§6.8.7-A）對應：日頻 core sync 完成後自動觸發 probe；月頻 doctrine audit 抽樣比對 probe 結果歷史趨勢

## 5. 本輪整體裁決

| 項目 | 結論 |
|---|---|
| 同步動作 | 完成，無 failed |
| Healthy universe 完整性 | 147 / 147 OK；FRED 4 序列 OK |
| 結構性異常 | 3 殭屍股 + 8 tpex margin + 1 KY 股利 — 不影響下游正確性，但須入憲分離 |
| `DB_COVERAGE_OK`（healthy 147） | **True** |
| `FINMIND_ALL_OK`（healthy 147 × 各表 effective denominator） | **True**（pending §6.8.8-A 修訂後正式定義） |
| `FRED_VALID_ALL_OK` | **True** |
| 後續憲章修改點 | 6 個問題（見 §4），其中 #1 #4 為 CRITICAL / HIGH |

## 6. 後續行動

1. 提交本紀錄至 git 作為 audit trail
2. 依 §4 六個問題草擬 v6.0.0-patch 修訂條文（建議分 P0 / P1）
3. 與 §6.8.6 bootstrap 機制保持相容；殭屍股清理走 special_rebalance audit，不得熱修改 committed snapshot
4. 在憲章下次修訂時新增 §6.8.8-B（universe-wide protocol）與 §7.6 A6（結構性 allowlist）
