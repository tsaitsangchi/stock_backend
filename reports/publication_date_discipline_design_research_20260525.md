# §8.5 第 9 條 Publication-date Discipline 強制契約 — 設計研究報告

- **產出日期**: 2026-05-25
- **產出者**: Codex (Opus 4.7, 1M context) session
- **觸發**: §0.1.3-B.7 C 類「需憲章先行入憲」第 1 項;對應 §0.1.3-B.5 發現 6(Publication-date leakage 未升強制契約);兌現 L5881 + §14.7-W 之「後續應研究」forward reference
- **位階**: **草案性提案(Level 1 強制契約草案)**;依 §0.0-G 紀律先入憲明文化治權規則,後續程式落地需另案授權
- **HEAD commit**: `8f40836`(§0.1.3-B + §14.7-AZ 入憲剛完成)
- **配套憲章節**: §8.5 anti-leakage 8 條(L4160-4171)/ §6.3 第 8 條 FRED vintage(L3185)/ §14.7-W feature_store_builder 研究承諾 / L5881 forward ref

---

## 一、執行摘要

| 項目 | 內容 |
|---|---|
| **核心發現** | DB 內 4 個 publication-date 候選欄位實證後**只有 2 個可用**:`Dividend.AnnouncementDate` 與 `FRED.realtime_start`;`MonthRevenue.create_time` 為 DB 寫入時間(lag 中位數 5 年)非公告日;`Shareholding.RecentlyDeclareDate` 之語意不明(lag 反向 161 天) |
| **草案策略修正** | 從原計畫「**一刀切 5 表 effective_date_for_audit gate**」修正為「**分層落地**」— Dividend/FRED 立即落地 + FinStmt 法定截止日推算 + MonthRevenue/Shareholding 另案研究 |
| **§8.5 第 9 條治權位階** | 全系統強制(Feature / Model / Prediction);**僅明文要求「對有可靠 publication-date 之 raw,必須以 publication-date 而非 statistical date 作 as-of gate」**;對無可靠 publication-date 之 raw,**容許**繼續用 statistical date 但**必須**在 `feature_definition.publication_date_source` 透明記述 |
| **追溯適用** | 既有 snapshot 不重 build(避免 destructive);新 snapshot(v0.4+)起適用;`audit_leakage.py v0.3` 加 `publication_date_check` |

---

## 二、現行 §8.5 8 條缺口確認

§8.5 anti-leakage 8 條原文(L4160-4171)涵蓋:

| 條 | 涵蓋層 | 缺口 |
|---|---|---|
| 1. as-of-strict filter | Feature/Model/Prediction `date <= as_of_date` | ❌ 不檢 publication-date |
| 2. label horizon 後置 | Model `label_date >= as_of_date + N` | — |
| 3. universe snapshot 鎖定 | All | — |
| 4. feature_set 鎖定 | Model/Prediction | — |
| 5. No hot-fix imputation | Feature | — |
| 6. No future split | Feature | — |
| 7. 零硬編預測 | Prediction | — |
| 8. 單一 SSOT(§6.7 延伸) | All | — |

**結構性缺口**:8 條皆隱含「raw data 一旦進 DB 即視為當日可得」,**無 vintage 校驗**。對基本面 raw(月營收 / 財報 / 股利)有 ~10-90 天的 publication delay,可能造成 forward-looking bias。

---

## 三、Publication-date 候選欄位之實證(2026-05-25 DB 查證)

對 13 表所有 publication-date 候選欄位,實測 lag 分布:

### 3.1 TaiwanStockMonthRevenue.create_time — ❌ 不可靠(DB 寫入時間非公告日)

| 統計 | 值 |
|---|---|
| total_rows | 40,960(每月 1 row × 2,343 stocks × 18 年) |
| has_create_time | 100%(40,960/40,960) |
| min_lag(create_time − date) | 1 day |
| **max_lag** | **8,753 days(~24 年)** |
| **avg_lag** | **2,988 days(~8 年)** |
| **median_lag** | **1,933 days(~5 年)** |
| p95_lag | 8,114 days(~22 年) |

**詮釋**:若 `create_time` 為「真實公告日」,lag 應為 ~10-30 天(每月 10 日前公告)。實際 median = 5 年 / max = 24 年,確認 `create_time` 之語意為「**FinMind 系統 / 本 DB 之 row 建立時間**」(歷史回補時批次寫入),**不是真實公告日**。

**裁決**:`MonthRevenue.create_time` **不適合作為 publication-date discipline 來源**;需另接真實公告日(可考慮硬編「每月 statistical month + 10 天」之保守上限規則)。

### 3.2 TaiwanStockDividend.AnnouncementDate — ✅ 可靠(lag ~23 天合理)

| 統計 | 值 |
|---|---|
| total_rows | 27,847 |
| has_announce | 100% |
| min(date − announce) | -10 days(少數 announce 後於除權息日 outlier) |
| max(date − announce) | 148 days |
| **avg(date − announce)** | **23.47 days** |

**詮釋**:`date` 為除權息基準日,`AnnouncementDate` 為公告日,**公告日先於除權息日約 23 天** — 符合台灣股利公告慣例(董事會公告 → 除權息日約 1 個月)。

**裁決**:`Dividend.AnnouncementDate` **可立即作為 publication-date discipline 來源**。

### 3.3 TaiwanStockShareholding.RecentlyDeclareDate — ❌ 語意不明(lag 反向)

| 統計 | 值 |
|---|---|
| total_rows | 8,216,504 |
| has_declare | 100% |
| min(declare − date) | -6,769 days(~19 年) |
| max(declare − date) | +1,985 days(~5 年) |
| **avg(declare − date)** | **-161 days(declare 在 date 之前 ~5 個月)** |

**詮釋**:`RecentlyDeclareDate` 平均比 `date`(每週統計日)早 ~5 個月,且分布極寬 — 不符合「公告日」語意。可能為「**上次完整申報日**」或「**最近一次外資持股結構大幅變動之申報日**」(stock-level snapshot 而非 row-level event)。

**裁決**:`Shareholding.RecentlyDeclareDate` **語意需另案研究**(未來研究方向 D2.1);本契約**不強制**使用此欄位作為 publication-date。

### 3.4 TaiwanStockFinancialStatements — ❌ 無 publication-date 欄位

DB schema(5 cols):`date / stock_id / type / value / origin_name` — **無 publication-date 欄位**。

**台灣財報法定截止日**(依 FSC 證券交易法施行細則):

| 季度 | statistical date | 法定截止日 | filing lag |
|---|---|---|---|
| Q1 | YYYY-03-31 | YYYY-05-15 | ~45 days |
| Q2 | YYYY-06-30 | YYYY-08-14 | ~45 days |
| Q3 | YYYY-09-30 | YYYY-11-14 | ~45 days |
| Q4 | YYYY-12-31 | YYYY+1-03-31 | ~90 days |

**裁決**:FinStmt 無 publication-date 欄位 → 採**法定截止日 + filing lag 推算**(保守上限,實際公告日 ≤ 法定截止日);此推算屬「硬編 filing lag rule」,須在 `feature_definition.publication_date_source = 'statutory_filing_deadline'` 透明記述。

### 3.5 FredData.realtime_start / realtime_end — ✅ vintage 機制(§6.3 第 8 條已入憲)

| Series | rows | min_lag | max_lag | avg_lag | median_lag |
|---|---|---|---|---|---|
| DFF | 26,257 | 1 | 26,257 | 13,129 | 13,129 |
| T10Y2Y | 12,490 | 0 | 18,251 | 9,120 | 9,120 |
| UNRATE | 939 | 46 | 28,626 | 14,351 | 14,350 |
| VIXCLS | 9,190 | 1 | 13,288 | 6,632 | 6,629 |

**詮釋**:lag 中位數極大(數千天)— 確認 `realtime_start` 為 FRED 之 vintage 機制(該 row 在 FRED 系統 vintage 起算日);需用 `realtime_start <= as_of_date` 條件過濾。對 historical 資料而言,`realtime_start` 多為「2024 年才被修正過的歷史 row 之 vintage 開始日」— 直接用無問題,因為過濾條件意義是「**該 vintage 在 as_of_date 當下能取得**」。

**裁決**:`FRED.realtime_start` **可立即作為 publication-date discipline 來源**(§6.3 第 8 條已入憲,本條為其升至全系統強制契約之配套)。

### 3.6 其他表(無 publication delay 議題)

| 表 | publication-date 處理 |
|---|---|
| TaiwanStockPrice / PriceAdj | `date` = trading day,T 日收盤後即可得;§6.8.7-A 已調 cron 17:30 |
| TaiwanStockPER | 同上 |
| TaiwanStockInstitutionalInvestorsBuySell | `date` = T,TWSE 17:00 後公告;§6.8.7-A 已涵蓋 |
| TaiwanStockMarginPurchaseShortSale | 同上 |
| TaiwanStockInfo | metadata snapshot,無 publication delay |

---

## 四、§8.5 第 9 條條文草案

### 4.1 條文新增(對接 §8.5 既有 8 條表格)

| 規則 | 適用層 | 違例範例 | 強制執行載體 |
|---|---|---|---|
| **9. Publication-date discipline**(本條新增) | Feature / Model / Prediction(以基本面 / 宏觀 raw 為輸入者) | 在 `as_of_date=2024-04-05` SELECT `Dividend` 之 row(其 `AnnouncementDate=2024-04-10`,尚未公告);或 SELECT `FRED.UNRATE` 之 row(其 `realtime_start=2024-04-15`,當下尚未發布)| builder/trainer/engine 必須使用 **`effective_publication_date <= as_of_date`** 之 SQL gate,而非 `date <= as_of_date`;`effective_publication_date` 按表分派(見 §8.5-9.2 表)|

### 4.2 `effective_publication_date` 分派表(per dataset)

| 表 | effective_publication_date | publication_date_source | 治權位階 |
|---|---|---|---|
| TaiwanStockPrice / PriceAdj | `date` | `trading_day` | T 日收盤後可得 |
| TaiwanStockPER | `date` | `trading_day` | 同上 |
| TaiwanStockInstitutionalInvestorsBuySell | `date` | `trading_day_post_1730`(§6.8.7-A) | T 日 17:30 後可得 |
| TaiwanStockMarginPurchaseShortSale | `date` | `trading_day` | T 日收盤後可得 |
| **TaiwanStockDividend** ✨ | `AnnouncementDate` | `announcement_date` | **本條強制** |
| **TaiwanStockMonthRevenue** ⚠️ | `date + INTERVAL '10 days'`(硬編保守上限) | `statutory_disclosure_deadline` | **本條強制硬編規則**(因 `create_time` 不可靠;見 §3.1)|
| **TaiwanStockFinancialStatements** ⚠️ | Q1/Q2/Q3: `date + INTERVAL '45 days'` / Q4: `date + INTERVAL '90 days'` | `statutory_filing_deadline` | **本條強制硬編規則**(因無 publication-date 欄位;見 §3.4)|
| TaiwanStockShareholding | `date`(暫維持;`RecentlyDeclareDate` 語意不明) | `statistical_date_pending_research` | **本條容許過渡**(待 §14.7 子節研究 `RecentlyDeclareDate` 語意後升版)|
| **FredData** ✨ | `realtime_start` | `fred_vintage_start` | **本條強制**(§6.3 第 8 條已有條文,本條升至 §8.5 強制) |
| TaiwanStockInfo | `date` | `registry_snapshot_date` | metadata,無 delay |

### 4.3 透明性要求

- `feature_definition.publication_date_source` 必須記述每個 feature 之 effective_publication_date 來源(對齊 §8.5 第 5 條 `null_strategy` 透明慣例)
- `audit_leakage.py v0.3` 加 `publication_date_check`:對每個 feature 用其 `publication_date_source` 之規則重算 `effective_publication_date`,驗證 `feature_value` 之 SQL gate 是否對齊
- `data_audit_log` 補入 `publication_date_strategy` 欄位(由 builder 寫入)

### 4.4 升版觸發

- v6.2.0+ feature_store_builder v0.4 必須升 SQL:
  - 從 `WHERE date <= as_of_date`
  - 到 `WHERE COALESCE(<publication_date_col>, <hardcoded_deadline>) <= as_of_date`
- v6.2.0+ core_universe_builder v0.4 同步升 SQL
- audit_leakage v0.3 補 `publication_date_check`

---

## 五、與既有契約之相容裁決

### 5.1 與 §6.3 第 8 條 FRED vintage 之關係

§6.3 第 8 條(L3185):「FRED:以 `as_of_date` 可取得的 `realtime_start` / `realtime_end` 版本為準,避免使用未來才修正完成的宏觀資料」

- §6.3 第 8 條:**已入憲,builder 0% 動員**(§0.1.3-B 發現 5「條文活實作死」)
- 本條 §8.5-9:**升至全系統強制契約**(non-FRED 表亦適用);FRED 部分為 §6.3 第 8 條之**執行載體升版**
- **裁決**:本條**不修改 §6.3 第 8 條原文**,而是將其精神普及至全系統 anti-leakage 範圍。§6.3 第 8 條保留(治權內容描述);§8.5-9 提供強制執行載體。

### 5.2 與 §8.5 第 1 條 as-of-strict filter 之關係

§8.5 第 1 條(L4164):「as-of-strict filter:Feature/Model/Prediction;違例範例 `WHERE date <= as_of_date + N` (N>0)」

- 第 1 條:防「**未來資料**」進入(future bias)
- 第 9 條:防「**已 statistical date 但尚未公告之資料**」進入(publication bias)
- **二者正交**:第 9 條為第 1 條之精細化補強(同一治權方向,粒度更細)
- 第 9 條 implies 第 1 條(若 `effective_publication_date <= as_of_date` 則必有 `date <= as_of_date`,因 publication_date ≥ statistical date)

### 5.3 與 §14.7-W feature_store_builder 研究承諾之兌現

§14.7-W(L102 修訂歷程記述):「§14.7-W 登錄研究發現與後續待研究項(**財報/月營收公告日語意**...)」

- 本條為 §14.7-W「公告日語意」研究承諾之**部分兌現**
- 完全兌現需待:(a) MonthRevenue 真實公告日來源確認(可能需另接 TWSE MOPS API);(b) Shareholding `RecentlyDeclareDate` 語意研究

### 5.4 與 §0.1.3-A.1「資料現實裁決」之精神對齊

§0.1.3-A.1(L1490)揭露 ROE「資料現實裁決」典範:「該 type 之 value 與 IncomeAfterTaxes value 幾乎相同... 無真正股東權益欄位」

- 本條同為「**資料層揭露驅動治權升版**」之 §14.7-AX 治權元規則第二次跑通(首次為 §0.1.3-A.1 之 ROE 不可實作)
- 對應發現:`MonthRevenue.create_time` 不是公告日 / `Shareholding.RecentlyDeclareDate` 語意不明 — 屬同類「資料現實 ≠ 治權預期」之揭露

### 5.5 治權邊界嚴守(本條不改之事項)

- **不**修改 §8.5 既有 8 條原文(僅新增第 9 條)
- **不**修改 §6.3 第 8 條 FRED vintage 條文
- **不**修改 §6.4 治理欄位
- **不**修改 §9.1 / §9.2 / §9.9 強制契約
- **不**修改 CoreScore v0.2 公式與權重
- **不**修改 §0.1 / §0.1.1 / §0.1.3 / §0.1.3-A / §0.1.3-B 任何子節
- **不**修改任何 raw DDL
- **不**修改 `audit_leakage.py` v0.2 既有 18 項檢查邏輯(v0.3 為新增 1 項)
- **不**追溯重 build 既有 snapshot(避免 destructive change)

---

## 六、可證偽承諾(對接 §8.5 既有 audit 載體)

### 6.1 一致性檢驗(必過 / 不過則撤銷 §8.5-9 入憲)

| 指標 | 觀察期 | 通過門檻 | 不通過則裁決 |
|---|---|---|---|
| **T9.1 Dividend AnnouncementDate gate 之 IC 影響** | 滾動 5 年 | 引入 publication-date gate 前後,prediction IC 差異 < 1%(若差異大則 publication-date 確實有效) | 若差異 = 0,撤銷 Dividend 之 publication-date gate(non-load-bearing)|
| **T9.2 FinStmt 法定截止日推算之穩健性** | 滾動 5 年 | 用 `date + 45/90 days` gate 後,h20 IC 不應顯著下降(< 5%);若顯著下降代表現行 builder 確實依賴 publication-bias | 若下降 > 5%,進一步研究 filing lag 之 stock-level 分布 |
| **T9.3 FRED vintage gate 之 walk-forward 穩健性** | 滾動 10 年 | `realtime_start` gate 對 macro features 之 walk-forward IC 應穩定(< 0.05 stdev) | 若不穩定,撤回 FRED vintage 為強制要求(降為建議) |

### 6.2 audit 載體升版

- `audit_leakage.py v0.3` 加 `publication_date_check` 之 SQL gate 驗收
- 對每個 committed feature_set,執行「**重算 effective_publication_date,驗證 feature_values 之原 SQL 之 effective_publication_date 等於重算值**」
- FAIL gate 觸發 → ConstitutionalViolationError(per §9.2-D.1 模式)

---

## 七、追溯適用裁決

| 既有 snapshot/feature_set/model | 處理 |
|---|---|
| `feature_set_v0.1` ~ `v0.3`(2026-05-15 至 2026-05-20)| **不重 build**(避免 destructive change);標記 `publication_date_strategy='legacy_statistical_date'` |
| 新 snapshot(`v0.4+`,本條入憲後)| **強制適用**第 9 條;`publication_date_strategy='per_table_dispatch_v1'` |
| 既有 `audit_leakage.py v0.2` 報告 | **不重跑**;v0.3 為新增 1 項檢查,不取代 v0.2 之 18 項 |
| 既有 walk-forward h20/h30 panel(2024-2026)| **保留為 legacy evidence**;v0.4 起新 panel 採新規則 |
| 現行 production-current 升版 gate(`v6.1.1`,等 2026-06-04)| **不延後**;v6.1.1 仍走 legacy_statistical_date;**v6.2.0+ 強制 publication_date discipline** |

---

## 八、實作建議(v6.2.0 軌道,本研究不執行)

### 8.1 程式落地計畫

| 模組 | 升版 | 變更 |
|---|---|---|
| `data_schema.py` | v2.17 → v2.18 | DATASET_REGISTRY 加 `publication_date_strategy` per dataset(對齊 §8.5-9.2 分派表)|
| `feature_store_builder.py` | v0.3 → v0.4 | SQL `WHERE` clause 升至 `effective_publication_date` gate;`feature_definition` 補 `publication_date_source` |
| `core_universe_builder.py` | v0.3 → v0.4 | 與 feature_store_builder 同步升 SQL |
| `audit_leakage.py` | v0.2 → v0.3 | 加 `publication_date_check`(rule 19) |
| `prediction_engine.py` | v0.2 → v0.3 | 對齊 feature_set v0.4 之 publication_date_strategy |

### 8.2 落地序列建議

```
Step 1: data_schema v2.18 加 DATASET_REGISTRY[t]['publication_date_strategy']
Step 2: feature_store_builder v0.4 dry-run on as_of_date=2025-03-31(historical)
Step 3: 比對 v0.3 vs v0.4 之 feature_values 差異(預期 fundamental 群有 0-90 天 shift)
Step 4: 跑 walk-forward h20 panel(v0.4)+ ablation 對照 v0.3
Step 5: 若 IC 穩定(T9.1-T9.3 全通過),audit_leakage v0.3 加 publication_date_check
Step 6: prediction_engine v0.3 切到 v0.4 feature_set
Step 7: 升憲章 v6.2.0(§8.5 第 9 條 ACTIVE)
```

### 8.3 風險與回退方案

| 風險 | 機率 | 回退 |
|---|---|---|
| FinStmt 法定截止日推算造成 historical IC 下降 > 5% | 中 | 將 FinStmt 移至「容許過渡」分類(同 Shareholding) |
| `Dividend.AnnouncementDate` 之 -10 天 outlier 造成 audit FAIL | 低 | audit 容忍 ±3 天偏差(對齊 §6.8.8-C 時點漂移容忍) |
| FRED vintage 機制造成 macro features 大量 null(early as_of_date) | 中 | feature_definition `null_strategy='fred_vintage_padding'` 處理 |
| MonthRevenue 硬編 +10 天保守上限造成 revenue YoY 信號延遲 | 高 | 接受延遲(該 bias 屬「**正當保守**」,不算 leakage 反向錯誤) |

---

## 九、後續研究方向(本草案外)

| 編號 | 方向 | 對應憲章節 |
|---|---|---|
| **D2.1** | `Shareholding.RecentlyDeclareDate` 語意研究(對應 §3.3 揭露)| §14.7 子節 |
| **D2.2** | `MonthRevenue.create_time` 真實語意確認 + 另接 TWSE MOPS 公告 API 之可行性 | §14.7 + §7 sync 擴充 |
| **D2.3** | FinStmt 真實 filing date 抓取(目前 DB 無;TWSE MOPS API 提供 announcement_time)| §7 sync 擴充 |
| **D2.4** | FRED 之 ALFRED archival vintage 抓取(取代 realtime_start/end 之單一 vintage)| §7 sync 擴充 |
| **D2.5** | Backtest 用 «walk-forward h30» panel 驗證 §8.5-9 之 IC 影響 | §9.1 / walk-forward 框架 |

---

## 十、治權聲明

### 10.1 嚴守 §0.0-G 憲章先行紀律

本研究報告為**草案性提案**,先入憲明文化治權規則,後續程式落地另案授權:
- 第 1 步:憲章先行(本報告 → 入憲為 §8.5 第 9 條 + §14.7-BA 治權閉環記述)
- 第 2 步:程式落地(等用戶授權)— data_schema v2.18 / feature_store_builder v0.4 / audit_leakage v0.3
- 第 3 步:walk-forward 驗證(T9.1-T9.3 證偽指標)
- 第 4 步:憲章升版 v6.1.1 → v6.2.0(若驗證通過)

### 10.2 與 §0.1.3-B 之關係

本草案為 §0.1.3-B.7 C 類「需憲章先行入憲」之**第 1 項落地**(發現 6 對應之兌現)。同類:
- 第 2 項:§9.10 起草(VolatilityControl 升版;待 §9.9 ablation)
- 第 3 項:§6.4 第七維(宏觀 F')sub-score 對映

### 10.3 與 §14.7-AX 治權元規則對齊

本草案再次驗證「**資料層揭露驅動治權升版**」之機制(§14.7-AX):
- §0.1.3-B field 盤點(2026-05-25)→ 揭露 Publication-date leakage(發現 6)→ 本研究報告兌現 L5881 + §14.7-W forward reference

### 10.4 治權對映表

| §0 支柱 | 本草案對映 |
|---|---|
| §0.1 第一性原理 | 時間單向性 T1(L1338)— 本條為其第 4 層(anti-leakage 層 publication-date discipline)補完 |
| §0.4 可觀察性 | `audit_leakage.py v0.3` 之 publication_date_check 為 §0.4 在 leakage 層之延伸 |

---

## 十一、入憲與後續路徑

### 11.1 待用戶授權之入憲動作

| 入憲落點 | 內容 |
|---|---|
| **§8.5 第 9 條**(對接既有 8 條表格)| 新增規則行(對應本研究 §4.1)|
| **§8.5-9 子節**(新建,對應 §8.5-1 至 §8.5-8 之展開模式)| 5 個子子節:9.1 分派表 / 9.2 透明性要求 / 9.3 升版觸發 / 9.4 與既有契約相容裁決 / 9.5 追溯適用 |
| **§14.7-BA**(新建子節)| 治權閉環記述(觸發 / 5 個漸進釐清 / 實證 / §0.0-G 第 22 次跑通 / 模式對比) |
| **修訂歷程 entry** | v6.1.0-patch 2026-05-25 第三輪 |

### 11.2 待用戶授權之程式落地(v6.2.0 軌道)

| 階段 | 內容 |
|---|---|
| **Phase 1** | data_schema v2.18 加 publication_date_strategy(per dataset) |
| **Phase 2** | feature_store_builder v0.4 落地 SQL gate 升版 |
| **Phase 3** | audit_leakage v0.3 加 publication_date_check |
| **Phase 4** | walk-forward h20/h30 panel 驗證 T9.1-T9.3 |
| **Phase 5** | 升憲章至 v6.2.0(若 T9.1-T9.3 全通過) |

### 11.3 接續點

本草案完成後,用戶可選:
- 甲:授權入憲(§8.5 第 9 條 + §14.7-BA;憲章先行)
- 乙:擱置入憲,先做 Phase 1-2 程式 dry-run + walk-forward 驗證 → 再決定是否入憲
- 丙:擱置整案(因 §0.1.3-A.1 ROE 教訓:資料現實裁決可能否決強制要求)
- 丁:展開 D2.2-D2.4 sync 擴充(另接 TWSE MOPS / FRED ALFRED API)

---

## 附錄 A — DB 實證查詢全紀錄(2026-05-25)

```sql
-- MonthRevenue create_time lag(揭露為 DB 寫入時間)
SELECT COUNT(*), MIN(create_time - date), MAX(...), AVG(...), median, p95
FROM TaiwanStockMonthRevenue WHERE create_time IS NOT NULL;
-- 結果:median lag = 1,933 days (~5 年);max = 8,753 days (~24 年)

-- Dividend AnnouncementDate lag(揭露為公告日,合理)
SELECT COUNT(*), MIN(date - AnnouncementDate), MAX(...), AVG(...)
FROM TaiwanStockDividend WHERE AnnouncementDate IS NOT NULL;
-- 結果:avg = 23.47 days;範圍 -10 ~ 148 days

-- Shareholding RecentlyDeclareDate lag(揭露為語意不明)
SELECT COUNT(*), MIN(RecentlyDeclareDate - date), MAX(...), AVG(...)
FROM TaiwanStockShareholding WHERE RecentlyDeclareDate IS NOT NULL;
-- 結果:avg = -161 days(反向!);範圍 -6,769 ~ +1,985 days

-- FinStmt date 分布(揭露為無 publication-date 欄位)
SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT date)
FROM TaiwanStockFinancialStatements;
-- 結果:1990-03-31 → 2026-03-31,145 distinct quarter ends

-- FRED realtime_start lag per series(vintage 機制)
SELECT series_id, COUNT(*), MIN(realtime_start - date), MAX(...), AVG(...), median
FROM FredData WHERE realtime_start IS NOT NULL GROUP BY series_id;
-- 結果:DFF/T10Y2Y/UNRATE/VIXCLS 各 median 6,629 ~ 14,350 days(歷史 vintage)
```

## 附錄 B — 憲章條文 cross-ref(已實際驗證)

| 引用條文 | 入憲狀態 | 行號 |
|---|---|---|
| §6.3 第 8 條 FRED vintage | ✅ 已入憲 | L3185 |
| §8.5 anti-leakage 8 條 | ✅ 已入憲 | L4160-4171 |
| §14.7-W feature_store first-principles research | ✅ 已入憲(2026-05-19,修訂歷程 L102)| 修訂歷程 |
| L5881「後續應研究公告日 / create_time / filing lag rule」 | ✅ forward reference 已入憲 | L5881 |
| §0.1.3-A.1「資料現實裁決」(ROE 不可實作) | ✅ 已入憲(2026-05-24) | L1490 |
| §0.1.3-B(本草案之 parent) | ✅ 已入憲(2026-05-25;commit 8f40836) | L1493-1611 |
| §14.7-AX「資料層揭露驅動治權升版」 | ✅ 已入憲 | §14.7-AX |
| §0.0-G 憲章先行紀律 | ✅ 已入憲(永久強制) | §0.0-G |
| §9.2-D.1 ConstitutionalViolationError 模式 | ✅ 已入憲 | §9.2-D.1 |
| §6.8.8-C 時點漂移容忍 | ✅ 已入憲 | §6.8.8-C |
| §6.8.7-A 日頻 cron 17:30 | ✅ 已入憲 | §6.8.7-A |

## 附錄 C — 與 §0.1.3-B 之關係(雙視角後之第三步)

| 步驟 | 內容 | 時點 |
|---|---|---|
| §0.1.3-A | top-down 揭露(2026-05-24) | 完成 |
| §0.1.3-B | bottom-up 補完(2026-05-25 早) | 完成 |
| **本草案** | bottom-up 之發現 6 之強制契約起草(2026-05-25 晚) | **本研究** |
| 後續 1 | 入憲 §8.5 第 9 條 + §14.7-BA(待授權) | — |
| 後續 2 | Phase 1-5 程式落地(待授權) | — |
