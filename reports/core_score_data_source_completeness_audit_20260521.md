# Core Score Data Source Completeness Audit — 2026-05-21

**對象**：CoreScore v0.2（§6.3）/ §0.1 + §0.2 + §0.3 三支柱在「核心股挑選」之資料來源對映
**狀態**：**INFORMATIONAL（非 PASS/FAIL 性質之跨層治權對映 audit）**
**Tool version**：手動 audit（無自動工具；資料源 = DB live query + `core_universe_scores.score_detail` JSONB 解析 + 憲章 cross-reference）
**Audit type**：跨層治權對映 audit（不取代 `audit_api_schema_compliance.py` schema 層 audit，亦不取代 `audit_core_universe.py` governance 層 audit）

---

## 1. 環境快照 (Environment Snapshot)

| 項目 | 值 |
|---|---|
| 執行時點 | 2026-05-21 19:00 Taipei (UTC+8) |
| Git HEAD | `9449e55` (fix(sovereign_sync_engine): v1.19 → v1.20 — §14.7-AM 4 步序列治權範本) |
| DB | PostgreSQL @ localhost:5432 |
| Schema | `public` |
| 當前 committed snapshot | `core_universe_20260514_core_universe_policy_v0_2` |
| Snapshot `as_of_date` | 2026-05-14 |
| Snapshot 分層 | core 120 / convex 30 / research 2270 / quarantine 378 = **2798** |
| 下游 §6.7 SQL 取得名單 | **150** (core ∪ convex) |
| Raw API tables 當前 row 數 | **全部 0** (從零重建中) |
| Governance tables 當前 row 數 | `core_universe_membership`: 2798 ✅ / `core_universe_scores`: 2798 ✅ |
| 對應憲章 patch | v6.0.0-patch（含今日 §14.7-AL / §14.7-AM 入憲）|

> **注意**：本 audit 之 coverage 數值有**雙時點**：
> - **NOW (2026-05-21)**：直接查 raw tables，全為 0（DB 從零重建中）
> - **2026-05-14 commit time**：來自 `core_universe_scores.score_detail` JSONB；為當時 v0.2 formal scoring 之 snapshot baseline，**非當前可觀測值**

---

## 2. 方法論 (Methodology)

### 2.1 Coverage % 計算來源

| 來源 | 時點 | 用途 |
|---|---|---|
| `SELECT COUNT(*) FROM <table>` | NOW | 顯示 raw table 當前資料量 |
| `score_detail->>'price_coverage_252d'` JSONB | 2026-05-14 commit | 顯示當時計分依據之 coverage |
| 「通過 ≥0.5 門檻」之計數 | 2026-05-14 commit | 顯示 150 支中有多少符合 §6.3 硬門檻 |

### 2.2 三支柱對映依據

- §0.1 第一性原理 → §0.1.1 T1/T2/T3 分層 + §0.1.3 V 補強 + §6.3 公式六層中之四層
- §0.2 八二法則 → §0.2.1 冪律 + §0.2-C 實證 + §6.3 分層規則之 cut-off
- §0.3 康波週期 → §0.3-A 治權分層強制條款 + §0.3-D MBNRIC 字典治權

### 2.3 樣本

`core_universe_membership.snapshot_id = 'core_universe_20260514_core_universe_policy_v0_2'`，共 2798 列；其中 150 = core_universe (120) + convex_universe (30) 為下游 §6.7 SQL 之預設 universe lock。

---

## 3. CoreScore v0.2 公式回顧

```text
CoreScore =
  0.25 × DataQuality          ← 元數據門檻（基礎，非治權支柱）
+ 0.25 × LiquidityMass        ← §0.1 T1 流動性質量 M
+ 0.20 × FundamentalGravity   ← §0.1.3 T1 內在價值密度 V
+ 0.15 × ThemeResonance       ← §0.3 K-wave 唯一投影 channel
+ 0.10 × InstitutionalFlow    ← §0.1 T2 資訊力 F proxy
+ 0.05 × VolatilityControl    ← §0.1 T1 ΔlnP 衍生
- RiskPenalty                 ← 絕對扣分，cap=99
```

**三支柱在公式內之權重對映**：

| 支柱 | 公式內權重 | 公式內 sub-layer 數 | 公式外體現 |
|---|---|---|---|
| §0.1 第一性原理 | **60%** | 4 (LM/FG/IF/VC) | T1 元素全部下沉到可觀測物理量 |
| §0.3 康波週期 | **15%** | 1 (TR) | 唯一進入 score 之宏觀訊號；macro features 永久不進 |
| §0.2 八二法則 | **0%** | 0 | 體現於「切 150」+「core/convex 槓鈴」+「max 5%/stock sizing」 |
| 基礎 (DQ + RP) | 25% + penalty | 2 | 元數據完整性 + 風險絕對扣分 |

---

## 4. §0.1 第一性原理 — 60% 權重 × 4/4 sub-layer 全部 DB-driven ✅

### 4.1 LiquidityMass (25%) — 物理意義：流動性質量 M

| 項目 | 值 |
|---|---|
| 主資料表 | `TaiwanStockPriceAdj` |
| 主欄位 | `Trading_money`（成交金額，TWD）, `Trading_Volume`, `close`, `date`, `stock_id` |
| Lookback window | 252 個交易日（最近 ~ 1 年） |
| 計分式 | `value_score × 0.85 + price_coverage_252d × 15`；`value_score = log10(Trading_money)` 線性映射 1M→0、10B→100 |
| 2026-05-14 commit avg LM score | core: **46.60** / convex: **82.71** |
| 2026-05-14 commit avg price_coverage_252d | core: **0.937** (93.7%) / convex: **0.968** (96.8%) |
| 150 支通過 ≥0.5 門檻 | **146/150 = 97.3%** ✅ |
| NOW row count | **0** ❌（DB 從零重建中） |
| §0.1.1 等級 | T1（不可分解物理事實）|

### 4.2 FundamentalGravity (20%) — 物理意義：內在價值密度 V (§0.1.3)

| 項目 | 值 |
|---|---|
| 主資料表 1 | `TaiwanStockMonthRevenue` |
| 主欄位 1 | `revenue`, `revenue_year`, `revenue_month`, `country`, `date`, `stock_id` |
| 主資料表 2 | `TaiwanStockFinancialStatements` |
| 主欄位 2 | `type`, `value`, `origin_name`, `date`, `stock_id` |
| Lookback window | 月營收 24 月 + 財報 8 季 |
| 計分式 | 基準 50；月營收 YoY > 30% +25 / > 10% +15 / > 0 +8 / < -5% -10 / < -20% -20；近 4 季 EPS 正 +；無獲利 -15；coverage 加分最多 +10 |
| 2026-05-14 commit avg FG score | core: **72.46** / convex: **91.19** |
| 2026-05-14 commit avg revenue_coverage_24m | core: **0.922** (92.2%) / convex: **0.992** (99.2%) |
| 2026-05-14 commit avg financial_coverage_8q | core: **0.964** (96.4%) / convex: **1.000** (100.0%) |
| 150 支通過 revenue ≥0.5 | **140/150 = 93.3%** ✅ |
| 150 支通過 financial ≥0.5 | **147/150 = 98.0%** ✅ |
| NOW row count | **0** ❌（兩表皆從零重建中） |
| §0.1.1 等級 | T1（V 變數本身）+ T2（V 之 P/V 計算方式）|

### 4.3 InstitutionalFlow (10%) — 物理意義：資訊力 F proxy

| 項目 | 值 |
|---|---|
| 主資料表 | `TaiwanStockInstitutionalInvestorsBuySell` |
| 主欄位 | `name` (外資/投信/自營商), `buy`, `sell`, `date`, `stock_id` |
| Lookback window | 90 個 calendar days |
| 計分式 | 基準 50；外資淨買 > 100M 股 +25 / > 10M +15 / > 0 +5；淨賣 < -10M -10 / < -100M -20；投信淨買 > 50M +15 / > 0 +5 / < -50M -10 |
| 2026-05-14 commit avg IF score | core: **64.17** / convex: **44.00** |
| Coverage 在 score_detail 中未獨立量化（隱含於 score 值與 risk_reasons） | — |
| NOW row count | **0** ❌ |
| §0.1.1 等級 | T2（F 概念之可觀測 proxy） |

### 4.4 VolatilityControl (5%) — 物理意義：ΔlnP 衍生

| 項目 | 值 |
|---|---|
| 主資料表 | `TaiwanStockPriceAdj` |
| 主欄位 | `close`, `date`, `stock_id` |
| Lookback window | 252 個交易日 |
| 計分式 | 計算 `cv_close`（收盤價變異係數）；cv ≤ 0.05 → 95；≤ 0.10 → 85；≤ 0.15 → 75；≤ 0.20 → 65；≤ 0.30 → 50；≤ 0.40 → 35；more → 20 |
| 2026-05-14 commit avg VC score | core: **53.17** / convex: **61.33** |
| 可用價格日數 < 20 時給中性 50 | — |
| NOW row count | **0** ❌ |
| §0.1.1 等級 | T1（時間軸不可逆 + ΔlnP 之直接衍生） |

### 4.5 §0.1 小結

- **60% 的 CoreScore 權重 100% 落在 4 張 DB tables 上**（其中 `TaiwanStockPriceAdj` 服務 2 個 sub-layer：LM + VC）
- **全部可觀測、可重算、可稽核**；§0.1.1 T1/T2 等級對映清晰
- 是三支柱中**最紮實 / 最 data-driven** 的一支
- 當前 DB 0 rows 為 §14.7-AL/AM 從零重建中間態，**非實作缺陷**

---

## 5. §0.3 康波週期 — 15% 權重 × 半 DB 半 hardcoded ⚠️

### 5.1 ThemeResonance (15%) — 每股之 TR 分數

| 項目 | 值 |
|---|---|
| 主資料表 | `TaiwanStockInfo` |
| 主欄位 | `industry_category`（TWSE/TPEx 之中文產業字串）, `stock_id`, `date` |
| 計分式 | 字串對照 `THEME_KEYWORDS` 字典，命中取對應分數；無命中或缺漏給中性偏低 **30** |
| 2026-05-14 commit avg TR score | core: **99.04** / convex: **100.00** / research: 62.60 / quarantine: 30.00 |
| `industry_category` 在 150 支之覆蓋率 | 隱含於 metadata；score 99-100 暗示 150 支幾乎全部命中 ≥1 主題 |
| NOW row count | **0** ❌（`TaiwanStockInfo` 從零重建中） |

### 5.2 THEME_KEYWORDS 字典 — 9 主題對映分數（非 DB-driven）

| 主題 | 分數 | MBNRIC 對應 | 治權位階 |
|---|---|---|---|
| 半導體 | 100 | Nano + Information(AI) | §0.3-D 直接核心 |
| 生技 / 醫療 | 95 | Medicine + Biotech | §0.3-D 直接核心 |
| 資訊 | 90 | Information | §0.3-D 直接核心 |
| 電腦 / 通信 | 85 | Information + Cognitive | §0.3-D 上游 |
| 電子 / 機器 | 80 | Robotics | §0.3-D 上游 |
| 電機 / 綠能 | 75 | (能源轉型) | §0.3-D 弱關聯 |
| 光電 / 能源 | 70 | (能源轉型) | §0.3-D 弱關聯 |
| 航太 | 65 | (Nano 上游) | §0.3-D 弱關聯 |
| 汽車 | 60 | Robotics（電動車）| §0.3-D 弱關聯 |
| 無命中 / 缺漏 | 30 | 中性偏低 | — |

| 字典治權項目 | 值 |
|---|---|
| 載體 | 程式碼模組常數於 `scripts/core/core_universe_builder.py` |
| §0.3-D 規範 | 權重設定範圍 [30, 100]；中性 30；直接核心 90-100；上游 60-89；弱關聯 30-59 |
| 更新頻率 | 原則每年 12 月（§6.8 年度重選前一次）|
| 非年度修改 | 須 `--special-rebalance-reason "THEME_KEYWORDS update: <條由>"` |
| 完整度 | **100%（hardcoded 故不存在 DB 缺漏問題）** |

### 5.3 FRED 宏觀資料 — 完全不進 CoreScore（重要 caveat）

| 項目 | 值 |
|---|---|
| 主資料表 | `FredData` |
| 主欄位 | `series_id` (DFF/VIX/T10Y2Y/UNRATE/...), `value`, `date`, `realtime_start`, `realtime_end` |
| 在 CoreScore v0.2 公式之權重 | **0%** |
| 為何 0%？ | §0.3-A 禁令 #4：「禁止把 macro features 解讀為 K-wave 即時訊號」；§0.3-C 實證：macro features 之 `drop_minus_full ≈ 0.0000` |
| FRED 資料當前用途 | 僅進 Feature Store macro 群（h20/h30 prediction 用），不影響核心股選拔 |
| NOW row count | **0** ❌ |

### 5.4 §0.3 小結

- 個股 TR score **來自 DB**（`TaiwanStockInfo.industry_category`）
- 字典本身（9 主題 → 分數）**寫死在程式**，非 DB
- **FRED 宏觀 0% 進 CoreScore**——使用者直覺中可能以為「景氣循環」需大量宏觀資料；事實上 K-wave 只透過 industry_category × 字典實現
- §0.3-A 禁令 #6 明文：「字典必須維持靜態 + 年度治理可修訂」

---

## 6. §0.2 八二法則 — 0% in 公式 / 完全結構性 ❌

### 6.1 四個結構性機制

| 機制 | 量化值 | 資料來源 | 性質 |
|---|---|---|---|
| 從 2798 取 **150 支** (5.4%) | `core_limit=120` + `convex_limit=30` | **無 DB 資料源** | 硬編於 `core_universe_builder.py` 模組常數 |
| **core / convex 槓鈴分層** | 「先取 `theme_score ≥ 70` 的前 30 為 convex」「剩餘前 120 為 core」 | **無 DB 資料源**（規則寫死）| 邏輯位於 `_assign_tiers()` |
| **max 5%/stock** sizing 上限 | `0.05` | **無 DB 資料源** | 硬編於 `scripts/inference/portfolio_sizer.py`（§9.2）|
| **§0.2-E P1 Hill estimator α** | 理論值待估 | 理論上應對 `TaiwanStockPriceAdj.Trading_money` 尾部分布 | **目前未實作自動 audit** |

### 6.2 §0.2 與 DB 之關係

| 問題 | 答案 |
|---|---|
| §0.2 的決策（150 / 5% / 槓鈴）是否來自 DB 計算？ | **否**。是憲章先寫死 constants，再事後檢驗 alpha 是否相容 |
| §0.2.1 之「150/2798 = 5.4% ≈ 95/5」實證從何而來？ | 一次性實證寫入 §0.2.1 / §0.2-C；非每年重算 |
| §0.2-E 之 7 項證偽承諾是否有自動 audit？ | **目前皆無自動 audit**；屬「持續驗證承諾」性質 |

### 6.3 §0.2 小結

- **0 張 DB tables 直接餵 §0.2 的計算**
- 150 這個數字是**規範性**而非**經驗性**——程式不會「自動發現 5.4% 是對的」
- §0.2 之治權合法性來自憲章 §0.2.0 學術源流（Pareto / Mandelbrot / Taleb / Barabási）+ §0.2.1 數學基礎，非 DB 資料

---

## 7. DataQuality (25%) + RiskPenalty — 基礎元數據門檻

### 7.1 DataQuality (25%)

| 項目 | 值 |
|---|---|
| 計分式 | `price_coverage_252d × 40 + revenue_coverage_24m × 30 + financial_coverage_8q × 30` |
| 主要 inputs（3 張 tables） | `TaiwanStockPriceAdj` / `TaiwanStockMonthRevenue` / `TaiwanStockFinancialStatements` |
| 2026-05-14 commit avg DQ | core: **94.05** / convex: **98.48** / research: 0.00 / quarantine: 0.00 |
| DQ < 20 額外懲罰 | +10 RP |

### 7.2 RiskPenalty

| 規則 | 觸發加分 | 主要資料源 |
|---|---|---|
| metadata 缺 `stock_name` / `type` / `industry_category` | +40 | `TaiwanStockInfo` |
| 非普通股語意產業（ETF/ETN/指數/權證）| +100（強制 quarantine） | `TaiwanStockInfo.industry_category` |
| `emerging` 且未開 `include_emerging` | +30 | `TaiwanStockInfo.type` |
| unsupported type | +30 | `TaiwanStockInfo.type` |
| 高波動 + 低流動 (`cv_close > 0.4 AND LM < 30`) | +15 | `TaiwanStockPriceAdj` |
| `DataQuality < 20` | +10 | （derived from DQ） |
| 異常槓桿變化 | (entry in `risk_reasons`) | `TaiwanStockMarginPurchaseShortSale` |
| 2026-05-14 commit avg RP | core: **0.25** / convex: **0.00** / research: 14.61 / quarantine: **99.00** (cap) |

---

## 8. 總覽表（一張全景）

| 治權支柱 | 公式內權重 | 資料源類型 | DB Tables 數 | 2026-05-14 commit coverage | NOW (2026-05-21) |
|---|---|---|---|---|---|
| **§0.1 第一性原理** | **60%** | ✅ 純 DB-driven | **4 表**（`TaiwanStockPriceAdj` × 2 用途、`TaiwanStockMonthRevenue`、`TaiwanStockFinancialStatements`、`TaiwanStockInstitutionalInvestorsBuySell`）| price 97.3% / revenue 93.3% / financial 98.0% (150 支通過率) | **DB 0 rows，等 sync** |
| **§0.3 康波週期** | **15%** | ⚠️ 半 DB / 半 hardcoded | **1 表** (`TaiwanStockInfo.industry_category`) + 字典於程式 | TR avg 99.04 (core) / 100.00 (convex) | **DB 0 rows；字典 100%** |
| **§0.2 八二法則** | **0%** | ❌ 完全結構性 | **0 表** | 規範性 constants 不可量化 | **不受 DB 狀態影響** |
| **DataQuality (基礎)** | 25% | ✅ DB-driven | 3 表（同 §0.1 之子集）| core 94.05 / convex 98.48 | **DB 0 rows** |
| **RiskPenalty (基礎)** | 絕對扣分 | ✅ DB-driven | 1 額外表（`TaiwanStockMarginPurchaseShortSale`）| core 0.25 / convex 0.00 / quarantine 99.00 | **DB 0 rows** |
| **FRED 宏觀**（§0.3 但不入 CoreScore）| 0% | ✅ DB-driven (僅 Feature Store macro 用) | 1 表 (`FredData`) | — | **DB 0 rows** |

---

## 9. 11 Tables × Row count NOW

| Table | NOW rows | NOW stocks | 對映 CoreScore Layer / 治權支柱 |
|---|---|---|---|
| `TaiwanStockInfo` | **0** | 0 | §0.3 TR + 候選池 + RP metadata |
| `TaiwanStockPriceAdj` | **0** | 0 | §0.1 LM (25%) + §0.1 VC (5%) + DQ (price_cov) |
| `TaiwanStockPrice` | **0** | 0 | (cross-check only, 非 v0.2 主依據) |
| `TaiwanStockMonthRevenue` | **0** | 0 | §0.1 FG (V) + DQ (revenue_cov) |
| `TaiwanStockFinancialStatements` | **0** | 0 | §0.1 FG (V) + DQ (financial_cov) |
| `TaiwanStockInstitutionalInvestorsBuySell` | **0** | 0 | §0.1 IF (F proxy, 10%) |
| `TaiwanStockMarginPurchaseShortSale` | **0** | 0 | RP 槓桿風險 |
| `TaiwanStockPER` | **0** | 0 | (informational; not in v0.2 score) |
| `TaiwanStockShareholding` | **0** | 0 | (not in CoreScore v0.2) |
| `TaiwanStockDividend` | **0** | 0 | (not in CoreScore v0.2) |
| `FredData` | **0** | 0 | (Feature Store macro only; 0% 進 CoreScore) |
| `core_universe_membership` (governance) | **2798** ✅ | 2798 | 150 = 120 core + 30 convex 名單仍在 |
| `core_universe_scores` (governance) | **2798** ✅ | 2798 | score_detail JSONB 保留歷史 coverage |

---

## 10. 三個關鍵 Finding

### Finding 1: 三支柱之 data-driven 程度差異懸殊

| 支柱 | data-driven 程度 |
|---|---|
| §0.1 第一性原理 | **完全 data-driven**（60% × 4 tables；T1/T2 元素皆可觀測）|
| §0.3 康波週期 | **半 data-driven**（個股 score 來自 DB；字典常數寫死）|
| §0.2 八二法則 | **完全非 data-driven**（純治權結構選擇；憲章寫死 constants）|

### Finding 2:「景氣循環」進入核心股之**唯一 channel** 是 industry_category × 字典

使用者直覺中「景氣循環」應大量倚賴宏觀資料（GDP、利率、失業率、VIX 等），但**事實上**：

- FRED 宏觀 (DFF/VIX/T10Y2Y/UNRATE) **0% 進入 CoreScore**（憲章 §0.3-A 禁令 #4 明文禁止）
- K-wave 的 15% 權重**完全透過** `TaiwanStockInfo.industry_category` 對照 **9 主題字典**實現
- 此設計合憲（§0.3-C 工程落地實況），但與直覺有顯著落差

### Finding 3: 當前「raw 0 rows + governance 2798 rows」是 §14.7-AL/AM 設計中間態

| 時序 | 狀態 |
|---|---|
| 2026-05-14 | CoreScore v0.2 formal scoring + commit snapshot |
| ~ 2026-05-20 | DB 重建 / wipe（屬合憲操作）|
| 2026-05-21 早 | codex 推 22 個 commit 完善 §14.7-AL/AM「從零 → 全市場全天數 + FRED 全歷史」4 步序列治權範本 |
| 2026-05-21 NOW | governance snapshot intact（150 名單）/ raw tables 0 rows / 等 `sovereign_sync_engine.py --universe core --all` 或全市場補刷 |

此狀態**非實作 bug**，是 §14.7-AL/AM 4 步治權範本明文允許的「先 governance 後 sync」順序。

---

## 11. 建議後續 (Recommended Follow-ups)

| 編號 | 建議 | 治權位階 |
|---|---|---|
| FU-01 | §0.2-E 之 7 項證偽承諾無自動 audit；可考慮 v0.3 補入 `audit_doctrine_compliance.py` 之 P2 層（含 Hill estimator α 持續觀測）| §0.0-E.6 P2 |
| FU-02 | 「DB rebuild → sync → re-score → re-audit」標準閉環 SOP 可正式入 §14.7-AN 或寫入獨立 `reports/db_rebuild_sop.md` | §0.0-E.6 P1 |
| FU-03 | 此 audit 為新類別「跨層治權對映 audit」之首例；若未來持續產生，建議命名規範 `<domain>_<topic>_audit_<YYYYMMDD>.md` | §四.4 |
| FU-04 | 若使用者希望「景氣循環」之 macro features 真正影響選股，需先通過 §0.3-C 之未來改進方向：將 macro 轉為 stock-sensitive regime interaction features；此為 v6.2.0+ 升版範圍 | §0.0-E.6 P3 |
| FU-05 | 建議在 §6.3 公式說明後追加一段 cross-ref：「§0.1 60% / §0.3 15% / §0.2 0% in 公式」之三支柱對映表（提升憲章可讀性，不變更治權）| §0.0-E.6 P3 |

---

## 附錄 A: SQL Queries Used (可重現)

### A.1 查表存在 + row count + stock count + date range

```sql
SELECT
  'TaiwanStockInfo' AS tbl, COUNT(*) AS rows, COUNT(DISTINCT stock_id) AS stocks,
  MIN(date)::text AS dmin, MAX(date)::text AS dmax
FROM "TaiwanStockInfo"
UNION ALL SELECT 'TaiwanStockPriceAdj', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockPriceAdj"
UNION ALL SELECT 'TaiwanStockPrice', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockPrice"
UNION ALL SELECT 'TaiwanStockMonthRevenue', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockMonthRevenue"
UNION ALL SELECT 'TaiwanStockPER', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockPER"
UNION ALL SELECT 'TaiwanStockInstitutionalInvestorsBuySell', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockInstitutionalInvestorsBuySell"
UNION ALL SELECT 'TaiwanStockMarginPurchaseShortSale', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockMarginPurchaseShortSale"
UNION ALL SELECT 'TaiwanStockFinancialStatements', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockFinancialStatements"
UNION ALL SELECT 'TaiwanStockShareholding', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockShareholding"
UNION ALL SELECT 'TaiwanStockDividend', COUNT(*), COUNT(DISTINCT stock_id), MIN(date)::text, MAX(date)::text FROM "TaiwanStockDividend"
UNION ALL SELECT 'FredData', COUNT(*), COUNT(DISTINCT series_id), MIN(date)::text, MAX(date)::text FROM "FredData"
ORDER BY tbl;
```

### A.2 當前 committed snapshot 分層組成

```sql
SELECT s.snapshot_id, s.as_of_date::text, s.status,
       SUM(CASE WHEN m.core_tier='core_universe'      THEN 1 ELSE 0 END) AS core,
       SUM(CASE WHEN m.core_tier='convex_universe'    THEN 1 ELSE 0 END) AS convex,
       SUM(CASE WHEN m.core_tier='research_universe'  THEN 1 ELSE 0 END) AS research,
       SUM(CASE WHEN m.core_tier='quarantine_universe' THEN 1 ELSE 0 END) AS quar
FROM core_universe_snapshot s
LEFT JOIN core_universe_membership m ON m.snapshot_id = s.snapshot_id
WHERE s.status='committed'
GROUP BY s.snapshot_id, s.as_of_date, s.status
ORDER BY s.as_of_date DESC LIMIT 3;
```

### A.3 各 tier 之六層平均分數（含 coverage）

```sql
SELECT
  m.core_tier,
  COUNT(*) AS n,
  ROUND(AVG((s.score_detail->>'data_quality_score')::numeric)::numeric,2)        AS dq,
  ROUND(AVG((s.score_detail->>'liquidity_mass_score')::numeric)::numeric,2)      AS lm,
  ROUND(AVG((s.score_detail->>'fundamental_gravity_score')::numeric)::numeric,2) AS fg,
  ROUND(AVG((s.score_detail->>'theme_resonance_score')::numeric)::numeric,2)     AS tr,
  ROUND(AVG((s.score_detail->>'institutional_flow_score')::numeric)::numeric,2)  AS if_,
  ROUND(AVG((s.score_detail->>'volatility_control_score')::numeric)::numeric,2)  AS vc,
  ROUND(AVG((s.score_detail->>'risk_penalty')::numeric)::numeric,2)              AS rp,
  ROUND(AVG((s.score_detail->>'core_score')::numeric)::numeric,2)                AS final_cs
FROM core_universe_membership m
JOIN core_universe_scores s ON s.snapshot_id = m.snapshot_id AND s.stock_id = m.stock_id
WHERE m.snapshot_id = 'core_universe_20260514_core_universe_policy_v0_2'
GROUP BY m.core_tier
ORDER BY m.core_tier;
```

### A.4 150 支通過 coverage ≥ 0.5 硬門檻之計數

```sql
SELECT
  COUNT(*) FILTER (WHERE (s.score_detail->>'price_coverage_252d')::numeric  >= 0.5) AS price_pass,
  COUNT(*) FILTER (WHERE (s.score_detail->>'revenue_coverage_24m')::numeric >= 0.5) AS rev_pass,
  COUNT(*) FILTER (WHERE (s.score_detail->>'financial_coverage_8q')::numeric >= 0.5) AS fin_pass,
  COUNT(*) AS total
FROM core_universe_membership m
JOIN core_universe_scores s ON s.snapshot_id = m.snapshot_id AND s.stock_id = m.stock_id
WHERE m.snapshot_id = 'core_universe_20260514_core_universe_policy_v0_2'
  AND m.core_tier IN ('core_universe','convex_universe');
```

### A.5 score_detail JSONB 完整 keys 清單

```sql
SELECT jsonb_object_keys(s.score_detail) AS key, COUNT(*) AS n
FROM core_universe_scores s
JOIN core_universe_membership m ON m.snapshot_id=s.snapshot_id AND m.stock_id=s.stock_id
WHERE m.snapshot_id='core_universe_20260514_core_universe_policy_v0_2'
  AND m.core_tier='core_universe'
GROUP BY 1 ORDER BY 1;
```

驗證得 18 keys：`candidate_source_mode`, `constitution`, `core_score`, `data_quality_score`, `downstream_boundary`, `financial_coverage_8q`, `fundamental_gravity_score`, `institutional_flow_score`, `liquidity_mass_score`, `price_coverage_252d`, `revenue_coverage_24m`, `risk_penalty`, `risk_reasons`, `score_scope`, `theme_resonance_score`, `tool_version`, `volatility_control_score`, `weights`。

---

## 附錄 B: 對應憲章節點 Cross-Reference

| 本 audit 章節 | 對應憲章節點 |
|---|---|
| §3 CoreScore 公式 | 憲章 §6.3 CoreScore v0.2 完整挑選方法 |
| §4 §0.1 第一性原理 | 憲章 §0.1 / §0.1.0 / §0.1.1 / §0.1.3 / §0.1-A / §0.1-B / §0.1-C / §0.1-E |
| §5 §0.3 康波週期 | 憲章 §0.3 / §0.3.0 / §0.3-A / §0.3-C / §0.3-D / §0.3-E |
| §6 §0.2 八二法則 | 憲章 §0.2 / §0.2.0 / §0.2.1 / §0.2-A / §0.2-C / §0.2-E |
| §7 DQ + RP | 憲章 §6.3 DataQuality / RiskPenalty 子章節 |
| §8 總覽 | 憲章 §0.0-A 三核心思想統合框架 |
| §9 Tables row count | 憲章 §6.3 第 8 類最小輸入資料契約 + §3.2 raw table |
| §10 Findings | 憲章 §14.7-AL / §14.7-AM（從零 → 全市場 4 步序列）|
| §11 後續 | 憲章 §0.0-E.6 升版優先級裁決 |
| §6.7 SQL 契約 | 用於取得 150 名單（core ∪ convex）|

---

**Audit 完成時點**：2026-05-21 19:00 Taipei
**Audit 性質**：跨層治權對映 audit（INFORMATIONAL，非 PASS/FAIL）
**Audit trail**：本檔為 §0.0-G 憲章先行紀律之 audit trail 留存
**後續動作**：不修憲、不 commit、不 push；待使用者明示授權
