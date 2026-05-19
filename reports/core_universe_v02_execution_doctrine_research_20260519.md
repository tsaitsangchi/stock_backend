# core_universe_builder.py 三核心思想轉換研究記錄

**研究日期**: 2026-05-19
**研究對象**: `scripts/core/core_universe_builder.py`
**程式版本**: `core_universe_builder.py v0.2`
**憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §0.0-A / §0.1 / §0.2 / §0.3 / §0.4 / §6
**研究目的**: 說明第一性原理、八二法則、康波週期如何在第一支核心治理程式中轉成 `core_universe` / `convex_universe` / `research_universe` / `quarantine_universe`。

---

## 1. 研究結論

`core_universe_builder.py` 是目前最接近「從零開始」理解三核心思想工程化的第一支程式。它不是最底層的路徑、schema 或資料抓取工具，而是第一個把 §0 核心思想轉成可提交 DB 治理結果的程式。

它的核心作用是：

```text
全市場候選股票
  -> 資料完整性檢查 (§0.4)
  -> 六層 CoreScore (§0.1 + §0.2 + §0.3)
  -> 風險隔離與右尾集中 (§0.2)
  -> committed universe snapshot (§6.7 SQL SSOT)
```

因此本程式是「哲學層 → 治理層」的第一個主要轉換器。

---

## 2. 憲章統合框架對映

憲章 §0.0-A 定義三核心思想統合為：

```text
Right-Tail Opportunity
  = f(First-Principle Force, Pareto Concentration, Kondratiev Direction)
```

在 `core_universe_builder.py` 中，這不是直接寫成 hardcoded 公式，而是落成六層 CoreScore：

```text
CoreScore = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP
```

| CoreScore 層 | 程式函式 / 欄位 | 對應核心思想 | 說明 |
|---|---|---|---|
| `DataQuality` | `_data_quality_score_v2()` | §0.4 | 先確認 price / revenue / financial coverage，沒有資料不得進入哲學推論。 |
| `LiquidityMass` | `_liquidity_mass_score()` | §0.1 + §0.2 | 流動性是價格力學的 `M`，也是右尾集中與 preferential attachment 的基礎。 |
| `FundamentalGravity` | `_fundamental_gravity_score()` | §0.1 | 營收成長、EPS、稅後淨利作為 `V` intrinsic value density 的 proxy。 |
| `ThemeResonance` | `_theme_resonance_score()` / `THEME_KEYWORDS` | §0.3 | 用產業分類對映第六波 MBNRIC 方向，只作用於 universe 治理層。 |
| `InstitutionalFlow` | `_institutional_flow_score()` | §0.1 + §0.2 | 外資 / 投信買超代表外部資訊力與資金注意力集中。 |
| `VolatilityControl` | `_volatility_control_score()` | §0.2 | 用 CV 控制不穩定價格路徑，避免右尾候選變成左尾風險。 |
| `RiskPenalty` | `_risk_profile()` + extra penalty | §0.2 + §0.4 | ETF/ETN/權證、metadata 缺口、高波動低流動性、資料品質不足進入懲罰或隔離。 |

---

## 3. 從資料到 universe 的實際流程

### 3.1 Preflight: 先由 §0.4 裁判資料是否可用

程式先檢查治理表是否存在：

- `TaiwanStockInfo`
- `core_universe_policy`
- `core_universe_snapshot`
- `core_universe_membership`
- `core_universe_scores`
- `universe_revision_log`

接著用 `V02_INPUT_CONTRACT` 檢查八類輸入表：

- `TaiwanStockInfo`
- `TaiwanStockPriceAdj`
- `TaiwanStockMonthRevenue`
- `TaiwanStockPER`
- `TaiwanStockInstitutionalInvestorsBuySell`
- `TaiwanStockMarginPurchaseShortSale`
- `TaiwanStockFinancialStatements`
- `FredData`

這一段的意義是：三核心思想不能在空資料或錯 schema 上運作。§0.4 不是附屬章節，而是核心思想能否落地的裁判。

### 3.2 Candidate source: 決定候選 universe 來源

候選股票由 `TaiwanStockInfo` 產生。正式狀態優先使用：

```text
candidate_source_mode = as_of_filtered
```

只有在 bootstrap 候選數低於 `core_limit + convex_limit` 時，才會進入：

```text
candidate_source_mode = latest_registry_fallback
```

程式已把 fallback 明確標成 warning，代表它可以協助空 DB 重建，但不是正式 v0.2 評分的理想狀態。

### 3.3 Market data loading: 把原始資料轉為六層 proxy

`_load_market_data()` 讀取：

- 近一年 `TaiwanStockPriceAdj`: 交易金額、成交量、連續性、收盤價 CV
- 近兩年 `TaiwanStockMonthRevenue`: 月營收覆蓋率、近 12 月 vs 前 12 月 YoY
- 近兩年 `TaiwanStockFinancialStatements`: EPS、稅後淨利 / 淨利
- 近一年 `TaiwanStockInstitutionalInvestorsBuySell`: 外資、投信、自營商 net flow

這一層是 §0.1 第一性原理的資料轉換點：不是直接引用 `M` / `V` / `Delta_lnP` 這種哲學符號，而是用可觀測表格轉成 proxy。

### 3.4 Six-layer scoring: 將三核心思想壓成 CoreScore

每一支股票經 `_score_candidate()` 計算：

```text
DQ, LM, FG, TR, IF, VC, RP
```

其中：

- `DQ` 把資料完整性轉成分數。
- `LM` 把平均日成交金額與 price continuity 轉成流動性質量。
- `FG` 把營收成長與獲利能力轉成基本面重力。
- `TR` 把產業分類對映第六波主題。
- `IF` 把法人資金流轉成資訊力。
- `VC` 把價格穩定性轉成波動控制。
- `RP` 把不合格標的或資料重大缺口轉成懲罰。

這是本程式最核心的三思想轉換段。

### 3.5 Tier assignment: 八二法則轉成治理分層

`_assign_tiers()` 的實際分層邏輯：

1. 先把有 `exclusion_reason` 的股票放入 `quarantine_universe`。
2. 其餘股票依 `core_score` 降序、`theme_score` 次序排序。
3. `theme_score >= 70` 的候選先取 `convex_limit=30` 進入 `convex_universe`。
4. 剩餘高分者取 `core_limit=120` 進入 `core_universe`。
5. 其他合格但未入選者放入 `research_universe`。

此處是 §0.2 八二法則的具體落地：

```text
左尾隔離     -> quarantine_universe
中段低頻觀察 -> research_universe
右尾集中治理 -> core_universe + convex_universe
```

其中 `convex_universe` 是 §0.2 與 §0.3 的交會點：它不是單純高分，而是高分且具第六波主題共振。

---

## 4. 四個 universe 的治理語意

| Tier | 來源邏輯 | 核心思想 | 治理語意 |
|---|---|---|---|
| `core_universe` | 高 CoreScore、非 convex、非 quarantine | §0.1 + §0.2 | 穩定右尾候選，日常資料同步與下游特徵主體。 |
| `convex_universe` | `theme_score >= 70` 且排序靠前 | §0.2 + §0.3 | 高主題共振的上行凸性候選。 |
| `research_universe` | 合格但未進 core/convex | §0.2 長尾邊界 | 不刪除、不日頻投入，保留年度灌溉與未來升遷可能。 |
| `quarantine_universe` | metadata / type / industry / risk 不合格 | §0.2 左尾隔離 + §0.4 | 不讓低品質或非股票標的污染核心治理。 |

---

## 5. 寫入 DB 的治理契約

正式 `--commit` 時寫入順序：

```text
core_universe_policy
  -> core_universe_snapshot
  -> core_universe_membership
  -> core_universe_scores
  -> universe_revision_log
```

這個順序很重要：

- `policy` 保存評分與 eligibility 契約。
- `snapshot` 保存當次 universe 的 as-of 狀態。
- `membership` 保存每支股票的 tier 與下游資格欄位。
- `scores` 保存六層分數與 `score_detail` JSON。
- `revision_log` 保存治理可追溯紀錄。

§6.7 的核心股名單由 committed snapshot 查詢，因此本程式是核心 universe 的唯一入口，不應由其他程式繞過。

---

## 6. 已符合憲章的地方

1. **沒有 hardcoded stock list**
   候選從 `TaiwanStockInfo` 讀取，排序與分層由分數決定。

2. **沒有把哲學公式直接寫成 prediction**
   程式只做 universe governance，不保存 feature values、labels、model outputs、prediction signals。

3. **§0.4 在前，§0.1 / §0.2 / §0.3 在後**
   先做 schema / coverage preflight，再執行 CoreScore。

4. **年度重選 guard 存在**
   正式 commit 受 `_annual_rebalance_guard()` 限制，非年度例外必須留理由。

5. **右尾集中與左尾隔離已落地**
   `core + convex = 150`，`quarantine` 明確隔離。

6. **下游邊界清楚**
   membership 預設 `train_eligible=False`、`predict_eligible=False`、`backtest_eligible=False`、`downstream_ready=False`，避免 universe builder 越權進模型。

---

## 7. 值得後續研究或改善的地方

### 7.1 `ThemeResonance` 目前是字典式產業關鍵字

`THEME_KEYWORDS` 已對齊 MBNRIC 大方向，但仍是靜態 keyword map。後續可研究：

- 是否需要把 MBNRIC 映射抽成 policy table 或 JSON policy。
- 是否要記錄 keyword hit reason。
- 是否要加入 sector exposure summary，避免半導體過度集中只在後驗報告才發現。

### 7.2 `FundamentalGravity` 仍偏簡化

目前使用 revenue YoY、EPS sum、net income positive。這足以作 v0.2 universe governance，但若要更貼近 §0.1 的 `V`，後續可研究：

- 毛利率、營業利益率、ROE、free cash flow proxy。
- 財報 type / origin_name 的標準化風險。
- 不同行業基本面尺度差異。

### 7.3 `InstitutionalFlow` 對大型股天然有利

目前以絕對股數門檻判定外資 / 投信淨買超。這會偏向大股本股票。後續可研究：

- 改成成交量比例或市值比例。
- 以 rolling z-score 或 percentile 取代固定股數門檻。
- 分 TWSE / TPEx 或產業做 normalization。

### 7.4 `VolatilityControl` 是防守型，未完整表達 convexity

目前 CV 越低分數越高，較像穩定性控制；對 §0.2 的上行凸性捕捉仍有限。後續可研究：

- upside volatility / downside volatility 分離。
- max drawdown + rebound strength。
- breakout persistence 或 right-tail return capture。

### 7.5 `convex_universe` 目前依 theme_score 優先

`convex_pool = theme_score >= 70` 並取排序前 30，代表 convex 主要由第六波主題共振決定。這合憲，但後續應研究：

- convex 是否應同時要求更明確的 price / fundamental acceleration。
- convex 與 core 的 realized IC / return 是否顯著不同。
- convex 是否過度集中單一產業。

---

## 8. 逐程式研究的下一步建議

下一支建議研究：

```text
scripts/core/feature_store_builder.py
```

理由：`core_universe_builder.py` 只決定「哪些股票值得進入核心治理」。真正把 §0.1 第一性原理轉成模型可用特徵的是 `feature_store_builder.py`。因此下一步應研究：

1. 它是否只讀取 committed core+convex 150。
2. 它如何產生 liquidity / price / fundamental / institutional / macro / theme features。
3. 它如何處理 as-of cutoff 與 anti-leakage。
4. 它如何標記 missing / imputed / coverage。
5. 它是否忠實承接 §0.0-A 統合框架而沒有 hardcode 哲學公式。

---

## 9. 本研究裁決

`core_universe_builder.py` 是「三核心思想 → 核心股 universe」的正確第一支研究程式。

它的定位不是模型，不是預測，也不是配置；它是 **Core Universe Selection Authority**。它把：

- §0.1 第一性原理轉成 LM / FG / IF 等可觀測治理分數；
- §0.2 八二法則轉成 quarantine / research / core / convex 分層；
- §0.3 康波週期轉成 ThemeResonance；
- §0.4 可觀測性轉成 DataQuality、preflight、coverage 與 revision log；

最後產生 committed universe snapshot，成為後續 Feature Store、Model Trainer、Prediction Engine 的合法起點。
