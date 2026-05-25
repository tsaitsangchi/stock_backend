# §0.1 第一性原理之 DB Field 真實邊界研究報告 — Bottom-up 反推

- **產出日期**: 2026-05-25
- **產出者**: Codex (Opus 4.7, 1M context) session
- **觸發原因**: 用戶要求「從目前系統所有的 table field 資料來討論第一性原理」(對映 §0.1.3-A 之 top-down 揭露,本報告為 bottom-up 反推)
- **憲章版本**: v6.1.0-patch (`reports/系統架構大憲章_v6.1.0.md`)
- **本報告位階**: 揭露性實證報告(非治權變更);沿用 §0.1.3-A 之先例(§14.7-AX 治權元規則:資料層揭露驅動治權升版)
- **HEAD commit**: `2b79872`(v6.1.0-patch 程式落地 — data_schema v2.17 / db_utils v2.48 / dedup migration v0.1)
- **DB 基線**: snapshot `core_universe_20260521_core_universe_policy_v0_2`(5/21,builder v0.2 committed);資料截至 2026-05-21(TaiwanStockPriceAdj.MAX(date)=2026-05-21)

---

## 一、執行摘要

| 元素 | Raw 完整度 | §6 治理層 | §8/§9 預測層 | Gap 性質 | 修補路徑 |
|---|---|---|---|---|---|
| **M** 流動性質量 | ✅ 100%(治權內飽和)| ✅ 100% | ✅ 已實作 | **無** | — |
| **V** 內在價值密度 | ✅ raw 充足(**跨 4 表 ≥ 32 fields**)| ❌ ~30%(v0.3 已加 GM) | ⚠️ 部分對齊 | **動員 gap**(非資料層 gap) | §0.1.3-A Phase C/D(PER + Dividend) |
| **ΔlnP** 對數價位移 | ✅ raw 完整(5 fields × ≥5 變體) | ❌ **0%**(用 CV 對稱壓制凸性) | ✅ §8 + v0.3 upside/downside | **跨層治權 gap** | §9.10(forward reference,尚未起草) |
| **時間單向性** | ✅ 32+ 時間欄位 | ⚠️ L1+L2 滿,L3 空轉,L4 缺 | ✅ §8.5 8 條規則 PERFECT | **vintage + publication-date 雙缺** | L3:builder 啟用 FRED;L4:憲章先行入憲 |

**核心結論**:
1. §0.1 四個 T1 元素在 raw data 層**全部完整**(13 表 / 113 columns / 62 FinStmt types / 6 Institutional names / 4 FRED series 可完整對應);**第一性原理之 gap 不在資料層,在 builder 計算層**
2. 第一性原理之觀測粒度**與 §9.4 第 7 條全系統治權邊界(≥ 日級)結構性對齊**;tick-level / intraday / ms-level 屬永久禁區,**「資料層硬上限」概念不成立**
3. 已揭露兩個治權落差:(a) §6.3 第 8 條 FRED vintage 機制**有條文無實作**(builder 0% 動員 FRED);(b) Publication-date discipline 為**已知缺陷未入憲**(forward reference at L5881)

---

## 二、研究背景與方法論

### 2.1 觸發來源

§0.1.3-A(`first_principles_evidence_chain_gap_20260524.md`,2026-05-24 入憲)以 **top-down** 角度揭露 §0.1 F = f(M, V) × ΔlnP 之治權-實作落差,列出「7+ dead columns」但未做完整盤點。

本報告以 **bottom-up** 角度,從 DB 13 表所有 fields 出發,完整反推第一性原理在資料層的真實邊界,並對接 §0.1.1 T1/T2/T3 分層裁決。

### 2.2 治權邊界(嚴守 §0.1-A 6 條禁令)

本報告嚴守 §0.1-A 之 6 條永久禁令:

1. 不字面寫 F=M×ΔlnP 入 prediction 計算
2. 不實作 IFF Θ
3. 不用 SOC 作 trading trigger
4. 不寫「重力井邊緣 = 訊號觸發」入 §9.1
5. 不以「物理系統」跳過 §8.5 anti-leakage 8 條規則
6. 不用 §0.1 物理隱喻替代 backtest 證據
7. 不寫地緣事件敘事入 §6.3 / §8 / §9.1 / §9.2 計算邏輯(§0.1-A 第 7 條,2026-05-21 入憲)

### 2.3 方法論:三層治權邊界框架(對 M / V / ΔlnP)

對每個 T1 元素,審查三層治權邊界:

| 層 | 邊界 | 對應條文 |
|---|---|---|
| **L1 觀測粒度層** | §9.4 第 7 條(全系統 ≥ 日級,永禁 tick/intraday/ms)| L4849 |
| **L2 §6 核心股治理層** | CoreScore lookback:LM/VC/IF=252 trading days / FG=24m revenue + 8q financial | L2996 / §6.3 第 3-7 條 |
| **L3 §9.1 預測層** | horizon=30 forward log-return(per-stock + confidence + rank)| L4506-4518 |

對時間單向性,新增第四層:

| L4 Anti-leakage 層 | §8.5 8 條規則 + §6.3 第 8 條 FRED vintage + Publication-date(forward ref) | L4160-4171 / L3185 / L5881 |

### 2.4 「Horizon-適配」原則(從憲章三條明文推導)

- §9.1 預測 horizon = **30 個交易日**(L4511)
- §9.4 第 7 條(L4849):**「horizon=30 之治權邊界即排除日內、tick 級、毫秒級訊號」**
- ∴ 觀測粒度只需對齊 horizon 級別;日級資料對 horizon=30 之預測為 **結構性 sufficient**
- 若觀測粒度 < horizon(即 tick / intraday),反而增加微結構雜訊,降低 IC

**含意**:本研究不評估「資料層硬上限」,因為 §9.4 第 7 條已將 < 日級之觀測**明文排除於系統治權之外**。

---

## 三、DB Field 全盤點(2026-05-21 基線)

### 3.1 業務表 10 表 / 113 columns

| 表 (rows) | columns | 第一性對應 |
|---|---|---|
| `TaiwanStockPriceAdj` (10,481,069) | date, stock_id, Trading_Volume, Trading_money, open, max, min, close, spread, Trading_turnover | **M** raw + **ΔlnP** raw |
| `TaiwanStockPrice` (10,485,300) | (同 PriceAdj,未除權息) | M 副本(builder 用 PriceAdj) |
| `TaiwanStockMonthRevenue` (459,383) | date, stock_id, country, revenue, revenue_month, revenue_year, **create_time** | **V**(營收面)+ **publication-date 欄位** |
| `TaiwanStockFinancialStatements` (2,656,263 / **62 unique types**) | date, stock_id, **type**, value, origin_name | **V**(62 科目)|
| `TaiwanStockPER` (7,328,884) | date, stock_id, **PER, PBR, dividend_yield** | **V**(估值面)|
| `TaiwanStockInstitutionalInvestorsBuySell` (24,963,205) | date, stock_id, **name (6 種)**, buy, sell | **F proxy**(法人流)|
| `TaiwanStockMarginPurchaseShortSale` (7,696,032) | 16 cols(Margin / Short 系列)| **F proxy**(融資擁擠度)|
| `TaiwanStockShareholding` (8,353,006) | 13 cols(Foreign / Chinese Investment + Ratio)+ `RecentlyDeclareDate` | **F proxy**(結構性籌碼)|
| `TaiwanStockDividend` (29,262) | 22 cols(現金/股票股利 + 員工配發 + **4 個 date 欄位**)| **V**(長期品質)|
| `FredData` (48,876 / **4 series**) | date, series_id, value, **realtime_start, realtime_end** | **F'**(宏觀環境)+ **vintage 雙欄位** |

### 3.2 治理表 5 表

`core_universe_policy` / `core_universe_snapshot` / `core_universe_membership` / `core_universe_scores` / `universe_revision_log`

### 3.3 Infra 表 2 表

`data_audit_log` / `pipeline_execution_log`(含 timestamp 欄位)

### 3.4 子表細節

**TaiwanStockFinancialStatements 62 unique types(前 16 大占 95% rows)**:
```
EPS / OperatingExpenses / TAX / OperatingIncome / CostOfGoodsSold / GrossProfit /
IncomeFromContinuingOperations / Revenue / PreTaxIncome / TotalConsolidatedProfit /
IncomeAfterTaxes / TotalNonoperatingIncomeAndExpense / OtherComprehensiveIncome /
EquityAttributableToOwnersOfParent (⚠️ mislabel 已揭露 §0.1.3-A.1 = NetIncome value) /
NetIncome (僅至 2019-12-31,後改 IncomeAfterTaxes) / IncomeLossFromDiscontinuedOperation
```
**V 相關 types ≥ 12**(EPS / Revenue / GrossProfit / OperatingIncome / OperatingExpenses / CostOfGoodsSold / IncomeFromContinuingOperations / PreTaxIncome / TotalConsolidatedProfit / IncomeAfterTaxes / IncomeLossFromDiscontinuedOperation / NoncontrollingInterests 等)

**6 Institutional names**:
```
Foreign_Investor (5.6M)         → builder 已用
Investment_Trust (5.6M)         → builder 已用
Dealer_Hedging (4.5M)           → builder 未用
Dealer_self (4.5M)              → builder 未用
Foreign_Dealer_Self (3.6M)      → builder 未用
Dealer (1.1M)                   → builder 未用
```

**4 FRED series + 全歷史**:
```
DFF      26,257 rows  1954-07-01 → 2026-05-20    聯邦基準利率
T10Y2Y   12,490 rows  1976-06-01 → 2026-05-21    殖利率倒掛
VIXCLS    9,190 rows  1990-01-02 → 2026-05-20    波動率指數
UNRATE      939 rows  1948-01-01 → 2026-04-01    失業率
```

---

## 四、四個 T1 元素逐項審查

### 4.1 M 流動性質量

#### 4.1.1 §0.1.1 T1 等級裁決

T1 第一性元素 — 「即時可動員資本;可直接觀測,單位明確(TWD / 股 / 筆)」(L1336)

#### 4.1.2 Raw Field 對應

| 第一性子源 | DB 欄位 | 表 |
|---|---|---|
| 成交金額(主)| `Trading_money` numeric(20,6) | TaiwanStockPriceAdj |
| 成交股數 | `Trading_Volume` | TaiwanStockPriceAdj |
| 成交筆數 | `Trading_turnover` | TaiwanStockPriceAdj |

**完整對應**:3 fields(日級)

**驗證**:`spread` 經實證(2330 / 5/1~5/21 / residual=0.0000)= `close − prev_close`,屬日漲跌價差(ΔP),**不是流動性指標**,應歸 ΔlnP 而非 M。

#### 4.1.3 三層治權邊界對齊

| 治權層 | 邊界 | M 之對應 |
|---|---|---|
| L1 §9.4 第 7 條 | ≥ 日級(永禁 tick/intraday/ms)| ✅ 日級對齊 |
| L2 §6.3 LM 公式 | lookback = 252 trading days,`avg_daily_value` + `log10` value_score + `price_coverage_252d` | ✅ 已實作(`lookback_252` 變數 SSOT)|
| L3 §9.1 預測層 | M 在 feature_store 為 `avg_daily_value_log_*` / `turnover_mean_*` / `zero_volume_ratio_*` | ✅ 已實作(§0.1-C L1546)|

#### 4.1.4 「資料層硬上限」概念之裁決(本研究修正)

之前討論曾出現「FinMind 不提供 tick-level → M 之資料層硬上限」之描述,經憲章驗證:

**§9.4 第 7 條(L4849)明文**:「不為高頻交易服務:**horizon=30 之治權邊界即排除日內、tick 級、毫秒級訊號**;任何聲稱『即時訊號』之模組擴張即違憲。」

**§0.1 主章節 L1310**:Soros 反射理論「屬高頻、非結構性,由 §9.4 第 7 條禁區排除」

**裁決**:
- tick-level 不是「缺資料」,是 §9.4 第 7 條**永久禁區**
- 接 tick data 即觸 §9.4 違憲
- ∴ M 之 raw 完整度 = **100%**(治權內飽和;不存在「上限」可言)
- 日級 = **§9.4 第 7 條全系統治權邊界內之最細允許粒度**
- §6 治理層之評分窗口由 §6.3 LM lookback 252 trading days **獨立約束**,**不**由 horizon=30 推導

#### 4.1.5 落地度純粹裁決

| 維度 | 邊界 | 落地度 |
|---|---|---|
| 觀測粒度 | §9.4 第 7 條 | ✅ 日級 |
| 治理評分窗口 | §6.3 LM(252 trading days) | ✅ 已實作 |
| 預測 feature 對映 | §8.2 feature_store | ✅ 已實作 |
| Raw 物理觀測 | 3 fields | ✅ 100% |

**M 無 gap,無修補路徑可言**。

---

### 4.2 V 內在價值密度

#### 4.2.1 §0.1.3 + §0.1.1 等級裁決

§0.1.3(2026-05-19 入憲)補入 V 為「Fundamental 第四變數」;§0.1.1 等級:
- V 變數本身 = **T1 第一性**(可觀測,L1388)
- V 之計算方式(如 P/V 比率)= **T2 物理啟發類比**
- V 之絕對「真值」= **T3 操作隱喻**(內在價值絕對量化有主觀假設)

#### 4.2.2 Raw Field 對應(完整版,跨 4 表)

| 第一性子源 | 跨表分布 | 欄位數 |
|---|---|---|
| 月營收(top-line) | `TaiwanStockMonthRevenue.revenue`(+ `revenue_month`, `revenue_year`)| 1 直接 + 2 metadata |
| 損益表科目 | `TaiwanStockFinancialStatements`(**62 unique types,V 相關 ≥ 12**)| 12 V types(EPS / GrossProfit / OperatingIncome / OperatingExpenses / CostOfGoodsSold / Revenue / IncomeFromContinuingOperations / PreTaxIncome / TotalConsolidatedProfit / IncomeAfterTaxes / IncomeLossFromDiscontinuedOperation / NoncontrollingInterests)|
| 估值面 | `TaiwanStockPER`(`PER` / `PBR` / `dividend_yield`)| 3 |
| 股利政策(長期品質) | `TaiwanStockDividend`(StockEarningsDistribution / StockStatutorySurplus / CashEarningsDistribution / CashStatutorySurplus / CashIncreaseSubscriptionRate / ParticipateDistributionOfTotalShares / RatioOfEmployeeStockDividend / RatioOfEmployeeStockDividendOfTotal / RemunerationOfDirectorsAndSupervisors / TotalEmployeeCashDividend / TotalEmployeeStockDividend / TotalEmployeeStockDividendAmount / TotalNumberOfCashCapitalIncrease 等)| ≥ 14 V cols |

**完整對應**:**跨 4 表 ≥ 32 fields**(比 §0.1.3-A 估的「7+ dead」高 4 倍)

#### 4.2.3 三層治權邊界對齊

| 治權層 | 邊界 | V 之對應 |
|---|---|---|
| L1 §9.4 第 7 條 | ≥ 日級 | ✅ V 自然粒度(月/季/年)均 ≥ 日級,結構性滿足 |
| L2 §6.3 FG 公式 | lookback = 24 months revenue + 8 quarters financial | ✅ 已實作 |
| L3 §9.1 預測層 | V 在 §8 feature_store 為 `revenue_yoy_*` / `eps_*` / `net_income_*` | ⚠️ 部分實作 |

#### 4.2.4 V 之觀測頻率對應

| Raw 源 | 自然更新頻率 |
|---|---|
| `TaiwanStockMonthRevenue.revenue` | **月**(每月 10 日前公告)|
| `TaiwanStockFinancialStatements`(62 types)| **季**(Q1-Q3 季底後 45 日 / Q4 年底後 60 日)|
| `TaiwanStockPER` | **日**(由 close 與 EPS/equity/dividend 即時換算)|
| `TaiwanStockDividend` | **年**(除權息日)|

#### 4.2.5 V 在預測層之角色裁決

對 horizon=30 預測:
- 30 日內 V 樣本變動次數:Revenue 至多 1 次 / FinStmt 至多 0~1 次 / Dividend 大多 0 次
- ∴ V 在 §9.1 預測之角色為 **「level info」**(price/V 之均衡水平),非「change info」(ΔV/Δt)
- 對應 §0.1 重力井模型之 V 中心(§0.1.3)

#### 4.2.6 落地度純粹裁決

| 維度 | 邊界 | 落地度 |
|---|---|---|
| 觀測粒度 | §9.4 第 7 條 | ✅ 完美對齊(月/季/年均 ≥ 日級)|
| 治理評分窗口 | §6.3 FG(24m + 8q)| ✅ 已實作 |
| Raw 物理觀測 | 跨 4 表 ≥ 32 fields | ✅ **raw 充足** |
| **Builder 動員度** | §6.3 第 4 條 | ❌ **~30%**(v0.3 已加 GM;真實 gap)|
| 預測 feature 對映 | §8.2 feature_store | ⚠️ 部分對齊 |

#### 4.2.7 修補路徑(§0.1.3-A 預備之 Phase B-E)

| Phase | 內容 | 狀態 |
|---|---|---|
| **B** | GrossProfit / Revenue(毛利率) | ✅ 已完成(v0.3,2026-05-24) |
| **C** | `TaiwanStockPER` 整張(PER / PBR / dividend_yield) | 待研究 |
| **D** | `TaiwanStockDividend` 啟用(股利政策穩定性) | 待研究 |
| **E** | ΔlnP 凸性 sub-score(屬 ΔlnP 修補,非 V) | 待 §9.10 |

---

### 4.3 ΔlnP 對數價位移

#### 4.3.1 §0.1.1 T1 等級裁決

T1 第一性元素 — 「時間軸不可逆;對數變換可加性;單位無因次」(L1337)

#### 4.3.2 Raw Field 對應

| 第一性子源 | DB 欄位 | 衍生變體 |
|---|---|---|
| 收盤價 | `close` | `ln(C_t/C_{t-1})` close-to-close log-return |
| 開盤價 | `open` | `ln(C_t/O_t)` 日內 log-return |
| 最高價 | `max` | `ln(H_t/L_t)` high-low range(Parkinson)|
| 最低價 | `min` | (與 max 對組) |
| 漲跌價差 | `spread`(= `close − prev_close`) | ΔP 線性近似 |

**衍生變體 ≥ 5 種**:
- close-to-close log-return `ln(C_t/C_{t-1})`(基本 ΔlnP)
- intraday log-return `ln(C_t/O_t)`(日內動能)
- overnight gap `ln(O_t/C_{t-1})`(隔夜跳空 — 資訊衝擊 proxy)
- high-low range `ln(H_t/L_t)`(Parkinson volatility 基礎)
- Garman-Klass / Rogers-Satchell volatility(用全 OHLC 高效估計)

**完整對應**:5 fields(PriceAdj 日級)+ ≥5 變體

#### 4.3.3 三層治權邊界對齊(ΔlnP 之角色跨層大不同)

| 治權層 | ΔlnP 之角色 | 對應條文 |
|---|---|---|
| L3 §9.1 預測層 | **預測標的本身** — `log(P_{t+30}/P_t)` 即 ΔlnP × 30 | §9.1 表「預測形式」L4512 |
| L2 §8 Feature Store | 預測特徵 — `log_return_*d` / `volatility_*` / upside/downside 分離 | §8.2 / §9.9 |
| L2 §6 核心股治理層 | **應作為「凸性」觀測載體**(對應 §0.2 槓鈴右尾)| §6.3 VC(5%)|

#### 4.3.4 §6 治理層之 ΔlnP 失血(0% 落地)

§6.3 第 7 條(VC 公式):
```
cv_close = STDDEV(close) / AVG(close)
```
為傳統 Coefficient of Variation,**對稱壓制凸性**;`log_return` 在 `core_universe_builder.py` 全檔 grep 結果為 zero(§0.1.3-A L1427)。

對比:

| 對比 | 實作 CV | §0.1 要求之 ΔlnP |
|---|---|---|
| 公式 | `STDDEV(price)/MEAN(price)` | `ln(P_t)−ln(P_{t-1})` |
| 上下行對稱性 | **對稱**(壓制凸性) | **不對稱**(可分 upside / downside) |
| 凸性表達 | 無 | 有(上行無限 / 下行有界) |
| 與 §0.2 槓鈴相容 | **衝突**(壓制右尾) | **相容**(攻擊端凸性) |

#### 4.3.5 §9.9 已落地之第一階段修補(2026-05-20 入憲)

`feature_store_builder.py v0.3` 加入 4 個 upside/downside 分離 features(`feature_set_v0.3`,31 features):
- `upside_volatility_60d`
- `downside_volatility_60d`
- `upside_capture_60d`
- `downside_capture_60d`

**§9.9 範圍裁決明文**(L4905):
> 不立即修改 core_universe_builder.py VolatilityControl(避免 universe 重組之 destructive change);待新特徵 ablation IC > 0 實證後,再於 **§9.10 起草** VolatilityControl 升版契約。

#### 4.3.6 §9.10 之 forward reference 狀態(實際驗證)

| 證據 | 內容 |
|---|---|
| §9 章節最後一節 | **§9.9-I**(L5010)— 無 §9.10 條文 |
| §9.10 出現處 | L4905 + L5019(皆「**待 §9.10 起草時**」未來式)|
| **§9.10 治權狀態** | ❌ **尚未起草,屬 forward reference / 預備占位** |

#### 4.3.7 觸發鏈

```
Feature Store v0.3 (4 upside/downside features) [已落地]
         ↓
ablation IC > 0 實證 [待 §0.0-E.6 P1 v0.2 runbook 跑出]
         ↓
§9.10 起草 (VolatilityControl 升版契約) [尚未起草]
         ↓
core_universe_builder VolatilityControl v0.4 升版 [尚未實作]
         ↓
§6 ΔlnP 落地度 0% → > 0%
```

#### 4.3.8 落地度純粹裁決

| 維度 | 邊界 | 落地度 |
|---|---|---|
| 觀測粒度 | §9.4 第 7 條 | ✅ 日級對齊 |
| Raw 物理觀測 | 5 fields × ≥ 5 變體 | ✅ **raw 完整** |
| **§6 治理層動員度** | §6.3 VC 第 7 條 | ❌ **0%**(用 CV 非 log-return)|
| §8 Feature Store 動員度 | §0.1-C L1547 | ✅ 已實作 |
| §9.9 upside/downside 分離 | feature_set_v0.3 | ✅ 已落地(Feature Store 層)|
| §9.1 預測標的對齊 | horizon=30 forward log-return | ✅ 直接定義 |

#### 4.3.9 跨層治權張力

| 跨層比較 | §6 治理層 | §8 Feature Store | §9.1 預測層 |
|---|---|---|---|
| ΔlnP 公式 | `STDDEV/MEAN`(對稱)| `ln(C_t/C_{t-1})`(對稱)+ upside/downside(不對稱 v0.3)| `log(P_{t+30}/P_t)`(forward)|
| 上下行對稱性 | **對稱**(壓制凸性)| 對稱 + 不對稱並存 | 不對稱(forward)|
| 與 §0.2 槓鈴相容 | ❌ 衝突 | ⚠️ 部分(v0.3 補)| ✅ 相容 |

**ΔlnP 在 §8/§9.1 已 100% 落地,但在 §6 universe 治理層卻 0% 落地** — 同一物理量,跨層治權實作天差地遠。屬 §0.0-C.3「上行凸性系統性壓制」之 root cause 之一。

---

### 4.4 時間單向性(含 vintage + publication-date)

#### 4.4.1 §0.1.1 T1 等級裁決

T1 第一性元素 — 「物理常數;不可違反;對應 as-of-strict / §8.5 anti-leakage」(L1338)

#### 4.4.2 Raw Field 對應(完整版)

**業務表時間欄位 17 個**:
```
FredData                                  date, realtime_start, realtime_end   (3)
TaiwanStockDividend                       date, AnnouncementDate, AnnouncementTime,
                                          CashDividendPaymentDate, CashExDividendTradingDate,
                                          StockExDividendTradingDate                       (6)
TaiwanStockFinancialStatements            date                                              (1)
TaiwanStockInfo                           date                                              (1)
TaiwanStockInstitutionalInvestorsBuySell  date                                              (1)
TaiwanStockMarginPurchaseShortSale        date                                              (1)
TaiwanStockMonthRevenue                   date, create_time                                 (2)
TaiwanStockPER                            date                                              (1)
TaiwanStockPrice / PriceAdj               date (×2)                                         (2)
TaiwanStockShareholding                   date, RecentlyDeclareDate                         (2)
```

**治理表 + Infra 時間欄位 15+**(包含 `selected_at` / `effective_from/to` / `as_of_date` / `source_data_cutoff` / `created_at/updated_at` / `revision_time` / `data_audit_log.timestamp` / `pipeline_execution_log.start_time/end_time`)

**完整對應**:**業務表 17 + 治理 15 = 32+ 時間欄位**

**含三個第一性 anti-leakage 關鍵欄位**:
- **vintage 雙欄位**:`FredData.realtime_start / realtime_end`(2026-05-21 當下能取得的 2025-Q4 GDP 數字 vs 2024-Q4 GDP 數字之 vintage 標籤)
- **publication-date 欄位**:`TaiwanStockMonthRevenue.create_time`(資料發布日 vs 統計日)
- **announcement-date 欄位**:`TaiwanStockDividend` × 4 dates + `TaiwanStockShareholding.RecentlyDeclareDate`

#### 4.4.3 四層治權邊界對齊

| 治權層 | 邊界 | 條文位置 | 落地度 |
|---|---|---|---|
| **L1 §9.4 第 7 條** | ≥ 日級 | L4849 | ✅ 100%(日級時間戳已用)|
| **L2 §8.5 anti-leakage 8 條規則** | as-of-strict / label horizon / universe 鎖定 / feature_set 鎖定 / no hot-fix / no future split / 零硬編預測 / 單一 SSOT | L4160-4171 | ✅ 已實作(`audit_leakage.py` PERFECT) |
| **L3 §6.3 FRED vintage** | `as_of_date 可取得的 realtime_start/end 版本為準,避免使用未來才修正完成的宏觀資料` | L3185 | ❌ **空轉** — 條文已入憲,但 §6 builder 完全未用 FRED(動員度 0%) |
| **L4 Publication-date / filing lag** | 「目前以 `date <= as_of_date` 為資料可得邊界;**後續應研究公告日 / create_time / filing lag rule**」 | L5881 | ❌ **未入憲** — forward reference;builder 仍用 statistical date |

#### 4.4.4 §8.5 8 條規則(已實際入憲驗證)

| 規則 | 適用層 | 強制執行載體 |
|---|---|---|
| as-of-strict filter | Feature / Model / Prediction | `audit_leakage.py` |
| label horizon 後置 | Model | trainer 必須驗證 `label_date >= as_of_date + label_horizon` |
| universe snapshot 鎖定 | All | 三層 DDL 之 `universe_snapshot_id` 必須一致 |
| feature_set 鎖定 | Model / Prediction | `model_registry.feature_set_id` = `prediction_run.feature_set_id` |
| No hot-fix imputation | Feature | `feature_definition.null_strategy` 在 build 時鎖定 |
| No future split | Feature | 標準化參數鎖在 `feature_definition` 或 train artifact |
| 零硬編預測 | Prediction | `audit_supply_chain.py` 延伸掃描 |
| 單一 SSOT(§6.7 延伸) | All | 必須走 `db_utils.get_core_stocks_from_db()` |

**8 條規則皆涵蓋 universe / feature / model / prediction 流之 time discipline,未涵蓋 raw data publication delay 之 vintage discipline**。

#### 4.4.5 兩個關鍵發現(時間單向性特有)

**發現 L3 — Vintage 機制空轉(條文活實作死)**:
- §6.3 第 8 條(L3185)明文要求 FRED 使用 `as_of_date 可取得的 realtime_start/end 版本為準`
- 但 `core_universe_builder.py` 完全未 SELECT FRED(§6 盤點動員度 0%)
- ∴ vintage 機制有條文但完全未落地

**發現 L4 — Publication-date leakage 已知缺陷未升強制契約**:
- L5881 明文:「目前以 `date <= as_of_date` 為資料可得邊界;**後續應研究公告日 / create_time / filing lag rule**」
- §14.7-W feature_store_builder 研究亦列「**財報/月營收公告日語意**」為「**後續待研究項**」
- 即:builder 用 `MonthRevenue.date=2024-03-31`(統計月底)作 as_of_date 邊界
- 但實際公告日 `create_time` 可能是 2024-04-10(統計月後 10 天)
- 若 `as_of_date=2024-04-05`,builder 會誤把 2024-03 revenue 納入(實際公告日 2024-04-10,還未發布)
- → 理論上有 **~10 天的 publication-date leakage**
- 此屬「**已知缺陷 + 憲章僅承諾未來研究**」,**未升至 §8.5 強制契約**

---

## 五、四元素審查總結比較表

| 維度 | M | V | ΔlnP | 時間單向性 |
|---|---|---|---|---|
| §0.1.1 等級 | T1 | T1(本身)+ T2(計算)+ T3(絕對值)| T1 | T1 |
| Raw 欄位數 | 3(治權內飽和)| 跨 4 表 ≥ 32 | 5 + ≥5 變體 | 32+ 跨業務 + 治理 |
| Raw 完整度 | ✅ 100% | ✅ 充足 | ✅ 完整 | ✅ 完整 + 含 vintage |
| 治權層數 | 3 | 3 | 3 | **4** |
| §6 治理層動員 | ✅ 100% | ❌ ~30% | ❌ 0%(CV)| ⚠️ L1+L2 滿 / L3 空轉 / L4 缺 |
| §8/§9 預測層動員 | ✅ 已實作 | ⚠️ 部分 | ✅ + v0.3 upside/downside | ✅ §8.5 PERFECT |
| Gap 性質 | 無 gap | 動員 gap | 跨層治權 gap | vintage + publication 雙缺 |
| 修補狀態 | — | Phase B 已完成 / C/D 待研究 | §9.9 已落第一階段 / §9.10 forward ref | L3:builder 啟用 FRED;L4:憲章先行 |
| 與 §9.1 horizon 親緣性 | feature 之一 | feature 之一(level info)| **預測標的本身** | as-of-strict + label gate |

---

## 六、八大關鍵發現

### 發現 1 — 第一性 gap 不在資料層,在 builder 計算層
系統有 raw data 完整對應 §0.1 全部 T1 元素(M / V / ΔlnP / 時間)+ 五源 F proxy + 一源 F'(FRED 宏觀)。13 表 / 113 columns + 62 FinStmt types + 6 Institutional names + 4 FRED series 之動員度僅 ~20%(builder 直接使用 23/113 columns)。**修補路徑無需新 sync,純讀現有 DB**。

### 發現 2 — TaiwanStockPER 整張 dead 是 V 最大失血點
7,328,884 rows / 2016 stocks / 3 個直接 V 指標(PER / PBR / dividend_yield)— builder 從未 SELECT。§0.1.3-A 之 V 落地度 ~30%(v0.3 +GM)可推進至 ~60%(若 +PER/PBR/yield)。

### 發現 3 — F proxy 之多源覆蓋僅 ~20%
§0.1 之 F(資訊力)在現行 CoreScore 只透過 InstitutionalFlow 10% 體現,且只用 2/6 names(Foreign + Trust)。Margin (16 cols dead) / Shareholding (13 cols dead) / Dealer 系列 (4 names dead) — 共 33 fields + 4 names = **多源 F proxy 約 80% 失血**。

### 發現 4 — 第七維「宏觀 F'」結構性缺失
FRED 4 series 在 §6.4 CoreScore 完全無對應 sub-score。§0.1 主章節「F 包含宏觀數據」但落地時 v0.2/v0.3 為純微觀股級評分。屬 §0.1 + §0.3 K-wave 治權交界,需另立提案。

### 發現 5 — Vintage 機制空轉(已入憲 vs 未落地)
§6.3 第 8 條(L3185)明文 FRED vintage 對齊要求,但 builder 完全未用 FRED → 條文活實作死。

### 發現 6 — Publication-date leakage 未入強制契約
L5881 明文「後續應研究公告日 / `create_time` / filing lag rule」— 屬 forward reference;builder 仍用 statistical date,理論上有 ~10 天 leakage。§14.7-W 亦列為「後續待研究項」。

### 發現 7 — ΔlnP 跨層治權實作天差地遠
同一物理量在 §8/§9.1 已 100% 落地(預測標的本身),在 §6 universe 治理層卻 0% 落地(用 CV 對稱壓制)。§9.10 升版契約尚未起草(forward reference)。

### 發現 8 — 「資料層硬上限」概念於本系統不成立
§9.4 第 7 條(L4849)明文排除日內 / tick / ms 訊號;tick-level 不是「缺資料」是「**永久禁區**」。M 之 3 fields 對 horizon=30 預測為**結構性 sufficient**,不存在「上限」可言。

---

## 七、修補路徑彙整(治權邊界內)

### 7.1 純動員修補(無需憲章變更,無治權風險)

| Phase | 內容 | 對應元素 | 風險 |
|---|---|---|---|
| **C** | `TaiwanStockPER` 整張啟用(PER / PBR / dividend_yield)| V | 低 |
| **D** | `TaiwanStockDividend` 啟用(股利政策穩定性) | V | 低 |
| **F** | InstitutionalFlow 補入 Dealer 4 names(現只用 2/6) | F proxy | 低 |

### 7.2 條文活實作死之修補(需 builder 升版,治權邊界內)

| 動作 | 內容 | 對應發現 |
|---|---|---|
| **L3 vintage 落地** | builder 啟用 FRED 4 series,使用 `realtime_start/end` 對齊 | 發現 5 |
| **F'(宏觀)補入 CoreScore** | 第七維 sub-score(FRED) | 發現 4 |

### 7.3 需憲章先行入憲(治權升版)

| 提案 | 內容 | 預估治權位階 |
|---|---|---|
| **§8.5-9 Publication-date Discipline** | 把 `create_time` filing lag rule 升至 §8.5 強制契約第 9 條 | 發現 6 |
| **§9.10 VolatilityControl 升版契約** | 待 ablation IC > 0 後起草 ΔlnP 凸性 sub-score(§6 治理層)| 發現 7 |
| **§6.4 第七維 sub-score 對映** | 宏觀 F' 進入 CoreScore(若 ablation 證明有效) | 發現 4 |

### 7.4 高風險路徑(慎評)

| 動作 | 風險 |
|---|---|
| ΔlnP 凸性 sub-score 直接寫入 §9.1 prediction(z-score > 2 加權) | §0.1-D 第 3 條「未來可選研究」,但仍須通過 §9.4 七條治權邊界檢驗 |
| 把優先依附 $\Pi(i,t)$ 寫成 prediction 分數 | §0.1-A 禁令 #1 / §0.0-E.4 — **永久禁止** |

---

## 八、後續研究方向

| 編號 | 方向 | 對應憲章節 |
|---|---|---|
| **D1** | FinStmt 62 types 之完整 V/非V 分類(逐 type 對齊會計學分類)| §6.3 第 4 條 FG / §0.1.3 |
| **D2** | Dividend 22 cols 之逐欄位語意確認(哪些 V / 哪些 governance metadata)| §6.3 第 4 條 FG |
| **D3** | 驗證 MonthRevenue `create_time` 是否在 builder 中作為 leakage gate(目前已知未用) | §8.5 第 1 條 / L5881 |
| **D4** | 驗證 FRED vintage 4 series 之 `realtime_start` 分布(若都 = date 則 vintage 機制無效) | §6.3 第 8 條 / L3185 |
| **D5** | 從 5/21 snapshot 之 120 核心股,實證對照「若使用全部 dead fields 重排,結果會如何」(simulation,不 commit)| §6.4 |
| **D6** | 第七維宏觀 F'(FRED)之治權邊界研究(§0.1 + §0.3 交界,可能涉及憲章先行入憲) | §0.1 / §0.3 / §6.4 |

---

## 九、附錄

### 附錄 A — 憲章條文行號 cross-ref(已實際驗證)

| 引用條文 | 實際入憲狀態 | 行號 |
|---|---|---|
| §0.1.1 T1/T2/T3 分層 | ✅ 已入憲 | L1330-1351 |
| §0.1.3 Fundamental 第四變數 V | ✅ 已入憲(2026-05-19) | L1352-1404 |
| §0.1.3-A V 與 ΔlnP 工程落地實況揭露 | ✅ 已入憲(2026-05-24) | L1405-1460 |
| §0.1-A 6 條禁令(+ #7 地緣) | ✅ 已入憲 | L1492-1510 |
| §6.0 核心股治理三層分工 | ✅ 已入憲 | L2977-2989 |
| §6.3 CoreScore lookback 顯式化 | ✅ 已入憲(2026-05-24,commit `a4fa6f2`)| L2996 |
| §6.3 第 3-7 條 DQ/LM/FG/TR/IF/VC 公式 | ✅ 已入憲 | L3101-3145 |
| §6.3 第 8 條 FRED vintage 對齊 | ✅ 已入憲 | L3185 |
| §6.7 SQL 契約(150 SSOT) | ✅ 已入憲 | L3246-3276 |
| §8.5 anti-leakage 8 條規則 | ✅ 已入憲 | L4160-4171 |
| §9.1 30-day prediction contract | ✅ 已入憲 | L4506-4518 |
| §9.4 第 7 條(高頻禁區 + horizon=30 排除 tick)| ✅ 已入憲(永久強制,2026-05-17 起)| L4849 |
| §9.9 Upside/Downside Volatility Decomposition v0.1 | ✅ 已入憲(2026-05-20) | L4899-5019 |
| §9.10 VolatilityControl 升版契約 | ❌ **forward reference / 預備占位** | L4905 / L5019(僅 2 處引用,皆未來式)|
| **Publication-date / filing lag rule** | ❌ **未入憲**(forward reference) | L5881(明文「後續應研究」)|

### 附錄 B — DB 全表 row count 基線(2026-05-25 驗證)

```
TaiwanStockInfo                              2,799
TaiwanStockPriceAdj                     10,481,069
TaiwanStockPrice                        10,485,300
TaiwanStockPER                           7,328,884
TaiwanStockInstitutionalInvestorsBuySell 24,963,205
TaiwanStockMarginPurchaseShortSale       7,696,032
TaiwanStockFinancialStatements           2,656,263 (62 unique types)
TaiwanStockShareholding                  8,353,006
TaiwanStockMonthRevenue                    459,383
TaiwanStockDividend                         29,262
FredData                                    48,876 (4 series, all-history)

業務表合計                              72,701,279
```

### 附錄 C — 與 §0.1.3-A 之關係

| 對比 | §0.1.3-A(2026-05-24)| 本報告(2026-05-25)|
|---|---|---|
| 方法論 | top-down(從 §0.1 文字回推實作)| **bottom-up**(從 DB field 反推治權)|
| V dead columns 估計 | 7+ | **≥ 32**(跨 4 表完整盤點)|
| ΔlnP 落地度估計 | ~0% | **0%** 確認(grep `log_return / np.log / math.log(` zero)|
| 時間單向性審查 | 未涵蓋 | **新增**:vintage 空轉 + publication-date 雙缺 |
| 治權層數 | 未明文分層 | **明文 3/4 層**(M/V/ΔlnP 3 層 + 時間 4 層)|
| 修補路徑 | Phase B-E(列 5) | **整理為 7.1-7.3 三類**(純動員 / 條文活實作死 / 憲章先行)|

本報告之地位:**§0.1.3-A 之 bottom-up 補完**(同方向,雙視角)。

---

## 十、治權聲明

### 10.1 嚴守 §0.1-A 6 條禁令

本報告之所有結論皆嚴守 §0.1-A 治權邊界:
- 不字面寫 F=M×ΔlnP 入計算邏輯
- 不實作 IFF Θ
- 不用 SOC 作 trading trigger
- 不寫「重力井邊緣 = 訊號觸發」
- 不以「物理系統」為由跳過 §8.5
- 不用 §0.1 物理隱喻替代 backtest 證據
- 不寫地緣事件敘事入計算邏輯

### 10.2 嚴守 §0.0-G 憲章先行紀律

本報告為**揭露性實證報告,非治權變更**;不修改:
- §6.3 / §6.4 強制契約之既有條文
- §8.5 anti-leakage 8 條規則
- §9.1 / §9.2 預測 + 配置契約
- §0.1 / §0.1.1 / §0.1.3 / §0.1.3-A 既有文字
- 任何 raw DDL
- `core_universe_builder.py` / `feature_store_builder.py` / `prediction_engine.py` / `portfolio_sizer.py` 任何程式邏輯

若本報告結論觸發升版需求,須依 §0.0-G Level 1 流程另行起草強制契約(對映本報告 §7.3)。

### 10.3 與 §0.1.3-A 之關係

本報告**延伸**§0.1.3-A 之揭露(2026-05-24,commit 含 `1df1f5e` / `c33cf8b`);採同向 bottom-up 方法論完整盤點,**不取代**§0.1.3-A 原文,**不修改**Phase B v0.3 落地實況(commit `c33cf8b`)。

### 10.4 §14.7-AX 治權元規則對齊

本報告再次驗證「**資料層揭露驅動治權升版**」之機制(§14.7-AX,2026-05-24 入憲):
- bottom-up field 盤點 → 揭露 4 個發現(發現 4-7)觸發 §6.3 / §8.5 / §9.10 / §6.4 升版需求
- 對映 §0.0-G 憲章先行紀律之 Level 1 流程

---

## 十一、後續接續點

| 條件 | 動作 |
|---|---|
| 用戶確認本報告為 §0.1.3-A 之 bottom-up 補完 | 入憲為 §0.1.3-B(或 §14.7-AZ 子節) |
| 用戶要求展開 §7.1 純動員修補(Phase C/D/F) | 進入 `core_universe_builder.py v0.4` 設計研究 |
| 用戶要求展開 §7.3 憲章先行入憲 | 起草 §8.5-9 / §9.10 / §6.4 第七維對應強制契約草案 |
| 用戶要求展開 D5 simulation | 寫 simulation 腳本對照 5/21 snapshot |
