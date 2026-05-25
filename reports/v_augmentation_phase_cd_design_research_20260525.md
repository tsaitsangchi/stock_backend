# V 補強 Phase C/D + FinStmt — FG v0.3 → v0.5 設計研究報告

- **產出日期**: 2026-05-25
- **產出者**: Codex (Opus 4.7, 1M context) session
- **觸發**: 用戶 2026-05-25「以第一性原理與目前系統資料庫的對應」之回歸 + 選 §0.1.3-B.7 A 類「純動員修補」之 Phase C/D 完整(對映 §14.7-AX 治權元規則第四次跑通預備)
- **位階**: **草案性設計提案**(治權先行;依 §0.0-G 紀律入憲 §14.7-BC 後實作 core_universe_builder v0.5)
- **HEAD commit**: `f3a1dc4`(Phase 3 + helper SSOT 統一)
- **配套憲章節**: §0.1.3 / §0.1.3-A / §0.1.3-A.1(ROE 不可實作)/ §0.1.3-B / §6.3 第 4 條 FG / §6.4 治理欄位 / §14.7-W(feature_store research)/ §14.7-AX(資料層揭露驅動治權升版)

---

## 一、執行摘要

| 項目 | 內容 |
|---|---|
| **動機** | §0.1.3-B 揭露 V 動員 gap(實證後修正為 5/22 = 23% — 比原估 30% 低);TaiwanStockPER 整張 dead(7.3M rows / 2016 stocks)+ TaiwanStockDividend 整張 dead(29K rows / 14 V cols)+ FinStmt 4 V types dead |
| **目標** | FG sub-scores 從 v0.3 之 5 個擴至 v0.5 之 **11 個**;V 動員度從 23% 提升至 **77%**(17/22);FG 權重 20% **維持不變**(內部結構升;不打破 §6.4 公式總結構) |
| **新增 6 個 sub-scores** | PER 估值(industry-relative)/ PBR 估值(industry-relative)/ Dividend yield / 配息穩定性 / Operating Margin / Attributable Ratio |
| **政策升版** | `core_universe_policy_v0.3` → `core_universe_policy_v0.4`;`core_universe_builder v0.4 → v0.5` |
| **追溯適用** | 既有 v0.3 snapshot 不重 build;新 v0.4 snapshot 起適用 |
| **「資料現實裁決」第四次跑通預備** | dry-run 後若揭露 PER outlier / Dividend year 語意 / industry 分類粒度 等資料現實問題,觸發 §14.7-BC 後續追溯修正(類比 §14.7-BB 之模式) |

---

## 二、研究背景與動機

### 2.1 §0.1.3 + §0.1.3-A + §0.1.3-B 之 V 治權三層揭露鏈

| 層 | 子節 | 內容 |
|---|---|---|
| 1 | **§0.1.3**(2026-05-19 入憲)| Fundamental 第四變數補強:F = f(M, V) × ΔlnP;V = 內在價值密度;**「V 之 raw 充足」治權聲稱** |
| 2 | **§0.1.3-A**(2026-05-24 入憲)| top-down 揭露 V 落地度 ~15%(v0.2)→ ~30%(v0.3 +GM);ROE 不可實作 = §0.1.3-A.1 |
| 3 | **§0.1.3-B**(2026-05-25 入憲)| bottom-up 完整盤點:跨 4 表 ≥ 32 fields raw 充足;builder 動員 ~30%(實證修正為 23%);8 大發現 |
| **4(本研究)** | **§14.7-BC**(預備入憲)| **V 補強 Phase C/D + FinStmt 落地設計**:從 23% 補完至 77%;對映 §0.1.3-B.7 A 類「純動員修補」 |

### 2.2 §14.7-AX 治權元規則之第四次跑通預備

| 次 | 時點 | 揭露 | 修正 |
|---|---|---|---|
| 1 | 2026-05-24(§0.1.3-A.1)| ROE EquityAttributableToOwnersOfParent mislabel | ROE dropped |
| 2 | 2026-05-25 §14.7-BA | 5 個 publication-date 候選欄位 3 不可用 | 一刀切 → 分層落地 |
| 3 | 2026-05-25 §14.7-BB | FRED vintage strict gate 100% loss | strict → transitional |
| **4(預備)** | **2026-05-25 §14.7-BC**(本研究 dry-run 後可能觸發)| **若 PER outlier / Dividend year 語意 / industry 分類粒度 等資料現實揭露** | **預備 V 補強之分層落地策略 + 治權追溯** |

---

## 三、V Raw Field 之 DB 實證(2026-05-25)

### 3.1 TaiwanStockPER(7.3M rows / 2016 stocks)— ✅ 完全可用

| col | 全市場非空 | core 150 valid | 全市場 avg | core 150 latest avg | 風險 |
|---|---|---|---|---|---|
| **PER** | 100% | **149/150** | 30.11 | 168.14 | ⚠️ outlier max=14,700(需 winsorize)|
| **PBR** | 100% | **150/150** | 2.10 | 10.28 | industry 差異大(通信 34 vs 電腦 6)|
| **dividend_yield** | 100% | **146/150** | 3.63 | 1.77 | 低 |

**PER 全市場分布**(n=1982,排 NULL/0):
```
min 1.11 / p5 8.70 / p25 15.50 / median 27.07 / p75 66.17 / p95 443.45 / p99 1766.90 / max 14,700
```

**PER core 150 分布**:
```
min 6.82 / median 38.98 / p95 190.32 / max 14,700
PER > 100: 23 stocks / PER > 50: 58 stocks(58/150 = 39%)
```

**core 150 PBR by industry**(top 7):

| industry | n | avg PBR | min | max |
|---|---|---|---|---|
| 通信網路業 | 4 | **34.77** | 20.43 | 55.76 |
| 半導體業 | 34 | 14.03 | 1.56 | 104.26 |
| 電機機械 | 10 | 10.49 | 1.98 | 30.67 |
| 其他電子類 | 5 | 9.20 | 5.11 | 19.68 |
| 電子零組件業 | 25 | 8.82 | 1.05 | 35.05 |
| 電子工業 | 50 | 7.90 | 0.78 | 52.75 |
| 電腦及週邊設備業 | 16 | 6.64 | 1.67 | 21.68 |

**裁決**:PBR 跨產業差異 ~5×(通信 34 vs 電腦 6);**純 PBR 評分會偏向半導體股**,需 industry-relative 化解。

### 3.2 TaiwanStockDividend(29K rows / 22 cols)— 5 V cols 高度可用

**核心 V 訊號(≥ 30% 覆蓋)**:

| col | 覆蓋率 | V 意義 | core 150 (2024-26) |
|---|---|---|---|
| **CashEarningsDistribution** | **87.5%** | 現金股利(主訊號)| 147/150 / avg 6.29 元 / 320 events |
| ParticipateDistributionOfTotalShares | 42.4% | 參與分配股數 | — |
| RemunerationOfDirectorsAndSupervisors | 34.0% | 董監酬勞 | — |
| TotalEmployeeCashDividend | 31.5% | 員工現金股利 | — |
| StockEarningsDistribution | 21.1% | 股票股利 | — |

**Dividend 多年覆蓋之 stock 配息穩定性**(待 dry-run 驗證):
- 過去 5 年配息次數 5 次 = 穩定配息股(高品質)
- 過去 5 年配息 0-2 次 = 不穩定 / 新上市

### 3.3 TaiwanStockFinancialStatements V types — 13 真實可用 / 4 新加入

| type | core 150 latest | avg value | 可用性 | v0.3 → v0.5 |
|---|---|---|---|---|
| EPS | 2342 stocks | — | ✅ | v0.3 已用 |
| Revenue | 2323 | — | ✅ | v0.3 已用 |
| GrossProfit | 2331 | — | ✅ | v0.3 已用(latest_margin) |
| IncomeAfterTaxes | 2326 | — | ⚠️ | v0.3 部分用(via origin_name) |
| **OperatingIncome** | **150/150** | 81.7 億 | ✅ | **v0.5 加** |
| **PreTaxIncome** | **150/150** | 85.8 億 | ✅ | **v0.5 加** |
| **IncomeFromContinuingOperations** | **150/150** | 70.4 億 | ✅ | **v0.5 加** |
| **NoncontrollingInterests** | **131/150**(87%)| 2.27 億 | ✅ | **v0.5 加**(計算 Attributable Ratio)|
| ~~TotalConsolidatedProfit~~ | 0 stocks | — | ❌ | API 不回傳 |
| ~~NetIncome~~ | 1776 | 2019-12-31 | legacy | 2020+ 改 IncomeAfterTaxes |
| ~~EquityAttributableToOwnersOfParent~~ | 2143 | — | ❌ | mislabel(§0.1.3-A.1)→ ROE dropped |

---

## 四、FG v0.5 設計 — 6 個新 sub-scores

### 4.1 PER 估值(industry-relative)— ±20 範圍

```
rel_per = stock_PER / median(industry_PER)
        其中 industry_PER 為該 industry_category 之 core 150 stock latest PER 中位數

rel_per < 0.7         → +10  (顯著低估)
rel_per < 1.0         → +5   (合理偏低)
1.0 <= rel_per <= 1.5 → 0    (合理區間)
1.5 < rel_per <= 2.0  → -5   (明顯高估)
rel_per > 2.0         → -10  (極高估或泡沫)

PER < 0(虧損)         → -5 + risk_penalty[低品質扣分]
PER > p99=1766         → -5 + risk_penalty
```

**設計依據**:
- core 150 中 39% (58 stocks) PER > 50,絕對 PER cap 會誤殺
- industry-relative 化解跨產業可比性問題
- p99 cap 處理極端 outlier(全市場 max=14,700)

### 4.2 PBR 估值(industry-relative)— ±15 範圍

```
rel_pbr = stock_PBR / median(industry_PBR)

rel_pbr < 0.8         → +5   (低 PBR;可能 deep value)
0.8 <= rel_pbr <= 1.5 → 0    (合理)
1.5 < rel_pbr <= 2.5  → -3   (偏高)
rel_pbr > 2.5         → -8   (高估)

industry_category 缺失 / 異常 → fallback to absolute PBR cap (>=10 -5;>=5 -2)
金融業特殊處理:絕對 PBR <=1.5 +3(銀行 BV 計算特殊)
```

**設計依據**:
- 通信網路業 avg PBR 34.77 vs 電腦業 6.64,跨產業差異 ~5×
- 金融業 BV 計算特殊(資產負債性質不同)

### 4.3 Dividend Yield(主源 PER.dividend_yield)— ±10 範圍

```
yld = PER.dividend_yield  (% 已單位化)

yld > 5%               → +8  (高殖利率;但需檢 distress)
3% < yld <= 5%         → +5
1% < yld <= 3%         → 0
0 < yld <= 1%          → -2
yld == 0               → -5  (不配息)

yld > 10%(distress 警示)→ -5(可能因股價暴跌造成偽訊號)
```

**設計依據**:
- core 150 latest avg 1.77%(穩健配息為主)
- distress 之 yld > 10% 屬偽訊號(分母股價暴跌)

### 4.4 配息穩定性(from Dividend.CashEarningsDistribution)— ±10 範圍

```
past_5y_dividend_count = COUNT(DISTINCT year)
    FROM TaiwanStockDividend
    WHERE stock_id = X AND CashEarningsDistribution > 0
      AND year IN (last 5 years)

5 次配息 → +10  (穩定配息)
4 次     → +6
3 次     → +3
2 次     → 0
1 次     → -2
0 次     → -3  (5 年無配息)

新上市 < 5 年 → 用 (count / years_listed * 10) 比例化
```

**設計依據**:
- CashEarningsDistribution 87.5% 覆蓋,可作主訊號
- 穩定配息 = 高品質 V 訊號(對應「永續經營能力」)

### 4.5 Operating Margin(OperatingIncome / Revenue)— ±10 範圍

```
op_margin = latest_quarter_OperatingIncome / latest_quarter_Revenue

op_margin > 30%        → +10  (高獲利能力)
15% < op_margin <= 30% → +5
5% < op_margin <= 15%  → 0
0% < op_margin <= 5%   → -3
op_margin <= 0%        → -8   (虧損營運)
```

**設計依據**:
- core 150 之 OperatingIncome 100% 覆蓋 + Revenue 99% 覆蓋
- 補強現行 FG 只看 EPS(總損益)未看「**營運獲利能力**」之 gap

### 4.6 Attributable Ratio((IncomeAfterTaxes − NoncontrollingInterests) / IncomeAfterTaxes)— ±5 範圍

```
attr_ratio = (IncomeAfterTaxes - NoncontrollingInterests) / IncomeAfterTaxes
    當 IncomeAfterTaxes <= 0 或 NoncontrollingInterests > IncomeAfterTaxes → 異常

attr_ratio > 0.95      → +3   (高歸屬母公司比例)
0.85 < attr_ratio <= 0.95 → +1
0.7 < attr_ratio <= 0.85  → 0
attr_ratio <= 0.7         → -3 (大量非控股權益)
異常(NCI > NI)            → -5
```

**設計依據**:
- NoncontrollingInterests core 150 之 131 stocks(87%)有資料
- 補強現行 FG 只看「合併損益」未看「**歸屬母公司比例**」之 gap

---

## 五、FG v0.5 總分計算流程

```python
def _fundamental_gravity_score_v0_5(stock_data):
    base = 50
    score = base

    # === v0.3 既有(維持)===
    score += revenue_yoy_score(stock_data.revenue_yoy)        # ±25
    score += eps_score(stock_data.eps_sum, stock_data.net_pos)# ±15
    score += coverage_bonus(stock_data.coverage_avg)          # +10
    score += gross_margin_score(stock_data.gross_margin)      # ±10  (v0.3 加)

    # === v0.5 新增 ===
    score += per_industry_relative_score(stock_data.per, industry_median_per)  # ±20
    score += pbr_industry_relative_score(stock_data.pbr, industry_median_pbr)  # ±15
    score += dividend_yield_score(stock_data.div_yield)        # ±10
    score += dividend_stability_score(stock_data.div_count_5y) # ±10
    score += operating_margin_score(stock_data.op_margin)      # ±10
    score += attributable_ratio_score(stock_data.attr_ratio)   # ±5

    # === 邊界 clamp(必要)===
    return max(0, min(100, score))  # 0..100
```

**總範圍**:基準 50 + v0.3 既有 ±60 + v0.5 新增 ±70 = -80~180 → clamp 0..100

**典型 stock 分數模擬**:
- 「優質穩健股」(台積電型):50 + 8(yoy) + 5(eps) + 10(cov) + 10(GM 66%)+ 0(rel_per 1.2)+ -3(rel_pbr 1.8 半導體)+ 5(yld 1.6%)+ 10(配息 5y)+ 10(op_margin 40%)+ 3(attr 0.97)= **108 → clamp 100**
- 「穩配息成熟股」:50 + 0(yoy stagnant)+ 5(eps)+ 10(cov)+ 5(GM 30%)+ 5(rel_per 0.8)+ 0(rel_pbr 1.2)+ 8(yld 5%)+ 10(配息 5y)+ 5(op_margin 20%)+ 3 = **101 → clamp 100**
- 「成長股」:50 + 25(yoy 50%)+ 5(eps)+ 10(cov)+ 10(GM)+ -5(rel_per 1.7)+ -8(rel_pbr 3)+ -2(yld 0.5%)+ -2(配息 2y)+ 10(op_margin 30%)+ 1 = **94**
- 「distress」:50 + -10(yoy -8%)+ -15(無獲利)+ 5(cov 部分)+ -8(GM 3%)+ 0+ 0+ -5(distress yld 12%)+ -3(0 配息)+ -8(虧損營運)+ 0 = **6**

---

## 六、與既有 v0.3 之相容裁決

### 6.1 FG 權重 20% 維持不變

§6.4 CoreScore 公式 v0.2 之總結構保留:
```
CoreScore = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP
```
- FG 內部結構升(5 → 11 sub-scores)
- FG 權重 20% 不變(不打破公式)

### 6.2 既有 v0.3 snapshot 影響

- 既有 `core_universe_20260524_core_universe_policy_v0_2` snapshot **不重 build**
- 新 v0.4 snapshot(`core_universe_policy_v0.4`)起適用 v0.5 builder
- 預期 fundamental_score 分布變動:中位數預估上升(因新增正向訊號 +10~20),需重 calibrate clamp

### 6.3 §6.7 SSOT 不變

- 150 鎖定不變(core 120 + convex 30)
- 但 v0.4 snapshot 之**選股結果可能與 v0.3 略有差異**(因 FG 改變 → CoreScore 變 → 排序變 → top-150 略有 churn)
- 預期 churn rate < 15%(以往升 patch 模式)

### 6.4 治權邊界嚴守

本研究**不**修改:
- §6.4 CoreScore 公式總結構(6 維 + RP)
- §6.7 SSOT(150 鎖定)
- §0.1-A 6 條禁令
- §0.1.3-A.1 ROE dropped 裁決(本研究**仍維持** ROE = None)
- §6.3 第 4 條 FG 條文原文(留待 v6.2.0 升強制契約時再升條文)

---

## 七、證偽承諾(對接 §0.1-E 框架)

| 指標 | 觀察期 | 通過門檻 | 不通過裁決 |
|---|---|---|---|
| **T_FG_v0.5.1** | 滾動 5 年 | v0.5 prediction h20 IC ≥ v0.3 baseline | < baseline 則 v0.5 撤回,policy 回退 v0.3 |
| **T_FG_v0.5.2** | 滾動 5 年 | fundamental_score 與 industry-relative valuation 相關 > 0.4 | < 0.4 則重審 sub-score 設計 |
| **T_FG_v0.5.3** | walk-forward h20 panel | v0.5 IC stdev ≤ v0.3 stdev | 退步則停留 v0.3 |
| **T_FG_v0.5.4**(本研究新增)| dry-run | fundamental_score 分布之 mean/std 與 v0.3 差異 ∈ [+5, +20] | 差異 > +30 表示新 sub-score 過度疊加;< 0 表示新 sub-score 反向 |

---

## 八、追溯適用裁決

| 既有 artifact | 處理 |
|---|---|
| `core_universe_20260524_core_universe_policy_v0_2` snapshot | **不重 build**;標記 `policy_version='legacy_v0.2'` |
| `core_universe_policy_v0.3`(2026-05-24 入憲) | **保留為 audit trail**;v0.5 builder 走新 policy v0.4 |
| 既有 v0.3 之 FG sub-scores | **保留為 v0.3 baseline**(audit 對照用) |
| 既有 audit_core_universe.py v0.4 之 policy_version 識別 | **需升 v0.5 加 v0.4 policy 識別** |
| 既有 walk-forward h20/h30 panel | **保留**;新 v0.4 snapshot 用於下一輪 walk-forward |

---

## 九、風險與回退方案

| 風險 | 機率 | 回退 |
|---|---|---|
| PER outlier(> p99=1766)處理太寬讓泡沫股得高分 | 中 | cap at p95=190(core 150)+ risk_penalty |
| industry_category 太粗導致 mis-relative | 中 | fallback to absolute PBR cap(>=10 -5)|
| 上市晚之股配息歷史不足 5 年 | 中 | 用 3y 替代 + 寬鬆評分;按 years_listed 比例化 |
| v0.5 IC < v0.3 baseline | 低-中 | 政策版本回退 v0.3(audit_core_universe 支援雙版本) |
| 新 sub-score 過度集中於半導體股(已 100% 集中)| 中 | sector_cap 在下游 §9.2 portfolio_sizer 處理 |
| **dry-run 揭露新的「資料現實裁決」**(§14.7-BC 第四次跑通)| 中 | 追溯修正 sub-score 設計;類比 §14.7-BB 之模式 |

---

## 十、實作計畫(builder v0.4 → v0.5)

### 10.1 核心變更

| 模組 | 變更 |
|---|---|
| `core_universe_builder.py v0.4 → v0.5` | 新增 `_load_per()` / `_load_dividend()` / 擴張 `_load_financial()` 加 4 新 types;升 `_fundamental_gravity_score()` 為 v0.5 11-sub-score 版本;DEFAULT_POLICY_VERSION v0.3 → v0.4 |
| `audit_core_universe.py v?.? → v?.?+1` | 加 `core_universe_policy_v0.4` 識別;FG v0.5 sub-scores 之驗收 |
| `core_universe_schema.py`(若需擴張)| 評估是否需在 `core_universe_scores.score_detail` JSONB 內新增 v0.5 sub-score 鍵(預期不需要,score_detail 已是 JSONB 可彈性存) |

### 10.2 落地序列

```
Step 1: 設計研究報告(本文檔)
Step 2: 入憲 §14.7-BC(治權先行)
Step 3: 實作 core_universe_builder v0.5
Step 4: dry-run on as_of=2026-05-21
        對照 v0.4 vs v0.5 之 fundamental_score per stock
        對照 v0.4 vs v0.5 之分層差異(120/30/2239/378 churn rate)
Step 5: audit_core_universe.py 升版加 v0.4 policy 識別
Step 6: commit
Step 7: 「資料現實裁決」第四次跑通檢驗(若 dry-run 揭露新問題,追溯修正)
```

### 10.3 治權邊界嚴守

本實作**不**修改:
- §6.4 CoreScore 6 維權重結構
- §6.7 SSOT 150 鎖定
- §0.1-A 6 條禁令
- §0.1.3-A.1 ROE dropped 裁決
- raw DDL
- CLI 介面(`--dry-run / --commit / --as-of-date / --special-rebalance-reason / --policy-version`)
- annual_rebalance_guard / candidate_fallback / 5 張治理表寫入順序

---

## 十一、與既有契約之相容裁決 + 後續路徑

### 11.1 與 §6.3 第 4 條 FG 公式之相容

§6.3 第 4 條為**治權條文**(現行 v0.2 公式 + v0.3 GrossProfit 已實作但未升條文);本研究之 v0.5 FG 升版屬「**實作擴張**」非「條文升版」;§6.3 條文原文**不修改**;待 v6.2.0 升強制契約時再評估升條文。

### 11.2 與 §6.4 CoreScore 公式之相容

§6.4 CoreScore 6 維權重結構不變(`0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP`);FG 內部 11 sub-scores 之 clamp 仍為 0..100;FG × 0.20 維持。

### 11.3 與 §14.7-W feature_store research 之關係

§14.7-W 列「**財報/月營收公告日語意**」為後續待研究 — 已部分由 §14.7-BA Publication-date Discipline 兌現;本研究兌現另一部分「**Dividend 之 dead column 動員**」(Phase D)。

### 11.4 與 §14.7-AX 治權元規則之關係

**第四次跑通預備**:若 dry-run 揭露 PER outlier / Dividend year / industry 分類等資料現實問題,觸發 §14.7-BC 後續追溯修正(類比 §14.7-BB 對 §14.7-BA 之追溯)。

### 11.5 後續可選路徑

| 階段 | 內容 | 用戶授權 |
|---|---|---|
| **A(本研究)** | 設計研究 + 入憲 §14.7-BC + builder v0.5 落地 + dry-run | 進行中 |
| **B** | walk-forward IC simulation(等 §8 schema 表建立 + feature_store_builder v0.5 整合)| 後續 |
| **C** | 若 IC < v0.3 baseline → 政策回退 v0.3(audit_core_universe 雙版本) | 後續 |
| **D** | 若 IC ≥ v0.3 baseline + T_FG_v0.5.1-4 全通過 → 升憲章 v6.2.0,§6.3 第 4 條條文升版 | 後續 |

---

## 十二、治權聲明

### 12.1 嚴守 §0.0-G 憲章先行紀律

本研究報告為**草案性提案**,先入憲 §14.7-BC 治權閉環記述,後續實作 builder v0.5 + dry-run + commit。

### 12.2 嚴守 §0.1-A 6 條禁令

- 不寫物理隱喻字面公式(F = M × ΔlnP)
- 不實作 IFF Θ / SOC / 重力井邊緣 trigger
- 不寫地緣事件敘事
- 不用物理隱喻替代 backtest 證據(本研究有 T_FG_v0.5.1-4 證偽承諾)

### 12.3 與 §0.1.3-A.1 ROE 之關係

§0.1.3-A.1 ROE dropped 裁決**仍維持**(本研究**不**嘗試重新實作 ROE;`financial_data[sid]['roe'] = None` 占位保留)。

### 12.4 與 §14.7-AX「資料層揭露驅動治權升版」之關係

本研究再次驗證「**資料層揭露驅動治權升版**」之機制:
- §0.1.3-B field 盤點 → 揭露 V 動員 gap → 本研究設計 Phase C/D 補強
- 若 dry-run 揭露新資料現實問題 → §14.7-BC 第四次跑通

---

## 附錄 A — DB 實證查詢全紀錄(2026-05-25)

```sql
-- 1. PER 全市場分布
WITH latest AS (
    SELECT DISTINCT ON (stock_id) stock_id, "PER"
    FROM "TaiwanStockPER" WHERE date <= '2026-05-21' AND "PER" IS NOT NULL AND "PER" > 0
    ORDER BY stock_id, date DESC
)
SELECT COUNT(*), MIN("PER"), percentile_cont(0.05/0.25/0.5/0.75/0.95/0.99) WITHIN GROUP, MAX("PER")
FROM latest;
-- 結果:n=1982 / min 1.11 / p5 8.70 / p25 15.50 / median 27.07 / p75 66.17 / p95 443.45 / p99 1766.90 / max 14,700

-- 2. PER core 150 outlier
-- 結果:median 38.98 / p95 190.32 / PER>100: 23 / PER>50: 58

-- 3. PBR by industry
-- 結果:通信網路 34.77 / 半導體 14.03 / 電腦 6.64 / ... 跨產業差異 ~5×

-- 4. Dividend 22 cols 逐 col 覆蓋
-- 結果:CashEarningsDistribution 87.5% 為主訊號 / ParticipateDistribution 42% / RemunerationDirectors 34% / TotalEmployeeCashDividend 31.5%

-- 5. FinStmt 4 新 types core 150 stock 覆蓋
-- 結果:OperatingIncome/PreTaxIncome/IncomeFromContinuingOperations 150/150 / NoncontrollingInterests 131/150
```

## 附錄 B — 憲章條文 cross-ref(已實際驗證)

| 引用條文 | 入憲狀態 | 行號 |
|---|---|---|
| §0.1.3 Fundamental 第四變數 | ✅(2026-05-19)| §0.1.3 |
| §0.1.3-A V/ΔlnP 工程落地實況 | ✅(2026-05-24)| §0.1.3-A |
| §0.1.3-A.1 ROE 不可實作 | ✅(2026-05-24)| §0.1.3-A.1 |
| §0.1.3-B DB Field Bottom-up | ✅(2026-05-25 commit `8f40836`)| §0.1.3-B |
| §6.3 第 4 條 FG 公式 | ✅(v6.0.0 既有)| §6.3 |
| §6.4 CoreScore 6 維權重 | ✅(v6.0.0 既有)| §6.4 |
| §6.7 SSOT(150 鎖定)| ✅ | §6.7 |
| §14.7-W feature_store research(公告日語意 待研究)| ✅(2026-05-19)| 修訂歷程 L102 |
| §14.7-AX 資料層揭露驅動治權升版 | ✅(2026-05-24)| §14.7-AX |
| §14.7-BA Publication-date Discipline | ✅(2026-05-25)| §14.7-BA |
| §14.7-BB FRED strict → transitional 追溯 | ✅(2026-05-25)| §14.7-BB |
| **§14.7-BC V 補強 Phase C/D 治權閉環** | ❌ **本研究預備入憲** | — |

## 附錄 C — 與 §0.1.3-A.1 ROE dropped 之關係

§0.1.3-A.1(2026-05-24)裁決 ROE 不可實作 — 「資料現實裁決」第一次跑通。

本研究**仍維持** ROE = None,**不**嘗試重新實作 ROE。但本研究**補強其他可計算之 V 訊號**:
- 取代 ROE 之獲利能力訊號:Operating Margin(OperatingIncome / Revenue)
- 取代 ROE 之估值訊號:PER / PBR(industry-relative)
- 取代 ROE 之品質訊號:Attributable Ratio(歸屬母公司比例)+ 配息穩定性

**§14.7-AX 元規則跑通對照**:
- 第 1 次:§0.1.3-A.1 ROE 不可實作 → ROE dropped + GrossProfit 替代
- 第 2 次:§14.7-BA 5 個 publication-date 欄位 3 不可用 → 分層落地
- 第 3 次:§14.7-BB FRED vintage strict gate 不可行 → strict → transitional
- **第 4 次(本研究預備)**:**dry-run 後若揭露 PER outlier / Dividend year / industry 分類等資料現實問題 → §14.7-BC 後續追溯修正**

---

## 十三、後續接續點

| 條件 | 動作 |
|---|---|
| 用戶確認本研究 | 入憲 §14.7-BC + 修訂歷程 v6.1.0-patch 2026-05-25 第六輪 entry |
| 入憲完成 | 實作 core_universe_builder v0.5 + audit_core_universe 升版 |
| 實作完成 | dry-run on as_of=2026-05-21,對照 v0.4 vs v0.5 之 fundamental_score 變化 |
| dry-run 通過 | commit;準備 walk-forward IC simulation(等 §8 schema 表建立) |
| dry-run 揭露問題 | 「資料現實裁決」第四次跑通,追溯 §14.7-BC + builder v0.5 design |
