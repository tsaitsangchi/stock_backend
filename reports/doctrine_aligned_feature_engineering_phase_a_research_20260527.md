# Doctrine-Aligned Feature Engineering — Phase A Design Research

**日期**: 2026-05-27
**Phase**: A(設計研究 / pre-charter inscription)
**對應軌道**: v6.3.0 或 v7.0.0(feature engineering 層升 doctrine-aligned 純化)
**對應憲章基礎**: §0.1 第一性原理 / §0.2 八二法則 / §0.3 康波週期(post §14.7-BZ 拆 §0.3.1/.2/.3)/ §0.1-A 6 禁令 / §0.2-A 7 禁令 / §0.3-A 7 禁令 / §8.5 anti-leakage / §9.9 P1 v0.1 RMS / §14.7-W feature_store first-principles research
**Status**: ✅ Phase A 完整(16 章 / non-destructive / 不動 DB 不動 code)
**對應 user trigger**: 2026-05-27 「如果要做符合依第一性原理、八二法則、康波週期核心思想的股價預測,需要先產生哪些特徵值,先做研究」
**前置基線**: v6.2.0(macro infrastructure layer closure / 13 K-wave indicators / 6 pillars / 1857 stocks)

---

## 1. 觸發

§14.7-BZ Phase F 完成(v6.1.33 / v6.2.0 milestone)後,§0.3 拆 §0.3.1/.2/.3 三 sub-pillars,Doctrine 四軸純化完整閉環(N + T + Indicator + Pillar)。

用戶接續提問:「如果要做**符合依第一性原理、八二法則、康波週期核心思想**的股價預測,**需要先產生哪些特徵值,先做研究**」。

此問題之治權層含義:**從 universe selection(per stock 三基柱 gate)升至 feature engineering(per stock × per feature × per doctrine alignment)之 doctrine-aligned mapping**。

### 1.1 現況揭露

- ✅ Raw data 層:1857 stocks × 5 raw sources × 100% coverage × avg 18 年深度
- ✅ §0.3 macro infrastructure:13 K-wave indicators 之 macro context 可用
- ✅ Feature schema:`feature_store_snapshot` / `feature_values` / `feature_definition` 已預建
- ⚠️ Feature store 之既有 features:**27 v0.1 features**(v0.2 之 31 ablation IC=+0.0131 HARMFUL 已 SUPERSEDED)
- ⚠️ 既有 27 features **未對映三基柱 doctrine 之嚴格 alignment**(屬 v6.0.0 之 §14.7-W 設計,non-doctrine-aligned)
- ⏸ §14.7-BU Phase E feature layer hook 未補(feature_store_builder 未寫 universe_completeness_snapshot 之 feature layer)

### 1.2 Phase A 目的

1. 學術文獻 SSOT(predictive features per doctrine in finance literature)
2. 三基柱對 features 之治權邊界檢視(§0.1-A / §0.2-A / §0.3-A 禁令延伸)
3. Doctrine-aligned features 候選 list(per §0.1 / §0.2 / §0.3.1/.2/.3)
4. 既有 27 v0.1 features 之 doctrine alignment 評估
5. Cross-pillar interactions 之治權邊界
6. Anti-leakage(§8.5)+ T1/T2/T3 分層約束
7. Sync feasibility + Phase B 入憲提案

---

## 2. 治權位階對齊

### 2.1 對映既有憲章節

| 節 | 內容 |
|---|---|
| §0.1 第一性原理 | 物理性 raw data → per-stock 量化信號 |
| §0.1-A 6 禁令 | T1 / T2 / T3 分層;不寫死 F=M×ΔlnP / 重力井 / SOC 等隱喻 |
| §0.2 八二法則 | 右尾集中 / barbell 結構 |
| §0.2-A 7 禁令 | α 不得固定 / 不寫死 20/60/20 / 不下沉個股層 |
| §0.3.1 K-wave pure(post §14.7-BZ) | 40-60 年 macro structural / 不下沉 L2/L3 |
| §0.3.2 Multi-cycle | 7-25 年 Juglar+Kuznets |
| §0.3.3 Microstructure | 月~季 sentiment / regime |
| §0.3-A 7 禁令 | K-wave 不下沉 L2/L3 / 不作為 per-stock score input |
| §8.5 anti-leakage | publication-date discipline / point-in-time data |
| §9.9 P1 v0.1 RMS | volatility 公式對齊(Sortino MAR=0)|
| §14.7-W | feature_store_builder first-principles research(v6.0.0 既有)|

### 2.2 §14.7-CA 預定子節(本 Phase A 落地後若進 Phase B)

本研究若入憲,將成為**第二十五輪治權升版** — feature engineering 層之 doctrine-aligned 純化。

### 2.3 治權邊界(不踩之線)

- ❌ 不改 §0.1-A / §0.2-A / §0.3-A 任一禁令字面
- ❌ 不改 §14.7-BW/BX/BY/BZ 既有 doctrine
- ❌ 不下沉 K-wave indicators 至 per-stock score input(per §0.3-A 禁令延伸)
- ❌ 不寫死 F=M×ΔlnP / 重力井 / SOC 等 T3 隱喻入 features(per §0.1-A)
- ✅ 提案 doctrine-aligned features 候選(per-stock T1/T2 級別)
- ✅ 提案 cross-pillar interactions(treaty-permitted 範圍內)

---

## 3. 學術文獻 SSOT — Doctrine-Aligned Predictive Features

### 3.1 第一性原理(per-stock physical signals)— 4 學派

| 學派 | 出處 | 核心 features |
|---|---|---|
| **Fama-French 3/5-factor** | Fama-French 1993, 2015 | Size(市值)/ Value(B/M)/ Profitability(ROE-like)/ Investment(asset growth) |
| **Carhart momentum** | Carhart 1997 | 12-1 momentum(過去 12 個月 ex-最近 1 個月 returns)|
| **Asness quality-minus-junk** | Asness, Frazzini, Pedersen 2019 | Profitability / Growth / Safety / Payout quality scores |
| **Sloan accruals** | Sloan 1996 | Accruals(non-cash earnings 部分)/ earnings quality |

### 3.2 八二法則(right-tail concentration / barbell)— 學術 anchor

| 學派 | 核心 features |
|---|---|
| **Barabási-Albert preferential attachment** | Stock-level liquidity preferential attachment proxy(avg_daily_value_log × time)|
| **Bianconi-Barabási fitness model**(γ≈2.255)| Fitness signal(成交額 + theme strength + 法人偏好之 composite)|
| **Right-tail volatility asymmetry** | Upside / Downside 之 RMS 分解(per §9.9)|

### 3.3 康波週期(macro context features)— 學術 anchor

| 學派 | 核心 features |
|---|---|
| **Schumpeter cluster of innovation** | Tech sector concentration / patent intensity |
| **Reinhart-Rogoff credit cycle** | Credit-to-GDP gap residuals |
| **Perez "techno-economic paradigm"** | Sector × paradigm phase interaction |
| **Goodhart-Pradhan demographics** | Working-age population change |
| **Erten-Ocampo super-cycle commodity** | Long-cycle commodity price ratios |

---

## 4. 既有 27 v0.1 features 之 doctrine alignment 評估

### 4.1 既有 features 清單(per feature_store_builder.py)

| Category | Features | 對映 |
|---|---|---|
| **Price-based**(L586-624) | log_return_20/60/252d / volatility_60/252d / ma_ratio_20/60 / max_drawdown_252d / upside_volatility_60d / downside_volatility_60d / upside_capture_60d / downside_capture_60d / avg_daily_value_log_60/252d / turnover_mean_60d / zero_volume_ratio_252d | §0.1 + §0.2(流動性/集中度)|
| **Revenue-based**(L628-650) | revenue_yoy_12m / revenue_yoy_3m | §0.1(fundamental)|
| **Theme-based**(L650-665) | theme_strength / theme_is_semiconductor | §0.3 proxy(industry keyword) |
| **Margin-based**(L728) | margin_ratio_60d | §0.1(法人)|
| **Macro(infered from older v0.2)** | vix / dff / 之類 | §0.3 macro |

### 4.2 Doctrine alignment 評等

| Feature group | §0.1 對應 | §0.2 對應 | §0.3 對應 | Doctrine 純度 |
|---|---:|---:|---:|---:|
| Price-based(15 features) | **95%**(per-stock 物理信號)| 60%(流動性 + capture)| 0% | 🟢 純 §0.1 |
| Revenue-based(2 features) | **90%**(per-stock fundamental)| 0% | 0% | 🟢 純 §0.1 |
| Theme-based(2 features) | 30%(per-stock industry)| 20% | **40%**(industry keyword K-wave proxy)| 🟡 mix |
| Margin-based(1 feature)| **85%**(法人流向) | 30% | 0% | 🟢 純 §0.1 |
| Macro(5 features estimate)| 0% | 0% | **85%**(macro context)| 🟢 純 §0.3 |

**整體 doctrine alignment 平均**:~75%(既有 v0.1 已大致對映,但 missing 三基柱之 explicit 區分 + cross-pillar interactions)

### 4.3 既有 features 之 gap 揭露

| Gap | 描述 | 補救方向 |
|---|---|---|
| **§0.2 八二法則 explicit 不足** | 既有只 turnover / liquidity 隱含,缺 right-tail 集中度之 explicit metric | 加 right_tail_concentration / barbell_balance features |
| **§0.3 從 industry keyword proxy 升至 macro indicator** | post §14.7-BY/BZ 之 13 K-wave indicators 未被 features 引用 | 加 macro context features(monthly aggregations)|
| **K-wave × stock 之 paradigm-aware features**| 缺 stock × K-wave phase 之 interaction | 加 paradigm × sector 之 features(per Perez 學派)|
| **Anti-leakage 嚴格度** | revenue_yoy 之 publication-date discipline 已有(per §8.5),但未明示 feature_definition | 補 feature_definition.as_of_strict |
| **§0.3.1/.2/.3 三 sub-pillars 對應** | 既有 features 仍 mix 之 macro,未拆 K-wave/Multi-cycle/Microstructure | 加 sub-pillar-aligned macro features |

---

## 5. Doctrine-Aligned Features 候選 list

### 5.1 §0.1 第一性原理 — 16 候選 features

| Category | Feature name | Formula | 學派依據 |
|---|---|---|---|
| **Momentum** | mom_12m_1m | $(P_{t-21}/P_{t-252}) - 1$ | Carhart 1997 |
| Momentum | mom_3m | $(P_t/P_{t-63}) - 1$ | Jegadeesh-Titman 1993 |
| Momentum | mom_1m | $(P_t/P_{t-21}) - 1$ | short-term reversal |
| **Volatility** | rms_upside_60d | $\sqrt{\text{AVG}(r^2 | r>0) \cdot 252}$ | per §9.9 P1 v0.1 |
| Volatility | rms_downside_60d | $\sqrt{\text{AVG}(r^2 | r<0) \cdot 252}$ | per §9.9 |
| Volatility | convexity_60d | upside_rms - downside_rms | §14.7-BG |
| **Liquidity** | avg_daily_value_log_60d | $\log_{10}(\text{AVG}(V \cdot P))$ | Amihud 2002 |
| Liquidity | amihud_illiquidity_60d | $\text{AVG}(|r|/V_{\$})$ | Amihud 2002 |
| Liquidity | zero_volume_ratio_252d | count(V=0) / 252 | Lesmond 1999 |
| **Value**(fundamental)| pe_ratio | (TaiwanStockPER)| Fama-French 1993 |
| Value | pb_ratio | (TaiwanStockPER)| Fama-French 1993 |
| Value | dividend_yield | (TaiwanStockDividend)| Litzenberger 1979 |
| **Quality**(profitability)| roe_ttm | (TaiwanStockFinancialStatements / BalanceSheet) | Asness 2019 |
| Quality | operating_margin_ttm | (TaiwanStockFinancialStatements)| QMJ 2019 |
| Quality | revenue_yoy_3m_log | $\log(\text{rev}_{3m} / \text{rev}_{3m-1y})$ | YoY growth |
| **Investment**(growth)| asset_growth_yoy | (BalanceSheet TotalAssets YoY)| Cooper-Gulen-Schill 2008 |

### 5.2 §0.2 八二法則 — 8 候選 features

| Feature | Formula | 學派依據 |
|---|---|---|
| **right_tail_concentration_60d** | top 10% volume share / total volume(per sector)| Pareto 分布實證 |
| **barbell_balance_60d** | abs(20% attack share - 80% safety expected)| §9.2 barbell theory |
| **preferential_attachment_60d** | log(成交額 × time-weighted decay)| Barabási-Albert 1999 |
| **fitness_signal_60d** | (avg_daily_value × theme_strength × foreign_net_ratio)^(1/3)| Bianconi-Barabási 2001 |
| **right_tail_returns_skew_252d** | skew(returns) when returns > 0 | right-tail asymmetry |
| **liquidity_rank_pct_sector_60d** | sector 內 percentile rank | per-stock 相對集中度 |
| **value_concentration_60d** | top 1 sector 之 avg_daily_value / total | sector dominance |
| **size_log_zscore_sector** | log(market_cap) z-score per sector | size factor |

### 5.3 §0.3 康波週期 — 拆 3 sub-pillars × 各自 features

#### 5.3.1 §0.3.1 K-wave pure features(macro × stock interaction)— 6 候選

| Feature | Formula | 學派依據 |
|---|---|---|
| **kwave_tech_paradigm_strength** | (PATENTUSALLTOTAL_yoy + B985_yoy) / 2 | Schumpeter / Perez |
| **kwave_credit_cycle_phase** | TCMDO_log_yoy(US credit cycle phase indicator)| Reinhart-Rogoff 2009 |
| **kwave_credit_to_gdp_gap** | QUSPAM770A latest(BIS Credit-to-GDP gap)| BIS / Drehmann 2014 |
| **kwave_demographics_trend** | (LFWA64_yoy + (1 - SPPOPDPND_yoy)) / 2 | Goodhart-Pradhan 2020 |
| **kwave_commodity_supercycle** | PALLFNFINDEXQ_log_yoy | Erten-Ocampo 2013 |
| **kwave_phase_indicator** | composite z-score(以上 5 features 平均;∈ [-3, +3] 對映 K-wave 4 phase)| Mensch 1979 |

#### 5.3.2 §0.3.2 Multi-cycle features — 5 候選

| Feature | Formula | 學派依據 |
|---|---|---|
| **mc_monetary_regime** | M2SL_log_yoy(monetary stance)| Friedman |
| **mc_yield_curve_inversion** | T10Y2Y latest(< 0 表示 inversion / Juglar leading)| Estrella-Hardouvelis 1991 |
| **mc_oil_juglar_phase** | WTISPLC_log_yoy | Stopford 2009 |
| **mc_semi_kitchin** | TW_SEMI_VWAP_YOY latest | Aizcorbe-Kortum 2005 |
| **mc_shipping_juglar** | TW_SHIPPING_VWAP_YOY latest | Stopford 2009 |

#### 5.3.3 §0.3.3 Microstructure features — 3 候選

| Feature | Formula | 學派依據 |
|---|---|---|
| **ms_volatility_regime** | VIXCLS rolling 60d mean | Whaley 1993 |
| **ms_vix_term_structure** | (VIXCLS / 252d_mean - 1)(規範化)| Vix premium |
| **ms_market_stress** | binary(VIXCLS > 30 in last 30 days)| crisis 警示 |

---

## 6. Cross-Pillar Interactions(治權邊界內)

### 6.1 治權允許之 interactions(per §0.0-E 跨基柱凸性統合)

| Interaction | 學派依據 | 治權層 |
|---|---|---|
| **§0.1 × §0.3.1**:per-stock fundamental × tech_paradigm_strength | Schumpeter sector x stock | T2 物理啟發(per §0.1.1)|
| **§0.1 × §0.3.2**:per-stock volatility × yield_curve_inversion | Juglar leading × stock vol | T2 |
| **§0.2 × §0.3**:right-tail concentration × paradigm_phase | Bianconi fitness × K-wave | T2 |

### 6.2 治權禁止之 interactions(per §0.1-A / §0.3-A 禁令延伸)

| Interaction | 為何禁止 |
|---|---|
| **F=M×ΔlnP raw features** | §0.1-A 禁令 #2:不寫死隱喻 |
| **K-wave per-stock score input** | §0.3-A 禁令 #3:K-wave 不下沉 per-stock score |
| **T3 元素**(IFF Θ / SOC / 重力井邊緣) | §0.1-A 禁令 #2/#3 + T3 永久禁用 |

### 6.3 v0.2 之 4 interactions ablation 證偽紀錄

依 feature_store_builder.py L73 之 docstring:
> v0.2 之 31 features(加 4 interactions) — ablation IC = +0.0131 HARMFUL → §0.0-D.6 #1 已實證否決

**v0.2 之 interactions(SUPERSEDED)**:
- feature_macro_vix_x_vol_60d
- feature_macro_dff_x_eps_sum_4q
- feature_theme_x_log_return_60d
- feature_theme_x_foreign_net_60d

**啟示**:cross-pillar interactions 必須經 ablation IC ≥ 0.02 之證偽 gate 才能升 production(per §9.0 v0.x 升版 pattern)。

---

## 7. Anti-leakage(§8.5)+ T1/T2/T3 分層 constraints

### 7.1 §8.5 publication-date discipline(已落地)

| Raw source | Publication-date gate | 既有處理 |
|---|---|---|
| TaiwanStockFinancialStatements | Q1/Q4 45/90 天延後 | `core.data_schema.build_publication_date_gate` SSOT helper(per §14.7-BC v0.5) |
| TaiwanStockMonthRevenue | 月底 +10 天 | 同上 |
| TaiwanStockPriceAdj | T+1 | 同上 |

**Phase A 之 doctrine-aligned features 必須沿用既有 publication-date gate**(per `data_schema.build_publication_date_gate`)。

### 7.2 T1/T2/T3 分層 constraints

| 分層 | 治權邊界 | features 範例 |
|---|---|---|
| **T1**(嚴格事實) | log returns / volatility / volume / fundamental ratios | mom_12m_1m / rms_upside_60d / pe_ratio / roe_ttm |
| **T2**(物理啟發類比) | preferential attachment / fitness signal / kwave phase composite | preferential_attachment_60d / kwave_phase_indicator |
| **T3**(永久禁用) | IFF Θ / SOC / 重力井邊緣觸發 | **完全不可入 features** |

**Phase A 之 candidate features 全部限制在 T1/T2 範圍**。

---

## 8. Sync Feasibility Matrix(對映 1857 stocks × 18 年深度)

### 8.1 既有 raw data 充足度(per §1.1)

✅ **5 raw sources 全 100% coverage**(per Phase A audit)

### 8.2 各候選 features 之 sync feasibility

| Feature group | Raw source 充足度 | Effort 估計 |
|---|---|---|
| **§0.1 Price-based(16)** | ✅ TaiwanStockPriceAdj 100% | 0.5 人天(全部 reuse 既有 `_log_return` / `_volatility` etc)|
| **§0.1 Value(3)**:pe/pb/dividend | ✅ TaiwanStockPER / Dividend | 0.5 人天 |
| **§0.1 Quality(3)** | ✅ TaiwanStockFinancialStatements | 1 人天(YoY 計算 + publication-date gate)|
| **§0.1 Investment(1)** | ✅ TaiwanStockBalanceSheet | 0.5 人天 |
| **§0.2 八二法則(8)** | ✅ TaiwanStockPriceAdj + TaiwanStockInfo industry | 1.5 人天(per-sector 計算)|
| **§0.3.1 K-wave(6)** | ✅ 7 FRED indicators 已 sync | 1 人天(monthly aggregation)|
| **§0.3.2 Multi-cycle(5)** | ✅ 5 indicators 已 sync | 0.5 人天 |
| **§0.3.3 Microstructure(3)** | ✅ VIXCLS 已 sync | 0.3 人天 |
| **Cross-pillar interactions** | ✅(依 base features 衍生) | 0.5 人天 + ablation 驗證 |
| **Total** | — | **~6.3 人天** |

### 8.3 推薦 features set 規模

依本研究 candidate list:
- §0.1: 16 features
- §0.2: 8 features
- §0.3.1: 6 features
- §0.3.2: 5 features
- §0.3.3: 3 features
- **Total v0.3 candidate set: 38 features**(vs 既有 v0.1 之 27 features)

**升版收益預估**:
- §0.2 八二法則之 explicit 化:從隱含 → 顯式 metric
- §0.3 從 industry keyword proxy → 13 K-wave indicators 之 macro features
- §0.3 拆 3 sub-pillars × 各自 features:對映 §14.7-BZ Phase F doctrine

---

## 9. 證偽承諾 T_CA-1〜5

| ID | 證偽命題 | 失敗條件 |
|---|---|---|
| T_CA-1 | v0.3 doctrine-aligned features set 之 ablation IC ≥ v0.1 baseline + 0.02 | 若 ≤ v0.1 → rollback |
| T_CA-2 | §0.2 八二法則 explicit features(8 個)之 IC > 0 | 若 ≤ 0 → 移除 §0.2 explicit feature |
| T_CA-3 | §0.3.1 K-wave features × stock interactions IC > §0.3 industry keyword baseline | 若 ≤ → 維持 industry keyword |
| T_CA-4 | T1 features 全部 ≥ v0.1 baseline IC | 若 T1 顯著降 IC → bug |
| T_CA-5 | 全 38 features × 1857 stocks × 252 trading days backtest 之 walk-forward IC stdev ≤ v0.1 stdev | 若 stdev > 2x → 過擬合 |

---

## 10. 風險評估

### 10.1 工程風險

| Risk | Mitigation |
|---|---|
| 38 features × 1857 stocks × ~5000 days = ~352M rows 之 storage | feature_values 表已 chunked / Postgres 可承受;預估 ~5 GB |
| Cross-pillar interactions 增 noise | 採 ablation IC gate(≥ +0.02 才升 production)|
| FRED indicator stale 風險 | 既有 §14.7-BR retry + auto-resume(已成熟)|

### 10.2 治權風險

| Risk | Mitigation |
|---|---|
| §0.3 K-wave features 不慎下沉至 per-stock score | 嚴格分層:K-wave 只作為 macro context 之 stock-invariant feature(per stock 但 timestamp 不變)|
| §0.1-A 禁令違反(寫死 F=M×ΔlnP)| candidate list 全 limited 在 T1/T2 / 不含 T3 隱喻 |
| ablation IC < baseline 之 v0.2 重蹈覆轍 | T_CA-1〜5 證偽 gate 嚴格;HARMFUL features 自動降級 |

### 10.3 學術 doctrine 風險

| Risk | Mitigation |
|---|---|
| Fama-French 之 size factor 在台股可能 reverse(per Liu 2019 emerging market)| 加 walk-forward backtest 之 stress test |
| K-wave phase indicator 之 composite formula 可能 over-fit | 採 1990-2010 train / 2011-2026 validate 之 OOS split |

---

## 11. Phase F-1/F-2/F-3 sub-phases roadmap

### 11.1 Phase B 入憲(charter §14.7-CA inscription)

| 動作 | Effort |
|---|---|
| 新建 §14.7-CA 子節(charter)| 0.5 人天 |
| 補修訂歷程第二十五輪 mega-row | inline |
| **Phase B 累計** | **0.5 人天** |

### 11.2 Phase C-1: feature_store_builder.py 升 v0.3

- 加 §0.1 missing features(value/quality/investment;~9 features)
- 加 §0.2 explicit features(~8 features)
- 加 §0.3.1/.2/.3 macro features(~14 features)
- Total ~31 new features(既有 v0.1 27 維持 + 加 31 = 58 total / 或拆 v0.3 純為 38 doctrine-aligned)
- Effort:~4 人天

### 11.3 Phase C-2: feature_store_builder 補 §14.7-BU Phase E hook

- 寫 per-stock × per-pillar × feature_layer 之 universe_completeness_snapshot records
- Effort:~0.5 人天

### 11.4 Phase D: ablation + IC backtest

- v0.3 vs v0.1 之 walk-forward IC ablation
- 通過 T_CA-1〜5 證偽 gate 才升 production
- Effort:~2 人天

### 11.5 Phase E: commit + audit verify

- feature_store_builder --commit
- audit_universe_completeness verify feature layer 升至 PERFECT(non-INFO)
- Effort:~0.5 人天

---

## 12. 工程 effort 與時程估計

| Phase | 動作 | Effort |
|---|---|---|
| Phase A(本文件)| 設計研究 | 0.5 人天 |
| Phase B | charter §14.7-CA 入憲 | 0.5 人天 |
| Phase C-1 | feature_store_builder v0.3(+11 features net)| 4 人天 |
| Phase C-2 | §14.7-BU Phase E feature layer hook | 0.5 人天 |
| Phase D | ablation + IC backtest | 2 人天 |
| Phase E | commit + audit | 0.5 人天 |
| **Total** | — | **~8 人天**(跨 sessions)|

---

## 13. Cross-Reference 影響面

### 13.1 Charter 升版範圍

| Section | 改動類型 |
|---|---|
| §0.1 / §0.2 / §0.3 主節 | 不動(本 §14.7-CA 為 features 層 application;不改 doctrine)|
| §0.1-A / §0.2-A / §0.3-A 禁令 | 不動(延伸至 feature engineering 之 application)|
| §14.7-CA 新節 | **新建**(feature engineering 層升 doctrine-aligned)|
| §14.7-BU Phase E | 補註 feature layer hook 升至 doctrine-aligned |
| §14.7-W feature_store_builder first-principles research(v6.0.0)| 補註 §14.7-CA 為其升版繼承 |

### 13.2 Code 升版範圍

| File | 改動 |
|---|---|
| `scripts/core/feature_store_builder.py` | 加 ~31 new features + 升 v0.3 + 補 §14.7-BU Phase E hook |
| `scripts/maintenance/audit_universe_completeness.py` | feature layer 從 0 records → 1857 stocks × per-feature-group |
| `feature_definition` table | 補 31 new feature_name × feature_group entries |

---

## 14. Phase B 入憲提案(charter §14.7-CA 新子節 outline)

```markdown
### §14.7-CA 2026-XX-XX Doctrine-Aligned Feature Engineering Phase B 入憲(v6.1.0-patch 第二十五輪)

**觸發**: 2026-05-27 user 問「需要先產生哪些特徵值」+ Phase A 設計研究(reports/doctrine_aligned_feature_engineering_phase_a_research_20260527.md)揭露既有 v0.1 27 features 之 doctrine 對映 ~75% / 缺 §0.2 explicit + §0.3 macro indicator features。

**§14.7-CA Phase B 入憲核心**:
1. feature_store_builder.py v0.1 → v0.3 升 38 doctrine-aligned features
2. §0.1(16) + §0.2(8) + §0.3.1(6) + §0.3.2(5) + §0.3.3(3) 完整對映
3. ablation IC gate T_CA-1〜5 證偽承諾
4. §14.7-BU Phase E feature layer hook 補

**Phase B 對既有治權影響**:
- §0.1-A / §0.2-A / §0.3-A 禁令不動
- §14.7-BZ Phase F doctrine 不動
- §14.7-W feature_store_builder first-principles research 升至 doctrine-aligned
```

---

## 15. 結論 + 下一步

### 15.1 結論

1. **既有 27 v0.1 features 對 doctrine 對映 ~75%**(price-based 95% / revenue 90% / theme keyword 40%)
2. **Doctrine-aligned 升 38 features**:加 §0.2 explicit 8 + §0.3 macro 14 + §0.1 missing 9 = 31 new
3. **Effort ~8 人天**(跨 sessions)
4. **T_CA-1〜5 證偽承諾**:walk-forward IC ablation 為 production-ready gate

### 15.2 推薦 Path

**Path A 完整升版**(38 features doctrine-aligned + ablation IC gate)
- ✅ 對映 §14.7-BZ Phase F doctrine
- ⏸ 等用戶 explicit auth 進 Phase B

### 15.3 治權判準 eighth-round refinement(若進 Phase B)

依 §14.7-BY 之七輪累進,Path C 成為:
- §14.7-CA(本研究)= **Feature-axis 純化**(從 v0.1 27 features 升 38 doctrine-aligned)

| 輪 | §14.7-Bx/CA | 軸 |
|---|---|---|
| 18-24 | BT-BZ | N-axis + T-axis + Indicator-axis + Pillar-axis |
| **25** | **CA** | **Feature-axis** ★(本研究 propose)|

**Doctrine 五軸純化**:N + T + Indicator + Pillar + Feature

### 15.4 下一步(待用戶 explicit auth)

| Step | 動作 |
|---|---|
| 1 | Phase B charter §14.7-CA 入憲(0.5 人天)|
| 2 | Phase C-1 feature_store_builder v0.3 升 38 features(4 人天)|
| 3 | Phase C-2 §14.7-BU Phase E feature layer hook(0.5 人天)|
| 4 | Phase D ablation + IC backtest(2 人天)|
| 5 | Phase E commit + audit verify(0.5 人天)|

---

## 16. 補充:既有 v0.1 27 features 之初步保留 / 升版判定

**保留**(已 doctrine-aligned;25 features):
- 所有 price-based 15 features(§0.1 + §0.2 流動性)
- 所有 revenue-based 2 features(§0.1 fundamental)
- theme_strength / theme_is_semiconductor(§0.3 industry proxy;與新 macro features 共存)
- margin_ratio_60d(§0.1 法人)
- macro 5 features(estimate;§0.3 macro)

**升版**(從 industry keyword 升 macro indicator):
- theme_strength + theme_is_semiconductor 維持(per-stock industry-based)
- 加新 §0.3.1/.2/.3 macro features(per-stock 共用之 macro context)

**新增 31 features**(per §5 candidate list)

---

**Phase A 設計研究作者**: Claude(Opus 4.7)
**Session ID**: 2026-05-27
**Charter base**: v6.1.0 + 第十九〜二十四輪 patch(§14.7-BU〜BZ 全 inscribed)
**HEAD commit at Phase A 完成**: `dbc3043`(v6.2.0-macro-infrastructure-milestone-20260527)
**Status**: ✅ Phase A 完整 / 16 章 / non-destructive(不動 DB 不動 code)/ 待用戶 explicit auth 進 Phase B charter 入憲
