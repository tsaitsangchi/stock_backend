# Phase C-1c Cross-Session Planning — Doctrine-Aligned Features 分批落地

**日期**: 2026-05-27
**對應 charter**: §14.7-CA(v6.2.1)/ Phase A research(v6.2.0.1)/ IC mapping(v6.2.1.2)/ IC supplement(v6.2.1.3)
**前置基線**: v6.2.1.5(Phase C-1c partial:convexity_60d + amihud_illiquidity_60d 已加)
**Status**: ✅ Planning document(non-destructive)

---

## 1. Phase C-1c 完整 scope

依 §14.7-CA Phase A research §5 之 v0.3 spec 38 features,vs v0.4 既有 35 features + 本 session 加 2 = 37,**真正 missing ~25 features 待跨 session 加**(實際 ~28 因部分 overlap 細節)。

| 階段 | Features | Effort |
|---|---|---|
| Phase C-1c partial(本 session v6.2.1.5)| +2 | ~1.5 hour |
| **Phase C-1c remaining 跨 session** | **+25** | **~7.5 人天** |

---

## 2. 4 Sub-phases 分批策略

依 risk-adjusted prioritization(highest-IC + lowest-effort + lowest-dependency 優先):

### Sub-phase C-1c-1 — §0.1 Value(3 features)

| Feature | Sign | Lit IC | New raw load |
|---|:---:|:---:|---|
| `pe_ratio` | − | -0.04(TW)| ✅ `_load_per`(TaiwanStockPER)|
| `pb_ratio` | − | -0.05(TW Fama-French) | 同上 |
| `dividend_yield` | + | +0.02(Litzenberger 1979)| ✅ `_load_dividend`(TaiwanStockDividend)|

- **DB write**: feature_definition 37→40 / feature_values +5,571 rows
- **Effort**: ~1.5 人天
- **Tag**: `v6.2.2-phase-c1c-1-value-features-20260XXX`

### Sub-phase C-1c-2 — §0.1 Quality + Investment(4 features)

| Feature | Sign | Lit IC | New raw load |
|---|:---:|:---:|---|
| `roe_ttm` | + | **+0.07(TW;Asness QMJ)** | ✅ `_load_balance_sheet`(TaiwanStockBalanceSheet)|
| `operating_margin_ttm` | + | +0.05 | 同上 + FinancialStatements |
| `revenue_yoy_3m_log` | + | +0.04 | ❌ reuse MonthRevenue |
| `asset_growth_yoy` | − | -0.05(Cooper 2008)| ✅ `_load_balance_sheet` |

- **DB write**: feature_definition 40→44 / feature_values +7,428 rows
- **Effort**: ~1.5 人天
- **Tag**: `v6.2.3-phase-c1c-2-quality-investment-features-20260XXX`

### Sub-phase C-1c-3 — §0.3 Macro(14 features:K-wave 6 + Multi-cycle 5 + Microstructure 3)

| Pillar | Features | Strongest IC |
|---|:---:|:---:|
| **§0.3.1 K-wave pure** | 6 | kwave_credit_to_gdp_gap(-0.02 BIS)|
| **§0.3.2 Multi-cycle** | 5 | **mc_yield_curve_inversion(-0.04;Estrella 30+ 年 OOS 最強 macro)** |
| **§0.3.3 Microstructure** | 3 | ms_volatility_regime / ms_market_stress(-0.025;Whaley)|

- **Raw load**: ❌ reuse 既有 _load_macro(升 5 → 13 indicators per §14.7-BY/BZ)
- **DB write**: feature_definition 44→58 / feature_values +25,998 rows
- **Effort**: ~2.5 人天
- **Tag**: `v6.2.4-phase-c1c-3-macro-features-20260XXX`

### Sub-phase C-1c-4 — §0.2 八二法則 explicit(7 features;amihud 已 in partial)

| Feature | Sign | Lit IC | Notes |
|---|:---:|:---:|---|
| `right_tail_concentration_60d` | + | +0.015 | per-sector aggregation |
| `barbell_balance_60d` | + | conjecture | per §9.2 barbell |
| `preferential_attachment_60d` | + | +0.015 | Barabási-Albert 1999 |
| `fitness_signal_60d` | + | +0.02 | Bianconi-Barabási 2001 |
| `right_tail_returns_skew_252d` | ± | ±0.02 | regime-dependent |
| `liquidity_rank_pct_sector_60d` | + | +0.015 | sector percentile |
| `size_log_zscore_sector` | − | ±0.01 TW(emerging market regime risk)| Fama-French SMB |

- **Raw load**: ❌ 全 reuse 既有 PriceAdj + Info + InstFlow
- **Implementation challenge**: per-sector aggregation(較複雜)
- **DB write**: feature_definition 58→65 / feature_values +13,000 rows
- **Effort**: ~2 人天
- **Tag**: `v6.2.5-phase-c1c-4-pareto-features-20260XXX`

---

## 3. Total Effort + DB state predictions

### 3.1 Effort 總計

| Sub-phase | Features | Effort |
|---|---:|---|
| C-1c-1 §0.1 Value | 3 | ~1.5 人天 |
| C-1c-2 §0.1 Quality+Investment | 4 | ~1.5 人天 |
| C-1c-3 §0.3 Macro(K+M+μ)| 14 | ~2.5 人天 |
| C-1c-4 §0.2 八二 | 7 | ~2 人天 |
| **Sub-phases 全 closure** | **28(net 25 unique)** | **~7.5 人天** |
| Phase D ablation | — | ~2 人天 |
| Phase E audit + v6.3.0 milestone | — | ~0.5 人天 |
| **Total cross-session** | — | **~10 人天** |

### 3.2 DB state predictions

| Sub-phase | feature_definition | feature_values rows |
|---|---:|---:|
| v6.2.1.5(本 session)| 37 | 60,604 |
| Post C-1c-1 | 40 | ~66,175 |
| Post C-1c-2 | 44 | ~73,603 |
| Post C-1c-3 | 58 | ~99,601 |
| Post C-1c-4 | **65** | **~112,601** |

---

## 4. Cross-Session Execution Protocol

### 4.1 Next session entry

```bash
git fetch --all --tags
git checkout v6.2.1.5-phase-c1c-2-features-partial-20260527

# 讀本規劃 + 必要 references
cat reports/phase_c1c_cross_session_planning_20260527.md
cat reports/doctrine_aligned_feature_engineering_phase_a_research_20260527.md  # §5
cat reports/feature_ic_literature_mapping_phase_a_research_20260527.md  # §4-8

# 進 Phase C-1c-1(per planning §2)
```

### 4.2 Per sub-phase 5-step protocol

1. Read § sub-phase scope(本 planning §2 對應 sub-phase)
2. 寫新 `_load_*` + `_compute_*` methods
3. 加 FEATURE_DEFINITIONS entries(per features spec)
4. Dry-run + production commit + audit verify
5. Commit + tag + push

### 4.3 Recommended session 分配

| Session # | Sub-phase | 預估 |
|---|---|---|
| Next(+1)| C-1c-1 Value | ~1.5 人天 |
| +2 | C-1c-2 Quality+Investment | ~1.5 人天 |
| +3 | C-1c-3 Macro(可拆 +3a/+3b)| ~2.5 人天 |
| +4 | C-1c-4 Pareto | ~2 人天 |
| +5 | Phase D ablation | ~2 人天 |
| +6 | Phase E audit + v6.3.0 milestone | ~0.5 人天 |

---

## 5. 證偽承諾 gates(per §14.7-CA Phase B + IC supplement)

| Gate | Threshold | 對應 sub-phase |
|---|---|---|
| T_CA-1 | v0.3 ensemble IC ≥ v0.4 baseline + 0.02 | Phase D 全跑通後 |
| T_CA-2 | §0.2 explicit IC > 0 | C-1c-4 後 |
| T_CA-3 | §0.3.1 K-wave × stock IC > §0.3 industry baseline | C-1c-3 後 |
| T_CA-4 | T1 features 不降 IC | 每 sub-phase regression check |
| T_CA-5 | walk-forward IC stdev ≤ v0.1 × 1.5 | Phase D 後 |
| T_CA-IC-1 | Top-3 features IC ≥ +0.03 | amihud(已)+ roe(C-1c-2)+ yield_curve(C-1c-3)|
| T_CA-IC-2 | mc_yield_curve_inversion IC < -0.03 | C-1c-3 |
| T_CA-IC-3 | TW size effect IC ∈ [-0.03, +0.02] | C-1c-4 |
| T_CA-IC-4 | Cross-pillar interaction IC ≥ +0.02 | post Phase D |
| T_CA-IC-5 | v0.3 ensemble IC ≥ 0.06 | Phase D 全跑通後 |

---

## 6. Risk Assessment

### 6.1 Engineering risks

| Risk | Mitigation |
|---|---|
| `_load_balance_sheet` SQL 之 publication-date gate | reuse `data_schema.build_publication_date_gate` SSOT helper(v2.20)|
| 大量 features × 1857 × 多年 = ~大 storage | feature_values 已 indexed;Postgres 可承受 ~10-15 GB |
| Per-sector compute join complexity | SQL window functions / Python groupby |

### 6.2 Doctrine risks

| Risk | Mitigation |
|---|---|
| T3 元素誤入 features | Phase A research §7 已驗證全 T1+T2;每 sub-phase 落地時 audit |
| Cross-pillar interaction 過擬合 | ablation IC ≥ +0.02 gate(per T_CA-1)|

### 6.3 IC drift risks

| Risk | Mitigation |
|---|---|
| 累積 features 後 IC 反而 dilute | 識別 IC < +0.005 features 並 deprecate |
| TW-specific size reverse | C-1c-4 之 size_log_zscore_sector 之 ablation 留意 |

---

## 7. Phase E Audit & Phase G Milestone

### 7.1 audit_universe_completeness feature layer 升版(post C-1c)

- 既有 11,142 records(v6.2.1.4)
- C-1c 後 expected_items per pillar 升版(per §14.7-BU Phase E hook 之 hardcoded broadcast 可調整)
- 1857 stocks × ~65 expected = ~120,705 expected records

### 7.2 Phase G v6.3.0 milestone

post Phase E audit PERFECT,可 tag `v6.3.0-feature-axis-purification-milestone-20260XXX` 作為 Feature-axis 純化完整閉環。

---

## 8. Cross-Reference 影響面

### 8.1 Charter
- §14.7-CA 主節 + IC Supplement 不動(已 inscribed v6.2.1 / v6.2.1.3)
- Phase E closure 後可加 milestone supplement(若 IC 符合 T_CA-1/IC-1〜5)

### 8.2 Code
- `feature_store_builder.py`:+3 `_load_*` / +6 `_compute_*` / +28 FEATURE_DEFINITIONS / ~700-1,000 行
- `audit_universe_completeness.py`:feature layer expected count 升版(若需)

### 8.3 DB
- 0 schema 變動(全 reuse 既有 feature_values / feature_definition)
- 累計 ~52,000 new rows in feature_values

---

## 9. 結論 + 推薦執行順序

### 9.1 完整 closure path

```
Phase A research(v6.2.0.1)
Phase B charter(v6.2.1)
Phase A IC research(v6.2.1.2)
Phase B IC supplement(v6.2.1.3)
Phase C-1a hook(v6.2.1.1)
Phase C-1b baseline(v6.2.1.4)
Phase C-1c partial(v6.2.1.5)← 當前 HEAD
Phase C-1c-1(v6.2.2)← Next session
Phase C-1c-2(v6.2.3)
Phase C-1c-3(v6.2.4)
Phase C-1c-4(v6.2.5)
Phase D ablation(v6.2.6)
Phase E audit(v6.2.7)
Phase G v6.3.0 Feature-axis milestone
```

### 9.2 推薦 priority

1. Next session:**C-1c-1**(quick win;low risk;3 Value features)
2. +1:**C-1c-2**(含 roe_ttm 之 Top-2 strongest IC)
3. +2:**C-1c-3**(含 mc_yield_curve_inversion 之 Top-1 IC;可拆 +3a/+3b)
4. +3:**C-1c-4**(複雜度最高 / 留最後)
5. +4:**Phase D ablation** all sub-phases together
6. +5:**Phase E audit** + v6.3.0 milestone

---

**Planning 作者**: Claude(Opus 4.7)
**Session ID**: 2026-05-27
**Charter base**: v6.1.0 + 第十九〜二十五輪 patch(§14.7-BU〜CA inscribed)
**HEAD at planning 完成**: `982b025`(v6.2.1.5-phase-c1c-2-features-partial-20260527)
**Status**: ✅ Planning 完整 / non-destructive / Cross-session execution-ready
