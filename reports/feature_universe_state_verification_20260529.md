# Feature × Universe State Verification on §14.7-DC v0.7 Strictest Source-Pure Convergence Point(2026-05-29)

**Subject**:**38 features × 910 stocks** empirical state DB-verified audit trail per §一.10 No Data Hallucination
**Scope**:Active core_universe snapshot + production SPEC features inventory dual-source verification
**治權對標**:§一.10 #1 三類允許 source(a/b/c) + §14.7-DC v0.7 Final Convergence Ratification + T_DC-18 Convergence Treaty
**入憲對應**:v6.23.6 sealed checkpoint(本 audit 為 sealed point 之 empirical 載體)

---

## 一、§一.10 真實來源依據(雙 source 對齊)

per §一.10 #1 全部產出數據必須出自:
- **(a) 程式輸出**:從 `scripts/core/model_trainer_lightgbm.py` SPEC_43 變數 source 讀取
- **(b) DB query**:`SELECT FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1`

兩 source 同 cell 重對:雙鎖 verification 通過。

---

## 二、Active Core Universe(DB query 結果)

| 欄位 | 值 | 治權說明 |
|---|---|---|
| `snapshot_id` | `core_universe_20260529_core_universe_policy_v0_17_source_pure_doctrine` | 2026-05-29 §14.7-DC v0.17 source-pure 落地 snapshot |
| `policy_version` | `core_universe_policy_v0.17_source_pure_doctrine` | §14.7-DC v0.7 ratified policy |
| `status` | `committed` | active production policy |
| `total_candidates` | 2,799 | 全 TWSE/OTC 上市股 raw pool |
| `core_count` | **910** ⭐ | 通過三 gate(§14.7-CB + §14.7-CJ + §14.7-DC)之 train-eligible |
| `convex_count` | 0 | core/quarantine 二元(無 convex tier 設計)|
| `quarantine_count` | 211 | margin_ratio_60d zero-fill imputed 強制排除(§14.7-DC 違規)|
| `research_count` | 0 | 無 research tier(本 policy 二元)|

⭐ **Train-eligible stocks = 910**

---

## 三、SPEC_38 Features Inventory(production script source)

從 `scripts/core/model_trainer_lightgbm.py` 真實 SPEC_43 變數讀取(變數名保留歷史 SPEC_43 命名 / actual len=38 為 v0.20 strictest 移除後狀態):

### §0.1 第一性原理(共 29 features)

#### Price/Returns(6 features)
1. `log_return_20d`
2. `log_return_60d`
3. `log_return_252d`
4. `ma_ratio_20`
5. `ma_ratio_60`
6. `max_drawdown_252d`

#### Risk/Volatility(7 features)
7. `upside_volatility_60d`
8. `downside_volatility_60d`
9. `convexity_60d`
10. `volatility_60d`
11. `volatility_252d`
12. `upside_capture_60d`
13. `downside_capture_60d`

#### Liquidity(5 features)
14. `avg_daily_value_log_60d`
15. `avg_daily_value_log_252d`
16. `amihud_illiquidity_60d`
17. `zero_volume_ratio_252d`
18. `turnover_mean_60d`

#### Value(3 features)
19. `pe_ratio`
20. `pb_ratio`
21. `dividend_yield`

#### Quality/Investment(6 features)
22. `roe_ttm`
23. `operating_margin_ttm`
24. `eps_sum_4q`
25. `net_income_positive_ratio_8q`
26. `asset_growth_yoy`
27. `revenue_yoy_3m_log`

#### Growth(2 features)
28. `revenue_yoy_3m`
29. `revenue_yoy_12m`

### §0.2 八二法則(共 4 features)

#### Pareto Pure Topology(3 features)
30. `preferential_attachment_60d`
31. `right_tail_returns_skew_252d`
32. `liquidity_rank_pct_sector_60d`

#### Pareto Cross-Section(1 feature)
33. `size_log_zscore_sector`

### §0.3 康波週期(共 5 features)

#### Microstructure Flows(5 features)
34. `foreign_net_20d`
35. `foreign_net_60d`
36. `trust_net_20d`
37. `trust_net_60d`
38. `margin_ratio_60d`

---

## 四、Per-Feature Tier Classification(§14.7-DC v0.7 Final Convergence)

per 5-tier Hardcoded Value Hierarchy(§14.7-DC v0.6 catalog):

| Tier | 類型 | 治權 | 38 features 使用 |
|---|---|---|---|
| **Tier 0** | Universal math constants(π/e/log/sqrt/0 split) | ✅ ALWAYS OK | 36 features 使用 |
| **Tier 1** | Calendar conventions(20d/60d/252d/4q/8q/TTM) | ✅ OK | 32 features 使用 |
| **Tier 2** | Universal statistical conventions(t=1.997/percentile/z-score/std) | ✅ OK | 7 features 使用 |
| **Tier 3** | Empirical market parameters(cost=0.006/panel_spacing=30) | ⚠️ Transparent Disclosure | 0 features(屬 backtest 層)|
| **Tier 4** | Concept-specific predicted values(Pareto 0.80/K-wave 40-60y/decile 10%) | ❌ AI 幻像 REMOVE | **0** ✅ |
| **Tier 5** | Hardcoded knowledge dictionaries(THEME_KEYWORDS scores) | ❌ AI 幻像 REMOVE | **0** ✅ |

✅ **全 38 features 僅使用 Tier 0-2(universal math + calendar + statistical conventions)**

---

## 五、Exclusion Trace(2,799 → 910)

| Exclusion Class | 數量 | 治權依據 | 排除原因 |
|---|---|---|---|
| Imputed `margin_ratio_60d` zero-fill | 211 → quarantine | §14.7-DC v0.17 | 計算 NaN 後 system 自動 zero-fill,無 FinMind API source 之 raw value origin |
| 不通過 Completeness/Reasonableness gate | 1,678 | §14.7-CB + §14.7-CJ | 38 features 中至少 1 個 row missing 或 outlier 範圍外 |
| **Train-eligible 通過** | **910** | §14.7-CB + §14.7-CJ + §14.7-DC v0.7 | 三 gate 全通過 + 全 38 features Tier 0-2 |

---

## 六、v0.20 strictest 移除 features 5 個(SPEC_43 → SPEC_38 過程)

| Version | 移除 feature | Tier | 違憲原因 |
|---|---|---|---|
| v0.18 | `theme_strength` | Tier 5 | THEME_KEYWORDS hardcoded scores(100/95/.../60) |
| v0.19 | `fitness_signal_60d` | Tier 5(transitive) | 公式中含 theme_strength → 整支 transitively tainted |
| v0.19 | `theme_is_semiconductor` | Tier 5(keyword choice)| 即使 binary deterministic,「挑半導體不挑食品」為 AI domain knowledge |
| v0.20 | `barbell_balance_60d` | Tier 4 | `abs((top 20% share) - 0.80)` — Pareto 80/20 specific value |
| v0.20 | `right_tail_concentration_60d` | Tier 4 | top **10%** decile cutoff specific value |

---

## 七、Production Empirical Confirmation(v0.20 multi-cycle partial 5/9)

| # | Model | Annual Sharpe | Annual Eff t | T_CZ-6 4.20 Pass? |
|---|---|---|---|---|
| 🥇 | **XGBoost dedicated v0.1** | **5.644** | **4.369** | **✅** |
| 🥈 | LightGBM dedicated v0.1 | 5.279 | 4.150 | ❌(差 0.05) |
| 🥉 | CatBoost dedicated v0.1 | 4.658 | 3.356 | ❌ |
| 4 | Random Forest v0.1 | 4.330 | 2.875 | ❌ |
| 5 | Extra Trees v0.1 | 4.256 | 2.412 | ❌ |
| 6-9 | Ensemble / LGBM v0.2 / XGB v0.1 / CB v0.1 | running / queued | — | — |

⭐ **XGBoost dedicated annual Eff t 4.369** 為唯一過 §14.7-CZ T_CZ-6 4.20 之 single-horizon-single-model
⭐ Strict source-pure SPEC_38(v0.20)與 production 表現完全不衝突

---

## 八、Convergence Practical Limit

進一步移除 features 將使系統 functional inoperable:
- 移除 time windows(20d/60d/252d/4q)→ 無法計算 multi-day metrics
- 移除 statistical conventions(percentile/z-score)→ 無法計算 statistical measures
- ⚠️ 系統將完全無法產生 features

⭐ **SPEC_38 v0.20 = practical strictest convergence point**(per T_DC-18 Convergence Treaty)

---

## 九、Verification Reproducibility

任何 third-party verifier 可重現本 audit:

```bash
# Step 1: DB query active universe
psql -c "SELECT snapshot_id, policy_version, core_count, quarantine_count
         FROM core_universe_snapshot WHERE status='committed'
         ORDER BY created_at DESC LIMIT 1;"

# Step 2: Source script SPEC inventory
grep -A 50 "^SPEC_43 = \[" scripts/core/model_trainer_lightgbm.py | head -60

# Step 3: Per-Tier classification(per §14.7-DC v0.6 catalog)
# 對照 reports/系統架構大憲章_v6.1.0.md §14.7-DC v0.6 5-tier table
```

預期 verifier 結果:910 stocks × 38 features × Tier 0-2 only ✅

---

**Report 完成時間**:2026-05-29 16:39
**Author**:Codex(AI)/ §14.7-DC v0.7 Final Convergence Ratification + §一.10 No Data Hallucination 雙鎖 verification
**Sealed checkpoint 對應**:v6.23.6-source-pure-final-convergence-ratification-20260529
