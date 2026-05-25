# P1 v0.1 公式對齊 ablation evidence — RMS vs STDDEV

- **tool**: `ablation_rms_vs_stddev_20260525.py vv0.1`
- **constitution**: v6.1.0
- **as_of_date**: 2026-05-21
- **lookback**: 90d (~60 交易日);min_obs=20
- **n_stocks**: 2688
- **公式比對**: STDDEV (builder v0.7 / §14.7-BG / §9.10 起草) vs RMS (§9.9 強制契約)

## 1. Numerical 差異 (RMS − STDDEV)

| 變數 | mean | median | stdev | abs_mean | abs_p95 | min | max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `up_sigma_diff (RMS - STDDEV)` | +0.1728 | +0.1359 | 0.1318 | 0.1728 | 0.4474 | +0.0123 | +1.0299 |
| `down_sigma_diff (RMS - STDDEV)` | +0.1551 | +0.1354 | 0.1001 | 0.1551 | 0.3410 | +0.0118 | +1.4998 |
| `convexity_diff (RMS - STDDEV)` | +0.0177 | +0.0034 | 0.0721 | 0.0434 | 0.1466 | -1.3613 | +0.5553 |

## 2. Rank Correlation (Spearman ρ)

- `convexity_stddev` vs `convexity_rms`: **ρ = 0.9003**
- `score_stddev` vs `score_rms`:         **ρ = 0.8816**

## 3. Score 差異 (5 階梯 mapping 後)

- 完全相同:    **1751 / 2688 (65.1%)**
- 有差異:      937
- mean |Δscore|: 5.79
- max |Δscore|:  75.0

## 4. Top-120 核心股名單 (依 score 排序)

- STDDEV ∩ RMS: **88/120 (73.3%)**
- Jaccard:      0.5789

## 5. Top-10 |Δconvexity| Outliers

| stock_id | conv_STDDEV | conv_RMS | Δconv | score_S | score_R | n_obs | n_up | n_down | mean_up | mean_down |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6236 | -0.4547 | -1.8160 | -1.3613 | 20 | 20 | 58 | 3 | 2 | +0.02734 | -0.14384 |
| 6820 | +0.5173 | +1.0726 | +0.5553 | 95 | 95 | 58 | 26 | 32 | +0.12613 | -0.06554 |
| 6990 | +1.2569 | +1.7530 | +0.4961 | 95 | 95 | 58 | 27 | 31 | +0.14549 | -0.06613 |
| 3585 | +0.6379 | +1.1338 | +0.4959 | 95 | 95 | 58 | 28 | 30 | +0.13181 | -0.06878 |
| 3049 | +0.0020 | +0.4443 | +0.4423 | 75 | 95 | 58 | 22 | 35 | +0.07015 | -0.03796 |
| 3498 | +0.1092 | +0.5332 | +0.4240 | 95 | 95 | 58 | 29 | 28 | +0.06367 | -0.02772 |
| 4573 | +0.4286 | +0.8481 | +0.4195 | 95 | 95 | 58 | 29 | 28 | +0.11164 | -0.06499 |
| 5248 | +1.4782 | +1.8663 | +0.3881 | 95 | 95 | 58 | 19 | 34 | +0.13162 | -0.05530 |
| 3184 | +0.5243 | +0.9121 | +0.3878 | 95 | 95 | 58 | 23 | 32 | +0.09227 | -0.04384 |
| 7839 | +0.1352 | -0.2454 | -0.3806 | 95 | 20 | 58 | 16 | 10 | +0.02182 | -0.04208 |

## 6. 裁決方向 (依實證讀數)

依以下指標強弱,本 ablation 之 evidence 對應 §9.9 vs §14.7-BG/§9.10 公式裁決:

- **若 ρ_score ≥ 0.95 且 Top-120 overlap ≥ 95%**: STDDEV / RMS 排名一致 → 選項 C(治權成本最低)可採;但仍應在 §14.7-BG 加註 STDDEV 為 fast-track 近似
- **若 ρ_score ∈ [0.80, 0.95) 或 Top-120 overlap ∈ [80%, 95%)**: 兩公式 rank 有差異 → 選項 B(雙公式 + ablation IC 後選主軸)為合理裁決
- **若 ρ_score < 0.80 或 Top-120 overlap < 80%**: 兩公式產生顯著不同的核心股名單 → 選項 A(以 §9.9 RMS 為治權 SSOT;追溯修正 builder v0.7 → v0.7.1 RMS)為強制裁決

### 本次實證讀數

| 指標 | 讀數 | 區間判定 |
| :--- | ---: | :--- |
| ρ_convexity (Spearman) | **0.9003** | [0.80, 0.95) → B |
| ρ_score (Spearman) | **0.8816** | [0.80, 0.95) → B |
| **Top-120 overlap** | **88/120 = 73.3%** | **< 80% → A** |
| Jaccard (核心股名單) | 0.5789 | 中低 |
| Score 完全相同率 | 65.1% | 中等 |
| mean \|Δscore\| | 5.79 分 | 中等 |
| **max \|Δscore\|** | **75 分** (stock 7839: 95→20) | **boundary 極端 sensitive** |

### 裁決:強烈傾向 **選項 A**(以 §9.9 RMS 為治權 SSOT;追溯修正 builder v0.7)

**5 條治權層支持**:

1. **Top-120 overlap = 73.3% < 80% threshold**:本指標代表「實際 production 核心股名單會發生 27 stocks 進出」(120 - 88 = 32 名單異動,且 27 stocks 會被換掉),已落入「兩公式產生顯著不同核心股名單」之強制裁決區間。
2. **boundary 極端 sensitive(max |Δscore| = 75 分)**:stock 7839 之 convexity 從 STDDEV +0.1352(score 95) → RMS −0.2454(score 20),這是 **sign flip** 兼 **跨 4 個 boundary**;代表 STDDEV 公式會把「正報酬比負報酬多但散度更大的個股」誤判為高凸性,而 RMS 公式(對絕對水平敏感)正確捕捉到負報酬的絕對水平更大。
3. **DB 中尚無 v0.6 snapshot(剛剛實證查 DB:`Existing snapshots: [('core_universe_policy_v0.2',)]`)**:builder v0.7 從未在 production 跑出 v0.6 snapshot,**現在改公式對 production 零影響**;若 v0.6 snapshot 已落地後再改公式,則會引入 §6.8 annual_rebalance_guard 之 churn rate 不必要噪訊。
4. **治權位階 §9.9 > §9.10**:§9.9 為「**強制契約**」(L5125-5245),§9.10 為「**forward reference 起草**」(L8449+);若兩者衝突,§9.9 必勝。§14.7-BG 之 STDDEV 公式應視為「fast-track 試錯」而非治權升版 — 既然 ablation 已實證 STDDEV 與 §9.9 RMS 不等價,fast-track 試錯之 finding 應追溯修正而非升正式條文。
5. **金融學標準對齊**:Sortino ratio 之 downside deviation 採 √(Σmin(r−MAR,0)²/n)(MAR=0) = RMS-based;§9.9 公式對齊國際金融學主流定義。

### 選項 A 落地路徑(若用戶授權)

| 步驟 | 對象 | 內容 |
|---|---|---|
| 1 | `core_universe_builder.py v0.7 → v0.7.1` | `_load_market_data` 之 SQL:`STDDEV(lr) FILTER ...` → `SQRT(AVG(lr*lr)) FILTER ...`;score_scope `v0.7_VC_convexity_aligned` → `v0.7.1_VC_convexity_aligned_rms`;DEFAULT_POLICY_VERSION 不動(仍 v0.6) |
| 2 | `audit_core_universe.py v0.2` | `POLICY_SCORE_SCOPE_MAP["core_universe_policy_v0.6"]` 升 `v0.7.1_VC_convexity_aligned_rms`;EXPECTED_SCORE_DETAIL_KEYS 不動 |
| 3 | 憲章 §9.10 | 從「forward reference 起草」升「**正式條文**」;公式 STDDEV → **RMS** ;明文 §9.10 = §9.9 之 builder-layer 落地對齊 |
| 4 | 憲章 §14.7-BG | 補註「STDDEV 公式為 v0.7 fast-track 試錯;ablation 揭露不等價於 §9.9 RMS;v0.7.1 起追溯修正」;raw-first 路徑作為**架構選項**保留(不退治權) |
| 5 | 憲章新增 §14.7-BH | 「P1 v0.1 公式對齊 ablation 完成 — RMS 為治權 SSOT」子節;記錄本次 ablation 之 evidence + 「公式層揭露」第 7 次跑通(對映 §14.7-AX) |
| 6 | feature_store_builder v0.5 | **不動**(本就已對齊 §9.9 RMS;v0.6 升級延後) |

## 7. 治權交叉引用
- 憲章 §9.9 (P1 v0.1 強制契約 — RMS 公式)
- 憲章 §9.10 (起草 — VC 升版 STDDEV 公式;待 §9.9 ablation 後升正式條文)
- 憲章 §14.7-BG (raw-first fast-track 路徑入憲)
- 憲章 §0.0-C.3 (上行凸性壓制修補)
- 憲章 §6.3 第 7 條 (VC 公式條文原文)
- 憲章 §14.7-AX (資料層揭露驅動治權升版元規則 — 本次屬「公式層揭露」第 7 次跑通)

## 8. 對映 JSON 完整資料
- `p1_v01_rms_vs_stddev_ablation_data_20260525_1604.json`
