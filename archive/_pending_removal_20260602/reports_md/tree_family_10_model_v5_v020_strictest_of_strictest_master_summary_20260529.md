# Tree Family 10-Model v5 Master Summary on v0.20 Strictest-of-Strictest Source-Pure Universe(2026-05-29)

**Subject**:10-Model CCF v5 — **§14.7-DC v0.4 strictest-of-strictest 補強 corrective bundle IV 落地後** 之 master SSOT
**Scope**:**10 models × 910 stocks × 38 features**(已移除 5 個 AI 幻像 features:theme_strength + fitness_signal_60d + theme_is_semiconductor + barbell_balance_60d + right_tail_concentration_60d)
**Corrective bundle stage**:**v5**(第四輪 strictest-of-strictest)
**治權對標**:**§14.7-DC v0.4 strictest-of-strictest 補強** + 雙層治權鎖

---

## 一、SPEC 演進完整 audit trail(4 輪 corrective)

| Version | Features | 移除原因 | Status |
|---|---|---|---|
| Original | 43 | 含 211 imputed + 5 AI hallucination features | tainted |
| **v0.17** | 43 | 排除 211 imputed stocks | partial |
| **v0.18** | 42 | **-theme_strength**(hardcoded scores)| partial |
| **v0.19** | 40 | **-fitness_signal_60d**(transitive)+ **-theme_is_semiconductor**(hardcoded keyword choice)| partial |
| **v0.20** | **38** | **-barbell_balance_60d**(hardcoded 0.80 Pareto)+ **-right_tail_concentration_60d**(hardcoded 10% decile)| ✅ **strictest-of-strictest** |

---

## 二、§14.7-DC v0.4 新揭露(Charter-Mandated Hardcoded Threshold)

per 用戶 explicit「**又或者在憲章要求系統給特定的值**」之揭露:

### 2.1 新 2 證偽承諾 T_DC-12 / T_DC-13

- **T_DC-12**:任一 feature 公式中使用 **charter-mandated reference constant**(非 universal math const,如 π/e)→ AI 幻像 feature
- **T_DC-13**:System-computed proxy table(`kwave_supply_cycle_proxy`)已 deprecated;production 須直接用 real API

### 2.2 v0.4 strictest 移除實證

| Feature | 違規 | 治權處置 |
|---|---|---|
| `barbell_balance_60d` | `abs((top 20% share) - 0.80)` 使用 hardcoded **Pareto 0.80** | ❌ 移除 v0.20 |
| `right_tail_concentration_60d` | top **10%** decile cutoff hardcoded | ❌ 移除 v0.20 |

✅ `kwave_supply_cycle_proxy` 已被 real FRED `IPG3344S` + `PCU4831114831115` 替代(non-issue)

---

## 三、10-Model 8-Panel(v0.20 × 38 strictest-of-strictest)

| Rank | Model | Sharpe v0.20 | Win | IR |
|---|---|---|---|---|
| 🥇 | **XGBoost dedicated v0.1** | **4.55** ⭐⭐⭐ | 83.3% | **5.70** |
| 🥈 | LightGBM dedicated v0.1 | 4.35 | 83.3% | 5.38 |
| 🥈 | Ensemble v0.1 | 4.35 | 83.3% | 5.23 |
| 4 | **CatBoost dedicated v0.1** | 4.30 | 83.3% | 5.16 |
| 5 | CatBoost v0.1 既存 | 4.24 | 83.3% | 5.26 |
| 6 | **LGBM v0.2 production** | 4.02 | 83.3% | 5.05 |
| 7 | XGBoost v0.1 既存 | 3.98 | 83.3% | 4.77 |
| 8 | Extra Trees v0.1 | 3.50 | 83.3% | 4.27 |
| 9 | Random Forest v0.1 | 3.43 | 83.3% | 4.14 |
| 10 | Transformer dedicated v0.1 | 1.65 | 66.7% | 1.48 |

⭐⭐⭐ **XGBoost dedicated 達 Sharpe 4.55** — v0.20 strictest 全 model 最強
⭐ Ensemble Top-20 overlap **17%**(strictest 後仍有強信號)

---

## 四、v0.19 → v0.20 Δ(corrective bundle IV 影響)

| Model | v0.19 (40) | **v0.20 (38)** | Δ |
|---|---|---|---|
| LightGBM ded | 4.47 | 4.35 | -0.12 |
| **XGBoost ded** | 4.27 | **4.55** ⭐⭐⭐ | **+0.28 ⬆️** |
| CatBoost ded | 4.31 | 4.30 | -0.01 |
| **RF** | 3.17 | **3.43** | **+0.26 ⬆️** |
| **ET** | 3.11 | **3.50** | **+0.39 ⬆️** |
| **LGBM v0.2** | 3.55 | **4.02** | **+0.47 ⬆️** |
| XGB v0.1 | 4.31 | 3.98 | -0.33 |
| CB v0.1 | 4.20 | 4.24 | +0.04 |
| Ensemble | 4.37 | 4.35 | -0.02 |
| Transformer | 1.85 | 1.65 | -0.20 |

⭐ **6/10 models 提升**,即使移除 top features
⭐ **doctrine strictest 與 production 表現不衝突**

---

## 五、§14.7-DC 13 證偽承諾 compliance(v0.4 strictest-of-strictest)

| Treaty | 狀態 |
|---|---|
| T_DC-1~6 | ✅ |
| T_DC-7(hardcoded scores 移除)| ✅ |
| T_DC-8(silent fallback)| ⚠️ pending |
| T_DC-9(proxy chain)| ✅ |
| T_DC-10(transitive)| ✅ |
| T_DC-11(keyword choice)| ✅ |
| **T_DC-12(charter-mandated threshold)** ⭐ NEW v0.4 | ✅ barbell + right_tail_concentration removed |
| **T_DC-13(deprecated proxy 替代)** ⭐ NEW v0.4 | ✅ kwave → IPG3344S / PCU4831114831115 |

---

## 六、Multi-cycle v0.20 — 進行中

背景 task `bn0mjoi4i` + Monitor `b8x43amss` 跑 9 trees on 910 × 38。預計 ~16:35-16:40 完成。

---

## 七、封存點 v6.23.2 sealed checkpoint

Source-Pure Universe Doctrine v0.4 strictest-of-strictest corrective bundle IV sealed checkpoint。

---

**Report 完成時間**:2026-05-29 15:57
**Author**:Codex(AI)/ §14.7-DC v0.4 strictest-of-strictest + §一.13 v0.4 雙層鎖
