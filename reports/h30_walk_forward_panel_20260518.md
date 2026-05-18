# H30 Walk-Forward Panel (2026-05-18)

- **generated_at**: 2026-05-18 14:08 Asia/Taipei
- **constitution**: §9.1 / §14.7-S
- **purpose**: 強化 §9.1 v6.2.0 升版前置之 IC 跨時點穩定性證據
- **verdict**: READY_FOR_DRAFT_EVIDENCE

## Scope

- horizon: 30 (calendar days)
- timepoints: 12 monthly (2025-04-30 → 2026-03-31)
- universe_snapshot: `core_universe_20260515_core_universe_policy_v0_2`
- 執行時間: ~20 秒 (純 DB，無 API)

## Panel

| # | model_id | as_of | rows | IC | RMSE | label_date | prediction |
|---|---|---|---:|---:|---:|---|---|
| 1 | mdl_20250430_lgbm_h30_4c18a33d_v0_1 | 2025-04-30 | 144 | 0.3731 | 0.2081 | 2025-06-02 | deprecated |
| 2 | mdl_20250530_lgbm_h30_ceba8c11_v0_1 | 2025-05-30 | 145 | 0.3218 | 0.2696 | 2025-06-30 | deprecated |
| 3 | mdl_20250630_lgbm_h30_429f5d89_v0_1 | 2025-06-30 | 146 | 0.2779 | 0.2909 | 2025-07-30 | deprecated |
| 4 | mdl_20250731_lgbm_h30_06b1d9d7_v0_1 | 2025-07-31 | 146 | 0.2545 | 0.2229 | 2025-09-01 | deprecated |
| 5 | mdl_20250829_lgbm_h30_05c07bc1_v0_1 | 2025-08-29 | 146 | 0.3468 | 0.2373 | 2025-09-30 | deprecated |
| 6 | mdl_20250930_lgbm_h30_880c02ca_v0_1 | 2025-09-30 | 146 | 0.2992 | 0.2461 | 2025-10-30 | deprecated |
| 7 | mdl_20251031_lgbm_h30_ee782506_v0_1 | 2025-10-31 | 146 | 0.2843 | 0.2453 | 2025-12-01 | deprecated |
| 8 | mdl_20251128_lgbm_h30_f5825d31_v0_1 | 2025-11-28 | 146 | 0.1978 | 0.2673 | 2025-12-29 | deprecated |
| 9 | mdl_20251231_lgbm_h30_1ede4261_v0_1 | 2025-12-31 | 146 | 0.4043 | 0.3028 | 2026-01-30 | deprecated |
| 10 | mdl_20260130_lgbm_h30_acaead63_v0_1 | 2026-01-30 | 147 | 0.3113 | 0.2330 | 2026-03-02 | deprecated |
| 11 | mdl_20260227_lgbm_h30_9f9ddfbd_v0_1 | 2026-02-27 | 147 | 0.2786 | 0.2190 | 2026-03-30 | deprecated |
| 12 | mdl_20260331_lgbm_h30_e5d97ec0_v0_1 | 2026-03-31 | 147 | 0.4090 | 0.4467 | 2026-04-30 | deprecated |

## IC Stability

| Metric | h30 panel | h20 panel (§14.7-Q 對比) |
|---|---:|---:|
| n | 12 | 12 |
| min IC | 0.1978 | 0.1820 |
| max IC | 0.4090 | 0.4658 |
| mean IC | **0.3132** | 0.3242 |
| median IC | 0.3053 | 0.3268 |
| **stdev IC** | **0.0622** | 0.0851 |
| IC ≥ 0 | **12/12** ✅ | 12/12 ✅ |

**重要發現**：h30 panel 之 IC stdev (0.0622) **比 h20 panel (0.0851) 更低**，顯示更長 horizon 的訊號雜訊比更高，穩定性更佳。對 §9.1 終極目標而言是強力 evidence。

## Compliance

| Audit | Result |
|---|---|
| `audit_leakage v0.2` | ✅ 18/0/0 PERFECT (含 12 h30 models + 12 h20 models) |
| `audit_downstream_readiness v0.2` | ✅ 29/1/0 READY_FOR_DRAFT_EVIDENCE |
| Sole WARN | production-current label window |

## Final Delivery State

- **h20 prediction-backed delivery**: `pred_20260425_*` (唯一 committed)
- **h30 walk-forward models**: 12 個 committed in `model_registry`（evidence panel）
- **h30 predictions**: 全 12 個 status=deprecated（§8.8.8 對齊）

## Decision

1. §9.1 v6.2.0 升版前置 IC 穩定性證據**已 ready** — h30 panel 與 h20 panel 等量、且更穩定
2. h20 panel + h30 panel 共 24 個 walk-forward models，跨 11 個月 + 12 個月跨度
3. 下次 v6.2.0 升版提案可直接引用本 panel 作為 horizon=30 之主要證據
4. 當 production-current h20 升 v6.1.0 後（2026-06-04+），h30 也應立刻有對應的 production-current model commit
