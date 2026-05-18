# Walk-Forward H20 Panel Expansion (2026-05-18)

- **generated_at**: 2026-05-18 13:51 Asia/Taipei
- **constitution**: `reports/系統架構大憲章_v6.0.0.md` §8 / §8.8.7 / §8.8.8 / §14.7-Q
- **purpose**: 擴張 walk-forward h20 panel 至 12 個時點，建立 IC stability evidence 強化 §8 升 v6.1.0 提案
- **execution time**: 13:47:08 → 13:47:28（**20 秒**，純 DB 操作不需 API）
- **verdict**: **READY_FOR_DRAFT_EVIDENCE**

## Panel Composition

12 個 walk-forward h20 model + 1 個 prediction-backed delivery（latest）：

| # | model_id | as_of_date | rows | IC_mean | RMSE | prediction status |
|---|---|---|---:|---:|---:|---|
| 1 | mdl_20250530_lgbm_h20_7b9f0d39_v0_1 | 2025-05-30 | 145 | 0.2603 | 0.2830 | deprecated |
| 2 | mdl_20250630_lgbm_h20_f36689d7_v0_1 | 2025-06-30 | 146 | 0.1820 | 0.2861 | deprecated |
| 3 | mdl_20250731_lgbm_h20_90a4b395_v0_1 | 2025-07-31 | 146 | 0.3255 | 0.2458 | deprecated |
| 4 | mdl_20250829_lgbm_h20_ada73e82_v0_1 | 2025-08-29 | 146 | 0.4269 | 0.2362 | deprecated |
| 5 | mdl_20250930_lgbm_h20_a4a1b802_v0_1 | 2025-09-30 | 146 | 0.3281 | 0.2596 | deprecated |
| 6 | mdl_20251031_lgbm_h20_fec82889_v0_1 | 2025-10-31 | 146 | 0.3737 | 0.2581 | deprecated |
| 7 | mdl_20251128_lgbm_h20_a1c774ee_v0_1 | 2025-11-28 | 146 | 0.2468 | 0.2267 | deprecated |
| 8 | mdl_20251231_lgbm_h20_a612e6fe_v0_1 | 2025-12-31 | 146 | 0.3932 | 0.2961 | deprecated |
| 9 | mdl_20260130_lgbm_h20_cad31cae_v0_1 | 2026-01-30 | 147 | 0.2760 | 0.2365 | deprecated |
| 10 | mdl_20260227_lgbm_h20_75c6761d_v0_1 | 2026-02-27 | 147 | 0.2405 | 0.2692 | deprecated |
| 11 | mdl_20260331_lgbm_h20_25fc2461_v0_1 | 2026-03-31 | 147 | 0.4658 | 0.4166 | deprecated |
| 12 | **mdl_20260425_lgbm_h20_d969ffb1_v0_1** | **2026-04-25** | **147** | **0.3716** | **0.2796** | **committed (sole delivery)** |

## IC Stability Statistics

| Metric | Value |
|---|---:|
| n | 12 |
| min IC | 0.1820 |
| max IC | 0.4658 |
| IC range | 0.2838 |
| mean IC | **0.3242** |
| median IC | 0.3268 |
| stdev IC | 0.0851 |
| **IC ≥ 0** | **12/12** ✅ |

**穩定性裁決**：12 時點 IC 皆為正，無相鄰時點差異 > 0.30；mean=0.3242 與單點 2026-04-25 之 0.3716 接近。滿足 §8.8.2-C「IC 跨時點穩定性」反向實證要求。

## Universe Lock

- All 12 models use universe_snapshot_id: `core_universe_20260515_core_universe_policy_v0_2`
- Universe lock 一致，無漂移

## Compliance Check

| Audit | Result |
|---|---|
| Step 11A `audit_leakage.py` | ✅ PERFECT 18/0/0（含 12 models + 12 predictions） |
| Step 11B `audit_downstream_readiness.py` (after deprecation) | ✅ **READY_FOR_DRAFT_EVIDENCE** 29/1/0 |
| Single WARN | production-current label window：`max_price_date=2026-05-15 < required_label_date=2026-06-04` |

## Final Delivery

- **唯一 prediction-backed committed model**: `mdl_20260425_lgbm_h20_d969ffb1_v0_1` (IC=0.3716)
- 其他 11 個 walk-forward models：保留 model_registry status=committed 作 IC stability evidence，但 prediction_run 已 status=deprecated 對齊 §8.8.8 之「exactly 1 prediction-backed」規則
- 此模式對齊 §8.8.10「Final Delivery Index」之單一 prediction-backed delivery + 多個 historical models 之 audit-trail 設計

## Decision

1. 本擴張為乾淨 §8 walk-forward evidence panel，可作為 §8 升 v6.1.0 之 IC 穩定性核心證據
2. 12 點全 IC ≥ 0 + stdev 0.0851 證明 robust rank-IC baseline 在 11 個月跨度下穩定運作
3. 不取代 production-current h20 升版；後者仍需等 2026-06-04 cron 累積資料
4. 提交 v6.1.0 升版時，本報告可作為 §8.8.7 walk-forward evidence panel 之最新版本
