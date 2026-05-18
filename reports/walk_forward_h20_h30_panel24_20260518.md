# H20/H30 Walk-Forward Panel24 Expansion (2026-05-18)

- generated_at: 2026-05-18 14:46 Asia/Taipei
- constitution: `reports/系統架構大憲章_v6.0.0.md` §8 / §9 / §14.7-T
- purpose: expand h20 and h30 historical evidence from 12 points to 24 points each using existing data only
- verdict: READY_FOR_DRAFT_EVIDENCE

## Scope

- data gate: all label dates are `<= 2026-05-15`
- h20 added points: 2024-05-31 -> 2025-04-30
- h30 added points: 2024-04-30 -> 2025-03-31
- final committed historical models: 48 total
- final prediction-backed delivery: `pred_20260425_mdl_20260425_lgbm_h20_d969ffb1_v0_1`
- evidence-only predictions deprecated this run: 24

## IC Stability Summary

| Horizon | n | first_as_of | last_as_of | min IC | max IC | mean IC | median IC | stdev IC | IC >= 0 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| h20 | 24 | 2024-05-31 | 2026-04-25 | 0.1820 | 0.5184 | 0.3530 | 0.3718 | 0.0848 | 24/24 |
| h30 | 24 | 2024-04-30 | 2026-03-31 | 0.1978 | 0.5889 | 0.3482 | 0.3276 | 0.0923 | 24/24 |

## H20 Panel

| # | model_id | as_of | rows | IC | RMSE | label_date | prediction |
|---|---|---|---:|---:|---:|---|---|
| 1 | mdl_20240531_lgbm_h20_dc48a80d_v0_1 | 2024-05-31 | 142 | 0.3721 | 0.2893 | 2024-06-20 | deprecated |
| 2 | mdl_20240628_lgbm_h20_7067505f_v0_1 | 2024-06-28 | 143 | 0.1983 | 0.2334 | 2024-07-18 | deprecated |
| 3 | mdl_20240731_lgbm_h20_35f37f76_v0_1 | 2024-07-31 | 143 | 0.3341 | 0.2785 | 2024-08-20 | deprecated |
| 4 | mdl_20240830_lgbm_h20_13eae811_v0_1 | 2024-08-30 | 142 | 0.4547 | 0.2593 | 2024-09-30 | deprecated |
| 5 | mdl_20240930_lgbm_h20_1f061265_v0_1 | 2024-09-30 | 143 | 0.3514 | 0.2513 | 2024-10-21 | deprecated |
| 6 | mdl_20241030_lgbm_h20_b527a075_v0_1 | 2024-10-30 | 143 | 0.3405 | 0.2648 | 2024-11-19 | deprecated |
| 7 | mdl_20241129_lgbm_h20_313e114b_v0_1 | 2024-11-29 | 143 | 0.4053 | 0.2671 | 2024-12-19 | deprecated |
| 8 | mdl_20241231_lgbm_h20_acd998d4_v0_1 | 2024-12-31 | 143 | 0.4264 | 0.2580 | 2025-01-20 | deprecated |
| 9 | mdl_20250122_lgbm_h20_2221380e_v0_1 | 2025-01-22 | 144 | 0.3815 | 0.2138 | 2025-02-11 | deprecated |
| 10 | mdl_20250227_lgbm_h20_261baa1a_v0_1 | 2025-02-27 | 144 | 0.5184 | 0.2720 | 2025-03-19 | deprecated |
| 11 | mdl_20250331_lgbm_h20_8cb8e806_v0_1 | 2025-03-31 | 144 | 0.4085 | 0.3475 | 2025-04-21 | deprecated |
| 12 | mdl_20250430_lgbm_h20_d2f1c784_v0_1 | 2025-04-30 | 144 | 0.3900 | 0.2737 | 2025-05-20 | deprecated |
| 13 | mdl_20250530_lgbm_h20_7b9f0d39_v0_1 | 2025-05-30 | 145 | 0.2603 | 0.2830 | 2025-06-19 | deprecated |
| 14 | mdl_20250630_lgbm_h20_f36689d7_v0_1 | 2025-06-30 | 146 | 0.1820 | 0.2861 | 2025-07-21 | deprecated |
| 15 | mdl_20250731_lgbm_h20_90a4b395_v0_1 | 2025-07-31 | 146 | 0.3255 | 0.2458 | 2025-08-20 | deprecated |
| 16 | mdl_20250829_lgbm_h20_ada73e82_v0_1 | 2025-08-29 | 146 | 0.4269 | 0.2362 | 2025-09-18 | deprecated |
| 17 | mdl_20250930_lgbm_h20_a4a1b802_v0_1 | 2025-09-30 | 146 | 0.3281 | 0.2596 | 2025-10-20 | deprecated |
| 18 | mdl_20251031_lgbm_h20_fec82889_v0_1 | 2025-10-31 | 146 | 0.3737 | 0.2581 | 2025-11-20 | deprecated |
| 19 | mdl_20251128_lgbm_h20_a1c774ee_v0_1 | 2025-11-28 | 146 | 0.2468 | 0.2267 | 2025-12-18 | deprecated |
| 20 | mdl_20251231_lgbm_h20_a612e6fe_v0_1 | 2025-12-31 | 146 | 0.3932 | 0.2961 | 2026-01-20 | deprecated |
| 21 | mdl_20260130_lgbm_h20_cad31cae_v0_1 | 2026-01-30 | 147 | 0.2760 | 0.2365 | 2026-02-23 | deprecated |
| 22 | mdl_20260227_lgbm_h20_75c6761d_v0_1 | 2026-02-27 | 147 | 0.2405 | 0.2692 | 2026-03-19 | deprecated |
| 23 | mdl_20260331_lgbm_h20_25fc2461_v0_1 | 2026-03-31 | 147 | 0.4658 | 0.4166 | 2026-04-20 | deprecated |
| 24 | mdl_20260425_lgbm_h20_d969ffb1_v0_1 | 2026-04-25 | 147 | 0.3716 | 0.2796 | 2026-05-15 | committed |

## H30 Panel

| # | model_id | as_of | rows | IC | RMSE | label_date | prediction |
|---|---|---|---:|---:|---:|---|---|
| 1 | mdl_20240430_lgbm_h30_c8bac28b_v0_1 | 2024-04-30 | 142 | 0.2949 | 0.2290 | 2024-05-30 | deprecated |
| 2 | mdl_20240531_lgbm_h30_3caab1a8_v0_1 | 2024-05-31 | 142 | 0.3334 | 0.2413 | 2024-07-01 | deprecated |
| 3 | mdl_20240628_lgbm_h30_47e5b50e_v0_1 | 2024-06-28 | 143 | 0.3689 | 0.2915 | 2024-07-29 | deprecated |
| 4 | mdl_20240731_lgbm_h30_24343387_v0_1 | 2024-07-31 | 142 | 0.3578 | 0.2397 | 2024-08-30 | deprecated |
| 5 | mdl_20240830_lgbm_h30_f4cce711_v0_1 | 2024-08-30 | 142 | 0.4278 | 0.2163 | 2024-09-30 | deprecated |
| 6 | mdl_20240930_lgbm_h30_718e3a69_v0_1 | 2024-09-30 | 143 | 0.2966 | 0.2762 | 2024-10-30 | deprecated |
| 7 | mdl_20241030_lgbm_h30_d338aecb_v0_1 | 2024-10-30 | 143 | 0.3364 | 0.2739 | 2024-11-29 | deprecated |
| 8 | mdl_20241129_lgbm_h30_3dd9d4b1_v0_1 | 2024-11-29 | 143 | 0.2804 | 0.2866 | 2024-12-30 | deprecated |
| 9 | mdl_20241231_lgbm_h30_416ac1ca_v0_1 | 2024-12-31 | 143 | 0.5534 | 0.3173 | 2025-02-03 | deprecated |
| 10 | mdl_20250122_lgbm_h30_81dc6cf2_v0_1 | 2025-01-22 | 144 | 0.4782 | 0.3000 | 2025-02-21 | deprecated |
| 11 | mdl_20250227_lgbm_h30_1279061a_v0_1 | 2025-02-27 | 144 | 0.5889 | 0.3047 | 2025-03-31 | deprecated |
| 12 | mdl_20250331_lgbm_h30_eb7e7d2c_v0_1 | 2025-03-31 | 144 | 0.2824 | 0.2283 | 2025-04-30 | deprecated |
| 13 | mdl_20250430_lgbm_h30_4c18a33d_v0_1 | 2025-04-30 | 144 | 0.3731 | 0.2081 | 2025-06-02 | deprecated |
| 14 | mdl_20250530_lgbm_h30_ceba8c11_v0_1 | 2025-05-30 | 145 | 0.3218 | 0.2696 | 2025-06-30 | deprecated |
| 15 | mdl_20250630_lgbm_h30_429f5d89_v0_1 | 2025-06-30 | 146 | 0.2779 | 0.2909 | 2025-07-30 | deprecated |
| 16 | mdl_20250731_lgbm_h30_06b1d9d7_v0_1 | 2025-07-31 | 146 | 0.2545 | 0.2229 | 2025-09-01 | deprecated |
| 17 | mdl_20250829_lgbm_h30_05c07bc1_v0_1 | 2025-08-29 | 146 | 0.3468 | 0.2373 | 2025-09-30 | deprecated |
| 18 | mdl_20250930_lgbm_h30_880c02ca_v0_1 | 2025-09-30 | 146 | 0.2992 | 0.2461 | 2025-10-30 | deprecated |
| 19 | mdl_20251031_lgbm_h30_ee782506_v0_1 | 2025-10-31 | 146 | 0.2843 | 0.2453 | 2025-12-01 | deprecated |
| 20 | mdl_20251128_lgbm_h30_f5825d31_v0_1 | 2025-11-28 | 146 | 0.1978 | 0.2673 | 2025-12-29 | deprecated |
| 21 | mdl_20251231_lgbm_h30_1ede4261_v0_1 | 2025-12-31 | 146 | 0.4043 | 0.3028 | 2026-01-30 | deprecated |
| 22 | mdl_20260130_lgbm_h30_acaead63_v0_1 | 2026-01-30 | 147 | 0.3113 | 0.2330 | 2026-03-02 | deprecated |
| 23 | mdl_20260227_lgbm_h30_9f9ddfbd_v0_1 | 2026-02-27 | 147 | 0.2786 | 0.2190 | 2026-03-30 | deprecated |
| 24 | mdl_20260331_lgbm_h30_e5d97ec0_v0_1 | 2026-03-31 | 147 | 0.4090 | 0.4467 | 2026-04-30 | deprecated |

## Compliance

| Audit | Result |
|---|---|
| `audit_leakage.py v0.2` | 18/0/0 PERFECT; historical=48, production_current=0 |
| `audit_downstream_readiness.py v0.2` | 29/1/0 READY_FOR_DRAFT_EVIDENCE; committed_model_count=48 |
| Single WARN | production-current label window: `max_price_date=2026-05-15`, `required_label_date=2026-06-04` |

## Decision

The expanded 24-point h20/h30 panels confirm that positive IC stability is not limited to the most recent year. Both horizons remain fully historical evidence only; the production-current h20 gate remains blocked until `2026-06-04` or later label data is available.
