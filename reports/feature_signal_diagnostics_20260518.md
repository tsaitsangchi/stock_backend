# Feature Signal Diagnostics Report (2026-05-18)

- generated_at: 2026-05-18 Asia/Taipei
- scope: existing 48 committed historical models only
- horizons: h20 and h30
- model source: `model_registry.status='committed'`
- prediction source: all retained `prediction_values`, including deprecated evidence-only runs
- delivery invariant: only `pred_20260425_mdl_20260425_lgbm_h20_d969ffb1_v0_1` remains committed
- verdict: SIGNAL_BROADLY_VALID_WITH_SECTOR_CONCENTRATION_RISK

## Method

This report does not create new features, models, or predictions.

Feature group contribution uses model artifact `weights` joined to `feature_definition.feature_group`, plus `model_registry.metrics.top_rank_ic_features`.

Tier and sector diagnostics reconstruct labels with the same trainer rule:

- base price: latest `TaiwanStockPriceAdj.close` where `date <= as_of_date`
- label price: first `TaiwanStockPriceAdj.close` where `date >= as_of_date + label_horizon calendar days`
- label: `(future_close / base_close) - 1`
- score: Spearman rank-IC between `prediction_value` and reconstructed label within each subgroup

Sector split:

- `semiconductor`: `industry_category` contains `半導體`
- `non_semiconductor`: all other core+convex members

## Feature Group Contribution

| Horizon | Group | Mean abs weight | Mean signed weight | Top-feature hits | Mean abs top rank-IC |
|---|---|---:|---:|---:|---:|
| h20 | price | 0.0547 | 0.0082 | 108 | 0.2029 |
| h20 | liquidity | 0.0535 | 0.0178 | 55 | 0.1998 |
| h20 | fundamental | 0.0453 | 0.0158 | 42 | 0.1687 |
| h20 | institutional | 0.0334 | 0.0043 | 35 | 0.1328 |
| h20 | macro | 0.0000 | 0.0000 | 0 | N/A |
| h20 | theme | 0.0000 | 0.0000 | 0 | N/A |
| h30 | price | 0.0528 | -0.0029 | 99 | 0.1916 |
| h30 | liquidity | 0.0521 | -0.0026 | 55 | 0.1893 |
| h30 | fundamental | 0.0508 | 0.0199 | 52 | 0.1680 |
| h30 | institutional | 0.0332 | 0.0050 | 34 | 0.1342 |
| h30 | macro | 0.0000 | 0.0000 | 0 | N/A |
| h30 | theme | 0.0000 | 0.0000 | 0 | N/A |

## H20 vs H30 Consistency

| Group | h20 rank | h30 rank | Consistency |
|---|---:|---:|---|
| price | 1 | 1 | stable top contributor |
| liquidity | 2 | 2 | stable top contributor |
| fundamental | 3 | 3 | stable secondary contributor |
| institutional | 4 | 4 | stable but weaker |
| macro | 5 | 5 | no cross-sectional signal in current implementation |
| theme | 6 | 6 | no active model contribution in current implementation |

Decision: h20 and h30 use a consistent signal stack. The models are not being driven by an arbitrary single feature group; however, the dominant signal family is clearly price/liquidity.

## Tier Diagnostics

| Horizon | Segment | Models | Mean IC | Median IC | Min IC | Max IC | IC >= 0 | Avg N |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h20 | all | 24 | 0.3532 | 0.3737 | 0.1832 | 0.5177 | 24/24 | 144.7 |
| h20 | core_universe | 24 | 0.3231 | 0.3119 | 0.0537 | 0.4715 | 24/24 | 114.7 |
| h20 | convex_universe | 24 | 0.3018 | 0.3059 | -0.0438 | 0.6561 | 23/24 | 30.0 |
| h30 | all | 24 | 0.3479 | 0.3269 | 0.1979 | 0.5897 | 24/24 | 144.5 |
| h30 | core_universe | 24 | 0.3201 | 0.3149 | 0.0292 | 0.6255 | 24/24 | 114.5 |
| h30 | convex_universe | 24 | 0.2982 | 0.3128 | -0.0314 | 0.6828 | 22/24 | 30.0 |

Decision: core and convex tiers are both effective. Convex is noisier due to smaller sample size (`N=30`) and has a small number of negative subgroup IC observations, but its mean and median remain positive for both horizons.

## Sector Diagnostics

| Horizon | Segment | Models | Mean IC | Median IC | Min IC | Max IC | IC >= 0 | Avg N |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h20 | semiconductor | 24 | 0.3441 | 0.3577 | 0.1855 | 0.4894 | 24/24 | 116.6 |
| h20 | non_semiconductor | 24 | 0.1674 | 0.2503 | -0.4198 | 0.7504 | 15/24 | 28.1 |
| h30 | semiconductor | 24 | 0.3293 | 0.3011 | 0.1432 | 0.4979 | 24/24 | 116.3 |
| h30 | non_semiconductor | 24 | 0.1781 | 0.1824 | -0.4176 | 0.7192 | 20/24 | 28.1 |

Decision: the signal is materially stronger and more stable inside semiconductor names. Non-semiconductor names still have positive mean IC, but the subgroup is small and volatile, with several negative IC periods. This is a concentration risk, not a leakage failure.

## Findings

1. Feature group contribution is stable across h20 and h30: price and liquidity dominate, fundamental contributes meaningfully, institutional contributes weakly but consistently.
2. Macro and theme currently contribute zero model weight. For macro this is expected because current macro features are mostly cross-section constants at a given as-of date; rank-based cross-sectional trainer cannot extract stock-level dispersion from constants.
3. Core and convex tiers both show positive signal. Convex has higher volatility, consistent with smaller N and more optionality/edge-case names.
4. Semiconductor concentration is real. The model works best where the core universe is densest and most liquid; non-semiconductor evidence is positive on average but not consistently positive across all periods.
5. No new production-current claim is created by this report. It only diagnoses existing historical evidence.

## Recommendations

1. Do not add more historical panel points before production-current gate unless a specific weakness is being tested.
2. Before v6.1.0 production-current promotion, cite this report as evidence that h20/h30 signals are broad enough across tiers but sector-concentrated.
3. Future research branch: add sector-neutral diagnostics or sector-neutral ranking to test whether non-semiconductor stability improves without weakening all-universe IC.
4. Future feature work: macro features should be transformed into stock-sensitive exposures or regime interaction terms if macro is expected to contribute to cross-sectional predictions.
5. Future audit tooling: add an automated `audit_feature_signal_diagnostics.py` to regenerate this report from `model_registry`, `prediction_values`, and `feature_definition`.
