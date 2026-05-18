# Model Quality Research: Sector-Neutral and Ablation Tests (2026-05-18)

- generated_at: 2026-05-18 Asia/Taipei
- scope: existing 48 committed historical models
- horizons: h20 and h30
- production impact: none
- verdict: CURRENT_SIGNAL_STACK_VALID; SECTOR_NEUTRAL_NOT_ADOPTED

## Purpose

This report tests two follow-up questions from `feature_signal_diagnostics_20260518.md`:

1. Would sector-neutral ranking reduce semiconductor concentration without hurting IC?
2. Which feature groups are truly contributing when removed from the trained scoring stack?

No new Feature Store, model, or prediction artifacts were created.

## Method

For each of the 48 existing models:

- Load model artifact weights and preprocessing bounds.
- Reload the model feature set from `feature_values`.
- Rebuild the trainer-style cross-sectional rank-score transform.
- Reconstruct forward-return labels using the same trainer rule.
- Recompute full-model IC as the baseline.

Sector-neutral test:

- Compute the model score normally.
- Rank-normalize predictions separately within `semiconductor` and `non_semiconductor`.
- Combine the sector-local rank scores and recompute all-universe rank IC.

Ablation test:

- Drop one feature group at a time by setting that group's weights to zero.
- Recompute prediction score and all-universe rank IC.
- Report `drop_minus_full`; negative values mean the removed group was helpful.

## Sector-Neutral Ranking Result

| Horizon | Models | Full mean IC | Sector-neutral mean IC | Mean delta | Median delta | Improved models |
|---|---:|---:|---:|---:|---:|---:|
| h20 | 24 | 0.3530 | 0.3082 | -0.0448 | -0.0508 | 4/24 |
| h30 | 24 | 0.3482 | 0.3008 | -0.0474 | -0.0320 | 2/24 |

Decision: do not adopt sector-neutral ranking as the default model scoring rule. It reduces the semiconductor concentration risk, but it also removes a meaningful part of current alpha. Sector-neutral ranking may still be useful as a portfolio-level risk overlay, not as a replacement for model scoring.

## Feature Group Ablation

Values below are mean `drop_minus_full` IC deltas. Negative means dropping the group made performance worse.

| Horizon | Dropped group | Mean delta | Median delta | Min delta | Max delta | Harmful drop count |
|---|---|---:|---:|---:|---:|---:|
| h20 | price | -0.0682 | -0.0457 | -0.2012 | 0.0199 | 19/24 |
| h20 | fundamental | -0.0226 | -0.0136 | -0.0699 | 0.0038 | 23/24 |
| h20 | institutional | -0.0210 | -0.0186 | -0.0499 | -0.0018 | 24/24 |
| h20 | liquidity | -0.0124 | 0.0011 | -0.1356 | 0.0383 | 11/24 |
| h20 | macro | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/24 |
| h20 | theme | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/24 |
| h30 | price | -0.0563 | -0.0452 | -0.1894 | 0.0253 | 21/24 |
| h30 | fundamental | -0.0293 | -0.0211 | -0.1169 | 0.0082 | 22/24 |
| h30 | liquidity | -0.0188 | -0.0020 | -0.1216 | 0.0245 | 13/24 |
| h30 | institutional | -0.0162 | -0.0132 | -0.0554 | 0.0106 | 20/24 |
| h30 | macro | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/24 |
| h30 | theme | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/24 |

## Interpretation

1. Price is the strongest group. Dropping price causes the largest IC loss in both horizons.
2. Fundamental is consistently useful and becomes slightly more important for h30.
3. Institutional is smaller but unusually stable: h20 institutional drop hurts 24/24 models.
4. Liquidity is important but more regime-dependent; it has strong negative deltas in some periods but neutral or positive deltas in others.
5. Macro and theme currently have zero ablation impact. This confirms the previous diagnostic: these features do not contribute to the current cross-sectional rank model in their present form.
6. Sector-neutral ranking should not be used as the model's default scoring transform. It lowers mean IC by about 0.045 for both h20 and h30.

## Recommendation

Keep the current model scoring rule for historical evidence and future production-current h20. Handle semiconductor concentration later at the portfolio/risk layer, or test a research-only sector-neutral overlay after production-current v6.1.0 is complete.

Future research candidates:

1. Sector exposure cap or portfolio-level diversification overlay.
2. Sector interaction features rather than hard sector-neutral ranking.
3. Macro regime interaction features that create cross-sectional dispersion.
4. Automated ablation audit script for repeated model releases.
