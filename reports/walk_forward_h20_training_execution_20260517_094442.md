# Walk-Forward H20 Training Execution Report

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` §8 draft
Scope: 依指定 as-of dates 建立多組 Feature Store，全部使用 strict `h20` forward-return label 訓練。

## 1. Intent

User request:

- 使用 `2024-09-30`, `2024-10-31`, `2024-11-30`, `2024-12-31`, `2025-01-31`, `2025-02-28`, `2025-03-31`, `2025-04-25` 多個 as-of dates。
- 先確認並補齊資料。
- 每個 as-of date 建立 feature set。
- 全部使用 `label_horizon=20` 訓練。

Governance interpretation:

- Feature Store 使用最新 committed v0.2 core+convex universe:
  `core_universe_20260514_core_universe_policy_v0_2`
- Feature derivation keeps `WHERE date <= as_of_date`.
- Trainer enforces `label_date >= as_of_date + label_horizon`.
- No production-current `2026-05-14 h20` acceptance is implied by this historical walk-forward run.

## 2. Data Refill

Command:

```bash
python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 1100 --no-resume
```

Result:

- success items: `1321`
- warning items: `33`
- failed items: `0`
- skipped items: `0`
- total written rows: `1,017,333`
- elapsed: `663.48s`
- sovereign verdict: `WARNING`

Notes:

- The run completed without failed sync items.
- Warning items were mostly source/API zero-row cases for specific stock/table combinations.
- A pandas date parsing warning was observed in `sovereign_sync_engine.py`; it did not stop execution.

## 3. Coverage After Refill

Core+convex universe size: `150`

| as_of_date | price stocks / rows | revenue stocks / rows | financial stocks / rows | institutional stocks / rows | h20 labels |
|---|---:|---:|---:|---:|---:|
| 2024-09-30 | 144 / 47,231 | 134 / 2,123 | 145 / 12,641 | 134 / 215,470 | 143 |
| 2024-10-31 | 144 / 49,948 | 136 / 2,258 | 145 / 12,641 | 134 / 227,800 | 143 |
| 2024-11-30 | 144 / 52,951 | 138 / 2,395 | 145 / 12,641 | 134 / 241,275 | 143 |
| 2024-12-31 | 144 / 56,097 | 139 / 2,533 | 148 / 14,864 | 138 / 255,730 | 143 |
| 2025-01-31 | 145 / 58,256 | 140 / 2,672 | 148 / 14,864 | 139 / 265,740 | 144 |
| 2025-02-28 | 145 / 60,992 | 141 / 2,812 | 148 / 14,864 | 140 / 278,495 | 144 |
| 2025-03-31 | 145 / 64,016 | 141 / 2,952 | 148 / 17,030 | 141 / 292,640 | 144 |
| 2025-04-25 | 145 / 66,464 | 141 / 3,092 | 148 / 17,030 | 141 / 304,030 | 144 |

Price window coverage expanded to `2023-05-15` for normal long-history stocks.

Remaining price lifecycle/source gaps:

- `1729 必翔`: no `TaiwanStockPriceAdj` rows in DB.
- `3559 全智科`: no `TaiwanStockPriceAdj` rows in DB.
- `1701 中化`: price data ends at `2024-08-20`.
- `7810 捷創科技`: first price date `2025-01-03`.
- `7828 創新服務`: first price date `2025-05-08`.
- `7772 耀穎`: first price date `2025-06-26`.
- `6907 雅特力-KY`: first price date `2026-01-29`.

Decision:

- Data is sufficient to run historical `h20` walk-forward training.
- Missing stock counts are explained by source/lifecycle gaps, not by a failed refill.
- Rows trained are expected to be `143` or `144`, not the full `150`.

## 4. Feature Store Commits

| as_of_date | feature_set_id | rows | imputed | verdict |
|---|---|---:|---:|---|
| 2024-09-30 | `fs_20240930_feature_set_v0_1_h20_walk_forward_20240930` | 3,890 | 109 | PERFECT |
| 2024-10-31 | `fs_20241031_feature_set_v0_1_h20_walk_forward_20241031` | 3,892 | 109 | PERFECT |
| 2024-11-30 | `fs_20241130_feature_set_v0_1_h20_walk_forward_20241130` | 3,897 | 119 | PERFECT |
| 2024-12-31 | `fs_20241231_feature_set_v0_1_h20_walk_forward_20241231` | 3,907 | 98 | PERFECT |
| 2025-01-31 | `fs_20250131_feature_set_v0_1_h20_walk_forward_20250131` | 3,920 | 92 | PERFECT |
| 2025-02-28 | `fs_20250228_feature_set_v0_1_h20_walk_forward_20250228` | 3,922 | 87 | PERFECT |
| 2025-03-31 | `fs_20250331_feature_set_v0_1_h20_walk_forward_20250331` | 3,929 | 84 | PERFECT |
| 2025-04-25 | `fs_20250425_feature_set_v0_1_h20_walk_forward_20250425` | 3,938 | 83 | PERFECT |

All feature sets:

- status: `committed`
- total_stocks: `150`
- feature_count: `27`
- label_horizon: `20`
- universe snapshot: `core_universe_20260514_core_universe_policy_v0_2`

## 5. H20 Model Commits

| train_end | model_id | rows | ic_mean | rmse | label_date |
|---|---|---:|---:|---:|---|
| 2024-09-30 | `mdl_20240930_lgbm_h20_ac2a2c6e_v0_1` | 143 | 0.342902 | 0.249388 | 2024-10-21 |
| 2024-10-31 | `mdl_20241031_lgbm_h20_48cbcf59_v0_1` | 143 | 0.317958 | 0.276625 | 2024-11-20 |
| 2024-11-30 | `mdl_20241130_lgbm_h20_1a50093b_v0_1` | 143 | 0.358121 | 0.258425 | 2024-12-20 |
| 2024-12-31 | `mdl_20241231_lgbm_h20_6030ba03_v0_1` | 143 | 0.422609 | 0.262946 | 2025-01-20 |
| 2025-01-31 | `mdl_20250131_lgbm_h20_9d6ee9ac_v0_1` | 144 | 0.493401 | 0.301985 | 2025-02-20 |
| 2025-02-28 | `mdl_20250228_lgbm_h20_aeb5bb87_v0_1` | 144 | 0.482935 | 0.257108 | 2025-03-20 |
| 2025-03-31 | `mdl_20250331_lgbm_h20_de5ab902_v0_1` | 144 | 0.425975 | 0.345794 | 2025-04-21 |
| 2025-04-25 | `mdl_20250425_lgbm_h20_9d93a46e_v0_1` | 144 | 0.490061 | 0.299978 | 2025-05-15 |

All models:

- status: `committed`
- trainer: `robust_rank_ic_baseline_v0.1`
- label source: `TaiwanStockPriceAdj.forward_return_label_v0.1`
- label_horizon: `20`
- model_id includes `feature_set_version` hash.

## 6. Leakage Audit

Command:

```bash
python scripts/maintenance/audit_leakage.py
```

Result:

- PASS/WARN/FAIL: `18/0/0`
- sovereign verdict: `PERFECT`

Confirmed:

- All feature definitions are `as_of_strict`.
- Committed models satisfy `label_date_min >= as_of_date + label_horizon`.
- Committed models use h20 labels through `<= 2025-05-15`.
- Model ids include feature-set-version hash governance.

## 7. Readiness Audit Regression And Fix

After committing the 8 walk-forward models, `audit_downstream_readiness.py --no-report` initially failed because the previous rule required exactly one committed model.

Fix applied:

- Current delivery evidence is now selected as the unique prediction-backed committed model.
- Additional committed historical walk-forward models are allowed as validation evidence.
- All committed models are still checked for:
  - model_id hash governance
  - `label_horizon=20`
  - `label_date_max <= 2025-05-15`
  - positive `ic_mean`
  - existing model artifact

Post-fix result:

- `python scripts/maintenance/audit_downstream_readiness.py --no-report`
- PASS/WARN/FAIL: `29/1/0`
- verdict: `READY_FOR_DRAFT_EVIDENCE`
- only WARN: production-current label window remains blocked until `2026-06-03`.

## 8. Issues Recorded

1. Data refill verdict is `WARNING`, not `PERFECT`, because some source/API requests returned zero rows.
2. `1729` and `3559` have no adjusted price data in DB; they cannot participate in historical feature/label rows until a valid source is available.
3. `6907`, `7772`, `7828`, and `7810` have lifecycle-limited price histories and cannot appear in early historical as-of snapshots.
4. `1701` has adjusted price data ending `2024-08-20`; it is excluded from later h20 labels.
5. `2024-11-30` is a Saturday. The current builder uses strict `date <= as_of_date`, so actual price cutoff is `2024-11-29`; this is governance-compatible but should be noted in downstream reporting.

## 9. Final Decision

Historical walk-forward `h20` evidence is now complete for the requested dates.

This does not promote §8 to v5.4.23 by itself. It strengthens historical validation evidence, while production-current acceptance remains separately blocked until the `2026-05-14` production snapshot has its own `h20` label window available.
