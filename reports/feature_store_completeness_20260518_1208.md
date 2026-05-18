# Core 150 Feature Store Completeness Report

- **time**: 2026-05-18 12:08 Asia/Taipei
- **constitution**: 系統架構大憲章_v6.0.0.md §8.2 / §14.7-L
- **feature_set_id**: `fs_20260515_feature_set_v0_1_h20_core150_strict_source_20260518`
- **feature_set_version**: `feature_set_v0.1_h20_core150_strict_source_20260518`
- **as_of_date**: `2026-05-15`
- **source_data_cutoff**: `2026-05-18`
- **universe_snapshot_id**: `core_universe_20260515_core_universe_policy_v0_2`
- **label_horizon**: `20`
- **mode**: committed
- **verdict**: **PERFECT_WITH_EXPLAINED_GAPS**

## Source Prerequisite

Live source availability audit already passed:

```text
FinMind checked=1350, source_empty_ok=13, mismatch=0, api_errors=0
FRED checked=4, mismatch=0, api_errors=0
```

Therefore missing feature slots below are not caused by DB/API ingestion incompleteness.

## Build Result

```text
preflight PASS/WARN/FAIL = 14/0/0
stocks = 150
features = 27
theoretical slots = 4050
feature_value rows = 3980
missing dropped slots = 70
null imputed rows = 47
warnings = 0
failed = 0
```

## DB Verification

| Metric | Value |
|---|---:|
| committed snapshots | 1 |
| feature rows | 3980 |
| imputed rows | 47 |
| distinct stocks | 150 |
| distinct features | 27 |

## Coverage By Group

| Feature group | Rows | Imputed | Stocks covered | Features |
|---|---:|---:|---:|---:|
| fundamental | 581 | 4 | 150 | 4 |
| institutional | 750 | 43 | 150 | 5 |
| liquidity | 582 | 0 | 147 | 4 |
| macro | 600 | 0 | 150 | 4 |
| price | 1167 | 0 | 147 | 8 |
| theme | 300 | 0 | 150 | 2 |

## Dropped Slot Diagnosis

The 70 missing slots are all `null_strategy='drop'` features where the stock does not have enough valid as-of history for the feature window. They are concentrated in 10 stocks:

| stock_id | Missing slots | Diagnosis |
|---|---:|---|
| 1701 | 13 | Source lifecycle: API/DB price and revenue both end before `as_of_date`; not an ingestion miss |
| 1729 | 14 | Legacy inactive source lifecycle; no FinMind FinancialStatements source rows |
| 3559 | 14 | Legacy inactive source lifecycle; no FinMind FinancialStatements source rows |
| 6907 | 7 | New listing / short history; 252d price and 12m/3m revenue windows not fully available |
| 7772 | 7 | New listing / short history; 252d price and revenue windows not fully available |
| 7828 | 7 | New listing / short history; 252d price and revenue windows not fully available |
| 7751 | 2 | Revenue window not long enough for YoY features |
| 7770 | 2 | Revenue window not long enough for YoY features |
| 7810 | 2 | Revenue window not long enough for YoY features |
| 8102 | 2 | Revenue window not long enough for YoY features |

Missing by group:

| Group | Missing slots |
|---|---:|
| price | 33 |
| liquidity | 18 |
| fundamental | 19 |

Missing by feature:

| Feature | Missing slots |
|---|---:|
| revenue_yoy_3m | 10 |
| revenue_yoy_12m | 9 |
| log_return_252d | 6 |
| volatility_252d | 6 |
| max_drawdown_252d | 6 |
| avg_daily_value_log_252d | 6 |
| zero_volume_ratio_252d | 6 |
| log_return_20d | 3 |
| log_return_60d | 3 |
| volatility_60d | 3 |
| ma_ratio_20 | 3 |
| ma_ratio_60 | 3 |
| avg_daily_value_log_60d | 3 |
| turnover_mean_60d | 3 |

## Imputation Diagnosis

The 47 imputed rows are all features with explicit `zero_fill` strategy:

| Feature | Imputed rows | Diagnosis |
|---|---:|---|
| margin_ratio_60d | 31 | Legitimate zero-fill for no valid short-sale denominator or source-empty margin/short cases |
| foreign_net_20d | 3 | Legitimate zero-fill for no recent institutional flow rows |
| foreign_net_60d | 3 | Legitimate zero-fill for no recent institutional flow rows |
| trust_net_20d | 3 | Legitimate zero-fill for no recent investment-trust flow rows |
| trust_net_60d | 3 | Legitimate zero-fill for no recent investment-trust flow rows |
| eps_sum_4q | 2 | Source-empty FinancialStatements for `1729` and `3559` |
| net_income_positive_ratio_8q | 2 | Source-empty FinancialStatements for `1729` and `3559` |

## Model Gate Decision

Do **not** start production-current h20 training yet.

Reason: `TaiwanStockPriceAdj.MAX(date)=2026-05-15`, while a production h20 label for `as_of_date=2026-05-15` requires future price data after the 20-trading-day horizon. The Feature Store is ready and committed, but production model training should wait until the h20 label window is available.

Allowed next actions:

1. Run historical or walk-forward training for validation.
2. Wait for production-current label availability, then run `model_trainer.py` against this feature set.
3. Improve downstream filtering so inactive/short-history stocks are excluded or flagged before model training if the training policy requires full 27/27 feature coverage.
