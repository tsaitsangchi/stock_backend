# Core 150 API Source Start Dates Probe (2026-05-18)

## Scope

This report records the source-side availability probe for the current committed `core_universe ∪ convex_universe` universe defined by charter §6.7.

- Core universe size: 150
- FinMind scope: 150 stocks x 9 stock-level datasets
- FRED scope: 4 macro series
- Probe time: 2026-05-18 10:43:38
- API errors: 0
- Raw machine-readable result: `/tmp/api_start_dates_core150.json` during execution; this report is the tracked audit-trail summary.

## FRED Valid Numeric Coverage

| series_id | API raw rows | API valid rows | valid min date | valid max date |
|---|---:|---:|---|---|
| DFF | 26251 | 26251 | 1954-07-01 | 2026-05-14 |
| UNRATE | 940 | 939 | 1948-01-01 | 2026-04-01 |
| T10Y2Y | 13034 | 12486 | 1976-06-01 | 2026-05-15 |
| VIXCLS | 9488 | 9186 | 1990-01-02 | 2026-05-14 |

Verdict: FRED source availability is confirmed. Row-count validation must continue to use valid numeric observations, excluding `value='.'`, consistent with charter §6.8.8.

## FinMind Source Availability By Dataset

| Dataset | Stocks with source data | Source-empty stocks | Earliest source start | Latest source start |
|---|---:|---:|---|---|
| TaiwanStockPrice | 150 | 0 | 1992-01-04 | 2026-01-29 |
| TaiwanStockPriceAdj | 150 | 0 | 1992-01-06 | 2026-01-29 |
| TaiwanStockPER | 150 | 0 | 2005-09-02 | 2026-05-08 |
| TaiwanStockInstitutionalInvestorsBuySell | 150 | 0 | 2005-01-03 | 2026-05-08 |
| TaiwanStockMarginPurchaseShortSale | 142 | 8 | 2001-01-05 | 2026-01-14 |
| TaiwanStockShareholding | 150 | 0 | 2004-02-12 | 2026-05-07 |
| TaiwanStockFinancialStatements | 148 | 2 | 1990-03-31 | 2024-12-31 |
| TaiwanStockMonthRevenue | 150 | 0 | 2002-02-01 | 2026-03-01 |
| TaiwanStockDividend | 147 | 3 | 2005-06-10 | 2025-08-24 |

## Source-Empty Stock Lists

| Dataset | Source-empty stocks |
|---|---|
| TaiwanStockMarginPurchaseShortSale | 6708 天擎, 6907 雅特力-KY, 7751 竑騰, 7770 君曜, 7772 耀穎, 7810 捷創科技, 7828 創新服務, 8102 傑霖科技 |
| TaiwanStockFinancialStatements | 1729 必翔, 3559 全智科 |
| TaiwanStockDividend | 1729 必翔, 3559 全智科, 6907 雅特力-KY |

## Interpretation

The 2026-05-18 completeness probe found 38 FinMind DB/API row-count mismatches out of 1350 checks. The source-start-date probe shows these mismatches are primarily a start-date scope mismatch, not a latest-data ingestion failure:

- The DB full-history refill used `--days 11500`, which starts around 1994-11-22 for daily tables.
- FinMind API source data for legacy stocks can begin at 1992-01-04 for `TaiwanStockPrice`, 1992-01-06 for `TaiwanStockPriceAdj`, and 1990-03-31 for `TaiwanStockFinancialStatements`.
- Therefore row-count gaps such as `-820`, `-819`, and `-154` to `-266` are explained by early 1990s source rows existing before the DB refill cutoff.
- For production-current use, the mismatched legacy stocks mostly have DB max dates aligned with API max dates. The exception `1701` has API and DB both ending at 2024-08-20 for daily price tables, which is a source-side lifecycle condition rather than a new ingestion miss.

## Governance Verdict

1. The source side is confirmed queryable: FinMind and FRED returned no API errors for this probe.
2. Source-empty cases are legitimate source-side availability gaps and must not be treated as DB ingestion failures.
3. The existing `--days 11500` refill is sufficient for current production-current and §8 feature/model work, but it is not sufficient for strict all-source-history equality for legacy stocks whose API data begins before 1994-11-22.
4. If the governance objective becomes strict `API min(date) == DB min(date)` for all core 150 stock-level datasets, a targeted no-resume refill should be run for the affected legacy stock/table pairs with a wider historical window, for example `--days 14000`, instead of full-market resync.
5. Feature Store completeness checks must distinguish source-empty features, DB historical cutoff gaps, and builder-level missing values before classifying a missing feature as a bug.

## Strict All-Source-History Completion Addendum

After the governance objective was explicitly set to strict `API min(date) == DB min(date)` and exact row-count equality for every core 150 stock-level dataset, two targeted no-resume backfill rounds were executed.

### Round 1

- Log: `logs/rebuild_runs/rebuild_20260518_081311/p4_strict_fullhistory_backfill.log`
- Scope: 38 originally failing stock/dataset pairs from `p3_completeness_probe.log`
- Command form: `sovereign_sync_engine.py --id <stock_id> --dataset <dataset> --days 14000 --no-resume`
- Result: every pair returned `PERFECT`; no failed sync items, no skipped items, no 402-recovered events.

### Round 2

The first strict full-matrix DB comparison surfaced 12 additional early-history gaps that had previously passed the loose `<=5%` tolerance:

- `1717`: `TaiwanStockPrice`, `TaiwanStockPriceAdj`
- `1720`: `TaiwanStockFinancialStatements`
- `1721`: `TaiwanStockFinancialStatements`
- `1731`: `TaiwanStockFinancialStatements`
- `2329`: `TaiwanStockPrice`, `TaiwanStockPriceAdj`
- `2330`: `TaiwanStockPrice`, `TaiwanStockPriceAdj`, `TaiwanStockFinancialStatements`
- `5302`: `TaiwanStockFinancialStatements`
- `5351`: `TaiwanStockFinancialStatements`

Round 2 execution:

- Log: `logs/rebuild_runs/rebuild_20260518_081311/p5_strict_fullhistory_backfill_round2.log`
- Same command form: `--days 14000 --no-resume`
- Result: every pair returned `PERFECT`; no failed sync items, no skipped items, no 402-recovered events.

### Final Strict Verification

The final strict comparison used the source-side FinMind result captured in `/tmp/api_start_dates_core150.json` and compared every core stock/dataset pair against current DB `COUNT(*)`, `MIN(date)`, and `MAX(date)`.

Result:

```text
checked 1350
source_empty_ok 13
mismatch 0
```

Final verdict: under the strict definition, core 150 FinMind stock-level datasets are now aligned to source-side earliest available dates, with legitimate source-empty cases represented as zero DB rows. FRED valid numeric coverage was already aligned before the targeted backfill and remains governed by the valid-observation rule.

Note: a subsequent `audit_supply_chain.py --include-logs` run returned one log-window failure because the 24-hour `pipeline_execution_log` window contained `audit_downstream_readiness_v0.1=failed` from the known §8 ACTIVE DRAFT lazy-schema state, and the failed audit itself then logged a `post_schema_audit_v1.18=failed` row. This is a log-window cleanliness issue, not a DB/API data completeness failure.
