# Core Universe v0.2 Execution Log

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` §6.3 / §6.7
Purpose: 執行核心股確立與歷史資料就位驗收。

## 1. Execution Scope

本次執行範圍：

- `core_universe_builder.py v0.2`
- as_of_date: `2026-05-14`
- policy_version: `core_universe_policy_v0.2`
- formula: `CoreScore = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP`

本次未抓取新交易資料；2026-05-17 為週日，僅使用 DB 既有資料執行核心股治理與覆蓋驗收。

## 2. CoreScore v0.2 Execution

Commands:

```bash
python scripts/core/core_universe_builder.py --dry-run --as-of-date 2026-05-14
python scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-14
```

Result:

- snapshot_id: `core_universe_20260514_core_universe_policy_v0_2`
- status: committed
- total_candidates: 2799
- core_universe: 120
- convex_universe: 30
- research_universe: 2271
- quarantine_universe: 378
- written_rows: 5601
- preflight PASS/WARN/FAIL: 7/0/0
- v0.2 contract PASS/WARN/FAIL: 16/4/0
- verdict: WARNING

WARNING explanation:

- Warnings are from full-market coverage and `TaiwanStockInfo` as-of fallback:
  - market-wide price zero-coverage candidates: 2652
  - market-wide revenue zero-coverage candidates: 2651
  - market-wide financial zero-coverage candidates: 2651
  - TaiwanStockInfo as-of candidates 65 < 150, using latest registry fallback
- Core+convex 150 scope coverage is materially in place and suitable for current governance scoring.

## 3. Historical Data Coverage

DB coverage as of `2026-05-14`:

- `TaiwanStockPriceAdj`: 148 stocks / 69,880 rows
- `TaiwanStockMonthRevenue`: 148 stocks / 3,369 rows
- `TaiwanStockFinancialStatements`: 148 stocks / 17,450 rows
- `TaiwanStockInstitutionalInvestorsBuySell`: 148 stocks / 326,045 rows

Builder core-sync coverage summary:

- price coverage 252d: pass 146/150, avg coverage 0.954868
- revenue coverage 24m: pass 140/150, avg coverage 0.935833
- financial coverage 8q: pass 147/150, avg coverage 0.970833

## 4. Core Universe Audit

Command:

```bash
python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14
```

Result:

- report: `reports/core_universe_audit_20260517_0910.md`
- snapshot: `core_universe_20260514_core_universe_policy_v0_2`
- PASS/WARN/FAIL: 36/0/0
- verdict: PERFECT

## 5. Current Tier Counts

Current committed tier counts:

- convex_universe: 30
- core_universe: 120
- quarantine_universe: 378
- research_universe: 2271

Core+convex total: 150.

## 6. Top 10 CoreScore Names

Top 10 by `core_score` among core+convex:

| Rank | Stock | Name | Tier | CoreScore |
|---:|---|---|---|---:|
| 1 | 2408 | 南亞科 | convex_universe | 94.56 |
| 2 | 8299 | 群聯 | convex_universe | 90.63 |
| 3 | 6223 | 旺矽 | convex_universe | 90.21 |
| 4 | 3260 | 威剛 | convex_universe | 89.57 |
| 5 | 2330 | 台積電 | convex_universe | 89.56 |
| 6 | 2303 | 聯電 | convex_universe | 88.94 |
| 7 | 5274 | 信驊 | convex_universe | 88.81 |
| 8 | 3105 | 穩懋 | convex_universe | 88.80 |
| 9 | 4991 | 環宇-KY | convex_universe | 88.59 |
| 10 | 4749 | 新應材 | convex_universe | 87.65 |

## 7. Decision

Core Universe v0.2 is executed and committed.

Conclusion:

- 核心股已確立：120 core + 30 convex。
- 歷史資料已就位於 core+convex governance scope。
- `core_universe_builder.py v0.2` 已把核心股從 metadata bootstrap 名單推進為六層 CoreScore 量化治理名單。
- 本步屬 universe governance selection，不是交易預測或投資建議。

Remaining boundary:

- Full-market coverage remains incomplete; warnings are retained and transparent.
- Downstream Feature Store / Model / Prediction remains governed by §8 and has its own readiness gates.

