# 2026-05-17 DB 全表刪除後重建執行紀錄

**日期**: 2026-05-17  
**憲章基準**: `系統架構大憲章_v5.4.22.md`  
**任務**: 使用者已刪除 database all tables，依憲章從零重建至目前 accepted CoreScore v0.2 核心股 snapshot，並記錄執行狀況與問題。  
**重要裁決**: 本次 `core_universe_builder.py --commit --as-of-date 2026-05-14` 屬於 **DB full rebuild restore accepted snapshot**，不是日常核心股重選；因 `2026-05-14` 非年度最後交易日，必須以 `--special-rebalance-reason` 留痕，符合 §6 特別治理例外。

---

## 1. Bootstrap / Schema 重建

| Step | 指令 | 結果 | 摘要 |
|---|---|---|---|
| 1 | `python scripts/core/path_setup.py` | PERFECT | 25 維路徑對齊；DB schema pending 時 logging hook 安全降級為 `BOOTSTRAP-DEFERRED`。 |
| 2 | `python scripts/core/data_schema.py --init --force` | PERFECT ALIGNMENT | API contract PASS/WARN/FAIL = `11/0/0`；13 張 raw/log tables 建立完成。 |
| 2B | `python scripts/core/core_universe_schema.py --init` | PERFECT | 核心股 governance 7 tables 建立完成；preflight `9/0/0`。 |
| 2C | `python scripts/core/db_utils.py` | WARNING | DB 與 log tables 正常；§6.7 core universe rows = 0，屬合法 bootstrap warning。 |
| 3 | `python scripts/maintenance/audit_supply_chain.py --include-logs` | PERFECT | 報告 `compliance_audit_20260517_1327.md`；PASS/WARN/FAIL = `29/0/0`。 |

---

## 2. Seed Ingestion

| Step | 指令 | 結果 | 摘要 |
|---|---|---|---|
| 4 | `python scripts/ingestion/sovereign_sync_engine.py --seed` | PERFECT | `TaiwanStockInfo` 3402 rows fetched / DB distinct current summary 2798；FRED 4 series 成功；總寫入 `7287` rows。 |

---

## 3. CoreScore v0.2 Snapshot Restore

### 3.1 初始 dry-run

指令：

```bash
python scripts/core/core_universe_builder.py --dry-run --as-of-date 2026-05-14
```

結果：
- verdict: `WARNING`
- preflight PASS/WARN/FAIL: `8/0/0`
- v0.2 contract PASS/WARN/FAIL: `10/10/0`
- 原因：空 DB 尚無個股歷史資料，`TaiwanStockPriceAdj` / `MonthRevenue` / `PER` / `Institutional` / `Margin` / `FinancialStatements` 皆無 `as_of_date <= 2026-05-14` rows。
- 分層 dry-run：core `120`、convex `30`、research `2270`、quarantine `378`。

裁決：合法中繼狀態，只能用來建立 bootstrap membership 供後續 `--universe core` 補資料，不得宣稱為最終 CoreScore coverage。

### 3.2 Bootstrap commit

指令：

```bash
python scripts/core/core_universe_builder.py --commit \
  --as-of-date 2026-05-14 \
  --special-rebalance-reason "database full rebuild restore accepted CoreScore v0.2 snapshot bootstrap before historical refill"
```

結果：
- verdict: `WARNING`
- preflight PASS/WARN/FAIL: `7/1/0`
- v0.2 contract PASS/WARN/FAIL: `10/10/0`
- written rows: `5599`
- reason 留痕：`rebalance_mode=special`

裁決：此 commit 僅作 DB full rebuild restore 的 bootstrap membership，不是年度正式重選。

### 3.3 Core+Convex 730 日資料補刷

指令：

```bash
python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730
```

結果：
- success: `1308`
- warning: `46`
- failed: `0`
- skipped: `0`
- rows written: `688867`
- elapsed: `602.09 s`
- verdict: `WARNING`

問題與裁決：
- 46 warnings 均為部分股票 / dataset API 回傳 0 筆，包含部分 margin / dividend 等資料缺口。
- 無 failed，未阻斷 CoreScore v0.2 restore。

### 3.4 歷史資料補齊後 dry-run

指令：

```bash
python scripts/core/core_universe_builder.py --dry-run --as-of-date 2026-05-14
```

結果：
- verdict: `WARNING`
- preflight PASS/WARN/FAIL: `8/0/0`
- v0.2 contract PASS/WARN/FAIL: `16/4/0`
- Core sync coverage:
  - price coverage: `146/150`
  - revenue coverage: `140/150`
  - financial coverage: `147/150`
- market data loaded:
  - price: `147` stocks
  - revenue: `148` stocks
  - financial: `148` stocks
  - institutional: `147` stocks

問題與裁決：
- warnings 來自全市場未全量同步造成 market-level zero coverage，以及 `TaiwanStockInfo as-of candidates=61 < 150` 的 fallback warning。
- core scope coverage 已恢復至先前可接受基線，符合 v5.4.22 既有裁決。

### 3.5 最終 restore commit

指令：

```bash
python scripts/core/core_universe_builder.py --commit \
  --as-of-date 2026-05-14 \
  --special-rebalance-reason "database full rebuild restore accepted CoreScore v0.2 snapshot after historical refill"
```

結果：
- snapshot: `core_universe_20260514_core_universe_policy_v0_2`
- status: `committed`
- verdict: `WARNING`
- preflight PASS/WARN/FAIL: `7/1/0`
- v0.2 contract PASS/WARN/FAIL: `16/4/0`
- written rows: `5599`
- tiers:
  - core: `120`
  - convex: `30`
  - research: `2270`
  - quarantine: `378`
- notes include: `rebalance_mode=special; special_rebalance_reason=database full rebuild restore accepted CoreScore v0.2 snapshot after historical refill`

裁決：最終核心股 snapshot 已恢復；WARNING 屬特殊治理留痕與既知 coverage/fallback warning，不是失敗。

---

## 4. 驗收與下游 smoke check

| Step | 指令 | 結果 | 摘要 |
|---|---|---|---|
| 4C | `python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14` | PERFECT | 報告 `core_universe_audit_20260517_1338.md`；PASS/WARN/FAIL = `36/0/0`。 |
| 2C-rerun | `python scripts/core/db_utils.py` | PERFECT | §6.7 core assets = `150`。 |
| 8 schema | `python scripts/core/feature_store_schema.py --init` | PERFECT | Feature Store 3 tables 建立；PASS/WARN/FAIL = `6/0/0`。 |
| 9 dry-run | `python scripts/core/feature_store_builder.py --dry-run --as-of-date 2026-05-14 --feature-set-version feature_set_v0.1_h20_production_current --label-horizon 20` | PERFECT | 鎖定 `core_universe_20260514_core_universe_policy_v0_2`；150 stocks / 27 features / 3980 feature rows / 47 imputed；未寫入 DB。 |
| final audit | `python scripts/maintenance/audit_supply_chain.py --include-logs` | PERFECT | 報告 `compliance_audit_20260517_1339.md`；PASS/WARN/FAIL = `33/0/0`。 |

---

## 5. DB 最終摘要

| Table / Query | Result |
|---|---:|
| `TaiwanStockInfo` | `2798` |
| `FredData` | `3885` |
| `core_universe_snapshot` | `1` |
| `core_universe_membership` | `2798` |
| `core_universe_scores` | `2798` |
| `feature_store_snapshot` | `0` |
| `feature_definition` | `0` |
| `feature_values` | `0` |
| `TaiwanStockPriceAdj` date range | `2024-05-17` to `2026-05-15` |
| `TaiwanStockPriceAdj` rows / stocks | `69885` rows / `148` stocks |
| §6.7 current core assets | `150` |

Latest snapshot:

```text
snapshot_id = core_universe_20260514_core_universe_policy_v0_2
status = committed
total_candidates = 2798
core = 120
convex = 30
research = 2270
quarantine = 378
```

---

## 6. 問題紀錄與裁決

1. **`db_utils.py` bootstrap warning**
   - 現象：核心股尚未 commit 前，§6.7 core universe query returned 0 rows。
   - 裁決：合法 bootstrap warning；最終重跑後已 PERFECT，核心資產數 150。

2. **初始 CoreScore dry-run / commit warnings**
   - 現象：空 DB 無個股歷史資料，v0.2 contract 出現 10 warnings。
   - 裁決：合法中繼狀態；僅用於建立 bootstrap membership 以支援 `--universe core` 歷史補刷。

3. **`--special-rebalance-reason` 使用**
   - 現象：`2026-05-14` 非年度最後交易日，常規 commit 會被年度 guard 擋下。
   - 裁決：本次是 DB full rebuild restore accepted snapshot，屬 §6 特別治理例外；已在 snapshot notes 與 revision log 留痕，不是日常重選。

4. **核心資料補刷 WARNING=46**
   - 現象：部分股票 / dataset API 回傳 0 筆。
   - 裁決：無 failed，且核心必要 coverage 恢復到可接受基線；不阻斷核心股 snapshot restore。

5. **最終 CoreScore WARNING**
   - 現象：market-level coverage 仍有大量 zero coverage，且 `TaiwanStockInfo as-of candidates=61 < 150` 使用 latest registry fallback。
   - 裁決：符合既有 v5.4.22 裁決，因本次只補 core+convex 730 日資料，不是全市場歷史補刷；core scope coverage 已達標。

6. **下游資料未正式 commit**
   - 現象：`feature_store_snapshot` / `feature_definition` / `feature_values` 皆為 0。
   - 裁決：符合本次任務邊界；只做 Feature Store dry-run smoke check，不正式進行 production-current h20 training，也不升 v5.4.23。

7. **ad-hoc DB summary 查詢的 sandbox 連線錯誤**
   - 現象：一般 sandbox 下 `python -c` DB query 出現 `OperationalError`。
   - 裁決：執行環境權限 / DB 連線限制；改用授權模式查詢成功，不是 schema 或資料錯誤。

---

## 7. 最終裁決

本次已自使用者刪除 all tables 後，依 `系統架構大憲章_v5.4.22.md` 從零重建至目前 accepted CoreScore v0.2 核心股 snapshot。

最終狀態：

- Raw schema: **PERFECT**
- Core governance schema: **PERFECT**
- Supply-chain audit: **PERFECT**
- Seed ingestion: **PERFECT**
- Core+convex 730d sync: **WARNING but 0 failed**
- CoreScore v0.2 restore commit: **WARNING with audited special reason**
- Core universe audit: **PERFECT**
- §6.7 core assets: **150**
- Feature Store dry-run smoke check: **PERFECT**

結論：**目前核心股已恢復產生並通過驗收；本次重建符合憲章。後續仍不得把本次 special restore 解讀為日常重選，下一次常規核心股正式重選仍應等年度最後交易日後執行。**
