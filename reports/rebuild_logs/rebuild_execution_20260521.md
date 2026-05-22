# DB Rebuild 執行紀錄：從零 → 全市場全天數 + FRED 全歷史

- **開始時間**: 2026-05-21
- **觸發**: 使用者於本日刪除 DB 所有 tables（驗證：0 tables）
- **目標**: 依憲章 §二 序列 + §14.7-AM 治權範本執行
  - Step 0 → 1 → 2 → 2.5 → 2B → 2C → 3 → 4 → 4F → 8
- **reason 字串**: `DB rebuild bootstrap: 從零建立全市場全天數基準資料供 CoreScore v0.2 核心股選擇用`（44 字 ≥ 12 字 ✅）

## 階段對照表

| Phase | Steps | 預估時間 |
|---|---|---|
| Phase 1 | DB 狀態驗證 | 秒級 |
| Phase 2 | Step 0-3 重建（schema + governance + 雙稽核）| 5-10 分鐘 |
| Phase 3.1 | Step 4 種子灌溉 | 秒級 ~ 1 分鐘 |
| Phase 3.2 | Step 4F 全市場全天數 | **6-10 小時** ⚠️ |
| Phase 3.3 | Step 8 FRED 全歷史 | 秒級 |

## 執行紀錄

### Phase 1: DB 狀態驗證 ✅

- 時間: 開始時
- 結果: DB tables = 0（使用者已刪除）
- 治權判定: 觸發 §6.8.7 第 (4) 條 (1) DB rebuild bootstrap 合法情境

### Phase 2: Step 1-3 重建 ✅

| Step | 程式 | 結果 |
|---|---|---|
| 1 | `path_setup.py v4.46` | PERFECT (BOOTSTRAP-DEFERRED) |
| 2 | `data_schema.py v2.16 --init --force` | PERFECT (11/0/0 API + 13/13 DDL, 3.66s) |
| 2.5 | `audit_api_schema_compliance.py v0.3 --include-fred` | PERFECT (459/0, 2.77s) |
| 2B | `core_universe_schema.py v0.3 --init` | PERFECT (9/0/0 preflight + 7/7 DDL, 0.17s) |
| 2C | `db_utils.py v2.47` | WARNING (合法 BOOTSTRAP — §6.7 core universe query 0 rows) |
| 3 | `audit_supply_chain.py v1.19 --include-logs` | PERFECT (29/0/0, ~3s) |

### Phase 3.1: Step 4 種子灌溉 ✅ 但發現問題

執行: `sovereign_sync_engine.py v1.19 --seed`
結果: PERFECT (15.30s)

**寫入細項**:
- TaiwanStockInfo: 3,404 筆（**比 §14.7-AM 預期之 ~2,798 多 606 筆**；可能含 ETFs / 權證等）
- FRED/DFF: 26,256 筆（全歷史 1954-07-01 → 今天）
- FRED/UNRATE: 939 筆
- FRED/T10Y2Y: 12,489 筆
- FRED/VIXCLS: 9,189 筆

#### 🔴 問題 #1：`--seed` 隱含包含 FRED 全歷史灌溉

**觀察**：執行 `--seed` 不僅灌 `TaiwanStockInfo`，同時自動觸發 FRED 4 序列全歷史灌溉。

**治權影響**：
- 憲章 §14.7-AM 治權範本明文「Step 4 → Step 4F → Step 8」3 步序列，將 Step 8 (`--source fred`) 列為**獨立第 III 步**
- 但程式 `--seed` 模式預設行為已包含 FRED 全灌（per sync_fred() 在 seed_mode 也被調用）
- 此造成「**Step 8 在 Step 4 後執行屬重複動作**」之治權描述不一致

**後續修正建議**（暫不修正，記錄供後續入憲）：
1. 修正憲章 §14.7-AM：明文「Step 4 `--seed` 已包含 FRED 4 序列；Step 8 屬冗餘但 idempotent」
2. 或修正程式：`--seed` 不應自動觸發 FRED（FRED 應由 `--source fred` 獨立執行）
3. 或修正治權範本：刪除 Step 8 為獨立步驟之表述

#### 🟡 觀察 #1：TaiwanStockInfo 筆數 3,404 vs 預期 2,798

- 治權範本預期：2,798 支股票（per §6.8.7 第 (4) 條）
- 實際 API 回傳：3,404 筆 → 多 606 筆（22%）
- 推測：FinMind `TaiwanStockInfo` API 包含 ETFs / 權證 / 上市櫃 + 興櫃所有標的
- 後續驗證：Step 4F 灌資料時應觀察實際參與 sync 之 stock_id 集合大小

### Phase 3.2: Step 4F 全市場全天數 ❌ 重大治權邊界缺陷揭露

執行: `sovereign_sync_engine.py v1.19 --universe full --all --dataset-batched --workers 4 --dynamic-quota --special-full-market-reason "..."`
背景 PID: `b9pdxfvnf`（已於 14.02s 提前結束）

**結果**: WARNING (主權判定 — 0 個 FinMind 股票被 sync)
**寫入細項**:
- FinMind stocks: **0 筆**（❌ 全市場 0 標的）
- FRED/DFF: 26,256 筆（再次灌；idempotent）
- FRED/UNRATE: 939 筆
- FRED/T10Y2Y: 12,489 筆
- FRED/VIXCLS: 9,189 筆

#### 🔴🔴 問題 #2（CRITICAL）：`--universe full` 在「從零 DB rebuild bootstrap」情境下無法執行（雞與蛋治權缺陷）

**Root Cause**：
```
sovereign_sync_engine.py L767-776 之 _resolve_stocks():
  if universe == 'full':
    stocks = get_core_stocks_from_db(tiers=UNIVERSE_TIERS['full'])
    # UNIVERSE_TIERS['full'] = ('core_universe', 'convex_universe', 'research_universe', 'quarantine_universe')
    # get_core_stocks_from_db 之 §6.7 SQL 查詢:
    #   SELECT ... FROM core_universe_membership m
    #   JOIN core_universe_snapshot s ON m.snapshot_id = s.snapshot_id
    #   WHERE s.status = 'committed' AND m.core_tier IN (tiers)
  if not stocks: warning "full universe 無標的"
```

**治權邊界雞與蛋**：
- Step 4F (`--universe full`) **需要** `core_universe_membership` 已 committed
- `core_universe_membership` committed **需要** Step 4B (`core_universe_builder.py --commit`) 已執行
- Step 4B 評分**需要** raw OHLC data
- raw OHLC data 灌溉**需要** universe 已 commit...（循環依賴）

**治權影響**：
- 憲章 §14.7-AM 治權範本明文「Step 4F = 全市場全天數限定治理例外」之「DB rebuild bootstrap」合法情境**實際無法執行**
- §6.8.7 第 (4) 條五類合法情境之 (1) DB rebuild bootstrap 屬**治權邊界缺陷**

**後續修正建議**（暫不修正，記錄供使用者裁決）：

| 選項 | 動作 | 影響範圍 |
|---|---|---|
| **甲** | **修正 sovereign_sync_engine.py**：對 `--universe full` 加 `TaiwanStockInfo` fallback（當 universe 表為空時自動從 TaiwanStockInfo 取所有 stock_id 作為 bootstrap candidate pool）| 程式 v1.19 → v1.20；憲章 §14.7-AM 加 bootstrap fallback 註記 |
| **乙** | **改用等價序列**：Step 4 `--seed` → Step 4D `--universe research --all --full-history`（research universe 由 TaiwanStockInfo 候選池產生）→ Step 4B `--commit` → Step 4G `--universe core --all --full-history`（core 全天數補刷）| 不改程式；但偏離 §14.7-AM 治權範本 |
| **丙** | **修正治權範本**：明文「從零 DB rebuild bootstrap 不適用 Step 4F；應改用乙之等價序列」入憲 §14.7-AM 補強 | 不改程式；憲章 patch |
| **丁** | **強制 bootstrap snapshot**：先 `--seed` → `core_universe_builder --commit --special-rebalance-reason "..."` 強制產生 bootstrap snapshot（即使 0 OHLC 也以 fallback mode commit）→ Step 4F 此時可執行 | 程式不改；但 bootstrap snapshot 將為低品質 |

#### 🟡 觀察 #2：`--universe full` 觸發後仍會跑 FRED 灌溉（雙重灌入）

第一次 `--seed` 已灌 FRED 4 序列共 48,873 rows；本次 `--universe full --all` 又灌一次相同 FRED 4 序列（idempotent UPSERT）。
- 此確認問題 #1：sync engine 對 FRED 之觸發條件需細化（per universe 應該只跑 FinMind，不重複跑 FRED）

---

### 待裁決事項彙整

| # | 問題 | 嚴重性 | 建議修正方向 |
|---|---|---|---|
| 1 | `--seed` 隱含 FRED 全灌（治權範本記載 Step 8 為獨立步驟）| 中 | 文件 / 程式 對齊 |
| 2 | `--universe full` 在 bootstrap 情境無法執行（雞與蛋）| **高（CRITICAL）** | 甲/乙/丙/丁 4 選項待裁決 |
| 3 | `--universe full` 重複觸發 FRED 灌溉 | 低 | per universe 邏輯細化 |


