# 從零 → 全量 raw data 抓取:問題、根因、修法與可行序列(2026-06-08→09）

**用途**：記錄「刪除全部 table → 從零用 FinMind/FRED API 建出全市場 raw data」**實跑過程遭遇的問題、根因、修正、與最終可行序列**。供日後重建與跨機接續直接照走。
**治權對齊**：§14.7-DD(12-PHASE)+ §14.7-DJ(pure-generic ingestion)+ §14.7-AM(從零→全市場序列)+ §一.12/§二.6(長跑治權)。
**誠實聲明（§一.10 / AP-3）**：本檔記錄之問題/修法/序列為**實跑驗證**(DB query / log / 實際 error,source-traceable);**全量抓取本身為 in-progress**(本檔寫成時 TaiwanStockPrice ~69%,dataset 1/10);**完成 + PHASE 5 reconcile 結果待跑完補入**(不預先宣稱完成)。

---

## §0 背景：為何會有這些問題

- 用戶 drop 全部 table → 要求**從零**用 FinMind/FRED 建 raw data。
- **根本原因**：2026-06-08 之 **§14.7-DJ pure-generic 重構**(退役 DATASET_REGISTRY → raw 表改 generic auto-schema **於 sync 時才建**)**從未做 from-zero 端到端測試**;舊重建序列多支程式仍假設「`data_schema --init` 在 PHASE 1 預建 11 張空 raw 表」。table 建立**時機**從 PHASE 1 移到 PHASE 4 → 一連串「在表還沒建時就去查表」之 `UndefinedTable` / 順序錯。
- **這些都不是「資料來源/抓取」問題**(來源只有 FinMind+FRED,抓取本身 OK),全是**系統內部順序假設**未跟上時機改動。

---

## §1 遭遇問題 → 根因 → 修法(依實跑發現順序)

### 問題 1：治理/特徵 schema preflight 失敗（PHASE 1）
- **現象**：`core_universe_schema --init` / `feature_store_schema --init` → `[PREFLIGHT-FAILED] TaiwanStockInfo missing`。
- **根因**：§14.7-DJ 後 `TaiwanStockInfo` 改由 `--seed`(generic)建,非 `data_schema --init`;但這兩支 schema 之 preflight 要求它先存在。
- **修法（純順序,無 code）**：把 `core_universe_schema/feature_store_schema --init` 移到 **`--seed` 之後**。實測:seed 後 9/0/0 PREFLIGHT PASS、欄位正確繼承 `TaiwanStockInfo` 之 `VARCHAR(255)`。
- **raw-data-only 註**：這兩支屬下游(universe/feature),**純抓 raw data 不需要**。

### 問題 2：`fetch_fred_data` 從零 `UndefinedTable`（PHASE 2b）
- **現象**：`psycopg2.errors.UndefinedTable: relation "fred_series" does not exist`(`get_all_safe_starts` 查詢處)。
- **根因**：v3.3 退役 `DDL_FRED`(原會先建空 `fred_series`)後,`get_all_safe_starts`(resume 最佳化:讀 DB max date 決定接續抓點)仍在 generic 建表**之前**就查 `fred_series`。
- **修法（B / API-first,code）**：在 `fetch_fred_data.fetch_fred_series` 把 resume 查詢移到 **table-exists 守門之後**(`SELECT to_regclass('fred_series')`);table 不存在 → `latest={}` → 直接從 FRED API 全抓 → generic 建表。**API 為源頭,DB 表為抓取之結果**。實測:從零 24 series / 70683 rows。

### 問題 3：universe bootstrap 從零全壞（原 PHASE 3）
- **現象 A（legacy,預設 `--mode legacy-corescore`）**：`TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'`（`minimum_bootstrap_size = self.core_limit + self.convex_limit`;此二參數 §14.7-BW DEPRECATED → 預設 None）。
- **現象 B（`--mode doctrine-native`）**：Stage 1 K-wave gate 過 → Stage 2+3 per-stock 查 `TaiwanStockPriceAdj` → `UndefinedTable`。
- **根因**：bootstrap(兩 mode)都**直接 query raw 價量/基本面表**,但 §14.7-DJ 讓這些表 PHASE 4 才建 → 不存在。**循環依賴**:PHASE 4 sync 需 PHASE 3 的 universe 當標的;PHASE 3 卻需 PHASE 4 才建的 raw 表。舊世界靠 `data_schema --init` 預建空表 + §14.7-AM `latest_registry_fallback` 過渡解,被時機改動打斷。

### 問題 4（關鍵方法決策）：raw data 與選股解耦 → `--universe full` roster fallback
- **洞察**：**raw data = 全名冊每支都抓,根本不需要 universe 選股**(選股是下游)。
- **修法（code,`sovereign_sync_engine._resolve_stocks`）**：`--universe full` 在 committed `core_universe_membership` **空或治理表未建**時,**fallback 到 `TaiwanStockInfo` 全名冊**(`get_db_stock_ids()`,去重 `sorted(set(...))`)。
  - 子問題 4a：`get_core_stocks_from_db` 在治理表未建時 **raise**(非回空)→ 原 `except: return []` 在 fallback 前就 return;改為 full 模式時 `except → stocks=[]` 落 fallback。
  - 子問題 4b：`get_db_stock_ids` 無 DISTINCT → 回 3437(2814 distinct + 623 重複)→ `sorted(set())` 去重避免 ~22% 重抓浪費。
- **實測**：`_resolve_stocks(None,'full')` → **2814 去重**;`--universe full --all` 從零直接抓全名冊全史。**雞與蛋之資料層解**,不碰下游 bootstrap。

### 問題 5：監看 DB 連線間歇 `FATAL`（監看層,非 sync）
- **現象**：5-min HB 進度檢查 `connection ... FATAL`(127.0.0.1 亦然)。
- **釐清**：`max_connections=100`、實際僅 **10 連線**(非耗盡);sync workers=4 高頻開關連線 → 新連線間歇被拒(15× retry 才連上)。**sync 自身連線正常、健康寫入**。
- **修法（監看腳本,非 production）**：進度改 **log 解析**(`/tmp/fullsync_*.log` 之 `FinMind: <stock> / <dataset>` 行 → 當前 dataset + 已完成 distinct 股數)為主,DB rows 改 best-effort 12× retry。

---

## §2 修正後「從零 → 全量 raw data」可行序列（已實跑驗證至 sync 啟動）

> **前置**：AC 插電(`pmset -g batt`)+ import smoke pass + DB 空。

```bash
# PHASE 1 — infra DDL(只建 raw-data 所需;下游治理/特徵表不需)
./venv/bin/python scripts/core/path_setup.py
./venv/bin/python scripts/core/data_schema.py --init --force          # 2 infra log 表(INFRA_TABLE_SCHEMAS)
./venv/bin/python scripts/ingestion/initialize_market_data.py         # stocks + pipeline_execution_log + data_audit_log

# PHASE 2 — Genesis 名冊（generic 建 TaiwanStockInfo）
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py --seed   # → TaiwanStockInfo 2814 distinct（+ FredData 4 series）

# PHASE 2b — FRED（generic 建 fred_series；含問題 2 修法）
./venv/bin/python scripts/fetchers/fetch_fred_data.py                  # → fred_series 24 series / ~70683 rows

# PHASE 4 — 全市場全史 raw sync（含問題 4 修法；caffeinate 包裹）
caffeinate -dimsu ./venv/bin/python scripts/ingestion/sovereign_sync_engine.py \
    --universe full --all --dataset-batched --workers 4 --dynamic-quota \
    --special-full-market-reason "DB rebuild from-zero raw data 2026-06-08"
# → _resolve_stocks fallback 全名冊 2814 → generic 自動建 10 raw 表（Price/PriceAdj/PER/法人/融資券/外資持股/財報/資產負債/月營收/股利）× 全史
# 實測啟動 2 分鐘:TaiwanStockPrice 已建 + 1.78M rows;ETA ~5-6 hr（前例 5h19m / ~81M rows）

# PHASE 5 — DB↔API 對帳（抓取完成後;你要的「實際 API 驗證 raw data」）
./venv/bin/python scripts/audit/audit_full_db_vs_api_reconcile.py --scope all   # 待跑
```

**關鍵差異 vs 舊 guide**：跳過 PHASE 3 bootstrap(universe 選股)—— raw data 由 `--universe full` 之 roster fallback 直接抓全名冊;選股留待 raw data 齊全後(下游,另行修問題 3)。

---

## §3 程式改動清單（本次 from-zero 修法;狀態：未 commit）

| 檔案 | 改動 | 問題 |
|---|---|---|
| `scripts/fetchers/fetch_fred_data.py` | `fetch_fred_series`:resume 查詢移到 `to_regclass` table-exists 守門後(API-first)| 2 |
| `scripts/ingestion/sovereign_sync_engine.py` | `_resolve_stocks`:`--universe full` 在 membership 空/治理表未建時 fallback 全名冊(去重);except-path 落 fallback | 4/4a/4b |

> `core/db_utils.py` **未改**(問題 2 採 B = 在 fetch_fred 守門,不動共用 `get_all_safe_starts`)。
> 問題 1 為**純順序**(無 code);問題 3(bootstrap)**未修**(下游選股,raw-data 階段以 roster fallback 繞過)。
> 監看腳本 `/tmp/sync_progress.py` 為**暫存非 production**。

---

## §4 長跑治權（本次落實）

- **caffeinate**：`caffeinate -dimsu` 包裹;實測 `pmset -g assertions` = `PreventSystemSleep=1` + `PreventUserIdleSystemSleep=1`(SUPREME 準則)。
- **§一.12 5-min 回報**：Monitor `bdh0boch5` 每 300s 貼 dataset 進度(log 解析:當前 dataset / X-of-2814 股 / datasets_seen Z/10);sync 結束自停 + 完成通知。
- **§二.6 SHMM**：本次以「Monitor HB(log-based)+ background run 完成通知 + DB 為真實進度真相」組成;DB 連線間歇 FATAL 已以 log-based 規避。

---

## §5 完成驗收（待跑完補入,§一.10 不預先宣稱）

- [ ] 10 個 raw 表全建 + 全史 rows(實際數待 sync 完成 DB query)
- [ ] `audit_supply_chain --include-logs` PERFECT(PHASE 4 gate)
- [ ] `audit_full_db_vs_api_reconcile --scope all`：raw `TaiwanStockPrice` value_mismatch=0(PriceAdj 除權息回溯為良性 mismatch)
- [ ] 總耗時(以 sync stdout `Total elapsed` 為準)
- [ ] 2 code fix commit 封存

---

**事實來源（§一.10）**：問題/error 訊息自實跑 stdout;`max_connections=100`/連線 10、`_resolve_stocks→2814`、fred_series 24/70683、TaiwanStockPrice 1.78M rows 自 DB query;啟動 2026-06-08 23:55、TaiwanStockPrice ~69%(1950/2814)自 log + HB。完成數據待 sync 跑完補。
