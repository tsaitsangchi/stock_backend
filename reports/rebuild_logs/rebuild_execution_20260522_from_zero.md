# Rebuild Execution Log — 從零執行至 sovereign_sync_engine.py 全市場全天數 + FRED 全歷史

- **執行日期**: 2026-05-22
- **執行者**: Claude Code (Opus 4.7)
- **目的**: 依憲章 §二 維運矩陣 + §14.7-AM 4 步序列,從清空之資料庫從零執行至全市場全天數(FinMind + FRED)
- **憲章版本**: v6.0.0 (`reports/系統架構大憲章_v6.0.0.md`)
- **PROJECT_ROOT**: `/Users/hugo/project/stock_backend`

---

## 計劃執行序列 (per §14.7-AM 4 步序列 + §二 前置步驟)

| # | 階段 | 指令 | 預期耗時 | 狀態 |
|---|---|---|---|---|
| 0 | 環境檢查 | venv / pip / .env / DB / API tokens | — | 進行中 |
| 1 | 路徑自癒 | `python scripts/core/path_setup.py` | 秒級 | pending |
| 2 | Raw schema | `python scripts/core/data_schema.py --init --force` | 數秒 | pending |
| 2.5 | Schema audit | `python scripts/maintenance/audit_api_schema_compliance.py --include-fred` | 數秒 | pending |
| 2B | Governance schema | `python scripts/core/core_universe_schema.py --init` | 秒級 | pending |
| 2C | DB 依賴檢查 | `python scripts/core/db_utils.py` | 秒級 | pending |
| 3 | 供應鏈驗收 | `python scripts/maintenance/audit_supply_chain.py --include-logs` | 數秒 | pending |
| 4 (I)   | Seed | `python scripts/ingestion/sovereign_sync_engine.py --seed` | 秒級 | pending |
| 4B (II) | Bootstrap init | `python scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-22 --special-rebalance-reason "DB rebuild bootstrap 2026-05-22 init"` | 秒級 | pending |
| 4F (III) | FinMind 全市場全天數 | `python scripts/ingestion/sovereign_sync_engine.py --universe full --all --dataset-batched --workers 4 --dynamic-quota --special-full-market-reason "DB rebuild bootstrap 2026-05-22 full-market irrigation"` | ~6-10h | pending |
| 4B (IV) | Bootstrap final | `python scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-22 --special-rebalance-reason "DB rebuild bootstrap 2026-05-22 final"` | 1-2 分鐘 | pending |
| 8 (V) | FRED 全歷史 | `python scripts/ingestion/sovereign_sync_engine.py --source fred` | 秒級 | pending |

---

## 執行過程紀錄

### Step 0. 環境檢查 — ✅ Done (with issues)

**檢查項目**:
- venv (Python 3.12.13) ✓
- requirements.txt 安裝(189 套件)— 過程順利但需配套修補(見 Issue #1)
- .env 14 變數齊全 ✓(但 `PROJECT_ROOT` 路徑需修正,見 Issue #2)
- PostgreSQL 17.10 連線正常 ✓,清空狀態確認
- FinMind Token / FRED API Key 存在 ✓

**Issue #1**:xgboost 無法 import — 缺 OpenMP runtime
- 錯誤:`Library not loaded: @rpath/libomp.dylib`
- 修補:`brew install libomp`(11 files / 1.8MB)
- 後續修正建議:於 README.md / `CLAUDE.md` 補入 macOS 必要前置 `brew install libomp` 一行;或在 `requirements.txt` 上方加入 macOS 前置條件註解

**Issue #2**:`.env` `PROJECT_ROOT=/home/hugo/...`(Linux 路徑),macOS 實際是 `/Users/hugo/...`
- 雖然 macOS 上 `/home → Users` symlink 可解析,但 `path_setup.py v4.46` 對 `PROJECT_ROOT` 與物理路徑做**嚴格字串比較**,判定 `MISMATCHED` → FAILED
- 修補:把 `.env` 改為 `PROJECT_ROOT=/Users/hugo/project/stock_backend`
- 後續修正建議:`path_setup.py` 比較邏輯改用 `os.path.realpath()` 解析後再比對(支援 symlink 同義路徑);或於 `.env.example` 補入 macOS / Linux 路徑分流註解;此外 `.env.example` 內 `MLFLOW_TRACKING_URI=sqlite:///${PROJECT_ROOT}/...` 也預設 Linux,建議補上 OS 區分

---

### Step 1. path_setup.py 路徑自癒 — ✅ PERFECT

修正 `.env` 後重跑:`物理基準 ✓ / 錨點對齊 MATCHED / 25 維治理 / 主權判定 PERFECT`(155.81 ms)。
混合日誌 `BOOTSTRAP-DEFERRED`(尚未建 logging table,屬合憲過渡狀態)。

Log: `reports/rebuild_logs/step1_path_setup_20260522.log`

---

### Step 2. data_schema.py --init --force — ✅ PERFECT

建 13 張 raw DDL(11 業務表 + 2 logging table):
- API Contract First probe:11/0/0(TaiwanStockPrice / PriceAdj / PER / InstitutionalInvestors / MarginPurchase / Shareholding / FinancialStatements / MonthRevenue / Dividend / Info / FredData)
- DDL 重鑄:13/13 全 PASS,耗時 3.48 秒
- 主權判定:PERFECT ALIGNMENT

Log: `reports/rebuild_logs/step2_data_schema_20260522.log`

---

### Step 2.5. audit_api_schema_compliance --include-fred — ✅ PERFECT

9 層動態檢驗皆 PASS(此時 Layer E-I 為 vacuous PASS,資料尚未灌入):
- A DDL↔DB:119/0 / B API↔DDL Type:102/0 / C 長度精度:83/0 / D NULL ratio:103/0
- E PK Unique:11/0 / F Duplicate:13/0 / G Date Continuity:11/0 / H Referential:9/0 / I Value Range:7/0
- 耗時:3.73 秒;報告:`reports/api_schema_compliance_audit_20260522_2209.md`

Log: `reports/rebuild_logs/step2_5_audit_schema_20260522.log`

---

### Step 2B. core_universe_schema.py --init — ✅ PERFECT

建 7 張 derived governance tables(core_universe_policy / snapshot / membership / scores / theme_taxonomy / stock_theme_map / universe_revision_log)。
- preflight 9/0/0(`RAW_COLUMN_INHERITANCE` 全對齊)
- 耗時:279 ms;主權判定 PERFECT

Log: `reports/rebuild_logs/step2B_core_universe_schema_20260522.log`

---

### Step 2C. db_utils.py 前置依賴檢查 — ⚠️ BOOTSTRAP WARNING(預期)

- DB 連線 SUCCESS,延遲 38 ms
- pipeline_execution_log [8 欄完整] / data_audit_log ACTIVE
- WARNING:`§6.7 core universe query returned 0 rows`(此時 `core_universe_membership` 尚無 committed snapshot,屬合憲過渡;per §3.2 Step 2C 接受標準可進入 Step 3)

Log: `reports/rebuild_logs/step2C_db_utils_20260522.log`

---

### Step 3. audit_supply_chain.py --include-logs — ✅ PERFECT

PASS=29 / WARN=0 / FAIL=0
- API 契約驗收 + DB 13 表實體 schema + FRED series freshness + log 兩表全通過
- 報告:`reports/compliance_audit_20260522_2210.md`

Log: `reports/rebuild_logs/step3_audit_supply_chain_20260522.log`

---

### Step 4 (I). sovereign_sync_engine --seed — ✅ PERFECT

21.23 秒、52,286 筆寫入:
- TaiwanStockInfo: 3,409 筆(全市場資產名冊)
- FRED/DFF: 26,257 / UNRATE: 939 / T10Y2Y: 12,490 / VIXCLS: 9,191

⚠️ **意外發現(非問題,屬正向觀察)**:`--seed` 在 v1.21 中**已連帶自動觸發 FRED 4 序列全歷史灌溉**(原預期需另跑 Step 8 `--source fred`)。憲章 §14.7-AM 已有註記:「Step 4 `--seed` 與 Step 4F `--universe full` 之預設邏輯皆會自動觸發 FRED 全灌;獨立執行屬 idempotent UPSERT」,與本實證一致。

Log: `reports/rebuild_logs/step4_seed_20260522.log`

---

### Step 4B (II). core_universe_builder.py --commit bootstrap_init — ⚠️ WARNING(預期)

bootstrap snapshot 成功 committed,進入 `latest_registry_fallback` mode(per §6.4 邊界裁決,合憲過渡):
- snapshot_id: `core_universe_20260522_core_universe_policy_v0_2`(status: committed, as_of_date: 2026-05-22)
- total_candidates: 2,771 / core: 120 / convex: 30 / research: 2,243 / quarantine: 378
- WARNING 原因:raw OHLC 尚無資料 → V0.2-CONTRACT 各表 0 rows / coverage 全 0 → CoreScore = 0(合憲)
- 耗時:3.16 秒;written_rows: 5,545

Log: `reports/rebuild_logs/step4B_init_20260522.log`

---

### Step 4F (III). sovereign_sync_engine --universe full --all --full-history — 🟡 進行中

啟動參數:
```
--universe full --all
--dataset-batched --workers 4 --dynamic-quota
--special-full-market-reason "DB rebuild bootstrap 2026-05-22 full-market irrigation from-zero"
```

- 啟動時間:2026-05-22 ~22:11
- PID:20275
- 預計耗時:6-10 小時
- 涵蓋:FinMind 10 raw tables × 2,798 支 × 各 (stock_id, dataset) 自 API 最早可得日期 → 今天
- §6.8.7 第 (4) 條治理例外已觸發,reason 字串已寫入 lifecycle context
- 完成後將自動執行 audit_supply_chain.py + audit_source_availability.py + 留 reports/full_market_sync_*.md

監控設定:
- 背景 Monitor (task b1o8pha71) 追蹤 FAILED / ERROR / 402/403/429 / §7.6 A5 預警 / 主權判定
- log 檔:`reports/rebuild_logs/step4F_finmind_full_20260522.log`

執行中即時觀察:
- 開頭數十秒已平行 fetch 多檔 TaiwanStockPrice(00400A / 00401A / 00403A / 0050 / 0051 等);4 workers 並行運作
- 首分鐘約 80 個 fetch 啟動;dataset-batched 模式按 dataset 分批處理

---

### Step 4B (IV). bootstrap_final — ⏸ 等待 Step 4F 完成

待 Step 4F 完成後,重跑 `core_universe_builder --commit --as-of-date 2026-05-22 --special-rebalance-reason "DB rebuild bootstrap 2026-05-22 final"` 以 real-data snapshot 覆蓋 bootstrap snapshot。

---

### Step 8 (V optional). --source fred — ⏸ 等待

FRED 已在 Step 4 自動觸發;此步驟為 idempotent UPSERT,可選。

---

## 已發現問題彙整(供後續修正)

| # | 嚴重度 | 問題 | 修正建議 |
|---|---|---|---|
| 1 | Medium | macOS 上 xgboost 缺 `libomp.dylib` | 在 `README.md` / `CLAUDE.md` 補入 macOS 必要前置 `brew install libomp`;或於 `requirements.txt` 上方註解 |
| 2 | Medium | `.env` `PROJECT_ROOT` 用 Linux 路徑,在 macOS 上被 path_setup 嚴格比對判 FAILED | `path_setup.py` 改用 `os.path.realpath()` 解析 symlink 後再比較;`.env.example` 補上 OS 分流註解 |
| 3 | Low | 同 #2 連帶:`.env.example` 之 `MLFLOW_TRACKING_URI` 預設仍用 `${PROJECT_ROOT}` 衍生,Linux/macOS 自動跟隨;但若 PROJECT_ROOT 校正不及則波及 MLflow 路徑 | 同 #2 修補 |
| 4 | **High** | Step 4F 跑 16 分鐘後,4 個 worker **同時撞 FinMind HTTP 402**(資料集付費門檻),依 §7.4 各自進入 `sleep 1800s`(30 分),CPU → 0%,**整條 pipeline 集體停擺 30 分**。停擺前 5,316,095 筆 TaiwanStockPrice 已成功寫入。觸發點落在 stock_id ≈ 3388–3402(電子類)的同一個 dataset 切片內。| (a) `sovereign_sync_engine._throttled_request()` 之 402 處理改為**全局單一退避**(只休一次 1800s,其他 workers 繼續做其他 stock),避免 N 個 worker × 1800s 同步浪費;(b) 觸發 402 後對該 (stock_id, dataset) 立即 `mark_skipped` 並寫 `data_audit_log`,**不要**等 1800s 後再決定 skip(因為實際上若 stock 真為 paywall,30 min 後還是 402);(c) 憲章 §7.4 條款檢討是否該由「單次探測重試」改為「立即略過」對 paywall 類 402;(d) 文件補上「402 cascade」現象與緩解策略;(e) 考慮對 `--workers > 1` 加入 worker-level 402-circuit-breaker:若同一分鐘內 ≥ 2 worker 撞 402 即進入單一 throttle |

(以下持續更新 — Step 4F 進行中可能再揭露問題;目前 22:27 等待 22:41 workers 甦醒)

---
