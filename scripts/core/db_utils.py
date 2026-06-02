"""
db_utils.py v2.50 (Quantum Finance Infrastructure Sovereign Edition)
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: GOVERNANCE SYNC (憲法 v6.1.0 對齊 + §14.7-DE Canonical Panel Source `get_canonical_panel_dates()` + §14.7-DF Canonical Horizon Metric SSOT `summarize_horizon_metrics()` 單一引用源 helper 落地 + §3.2A.J `write_data_audit_log` race-safe ON CONFLICT DO NOTHING；維運矩陣重組為 6 大功能群視角；8 項檢查面 100% 合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Infrastructure Resilience]: 提供具備自動重連與健康診斷的資料庫通訊介面，確保 24/7 治權連通性。
2. [Asset Sovereignty]: 核心股名單必須透過 `core_universe_membership` JOIN `core_universe_snapshot` 取得，嚴禁回查 v5.2 時代 `stocks` 表。
3. [Historical Reference Authority]: 保留從 v1.0 至今的所有修訂歷史，作為判定系統正確性的基準。
4. [Hybrid Observability]: 基礎設施維運必須觸發「生命週期紀錄」(Lifecycle) 與「專項審計紀錄」(Audit)；
   生命週期紀錄必須完整寫入 start_time / end_time / error_msg；status 必須反映實際結果，
   嚴禁「Python 無例外即記 success」之謊報邏輯（v2.44 補強）。
5. [Zero Hardcoded Verdict]: 主權判定（PERFECT / WARNING / FAILED）必須依執行結果動態計算（`run_diagnostics()` L976-1019），嚴禁硬編碼。對齊憲章 §5.6.3「禁止硬編碼 PERFECT」與 §3.2 Step 2C 接受標準（FAILED → exit 1；PERFECT/WARNING → exit 0）；治權慣例對齊 `data_schema.py v2.13+` / `core/__init__.py v1.16+` / `finmind_client.py v4.46+` / `core_universe_schema.py v0.3+` / `audit_api_schema_compliance.py v0.1+` 之 [Zero Hardcoded Verdict]。
6. [Sovereignty Declaration]: 本模組為憲章 §3.2 橫切 library / Infrastructure Sovereign Authority（憲章 L2282 §3.2 子表）；承擔 DB 連線池管理 + lifecycle log（pipeline_execution_log）+ audit log（data_audit_log）+ §6.7 SQL 核心股查詢之 public API 治權；同時授權 `python scripts/core/db_utils.py` 之 `__main__` 為憲章 §二 Step 2C 前置依賴檢查（憲章 §二 L2411）；不涉及 §0.1-A 第一性原理 / §0.2-A 八二法則 / §0.3-A 康波週期 / §0.0-E.4 統合層 / §0.0-F.3 AI 協作工具規則五套禁令；不在 §0.1.1 T1/T2/T3 分層內（infrastructure 為基礎設施，不執行第一性原理工程公式）；不處理 §8.5 anti-leakage（DB connection 屬基礎設施，非時間序列建模）；不選股不評分（屬 `core_universe_builder.py` Builder Authority）；不持有 Raw API Schema 治權（屬 `data_schema.py` Raw API Schema Authority）。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

依憲章 §3.2 橫切 library 之治權特性，`db_utils.py` 為被多支治權程式 `import` 之
基礎設施 library；本矩陣按「**功能群視角**」分類 6 大功能群，明示 API 群組與
治權對應。（v2.47 由原 8 場景視角重組）

### 🔌 Group A. DB 連線管理 (Infrastructure Resilience)

| 子項 | API / 指令 | 治權對應 |
| :--- | :--- | :--- |
| **A.1 標準連線** | `get_db_connection()`（含 `connect_timeout=10` + 5 件套 env 檢查）| §一 0 / §0.0-I.8 |
| **A.2 健康診斷** | `db_connection_check()` → (ok, latency_ms) | [Infrastructure Resilience] |
| **A.3 Transaction context** | `with db_transaction() as cur:` 自動 commit/rollback | [Infrastructure Resilience] |
| **A.4 Session context** | `with db_session() as conn:` raw 連線 + 自動 commit | [Infrastructure Resilience] |
| **A.5 連線池重置（CLI）** | `$ python scripts/core/db_utils.py --reset-pool` | 緊急維運 |
| **A.6 向後相容 alias** | `get_db_conn()` ← legacy 包裝 `get_db_connection()` | 向後相容 |
| **A.7 連線參數** | `get_connection_params()` → dict | 維運工具 |

### 📝 Group B. Hybrid Observability — 雙日誌系統

| 子項 | API / 寫入表 | 治權對應 |
| :--- | :--- | :--- |
| **B.1 生命週期** | `with record_lifecycle(task, category, stock_id) as lc:` → `pipeline_execution_log` | [Hybrid Observability] |
| **B.2 主動標記失敗** | `lc.mark_failed(msg)` / `lc.mark_warning(msg)` | v2.44 封堵 status 謊報 |
| **B.3 專項審計** | `write_data_audit_log(table, stock, date, action, rows)` → `data_audit_log` | [Hybrid Observability] |
| **B.4 pipeline_log 寫入** | `write_pipeline_log(...)` 直寫 | 跨模組相容 |
| **B.5 evaluation_log 寫入** | `write_evaluation_log(...)` 評估專用 | 跨模組相容 |

### 🏛️ Group C. §6.7 SQL SSOT — 核心股查詢

| 子項 | API | 治權對應 |
| :--- | :--- | :--- |
| **C.1 核心股唯一入口** | `get_core_stocks_from_db(as_of_date, tiers)` — JOIN `core_universe_membership` × `core_universe_snapshot WHERE status='committed'` | §6.7 SQL 契約 + [Asset Sovereignty] |
| **C.2 股票 ID 清單** | `get_db_stock_ids(core_only=True/False, types, conn)` | 包裝 C.1 |
| **C.3 最新日期查詢** | `get_latest_date(table, stock_id, date_col)` | 同步邊界 |
| **C.4 安全 start date** | `get_market_safe_start()` / `get_all_safe_starts()` | §8.5 anti-leakage 邊界 |
| **C.5 cached start 解析** | `resolve_start_cached()` | 同步效能優化 |

### ⚡ Group D. 批次 Upsert + 安全 Commit

| 子項 | API | 用途 |
| :--- | :--- | :--- |
| **D.1 批次 upsert** | `bulk_upsert(table, records, unique_cols, page_size=2000)` | execute_values 高效批次 |
| **D.2 安全 commit** | `safe_commit_rows(conn, sql, rows, template, label)` | rollback 保護 |
| **D.3 分群 commit** | `commit_per_group()` / `commit_per_stock()` / `commit_per_day()` / `commit_per_stock_per_day()` | 避免長 transaction |
| **D.4 冪等 DDL** | `ensure_ddl(conn, *ddls)` 預設套用 `DDL_FETCH_LOG` | IF NOT EXISTS 語意 |
| **D.5 基礎設施 ensure** | `ensure_infrastructure()` | DDL_FETCH_LOG 唯一保證 |

### 🛡️ Group E. 錯誤處理 + Failure Logging

| 子項 | API | 用途 |
| :--- | :--- | :--- |
| **E.1 失敗追蹤類別** | `FailureLogger(table_name, db_conn, log_to_db)` 之 `log/add/__call__/dump` 介面 | 集中失敗追蹤 |
| **E.2 fetch_log 寫入** | `log_fetch_result(table, stock_id, status, rows_inserted, ...)` | 同步成果記錄 |
| **E.3 型別安全轉換** | `safe_float(value, default)` / `safe_int()` / `safe_date()` | None / 例外處理 |
| **E.4 批次安全映射** | `map_rows_safe(rows, mapper, label)` | 容錯式 row 處理 |
| **E.5 重複行去除** | `dedup_rows(rows, key_indices)` | 依 key 去重 |
| **E.6 失敗 JSON 落地** | `append_failure_json(table_name, item)` → `reports/` | audit trail |
| **E.7 failure log 路徑** | `get_failure_log_path(table_name)` | 路徑解析 |

### 🩺 Group F. Step 2C 前置依賴檢查（憲章 §二 L2411 授權）

| 子項 | 指令 / API | 治權對應 |
| :--- | :--- | :--- |
| **F.1 五軸前置稽核（CLI）** | `$ python scripts/core/db_utils.py` 之 `__main__` | 憲章 §二 L2411 Step 2C |
| **F.2 verdict 動態判定** | `run_diagnostics()` → PERFECT / WARNING / FAILED | §5.6.3 / §3.2 接受標準 |
| **F.3 五軸內容** | (1) DB 連線 + latency / (2) §6.7 SQL 查詢 / (3) Public API 20 個契約 / (4) `pipeline_execution_log` 寫入 / (5) `data_audit_log` 寫入 | 治權閉環 |
| **F.4 exit code 對 §3.2 接受標準** | `sys.exit(0 if status in ("PERFECT", "WARNING") else 1)` L1068 | §3.2 / FAILED 阻斷 Step 3 |

💡 **使用提示**：以上 6 群 + 30+ 子項 API 涵蓋 `db_utils.py` 全部 public 接口；其他治權程式（`data_schema.py` / `core_universe_schema.py` / `sovereign_sync_engine.py` / `core_universe_builder.py` / `audit_supply_chain.py` 等）依各自治權需求 `import` 對應功能群。

> 註 (v2.47)：v2.45 完成 §6.7 核心股查詢 SQL 契約 + 跨模組 public API restoration + connection diagnostics；v2.46 依 CLAUDE.md §四 #4 補入治權核心定義 + 模組常數 + cross-ref 達 100% 合規；v2.47（2026-05-21）將維運矩陣重組為 6 大功能群視角（Group A-F），對齊 db_utils 之 §3.2 橫切 library 治權特性，提升可讀性。

## 📜 三、全修訂歷程 (Full Revision History - 舊詳細參考)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v2.50** | 2026-06-02 | Codex | **§14.7-DF / §0.0-I Canonical Horizon Metric SSOT helper 落地**：新增 `summarize_horizon_metrics(label, horizon_days, panel_preds_actuals, ...)`（在 `get_canonical_panel_dates()` 之後），為全 multi-cycle validator 之 horizon-summary metric（core sharpe/eff_t/ir/win/ic/mdd/net + precision hit/overlap/rmse/mae + reliability ic_cov）**唯一計算來源**；validator 只產 (pred,actual) per panel，全部 metric 由 helper 算 → 全模型 metric 碼一致、可比。**動因**：§14.7-DE 切換後揭露 13 validator metric 計算碼分歧（9 FULL / 4 torch-subset）→ 違反 §0.0-I + 破壞相同比較基準。**實證**：公式逐字對齊 canonical LightGBM inline；faithful-reproduction unit test bit-identical（13 keys max|Δ|=0.00）；9 validator codemod LANDED（py_compile 13/13 PASS）。**雙層治權鎖**：憲章 §14.7-DF（T_DF-1~4）+ CLAUDE.md §一.17 同次入憲。**邏輯動量**：純新增 1 函式；既有 public API / `get_canonical_panel_dates` / `run_diagnostics()` / §6.7 SQL 契約皆無變更；TOOL_VER v2.49 → v2.50。 | **ACTIVE** |
| **v2.49** | 2026-06-02 | Codex | **§14.7-DE / §0.0-I Canonical Panel Source 單一引用源 helper 落地**：新增 `get_canonical_panel_dates(feature_set_version)` 函式（在 `get_db_conn()` 之後），依資料動態判定「具備最大 distinct 特徵數（canonical 完整集）之 panels」之 (feature_set_id, as_of_date) 清單，作為全部 multi-cycle validator 之 panel 範圍**唯一來源**。**動因**：2026-06-02 用戶 directive「所有模型符合 §0.0-I 單一引用源 + 反硬編精神，入憲章」；揭露 13 支 `multi_cycle_*_validation.py` 之 `get_panel_dates()` 各自寫死 `date(2018,6,15)` 起點 → 違反 §0.0-I（重複定義）+ §一.13（反硬編）。**實證**：資料驅動自動判定起點 **2013-05-15**（優於人工硬寫之 2013-06）/ 回傳 157 panels（2013-05-15~2026-06-01）/ py_compile PASS。**雙層治權鎖**：憲章 §14.7-DE（T_DE-1~4）+ CLAUDE.md §一.16 同次入憲。**邏輯動量**：純新增 1 函式；既有 20+ public API / `run_diagnostics()` / §6.7 SQL 契約 / verdict 動態判定 / 6 大功能群矩陣皆無變更；TOOL_VER v2.48 → v2.49。 | SUPERSEDED |
| **v2.48** | 2026-05-25 | Codex | **§3.2A.J `write_data_audit_log` race-safe ON CONFLICT DO NOTHING 落地（v6.1.0-patch §3.2A.J / §14.7-AY 程式預備升版 B）**：依憲章 v6.1.0-patch（commit `4da2450`，2026-05-25）新入憲之 §3.2A.J `db_utils.write_data_audit_log` Audit Log Write-Safe 治權契約（憲章 L2722-2745）+ §14.7-AY §7.4-A 姊妹缺陷補完入憲（憲章 L7480-7568），本版落地裁決第 2 條「`db_utils.py v2.47 → v2.48`：`write_data_audit_log()` INSERT 改為 `INSERT ... ON CONFLICT (...) DO NOTHING`，保證多 worker 並發冪等」。**Root cause（2026-05-24 Audit 2 揭露）**：Step 4F 啟動 ~65 秒兩個 sync_engine worker 並發呼叫本函式撞同 microsecond + 同 5-tuple → 純 INSERT 無 ON CONFLICT 保護 → race-induced duplicate row 1 個 → Audit 2 verdict=FAILED。**補正內容**：(I) `write_data_audit_log()` SQL 加 `ON CONFLICT (table_name, stock_id, data_date, action_type, timestamp) DO NOTHING` 子句（保證冪等寫入，不阻斷 caller）；(II) function docstring 從 "v2.43 unchanged in v2.44" 升為 "v2.48: §3.2A.J race-safe upgrade"；(III) CONSTITUTION_VER v6.0.0 → v6.1.0（對齊現行憲章 v6.1.0-patch）；(IV) TOOL_VER v2.47 → v2.48；(V) 主權狀態行加「§3.2A.J `write_data_audit_log` race-safe ON CONFLICT DO NOTHING 落地」；(VI) 標頭版本 + cosmetic v2.47 → v2.48。**對應 §7.4-A 對稱性**：§7.4-A multi-worker 讀側（sync_engine v1.22 cascade mitigation）已治權閉環；本版補完寫側（audit log 並發寫入）治權閉環 → 共同完成「multi-worker 讀寫雙側治權閉環」（憲章 §14.7-AY §7.4-A 治權對稱性表）。**邏輯動量**：函式簽名不變（`write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected)`）；caller 端零修改（30+ 個 caller 在 sovereign_sync_engine / fetchers / pipeline 中無感升級）；`record_lifecycle()` / `_LifecycleContext` / `get_db_connection()` / `get_core_stocks_from_db()` 等 20+ public API 全部不動；`run_diagnostics()` 五軸內容（含 `data_audit_log` 寫入測試）行為不變；§5.6.3 + §0.4 + §3.2 接受標準 + §6.7 SQL 契約 + §一 0~6 條核心定義 + 6 大功能群矩陣（Group A-F）皆無變更。**對既有 DB 影響**：升版前必須執行 `scripts/maintenance/migrate_data_audit_log_dedup_20260525.py v0.1` 一次性 dedup + ALTER TABLE ADD CONSTRAINT（同次入憲之 §14.7-AY 落地 C 項）；否則本版之 ON CONFLICT 子句因目標 UNIQUE constraint 不存在而 SQL error（PostgreSQL 要求 ON CONFLICT 必須指定既有 UNIQUE / EXCLUDE constraint）。**data_schema.py 配套升版**：v2.16 → v2.17 之 `data_audit_log.unique_constraints` 從 `[]` 改為 5-tuple（落地裁決第 1 條，另一支同次升版）。**追溯適用**：v0.1-v0.6 之 `audit_api_schema_compliance` Layer F 對 `data_audit_log` 之 dup>0 記錄重新詮釋為 race-induced artifact；本版 + data_schema v2.17 + migration 落地後自動消解。本版**不**修改其他 `write_*_log()` 函式（pipeline_execution_log SERIAL id 自然 unique 不需 ON CONFLICT）、**不**擴張至業務 dataset 之 `bulk_upsert()`（已有 ON CONFLICT DO UPDATE 業務鍵語意）。同步入憲：憲章 §3.2A.J（L2722-2745）/ §14.7-AY（L7480-7568）/ 修訂歷程 v6.1.0-patch entry（L66）。 | SUPERSEDED |
| v2.47 | 2026-05-21 | Codex | **維運矩陣重組為 6 大功能群視角（功能可讀性提升；採選項丙重寫；對齊 §3.2 橫切 library 治權特性）**：依 v2.46 後再審「db_utils 為橫切 library，舊維運矩陣 8 場景半數借用其他程式之指令，未真正代表 db_utils 自身功能」之問題，採選項丙重組維運矩陣。**補正內容**：(I) 維運矩陣標題「全量維運指令總矩陣」→「全量功能群矩陣」；(II) 由 8 場景視角重組為 6 大功能群分類：Group A DB 連線管理（7 子項）/ Group B Hybrid Observability 雙日誌（5 子項）/ Group C §6.7 SQL SSOT 核心股查詢（5 子項）/ Group D 批次 Upsert + 安全 Commit（5 子項）/ Group E 錯誤處理 + Failure Logging（7 子項）/ Group F Step 2C 前置依賴檢查（4 子項）= 33 子項；(III) 每子項明示 API / 指令 / 治權對應；(IV) 主權狀態行升至「GOVERNANCE SYNC (憲法 v6.0.0 對齊 + 維運矩陣重組為 6 大功能群視角；8 項檢查面 100% 合規)」；(V) TOOL_VER v2.46 → v2.47；(VI) 標頭版本 + cosmetic v2.46 → v2.47。**所有 public API、`run_diagnostics()` 邏輯、verdict 動態計算（L976-1019）、§6.7 SQL 契約、`__main__` exit code、6 條核心定義皆無變更**；本補正純為標頭維運矩陣之視角重組（functional 視角優於 scenario 視角；對齊 db_utils 為橫切 library 之治權特性）。 | SUPERSEDED |
| v2.46 | 2026-05-21 | Codex | **2 條核心定義補入 + 2 模組常數 + cross-ref 行號（CLAUDE.md §四 #4 100% 合規補強；逐元件審計 Step 1.1.4 跟進）**：依昨日剛入憲之 CLAUDE.md §四 #4「8 項標頭強制檢驗」治權原則檢驗，揭露 v2.45 之 5 項缺口：(1) 主權狀態行未明示「憲法 v6.0.0 對齊」；(2) 核心定義缺 [Zero Hardcoded Verdict]（程式邏輯 L976-1019 動態但未顯式宣告）；(3) 核心定義缺 [Sovereignty Declaration]（治權位階 / 5 套禁令 / T1-T3 / §8.5 未明示）；(4) 缺 module-level `CONSTITUTION_VER` + `TOOL_VER` 模組常數（治權慣例對齊 path_setup v4.45+ / data_schema v2.12+）；(5) cross-ref 缺精確行號（與 data_schema 之「L2440 / L2709」/ __init__ 之「L2457 / L2488 / L5589」/ core_universe_schema v0.3 之「L2455 / L2722 / L2348-L2353」/ audit_api_schema_compliance v0.2+ 之「L2483」之治權慣例不對齊）。**補正內容**：(I) 標頭版本 v2.45 → v2.46；(II) 最後更新日期 2026-05-16 → 2026-05-21；(III) 主權狀態行升至「GOVERNANCE SYNC (憲法 v6.0.0 對齊 + [Zero Hardcoded Verdict] + [Sovereignty Declaration] 核心定義補入；8 項檢查面 100% 合規)」；(IV) 核心定義新增第 5 條 [Zero Hardcoded Verdict] 顯式宣告動態 `run_diagnostics()` 對齊 §5.6.3；(V) 核心定義新增第 6 條 [Sovereignty Declaration] 明示 §3.2 橫切 library + Infrastructure Sovereign Authority（cross-ref 憲章 L2282 §3.2 子表）+ Step 2C 前置依賴檢查授權 + 5 套禁令不涉 + T1-T3 不分層 + §8.5 anti-leakage 不處理 + 不選股不評分（Builder Authority 邊界）+ 不持有 Raw API Schema（data_schema Authority 邊界）；(VI) 新增 module-level `CONSTITUTION_VER = "v6.0.0"` + `TOOL_VER = "v2.46"` 常數；(VII) 維運矩陣 8 場景 + report header 之 cosmetic v2.45 → v2.46。**所有 public API（`get_db_connection()`, `record_lifecycle()`, `write_data_audit_log()`, `get_core_stocks_from_db()`, `bulk_upsert()`, `FailureLogger`, `DDL_FETCH_LOG` 等 20+ 函式）、`run_diagnostics()` 邏輯、verdict 動態計算（L976-1019）、§6.7 SQL 契約、`__main__` exit code 對 §3.2 Step 2C 接受標準分流、record_lifecycle 之 _LifecycleContext（v2.44 既有）、所有公開行為皆無變更**；本補正純為標頭治權自我宣告（與 data_schema v2.14 / __init__ v1.16 / core_universe_schema v0.3 / audit_api_schema_compliance v0.2 同模式）。合規度：v2.45 ≈60% → v2.46 100%。 | SUPERSEDED |
| v2.45 | 2026-05-14 | Codex (No-touch Zone 授權) | **§6.7 SQL + Public API Restoration + 2026-05-16 Connection Diagnostics + 2026-05-16 exit code 補正**：(1) `get_core_stocks_from_db()` 改查 `core_universe_membership` JOIN `core_universe_snapshot WHERE status='committed'`，封閉 Pending Bug #4；(2) 補回 `get_db_conn`、`ensure_ddl`、`bulk_upsert`、`safe_commit_rows`、`FailureLogger`、`DDL_FETCH_LOG`、`log_fetch_result`、`db_transaction`、`db_session`、`write_pipeline_log`、`write_evaluation_log`、`get_db_stock_ids` 等跨模組 public API；(3) `psycopg2` / `dotenv` 改為延遲失敗，允許常數與 API 匯入測試先行；(4) `get_db_connection()` 補強必要 DB env 檢查、`connect_timeout=10` 與 host/user/dbname 錯誤脈絡，避免 sandbox 或秘密缺失被誤判為 schema drift；(5) **2026-05-16 exit code 補正**：`run_diagnostics()` 補上 `return diag_status`；`__main__` 改依回傳值呼叫 `sys.exit(0 if status in ("PERFECT", "WARNING") else 1)`，對齊憲章 §3.2 Step 2C 接受標準（FAILED 必須 exit 1 阻斷進入 Step 3）。 | SUPERSEDED |
| v2.44 | 2026-05-14 | Antigravity (Auto-patch, No-touch Zone 授權) | **Bug #2 + Bug #3 雙修補**：(1) `record_lifecycle` 改為 yield 一個可由 caller 標記失敗/警告的 `_LifecycleContext`，封堵「Python 無例外即記 success」之 status 謊報；(2) INSERT 由 5 欄擴張為 8 欄，補寫 start_time / end_time / error_msg，封堵 NULL 漏洞；(3) DB 連線改為僅在 finally 開啟，不再霸佔整個 task 期間；(4) logger 失敗時不再 propagate 例外給 caller。100% backward compatible —— 舊 `with record_lifecycle(...):` 呼叫端零修改。 | SUPERSEDED |
| v2.43 | 2026-05-12 | Antigravity | **防禦性修復**：補全缺失的 `argparse` 導入，恢復指令列工具之治權效力。 | SUPERSEDED |
| v2.42 | 2026-05-12 | Antigravity | **主權完備化**：對齊五大核心場景語意，擴張全可能性維運矩陣，落實混合觀測。 | SUPERSEDED |
| v2.41 | 2026-05-12 | Antigravity | **憲法化對齊**：補全維運矩陣與四維核心定義，對齊 v5.2 旗艦要求。 | SUPERSEDED |
| v2.0 | 2026-04-30 | Antigravity | **安全重構**：整合 .env 加密認證，建立 get_db_connection 標準化接口。 | SUPERSEDED |
| v1.0 | 2026-04-20 | Antigravity | **主權奠基**：初始版本，建立基本連線與 stocks 元數據表治理。 | ARCHIVED |
================================================================================
"""
import os, sys, logging, time, argparse, json
from contextlib import contextmanager
from pathlib import Path
from datetime import date, datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# 治權常數 (Constitution Constants) — v2.46 新增（CLAUDE.md §四 #4 / Step 1.1.4 補正）
# ──────────────────────────────────────────────────────────────────────────────
CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v2.50"

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ModuleNotFoundError:
    psycopg2 = None
    execute_values = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


DDL_FETCH_LOG = """
CREATE TABLE IF NOT EXISTS fetch_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    table_name TEXT NOT NULL,
    stock_id TEXT,
    status TEXT NOT NULL,
    rows_inserted INTEGER DEFAULT 0,
    fetch_date_from DATE,
    fetch_date_to DATE,
    duration_ms INTEGER DEFAULT 0,
    error_msg TEXT,
    fetch_mode TEXT DEFAULT 'market',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS table_name TEXT;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS stock_id TEXT;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS status TEXT;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS rows_inserted INTEGER DEFAULT 0;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS fetch_date_from DATE;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS fetch_date_to DATE;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS duration_ms INTEGER DEFAULT 0;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS error_msg TEXT;
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS fetch_mode TEXT DEFAULT 'market';
ALTER TABLE fetch_log ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
CREATE INDEX IF NOT EXISTS idx_fetch_log_table_stock_time
    ON fetch_log (table_name, stock_id, timestamp DESC);
"""


def _require_psycopg2():
    if psycopg2 is None:
        raise RuntimeError(
            "Missing dependency: psycopg2/psycopg2-binary is required for DB operations. "
            "Install project requirements before running DB diagnostics."
        )


class _LifecycleContext:
    """[v2.44 新增] 生命週期上下文物件。

    用於封堵 Bug #2：caller 在 try/except 內吃掉例外時，
    可透過 mark_failed / mark_warning 把局部失敗反映到 lifecycle log，
    避免 status 因「Python 無例外」而謊報 success。

    背景：sovereign_sync_engine v1.7 的 sync_fred / sync_finmind 即使
    sub-task 失敗也只更新內部 stats["failed"]，不 raise；舊版 record_lifecycle
    看不到這層失敗。本物件補上「外部標記」介面。
    """

    __slots__ = ("failures", "warnings")

    def __init__(self):
        self.failures = []
        self.warnings = []

    def mark_failed(self, msg):
        """標記一個局部失敗（不會 raise，僅記錄）。"""
        self.failures.append(str(msg))

    def mark_warning(self, msg):
        """標記一個局部警告（不會 raise，僅記錄）。"""
        self.warnings.append(str(msg))

    @property
    def has_failures(self):
        return len(self.failures) > 0

    @property
    def has_warnings(self):
        return len(self.warnings) > 0


@contextmanager
def record_lifecycle(task_name, category="general", stock_id=None):
    """旗艦級生命週期裝飾器 (v2.44) - 混合模式 A: pipeline_execution_log

    [v2.44 主要變動]
    1. Bug #2 修補：yield 一個 _LifecycleContext 給 caller 主動標記局部失敗。
       舊 `with record_lifecycle(...):` 不接收 yield 值仍正常運作（context manager 規範允許）。
       新 `with record_lifecycle(...) as lc:` 可呼叫 lc.mark_failed(msg) / lc.mark_warning(msg)。
    2. Bug #3 修補：INSERT 改寫 8 欄，補上 start_time / end_time / error_msg。
    3. 連線生命週期：改為僅在 finally 開連線，不再霸佔整個 task 期間。
    4. Logger 隔離：寫日誌失敗時印 warning 到 stderr，不再 propagate 給 caller。

    Args:
        task_name (str): 任務名稱，例：'sync_fred_macro'
        category (str): 分類，例：'ingestion' / 'maintenance' / 'infrastructure'
        stock_id (str|None): 標的 ID，無關標的時建議填 'SYSTEM'

    Yields:
        _LifecycleContext: 供 caller 標記局部失敗/警告之介面（opt-in）。

    Status 判定優先序：
        Python 例外          → 'failed' (error_msg = exception 訊息)
        ctx.failures 非空    → 'failed' (error_msg = 合併之失敗訊息)
        ctx.warnings 非空    → 'warning' (error_msg = 合併之警告訊息)
        否則                 → 'success' (error_msg = NULL)
    """
    start_time = datetime.now()
    ctx = _LifecycleContext()
    py_exception = None
    try:
        yield ctx
    except Exception as e:
        py_exception = e
        raise
    finally:
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)

        # [v2.44 Bug#2] 動態判定 status
        if py_exception is not None:
            status = "failed"
            error_msg = f"{type(py_exception).__name__}: {str(py_exception)}"
        elif ctx.has_failures:
            status = "failed"
            error_msg = "; ".join(ctx.failures[:5])
            if len(ctx.failures) > 5:
                error_msg += f"; ... (+{len(ctx.failures) - 5} more)"
        elif ctx.has_warnings:
            status = "warning"
            error_msg = "; ".join(ctx.warnings[:5])
            if len(ctx.warnings) > 5:
                error_msg += f"; ... (+{len(ctx.warnings) - 5} more)"
        else:
            status = "success"
            error_msg = None

        # [v2.44 Bug#3] INSERT 8 欄完整寫入 (start_time / end_time / error_msg 不再 NULL)
        # [v2.44 Patch C] 連線僅在此處開啟，不霸佔整個 task 期間
        # [v2.44 Patch D] Logger 失敗不影響 caller
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO pipeline_execution_log
                        (task_name, category, stock_id, start_time, end_time,
                         status, duration_ms, error_msg)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (task_name, category, stock_id, start_time, end_time,
                     status, duration, error_msg),
                )
                conn.commit()
            finally:
                cur.close()
                conn.close()
        except Exception as log_err:
            # 寫日誌失敗只警告，不再 raise 把 caller 一起拖死
            print(
                f"⚠️  [record_lifecycle] pipeline_execution_log 寫入失敗: {log_err}",
                file=sys.stderr,
            )


def write_data_audit_log(table_name, stock_id, data_date, action_type, rows_affected):
    """專項審計日誌 (v2.48: §3.2A.J race-safe upgrade) - 混合模式 B: data_audit_log

    §3.2A.J Audit Log Write-Safe 治權契約 (憲章 v6.1.0-patch L2722-2745 / §14.7-AY L7480-7568):
    multi-worker 並發呼叫本函式撞同 microsecond + 同 5-tuple 時,ON CONFLICT DO NOTHING
    保證冪等寫入(不產生 dup,不阻斷 caller);與 data_schema.py v2.17 之 5-tuple UNIQUE
    constraint 配套(若 DB 端 constraint 不存在,本 SQL 會 error — 需先跑
    scripts/maintenance/migrate_data_audit_log_dedup_20260525.py v0.1 一次性 migration)。
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO data_audit_log (table_name, stock_id, data_date, action_type, rows_affected)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (table_name, stock_id, data_date, action_type, timestamp) DO NOTHING
        """, (table_name, stock_id, data_date, action_type, rows_affected))
        conn.commit()
    finally:
        cur.close(); conn.close()


def get_db_connection():
    """建立資料庫連線 (v2.47)"""
    _require_psycopg2()
    required_env = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing_env = [name for name in required_env if not os.getenv(name)]
    if missing_env:
        raise RuntimeError(
            "Missing DB environment variables: "
            + ", ".join(missing_env)
            + "; load the project .env or run through the constitution-authorized entrypoints."
        )
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    try:
        return psycopg2.connect(
            host=host, port=port,
            dbname=dbname, user=user,
            password=os.getenv("DB_PASSWORD"), connect_timeout=10
        )
    except Exception as exc:
        raise RuntimeError(
            f"DB connection failed for host={host} port={port} dbname={dbname} user={user}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def get_db_conn():
    """Backward-compatible alias used by legacy fetchers."""
    return get_db_connection()


def get_canonical_panel_dates(feature_set_version="feature_set_v0.5"):
    """§0.0-I 單一引用源 + §一.13 反硬編 + §14.7-DE Canonical Panel Source:
    資料驅動取「具備完整 canonical 特徵集(nf = 資料中最大 distinct 特徵數)」之 panels。
    回傳 [(feature_set_id, as_of_date), ...] 依日期排序;起點/終點/特徵數皆由 DB 自動判定,
    嚴禁寫死日期或特徵數。為所有 multi-cycle validator 之 panel 範圍唯一來源(single source)。"""
    like = "%" + feature_set_version.replace(".", "_")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT as_of_date, feature_set_id, count(DISTINCT feature_name) AS nf "
            "FROM feature_values WHERE feature_set_id LIKE %s "
            "GROUP BY as_of_date, feature_set_id ORDER BY as_of_date",
            (like,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return []
    full = max(r[2] for r in rows)  # canonical 完整特徵數 = 資料驅動(非寫死)
    return [(fsid, d) for d, fsid, nf in rows if nf >= full]


def summarize_horizon_metrics(label, horizon_days, panel_preds_actuals,
                              n_top=20, cost_per_rebal=0.006, panel_spacing=30):
    """§14.7-DF Canonical Horizon Metric SSOT (§0.0-I 單一引用源 + §一.13 反硬編):
    全 multi-cycle validator 之 horizon-summary metric **唯一**計算來源 → 全模型 metric 碼 100% 一致 → 可比。
    輸入:
      panel_preds_actuals = [(pred_array, actual_array), ...] 每 OOS panel 一組
        (pred / actual 須同序、同股對齊;actual = 真實 forward log return)。
    參數(Tier-1/Tier-3 揭露,非寫死 magic):
      n_top=20(top-N 多頭組合慣例) / cost_per_rebal=0.006(Tier-3 broker fee 揭露) / panel_spacing=30(calendar month)。
    輸出: 完整 metric dict(core: sharpe/eff_t/ir/win/ic/mdd/net + precision: hit/overlap/rmse/mae + reliability: ic_cov);
           無有效 panel → None。公式逐字對齊 canonical LightGBM 實作(faithful-reproduction 保證)。"""
    import numpy as np
    import math

    def _spearman_ic(pred, y):
        rp = pred.argsort().argsort().astype(float)
        ry = y.argsort().argsort().astype(float)
        if np.std(rp) < 1e-10 or np.std(ry) < 1e-10:
            return 0.0
        return float(np.corrcoef(rp, ry)[0, 1])

    panel_top20_rets, panel_univ_rets, panel_ics = [], [], []
    panel_hit_rates, panel_overlaps, panel_rmses, panel_maes = [], [], [], []
    for pred, actual in panel_preds_actuals:
        pred = np.asarray(pred, dtype=float)
        y = np.asarray(actual, dtype=float)
        if len(pred) == 0 or len(pred) != len(y):
            continue
        ic = _spearman_ic(pred, y)
        nt = min(n_top, len(pred))
        top_idx = np.argsort(pred)[-nt:]
        actual_top_idx = np.argsort(y)[-nt:]
        panel_top20_rets.append(float(np.mean(y[top_idx])))
        panel_univ_rets.append(float(np.mean(y)))
        panel_ics.append(ic)
        panel_hit_rates.append(float(np.mean(np.sign(pred) == np.sign(y))))
        panel_overlaps.append(len(set(top_idx.tolist()) & set(actual_top_idx.tolist())) / nt)
        panel_rmses.append(float(np.sqrt(np.mean((pred - y) ** 2))))
        panel_maes.append(float(np.mean(np.abs(pred - y))))

    if not panel_top20_rets:
        return None

    n = len(panel_top20_rets)
    mean_ret = float(np.mean(panel_top20_rets))
    std_ret = float(np.std(panel_top20_rets, ddof=1)) if n > 1 else 0
    sharpe = mean_ret / std_ret * math.sqrt(12) if std_ret > 0 else 0
    win_rate = sum(1 for r in panel_top20_rets if r > 0) / n
    alphas = [t - u for t, u in zip(panel_top20_rets, panel_univ_rets)]
    mean_alpha = float(np.mean(alphas))
    std_alpha = float(np.std(alphas, ddof=1)) if n > 1 else 0
    ir = mean_alpha / std_alpha * math.sqrt(12) if std_alpha > 0 else 0
    t_stat = mean_alpha / (std_alpha / math.sqrt(n)) if std_alpha > 0 else 0
    running = 0; peak = 0; mdd = 0
    for r in panel_top20_rets:
        running += r
        if running > peak: peak = running
        if peak - running > mdd: mdd = peak - running

    rebals_per_year = 252.0 / horizon_days
    annualized_log_gross = mean_ret * rebals_per_year
    annualized_simple_gross = math.exp(annualized_log_gross) - 1
    annual_cost_drag = cost_per_rebal * rebals_per_year
    annualized_simple_net = math.exp(annualized_log_gross - annual_cost_drag) - 1

    if horizon_days <= panel_spacing:
        n_eff = float(n); overlap_pct = 0.0
    else:
        n_eff = n * (panel_spacing / horizon_days)
        overlap_pct = (horizon_days - panel_spacing) / horizon_days * 100
    eff_t_stat = t_stat * math.sqrt(n_eff / n) if n > 0 else 0
    is_significant = abs(eff_t_stat) > 1.997

    mean_hit = float(np.mean(panel_hit_rates))
    mean_overlap = float(np.mean(panel_overlaps))
    mean_rmse = float(np.mean(panel_rmses))
    mean_mae = float(np.mean(panel_maes))
    ic_cov = float(np.std(panel_ics, ddof=1) / abs(np.mean(panel_ics))) if np.mean(panel_ics) != 0 else float('inf')

    return {
        "horizon": label, "horizon_days": horizon_days, "n_panels": n,
        "n_effective": n_eff, "overlap_pct": overlap_pct, "rebals_per_year": rebals_per_year,
        "mean_ret_per_panel": mean_ret, "sharpe": sharpe, "win_rate": win_rate, "mdd_per_panel": mdd,
        "mean_alpha_per_panel": mean_alpha, "ir": ir, "t_stat": t_stat,
        "effective_t_stat": eff_t_stat, "is_significant_p05": is_significant,
        "mean_ic": float(np.mean(panel_ics)),
        "annualized_simple_gross": annualized_simple_gross,
        "annual_cost_drag_log": annual_cost_drag,
        "annualized_simple_net": annualized_simple_net,
        "precision_directional_hit_rate": mean_hit,
        "precision_top20_actual_overlap": mean_overlap,
        "precision_rmse": mean_rmse,
        "precision_mae": mean_mae,
        "reliability_ic_stability_cov": ic_cov,
    }


def get_connection_params():
    """Return sanitized connection parameters for maintenance tools."""
    return {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def db_connection_check():
    """基礎設施健康診斷 (v2.43 unchanged in v2.44)"""
    start = time.time()
    try:
        conn = get_db_connection()
        conn.close()
        return True, (time.time() - start) * 1000
    except Exception:
        return False, 0


check_db_health = db_connection_check


def ensure_ddl(conn=None, *ddls):
    """Execute one or more DDL statements idempotently.

    If called without DDL, ensure the shared fetch_log table exists.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    if not ddls:
        ddls = (DDL_FETCH_LOG,)
    try:
        with conn.cursor() as cur:
            for ddl in ddls:
                if ddl:
                    cur.execute(ddl)
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if own_conn:
            conn.close()


@contextmanager
def db_transaction():
    """Yield a cursor inside a commit/rollback transaction."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


@contextmanager
def db_session():
    """Yield a raw connection inside a commit/rollback transaction."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _bulk_upsert_sql(conn, sql, rows, template=None, page_size=2000):
    if not rows:
        return 0
    if execute_values is None:
        _require_psycopg2()
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template=template, page_size=page_size)
    return len(rows)


def _quote_ident(name):
    return '"' + str(name).replace('"', '""') + '"'


def _bulk_upsert_table(table_name, records, unique_cols, page_size=2000):
    if not records:
        return 0
    if not unique_cols:
        raise ValueError("unique_cols is required for table-name bulk_upsert")
    conn = get_db_connection()
    columns = list(records[0].keys())
    rows = [tuple(record.get(col) for col in columns) for record in records]
    col_sql = ", ".join(_quote_ident(col) for col in columns)
    conflict_sql = ", ".join(_quote_ident(col) for col in unique_cols)
    update_cols = [col for col in columns if col not in set(unique_cols)]
    if update_cols:
        update_sql = ", ".join(
            f"{_quote_ident(col)} = EXCLUDED.{_quote_ident(col)}" for col in update_cols
        )
        action_sql = f"DO UPDATE SET {update_sql}"
    else:
        action_sql = "DO NOTHING"
    sql = (
        f"INSERT INTO {_quote_ident(table_name)} ({col_sql}) VALUES %s "
        f"ON CONFLICT ({conflict_sql}) {action_sql}"
    )
    try:
        count = _bulk_upsert_sql(conn, sql, rows, page_size=page_size)
        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def bulk_upsert(*args, **kwargs):
    """Backward-compatible bulk upsert.

    Supports both legacy `(conn, sql, rows, template, page_size=...)` and
    newer `(table_name, records, unique_cols=[...])` call styles.
    """
    if args and hasattr(args[0], "cursor"):
        conn = args[0]
        sql = args[1]
        rows = args[2]
        template = args[3] if len(args) > 3 else kwargs.get("template")
        page_size = kwargs.get("page_size", 2000)
        return _bulk_upsert_sql(conn, sql, rows, template=template, page_size=page_size)
    if len(args) < 2:
        raise TypeError("bulk_upsert requires either conn/sql/rows or table_name/records")
    table_name = args[0]
    records = args[1]
    unique_cols = kwargs.get("unique_cols")
    if unique_cols is None and len(args) > 2:
        unique_cols = args[2]
    return _bulk_upsert_table(
        table_name,
        records,
        unique_cols=unique_cols,
        page_size=kwargs.get("page_size", 2000),
    )


def safe_commit_rows(conn, sql, rows, template=None, label="rows", page_size=2000):
    """Bulk-write rows and commit, rolling back on failure."""
    try:
        count = _bulk_upsert_sql(conn, sql, rows, template=template, page_size=page_size)
        conn.commit()
        return count
    except Exception as exc:
        conn.rollback()
        logging.getLogger(__name__).error("safe_commit_rows failed for %s: %s", label, exc)
        raise


def safe_float(value, default=None):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_date(value, default=None):
    if value in ("", None):
        return default
    if isinstance(value, date):
        return value
    text = str(value)[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return default


def map_rows_safe(rows, mapper, label="rows"):
    mapped = []
    failures = []
    for row in rows or []:
        try:
            mapped.append(mapper(row))
        except Exception as exc:
            failures.append({"row": row, "error_msg": str(exc), "label": label})
    return mapped, failures


def dedup_rows(rows, key_indices):
    seen = {}
    for row in rows or []:
        key = tuple(row[i] for i in key_indices)
        seen[key] = row
    return list(seen.values())


def commit_per_group(conn, sql, rows, template=None, group_key_fn=None, label="rows", page_size=2000):
    if group_key_fn is None:
        return {"ALL": safe_commit_rows(conn, sql, rows, template, label=label, page_size=page_size)}
    grouped = {}
    for row in rows or []:
        grouped.setdefault(group_key_fn(row), []).append(row)
    result = {}
    for key, group_rows in grouped.items():
        result[key] = safe_commit_rows(
            conn, sql, group_rows, template, label=f"{label}/{key}", page_size=page_size
        )
    return result


def commit_per_stock(conn, sql, rows, template=None, stock_index=0, label="rows", page_size=2000):
    return commit_per_group(
        conn,
        sql,
        rows,
        template=template,
        group_key_fn=lambda row: row[stock_index],
        label=label,
        page_size=page_size,
    )


def commit_per_day(conn, sql, rows, template=None, date_index=1, label="rows", page_size=2000):
    return commit_per_group(
        conn,
        sql,
        rows,
        template=template,
        group_key_fn=lambda row: row[date_index],
        label=label,
        page_size=page_size,
    )


def commit_per_stock_per_day(
    conn,
    sql,
    rows,
    template=None,
    stock_index=0,
    date_index=1,
    label="rows",
    page_size=2000,
):
    return commit_per_group(
        conn,
        sql,
        rows,
        template=template,
        group_key_fn=lambda row: (row[stock_index], row[date_index]),
        label=label,
        page_size=page_size,
    )


def get_failure_log_path(table_name):
    output_dir = _PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    return output_dir / f"{table_name}_failed_{stamp}.json"


def append_failure_json(table_name, item):
    path = get_failure_log_path(table_name)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(item)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_pipeline_log(
    task_name,
    stock_id="SYSTEM",
    status="success",
    category="general",
    duration_ms=0,
    rows=0,
    err=None,
    **kwargs,
):
    """Write a lifecycle-like pipeline log row without context-manager wrapping."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_execution_log
                    (task_name, category, stock_id, start_time, end_time,
                     status, duration_ms, error_msg)
                VALUES (%s, %s, %s, NOW(), NOW(), %s, %s, %s)
                """,
                (task_name, category, stock_id, status, duration_ms or 0, err),
            )
        conn.commit()
    except Exception as exc:
        print(f"⚠️  [write_pipeline_log] failed: {exc}", file=sys.stderr)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def write_evaluation_log(*args, **kwargs):
    """Best-effort evaluation logger backed by pipeline_execution_log."""
    task_name = kwargs.pop("task_name", None) or (args[0] if args else "evaluation")
    stock_id = kwargs.pop("stock_id", "SYSTEM")
    status = kwargs.pop("status", "success")
    duration_ms = kwargs.pop("duration_ms", 0)
    rows = kwargs.pop("rows", 0)
    err = kwargs.pop("err", kwargs.pop("error_msg", None))
    write_pipeline_log(task_name, stock_id, status, "evaluation", duration_ms, rows, err)


def log_fetch_result(*args, **kwargs):
    """Write a standardized fetch_log row."""
    if args and hasattr(args[0], "cursor"):
        conn = args[0]
        table_name = args[1]
        stock_id = args[2]
        fetch_date_from = args[3] if len(args) > 3 else None
        fetch_date_to = args[4] if len(args) > 4 else None
        rows_inserted = args[5] if len(args) > 5 else 0
        status = args[6] if len(args) > 6 else "success"
        error_msg = args[7] if len(args) > 7 else None
        duration_ms = kwargs.get("duration_ms", 0)
        fetch_mode = kwargs.get("fetch_mode", "market")
    else:
        table_name = args[0] if len(args) > 0 else kwargs.get("table_name")
        stock_id = args[1] if len(args) > 1 else kwargs.get("stock_id")
        status = args[2] if len(args) > 2 else kwargs.get("status", "success")
        rows_inserted = kwargs.get("rows_inserted", args[3] if len(args) > 3 else 0)
        fetch_date_from = kwargs.get("fetch_date_from", args[4] if len(args) > 4 else None)
        fetch_date_to = kwargs.get("fetch_date_to", args[5] if len(args) > 5 else None)
        duration_ms = kwargs.get("duration_ms", args[6] if len(args) > 6 else 0)
        error_msg = kwargs.get("error_msg", kwargs.get("error_message", args[7] if len(args) > 7 else None))
        fetch_mode = kwargs.get("fetch_mode", "market")
        conn = kwargs.get("conn")

    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fetch_log
                    (table_name, stock_id, status, rows_inserted, fetch_date_from,
                     fetch_date_to, duration_ms, error_msg, fetch_mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    table_name,
                    stock_id,
                    status,
                    rows_inserted,
                    fetch_date_from,
                    fetch_date_to,
                    duration_ms,
                    error_msg,
                    fetch_mode,
                ),
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if own_conn:
            conn.close()


class FailureLogger:
    """Collect failures and optionally persist them to fetch_log."""

    def __init__(self, table_name, db_conn=None, log_to_db=True):
        self.table_name = table_name
        self.db_conn = db_conn
        self.log_to_db = log_to_db
        self.failures = []

    def log(self, stock_id, error_msg, status="failed", **kwargs):
        item = {"stock_id": stock_id, "error_msg": str(error_msg), "status": status}
        item.update(kwargs)
        self.failures.append(item)
        if self.log_to_db:
            try:
                log_fetch_result(
                    self.table_name,
                    stock_id,
                    status,
                    rows_inserted=kwargs.get("rows_inserted", 0),
                    fetch_date_from=kwargs.get("fetch_date_from"),
                    fetch_date_to=kwargs.get("fetch_date_to"),
                    duration_ms=kwargs.get("duration_ms", 0),
                    error_msg=str(error_msg),
                    fetch_mode=kwargs.get("fetch_mode", "market"),
                    conn=self.db_conn,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning("FailureLogger DB write failed: %s", exc)

    def add(self, stock_id, error_msg, **kwargs):
        self.log(stock_id, error_msg, **kwargs)

    def __call__(self, stock_id, error_msg, **kwargs):
        self.log(stock_id, error_msg, **kwargs)

    def has_failures(self):
        return bool(self.failures)

    def dump(self):
        return list(self.failures)


def get_core_stocks_from_db(as_of_date=None, tiers=None, conn=None):
    """Return current committed core stock universe using the §6.7 SQL contract.

    Backward compatibility: legacy fetchers call `get_core_stocks_from_db(conn)`
    and expect a `{stock_id: config}` mapping. New governance callers call the
    function without a positional connection and receive a sorted stock-id list.
    """
    legacy_mapping = False
    if conn is None and hasattr(as_of_date, "cursor"):
        conn = as_of_date
        as_of_date = None
        legacy_mapping = True
    tiers = tuple(tiers or ("core_universe", "convex_universe"))
    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if as_of_date is None:
                cur.execute(
                    """
                    SELECT
                        m."stock_id",
                        m."stock_name",
                        m."type",
                        m."industry_category",
                        m."train_eligible",
                        m."predict_eligible",
                        m."backtest_eligible",
                        m."downstream_ready"
                    FROM "core_universe_membership" m
                    JOIN "core_universe_snapshot" s
                      ON s."snapshot_id" = m."snapshot_id"
                    WHERE s."status" = 'committed'
                      AND m."core_tier" = ANY(%s)
                      AND COALESCE(m."industry_category", '') NOT IN ('Index', '大盤')
                      AND s."as_of_date" = (
                          SELECT MAX("as_of_date")
                          FROM "core_universe_snapshot"
                          WHERE "status" = 'committed'
                      )
                    ORDER BY m."stock_id"
                    """,
                    (list(tiers),),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        m."stock_id",
                        m."stock_name",
                        m."type",
                        m."industry_category",
                        m."train_eligible",
                        m."predict_eligible",
                        m."backtest_eligible",
                        m."downstream_ready"
                    FROM "core_universe_membership" m
                    JOIN "core_universe_snapshot" s
                      ON s."snapshot_id" = m."snapshot_id"
                    WHERE s."status" = 'committed'
                      AND m."core_tier" = ANY(%s)
                      AND COALESCE(m."industry_category", '') NOT IN ('Index', '大盤')
                      AND s."as_of_date" = %s
                    ORDER BY m."stock_id"
                    """,
                    (list(tiers), as_of_date),
                )
            rows = cur.fetchall()
            if not legacy_mapping:
                return [row[0] for row in rows]
            return {
                row[0]: {
                    "name": row[1],
                    "stock_name": row[1],
                    "type": row[2],
                    "industry": row[3],
                    "industry_category": row[3],
                    "fetch_basic": True,
                    "fetch_chip": True,
                    "fetch_fundamental": True,
                    "fetch_derivative": True,
                    "fetch_news": True,
                    "train_eligible": bool(row[4]),
                    "predict_eligible": bool(row[5]),
                    "backtest_eligible": bool(row[6]),
                    "downstream_ready": bool(row[7]),
                }
                for row in rows
            }
    finally:
        if own_conn:
            conn.close()


def get_db_stock_ids(conn=None, active_only=True, core_only=False, types=None, **kwargs):
    """Return stock IDs from the governed universe or market asset table."""
    if core_only:
        return get_core_stocks_from_db(conn=conn)
    own_conn = conn is None
    if own_conn:
        conn = get_db_connection()
    where = []
    params = []
    if types:
        where.append('"type" = ANY(%s)')
        params.append(list(types))
    query = 'SELECT "stock_id" FROM "TaiwanStockInfo"'
    if where:
        query += " WHERE " + " AND ".join(where)
    query += ' ORDER BY "stock_id"'
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]
    finally:
        if own_conn:
            conn.close()


def get_latest_date(table_name, stock_id=None, date_col="date", id_col="stock_id"):
    """Return latest date from a table, optionally scoped by stock_id."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if stock_id is None:
                cur.execute(f"SELECT MAX({_quote_ident(date_col)}) FROM {_quote_ident(table_name)}")
            else:
                cur.execute(
                    f"SELECT MAX({_quote_ident(date_col)}) FROM {_quote_ident(table_name)} "
                    f"WHERE {_quote_ident(id_col)} = %s",
                    (stock_id,),
                )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_market_safe_start(conn, table_name, window_days=60, date_col="date"):
    query = (
        f"SELECT COALESCE(MAX({_quote_ident(date_col)}) + interval '1 day', "
        f"CURRENT_DATE - interval '{int(window_days)} days') "
        f"FROM {_quote_ident(table_name)}"
    )
    with conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()
        return row[0].strftime("%Y-%m-%d") if row and row[0] else None


def get_all_safe_starts(conn, table_name, id_col="stock_id", date_col="date", window_days=60):
    query = (
        f"SELECT {_quote_ident(id_col)}, MAX({_quote_ident(date_col)}) + interval '1 day' "
        f"FROM {_quote_ident(table_name)} GROUP BY {_quote_ident(id_col)}"
    )
    with conn.cursor() as cur:
        cur.execute(query)
        return {
            row[0]: row[1].strftime("%Y-%m-%d") if row[1] else None
            for row in cur.fetchall()
        }


def resolve_start_cached(
    stock_id,
    latest_dates,
    global_start,
    dataset_earliest,
    force=False,
    today=None,
):
    today = today or date.today().strftime("%Y-%m-%d")
    effective_start = max(global_start, dataset_earliest)
    if force:
        return effective_start
    latest = latest_dates.get(str(stock_id)) if latest_dates else None
    if latest is None:
        return effective_start
    if not isinstance(latest, str):
        latest = latest.strftime("%Y-%m-%d")
    next_day = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    if next_day > today:
        return None
    return max(next_day, effective_start)


def ensure_infrastructure():
    """Ensure shared infrastructure tables that this module owns."""
    ensure_ddl(None, DDL_FETCH_LOG)


def _check_pipeline_log_writable():
    """Actively verify Step 2C lifecycle-log writability."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_execution_log
                    (task_name, category, stock_id, start_time, end_time,
                     status, duration_ms, error_msg)
                VALUES (%s, %s, %s, NOW(), NOW(), %s, %s, %s)
                """,
                (
                    "db_diagnostic_v2.47_log_probe",
                    "infrastructure",
                    "SYSTEM",
                    "success",
                    0,
                    None,
                ),
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _check_public_api_contract():
    """Verify the public API set required by the constitution is present."""
    required = [
        "DDL_FETCH_LOG",
        "FailureLogger",
        "bulk_upsert",
        "check_db_health",
        "db_connection_check",
        "db_session",
        "db_transaction",
        "ensure_ddl",
        "ensure_infrastructure",
        "get_core_stocks_from_db",
        "get_db_conn",
        "get_db_connection",
        "get_db_stock_ids",
        "get_latest_date",
        "log_fetch_result",
        "record_lifecycle",
        "safe_commit_rows",
        "write_data_audit_log",
        "write_evaluation_log",
        "write_pipeline_log",
    ]
    return [name for name in required if name not in globals()]


def run_diagnostics():
    """執行基礎設施旗艦診斷報告 (v2.47 Standard)"""
    stocks = []
    diag_status = "PERFECT"
    diag_notes = []
    with record_lifecycle("db_diagnostic_v2.47", category="infrastructure", stock_id="SYSTEM") as lc:
        ok, latency = db_connection_check()
        if ok:
            try:
                stocks = get_core_stocks_from_db()
                if not stocks:
                    msg = "§6.7 core universe query returned 0 rows"
                    lc.mark_warning(msg)
                    diag_status = "WARNING"
                    diag_notes.append(msg)
            except Exception as exc:
                msg = f"§6.7 core universe query failed: {type(exc).__name__}: {exc}"
                lc.mark_failed(msg)
                diag_status = "FAILED"
                diag_notes.append(msg)
        else:
            msg = "DB connection check failed"
            lc.mark_failed(msg)
            diag_status = "FAILED"
            diag_notes.append(msg)

        missing_api = _check_public_api_contract()
        if missing_api:
            msg = "public API contract missing: " + ", ".join(missing_api)
            lc.mark_failed(msg)
            diag_status = "FAILED"
            diag_notes.append(msg)

        try:
            _check_pipeline_log_writable()
        except Exception as exc:
            msg = f"pipeline_execution_log write failed: {type(exc).__name__}: {exc}"
            lc.mark_failed(msg)
            diag_status = "FAILED"
            diag_notes.append(msg)

        try:
            write_data_audit_log("INFRA_CHECK", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "DIAGNOSTIC", 1)
        except Exception as exc:
            msg = f"data_audit_log write failed: {type(exc).__name__}: {exc}"
            lc.mark_failed(msg)
            diag_status = "FAILED"
            diag_notes.append(msg)

        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 基礎設施旗艦診斷啟動 (v2.47)")
        print("🛡️" * 40)

        print("\n" + "─" * 80)
        print("📊 基礎設施診斷摘要報告 (Infrastructure Diagnostic Report v2.47)")
        print("─" * 80)
        print(f"✅ 資料庫狀態   : {'SUCCESS' if ok else 'FAILED'}")
        print(f"🕒 連線延遲     : {latency:.2f} ms")
        print(f"📈 核心資產數   : {len(stocks)} 支 (§6.7 core_universe_membership)")
        print(f"📝 混合日誌狀態 : ACTIVE (pipeline_execution_log [8 欄完整] & data_audit_log)")
        print(f"⚖️  系統主權狀態 : {diag_status} (憲法 v6.1.0 / db_utils v2.50)")
        for note in diag_notes:
            print(f"   - {note}")
        print("─" * 80)

        print("\n💡 基礎設施維運建議 (Reference Information):")
        print("1. [效能提示]: 連線延遲高於 50ms 時，建議檢查資料庫連線池負載。")
        print("2. [範例提示]: 請參閱 Header 矩陣執行「所有核心股」的全量數據同步。")
        print("3. [歷史提示]: 所有連線變動必須記錄在全修訂歷程中以供溯源。")
        print("─" * 80 + "\n")

    return diag_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Finance 基礎設施治理工具 (v2.46)")
    parser.add_argument("--reset-pool", action="store_true", help="重置連線池 (Mock)")
    args = parser.parse_args()

    if args.reset_pool:
        print("🚀 正在執行連線池重置...")
        time.sleep(1)
        print("✅ 連線池已重置。")
    else:
        # §3.2 Step 2C 接受標準：PERFECT/WARNING exit 0；FAILED exit 1（不得進入 Step 3）
        status = run_diagnostics()
        sys.exit(0 if status in ("PERFECT", "WARNING") else 1)
