"""
audit_source_availability.py v0.7 (Strict Source Availability Audit · §6.8.8-C Time-Drift Tolerance · §6.8.8-D Full-Market Mode · §0.4 Progress Heartbeat · §6.8.8-E Quota-Error Detection · §14.7-AS Time-Drift Hotfix · §6.8.8-E.1 Transient Timeout Retry · §3.2A.I Parallel Workers)
================================================================================
**最後更新日期**: 2026-05-23
**主權狀態**: ACTIVE (憲法 v6.1.0 §14.7-L 對齊 + §6.8.8-C 時點漂移容忍規則落地 + §14.7-AP 治權閉環 + §6.8.8-D 全市場驗證模式 + §14.7-AQ 範圍對稱性補齊 + §0.4 [Hybrid Observability] 進度心跳 + §6.8.8-E Quota-Error 分類 + §14.7-AR 錯誤分類治權補齊 + §14.7-AS `_is_time_drift_ok` 字串減法 latent bug hotfix + **§6.8.8-E.1 transient timeout retry (`[30s, 300s]` 退避)** + **§3.2A.I parallel workers (`--workers ≤ 4` 平行 audit;預期 8h+ → ~2-3h)** + 對齊 §14.7-AU v6.1.0 升版)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:**來源可得性稽核**(§6.8.8):各 dataset 自 API 來源端最早可得日期。

**輸入 → 輸出**:API → 各 dataset 起始日

**為什麼需要它**:決定「全天數」的真實起點。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Strict Source Authority]: 驗證 core+convex 全集(N dynamic per §14.7-BW pure doctrine,無 hardcoded 150)資料是否符合憲章 §14.7-L 之嚴格定義
   — 每一個 FinMind `stock_id + dataset` 必須從 API 最早可得日期完整對齊 DB；
   `start_date=1990-01-01` 為來源端可得下界。
2. [Dual-Source Verification]: 對每個 `stock_id + dataset` 比對 API 與 DB 的
   `row_count / min(date) / max(date)`；API source-empty 時 DB 必須為 0 rows
   才視為 `SOURCE_EMPTY_OK`；任何差異即為 mismatch。
3. [FRED Valid Numeric Coverage]: `--include-fred` 啟用時，另對 DFF / UNRATE /
   T10Y2Y / VIXCLS 四序列以「可轉為數值的有效 observation」（排除 `.` 缺值列）
   為驗收口徑，row_count / min / max 必須與 DB 完全一致。
4. [Strict Mode Exit Contract]: `--strict` 模式下任何 mismatch 即 exit 1 並
   產出 `reports/source_availability_audit_*.md` 與 targeted backfill commands；
   對齊憲章 §3.2 接受標準（FAIL → exit 1）。
5. [Hybrid Observability]: 維運行為觸發 `record_lifecycle` 與 `write_data_audit_log`；
   主權狀態依實況動態計算（PERFECT / WARNING / FAILED），嚴禁硬編
   （對齊憲章 §5.6.3 [Zero Hardcoded Verdict]）。
   **v0.4 補強**：對長時運行（全市場 ~5 hr）新增 per-N-stock progress heartbeat
   （`--progress-interval N`；預設 100；N=0 為靜默模式相容 v0.3）；每條 heartbeat 含
   `idx/total | checked | source_empty_ok | time_drift_ok | mismatch | api_errors | elapsed | eta`，
   解決「audit 中段 25,000 probe 完全靜默」之觀察缺口。
6. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準。
7. **[Time-Drift Tolerance]** (v0.2, 憲法 §6.8.8-C / §14.7-AP)：對「**audit 觀察時點之自然時間漂移**」之容忍：
   - **`(api_date_max - db_date_max) ≤ N_calendar_days`** 且 **`abs(api_rows - db_rows) ≤ N`** → 標記為
     `TIME_DRIFT_OK`（**不**視為 mismatch；不計入 exit 1）
   - **預設 N = 3 個日曆日**（覆蓋週末 + 1 個工作日緩衝）
   - CLI: `--drift-tolerance N`（N=0 為嚴格模式；N>0 為容忍模式；預設 3）
   - 對齊憲章 §6.8.8-C 治權契約：「sync 時點 vs API publish 時點之競爭」+「audit 觀察時點之自然延遲」屬合法漂移
8. **[Sovereignty Declaration]** (v0.2, 憲章 §3.2A 橫切稽核工具)：本程式為 §3.2A 橫切基礎設施稽核工具
   （非 §3.1 序列模組）；落實 §14.7-L strict source availability + §6.8.8-C 時點漂移容忍；不涉及 §0.1-A / §0.2-A /
   §0.3-A / §0.0-E.4 / §0.0-F.3 五套禁令；不在 §0.1.1 T1/T2/T3 分層內；不處理 §8.5 anti-leakage；不調度 universe；
   不持有 schema 定義；DATASET_REGISTRY + FINMIND_API_TABLES 為唯一 schema 引用源。
9. **[Full-Market Audit Mode]** (v0.3, 憲法 §6.8.8-D / §14.7-AQ；2026-05-22 入憲)：對齊 `sovereign_sync_engine` 之
   §6.8.7 第 (4) 條全市場限定治理例外，落地 audit 側對等驗證範圍：
   - `--universe full` 解鎖 `core ∪ convex ∪ research ∪ quarantine` ≈ 2,798 支 × 10 datasets 之全市場 audit
   - **必須**附 `--special-full-market-reason "<≥12 字理由>"`（與 sovereign_sync_engine 同口徑）；缺 reason / reason < 12 字 → preflight exit 1
   - 五類合法情境：DB rebuild bootstrap / Sovereign rebuild / pre-annual audit / 資料源治權變更 / 重大合規事件
   - reason 寫入報表 metadata + 終端 summary 留 audit trail
   - `_resolve_stocks(universe="full")` 對齊 `sovereign_sync_engine.UNIVERSE_TIERS["full"]` SSOT（4-tier union；避免兩工具範圍漂移）
   - 報表標題去 "Core 150" 字樣（為「Strict source availability audit」）；scope 欄位升級為 `stocks=N (universe=core|full)`
   - 跨工具治權範本：未來 maintenance audit 工具支援 `--universe full` 一律比照（必須附 reason）
10. **[Quota-Error Detection]** (v0.5, 憲法 §6.8.8-E / §14.7-AR；2026-05-22 入憲)：區分 FinMind quota 暫時性錯誤
    vs 真實 SOURCE_EMPTY vs 正常資料三類響應：
    - `_fetch_api_summary` 內 msg 檢測：含 `"upper limit"` / `"rate limit"` 字串 → `raise RuntimeError`
    - 既有 `msg not in (None, "", "success")` 邏輯保留為 catch-all
    - quota error → `api_status="ERROR"` → `_classify` 回 `API_ERROR` → 計入 `self.api_errors`（不入 mismatch）
    - **計數邏輯解耦**：v0.5 修正既有 bug：v0.2-v0.4 將 API_ERROR 同時計入 api_errors + mismatch（雙重計）；v0.5 僅計 api_errors
    - 報表 mismatch table 新增 `error` 欄位，明示 API_ERROR row 之 raw exception / msg
    - 跨工具治權範本：所有 maintenance audit 工具區分 quota error vs SOURCE_EMPTY 一律比照

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [標準執行：core+convex 全集嚴格驗證(N dynamic per §14.7-BW)]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict` | audit_source_availability v0.6 |
| **2. [單一個股全表驗證]** | `$ python scripts/maintenance/audit_source_availability.py --id 2330 --all --strict` | audit_source_availability v0.6 |
| **3. [來源端 snapshot 寫出]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --strict --snapshot-out /tmp/api_start_dates_core150.json` | audit_source_availability v0.6 |
| **4. [離線重跑：用既有 snapshot]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --strict --snapshot-in /tmp/api_start_dates_core150.json` | audit_source_availability v0.6 |
| **5. [僅 FinMind（略 FRED）]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --strict` | audit_source_availability v0.6 |
| **6. [§6.8.8-C 時點漂移容忍：預設 3 個日曆日]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict --drift-tolerance 3` | audit_source_availability v0.6 |
| **7. [§6.8.8-C 嚴格模式：相容 v0.1]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict --drift-tolerance 0` | audit_source_availability v0.6 |
| **8. [§6.8.8-D 全市場驗證：~2,798 支]** | `$ python scripts/maintenance/audit_source_availability.py --universe full --all --include-fred --strict --special-full-market-reason "DB rebuild bootstrap 2026-05-22 full-market validation"` | audit_source_availability v0.6 |
| **9. [§6.8.8-D preflight 拒絕：缺 reason]** | `$ python scripts/maintenance/audit_source_availability.py --universe full --all` → **exit 1**（缺 reason / reason < 12 字 / reason 給但 universe 非 full）| audit_source_availability v0.3 |
| **10. [§0.4 進度心跳：每 100 stock]** | `$ python scripts/maintenance/audit_source_availability.py --universe full --all --include-fred --strict --special-full-market-reason "<reason>" --progress-interval 100` | audit_source_availability v0.6 |
| **11. [§0.4 靜默模式：相容 v0.3]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict --progress-interval 0` | audit_source_availability v0.6 |
| **12. [§6.8.8-E quota-error 分類]** | quota 耗盡時 audit 自動歸為 `api_errors`（不入 mismatch）；報表 mismatch table 之 API_ERROR row 顯示 `error` 欄位含 raw msg | audit_source_availability v0.6 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **non-strict** | 移除 `--strict` | 報告 mismatch 但 exit 0；用於診斷掃描 |
| **snapshot-only** | `--snapshot-out <path>` 不加 `--strict` | 僅產生來源端 snapshot，不做 DB 比對 |
| **fred-only** | `--include-fred --datasets fred` | 略過 FinMind，僅對 FRED 四序列驗證 |
| **id-list** | `--id 2330 --id 2454 ...` | 多個指定 stock_id 並列驗證 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.7** | 2026-05-23 | Codex | **§6.8.8-E.1 transient timeout retry + §3.2A.I parallel workers + §7.4-A cascade-aware（憲章 v6.1.0 §6.8.8-E.1 + §3.2A.I + §14.7-AU 入憲對應之三契約合併單程式升版）**：依憲章 v6.1.0 §6.8.8-E.1 + §3.2A.I + §14.7-AT + §14.7-AU（2026-05-23 入憲）合併落地三個 audit-side 治權契約。**Root cause 實證**：2026-05-23 from-zero rebuild 後置 audit 3 全市場 mode 8h15m / 24,939 probe / verdict=FAILED；FAIL 原因為 1 mismatch（已 backfill 修復）+ 1 api_error（stock 5490 純 30s 網路 timeout，DB 內容實則 100% 正確）。**功能變更 6 大塊**：(I) `__init__` 新增 `api_timeout` / `api_retry` / `api_retry_backoffs` / `workers` 參數 + `_stats_lock` thread-safe Lock + `transient_retries_used` 計數；(II) `SimpleThrottle.acquire()` 加 `threading.Lock` 對齊 sync_engine `FinMindThrottle` 設計;sleep 期間釋放 lock 讓其他 worker 同步 prune window；(III) `_fetch_api_summary` 對 `requests.exceptions.Timeout` / `ConnectionError` / `HTTPError(5xx)` 退避重試（預設 `[30s, 300s]` 兩階段；quota 訊息 raise 不重試;對齊 §6.8.8-E 既有契約）；error 訊息附「after N retries [...]」；(IV) `_fetch_fred_summary` 同步加 transient retry；(V) `run()` 重構為 serial / parallel 雙路徑：`workers <= 1` 走原 serial loop（v0.6 兼容）；`workers > 1` 走 `ThreadPoolExecutor`，每 worker 持自己的 DB connection（`conn_pool`），stats 計數透過 `_stats_lock` 批次更新降低 lock 競爭，progress 心跳基於 atomic `done` counter；(VI) progress 心跳新增 `retries=` 欄位顯示 transient 重試次數。**CLI 新增 4 flag**：`--api-timeout N`（預設 30）/ `--api-retry N`（預設 1；0 = 完全相容 v0.6）/ `--retry-backoff "30,300"`（csv；空字串 = 不退避立即 raise）/ `--workers N`（預設 1；最大 4）。**FRED 區段**：仍序列執行（4 series 數量少不需平行）;自管 conn 生命週期。**標頭變更**：(a) L2 副標補入「§6.8.8-E.1 Transient Timeout Retry + §3.2A.I Parallel Workers」；(b) L5 主權狀態行補入 §6.8.8-E.1 + §3.2A.I 落地 + 憲法 v6.0.0 → v6.1.0；(c) TOOL_VER v0.6 → v0.7 + CONSTITUTION_VER v6.0.0 → v6.1.0；(d) 最後更新日期 2026-05-22 → 2026-05-23；(e) v0.6 SUPERSEDED + 新 v0.7 ACTIVE entry。**Smoke test 通過**：(i) compile OK;(ii) --id 2330 --dataset TaiwanStockPrice serial PERFECT 1/0/0/0 兼容 v0.6;(iii) --universe core --dataset TaiwanStockPrice --workers 4 並行 150 stocks 101 秒 PERFECT 150/0/0/0/0 retries=0。**治權邊界嚴守**：所有既有 11 CLI flag / verdict 計算 / classify 邏輯 / time-drift / quota 分類 / record_lifecycle / write_data_audit_log 接線**全保留**；本契約**僅**新增 4 flag + 內部 retry + 並行能力;介面對外**向後相容 v0.6**（`--workers 1 --api-retry 0` 完全等同 v0.6 行為）；**不**改 schema / DDL / API contract / DATASET_REGISTRY 引用 / `--include-fred` / `--strict` / `--snapshot-in/out` / `--drift-tolerance` / `--special-full-market-reason` / `--progress-interval` 既有 flag 語意；**不**改 §6.8.7 第 (4) 條 / §6.8.8 / §6.8.8-A〜E 既有條款（§6.8.8-E.1 為延伸補強而非取代）；**不**改 §3.1 / §3.2 / §6.7 / §7 / §8 / §9 強制契約。**性能預估**：全市場 audit 8h15m → `--workers 4` 預期 ~2-3h（3x 加速；受 FinMind 5500/hr quota 限制）；transient timeout 自動 retry 後在 strict 模式不再因偶發網路抖動誤判 FAILED。**對應憲章 v6.0.0 → v6.1.0 升版**：本程式為 §14.7-AU 預備 7 程式之第五個（最後一個）落地；首例「三契約單程式合併升版」之治權合併（§6.8.8-E.1 + §3.2A.I + §7.4-A audit 端對應）。 | **ACTIVE** |
| v0.6 | 2026-05-22 | Codex | **§14.7-AS `_is_time_drift_ok` 字串減法 latent bug hotfix 落地**：依憲章 v6.0.0-patch §14.7-AS（commit `aa1c03f`；2026-05-22 入憲）落地 `_is_time_drift_ok()` 之 latent bug 修正。**Root Cause**：v0.2-v0.5 五版 `_is_time_drift_ok` 之 `(api_max - db_max).days` 對 str 相減直接拋 `TypeError` 被 except 捕獲返 False → 真實 1 day drift 被誤判 MISMATCH 而非 TIME_DRIFT_OK。v0.5 smoke test (`reports/source_availability_audit_20260522_1521.md`) 首次浮現：2330 TaiwanStockPrice api_max=2026-05-22 / db_max=2026-05-21（1 day drift 在 drift_tolerance=3 容忍範圍內）卻被誤判 MISMATCH。**功能變更 1 點**（純 hotfix）：`_is_time_drift_ok()` 內加入字串轉 date 邏輯 — `api_max_date = datetime.strptime(api_max, "%Y-%m-%d").date() if isinstance(api_max, str) else api_max`；db_max 同；再 subtract 計算 `date_drift_days`。**標頭變更**：(a) L2 副標補入「§14.7-AS Time-Drift Hotfix」；(b) L5 主權狀態行補入 §14.7-AS hotfix；(c) 矩陣場景 1-12 對齊 v0.6；(d) TOOL_VER v0.5 → v0.6。**治權邊界嚴守**：介面零變動（無新 flag；行為純內部 hotfix）；不修改 §6.8.8 / §6.8.8-A / §6.8.8-B / §6.8.8-C / §6.8.8-D / §6.8.8-E 既有條款（§6.8.8-C drift 容忍規則完整有效，本次屬實作對齊）；不改 §3.1 / §3.2 / §6.7 / §7 / §8 / §9 強制契約；不改 audit 工具邏輯（hotfix 非邏輯改寫）。**治權位階**：**P1 hotfix**（§0.0-E.6 升版優先級）；§6.8.8-C 落地實作 bug 修正，不新增治權條款。**驗證**：compile OK；smoke `--id 2330 --dataset TaiwanStockPrice` 預期 verdict=PERFECT + status=TIME_DRIFT_OK（v0.5 之 MISMATCH artifact 消解）。**對應 §0.0-G 第 17 次跑通**（§14.7-AS）落地 Phase 2 - hotfix 單修。 | SUPERSEDED |
| v0.5 | 2026-05-22 | Codex | **§6.8.8-E quota-error 判別 + §14.7-AR 錯誤分類治權補齊落地**：依憲章 v6.0.0-patch §6.8.8-E + §14.7-AR（commit `5fd42a1`；2026-05-22 入憲）落地 quota-throttled response 判別治權規則。**真實 Root Cause**（讀程式碼確認後）：v0.2-v0.4 `_fetch_api_summary` 已有 `payload.msg not in (None, "", "success") → raise` 邏輯，**但** `run()` 計數邏輯之 `elif row["status"] not in {"OK"}: self.mismatch += 1` 將 API_ERROR row 與 SOURCE_EMPTY_DB_HAS_ROWS 一併歸為 mismatch → **API_ERROR 雙重計數**（既在 except 區塊計入 `self.api_errors`、又在 run() 計數邏輯計入 `self.mismatch`）。**功能變更 3 點**：(I) **計數邏輯解耦 API_ERROR**：FinMind loop + FRED loop 各自新增 `elif row["status"] == "API_ERROR": pass` 分支，避免雙重計（v0.2-v0.4 bug 修正）；(II) `_fetch_api_summary` 強化 quota substring 檢測（含 `"upper limit"` / `"rate limit"` → `raise RuntimeError`，與既有 `msg not in (None,"","success")` 並列，明示 §6.8.8-E 治權契約）；(III) 報表 mismatch table 新增 `error` 欄位，明示 API_ERROR row 之 raw exception / msg（如 `RuntimeError: FinMind quota throttled: Requests reach the upper limit`），不再僅顯示 api_rows=null。**標頭變更**：(a) L2 副標補入「§6.8.8-E Quota-Error Detection」；(b) L5 主權狀態行補入 §6.8.8-E + §14.7-AR 錯誤分類治權補齊；(c) 核心定義 9 條 → 10 條：新增 [Quota-Error Detection] (v0.5, §6.8.8-E / §14.7-AR；6 條治權邊界含計數解耦明文)；(d) 維運矩陣補入場景 12（quota 自動歸 api_errors）；(e) 矩陣場景 1-9 對齊模組統一更新至 v0.5；(f) TOOL_VER v0.4 → v0.5。**治權邊界嚴守**：介面零變動（無新 flag；行為純內部修正）；既有 v0.4 之 quota raise 邏輯保留為 catch-all（無破壞性變更）；不修改 §6.8.7 第 (4) 條五類合法情境、不修改 §6.8.8 / §6.8.8-A / §6.8.8-B / §6.8.8-C / §6.8.8-D 既有條款、不改 §3.1 / §3.2 / §6.7 / §7 / §8 / §9 強制契約、不改 CoreScore v0.2 與 ThemeResonance 15%、不改 FinMind API §7 throttle 5500/hr 上限。**追溯適用**：既有 v0.2-v0.4 audit 報告中 mismatch row 若顯示 `api_rows=null / api_min=null / api_max=null` 而 `db_rows>0` 者，**重新詮釋為「API_ERROR row 之雙重計數 artifact」**；v0.5 落地後此類誤判將自動歸為 `api_errors`，verdict 不變但 mismatch 數字下降。**驗證**：compile OK；--help 不變；smoke `--id 2330 --dataset TaiwanStockPrice` verdict=PERFECT；計數解耦邏輯驗證（force quota → api_errors > 0 而 mismatch=0）。**對應 §0.0-G 第 16 次跑通**（§14.7-AR）落地 Phase 2 - 程式單修。 | SUPERSEDED |
| v0.4 | 2026-05-22 | Codex | **§0.4 [Hybrid Observability] 進度心跳落地 — 全市場 5h audit 中段觀察缺口補齊**：v0.3 首戰全市場 audit (PID 1128773) 啟動 13 分鐘後揭露**觀察缺口**：log 0 bytes / 無 per-stock print / 無中段 DB 寫入 → 對 ~25,000 probe / ~5h 之長時運行完全無法掌握進度。**功能變更 3 點**：(I) 新增模組級常數 `PROGRESS_INTERVAL_DEFAULT = 100`；(II) `__init__` 新增 `progress_interval` 參數 + argparse `--progress-interval N` flag（預設 100；N=0 為靜默模式相容 v0.3）；(III) `run()` 主迴圈每 N 個 stock 印一條 progress heartbeat：`stocks=idx/total (%) | checked | source_empty_ok | time_drift_ok | mismatch | api_errors | elapsed | eta`；run start 印一條 init line；最後一個 stock 強制印一條 final progress。**標頭變更**：(a) L2 副標補入「§0.4 Progress Heartbeat」；(b) L5 主權狀態行補入 §0.4 [Hybrid Observability] 進度心跳；(c) 核心定義 5. [Hybrid Observability] 擴增「v0.4 補強」說明（不另立新條，純功能擴增）；(d) 維運矩陣新增場景 10 (`--progress-interval 100`) + 場景 11 (`--progress-interval 0` 相容 v0.3)；(e) TOOL_VER v0.3 → v0.4。**治權位階**：**P3 觀察性升級**（§0.0-E.6 升版優先級），**不**改 verdict 邏輯 / **不**改 schema / **不**改 audit 範圍邏輯 / **不**需要 §0.0-G 入憲（純 UX）；對齊憲章 §0.4 [Hybrid Observability] + §5.6.3 [Zero Hardcoded Verdict] 既有原則。**對應 sovereign_sync_engine 之 `_detail()` print 慣例**（同類進度可見性設計範本）。**驗證已通過**：compile OK；--help 顯示 --progress-interval；smoke 單一個股仍 verdict=PERFECT；--progress-interval 0 完全靜默相容 v0.3。**§14.7-AQ Phase 3 全市場 audit 首戰實證**將用 v0.4 重啟（reason 不變）。 | SUPERSEDED |
| v0.3 | 2026-05-22 | Codex | **§6.8.8-D 全市場驗證模式 + §14.7-AQ 治權範圍對稱性補齊落地**：依憲章 v6.0.0-patch §6.8.8-D + §14.7-AQ（commit `52e4511`；2026-05-22 入憲）落地 audit 側對等於 `sovereign_sync_engine v1.21 --universe full` 之全市場 audit 能力，閉合 §14.7-AQ 識別之治權範圍對稱性缺口（~26,500 對全市場資料原無治權內驗證機制）。**功能變更 5 點**：(I) argparse `--universe choices=["core"]` → `["core", "full"]`；(II) argparse 新增 `--special-full-market-reason` flag + main() preflight 3 分支治權檢查（缺 reason / reason < 12 字 / reason 給但 universe 非 full → exit 1）；(III) `_resolve_stocks(universe="full")` 對齊 `sovereign_sync_engine.UNIVERSE_TIERS["full"]` SSOT，回傳 `core_universe ∪ convex_universe ∪ research_universe ∪ quarantine_universe` 4-tier union ≈ 2,798 支；(IV) `__init__` 新增 `special_full_market_reason` 參數；reason 寫入報表 metadata + 終端 summary 留 audit trail；(V) 新增模組級常數 `FULL_MARKET_REASON_MIN_CHARS = 12` + `FULL_MARKET_REQUIRED_UNIVERSE = "full"`（獨立定義避免 maintenance → ingestion 反向耦合 import；對映 sovereign_sync_engine 同名常數）。**標頭變更**：(a) L2 副標補入「§6.8.8-D Full-Market Mode」；(b) L5 主權狀態行補入 §6.8.8-D + §14.7-AQ；(c) 核心定義 8 條 → 9 條：新增 [Full-Market Audit Mode] (v0.3, §6.8.8-D / §14.7-AQ；7 條治權邊界)；(d) 維運矩陣補入場景 8（`--universe full` 含 reason）+ 場景 9（preflight 拒絕 invalid case）；(e) 報表標題去 "Core 150" → "Strict source availability audit"；(f) scope 欄位升級為 `stocks=N (universe=core|full)`；(g) constitution 欄位加 §6.8.8-D + §14.7-AQ；(h) `_print_summary` 加 universe + special_full_market_reason 顯示；(i) TOOL_VER v0.2 → v0.3。**治權邊界嚴守**：所有既有 `--universe core` 行為**完全不變**（v0.2 → v0.3 為新功能擴充非邏輯改寫）；不修改 §6.8.7 第 (4) 條五類合法情境、不修改 §6.8.8 / §6.8.8-A / §6.8.8-B / §6.8.8-C 既有條款、不改 §3.1 / §3.2 / §6.7 / §7 / §8 / §9 強制契約、不改 CoreScore v0.2 與 ThemeResonance 15%、不改 schema 定義（DATASET_REGISTRY + FINMIND_API_TABLES 仍為唯一引用源）。**驗證已通過**：compile OK；preflight 3 分支全部 exit 1 ✓；smoke `--id 2330 --dataset TaiwanStockPrice` verdict=PERFECT；`--help` 顯示 `choices=["core", "full"]` + `--special-full-market-reason`。**對應 §0.0-G 第 15 次跑通**（§14.7-AQ）落地 Phase 2 - 程式單修。 | SUPERSEDED |
| v0.2 | 2026-05-22 | Codex | **§6.8.8-C 時點漂移容忍規則落地 + §14.7-AP 治權閉環延伸實證**：依憲章 v6.0.0-patch §6.8.8-C + §14.7-AP（commit `4d990d0`；2026-05-22 入憲）落地實作 audit 觀察時點漂移之容忍規則。**補正內容**：(I) 新增 `--drift-tolerance N` argparse flag（預設 N=3 個日曆日；N=0 為嚴格模式相容 v0.1）；(II) `_classify()` 邏輯擴增 TIME_DRIFT_OK 分支：當 `(api_date_max - db_date_max).days ≤ N` 且 `abs(api_rows - db_rows) ≤ N` 時標記為 TIME_DRIFT_OK（**不**計入 mismatch / 不影響 exit code）；(III) `_classify_fred()` 同步擴增 TIME_DRIFT_OK 分支；(IV) stats 新增 `time_drift_ok` + `fred_time_drift_ok` 計數器；(V) 報告新增 TIME_DRIFT_OK 獨立分類段落；(VI) 核心定義新增 [Time-Drift Tolerance] + [Sovereignty Declaration] 兩條治權慣例；(VII) 維運矩陣補入 6/7 兩個 --drift-tolerance scenarios；(VIII) 新增模組級 CONSTITUTION_VER + TOOL_VER 常數。**§14.7-AP 治權閉環實證**：本 commit 與 charter commit `4d990d0` 之治權契約完全對齊；2026-05-22 11:13 已實證 PERFECT 0/0（mismatch 全部消解後本 v0.2 之容忍規則為日後增量 sync 時自然漂移之預防性容忍）。**介面零變動**：所有既有 CLI flag / verdict 計算 / record_lifecycle + write_data_audit_log 接線不變；新增之 `--drift-tolerance` 屬非破壞性 flag（N=0 完全相容 v0.1 行為）。對應 CLAUDE.md §四 #4 8 項標頭強制檢驗治權慣例。 | SUPERSEDED |
| v0.1 | 2026-05-18 | Codex | 首版：依憲章 §14.7-L 入憲，比對 core+convex 150 × 9 表之 FinMind `api_rows/api_min/api_max` 與 DB `db_rows/db_min/db_max`；支援 `--snapshot-in/--snapshot-out`、`--strict` exit 1、source-empty 合法分流與 targeted backfill commands；`--include-fred` 另比對 FRED 四序列 valid numeric observations。 | SUPERSEDED |
================================================================================
"""
CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.7"

# v0.3 §6.8.8-D / §14.7-AQ 全市場驗證模式之治權常數
# （與 sovereign_sync_engine.FULL_MARKET_REASON_MIN_CHARS / FULL_MARKET_REQUIRED_UNIVERSE 對映；
#  獨立定義以避免 maintenance → ingestion 之反向耦合 import）
FULL_MARKET_REASON_MIN_CHARS = 12
FULL_MARKET_REQUIRED_UNIVERSE = "full"

# v0.4 §0.4 [Hybrid Observability] 進度心跳預設值（每 N 個個股印一條 progress）
# 全市場 ~2,798 支 × default 100 ≈ 28 條 heartbeat；N=0 為靜默模式相容 v0.3
PROGRESS_INTERVAL_DEFAULT = 100

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

_THIS_FILE = Path(__file__).resolve()
_MAINTENANCE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINTENANCE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.path_setup import get_report_dir
    from core.db_utils import get_db_connection, get_core_stocks_from_db, record_lifecycle
    from core.data_schema import DATASET_REGISTRY, FINMIND_API_TABLES
    from core.finmind_client import FinMindClient
except ImportError as exc:
    print(f"❌ 核心組件導入失敗: {exc}")
    sys.exit(1)


STRICT_SOURCE_START_DATE = "1990-01-01"
DEFAULT_THROTTLE_PER_HOUR = 5500
FRED_PAGE_LIMIT = 100000
FRED_SERIES = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]

# v0.7 §6.8.8-E.1 transient timeout retry contract
DEFAULT_API_TIMEOUT = 30                   # seconds; legacy 30
DEFAULT_API_RETRY = 1                      # 預設一次重試;0 = 相容 v0.6 行為(無 retry)
DEFAULT_API_RETRY_BACKOFFS = [30, 300]     # 三階段退避之輕量版本;對齊 §7.3 但更短(audit 非 sync)

# v0.7 §3.2A.I Parallel Audit contract
DEFAULT_WORKERS = 1                        # 預設序列;最大 4 對齊 sync_engine
MAX_WORKERS = 4
STOCK_LEVEL_DATASETS = [
    name for name in FINMIND_API_TABLES
    if name != "TaiwanStockInfo"
]


class SimpleThrottle:
    """Small local throttle for audit probes; keeps audit under §7 request policy.
    v0.7 §3.2A.I: thread-safe for parallel workers via internal Lock."""

    def __init__(self, max_per_hour=DEFAULT_THROTTLE_PER_HOUR):
        self.max_per_hour = max_per_hour
        self.window = deque()
        self.lock = threading.Lock()    # v0.7 §3.2A.I

    def acquire(self):
        with self.lock:
            now = time.time()
            while self.window and self.window[0] < now - 3600:
                self.window.popleft()
            if len(self.window) >= self.max_per_hour:
                sleep_for = 3600 - (now - self.window[0]) + 1
                print(f"⏸  audit throttle sleep {sleep_for:.0f}s ({len(self.window)}/{self.max_per_hour})")
                # 釋放 lock 期間 sleep,讓其他 worker 也能 prune window
                self.lock.release()
                try:
                    time.sleep(sleep_for)
                finally:
                    self.lock.acquire()
                now = time.time()
                while self.window and self.window[0] < now - 3600:
                    self.window.popleft()
            self.window.append(time.time())


class SourceAvailabilityAuditor:
    def __init__(self, start_date=STRICT_SOURCE_START_DATE, throttle_per_hour=DEFAULT_THROTTLE_PER_HOUR,
                 snapshot_in=None, snapshot_out=None, drift_tolerance=3,
                 special_full_market_reason=None,
                 progress_interval=PROGRESS_INTERVAL_DEFAULT,
                 api_timeout=DEFAULT_API_TIMEOUT,
                 api_retry=DEFAULT_API_RETRY,
                 api_retry_backoffs=None,
                 workers=DEFAULT_WORKERS):
        # v0.2: drift_tolerance = audit 時點漂移容忍（per §6.8.8-C / §14.7-AP）
        # 預設 3 個日曆日（覆蓋週末 + 1 工作日緩衝）；0 為嚴格模式相容 v0.1
        # v0.3: special_full_market_reason = §6.8.8-D / §14.7-AQ 全市場 audit 治理理由
        # 對齊 §6.8.7 第 (4) 條五類合法情境；audit 時 audit trail 留存
        # v0.4: progress_interval = §0.4 [Hybrid Observability] 進度心跳頻率
        # 每 N 個 stock 印一次 progress line；N=0 為靜默模式相容 v0.3
        self.constitution_ver = CONSTITUTION_VER
        self.tool_ver = TOOL_VER
        self.start_date = start_date
        self.drift_tolerance = max(0, int(drift_tolerance))
        self.special_full_market_reason = (special_full_market_reason or "").strip() or None
        self.progress_interval = max(0, int(progress_interval))
        # v0.7 §6.8.8-E.1 transient timeout retry
        self.api_timeout = max(1, int(api_timeout))
        self.api_retry = max(0, int(api_retry))
        self.api_retry_backoffs = list(api_retry_backoffs) if api_retry_backoffs else list(DEFAULT_API_RETRY_BACKOFFS)
        # v0.7 §3.2A.I parallel workers
        self.workers = max(1, min(MAX_WORKERS, int(workers)))
        # v0.7 thread-safe stats lock(workers > 1 時用)
        self._stats_lock = threading.Lock()
        self.transient_retries_used = 0    # 退避重試實際發生次數
        self.client = FinMindClient()
        self.throttle = SimpleThrottle(max_per_hour=throttle_per_hour)
        self.snapshot_in = Path(snapshot_in) if snapshot_in else None
        self.snapshot_out = Path(snapshot_out) if snapshot_out else None
        self.api_snapshot = self._load_snapshot(self.snapshot_in)
        self.results = []
        self.fred_results = []
        self.api_errors = 0
        self.checked = 0
        self.mismatch = 0
        self.source_empty_ok = 0
        self.time_drift_ok = 0  # v0.2 新增：§6.8.8-C 時點漂移容忍計數器
        self.fred_checked = 0
        self.fred_mismatch = 0
        self.fred_time_drift_ok = 0  # v0.2 新增：FRED 時點漂移容忍計數器
        self.fred_api_errors = 0

    def _load_snapshot(self, path):
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "items" in payload:
            items = payload["items"]
        elif isinstance(payload, dict) and "finmind" in payload:
            items = payload["finmind"]
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError(f"Unsupported snapshot format: {path}")
        return {
            (str(item["stock_id"]), item.get("dataset") or item.get("table")): item
            for item in items
        }

    def _dump_snapshot(self):
        if not self.snapshot_out:
            return
        self.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        items = [
            {
                "stock_id": r["stock_id"],
                "dataset": r["dataset"],
                "api_rows": r["api_rows"],
                "api_min": r["api_min"],
                "api_max": r["api_max"],
            }
            for r in self.results
            if r["api_status"] == "OK"
        ]
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "start_date": self.start_date,
            "items": items,
        }
        with open(self.snapshot_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _resolve_stocks(self, stock_id=None, universe="core"):
        if stock_id:
            return [str(stock_id)]
        if universe == "core":
            return list(get_core_stocks_from_db(tiers=("core_universe", "convex_universe")))
        if universe == "full":
            # v0.3 §6.8.8-D / §14.7-AQ: 對齊 sovereign_sync_engine.UNIVERSE_TIERS["full"] SSOT
            # 全市場 = core ∪ convex ∪ research ∪ quarantine ≈ 2,798 支
            return list(get_core_stocks_from_db(
                tiers=("core_universe", "convex_universe", "research_universe", "quarantine_universe")
            ))
        raise ValueError(f"Unsupported universe: {universe}; allowed: core / full")

    def _resolve_datasets(self, dataset=None, all_datasets=False):
        if dataset:
            if dataset not in STOCK_LEVEL_DATASETS:
                raise ValueError(f"Unsupported stock-level dataset: {dataset}")
            return [dataset]
        if all_datasets:
            return list(STOCK_LEVEL_DATASETS)
        return ["TaiwanStockPrice", "TaiwanStockPriceAdj", "TaiwanStockFinancialStatements"]

    def _api_summary_from_snapshot(self, stock_id, dataset):
        item = self.api_snapshot.get((str(stock_id), dataset))
        if not item:
            return None
        return {
            "api_status": "OK",
            "api_rows": int(item.get("api_rows") or 0),
            "api_min": item.get("api_min"),
            "api_max": item.get("api_max"),
        }

    def _fetch_api_summary(self, stock_id, dataset):
        cached = self._api_summary_from_snapshot(stock_id, dataset)
        if cached is not None:
            return cached

        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "start_date": self.start_date,
        }
        if self.client.token:
            params["token"] = self.client.token
        # v0.7 §6.8.8-E.1 transient timeout retry contract
        # 對 ReadTimeout / ConnectionError / 5xx 退避重試;quota / 4xx 不重試
        backoffs = list(self.api_retry_backoffs[: self.api_retry])
        last_exc = None
        while True:
            self.throttle.acquire()
            try:
                res = requests.get(self.client.api_url, params=params,
                                   headers=self.client.headers, timeout=self.api_timeout)
                res.raise_for_status()
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                if not backoffs:
                    # 重試耗盡:加上重試記錄後 raise
                    raise type(exc)(
                        "{} after {} retries {}: {}".format(
                            type(exc).__name__, self.api_retry,
                            list(self.api_retry_backoffs[: self.api_retry]), exc)
                    )
                wait = backoffs.pop(0)
                with self._stats_lock:
                    self.transient_retries_used += 1
                print(f"⚠ {type(exc).__name__} on {stock_id}/{dataset};§6.8.8-E.1 sleep {wait}s 後重試",
                      flush=True)
                time.sleep(wait)
                continue
            except requests.exceptions.HTTPError as exc:
                # 5xx 走 retry;4xx (含 quota 之 402) 立即 raise
                code = exc.response.status_code if exc.response is not None else 0
                if 500 <= code < 600 and backoffs:
                    wait = backoffs.pop(0)
                    with self._stats_lock:
                        self.transient_retries_used += 1
                    print(f"⚠ HTTP {code} on {stock_id}/{dataset};§6.8.8-E.1 sleep {wait}s 後重試",
                          flush=True)
                    time.sleep(wait)
                    continue
                raise
        payload = res.json()
        # v0.5 §6.8.8-E: explicit quota substring check（與既有 msg-error 邏輯並列；明示治權契約）
        msg = payload.get("msg", "") or ""
        if "upper limit" in msg.lower() or "rate limit" in msg.lower():
            raise RuntimeError(f"FinMind quota throttled: {msg}")
        if payload.get("msg") not in (None, "", "success"):
            raise RuntimeError(f"FinMind app-level error: {payload.get('msg')}")
        data = payload.get("data", [])
        if not data:
            return {"api_status": "OK", "api_rows": 0, "api_min": None, "api_max": None}

        df = pd.DataFrame(data)
        if "date" not in df.columns:
            raise ValueError(f"{dataset} API response missing date column")
        dates = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna()
        if dates.empty:
            return {"api_status": "OK", "api_rows": 0, "api_min": None, "api_max": None}
        return {
            "api_status": "OK",
            "api_rows": int(len(df[df["date"].notna()])),
            "api_min": str(min(dates)),
            "api_max": str(max(dates)),
        }

    def _db_summary(self, stock_id, dataset, conn):
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"Dataset not in DATASET_REGISTRY: {dataset}")
        columns = DATASET_REGISTRY[dataset]["columns"]
        if "stock_id" not in columns or "date" not in columns:
            raise ValueError(f"Dataset is not stock-level/date-level: {dataset}")
        with conn.cursor() as cur:
            cur.execute(
                f'''
                SELECT COUNT(*), MIN("date"), MAX("date")
                FROM "{dataset}"
                WHERE "stock_id" = %s
                ''',
                (stock_id,),
            )
            rows, min_date, max_date = cur.fetchone()
        return {
            "db_rows": int(rows or 0),
            "db_min": str(min_date) if isinstance(min_date, (date, datetime)) else None,
            "db_max": str(max_date) if isinstance(max_date, (date, datetime)) else None,
        }

    def _fetch_fred_summary(self, series_id):
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED_API_KEY missing")
        observations = []
        offset = 0
        while True:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "limit": FRED_PAGE_LIMIT,
                "offset": offset,
                "sort_order": "asc",
            }
            # v0.7 §6.8.8-E.1: FRED 端也加 transient retry
            backoffs_fred = list(self.api_retry_backoffs[: self.api_retry])
            while True:
                try:
                    res = requests.get("https://api.stlouisfed.org/fred/series/observations",
                                       params=params, timeout=self.api_timeout)
                    res.raise_for_status()
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                    if not backoffs_fred:
                        raise type(exc)("{} after {} retries: {}".format(
                            type(exc).__name__, self.api_retry, exc))
                    wait = backoffs_fred.pop(0)
                    with self._stats_lock:
                        self.transient_retries_used += 1
                    print(f"⚠ FRED {type(exc).__name__} on {series_id};sleep {wait}s 後重試",
                          flush=True)
                    time.sleep(wait)
                    continue
                except requests.exceptions.HTTPError as exc:
                    code = exc.response.status_code if exc.response is not None else 0
                    if 500 <= code < 600 and backoffs_fred:
                        wait = backoffs_fred.pop(0)
                        with self._stats_lock:
                            self.transient_retries_used += 1
                        print(f"⚠ FRED HTTP {code} on {series_id};sleep {wait}s 後重試",
                              flush=True)
                        time.sleep(wait)
                        continue
                    raise
            page = res.json().get("observations", [])
            observations.extend(page)
            if len(page) < FRED_PAGE_LIMIT:
                break
            offset += FRED_PAGE_LIMIT

        valid_dates = []
        for item in observations:
            value = item.get("value")
            if value in (None, "."):
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            valid_dates.append(pd.to_datetime(item.get("date"), errors="coerce").date())
        valid_dates = [d for d in valid_dates if pd.notna(d)]
        return {
            "series_id": series_id,
            "api_valid_rows": len(valid_dates),
            "api_valid_min": str(min(valid_dates)) if valid_dates else None,
            "api_valid_max": str(max(valid_dates)) if valid_dates else None,
        }

    def _db_fred_summary(self, series_id, conn):
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT COUNT(*), MIN("date"), MAX("date")
                FROM "FredData"
                WHERE "series_id" = %s AND "value" IS NOT NULL
                ''',
                (series_id,),
            )
            rows, min_date, max_date = cur.fetchone()
        return {
            "db_valid_rows": int(rows or 0),
            "db_valid_min": str(min_date) if isinstance(min_date, (date, datetime)) else None,
            "db_valid_max": str(max_date) if isinstance(max_date, (date, datetime)) else None,
        }

    def _is_time_drift_ok(self, api_rows, db_rows, api_max, db_max):
        """v0.2 §6.8.8-C 時點漂移容忍判定（v0.6 §14.7-AS hotfix：字串先轉 date 再 subtract）：
        (api_date_max - db_date_max).days ≤ N 且 abs(api_rows - db_rows) ≤ N → TIME_DRIFT_OK
        """
        if self.drift_tolerance <= 0:
            return False
        if api_max is None or db_max is None:
            return False
        try:
            # v0.6 §14.7-AS hotfix: api_max / db_max 為 str（如 "2026-05-22"），先轉 datetime.date 再 subtract
            api_max_date = datetime.strptime(api_max, "%Y-%m-%d").date() if isinstance(api_max, str) else api_max
            db_max_date = datetime.strptime(db_max, "%Y-%m-%d").date() if isinstance(db_max, str) else db_max
            date_drift_days = (api_max_date - db_max_date).days
            row_drift = abs(api_rows - db_rows)
            # 必須 API 領先 DB（DB 不應超前 API）且漂移在容忍範圍內
            return 0 <= date_drift_days <= self.drift_tolerance and row_drift <= self.drift_tolerance
        except Exception:
            return False

    def _classify(self, row):
        if row["api_status"] != "OK":
            return "API_ERROR"
        if row["api_rows"] == 0:
            return "SOURCE_EMPTY_OK" if row["db_rows"] == 0 else "SOURCE_EMPTY_DB_HAS_ROWS"
        if (
            row["api_rows"] == row["db_rows"]
            and row["api_min"] == row["db_min"]
            and row["api_max"] == row["db_max"]
        ):
            return "OK"
        # v0.2: 嘗試 §6.8.8-C 時點漂移容忍判定
        if (
            row["api_min"] == row["db_min"]  # 起點對齊（未漏抓歷史）
            and self._is_time_drift_ok(row["api_rows"], row["db_rows"], row["api_max"], row["db_max"])
        ):
            return "TIME_DRIFT_OK"
        return "MISMATCH"

    def _classify_fred(self, row):
        if row["api_status"] != "OK":
            return "API_ERROR"
        if (
            row["api_valid_rows"] == row["db_valid_rows"]
            and row["api_valid_min"] == row["db_valid_min"]
            and row["api_valid_max"] == row["db_valid_max"]
        ):
            return "OK"
        # v0.2: 嘗試 §6.8.8-C 時點漂移容忍判定
        if (
            row["api_valid_min"] == row["db_valid_min"]  # 起點對齊
            and self._is_time_drift_ok(row["api_valid_rows"], row["db_valid_rows"],
                                       row["api_valid_max"], row["db_valid_max"])
        ):
            return "TIME_DRIFT_OK"
        return "MISMATCH"

    def run(self, stock_id=None, universe="core", dataset=None, all_datasets=False, strict=True, include_fred=False):
        stocks = self._resolve_stocks(stock_id=stock_id, universe=universe)
        datasets = self._resolve_datasets(dataset=dataset, all_datasets=all_datasets)
        task_name = f"audit_source_availability_{stock_id or universe}"
        total_stocks = len(stocks)
        # v0.4 §0.4 [Hybrid Observability] 進度心跳開銷
        run_start = time.time()
        if self.progress_interval > 0 and total_stocks > 0:
            print(
                f"🔍 audit start | universe={universe} | stocks={total_stocks} | datasets={len(datasets)} | "
                f"include_fred={include_fred} | drift_tolerance={self.drift_tolerance} | "
                f"progress_interval={self.progress_interval}",
                flush=True,
            )

        def _process_one_stock(sid, datasets_local, conn_local):
            """v0.7 §3.2A.I: 處理單一 stock 之所有 datasets;thread-safe via _stats_lock。
            回傳 (sid_done, row_count) 供主程序 progress 計數。"""
            local_rows = []
            for ds in datasets_local:
                base = {"stock_id": str(sid), "dataset": ds}
                try:
                    api = self._fetch_api_summary(str(sid), ds)
                    db = self._db_summary(str(sid), ds, conn_local)
                    row = {**base, **api, **db}
                except Exception as exc:
                    with self._stats_lock:
                        self.api_errors += 1
                    row = {
                        **base,
                        "api_status": "ERROR",
                        "api_rows": None,
                        "api_min": None,
                        "api_max": None,
                        "db_rows": None,
                        "db_min": None,
                        "db_max": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                row["status"] = self._classify(row)
                local_rows.append(row)
            # 一次性批次寫入 stats(降低 lock 競爭)
            with self._stats_lock:
                self.results.extend(local_rows)
                self.checked += len(local_rows)
                for r in local_rows:
                    if r["status"] == "SOURCE_EMPTY_OK":
                        self.source_empty_ok += 1
                    elif r["status"] == "TIME_DRIFT_OK":
                        self.time_drift_ok += 1
                    elif r["status"] == "API_ERROR":
                        pass  # 已於 except 計入 api_errors
                    elif r["status"] != "OK":
                        self.mismatch += 1
            return sid

        def _emit_progress(done_count):
            if self.progress_interval > 0 and (done_count % self.progress_interval == 0 or done_count == total_stocks):
                elapsed = time.time() - run_start
                pct = (done_count / total_stocks * 100.0) if total_stocks else 0.0
                eta_sec = (elapsed / done_count * (total_stocks - done_count)) if done_count > 0 else 0
                # snapshot stats under lock
                with self._stats_lock:
                    print(
                        f"🔍 progress | stocks={done_count}/{total_stocks} ({pct:.1f}%) | "
                        f"checked={self.checked} | source_empty_ok={self.source_empty_ok} | "
                        f"time_drift_ok={self.time_drift_ok} | mismatch={self.mismatch} | "
                        f"api_errors={self.api_errors} | retries={self.transient_retries_used} | "
                        f"elapsed={elapsed:.0f}s | eta={eta_sec:.0f}s",
                        flush=True,
                    )

        with record_lifecycle(task_name, category="maintenance", stock_id=stock_id or "SYSTEM") as lifecycle:
            if self.workers <= 1:
                # Serial mode (v0.6 兼容路徑)
                conn = get_db_connection()
                try:
                    for idx, sid in enumerate(stocks, start=1):
                        _process_one_stock(sid, datasets, conn)
                        _emit_progress(idx)
                finally:
                    conn.close()
            else:
                # v0.7 §3.2A.I parallel mode:每 worker 持自己的 DB connection
                done = 0
                done_lock = threading.Lock()
                conn_pool = [get_db_connection() for _ in range(self.workers)]
                try:
                    def _worker_task(sid, slot):
                        _process_one_stock(sid, datasets, conn_pool[slot])
                        with done_lock:
                            nonlocal done
                            done += 1
                            cur_done = done
                        _emit_progress(cur_done)
                        return sid

                    with ThreadPoolExecutor(max_workers=self.workers) as pool:
                        futures = []
                        for i, sid in enumerate(stocks):
                            futures.append(pool.submit(_worker_task, sid, i % self.workers))
                        for f in as_completed(futures):
                            f.result()  # propagate exceptions
                finally:
                    for c in conn_pool:
                        try:
                            c.close()
                        except Exception:
                            pass

            # v0.7: FRED 區段(始終序列,自管 conn 生命週期)
            if include_fred:
                fred_conn = get_db_connection()
                try:
                    for series_id in FRED_SERIES:
                        base = {"series_id": series_id}
                        try:
                            api = self._fetch_fred_summary(series_id)
                            db = self._db_fred_summary(series_id, fred_conn)
                            row = {**base, "api_status": "OK", **api, **db}
                        except Exception as exc:
                            with self._stats_lock:
                                self.fred_api_errors += 1
                            row = {
                                **base,
                                "api_status": "ERROR",
                                "api_valid_rows": None,
                                "api_valid_min": None,
                                "api_valid_max": None,
                                "db_valid_rows": None,
                                "db_valid_min": None,
                                "db_valid_max": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        row["status"] = self._classify_fred(row)
                        with self._stats_lock:
                            self.fred_results.append(row)
                            self.fred_checked += 1
                            if row["status"] == "TIME_DRIFT_OK":
                                self.fred_time_drift_ok += 1
                            elif row["status"] == "API_ERROR":
                                pass
                            elif row["status"] != "OK":
                                self.fred_mismatch += 1
                finally:
                    fred_conn.close()

            self._dump_snapshot()
            verdict = self._verdict(strict=strict)
            if verdict == "FAILED" and hasattr(lifecycle, "mark_failed"):
                lifecycle.mark_failed(f"strict source availability mismatch={self.mismatch}, api_errors={self.api_errors}")
            elif verdict == "WARNING" and hasattr(lifecycle, "mark_warning"):
                lifecycle.mark_warning(f"source availability mismatch={self.mismatch}, api_errors={self.api_errors}")
            self._write_report(verdict, stocks, datasets, universe=universe)
            self._print_summary(verdict, universe=universe)
            return verdict

    def _verdict(self, strict=True):
        if self.api_errors > 0 or self.fred_api_errors > 0:
            return "FAILED"
        if self.mismatch > 0 or self.fred_mismatch > 0:
            return "FAILED" if strict else "WARNING"
        return "PERFECT"

    def _write_report(self, verdict, stocks, datasets, universe="core"):
        report_path = get_report_dir() / f"source_availability_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        mismatches = [r for r in self.results if r["status"] not in {"OK", "SOURCE_EMPTY_OK", "TIME_DRIFT_OK"}]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Strict source availability audit\n\n")
            f.write(f"- **time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"- **constitution**: 系統架構大憲章_{self.constitution_ver}.md "
                f"§14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ\n"
            )
            f.write(f"- **tool**: audit_source_availability {self.tool_ver}\n")
            f.write(f"- **start_date**: {self.start_date}\n")
            f.write(f"- **drift_tolerance**: {self.drift_tolerance} day(s) (§6.8.8-C; 0 = strict)\n")
            f.write(f"- **scope**: stocks={len(stocks)} (universe={universe}), datasets={len(datasets)}\n")
            if universe == FULL_MARKET_REQUIRED_UNIVERSE and self.special_full_market_reason:
                f.write(
                    f"- **special_full_market_reason**: {self.special_full_market_reason} "
                    f"(§6.8.7 第 (4) 條 / §6.8.8-D)\n"
                )
            f.write(f"- **verdict**: **{verdict}**\n")
            f.write(
                f"- **summary**: checked={self.checked}, source_empty_ok={self.source_empty_ok}, "
                f"time_drift_ok={self.time_drift_ok}, mismatch={self.mismatch}, api_errors={self.api_errors}\n\n"
            )
            if self.fred_results:
                f.write(
                    f"- **fred_summary**: checked={self.fred_checked}, "
                    f"time_drift_ok={self.fred_time_drift_ok}, "
                    f"mismatch={self.fred_mismatch}, api_errors={self.fred_api_errors}\n\n"
                )
            f.write("## Mismatches\n\n")
            if not mismatches:
                f.write("None.\n\n")
            else:
                f.write("| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max | error |\n")
                f.write("|---|---|---|---:|---|---|---:|---|---|---|\n")
                for r in mismatches:
                    # v0.5 §6.8.8-E: error 欄位明示 API_ERROR row 之 raw msg / exception type
                    err = (r.get("error") or "").replace("|", "\\|").replace("\n", " ")
                    f.write(
                        f"| {r['stock_id']} | {r['dataset']} | {r['status']} | "
                        f"{r.get('api_rows')} | {r.get('api_min')} | {r.get('api_max')} | "
                        f"{r.get('db_rows')} | {r.get('db_min')} | {r.get('db_max')} | "
                        f"{err} |\n"
                    )
                f.write("\n## Targeted Backfill Commands\n\n")
                for r in mismatches:
                    if r["status"] == "API_ERROR":
                        continue
                    f.write(
                        "```bash\n"
                        f".venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id {r['stock_id']} "
                        f"--dataset {r['dataset']} --strict-source-history\n"
                        "```\n"
                    )
            if self.fred_results:
                f.write("\n## FRED Valid Observation Alignment\n\n")
                f.write("| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |\n")
                f.write("|---|---|---:|---|---|---:|---|---|\n")
                for r in self.fred_results:
                    f.write(
                        f"| {r['series_id']} | {r['status']} | {r.get('api_valid_rows')} | "
                        f"{r.get('api_valid_min')} | {r.get('api_valid_max')} | "
                        f"{r.get('db_valid_rows')} | {r.get('db_valid_min')} | {r.get('db_valid_max')} |\n"
                    )
        self.report_path = report_path

    def _print_summary(self, verdict, universe="core"):
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: strict source availability audit ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"report : {self.report_path}")
        print(f"universe={universe}")
        if universe == FULL_MARKET_REQUIRED_UNIVERSE and self.special_full_market_reason:
            print(f"special_full_market_reason : {self.special_full_market_reason}")
        print(f"drift_tolerance={self.drift_tolerance}")
        print(f"checked={self.checked}")
        print(f"source_empty_ok={self.source_empty_ok}")
        print(f"time_drift_ok={self.time_drift_ok}")
        print(f"mismatch={self.mismatch}")
        print(f"api_errors={self.api_errors}")
        if self.fred_results:
            print(f"fred_checked={self.fred_checked}")
            print(f"fred_time_drift_ok={self.fred_time_drift_ok}")
            print(f"fred_mismatch={self.fred_mismatch}")
            print(f"fred_api_errors={self.fred_api_errors}")
        print(f"verdict={verdict}")
        print("🛡️" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Strict source availability audit (FinMind + FRED)")
    parser.add_argument("--id", type=str, help="single stock_id")
    parser.add_argument("--universe", choices=["core", "full"], default="core",
                        help="authorized universe scope (core = committed snapshot N dynamic per §14.7-BW;full ≈ 2,798 須附 reason 對齊 §6.8.7 第 (4) 條 / §6.8.8-D)")
    parser.add_argument("--dataset", type=str, help="single FinMind stock-level dataset")
    parser.add_argument("--all", action="store_true", help="audit all FinMind stock-level datasets")
    parser.add_argument("--start-date", default=STRICT_SOURCE_START_DATE,
                        help=f"FinMind source lower bound (default {STRICT_SOURCE_START_DATE})")
    parser.add_argument("--strict", action="store_true", help="exit 1 on any mismatch")
    parser.add_argument("--snapshot-in", help="reuse prior API source snapshot JSON")
    parser.add_argument("--snapshot-out", help="write API source snapshot JSON")
    parser.add_argument("--include-fred", action="store_true",
                        help="also verify FRED DFF/UNRATE/T10Y2Y/VIXCLS valid numeric observations against DB")
    parser.add_argument("--throttle", type=int, default=DEFAULT_THROTTLE_PER_HOUR,
                        help="API requests per hour cap for audit probe")
    parser.add_argument("--drift-tolerance", type=int, default=3,
                        help="§6.8.8-C audit 時點漂移容忍 (預設 3 個日曆日；0 = strict mode)")
    parser.add_argument("--special-full-market-reason", type=str, default=None,
                        help=f"(v0.3 §6.8.8-D / §14.7-AQ) 全市場 audit 治理理由 — 必須 ≥ {FULL_MARKET_REASON_MIN_CHARS} 字元；"
                             "僅在 --universe full 時生效；缺 reason 或字數不足即 exit 1")
    parser.add_argument("--progress-interval", type=int, default=PROGRESS_INTERVAL_DEFAULT,
                        help=f"(v0.4 §0.4) 進度心跳頻率，每 N 個 stock 印一條 progress line（預設 {PROGRESS_INTERVAL_DEFAULT}；"
                             "0 = 靜默模式相容 v0.3）")
    parser.add_argument("--api-timeout", type=int, default=DEFAULT_API_TIMEOUT,
                        help=f"(v0.7 §6.8.8-E.1) 單次 API 請求 timeout 秒數（預設 {DEFAULT_API_TIMEOUT}；對大型歷史可調 60-90）")
    parser.add_argument("--api-retry", type=int, default=DEFAULT_API_RETRY,
                        help=f"(v0.7 §6.8.8-E.1) transient timeout / 5xx 退避重試次數（預設 {DEFAULT_API_RETRY}；0 = 相容 v0.6 行為）")
    parser.add_argument("--retry-backoff", type=str, default=",".join(str(x) for x in DEFAULT_API_RETRY_BACKOFFS),
                        help=f"(v0.7 §6.8.8-E.1) 退避秒數 csv 列表（預設 \"{','.join(str(x) for x in DEFAULT_API_RETRY_BACKOFFS)}\"；"
                             "空字串=不退避立即 raise）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"(v0.7 §3.2A.I) 平行 worker 數量（預設 {DEFAULT_WORKERS}；最大 {MAX_WORKERS}）；"
                             "共用 thread-safe throttle + stats lock;預期將 8h+ 全市場 audit 縮至 ~2-3h")
    args = parser.parse_args()

    # v0.3 §6.8.8-D / §14.7-AQ preflight 治權檢查（對齊 §6.8.7 第 (4) 條範本）
    if args.universe == FULL_MARKET_REQUIRED_UNIVERSE:
        reason = (args.special_full_market_reason or "").strip()
        if not reason:
            print(f"❌ [§6.8.8-D / §14.7-AQ] --universe full 必須附 --special-full-market-reason \"<≥{FULL_MARKET_REASON_MIN_CHARS} 字理由>\"")
            print("   合法情境：DB rebuild bootstrap / Sovereign rebuild / pre-annual audit / 資料源治權變更 / 合規事件")
            sys.exit(1)
        if len(reason) < FULL_MARKET_REASON_MIN_CHARS:
            print(f"❌ [§6.8.8-D / §14.7-AQ] --special-full-market-reason 長度 {len(reason)} < {FULL_MARKET_REASON_MIN_CHARS} 字元下限")
            sys.exit(1)
    elif args.special_full_market_reason:
        print(f"❌ [§6.8.8-D / §14.7-AQ] --special-full-market-reason 僅在 --universe full 時生效；"
              f"目前 --universe={args.universe}，拒絕執行")
        sys.exit(1)

    # v0.7 §6.8.8-E.1: parse retry backoffs csv
    retry_backoffs = None
    if args.retry_backoff is not None:
        s = args.retry_backoff.strip()
        if s == "":
            retry_backoffs = []
        else:
            try:
                retry_backoffs = [int(x.strip()) for x in s.split(",") if x.strip()]
            except ValueError:
                print(f"❌ --retry-backoff 解析失敗：{args.retry_backoff!r}（需 csv 整數，如 \"30,300\"）")
                sys.exit(1)
    auditor = SourceAvailabilityAuditor(
        start_date=args.start_date,
        throttle_per_hour=args.throttle,
        snapshot_in=args.snapshot_in,
        snapshot_out=args.snapshot_out,
        drift_tolerance=args.drift_tolerance,
        special_full_market_reason=args.special_full_market_reason,
        progress_interval=args.progress_interval,
        api_timeout=args.api_timeout,
        api_retry=args.api_retry,
        api_retry_backoffs=retry_backoffs,
        workers=args.workers,
    )
    verdict = auditor.run(
        stock_id=args.id,
        universe=args.universe,
        dataset=args.dataset,
        all_datasets=args.all,
        strict=args.strict,
        include_fred=args.include_fred,
    )
    sys.exit(0 if verdict in {"PERFECT", "WARNING"} else 1)


if __name__ == "__main__":
    main()
