"""
sovereign_sync_engine.py v1.20 (Quantum Finance Market Universe Seed Engine · Functional Group Matrix Edition · §14.7-AM 雞與蛋缺陷補強：4 步序列 + Cross-ref Calibration #3)
================================================================================
**最後更新日期**: 2026-05-21
**主權狀態**: SUPPLY CHAIN RATE SOVEREIGNTY ALIGNED + STRICT SOURCE HISTORY + FULL-MARKET RESTRICTED GOVERNANCE EXCEPTION + AUTO STRICT-SOURCE-HISTORY ON FULL UNIVERSE + --full-history ALIAS FOR CORE FULL-HISTORY MODE + §14.7-AL CROSS-REF CALIBRATION + §14.7-AM ZERO-TO-FULL-MARKET+FRED SEQUENCE TREATY + §14.7-AM POST-INSCRIPTION CROSS-REF CALIBRATION #2 + §14.7-AM 雞與蛋缺陷補強 4 步序列 + CROSS-REF CALIBRATION #3 (憲法 v6.0.0-FINAL §7 / §14.7-L / §6.8.7 第 (1A) / 第 (4) 條對齊 + §3.1 序列模組身分自我宣告 + 維運矩陣重組為 8 大功能群視角 + §14.7-AL/AM 雙入憲後行號 3 次校準累計 + §14.7-AM 雞與蛋缺陷補強之 4 步序列治權範本明文化；8 項標頭強制檢驗 100% 合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Market Universe Seed]: 第 4 步（憲章 §二 Step 4 L2428）取得 `TaiwanStockInfo` 市場個股清單，並同步 FRED 核心宏觀資料。
2. [Schema Sovereignty]: 所有寫入欄位必須 100% 對齊 `data_schema.py` 當前版本之 `DATASET_REGISTRY`（透過 import；非硬鎖版本）。
3. [Hybrid Observability]: 使用 `record_lifecycle(... ) as lc`，將 warning / failed 回寫 pipeline lifecycle。
4. [Zero Silent Drop]: API 空回應、HTTP 4xx/5xx、dropna 全空、DB upsert 失敗皆必須記入 stats 與 terminal report。
5. [Idempotency]: 使用 ON CONFLICT upsert，確保 seed / sync 可重跑。
6. [Phase-Appropriate Lookback]: 預設 `--days 30` 為日常增量；CoreScore v0.2 選股 phase 建議 `--days 730`
   （對應憲章 6.4 price_coverage_252d + revenue_coverage_24m + financial_coverage_8q）。
7. **[Supply Chain Rate Sovereignty]** (v1.10, 憲法 §7)：對 FinMind 6000/hr 上限與 30 分鐘重置週期實作三層防禦：
   - **L1 事前預防**：滑動窗節流 5500/hr（保留 8% 餘裕）
   - **L2 事中應對**：402 單次 1800s 探測重試；403/429/5xx/Timeout 三階段退避 [30s, 300s, 1800s]
   - **L3 事後續跑**：DB-driven checkpoint，已同步之 (stock_id, dataset, ≥start_date) 不再呼叫 API
8. **[402 vs 403 分流]** (v1.10, 憲法 §7.4)：402 預設視為「資料集付費門檻」單次重試；403/429 視為「速率超限」完整三階段退避。
9. **[Full-Market Restricted Governance Exception]** (v1.13, 憲法 §6.8.7 第 (4) 條 / §二 Step 4F L2434)：`--universe full` 解鎖
   `core ∪ convex ∪ research ∪ quarantine` 全 2,798 支 sync；**必須**附 `--special-full-market-reason "<≥12 字理由>"`
   且情境屬 §6.8.7 第 (4) 條五類合法清單（DB rebuild bootstrap / Sovereign rebuild / pre-annual audit /
   重大資料源治權變更 / 重大合規事件）；缺 reason 或 reason < 12 字即 preflight FAILED exit 1；
   reason 字串寫入 lifecycle context 與報表，留 audit trail。
10. **[Full-History Per-Stock Earliest-Date Semantic]** (v1.14, 憲法 §6.8.7 第 (4) 條)：「全天數」之嚴格定義
    為「每 (stock_id, dataset) 對自 FinMind/FRED API 來源端最早可得日期 → DB 最後一個交易日」；**不**等同
    `--days N` 固定天數窗、**不**強制統一 start_date。`--universe full` 觸發時 main() preflight **自動啟用**
    `--strict-source-history`（強制 `start_date=1990-01-01` + 停用 §7.5 L3 resume，FinMind API 自動回傳真實最早
    可得日期）；FRED 同步維持既有 `asc + offset` 全歷史分頁。`--days` 在 full mode 下退為 safety floor，
    實際同步起點由 strict-source-history 決定；使用者毋須另下 `--strict-source-history` 旗標。
11. **[Core Full-History Mode `--full-history` Alias]** (v1.15, 憲法 §6.8.7 第 (1A) 條 / §二 Step 4G L2435)：核心股全天數補刷模式，
    為 `--strict-source-history` 之直觀別名旗標；兩旗標**等價**，任一啟用即觸發 §14.7-L 行為
    （start_date=1990-01-01 + §7.5 L3 resume 停用）。適用 `--universe core/convex/research` 或 `--id <stock_id>`，
    **無需** `--special-full-market-reason`（核心股範圍合憲，屬 §6.8.7 第 (1) 條合法範圍）。標準用法：
    `--universe core --all --full-history --dataset-batched --workers 4`，耗時 ~5-10 分鐘、~1-2M rows。
    對 `--universe full` 之語意不變（仍須 reason）。
12. **[Zero Hardcoded Verdict]** (v1.16, 憲法 §5.6.3)：主權判定動態計算（`_apply_lifecycle_verdict()`）：
    依 `stats.failed` / `a5_warn_count` / `a5_pause_count` 動態決定 lifecycle marker；§7.6 A5 觸發 (warn_count > 0 或 pause_count > 0)
    即使 stats.failed=0 也升級為 WARNING；不硬編 PERFECT/WARNING/FAILED 結論。對齊全系統治權慣例
    （path_setup v4.46 / data_schema v2.16 / db_utils v2.47 / audit_supply_chain v1.19）。
13. **[Sovereignty Declaration]** (v1.16, 憲法 §3.1 序列模組身分；v1.17 cross-ref 行號校準)：本程式為 **§3.1 序列執行模組第 5/9 員**
    （cross-ref 憲章 §3.1 子表 L2459「`sovereign_sync_engine.py`」），對應 **§二 維運矩陣 Step 4 / 4D / 4E / 4F / 4G / 5 / 6 / 7 / 8**
    （L2428 / L2432 / L2433 / L2434 / L2435 / L2436 / L2437 / L2438 / L2439），唯一授權之 ingestion 載體。
    **v6.0.0-patch §14.7-AL（2026-05-21 入憲）關聯**：本程式 Step 4 之執行為 hub 治權閉環之**必要前置**——hub PERFECT
    verdict 之 floor 門檻 = Step 4B 後（`core_universe_builder --commit` 之後）、ceiling 時點 = Step 4C 後，
    Step 4.5 [hub 治權閉環確認] 屬非必須之 ceiling time-point 選擇；本程式自身執行 Step 4 種子灌溉，**不**直接觸及 Step 4.5
    （hub 屬 §3.2 橫切；本程式屬 §3.1 序列）。
    **v6.0.0-patch §14.7-AM（2026-05-21 入憲；同日雞與蛋缺陷補強為 4 步序列）關聯**：本程式為「從零 → 全市場全天數 + FRED 全歷史」**4 步序列**治權範本之**唯一執行載體**（Step 4 `--seed` → **Step 4B bootstrap_init `core_universe_builder --commit --special-rebalance-reason "..."`** → Step 4F `--universe full --all --special-full-market-reason "<≥12 字>"` → **Step 4B bootstrap_final** → Step 8 optional `--source fred`）；
    **⚠️ 雞與蛋缺陷實證**：本程式 `_resolve_stocks()` L767-776 之 `--universe full` 查詢 `core_universe_membership` committed snapshot，故**必須先執行 Step 4B bootstrap_init**（以 `latest_registry_fallback` mode 強制 commit）才能 Step 4F；本日實證揭露 → 同日入憲補強（首例「實證即補強」治權閉環）。
    FinMind 與 FRED 屬不同來源，**無單一指令可同時涵蓋**——`--universe full --all` 僅同步 `FINMIND_API_TABLES`（10 raw tables），不觸及 FRED；必須依 4 步序列執行（詳見 Group D + Group F + 治權範本子節）。
    治權邊界：(a) §3.1 序列模組執行 ingestion；(b) 五套禁令（§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §6.8）不涉；
    (c) **T1-T3 不分層**；(d) **§8.5 anti-leakage 不處理**（由 `audit_leakage.py` 負責）；
    (e) **不選股不評分**（由 `core_universe_builder.py` 負責）；(f) **不持有 Raw API Schema**（由 `data_schema.py` 持有）；
    (g) **不建 governance tables**（由 `core_universe_schema.py` 負責）；(h) 唯一職責：對齊 `DATASET_REGISTRY` 寫入 FinMind + FRED 原始資料。
14. **[Historical Reference Authority]** (v1.16; v1.17 cross-ref 行號校準)：本程式之 `schema_ver` 屬於記述性快照（記載當下對齊之 `data_schema` 版本），
    非權威來源；`DATASET_REGISTRY` 之權威來源永遠是 `data_schema.py` 當前版本之 import 結果。§3.1 子表 L2459 之
    「對齊 `data_schema.py v2.11`」表述為治權記述，非硬鎖版本（憲章本身為快照記錄，本程式對齊當前 `data_schema v2.16`）。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

> 本程式作為 §3.1 序列模組第 5/9 員（唯一 ingestion 載體），依「灌溉面向」拆分為 8 大功能群；每群對應憲章治權契約。
> 對齊憲章 §二 維運矩陣 Step 4 / 4D / 4E / 4F / 4G / 5 / 6 / 7 / 8（L2428, L2432-2439）。**v6.0.0-patch §14.7-AL/AM 入憲後行號三次校準**：(I) §14.7-AL（Step 4.5 hub 治權閉環新行插入）→ v1.17 第 1 次校準；(II) §14.7-AM（修訂歷程頂部 +1 entry）→ v1.19 第 2 次校準；(III) **§14.7-AM 雞與蛋缺陷補強（修訂歷程頂部再 +1 entry）→ v1.20 第 3 次校準（本版）**。

### Group A. 種子灌溉 (Seed Ingestion) — `--seed`
| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 `TaiwanStockInfo` 全市場資產名冊灌入（~2,798 支）| `sync_finmind(["TaiwanStockInfo"], seed_mode=True)` | §二 Step 4 L2428 |
| A.2 可與其他旗標組合（如 `--seed --source fred`）| 多軸並行 | §3.1 ingestion 載體 |
| 對應 CLI | `--seed` | — |

### Group B. 個股同步 (Single-Stock Sync) — `--id <stock>`
| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 單一個股單一 dataset | `--id 2330 --dataset TaiwanStockPrice` | §二 Step 5 L2436 |
| B.2 單一個股核心 4 表預設 | `--id 2330` | §二 Step 6 L2437 |
| B.3 單一個股全 datasets | `--id 2330 --all` | §3.1 ingestion |
| 對應 CLI | `--id <stock_id>` (+ optional `--dataset` / `--all`) | — |

### Group C. Universe 階段灌溉 (Universe Tier Ingestion) — `--universe research/convex/core`
| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 研究宇宙第一階段（`--universe research --all --days 730`）| 730d for CoreScore v0.2 | §二 Step 4D L2432 |
| C.2 凸性宇宙第二階段補刷（`--universe convex --all --days 730`）| 730d | §二 Step 4E L2433 |
| C.3 核心宇宙最終同步（`--universe core --all --days 730`）| 730d | §二 Step 7 L2438 / 5.5.3 第 5 條 |
| C.4 核心宇宙日常維運（`--universe core`，不帶 `--all`）| 預設 30d 增量 | §二 Step 7 L2438 |
| C.5 §6.7 SQL 契約查詢 tier | `UNIVERSE_TIERS["research/convex/core"]` | §6.7 SSOT |
| 對應 CLI | `--universe research|convex|core [--all] [--days N]` | — |

### Group D. 全市場治理例外 (Full-Market Restricted Governance Exception) — `--universe full`
| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 全 2,798 支（`core ∪ convex ∪ research ∪ quarantine`）| `UNIVERSE_TIERS["full"]` | §6.8.7 第 (4) 條 |
| D.2 強制 `--special-full-market-reason "<≥12 字理由>"` | preflight 驗證；缺 reason 或 < 12 字即 exit 1 | §6.8.7 第 (4) 條五類合法清單 |
| D.3 auto strict-source-history（v1.14 自動啟用）| `start_date=1990-01-01` + L3 resume off | §6.8.7 第 (4) 條 / §14.7-L |
| D.4 reason 寫入 lifecycle context + 終端報表 | audit trail | §0.4 可觀察性 |
| D.5 五類合法情境 | DB rebuild / Sovereign rebuild / pre-annual audit / 資料源治權變更 / 合規事件 | §6.8.7 第 (4) 條 |
| D.6 「從零 → 全市場全天數」**4 步序列**之第 III 步 | Step 4 → **Step 4B bootstrap_init** → **Step 4F (本群組)** → Step 4B bootstrap_final → Step 8 optional | §14.7-AM 補強 (2026-05-21) |
| **D.7 ⚠️ 雞與蛋 precondition**：必須先有 `core_universe_membership` committed | `_resolve_stocks()` L767-776 之 `--universe full` 查詢 committed snapshot；空 DB 情境必須先跑 Step 4B `core_universe_builder --commit --special-rebalance-reason "..."`（`latest_registry_fallback` mode） | §14.7-AM 雞與蛋實證 (2026-05-21) |
| 對應 CLI | `--universe full --all --dataset-batched --workers 4 --dynamic-quota --special-full-market-reason "<≥12 字>"` | §二 Step 4F L2434 |

### Group E. 核心股全天數補刷 (Core Full-History Mode) — `--full-history` / `--strict-source-history`
| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| E.1 `--full-history` 為 `--strict-source-history` 之直觀別名 | argparse `dest='strict_source_history'` 共用 | §6.8.7 第 (1A) 條 |
| E.2 適用 `--universe core/convex/research` 或 `--id <stock>` | 無需 reason（核心股範圍合憲）| §6.8.7 第 (1) 條合法範圍 |
| E.3 觸發 §14.7-L 行為 | `start_date=1990-01-01` + §7.5 L3 resume 停用 | §14.7-L |
| E.4 標準用法效能 | `--universe core --all --full-history --dataset-batched --workers 4` ~5-10 分鐘 / ~1-2M rows | §6.8.7 第 (1A) 條 |
| E.5 對 `--universe full` 仍須 reason | 不與 D 衝突 | §6.8.7 第 (4) 條 |
| 對應 CLI | `--full-history` 或 `--strict-source-history`（任一）| §二 Step 4G L2435 |

### Group F. FRED 宏觀同步 (FRED Macro Sync) — `--source fred`
| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| F.1 4 大序列同步（DFF / UNRATE / T10Y2Y / VIXCLS）| `sync_fred()` (L691-715) | §二 Step 8 L2439 |
| F.2 「**全歷史**」= 自 FRED API 各序列可得最早日期 → 今天（預設行為）| **無 `observation_start` / `observation_end` 參數** | §14.7-AM (2026-05-21 入憲) |
| F.3 4 序列各自最早可得日期 | UNRATE 1948-01-01 / DFF 1954-07-01 / T10Y2Y 1976-06-01 / VIXCLS 1990-01-02 | §6.8.8 L3266-3271 / §14.7-AM |
| F.4 `asc + offset` 全歷史分頁 | `sort_order=asc` + `FRED_PAGE_LIMIT = 100000`；`while len(page) >= FRED_PAGE_LIMIT: offset += FRED_PAGE_LIMIT` | FRED API contract |
| F.5 `--days N` 在 FRED 路徑**無影響** | 純針對 FinMind stock-level sync | §14.7-AM |
| F.6 **無需 reason**（FRED 全歷史屬預設行為，非限定治理例外）| 不需 `--special-full-market-reason` | §14.7-AM (vs Group D 對照)|
| F.7 dropna 後空集合分支防禦 | v1.7 補正 | [Zero Silent Drop] |
| F.8 empty-data 失敗分支 | v1.7 補正 | [Zero Silent Drop] |
| F.9 「從零 → 全市場全天數 + FRED 全歷史」3 步序列之第 III 步 | Step 4 `--seed` → Step 4F → **Step 8 (本群組)** | §14.7-AM |
| 對應 CLI | `--source fred` | §二 Step 8 L2439 |

### Group G. §7 三層防禦 + §7.6 進階優化 (Supply Chain Defenses)
| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| G.1 L1 滑動窗節流 5500/hr | `FinMindThrottle` (`DEFAULT_THROTTLE_PER_HOUR=5500`)；禁止 ≥ 6000 | §7.2 |
| G.2 L2 三階段退避 [30s, 300s, 1800s] | `fetch_with_retry()` (`RETRY_BACKOFFS_FULL`) | §7.3 |
| G.3 L3 DB-driven resume | `is_already_synced()` checkpoint | §7.5 |
| G.4 402 單次 1800s 探測重試 | `RETRY_BACKOFF_402=[1800]` | §7.4 |
| G.5 §7.6 A1 dataset 優先迴圈 | `--dataset-batched` | §7.6 A1 |
| G.6 §7.6 A2 thread-safe 平行 worker | `--workers N` (`threading.Lock`) | §7.6 A2 |
| G.7 §7.6 A3 動態配額查詢 | `--dynamic-quota --quota-interval N` (N≥100) | §7.6 A3 |
| G.8 §7.6 A4 per-dataset 配額快照 | `write_data_audit_log(op_type='QUOTA_HOURLY_SNAPSHOT')` | §7.6 A4 |
| G.9 §7.6 A5 4800/hr WARN + 5500/hr 自動暫停 300s | `A5_WARN_THRESHOLD` / `A5_PAUSE_THRESHOLD` / `A5_PAUSE_DURATION` | §7.6 A5 |
| 對應 CLI | `--no-resume` / `--throttle N` / `--dataset-batched` / `--workers N` / `--dynamic-quota` / `--quota-interval N` | — |

### Group H. Lifecycle + 動態主權判定 (Verdict)
| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| H.1 `record_lifecycle()` context 包裹整個 sync 流程 | `run()` 內之 `with record_lifecycle(...)` | §0.4 可觀察性 |
| H.2 `stats.failed > 0` → lifecycle FAILED | `_apply_lifecycle_verdict()` | §5.6.3 動態 verdict |
| H.3 `a5_warn_count > 0` 或 `a5_pause_count > 0` → lifecycle WARNING（即使 stats.failed=0）| v1.11a 邊界補正 | §7.6 A5 升級邏輯 |
| H.4 PERFECT/WARNING → exit 0；FAILED → exit 1 | `run()` 末段 | §3.1 接受標準 |
| H.5 `write_data_audit_log()` per-stock × per-dataset 寫入 | op_type 擴充 `RETRY_402_RECOVERED` / `RESUME_SKIP` / `QUOTA_HOURLY_SNAPSHOT` | §0.4 audit trail |
| 對應 CLI | 所有模式皆觸發 | — |

### 對齊憲章 §二 維運矩陣（標準場景索引）
| 場景 | 指令 | 對應功能群 |
| :--- | :--- | :--- |
| **4. [種子灌溉]** (L2428) | `python scripts/ingestion/sovereign_sync_engine.py --seed` | A + H |
| **4D. [研究宇宙第一階段]** (L2432) | `--universe research --all --days 730` | C + G + H |
| **4E. [凸性宇宙第二階段]** (L2433) | `--universe convex --all --days 730` | C + G + H |
| **4F. [全市場治理例外]** (L2434) | `--universe full --all --dataset-batched --workers 4 --dynamic-quota --special-full-market-reason "<≥12 字>"` | D + G + H |
| **4G. [核心股全天數補刷]** (L2435) | `--universe core --all --full-history --dataset-batched --workers 4` | E + G + H |
| **5. [單一標的指定 dataset]** (L2436) | `--id 2330 --dataset TaiwanStockPrice` | B + G + H |
| **6. [單一標的核心 datasets]** (L2437) | `--id 2330` | B + G + H |
| **7. [核心 Universe 同步]** (L2438) | `--universe core --all --days 730` | C + G + H |
| **8. [FRED 宏觀]** (L2439) | `--source fred` | F + H |

### 「從零 → 全市場全天數 + FRED 全歷史」執行序列治權範本（§14.7-AM 入憲；2026-05-21）

> 當情境屬 §6.8.7 第 (4) 條五類合法之 **(1) DB rebuild bootstrap** 時，標準執行序列為 3 步：

| 步驟 | 指令 | 涵蓋範圍 | 對應功能群 | 治權契約 |
| :--- | :--- | :--- | :--- | :--- |
| **I. Step 4** | `python scripts/ingestion/sovereign_sync_engine.py --seed` | `TaiwanStockInfo` 全市場資產名冊（~2,798 支）| **A** + H | §二 L2428 |
| **II. Step 4B bootstrap_init** ⚠️ | `python scripts/core/core_universe_builder.py --commit --as-of-date <date> --special-rebalance-reason "DB rebuild bootstrap <YYYY-MM-DD> init"` | **解開雞與蛋邊界** — 強制 commit bootstrap snapshot (via `latest_registry_fallback` mode；CoreScore=0 屬合憲過渡) | （外部 core_universe_builder v0.2）| §6.4 / §6.8.6 / §14.7-AM 雞與蛋實證 |
| **III. Step 4F** | `python scripts/ingestion/sovereign_sync_engine.py --universe full --all --dataset-batched --workers 4 --dynamic-quota --special-full-market-reason "<≥12 字理由>"` | FinMind 10 raw tables × 2,798 支 × 各 (stock_id, dataset) 自 API 最早可得日期 → 今天；自動啟用 `--strict-source-history`；現 universe 已 committed 可成功 sync | **D** + G + H | §二 L2434 / §6.8.7 第 (4) 條 |
| **IV. Step 4B bootstrap_final** | `python scripts/core/core_universe_builder.py --commit --as-of-date <date> --special-rebalance-reason "DB rebuild bootstrap <YYYY-MM-DD> final"` | real-data snapshot 覆蓋 bootstrap snapshot（per §6.8.6 stage 規範）| （外部 core_universe_builder v0.2）| §6.4 / §6.8.6 |
| **V. Step 8** (optional) | `python scripts/ingestion/sovereign_sync_engine.py --source fred` | FRED 4 序列 × 各自最早可得日期 → 今天（**無需 reason**；已被 Step I + III 自動觸發；獨立執行屬 idempotent）| **F** + H | §二 L2439 / §6.8.8 |

**治權邊界 4 點**：
- (a) FinMind 與 FRED 屬不同來源，**無單一指令可同時涵蓋兩者**——`--universe full --all` 僅同步 `FINMIND_API_TABLES`（10 raw tables），不觸及 FRED；必須分別執行 Step 4F + Step 8
- (b) Step 4F 需 `--special-full-market-reason ≥ 12 字`；Step 8 無需 reason（FRED 全歷史屬預設行為）
- (c) Step 4F 完成後依 §6.8.7-B 強制長跑持久化規範（`tmux` / `systemd-run --user`）與 30 分鐘監控回報
- (d) Step 4F + Step 8 兩者完成後必須執行雙稽核（`audit_supply_chain.py` + `audit_api_schema_compliance.py --include-fred`）並留 `reports/full_market_sync_*.md` 實證

**典型耗時**：FinMind 全市場全天數約 6-10 小時級（依 quota；§7 三層防禦自動 throttle）；FRED 全歷史約秒級（4 序列 × `asc + offset` pagination）。

### 不提供之旗標 (Intentionally Omitted)
- `--force` (全歷史 from 2000-01-01)：仍維持不提供，避免在 v0.2 builder 完工前耗盡 FinMind quota。使用者應改用 `--full-history` 或 `--strict-source-history`（屬 Group E）。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.20** | 2026-05-21 | Codex | **§14.7-AM 雞與蛋缺陷補強之 4 步序列治權範本明文化 + cross-ref 第 3 次校準（16 處）**：依憲章 v6.0.0-patch §14.7-AM 雞與蛋缺陷補強（commit `fea89bf`；2026-05-21 入憲）將「從零 → 全市場全天數」**3 步序列 → 4 步序列**之治權正解寫入標頭。**補正內容 8 項**：(a) L2 header v1.19 → v1.20 + 副標補入「§14.7-AM 雞與蛋缺陷補強：4 步序列 + Cross-ref Calibration #3」；(b) L5 主權狀態補入 v1.20 修補摘要（含 3 次行號校準累計）；(c) [Sovereignty Declaration] 補入「§14.7-AM 雞與蛋缺陷實證 + 4 步序列治權範本」關聯 + 雞與蛋實證明文（`_resolve_stocks()` L767-776 查詢 committed snapshot 之 chicken-and-egg）；(d) Group D 新增 D.6 「4 步序列之第 III 步」+ D.7 「⚠️ 雞與蛋 precondition」兩條目；(e) 治權範本 sub-section **3 步序列 (I/II/III) → 4 步序列 (I/II/III/IV) + V optional**：新增 Step 4B bootstrap_init (II) + Step 4B bootstrap_final (IV) 兩個外部 core_universe_builder 階段；(f) 16 處 cross-ref 行號 +1 校準（憲章 v6.0.0-patch §14.7-AM 補強 commit `fea89bf` 之修訂歷程頂部 +1 entry 造成）；(g) TOOL_VER v1.19 → v1.20；(h) 修訂歷程 v1.19 → SUPERSEDED + 新 v1.20 ACTIVE entry。**介面零變動**：所有 CLI flag / preflight 邏輯 / UNIVERSE_TIERS / 節流 / 退避 / FRED pagination / verdict 邏輯全保留；本程式自身**不處理 bootstrap commit**（屬 `core_universe_builder.py` v0.2 之 `--special-rebalance-reason` + `latest_registry_fallback` mode 治權）。**本日第 3 次治權邊界相對性循環實證**：v1.16→v1.17（§14.7-AL 10 處）→ v1.18→v1.19（§14.7-AM #1 15 處 + 雙 ACTIVE 修正）→ v1.20（§14.7-AM 雞與蛋補強 16 處）；累計 41 處校準 + 1 治權違規修正 + 4 步序列治權範本明文化；對應 CLAUDE.md §四 #4 8 項標頭強制檢驗治權慣例。 | **ACTIVE** |
| v1.19 | 2026-05-21 | Codex | **§14.7-AM 入憲後 cross-ref 行號第 2 次校準（15 處；本日第 2 次治權邊界相對性循環）**：依憲章 v6.0.0-patch §14.7-AM（commit `17b3c69`；2026-05-21 入憲）之修訂歷程頂部新增 +1 entry 造成 §二 維運矩陣行號 +1 偏移（Step 4 額外 +1，總 +2），本標頭 v1.18 之 15 處 cross-ref 行號漂移已校準：(a) [Market Universe Seed] Step 4 L2425 → L2427；(b) [Full-Market Restricted] Step 4F L2432 → L2433；(c) [Core Full-History] Step 4G L2433 → L2434；(d) [Sovereignty Declaration] §3.1 子表 L2455 → L2458；(e) [Sovereignty Declaration] Step 4-8 範圍 L2425/L2430-2437 → L2427/L2431-2438；(f) [Historical Reference Authority] §3.1 子表 L2455 → L2458；(g) 維運矩陣 sub-title L2425, L2430-2437 → L2427, L2431-2438；(h) 功能群 A.1 Step 4 L2425 → L2427；(i) 功能群 B.1/B.2 Step 5/6 L2434/L2435 → L2435/L2436；(j) 功能群 C.1/C.2/C.3/C.4 L2430/L2431/L2436/L2436 → L2431/L2432/L2437/L2437；(k) 功能群 D 對應 CLI Step 4F L2432 → L2433；(l) 功能群 E 對應 CLI Step 4G L2433 → L2434；(m) 功能群 F.1/對應 CLI Step 8 L2437 → L2438；(n) 維運矩陣場景索引 9 行 L2425/L2430-2437 → L2427/L2431-2438；(o) 治權範本 sub-section Step 引用 L2425/L2432/L2437 → L2427/L2433/L2438。**TOOL_VER v1.18 → v1.19；主權狀態行補入「+ §14.7-AM POST-INSCRIPTION CROSS-REF CALIBRATION #2」摘要**。**介面零變動**：所有 CLI flag / preflight 邏輯 / UNIVERSE_TIERS / 節流 / 退避 / FRED pagination / verdict 邏輯全保留。**本日第 2 次治權邊界相對性循環實證**：v1.16→v1.17（§14.7-AL 入憲後 15 處校準）→ v1.18→v1.19（§14.7-AM 入憲後 15 處再校準），共 30 處校準；對應 CLAUDE.md §四 #4 8 項標頭強制檢驗治權慣例。 | SUPERSEDED |
| v1.18 | 2026-05-21 | Codex | **§14.7-AM「從零 → 全市場全天數 + FRED 全歷史」執行序列治權範本明文化**：依憲章 v6.0.0-patch §14.7-AM（commit `17b3c69`；2026-05-21 入憲）將本程式之「從零 → 全市場全天數 + FRED 全歷史」3 步序列治權範本同步寫入標頭。**補正內容 5 處**：(a) 主權狀態行補入「+ §14.7-AM ZERO-TO-FULL-MARKET+FRED SEQUENCE TREATY」+ v1.18 修補摘要；(b) [Sovereignty Declaration] 補入 §14.7-AM 關聯說明（本程式為「從零 → 全市場全天數 + FRED 全歷史」3 步序列治權範本之唯一執行載體 + FinMind 與 FRED 無單一指令可同時涵蓋之治權邊界）；(c) Group D 新增 D.6「3 步序列之第 II 步」cross-ref；(d) Group F 從 4 子項擴至 9 子項：F.1 sync_fred() L691-715 / F.2「全歷史」明文（無 observation_start/end）/ F.3 4 序列最早日期（UNRATE 1948-01-01 / DFF 1954-07-01 / T10Y2Y 1976-06-01 / VIXCLS 1990-01-02）/ F.4 asc+offset pagination / F.5 --days 無影響 / F.6 無需 reason / F.7-F.8 既有 zero-silent-drop / F.9「3 步序列之第 III 步」；(e) 9 場景索引後新增「從零 → 全市場全天數 + FRED 全歷史」治權範本 sub-section（3 步序列 I/II/III 表 + 治權邊界 4 點 + 典型耗時）；(f) TOOL_VER v1.17 → v1.18。**介面零變動**：所有 CLI flag / preflight 邏輯 / UNIVERSE_TIERS / 節流 / 退避 / FRED pagination / verdict 邏輯全保留。本補正屬「憲章入憲 → 標頭明文同步」之自然對齊；對應 CLAUDE.md §四 #4 8 項標頭強制檢驗治權慣例 + §0.0-I 單一引用源原則（憲章 §14.7-AM ↔ 本標頭治權範本 sub-section ↔ Group D/F 條目同步）。 | SUPERSEDED |
| v1.17 | 2026-05-21 | Codex | **§14.7-AL 入憲後 cross-ref 行號校準 + Step 4.5 hub 治權閉環關聯補入**：依憲章 v6.0.0-patch §14.7-AL（2026-05-21 入憲；commit `961a55f`）於 §二 維運矩陣 Step 4C 後新增 **Step 4.5 [hub 治權閉環確認 ceiling time-point]** 行（憲章 L2429），造成原 L2428 之後行號 **+2 偏移**。本標頭 v1.16 之 10 處 cross-ref 行號漂移已校準：(a) [Sovereignty Declaration] 內 Step 4D-8 範圍 L2428-2435 → L2430-2437；(b) [Sovereignty Declaration] 內 §3.1 子表 L2453 → L2455；(c) [Historical Reference Authority] 內 §3.1 子表 L2453 → L2455；(d) [Full-Market Restricted Governance Exception] L2430 → L2432；(e) [Core Full-History Mode] L2431 → L2433；(f) 維運矩陣場景索引 8 行（4D-8）行號 +2 校準；(g) 維運矩陣 sub-title 範圍 L2425-2435 → L2425, L2430-2437；(h) [Sovereignty Declaration] 補入 §14.7-AL 關聯說明（本程式 Step 4 為 hub 治權閉環必要前置；本程式不直接觸及 Step 4.5 — hub 屬 §3.2，本程式屬 §3.1）；(i) TOOL_VER v1.16 → v1.17；(j) 主權狀態行補入「+ §14.7-AL CROSS-REF CALIBRATION」摘要。**介面零變動**：所有 CLI flag / preflight 邏輯 / UNIVERSE_TIERS / 節流 / 退避 / FRED pagination / verdict 邏輯全保留。本補正屬「憲章演進造成下游行號偏移」之自然校準（非治權違規）；對應 CLAUDE.md §四 #4 8 項標頭強制檢驗治權慣例。 | SUPERSEDED |
| v1.16 | 2026-05-21 | Codex | **8 項標頭強制檢驗 100% 合規 + 維運矩陣重組為 8 大功能群視角 + 治權慣例 3 條入憲**：(a) 主權狀態行補入 v1.16 修補摘要 + §3.1 序列模組身分宣告；(b) 最後更新日期 2026-05-18 → 2026-05-21；(c) 核心定義 11 條 → 14 條：新增 [Zero Hardcoded Verdict] §5.6.3 動態 verdict 對齊 + [Sovereignty Declaration] §3.1 序列模組第 5/9 員身分自我宣告（8 條治權邊界 a-h）+ [Historical Reference Authority]；(d) cross-ref 精確行號補入（§二 Step 4 L2425 / Step 4D-8 L2428-2435 + §3.1 子表 L2453）；(e) 維運矩陣重組為 **8 大功能群**（A. 種子灌溉 / B. 個股同步 / C. Universe 階段灌溉 / D. 全市場治理例外 / E. 核心股全天數 / F. FRED 宏觀 / G. §7 三層防禦 + §7.6 進階優化 / H. Lifecycle + Verdict 動態判定）+ 對齊 §二 維運矩陣 9 個標準場景索引；(f) cosmetic：`data_schema.py v2.11` → 動態對齊當前 `v2.16`；對齊憲章 v6.0.0-FINAL；補入模組級 `CONSTITUTION_VER` + `TOOL_VER` 常數；`self.schema_ver` 更新至 v2.16；**修正 v1.11 ACTIVE → SUPERSEDED 狀態錯誤**（修訂歷程內部矛盾封閉）。介面零變動：所有 CLI flag / preflight 邏輯 / UNIVERSE_TIERS / 節流 / 退避 / FRED pagination / verdict 邏輯全保留。對應 CLAUDE.md §四 #4 8 項標頭強制檢驗治權慣例。 | SUPERSEDED |
| v1.15 | 2026-05-18 | Codex | **§6.8.7 第 (1A) 條核心股全天數補刷 + `--full-history` 別名旗標**：依憲章 v6.0.0-patch §6.8.7 第 (1A) 條新增「核心股全天數補刷」場景，引入 `--full-history` 作為 `--strict-source-history` 之直觀別名（兩旗標等價，任一啟用即觸發 §14.7-L 行為：`start_date=1990-01-01` + §7.5 L3 resume 停用）。argparse 以 `dest='strict_source_history'` 將兩旗標合併為同一狀態變數，使用者下任一即生效；help 文字明示適用場景（core/convex/research 或 --id，無需 reason；對 --universe full 仍須 reason）。同步登錄 §2 維運矩陣 Step 4F、B 子節新增 F「核心股全天數補刷」。本版不改 main() preflight 邏輯、UNIVERSE_TIERS、節流、退避、FRED pagination 或 v1.14 auto-strict 行為。 | SUPERSEDED |
| v1.14 | 2026-05-18 | Codex | **§6.8.7 第 (4) 條「全天數」定義明文化 + auto strict-source-history on full universe**：依憲章 v6.0.0-patch 對「全天數」之嚴格定義（每 (stock_id, dataset) 對自 API 來源端最早可得日期 → DB 最後交易日）落地實作：main() preflight 在 `--universe full` 觸發時 **自動啟用** `--strict-source-history`（強制 `start_date=1990-01-01` + 停用 §7.5 L3 resume），使用者毋須另下旗標；`--days` 在 full mode 下退為 safety floor，實際同步起點由 strict-source-history 決定。終端報表加 INFO 列說明此自動行為。FRED 同步維持既有 `asc + offset` 全歷史分頁，不變。本版不改 UNIVERSE_TIERS、節流、退避、reason 驗證或既有旗標語意。 | SUPERSEDED |
| v1.13 | 2026-05-18 | Codex | **§6.8.7 第 (4) 條全市場全天數限定治理例外**：依憲章 v6.0.0-patch §6.8.7 第 (4) 條修訂，原「全市場 2,798 支全抓禁止」改為限定治理例外。新增 `UNIVERSE_TIERS["full"] = (core, convex, research, quarantine)`；新增 `--special-full-market-reason <reason>` argparse + preflight 驗證（缺 reason 或 reason < 12 字即 exit 1）；reason 寫入 `record_lifecycle` context 與終端報表「special override」列；__main__ 加治權檢查 — `--universe full` 時必須附 reason，反之 reason 不得用於非 full universe；§7.6 A1+A2 平行與 A3 動態配額之必要性提示（不強制 enforce，避免破壞既有 dry-run）。本版不改 schema、upsert、節流、退避、FRED pagination 或 strict-source-history 語意。 | SUPERSEDED |
| v1.12 | 2026-05-18 | Codex | **§14.7-L strict all-source-history mode**：新增 `--strict-source-history`，使 FinMind 個股同步以 `1990-01-01` 作為授權來源端探測下界，並強制停用 §7.5 L3 resume，避免 partial DB 因「存在任一 >= start_date row」被誤跳過。此模式只改變 start_date/resume 語意，不改 schema、upsert、節流、退避或 FRED pagination；應搭配 `audit_source_availability.py --strict` 產生之 mismatch 清單做 core 150 小範圍精準補刷，不得作為全市場全歷史重抓入口。 | SUPERSEDED |
| v1.11a | 2026-05-17 | Codex | **§7.6 A3/A5 治權邊界補正**：(A3 修正) `_query_remaining_quota()` 原直接呼叫 `get_user_info()` 未計入 throttle 配額，違反 §7.6 A3「查詢動作本身計入配額」邊界；補正為呼叫成功後在 `throttle.lock` 內遞增 `window` 與 `total_acquired`，避免遞迴 acquire 造成死鎖。(A5 修正) `_apply_lifecycle_verdict()` 原僅輸出 stdout，未對齊 §7.6 A5「達 4,800/hr 時 **lifecycle 寫入 warning**」；補正為當 `a5_warn_count > 0` 或 `a5_pause_count > 0` 時，即使主流程 success 也升級為 lifecycle WARNING；既有 warning/failed 分支則 append A5 訊息進入 lifecycle marker。本補正不改動 CLI、不改動既有節流邏輯，僅封閉 v1.11 對 §7.6 條文之邊界漏洞。 | SUPERSEDED |
| v1.11 | 2026-05-17 | Codex | §7.6 A1〜A5 進階優化落地版：(A1) 新增 `--dataset-batched` 改外層迴圈優先 dataset，降低單批請求量；(A2) 新增 `--workers N` 平行 worker，共用 thread-safe `FinMindThrottle` (`threading.Lock`)；(A3) 新增 `--dynamic-quota` 與 `--quota-interval N` (N≥100)，每 N 次請求查 FinMind 帳號 API 動態調整節流上限；(A4) `FinMindThrottle` 新增 per-dataset 滑動窗統計，引擎結束時自動寫入 `data_audit_log` op_type=`QUOTA_HOURLY_SNAPSHOT`，不改動既有主鍵；(A5) 4800/hr 觸發一次性 WARN，5500/hr 觸發自動暫停 300s（次數計入 stats）。預設行為 (workers=1, dataset-batched=off, dynamic-quota=off) 完全相容 v1.10。 | SUPERSEDED |
| v1.10 | 2026-05-15 | Codex | §7 供應鏈速率主權落地版：(1) 新增 `FinMindThrottle` 滑動窗節流 5500/hr；(2) 新增 `fetch_with_retry()` 三階段退避 [30s, 300s, 1800s] 與 402 單次探測重試；(3) 新增 `is_already_synced()` DB-driven L3 斷點續傳；(4) 重寫 `sync_finmind()` 整合三層防禢，新增 `skipped` 與 `recovered` stats 類別；(5) `write_data_audit_log` op_type 擴充 `RETRY_402_RECOVERED` / `RESUME_SKIP`；(6) 連續三次失敗即 FAILED 並寫入 lifecycle；(7) CLI 不變、`--no-resume`、`--throttle` 為非破壞性新增。對齊憲法 v5.4.22 §7.1–7.8 全部條文。 | SUPERSEDED |
| v1.9 | 2026-05-14 | Auto-patch | 5.5.3 對齊版：矩陣表補滿；--all 旗標語意正式登錄；Phase-Appropriate Lookback。 | SUPERSEDED |
| v1.8 | 2026-05-14 | Codex | 市場個股資料取得與種子灌溉治理；lifecycle context 接入；動態 PERFECT/WARNING/FAILED。 | SUPERSEDED |
| v1.7 | 2026-05-13 | Auto-patch | Bug #1 修補：sync_fred() 補完 empty-data 失敗分支與 dropna 後空集合分支。 | ARCHIVED |
| v1.6 | 2026-05-13 | Antigravity | 創世圓滿：對齊憲法 v5.4.18。 | ARCHIVED |
================================================================================
"""
import argparse
import os
import re
import sys
import threading
import time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

_THIS_FILE = Path(__file__).resolve()
_INGESTION_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _INGESTION_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import (
        get_db_connection,
        record_lifecycle,
        write_data_audit_log,
        get_core_stocks_from_db,
    )
    from core.data_schema import DATASET_REGISTRY, FINMIND_API_TABLES
    from core.finmind_client import FinMindClient
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v1.20"


# v1.10 phase-aware constants inherited from v1.9
SELECTION_PHASE_DAYS = 730
UNIVERSE_TIERS = {
    "research": ("research_universe",),
    "convex": ("convex_universe",),
    "core": ("core_universe", "convex_universe"),
    # v1.13 §6.8.7 第 (4) 條限定治理例外：全市場全天數 sync
    "full": ("core_universe", "convex_universe", "research_universe", "quarantine_universe"),
}

# v1.13 §6.8.7 第 (4) 條全市場限定治理例外常數
FULL_MARKET_REASON_MIN_CHARS = 12      # special_full_market_reason 字串長度下限
FULL_MARKET_REQUIRED_UNIVERSE = "full"  # 必須 --universe full 才能 / 才需 reason

# v1.10 §7 supply chain rate sovereignty constants
DEFAULT_THROTTLE_PER_HOUR = 5500          # §7.2 主權預設 (8% 餘裕)
ABSOLUTE_THROTTLE_CEILING = 6000          # §7.2 禁止 ≥ 6000
RETRY_BACKOFFS_FULL = [30, 300, 1800]     # §7.3 三階段退避
RETRY_BACKOFF_402 = [1800]                # §7.4 402 單次探測
RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}

# v1.11 §7.6 A1〜A5 進階優化常數（憲法固定值，不得由 caller 調整）
A5_WARN_THRESHOLD = 4800                  # §7.6 A5: 80% 觸發 lifecycle warning（4800 = 6000 × 80%）
A5_PAUSE_THRESHOLD = 5500                 # §7.6 A5: 達 5500/hr 自動暫停
A5_PAUSE_DURATION = 300                   # §7.6 A5: 暫停 5 分鐘
A3_QUOTA_INTERVAL_MIN = 100               # §7.6 A3: N 不得小於 100
FRED_PAGE_LIMIT = 100000                  # FRED max page size; enough for current core daily/monthly series
STRICT_SOURCE_HISTORY_START_DATE = "1990-01-01"  # §14.7-L FinMind source availability lower bound


class FinMindThrottle:
    """§7.2 L1 事前預防：滑動窗節流（v1.11: thread-safe + §7.6 A4/A5）。
    憲法級常數，禁止調至 >= 6000/hr。
    """

    def __init__(self, max_per_hour=DEFAULT_THROTTLE_PER_HOUR,
                 quota_query_fn=None, quota_check_interval=A3_QUOTA_INTERVAL_MIN):
        if max_per_hour >= ABSOLUTE_THROTTLE_CEILING:
            raise ValueError(
                f"§7.2 違憲：throttle 上限 {max_per_hour} >= {ABSOLUTE_THROTTLE_CEILING}; "
                f"主權預設為 {DEFAULT_THROTTLE_PER_HOUR}"
            )
        if quota_query_fn is not None and quota_check_interval < A3_QUOTA_INTERVAL_MIN:
            raise ValueError(
                f"§7.6 A3 違憲：quota_check_interval={quota_check_interval} < {A3_QUOTA_INTERVAL_MIN}"
            )
        self.max = max_per_hour
        self.window = deque()
        self.lock = threading.Lock()              # v1.11 §7.6 A2 thread-safe
        self.total_acquired = 0
        self.total_throttled_sleep = 0.0
        # v1.11 §7.6 A5 主動配額預警
        self._warn_emitted = False
        self.a5_warn_count = 0
        self.a5_pause_count = 0
        self.total_pause_sleep = 0.0
        # v1.11 §7.6 A4 per-dataset 滑動窗
        self.per_dataset_window = defaultdict(deque)
        # v1.11 §7.6 A3 動態配額查詢
        self.quota_query_fn = quota_query_fn
        self.quota_check_interval = quota_check_interval
        self.dynamic_quota_adjustments = 0

    def acquire(self, dataset=None):
        """獲取一個請求 slot；若已達上限則阻塞。v1.11 thread-safe。"""
        # A3 動態配額查詢必須在 lock 之外（避免 HTTP 期間阻塞其他 worker）
        do_quota_check = False
        with self.lock:
            now = time.time()
            # 主窗口 prune
            while self.window and self.window[0] < now - 3600:
                self.window.popleft()
            # A4 per-dataset 窗口 prune
            if dataset:
                dq = self.per_dataset_window[dataset]
                while dq and dq[0] < now - 3600:
                    dq.popleft()
            window_size = len(self.window)

            # §7.6 A5 主動配額預警（4800/hr → WARN；5500/hr → 暫停 300s）
            if window_size >= A5_PAUSE_THRESHOLD:
                self.a5_pause_count += 1
                print(f"⏸  §7.6 A5 自動暫停：window={window_size}/{A5_PAUSE_THRESHOLD}，"
                      f"sleep {A5_PAUSE_DURATION}s 讓配額自然回收 (第 {self.a5_pause_count} 次)")
                time.sleep(A5_PAUSE_DURATION)
                self.total_pause_sleep += A5_PAUSE_DURATION
                # 重新 prune
                now = time.time()
                while self.window and self.window[0] < now - 3600:
                    self.window.popleft()
                window_size = len(self.window)
            elif window_size >= A5_WARN_THRESHOLD and not self._warn_emitted:
                self.a5_warn_count += 1
                self._warn_emitted = True
                print(f"⚠️  §7.6 A5 預警：window={window_size}/{A5_WARN_THRESHOLD} (80%)；"
                      f"後續可能進入 5500/hr 暫停")

            # §7.2 主節流 (原 v1.10 邏輯)
            if window_size >= self.max:
                sleep_for = 3600 - (now - self.window[0]) + 1
                print(f"⏸  §7.2 節流啟動：sleep {sleep_for:.0f}s 等待 1 小時窗口釋放 "
                      f"(目前窗內 {window_size}/{self.max})")
                time.sleep(sleep_for)
                self.total_throttled_sleep += sleep_for
                now = time.time()
                while self.window and self.window[0] < now - 3600:
                    self.window.popleft()

            self.window.append(time.time())
            self.total_acquired += 1
            if dataset:
                self.per_dataset_window[dataset].append(time.time())

            # A5 hysteresis: window 回落到 80% 以下時 reset warn flag
            if len(self.window) < A5_WARN_THRESHOLD:
                self._warn_emitted = False

            # A3 觸發判定（呼叫移到 lock 外）
            if (self.quota_query_fn is not None
                    and self.total_acquired > 0
                    and self.total_acquired % self.quota_check_interval == 0):
                do_quota_check = True

        if do_quota_check:
            try:
                remaining = self.quota_query_fn()
                if isinstance(remaining, int) and remaining >= 0:
                    with self.lock:
                        # 動態調整：若剩餘配額少，降低 throttle 上限
                        used = len(self.window)
                        suggested = max(used + remaining - 100, 100)
                        new_max = min(suggested, DEFAULT_THROTTLE_PER_HOUR)
                        if new_max != self.max:
                            print(f"📊 §7.6 A3 動態配額調整：max {self.max} → {new_max} "
                                  f"(used={used}, remaining={remaining})")
                            self.max = new_max
                            self.dynamic_quota_adjustments += 1
            except Exception as exc:
                print(f"⚠️  §7.6 A3 配額查詢失敗（忽略）：{type(exc).__name__}: {exc}")

    def per_dataset_snapshot(self):
        """v1.11 §7.6 A4：取 per-dataset 1 小時窗內請求數快照。"""
        with self.lock:
            now = time.time()
            snapshot = {}
            for ds, dq in self.per_dataset_window.items():
                while dq and dq[0] < now - 3600:
                    dq.popleft()
                if dq:
                    snapshot[ds] = len(dq)
            return snapshot


class SovereignSyncEngine:
    FRED_LIST = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]
    DEFAULT_FINMIND_DATASETS = [
        "TaiwanStockPrice",
        "TaiwanStockInstitutionalInvestorsBuySell",
        "TaiwanStockMarginPurchaseShortSale",
        "TaiwanStockPER",
    ]

    def __init__(self, throttle_per_hour=DEFAULT_THROTTLE_PER_HOUR, resume_enabled=True,
                 workers=1, dataset_batched=False, dynamic_quota=False,
                 quota_check_interval=A3_QUOTA_INTERVAL_MIN):
        self.fm_client = FinMindClient()
        self.fred_key = os.getenv("FRED_API_KEY")
        self.constitution_ver = CONSTITUTION_VER
        self.schema_ver = "v2.16"
        self.tool_ver = TOOL_VER
        # v1.11 §7.6 A3 動態配額查詢 callback
        quota_fn = self._query_remaining_quota if dynamic_quota else None
        self.throttle = FinMindThrottle(
            max_per_hour=throttle_per_hour,
            quota_query_fn=quota_fn,
            quota_check_interval=quota_check_interval,
        )
        self.resume_enabled = resume_enabled
        # v1.11 §7.6 A1 / A2 旗標
        self.dataset_batched = dataset_batched
        self.workers = max(1, int(workers))
        self.dynamic_quota = dynamic_quota
        self.stats_lock = threading.Lock()  # v1.11 §7.6 A2 thread-safe stats
        self.stats = {
            "success": 0,
            "warning": 0,
            "failed": 0,
            "skipped": 0,         # v1.10: L3 斷點續傳跳過
            "recovered_402": 0,   # v1.10: 402 探測重試成功
            "rows": 0,
            "details": [],
        }

    # ---------- v1.11 §7.6 A3 動態配額查詢 callback ----------

    def _query_remaining_quota(self):
        """A3: 回傳 FinMind 帳號剩餘小時配額；此查詢本身計入配額（憲法 §7.6 A3 邊界）。
        若 API 不提供 remaining 欄位則回傳 None，不調整 throttle。"""
        try:
            info = self.fm_client.get_user_info()
        except Exception:
            return None
        # §7.6 A3 邊界：查詢動作本身計入配額（憲法強制）
        # 直接遞增 throttle window 與計數器，避免遞迴呼叫 acquire 造成死鎖
        with self.throttle.lock:
            self.throttle.window.append(time.time())
            self.throttle.total_acquired += 1
        # FinMind get_user_info 回應結構：{"msg": "...", "user_count": N, "api_request_limit": M, ...}
        # 不同版本可能用不同欄位名；嘗試幾個常見鍵
        if not isinstance(info, dict):
            return None
        for key in ("api_request_limit", "remaining", "request_remaining", "quota_remaining"):
            v = info.get(key)
            if isinstance(v, (int, float)):
                return int(v)
        return None

    # ---------- detail / error helpers ----------

    def _detail(self, status, message):
        # v1.11: thread-safe (§7.6 A2)
        icon_map = {
            "success": "✅",
            "warning": "⚠️",
            "failed": "❌",
            "skipped": "⏭ ",
            "recovered_402": "♻️",
        }
        with self.stats_lock:
            if status in self.stats:
                self.stats[status] += 1
            icon = icon_map.get(status, "•")
            self.stats["details"].append(f"{icon} {message}")

    def _add_rows(self, n):
        # v1.11: thread-safe rows counter
        with self.stats_lock:
            self.stats["rows"] += n

    def _safe_error(self, exc):
        message = f"{type(exc).__name__}: {exc}"
        return re.sub(r"([?&]token=)[^&\s]+", r"\1<redacted>", message)

    # ---------- v1.10 §7.5 L3 checkpoint ----------

    def is_already_synced(self, stock_id, dataset_name, start_date):
        """
        §7.5 DB-driven L3 斷點續傳。
        若 (dataset_name) 表內存在 stock_id 且 date >= start_date 的紀錄，
        視為「該 (stock_id, dataset, >= start_date) 已同步」並回傳 True。
        對 TaiwanStockInfo 等市場級或無 stock_id 表，不啟用 checkpoint。
        """
        if not self.resume_enabled:
            return False
        if not stock_id:
            return False
        config = DATASET_REGISTRY.get(dataset_name)
        if not config:
            return False
        columns = config.get("columns", {})
        if "stock_id" not in columns or "date" not in columns:
            return False

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{dataset_name}" '
                f'WHERE "stock_id" = %s AND "date" >= %s LIMIT 1',
                (stock_id, start_date),
            )
            row = cur.fetchone()
            cur.close()
            return row is not None
        except Exception:
            return False
        finally:
            conn.close()

    # ---------- v1.10 §7.2-7.4 retry / throttle wrapper ----------

    def fetch_finmind(self, params):
        """
        §7.2 / §7.4 統一進場：節流 → 請求 → 狀態碼分流。
        回傳 (payload, recovered_402_flag)；任何最終失敗即拋例外。
        v1.11: 將 dataset 名稱傳給 throttle 以支援 §7.6 A4 per-dataset 統計。
        """
        url = self.fm_client.api_url
        headers = self.fm_client.headers
        ds_label = params.get("dataset") if isinstance(params, dict) else None

        # 402 與 403 走不同 backoff 軌道，但同一次呼叫中可能先遇 200，後遇 403
        backoff_403 = list(RETRY_BACKOFFS_FULL)  # [30, 300, 1800]
        backoff_402 = list(RETRY_BACKOFF_402)    # [1800]
        recovered_402 = False
        last_status = None

        while True:
            self.throttle.acquire(dataset=ds_label)
            try:
                res = requests.get(url, params=params, headers=headers, timeout=30)
            except (requests.Timeout, requests.ConnectionError) as exc:
                # 視同 5xx，走 backoff_403 軌道
                if not backoff_403:
                    raise
                wait = backoff_403.pop(0)
                print(f"⏱ {exc.__class__.__name__}; sleep {wait}s 後重試")
                time.sleep(wait)
                continue

            last_status = res.status_code
            if res.status_code == 200:
                payload = res.json()
                # FinMind 應用層錯誤（msg != success）
                msg = payload.get("msg")
                if msg not in (None, "success", ""):
                    raise RuntimeError(f"FinMind app-level error: {msg}")
                return payload, recovered_402

            if res.status_code == 402:
                if not backoff_402:
                    raise requests.HTTPError(f"402 Payment Required (permanent after probe): {res.text[:200]}")
                wait = backoff_402.pop(0)
                print(f"⚠ HTTP 402 探測重試：sleep {wait}s（§7.4 單次探測）")
                time.sleep(wait)
                recovered_402 = True  # 若下次成功則標註為 recovered
                continue

            if res.status_code in RETRYABLE_STATUS_CODES:
                if not backoff_403:
                    raise requests.HTTPError(f"{res.status_code} after {len(RETRY_BACKOFFS_FULL)} retries")
                wait = backoff_403.pop(0)
                print(f"⚠ HTTP {res.status_code}; §7.3 退避 sleep {wait}s")
                time.sleep(wait)
                continue

            # 其他 4xx → 立即拋
            res.raise_for_status()
            return None, recovered_402  # unreachable but for linter

    # ---------- schema / DB helpers (v1.10; schema contract inherited from v1.9) ----------

    def _convert_type(self, series, sql_type):
        if sql_type.startswith("DATE"):
            return pd.to_datetime(series, errors="coerce").dt.date
        if sql_type.startswith("TIMESTAMP"):
            return pd.to_datetime(series, errors="coerce")
        if sql_type.startswith("NUMERIC") or sql_type.startswith("INTEGER") or sql_type.startswith("BIGINT"):
            return pd.to_numeric(series, errors="coerce")
        return series.where(series.notna(), None)

    def _align_to_schema(self, table_name, df):
        config = DATASET_REGISTRY.get(table_name)
        if not config:
            raise ValueError(f"憲章未定義表名: {table_name}")

        expected_cols = list(config["columns"].keys())
        actual_cols = list(df.columns)
        missing = [col for col in expected_cols if col not in actual_cols]
        extra = [col for col in actual_cols if col not in expected_cols]
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing={missing}")
            if extra:
                problems.append(f"extra={extra}")
            raise ValueError(f"{table_name} API/schema 欄位不一致: {'; '.join(problems)}")

        aligned = df[expected_cols].copy()
        for col, sql_type in config["columns"].items():
            aligned[col] = self._convert_type(aligned[col], sql_type)

        unique_cols = config.get("unique_constraints", [])
        if unique_cols:
            before = len(aligned)
            aligned = aligned.dropna(subset=unique_cols)
            dropped = before - len(aligned)
            if dropped:
                self._detail("warning", f"{table_name}: 已丟棄 {dropped} 筆 unique key 欄位為空的資料")

        aligned = aligned.where(pd.notnull(aligned), None)
        if aligned.empty:
            raise ValueError(f"{table_name} 清洗後資料為空")
        return aligned

    def _db_value(self, value):
        if pd.isna(value):
            return None
        return value

    def _upsert_to_db(self, table_name, df):
        if df.empty:
            raise ValueError(f"{table_name} 無可寫入資料")

        config = DATASET_REGISTRY[table_name]
        unique_cols = config.get("unique_constraints", [])
        if not unique_cols:
            raise ValueError(f"{table_name} 未定義 unique_constraints，禁止 upsert")

        cols = list(df.columns)
        quoted_cols = [f'"{col}"' for col in cols]
        placeholders = ", ".join(["%s"] * len(cols))
        conflict_cols = ", ".join([f'"{col}"' for col in unique_cols])
        update_cols = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in cols if col not in unique_cols])
        update_clause = "DO NOTHING" if not update_cols else f"DO UPDATE SET {update_cols}"

        sql = f'''
            INSERT INTO "{table_name}" ({", ".join(quoted_cols)})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_cols})
            {update_clause}
        '''

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            data = [
                tuple(self._db_value(value) for value in row)
                for row in df.itertuples(index=False, name=None)
            ]
            cur.executemany(sql, data)
            conn.commit()
            rows = len(df)
            audit_date = df.iloc[0].get("date", datetime.now().date()) if hasattr(df.iloc[0], "get") else datetime.now().date()
            write_data_audit_log(table_name, "SYNC", audit_date, "UPSERT", rows)
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    # ---------- v1.10 重寫的 sync_finmind ----------

    def sync_finmind(self, stock_id, dataset_name, start_date):
        # L3 §7.5 斷點續傳：先看 DB 是否已有資料
        if self.is_already_synced(stock_id, dataset_name, start_date):
            self._detail("skipped", f"{dataset_name} ({stock_id}): DB 已有 ≥ {start_date} 資料 (§7.5 resume)")
            try:
                write_data_audit_log(dataset_name, "SYNC", datetime.now().date(), "RESUME_SKIP", 0)
            except Exception:
                pass
            return

        try:
            print(f"📡 正在獲取 FinMind: {stock_id} / {dataset_name}...")
            params = {"dataset": dataset_name, "start_date": start_date}
            if stock_id is not None:
                params["data_id"] = stock_id
            if self.fm_client.token:
                params["token"] = self.fm_client.token

            payload, recovered_402 = self.fetch_finmind(params)
            data = payload.get("data", [])
            if not data:
                self._detail("warning", f"{dataset_name} ({stock_id or 'MARKET'}): API 回傳 0 筆")
                return

            df = self._align_to_schema(dataset_name, pd.DataFrame(data))
            rows = self._upsert_to_db(dataset_name, df)
            self._add_rows(rows)

            if recovered_402:
                # §7.4: 寫入 audit log 標籤
                try:
                    audit_date = df.iloc[0].get("date", datetime.now().date()) if hasattr(df.iloc[0], "get") else datetime.now().date()
                    write_data_audit_log(dataset_name, "SYNC", audit_date, "RETRY_402_RECOVERED", rows)
                except Exception:
                    pass
                self._detail("recovered_402",
                             f"{dataset_name} ({stock_id or 'MARKET'}): {rows} 筆 UPSERT 成功（402-recovered）")
                # 仍計入 success
                self.stats["success"] += 1
            else:
                self._detail("success", f"{dataset_name} ({stock_id or 'MARKET'}): {rows} 筆 UPSERT 成功")
        except Exception as exc:
            self._detail("failed", f"{dataset_name} ({stock_id or 'MARKET'}) 失敗: {self._safe_error(exc)}")

    def sync_fred(self, series_id):
        try:
            print(f"📡 正在獲取 FRED: {series_id}...")
            if not self.fred_key:
                raise RuntimeError("FRED_API_KEY missing")
            url = "https://api.stlouisfed.org/fred/series/observations"
            data = []
            offset = 0
            while True:
                params = {
                    "series_id": series_id,
                    "api_key": self.fred_key,
                    "file_type": "json",
                    "limit": FRED_PAGE_LIMIT,
                    "offset": offset,
                    "sort_order": "asc",
                }
                res = requests.get(url, params=params, timeout=30)
                res.raise_for_status()
                payload = res.json()
                page = payload.get("observations", [])
                data.extend(page)
                if len(page) < FRED_PAGE_LIMIT:
                    break
                offset += FRED_PAGE_LIMIT
            if not data:
                raise RuntimeError("API 回傳 0 個 observation")

            df = pd.DataFrame(data)
            df["series_id"] = series_id
            df = self._align_to_schema("FredData", df)
            df = df.dropna(subset=["value"])
            if df.empty:
                raise ValueError("全部 observation 在數據聖潔清洗後為空")

            rows = self._upsert_to_db("FredData", df)
            self._add_rows(rows)
            self._detail("success", f"FRED/{series_id}: {rows} 筆 UPSERT 成功")
        except Exception as exc:
            self._detail("failed", f"FRED/{series_id} 失敗: {self._safe_error(exc)}")

    # ---------- universe / dataset resolution ----------

    def _resolve_stocks(self, stock_id, universe):
        if stock_id:
            return [stock_id]
        if universe in UNIVERSE_TIERS:
            try:
                stocks = get_core_stocks_from_db(tiers=UNIVERSE_TIERS[universe])
            except Exception as exc:
                self._detail("failed", f"{universe} universe 讀取失敗: {type(exc).__name__}: {exc}")
                return []
            if not stocks:
                self._detail("warning", f"{universe} universe 無標的")
            return stocks
        return []

    def _target_datasets(self, dataset, all_datasets):
        if dataset:
            return [dataset]
        if all_datasets:
            return [name for name in FINMIND_API_TABLES if name != "TaiwanStockInfo"]
        return self.DEFAULT_FINMIND_DATASETS

    def _apply_lifecycle_verdict(self, lifecycle):
        if lifecycle is None:
            return
        # §7.6 A5：達 4800/hr 預警次數或 5500/hr 自動暫停次數任一 > 0 → lifecycle WARN
        a5_msgs = []
        if self.throttle.a5_warn_count > 0:
            a5_msgs.append(f"§7.6 A5 預警觸發 {self.throttle.a5_warn_count} 次 (≥4800/hr)")
        if self.throttle.a5_pause_count > 0:
            a5_msgs.append(f"§7.6 A5 自動暫停 {self.throttle.a5_pause_count} 次 (≥5500/hr, 累計 {self.throttle.total_pause_sleep:.0f}s)")

        if self.stats["failed"] > 0 and hasattr(lifecycle, "mark_failed"):
            lifecycle.mark_failed("; ".join(self.stats["details"][:5]))
        elif self.stats["warning"] > 0 and hasattr(lifecycle, "mark_warning"):
            lifecycle.mark_warning("; ".join(self.stats["details"][:5] + a5_msgs))
        elif a5_msgs and hasattr(lifecycle, "mark_warning"):
            # 即使主流程 success，A5 觸發也須升級為 WARNING 寫入 lifecycle
            lifecycle.mark_warning("; ".join(a5_msgs))

    def _phase_appropriate_hint(self, stock_id, universe, days, dataset, seed, all_datasets):
        if seed or stock_id or dataset:
            return
        if universe in UNIVERSE_TIERS and days < SELECTION_PHASE_DAYS:
            print(f"💡 [Phase Hint] 對 `--universe {universe}` 使用 `--days {days}` 只取增量。")
            print(f"   選股 phase 建議 `--days {SELECTION_PHASE_DAYS}`（~2 年），對應憲章 6.4 三項 coverage。")

    def _iter_sync_pairs(self, stocks, datasets):
        """v1.11 §7.6 A1: dataset-batched=True 時改外層 dataset、內層 stock。
        預設順序保留 v1.10 行為（外層 stock）。"""
        if self.dataset_batched:
            for ds in datasets:
                for sid in stocks:
                    yield sid, ds
        else:
            for sid in stocks:
                for ds in datasets:
                    yield sid, ds

    def _execute_pairs(self, pairs, start_date):
        """v1.11 §7.6 A2: workers=1 為串行（與 v1.10 完全相容）；workers>1 走 ThreadPoolExecutor。
        共用同一 FinMindThrottle，因此節流仍受 §7.2 主權保護。"""
        if self.workers <= 1:
            for sid, ds in pairs:
                self.sync_finmind(sid, ds, start_date)
            return
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="sync") as ex:
            futures = [ex.submit(self.sync_finmind, sid, ds, start_date) for sid, ds in pairs]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:
                    self._detail("failed", f"thread exception: {type(exc).__name__}: {exc}")

    def _flush_quota_audit(self):
        """v1.11 §7.6 A4: 引擎結束時將 per-dataset 一小時請求量寫入 data_audit_log。
        op_type='QUOTA_HOURLY_SNAPSHOT'；不改既有主鍵與必填欄位。"""
        snapshot = self.throttle.per_dataset_snapshot()
        if not snapshot:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        for dataset, count in sorted(snapshot.items()):
            try:
                write_data_audit_log(dataset, "SYSTEM", today, "QUOTA_HOURLY_SNAPSHOT", count)
            except Exception as exc:
                self._detail("warning", f"§7.6 A4 quota flush failed for {dataset}: {type(exc).__name__}: {exc}")

    def run(self, stock_id=None, universe=None, source=None, dataset=None, days=30, seed=False, all_datasets=False,
            strict_source_history=False, special_full_market_reason=None):
        start_time = time.time()
        if strict_source_history:
            start_date = STRICT_SOURCE_HISTORY_START_DATE
            self.resume_enabled = False
        else:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        task_name = f"sync_{source or 'all'}_{stock_id or universe or ('seed' if seed else 'macro')}"

        self._phase_appropriate_hint(stock_id, universe, days, dataset, seed, all_datasets)

        if universe == FULL_MARKET_REQUIRED_UNIVERSE and special_full_market_reason:
            # §6.8.7 第 (4) 條：full-market reason 必須留 audit trail
            print(f"⚠️ [§6.8.7 第 (4) 條] 全市場全天數例外觸發 — reason: {special_full_market_reason}")
            print(f"   完成後必須執行：audit_supply_chain.py --include-logs + audit_source_availability.py --strict")
            print(f"   並提交 reports/full_market_sync_<YYYYMMDD_HHMM>.md 實證報告")

        with record_lifecycle(task_name, category="ingestion", stock_id=stock_id or "SYSTEM") as lifecycle:
            if special_full_market_reason and hasattr(lifecycle, "mark_warning"):
                # 全市場例外即使全成功也應 lifecycle WARNING + reason
                self._detail("warning", f"§6.8.7 全市場例外 reason: {special_full_market_reason}")

            if source in (None, "finmind"):
                if seed or dataset == "TaiwanStockInfo":
                    # TaiwanStockInfo 為市場級表，單次呼叫；不走平行
                    self.sync_finmind("", "TaiwanStockInfo", start_date)

                stocks = self._resolve_stocks(stock_id, universe)
                datasets = [ds for ds in self._target_datasets(dataset, all_datasets) if ds != "TaiwanStockInfo"]
                pairs = list(self._iter_sync_pairs(stocks, datasets))
                if pairs:
                    self._execute_pairs(pairs, start_date)

            if source in (None, "fred") and not stock_id:
                # FRED 不計入 FinMind 配額；維持串行
                for series_id in self.FRED_LIST:
                    self.sync_fred(series_id)

            # v1.11 §7.6 A4: lifecycle 結束前 flush per-dataset quota snapshot
            self._flush_quota_audit()
            self._apply_lifecycle_verdict(lifecycle)
            self.report_results(start_time, days, universe, strict_source_history=strict_source_history,
                                special_full_market_reason=special_full_market_reason)

        return self.stats["failed"] == 0

    def report_results(self, start_time, days, universe, strict_source_history=False,
                       special_full_market_reason=None):
        if self.stats["failed"] > 0:
            verdict = "FAILED"
        elif self.stats["warning"] > 0:
            verdict = "WARNING"
        else:
            verdict = "PERFECT"

        if strict_source_history:
            phase_label = f"strict source history (from {STRICT_SOURCE_HISTORY_START_DATE})"
        elif days >= SELECTION_PHASE_DAYS:
            phase_label = f"選股 phase ({days} 天)"
        elif days <= 60:
            phase_label = f"增量 phase ({days} 天)"
        else:
            phase_label = f"自訂窗 ({days} 天)"

        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: 主權同步引擎執行摘要 ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{self.constitution_ver}.md (§7 對齊)")
        print(f"schema 基準 : data_schema {self.schema_ver}")
        if universe:
            print(f"執行 universe : {universe}")
        if special_full_market_reason:
            print(f"§6.8.7 special override : {special_full_market_reason}")
        print(f"執行 phase : {phase_label}")
        print(f"§7 節流統計 : acquired={self.throttle.total_acquired}, throttle_sleep={self.throttle.total_throttled_sleep:.0f}s")
        print(f"§7 L3 續跑 : skipped={self.stats['skipped']}, 402_recovered={self.stats['recovered_402']}")
        print(f"§7.6 A2 workers={self.workers}, A1 dataset_batched={self.dataset_batched}, "
              f"A3 dynamic_quota={self.dynamic_quota} (adjustments={self.throttle.dynamic_quota_adjustments})")
        print(f"§7.6 A5 預警次數={self.throttle.a5_warn_count}, 自動暫停次數={self.throttle.a5_pause_count}, "
              f"暫停總時長={self.throttle.total_pause_sleep:.0f}s")
        print("─" * 80)
        for detail in self.stats["details"]:
            print(detail)
        print("─" * 80)
        print(f"📈 成功同步項目 : {self.stats['success']}")
        print(f"⚠️  警告同步項目 : {self.stats['warning']}")
        print(f"❌ 失敗同步項目 : {self.stats['failed']}")
        print(f"⏭  跳過同步項目 : {self.stats['skipped']}")
        print(f"♻️  402-recovered : {self.stats['recovered_402']}")
        print(f"📝 總計寫入筆數 : {self.stats['rows']}")
        print(f"🕒 總計耗時     : {(time.time() - start_time):.2f} s")
        print(f"⚖️  主權判定     : {verdict}")
        print("🛡️" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Quantum Finance 主權同步引擎 ({TOOL_VER} — §7 速率主權 + §14.7-L strict source history + §6.8.7 第 (1A) 核心股全天數 + 第 (4) 條全市場限定例外 + §3.1 序列模組身分 + 8 大功能群矩陣)",
        epilog="選股 phase 範例：python scripts/ingestion/sovereign_sync_engine.py --universe research --all --days 730；"
               "核心 phase 範例：python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730；"
               "核心股全天數補刷 (v1.15)：python scripts/ingestion/sovereign_sync_engine.py --universe core --all --full-history --dataset-batched --workers 4；"
               "全市場全天數限定例外：python scripts/ingestion/sovereign_sync_engine.py --universe full --all "
               "--dataset-batched --workers 4 --dynamic-quota --special-full-market-reason \"DB rebuild bootstrap YYYY-MM-DD full-market irrigation\""
               "（『全天數』= 每 (stock_id, dataset) 自 API 最早可得日期 → 最新交易日）",
    )
    parser.add_argument("--id", type=str, help="指定標的 ID (如 2330)")
    parser.add_argument("--universe", type=str, choices=["research", "convex", "core", "full"],
                        help="指定標的範圍（research / convex / core / full）；"
                             "full = core+convex+research+quarantine 全 2,798 支，"
                             "屬 §6.8.7 第 (4) 條限定治理例外，必須附 --special-full-market-reason")
    parser.add_argument("--source", type=str, choices=["finmind", "fred"], help="指定數據源")
    parser.add_argument("--dataset", type=str, help="指定單一 dataset 名稱")
    parser.add_argument("--seed", action="store_true",
                        help="種子灌溉模式（同步 TaiwanStockInfo 全市場資產名冊）")
    parser.add_argument("--all", action="store_true",
                        help="使用 FINMIND_API_TABLES 全表 (除 TaiwanStockInfo) 取代 DEFAULT_FINMIND_DATASETS")
    parser.add_argument("--days", type=int, default=30,
                        help="同步天數 (預設 30；選股 phase 建議 730)")
    # v1.10 新增（非破壞性）
    parser.add_argument("--no-resume", action="store_true",
                        help="(v1.10) 停用 §7.5 L3 斷點續傳；除錯用，正式運行不建議")
    parser.add_argument("--throttle", type=int, default=DEFAULT_THROTTLE_PER_HOUR,
                        help=f"(v1.10) §7.2 節流上限/小時 (預設 {DEFAULT_THROTTLE_PER_HOUR}，禁止 ≥ {ABSOLUTE_THROTTLE_CEILING})")
    # v1.11 §7.6 A1〜A5 新增（非破壞性）
    parser.add_argument("--dataset-batched", action="store_true",
                        help="(v1.11 §7.6 A1) 改 dataset 優先迴圈，單批請求量遠低於 6000")
    parser.add_argument("--workers", type=int, default=1,
                        help="(v1.11 §7.6 A2) 平行 worker 數量，預設 1 (串行)；共用 thread-safe throttle")
    parser.add_argument("--dynamic-quota", action="store_true",
                        help="(v1.11 §7.6 A3) 每 N 次請求查 FinMind 帳號 API 動態調整節流上限")
    parser.add_argument("--quota-interval", type=int, default=A3_QUOTA_INTERVAL_MIN,
                        help=f"(v1.11 §7.6 A3) 動態配額查詢間隔；憲法下限 {A3_QUOTA_INTERVAL_MIN}")
    parser.add_argument("--strict-source-history", action="store_true",
                        help="(v1.12 §14.7-L) FinMind 個股表自 1990-01-01 補刷並自動停用 resume；用於 core 150 精準全歷史對齊")
    parser.add_argument("--full-history", dest="strict_source_history", action="store_true",
                        help="(v1.15 §6.8.7 第 (1A) 條) --strict-source-history 之直觀別名；兩旗標等價；"
                             "適用 --universe core/convex/research 或 --id <stock>，無需 reason；"
                             "對 --universe full 仍須 --special-full-market-reason")
    parser.add_argument("--special-full-market-reason", type=str, default=None,
                        help=f"(v1.13 §6.8.7 第 (4) 條) 全市場全天數 sync 之治理理由 — 必須 ≥ {FULL_MARKET_REASON_MIN_CHARS} 字元；"
                             "僅在 --universe full 時生效；缺 reason 或字數不足即 exit 1")
    args = parser.parse_args()

    # v1.13 §6.8.7 第 (4) 條 preflight 治權檢查
    if args.universe == FULL_MARKET_REQUIRED_UNIVERSE:
        reason = (args.special_full_market_reason or "").strip()
        if not reason:
            print("❌ [§6.8.7 第 (4) 條] --universe full 必須附 --special-full-market-reason \"<≥12 字理由>\"")
            print("   合法情境：DB rebuild bootstrap / Sovereign rebuild / pre-annual audit / 資料源治權變更 / 合規事件")
            sys.exit(1)
        if len(reason) < FULL_MARKET_REASON_MIN_CHARS:
            print(f"❌ [§6.8.7 第 (4) 條] --special-full-market-reason 長度 {len(reason)} < {FULL_MARKET_REASON_MIN_CHARS} 字元下限")
            sys.exit(1)
        # v1.14 §6.8.7 第 (4) 條「全天數」定義落地：--universe full 自動啟用 strict-source-history
        # 「全天數」= 每 (stock_id, dataset) 自 API 最早可得日期 → 最新交易日；不可由 --days 決定
        if not args.strict_source_history:
            args.strict_source_history = True
            print(f"ℹ️ [§6.8.7 第 (4) 條] --universe full 觸發「全天數」語意：")
            print(f"   自動啟用 --strict-source-history（start_date={STRICT_SOURCE_HISTORY_START_DATE}, "
                  f"§7.5 L3 resume 停用），每支個股自 FinMind/FRED API 最早可得日期同步至最新交易日。")
            print(f"   --days={args.days} 退為 safety floor，不限制實際同步起點。")
    elif args.special_full_market_reason:
        print(f"❌ [§6.8.7 第 (4) 條] --special-full-market-reason 僅在 --universe full 時生效；"
              f"目前 --universe={args.universe or 'None'}，拒絕執行")
        sys.exit(1)

    engine = SovereignSyncEngine(
        throttle_per_hour=args.throttle,
        resume_enabled=(not args.no_resume),
        workers=args.workers,
        dataset_batched=args.dataset_batched,
        dynamic_quota=args.dynamic_quota,
        quota_check_interval=args.quota_interval,
    )
    ok = engine.run(
        stock_id=args.id,
        universe=args.universe,
        source=args.source,
        dataset=args.dataset,
        days=args.days,
        seed=args.seed,
        all_datasets=args.all,
        strict_source_history=args.strict_source_history,
        special_full_market_reason=args.special_full_market_reason,
    )
    sys.exit(0 if ok else 1)
