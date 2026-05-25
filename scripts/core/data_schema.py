"""
data_schema.py v2.20 (Quantum Finance API-First Schema Sovereignty Edition)
================================================================================
**最後更新日期**: 2026-05-25
**主權狀態**: API CONTRACT FIRST (憲法 v6.1.0-patch 對齊 + §3.2A.J data_audit_log 5-tuple UNIQUE constraint 落地（v2.17）+ §8.5 第 9 條 Publication-date Discipline 之 PUBLICATION_DATE_STRATEGY_REGISTRY 落地（v2.18 Phase 1）+ FredData strict → transitional 追溯修正（v2.19 §14.7-BB Phase 2 dry-run 揭露）+ build_publication_date_gate() helper 升 SSOT（v2.20 Phase 3 配套;兩個 builder 共用）；維運矩陣場景齊全（含 Step 2A 離線復原）；8 項檢查面 100% 合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [API Contract First]: `--init` 進入 DDL 前必須先詢問 FinMind / FRED API，確認外部欄位契約。
2. [Absolute Case Sovereignty]: 強制執行雙引號封裝 DDL，確保物理層與 API 原始大小寫 1:1 鏡像。
3. [Defensive Architecture]: 統一字串為 VARCHAR(255)，數值為 NUMERIC(20, 6)。
4. [Hybrid Observability]: 整合 pipeline_execution_log 與詳細之終端重鑄報告。
5. [Zero Hardcoded Verdict]: 主權判定（PERFECT / WARNING / FAILED）必須依執行結果動態計算，嚴禁硬編碼。對齊憲章 §5.6.3「禁止硬編碼 PERFECT」與 §3.2 Step 2 接受標準。
6. [Sovereignty Declaration]: 本模組為憲章 §3.1 序列模組 / Raw API Schema Authority（憲章 L2440 / §6.0A L2709）；不涉及 §0.1-A 第一性原理 / §0.2-A 八二法則 / §0.3-A 康波週期 / §0.0-E.4 統合層 / §0.0-F.3 AI 協作工具規則五套禁令；不在 §0.1.1 T1/T2/T3 分層內；**為 §8.5 anti-leakage 8 條 + 第 9 條 Publication-date Discipline 提供 per-dataset SSOT metadata（PUBLICATION_DATE_STRATEGY_REGISTRY）但不執行**（執行載體為 feature_store_builder / core_universe_builder v0.4+ / audit_leakage v0.3+ rule 19）；不得承擔核心股 derived governance schema（憲章 L2440 / L2710 邊界）。
7. [Publication-date Strategy SSOT]: `PUBLICATION_DATE_STRATEGY_REGISTRY` 為每個 dataset 之 publication-date 規則 SSOT（v2.18 新增；模組級獨立 dict，對齊既有 `INFRA_TABLES` 慣例不擴張 `DATASET_REGISTRY` schema），對齊憲章 §8.5-9.2 分派表；含 5 種 enforcement：`strict`（直接用 publication-date column）/ `hardcoded_conservative`（法定截止日推算 +offset_days）/ `transitional`（暫維持 date，待研究升版）/ `native_aligned`（date = trading day 本就無 delay）/ `infrastructure`（infra 表不適用）。Builder/audit 模組透過此 SSOT 取得 SQL gate 規則，**不得自行定義 publication-date 規則**（對齊 §0.0-I 單一引用源原則）。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario)   | 權威指令 / 建議用法 (Exhaustive Examples)                             | 對齊模組 |
| :----------------------- | :-------------------------------------------------------------------- | :--- |
| **1. [重鑄：資料庫主權初始化]**       | `$ python scripts/core/data_schema.py --init --force`                              | data_schema v2.20 |
| **2. [重鑄：單一表主權重鑄]**         | `$ python scripts/core/data_schema.py --init --table [Name]`                       | data_schema v2.20 |
| **3. [離線/災難復原：略過 API 契約探測]** | `$ python scripts/core/data_schema.py --init --force --skip-api-contract` | data_schema v2.20 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v2.20** | 2026-05-25 | Codex | **`build_publication_date_gate()` helper 升至 data_schema SSOT(Phase 3 配套;helper 從 feature_store_builder v0.4 之 local def 移至 data_schema 模組級;對齊 §0.0-I 單一引用源原則)**:依憲章 §8.5-9 之 Phase 3 落地(core_universe_builder v0.3 → v0.4 之 SQL gate 升版),為避免 helper 在 feature_store_builder + core_universe_builder 重複定義(違 DRY/SSOT),將 helper 從 `feature_store_builder.py v0.4` 之 local `_publication_date_gate()` 升至 `data_schema.py` 模組級 `build_publication_date_gate()`(rename 反映 utility 性質,移除 private underscore prefix)。**設計合理性**:helper 為 `PUBLICATION_DATE_STRATEGY_REGISTRY` 之直接衍生(讀 strategy → 構造 SQL clause 字串),屬同治權位階(data_schema = Raw API Schema + metadata SSOT;helper 為 metadata 之衍生工具,**不執行 SQL,只構造字串**;未違反 data_schema [Sovereignty Declaration] 之治權邊界);對齊既有 `INFRA_TABLES` set 之模組級 utility 慣例。**補正內容**:(I) 新增模組級函式 `build_publication_date_gate(table: str, as_of_param_placeholder: str = "%s") -> tuple[str, int]`(從 feature_store_builder v0.4 之 `_publication_date_gate` 完整移入,介面零變動);(II) TOOL_VER v2.19 → v2.20;(III) 主權狀態行加「build_publication_date_gate() helper 升 SSOT(v2.20 Phase 3 配套;兩個 builder 共用)」;(IV) 維運矩陣 3 場景 cosmetic v2.19 → v2.20。**邏輯動量**:`PUBLICATION_DATE_STRATEGY_REGISTRY` 13 entries 不變(v2.19 之 FRED transitional 追溯維持);`PUBLICATION_DATE_ENFORCEMENT_TYPES` 5 種不變;13 張 DATASET_REGISTRY 不變;API contract probe / CLI / verdict 邏輯不變;helper 介面與行為與 feature_store_builder v0.4 之 local 版本**完全一致**(僅位置移轉)。**對下游 builder 影響**:`feature_store_builder v0.5` 刪 local def 改 `from core.data_schema import build_publication_date_gate`;`core_universe_builder v0.4` 加同 import,12 處 SQL gate 透過 helper 升版;**兩個 builder 共用單一 SSOT helper**。**對既有 DB / snapshot 影響**:**零**(本版純為 helper 位置移轉,無 DDL 變更;builder 升版另案落地)。本版**不**修改 DATASET_REGISTRY 任何內容、**不**改 PUBLICATION_DATE_STRATEGY_REGISTRY、**不**改 PUBLICATION_DATE_ENFORCEMENT_TYPES、**不**改 INFRA_TABLES、**不**改 API contract probe / DDL / CLI / verdict、**不**修改既有任何函式介面(僅新增 1 個 module-level helper)。同步配套:`feature_store_builder.py v0.4 → v0.5`(刪 local;import)+ `core_universe_builder.py v0.3 → v0.4`(import + SQL gate 升版,Phase 3 落地)。 | **ACTIVE** |
| v2.19 | 2026-05-25 | Codex | **PUBLICATION_DATE_STRATEGY_REGISTRY['FredData'] 追溯修正 strict → transitional(憲章 §14.7-BB Phase 2 dry-run 揭露驅動;v6.1.0-patch 第五輪 charter 同次配套程式)**:依憲章 v6.1.0-patch §14.7-BB(2026-05-25 夜深入憲),基於 Phase 2 dry-run 實證:5 個 historical as_of_date × 4 FRED series 之 strict vintage gate(`realtime_start <= as_of_date`)**100% loss**。**Root cause**:DB 內 `FredData.realtime_start` 為 ingest 日期(2026-05-21~22 從零重建)非真實 ALFRED vintage。**追溯修正 PUBLICATION_DATE_STRATEGY_REGISTRY['FredData']**:(I) enforcement: `strict` → **`transitional`**(對齊 Shareholding 過渡分類);(II) source: `fred_vintage_start` → **`fred_vintage_pending_alfred`**;(III) column: `realtime_start` → **`date`**(暫維持,同 Shareholding);(IV) description 補入 Phase 2 dry-run 揭露;(V) TOOL_VER v2.18 → v2.19;(VI) 主權狀態行加「FredData strict → transitional 追溯修正(v2.19 §14.7-BB)」;(VII) 維運矩陣 cosmetic v2.18 → v2.19。**邏輯動量**:13 張 DATASET_REGISTRY 不變;PUBLICATION_DATE_STRATEGY_REGISTRY 13 entries 不變(僅 FredData 單筆內容追溯);PUBLICATION_DATE_ENFORCEMENT_TYPES 5 種不變;API contract probe / CLI / verdict 邏輯不變。**對既有 DB 影響**:**零**(本版純為 metadata 註冊追溯,無 DDL 變更)。**對下游 builder/audit 影響(配套 Phase 2 落地之 feature_store_builder v0.4)**:讀此 metadata 時 FRED 走 transitional 路徑(SQL gate 維持 `date <= as_of_date`),與其他 native_aligned 表行為相同。**「資料現實裁決」第三次跑通**(對映 §14.7-AX):2026-05-24 §0.1.3-A.1 ROE → 2026-05-25 §14.7-BA 5 欄位 → **2026-05-25 §14.7-BB FRED**。**追溯適用**:既有 audit / feature_set / model / prediction **皆不受影響**(metadata-only;v2.18 入憲後 builder 尚未實際讀取 strategy,追溯無 builder 副作用)。本版**不**修改 DATASET_REGISTRY 任何內容、**不**改其他 12 表之 publication_date_strategy(僅 FredData 追溯)、**不**改 PUBLICATION_DATE_ENFORCEMENT_TYPES、**不**改 INFRA_TABLES、**不**改 API contract probe / DDL / CLI / verdict。同步入憲:憲章 §14.7-BB(新建子節) / §8.5-9 4 處追溯修正 / 修訂歷程 v6.1.0-patch 2026-05-25 第五輪 entry。 | SUPERSEDED |
| v2.18 | 2026-05-25 | Codex | **§8.5 第 9 條 Publication-date Discipline Phase 1 落地（v6.1.0-patch §8.5-9 / §14.7-BA 程式預備升版 A;commit `669b5c8` 入憲後第一支程式落地）**：依憲章 v6.1.0-patch（commit `669b5c8`,2026-05-25 夜）新入憲之 §8.5 第 9 條 Publication-date Discipline 強制契約 + §8.5-9 子節 8 個子子節 + §14.7-BA 治權閉環,本版落地 Phase 1（per §8.5-9.7 升版觸發表）:**PUBLICATION_DATE_STRATEGY_REGISTRY 模組級獨立 dict,對齊 §8.5-9.2 分派表（11 業務+metadata 表 + 2 infra 表 = 13）**。**5 種 enforcement 類別**:(1) `strict`（直接用 publication-date column）:`TaiwanStockDividend.AnnouncementDate` / `FredData.realtime_start`;(2) `hardcoded_conservative`（法定截止日推算）:`TaiwanStockMonthRevenue date + 10 天` / `TaiwanStockFinancialStatements date + 45 天 (Q1-Q3) / 90 天 (Q4)`;(3) `transitional`（暫維持 `date`,待研究升版）:`TaiwanStockShareholding`（`RecentlyDeclareDate` 語意不明,§14.7-BA §3.3 揭露;D2.1 研究方向）;(4) `native_aligned`（`date` = trading day,本就無 publication delay）:`TaiwanStockPrice / PriceAdj / PER / Margin / InstitutionalInvestorsBuySell / TaiwanStockInfo`(InstitutionalInvestorsBuySell 之 T 日 17:30 cron 對齊已於 §6.8.7-A 涵蓋);(5) `infrastructure`（infra 觀測表不適用 publication-date）:`pipeline_execution_log` / `data_audit_log`。**補正內容**:(I) `PUBLICATION_DATE_STRATEGY_REGISTRY` 模組級獨立 dict 涵蓋 13 表項目(source / column / offset_days / enforcement / description 5 鍵);(II) 核心定義新增第 7 條 [Publication-date Strategy SSOT] 顯式宣告 PUBLICATION_DATE_STRATEGY_REGISTRY 為憲章 §8.5-9 之 per-dataset metadata SSOT(對齊 §0.0-I 單一引用源原則);(III) 核心定義第 6 條 [Sovereignty Declaration] 補入「為 §8.5 anti-leakage 8 條 + 第 9 條提供 SSOT metadata 但不執行」之治權邊界明示;(IV) TOOL_VER v2.17 → v2.18;(V) 主權狀態行加「§8.5 第 9 條 Publication-date Discipline 之 PUBLICATION_DATE_STRATEGY_REGISTRY 落地（v2.18 Phase 1）」;(VI) CONSTITUTION_VER 維持 v6.1.0(主版號,patch 為修訂歷程記述);(VII) 維運矩陣 3 場景之 cosmetic v2.17 → v2.18。**邏輯動量**:13 張 DATASET_REGISTRY 數量不變;12 業務 + infra 表之 columns / unique_constraints / DDL 邏輯不變;API contract probe 邏輯不變;--init / --force / --table / --skip-api-contract CLI 介面不變;verdict 動態計算邏輯不變;§5.6.3 + §0.4 + §0.0-G + §0.0-I 全部不違反。**對既有 DB 影響**:**零**(本版純為 metadata 註冊,無 DDL 變更);既有 snapshot / feature_set / model / prediction 全部不受影響;`--init --force` 之 DROP + CREATE 行為不變。**對下游 builder/audit 影響(待 Phase 2-5)**:`feature_store_builder v0.4` 升 SQL 時可讀此 metadata 取得 effective_publication_date gate 規則;`audit_leakage v0.3` 之 rule 19 publication_date_check 透過此 metadata 驗收。**Phase 1 完成 → Phase 2 預備**:feature_store_builder v0.3 → v0.4 之 SQL gate 升版(讀 PUBLICATION_DATE_STRATEGY_REGISTRY[t])。**追溯適用**:既有 audit 報告 / feature_set v0.1-v0.3 / model_registry / prediction_run **皆不受影響**(本版為 metadata-only 升版;builder 邏輯升 v0.4 後新 snapshot 起適用)。本版**不**修改其他 12 表之 columns / unique_constraints、**不**改 `pipeline_execution_log` / `data_audit_log` DDL、**不**改 13 表之 API contract probe 邏輯、**不**改 CLI 介面、**不**改 verdict 計算邏輯。同步入憲:憲章 §8.5 第 9 條(L4292) / §8.5-9 子節 8 個子子節(L4296-4392) / §14.7-BA(L7844-7928) / 修訂歷程 v6.1.0-patch 2026-05-25 第三輪 entry(L66);程式落地序列 §8.5-9.7 之 Phase 1(本版)。 | SUPERSEDED |
| v2.17 | 2026-05-25 | Codex | **§3.2A.J `data_audit_log` 5-tuple UNIQUE constraint 落地（v6.1.0-patch §3.2A.J / §14.7-AY 程式預備升版 A）**：依憲章 v6.1.0-patch（commit `4da2450`，2026-05-25）新入憲之 §3.2A.J `db_utils.write_data_audit_log` Audit Log Write-Safe 治權契約（憲章 L2722-2745）+ §14.7-AY §7.4-A 姊妹缺陷補完入憲（憲章 L7480-7568），本版落地裁決第 1 條「`data_schema.py v2.16 → v2.17`：`data_audit_log.unique_constraints` 從 `[]` 改為 5-tuple」。**Root cause（2026-05-24 Audit 2 揭露）**：Step 4F 啟動 ~65 秒兩個 sync_engine worker 並發寫入 `data_audit_log` 撞同 microsecond + 同 5-tuple → race-induced duplicate row 1 個 → Audit 2 verdict=FAILED（業務 dataset 全 PASS，唯一 FAIL 在 infra 觀測表）。**補正內容**：(I) `DATASET_REGISTRY["data_audit_log"]["unique_constraints"]`: `[]` → `["table_name", "stock_id", "data_date", "action_type", "timestamp"]`（5-tuple，timestamp 為 microsecond 精度作 race boundary）；(II) TOOL_VER v2.16 → v2.17；(III) CONSTITUTION_VER v6.0.0 → v6.1.0（對齊現行憲章 v6.1.0-patch）；(IV) 主權狀態行加「§3.2A.J data_audit_log 5-tuple UNIQUE constraint 落地」；(V) 維運矩陣 3 場景之 cosmetic v2.16 → v2.17；(VI) report header v2.16 → v2.17。**邏輯動量**：13 張 DATASET_REGISTRY 數量不變；其他 12 表 unique_constraints 不變；API contract probe 邏輯不變；--init / --force / --table / --skip-api-contract CLI 介面不變；verdict 動態計算邏輯不變；§5.6.3 + §0.4 + §0.0-G + §0.0-I 全部不違反。**對既有 DB 影響**：既存 `data_audit_log` 表若有 race-induced dup（如 2026-05-24 從零驗證留下之 1 個 dup），`--init --force` DROP + CREATE 會清空；非 force 模式需透過 `scripts/maintenance/migrate_data_audit_log_dedup_20260525.py v0.1`（同次入憲之 §14.7-AY 落地 C 項）執行 dedup + ALTER TABLE ADD CONSTRAINT。**db_utils.py 配套升版**：v2.47 → v2.48 之 `write_data_audit_log()` 加 ON CONFLICT DO NOTHING（落地裁決第 2 條，另一支同次升版）。**追溯適用**：v0.1-v0.6 之 `audit_api_schema_compliance` Layer F 對 `data_audit_log` 之 dup>0 記錄重新詮釋為 race-induced artifact；v2.17 / db_utils v2.48 + migration 落地後自動消解。本版**不**修改其他 12 張 dataset 之 unique_constraints、**不**改 `pipeline_execution_log` DDL（其 SERIAL id 自然 unique 不需新約束）、**不**擴張至業務 dataset（已有業務鍵 UNIQUE）。同步入憲：憲章 §3.2A.J（L2722-2745）/ §14.7-AY（L7480-7568）/ 修訂歷程 v6.1.0-patch entry（L66）。 | SUPERSEDED |
| v2.16 | 2026-05-21 | Codex | **維運矩陣補入 Step 2A「離線/災難復原」場景（CLAUDE.md §四 #4「8 項標頭強制檢驗」第 5 項首例實證；達 100% 合規）**：依昨日剛入憲之 CLAUDE.md §四 #4「完整度評估必須先檢驗標頭 docstring」治權原則第 5 項「全量維運指令總矩陣場景齊全」之檢驗，揭露 v2.15 維運矩陣只列 2 場景（`--init --force` / `--init --table`），但 CLI 實際支援 4 個 flag（含 `--skip-api-contract` 之離線/災難復原 = 憲章 §二 Step 2A）— 矩陣未對齊治權現況。**補正內容**：(I) 維運矩陣新增第 3 場景「[離線/災難復原：略過 API 契約探測]」對應 CLI `--init --force --skip-api-contract`；(II) 主權狀態行升至「(憲法 v6.0.0 對齊 + 維運矩陣場景齊全（含 Step 2A 離線復原）；8 項檢查面 100% 合規)」；(III) TOOL_VER v2.15 → v2.16；(IV) 維運矩陣 3 場景之 cosmetic v2.15 → v2.16；(V) report header v2.15 → v2.16。**API、DDL、CLI 介面（4 flag 不變）、`probe_api_contracts()` 邏輯、13 張 DATASET_REGISTRY、`init_tables(skip_api_contract=False)` 已有之 `--skip-api-contract` 處理邏輯 L309/L314（v2.10 既有）、verdict 動態計算、所有公開行為皆無變更**；本補正純為標頭維運矩陣完整化（對齊 CLAUDE.md §四 #4 第 5 項）。合規度：v2.15 ≈98%（缺 Step 2A 場景）→ v2.16 100%。 | SUPERSEDED |
| v2.15 | 2026-05-20 | Codex | **DDL hotfix：TaiwanStockMonthRevenue.create_time TIMESTAMP → DATE（對齊 API 物理本質；對應憲章 §14.7-AK 補登）**：依 `audit_api_schema_compliance v0.1` 首次實測揭露之 Layer B FAIL（commit `608c5e8` / §14.7-AJ Step 3 實證）：DDL 宣告 `create_time TIMESTAMP`，但 FinMind API 實際回傳 10 字元 DATE 字串（如 `'2026-04-21'`），不符 19 字元 `YYYY-MM-DD HH:MM:SS` 格式。實質上 `TaiwanStockMonthRevenue` 為月度資料，`create_time` 物理本質即為「資料發佈日」，DATE 比 TIMESTAMP 更精確。**裁決**：依使用者「audit 工具須嚴格、不得包容違規」之治權原則，採甲案修 DDL 對齊 API 物理本質，保留 audit 嚴格性。**補正內容**：(I) `DATASET_REGISTRY["TaiwanStockMonthRevenue"]["columns"]["create_time"]`: `"TIMESTAMP"` → `"DATE"`；(II) TOOL_VER v2.14 → v2.15；(III) 標頭主權狀態加「v2.15 hotfix：TaiwanStockMonthRevenue.create_time TIMESTAMP → DATE 對齊 API 物理本質」；(IV) 維運矩陣 / report header v2.14 → v2.15。**邏輯動量**：API contract probe 邏輯不變（仍比對 column name set，DATE 之 API 回傳值原本即合法）；對既有 DB 影響——既存 rows（被 PostgreSQL 自動補 `00:00:00` 之 TIMESTAMP）需透過 `--init --force` 重建（DROP + CREATE）或 `ALTER TABLE ALTER COLUMN create_time TYPE DATE USING create_time::DATE` 截為 DATE（資訊無損失，原本時分秒即虛值）。13 張 DATASET_REGISTRY 數量不變；__init/--force/--table/--skip-api-contract CLI 介面不變；§5.6.3 verdict 動態計算邏輯不變；§1.4 + §5.6 + §0.4 + §0.0-G + §0.0-I 全部不違反。同步入憲：憲章 §二 L2408 / §3.1 L2440 / §3.2 L2483 模組登錄版本升至 v2.15；§14.7-AK 新增 hotfix 紀錄。 | SUPERSEDED |
| v2.14 | 2026-05-20 | Codex | **[Sovereignty Declaration] 核心定義第 6 條補入（8 項檢查面 100% 合規補強；對應憲章 §14.7-AH 補登）**：依 v2.13 後 8 項檢查面審計（per_program_audit §7.5 模板）揭露之 4 項標頭治權自我宣告缺口（第 1/5/6/7 項：治權位階 Type 未明示 / 5 套禁令未明示 / T1-T3 分層未明示 / §8.5 anti-leakage 未明示），補入核心定義第 6 條 [Sovereignty Declaration] 一次性涵蓋。**補正內容**：(I) 核心定義新增第 6 條 [Sovereignty Declaration]：§3.1 序列模組 / Raw API Schema Authority（憲章 L2440 / L2709）；不涉及 §0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §0.0-F.3 五套禁令；不在 §0.1.1 T1/T2/T3 分層內；不處理 §8.5 anti-leakage；不得承擔核心股 derived governance schema（憲章 L2440 / L2710 邊界）；(II) 主權狀態行升至「API CONTRACT FIRST (憲法 v6.0.0 對齊 + [Sovereignty Declaration] 核心定義第 6 條補入；8 項檢查面 100% 合規)」；(III) TOOL_VER v2.13 → v2.14；(IV) 維運矩陣 / report header v2.13 → v2.14；(V) 修訂歷程補入標準 markdown 表格 header。**API、DDL、CLI 介面、13 張 DATASET_REGISTRY、`probe_api_contracts()` 邏輯、verdict 動態計算（L393-396）、所有公開行為皆無變更**；本補正純為標頭治權自我宣告（與 `core/__init__.py v1.15` 標頭治權對齊風格一致）。同步入憲：憲章 §二 L2408 / §3.1 L2440 / §3.2 L2483 模組登錄版本升至 v2.14；§14.7-AH 新增逐元件審計修訂紀錄。 | SUPERSEDED |
| v2.13 | 2026-05-20 | Codex | **[Zero Hardcoded Verdict] 核心定義第 5 條補入（逐元件審計 Step 1.1.2 100% 合規補強）**：依逐元件治權合規審計 Step 1.1.2 修補後再審 minor 補強建議，補入核心定義第 5 條 [Zero Hardcoded Verdict] 顯式對齊憲章 §5.6.3。**補正內容**：(I) 核心定義新增第 5 條「[Zero Hardcoded Verdict]: 主權判定（PERFECT / WARNING / FAILED）必須依執行結果動態計算，嚴禁硬編碼。對齊憲章 §5.6.3 與 §3.2 Step 2 接受標準」；(II) 主權狀態行更新為「(憲法 v6.0.0 對齊 + [Zero Hardcoded Verdict] 核心定義補入；100% 合規)」；(III) TOOL_VER v2.12 → v2.13；(IV) 維運矩陣 / report header v2.12 → v2.13。**程式邏輯（L391-393 之 verdict 計算）原已 §5.6.3 合規**，本次純為核心定義條之顯式宣告（與 `finmind_client.py v4.46` 之第 5 條 [Zero Hardcoded Verdict] 治權慣例對齊）。API、DDL、CLI 介面、13 張 DATASET_REGISTRY、所有公開行為皆無變更。 | SUPERSEDED |
| v2.12 | 2026-05-20 | Codex | **v6.0.0 標頭治權對齊 + `CONSTITUTION_VER` 模組常數補入（逐元件審計 Step 1.1.2 補正）**：依逐元件治權合規審計 Step 1.1.2 揭露之兩項違規：(1) 缺 `CONSTITUTION_VER` 模組常數（違反憲章 L26「所有 §3.1/§3.2 登錄模組之 `CONSTITUTION_VER` 同步至 v6.0.0」；先前只有 `self.constitution_ver` attribute）；(2) 修訂歷程缺 v6.0.0 升版條目（雖 self.constitution_ver = "v6.0.0" 已在 L204，但歷程未追蹤升版）；(3) L307 docstring 殘留 v5.4.22 cosmetic 字串。本次補正：(I) 新增 `CONSTITUTION_VER = "v6.0.0"` + `TOOL_VER = "v2.12"` 模組常數；(II) 補入本 v2.12 修訂條目；(III) L307 docstring「執行憲法 v5.4.22 API-first 標準」→「執行憲法 v6.0.0 API-first 標準」；(IV) 維運矩陣 / report header 之 v2.11 cosmetic → v2.12。**API Contract First 邏輯、13 張 DATASET_REGISTRY、雙引號 DDL 封裝、--init/--force/--table/--skip-api-contract CLI 與所有公開行為皆無變更**；本補正純為標頭治權對齊。 | SUPERSEDED |
| v2.11 | 2026-05-14 | Codex | **API 欄位鏡像修正**：補齊 FinMind API 實際回傳欄位大小寫；移除 `TaiwanStockPrice.spread_per` schema-only 欄位，使 API contract probe 全量通過。 | SUPERSEDED |
| v2.10 | 2026-05-14 | Codex | **API Contract First**：`--init` / `--force` 前先探測 FinMind 與 FRED API 契約；契約失敗時停止 DDL，離線復原需明示 `--skip-api-contract`。 | SUPERSEDED |
| v2.9 | 2026-05-13 | Antigravity | **創世圓滿**：對齊憲法 v5.4.18；對齊「大憲章」命名體系。 | SUPERSEDED |
| v2.8 | 2026-05-13 | Antigravity | **崩潰修復**：對齊憲法 v5.4.17；補齊 pipeline_execution_log 欄位。 | ARCHIVED |
| v2.7 | 2026-05-13 | Antigravity | **全量實證對齊**：對齊憲法 v5.4.16；確立全譜系大同步地位。 | ARCHIVED |
| v2.6 | 2026-05-13 | Antigravity | **全量大同步**：對齊憲法 v5.4.15；達成檔案治理全量收斂。 | ARCHIVED |
| v2.5 | 2026-05-13 | Antigravity | **治權完備**：對齊憲法 v5.4.14；確立創世序列與檔案累積規範。 | ARCHIVED |
| v2.4 | 2026-05-13 | Antigravity | **旗艦對齊版**：對齊憲法 v5.4.12，實作絕對大小寫主權。 | ARCHIVED |
| v2.3 | 2026-05-13 | Antigravity | **主權重鑄版**：解決創世悖論，對齊基礎設施表定義。 | ARCHIVED |
================================================================================
"""
import os, sys, time, requests
from pathlib import Path
from datetime import datetime
import argparse

# ──────────────────────────────────────────────────────────────────────────────
# 治權常數 (Constitution Constants) — v2.12 新增（憲章 L26 / Step 1.1.2 補正）
# ──────────────────────────────────────────────────────────────────────────────
CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v2.21"

# ──────────────────────────────────────────────────────────────────────────────
# §8.5-9 Publication-date Discipline Strategy Enforcement Enum
# (v2.18 新增；對齊憲章 §8.5-9.2 分派表 + §14.7-BA 治權閉環)
# ──────────────────────────────────────────────────────────────────────────────
PUBLICATION_DATE_ENFORCEMENT_TYPES = {
    "strict",                    # 直接 use publication-date column (Dividend, FRED)
    "hardcoded_conservative",    # 法定截止日推算 (MonthRevenue, FinStmt)
    "transitional",              # 暫維持 date,待研究升版 (Shareholding)
    "native_aligned",            # date = trading day,本就無 publication delay
    "infrastructure",            # infra 觀測表不適用
}

# ── 系統級架構引導 ──
_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
    from core.finmind_client import FinMindClient
except ImportError:
    print("❌ 核心組件導入失敗，請確認 db_utils.py")
    sys.exit(1)

# 🏛️ 13 張全譜數據契約註冊表 (2 infra + 10 FinMind raw + 1 FRED；1:1 API 鏡像，嚴格遵循大小寫)
DATASET_REGISTRY = {
    # --- Infrastructure (治權基礎設施 - 優先建立) ---
    "pipeline_execution_log": {
        "columns": {
            "id": "SERIAL PRIMARY KEY", "task_name": "VARCHAR(255)", 
            "category": "VARCHAR(255)", "stock_id": "VARCHAR(255)",
            "start_time": "TIMESTAMP", "end_time": "TIMESTAMP", 
            "status": "VARCHAR(255)", "duration_ms": "BIGINT", "error_msg": "TEXT"
        },
        "unique_constraints": [] # 使用 Primary Key
    },
    "data_audit_log": {
        "columns": {
            "id": "SERIAL PRIMARY KEY", "table_name": "VARCHAR(255)",
            "stock_id": "VARCHAR(255)", "data_date": "DATE",
            "action_type": "VARCHAR(255)", "rows_affected": "INTEGER",
            "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        },
        # §3.2A.J data_audit_log Write-Safe (v2.17 / 2026-05-25): 5-tuple UNIQUE constraint 阻擋
        # multi-worker race-induced dup; timestamp microsecond 精度作 race boundary;
        # 配套 db_utils.write_data_audit_log() v2.48 之 ON CONFLICT DO NOTHING (憲章 §3.2A.J / §14.7-AY)
        "unique_constraints": ["table_name", "stock_id", "data_date", "action_type", "timestamp"]
    },

    # --- Technical (技術面) ---
    "TaiwanStockPrice": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "Trading_Volume": "NUMERIC(20, 6)", "Trading_money": "NUMERIC(20, 6)",
            "open": "NUMERIC(20, 6)", "max": "NUMERIC(20, 6)", "min": "NUMERIC(20, 6)", "close": "NUMERIC(20, 6)",
            "spread": "NUMERIC(20, 6)", "Trading_turnover": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    "TaiwanStockPriceAdj": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "Trading_Volume": "NUMERIC(20, 6)", "Trading_money": "NUMERIC(20, 6)",
            "open": "NUMERIC(20, 6)", "max": "NUMERIC(20, 6)", "min": "NUMERIC(20, 6)", "close": "NUMERIC(20, 6)",
            "spread": "NUMERIC(20, 6)", "Trading_turnover": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    "TaiwanStockPER": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "dividend_yield": "NUMERIC(20, 6)", "PER": "NUMERIC(20, 6)", "PBR": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    
    # --- Chip (籌碼面) ---
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "buy": "NUMERIC(20, 6)", "name": "VARCHAR(255)", "sell": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id", "name"]
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)",
            "MarginPurchaseBuy": "NUMERIC(20, 6)", "MarginPurchaseSell": "NUMERIC(20, 6)",
            "MarginPurchaseCashRepayment": "NUMERIC(20, 6)", "MarginPurchaseLimit": "NUMERIC(20, 6)",
            "MarginPurchaseTodayBalance": "NUMERIC(20, 6)", "MarginPurchaseYesterdayBalance": "NUMERIC(20, 6)",
            "ShortSaleBuy": "NUMERIC(20, 6)", "ShortSaleSell": "NUMERIC(20, 6)",
            "ShortSaleCashRepayment": "NUMERIC(20, 6)", "ShortSaleLimit": "NUMERIC(20, 6)",
            "ShortSaleTodayBalance": "NUMERIC(20, 6)", "ShortSaleYesterdayBalance": "NUMERIC(20, 6)",
            "OffsetLoanAndShort": "NUMERIC(20, 6)", "Note": "VARCHAR(255)"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    "TaiwanStockShareholding": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", "stock_name": "VARCHAR(255)",
            "InternationalCode": "VARCHAR(255)", "ForeignInvestmentRemainingShares": "NUMERIC(20, 6)",
            "ForeignInvestmentShares": "NUMERIC(20, 6)", "ForeignInvestmentRemainRatio": "NUMERIC(20, 6)",
            "ForeignInvestmentSharesRatio": "NUMERIC(20, 6)", "NumberOfSharesIssued": "NUMERIC(20, 6)",
            "ForeignInvestmentUpperLimitRatio": "NUMERIC(20, 6)", "ChineseInvestmentUpperLimitRatio": "NUMERIC(20, 6)",
            "RecentlyDeclareDate": "DATE", "note": "VARCHAR(255)"
        },
        "unique_constraints": ["date", "stock_id"]
    },

    # --- Fundamental (基本面) ---
    "TaiwanStockFinancialStatements": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)",
            "type": "VARCHAR(255)", "value": "NUMERIC(20, 6)", "origin_name": "VARCHAR(255)"
        },
        "unique_constraints": ["date", "stock_id", "type", "origin_name"]
    },
    "TaiwanStockBalanceSheet": {
        # v2.21 (2026-05-25): §14.7-BI ROE 解鎖 — FinStmt 之姊妹表,提供真權益 (Equity / EquityAttributableToOwnersOfParent / CapitalStock / RetainedEarnings 等 101 types)
        # 同 FinStmt schema (date, stock_id, type, value, origin_name);quarterly;2011-12-01 ~ now
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)",
            "type": "VARCHAR(255)", "value": "NUMERIC(20, 6)", "origin_name": "VARCHAR(255)"
        },
        "unique_constraints": ["date", "stock_id", "type", "origin_name"]
    },
    "TaiwanStockMonthRevenue": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", 
            "country": "VARCHAR(255)", "revenue": "NUMERIC(20, 6)", 
            "revenue_month": "NUMERIC(20, 6)", "revenue_year": "NUMERIC(20, 6)", "create_time": "DATE"
        },
        "unique_constraints": ["date", "stock_id"]
    },
    "TaiwanStockDividend": {
        "columns": {
            "date": "DATE", "stock_id": "VARCHAR(255)", "year": "VARCHAR(255)",
            "StockEarningsDistribution": "NUMERIC(20, 6)", "StockStatutorySurplus": "NUMERIC(20, 6)",
            "CashEarningsDistribution": "NUMERIC(20, 6)", "CashStatutorySurplus": "NUMERIC(20, 6)",
            "AnnouncementDate": "DATE", "AnnouncementTime": "VARCHAR(255)",
            "CashDividendPaymentDate": "DATE", "CashExDividendTradingDate": "DATE",
            "CashIncreaseSubscriptionRate": "NUMERIC(20, 6)", "CashIncreaseSubscriptionpRrice": "NUMERIC(20, 6)",
            "ParticipateDistributionOfTotalShares": "NUMERIC(20, 6)", "RatioOfEmployeeStockDividend": "NUMERIC(20, 6)",
            "RatioOfEmployeeStockDividendOfTotal": "NUMERIC(20, 6)", "RemunerationOfDirectorsAndSupervisors": "NUMERIC(20, 6)",
            "StockExDividendTradingDate": "DATE", "TotalEmployeeCashDividend": "NUMERIC(20, 6)",
            "TotalEmployeeStockDividend": "NUMERIC(20, 6)", "TotalEmployeeStockDividendAmount": "NUMERIC(20, 6)",
            "TotalNumberOfCashCapitalIncrease": "NUMERIC(20, 6)"
        },
        "unique_constraints": ["date", "stock_id", "year"]
    },

    # --- Macro (FRED 宏觀主權；單表多 series_id，預設核心序列 DFF/UNRATE/T10Y2Y/VIXCLS) ---
    "FredData": {
        "columns": {
            "date": "DATE", "series_id": "VARCHAR(255)", 
            "value": "NUMERIC(20, 6)", "realtime_start": "DATE", "realtime_end": "DATE"
        },
        "unique_constraints": ["date", "series_id"]
    },
    
    # --- Market (市場總覽) ---
    "TaiwanStockInfo": {
        "columns": {
            "stock_id": "VARCHAR(255)", "stock_name": "VARCHAR(255)", 
            "industry_category": "VARCHAR(255)", "type": "VARCHAR(255)", "date": "DATE"
        },
        "unique_constraints": ["stock_id"]
    }
}
FINMIND_API_TABLES = {
    "TaiwanStockPrice": {"dataset": "TaiwanStockPrice", "data_id": "2330", "start_date": "2024-05-01"},
    "TaiwanStockPriceAdj": {"dataset": "TaiwanStockPriceAdj", "data_id": "2330", "start_date": "2024-05-01"},
    "TaiwanStockPER": {"dataset": "TaiwanStockPER", "data_id": "2330", "start_date": "2024-05-01"},
    "TaiwanStockInstitutionalInvestorsBuySell": {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": "2330", "start_date": "2024-05-01"},
    "TaiwanStockMarginPurchaseShortSale": {"dataset": "TaiwanStockMarginPurchaseShortSale", "data_id": "2330", "start_date": "2024-05-01"},
    "TaiwanStockShareholding": {"dataset": "TaiwanStockShareholding", "data_id": "2330", "start_date": "2024-05-01"},
    "TaiwanStockFinancialStatements": {"dataset": "TaiwanStockFinancialStatements", "data_id": "2330", "start_date": "2024-01-01"},
    "TaiwanStockBalanceSheet": {"dataset": "TaiwanStockBalanceSheet", "data_id": "2330", "start_date": "2024-01-01"},
    "TaiwanStockMonthRevenue": {"dataset": "TaiwanStockMonthRevenue", "data_id": "2330", "start_date": "2024-01-01"},
    "TaiwanStockDividend": {"dataset": "TaiwanStockDividend", "data_id": "2330", "start_date": "2020-01-01"},
    "TaiwanStockInfo": {"dataset": "TaiwanStockInfo", "data_id": "", "start_date": "2024-01-01"},
}

FRED_CONTRACT_SERIES = "DFF"
LOCAL_DERIVED_COLUMNS = {"FredData": {"series_id"}}
INFRA_TABLES = {"pipeline_execution_log", "data_audit_log"}

# ──────────────────────────────────────────────────────────────────────────────
# §8.5-9 Publication-date Strategy Registry (v2.18 新增;對齊憲章 §8.5-9.2 分派表)
# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset publication-date 規則 SSOT;Builder/audit 透過此 dict 取得 SQL gate 規則,
# 不得自行定義 publication-date 規則(對齊 §0.0-I 單一引用源原則 + 核心定義第 7 條)。
#
# Schema(每個 dataset):
#   {
#     "source":       str   — rule ID,對齊 §8.5-9.2 之 publication_date_source 欄
#     "column":       str | None — effective_publication_date 來源欄位(None = 用 date)
#     "offset_days":  int | dict | None — 法定推算偏移(int 或 quarter-aware dict)
#     "enforcement":  str   — 5 種 enforcement 之一(見 PUBLICATION_DATE_ENFORCEMENT_TYPES)
#     "description":  str   — 一句話說明
#   }
PUBLICATION_DATE_STRATEGY_REGISTRY = {
    # === Native-aligned: date = trading day, 本就無 publication delay ===
    "TaiwanStockPrice": {
        "source": "trading_day",
        "column": "date",
        "offset_days": 0,
        "enforcement": "native_aligned",
        "description": "T 日收盤後可得;`date` 即實際可觀測日",
    },
    "TaiwanStockPriceAdj": {
        "source": "trading_day",
        "column": "date",
        "offset_days": 0,
        "enforcement": "native_aligned",
        "description": "T 日收盤後可得;`date` 即實際可觀測日",
    },
    "TaiwanStockPER": {
        "source": "trading_day",
        "column": "date",
        "offset_days": 0,
        "enforcement": "native_aligned",
        "description": "T 日收盤後即時換算 PER/PBR/yield;`date` 即實際可觀測日",
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "source": "trading_day_post_1730",
        "column": "date",
        "offset_days": 0,
        "enforcement": "native_aligned",
        "description": "T 日 17:30 後可得(TWSE 17:00 後公告 + FinMind ~30 min);§6.8.7-A 已調 cron 17:30 對齊",
    },
    "TaiwanStockMarginPurchaseShortSale": {
        "source": "trading_day",
        "column": "date",
        "offset_days": 0,
        "enforcement": "native_aligned",
        "description": "T 日收盤後可得;`date` 即實際可觀測日",
    },
    "TaiwanStockInfo": {
        "source": "registry_snapshot_date",
        "column": "date",
        "offset_days": 0,
        "enforcement": "native_aligned",
        "description": "市場資產 metadata snapshot,無 publication delay",
    },
    # === Strict: 直接 use publication-date column ===
    "TaiwanStockDividend": {
        "source": "announcement_date",
        "column": "AnnouncementDate",
        "offset_days": 0,
        "enforcement": "strict",
        "description": "公告日先於除權息日 avg ~23 天(實證範圍 -10 ~ +148 days);符合台灣股利公告慣例",
    },
    "FredData": {
        # v2.19 追溯修正(§14.7-BB Phase 2 dry-run 揭露):DB realtime_start 為 ingest 日期(2026-05-21~22)
        # 非真實 ALFRED vintage;strict gate 對 historical as_of_date < 2026-05-21 100% loss → 降為 transitional
        "source": "fred_vintage_pending_alfred",
        "column": "date",
        "offset_days": 0,
        "enforcement": "transitional",
        "description": "v2.19 追溯降為 transitional(§14.7-BB):DB realtime_start = ingest 日期非真實 vintage;暫維持 `date` gate(同 Shareholding);§6.3 第 8 條治權目標保留,待 D2.4 ALFRED archival API 整合後升回 strict",
    },
    # === Hardcoded conservative: 法定截止日推算 ===
    "TaiwanStockMonthRevenue": {
        "source": "statutory_disclosure_deadline",
        "column": "date",       # statistical month-end
        "offset_days": 10,      # 台灣公司法 每月 10 日前公告
        "enforcement": "hardcoded_conservative",
        "description": "硬編 +10 天保守上限(因 `create_time` 經實證為 DB 寫入時間 median lag 5 年非公告日;§14.7-BA 揭露);實際公告 ≤ 統計月後 10 天",
    },
    "TaiwanStockFinancialStatements": {
        "source": "statutory_filing_deadline",
        "column": "date",       # statistical quarter-end
        "offset_days": {"Q1": 45, "Q2": 45, "Q3": 45, "Q4": 90},  # FSC 證交法施行細則
        "enforcement": "hardcoded_conservative",
        "description": "硬編法定截止日推算(Q1-Q3 +45 天 / Q4 +90 天);DB 無 publication-date 欄位(§14.7-BA 揭露);實際公告 ≤ 法定截止日",
    },
    "TaiwanStockBalanceSheet": {
        "source": "statutory_filing_deadline",
        "column": "date",       # statistical quarter-end(同 FinStmt)
        "offset_days": {"Q1": 45, "Q2": 45, "Q3": 45, "Q4": 90},  # 同 FinStmt(同為證交法 quarterly filing)
        "enforcement": "hardcoded_conservative",
        "description": "硬編法定截止日推算(同 FinStmt;Q1-Q3 +45 / Q4 +90);v2.21 §14.7-BI ROE 解鎖新表(Equity / RetainedEarnings 真權益來源,消解 §0.1.3-A.1 FinStmt mislabel)",
    },
    # === Transitional: 暫維持 date, 待研究升版 ===
    "TaiwanStockShareholding": {
        "source": "statistical_date_pending_research",
        "column": "date",
        "offset_days": 0,
        "enforcement": "transitional",
        "description": "`RecentlyDeclareDate` 語意不明 avg lag -161 days 反向(§14.7-BA §3.3 揭露);暫維持 `date`,待 D2.1 研究方向確認後升 strict",
    },
    # === Infrastructure: infra 觀測表不適用 publication-date ===
    "pipeline_execution_log": {
        "source": "infrastructure_no_publication_date",
        "column": None,
        "offset_days": None,
        "enforcement": "infrastructure",
        "description": "infra 觀測表;不參與 anti-leakage 流;publication-date 不適用",
    },
    "data_audit_log": {
        "source": "infrastructure_no_publication_date",
        "column": None,
        "offset_days": None,
        "enforcement": "infrastructure",
        "description": "infra 觀測表;不參與 anti-leakage 流;publication-date 不適用",
    },
}


def build_publication_date_gate(table: str, as_of_param_placeholder: str = "%s") -> tuple[str, int]:
    """
    依憲章 §8.5 第 9 條 Publication-date Discipline 為 table 構造 SQL gate clause + 參數計數。

    v2.20 升 SSOT(從 feature_store_builder v0.4 之 local _publication_date_gate 移入;
    rename 移除 private prefix 反映 utility 性質)。對應憲章 §8.5-9.2 分派表 + §0.0-I 單一引用源。

    讀 `PUBLICATION_DATE_STRATEGY_REGISTRY[table]` 依 enforcement 分派:
      - native_aligned / transitional: `date <= %s`(1 個 as_of_date 參數)
      - strict: `"<column>" <= %s`(1 個 as_of_date 參數;column 大寫者加雙引號)
      - hardcoded_conservative (offset_days = int): `(date + INTERVAL 'N days') <= %s`(1 個)
      - hardcoded_conservative (offset_days = quarter dict): quarter-aware
            `((EXTRACT(QUARTER FROM date) IN (1,2,3) AND (date + INTERVAL '45 days') <= %s)
              OR (EXTRACT(QUARTER FROM date) = 4 AND (date + INTERVAL '90 days') <= %s))`
            (2 個 as_of_date 參數)
      - infrastructure: raise ValueError(infra 表不參與 feature/universe builder 流)

    Returns:
        (where_clause, extra_param_count)
        e.g. ("\"AnnouncementDate\" <= %s", 1)

    對齊憲章 §8.5-9.2 分派表;§8.5-9.3 透明性要求;v2.20 SSOT 化。
    """
    strategy = PUBLICATION_DATE_STRATEGY_REGISTRY.get(table)
    if strategy is None:
        raise ValueError(f"table {table!r} 不在 PUBLICATION_DATE_STRATEGY_REGISTRY 內")

    enforcement = strategy["enforcement"]
    column = strategy["column"]
    offset = strategy["offset_days"]
    ap = as_of_param_placeholder

    if enforcement == "infrastructure":
        raise ValueError(f"table {table!r} 屬 infrastructure(不參與 feature/universe 流);不得在 builder 內使用")

    if enforcement in ("native_aligned", "transitional", "strict"):
        col_quoted = f'"{column}"' if any(c.isupper() for c in column) else column
        return f"{col_quoted} <= {ap}", 1

    if enforcement == "hardcoded_conservative":
        if isinstance(offset, int):
            return f"(date + INTERVAL '{offset} days') <= {ap}", 1
        if isinstance(offset, dict):
            q123_offset = offset.get("Q1", offset.get("Q2", offset.get("Q3", 45)))
            q4_offset = offset.get("Q4", 90)
            clause = (
                f"((EXTRACT(QUARTER FROM date) IN (1,2,3) "
                f"AND (date + INTERVAL '{q123_offset} days') <= {ap}) "
                f"OR (EXTRACT(QUARTER FROM date) = 4 "
                f"AND (date + INTERVAL '{q4_offset} days') <= {ap}))"
            )
            return clause, 2

    raise ValueError(f"unknown enforcement {enforcement!r} for table {table!r}")


class SovereignSchemaManager:
    def __init__(self):
        self.stats = {"success": 0, "failed": 0, "details": []}
        self.contract_stats = {"pass": 0, "warn": 0, "failed": 0, "details": []}
        self.constitution_ver = "v6.0.0"

    def _record_contract(self, status, item, detail):
        self.contract_stats[status] += 1
        icon = {"pass": "✅", "warn": "⚠️", "failed": "❌"}[status]
        self.contract_stats["details"].append(f"{icon} [API-{status.upper()}] {item} - {detail}")

    def _api_tables_for_target(self, target_table):
        if target_table:
            if target_table in INFRA_TABLES:
                return []
            return [target_table]
        return list(FINMIND_API_TABLES.keys()) + ["FredData"]

    def _compare_columns(self, table_name, api_columns):
        schema_columns = set(DATASET_REGISTRY[table_name]["columns"].keys())
        api_columns = set(api_columns)
        derived = LOCAL_DERIVED_COLUMNS.get(table_name, set())
        missing_from_api = sorted(schema_columns - api_columns - derived)
        missing_from_schema = sorted(api_columns - schema_columns)
        return missing_from_api, missing_from_schema

    def _probe_finmind_contract(self, table_name):
        probe = FINMIND_API_TABLES[table_name]
        client = FinMindClient()
        params = {
            "dataset": probe["dataset"],
            "start_date": probe["start_date"],
        }
        if probe.get("data_id") is not None:
            params["data_id"] = probe.get("data_id", "")
        if client.token:
            params["token"] = client.token

        res = requests.get(client.api_url, params=params, headers=client.headers, timeout=30)
        res.raise_for_status()
        payload = res.json()
        if payload.get("msg") not in (None, "success"):
            raise RuntimeError(payload.get("msg"))
        data = payload.get("data", [])
        if not data:
            self._record_contract("warn", table_name, "API reachable but returned empty sample; DDL will use local registry")
            return

        missing_from_api, missing_from_schema = self._compare_columns(table_name, data[0].keys())
        if missing_from_api or missing_from_schema:
            problems = []
            if missing_from_api:
                problems.append(f"schema-only={missing_from_api}")
            if missing_from_schema:
                problems.append(f"api-only={missing_from_schema}")
            self._record_contract("failed", table_name, "; ".join(problems))
        else:
            self._record_contract("pass", table_name, f"{len(data[0].keys())} columns matched")

    def _probe_fred_contract(self):
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            self._record_contract("failed", "FredData", "FRED_API_KEY missing")
            return
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": FRED_CONTRACT_SERIES, "api_key": api_key, "file_type": "json", "limit": 1}
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        payload = res.json()
        data = payload.get("observations", [])
        if not data:
            self._record_contract("failed", "FredData", "FRED returned empty observations")
            return
        missing_from_api, missing_from_schema = self._compare_columns("FredData", data[0].keys())
        if missing_from_api or missing_from_schema:
            problems = []
            if missing_from_api:
                problems.append(f"schema-only={missing_from_api}")
            if missing_from_schema:
                problems.append(f"api-only={missing_from_schema}")
            self._record_contract("failed", "FredData", "; ".join(problems))
        else:
            self._record_contract("pass", "FredData", f"{len(data[0].keys())}+1 derived columns matched")

    def probe_api_contracts(self, target_table=None):
        """DDL 前置 API 契約探測。失敗時不得重鑄 schema。"""
        targets = self._api_tables_for_target(target_table)
        if not targets:
            self._record_contract("pass", target_table or "infrastructure", "no external API contract required")
            return True

        print(f"🔎 正在執行 API-first 契約探測 ({self.constitution_ver})...")
        for table_name in targets:
            if table_name not in DATASET_REGISTRY:
                self._record_contract("failed", table_name, "table is not registered in DATASET_REGISTRY")
                continue
            try:
                if table_name == "FredData":
                    self._probe_fred_contract()
                else:
                    self._probe_finmind_contract(table_name)
            except Exception as e:
                self._record_contract("failed", table_name, f"{type(e).__name__}: {e}")

        return self.contract_stats["failed"] == 0

    def init_tables(self, target_table=None, force=False, skip_api_contract=False):
        """執行憲法 v6.0.0 API-first 標準之物理初始化"""
        start_time = time.time()
        if not skip_api_contract and not self.probe_api_contracts(target_table=target_table):
            self.stats["failed"] += 1
            self.stats["details"].append("❌ [FAILED] API 契約探測失敗；已停止 DDL 重鑄")
            self.report_results(start_time, ddl_executed=False)
            return False
        if skip_api_contract:
            self._record_contract("warn", "API-first", "--skip-api-contract used; DDL executed from local registry")

        conn = get_db_connection()
        cur = conn.cursor()

        print("🛠️  正在啟動主權初始化程序...")

        tables = [target_table] if target_table else DATASET_REGISTRY.keys()

        for table_name in tables:
            if table_name not in DATASET_REGISTRY:
                self.stats["failed"] += 1
                self.stats["details"].append(f"❌ [FAILED] 表名: \"{table_name}\" - 未登錄於 DATASET_REGISTRY")
                continue
            config = DATASET_REGISTRY[table_name]

            try:
                if force:
                    cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')

                cols_def = ", ".join([f'"{k}" {v}' for k, v in config["columns"].items()])
                cur.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_def})')

                if config.get("unique_constraints"):
                    constraint_name = f"uq_{table_name.lower()}"
                    cur.execute(f'ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS {constraint_name}')
                    cols_str = ", ".join([f'"{c}"' for c in config["unique_constraints"]])
                    cur.execute(f'ALTER TABLE "{table_name}" ADD CONSTRAINT {constraint_name} UNIQUE ({cols_str})')

                if "date" in config["columns"] and "log" not in table_name:
                    cur.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name.lower()}_date" ON "{table_name}" ("date")')

                conn.commit()
                self.stats["success"] += 1
                self.stats["details"].append(f"✅ [SUCCESS] 表名: \"{table_name}\" - 絕對大小寫封印完成")

                if "log" not in table_name:
                    write_data_audit_log(table_name, "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "SCHEMA_INIT", 1)

            except Exception as e:
                conn.rollback()
                self.stats["failed"] += 1
                self.stats["details"].append(f"❌ [FAILED] 表名: \"{table_name}\" - 錯誤: {str(e)}")

        cur.close()
        conn.close()
        self.report_results(start_time, ddl_executed=True)
        return self.stats["failed"] == 0

    def report_results(self, start_time, ddl_executed=True):
        """顯示詳細結果訊息 (憲法 5.6 條款)"""
        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 資料庫主權初始化報告 (v2.16)")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{self.constitution_ver}.md")
        print(f"核心技術 : API Contract First + Absolute Case Sovereignty")
        print("─" * 80)
        for d in self.contract_stats["details"]:
            print(d)
        print("─" * 80)
        for d in self.stats["details"]:
            print(d)
        print("─" * 80)
        print(f"🔎 API PASS/WARN/FAIL : {self.contract_stats['pass']}/{self.contract_stats['warn']}/{self.contract_stats['failed']}")
        print(f"📈 總計項目 : {len(DATASET_REGISTRY)}")
        print(f"✅ 成功重鑄 : {self.stats['success']}")
        print(f"❌ 失敗項目 : {self.stats['failed']}")
        print(f"🧱 DDL 執行 : {'YES' if ddl_executed else 'NO'}")
        print(f"🕒 總計耗時 : {(time.time() - start_time)*1000:.2f} ms")
        verdict = "PERFECT ALIGNMENT" if self.stats["failed"] == 0 and self.contract_stats["failed"] == 0 else "FAILED"
        if verdict == "PERFECT ALIGNMENT" and self.contract_stats["warn"] > 0:
            verdict = "WARNING"
        print(f"⚖️  主權判定 : {verdict}")
        print("🛡️" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="啟動主權初始化")
    parser.add_argument("--force", action="store_true", help="強制重置現有表")
    parser.add_argument("--table", type=str, help="指定單一表名")
    parser.add_argument("--skip-api-contract", action="store_true", help="離線/災難復原：略過 API-first 契約探測")
    args = parser.parse_args()
    
    manager = SovereignSchemaManager()
    if args.init:
        ok = manager.init_tables(target_table=args.table, force=args.force, skip_api_contract=args.skip_api_contract)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()
