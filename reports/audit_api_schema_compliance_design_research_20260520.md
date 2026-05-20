# `audit_api_schema_compliance.py v0.1` 設計研究報告（v0.2 含使用者裁決後修訂）

**研究日期**：2026-05-20 (Asia/Taipei)
**研究階段**：依憲章 **§0.0-G 憲章先行紀律 — Step 1（研究）**
**修訂時點**：2026-05-20 — 使用者完成 6 項裁決，研究範圍由 4 層擴張至 **9 層**
**研究主題**：FinMind + FRED API 抓取資料之 schema 命名 / 型態 / 大小 + 資料完整性 audit 工具設計
**對應憲章**：§1.4 / L2388 / §3.2A / §5.6.3 / §0.0-G / §0.4 / §6.7
**裁決狀態**：✅ **使用者已完成全部 6 項裁決** → 進入 **§0.0-G Step 2（入憲）**
**Git HEAD**：`e3b1497`（含 `data_schema.py v2.14` + `core/__init__.py v1.16`）

---

## 🗳️ 使用者裁決紀錄（2026-05-20）

| # | 議題 | 裁決 | 對比研究推薦 |
|---|---|---|---|
| 1 | 方案選擇 | **乙：新建獨立工具 `audit_api_schema_compliance.py v0.1`** | ✅ 同推薦 |
| 2 | 檢驗深度 | **深：Layer A+B+C+D**（後升級為 9 層 A-I） | ⬆️ 比推薦更深 |
| 3 | 失敗處理 | **嚴格：任何 FAILED → exit 1** | ⬆️ 比推薦更嚴格（不分 --strict） |
| 4 | 入憲時機 | **嚴格分離：先 Step 2 入憲 commit，後 Step 3 落地 commit** | ⬆️ 比推薦更嚴格 |
| 5 | 覆蓋範圍 | **路徑 2：v0.1 所有 9 層**（schema + 資料完整性一次達陣） | ⬆️ 範圍擴張 |
| 6 | 取樣大小 | （未明示） → **採推薦預設 100** | ✅ 同推薦 |

**裁決影響**：
- 工程量自 500-700 行升至 **1000-1500 行**
- 範圍自 schema-only 升至 **schema + 資料完整性一次達陣**
- 入憲流程嚴格分離 → **兩個 commit**（Step 2 入憲 / Step 3 落地）
- 任何 FAILED 一律 exit 1 → 無 `--strict` flag（或 `--strict` 改為預設）

---

## 一、研究背景

### 1.1 使用者需求陳述

> 「要針對所有從外部抓取的資料 finmind api 與 fred api 都要有對應的 audit 程式來確認正確性。如 table schema 命名、型態、大小均要符合系統架構大憲章_v6.0.0.md 規則」

裁決後延伸：「研究報告有包含資料的完整性嗎?」→ 將範圍擴至**廣義資料完整性**

### 1.2 需求拆解

| 檢驗項 | 具體內涵 |
|---|---|
| 命名 | column name + 大小寫對齊（憲章 §1.4 Absolute Case Sovereignty） |
| 型態 | DDL data_type（VARCHAR / NUMERIC / DATE / INTEGER / TIMESTAMP）對齊 DB 物理 type + 對齊 API 回傳值 type |
| 大小 | VARCHAR length（憲章預設 255）+ NUMERIC precision/scale（憲章預設 20/6） |
| **資料完整性**（裁決後擴張） | PK/Unique 衝突、duplicate row、date 連續性、referential integrity、value range sanity、NULL ratio |

---

## 二、憲章規則引用

| 憲章節 | 規範內容 | 與本提案關係 |
|---|---|---|
| §1.4 [Defensive Architecture] | VARCHAR(255) 防禦性寬容 + NUMERIC(20,6) 統一精度 | Layer A/C 治權依據 |
| §1.4 [Absolute Case Sovereignty] | 雙引號封裝 1:1 大小寫 | Layer A 治權依據 |
| **L2388 Derived Schema 欄位繼承原則** | 「SQL 型別寬度不得更窄」 | **Layer A/C 核心治權任務** |
| §3.2A 橫切稽核工具子表（L2459-2473） | 5 個 audit 工具 | 本工具為**第 6 個** |
| §3.2A 治權邊界（L2473） | 須遵守 §5.6.3 / §0.4 / §3.2 接受標準 | 本工具設計規範 |
| §5.6.3 [Zero Hardcoded Verdict] | PERFECT / WARNING / FAILED 必須動態計算 | verdict 邏輯規範 |
| §3.2 接受標準 | PERFECT/WARNING → exit 0；FAILED → exit 1 | exit code 規範 |
| §0.4 可觀察性 | record_lifecycle + write_data_audit_log | 觀測接線 |
| §0.0-G 憲章先行紀律 | 程式變動前先動憲章 | **本研究即 Step 1** |
| §6.7 SQL 契約 | core_universe_membership JOIN core_universe_snapshot WHERE status='committed' | Layer H referential integrity 引用 |
| §8.5 anti-leakage 第 5 條（時間防漏） | 時間欄位 date 不得超過 cutoff | Layer G date 連續性間接相關 |

---

## 三、現況工具地圖與覆蓋面分析

### 3.1 既有 6 個 audit 工具職責盤點

| 工具 | 版本 | 職責 | 覆蓋 |
|---|---|---|---|
| `data_schema.probe_api_contracts()` | v2.14 | API 欄位名稱對齊 | ✅ 名稱 |
| `audit_supply_chain.audit_db_schema()` | v1.18 | DB column 名稱比對 | ✅ 名稱 |
| `audit_source_availability.py` | v0.1 | row count + min/max date | ✅ row count + date 範圍 |
| `audit_core_universe.py` | v0.1 | 核心股治理表 41 檢驗項 | N/A（不涉 raw API）|
| `audit_leakage.py` | v0.1 | §8.5 時間防漏 | N/A |
| `audit_downstream_readiness.py` | v0.1 | §8 升版 readiness | N/A |
| `audit_doctrine_compliance.py` | v0.1 | §0 核心思想對映 | N/A |

### 3.2 覆蓋缺口完整盤點（裁決後更新）

| 檢驗項 | 既有覆蓋 | 缺口 |
|---|---|---|
| column name + 大小寫 | ✅ data_schema + supply_chain | 無 |
| DB 物理 data_type | ❌ | **缺**（Layer A） |
| DB 物理 length / precision / scale | ❌ | **缺**（Layer A） |
| API 回傳值之 type cast 可行性 | ❌ | **缺**（Layer B） |
| API 字串長度 / 數值範圍 | ❌ | **缺**（Layer C） |
| 樣本 NULL 比例 | ❌ | **缺**（Layer D） |
| PK / Unique constraints 衝突 | ❌ | **缺**（Layer E） |
| Duplicate row 偵測 | ❌ | **缺**（Layer F） |
| Date 序列連續性 | ❌ | **缺**（Layer G） |
| Referential integrity（跨表 FK） | ❌ | **缺**（Layer H） |
| Value range sanity（負值 / 異常分佈） | ❌ | **缺**（Layer I） |

---

## 四、缺口識別 → 9 層治權任務

| 治權任務 | 對應 Layer | 嚴重度 | 對應憲章 |
|---|---|---|---|
| DDL ↔ DB Physical 一致性 | A | HIGH | §1.4 / L2388 |
| API ↔ DDL Type Compatibility | B | HIGH | §3.2 Step 2 深度 |
| Length / Precision Range | C | HIGH | §1.4 / L2388 |
| NULL Ratio Sanity | D | MEDIUM | §0.4 觀察性 |
| PK / Unique Uniqueness | E | HIGH | DDL unique_constraints 強制 |
| Duplicate Row Detection | F | MEDIUM | §0.4 觀察性 |
| Date Series Continuity | G | MEDIUM | §8.5 anti-leakage 間接 |
| Referential Integrity | H | HIGH | §6.7 SQL 契約 |
| Value Range Sanity | I | MEDIUM | §0.4 觀察性 |

---

## 五、三案比較（裁決結果 → 採乙案）

| 方案 | 動作 | 工程量 | 裁決 |
|---|---|---|---|
| 甲 | 強化 audit_supply_chain v1.18 → v1.19 | 中 | ❌ |
| **乙** | **新建獨立工具 audit_api_schema_compliance v0.1** | 大（1000-1500 行）| ✅ **採用** |
| 丙 | 甲 + 乙雙軌 | 最大 | ❌ |

---

## 六、乙案設計細節（裁決後升級至 9 層）

### 6.1 九層動態檢驗

| Layer | 名稱 | 檢驗對象 | 嚴格度 | 對應憲章 |
|---|---|---|---|---|
| **A** | **DDL ↔ DB Physical Consistency** | DATASET_REGISTRY DDL 字串 vs `information_schema.columns` (data_type / character_maximum_length / numeric_precision / numeric_scale / is_nullable) | **FAILED** | §1.4 / L2388 |
| **B** | **API Sample ↔ DDL Type Compatibility** | FinMind/FRED 取樣 100 筆，逐欄位 cast 為對應 DDL type | **FAILED** | §3.2 Step 2 深度延伸 |
| **C** | **API Sample Length / Precision Range** | 樣本字串 max length vs DDL VARCHAR(N)；數值 max abs vs NUMERIC(p, s) 範圍 | **FAILED**（裁決升級）| §1.4 / L2388 |
| **D** | **NULL Ratio Sanity** | 樣本中 NULL 比例 > 50% 且為 unique constraint 成員 | **FAILED**（裁決升級）| §0.4 + DDL |
| **E** | **PK / Unique Constraint Uniqueness** | SQL `SELECT COUNT(*) - COUNT(DISTINCT pk_cols)` 應為 0 | **FAILED** | DDL 強制 |
| **F** | **Duplicate Row Detection** | 非 unique 欄位之 row-level 完全重複 | **FAILED**（裁決升級）| §0.4 |
| **G** | **Date Series Continuity** | 交易日連續性（扣除已知假日）；日序漏洞偵測 | **FAILED**（裁決升級）| §8.5 anti-leakage 間接 |
| **H** | **Referential Integrity** | 跨表 stock_id 存在性（如 TaiwanStockPrice.stock_id ∈ TaiwanStockInfo.stock_id） | **FAILED** | §6.7 SQL 契約 |
| **I** | **Value Range Sanity** | 負值 / 異常大值 / 異常分佈（如 Trading_Volume < 0、close > 100000、PER < 0） | **FAILED**（裁決升級）| §0.4 + 物理常識 |

**關鍵裁決變更**：原本 Layer C/D/F/G/I 為 WARNING，依使用者「嚴格：任何 FAILED → exit 1」之裁決，**全升至 FAILED**。

### 6.2 CLI 介面（無 `--strict` flag — 預設即嚴格）

對齊 `audit_source_availability.py v0.1` 之 SOP，但**任何 FAILED 一律 exit 1**：

```bash
# 標準執行：全 13 表（FinMind 10 + FRED 1 + infra 2）+ 9 層全跑
venv/bin/python scripts/maintenance/audit_api_schema_compliance.py

# 含 FRED 深度檢驗（預設不含 FRED 之 sample probe；可選含）
venv/bin/python scripts/maintenance/audit_api_schema_compliance.py --include-fred

# 單一表
venv/bin/python scripts/maintenance/audit_api_schema_compliance.py --table TaiwanStockPrice

# 離線：只查 DDL ↔ DB Physical（Layer A / E / F / G / H / I 即 DB-side），略過 Layer B/C/D 之 API 探測
venv/bin/python scripts/maintenance/audit_api_schema_compliance.py --skip-api-probe

# 自訂取樣大小（Layer B/C/D 用，預設 100）
venv/bin/python scripts/maintenance/audit_api_schema_compliance.py --sample-size 500

# 自訂 layer 範圍（debug 用）
venv/bin/python scripts/maintenance/audit_api_schema_compliance.py --layers A,B,E

# 報告寫入路徑（預設 reports/api_schema_compliance_audit_<YYYYMMDD_HHMM>.md）
venv/bin/python scripts/maintenance/audit_api_schema_compliance.py --report-out /tmp/foo.md
```

### 6.3 verdict 動態計算邏輯（對齊 §5.6.3）

```python
# 偽碼（嚴禁硬編碼）
def _compute_verdict(failures: int, warnings: int) -> str:
    if failures > 0:
        return "FAILED"
    if warnings > 0:
        return "WARNING"  # 罕見情況；本工具預設將 mismatch 視為 FAILED
    return "PERFECT"
```

注：裁決後幾乎不會回 WARNING（mismatch 一律升 FAILED），但保留邏輯以便將來新增 INFO/WARN 級檢查。

### 6.4 exit code 對齊 §3.2 接受標準（裁決後嚴格化）

| Verdict | exit code |
|---|---|
| PERFECT | 0 |
| WARNING | 0（觀察用，僅記錄）|
| **FAILED** | **1**（無條件阻斷下游 maintenance / ingestion；無 `--strict` flag）|

### 6.5 與既有 audit 工具的協作

| 既有工具 | 與本工具關係 |
|---|---|
| `data_schema.probe_api_contracts()` | 本工具 Layer A/B 之輕量版 |
| `audit_supply_chain.audit_db_schema()` | 本工具 Layer A 之深度版 |
| `audit_source_availability.py v0.1` | **互補**：對方查 row count / date min-max；本工具 Layer G 含日序連續性 |
| `audit_doctrine_compliance.py` | 不直接關聯；但本工具自身須通過 doctrine_compliance 之 §0 對映檢驗 |

### 6.6 input 契約

| 輸入源 | 用途 |
|---|---|
| `core.data_schema.DATASET_REGISTRY` | DDL 字串解析 ground truth |
| `core.data_schema.FINMIND_API_TABLES` | FinMind 11 個 dataset probe 入口 |
| `core.data_schema.FRED_CONTRACT_SERIES` | FRED 4 series probe 入口 |
| `core.db_utils.get_db_connection()` | PostgreSQL `information_schema.columns` + table data 查詢 |
| `core.finmind_client.FinMindClient` | FinMind API 取樣 |
| `requests + FRED_API_KEY` | FRED API 取樣 |
| `core.path_setup.get_report_dir()` | 報告寫入路徑 |
| 台灣交易日曆（待補；可從 TaiwanStockPrice 已有 date 反推） | Layer G date 連續性檢驗 |

### 6.7 output 契約

寫入：`reports/api_schema_compliance_audit_<YYYYMMDD_HHMM>.md`

含：
- 環境快照（DB conn / API key 狀態 / 取樣大小 / 9 個 Layer 啟用狀態）
- 9 Layer 逐項結果表（每個 table × 每個 column × 9 Layer = 細粒度）
- verdict 動態計算明細
- targeted 修補建議（DDL `ALTER TABLE` / API 報修 / row 清理 SQL）
- lifecycle 接線紀錄（pipeline_execution_log + data_audit_log 寫入確認）

---

## 七、入憲提案（§0.0-G Step 2）

### 7.1 §3.2A 子表新增第 6 個 audit（L2466-2471 之後）

```markdown
| **`audit_api_schema_compliance.py`** | **v0.1** | **9 層動態檢驗（A DDL↔DB Physical / B API↔Type / C Length-Precision / D NULL Ratio / E PK-Unique / F Duplicate Row / G Date Continuity / H Referential / I Value Range）；落實憲章 L2388「SQL 型別寬度不得更窄」+ §6.7 SQL 契約 referential integrity 之治權強制；任何 FAILED → exit 1** | **`--include-fred` / `--table <name>` / `--skip-api-probe` / `--sample-size <N>` / `--layers <A,B,...>`** | **§1.4、L2388、§3.2A、§6.7** | **REAL ✅** |
```

### 7.2 §二維運矩陣新增 SCHEMA-AUDIT 步驟

建議插入位置：**Step 3.5**（在 audit_supply_chain Step 3 之後、Step 4 之前）

```markdown
| **3.5. [標準序列：API Schema 合規 + 資料完整性 9 層深度驗收]** | `$ python scripts/maintenance/audit_api_schema_compliance.py --include-fred` | `audit_api_schema_compliance v0.1` |
```

理由：
- Step 3 audit_supply_chain 只查名稱（淺層）
- 本工具補 9 層深度檢驗（schema + 資料完整性）
- 在 Step 4 sovereign_sync_engine 灌溉前確認 schema + 完整性安全

### 7.3 §14.7-AJ 補登研究 + 入憲紀錄（Step 2）

§14 末尾（§14.7-AI 之後）新增：

```markdown
### §14.7-AJ 2026-05-20 audit_api_schema_compliance.py v0.1 設計研究 + 入憲（補齊 L2388 規則 + §6.7 referential integrity audit 缺口）

依使用者要求「FinMind + FRED API 抓的資料要有對應 audit 程式確認 schema 命名 / 型態 / 大小 + 資料完整性符合憲章」，
經 §0.0-G 憲章先行紀律之三步驟：

**Step 1（研究）**：reports/audit_api_schema_compliance_design_research_20260520.md v0.2
- 揭露 L2388「SQL 型別寬度不得更窄」之治權規則無 audit 落實
- 揭露 §6.7 SQL 契約之 referential integrity 無 audit 落實
- 對比現有 6 個 audit 工具，識別 schema + 完整性為 §3.2A 之第 6 個缺口
- 使用者完成 6 項裁決：採乙案 / 9 層覆蓋 / 嚴格 exit 1 / 嚴格分離雙 commit / 取樣 100

**Step 2（入憲）**：本節 + §3.2A L2466-2471 表第 6 行 + §二維運矩陣 Step 3.5

**Step 3（落地）**：scripts/maintenance/audit_api_schema_compliance.py v0.1（待後續 commit 補登實證）

§0.0-G 第七次跑通（前六次：§9.2 / §9.9 / §9.1-A〜I / §14.7-AG / §14.7-AG 自我修正 / §0.1-F）。
本補登（Step 2）不改 §1〜§7 強制契約、§6.7 SQL、25 維路徑、§8/§9 條文；僅補齊 §3.2A 之第 6 個 audit 工具 +
§二維運矩陣 Step 3.5 + §14.7-AJ 研究紀錄。Step 3 落地後本節將補登實證段。
```

### 7.4 不動其他憲章節

| 不動 | 原因 |
|---|---|
| §1〜§7 強制契約 | 本工具是 audit，不改 schema/SQL/路徑/序列 |
| §6.7 SQL | 不涉核心股查詢；本工具只讀 §6.7 結果做 referential check |
| §8/§9 | 不涉 feature/model/prediction/sizer |
| §0.0-A〜§0.0-I 元規則 | 本工具落地不改元規則 |
| §1.4 / L2388 | 本工具是落實，非修改 |

---

## 八、跨基柱影響評估

| 基柱 | 影響 | 裁決 |
|---|---|---|
| §0.0-B 第一性原理 | 不涉及 | N/A |
| §0.0-C 八二法則 | 不涉及 | N/A |
| §0.0-D 康波週期 | 不涉及 | N/A |
| §0.0-E 統合層 | 不違反（屬 §3.2A 工程實作層） | 安全 |
| §0.0-F AI 協作工具規則 | 不違反 | 安全 |
| §0.0-G 憲章先行紀律 | **本研究即執行 §0.0-G** | 第 7 次跑通 |
| §0.0-H 強制契約通用模板 | 適用 | v0.1 採 ACTIVE 條款 |
| §0.0-I 單一引用源 | 適用 | DATASET_REGISTRY 為唯一 schema SSOT |

---

## 九、§0.0-G 三步驟路線圖（裁決後嚴格分離模式）

| 步驟 | 狀態 | 動作 | 觸動範圍 | Commit |
|---|---|---|---|---|
| **Step 1：研究** | ✅ **本報告 v0.2** | 設計分析 + 三案比較 + 使用者裁決 | 1 個研究報告（本檔，~520 行） | （含於 Step 2 commit） |
| **Step 2：入憲** | ⏸️ **待執行** | 改 `reports/系統架構大憲章_v6.0.0.md` 之 §3.2A / §二 / §14.7-AJ | 憲章 3 個 Edit + 研究報告 | **第一 commit + push** |
| **Step 3：落地** | ⏸️ **待 Step 2 完成後執行** | 寫 `audit_api_schema_compliance.py v0.1` + 實測 + 補登實證 | 新檔 ~1000-1500 行 + 報告 + §14.7-AJ 落地紀錄 | **第二 commit + push** |

---

## 十、已裁決事項彙整

### 10.1 方案選擇：**乙案**（新建獨立工具）

### 10.2 檢驗深度：**9 層全跑**（A-I）

### 10.3 失敗處理：**嚴格**（任何 FAILED → exit 1；無 `--strict` flag）

### 10.4 入憲時機：**嚴格分離**（Step 2 + Step 3 兩個 commit）

### 10.5 取樣大小：**100**（預設）

### 10.6 排程：（裁決時未明示，採研究推薦）**on-demand + ingestion 後**（由 sovereign_sync_engine 在完成核心同步後自動呼叫；本研究 v0.2 不展開排程細節，留待 v0.2 工具落地後另案處理）

---

## 十一、實作風險與緩解

| 風險 | 緩解 |
|---|---|
| FinMind / FRED API rate limit | 重用 §7 三層防禦；取樣 ≤ 500 |
| DDL 字串解析複雜（`NUMERIC(20, 6)` 含空格） | regex + AST 雙保險 + 13 表單元測試 |
| API 回傳 None / "." / 空字串 | 對齊憲章 §3.2 既有 FRED `.` 缺值處理 |
| `information_schema.columns` 查不到（schema 未建） | bootstrap 階段 graceful skip（§3.2A 治權邊界要求） |
| audit 本身 crash 阻斷下游 | try/except + record_lifecycle.mark_warning，本身不上升為 FAILED |
| 9 層全跑耗時可能 > 1 分鐘 | `--layers` flag 可分批；`--skip-api-probe` 可只跑 DB-side 6 層 |
| Layer G 台灣交易日曆缺乏 SSOT | v0.1 採「DB 已有 date 之最大連續區間」反推；v0.2 可考慮引入 twstock 等 library |
| Layer H referential integrity 跨表查詢可能慢 | 用 `EXISTS` subquery；對齊 `db_utils.get_core_stocks_from_db()` 既有 SQL 模式 |

---

## 十二、本工具自我合規檢驗（§3.2A L2473）

依 §3.2A 治權邊界，本工具自身必須通過：

| 規範 | 實作對應 |
|---|---|
| §5.6.3 零硬編 PERFECT | `_compute_verdict()` 動態計算 |
| §0.4 可觀察性 | `with record_lifecycle('api_schema_compliance_audit'):` + `write_data_audit_log` |
| §3.2 接受標準（嚴格化） | `sys.exit(0 if verdict in ('PERFECT', 'WARNING') else 1)` |
| §0.0-G.0 Type-3 實作層 | 標頭宣告 Type-3 |
| 標頭 8 項檢查面（per_program_audit §7.5） | v0.1 須含 6 條核心定義（[Zero Hardcoded Verdict] + [Sovereignty Declaration]） |
| 標準 markdown 表格 header | v0.1 即遵守 |
| §0.0-I 單一引用源 | DATASET_REGISTRY 為唯一 schema SSOT；本工具不另立 schema 定義 |

---

## 十三、結論與後續路線圖

### 13.1 研究結論

✅ **本研究確認需求合理 + 治權缺口真實 + 方案明確 + 裁決完整**：

1. 憲章 **L2388** 已有「SQL 型別寬度不得更窄」規則 + **§6.7 SQL 契約** referential integrity 規則，但兩條皆**無工具落實**，本提案補齊
2. 採方案：**乙**（新建獨立工具 `audit_api_schema_compliance.py v0.1`）
3. 範圍：**9 層**（A DDL↔DB Physical / B API↔Type / C Length-Precision / D NULL Ratio / E PK-Unique / F Duplicate Row / G Date Continuity / H Referential / I Value Range）
4. 失敗處理：任何 FAILED → exit 1（嚴格）
5. 入憲：嚴格分離 Step 2 / Step 3 兩個 commit
6. 工程量：v0.1 約 1000-1500 行；Step 2 約 3 個憲章 Edit + 1 個研究報告；Step 3 約 1 新檔 + 1 實證報告 + 1 個憲章補登

### 13.2 §0.0-G 後續路線圖（嚴格分離模式）

| Phase | 動作 | 完成標誌 |
|---|---|---|
| **Phase 1（Step 1.5）** | ✅ 本研究報告 v0.2 更新（裁決後） | 本檔寫入 |
| **Phase 2（Step 2）** | ⏸️ 改憲章 3 處（§3.2A / §二 / §14.7-AJ） | **第一 commit + push** |
| **Phase 3（Step 3）** | ⏸️ 寫程式 + 實測 + 報告 + §14.7-AJ 補登實證 | **第二 commit + push** |

### 13.3 自我合規宣言

本研究報告之撰寫**本身就遵守** §0.0-G 憲章先行紀律：先研究、先記錄、先擴張範圍至 9 層、先列出待裁決事項、先收集使用者裁決、後進入入憲與落地。本紀錄為 §0.0-G **第 7 次跑通**之開端。

---

**研究報告版本**：v0.2（含裁決後修訂）
**對應 §0.0-G 階段**：Step 1 已完成
**HEAD commit at research time**：`e3b1497`
**接續處理**：進入 Phase 2（Step 2 入憲 commit）
