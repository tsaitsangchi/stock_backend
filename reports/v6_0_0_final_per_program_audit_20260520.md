# v6.0.0-FINAL 逐元件治權合規審計報告（依視角 A：系統依賴順序）

- **生成時間**: 2026-05-20 Asia/Taipei
- **基準**: HEAD `aecd50e` / Tag `v6.0.0-FINAL-Phase-A-quarantined`
- **憲章基準**: `reports/系統架構大憲章_v6.0.0.md` v6.0.0-FINAL
- **配套文件**:
  - `reports/v6_0_0_final_code_audit_20260520.md`（Phase A 程式碼治權審計）
  - `reports/v6_0_0_final_db_rebuild_runbook_20260520.md`（Phase B DB 重建 runbook）
- **本報告目的**: 依「**系統依賴順序**」逐一檢視 v6.0.0-FINAL 治權核心 60 個元件 + .env，產出問題記錄供憲章修訂依據

---

## 0. 摘要與審查方法

### 0.1 一句話結論

> *待審查完成後填入*

### 0.2 審查視角

採憲章 §0.0-G.0 Type-(n) 由 Type-(n-1) 授權之精神，**自底向上**（bottom-up）依依賴鏈順序審查。

### 0.3 審查範圍與順序

| Step | 範圍 | 元件數 | 預估時間 |
|---|---|---|---|
| **Step 0** | `.env` 環境變數設定 | 1 | 10-20 分鐘 |
| **階段 1.1** | `scripts/core/` 治權支援 | 9 | 1-2 小時 |
| **階段 1.2** | `scripts/core/` 五支落地鏈 | 5 | 2-3 小時 |
| **階段 2** | `scripts/maintenance/` audit 工具 | 21 | 3-4 小時 |
| **階段 3** | `scripts/ingestion/` 同步工具 | 25 | 3-4 小時 |
| **總計** | | **60 + .env** | **9-13 小時** |

### 0.4 審查標準（每元件）

依憲章 v6.0.0-FINAL 之治權結構，每元件檢查：

1. **治權位階**（依 §0.0-G.0）
2. **依賴關係**（是否符合 §8.4 治權邊界）
3. **強制契約對齊**（§9.1 / §9.2 / §9.9 等）
4. **檔頭元數據**（主權狀態 / 修訂歷程 / 對應 §14.7-X 研究）
5. **治權禁令遵守**（§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §0.0-F.3）
6. **T1/T2/T3 分層遵守**（§0.1.1）
7. **anti-leakage 遵守**（§8.5）
8. **§0.0-G 憲章先行紀律遵守**

### 0.5 問題記錄格式

每元件之記錄包含：

- ✅ **符合項**：列出對齊憲章之點
- ❌ **違規項**：需修補（標明憲章節依據）
- ⚠️ **待釐清項**：實證或文獻佐證待確認
- 💡 **憲章補強建議**：憲章未涵蓋之元件行為
- 📝 **修訂建議書條目**：累積成 v6.1.0 升版依據

---

## Step 0: `.env` 環境設定檢查

### 0.1 元件性質

- **類別**: 環境變數設定檔（不是 .py 程式）
- **治權位階**: 依憲章 §0.0-G.0，**不在 Type-3 實作層之列**
- **系統依賴順序**: **最上游**（所有 v6.0.0-FINAL 治權核心程式間接依賴）
- **載入機制**: `scripts/core/path_setup.py:load_dotenv()` / `scripts/core/db_utils.py:os.getenv()` / `scripts/core/finmind_client.py:load_dotenv()`

### 0.2 對應憲章節

| 憲章節 | 引用內容 |
|---|---|
| §3.1（25-Dim 路徑治權）| sovereign_sync_engine.py 為唯一授權同步載體；依 `.env` 設定 |
| §5.6.3（禁止硬編碼 PERFECT）| 設定值不應硬編碼於程式 |
| §6.7（SQL SSOT）| DB 連線設定依 `.env` |
| `scripts/core/db_utils.py v2.45`「DB env 檢查」 | 必要變數：DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD |
| `scripts/core/finmind_client.py v4.46`「§3.1 唯一接口」 | FINMIND_TOKEN |
| `CLAUDE.md` | AI 工具規則 |

### 0.3 檢查項清單

*以下為審計待執行項目*

#### 0.3.1 檔案存在性

- [ ] `.env` 是否存在於 PROJECT_ROOT
- [ ] `.env.example` 是否存在（範本檔）
- [ ] `.gitignore` 是否正確排除 `.env`（防泄密）

#### 0.3.2 必要變數齊全

- [ ] `DB_HOST`
- [ ] `DB_PORT`
- [ ] `DB_NAME`
- [ ] `DB_USER`
- [ ] `DB_PASSWORD`
- [ ] `FINMIND_TOKEN`

#### 0.3.3 與程式之對齊

- [ ] `path_setup.py` 之 `load_dotenv(PROJECT_ROOT / ".env")` 路徑正確
- [ ] `db_utils.py` 之 `get_db_connection()` 之 5 個變數名一致
- [ ] `finmind_client.py` 之 token 變數名一致

#### 0.3.4 安全性

- [ ] `.env` 不在 git 追蹤中
- [ ] `.env` 內無 hardcoded secret 之 placeholder（如 `your_token_here`）
- [ ] `.env.example` 不包含真實密鑰

#### 0.3.5 治權檔案引用

- [ ] `README.md` 是否提及 `.env` 設定
- [ ] `CLAUDE.md` 是否提及 `.env` 治權地位
- [ ] 憲章是否引用 `.env`（v6.0.0-FINAL 應有提及）

### 0.4 審查實證

執行時點：2026-05-20

#### 0.4.1 .env 與 .env.example 存在性檢查

```
.env         存在 (1642 bytes, 2026-05-15 08:36)
.env.example 存在 (1205 bytes, 2026-05-15 08:01)
```

| 檢查項 | 結果 |
|---|---|
| `.env` 存在於 PROJECT_ROOT | ✅ |
| `.env.example` 存在（範本檔）| ✅ |
| `.gitignore` 排除 `.env` | ✅（`git check-ignore .env` 回傳 `.env`）|
| `.env` 不在 git 追蹤中 | ✅（`git ls-files .env` 為空）|

**裁決**：存在性與 git 排除設定 PERFECT。

#### 0.4.2 必要變數對齊檢查

**.env 之 14 個變數**：
```
ENV
LOG_LEVEL
TZ
PROJECT_ROOT
MLFLOW_TRACKING_URI
DB_HOST              ← v6.0.0 治權核心依賴
DB_PORT              ← v6.0.0 治權核心依賴
DB_NAME              ← v6.0.0 治權核心依賴
DB_USER              ← v6.0.0 治權核心依賴
DB_PASSWORD          ← v6.0.0 治權核心依賴
GEMINI_API_KEY
FINMIND_TOKEN        ← v6.0.0 治權核心依賴
FRED_API_KEY
GITHUB_TOKEN
```

**v6.0.0 核心 6 個必要變數**（依 db_utils.py + finmind_client.py）：

| 變數 | .env 狀態 | 長度 |
|---|---|---|
| `DB_HOST` | ✅ SET | 9 |
| `DB_PORT` | ✅ SET | 4 |
| `DB_NAME` | ✅ SET | 5 |
| `DB_USER` | ✅ SET | 5 |
| `DB_PASSWORD` | ✅ SET | 5 |
| `FINMIND_TOKEN` | ✅ SET | 184 |

**裁決**：v6.0.0 核心 6 個必要變數全部就位。

**❌ 違規項：.env.example 缺漏關鍵變數**

`.env.example` 只列 5 個變數（FINMIND_TOKEN / FRED_API_KEY / DB_PASSWORD / DB_HOST / DB_PORT），**遺漏 `DB_NAME` 與 `DB_USER`**！

| 變數 | .env | .env.example | 缺漏？ |
|---|---|---|---|
| DB_HOST | ✅ | ✅ | - |
| DB_PORT | ✅ | ✅ | - |
| **DB_NAME** | ✅ | ❌ | **缺漏** |
| **DB_USER** | ✅ | ❌ | **缺漏** |
| DB_PASSWORD | ✅ | ✅ | - |
| FINMIND_TOKEN | ✅ | ✅ | - |

**影響**：後繼者依 `.env.example` 設定時將遺漏 `DB_NAME` 與 `DB_USER`，導致 `get_db_connection()` 之 G1 必要 env 檢查失敗。

#### 0.4.3 與 scripts/core/ 載入機制對齊

**path_setup.py v4.44**：
```python
PROJECT_ROOT_CALC = _SCRIPTS_DIR.parent
load_dotenv(PROJECT_ROOT_CALC / ".env")   # 載入路徑明確
```

**db_utils.py v2.45**：
```python
required_env = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
# 5 個變數全部用 os.getenv() 取得，連線失敗時拋出 RuntimeError
```

**finmind_client.py v4.46**：
```python
load_dotenv(_PROJECT_ROOT / ".env")
self.token = os.getenv("FINMIND_TOKEN")
```

| 對齊項 | 結果 |
|---|---|
| path_setup.py 載入路徑 | ✅ PROJECT_ROOT / .env |
| db_utils.py 5 個變數名一致 | ✅ DB_HOST/PORT/NAME/USER/PASSWORD |
| finmind_client.py token 變數名一致 | ✅ FINMIND_TOKEN |
| `.env` 之 `PROJECT_ROOT` 與 path_setup 計算之物理路徑一致 | ⚠️ 待主環境驗證（依憲章 §3.1 Boundary Drift 規範）|

**裁決**：基本對齊 PERFECT，僅 PROJECT_ROOT 一致性待主環境驗證。

#### 0.4.4 安全性檢查

| 安全項 | 結果 |
|---|---|
| `.env` 不在 git 追蹤 | ✅ |
| `.env` 在 .gitignore 中 | ✅ |
| FINMIND_TOKEN 長度合理 (184) | ✅（真實 token 約 150-200）|
| 無 hardcoded placeholder | ✅（無 `your_token_here` / `change_me` 等）|
| 程式碼中無泄漏密鑰 | ✅（無 hardcoded token）|

**裁決**：安全性 PERFECT。

#### 0.4.5 憲章引用檢查

**憲章 v6.0.0-FINAL 對 `.env` 之引用（10 處）**：

| 行號 | 引用內容 | 治權地位 |
|---|---|---|
| L31 | v5.4.19 將 `.env` Bootstrap Anchor 入憲 | 歷史紀錄 |
| L51 | 啟動序列 Step 0：`.env` 啟動錨點確認 | 操作流程 |
| L2317 | `.env` 是系統架構的 Bootstrap Anchor | **治權定位明文** |
| L2318 | `.env` **不得**成為 25 維路徑清單的 SSOT | **治權禁令** |
| L2319 | 密鑰只能存在於 `.env`，禁止寫入程式碼 | **安全治權** |
| L2329 | `.env` / path_setup.py 屬 pre-schema bootstrap | 階段定位 |
| L2334 | `path_setup.py` 是 Path SSOT，不由 `.env` 分散定義 | 治權邊界 |
| L2335 | `.env` 之 `PROJECT_ROOT` 必須與 path_setup 計算物理根目錄一致 | **Boundary Drift 規範** |

**裁決**：憲章對 `.env` 之治權地位明文且完整。

**CLAUDE.md 引用**（1 處）：
- L37：「不在 commit 中含 `.env` / credentials / API key」

**裁決**：CLAUDE.md 工具規則對齊 ✅。

**README.md 引用**：未提及 `.env` 設定。

**💡 補強建議**：README.md 應補入「啟動前先設定 `.env`」之指引。

### 0.5 Step 0 裁決

**綜合 PASS/WARN/FAIL 統計**：

- ✅ 符合項（10 項）：
  1. .env 存在於 PROJECT_ROOT
  2. .env.example 範本存在
  3. .gitignore 正確排除 .env
  4. .env 不在 git 追蹤
  5. v6.0.0 核心 6 個必要變數全部就位
  6. path_setup.py 載入路徑明確
  7. db_utils.py 5 個變數名一致
  8. finmind_client.py token 變數名一致
  9. 無 hardcoded placeholder
  10. 憲章對 `.env` 之治權地位明文且完整（10 處引用）

- ❌ 違規項（1 項）：
  1. **.env.example 遺漏 `DB_NAME` 與 `DB_USER`**（後繼者依範本設定時將遺漏，導致 G1 必要 env 檢查失敗）

- ⚠️ 待釐清項（1 項）：
  1. `.env` 之 `PROJECT_ROOT` 與 path_setup 計算物理根目錄一致性待主環境驗證

- 💡 憲章補強建議（2 項）：
  1. README.md 補入「啟動前先設定 `.env`」之指引
  2. 憲章可考慮新增「`.env.example` 必須與 `.env` 變數名完全一致」之治權條款（避免遺漏）

- 📝 修訂建議書條目（2 項）：
  1. **修補 `.env.example`**：補入 `DB_NAME=` 與 `DB_USER=`（Level 2 內容更新）
  2. **README.md 補強**：新增「環境變數設定」章節，說明 .env 啟動必要性

### 0.6 Step 0 完成標記

**狀態**：✅ Step 0 .env 環境設定檢查完成

**結論**：.env 設定本身 PERFECT（10 項符合），唯 1 項違規（.env.example 變數遺漏）需立即修補；其餘為 Level 2 補強。

**下一步**：階段 1.1.1 `scripts/core/path_setup.py`

---

## 階段 1.1: `scripts/core/` 治權支援（9 個）

依依賴順序：

### 1.1.1 `scripts/core/path_setup.py`

#### 1.1.1.1 元件性質

- **類別**：Type-3 治權支援（Path SSOT + Bootstrap Anchor 對齊）
- **版本**：v4.44（2026-05-15，最後更新）
- **治權位階**：依憲章 §3.2 / §0.0-G.0 — Type-3 實作層；同時為 Type-2 派生規則之直接落實檔（憲章 §2317-§2335 之 Path Sovereignty 治權邊界）
- **依賴**：`os` / `sys` / `time` / `pathlib.Path` / `datetime` / `dotenv.load_dotenv`（外部唯一第三方）+ `core.db_utils`（lazy，於 `_load_logging_hooks` 內）
- **下游依賴方**：`scripts/core/__init__.py`（hub）+ 27 個 `scripts/ingestion/*.py` + 9 個 `scripts/maintenance/*.py` + 18 個 `scripts/fetchers/*.py`（legacy）

#### 1.1.1.2 對應憲章節

| 憲章節 | 引用內容 |
|---|---|
| §2317 | 「`.env` 是系統架構的 Bootstrap Anchor」 |
| §2318 | 「`.env` **不得**成為 25 維路徑清單的 SSOT」 |
| §2329 | 「`.env` / `path_setup.py` 屬 pre-schema bootstrap」 |
| §2334 | 「`path_setup.py` 是 Path SSOT」（25 維） |
| §2335 | 「`.env` 之 `PROJECT_ROOT` 必須與 `path_setup` 計算物理根目錄一致」（Boundary Drift 規範） |
| §3.2 | path_setup 為治權路徑配發中心 |
| §0.0-G.0 | Type-3 由 Type-2 授權之精神 |

#### 1.1.1.3 檢查清單與審查實證

**A. 檔頭元數據對齊（§0.0-G.7）**

| 檢查項 | 結果 |
|---|---|
| 主權狀態明文 | ✅「PERFECT (憲法 v6.0.0 啟動治理對齊 + 同日 hub 補充相容)」 |
| 最後更新日期明文 | ✅ 2026-05-15 |
| 全修訂歷程列表 | ✅ v1.0〜v4.44 6 個版本完整 |
| 對應 §14.7-X 研究 | ⚠️ **未明文引用任何 §14.7-X 研究報告** |
| 八子節結構（§0.0-H）| ⚠️ **不適用**（path_setup 為 pre-schema bootstrap，不歸屬 §9.1〜§9.9 之 8 子節範式；憲章未規定 path_setup 須遵守 §0.0-H） |

**B. 治權位階與依賴邊界（§0.0-G.0 / §8.4）**

| 檢查項 | 結果 |
|---|---|
| 不繞過 `path_setup` 之 25 維 SSOT | ✅（不存在硬編碼路徑於程式邏輯內）|
| 載入 `.env` 後僅讀取 `PROJECT_ROOT`（不讀 DB env）| ✅ |
| `_load_logging_hooks` 採 lazy import core.db_utils | ✅（避免循環依賴）|
| BOOTSTRAP-DEFERRED 模式避免 DB schema 未建立時阻斷 | ✅（v4.43 已固化）|

**C. 強制契約對齊**

| 契約 | 結果 |
|---|---|
| 25 維路徑清單與憲章「25 維治理維度」一致 | ✅（`ALL_PATHS` 共 25 個元素，`get_evaluation_dir()` 為相容別名不計）|
| `_evaluate_anchor()` 之 MISSING/MISMATCHED 處理 | ✅（v4.43 已固化）|
| Sovereignty Status 三態 (PERFECT/WARNING/FAILED) | ✅ |
| `ensure_all_dirs()` 自癒契約 | ✅（mkdir(parents=True, exist_ok=True)）|

**D. 治權禁令遵守（§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §0.0-F.3）**

| 禁令 | 結果 |
|---|---|
| §0.1-A 無泛化「第一性原理」 | ✅ |
| §0.2-A 無泛化「八二法則」 | ✅ |
| §0.3-A 無泛化「康波週期」 | ✅ |
| §0.0-E.4 統合三概念之禁令 | ✅（path_setup 不涉及配置層）|
| §0.0-F.3 不引用 AI 工具規則 | ✅（無 AI 治權引用，符合）|

**E. T1/T2/T3 分層遵守（§0.1.1）**

| 檢查項 | 結果 |
|---|---|
| 程式內部嚴格 T1（事實）| ✅ |
| 註解 / 終端報表用語不誇張 | ✅（「🛡️」emoji 為純視覺裝飾，不違反 T1）|
| 無「主權」「絕對」誇張形容跨層 | ⚠️ 用語包含「主權」「絕對物理基準」「封印」等強烈詞彙，但屬路徑治權內部用語（§3.2 治權允許） |

**F. anti-leakage 遵守（§8.5）**

| 檢查項 | 結果 |
|---|---|
| 不寫入 feature_store | ✅（不適用）|
| 不訪問未來資料 | ✅（不適用）|

**G. §0.0-G 憲章先行紀律**

| 檢查項 | 結果 |
|---|---|
| 程式修改前是否先入憲（v4.44 標頭明示「憲法 v5.4.22 啟動治理對齊」）| ✅ |
| 修訂歷程版本對齊憲章 | ✅（v4.43 對齊 v5.4.21；v4.44 對齊 v5.4.22 + v6.0.0）|

#### 1.1.1.4 下游使用面 API 一致性審查（**重大發現**）

> **問題定位**：23 of 25 ingestion 治權核心程式 + 多支 maintenance/fetcher 程式 import `from core.path_setup import ensure_scripts_on_path`，但 `path_setup.py v4.44` **不定義此函式**。

**A. 下游引用之缺失 API 清單**

```bash
# 實證命令
$ python3 -c "from core.path_setup import ensure_scripts_on_path"
ImportError: cannot import name 'ensure_scripts_on_path' from 'core.path_setup'

# 進一步測試
$ python3 -c "import importlib.util; spec=importlib.util.spec_from_file_location(
    'm','scripts/ingestion/ingest_technical_data.py');
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
AttributeError: module 'path_setup' has no attribute 'ensure_scripts_on_path'
```

| API 名稱 | path_setup v4.44 | 引用方數量 | 影響範圍 |
|---|---|---|---|
| `ensure_scripts_on_path` | ❌ 未定義 | **23 個 ingestion + 9 個 maintenance/fetcher** = 32+ 處 | ingestion 治權核心 92% 失能 |
| `ensure_dirs_exist` | ❌ 未定義 | 1（fetchers/parallel_fetch.py）| legacy |
| `get_logs_dir` | ❌ 未定義（實際是 `get_log_dir`，單複數不一）| 1 | legacy |
| `get_outputs_dir` | ❌ 未定義（實際是 `get_output_dir`，單複數不一）| 2 | ingestion + fetcher |
| `get_checkpoints_dir` | ❌ 未定義 | 2 | ingestion + fetcher |

**B. ingestion 治權核心受影響清單（23/25）**

未受影響：`sovereign_sync_engine.py`（治權主檔，無依賴）/ `initialize_market_data.py`（未審）

受影響：`ingest_technical_data.py` / `ingest_derivative_sentiment_data.py` / `ingest_chip_data.py` / `ingest_fred_data.py` / `parallel_ingestion.py` / `ingest_cash_flows_data.py` / `ingest_macro_fundamental_data.py` / `ingest_international_data.py` / `backfill_from_gaps.py` / `ingest_sponsor_chip_data.py` / `search_finmind_datasets.py` / `ingest_block_trading.py` / `ingest_advanced_chip_data.py` / `ingest_extended_derivative_data.py` / `ingest_news_data.py` / `ingest_macro_data.py` / `ingest_month_revenue.py` / `ingest_event_risk_data.py` / `ingest_price_adj_data.py` / `ingest_derivative_data.py` / `ingest_missing_stocks_data.py` / `ingest_fundamental_data.py` / `ingest_total_return_index.py`

**C. 為何主線運行未失敗？**

關鍵：v6.0.0-FINAL **不直接調用 `python scripts/ingestion/ingest_*.py`**。所有 ingestion 由 `sovereign_sync_engine.py`（唯一授權同步載體）內部執行 FinMind/FRED API 抓取，不 import 任何 `ingest_*.py` 模組。

但這意味著：
1. 23 支 ingest_*.py 為「**治權核心內無法運行的死碼**」（Type-3 但 import fails）
2. 若有 ad-hoc 操作員依憲章 §0.0-A 嘗試獨立執行任一 ingest_*.py，將立即遇到 ImportError → AttributeError
3. Phase A 審計將其判為「治權核心」是基於檔案位置而非可運行性

#### 1.1.1.5 Step 1.1.1 裁決

**綜合 PASS/WARN/FAIL 統計**：

- ✅ 符合項（11 項）：
  1. 主權狀態 / 修訂歷程 / 版本標頭明文
  2. v1.0〜v4.44 6 個版本歷程完整
  3. 25 維路徑清單與憲章一致
  4. `_evaluate_anchor()` MISSING/MISMATCHED 處理完備
  5. BOOTSTRAP-DEFERRED 三態 (PERFECT/WARNING/FAILED) 完備
  6. `ensure_all_dirs()` 自癒契約完備
  7. `_load_logging_hooks` lazy import 避免循環依賴
  8. 不繞過 25 維 SSOT（無硬編碼路徑）
  9. 治權禁令全遵守（§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §0.0-F.3）
  10. anti-leakage（不適用，符合）
  11. §0.0-G 憲章先行紀律（v4.44 對齊 v5.4.22 + v6.0.0）

- ❌ 違規項（**重大** 1 項）：
  1. **下游使用面 API 殘缺**：v4.44 不定義 `ensure_scripts_on_path`、`ensure_dirs_exist`、`get_logs_dir`、`get_outputs_dir`、`get_checkpoints_dir`，但 23/25 ingestion 治權核心 + 多支 maintenance/fetcher 引用。此導致治權核心內 32+ 處 import-time failure（AttributeError）

- ⚠️ 待釐清項（2 項）：
  1. 標頭未明文引用 §14.7-X 研究報告（憲章未規定 path_setup 須引用）
  2. T1/T2/T3 用語「主權」「絕對」「封印」屬路徑治權內部用語（§3.2 允許）但跨層讀者可能誤解

- 💡 憲章補強建議（2 項）：
  1. **API 契約入憲**：憲章 §3.2 應明文 path_setup 對下游揭露之函式名單（25 維 + ALL_PATHS + ensure_all_dirs），並規定**新增/移除任何函式須先入憲**
  2. **死碼治權判別**：Phase A 程式碼治權審計應補入「**import-time runnability** 檢查」，不可只依檔案位置判定治權地位

- 📝 修訂建議書條目（3 項）：
  1. **【選項甲 - 補強 path_setup.py】** 補入 4 個 legacy API：`ensure_scripts_on_path(file)` / `ensure_dirs_exist()` / `get_logs_dir()` / `get_outputs_dir()` / `get_checkpoints_dir()` 為相容別名（與 `get_evaluation_dir` 相同模式）
  2. **【選項乙 - 重構 ingest_*.py】** 統一改為 `from core.path_setup import ensure_all_dirs` 並補上 sys.path bootstrap 樣板（23+ 處改動）
  3. **【選項丙 - 治權重分類】** Phase A 審計重判：將 23 支非可運行 ingest_*.py 自治權核心退至「歷史/實驗層」，由 sovereign_sync_engine 接管實際同步

**狀態**：⚠️ **Step 1.1.1 PASS（11 PERFECT）+ 1 重大違規（治權邊界 API 殘缺）**

**結論**：path_setup.py 本身 PERFECT 對齊憲章 §3.2 / §2317-§2335，但下游使用面契約不一致——23/25 ingestion 治權核心引用之 4 個 API（ensure_scripts_on_path 等）未定義。建議優先採選項甲（最小衝擊），同時導入「死碼治權判別」機制。

**下一步**：階段 1.1.2 `scripts/core/db_utils.py`

### 1.1.2 `scripts/core/db_utils.py`

### 1.1.3 `scripts/core/finmind_client.py`

### 1.1.4 `scripts/core/data_schema.py`

### 1.1.5 `scripts/core/core_universe_schema.py`

### 1.1.6 `scripts/core/feature_store_schema.py`

### 1.1.7 `scripts/core/model_metadata.py`

### 1.1.8 `scripts/core/migrate_stocks_config.py`

### 1.1.9 `scripts/core/__init__.py`

---

## 階段 1.2: `scripts/core/` 五支落地鏈（5 個）

依 §0.0-A.1〜A.5 順序：

### 1.2.1 `scripts/core/core_universe_builder.py` — §0.0-A.1

### 1.2.2 `scripts/core/feature_store_builder.py` — §0.0-A.2

### 1.2.3 `scripts/core/model_trainer.py` — §0.0-A.3

### 1.2.4 `scripts/core/prediction_engine.py` — §0.0-A.4

### 1.2.5 `scripts/core/portfolio_sizer.py` — §0.0-A.5

---

## 階段 2: `scripts/maintenance/` audit 工具（21 個）

*詳細列表待階段 1 完成後展開*

---

## 階段 3: `scripts/ingestion/` 同步工具（25 個）

*詳細列表待階段 2 完成後展開*

---

## 三、跨元件發現之共通問題

*待全部審查完成後填入*

---

## 四、憲章修訂建議書

*待全部審查完成後填入；列出建議憲章修訂之事項*

---

## 五、程式修補建議書

*待全部審查完成後填入；列出程式需修補之事項*

---

## 六、結論

*待全部審查完成後填入*

---

## 七、接續執行指引（2026-05-20 換機接力點）

### 7.1 當前狀態快照

**作業日期**：2026-05-20（Asia/Taipei）
**HEAD commit**：`4822795`
**Tag 已封存**：`v6.0.0-FINAL-audit-step0-env-remediation-completed`

**今日完成（11 個 commits）**：

| Commit | 範圍 |
|---|---|
| `4822795` | data_schema v2.13（[Zero Hardcoded Verdict] 補入 / 100% 合規）|
| `229967b` | path_setup v4.45 + __init__ v1.15 標頭 v6.0.0 對齊（選項乙）|
| `00494c6` | 隔離 Dockerfile（v5.x；CMD 指向不存在路徑）|
| `788bdd2` | 隔離 create_table.sql（§9.1-B 禁止來源）|
| `1cd9b23` | 隔離 3 root .txt |
| `ac03ac3` | 隔離 8 root .md |
| `15021fa` | 隔離 16 .py（3 root + 13 maintenance；路徑丙 Y）|
| `4510e93` | requirements.txt 補 requests（治權核心 6 檔依賴）|
| `cdba6c9` | Step 1.1.1 path_setup.py 審查記錄 |
| `08af007` | Step 0 .env 修補包（憲章 §0.0-I.8 + .env.example + README）|
| `4a29b3c` | Step 0 .env 環境設定檢查 |

**v6.0.0-FINAL 治權核心邊界縮容**：

| 範圍 | 變動前 | 變動後 |
|---|---|---|
| 根目錄 .md | 10 | 2（CLAUDE.md / README.md）|
| 根目錄 .txt | 4 | 1（requirements.txt）|
| 根目錄 .py | 3 | 0 |
| 根目錄 .sql | 1 | 0 |
| 根目錄 Dockerfile | 1 | 0 |
| scripts/maintenance/ | 21 | 8 |
| 隔離區總檔數 | 64 | 82 |

### 7.2 換機接力第 1 步：執行 `data_schema.py --init --force`

**接續執行起點**：

```bash
cd /home/hugo/project/stock_backend
git pull origin master                                    # 取得 11 個 commits + 修補
.venv/bin/python3 scripts/core/data_schema.py --init --force
```

**預期執行結果**（v2.13）：

1. **Step A：API Contract First Probe**
   - 探測 FinMind 10 個 dataset 之欄位契約（須網路通；無 token 即離線回退）
   - 探測 FRED `DFF` series 契約
   - PASS / WARN / FAIL 三態動態判定

2. **Step B：DDL 主權重鑄**（API probe 通過後）
   - 重建 13 張表（2 infra + 10 FinMind raw + 1 FRED）
   - 雙引號封裝 / VARCHAR(255) 防禦性寬容 / `idx_*_date` 索引

3. **Step C：報告輸出**
   - PERFECT ALIGNMENT / WARNING / FAILED
   - sys.exit(0/1) 依結果

**離線復原指令**（若 FinMind/FRED API 失效）：

```bash
.venv/bin/python3 scripts/core/data_schema.py --init --force --skip-api-contract
```

### 7.3 換機接力第 2-N 步：後續審計順序

依憲章 §二 全量維運矩陣序列：

| 順位 | 程式 | 對應 Step | 動作 |
|---|---|---|---|
| **下一支** | `scripts/core/core_universe_schema.py` | 憲章 Step 2B | 階段 1.1.3 審查 |
| 後續 | `scripts/core/db_utils.py` | 憲章 Step 2C | 階段 1.1.4 審查 |
| 後續 | `scripts/core/finmind_client.py` | §3.2 橫切 library | 階段 1.1.5 審查 |
| 後續 | `scripts/core/feature_store_schema.py` | feature store schema | 階段 1.1.6 審查 |
| 後續 | `scripts/core/model_metadata.py` | 模型元數據 | 階段 1.1.7 審查 |
| 後續 | `scripts/core/migrate_stocks_config.py` | 一次性遷移 | 階段 1.1.8 審查 |
| 後續 | `scripts/core/__init__.py` | hub | 階段 1.1.9 審查（部分已於 1.1.1 連帶完成）|
| 階段 1.2 | 五支落地鏈（core_universe_builder → portfolio_sizer）| §0.0-A.1-A.5 | 5 支 |
| 階段 2 | scripts/maintenance/（8 殘留治權）| §3.2A 橫切稽核 | 8 支 |
| 階段 3 | scripts/ingestion/（25 支）| §3.1 序列模組 | 25 支 |

### 7.4 接續執行 SOP（「中度確認」工作流程）

依 `/home/hugo/.claude/plans/imperative-wiggling-key.md` 之中度確認模式：

```
1. Claude 讀程式 → 對照 8 項檢查面 → 寫初步審計區塊到報告
2. Claude 列出本支之 ✅ 符合 N 項 / ❌ 違規 M 項 / ⚠️ 待釐清 K 項 / 💡 補強 J 項
3. 如有違規：列出修補選項（甲/乙/丙 + 衝擊比較）
4. 暫停，等待使用者裁決：
   - 「下一支」→ 維持目前審計記錄，進入下一支
   - 「採選項甲/乙/丙」→ Claude 寫入修訂建議書，不執行修補
   - 「先修補再下一支」→ 切換至「重度確認」模式，先實作修補與驗證
   - 「補充 / 質疑」→ Claude 補強審計或重做某項
5. 報告寫入此檔
6. 每階段（1.1 / 1.2 / 2 / 3）末做一次 commit + tag
```

### 7.5 已建立之治權治理模板（每元件審計時對照）

**完整 8 項檢查面**（依憲章 v6.0.0）：

1. **治權位階**（§0.0-G.0 Type-3 實作層）
2. **依賴關係**（§8.4 治權邊界 / §3.1 序列模組 vs §3.2 橫切 library）
3. **強制契約對齊**（§9.1 / §9.2 / §9.9 等 + §6.7 SQL SSOT）
4. **檔頭元數據**（憲章 §一 1. Authority of Historical Truth；CONSTITUTION_VER 模組常數 + self.constitution_ver 屬性同步至 v6.0.0；§5.6.3 [Zero Hardcoded Verdict]）
5. **治權禁令遵守**（§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §0.0-F.3 共 5 套禁令）
6. **T1/T2/T3 分層遵守**（§0.1.1）
7. **anti-leakage 遵守**（§8.5）
8. **§0.0-G 憲章先行紀律遵守**（程式版本與憲章一致）

**標準修補模板**（依 Step 1.1.1 / 1.1.2 已驗證之路徑）：

- **小修補（甲）**：補 `CONSTITUTION_VER` + `TOOL_VER` 模組常數 + 修訂歷程 v6.0.0 條目 + cosmetic v5.4.x → v6.0.0
- **中修補（乙）**：甲 + 連帶 hub `__init__.py` 同步更新
- **完整修補（v0.X.X 升級）**：甲乙 + 補入 [Zero Hardcoded Verdict] 核心定義第 5 條
- **重大違規（如 path_setup 下游 API 殘缺）**：留待後續批次（修補建議書）

### 7.6 已知 pending 修補建議書條目

| # | 項目 | 來源 |
|---|---|---|
| 1 | path_setup 下游 5 個 API 殘缺（ensure_scripts_on_path / ensure_dirs_exist / get_logs_dir / get_outputs_dir / get_checkpoints_dir）影響 23/25 ingestion 治權核心 | Step 1.1.1 重大違規 |
| 2 | .env.example 缺 DB_NAME / DB_USER（已修補於 Step 0 之 §0.0-I.8 入憲）| — |
| 3 | requirements.txt 補 requests（已修補於 commit `4510e93`）| — |
| 4 | scipy / pandas_ta / pytest 未列於 requirements（留待後續批次 minor）| 隔離 16 檔之 audit |
| 5 | maintenance 8 個 PERFECT 標頭但憲章 0 引用之灰色檔已隔離（驗證後可刪除）| 路徑丙 Y |
| 6 | scripts/fetchers/ 為 v5.x legacy，sovereign_sync_engine 已取代，待後續批次處理 | Step 1.1.1 觀察 |
| 7 | scripts/pipeline/portfolio_optimizer.py 為 legacy（讀 §9.1-B 禁止之 stock_forecast_daily 表）| commit `788bdd2` 觀察 |

### 7.7 接續換機檢核 SOP

1. `cd /home/hugo/project/stock_backend && git pull origin master` — 取得 11 個 commits
2. `.venv/bin/python3 scripts/core/path_setup.py` — 確認 PERFECT（v4.45 對齊 v6.0.0）
3. `.venv/bin/python3 scripts/core/data_schema.py --init --force` — **執行此命令並觀察 PASS/WARN/FAIL；本接續第 1 步**
4. 將執行結果（PERFECT / WARNING / FAILED 之終端輸出）回報，再進入階段 1.1.3 `core_universe_schema.py` 審查

### 7.8 強制原則：外部資料 ↔ Audit 程式配對

**使用者明示之治權原則**：

> **所有從外部抓取的資料都要有對應的 audit 程式來確認正確性。**

**對齊憲章 §一 4. [Supply Chain Sovereignty]**：

- 「數據契約必須 100% 鏡像對齊外部數據源」
- 「全譜供應鏈合規稽核必須交叉比對 API + DB 實況（v1.17 補強），不可僅查 API」

#### 7.8.1 現行外部資料 ↔ 同步 + Audit 對映

| 外部來源 | 13 張表 | 同步載體（依 §3.1）| Audit 程式（依 §3.2A）|
|---|---|---|---|
| **FinMind API** | TaiwanStockPrice / PriceAdj / PER / InstitutionalInvestorsBuySell / MarginPurchaseShortSale / Shareholding / FinancialStatements / MonthRevenue / Dividend / Info（10 張）| `sovereign_sync_engine.py v1.15`（唯一授權同步載體）| `audit_supply_chain.py v1.18`（schema + DB + logs 三重對齊）|
| **FRED API** | FredData（1 張）| `sovereign_sync_engine.py v1.15`（透過 `--source fred`）| `audit_supply_chain.py v1.18` |
| **Infrastructure** | pipeline_execution_log / data_audit_log（2 張）| `data_schema.py --init --force`（一次性建立）| `audit_supply_chain.py v1.18`（含 lifecycle 驗收 + record_lifecycle 回寫驗證）|

**現行配對覆蓋率**：13/13 = **100%**（皆由 `audit_supply_chain.py` 統一驗收 API + DB + logs 三重對齊）。

#### 7.8.2 衍生治權層 audit 矩陣

依憲章 §3.2A 橫切稽核工具 6 員：

| Audit 程式 | 對應外部資料 / 衍生資料 |
|---|---|
| `audit_supply_chain.py` | **13 張 raw 表**（FinMind 10 + FRED 1 + infra 2）API ↔ DB ↔ logs |
| `audit_core_universe.py` | `core_universe_membership` JOIN `core_universe_snapshot`（衍生 §6.7 治權表）|
| `audit_downstream_readiness.py` | `feature_set_*` / `prediction_run` / `model_registry` 升版 readiness |
| `audit_leakage.py` | feature/model/prediction 之 §8.5 anti-leakage（時間邊界）|
| `audit_source_availability.py` | Core 150 strict source availability（§14.7-L mismatch）|
| `audit_doctrine_compliance.py` | §0 四大支柱 doctrine compliance |

**衍生資料覆蓋率**：100%（每一治權衍生表 / 流程皆有對應 audit）。

#### 7.8.3 未來新增外部資料來源之強制要求

當未來新增任何外部資料來源（如新增 Bloomberg / S&P / 證交所 SOP / 其他 API），**必須同步建立**：

1. **`data_schema.py` 中新增 DDL 條目**（API contract first probe + 雙引號封裝 DDL）
2. **`sovereign_sync_engine.py` 中新增同步邏輯**（§7 三層防禦：節流 + 退避 + DB resume）
3. **`audit_supply_chain.py` 中新增驗收項**（API + DB + logs 三重對齊）
4. **憲章 §3.1 / §3.2 / §3.2A 中登錄新模組**（如另立獨立 audit 程式時）
5. **任一外部變數須先入 §0.0-I.8**（依憲章先行紀律）

**違反裁決**：未對應 audit 程式之外部資料 = 違反 §一 4. [Supply Chain Sovereignty]，**判為非法資料源**，禁止寫入治權層。

#### 7.8.4 後續逐元件審計時須核對之問題

每審查一支 ingestion / fetcher 程式時，必須交叉確認：

- [ ] 該程式同步之資料表是否在 `DATASET_REGISTRY`（憲章 13 張表基線）？
- [ ] 對應 audit 程式是否能正確驗收此資料（API 欄位 + DB 列數 + lifecycle log）？
- [ ] 是否有 row count / freshness / coverage 之動態檢驗（§5.6.3 動態判定）？
- [ ] 是否寫入 `pipeline_execution_log` + `data_audit_log` 雙日誌（§一 2. [Hybrid Observability]）？

任一項 ❌ 即為**重大違規**，需立即補配 audit。

#### 7.8.5 階段 3 ingestion 25 支審計檢核項目

階段 3 審計每一支 ingestion 程式時，須**在審計報告中明示其對應之 audit 程式**：

| ingestion 程式（25 支）| 同步資料表 | 對應 audit 程式 | 已配對？ |
|---|---|---|---|
| `sovereign_sync_engine.py`（主載體）| 全 13 張 | `audit_supply_chain.py` | ✅ |
| `ingest_*.py` 23 支（皆 ImportError 死碼，Step 1.1.1 已揭露）| — | — | ⚠️ 因死碼無實際同步 |
| `initialize_market_data.py` | 初始化 | （待 1.1.X 審查時釐清）| ⏳ |
| `parallel_ingestion.py` | parallel 載體 | （待釐清）| ⏳ |

待逐一審計後填入完整對映表（§ 三 跨元件發現之共通問題章）。

### 7.9 hub 執行時點之治權建議（治權詮釋；非憲章條款）

**性質聲明**：本子節為**治權詮釋**之記錄；不修改憲章 v6.0.0 任何條款；不取消 §3.2 hub「**任何階段可執行**」之橫切性質（憲章 L47 / L2500）。本記錄僅補入「**hub 治權閉環時點之最佳時機**」之詮釋作為未來執行參考。

#### 7.9.1 4 層稽核 verdict 邏輯（依憲章 L2436）

`core/__init__.py` 之 `run_sovereign_hub_audit()` 執行 4 層動態稽核：

| Layer | 檢查項 | 治權依據 |
|---|---|---|
| A | import sanity（path_setup + db_utils）| §3.1 / §3.2 模組可正常 import |
| B | 25 維路徑接口維度 | §3.2 / §一 3. [Boundary Integrity] |
| C | DB 連線狀態 | §6.7 / §3.2 |
| D | §6.7 SQL 查詢（`core_universe_membership` JOIN `core_universe_snapshot`）| §6.7 SQL SSOT |

**Verdict 動態計算**（依憲章 §5.6.3 + L2500）：

```text
hub PERFECT  ⟺  Layer A ∧ Layer B ∧ Layer C ∧ Layer D 皆 PASS
hub WARNING  ⟸  Layer D 0-row（DB 連線正常但 §6.7 0 rows；bootstrap state）
hub FAILED   ⟸  任一 Layer 例外（如 §6.7 query 抛 UndefinedTable）
```

#### 7.9.2 Layer D PASS 之最早門檻 = Step 4B 之後

| 時點 | `core_universe_membership` 狀態 | Layer D | 整體 verdict |
|---|---|---|---|
| Step 0 → Step 2 之間 | 表不存在 | ❌ FAILED（UndefinedTable）| FAILED exit 1 |
| Step 2B → Step 4B 之間 | 表存在但 0 committed rows | ⚠️ WARNING（0-row bootstrap）| WARNING exit 0 |
| **Step 4B 之後**（commit）| 表存在 + ≥ 1 committed rows（如 150 支）| ✅ **PASS** | **PERFECT exit 0** |

→ Layer D 達成 PASS 之最早時點 = **`core_universe_builder.py --commit` 之後**。

#### 7.9.3 hub 治權閉環時點之治權詮釋

依「**4 層 PASS / PERFECT**」為治權標準，hub 之**最佳執行時點**：

| 治權層級 | 時點 | 預期 verdict | 治權意義 |
|---|---|---|---|
| **🥉 早期 sanity check** | Step 1 後 | FAILED（預期）| import + path 早期診斷；接受 Layer C/D 不 PASS |
| **🥈 早期不 FAILED** | Step 2B 後 | WARNING exit 0 | 表結構 ready，但 0-row bootstrap |
| **🥇 治權閉環 PERFECT** | **Step 4B 後（commit）**| ✅ PERFECT | **最早完整 4 層 PASS** |
| **💎 audit-validated PERFECT** | Step 4C 後 | ✅ PERFECT + audit 驗證 | **最完整治權閉環** |

#### 7.9.4 治權邊界（保持與憲章 §3.2 一致）

本治權詮釋**不**：

- ❌ 不修改憲章 §3.2 hub「任何階段可執行」之橫切性質
- ❌ 不取消「hub 早期執行得到 WARNING / FAILED 為合法 bootstrap state」之既有規則
- ❌ 不將 hub 升級為主執行序列之必要步驟
- ❌ 不修改 §二 維運矩陣之 HUB 列定位

本治權詮釋**僅**：

- ✅ 詮釋「**hub PERFECT verdict 之最早門檻 = Step 4B 後**」
- ✅ 詮釋「**hub 完整治權閉環時點 = Step 4C 後**」
- ✅ 提供「**hub 執行時點選擇之治權建議**」記錄

#### 7.9.5 未來升憲提案候選（v6.0.0-patch / v6.1.0）

若未來累積實證證明本治權詮釋值得入憲，可考慮：

- §二 維運矩陣補入「Step 4.5 治權閉環確認」之 hub 建議時點（仍保留 §3.2 任何階段可執行性質）
- __init__.py [Canonical Execution Order] 核心定義第 3 條補入「治權閉環確認步」之治權建議
- §3.2 hub 子節補入「最佳執行時點」之治權詮釋

→ **本次不執行**升憲；僅記錄治權詮釋於本審計報告 §7.9。

---

**本報告為「逐元件治權合規審計」之主檔。每完成一個 step 即 commit + tag，可隨時暫停與繼續。**
