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

*待 Step 0 完成後啟動*

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

**本報告為「逐元件治權合規審計」之主檔。每完成一個 step 即 commit + tag，可隨時暫停與繼續。**
