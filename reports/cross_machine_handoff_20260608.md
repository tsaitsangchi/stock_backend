# 跨機交接記錄 — 專案執行階段現況(2026-06-08)

**用途**:本檔記錄本專案當前執行階段現況,供**換另一台電腦接續**時快速對齊。全部現況為實測事實(git / DB query / 檔案 header / env,§一.10 source-traceable)。

---

## 0. 一句話現況

通用 ingester(generic auto-schema)+ 完整 82-表 schema 字典(逐欄中文定義、81/82 live-API 驗證)已完成並封存上 GitHub;**DB 目前為空(0 表)**,新機接續之主要待辦 = **從零重建(§14.7-DD 12-PHASE)** 把全市場全史資料經 generic ingester 落地。

---

## 1. Git 現況(取得方式)

| 項目 | 值 |
|---|---|
| remote | `https://github.com/tsaitsangchi/stock_backend.git` |
| branch | `master` |
| HEAD | `280fb2f`(與 origin/master 同步 0/0,工作樹 clean) |
| 最新封存 tag | `v6.33.3-section14-7-DJ-82table-live-api-verified-percolumn-zhdefs-20260608` |

**新機取得**:
```bash
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git checkout master            # 或 git checkout v6.33.3-...(該封存點)
git log --oneline -6           # 確認 HEAD=280fb2f
```

**本 session commit 鏈(自舊封存點起)**:
```
3b63213 §14.7-DJ FRED-generic + DATASET_REGISTRY 徹底退役 → 零白名單全表通用
1f90d09 FRED 2-table generic 修正 + 字串 floor VARCHAR 100→255 + infra log ≥255
b2b0090 4 處 ⚡ live-API 型別修正 + 標 8 ᵈ
bb2fb58 8 ᵈ → 4 ᵈ(攻克 USD/WTI/2330/2603)
c29b6d6 4 ᵈ → 1 ᵈ(攻克 GovernmentBankBuySell/MarketValueWeight/InterestRate)
280fb2f 全 82 表逐欄補中文定義(2-col → 3-col)
```

---

## 2. DB 現況 ⚠️ 空(0 表)

- `public` schema 表數 = **0**(本 session 經用戶授權 `DROP` 全部 table;實測 information_schema 確認 0)。
- **無 DB dump**;新機若要資料 → 必須**從零重建**(下方 §6 待辦)。
- schema 不需手建:generic auto-schema 於首次 ingest 自動建表(零白名單,任意 dataset 經 `--dataset X` 即建)。

---

## 3. 本 session 完成內容(schema 字典軸)

1. **零白名單、全表通用 ingester**(§14.7-DJ):退役 `DATASET_REGISTRY` schema 白名單 → `core/generic_schema.py` 為 §0.0-I 單一引用源;`sovereign_sync_engine._target_datasets` 之 `if dataset: return [dataset]` 任意 dataset 放行;`data_schema.py` 僅留 `INFRA_TABLE_SCHEMAS`(2 infra log)。
2. **FRED 全 generic 化**:`FredData`(`sync_fred`,4 series,5 欄)+ `fred_series`(`fetch_fred_data`,24 series,3 欄)皆 generic,PK `(series_id, date)`(KEY_CANDIDATES 將 series_id 前置於 date)。
3. **字串型別下限 VARCHAR 100 → 255**(用戶 directive):`generic_schema.MIN_VARCHAR=255`,所有 generic-built 字串欄 ≥255;infra log 2 表同步 255。
4. **完整 82-表 schema 字典 + 逐欄中文定義**:`reports/finmind_full_table_schema_20260608.md`(SSOT,3-col:欄位|型態大小|中文定義,650+ 欄 0 漏);**81/82 用實際 FinMind/FRED API live 驗證**(僅 `ExchangeRate` FinMind 端回 0-rows 為建檔值);4 處 ⚡ sample-dependent 型別修正已套用。

---

## 4. 關鍵檔案 + 現行版本(本 session 改動)

| 檔案 | 版本 | 角色 |
|---|---|---|
| `scripts/core/generic_schema.py` | **v1.2** | generic auto-schema SSOT(infer_schema/detect_keys/ensure_table/upsert);MIN_VARCHAR=255 |
| `scripts/ingestion/sovereign_sync_engine.py` | **v1.25** | 全市場全史 sync 引擎;`_generic_ingest` + FRED generic;零白名單 |
| `scripts/core/data_schema.py` | **v2.23** | `INFRA_TABLE_SCHEMAS`(2 infra)+ `get_dataset_columns/get_dataset_keys`(DB-derived) |
| `scripts/fetchers/fetch_fred_data.py` | **v3.3** | FRED `fred_series` generic ingest(24 series) |
| `scripts/ingestion/finmind_generic_ingest.py` | **v0.3**(修訂歷程;標題行仍 v0.2 待同步) | 單 dataset 探索 ingester(import generic_schema 繼承 floor) |
| `scripts/ingestion/initialize_market_data.py` | **v1.22** | 建 2 infra log 表 + legacy stocks 表(infra log 已升 255) |
| `reports/系統架構大憲章_v6.1.0.md` | v6.1.0 | 治權 SSOT(§14.7-DJ §二 含 82-表字典) |
| `reports/finmind_full_table_schema_20260608.md` | — | **完整 per-column schema + 中文定義 SSOT** |
| `reports/generic_ingester_data_dictionary_20260608.md` | — | 13 feature-input 表詳細字典(companion) |

---

## 5. 治權入憲(雙層治權鎖)

- **§14.7-DJ**(主憲章)+ **CLAUDE.md §一.22**:Pure-Generic Ingestion 退役 DATASET_REGISTRY 白名單 + 82-表資料字典。
- **§14.7-DD**(主憲章)+ **CLAUDE.md §一.14**:Tree-family 從零重建 12-PHASE 序列(從零重建之治權範本)。
- **§14.7-DC / §一.13**:Source-Pure Doctrine(imputed 股排除核心、no AI 幻像值)。
- 字串 floor 255 為用戶 2026-06-08 directive;記於 generic_schema v1.2 + §14.7-DJ + CLAUDE.md §一.22。

---

## 6. 待辦 / 下一步(新機接續)⚠️

1. **(主要)從零重建 §14.7-DD 12-PHASE**:DB 空 → 依 PHASE 0-11 重建(環境→schema DDL→genesis 名冊→bootstrap 宇宙→全市場全史 sync + FRED→raw audit→最終宇宙→feature store→宇宙 gate→feature audit→模型訓練→模型驗證)。全市場全史 sync(PHASE 4)估 ~6-10 hr。
   - ⚠️ **≥30 min 長跑 → 必須 `caffeinate -dimsu` 包裹啟動**(最高運行準則,防機器睡眠殺進程)+ **§二.6 SHMM**(N≥3 Monitor heartbeat + sentinel + watchdog)+ **§一.12 每 5 分鐘回報**。先驗 AC 插電(`pmset -g batt`)。
   - ⚠️ B1-B4 code-level blockers 見 CLAUDE.md §一.14(SPEC=37 gate、feature audit SPEC list、base trainer 先建表、T_CZ-6 人工裁決)。
   - **注意 SPEC 已演進**:CLAUDE.md §一.21 v0.6 記 `feature_set_v0.6`(37→32 features,294 純核心);與本 session 的 82-表 raw schema 軸正交(raw 表 vs feature set)。
2. **1 個 ᵈ 待解**:`ExchangeRate`(FinMind 全參數回 0-rows,疑 deprecated/空集)— 非缺表,schema 為建檔實打值;新機可再試 FinMind 是否恢復供應。
3. **finmind_generic_ingest.py 標題行 v0.2 → v0.3 同步**(docstring/修訂歷程已 v0.3,標題行 cosmetic 待補)。
4. 既有 model artifacts(若重建後)須依 §一.12 + §二.6 retrain;未授權不 auto-run。

---

## 7. 新機環境前置(§二.7 / §0.0-I.9-10)⚠️ 必做

1. **OS 原生依賴**(xgboost/lightgbm 需 OpenMP;psycopg2 需 PG client headers):
   - macOS:`brew install libomp postgresql@17`
   - Linux:`sudo apt-get install -y libgomp1 libpq-dev`
2. **venv 重建**(本機有 `venv`[tree+外部權重] + `venv_fm`[foundation torch];新機需各自重建並 `pip install -r`):
   ```bash
   python3 -m venv venv && venv/bin/pip install -r requirements.txt   # 檔名以 repo 實際為準
   ```
3. **`.env` 設定**(本機已有,新機須重建;**勿 commit .env**):
   - `FINMIND_TOKEN`(sponsor tier;本 session 實測有效)、`FRED_API_KEY`(32 char hex)
   - `DB_HOST/DB_NAME/DB_USER/DB_PASSWORD/DB_PORT`(PostgreSQL 連線)
   - ⚠️ **`PROJECT_ROOT` 必須改為新機物理路徑**:本機記為 `/home/hugo/project/stock_backend`(macOS 經 symlink),新機須對齊該機 `os.path.realpath` 後之實際路徑(§0.0-I.10;`path_setup.py` 用 realpath 比對)。
4. **PostgreSQL**:建 role/db(對齊 .env);DB 起始為空,由重建流程建表。
5. **Import smoke test(通過才進後續)**:
   ```bash
   venv/bin/python -c "import psycopg2,pandas,polars,numpy,requests,sklearn,xgboost,lightgbm; print('✅ all imports OK')"
   ```
   失敗 → 先補 OS 層依賴(步驟 1)再執行任何 sync/audit。

---

## 8. 核心原則提醒(跨機不可違)

- **零 DATASET_REGISTRY 白名單**:全部表由 generic ingester 建;不得復辟「只認 N 表」白名單(用戶 explicit directive ×4)。
- **資料最小單位=日**(T_DI-7):intraday(tick/分K/5秒)raw 不入庫;intraday 衍生值以日級 derive 儲存。
- **Source-Pure / No AI 幻像值**(§14.7-DC / §一.10):imputed 值無 FinMind/FRED API source → 該股排除核心;任何數據須 trace 回 (a)程式輸出/(b)DB query/(c)API response。
- **No 手動補資料**(memory: no_manual_data_fill):永不手動 UPDATE/INSERT 補值;只 DELETE 過期列或改 writer code 重 build。
- **大小寫=API 逐字**(§14.7-CC):表名/欄位名與 FinMind/FRED API 完全一致(generic 雙引號封裝保留)。
- **對話一律繁體中文**(CLAUDE.md §一.18 / 全域 §0)。
- **commit/push 需用戶明示授權**(§二.2);**≥30 min 長跑必 caffeinate + SHMM + 5-min 回報**。

---

## 9. 重要參考檔索引

| 檔案 | 內容 |
|---|---|
| `reports/finmind_full_table_schema_20260608.md` | **82-表完整 per-column schema + 中文定義 SSOT**(本 session 主產物) |
| `reports/generic_ingester_data_dictionary_20260608.md` | 13 feature-input 表詳細字典 |
| `reports/系統架構大憲章_v6.1.0.md` §14.7-DJ §二 | 憲章內 82-表字典 + long-format(財報 17 / 資產負債 101)type 科目枚舉 |
| `reports/系統架構大憲章_v6.1.0.md` §14.7-DD | 從零重建 12-PHASE 治權範本 |
| `CLAUDE.md` | AI 協作工具規則 SSOT(§一.13/14/21/22 等) |
| `reports/tree_based_from_zero_build_runbook_20260531.md` | tree-family 從零重建 runbook(若存在) |

---

**本檔事實來源(§一.10)**:git(HEAD 280fb2f / tag v6.33.3 / clean / synced 0/0)、DB query(public schema 0 表)、檔案 header(各版本)、env(FINMIND_TOKEN/FRED_API_KEY SET / PROJECT_ROOT=/home/hugo/...)、本 session commit 鏈,皆 2026-06-08 實測。
