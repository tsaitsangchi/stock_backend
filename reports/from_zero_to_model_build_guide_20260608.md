# 從零到模型產生與驗證 — 全系統重建 Runbook（2026-06-08 · §14.7-DJ pure-generic 版）

**文件性質**：本檔為「刪除全部環境與資料後,從零完整重建本系統」之**可執行 runbook**。涵蓋:環境 → 資料產生(generic ingester)→ 資料 audit(DB↔API)→ 核心股 → 特徵 → 模型訓練 → 模型驗證。
**治權對齊**：主憲章 **§14.7-DD**(Tree-Family From-Zero → Model → Validation 12-PHASE 範本)+ **§14.7-DJ**(Pure-Generic Ingestion,2026-06-08)為治權 SSOT;本檔為 implementation reference。
**前序**：supersede `from_zero_to_model_build_guide_20260602.md`(整合 2026-06-08 §14.7-DJ generic 改動:raw 表改 generic 於 sync 時自動建、FRED 2 表 generic、字串 floor 255、82-表 live-API catalog)。
**誠實聲明（§一.10）**：時間預算為 **ESTIMATE**(實跑以各程式 stdout `Total elapsed` 為準);PHASE 4 全市場同步 2026-06-01 實測 ~5h19m / ~81M rows。**raw schema 數字** trace 自 `reports/finmind_full_table_schema_20260608.md`(82 表,81/82 用實際 FinMind/FRED API live 驗證);**feature/core 數**(v0.5 37/397 vs §一.21 v0.6 32/294)以 code 當前 `feature_store_builder.FEATURE_DEFINITIONS` 為準,執行時 verify(本檔不硬寫,避免幻像)。

---

## §0 重大變更(2026-06-08 §14.7-DJ;與 2026-06-02 版差異)

1. **零 DATASET_REGISTRY 白名單**:FinMind/FRED raw 表**不再由 explicit DDL 預建**,改由**通用 ingester**(`core/generic_schema.py` v1.2 = §0.0-I 單一引用源)於**首次 sync 時自動建表**(看 API 回應推型別/建表/upsert)。`data_schema.py --init` 現**只建 2 個 infra log 表**(`INFRA_TABLE_SCHEMAS`),**非 11 raw 表**。
2. **raw 表在 PHASE 2/2b/4 sync 時才出現**(generic 建);PHASE 1 後 DB 只有 infra + stocks + universe-gov + feature-store schema 表,**不要預期 11 raw 表已存在**。
3. **FRED 2 表 generic**:`FredData`(`sync_fred`,4 series:DFF/UNRATE/T10Y2Y/VIXCLS,5 欄)+ `fred_series`(`fetch_fred_data`,24 series,3 欄),PK 皆 `(series_id, date)`(generic detect_keys,`KEY_CANDIDATES` series_id 前置於 date)。
4. **字串型別下限 VARCHAR 100→255**(generic_schema v1.2 MIN_VARCHAR;數字 ≥ NUMERIC(20,6);值超界自動加大)。
5. **完整 82-表 schema(逐欄中文定義 + 型別)SSOT = `reports/finmind_full_table_schema_20260608.md`**(FinMind Taiwan 67 + 非Taiwan 13 + FRED 2;81/82 用實際 API live 驗證,僅 `ExchangeRate` FinMind 端回 0-rows 為建檔值)。表名/欄位大小寫與 API 逐字一致(§14.7-CC)。

---

## §1 環境建立（PHASE 0）

```bash
# 1. OS 原生依賴(§0.0-I.9)
sudo apt-get install -y libgomp1 libpq-dev postgresql postgresql-contrib   # Linux
# macOS: brew install libomp postgresql@17

# 2. PostgreSQL role + db
sudo -u postgres psql -c "CREATE ROLE <user> LOGIN PASSWORD '<pw>';"
sudo -u postgres psql -c "CREATE DATABASE <db> OWNER <user>;"

# 3. venv + 套件
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

# 4. .env(PROJECT_ROOT 對齊本機物理路徑 / DB_* / FINMIND_TOKEN / FRED_API_KEY)
#    macOS /home→/Users symlink 用 path_setup.py os.path.realpath 解析(§0.0-I.10)

# 5. Import smoke test(必過才繼續)
./venv/bin/python -c "import psycopg2,pandas,polars,numpy,requests,sklearn,xgboost,lightgbm,catboost,torch; print('✅ imports OK')"
```
> ⚠️ **最高運行準則**:PHASE 4/10/11 之 ≥30 min 長跑**必須 `caffeinate -dimsu` 包裹啟動**(防 macOS 睡眠殺進程)+ 先驗 AC 插電(`pmset -g batt`);Linux 用 `systemd-inhibit`。

---

## §2 12-PHASE 序列總表（不可跳 / 不可改 order;每 PHASE audit gate 通過才進下一）

| PHASE | 階段 | 主程式 | 預期產出 |
| :--- | :--- | :--- | :--- |
| 0 | 環境 | (上方) | imports OK |
| 1 | Schema DDL | `path_setup.py` → `data_schema.py --init`(**2 infra**)→ `initialize_market_data.py` → `core_universe_schema.py` → `feature_store_schema.py` | infra log + stocks + universe-gov + feature-store schema 表(**raw 表尚未建**) |
| 2 | Genesis 名冊 | `sovereign_sync_engine.py --seed` | `TaiwanStockInfo`(generic 建,~2800+ 檔)|
| 2b | **FRED 前置** ⚠️(B5)| `fetchers/fetch_fred_data.py` | `fred_series`(generic 建,24 series 含 13 KWAVE)|
| 3 | Bootstrap 宇宙 ⚠️(B6)| `core_universe_builder.py bootstrap_init --bootstrap --commit` | 過渡 research_universe 成員 |
| 4 | 全市場全歷史 sync | `sovereign_sync_engine.py --universe full --all` | **raw 表 generic 自動建** + ~81M rows(實測 ~5h19m)|
| 5 | **Raw audit(DB↔API 對帳)** | `audit/audit_full_db_vs_api_reconcile.py --scope core` | byte-level DB↔API 驗證(無 AI 幻像)|
| 6 | 最終宇宙 | `core_universe_builder.py bootstrap_final --commit` | committed core snapshot(CoreScore 五層)|
| 7 | Feature Store ⚠️(B7)| `feature_store_builder.py --commit` + `build_historical_panels.py --commit` | feature_values panels(feature 數見 §6 caveat)|
| 8 | 宇宙加 gate(source-pure)| `core_universe_builder.py --with-feature-gate --commit` | pan-historical source-pure core |
| 9 | Feature audit(訓練前)| 4 audit 程式(見 §5)| IC / sign / necessity / quality |
| 10 | 模型訓練 | base `model_trainer.py --commit` → 9 tree trainers | model artifacts + registry |
| 11 | 模型驗證 | `universe_completeness_schema.py` → 13 multi_cycle validators | 各週期 metrics + T_CZ-6 裁決 |

> ⚠️ = 從零序列之**順序缺口**(B5/B6/B7,見 §7),不補會在空 DB 卡住。

---

## §3 逐 PHASE 詳述

### PHASE 1 — Schema DDL（§14.7-DJ 後:只建 infra + 治理/特徵 schema,raw 表由 generic 後建）
```bash
# 1.0 路徑主權自癒(canonical 序列第一步;path_setup.py 被 ~111 支 import,絕不可移除)
./venv/bin/python scripts/core/path_setup.py
# 1.1 schema DDL
./venv/bin/python scripts/core/data_schema.py --init --force          # ⭐ 只建 2 infra log 表(INFRA_TABLE_SCHEMAS);raw FinMind/FRED 表改 generic 於 sync 時建
./venv/bin/python scripts/ingestion/initialize_market_data.py         # stocks 名冊表 + pipeline_execution_log + data_audit_log(字串欄 ≥VARCHAR(255))
./venv/bin/python scripts/core/core_universe_schema.py --init          # §6.7 universe governance 表
./venv/bin/python scripts/core/feature_store_schema.py --init          # feature_values / feature_set
```
**audit gate**:四程式 exit 0;DB 內 infra/stocks/universe-gov/feature-store schema 表存在。**注意**:此時**不應有** TaiwanStockPrice 等 raw 表(它們在 PHASE 2/4 由 generic 建)。

### PHASE 2 — Genesis 名冊
```bash
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py --seed
```
產 `TaiwanStockInfo`(generic auto-schema 建表;全市場 ~2800+ 檔)。

### PHASE 2b — FRED 前置 ⚠️（B5 gap）
```bash
./venv/bin/python scripts/fetchers/fetch_fred_data.py    # ⭐ generic provision_and_upsert 建 fred_series(v3.3 退役 hardcoded DDL_FRED);§0.3 FRED 唯一 ingestion 載體
```
產 `fred_series`(generic 建,24 series 含全 13 KWAVE;PK (series_id, date))。
> ⚠️ **B5 ordering 校正**:PHASE 3 bootstrap_init 之 `_check_kwave_market` Stage-1 gate 需 `fred_series` → FRED **必須提前到 PHASE 3 之前(本 2b)**,否則 PHASE 3 abort(`fred_series` UndefinedTable)。

### PHASE 3 — Bootstrap 宇宙 ⚠️（B6 gap）
```bash
./venv/bin/python scripts/core/core_universe_builder.py bootstrap_init --bootstrap --commit
```
**`--bootstrap` 不可省** —— 將 candidates 以 `research_universe` 過渡成員 commit,否則 PHASE 4「無標的」。

### PHASE 4 — 全市場全歷史 sync（最長階段;generic 建 raw 表）
```bash
# 必 caffeinate 包裹(最高運行準則);先驗 pmset -g batt 插電
caffeinate -dimsu ./venv/bin/python scripts/ingestion/sovereign_sync_engine.py --universe full --all
./venv/bin/python scripts/maintenance/audit_supply_chain.py --include-logs   # ⭐ PHASE 4 audit gate:須 PERFECT + FRED ≥ 4 series
```
- **raw 表(`TaiwanStockPrice`/`PriceAdj`/`PER`/法人/融資券/外資持股/財報/資產負債/月營收/股利…)由 generic ingester 於此階段首次 ingest 時自動建**(schema 對應 `reports/finmind_full_table_schema_20260608.md` 82-表 catalog;floor VARCHAR(255)/NUMERIC(20,6),值超界自動加大)。
- **≥30 min → §二.6 SHMM 強制**(N≥3 Monitor heartbeat + sentinel + watchdog);**≥5 min → §一.12 5-min 回報**。實測 ~5h19m / ~81M rows。FinMind sponsor ~6000/hr;§7.6 A5 auto-pause 5500/hr;§7.4-A 402 cascade 1800s cooldown。**Audit gate**:`audit_supply_chain.py` PERFECT 才進 PHASE 5。

### PHASE 5 — Raw audit（DB↔API 對帳;用戶要的「實際 API 驗證 raw data」)
```bash
./venv/bin/python scripts/audit/audit_full_db_vs_api_reconcile.py --scope core   # 或 --scope all
```
**byte-level 驗 raw DB rows ↔ FinMind/FRED API**(value/count/missing;無 AI 幻像)。**良性差異**:`TaiwanStockPriceAdj` 因除權息回溯重算 → value_mismatch(以 raw `TaiwanStockPrice` mismatch=0 為準)。

### PHASE 6 — 最終宇宙（real-data CoreScore）
```bash
./venv/bin/python scripts/core/core_universe_builder.py bootstrap_final --commit
./venv/bin/python scripts/maintenance/audit_core_universe.py   # ⭐ PHASE 6 audit gate:須 PERFECT
```
產 committed core snapshot(CoreScore 五層:30% DataQuality + 30% LiquidityMass + 20% FundamentalGravity + 15% InstitutionalFlow + 5% VolatilityControl − RiskPenalty)。

### PHASE 7 — Feature Store ⚠️（B7 gap）
```bash
# B7:feature_store_builder 寫 universe_completeness_snapshot;且 panel 預測 FK→model_registry → 必須先建這兩表
./venv/bin/python scripts/core/universe_completeness_schema.py --init
./venv/bin/python -c "import sys;sys.path.insert(0,'scripts');import core.model_trainer as mt,core.db_utils as du;\
cur=du.get_db_conn().cursor();cur.execute(mt.DDL_MODEL_REGISTRY);cur.execute(mt.DDL_MODEL_TRAINING_RUN);cur.connection.commit()"
# 再建特徵
./venv/bin/python scripts/core/feature_store_builder.py --commit                 # current panel
./venv/bin/python scripts/core/build_historical_panels.py --commit               # historical monthly panels
```
產 `feature_set`(source-pure 特徵;feature 數見 §6 caveat —— v0.5=37 或 §一.21 v0.6=32,以 code SPEC 為準)。

### PHASE 8 — 宇宙加 feature gate（pan-historical source-pure 收緊）
```bash
./venv/bin/python scripts/core/core_universe_builder.py --with-feature-gate --commit
```
產 pan-historical source-pure core(任一 panel 含 imputed → 排除;§14.7-DC v0.17)。
> ⚠️ **B1**:feature gate 須 `n >= len(SPEC_FEATURES)`(=當前特徵數),不可 hardcode `>=38`(amihud 移除後之 self-regression),否則清空核心宇宙。

### PHASE 9 — Feature audit（訓練前)
```bash
./venv/bin/python scripts/audit/audit_feature_ic_vs_future_return.py
./venv/bin/python scripts/audit/audit_feature_sign_stability.py
./venv/bin/python scripts/audit/audit_feature_necessity.py
./venv/bin/python scripts/audit/audit_feature_data_quality_bias.py
```
> ⚠️ **B2**:audit SPEC list 須 set-match SSOT `feature_store_builder.FEATURE_DEFINITIONS`,snapshot dynamic 讀 latest committed,不可 hardcode `fs_v0_4`/38。

### PHASE 10 — 模型訓練
```bash
caffeinate -dimsu ./venv/bin/python scripts/core/model_trainer.py --commit    # B3:base trainer 先跑(建 model_registry/training_run DDL)
# 再跑 9 tree family trainers(model_trainer_{lightgbm,lgbm_v2,xgboost,xgboost_dedicated,catboost,catboost_dedicated,random_forest,extra_trees,ensemble}.py --commit)
```
產 model artifacts(`data/models/`)+ registry 列。

### PHASE 11 — 模型驗證（self-contained walk-forward）
```bash
./venv/bin/python scripts/core/universe_completeness_schema.py --init   # 若 PHASE 7 已建可略
# 13 multi_cycle_*_validation.py;panel 窗由 core.db_utils.get_canonical_panel_dates() 動態取(§14.7-DE,禁寫死);metric 由 summarize_horizon_metrics() 算(§14.7-DF)
for m in lightgbm xgboost catboost random_forest extra_trees ensemble xgboost_dedicated catboost_dedicated transformer_dedicated; do
  for s in 5422 1009 7331; do
    caffeinate -dimsu ./venv/bin/python scripts/evaluation/multi_cycle_${m}_validation.py --commit --seed $s --output /tmp/${m}_s${s}.json
  done
done
```
> ⚠️ **B4**:T_CZ-6 gate(Eff t≥4.20 ∧ Sharpe≥2.40 ∧ Win≥79%)為 docstring,code 唯一硬判定 `|eff_t|>1.997`(p<0.05)→「過 T_CZ-6」須**人工**對照 metrics 裁決。validator 為**雙軌獨立**(不讀 PHASE 10 artifacts,自做 train→predict),唯一前置 = PHASE 7 feature_values。

---

## §4 Canonical 程式清單（含本 session 版本）

| 程式 | 角色 | PHASE |
| :--- | :--- | :--- |
| `core/path_setup.py` | 路徑主權(25 維接口,realpath 跨平台)| 0 |
| `core/db_utils.py` v2.50 | DB 連線 + §6.7 SQL + `get_canonical_panel_dates()`/`summarize_horizon_metrics()` 單一引用源 | 全 |
| **`core/generic_schema.py` v1.2** | ⭐ **通用 auto-schema SSOT**(§0.0-I):看 API 回應推型別/建表/補欄/冪等 upsert;MIN_VARCHAR=255;退役 DATASET_REGISTRY 白名單後之全 raw 表建表機制 | 2,2b,4 |
| `core/data_schema.py` v2.23 | `INFRA_TABLE_SCHEMAS`(2 infra log)+ `get_dataset_columns/get_dataset_keys`(DB-derived;不再 index DATASET_REGISTRY)| 1 |
| `core/core_universe_schema.py` | §6.7 universe governance 表 DDL | 1 |
| `core/feature_store_schema.py` | feature_values/feature_set DDL | 1 |
| `core/universe_completeness_schema.py` | universe_completeness_snapshot DDL | 7/11 |
| `ingestion/initialize_market_data.py` v1.22 | stocks 名冊表 + pipeline_execution_log + data_audit_log(infra log 字串欄 ≥VARCHAR(255))| 1 |
| `ingestion/sovereign_sync_engine.py` v1.25 | 全市場 raw sync(seed/full;`_generic_ingest` 建表;FRED `FredData` 亦 generic;零白名單 `_target_datasets: if dataset: return [dataset]`)| 2,4 |
| `ingestion/finmind_generic_ingest.py` v0.3 | 單 dataset 探索 ingester(import generic_schema 繼承 floor)| 探索 |
| `fetchers/fetch_fred_data.py` v3.3 | FRED `fred_series` ingestion(**generic provision_and_upsert**,退役 DDL_FRED;§0.3 唯一 FRED 載體)| 2b,4 |
| `core/core_universe_builder.py` | CoreScore 選股 + pan-historical source-pure gate | 3,6,8 |
| `core/feature_store_builder.py` | source-pure 特徵計算(current panel;FEATURE_DEFINITIONS = SPEC SSOT)| 7 |
| `core/build_historical_panels.py` | historical monthly panels | 7 |
| `audit/audit_full_db_vs_api_reconcile.py` | **raw DB↔API byte-level 對帳** | 5 |
| `audit/audit_feature_{ic_vs_future_return,sign_stability,necessity,data_quality_bias}.py` | 訓練前特徵 audit | 9 |
| `maintenance/audit_supply_chain.py` / `audit_core_universe.py` | PHASE 4/6 audit gate | 4,6 |
| `core/model_trainer.py` + 9 family trainers | model artifacts + registry | 10 |
| 13 `evaluation/multi_cycle_*_validation.py` | multi-cycle walk-forward 驗證(共用 helper)| 11 |

---

## §5 資料 Audit 總覽（何時 audit / 驗什麼）

| Audit | PHASE | 驗證內容 |
| :--- | :--- | :--- |
| `audit_supply_chain --include-logs` | 4 | sync 完整性 + FRED ≥ 4 series(PERFECT gate)|
| `audit_full_db_vs_api_reconcile` | 5 | **raw DB rows ↔ FinMind/FRED API**(value/count/missing;byte-level)|
| `audit_core_universe` | 6 | committed core snapshot 完整性(PERFECT gate)|
| `audit_feature_ic_vs_future_return` | 9 | 各特徵 cross-sectional IC vs 未來報酬 |
| `audit_feature_sign_stability` | 9 | 特徵 IC 符號跨期穩定性 |
| `audit_feature_necessity` | 9 | ablation:移除特徵之 IC 影響 |
| `audit_feature_data_quality_bias` | 9 | imputed/zero-fill/survivorship 偏差 |

**核心 doctrine**(§14.7-DC / §一.10):任一特徵 `is_null_imputed=True` 之 stock 必須排除 core(無 FinMind/FRED source 之值 = AI 幻像);raw 值須 byte-equal API(§14.7-CE)。

---

## §6 成功驗證（重建完成的 final state）

- **Raw schema**:82 表(67 FinMind Taiwan + 13 非Taiwan + 2 FRED)由 generic ingester 建,schema = `reports/finmind_full_table_schema_20260608.md`(81/82 用實際 API live 驗證;大小寫 API 逐字;floor VARCHAR(255)/NUMERIC(20,6))。
- **Reconcile**(PHASE 5):raw `TaiwanStockPrice` value_mismatch=0(adjusted-price 回溯為良性 mismatch)。
- **Universe**:committed snapshot pan-historical source-pure(0 imputed-among-core)。
- **Feature**:`feature_set`(`get_canonical_panel_dates()` 動態 panel 窗);**feature/core 數以 code 當前 SPEC 為準** —— 2026-06-01 實跑基準 = v0.5/37 features/397 core/96 panels;**§一.21 v0.6 已演進至 32 features/294 core**(再移 5 死重/冗餘:trust_net×2/net_income_positive/preferential_attachment/zero_volume),執行時以 `feature_store_builder.FEATURE_DEFINITIONS` 實數為準(§一.10 不硬寫)。
- **Models**:`data/models/` 有 artifacts + `model_registry` 有列。
- **Validation**:tree-family validators 各週期 metrics;annual(252d)應過 T_CZ-6(實證:LGBM/XGB/CatBoost/RF/ET/Ensemble annual 皆過)。

---

## §7 從零 Gotchas（B1-B7 + 本 session generic 軸,不補會卡）

| # | PHASE | 問題 | 修法 |
| :--- | :--- | :--- | :--- |
| **G** | 1 | (§14.7-DJ NEW)PHASE 1 後**沒有** raw 表 → 誤判失敗 | 正常:raw 表由 generic 於 PHASE 2/4 sync 時自動建;PHASE 1 只建 infra+治理+特徵 schema |
| B5 | 2b | `fred_series` UndefinedTable(PHASE 3 需)| 先跑 `fetchers/fetch_fred_data.py`(generic 建 fred_series)|
| B6 | 3 | bootstrap commit 0 成員 → PHASE 4 無標的 | `bootstrap_init --bootstrap --commit` |
| B7 | 7 | `universe_completeness_snapshot` / `model_registry` 不存在 → panel BUILD-FAILED | PHASE 7 前先建這兩表 |
| B1 | 8 | feature gate hardcode `>=38`(amihud 移除後)→ 清空宇宙 | gate 用 `len(SPEC_FEATURES)`(=當前特徵數)|
| B2 | 9 | audit SPEC hardcode 38 或 `fs_v0_4` | set-match 當前 SPEC + dynamic latest panel |
| B3 | 10 | family trainer 只 INSERT,registry 表未建 | base `model_trainer.py --commit` 先跑 |
| B4 | 11 | T_CZ-6 非 code-enforced | 人工對照印出 metrics 裁決 |

---

## §8 治權 cross-ref + 誠實 caveat

- **治權 SSOT**:§14.7-DD(12-PHASE)+ **§14.7-DJ(pure-generic ingestion + 82-表字典)** + §14.7-DC(pan-historical source-pure)+ §14.7-DE/DF(單一引用源)+ §14.7-CZ/T_CZ-6(validation gate)。
- **最高運行準則**:PHASE 4/10/11 ≥30 min → **`caffeinate -dimsu` 包裹 + AC 插電**(防睡眠殺進程)+ §二.6 SHMM;≥5 min → §一.12 5-min 回報。
- **資料真實性**:全數字 trace (a) 程式 stdout / (b) DB query / (c) API response;**無 AI 補值/幻像**(§一.10);raw 值 byte-equal API(§14.7-CE);imputed 股排除 core(§14.7-DC)。
- **零白名單**:任意 FinMind dataset 經 `--dataset X` 即由 generic 建表,不需預註冊(用戶 2026-06-08 directive);intraday(tick/分K/5秒)raw 不入庫(T_DI-7,日為最小單位)。
- **時間預算 = ESTIMATE**:總首建 ~7-14 hr(PHASE 4 主導 ~5-10 hr)。
- **完整 82-表 schema(逐欄中文定義/型別/大小寫)**:見 `reports/finmind_full_table_schema_20260608.md`(本檔 raw 層之 schema SSOT)。
- **本 runbook 基於**:2026-06-01 本機實跑(PHASE 4 5h19m / ~81M rows)+ 2026-06-08 §14.7-DJ generic 改動 + 82-表 live-API 驗證。
