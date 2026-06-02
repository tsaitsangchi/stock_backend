# 從零到模型產生與驗證 — 全系統重建 Runbook（2026-06-02 comprehensive）

**文件性質**：本檔為「刪除全部環境與資料後,從零完整重建本系統」之**可執行 runbook**。涵蓋:環境建立 → 資料產生 → 資料 audit → 核心股 → 特徵 → 模型訓練 → 模型驗證。
**治權對齊**：主憲章 **§14.7-DD**(Tree-Family From-Zero → Model → Validation 12-PHASE 範本)為治權 SSOT;本檔為其 implementation reference,並反映 **2026-06-01 本機實跑**(v0.18 / 397 core / 96 panels / PHASE 4 ~5h19m)+ 現況(37 source-pure 特徵 / §14.7-DE/DF 單一引用源 helper)。
**前序**：supersede `tree_based_from_zero_build_runbook_20260531.md` + `from_zero_to_model_build_guide_20260528.md`(整合 + 現況化)。
**誠實聲明（§一.10）**：時間預算為 **ESTIMATE**(實跑以各程式 stdout `Total elapsed` 為準);PHASE 4 全市場同步本機實測 ~5h19m / ~81M rows。

---

## §0 檔案整理建議（先讀:整理 vs 重建的順序）

**建議:先用本 runbook 定義 canonical 程式集,再隔離非 canonical 檔案**(避免誤殺)。
- **判準**:凡不在下方 §4 canonical 清單、且不被任一 PHASE 程式 `import` 之 `.py`,即「沒用」候選。
- **既有隔離區**(per `.gitignore`):`scripts/archive/`、`scripts/_patch_backup/`、`archive/_pending_removal_*/`。
- **流程**:(1) 跑 `grep -rl "import X"` 確認某程式無人引用 → (2) `git mv` 到 `archive/_pending_removal_20260602/` → (3) 跑一次本 runbook PHASE 0-1 import smoke + PHASE 11 確認無 breakage → (4) 確認後才真刪。**不直接 rm**(§一.6)。

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

# 4. .env(PROJECT_ROOT 對齊本機物理路徑 / DB_* / FINMIND_TOKEN / FRED_API_KEY / GITHUB_TOKEN)
#    macOS /home→/Users symlink 用 path_setup.py os.path.realpath 解析

# 5. Import smoke test(必過才繼續)
./venv/bin/python -c "import psycopg2,pandas,polars,numpy,requests,sklearn,xgboost,lightgbm,catboost,torch; print('✅ imports OK')"
```

---

## §2 12-PHASE 序列總表（不可跳 / 不可改 order;每 PHASE audit gate 通過才進下一）

| PHASE | 階段 | 主程式 | 預期產出 |
| :--- | :--- | :--- | :--- |
| 0 | 環境 | (上方) | imports OK |
| 1 | Schema DDL | `path_setup.py`(自癒)→ `data_schema.py` → `initialize_market_data.py` → `core_universe_schema.py` → `feature_store_schema.py` | 空表 + stocks/pipeline_log/audit_log 建立 |
| 2 | Genesis 名冊 | `sovereign_sync_engine.py --seed` | TaiwanStockInfo 全市場資產 |
| 2b | **FRED 前置** ⚠️(B5)| `fetchers/fetch_fred_data.py` | fred_series(13 KWAVE + 其他)|
| 3 | Bootstrap 宇宙 ⚠️ | `core_universe_builder.py bootstrap_init --bootstrap` | 過渡 research_universe 成員 |
| 4 | 全市場全歷史 + FRED | `sovereign_sync_engine.py --universe full --all` | ~81M rows(實測 ~5h19m)|
| 5 | Raw audit | `audit_full_db_vs_api_reconcile.py` | DB↔API 對帳 |
| 6 | 最終宇宙 | `core_universe_builder.py bootstrap_final --commit` | committed core snapshot |
| 7 | Feature Store ⚠️ | `feature_store_builder.py --commit` + `build_historical_panels.py` | 96 panels × 37 features(v0_5)|
| 8 | 宇宙加 gate(source-pure)| `core_universe_builder.py --with-feature-gate --commit` | v0.18 / 397 core |
| 9 | Feature audit(訓練前)| 4 audit 程式(見 §5)| IC / sign / necessity / quality |
| 10 | 模型訓練 | base `model_trainer.py --commit` → 9 tree trainers | model artifacts + registry |
| 11 | 模型驗證 | `universe_completeness_schema.py` → 13 multi_cycle validators | 各週期 metrics + T_CZ-6 裁決 |

> ⚠️ = 從零序列之**順序缺口**(B5/B6/B7,見 §7),不補會在空 DB 卡住。

---

## §3 逐 PHASE 詳述

### PHASE 1 — Schema DDL
```bash
# 1.0 路徑主權自癒(憲章 §20 #6 canonical 序列第一步:.env → path_setup → data_schema;推導 25 維路徑)
#     path_setup.py 為 §3.2 [Path Sovereignty] 基石,被 ~111 支程式 import;絕不可隔離/移除
./venv/bin/python scripts/core/path_setup.py
# 1.1 schema DDL
./venv/bin/python scripts/core/data_schema.py --init --force          # raw 表(TaiwanStockPriceAdj 等 11 表)
./venv/bin/python scripts/ingestion/initialize_market_data.py         # ⭐ stocks 名冊表 + pipeline_execution_log + data_audit_log(PHASE 2 seed 之前置;憲章 §14.7-DD PHASE 1 明列)
./venv/bin/python scripts/core/core_universe_schema.py --init          # §6.7 universe governance 表
./venv/bin/python scripts/core/feature_store_schema.py --init          # feature_values / feature_set
```
**audit gate**:三 schema 程式 exit 0;表存在。

### PHASE 2 — Genesis 名冊
```bash
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py --seed
```
產 `TaiwanStockInfo` 全市場清單(~2800+ 檔)。

### PHASE 2b — FRED 前置 ⚠️（B5 gap）
```bash
./venv/bin/python scripts/fetchers/fetch_fred_data.py    # ⭐ 正解:fetchers/ 此支含 DDL_FRED(建 fred_series)+ 為憲章 §0.3 FRED 唯一 ingestion 載體(非 ingestion/ingest_fred_data.py)
```
產 `fred_series`(24 series 含全 13 KWAVE)。
> ⚠️ **B5 修正(本 runbook 對憲章 §14.7-DD 之 ordering 校正)**:憲章 §14.7-DD 表將 FRED 列於 **PHASE 4**,但 2026-06-01 實跑揭露 **PHASE 3 bootstrap_init 之 `core_universe_builder._check_kwave_market` Stage-1 gate 需 fred_series** → 若按憲章 PHASE-4 放,PHASE 3 會 abort(`fred_series` UndefinedTable)。故 FRED **必須提前到 PHASE 3 之前(本 2b)**。此為實證 ordering 校正。

### PHASE 3 — Bootstrap 宇宙 ⚠️（B6 gap）
```bash
./venv/bin/python scripts/core/core_universe_builder.py bootstrap_init --bootstrap --commit
```
**`--bootstrap` 不可省** —— 將 candidates 以 `research_universe` 過渡成員 commit,否則 PHASE 4 「無標的」。

### PHASE 4 — 全市場全歷史 sync + FRED（最長階段）
```bash
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py --universe full --all
./venv/bin/python scripts/maintenance/audit_supply_chain.py --include-logs   # ⭐ PHASE 4 audit gate(憲章 §14.7-DD):須 PERFECT + FRED ≥ 4 series
```
**≥30 min → §二.6 SHMM 強制**(N≥3 Monitor heartbeat + sentinel + watchdog);**≥5 min → §一.12 5-min 回報**。實測 ~5h19m / ~81M rows。FinMind sponsor tier ~6000/hr;§7.6 A5 auto-pause 5500/hr;§7.4-A 402 cascade 1800s cooldown。**Audit gate**:`audit_supply_chain.py` PERFECT 才進 PHASE 5。

### PHASE 5 — Raw audit（DB↔API 對帳）
```bash
./venv/bin/python scripts/audit/audit_full_db_vs_api_reconcile.py --scope core   # 或 --scope all
```
驗 raw 完整性。**良性差異**:TaiwanStockPriceAdj 因除權息回溯重算 → value_mismatch(raw Price mismatch=0 為準)。

### PHASE 6 — 最終宇宙（real-data CoreScore）
```bash
./venv/bin/python scripts/core/core_universe_builder.py bootstrap_final --commit
./venv/bin/python scripts/maintenance/audit_core_universe.py   # ⭐ PHASE 6 audit gate(憲章 §14.7-DD):須 PERFECT
```
產 committed core snapshot(CoreScore 五層:30% DataQuality + 30% LiquidityMass + 20% FundamentalGravity + 15% InstitutionalFlow + 5% VolatilityControl − RiskPenalty)。**Audit gate**:`audit_core_universe.py` PERFECT 才進 PHASE 7。

### PHASE 7 — Feature Store ⚠️（B7 gap）
```bash
# B7:feature_store_builder 寫 universe_completeness_snapshot;且 panel 預測 FK→model_registry
#     → 必須先建這兩表(否則全 panel BUILD-FAILED)
./venv/bin/python scripts/core/universe_completeness_schema.py --init
./venv/bin/python -c "import sys;sys.path.insert(0,'scripts');import core.model_trainer as mt,core.db_utils as du;\
cur=du.get_db_conn().cursor();cur.execute(mt.DDL_MODEL_REGISTRY);cur.execute(mt.DDL_MODEL_TRAINING_RUN);cur.connection.commit()"
# 再建特徵
./venv/bin/python scripts/core/feature_store_builder.py --commit                 # current panel(37 features)
./venv/bin/python scripts/core/build_historical_panels.py --commit               # 95 historical monthly panels
```
產 `feature_set_v0.5`:**96 panels × 37 source-pure 特徵**(K-wave 7 死重已於 v6.26.2 移除)。

### PHASE 8 — 宇宙加 feature gate（source-pure 收緊）
```bash
./venv/bin/python scripts/core/core_universe_builder.py --with-feature-gate --commit
```
產 **v0.18 / 397 core**(PAN-HISTORICAL gate:任一 panel 含 imputed → 排除)。
> ⚠️ B1:feature gate 須 `n >= len(SPEC_38_FEATURES)`(=37),不可 hardcode `>=38`(amihud 移除後之 self-regression),否則清空核心宇宙。

### PHASE 9 — Feature audit（訓練前)
```bash
./venv/bin/python scripts/audit/audit_feature_ic_vs_future_return.py
./venv/bin/python scripts/audit/audit_feature_sign_stability.py
./venv/bin/python scripts/audit/audit_feature_necessity.py
./venv/bin/python scripts/audit/audit_feature_data_quality_bias.py
```
> ⚠️ B2:audit SPEC list 須 = 37(set-match SSOT `feature_store_builder.FEATURE_DEFINITIONS`),snapshot dynamic 讀 latest committed,不可 hardcode `fs_v0_4`。

### PHASE 10 — 模型訓練
```bash
./venv/bin/python scripts/core/model_trainer.py --commit          # B3:base trainer 先跑(建 model_registry/training_run DDL)
# 再跑 9 tree family trainers(model_trainer_{lightgbm,lgbm_v2,xgboost,xgboost_dedicated,catboost,catboost_dedicated,random_forest,extra_trees,ensemble}.py --commit)
```
產 model artifacts(`data/models/`)+ registry 列。

### PHASE 11 — 模型驗證（self-contained walk-forward）
```bash
./venv/bin/python scripts/core/universe_completeness_schema.py --init   # 若 PHASE 7 已建可略
# 13 multi_cycle_*_validation.py;panel 窗由 core.db_utils.get_canonical_panel_dates() 動態取(§14.7-DE,禁寫死)
for m in lightgbm xgboost catboost random_forest extra_trees ensemble xgboost_dedicated catboost_dedicated transformer_dedicated; do
  for s in 5422 1009 7331; do
    ./venv/bin/python scripts/evaluation/multi_cycle_${m}_validation.py --commit --seed $s --output /tmp/${m}_s${s}.json
  done
done
```
> ⚠️ B4:T_CZ-6 gate(Eff t≥4.20 ∧ Sharpe≥2.40 ∧ Win≥79%)為 docstring,code 唯一硬判定 `|eff_t|>1.997`(p<0.05)→ 「過 T_CZ-6」須人工對照 metrics 裁決。validator 為**雙軌獨立**(不讀 PHASE 10 artifacts,自做 train→predict),唯一前置 = PHASE 7 feature_values。

---

## §4 Canonical 程式清單（per-program 一句話）

| 程式 | 角色 | PHASE |
| :--- | :--- | :--- |
| `path_setup.py` | 路徑主權(25 維接口,realpath 跨平台)| 0 |
| `core/db_utils.py` v2.50 | DB 連線 + §6.7 SQL + **get_canonical_panel_dates()/summarize_horizon_metrics()** 單一引用源 helper | 全 |
| `core/data_schema.py` | raw 11 表 DDL + API schema 主權 | 1 |
| `core/core_universe_schema.py` | §6.7 universe governance 表 DDL | 1 |
| `core/feature_store_schema.py` | feature_values/feature_set DDL | 1 |
| `core/universe_completeness_schema.py` | universe_completeness_snapshot DDL | 7/11 |
| `ingestion/initialize_market_data.py` | stocks 名冊表 + pipeline_execution_log + data_audit_log DDL | 1 |
| `ingestion/sovereign_sync_engine.py` | 全市場 raw sync(seed / full / SHMM)| 2,4 |
| `fetchers/fetch_fred_data.py` | FRED macro ingestion(建 fred_series + DDL_FRED;§0.3 唯一 FRED 載體)| 2b,4 |
| `core/core_universe_builder.py` | CoreScore 選股 + source-pure gate | 3,6,8 |
| `core/feature_store_builder.py` v0.8 | 37 source-pure 特徵計算(current panel)| 7 |
| `core/build_historical_panels.py` | 95 historical monthly panels | 7 |
| `audit/audit_full_db_vs_api_reconcile.py` | raw DB↔API 對帳 | 5 |
| `audit/audit_feature_{ic_vs_future_return,sign_stability,necessity,data_quality_bias}.py` | 訓練前特徵 audit | 9 |
| `core/model_trainer.py` + 9 family trainers | model artifacts + registry | 10 |
| 13 `evaluation/multi_cycle_*_validation.py` | multi-cycle walk-forward 驗證(用共用 helper)| 11 |

---

## §5 資料 Audit 總覽（何時 audit / 驗什麼）

| Audit | PHASE | 驗證內容 |
| :--- | :--- | :--- |
| `audit_full_db_vs_api_reconcile` | 5 | raw DB rows ↔ FinMind/FRED API(value/count/missing)|
| `audit_feature_ic_vs_future_return` | 9 | 各特徵 cross-sectional IC vs 未來報酬 |
| `audit_feature_sign_stability` | 9 | 特徵 IC 符號跨期穩定性 |
| `audit_feature_necessity` | 9 | ablation:移除特徵之 IC 影響 |
| `audit_feature_data_quality_bias` | 9 | imputed/zero-fill/survivorship 偏差 |
| (選)`audit_per_stock_source_authority` / `audit_survivorship_bias` / `audit_universe_selection_bias` | 5-9 | 個股來源主權 / 倖存者偏差 / 選股偏差 |

**核心 doctrine**(§14.7-DC / §一.10):任一特徵 `is_null_imputed=True` 之 stock 必須排除 core(無 FinMind/FRED source 之值 = AI 幻像)。

---

## §6 成功驗證（重建完成的 final state）

- **Universe**:committed snapshot policy v0.18 / **397 core** / source-pure(0 imputed-among-core)
- **Feature**:`feature_set_v0.5` / 96 panels / **37 features** / `get_canonical_panel_dates()` 回傳 157 panels(2013-05-15~2026-06-01)
- **Models**:`data/models/` 有 artifacts + `model_registry` 有列
- **Validation**:tree-family validators 跑出各週期 metrics;**annual(252d)應過 T_CZ-6**(實證:LGBM/XGB/CatBoost/RF/ET/Ensemble annual 皆過,net +23~33%/yr)
- **Reconcile**:raw Price value_mismatch=0(adjusted-price 回溯為良性)

---

## §7 從零 Gotchas（B1-B7,不補會卡）

| # | PHASE | 問題 | 修法 |
| :--- | :--- | :--- | :--- |
| B5 | 2b | `fred_series` UndefinedTable(PHASE 3 需)| 先跑 `ingest_fred_data.py` |
| B6 | 3 | bootstrap commit 0 成員 → PHASE 4 無標的 | `bootstrap_init --bootstrap` |
| B7 | 7 | `universe_completeness_snapshot` / `model_registry` 不存在 → panel BUILD-FAILED | PHASE 7 前先建這兩表 |
| B1 | 8 | feature gate hardcode `>=38`(amihud 移除後)→ 清空宇宙 | gate 用 `len(SPEC_38_FEATURES)`=37 |
| B2 | 9 | audit SPEC hardcode 38 或 `fs_v0_4` | set-match 37 + dynamic latest panel |
| B3 | 10 | family trainer 只 INSERT,registry 表未建 | base `model_trainer.py --commit` 先跑 |
| B4 | 11 | T_CZ-6 非 code-enforced | 人工對照印出 metrics 裁決 |

---

## §8 治權 cross-ref + 誠實 caveat

- **治權 SSOT**:§14.7-DD(12-PHASE)+ §14.7-CZ(8-phase LGBM 前身)+ §14.7-DC(source-pure)+ §14.7-DE/DF(單一引用源)+ §14.7-CY/CZ/T_CZ-6(validation gate)。
- **Long-running**:PHASE 4/10/11 ≥5min → §一.12 5-min 回報;≥30min → §二.6 SHMM。
- **資料真實性**:全數字 trace (a) 程式 stdout / (b) DB query / (c) API response;**無 AI 補值**(§一.10)。
- **時間預算 = ESTIMATE**:總首建 ~7-14 hr(PHASE 4 主導 ~5-10 hr)。
- **外部權重模型**(Chronos/foundation,若納入):非 source-pure,須醒目揭露(§14.7-DC caveat)。
- **本 runbook 基於 2026-06-01 本機實跑**(v0.18/397/96 panels/PHASE 4 5h19m)+ 現況 code(37 features / db_utils v2.50)。
