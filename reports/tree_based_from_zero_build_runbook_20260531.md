# Tree-Based(Production 主軸)從零重建到模型驗證 完整 Runbook

**文件性質**：本檔為「環境與資料全部刪除後，依本檔可從零完整重建至模型產生 + 模型驗證」之執行序列 runbook。
**治權位階**：本檔為操作層 runbook，受憲章治權約束；治權判準以 `reports/系統架構大憲章_v6.1.0.md` 為準，本檔僅承載「執行順序 + 各程式說明 + 資料 audit」之落地序列。
**對齊憲章錨點**：§14.7-AM(從零→全市場全歷史+FRED 4 步序列治權範本)/ §14.7-CB(Feature Completeness Gate)/ §14.7-CJ(Reasonableness Gate)/ §14.7-CE(Per-Stock Empirical API Verification)/ §14.7-CF(Core Stock Selection SSOT)/ §14.7-DC(Source-Pure Doctrine)/ §14.7-BQ(Model Trainer Contract)/ §14.7-CW(Model Metric Gates)/ §14.7-CZ(Multi-cycle Validation Gates)。
**Scope**：Tree-based production 主軸 = 9 tree 模型(xgboost / xgboost_dedicated / lightgbm / lgbm_v2 / catboost / catboost_dedicated / random_forest / extra_trees / ensemble)+ 對應 9 multi-cycle validators。⚠️ Foundation/Transformer 模型(transformer_dedicated / itransformer / patchtst / tft / chronos)為**獨立研究軸,不在本 runbook scope**。
**最後更新**：2026-05-31
**資料真實性聲明(per §一.10)**：本檔標註 ✅DB-verified 之數字 source = 活 PostgreSQL READ-ONLY query;標註 ⚠️估計 之數字為依程式結構推估(未實跑 profiling),非 deterministic fact。

---

## 〇、執行前必讀 — 已知 BLOCKERS(依憲章「如有不足先修訂」原則之揭露)

本 runbook 在研究階段以 4 個 research agent + 本人 code/DB 三方驗證,**發現以下會阻斷乾淨重建之 code-level 缺陷**。這些**不是憲章缺口**(憲章治權條文齊備),而是 code 與最新 doctrine(amihud 移除 → 37 features)未同步之 **regression / 命名債**。

> ✅ **狀態更新(2026-05-31,依用戶 Q2「修正 B1-B3」授權)**:**B1+B2 已 code-fix(py_compile 全 PASS,僅改 code,未 git commit、未 DB write)**;B3 為 resolved-by-ordering(DDL idempotent,無需改 code);B4 為治權揭露(人工裁決,無需改 code)。下表「修正狀態」欄為各項當前狀態。**4 項已全部入憲主憲章 §14.7-DD + CLAUDE.md §一.14**。

| # | 缺陷 | 位置 | 影響 | 修正狀態 | 觸發 PHASE |
|---|---|---|---|---|---|
| **B1** | feature gate 常數未隨 amihud 移除更新:`SPEC_38_FEATURES` 已含 **37** 個 feature,但 gate 仍比對 `if n >= 38` | [core_universe_builder.py:2495](scripts/core/core_universe_builder.py:2495)(list 在 :2303)| 用 `--with-feature-gate` 時**每一支股票都被拒絕** → 核心宇宙變空 | ✅ **已修(code-only,未 commit)**:`n >= 38` → `n >= len(self.SPEC_38_FEATURES)`(=37,AST-confirmed);py_compile PASS | PHASE 8 |
| **B2** | 5 個 feature audit 仍列舉 **~43** 舊 SPEC(含已移除的 theme_strength / theme_is_semiconductor / fitness_signal_60d / barbell_balance_60d / right_tail_concentration_60d),且 necessity/sign/quality hardcode 舊 `fs_v0_4` snapshot | `audit_per_stock_feature_validity.py` / `audit_feature_necessity.py` / `audit_feature_sign_stability.py` / `audit_feature_data_quality_bias.py` / `audit_feature_ic_vs_future_return.py` | 對 v0_5(37 features)會回報 **false ❌ missing/incomplete**;blocking audit 卡住訓練 | ✅ **已修(code-only,未 commit)**:5 audit SPEC list 同步 37(set-match SSOT FEATURE_DEFINITIONS)+ 移除 dead range/literature/zero-fill entry + 2 hardcode snapshot 改 dynamic `pick_forward_window()`;py_compile PASS | PHASE 9 |
| **B3** | `model_registry` / `model_training_run` 表**不由 9 family trainer 建立**(family trainer 只 `INSERT ... ON CONFLICT`)| DDL 在 [model_trainer.py:266](scripts/core/model_trainer.py:266)(`DDL_MODEL_REGISTRY` / `DDL_MODEL_TRAINING_RUN`,IF NOT EXISTS,commit 路徑 L352-353 執行)| family trainer 在表不存在時 INSERT 失敗;`universe_completeness_schema.py` 亦以 model_registry 為前置 | ✅ **resolved-by-ordering(DDL idempotent,無需改 code)**:重建序列須**先跑 base `model_trainer.py --commit` 一次**(建表+訓練 §10 baseline),再跑 9 family trainer | PHASE 10 |
| **B4**(非缺陷,治權揭露)| multi-cycle validator 之 §14.7-CZ T_CZ-6 gate(Eff t≥4.20 / Sharpe≥2.40 / Win≥79%)**僅存在於 docstring,非 code-enforced**;code 唯一硬判定為 `abs(eff_t_stat) > 1.997`(p<0.05)| validator [multi_cycle_xgboost_validation.py:283](scripts/evaluation/multi_cycle_xgboost_validation.py:283)| 「PASS T_CZ-6」須**人工**對照印出的 metrics 判定,非自動 gate | ⚠️ **治權揭露(無需改 code)**:PHASE 11 驗收須人工裁決(見 §11)| PHASE 11 |

> **§一.8 誠實聲明**:B1 為本人於前一階段 amihud dead-feature 移除(SPEC 38→37)時遺漏同步的 gate 常數,屬本次移除引入之 self-introduced regression;B2/B3 為先前即存在之命名債/結構,非本次引入。**狀態(2026-05-31,依用戶 Q2 授權「修正 B1-B3」)**:B1+B2 **已 code-fix(py_compile 全 PASS),但僅改 code — 未 git commit、未 DB write、未 retrain**(commit 仍另需 §二.2 explicit 授權);B3 為 resolved-by-ordering;B4 為治權揭露。4 項已全數入憲主憲章 §14.7-DD + CLAUDE.md §一.14。

---

## 一、系統概觀與資料流

```
[FinMind API] ──┐
                ├─► sovereign_sync_engine.py ──► 10 raw tables + TaiwanStockInfo (PostgreSQL)
[FRED API] ─────┘                              + FredData
                                                     │
                                                     ▼
                                  core_universe_builder.py ──► core_universe_snapshot/membership
                                       (CoreScore 5 層 + 3 gate)      (committed, core_tier='core_universe')
                                                     │
                                                     ▼
                                  feature_store_builder.py ──► feature_values (feature_set_v0_5, 37 features)
                                       (is_null_imputed flag)         + feature_store_snapshot
                                                     │
                          ┌──────────────────────────┼──────────────────────────┐
                          ▼                           ▼                           ▼
                  feature audits             9 tree trainers            9 multi-cycle validators
            (per-stock/IC/necessity/    (model_trainer_*.py)        (multi_cycle_*_validation.py)
             sign/quality/reconcile)     → data/models/<id>/         → reports/multi_cycle_*.json
                  [PHASE 9]                  metrics.json [PHASE 10]      (4-horizon walk-forward) [PHASE 11]
```

**治權核心 doctrine(貫穿全流程)**:
- **§14.7-DC Source-Pure**:任一 feature `is_null_imputed=TRUE`(無 FinMind/FRED API source 之 AI 自補值)→ 該股 quarantine,不入核心宇宙、不入訓練。
- **§一.10 No Data Hallucination**:所有數字 trace 回 (a)程式輸出 / (b)DB query / (c)API response。
- **三 gate**:§14.7-CB Completeness(全 feature 有 row)∧ §14.7-CJ Reasonableness(無 outlier/NaN)∧ §14.7-DC Source-Pure(無 imputed)。

---

## PHASE 0 — 環境建立(Environment Setup)

### 0.1 OS-native 依賴(per §二.7 / 憲章 §0.0-I.9)

```bash
# macOS
brew install libomp            # xgboost / lightgbm 需 OpenMP runtime
brew install postgresql@17     # PostgreSQL 17 + psql client

# Linux (Debian/Ubuntu)
sudo apt-get install -y libgomp1 libpq-dev
```

### 0.2 PostgreSQL 資料庫 + 角色建立(⚠️ 從零無此步會卡住)

> 既有程式皆假設 db=`stock` / user=`stock` 已存在。從零須先建立:

```bash
# 啟動 PostgreSQL 17(macOS brew)
brew services start postgresql@17

# 建立角色與資料庫(密碼請自行設定,勿寫入版本控制)
/usr/local/opt/postgresql@17/bin/createuser --pwprompt stock
/usr/local/opt/postgresql@17/bin/createdb -O stock stock
```

### 0.3 Python 環境

```bash
cd /Users/hugo/project/stock_backend
python -m venv venv          # 或 pyenv local 3.12.13 後 venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install catboost   # ⚠️ requirements.txt 內 catboost 被註解,必須手動安裝
```

### 0.4 `.env` 設定(7 個強制環境變數 + FRED)

於 PROJECT_ROOT 建立 `.env`(**勿 commit;勿印出 secret 值**):

```
PROJECT_ROOT=/Users/hugo/project/stock_backend     # ⚠️ 見 GAP-0a:必須對齊本機物理路徑
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=stock
DB_USER=stock
DB_PASSWORD=<在此填入,勿外洩>
FINMIND_TOKEN=<FinMind sponsor token>
FRED_API_KEY=<FRED API key>
```

> **GAP-0a(路徑)**：歷史 `.env` 之 `PROJECT_ROOT=/home/hugo/...`(Linux 路徑)。在 macOS 須改為 `/Users/hugo/...`,或建 symlink。`path_setup.py` 以 `os.path.realpath()` 比對 anchor,路徑不符會在 import 期報錯。
> **GAP-0b(config.py 讀錯 .env)**：[config.py](scripts/config.py) 以 `BASE_DIR = Path(__file__).parent`(=`scripts/`)再 `load_dotenv(BASE_DIR/.env)`,會找不存在的 `scripts/.env`;且 `FINMIND_TOKEN = os.environ["FINMIND_TOKEN"]` 在環境未先載入時直接 `KeyError`。**緩解**:在 shell 先 `export $(grep -v '^#' .env | xargs)` 或於 PROJECT_ROOT 執行並確保 root `.env` 被 `path_setup` 載入。

### 0.5 Import smoke test(通過才可進入後續)

```bash
./venv/bin/python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm, catboost; print('✅ all imports OK')"
```

### 0.6 連線驗證(密碼以環境變數帶入,不印出)

```bash
export PGPASSWORD="$(grep '^DB_PASSWORD=' .env | cut -d= -f2-)"
/usr/local/opt/postgresql@17/bin/psql -h 127.0.0.1 -U stock -d stock -c "SELECT version();"
unset PGPASSWORD
```

---

## PHASE 1 — Schema DDL 建表

> **依賴鏈(必須依序)**：`data_schema.py` → `initialize_market_data.py` → `core_universe_schema.py` → `feature_store_schema.py`。`model_registry`(PHASE 10 由 base trainer 建)→ `universe_completeness_schema.py`(PHASE 11)。
> **GAP-1a(三個衝突的 `pipeline_execution_log` DDL)**:`data_schema.py` / `initialize_market_data.py` / `init_monitoring_schema.py` 各有版本不同的 `pipeline_execution_log`。**務必以 `data_schema.py --init` 為準先跑**;**勿跑 `init_monitoring_schema.py`**(legacy,且其 `ALTER TABLE stocks` 在 stocks 未建時失敗)。

### Step 1.1 — `data_schema.py`(13 表,RUN FIRST)

```bash
./venv/bin/python scripts/core/data_schema.py --init
```
- 建 13 表:`pipeline_execution_log`、`data_audit_log` + 10 FinMind raw(`TaiwanStockPrice`、`TaiwanStockPriceAdj`、`TaiwanStockPER`、`TaiwanStockInstitutionalInvestorsBuySell`、`TaiwanStockMarginPurchaseShortSale`、`TaiwanStockShareholding`、`TaiwanStockFinancialStatements`、`TaiwanStockBalanceSheet`、`TaiwanStockMonthRevenue`、`TaiwanStockDividend`)+ `TaiwanStockInfo` + `FredData`。
- CLI:`--init --force(DROP CASCADE)--table <name> --skip-api-contract`。
- ⚠️ 預設會探測 live FinMind + FRED(API contract 檢查);無網路或無 token 時加 `--skip-api-contract`。

### Step 1.2 — `initialize_market_data.py`(genesis:`stocks` 表)

```bash
./venv/bin/python scripts/ingestion/initialize_market_data.py
```
- 建 `stocks` 表(**≠ `TaiwanStockInfo`**,為內部名冊變體)。須在 `data_schema.py --init` 之後。

### Step 1.3 — `core_universe_schema.py`(7 表)

```bash
./venv/bin/python scripts/core/core_universe_schema.py --init
```
- 建 `core_universe_policy`、`core_universe_snapshot`、`core_universe_membership`、`core_universe_scores`、`theme_taxonomy`、`stock_theme_map`、`universe_revision_log`。
- Preflight 需 `pipeline_execution_log` / `data_audit_log` / `TaiwanStockInfo` 已存在(可 `--skip-preflight`)。

### Step 1.4 — `feature_store_schema.py`(3 表)

```bash
./venv/bin/python scripts/core/feature_store_schema.py --init
```
- 建 `feature_store_snapshot`、`feature_definition`、`feature_values`。
- Preflight 需 2 infra log + `TaiwanStockInfo` + `core_universe_snapshot` + `core_universe_policy`。

---

## PHASE 2 — Genesis 名冊同步(憲章 §二 Step 4)

```bash
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py --seed
```
- `--seed` 同步 `TaiwanStockInfo` 全市場名冊(~2,800 支;DB-verified 最近 committed total=2803)。
- `sovereign_sync_engine.py`(v1.22)為憲章 §3.1 **唯一授權 ingestion 載體**。

---

## PHASE 3 — Bootstrap 核心宇宙(破雞與蛋,憲章 §14.7-AM Step 4B bootstrap_init)

> **雞與蛋(charter §14.7-AM 已入憲解法)**:`--universe full/core` 同步需要 committed `core_universe_membership`,但 membership 需要 raw data 評分,raw data 又需要同步 → 循環。解法:先以 `latest_registry_fallback`(從 `TaiwanStockInfo` 取候選強制 commit,CoreScore=0 為合憲過渡)建立 bootstrap snapshot。

```bash
./venv/bin/python scripts/core/core_universe_builder.py --commit \
    --mode doctrine-native \
    --special-rebalance-reason "DB rebuild bootstrap 2026-05-31"
```
- ⚠️ **此步不加任何 gate flag**(features 尚未存在;且避開 B1 `>=38` bug)。
- `--special-rebalance-reason`(≥12 字)用以繞過 annual-rebalance guard(off-cycle 重建必需)。
- 產出:committed `core_universe_snapshot`(policy v0.13,無 gate)+ membership。
- doctrine-native 候選集 = `SELECT DISTINCT stock_id FROM TaiwanStockInfo WHERE industry_category IS NOT NULL`([core_universe_builder.py:2372](scripts/core/core_universe_builder.py:2372)),**不需先有 committed universe** → 這正是破雞與蛋的關鍵。

---

## PHASE 4 — 全市場全歷史同步(憲章 §二 Step 4F + Step 8)

> ⚠️ **此為 ≥30 分鐘 long-running workflow**(全市場全歷史 ⚠️估計 6–10 小時)→ **強制 §二.6 SHMM 心跳** + **§一.12 每 5 分鐘回報**。建議以 tmux / `systemd-run` 背景執行 + Monitor heartbeat。

### Step 4.1 — FinMind 10 raw tables × 全市場 × 全歷史

```bash
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py \
    --universe full --all \
    --dataset-batched --workers 4 --dynamic-quota \
    --special-full-market-reason "from-zero full rebuild 20260531"
```
- `--all` = 10 stock tables(**排除 `TaiwanStockInfo`** — 那由 `--seed` 處理)。
- `--universe full` 自動啟用 `--strict-source-history`(start=1990-01-01,resume off);FinMind API 回傳各股真實最早可得日期 → DB 最後交易日。
- `--special-full-market-reason`(≥12 字)強制必填(§6.8.7 第(4)條限定治理例外)。
- FinMind sponsor tier 上限 6000/hr;`--dynamic-quota` 動態節流(預設 throttle 5500/hr,8% headroom);30 分鐘 reset window。

### Step 4.2 — FRED 4 序列全歷史(idempotent;通常已被自動觸發)

```bash
./venv/bin/python scripts/ingestion/sovereign_sync_engine.py --source fred
```
- FRED 4 序列:`UNRATE`(1948-)、`DFF`(1954-)、`T10Y2Y`(1976-)、`VIXCLS`(1990-)→ 各自最早可得日期 → 今天(預設行為,無需 reason)。

---

## PHASE 5 — Raw 資料 audit(資料真實性閘門)

### Step 5.1 — DB ↔ API byte-level 對帳(§14.7-CE Per-Stock Empirical Verification)

```bash
./venv/bin/python scripts/audit/audit_full_db_vs_api_reconcile.py --scope core
```
- 逐股 × 10 表 + TaiwanStockInfo + FRED 與 live API byte-level 比對。
- **PASS 判準**:`value_mismatch == 0`。**這是資料完整性硬閘**(在建宇宙/特徵前確認 raw 真實)。
- CLI:`--scope {core,candidates,all}`、`--tables`、`--start(1990-01-01)`、`--end`、`--limit`、`--workers 8`、`--max-calls 6000`(估計超過即中止)、`--rate-per-hour 5500`、`--no-fred`、`--no-info`、`--sample-cap 20`。
- ⚠️ 需 `FINMIND_TOKEN` + `FRED_API_KEY`;會消耗 API quota → 注意與 PHASE 4 共用 hourly window。

---

## PHASE 6 — 最終核心宇宙(real-data,憲章 §14.7-AM Step 4B bootstrap_final)

```bash
./venv/bin/python scripts/core/core_universe_builder.py --commit \
    --mode doctrine-native \
    --special-rebalance-reason "DB rebuild bootstrap 2026-05-31 final"
```
- 以**真實 raw data** 重算 CoreScore,覆蓋 PHASE 3 的 bootstrap snapshot。
- ⚠️ **此步仍不加 feature gate**(feature_values 尚未建)。
- **CoreScore 5 層**(ThemeResonance 15% 已移除):`0.30·DataQuality + 0.30·LiquidityMass + 0.20·FundamentalGravity + 0.15·InstitutionalFlow + 0.05·VolatilityControl − RiskPenalty`。
- 產出:committed snapshot + membership(`core_tier='core_universe'` / `'quarantine'`)。

---

## PHASE 7 — Feature Store 建特徵(feature_set_v0_5,37 features)

> **依賴**：`feature_store_builder.py` preflight **硬性要求** latest committed `core_universe_snapshot`([feature_store_builder.py:288](scripts/core/feature_store_builder.py:288)「no committed core_universe_snapshot; run core_universe_builder.py --commit first」)→ 故必須在 PHASE 6 之後。

```bash
./venv/bin/python scripts/core/feature_store_builder.py --commit
```
- `--feature-set-version`(預設 `feature_set_v0.5`)、`--label-horizon`(預設 20)。**無 `--mode` flag**。
- 讀 raw FinMind 表(經 publication-date gate 防洩漏:MonthRevenue +10d、FinStmt Q1-Q3+45/Q4+90、Institutional/Margin/PER native);依 `core+convex` membership 計算。
- 寫 `feature_definition` + `feature_values`(欄位含 **`is_null_imputed`**)+ `feature_store_snapshot`。
- **`is_null_imputed=TRUE`**:當值缺失且套用了 `zero_fill`(此 flag 為 §14.7-DC source-pure gate quarantine 的依據)。
- 37 features = feature_set_v0_5(amihud 已移除;theme/macro/interaction 等 AI 幻像 feature 早於 v0.18-v0.20 移除)。

---

## PHASE 8 — 核心宇宙加 gate 收緊(source-pure,policy v0.18)

> ⚠️ **執行前必須先修正 B1**(否則所有股票被拒、宇宙變空):
> ```python
> # core_universe_builder.py:2495
> -            if n >= 38:
> +            if n >= len(self.SPEC_38_FEATURES):   # =37(amihud 移除後)
> ```

```bash
./venv/bin/python scripts/core/core_universe_builder.py --commit \
    --mode doctrine-native \
    --with-feature-gate --with-reasonableness-gate --with-source-pure-gate \
    --special-rebalance-reason "DB rebuild source-pure gate 20260531"
```
- gate → policy 版本:source-pure → **v0.18**;reasonableness → v0.15;feature → v0.14。
- **§14.7-CB Completeness**(全 feature 有 row)∧ **§14.7-CJ Reasonableness**(REASONABLE_BOUNDS 內)∧ **§14.7-DC Source-Pure**(`is_null_imputed` 者 quarantine)。
- 產出:最終 committed snapshot(DB-verified 最近狀態 = 398 core / 603 quarantine / 2803 total,policy v0.18)。
- ⚠️ **若不修 B1**:退而求其次可**跳過此步**,直接用 PHASE 6 的 universe + 倚賴 trainer 自身的 `is_null_imputed IS NOT TRUE` 載入過濾(見 PHASE 10)。但這會犧牲 universe 層的 source-pure 收緊,**不建議**。

---

## PHASE 9 — Feature audit(訓練前閘門)

> ⚠️ **執行前必須先修正 B2**(將各 audit 的 SPEC list 43→37 + snapshot 改讀 latest committed),否則對 v0_5 會 false ❌。

| 順序 | 程式 | 驗證內容 | PASS 判準 | 性質 |
|---|---|---|---|---|
| 9.1 | `audit_per_stock_feature_validity.py` | 每股完整度 = features 數 ∧ 全在 FEATURE_RANGES ∧ 100% 股票 | completeness==37 ∧ in-range | **BLOCKING(硬閘)** |
| 9.2 | `audit_feature_data_quality_bias.py` | look-ahead=0 ∧ zero_fill≤30% ∧ \|rho\|>0.95 配對數 ≤ 門檻 | all_pass(容忍 ⚠️) | **BLOCKING**(否則「§10 model_trainer cannot land」) |
| 9.3 | `audit_feature_necessity.py` | 0 個 NOT_NECESSARY(4-path:Literature/W1/W2/Doctrine) | 0 NOT_NECESSARY | **BLOCKING(部署)** |
| 9.4 | `audit_feature_ic_vs_future_return.py` | feature × forward-return Spearman IC | mean_abs_ic>0.03 ∧ ≥30% 顯著 | Advisory |
| 9.5 | `audit_feature_sign_stability.py` | 符號穩定性 vs literature | sign-stable ≥11 ∧ lit-mismatch<6 | Advisory |

```bash
# Blocking(全 PASS 才可進 PHASE 10)
./venv/bin/python scripts/audit/audit_per_stock_feature_validity.py
./venv/bin/python scripts/audit/audit_feature_data_quality_bias.py
./venv/bin/python scripts/audit/audit_feature_necessity.py
# Advisory
./venv/bin/python scripts/audit/audit_feature_ic_vs_future_return.py --horizon 14
./venv/bin/python scripts/audit/audit_feature_sign_stability.py
```

---

## PHASE 10 — 模型訓練(model_registry 建表 + 9 tree trainers)

> ⚠️ **必須先修正/處理 B3**:`model_registry` + `model_training_run` 由 base [model_trainer.py:266](scripts/core/model_trainer.py:266) 的 DDL 建立(在其 `--commit` 路徑套用)。9 個 family trainer 只 INSERT,**不建表**。

### Step 10.1 — base trainer 建表 + §10 baseline

```bash
./venv/bin/python scripts/core/model_trainer.py --commit   # 建 model_registry / model_training_run + 訓練 §10 baseline
```

### Step 10.2 — 9 個 tree family trainer(production 主軸)

```bash
for M in lgbm_v2 xgboost xgboost_dedicated lightgbm catboost catboost_dedicated random_forest extra_trees ensemble; do
    ./venv/bin/python scripts/core/model_trainer_${M}.py --commit
done
```

**9 trainer 共通模式(已驗證一致)**:
- CLI:`--dry-run`/`--commit`(互斥;預設 dry-run)、`--label-horizon`(預設 30)、`--panel-feature-sets`。SEED=5422 凍結於 policy(不開放 CLI)。
- `DEFAULT_PANELS` = 8 個 v0_5 panel:`fs_20250915` ~ `fs_20260415`(各 `_feature_set_v0_5`),8-panel walk-forward。
- DB 讀:universe(`core_universe_membership` JOIN `snapshot` WHERE `status='committed'` ∧ `core_tier='core_universe'`)+ `feature_values` + forward return(`TaiwanStockPriceAdj` 的 `LN(t1/t0)`)。
- **防禦過濾**:載入 query 含 `AND is_null_imputed IS NOT TRUE`(9/9 確認)。
- SPEC list = **37**(amihud 移除;變數名 `SPEC_43` 保留為命名債)。
- 產出:`data/models/<model_id>/metrics.json` + 模型二進位。metrics.json 含 in_sample_ic、cross_panel_ic_mean/std、sharpe、win_rate、mdd、mean_alpha、information_ratio、t_statistic、top_features、**verdict**。
- **verdict**(§14.7-CW 4 gate):Sharpe>0 ∧ Win≥50% ∧ MDD≤30% ∧ mean_alpha>0 全過 → `PERFECT`,否則 `WARNING`。

**各 trainer 差異**:

| 程式 | API / 二進位 | 備註 |
|---|---|---|
| `model_trainer_lgbm_v2.py` | native `lgb.train` / model.txt | PRODUCTION baseline;⚠️ universe query 少了其他 8 個有的 `snapshot_id=(latest)` 子查詢(多 snapshot 共存時的潛在不一致) |
| `model_trainer_xgboost.py` | native `xgb.train` / model.json | gain importance |
| `model_trainer_catboost.py` | `CatBoostRegressor` / model.cbm | l2_leaf_reg=3 |
| `model_trainer_ensemble.py` | LGBM+XGB+CatBoost / 3 子模型 | 等權平均;metrics 加 per_model + precision_top20_overlap + reliability_disagreement |
| `model_trainer_random_forest.py` | `RandomForestRegressor` / model.pkl | sqrt features,bootstrap=True,min_samples_leaf=30 |
| `model_trainer_extra_trees.py` | `ExtraTreesRegressor` / model.pkl | ET-vs-RF ablation |
| `model_trainer_lightgbm.py` | `LGBMRegressor` / model.txt | dedicated;num_leaves=20 |
| `model_trainer_xgboost_dedicated.py` | `XGBRegressor.fit` / model.json | dedicated |
| `model_trainer_catboost_dedicated.py` | `CatBoostRegressor` / model.cbm | +bootstrap_type=Bernoulli |

- 共通 hyperparams:200 trees / lr 0.05 / depth 5 / seed 5422;GBT 用 subsample/colsample 0.8;RF/ET 用 min_samples_leaf=30。
- ⚠️估計訓練時間:8-panel,每模型數秒~數分鐘(RF/ET 最慢)。觸發 §一.12 5-min 回報。

---

## PHASE 11 — 模型驗證(universe_completeness_schema + 9 multi-cycle validators)

### Step 11.1 — `universe_completeness_schema.py`(需 model_registry 已存在)

```bash
./venv/bin/python scripts/core/universe_completeness_schema.py --init
```
- 建 `prediction_run`、`predictions`、`universe_completeness_snapshot` + MV `universe_completeness_matrix_current`。前置含 `model_registry`(PHASE 10 已建)+ `feature_store_snapshot`。

### Step 11.2 — 9 個 multi-cycle validator(4-horizon walk-forward)

```bash
for V in validation xgboost_validation xgboost_dedicated_validation \
         lightgbm_validation catboost_validation catboost_dedicated_validation \
         random_forest_validation extra_trees_validation ensemble_validation; do
    ./venv/bin/python scripts/evaluation/multi_cycle_${V}.py --commit
done
```

**驗證器共通模式(已驗證一致)**:
- CLI:`--dry-run`/`--commit`(預設 dry-run)、`--horizons`(預設 `"5,20,60,252"`)、`--output`、**`--seed`(預設 5422,validator 有開放)**。
- 4 horizons:weekly 5d / monthly 20d / quarterly 60d / annual 252d。
- panel:`get_panel_dates()` 生成 ~95 個月中 panel(`fs_YYYYMMDD_feature_set_v0_5`,2018-06-15 → 2026-04-30 全歷史,非 trainer 的 8 panel)。walk-forward expanding window:`train [0..i-1] → test i`(min 100 train rows)。
- metrics/horizon:Sharpe(×√12)、win_rate、MDD、mean_alpha、IR、t_stat、mean_ic;7/9 另含 precision(方向命中率/top-20 overlap/RMSE/MAE)+ reliability(IC CoV)。`cost_per_rebal=0.006`、`panel_spacing=30`、`rebals_per_year=252/horizon`、honest annualization。
- overlap 校正:`horizon≤30 → n_eff=n`;else `n_eff=n×(30/horizon)`;`eff_t_stat=t_stat×√(n_eff/n)`。
- 產出:`reports/multi_cycle_<family>_<timestamp>.json`(`--output` 或 `--commit` 時寫)。**僅 `multi_cycle_validation.py`(LGBM)額外寫 `evaluation_log` DB 表**;其餘 8 個純 read-only。

### §11 驗收判準(⚠️ 注意 B4)

- **Code 唯一硬判定**:`is_significant = abs(eff_t_stat) > 1.997`(p<0.05)。
- **§14.7-CZ T_CZ-6 gate(Eff t≥4.20 / Sharpe≥2.40 / Win≥79%)為 docstring 參考,非 code-enforced** → 須**人工**對照各 validator 印出的 quarterly horizon metrics,判定是否達 T_CZ-6 production gate。
- **驗證器 self-contained**:每個 validator 在執行時自行 walk-forward 訓練 ~95 panel 的 fold,**不讀 `data/models/` 的訓練產物、不讀 `model_registry`** → 故 validator **不需要 PHASE 10 的模型先訓練完成**(兩軌獨立)。PHASE 10 產出 production artifacts + registry,PHASE 11 產出實證證據報告。

> **divergence 揭露(per §一.8)**:validator 的 `load_features` **未**套用 `is_null_imputed IS NOT TRUE`(只跳過 `val is None`)→ validator 倚賴 v0_5 panel 在 DB 層已無 imputed。若 PHASE 8 source-pure gate 確實執行,則 v0_5 panel 應已乾淨,此 divergence 無實質影響;若跳過 PHASE 8,則 trainer 與 validator 的 universe 純度會有落差。

---

## PHASE 12 — 週度 doctrine recommit(維運,非從零必需)

```bash
./venv/bin/python scripts/maintenance/run_weekly_doctrine_recommit.py --commit
```
- 編排器(subprocess orchestrator):FRED sync → API audit → `core_universe_builder --mode doctrine-native --with-feature-gate --with-reasonableness-gate --commit` → completeness/drift/feature audits。
- ⚠️ **非從零建構器**([Phase C-2 Pre-condition] 假設 schema/data/model 已存在)→ 僅用於既有系統的週度維運,不在從零序列內。

---

## 二、各程式說明(逐程式)

| 程式 | 角色 | 憲章治權 | PHASE |
|---|---|---|---|
| `scripts/core/path_setup.py` | 路徑 SSOT;`ensure_scripts_on_path()` bootstrap;realpath anchor 比對 | §0.0-I.10 | 0(import 期) |
| `scripts/config.py` | 全域設定;讀 FINMIND_TOKEN / DB creds(⚠️ GAP-0b) | — | 0 |
| `scripts/core/db_utils.py` | `get_db_connection()`(讀 DB_* env,connect_timeout=10);`get_core_stocks_from_db()` | §6.7 | 全程 |
| `scripts/core/data_schema.py` | 13 raw/infra 表 DDL;live API contract 探測 | §3.2 | 1 |
| `scripts/ingestion/initialize_market_data.py` | genesis `stocks` 表 | — | 1 |
| `scripts/core/core_universe_schema.py` | 7 universe 表 DDL | §6.7 | 1 |
| `scripts/core/feature_store_schema.py` | 3 feature 表 DDL | §14.7-CA | 1 |
| `scripts/ingestion/sovereign_sync_engine.py` | **唯一授權 ingestion 載體**;FinMind 10 表 + TaiwanStockInfo + FRED | §3.1 / §6.8 / §7 | 2,4 |
| `scripts/core/core_universe_builder.py` | CoreScore 5 層 + 3 gate;核心宇宙 SSOT | §6.4 / §14.7-CF/CB/CJ/DC | 3,6,8 |
| `scripts/core/feature_store_builder.py` | 37 feature(v0_5)SSOT;`is_null_imputed` flag | §14.7-CA/CB/DC | 7 |
| `scripts/audit/audit_full_db_vs_api_reconcile.py` | DB↔API byte-level 對帳 | §14.7-CE | 5 |
| `scripts/audit/audit_per_stock_feature_validity.py` | 每股 feature 完整度/range(硬閘) | §14.7-CI/CK/CL | 9 |
| `scripts/audit/audit_feature_data_quality_bias.py` | look-ahead/zero-fill/共線性(硬閘) | §14.7-CP/H4 | 9 |
| `scripts/audit/audit_feature_necessity.py` | feature 必要性 4-path(硬閘) | §14.7-CN | 9 |
| `scripts/audit/audit_feature_ic_vs_future_return.py` | IC vs forward return(advisory) | §14.7-CM | 9 |
| `scripts/audit/audit_feature_sign_stability.py` | 符號穩定性(advisory) | §14.7-CO/CQ/CR | 9 |
| `scripts/core/model_trainer.py` | base §10 trainer;建 model_registry/model_training_run | §14.7-BQ/CW | 10 |
| `scripts/core/model_trainer_*.py`(9) | tree family trainer;8-panel walk-forward | §14.7-CW | 10 |
| `scripts/core/universe_completeness_schema.py` | prediction/completeness 表 + MV | §14.7-BU | 11 |
| `scripts/evaluation/multi_cycle_*_validation.py`(9) | 4-horizon walk-forward 驗證 | §14.7-CY/CZ | 11 |
| `scripts/maintenance/run_weekly_doctrine_recommit.py` | 週度維運編排器 | §14.7-BX | 12(維運) |

---

## 三、執行順序總表(從零)

| # | PHASE | 指令摘要 | 阻斷性 | 預估時間 |
|---|---|---|---|---|
| 0 | 環境 | brew/apt deps + createdb + venv + .env + smoke test | 必須 | ~15 min |
| 1 | Schema | data_schema → initialize_market_data → core_universe_schema → feature_store_schema(各 `--init`)| 必須(依序)| ~2 min |
| 2 | 名冊 | `sovereign_sync_engine --seed` | 必須 | ~5 min |
| 3 | Bootstrap 宇宙 | `core_universe_builder --commit ... bootstrap`(無 gate)| 必須(破雞與蛋)| ~3 min |
| 4 | 全市場同步 | `sovereign_sync_engine --universe full --all ...` + `--source fred` | 必須 | ⚠️估計 6–10 hr(SHMM)|
| 5 | Raw audit | `audit_full_db_vs_api_reconcile --scope core` | 建議(資料完整性)| ⚠️估計 ~30–60 min |
| 6 | 最終宇宙 | `core_universe_builder --commit ... final`(無 gate)| 必須 | ~5 min |
| 7 | Feature store | `feature_store_builder --commit` | 必須 | ⚠️估計 ~10–30 min |
| 8 | 宇宙加 gate | `core_universe_builder --commit --with-*-gate`(⚠️先修 B1)| 建議(source-pure)| ~5 min |
| 9 | Feature audit | per-stock/quality/necessity(blocking)+ IC/sign(advisory)(⚠️先修 B2)| Blocking 3 個 | ⚠️估計 ~10–20 min |
| 10 | 訓練 | base `model_trainer --commit`(建表,⚠️B3)+ 9 family trainer | production 軌 | ⚠️估計 ~10–40 min |
| 11 | 驗證 | `universe_completeness_schema --init` + 9 validator | 證據軌(獨立)| ⚠️估計 數十分鐘~數小時(ensemble 最久)|
| 12 | 週度維運 | `run_weekly_doctrine_recommit --commit` | 非從零必需 | — |

> **PHASE 10 與 11 兩軌獨立**:validator self-contained,可與 trainer 任一順序執行(只要 PHASE 7 feature_values 已 committed)。

---

## 四、憲章充分性評估(per 用戶「如有不足先修訂憲章」)

### 4.1 充分的部分 ✅

- **同步序列**：§14.7-AM 已完整入憲「從零 → 全市場全歷史 + FRED」4 步序列範本(含雞與蛋解法 latest_registry_fallback + bootstrap_init/final 兩階段),PHASE 2–4 直接對齊。
- **各層治權**：feature engineering(§14.7-CA)、completeness gate(§14.7-CB)、reasonableness(§14.7-CJ)、source-pure(§14.7-DC)、empirical verification(§14.7-CE)、IC(§14.7-CM)、necessity(§14.7-CN)、model trainer contract(§14.7-BQ)、model metric gate(§14.7-CW)、multi-cycle validation(§14.7-CY/CZ)— **每個 PHASE 的治權條文都已存在**。

### 4.2 曾經不足的部分(charter gap)→ ✅ 已於 2026-05-31 入憲 §14.7-DD 補齊

**原 charter gap**：§14.7-AM 的「從零」序列範本**止於 Step 8(FRED sync)**;§14.7-CZ 雖補 environment→validation 8-phase 但以 single LGBM 為 Phase 7 anchor。**兩者皆未將 tree-family(9 trainers + 9 validators)「從零 → 模型產生 → 模型驗證」完整端到端序列明文化為單一範本**。各尾段治權條文雖散存(§14.7-CA/CB/CJ/CM/CN/BQ/CW/CZ/DC),但無一節編成 tree-family 序列範本。

**→ 已補齊**:**§14.7-DD「Tree-Family From-Zero → Model → Validation 12-PHASE 全序列治權範本」已於 2026-05-31 入憲**(用戶 AskUserQuestion Q1=「現在入憲 §14.7-DD」授權),將 PHASE 0–11 明文化為 charter 序列,延伸 §14.7-AM 的 4-step + §14.7-CZ 的 8-phase 至 tree-family 12-PHASE。

### 4.3 已落地的憲章修訂(§14.7-DD,2026-05-31 入憲完成)

**§14.7-DD「Tree-Family From-Zero → Model → Validation 12-PHASE 全序列治權範本」** 已 landed,內容:
1. PHASE 0–11 明文化為 charter 序列,各 PHASE cross-ref 既有 §14.7-X 治權(§14.7-CA/CB/CJ/CM/CN/BQ/CW/CZ/DC + §14.7-AM/CE);治權契約 + 新特性 7 條 + 入憲規則 7 條 + 證偽承諾 T_DD-1〜7。
2. B1/B2/B3/B4 四 code-level 發現入憲為治權揭露(§一.8 誠實):B1 = 本人 amihud 移除(SPEC 38→37)引入之 self-introduced regression(**已 code-fix**);B2 = audit staleness + hardcode snapshot(**已 code-fix** 為 37 + dynamic `pick_forward_window()`);B3 = model_registry DDL idempotent / 由 base trainer 建(resolved by ordering,無需改 code);B4 = T_CZ-6 為 docstring-reference 非 code-gate 之治權釐清(人工裁決)。
3. 雙層治權鎖(per T_DC-6):主憲章 §14.7-DD + CLAUDE.md **§一.14** 同次入憲。

> ✅ **授權與執行邊界(per §二.2 + §一.8)**:憲章 §14.7-DD + CLAUDE.md §一.14 入憲已依用戶 Q1 授權完成;B1+B2 code 修正已依用戶 Q2 授權「修正 B1-B3」完成(**僅改 code,py_compile 全 PASS,未 git commit、未 DB write、未 retrain**)。git commit / DB write / 重訓 仍各需 §二.2 explicit 授權,**本 runbook 未執行**。

---

## 五、Long-running workflow 治權(PHASE 4 / 10 / 11)

- **§二.6 SHMM**(≥30 min)：N≥3 個非整數倍對齊 Monitor heartbeat(15/20/25/30 min)+ sentinel timestamp + 60s watchdog + self-healing。PHASE 4 全市場同步必備。
- **§一.12 5-min 回報**(≥5 min 任何 AI-triggered task)：每 5 分鐘回報已完成階段 + elapsed + 剩餘估計 + 已知 metrics + warning。PHASE 4/10/11 適用。
- 建議以 `tmux` 或 `systemd-run --user` 背景執行,主控以 Monitor sleep loop 為主軸(CronCreate 不可獨用)。

---

## 六、資料真實性附註(per §一.10)

- ✅ DB-verified（活 PostgreSQL READ-ONLY）：committed snapshot 398 core / 603 quarantine / 2803 total（policy v0.18）；feature_set_v0_5 = 37 distinct features / imputed-among-core = 0；8 DEFAULT_PANELS = fs_20250915 ~ fs_20260415。
- ✅ Code-verified（grep + Read）：B1（[core_universe_builder.py:2495](scripts/core/core_universe_builder.py:2495) `n >= 38` vs 37-name list）；B3（[model_trainer.py:266](scripts/core/model_trainer.py:266) DDL）；B4（[multi_cycle_xgboost_validation.py:283](scripts/evaluation/multi_cycle_xgboost_validation.py:283) 唯一硬判定 1.997）。
- ⚠️ 估計（未實跑 profiling）：PHASE 4 全市場 6–10 hr；PHASE 5/7/9/10/11 之分鐘數 — 皆為依程式結構推估，**非 deterministic fact**，實跑請以各程式 stdout 的 `Total elapsed` 為準。
- FinMind sponsor tier（6000/hr / 30-min reset）為 §C 外部資源 protocol 之既有事實，重建時請以 `/api/v4/user_info` 實查確認 tier 仍有效。
