# 跨機接續快照 — 多模型「能否賺錢」驗證工作流 (2026-06-02)

**文件性質**：換機接續 handoff（本機 macOS → 另一台電腦）。記錄當前執行階段、未提交資產、環境重建步驟、下一步 standing order。
**資料真實性**：本檔所有數字皆 trace 回 (a) 程式 stdout / (b) git/檔案系統實查 / (c) JSON `_meta` 實讀（§一.10）。檢查時點 = 2026-06-02 ~07:10 CST。
**位階**：協作接續用，不上升治權；治權以憲章 + CLAUDE.md 為準。

---

## ⚠️ 〇、換機最關鍵警告（先讀這段）

換機 `git clone` / `git pull` **拿不到以下資產**，必須手動轉移（rsync / scp / 隨身碟），否則工作中斷無法接續：

### A. 4 個「未追蹤」檔（未 commit、不在 origin）

| 檔案 | 角色 | 狀態 |
|---|---|---|
| `scripts/evaluation/build_model_comparison_report.py` | Step1 跨模型比較產生器（leaderboard 引擎）| 未追蹤 |
| `scripts/evaluation/multi_cycle_timesfm_validation.py` | Step3 TimesFM validator（已 py_compile PASS，未跑）| 未追蹤 |
| `scripts/evaluation/multi_cycle_stockformer_validation.py` | Step3 Stockformer validator（已 py_compile PASS，未跑）| 未追蹤 |
| `scripts/evaluation/multi_cycle_hist_validation.py` | Step3 HIST validator（已 py_compile PASS，未跑）| 未追蹤 |
| `reports/compliance_audit_20260601_1831.md` | 稽核報告 | 未追蹤 |

> git branch = `master`，與 `origin/master` **同步（ahead 0 / behind 0）**。最新 commit = `fb25ac1`（v6.26.2 sealed，37 source-pure features / v0_5）。即committed 程式（含 tree validators + chronos/itransformer/patchtst/tft/transformer_dedicated validators）換機 pull 後都在；**唯獨上表 5 檔不在**。

### B. 所有驗證結果 JSON 被 gitignore（不在 git）

`reports/multi_cycle_*.json` 經 `git check-ignore` 確認**全部被忽略** → 換機後**所有已跑出的結果會消失**。需手動轉移的關鍵 JSON：

- Step2 time-series（本批，n_universe=914）：`reports/multi_cycle_{chronos,itransformer,patchtst}_20260601_193610.json`（已完成）+ `transformer_dedicated_20260601_193610.json`（**跑完才會生成**）
- Step1 tree-family（n_universe=397）：`reports/multi_cycle_{validation,lightgbm,xgboost,xgboost_dedicated,catboost,catboost_dedicated,random_forest,extra_trees,ensemble}_20260601_1[45]*.json`

> **建議換機前先在本機把這 5 檔 + 全部 `reports/multi_cycle_*_20260601_*.json` 打包帶走；或經 §二.2 授權後 commit + push（含 gitignore 例外處理）。**

---

## 一、專案總目標（為何在做這些）

對核心股建立並驗證 **5 大類股價預測模型**，**用真實 DB 資料**確認**能否賺錢**：

1. **Tree**：LightGBM / XGBoost / CatBoost（+ dedicated 變體）
2. **Transformer**：TFT / iTransformer / PatchTST / transformer_dedicated
3. **Foundation**：TimesFM / Chronos
4. **Stock-specific**：Stockformer / HIST
5. **Ensemble**：multi-tree

每模型一支獨立程式，**不在 AI 環境跑**，由本系統程式對真實 PostgreSQL 實算。產出方法論 + 驗證報告 + 從 DB 取的真實股價變化 + 精準度/信任度分析，**共同比較基準**。用戶授權「全部，分階段 A→B→C」+「沿用入憲 T_CZ-6 框架」。

---

## 二、共同比較基準（T_CZ-6 + 4-horizon，SSOT 已鎖）

SSOT 文件：`reports/model_comparison_baseline_spec_20260601.md`（v1.0）。框架實作 SSOT：`scripts/evaluation/multi_cycle_validation.py`。

- **4 horizon walk-forward**：週 5d / 月 20d / 季 60d / 年 252d，expanding-window OOS（第 i 面板只用 [0..i-1] 訓練）
- **Portfolio**：每面板取 top-20 等權 long-only vs 全宇宙等權；alpha = top20 − univ
- **指標**：Spearman rank IC、Sharpe、Win rate、α、IR、Eff t（overlap 校正 n_eff）
- **成本**：cost_per_rebal=0.006；panel_spacing=30（月度格）；seed=5422
- **n_eff overlap 校正**：horizon ≤ 30 → n_eff=n；否則 n_eff = n×(30/horizon)。實測 weekly≈65 / monthly≈65 / quarterly≈32 / annual≈7.3（年線樣本脆弱，須附 caveat）
- **T_CZ-6 production 門檻**：Eff t ≥ 4.20 **AND** Sharpe ≥ 2.40 **AND** Win ≥ 79%（三者全過才算 production 級）。validator 內只設 `is_significant_p05 = abs(eff_t) > 1.997`；**完整 T_CZ-6 由 `build_model_comparison_report.py` 套用**。

### ⚠️ universe 不一致（誠實標記，務必知悉）

JSON `_meta` 實讀：

| 模型族 | n_universe | 來源 |
|---|---|---|
| tree-family（9 支）| **397** | core_tier='core_universe' |
| time-series（chronos/itransformer/patchtst/transformer_dedicated）| **914** | validator `get_universe()` 取較廣 committed 成員 |

→ 兩族**不在同一 universe**。`build_model_comparison_report.py` 依 `(model_key, n_universe)` 分組，故 tree(397) 與 time-series(914) 自動落在不同組（行為正確，但**不可宣稱兩族 apples-to-apples**）。換機接續若要嚴格同 universe 比較，需統一 `get_universe()` 定義後重跑其中一族。**此差異尚未解決，須在最終報告明確揭露。**

---

## 三、當前執行階段（A→B→C 進度）

| Step | 內容 | 狀態 |
|---|---|---|
| Step 0 | 鎖定共同基準 SSOT 文件 | ✅ DONE（`model_comparison_baseline_spec_20260601.md`）|
| Step 1 | 比較產生器 + tree leaderboard | ✅ DONE（9 tree JSON 已跑，n_universe=397）|
| Step 2 | 跑未跑過的 transformer/foundation validators（Option A 忠實全跑）| 🚧 **3/4 完成** |
| Step 3 | 新建 TimesFM/Stockformer/HIST 程式 | 🚧 **3 支已寫好 + py_compile PASS，未跑** |
| TFT 修復 | venv_fm 補 pytorch_forecasting+lightning | ⏳ pending（task #43）|

### Step 2 細節（本機正在跑）

執行 log：`reports/_step2_nontree_run_20260601_193610.log`（marker `[STEP2]`）。批次 TS = `20260601_193610`。

- ✅ chronos DONE（19:42:59）→ `reports/multi_cycle_chronos_20260601_193610.json`
- ✅ itransformer DONE（02:10:48）→ `..._itransformer_...json`
- ✅ patchtst DONE（02:32:11）→ `..._patchtst_...json`
- 🚧 **transformer_dedicated 執行中**（本機 PID 70605，`./venv_fm/bin/python ... --dry-run --seed 5422`）

**transformer_dedicated 為 fold-based**，每 horizon 約 64-65 folds，依序 weekly→monthly→quarterly→annual。截至 ~07:10 檢查（健康 STAT=RN / %CPU≈384 / elapsed≈4h38m）：

| horizon | folds | 結果（誠實）|
|---|---|---|
| weekly | 65/65 ✅ | Sharpe +0.31｜Win 58.5%｜α −0.20%｜Eff t **−0.828** ❌ 不賺錢 |
| monthly | 65/65 ✅ | Sharpe +1.17｜Win 66.2%｜α +0.96%｜Eff t **+1.900** ❌ 未過 p<0.05 與 T_CZ-6 |
| quarterly | 40/64 🚧 | last IC=+0.1480（季線 IC 轉強）|
| annual | 0/~65 ⏳ | 尚未開始 |

> **observed 放緩**：per-fold 隨 expanding-window 訓練量增 + 疑似長時間 CPU 飽和熱節流而漸慢（weekly 末段 ~105s/fold；monthly 末 ~200s/fold；quarterly 30→40 約 29 分）。CPU 全程未 stall。**完成時間顯著晚於原估 ~3.5hr。**

> **換機注意**：PID 70605 為本機 process，**不會跟著換機**。換機後 transformer_dedicated 須**重跑**（`./venv_fm/bin/python scripts/evaluation/multi_cycle_transformer_dedicated_validation.py --dry-run --seed 5422 --output reports/multi_cycle_transformer_dedicated_<新TS>.json`）。

### Step 3 細節（已寫好，等 venv_fm 空出才跑）

3 支已建妥 + `python -m py_compile` PASS，**尚未執行**（venv_fm 被 transformer_dedicated 佔用，不可同時 pip / 重 IO）：

- `multi_cycle_timesfm_validation.py`（Foundation, zero-shot, `google/timesfm-2.0-500m-pytorch`，首跑會下載 ~2GB 到 HF cache）
- `multi_cycle_stockformer_validation.py`（Stock-specific 時序 transformer, RevIN+SeriesDecomp）
- `multi_cycle_hist_validation.py`（Graph-based, 用 `TaiwanStockInfo.industry_category` 當對稱圖拓樸，**不寫 feature_values、無 hardcoded score**）

3 支皆 verbatim 沿用 patchtst 的 Group-A 共同基準函式（get_universe→914 / get_panel_dates→95 mid-month / load_forward_returns / aggregate_horizon / spearman_ic）。

---

## 四、環境（換機必重建）— venv 雙環境分裂

⚠️ **本專案有兩個 venv，缺一不可**。CPU-only（cuda/mps 皆 False）。py 3.12.13。

### `./venv`（主環境，tree validators + 比較產生器 + DB ops）
- numpy **2.3.5**（會 break numpy-1.x 編譯的模組，例如 torch 2.2.2）
- 有 pytorch_forecasting + lightning
- **用途**：9 tree validators、`build_model_comparison_report.py`、DB 讀寫、py 工具

### `./venv_fm`（時序/foundation 環境）
- numpy **1.26.4** + torch **2.2.2**（可用）+ transformers **4.57.6** + einops **0.8.2**
- **`TimesFmModelForPrediction` native 可用**（transformers 內建，已驗證 import OK）
- chronos 可用
- ❌ **缺 pytorch_forecasting + lightning**（confirmed import 失敗）→ 這是 **TFT 的 blocker**
- **用途**：chronos / itransformer / patchtst / transformer_dedicated / Step3（timesfm/stockformer/hist）

### 重建要點（換機）
1. 兩 venv 都建在專案根（`./venv`、`./venv_fm`），py 3.12。
2. **OS 原生依賴**（per CLAUDE.md §二.7）：macOS `brew install libomp postgresql@17`；Linux `apt-get install -y libgomp1 libpq-dev`。
3. venv_fm 鎖 numpy 1.26.4 + torch 2.2.2（**勿升 numpy 到 2.x**，會 break torch）。
4. import smoke test：`python -c "import psycopg2,pandas,polars,numpy,requests,sklearn,xgboost,lightgbm; print('OK')"`。
5. **TFT 修復（task #43）**：`./venv_fm/bin/pip install pytorch_forecasting lightning`，裝完**驗證 torch 仍 2.2.2 / numpy 仍 1.26.4**（pip 可能想升 numpy，要擋），再跑 `multi_cycle_tft_validation.py`。
6. ⚠️ **venv_fm 有 process 在跑時，禁止 pip install**（會壞環境）。

### DB 連線
- validators 用 `from core.db_utils import get_db_conn`；臨時 read-only /tmp script 用 `sys.path.insert(0, ".../scripts")` 後 `from core.db_utils import get_db_conn`。
- `.env` 在專案根（含 DB_HOST/PORT/NAME/USER/PASSWORD 等），**值絕不外印**。換機需確認新機 `.env` 與本機 PostgreSQL 一致（或同步 DB dump）。
- ⚠️ **本機 DB 與另一機是否同一份**？若換機是另一份 PostgreSQL，須先確認 committed universe snapshot 一致（最新 commit 提及 v0.18 / 397 core / 96 panels / v0_5 / 37 features），否則結果不可比。

---

## 五、下一步 standing order（換機後接續執行順序）

> 前提：transformer_dedicated 已跑完（log 出現 `[STEP2] ALL DONE`，或換機後重跑完成並生成 `transformer_dedicated` JSON）。

1. **完成處理**：parse 4 個 `reports/multi_cycle_{chronos,itransformer,patchtst,transformer_dedicated}_<TS>.json`。
2. 用 **`./venv/bin/python scripts/evaluation/build_model_comparison_report.py`** 重新產生 `reports/model_comparison_20260601.md` + `.json`（transformer/foundation 的 IC 讀 `rank_ic_mean`），涵蓋 tree + transformer + foundation。
   - ⚠️ 注意 universe 分組（tree 397 / time-series 914，見 §二）。
3. **TFT venv 修復（task #43）**：`./venv_fm/bin/pip install pytorch_forecasting lightning` → 驗證 torch 2.2.2 / numpy 1.26.4 → 跑 `multi_cycle_tft_validation.py`。
4. **Step3（task #42）**：`./venv_fm/bin/python` 跑 3 支 → `reports/multi_cycle_{timesfm,stockformer,hist}_<TS>.json`。跑完併入 leaderboard。
5. 最終：方法論 + 5 類模型完整 leaderboard + 精準度/信任度 + universe 差異揭露 + 賺錢裁決報告。

---

## 六、治權約束（換機仍適用，VERBATIM）

- **未經 §二.2 授權，不 git commit、不 git push、不開 PR**。
- **不手動補 DB 值**（[[no-manual-data-fill]]）：只能 DELETE 過期/重複列或改 writer code 重 build；**永不** hand-fill/UPDATE/INSERT 補值。
- **§一.8 報告誠實**：描述「做了什麼」，工具失敗要明說，不佯稱完成。
- **§一.10 No Data Hallucination**：每個數字 trace 回 (a) 程式輸出 / (b) DB query / (c) API response。
- **§一.12**：≥5min 工作每 5 分鐘回報；**§二.6 SHMM**：≥30min 工作要 N≥3 Monitor heartbeat + sentinel + watchdog。
- **§一.11**：新 Python script 需三段式 docstring 標頭（核心定義 / 功能群矩陣 / 修訂歷程）。Step3 三支若要正式 commit，須補齊標頭。
- 機密值（DB_PASSWORD / FINMIND_TOKEN / FRED_API_KEY / GITHUB_TOKEN / GEMINI_API_KEY）**只讀進 env var，絕不外印/echo**。
- 所有編輯指向專案根（本機 `/Users/hugo/project/stock_backend/`；Linux 另一機對應 `/home/<user>/...`），**不寫入 `.claude/worktrees/`**。
- venv_fm 忙碌時**不 pip install**。

---

## 七、檔案索引（換機核對清單）

### 程式（committed，pull 即有）
- `scripts/evaluation/multi_cycle_validation.py`（base，框架 SSOT）
- 8 支 tree validator（lightgbm/xgboost/xgboost_dedicated/catboost/catboost_dedicated/random_forest/extra_trees/ensemble）
- `multi_cycle_{chronos,itransformer,patchtst,tft,transformer_dedicated}_validation.py`

### 程式（未追蹤，務必手動帶走）
- `scripts/evaluation/build_model_comparison_report.py`
- `scripts/evaluation/multi_cycle_{timesfm,stockformer,hist}_validation.py`

### 文件 / 結果
- `reports/model_comparison_baseline_spec_20260601.md`（committed）
- `reports/_step2_nontree_run_20260601_193610.log`（Step2 執行 log）
- `reports/multi_cycle_*.json`（**gitignored，務必手動帶走**）
- 本檔 `reports/cross_machine_handoff_20260602.md`

### task 進度（task tool）
- #1–#40 completed
- #41 Step2 in_progress（3/4，transformer_dedicated 跑中）
- #42 Step3 in_progress（3 支已建未跑）
- #43 TFT venv 修復 pending
