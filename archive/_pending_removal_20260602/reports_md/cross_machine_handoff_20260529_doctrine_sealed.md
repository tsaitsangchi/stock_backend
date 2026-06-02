# 跨機接續完整 Handoff:§14.7-DC v0.7 Final Convergence + v0.20 9-Tree Multi-Cycle Complete(2026-05-29 v6.23.7 sealed)

**Subject**:用戶換另一台電腦繼續本專案之 cross-machine session handoff
**Sealed checkpoint**:`v6.23.7-source-pure-final-convergence-9tree-complete-20260529`
**HEAD commit**:`e9aba76`
**Charter**:`reports/系統架構大憲章_v6.1.0.md`(§14.7-DC v0.7)
**對應任務**:#116 v0.20 multi-cycle 完整收尾(已 completed)

---

## 一、本封存點核心狀態(快照)

### 1.1 治權狀態(charter doctrine state)

- **§14.7-DC Source-Pure Universe Doctrine v0.7 Final Convergence Ratification** 已落地
- **SPEC_38 = strict source-pure convergence point**(per T_DC-18 Convergence Treaty)
- **18 證偽承諾(T_DC-1 ~ T_DC-18)全 inscribed**;T_DC-8 silent fallback flag pending(empirically 0 NULL)
- 主憲章 § + CLAUDE.md §一.13 雙鎖入憲 + 本 master summary v6 三層治權鎖

### 1.2 Universe 狀態(DB query verified per §一.10)

```
snapshot_id     : core_universe_20260529_core_universe_policy_v0_17_source_pure_doctrine
policy_version  : core_universe_policy_v0.17_source_pure_doctrine
status          : committed
total_candidates: 2,799
core_count      : 910 ⭐(train-eligible)
quarantine_count: 211(imputed margin_ratio_60d 強制排除)
```

### 1.3 SPEC 狀態(source script verified per §一.10)

- **SPEC_38** in `scripts/core/model_trainer_lightgbm.py`(變數仍命名 SPEC_43 / actual len=38)
- 跨 20 scripts 同步(10 trainer + 10 multi-cycle validator)
- **Tier 0-2 only**(Tier 4/5 0 use)→ 全 mathematically pure

```
§0.1 第一性原理: 29 features
  - Price/Returns 6 + Risk/Vol 7 + Liquidity 5 + Value 3 + Quality 6 + Growth 2
§0.2 八二法則:   4 features
  - Pareto Topology 3 + Cross-Section 1
§0.3 康波週期:   5 features
  - Microstructure Flows 5
TOTAL:           38 features ✅
```

### 1.4 Model 狀態(9-tree × 4-horizon multi-cycle complete)

| Rank | Model | Annual Sharpe | Annual Win | Annual EffT | T_CZ-6 ≥ 4.20 |
|---|---|---:|---:|---:|---|
| 🥇 | **XGB v0.1 既存** | **6.051** | **98.4%** | **4.612** | **✅** ⭐⭐⭐ |
| 🥈 | XGBoost dedicated v0.1 | 5.644 | 96.7% | 4.369 | ✅ ⭐⭐ |
| 🥉 | Ensemble v0.1 | 5.305 | 91.8% | 4.070 | ❌ ⭐ |
| 4 | LGBM v0.2 production | 5.281 | 91.8% | 4.027 | ❌ |
| 5 | LightGBM dedicated v0.1 | 5.279 | 90.2% | 4.150 | ❌ |
| 6 | CB v0.1 既存 | 4.737 | 91.8% | 3.548 | ❌ |
| 7 | CatBoost dedicated v0.1 | 4.658 | 90.2% | 3.356 | ❌ |
| 8 | Random Forest v0.1 | 4.330 | 90.2% | 2.875 | ❌ |
| 9 | Extra Trees v0.1 | 4.256 | 88.5% | 2.412 | ❌ |
| — | Transformer dedicated | aborted | — | — | defer v6.24.x |

⭐⭐⭐ **2/9 models pass T_CZ-6 annual EffT ≥ 4.20**
⭐⭐ Strictest source-pure SPEC_38 v0.20 與 production 表現完全不衝突
⭐ 9 models 全 annual Sharpe ≥ 4.26 / Win ≥ 88.5%

---

## 二、7 user directives → 6 corrective bundles → SPEC_38 收斂演進

| Iteration | Date | Trigger | Action | SPEC | tag |
|---|---|---|---|---|---|
| 1 | 2026-05-29 | imputed exclusion | -211 imputed stocks → 910 universe | SPEC_43 | v6.23.0 |
| 2 | 2026-05-29 | THEME_KEYWORDS hardcoded scores | -theme_strength | SPEC_42 | v6.23.0 |
| 3 | 2026-05-29 | transitive + keyword choice | -fitness_signal_60d -theme_is_semiconductor | SPEC_40 | v6.23.1 |
| 4 | 2026-05-29 | charter-mandated hardcoded thresholds | -barbell_balance_60d -right_tail_concentration_60d | **SPEC_38** | v6.23.2 |
| 5 | 2026-05-29 | Concept vs Specific Value clarification | retroactive validation of 1-4 | SPEC_38 | v6.23.3 |
| 6 | 2026-05-29 | 5-tier Hardcoded Value Catalog | systematic Tier 0-5 audit | SPEC_38 | v6.23.4 |
| 7 | 2026-05-29 | Final Convergence Ratification | T_DC-18 inscribed | SPEC_38 ⭐ | **v6.23.7** |

---

## 三、Git 狀態(完整 audit trail)

### 3.1 最近 commits(v6.23.x 全 sequence)

```
e9aba76 docs+empirical: §14.7-DC v0.7 9-tree × 4-horizon multi-cycle complete + master summary v6
8967d9f docs: Feature × Universe State Verification on §14.7-DC v0.7 + .gitignore allowlist 補充
743197e docs+governance: §14.7-DC v0.7 final-convergence-ratification — SPEC_38 convergence point + T_DC-18
5fc6db1 docs: 補追 v0.20 strictest-of-strictest partial multi-cycle JSON(LightGBM ded + XGBoost ded)
ee63a8e docs+governance: §14.7-DC v0.6 ultimate-final 補強 — Comprehensive 5-tier Hardcoded Value Catalog
1528e13 docs+governance: §14.7-DC v0.5 ultimate-strictest 補強 — Concept vs Specific Value clarification
1e65206 docs: 補追 v0.19 partial multi-cycle JSON(LightGBM dedicated only)
02fa451 feat+docs+governance: §14.7-DC v0.4 strictest-of-strictest 補強 + corrective bundle IV(SPEC_38)
```

### 3.2 最近 tags(全 v6.23.x sealed checkpoints)

```
v6.23.7-source-pure-final-convergence-9tree-complete-20260529       ⭐ HEAD
v6.23.6.1-feature-universe-state-verification-20260529
v6.23.6-source-pure-final-convergence-ratification-20260529
v6.23.5-source-pure-strictest-multi-cycle-partial-20260529
v6.23.4-source-pure-ultimate-final-5tier-catalog-20260529
v6.23.3-source-pure-ultimate-strictest-concept-vs-value-doctrine-20260529
v6.23.2-source-pure-strictest-of-strictest-corrective-bundle-IV-20260529
v6.23.1-source-pure-strictest-corrective-bundle-III-20260529
```

### 3.3 GitHub URL

- Repo: https://github.com/tsaitsangchi/stock_backend
- HEAD: https://github.com/tsaitsangchi/stock_backend/commit/e9aba76
- v6.23.7 release: https://github.com/tsaitsangchi/stock_backend/releases/tag/v6.23.7-source-pure-final-convergence-9tree-complete-20260529

---

## 四、新機器 setup 步驟(per §0.0-I.9 / §0.0-I.10 / §14.7-AU 跨平台環境前置)

### 4.1 OS-level dependencies(per CLAUDE.md §二.7)

```bash
# macOS
brew install libomp postgresql@17

# Linux(Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libgomp1 libpq-dev postgresql-17

# Windows
# 通常無需(內含 vcomp140.dll)
```

### 4.2 Repo pull

```bash
cd ~/project  # 或新機器之 PROJECT_ROOT 父目錄
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend

# Pull all tags
git fetch --tags

# Checkout v6.23.7 sealed checkpoint
git checkout v6.23.7-source-pure-final-convergence-9tree-complete-20260529

# Optional: switch back to master HEAD if continuing work
git checkout master
git pull
```

### 4.3 .env 設定(per §0.0-I.10 跨平台路徑)

```bash
# Copy template + 修改路徑
cp .env.example .env

# .env 必須含(per CLAUDE.md §二.7):
# PROJECT_ROOT=<absolute path to stock_backend>(必須對齊本機物理路徑)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=stock_db
# DB_USER=<your_user>
# DB_PASSWORD=<your_password>
# FINMIND_TOKEN=<sponsor tier token>
# FRED_API_KEY=<your fred key>
# ANTHROPIC_API_KEY=<your anthropic key>(optional)
```

⚠️ **macOS 注意**:`/home/<user>` 為 `/Users/<user>` 之 symlink,`path_setup.py v4.47+` 用 `os.path.realpath()` 解析,新機器 `PROJECT_ROOT` 必須對齊本機物理路徑(macOS 應用 `/Users/<user>/project/stock_backend`)。

### 4.4 venv + pip dependencies

```bash
# 建 venv
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate.bat  # Windows

# 安裝 requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.5 Import smoke test(per CLAUDE.md §二.7 必須通過)

```bash
python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm, catboost, torch; print('✅ all imports OK')"
```

⚠️ 失敗(如 `OSError: libomp.dylib not found`)→ 補 OS-level dependency 再 retry。

### 4.6 DB 設定(從 backup restore 或新建)

```bash
# Option A: restore from backup(若新機器要繼續使用相同 historic data)
psql -U postgres -d stock_db < /path/to/dump.sql

# Option B: 新建 DB + 跑全 sync(從 zero)
psql -U postgres -c "CREATE DATABASE stock_db;"
python scripts/maintenance/universe_completeness_schema.py --init

# 然後 from-zero sync(per §14.7-CZ Build Sequence Doctrine):
python scripts/ingestion/sovereign_sync_engine.py --mode commit  # ~9-15 hours
```

### 4.7 DB 狀態 verification(per §一.10)

```bash
psql -U <user> -d stock_db -c "
SELECT snapshot_id, policy_version, core_count, quarantine_count, status
FROM core_universe_snapshot WHERE status='committed'
ORDER BY created_at DESC LIMIT 1;"
# Expected: core_universe_20260529_core_universe_policy_v0_17_source_pure_doctrine / 910 / 211 / committed
```

---

## 五、SPEC × Universe 雙 source 驗證(本封存點之 v6.23.6.1 audit trail)

per `reports/feature_universe_state_verification_20260529.md`:

```bash
# Source (a) Program file:38 features
grep -A 50 "^SPEC_43 = \[" scripts/core/model_trainer_lightgbm.py | head -60

# Source (b) DB query:910 stocks
psql -c "SELECT core_count FROM core_universe_snapshot
        WHERE status='committed' ORDER BY created_at DESC LIMIT 1;"

# Expected:雙 source 對齊 38 × 910 = 34,580 cells / 0 imputed / Tier 0-2 only
```

---

## 六、9-tree empirical results 重現(per §14.7-CY multi-cycle protocol)

### 6.1 重現單一 model multi-cycle(deterministic)

```bash
source .venv/bin/activate

# Re-run XGB v0.1 既存(target Sharpe 6.051)
python scripts/evaluation/multi_cycle_xgboost_validation.py \
  --output reports/multi_cycle_xgboost_v01_v020_20260529_reproduced.json

# Compare
diff <(jq -S . reports/multi_cycle_xgboost_v01_v020_20260529.json) \
     <(jq -S . reports/multi_cycle_xgboost_v01_v020_20260529_reproduced.json)
# Expected:0 diff(deterministic on same input)
```

### 6.2 全 9-tree 重跑(reference timing 60m24s on 8-core / 16GB)

```bash
# 用既有 background script(per §一.12 5-min reporting)
nohup bash -c "
  python scripts/evaluation/multi_cycle_lightgbm_validation.py --output reports/multi_cycle_lightgbm_v020_$(date +%Y%m%d).json
  python scripts/evaluation/multi_cycle_xgboost_dedicated_validation.py --output reports/multi_cycle_xgboost_dedicated_v020_$(date +%Y%m%d).json
  python scripts/evaluation/multi_cycle_catboost_dedicated_validation.py --output reports/multi_cycle_catboost_dedicated_v020_$(date +%Y%m%d).json
  python scripts/evaluation/multi_cycle_random_forest_validation.py --output reports/multi_cycle_random_forest_v020_$(date +%Y%m%d).json
  python scripts/evaluation/multi_cycle_extra_trees_validation.py --output reports/multi_cycle_extra_trees_v020_$(date +%Y%m%d).json
  python scripts/evaluation/multi_cycle_ensemble_validation.py --output reports/multi_cycle_ensemble_v020_$(date +%Y%m%d).json
  python scripts/evaluation/multi_cycle_validation.py --output reports/multi_cycle_lgbm_v2_v020_$(date +%Y%m%d).json
  python scripts/evaluation/multi_cycle_xgboost_validation.py --output reports/multi_cycle_xgboost_v01_v020_$(date +%Y%m%d).json
  python scripts/evaluation/multi_cycle_catboost_validation.py --output reports/multi_cycle_catboost_v01_v020_$(date +%Y%m%d).json
" > /tmp/all_trees_mc.log 2>&1 &
```

---

## 七、Pending tasks(non-blocking,可在新機器 resume)

### 7.1 P0(本封存點未含)

- 無 P0 pending(v6.23.7 為 final-final closure)

### 7.2 P1(可延後)

| # | 任務 | 對應治權 | 預期 effort |
|---|---|---|---|
| 1 | Transformer dedicated v0.20 rebuild + multi-cycle | §14.7-DC v0.7 + §14.7-CY | ~10-12h(per-fold time 涉 expanding window)|
| 2 | T_DC-8 silent fallback flag 補(builder L345/L364/L377/L391/L420-421/L442/L496/L845)| §14.7-DC v0.7 | ~1-2h |
| 3 | Annual horizon 邊界 model hyperparameter tuning | §14.7-CW T_CW-6 + §14.7-CZ T_CZ-6 | ~4-6h per model |

### 7.3 P2(future research)

| # | 任務 | 對應治權 |
|---|---|---|
| 1 | Cross-pillar interaction features(§0.1 × §0.3 / Pareto × Macro)| §14.7-CB Phase D-2 |
| 2 | Multi-cycle weekly horizon dedicated optimization | §14.7-CY |
| 3 | Down-stream production sync + portfolio sizer v0.20 alignment | §9.2 + §14.7-CU |

---

## 八、Background tasks(本 session 結束前)

```
無 background python 程序執行中(multi-cycle 全 9/9 完成 @ 16:57:21)
無 active Monitor task(全 task 已完成或 timed out)
無 ScheduledWakeup / CronCreate task
```

---

## 九、Key files reference(新機器接續時優先讀)

### 9.1 Charter / doctrine

```
reports/系統架構大憲章_v6.1.0.md       # 主憲章(7,344 行,Operations Reality Patch)
  - §14.7-DC v0.7                        # Source-Pure Universe Doctrine(v0.1~v0.7)
  - §14.7-CY                              # Multi-Cycle Horizon Validation Doctrine
  - §14.7-CZ                              # T_CZ-6 Reality Check
CLAUDE.md                                 # AI 協作工具規則 SSOT
  - §一.10                                # No Data Hallucination
  - §一.11                                # Mandatory Header Three-Section
  - §一.12                                # 5-min Progress Reporting
  - §一.13                                # Source-Pure Universe Doctrine(雙鎖 with §14.7-DC)
```

### 9.2 Empirical evidence

```
reports/tree_family_10_model_v6_v020_multi_cycle_complete_master_summary_20260529.md   ⭐ HEAD master summary v6
reports/tree_family_10_model_v5_v020_strictest_of_strictest_master_summary_20260529.md
reports/feature_universe_state_verification_20260529.md  # 38 × 910 雙 source audit
```

### 9.3 v0.20 multi-cycle JSON files(9 個)

```
reports/multi_cycle_lightgbm_v020_20260529.json
reports/multi_cycle_xgboost_dedicated_v020_20260529.json
reports/multi_cycle_catboost_dedicated_v020_20260529.json
reports/multi_cycle_random_forest_v020_20260529.json
reports/multi_cycle_extra_trees_v020_20260529.json
reports/multi_cycle_ensemble_v020_20260529.json
reports/multi_cycle_lgbm_v2_v020_20260529.json
reports/multi_cycle_xgboost_v01_v020_20260529.json
reports/multi_cycle_catboost_v01_v020_20260529.json
```

### 9.4 Production scripts(per CLAUDE.md §一.11 mandatory 3-section header)

```
scripts/ingestion/sovereign_sync_engine.py  # §一.11 範本(L1-230)
scripts/core/
  ├── core_universe_builder.py               # §6.7 + §14.7-CC/CD/CG
  ├── feature_store_builder.py               # §0.0-D + §14.7-CA/CK/CL
  ├── model_trainer_*.py(× 10)               # SPEC_38 同步
  ├── prediction_engine.py                    # §9.1
  └── portfolio_sizer.py                      # §9.2 + §14.7-CU
scripts/evaluation/
  └── multi_cycle_*_validation.py(× 10)      # §14.7-CY
```

---

## 十、Cross-machine pull-and-verify 快速 protocol(新機器執行)

```bash
# Step 1: Pull
cd ~/project
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git fetch --tags
git checkout v6.23.7-source-pure-final-convergence-9tree-complete-20260529

# Step 2: OS deps + venv + imports
# (per §4.1, §4.4, §4.5 above)

# Step 3: 雙 source verification(per §一.10)
source .venv/bin/activate
grep "^SPEC_43 = \[" scripts/core/model_trainer_lightgbm.py
# Plus: count features in SPEC list

# Step 4: 讀 master summary v6 + this handoff
less reports/tree_family_10_model_v6_v020_multi_cycle_complete_master_summary_20260529.md
less reports/cross_machine_handoff_20260529_doctrine_sealed.md

# Step 5: 確認跨機接續完成,select next direction:
#   A. Transformer dedicated rebuild(P1)
#   B. T_DC-8 silent fallback flag 補(P1)
#   C. Hyperparameter tuning(P1)
#   D. Cross-pillar interaction features(P2)
#   E. Other
```

---

## 十一、Session 治權狀態(本接力點)

- **Charter**:v6.1.0(7,344+ 行,含 §14.7-DC v0.7 七輪入憲)
- **CLAUDE.md**:§一.13 v0.7 雙鎖入憲完成
- **SPEC**:SPEC_38(strict source-pure convergence point per T_DC-18)
- **Universe**:910 stocks(v0.17 source-pure doctrine snapshot active)
- **Models**:9-tree × 4-horizon multi-cycle complete + 2 pass T_CZ-6
- **Doctrine**:Tier 0-2 only / 0 AI 幻像 / 0 imputed / 0 hardcoded knowledge

---

**Report 完成時間**:2026-05-29 17:01
**Author**:Codex(AI)/ Cross-machine handoff per §一.8 報告誠實 + §14.7-DC v0.7 三層治權鎖
**Sealed checkpoint**:`v6.23.7-source-pure-final-convergence-9tree-complete-20260529`
**HEAD**:`e9aba76`
