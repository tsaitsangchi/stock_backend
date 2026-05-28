# Quantum Finance 從零到模型 + 驗證完整建構指南

**Doctrine Anchor**:`§14.7-CZ From-Zero Production Build Sequence Doctrine`(charter L11833 / v6.20.0)
**對映用戶 directive**(2026-05-28):「如果要從零開始到模型產生及之後的模型驗証要來建構此系統,請寫出一份如何從零開始到模型產生架構此系統的報告」
**用法**:全環境 + DB 刪除後依本報告 8 phase canonical sequence 重建,**總計 ~7-14 hr first-time build**

---

## 📋 Pipeline 全景圖

```
Phase 0  Environment Bootstrap    [OS deps + Python venv + .env]            ~10 min
   ↓
Phase 1  DB Schema Init           [3 schema init scripts]                   ~5 min
   ↓
Phase 2  Raw API Sync(§14.7-AM)  [4-step zero-to-full-market+FRED]        6-12 hr  ⭐ 最耗時
   ↓
Phase 3  FRED Macro Sync          [11 P0 + 2 P1 series]                   ~5 min
   ↓
Phase 4  Core Universe Selection  [§14.7-CG native gate / 7 gates]        ~10 min
   ↓
Phase 5  Feature Store            [current + 95 historical monthly]       ~20 min
   ↓
Phase 6  Feature Audits           [IC / sign / necessity]                 ~5 min
   ↓
Phase 7  LGBM Model Training      [scripts/core/model_trainer_lgbm_v2.py] ~3 min
   ↓
Phase 8  Multi-Layer Validation   [walk-forward + multi-cycle]            ~5 min
   ↓
✅ PRODUCTION-READY                Quarterly horizon @ Eff t≥4.20 / Sharpe≥2.4
```

**Time budget realistic**(per §14.7-CZ T_CZ-3):**total ~7-14 hr first-time build**

---

# 📍 Phase 0 — Environment Bootstrap(~10 min)

## 0.1 System Requirements

| 項目 | 需求 | 來源治權 |
|---|---|---|
| OS | Linux(Ubuntu 22.04+ ideal)/ macOS(arm64 or x86_64)/ Windows(限制較多)| §0.0-I.10 |
| Python | **3.11+** | §0.0-I.9 |
| PostgreSQL | **17.x**(client headers)| 本 doctrine |
| OpenMP runtime | **libgomp1**(Linux)/ **libomp**(macOS)| §0.0-I.9 |
| Disk space | ~15 GB(13 GB DB + 2 GB code/artifacts)| 本 doctrine |
| RAM | ≥ 8 GB(LGBM peak ~4 GB)| 本 doctrine |

## 0.2 OS 原生依賴安裝(per §0.0-I.9)

```bash
# Linux(Ubuntu / Debian)
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    libgomp1 libpq-dev \
    postgresql-17 postgresql-client-17 \
    git curl

# macOS
brew install python@3.11 libomp postgresql@17 git
brew services start postgresql@17
```

## 0.3 Clone Repository

```bash
cd ~/project  # or your projects dir
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend

# Checkout sealed checkpoint(latest production-ready state)
git checkout session-final-20260528-v6.19.0-multi-cycle-sealed
# Or latest master:
# git pull origin master
```

## 0.4 Python venv + pip 套件

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
# 或手動:
# pip install psycopg2-binary pandas polars numpy requests \
#             scikit-learn xgboost lightgbm scipy joblib \
#             threadpoolctl tqdm python-dotenv pyarrow
```

## 0.5 Import Smoke Test(per §0.0-I.9 / CLAUDE.md §二.7)

```bash
python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅ all imports OK')"
```

**Expected**:`✅ all imports OK`

## 0.6 .env Configuration(per §0.0-I.8)

```bash
cat > .env <<'EOF'
PROJECT_ROOT=/home/<your-user>/project/stock_backend
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock
DB_USER=<your-db-user>
DB_PASSWORD=<your-db-password>
FINMIND_TOKEN=<your-finmind-token>
FRED_API_KEY=<your-fred-api-key>
TZ=Asia/Taipei
EOF
```

**注意**(per §0.0-I.10):
- macOS:PROJECT_ROOT 用 `/Users/<user>/...`,**不是** `/home/<user>/...`
- Linux:用物理絕對路徑,避免 symlink
- `path_setup.py v4.47+` 用 `os.path.realpath()` 處理 symlink

## 0.7 PostgreSQL DB 建立

```bash
sudo -u postgres psql <<SQL
CREATE USER <your-db-user> WITH PASSWORD '<your-db-password>';
CREATE DATABASE stock OWNER <your-db-user>;
GRANT ALL PRIVILEGES ON DATABASE stock TO <your-db-user>;
SQL
```

## ✅ Phase 0 Audit Gate

```bash
source .venv/bin/activate
python -c "
import psycopg2, lightgbm
from dotenv import load_dotenv; load_dotenv()
import os
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
    dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'))
print(f'✅ DB connection OK')
print(f'✅ lightgbm {lightgbm.__version__}')
"
```

**Phase 0 PASS** = DB connection OK + imports OK

---

# 📍 Phase 1 — DB Schema Init(~5 min)

## 1.1 Raw API Tables Schema(`data_schema.py`)

```bash
python scripts/core/data_schema.py --init --force
```

**Creates**:
- TaiwanStockPrice / TaiwanStockPriceAdj
- TaiwanStockInfo
- TaiwanStockMonthRevenue / FinancialStatements / BalanceSheet
- TaiwanStockInstitutionalInvestorsBuySell
- TaiwanStockMarginPurchaseShortSale / Shareholding / PER / Dividend
- fred_series
- pipeline_execution_log / data_audit_log

## 1.2 Core Universe Governance Schema(`core_universe_schema.py`)

```bash
python scripts/core/core_universe_schema.py --init
```

**Creates**(per §6.7):
- core_universe_snapshot(snapshot governance)
- core_universe_membership(stock × tier 對映)
- core_universe_scores(CoreScore v0.2 六層分數)
- model_registry / prediction_run
- evaluation_log / feature_store_snapshot / feature_values / feature_definition

## 1.3 Universe Completeness Schema(per §14.7-CB)

```bash
python scripts/core/universe_completeness_schema.py --init
```

**Creates**:feature_completeness_audit table(用於 §14.7-CB Feature Completeness Gate)

## 1.4 db_utils 連線驗證

```bash
python scripts/core/db_utils.py
```

**Expected**:列出 §6.7 SSOT 對映 / connection PERFECT

## ✅ Phase 1 Audit Gate

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from core.db_utils import get_db_conn
conn = get_db_conn()
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=\\'public\\'')
n = cur.fetchone()[0]
print(f'Tables: {n}')
assert n >= 33, f'Expected ≥ 33 tables, got {n}'
print('✅ Phase 1 PASS')
"
```

**Phase 1 PASS** = ≥ 33 tables created

---

# 📍 Phase 2 — Raw API Sync 4-Step(~6-12 hr ⭐ 最耗時)

依 **§14.7-AM 雞與蛋缺陷補強之 4 步序列**(2026-05-21 入憲)。

## 2.1 Step A:Seed Ingestion(small core)

```bash
python scripts/ingestion/sovereign_sync_engine.py --seed
```

**作用**:
- 取得 TaiwanStockInfo(2,799 stocks metadata)
- 為 core 候選股 fetch 短期 history(~ recent 250 days)
- 建立 minimal_seed candidate_fallback

**耗時**:~30-60 min

**Audit**:
```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from core.db_utils import get_db_conn
conn = get_db_conn()
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM \"TaiwanStockInfo\"')
print(f'TaiwanStockInfo: {cur.fetchone()[0]}')
cur.execute('SELECT COUNT(DISTINCT stock_id) FROM \"TaiwanStockPriceAdj\"')
print(f'PriceAdj stocks: {cur.fetchone()[0]}')
"
```

## 2.2 Step B:bootstrap_init(per §14.7-AM 雞與蛋缺陷補強)

`sovereign_sync_engine.py --universe full` 需要 `core_universe_membership` committed,但 universe builder 又需要 raw data。**解法**:先用 latest_registry_fallback bootstrap commit 一個 minimal snapshot。

```bash
python scripts/core/core_universe_builder.py --commit \
    --as-of-date $(date +%Y-%m-%d) \
    --special-rebalance-reason "DB rebuild bootstrap_init"
```

**作用**:寫入第一份 `core_universe_membership` snapshot,作為 `sovereign_sync_engine.py --universe full` 之 precondition。

## 2.3 Step C:全市場全天數 sync(per §6.8.7 第 (4) 條)

```bash
python scripts/ingestion/sovereign_sync_engine.py \
    --universe full --all \
    --special-full-market-reason "DB rebuild full-market full-history sync"
```

**作用**:全 1,800-2,800 stocks × ~34 years history 完整 sync(從 1992-01 起所有可得 raw rows)

**耗時**:~5-10 hr(per §7.4-A 402 cascade mitigation + parallel workers)
**Rows added**:~76M raw rows

**監控**(per §6.8.7-B 30 分鐘監控規範):
```bash
# 在另一 terminal:
watch -n 1800 'python -c "
import sys; sys.path.insert(0, \"scripts\")
from core.db_utils import get_db_conn
conn = get_db_conn(); cur = conn.cursor()
cur.execute(\"SELECT COUNT(*) FROM \\\"TaiwanStockPriceAdj\\\"\")
print(f\"PriceAdj rows: {cur.fetchone()[0]:,}\")
"'
```

## 2.4 Step D:bootstrap_final(per §14.7-AM)

```bash
python scripts/core/core_universe_builder.py --commit \
    --as-of-date $(date +%Y-%m-%d) \
    --special-rebalance-reason "DB rebuild bootstrap_final after full-market sync"
```

**作用**:用 full-market data 重 build 完整 core_universe_membership(取代 bootstrap_init 之 minimal snapshot)

## ✅ Phase 2 Audit Gate

```bash
python scripts/maintenance/audit_supply_chain.py --include-logs
```

**Expected**:`PERFECT (PASS=33 / WARN=0 / FAIL=0)`

**Phase 2 PASS** = audit_supply_chain PERFECT

---

# 📍 Phase 3 — FRED Macro Sync(~5 min)

依 §0.3 康波週期 13 indicator(per §14.7-BY:11 P0 + Path E P1 2 個 BIS Credit / EIA Oil)。

## 3.1 FRED 完整 sync

```bash
python scripts/fetchers/fetch_fred_data.py
# 或 via sovereign_sync_engine:
# python scripts/ingestion/sovereign_sync_engine.py --source fred
```

**作用**:fetch FRED 24 series(11 §0.3 P0 + Path E + 其他輔助)

## ✅ Phase 3 Audit Gate

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from core.db_utils import get_db_conn
conn = get_db_conn(); cur = conn.cursor()
cur.execute('SELECT COUNT(DISTINCT series_id) FROM fred_series')
n = cur.fetchone()[0]
print(f'FRED series: {n}')
assert n >= 13, f'Expected ≥ 13, got {n}'
print('✅ Phase 3 PASS')
"
```

**Phase 3 PASS** = ≥ 13 distinct FRED series

---

# 📍 Phase 4 — Core Universe Selection(~10 min)

依 §14.7-CF SSOT + §14.7-CG native gate(v0.13)+ §14.7-CI/CJ/CK 三重 quality gate(v0.14-v0.16)。

## 4.1 Native Gate Build(per §14.7-CG)

```bash
python scripts/core/core_universe_builder.py \
    --mode doctrine-native --commit \
    --as-of-date $(date +%Y-%m-%d)
```

**Pipeline**(per §14.7-CG):
1. Stage 1:§0.3 K-wave 13 FRED series 存在性檢查
2. Stage 2:§0.1 第一性原理 8 raw sources × thresholds
3. Stage 3:§0.2 八二法則 3 raw sources × thresholds
4. Stage 4:doctrine-pass union → core_universe
5. Stage 5:atomic supersede write(per §14.7-BX)

## 4.2 Feature Completeness Gate(per §14.7-CB)

builder 內含 37/37 feature completeness 檢查;**個股 feature 不完整即排除**(per 用戶 7 chain directive 之第 4 條)。

## 4.3 Strict Feature Validity Gate(per §14.7-CI v0.14)

feature 計算後仍 invalid → 排除。

## 4.4 Feature Reasonableness Gate(per §14.7-CJ v0.15)

feature 為 outlier(z-score > N)→ 排除。

## 4.5 Feature Effectiveness Doctrine(per §14.7-CK v0.16)

broadcast features(macro 14 個無 cross-section variation)→ 從 43-feature SPEC 移除。

## ✅ Phase 4 Audit Gate

```bash
python scripts/maintenance/audit_core_universe.py \
    --as-of-date $(date +%Y-%m-%d)
```

**Expected**:`PERFECT (PASS=36 / WARN=0 / FAIL=0)`

**Phase 4 PASS** = audit_core_universe PERFECT + N ≈ 1,121(super-strict 預期值)

---

# 📍 Phase 5 — Feature Store(~20 min)

依 §14.7-CA(v0.3 doctrine-aligned)+ §14.7-CL(43 canonical features)。

## 5.1 Current Snapshot(per §14.7-CA)

```bash
python scripts/core/feature_store_builder.py --commit \
    --as-of-date $(date +%Y-%m-%d) \
    --feature-set-version feature_set_v0.4 \
    --label-horizon 30
```

**Output**:`fs_<YYYYMMDD>_feature_set_v0_4`(43 features × 1,121 stocks ≈ 48K rows)

## 5.2 Historical 95 Monthly Panels(per §14.7-CX)

```bash
python scripts/evaluation/build_historical_panels.py
```

**作用**:build 2018-06-15 ~ 2026-04-15 mid-month 95 monthly snapshots
**耗時**:~16 min(每 panel ~10s × 95)
**Output**:95 × 48K = ~4.5M new feature_values rows

## ✅ Phase 5 Audit Gate

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from core.db_utils import get_db_conn
conn = get_db_conn(); cur = conn.cursor()
cur.execute('''
    SELECT COUNT(*), MIN(as_of_date), MAX(as_of_date)
    FROM feature_store_snapshot
    WHERE status=\\'committed\\' AND feature_set_version=\\'feature_set_v0.4\\'
''')
n, mn, mx = cur.fetchone()
print(f'feature_store_snapshot: {n}({mn} → {mx})')
assert n >= 96, f'Expected ≥ 96 snapshots, got {n}'
cur.execute('SELECT COUNT(*) FROM feature_values')
rows = cur.fetchone()[0]
print(f'feature_values rows: {rows:,}')
assert rows > 4_000_000, f'Expected > 4M rows, got {rows:,}'
print('✅ Phase 5 PASS')
"
```

**Phase 5 PASS** = ≥ 96 snapshots committed + ≥ 4M feature_values rows

---

# 📍 Phase 6 — Feature Audits(~5 min)

依 §14.7-CM(empirical IC)+ §14.7-CO(sign stability)+ §14.7-CN(necessity)。

## 6.1 Empirical IC vs Future Returns(per §14.7-CM)

```bash
python scripts/audit/audit_feature_ic_vs_future_return.py
```

**作用**:每 feature 跨 95 panels × forward 30d returns 計算 Spearman IC

## 6.2 Feature Sign Stability(per §14.7-CO)

```bash
python scripts/audit/audit_feature_sign_stability.py
```

**作用**:每 feature 之 IC sign(+/-)across panels 之穩定度

## 6.3 Feature Necessity(per §14.7-CN)

```bash
python scripts/audit/audit_feature_necessity.py
```

**作用**:4-path necessity verdict(§0.1 第一性原理 / §0.2 八二法則 / 學術文獻 / 統計顯著)

## ✅ Phase 6 Audit Gate

**Phase 6 PASS** = 各 audit 跑出 t-stat / sign verdict reports

---

# 📍 Phase 7 — LGBM Model Training(~3 min)

依 §14.7-CW LGBM tree v0.2(2026-05-28 入憲)。

## 7.1 Production Model Training

```bash
python scripts/core/model_trainer_lgbm_v2.py --commit
```

**作用**:
- 跑 8-panel walk-forward expanding window OOS
- Train LGBM tree(n_estimators=200, max_depth=5, num_leaves=20, seed=5422)
- Save model artifact 至 `data/models/<model_id>/`
- Insert row 至 `model_registry`

## 7.2 Verify Model Artifact

```bash
ls -la data/models/mdl_$(date +%Y%m%d)_lgbm_h30_*/
```

**Expected**:`model.txt`(~340KB)+ `metrics.json` + `hyperparams.json`

## ✅ Phase 7 Audit Gate

從 stdout 確認 **Treaty Gates 4/4 PASS**:
- Gate CW-1 Sharpe > 0 ✅
- Gate CW-2 Win rate ≥ 50% ✅
- Gate CW-3 MDD ≤ 30% ✅
- Gate CW-4 Mean alpha > 0 ✅

**Phase 7 PASS** = 4/4 Treaty Gates PASS + model_registry row created

---

# 📍 Phase 8 — Multi-Layer Validation(~5 min)

依 §14.7-CV(backtest)+ §14.7-CX(8-year OOS)+ §14.7-CY(multi-cycle horizon)。

## 8.1 Walk-Forward Backtest(per §14.7-CV)

```bash
python scripts/evaluation/audit_backtest_walk_forward.py
```

**作用**:8-panel top-20 long strategy backtest;Sharpe / Win / α / MDD per panel

## 8.2 8-Year Historical OOS(per §14.7-CX)

由 model_trainer_lgbm_v2.py(--panel-feature-sets)或 multi_cycle_validation.py 內含:

```bash
PANELS=$(python -c "
from datetime import date
panels = []
current = date(2018, 6, 15)
while current <= date(2026, 4, 30):
    panels.append(f'fs_{current.strftime(\"%Y%m%d\")}_feature_set_v0_4')
    if current.month == 12: current = date(current.year+1, 1, 15)
    else: current = date(current.year, current.month+1, 15)
print(','.join(panels))")
python scripts/core/model_trainer_lgbm_v2.py --dry-run --panel-feature-sets "$PANELS"
```

**Expected**:Sharpe 1.67 / Win 67.7% / α +1.94% / 65 OOS panels

## 8.3 Multi-Cycle Horizon Validation(per §14.7-CY)

```bash
python scripts/evaluation/multi_cycle_validation.py --dry-run \
    --horizons 5,20,60,252 \
    --output reports/multi_cycle_validation_$(date +%Y%m%d).json
```

**Expected cross-cycle matrix**:

| Horizon | Eff t-stat | Sig p<0.05 | Sharpe | Win | Net Annual |
|---|---|---|---|---|---|
| weekly(5d) | +1.59 | ❌ | 0.89 | 67.7% | +13.99% |
| monthly(20d) | +1.41 | ❌ | 0.97 | 64.6% | +17.41% |
| **quarterly(60d)** | **+4.20** | **✅** | **2.55** | **79.7%** | **+24.44%** |
| annual(252d) | +3.58 | ✅(small n_eff)| 4.81 | 91.8% | +29.69% |

## ✅ Phase 8 Audit Gate(Terminal Reality Check per §14.7-CZ T_CZ-6)

```bash
# Reality check verdict
python -c "
import json
with open('reports/multi_cycle_validation_$(date +%Y%m%d).json') as f:
    data = json.load(f)
q = data['quarterly']
print(f'Quarterly:')
print(f'  Eff t-stat:    {q[\"effective_t_stat\"]:+.3f}')
print(f'  Sharpe:        {q[\"sharpe\"]:+.3f}')
print(f'  Win rate:      {q[\"win_rate\"]*100:.1f}%')
print(f'  Net annual:    {q[\"annualized_simple_net\"]*100:+.2f}%')
assert q['effective_t_stat'] >= 4.0, 'Eff t < 4.0'
assert q['sharpe'] >= 2.4, 'Sharpe < 2.4'
assert q['win_rate'] >= 0.79, 'Win < 79%'
print('✅ Phase 8 PASS — production-ready per §14.7-CZ T_CZ-6')
"
```

**Phase 8 PASS** = Quarterly Eff t ≥ 4.0 / Sharpe ≥ 2.4 / Win ≥ 79%

---

# 🎯 各 Script 詳細說明

## scripts/core/data_schema.py
- **角色**:Raw API tables 之 schema SSOT
- **CLI**:`--init --force`(必須 --force 才會 drop 重建)
- **Creates**:11 FinMind tables + fred_series + 治權 audit tables

## scripts/core/core_universe_schema.py
- **角色**:§6.7 SSOT governance schema(snapshot / membership / scores / registry / log)
- **CLI**:`--init`
- **Creates**:9 governance tables

## scripts/core/universe_completeness_schema.py
- **角色**:§14.7-CB feature_completeness_audit table
- **CLI**:`--init`

## scripts/ingestion/sovereign_sync_engine.py
- **角色**:§7 唯一 sync 引擎(§3.1 序列模組)
- **CLI 重要 mode**:
  - `--seed`(Group A:種子灌溉 / TaiwanStockInfo + minimal core)
  - `--universe full --all`(Group F:全市場全天數)
  - `--universe core --all --full-history`(Group E:核心股全天數補刷)
  - `--source fred`(FRED-only sync)
  - `--disable-402-cascade-mitigation`(per §7.4-A v1.22 flag)
- **耗時**:--seed ~30-60 min / --universe full ~5-10 hr

## scripts/fetchers/fetch_fred_data.py
- **角色**:FRED API fetcher(per §14.7-BY v0.3 / 11 P0 + Path E P1)
- **CLI**:無 mode(直接跑)

## scripts/core/core_universe_builder.py
- **角色**:§14.7-CF/CG 核心股 selection
- **CLI 重要 mode**:
  - `--mode doctrine-native`(per §14.7-CG v0.13)
  - `--mode coreScore`(legacy CoreScore v0.2)
  - `--dry-run / --commit`
  - `--special-rebalance-reason "<≥12 字>"`(per §6.8 annual_rebalance_guard 例外)
- **Output**:core_universe_snapshot + membership(N ≈ 1,121 expected per §14.7-CJ)

## scripts/core/feature_store_builder.py
- **角色**:§14.7-CA/CL feature computation(43 canonical features)
- **CLI**:
  - `--dry-run / --commit`
  - `--as-of-date YYYY-MM-DD`(anti-leakage per §8.5)
  - `--feature-set-version feature_set_v0.4`
  - `--label-horizon 30`
- **Anti-leakage**:per §8.5-9 publication_date_strategy

## scripts/evaluation/build_historical_panels.py(本 session 新建)
- **角色**:批量 build 95 historical monthly panels for §14.7-CX
- **耗時**:~16 min(每 panel ~10s × 95)

## scripts/core/model_trainer_lgbm_v2.py(本 session 新建)
- **角色**:§14.7-CW LGBM tree v0.2 production trainer
- **CLI**:
  - `--dry-run / --commit`
  - `--panel-feature-sets <fs1,fs2,...>`(walk-forward training panels)
  - `--label-horizon 30`
- **Output**:model.txt + metrics.json + model_registry row

## scripts/evaluation/multi_cycle_validation.py(本 session 新建)
- **角色**:§14.7-CY 4-horizon × 95-panel cross-cycle validation
- **CLI**:
  - `--dry-run / --commit`
  - `--horizons 5,20,60,252`
  - `--output <path>.json`
- **Output**:per-horizon metrics + cross-cycle comparison + JSON persistence

## scripts/maintenance/audit_supply_chain.py
- **角色**:§3.2A.H audit performance / supply chain integrity check
- **CLI**:`--include-logs`
- **Expected**:PASS=33 / WARN=0 / FAIL=0

## scripts/maintenance/audit_core_universe.py
- **角色**:§6.7 core_universe_snapshot 驗收稽核
- **CLI**:`--as-of-date YYYY-MM-DD`
- **Expected**:PASS=36 / WARN=0 / FAIL=0

## scripts/audit/audit_feature_ic_vs_future_return.py
- **角色**:§14.7-CM per-feature × future return IC empirical computation

## scripts/audit/audit_feature_sign_stability.py
- **角色**:§14.7-CO feature IC sign across panels stability check

## scripts/audit/audit_feature_necessity.py
- **角色**:§14.7-CN 4-path necessity verdict

## scripts/evaluation/audit_backtest_walk_forward.py
- **角色**:§14.7-CV 8-panel real walk-forward backtest

---

# ⏱️ 真實 Time Budget(per §14.7-CZ T_CZ-3)

| Phase | 耗時 | 累計 |
|---|---|---|
| Phase 0 Environment | ~10 min | 10 min |
| Phase 1 DB Schema | ~5 min | 15 min |
| **Phase 2 Raw API Sync** | **~6-12 hr** ⭐ | **~6-12 hr** |
| Phase 3 FRED Sync | ~5 min | +5 min |
| Phase 4 Universe Build | ~10 min | +10 min |
| Phase 5 Feature Store(current + 95)| ~20 min | +20 min |
| Phase 6 Feature Audits | ~5 min | +5 min |
| Phase 7 Model Training | ~3 min | +3 min |
| Phase 8 Validation Suite | ~5 min | +5 min |
| **TOTAL first-time build** | — | **~7-14 hr** |

**Re-build(已有 dump)**:Phase 2 改為 pg_restore ~30 min,total ~1-2 hr

---

# 🚨 常見問題 + 解法

| 問題 | 解法 |
|---|---|
| psycopg2 missing libpq | `sudo apt-get install libpq-dev` |
| lightgbm missing libomp | Linux: `apt-get install libgomp1` / macOS: `brew install libomp` |
| PROJECT_ROOT path mismatch | macOS `/Users` vs Linux `/home`;`.env` 寫物理絕對路徑 |
| 雞與蛋 precondition fail | 必須先跑 Phase 2 Step B bootstrap_init |
| FinMind quota cap(402)| `--disable-402-cascade-mitigation` flag(per §7.4-A v1.22)/ 等 hourly window 重置 |
| LGBM Sharpe variance ±15% | 跑 ≥ 3 runs(per §14.7-CW T_CW-6)|
| Charter line numbers drift | 升版後重 grep `^### §14.7-` |
| timezone mismatch | `.env` 設 `TZ=Asia/Taipei` |
| PG version不對 | DB dump v17 須用 client v17+ |

---

# ✅ 最終 production-ready 驗收 checklist

完成 Phase 0-8 後執行:

- [ ] `git describe --tags` 對齊 sealed checkpoint
- [ ] DB tables ≥ 33 / `audit_supply_chain.py` PERFECT
- [ ] FRED series ≥ 13 distinct
- [ ] `core_universe_membership` ≥ 1,100 stocks(super-strict expected ~1,121)
- [ ] `feature_store_snapshot` ≥ 96 committed
- [ ] `feature_values` ≥ 4,000,000 rows
- [ ] `model_registry` 含 `mdl_*_lgbm_h30_*_v0_2` row
- [ ] `data/models/mdl_*/model.txt` exists ~340KB
- [ ] LGBM Treaty Gates 4/4 PASS
- [ ] Multi-cycle quarterly Eff t ≥ 4.0 / Sharpe ≥ 2.4 / Win ≥ 79%(per §14.7-CZ T_CZ-6)

**Pass all 10 → 系統 production-ready per §14.7-CZ canonical sequence。**

---

# 📚 Reference Documents

| Doc | Purpose |
|---|---|
| `reports/系統架構大憲章_v6.1.0.md` | 治權 SSOT(67 §14.7-* sections)|
| `CLAUDE.md` | AI 協作工具規則(10 §一 rules / per §一.10 No Data Hallucination)|
| `reports/cross_machine_handoff_20260528_v6.19.0_multi_cycle_sealed.md` | Cross-machine handoff(502 行)|
| `reports/multi_cycle_validation_report_20260528.md` | Multi-cycle 完整 reality report |
| `reports/from_zero_to_model_build_guide_20260528.md`(本檔)| **§14.7-CZ implementation reference** |

---

**報告生成時間**:2026-05-28(per §14.7-CZ inscription session)
**Charter Anchor**:`§14.7-CZ From-Zero Production Build Sequence Doctrine`(charter L11833)
**Repository**:https://github.com/tsaitsangchi/stock_backend
**Latest sealed**:`session-final-20260528-v6.19.0-multi-cycle-sealed`
**Source compliance**:per CLAUDE.md §一.10 — 全 charter / system script-traceable
