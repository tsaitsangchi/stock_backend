# From-Zero to Model Production + Validation Build Guide(v0.16)

**Charter anchor**: §14.7-CZ(8-phase canonical sequence)+ §14.7-DA(v0.16 triple-gate)+ §14.7-DB(reconciliation)
**Target end-state**: v0.16 N=1,002 active / LGBM v0.2 model committed / Multi-cycle validation PASS(quarterly Eff t ≥ 3.26 / annual Eff t ≥ 3.96)
**Total time(first build)**: ~7-14 hr(dominated by Phase 2 raw API sync 6-12 hr)
**Re-build from dump**: ~30 min(Phase 2 → pg_restore)+ ~1 hr(Phase 4-8)= **~1.5-2 hr**
**Audience**: 新機從零重建 / DB full drop 後 recovery / 跨機 reproducibility 驗證

---

## 📋 Pipeline 全景圖

```
Phase 0  Environment Bootstrap         §0.0-I.9/I.10        ~10 min
   ↓
Phase 1  DB Schema Init                §3.2/§6.7            ~5 min
   ↓
Phase 2  Raw API Sync(§14.7-AM 4-step)§14.7-AM             ~6-12 hr ⚠️ longest
   ↓
Phase 3  FRED Macro Sync               §0.3/§BY/§BZ         ~10 min
   ↓
Phase 4a Production Gate v0.13         §CG                  ~10 min
Phase 4b Backtest Triple-Gate v0.16    §DA                  ~1 min  ⭐ NEW per §DB
   ↓
Phase 5  Feature Store(1 + 95 panels)§CA/§CL              ~80 min
   ↓
Phase 6  Feature audits                §CM/CN/CO/CQ/CR      ~5 min
   ↓
Phase 7  LGBM v0.2 Production Training §CS/§CW              ~3 min
   ↓
Phase 8  Multi-Cycle Validation        §CV/§CX/§CY          ~5 min
   ↓
🎯 Production reality(per T_DB-6 / v0.16):
   Quarterly: Eff t=+3.26 / Sharpe 2.22 / Win 81.2% / Net +17.35%/年
   Annual:    Eff t=+3.96 / Sharpe 5.09 / Win 91.7% / Net +31.09%/年
```

---

# Phase 0 — Environment Bootstrap

**Charter**: §0.0-I.9 / §0.0-I.10 / CLAUDE.md §二.7
**Time**: ~10 min

## 0.1 System requirements

- macOS 10.15+ / Linux(Ubuntu 22.04+)
- Python **3.12.x** mandatory(3.13+ 不支援部分依賴)
- PostgreSQL **17.x** mandatory
- ~50 GB free disk(raw data + feature_store)
- 16 GB RAM minimum / 32 GB recommended
- Network access to `api.finmindtrade.com` + `api.stlouisfed.org`

## 0.2 OS native deps(per §0.0-I.9)

```bash
# macOS
brew install libomp postgresql@17

# Linux(Ubuntu)
sudo apt-get update
sudo apt-get install -y libgomp1 libpq-dev postgresql-17
```

## 0.3 Clone repository

```bash
cd ~/project
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git log --oneline -3  # confirm HEAD = f4f78ae or newer
```

## 0.4 Python venv + pip

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 0.5 Import smoke test(必須 PASS)

```bash
python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅ all imports OK')"
```
**FAIL**:回 0.2 補 OS deps;確認 Python 3.12.x;`pip install` 重跑。

## 0.6 `.env` configuration(per §0.0-I.8)

```bash
# 必填(per §0.0-I.8 14 變數契約)
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=stock
DB_USER=stock
DB_PASSWORD=stock
FINMIND_TOKEN=<your sponsor-tier FinMind token>
FRED_API_KEY=<your FRED API key>
# 可選
GEMINI_API_KEY=<optional>
GITHUB_TOKEN=<optional>
```

**Token acquisition**:
- FinMind: https://finmindtrade.com/analysis/UserInfo(sponsor tier 6,000/hr 推薦)
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html

## 0.7 PostgreSQL DB 建立

```bash
brew services start postgresql@17   # macOS
# or: sudo systemctl start postgresql

psql -h 127.0.0.1 -p 5432 -U postgres -c "
  CREATE USER stock WITH PASSWORD 'stock' SUPERUSER;
  CREATE DATABASE stock OWNER stock;
"
```

## ✅ Phase 0 Audit Gate

```bash
# Verify all
pg_isready -h 127.0.0.1 -p 5432                  # accepting connections
python -c "import psycopg2; psycopg2.connect(host='127.0.0.1',user='stock',password='stock',dbname='stock'); print('DB OK')"
python --version | grep "3.12"                    # Python 3.12.x
cat .env | grep -E "FINMIND_TOKEN|FRED_API_KEY"   # tokens set
```
**All PASS → continue to Phase 1**

---

# Phase 1 — DB Schema Init

**Charter**: §3.2 / §6.7
**Time**: ~5 min

## 1.1 Raw API tables schema

```bash
source venv/bin/activate && set -a && source .env && set +a
python scripts/core/data_schema.py --init --force
```

Creates 28+ raw API tables(TaiwanStockPriceAdj / TaiwanStockPER / ... / fred_series / ...).

## 1.2 Core universe governance schema

```bash
python scripts/core/core_universe_schema.py --init
```

Creates 5 governance tables:
- `core_universe_policy`
- `core_universe_snapshot`
- `core_universe_membership`
- `core_universe_scores`
- `universe_revision_log`

## 1.3 Universe completeness schema(per §14.7-CB)

```bash
python scripts/core/universe_completeness_schema.py --init
```

## ✅ Phase 1 Audit Gate

```bash
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -U stock -d stock -t -c "
SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';"
# Expected: ≥ 33 tables
```
**PASS = ≥ 33 → continue to Phase 2**

---

# Phase 2 — Raw API Sync(§14.7-AM 4-step)⚠️ LONGEST

**Charter**: §14.7-AM / §7 / §7.4-A
**Time**: ~6-12 hr(視 FinMind quota / 全市場 ~3,000 stocks × 多 tables × ~30 年歷史)
**Quota note**: FinMind sponsor tier 6,000/hr。若 hit cap → 等 1 hour 自動 recover。

## 2.1 Step A:Seed Ingestion(small core)

```bash
python scripts/ingestion/sovereign_sync_engine.py --seed
# 預期 ~5-10 min;只 sync 少量 seed stocks 以 init governance
```

## 2.2 Step B:bootstrap_init(per §14.7-AM 雞與蛋缺陷補強)

```bash
python scripts/core/core_universe_builder.py --bootstrap-init --commit
# 預期 ~2 min;建 v0.1 universe
```

## 2.3 Step C:全市場全天數 sync(per §6.8.7 第 (4) 條)

```bash
nohup python scripts/ingestion/sovereign_sync_engine.py \
  --universe full --all > /tmp/sync.log 2>&1 &
# 預期 ~6-12 hr;sync 全市場所有 tables 全歷史

# 監控
tail -f /tmp/sync.log
# 等 "✅ Full sync complete" 訊息
```

## 2.4 Step D:bootstrap_final(per §14.7-AM)

```bash
python scripts/core/core_universe_builder.py --bootstrap-final --commit
```

## ✅ Phase 2 Audit Gate

```bash
python scripts/maintenance/audit_supply_chain.py --include-logs
# 預期 PERFECT verdict / 29+/0/0
```

DB row sanity check:
```sql
SELECT 'PriceAdj' tbl, COUNT(*) FROM "TaiwanStockPriceAdj"
UNION ALL SELECT 'PER', COUNT(*) FROM "TaiwanStockPER"
UNION ALL SELECT 'Inst', COUNT(*) FROM "TaiwanStockInstitutionalInvestorsBuySell";
-- Expected:
--   PriceAdj ≥ 10,000,000
--   PER ≥ 7,000,000
--   Inst ≥ 25,000,000
```

**PASS → continue to Phase 3**

---

# Phase 3 — FRED Macro Sync

**Charter**: §0.3 / §14.7-BY / §14.7-BZ
**Time**: ~10 min

## 3.1 FRED 完整 sync(24 series:14 base + 6 K-wave Phase C + 2 §CC + 2 Path E)

```bash
python scripts/fetchers/fetch_fred_data.py
# 24 series × 30+ years ≈ ~70K rows
```

## ✅ Phase 3 Audit Gate

```sql
SELECT COUNT(DISTINCT series_id) FROM fred_series;
-- Expected: ≥ 13 (K-wave 7 + Multi-cycle 5 + Microstructure 1)
-- Optimal: 24 (full set)
```

**PASS → continue to Phase 4**

---

# Phase 4a — Production Universe(§CG v0.13 native gate)

**Charter**: §14.7-CB / §CC / §CD / §CE / §CF / §CG
**Time**: ~10 min

## 4a.1 Build v0.13

```bash
python scripts/core/core_universe_builder.py --mode doctrine-native --commit
# Stage 1: K-wave 13 series binary gate(macro)
# Stage 2: §0.1 8 raw sources × thresholds(per-stock)
# Stage 3: §0.2 3 raw sources × thresholds(per-stock)
# Stage 4: doctrine-pass union → core_universe
# Stage 5: atomic supersede write
```

**Expected output**:`core_universe_policy_v0.13_doctrine_native_gate` / **N≈1,583**(full doctrine-pass)

---

# Phase 4b — Backtest Triple-Gate Universe v0.16 ⭐(per §14.7-DA + §DB)

**Charter**: §14.7-DA / §14.7-DB
**Time**: ~1 min
**Purpose**: 在 v0.13(1,583)之上加 history≥8y + reasonable bounds filter,產出 backtest-doctrine-compliant N=1,002

## 4b.1 SQL post-process(triple-gate intersection)

```bash
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -p 5432 -U stock -d stock << 'SQL'
BEGIN;

-- Step 1: insert v0.16 policy
INSERT INTO core_universe_policy (policy_version, policy_name, description, active, effective_from)
VALUES (
  'core_universe_policy_v0.16_backtest_doctrine_compliant',
  '§14.7-CD ∩ history≥8y ∩ §14.7-CJ Reasonable',
  'Backtest universe: A=raw complete ∩ B=history>=8y ∩ C=reasonable = N stocks',
  TRUE, CURRENT_DATE
) ON CONFLICT (policy_version) DO NOTHING;

-- Step 2: compute A∩B∩C + insert snapshot + members
WITH
  v13 AS (
    SELECT stock_id FROM core_universe_membership m
    JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
    WHERE s.policy_version='core_universe_policy_v0.13_doctrine_native_gate'
      AND s.status='committed'
      AND m.core_tier='core_universe'
  ),
  hist_8y AS (
    SELECT v.stock_id FROM v13 v
    JOIN "TaiwanStockPriceAdj" p ON p.stock_id=v.stock_id
    GROUP BY v.stock_id HAVING MIN(p.date) <= '2018-01-01'
  ),
  -- (Note: feature_5 + reasonable need Phase 5 today panel; 若先做 4b 則 reasonable filter 跳過,僅用 A∩B = 1,418)
  reasonable_full AS (
    SELECT v.stock_id FROM v13 v
    WHERE EXISTS (
      SELECT 1 FROM feature_values fv
      WHERE fv.stock_id=v.stock_id
        AND fv.feature_name IN ('pe_ratio','pb_ratio','roe_ttm','operating_margin_ttm','dividend_yield')
      GROUP BY fv.stock_id
      HAVING
        MAX(CASE WHEN fv.feature_name='pe_ratio' THEN fv.feature_value::numeric END) BETWEEN 0.001 AND 500.0
        AND MAX(CASE WHEN fv.feature_name='pb_ratio' THEN fv.feature_value::numeric END) BETWEEN 0.001 AND 30.0
        AND MAX(CASE WHEN fv.feature_name='roe_ttm' THEN fv.feature_value::numeric END) BETWEEN -1.0 AND 1.0
        AND MAX(CASE WHEN fv.feature_name='operating_margin_ttm' THEN fv.feature_value::numeric END) BETWEEN -1.0 AND 1.0
        AND MAX(CASE WHEN fv.feature_name='dividend_yield' THEN fv.feature_value::numeric END) BETWEEN 0.0 AND 30.0
    )
  ),
  doctrine_compliant AS (
    SELECT v.stock_id FROM v13 v
    INNER JOIN hist_8y USING(stock_id)
    INNER JOIN reasonable_full USING(stock_id)
  )
INSERT INTO core_universe_snapshot
  (snapshot_id, as_of_date, source_data_cutoff, policy_version,
   total_candidates, core_count, status, notes, created_at)
SELECT
  'core_universe_' || to_char(CURRENT_DATE,'YYYYMMDD') || '_core_universe_policy_v0_16_backtest_doctrine_compliant',
  CURRENT_DATE, CURRENT_DATE,
  'core_universe_policy_v0.16_backtest_doctrine_compliant',
  (SELECT COUNT(*) FROM v13), (SELECT COUNT(*) FROM doctrine_compliant),
  'committed',
  '§14.7-DA triple-gate; supersedes v0.13', NOW();

-- Step 3: insert memberships(repeat CTE)
WITH
  v13 AS (
    SELECT stock_id FROM core_universe_membership m
    JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
    WHERE s.policy_version='core_universe_policy_v0.13_doctrine_native_gate'
      AND s.status='committed' AND m.core_tier='core_universe'
  ),
  hist_8y AS (
    SELECT v.stock_id FROM v13 v JOIN "TaiwanStockPriceAdj" p ON p.stock_id=v.stock_id
    GROUP BY v.stock_id HAVING MIN(p.date) <= '2018-01-01'
  ),
  reasonable_full AS (
    SELECT v.stock_id FROM v13 v
    WHERE EXISTS (
      SELECT 1 FROM feature_values fv
      WHERE fv.stock_id=v.stock_id
        AND fv.feature_name IN ('pe_ratio','pb_ratio','roe_ttm','operating_margin_ttm','dividend_yield')
      GROUP BY fv.stock_id HAVING
        MAX(CASE WHEN fv.feature_name='pe_ratio' THEN fv.feature_value::numeric END) BETWEEN 0.001 AND 500.0
        AND MAX(CASE WHEN fv.feature_name='pb_ratio' THEN fv.feature_value::numeric END) BETWEEN 0.001 AND 30.0
        AND MAX(CASE WHEN fv.feature_name='roe_ttm' THEN fv.feature_value::numeric END) BETWEEN -1.0 AND 1.0
        AND MAX(CASE WHEN fv.feature_name='operating_margin_ttm' THEN fv.feature_value::numeric END) BETWEEN -1.0 AND 1.0
        AND MAX(CASE WHEN fv.feature_name='dividend_yield' THEN fv.feature_value::numeric END) BETWEEN 0.0 AND 30.0
    )
  )
INSERT INTO core_universe_membership
  (snapshot_id, stock_id, core_tier, active, selected_at, selection_reason)
SELECT
  (SELECT snapshot_id FROM core_universe_snapshot WHERE policy_version='core_universe_policy_v0.16_backtest_doctrine_compliant' AND status='committed' ORDER BY as_of_date DESC LIMIT 1),
  v.stock_id, 'core_universe', TRUE, NOW(), '§14.7-DA triple-gate'
FROM v13 v INNER JOIN hist_8y USING(stock_id) INNER JOIN reasonable_full USING(stock_id);

-- Step 4: supersede v0.13
UPDATE core_universe_snapshot SET status='superseded'
WHERE policy_version='core_universe_policy_v0.13_doctrine_native_gate' AND status='committed';

COMMIT;

SELECT snapshot_id, status, core_count FROM core_universe_snapshot WHERE status='committed';
SQL
```

**重要 note**:Phase 4b 中 reasonableness filter 須有今日 feature panel,所以實務上順序為:
1. **Phase 4a build v0.13(1,583)**
2. **Phase 5 first iteration:build today's feature panel on v0.13 universe**
3. **Phase 4b:apply triple-gate(用 step 2 之 feature panel filter reasonable)→ v0.16 N=1,002 committed,v0.13 superseded**
4. **Phase 5 second iteration:rebuild today's feature panel on v0.16 universe**(現在 1,002 stocks)
5. **Phase 5 build 95 historical panels on v0.16**

簡化版(Skip reasonable check):
- 用 A∩B 兩 gate(raw + history)→ N≈1,418
- 不過此非完整 §14.7-DA(缺 C reasonable)

## ✅ Phase 4b Audit Gate

```sql
SELECT snapshot_id, status, core_count FROM core_universe_snapshot WHERE status='committed';
-- Expected: 1 row / status='committed' / core_count = 1,002(±5)
```

**PASS = active committed v0.16 N≈1,002 → continue to Phase 5**

---

# Phase 5 — Feature Store(today + 95 historical panels)

**Charter**: §14.7-CA / §CL
**Time**: ~80 min(主要 95 panel build,每 panel ~50s)

## 5.1 Today's panel(v0.16 bound)

```bash
python scripts/core/feature_store_builder.py --commit
# 預期 Feature Set ID: fs_<TODAY>_feature_set_v0_4 / 1,002 × 43 = 43,081 rows
```

## 5.2 Historical 95 panels(2018-06-15 → 2026-04-15 monthly)

```bash
nohup python scripts/evaluation/build_historical_panels.py > /tmp/build_panels.log 2>&1 &
# 預期 ~80 min:每 panel ~50s × 95
tail -f /tmp/build_panels.log
```

## ✅ Phase 5 Audit Gate

```sql
SELECT COUNT(*) FROM feature_store_snapshot
WHERE feature_set_id LIKE 'fs_%_feature_set_v0_4' AND total_stocks = 1002;
-- Expected: ≥ 96 (95 historical + today's)
```

**PASS → continue to Phase 6**

---

# Phase 6 — Feature Audits(IC + sign + necessity)

**Charter**: §14.7-CM / §CN / §CO / §CP / §CQ / §CR
**Time**: ~5 min

## 6.1 §14.7-CM Empirical IC vs Future Returns

```bash
python scripts/audit/audit_feature_ic_vs_future_return.py --as-of <historical-date>
# 例如 --as-of 2026-04-15(需有 ≥ 14 trading days 之 forward returns)
```

**Expected**:Mean \|IC\| > 0.03(treaty baseline)/ ≥ 30% features p<.05;**v0.16 reference: 0.2195 / 87.2%**

## 6.2 §14.7-CO Sign Stability

```bash
python scripts/audit/audit_feature_sign_stability.py
# 預期需 3 panels: fs_20260316 / fs_20260430 / fs_20260506
```

**Expected**:sign-stable ≥ 25% / lit-mismatch ≤ 5 / 0 indeterminate(§CQ);**v0.16 reference: 23.3% / 10 mismatch / 0?**

## 6.3 §14.7-CN Feature Necessity(可選)

```bash
python scripts/audit/audit_feature_necessity.py
```

## ✅ Phase 6 Audit Gate

- §14.7-CM Mean \|IC\| > 0.03 ✅
- §14.7-CQ 0 indeterminate(40+/3-) ✅
- §14.7-CR LITERATURE_SIGN 0 ± ✅

**PASS → continue to Phase 7**

---

# Phase 7 — LGBM v0.2 Production Training

**Charter**: §14.7-CS / §CW
**Time**: ~3 min

## 7.1 Training command

```bash
# 用最近 8 個 mid-month panels 之 walk-forward(2025-09-15 → 2026-04-15)
PANELS="fs_20250915_feature_set_v0_4,fs_20251015_feature_set_v0_4,fs_20251115_feature_set_v0_4,fs_20251215_feature_set_v0_4,fs_20260115_feature_set_v0_4,fs_20260215_feature_set_v0_4,fs_20260315_feature_set_v0_4,fs_20260415_feature_set_v0_4"

python scripts/core/model_trainer_lgbm_v2.py --panel-feature-sets "$PANELS"
```

## 7.2 Verify Treaty Gates

| Gate | Threshold | v0.16 reference |
|---|---|---:|
| CW-1 Sharpe > 0 | > 0 | +1.82 |
| CW-2 Win ≥ 50% | ≥ 50% | 75% |
| CW-3 MDD ≤ 30% | ≤ 30% | 6.6% |
| CW-4 Mean α > 0 | > 0 | +3.43%/30d |

**Expected verdict**:PERFECT(4/4 PASS)

## 7.3 Optionally persist to model_registry

```sql
-- See reports/v0_16_local_evidence_20260528.md §5 for exact INSERT pattern
INSERT INTO model_registry (model_id, model_policy_version, model_family, ...)
VALUES ('lgbm_v0_2_<date>_<host>', ...);
INSERT INTO model_training_run (model_id, ...) VALUES (...);
```

## ✅ Phase 7 Audit Gate

- Treaty Gates 4/4 PASS
- (optional)model_registry committed

**PASS → continue to Phase 8**

---

# Phase 8 — Multi-Cycle Validation

**Charter**: §14.7-CV / §CX / §CY
**Time**: ~5 min(380 LGBM walk-forward trains)

## 8.1 Multi-Cycle 4-horizon walk-forward

```bash
python scripts/evaluation/multi_cycle_validation.py
# 95 panels × 4 horizons(5/20/60/252d)= 380 LGBM trains
# 預期 ~5 min
```

## 8.2 NEW T_DB-6 v0.16 reality check(per §14.7-DB)

| Horizon | v0.16 expected | Sig? |
|---|---:|:---:|
| weekly(5d)| Eff t ≥ +1.87 | ❌ |
| **monthly(20d)** | **Eff t ≥ +2.09** | **✅** |
| **quarterly(60d) ⭐** | **Eff t ≥ +3.26 / Sharpe ≥ 2.22 / Win ≥ 81.2% / Net ≥ +17.35%/年** | **✅** |
| **annual(252d)** | **Eff t ≥ +3.96 / Sharpe ≥ 5.09 / Win ≥ 91.7% / Net ≥ +31.09%/年** | **✅** |

**Acceptance**:**3/4 horizons 統計顯著(p<.05)** + Quarterly Eff t > 1.997 + n_eff ≥ 30 + cost drag < 5%/yr

## ✅ Phase 8 Audit Gate(Terminal Reality Check per §14.7-DB T_DB-3)

**ALL OF**:
- 3/4 horizons sig(monthly / quarterly / annual)
- Quarterly:Eff t ≥ 3.26 / Sharpe ≥ 2.22 / Win ≥ 81%
- Annual:Eff t ≥ 3.96 / Sharpe ≥ 5.09 / Win ≥ 91%

**PASS → 🎯 System fully rebuilt and reality-validated**

---

# 🎯 Final State After Full Build

```
HEAD:                  current(git pull)
Active universe:       v0.16 / N=1,002 / status=committed
Feature panels:        ≥ 96 in DB
Model committed:       lgbm_v0_2_v0_16_local_<date>
Multi-cycle PASS:      3/4 horizons sig
Charter:               §CE~§DB 28 sections inscribed
Production strategy:   Quarterly rebalance / Net +17.35%/年 / Sharpe 2.22
```

---

# 📁 程式說明(per script)

## scripts/core/data_schema.py
- 用途:Init 全部 raw API tables(28+ tables)
- CLI:`--init --force`
- Idempotent:`CREATE TABLE IF NOT EXISTS`

## scripts/core/core_universe_schema.py
- 用途:Init 5 個 governance tables
- CLI:`--init`

## scripts/core/universe_completeness_schema.py
- 用途:Init `universe_completeness_snapshot`(per §14.7-CB Phase E hook)
- CLI:`--init`

## scripts/ingestion/sovereign_sync_engine.py
- 用途:全市場 raw API sync(§14.7-AM 4-step)
- CLI:`--seed` / `--universe full --all` / `--id <stock> --dataset <table>`
- Idempotent:ON CONFLICT DO UPDATE

## scripts/fetchers/fetch_fred_data.py
- 用途:FRED 24 series sync
- CLI:`--ids <series>` / `--force` / `--gap-fill 30`
- API endpoint:`api.stlouisfed.org`

## scripts/core/core_universe_builder.py
- 用途:Core universe selection
- 主要 mode:
  - `--mode doctrine-native`(§14.7-CG v0.13;native 11 source thresholds)
  - `--bootstrap-init / --bootstrap-final`(§14.7-AM 雞蛋缺陷補強)
- CLI:`--commit / --dry-run`

## scripts/core/feature_store_builder.py
- 用途:Feature store build(43 canonical features)
- CLI:`--commit --as-of-date <YYYY-MM-DD>`
- Reads latest committed universe snapshot

## scripts/evaluation/build_historical_panels.py
- 用途:Build 95 monthly historical feature panels(2018-06-15 → 2026-04-15)
- CLI:無參數;自動 iterate dates
- 預期 ~80 min on v0.16(1,002 stocks)

## scripts/core/model_trainer_lgbm_v2.py
- 用途:§14.7-CW LGBM tree v0.2 production training
- CLI:`--panel-feature-sets <fs_id_csv>`(8-panel walk-forward)
- Treaty Gates:CW-1 Sharpe>0 / CW-2 Win≥50% / CW-3 MDD≤30% / CW-4 α>0

## scripts/evaluation/multi_cycle_validation.py
- 用途:§14.7-CY 4-horizon multi-cycle walk-forward(380 trains)
- CLI:無參數 default(use all 95 panels × 4 horizons)
- Output:per-horizon Sharpe / Win / Eff t / Net annualized

## scripts/audit/audit_feature_ic_vs_future_return.py
- §14.7-CM Empirical IC
- CLI:`--as-of <date> --horizon 14`

## scripts/audit/audit_feature_sign_stability.py
- §14.7-CO / CQ / CR Sign verdict
- CLI:無參數(用 3 hardcoded historical panels)

## scripts/audit/audit_feature_necessity.py
- §14.7-CN 4-path necessity verdict

## scripts/maintenance/audit_supply_chain.py
- §7 全 raw table coverage audit
- CLI:`--include-logs`

## scripts/maintenance/audit_core_universe.py
- §6.7 core_universe_snapshot integrity audit
- CLI:`--as-of-date <date> --policy-version <name>`

---

# 🔄 Re-build from dump(若有 backup,推薦)

```bash
# 在 source machine 做 dump
pg_dump -h 127.0.0.1 -p 5432 -U stock -d stock -F c -f stock_backup_$(date +%Y%m%d).dump

# 在 target machine 還原
brew services start postgresql@17  # macOS
createdb -h 127.0.0.1 -p 5432 -U stock stock
pg_restore -h 127.0.0.1 -p 5432 -U stock -d stock stock_backup_<date>.dump

# 然後 skip Phase 2,直接從 Phase 4a 開始(或直接驗證 active state)
```

Restore time:~30 min(vs 6-12 hr API sync)+ Phase 4-8 ~1.5 hr = **~2 hr total**

---

# 📊 Pre-Phase 7 sanity checklist(每 Phase 結束後 check)

```bash
# After Phase 2
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -U stock -d stock -c "
SELECT 'PriceAdj' tbl, COUNT(*) FROM \"TaiwanStockPriceAdj\"
UNION ALL SELECT 'PER', COUNT(*) FROM \"TaiwanStockPER\";
"
# Expected: PriceAdj ≥ 10M / PER ≥ 7M

# After Phase 3
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -U stock -d stock -c "
SELECT COUNT(DISTINCT series_id) FROM fred_series;
"
# Expected: ≥ 13(optimal 24)

# After Phase 4a + 4b
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -U stock -d stock -c "
SELECT snapshot_id, core_count FROM core_universe_snapshot WHERE status='committed';
"
# Expected: 1 row / N=1,002 / policy=v0.16

# After Phase 5
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -U stock -d stock -c "
SELECT COUNT(*) FROM feature_store_snapshot WHERE total_stocks=1002;
"
# Expected: ≥ 96

# After Phase 6
# audit_feature_ic_vs_future_return Mean |IC| > 0.03
# audit_feature_sign_stability §CQ PASS(0 indeterminate)

# After Phase 7
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -U stock -d stock -c "
SELECT model_id, status FROM model_registry;
"
# Expected: ≥ 1 committed model

# After Phase 8
# multi_cycle output: 3/4 horizons sig + Quarterly Eff t ≥ 3.26
```

---

# 🎯 Production deployment(post-rebuild)

After Phase 8 PASS,系統可進入 production:

1. **Recommended strategy**:Quarterly rebalance / Net +17.35%/年 / Sharpe 2.22 / Win 81.2%
2. **Alternative high-α**:Annual rebalance / Net +31.09%/年 / Sharpe 5.09 / Win 91.7%(但 n_eff 較小)
3. **Weekly cron**:`scripts/maintenance/run_weekly_doctrine_recommit.py`(per §14.7-BX)— 每週五 13:30+ 自動重跑 Phase 4a-8
4. **Monitoring**:
   - `audit_feature_ic_vs_future_return.py` weekly(per §14.7-CM)
   - `multi_cycle_validation.py` monthly(per §14.7-CY)— drift detection
   - Model retrain quarterly(per panel cadence)

---

# 📜 Charter Cross-Reference

| Phase | Charter sections |
|---|---|
| 0 | §0.0-I.8(env vars)/ §0.0-I.9(OS deps)/ §0.0-I.10(paths) |
| 1 | §3.2(schema)/ §6.7(SQL contract) |
| 2 | §14.7-AM(4-step zero-to-full)/ §7(supply chain)/ §7.4-A(402 cascade) |
| 3 | §0.3 / §14.7-BY(K-wave purity)/ §14.7-BZ(macro pillar split) |
| 4a | §14.7-CB(feature completeness gate)/ §CC(source authority)/ §CD(raw threshold)/ §CE(empirical proof)/ §CF(unified SSOT)/ §CG(native gate v0.13) |
| 4b | **§14.7-DA**(triple-gate v0.16)/ **§14.7-DB**(from-zero v0.16 reconciliation) |
| 5 | §14.7-CA(43 canonical features)/ §CL(canonical scope) |
| 6 | §14.7-CM(empirical IC)/ §CN(necessity)/ §CO(sign stability)/ §CQ(sign commitment)/ §CR(math realism) |
| 7 | §14.7-CS(model training landing)/ §CW(LGBM tree production) |
| 8 | §14.7-CV(walk-forward)/ §CX(8-year OOS)/ §CY(multi-cycle)/ §CZ(from-zero sequence) |

---

**🏛️ Per §14.7-DB:本 guide 為 from-zero rebuild 之 implementation SSOT;v0.16 N=1,002 為 expected active state;reality reference per §14.7-DA evidence(`reports/v0_16_local_evidence_20260528.md` + `reports/multi_cycle_validation_20260528_v0_16_local.json`)。**
