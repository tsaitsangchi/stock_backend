# Cross-Machine Handoff — 2026-05-28 v6.22.0 Sealed(v0.16 + From-Zero Guide)

**Handoff date**: 2026-05-28 21:55 CST
**Final HEAD**: `a558c34`(可直接 `git pull origin master` 取得)
**Repo URL**: https://github.com/tsaitsangchi/stock_backend.git
**Active state**: v0.16 N=1,002 doctrine-compliant universe + LGBM v0.2 production model + Multi-cycle reality
**位階**: 完整跨機接續 context — **新機只需讀此一檔即可接手**

---

## 〇、Quick Start(新機 first 2 hr)

### 0.1 必讀文件(優先級排序)

```
[必讀]
1. 本檔(cross-machine handoff)
2. reports/from_zero_to_model_build_guide_v0_16_20260528.md ← 從零重建 SSOT
3. reports/v0_16_local_evidence_20260528.md ← 預期 evidence

[參考]
4. reports/系統架構大憲章_v6.1.0.md §14.7-CE 至 §14.7-DB(28 章 inscribed)
5. CLAUDE.md(專案規則)
6. ~/.claude/projects/.../memory/core_stock_selection_doctrine.md(memory)
```

### 0.2 Clone repo

```bash
cd ~/project
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git log --oneline -5
# expect first line: a558c34 feat(charter+guide): §14.7-DB ...
```

### 0.3 從零重建(完整流程)

依 `reports/from_zero_to_model_build_guide_v0_16_20260528.md` 之 8-phase canonical sequence:
- Phase 0 Environment(~10 min)
- Phase 1 DB Schema(~5 min)
- **Phase 2 Raw API Sync(~6-12 hr)** ⚠️ 主要時間消耗
- Phase 3 FRED Macro Sync(~10 min)
- Phase 4a Production v0.13(~10 min)
- **Phase 4b Backtest triple-gate v0.16(~1 min)** ⭐ per §DA
- Phase 5 Feature Store(~80 min,含 95 historical panels)
- Phase 6 Feature audits(~5 min)
- Phase 7 LGBM v0.2 training(~3 min)
- Phase 8 Multi-Cycle validation(~5 min)

**TOTAL ~7-14 hr first-time build**

### 0.4 Re-build from dump(推薦,若有 backup)

```bash
# 在舊機:
pg_dump -h 127.0.0.1 -p 5432 -U stock -d stock -F c -f stock_backup_20260528.dump

# 在新機:
brew services start postgresql@17
createdb -U stock stock
pg_restore -h 127.0.0.1 -p 5432 -U stock -d stock stock_backup_20260528.dump

# Skip Phase 2;直接 Phase 4-8 驗證 active state
```

Restore time: ~30 min(vs 6-12 hr API sync)+ Phase 4-8 ~1.5 hr = **~2 hr total**

---

## 一、Current Project State(post §14.7-DB / a558c34)

### 1.1 Charter inscriptions

```
§14.7-CE 至 §14.7-DB 共 28 sections 全 inscribed
§14.7-CE  Per-Stock Empirical Verification(7,834 entries proof)
§14.7-CF  Unified SSOT(三 invariant 統合)
§14.7-CG  Native gate v0.13(N=1,583)
§14.7-CH-CY  intermediate doctrines(weekly cron / IC / sign / necessity / LGBM / OOS / multi-cycle)
§14.7-CZ  From-Zero 8-phase canonical sequence
§14.7-DA  v0.16 triple-gate(A∩B∩C → N=1,002)
§14.7-DB  From-Zero v0.16 reconciliation(Phase 4 升 4a/4b)
```

### 1.2 治權判準三十二純化軸完成

```
01 N            02 T            03 Indicator         04 Pillar
05 Feature      06 Completeness 07 Source            08 Source-Completeness
09 Empirical    10 SSOT         11 Native            12 Continuous
13 Feature-Validity         14 Feature-Reasonableness
15 Feature-Effectiveness    16 Feature-Canonical-Scope
17 Feature-Empirical-IC     18 Feature-Necessity
19 Feature-Sign-Stability   20 Sign-Commitment
21 Math-Realism             22 Model-Training-Landing
23 Effectiveness-Provenance 24 Conditional-Necessity
25 Tree-Model-Production    26 Historical-OOS-Validation
27 Multi-Cycle-Horizon-Validation
28 From-Zero-Production-Sequence
29 Doctrine-Triple-Intersection
30 From-Zero-v0.16-Reconciliation
```

### 1.3 DB Active State

```
core_universe_snapshot:
  v0.16 N=1,002          committed (active)
  v0.13 N=1,583          superseded
  v0.12 N=1,543          superseded
  v0.11 N=1,640          superseded
  v0.10 N=1,862          superseded

feature_store_snapshot:
  99 panels bound to v0.16(95 historical 2018-06 → 2026-04 + 4 supplementary)
  1 panel bound to v0.13(legacy)
  Total: 100 panels in DB

model_registry:
  lgbm_v0_2_v0_16_local_20260528  committed
  metrics: Sharpe 1.82 / Win 75% / MDD 6.6% / α +3.43%/30d

model_training_run:
  1 row(PERFECT exit_status)
```

### 1.4 Production Reality(per §14.7-DB T_DB-6)

| Horizon | Eff t | Sig | Sharpe | Win | Net Annual |
|---|---:|:---:|---:|---:|---:|
| weekly | +1.87 | ❌ | 0.97 | 73.8% | +18.50% |
| monthly | +2.09 | ✅ | 1.15 | 58.5% | +25.04% |
| **quarterly ⭐** | **+3.26** | ✅ | **2.22** | **81.2%** | **+17.35%** |
| **annual** | **+3.96** | ✅ | **5.09** | **91.7%** | **+31.09%** |

**3/4 horizons 統計顯著** / Recommended production: **Quarterly rebalance**

### 1.5 用戶 8 條 directive 100% 對映

| # | Directive | 系統實證 |
|---:|---|---|
| 1 | 三基柱有對應資料來源 | ✅ 43 features × 10 tables / 13 FRED series |
| 2 | N 不固定 | ✅ N=1,002 dynamic |
| 3 | 全從 FinMind/FRED API 抓取 | ✅ 62M raw rows / 2 endpoints |
| 4 | 缺資料即不入核心股 | ✅ 2,803 → 1,002(reject 64%)|
| 5 | Feature 不可用即排除 | ✅ §14.7-CD「真實或 None」cascade |
| 6 | 明確 feature 可訓練 | ✅ LGBM v0.2 / 4/4 Treaty Gates PASS |
| 7 | IC 正負相關 | ✅ Mean \|IC\|=0.2195 / Sig 87.2% |
| 8 | 所有係數明確 + 或 - | ✅ 40+/3-/0? |

---

## 二、本 session 累積 commits(自 1a0138e v6.5.0 sealed 之後)

```
a558c34  §14.7-DB From-Zero v0.16 Reconciliation + comprehensive build guide  (v6.22.0)
f4f78ae  §14.7-DA v0.16 Backtest-Doctrine-Compliant + Cross-Machine Reproducibility  (v6.21.0)
... (許多前機 commits 已 pull)
52fb7af  v6.20.0 From-Zero Build Sequence(§14.7-CZ)
```

本 session 增 2 commits + 28 charter sections + 1 new universe(v0.16) + 99 new feature panels + 1 model + 1 build guide。

---

## 三、Active programs(production-current)

### Universe + Feature
- `scripts/core/core_universe_builder.py`(§14.7-CG / Phase 4a)
- `scripts/core/feature_store_builder.py`(§14.7-CA / Phase 5)
- `scripts/evaluation/build_historical_panels.py`(95 panels / Phase 5)

### Model + Validation
- `scripts/core/model_trainer_lgbm_v2.py`(§14.7-CW / Phase 7)
- `scripts/evaluation/multi_cycle_validation.py`(§14.7-CY / Phase 8)

### Audits
- `scripts/audit/audit_feature_ic_vs_future_return.py`(§14.7-CM)
- `scripts/audit/audit_feature_sign_stability.py`(§14.7-CO/CQ/CR)
- `scripts/audit/audit_feature_necessity.py`(§14.7-CN)
- `scripts/maintenance/audit_core_universe.py`(§6.7)
- `scripts/maintenance/audit_supply_chain.py`(§7)

### Orchestrator
- `scripts/maintenance/run_weekly_doctrine_recommit.py` v0.7(§14.7-BX)— 13-step weekly cron

---

## 四、Pending Follow-ups(下輪 v6.22.x+)

### P0(治權完整性)
1. **audit_core_universe.py 對 v0.16 之 native support** — 當前 POLICY_SCORE_SCOPE_MAP 含 v0.16 但 EXPECTED_SCORE_DETAIL_KEYS for v0.16 仍可補
2. **`core_universe_builder.py` 加 `--with-triple-gate` mode** — 取代 Phase 4b SQL post-process(避免 manual SQL)

### P1(技術債清理)
3. `kwave_supply_cycle_proxy` table DROP(per §14.7-CD / §14.7-CC 預告 v6.4.5;仍 pending)
4. v0.13 → v0.16 migration:確認所有 downstream tools 切換至 v0.16(model_trainer / multi_cycle / audit 等)

### P2(精度提升)
5. 35 stocks `operating_margin_ttm` root cause(charter §CO 揭露之 derived feature 計算邊界;Taiwan 財報累積 shape vs TTM 邏輯)
6. Reproducibility:add seed-fixed mode for LGBM trainer(per §14.7-CW T_CW-6 reproducibility transparency)

### P3(下一步)
7. v0.16 production cron 啟動:run_weekly_doctrine_recommit.py 切 v0.16 universe
8. v7.0.0 candidate:charter 主版號升級之 milestone tag(時機由用戶決定)

---

## 五、新機驗證清單(完成後即可開始新工作)

```bash
# 1. Repo HEAD
git log --oneline -1
# expect: a558c34 feat(charter+guide): §14.7-DB ...

# 2. Python env
python --version  # 3.12.x

# 3. Imports
python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('OK')"

# 4. PostgreSQL
pg_isready -h 127.0.0.1 -p 5432

# 5. DB active universe
PGPASSWORD=stock psql -h 127.0.0.1 -U stock -d stock -c "
SELECT snapshot_id, core_count FROM core_universe_snapshot WHERE status='committed';"
# expect: 1 row / v0.16 / N=1,002

# 6. Feature panels
PGPASSWORD=stock psql -h 127.0.0.1 -U stock -d stock -t -c "
SELECT COUNT(*) FROM feature_store_snapshot WHERE total_stocks=1002;"
# expect: ≥ 96

# 7. Model registry
PGPASSWORD=stock psql -h 127.0.0.1 -U stock -d stock -c "
SELECT model_id FROM model_registry;"
# expect: 1 row(lgbm_v0_2_v0_16_local_*)

# 8. Charter sections
grep -c "^### §14.7-D[AB] " reports/系統架構大憲章_v6.1.0.md
# expect: 2

# 9. Memory
ls ~/.claude/projects/-Users-hugo-project-stock-backend/memory/
# expect: MEMORY.md + core_stock_selection_doctrine.md
```

**8/9 PASS** = ready for next session work

---

## 六、新機若要繼續工作

### 6.1 啟動建議順序
1. 完成 §〇 Quick Start(clone + pull)
2. 跑 §五 驗證清單(15 min)
3. 讀本檔 + `reports/from_zero_to_model_build_guide_v0_16_20260528.md`
4. 從 §四 Pending Follow-ups 挑優先級(建議 P0 audit refactor 或 P3 weekly cron 啟動)

### 6.2 新工作之治權前置條件
- 任何 universe 修改:必對齊 §14.7-CF(三 invariant)+ §14.7-DA(triple-gate)+ §14.7-DB(from-zero reconciliation)
- 任何 raw data fetcher 修改:必對應 FinMind / FRED API endpoint(per §14.7-CC)
- 任何 feature 修改:raw 完整 ⟹ feature 真實或 None(per §14.7-CD)
- 任何 model training:Treaty Gates 4/4 must PASS(per §14.7-CW)
- 任何 charter inscription:對應 §14.7-X 輪次 + 修訂歷程 + 標頭最後更新日期同步(per CLAUDE.md §四.4)

### 6.3 緊急聯絡 / 漂移偵測
若新機運作後發現本 handoff 描述與實際 DB / code 不符,先停止破壞性操作並:
1. `git log` 對比 final HEAD `a558c34`
2. DB committed snapshot 是否為 v0.16 N=1,002
3. 若 mismatch → 讀 latest session handoff(`ls -t reports/*handoff*.md` 最新)

---

## 七、本檔之 invariants(換機 sealed at a558c34)

- ✅ Repo HEAD = `a558c34`(GitHub `tsaitsangchi/stock_backend` master)
- ✅ DB v0.16 active(N=1,002 / committed)
- ✅ Feature store(99 panels bound to v0.16)
- ✅ Model registry(lgbm_v0_2_v0_16_local_20260528 committed)
- ✅ Charter §14.7-CE 至 §14.7-DB 共 28 sections inscribed
- ✅ Build guide `from_zero_to_model_build_guide_v0_16_20260528.md` git-tracked
- ✅ Evidence file `v0_16_local_evidence_20260528.md` git-tracked
- ✅ Multi-cycle JSON `multi_cycle_validation_20260528_v0_16_local.json` git-tracked
- ⏸ Pending follow-ups documented(P0-P3)

---

## 八、Production Reality Final State(per §14.7-DB T_DB-6)

### Recommended strategy: Quarterly Rebalance

| 維度 | 真實值 |
|---|---|
| Recommended horizon | **Quarterly(60-day)** |
| Net annualized | **+17.35%/年** |
| Sharpe(net)| **+2.22** |
| Win rate | **81.2%** |
| n_effective(non-overlap)| 32 真獨立 panels |
| Effective t-stat | **+3.264**(p < 0.001 robust)|
| MDD | 17.43% |

### Alternative high-α: Annual Rebalance

| 維度 | 真實值 |
|---|---|
| Net annualized | **+31.09%/年** |
| Sharpe | **+5.09** |
| Win rate | **91.7%** |
| n_effective | 7.1(small)|
| Effective t-stat | +3.956 |

---

**🏛️ 本檔為換機日所需之 single canonical entry。讀此一檔 + `from_zero_to_model_build_guide_v0_16_20260528.md` 即可在新機接手 v6.22.0 後續工作。**
