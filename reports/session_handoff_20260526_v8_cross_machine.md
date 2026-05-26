# Session Handoff v7 → v8 — Cross-Machine Continuation Handoff

**版本**: v8(cross-machine focus)
**日期**: 2026-05-26
**前置 handoff**: `session_handoff_20260526_v7.md`
**本 handoff 觸發**: 用戶明示「再晚一點會換另一台電腦做此專案」
**HEAD(at handoff time)**: `73228ba` (已 push origin/master)
**Branch**: master / **Working tree**: clean
**最新 tag**: `v6.3.1-feature-store-v08-rebuilt`

---

## 一、跨機接續最重要 5 件事(讀完即可開工)

### 1. Repo 與 HEAD 狀態
- Remote: `https://github.com/tsaitsangchi/stock_backend.git`
- HEAD: `73228ba`(已 push;新機 `git pull` 即同步)
- Branch: `master`
- Working tree: **clean**(0 uncommitted / 0 untracked)
- Latest 2 tags: `v6.3.1-feature-store-v08-rebuilt`(milestone) / `v6.3.0-trinity-charter-audit-seal-20260526`(seal)

### 2. 新機 setup 4 步(per CLAUDE.md §二 #7)

```bash
# Step 1: clone + checkout
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git checkout master && git pull origin master  # 確認在 73228ba

# Step 2: OS 原生依賴(per §0.0-I.9)
# macOS:
brew install libomp postgresql@17
# Linux:
sudo apt-get install -y libgomp1 libpq-dev
# Windows: 通常無需

# Step 3: Python venv + dependencies
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Step 4: 必跑 import smoke test(per §0.0-I.9)
.venv/bin/python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅ all imports OK')"
# 失敗則先補 OS 層依賴,否則勿執行任何 sync/audit
```

### 3. .env 14 變數設定(per §0.0-I.8;新機必設)

需在新機建 `/path/to/stock_backend/.env` 並填:

**強制 7 變數**:
- `PROJECT_ROOT` = 新機絕對路徑(macOS 注意 `/Users` 不是 `/home`;path_setup v4.48 用 `os.path.realpath()` 解析)
- `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASS`(PostgreSQL 連線五件套)
- `FINMIND_TOKEN` / `FRED_API_KEY`(API 金鑰)

**可選 7 變數**: `ENV` / `LOG_LEVEL` / `TZ` / `MLFLOW_TRACKING_URI` / `GEMINI_API_KEY` / `GITHUB_TOKEN` / `STORAGE_BACKEND`

**警示**: `.env` 在 `.gitignore` 內(不會被 push);**新機必須手動 setup**,或從舊機加密複製。

### 4. DB 狀態(若新機亦走連同一 PostgreSQL 則自動同步;否則需 rebuild)

**現行 production-current snapshot**:
- `core_universe_20260521_core_universe_policy_v0_8_dynamic`
- 119 stocks(83 core + 36 convex)
- policy: `core_universe_policy_v0.8_dynamic`(per §6.7.1 dynamic annex)

**3 個 schema layer 狀態(per 73228ba)**:
- governance(5 tables): ✅ committed
- feature_store(3 tables): ✅ init done(本 session 跑 `feature_store_schema.py --init`)
- feature_values: ⏸ dry-run only(未 commit;Pri #1 next session 推)
- prediction(0 tables): ⚠️ 完全空缺(per §14.7-BU Phase A 設計研究指出)

**若新機獨立 DB**,從零重建序列(per §10 創世圓滿宣言第 6 條):
```bash
.venv/bin/python scripts/core/path_setup.py
.venv/bin/python scripts/core/data_schema.py --init --force
.venv/bin/python scripts/core/core_universe_schema.py --init
.venv/bin/python scripts/core/db_utils.py
.venv/bin/python scripts/maintenance/audit_supply_chain.py --include-logs
.venv/bin/python scripts/ingestion/sovereign_sync_engine.py --seed
.venv/bin/python scripts/core/core_universe_builder.py --commit \
  --policy-version core_universe_policy_v0.8_dynamic \
  --special-rebalance-reason "cross-machine rebuild 2026-05-26 v6.3.1"
.venv/bin/python scripts/maintenance/audit_core_universe.py
.venv/bin/python scripts/core/feature_store_schema.py --init
```

### 5. Next session 第一個指令(直接執行即可推進)

**Pri #1 — Feature Store production commit**(~5 min):
```bash
.venv/bin/python scripts/core/feature_store_builder.py --commit \
  --as-of-date 2026-05-21 \
  --feature-set-version feature_set_v0.4_v08_universe_production \
  --label-horizon 20
```

---

## 二、本 session 至 73228ba 完整 increment(v6 closure 後 4 commits / 2 tags)

### 2.1 Commits 表(依時序)

| Commit | 內容 | 對應 priority |
|---|---|---|
| `542fbfd` | core_universe_audit v0.3 execution snapshot evidence(37/1/4) | v6 §七 #1 留存 |
| `98dbc67` | feature_store v0.8 universe rebuild dry-run PERFECT(119 × 31 × 3,689) | v6 §七 #2 ✅ |
| `302d5d2` | session handoff v6 → v7 | meta |
| `73228ba` | §14.7-BU Phase A design research(15 章 / Path C hybrid 推薦) | universe_completeness governance Phase A |

### 2.2 Tags 表

| Tag | Commit | 意義 |
|---|---|---|
| `v6.3.0-trinity-charter-audit-seal-20260526` | 542fbfd | 三基柱 + Dynamic Universe 入憲 seal |
| `v6.3.1-feature-store-v08-rebuilt` | 98dbc67 | Feature Store v0.8 rebuild milestone |

### 2.3 Reports 增量

| Report | 主題 |
|---|---|
| `reports/core_universe_audit_20260526_1603.md` | audit v0.3 執行證據 |
| `reports/feature_store_v08_implementation_audit_20260526.md` | Feature Store v0.8 rebuild evidence(v6.3.1 核心) |
| `reports/session_handoff_20260526_v7.md` | handoff v7 |
| `reports/universe_completeness_governance_design_research_20260526.md` | §14.7-BU Phase A 設計研究 |
| `reports/session_handoff_20260526_v8_cross_machine.md` | **本 handoff(v8)** |

### 2.4 .gitignore 修改

- 新增 L100: `!reports/universe_completeness_*.md`(為 §14.7-BU Phase B-G 後續 reports 預備)

---

## 三、Pending 決策(新機接續時用戶須明示)

### 3.1 §14.7-BU 後續 Phase 路徑(Path C hybrid 推薦)

| Phase | 內容 | 時間 | 是否 destructive |
|---|---|---|---|
| **B** | 入憲 §14.7-BU + 修訂歷程第十九輪 | ~30 min | 否(charter 修改) |
| **C** | universe_completeness_schema.py --init(3 表 + 1 view) | ~30 min | 否(only INIT 新表) |
| **D** | prediction layer schema 補建(prediction_run + predictions) | ~20 min | 否(only INIT 新表) |
| **E** | 4 builders 補 audit hooks(fetcher / feature_store / model_trainer / prediction_engine) | ~2-3h | 否(只加新 write) |
| **F** | audit_universe_completeness.py v0.1 + materialized view refresh | ~1h | 否 |
| **G** | v6.3.2(或 v6.4.0) milestone tag + handoff v9 | ~30 min | 否 |

**用戶可選 3 option**:
- α(推進): Phase B+C+D(~80 min)+ 跨 session Phase E-G
- β(穩健): Phase B only(~30 min)+ 跨 session Phase C-G
- γ(全推): Phase B → G(~6-8h)需 SHMM 心跳保護

### 3.2 Feature Store cascading rebuild 4 priority(原 v7 handoff 已列)

| Pri | 任務 | 時間 |
|---|---|---|
| **#1** | feature_store_builder `--commit` 產線寫入 | ~5 min |
| **#2** | model_trainer 重訓 + walk-forward IC validation | ~30-60 min |
| **#3** | prediction_engine 切 v0.8 | ~10 min |
| **#4** | portfolio_sizer dynamic universe review | ~20-30 min |

### 3.3 Independent priorities(非 cascading)

| Pri | 任務 | 對應憲章 |
|---|---|---|
| #5 | audit_core_universe v0.3 之 4 FAIL refinement | handoff v6 §七 #1 |
| #6 | feature_set v0.4 → v0.5 升版(加 M2SL + TW proxies + CashFlow) | §14.7-BR 5/5 |
| #7 | §14.7-BO Phase B(CashFlow sync) | §14.7-BO |
| #8 | §14.7-BM Phase B(金融業 ROE 落地 post BS sync) | §14.7-BM |

---

## 四、本 session marathon 累積總成(自 v4 closure 後)

### 4.1 全部 tags(新到舊)

```
v6.3.1-feature-store-v08-rebuilt           ← v6.3.1 milestone
v6.3.0-trinity-charter-audit-seal-20260526  ← v6.3.0 seal
v6.3.0-dynamic-universe-milestone-landing   ← v6.3.0 milestone
v6.1.28.4-dynamic-universe-phase-d2-snapshot-committed
v6.1.28.3-dynamic-universe-phase-d1-deprecate
v6.1.28.2-dynamic-universe-phase-c-builder-rewritten
v6.1.28.1-dynamic-universe-phase-b-inscribed
v6.1.28-final-5-of-5                        ← §14.7-BR closure
v6.1.28-kwave-c4-shipping-landed-5-of-5
v6.1.28-kwave-c3-audit-landed
... (其餘 22+ tags per v6 handoff)
```

### 4.2 主要 charter inscriptions

- v6.1.0-patch 第十六輪: §14.7-BR Phase B(K-wave 5/5 leading indicators 治權)
- v6.1.0-patch 第十七輪: §14.7-BR Phase C-4 + §0.3.8.3 BDI 追溯
- v6.1.0-patch 第十八輪: §14.7-BT Phase B(§6.7.1 dynamic annex + 取消 150 hardcode)
- v6.1.0-patch 第十九輪: ⏸ pending(§14.7-BU Phase B 待入憲)

### 4.3 主要 milestones

- v6.2.0(`v6.2.0-model-trainer-phase-c-d-complete`): §10 model_trainer Phase C-D
- v6.3.0(`v6.3.0-dynamic-universe-milestone-landing`): §14.7-BT 取消 150 hardcode
- v6.3.1(`v6.3.1-feature-store-v08-rebuilt`): Feature Store v0.8 rebuild(本 handoff 主)
- v6.3.2 / v6.4.0: ⏸ pending(§14.7-BU completeness governance 待決路徑)

---

## 五、跨機環境注意事項(per CLAUDE.md §二 #7 + §6)

### 5.1 路徑解析(per §0.0-I.10)
- macOS: `/Users/<user>` ≠ `/home/<user>`(後者為 Linux);
- `path_setup.py v4.48+` 用 `os.path.realpath()` 解析,自動處理 symlink
- 若新機路徑差異大,須改 `.env PROJECT_ROOT`

### 5.2 SHMM 心跳保護(若 long-running 跑 >30 min)
- N ≥ 3 個 Monitor heartbeat(15/20/25 min 組合)
- Sentinel timestamp: `/tmp/claude_loop_last_fire.txt`
- Watchdog: 每 60s 檢查 sentinel age
- 詳見 CLAUDE.md §二 #6 SHMM protocol

### 5.3 外部資源驗證 protocol(per §14.7-AX(E))
- 任何 API tier 判定前必須先 call `user_info`(如 FinMind `/api/v4/user_info`)
- 不從 sync error 直接推測 tier
- 區分「tier 不足(永久 blocked)」vs「quota 暫時耗盡(可重試)」

---

## 六、Session 累積統計

| 維度 | 累積 |
|---|---|
| 自 v4 closure(0cb61a4)後 commits | 9 |
| 自 v6.2.0.1 後 tags | 9 |
| Charter inscriptions | 3 rounds(第十六/十七/十八輪)+ 1 pending(第十九輪) |
| Major milestones | v6.2.0 / v6.3.0 / v6.3.1 + v6.3.2 pending |
| Design research reports | 4(§14.7-BR Phase A / §14.7-BT Phase A / dynamic universe Phase A / §14.7-BU Phase A) |
| Empirical evidence reports | 5+(含 5/5 leading indicators / Phase D-2 / v6.2.0 production smoke / audit v0.3 / feature_store v0.8) |
| Honest amendment | 1(v6_2_0_honest_amendment_20260526) |
| Cross-machine handoffs | 3(v6 / v7 / v8 本) |

---

## 七、治權誠實三軸(cross-machine handoff time)

| 軸 | 狀態 | 證據 |
|---|---|---|
| **Implementation completion** | ✅ ~100%(at code/dry-run level) | v6.3.1 PERFECT |
| **Charter alignment** | ✅ ~95%(charter inscribed for all landed milestones) | 第十八輪 inscribed;第十九輪 pending |
| **Production-empirical IC validation** | ⏸ 0%(待 Pri #2-#3) | 待 model retrain + walk-forward |
| **Epistemological ceiling** | ~95% | per §0.3 decades constraint |

---

## 八、新機第一個 session 啟動建議流程

```
1. git clone + checkout master + pull → 確認在 73228ba
2. OS 依賴 + Python venv + import smoke test
3. .env setup(14 變數;從舊機加密複製或手動填)
4. (可選)DB rebuild(若新機獨立 DB)
5. 讀 4 文件(per CLAUDE.md §四 #1):
   a. 本 handoff(session_handoff_20260526_v8_cross_machine.md)
   b. universe_completeness_governance_design_research_20260526.md(§14.7-BU Phase A)
   c. feature_store_v08_implementation_audit_20260526.md(v6.3.1 milestone)
   d. CLAUDE.md(治權規則 SSOT)
6. 詢問用戶第一個 priority(本 handoff §三 列出 9 個 candidates)
7. 啟動 work loop
```

---

## 九、結論

**本 session 至 73228ba 已完整 commit + push 至 GitHub**;working tree clean;可隨時切換到新機接續。

**最關鍵 3 件事**:
1. **HEAD 為 `73228ba`**(已 push)
2. **`.env` 必須在新機手動 setup**(14 變數;不在 repo 內)
3. **DB 狀態**:若新機獨立 PostgreSQL,需走憲章 §10 第 6 條 9 步重建序列;若連同舊機 DB 則自動同步

**Next session 建議第一動作**:讀本 handoff + 詢問用戶 §三 9 個 priority 候選之偏好。

---

**Handoff 作者**: Claude(本 session ultra-ultra-ultra-ultra-ultra-long marathon continuation)
**Cross-machine reading order**:
1. 本 handoff(`session_handoff_20260526_v8_cross_machine.md`)← **先讀**
2. `universe_completeness_governance_design_research_20260526.md`(最新設計研究)
3. `feature_store_v08_implementation_audit_20260526.md`(v6.3.1 milestone evidence)
4. `session_handoff_20260526_v7.md`(若需 v7 細節 context)
5. `CLAUDE.md`(治權規則)
6. 憲章 §6.7.1 / §14.7-BT / §14.7-BU pending(若需深度 charter context)
