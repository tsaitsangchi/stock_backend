# Session Handoff v12 — 跨機接續完整 context(2026-05-26 結束 → 2026-05-27+ 另機開工)

**日期**: 2026-05-26 結束 → 明日另機接續
**HEAD**: `b923e4c`(已 push 到 origin/master)
**封存 tag**: **`v6.1.19-doctrine-pure-weekly-recommit-20260526`**(已 push)
**位階**: 完整跨機接續 context(本檔 standalone;另機只需讀此一檔即可接手 / v9-v11 為歷史參考)
**前置必讀(若想看歷史)**: [v9](session_handoff_20260526_v9_4round_doctrine_closure.md) → [v10](session_handoff_20260526_v10_5round_doctrine_closure.md) → [v11](session_handoff_20260526_v11_phase_c_landing.md)

---

## 〇、Quick Start(另機 first 30 min)

### 0.1 Clone repo + 環境設置

```bash
# Step 1: Clone
cd ~/project          # 或你的 project root
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend

# Step 2: 切到本 session 之封存點
git checkout v6.1.19-doctrine-pure-weekly-recommit-20260526
# (建議再 checkout master:`git checkout master`;tag 為固定 reference)

# Step 3: 設 venv(Python 3.12)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt    # (若無 requirements.txt;見 §一.5 manual install)

# Step 4: OS 原生依賴(per CLAUDE.md §二.7)
# macOS:
brew install libomp postgresql@17
# Linux:
sudo apt-get install -y libgomp1 libpq-dev

# Step 5: Import smoke test(必須通過)
python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅ all imports OK')"

# Step 6: 從上一台機複製 .env(含 DB_*, FINMIND_TOKEN, FRED_API_KEY)
# ⚠️ GITHUB_TOKEN 已在 chat transcript 暴露,**先 revoke 重生**
cp /path/from/old/machine/.env .

# Step 7: 驗證 DB 連線
source venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'scripts')
from core.db_utils import get_db_connection
conn = get_db_connection()
cur = conn.cursor()
cur.execute('SELECT snapshot_id, status FROM core_universe_snapshot WHERE status=%s', ('committed',))
print(cur.fetchall())
conn.close()
"
# 預期輸出:[('core_universe_20260526_core_universe_policy_v0_10_pure_doctrine_weekly', 'committed')]
```

### 0.2 必讀順序(another AI 接手 protocol)

| 順序 | 文件 | 用途 |
|---|---|---|
| 1 | **本 handoff v12**(本檔) | 整體 context 之 entry point |
| 2 | `reports/系統架構大憲章_v6.1.0.md` §14.7-BX 子節(L9655)+ 修訂歷程 第二十二輪 row(L66)| 最新治權 |
| 3 | `reports/系統架構大憲章_v6.1.0.md` §14.7-BW 子節(L9551)| Pure doctrine(N-axis 純化)|
| 4 | `reports/系統架構大憲章_v6.1.0.md` §14.7-BU 子節(L9378)| Governance SSOT |
| 5 | Memory: `/Users/<user>/.claude/projects/-Users-<user>-project-stock-backend/memory/core_stock_selection_doctrine.md` v4 | Doctrine quick reference |
| 6 | (optional)各輪 Phase A research:`reports/{universe_completeness_governance,doctrine_gate_selection,pure_doctrine_drift_closure,weekly_doctrine_recommit}_phase_a_research_20260526.md` | 細節設計 |
| 7 | (optional)`reports/session_handoff_20260526_v9_4round_doctrine_closure.md` → v10 → v11 | 漸進 narrative |

---

## 一、本 Session 整體成果(2026-05-26 / 全 8 commits)

### 1.1 治權層 5 輪累進(BT→BU→BV→BW→BX)

| 第輪 | 條目 | Doctrine 維度 | Phase B/C 落地狀態 |
|---|---|---|---|
| 第十八輪 §14.7-BT | 取消 §6.7 SSOT 150 hardcode prescription | N-prescription | ✅ (prior session)|
| 第十九輪 §14.7-BU | 建 universe_completeness_snapshot SSOT | governance infra | ✅ B+C+D+F done |
| 第二十輪 §14.7-BV | 改 selection criterion 為 doctrine-gate-first | criterion(含 hidden hardcode)| ✅ B done / Path C drift 後 superseded by BW |
| 第二十一輪 §14.7-BW | 揭露 BV drift + 升 Path D pure doctrine | **N-axis 純化** | ✅ B done |
| 第二十二輪 §14.7-BX | weekly recommit(annual freeze 為 hidden temporal hardcode)| **T-axis 純化** | ✅ B+C-1+C-3+C-4 done |

### 1.2 用戶 6 次累進明示之 doctrine 觸發

| 次 | 明示內容 | 觸發 |
|---|---|---|
| 1 | 「沒有一定要多少支核心股,但必須符合三基柱具有對應資料來源為依據」 | doctrine 釐清 framing |
| 2 | 「取消 150 在任何地方」 | active deprecation |
| 3 | 「取消 150 支核心股」+ restate | 強化 |
| 4 | 「先全部取消 150 支核心股」 | imperative execute |
| 5 | 「取消所有 200 支及 150 支」 | 揭露 Path C hidden hardcode → §14.7-BW |
| 6 | 「每週跑 builder v0.10 + commit 新 snapshot。改憲章」 | T-axis 升版 → §14.7-BX |

---

## 二、Pushed Commits(已 on origin/master)

```
b923e4c  docs(handoff) v11               — §14.7-BX Phase C 落地
9561c9a  feat(code): §14.7-BX Phase C-1+C-3+C-4 落地 — Weekly Recommit Pipeline + Schema enum + Atomic Supersede
2cdaa25  docs(handoff) v10               — 5-round doctrine treaty closure
1d4cad1  feat(charter+research): §14.7-BX Phase A+B closure — Weekly Doctrine-Driven Recommit
24cdbf9  docs(handoff) v9                — 4-round doctrine treaty closure
532d589  feat(charter+research+code): §14.7-BW Phase A+B closure — Pure Doctrine Drift Closure
31a9c7a  feat(charter+research+evidence): §14.7-BV Phase A+B closure — Doctrine-Gate-First Universe Selection
1c61eed  feat(charter+schema+audit): §14.7-BU Phase B+C+D+F closure — Cross-Layer × Cross-Pillar Universe Completeness Governance
```

**封存點 tag**:`v6.1.19-doctrine-pure-weekly-recommit-20260526`(annotated tag with full milestone notes)

GitHub URLs:
- Master: https://github.com/tsaitsangchi/stock_backend
- Tag: https://github.com/tsaitsangchi/stock_backend/releases/tag/v6.1.19-doctrine-pure-weekly-recommit-20260526

---

## 三、Charter State(reports/系統架構大憲章_v6.1.0.md)

### 3.1 本 session inscribed sections

| Section | Charter line | Topic |
|---|---|---|
| §14.7-BU | L9378 | Cross-Layer × Cross-Pillar Universe Completeness Governance |
| §14.7-BV | L9462 | Doctrine-Gate-First Universe Selection(Path C / **historical narrative** after BW)|
| §14.7-BW | L9551 | Pure Doctrine Drift Closure(N-axis 純化;**升 Path D 為正式 doctrine**)|
| §14.7-BX | L9655 | Weekly Doctrine-Driven Recommit(T-axis 純化;§6.8 frequency clause **不適用**)|

修訂歷程 rows at L66:第十九/二十/二十一/二十二輪。

Charter total lines:**9,777**(prior session 9,406 → +371)。

### 3.2 治權層讀法指引(critical)

- **§14.7-BV Path C** 在 charter L9462 仍存在,但 **讀為 historical narrative**(§14.7-BW 已 supersede)
- **§14.7-BV Stage 4 之 70/30 tier split / N_max=200 cap / N_min=100 floor**:全 deprecated by §14.7-BW
- **§6.7.1 dynamic size annex**(L3421,§14.7-BT)之 N_max/N_min/core_pct values:**Path D 下不適用**(historical)
- **§6.4 CoreScore 公式不動,治權位階降為 INFO display only**(§14.7-BW 第 5 條)
- **§6.8 制度穩定性條文不撤銷,但 annual frequency clause 在 Path B/BX 下不適用**(§14.7-BX 第 2-4 條)
- **§14.7-BU governance infrastructure 不動**(universe_completeness_snapshot + matview + audit)

---

## 四、DB State(post 本 session)

### 4.1 Snapshot inventory

```
✅ [committed  ] 2026-05-26 core= 1862  core_universe_20260526_core_universe_policy_v0_10_pure_doctrine_weekly
⤴️ [superseded ] 2026-05-26 core= 1862  core_universe_20260526_core_universe_policy_v0_10_pure_doctrine
❌ [deprecated ] 2026-05-26 core=  140  core_universe_20260526_core_universe_policy_v0_9_doctrine_gate
❌ [deprecated ] 2026-05-24 core=  120  core_universe_20260524_core_universe_policy_v0_2
❌ [deprecated ] 2026-05-22 core=  120  core_universe_20260522_core_universe_policy_v0_7
❌ [deprecated ] 2026-05-22 core=  120  core_universe_20260522_core_universe_policy_v0_3
```

### 4.2 §6.7 SSOT 治權查詢預期

```sql
SELECT m.stock_id FROM core_universe_membership m
JOIN core_universe_snapshot s ON m.snapshot_id = s.snapshot_id
WHERE s.status='committed' AND m.core_tier IN ('core_universe','convex_universe');
-- 預期返回 1,862 stock_ids
```

### 4.3 §14.7-BU governance tables

```
prediction_run:                          0 rows(待 prediction_engine 落地)
predictions:                             0 rows(同上)
universe_completeness_snapshot:          ~11,172 rows(2 snapshots × 3 pillars × 1862;v0.10 + v0.10_weekly)
universe_completeness_matrix_current:    matview;refresh 後展示當前 1862 records
```

### 4.4 §0.3 K-wave indicators(已 sync 5/5)

```
fred_series.M2SL:                      435 rows  / 1990-01 → 2026-03
fred_series.T10Y2Y:                    9,104 rows / 1990-01 → 2026-05
fred_series.VIXCLS:                    9,191 rows / 1990-01 → 2026-05
kwave_supply_cycle_proxy.TW_SEMI:      401 rows  / 1992-01 → 2026-05
kwave_supply_cycle_proxy.TW_SHIPPING:  401 rows  / 1992-01 → 2026-05
```

### 4.5 §0.1 raw data freshness

```
TaiwanStockPriceAdj                       max=2026-05-22 / 10.5M rows
TaiwanStockPrice                          max=2026-05-25 / 10.5M rows
TaiwanStockFinancialStatements            max=2026-03-31 / 2.66M rows
TaiwanStockMonthRevenue                   max=2026-05-01 / 460K rows
TaiwanStockInstitutionalInvestorsBuySell  max=2026-05-25 / 25M rows
TaiwanStockMarginPurchaseShortSale        max=2026-05-25 / 7.7M rows
```

### 4.6 status enum constraint

```
CHECK (status IN ('committed', 'superseded', 'deprecated', 'draft'))
```

---

## 五、Tool Inventory(本 session 落地之可用工具)

### 5.1 Builder

**`scripts/maintenance/build_doctrine_gate_universe.py` v0.10+§14.7-BX**
- One-shot mode:`--commit`(legacy v0.10 pure doctrine 風格)
- Weekly mode:`--commit --weekly-mode`(§14.7-BX:atomic supersede prior committed + `_weekly` suffix snapshot_id)
- Dry-run:`--dry-run`(不寫 DB;顯示 4-stage pipeline 結果)

### 5.2 Weekly Pipeline Orchestrator(§14.7-BX Phase C-3)

**`scripts/maintenance/run_weekly_doctrine_recommit.py` v0.1**

5-step pipeline:
1. FRED sync(可 `--skip-fred-sync`)
2. TW_SEMI_VWAP_YOY proxy
3. TW_SHIPPING_VWAP_YOY proxy
4. builder --weekly-mode --commit
5. audit_universe_completeness.py
6. Drift report 生成

執行:
```bash
# Dry-run (看 pipeline 不執行):
python scripts/maintenance/run_weekly_doctrine_recommit.py --dry-run --force-now

# Commit (真跑):
python scripts/maintenance/run_weekly_doctrine_recommit.py --commit
# (若非週五 13:30 後則需 --force-now)
```

### 5.3 Schema Migrations

**`scripts/maintenance/migrate_snapshot_status_superseded_20260526.py` v0.1**
- 冪等(idempotent);加 CHECK constraint enforcing status enum {committed, superseded, deprecated, draft}
- 已在本機跑過;新機 clone 後須跑一次:`--commit`

### 5.4 Governance Schemas(per §14.7-BU)

**`scripts/core/universe_completeness_schema.py` v0.1**
- 建 3 tables(prediction_run / predictions / universe_completeness_snapshot)+ 1 matview
- 已在本機跑過;新機跑:`--init`

### 5.5 Audit Tool(per §14.7-BU Phase F)

**`scripts/maintenance/audit_universe_completeness.py` v0.1**
- 12 checks(C1-C4 schema strict / C5-C12 data integrity)
- Verdict:PERFECT 🎯 / EMPTY_AWAITING_PHASE_E / WARNING / FAILED
- Console + JSON 輸出

---

## 六、Environment & Auth

### 6.1 .env 必備 keys(從舊機複製)

```
DB_HOST=...        DB connection
DB_PORT=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
FINMIND_TOKEN=...  FinMind sponsor token(v0.3 active;2026-06-24 到期)
FRED_API_KEY=...   FRED 之 macro indicators sync 用
GITHUB_TOKEN=...   ⚠️ 已暴露於 chat transcript;革命此 token 後重生
```

### 6.2 Python 環境

- Python 3.12.13(per `venv/pyvenv.cfg`)
- 主要依賴:psycopg2-binary / pandas / polars / numpy / requests / scikit-learn / xgboost / lightgbm
- 若 `requirements.txt` 缺,從 `pip freeze` 既有 venv 取

### 6.3 OS dependencies(per CLAUDE.md §二.7 / 跨平台前置)

| OS | Setup |
|---|---|
| macOS | `brew install libomp postgresql@17` |
| Linux | `sudo apt-get install -y libgomp1 libpq-dev` |
| Windows | 通常無需(vcomp140.dll 內建)|

---

## 七、Pending Cross-Session Work(優先級排序)

### 7.1 第一優先(blocking weekly cron 啟動)

**Phase C-2:M1/M2/M3 model retrain 策略治權選定**

Per §14.7-BX Phase A research §5.2,3 個 sub-options 擇一:

| Option | 含義 | Pros | Cons |
|---|---|---|---|
| **M1 Incremental(推薦)** | model fine-tune from prior week's checkpoint | ~5x annual cost / 連續性 | 需 checkpoint 機制 / catastrophic forgetting 風險 |
| M2 Lazy trigger | weekly commit 但 retrain 僅在 drift > 5% | compute 低 | trigger threshold 本身為 implicit hardcode |
| M3 Parallel | weekly walk-forward IC / model 不重 train | compute 最低 | distribution shift 風險 |

**Action**:next session 用戶 explicit 選定後,進 Phase D-2 落地。

### 7.2 第二優先(weekly cron 啟動之前置)

**Phase D-2:model_trainer / feature_store weekly mode 升版**

- `scripts/core/model_trainer.py` 加 weekly-aware mode(per C-2 選定之 sub-option)
- `scripts/core/feature_store_builder.py` 加 weekly feature_set 升版機制

預估 ~5-8h cross-session work。

### 7.3 第三優先(可獨立並行)

**Phase D-3:拆 3 HARD BLOCK(downstream production code)**

- `scripts/core/prediction_engine.py` L238-239:`if len(self.rows) != 150: FAIL` → dynamic N from snapshot
- `scripts/core/portfolio_sizer.py` L116:`"required_coverage": 150` → dynamic
- `scripts/core/portfolio_sizer.py` L174-175:G2 gate 150 → dynamic
- `scripts/core/portfolio_sizer.py` L467:cosmetic log
- `scripts/core/feature_store_builder.py` L271:WARN 150 → dynamic

預估 ~2-3h。可獨立於 Phase C-2/D-2 並行做。

### 7.4 後續

- Phase D-1:launchd / cron 排程配置(C-2 + D-2 完成後;~30 min)
- Phase E:audit 升版加 weekly snapshot 識別 + storage archive policy(~1h)
- Phase F:first 4-week integration test(post-deploy / ~1 month real-time observation)
- Phase G:v6.4.0 / v7.0.0 milestone tag(~30 min)
- §9.2 portfolio_sizer v0.4 tier-less 升版(§14.7-BW Phase D / sub-option X/Y/Z 待研究)

---

## 八、Critical 禁令(next session 不可違)

### 8.1 治權邊界禁令

- ❌ **不可寫「目前核心股 = 150 / N = 200 / top X% / N_MAX=200 cap」之 active prescription** 到 charter / scripts / reports / commit messages
- ❌ **不可寫「年度頻率 / annual rebalance / monthly recommit」對 Path B/BX 文脈之 active prescription**
- ❌ **不可將 CoreScore 用作 selection gate**(只可 INFO display)
- ❌ **不可未經 explicit auth 跑 §9.2 v0.4 升版**(sub-option X/Y/Z 須先治權研究)
- ❌ **不可未經 explicit auth 拆 3 HARD BLOCK**(production code 改動)
- ❌ **不可未經 explicit auth 修改 charter / 入憲新治權 section**

### 8.2 自動化禁令

- ❌ **不可直接掛 launchd / cron 跑 `run_weekly_doctrine_recommit.py` 自動跑**(Phase C-2 + D-2 未完成前)
  - 違反後果:model 用舊 N 預期跑新 N=1862 → distribution shift / IC 崩
- ✅ **可手動跑 runner**(治權者 ad-hoc trigger;本 session 已 end-to-end 驗證)
- ✅ **可讀 charter + memory + 跑 audit + dry-run**

### 8.3 治權層 Don't-touch(per §0.0-I + §14.7 precedent)

- §14.7-BU/BV/BW/BX inscribed sections **不追溯修正**;任何治權變動須新建 §14.7-BY...
- §6.7.1 annex(§14.7-BT)values 不刪除(historical record)
- §6.8 條文不撤銷(只 frequency clause 失效;sync infrastructure clauses 維持)
- §6.4 CoreScore 公式不動(只治權位階改動 by §14.7-BW)
- v0.10_pure_doctrine_weekly committed snapshot 不重 build(下次 weekly cron 啟動會自動 supersede)

---

## 九、Open Questions(待用戶決定)

| Q | Context | Status |
|---|---|---|
| Q1 | **§14.7-BX M1/M2/M3 sub-option 選定** | **next session 第一優先;blocking weekly cron 啟動** |
| Q2 | §9.2 v0.4 sub-option(Option X/Y/Z barbell 重設計)| §14.7-BW Phase D pending |
| Q3 | builder canonicalization(maintenance vs core 之 file location)| §14.7-BW Phase C / **next-session optional** |
| Q4 | v6.4.0 vs v7.0.0 升版判斷 | Phase G milestone tag |
| Q5 | Weekly frequency fine-tune(daily / bi-weekly / monthly)| Phase F post-deploy / 1 month observation 後決定 |
| Q6 | GitHub PAT revoke + 重生 token 是否已執行 | ⚠️ 本檔提醒(已暴露於 chat) |

---

## 十、Memory(跨機需重新建立 / 或從舊機複製)

### 10.1 Memory 位置

舊機:`/Users/<user>/.claude/projects/-Users-<user>-project-stock-backend/memory/`

新機:相同路徑(若 Claude Code 行為一致)— Claude Code 之 memory 為 per-machine 之 local file。新機 fresh AI 接手時,memory 為空。

### 10.2 重要 memory file 之內容(可手動 recreate)

**`core_stock_selection_doctrine.md` v4**(本 session 累積 update):
- 核心股挑選 doctrine:三基柱資料源依據(§0.1 + §0.2 + §0.3)
- N 為判準結果,不為 hardcode(無 cap / floor / target / tier % / percentile)
- 頻率為 weekly(per §14.7-BX 第二十二輪)
- 不在任何地方寫「目前核心股 = 150 / N = 200 / top X% / N_MAX cap / 年度頻率」之 prescription
- §0.3 為 market-level(non-per-stock)
- 6 次累進 user 明示之 trigger 紀錄

### 10.3 新機 fresh AI 之 memory 啟動建議

新機 AI 第一次跑時,從本 handoff §1.2 + 治權禁令(§八)reconstruct memory file。

---

## 十一、Session 統計(2026-05-26 全 session)

| Metric | 值 |
|---|---:|
| Commits added(本機後 push)| 8 |
| Charter sections inscribed | 4(§14.7-BU/BV/BW/BX)|
| Charter 修訂歷程 rows | 4(第十九-二十二輪)|
| Charter lines added | +371 |
| Phase A research reports | 4(BU 523 / BV 566 / BW 372 / BX 317)|
| Handoff snapshots | 4(v9 / v10 / v11 / v12 本檔)|
| Scripts created | 5(schema + audit + builder + runner + migration)|
| Scripts modified | 1(builder +weekly-mode + docstring)|
| DB schema changes | 1(status enum CHECK constraint)|
| Treaty rounds | 4(BU/BV/BW/BX)|
| User explicit directives | 6 次 doctrine 累進 |
| End-to-end weekly pipeline | ✅ verified |
| GitHub milestone tag | `v6.1.19-doctrine-pure-weekly-recommit-20260526` |
| Total session duration | ~8 hours dense charter + code work |

---

## 十二、技術債(remaining;新機接手後可處理)

- **9.1** Builder file location 模糊(maintenance vs core 之 canonical)
- **9.2** Downstream 3 HARD BLOCK(prediction_engine / portfolio_sizer × 2)— Phase D-3
- **9.3** §9.2 portfolio_sizer barbell tier-less 重設計(convex 暫缺)— sub-option 待研究
- **9.4** §14.7-BU Phase E feature/model/prediction layer hooks(目前只 data layer)
- **9.5** data_audit_log ON CONFLICT 警告(pre-existing;若日後跑 `migrate_data_audit_log_dedup_20260525.py` 可消解)
- **9.8** ⏸ M1/M2/M3 sub-option 治權選定(blocking C-2)
- **9.9** ⏸ Weekly cron / launchd 觸發機制(blocked by C-2 + D-2)
- **9.10** ⏸ Storage archive policy(weekly snapshot 一年 ~52 × N rows;Phase E)

---

## 十三、結論

本 session(2026-05-26)為**doctrine 純化之集大成 session** — 完成治權層 5 輪累進 + Phase C 工具落地,雙軸純化 doctrine(N-axis + T-axis)入憲 + 程式可用。

**對另機 AI(接手 protocol)**:

1. **讀本 handoff v12**(整體 context;standalone)
2. **clone repo + 切到 tag `v6.1.19-doctrine-pure-weekly-recommit-20260526`**(或直接 master)
3. **從舊機複製 .env**;**先 revoke GitHub PAT + 重生**(已在 chat 暴露)
4. **跑 import smoke test**;若 fail 補 OS dependencies
5. **驗證 DB 連線**(per §0.1 Quick Start);若另機與舊機共用 DB,§6.7 SSOT 預期返回 1,862 stocks
6. **第一優先**:用戶決定 Phase C-2 之 M1/M2/M3 sub-option(blocking weekly cron 啟動)
7. **第二優先**:Phase D-2 model_trainer / feature_store weekly mode 升版
8. **不可違**:§八 治權邊界禁令(尤其不可掛 cron until D 完成 / 不可寫 hardcode N prescription / 不可改 charter without auth)

**Cross-machine continuity 保證**:
- 全 8 commits + tag 已在 origin/master
- Charter + scripts + memory 均可從 git clone 取回
- DB 為 stateful(若同 host 則 state 自動 carry;若不同 host 需 re-sync)

---

**Handoff 作者**: Claude(Opus 4.7)
**Session ID**: 2026-05-26
**Charter base**: v6.1.0(+第十八-二十二輪 patch)
**HEAD commit**: `b923e4c`(已 push 至 origin/master)
**封存點 tag**: `v6.1.19-doctrine-pure-weekly-recommit-20260526`
**Status**: ✅ Session 完整 closure / 4 treaty rounds + Phase C tools landed / 跨機接續 ready
