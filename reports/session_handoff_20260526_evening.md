# Session Handoff — 2026-05-26 Evening(v6.1.19 完整封存 + 本機 DB 失調揭露)

- **時間**: 2026-05-26 晚(承接他機 v6.1.18.2 之後本機 session)
- **角色分工**:
  - 本機(Linux Codex):本 session 工作 — pure git pull + 設計研究 + analysis(無 DB 寫入)
  - 另一機(macOS Codex):2026-05-25 19:18 → 2026-05-26 00:57 完成 ROE 解鎖 + 三柱 evidence + production v0.7 commit
- **前次 handoff**: `reports/session_handoff_20260526.md`(另一機,07:02 早上 v6.1.18.2 之後)
- **檔案位階**: 永久追蹤(`.gitignore` `!reports/session_handoff_*.md` whitelist)

---

## 📌 一、Git 接續錨點

| 項目 | 值 |
|---|---|
| **Repo** | `https://github.com/tsaitsangchi/stock_backend` |
| **Branch** | `master` |
| **HEAD commit** | `59bfc8f` |
| **Latest tag(annotated)** | `v6.1.19-portfolio-sizer-v03-design-research-20260526` |
| **遠端同步狀態** | `master...origin/master` 完全同步(0 ahead / 0 behind)|

### v6.1.x tag 完整序列(本封存點為 v6.1.19)

```
v6.1.19-portfolio-sizer-v03-design-research-20260526  ← 本 session 封存
v6.1.18.2-session-handoff-cross-machine-20260526       (他機 morning handoff)
v6.1.18.1-session-roe-and-three-pillars-final-20260526 (他機 final)
v6.1.18-k-wave-evidence-and-l1-implementation
v6.1.17-pareto-evidence-and-universe-diff
v6.1.16-triple-task-bk-v07-launchd
v6.1.15-f-promotion-research-phase-a
v6.1.14-external-resource-verification-protocol-meta-rule
v6.1.13-roe-unlocked-via-sponsor-success
v6.1.12-roe-path-a-paywall-blocked-20260525
v6.1.11-session-handoff-cross-machine-20260525         (本機 morning closure)
v6.1.10-v06-dryrun-empirical-evidence-20260525
v6.1.9-rms-aligned-three-layer-closure-20260525
v6.1.8-audit-source-availability-v06-quad-treaty-closure-2026-05-22
```

### 跨機 clone 指令

```bash
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git checkout master           # 應自動到 59bfc8f
git log --oneline -3          # 確認 HEAD = 59bfc8f
git tag --sort=-v:refname | head -3
```

---

## 🚨 二、本機 DB 失調(最關鍵 — 需用戶 explicit 解決)

### 2.1 失調實證

| 維度 | 本機(Linux Codex) | 另一機(macOS production) |
|---|---|---|
| GitHub HEAD | `59bfc8f` ✅ 同步 | `59bfc8f` ✅ 同步 |
| **DB snapshots** | **v0.2 only**(2026-05-21)| v0.2 + v0.3 + **v0.7** |
| **TaiwanStockBalanceSheet 表** | **❌ 不存在** | ✅ 存在(8.25M rows / 2353 stocks)|
| `data_schema.py` | v2.21(含 BS DDL,已 pull)| v2.21 |
| `core_universe_builder.py` | **v0.8(讀 BS 表)已 pull** | v0.8 |
| **可否跑 builder?** | ❌ **NO**(會 fail — 因 BS 表不存在)| ✅ |

### 2.2 失調的後果

- 本機跑 `python3 scripts/core/core_universe_builder.py --commit --policy-version core_universe_policy_v0.7` 會 fail
- 任何 production-level sanity check 必須在另一機做
- 本機只能讀 reports/ + 跑 v0.2 audit + 純 charter / 純設計研究類工作

### 2.3 v0.6 vs v0.7 snapshot 之歷史釐清

| Snapshot | Policy | Builder | 何時 commit | DB 位置 |
|---|---|---|---|---|
| v0.2 | core_universe_policy_v0.2 | v0.2(六層 CoreScore)| 2026-05-21 commit | **本機 + 他機** |
| v0.3 | core_universe_policy_v0.3 | v0.3(FG GrossMargin)| 2026-05-22 commit | 他機 only |
| v0.4 / v0.5 | core_universe_policy_v0.4/5 | v0.5 / v0.6 | **從未 commit** | — |
| **v0.6** | core_universe_policy_v0.6 | v0.7.1(RMS)| **從未 commit;被 v0.7 替代** | — |
| **v0.7** | core_universe_policy_v0.7 | v0.8(ROE)| 2026-05-22 commit | **他機 only** ✨ production |

**v0.6 snapshot 已被歷史略過** — §14.7-BI ROE 解鎖(5/25 晚)後,production 直接從 v0.2 跳到 v0.7。v0.6 evidence 永久保留於 `reports/v06_dryrun_vs_v02_baseline_universe_diff_20260525.md`(74.2% overlap)。

### 2.4 3 個 sync 方向(用戶選一)

#### 方向甲:本機 sync 至 v0.7 production(~65 min,需 FINMIND_TOKEN sponsor)

```bash
cd /home/hugo/project/stock_backend
source .venv/bin/activate
# Step 1: user_info verify (per §14.7-AX(E) protocol)
python3 -c "
import os, requests
from dotenv import load_dotenv; load_dotenv('.env')
r = requests.get('https://api.finmindtrade.com/api/v4/user_info',
  headers={'Authorization': f'Bearer {os.environ[\"FINMIND_TOKEN\"]}'})
print(r.json())  # level >= 3 / sponsor tier
"
# Step 2: 建表
python3 scripts/core/data_schema.py --init --table TaiwanStockBalanceSheet
# Step 3: 跑 sync (tmux ~64 min)
tmux new -d -s bs_sync 'python3 scripts/core/sovereign_sync_engine.py --table TaiwanStockBalanceSheet --universe full > /tmp/bs_sync.log 2>&1'
# Step 4: 完成後 commit v0.7
python3 scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-21 \
  --policy-version core_universe_policy_v0.7 \
  --special-rebalance-reason "本機 sync 至 production v0.7 ROE 解鎖"
# Step 5: audit
python3 scripts/maintenance/audit_core_universe.py --policy-version core_universe_policy_v0.7
```

#### 方向乙:跳過 sync(接受本機 DB stranded)

- 不做任何 DB 操作
- 本機可繼續做純 charter / 設計研究 / pure git 工作
- 任何 production sanity check 在另一機跑

#### 方向丙:PostgreSQL dump/restore(~5-10 min,需另一機操作)

```bash
# === 另一機 ===
pg_dump -U stockuser -t '"TaiwanStockBalanceSheet"' \
        -t '"core_universe_snapshot"' \
        -t '"core_universe_membership"' \
        -t '"core_universe_scores"' \
        -t '"core_universe_policy"' \
        -t '"universe_revision_log"' \
        stock_db > /tmp/v07_snapshot.sql

# === scp 至本機 ===
scp other-machine:/tmp/v07_snapshot.sql /tmp/

# === 本機 restore ===
psql -U stockuser stock_db < /tmp/v07_snapshot.sql
```

---

## 📦 三、本 session 完成(2026-05-26 evening,~3-4 小時)

### Commits 軌跡(自他機 v6.1.18.2 起 1 個 commit)

| Commit | Tag | 內容 |
|---|---|---|
| `59a5e5b` | (他機 morning)| handoff 20260526(承接 v6.1.18.1)|
| **`59bfc8f`** | **v6.1.19** | **portfolio_sizer v0.3 設計研究(384 行/15 章/治權先行 Phase A)** |

### 本 session 完成的工作

1. **跨機 sync**(git fetch + git pull,9 個新 tags + 多個新 reports + audit_core_universe.py 升至 v0.7 policy)
2. **§14.7-BI / §14.7-BJ / §14.7-BK / §14.7-AX(E) 4 章完整 review**(2400+ 行 charter 內容)
3. **§0.2 八二法則 raw data 補強實證**(全市場 2767 stocks 之 Pareto 分析)
   - 揭露:Trading_money top 5% 拿 **74.46%**,但 CoreScore top 5% 只拿 **8.1%**
   - Gini coefficient: 0.2771(接近均勻)
   - LM sub-score Gini 0.303(最高)/ IF Gini 0.064(最低)
4. **CoreScore BY DESIGN 平均化結構性論證**(治權三分權責清晰)
   - L1 Universe:**必須平均化**(selection-oriented)
   - L2 Tactical:不直接 weighting
   - L3 Sizing:**必須集中右尾**(weighting-oriented;12 FAIL gate)
5. **portfolio_sizer v0.3 設計研究 Phase A**(`reports/portfolio_sizer_v03_design_research_20260526.md` 384 行 15 章)
   - DEFAULT_POLICY 升版(5 新 params)
   - 新增 G13/G14/G15 audit gates
   - ROE-weighted Pareto 公式 + 數值例
   - 證偽承諾 T_PS_v0.3-1〜5
   - **Root cause 誠實聲明**(v0.3 治標 / §10 治本)
6. **本機 DB 失調揭露**(本 handoff §二)
7. **v6.1.19 tag 封存 + push**(Phase A closure)

---

## 🏛️ 四、憲章治權層當前狀態(截至 v6.1.19)

### 本 session 新增 reports(設計研究類)

- `reports/portfolio_sizer_v03_design_research_20260526.md`(384 行/15 章)

### 既有 reports(從他機 pull,v6.1.12~18.2 累積)

- `roe_unlock_path_a_paywall_blocked_20260525.md`(199 行)
- `roe_unlock_success_evidence_20260525.md`(151 行)
- `f_promotion_to_t1_decision_research_20260525.md`(284 行)
- `pareto_law_evidence_and_v07_universe_diff_20260526.md`(198 行)
- `k_wave_evidence_and_l1_implementation_20260526.md`(203 行)
- `session_handoff_20260526.md`(301 行 — 他機 morning handoff)
- `core_universe_audit_20260525_2203.md` + `_20260526_0013.md`
- `daily_sync_and_audit.sh` + `weekly_audit3.sh`(launchd 自動化)
- `deploy/launchd/*.plist`(macOS automation)

### 憲章 §14.7 子節進度(截至 §14.7-BK)

| 子節 | 主題 | 狀態 |
|---|---|---|
| §14.7-AY ~ BH | (本機 v6.1.10 之前已完成) | ✅ |
| **§14.7-BI** | **ROE 解鎖 SUCCESS via sponsor (V 64% → 73%)** | ✅ |
| **§14.7-BJ** | **ROE Path A paywall blocked (歷史記述)** | ✅ |
| **§14.7-BK** | **F/IF 升 §0.1 T1 Phase A 治權預備** | ✅ |
| **§14.7-AX(E)** | **外部資源驗證 protocol (元規則第 9 條)** | ✅ |

### 待入憲(下個 session)

- **§9.2-I (v0.3 補強條款)** 或 **§14.7-BL (v0.3 治權升版預備)** ← T24
- 可能 §14.7-BM (v0.3 落地完成記述) ← T26 之後

---

## 🛠️ 五、程式層當前版本(截至 v6.1.19)

| 模組 | 版本 | 治權對齊 |
|---|---|---|
| `data_schema.py` | **v2.21** | + TaiwanStockBalanceSheet DDL |
| `core_universe_builder.py` | **v0.8** | + BS SQL + `_roe_score()` + ROE sub-score |
| `audit_core_universe.py` | **v0.2** | + v0.7 policy entry + v0.8_roe_unlocked score_scope |
| `feature_store_builder.py` | v0.5 | (不動 — 早已對齊 §9.9 RMS) |
| `portfolio_sizer.py` | **v0.2**(97.5% 合規)| ⏸ 待 v0.3 升版 |
| `sovereign_sync_engine.py` | v1.22 | (不動) |
| `path_setup.py` | v4.47+ | macOS/Linux symlink 跨平台對齊 |

### 本 session 新增工具

- 無新工具(設計研究類為主)

---

## 📋 六、Unfinished items + 4 phases 接力指南

### portfolio_sizer v0.3 完成 phases(本 session 完成 Phase A,Phase B-D 待續)

| Phase | 內容 | 狀態 | 對應 Task |
|---|---|---|---|
| **Phase A** | 設計研究 + commit + push + tag v6.1.19 | ✅ 本 session 完成 | T23 |
| **Phase B** | 入憲 §9.2-I 或 §14.7-BL(charter 編輯) | ⏸ 待下個 session | T24 |
| **Phase C** | portfolio_sizer.py v0.2 → v0.3 程式落地 | ⏸ 待下個 session | T25 |
| **Phase D** | smoke test + commit + tag v6.1.20 | ⏸ 待下個 session | T26 |

### 6 個 critical decisions pending(優先級高 → 低)

| # | Issue | 阻塞於 | 評估 |
|---|---|---|---|
| 1 | **本機 DB 失調(v0.2 vs production v0.7)** | 等用戶選甲/乙/丙 | 立即決定 |
| 2 | portfolio_sizer v0.3 Phase B-D | 下個 session T24-T26 | ~2-3 小時 |
| 3 | §10 model_trainer 治本 | v6.2.0 軌道 | ~2-3 週 |
| 4 | F 升 T1 Phase B-D | 等 §10 | 等 §10 |
| 5 | 電子業 86% 集中違 §0.2-A 禁令 #3 | 等 portfolio_sizer + §10 | 等補救 |
| 6 | 金融業 ROE 對齊(國泰/中信無 ROE 值)| BS 對金融業 EAOP 對應特殊 case | 等 §14.7-BL 候選 |

### 重新評估時點

- 2026-06-13(v6.1.1 production-current h20 gate 解除)
- 2026-06-24(FinMind sponsor 到期,需續訂)
- v6.2.0(預估 Q3,§10 model_trainer + §9.2 portfolio_sizer 完整 v0.1)

---

## 🌐 七、跨機環境前置(若新電腦 setup)

### Step 1 — OS 原生依賴

**macOS**: `brew install libomp postgresql@17`
**Linux**: `sudo apt-get install -y libgomp1 libpq-dev`

### Step 2 — Python 環境

```bash
cd stock_backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅')"
```

### Step 3 — .env 跨平台路徑對齊

```env
PROJECT_ROOT=/home/<user>/project/stock_backend   # Linux
# 或 /Users/<user>/project/stock_backend          # macOS
DB_HOST=...
DB_PORT=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
FINMIND_TOKEN=eyJ0eXAi...   # sponsor 到 2026-06-24
FRED_API_KEY=...
GITHUB_TOKEN=...            # 若需 push
```

### Step 4 — DB 同步(per 本 handoff §二)

選甲 / 乙 / 丙 之一。

---

## 🎯 八、新 session 建議起手式

```
1. 確認 git pull 同步至 59bfc8f(v6.1.19)
2. 讀本 handoff(reports/session_handoff_20260526_evening.md)
3. 決定本機 DB 是否要 sync(per §二 三方向)
4. 開新主題(對應 phase ABCD 順序):
   a. portfolio_sizer v0.3 Phase B(入憲 §9.2-I 或 §14.7-BL)
   b. portfolio_sizer v0.3 Phase C(程式落地 v0.2 → v0.3)
   c. portfolio_sizer v0.3 Phase D(smoke + commit + tag v6.1.20)
   d. 其他(例:§10 model_trainer 設計研究 / ROE 金融業特殊 case 研究)
```

---

## 📊 九、本 session 最終結算

| 指標 | 數值 |
|---|---|
| 本 session commits | **1**(`59bfc8f` 設計研究)|
| 本 session 新 tags | **1**(`v6.1.19`)|
| pull 自他機 commits | 1(`59a5e5b` morning handoff)|
| pull 自他機 tags | **9**(v6.1.12 ~ v6.1.18.2)|
| 設計研究產出 | 1 份(384 行/15 章 portfolio_sizer v0.3)|
| 補強 evidence 產出 | 1 份(§0.2 八二法則全市場 Pareto 分析,inline 文字)|
| 用戶 anchor echoes 回應 | 7 次(包含 §6 八二法則 / CoreScore BY DESIGN / v0.3 升版 / v0.6 snapshot / 完整封存)|
| 寫入 DB | **0**(純讀;DB unchanged) |
| §0.0-G 跑通計數 | 不另計(設計研究屬 Level 1 Phase A) |

---

## 🔚 結語

本 session 為**跨機接續 + Phase A 治權先行 closure**:
1. 從他機 v6.1.18.2 sync 至本機
2. 完成 portfolio_sizer v0.3 設計研究(治標 / Root cause 揭露)
3. 揭露本機 DB 失調(待用戶決策 sync 方向)
4. 封存 v6.1.19 + push

**最關鍵成就**: portfolio_sizer v0.3 設計研究完整 closure,含治權三分權責論證 + §0.2 八二法則資料層補強 + ROE-weighted Pareto 公式設計。

**最關鍵未解**: 本機 DB 失調(等用戶決定甲/乙/丙)+ portfolio_sizer v0.3 Phase B-D。

跨機接續或本機繼續工作,皆可從本 handoff + v6.1.19 tag 開始。

---

*Handoff generated 2026-05-26 evening by Claude Sonnet 4.7 session*
*Session covered: cross-machine sync + portfolio_sizer v0.3 design research (Phase A)*
*HEAD: 59bfc8f / Tag: v6.1.19-portfolio-sizer-v03-design-research-20260526*
*下個 session: T24 入憲 → T25 程式 → T26 smoke + v6.1.20 tag*
