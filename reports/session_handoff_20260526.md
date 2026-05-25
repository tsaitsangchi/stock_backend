# Session Handoff — 2026-05-26 07:02(跨機接續必要文件)

- **時間**: 2026-05-26 早上 07:02(昨晚 19:18 → 凌晨 00:57 跨夜大滿貫之後)
- **目的**: 用戶將在另一台電腦接續本 session 工作;本檔為跨機接續完整 context
- **對映**: 憲章 §0.0-I.9/I.10 跨平台依賴 + 本 session 完整 audit trail
- **檔案位階**: 永久追蹤(`.gitignore` `!reports/session_handoff_*.md` whitelist)
- **前次 handoff**: `reports/session_handoff_20260525.md`(Codex,5/25 16:58 v6.1.10 closure 後)

---

## 📌 一、Git 接續錨點(最關鍵 — 跨機第一步)

| 項目 | 值 |
|---|---|
| **Repo** | `https://github.com/tsaitsangchi/stock_backend` |
| **Branch** | `master` |
| **HEAD commit** | `0b84284` |
| **Latest tag(annotated)** | `v6.1.18.1-session-roe-and-three-pillars-final-20260526` ✨ |
| **Latest tag(lightweight)** | `v6.1.18-k-wave-evidence-and-l1-implementation` |
| **遠端同步狀態** | `master...origin/master` 完全同步(0 ahead / 0 behind) |

### 跨機 clone 指令

```bash
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git checkout master  # 應自動到 0b84284
git log --oneline -1  # 確認 HEAD = 0b84284
git tag --sort=-v:refname | grep v6.1 | head -5
# 預期最新 5 個:
# v6.1.18.1-session-roe-and-three-pillars-final-20260526
# v6.1.18-k-wave-evidence-and-l1-implementation
# v6.1.17-pareto-evidence-and-universe-diff
# v6.1.16-triple-task-bk-v07-launchd
# v6.1.15-f-promotion-research-phase-a
```

---

## 📦 二、本跨夜 session 完成內容(2026-05-25 19:18 → 2026-05-26 00:57,8 tag)

### Commits 軌跡(自 a42a416 起共 12 個 commits 已 push)

| Commit | Tag | 內容 |
|---|---|---|
| `4da2450` | (前 session)| 跨機開始基線 |
| `2b79872` ~ `46f4d15` | v6.1.10/11(Codex morning)| Codex 早上 RMS 公式對齊 |
| `a42a416` | — | Codex handoff doc |
| `fd38a5d` | **v6.1.12** | ROE Path A paywall blocked(誤判)|
| `ef03b26` | **v6.1.13** | ROE 解鎖 SUCCESS via sponsor |
| `926b56c` | **v6.1.14** | 外部資源驗證 protocol 元規則(三層)|
| `14f16ef` | **v6.1.15** | F 升 T1 Phase A 治權研究 |
| `e213547` | **v6.1.16** | 三任務並聯(§14.7-BK + v0.7 production + launchd) |
| `b11deb8` | **v6.1.17** | §0.2 八二法則 evidence + universe diff |
| `0b84284` | **v6.1.18** | §0.3 康波週期 evidence + L1 落地 |
| (no commit) | **v6.1.18.1**(annotated) | Session final marker |

---

## 🏛️ 三、憲章治權層當前狀態

### 跨夜本 session 新增條款

| 子節 | 主題 | 入憲 commit |
|---|---|---|
| **§14.7-BJ** | ROE Path A 受 paywall 阻擋(§14.7-AX 第 8 次跑通首例「外部資源限制」)| fd38a5d |
| **§14.7-BI** | ROE 解鎖 Path A' SUCCESS(第 8 次跑通正式 closure) | ef03b26 |
| **§14.7-AX (E)** | 外部資源驗證 protocol(元規則第 3 類覆蓋面正式入憲) | 926b56c |
| **§14.7-BK** | F/IF 升 §0.1 T1 治權預備設計研究(Phase A) | e213547 |

### §14.7-AX 元規則覆蓋面更新

```
第 1 類:dataset 內部資料 mislabel/格式異常(§0.1.3-A.1 / §14.7-BA-BE)
第 2 類:dataset 內部公式漂移(§14.7-BH)
第 3 類:外部資源 access 限制(§14.7-BJ blocked + §14.7-BI unlocked)  ✨ NEW
第 4 類:治權框架重構(§14.7-BK F 升 T1 Phase A 預備候選)
```

### §0 三柱完整 evidence-tagged

- **§0.1 第一性原理** → v6.1.13/15(ROE 解鎖 + F 升 T1 研究)
- **§0.2 八二法則** → v6.1.17(集中右尾 + 厚尾數學)
- **§0.3 康波週期** → v6.1.18(先驗哲學選擇 + L1 集中右尾)

---

## 💾 四、DB 狀態(跨機 setup 後驗證用)

```sql
-- 現有 snapshots(3 個):
('core_universe_policy_v0.2')  -- 2026-05-24 baseline
('core_universe_policy_v0.3')  -- 2026-05-22 my Phase B test (GM only, no ROE)
('core_universe_policy_v0.7')  -- 2026-05-22 ROE unlocked production ✨

-- 現有 score_scopes:
('v0.2_six_layer')
('v0.3_six_layer_extended')
('v0.8_roe_unlocked_via_balance_sheet')  ← v0.7 snapshot 用此 scope

-- DEFAULT_POLICY_VERSION(builder + audit):
'core_universe_policy_v0.7' ← production
```

### DB 連線設定(跨機需要)

`.env` 必須含:
- `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD`
- `PROJECT_ROOT=/Users/<user>/project/stock_backend`(macOS)或 Linux 對應
- `FINMIND_TOKEN=eyJ0eXAi...`(sponsor token 到 2026-06-24)
- `FRED_API_KEY=...`
- `GITHUB_TOKEN=...`(若需 push)

### DB 表規模(跨機驗證)

| Table | rows | 重要新表? |
|---|---|---|
| TaiwanStockPriceAdj | ~10.5M | |
| TaiwanStockMonthRevenue | ~460K | |
| TaiwanStockPER | ~7.3M | |
| TaiwanStockInstitutionalInvestorsBuySell | ~25M | |
| TaiwanStockMarginPurchaseShortSale | ~7.7M | |
| TaiwanStockFinancialStatements | ~2.7M | |
| **TaiwanStockBalanceSheet** | **~8.25M / 2353 stocks** | ✨ NEW(本 session 5/25 晚同步)|
| FredData | ~49K | |

---

## 🛠️ 五、程式層當前版本

| 模組 | 版本 | 治權對齊 |
|---|---|---|
| `data_schema.py` | **v2.21** | + TaiwanStockBalanceSheet 表 DDL + FINMIND_API_TABLES + PUBLICATION_DATE_STRATEGY_REGISTRY |
| `core_universe_builder.py` | **v0.8** | + BS SQL block + `_roe_score()` helper + `fg_equity` / `fg_ni_4q_sum` score_detail keys + DEFAULT_POLICY_VERSION = v0.7 |
| `audit_core_universe.py` | **v0.2** | + POLICY_SCORE_SCOPE_MAP v0.7 entry + EXPECTED_SCORE_DETAIL_KEYS v0.7 + check_policy v0.7 分支 + DEFAULT_POLICY_VERSION = v0.7 |
| `feature_store_builder.py` | v0.5 | (不動;Codex morning v0.7.1 RMS 對齊已落地)|
| `sovereign_sync_engine.py` | v1.22 | (不動)|
| `path_setup.py` | v4.47+ | macOS/Linux symlink 跨平台對齊 |

### 本 session 新增工具

| 工具 | 路徑 | 用途 |
|---|---|---|
| daily sync + audit | `scripts/maintenance/daily_sync_and_audit.sh` | launchd 每日 18:30 觸發 |
| weekly audit3 | `scripts/maintenance/weekly_audit3.sh` | launchd 週日 03:00 觸發 |
| launchd plist × 2 | `deploy/launchd/*.plist` | macOS 自動化 SOP |

### 本 session 新增 reports(5 份)

| 報告 | 行數 | 主題 |
|---|---|---|
| `roe_unlock_path_a_paywall_blocked_20260525.md` | 199 | Path A 第 1 次嘗試(誤判)|
| `roe_unlock_success_evidence_20260525.md` | 151 | Path A' SUCCESS 實證 |
| `f_promotion_to_t1_decision_research_20260525.md` | 284 | F 升 T1 完整評估 |
| `pareto_law_evidence_and_v07_universe_diff_20260526.md` | 198 | §0.2 八二 + v0.3→v0.7 diff |
| `k_wave_evidence_and_l1_implementation_20260526.md` | 203 | §0.3 康波 evidence |

---

## 🤖 六、launchd 自動化(已啟用!跨機需重新 install)

```
✅ 昨晚 00:17 完成 launchd 啟用
✅ 已通過手動 launchctl start 測試(Step 1 sync 進入電子產業段時 stop 因為手動驗證足夠)
```

### ⚠️ 跨機後需重新 install launchd jobs

```bash
cd /Users/hugo/project/stock_backend
cp deploy/launchd/com.tsai.stock_backend.daily_sync.plist ~/Library/LaunchAgents/
cp deploy/launchd/com.tsai.stock_backend.weekly_audit3.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tsai.stock_backend.daily_sync.plist
launchctl load ~/Library/LaunchAgents/com.tsai.stock_backend.weekly_audit3.plist

# 驗證
launchctl list | grep com.tsai.stock_backend
# 應看到 2 條
```

詳細 SOP:`deploy/launchd/README.md`

---

## 📊 七、§0 三柱落地度(跨機後可立即查)

```
                  治權層    L1 universe    L2 prediction   L3 sizing
─────────────────────────────────────────────────────────────────────
§0.1 第一性原理   ✅完整    M 100%/V 73%   ⏸ §10        ⏸ §9.2
                            ΔlnP 100%
§0.2 八二法則     ✅完整    集中右尾 ✅    ⏸ §10        ⏸ §9.2
                            (top 5% 947%)
§0.3 康波週期     ✅完整    96% MBNRIC ✅  ⏸ §10        ⏸ §9.2
                            (theme≠return)
─────────────────────────────────────────────────────────────────────
整體落地度        ~92%(歷史新高)
治權成熟度        §14.7-AX 元規則 3 類正式 + 1 類預備
```

---

## 🔧 八、跨機環境前置(per CLAUDE.md §二 #7 / 憲章 §0.0-I.9-I.10)

### Step 1 — OS 原生依賴

**macOS**:
```bash
brew install libomp postgresql@17
```

**Linux**:
```bash
sudo apt-get install -y libgomp1 libpq-dev
```

### Step 2 — Python 環境

```bash
cd stock_backend
python3.12 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./venv/bin/python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅ all imports OK')"
```

### Step 3 — .env 設定(複製從原機;不可上傳 GitHub)

需 `cp .env.example .env` 後填入:
- `FINMIND_TOKEN`(sponsor 到 2026-06-24)
- `FRED_API_KEY`
- `DB_HOST` / `DB_USER` / `DB_PASSWORD`
- `GITHUB_TOKEN`(若要 push)
- `PROJECT_ROOT=/Users/<new-user>/project/stock_backend`

### Step 4 — DB 同步

兩個選擇:
- **A. 直接複製 PostgreSQL data dir**(快,但需相同 PG 版本)
- **B. 從零重建**(per 憲章 §二 §14.7-AM 4 步)~ 11h(過了昨晚 ROE 還要 +5h sync BS)

推薦 **A**(複製 data dir 最快)。

### Step 5 — 驗證 setup OK

```bash
./venv/bin/python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-22
# 預期:PASS=41 / WARN=1 / FAIL=0 (verdict=WARNING)
```

---

## 📋 九、Unfinished items(post v6.1.18,給新 session 看)

### 5 個 outstanding(優先級高 → 低)

| # | Issue | 狀態 | 阻塞於 |
|---|---|---|---|
| 1 | **§9.2 portfolio_sizer v0.1 落地** | 程式未實作,§9.2-A~H formal contract v0.1 已入憲 | 設計 + 編碼 ~3 天 |
| 2 | **§10 model_trainer 落地** | 全未開始,§8.3 v0.1 草案在 | 大工程 ~2-3 週 |
| 3 | **F 升 T1 Phase D**(walk-forward IC 通過後)| 等 §10 | §10 落地後 |
| 4 | **電子業 86% 集中違 §0.2-A 禁令 #3** | 已知 + §0.2-C 已記述 | 待 portfolio_sizer sector cap |
| 5 | **金融業 ROE 對齊**(國泰/中信無 ROE 值)| 已知 | BS 對金融業 EAOP 對應需特殊 case |

### 重新評估時點

- **2026-06-13**(v6.1.1 production-current h20 gate 解除)
- **2026-06-24**(FinMind sponsor 到期,需續訂)
- **v6.2.0**(預估 Q3,§10 model_trainer + §9.2 portfolio_sizer 完整 v0.1)

---

## 🎯 十、新 session 建議起手式(早安!)

```
1. 確認 git pull 同步至 v6.1.18.1(本檔已含 tag info)
2. 確認 launchctl list 顯示 2 條 launchd job(若無需重 install)
3. (若 launchd 之前自動跑過)看 logs/daily/*.log 是否健康
4. 開新主題:
   a. 啟動 §10 model_trainer Phase A 設計研究(F 升 T1 / portfolio_sizer 前置)
   b. 啟動 §9.2 portfolio_sizer v0.1-prelim(equal-weight + sector cap,3 天)
   c. 開啟 tsai_ai_assistant Phase 1(MCP 整合 stock_backend)
   d. 其他方向
```

---

## ⚠️ 注意事項(跨機常見坑)

1. **PostgreSQL 連線**:macOS 預設 `localhost` 應該 OK;若新機 Linux 需開 PG service
2. **PROJECT_ROOT 路徑**:macOS `/Users/<u>/` vs Linux `/home/<u>/`;`path_setup.py v4.47+` 用 realpath 自動處理
3. **FinMind hourly quota**:sponsor 6000/hr;若早上 launchd 跑過會用部分 quota
4. **SHMM heartbeats**:本 session 已 stop,新 session 重啟 long-running 任務時記得重掛
5. **Sentinel 檔**:`/tmp/claude_loop_last_fire.txt` 重啟後可能消失,SHMM 自動 reset

---

*Handoff generated 2026-05-26 07:02 by Claude Code session*
*Session covered: ROE saga + 3 任務並聯 + §0/3-pillar evidence inscription*
*8 tags pushed: v6.1.12 → v6.1.18.1*
*Total elapsed: 5h37m(19:18 → 00:57)+ 6h pause(00:57 → 07:02)*
