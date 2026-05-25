# Session Handoff — 2026-05-25 v6.1.10 closure(跨機接續必要文件)

- **時間**: 2026-05-25 結束時
- **目的**: 用戶將在另一台電腦接續本 session 工作;本檔為跨機接續完整 context
- **對映**: 憲章 §0.0-I.9/I.10 跨平台依賴 + 本 session 完整 audit trail
- **檔案位階**: 永久追蹤(`.gitignore` `!reports/session_handoff_*.md` whitelist)

---

## 📌 一、Git 接續錨點(最關鍵 — 跨機第一步)

| 項目 | 值 |
|---|---|
| **Repo** | `https://github.com/tsaitsangchi/stock_backend` |
| **Branch** | `master` |
| **HEAD commit** | `46f4d15` |
| **Latest tag** | `v6.1.10-v06-dryrun-empirical-evidence-20260525` |
| **Previous tag** | `v6.1.9-rms-aligned-three-layer-closure-20260525` |
| **遠端同步狀態** | `master...origin/master` 完全同步(0 ahead / 0 behind)|

**跨機 clone 指令**:

```bash
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git checkout master  # 應自動到 46f4d15
git log --oneline -1  # 確認 HEAD = 46f4d15
git tag --sort=-v:refname | head -3  # 確認三個最新 tag
```

---

## 📦 二、本 session 完成內容(16 commits / 2 tags)

### Commits 軌跡(自 `2b79872` 起共 16 個 commits 已 push 至 origin)

| Commit | Phase | 內容 |
|---|---|---|
| `2b79872` | (前 session)| 本 session 起點 baseline |
| (12 commits) | v6.1.0-patch 第七~十一輪 | V 補強(FG 11 sub)/ F proxy(IF 12 sub)/ ΔlnP 凸性對齊 builder v0.7 |
| `88cc617` | v6.1.0-patch 第十二輪 | audit_core_universe v0.1 → v0.2 + P1 v0.1 RMS vs STDDEV ablation closure |
| `37bc687` | v6.1.0-patch 第十三輪 | charter §14.7-BH/§9.10/§14.7-BG + builder v0.7 → v0.7.1 + audit RMS 配套(三層治權同步升版)|
| `9ea2ba0` | Archive | 4 audit reports(v0.2 smoke + 5/22 source_availability + 早 5/25 compliance)|
| `46f4d15` | **本 session 結束** | dry-run evidence + compare_script + .gitignore whitelist |

### Tag 序列(最新 3 個)

```
v6.1.10-v06-dryrun-empirical-evidence-20260525     ← 當前 HEAD 對齊
v6.1.9-rms-aligned-three-layer-closure-20260525    ← 三層治權同步升版 closure
v6.1.8-audit-source-availability-v06-quad-treaty-closure-2026-05-22
```

---

## 🏛️ 三、憲章治權層當前狀態

### v6.1.0-patch 修訂歷程已新增 13 條 entries(2026-05-25 之內)

| 輪次 | 子節 / 主題 |
|---|---|
| 第一輪 | §0.1.3-B DB Field Bottom-up 反推治權閉環 + §14.7-AY 新建 |
| 第二輪 | §8.5 第 9 條 Publication-date Discipline 起草入憲 + §14.7-AZ 新建 |
| 第三輪 | §8.5-9 FredData 追溯 strict → transitional + §14.7-BA 新建 |
| 第四輪 | data_schema v2.19 同步追溯修正 FRED strategy + §14.7-BB 新建 |
| 第五輪 | builder v0.3 → v0.4 + feature_store v0.4 + 配套 |
| 第六輪 | §14.7-BC V 補強 Phase C/D + FinStmt 治權預備 |
| 第七輪 | builder v0.4 → v0.5(FG 11 sub-scores 落地)|
| 第八輪 | §14.7-BE「資料現實裁決」第 5 次跑通 + builder v0.5.1 |
| 第九輪 | §14.7-BF F proxy 補強 Phase F.1-F.3 治權預備 |
| 第十輪 | builder v0.5.1 → v0.6(IF 12 sub-scores 落地)|
| 第十一輪 | §14.7-BG VC 凸性對齊 + §9.10 起草 + builder v0.6 → v0.7 |
| **第十二輪** | **audit_core_universe v0.1 → v0.2(配套 v0.7 builder)** |
| **第十三輪** | **§14.7-BH P1 v0.1 公式對齊 ablation + §9.10 升正式條文 + builder v0.7 → v0.7.1 + audit v0.2 RMS 配套** |

### §14.7 子節進度(到 §14.7-BH)

| 子節 | 主題 | 狀態 |
|---|---|---|
| §14.7-AY | §0.1.3-B 治權閉環(V/F/M/ΔlnP/時間單向性 完整論證)| ✅ ACTIVE |
| §14.7-AZ | §8.5 第 9 條 Publication-date Discipline 強制契約 | ✅ ACTIVE |
| §14.7-BA | §8.5-9.2 5-enforcement strategy(native/strict/hardcoded/transitional/infra)| ✅ ACTIVE |
| §14.7-BB | data_schema v2.19 FRED strategy 追溯修正 | ✅ ACTIVE |
| §14.7-BC | V 補強 Phase C/D + FinStmt(FG 11 sub-scores)| ✅ ACTIVE |
| §14.7-BD | 「資料現實裁決」第 4 次跑通(Dividend 民國年)| ✅ ACTIVE |
| §14.7-BE | 「資料現實裁決」第 5 次跑通(Dividend 4 cols sunset)| ✅ ACTIVE |
| §14.7-BF | F proxy 補強 Phase F.1-F.3(IF 12 sub-scores)| ✅ ACTIVE |
| §14.7-BG | VC 凸性對齊 + §9.10 起草(raw-first 路徑首例)| ✅ ACTIVE(STDDEV 補註 fast-track 試錯)|
| **§14.7-BH** | **P1 v0.1 公式對齊 ablation + §9.10 升正式條文** | ✅ **ACTIVE(最新)** |

### §14.7-AX 治權元規則(資料/公式層揭露驅動治權升版)

**累計 7 次跑通**:
1. ROE mislabel(§0.1.3-A.1)— v0.3 builder dropped
2. Publication-date strict gate — §8.5-9 + §14.7-BA
3. FRED vintage gate 100% loss — §14.7-BB 追溯
4. Dividend 民國年格式 — §14.7-BD
5. Dividend 4 cols sunset — §14.7-BE
6. F proxy r 矩陣分析 — §14.7-BF
7. **§9.9 RMS vs §14.7-BG/§9.10 STDDEV 公式不一致**(本 session 第 7 次;**首例「公式層揭露」**)

### §9.10 正式契約(本 session 升版)

由 §14.7-BG 起草 → §14.7-BH 升正式條文:

- §9.10-A Identity: `scripts/core/core_universe_builder.py` VC sub-score v0.7.1
- §9.10-B 強制輸入: `TaiwanStockPriceAdj.close`(raw close;§8.5-9.2 native_aligned)
- §9.10-C 強制輸出: `volatility_control_score` + `vc_convexity_60d` / `vc_upside_rms_60d` / `vc_downside_rms_60d` / `vc_cc_sigma_60d`
- §9.10-D 公式: `convexity = upside_RMS − downside_RMS`(annualized;60d window)
  - `upside_RMS = SQRT(AVG(log_return²) FILTER (WHERE log_return > 0)) × √252`
  - 5 階梯 score map: > +0.10 → 95 / > +0.05 → 85 / > 0 → 75 / > -0.05 → 60 / > -0.10 → 40 / ≤ -0.10 → 20
- §9.10-E 強制 Policy: 60d / SQL window function / √252 / 替代 cv_close
- §9.10-F 強制 FAIL Gate: 窗口 ≠ 60 FAILED;n_obs < 20 → 50
- §9.10-G 與 §9.9 對齊聲明: §9.10 為 §9.9 之 builder-layer 落地對齊

---

## 💾 四、DB 狀態(剛實證查詢)

```sql
-- Existing snapshots:
('core_universe_policy_v0.2',)  -- 唯一已建 snapshot

-- Defined policies:
('core_universe_policy_v0.2',)  -- 唯一已定義 policy

-- Existing score_scopes:
('v0.2_six_layer',)             -- 唯一已寫入 scope
```

**關鍵**: builder v0.7.1 + 三層治權升版 + 4 個新 policy(v0.3/v0.4/v0.5/v0.6)都在程式層就緒,但 **DB 中只有 v0.2 snapshot**。v0.6 dry-run 已跑出名單(74.2% overlap),但**未 commit** 至 DB(避免 over-fit 風險)。

### DB 連線設定(跨機接續需要)

`.env` 必須含:
- `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD`
- `PROJECT_ROOT=/home/hugo/project/stock_backend`(或新機之絕對路徑)

**跨平台路徑提醒**(per 憲章 §0.0-I.10):
- macOS: `/Users/<user>/project/stock_backend`(`/home/<user>` 為 symlink)
- Linux: `/home/<user>/project/stock_backend`
- `path_setup.py v4.47+` 用 `os.path.realpath()` 自動處理

### DB 主要表格規模(2026-05-21 as_of)

| Table | rows | date_range | stock_id 數 |
|---|---:|---|---:|
| TaiwanStockInfo | 2,767 | 2026-04-21 to 2026-05-21 | 2,767 |
| TaiwanStockPriceAdj | 10,481,069 | 1992-01-04 to 2026-05-21 | 2,766 |
| TaiwanStockMonthRevenue | 459,383 | 2002-02-01 to 2026-05-01 | 2,357 |
| TaiwanStockPER | 7,328,884 | 2005-09-02 to 2026-05-21 | 2,012 |
| TaiwanStockInstitutionalInvestorsBuySell | 24,963,205 | 2005-01-03 to 2026-05-21 | 2,754 |
| TaiwanStockMarginPurchaseShortSale | 7,696,032 | 2001-01-05 to 2026-05-21 | 2,222 |
| TaiwanStockFinancialStatements | 2,656,263 | 1990-03-31 to 2026-03-31 | 2,342 |
| FredData | 48,876 | 1948-01-01 to 2026-05-21 | series_id=4 |

---

## 🛠️ 五、程式層當前版本

| 模組 | 版本 | 治權對齊 |
|---|---|---|
| `data_schema.py` | **v2.20** | PUBLICATION_DATE_STRATEGY_REGISTRY + build_publication_date_gate SSOT helper |
| `core_universe_builder.py` | **v0.7.1** | 6 維 CoreScore + FG 11 sub + IF 12 sub + **VC RMS 凸性** |
| `feature_store_builder.py` | **v0.5** | §9.9 RMS upside/downside_volatility_60d(早已對齊;不動)|
| `audit_core_universe.py` | **v0.2** | POLICY_SCORE_SCOPE_MAP + EXPECTED_SCORE_DETAIL_KEYS + check_score_detail_keys |
| `db_utils.py` | v2.48 | data_audit_log ON CONFLICT DO NOTHING |
| `path_setup.py` | v4.47+ | macOS/Linux symlink 跨平台對齊 |

### 新增工具(本 session)

| 工具 | 路徑 | 用途 |
|---|---|---|
| ablation_rms_vs_stddev | `scripts/maintenance/ablation_rms_vs_stddev_20260525.py v0.1` | 全市場 RMS vs STDDEV 公式對照 |
| compare_v06_dryrun | `scripts/maintenance/compare_v06_dryrun_vs_v02_baseline_20260525.py v0.1` | v0.7.1 dry-run vs v0.2 baseline universe 對照 |

---

## 📊 六、V/F/ΔlnP 三軸落地對齊度

| §0.1 T1 元素 | 落地組件 | 動員 cols | 治權狀態 | DB 實證 |
|---|---|---|---|---|
| **M** 流動性質量 | LM 25% | 3/3 = 100% | §9.4 第 7 條 horizon=30 ✅ | ✅ v0.2 snapshot |
| **V** 內在價值密度 | FG 20% | 14/22 = **64%** | §14.7-BC/BE + §6.3 第 4 條 ✅ | ⏸ v0.4 snapshot 未建 |
| **F** 機構/外生力 | IF 10% | 22/25 = **88%** | §14.7-BF + §6.3 第 6 條 ✅ | ⏸ v0.5 snapshot 未建 |
| **ΔlnP** 價格訊號 | VC 5% + LM/IF/TR | **RMS 對齊 100%** | §9.9/§9.10 + §14.7-BG/BH ✅ | ⏸ v0.6 snapshot 未建(dry-run only)|
| **時間單向性** | publication_date | 13 datasets × 5 strategy | §8.5 第 9 條 ✅ | ✅ SSOT helper 落地 |

---

## 📄 七、重要 evidence 報告位置

### Ablation Evidence Chain(§14.7-BH)
- `reports/p1_v01_rms_vs_stddev_ablation_evidence_20260525_1604.md` — Evidence + 裁決(2688 stocks ρ_score 0.8816)
- `reports/p1_v01_rms_vs_stddev_ablation_data_20260525_1604.json` — Full data

### Dry-run Empirical Evidence(v6.1.10)
- **`reports/v06_dryrun_vs_v02_baseline_universe_diff_20260525.md`** — 11 章完整分析(74.2% overlap / 6 structural patterns)

### 設計研究(本 session 累積)
- `reports/audit_log_write_safe_design_research_20260525.md`
- `reports/publication_date_discipline_design_research_20260525.md`(11 章 + 3 附錄)
- `reports/first_principles_field_inventory_research_20260525.md`(11 章 + 3 附錄)
- `reports/v_augmentation_phase_cd_design_research_20260525.md`(13 章 + 3 附錄)
- `reports/f_proxy_augmentation_phase_f_design_research_20260525.md`(15 章 + 3 附錄)
- `reports/vc_convexity_alignment_design_research_20260525.md`

### 憲章
- `reports/系統架構大憲章_v6.1.0.md`(8650+ 行;v6.1.0-patch 第十三輪 entries 已加;§14.7-BH 已新建)

---

## ⏸ 八、Unfinished items + 下一步建議

### 5 個 outstanding(post v6.1.10)

| # | Issue | 性質 | 動員度影響 |
|---|---|---|---|
| 1 | **v0.6 policy snapshot 仍未在 DB 產出** | dry-run 已完成,commit 未做 | 0%(程式就緒,差「按 enter」)|
| 2 | **ROE 仍 None 占位**(§0.1.3-A.1) | raw schema mislabel 鎖死 | V 64% → ~73%(若解開) |
| 3 | **F 是否升 §0.1 T1 第 5 元素治權位階** | IF 88% > V 64% 動員度 | 治權框架重構 |
| 4 | **§9.9 P1 v0.1 walk-forward IC ablation** | 長期 v6.2.0 強制 gate | 待 §10 model_trainer 落地 |
| 5 | **電子業集中 over-fit risk**(本 dry-run 揭露) | 待 walk-forward IC 裁決 | 影響 universe selection 哲學 |

### 三個合理下一步方向

**方向 A — commit v0.6 snapshot 為 production(勇進派)**

```bash
python3 scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-21 \
  --policy-version core_universe_policy_v0.6 \
  --special-rebalance-reason "v6.1.10 v0.7.1 RMS 三層治權升版後首個 production v0.6 snapshot"
python3 scripts/maintenance/audit_core_universe.py --policy-version core_universe_policy_v0.6
```
風險:74.2% overlap = 27 stocks 進出;可能被 over-fit pattern 主導(電子業集中)

**方向 B — §10 model_trainer 設計研究(穩健派)**

不動 DB;深入研究 walk-forward IC 框架 + LSTM/XGBoost 選型 + sharpe 裁決機制 → 入憲 §10 model_trainer policy + 設計研究 → 後續再 commit v0.6 snapshot

**方向 C — ROE 第二次資料現實裁決(深治權派)**

研究 BalanceSheet 之 TotalEquity / CommonStock / RetainedEarnings 組合是否能重算真 Equity → ROE 復活 → V 動員度 64% → ~73%;若可行則入憲 §14.7-BI

---

## 🌐 九、跨機環境前置檢查清單(per CLAUDE.md §二 #7 / 憲章 §0.0-I.9-I.10)

### Step 1 — OS 原生依賴

**macOS**:
```bash
brew install libomp                    # xgboost / lightgbm 需 OpenMP
brew install postgresql@17             # 若 psycopg2-binary 找不到 client headers
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install -y libgomp1 libpq-dev
```

**Windows**: 通常無需(內含 vcomp140.dll)

### Step 2 — Python venv setup

```bash
cd stock_backend
python3 -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Step 3 — `.env` 跨平台路徑對齊(per 憲章 §0.0-I.10)

```env
PROJECT_ROOT=/home/<your-user>/project/stock_backend  # Linux
# 或
PROJECT_ROOT=/Users/<your-user>/project/stock_backend # macOS

DB_HOST=...
DB_PORT=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
```

`path_setup.py v4.47+` 用 `os.path.realpath()` 自動處理 symlink。

### Step 4 — Import smoke test(必須通過才能進入後續執行)

```bash
python3 -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅ all imports OK')"
```

如失敗,先補 OS 層依賴(Step 1)再重試,避免中途 rollback DB。

### Step 5 — DB 連線驗證

```bash
python3 -c "
from dotenv import load_dotenv
load_dotenv('.env')
import sys
sys.path.insert(0, 'scripts')
from core.db_utils import get_db_connection
conn = get_db_connection()
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM \"TaiwanStockInfo\";')
print(f'✅ DB connected; TaiwanStockInfo rows: {cur.fetchone()[0]}')
cur.close(); conn.close()
"
```

---

## 🔄 十、跨機接續操作 step-by-step

### Step 1 — Clone + checkout

```bash
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git log --oneline -3
# 應看到:
# 46f4d15 chore(maint+evidence): v0.7.1 RMS dry-run...
# 9ea2ba0 chore(reports): archive 4 audit outputs...
# 37bc687 feat(charter+core+audit): v6.1.0-patch 第十三輪...
```

### Step 2 — 環境前置(per §九)

### Step 3 — 驗證系統狀態

```bash
# audit_supply_chain 確認 schema 沒退化
python3 scripts/maintenance/audit_supply_chain.py

# audit_core_universe v0.2 確認 v0.2 baseline 仍 healthy
python3 scripts/maintenance/audit_core_universe.py --policy-version core_universe_policy_v0.2
# 預期: PASS=41, WARN=1(audit_self_log infra), FAIL=0

# 重跑 dry-run 確認 universe 結果一致
python3 scripts/maintenance/compare_v06_dryrun_vs_v02_baseline_20260525.py
# 預期: Core overlap 89/120 = 74.2%
```

### Step 4 — 讀本檔 + memory 對齊 context

讀以下文件:
1. **本檔**(`reports/session_handoff_20260525.md`)— 跨機 context
2. `reports/系統架構大憲章_v6.1.0.md`(尤其 §14.7-AY 至 §14.7-BH;v6.1.0-patch 13 entries)
3. `reports/v06_dryrun_vs_v02_baseline_universe_diff_20260525.md`(最新實證對照)
4. `reports/p1_v01_rms_vs_stddev_ablation_evidence_20260525_1604.md`(§14.7-BH 證據)
5. `CLAUDE.md`(AI 協作規則;尤其 §一 / §二 / §四)

### Step 5 — 啟動下個 session(依用戶選擇方向)

依 §八 之三個方向(A/B/C)選擇下一步。

---

## ⏳ 十一、Critical decisions pending(待用戶裁決事項)

| # | 待裁決事項 | 治權位階 | 影響 |
|---|---|---|---|
| 1 | v0.6 snapshot 是否 commit 至 DB(取代 v0.2 為 production)| §6.7 / §6.8 annual_guard | universe 重組 27 stocks |
| 2 | 電子業集中是否為 over-fit | §10 walk-forward IC | universe selection 哲學 |
| 3 | F 是否升 §0.1 T1 第 5 元素 | §0.1 + §6.4 weights 調整 | 治權框架重構 |
| 4 | ROE 第二次資料現實裁決(BalanceSheet 重算) | §0.1.3-A.1 + §14.7-BI(若新建) | V 動員度提升 |
| 5 | §10 model_trainer 設計研究啟動時機 | §10 + v6.2.0 強制 gate | walk-forward IC 框架 |

---

## 📊 十二、本 session 最終結算

| 指標 | 數值 |
|---|---|
| 總 commits 已 push GitHub | **16** |
| 新 tags 已 push GitHub | **2**(v6.1.9 + v6.1.10)|
| 憲章修訂歷程加入 entries | **13 條 v6.1.0-patch** |
| 新 charter sections | §14.7-AY 至 §14.7-BH **11 個** |
| 模組升版 | builder v0.2 → v0.7.1 / audit v0.1 → v0.2 / data_schema v2.16 → v2.20 / feature_store v0.3 → v0.5 |
| 設計研究報告 | **6 份** |
| 程式工具新增 | **3 個**(migrate + 2 個 ablation/compare 工具) |
| Evidence 報告新增 | **2 份**(ablation + dry-run)|
| §0.0-G 跑通次數 | **32 次** |
| §14.7-AX 跑通次數 | **7 次**(含首例「公式層揭露」)|
| V/F/ΔlnP 三軸動員度 | V 64% / F 88% / ΔlnP 100% 治權對齊(RMS)|

---

## 🔚 結語

本 session 完成從「§0.1.3-B DB Field Bottom-up 反推治權閉環」起步,到「§14.7-BH P1 v0.1 公式對齊 ablation + RMS 三層治權同步升版 + v0.6 dry-run 實證對照」之完整 13 commits 鏈。

**最關鍵成就**:
1. V/F/ΔlnP 三軸補強落地(FG 11 + IF 12 + VC RMS)
2. §9.10 由起草升正式條文(對齊 §9.9 RMS 強制契約)
3. §14.7-AX 治權元規則擴充為「資料/公式層揭露雙料含義」
4. 全市場 dry-run 證實 74.2% core universe 重組(31 stocks 進出)
5. GitHub 同步 + 2 個封存 tag(v6.1.9 + v6.1.10)

**唯一未跨越**:v0.6 snapshot 未 commit 至 DB(留待 §10 walk-forward IC 驗證後裁決)。

跨機接續時,從本檔開始讀 + 依 §九 環境前置 + 依 §十 step-by-step 操作 + 依 §八 選擇下一步方向。

---

**檔案產出**: 2026-05-25 by Claude Opus 4.7
**用戶授權**: 「請記錄目前此系統的作業現況,再晚一點我會換另外一台電腦再往下做」
