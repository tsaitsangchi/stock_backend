# Cross-Machine Handoff — 2026-05-28(v6.5.0 native gate active / 預備換機)

**Handoff date**: 2026-05-27 → 2026-05-28(換機日)
**Final HEAD**: `1a0138e`(可直接 `git pull origin master` 取得)
**Final state**: v6.5.0 native gate 全閉環 sealed
**Repo URL**: https://github.com/tsaitsangchi/stock_backend.git
**位階**: 完整跨機接續 context(本檔 standalone;另機只需讀此一檔即可接手)

---

## 〇、Quick Start(新機 first 30 min checklist)

### 0.1 Clone + git pull

```bash
cd ~/project
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend

# 驗證在 1a0138e
git log --oneline -3
# expect:
# 1a0138e docs(handoff): 2026-05-27 v6.5.0 native gate sealed — 封存點 session handoff
# 5f40b3b feat(governance+db+code): §14.7-CG Phase D+E 全閉環 — v6.5.0 Native Gate Integration 落地
# 51668ef feat(charter+code+research): §14.7-CG v6.5.0 Native Gate Integration Phase A-C 閉環
```

### 0.2 Python 環境

```bash
# Python 3.12(必須;舊機為 /Users/hugo/.pyenv/versions/3.12.13)
python3 --version  # >= 3.12

# Create venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 0.3 OS 原生依賴(per CLAUDE.md §二.7 / 憲章 §0.0-I.9)

```bash
# macOS
brew install libomp postgresql@17

# Linux
sudo apt-get install -y libgomp1 libpq-dev postgresql-17

# Import smoke test(必須通過才能進入後續執行)
python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm; print('✅ all imports OK')"
```

### 0.4 環境變數 `.env`(從舊機複製;新機需重新取得)

```bash
# 從舊機複製整份 .env(含以下變數):
# - DB_HOST=127.0.0.1
# - DB_PORT=5432
# - DB_NAME=stock
# - DB_USER=stock
# - DB_PASSWORD=stock
# - FINMIND_TOKEN=<舊機之 sponsor tier token;舊機 .env 有>
# - FRED_API_KEY=<舊機之 FRED API key;舊機 .env 有>
# - GEMINI_API_KEY=<舊機之 Gemini key;可選>
# - GITHUB_TOKEN=<舊機之 GitHub token;若 chat transcript 已暴露須先 revoke 重生>

# 建議用 scp / rsync / USB:
scp old-machine:~/project/stock_backend/.env ./

# 或手動寫入(需取得各 token)
cp .env.example .env  # 若有 example
# 然後填入各 token

# ⚠️ Tokens 若曾在 chat / log 中暴露,先 revoke 重生:
# - FinMind: https://finmindtrade.com/analysis/UserInfo(重生 token)
# - FRED: https://fred.stlouisfed.org/docs/api/api_key.html(可申請新 key)
# - GitHub: https://github.com/settings/tokens(revoke + regenerate)
```

### 0.5 PostgreSQL 啟動 + DB 還原

**選項 A**:從舊機 dump 還原(推薦,可即刻 reproduce v0.13 state)

```bash
# 舊機:
pg_dump -h 127.0.0.1 -U stock -d stock -Fc -f /tmp/stock_db_20260527.dump

# 新機:
brew services start postgresql@17  # macOS
# or: sudo systemctl start postgresql

createdb -U stock stock  # 若不存在
pg_restore -h 127.0.0.1 -U stock -d stock --clean /tmp/stock_db_20260527.dump
```

**選項 B**:從零重建(需 ~9-15h FinMind/FRED sync)

```bash
# 1. 建 schema
python scripts/core/data_schema.py --init --force
python scripts/core/core_universe_schema.py --init

# 2. Full market sync(慢)
python scripts/fetchers/fetch_stock_info.py
python scripts/fetchers/fetch_price_adj_data.py
python scripts/fetchers/fetch_fundamental_data.py
python scripts/fetchers/fetch_balance_sheet_data.py
python scripts/fetchers/fetch_chip_data.py
python scripts/fetchers/fetch_fred_data.py
# (依憲章 §二 standard 序列;預計 ~9-15h on sponsor tier 6000/hr quota)

# 3. Build v0.13 snapshot
python scripts/core/core_universe_builder.py --mode doctrine-native --commit
# expect: N=1,583 / Stage 1-4 PASS

# 4. Feature store
python scripts/core/feature_store_builder.py --commit
# expect: 1,583 stocks × 65 features = 96,516 rows
```

### 0.6 DB 狀態驗證(新機應跑通)

```bash
source venv/bin/activate && set -a && source .env && set +a

# 驗證 1: v0.13 active snapshot
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -p 5432 -U stock -d stock -c "
SELECT snapshot_id, status, core_count, policy_version
FROM core_universe_snapshot
WHERE status='committed' LIMIT 1;
"
# expect:
# core_universe_20260527_core_universe_policy_v0_13_doctrine_native_gate | committed | 1583 | core_universe_policy_v0.13_doctrine_native_gate

# 驗證 2: feature_store_snapshot 綁定 v0.13
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -p 5432 -U stock -d stock -c "
SELECT feature_set_id, universe_snapshot_id, total_stocks, feature_count
FROM feature_store_snapshot
WHERE status='committed' ORDER BY as_of_date DESC LIMIT 1;
"
# expect: fs_20260527_feature_set_v0_4 | ...v0_13... | 1583 | 65

# 驗證 3: native builder dry-run
python scripts/core/core_universe_builder.py --mode doctrine-native --dry-run
# expect: Stage 1-4 PASS / N=1,583
```

---

## 一、Current Project State(v6.5.0 native gate sealed)

### 1.1 Charter sections(治權閉環)

```
§14.7-BW (2026-05-26;N-axis)
   ↓ N 動態 doctrine-derived
§14.7-CC (2026-05-27;Source-axis)
   ↓ FinMind/FRED API uniqueness
§14.7-CD (2026-05-27;Source-Completeness-axis)
   ↓ 11 raw source × thresholds per-stock
§14.7-CE (2026-05-27;Empirical-Verification-axis)
   ↓ 7,834 entries byte-level proof / 0 synthetic
§14.7-CF (2026-05-27;SSOT-Unification-axis)
   ↓ 三 invariant 統合入口
§14.7-CG (2026-05-27;Native-Implementation-axis)
   ↓ 3 step → 1 program
🏛️ v6.5.0 sealed at 1a0138e
```

### 1.2 治權判準十一純化軸完成

N + T + Indicator + Pillar + Feature + Completeness + Source + Source-Completeness + Empirical-Verification + SSOT-Unification + **Native-Implementation**

### 1.3 DB Snapshots(v0.13 為 active)

| Snapshot | Policy | N | Status |
|---|---|---:|---|
| `core_universe_20260527_core_universe_policy_v0_13_doctrine_native_gate` | v0.13_doctrine_native_gate | **1,583** | **committed (active)** |
| v0.12 raw_data_completeness_gate | — | 1,543 | superseded |
| v0.11 feature_completeness_gate | — | 1,640 | superseded |
| v0.10 pure_doctrine | — | 1,862 | superseded |

### 1.4 Active programs(production-current)

- **`scripts/core/core_universe_builder.py`** — v0.13 native gate
  - `DoctrineNativeGateBuilder` class
  - CLI: `--mode doctrine-native [--commit | --dry-run] [--with-feature-gate]`
  - 5 stages: K-wave + §0.1 + §0.2 + union + atomic supersede

- **`scripts/core/feature_store_builder.py`** — feature_set_v0.4(65 features × 1,583 stocks)

- **`scripts/maintenance/run_weekly_doctrine_recommit.py`** — weekly orchestrator
  - Step 4 已切換為 `core_universe_builder.py --mode doctrine-native --commit`

### 1.5 DEPRECATED programs(已標 header;v6.5.x 後完全下架)

- `scripts/maintenance/build_doctrine_gate_universe.py`(§14.7-BV/BW v0.10;邏輯已併入 DoctrineNativeGateBuilder Stage 1+2)
- `scripts/maintenance/apply_feature_completeness_gate.py`(§14.7-CB v0.11;Stage 4 optional)
- `scripts/maintenance/apply_raw_data_completeness_gate.py`(§14.7-CD v0.12;Stage 2+3 移植)
- `scripts/maintenance/compute_semi_supply_cycle_proxy.py`(§14.7-CC FRED-native 取代)

### 1.6 N 三層關係(Phase D-3 揭露)

```
N=2,803 candidates (full TaiwanStockInfo)
  ↓ §14.7-CG (13 K-wave + 11 raw thresholds)
N=1,583 §14.7-CF pure (Reading A+C / active v0.13)
  ↓ +§14.7-CB 37/37 feature gate (post-rebuild)
N=1,548 Reading B (35 stocks 缺 derived feature:31 缺 operating_margin_ttm + 其餘)
  ↓ +90d recency check (v0.11 step)
N=1,543 v0.12 historical (superseded)
```

---

## 二、Pending Follow-up(下輪 v6.5.x 工作)

按優先級排序:

### P0(治權完整性)
1. **audit_core_universe.py refactor 對 v0.13 native gate**
   - 當前:對 v0.13 出 11 FAIL(by-design 介面差異;v0.13 不寫 scores/research/mirror columns per §14.7-BW)
   - 需要:加 `NATIVE_GATE_POLICIES` set + skip 相應 checks(policy_source/membership_count/scores_count/raw_column_mirror/revision_log/data_audit_log)
   - 詳見:`reports/core_universe_audit_20260527_2213.md`(FAIL 列表)

### P1(技術債清理)
2. **`kwave_supply_cycle_proxy` table DROP**
   - 當前:DB 中表還在,僅 audit trail;active builder 不讀
   - 需要:`DROP TABLE kwave_supply_cycle_proxy CASCADE` + 從 data_schema 移除 + 4 個 deprecated script 完整下架
   - charter §14.7-CD 預告 v6.4.5;§14.7-CG 預告 v6.5.x

3. **`build_doctrine_gate_universe.py` / `apply_*_completeness_gate.py` / `compute_semi_supply_cycle_proxy.py` 完整下架**
   - 當前:標 DEPRECATED header;邏輯仍存
   - 需要:確認無 cron / external scripts 引用後刪除檔案

### P2(精度提升)
4. **35 stocks `operating_margin_ttm` root cause**
   - 31 stocks raw FinStmt 4Q 通過 §14.7-CD threshold,但 feature_store_builder TTM 計算 fail
   - 假設:Taiwan 財報「累積」shape vs feature_store_builder TTM 計算邊界
   - 需要:逐 stock 抽樣審視 FinStmt rows(Q1/H1/9M/FY cumulative pattern)+ 修補 feature_store_builder logic
   - 若修補成功 → Reading B 完全收斂 → N_with_feature_gate 從 1,548 升至 1,583

### P3(預設切換)
5. **v6.6.0 預設 mode 切 doctrine-native**
   - 當前:`core_universe_builder.py` CLI 預設 `--mode legacy-corescore`
   - 需要:切預設為 `doctrine-native`;legacy 仍可用 `--mode legacy-corescore` 顯式呼叫
   - charter 升 v6.6.0(major version bump per §14.7-CG 計畫)

---

## 三、用戶治權 directives 完整對映(本 session)

| 用戶 directive | charter 入憲節 | 落地 |
|---|---|---|
| 比對每一支個股,非抽樣 | §14.7-CE | 4 軸 × 1,543 stocks × byte-level = 7,834 entries / 0 synthetic |
| 三基柱對應資料來源 | §14.7-CF Invariant 1 | DoctrineNativeGateBuilder Stage 1+2+3 |
| 没有固定 N | §14.7-CF Invariant 2 | N 動態 = gate 結果(當前 1,583) |
| 全部資料 FinMind/FRED API | §14.7-CF Invariant 3 | API endpoint uniqueness + §14.7-CE proof |
| 啟動 v6.5.0 native gate | §14.7-CG | `scripts/core/core_universe_builder.py --mode doctrine-native` |
| Reading A+C → Reading B 收斂 | §14.7-CG Phase D-3 | 三層關係 inscribed |

---

## 四、新機驗證清單(完成後即可開始新工作)

| # | 驗證項 | 預期結果 |
|---|---|---|
| 1 | `git log --oneline -3` 第一行 | `1a0138e docs(handoff)...` |
| 2 | Python imports smoke | psycopg2/pandas/polars/numpy/requests/sklearn/xgboost/lightgbm 全 OK |
| 3 | PostgreSQL 連線 | `pg_isready` accepting connections |
| 4 | DB 有 v0.13 committed | `core_universe_snapshot` 有 1 row status='committed' / N=1583 |
| 5 | feature_store 綁定 v0.13 | `fs_20260527_feature_set_v0_4` / 96,516 rows |
| 6 | Native builder dry-run | `python scripts/core/core_universe_builder.py --mode doctrine-native --dry-run` → Stage 1-4 PASS / N=1,583 |
| 7 | Charter 完整 | `wc -l reports/系統架構大憲章_v6.1.0.md` ≈ 10,461 lines / `grep "§14.7-CG" reports/系統架構大憲章_v6.1.0.md \| wc -l` ≥ 10 hits |
| 8 | Memory 在 ~/.claude | `cat ~/.claude/projects/-Users-hugo-project-stock-backend/memory/core_stock_selection_doctrine.md` 應反映 v0.13 N=1,583 |

---

## 五、若新機環境差異(troubleshooting)

### 5.1 Python 版本差異
- 若新機 Python ≠ 3.12 → 安裝 pyenv:`brew install pyenv` + `pyenv install 3.12.13`
- 不可用 Python 3.13+(部分依賴未支援)

### 5.2 PostgreSQL 版本差異
- 必須 PostgreSQL 17(per CLAUDE.md §二.7);新機若已有 ≠17 版本,並存安裝 17
- 若 dump 為 17 製,還原至 ≠17 可能 schema 問題

### 5.3 PROJECT_ROOT 路徑差異
- 舊機:`/Users/hugo/project/stock_backend`(macOS;Hugo)
- 新機若不同 username / 路徑:更新 `.env PROJECT_ROOT`(per 憲章 §0.0-I.10);`path_setup.py v4.47+` 用 `os.path.realpath()` 解析

### 5.4 Token revocation
- 若 chat transcript 曾暴露 FINMIND_TOKEN / FRED_API_KEY / GITHUB_TOKEN,**先 revoke 重生**才設新機
- 之後 sync test:`python scripts/fetchers/fetch_fred_data.py --ids T10Y2Y`(應 200 OK)

### 5.5 Charter v6.1.0 vs v6.5.0 命名
- 注意:**檔名是 `系統架構大憲章_v6.1.0.md`**(主 charter 版號)
- v6.4.3 / v6.5.0 / v6.6.0 為 **patch round** 編號(於 charter 內第 N 輪 entry 標記),非主版號
- 主 charter 升 v7.0.0 為 breaking change reserved;v6.1.0 + patch entries 仍為當前 active

---

## 六、新機若要繼續工作

### 6.1 啟動建議順序
1. 完成 §〇 Quick Start(30 min)
2. 跑 §四 驗證清單(15 min)
3. 讀本檔 + `session_handoff_20260527_v6_5_0_native_gate_sealed.md`(對 night session 細節)+ charter §14.7-CE/CF/CG(對 doctrine 細節)
4. 從 §二 Pending Follow-up 挑優先級開始(建議 P0 audit refactor 為下一輪起點)

### 6.2 新工作之治權前置條件
- 任何核心股 selection 修改:**必以 §14.7-CG 為 implementation 基礎 + §14.7-CF 為 doctrine 基礎**;不可走舊 3 step pipeline 路徑
- 任何 raw data fetcher 修改:**必對應 FinMind / FRED API endpoint**;不可寫 synthetic / proxy(per §14.7-CC)
- 任何 feature 修改:**raw 完整 ⟹ feature 真實或 None;廢棄 fallback**(per §14.7-CD)
- 任何 charter inscription:**對應 §14.7-X 輪次 + 修訂歷程 entry + 標頭最後更新日期同步**(per CLAUDE.md §四.4)

### 6.3 緊急聯絡 / 治權升級
- 若新機運作後發現本 handoff 描述與實際 DB / code 不符,**先停止任何 destructive 操作**並回看:
  1. `git log` 對比此檔 final HEAD `1a0138e`
  2. DB committed snapshot 是否為 v0.13 N=1,583
  3. 若 mismatch → 可能是 next session 已動過;讀 latest session handoff(`ls -t reports/session_handoff_*` 最新)

---

## 七、本檔之 invariants(換機 sealed at 1a0138e)

- ✅ Repo HEAD = `1a0138e`(GitHub `tsaitsangchi/stock_backend` master)
- ✅ DB v0.13 active(`core_universe_policy_v0.13_doctrine_native_gate` / N=1,583 / committed)
- ✅ Feature store rebuilt(fs_20260527_feature_set_v0_4 / 96,516 rows / 綁定 v0.13)
- ✅ Charter §14.7-CE/CF/CG 三節 inscribed at v6.1.0-patch 第二十九/三十/三十一輪
- ✅ Memory updated(`~/.claude/projects/-Users-hugo-project-stock-backend/memory/core_stock_selection_doctrine.md` 升 v9)
- ✅ 4 個舊 script DEPRECATED header
- ✅ Weekly orchestrator Step 4 切換新 builder
- ⏸ audit_core_universe v0.3 對 v0.13 出 11 FAIL(by-design;下輪 P0 refactor)
- ⏸ `kwave_supply_cycle_proxy` table 仍在 DB(下輪 DROP)

---

**🏛️ 本檔為換機日所需之 single canonical entry。讀此一檔即可在新機接手 v6.5.0 native gate 後續工作。**
