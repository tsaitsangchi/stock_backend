# 跨機接力封存點 — v6.27.0 全市場增量入憲 (Cross-Machine Handoff)

**日期**：2026-06-02
**接力點 HEAD**：`96b6ff4`（master）
**最新 tag**：`v6.27.0-section6-8-7-clause5-full-market-incremental-sovereign-v1.23-20260602`
**Remote**：`github.com/tsaitsangchi/stock_backend`（master 已同步至 96b6ff4，本 session 全部成果已 push）
**性質**：跨機接力載體（§二.4）。用戶 explicit「會換另一台電腦做此專案」。

---

## 🎯 零、白話：新機第一步做什麼（給人看的）

1. `git clone https://github.com/tsaitsangchi/stock_backend` → `git checkout master`（到 `96b6ff4`）。
2. 建環境（venv + `.env` + OS deps + import smoke test，見 §2）。
3. **⚠️ 最關鍵**：**DB 不隨 git 走**（PostgreSQL ~81M rows 是本機資料）。新機要嘛 (a) 從零重建（§14.7-DD 12-PHASE，~6-10hr）、要嘛 (b) 還原備份。見 §3。
4. 讀本檔 §4 了解上一機做到哪、§6 待續事項。

**一句話現況**：程式碼/憲章/報告全在 git（乾淨、已 push）；**DB + 模型 artifacts + memory + 背景跑的 pipeline 都是本機資產，不隨 git 轉移**，新機須自行重建。

---

## 一、git 接力點

```bash
git clone https://github.com/tsaitsangchi/stock_backend
cd stock_backend && git checkout master        # → 96b6ff4
git log --oneline -1                             # 確認 96b6ff4
```

- 工作樹乾淨、HEAD = origin/master = `96b6ff4`、0 ahead / 0 behind。
- 本 session 11 個封存 tag（v6.26.6 → v6.27.0）全在 remote。

## 二、環境建立（新機，per CLAUDE.md §二.7 + §0.0-I.8/I.9）

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
# OS deps：Linux: sudo apt-get install -y libgomp1 libpq-dev ; macOS: brew install libomp postgresql@17
# import smoke test（必過才往下）：
./venv/bin/python -c "import psycopg2,pandas,polars,numpy,requests,sklearn,xgboost,lightgbm; print('✅ imports OK')"
```

- **`.env` 須重建**（14 強制變數，per §0.0-I.8）：`PROJECT_ROOT`（對齊新機物理路徑）、PostgreSQL role/db/host/port/password、`FINMIND_API_TOKEN`、`FRED_API_KEY`、`GITHUB_TOKEN`（push 用）等。**`.env` 不在 git**（機密）。
- PostgreSQL：建 role + db（見 §14.7-DD PHASE 0）。

## 三、⚠️ DB 狀態 + 重建（最關鍵 — DB 不在 git）

**上一機 DB 現況（僅供參考，新機須自建）**：
- committed snapshot：`core_universe_20260601_core_universe_policy_v0_18_source_pure_panhistorical_gate`
- **397 core + 730 quarantine**（research/convex tier 空）
- 日頻表（TaiwanStockPrice 等）max = **2026-06-02**（核心 397 已增量補到今天）；FRED `FredData` / `fred_series` max = 2026-06-01
- feature_store：`feature_set_v0.5`（37 features，SPEC_37）/ 96 panels

**新機重建路徑（二擇一）**：
- **(a) 從零重建（推薦,最乾淨）**：依 `reports/tree_based_from_zero_build_runbook_20260531.md`（§14.7-DD 12-PHASE）+ memory `from_zero_rebuild_runbook_gaps_20260601`（3 個空-DB 順序缺口修法：fred_series 前置 / PHASE3 `--bootstrap` / PHASE7 需 universe_completeness+model_registry 提前建）。PHASE 4 全市場同步 ~5-6hr（~81M rows）。
- **(b) 還原備份**：上一機備份曾在 `/tmp/rebuild_from_zero/`（**本機 /tmp，不隨 git 轉移**；若未另存則只能走 (a)）。

**補資料到今天**（DB 建好後）：見 memory `sync_engine_catchup_modes`：核心增量 `sovereign_sync_engine.py --universe core --all`（~8min）;全市場增量（本 session 新增 §6.8.7 第(5)條）`--universe full --incremental [--roster] --special-full-market-reason "..."`。

## 四、本 session 成果（v6.26.6 → v6.27.0，2026-06-02）

| 階段 | 內容 | tag |
|---|---|---|
| Metric helper 統一 | §0.0-I 雙 helper（panel-date §14.7-DE + metric §14.7-DF）落 `db_utils v2.50`；13 validator 切換；§一.11 第零段白話入憲 | v6.26.3–6.26.6（前序）|
| §一.11 標頭全面 retrofit | 49 程式補白話段 + charter-core 9 補 4 段 + B 群 20 補標頭 | v6.26.7/8/10 |
| 檔案清理（兩軸：標頭完整度 ⊥ 有用性）| 隔離 **120 .py**（28 debug/app + 34 Trinity + 58 fetch/ingest legacy）+ **97 .md**（稽核快照/舊模型/v5.x 憲章副本/project_structure dump 等）至 `archive/_pending_removal_20260602/`（git mv 可逆）；scripts/ 207→101、reports/ 265→169；0 breakage | v6.26.9/11/12/13 |
| reports 殘留阻擋分析 | ~16 阻擋檔追查 → 0 可乾淨清除（皆 load-bearing：baseline_v1 torch 事實基準 + 憲章 T2 §14 audit-trail）；建議停 | v6.26.14 |
| **全市場增量（本輪主軸）** | **憲章 §6.8.7 第(5)條** 全市場增量維運 + `sovereign_sync_engine.py v1.23`（`--incremental` 抑制 auto-strict + `--roster` 全名冊解析）；向後相容零破壞;dry-run 全通過 | **v6.27.0** |

## 五、進行中 / 背景（⚠️ 不隨 git 轉移）

- **模型 pipeline 在上一機背景跑**：`multi_cycle_transformer_dedicated_validation.py --commit`（#7 torch，PID 609791，16:09 載入 107 panels）。**torch on CPU 極慢（數天）**。**此 process + /tmp logs 不轉移**；新機若要續跑須重啟（DB 建好後）。
- 隊列原規劃（記於 [[session_metric_helper_unification_20260602]]）：transformer_dedicated → tft/itransformer/patchtst/chronos（窗對齊 + chronos pip）→ foundation（TTM/Time-MoE/Moirai/Lag-Llama 新寫）→ deep（DLinear 等新寫）→ master comparison。
- **6 樹模型已驗證完成**（報告在 reports/，annual 全過 T_CZ-6；XGB≈LGBM>Ensemble>CatBoost>RF>ET；net +23-33%/yr）。

## 六、待續事項（pending）

1. **模型 pipeline 續跑**（新機 DB 建好後）：torch 序列 + foundation/deep 新模型（多為「輪到才寫」未實作）。
2. **3 互依 .py cluster** deferred（`pipeline/data_pipeline` + `pipeline/signal_filter` + `inference/signal_filter`，與 B 群 backtest cluster 互依，須 joint 決定才隔離）。
3. **reports 16 阻擋檔**：load-bearing，建議不動（見 `reports/reports_md_residual_blocked_load_bearing_analysis_20260602.md`）;自然退場 = torch 在新宇宙重跑後 baseline_v1 + 4 torch 報告一起退役。
4. **全市場增量未實跑**：v6.27.0 僅 dry-run 驗證路徑;真要跑 `--roster` ~5hr（call-bound）。

## 七、隔離區（archive/_pending_removal_20260602/，可逆）

- **120 .py + 97 .md**，`git mv` 移入（git 史保留）。manifest = `archive/_pending_removal_20260602/README.md`。
- 恢復：`git mv archive/_pending_removal_20260602/<path> scripts/<原目錄>/`（或 reports_md/ → reports/）。
- 全在 git → **隨 clone 轉移**（與 DB/models 不同）。

## 八、不隨 git 轉移的 machine-local 資產（新機須留意）

| 資產 | 位置 | 處置 |
|---|---|---|
| **PostgreSQL DB**（~81M rows）| 本機 PG | 新機重建（§3）|
| **模型 artifacts**（68 子目錄）| `data/models/`（多 gitignored）| 新機重訓 or 手動複製 |
| **memory（11 檔）** | `~/.claude/projects/.../memory/` | **不隨 git**;canonical 接力 = 本報告。關鍵 memory：`from_zero_rebuild_runbook_gaps_20260601` / `sync_engine_catchup_modes` / `session_metric_helper_unification_20260602` / `project_charter_v6_1_0`（內容已濃縮於本報告 §3-6）|
| **/tmp 備份 + logs** | `/tmp/rebuild_from_zero/`、`/tmp/*.log` | 不轉移 |
| **背景 process** | PID 609791 等 | 不轉移 |

## 九、關鍵治權索引

- **憲章 SSOT**：`reports/系統架構大憲章_v6.1.0.md`（本 session 加 §6.8.7 第(5)條）。
- **AI 工具規則**：`CLAUDE.md`（§一.16/§一.17 單一引用源雙 helper、§一.11 三段式+第零段白話、§一.10 資料真實性、§一.13 source-pure、§14.7-DD 12-PHASE）。
- **from-zero runbook**：`reports/tree_based_from_zero_build_runbook_20260531.md` + `reports/from_zero_to_model_build_guide_20260602.md`。
- **同步指令**：memory `sync_engine_catchup_modes` + 憲章 §6.8.7（第(1)/(1A)/(4)/(5) 條）。

**證據基礎**：本報告全數字出自實查（git rev-parse + ps + DB query + find），無臆測（§一.10）。HEAD 96b6ff4 / DB v0.18/397 / 背景 transformer_dedicated PID 609791 / 隔離 120+97 / data/models 68 子目錄 皆 verified。
