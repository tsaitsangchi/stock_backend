# 隔離區 _pending_removal_20260602

**建立**:2026-06-02 | **授權**:用戶 explicit「依 from_zero_to_model_build_guide_20260602.md 將没用的檔案移到隔離區」+ AskUserQuestion「只移 A 層」
**性質**:**可逆隔離,非刪除**。以 `git mv`(rename)移入,完整 git 歷史保留;誤判可 `git mv` 移回原路徑。

## 判準
依 `reports/from_zero_to_model_build_guide_20260602.md` canonical 程式集 + 唯讀依賴分析:本批 14 支為 **A 層(明確可棄)** —— 全部 **(1) 從未被任何檔 import、(2) 非 12-PHASE entry-point、(3) 不在憲章/runbook、(4) 屬 debug/test/一次性 migrate 性質**。移前逐支 `grep` 確認 0 import(安全)。

## 清單(原路徑 → 隔離後)+ 棄置理由

| 原路徑 | 理由 |
| :--- | :--- |
| `scripts/training/debug_8banks_2days.py` | 8-banks 除錯一次性腳本 |
| `scripts/training/debug_8banks_400.py` | 同上 |
| `scripts/training/debug_8banks_400_msg.py` | 同上 |
| `scripts/training/debug_8banks_single_day_2.py` | 同上 |
| `scripts/training/debug_8banks_week.py` | 同上 |
| `scripts/pipeline/debug_feat_crash.py` | feature crash 除錯一次性 |
| `scripts/training/test_adj_access.py` | 臨時測試腳本 |
| `scripts/fetchers/test_block_trading.py` | 臨時測試腳本 |
| `scripts/tests/test_round3_fixes.py` | 一次性回歸測試(round3,已過時)|
| `scripts/tests/test_round5_fixes.py` | 一次性回歸測試(round5,已過時)|
| `scripts/training/header_fix.py` | 一次性標頭修正工具 |
| `scripts/maintenance/migrate_snapshot_status_superseded_20260526.py` | 一次性 DB migration(已套用)|
| `scripts/core/migrate_stocks_config.py` | 一次性 config migration(已套用)|
| `scripts/maintenance/compare_v06_dryrun_vs_v02_baseline_20260525.py` | 一次性比較分析腳本 |

## app/ 整套子系統(14 檔,2026-06-02 第二批,用戶 explicit「整個 app/ 可安全隔離」)

`app/`(clean-architecture FastAPI "StockFront Auth Service":domain/entities+repositories、application/use_cases+commands+uow、infrastructure/auth_service+database、interfaces/api/auth_routes)整套移入 `_pending_removal_20260602/app/`(保留套件結構)。

**死碼確認證據**:(1) app/ 外**零** import(股票 pipeline 從不碰);(2) app/ 內**無** entry-point(無 FastAPI()/uvicorn/main);(3) 其**唯一 runner** `root main.py`(`FastAPI(title="StockFront Auth Service")` + `uvicorn.run(port=8001)`)**早在 2026-05-20 已被隔離**至 `_pending_removal_20260520/root_legacy_py/main.py`;(4) 憲章 0 引用;(5) 最後改 2026-05-10。→ runner 已走、無人 import = 確認整套死碼。

**附帶**:`requirements.txt` L92-93 `fastapi`/`uvicorn` 為此死服務依賴,隔離後成 unused(留著無害,日後可一併清)。

## Trinity v5.5.x legacy 子系統(34 檔,2026-06-02 第三批,用戶 explicit「隔離 39 legacy」→ 封閉安全集 34)

依**現行憲章可達性判斷**:以下為「Trinity Core Final v5.5.x」舊子系統層(daily 自動化 / monitor / backtest / training),**未被 charter-core(§3.1/§9.x/§14.7-CY/CZ/DD)遞移 import、憲章 0 引用** → legacy。其角色已被新框架(multi_cycle validators + §14.7-DD from-zero + §14.7-BX `run_weekly_doctrine_recommit`)取代。保留子目錄結構移入。

| 子目錄 | 數 | 內容 |
|---|---|---|
| `monitor/` | 11 | action_runner / backtest_audit / dashboard / db_health_check / db_optimize / init_monitoring_schema / log_manager / model_health_check / model_quality_audit / pipeline_monitor / update_daily_status |
| `training/` | 15 | auto_predict_manager / auto_train_manager / batch_predict_all / batch_tune / check_finmind_token / compute_stock_dynamics / feature_analysis / historical_backfill / parallel_train / path_fix / predict / train_evaluate / train_evaluate_body / tune_hyperparameters / update_feature_store |
| `pipeline/` | 3 | compute_stock_dynamics / faithful_portfolio_backtest / strategy_tester |
| `evaluation/` | 3 | faithful_portfolio_backtest / run_batch_backtest / strategy_tester |
| `utils/` | 2 | feature_selection / model_loader |

**封閉安全集驗證**:對 39 候選跑 transitive-closure —— 集合外**無任何檔 import 集合內** 才移(34 安全)。**5 支排除留下**(有外部 importer,日後與其 importer 一起隔離):`pipeline/data_pipeline`、`inference/signal_filter`、`monitor/data_audit_engine`、`monitor/data_integrity_audit`、`monitor/sync_trinity_db`。移後 import smoke + 關鍵程式 py_compile 全 PASS,**0 breakage**。

> ⚠️ caveat:training/monitor 含 daily-automation CLI,若仍有 cron 排程跑舊 Trinity 自動化,需自 archive 取回(本隔離假設新框架已取代)。

## C 群 legacy 大批次(58 檔,2026-06-02 第四批,用戶「C 沒用 → 隔離」)

依現行憲章兩軸判定(有用性 vs 標頭),以下為「沒用(charter 不可達 + 0 憲章引用)」之 legacy,封閉安全集隔離(集合外無人 import):

| 子目錄 | 數 | 內容 |
|---|---|---|
| `fetchers/` | 24 | fetch_*_data(18,sovereign_sync_engine §3.1 唯一同步載體取代)+ check_*(3)+ backfill_from_gaps + parallel_fetch + search_finmind_datasets |
| `ingestion/` | 23 | ingest_*_data(21,同 sovereign 取代)+ parallel_ingestion + search_finmind_datasets |
| `models/` | 4 | 舊 model 子系統 |
| `monitor/` | 3 | data_audit_engine / data_integrity_audit / sync_trinity_db(第三批 deferred 之 3,其 importer 此批一起移)|
| `audit/` `features/` `inference/` `reports/` | 各 1 | phase_d_ablation / run_feature_engineering / (inference orphan) / top_roi_stocks |

**封閉安全集**:對 62 候選跑 transitive-closure,移 59 安全(集合外無人 import;實移 58 + 1 前批已隔離)。**3 排除留下**(`pipeline/data_pipeline`、`pipeline/signal_filter`、`inference/signal_filter`)—— 被 **B 群(已補標頭)pipeline/portfolio backtest** import,須與該 backtest cluster 一起決定(deferred)。移後 import smoke + 全 101 剩餘 scripts/ .py py_compile PASS,**0 breakage**。

**charter 依據**:fetch_*/ingest_* 為 sovereign_sync_engine(§3.1)取代之 per-dataset legacy;其餘為 charter 不可達 orphan/舊子系統。

## reports/ 沒用 .md 大批次(89 檔,2026-06-02 第五批,用戶「reports 沒用 .md 依最新憲章判斷」+ AskUserQuestion「Tier 1 + Tier 2」)

依**引用可達性 + 取代軸**判定(265 .md 中 96 受憲章/CLAUDE/memory 引用 = 受保護必留)。對 0-引用候選套「系列內是否被更新版取代」+ **封閉安全集固定點**(只移「無未隔離活躍檔引用」者)後,移入 `reports_md/`(平面化 basename,無碰撞)。移後活躍引用斷裂 = **0**。

| 群 | 數 | 棄置理由(憲章) |
|---|---|---|
| 自動產生稽核快照(api_schema 14 / source_avail 14 / core_universe 10 / compliance 7 / doctrine 7 / db_vs_api 2)| **54** | 稽核程式可隨時重生;SSOT 為活 DB/API(§一.10);各系列**最新 1 份留在 reports/**,舊快照冗餘 |
| 被取代 handoff(留最新 `cross_machine_handoff_20260601_v6_26_1`)| 13 | session 歷程中間態,被最新封存點取代 |
| 05-29 舊模型產物(base 報告 7 / master summary 5 / dedicated 2)| 14 | 今天 06-02 已在重建後 397 核心 × 37 特徵 × §0.0-I 單一源重跑;06-02 為當前 SSOT |
| v6.0.0-era 報告 | 3 | v6.1.0 為憲章 SSOT;v6.0.0 final audit/runbook 被取代 |
| 舊版憲章/架構副本(系統架構大憲章 v5.4.19/20/21)| 3 | v6.1.0 為 SSOT;v5.x 全被取代 |
| 其他一次性 log/draft(rebuild_execution / v03_draft 等)| 2 | 一次性執行紀錄/草稿 |

**23 檔阻擋未移**(被未隔離之活躍檔引用 → 留以維持連結一致性,0 breakage):
- `Quantum_Finance v5.1/Asset_Matrix/Roadmap`(3,反憲章 §0.1-A 物理隱喻,但被 `reports/project_structure.md` 連結)
- `系統架構_v5.2 / 系統架構大憲章_v5.4.18 / 系統架構設計報告`(3,被活躍 `系統核心完整度評估報告.md` 連結,CLAUDE.md §六)
- 部分稽核快照 + handoff(被活躍 research/sync/handoff 檔連結);`common_model_comparison_baseline_v1`(被 chronos/itransformer 報告連結)
> 若日後一併處理 `project_structure.md` / `系統核心完整度評估報告.md` 之 stale 連結,可解除這批阻擋。

### 第五批 follow-up(+7 檔,2026-06-02,用戶「處理 2 索引檔 stale 連結」+ AskUserQuestion「全清 6 檔」)

解除上批 23 阻擋中之 7 檔(reports_md/ 89→96):

| 檔 | 數 | 處置 |
|---|---|---|
| `project_structure.md` | 1 | 5/9 自動產生之專案結構 dump(無產生器、含 53 個已隔離檔的 stale 連結、早於整個重建+清理)→ 整檔過時隔離。其殘留引用全為 `.gitignore` allowlist 與憲章內 gitignore-doc 引述區塊(`!reports/project_structure.md` glob,隔離後自然 no-op)|
| Quantum_Finance v5(`v5.1` / `v5_Asset_Matrix` / `v5_Roadmap`)| 3 | §0.1-A 明令禁止之物理隱喻(F=M×a / 重力井 / 20-60-20)v5 原始文件;雖 .gitignore glob 列「T1 治權文件」,經 AskUserQuestion 確認全清(可逆,git 史保留)|
| 舊版憲章/架構(`系統架構_v5.2` / `系統架構大憲章_v5.4.18` / `系統架構設計報告`)| 3 | v6.1.0 為 SSOT;v5.x 全被取代 |

**`系統核心完整度評估報告.md`(CLAUDE.md §六 引用,保留)之 stale 連結處理**:該報告為 2026-05-19 快照(基準 v6.0.0),§1.1 文件清單以歷史目錄列此 6 檔。已於報告頂部置入 **§一.11 SUPERSEDED banner**,聲明其為 5/19 快照且清單中多份已於 2026-06-02 隔離 → **保留歷史目錄完整,不逐行破壞記錄**。故該報告仍 textually 提及此 6 檔名(intentional 歷史目錄,banner 已註),**非斷裂連結**。

> 仍未解除之 23 阻擋餘 16(被其他活躍 research/sync/handoff 檔引用,非本 2 索引檔)— deferred。

## 未移(保留待審)
- **B 層 診斷 CLI**(5):`check_db_locks` / `check_finmind_datalist` / `check_finmind_quota` / `check_finmind_token` / `search_finmind_datasets` —— 獨立診斷工具,可能偶爾手動跑。
- **C 層 疑似工作流工具**(10):`auto_predict_manager` / `batch_predict_all` / `batch_tune` / `historical_backfill` / `train_evaluate_body` / `update_feature_store` / `parallel_ingestion` / `phase_d_ablation` / `run_batch_backtest` / `top_roi_stocks` —— 須逐個確認用途。

## 恢復
```bash
git mv archive/_pending_removal_20260602/<file>.py scripts/<原子目錄>/<file>.py
```
