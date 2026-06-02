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

## 未移(保留待審)
- **B 層 診斷 CLI**(5):`check_db_locks` / `check_finmind_datalist` / `check_finmind_quota` / `check_finmind_token` / `search_finmind_datasets` —— 獨立診斷工具,可能偶爾手動跑。
- **C 層 疑似工作流工具**(10):`auto_predict_manager` / `batch_predict_all` / `batch_tune` / `historical_backfill` / `train_evaluate_body` / `update_feature_store` / `parallel_ingestion` / `phase_d_ablation` / `run_batch_backtest` / `top_roi_stocks` —— 須逐個確認用途。

## 恢復
```bash
git mv archive/_pending_removal_20260602/<file>.py scripts/<原子目錄>/<file>.py
```
