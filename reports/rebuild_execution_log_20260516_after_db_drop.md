# Quantum Finance v5.4.22 刪表後重建執行紀錄

**執行日期**: 2026-05-16  
**執行者**: Codex  
**依據文件**: `reports/系統架構大憲章_v5.4.22.md`  
**情境**: 使用者已刪除 database all table，依新修改憲章與程式碼從頭重建。  
**原則**: 依憲章 v5.4.22 第二章合法序列執行；如遇問題，記錄實際狀況、判定與處置。

---

## 0. 起始狀態

- 工作樹已有本次憲章與程式碼對齊修改，未回復。
- 使用者宣告 DB all table 已刪除，因此本次視為空 DB / 刪表後重建。
- 本次不修改行為程式碼；若發現問題，先記錄並依憲章序列處置。

## 1. 執行序列紀錄

| Step | 指令 | 結果 | 重點摘要 |
| :--- | :--- | :--- | :--- |
| 1 | `python scripts/core/path_setup.py` | PERFECT | v4.44；`.env` anchor MATCHED；25 維路徑對齊；因 DB table 尚未建立，log 模式為 `BOOTSTRAP-DEFERRED (OperationalError)`，屬空 DB 起始期合理狀態。 |
| 2 | `python scripts/core/data_schema.py --init --force` | PERFECT ALIGNMENT | API-first probe PASS/WARN/FAIL=11/0/0；13 張 raw / infra tables 建立成功，DDL=YES。 |
| 2B | `python scripts/core/core_universe_schema.py --init` | PERFECT | 第一次因誤平行執行而 FAILED；Step 2 完成後重跑成功。preflight PASS/WARN/FAIL=9/0/0；7 張核心股治理表建立成功，DDL=YES。 |
| 2C | `python scripts/core/db_utils.py` | WARNING | DB 連線成功；hybrid logs ACTIVE；§6.7 core universe query = 0 rows，屬 commit 前合法 bootstrap warning。 |
| 3 | `python scripts/maintenance/audit_supply_chain.py --include-logs` | PERFECT | 報告 `compliance_audit_20260516_1632.md`；PASS=29 / WARN=0 / FAIL=0。 |
| 4 | `python scripts/ingestion/sovereign_sync_engine.py --seed` | PERFECT | `sovereign_sync_engine v1.10`；TaiwanStockInfo 3403 筆 UPSERT；FRED DFF/UNRATE/T10Y2Y/VIXCLS 共 3885 筆；總寫入 7288；SUCCESS=5 / WARN=0 / FAIL=0。 |
| 4B-1 | `python scripts/core/core_universe_builder.py --dry-run --as-of-date 2026-05-14` | WARNING | v0.2-preflight；PREFLIGHT PASS/WARN/FAIL=7/0/0；V0.2 CONTRACT PASS/WARN/FAIL=10/10/0；total_candidates=2799；core=120；convex=30；research=2271；quarantine=378；`candidate_source=latest_registry_fallback`。WARNING 來自 v0.2 coverage 未完成與 as-of candidates=65 < 150。 |
| 4B-2 | `python scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-14` | WARNING | snapshot=`core_universe_20260514_core_universe_policy_v0_1`；written_rows=5601；分層同 dry-run；WARNING 同 dry-run，屬 v0.2 正式評分尚未 ready。 |
| 4C | `python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14` | PERFECT | 報告 `core_universe_audit_20260516_1633.md`；PASS=36 / WARN=0 / FAIL=0；membership=2799；scores=2799；core=120；convex=30；research=2271；quarantine=378。 |
| Final | `python scripts/core/db_utils.py` | PERFECT | commit 後 §6.7 核心同步資產數=150；DB、pipeline log、data audit log 均正常。 |
| Final-HUB | `python scripts/core/__init__.py` | PERFECT | 四層 hub 稽核通過；DB reachable；§6.7 core_universe ∪ convex_universe=150。 |
| Final-DB-Audit | `python scripts/maintenance/audit_supply_chain.py --db-only --include-logs` | PERFECT | 報告 `compliance_audit_20260516_1634.md`；PASS=22 / WARN=0 / FAIL=0；此為 seed 後 DB-only 快照，不取代 Step 3。 |

## 2. 問題紀錄

1. **操作順序問題：Step 2B 誤與 Step 2 平行執行**
   - 現象：`core_universe_schema.py --init` 回報 `pipeline_execution_log missing`、`data_audit_log missing`、`TaiwanStockInfo missing`，停止核心股治理 DDL。
   - 判定：這是執行方式違反憲章序列造成，不是程式碼問題；憲章要求 `data_schema.py --init --force` 完成後才能執行 `core_universe_schema.py --init`。
   - 處置：已等待 Step 2 完成；後續依序重跑 Step 2B。

2. **合法 bootstrap warning：Step 2C commit 前 §6.7 = 0**
   - 現象：`db_utils.py` 在核心股 builder commit 前顯示核心資產數 0，主權狀態 WARNING。
   - 判定：符合憲章 v5.4.22；治理表已建立但尚未產生 committed membership 時，0-row 是合法 bootstrap warning。
   - 處置：完成 builder commit 後重跑 `db_utils.py`，已收斂為 PERFECT，核心同步資產數 150。

3. **v0.2-preflight warning：完整 CoreScore v0.2 尚未 ready**
   - 現象：`core_universe_builder.py --dry-run/--commit` 皆為 WARNING；V0.2 CONTRACT PASS/WARN/FAIL=10/10/0。
   - 細節：`TaiwanStockPriceAdj`、`TaiwanStockMonthRevenue`、`TaiwanStockPER`、`TaiwanStockInstitutionalInvestorsBuySell`、`TaiwanStockMarginPurchaseShortSale`、`TaiwanStockFinancialStatements` 在 `2026-05-14` 前無 rows；`price_coverage_252d`、`revenue_coverage_24m`、`financial_coverage_8q` 為 0 coverage；`TaiwanStockInfo` as-of candidates=65 < 150，故使用 `candidate_source=latest_registry_fallback`。
   - 判定：符合憲章；本次 commit 僅代表 v0.1 metadata bootstrap + v0.2 input contract preflight，不代表完整量價 / 基本面 / 籌碼 CoreScore 已完成。
   - 後續：完整 v0.2 正式評分必須先完成歷史資料 coverage，並在 `candidate_source_mode=as_of_filtered` 下重新驗收。

4. **sandbox ad-hoc DB 查詢限制**
   - 現象：以 `python -c` 直接 import `core.db_utils.get_db_connection()` 查表時，在 sandbox 內發生 `psycopg2.OperationalError`；但正式腳本 `db_utils.py`、audit、builder 均可連線並通過。
   - 判定：這是 ad-hoc shell 查詢的環境注入邊界，不影響憲章授權序列；正式入口已驗證 DB 可用。
   - 處置：以正式稽核報告與授權腳本輸出作為本次紀錄依據。

## 3. 最終結論

刪表後重建已依憲章 v5.4.22 完成至 Step 4C，並補跑 commit 後 `db_utils.py`、hub 與 DB-only audit。最終狀態：

- Schema / infra / raw API tables：重建完成。
- 核心股治理 tables：重建完成。
- Seed ingestion：PERFECT，總寫入 7288。
- Core universe metadata bootstrap：commit 完成，membership/scores 各 2799。
- §6.7 核心同步資產數：150。
- 最終核心股驗收：PERFECT（PASS=36 / WARN=0 / FAIL=0）。
- 最終 DB-only 供應鏈驗收：PERFECT（PASS=22 / WARN=0 / FAIL=0）。

本次唯一阻斷型失敗為 Step 2B 誤平行執行造成的順序錯誤，已依憲章順序重跑修復；其餘 WARNING 均為憲章承認的 bootstrap / v0.2-preflight 邊界。
