# Quantum Finance v5.4.22 刪表後重建執行紀錄

**執行時間**: 2026-05-16 16:43:12 Asia/Taipei  
**執行者**: Codex  
**依據文件**: `reports/系統架構大憲章_v5.4.22.md`  
**情境**: 使用者再次刪除 database all table，依新修改憲章與修改後程式碼從頭重建。  
**原則**: 嚴格依憲章 v5.4.22 合法序列執行；有依賴的步驟不平行執行；所有問題記錄於本檔。

---

## 0. 起始狀態

- 使用者宣告 database all table 已刪除。
- 本次以空 DB / 刪表後狀態重建。
- 使用已修改後的程式碼與 `系統架構大憲章_v5.4.22.md`。

## 1. 執行序列紀錄

| Step | 指令 | 結果 | 重點摘要 |
| :--- | :--- | :--- | :--- |
| 1 | `python scripts/core/path_setup.py` | PERFECT | v4.44；`.env` anchor MATCHED；25 維路徑對齊；DB logging hook 在刪表/空 DB 起始期不可用，安全降級為 `BOOTSTRAP-DEFERRED (DB hook unavailable: ... OperationalError)`，不阻斷路徑自癒。 |
| 2 | `python scripts/core/data_schema.py --init --force` | PERFECT ALIGNMENT | API-first probe PASS/WARN/FAIL=11/0/0；13 張 infra/raw tables 重建成功；DDL=YES。 |
| 2B | `python scripts/core/core_universe_schema.py --init` | PERFECT | preflight PASS/WARN/FAIL=9/0/0；`RAW_COLUMN_INHERITANCE` 驗證完整執行；7 張核心股治理 tables 建立成功，DDL=YES。 |
| 2C | `python scripts/core/db_utils.py` | WARNING | DB 連線成功；hybrid logs ACTIVE；§6.7 core universe query=0 rows，屬 builder commit 前合法 bootstrap warning。 |
| 3 | `python scripts/maintenance/audit_supply_chain.py --include-logs` | PERFECT | 報告 `compliance_audit_20260516_1644.md`；PASS=29 / WARN=0 / FAIL=0。 |
| 4 | `python scripts/ingestion/sovereign_sync_engine.py --seed` | PERFECT | `sovereign_sync_engine v1.10`；TaiwanStockInfo 3403 筆 UPSERT；FRED DFF/UNRATE/T10Y2Y/VIXCLS 共 3885 筆；總寫入 7288；SUCCESS=5 / WARN=0 / FAIL=0。 |
| 4B-1 | `python scripts/core/core_universe_builder.py --dry-run --as-of-date 2026-05-14` | WARNING | v0.2-preflight；PREFLIGHT PASS/WARN/FAIL=7/0/0；V0.2 CONTRACT PASS/WARN/FAIL=10/10/0；total_candidates=2799；core=120；convex=30；research=2271；quarantine=378；`candidate_source=latest_registry_fallback`。WARNING 來自歷史資料 coverage 未完成與 as-of candidates=65 < 150。 |
| 4B-2 | `python scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-14` | WARNING | snapshot=`core_universe_20260514_core_universe_policy_v0_1`；written_rows=5601；total_candidates=2799；core=120；convex=30；research=2271；quarantine=378；WARNING 同 dry-run，屬 v0.2 正式評分尚未 ready。 |
| 4C | `python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14` | PERFECT | 報告 `core_universe_audit_20260516_1645.md`；PASS=36 / WARN=0 / FAIL=0；snapshot=`core_universe_20260514_core_universe_policy_v0_1`。 |
| Final | `python scripts/core/db_utils.py` | PERFECT | commit 後 §6.7 核心同步資產數=150；DB、pipeline log、data audit log 均正常。 |
| Final-HUB | `python scripts/core/__init__.py` | PERFECT | 四層 hub 稽核通過；DB reachable；§6.7 core_universe ∪ convex_universe=150。 |
| Final-DB-Audit | `python scripts/maintenance/audit_supply_chain.py --db-only --include-logs` | PERFECT | 報告 `compliance_audit_20260516_1646.md`；PASS=22 / WARN=0 / FAIL=0；此為 seed/commit 後 DB-only 快照，不取代 Step 3。 |

## 2. 問題紀錄

1. **Step 1 DB logging hook 安全降級**
   - 現象：`path_setup.py` 顯示 `BOOTSTRAP-DEFERRED (DB hook unavailable: ... OperationalError)`。
   - 判定：符合憲章；Step 1 負責路徑與 `.env` anchor，自癒不應因 DB logging hook 不可用而阻斷。
   - 處置：繼續 Step 2，由 `data_schema.py --init --force` 重建 DB infra tables。

2. **Step 2C 合法 bootstrap warning**
   - 現象：`db_utils.py` 顯示 §6.7 core universe query=0 rows，主權狀態 WARNING。
   - 判定：符合憲章 v5.4.22；核心股治理表已建立但尚未執行 builder commit 前，committed membership 為 0 是合法 bootstrap warning。
   - 處置：繼續 Step 3；builder commit 後需重跑 `db_utils.py` 驗證收斂為 PERFECT。

3. **v0.2-preflight coverage warning**
   - 現象：`core_universe_builder.py --dry-run` 判定 WARNING；V0.2 CONTRACT PASS/WARN/FAIL=10/10/0。
   - 細節：`TaiwanStockPriceAdj`、`TaiwanStockMonthRevenue`、`TaiwanStockPER`、`TaiwanStockInstitutionalInvestorsBuySell`、`TaiwanStockMarginPurchaseShortSale`、`TaiwanStockFinancialStatements` 在 `2026-05-14` 前無 rows；coverage 為 0；`TaiwanStockInfo` as-of candidates=65 < 150，使用 `latest_registry_fallback`。
   - 判定：符合憲章 v5.4.22；本階段只代表 v0.1 metadata bootstrap + v0.2 input contract preflight，不代表完整 CoreScore v0.2 ready。
   - 處置：dry-run 無 FAIL，繼續 commit；完整 v0.2 後續需補歷史 coverage 並以 `as_of_filtered` 驗收。

## 3. 最終結論

刪表後重建已依憲章 v5.4.22 完成，並補跑 commit 後 `db_utils.py`、hub 與 DB-only audit。最終狀態：

- Schema / infra / raw API tables：重建完成。
- 核心股治理 tables：重建完成。
- Seed ingestion：PERFECT，總寫入 7288。
- Core universe metadata bootstrap：commit 完成，written_rows=5601。
- `core_universe_membership` / `core_universe_scores`：各 2799。
- 分層：`core_universe` 120、`convex_universe` 30、`research_universe` 2271、`quarantine_universe` 378。
- §6.7 核心同步資產數：150。
- 最終核心股驗收：PERFECT（PASS=36 / WARN=0 / FAIL=0）。
- 最終 DB-only 供應鏈驗收：PERFECT（PASS=22 / WARN=0 / FAIL=0）。

本次沒有新的程式阻斷錯誤。記錄到的問題均屬憲章承認的空 DB bootstrap / v0.2-preflight 邊界：Step 1 DB logging hook 安全降級、Step 2C commit 前 §6.7 0-row warning、以及 v0.2 coverage 尚未完成。
