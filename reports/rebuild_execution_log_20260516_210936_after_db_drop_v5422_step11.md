# Rebuild Execution Log - 2026-05-16 21:09:36 CST

依使用者要求：「我把 database all table 刪除了，請依新修改的系統架構大憲章_v5.4.22.md及修改後的程式碼，從頭開始執行並記錄執行狀況與問題。」

本輪同時納入使用者補充之憲章 §8 草案要求：Feature Store / Model Registry / Prediction Table，下游 Step 9 / Step 10 / Step 11 / Step 11A。

## 起始狀態

- 工作目錄：`/home/hugo/project/stock_backend`
- 憲章：`reports/系統架構大憲章_v5.4.22.md`
- 觀察到的起始狀態：Step 1 `path_setup.py` 顯示 `REAL (DB-Linked)`，Step 2C `db_utils.py` 顯示 §6.7 核心資產數已為 150。
- 裁決：本輪 DB 並非完全空白狀態；至少 `pipeline_execution_log` / `data_audit_log` 及既有核心股治理資料仍存在。依安全規則，本輪未自行執行 destructive drop。

## 執行序列與結果

| Step | 指令 | 結果 | 摘要 |
| :--- | :--- | :--- | :--- |
| 1 | `python scripts/core/path_setup.py` | PERFECT | `.env` anchor MATCHED；25 維路徑對齊；log mode=`REAL (DB-Linked)`。 |
| 2 | `python scripts/core/data_schema.py --init --force` | PERFECT ALIGNMENT | API PASS/WARN/FAIL=11/0/0；13 張 raw/infra tables 重鑄。 |
| 2B | `python scripts/core/core_universe_schema.py --init` | PERFECT | preflight PASS/WARN/FAIL=9/0/0；7 張核心股治理 tables 存在/建立完成。 |
| 2C | `python scripts/core/db_utils.py` | PERFECT | DB SUCCESS；§6.7 核心資產數=150。此結果再次證明 DB 起始非空。 |
| 3 | `python scripts/maintenance/audit_supply_chain.py --include-logs` | PERFECT | 報告：`reports/compliance_audit_20260516_2104.md`；PASS=29/WARN=0/FAIL=0。 |
| 4 | `python scripts/ingestion/sovereign_sync_engine.py --seed` | PERFECT | `TaiwanStockInfo` 3403 UPSERT；FRED 3885 UPSERT；總寫入 7288。 |
| 7A | `python scripts/core/core_universe_builder.py --dry-run --as-of-date 2026-05-14` | WARNING | Snapshot=`core_universe_20260514_core_universe_policy_v0_2`；v0.2 contract PASS/WARN/FAIL=10/10/0；coverage 仍不足但無 FAIL。 |
| 7B | `python scripts/core/core_universe_builder.py --commit --as-of-date 2026-05-14` | WARNING | committed v0.2 snapshot；written_rows=5601；WARNING 來自歷史資料 coverage 不足與 latest_registry_fallback。 |
| 8 初跑 | `python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14` | FAILED | 報告：`reports/core_universe_audit_20260516_2104.md`；預設仍指向 `core_universe_policy_v0.1`，命中 superseded snapshot；FAIL=2。 |
| 8 v0.2 指定重跑 | `python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14 --policy-version core_universe_policy_v0.2` | FAILED | 同報告檔被覆寫；FAIL=4；auditor 仍用 v0.1 pending-score 規則驗 v0.2 six-layer scores。 |
| 修正 | `scripts/maintenance/audit_core_universe.py` | APPLIED | 將 default policy 改為 v0.2；v0.2 允許 six-layer score columns 與 `score_scope=v0.2_six_layer`；lifecycle 接受 `core_universe_builder_v0.2` 且 status 可為 success/warning。 |
| 8 修正後 | `python -m py_compile scripts/maintenance/audit_core_universe.py` | PASS | 語法檢查通過。 |
| 8 修正後 | `python scripts/maintenance/audit_core_universe.py --as-of-date 2026-05-14` | PERFECT | 報告：`reports/core_universe_audit_20260516_2105.md`；Snapshot=`core_universe_20260514_core_universe_policy_v0_2`；PASS=36/WARN=0/FAIL=0。 |
| Final | `python scripts/core/db_utils.py` | PERFECT | §6.7 核心資產數=150。 |
| Final-HUB | `python scripts/core/__init__.py` | PERFECT | 四層動態稽核通過；§6.7 core+convex=150。 |
| §8 DDL | `python scripts/core/feature_store_schema.py --init` | PERFECT | 初次 sandbox 執行因 DB 連線限制失敗；非 sandbox 重跑成功；feature_store 三表建立完成。 |
| Step 9 dry-run | `python scripts/core/feature_store_builder.py --dry-run --as-of-date 2026-05-14` | PERFECT | 初次 sandbox 執行因 DB 連線限制失敗；非 sandbox 重跑成功；27 features × 150 stocks；would write 1950 feature rows；null_imputed=1050。 |
| Step 9 commit | `python scripts/core/feature_store_builder.py --commit --as-of-date 2026-05-14` | PERFECT | `feature_store_snapshot` 1；`feature_definition` 27；`feature_values` 1950。 |
| Step 10 | `scripts/core/model_trainer.py` | MISSING | 憲章 §8 草案已有條目，但程式檔不存在，無法執行訓練。 |
| Step 11 | `scripts/core/prediction_engine.py` | MISSING | 憲章 §8 草案已有條目，但程式檔不存在，無法執行推論。 |
| Step 11A | `scripts/maintenance/audit_leakage.py` | MISSING | 憲章 §8 草案已有條目，但稽核程式檔不存在，無法執行 leakage audit。 |
| Final-DB-Audit | `python scripts/maintenance/audit_supply_chain.py --db-only --include-logs` | FAILED | 報告：`reports/compliance_audit_20260516_2108.md`；PASS=21/WARN=0/FAIL=1；原因為本輪修正前產生兩筆 `audit_core_universe_v0.1=failed` lifecycle，污染 24 小時 log window。 |

## 最終 DB 實況

```text
TaiwanStockInfo             2799
FredData                    3885
core_universe_snapshot      2
core_universe_membership    5598
core_universe_scores        5598
feature_store_snapshot      1
feature_definition          27
feature_values              1950
```

Committed §6.7 分層：

```text
core_universe          120
convex_universe         30
research_universe     2271
quarantine_universe    378
```

## 問題紀錄與裁決

1. **DB 起始狀態非全空**：Step 1 已可 DB-linked，Step 2C 初次即顯示核心資產數 150。這與「database all table 已刪除」不一致。本輪未自行 drop tables。
2. **`audit_core_universe.py` 與 v0.2 builder 漂移**：auditor 原本預設 v0.1 policy，且以 v0.1 pending-score 規則驗 v0.2 six-layer scores。已修正並重跑通過。
3. **Step 9 sandbox DB 連線限制**：`feature_store_schema.py` / `feature_store_builder.py` 在 sandbox 內出現 `DB connection failed for host=127.0.0.1`；非 sandbox 重跑通過。裁決為執行環境限制。
4. **Step 9 可執行且通過**：Feature Store DDL 與 builder v0.1 已可執行，commit 後產生 27 個 feature definitions 與 1950 筆 feature values。
5. **Step 10 / Step 11 / Step 11A 尚未落地**：`model_trainer.py`、`prediction_engine.py`、`audit_leakage.py` 不存在。§8 仍只能停留在草案，不可升 v5.4.23。
6. **Final DB-only audit with logs 失敗**：原因不是現行 schema 失敗，而是本輪修正前留下的 `audit_core_universe_v0.1=failed` lifecycle rows。依既有 §3 log-window 規則，24 小時內失敗 log 會使 `--include-logs` 驗收 FAILED。需等待視窗外、或在明確授權的非生產環境中清理該 observation window 後再做正式 final audit。

## 本輪結論

- 上游 Step 1〜8 已可在修正後收斂至 PERFECT。
- §8 Step 9 Feature Store 已可 dry-run + commit 並 PERFECT。
- §8 Step 10 / Step 11 / Step 11A 因模組缺失無法執行。
- 本輪不符合「完全從空 DB」的實證前提，因初始 DB 仍保留核心股治理資料；但已依現有 DB 狀態完成重建序列與問題紀錄。
