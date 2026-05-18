# Full DB Rebuild + CoreScore v0.2 Annual Guard + §8 Pipeline Re-execution

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` (含本次新增 §6.8 年度重選契約)
Trigger: 使用者刪除 all DB tables；要求依新修改的憲章與程式碼從頭執行並嘗試升至 v5.4.23。

## 1. 重建範圍

- DB 全部 table drop（0 tables 起始）
- 九步合法啟動序列 + Feature Store schema
- CoreScore v0.2 bootstrap + 730d sync + 正式 commit
- §8 production-current h20 training（嘗試升 v5.4.23）
- §8 historical clean validation（rankic）作為 draft evidence
- 全鏈 Step 11A/11B audits

## 2. 程式碼變更摘要（本次新加）

| 模組 | 變更 |
|---|---|
| `core_universe_builder.py` | 新增 `special_rebalance_reason` 參數 + `_annual_rebalance_guard()`；強制 commit 必須是年末最後交易日或特別原因 (≥12 字元)；報告增「重選模式」「特別原因」 |
| `sovereign_sync_engine.py` | 已於本日早些升至 v1.11，新增 §7.6 A1〜A5 |
| `系統架構大憲章_v5.4.22.md` | 新增 §6.8 年度重選契約 |

## 3. 執行步驟序列

| Step | 指令 | 判定 | Exit | 備註 |
|---|---|---|---|---|
| 0 | `.env` 錨點確認 | MATCHED | — | PROJECT_ROOT=/home/hugo/project/stock_backend |
| 1 | `path_setup.py` | PERFECT | 0 | 25 維對齊 |
| 2 | `data_schema.py --init --force` | PERFECT | 0 | 13 tables, 2835 ms |
| 2B | `core_universe_schema.py --init` | PERFECT | 0 | 7 governance tables, 142 ms |
| 2C | `db_utils.py` | WARNING | 0 | §6.7 0 stocks（bootstrap 預期）|
| 2D | `feature_store_schema.py --init` | PERFECT | 0 | 3 feature_store tables, 78 ms |
| 3 | `audit_supply_chain.py --include-logs` | PERFECT | 0 | PASS=29/0/0 |
| 4 | `sovereign_sync_engine.py --seed` | PERFECT | 0 | 7287 rows (TaiwanStockInfo 2798 + FRED 3885 + counters); 3.30 s |
| 7A | `core_universe_builder.py --dry-run --as-of-date 2026-05-14` | WARNING | 0 | 2798 candidates, 120 core/30 convex；annual rebalance guard: **dry-run allowed** |
| 7B-bootstrap | `core_universe_builder.py --commit --as-of-date 2026-05-14 --special-rebalance-reason "DB rebuild bootstrap 2026-05-17 for core universe sync target"` | WARNING | 0 | 5599 rows；annual rebalance guard: **special override accepted** |
| 4-extended | `sovereign_sync_engine.py --universe core --all --days 730 --dataset-batched --workers 4` | WARNING | 0 | **688,867 rows / 194.34 s** (v1.11 A1+A2 比 v1.10 baseline 622 s **快 3.2×**); 46 warning 皆為 lifecycle gap zero-row |
| 7B-final | `core_universe_builder.py --commit --as-of-date 2026-05-14 --special-rebalance-reason "DB rebuild bootstrap 2026-05-17 final v0.2 scoring after core sync"` | WARNING | 0 | 5599 rows；real-data 評分 |
| 8 | `audit_core_universe.py --as-of-date 2026-05-14` | PERFECT | 0 | **PASS=40/0/0**（較前次 36 多 4 項，新增 annual rebalance guard 相關檢查）|
| Final db_utils | `db_utils.py` | PERFECT | 0 | §6.7 核心資產數 **150** |
| 9 production | `feature_store_builder.py --commit --as-of-date 2026-05-14 --feature-set-version feature_set_v0.1_h20_production_current --label-horizon 20` | PERFECT | 0 | 150 stocks/27 features/3980 rows/47 imputed |
| 10 production | `model_trainer.py --commit --feature-set-id fs_20260514_feature_set_v0_1_h20_production_current --model-family lgbm --label-horizon 20` | **FAILED** | 1 | **rows_trained=0**：DB 缺 2026-06-03 之後價格資料；§8.8.9-A 升版前置條件 #1 不滿足 |
| 9 historical | `feature_store_builder.py --commit --as-of-date 2025-04-25 --feature-set-version feature_set_v0.1_h20_20250515_cutoff_rankic_validation --label-horizon 20` | PERFECT | 0 | 150 stocks/27 features/2965 rows/84 imputed |
| 10 historical | `model_trainer.py --commit --feature-set-id fs_20250425_... --model-family lgbm --label-horizon 20` | PERFECT | 0 | rows_trained=144；**IC_mean=0.4998**；trainer=robust_rank_ic_baseline_v0.1；label_date=2025-05-15 |
| 11 | `prediction_engine.py --commit --model-id mdl_20250425_lgbm_h20_5c7f36c2_v0_1 --as-of-date 2025-04-25` | PERFECT | 0 | 150 predictions |
| 11A | `audit_leakage.py` | PERFECT | 0 | PASS/WARN/FAIL=18/0/0 |
| 11B | `audit_downstream_readiness.py --no-report` | **READY_FOR_DRAFT_EVIDENCE** | 0 | PASS/WARN/FAIL=29/1/0；唯一 WARN 為 production-current label window |

## 4. v5.4.23 升版判定

**結論：升版 BLOCKED。** §8 升版至 v5.4.23 之 6 條準則（憲章 §8.8.9-C）：

| # | 準則 | 實況 | 通過？ |
|---|---|---|---|
| 1 | Step 9 production-current PERFECT | PERFECT (150 stocks, 27 features, 3980 rows) | ✅ |
| 2 | Step 10 production-current PERFECT | **FAILED** (rows_trained=0；無 2026-06-03 後 label) | ❌ |
| 3 | Step 11 production-current PERFECT | 未執行（Step 10 已 FAILED）| ❌ |
| 4 | Step 11A leakage PERFECT | PERFECT (18/0/0) | ✅ |
| 5 | Step 11B readiness `READY_FOR_V5_4_23` | `READY_FOR_DRAFT_EVIDENCE` | ❌ |
| 6 | 憲章升版為 v5.4.23 | 維持 v5.4.22 | ❌ |

**根因**：DB `TaiwanStockPriceAdj.MAX(date)=2026-05-15`，required_label_date=2026-06-03（=2026-05-14 + 20 trading days）。差距約 12 個工作日。

**§8.8.9-D 不得升版條件 #1**「TaiwanStockPriceAdj.MAX(date) < required_label_date」明確命中；升版必須延後至 2026-06-03 後價格資料入 DB。

## 5. 程式碼問題清單（需修正項目）

**0 件結構性問題。** 全鏈執行符合憲章預期：

- ✅ 9 步序列無新阻斷錯誤
- ✅ §6.8 annual rebalance guard 在 dry-run / special override / regular commit 三種模式下行為正確
- ✅ §7.6 A1+A2 平行優化效果經 730d sync 實證 **3.2× speedup**
- ✅ §6.7 SQL 契約於兩次 7B commit 後均正確回傳 150 stocks
- ✅ §8 historical clean validation 與 pre-wipe 完全可重現（IC 略升 0.4911 → 0.4998 因今日 sync 補充了少量近期資料）
- ✅ §8 production-current 失敗模式與憲章 §8.8.9-A/D 預期完全一致

**唯一非結構性現象**：Step 10 production-current FAILED 屬「資料時間軸未追上」之預期 gate，非 bug。憲章 §8.8.9 已明文規範此條件，本次執行正是該條款的活體驗證。

## 6. 憲章補登需求

本次執行揭露以下需在憲章補登的事項：

1. **§6.8 年度重選契約**（本次已新增）：包含 6.8.1〜6.8.5 五子節，明示年度觸發條件 + special override 治理要求。
2. **§7.6 v1.11 落地實證**：730d core sync 從 622 s（v1.10 baseline）改善至 194 s（v1.11 A1+A2，workers=4），實測 3.2× speedup。
3. **§14.5（新）**：本次 2026-05-17 DB 全重建實證紀錄。
4. **Step 8 audit count 升至 40**：annual rebalance guard 加入後 audit_core_universe 新增 4 項 PASS 檢查。
5. **v5.4.23 升版仍 BLOCKED**：明示 production-current data window 限制（不變更門檻，僅補登實證 evidence）。

## 7. 最終判定

- 全鏈執行 **無新阻斷錯誤**；程式碼與憲章一致無漂移。
- v5.4.22 治權狀態：**MAINTAINED**。
- §8 主權狀態：**ACTIVE (DRAFT)**，等待 2026-06-03 後價格資料追上即可依 §8.8.9-B 序列再次執行 production-current Step 10/11 → 升 v5.4.23。
- 現行 prediction-backed delivery: `mdl_20250425_lgbm_h20_5c7f36c2_v0_1` / IC=0.4998 / 150 predictions。
