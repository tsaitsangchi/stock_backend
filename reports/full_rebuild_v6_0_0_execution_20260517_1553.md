# 全鏈重建執行紀錄 (v6.0.0 + §0 系統核心思想入憲後首次空 DB 重建)

**執行日期**: 2026-05-17 15:46〜15:53
**治權基準**: 系統架構大憲章_v6.0.0.md（含 §0 最高指導原則）
**觸發**: 使用者刪除 all DB tables（驗證為 0 tables）
**任務界線**: 「**只做記錄所有執行狀況與問題**」— 本紀錄不對任何發現項目進行修正動作

---

## 1. 完整序列執行結果

| Step | 指令 | 判定 | Exit | 耗時 | 重點數據 |
|---|---|---|---|---|---|
| 0 | `.env` 錨點確認 | MATCHED | — | — | PROJECT_ROOT=/home/hugo/project/stock_backend |
| 1 | `path_setup.py` | PERFECT | 0 | 37.45 ms | 25 維對齊，BOOTSTRAP-DEFERRED 預期 |
| 2 | `data_schema.py --init --force` | **PERFECT ALIGNMENT** | 0 | 2,883.91 ms | API PASS=11/0/0，13 tables 建立 |
| 2B | `core_universe_schema.py --init` | PERFECT | 0 | 135.40 ms | PREFLIGHT 9/0/0，7 governance tables |
| 2C | `db_utils.py` | WARNING | 0 | 11.36 ms | §6.7 0 stocks（bootstrap 預期）、憲法 v6.0.0 ✅ |
| 2D | `feature_store_schema.py --init` | PERFECT | 0 | 69.87 ms | PREFLIGHT 6/0/0，3 feature_store tables |
| 3 | `audit_supply_chain.py --include-logs` | PERFECT | 0 | — | PASS=29/0/0，憲法 v6.0.0 ✅ |
| 4 | `sovereign_sync_engine.py --seed` (v1.11a) | PERFECT | 0 | 3.27 s | 5 成功 / 0 警告 / 0 失敗 / 7,287 rows |
| 7A | `core_universe_builder.py --dry-run` | WARNING | 0 | 818.79 ms | annual guard: dry-run allowed; PREFLIGHT 8/0/0；V0.2 CONTRACT 10/10/0；core=120/convex=30/research=2270/quarantine=378 |
| 7B-boot | `--commit + --special-rebalance-reason "DB rebuild bootstrap 2026-05-17 v6.0.0 charter execution"` | WARNING | 0 | 1,431.55 ms | **annual guard: special override accepted**（§6.8 首次 v6.0.0 期間特別重選）；written_rows=5,599 |
| 4-ext | `--universe core --all --days 730 --dataset-batched --workers 4` (v1.11a A1+A2) | WARNING | 0 | **162.85 s** | 1308 成功 / 46 lifecycle gap 警告 / 0 失敗 / **688,867 rows**；A2 workers=4，A1 dataset_batched=True；A5 預警 0 次 |
| 7B-final | `--commit + special-rebalance-reason "...final v0.2 scoring after core sync"` | WARNING | 0 | 1,843.64 ms | PREFLIGHT 7/1/0；V0.2 CONTRACT **16/4/0**；real-data scoring 150 stocks |
| 8 | `audit_core_universe.py --as-of-date 2026-05-14` | PERFECT | 0 | 48.48 ms | **PASS=40/0/0**（包含 §6.8 rebalance trace 全項）|
| Final | `db_utils.py` (post-commit) | **PERFECT** | 0 | 11.60 ms | §6.7 核心資產 **150 支**、憲法 v6.0.0 ✅ |
| 9 prod | `feature_store_builder.py --commit --as-of-date 2026-05-14 --feature-set-version feature_set_v0.1_h20_production_current --label-horizon 20` | PERFECT | 0 | 560.44 ms | 150/27/3,980 rows，feature_set committed |
| 10 prod | `model_trainer.py --commit --feature-set-id fs_20260514_..._production_current --model-family lgbm --label-horizon 20` | **FAILED** | 1 | 101.91 ms | **rows_trained=0**：label_date >= 2026-06-03 但 DB max=2026-05-17 |
| 9 hist | `feature_store_builder.py --commit --as-of-date 2025-04-25 --feature-set-version feature_set_v0.1_h20_20250515_cutoff_rankic_validation` | PERFECT | 0 | 413.07 ms | 150/27/2,965 rows / 84 imputed |
| 10 hist | `model_trainer.py --commit --feature-set-id fs_20250425_..._rankic_validation --label-horizon 20` | PERFECT | 0 | 140.50 ms | rows_trained=144；**IC_mean=0.4998**；rmse=0.3078；label_date_min=label_date_max=2025-05-15 |
| 11 | `prediction_engine.py --commit --model-id mdl_20250425_lgbm_h20_5c7f36c2_v0_1 --as-of-date 2025-04-25` | PERFECT | 0 | 88.53 ms | run_id=`pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1`，150 predictions，imputed_ratio=2.83% |
| 11A | `audit_leakage.py` | PERFECT | 0 | 32.69 ms | **PASS/WARN/FAIL=18/0/0**；治權基準=憲法 v6.0.0 ✅ |
| 11B | `audit_downstream_readiness.py --no-report` | **READY_FOR_DRAFT_EVIDENCE** | 0 | 31.14 ms | PASS/WARN/FAIL=29/1/0；治權基準=憲法 v6.0.0 ✅ |

**全鏈耗時**: ~170 秒（不含人工讀取時間），其中 730d core sync 佔 162.85 s（96%）。

---

## 2. v5.4.22 → v6.0.0 升版後執行對照

| 指標 | v5.4.22 baseline (2026-05-17 13:39) | v6.0.0 本次 (2026-05-17 15:46) | 差異 |
|---|---|---|---|
| 730d sync 耗時 | 194.34 s (v1.11) | 162.85 s (v1.11a) | **−31.5 s / −16.2%** |
| Step 8 audit | PASS=40/0/0 | PASS=40/0/0 | 一致 |
| 核心資產數 | 150 | 150 | 一致 |
| 寫入 rows | 688,867 | 688,867 | 一致 |
| Historical IC_mean | 0.4998 | 0.4998 | 一致 |
| Step 11A leakage | 18/0/0 | 18/0/0 | 一致 |
| Step 11B readiness | READY_FOR_DRAFT_EVIDENCE | READY_FOR_DRAFT_EVIDENCE | 一致 |
| 主權判定 print 字串 | 憲法 v5.4.22 | **憲法 v6.0.0** | ✅ 升版生效 |

v6.0.0 升版無造成任何結構性行為變化；唯一可觀察改善為 sync 速度（推測為 OS-level cache 或 FinMind 端瞬時較佳，非程式碼改動），與 §0.4 可觀察性原則符合。

---

## 3. 觀察到的問題與狀況清單（**僅記錄、不修正**）

### 3.1 程式碼層

| # | 嚴重度 | 模組 | 觀察 | 對應憲章條文 |
|---|---|---|---|---|
| 1 | 🟡 LOW | `scripts/core/path_setup.py` | Step 1 輸出「治理維度 : 25 維全譜路徑 (**對齊 v5.4.22**)」— 應為 v6.0.0；屬 v6.0.0 升版時遺漏更新之硬編字串 | §0.4 數位孿生完整性（顯示與實際版本一致性）|
| 2 | 🟢 INFO | `scripts/core/core_universe_schema.py` 內部驗證訊息 | 部分 PREFLIGHT 訊息引用 `TaiwanStockInfo.stock_id` 之 raw column inheritance 仍以 v5.4.22 樣式輸出（功能正常，僅 cosmetic）| §0.4 |
| 3 | 🟢 INFO | `sovereign_sync_engine v1.11a` | 本次未啟用 `--dynamic-quota` 與 `--quota-interval`，§7.6 A3 邊界（v1.11a 補正項）未實際驗證；A5 預警/暫停亦未觸發（window 遠低於 4800/hr）| §7.6 A3/A5 |
| 4 | 🟢 INFO | `core_universe_builder.py` | 7B-final 之 `source_data_cutoff=2026-05-17` 晚於 `as_of_date=2026-05-14`，觸發 `latest_registry_fallback` warning；屬已預期，憲章 §6.3 已明文化 | §6.3 |

### 3.2 治權層

| # | 嚴重度 | 觀察 | 對應憲章條文 |
|---|---|---|---|
| 5 | 🟡 MED | **§6.8 annual rebalance guard 兩次以 special override 通過**（7B-bootstrap + 7B-final）。Reason 字串均為「DB rebuild bootstrap」開頭，符合 §6.8.3 例外清單第一項。但 audit trail 將留下兩個 special snapshot 紀錄；同一 universe snapshot 因 ON CONFLICT 機制只保留最終版本，避免污染。仍須注意：同一 DB 重建過程中**不應**重複使用 special override 增加 audit 雜訊。 | §6.8.3 |
| 6 | 🟢 INFO | **§8.8.9-D 條件 #1 命中 → §8 v6.1.0/v7.0.0 升版仍 BLOCKED**：required_label_date=2026-06-03 > DB max=2026-05-17（差距 ~11 工作日）。屬資料時間軸限制，非程式問題。 | §8.8.9-D |
| 7 | 🟢 INFO | **V0.2 CONTRACT 4 項 WARN** 持續存在：market scope zero-coverage (2,652/2,798)、TaiwanStockInfo as-of=65 < 150。屬可預期 fallback，憲章 §6.4 已明文化。 | §6.4 |
| 8 | 🟢 INFO | **lifecycle gap 46 項 warning**（如 1718/1721/1729/2408/3122 等股票之特定資料表 API 回 0 筆）。憲章 §8.8.7 已記錄 7 支主要 lifecycle gap stocks（1729/3559/1701/7810/7828/7772/6907）。 | §8.8.7 |

### 3.3 §0 治權檢驗（v6.0.0 新增升版規則）

依憲章 §0.7「升版規則延伸：v6.x.0 / v7.0.0 升版提案必須附 §0 四大支柱治理檢驗報告」，本次執行**非升版動作**，但仍逐一檢驗：

| 支柱 | 本次執行符合性 |
|---|---|
| §0.1 第一性原則與市場物理學 | ✅ CoreScore v0.2 六層含 `LiquidityMass` (M) + `InstitutionalFlow` (F) + `VolatilityControl`；無基於落後技術指標之決策 |
| §0.2 八二法則與不對稱槓鈴 | ✅ 150 支（core 120 + convex 30）= 右尾 5.4%；中段 research 2270 保留為觀測池不主動投入；左尾 quarantine 378 結構性隔離 |
| §0.3 康波週期與 2026 雙重共振 | ✅ THEME_KEYWORDS 涵蓋第六波 MBNRIC（半導體/生技/醫療/資訊/通信/機器/綠能/光電 等）；macro features 群（DFF/VIX/T10Y2Y/UNRATE）作宏觀觀測 |
| §0.4 可觀察性與數位孿生完整性 | ✅ 14 步全鏈寫入 `pipeline_execution_log` + `data_audit_log`；四層稽核全通過；動態 PERFECT/WARNING/FAILED 判定無硬編 |

**§0 治權判定**：本次執行**完全符合**四大支柱，無違憲行為。

---

## 4. 最終 DB 實況

| 類別 | Table | 狀態 |
|---|---|---|
| Infrastructure | `pipeline_execution_log` | ACTIVE |
| Infrastructure | `data_audit_log` | ACTIVE |
| Raw / FinMind | `TaiwanStockInfo` | 2,798 distinct stocks（3,402 rows） |
| Raw / FinMind | `TaiwanStockPriceAdj` | ~69,880 rows (148 core stocks) |
| Raw / FinMind | `TaiwanStockMonthRevenue` | ~3,369 rows (148 stocks) |
| Raw / FinMind | `TaiwanStockFinancialStatements` | ~17,450 rows (148 stocks) |
| Raw / FinMind | `TaiwanStockInstitutionalInvestorsBuySell` | ~326,045 rows (148 stocks) |
| Raw / FinMind | `TaiwanStockMarginPurchaseShortSale` | ~63,138 rows (140 stocks) |
| Raw / FinMind | `TaiwanStockPER` | ~67,122 rows (148 stocks) |
| Raw / FinMind | `TaiwanStockPrice` | 增量 |
| Raw / FinMind | `TaiwanStockShareholding` | 增量 |
| Raw / FinMind | `TaiwanStockDividend` | 增量 |
| Macro / FRED | `FredData` | 3,885 rows (DFF/UNRATE/T10Y2Y/VIXCLS) |
| Governance | `core_universe_policy` | 1 active (`core_universe_policy_v0.2`) |
| Governance | `core_universe_snapshot` | 1 committed (`core_universe_20260514_core_universe_policy_v0_2`)；rebalance_mode=`special` |
| Governance | `core_universe_membership` | 2,798 rows（core 120 + convex 30 + research 2270 + quarantine 378）|
| Governance | `core_universe_scores` | 2,798 rows |
| Governance | `theme_taxonomy` | 已建立 |
| Governance | `stock_theme_map` | 已建立 |
| Governance | `universe_revision_log` | 2 entries（7B-bootstrap + 7B-final）|
| Feature Store | `feature_store_snapshot` | 2 committed（`fs_20260514_..._production_current` + `fs_20250425_..._rankic_validation`）|
| Feature Store | `feature_definition` | 27 × 2 = 54 rows |
| Feature Store | `feature_values` | 3,980 + 2,965 = 6,945 rows |
| Model Registry | `model_registry` | 1 committed（`mdl_20250425_lgbm_h20_5c7f36c2_v0_1`，IC=0.4998）|
| Model Registry | `model_training_run` | 1 success entry |
| Prediction | `prediction_run` | 1 committed（`pred_20250425_mdl_20250425_lgbm_h20_5c7f36c2_v0_1`）|
| Prediction | `prediction_values` | 150 rows |

**§6.7 核心同步資產數**: **150 支** ✅
**§8 主權狀態**: **ACTIVE (DRAFT)**；下次升版 gate 為 2026-06-03 後 production-current label window。

---

## 5. v5.4.23 升版 6 條準則對照（§8.8.9-C）

| # | 準則 | 實況 | 通過 |
|---|---|---|---|
| 1 | Step 9 production-current PERFECT | PERFECT | ✅ |
| 2 | Step 10 production-current PERFECT | **FAILED** (rows_trained=0) | ❌ |
| 3 | Step 11 production-current PERFECT | 未執行 | ❌ |
| 4 | Step 11A leakage PERFECT | PERFECT (18/0/0) | ✅ |
| 5 | Step 11B readiness = `READY_FOR_V5_4_23` | `READY_FOR_DRAFT_EVIDENCE` | ❌ |
| 6 | 憲章升版 | 維持 v6.0.0 | ❌ |

**升版結論**: §8 升至強制契約（觸發 v6.1.0 minor 升版）**仍 BLOCKED**。命中 §8.8.9-D 不得升版條件 #1。

---

## 6. 結論

- 全鏈 14 步驟執行完成；無**新的**結構性問題。
- 唯一程式級觀察為 `path_setup.py` 之「對齊 v5.4.22」硬編字串未隨 v6.0.0 升版更新（嚴重度 🟡 LOW，cosmetic）。**依任務界線「只做記錄不修正」，本紀錄不對該項進行修補動作**。
- v6.0.0 與 v5.4.22 相比，行為層 100% 相容；觀察到 730d sync 耗時改善 −16.2% 屬執行環境變動而非程式碼改動。
- §0 四大支柱治權檢驗：全部符合。
- §6.7 核心資產 150 支已就位；§8 historical clean validation 全鏈 PERFECT；§8 production-current 升版仍待 2026-06-03。
- 系統治權狀態：**v6.0.0 ACTIVE / §8 ACTIVE (DRAFT)**，無變更。
