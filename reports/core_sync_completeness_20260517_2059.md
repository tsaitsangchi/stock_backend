# Core Universe Sync 完整性驗證紀錄（第 2 輪）— 2026-05-17 20:59

> 依憲章 v6.0.0-patch §6.8.7 / §6.8.8 / §6.8.8-A / §6.8.8-B 執行。本輪為 §6.8.8-B 入憲後首次完整循環，做為**憲章 ↔ 程式碼一致性**檢驗。

---

## 0. 執行概要

- snapshot：`core_universe_20260514_core_universe_policy_v0_2`（committed）
- 命令：`venv/bin/python scripts/ingestion/sovereign_sync_engine.py --universe core --days 7`
- 引擎：`sovereign_sync_engine.py v1.11a`
- 時點：2026-05-17 20:58（Sun，盤後）
- 最近交易日：2026-05-15
- 本輪 vs 上輪基線（20:46）：**byte-identical**

## 1. 對映 §6.8.8-B (V) 基線

| 維度 | 基線（§6.8.8-B V）| 本輪實測 | 狀態 |
|---|---|---|---|
| `full_universe_size` | 150 | 150 | ✅ |
| `|zombies|` | 3 | 3 | ✅ |
| `healthy_universe_size` | 147 | 147 | ✅ |
| `structural_na_dict.size` | 9 | 9 | ✅ |
| Price/PriceAdj/Inst/Share/PER coverage | 147/147 | 147/147 | ✅ |
| Margin effective coverage | 139/139 | 139/139 | ✅ |
| FRED 4 序列 | 各自頻率 OK | 各自頻率 OK | ✅ |
| 同步主權判定 | WARNING（20 WARN）| WARNING（20 WARN）| ⚠️ 完全相同 |
| `HEALTHY_UNIVERSE_COVERAGE_OK` | True | **True** | ✅ |

## 2. 同步引擎輸出（與上輪對照）

| 項目 | 第 1 輪 20:45 | 第 2 輪 20:46 | 第 3 輪 20:58 |
|---|---|---|---|
| success | 4 | 4 | 4 |
| warning | 20 | 20 | 20 |
| failed | 0 | 0 | 0 |
| skipped | 580 | 580 | 580 |
| rows | 48,862 | 48,862 | 48,862 |
| 耗時 | 29.47 s | 27.x s | 30.13 s |

**裁決**：三輪 sync 之 success/warning/failed/skipped/rows 完全相同；§7.5 resume + §7 throttle 行為具決定性。**穩定性本身即為一項實證**。

## 3. 本輪新發現問題（憲章後續修改依據）

### 問題 #7：§6.8.8-B 異常清單僅存於 charter markdown，未進入 DB / 程式碼（CRITICAL）

- **現象**：§6.8.8-B (II) 之 zombies = `{1701, 1729, 3559}` 與 structural NA dict（9 筆）只存在於 `reports/系統架構大憲章_v6.0.0.md` 第 1042-1052 行；DB 不存在 `structural_na_dict` 或 `universe_exclusion_list` 表；`sovereign_sync_engine.py` 與 `audit_core_universe.py` 內 grep 不到 `1701` / `zombie` / `structural_na` / `expected_empty` 任何字串
- **驗證**：
  ```
  SELECT EXISTS (... WHERE table_name='structural_na_dict')    -- false
  SELECT EXISTS (... WHERE table_name='universe_exclusion_list')   -- false
  grep "1701|zombie|structural_na" scripts/  -> 0 matches
  ```
- **後果**：
  - §6.8.8-B (III) 四條永久強制原則**無法機器強制執行**，僅為文件層級宣告
  - sync 引擎每日仍對 zombies 打 12 條無謂 API request（浪費 §7.6 A3 配額）
  - `audit_core_universe.py` 不會 flag committed snapshot 中之 zombie 成員
  - 下次年度重選若忘記從清單剔除，charter ↔ code 漂移將靜默累積
- **建議入憲修改**：
  1. 新增 §6.8.8-C「Anomaly Registry as Code」：將 §6.8.8-B (II) 之清單存入 DB 表 `universe_anomaly_registry`（schema：`anomaly_class CHAR(1)`, `stock_id`, `dataset`, `effective_from`, `effective_to`, `reason`, `committed_by`, `audit_trail_ref`）
  2. `sovereign_sync_engine.py` 啟動時 SELECT 此表，對命中 `class='A'` 或 `(class='D', dataset=current)` 之組合 short-circuit
  3. `audit_core_universe.py` 新增 `check_zombie_in_committed_snapshot()`：若 committed snapshot 之 `core_universe`/`convex_universe` membership 含 `class='A'` 標的且 effective，回報 WARN
  4. `audit_doctrine_compliance.py` 新增 P2 支柱檢驗：`universe_anomaly_registry` 內容須與 charter §6.8.8-B (II) 清單一致

### 問題 #8：`audit_doctrine_compliance.py` 未涵蓋 §6.8.8-B（HIGH）

- **現象**：§6.8.8-B 為新入憲條文，但 `audit_doctrine_compliance.py` 的 P1/P2/P3/P4 支柱檢驗（基礎模式 14 項）未對映 §6.8.8-B 任一條原則
- **後果**：§0.7「升版提案必須附 §0 四大支柱治理檢驗報告；無法明示對映即不得進入正式 review」之治權對 §6.8.8-B 無法執行；下次 v6.0.0-patch → v6.1.0 升版時，§6.8.8-B 之合規性無法被機器驗證
- **建議入憲修改**：
  1. `audit_doctrine_compliance.py` 之 P2 支柱（八二槓鈴）新增 1 項：`check_universe_anomaly_registry_alignment()` — 比對 DB `universe_anomaly_registry` 與 §6.8.8-B (II) charter 清單；偏離即 FAIL
  2. 基礎檢驗從 14 項升至 15 項
  3. `for-promotion` 模式對 v6.1.0+ 必須包含此檢驗

### 問題 #9：sync 引擎 WARN 噪音持續（confirmed from last round）

- **狀態**：上輪 (20:46) 已記錄為「問題 #5」；本輪 (20:58) 三次連續 sync 完全相同的 20 條 WARN 確認**未實作**
- **建議入憲修改**：與問題 #7 合併實作 — 一旦 `universe_anomaly_registry` 落地，sync 引擎對 short-circuit 之組合不報 WARN（僅寫 INFO），sovereign 判定改以「非 registry 之 warning 計數 > 0」為 WARNING 觸發條件

### 問題 #10：核心股 universe-wide completeness probe 仍為手動執行（confirmed from last round）

- **狀態**：上輪「問題 #6」未實作；目前每輪都由 Claude 手動跑 SQL probe，無自動化 SOP
- **建議入憲修改**：在 §6.8.7-A cron 排程建議中加入第四條：
  ```
  # 每交易日 16:30（core sync 完成後 30 分鐘）
  30 16 * * 1-5  cd /home/hugo/project/stock_backend && venv/bin/python scripts/maintenance/check_universe_completeness.py --universe core --apply-registry >> logs/universe_completeness.log 2>&1
  ```
  新增 `scripts/maintenance/check_universe_completeness.py` 為 §6.8.8-B (V) 之程式碼載體；產出 `reports/universe_completeness_<timestamp>.md`

### 問題 #11：穩定性反成為盲點（MEDIUM，新觀察）

- **現象**：三輪 sync byte-identical（success/warning/failed/skipped/rows 完全相同），表面是極佳的決定性；但若**真實上游發生小型 regression**（例如 FinMind 某天某股回 0 rows），會被現有 20 條 noise WARN 淹沒，操作員難以察覺
- **後果**：noise floor 升高，operator 對 WARNING 判定逐漸鈍化
- **建議入憲修改**：與問題 #7 / #9 連動 — registry 落地後 sovereign 判定純度恢復，新增之 WARN 必為真實 regression；同時 `audit_supply_chain.py` 應新增「sync output diff vs last run」之檢驗，偵測 row count / warning set 之異常飄移

## 4. 整體裁決

| 項目 | 結論 |
|---|---|
| 同步動作 | 完成，與 §6.8.8-B (V) 基線 byte-identical |
| Healthy universe 完整性 | 147/147 OK；Margin effective 139/139 OK；FRED 4 序列 OK |
| `HEALTHY_UNIVERSE_COVERAGE_OK` | **True** |
| 本輪新發現問題 | 5 項（#7〜#11），其中 #7 #8 為 CRITICAL / HIGH |
| 主要結論 | §6.8.8-B 完整性驗收口徑正確；下一步為 §6.8.8-C「Anomaly Registry as Code」憲章化 + 程式碼落地 |

## 5. 後續行動

1. 提交本輪紀錄至 git audit trail
2. 起草 §6.8.8-C 與 `audit_doctrine_compliance.py` P2 第 5 項條文
3. 與既有 §6.8.6 special_rebalance / §7.6 / §8 imputation 保持相容
4. 不熱修改 committed snapshot；registry 表為新增之 DB 物件，不影響 §6.7 SQL SSOT
