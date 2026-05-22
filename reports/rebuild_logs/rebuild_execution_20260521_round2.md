# DB Rebuild 執行紀錄 #2：從零 → 新 4 步序列（§14.7-AM 雞與蛋補強後）

- **開始時間**: 2026-05-21（第 2 輪）
- **觸發**: 使用者再次刪除 DB 所有 tables（驗證：0 tables）
- **依憲章**: v6.0.0-FINAL + v6.0.0-patch §14.7-AM 雞與蛋補強（commit `fea89bf`）
- **依程式**: sovereign_sync_engine.py v1.20 + core_universe_builder.py v0.2
- **目標 4 步序列**:
  - Phase 2: Step 0 → 1 → 2 → 2.5 → 2B → 2C → 3（schema + governance + 雙稽核）
  - Phase 3.I: Step 4 `--seed`
  - Phase 3.II: Step 4B bootstrap_init (`core_universe_builder --commit --special-rebalance-reason "..."`)
  - Phase 3.III: Step 4F `--universe full --all --special-full-market-reason "..."`
  - Phase 3.IV: Step 4B bootstrap_final
  - (Phase 3.V optional: Step 8 `--source fred`)

---

## 執行紀錄

### Phase 1: DB 狀態驗證 ✅
- 結果: 0 tables

### Phase 2: Step 1-3 重建 ✅
| Step | 程式 | 結果 |
|---|---|---|
| 1 | `path_setup.py v4.46` | PERFECT |
| 2 | `data_schema.py v2.16 --init --force` | PERFECT (13/13 DDL, 3.43s) |
| 2.5 | `audit_api_schema_compliance.py v0.3 --include-fred` | PERFECT |
| 2B | `core_universe_schema.py v0.3 --init` | PERFECT (7/7 tables) |
| 2C | `db_utils.py v2.47` | WARNING (合法 BOOTSTRAP — §6.7 0 rows) |
| 3 | `audit_supply_chain.py v1.19 --include-logs` | PERFECT (29/0/0) |

### Phase 3.I: Step 4 種子灌溉 ✅
- 程式: `sovereign_sync_engine.py v1.20 --seed`
- 結果: PERFECT (52,277 rows / 11.86s)
- 細項: TaiwanStockInfo 3,404 + FRED 4 序列 48,873

### Phase 3.II: Step 4B bootstrap_init ✅
- 程式: `core_universe_builder.py v0.2 --commit --as-of-date 2026-05-21 --special-rebalance-reason "DB rebuild bootstrap 2026-05-21 init"`
- 結果: **WARNING** (合法 — coverage=0 屬 0 OHLC bootstrap 預期)
- 細項:
  - 總候選: 2,767（as_of_filtered from TaiwanStockInfo）
  - **core_universe: 120**
  - **convex_universe: 30**
  - **research_universe: 2,239**
  - **quarantine: 378**
  - committed rows: 5,537
  - 耗時: 1.79s

#### 🟢 觀察 #3：bootstrap_init 之 universe 分配比預期合理
- 5 月雞與蛋補強治權範本明文「latest_registry_fallback mode」實際採用 `as_of_filtered` mode（程式輸出 `candidate_source_mode=as_of_filtered`）
- 結果合憲：120 core + 30 convex 之合計 150 與既有 v6.0.0 治權結構之 150 core+convex 一致
- coverage=0 屬合法（per §6.4 治權詮釋）

### Phase 3.III: Step 4F 全市場全天數（**正在背景執行中**）
- PID: `746736`
- 啟動: 2026-05-21 15:38
- 背景任務 ID: `bzgy8buwd`
- 主 log: `logs/full_market_sync_20260521_r2.log`
- 命令:
  ```bash
  sovereign_sync_engine.py --universe full --all --dataset-batched --workers 4 --dynamic-quota \
      --special-full-market-reason "DB rebuild bootstrap: 從零建立全市場全天數基準資料供 CoreScore v0.2 核心股選擇用"
  ```
- 治權確認: universe full = 2,767 stocks committed ✅
- 預期耗時: 6-10 小時級
- 預期 row 數: 數千萬 ~ 上億
- 後續動作: 完成後自動通知 → 執行 Phase 3.IV bootstrap_final + 雙稽核

#### 🟡 觀察 #4：Bash tool CWD 繼承不一致（操作面提示，非治權問題）

**現象**：
- `run_in_background=true` 啟動 nohup 進程繼承「上次 cd 過的 CWD」（main repo）
- 後續同步 Bash 查詢若**未明示 `cd /home/hugo/project/stock_backend`**，CWD 預設回到 worktree（primary）
- 造成第一次 `tail logs/full_market_sync_*.log` 找不到檔案，誤判「log path 異常」
- 實際 log 正確產生於主目錄 `/home/hugo/project/stock_backend/logs/full_market_sync_20260521_r2.log`（符合 CLAUDE.md §二 #1 治權慣例）

**根因**：Bash tool CWD 繼承策略對 `run_in_background=true` 與一般同步呼叫不一致

**操作面修正建議**（非治權問題；不需入憲）：
- 所有讀 log 之 Bash 指令統一加 `cd /home/hugo/project/stock_backend &&` 前綴
- 或改用絕對路徑 `tail /home/hugo/project/stock_backend/logs/...`



