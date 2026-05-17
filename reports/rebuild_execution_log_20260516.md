# 九步全量重建結果 (2026-05-16)

| 步驟 | 指令 | 判定 | Exit |
|---|---|---|---|
| Step 0 | `.env` 錨點確認 | MATCHED | — |
| Step 1 | `path_setup.py` | PERFECT | 0 |
| Step 2 | `data_schema.py --init --force` | PERFECT | 0 |
| Step 2B | `core_universe_schema.py --init` | PERFECT | 0 |
| Step 2C | `db_utils.py` | WARNING | 0 |
| Step 3 | `audit_supply_chain.py --include-logs` | PERFECT | 0 |
| Step 4 | `sovereign_sync_engine.py --seed` | PERFECT | 0 |
| Step 7A | `core_universe_builder.py --dry-run` | WARNING | 0 |
| Step 7B | `core_universe_builder.py --commit` | WARNING | 0 |
| Step 8 | `audit_core_universe.py` | PERFECT | 0 |
| Final | `db_utils.py` (post-commit) | PERFECT | 0 |

**整體判定: PERFECT** — 無任何 FAILED，全序列 exit 0。

## 關鍵數據
*   **TaiwanStockInfo**: 3403 支股票已同步
*   **FredData**: 3885 rows (DFF/UNRATE/T10Y2Y/VIXCLS)
*   **Core Universe**: 120 支 core + 30 支 convex + 2271 研究池
*   **Final §6.7 核心資產數**: 150 支 (core + convex 合計)

## 已知 WARNING 說明 (皆為預期，非問題)
*   **Step 2C**: membership 空表 → 0 stocks（Step 7B 前正常）
*   **Step 7A/7B**: 個股行情/財務/籌碼資料尚未同步（`--seed` 只含 TaiwanStockInfo + FRED），v0.2 scoring 進入 WARNING 模式，v0.1 metadata bootstrap 正常完成

## 後續建議
執行 `sovereign_sync_engine.py --universe core` 同步 120 支 core 股個股資料，再重跑 Step 7A/7B 以取得完整 v0.2 scoring。
