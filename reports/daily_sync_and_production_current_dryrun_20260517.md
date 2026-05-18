# Daily Sync 維護 + Production-Current Dry-Run 排演

Date: 2026-05-17
Constitution: `系統架構大憲章_v5.4.22.md` §7 + §8.8.9
Purpose: 執行 §8 升版前的等待期維護作業 — 每日 sync 保持 DB 最新 + 預先 dry-run 排演 §8.8.9-B 序列。

## 1. Track A — Daily Sync 維護

Command:

```bash
python scripts/ingestion/sovereign_sync_engine.py --universe core --days 7
```

Result:

- success: 4
- warning: 20
- failed: **0**
- skipped: 580
- 402-recovered: 0
- 總寫入: 3,885 rows
- 耗時: 23.31 s
- sovereign verdict: WARNING

Sync 行為解讀:

- 580 skipped 皆為 §7.5 L3 DB-driven resume：DB 已有 ≥ 2026-05-10 資料，無需重抓。
- 20 warning 皆為個別 stock/table API 回傳 zero rows（lifecycle gap stock 與 ETF 無融資融券屬正常）。
- 4 success 為 FRED 四序列（DFF 1000 / UNRATE 939 / T10Y2Y 958 / VIXCLS 988）。
- 0 failed 確認 §7 三層防禦（節流 5500/hr + 退避 [30s, 300s, 1800s] + DB resume）運作正常。

## 2. §8.8.9-A Precheck 結果

Precheck command output:

```text
max_price_date 2026-05-15
latest_snapshot ('core_universe_20260514_core_universe_policy_v0_2', datetime.date(2026, 5, 14))
```

Gate 判定:

| 條件 | 期望 | 實況 | 通過？ |
|---|---|---|---|
| `TaiwanStockPriceAdj.MAX(date) >= 2026-06-03` | ≥ 2026-06-03 | 2026-05-15 | ❌ |
| latest committed snapshot = `core_universe_20260514_core_universe_policy_v0_2` | matched | matched | ✅ |

**裁決**: production-current 升版仍 BLOCKED；前置條件 1 未滿足，須等待至少 12 個交易日的價格資料追上。

## 3. Track B — Production-Current Dry-Run 排演

Command (依憲章 §8.8.9-B):

```bash
python scripts/core/feature_store_builder.py --dry-run \
  --as-of-date 2026-05-14 \
  --feature-set-version feature_set_v0.1_h20_production_current \
  --label-horizon 20
```

Result:

- feature_set_id: `fs_20260514_feature_set_v0_1_h20_production_current`（符合 §8.8.9-B 預期）
- feature_set_version: `feature_set_v0.1_h20_production_current`
- universe_snapshot_id: `core_universe_20260514_core_universe_policy_v0_2`（鎖定當期 v0.2 snapshot）
- as_of_date: 2026-05-14
- label_horizon: 20
- stocks scored: 150
- features defined: 27
- value rows: 3,980
- null imputed: 47
- preflight PASS/WARN/FAIL: **14/0/0**
- warnings: 0
- failed: 0
- 耗時: 360.53 ms
- verdict: **PERFECT**

Dry-Run 排演結論:

- §8.8.9-B Step 9 指令字串無漏字、無漂移。
- builder 正確鎖定當期 CoreScore v0.2 universe，未漂移至舊 v0.1 snapshot。
- 150 stocks、27 features、3,980 rows 與 §8.8.11 dry-run 對齊（小幅差異於 imputed 數量 47 vs 47，因兩次 dry-run 間 daily sync 已刷新部分 institutional 資料）。
- DB 未寫入任何 governance row（dry-run 邊界正確）。
- 此次 dry-run **不 commit**；待 2026-06-03 後 production-current 序列正式啟動時，再以同一指令改 `--commit` 落地。

## 4. SOP 漏洞掃描結果

依本次 dry-run 對 §8.8.9-B 第 Step 9 條目進行字面對照:

- ✅ CLI 旗標 `--commit/--dry-run/--as-of-date/--feature-set-version/--label-horizon` 全部就位。
- ✅ feature_set_id 命名規則與憲章預期完全一致。
- ✅ universe_snapshot 鎖定為當期最新 committed v0.2。
- ✅ label_horizon=20 被正確攜帶至 snapshot 與 metrics。
- ⚠️ `source_cutoff=2026-05-16` 晚於 `as_of_date=2026-05-14`：屬合法 fallback，但 production-current commit 時 source_cutoff 預期將推進至 ≥ 2026-06-03。
- 未發現需修改的 SOP 漏洞。

## 5. 後續排程建議

1. 自動化每日 sync (建議 cron 或排程任務於每個交易日盤後)：
   ```bash
   python scripts/ingestion/sovereign_sync_engine.py --universe core --days 7
   ```
2. 每週驗證一次 §8.8.9-A precheck，確認 max_price_date 進度與 snapshot 未漂移。
3. 當 max_price_date ≥ 2026-06-03 滿足後，依 §8.8.9-B 5 步序列正式執行 production-current 升版。
4. 升版完成後憲章升至 v5.4.23，§8 由 DRAFT 改為強制契約。

## 6. 裁決

- Track A 維護: PERFECT 等價（WARNING exit 0 為 incremental sync 預期）。
- Track B SOP 排演: **PERFECT**（14/0/0）。
- §8 升版狀態: 仍 BLOCKED，但所有可預先驗證之治權條件皆已通過；剩餘唯一阻塞為 DB 價格資料尚未追上 required label window。
- §8 主權狀態: 維持 **ACTIVE (DRAFT)**。
