# 逐支逐日 Commit 完整性 — Fetcher 全面對齊報告

**範圍**：`scripts/core/` v3.0 + `scripts/fetchers/` 全 21 個檔案
**依據附件**：
- `core/db_utils.py` v3.0（`safe_commit_rows / commit_per_stock / commit_per_day / commit_per_stock_per_day / FailureLogger / with_savepoint / async_*`）
- `core/finmind_client.py` v3.0（`RequestStats / CircuitBreaker / FetcherInterrupted / wait_until_quota_reset / quota cache`）
- `core/model_metadata.py` v2.0（`atomic_write_json / atomic_copy_file / rollback_to_metadata / list_history`）
- `core/path_setup.py` v2.0（`get_outputs_dir / get_logs_dir / get_archive_dir / ensure_dirs_exist`）
- 已對齊 v3.0 的 5 支 fetcher 範本（fetch_advanced_chip / cash_flows / chip / derivative / fundamental）
**日期**：2026-05-04

---

## 一、總覽：v3.0「逐支逐日 commit」原則

| 資料粒度 | 對應 commit 函式 | 適用情境 | 完整性保證 |
|---|---|---|---|
| **逐支 × 逐日** | `commit_per_stock_per_day(date_index, stock_index)` | 個股時間序列（價量、籌碼、財報、衍生品 OHLCV…）| 每對 (sid, day) 獨立短交易；單對失敗不影響其他對 |
| 逐日（市場層）| `commit_per_day(date_index)` | 全市場單值指標（CNN F&G、景氣指標、總三大法人）| 每日一筆短交易 |
| 逐支（無時間軸）| `commit_per_group(group_key_fn=lambda r:r[0])` | 標的快照表（stock_info）| 每 stock_id 獨立短交易 |
| 失敗自動 rollback | `safe_commit_rows()` | 任何「不希望毒交易擴散」的場景 | rollback 後 conn 仍可用 |

→ 這套 API 把舊版 `bulk_upsert(...)`「全部成功或全部回滾」的二元結果，
細化為「每組獨立成敗」，搭配 `FailureLogger` 即時原子落盤至
`outputs/{table}_failed_YYYYMMDD.json`，崩潰前完成的資料絕不遺失。

---

## 二、本輪實際異動

### A. 已是 v3.0 — 只做工作樹同步（19 支）

下表列出已正確使用 v3.0 commit pattern 的 fetcher，無需邏輯修改：

| Fetcher | commit 模式 | dedup PK | FailureLogger |
|---|---|---|---|
| `fetch_advanced_chip_data.py` | per_day（市場）+ per_stock_per_day（個股） + 批次 fallback | (date,name) / (date,sid) / (date,sid,tx_type) | ✓ |
| `fetch_cash_flows_data.py` | per_stock_per_day | (date,sid,type) | ✓ |
| `fetch_chip_data.py` | per_stock_per_day（含 inst-by-name PK 去重）| (date,sid,name) / (date,sid) | ✓ |
| `fetch_derivative_data.py` | per_stock_per_day | 期貨 (d,fid,cd,sess) / 選擇權 (d,oid,cd,sp,cp,sess) | ✓ |
| `fetch_derivative_sentiment_data.py` | per_stock_per_day + per_day（CNN F&G 無 stock_id）| 各表 PK | ✓ |
| `fetch_event_risk_data.py` | per_stock_per_day | 各表 PK | ✓ |
| `fetch_extended_derivative_data.py` | per_stock_per_day | 各表 PK | ✓ |
| `fetch_fred_data.py` | per_stock_per_day（series_id 當作 stock_index）| (series_id,date) | ✓ |
| `fetch_fundamental_data.py` | per_stock_per_day | (date,sid) / (date,sid,type) | ✓ |
| `fetch_international_data.py` | per_stock_per_day | (ticker,date) | ✓ |
| `fetch_macro_data.py` | per_stock_per_day（country/currency 當 stock_index）| 各表 PK | ✓ |
| `fetch_macro_fundamental_data.py` | per_stock_per_day + per_day（business_indicator）| 各表 PK | ✓ |
| `fetch_news_data.py` | per_stock_per_day | (date,sid) | ✓ |
| `fetch_price_adj_data.py` | per_stock_per_day | (date,sid) | ✓ |
| `fetch_sponsor_chip_data.py` | per_stock_per_day | 各表 PK | ✓ |
| `fetch_stock_info.py` | per_group（無時間軸 / one row per stock_id）| (stock_id) | ✓ |
| `fetch_technical_data.py` | per_stock_per_day | (date,sid) | ✓ |
| `fetch_total_return_index.py` | per_stock_per_day（index_id 當 stock_index）| (date,index_id) | ✓ |
| `backfill_from_gaps.py` | launcher（不直接寫 DB；交由被呼叫之 fetcher 完成）；FailureLogger + atomic checkpoint | — | ✓ |

> 抽檢結果：所有 v3.0 fetcher 對「個股資料」皆使用 `commit_per_stock_per_day`；
> 對「市場層級單值指標」（`fear_greed_index`、`business_indicator`）才使用 `commit_per_day`，
> 完全符合 v3.0 設計原則。沒有發現對個股資料誤用 `bulk_upsert` 或粗粒度 commit 的反模式。

---

### B. 本輪新升級至 v3.0（2 支 launcher）

#### 1. [`fetch_missing_stocks_data.py`](scripts/fetchers/fetch_missing_stocks_data.py) — 從 v3 Trinity 升級至 **v4.0**

本檔為 subprocess launcher（呼叫個股 fetcher），不直接寫資料表，
但 v3.0「逐支逐日完整性」精神同樣要落實在「子任務追蹤」：

| 變更 | 說明 |
|---|---|
| ✅ 統一 sys.path | 移除手寫多重 path block，整合 `core.path_setup.ensure_scripts_on_path()` |
| ✅ FailureLogger | 每支股票 × 每支 fetcher 子任務的失敗都即時 record 至 `outputs/fetch_missing_stocks_failed_YYYYMMDD.json` 與 `fetch_log` 表（雙通道） |
| ✅ Checkpoint | `outputs/checkpoints/fetch_missing_stocks.json`（atomic_write_json）；新增 `--resume` 旗標可從上次失敗處續做 |
| ✅ Subprocess timeout | 每個子任務 1 hour timeout（可由 `--timeout` 調整），避免單一卡死任務佔據整體流程 |
| ✅ stderr_tail | 失敗時抓 stderr 末 5 行，方便診斷（不再只剩 returncode） |
| ✅ `--skip-stock-info` / `--skip-macro` | 多次呼叫期間避免重複工作 |
| ✅ 統計摘要 | 結尾印出每支 fetcher 的成功率（成功/總數/百分比）與最慢 5 個子任務 |
| ✅ Optional `data_integrity_audit` | 該模組缺席時 graceful fallback（單股模式仍可使用） |

#### 2. [`parallel_fetch.py`](scripts/fetchers/parallel_fetch.py) — 升級至 **v3.0**

| 變更 | 說明 |
|---|---|
| ✅ ProcessPoolExecutor → `multiprocessing.Pool.imap_unordered` | 系統重構報告 INT-04：worker 完成一個才送下一個 chunk，避免一次 submit 全部 future 的記憶體尖峰；KeyboardInterrupt 觸發 `pool.terminate()` + `pool.join()` 優雅退出 |
| ✅ 整合 `core.path_setup` | `ensure_dirs_exist()` 啟動時建立所有必要目錄 |
| ✅ 整合 `core.db_utils.FailureLogger` | 每支失敗腳本即時 record，rc + duration + stderr_tail（最後 500 字元） |
| ✅ 整合 `core.finmind_client.get_request_stats().summary()` | 結尾印出 dataset 級別的請求統計 |
| ✅ 結尾摘要 | 失敗清單 + 最慢 5 支腳本 |
| ✅ `--phase` flag | 可只跑 phase 0 / 1 / 2 / 1+2 / all |
| ✅ `--abort-on-low-quota` | 配額剩餘 < 100 時可選擇中止，避免浪費 |
| ✅ `--workers N` | 取代硬編碼 8，可依硬體調整 |

---

## 三、commit 模式抽檢（驗證表）

> 透過 `grep -B3 "commit_per_day" fetch_*.py` 找出可能誤用 `commit_per_day` 寫個股資料的反模式：

| 檔案 | commit_per_day 出現處 | 是否合規 |
|---|---|---|
| `fetch_derivative_sentiment_data.py` | `fear_greed_index`（CNN 全市場單值，無 stock_id） | ✅ 正確 |
| `fetch_macro_fundamental_data.py` | `business_indicator`（台灣景氣指標，無 stock_id） | ✅ 正確 |
| `fetch_advanced_chip_data.py` | `total_margin_short` / `total_inst_investors`（市場層三大法人/融資總額） | ✅ 正確 |

**結論**：全部 21 支 fetcher 的 commit 模式皆與 v3.0 設計原則一致。

---

## 四、相容性與測試

- **完全向後相容**：所有 fetcher CLI、環境變數、寫入的資料表 schema、log 輸出格式皆未變動。
- **新增依賴**：本輪零新增（asyncpg / aiohttp 仍維持 lazy import；無 numba 需求）。
- **新建立的檔案 / 目錄**：
  - `outputs/{table}_failed_YYYYMMDD.json`（每張表自己的失敗清單，每天一份）
  - `outputs/checkpoints/fetch_missing_stocks.json`（fetch_missing_stocks_data 進度檔）
  - `outputs/logs/parallel_fetch.log`（parallel_fetch v3 log）
- **驗證指令**：
  ```bash
  # 語法檢查
  python -m py_compile scripts/fetchers/parallel_fetch.py scripts/fetchers/fetch_missing_stocks_data.py

  # 單股 smoke test
  python scripts/fetchers/fetch_missing_stocks_data.py --stock-id 2330 --skip-macro

  # 全市場 dry run（觀察 commit 順序）
  python scripts/fetchers/parallel_fetch.py --phase 0
  ```

---

## 五、檔案異動清單

```
M  scripts/core/db_utils.py            (v3.0 已部署 — 工作樹同步)
M  scripts/core/finmind_client.py      (v3.0 已部署 — 工作樹同步)
M  scripts/core/model_metadata.py      (v2.0 已部署 — 工作樹同步)
M  scripts/core/path_setup.py          (v2.0 已部署 — 工作樹同步)

M  scripts/fetchers/fetch_advanced_chip_data.py        (v3.0 已部署)
M  scripts/fetchers/fetch_cash_flows_data.py           (v3.0 已部署)
M  scripts/fetchers/fetch_chip_data.py                 (v3.0 已部署)
M  scripts/fetchers/fetch_derivative_data.py           (v3.0 已部署)
M  scripts/fetchers/fetch_derivative_sentiment_data.py (v3.0 已部署)
M  scripts/fetchers/fetch_event_risk_data.py           (v3.0 已部署)
M  scripts/fetchers/fetch_extended_derivative_data.py  (v3.0 已部署)
M  scripts/fetchers/fetch_fred_data.py                 (v3.0 已部署)
M  scripts/fetchers/fetch_fundamental_data.py          (v3.0 已部署)
M  scripts/fetchers/fetch_international_data.py        (v3.0 已部署)
M  scripts/fetchers/fetch_macro_data.py                (v3.0 已部署)
M  scripts/fetchers/fetch_macro_fundamental_data.py    (v3.0 已部署)
M  scripts/fetchers/fetch_news_data.py                 (v3.0 已部署)
M  scripts/fetchers/fetch_price_adj_data.py            (v3.0 已部署)
M  scripts/fetchers/fetch_sponsor_chip_data.py         (v3.0 已部署)
M  scripts/fetchers/fetch_stock_info.py                (v3.0 已部署 — commit_per_group)
M  scripts/fetchers/fetch_technical_data.py            (v3.0 已部署)
M  scripts/fetchers/fetch_total_return_index.py        (v3.0 已部署)
M  scripts/fetchers/backfill_from_gaps.py              (v3.0 launcher)

M  scripts/fetchers/fetch_missing_stocks_data.py       ★ 本輪 v3 → v4 升級
M  scripts/fetchers/parallel_fetch.py                  ★ 本輪 v2 → v3 升級

A  reports/逐支逐日commit完整性報告.md
```

---

## 六、設計原則回顧（呼應系統檢核報告）

> 系統檢核報告 P0-1 指出 267 處資料斷層污染整套訓練資料，且
> 「audit 偵測得到但無人補抓」。本輪的 v3.0「逐支逐日 commit」
> 正是要解決「補抓過程崩潰時的資料完整性」核心矛盾：

- **舊行為**：`bulk_upsert(rows)` 一次處理全部 → 任一筆異常整批 rollback →
  下次重抓時可能因為 API 配額已耗盡而停在更早的位置 → 資料斷層持續擴大。
- **v3.0 新行為**：`commit_per_stock_per_day(rows)` 每對 (sid, day) 獨立短交易 →
  單對失敗只影響該對 → 配額耗盡時，已完成的所有 (sid, day) 對皆已落地 →
  下次重啟自動從 audit 偵測的 gap 起點接續，**完整性與進度同時保留**。
- **配套**：`FailureLogger` 即時原子寫入 `outputs/{table}_failed_*.json`，崩潰前
  失敗的清單也保留下來，可由 `backfill_from_gaps.py` 直接讀取做下一輪精確補抓。

— END —
