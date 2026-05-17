"""
sovereign_sync_engine.py v1.11 (Quantum Finance Market Universe Seed Engine)
================================================================================
**最後更新日期**: 2026-05-17
**主權狀態**: SUPPLY CHAIN RATE SOVEREIGNTY ALIGNED + §7.6 A1〜A5 進階優化已實作 (憲法 v5.4.22 §7 對齊)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Market Universe Seed]: 第 4 步取得 `TaiwanStockInfo` 市場個股清單，並同步 FRED 核心宏觀資料。
2. [Schema Sovereignty]: 所有寫入欄位必須 100% 對齊 `data_schema.py v2.11` 的 `DATASET_REGISTRY`。
3. [Hybrid Observability]: 使用 `record_lifecycle(... ) as lc`，將 warning / failed 回寫 pipeline lifecycle。
4. [Zero Silent Drop]: API 空回應、HTTP 4xx/5xx、dropna 全空、DB upsert 失敗皆必須記入 stats 與 terminal report。
5. [Idempotency]: 使用 ON CONFLICT upsert，確保 seed / sync 可重跑。
6. [Phase-Appropriate Lookback]: 預設 `--days 30` 為日常增量；CoreScore v0.2 選股 phase 建議 `--days 730`
   （對應憲章 6.4 price_coverage_252d + revenue_coverage_24m + financial_coverage_8q）。
7. **[Supply Chain Rate Sovereignty]** (v1.10, 憲法 §7)：對 FinMind 6000/hr 上限與 30 分鐘重置週期實作三層防禦：
   - **L1 事前預防**：滑動窗節流 5500/hr（保留 8% 餘裕）
   - **L2 事中應對**：402 單次 1800s 探測重試；403/429/5xx/Timeout 三階段退避 [30s, 300s, 1800s]
   - **L3 事後續跑**：DB-driven checkpoint，已同步之 (stock_id, dataset, ≥start_date) 不再呼叫 API
8. **[402 vs 403 分流]** (v1.10, 憲法 §7.4)：402 預設視為「資料集付費門檻」單次重試；403/429 視為「速率超限」完整三階段退避。

## 📊 二、全量維運指令總矩陣 (per 憲法 v5.4.22 Section 二 / 5.5.3 五大標準)

### A. 5.5.3 五大標準場景 (Core 5 Scenarios)
| 維運需求場景 (Scenario)                       | 權威指令 / 建議用法                                                                                          | 對齊模組 |
| :-------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--- |
| **1. [個股同步]**                              | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330`                                              | sovereign_sync_engine v1.10 |
| **2. [單一 Table 同步]**                       | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPrice`                   | sovereign_sync_engine v1.10 |
| **3. [單一個股所有 Table 同步]**                | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --all`                                        | sovereign_sync_engine v1.10 |
| **4. [研究宇宙第一階段灌溉]**                  | `$ python scripts/ingestion/sovereign_sync_engine.py --universe research --all --days 730`                   | sovereign_sync_engine v1.10 |
| **5. [核心宇宙最終同步]**                      | `$ python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730`                        | sovereign_sync_engine v1.10 |

### B. 補充運行模式 (Auxiliary Modes)
| 場景                          | 指令                                                                                            | 用途 |
| :---------------------------- | :--------------------------------------------------------------------------------------------- | :--- |
| A. 種子灌溉 (TaiwanStockInfo)  | `$ python scripts/ingestion/sovereign_sync_engine.py --seed`                                    | 第 4 步全市場資產名冊重灌 |
| B. 宏觀指標 (FRED 全譜)        | `$ python scripts/ingestion/sovereign_sync_engine.py --source fred`                             | DFF/UNRATE/T10Y2Y/VIXCLS |
| C. 選股 phase 全宇宙補刷       | `$ python scripts/ingestion/sovereign_sync_engine.py --all --days 730`                          | 為 CoreScore v0.2 準備 2 年歷史 |

### C. 旗標語意 (Flag Semantics)
- `--id <stock_id>`：指定單一標的 ID（如 2330）。
- `--universe research|convex|core`：依 §6.7 SQL 契約讀取對應 tier 名單。
- `--dataset <name>`：只同步單一 dataset；不傳則用 `DEFAULT_FINMIND_DATASETS`（4 核心表）。
- `--all`：取代「DEFAULT 4 表」為 `FINMIND_API_TABLES`（除 `TaiwanStockInfo` 外的全部 stock-data tables）。
- `--days <N>`：增量同步天數（預設 30）。**研究宇宙 / 選股 phase 建議 730**。
- `--seed`：執行 `TaiwanStockInfo` 種子灌溉（與其他旗標可組合）。
- `--source [finmind|fred]`：限定數據源。
- **v1.10 新增 (非破壞性)**：`--no-resume` 停用 L3 斷點續傳（除錯用）；`--throttle N` 自訂節流上限（預設 5500/hr，禁止 ≥ 6000）。
- **v1.11 新增 (非破壞性, §7.6 A1〜A5)**：`--dataset-batched` 改 dataset 優先迴圈 (A1)；`--workers N` 共享 throttle 平行 worker (A2)；`--dynamic-quota --quota-interval N` 動態配額查詢 (A3，N≥100)；A4 per-dataset 配額自動寫入 `data_audit_log`；A5 達 4800/hr 主動 WARN、達 5500/hr 自動暫停 300s。

### D. 不提供之旗標 (Intentionally Omitted)
- `--force` (全歷史 from 2000-01-01)：仍維持不提供，避免在 v0.2 builder 完工前耗盡 FinMind quota。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.11** | 2026-05-17 | Codex | **§7.6 A1〜A5 進階優化落地版**：(A1) 新增 `--dataset-batched` 改外層迴圈優先 dataset，降低單批請求量；(A2) 新增 `--workers N` 平行 worker，共用 thread-safe `FinMindThrottle` (`threading.Lock`)；(A3) 新增 `--dynamic-quota` 與 `--quota-interval N` (N≥100)，每 N 次請求查 FinMind 帳號 API 動態調整節流上限；(A4) `FinMindThrottle` 新增 per-dataset 滑動窗統計，引擎結束時自動寫入 `data_audit_log` op_type=`QUOTA_HOURLY_SNAPSHOT`，不改動既有主鍵；(A5) 4800/hr 觸發一次性 WARN，5500/hr 觸發自動暫停 300s（次數計入 stats）。預設行為 (workers=1, dataset-batched=off, dynamic-quota=off) 完全相容 v1.10。 | **ACTIVE** |
| v1.10 | 2026-05-15 | Codex | §7 供應鏈速率主權落地版：(1) 新增 `FinMindThrottle` 滑動窗節流 5500/hr；(2) 新增 `fetch_with_retry()` 三階段退避 [30s, 300s, 1800s] 與 402 單次探測重試；(3) 新增 `is_already_synced()` DB-driven L3 斷點續傳；(4) 重寫 `sync_finmind()` 整合三層防禢，新增 `skipped` 與 `recovered` stats 類別；(5) `write_data_audit_log` op_type 擴充 `RETRY_402_RECOVERED` / `RESUME_SKIP`；(6) 連續三次失敗即 FAILED 並寫入 lifecycle；(7) CLI 不變、`--no-resume`、`--throttle` 為非破壞性新增。對齊憲法 v5.4.22 §7.1–7.8 全部條文。 | SUPERSEDED |
| v1.9 | 2026-05-14 | Auto-patch | 5.5.3 對齊版：矩陣表補滿；--all 旗標語意正式登錄；Phase-Appropriate Lookback。 | SUPERSEDED |
| v1.8 | 2026-05-14 | Codex | 市場個股資料取得與種子灌溉治理；lifecycle context 接入；動態 PERFECT/WARNING/FAILED。 | SUPERSEDED |
| v1.7 | 2026-05-13 | Auto-patch | Bug #1 修補：sync_fred() 補完 empty-data 失敗分支與 dropna 後空集合分支。 | ARCHIVED |
| v1.6 | 2026-05-13 | Antigravity | 創世圓滿：對齊憲法 v5.4.18。 | ARCHIVED |
================================================================================
"""
import argparse
import os
import re
import sys
import threading
import time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

_THIS_FILE = Path(__file__).resolve()
_INGESTION_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _INGESTION_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.db_utils import (
        get_db_connection,
        record_lifecycle,
        write_data_audit_log,
        get_core_stocks_from_db,
    )
    from core.data_schema import DATASET_REGISTRY, FINMIND_API_TABLES
    from core.finmind_client import FinMindClient
except ImportError as exc:
    print(f"❌ 核心組件導入失敗，請確認 core/ 目錄: {exc}")
    sys.exit(1)


# v1.10 phase-aware constants inherited from v1.9
SELECTION_PHASE_DAYS = 730
UNIVERSE_TIERS = {
    "research": ("research_universe",),
    "convex": ("convex_universe",),
    "core": ("core_universe", "convex_universe"),
}

# v1.10 §7 supply chain rate sovereignty constants
DEFAULT_THROTTLE_PER_HOUR = 5500          # §7.2 主權預設 (8% 餘裕)
ABSOLUTE_THROTTLE_CEILING = 6000          # §7.2 禁止 ≥ 6000
RETRY_BACKOFFS_FULL = [30, 300, 1800]     # §7.3 三階段退避
RETRY_BACKOFF_402 = [1800]                # §7.4 402 單次探測
RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}

# v1.11 §7.6 A1〜A5 進階優化常數（憲法固定值，不得由 caller 調整）
A5_WARN_THRESHOLD = 4800                  # §7.6 A5: 80% 觸發 lifecycle warning（4800 = 6000 × 80%）
A5_PAUSE_THRESHOLD = 5500                 # §7.6 A5: 達 5500/hr 自動暫停
A5_PAUSE_DURATION = 300                   # §7.6 A5: 暫停 5 分鐘
A3_QUOTA_INTERVAL_MIN = 100               # §7.6 A3: N 不得小於 100


class FinMindThrottle:
    """§7.2 L1 事前預防：滑動窗節流（v1.11: thread-safe + §7.6 A4/A5）。
    憲法級常數，禁止調至 >= 6000/hr。
    """

    def __init__(self, max_per_hour=DEFAULT_THROTTLE_PER_HOUR,
                 quota_query_fn=None, quota_check_interval=A3_QUOTA_INTERVAL_MIN):
        if max_per_hour >= ABSOLUTE_THROTTLE_CEILING:
            raise ValueError(
                f"§7.2 違憲：throttle 上限 {max_per_hour} >= {ABSOLUTE_THROTTLE_CEILING}; "
                f"主權預設為 {DEFAULT_THROTTLE_PER_HOUR}"
            )
        if quota_query_fn is not None and quota_check_interval < A3_QUOTA_INTERVAL_MIN:
            raise ValueError(
                f"§7.6 A3 違憲：quota_check_interval={quota_check_interval} < {A3_QUOTA_INTERVAL_MIN}"
            )
        self.max = max_per_hour
        self.window = deque()
        self.lock = threading.Lock()              # v1.11 §7.6 A2 thread-safe
        self.total_acquired = 0
        self.total_throttled_sleep = 0.0
        # v1.11 §7.6 A5 主動配額預警
        self._warn_emitted = False
        self.a5_warn_count = 0
        self.a5_pause_count = 0
        self.total_pause_sleep = 0.0
        # v1.11 §7.6 A4 per-dataset 滑動窗
        self.per_dataset_window = defaultdict(deque)
        # v1.11 §7.6 A3 動態配額查詢
        self.quota_query_fn = quota_query_fn
        self.quota_check_interval = quota_check_interval
        self.dynamic_quota_adjustments = 0

    def acquire(self, dataset=None):
        """獲取一個請求 slot；若已達上限則阻塞。v1.11 thread-safe。"""
        # A3 動態配額查詢必須在 lock 之外（避免 HTTP 期間阻塞其他 worker）
        do_quota_check = False
        with self.lock:
            now = time.time()
            # 主窗口 prune
            while self.window and self.window[0] < now - 3600:
                self.window.popleft()
            # A4 per-dataset 窗口 prune
            if dataset:
                dq = self.per_dataset_window[dataset]
                while dq and dq[0] < now - 3600:
                    dq.popleft()
            window_size = len(self.window)

            # §7.6 A5 主動配額預警（4800/hr → WARN；5500/hr → 暫停 300s）
            if window_size >= A5_PAUSE_THRESHOLD:
                self.a5_pause_count += 1
                print(f"⏸  §7.6 A5 自動暫停：window={window_size}/{A5_PAUSE_THRESHOLD}，"
                      f"sleep {A5_PAUSE_DURATION}s 讓配額自然回收 (第 {self.a5_pause_count} 次)")
                time.sleep(A5_PAUSE_DURATION)
                self.total_pause_sleep += A5_PAUSE_DURATION
                # 重新 prune
                now = time.time()
                while self.window and self.window[0] < now - 3600:
                    self.window.popleft()
                window_size = len(self.window)
            elif window_size >= A5_WARN_THRESHOLD and not self._warn_emitted:
                self.a5_warn_count += 1
                self._warn_emitted = True
                print(f"⚠️  §7.6 A5 預警：window={window_size}/{A5_WARN_THRESHOLD} (80%)；"
                      f"後續可能進入 5500/hr 暫停")

            # §7.2 主節流 (原 v1.10 邏輯)
            if window_size >= self.max:
                sleep_for = 3600 - (now - self.window[0]) + 1
                print(f"⏸  §7.2 節流啟動：sleep {sleep_for:.0f}s 等待 1 小時窗口釋放 "
                      f"(目前窗內 {window_size}/{self.max})")
                time.sleep(sleep_for)
                self.total_throttled_sleep += sleep_for
                now = time.time()
                while self.window and self.window[0] < now - 3600:
                    self.window.popleft()

            self.window.append(time.time())
            self.total_acquired += 1
            if dataset:
                self.per_dataset_window[dataset].append(time.time())

            # A5 hysteresis: window 回落到 80% 以下時 reset warn flag
            if len(self.window) < A5_WARN_THRESHOLD:
                self._warn_emitted = False

            # A3 觸發判定（呼叫移到 lock 外）
            if (self.quota_query_fn is not None
                    and self.total_acquired > 0
                    and self.total_acquired % self.quota_check_interval == 0):
                do_quota_check = True

        if do_quota_check:
            try:
                remaining = self.quota_query_fn()
                if isinstance(remaining, int) and remaining >= 0:
                    with self.lock:
                        # 動態調整：若剩餘配額少，降低 throttle 上限
                        used = len(self.window)
                        suggested = max(used + remaining - 100, 100)
                        new_max = min(suggested, DEFAULT_THROTTLE_PER_HOUR)
                        if new_max != self.max:
                            print(f"📊 §7.6 A3 動態配額調整：max {self.max} → {new_max} "
                                  f"(used={used}, remaining={remaining})")
                            self.max = new_max
                            self.dynamic_quota_adjustments += 1
            except Exception as exc:
                print(f"⚠️  §7.6 A3 配額查詢失敗（忽略）：{type(exc).__name__}: {exc}")

    def per_dataset_snapshot(self):
        """v1.11 §7.6 A4：取 per-dataset 1 小時窗內請求數快照。"""
        with self.lock:
            now = time.time()
            snapshot = {}
            for ds, dq in self.per_dataset_window.items():
                while dq and dq[0] < now - 3600:
                    dq.popleft()
                if dq:
                    snapshot[ds] = len(dq)
            return snapshot


class SovereignSyncEngine:
    FRED_LIST = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]
    DEFAULT_FINMIND_DATASETS = [
        "TaiwanStockPrice",
        "TaiwanStockInstitutionalInvestorsBuySell",
        "TaiwanStockMarginPurchaseShortSale",
        "TaiwanStockPER",
    ]

    def __init__(self, throttle_per_hour=DEFAULT_THROTTLE_PER_HOUR, resume_enabled=True,
                 workers=1, dataset_batched=False, dynamic_quota=False,
                 quota_check_interval=A3_QUOTA_INTERVAL_MIN):
        self.fm_client = FinMindClient()
        self.fred_key = os.getenv("FRED_API_KEY")
        self.constitution_ver = "v5.4.22"
        self.schema_ver = "v2.11"
        self.tool_ver = "v1.11"
        # v1.11 §7.6 A3 動態配額查詢 callback
        quota_fn = self._query_remaining_quota if dynamic_quota else None
        self.throttle = FinMindThrottle(
            max_per_hour=throttle_per_hour,
            quota_query_fn=quota_fn,
            quota_check_interval=quota_check_interval,
        )
        self.resume_enabled = resume_enabled
        # v1.11 §7.6 A1 / A2 旗標
        self.dataset_batched = dataset_batched
        self.workers = max(1, int(workers))
        self.dynamic_quota = dynamic_quota
        self.stats_lock = threading.Lock()  # v1.11 §7.6 A2 thread-safe stats
        self.stats = {
            "success": 0,
            "warning": 0,
            "failed": 0,
            "skipped": 0,         # v1.10: L3 斷點續傳跳過
            "recovered_402": 0,   # v1.10: 402 探測重試成功
            "rows": 0,
            "details": [],
        }

    # ---------- v1.11 §7.6 A3 動態配額查詢 callback ----------

    def _query_remaining_quota(self):
        """A3: 回傳 FinMind 帳號剩餘小時配額；此查詢本身計入配額（憲法 §7.6 A3 邊界）。
        若 API 不提供 remaining 欄位則回傳 None，不調整 throttle。"""
        try:
            info = self.fm_client.get_user_info()
        except Exception:
            return None
        # FinMind get_user_info 回應結構：{"msg": "...", "user_count": N, "api_request_limit": M, ...}
        # 不同版本可能用不同欄位名；嘗試幾個常見鍵
        if not isinstance(info, dict):
            return None
        for key in ("api_request_limit", "remaining", "request_remaining", "quota_remaining"):
            v = info.get(key)
            if isinstance(v, (int, float)):
                return int(v)
        return None

    # ---------- detail / error helpers ----------

    def _detail(self, status, message):
        # v1.11: thread-safe (§7.6 A2)
        icon_map = {
            "success": "✅",
            "warning": "⚠️",
            "failed": "❌",
            "skipped": "⏭ ",
            "recovered_402": "♻️",
        }
        with self.stats_lock:
            if status in self.stats:
                self.stats[status] += 1
            icon = icon_map.get(status, "•")
            self.stats["details"].append(f"{icon} {message}")

    def _add_rows(self, n):
        # v1.11: thread-safe rows counter
        with self.stats_lock:
            self.stats["rows"] += n

    def _safe_error(self, exc):
        message = f"{type(exc).__name__}: {exc}"
        return re.sub(r"([?&]token=)[^&\s]+", r"\1<redacted>", message)

    # ---------- v1.10 §7.5 L3 checkpoint ----------

    def is_already_synced(self, stock_id, dataset_name, start_date):
        """
        §7.5 DB-driven L3 斷點續傳。
        若 (dataset_name) 表內存在 stock_id 且 date >= start_date 的紀錄，
        視為「該 (stock_id, dataset, >= start_date) 已同步」並回傳 True。
        對 TaiwanStockInfo 等市場級或無 stock_id 表，不啟用 checkpoint。
        """
        if not self.resume_enabled:
            return False
        if not stock_id:
            return False
        config = DATASET_REGISTRY.get(dataset_name)
        if not config:
            return False
        columns = config.get("columns", {})
        if "stock_id" not in columns or "date" not in columns:
            return False

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{dataset_name}" '
                f'WHERE "stock_id" = %s AND "date" >= %s LIMIT 1',
                (stock_id, start_date),
            )
            row = cur.fetchone()
            cur.close()
            return row is not None
        except Exception:
            return False
        finally:
            conn.close()

    # ---------- v1.10 §7.2-7.4 retry / throttle wrapper ----------

    def fetch_finmind(self, params):
        """
        §7.2 / §7.4 統一進場：節流 → 請求 → 狀態碼分流。
        回傳 (payload, recovered_402_flag)；任何最終失敗即拋例外。
        v1.11: 將 dataset 名稱傳給 throttle 以支援 §7.6 A4 per-dataset 統計。
        """
        url = self.fm_client.api_url
        headers = self.fm_client.headers
        ds_label = params.get("dataset") if isinstance(params, dict) else None

        # 402 與 403 走不同 backoff 軌道，但同一次呼叫中可能先遇 200，後遇 403
        backoff_403 = list(RETRY_BACKOFFS_FULL)  # [30, 300, 1800]
        backoff_402 = list(RETRY_BACKOFF_402)    # [1800]
        recovered_402 = False
        last_status = None

        while True:
            self.throttle.acquire(dataset=ds_label)
            try:
                res = requests.get(url, params=params, headers=headers, timeout=30)
            except (requests.Timeout, requests.ConnectionError) as exc:
                # 視同 5xx，走 backoff_403 軌道
                if not backoff_403:
                    raise
                wait = backoff_403.pop(0)
                print(f"⏱ {exc.__class__.__name__}; sleep {wait}s 後重試")
                time.sleep(wait)
                continue

            last_status = res.status_code
            if res.status_code == 200:
                payload = res.json()
                # FinMind 應用層錯誤（msg != success）
                msg = payload.get("msg")
                if msg not in (None, "success", ""):
                    raise RuntimeError(f"FinMind app-level error: {msg}")
                return payload, recovered_402

            if res.status_code == 402:
                if not backoff_402:
                    raise requests.HTTPError(f"402 Payment Required (permanent after probe): {res.text[:200]}")
                wait = backoff_402.pop(0)
                print(f"⚠ HTTP 402 探測重試：sleep {wait}s（§7.4 單次探測）")
                time.sleep(wait)
                recovered_402 = True  # 若下次成功則標註為 recovered
                continue

            if res.status_code in RETRYABLE_STATUS_CODES:
                if not backoff_403:
                    raise requests.HTTPError(f"{res.status_code} after {len(RETRY_BACKOFFS_FULL)} retries")
                wait = backoff_403.pop(0)
                print(f"⚠ HTTP {res.status_code}; §7.3 退避 sleep {wait}s")
                time.sleep(wait)
                continue

            # 其他 4xx → 立即拋
            res.raise_for_status()
            return None, recovered_402  # unreachable but for linter

    # ---------- schema / DB helpers (v1.10; schema contract inherited from v1.9) ----------

    def _convert_type(self, series, sql_type):
        if sql_type.startswith("DATE"):
            return pd.to_datetime(series, errors="coerce").dt.date
        if sql_type.startswith("TIMESTAMP"):
            return pd.to_datetime(series, errors="coerce")
        if sql_type.startswith("NUMERIC") or sql_type.startswith("INTEGER") or sql_type.startswith("BIGINT"):
            return pd.to_numeric(series, errors="coerce")
        return series.where(series.notna(), None)

    def _align_to_schema(self, table_name, df):
        config = DATASET_REGISTRY.get(table_name)
        if not config:
            raise ValueError(f"憲章未定義表名: {table_name}")

        expected_cols = list(config["columns"].keys())
        actual_cols = list(df.columns)
        missing = [col for col in expected_cols if col not in actual_cols]
        extra = [col for col in actual_cols if col not in expected_cols]
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing={missing}")
            if extra:
                problems.append(f"extra={extra}")
            raise ValueError(f"{table_name} API/schema 欄位不一致: {'; '.join(problems)}")

        aligned = df[expected_cols].copy()
        for col, sql_type in config["columns"].items():
            aligned[col] = self._convert_type(aligned[col], sql_type)

        unique_cols = config.get("unique_constraints", [])
        if unique_cols:
            before = len(aligned)
            aligned = aligned.dropna(subset=unique_cols)
            dropped = before - len(aligned)
            if dropped:
                self._detail("warning", f"{table_name}: 已丟棄 {dropped} 筆 unique key 欄位為空的資料")

        aligned = aligned.where(pd.notnull(aligned), None)
        if aligned.empty:
            raise ValueError(f"{table_name} 清洗後資料為空")
        return aligned

    def _db_value(self, value):
        if pd.isna(value):
            return None
        return value

    def _upsert_to_db(self, table_name, df):
        if df.empty:
            raise ValueError(f"{table_name} 無可寫入資料")

        config = DATASET_REGISTRY[table_name]
        unique_cols = config.get("unique_constraints", [])
        if not unique_cols:
            raise ValueError(f"{table_name} 未定義 unique_constraints，禁止 upsert")

        cols = list(df.columns)
        quoted_cols = [f'"{col}"' for col in cols]
        placeholders = ", ".join(["%s"] * len(cols))
        conflict_cols = ", ".join([f'"{col}"' for col in unique_cols])
        update_cols = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in cols if col not in unique_cols])
        update_clause = "DO NOTHING" if not update_cols else f"DO UPDATE SET {update_cols}"

        sql = f'''
            INSERT INTO "{table_name}" ({", ".join(quoted_cols)})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_cols})
            {update_clause}
        '''

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            data = [
                tuple(self._db_value(value) for value in row)
                for row in df.itertuples(index=False, name=None)
            ]
            cur.executemany(sql, data)
            conn.commit()
            rows = len(df)
            audit_date = df.iloc[0].get("date", datetime.now().date()) if hasattr(df.iloc[0], "get") else datetime.now().date()
            write_data_audit_log(table_name, "SYNC", audit_date, "UPSERT", rows)
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    # ---------- v1.10 重寫的 sync_finmind ----------

    def sync_finmind(self, stock_id, dataset_name, start_date):
        # L3 §7.5 斷點續傳：先看 DB 是否已有資料
        if self.is_already_synced(stock_id, dataset_name, start_date):
            self._detail("skipped", f"{dataset_name} ({stock_id}): DB 已有 ≥ {start_date} 資料 (§7.5 resume)")
            try:
                write_data_audit_log(dataset_name, "SYNC", datetime.now().date(), "RESUME_SKIP", 0)
            except Exception:
                pass
            return

        try:
            print(f"📡 正在獲取 FinMind: {stock_id} / {dataset_name}...")
            params = {"dataset": dataset_name, "start_date": start_date}
            if stock_id is not None:
                params["data_id"] = stock_id
            if self.fm_client.token:
                params["token"] = self.fm_client.token

            payload, recovered_402 = self.fetch_finmind(params)
            data = payload.get("data", [])
            if not data:
                self._detail("warning", f"{dataset_name} ({stock_id or 'MARKET'}): API 回傳 0 筆")
                return

            df = self._align_to_schema(dataset_name, pd.DataFrame(data))
            rows = self._upsert_to_db(dataset_name, df)
            self._add_rows(rows)

            if recovered_402:
                # §7.4: 寫入 audit log 標籤
                try:
                    audit_date = df.iloc[0].get("date", datetime.now().date()) if hasattr(df.iloc[0], "get") else datetime.now().date()
                    write_data_audit_log(dataset_name, "SYNC", audit_date, "RETRY_402_RECOVERED", rows)
                except Exception:
                    pass
                self._detail("recovered_402",
                             f"{dataset_name} ({stock_id or 'MARKET'}): {rows} 筆 UPSERT 成功（402-recovered）")
                # 仍計入 success
                self.stats["success"] += 1
            else:
                self._detail("success", f"{dataset_name} ({stock_id or 'MARKET'}): {rows} 筆 UPSERT 成功")
        except Exception as exc:
            self._detail("failed", f"{dataset_name} ({stock_id or 'MARKET'}) 失敗: {self._safe_error(exc)}")

    def sync_fred(self, series_id):
        try:
            print(f"📡 正在獲取 FRED: {series_id}...")
            if not self.fred_key:
                raise RuntimeError("FRED_API_KEY missing")
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_key,
                "file_type": "json",
                "limit": 1000,
                "sort_order": "desc",
            }
            res = requests.get(url, params=params, timeout=30)
            res.raise_for_status()
            payload = res.json()
            data = payload.get("observations", [])
            if not data:
                raise RuntimeError(payload.get("error_message") or "API 回傳 0 個 observation")

            df = pd.DataFrame(data)
            df["series_id"] = series_id
            df = self._align_to_schema("FredData", df)
            df = df.dropna(subset=["value"])
            if df.empty:
                raise ValueError("全部 observation 在數據聖潔清洗後為空")

            rows = self._upsert_to_db("FredData", df)
            self._add_rows(rows)
            self._detail("success", f"FRED/{series_id}: {rows} 筆 UPSERT 成功")
        except Exception as exc:
            self._detail("failed", f"FRED/{series_id} 失敗: {self._safe_error(exc)}")

    # ---------- universe / dataset resolution ----------

    def _resolve_stocks(self, stock_id, universe):
        if stock_id:
            return [stock_id]
        if universe in UNIVERSE_TIERS:
            try:
                stocks = get_core_stocks_from_db(tiers=UNIVERSE_TIERS[universe])
            except Exception as exc:
                self._detail("failed", f"{universe} universe 讀取失敗: {type(exc).__name__}: {exc}")
                return []
            if not stocks:
                self._detail("warning", f"{universe} universe 無標的")
            return stocks
        return []

    def _target_datasets(self, dataset, all_datasets):
        if dataset:
            return [dataset]
        if all_datasets:
            return [name for name in FINMIND_API_TABLES if name != "TaiwanStockInfo"]
        return self.DEFAULT_FINMIND_DATASETS

    def _apply_lifecycle_verdict(self, lifecycle):
        if lifecycle is None:
            return
        if self.stats["failed"] > 0 and hasattr(lifecycle, "mark_failed"):
            lifecycle.mark_failed("; ".join(self.stats["details"][:5]))
        elif self.stats["warning"] > 0 and hasattr(lifecycle, "mark_warning"):
            lifecycle.mark_warning("; ".join(self.stats["details"][:5]))

    def _phase_appropriate_hint(self, stock_id, universe, days, dataset, seed, all_datasets):
        if seed or stock_id or dataset:
            return
        if universe in UNIVERSE_TIERS and days < SELECTION_PHASE_DAYS:
            print(f"💡 [Phase Hint] 對 `--universe {universe}` 使用 `--days {days}` 只取增量。")
            print(f"   選股 phase 建議 `--days {SELECTION_PHASE_DAYS}`（~2 年），對應憲章 6.4 三項 coverage。")

    def _iter_sync_pairs(self, stocks, datasets):
        """v1.11 §7.6 A1: dataset-batched=True 時改外層 dataset、內層 stock。
        預設順序保留 v1.10 行為（外層 stock）。"""
        if self.dataset_batched:
            for ds in datasets:
                for sid in stocks:
                    yield sid, ds
        else:
            for sid in stocks:
                for ds in datasets:
                    yield sid, ds

    def _execute_pairs(self, pairs, start_date):
        """v1.11 §7.6 A2: workers=1 為串行（與 v1.10 完全相容）；workers>1 走 ThreadPoolExecutor。
        共用同一 FinMindThrottle，因此節流仍受 §7.2 主權保護。"""
        if self.workers <= 1:
            for sid, ds in pairs:
                self.sync_finmind(sid, ds, start_date)
            return
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="sync") as ex:
            futures = [ex.submit(self.sync_finmind, sid, ds, start_date) for sid, ds in pairs]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:
                    self._detail("failed", f"thread exception: {type(exc).__name__}: {exc}")

    def _flush_quota_audit(self):
        """v1.11 §7.6 A4: 引擎結束時將 per-dataset 一小時請求量寫入 data_audit_log。
        op_type='QUOTA_HOURLY_SNAPSHOT'；不改既有主鍵與必填欄位。"""
        snapshot = self.throttle.per_dataset_snapshot()
        if not snapshot:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        for dataset, count in sorted(snapshot.items()):
            try:
                write_data_audit_log(dataset, "SYSTEM", today, "QUOTA_HOURLY_SNAPSHOT", count)
            except Exception as exc:
                self._detail("warning", f"§7.6 A4 quota flush failed for {dataset}: {type(exc).__name__}: {exc}")

    def run(self, stock_id=None, universe=None, source=None, dataset=None, days=30, seed=False, all_datasets=False):
        start_time = time.time()
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        task_name = f"sync_{source or 'all'}_{stock_id or universe or ('seed' if seed else 'macro')}"

        self._phase_appropriate_hint(stock_id, universe, days, dataset, seed, all_datasets)

        with record_lifecycle(task_name, category="ingestion", stock_id=stock_id or "SYSTEM") as lifecycle:
            if source in (None, "finmind"):
                if seed or dataset == "TaiwanStockInfo":
                    # TaiwanStockInfo 為市場級表，單次呼叫；不走平行
                    self.sync_finmind("", "TaiwanStockInfo", start_date)

                stocks = self._resolve_stocks(stock_id, universe)
                datasets = [ds for ds in self._target_datasets(dataset, all_datasets) if ds != "TaiwanStockInfo"]
                pairs = list(self._iter_sync_pairs(stocks, datasets))
                if pairs:
                    self._execute_pairs(pairs, start_date)

            if source in (None, "fred") and not stock_id:
                # FRED 不計入 FinMind 配額；維持串行
                for series_id in self.FRED_LIST:
                    self.sync_fred(series_id)

            # v1.11 §7.6 A4: lifecycle 結束前 flush per-dataset quota snapshot
            self._flush_quota_audit()
            self._apply_lifecycle_verdict(lifecycle)
            self.report_results(start_time, days, universe)

        return self.stats["failed"] == 0

    def report_results(self, start_time, days, universe):
        if self.stats["failed"] > 0:
            verdict = "FAILED"
        elif self.stats["warning"] > 0:
            verdict = "WARNING"
        else:
            verdict = "PERFECT"

        if days >= SELECTION_PHASE_DAYS:
            phase_label = f"選股 phase ({days} 天)"
        elif days <= 60:
            phase_label = f"增量 phase ({days} 天)"
        else:
            phase_label = f"自訂窗 ({days} 天)"

        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: 主權同步引擎執行摘要 ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"治權基準 : 系統架構大憲章_{self.constitution_ver}.md (§7 對齊)")
        print(f"schema 基準 : data_schema {self.schema_ver}")
        if universe:
            print(f"執行 universe : {universe}")
        print(f"執行 phase : {phase_label}")
        print(f"§7 節流統計 : acquired={self.throttle.total_acquired}, throttle_sleep={self.throttle.total_throttled_sleep:.0f}s")
        print(f"§7 L3 續跑 : skipped={self.stats['skipped']}, 402_recovered={self.stats['recovered_402']}")
        print(f"§7.6 A2 workers={self.workers}, A1 dataset_batched={self.dataset_batched}, "
              f"A3 dynamic_quota={self.dynamic_quota} (adjustments={self.throttle.dynamic_quota_adjustments})")
        print(f"§7.6 A5 預警次數={self.throttle.a5_warn_count}, 自動暫停次數={self.throttle.a5_pause_count}, "
              f"暫停總時長={self.throttle.total_pause_sleep:.0f}s")
        print("─" * 80)
        for detail in self.stats["details"]:
            print(detail)
        print("─" * 80)
        print(f"📈 成功同步項目 : {self.stats['success']}")
        print(f"⚠️  警告同步項目 : {self.stats['warning']}")
        print(f"❌ 失敗同步項目 : {self.stats['failed']}")
        print(f"⏭  跳過同步項目 : {self.stats['skipped']}")
        print(f"♻️  402-recovered : {self.stats['recovered_402']}")
        print(f"📝 總計寫入筆數 : {self.stats['rows']}")
        print(f"🕒 總計耗時     : {(time.time() - start_time):.2f} s")
        print(f"⚖️  主權判定     : {verdict}")
        print("🛡️" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantum Finance 主權同步引擎 (v1.11 — §7 供應鏈速率主權 + §7.6 A1〜A5 進階優化)",
        epilog="選股 phase 範例：python scripts/ingestion/sovereign_sync_engine.py --universe research --all --days 730；"
               "核心 phase 範例：python scripts/ingestion/sovereign_sync_engine.py --universe core --all --days 730",
    )
    parser.add_argument("--id", type=str, help="指定標的 ID (如 2330)")
    parser.add_argument("--universe", type=str, choices=["research", "convex", "core"],
                        help="指定標的範圍（research / convex / core）")
    parser.add_argument("--source", type=str, choices=["finmind", "fred"], help="指定數據源")
    parser.add_argument("--dataset", type=str, help="指定單一 dataset 名稱")
    parser.add_argument("--seed", action="store_true",
                        help="種子灌溉模式（同步 TaiwanStockInfo 全市場資產名冊）")
    parser.add_argument("--all", action="store_true",
                        help="使用 FINMIND_API_TABLES 全表 (除 TaiwanStockInfo) 取代 DEFAULT_FINMIND_DATASETS")
    parser.add_argument("--days", type=int, default=30,
                        help="同步天數 (預設 30；選股 phase 建議 730)")
    # v1.10 新增（非破壞性）
    parser.add_argument("--no-resume", action="store_true",
                        help="(v1.10) 停用 §7.5 L3 斷點續傳；除錯用，正式運行不建議")
    parser.add_argument("--throttle", type=int, default=DEFAULT_THROTTLE_PER_HOUR,
                        help=f"(v1.10) §7.2 節流上限/小時 (預設 {DEFAULT_THROTTLE_PER_HOUR}，禁止 ≥ {ABSOLUTE_THROTTLE_CEILING})")
    # v1.11 §7.6 A1〜A5 新增（非破壞性）
    parser.add_argument("--dataset-batched", action="store_true",
                        help="(v1.11 §7.6 A1) 改 dataset 優先迴圈，單批請求量遠低於 6000")
    parser.add_argument("--workers", type=int, default=1,
                        help="(v1.11 §7.6 A2) 平行 worker 數量，預設 1 (串行)；共用 thread-safe throttle")
    parser.add_argument("--dynamic-quota", action="store_true",
                        help="(v1.11 §7.6 A3) 每 N 次請求查 FinMind 帳號 API 動態調整節流上限")
    parser.add_argument("--quota-interval", type=int, default=A3_QUOTA_INTERVAL_MIN,
                        help=f"(v1.11 §7.6 A3) 動態配額查詢間隔；憲法下限 {A3_QUOTA_INTERVAL_MIN}")
    args = parser.parse_args()

    engine = SovereignSyncEngine(
        throttle_per_hour=args.throttle,
        resume_enabled=(not args.no_resume),
        workers=args.workers,
        dataset_batched=args.dataset_batched,
        dynamic_quota=args.dynamic_quota,
        quota_check_interval=args.quota_interval,
    )
    ok = engine.run(
        stock_id=args.id,
        universe=args.universe,
        source=args.source,
        dataset=args.dataset,
        days=args.days,
        seed=args.seed,
        all_datasets=args.all,
    )
    sys.exit(0 if ok else 1)
