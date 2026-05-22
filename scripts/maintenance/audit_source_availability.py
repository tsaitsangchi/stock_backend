"""
audit_source_availability.py v0.4 (Strict Source Availability Audit · §6.8.8-C Time-Drift Tolerance · §6.8.8-D Full-Market Mode · §0.4 Progress Heartbeat)
================================================================================
**最後更新日期**: 2026-05-22
**主權狀態**: ACTIVE (憲法 v6.0.0 §14.7-L 對齊 + §6.8.8-C 時點漂移容忍規則落地 + §14.7-AP 治權閉環 + §6.8.8-D 全市場驗證模式 + §14.7-AQ 範圍對稱性補齊 + §0.4 [Hybrid Observability] 進度心跳)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Strict Source Authority]: 驗證 core+convex 150 個股資料是否符合憲章 §14.7-L 之嚴格定義
   — 每一個 FinMind `stock_id + dataset` 必須從 API 最早可得日期完整對齊 DB；
   `start_date=1990-01-01` 為來源端可得下界。
2. [Dual-Source Verification]: 對每個 `stock_id + dataset` 比對 API 與 DB 的
   `row_count / min(date) / max(date)`；API source-empty 時 DB 必須為 0 rows
   才視為 `SOURCE_EMPTY_OK`；任何差異即為 mismatch。
3. [FRED Valid Numeric Coverage]: `--include-fred` 啟用時，另對 DFF / UNRATE /
   T10Y2Y / VIXCLS 四序列以「可轉為數值的有效 observation」（排除 `.` 缺值列）
   為驗收口徑，row_count / min / max 必須與 DB 完全一致。
4. [Strict Mode Exit Contract]: `--strict` 模式下任何 mismatch 即 exit 1 並
   產出 `reports/source_availability_audit_*.md` 與 targeted backfill commands；
   對齊憲章 §3.2 接受標準（FAIL → exit 1）。
5. [Hybrid Observability]: 維運行為觸發 `record_lifecycle` 與 `write_data_audit_log`；
   主權狀態依實況動態計算（PERFECT / WARNING / FAILED），嚴禁硬編
   （對齊憲章 §5.6.3 [Zero Hardcoded Verdict]）。
   **v0.4 補強**：對長時運行（全市場 ~5 hr）新增 per-N-stock progress heartbeat
   （`--progress-interval N`；預設 100；N=0 為靜默模式相容 v0.3）；每條 heartbeat 含
   `idx/total | checked | source_empty_ok | time_drift_ok | mismatch | api_errors | elapsed | eta`，
   解決「audit 中段 25,000 probe 完全靜默」之觀察缺口。
6. [Historical Reference Authority]: 保留完整修訂歷程作為判定系統正確性之基準。
7. **[Time-Drift Tolerance]** (v0.2, 憲法 §6.8.8-C / §14.7-AP)：對「**audit 觀察時點之自然時間漂移**」之容忍：
   - **`(api_date_max - db_date_max) ≤ N_calendar_days`** 且 **`abs(api_rows - db_rows) ≤ N`** → 標記為
     `TIME_DRIFT_OK`（**不**視為 mismatch；不計入 exit 1）
   - **預設 N = 3 個日曆日**（覆蓋週末 + 1 個工作日緩衝）
   - CLI: `--drift-tolerance N`（N=0 為嚴格模式；N>0 為容忍模式；預設 3）
   - 對齊憲章 §6.8.8-C 治權契約：「sync 時點 vs API publish 時點之競爭」+「audit 觀察時點之自然延遲」屬合法漂移
8. **[Sovereignty Declaration]** (v0.2, 憲章 §3.2A 橫切稽核工具)：本程式為 §3.2A 橫切基礎設施稽核工具
   （非 §3.1 序列模組）；落實 §14.7-L strict source availability + §6.8.8-C 時點漂移容忍；不涉及 §0.1-A / §0.2-A /
   §0.3-A / §0.0-E.4 / §0.0-F.3 五套禁令；不在 §0.1.1 T1/T2/T3 分層內；不處理 §8.5 anti-leakage；不調度 universe；
   不持有 schema 定義；DATASET_REGISTRY + FINMIND_API_TABLES 為唯一 schema 引用源。
9. **[Full-Market Audit Mode]** (v0.3, 憲法 §6.8.8-D / §14.7-AQ；2026-05-22 入憲)：對齊 `sovereign_sync_engine` 之
   §6.8.7 第 (4) 條全市場限定治理例外，落地 audit 側對等驗證範圍：
   - `--universe full` 解鎖 `core ∪ convex ∪ research ∪ quarantine` ≈ 2,798 支 × 10 datasets 之全市場 audit
   - **必須**附 `--special-full-market-reason "<≥12 字理由>"`（與 sovereign_sync_engine 同口徑）；缺 reason / reason < 12 字 → preflight exit 1
   - 五類合法情境：DB rebuild bootstrap / Sovereign rebuild / pre-annual audit / 資料源治權變更 / 重大合規事件
   - reason 寫入報表 metadata + 終端 summary 留 audit trail
   - `_resolve_stocks(universe="full")` 對齊 `sovereign_sync_engine.UNIVERSE_TIERS["full"]` SSOT（4-tier union；避免兩工具範圍漂移）
   - 報表標題去 "Core 150" 字樣（為「Strict source availability audit」）；scope 欄位升級為 `stocks=N (universe=core|full)`
   - 跨工具治權範本：未來 maintenance audit 工具支援 `--universe full` 一律比照（必須附 reason）

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
| 維運需求場景 (Scenario) | 權威指令 / 建議用法 | 對齊模組 |
| :--- | :--- | :--- |
| **1. [標準執行：core+convex 150 嚴格驗證]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict` | audit_source_availability v0.3 |
| **2. [單一個股全表驗證]** | `$ python scripts/maintenance/audit_source_availability.py --id 2330 --all --strict` | audit_source_availability v0.3 |
| **3. [來源端 snapshot 寫出]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --strict --snapshot-out /tmp/api_start_dates_core150.json` | audit_source_availability v0.3 |
| **4. [離線重跑：用既有 snapshot]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --strict --snapshot-in /tmp/api_start_dates_core150.json` | audit_source_availability v0.3 |
| **5. [僅 FinMind（略 FRED）]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --strict` | audit_source_availability v0.3 |
| **6. [§6.8.8-C 時點漂移容忍：預設 3 個日曆日]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict --drift-tolerance 3` | audit_source_availability v0.3 |
| **7. [§6.8.8-C 嚴格模式：相容 v0.1]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict --drift-tolerance 0` | audit_source_availability v0.3 |
| **8. [§6.8.8-D 全市場驗證：~2,798 支]** | `$ python scripts/maintenance/audit_source_availability.py --universe full --all --include-fred --strict --special-full-market-reason "DB rebuild bootstrap 2026-05-22 full-market validation"` | audit_source_availability v0.3 |
| **9. [§6.8.8-D preflight 拒絕：缺 reason]** | `$ python scripts/maintenance/audit_source_availability.py --universe full --all` → **exit 1**（缺 reason / reason < 12 字 / reason 給但 universe 非 full）| audit_source_availability v0.3 |
| **10. [§0.4 進度心跳：每 100 stock]** | `$ python scripts/maintenance/audit_source_availability.py --universe full --all --include-fred --strict --special-full-market-reason "<reason>" --progress-interval 100` | audit_source_availability v0.4 |
| **11. [§0.4 靜默模式：相容 v0.3]** | `$ python scripts/maintenance/audit_source_availability.py --universe core --all --include-fred --strict --progress-interval 0` | audit_source_availability v0.4 |

### B. 補充運行模式 (Auxiliary Modes)
| 模式 | 指令旗標 | 用途 |
| :--- | :--- | :--- |
| **non-strict** | 移除 `--strict` | 報告 mismatch 但 exit 0；用於診斷掃描 |
| **snapshot-only** | `--snapshot-out <path>` 不加 `--strict` | 僅產生來源端 snapshot，不做 DB 比對 |
| **fred-only** | `--include-fred --datasets fred` | 略過 FinMind，僅對 FRED 四序列驗證 |
| **id-list** | `--id 2330 --id 2454 ...` | 多個指定 stock_id 並列驗證 |

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v0.4** | 2026-05-22 | Codex | **§0.4 [Hybrid Observability] 進度心跳落地 — 全市場 5h audit 中段觀察缺口補齊**：v0.3 首戰全市場 audit (PID 1128773) 啟動 13 分鐘後揭露**觀察缺口**：log 0 bytes / 無 per-stock print / 無中段 DB 寫入 → 對 ~25,000 probe / ~5h 之長時運行完全無法掌握進度。**功能變更 3 點**：(I) 新增模組級常數 `PROGRESS_INTERVAL_DEFAULT = 100`；(II) `__init__` 新增 `progress_interval` 參數 + argparse `--progress-interval N` flag（預設 100；N=0 為靜默模式相容 v0.3）；(III) `run()` 主迴圈每 N 個 stock 印一條 progress heartbeat：`stocks=idx/total (%) | checked | source_empty_ok | time_drift_ok | mismatch | api_errors | elapsed | eta`；run start 印一條 init line；最後一個 stock 強制印一條 final progress。**標頭變更**：(a) L2 副標補入「§0.4 Progress Heartbeat」；(b) L5 主權狀態行補入 §0.4 [Hybrid Observability] 進度心跳；(c) 核心定義 5. [Hybrid Observability] 擴增「v0.4 補強」說明（不另立新條，純功能擴增）；(d) 維運矩陣新增場景 10 (`--progress-interval 100`) + 場景 11 (`--progress-interval 0` 相容 v0.3)；(e) TOOL_VER v0.3 → v0.4。**治權位階**：**P3 觀察性升級**（§0.0-E.6 升版優先級），**不**改 verdict 邏輯 / **不**改 schema / **不**改 audit 範圍邏輯 / **不**需要 §0.0-G 入憲（純 UX）；對齊憲章 §0.4 [Hybrid Observability] + §5.6.3 [Zero Hardcoded Verdict] 既有原則。**對應 sovereign_sync_engine 之 `_detail()` print 慣例**（同類進度可見性設計範本）。**驗證已通過**：compile OK；--help 顯示 --progress-interval；smoke 單一個股仍 verdict=PERFECT；--progress-interval 0 完全靜默相容 v0.3。**§14.7-AQ Phase 3 全市場 audit 首戰實證**將用 v0.4 重啟（reason 不變）。 | **ACTIVE** |
| v0.3 | 2026-05-22 | Codex | **§6.8.8-D 全市場驗證模式 + §14.7-AQ 治權範圍對稱性補齊落地**：依憲章 v6.0.0-patch §6.8.8-D + §14.7-AQ（commit `52e4511`；2026-05-22 入憲）落地 audit 側對等於 `sovereign_sync_engine v1.21 --universe full` 之全市場 audit 能力，閉合 §14.7-AQ 識別之治權範圍對稱性缺口（~26,500 對全市場資料原無治權內驗證機制）。**功能變更 5 點**：(I) argparse `--universe choices=["core"]` → `["core", "full"]`；(II) argparse 新增 `--special-full-market-reason` flag + main() preflight 3 分支治權檢查（缺 reason / reason < 12 字 / reason 給但 universe 非 full → exit 1）；(III) `_resolve_stocks(universe="full")` 對齊 `sovereign_sync_engine.UNIVERSE_TIERS["full"]` SSOT，回傳 `core_universe ∪ convex_universe ∪ research_universe ∪ quarantine_universe` 4-tier union ≈ 2,798 支；(IV) `__init__` 新增 `special_full_market_reason` 參數；reason 寫入報表 metadata + 終端 summary 留 audit trail；(V) 新增模組級常數 `FULL_MARKET_REASON_MIN_CHARS = 12` + `FULL_MARKET_REQUIRED_UNIVERSE = "full"`（獨立定義避免 maintenance → ingestion 反向耦合 import；對映 sovereign_sync_engine 同名常數）。**標頭變更**：(a) L2 副標補入「§6.8.8-D Full-Market Mode」；(b) L5 主權狀態行補入 §6.8.8-D + §14.7-AQ；(c) 核心定義 8 條 → 9 條：新增 [Full-Market Audit Mode] (v0.3, §6.8.8-D / §14.7-AQ；7 條治權邊界)；(d) 維運矩陣補入場景 8（`--universe full` 含 reason）+ 場景 9（preflight 拒絕 invalid case）；(e) 報表標題去 "Core 150" → "Strict source availability audit"；(f) scope 欄位升級為 `stocks=N (universe=core|full)`；(g) constitution 欄位加 §6.8.8-D + §14.7-AQ；(h) `_print_summary` 加 universe + special_full_market_reason 顯示；(i) TOOL_VER v0.2 → v0.3。**治權邊界嚴守**：所有既有 `--universe core` 行為**完全不變**（v0.2 → v0.3 為新功能擴充非邏輯改寫）；不修改 §6.8.7 第 (4) 條五類合法情境、不修改 §6.8.8 / §6.8.8-A / §6.8.8-B / §6.8.8-C 既有條款、不改 §3.1 / §3.2 / §6.7 / §7 / §8 / §9 強制契約、不改 CoreScore v0.2 與 ThemeResonance 15%、不改 schema 定義（DATASET_REGISTRY + FINMIND_API_TABLES 仍為唯一引用源）。**驗證已通過**：compile OK；preflight 3 分支全部 exit 1 ✓；smoke `--id 2330 --dataset TaiwanStockPrice` verdict=PERFECT；`--help` 顯示 `choices=["core", "full"]` + `--special-full-market-reason`。**對應 §0.0-G 第 15 次跑通**（§14.7-AQ）落地 Phase 2 - 程式單修。 | SUPERSEDED |
| v0.2 | 2026-05-22 | Codex | **§6.8.8-C 時點漂移容忍規則落地 + §14.7-AP 治權閉環延伸實證**：依憲章 v6.0.0-patch §6.8.8-C + §14.7-AP（commit `4d990d0`；2026-05-22 入憲）落地實作 audit 觀察時點漂移之容忍規則。**補正內容**：(I) 新增 `--drift-tolerance N` argparse flag（預設 N=3 個日曆日；N=0 為嚴格模式相容 v0.1）；(II) `_classify()` 邏輯擴增 TIME_DRIFT_OK 分支：當 `(api_date_max - db_date_max).days ≤ N` 且 `abs(api_rows - db_rows) ≤ N` 時標記為 TIME_DRIFT_OK（**不**計入 mismatch / 不影響 exit code）；(III) `_classify_fred()` 同步擴增 TIME_DRIFT_OK 分支；(IV) stats 新增 `time_drift_ok` + `fred_time_drift_ok` 計數器；(V) 報告新增 TIME_DRIFT_OK 獨立分類段落；(VI) 核心定義新增 [Time-Drift Tolerance] + [Sovereignty Declaration] 兩條治權慣例；(VII) 維運矩陣補入 6/7 兩個 --drift-tolerance scenarios；(VIII) 新增模組級 CONSTITUTION_VER + TOOL_VER 常數。**§14.7-AP 治權閉環實證**：本 commit 與 charter commit `4d990d0` 之治權契約完全對齊；2026-05-22 11:13 已實證 PERFECT 0/0（mismatch 全部消解後本 v0.2 之容忍規則為日後增量 sync 時自然漂移之預防性容忍）。**介面零變動**：所有既有 CLI flag / verdict 計算 / record_lifecycle + write_data_audit_log 接線不變；新增之 `--drift-tolerance` 屬非破壞性 flag（N=0 完全相容 v0.1 行為）。對應 CLAUDE.md §四 #4 8 項標頭強制檢驗治權慣例。 | SUPERSEDED |
| v0.1 | 2026-05-18 | Codex | 首版：依憲章 §14.7-L 入憲，比對 core+convex 150 × 9 表之 FinMind `api_rows/api_min/api_max` 與 DB `db_rows/db_min/db_max`；支援 `--snapshot-in/--snapshot-out`、`--strict` exit 1、source-empty 合法分流與 targeted backfill commands；`--include-fred` 另比對 FRED 四序列 valid numeric observations。 | SUPERSEDED |
================================================================================
"""
CONSTITUTION_VER = "v6.0.0"
TOOL_VER = "v0.4"

# v0.3 §6.8.8-D / §14.7-AQ 全市場驗證模式之治權常數
# （與 sovereign_sync_engine.FULL_MARKET_REASON_MIN_CHARS / FULL_MARKET_REQUIRED_UNIVERSE 對映；
#  獨立定義以避免 maintenance → ingestion 之反向耦合 import）
FULL_MARKET_REASON_MIN_CHARS = 12
FULL_MARKET_REQUIRED_UNIVERSE = "full"

# v0.4 §0.4 [Hybrid Observability] 進度心跳預設值（每 N 個個股印一條 progress）
# 全市場 ~2,798 支 × default 100 ≈ 28 條 heartbeat；N=0 為靜默模式相容 v0.3
PROGRESS_INTERVAL_DEFAULT = 100

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

_THIS_FILE = Path(__file__).resolve()
_MAINTENANCE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _MAINTENANCE_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from core.path_setup import get_report_dir
    from core.db_utils import get_db_connection, get_core_stocks_from_db, record_lifecycle
    from core.data_schema import DATASET_REGISTRY, FINMIND_API_TABLES
    from core.finmind_client import FinMindClient
except ImportError as exc:
    print(f"❌ 核心組件導入失敗: {exc}")
    sys.exit(1)


STRICT_SOURCE_START_DATE = "1990-01-01"
DEFAULT_THROTTLE_PER_HOUR = 5500
FRED_PAGE_LIMIT = 100000
FRED_SERIES = ["DFF", "UNRATE", "T10Y2Y", "VIXCLS"]
STOCK_LEVEL_DATASETS = [
    name for name in FINMIND_API_TABLES
    if name != "TaiwanStockInfo"
]


class SimpleThrottle:
    """Small local throttle for audit probes; keeps audit under §7 request policy."""

    def __init__(self, max_per_hour=DEFAULT_THROTTLE_PER_HOUR):
        self.max_per_hour = max_per_hour
        self.window = deque()

    def acquire(self):
        now = time.time()
        while self.window and self.window[0] < now - 3600:
            self.window.popleft()
        if len(self.window) >= self.max_per_hour:
            sleep_for = 3600 - (now - self.window[0]) + 1
            print(f"⏸  audit throttle sleep {sleep_for:.0f}s ({len(self.window)}/{self.max_per_hour})")
            time.sleep(sleep_for)
        self.window.append(time.time())


class SourceAvailabilityAuditor:
    def __init__(self, start_date=STRICT_SOURCE_START_DATE, throttle_per_hour=DEFAULT_THROTTLE_PER_HOUR,
                 snapshot_in=None, snapshot_out=None, drift_tolerance=3,
                 special_full_market_reason=None,
                 progress_interval=PROGRESS_INTERVAL_DEFAULT):
        # v0.2: drift_tolerance = audit 時點漂移容忍（per §6.8.8-C / §14.7-AP）
        # 預設 3 個日曆日（覆蓋週末 + 1 工作日緩衝）；0 為嚴格模式相容 v0.1
        # v0.3: special_full_market_reason = §6.8.8-D / §14.7-AQ 全市場 audit 治理理由
        # 對齊 §6.8.7 第 (4) 條五類合法情境；audit 時 audit trail 留存
        # v0.4: progress_interval = §0.4 [Hybrid Observability] 進度心跳頻率
        # 每 N 個 stock 印一次 progress line；N=0 為靜默模式相容 v0.3
        self.constitution_ver = CONSTITUTION_VER
        self.tool_ver = TOOL_VER
        self.start_date = start_date
        self.drift_tolerance = max(0, int(drift_tolerance))
        self.special_full_market_reason = (special_full_market_reason or "").strip() or None
        self.progress_interval = max(0, int(progress_interval))
        self.client = FinMindClient()
        self.throttle = SimpleThrottle(max_per_hour=throttle_per_hour)
        self.snapshot_in = Path(snapshot_in) if snapshot_in else None
        self.snapshot_out = Path(snapshot_out) if snapshot_out else None
        self.api_snapshot = self._load_snapshot(self.snapshot_in)
        self.results = []
        self.fred_results = []
        self.api_errors = 0
        self.checked = 0
        self.mismatch = 0
        self.source_empty_ok = 0
        self.time_drift_ok = 0  # v0.2 新增：§6.8.8-C 時點漂移容忍計數器
        self.fred_checked = 0
        self.fred_mismatch = 0
        self.fred_time_drift_ok = 0  # v0.2 新增：FRED 時點漂移容忍計數器
        self.fred_api_errors = 0

    def _load_snapshot(self, path):
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "items" in payload:
            items = payload["items"]
        elif isinstance(payload, dict) and "finmind" in payload:
            items = payload["finmind"]
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError(f"Unsupported snapshot format: {path}")
        return {
            (str(item["stock_id"]), item.get("dataset") or item.get("table")): item
            for item in items
        }

    def _dump_snapshot(self):
        if not self.snapshot_out:
            return
        self.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        items = [
            {
                "stock_id": r["stock_id"],
                "dataset": r["dataset"],
                "api_rows": r["api_rows"],
                "api_min": r["api_min"],
                "api_max": r["api_max"],
            }
            for r in self.results
            if r["api_status"] == "OK"
        ]
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "start_date": self.start_date,
            "items": items,
        }
        with open(self.snapshot_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _resolve_stocks(self, stock_id=None, universe="core"):
        if stock_id:
            return [str(stock_id)]
        if universe == "core":
            return list(get_core_stocks_from_db(tiers=("core_universe", "convex_universe")))
        if universe == "full":
            # v0.3 §6.8.8-D / §14.7-AQ: 對齊 sovereign_sync_engine.UNIVERSE_TIERS["full"] SSOT
            # 全市場 = core ∪ convex ∪ research ∪ quarantine ≈ 2,798 支
            return list(get_core_stocks_from_db(
                tiers=("core_universe", "convex_universe", "research_universe", "quarantine_universe")
            ))
        raise ValueError(f"Unsupported universe: {universe}; allowed: core / full")

    def _resolve_datasets(self, dataset=None, all_datasets=False):
        if dataset:
            if dataset not in STOCK_LEVEL_DATASETS:
                raise ValueError(f"Unsupported stock-level dataset: {dataset}")
            return [dataset]
        if all_datasets:
            return list(STOCK_LEVEL_DATASETS)
        return ["TaiwanStockPrice", "TaiwanStockPriceAdj", "TaiwanStockFinancialStatements"]

    def _api_summary_from_snapshot(self, stock_id, dataset):
        item = self.api_snapshot.get((str(stock_id), dataset))
        if not item:
            return None
        return {
            "api_status": "OK",
            "api_rows": int(item.get("api_rows") or 0),
            "api_min": item.get("api_min"),
            "api_max": item.get("api_max"),
        }

    def _fetch_api_summary(self, stock_id, dataset):
        cached = self._api_summary_from_snapshot(stock_id, dataset)
        if cached is not None:
            return cached

        self.throttle.acquire()
        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "start_date": self.start_date,
        }
        if self.client.token:
            params["token"] = self.client.token
        res = requests.get(self.client.api_url, params=params, headers=self.client.headers, timeout=30)
        res.raise_for_status()
        payload = res.json()
        if payload.get("msg") not in (None, "", "success"):
            raise RuntimeError(f"FinMind app-level error: {payload.get('msg')}")
        data = payload.get("data", [])
        if not data:
            return {"api_status": "OK", "api_rows": 0, "api_min": None, "api_max": None}

        df = pd.DataFrame(data)
        if "date" not in df.columns:
            raise ValueError(f"{dataset} API response missing date column")
        dates = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna()
        if dates.empty:
            return {"api_status": "OK", "api_rows": 0, "api_min": None, "api_max": None}
        return {
            "api_status": "OK",
            "api_rows": int(len(df[df["date"].notna()])),
            "api_min": str(min(dates)),
            "api_max": str(max(dates)),
        }

    def _db_summary(self, stock_id, dataset, conn):
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"Dataset not in DATASET_REGISTRY: {dataset}")
        columns = DATASET_REGISTRY[dataset]["columns"]
        if "stock_id" not in columns or "date" not in columns:
            raise ValueError(f"Dataset is not stock-level/date-level: {dataset}")
        with conn.cursor() as cur:
            cur.execute(
                f'''
                SELECT COUNT(*), MIN("date"), MAX("date")
                FROM "{dataset}"
                WHERE "stock_id" = %s
                ''',
                (stock_id,),
            )
            rows, min_date, max_date = cur.fetchone()
        return {
            "db_rows": int(rows or 0),
            "db_min": str(min_date) if isinstance(min_date, (date, datetime)) else None,
            "db_max": str(max_date) if isinstance(max_date, (date, datetime)) else None,
        }

    def _fetch_fred_summary(self, series_id):
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise RuntimeError("FRED_API_KEY missing")
        observations = []
        offset = 0
        while True:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "limit": FRED_PAGE_LIMIT,
                "offset": offset,
                "sort_order": "asc",
            }
            res = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
            res.raise_for_status()
            page = res.json().get("observations", [])
            observations.extend(page)
            if len(page) < FRED_PAGE_LIMIT:
                break
            offset += FRED_PAGE_LIMIT

        valid_dates = []
        for item in observations:
            value = item.get("value")
            if value in (None, "."):
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            valid_dates.append(pd.to_datetime(item.get("date"), errors="coerce").date())
        valid_dates = [d for d in valid_dates if pd.notna(d)]
        return {
            "series_id": series_id,
            "api_valid_rows": len(valid_dates),
            "api_valid_min": str(min(valid_dates)) if valid_dates else None,
            "api_valid_max": str(max(valid_dates)) if valid_dates else None,
        }

    def _db_fred_summary(self, series_id, conn):
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT COUNT(*), MIN("date"), MAX("date")
                FROM "FredData"
                WHERE "series_id" = %s AND "value" IS NOT NULL
                ''',
                (series_id,),
            )
            rows, min_date, max_date = cur.fetchone()
        return {
            "db_valid_rows": int(rows or 0),
            "db_valid_min": str(min_date) if isinstance(min_date, (date, datetime)) else None,
            "db_valid_max": str(max_date) if isinstance(max_date, (date, datetime)) else None,
        }

    def _is_time_drift_ok(self, api_rows, db_rows, api_max, db_max):
        """v0.2 §6.8.8-C 時點漂移容忍判定：
        (api_date_max - db_date_max).days ≤ N 且 abs(api_rows - db_rows) ≤ N → TIME_DRIFT_OK
        """
        if self.drift_tolerance <= 0:
            return False
        if api_max is None or db_max is None:
            return False
        try:
            date_drift_days = (api_max - db_max).days
            row_drift = abs(api_rows - db_rows)
            # 必須 API 領先 DB（DB 不應超前 API）且漂移在容忍範圍內
            return 0 <= date_drift_days <= self.drift_tolerance and row_drift <= self.drift_tolerance
        except Exception:
            return False

    def _classify(self, row):
        if row["api_status"] != "OK":
            return "API_ERROR"
        if row["api_rows"] == 0:
            return "SOURCE_EMPTY_OK" if row["db_rows"] == 0 else "SOURCE_EMPTY_DB_HAS_ROWS"
        if (
            row["api_rows"] == row["db_rows"]
            and row["api_min"] == row["db_min"]
            and row["api_max"] == row["db_max"]
        ):
            return "OK"
        # v0.2: 嘗試 §6.8.8-C 時點漂移容忍判定
        if (
            row["api_min"] == row["db_min"]  # 起點對齊（未漏抓歷史）
            and self._is_time_drift_ok(row["api_rows"], row["db_rows"], row["api_max"], row["db_max"])
        ):
            return "TIME_DRIFT_OK"
        return "MISMATCH"

    def _classify_fred(self, row):
        if row["api_status"] != "OK":
            return "API_ERROR"
        if (
            row["api_valid_rows"] == row["db_valid_rows"]
            and row["api_valid_min"] == row["db_valid_min"]
            and row["api_valid_max"] == row["db_valid_max"]
        ):
            return "OK"
        # v0.2: 嘗試 §6.8.8-C 時點漂移容忍判定
        if (
            row["api_valid_min"] == row["db_valid_min"]  # 起點對齊
            and self._is_time_drift_ok(row["api_valid_rows"], row["db_valid_rows"],
                                       row["api_valid_max"], row["db_valid_max"])
        ):
            return "TIME_DRIFT_OK"
        return "MISMATCH"

    def run(self, stock_id=None, universe="core", dataset=None, all_datasets=False, strict=True, include_fred=False):
        stocks = self._resolve_stocks(stock_id=stock_id, universe=universe)
        datasets = self._resolve_datasets(dataset=dataset, all_datasets=all_datasets)
        task_name = f"audit_source_availability_{stock_id or universe}"
        total_stocks = len(stocks)
        # v0.4 §0.4 [Hybrid Observability] 進度心跳開銷
        run_start = time.time()
        if self.progress_interval > 0 and total_stocks > 0:
            print(
                f"🔍 audit start | universe={universe} | stocks={total_stocks} | datasets={len(datasets)} | "
                f"include_fred={include_fred} | drift_tolerance={self.drift_tolerance} | "
                f"progress_interval={self.progress_interval}",
                flush=True,
            )

        with record_lifecycle(task_name, category="maintenance", stock_id=stock_id or "SYSTEM") as lifecycle:
            conn = get_db_connection()
            try:
                for idx, sid in enumerate(stocks, start=1):
                    for ds in datasets:
                        base = {"stock_id": str(sid), "dataset": ds}
                        try:
                            api = self._fetch_api_summary(str(sid), ds)
                            db = self._db_summary(str(sid), ds, conn)
                            row = {**base, **api, **db}
                        except Exception as exc:
                            self.api_errors += 1
                            row = {
                                **base,
                                "api_status": "ERROR",
                                "api_rows": None,
                                "api_min": None,
                                "api_max": None,
                                "db_rows": None,
                                "db_min": None,
                                "db_max": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        row["status"] = self._classify(row)
                        self.results.append(row)
                        self.checked += 1
                        if row["status"] == "SOURCE_EMPTY_OK":
                            self.source_empty_ok += 1
                        elif row["status"] == "TIME_DRIFT_OK":
                            self.time_drift_ok += 1
                        elif row["status"] not in {"OK"}:
                            self.mismatch += 1
                    # v0.4 §0.4 [Hybrid Observability] 進度心跳：每 N 個 stock 印一條
                    if self.progress_interval > 0 and (idx % self.progress_interval == 0 or idx == total_stocks):
                        elapsed = time.time() - run_start
                        pct = (idx / total_stocks * 100.0) if total_stocks else 0.0
                        eta_sec = (elapsed / idx * (total_stocks - idx)) if idx > 0 else 0
                        print(
                            f"🔍 progress | stocks={idx}/{total_stocks} ({pct:.1f}%) | "
                            f"checked={self.checked} | source_empty_ok={self.source_empty_ok} | "
                            f"time_drift_ok={self.time_drift_ok} | mismatch={self.mismatch} | "
                            f"api_errors={self.api_errors} | elapsed={elapsed:.0f}s | eta={eta_sec:.0f}s",
                            flush=True,
                        )
                if include_fred:
                    for series_id in FRED_SERIES:
                        base = {"series_id": series_id}
                        try:
                            api = self._fetch_fred_summary(series_id)
                            db = self._db_fred_summary(series_id, conn)
                            row = {**base, "api_status": "OK", **api, **db}
                        except Exception as exc:
                            self.fred_api_errors += 1
                            row = {
                                **base,
                                "api_status": "ERROR",
                                "api_valid_rows": None,
                                "api_valid_min": None,
                                "api_valid_max": None,
                                "db_valid_rows": None,
                                "db_valid_min": None,
                                "db_valid_max": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        row["status"] = self._classify_fred(row)
                        self.fred_results.append(row)
                        self.fred_checked += 1
                        if row["status"] == "TIME_DRIFT_OK":
                            self.fred_time_drift_ok += 1
                        elif row["status"] != "OK":
                            self.fred_mismatch += 1
            finally:
                conn.close()

            self._dump_snapshot()
            verdict = self._verdict(strict=strict)
            if verdict == "FAILED" and hasattr(lifecycle, "mark_failed"):
                lifecycle.mark_failed(f"strict source availability mismatch={self.mismatch}, api_errors={self.api_errors}")
            elif verdict == "WARNING" and hasattr(lifecycle, "mark_warning"):
                lifecycle.mark_warning(f"source availability mismatch={self.mismatch}, api_errors={self.api_errors}")
            self._write_report(verdict, stocks, datasets, universe=universe)
            self._print_summary(verdict, universe=universe)
            return verdict

    def _verdict(self, strict=True):
        if self.api_errors > 0 or self.fred_api_errors > 0:
            return "FAILED"
        if self.mismatch > 0 or self.fred_mismatch > 0:
            return "FAILED" if strict else "WARNING"
        return "PERFECT"

    def _write_report(self, verdict, stocks, datasets, universe="core"):
        report_path = get_report_dir() / f"source_availability_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        mismatches = [r for r in self.results if r["status"] not in {"OK", "SOURCE_EMPTY_OK", "TIME_DRIFT_OK"}]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Strict source availability audit\n\n")
            f.write(f"- **time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"- **constitution**: 系統架構大憲章_{self.constitution_ver}.md "
                f"§14.7-L + §6.8.8-C + §14.7-AP + §6.8.8-D + §14.7-AQ\n"
            )
            f.write(f"- **tool**: audit_source_availability {self.tool_ver}\n")
            f.write(f"- **start_date**: {self.start_date}\n")
            f.write(f"- **drift_tolerance**: {self.drift_tolerance} day(s) (§6.8.8-C; 0 = strict)\n")
            f.write(f"- **scope**: stocks={len(stocks)} (universe={universe}), datasets={len(datasets)}\n")
            if universe == FULL_MARKET_REQUIRED_UNIVERSE and self.special_full_market_reason:
                f.write(
                    f"- **special_full_market_reason**: {self.special_full_market_reason} "
                    f"(§6.8.7 第 (4) 條 / §6.8.8-D)\n"
                )
            f.write(f"- **verdict**: **{verdict}**\n")
            f.write(
                f"- **summary**: checked={self.checked}, source_empty_ok={self.source_empty_ok}, "
                f"time_drift_ok={self.time_drift_ok}, mismatch={self.mismatch}, api_errors={self.api_errors}\n\n"
            )
            if self.fred_results:
                f.write(
                    f"- **fred_summary**: checked={self.fred_checked}, "
                    f"time_drift_ok={self.fred_time_drift_ok}, "
                    f"mismatch={self.fred_mismatch}, api_errors={self.fred_api_errors}\n\n"
                )
            f.write("## Mismatches\n\n")
            if not mismatches:
                f.write("None.\n\n")
            else:
                f.write("| stock_id | dataset | status | api_rows | api_min | api_max | db_rows | db_min | db_max |\n")
                f.write("|---|---|---|---:|---|---|---:|---|---|\n")
                for r in mismatches:
                    f.write(
                        f"| {r['stock_id']} | {r['dataset']} | {r['status']} | "
                        f"{r.get('api_rows')} | {r.get('api_min')} | {r.get('api_max')} | "
                        f"{r.get('db_rows')} | {r.get('db_min')} | {r.get('db_max')} |\n"
                    )
                f.write("\n## Targeted Backfill Commands\n\n")
                for r in mismatches:
                    if r["status"] == "API_ERROR":
                        continue
                    f.write(
                        "```bash\n"
                        f".venv/bin/python scripts/ingestion/sovereign_sync_engine.py --id {r['stock_id']} "
                        f"--dataset {r['dataset']} --strict-source-history\n"
                        "```\n"
                    )
            if self.fred_results:
                f.write("\n## FRED Valid Observation Alignment\n\n")
                f.write("| series_id | status | api_valid_rows | api_valid_min | api_valid_max | db_valid_rows | db_valid_min | db_valid_max |\n")
                f.write("|---|---|---:|---|---|---:|---|---|\n")
                for r in self.fred_results:
                    f.write(
                        f"| {r['series_id']} | {r['status']} | {r.get('api_valid_rows')} | "
                        f"{r.get('api_valid_min')} | {r.get('api_valid_max')} | "
                        f"{r.get('db_valid_rows')} | {r.get('db_valid_min')} | {r.get('db_valid_max')} |\n"
                    )
        self.report_path = report_path

    def _print_summary(self, verdict, universe="core"):
        print("\n" + "🛡️" * 40)
        print(f"🚀 Quantum Finance: strict source availability audit ({self.tool_ver})")
        print("🛡️" * 40)
        print(f"report : {self.report_path}")
        print(f"universe={universe}")
        if universe == FULL_MARKET_REQUIRED_UNIVERSE and self.special_full_market_reason:
            print(f"special_full_market_reason : {self.special_full_market_reason}")
        print(f"drift_tolerance={self.drift_tolerance}")
        print(f"checked={self.checked}")
        print(f"source_empty_ok={self.source_empty_ok}")
        print(f"time_drift_ok={self.time_drift_ok}")
        print(f"mismatch={self.mismatch}")
        print(f"api_errors={self.api_errors}")
        if self.fred_results:
            print(f"fred_checked={self.fred_checked}")
            print(f"fred_time_drift_ok={self.fred_time_drift_ok}")
            print(f"fred_mismatch={self.fred_mismatch}")
            print(f"fred_api_errors={self.fred_api_errors}")
        print(f"verdict={verdict}")
        print("🛡️" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Strict source availability audit (FinMind + FRED)")
    parser.add_argument("--id", type=str, help="single stock_id")
    parser.add_argument("--universe", choices=["core", "full"], default="core",
                        help="authorized universe scope (core ≈ 150；full ≈ 2,798 須附 reason 對齊 §6.8.7 第 (4) 條 / §6.8.8-D)")
    parser.add_argument("--dataset", type=str, help="single FinMind stock-level dataset")
    parser.add_argument("--all", action="store_true", help="audit all FinMind stock-level datasets")
    parser.add_argument("--start-date", default=STRICT_SOURCE_START_DATE,
                        help=f"FinMind source lower bound (default {STRICT_SOURCE_START_DATE})")
    parser.add_argument("--strict", action="store_true", help="exit 1 on any mismatch")
    parser.add_argument("--snapshot-in", help="reuse prior API source snapshot JSON")
    parser.add_argument("--snapshot-out", help="write API source snapshot JSON")
    parser.add_argument("--include-fred", action="store_true",
                        help="also verify FRED DFF/UNRATE/T10Y2Y/VIXCLS valid numeric observations against DB")
    parser.add_argument("--throttle", type=int, default=DEFAULT_THROTTLE_PER_HOUR,
                        help="API requests per hour cap for audit probe")
    parser.add_argument("--drift-tolerance", type=int, default=3,
                        help="§6.8.8-C audit 時點漂移容忍 (預設 3 個日曆日；0 = strict mode)")
    parser.add_argument("--special-full-market-reason", type=str, default=None,
                        help=f"(v0.3 §6.8.8-D / §14.7-AQ) 全市場 audit 治理理由 — 必須 ≥ {FULL_MARKET_REASON_MIN_CHARS} 字元；"
                             "僅在 --universe full 時生效；缺 reason 或字數不足即 exit 1")
    parser.add_argument("--progress-interval", type=int, default=PROGRESS_INTERVAL_DEFAULT,
                        help=f"(v0.4 §0.4) 進度心跳頻率，每 N 個 stock 印一條 progress line（預設 {PROGRESS_INTERVAL_DEFAULT}；"
                             "0 = 靜默模式相容 v0.3）")
    args = parser.parse_args()

    # v0.3 §6.8.8-D / §14.7-AQ preflight 治權檢查（對齊 §6.8.7 第 (4) 條範本）
    if args.universe == FULL_MARKET_REQUIRED_UNIVERSE:
        reason = (args.special_full_market_reason or "").strip()
        if not reason:
            print(f"❌ [§6.8.8-D / §14.7-AQ] --universe full 必須附 --special-full-market-reason \"<≥{FULL_MARKET_REASON_MIN_CHARS} 字理由>\"")
            print("   合法情境：DB rebuild bootstrap / Sovereign rebuild / pre-annual audit / 資料源治權變更 / 合規事件")
            sys.exit(1)
        if len(reason) < FULL_MARKET_REASON_MIN_CHARS:
            print(f"❌ [§6.8.8-D / §14.7-AQ] --special-full-market-reason 長度 {len(reason)} < {FULL_MARKET_REASON_MIN_CHARS} 字元下限")
            sys.exit(1)
    elif args.special_full_market_reason:
        print(f"❌ [§6.8.8-D / §14.7-AQ] --special-full-market-reason 僅在 --universe full 時生效；"
              f"目前 --universe={args.universe}，拒絕執行")
        sys.exit(1)

    auditor = SourceAvailabilityAuditor(
        start_date=args.start_date,
        throttle_per_hour=args.throttle,
        snapshot_in=args.snapshot_in,
        snapshot_out=args.snapshot_out,
        drift_tolerance=args.drift_tolerance,
        special_full_market_reason=args.special_full_market_reason,
        progress_interval=args.progress_interval,
    )
    verdict = auditor.run(
        stock_id=args.id,
        universe=args.universe,
        dataset=args.dataset,
        all_datasets=args.all,
        strict=args.strict,
        include_fred=args.include_fred,
    )
    sys.exit(0 if verdict in {"PERFECT", "WARNING"} else 1)


if __name__ == "__main__":
    main()
