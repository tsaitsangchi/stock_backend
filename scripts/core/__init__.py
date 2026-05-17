"""
core/__init__.py v1.14 (Quantum Finance Sovereign Hub - Constitution v5.4.22 Aligned)
================================================================================
**最後更新日期**: 2026-05-15
**主權狀態**: ACTIVE (憲法 v6.0.0 演進定稿對齊 + 動態四層稽核)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Hub Sovereignty]: 本模組為 `scripts/core` package 之唯一接口中樞，所有
   外部調用必須經由此 25 維全譜路徑接口與 db_utils 公共 API 集中導出。
2. [Boundary Integrity]: 路徑接口維度鎖定為 **25 維**，嚴禁回到 v5.2/v5.3
   時代之 27 維誤宣告（對齊 `path_setup.py v4.44` 之 `ALL_PATHS`）。
3. [Canonical Execution Order]: 維運矩陣必須鏡像憲章 v5.4.22 第二章九步序列：
   `.env` → `path_setup` → `data_schema --init --force` → `core_universe_schema --init`
   → `db_utils` → `audit_supply_chain --include-logs` → `sovereign_sync_engine --seed`
   → `core_universe_builder --dry-run/--commit` → `audit_core_universe`。
4. [No Hardcoded Verdict]: 中樞稽核之最終判定必須依「import / path / db / §6.7」
   四層實況動態計算，**嚴禁硬編碼 PERFECT**（對齊憲章 §5.6 第 3 條「零靜默丟失原則」與
   §1.2「混合觀測條款」之動態主權判定）。
5. [§6.7 SQL Contract]: 核心股名單必須透過 `db_utils.get_core_stocks_from_db()`
   取得；該函式封裝 `core_universe_membership` JOIN `core_universe_snapshot
   WHERE status='committed'`，嚴禁回查 v5.2 時代之 `stocks` 表。
6. [Historical Reference Authority]: 嚴格保留從 v1.0 至今的所有歷史紀錄，
   作為判定系統治權一致性的權威參考。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix - v5.4.22 對齊)

| Step | 維運需求場景 (Scenario) | 權威指令 (Authoritative Command) | 對齊模組 |
| :--- | :--- | :--- | :--- |
| **0**   | [啟動錨點：`.env` 設定確認]                | `PROJECT_ROOT=/home/hugo/project/stock_backend`                                                          | `.env Bootstrap Anchor` |
| **1**   | [環境前置：路徑與目錄自癒]                 | `$ python scripts/core/path_setup.py`                                                                    | `path_setup v4.44` |
| **2**   | [標準序列：Raw API Schema 主權重鑄]        | `$ python scripts/core/data_schema.py --init --force`                                                    | `data_schema v2.11` |
| **2A**  | [離線復原：略過 API 契約探測]              | `$ python scripts/core/data_schema.py --init --force --skip-api-contract`                                | `data_schema v2.11` |
| **2B**  | [標準序列：核心股治理 Schema 初始化]       | `$ python scripts/core/core_universe_schema.py --init`                                                   | `core_universe_schema v0.2` |
| **2C**  | [前置依賴檢查：DB / lifecycle / §6.7 SQL]  | `$ python scripts/core/db_utils.py`                                                                      | `db_utils v2.45` |
| **3**   | [標準序列：Schema 後供應鏈驗收]            | `$ python scripts/maintenance/audit_supply_chain.py --include-logs`                                      | `audit_supply_chain v1.18` |
| **3A**  | [離線供應鏈驗收：僅 DB 實況]               | `$ python scripts/maintenance/audit_supply_chain.py --db-only --include-logs`                            | `audit_supply_chain v1.18` |
| **4**   | [標準序列：市場個股資料種子灌溉]           | `$ python scripts/ingestion/sovereign_sync_engine.py --seed`                                             | `sovereign_sync_engine v1.10` |
| **4B**  | [標準序列：核心股 Universe metadata bootstrap + v0.2 input preflight] | `$ python scripts/core/core_universe_builder.py --dry-run --as-of-date <YYYY-MM-DD>` / `--commit`    | `core_universe_builder v0.2-preflight` |
| **4C**  | [標準序列：核心股結果驗收稽核]             | `$ python scripts/maintenance/audit_core_universe.py --as-of-date <YYYY-MM-DD>`                          | `audit_core_universe v0.1` |
| **5**   | [同步操作：單一標的指定資料集]             | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPrice`               | `sovereign_sync_engine v1.10` |
| **6**   | [同步操作：單一標的核心數據集全同步]       | `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330`                                          | `sovereign_sync_engine v1.10` |
| **7**   | [同步操作：核心股 Universe 全量同步]       | `$ python scripts/ingestion/sovereign_sync_engine.py --universe core` （§6.7 SQL 契約）                  | `sovereign_sync_engine v1.10` |
| **8**   | [同步操作：FRED 宏觀指標全譜同步]          | `$ python scripts/ingestion/sovereign_sync_engine.py --source fred`                                      | `sovereign_sync_engine v1.10` |
| **HUB** | [中樞主權完整性稽核]                       | `$ python scripts/core/__init__.py`                                                                      | `core/__init__ v1.14` |

💡 **範例完整性說明**：矩陣完整對齊憲章 v5.4.22 第二章九步序列；Step 7 必須透過
   §6.7 SQL 契約讀取 `core_universe_membership`，嚴禁查 v5.2 時代之 `stocks` 表。

## ⚔️ 三、五大核心場景對齊 (Five Canonical Scenarios — 憲章 §5.5 第 3 條)
1. **[個股同步]**           : `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330`
2. **[單一資料集同步]**     : `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330 --dataset TaiwanStockPrice`
3. **[單一個股所有資料集]** : `$ python scripts/ingestion/sovereign_sync_engine.py --id 2330`
4. **[所有核心股同步]**     : `$ python scripts/ingestion/sovereign_sync_engine.py --universe core`  (§6.7 SQL)
5. **[FRED 宏觀全量同步]**  : `$ python scripts/ingestion/sovereign_sync_engine.py --source fred`

## 📜 四、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.14** | 2026-05-15 | Codex | **憲法 v5.4.22 演進定稿對齊**：(1) 路徑接口維度由 27 維修正為 **25 維**，
            對齊 `path_setup v4.44` 之 `ALL_PATHS`；(2) 全量維運矩陣重寫為憲章 v5.4.22 九步
            正典序列，新增 Step 2B/2C/3/4B/4C/5/6/7/8；(3) `template_fetcher.py` 全部替換為
            `sovereign_sync_engine.py v1.10`；(4) 中樞稽核改為「import / path / db / §6.7」
            四層動態判定，**移除硬編碼 PERFECT**；(5) 修補既有 latent bug：舊版 import
            `PROJECT_ROOT` 但 `path_setup.py` 僅匯出 `PROJECT_ROOT_CALC`，改為正確匯入並
            重新導出別名；(6) 補上 `get_evaluation_dir()` 相容別名匯入（不計入 25 維）；
            (7) 補上 §6.7 核心同步資產數即時稽核；(8) 補上 `__all__` 對外匯出清單，
            鎖定 25 維路徑接口 + db_utils 公共 API 鏡像匯出邊界。 | **ACTIVE** |
| v1.13 | 2026-05-12 | Antigravity | **治權終極校準**：補全「場景治權」與「範例窮舉」條款，對齊 v5.2 旗艦版憲法新條款。 | ARCHIVED |
| v1.12 | 2026-05-12 | Antigravity | **旗艦化重鑄**：注入「中樞主權校驗」邏輯，對齊全量維運矩陣。 | SUPERSEDED |
| v1.11 | 2026-05-12 | Antigravity | **憲法化對齊**：對齊 v5.2 主權標準。 | SUPERSEDED |
| v1.0  | 2026-04-20 | Antigravity | **主權奠基**：初始接口導出定義。 | ARCHIVED |
================================================================================
"""
import sys
import time
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# 治權常數 (Constitution Constants)
# ──────────────────────────────────────────────────────────────────────────────
CONSTITUTION_VER = "v6.0.0"
HUB_VER = "v1.14"
EXPECTED_PATH_DIM = 25  # 對齊憲章 §1 第 3 條 + §4 25 維清單


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1: path_setup 匯入 (25 維全譜路徑 + Bootstrap Anchor)
#
# 修補 v1.13 latent bug:
#   舊版 `from core.path_setup import PROJECT_ROOT, ...` 會失敗,
#   因為 path_setup.py 實際匯出的是 PROJECT_ROOT_CALC,沒有 PROJECT_ROOT。
#   v1.14 改為匯入 PROJECT_ROOT_CALC 並建立 PROJECT_ROOT 別名以維持下游相容。
# ──────────────────────────────────────────────────────────────────────────────
PATH_IMPORT_OK = False
PATH_IMPORT_ERR = None
ALL_PATHS = []  # safe fallback


def _do_path_import():
    """執行 path_setup 匯入。失敗時啟動 sys.path 自癒並重試一次。"""
    global PATH_IMPORT_OK, PATH_IMPORT_ERR, ALL_PATHS

    last_error = None
    for attempt in (1, 2):
        try:
            from core.path_setup import (
                PROJECT_ROOT_CALC,
                get_root_dir, get_core_dir, get_utils_dir, get_maintenance_dir,
                get_data_dir, get_raw_data_dir, get_ingestion_dir,
                get_feature_dir, get_feature_store_dir,
                get_model_dir, get_model_weights_dir, get_model_scalers_dir,
                get_training_dir, get_archive_dir, get_mlflow_dir,
                get_infer_dir, get_prediction_dir, get_eval_dir, get_evaluation_dir,
                get_output_dir, get_report_dir, get_scratch_dir, get_log_dir,
                get_pipeline_dir, get_monitor_dir, get_test_dir,
                ALL_PATHS as _ALL_PATHS,
                ensure_all_dirs as _ensure_all_dirs,
            )
            ALL_PATHS = _ALL_PATHS
            mod = globals()
            mod.update({
                "PROJECT_ROOT_CALC": PROJECT_ROOT_CALC,
                "PROJECT_ROOT": PROJECT_ROOT_CALC,  # backward-compat alias
                "ensure_all_dirs": _ensure_all_dirs,
                "get_root_dir": get_root_dir,
                "get_core_dir": get_core_dir,
                "get_utils_dir": get_utils_dir,
                "get_maintenance_dir": get_maintenance_dir,
                "get_data_dir": get_data_dir,
                "get_raw_data_dir": get_raw_data_dir,
                "get_ingestion_dir": get_ingestion_dir,
                "get_feature_dir": get_feature_dir,
                "get_feature_store_dir": get_feature_store_dir,
                "get_model_dir": get_model_dir,
                "get_model_weights_dir": get_model_weights_dir,
                "get_model_scalers_dir": get_model_scalers_dir,
                "get_training_dir": get_training_dir,
                "get_archive_dir": get_archive_dir,
                "get_mlflow_dir": get_mlflow_dir,
                "get_infer_dir": get_infer_dir,
                "get_prediction_dir": get_prediction_dir,
                "get_eval_dir": get_eval_dir,
                "get_evaluation_dir": get_evaluation_dir,  # 不計入 25 維,僅相容別名
                "get_output_dir": get_output_dir,
                "get_report_dir": get_report_dir,
                "get_scratch_dir": get_scratch_dir,
                "get_log_dir": get_log_dir,
                "get_pipeline_dir": get_pipeline_dir,
                "get_monitor_dir": get_monitor_dir,
                "get_test_dir": get_test_dir,
            })
            PATH_IMPORT_OK = True
            PATH_IMPORT_ERR = None
            return
        except ImportError as exc:
            last_error = exc
            if attempt == 1:
                _THIS_FILE = Path(__file__).resolve()
                _CORE_DIR = _THIS_FILE.parent
                _SCRIPTS_DIR = _CORE_DIR.parent
                if str(_SCRIPTS_DIR) not in sys.path:
                    sys.path.insert(0, str(_SCRIPTS_DIR))
                continue
            PATH_IMPORT_OK = False
            PATH_IMPORT_ERR = "{}: {}".format(type(last_error).__name__, last_error)
            return


_do_path_import()


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2: db_utils 匯入 (橫切基礎設施 library v2.45 - §3.2 + §6.7)
# ──────────────────────────────────────────────────────────────────────────────
DB_IMPORT_OK = False
DB_IMPORT_ERR = None
LOG_MODE = "MOCK (db_utils unavailable)"

try:
    from core.db_utils import (
        DDL_FETCH_LOG,
        FailureLogger,
        append_failure_json,
        bulk_upsert,
        check_db_health,
        commit_per_day,
        commit_per_stock,
        commit_per_stock_per_day,
        db_connection_check,
        db_session,
        db_transaction,
        dedup_rows,
        ensure_ddl,
        ensure_infrastructure,
        get_all_safe_starts,
        get_connection_params,
        get_core_stocks_from_db,
        get_db_conn,
        get_db_connection,
        get_db_stock_ids,
        get_failure_log_path,
        get_latest_date,
        get_market_safe_start,
        log_fetch_result,
        map_rows_safe,
        record_lifecycle,
        resolve_start_cached,
        safe_commit_rows,
        safe_date,
        safe_float,
        safe_int,
        write_data_audit_log,
        write_evaluation_log,
        write_pipeline_log,
    )
    DB_IMPORT_OK = True
    LOG_MODE = "REAL (DB-Linked / db_utils v2.45)"
except ImportError as exc:
    from contextlib import contextmanager

    DB_IMPORT_ERR = "{}: {}".format(type(exc).__name__, exc)
    LOG_MODE = "MOCK (db_utils unavailable)"

    class _MockLifecycleContext:
        """Fallback context that mimics db_utils._LifecycleContext."""
        def mark_failed(self, *args, **kwargs):
            pass

        def mark_warning(self, *args, **kwargs):
            pass

    @contextmanager
    def record_lifecycle(task_name, **kwargs):
        yield _MockLifecycleContext()

    def write_data_audit_log(*args, **kwargs):
        return None

    def _missing_db_utils(*args, **kwargs):
        raise ImportError(
            "core.db_utils is not importable; install psycopg2-binary or repair "
            "PYTHONPATH. Original error: {}".format(DB_IMPORT_ERR)
        )

    DDL_FETCH_LOG = ""
    FailureLogger = None
    append_failure_json = _missing_db_utils
    bulk_upsert = _missing_db_utils
    check_db_health = _missing_db_utils
    commit_per_day = _missing_db_utils
    commit_per_stock = _missing_db_utils
    commit_per_stock_per_day = _missing_db_utils
    db_connection_check = _missing_db_utils
    db_session = _missing_db_utils
    db_transaction = _missing_db_utils
    dedup_rows = _missing_db_utils
    ensure_ddl = _missing_db_utils
    ensure_infrastructure = _missing_db_utils
    get_all_safe_starts = _missing_db_utils
    get_connection_params = _missing_db_utils
    get_core_stocks_from_db = _missing_db_utils
    get_db_conn = _missing_db_utils
    get_db_connection = _missing_db_utils
    get_db_stock_ids = _missing_db_utils
    get_failure_log_path = _missing_db_utils
    get_latest_date = _missing_db_utils
    get_market_safe_start = _missing_db_utils
    log_fetch_result = _missing_db_utils
    map_rows_safe = _missing_db_utils
    resolve_start_cached = _missing_db_utils
    safe_commit_rows = _missing_db_utils
    safe_date = _missing_db_utils
    safe_float = _missing_db_utils
    safe_int = _missing_db_utils
    write_evaluation_log = _missing_db_utils
    write_pipeline_log = _missing_db_utils


# ──────────────────────────────────────────────────────────────────────────────
# 四層動態稽核 (Four-Layer Dynamic Audit — 對齊 §1.2 / §5.6.3 / §3.2 Step 2C)
# ──────────────────────────────────────────────────────────────────────────────

def _audit_import_layer():
    """Layer A: 模組匯入健康度。"""
    failures, warnings_list = [], []
    if not PATH_IMPORT_OK:
        failures.append("path_setup import failed: {}".format(PATH_IMPORT_ERR))
    if not DB_IMPORT_OK:
        warnings_list.append(
            "db_utils import degraded -> MOCK mode: {}".format(DB_IMPORT_ERR)
        )
    return failures, warnings_list


def _audit_path_layer():
    """Layer B: 25 維路徑接口完整性。"""
    failures, warnings_list = [], []
    info = {"dim": 0, "msg": "deferred"}
    if not PATH_IMPORT_OK:
        warnings_list.append("path dimension check deferred (path_setup unavailable)")
        return failures, warnings_list, info
    dim = len(ALL_PATHS) if ALL_PATHS else 0
    info["dim"] = dim
    if dim != EXPECTED_PATH_DIM:
        msg = "path dimension mismatch: expected {}, got {}".format(EXPECTED_PATH_DIM, dim)
        failures.append(msg)
        info["msg"] = msg
    else:
        info["msg"] = "{}-dimensional path SSOT aligned".format(EXPECTED_PATH_DIM)
    return failures, warnings_list, info


def _audit_db_layer():
    """Layer C: DB 連線健康度。"""
    failures, warnings_list = [], []
    info = {"ok": False, "latency_ms": 0.0, "msg": "deferred"}
    if not DB_IMPORT_OK:
        warnings_list.append("DB health check deferred (db_utils unavailable)")
        return failures, warnings_list, info
    try:
        ok, latency_ms = db_connection_check()
        info["ok"] = bool(ok)
        info["latency_ms"] = float(latency_ms)
        if ok:
            info["msg"] = "DB reachable"
        else:
            info["msg"] = "DB connection check returned False"
            warnings_list.append(info["msg"])
    except Exception as exc:
        info["msg"] = "db_connection_check raised: {}: {}".format(type(exc).__name__, exc)
        warnings_list.append(info["msg"])
    return failures, warnings_list, info


def _audit_core_universe_layer(db_ok):
    """Layer D: §6.7 SQL 契約即時稽核（核心股 Universe 同步資產數）。"""
    failures, warnings_list = [], []
    info = {"scope": "deferred", "count": 0, "msg": "deferred"}
    if not DB_IMPORT_OK or not db_ok:
        info["msg"] = "§6.7 probe deferred (DB layer unavailable)"
        warnings_list.append(info["msg"])
        return failures, warnings_list, info
    try:
        stocks = get_core_stocks_from_db()
        count = len(stocks) if stocks is not None else 0
        info["count"] = count
        if count == 0:
            info["scope"] = "warning"
            info["msg"] = (
                "§6.7 returned 0 rows (bootstrap warning — "
                "core_universe_builder not committed yet)"
            )
            warnings_list.append(info["msg"])
        else:
            info["scope"] = "ok"
            info["msg"] = "§6.7 core_universe ∪ convex_universe = {} stocks".format(count)
    except Exception as exc:
        info["scope"] = "failed"
        info["msg"] = "§6.7 query raised: {}: {}".format(type(exc).__name__, exc)
        failures.append(info["msg"])
    return failures, warnings_list, info


def _compute_verdict(failures, warnings_list):
    if failures:
        return "FAILED"
    if warnings_list:
        return "WARNING"
    return "PERFECT"


def run_sovereign_hub_audit():
    """執行中樞主權完整性稽核 (v1.14 / 憲法 v5.4.22)。

    動態四層稽核 (import / path / db / §6.7) — 嚴禁硬編碼 PERFECT。

    Returns:
        str: verdict in {"PERFECT", "WARNING", "FAILED"}
    """
    start_time = time.time()
    all_failures = []
    all_warnings = []

    with record_lifecycle(
        "sovereign_hub_audit_v1.14",
        category="governance",
        stock_id="SYSTEM",
    ) as lc:
        # Layer A: import sanity
        fa, wa = _audit_import_layer()
        all_failures.extend(fa)
        all_warnings.extend(wa)

        # Layer B: 25-dimensional path SSOT
        fb, wb, path_info = _audit_path_layer()
        all_failures.extend(fb)
        all_warnings.extend(wb)

        # Layer C: DB infrastructure
        fc, wc, db_info = _audit_db_layer()
        all_failures.extend(fc)
        all_warnings.extend(wc)

        # Layer D: §6.7 core universe SQL contract
        fd, wd, scope_info = _audit_core_universe_layer(db_info["ok"])
        all_failures.extend(fd)
        all_warnings.extend(wd)

        # 旁系: data_audit_log 寫入測試 (best-effort, §1.2 補強)
        try:
            write_data_audit_log(
                "HUB_AUDIT",
                "SYSTEM",
                datetime.now().strftime("%Y-%m-%d"),
                "AUDIT_" + HUB_VER,
                1,
            )
        except Exception as exc:
            all_warnings.append(
                "data_audit_log downgraded: {}: {}".format(type(exc).__name__, exc)
            )

        # 動態 lifecycle 標記 (對齊 §1.2 / §5.6.3 - 嚴禁硬編碼 PERFECT)
        for f in all_failures:
            lc.mark_failed(f)
        for w in all_warnings:
            lc.mark_warning(w)

        verdict = _compute_verdict(all_failures, all_warnings)
        latency = (time.time() - start_time) * 1000

        # 預先格式化避免 f-string 跳脫風險
        db_latency_str = ""
        if db_info["ok"]:
            db_latency_str = " / {:.2f} ms".format(db_info["latency_ms"])
        scope_icons = {"ok": "✅", "warning": "⚠️ ", "failed": "❌", "deferred": "⏸️ "}
        scope_icon = scope_icons.get(scope_info["scope"], "❔")
        path_icon = "✅" if path_info["dim"] == EXPECTED_PATH_DIM else "❌"
        db_icon = "✅" if db_info["ok"] else "⚠️ "
        path_import_label = "✅ OK" if PATH_IMPORT_OK else "❌ FAILED"
        db_import_label = "✅ OK" if DB_IMPORT_OK else "⚠️  MOCK"

        # ── 終端摘要報告 (對齊 §5.6 第 2 條 詳細結果展示) ──
        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 系統治權中樞 ({})".format(HUB_VER))
        print("🛡️" * 40)
        print("治權基準 : 系統架構大憲章_{}.md".format(CONSTITUTION_VER))
        print("核心職責 : 25 維全譜路徑接口 + db_utils 橫切 library 鏡像匯出")
        print("─" * 80)
        print("📊 中樞主權稽核摘要報告 (Hub Sovereignty Report — Four-Layer Dynamic Audit)")
        print("─" * 80)

        # Layer A
        path_err_tail = " ({})".format(PATH_IMPORT_ERR) if PATH_IMPORT_ERR else ""
        db_err_tail = " ({})".format(DB_IMPORT_ERR) if DB_IMPORT_ERR else ""
        print("📦 [Layer A] path_setup 匯入   : {}{}".format(path_import_label, path_err_tail))
        print("📦 [Layer A] db_utils 匯入     : {}{}".format(db_import_label, db_err_tail))

        # Layer B
        print("📐 [Layer B] 路徑接口維度       : {} {} 維 ({})".format(
            path_icon, path_info["dim"], path_info["msg"]
        ))

        # Layer C
        print("🔌 [Layer C] DB 連線狀態        : {} {}{}".format(
            db_icon, db_info["msg"], db_latency_str
        ))

        # Layer D (§6.7)
        print("📈 [Layer D] §6.7 核心同步資產 : {} {} 支 — {}".format(
            scope_icon, scope_info["count"], scope_info["msg"]
        ))

        # Meta
        print("📝 混合日誌模式                 : {}".format(LOG_MODE))
        print("🕒 稽核延遲                     : {:.2f} ms".format(latency))
        print("⚖️  系統主權狀態                 : {} (憲法 {} 動態判定)".format(
            verdict, CONSTITUTION_VER
        ))

        if all_failures:
            print("─" * 80)
            print("❌ 違憲項目 (Failures):")
            for f in all_failures:
                print("   - {}".format(f))
        if all_warnings:
            print("─" * 80)
            print("⚠️  警告項目 (Warnings):")
            for w in all_warnings:
                print("   - {}".format(w))
        print("─" * 80)

        print("\n💡 治權維運建議 (Reference Information):")
        print("1. [治權提示]: 本模組為系統唯一接口導出中樞，嚴禁繞過 core/__init__.py "
              "調用 25 維路徑或 db_utils 公共 API。")
        print("2. [序列提示]: 請依憲章 {} 第二章九步序列執行 (.env → path_setup → "
              "data_schema → core_universe_schema → db_utils → audit_supply_chain → "
              "sovereign_sync_engine --seed → core_universe_builder → "
              "audit_core_universe)。".format(CONSTITUTION_VER))
        print("3. [§6.7 提示]: 核心股名單必須透過 db_utils.get_core_stocks_from_db() "
              "取得，封裝 core_universe_membership JOIN core_universe_snapshot "
              "WHERE status='committed'，嚴禁查 v5.2 時代之 stocks 表。")
        print("4. [動態判定]: 本中樞之 PERFECT/WARNING/FAILED 為四層 "
              "(import/path/db/§6.7) 動態計算，禁止硬編碼。")
        print("─" * 80 + "\n")

        return verdict


# ──────────────────────────────────────────────────────────────────────────────
# Public Re-exports (對外匯出清單)
# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Constitution constants
    "CONSTITUTION_VER", "HUB_VER", "EXPECTED_PATH_DIM",
    # Path interfaces (25 dims + 1 backward-compat alias get_evaluation_dir)
    "PROJECT_ROOT", "PROJECT_ROOT_CALC", "ALL_PATHS", "ensure_all_dirs",
    "get_root_dir", "get_core_dir", "get_utils_dir", "get_maintenance_dir",
    "get_data_dir", "get_raw_data_dir", "get_ingestion_dir",
    "get_feature_dir", "get_feature_store_dir",
    "get_model_dir", "get_model_weights_dir", "get_model_scalers_dir",
    "get_training_dir", "get_archive_dir", "get_mlflow_dir",
    "get_infer_dir", "get_prediction_dir", "get_eval_dir", "get_evaluation_dir",
    "get_output_dir", "get_report_dir", "get_scratch_dir", "get_log_dir",
    "get_pipeline_dir", "get_monitor_dir", "get_test_dir",
    # db_utils public API (§3.2 + §6.7)
    "DDL_FETCH_LOG", "FailureLogger",
    "append_failure_json", "bulk_upsert", "check_db_health",
    "commit_per_day", "commit_per_stock", "commit_per_stock_per_day",
    "db_connection_check", "db_session", "db_transaction",
    "dedup_rows", "ensure_ddl", "ensure_infrastructure",
    "get_all_safe_starts", "get_connection_params",
    "get_core_stocks_from_db", "get_db_conn", "get_db_connection",
    "get_db_stock_ids", "get_failure_log_path",
    "get_latest_date", "get_market_safe_start",
    "log_fetch_result", "map_rows_safe", "record_lifecycle",
    "resolve_start_cached", "safe_commit_rows",
    "safe_date", "safe_float", "safe_int",
    "write_data_audit_log", "write_evaluation_log", "write_pipeline_log",
    # Hub interfaces & diagnostics
    "run_sovereign_hub_audit", "LOG_MODE",
    "PATH_IMPORT_OK", "PATH_IMPORT_ERR", "DB_IMPORT_OK", "DB_IMPORT_ERR",
]


if __name__ == "__main__":
    verdict = run_sovereign_hub_audit()
    # 對齊憲章 §3.2 Step 2C 接受標準:
    #   PERFECT/WARNING -> exit 0 (允許進入下一步)
    #   FAILED          -> exit 1 (阻斷)
    sys.exit(0 if verdict in ("PERFECT", "WARNING") else 1)
