"""
path_setup.py v4.44 (Quantum Finance Bootstrap-Aligned Path SSOT Edition)
================================================================================
**最後更新日期**: 2026-05-15
**主權狀態**: PERFECT (憲法 v5.4.22 啟動治理對齊 + 同日 hub 補充相容)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. [Path Sovereignty]: 確立絕對物理基準 (PROJECT_ROOT)，所有子路徑接口必須以此為唯一根節點擴展。
2. [Self-Healing Mechanism]: 具備自動化目錄重建與權限修正能力，確保系統環境「零配置」主權啟動。
3. [Historical Reference Authority]: 保留從 v1.0 至今的所有歷史，作為判定系統物理架構正確性的基準。
4. [Boundary Integrity]: 確保 25 維全譜路徑接口與實體目錄 100% 同步，封印任何執行鏈中斷之可能性。
5. [Bootstrap Anchor Alignment]: `.env` 僅作為 Bootstrap Anchor 錨定 `PROJECT_ROOT`；`path_setup.py` 保持 Path SSOT，負責從根目錄推導 25 維路徑。
6. [Hub Compatibility]: 本檔之 `PROJECT_ROOT_CALC`、25 維 `get_*_dir()` 與 `get_evaluation_dir()` 相容別名、`ALL_PATHS`、`ensure_all_dirs()` 為 `core/__init__.py v1.14` hub 鏡像匯出之 import 邊界；任何改名或移除都會直接破壞 hub 治權。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix - 100% Coverage)
| 維運需求場景 (Scenario)   | 權威指令 / 建議用法 (Exhaustive Examples)                             | 對齊模組 |
| :----------------------- | :-------------------------------------------------------------------- | :--- |
| **1. [路徑治理：全維度稽核]** | `$ python scripts/core/path_setup.py`                                 | path_setup v4.44 |
| **2. [個股同步：環境路徑初始化]** | `$ python scripts/core/path_setup.py`                                 | path_setup v4.44 |
| **3. [啟動錨點：`.env` 與 `PROJECT_ROOT` 對齊驗證]** | `$ python scripts/core/path_setup.py`                                 | path_setup v4.44 |

## 📜 三、全修訂歷程 (Full Revision History - 舊詳細參考)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v4.44** | 2026-05-15 | Codex | **憲法 v5.4.22 同日補充標籤對齊 + 2026-05-16 bootstrap 診斷補正**：標頭由「憲法 v5.4.21 啟動治理對齊」升至「憲法 v5.4.22 啟動治理對齊 + 同日 hub 補充相容」；新增 [Hub Compatibility] 核心定義第 6 條，明定本檔對 `core/__init__.py v1.14` hub 之 import 邊界承諾（`PROJECT_ROOT_CALC` / 25 維 `get_*_dir()` / `get_evaluation_dir()` / `ALL_PATHS` / `ensure_all_dirs()`）。2026-05-16 補正 DB logging hook 不可用時的 `BOOTSTRAP-DEFERRED (DB hook unavailable: ...)` 診斷文字；**API、ALL_PATHS、自癒邏輯與 `_evaluate_anchor()` 皆無變更**。 | **ACTIVE** |
| v4.43 | 2026-05-14 | Codex | **憲法 v5.4.21 啟動治理對齊**：補齊 Bootstrap Anchor / Path SSOT 契約；新增 `get_evaluation_dir()` 相容別名；`.env` 錨點異常不再回報 PERFECT；DB schema 尚未初始化時進入 `BOOTSTRAP-DEFERRED`；真實 DB audit log 寫入失敗才降級為 warning，不阻斷路徑自癒。 | SUPERSEDED |
| v4.42 | 2026-05-13 | Antigravity | **憲法 v5.4 對齊**：校正治理維度為 25 維；修正標頭為對齊 v5.4；新增 .env 錨點對齊校驗邏輯。 | SUPERSEDED |
| v4.41 | 2026-05-12 | Antigravity | 主權完備化：補全全場景路徑維運矩陣。 | SUPERSEDED |
| v1.0 | 2026-04-20 | Antigravity | 主權奠基：初始路徑管理版本。 | ARCHIVED |
================================================================================
"""
import os, sys, time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ── 系統級架構引導 ──
_THIS_FILE = Path(__file__).resolve()
_CORE_DIR = _THIS_FILE.parent
_SCRIPTS_DIR = _CORE_DIR.parent
PROJECT_ROOT_CALC = _SCRIPTS_DIR.parent

# 載入 .env 進行錨點稽核
load_dotenv(PROJECT_ROOT_CALC / ".env")
PROJECT_ROOT_ENV = os.getenv("PROJECT_ROOT")

# 25 維路徑接口註冊中心 (v4.44 完備版)
def get_root_dir(): return PROJECT_ROOT_CALC
def get_core_dir(): return PROJECT_ROOT_CALC / "scripts" / "core"
def get_utils_dir(): return PROJECT_ROOT_CALC / "scripts" / "utils"
def get_maintenance_dir(): return PROJECT_ROOT_CALC / "scripts" / "maintenance"
def get_data_dir(): return PROJECT_ROOT_CALC / "data"
def get_raw_data_dir(): return PROJECT_ROOT_CALC / "data" / "raw"
def get_ingestion_dir(): return PROJECT_ROOT_CALC / "data" / "ingestion"
def get_feature_dir(): return PROJECT_ROOT_CALC / "data" / "features"
def get_feature_store_dir(): return PROJECT_ROOT_CALC / "data" / "feature_store"
def get_model_dir(): return PROJECT_ROOT_CALC / "models"
def get_model_weights_dir(): return PROJECT_ROOT_CALC / "models" / "weights"
def get_model_scalers_dir(): return PROJECT_ROOT_CALC / "models" / "scalers"
def get_training_dir(): return PROJECT_ROOT_CALC / "models" / "training"
def get_archive_dir(): return PROJECT_ROOT_CALC / "models" / "archive"
def get_mlflow_dir(): return PROJECT_ROOT_CALC / "models" / "mlflow"
def get_infer_dir(): return PROJECT_ROOT_CALC / "inference"
def get_prediction_dir(): return PROJECT_ROOT_CALC / "inference" / "predictions"
def get_eval_dir(): return PROJECT_ROOT_CALC / "inference" / "evaluation"
def get_evaluation_dir(): return get_eval_dir()  # Legacy alias; not counted as an extra path dimension.
def get_output_dir(): return PROJECT_ROOT_CALC / "outputs"
def get_report_dir(): return PROJECT_ROOT_CALC / "reports"
def get_scratch_dir(): return PROJECT_ROOT_CALC / "scratch"
def get_log_dir(): return PROJECT_ROOT_CALC / "logs"
def get_pipeline_dir(): return PROJECT_ROOT_CALC / "logs" / "pipeline"
def get_monitor_dir(): return PROJECT_ROOT_CALC / "logs" / "monitor"
def get_test_dir(): return PROJECT_ROOT_CALC / "tests"

ALL_PATHS = [
    get_root_dir(), get_core_dir(), get_utils_dir(), get_maintenance_dir(),
    get_data_dir(), get_raw_data_dir(), get_ingestion_dir(),
    get_feature_dir(), get_feature_store_dir(),
    get_model_dir(), get_model_weights_dir(), get_model_scalers_dir(),
    get_training_dir(), get_archive_dir(), get_mlflow_dir(),
    get_infer_dir(), get_prediction_dir(), get_eval_dir(),
    get_output_dir(), get_report_dir(), get_scratch_dir(), get_log_dir(),
    get_pipeline_dir(), get_monitor_dir(), get_test_dir()
]

def _load_logging_hooks():
    """載入混合日誌鉤子；DB schema 尚未建立時進入 bootstrap deferred。"""
    from contextlib import contextmanager

    @contextmanager
    def mock_lifecycle(task_name, **kwargs):
        yield None

    def mock_audit_log(*args, **kwargs):
        return None

    try:
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT
                    to_regclass('public.pipeline_execution_log') IS NOT NULL,
                    to_regclass('public.data_audit_log') IS NOT NULL
                """
            )
            has_pipeline_log, has_audit_log = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        if has_pipeline_log and has_audit_log:
            return record_lifecycle, write_data_audit_log, "REAL (DB-Linked)"
        return mock_lifecycle, mock_audit_log, "BOOTSTRAP-DEFERRED (DB schema pending)"
    except Exception as e:
        reason = str(e).strip() or type(e).__name__
        return mock_lifecycle, mock_audit_log, f"BOOTSTRAP-DEFERRED (DB hook unavailable: {reason})"


def _mark_lifecycle(lifecycle, level, message):
    if lifecycle is None:
        return
    method_name = "mark_failed" if level == "failed" else "mark_warning"
    marker = getattr(lifecycle, method_name, None)
    if callable(marker):
        marker(message)


def _evaluate_anchor():
    calculated = str(PROJECT_ROOT_CALC)
    if not PROJECT_ROOT_ENV:
        return "failed", "MISSING (PROJECT_ROOT not set in .env)", "PROJECT_ROOT missing in .env"
    if PROJECT_ROOT_ENV != calculated:
        return (
            "failed",
            f"MISMATCHED (env={PROJECT_ROOT_ENV})",
            f"PROJECT_ROOT mismatch: env={PROJECT_ROOT_ENV}, calc={calculated}",
        )
    return "success", "MATCHED", None


def ensure_all_dirs():
    """執行物理路徑對齊與自癒 (v4.44 / v5.4.22 啟動治理版)"""
    start_time = time.time()
    record_lifecycle, write_data_audit_log, log_mode = _load_logging_hooks()
    warnings = []

    for p in ALL_PATHS:
        p.mkdir(parents=True, exist_ok=True)

    anchor_level, anchor_display, anchor_message = _evaluate_anchor()
    if log_mode.startswith("MOCK"):
        warnings.append(f"Hybrid logging downgraded: {log_mode}")

    with record_lifecycle("path_setup_v4.44", category="maintenance", stock_id="SYSTEM") as lifecycle:
        if anchor_message:
            _mark_lifecycle(lifecycle, anchor_level, anchor_message)

        try:
            write_data_audit_log(
                "PATH_SOVEREIGNTY",
                "SYSTEM",
                datetime.now().strftime("%Y-%m-%d"),
                "SELF_HEAL_v4.44",
                len(ALL_PATHS),
            )
        except Exception as e:
            msg = f"data_audit_log downgraded: {type(e).__name__}: {e}"
            warnings.append(msg)
            _mark_lifecycle(lifecycle, "warning", msg)

        for msg in warnings:
            _mark_lifecycle(lifecycle, "warning", msg)

        if anchor_level != "success":
            sovereignty_status = "FAILED (Bootstrap Anchor Drift)"
        elif warnings:
            sovereignty_status = "WARNING (路徑已對齊；日誌降級)"
        else:
            sovereignty_status = "PERFECT (已對齊/自癒)"

        # ── 執行後路徑稽核摘要 (Flagship Report v4.44) ──
        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 路徑主權治理中心 (v4.44)")
        print("🛡️" * 40)

        print("\n" + "─" * 80)
        print("📊 物理路徑稽核摘要報告 (Path Sovereignty Report v4.44)")
        print("─" * 80)
        print(f"✅ 物理基準 (ROOT) : {PROJECT_ROOT_CALC}")
        print(f"⚓ 錨點對齊 (.env) : {anchor_display}")
        print("✅ 治理維度        : 25 維全譜路徑 (對齊 v5.4.22)")
        print(f"🕒 處理時長        : {(time.time() - start_time)*1000:.2f} ms")
        print(f"📝 混合日誌模式    : {log_mode}")
        if warnings:
            print(f"⚠️  降級警告        : {'; '.join(warnings[:3])}")
        print(f"⚖️  路徑主權狀態    : {sovereignty_status}")
        print("─" * 80)

        print("\n💡 路徑維運建議 (Reference Information):")
        print("1. [治權提示]: 系統已鎖定絕對路徑，嚴禁在子模組中使用 os.chdir()。")
        print("2. [解耦提示]: .env 僅錨定 PROJECT_ROOT；path_setup.py 承擔 25 維 Path SSOT 職責。")
        print("3. [範例提示]: 請參閱 Header 矩陣以確保維運環境的一致性。")
        print("─" * 80 + "\n")

    return anchor_level == "success"


if __name__ == "__main__":
    sys.exit(0 if ensure_all_dirs() else 1)
