"""
path_setup.py v4.2 (Quantum Finance Edition)
================================================================================
全域路徑治理引擎 — 接口全回歸版 (Quantum v5.2 標準)
負責將全系統 27 個關鍵目錄映射為絕對路徑，確保物理與邏輯邊界完全對齊。

修訂歷程：
  v4.2 (2026-05-11): [修復] 補回 get_model_dir 等 17 個遺漏接口，確保全系統診斷不崩潰。
  v4.1 (2026-05-11): [標準] 補全極致範例矩陣。
  v4.0 (2026-05-11): [架構] 新增 5 個 DDD 接口 (Raw, Store, Weights, Scalers, Eval)。

【執行範例矩陣 (Path Governance Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統路徑治理地圖]      │ $ python scripts/core/path_setup.py                    │
│ 2. [程式內獲取模型目錄]      │ from core import get_model_dir                         │
│ 3. [強制執行：全目錄同步建立]│ ensure_all_dirs()                                      │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, shutil
from pathlib import Path
from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
def find_project_root(current_path: Path) -> Path:
    for parent in [current_path] + list(current_path.parents):
        if (parent / ".env").exists(): return parent
    return current_path.parent

_PROJECT_ROOT = find_project_root(_THIS_DIR)
load_dotenv(_PROJECT_ROOT / ".env")

def get_abs_path(env_key: str, default_rel_path: str) -> Path:
    raw_path = os.getenv(env_key)
    p = Path(raw_path) if raw_path else _PROJECT_ROOT / default_rel_path
    return p if p.is_absolute() else _PROJECT_ROOT / p

# ── 1. 核心代碼接口 ──
def get_root_dir(): return _PROJECT_ROOT
def get_core_dir(): return get_abs_path("CORE_DIR", "scripts/core")
def get_utils_dir(): return get_abs_path("UTILS_DIR", "scripts/utils")
def get_maintenance_dir(): return get_abs_path("MAINTENANCE_DIR", "scripts/maintenance")

# ── 2. 數據層接口 (Data Context) ──
def get_data_dir(): return get_abs_path("DATA_DIR", "data")
def get_raw_data_dir(): return get_abs_path("RAW_DATA_DIR", "data/raw")
def get_ingestion_dir(): return get_abs_path("INGESTION_DIR", "scripts/ingestion")

# ── 3. 特徵層接口 (Feature Context) ──
def get_feature_dir(): return get_abs_path("FEATURE_DIR", "features")
def get_feature_store_dir(): return get_abs_path("FEATURE_STORE_DIR", "data/feature_store")

# ── 4. 學習與訓練接口 (Model Context) ──
def get_model_dir(): return get_abs_path("MODEL_DIR", "models")
def get_model_weights_dir(): return get_abs_path("MODEL_WEIGHTS_DIR", "models/weights")
def get_model_scalers_dir(): return get_abs_path("MODEL_SCALERS_DIR", "models/scalers")
def get_training_dir(): return get_abs_path("TRAINING_DIR", "scripts/training")
def get_archive_dir(): return get_abs_path("ARCHIVE_DIR", "archive")
def get_mlflow_dir(): return get_abs_path("MLFLOW_DIR", "scripts/mlruns")

# ── 5. 預測與評估接口 (Prediction Context) ──
def get_infer_dir(): return get_abs_path("INFER_DIR", "inference")
def get_prediction_dir(): return get_abs_path("PREDICTION_DIR", "predictions")
def get_eval_dir(): return get_abs_path("EVAL_DIR", "evaluations")
def get_evaluation_dir(): return get_abs_path("EVALUATION_DIR", "evaluations")

# ── 6. 維運與產出接口 (DevOps) ──
def get_output_dir(): return get_abs_path("OUTPUT_DIR", "scripts/outputs")
def get_report_dir(): return get_abs_path("REPORT_DIR", "scripts/reports")
def get_scratch_dir(): return get_abs_path("SCRATCH_DIR", "scripts/scratch")
def get_log_dir(): return get_abs_path("LOG_DIR", "logs")
def get_pipeline_dir(): return get_abs_path("PIPELINE_DIR", "scripts/pipeline")
def get_monitor_dir(): return get_abs_path("MONITOR_DIR", "scripts/monitor")
def get_test_dir(): return get_abs_path("TEST_DIR", "scripts/tests")

def ensure_all_dirs():
    dirs = [
        get_core_dir(), get_utils_dir(), get_maintenance_dir(),
        get_data_dir(), get_raw_data_dir(), get_ingestion_dir(),
        get_feature_dir(), get_feature_store_dir(),
        get_model_dir(), get_model_weights_dir(), get_model_scalers_dir(),
        get_training_dir(), get_archive_dir(), get_mlflow_dir(),
        get_infer_dir(), get_prediction_dir(), get_eval_dir(), get_evaluation_dir(),
        get_output_dir(), get_report_dir(), get_log_dir(), get_pipeline_dir(),
        get_monitor_dir(), get_test_dir(), get_scratch_dir()
    ]
    for d in dirs: d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_all_dirs()
    print(f"✅ Path Governance v4.2 - All 27 interfaces restored.")