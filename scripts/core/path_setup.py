"""
path_setup.py v4.0 (Quantum Finance Edition)
================================================================================
全域路徑治理引擎 — DDD 邊界對齊版 (Quantum v5.2 標準)
負責將全系統 27 個關鍵目錄映射為絕對路徑，確保數據管線與模型實驗的物理秩序。

修訂歷程：
  v4.0 (2026-05-11): [架構] 新增 5 個 DDD 接口 (Raw Data, Feature Store, Weights, Scalers, Evaluation)。
  v3.11 (2026-05-11): [修復] 強制絕對路徑解析，消除目錄層級重疊 Bug。
  v3.10 (2026-05-11): [治理] 實作全域 .env 感知，確立 PROJECT_ROOT 為定錨點。

【執行範例矩陣 (Path Governance Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [查看全路徑治理地圖]      │ $ python scripts/core/path_setup.py                    │
│ 2. [程式內獲取特徵目錄]      │ from core.path_setup import get_feature_dir            │
│ 3. [程式內獲取模型權重]      │ from core.path_setup import get_model_weights_dir      │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, shutil
from pathlib import Path
from dotenv import load_dotenv

# ── 核心定錨邏輯 ──
_THIS_DIR = Path(__file__).resolve().parent

def find_project_root(current_path: Path) -> Path:
    """自動向上尋找包含 .env 的專案根目錄"""
    for parent in [current_path] + list(current_path.parents):
        if (parent / ".env").exists(): return parent
    return current_path.parent

_PROJECT_ROOT = find_project_root(_THIS_DIR)
load_dotenv(_PROJECT_ROOT / ".env")

def get_abs_path(env_key: str, default_rel_path: str) -> Path:
    """從 .env 讀取路徑，若無則以根目錄為基準建立絕對路徑"""
    raw_path = os.getenv(env_key)
    if raw_path:
        # 確保 .env 內的內容如果是相對路徑，也能轉為相對於根目錄的絕對路徑
        p = Path(raw_path)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return _PROJECT_ROOT / default_rel_path

# ── 1. 核心代碼接口 ──
def get_root_dir() -> Path: return _PROJECT_ROOT
def get_core_dir() -> Path: return get_abs_path("CORE_DIR", "scripts/core")
def get_utils_dir() -> Path: return get_abs_path("UTILS_DIR", "scripts/utils")
def get_maintenance_dir() -> Path: return get_abs_path("MAINTENANCE_DIR", "scripts/maintenance")

# ── 2. 數據攝取與儲存接口 (Data Context) ──
def get_data_dir() -> Path: return get_abs_path("DATA_DIR", "data")
def get_raw_data_dir() -> Path: return get_abs_path("RAW_DATA_DIR", "data/raw")
def get_ingestion_dir() -> Path: return get_abs_path("INGESTION_DIR", "scripts/ingestion")

# ── 3. 特徵工程接口 (Feature Context) ──
def get_feature_dir() -> Path: return get_abs_path("FEATURE_DIR", "features")
def get_feature_store_dir() -> Path: return get_abs_path("FEATURE_STORE_DIR", "data/feature_store")

# ── 4. 模型與訓練接口 (Model Context) ──
def get_model_dir() -> Path: return get_abs_path("MODEL_DIR", "models")
def get_model_weights_dir() -> Path: return get_abs_path("MODEL_WEIGHTS_DIR", "models/weights")
def get_model_scalers_dir() -> Path: return get_abs_path("MODEL_SCALERS_DIR", "models/scalers")
def get_training_dir() -> Path: return get_abs_path("TRAINING_DIR", "scripts/training")
def get_archive_dir() -> Path: return get_abs_path("ARCHIVE_DIR", "archive")
def get_mlflow_dir() -> Path: return get_abs_path("MLFLOW_DIR", "scripts/mlruns")

# ── 5. 預測與評估接口 (Prediction Context) ──
def get_infer_dir() -> Path: return get_abs_path("INFER_DIR", "inference")
def get_prediction_dir() -> Path: return get_abs_path("PREDICTION_DIR", "predictions")
def get_eval_dir() -> Path: return get_abs_path("EVAL_DIR", "evaluations")
def get_evaluation_dir() -> Path: return get_abs_path("EVALUATION_DIR", "evaluations")

# ── 6. 產出與維運接口 (DevOps) ──
def get_output_dir() -> Path: return get_abs_path("OUTPUT_DIR", "scripts/outputs")
def get_report_dir() -> Path: return get_abs_path("REPORT_DIR", "scripts/reports")
def get_scratch_dir() -> Path: return get_abs_path("SCRATCH_DIR", "scripts/scratch")
def get_log_dir() -> Path: return get_abs_path("LOG_DIR", "logs")
def get_pipeline_dir() -> Path: return get_abs_path("PIPELINE_DIR", "scripts/pipeline")
def get_monitor_dir() -> Path: return get_abs_path("MONITOR_DIR", "scripts/monitor")
def get_test_dir() -> Path: return get_abs_path("TEST_DIR", "scripts/tests")

def ensure_all_dirs():
    """自動確保所有定義的目錄均已物理存在"""
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

def show_path_dashboard():
    ensure_all_dirs()
    print("\n" + "👑"*45)
    print(f"🚀 Quantum Finance: 全域路徑治理總表 (v4.0 - DDD Edition)")
    print(f"✅ 執行結果  : SUCCESS | 📍 Root: {_PROJECT_ROOT}")
    
    sections = {
        "📊 數據與特徵 (Data Context)": [
            ("Raw Data", get_raw_data_dir()),
            ("Features", get_feature_dir()),
            ("Feat Store", get_feature_store_dir()),
            ("Ingestion", get_ingestion_dir()),
        ],
        "🤖 學習與資產 (Model Context)": [
            ("Training", get_training_dir()),
            ("Models", get_model_dir()),
            ("Weights", get_model_weights_dir()),
            ("Scalers", get_model_scalers_dir()),
            ("Archive", get_archive_dir()),
        ],
        "🔮 推論與評估 (Prediction Context)": [
            ("Inference", get_infer_dir()),
            ("Prediction", get_prediction_dir()),
            ("Evaluation", get_evaluation_dir()),
        ],
        "📈 維運與日誌 (DevOps)": [
            ("Logs", get_log_dir()),
            ("Pipeline", get_pipeline_dir()),
            ("Reports", get_report_dir()),
            ("Outputs", get_output_dir()),
        ]
    }

    for section, items in sections.items():
        print(f"\n{section}")
        print("-" * 90)
        for label, path in items:
            _, _, free = shutil.disk_usage(path.parent if not path.exists() else path)
            print(f"  {label:<15} : {path} | [可用: {free // (2**30)} GB]")

    print("-" * 90 + "\n")

if __name__ == "__main__":
    show_path_dashboard()