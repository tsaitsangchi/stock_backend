"""
path_setup.py v3.11 (Quantum Finance Edition)
================================================================================
資源路徑總司令部 — 絕對路徑強制版 (Quantum v5.2 標準)
負責接管全系統 22 個關鍵路徑，確保路徑解析不隨執行目錄 (CWD) 變動。

修訂歷程：
  v3.11 (2026-05-11): [修復] 強制 relative path 相對於 PROJECT_ROOT，消除 scripts/scripts 重疊。
  v3.10 (2026-05-11): [主權化] 補齊 22 個接口。
================================================================================
"""
import os, sys, logging, shutil, platform
from pathlib import Path
from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
def find_project_root(current_path: Path) -> Path:
    for parent in [current_path] + list(current_path.parents):
        if (parent / ".env").exists(): return parent
    return current_path.parent
_PROJECT_ROOT = find_project_root(_THIS_DIR)
load_dotenv(_PROJECT_ROOT / ".env")

def get_resource_path(env_var: str, default_subdir: str) -> (Path, bool):
    path_str = os.getenv(env_var)
    # 核心修復：如果 path_str 是相對路徑，則強制相對於 _PROJECT_ROOT
    if path_str:
        path = Path(path_str)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        else:
            path = path.resolve()
    else:
        path = _PROJECT_ROOT / default_subdir
    
    healed = False
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True); healed = True
    return path, healed

# ── 22 核心資源路徑定義 (Getters) ──
def get_data_dir() -> Path: return get_resource_path("DATA_DIR", "data")[0]
def get_feature_dir() -> Path: return get_resource_path("FEATURE_DIR", "features")[0]
def get_model_dir() -> Path: return get_resource_path("MODEL_DIR", "models")[0]
def get_training_dir() -> Path: return get_resource_path("TRAINING_DIR", "scripts/training")[0]
def get_archive_dir() -> Path: return get_resource_path("ARCHIVE_DIR", "archive")[0]
def get_mlflow_dir() -> Path: return get_resource_path("MLFLOW_DIR", "scripts/mlruns")[0]
def get_infer_dir() -> Path: return get_resource_path("INFER_DIR", "inference")[0]
def get_prediction_dir() -> Path: return get_resource_path("PREDICTION_DIR", "predictions")[0]
def get_eval_dir() -> Path: return get_resource_path("EVAL_DIR", "evaluations")[0]
def get_data_ingestion_dir() -> Path: return get_resource_path("INGESTION_DIR", "scripts/ingestion")[0]
def get_log_dir() -> Path: return get_resource_path("LOG_DIR", "logs")[0]
def get_output_dir() -> Path: return get_resource_path("OUTPUT_DIR", "scripts/outputs")[0]
def get_report_dir() -> Path: return get_resource_path("REPORT_DIR", "scripts/reports")[0]
def get_scratch_dir() -> Path: return get_resource_path("SCRATCH_DIR", "scripts/scratch")[0]
def get_pipeline_dir() -> Path: return get_resource_path("PIPELINE_DIR", "scripts/pipeline")[0]
def get_monitor_dir() -> Path: return get_resource_path("MONITOR_DIR", "scripts/monitor")[0]
def get_test_dir() -> Path: return get_resource_path("TEST_DIR", "scripts/tests")[0]
def get_core_dir() -> Path: return get_resource_path("CORE_DIR", "scripts/core")[0]
def get_utils_dir() -> Path: return get_resource_path("UTILS_DIR", "scripts/utils")[0]
def get_maintenance_dir() -> Path: return get_resource_path("MAINTENANCE_DIR", "scripts/maintenance")[0]

def update_latest_link(src_file: str, link_name: str):
    target = get_archive_dir() / link_name
    if target.is_symlink() or target.exists(): target.unlink()
    os.symlink(src_file, target)

def show_grand_dashboard(healed_count: int):
    print("\n" + "👑"*40)
    print(f"🚀 Quantum Finance: 全域路徑治理總表 (v3.11)")
    print(f"✅ 執行結果  : SUCCESS | 📍 Root: {_PROJECT_ROOT}")
    
    sections = {
        "📊 數據與特徵": [("Data", get_data_dir()), ("Features", get_feature_dir()), ("Ingestion", get_data_ingestion_dir())],
        "🤖 學習與實驗": [("Training", get_training_dir()), ("Models", get_model_dir()), ("Archive", get_archive_dir()), ("MLflow", get_mlflow_dir())],
        "🔮 推論與預測": [("Inference", get_infer_dir()), ("Prediction", get_prediction_dir()), ("Eval", get_eval_dir())],
        "📈 產出與維運": [("Outputs", get_output_dir()), ("Reports", get_report_dir()), ("Logs", get_log_dir()), ("Pipeline", get_pipeline_dir())]
    }
    
    for section, paths in sections.items():
        print(f"\n{section}"); print("-" * 80)
        for label, path in paths:
            _, _, free = shutil.disk_usage(path)
            print(f"  {label:<15} : {path} | [剩餘: {free // (2**30)} GB]")
    print("-" * 80 + "\n")

if __name__ == "__main__":
    show_grand_dashboard(0)