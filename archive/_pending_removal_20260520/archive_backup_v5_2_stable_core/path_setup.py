"""
path_setup.py v4.1 (Quantum Finance Edition)
================================================================================
全域路徑治理引擎 — 極致範例版 (Quantum v5.2 標準)
負責將全系統 27 個關鍵目錄映射為絕對路徑，確保物理與邏輯邊界對齊。

修訂歷程：
  v4.1 (2026-05-11): [標準] 補全極致範例矩陣，包含路徑地圖生成與單一接口查詢範例。
  v4.0 (2026-05-11): [架構] 新增 5 個 DDD 接口 (Raw, Store, Weights, Scalers, Eval)。
  v3.11 (2026-05-11): [修復] 強制絕對路徑解析。

【執行範例矩陣 (Path Governance Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統路徑治理地圖]      │ $ python scripts/core/path_setup.py                    │
│ 2. [單一接口：獲取權重目錄]  │ path = get_model_weights_dir()                         │
│ 3. [單一接口：獲取特徵目錄]  │ path = get_feature_store_dir()                         │
│ 4. [強制執行：全目錄同步建立]│ ensure_all_dirs()                                      │
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

def get_root_dir(): return _PROJECT_ROOT
def get_core_dir(): return get_abs_path("CORE_DIR", "scripts/core")
def get_data_dir(): return get_abs_path("DATA_DIR", "data")
def get_raw_data_dir(): return get_abs_path("RAW_DATA_DIR", "data/raw")
def get_feature_store_dir(): return get_abs_path("FEATURE_STORE_DIR", "data/feature_store")
def get_model_weights_dir(): return get_abs_path("MODEL_WEIGHTS_DIR", "models/weights")
def get_model_scalers_dir(): return get_abs_path("MODEL_SCALERS_DIR", "models/scalers")
def get_evaluation_dir(): return get_abs_path("EVALUATION_DIR", "evaluations")
def get_ingestion_dir(): return get_abs_path("INGESTION_DIR", "scripts/ingestion")
def get_log_dir(): return get_abs_path("LOG_DIR", "logs")

def ensure_all_dirs():
    dirs = [get_core_dir(), get_data_dir(), get_raw_data_dir(), get_feature_store_dir(), get_model_weights_dir(), get_model_scalers_dir(), get_evaluation_dir(), get_log_dir()]
    for d in dirs: d.mkdir(parents=True, exist_ok=True)

def show_path_dashboard():
    ensure_all_dirs()
    print("\n" + "👑"*40)
    print(f"🚀 Quantum Finance: 路徑治理地圖 (v4.1)\n✅ 根目錄: {_PROJECT_ROOT}")
    print(f"📊 核心接口: RawData={get_raw_data_dir().name} | Weights={get_model_weights_dir().name}")
    print("👑"*40 + "\n")

if __name__ == "__main__":
    show_path_dashboard()