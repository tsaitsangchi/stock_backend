"""
core/__init__.py v1.9 (Quantum Finance Edition)
================================================================================
核心庫總匯導出 — 旗艦編年史版 (Quantum v5.2 標準)
負責統一暴露 27 個路徑接口、資料庫自癒工具與全量模型管理功能。

【核心定義說明 (Core Definitions)】
1. [Centralized Interface]: 作為系統「門戶」，隱藏內部文件夾層次，提供統一的調用路徑。
2. [Encapsulation Policy]: 強制對外層腳本隱藏底層實作細節，僅暴露 get_xxx_dir 與核心類。
3. [Global Alignment]: 確保 Ingestion, Feature, Inference 各層引用的是同一套路徑邏輯。

【執行範例矩陣 (Historical & Active Matrix)】
┌──────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運場景                             │ 建議用法                                               │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [外層程式：獲取特徵儲存路徑]      │ from core import get_feature_store_dir; path = ...      │
│ 2. [系統級：一鍵建立全目錄結構]      │ from core import ensure_all_dirs; ensure_all_dirs()     │
│ 3. [舊版範例 (v1.0)：手動匯入組件]   │ from core.db_utils import ... (已整合至總匯)           │
│ 4. [標準範例 (v1.5)：22 接口調用]    │ from core import get_data_dir, get_model_dir...        │
│ 5. [旗艦範例 (v1.8)：27 接口全對齊]  │ from core import get_mlflow_dir, get_scalers_dir...    │
└──────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v1.9 (2026-05-12): [憲法] 注入詳細核心定義說明、舊歷程保留規範，對齊 2026-05-12 旗艦要求。
  v1.8 (2026-05-11): [旗艦] 補回 get_model_dir, get_feature_dir 等 27 個接口，確立 v5.2 標準。
  v1.7 (2026-05-11): [標準] 整合並匯出 v2.27 與 v4.1 路徑規範。
  v1.5 (2026-05-11): [主權] 補齊 22 個路徑接口導出。
  v1.0 (2026-04-20): [奠基] 初始版本，建立核心庫導出機制。
================================================================================
"""
from core.path_setup import (
    get_root_dir, get_core_dir, get_utils_dir, get_maintenance_dir,
    get_data_dir, get_raw_data_dir, get_ingestion_dir,
    get_feature_dir, get_feature_store_dir,
    get_model_dir, get_model_weights_dir, get_model_scalers_dir,
    get_training_dir, get_archive_dir, get_mlflow_dir,
    get_infer_dir, get_prediction_dir, get_eval_dir, get_evaluation_dir,
    get_output_dir, get_report_dir, get_scratch_dir, get_log_dir,
    get_pipeline_dir, get_monitor_dir, get_test_dir, ensure_all_dirs
)

from core.db_utils import (
    get_db_connection, db_connection, get_core_stocks_from_db,
    record_lifecycle, write_data_audit_log
)

from core.finmind_client import FinMindClient
from core.data_schema import DATASET_SCHEMA_MAP
