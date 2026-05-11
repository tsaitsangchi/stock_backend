"""
core/__init__.py v1.8 (Quantum Finance Edition)
================================================================================
核心庫總匯導出 — 全面接口版 (Quantum v5.2 標準)
負責統一暴露 27 個路徑接口、資料庫自癒工具與模型管理功能。

修訂歷程：
  v1.8 (2026-05-11): [修復] 補回 get_model_dir, get_feature_dir 等所有 27 個接口的導出。
  v1.7 (2026-05-11): [標準] 整合並匯出 v2.27 與 v4.1。
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
    db_transaction, db_connection, record_lifecycle, ensure_infrastructure,
    get_db_stock_ids, bulk_upsert, get_latest_date,
    write_pipeline_log, write_data_audit_log
)

from core.finmind_client import FinMindClient

from core.model_metadata import ModelMetadata

