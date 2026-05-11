"""
core/__init__.py v1.5 (Quantum Finance Edition)
================================================================================
核心組件統一調度中心 — 全頻譜對齊版 (Quantum v5.2 標準)
負責封裝並暴露全系統 22 個資源路徑與元數據管理函式。

修訂歷程：
  v1.5 (2026-05-11): [主權化] 補齊 get_training_dir, get_prediction_dir 等全頻譜路徑接口。
================================================================================
"""
# 1. 資源路徑 (Path Governance - 22 接口)
from core.path_setup import (
    get_data_dir, get_feature_dir, get_model_dir, get_training_dir,
    get_archive_dir, get_mlflow_dir, get_infer_dir, get_prediction_dir,
    get_eval_dir, get_data_ingestion_dir, get_log_dir, get_output_dir,
    get_report_dir, get_scratch_dir, get_pipeline_dir, get_monitor_dir,
    get_test_dir, get_core_dir, get_utils_dir, get_maintenance_dir,
    update_latest_link
)

# 2. 資料庫與生命週期
from core.db_utils import (
    db_transaction, db_connection, ensure_infrastructure,
    record_lifecycle, bulk_upsert, get_latest_date,
    write_data_audit_log, get_db_stock_ids
)

# 3. API 供應鏈
from core.finmind_client import FinMindClient

# 4. 模型註冊中心
from core.model_metadata import audit_registry, register_model
