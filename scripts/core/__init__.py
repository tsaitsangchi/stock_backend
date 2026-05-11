"""
core/__init__.py v1.6 (Quantum Finance Edition)
================================================================================
核心庫導出定義 — DDD 完整對齊版 (Quantum v5.2 標準)
負責統一暴露 27 個路徑接口、資料庫工具與模型註冊功能。

修訂歷程：
  v1.6 (2026-05-11): [標準] 對齊 path_setup v4.0 的 27 個接口命名。
  v1.5 (2026-05-11): [治理] 暴露 22 個路徑變數與模型註冊功能。
================================================================================
"""

# ── 1. 路徑治理接口 (對齊 path_setup v4.0) ──
from core.path_setup import (
    get_root_dir,
    get_core_dir,
    get_utils_dir,
    get_maintenance_dir,
    
    # 數據層
    get_data_dir,
    get_raw_data_dir,
    get_ingestion_dir,
    
    # 特徵層
    get_feature_dir,
    get_feature_store_dir,
    
    # 學習層
    get_model_dir,
    get_model_weights_dir,
    get_model_scalers_dir,
    get_training_dir,
    get_archive_dir,
    get_mlflow_dir,
    
    # 預測層
    get_infer_dir,
    get_prediction_dir,
    get_eval_dir,
    get_evaluation_dir,
    
    # 維運層
    get_output_dir,
    get_report_dir,
    get_scratch_dir,
    get_log_dir,
    get_pipeline_dir,
    get_monitor_dir,
    get_test_dir
)

# ── 2. 資料庫核心工具 (對齊 db_utils v2.26) ──
from core.db_utils import (
    db_transaction,
    db_connection,
    record_lifecycle,
    ensure_infrastructure,
    get_db_stock_ids,
    bulk_upsert,
    get_latest_date,
    write_pipeline_log,
    write_data_audit_log
)

# ── 3. API 供應鏈工具 (對齊 finmind_client v4.15) ──
from core.finmind_client import FinMindClient

# ── 4. 模型註冊工具 (對齊 model_metadata v2.11) ──
from core.model_metadata import (
    audit_registry,
    register_model
)
