"""
core/__init__.py v1.7 (Quantum Finance Edition)
================================================================================
核心庫總匯導出 — 極致標準版 (Quantum v5.2 標準)
負責統一暴露 27 個路徑接口與資料庫自癒工具。

修訂歷程：
  v1.7 (2026-05-11): [標準] 整合並匯出 v2.27 與 v4.1。
  v1.6 (2026-05-11): [標準] 對齊 path_setup v4.0。
================================================================================
"""
from core.path_setup import (
    get_root_dir, get_core_dir, get_data_dir, get_raw_data_dir,
    get_feature_store_dir, get_model_weights_dir, get_model_scalers_dir,
    get_evaluation_dir, get_ingestion_dir, get_log_dir, ensure_all_dirs
)

from core.db_utils import (
    db_transaction, db_connection, record_lifecycle, ensure_infrastructure,
    get_db_stock_ids, bulk_upsert, get_latest_date,
    write_pipeline_log, write_data_audit_log
)

from core.finmind_client import FinMindClient

from core.model_metadata import (
    audit_registry, register_model, ensure_model_table_integrity
)
