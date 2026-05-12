"""
core/__init__.py v1.11 (Quantum Finance Edition)
================================================================================
系統治權中樞 — 憲法完整版 (Quantum v5.2 標準)
負責 27 維全譜路徑接口導出、路徑自愈引導與系統導入鏈之絕對主權維護。

【核心定義說明 (Core Definitions)】
1. [Central Hub Sovereignty]: 確立 core 為系統的唯一通訊中樞，所有跨模組調用必須透過此入口。
2. [Interface Integrity]: 鎖定 27 維全譜路徑接口，確保導入鏈具備 100% 的邊界完整性。
3. [Historical Reference Authority]: 保留從 v1.0 到 v1.11 的所有歷史歷程，作為判定系統正確性的基準。
4. [Hybrid Observability]: 導入鏈的任何崩潰或 MOCK 降級必須能透過供應鏈診斷即時察覺。

【全量執行範例矩陣 (The Complete Operational Matrix)】
┌──────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運需求場景                             │ 建議指令 / 用法                                        │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [中樞治理：全系統導入鏈完整性稽核]    │ $ python scripts/core/finmind_client.py                │
│ 2. [個股同步 / 路徑對齊初始化]           │ $ python scripts/core/path_setup.py                    │
│ 3. [所有核心股 / 數據契約重鑄]           │ $ python scripts/core/data_schema.py --init --force     │
│ 4. [所有核心股 / 所有表：全量數據同步]   │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --universe core --all_datasets                       │
│ 5. [所有核心股 / 所有表：全量強制更新]   │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --universe core --all_datasets --force               │
└──────────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v1.11 (2026-05-12): [憲法] 補全全量維運矩陣與四維核心定義，對齊 v5.2 旗艦要求。
  v1.10 (2026-05-12): [修復] 移除不存在的 db_connection 引用，終結導入鏈崩潰，恢復 REAL 模式。
  v1.8  (2026-05-11): [標準] 對齊 27 維全譜接口導出 (新增 RAW_DATA, MLFLOW 等)。
  v1.0  (2026-04-20): [奠基] 初始接口導出定義。
================================================================================
"""
# 封裝 path_setup 以確保導入鏈具備自愈能力
from core.path_setup import (
    PROJECT_ROOT, get_root_dir, get_core_dir, get_utils_dir, get_maintenance_dir,
    get_data_dir, get_raw_data_dir, get_ingestion_dir, get_feature_dir, get_feature_store_dir,
    get_model_dir, get_model_weights_dir, get_model_scalers_dir, get_training_dir,
    get_archive_dir, get_mlflow_dir, get_infer_dir, get_prediction_dir, get_eval_dir,
    get_evaluation_dir, get_output_dir, get_report_dir, get_scratch_dir, get_log_dir,
    get_pipeline_dir, get_monitor_dir, get_test_dir
)

# 導出核心工具
from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
from core.finmind_client import FinMindClient
from core.data_schema import DATASET_SCHEMA_MAP

__all__ = [
    'PROJECT_ROOT', 'get_root_dir', 'get_core_dir', 'get_utils_dir', 'get_maintenance_dir',
    'get_data_dir', 'get_raw_data_dir', 'get_ingestion_dir', 'get_feature_dir', 'get_feature_store_dir',
    'get_model_dir', 'get_model_weights_dir', 'get_model_scalers_dir', 'get_training_dir',
    'get_archive_dir', 'get_mlflow_dir', 'get_infer_dir', 'get_prediction_dir', 'get_eval_dir',
    'get_evaluation_dir', 'get_output_dir', 'get_report_dir', 'get_scratch_dir', 'get_log_dir',
    'get_pipeline_dir', 'get_monitor_dir', 'get_test_dir',
    'get_db_connection', 'record_lifecycle', 'write_data_audit_log',
    'FinMindClient', 'DATASET_SCHEMA_MAP'
]
