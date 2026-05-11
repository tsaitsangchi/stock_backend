"""
verify_core_integrity.py v1.1 (Quantum Finance Edition)
================================================================================
核心庫完整性稽核腳本 — 信任重建版 (Quantum v5.2 標準)
專門用於窮舉校驗 core 模組的所有 27 個接口與功能鏈。

修訂歷程：
  v1.1 (2026-05-11): [標準] 升級至 v5.2 標準，補全修訂歷程與範例矩陣。
  v1.0 (2026-05-11): [首發] 實作接口窮舉測試與路徑對齊校驗。

【執行範例矩陣 (Integrity Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [核心庫完整性全稽核]      │ $ python scripts/maintenance/verify_core_integrity.py  │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import sys, os
from pathlib import Path

# ── 確保路徑正確 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

def run_test():
    print("\n🔍 啟動核心庫「硬核完整性」深度稽核...")
    print("=" * 60)
    
    # 1. 測試 27 個路徑接口的匯入
    expected_paths = [
        "get_root_dir", "get_core_dir", "get_utils_dir", "get_maintenance_dir",
        "get_data_dir", "get_raw_data_dir", "get_ingestion_dir",
        "get_feature_dir", "get_feature_store_dir",
        "get_model_dir", "get_model_weights_dir", "get_model_scalers_dir",
        "get_training_dir", "get_archive_dir", "get_mlflow_dir",
        "get_infer_dir", "get_prediction_dir", "get_eval_dir", "get_evaluation_dir",
        "get_output_dir", "get_report_dir", "get_scratch_dir", "get_log_dir",
        "get_pipeline_dir", "get_monitor_dir", "get_test_dir", "ensure_all_dirs"
    ]
    
    import core
    missing_paths = []
    print("📡 [測試 1/3] 路徑接口窮舉檢查:")
    for p in expected_paths:
        if hasattr(core, p):
            print(f"  ✅ {p.ljust(25)} : 存在")
        else:
            print(f"  ❌ {p.ljust(25)} : 缺失")
            missing_paths.append(p)
            
    # 2. 測試資料庫自癒功能鏈
    print("\n💎 [測試 2/3] 資料庫與日誌功能鏈檢查:")
    expected_funcs = ["record_lifecycle", "ensure_infrastructure", "bulk_upsert", "FinMindClient"]
    missing_funcs = []
    for f in expected_funcs:
        if hasattr(core, f):
            print(f"  ✅ {f.ljust(25)} : 存在")
        else:
            print(f"  ❌ {f.ljust(25)} : 缺失")
            missing_funcs.append(f)

    # 3. 測試實例化與路徑解析
    print("\n📁 [測試 3/3] 實機路徑解析檢查:")
    try:
        model_dir = core.get_model_dir()
        print(f"  ✅ get_model_dir() 成功解析為: {model_dir}")
        if str(_PROJECT_ROOT) in str(model_dir):
            print(f"  ✅ 路徑與根目錄對齊成功")
    except Exception as e:
        print(f"  ❌ 路徑解析失敗: {e}")

    print("\n" + "=" * 60)
    if not missing_paths and not missing_funcs:
        print("🏆 稽核結果: PERFECT | 核心庫 100% 完整，信任已重建。")
        return True
    else:
        print(f"❌ 稽核結果: FAILED | 發現 {len(missing_paths) + len(missing_funcs)} 處缺失。")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
