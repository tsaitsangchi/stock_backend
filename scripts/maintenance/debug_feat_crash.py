"""
debug_feat_crash.py v2.1 (Quantum Finance Edition)
================================================================================
特徵崩潰調試器 — 系統穩定性診斷工具 (Quantum v5.2 標準)
負責主動掃描流水線日誌，定位特徵生成或訓練過程中的崩潰原因，並輸出診斷報告。

修訂歷程：
  v2.1 (2026-05-11): [標準化] 實作主動日誌掃描邏輯、補全範例矩陣與 Hybrid Logging 對接。
  v5.5.7 (2026-05-09): [核心] 導入 Hybrid Logging 混合日誌。

【執行範例矩陣 (Debug Operations Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / SQL                                         │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [執行全系統崩潰診斷]      │ $ python scripts/maintenance/debug_feat_crash.py       │
│ 2. [查看最新診斷發現]        │ SELECT message, created_at FROM data_audit_log         │
│                              │ WHERE table_name = 'crash_diagnosis' ORDER BY id DESC; │
│ 3. [定位特定任務錯誤]        │ SELECT error_msg FROM pipeline_execution_log           │
│                              │ WHERE status = 'failed' ORDER BY created_at DESC;      │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【業務邏輯說明】
  - 診斷來源: pipeline_execution_log 中狀態為 'failed' 的紀錄。
  - 核心價值: 快速提取 traceback 中的關鍵錯誤關鍵字 (如 MemoryError, Timeout)。
================================================================================
"""
import sys, logging, time, argparse
from pathlib import Path
from datetime import datetime

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent if _THIS_DIR.name != "scripts" else _THIS_DIR
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_transaction, record_lifecycle, write_data_audit_log
except ImportError as e:
    print(f"[FATAL] 無法匯入核心配置: {e}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def show_debug_dashboard(stats: dict):
    """執行後的診斷儀表板。"""
    print("\n" + "🚑"*35)
    print("🚀 Quantum Finance: 特徵崩潰診斷報告 (v2.1)")
    print("🚑"*35)
    print(f"✅ 診斷狀態  : 完成")
    print(f"🔍 掃描任務數: {stats['scanned_tasks']} 筆")
    print(f"⚠️ 偵測異常數: {stats['anomaly_found']} 筆")
    
    if stats['anomaly_found'] > 0:
        print("-" * 70)
        print(f"🔴 關鍵診斷：{stats['last_error'][:60]}...")
        print(f"🛠️  修復建議：請檢查資料庫連線池、記憶體佔用或 API 響應狀態。")
    else:
        print("-" * 70)
        print("🟢 診斷良好：最近 24 小時內未偵測到顯著的系統崩潰紀錄。")
        
    print("-" * 70)
    print("📝 日誌同步: pipeline_execution_log & data_audit_log (crash_diagnosis)")
    print("🚑"*35 + "\n")

def run_debug():
    """啟動崩潰診斷流程。"""
    with record_lifecycle("feature_crash_debug", category="sys", stock_id="SYSTEM"):
        stats = {"scanned_tasks": 0, "anomaly_found": 0, "last_error": "None"}
        try:
            logger.info("🚑 啟動特徵生成崩潰原因調試...")
            
            with db_transaction() as cur:
                # 掃描最近 24 小時的失敗任務
                query = "SELECT task_name, error_msg FROM pipeline_execution_log WHERE status = 'failed' AND created_at > now() - interval '24 hours';"
                cur.execute(query)
                failures = cur.fetchall()
                
                stats['scanned_tasks'] = len(failures)
                if failures:
                    stats['anomaly_found'] = len(failures)
                    raw_msg = failures[0]['error_msg']
                    stats['last_error'] = raw_msg if raw_msg else "(Legacy log without error details)"
                    
                    # 將關鍵診斷紀錄到審計表
                    write_data_audit_log("crash_diagnosis", "SYSTEM", 
                                         datetime.now().strftime("%Y-%m-%d"), 
                                         "DIAGNOSE", stats['anomaly_found'])

            show_debug_dashboard(stats)
            return True
        except Exception as e:
            logger.error(f"❌ 調試任務失敗: {e}")
            raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_debug()
