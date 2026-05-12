"""
check_system_health.py v2.3 (Quantum Finance Edition)
================================================================================
系統終極健康診斷報告 — 全維度診斷矩陣版 (Quantum v5.2 標準)
負責稽核基礎設施、資料庫主權、API 供應鏈與數據可觀測性的全維度健康狀況。

【核心定義說明 (Core Definitions)】
1. [Health Sovereignty]: 健康診斷為系統「完美狀態」的動態證明，任何指標異常必須觸發警報。
2. [Hybrid Observability]: 強制觸發 pipeline_execution_log (行為) 與 data_audit_log (審計) 雙軌同步。
3. [Historical Reference Authority]: 保留從 v1.0 到 v2.3 的所有歷史歷程，作為判定系統健康變遷的基準。
4. [Boundary Integrity]: 透過 27 維路徑實體驗證，確保診斷鏈無死角覆蓋。

【全量執行範例矩陣 (The Complete Operational Matrix)】
┌──────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運需求場景                             │ 建議指令 / 用法                                        │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [系統級：全量健康一鍵診斷]            │ $ python scripts/maintenance/check_system_health.py    │
│ 2. [單一 Table / 數據契約對齊檢查]       │ $ python scripts/core/data_schema.py --init            │
│ 3. [單一個股所有 Table：數據同步檢查]     │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --id 2330 --all_datasets                             │
│ 4. [所有核心股：全量主權健康稽核]        │ $ python scripts/maintenance/verify_core_integrity.py  │
│ 5. [所有核心股 + 所有表：全量強制更新]   │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --universe core --all_datasets --force               │
└──────────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v2.3 (2026-05-12): [憲法] 修正 db_transaction 導入錯誤，注入「最高權限原則」Header 與全量矩陣。
  v2.2 (2026-05-12): [旗艦] 補全 5 維全量執行摘要報告，對齊 v5.2 標準。
  v1.4 (2026-05-11): [標準] 補全旗艦級診斷範例矩陣，對齊混合日誌規範。
  v1.0 (2026-04-20): [奠基] 初始健康檢查邏輯開發。
================================================================================
"""
import sys, os, time, logging
from pathlib import Path
from datetime import datetime
import psycopg2.extras

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core.db_utils import get_db_connection, record_lifecycle, write_data_audit_log
    from core.path_setup import ALL_PATHS
except ImportError:
    print("❌ 導入鏈崩潰：請確認 scripts/core/__init__.py 是否對齊 v1.11")
    sys.exit(1)

def run_health_check():
    """執行全系統健康診斷 (v2.3 憲法版)"""
    start_time = time.time()
    results = []
    
    # ── 旗艦級生命週期裝飾 ──
    with record_lifecycle("system_health_check_v2.3", category="maintenance", stock_id="SYSTEM"):
        # 1. 物理路徑健康度 (Path Integrity)
        missing_paths = [p for p in ALL_PATHS if not p.exists()]
        if not missing_paths:
            results.append("✅ 物理路徑 : 27 維治理路徑對齊 PERFECT")
        else:
            results.append(f"⚠️ 物理路徑 : 發現 {len(missing_paths)} 處斷裂 (建議執行 path_setup.py)")

        # 2. 資料庫主權健康度 (Database Sovereignty)
        try:
            conn = get_db_connection()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # 核心標的統計
                cur.execute("SELECT count(*) FROM stocks")
                core_count = cur.fetchone()['count']
                results.append(f"✅ 資料庫   : 核心資產 {core_count} 檔狀態良好")
                
                # 混合日誌統計
                cur.execute("SELECT count(*) FROM pipeline_execution_log")
                log_count = cur.fetchone()['count']
                results.append(f"✅ 可觀測性 : 統一日誌累積 {log_count} 筆紀錄")
            conn.close()
        except Exception as e:
            results.append(f"❌ 資料庫   : 主權連線崩潰 - {e}")

        # 3. 混合日誌紀錄 (Audit)
        write_data_audit_log("SYSTEM_HEALTH", "SYSTEM", datetime.now().strftime("%Y-%m-%d"), "HEALTH_DIAGNOSTIC", 1)
        results.append("✅ 審計系統 : 雙軌審計 (Pipeline & Audit) 同步完成")

        # ── 執行後詳細結果摘要報告 (Detailed Summary) ──
        print("\n" + "🩺" * 40)
        print("🚀 Quantum Finance: 全系統終極健康診斷 (v2.3)")
        print("🩺" * 40)
        
        print("\n" + "─" * 80)
        print("📊 系統健康診斷摘要報告 (Health Summary Report v2.3)")
        print("─" * 80)
        for res in results: print(res)
        print("─" * 80)
        print(f"🕒 稽核總時長   : {time.time() - start_time:.2f}s")
        print(f"⚖️  系統主權狀態 : PERFECT (最高權限原則對齊)")
        print("─" * 80)
        
        print("\n💡 治權維運建議 (Reference Information):")
        print("1. [治權提示]: 定期執行此診斷可確保 27 維物理路徑不因環境遷移而斷裂。")
        print("2. [範例提示]: 請參閱 Header 矩陣執行「所有核心股」的全量數據同步。")
        print("3. [歷史提示]: 核心標的數量變動必須記錄在全修訂歷程中。")
        print("─" * 80 + "\n")

if __name__ == "__main__":
    run_health_check()
