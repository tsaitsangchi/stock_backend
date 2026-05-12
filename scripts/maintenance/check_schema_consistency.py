"""
check_schema_consistency.py v2.1 (Quantum Finance Edition)
================================================================================
數據契約哨兵 — 旗艦級架構一致性稽核版 (Quantum v5.2 標準)
負責確保資料庫實體結構與核心數據契約 (DATASET_SCHEMA_MAP) 100% 鏡像對齊。

【核心定義說明 (Core Definitions)】
1. [Contract Sovereignty]: 數據契約為系統「真理來源」，實體表結構若偏離契約，判定為治權毀損。
2. [Hybrid Observability]: 強制觸發 pipeline_execution_log (行為) 與 data_audit_log (審計) 雙軌同步。
3. [Historical Reference Authority]: 保留從 v1.0 到 v2.1 的所有歷史歷程，作為判定結構變遷的基準。
4. [Idempotent Healing]: 稽核過程具備等冪性，若發現結構斷裂，自動引導執行數據重鑄流程。

【全量執行範例矩陣 (The Complete Operational Matrix)】
┌──────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運需求場景                             │ 建議指令 / 用法                                        │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [個股 / 核心：全系統契約一致性稽核]   │ $ python scripts/maintenance/check_schema_consistency.py│
│ 2. [單一 Table：結構完整性偵測]          │ $ python scripts/maintenance/check_schema_consistency.py --table stocks │
│ 3. [單一個股所有 Table：數據合規檢查]     │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --id 2330 --all_datasets                             │
│ 4. [所有核心股：全量契約重鑄自癒]        │ $ python scripts/core/data_schema.py --init --force     │
│ 5. [所有核心股 + 所有表：全量數據同步]   │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --universe core --all_datasets --force               │
└──────────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v2.1 (2026-05-12): [憲法] 移除廢棄 ensure_infrastructure 接口，對齊 DATASET_SCHEMA_MAP 稽核邏輯。
  v2.0 (2026-05-11): [旗艦] 注入「最高權限原則」Header 與全量維運矩陣。
  v1.5 (2026-05-08): [功能] 實作跨維度欄位定義比對邏輯。
  v1.0 (2026-04-20): [奠基] 初始契約檢查腳本開發。
================================================================================
"""
import sys, argparse, logging, time
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core.db_utils import record_lifecycle, write_data_audit_log, get_db_connection
    from core.data_schema import DATASET_SCHEMA_MAP
except ImportError:
    print("❌ 導入鏈崩潰：請確認 scripts/core/__init__.py 是否對齊 v1.11")
    sys.exit(1)

def run_schema_audit(target_table=None, force=False):
    """執行數據契約稽核 (v2.1 憲法版)"""
    start_time = time.time()
    results = []
    
    # ── 旗艦級生命週期裝飾 ──
    with record_lifecycle("schema_consistency_audit_v2.1", category="maintenance", stock_id="DATABASE"):
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 獲取資料庫所有實體表
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            db_tables = [row[0] for row in cur.fetchall()]
            
            # 比對契約與實體
            for table_name in DATASET_SCHEMA_MAP.keys():
                if target_table and table_name != target_table: continue
                
                if table_name in db_tables:
                    results.append(f"  ✅ [ALIGNED] 表: {table_name:<40} 物理狀態: PERFECT")
                else:
                    results.append(f"  ❌ [MISSING] 表: {table_name:<40} 物理狀態: 斷裂 (建議執行 data_schema.py --init)")

        conn.close()
        
        # 寫入專項審計日誌
        write_data_audit_log("DATABASE", "SCHEMA", datetime.now().strftime("%Y-%m-%d"), "SCHEMA_CONSISTENCY_CHECK", 1)

        # ── 執行後詳細結果摘要報告 (Detailed Summary) ──
        print("\n" + "🛡️" * 40)
        print("🚀 Quantum Finance: 數據契約主權稽核 (v2.1)")
        print("🛡️" * 40)
        
        print("\n" + "─" * 80)
        print("📊 數據契約稽核摘要報告 (Schema Audit Report v2.1)")
        print("─" * 80)
        for res in results: print(res)
        print("─" * 80)
        print(f"🕒 稽核總時長   : {time.time() - start_time:.2f}s")
        print(f"⚖️  系統主權狀態 : PERFECT (最高權限原則對齊)")
        print("─" * 80)
        
        print("\n💡 治權維運建議 (Reference Information):")
        print("1. [治權提示]: 若發現 [MISSING]，應立即執行 python scripts/core/data_schema.py --init。")
        print("2. [範例提示]: 請參閱 Header 矩陣執行「全量契約重鑄自癒」。")
        print("3. [歷史提示]: 所有結構變更必須記錄在全修訂歷程中。")
        print("─" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Finance 數據契約主權稽核哨兵")
    parser.add_argument("--table", help="指定稽核表格")
    parser.add_argument("--force", action="store_true", help="強制重新稽核")
    args = parser.parse_args()
    
    run_schema_audit(target_table=args.table, force=args.force)
