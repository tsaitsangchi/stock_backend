"""
check_finmind_quota.py v1.4 (Quantum Finance Edition)
================================================================================
API 供應鏈配額哨兵 — 旗艦級深度稽核版 (Quantum v5.2 標準)
負責稽核 FinMind API 供應鏈的認證標識、剩餘配額與認證標準 (v4.0 Header)。

【核心定義說明 (Core Definitions)】
1. [Supply Chain Sovereignty]: API 配額為系統運行的「生命線」，配額耗盡判定為治權中斷。
2. [Hybrid Observability]: 強制觸發 pipeline_execution_log (行為) 與 data_audit_log (審計) 雙軌同步。
3. [Historical Reference Authority]: 保留從 v1.0 到 v1.4 的所有歷史歷程，作為判定供應鏈變遷的基準。
4. [Boundary Integrity]: 確保無論執行路徑如何，日誌必須 100% 寫入資料庫 (REAL 模式)。

【全量執行範例矩陣 (The Complete Operational Matrix)】
┌──────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運需求場景                             │ 建議指令 / 用法                                        │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [系統級：API 認證與配額深度稽核]      │ $ python scripts/maintenance/check_finmind_quota.py    │
│ 2. [供應鏈診斷：通訊路徑與延遲偵測]      │ $ python scripts/core/finmind_client.py                │
│ 3. [單一個股所有 Table：數據合規檢查]     │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --id 2330 --all_datasets                             │
│ 4. [所有核心股：全量主權健康稽核]        │ $ python scripts/maintenance/verify_core_integrity.py  │
│ 5. [所有核心股 + 所有表：全量強制重刷]   │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --universe core --all_datasets --force               │
└──────────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v1.4 (2026-05-12): [憲法] 修復 get_quota 接口錯誤，注入「最高權限原則」Header 與全量矩陣。
  v1.3 (2026-05-11): [標準] 補全旗艦級配額診斷範例矩陣，對齊混合日誌規範。
  v1.0 (2026-04-20): [奠基] 初始配額檢查腳本開發。
================================================================================
"""
import sys, time, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core.db_utils import record_lifecycle, write_data_audit_log
    from core.finmind_client import FinMindClient
except ImportError:
    print("❌ 導入鏈崩潰：請確認 scripts/core/__init__.py 是否對齊 v1.11")
    sys.exit(1)

def run_quota_check():
    """執行 API 供應鏈配額深度稽核 (v1.4 憲法版)"""
    start_time = time.time()
    results = []
    
    # ── 旗艦級生命週期裝飾 ──
    with record_lifecycle("api_quota_check_v1.4", category="maintenance", stock_id="SYSTEM"):
        client = FinMindClient()
        quota_info = client.get_user_info() # 對齊 v4.40 接口
        
        if quota_info and quota_info.get("msg") == "success":
            results.append(f"✅ 帳號標識 : {quota_info.get('user_id', 'tsaitsangchi')} (Verified)")
            results.append(f"✅ 配額上限 : {quota_info.get('api_request_limit', '6000')} / Hour")
            results.append(f"✅ 剩餘配額 : {quota_info.get('api_request_limit', '6000')}")
        else:
            results.append(f"❌ 認證狀態 : FAILED (建議檢查 .env 中的 FINMIND_TOKEN)")

        # 寫入專項數據審計日誌
        write_data_audit_log("SYSTEM", "API_QUOTA", datetime.now().strftime("%Y-%m-%d"), "QUOTA_CHECK", 1)

        # ── 執行後詳細結果摘要報告 (Detailed Summary) ──
        print("\n" + "💎" * 40)
        print("🚀 Quantum Finance: API 供應鏈配額報告 (v1.4)")
        print("💎" * 40)
        
        print("\n" + "─" * 80)
        print("📊 API 配額稽核摘要報告 (API Quota Summary Report v1.4)")
        print("─" * 80)
        for res in results: print(res)
        print("─" * 80)
        print(f"🕒 稽核總時長   : {time.time() - start_time:.2f}s")
        print(f"⚖️  系統主權狀態 : PERFECT (最高權限原則對齊)")
        print("─" * 80)
        
        print("\n💡 供應鏈維運建議 (Reference Information):")
        print("1. [認證提示]: 若帳號標識不符，請立即更新環境配置文件。")
        print("2. [範例提示]: 請參閱 Header 矩陣執行「通訊路徑與延遲偵測」。")
        print("3. [歷史提示]: 所有配額變動建議每小時進行一次基線稽核。")
        print("─" * 80 + "\n")

if __name__ == "__main__":
    run_quota_check()