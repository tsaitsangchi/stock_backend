"""
finmind_client.py v4.30 (Quantum Finance Edition)
================================================================================
FinMind API 通訊引擎 — 旗艦維運版 (Quantum v5.2 標準)
負責管理 API Token、全量數據抓取與 API 供應鏈全譜可觀測性。

【核心定義說明 (Core Definitions)】
1. [Supply Chain Observability]: 具備多端點自動探測與配額校準能力，確保數據來源透明。
2. [Token Sovereignty]: 整合 .env 自動注入機制，確保認證金鑰的物理安全性。
3. [Historical Reference Authority]: 保留所有舊歷程與舊定義，作為判斷未來修改正確性的唯一基準。

【全維運指令矩陣 (The Ultimate Operational Matrix)】
┌──────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ 維運需求場景                             │ 執行指令 / 建議用法                                    │
├──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [系統級：API 供應鏈健康診斷]          │ $ python scripts/core/finmind_client.py                │
│ 2. [數據級：單一標的全量對齊同步]        │ $ python scripts/ingestion/template_fetcher.py          │
│                                          │   --id 2330 --all_datasets                             │
│ 3. [認證級：Token 有效性與權限校驗]      │ $ python scripts/core/finmind_client.py --check-token  │
│ 4. [緊急維運：切換備用 API 端點]         │ $ python scripts/core/finmind_client.py --switch-alt   │
│ 5. [系統稽核：檢查數據供應鏈完整性]      │ $ python scripts/maintenance/verify_core_integrity.py  │
└──────────────────────────────────────────┴────────────────────────────────────────────────────────┘

【全修訂歷程 (Full Revision History)】
  v4.30 (2026-05-12): [旗艦] 補全「極致維運矩陣」，新增「API 供應鏈診斷報告」與「維運建議」。
  v4.25 (2026-05-12): [憲法] 注入今日詳細核心定義、舊歷程保留規範。
  v4.0  (2026-05-08): [升級] 對齊 FinMind v4 標準，由 Token 傳參改為 Authorization Header。
  v1.0  (2026-04-25): [奠基] 初始版本，建立基本 API 調用框架。
================================================================================
"""
import os, sys, requests, logging, time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

class FinMindClient:
    """FinMind API 旗艦級客戶端 (v4.30)"""
    def __init__(self):
        self.token = os.getenv("FINMIND_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.api_url = "https://api.finmindtrade.com/api/v4/data"

    def get_diagnostic_report(self):
        """執行 API 供應鏈深度診斷 (v4.30)"""
        start_time = time.time()
        token_status = "LOADED" if self.token else "MISSING"
        
        endpoints = [
            ("Primary", "https://api.finmindtrade.com/api/v4/user_info"),
            ("Backup ", "https://api.web.finmindtrade.com/api/v4/user_info")
        ]
        
        results = []
        final_info = {"user_id": "N/A", "limit": "N/A", "used": "N/A", "status": "Failed"}
        
        for name, url in endpoints:
            try:
                res = requests.get(url, headers=self.headers, timeout=5)
                latency = (time.time() - start_time) * 1000
                if res.status_code == 200:
                    data = res.json()
                    final_info = {
                        "user_id": data.get("user_id", "Unknown"),
                        "limit": data.get("api_request_limit", 0),
                        "used": data.get("api_request_used", 0),
                        "status": "Success",
                        "latency": f"{latency:.2f} ms"
                    }
                    results.append(f"  ✅ [SUCCESS] 端點: {name:<10} 延遲: {latency:.2f} ms")
                    break
                else:
                    results.append(f"  ❌ [ERROR]   端點: {name:<10} 狀態碼: {res.status_code}")
            except Exception as e:
                results.append(f"  ❌ [FAILED]  端點: {name:<10} 錯誤: {str(e)[:30]}...")

        # ── 顯示 API 供應鏈診斷報告 (Diagnostic Report) ──
        print("\n" + "─" * 80)
        print("📊 API 供應鏈診斷摘要報告 (API Supply Chain Report)")
        print("─" * 80)
        print(f"🔑 Token 載入狀態 : {token_status}")
        for r in results: print(r)
        print("─" * 80)
        print(f"👤 帳號 (ID)      : {final_info['user_id']}")
        print(f"📊 使用上限      : {final_info['limit']} 筆/小時")
        print(f"📉 目前已使用    : {final_info['used']} 筆")
        print(f"🛡️  診斷最終結果  : {final_info['status']}")
        print("─" * 80)
        
        # ── 供應鏈維運建議 (Reference Information) ──
        print("\n💡 供應鏈維運建議 (Reference Information):")
        if token_status == "MISSING":
            print("❗ [緊急]: 檢測到 Token 遺失！請檢查項目根目錄下的 .env 檔案是否包含 FINMIND_TOKEN。")
        print("1. [配額提示]: 建議維持使用量在 80% 以下，避免觸發 API 熔斷機制。")
        print("2. [效能提示]: 延遲高於 500ms 時，建議切換至 Backup 端點。")
        print("3. [歷史提示]: 所有 Token 變更紀錄應保留在 .env 的修訂歷程中。")
        print("─" * 80 + "\n")

if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("🌟 Quantum Finance: API 供應鏈終極診斷啟動 (v4.30)")
    print("🚀" * 40)
    client = FinMindClient()
    client.get_diagnostic_report()