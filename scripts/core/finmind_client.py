"""
finmind_client.py v4.34 (Quantum Finance Edition)
================================================================================
FinMind API 通訊引擎 — 認證回歸版 (Quantum v5.2 標準)
負責管理 API Token、全量數據抓取與 API 供應鏈全譜可觀測性。

【修訂歷程】
  v4.34 (2026-05-12): [修復] 認證回歸至 v4.0 Header 標準 (解決 v4.33 的 400 錯誤)，並改用數據感測器偵測供應鏈。
  v4.33 (2026-05-12): [憲法] 補全全量修訂歷程，確保 2026-04-25 至今的技術軌跡完整性。
  v4.32 (2026-05-12): [協議] 將診斷端點改為 POST 請求 (解決 405 錯誤)，並優化配額解析邏輯。
  v4.31 (2026-05-12): [校準] 嘗試修正 404 問題，確定 GET 協議在某些端點不被支持。
  v4.30 (2026-05-12): [旗艦] 補全「極致維運矩陣」，新增「API 供應鏈診斷報告」。
  v4.25 (2026-05-12): [憲法] 注入今日詳細核心定義、舊歷程保留規範，對齊 2026-05-12 旗艦要求。
  v4.24 (2026-05-11): [標準] 補全極致範例矩陣與歷史歷程，確立為 v5.2 供應鏈憲法。
  v4.23 (2026-05-11): [修復] 加入 load_dotenv() 確保 Token 自動載入，解決環境變數漂移問題。
  v4.22 (2026-05-11): [診斷] 實現多端點自動探測與手動配額校準邏輯。
  v4.0  (2026-05-08): [升級] 對齊 FinMind v4 標準，由 Token 傳參改為 Authorization Header。
  v1.0  (2026-04-25): [奠基] 初始版本，建立基本 API 調用與數據解析框架。
================================================================================
"""
import os, sys, requests, time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

class FinMindClient:
    """FinMind API 旗艦級客戶端 (v4.34 認證回歸版)"""
    def __init__(self):
        self.token = os.getenv("FINMIND_TOKEN")
        # 回歸 v4.0 標準：使用 Authorization Header
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.api_url = "https://api.finmindtrade.com/api/v4/data"

    def get_diagnostic_report(self):
        """執行 API 供應鏈深度診斷 (v4.34 數據感測器模式)"""
        start_time = time.time()
        token_status = "LOADED" if self.token else "MISSING"
        
        # 認證回歸判定：改用 /data 端點請求 TaiwanStockInfo 進行存取權限測試
        test_params = {
            "dataset": "TaiwanStockInfo",
            "data_id": "2330",  # 請求台積電元數據進行感測
            "start_date": "2026-01-01"
        }
        
        results = []
        final_info = {"user_id": "N/A", "limit": "N/A", "used": "N/A", "status": "Failed"}
        
        try:
            # 使用 v4.0 標準 Header 進行感測
            res = requests.get(self.api_url, params=test_params, headers=self.headers, timeout=5)
            latency = (time.time() - start_time) * 1000
            
            if res.status_code == 200:
                data = res.json()
                if data.get("msg") == "success":
                    final_info = {
                        "user_id": "FinMind User (Verified)",
                        "limit": "See Web Console",
                        "used": "Verified (Data Path OK)",
                        "status": "Success (Data Sensor OK)",
                        "latency": f"{latency:.2f} ms"
                    }
                    results.append(f"  ✅ [SUCCESS] 感測器: 數據介面 (/data) 延遲: {latency:.2f} ms")
                else:
                    results.append(f"  ❌ [DENIED]  感測器: 數據介面權限不足: {data.get('msg')}")
            else:
                results.append(f"  ❌ [ERROR]   感測器: 數據介面狀態碼: {res.status_code}")
                # 嘗試舊版 URL 傳參備援
                res_v1 = requests.get(f"{self.api_url}?token={self.token}", params=test_params, timeout=5)
                if res_v1.status_code == 200:
                    results.append(f"  ⚠️ [RECOVERY] v1.0 URL 參數傳參模式可用。")
                    final_info["status"] = "Success (Legacy Mode)"
        except Exception as e:
            results.append(f"  ❌ [FAILED]  感測器偵測錯誤: {str(e)[:30]}...")

        print("\n" + "─" * 80)
        print("📊 API 供應鏈診斷摘要報告 (API Supply Chain Report v4.34)")
        print("─" * 80)
        print(f"🔑 Token 載入狀態 : {token_status}")
        for r in results: print(r)
        print("─" * 80)
        print(f"👤 帳號 (ID)      : {final_info['user_id']}")
        print(f"📊 使用上限      : {final_info['limit']}")
        print(f"📉 目前已使用    : {final_info['used']}")
        print(f"🛡️  診斷最終結果  : {final_info['status']}")
        print("─" * 80 + "\n")

if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("🌟 Quantum Finance: API 供應鏈終極診斷啟動 (v4.34)")
    print("🚀" * 40)
    client = FinMindClient()
    client.get_diagnostic_report()