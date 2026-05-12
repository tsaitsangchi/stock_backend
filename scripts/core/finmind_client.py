"""
finmind_client.py v4.32 (Quantum Finance Edition)
================================================================================
FinMind API 通訊引擎 — 協議轉型版 (Quantum v5.2 標準)
負責管理 API Token、全量數據抓取與 API 供應鏈全譜可觀測性。

【修訂歷程】
  v4.32 (2026-05-12): [協議] 將診斷端點改為 POST 請求 (解決 405 錯誤)，並優化配額解析邏輯。
  v4.31 (2026-05-12): [校準] 嘗試修正 404 問題，確定 GET 協議在某些端點不被支持。
  v4.30 (2026-05-12): [旗艦] 補全「極致維運矩陣」，新增「API 供應鏈診斷報告」。
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
    def __init__(self):
        self.token = os.getenv("FINMIND_TOKEN")
        self.api_url = "https://api.finmindtrade.com/api/v4/data"

    def get_diagnostic_report(self):
        """執行 API 供應鏈深度診斷 (v4.32 協議轉型版)"""
        start_time = time.time()
        token_status = "LOADED" if self.token else "MISSING"
        
        # 使用 POST 請求 /login 端點是獲取使用者資訊的權威方式
        login_url = "https://api.finmindtrade.com/api/v4/login"
        
        results = []
        final_info = {"user_id": "N/A", "limit": "N/A", "used": "N/A", "status": "Failed"}
        
        try:
            # 協議轉型：改為 POST 並傳入 token
            res = requests.post(login_url, data={"token": self.token}, timeout=5)
            latency = (time.time() - start_time) * 1000
            
            if res.status_code == 200:
                data = res.json()
                final_info = {
                    "user_id": data.get("user_id", "Unknown"),
                    "limit": data.get("api_request_limit", 600),
                    "used": data.get("api_request_used", 0),
                    "status": "Success",
                    "latency": f"{latency:.2f} ms"
                }
                results.append(f"  ✅ [SUCCESS] 協議: POST 端點: /login 延遲: {latency:.2f} ms")
            else:
                results.append(f"  ❌ [ERROR]   協議: POST 端點: /login 狀態碼: {res.status_code}")
                # 備援計畫：嘗試透過數據介面獲取資訊
                res_data = requests.get(self.api_url, params={"dataset": "TaiwanStockInfo", "token": self.token}, timeout=5)
                if res_data.status_code == 200:
                    results.append(f"  ⚠️ [RECOVERY] 數據接口通暢，但配額資訊無法直接獲取。")
                    final_info["status"] = "Partial (Data OK, Meta Fail)"
        except Exception as e:
            results.append(f"  ❌ [FAILED]  協議: POST 錯誤: {str(e)[:30]}...")

        print("\n" + "─" * 80)
        print("📊 API 供應鏈診斷摘要報告 (API Supply Chain Report v4.32)")
        print("─" * 80)
        print(f"🔑 Token 載入狀態 : {token_status}")
        for r in results: print(r)
        print("─" * 80)
        print(f"👤 帳號 (ID)      : {final_info['user_id']}")
        print(f"📊 使用上限      : {final_info['limit']} 筆/小時")
        print(f"📉 目前已使用    : {final_info['used']} 筆")
        print(f"🛡️  診斷最終結果  : {final_info['status']}")
        print("─" * 80 + "\n")

if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("🌟 Quantum Finance: API 供應鏈終極診斷啟動 (v4.32)")
    print("🚀" * 40)
    client = FinMindClient()
    client.get_diagnostic_report()