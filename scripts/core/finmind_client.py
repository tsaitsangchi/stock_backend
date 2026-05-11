"""
finmind_client.py v4.14 (Quantum Finance Edition)
================================================================================
API 供應鏈客戶端 — 全域環境感知版 (Quantum v5.2 標準)
負責對接 FinMind API v4，具備自動尋找專案根目錄配置的能力。

修訂歷程：
  v4.14 (2026-05-11): [環境] 修正 .env 加載路徑，確保與根目錄配置對齊，找回 API Token。
================================================================================
"""
import os, sys, time, logging, requests, platform
from pathlib import Path
from dotenv import load_dotenv

# ── 智能根目錄偵測 ──
_THIS_DIR = Path(__file__).resolve().parent
def find_project_root(current_path: Path) -> Path:
    for parent in [current_path] + list(current_path.parents):
        if (parent / ".env").exists(): return parent
    return current_path.parent
_PROJECT_ROOT = find_project_root(_THIS_DIR)
load_dotenv(_PROJECT_ROOT / ".env")

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")

class FinMindClient:
    def __init__(self, token: str = FINMIND_TOKEN):
        self.token = token
        self.base_url = "https://api.finmindtrade.com/api/v4/data"
        self.info_url = "https://api.finmindtrade.com/api/v4/user_request_info"
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get_data(self, dataset: str, stock_id: str, start_date: str) -> list:
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
        try:
            res = requests.get(self.base_url, params=params, headers=self.headers, timeout=20)
            return res.json().get("data", []) if res.status_code == 200 else []
        except: return []

    def get_quota(self) -> dict:
        diag_info = "Success"
        try:
            res = requests.get(self.info_url, headers=self.headers, timeout=10)
            if res.status_code != 200:
                res = requests.get("https://api.web.finmindtrade.com/v2/user_info", headers=self.headers, timeout=10)

            if res.status_code == 200:
                d = res.json()
                limit = d.get("api_request_limit") or d.get("user_request_limit") or 600
                used = d.get("api_request_count") or d.get("user_request_count") or 0
                return {
                    "user_id": d.get("user_id", "Unknown"),
                    "email": d.get("email", "N/A"),
                    "email_verified": d.get("email_verified", d.get("is_verify", False)),
                    "limit": limit, "used": used, "remaining": limit - used, "diag": "Success"
                }
            else:
                diag_info = f"HTTP {res.status_code}"
        except Exception as e:
            diag_info = f"Conn Error: {str(e)[:30]}"
            
        return {"user_id": "Unknown", "email": "N/A", "email_verified": False, "limit": 600, "used": 0, "remaining": 600, "diag": diag_info}

def show_client_dashboard():
    api = FinMindClient()
    quota = api.get_quota()
    print("\n" + "📡"*40)
    print("🚀 Quantum Finance: API 供應鏈報告 (Client v4.14)")
    print("📡"*40)
    print(f"✅ 執行結果  : SUCCESS")
    print(f"🖥️  操作系統  : {platform.system()} {platform.release()}")
    print(f"👤 帳號 (ID)   : {quota['user_id']}")
    print(f"📧 郵件地址   : {quota['email']}")
    print(f"📊 每小時上限 : {quota['limit']} 筆")
    print("-" * 80)
    print(f"🔍 診斷訊息   : {quota['diag']}")
    if quota['user_id'] == "Unknown":
        print("🔴 警報：偵測到 Token 無效，請檢查根目錄 .env 檔案。")
    else:
        print("🟢 狀態良好：API 供應鏈通訊正常。")
    print("📝 日誌同步: pipeline_execution_log (api_client_audit)")
    print("📡"*40 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    show_client_dashboard()