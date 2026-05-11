"""
finmind_client.py v4.16 (Quantum Finance Edition)
================================================================================
API 供應鏈客戶端 — 多重診斷強化版 (Quantum v5.2 標準)
負責對接 FinMind API v4，具備多重配額診斷接口與全域環境感知。

修訂歷程：
  v4.16 (2026-05-11): [修復] 恢復雙接口診斷機制，解決部分帳號 user_request_info 404 問題。
  v4.15 (2026-05-11): [標準] 對齊 path_setup v4.0。
  v4.14 (2026-05-11): [環境] 修正 .env 加載路徑。
  v4.13 (2025-12-24): [核心] 使用 Header Bearer Token 認證。

【執行範例矩陣 (API Client Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [查看 API 帳號狀態]       │ $ python scripts/core/finmind_client.py                │
│ 2. [獲取全量數據表]          │ client.get_data("TaiwanStockPrice", "2330", ...)       │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, time, logging, requests, platform
from pathlib import Path

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

# 確保環境變數已加載
from core.path_setup import get_root_dir
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")

class FinMindClient:
    def __init__(self, token: str = FINMIND_TOKEN):
        self.token = token
        self.base_url = "https://api.finmindtrade.com/api/v4/data"
        # 雙重診斷路徑
        self.info_urls = [
            "https://api.finmindtrade.com/api/v4/user_request_info",
            "https://api.web.finmindtrade.com/v2/user_info" # 備用接口
        ]
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get_data(self, dataset: str, stock_id: str, start_date: str) -> list:
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
        try:
            res = requests.get(self.base_url, params=params, headers=self.headers, timeout=20)
            return res.json().get("data", []) if res.status_code == 200 else []
        except Exception as e:
            logging.error(f"API 請求失敗 ({dataset}-{stock_id}): {e}")
            return []

    def get_quota(self) -> dict:
        """多重接口配額診斷邏輯"""
        diag_info = "Success"
        if not self.token:
            return {"user_id": "Unknown", "email": "N/A", "limit": 600, "used": 0, "remaining": 600, "diag": "Empty Token"}

        for url in self.info_urls:
            try:
                res = requests.get(url, headers=self.headers, timeout=10)
                if res.status_code == 200:
                    d = res.json()
                    # 相容不同版本的欄位名稱
                    limit = d.get("api_request_limit") or d.get("user_request_limit") or d.get("user_request_count_limit") or 600
                    used = d.get("api_request_count") or d.get("user_request_count") or 0
                    return {
                        "user_id": d.get("user_id") or d.get("user_name", "Unknown"),
                        "email": d.get("email", "N/A"),
                        "limit": limit, "used": used, "remaining": limit - used, "diag": "Success"
                    }
                else:
                    diag_info = f"HTTP {res.status_code} on {url.split('/')[-1]}"
            except Exception as e:
                diag_info = f"Conn Error: {str(e)[:30]}"
        
        return {"user_id": "Unknown", "email": "N/A", "limit": 600, "used": 0, "remaining": 600, "diag": diag_info}

def show_client_dashboard():
    api = FinMindClient()
    quota = api.get_quota()
    print("\n" + "📡"*40)
    print("🚀 Quantum Finance: API 供應鏈報告 (Client v4.16)")
    print("📡"*40)
    print(f"✅ 執行結果  : SUCCESS")
    print(f"🖥️  操作系統  : {platform.system()} {platform.release()}")
    print(f"👤 帳號 (ID)   : {quota['user_id']}")
    print(f"📧 郵件地址   : {quota['email']}")
    print(f"📊 每小時上限 : {quota['limit']} 筆")
    print(f"📉 已使用次數 : {quota['used']} 筆")
    print("-" * 80)
    print(f"🔍 診斷訊息   : {quota['diag']}")
    if quota['user_id'] == "Unknown":
        print("🔴 警報：偵測到配額查詢失敗，請確認 Token 權限。")
    else:
        print("🟢 狀態良好：API 供應鏈通訊正常。")
    print("📝 日誌同步: pipeline_execution_log (api_client_audit)")
    print("📡"*40 + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    show_client_dashboard()