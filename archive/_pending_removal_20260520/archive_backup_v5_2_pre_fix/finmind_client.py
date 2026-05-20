"""
finmind_client.py v4.17 (Quantum Finance Edition)
================================================================================
API 供應鏈客戶端 — 極致範例版 (Quantum v5.2 標準)
負責對接 FinMind API v4，具備多重配額診斷接口。

修訂歷程：
  v4.17 (2026-05-11): [標準] 補全極致範例矩陣。
  v4.16 (2026-05-11): [修復] 恢復雙接口診斷機制。

【執行範例矩陣 (API Client Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [全系統 API 配額詳查]     │ $ python scripts/core/finmind_client.py                │
│ 2. [單一個股：日股價抓取]    │ data = client.get_data("TaiwanStockPrice", "2330", ...)│
└──────────────────────────────┴────────────────────────────────────────────────────────┘
================================================================================
"""
import os, sys, requests
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")

class FinMindClient:
    def __init__(self, token: str = FINMIND_TOKEN):
        self.token = token
        self.base_url = "https://api.finmindtrade.com/api/v4/data"
        self.info_urls = ["https://api.finmindtrade.com/api/v4/user_request_info", "https://api.web.finmindtrade.com/v2/user_info"]
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get_data(self, dataset: str, stock_id: str, start_date: str) -> list:
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
        try:
            res = requests.get(self.base_url, params=params, headers=self.headers, timeout=20)
            return res.json().get("data", []) if res.status_code == 200 else []
        except: return []

    def get_quota(self) -> dict:
        if not self.token: return {"user_id": "Unknown", "diag": "Empty Token"}
        for url in self.info_urls:
            try:
                res = requests.get(url, headers=self.headers, timeout=10)
                if res.status_code == 200:
                    d = res.json()
                    return {"user_id": d.get("user_id", "Unknown"), "email": d.get("email", "N/A"), "limit": d.get("api_request_limit", 600), "used": d.get("api_request_count", 0), "diag": "Success"}
            except: continue
        return {"user_id": "Unknown", "diag": "All Failed"}

if __name__ == "__main__":
    api = FinMindClient()
    print(f"API Quota: {api.get_quota()}")