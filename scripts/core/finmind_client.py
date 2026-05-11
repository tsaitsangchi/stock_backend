"""
finmind_client.py v4.20 (Quantum Finance Edition)
================================================================================
FinMind API 通訊引擎 — 高可用與配額稽核旗艦版 (Quantum v5.2 標準)
負責管理 API Token、自動配額追蹤與高效能數據抓取。

修訂歷程：
  v4.20 (2026-05-11): [標準] 補全旗艦級範例矩陣，整合 API 呼叫審計日誌。
  v4.19 (2026-05-08): [修復] 更新授權 Header，對齊 FinMind API v4 標準。

【執行範例矩陣 (API Client Matrix)】
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ 需求場景                     │ 建議指令 / 用法                                        │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. [系統級：API 配額診斷]    │ $ python scripts/core/finmind_client.py                │
│ 2. [單一標的：抓取股價數據]  │ data = client.get_data("TaiwanStockPrice", "2330", ...)│
│ 3. [所有核心股：配額壓力測試]│ $ python scripts/maintenance/check_finmind_quota.py    │
└──────────────────────────────┴────────────────────────────────────────────────────────┘

【可觀測性紀錄 (Observability)】
  - 統一本誌 (Unified): pipeline_execution_log (Task: api_call_{dataset})
  - 專項審計 (Audit): data_audit_log (Action: API_REQUEST)
================================================================================
"""
import os, sys, time, requests, logging
from pathlib import Path
from datetime import datetime

# ── 系統級架構引導 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core.db_utils import record_lifecycle, write_data_audit_log
except ImportError:
    def record_lifecycle(*args, **kwargs): pass
    def write_data_audit_log(*args, **kwargs): pass

class FinMindClient:
    def __init__(self):
        self.token = os.getenv("FINMIND_TOKEN", "your_token")
        self.api_url = "https://api.finmindtrade.com/api/v4/data"
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get_data(self, dataset, data_id, start_date, end_date=None):
        """抓取數據並紀錄審計日誌"""
        params = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start_date,
        }
        if end_date: params["end_date"] = end_date
        
        with record_lifecycle(f"api_call_{dataset}", category="api", stock_id=data_id):
            try:
                resp = requests.get(self.api_url, params=params, headers=self.headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    # 🔴 專項審計
                    write_data_audit_log(dataset, data_id, start_date, "API_REQUEST", len(data))
                    return data
                else:
                    logging.error(f"❌ API 錯誤: {resp.status_code} {resp.text}")
                    return []
            except Exception as e:
                logging.error(f"❌ API 連線失敗: {e}")
                return []

    def get_quota(self):
        url = "https://api.finmindtrade.com/api/v4/user_info"
        resp = requests.get(url, headers=self.headers)
        return resp.json()

if __name__ == "__main__":
    client = FinMindClient()
    print("-" * 50)
    print(f"🚀 FinMindClient v4.20 自我診斷啟動...")
    try:
        quota = client.get_quota()
        print(f"📡 API 配額狀態 : {quota}")
    except Exception as e:
        print(f"❌ 診斷失敗 : {e}")
    print("-" * 50)