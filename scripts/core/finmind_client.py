"""
finmind_client.py v6.0 (Trinity Core Final)
================================================================================
修訂歷程：
  v6.0 (2026-05-10): [修正] 強化路徑自癒 Bootstrap，解決 No module named 'core'。
"""
import os, sys, time, requests, logging
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS = None
for p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent]:
    if p.name == "scripts" or (p / "scripts").exists():
        _SCRIPTS = p if p.name == "scripts" else (p / "scripts")
        break
if _SCRIPTS:
    if str(_SCRIPTS) not in sys.path: sys.path.insert(0, str(_SCRIPTS))
    if str(_SCRIPTS.parent) not in sys.path: sys.path.insert(0, str(_SCRIPTS.parent))

try: from core.db_utils import write_pipeline_log
except ImportError:
    try: from db_utils import write_pipeline_log
    except ImportError:
        def write_pipeline_log(*a, **k): pass

class FinMindClient:
    def __init__(self):
        self.api_url = "https://api.finmindtrade.com/api/v4/data"
        self.token = os.getenv("FINMIND_TOKEN", "")
        logging.info("🟢 FinMindClient v6.0 初始化完成")

    def get_data(self, dataset: str, stock_id: str, start_date: str):
        t0 = time.monotonic()
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date, "token": self.token}
        try:
            resp = requests.get(self.api_url, params=params, timeout=20)
            data = resp.json().get("data", [])
            status = "success"; rows = len(data)
        except Exception as e:
            data = []; status = "failed"; rows = 0
            logging.error(f"❌ API 失敗 ({dataset}): {e}")
        
        write_pipeline_log(f"api_{dataset}", stock_id, status, "ingestion", int((time.monotonic()-t0)*1000), rows)
        return data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    api = FinMindClient(); res = api.get_data("TaiwanStockPrice", "2330", "2024-05-01")
    print(f"✅ 測試抓取筆數: {len(res)}")