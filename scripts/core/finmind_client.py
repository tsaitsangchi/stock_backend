"""
finmind_client.py v3.8 (Quantum Finance Edition)
================================================================================
FinMind API 客戶端 — 混合日誌與韌性觀測版 (Quantum v5.1 標準)
整合 Token Bucket 速率限制、斷路器、以及生命週期監測。

修訂歷程：
  v3.8 (2026-05-11): [標準化] 擴充全場景執行範例，強化配額診斷邏輯。
  v3.7 (2026-05-11): [診斷] 強化 user_info 偵錯，支援 v4 端點與暴力解析。
  v3.6 (2026-05-11): [修復] 強化驗證相容性，支援 Header/Param 雙傳參。

執行範例 (Comprehensive Usage Examples):
  1. [配額與帳號診斷] 查看 API 剩餘配額與 Token 狀態:
     python scripts/core/finmind_client.py

  2. [個股 + 單表] 抓取台積電(2330)日成交資料:
     client = FinMindClient()
     data = client.get_data("TaiwanStockPrice", "2330", "2024-05-01")

  3. [單一個股 + 所有表] 檢查特定個股並補全缺失數據:
     sid = "2330"
     datasets = ["TaiwanStockPrice", "TaiwanStockFinancialStatements"]
     for ds in datasets:
         # 結合 db_utils.get_latest_date 獲取 start_date
         pass

  4. [所有核心股 + 所有表] 強制更新 (Forced Update / Full Reload):
     from core.db_utils import get_db_stock_ids
     stocks = get_db_stock_ids()
     for sid in stocks:
         for ds in datasets:
             # 強制從遠古日期重抓數據
             data = client.get_data(ds, sid, "2010-01-01")
================================================================================
"""
import os, sys, time, requests, logging, random
from pathlib import Path
from typing import Dict, Any, List, Optional
from threading import Lock
from datetime import datetime

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

try:
    from core.path_setup import ensure_scripts_on_path, get_scripts_dir
    ensure_scripts_on_path(__file__)
except ImportError:
    import path_setup
    path_setup.ensure_scripts_on_path(__file__)
    from path_setup import get_scripts_dir

try:
    from dotenv import load_dotenv
    _ENV_PATH = get_scripts_dir().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    else:
        load_dotenv()
except: pass

try:
    from core.db_utils import write_pipeline_log
except ImportError:
    def write_pipeline_log(*a, **k): pass

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = {}
        self.open_until = {}
        self._lock = Lock()

    def check(self, dataset: str):
        with self._lock:
            if dataset in self.open_until:
                if time.time() < self.open_until[dataset]: return False
                else: del self.open_until[dataset]
        return True

    def record_success(self, dataset: str):
        with self._lock: self.failures[dataset] = 0

    def record_failure(self, dataset: str):
        with self._lock:
            count = self.failures.get(dataset, 0) + 1
            self.failures[dataset] = count
            if count >= self.failure_threshold:
                self.open_until[dataset] = time.time() + self.recovery_timeout
                logger.warning(f"⚠️ [Circuit] {dataset} 已斷路")

class TokenBucketRateLimiter:
    def __init__(self, capacity: int = 600, refill_rate: float = 0.167):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self):
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1; return True
            return False

global_circuit_breaker = CircuitBreaker()
_FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
global_rate_limiter = TokenBucketRateLimiter(
    capacity=6000 if _FINMIND_TOKEN else 600, 
    refill_rate=1.67 if _FINMIND_TOKEN else 0.167
)

class FinMindClient:
    def __init__(self):
        self.api_url = "https://api.finmindtrade.com/api/v4/data"
        self.token = os.getenv("FINMIND_TOKEN", "")
        if self.token:
            logger.info(f"🟢 FinMindClient v3.8 初始化 (Token: Yes, Limit: 6000/hr)")
        else:
            logger.warning(f"🟡 FinMindClient v3.8 初始化 (Token: No, Limit: 600/hr)")

    def get_user_info(self) -> Dict[str, Any]:
        """多重測試不同的 UserInfo 端點以獲獲取資訊"""
        if not self.token: return {}
        endpoints = ["https://api.web.finmindtrade.com/v2/user_info", "https://api.finmindtrade.com/api/v4/user_info"]
        for url in endpoints:
            try:
                resp = requests.get(url, headers={"Authorization": f"Bearer {self.token}"}, timeout=10)
                if resp.status_code != 200:
                    resp = requests.get(url, params={"token": self.token}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if "data" in data and isinstance(data["data"], dict): return data["data"]
                    return data
            except: continue
        return {}

    def get_data(self, dataset: str, stock_id: str, start_date: str, max_retries: int = 3):
        t0 = time.monotonic()
        if not global_circuit_breaker.check(dataset): return []
        while not global_rate_limiter.acquire(): time.sleep(0.5)
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date, "token": self.token}
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(self.api_url, params=params, timeout=20)
                if resp.status_code == 402: time.sleep(65); continue
                if resp.status_code == 422: break
                resp.raise_for_status()
                data = resp.json().get("data", [])
                global_circuit_breaker.record_success(dataset)
                duration = int((time.monotonic() - t0) * 1000)
                write_pipeline_log(f"api_{dataset}", stock_id, "success", "ingestion", duration, len(data))
                return data
            except Exception as e:
                last_err = e
                global_circuit_breaker.record_failure(dataset)
                time.sleep(2 ** attempt)
        duration = int((time.monotonic() - t0) * 1000)
        write_pipeline_log(f"api_{dataset}", stock_id, "failed", "ingestion", duration, 0, err=last_err)
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    client = FinMindClient()
    info = client.get_user_info()
    print("\n" + "="*45)
    print("📊 Quantum Finance: FinMind 配額診斷報告 (v3.8)")
    print("="*45)
    if info:
        u_id = info.get('user_id') or info.get('user_name') or info.get('uid') or info.get('account', 'N/A')
        print(f"帳號 (User Identity) : {u_id}")
        print(f"電子郵件 (Email)      : {info.get('email', 'N/A')}")
        is_verified = info.get("email_verify") or info.get("is_verify")
        print(f"驗證狀態 (Verified)  : {'✅ 已驗證' if is_verified else '❌ 未驗證 / 狀態未知'}")
        print(f"每小時使用上限 (Limit): {info.get('api_request_limit', '6000')}")
        print(f"已使用次數 (Used)     : {info.get('user_count', '0')}")
    else: print("❌ 無法獲取使用者資訊。請檢查 Token 有效性。")
    print("="*45 + "\n")