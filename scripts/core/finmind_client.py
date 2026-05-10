"""
finmind_client.py v3.4 (Quantum Finance Edition)
================================================================================
FinMind API 客戶端 — 混合日誌與韌性觀測版 (Quantum v5.1 標準)
整合 Token Bucket 速率限制、斷路器、以及多層級異常捕獲。

修訂歷程：
  v3.4 (2026-05-10): [修正] 修復重試邏輯中 e 變數範圍問題，優化 422 錯誤日誌。
  v3.2 (2026-05-10): [修正] 導入 load_dotenv 自動加載根目錄 .env。
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
        if self.token: logger.info(f"🟢 FinMindClient v3.2 初始化完成 (Token: Yes, Limit: 6000/hr)")
        else: logger.warning(f"🟡 FinMindClient v3.2 初始化完成 (Token: No, Limit: 600/hr)")

    def get_data(self, dataset: str, stock_id: str, start_date: str, max_retries: int = 3):
        t0 = time.monotonic()
        if not global_circuit_breaker.check(dataset): return []

        while not global_rate_limiter.acquire(): time.sleep(0.5)

        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date, "token": self.token}
        last_err = "Unknown Error"
        
        for attempt in range(max_retries):
            try:
                resp = requests.get(self.api_url, params=params, timeout=20)
                if resp.status_code == 402:
                    time.sleep(65); continue
                if resp.status_code == 422:
                    logger.error(f"❌ [API] 422 參數錯誤: Dataset={dataset}, ID={stock_id}, Start={start_date}")
                    break # 參數錯誤不需要重試
                
                resp.raise_for_status()
                data = resp.json().get("data", [])
                global_circuit_breaker.record_success(dataset)
                write_pipeline_log(f"api_{dataset}", stock_id, "success", "ingestion", int((time.monotonic()-t0)*1000), len(data))
                return data
            except Exception as e:
                last_err = str(e)
                logger.error(f"❌ [API] {dataset} 抓取失敗 ({attempt+1}): {e}")
                global_circuit_breaker.record_failure(dataset)
                time.sleep(2 ** attempt)
        
        write_pipeline_log(f"api_{dataset}", stock_id, "failed", "ingestion", int((time.monotonic()-t0)*1000), 0, last_err)
        return []