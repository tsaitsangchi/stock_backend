"""
finmind_client.py v4.0
量化系統核心：FinMind API 企業級客戶端 (整合快取與非同步 I/O)
================================================================================
v4.0 重大升級：
  · 引入 SQLite 本機快取：相同條件的請求在 24 小時內直接從本機讀取，0 延遲且不扣 API 配額。
  · 引入 Async I/O：支援 `async_get_data`，允許未來系統使用 asyncio 進行極速併發抓取。
  · 非阻塞令牌桶 (Async Token Bucket)：並行抓取時不會因為 time.sleep() 阻塞整個 Event Loop。

執行範例（同步模式 - 向下相容現有 fetchers）：
    from core.finmind_client import FinMindClient
    api = FinMindClient()
    # 第一次會消耗配額並寫入本機 sqlite 快取
    data1 = api.get_data("TaiwanStockPrice", "2330", "2023-01-01", "2023-01-10")
    # 第二次會直接從快取返回，耗時 < 5ms
    data2 = api.get_data("TaiwanStockPrice", "2330", "2023-01-01", "2023-01-10")

執行範例（非同步模式 - 適合未來的高頻併發管線）：
    import asyncio
    from core.finmind_client import FinMindClient
    
    async def fetch_multiple():
        api = FinMindClient()
        tasks = [
            api.async_get_data("TaiwanStockPrice", sid, "2023-01-01", "2023-01-10")
            for sid in ["2330", "2317", "2454"]
        ]
        results = await asyncio.gather(*tasks)
        print(f"完成併發抓取，取得 {len(results)} 檔股票資料")
        
    asyncio.run(fetch_multiple())
"""

import os
import time
import json
import logging
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

# 嘗試載入 aiohttp 以支援非同步操作，若無則優雅降級
try:
    import aiohttp
    import asyncio
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# Compatibility Layer (Shims for old scripts)
# =====================================================================

class BatchNotSupportedError(Exception):
    """Shim for scripts expecting this error class."""
    pass

def finmind_get(dataset: str, data_id: str, start_date: str, end_date: str, **kwargs) -> List[Dict]:
    """Shim for old scripts calling finmind_get directly."""
    return FinMindClient().get_data(dataset, data_id, start_date, end_date)

async def finmind_get_async(dataset: str, data_id: str, start_date: str, end_date: str, **kwargs) -> List[Dict]:
    """Shim for old scripts calling finmind_get_async directly."""
    return await FinMindClient().async_get_data(dataset, data_id, start_date, end_date)

# =====================================================================
# 1. 狀態統計與快取層
# =====================================================================

class RequestStats:
    """追蹤 API 請求的健康度與快取命中率"""
    def __init__(self):
        self.success = 0
        self.failed = 0
        self.cache_hits = 0
        self.lock = threading.Lock()

    def record_success(self):
        with self.lock: self.success += 1

    def record_failure(self):
        with self.lock: self.failed += 1
        
    def record_cache_hit(self):
        with self.lock: self.cache_hits += 1

    def summary(self):
        total = self.success + self.failed + self.cache_hits
        logger.info(f"📊 FinMind API 統計: 總請求={total} | 成功={self.success} | 快取命中={self.cache_hits} | 失敗={self.failed}")

class SQLiteCache:
    """輕量級本機資料快取，防止開發與重試時浪費 API 額度"""
    def __init__(self, db_path="outputs/finmind_cache.db", ttl_hours=24):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME
                )
            ''')

    def _generate_key(self, dataset: str, data_id: str, start_date: str, end_date: str) -> str:
        key_data = f"{dataset}_{data_id}_{start_date}_{end_date}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()

    def get(self, dataset: str, data_id: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        key = self._generate_key(dataset, data_id, start_date, end_date)
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT response, timestamp FROM api_cache WHERE cache_key=?", (key,))
                row = cur.fetchone()
                if row:
                    cached_time = datetime.fromisoformat(row[1])
                    if datetime.now() - cached_time < self.ttl:
                        return json.loads(row[0])
                    else:
                        # 快取過期，清理空間
                        cur.execute("DELETE FROM api_cache WHERE cache_key=?", (key,))
        return None

    def set(self, dataset: str, data_id: str, start_date: str, end_date: str, response_data: List[Dict]):
        key = self._generate_key(dataset, data_id, start_date, end_date)
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "REPLACE INTO api_cache (cache_key, response, timestamp) VALUES (?, ?, ?)",
                    (key, json.dumps(response_data), datetime.now().isoformat())
                )

# =====================================================================
# 2. 流量防護層 (Token Bucket & Circuit Breaker)
# =====================================================================

class TokenBucket:
    """支援同步與非同步的速率限制器 (Rate Limiter)"""
    def __init__(self, capacity: int = 300, fill_rate: float = 300/3600): # 預設 300次/小時
        self.capacity = float(capacity)
        self.fill_rate = fill_rate
        self.tokens = float(capacity)
        self.last_fill = time.monotonic()
        self.lock = threading.Lock()

    def _add_tokens(self):
        now = time.monotonic()
        new_tokens = (now - self.last_fill) * self.fill_rate
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_fill = now

    def consume(self, tokens: int = 1):
        """同步等待直到有足夠的 Token"""
        while True:
            with self.lock:
                self._add_tokens()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
            time.sleep(0.1)

    async def async_consume(self, tokens: int = 1):
        """非同步等待，不阻塞 Event Loop"""
        while True:
            with self.lock:
                self._add_tokens()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
            await asyncio.sleep(0.1)

class CircuitBreaker:
    """斷路器模式，防止目標伺服器崩潰時引發連鎖雪崩"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED" # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()

    def record_failure(self):
        with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"⚠️ Circuit Breaker OPEN! 暫停請求 {self.recovery_timeout} 秒。")

    def record_success(self):
        with self.lock:
            self.failures = 0
            self.state = "CLOSED"

    def check(self):
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("🔄 Circuit Breaker HALF_OPEN, 嘗試恢復連線...")
                    return True
                return False
            return True

# =====================================================================
# 3. 核心 API 客戶端
# =====================================================================

class FinMindClient:
    """
    FinMind 企業級量化客戶端 (Singleton)
    整合快取、斷路器、速率限制與非同步支援。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FinMindClient, cls).__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        self.api_url = "https://api.finmindtrade.com/api/v4/data"
        self.api_token = os.environ.get("FINMIND_TOKEN", "")
        
        # 組件初始化
        self.stats = RequestStats()
        self.circuit_breaker = CircuitBreaker()
        self.cache = SQLiteCache()
        
        # 依照有無 Token 設定速率
        if self.api_token:
            self.token_bucket = TokenBucket(capacity=600, fill_rate=600/3600)
            logger.info("🟢 FinMindClient 初始化 (附帶 API Token，高配額模式)")
        else:
            self.token_bucket = TokenBucket(capacity=300, fill_rate=300/3600)
            logger.info("🟡 FinMindClient 初始化 (無 API Token，一般配額模式)")

    def _prepare_params(self, dataset: str, data_id: str, start_date: str, end_date: str) -> dict:
        params = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start_date,
            "end_date": end_date
        }
        if self.api_token:
            params["token"] = self.api_token
        return params

    def get_data(self, dataset: str, data_id: str, start_date: str, end_date: str) -> List[Dict]:
        """[同步版本] 獲取資料，向下相容於現有 scripts"""
        # 1. 檢查快取
        cached_data = self.cache.get(dataset, data_id, start_date, end_date)
        if cached_data is not None:
            self.stats.record_cache_hit()
            logger.debug(f"⚡ Cache Hit: {dataset} {data_id} ({start_date}~{end_date})")
            return cached_data

        # 2. 檢查斷路器
        if not self.circuit_breaker.check():
            raise Exception("Circuit Breaker is OPEN. API 請求暫時阻斷。")

        # 3. 消耗 Token (速率限制)
        self.token_bucket.consume(1)

        # 4. 實際請求
        params = self._prepare_params(dataset, data_id, start_date, end_date)
        try:
            resp = requests.get(self.api_url, params=params, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            if result.get("msg") != "success":
                raise Exception(f"API 回應異常: {result.get('msg')}")

            data = result.get("data", [])
            self.stats.record_success()
            self.circuit_breaker.record_success()
            
            # 5. 寫入快取
            if data:
                self.cache.set(dataset, data_id, start_date, end_date, data)
                
            return data

        except Exception as e:
            self.stats.record_failure()
            self.circuit_breaker.record_failure()
            logger.error(f"❌ FinMind API 請求失敗 ({dataset}-{data_id}): {e}")
            raise

    async def async_get_data(self, dataset: str, data_id: str, start_date: str, end_date: str) -> List[Dict]:
        """[非同步版本] 獲取資料，適合未來的高頻併發管線"""
        if not HAS_AIOHTTP:
            raise RuntimeError("未安裝 aiohttp。請執行 `pip install aiohttp` 以啟用非同步抓取。")

        # 1. 檢查快取 (SQLite 操作使用同步，因輕量不致嚴重阻塞)
        cached_data = self.cache.get(dataset, data_id, start_date, end_date)
        if cached_data is not None:
            self.stats.record_cache_hit()
            return cached_data

        if not self.circuit_breaker.check():
            raise Exception("Circuit Breaker is OPEN. API 請求暫時阻斷。")

        # 2. 非同步消耗 Token
        await self.token_bucket.async_consume(1)

        params = self._prepare_params(dataset, data_id, start_date, end_date)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params, timeout=10) as resp:
                    resp.raise_for_status()
                    result = await resp.json()

                    if result.get("msg") != "success":
                        raise Exception(f"API 回應異常: {result.get('msg')}")

                    data = result.get("data", [])
                    self.stats.record_success()
                    self.circuit_breaker.record_success()
                    
                    if data:
                        self.cache.set(dataset, data_id, start_date, end_date, data)
                        
                    return data
                    
        except Exception as e:
            self.stats.record_failure()
            self.circuit_breaker.record_failure()
            logger.error(f"❌ FinMind Async API 請求失敗 ({dataset}-{data_id}): {e}")
            raise

def get_request_stats():
    """提供全域函式供外部（如 backfill_from_gaps.py）調用統計結果"""
    client = FinMindClient()
    return client.stats