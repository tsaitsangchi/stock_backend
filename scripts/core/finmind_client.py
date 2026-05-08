"""
core/finmind_client.py — FinMind API Client v3.1 (Quantum Finance v5.1 Edition)
=============================================================================
整合 Token Bucket 速率限制、斷路器、以及全非同步 I/O 支援的高頻量化網路客戶端。
依據 Quantum Finance v5.1 物理資訊系統範例重構。
"""

import os
import sys
import time
import random
import asyncio
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

import requests
import aiohttp

# ── 設定區域 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import FINMIND_TOKEN

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.finmindtrade.com/api/v4/data"
USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"

# ─────────────────────────────────────────────
# 自訂例外
# ─────────────────────────────────────────────
class FetcherInterrupted(Exception):
    """當 API 配額耗盡且設定為不自動等待時拋出。"""
    pass

class CircuitOpenError(Exception):
    """當斷路器處於開啟狀態時拋出，防止對已失效端點進行無效連線。"""
    pass

class BatchNotSupportedError(Exception):
    """當資料集不支援批次（data_id 清單）請求時拋出。"""
    pass

# ─────────────────────────────────────────────
# 1. 請求統計追蹤器 (RequestStats)
# ─────────────────────────────────────────────
class RequestStats:
    """執行緒安全的 API 請求狀態追蹤器，用於量化管線監控。"""
    def __init__(self):
        self._lock = threading.Lock()
        self.stats = {}

    def record(self, dataset: str, success: bool, latency: float, error_msg: str = ""):
        with self._lock:
            if dataset not in self.stats:
                self.stats[dataset] = {"success": 0, "fail": 0, "total_latency": 0.0, "last_error": ""}
            
            if success:
                self.stats[dataset]["success"] += 1
            else:
                self.stats[dataset]["fail"] += 1
                self.stats[dataset]["last_error"] = error_msg
            self.stats[dataset]["total_latency"] += latency

    def summary(self):
        with self._lock:
            if not self.stats:
                return
            logger.info("=" * 60)
            logger.info("    FinMind API 請求統計摘要")
            logger.info("=" * 60)
            for ds, data in self.stats.items():
                total = data["success"] + data["fail"]
                avg_lat = data["total_latency"] / total if total > 0 else 0
                logger.info(f" [{ds:20s}] 成功: {data['success']:4d} | 失敗: {data['fail']:2d} | 平均耗時: {avg_lat:6.3f}s | 最後錯誤: {data['last_error']}")
            logger.info("=" * 60)

global_stats = RequestStats()

def get_request_stats():
    return global_stats

# ─────────────────────────────────────────────
# 2. 斷路器機制 (CircuitBreaker)
# ─────────────────────────────────────────────
class CircuitBreaker:
    """資料集級別的斷路器機制，防止單一失效端點引發整體系統延遲。"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = {}
        self.open_until = {}
        self._lock = threading.Lock()

    def check(self, dataset: str):
        with self._lock:
            if dataset in self.open_until:
                if time.time() < self.open_until[dataset]:
                    raise CircuitOpenError(f"Circuit for {dataset} is OPEN until {datetime.fromtimestamp(self.open_until[dataset])}")
                else:
                    # 進入 HALF_OPEN 狀態，允許單次探測
                    del self.open_until[dataset]
                    return True
        return True

    def record_success(self, dataset: str):
        with self._lock:
            self.failures[dataset] = 0
            if dataset in self.open_until:
                del self.open_until[dataset]

    def record_failure(self, dataset: str):
        with self._lock:
            count = self.failures.get(dataset, 0) + 1
            self.failures[dataset] = count
            if count >= self.failure_threshold:
                self.open_until[dataset] = time.time() + self.recovery_timeout
                logger.warning(f"CircuitBreaker OPENED for {dataset} due to {count} consecutive failures.")

global_circuit_breaker = CircuitBreaker()

# ─────────────────────────────────────────────
# 3. 權杖桶速率限制器 (TokenBucketRateLimiter)
# ─────────────────────────────────────────────
class TokenBucketRateLimiter:
    """實作動態速率限制與抖動退避重試的權杖桶演算法。"""
    def __init__(self, capacity: int = 600, refill_rate: float = 0.167):
        # refill_rate 0.167 代表每秒補充 0.167 個權杖 (一小時約 600 個)
        # 若有 Token，FinMind 限制通常放寬至每小時 6000 次
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self):
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def acquire_async(self):
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            # 加入抖動因子防止連線風暴
            jitter = random.uniform(0, 0.3)
            await asyncio.sleep(0.5 + jitter)

    def full_reset(self):
        with self._lock:
            self.tokens = self.capacity
            self.last_refill = time.time()

# 根據是否有 Token 自動調整頻率上限
_limit = 6000 if FINMIND_TOKEN else 600
_rate = _limit / 3600.0
global_rate_limiter = TokenBucketRateLimiter(capacity=_limit, refill_rate=_rate)

# ─────────────────────────────────────────────
# 4. 配額與資訊工具
# ─────────────────────────────────────────────
_QUOTA_CACHE = {"used": -1, "limit": -1, "ts": 0}

def check_api_quota(force: bool = False) -> tuple[int, int]:
    """
    檢查目前帳號 API 配額，具備 60 秒快取。
    """
    now = time.time()
    if not force and (now - _QUOTA_CACHE["ts"]) < 60:
        return _QUOTA_CACHE["used"], _QUOTA_CACHE["limit"]

    try:
        res = requests.get(USER_INFO_URL, params={"token": FINMIND_TOKEN}, timeout=10)
        res.raise_for_status()
        data = res.json().get("data", {})
        used = data.get("api_request", -1)
        limit = data.get("api_request_limit", -1)
        _QUOTA_CACHE.update({"used": used, "limit": limit, "ts": now})
        return used, limit
    except Exception as e:
        logger.warning(f"無法檢查 API 配額：{e}")
        return -1, -1

def wait_until_quota_reset():
    """精確計算至下一個小時重置時間的等待邏輯，附帶 65 秒緩衝期。"""
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_seconds = (next_hour - now).total_seconds() + 65.0
    logger.info(f"API 配額耗盡。正在等待至下一整點重置（約 {wait_seconds:.0f} 秒）...")
    time.sleep(wait_seconds)
    global_rate_limiter.full_reset()
    return wait_seconds

def wait_until_next_hour():
    """向後相容介面。"""
    return wait_until_quota_reset()

# ─────────────────────────────────────────────
# 5. 核心抓取函式 (Sync & Async)
# ─────────────────────────────────────────────
def finmind_get(dataset: str, params: Dict[str, Any], max_retries: int = 3, 
                use_rate_limiter: bool = True, raise_on_quota: bool = False, 
                raise_on_error: bool = False, delay: float = 0.0,
                raise_on_batch_400: bool = False) -> List:
    """
    同步抓取函式：整合速率限制、斷路器與重試機制。
    新增 raise_on_batch_400：當 API 因不支援批次（data_id）而報錯時拋出專屬例外。
    """
    if use_rate_limiter:
        while not global_rate_limiter.acquire():
            time.sleep(0.5 + random.uniform(0, 0.3))
    
    # 手動延遲
    if delay > 0:
        time.sleep(delay)
            
    try:
        global_circuit_breaker.check(dataset)
    except CircuitOpenError as e:
        logger.error(f"斷路器開啟中，跳過 {dataset}: {e}")
        if raise_on_error: raise
        return []
    
    merged_params = params.copy()
    if FINMIND_TOKEN and "token" not in merged_params:
        merged_params["token"] = FINMIND_TOKEN
    merged_params["dataset"] = dataset
    
    backoff = 1.0
    for attempt in range(max_retries):
        start_time = time.time()
        try:
            response = requests.get(API_BASE_URL, params=merged_params, timeout=20)
            latency = time.time() - start_time
            
            if response.status_code == 402:
                if raise_on_quota:
                    raise FetcherInterrupted("FinMind API 配額耗盡 (402)。")
                wait_until_quota_reset()
                continue
            
            # 處理不支援批次的情況 (400 Bad Request)
            if response.status_code == 400 and raise_on_batch_400:
                msg = response.json().get("msg", "").lower()
                if "parameter" in msg or "data_id" in msg:
                    raise BatchNotSupportedError(f"資料集 {dataset} 不支援批次請求")
                
            response.raise_for_status()
            data = response.json()
            
            res_data = data.get("data")
            if res_data is None or not isinstance(res_data, list):
                # 檢查是否為 batch 錯誤訊息
                msg = data.get("msg", "").lower()
                if raise_on_batch_400 and ("parameter" in msg or "data_id" in msg):
                    raise BatchNotSupportedError(f"資料集 {dataset} 格式異常，疑似不支援批次: {msg}")
                raise ValueError(f"API 回傳格式異常 (dataset={dataset}): {msg}")
                
            global_circuit_breaker.record_success(dataset)
            global_stats.record(dataset, True, latency)
            return res_data
            
        except BatchNotSupportedError:
            # 批次錯誤不重試，直接向上拋出供切換模式
            raise
        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e)
            global_circuit_breaker.record_failure(dataset)
            global_stats.record(dataset, False, latency, error_msg)
            
            if attempt == max_retries - 1:
                logger.error(f"抓取 {dataset} 失敗，已達最大重試次數 ({max_retries}): {error_msg}")
                if raise_on_error: raise
                return []
                
            jitter = random.uniform(0, 1.0)
            time.sleep(backoff + jitter)
            backoff *= 2

    return []

async def finmind_get_async(session: aiohttp.ClientSession, dataset: str, params: Dict[str, Any]) -> List:
    """
    非同步抓取函式：為高吞吐量管線設計，與全域速率限制器協同運作。
    """
    await global_rate_limiter.acquire_async()
    
    try:
        global_circuit_breaker.check(dataset)
    except CircuitOpenError:
        return []

    merged_params = params.copy()
    if FINMIND_TOKEN and "token" not in merged_params:
        merged_params["token"] = FINMIND_TOKEN
    merged_params["dataset"] = dataset
    
    start_time = time.time()
    try:
        async with session.get(API_BASE_URL, params=merged_params, timeout=20) as response:
            latency = time.time() - start_time
            if response.status == 402:
                logger.warning(f"非同步 Worker 遭遇 402 配額耗盡 (dataset={dataset})。")
                return []
            
            response.raise_for_status()
            data = await response.json()
            
            res_data = data.get("data")
            if res_data is None or not isinstance(res_data, list):
                raise ValueError(f"非同步 API 回傳格式異常: {data.get('msg', 'no msg')}")
                
            global_circuit_breaker.record_success(dataset)
            global_stats.record(dataset, True, latency)
            return res_data
            
    except Exception as e:
        latency = time.time() - start_time
        global_circuit_breaker.record_failure(dataset)
        global_stats.record(dataset, False, latency, str(e))
        return []