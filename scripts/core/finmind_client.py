"""
core/finmind_client.py — 統一的 FinMind API 客戶端 v3.0（穩健觀測版）
=====================================================================
v3.0 重大改進（搭配 db_utils v3.0 的逐支逐日 commit）：
  ★ RequestStats          : 請求成功/失敗 / 耗時統計（per dataset），fetcher 可在收尾印出。
  ★ CircuitBreaker        : 同一 dataset 連續失敗 N 次後，後續請求直接快速失敗，避免拖垮整體
                             pipeline；冷卻時間到後自動半開重試。
  ★ FetcherInterrupted    : 配額耗盡時可選擇拋出（給呼叫端決定 partial commit），不必固定等待。
  ★ wait_until_quota_reset(): 取代 wait_until_next_hour，統一名稱並回傳實際等待秒數。
  ★ finmind_get() 新增 raise_on_quota 參數；預設行為仍與 v2 相同（自動等到下一整點）。
  ★ check_api_quota() 加入快取，60 秒內不重複呼叫，避免大量 fetcher 並發時把 user_info 灌爆。
  ★ 結果筆數驗證：API 回傳的 list 為 None / 非 list 時記為失敗。
  ★ 所有錯誤路徑都更新 RequestStats，failures 即時可觀察。
  ★ 向後相容：finmind_get() / finmind_get_async() / wait_until_next_hour() 介面不變。

v2.0 既有：
  - TokenBucketRateLimiter：智慧型動態速率限制
  - finmind_get_async()：基於 aiohttp 的非同步請求
  - 指數退避 + 抖動因子重試
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── 自我修復 sys.path（讓本檔可從任何位置直接執行 / import）──
_THIS_DIR = Path(__file__).resolve().parent      # scripts/core
_SCRIPTS_DIR = _THIS_DIR.parent                  # scripts
for _p in (_SCRIPTS_DIR, _THIS_DIR):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import requests

from config import FINMIND_TOKEN

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"

# FinMind 官方配額：每小時 600 次（含授權 Token）
FINMIND_HOURLY_LIMIT = 6000

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 自訂例外
# ─────────────────────────────────────────────
class BatchNotSupportedError(Exception):
    """批次請求（不帶 data_id）被 FinMind 拒絕。可 fallback 為逐支模式。"""


class FetcherInterrupted(Exception):
    """配額耗盡且呼叫端要求快速失敗（raise_on_quota=True）。允許上層做 partial commit。"""


class CircuitOpenError(Exception):
    """斷路器開啟期間，後續請求直接快速失敗。"""


# ─────────────────────────────────────────────
# 請求統計（per dataset）
# ─────────────────────────────────────────────
@dataclass
class _DatasetStat:
    success: int = 0
    failure: int = 0
    quota_wait: int = 0
    total_seconds: float = 0.0
    last_error: str | None = None


class RequestStats:
    """
    記錄各 dataset 的成功 / 失敗 / 耗時，fetcher 結束時可呼叫 summary() 列印報表。
    全執行緒共享同一 instance，使用 lock 保證一致性。
    """

    def __init__(self):
        self._stats: dict[str, _DatasetStat] = defaultdict(_DatasetStat)
        self._lock = threading.Lock()

    def record_success(self, dataset: str, elapsed: float) -> None:
        with self._lock:
            s = self._stats[dataset]
            s.success += 1
            s.total_seconds += elapsed

    def record_failure(self, dataset: str, error: str) -> None:
        with self._lock:
            s = self._stats[dataset]
            s.failure += 1
            s.last_error = error[:200]

    def record_quota_wait(self, dataset: str) -> None:
        with self._lock:
            self._stats[dataset].quota_wait += 1

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                k: {
                    "success": v.success,
                    "failure": v.failure,
                    "quota_wait": v.quota_wait,
                    "avg_seconds": (v.total_seconds / v.success) if v.success else 0.0,
                    "last_error": v.last_error,
                }
                for k, v in self._stats.items()
            }

    def summary(self) -> None:
        snap = self.snapshot()
        if not snap:
            return
        logger.info("=== FinMind 請求統計 ===")
        logger.info(f"{'dataset':<48} {'ok':>6} {'fail':>5} {'quota_wait':>10} {'avg(s)':>7}")
        for ds, s in sorted(snap.items()):
            logger.info(
                f"{ds:<48} {s['success']:>6} {s['failure']:>5} "
                f"{s['quota_wait']:>10} {s['avg_seconds']:>7.2f}"
            )
            if s["last_error"]:
                logger.info(f"    last_error: {s['last_error']}")

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()


_global_stats = RequestStats()


def get_request_stats() -> RequestStats:
    """取得全域 RequestStats 實例。"""
    return _global_stats


# ─────────────────────────────────────────────
# 斷路器（per dataset）
# ─────────────────────────────────────────────
@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    open_until: float = 0.0   # monotonic 秒數
    half_open: bool = False


class CircuitBreaker:
    """
    Per-dataset 斷路器，連續失敗 N 次後進入 OPEN 狀態，後續請求直接快速失敗。
    冷卻時間（cool_down_sec）到後進入 HALF_OPEN，允許 1 次試探請求；成功 → CLOSED，
    失敗 → 重新 OPEN。

    Parameters
    ----------
    failure_threshold : int   連續失敗幾次後開斷路器（預設 5）
    cool_down_sec     : float OPEN 後冷卻時間（預設 120）
    """

    def __init__(self, failure_threshold: int = 5, cool_down_sec: float = 120.0):
        self.failure_threshold = failure_threshold
        self.cool_down_sec = cool_down_sec
        self._state: dict[str, _BreakerState] = defaultdict(_BreakerState)
        self._lock = threading.Lock()

    def check(self, dataset: str) -> None:
        """請求前呼叫；若斷路器 OPEN 拋 CircuitOpenError。"""
        now = time.monotonic()
        with self._lock:
            st = self._state[dataset]
            if st.open_until > now:
                raise CircuitOpenError(
                    f"[{dataset}] circuit open，將於 {st.open_until - now:.0f}s 後重試"
                )
            if st.open_until and st.open_until <= now and not st.half_open:
                # 進入 HALF_OPEN
                st.half_open = True
                logger.info(f"[{dataset}] 斷路器進入 HALF_OPEN，允許試探請求")

    def record_success(self, dataset: str) -> None:
        with self._lock:
            st = self._state[dataset]
            if st.half_open or st.consecutive_failures > 0:
                logger.info(f"[{dataset}] 斷路器已 CLOSE")
            st.consecutive_failures = 0
            st.open_until = 0.0
            st.half_open = False

    def record_failure(self, dataset: str) -> None:
        with self._lock:
            st = self._state[dataset]
            st.consecutive_failures += 1
            if st.consecutive_failures >= self.failure_threshold:
                st.open_until = time.monotonic() + self.cool_down_sec
                st.half_open = False
                logger.warning(
                    f"[{dataset}] 連續失敗 {st.consecutive_failures} 次，"
                    f"斷路器 OPEN，冷卻 {self.cool_down_sec:.0f}s"
                )

    def reset(self, dataset: str | None = None) -> None:
        with self._lock:
            if dataset:
                self._state.pop(dataset, None)
            else:
                self._state.clear()


_global_breaker = CircuitBreaker()


def get_circuit_breaker() -> CircuitBreaker:
    return _global_breaker


# ─────────────────────────────────────────────
# Token Bucket 速率限制器（執行緒安全）
# ─────────────────────────────────────────────
class TokenBucketRateLimiter:
    """智慧型令牌桶速率限制器，取代固定 time.sleep。"""

    def __init__(
        self,
        rate_limit: int = FINMIND_HOURLY_LIMIT,
        period: float = 3600.0,
        min_interval: float = 0.5,
    ):
        self._capacity = rate_limit
        self._tokens = float(rate_limit)
        self._refill_rate = rate_limit / period
        self._min_interval = min_interval
        self._last_refill = time.monotonic()
        self._last_request = 0.0
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now

    def acquire(self, jitter_range: float = 0.3) -> None:
        wait_base = 1.0
        while True:
            with self._lock:
                self._refill()
                since_last = time.monotonic() - self._last_request
                if since_last < self._min_interval:
                    time.sleep(self._min_interval - since_last)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._last_request = time.monotonic()
                    return
            jitter = random.uniform(0, jitter_range)
            wait = wait_base + jitter
            logger.info(f"[RateLimiter] 令牌不足(剩餘 {self._tokens:.2f})，等待 {wait:.2f}s...")
            time.sleep(wait)
            wait_base = min(wait_base * 2, 30.0)

    async def acquire_async(self, jitter_range: float = 0.3) -> None:
        wait_base = 1.0
        while True:
            with self._lock:
                self._refill()
                since_last = time.monotonic() - self._last_request
                if since_last < self._min_interval:
                    await asyncio.sleep(self._min_interval - since_last)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._last_request = time.monotonic()
                    return
            jitter = random.uniform(0, jitter_range)
            wait = wait_base + jitter
            await asyncio.sleep(wait)
            wait_base = min(wait_base * 2, 30.0)


_rate_limiter = TokenBucketRateLimiter()


def get_rate_limiter() -> TokenBucketRateLimiter:
    return _rate_limiter


# ─────────────────────────────────────────────
# 配額重置等待
# ─────────────────────────────────────────────
def wait_until_quota_reset(buffer_sec: float = 65.0) -> float:
    """
    配額耗盡時呼叫，等到下一整點 + buffer_sec 秒後返回。
    重置令牌桶，使等待後可立即全速請求。
    回傳實際等待秒數。
    """
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_sec = (next_hour - now).total_seconds() + buffer_sec
    logger.warning(
        f"API 配額達上限（402），等待 {wait_sec:.0f} 秒至 "
        f"{next_hour.strftime('%H:%M')} 後恢復…"
    )
    time.sleep(wait_sec)
    with _rate_limiter._lock:
        _rate_limiter._tokens = float(_rate_limiter._capacity)
        _rate_limiter._last_refill = time.monotonic()
    logger.info("等待結束，恢復請求。")
    return wait_sec


def wait_until_next_hour() -> None:
    """v2.0 介面別名（保留向後相容）。"""
    wait_until_quota_reset()


# ─────────────────────────────────────────────
# 同步核心請求函式（v3.0）
# ─────────────────────────────────────────────
def finmind_get(
    dataset: str,
    params: dict,
    delay: float = 1.2,
    max_retries: int = 3,
    raise_on_batch_400: bool = False,
    use_rate_limiter: bool = True,
    raise_on_quota: bool = False,
    raise_on_error: bool = False,
) -> list:
    """
    統一的 FinMind API 同步請求函式（v3）。

    v3 改進：
      · 整合 RequestStats（成功/失敗/耗時統計）
      · 整合 CircuitBreaker（同一 dataset 連續失敗 N 次後快速失敗）
      · raise_on_quota=True 時，配額耗盡會拋 FetcherInterrupted（呼叫端可
        partial commit 後再決定是否等待重啟）；預設仍維持 v2 行為（等待下一整點）
      · raise_on_error=True 時，API 400/403/422 或逾時將拋出異常，
        供呼叫端的 FailureLogger 擷取；預設為 False（維持舊版回傳空列表）。

    Returns
    -------
    list  成功時回傳 data 列表；失敗時回傳空列表 []
    """
    # ── 斷路器預檢 ──
    try:
        _global_breaker.check(dataset)
    except CircuitOpenError as e:
        logger.warning(str(e))
        _global_stats.record_failure(dataset, "circuit_open")
        return []

    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}
    is_batch = "data_id" not in params
    t0 = time.monotonic()

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                FINMIND_API_URL,
                headers=headers,
                params=req_params,
                timeout=(60, 600),
            )

            # ── 402：配額耗盡 ──
            if resp.status_code == 402:
                _global_stats.record_quota_wait(dataset)
                if raise_on_quota:
                    raise FetcherInterrupted(f"[{dataset}] 配額耗盡（HTTP 402）")
                wait_until_quota_reset()
                attempt = 1
                continue

            # ── 403：權限不足 ──
            if resp.status_code == 403:
                msg = f"🚫 [Perm] 403 Forbidden: {dataset}. 自動跳過。"
                logger.warning(msg)
                _global_stats.record_failure(dataset, "403")
                if raise_on_error:
                    raise PermissionError(msg)
                return []

            # ── 400：Bad Request ──
            if resp.status_code == 400:
                if raise_on_batch_400 and is_batch:
                    raise BatchNotSupportedError(
                        f"[{dataset}] 批次請求被拒絕 (HTTP 400)"
                    )
                msg = f"⚠️ [Skip] 400 Bad Request: {dataset}. 已跳過（資料缺口或格式不支援）。"
                logger.warning(msg)
                _global_stats.record_failure(dataset, "400")
                if raise_on_error:
                    raise ValueError(msg)
                return []

            # ── 401：Token 無效 ──
            if resp.status_code == 401:
                msg = f"[{dataset}] HTTP 401 Unauthorized: Token 過期或無效。"
                logger.error(msg)
                _global_stats.record_failure(dataset, "401")
                if raise_on_error:
                    raise ConnectionRefusedError(msg)
                return []

            resp.raise_for_status()
            payload = resp.json()

            # ── payload status 402 ──
            if payload.get("status") == 402:
                _global_stats.record_quota_wait(dataset)
                if raise_on_quota:
                    raise FetcherInterrupted(f"[{dataset}] 配額耗盡（payload 402）")
                wait_until_quota_reset()
                attempt = 1
                continue

            if payload.get("status") != 200:
                if (
                    raise_on_batch_400
                    and is_batch
                    and payload.get("status") == 400
                ):
                    raise BatchNotSupportedError(
                        f"[{dataset}] 批次請求被拒絕（status=400），"
                        f"msg={payload.get('msg')}"
                    )
                msg = f"[{dataset}] status={payload.get('status')}, msg={payload.get('msg')}"
                logger.warning(msg)
                _global_stats.record_failure(
                    dataset, f"status={payload.get('status')}: {payload.get('msg')}"
                )
                if raise_on_error:
                    raise ValueError(msg)
                return []

            # ── 成功：速率控制 + 統計 ──
            data = payload.get("data", [])
            if data is None:
                data = []
            if not isinstance(data, list):
                logger.warning(f"[{dataset}] payload['data'] 非 list（{type(data).__name__}），回空")
                _global_stats.record_failure(dataset, f"non_list_data:{type(data).__name__}")
                return []

            if use_rate_limiter:
                _rate_limiter.acquire()
            else:
                time.sleep(delay)

            _global_stats.record_success(dataset, time.monotonic() - t0)
            _global_breaker.record_success(dataset)
            return data

        except (BatchNotSupportedError, FetcherInterrupted):
            raise
        except requests.exceptions.Timeout:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次逾時")
        except requests.exceptions.ConnectionError:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次連線失敗")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 402:
                _global_stats.record_quota_wait(dataset)
                if raise_on_quota:
                    raise FetcherInterrupted(f"[{dataset}] 配額耗盡（HTTPError 402）")
                wait_until_quota_reset()
                attempt = 1
                continue
            if code == 422:
                msg = f"⚠️ [Skip] 422 Unprocessable Entity: {dataset}. 標的可能不支援，已跳過。"
                logger.warning(msg)
                _global_stats.record_failure(dataset, "422")
                if raise_on_error:
                    raise ValueError(msg)
                return []
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次 HTTP {code} 錯誤：{e}")
            if raise_on_error and attempt == max_retries:
                raise
        except Exception as exc:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次異常：{exc}")

        if attempt < max_retries:
            backoff = delay * (2 ** attempt) + random.uniform(0, 1.0)
            logger.info(f"[{dataset}] {backoff:.1f}s 後重試…")
            time.sleep(backoff)
        else:
            msg = f"[{dataset}] 已重試 {max_retries} 次，放棄。"
            logger.error(msg)
            _global_stats.record_failure(dataset, f"max_retries={max_retries}")
            _global_breaker.record_failure(dataset)
            if raise_on_error:
                raise RuntimeError(msg)
            return []

    return []


# ─────────────────────────────────────────────
# 非同步請求函式（aiohttp）
# ─────────────────────────────────────────────
async def finmind_get_async(
    dataset: str,
    params: dict,
    session,  # aiohttp.ClientSession
    max_retries: int = 3,
    raise_on_batch_400: bool = False,
    raise_on_quota: bool = False,
    raise_on_error: bool = False,
) -> list:
    """
    FinMind API 非同步請求函式（v3，整合 RequestStats / CircuitBreaker）。
    """
    try:
        _global_breaker.check(dataset)
    except CircuitOpenError as e:
        logger.warning(str(e))
        _global_stats.record_failure(dataset, "circuit_open")
        return []

    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}
    is_batch = "data_id" not in params
    t0 = time.monotonic()

    for attempt in range(1, max_retries + 1):
        try:
            await _rate_limiter.acquire_async()

            async with session.get(
                FINMIND_API_URL,
                headers=headers,
                params=req_params,
                timeout=__import__("aiohttp").ClientTimeout(connect=60, total=660, sock_read=600),
            ) as resp:
                if resp.status == 402:
                    _global_stats.record_quota_wait(dataset)
                    if raise_on_quota:
                        raise FetcherInterrupted(f"[async][{dataset}] 配額耗盡 402")
                    now = datetime.now()
                    next_hour = (now + timedelta(hours=1)).replace(
                        minute=0, second=0, microsecond=0
                    )
                    wait_sec = (next_hour - now).total_seconds() + 65
                    logger.warning(
                        f"[async][{dataset}] 402 配額耗盡，等待 {wait_sec:.0f}s"
                    )
                    await asyncio.sleep(wait_sec)
                    with _rate_limiter._lock:
                        _rate_limiter._tokens = float(_rate_limiter._capacity)
                    attempt = 1
                    continue

                if resp.status == 403:
                    msg = f"🚫 [async][Perm] 403 Forbidden: {dataset}. 自動跳過。"
                    logger.warning(msg)
                    _global_stats.record_failure(dataset, "async_403")
                    if raise_on_error: raise PermissionError(msg)
                    return []
                if resp.status == 401:
                    msg = f"[async][{dataset}] HTTP 401 Unauthorized: Token 無效。"
                    logger.error(msg)
                    _global_stats.record_failure(dataset, "async_401")
                    if raise_on_error: raise ConnectionRefusedError(msg)
                    return []
                if resp.status == 400:
                    if raise_on_batch_400 and is_batch:
                        raise BatchNotSupportedError(
                            f"[async][{dataset}] 批次請求被拒絕 (HTTP 400)"
                        )
                    msg = f"⚠️ [async][Skip] 400 Bad Request: {dataset}. 已跳過。"
                    logger.warning(msg)
                    _global_stats.record_failure(dataset, "async_400")
                    if raise_on_error: raise ValueError(msg)
                    return []

                resp.raise_for_status()
                payload = await resp.json()

                if payload.get("status") == 402:
                    _global_stats.record_quota_wait(dataset)
                    if raise_on_quota:
                        raise FetcherInterrupted(f"[async][{dataset}] 配額耗盡 payload 402")
                    now = datetime.now()
                    next_hour = (now + timedelta(hours=1)).replace(
                        minute=0, second=0, microsecond=0
                    )
                    wait_sec = (next_hour - now).total_seconds() + 65
                    await asyncio.sleep(wait_sec)
                    attempt = 1
                    continue

                if payload.get("status") != 200:
                    if (
                        raise_on_batch_400
                        and is_batch
                        and payload.get("status") == 400
                    ):
                        raise BatchNotSupportedError(
                            f"[async][{dataset}] 批次請求被拒絕 status=400"
                        )
                    msg = f"[async][{dataset}] status={payload.get('status')}, msg={payload.get('msg')}"
                    logger.warning(msg)
                    _global_stats.record_failure(
                        dataset, f"async_status={payload.get('status')}"
                    )
                    if raise_on_error: raise ValueError(msg)
                    return []

                data = payload.get("data", [])
                if data is None:
                    data = []
                if not isinstance(data, list):
                    _global_stats.record_failure(dataset, f"async_non_list:{type(data).__name__}")
                    return []
                _global_stats.record_success(dataset, time.monotonic() - t0)
                _global_breaker.record_success(dataset)
                return data

        except (BatchNotSupportedError, FetcherInterrupted):
            raise
        except Exception as exc:
            if attempt < max_retries:
                backoff = 2.0 ** attempt + random.uniform(0, 1.0)
                logger.warning(
                    f"[async][{dataset}] 第 {attempt}/{max_retries} 次失敗：{exc}，"
                    f"{backoff:.1f}s 後重試"
                )
                await asyncio.sleep(backoff)
            else:
                msg = f"[async][{dataset}] 已重試 {max_retries} 次，放棄：{exc}"
                logger.error(msg)
                _global_stats.record_failure(dataset, f"async_max_retries:{exc}")
                _global_breaker.record_failure(dataset)
                if raise_on_error: raise RuntimeError(msg)
                return []

    return []


# ─────────────────────────────────────────────
# API 配額預查（含快取）
# ─────────────────────────────────────────────
_quota_cache_lock = threading.Lock()
_quota_cache: dict = {"ts": 0.0, "used": -1, "limit": FINMIND_HOURLY_LIMIT}


def check_api_quota(cache_ttl: float = 60.0) -> tuple[int, int]:
    """
    查詢當前 FinMind API 使用量與上限。
    回傳 (used, limit)。
    含 cache_ttl 秒快取，避免並行 fetcher 各自打 user_info。
    若查詢失敗，回傳 (-1, 600)（不阻斷執行）。
    """
    now = time.monotonic()
    with _quota_cache_lock:
        if now - _quota_cache["ts"] < cache_ttl and _quota_cache["used"] >= 0:
            return _quota_cache["used"], _quota_cache["limit"]

    try:
        resp = requests.get(
            FINMIND_USER_INFO_URL,
            headers={"Authorization": f"Bearer {FINMIND_TOKEN}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        used  = int(data.get("user_count", 0))
        limit = int(data.get("api_request_limit", 600))
        with _quota_cache_lock:
            _quota_cache.update({"ts": now, "used": used, "limit": limit})
        return used, limit
    except Exception as e:
        logger.warning(f"無法查詢 API 配額：{e}")
        return -1, FINMIND_HOURLY_LIMIT


__all__ = [
    "FINMIND_API_URL", "FINMIND_USER_INFO_URL", "FINMIND_HOURLY_LIMIT",
    "BatchNotSupportedError", "FetcherInterrupted", "CircuitOpenError",
    "TokenBucketRateLimiter", "get_rate_limiter",
    "RequestStats", "get_request_stats",
    "CircuitBreaker", "get_circuit_breaker",
    "wait_until_quota_reset", "wait_until_next_hour",
    "finmind_get", "finmind_get_async",
    "check_api_quota",
]