"""
core/finmind_client.py — 統一的 FinMind API 客戶端 v2.0
=====================================================
v2.0 新增（優化報告建議）：
  - TokenBucketRateLimiter：智慧型動態速率限制，取代固定 time.sleep
    · 根據 FinMind 官方配額（每小時 600 次）動態管理令牌
    · 含抖動因子（Jitter）指數退避，防止並行任務同時甦醒造成連線風暴
  - finmind_get_async()：基於 aiohttp + asyncio 的非同步請求函式
    · 單一執行緒協程並行，突破同步阻塞瓶頸
    · 與 TokenBucketRateLimiter 整合，確保非同步模式下速率合規
  - 向後相容：finmind_get() 同步介面保持不變

原有特性：
  - 指數退避重試（delay × 2^n），最多 max_retries 次
  - 402 配額耗盡：無限等待至下一整點後重試，不計入重試次數
  - 400 Bad Request：直接返回空列表（參數錯誤，無需重試）
  - timeout=(15, 120)：connect 15s / read 120s
  - check_api_quota()：並行抓取前預查剩餘配額
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import threading
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import FINMIND_TOKEN

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"

# FinMind 官方配額：每小時 600 次（含授權 Token）
FINMIND_HOURLY_LIMIT = 600

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 自訂例外
# ─────────────────────────────────────────────
class BatchNotSupportedError(Exception):
    """
    批次請求（不帶 data_id）被 FinMind 拒絕。
    呼叫端可捕捉此例外，自動 fallback 為逐支請求模式。
    """


# ─────────────────────────────────────────────
# Token Bucket 速率限制器（執行緒安全）
# ─────────────────────────────────────────────
class TokenBucketRateLimiter:
    """
    智慧型令牌桶速率限制器，取代固定 time.sleep。

    工作原理：
      · 桶容量 = rate_limit（每 period 秒允許的最大請求數）
      · 令牌以 rate_limit / period 的速率持續補充
      · 每次請求消耗 1 個令牌
      · 桶空時自動等待（含抖動因子，防止並行任務同時甦醒）

    Parameters
    ----------
    rate_limit : 時間窗口內允許的最大請求數（預設 600）
    period     : 時間窗口秒數（預設 3600 = 1 小時）
    min_interval : 兩次請求間的最小間隔秒數（防止瞬間爆發，預設 0.5）
    """

    def __init__(
        self,
        rate_limit: int = FINMIND_HOURLY_LIMIT,
        period: float = 3600.0,
        min_interval: float = 0.5,
    ):
        self._capacity = rate_limit
        self._tokens = float(rate_limit)
        self._refill_rate = rate_limit / period   # 每秒補充令牌數
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
        """
        消耗 1 個令牌。桶空時以帶抖動因子的指數退避等待，
        防止多個並行協程在同一毫秒同時甦醒造成連線風暴。
        """
        wait_base = 1.0
        while True:
            with self._lock:
                self._refill()
                # 確保最小請求間隔
                since_last = time.monotonic() - self._last_request
                if since_last < self._min_interval:
                    time.sleep(self._min_interval - since_last)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._last_request = time.monotonic()
                    return
            # 令牌不足：帶抖動因子等待
            jitter = random.uniform(0, jitter_range)
            wait = wait_base + jitter
            logger.debug(f"[RateLimiter] 令牌不足，等待 {wait:.2f}s")
            time.sleep(wait)
            wait_base = min(wait_base * 2, 30.0)  # 上限 30s

    async def acquire_async(self, jitter_range: float = 0.3) -> None:
        """acquire() 的非同步版本（asyncio 協程）。"""
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


# 全域單例速率限制器（所有模組共用）
_rate_limiter = TokenBucketRateLimiter()


def get_rate_limiter() -> TokenBucketRateLimiter:
    """取得全域 Token Bucket 速率限制器實例。"""
    return _rate_limiter


# ─────────────────────────────────────────────
# 402 配額限制等待
# ─────────────────────────────────────────────
def wait_until_next_hour() -> None:
    """
    API 配額耗盡（HTTP 402）時呼叫。
    FinMind 在整點重置配額，等待至下一整點 + 65 秒緩衝。
    同時重置速率限制器令牌桶，使等待後可立即全速請求。
    """
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_sec = (next_hour - now).total_seconds() + 65
    logger.warning(
        f"API 配額達上限（402），等待 {wait_sec:.0f} 秒至 "
        f"{next_hour.strftime('%H:%M')} 後恢復…"
    )
    time.sleep(wait_sec)
    # 重置令牌桶，配額已重置可全速請求
    with _rate_limiter._lock:
        _rate_limiter._tokens = float(_rate_limiter._capacity)
        _rate_limiter._last_refill = time.monotonic()
    logger.info("等待結束，恢復請求。")


# ─────────────────────────────────────────────
# 同步核心請求函式（v2 — Token Bucket + Jitter）
# ─────────────────────────────────────────────
def finmind_get(
    dataset: str,
    params: dict,
    delay: float = 1.2,
    max_retries: int = 3,
    raise_on_batch_400: bool = False,
    use_rate_limiter: bool = True,
) -> list:
    """
    統一的 FinMind API 同步請求函式（v2）。

    v2 改進：
      · use_rate_limiter=True（預設）：改用 TokenBucketRateLimiter 取代固定 sleep，
        配額充裕時全速請求，配額緊張時動態降速
      · delay 參數保留向後相容，use_rate_limiter=False 時仍使用固定 sleep
      · 指數退避含抖動因子，防止並行連線風暴

    Parameters
    ----------
    dataset            : FinMind dataset 名稱
    params             : 額外參數 dict
    delay              : 固定延遲秒數（use_rate_limiter=False 時生效）
    max_retries        : 非 402 錯誤的最大重試次數
    raise_on_batch_400 : True 時批次請求收到 400 拋出 BatchNotSupportedError
    use_rate_limiter   : True（預設）使用 Token Bucket；False 使用固定 sleep

    Returns
    -------
    list  成功時回傳 data 列表；失敗時回傳空列表 []
    """
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}
    is_batch = "data_id" not in params

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                FINMIND_API_URL,
                headers=headers,
                params=req_params,
                timeout=(15, 120),
            )

            # ── 402：配額耗盡 ──
            if resp.status_code == 402:
                wait_until_next_hour()
                attempt = 1
                continue

            # ── 400：參數錯誤，不重試 ──
            if resp.status_code == 400:
                if raise_on_batch_400 and is_batch:
                    raise BatchNotSupportedError(
                        f"[{dataset}] 批次請求（HTTP 400）被拒絕，可能帳號等級不足"
                    )
                logger.debug(
                    f"[{dataset}] 400 Bad Request，跳過"
                    f"（data_id={params.get('data_id')}，"
                    f"start={params.get('start_date')}）"
                )
                return []

            resp.raise_for_status()
            payload = resp.json()

            if payload.get("status") == 402:
                wait_until_next_hour()
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
                logger.warning(
                    f"[{dataset}] status={payload.get('status')}, "
                    f"msg={payload.get('msg')}"
                )
                return []

            # ── 成功：速率控制 ──
            if use_rate_limiter:
                _rate_limiter.acquire()
            else:
                time.sleep(delay)

            return payload.get("data", [])

        except BatchNotSupportedError:
            raise
        except requests.exceptions.Timeout:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次逾時")
        except requests.exceptions.ConnectionError:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次連線失敗")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 402:
                wait_until_next_hour()
                attempt = 1
                continue
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次 HTTP {code} 錯誤：{e}")
        except Exception as exc:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次異常：{exc}")

        if attempt < max_retries:
            # 指數退避 + 抖動因子
            backoff = delay * (2 ** attempt) + random.uniform(0, 1.0)
            logger.info(f"[{dataset}] {backoff:.1f}s 後重試…")
            time.sleep(backoff)
        else:
            logger.error(f"[{dataset}] 已重試 {max_retries} 次，放棄。")
            return []

    return []


# ─────────────────────────────────────────────
# 非同步請求函式（aiohttp，v2 新增）
# ─────────────────────────────────────────────
async def finmind_get_async(
    dataset: str,
    params: dict,
    session,  # aiohttp.ClientSession
    max_retries: int = 3,
    raise_on_batch_400: bool = False,
) -> list:
    """
    FinMind API 非同步請求函式（基於 aiohttp）。

    使用前需在外部建立 aiohttp.ClientSession，並以 asyncio.gather 調度多個協程：

        async with aiohttp.ClientSession() as session:
            tasks = [finmind_get_async(ds, p, session) for p in params_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

    與全域 TokenBucketRateLimiter 整合，確保非同步並行模式下速率合規。

    Parameters
    ----------
    dataset            : FinMind dataset 名稱
    params             : 額外參數 dict
    session            : aiohttp.ClientSession 實例
    max_retries        : 非 402 錯誤的最大重試次數
    raise_on_batch_400 : True 時批次請求收到 400 拋出 BatchNotSupportedError

    Returns
    -------
    list  成功時回傳 data 列表；失敗時回傳空列表 []
    """
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}
    is_batch = "data_id" not in params

    for attempt in range(1, max_retries + 1):
        try:
            # 非同步速率限制
            await _rate_limiter.acquire_async()

            async with session.get(
                FINMIND_API_URL,
                headers=headers,
                params=req_params,
                timeout=__import__("aiohttp").ClientTimeout(connect=15, total=135),
            ) as resp:
                if resp.status == 402:
                    # 協程中等待配額重置（用 asyncio.sleep 不阻塞事件迴圈）
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

                if resp.status == 400:
                    if raise_on_batch_400 and is_batch:
                        raise BatchNotSupportedError(
                            f"[async][{dataset}] 批次請求（HTTP 400）被拒絕"
                        )
                    return []

                resp.raise_for_status()
                payload = await resp.json()

                if payload.get("status") == 402:
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
                    logger.warning(
                        f"[async][{dataset}] status={payload.get('status')}, "
                        f"msg={payload.get('msg')}"
                    )
                    return []

                return payload.get("data", [])

        except BatchNotSupportedError:
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
                logger.error(f"[async][{dataset}] 已重試 {max_retries} 次，放棄：{exc}")
                return []

    return []


# ─────────────────────────────────────────────
# API 配額預查（供 parallel_fetch.py 使用）
# ─────────────────────────────────────────────
def check_api_quota() -> tuple[int, int]:
    """
    查詢當前 FinMind API 使用量與上限。
    回傳 (used, limit)。
    若查詢失敗，回傳 (-1, 600)（不阻斷執行）。
    """
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
        return used, limit
    except Exception as e:
        logger.warning(f"無法查詢 API 配額：{e}")
        return -1, 600
