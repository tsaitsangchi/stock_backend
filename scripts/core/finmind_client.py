"""
core/finmind_client.py — 統一的 FinMind API 客戶端
=====================================================
[P0 修正] 取代散落在 10 個 fetch_*.py 中各自不同的 finmind_get 實作。

統一特性：
  - 指數退避重試（delay × 2^n），最多 max_retries 次
  - 402 配額耗盡：無限等待至下一整點後重試，不計入重試次數
  - 400 Bad Request：直接返回空列表（參數錯誤，無需重試）
  - timeout=(15, 120)：connect 15s / read 120s（所有腳本一致）
  - check_api_quota()：並行抓取前預查剩餘配額，避免快速耗盡
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import requests

from config import FINMIND_TOKEN

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 402 配額限制等待
# ─────────────────────────────────────────────
def wait_until_next_hour() -> None:
    """
    API 配額耗盡（HTTP 402）時呼叫。
    FinMind 在整點重置配額，等待至下一整點 + 65 秒緩衝。
    """
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_sec = (next_hour - now).total_seconds() + 65
    logger.warning(
        f"API 配額達上限（402），等待 {wait_sec:.0f} 秒至 "
        f"{next_hour.strftime('%H:%M')} 後恢復…"
    )
    time.sleep(wait_sec)
    logger.info("等待結束，恢復請求。")


# ─────────────────────────────────────────────
# 核心請求函式
# ─────────────────────────────────────────────
def finmind_get(
    dataset: str,
    params: dict,
    delay: float = 1.2,
    max_retries: int = 3,
) -> list:
    """
    統一的 FinMind API 請求函式。

    Parameters
    ----------
    dataset     : FinMind dataset 名稱（如 "TaiwanStockPrice"）
    params      : 額外參數 dict（如 {"data_id": "2330", "start_date": "..."}）
    delay       : 成功後的休眠秒數（避免過快耗盡配額）
    max_retries : 非 402 錯誤的最大重試次數

    Returns
    -------
    list  成功時回傳 data 列表；失敗時回傳空列表 []
    """
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}
    req_params = {"dataset": dataset, **params}

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                FINMIND_API_URL,
                headers=headers,
                params=req_params,
                timeout=(15, 120),  # connect=15s, read=120s
            )

            # ── 402：配額耗盡，等待整點重置後繼續（不計入重試次數）──
            if resp.status_code == 402:
                wait_until_next_hour()
                attempt = 1  # 重置嘗試計數
                continue

            # ── 400：請求參數錯誤，直接跳過（不重試）──
            if resp.status_code == 400:
                logger.debug(
                    f"[{dataset}] 400 Bad Request，跳過"
                    f"（data_id={params.get('data_id')}，"
                    f"start={params.get('start_date')}）"
                )
                return []

            resp.raise_for_status()
            payload = resp.json()

            # ── API 業務邏輯 402 ──
            if payload.get("status") == 402:
                wait_until_next_hour()
                attempt = 1
                continue

            if payload.get("status") != 200:
                logger.warning(
                    f"[{dataset}] status={payload.get('status')}, "
                    f"msg={payload.get('msg')}"
                )
                return []

            time.sleep(delay)
            return payload.get("data", [])

        except requests.exceptions.Timeout:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次逾時")
        except requests.exceptions.ConnectionError:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次連線失敗")
        except requests.exceptions.HTTPError as e:
            # raise_for_status 拋出的 HTTPError（402 已在上方處理）
            code = e.response.status_code if e.response is not None else 0
            if code == 402:
                wait_until_next_hour()
                attempt = 1
                continue
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次 HTTP {code} 錯誤：{e}")
        except Exception as exc:
            logger.warning(f"[{dataset}] 第 {attempt}/{max_retries} 次異常：{exc}")

        if attempt < max_retries:
            backoff = delay * (2 ** attempt)  # 指數退避
            logger.info(f"[{dataset}] {backoff:.1f}s 後重試…")
            time.sleep(backoff)
        else:
            logger.error(f"[{dataset}] 已重試 {max_retries} 次，放棄。")
            return []

    return []  # 理論上不會到達這裡


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
