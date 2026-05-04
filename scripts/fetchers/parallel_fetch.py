"""
parallel_fetch.py v3.0 — 並行資料抓取管理器（core v3.0 完整性對齊版）
================================================================================
同時啟動多個 fetch 腳本，大幅縮短每日更新時間。

對齊 core v3.0 的「逐支逐日 commit」精神：
本檔本身為 subprocess launcher，不直接寫資料表；真正的逐支逐日落盤由各被
呼叫的 fetcher 內部的 commit_per_stock_per_day 完成。本檔負責：
  · 把 phase 排程併發跑滿，但維持單一全域速率限制（FinMind Token Bucket）
  · 對「子腳本層級」的成敗 / 耗時做完整記錄（fetch_log + FailureLogger）
  · 最後印出 RequestStats 統計與最慢子任務

v3.0 重大改進：
  ★ ProcessPoolExecutor → multiprocessing.Pool.imap_unordered（INT-04 串流模式）
    避免一次 submit 全部 future 帶來的記憶體尖峰；KeyboardInterrupt 可優雅退出
  ★ 整合 core.path_setup（ensure_dirs_exist）
  ★ 整合 core.db_utils.FailureLogger：每支失敗腳本都被原子記錄
  ★ 整合 core.finmind_client.get_request_stats：結尾印出每個 dataset 的請求統計
  ★ 子腳本失敗時抓 stderr 末 5 行，方便診斷（不再只剩 returncode）
  ★ 配額不足時可設 --abort-on-low-quota，停止 phase2 避免浪費
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path 自我修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "fetchers", "monitor"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)

import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import time
from datetime import date

try:
    from core.path_setup import ensure_scripts_on_path, ensure_dirs_exist, get_logs_dir
    ensure_scripts_on_path(__file__)
    from core.finmind_client import check_api_quota, get_request_stats
    from core.db_utils import get_db_conn, log_fetch_result, FailureLogger
    _CORE_OK = True
except Exception as _e:
    _CORE_OK = False
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"core v3.0 helpers 未到位，使用 fallback：{_e}")
    # 最小 fallback
    from core.finmind_client import check_api_quota  # type: ignore
    from core.db_utils import get_db_conn, log_fetch_result  # type: ignore

# ─────────────────────────────────────────────
# 路徑常數
# ─────────────────────────────────────────────
SCRIPTS_DIR = _THIS_DIR
BASE_DIR = SCRIPTS_DIR.parent
VENV_PYTHON = os.environ.get(
    "VENV_PYTHON", str(BASE_DIR.parent / "venv" / "bin" / "python3"),
)

# ─────────────────────────────────────────────
# Phase 規劃
# ─────────────────────────────────────────────
PHASE_0 = [
    str(SCRIPTS_DIR / "fetch_stock_info.py"),
]

PHASE_1 = [
    str(SCRIPTS_DIR / "fetch_technical_data.py"),
    str(SCRIPTS_DIR / "fetch_price_adj_data.py"),
    str(SCRIPTS_DIR / "fetch_chip_data.py"),
    str(SCRIPTS_DIR / "fetch_advanced_chip_data.py"),
    str(SCRIPTS_DIR / "fetch_international_data.py"),
    str(SCRIPTS_DIR / "fetch_macro_data.py"),
    str(SCRIPTS_DIR / "fetch_fundamental_data.py"),
    str(SCRIPTS_DIR / "fetch_cash_flows_data.py"),
    str(SCRIPTS_DIR / "fetch_derivative_data.py"),
    str(SCRIPTS_DIR / "fetch_extended_derivative_data.py"),
    str(SCRIPTS_DIR / "fetch_sponsor_chip_data.py"),
    str(SCRIPTS_DIR / "fetch_macro_fundamental_data.py"),
    str(SCRIPTS_DIR / "fetch_derivative_sentiment_data.py"),
]

PHASE_2 = [
    str(SCRIPTS_DIR / "fetch_event_risk_data.py"),
    str(SCRIPTS_DIR / "fetch_news_data.py"),
    str(SCRIPTS_DIR / "fetch_fred_data.py"),
]

DEFAULT_MAX_WORKERS = 8
MAX_SCRIPT_TIMEOUT = 3 * 3600
API_QUOTA_MIN = 100

# ─────────────────────────────────────────────
# 日誌
# ─────────────────────────────────────────────
LOG_DIR = (
    get_logs_dir() if _CORE_OK else (BASE_DIR / "outputs" / "logs")
)
if not _CORE_OK:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "parallel_fetch.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 子任務執行
# ─────────────────────────────────────────────
def run_script(script_path: str) -> tuple[str, bool, int, float, str]:
    """
    執行單一 fetcher。回傳 (script_path, success, returncode, duration, stderr_tail)。
    """
    if not os.path.exists(script_path):
        return script_path, False, -127, 0.0, f"找不到檔案: {script_path}"

    start = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BASE_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"

    rc = 0
    success = False
    stderr_tail = ""
    try:
        proc = subprocess.run(
            [VENV_PYTHON, script_path],
            env=env,
            check=False,
            timeout=MAX_SCRIPT_TIMEOUT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        rc = proc.returncode
        success = rc == 0
        if not success:
            tail = (proc.stderr or "").strip().split("\n")[-5:]
            stderr_tail = "\n".join(tail)[-500:]
    except subprocess.TimeoutExpired:
        rc = -1
        stderr_tail = f"timeout > {MAX_SCRIPT_TIMEOUT}s"
    except Exception as e:
        rc = -1
        stderr_tail = str(e)[:500]

    duration = time.monotonic() - start
    return script_path, success, rc, duration, stderr_tail


def _record_run_result(
    script_path: str, success: bool, rc: int, duration: float,
    stderr_tail: str, flog: "FailureLogger | None",
) -> None:
    """寫入 fetch_log + FailureLogger（雙通道，皆 graceful fallback）。"""
    script_name = Path(script_path).name
    if success:
        logger.info(f"✅ 完成: {script_name} ({duration:.1f}s)")
    else:
        logger.error(f"❌ 失敗: {script_name} ({duration:.1f}s) rc={rc}")
        if stderr_tail:
            logger.error(f"   stderr_tail: {stderr_tail}")

    # fetch_log
    try:
        conn = get_db_conn()
        try:
            log_fetch_result(
                conn, script_name, "SYSTEM",
                date.today().isoformat(), date.today().isoformat(),
                0, "SUCCESS" if success else "FAILED",
                None if success else f"rc={rc}: {stderr_tail[:300]}",
            )
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"無法寫入 fetch_log: {e}")

    # FailureLogger
    if not success and flog is not None:
        try:
            flog.record(
                script=script_name, returncode=rc,
                duration=round(duration, 2), error=stderr_tail or "",
            )
        except Exception:
            pass


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="並行資料抓取管線 v3.0")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"並行 worker 數（預設 {DEFAULT_MAX_WORKERS}）")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "0", "1", "2", "1+2"],
                        help="執行哪個 phase")
    parser.add_argument("--abort-on-low-quota", action="store_true",
                        help=f"配額剩餘 < {API_QUOTA_MIN} 時直接中止")
    args = parser.parse_args()

    if _CORE_OK:
        ensure_dirs_exist()

    total_start = time.monotonic()
    logger.info("=" * 60)
    logger.info("    並行資料抓取管線 v3.0 啟動")
    logger.info("=" * 60)

    # ── 配額預查 ──
    try:
        used, limit = check_api_quota()
        remaining = limit - used if used >= 0 else -1
        logger.info(f"API 配額：已用 {used}/{limit}，剩餘 {remaining}")
        if remaining >= 0 and remaining < API_QUOTA_MIN:
            msg = f"配額剩餘 {remaining} < {API_QUOTA_MIN} 警戒值"
            if args.abort_on_low_quota:
                logger.error(f"❌ {msg}，--abort-on-low-quota 啟用，中止執行")
                return
            else:
                logger.warning(f"⚠️ {msg}（仍會繼續執行）")
    except Exception as e:
        logger.warning(f"配額檢查跳過：{e}")

    flog: FailureLogger | None = None
    if _CORE_OK:
        try:
            flog = FailureLogger("parallel_fetch")
        except Exception:
            flog = None

    results: list[tuple[str, bool, int, float, str]] = []

    # ── Phase 0：序列 ──
    if args.phase in ("all", "0"):
        logger.info(f"\n[Phase 0] 序列執行 {len(PHASE_0)} 支基礎腳本")
        for script in PHASE_0:
            res = run_script(script)
            results.append(res)
            _record_run_result(*res, flog=flog)

    # ── Phase 1+2：併發（imap_unordered 串流）──
    parallel_targets: list[str] = []
    if args.phase in ("all", "1", "1+2"):
        parallel_targets += PHASE_1
    if args.phase in ("all", "2", "1+2"):
        parallel_targets += PHASE_2

    if parallel_targets:
        logger.info(
            f"\n[Hyper-Fetch] 並行執行 {len(parallel_targets)} 支腳本"
            f"（workers={args.workers}, mode=imap_unordered 串流）"
        )
        # multiprocessing.Pool.imap_unordered：worker 完成一個才送下一個 chunk，
        # 不會像 ProcessPoolExecutor 一次將所有 future 列舉到記憶體
        pool = mp.Pool(processes=max(1, args.workers))
        try:
            for res in pool.imap_unordered(run_script, parallel_targets):
                results.append(res)
                _record_run_result(*res, flog=flog)
            pool.close()
        except KeyboardInterrupt:
            logger.warning("收到 KeyboardInterrupt，正在優雅關閉 worker pool …")
            pool.terminate()
            raise
        finally:
            pool.join()

    # ── 摘要 ──
    total_duration = time.monotonic() - total_start
    success = sum(1 for _, ok, *_ in results if ok)
    failed = len(results) - success

    logger.info("\n" + "=" * 60)
    logger.info(f"任務結束。成功: {success}, 失敗: {failed}")
    logger.info(f"總耗時: {total_duration / 60:.2f} 分鐘")
    logger.info("=" * 60)

    # 失敗清單
    if failed:
        logger.info("失敗腳本：")
        for path, ok, rc, dur, err in results:
            if not ok:
                logger.info(f"  ❌ {Path(path).name}  rc={rc}  ({dur:.1f}s)  {err[:200]}")

    # 最慢前 5
    slowest = sorted(results, key=lambda x: -x[3])[:5]
    if slowest:
        logger.info("最慢 5 支腳本：")
        for path, ok, rc, dur, _ in slowest:
            mark = "✅" if ok else "❌"
            logger.info(f"  {mark} {Path(path).name:<40s} {dur:7.1f}s")

    if flog is not None:
        flog.summary()

    # FinMind 請求統計（v3.0）— 父進程通常沒呼叫 finmind_get，這裡會是空表，
    # 但保留以利人工模式直接調用本檔的場景
    if _CORE_OK:
        try:
            stats = get_request_stats()
            stats.summary()
        except Exception:
            pass


if __name__ == "__main__":
    main()
