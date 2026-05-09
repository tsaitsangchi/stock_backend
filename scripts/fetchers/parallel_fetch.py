"""
parallel_fetch.py — 並行資料抓取管線管理器 (v3.4 標準日誌版)
================================================================================
v3.4 改進：
  ★ 整合 `db_utils.write_fetch_log`：移除本地冗餘日誌函式與 Schema 升級邏輯。

v3.2 既有 (配合 db_utils v3.0, finmind_client v3.1, path_setup v2.0)：
  ★ 標準化 `fetch_log`：對齊個股抓取腳本，將子任務狀態、耗時與參數精確落盤。
  ★ 配額防護：啟動前自動檢查 FinMind API 剩餘配額。

執行範例：
    python scripts/fetchers/parallel_fetch.py --phase 1
"""

from __future__ import annotations

import sys
import logging
import time
import os
import subprocess
import multiprocessing as mp
import argparse
from pathlib import Path
from datetime import date, datetime

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path, ensure_dirs_exist, get_logs_dir
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.finmind_client import check_api_quota, get_request_stats
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    write_fetch_log,
    FailureLogger,
    DDL_FETCH_LOG
)

# ─────────────────────────────────────────────
# 常數與 Phase 定義
# ─────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
VENV_PYTHON = sys.executable
_CLI_ARGS_STR = " ".join(sys.argv)

PHASE_0 = [str(SCRIPTS_DIR / "fetch_stock_info.py")]

PHASE_1 = [
    str(SCRIPTS_DIR / "fetch_technical_data.py"),
    str(SCRIPTS_DIR / "fetch_price_adj_data.py"),
    str(SCRIPTS_DIR / "fetch_sponsor_chip_data.py"),
    str(SCRIPTS_DIR / "fetch_fundamental_data.py"),
    str(SCRIPTS_DIR / "fetch_macro_data.py"),
    str(SCRIPTS_DIR / "fetch_international_data.py"),
    str(SCRIPTS_DIR / "fetch_macro_fundamental_data.py"),
    str(SCRIPTS_DIR / "fetch_total_return_index.py"),
    str(SCRIPTS_DIR / "fetch_chip_data.py"),
    str(SCRIPTS_DIR / "fetch_advanced_chip_data.py"),
    str(SCRIPTS_DIR / "fetch_cash_flows_data.py"),
    str(SCRIPTS_DIR / "fetch_derivative_data.py"),
    str(SCRIPTS_DIR / "fetch_derivative_sentiment_data.py"),
]

PHASE_2 = [
    str(SCRIPTS_DIR / "fetch_event_risk_data.py"),
    str(SCRIPTS_DIR / "fetch_news_data.py"),
    str(SCRIPTS_DIR / "fetch_fred_data.py"),
    str(SCRIPTS_DIR / "fetch_extended_derivative_data.py"),
]

NON_STOCK_SCRIPTS = {
    "fetch_stock_info.py",
    "fetch_international_data.py",
    "fetch_macro_data.py",
    "fetch_macro_fundamental_data.py",
    "fetch_total_return_index.py",
    "fetch_fred_data.py",
    "fetch_derivative_sentiment_data.py",
}

DEFAULT_MAX_WORKERS = 8
MAX_SCRIPT_TIMEOUT = 12 * 3600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 日誌與監控
# ──────────────────────────────────────────────
# ─────────────────────────────────────────────
# 子任務執行
# ─────────────────────────────────────────────
def run_script_with_args(task: tuple[str, list[str]]) -> tuple[str, bool, int, int, str]:
    """執行單一 fetcher"""
    script_path, extra_args = task
    script_name = Path(script_path).name
    
    filtered_args = []
    if extra_args:
        i = 0
        while i < len(extra_args):
            if extra_args[i] == "--stock-id" and script_name in NON_STOCK_SCRIPTS:
                i += 2
            else:
                filtered_args.append(extra_args[i])
                i += 1

    cmd = [VENV_PYTHON, script_path] + filtered_args
    start_ts = time.monotonic()
    
    try:
        proc = subprocess.run(
            cmd, check=False, timeout=MAX_SCRIPT_TIMEOUT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        duration_ms = int((time.monotonic() - start_ts) * 1000)
        success = (proc.returncode == 0)
        stderr_tail = (proc.stderr or "").strip().split("\n")[-3:]
        return script_path, success, proc.returncode, duration_ms, "\n".join(stderr_tail)
    except subprocess.TimeoutExpired:
        return script_path, False, -1, int((time.time() - start_ts) * 1000), "Timeout"
    except Exception as e:
        return script_path, False, -1, int((time.time() - start_ts) * 1000), str(e)


def _record_task_result(res: tuple, flog: FailureLogger | None, stock_id: str):
    """記錄結果"""
    path, success, rc, dur_ms, err = res
    script_name = Path(path).name
    
    if success:
        logger.info(f"  ✅ 完成: {script_name} ({dur_ms/1000:.1f}s)")
    else:
        logger.error(f"  ❌ 失敗: {script_name} ({dur_ms/1000:.1f}s) rc={rc} | Error: {err}")

    conn = get_db_conn()
    try:
        write_fetch_log(
            conn, script_name, stock_id, 
            status="success" if success else "failed",
            fetch_mode="parallel_launcher",
            duration_ms=dur_ms, 
            error_message=err if not success else None
        )
    finally:
        conn.close()

    if not success and flog:
        flog.record(script=script_name, stock_id=stock_id, error=err, duration=dur_ms/1000)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="並行資料抓取管線 (v3.3 — 修正長字串警告)")
    p.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--phase", type=str, default="all", choices=["all", "0", "1", "2", "1+2"])
    p.add_argument("--stock-id", help="指定單一標的 ID")
    p.add_argument("--force", action="store_true")
    p.add_argument("--abort-on-low-quota", action="store_true")
    p.add_argument("--quota-min", type=int, default=100)
    args = p.parse_args()

    ensure_dirs_exist()
    conn = get_db_conn()
    ensure_ddl(conn, DDL_FETCH_LOG)
    conn.close()

    total_start = time.time()
    logger.info("=" * 65)
    logger.info(f"    Quantum Finance 並行抓取管線 v3.3 啟動 ({datetime.now().strftime('%H:%M:%S')})")
    logger.info("=" * 65)

    try:
        used, limit = check_api_quota()
        remaining = limit - used
        logger.info(f"FinMind API 配額：已用 {used:,} / 限制 {limit:,} (剩餘 {remaining:,})")
        if remaining < args.quota_min:
            msg = f"API 配額過低 ({remaining} < {args.quota_min})"
            if args.abort_on_low_quota:
                logger.error(f"❌ {msg}，基於保護機制中止執行。")
                return
            logger.warning(f"⚠️ {msg}")
    except Exception as e:
        logger.warning(f"無法獲取 API 配額資訊: {e}")

    flog = FailureLogger("parallel_fetch", db_conn=get_db_conn(), log_to_db=True)
    extra_args = []
    if args.stock_id: extra_args.extend(["--stock-id", args.stock_id])
    if args.force: extra_args.append("--force")
    
    log_sid = args.stock_id if args.stock_id else "ALL_CORE"
    results = []

    if args.phase in ("all", "0"):
        logger.info(f"\n[Phase 0] 序列執行基礎資訊腳本...")
        for script in PHASE_0:
            res = run_script_with_args((script, extra_args))
            results.append(res)
            _record_task_result(res, flog, log_sid)

    parallel_targets = []
    if args.phase in ("all", "1", "1+2"): parallel_targets += PHASE_1
    if args.phase in ("all", "2", "1+2"): parallel_targets += PHASE_2

    if parallel_targets:
        logger.info(f"\n[Phase 1/2] 啟動 Hyper-Fetch 並行處理 (Workers={args.workers})")
        pool = mp.Pool(processes=max(1, args.workers))
        try:
            tasks = [(script, extra_args) for script in parallel_targets]
            for res in pool.imap_unordered(run_script_with_args, tasks):
                results.append(res)
                _record_task_result(res, flog, log_sid)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            raise
        finally:
            pool.join()

    total_dur = time.time() - total_start
    success_count = sum(1 for r in results if r[1])
    fail_count = len(results) - success_count

    print("\n" + "=" * 65)
    print(f"  管線任務結束摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("-" * 65)
    print(f"  總執行時間 : {total_dur / 60:.2f} 分鐘")
    print(f"  成功腳本數 : {success_count}")
    print(f"  失敗腳本數 : {fail_count}")
    
    slowest = sorted(results, key=lambda x: -x[3])[:5]
    print("\n  [最慢任務 TOP 5]")
    for r in slowest:
        print(f"  ⏱️ {Path(r[0]).name:<35s} {r[3]/1000:7.1f}s")
    print("=" * 65 + "\n")

    flog.summary()
    get_request_stats().summary()

if __name__ == "__main__":
    main()