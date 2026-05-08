from __future__ import annotations

import sys
import logging
import time
import json
import os
import subprocess
import multiprocessing as mp
import argparse
from pathlib import Path
from datetime import date

# ── sys.path 自我修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "fetchers", "monitor"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)

"""
parallel_fetch.py v3.1 — 並行資料抓取管理器（監控整合標準版）
================================================================================
v3.1 重大改進：
  · 整合 fetch_log v3.1：對每一個子任務執行結果進行標準化監控記錄。
  · 效能監控：記錄各腳本執行耗時（duration_ms），便於分析管線瓶頸。
  · 錯誤診斷：子腳本失敗時自動擷取 stderr 末尾資訊並存入 fetch_log。
  · 配額防護：執行前自動檢查 FinMind API 配額，防止過度請求。
  · 完整註解：提供多樣化執行範例，便於維運。

執行範例（常規）：
    python scripts/fetchers/parallel_fetch.py                # 並行執行所有 Phase 的抓取任務 (預設全市場)
    python scripts/fetchers/parallel_fetch.py --workers 4    # 指定並行數

執行範例（單一個股或強制重抓）：
    python scripts/fetchers/parallel_fetch.py --stock-id 2330
    python scripts/fetchers/parallel_fetch.py --stock-id 2330 --force
    python scripts/fetchers/parallel_fetch.py --force

執行範例（分段）：
    python scripts/fetchers/parallel_fetch.py --phase 1      # 僅執行 Phase 1 (核心資料)
"""

try:
    from core.path_setup import ensure_scripts_on_path, ensure_dirs_exist, get_logs_dir
    ensure_scripts_on_path(__file__)
    from core.finmind_client import check_api_quota, get_request_stats
    from core.db_utils import get_db_conn, FailureLogger
    _CORE_OK = True
except Exception as _e:
    _CORE_OK = False
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"core v3.1 helpers 未到位，使用 fallback：{_e}")
    from core.finmind_client import check_api_quota  # type: ignore
    from core.db_utils import get_db_conn  # type: ignore

# ─────────────────────────────────────────────
# 路徑與常數
# ─────────────────────────────────────────────
BASE_DIR = _THIS_DIR.parent
VENV_PYTHON = os.environ.get(
    "VENV_PYTHON", str(BASE_DIR.parent / "venv" / "bin" / "python3"),
)

PHASE_0 = [str(_THIS_DIR / "fetch_stock_info.py")]
PHASE_1 = [
    str(_THIS_DIR / "fetch_technical_data.py"),
    str(_THIS_DIR / "fetch_price_adj_data.py"),
    str(_THIS_DIR / "fetch_chip_data.py"),
    str(_THIS_DIR / "fetch_advanced_chip_data.py"),
    str(_THIS_DIR / "fetch_international_data.py"),
    str(_THIS_DIR / "fetch_macro_data.py"),
    str(_THIS_DIR / "fetch_fundamental_data.py"),
    str(_THIS_DIR / "fetch_cash_flows_data.py"),
    str(_THIS_DIR / "fetch_derivative_data.py"),
    str(_THIS_DIR / "fetch_extended_derivative_data.py"),
    str(_THIS_DIR / "fetch_sponsor_chip_data.py"),
    str(_THIS_DIR / "fetch_macro_fundamental_data.py"),
    str(_THIS_DIR / "fetch_derivative_sentiment_data.py"),
    str(_THIS_DIR / "fetch_total_return_index.py"),
]
PHASE_2 = [
    str(_THIS_DIR / "fetch_event_risk_data.py"),
    str(_THIS_DIR / "fetch_news_data.py"),
    str(_THIS_DIR / "fetch_fred_data.py"),
]

NON_STOCK_SCRIPTS = {
    "fetch_stock_info.py",
    "fetch_international_data.py",
    "fetch_macro_data.py",
    "fetch_macro_fundamental_data.py",
    "fetch_total_return_index.py",
    "fetch_event_risk_data.py",
    "fetch_fred_data.py",
    "fetch_derivative_sentiment_data.py",
    "fetch_extended_derivative_data.py",
    "fetch_derivative_data.py",
}

DEFAULT_MAX_WORKERS = 8
MAX_SCRIPT_TIMEOUT = 12 * 3600  # 初始全量補抓建議 12 小時
API_QUOTA_MIN = 100
_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 日誌與監控
# ─────────────────────────────────────────────
LOG_DIR = get_logs_dir() if _CORE_OK else (BASE_DIR / "outputs" / "logs")
if not _CORE_OK: LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "parallel_fetch.log"

_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
try:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _handlers.append(logging.FileHandler(str(LOG_FILE)))
except (PermissionError, OSError) as _log_err:
    # 無法寫入日誌檔案時，降級為僅輸出至終端（不中斷程式）
    logging.getLogger(__name__).warning(f"無法寫入日誌檔：{_log_err}，改為僅輸出至終端")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger(__name__)

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None):
    """v3.1 標準化日誌寫入"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (
                    run_ts, table_name, stock_id, status, rows_inserted, 
                    fetch_date_from, fetch_date_to, duration_ms, error_message, cli_args
                ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (table_name, stock_id, status, rows_inserted, fetch_date_from, fetch_date_to, duration_ms, error_message, _CLI_ARGS_STR))
        conn.commit()
    except Exception as e:
        logger.warning(f"無法寫入 fetch_log: {e}")

# ─────────────────────────────────────────────
# 子任務執行
# ─────────────────────────────────────────────
def run_script_with_args(task: tuple[str, list[str]]) -> tuple[str, bool, int, float, str]:
    """執行單一 fetcher，並可傳入額外參數。回傳 (script_path, success, returncode, duration, stderr_tail)"""
    script_path, extra_args = task
    if not os.path.exists(script_path):
        return script_path, False, -127, 0.0, f"找不到檔案: {script_path}"

    start = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BASE_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"

    rc = 0
    success = False
    stderr_tail = ""
    script_name = Path(script_path).name
    
    # Filter out --stock-id for scripts that don't support it
    filtered_args = []
    if extra_args:
        i = 0
        while i < len(extra_args):
            if extra_args[i] == "--stock-id" and script_name in NON_STOCK_SCRIPTS:
                i += 2  # skip --stock-id and its value
            else:
                filtered_args.append(extra_args[i])
                i += 1

    try:
        cmd = [VENV_PYTHON, script_path] + filtered_args
        proc = subprocess.run(
            cmd,
            env=env, check=False, timeout=MAX_SCRIPT_TIMEOUT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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

def _record_run_result(script_path: str, success: bool, rc: int, duration: float, stderr_tail: str, flog: FailureLogger | None, stock_id: str = "SYSTEM") -> None:
    """記錄執行結果至日誌與資料庫"""
    script_name = Path(script_path).name
    duration_ms = int(duration * 1000)
    
    if success:
        logger.info(f"✅ 完成: {script_name} ({duration:.1f}s)")
    else:
        logger.error(f"❌ 失敗: {script_name} ({duration:.1f}s) rc={rc}")
        if stderr_tail: logger.error(f"   stderr_tail: {stderr_tail}")

    # 寫入 fetch_log
    conn = get_db_conn()
    try:
        _write_fetch_log(
            conn, script_name, stock_id, 
            "success" if success else "failed",
            duration_ms=duration_ms, 
            error_message=None if success else f"rc={rc}: {stderr_tail[:300]}"
        )
    finally:
        conn.close()

    # 寫入 FailureLogger
    if not success and flog is not None:
        try: flog.record(script=script_name, returncode=rc, duration=round(duration, 2), error=stderr_tail or "")
        except Exception: pass

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="並行資料抓取管線 (v3.1 — 監控整合標準版)")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help=f"並行 worker 數 (預設 {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "0", "1", "2", "1+2"], help="執行哪個 phase")
    parser.add_argument("--abort-on-low-quota", action="store_true", help=f"配額剩餘 < {API_QUOTA_MIN} 時中止")
    parser.add_argument("--stock-id", type=str, help="指定抓取單一股號 (例如 2330)")
    parser.add_argument("--force", action="store_true", help="強制重抓 (略過 cache)")
    args = parser.parse_args()

    extra_args = []
    if args.stock_id:
        extra_args.extend(["--stock-id", args.stock_id])
    if args.force:
        extra_args.append("--force")
        
    log_stock_id = args.stock_id if args.stock_id else "SYSTEM"

    if _CORE_OK: ensure_dirs_exist()

    total_start = time.monotonic()
    logger.info("=" * 60)
    logger.info("    並行資料抓取管線 v3.1 啟動")
    logger.info("=" * 60)

    # ── 配額預查 ──
    try:
        used, limit = check_api_quota()
        remaining = limit - used if used >= 0 else -1
        logger.info(f"API 配額：已用 {used}/{limit}，剩餘 {remaining}")
        if 0 <= remaining < API_QUOTA_MIN:
            msg = f"配額剩餘 {remaining} < {API_QUOTA_MIN} 警戒值"
            if args.abort_on_low_quota:
                logger.error(f"❌ {msg}，中止執行")
                return
            else:
                logger.warning(f"⚠️ {msg}")
    except Exception as e:
        logger.warning(f"配額檢查跳過：{e}")

    flog = None
    if _CORE_OK:
        try: flog = FailureLogger("parallel_fetch")
        except Exception: pass

    results = []

    # ── Phase 0：序列 ──
    if args.phase in ("all", "0"):
        logger.info(f"\n[Phase 0] 序列執行 {len(PHASE_0)} 支基礎腳本")
        for script in PHASE_0:
            res = run_script_with_args((script, extra_args))
            results.append(res)
            _record_run_result(*res, flog=flog, stock_id=log_stock_id)

    # ── Phase 1+2：併發 ──
    parallel_targets = []
    if args.phase in ("all", "1", "1+2"): parallel_targets += PHASE_1
    if args.phase in ("all", "2", "1+2"): parallel_targets += PHASE_2

    if parallel_targets:
        logger.info(f"\n[Hyper-Fetch] 並行執行 {len(parallel_targets)} 支腳本 (workers={args.workers})")
        pool = mp.Pool(processes=max(1, args.workers))
        try:
            tasks = [(script, extra_args) for script in parallel_targets]
            for res in pool.imap_unordered(run_script_with_args, tasks):
                results.append(res)
                _record_run_result(*res, flog=flog, stock_id=log_stock_id)
            pool.close()
        except KeyboardInterrupt:
            logger.warning("收到 KeyboardInterrupt，關閉中...")
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

    if failed:
        logger.info("失敗腳本：")
        for path, ok, rc, dur, err in results:
            if not ok: logger.info(f"  ❌ {Path(path).name}  rc={rc}  ({dur:.1f}s)  {err[:200]}")

    slowest = sorted(results, key=lambda x: -x[3])[:5]
    if slowest:
        logger.info("最慢 5 支腳本：")
        for path, ok, rc, dur, _ in slowest:
            logger.info(f"  {'✅' if ok else '❌'} {Path(path).name:<40s} {dur:7.1f}s")

    if flog: flog.summary()
    if _CORE_OK:
        try: get_request_stats().summary()
        except Exception: pass

if __name__ == "__main__":
    main()
