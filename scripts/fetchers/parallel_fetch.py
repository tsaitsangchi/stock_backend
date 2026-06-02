"""
fetchers/parallel_fetch.py v6 (並行抓取工具(fetchers))
================================================================================
**最後更新日期**: 2026-06-02
**主權狀態**: ACTIVE (parallel fetch util)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:並行抓取工具(fetchers/ 版;多 worker 平行 fetch)。

**輸入 → 輸出**:dataset 清單 → 並行抓取

**為什麼需要它**:data_pipeline 子系統之並行抓取元件。

## 📜 一、核心定義說明 (Core Definitions)

1. **[Parallel Fetch]**:多 worker 並行抓取
2. **[Sovereignty Declaration]**:本程式為**非 charter-core 子系統**工具(charter 可達/引用),不涉 §3.1/§3.2 序列治權主軸、不持五套禁令、不處理 §8.5 anti-leakage。
3. **[Historical Reference Authority]**:本檔標頭版本為記述性快照,非權威來源(權威為憲章 + 程式現行碼)。

## 📊 二、全量功能群矩陣 (Functional Group Matrix)

| 功能 / 指令 | 說明 |
| :--- | :--- |
| parallel_fetch | 並行抓取多 dataset |

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v6 | 2026-06-02 | Codex | §一.11 標頭三段式 + 白話補正;原邏輯不變。 | **ACTIVE** |

## 原始說明
parallel_fetch.py — 並行資料抓取管線管理器 (v3.3 Schema 升級修復版)
================================================================================
v3.3 改進：
  ★ 導入 `_upgrade_schema()`：自動將 `fetch_log.fetch_mode` 欄位擴充至 VARCHAR(50)，
    修復因 'parallel_launcher' 字串過長導致的寫入警告。

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
    str(SCRIPTS_DIR / "fetch_derivative_sentiment_data.py"),
]

PHASE_2 = [
    str(SCRIPTS_DIR / "fetch_event_risk_data.py"),
    str(SCRIPTS_DIR / "fetch_news_data.py"),
    str(SCRIPTS_DIR / "fetch_fred_data.py"),
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
def _upgrade_schema(conn) -> None:
    """自動擴充 fetch_log 欄位以支援長字串標籤。"""
    try:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE fetch_log ALTER COLUMN fetch_mode TYPE VARCHAR(50);")
        conn.commit()
        logger.info("✅ Schema 升級：已將 fetch_log.fetch_mode 擴充至 VARCHAR(50)。")
    except Exception as e:
        conn.rollback()
        logger.debug(f"Schema 升級略過: {e}")

def _write_fetch_log(conn, table_name, stock_id, status, rows_inserted=0, fetch_date_from=None, fetch_date_to=None, duration_ms=0, error_message=None):
    """標準化 fetch_log 寫入"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fetch_log (
                    run_ts, table_name, stock_id, fetch_mode, status, rows_inserted, 
                    fetch_date_from, fetch_date_to, duration_ms, error_message, cli_args
                ) VALUES (NOW(), %s, %s, 'parallel_launcher', %s, %s, %s, %s, %s, %s, %s)
            """, (table_name, stock_id, status, rows_inserted, fetch_date_from, fetch_date_to, duration_ms, error_message, _CLI_ARGS_STR))
        conn.commit()
    except Exception as e:
        logger.warning(f"無法寫入 parallel 執行日誌: {e}")


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
    start_ts = time.time()
    
    try:
        proc = subprocess.run(
            cmd, check=False, timeout=MAX_SCRIPT_TIMEOUT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        duration_ms = int((time.time() - start_ts) * 1000)
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
        _write_fetch_log(conn, script_name, stock_id, "success" if success else "failed", duration_ms=dur_ms, error_message=err if not success else None)
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
    _upgrade_schema(conn) # ⭐ 執行欄位擴充
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