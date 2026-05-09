"""
fetch_missing_stocks_data.py — 自動補齊新增標的之歷史數據 (v4.2 路徑類型修正版)
================================================================================
v4.2 改進：
  ★ 修復 `TypeError`: 在呼叫 `atomic_write_json` 前將 `Path` 物件顯式轉為 `str`，
    解決與 `model_metadata.py` 的相容性問題。

v4.1 既有：
  ★ 導入 `core.path_setup` 統一處理路徑。
  ★ 標準化 `fetch_log`：將子任務狀態、耗時與 CLI 參數落盤。
  ★ 整合 Checkpoint：支援 `--resume` 續做模式。

執行範例：
    python scripts/fetchers/fetch_missing_stocks_data.py --stock-id 2330 --force
"""

from __future__ import annotations

import sys
import logging
import subprocess
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path, get_outputs_dir, get_checkpoints_dir
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    FailureLogger,
    write_fetch_log,
    DDL_FETCH_LOG
)
from core.model_metadata import atomic_write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 常數與配置
# ─────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
VENV_PYTHON = sys.executable
SUBPROCESS_TIMEOUT = 3600

PER_STOCK_SCRIPTS = {
    "fetch_technical_data.py":      "--stock-id",
    "fetch_price_adj_data.py":      "--stock-id",
    "fetch_fundamental_data.py":    "--stock-id",
    "fetch_sponsor_chip_data.py":   "--stock-id",
    "fetch_event_risk_data.py":     "--stock-id",
}

MACRO_SCRIPTS = [
    "fetch_macro_data.py",
    "fetch_fred_data.py",
    "fetch_macro_fundamental_data.py",
    "fetch_total_return_index.py",
    "fetch_international_data.py",
    "fetch_derivative_sentiment_data.py",
]

_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────

def get_missing_data_manifest() -> list[dict]:
    manifest_path = get_outputs_dir() / "integrity_gaps.json"
    try:
        from data_integrity_audit import IntegrityAuditor
        auditor = IntegrityAuditor(days_window=1000)
        auditor.dump_gaps_json(str(manifest_path))
    except ImportError:
        logger.warning("未安裝 data_integrity_audit，讀取現有報告。")

    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"讀取審計報告失敗：{e}")
    return []

def run_script(script_name: str, args: list[str], timeout: int) -> tuple[bool, int, int, str]:
    cmd = [VENV_PYTHON, str(SCRIPTS_DIR / script_name), *args]
    logger.info(f"🚀 啟動子任務: {script_name} {' '.join(args)}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, check=False, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        duration_ms = int((time.time() - t0) * 1000)
        stderr_tail = (proc.stderr or "").strip().split("\n")[-3:]
        return (proc.returncode == 0), proc.returncode, duration_ms, "\n".join(stderr_tail)
    except subprocess.TimeoutExpired:
        return False, -1, int((time.time() - t0) * 1000), f"Timeout > {timeout}s"
    except Exception as e:
        return False, -1, int((time.time() - t0) * 1000), str(e)

# ─────────────────────────────────────────────
# Checkpoint 管理 (修復處)
# ─────────────────────────────────────────────
def _ckpt_path() -> Path:
    return get_checkpoints_dir() / "fetch_missing_stocks.json"

def _load_checkpoint() -> set[tuple[str, str]]:
    p = _ckpt_path()
    if not p.exists(): return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {(d["stock_id"], d["script"]) for d in data.get("done", [])}
    except Exception: return set()

def _save_checkpoint(done_set: set[tuple[str, str]]):
    payload = {
        "updated_at": datetime.now().isoformat(),
        "done": [{"stock_id": sid, "script": scr} for (sid, scr) in done_set]
    }
    # ⭐ 修正：顯式轉為 str 解決 atomic_write_json 類別衝突 ⭐
    atomic_write_json(str(_ckpt_path()), payload)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="自動補齊個股資料庫 (v4.2 — 修正路徑相容性)")
    p.add_argument("--stock-id", help="指定單一標的 ID (例如 2330)")
    p.add_argument("--force", action="store_true", help="強制重新抓取")
    p.add_argument("--resume", action="store_true", help="續做模式")
    p.add_argument("--skip-stock-info", action="store_true")
    p.add_argument("--skip-macro", action="store_true")
    p.add_argument("--timeout", type=int, default=SUBPROCESS_TIMEOUT)
    args = p.parse_args()

    conn = get_db_conn()
    flog = FailureLogger("missing_stocks_pipeline", db_conn=conn)
    
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        done_set = _load_checkpoint() if args.resume else set()

        if not args.skip_stock_info:
            ok, rc, dur, err = run_script("fetch_stock_info.py", [], args.timeout)
            write_fetch_log(conn, table_name="fetch_stock_info.py", stock_id="SYSTEM", status="success" if ok else "failed", duration_ms=dur, error_message=err, fetch_mode="subprocess")

        target_map: dict[str, str] = {}
        if args.stock_id:
            target_map[args.stock_id] = "1994-10-01" if args.force else "2024-01-01"
        else:
            gaps = get_missing_data_manifest()
            for g in gaps:
                sid, s_date = g["stock_id"], g["gap_start"]
                if sid not in target_map or s_date < target_map[sid]:
                    target_map[sid] = s_date

        for sid, start_date in target_map.items():
            for script, flag in PER_STOCK_SCRIPTS.items():
                if (sid, script) in done_set: continue
                
                run_args = [flag, sid, "--start", start_date]
                if args.force: run_args.append("--force")
                
                ok, rc, dur, err = run_script(script, run_args, args.timeout)
                write_fetch_log(conn, table_name=script, stock_id=sid, status="success" if ok else "failed", fetch_date_from=start_date, duration_ms=dur, error_message=err, fetch_mode="subprocess")
                
                if ok:
                    done_set.add((sid, script))
                    _save_checkpoint(done_set)
                else:
                    flog.record(stock_id=sid, script=script, error=err, duration=dur)

        if not args.skip_macro:
            for script in MACRO_SCRIPTS:
                if ("MACRO", script) in done_set: continue
                ok, rc, dur, err = run_script(script, [], args.timeout)
                write_fetch_log(conn, table_name=script, stock_id="MACRO", status="success" if ok else "failed", duration_ms=dur, error_message=err, fetch_mode="subprocess")
                if ok:
                    done_set.add(("MACRO", script))
                    _save_checkpoint(done_set)
                else:
                    flog.record(stock_id="MACRO", script=script, error=err, duration=dur)

    finally:
        flog.summary()
        conn.close()
        logger.info("\n✨ 補件任務全數完成。")

if __name__ == "__main__":
    main()