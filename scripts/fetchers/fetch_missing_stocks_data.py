"""
fetch_missing_stocks_data.py v5.1 (Trinity Core Edition)
================================================================================
自動補齊新增標的歷史數據管線 — 完美對接 core/ 五大核心模組
此模組為自動化維運工具，負責根據數據完整性報告 (integrity_gaps.json)，補齊新標的之歷史數據。

核心功能：
  · 自動補件管線     ─ 依序執行價量、還原、基本面、籌碼面、事件風險等補件任務。
  · 斷路器相容       ─ 子任務腳本 (v5.1) 內建斷路器保護，防止因單一標的錯誤導致管線潰縮。
  · 續做模式 (Checkpoint) ─ 紀錄已完成的任務，支援意外中斷後的智慧續做。
  · 非法 ID 跳過     ─ 繼承子腳本之過濾特性，自動過濾 Automobile 等無效 ID。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 事務原子性 + 筆數追蹤
  · model_metadata v3.0      ─ 原子化 JSON 讀寫 (Checkpoint 管理)
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 移除 get_db_conn，全面改用 db_session 與 db_transaction。
    - [整合] 確保子任務呼叫對接 v5.1 系列之 CLI 參數規範。
    - [路徑] 對接 path_setup v3.0，自動定位 outputs 與 checkpoints 目錄。
  v4.2 (2024-05-01):
    - [修復] 修正 Path 物件轉 JSON 的相容性問題。

執行範例：
    # 範例 1：全自動補齊所有缺失數據
    python scripts/fetchers/fetch_missing_stocks_data.py
    
    # 範例 2：從上次中斷處續做
    python scripts/fetchers/fetch_missing_stocks_data.py --resume
    
    # 範例 3：針對特定新標的強制補齊歷史數據 (1994 起)
    python scripts/fetchers/fetch_missing_stocks_data.py --stock-id 6861 --force
"""

import sys
import argparse
import logging
import subprocess
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple, Set

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path, get_outputs_dir, get_checkpoints_dir
    ensure_scripts_on_path(__file__)
    from core.db_utils import (
        db_session, db_transaction, ensure_ddl, write_fetch_log, FailureLogger
    )
    from core.model_metadata import atomic_write_json
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. 常數與腳本配置
# =====================================================================

VENV_PYTHON = sys.executable
SUBPROCESS_TIMEOUT = 3600

PER_STOCK_SCRIPTS = {
    "fetch_technical_data.py":      "--stock-id",
    "fetch_price_adj_data.py":      "--stock-id",
    "fetch_fundamental_data.py":    "--stock-id",
    "fetch_sponsor_chip_data.py":   "--stock-id",
    "fetch_event_risk_data.py":     "--stock-id",
    "fetch_chip_data.py":           "--stock-id",
    "fetch_advanced_chip_data.py":  "--stock-id",
}

MACRO_SCRIPTS = [
    "fetch_macro_data.py",
    "fetch_fred_data.py",
    "fetch_macro_fundamental_data.py",
    "fetch_total_return_index.py",
    "fetch_international_data.py",
    "fetch_derivative_sentiment_data.py",
]

# =====================================================================
# 2. 工具函式
# =====================================================================

def run_script(script_name: str, args: List[str], timeout: int) -> Tuple[bool, int, int, str]:
    cmd = [VENV_PYTHON, str(_THIS_DIR / script_name), *args]
    logger.info(f"🚀 啟動子任務: {script_name} {' '.join(args)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=False, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration_ms = int((time.time() - t0) * 1000)
        stderr_tail = (proc.stderr or "").strip().split("\n")[-3:]
        return (proc.returncode == 0), proc.returncode, duration_ms, "\n".join(stderr_tail)
    except subprocess.TimeoutExpired:
        return False, -1, int((time.time() - t0) * 1000), f"Timeout > {timeout}s"
    except Exception as e:
        return False, -1, int((time.time() - t0) * 1000), str(e)

def _ckpt_path() -> Path:
    return get_checkpoints_dir() / "fetch_missing_stocks.json"

def _load_checkpoint() -> Set[Tuple[str, str]]:
    p = _ckpt_path()
    if not p.exists(): return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {(d["stock_id"], d["script"]) for d in data.get("done", [])}
    except: return set()

def _save_checkpoint(done_set: Set[Tuple[str, str]]):
    payload = {
        "updated_at": datetime.now().isoformat(),
        "done": [{"stock_id": sid, "script": scr} for (sid, scr) in done_set]
    }
    atomic_write_json(str(_ckpt_path()), payload)

def get_missing_data_manifest() -> List[dict]:
    manifest_path = get_outputs_dir() / "integrity_gaps.json"
    if manifest_path.exists():
        try: return json.loads(manifest_path.read_text(encoding="utf-8"))
        except: pass
    return []

# =====================================================================
# 3. Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="自動補齊缺失數據 v5.1 (Trinity Core Edition)")
    parser.add_argument("--stock-id", help="指定單一標的 ID")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=SUBPROCESS_TIMEOUT)
    args = parser.parse_args()

    fail_log = FailureLogger("missing_stocks_pipeline")
    done_set = _load_checkpoint() if args.resume else set()

    # 1. 更新基礎股票清單
    ok, rc, dur, err = run_script("fetch_stock_info.py", [], args.timeout)
    write_fetch_log("stocks", "SYSTEM", "success" if ok else "failed", "subprocess", None, None, dur, 0, err)

    # 2. 決定補件目標
    target_map = {}
    if args.stock_id:
        target_map[args.stock_id] = "1994-10-01" if args.force else "2024-01-01"
    else:
        gaps = get_missing_data_manifest()
        for g in gaps:
            sid, s_date = g["stock_id"], g["gap_start"]
            if sid not in target_map or s_date < target_map[sid]:
                target_map[sid] = s_date

    # 3. 執行個股補件
    for sid, start_date in target_map.items():
        for script, flag in PER_STOCK_SCRIPTS.items():
            if (sid, script) in done_set: continue
            
            run_args = [flag, sid, "--start", start_date]
            if args.force: run_args.append("--force")
            
            ok, rc, dur, err = run_script(script, run_args, args.timeout)
            write_fetch_log(script, sid, "success" if ok else "failed", "subprocess", start_date, None, dur, 0, err)
            
            if ok:
                done_set.add((sid, script))
                _save_checkpoint(done_set)
            else:
                fail_log.log_failure(script, sid, start_date, None, err)

    # 4. 執行宏觀數據補件
    for script in MACRO_SCRIPTS:
        if ("MACRO", script) in done_set: continue
        ok, rc, dur, err = run_script(script, [], args.timeout)
        write_fetch_log(script, "MACRO", "success" if ok else "failed", "subprocess", None, None, dur, 0, err)
        if ok:
            done_set.add(("MACRO", script))
            _save_checkpoint(done_set)
        else:
            fail_log.log_failure(script, "MACRO", None, None, err)

    logger.info("✨ 補件任務全數完成。")

if __name__ == "__main__":
    main()