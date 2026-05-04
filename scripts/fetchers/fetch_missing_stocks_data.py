"""
fetch_missing_stocks_data.py v4.0 — 自動補齊 config.py 中新增個股的歷史數據
[Trinity Edition × core v3.0 完整性版]

對齊 core v3.0「逐支逐日 commit」精神：
本檔本身為 subprocess launcher，不直接寫 DB；真正的逐支逐日落盤由各被
呼叫之 fetcher（fetch_technical_data / fetch_chip_data / ...）內部的
commit_per_stock_per_day 完成。

v4.0 改進（呼應 core v3.0 完整性原則）：
  ★ 改用 core.path_setup 統一 sys.path 設定，移除手寫多重 path block
  ★ 新增 FailureLogger：追蹤每支個股 × 每支 fetcher 子任務的成敗
  ★ 新增 fetch_log 寫入：每次 subprocess 呼叫的 status/duration 直接落 DB
  ★ subprocess 呼叫加 timeout 保護（預設 1 小時）+ 子任務失敗不阻斷其他標的
  ★ run_script() 回傳 (success, returncode, duration, stderr_tail)，方便診斷
  ★ checkpoint：完成的 (sid, script) 寫入 checkpoint 檔，--resume 可續做
  ★ 結尾統計：印出每支 fetcher 的成功率與最慢的 5 個任務

修改摘要：
1. 強化缺失偵測：從單純檢查 stock_price (count < 100)，升級為多表聯動校驗。
2. 引入效能機制：利用 data_integrity_check 的邏輯，精確識別需要補件的標的。
3. 智能補抓觸發：核心表任一出現嚴重缺漏，即觸發該標的的全量補抓。
4. v4.0：每支 fetcher 子任務的成敗都被獨立 record，不再「跑了就忘」。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path 自我修復（與 core v3.0 對齊）──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core", "fetchers", "monitor"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)

import argparse
import json
import logging
import subprocess
import time
from collections import defaultdict
from datetime import date

# 載入 core v3.0 helpers（fallback 友善）
try:
    from core.path_setup import (
        ensure_scripts_on_path, get_outputs_dir, get_checkpoints_dir, ensure_dirs_exist,
    )
    ensure_scripts_on_path(__file__)
    from core.db_utils import (
        get_db_conn, ensure_ddl, log_fetch_result, FailureLogger,
    )
    from core.model_metadata import atomic_write_json
    _CORE_OK = True
except Exception as _e:
    _CORE_OK = False
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(
        f"無法載入 core v3.0 helpers，使用 fallback：{_e}"
    )

from config import STOCK_CONFIGS  # noqa: E402

# data_integrity_audit 為可選相依（單股模式不需要）
try:
    from data_integrity_audit import IntegrityAuditor
    _AUDIT_OK = True
except Exception:
    _AUDIT_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 路徑與常數
# ─────────────────────────────────────────────
SCRIPTS_DIR = _THIS_DIR  # = scripts/fetchers
BASE_DIR = SCRIPTS_DIR.parent  # = scripts
VENV_PYTHON = str(BASE_DIR.parent / "venv" / "bin" / "python3")
SUBPROCESS_TIMEOUT = 60 * 60  # 1 hour per fetcher

# 個股相關補件腳本及其對應的 ID 參數名稱
PER_STOCK_SCRIPTS_CONFIG = {
    "fetch_technical_data.py":      "--stock-id",
    "fetch_price_adj_data.py":      "--stock-id",
    "fetch_fundamental_data.py":    "--stock-id",
    "fetch_chip_data.py":           "--stock-id",
    "fetch_advanced_chip_data.py":  "--stock-id",
    "fetch_sponsor_chip_data.py":   "--stock-id",
    "fetch_event_risk_data.py":     "--stock-id",
    "fetch_cash_flows_data.py":     "--stock-id",
}

# 全域宏觀資料腳本（不分股票）
MACRO_SCRIPTS = [
    "fetch_macro_data.py",
    "fetch_fred_data.py",
    "fetch_macro_fundamental_data.py",
    "fetch_total_return_index.py",
    "fetch_extended_derivative_data.py",
    "fetch_derivative_sentiment_data.py",
    "fetch_derivative_data.py",
    "fetch_international_data.py",
]


def _get_outputs_dir() -> Path:
    if _CORE_OK:
        return get_outputs_dir()
    p = BASE_DIR / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_checkpoints_dir() -> Path:
    if _CORE_OK:
        return get_checkpoints_dir()
    p = BASE_DIR / "outputs" / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────
# subprocess 包裝（v4.0：含 timeout、duration、stderr 截斷）
# ─────────────────────────────────────────────
def run_script(
    script_name: str,
    extra_args: list[str] | None = None,
    timeout: int = SUBPROCESS_TIMEOUT,
) -> tuple[bool, int, float, str]:
    """
    執行一支 fetcher。回傳 (success, returncode, duration_sec, stderr_tail)。
    任何例外都被吃掉，回傳 success=False，由呼叫端決定是否寫入 FailureLogger。
    """
    extra_args = extra_args or []
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        return False, -127, 0.0, f"找不到腳本: {script_path}"

    cmd = [VENV_PYTHON, str(script_path), *extra_args]
    t0 = time.monotonic()
    logger.info(f"🚀 {script_name} {' '.join(extra_args)}")
    try:
        proc = subprocess.run(
            cmd, check=False, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        duration = time.monotonic() - t0
        success = proc.returncode == 0
        stderr_tail = (proc.stderr or "").strip().split("\n")[-5:]
        return success, proc.returncode, duration, "\n".join(stderr_tail)[-500:]
    except subprocess.TimeoutExpired:
        return False, -1, time.monotonic() - t0, f"timeout > {timeout}s"
    except Exception as e:
        return False, -1, time.monotonic() - t0, str(e)[:500]


# ─────────────────────────────────────────────
# 完整性審計
# ─────────────────────────────────────────────
def get_missing_data_manifest() -> list[dict]:
    """執行深度審計並取得斷層清單。"""
    if not _AUDIT_OK:
        logger.warning("data_integrity_audit 不可用，跳過自動掃描")
        return []
    out_dir = _get_outputs_dir()
    manifest_path = out_dir / "integrity_gaps.json"

    auditor = IntegrityAuditor(days_window=1000)
    auditor.dump_gaps_json(str(manifest_path))

    if not manifest_path.exists():
        return []
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"讀取 integrity_gaps.json 失敗：{e}")
        return []


# ─────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────
def _ckpt_path() -> Path:
    return _get_checkpoints_dir() / "fetch_missing_stocks.json"


def _load_checkpoint() -> set[tuple[str, str]]:
    """回傳已完成的 (stock_id, script_name) 集合。"""
    p = _ckpt_path()
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {(d["stock_id"], d["script"]) for d in data.get("done", [])}
    except Exception as e:
        logger.warning(f"checkpoint 讀取失敗：{e}")
        return set()


def _save_checkpoint(done: set[tuple[str, str]]) -> None:
    payload = {
        "updated_at": date.today().isoformat(),
        "done": [{"stock_id": s, "script": k} for (s, k) in done],
    }
    if _CORE_OK:
        atomic_write_json(_ckpt_path(), payload)
    else:
        _ckpt_path().parent.mkdir(parents=True, exist_ok=True)
        _ckpt_path().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="自動補齊或強制更新個股資料庫 v4.0")
    parser.add_argument("--stock-id", help="指定單一標的 ID（例如 2330）")
    parser.add_argument("--force", action="store_true", help="強制重新抓取所有腳本（不論是否有斷層）")
    parser.add_argument("--resume", action="store_true",
                        help="從上次 checkpoint 續做，已完成的 (stock_id, script) 跳過")
    parser.add_argument("--skip-stock-info", action="store_true",
                        help="不執行 fetch_stock_info.py（適合多次呼叫期間避免重複）")
    parser.add_argument("--skip-macro", action="store_true",
                        help="不執行 macro 階段")
    parser.add_argument("--timeout", type=int, default=SUBPROCESS_TIMEOUT,
                        help=f"單一 fetcher 子任務 timeout 秒數（預設 {SUBPROCESS_TIMEOUT}）")
    args = parser.parse_args()

    if _CORE_OK:
        ensure_dirs_exist()

    logger.info("=" * 60)
    logger.info("    Phase 2 精確補件管線（v4.0 / core v3.0 對齊）啟動")
    logger.info("=" * 60)

    # ── failure logger / checkpoint ──
    flog = None
    db_conn = None
    if _CORE_OK:
        try:
            db_conn = get_db_conn()
            flog = FailureLogger("fetch_missing_stocks", db_conn=db_conn, log_to_db=True)
        except Exception as e:
            logger.warning(f"FailureLogger 初始化失敗（將僅輸出至檔案）：{e}")
            flog = FailureLogger("fetch_missing_stocks") if _CORE_OK else None

    done_set: set[tuple[str, str]] = _load_checkpoint() if args.resume else set()
    if args.resume and done_set:
        logger.info(f"[resume] 已完成 {len(done_set)} 個 (stock_id, script) 將被跳過")

    # ── Step 1：基礎資訊（除非 --skip-stock-info）──
    if not args.skip_stock_info:
        logger.info("Step 1: 更新 stock_info ...")
        ok, rc, dur, err = run_script("fetch_stock_info.py", timeout=args.timeout)
        if not ok and flog is not None:
            flog.record(stock_id="STOCK_INFO", script="fetch_stock_info.py",
                        returncode=rc, duration=dur, error=err)
        if db_conn is not None:
            try:
                log_fetch_result(
                    db_conn, "fetch_stock_info.py", "SYSTEM",
                    date.today().isoformat(), date.today().isoformat(),
                    0, "SUCCESS" if ok else "FAILED", err if not ok else None,
                )
            except Exception:
                pass

    # ── Step 2：決定 target_stocks dict[sid -> start_date] ──
    target_stocks: dict[str, str] = {}
    if args.stock_id:
        logger.info(f"模式: 指定單一標的 {args.stock_id}")
        start_date = "1994-10-01" if args.force else "2024-01-01"
        target_stocks[args.stock_id] = start_date
    else:
        gap_manifest = get_missing_data_manifest()
        if not gap_manifest:
            logger.info("✅ 所有標的核心資料皆已完整，無需執行補件。")
        else:
            logger.info(f"發現 {len(gap_manifest)} 處資料斷層，啟動精確補件...")
            grouped = defaultdict(list)
            for gap in gap_manifest:
                grouped[gap["stock_id"]].append(gap)
            for sid, gaps in grouped.items():
                target_stocks[sid] = min(g["gap_start"] for g in gaps)

    # ── Step 3：逐支股票 × 逐個 fetcher 執行（記錄每個子任務）──
    durations: list[tuple[str, str, float]] = []  # (sid, script, duration)
    success_by_script: dict[str, int] = defaultdict(int)
    total_by_script: dict[str, int] = defaultdict(int)

    for sid, start_date in target_stocks.items():
        logger.info(f"\n>>> 處理 {sid}（起點：{start_date}）")
        for script, id_flag in PER_STOCK_SCRIPTS_CONFIG.items():
            if (sid, script) in done_set:
                logger.info(f"  [skip-resume] {sid} / {script}")
                continue
            run_args = [id_flag, sid, "--start", start_date]
            if args.force:
                run_args.append("--force")

            ok, rc, dur, err = run_script(script, run_args, timeout=args.timeout)
            durations.append((sid, script, dur))
            total_by_script[script] += 1
            if ok:
                success_by_script[script] += 1
                done_set.add((sid, script))
                _save_checkpoint(done_set)
            else:
                if flog is not None:
                    flog.record(stock_id=sid, script=script,
                                returncode=rc, duration=round(dur, 2), error=err)

            # fetch_log 寫入（即使失敗）
            if db_conn is not None:
                try:
                    log_fetch_result(
                        db_conn, script, sid,
                        start_date, date.today().isoformat(),
                        0, "SUCCESS" if ok else "FAILED", err if not ok else None,
                    )
                except Exception:
                    pass

    # ── Step 4：宏觀資料更新（除非 --skip-macro）──
    if not args.skip_macro:
        logger.info("\nStep 4: 更新全域宏觀資料 ...")
        for script in MACRO_SCRIPTS:
            ok, rc, dur, err = run_script(script, timeout=args.timeout)
            durations.append(("MACRO", script, dur))
            total_by_script[script] += 1
            if ok:
                success_by_script[script] += 1
            elif flog is not None:
                flog.record(stock_id="MACRO", script=script,
                            returncode=rc, duration=round(dur, 2), error=err)
            if db_conn is not None:
                try:
                    log_fetch_result(
                        db_conn, script, "MACRO",
                        date.today().isoformat(), date.today().isoformat(),
                        0, "SUCCESS" if ok else "FAILED", err if not ok else None,
                    )
                except Exception:
                    pass

    # ── 統計摘要 ──
    logger.info("\n" + "=" * 60)
    logger.info("執行摘要（成功率）")
    logger.info("=" * 60)
    for script in sorted(total_by_script.keys()):
        s = success_by_script[script]
        t = total_by_script[script]
        pct = 100.0 * s / t if t else 0
        logger.info(f"  {script:<40s} {s}/{t}  ({pct:5.1f}%)")

    if durations:
        durations.sort(key=lambda x: -x[2])
        logger.info("\n最慢 5 個子任務：")
        for sid, script, dur in durations[:5]:
            logger.info(f"  {sid:<8s} {script:<40s} {dur:7.1f}s")

    if flog is not None:
        flog.summary()
    if db_conn is not None:
        try:
            db_conn.close()
        except Exception:
            pass

    logger.info("\n✨ 任務全數完成。")


if __name__ == "__main__":
    main()
