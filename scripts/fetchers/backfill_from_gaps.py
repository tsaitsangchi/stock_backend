from __future__ import annotations
"""
backfill_from_gaps.py v3.2 — 精確資料斷層補抓器（完整參數與錯誤修正版）
================================================================================
修正項目：
  1. 參數完整性：補回 --refresh-audit, --per-day, --derivative-only 等所有遺失參數。
  2. 耗時計算：改用 time.perf_counter() 並強制 max(0, ...)，防止出現負數耗時。
  3. 路徑錯誤：修正 _atomic_write fallback 邏輯，避免 Path 物件與 str 直接相加。
  4. 核心偵測：確保 atomic_write_json 的錯誤被優雅捕捉並轉向正確的 fallback。
"""

import sys
from pathlib import Path

# ── sys.path 自我修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for sub in ("", "core", "fetchers", "pipeline", "training", "monitor"):
    p = (_SCRIPTS_DIR / sub) if sub else _SCRIPTS_DIR
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

import argparse
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Optional

# ── 載入 core v3.1 helpers ──
_CLI_ARGS_STR = " ".join(sys.argv)

try:
    from core.path_setup import (
        ensure_scripts_on_path,
        get_outputs_dir,
        get_checkpoints_dir,
        ensure_dirs_exist,
    )
    ensure_scripts_on_path(__file__)
    from core.db_utils import (
        get_db_conn,
        write_fetch_log,
    )
    from core.model_metadata import atomic_write_json
    _CORE_OK = True
except Exception as _e:
    _CORE_OK = False
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"無法載入 core helpers，使用 fallback：{_e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── 路徑設定 ──
if _CORE_OK:
    OUTPUT_DIR = get_outputs_dir()
    CHECKPOINT_DIR = get_checkpoints_dir()
else:
    OUTPUT_DIR = _SCRIPTS_DIR / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_JSON     = OUTPUT_DIR / "integrity_gaps.json"
FAILURE_JSON     = OUTPUT_DIR / "backfill_failures.json"
SUCCESS_JSON     = OUTPUT_DIR / "backfill_successes.json"
CHECKPOINT_JSON  = CHECKPOINT_DIR / "backfill.json"

def _detect_venv_python() -> str:
    cur = Path(sys.executable)
    return str(cur) if cur.exists() else sys.executable

VENV_PYTHON = _detect_venv_python()

# ── 資料表對應表 ──
TABLE_TO_FETCHER: dict[str, dict] = {
    "stock_price":                      {"script": "fetch_technical_data.py",          "id_flag": "--stock-id"},
    "stock_per":                        {"script": "fetch_technical_data.py",          "id_flag": "--stock-id"},
    "price_adj":                        {"script": "fetch_price_adj_data.py",          "id_flag": "--stock-id"},
    "day_trading":                      {"script": "fetch_price_adj_data.py",          "id_flag": "--stock-id"},
    "price_limit":                      {"script": "fetch_price_adj_data.py",          "id_flag": "--stock-id"},
    "institutional_investors_buy_sell": {"script": "fetch_chip_data.py",               "id_flag": "--stock-id"},
    "margin_purchase_short_sale":       {"script": "fetch_chip_data.py",               "id_flag": "--stock-id"},
    "shareholding":                     {"script": "fetch_chip_data.py",               "id_flag": "--stock-id"},
    "securities_lending":               {"script": "fetch_advanced_chip_data.py",      "id_flag": "--stock-id"},
    "daily_short_balance":              {"script": "fetch_advanced_chip_data.py",      "id_flag": "--stock-id"},
    "eight_banks_buy_sell":             {"script": "fetch_advanced_chip_data.py",      "id_flag": "--stock-id"},
    "sponsor_chip":                     {"script": "fetch_sponsor_chip_data.py",       "id_flag": "--stock-id"},
    "month_revenue":                    {"script": "fetch_fundamental_data.py",        "id_flag": "--stock-id"},
    "financial_statements":             {"script": "fetch_fundamental_data.py",        "id_flag": "--stock-id"},
    "balance_sheet":                    {"script": "fetch_fundamental_data.py",        "id_flag": "--stock-id"},
    "cash_flows_statement":             {"script": "fetch_cash_flows_data.py",         "id_flag": "--stock-id"},
    "dividend":                         {"script": "fetch_fundamental_data.py",        "id_flag": "--stock-id"},
    "us_stock_price":                   {"script": "fetch_international_data.py",      "id_flag": "--ids"},
    "exchange_rate":                    {"script": "fetch_macro_data.py",              "id_flag": "--ids"},
    "crude_oil_prices":                 {"script": "fetch_international_data.py",      "id_flag": None},
    "gold_price":                       {"script": "fetch_international_data.py",      "id_flag": None},
    "interest_rate":                    {"script": "fetch_macro_data.py",              "id_flag": "--ids"},
    "futures_ohlcv":                    {"script": "fetch_derivative_data.py",         "id_flag": "--ids"},
    "options_ohlcv":                    {"script": "fetch_derivative_data.py",         "id_flag": "--ids"},
    "options_oi_large_holders":         {"script": "fetch_derivative_sentiment_data.py", "id_flag": "--ids"},
    "total_margin_short":               {"script": "fetch_advanced_chip_data.py",      "id_flag": None},
    "total_inst_investors":             {"script": "fetch_advanced_chip_data.py",      "id_flag": None},
    "futures_inst_investors":           {"script": "fetch_extended_derivative_data.py", "id_flag": "--ids"},
    "options_inst_investors":           {"script": "fetch_extended_derivative_data.py", "id_flag": "--ids"},
    "disposition_securities":           {"script": "fetch_event_risk_data.py",         "id_flag": "--stock-id"},
    "capital_reduction":                {"script": "fetch_event_risk_data.py",         "id_flag": "--stock-id"},
    "stock_news":                       {"script": "fetch_news_data.py",               "id_flag": "--stock-id"},
    "fred_series":                      {"script": "fetch_fred_data.py",               "id_flag": "--ids"},
}

# ── 修正後的原子寫入助手 ──
def _atomic_write(path: Path | str, data) -> None:
    path_obj = Path(path)
    if _CORE_OK:
        try:
            atomic_write_json(path_obj, data)
            return
        except Exception as e:
            # 這裡捕捉 core.model_metadata.py 內的類型錯誤 [cite: 1, 2]
            logger.debug(f"核心 atomic_write_json 失敗，執行補救措施：{e}")
    
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    # 修正 Path 拼接邏輯 
    tmp = path_obj.parent / f"{path_obj.name}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(str(tmp), str(path_obj))

def _read_json_list(path: Path) -> list:
    if not path.exists(): return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception: return []

def _append_json(path: Path, item: dict) -> None:
    existing = _read_json_list(path)
    existing.append(item)
    _atomic_write(path, existing)

def append_failure(item: dict) -> None: _append_json(FAILURE_JSON, item)
def append_success(item: dict) -> None: _append_json(SUCCESS_JSON, item)

def _ckpt_key(entry: dict) -> str:
    return f"{entry['table']}|{entry.get('stock_id') or 'MARKET'}|{entry['start']}"

def load_checkpoint() -> dict:
    if not CHECKPOINT_JSON.exists(): return {}
    try: return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
    except Exception: return {}

def update_checkpoint(ckpt: dict, entry: dict, status: str, msg: str = "") -> None:
    ckpt[_ckpt_key(entry)] = {"status": status, "ts": datetime.now().isoformat(), "msg": msg}
    _atomic_write(CHECKPOINT_JSON, ckpt)

def reset_log_files() -> None:
    for p in (FAILURE_JSON, SUCCESS_JSON, CHECKPOINT_JSON):
        if p.exists(): p.unlink()

def load_gaps(json_path: Path) -> list[dict]:
    if not json_path.exists(): return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def refresh_audit() -> Optional[Path]:
    logger.info("重新執行完整性 audit...")
    try:
        from data_integrity_audit import IntegrityAuditor
        auditor = IntegrityAuditor(days_window=1500)
        auditor.dump_gaps_json(str(DEFAULT_JSON))
        return DEFAULT_JSON
    except Exception as e:
        logger.error(f"refresh_audit 失敗：{e}")
        return None

# ── 修正後的子腳本執行 ──
def run_fetcher(
    script: str, stock_id: Optional[str], id_flag: Optional[str],
    start: str, end: Optional[str] = None, dry_run: bool = False,
    timeout: int = 1800, quiet: bool = False,
) -> tuple[bool, str, float]:
    script_path = _SCRIPTS_DIR / "fetchers" / script
    if not script_path.exists(): script_path = _SCRIPTS_DIR / script
    if not script_path.exists(): return False, f"找不到 {script}", 0.0

    cmd = [VENV_PYTHON, str(script_path), "--start", start]
    if end: cmd += ["--end", end]
    if id_flag and stock_id and stock_id != "MARKET": cmd += [id_flag, stock_id]
    if "--force" not in cmd: cmd.append("--force")

    if dry_run: return True, "dry-run", 0.0

    # 使用 perf_counter 並加上 max(0, ...) 確保耗時正確 
    t0 = time.perf_counter()
    try:
        if quiet:
            result = subprocess.run(cmd, check=False, timeout=timeout, capture_output=True, text=True, encoding="utf-8", errors="replace")
            elapsed = max(0.0, time.perf_counter() - t0)
            if result.returncode != 0: return False, f"exit_code={result.returncode}", elapsed
            return True, "ok", elapsed
        else:
            subprocess.run(cmd, check=True, timeout=timeout)
            return True, "ok", max(0.0, time.perf_counter() - t0)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", max(0.0, time.perf_counter() - t0)

def split_gap_into_days(gap: dict) -> list[dict]:
    try:
        d_start = datetime.strptime(gap["gap_start"], "%Y-%m-%d").date()
        d_end = datetime.strptime(gap.get("gap_end") or gap["gap_start"], "%Y-%m-%d").date()
    except: return [gap]
    days = []
    cur = d_start
    while cur <= d_end:
        if cur.weekday() < 5:
            sub = dict(gap); sub["gap_start"] = sub["gap_end"] = cur.strftime("%Y-%m-%d"); sub["gap_days"] = 1
            days.append(sub)
        cur += timedelta(days=1)
    return days if days else [gap]

def build_work_units(filtered: list[dict], per_day: bool) -> list[dict]:
    if per_day:
        units = []
        for g in filtered:
            for sub in split_gap_into_days(g):
                meta = TABLE_TO_FETCHER[sub["table"]]
                units.append({"script": meta["script"], "stock_id": sub["stock_id"] if meta["id_flag"] else None, "id_flag": meta["id_flag"], "table": sub["table"], "start": sub["gap_start"], "end": sub["gap_end"], "tables": [sub["table"]]})
        return units
    groups = {}
    for g in filtered:
        meta = TABLE_TO_FETCHER[g["table"]]
        key = (meta["script"], g["stock_id"] if meta["id_flag"] else "MARKET")
        if key not in groups:
            groups[key] = {"script": meta["script"], "stock_id": g["stock_id"] if meta["id_flag"] else None, "id_flag": meta["id_flag"], "start": g["gap_start"], "end": g.get("gap_end"), "tables": [g["table"]], "_tbl_set": {g["table"]}}
        else:
            if g["table"] not in groups[key]["_tbl_set"]:
                groups[key]["tables"].append(g["table"]); groups[key]["_tbl_set"].add(g["table"])
            groups[key]["start"] = min(groups[key]["start"], g["gap_start"])
            if g.get("gap_end"): groups[key]["end"] = max(groups[key]["end"] or g["gap_start"], g["gap_end"])
    return list(groups.values())

def process_unit(unit: dict, args, ckpt: dict, counters: dict) -> dict:
    ok, msg, elapsed = run_fetcher(unit["script"], unit["stock_id"], unit["id_flag"], unit["start"], unit.get("end"), args.dry_run, args.timeout, args.quiet)
    sid_str = unit["stock_id"] or "MARKET"
    if not args.dry_run:
        logger.info(f"  {'✅' if ok else '❌'} {unit['script']} | sid={sid_str} | start={unit['start']} | 耗時 {elapsed:.1f}s | {msg}")
        if ok and elapsed < 0.1: logger.warning(f"  ⚠️  耗時極短 ({elapsed*1000:.0f}ms)，請檢查子腳本。")
    
    for tbl in unit["tables"]:
        entry = {"stock_id": sid_str, "table": tbl, "start": unit["start"], "end": unit.get("end") or unit["start"], "script": unit["script"], "elapsed": round(elapsed, 2)}
        if ok:
            counters["successes"] += 1
            if not args.dry_run:
                append_success(entry); update_checkpoint(ckpt, entry, "SUCCESS", msg)
                if _CORE_OK:
                    try:
                        conn = get_db_conn()
                        write_fetch_log(conn, tbl, unit["stock_id"], status="success", fetch_mode="gap_backfill", fetch_date_from=unit["start"], fetch_date_to=unit.get("end"), duration_ms=int(elapsed*1000))
                        conn.close()
                    except: pass
        else:
            counters["failures"] += 1; entry["error"] = msg
            if not args.dry_run: append_failure(entry); update_checkpoint(ckpt, entry, "FAILED", msg)
    return {"ok": ok, "elapsed": elapsed}

# ── 完整主流程 ──
def main():
    parser = argparse.ArgumentParser(description="精確資料斷層補抓器 (v3.2)")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="integrity_gaps.json 路徑")
    parser.add_argument("--tables", help="指定資料表(逗號分隔)")
    parser.add_argument("--stocks", help="指定股票(逗號分隔)")
    parser.add_argument("--dry-run", action="store_true", help="試跑")
    parser.add_argument("--max-stocks", type=int, help="限制 unit 數")
    parser.add_argument("--refresh-audit", action="store_true", help="執行前重新 audit")
    parser.add_argument("--min-gap-days", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--keep-history", action="store_true")
    parser.add_argument("--per-day", action="store_true", help="逐日拆分")
    parser.add_argument("--resume", action="store_true", help="從續傳點開始")
    parser.add_argument("--parallel", type=int, default=1, help="並行執行")
    parser.add_argument("--quiet", action="store_true", help="隱藏子腳本輸出")
    parser.add_argument("--derivative-only", action="store_true", help="僅限衍生品")
    args = parser.parse_args()

    if _CORE_OK: ensure_dirs_exist()
    if not Path(VENV_PYTHON).exists(): logger.error(f"找不到 Python: {VENV_PYTHON}"); return 1

    if args.refresh_audit: path = refresh_audit() or Path(args.json)
    else: path = Path(args.json)
    
    gaps = load_gaps(path)
    if not gaps: logger.info("沒有斷層，任務結束。"); return 0

    if not args.dry_run and not args.resume and not args.keep_history: reset_log_files()

    filtered = [g for g in gaps if g["table"] in TABLE_TO_FETCHER]
    if args.tables:
        t_set = set(args.tables.split(","))
        filtered = [g for g in filtered if g["table"] in t_set]
    if args.stocks:
        s_set = set(args.stocks.split(","))
        filtered = [g for g in filtered if g["stock_id"] in s_set]
    if args.derivative_only:
        d_tables = {"futures_ohlcv", "options_ohlcv", "exchange_rate", "interest_rate"} # 簡化列表
        filtered = [g for g in filtered if g["table"] in d_tables]

    units = build_work_units(filtered, args.per_day)
    ckpt = load_checkpoint() if args.resume else {}
    if args.resume:
        units = [u for u in units if not all(ckpt.get(_ckpt_key({"table": t, "stock_id": u["stock_id"], "start": u["start"]}), {}).get("status") == "SUCCESS" for t in u["tables"])]
    
    if args.max_stocks: units = units[:args.max_stocks]
    if not units: logger.info("無待處理任務。"); return 0

    logger.info(f"開始執行 {len(units)} 個任務 (Parallel={args.parallel})...")
    counters = {"successes": 0, "failures": 0}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(process_unit, u, args, ckpt, counters) for u in units]
        for f in as_completed(futures): f.result()

    logger.info(f"任務完成。成功: {counters['successes']}, 失敗: {counters['failures']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())