"""
backfill_from_gaps.py v5.1 (Trinity Core Edition)
================================================================================
精確資料斷層自動補抓器 — 完美對接 core/ 五大核心模組
此模組負責讀取審計報告 (integrity_gaps.json)，並自動調用對應的抓取器進行資料補漏。

核心功能：
  · 自動化審計對接 ─ 讀取資料完整性報告，精確識別各資料表的時間斷層。
  · 任務並行調度   ─ 支援多執行緒並發補件，加速歷史數據的回補效率。
  · 續傳檢查機制   ─ 支援斷點續傳，避免對相同時間區間的重複請求。

對接核心模組 (scripts/core/)：
  · db_utils v4.6            ─ 連線池 + 任務日誌落盤 (fetch_log)
  · finmind_client v5.1      ─ Singleton + 智慧斷路器與配額管理
  · path_setup v3.0          ─ 跨環境路徑自動導引

修訂歷程：
  v5.1 (2026-05-09):
    - [核心] 全面對接 Trinity Core v5.1，使用 db_session 與 FinMindClient 單例。
    - [監控] 統一採用 write_fetch_log 紀錄補件狀態，標記模式為 gap_v5.1。
    - [穩定] 強化 subprocess 調用安全性，優化原子寫入邏輯以防止 JSON 損壞。
  v3.2 (2026-04-30):
    - [修復] 補回遺失參數並修正耗時計算邏輯。

執行範例：
    # 範例 1：根據審計報告自動補齊所有斷層 (並行 4 執行緒)
    python scripts/fetchers/backfill_from_gaps.py --parallel 4
    
    # 範例 2：僅針對特定資料表進行逐日精確補件
    python scripts/fetchers/backfill_from_gaps.py --tables stock_price --per-day
    
    # 範例 3：重新執行審計後立即開始補件
    python scripts/fetchers/backfill_from_gaps.py --refresh-audit --parallel 4
"""

from __future__ import annotations
import sys
import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

# ── 系統路徑修復 (對接 path_setup v3.0) ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.path_setup import ensure_scripts_on_path, get_outputs_dir, get_checkpoints_dir, ensure_dirs_exist
    ensure_scripts_on_path(__file__)
    from core.db_utils import db_session, write_fetch_log, get_db_stock_ids
    from core.finmind_client import FinMindClient
    from core.model_metadata import atomic_write_json
except ImportError as e:
    print(f"[FATAL] 無法匯入 core 模組: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 路徑設定 ──
OUTPUT_DIR = get_outputs_dir()
CHECKPOINT_DIR = get_checkpoints_dir()
DEFAULT_JSON     = OUTPUT_DIR / "integrity_gaps.json"
FAILURE_JSON     = OUTPUT_DIR / "backfill_failures.json"
SUCCESS_JSON     = OUTPUT_DIR / "backfill_successes.json"
CHECKPOINT_JSON  = CHECKPOINT_DIR / "backfill.json"

VENV_PYTHON = sys.executable

# ── 資料表對應表 (v5.1 更新版) ──
TABLE_TO_FETCHER: Dict[str, Dict] = {
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
    "dividend_result":                  {"script": "fetch_cash_flows_data.py",         "id_flag": "--stock-id"},
    "us_stock_price":                   {"script": "fetch_international_data.py",      "id_flag": "--ids"},
    "exchange_rate":                    {"script": "fetch_macro_data.py",              "id_flag": "--ids"},
    "futures_ohlcv":                    {"script": "fetch_derivative_data.py",         "id_flag": "--ids"},
    "options_ohlcv":                    {"script": "fetch_derivative_data.py",         "id_flag": "--ids"},
    "options_oi_large_holders":         {"script": "fetch_derivative_sentiment_data.py", "id_flag": "--ids"},
    "disposition_securities":           {"script": "fetch_event_risk_data.py",         "id_flag": "--stock-id"},
    "capital_reduction":                {"script": "fetch_event_risk_data.py",         "id_flag": "--stock-id"},
    "stock_news":                       {"script": "fetch_news_data.py",               "id_flag": "--stock-id"},
    "fred_series":                      {"script": "fetch_fred_data.py",               "id_flag": "--ids"},
}

# =====================================================================
# 1. JSON 持久化助手
# =====================================================================

def _atomic_write(path: Path, data) -> None:
    try: atomic_write_json(path, data)
    except:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(path)

def _append_json(path: Path, item: dict) -> None:
    existing = []
    if path.exists():
        try: existing = json.loads(path.read_text(encoding="utf-8"))
        except: pass
    existing.append(item)
    _atomic_write(path, existing)

def update_checkpoint(ckpt: dict, entry: dict, status: str, msg: str = "") -> None:
    key = f"{entry['table']}|{entry.get('stock_id') or 'MARKET'}|{entry['start']}"
    ckpt[key] = {"status": status, "ts": datetime.now().isoformat(), "msg": msg}
    _atomic_write(CHECKPOINT_JSON, ckpt)

# =====================================================================
# 2. 核心補件邏輯
# =====================================================================

def run_fetcher(script: str, sid: Optional[str], flag: Optional[str], start: str, end: str) -> Tuple[bool, str, float]:
    script_path = _THIS_DIR / script
    cmd = [VENV_PYTHON, str(script_path), "--start", start, "--end", end, "--force"]
    if flag and sid and sid != "MARKET": cmd += [flag, sid]
    
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, check=False, timeout=1800, capture_output=True, text=True, encoding="utf-8")
        elapsed = max(0.0, time.perf_counter() - t0)
        return (proc.returncode == 0), "ok" if proc.returncode == 0 else proc.stderr.strip().split("\n")[-1], elapsed
    except Exception as e:
        return False, str(e), max(0.0, time.perf_counter() - t0)

def process_unit(unit: dict, args, ckpt: dict, counters: dict) -> None:
    ok, msg, elapsed = run_fetcher(unit["script"], unit["stock_id"], unit["id_flag"], unit["start"], unit["end"])
    sid_str = unit["stock_id"] or "MARKET"
    logger.info(f"  {'✅' if ok else '❌'} {unit['script']} | sid={sid_str} | {unit['start']} ~ {unit['end']} | {elapsed:.1f}s")

    for tbl in unit["tables"]:
        entry = {"stock_id": sid_str, "table": tbl, "start": unit["start"], "end": unit["end"], "script": unit["script"]}
        if ok:
            counters["successes"] += 1
            _append_json(SUCCESS_JSON, entry)
            update_checkpoint(ckpt, entry, "SUCCESS")
            write_fetch_log(tbl, unit["stock_id"], "success", "gap_v5.1", unit["start"], unit["end"], int(elapsed*1000))
        else:
            counters["failures"] += 1
            entry["error"] = msg
            _append_json(FAILURE_JSON, entry)
            update_checkpoint(ckpt, entry, "FAILED", msg)
            write_fetch_log(tbl, unit["stock_id"], "failed", "gap_v5.1", unit["start"], unit["end"], int(elapsed*1000), 0, msg)

def main():
    parser = argparse.ArgumentParser(description="Precision Gap Backfiller v5.1 (Trinity Core Edition)")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tables", type=str, default="")
    parser.add_argument("--refresh-audit", action="store_true")
    parser.add_argument("--per-day", action="store_true")
    args = parser.parse_args()

    ensure_dirs_exist()
    if args.refresh_audit:
        try:
            from data_integrity_audit import IntegrityAuditor
            auditor = IntegrityAuditor(days_window=1000)
            auditor.dump_gaps_json(str(DEFAULT_JSON))
        except Exception as e: logger.error(f"審計重新整理失敗: {e}")

    if not DEFAULT_JSON.exists():
        logger.warning("找不到 gaps.json，請先執行審計。")
        return

    gaps = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    filtered = [g for g in gaps if g["table"] in TABLE_TO_FETCHER]
    
    if args.tables:
        t_set = set(args.tables.split(","))
        filtered = [g for g in filtered if g["table"] in t_set]

    # 建構工作單元
    units = []
    for g in filtered:
        meta = TABLE_TO_FETCHER[g["table"]]
        units.append({
            "script": meta["script"], "stock_id": g["stock_id"] if meta["id_flag"] else None,
            "id_flag": meta["id_flag"], "table": g["table"], "start": g["gap_start"], 
            "end": g.get("gap_end") or g["gap_start"], "tables": [g["table"]]
        })

    ckpt = {}
    if args.resume and CHECKPOINT_JSON.exists():
        ckpt = json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
        units = [u for u in units if f"{u['table']}|{u['stock_id'] or 'MARKET'}|{u['start']}" not in ckpt]

    if not units:
        logger.info("無待處理補件任務。")
        return

    logger.info("=" * 70)
    logger.info(f"  Gap Backfiller Pipeline v5.1  ({datetime.now():%H:%M:%S})")
    logger.info(f"  待處理任務: {len(units)}  (Parallel: {args.parallel})")
    logger.info("=" * 70)

    counters = {"successes": 0, "failures": 0}
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(process_unit, u, args, ckpt, counters) for u in units]
        for f in as_completed(futures): f.result()

    logger.info(f"🎉 全部完成。成功: {counters['successes']}, 失敗: {counters['failures']}")

if __name__ == "__main__":
    main()