from __future__ import annotations
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]:
    sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))

"""
backfill_from_gaps.py — 精確資料斷層補抓器（v2.2 逐筆 append failure 完整性版）
=====================================================================
讀取 outputs/integrity_gaps.json，按 (stock_id, table) 精確補抓資料斷層，
避免 fetch_missing_stocks_data.py 的「整支股票全量重抓」浪費 API 配額。

v2.2 改進：
  · 失敗清單 append 寫檔（每筆失敗即時落盤），子腳本中途崩潰仍可保留進度。
  · 子腳本失敗時也呼叫 write_fetch_log，補上「正在抓但未完成」的訊號。
  · 子腳本本身已升級為「逐支 commit」設計（v2.2 系列），相當於
    本檔同時取得「(table, stock_id) 層級」與「(stock_id) 層級」雙重完整性保護。

執行：
    python backfill_from_gaps.py
    python backfill_from_gaps.py --json outputs/integrity_gaps.json
    python backfill_from_gaps.py --tables stock_per,price_adj
    python backfill_from_gaps.py --stocks 2330,2317
    python backfill_from_gaps.py --dry-run
    python backfill_from_gaps.py --refresh-audit
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_JSON = OUTPUT_DIR / "integrity_gaps.json"
FAILURE_JSON = OUTPUT_DIR / "backfill_failures.json"
SUCCESS_JSON = OUTPUT_DIR / "backfill_successes.json"

VENV_PYTHON = os.environ.get(
    "VENV_PYTHON",
    str(BASE_DIR.parent / "venv" / "bin" / "python3"),
)


# ─────────────────────────────────────────────
# (table → fetcher script, id flag) 對應表
# ─────────────────────────────────────────────
TABLE_TO_FETCHER: dict[str, dict] = {
    # 核心價量
    "stock_price":                      {"script": "fetch_technical_data.py",          "id_flag": "--stock-id"},
    "stock_per":                        {"script": "fetch_technical_data.py",          "id_flag": "--stock-id"},
    "price_adj":                        {"script": "fetch_price_adj_data.py",          "id_flag": "--stock-id"},
    "day_trading":                      {"script": "fetch_price_adj_data.py",          "id_flag": "--stock-id"},
    "price_limit":                      {"script": "fetch_price_adj_data.py",          "id_flag": "--stock-id"},
    # 籌碼面
    "institutional_investors_buy_sell": {"script": "fetch_chip_data.py",               "id_flag": "--stock-id"},
    "margin_purchase_short_sale":       {"script": "fetch_chip_data.py",               "id_flag": "--stock-id"},
    "shareholding":                     {"script": "fetch_chip_data.py",               "id_flag": "--stock-id"},
    "securities_lending":               {"script": "fetch_advanced_chip_data.py",      "id_flag": "--stock-id"},
    "daily_short_balance":              {"script": "fetch_advanced_chip_data.py",      "id_flag": "--stock-id"},
    "eight_banks_buy_sell":             {"script": "fetch_advanced_chip_data.py",      "id_flag": "--stock-id"},
    "sponsor_chip":                     {"script": "fetch_sponsor_chip_data.py",       "id_flag": "--stock-id"},
    # 基本面
    "month_revenue":                    {"script": "fetch_fundamental_data.py",        "id_flag": "--stock-id"},
    "financial_statements":             {"script": "fetch_fundamental_data.py",        "id_flag": "--stock-id"},
    "balance_sheet":                    {"script": "fetch_fundamental_data.py",        "id_flag": "--stock-id"},
    "cash_flows_statement":             {"script": "fetch_cash_flows_data.py",         "id_flag": "--stock-id"},
    "dividend":                         {"script": "fetch_fundamental_data.py",        "id_flag": "--stock-id"},
    # 國際 / 衍生品
    "us_stock_price":                   {"script": "fetch_international_data.py",      "id_flag": None},
    "exchange_rate":                    {"script": "fetch_international_data.py",      "id_flag": None},
    "interest_rate":                    {"script": "fetch_macro_data.py",              "id_flag": None},
    "futures_ohlcv":                    {"script": "fetch_derivative_data.py",         "id_flag": None},
    "options_ohlcv":                    {"script": "fetch_derivative_data.py",         "id_flag": None},
    "options_oi_large_holders":         {"script": "fetch_derivative_sentiment_data.py", "id_flag": None},
    # 市場層級
    "total_margin_short":               {"script": "fetch_advanced_chip_data.py",      "id_flag": None},
    "total_inst_investors":             {"script": "fetch_advanced_chip_data.py",      "id_flag": None},
    "futures_inst_investors":           {"script": "fetch_extended_derivative_data.py", "id_flag": None},
    "options_inst_investors":           {"script": "fetch_extended_derivative_data.py", "id_flag": None},
    # 事件 / 另類
    "disposition_securities":           {"script": "fetch_event_risk_data.py",         "id_flag": "--stock-id"},
    "capital_reduction":                {"script": "fetch_event_risk_data.py",         "id_flag": "--stock-id"},
    "stock_news":                       {"script": "fetch_news_data.py",               "id_flag": "--stock-id"},
    "fred_series":                      {"script": "fetch_fred_data.py",               "id_flag": None},
}


# ─────────────────────────────────────────────
# 載入斷層清單
# ─────────────────────────────────────────────
def load_gaps(json_path: Path) -> list[dict]:
    if not json_path.exists():
        logger.error(f"找不到 {json_path}，請先執行 data_integrity_audit 產生。")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        gaps = json.load(f)
    logger.info(f"從 {json_path} 載入 {len(gaps):,} 處斷層")
    return gaps


def refresh_audit() -> Optional[Path]:
    logger.info("先執行完整性 audit 重新偵測斷層…")
    try:
        from data_integrity_audit import IntegrityAuditor
        auditor = IntegrityAuditor(days_window=1500)
        auditor.dump_gaps_json(str(DEFAULT_JSON))
        return DEFAULT_JSON
    except Exception as e:
        logger.error(f"refresh_audit 失敗：{e}")
        return None


# ─────────────────────────────────────────────
# 失敗 / 成功清單即時 append 寫檔
# ─────────────────────────────────────────────
def _append_json(path: Path, item: dict) -> None:
    try:
        existing: list = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.append(item)
        path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"append {path.name} 失敗：{e}")


def append_failure(item: dict) -> None:
    _append_json(FAILURE_JSON, item)


def append_success(item: dict) -> None:
    _append_json(SUCCESS_JSON, item)


def reset_log_files() -> None:
    """每次啟動補抓任務前，把上次的 success/failure 清單清空，避免混淆。"""
    for p in (FAILURE_JSON, SUCCESS_JSON):
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            logger.warning(f"清空 {p.name} 失敗：{e}")


# ─────────────────────────────────────────────
# 子腳本執行
# ─────────────────────────────────────────────
def run_fetcher(
    script: str,
    stock_id: Optional[str],
    id_flag: Optional[str],
    start: str,
    dry_run: bool = False,
    timeout: int = 1800,
) -> tuple[bool, str]:
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        return False, f"找不到 {script_path}"

    cmd = [VENV_PYTHON, str(script_path), "--start", start]
    if id_flag and stock_id:
        cmd += [id_flag, stock_id]

    pretty = " ".join(cmd[1:])
    if dry_run:
        logger.info(f"[DRY-RUN] {pretty}")
        return True, "dry-run"

    logger.info(f"🚀 {pretty}")
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, f"timeout({timeout}s)"
    except subprocess.CalledProcessError as e:
        return False, f"exit_code={e.returncode}"
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# 寫入 fetch_log
# ─────────────────────────────────────────────
def write_fetch_log(table_name: str, stock_id: Optional[str],
                    start: str, status: str, message: str) -> None:
    try:
        from core.db_utils import get_db_conn, ensure_ddl, log_fetch_result
        conn = get_db_conn()
        try:
            ensure_ddl(conn)
            log_fetch_result(
                conn,
                table_name=table_name,
                stock_id=stock_id or "MARKET",
                start_date=start,
                end_date=date.today().strftime("%Y-%m-%d"),
                rows_count=0,
                status=status,
                error_msg=message if status == "FAILED" else None,
            )
        finally:
            conn.close()
    except Exception as e:
        logger.debug(f"fetch_log 寫入失敗（不影響主流程）：{e}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="按 (stock_id, table) 精確補抓資料斷層 v2.2")
    parser.add_argument("--json", default=str(DEFAULT_JSON),
                        help=f"integrity_gaps.json 路徑（預設 {DEFAULT_JSON}）")
    parser.add_argument("--tables", default=None,
                        help="只處理指定資料表，多筆用逗號分隔")
    parser.add_argument("--stocks", default=None,
                        help="只處理指定股票，多筆用逗號分隔")
    parser.add_argument("--dry-run", action="store_true", help="試跑，不實際呼叫 fetcher")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="限制每張表最多處理多少支股票（節流 API 配額）")
    parser.add_argument("--refresh-audit", action="store_true",
                        help="補抓前先跑 data_integrity_audit 重新產生 integrity_gaps.json")
    parser.add_argument("--min-gap-days", type=int, default=1,
                        help="只補抓 gap_days >= 此值的斷層（預設 1）")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="單一 fetcher 最長執行秒數（預設 1800）")
    parser.add_argument("--keep-history", action="store_true",
                        help="保留前次 success/failure 紀錄（預設啟動時清空）")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 65)
    logger.info("  資料斷層精確補抓 v2.2 (P0-1 / QW-1)")
    logger.info(f"  模式：{'DRY-RUN' if args.dry_run else '正式執行'}")
    logger.info("=" * 65)

    # ── 0) 重置 success/failure 清單（除非 --keep-history） ──
    if not args.dry_run and not args.keep_history:
        reset_log_files()

    # ── 1) 取得最新 gap 清單 ───────────────────
    if args.refresh_audit:
        path = refresh_audit() or Path(args.json)
    else:
        path = Path(args.json)
    gaps = load_gaps(path)
    if not gaps:
        logger.info("沒有偵測到任何斷層，提前結束 ✨")
        return 0

    # ── 2) 套用篩選條件 ───────────────────────
    table_filter = set(args.tables.split(",")) if args.tables else None
    stock_filter = set(args.stocks.split(",")) if args.stocks else None

    filtered: list[dict] = []
    for g in gaps:
        if g.get("gap_days", 0) < args.min_gap_days:
            continue
        if table_filter and g["table"] not in table_filter:
            continue
        if stock_filter and g["stock_id"] not in stock_filter:
            continue
        if g["table"] not in TABLE_TO_FETCHER:
            logger.warning(f"未知資料表 {g['table']}，跳過")
            continue
        filtered.append(g)

    logger.info(f"套用篩選後共 {len(filtered):,} 筆斷層待補抓")
    if not filtered:
        logger.info("篩選後無任何斷層，結束。")
        return 0

    # ── 3) 依 (table → 最早 gap_start) 分組 ────
    groups: dict[tuple[str, str], dict] = {}
    for g in filtered:
        meta = TABLE_TO_FETCHER[g["table"]]
        key = (meta["script"], g["stock_id"] if meta["id_flag"] else "MARKET")
        if key not in groups:
            groups[key] = {
                "script":   meta["script"],
                "stock_id": g["stock_id"] if meta["id_flag"] else None,
                "id_flag":  meta["id_flag"],
                "earliest": g["gap_start"],
                "tables":   [g["table"]],
            }
        else:
            groups[key]["tables"].append(g["table"])
            if g["gap_start"] < groups[key]["earliest"]:
                groups[key]["earliest"] = g["gap_start"]

    # ── 4) 逐組執行 ───────────────────────────
    if args.max_stocks:
        groups = dict(list(groups.items())[: args.max_stocks])

    successes_count, failures_count = 0, 0
    for i, ((script, sid_key), info) in enumerate(groups.items(), 1):
        logger.info(f"\n[{i}/{len(groups)}] {script} | sid={sid_key} | "
                    f"start={info['earliest']} | tables={info['tables']}")
        ok, msg = run_fetcher(
            script=info["script"],
            stock_id=info["stock_id"],
            id_flag=info["id_flag"],
            start=info["earliest"],
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
        for tbl in info["tables"]:
            entry = {
                "stock_id": info["stock_id"] or "MARKET",
                "table":    tbl,
                "start":    info["earliest"],
                "script":   info["script"],
            }
            if ok:
                successes_count += 1
                if not args.dry_run:
                    append_success(entry)
                    write_fetch_log(tbl, info["stock_id"], info["earliest"], "SUCCESS", msg)
            else:
                entry["error"] = msg
                failures_count += 1
                if not args.dry_run:
                    append_failure(entry)
                    write_fetch_log(tbl, info["stock_id"], info["earliest"], "FAILED", msg)

    # ── 5) 摘要 ────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 65)
    logger.info(
        f"  完成：{len(groups)} 組  |  成功 {successes_count}，失敗 {failures_count}  |  "
        f"耗時 {elapsed:.1f}s"
    )
    if failures_count and not args.dry_run:
        logger.info(f"  失敗清單：{FAILURE_JSON}")
    if successes_count and not args.dry_run:
        logger.info(f"  成功清單：{SUCCESS_JSON}")

    # ── 6) 自動同步至 Trinity 健康度矩陣 ─────────
    if successes_count and not args.dry_run:
        try:
            from sync_trinity_db import sync_health_matrix
            logger.info("正在將最新補抓結果同步至全系統健康度矩陣...")
            sync_health_matrix()
            logger.info("✅ 健康度矩陣同步完成。")
        except Exception as e:
            logger.warning(f"自動同步失敗（不影響補抓）：{e}")

    return 0 if not failures_count else 2


if __name__ == "__main__":
    sys.exit(main())