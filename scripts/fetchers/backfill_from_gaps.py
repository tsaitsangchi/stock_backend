from __future__ import annotations
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]:
    sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))

"""
backfill_from_gaps.py — 精確資料斷層補抓器（P0-1 / QW-1）
=====================================================================
讀取 outputs/integrity_gaps.json，按 (stock_id, table) 精確補抓
267 個資料斷層，避免 fetch_missing_stocks_data.py 的「整支股票全量重抓」
浪費 API 配額。

設計重點：
  - 只針對「真正缺資料的 (stock_id, table)」呼叫對應 fetcher
  - --start 起點精確對齊 gap_start，不額外重抓已有資料
  - 支援 --dry-run 試跑、--max-stocks 限制每次處理量
  - 失敗自動跳過並寫入 outputs/backfill_failures.json
  - 透過 fetch_log 表持久化結果，data_integrity_audit 第 6 維度可追蹤

執行：
    python backfill_from_gaps.py                            # 補抓全部斷層
    python backfill_from_gaps.py --json outputs/integrity_gaps.json
    python backfill_from_gaps.py --tables stock_per,price_adj
    python backfill_from_gaps.py --stocks 2330,2317
    python backfill_from_gaps.py --dry-run                  # 試跑
    python backfill_from_gaps.py --refresh-audit            # 補抓前先重跑 audit
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
DEFAULT_JSON = OUTPUT_DIR / "integrity_gaps.json"
FAILURE_JSON = OUTPUT_DIR / "backfill_failures.json"

VENV_PYTHON = os.environ.get(
    "VENV_PYTHON",
    str(BASE_DIR.parent / "venv" / "bin" / "python3"),
)


# ─────────────────────────────────────────────
# (table → fetcher script, id flag) 對應表
# 精確對應「補抓單表」所需的最小 fetcher 集合
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
    # 國際 / 衍生品（多為 market-level，無 stock_id）
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
    """執行 data_integrity_audit.IntegrityAuditor.dump_gaps_json 重新產生清單。"""
    logger.info("先執行完整性 audit 重新偵測斷層…")
    try:
        # 使用內部呼叫避免 subprocess 環境變數問題
        from data_integrity_audit import IntegrityAuditor
        auditor = IntegrityAuditor(days_window=1500)
        auditor.dump_gaps_json(str(DEFAULT_JSON))
        return DEFAULT_JSON
    except Exception as e:
        logger.error(f"refresh_audit 失敗：{e}")
        return None


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
    """
    呼叫 fetchers/{script}，傳入 --start {start}（與，若有 stock_id，--stock-id {stock_id}）。

    Returns:
        (success, message)
    """
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
# 寫入 fetch_log（與 parallel_fetch 對齊）
# ─────────────────────────────────────────────

def write_fetch_log(table_name: str, stock_id: Optional[str],
                    start: str, status: str, message: str) -> None:
    """將補抓結果寫入 fetch_log 表，data_integrity_audit 第 6 維度復活。"""
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
    parser = argparse.ArgumentParser(description="按 (stock_id, table) 精確補抓資料斷層")
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
                        help="只補抓 gap_days >= 此值的斷層（預設 1，避免微小延遲噪音）")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="單一 fetcher 最長執行秒數（預設 1800）")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 65)
    logger.info("  資料斷層精確補抓 (P0-1 / QW-1)")
    logger.info(f"  模式：{'DRY-RUN' if args.dry_run else '正式執行'}")
    logger.info("=" * 65)

    # ── 1) 取得最新 gap 清單 ─────────────────────────────────
    if args.refresh_audit:
        path = refresh_audit() or Path(args.json)
    else:
        path = Path(args.json)
    gaps = load_gaps(path)
    if not gaps:
        logger.info("沒有偵測到任何斷層，提前結束 ✨")
        return 0

    # ── 2) 套用篩選條件 ───────────────────────────────────────
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

    # ── 3) 依 (table → 最早 gap_start) 分組，避免重複呼叫同 fetcher ─
    # 例如同個 fetcher 同支股票多個表都有 gap，取最早起點補一次
    groups: dict[tuple[str, str], dict] = {}
    for g in filtered:
        meta = TABLE_TO_FETCHER[g["table"]]
        # key = (script, stock_id or 'MARKET')
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

    # ── 4) 逐組執行 ──────────────────────────────────────────
    if args.max_stocks:
        # 簡化版：限制總筆數
        groups = dict(list(groups.items())[: args.max_stocks])

    successes, failures = [], []
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
            if ok:
                successes.append({"stock_id": info["stock_id"] or "MARKET",
                                  "table": tbl, "start": info["earliest"]})
                write_fetch_log(tbl, info["stock_id"], info["earliest"], "SUCCESS", msg)
            else:
                failures.append({"stock_id": info["stock_id"] or "MARKET",
                                 "table": tbl, "start": info["earliest"], "error": msg})
                write_fetch_log(tbl, info["stock_id"], info["earliest"], "FAILED", msg)

    # ── 5) 摘要 + 失敗清單寫檔 ───────────────────────────────
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 65)
    logger.info(f"  補抓完成！總耗時 {elapsed:.1f} 秒")
    logger.info(f"  成功：{len(successes):,}  失敗：{len(failures):,}")
    logger.info("=" * 65)

    if failures and not args.dry_run:
        FAILURE_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILURE_JSON, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"  失敗清單已寫入：{FAILURE_JSON}")

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
