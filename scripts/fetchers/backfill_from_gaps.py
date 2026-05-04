from __future__ import annotations
"""
backfill_from_gaps.py v3.0 — 精確資料斷層補抓器（逐支逐日 commit 完整性版）
==============================================================================
讀取 outputs/integrity_gaps.json，按 (table, stock_id) 精確補抓資料斷層。
搭配 core v3.0 的「逐支逐日 commit」設計，確保任何中斷都能保留已完成資料。

v3.0 改進（與 core v3.0 完整對齊）：
  ★ 改用 core.db_utils.FailureLogger / append_failure_json — 失敗即時原子落盤
  ★ 改用 core.model_metadata.atomic_write_json — success/failure 清單 tmp+rename 寫入
  ★ 改用 core.path_setup.get_outputs_dir / get_checkpoints_dir — 統一路徑
  ★ 新增 --per-day 模式：每日獨立 fetcher 呼叫，達到 (table, stock_id, day) 最細粒度
  ★ 新增 --resume 模式：讀取上次 successes，跳過已完成的 (table, stock_id, start)
  ★ 新增 --parallel N 模式：並行執行 N 個 fetcher subprocess（仍受全域速率限制）
  ★ 新增 checkpoint 檔（outputs/checkpoints/backfill.json）— 中斷時保留進度
  ★ 每個 (table, stock_id, gap) 獨立寫 fetch_log，崩潰前的所有完成項都已落地
  ★ 結尾印出統計摘要（成功/失敗/略過/平均耗時）

執行：
    python backfill_from_gaps.py
    python backfill_from_gaps.py --tables stock_per,price_adj
    python backfill_from_gaps.py --stocks 2330,2317
    python backfill_from_gaps.py --dry-run
    python backfill_from_gaps.py --refresh-audit
    python backfill_from_gaps.py --per-day                # 逐日呼叫（最細粒度）
    python backfill_from_gaps.py --resume                 # 從上次 successes 續做
    python backfill_from_gaps.py --parallel 4             # 4 條 fetcher 並行
"""

import sys
from pathlib import Path

# ── sys.path 自我修復 ──
_THIS_DIR = Path(__file__).resolve().parent       # scripts/ 或 scripts/<sub>/
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

# ── 載入 core v3.0 helpers ──
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
        ensure_ddl,
        log_fetch_result,
        FailureLogger,
        append_failure_json,
        get_failure_log_path,
    )
    from core.model_metadata import atomic_write_json
    _CORE_OK = True
except Exception as _e:
    # core 模組還沒部署的退化路徑
    _CORE_OK = False
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(
        f"無法載入 core v3.0 helpers，使用 fallback：{_e}"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 路徑設定（優先用 core path_setup；否則 fallback）
# ─────────────────────────────────────────────
SCRIPTS_DIR = _SCRIPTS_DIR
BASE_DIR = SCRIPTS_DIR.parent

if _CORE_OK:
    OUTPUT_DIR = get_outputs_dir()
    CHECKPOINT_DIR = get_checkpoints_dir()
else:
    OUTPUT_DIR = SCRIPTS_DIR / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_JSON     = OUTPUT_DIR / "integrity_gaps.json"
FAILURE_JSON     = OUTPUT_DIR / "backfill_failures.json"
SUCCESS_JSON     = OUTPUT_DIR / "backfill_successes.json"
CHECKPOINT_JSON  = CHECKPOINT_DIR / "backfill.json"

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
# Atomic JSON helpers（優先用 model_metadata.atomic_write_json，否則 fallback）
# ─────────────────────────────────────────────
def _atomic_write(path: Path, data) -> None:
    if _CORE_OK:
        try:
            atomic_write_json(path, data)
            return
        except Exception as e:
            logger.warning(f"atomic_write_json 失敗，改用 fallback：{e}")
    # fallback
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _append_json(path: Path, item: dict) -> None:
    """即時 append 至 JSON 清單（每筆都原子寫入）。"""
    existing = _read_json_list(path)
    existing.append(item)
    _atomic_write(path, existing)


def append_failure(item: dict) -> None:
    _append_json(FAILURE_JSON, item)


def append_success(item: dict) -> None:
    _append_json(SUCCESS_JSON, item)


# ─────────────────────────────────────────────
# Checkpoint（中斷時保留進度，下次 --resume 可續做）
# ─────────────────────────────────────────────
def _ckpt_key(entry: dict) -> str:
    """生成單一 (table, stock_id, start) 的 checkpoint key。"""
    return f"{entry['table']}|{entry.get('stock_id') or 'MARKET'}|{entry['start']}"


def load_checkpoint() -> dict:
    """讀取 checkpoint：{ key: { 'status': 'SUCCESS'/'FAILED', 'ts': ... } }"""
    if not CHECKPOINT_JSON.exists():
        return {}
    try:
        d = json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_checkpoint(ckpt: dict) -> None:
    _atomic_write(CHECKPOINT_JSON, ckpt)


def update_checkpoint(ckpt: dict, entry: dict, status: str, msg: str = "") -> None:
    """單筆即時 checkpoint，崩潰前的進度都已落地。"""
    ckpt[_ckpt_key(entry)] = {
        "status": status,
        "ts": datetime.now().isoformat(),
        "msg": msg,
    }
    # 即時落盤（每筆都寫一次，性能 OK 因為單檔不大）
    save_checkpoint(ckpt)


def reset_log_files() -> None:
    """每次啟動補抓任務前，把上次的 success/failure/checkpoint 清單清空。"""
    for p in (FAILURE_JSON, SUCCESS_JSON, CHECKPOINT_JSON):
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            logger.warning(f"清空 {p.name} 失敗：{e}")


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
# 子腳本執行
# ─────────────────────────────────────────────
def run_fetcher(
    script: str,
    stock_id: Optional[str],
    id_flag: Optional[str],
    start: str,
    end: Optional[str] = None,
    dry_run: bool = False,
    timeout: int = 1800,
) -> tuple[bool, str, float]:
    """
    呼叫指定 fetcher 子腳本。
    Returns (success, message, elapsed_sec)
    """
    script_path = SCRIPTS_DIR / "fetchers" / script
    # fallback：scripts/script.py（舊路徑）
    if not script_path.exists():
        script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        return False, f"找不到 {script}", 0.0

    cmd = [VENV_PYTHON, str(script_path), "--start", start]
    if end:
        cmd += ["--end", end]
    if id_flag and stock_id:
        cmd += [id_flag, stock_id]

    pretty = " ".join(cmd[1:])
    if dry_run:
        logger.info(f"[DRY-RUN] {pretty}")
        return True, "dry-run", 0.0

    logger.info(f"🚀 {pretty}")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        return True, "ok", time.time() - t0
    except subprocess.TimeoutExpired:
        return False, f"timeout({timeout}s)", time.time() - t0
    except subprocess.CalledProcessError as e:
        return False, f"exit_code={e.returncode}", time.time() - t0
    except Exception as e:
        return False, str(e), time.time() - t0


# ─────────────────────────────────────────────
# 寫入 fetch_log（per-entry 即時 commit）
# ─────────────────────────────────────────────
def write_fetch_log(table_name: str, stock_id: Optional[str],
                    start: str, end: Optional[str],
                    status: str, message: str) -> None:
    """每筆 (table, stock_id, start, end) 獨立寫 fetch_log，立即 commit。"""
    if not _CORE_OK:
        return
    try:
        conn = get_db_conn()
        try:
            ensure_ddl(conn)
            log_fetch_result(
                conn,
                table_name=table_name,
                stock_id=stock_id or "MARKET",
                start_date=start,
                end_date=end or date.today().strftime("%Y-%m-%d"),
                rows_count=0,
                status=status,
                error_msg=message if status == "FAILED" else None,
            )
        finally:
            conn.close()
    except Exception as e:
        logger.debug(f"fetch_log 寫入失敗（不影響主流程）：{e}")


# ─────────────────────────────────────────────
# 將多日 gap 拆成「逐日工作項」（--per-day 模式用）
# ─────────────────────────────────────────────
def split_gap_into_days(gap: dict) -> list[dict]:
    """
    把單一 gap (gap_start ~ gap_end) 拆成 N 個單日子項。
    每個子項對應一次 fetcher 呼叫，達到「逐日 commit」的最細粒度。
    """
    try:
        d_start = datetime.strptime(gap["gap_start"], "%Y-%m-%d").date()
    except Exception:
        return [gap]  # 異常筆，保留原樣

    d_end_str = gap.get("gap_end") or gap["gap_start"]
    try:
        d_end = datetime.strptime(d_end_str, "%Y-%m-%d").date()
    except Exception:
        d_end = d_start

    if d_end <= d_start:
        return [gap]

    days = []
    cur = d_start
    while cur <= d_end:
        # 跳過週末（台股無交易）
        if cur.weekday() < 5:
            sub = dict(gap)
            sub["gap_start"] = cur.strftime("%Y-%m-%d")
            sub["gap_end"]   = cur.strftime("%Y-%m-%d")
            sub["gap_days"]  = 1
            days.append(sub)
        cur += timedelta(days=1)

    return days if days else [gap]


# ─────────────────────────────────────────────
# 構建工作清單（每個 entry 是一個獨立 commit 單元）
# ─────────────────────────────────────────────
def build_work_units(filtered: list[dict], per_day: bool) -> list[dict]:
    """
    將 filtered gaps 轉成工作清單。
    一般模式：(script, stock_id) 取最早 gap_start → 一次 fetcher 涵蓋多 table
    per-day 模式：每個 (table, stock_id, day) 獨立成一個 fetcher 呼叫
    """
    if per_day:
        # 逐日模式：每個 (table, stock_id, day) 都是一個 unit
        days_units: list[dict] = []
        for g in filtered:
            for sub in split_gap_into_days(g):
                meta = TABLE_TO_FETCHER[sub["table"]]
                days_units.append({
                    "script":   meta["script"],
                    "stock_id": sub["stock_id"] if meta["id_flag"] else None,
                    "id_flag":  meta["id_flag"],
                    "table":    sub["table"],
                    "start":    sub["gap_start"],
                    "end":      sub["gap_end"],
                    "tables":   [sub["table"]],   # 兼容舊 entry 結構
                })
        return days_units

    # 一般模式：依 (script, stock_id) 分組，取該組最早 gap_start
    groups: dict[tuple[str, str], dict] = {}
    for g in filtered:
        meta = TABLE_TO_FETCHER[g["table"]]
        key = (meta["script"], g["stock_id"] if meta["id_flag"] else "MARKET")
        if key not in groups:
            groups[key] = {
                "script":   meta["script"],
                "stock_id": g["stock_id"] if meta["id_flag"] else None,
                "id_flag":  meta["id_flag"],
                "start":    g["gap_start"],
                "end":      g.get("gap_end"),
                "tables":   [g["table"]],
            }
        else:
            groups[key]["tables"].append(g["table"])
            if g["gap_start"] < groups[key]["start"]:
                groups[key]["start"] = g["gap_start"]
            if g.get("gap_end") and (
                groups[key]["end"] is None or g["gap_end"] > groups[key]["end"]
            ):
                groups[key]["end"] = g["gap_end"]
    return list(groups.values())


# ─────────────────────────────────────────────
# 單一 unit 處理（並行模式的 worker function）
# ─────────────────────────────────────────────
def process_unit(
    unit: dict,
    args,
    ckpt: dict,
    counters: dict,
) -> dict:
    """
    處理單一工作 unit（一次 fetcher 呼叫）。
    回傳結果 dict 給主執行緒整合。
    """
    ok, msg, elapsed = run_fetcher(
        script=unit["script"],
        stock_id=unit["stock_id"],
        id_flag=unit["id_flag"],
        start=unit["start"],
        end=unit.get("end"),
        dry_run=args.dry_run,
        timeout=args.timeout,
    )

    # 一個 fetcher 呼叫可能涵蓋多個 table，每個 table 獨立記錄
    out_entries = []
    for tbl in unit["tables"]:
        entry = {
            "stock_id": unit["stock_id"] or "MARKET",
            "table":    tbl,
            "start":    unit["start"],
            "end":      unit.get("end") or unit["start"],
            "script":   unit["script"],
            "elapsed":  round(elapsed, 2),
        }
        if ok:
            counters["successes"] += 1
            if not args.dry_run:
                append_success(entry)
                update_checkpoint(ckpt, entry, "SUCCESS", msg)
                write_fetch_log(tbl, unit["stock_id"], unit["start"],
                                unit.get("end"), "SUCCESS", msg)
        else:
            entry["error"] = msg
            counters["failures"] += 1
            if not args.dry_run:
                append_failure(entry)
                update_checkpoint(ckpt, entry, "FAILED", msg)
                write_fetch_log(tbl, unit["stock_id"], unit["start"],
                                unit.get("end"), "FAILED", msg)
        out_entries.append(entry)

    return {"ok": ok, "entries": out_entries, "elapsed": elapsed}


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="按 (table, stock_id) 精確補抓資料斷層 v3.0")
    parser.add_argument("--json", default=str(DEFAULT_JSON),
                        help=f"integrity_gaps.json 路徑（預設 {DEFAULT_JSON}）")
    parser.add_argument("--tables", default=None,
                        help="只處理指定資料表，多筆用逗號分隔")
    parser.add_argument("--stocks", default=None,
                        help="只處理指定股票，多筆用逗號分隔")
    parser.add_argument("--dry-run", action="store_true", help="試跑，不實際呼叫 fetcher")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="限制最多處理多少 unit（節流 API 配額）")
    parser.add_argument("--refresh-audit", action="store_true",
                        help="補抓前先跑 data_integrity_audit 重新產生 integrity_gaps.json")
    parser.add_argument("--min-gap-days", type=int, default=1,
                        help="只補抓 gap_days >= 此值的斷層（預設 1）")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="單一 fetcher 最長執行秒數（預設 1800）")
    parser.add_argument("--keep-history", action="store_true",
                        help="保留前次 success/failure 紀錄（預設啟動時清空）")
    parser.add_argument("--per-day", action="store_true",
                        help="逐日模式：將每個 gap 拆成「每天一次 fetcher 呼叫」"
                             "（最細粒度但 API 用量倍增）")
    parser.add_argument("--resume", action="store_true",
                        help="從上次 checkpoint 續做，跳過已成功的 unit")
    parser.add_argument("--parallel", type=int, default=1,
                        help="並行 fetcher 數（預設 1；建議 ≤4，受 FinMind 速率限制）")
    args = parser.parse_args()

    # ── 啟動時建立所有必要目錄 ──
    if _CORE_OK:
        ensure_dirs_exist()

    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  資料斷層精確補抓 v3.0（逐支逐日 commit 完整性版）")
    logger.info(f"  模式：{'DRY-RUN' if args.dry_run else '正式執行'}"
                f" | per-day={args.per_day} | parallel={args.parallel}"
                f" | resume={args.resume}")
    logger.info("=" * 70)

    # ── 0) 重置或保留歷史紀錄 ──
    if not args.dry_run:
        if args.resume:
            logger.info("--resume 啟用：保留 checkpoint，跳過已成功項目。")
        elif not args.keep_history:
            reset_log_files()

    # ── 1) 取得最新 gap 清單 ──
    if args.refresh_audit:
        path = refresh_audit() or Path(args.json)
    else:
        path = Path(args.json)
    gaps = load_gaps(path)
    if not gaps:
        logger.info("沒有偵測到任何斷層，提前結束 ✨")
        return 0

    # ── 2) 套用篩選條件 ──
    table_filter = set(args.tables.split(",")) if args.tables else None
    stock_filter = set(args.stocks.split(",")) if args.stocks else None

    filtered: list[dict] = []
    skipped_unknown = 0
    for g in gaps:
        if g.get("gap_days", 0) < args.min_gap_days:
            continue
        if table_filter and g["table"] not in table_filter:
            continue
        if stock_filter and g["stock_id"] not in stock_filter:
            continue
        if g["table"] not in TABLE_TO_FETCHER:
            skipped_unknown += 1
            continue
        filtered.append(g)

    if skipped_unknown:
        logger.warning(f"未知資料表已跳過 {skipped_unknown} 筆")
    logger.info(f"套用篩選後共 {len(filtered):,} 筆斷層待補抓")
    if not filtered:
        logger.info("篩選後無任何斷層，結束。")
        return 0

    # ── 3) 構建工作清單（一般模式 / per-day 模式）──
    units = build_work_units(filtered, per_day=args.per_day)
    logger.info(f"工作 units 數：{len(units):,}（per_day={args.per_day}）")

    # ── 4) Resume：跳過 checkpoint 中已成功的 unit ──
    ckpt = load_checkpoint() if args.resume else {}
    if args.resume and ckpt:
        before = len(units)
        units = [
            u for u in units
            if not all(
                ckpt.get(_ckpt_key({
                    "table": tbl,
                    "stock_id": u["stock_id"],
                    "start": u["start"],
                }), {}).get("status") == "SUCCESS"
                for tbl in u["tables"]
            )
        ]
        logger.info(f"--resume：跳過已成功 {before - len(units)} units，剩餘 {len(units)}")

    if args.max_stocks:
        units = units[: args.max_stocks]
        logger.info(f"--max-stocks：限制處理 {len(units)} units")

    if not units:
        logger.info("沒有需要處理的 unit，結束。")
        return 0

    # ── 5) 執行（單執行緒 / 並行）──
    counters = {"successes": 0, "failures": 0}

    if args.parallel <= 1 or args.dry_run:
        # 單執行緒（dry-run 也走這條，避免噪音）
        for i, unit in enumerate(units, 1):
            logger.info(
                f"\n[{i}/{len(units)}] {unit['script']} | "
                f"sid={unit['stock_id'] or 'MARKET'} | "
                f"start={unit['start']} | tables={unit['tables']}"
            )
            process_unit(unit, args, ckpt, counters)
            if i % 50 == 0:
                _log_progress(i, len(units), counters, t0)
    else:
        # 並行（ThreadPoolExecutor，subprocess 之間天然隔離）
        logger.info(f"並行模式啟用：{args.parallel} 條 worker")
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(process_unit, u, args, ckpt, counters): (i, u)
                for i, u in enumerate(units, 1)
            }
            done = 0
            for fut in as_completed(futures):
                i, unit = futures[fut]
                done += 1
                try:
                    res = fut.result()
                    status = "✅" if res["ok"] else "❌"
                    logger.info(
                        f"[{done}/{len(units)}] {status} {unit['script']} | "
                        f"sid={unit['stock_id'] or 'MARKET'} | "
                        f"耗時 {res['elapsed']:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"[{done}/{len(units)}] worker 例外：{e}")
                if done % 50 == 0:
                    _log_progress(done, len(units), counters, t0)

    # ── 6) 摘要 ──
    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(
        f"  完成：{len(units)} units  |  "
        f"成功 {counters['successes']}，失敗 {counters['failures']}  |  "
        f"耗時 {elapsed:.1f}s（avg {elapsed / max(len(units), 1):.2f}s/unit）"
    )
    if counters["failures"] and not args.dry_run:
        logger.info(f"  失敗清單：{FAILURE_JSON}")
    if counters["successes"] and not args.dry_run:
        logger.info(f"  成功清單：{SUCCESS_JSON}")
    if not args.dry_run:
        logger.info(f"  Checkpoint：{CHECKPOINT_JSON}（下次可用 --resume 續做）")

    # ── 7) 自動同步至 Trinity 健康度矩陣 ──
    if counters["successes"] and not args.dry_run:
        try:
            from sync_trinity_db import sync_health_matrix
            logger.info("正在將最新補抓結果同步至全系統健康度矩陣...")
            sync_health_matrix()
            logger.info("✅ 健康度矩陣同步完成。")
        except Exception as e:
            logger.warning(f"自動同步失敗（不影響補抓）：{e}")

    # ── 8) 印出 FinMind 請求統計（若 fetcher 在同程序中發過 API）──
    try:
        from core.finmind_client import get_request_stats
        get_request_stats().summary()
    except Exception:
        pass

    return 0 if not counters["failures"] else 2


def _log_progress(done: int, total: int, counters: dict, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / max(elapsed, 0.01)
    eta = (total - done) / max(rate, 0.01)
    logger.info(
        f"  進度：{done}/{total}（{done/total*100:.1f}%）"
        f"  ✅ {counters['successes']}  ❌ {counters['failures']}"
        f"  ETA {eta:.0f}s"
    )


if __name__ == "__main__":
    sys.exit(main())