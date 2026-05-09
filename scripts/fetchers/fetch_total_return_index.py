"""
fetch_total_return_index.py — 台股報酬指數（v3.4 Schema Auto-Migration 版）
================================================================================
v3.4 改進：
  ★ 導入 `_upgrade_schema()`：自動偵測舊版資料庫中可能被命名為 `value` 或 `return_index` 的欄位，
    並自動進行 `RENAME COLUMN` 升級為 `total_return_index`，徹底解決 column does not exist 錯誤。

v3.3 既有：
  ★ 修正 SQL 語法：明確宣告 INSERT INTO 欄位。
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 結合 `FailureLogger` 與 `commit_per_stock_per_day` 進行雙層粒度原子寫入。

執行範例（常規）：
    python scripts/fetchers/fetch_total_return_index.py
    python scripts/fetchers/fetch_total_return_index.py --ids TAIEX --force
"""

from __future__ import annotations

import sys
import logging
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.finmind_client import finmind_get, get_request_stats
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    write_fetch_log,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
    DDL_FETCH_LOG
)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATASET_START = {
    "total_return_index": "2003-01-01"
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")
DEFAULT_IDS = ["TAIEX", "TPEx"]

_CLI_ARGS_STR = " ".join(sys.argv)


# ─────────────────────────────────────────────
# 日誌與 SQL
# ─────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")

DDL_TOTAL_RETURN = """
CREATE TABLE IF NOT EXISTS total_return_index (
    date DATE, 
    stock_id VARCHAR(50), 
    total_return_index NUMERIC(20,6), 
    PRIMARY KEY (date, stock_id)
);
"""

def _upgrade_schema(conn) -> None:
    """
    自動化 Schema 移轉 (Auto-Migration)
    解決舊版資料庫可能使用 value、return_index 或大寫名稱作為欄位名的問題。
    """
    try:
        with conn.cursor() as cur:
            # 取得 total_return_index 表的所有欄位
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'total_return_index';
            """)
            columns = [row[0] for row in cur.fetchall()]
            
            # 如果發現沒有 total_return_index 欄位，代表是舊版 Schema
            if columns and "total_return_index" not in columns:
                # 找出除了 date, stock_id 以外的那個舊版資料欄位
                for col in columns:
                    if col not in ('date', 'stock_id'):
                        logger.warning(f"🛠️ 偵測到舊版欄位名稱 '{col}'，正在自動進行 Schema 升級 (改名為 total_return_index)...")
                        cur.execute(f'ALTER TABLE total_return_index RENAME COLUMN "{col}" TO total_return_index;')
                        logger.info("✅ Schema 升級完成！")
                        break
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.debug(f"Schema 升級檢查失敗 (可能是首次建表，可忽略): {e}")

UPSERT_TOTAL_RETURN = """
INSERT INTO total_return_index (date, stock_id, total_return_index) VALUES %s 
ON CONFLICT (date, stock_id) DO UPDATE SET 
    total_return_index = EXCLUDED.total_return_index;
"""


# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_tr(r: dict) -> tuple:
    return (r["date"], str(r.get("stock_id", "")), safe_float(r.get("price")))


# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_total_return_index(
    conn, target_ids: list[str], start: str, end: str, 
    delay: float, force: bool, fetch_mode_override: str | None = None
):
    table = "total_return_index"
    logger.info(f"=== [{table}] 開始 ===")
    
    # 執行 DDL 與 Schema 升級
    ensure_ddl(conn, DDL_TOTAL_RETURN)
    conn.commit()
    _upgrade_schema(conn)
    
    flog = FailureLogger(table, db_conn=conn)
    latest = get_all_safe_starts(conn, table)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    tmpl = "(%s::date, %s, %s::numeric)"

    for sid in target_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START[table], force)
        if not s:
            write_fetch_log(conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            data = finmind_get(
                dataset="TaiwanStockTotalReturnIndex", 
                params={"data_id": sid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            duration_ms = int((time.time() - t0) * 1000)
            
            if data:
                rows = [map_tr(r) for r in data]
                rows = dedup_rows(rows, (0, 1))
                
                res = commit_per_stock_per_day(conn, UPSERT_TOTAL_RETURN, rows, tmpl, label_prefix=table, failure_logger=flog)
                n = sum(res.values())
                total_rows += n
                
                write_fetch_log(
                    conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=n, 
                    duration_ms=duration_ms, status="success" if n > 0 else "partial"
                )
            else:
                write_fetch_log(
                    conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                    duration_ms=duration_ms, status="no_new_data"
                )
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            write_fetch_log(
                conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                duration_ms=duration_ms, status="failed", error_message=str(e)
            )
            
    logger.info(f"  [{table}] 總共寫入 {total_rows} 筆")
    flog.summary()


# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int, target_tables: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    sql = """
    WITH recent AS (
        SELECT table_name, stock_id, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY table_name, stock_id ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = ANY(%s) AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT table_name, stock_id FROM recent WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (target_tables, str(days)))
            for tbl, sid in cur.fetchall():
                targets[tbl].append(sid)
    except Exception as e:
        logger.error(f"[retry-failed] 查詢失敗：{e}")
        return {}

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [retry-failed/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets

def query_gap_targets(conn, days: int, target_tables: list[str], all_stock_ids: list[str]) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = defaultdict(list)
    for tbl in target_tables:
        sql = f"SELECT DISTINCT stock_id FROM fetch_log WHERE table_name = %s AND status = 'success' AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);"
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (tbl, str(days), all_stock_ids))
                have_success = {row[0] for row in cur.fetchall()}
            missing = [sid for sid in all_stock_ids if sid not in have_success]
            targets[tbl].extend(missing)
        except Exception as e:
            logger.error(f"[gap-fill/{tbl}] 查詢失敗：{e}")

    for tbl, sids in targets.items():
        sample = sids[:5]
        logger.info(f"  [gap-fill/{tbl}] {len(sids)} 個目標 (例：{sample})")
    return targets


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="台股報酬指數抓取 (v3.4 — 核心模組升級版)")
    p.add_argument("--ids", nargs="+", help="指定報酬指數代碼 (例如 TAIEX, TPEx)", default=DEFAULT_IDS)
    p.add_argument("--start", default="2003-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    conn = get_db_conn()
    tables = ["total_return_index"]
    
    try:
        _ensure_fetch_log_table(conn)
        
        target_ids = args.ids
        
        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, tables)
            if targets and "total_return_index" in targets: 
                fetch_total_return_index(conn, targets["total_return_index"], args.start, args.end, args.delay, force=True, fetch_mode_override="retry")
            else: 
                logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, tables, target_ids)
            if targets and "total_return_index" in targets: 
                fetch_total_return_index(conn, targets["total_return_index"], args.start, args.end, args.delay, force=True, fetch_mode_override="gap_fill")
            else: 
                logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        fetch_total_return_index(conn, target_ids, args.start, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        get_request_stats().summary()

if __name__ == "__main__":
    main()