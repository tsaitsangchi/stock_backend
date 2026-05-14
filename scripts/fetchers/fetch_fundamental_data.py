"""
fetch_fundamental_data.py — 基本面財報資料 (v3.4 Schema Auto-Migration 版)
================================================================================
v3.4 改進：
  ★ 強化 `_upgrade_schema()`：自動將舊版的 VARCHAR(50) 欄位擴充為 VARCHAR(255)，
    徹底解決 `value too long for type character varying(50)` 的寫入崩潰問題。

v3.3 既有：
  ★ 自動移轉 Primary Key 至 (stock_id, date, type, origin_name)。
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 結合 `FailureLogger` 與 `commit_per_stock_per_day` 進行雙層粒度原子寫入。
  ★ 財報發布遞延邏輯：將原始的會計結算日推遲 45 天作為發布日，以消除前視偏誤 (Look-ahead Bias)。

執行範例（常規）：
    python scripts/fetchers/fetch_fundamental_data.py
    python scripts/fetchers/fetch_fundamental_data.py --stock-id 2330,2454 --force
"""

from __future__ import annotations

import sys
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
import argparse
import time
from pathlib import Path

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
    get_db_stock_ids,
    get_core_stocks_from_db,
    ensure_ddl,
    safe_float,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    commit_per_stock_per_day,
    dedup_rows,
    DDL_FETCH_LOG
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

DATASET_START = {
    "financial_statements": "2010-01-01"
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

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

def _write_fetch_log(conn, **kwargs):
    """寫入 fetch_log，失敗不影響主流程。"""
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO fetch_log (
                run_ts, table_name, stock_id, fetch_mode,
                fetch_date_from, fetch_date_to,
                rows_inserted, rows_updated, duration_ms,
                status, error_message, cli_args
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
            """
            cur.execute(sql, (
                kwargs.get("table_name"), kwargs.get("stock_id"), kwargs.get("fetch_mode", "per_stock"),
                kwargs.get("fetch_date_from"), kwargs.get("fetch_date_to"),
                kwargs.get("rows_inserted", 0), kwargs.get("duration_ms", 0),
                kwargs.get("status"), kwargs.get("error_message"), _CLI_ARGS_STR
            ))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.debug(f"fetch_log 寫入失敗：{e}")

# 這裡也把 DDL 擴充到 VARCHAR(255)
DDL_FINANCIAL_STATEMENT = """
CREATE TABLE IF NOT EXISTS financial_statements (
    stock_id    VARCHAR(20),
    date        DATE,
    type        VARCHAR(255),
    value       NUMERIC(20,4),
    origin_name VARCHAR(255),
    PRIMARY KEY (stock_id, date, type, origin_name)
);
CREATE INDEX IF NOT EXISTS idx_fin_stock_date ON financial_statements (stock_id, date);
"""

UPSERT_FINANCIAL_STATEMENT = """
INSERT INTO financial_statements (stock_id, date, type, value, origin_name)
VALUES %s
ON CONFLICT (stock_id, date, type, origin_name) 
DO UPDATE SET value = EXCLUDED.value;
"""

def _upgrade_schema(conn) -> None:
    """
    自動化 Schema 移轉 (Auto-Migration)
    確保欄位長度足夠，並更新 Primary Key。
    """
    try:
        with conn.cursor() as cur:
            # 1. 強制擴充欄位長度，解決 VARCHAR(50) 不足的問題
            cur.execute("ALTER TABLE financial_statements ALTER COLUMN type TYPE VARCHAR(255);")
            cur.execute("ALTER TABLE financial_statements ALTER COLUMN origin_name TYPE VARCHAR(255);")
            
            # 2. 檢查並更新 Primary Key
            cur.execute("""
                SELECT a.attname
                FROM   pg_index i
                JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                     AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = 'financial_statements'::regclass
                AND    i.indisprimary;
            """)
            pk_columns = {row[0] for row in cur.fetchall()}
            
            if pk_columns and "origin_name" not in pk_columns:
                logger.warning("🛠️ 偵測到舊版 Primary Key，正在自動進行 Schema 升級 (加入 origin_name)...")
                cur.execute("ALTER TABLE financial_statements DROP CONSTRAINT IF EXISTS financial_statements_pkey;")
                cur.execute("ALTER TABLE financial_statements ADD PRIMARY KEY (stock_id, date, type, origin_name);")
                logger.info("✅ Schema 升級完成！已支援 origin_name 且欄位已擴充至 VARCHAR(255)。")
                
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.debug(f"Schema 升級檢查失敗 (可能是首次建表，可忽略): {e}")


# ─────────────────────────────────────────────
# Mappers
# ─────────────────────────────────────────────
def map_financial_statement(r: dict) -> tuple | None:
    """處理財報資料，並包含遞延 45 天的前視偏誤消除邏輯。"""
    try:
        original_date = datetime.strptime(r["date"], "%Y-%m-%d")
        publish_date = original_date + timedelta(days=45)
        return (
            str(r["stock_id"]), 
            publish_date.strftime("%Y-%m-%d"), 
            str(r.get("type", ""))[:255],  # 配合資料庫長度，截斷保護
            safe_float(r.get("value")), 
            str(r.get("origin_name", r.get("type", "")))[:255] # 配合資料庫長度，截斷保護
        )
    except Exception as e:
        logger.debug(f"財報資料 mapper 錯誤跳過: {e}")
        return None

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_financial_statements(
    conn, stock_ids: list[str], start: str, end: str, 
    delay: float, force: bool, fetch_mode_override: str | None = None
):
    table = "financial_statements"
    logger.info(f"=== [{table}] 開始 ===")
    
    # 執行 DDL 與 Schema 升級
    ensure_ddl(conn, DDL_FINANCIAL_STATEMENT)
    conn.commit()
    _upgrade_schema(conn)
    
    flog = FailureLogger(table, db_conn=conn)
    latest = get_all_safe_starts(conn, table, key_col="stock_id")
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"
    tmpl = "(%s, %s::date, %s, %s::numeric, %s)"
    
    for sid in stock_ids:
        s = resolve_start_cached(sid, latest, start, DATASET_START[table], force)
        if not s:
            _write_fetch_log(conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue
        
        t0 = time.time()
        try:
            data = finmind_get(
                dataset="TaiwanStockFinancialStatements", 
                params={"data_id": sid, "start_date": s, "end_date": end}, 
                delay=delay,
                raise_on_error=True
            )
            dur = int((time.time() - t0) * 1000)
            
            if data:
                rows = []
                for r in data:
                    mapped = map_financial_statement(r)
                    if mapped:
                        rows.append(mapped)
                
                rows = dedup_rows(rows, (0, 1, 2, 4)) # PK: stock_id, date, type, origin_name
                
                res = commit_per_stock_per_day(
                    conn, UPSERT_FINANCIAL_STATEMENT, rows, tmpl, 
                    date_index=1, stock_index=0, 
                    label_prefix=table, failure_logger=flog
                )
                n = sum(res.values())
                total_rows += n
                _write_fetch_log(
                    conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=n, 
                    duration_ms=dur, status="success" if n > 0 else "partial"
                )
            else:
                _write_fetch_log(
                    conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                    duration_ms=dur, status="no_new_data"
                )
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.record(stock_id=sid, error=str(e), start_date=s, end_date=end)
            _write_fetch_log(
                conn, table_name=table, stock_id=sid, fetch_mode=fetch_mode, 
                fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                duration_ms=dur, status="failed", error_message=str(e)
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

def _run_targeted(conn, targets: dict[str, list[str]], args, fetch_mode: str):
    for tbl, sids in targets.items():
        if not sids: continue
        if tbl == "financial_statements":
            fetch_financial_statements(conn, sids, args.start, args.end, args.delay, force=True, fetch_mode_override=fetch_mode)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+", choices=["financial_statements", "all"], default=["all"])
    p.add_argument("--stock-id", default=None)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    tables = ["financial_statements"] if "all" in args.tables else args.tables
    conn = get_db_conn()
    
    try:
        _ensure_fetch_log_table(conn)
        
        # 決定要抓取的標的
        if args.stock_id:
            stock_ids = [s.strip() for s in args.stock_id.split(",")]
        else:
            # 優先使用動態核心配置，並過濾 fetch_fundamental = TRUE
            stock_configs = get_core_stocks_from_db(conn)
            if stock_configs:
                stock_ids = [sid for sid, cfg in stock_configs.items() if cfg.get("is_active", True) and cfg.get("fetch_fundamental", True)]
            else:
                stock_ids = get_db_stock_ids(conn)
                
        if not stock_ids:
            logger.warning("未找到任何需要抓取基本面的標的。")
            return
            
        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed, tables)
            if targets: _run_targeted(conn, targets, args, fetch_mode="retry")
            else: logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, tables, stock_ids)
            if targets: _run_targeted(conn, targets, args, fetch_mode="gap_fill")
            else: logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        if "financial_statements" in tables: 
            fetch_financial_statements(conn, stock_ids, args.start, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")
        get_request_stats().summary()

if __name__ == "__main__":
    main()