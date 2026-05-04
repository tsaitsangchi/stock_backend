from __future__ import annotations
import sys
import json
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_extended_derivative_data.py — 期貨/選擇權 II（逐支 commit 完整性版）
================================================================================
v2.2 改進：
  · safe_commit_rows()：每個商品 (futures_id / option_id) 寫入後立即 commit，失敗 rollback。
  · 主迴圈以 try/except 包單一商品，單個失敗不影響其他商品。
  · 失敗清單寫入 outputs/{table}_failed_{date}.json。

執行：
    python fetch_extended_derivative_data.py
    python fetch_extended_derivative_data.py --tables futures_inst_investors options_inst_investors
    python fetch_extended_derivative_data.py --force --start 2018-06-05
"""

import argparse
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta

from config import DB_CONFIG  # noqa: F401
from core.finmind_client import finmind_get, BatchNotSupportedError
from core.db_utils import (
    get_db_conn, ensure_ddl, bulk_upsert,
    safe_float, safe_int,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = _base_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_START = {
    "futures_inst_investors":   "2018-06-05",
    "options_inst_investors":   "2018-06-05",
    "futures_inst_after_hours": "2021-10-12",
    "options_inst_after_hours": "2021-10-12",
    "futures_dealer_volume":    "2021-04-01",
    "options_dealer_volume":    "2021-04-01",
}
DEFAULT_END = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
DDL_FUT_INST = """
CREATE TABLE IF NOT EXISTS futures_inst_investors (
    date                                       DATE,
    futures_id                                 VARCHAR(50),
    institutional_investors                    VARCHAR(100),
    long_deal_volume                           BIGINT,
    long_deal_amount                           NUMERIC(20,2),
    short_deal_volume                          BIGINT,
    short_deal_amount                          NUMERIC(20,2),
    long_open_interest_balance_volume          BIGINT,
    long_open_interest_balance_amount          NUMERIC(20,2),
    short_open_interest_balance_volume         BIGINT,
    short_open_interest_balance_amount         NUMERIC(20,2),
    PRIMARY KEY (date, futures_id, institutional_investors)
);
CREATE INDEX IF NOT EXISTS idx_fii_date ON futures_inst_investors (date);
"""

DDL_OPT_INST = """
CREATE TABLE IF NOT EXISTS options_inst_investors (
    date                                       DATE,
    option_id                                  VARCHAR(50),
    call_put                                   VARCHAR(10),
    institutional_investors                    VARCHAR(100),
    long_deal_volume                           BIGINT,
    long_deal_amount                           NUMERIC(20,2),
    short_deal_volume                          BIGINT,
    short_deal_amount                          NUMERIC(20,2),
    long_open_interest_balance_volume          BIGINT,
    long_open_interest_balance_amount          NUMERIC(20,2),
    short_open_interest_balance_volume         BIGINT,
    short_open_interest_balance_amount         NUMERIC(20,2),
    PRIMARY KEY (date, option_id, call_put, institutional_investors)
);
CREATE INDEX IF NOT EXISTS idx_oii_date ON options_inst_investors (date);
"""

DDL_FUT_AH = """
CREATE TABLE IF NOT EXISTS futures_inst_after_hours (
    date                    DATE,
    futures_id              VARCHAR(50),
    institutional_investors VARCHAR(100),
    long_deal_volume        BIGINT,
    long_deal_amount        NUMERIC(20,2),
    short_deal_volume       BIGINT,
    short_deal_amount       NUMERIC(20,2),
    PRIMARY KEY (date, futures_id, institutional_investors)
);
CREATE INDEX IF NOT EXISTS idx_fiah_date ON futures_inst_after_hours (date);
"""

DDL_OPT_AH = """
CREATE TABLE IF NOT EXISTS options_inst_after_hours (
    date                    DATE,
    option_id               VARCHAR(50),
    call_put                VARCHAR(10),
    institutional_investors VARCHAR(100),
    long_deal_volume        BIGINT,
    long_deal_amount        NUMERIC(20,2),
    short_deal_volume       BIGINT,
    short_deal_amount       NUMERIC(20,2),
    PRIMARY KEY (date, option_id, call_put, institutional_investors)
);
CREATE INDEX IF NOT EXISTS idx_oiah_date ON options_inst_after_hours (date);
"""

DDL_FUT_DEALER = """
CREATE TABLE IF NOT EXISTS futures_dealer_volume (
    date         DATE,
    futures_id   VARCHAR(50),
    dealer_code  VARCHAR(50),
    dealer_name  VARCHAR(200),
    volume       BIGINT,
    is_after_hour BOOLEAN,
    PRIMARY KEY (date, futures_id, dealer_code, is_after_hour)
);
CREATE INDEX IF NOT EXISTS idx_fdv_date ON futures_dealer_volume (date);
"""

DDL_OPT_DEALER = """
CREATE TABLE IF NOT EXISTS options_dealer_volume (
    date         DATE,
    option_id    VARCHAR(50),
    dealer_code  VARCHAR(50),
    dealer_name  VARCHAR(200),
    volume       BIGINT,
    is_after_hour BOOLEAN,
    PRIMARY KEY (date, option_id, dealer_code, is_after_hour)
);
CREATE INDEX IF NOT EXISTS idx_odv_date ON options_dealer_volume (date);
"""

# ─────────────────────────────────────────────
# Upsert SQL
# ─────────────────────────────────────────────
UPSERT_FUT_INST = """
INSERT INTO futures_inst_investors VALUES %s
ON CONFLICT (date, futures_id, institutional_investors) DO UPDATE SET
    long_deal_volume                   = EXCLUDED.long_deal_volume,
    long_deal_amount                   = EXCLUDED.long_deal_amount,
    short_deal_volume                  = EXCLUDED.short_deal_volume,
    short_deal_amount                  = EXCLUDED.short_deal_amount,
    long_open_interest_balance_volume  = EXCLUDED.long_open_interest_balance_volume,
    long_open_interest_balance_amount  = EXCLUDED.long_open_interest_balance_amount,
    short_open_interest_balance_volume = EXCLUDED.short_open_interest_balance_volume,
    short_open_interest_balance_amount = EXCLUDED.short_open_interest_balance_amount;
"""

UPSERT_OPT_INST = """
INSERT INTO options_inst_investors VALUES %s
ON CONFLICT (date, option_id, call_put, institutional_investors) DO UPDATE SET
    long_deal_volume                   = EXCLUDED.long_deal_volume,
    long_deal_amount                   = EXCLUDED.long_deal_amount,
    short_deal_volume                  = EXCLUDED.short_deal_volume,
    short_deal_amount                  = EXCLUDED.short_deal_amount,
    long_open_interest_balance_volume  = EXCLUDED.long_open_interest_balance_volume,
    long_open_interest_balance_amount  = EXCLUDED.long_open_interest_balance_amount,
    short_open_interest_balance_volume = EXCLUDED.short_open_interest_balance_volume,
    short_open_interest_balance_amount = EXCLUDED.short_open_interest_balance_amount;
"""

UPSERT_FUT_AH = """
INSERT INTO futures_inst_after_hours VALUES %s
ON CONFLICT (date, futures_id, institutional_investors) DO UPDATE SET
    long_deal_volume  = EXCLUDED.long_deal_volume,
    long_deal_amount  = EXCLUDED.long_deal_amount,
    short_deal_volume = EXCLUDED.short_deal_volume,
    short_deal_amount = EXCLUDED.short_deal_amount;
"""

UPSERT_OPT_AH = """
INSERT INTO options_inst_after_hours VALUES %s
ON CONFLICT (date, option_id, call_put, institutional_investors) DO UPDATE SET
    long_deal_volume  = EXCLUDED.long_deal_volume,
    long_deal_amount  = EXCLUDED.long_deal_amount,
    short_deal_volume = EXCLUDED.short_deal_volume,
    short_deal_amount = EXCLUDED.short_deal_amount;
"""

UPSERT_FUT_DEALER = """
INSERT INTO futures_dealer_volume VALUES %s
ON CONFLICT (date, futures_id, dealer_code, is_after_hour) DO UPDATE SET
    dealer_name = EXCLUDED.dealer_name,
    volume      = EXCLUDED.volume;
"""

UPSERT_OPT_DEALER = """
INSERT INTO options_dealer_volume VALUES %s
ON CONFLICT (date, option_id, dealer_code, is_after_hour) DO UPDATE SET
    dealer_name = EXCLUDED.dealer_name,
    volume      = EXCLUDED.volume;
"""


# ─────────────────────────────────────────────
# 逐支 commit 工具函式
# ─────────────────────────────────────────────
def safe_commit_rows(conn, upsert_sql: str, rows: list, template: str,
                      label: str = "") -> int:
    if not rows:
        return 0
    try:
        n = bulk_upsert(conn, upsert_sql, rows, template)
        conn.commit()
        return n
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error(f"  [{label}] 寫入失敗，已 rollback：{e}")
        return 0


def dump_failures(table: str, failures: list) -> None:
    if not failures:
        return
    out = OUTPUT_DIR / f"{table}_failed_{date.today().strftime('%Y%m%d')}.json"
    try:
        out.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"  失敗清單已寫入：{out}（{len(failures)} 筆）")
    except Exception as e:
        logger.warning(f"  寫入失敗清單時發生錯誤：{e}")


# ─────────────────────────────────────────────
# Mapper
# ─────────────────────────────────────────────
def _f(r, k): return safe_float(r.get(k))
def _i(r, k): return safe_int(r.get(k))

def map_fut_inst(r): return (
    r["date"], r.get("futures_id") or r.get("name"),
    r.get("institutional_investors"),
    _i(r, "long_deal_volume"),  _f(r, "long_deal_amount"),
    _i(r, "short_deal_volume"), _f(r, "short_deal_amount"),
    _i(r, "long_open_interest_balance_volume"),  _f(r, "long_open_interest_balance_amount"),
    _i(r, "short_open_interest_balance_volume"), _f(r, "short_open_interest_balance_amount"),
)

def map_opt_inst(r): return (
    r["date"], r.get("option_id") or r.get("name"),
    r.get("call_put"), r.get("institutional_investors"),
    _i(r, "long_deal_volume"),  _f(r, "long_deal_amount"),
    _i(r, "short_deal_volume"), _f(r, "short_deal_amount"),
    _i(r, "long_open_interest_balance_volume"),  _f(r, "long_open_interest_balance_amount"),
    _i(r, "short_open_interest_balance_volume"), _f(r, "short_open_interest_balance_amount"),
)

def map_fut_ah(r): return (
    r["date"], r.get("futures_id"),
    r.get("institutional_investors"),
    _i(r, "long_deal_volume"),  _f(r, "long_deal_amount"),
    _i(r, "short_deal_volume"), _f(r, "short_deal_amount"),
)

def map_opt_ah(r): return (
    r["date"], r.get("option_id"),
    r.get("call_put"), r.get("institutional_investors"),
    _i(r, "long_deal_volume"),  _f(r, "long_deal_amount"),
    _i(r, "short_deal_volume"), _f(r, "short_deal_amount"),
)

def map_fut_dealer(r): return (
    r["date"], r.get("futures_id"),
    r.get("dealer_code"), r.get("dealer_name"),
    _i(r, "volume"),
    bool(r.get("is_after_hour")),
)

def map_opt_dealer(r): return (
    r["date"], r.get("option_id"),
    r.get("dealer_code"), r.get("dealer_name"),
    _i(r, "volume"),
    bool(r.get("is_after_hour")),
)

# ─────────────────────────────────────────────
# 抓取邏輯
# ─────────────────────────────────────────────
def latest_market_date(conn, table: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(date) FROM {table}")
        row = cur.fetchone()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d")
    return None


def fetch_market_dataset(
    conn, dataset: str, table: str, ddl: str,
    upsert_sql: str, template: str, mapper, dataset_key: str,
    start: str, end: str, delay: float, force: bool,
    chunk_days: int = 90,
):
    """Sponsor 一次抓全期貨/選擇權，分段請求並依商品 id 逐支 commit。"""
    ensure_ddl(conn, ddl)
    conn.commit()
    s = start
    if not force:
        last = latest_market_date(conn, table)
        if last:
            next_d = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            if next_d > end:
                logger.info(f"[{table}] 已最新（{last}），跳過")
                return
            s = max(next_d, DATASET_START[dataset_key])
    s = max(s, DATASET_START[dataset_key])

    logger.info(f"[{table}] 抓取 {s} ~ {end}（chunk={chunk_days} 天）")
    seg_start = s
    seg_end_dt = datetime.strptime(end, "%Y-%m-%d")
    total_rows = 0
    api_calls = 0
    failures: list[dict] = []

    while True:
        seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
        if seg_start_dt > seg_end_dt:
            break
        seg_end = min(
            (seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"),
            end,
        )

        # API 呼叫獨立 try/except，單 chunk 失敗不影響整支 fetcher
        try:
            data = finmind_get(
                dataset, {"start_date": seg_start, "end_date": seg_end},
                delay, raise_on_batch_400=True,
            )
        except BatchNotSupportedError as e:
            logger.error(f"[{table}] 帳號等級不支援批次：{e}")
            return
        except Exception as e:
            logger.error(f"[{table}] {seg_start}~{seg_end} API 失敗：{e}")
            failures.append({"chunk": f"{seg_start}~{seg_end}", "error": str(e)})
            seg_start = (
                datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            continue

        api_calls += 1
        if data:
            rows = []
            for r in data:
                try:
                    rows.append(mapper(r))
                except Exception as e:
                    logger.warning(f"  [{table}] mapper 異常筆，跳過：{e}")

            # ── 依商品代號分組逐支 commit ──
            rows_by_id: dict[str, list] = defaultdict(list)
            for r in rows:
                iid = r[1]
                rows_by_id[iid].append(r)

            for iid, s_rows in rows_by_id.items():
                try:
                    n = safe_commit_rows(
                        conn, upsert_sql, s_rows, template,
                        label=f"{table}/{iid}"
                    )
                    total_rows += n
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    failures.append({"id": iid, "chunk": f"{seg_start}~{seg_end}",
                                     "error": str(e)})
                    logger.error(f"  [{table}/{iid}] 寫入失敗：{e}")

            logger.info(f"    → 寫入 {len(rows_by_id)} 支商品（共 {len(rows)} 筆）")

        seg_start = (
            datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

    dump_failures(table, failures)
    logger.info(f"[{table}] 完成 API:{api_calls} 寫入:{total_rows} 失敗:{len(failures)}")

# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", nargs="+",
                   choices=list(DATASET_START.keys()),
                   default=list(DATASET_START.keys()))
    p.add_argument("--start", type=str, default="2018-06-05")
    p.add_argument("--end",   type=str, default=DEFAULT_END)
    p.add_argument("--delay", type=float, default=1.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--chunk-days", type=int, default=60,
                   help="每次批次請求涵蓋的天數（衍生品資料量較大，建議 60 天）")
    args = p.parse_args()

    conn = get_db_conn()
    try:
        configs = [
            ("futures_inst_investors", "TaiwanFuturesInstitutionalInvestors",
             DDL_FUT_INST, UPSERT_FUT_INST,
             "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", map_fut_inst),
            ("options_inst_investors", "TaiwanOptionInstitutionalInvestors",
             DDL_OPT_INST, UPSERT_OPT_INST,
             "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", map_opt_inst),
            ("futures_inst_after_hours", "TaiwanFuturesInstitutionalInvestorsAfterHours",
             DDL_FUT_AH, UPSERT_FUT_AH,
             "(%s, %s, %s, %s, %s, %s, %s)", map_fut_ah),
            ("options_inst_after_hours", "TaiwanOptionInstitutionalInvestorsAfterHours",
             DDL_OPT_AH, UPSERT_OPT_AH,
             "(%s, %s, %s, %s, %s, %s, %s, %s)", map_opt_ah),
            ("futures_dealer_volume", "TaiwanFuturesDealerTradingVolumeDaily",
             DDL_FUT_DEALER, UPSERT_FUT_DEALER,
             "(%s, %s, %s, %s, %s, %s)", map_fut_dealer),
            ("options_dealer_volume", "TaiwanOptionDealerTradingVolumeDaily",
             DDL_OPT_DEALER, UPSERT_OPT_DEALER,
             "(%s, %s, %s, %s, %s, %s)", map_opt_dealer),
        ]
        for key, dataset, ddl, upsert, tmpl, mapper in configs:
            if key not in args.tables:
                continue
            try:
                fetch_market_dataset(
                    conn, dataset, key, ddl, upsert, tmpl, mapper, key,
                    args.start, args.end, args.delay, args.force, args.chunk_days,
                )
            except Exception as e:
                try: conn.rollback()
                except Exception: pass
                logger.error(f"[{key}] 未預期錯誤：{e}")
    finally:
        conn.close()
    logger.info("全部完成")

if __name__ == "__main__":
    main()