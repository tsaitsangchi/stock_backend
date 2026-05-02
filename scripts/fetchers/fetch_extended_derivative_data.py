from __future__ import annotations
import sys
from pathlib import Path
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
"""
fetch_extended_derivative_data.py — 期貨/選擇權 II + 各券商交易量 + 夜盤 II
================================================================================
[新增] 補齊衍生品的「人」維度資料：
       原有 derivative_data 只有期貨/選擇權的 OHLCV，
       原有 derivative_sentiment 只有 options_large_oi（OI 持倉結構）。
       本檔補上「成交層」的三大法人方向 + 夜盤動向 + 各券商交易強度。

  1. futures_inst_investors ← TaiwanFuturesInstitutionalInvestors
     資料範圍：2018-06-05 ~ now（Free per-data_id，Sponsor 批次）
     用途：期貨日盤三大法人多空成交與未平倉。常用 contract = TX、MTX、TJF。
           外資期貨多單未平倉 = 對大盤的真實看法（與現貨買賣方向交叉驗證）。

  2. options_inst_investors ← TaiwanOptionInstitutionalInvestors
     資料範圍：2018-06-05 ~ now（Free per-data_id，Sponsor 批次）
     用途：選擇權三大法人 call/put 多空成交。
           外資 put 賣出（賣保險）vs put 買進（買保險）的差距是強烈情緒指標。

  3. futures_inst_after_hours ← TaiwanFuturesInstitutionalInvestorsAfterHours（Sponsor）
     資料範圍：2021-10-12 ~ now
     用途：**夜盤反映外資真實意圖**（國際資金時段，雜訊低，比日盤乾淨）。

  4. options_inst_after_hours ← TaiwanOptionInstitutionalInvestorsAfterHours（Sponsor）
     資料範圍：2021-10-12 ~ now

  5. futures_dealer_volume ← TaiwanFuturesDealerTradingVolumeDaily（Free）
     資料範圍：2021-04-01 ~ now
     用途：各券商每日期貨交易量。可建構「主力券商期貨持倉」特徵，
           準確度遠高於只看法人總和。

  6. options_dealer_volume ← TaiwanOptionDealerTradingVolumeDaily（Free）

衍生因子（建議在 feature_engineering.py 補上）：
  - foreign_futures_oi_chg    = 外資期貨 OI 日變化
  - foreign_put_buy_intensity = 外資 put 買進金額 / put 總成交金額
  - night_session_premium     = 夜盤外資多空偏向（與日盤對照）
  - put_call_ratio_oi         = 全市場 put OI / call OI
  - top_dealer_concentration  = 前 5 大券商成交量佔比

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
# Mapper
# ─────────────────────────────────────────────
def _f(r, k): return safe_float(r.get(k))
def _i(r, k): return safe_int(r.get(k))

def map_fut_inst(r): return (
    r["date"], r.get("futures_id") or r.get("name"),  # 部分 dataset 用 name
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
    """
    Sponsor 可不帶 data_id 一次抓全期貨/選擇權，但回傳量大需分段。
    """
    ensure_ddl(conn, ddl)
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

    try:
        while True:
            seg_start_dt = datetime.strptime(seg_start, "%Y-%m-%d")
            if seg_start_dt > seg_end_dt:
                break
            seg_end = min(
                (seg_start_dt + timedelta(days=chunk_days - 1)).strftime("%Y-%m-%d"),
                end,
            )
            data = finmind_get(
                dataset, {"start_date": seg_start, "end_date": seg_end},
                delay, raise_on_batch_400=True,
            )
            api_calls += 1
            if data:
                rows = [mapper(r) for r in data]
                bulk_upsert(conn, upsert_sql, rows, template)
                total_rows += len(rows)
            seg_start = (
                datetime.strptime(seg_end, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
    except BatchNotSupportedError as e:
        logger.error(f"[{table}] 帳號等級不支援批次：{e}")
        return

    logger.info(f"[{table}] 完成 API:{api_calls} 寫入:{total_rows}")

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
            fetch_market_dataset(
                conn, dataset, key, ddl, upsert, tmpl, mapper, key,
                args.start, args.end, args.delay, args.force, args.chunk_days,
            )
    finally:
        conn.close()
    logger.info("全部完成")

if __name__ == "__main__":
    main()
