"""
monitor/update_daily_status.py
─────────────────────────────────────────────────────────────────────
從事件表聚合出 stock_daily_status 每日狀態快照表，作為監控 dashboard
與健康燈號告警的核心數據源。

事件源（append-only）→ 狀態源（每股每天一筆）：

    [fetch_log]              ─┐
    [signal_history]         ─┼─→ aggregate ─→ [stock_daily_status]
    [model_training_log]     ─┤                       │
    [stock_price] (既有)     ─┘                       ▼
                                              dashboard / 告警 / 反事實

執行方式：
    # 每日例行（建議排程在 18:00）
    python -m scripts.monitor.update_daily_status

    # 指定日期
    python -m scripts.monitor.update_daily_status --date 2026-05-07

    # 補抓過去 30 天（重建歷史狀態）
    python -m scripts.monitor.update_daily_status --backfill 30

    # 除錯：只跑單股
    python -m scripts.monitor.update_daily_status --stock 2330

    # 試跑：不寫入 DB
    python -m scripts.monitor.update_daily_status --dry-run

依賴：
    必要：stock_price（既有）、STOCK_CONFIGS、TABLE_REGISTRY（config.py）
    可選：fetch_log、signal_history、feature_audit、model_training_log
          缺哪張就只是該維度為 NULL，不會 crash
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

# ─────────────────────────────────────────────
# 將 scripts/ 加入 sys.path 以便 import config
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DB_CONFIG, TABLE_REGISTRY, LOG_DIR  # noqa: E402
from core.db_utils import get_core_stocks_from_db  # noqa: E402

try:
    from config import get_all_features  # noqa: E402
except ImportError:
    get_all_features = None

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_FILE = Path(LOG_DIR) / "update_daily_status.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 健康燈號門檻（之後可抽到 config.HEALTH_CONFIG）
# ─────────────────────────────────────────────
HEALTH_THRESHOLDS = {
    "data_lag_red":          3,    # data_lag_days >  3 → red
    "data_lag_yellow":       1,    # data_lag_days >  1 → yellow
    "missing_tables_red":    5,    # 缺 >= 5 張日頻表 → red
    "missing_tables_yellow": 2,
    "nan_rate_red":          0.30,
    "nan_rate_yellow":       0.10,
    "feature_count_min":     150,  # 應為 175，<150 視為異常
}

# 從 TABLE_REGISTRY 抽出「日頻」表，作為當日 fetch 覆蓋率的分母
DAILY_TABLES: set[str] = {
    name for name, cfg in TABLE_REGISTRY.items()
    if cfg.get("type") == "daily"
}


# ─────────────────────────────────────────────
# Data Class
# ─────────────────────────────────────────────
@dataclass
class DailyStatus:
    date: date
    stock_id: str
    fetch_completed_tables: int
    fetch_missing_tables: list[str]
    last_data_date: Optional[date]
    data_lag_days: Optional[int]
    feature_built: Optional[bool]
    feature_nan_rate: Optional[float]
    feature_count: Optional[int]
    prediction_generated: bool
    prob_up: Optional[float]
    decision: Optional[str]
    model_version: Optional[str]
    health_status: str
    health_reasons: list[str]

    def to_db_row(self) -> dict:
        return {
            "date":                   self.date,
            "stock_id":               self.stock_id,
            "fetch_completed_tables": self.fetch_completed_tables,
            "fetch_missing_tables":   json.dumps(self.fetch_missing_tables, ensure_ascii=False),
            "last_data_date":         self.last_data_date,
            "data_lag_days":          self.data_lag_days,
            "feature_built":          self.feature_built,
            "feature_nan_rate":       self.feature_nan_rate,
            "feature_count":          self.feature_count,
            "prediction_generated":   self.prediction_generated,
            "prob_up":                self.prob_up,
            "decision":               self.decision,
            "model_version":          self.model_version,
            "health_status":          self.health_status,
            "health_reasons":         json.dumps(self.health_reasons, ensure_ascii=False),
        }


# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────
def _connect():
    return psycopg2.connect(**DB_CONFIG)


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        )
    """, (table_name,))
    return cur.fetchone()[0]


def ensure_status_table(conn) -> None:
    """若 stock_daily_status 不存在則建立（含索引）。"""
    DDL = """
    CREATE TABLE IF NOT EXISTS stock_daily_status (
        date                    DATE          NOT NULL,
        stock_id                VARCHAR(20)   NOT NULL,
        fetch_completed_tables  INTEGER,
        fetch_missing_tables    TEXT,
        last_data_date          DATE,
        data_lag_days           INTEGER,
        feature_built           BOOLEAN,
        feature_nan_rate        NUMERIC(5,4),
        feature_count           INTEGER,
        prediction_generated    BOOLEAN,
        prob_up                 NUMERIC(5,4),
        decision                VARCHAR(16),
        model_version           VARCHAR(32),
        health_status           VARCHAR(16),
        health_reasons          TEXT,
        updated_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
        PRIMARY KEY (date, stock_id)
    );
    CREATE INDEX IF NOT EXISTS idx_sds_health
        ON stock_daily_status(date DESC, health_status);
    CREATE INDEX IF NOT EXISTS idx_sds_stock
        ON stock_daily_status(stock_id, date DESC);
    """
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    logger.debug("ensure_status_table OK")


# ─────────────────────────────────────────────
# 各維度聚合查詢
# ─────────────────────────────────────────────
def fetch_completed_per_stock(
    cur, target_date: date, stock_ids: list[str]
) -> dict[str, set[str]]:
    """
    從 fetch_log 統計指定日期當天，每支股票成功抓取的「表名」集合。
    market-level 表（stock_id IS NULL）視為「全體適用」，所有股票都加上。
    """
    per_stock: dict[str, set[str]] = {sid: set() for sid in stock_ids}

    if not _table_exists(cur, "fetch_log"):
        logger.warning("[fetch_log] 表不存在 → fetch 維度全部視為缺")
        return per_stock

    cur.execute("""
        SELECT stock_id, table_name
        FROM fetch_log
        WHERE DATE(run_ts) = %s
          AND status IN ('success', 'partial')
    """, (target_date,))

    market_tables: set[str] = set()
    for sid, tbl in cur.fetchall():
        if sid is None:
            market_tables.add(tbl)
        elif sid in per_stock:
            per_stock[sid].add(tbl)

    # 市場層級表：所有股票都繼承
    if market_tables:
        for sid in per_stock:
            per_stock[sid] |= market_tables

    return per_stock


def last_data_dates(cur, stock_ids: list[str]) -> dict[str, date]:
    """每支股票在 stock_price 表中的最新日期。"""
    if not _table_exists(cur, "stock_price"):
        logger.error("[stock_price] 表不存在 → data_lag 全部 NULL")
        return {}

    cur.execute("""
        SELECT stock_id, MAX(date) AS last_date
        FROM stock_price
        WHERE stock_id = ANY(%s)
        GROUP BY stock_id
    """, (stock_ids,))
    return {sid: d for sid, d in cur.fetchall()}


def signal_per_stock(
    cur, target_date: date, stock_ids: list[str]
) -> dict[str, dict]:
    """從 signal_history 取出當天每股的訊號。"""
    if not _table_exists(cur, "signal_history"):
        logger.warning("[signal_history] 表不存在 → predict 維度為空")
        return {}

    # 兼容兩種 schema：有沒有 prob_up_15d 欄位
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='public' AND table_name='signal_history'
    """)
    cols = {row[0] for row in cur.fetchall()}

    prob_col = "prob_up_15d" if "prob_up_15d" in cols else "prob_up"
    run_id_col = "model_run_id" if "model_run_id" in cols else "NULL::text"

    cur.execute(f"""
        SELECT stock_id, {prob_col}, decision, {run_id_col}
        FROM signal_history
        WHERE date = %s AND stock_id = ANY(%s)
    """, (target_date, stock_ids))

    return {
        sid: {"prob_up": p, "decision": dc, "model_run_id": mri}
        for sid, p, dc, mri in cur.fetchall()
    }


def feature_audit_per_stock(
    cur, target_date: date, stock_ids: list[str]
) -> dict[str, dict]:
    """
    若有 feature_audit 表（feature_engineering 寫入）則讀取最新一次的記錄。
    無此表時回傳 {}，後續欄位以 None 表示。
    """
    if not _table_exists(cur, "feature_audit"):
        return {}

    cur.execute("""
        SELECT DISTINCT ON (stock_id)
               stock_id, nan_rate_60d, feature_count, built_at
        FROM feature_audit
        WHERE stock_id = ANY(%s) AND DATE(built_at) <= %s
        ORDER BY stock_id, built_at DESC
    """, (stock_ids, target_date))

    return {
        sid: {"nan_rate": nr, "feature_count": fc, "built_at": ba}
        for sid, nr, fc, ba in cur.fetchall()
    }


# ─────────────────────────────────────────────
# 健康燈號計算
# ─────────────────────────────────────────────
def compute_health(
    data_lag_days: Optional[int],
    fetch_missing_tables: list[str],
    feature_built: Optional[bool],
    feature_nan_rate: Optional[float],
    feature_count: Optional[int],
    prediction_generated: bool,
) -> tuple[str, list[str]]:
    """根據各項指標決定 green/yellow/red，並回傳人類可讀的原因。"""
    reasons: list[str] = []
    is_red = False
    is_yellow = False

    # 1. 資料鮮度
    if data_lag_days is None:
        is_red = True
        reasons.append("⛔ stock_price 找不到任何資料")
    elif data_lag_days > HEALTH_THRESHOLDS["data_lag_red"]:
        is_red = True
        reasons.append(f"⛔ 資料延遲 {data_lag_days} 天")
    elif data_lag_days > HEALTH_THRESHOLDS["data_lag_yellow"]:
        is_yellow = True
        reasons.append(f"⚠️ 資料延遲 {data_lag_days} 天")

    # 2. 日頻表缺漏
    miss_n = len(fetch_missing_tables)
    if miss_n >= HEALTH_THRESHOLDS["missing_tables_red"]:
        is_red = True
        reasons.append(f"⛔ 今日缺 {miss_n} 張日頻表未抓")
    elif miss_n >= HEALTH_THRESHOLDS["missing_tables_yellow"]:
        is_yellow = True
        reasons.append(f"⚠️ 今日缺 {miss_n} 張日頻表")

    # 3. 特徵
    if feature_built is False:
        is_red = True
        reasons.append("⛔ 特徵未建立")
    if feature_count is not None and feature_count < HEALTH_THRESHOLDS["feature_count_min"]:
        is_red = True
        reasons.append(f"⛔ feature_count={feature_count}（< 150 異常）")
    if feature_nan_rate is not None:
        if feature_nan_rate > HEALTH_THRESHOLDS["nan_rate_red"]:
            is_red = True
            reasons.append(f"⛔ 特徵 NaN 率 {feature_nan_rate:.1%}")
        elif feature_nan_rate > HEALTH_THRESHOLDS["nan_rate_yellow"]:
            is_yellow = True
            reasons.append(f"⚠️ 特徵 NaN 率 {feature_nan_rate:.1%}")

    # 4. 預測
    if not prediction_generated:
        is_yellow = True
        reasons.append("⚠️ 今日無預測訊號")

    if is_red:
        return "red", reasons
    if is_yellow:
        return "yellow", reasons
    return "green", reasons or ["✅ 全部正常"]


# ─────────────────────────────────────────────
# 主聚合：對指定日期產出所有股票的 DailyStatus
# ─────────────────────────────────────────────
def build_status_for_date(
    conn,
    target_date: date,
    stock_ids: list[str],
) -> list[DailyStatus]:
    rows: list[DailyStatus] = []

    # feature_count 預期值（用於 feature_audit 缺席時填補）
    expected_feature_count: Optional[int] = None
    if get_all_features is not None and stock_ids:
        try:
            expected_feature_count = len(get_all_features(stock_ids[0]))
        except Exception as e:
            logger.warning(f"get_all_features 失敗：{e}")

    with conn.cursor() as cur:
        fetched     = fetch_completed_per_stock(cur, target_date, stock_ids)
        last_dates  = last_data_dates(cur, stock_ids)
        signals     = signal_per_stock(cur, target_date, stock_ids)
        feat_audits = feature_audit_per_stock(cur, target_date, stock_ids)

    for sid in stock_ids:
        completed = fetched.get(sid, set())
        # 只看日頻表的覆蓋率（月/季表用獨立邏輯，避免每天都喊 missing）
        completed_daily = completed & DAILY_TABLES
        missing_daily = sorted(DAILY_TABLES - completed_daily)

        last_d = last_dates.get(sid)
        lag = (target_date - last_d).days if last_d else None

        sig = signals.get(sid) or {}
        feat = feat_audits.get(sid) or {}

        feature_built = bool(feat) if feat_audits else None
        feature_nan_rate = (
            float(feat["nan_rate"]) if feat.get("nan_rate") is not None else None
        )
        feature_count = feat.get("feature_count") or expected_feature_count

        health, reasons = compute_health(
            data_lag_days=lag,
            fetch_missing_tables=missing_daily,
            feature_built=feature_built,
            feature_nan_rate=feature_nan_rate,
            feature_count=feature_count,
            prediction_generated=bool(sig),
        )

        rows.append(DailyStatus(
            date=target_date,
            stock_id=sid,
            fetch_completed_tables=len(completed_daily),
            fetch_missing_tables=missing_daily,
            last_data_date=last_d,
            data_lag_days=lag,
            feature_built=feature_built,
            feature_nan_rate=feature_nan_rate,
            feature_count=feature_count,
            prediction_generated=bool(sig),
            prob_up=float(sig["prob_up"]) if sig.get("prob_up") is not None else None,
            decision=sig.get("decision"),
            model_version=sig.get("model_run_id"),
            health_status=health,
            health_reasons=reasons,
        ))

    return rows


# ─────────────────────────────────────────────
# UPSERT
# ─────────────────────────────────────────────
UPSERT_SQL = """
INSERT INTO stock_daily_status (
    date, stock_id, fetch_completed_tables, fetch_missing_tables,
    last_data_date, data_lag_days, feature_built, feature_nan_rate,
    feature_count, prediction_generated, prob_up, decision,
    model_version, health_status, health_reasons, updated_at
) VALUES (
    %(date)s, %(stock_id)s, %(fetch_completed_tables)s, %(fetch_missing_tables)s,
    %(last_data_date)s, %(data_lag_days)s, %(feature_built)s, %(feature_nan_rate)s,
    %(feature_count)s, %(prediction_generated)s, %(prob_up)s, %(decision)s,
    %(model_version)s, %(health_status)s, %(health_reasons)s, NOW()
)
ON CONFLICT (date, stock_id) DO UPDATE SET
    fetch_completed_tables = EXCLUDED.fetch_completed_tables,
    fetch_missing_tables   = EXCLUDED.fetch_missing_tables,
    last_data_date         = EXCLUDED.last_data_date,
    data_lag_days          = EXCLUDED.data_lag_days,
    feature_built          = EXCLUDED.feature_built,
    feature_nan_rate       = EXCLUDED.feature_nan_rate,
    feature_count          = EXCLUDED.feature_count,
    prediction_generated   = EXCLUDED.prediction_generated,
    prob_up                = EXCLUDED.prob_up,
    decision               = EXCLUDED.decision,
    model_version          = EXCLUDED.model_version,
    health_status          = EXCLUDED.health_status,
    health_reasons         = EXCLUDED.health_reasons,
    updated_at             = NOW();
"""


def upsert_rows(conn, rows: list[DailyStatus], dry_run: bool = False) -> None:
    if not rows:
        logger.warning("沒有 row 需要寫入")
        return

    payload = [r.to_db_row() for r in rows]

    if dry_run:
        logger.info(f"[DRY RUN] 將會 UPSERT {len(payload)} 筆，前 3 筆預覽：")
        for r in rows[:3]:
            logger.info(
                f"  {r.stock_id} {r.date} health={r.health_status:6s} "
                f"lag={r.data_lag_days} miss={len(r.fetch_missing_tables)} "
                f"reasons={r.health_reasons[:1]}"
            )
        return

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, UPSERT_SQL, payload, page_size=200)
    conn.commit()
    logger.info(f"已 UPSERT {len(payload)} 筆 stock_daily_status")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def run(
    target_date: Optional[date] = None,
    backfill_days: int = 0,
    stock_filter: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    if target_date is None:
        target_date = date.today()

    # ── 獲取標的名單 (DB Driven) ──
    conn = _connect()
    try:
        stock_configs = get_core_stocks_from_db(conn)
        stock_ids = list(stock_configs.keys())
    finally:
        conn.close()

    if stock_filter:
        if stock_filter in stock_ids:
            stock_ids = [stock_filter]
        else:
            logger.error(f"找不到股票 {stock_filter}（不具備核心標記或已停用）")
            return

    logger.info(f"目標股票數：{len(stock_ids)}（過濾：{stock_filter or 'ALL'}）")
    logger.info(f"日頻表清單（{len(DAILY_TABLES)} 張）：{sorted(DAILY_TABLES)}")

    if backfill_days > 0:
        dates = [target_date - timedelta(days=i) for i in range(backfill_days)]
    else:
        dates = [target_date]

    conn = _connect()
    try:
        ensure_status_table(conn)

        total = 0
        red_total = 0
        yellow_total = 0
        green_total = 0

        for d in dates:
            logger.info(f"─── 處理日期 {d} ───")
            rows = build_status_for_date(conn, d, stock_ids)
            upsert_rows(conn, rows, dry_run=dry_run)

            day_red    = sum(1 for r in rows if r.health_status == "red")
            day_yellow = sum(1 for r in rows if r.health_status == "yellow")
            day_green  = sum(1 for r in rows if r.health_status == "green")
            logger.info(
                f"   健康燈號：🟢 {day_green}  🟡 {day_yellow}  🔴 {day_red}"
            )

            total += len(rows)
            red_total += day_red
            yellow_total += day_yellow
            green_total += day_green

            # 列出當日紅燈標的供告警系統取用
            if day_red > 0 and not dry_run:
                red_list = [r.stock_id for r in rows if r.health_status == "red"]
                logger.warning(f"   🔴 紅燈清單（{len(red_list)}）：{red_list[:20]}"
                               + ("..." if len(red_list) > 20 else ""))

        logger.info(
            f"═══ 完成：共 {total} 筆 │ 🟢 {green_total} │ "
            f"🟡 {yellow_total} │ 🔴 {red_total} ═══"
        )
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="從事件表（fetch_log/signal_history/...）聚合 stock_daily_status",
    )
    parser.add_argument("--date", type=str,
                        help="目標日期 YYYY-MM-DD（預設 today）")
    parser.add_argument("--backfill", type=int, default=0,
                        help="補抓過去 N 天（含 target_date）")
    parser.add_argument("--stock", type=str,
                        help="只處理單一 stock_id（除錯用）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不寫入 DB，只印出統計")
    args = parser.parse_args()

    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"--date 格式錯誤：{args.date}（應為 YYYY-MM-DD）")
            sys.exit(2)

    run(
        target_date=target_date,
        backfill_days=args.backfill,
        stock_filter=args.stock,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
