"""
monitor/init_monitoring_schema.py
─────────────────────────────────────────────────────────────────────
一次建好「監控與可信度骨架」所需的全部資料表：

事件表（append-only，記錄每件事發生時的快照）：
  1. fetch_log              — 每次 fetcher 跑一筆
  2. model_training_log     — 每次訓練跑一筆
  3. prediction_output      — 每次 predict 一筆（pre-filter，模型原始輸出）
  4. signal_history         — 每次 signal_filter 一筆（post-filter，最終決策）

狀態表（每日聚合，dashboard 直接 query 它）：
  5. stock_daily_status     — 每股每天一筆健康燈號

執行方式：
    # 建表（idempotent，重複執行安全）
    python -m scripts.monitor.init_monitoring_schema

    # 只檢查目前狀態，不建表
    python -m scripts.monitor.init_monitoring_schema --check

    # 印出 DDL 但不執行（給 DBA review）
    python -m scripts.monitor.init_monitoring_schema --dry-run

職責切分：
    fetch_log              ─→「資料抓進來了沒」
    model_training_log     ─→「訓了哪個版本，OOF 多少」
    prediction_output      ─→「模型算出什麼」（pre-filter）
    signal_history         ─→「決定做什麼」（post-filter）
    stock_daily_status     ─→「整條 pipeline 健康嗎」（聚合視圖）
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import psycopg2

# ─────────────────────────────────────────────
# 將 scripts/ 加入 sys.path
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DB_CONFIG, LOG_DIR  # noqa: E402

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_FILE = Path(LOG_DIR) / "init_monitoring_schema.log"
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
# DDL 區段（每張表一段，方便單獨維護）
# ─────────────────────────────────────────────

# 1. fetch_log — 抓取事件
DDL_FETCH_LOG = """
CREATE TABLE IF NOT EXISTS fetch_log (
    id               BIGSERIAL    PRIMARY KEY,
    run_ts           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    table_name       VARCHAR(64)  NOT NULL,                 -- 對應 TABLE_REGISTRY 的 key
    stock_id         VARCHAR(20),                            -- NULL 表示市場層級表
    fetch_mode       VARCHAR(16),                            -- market / batch / per_stock / gap_fill / retry
    fetch_date_from  DATE,
    fetch_date_to    DATE,
    rows_inserted    INTEGER,
    rows_updated     INTEGER,
    duration_ms      INTEGER,
    status           VARCHAR(16)  NOT NULL,                  -- success / no_new_data / partial / failed / rate_limited / skipped
    error_message    TEXT,
    api_quota_left   INTEGER,                                -- FinMind quota 監控
    cli_args         TEXT                                     -- 執行時的 CLI 命令（可追溯）
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_table_ts  ON fetch_log(table_name, run_ts DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_log_stock_ts  ON fetch_log(stock_id,   run_ts DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_log_status_ts ON fetch_log(status,     run_ts DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_log_lookup    ON fetch_log(table_name, stock_id, run_ts DESC);

COMMENT ON TABLE fetch_log IS '每次 fetcher 執行寫一筆，data_integrity_audit 第 6 維度的數據源';
COMMENT ON COLUMN fetch_log.fetch_mode  IS 'market=市場層級不需 stock_id；batch=多股同一 API 呼叫；per_stock=逐支；gap_fill=由 fetch_log 反推補抓；retry=重試 failed';
COMMENT ON COLUMN fetch_log.status      IS 'success/no_new_data/partial/failed/rate_limited/skipped';
"""

# 1.5 feature_log — 特徵工程事件
DDL_FEATURE_LOG = """
CREATE TABLE IF NOT EXISTS feature_log (
    id               BIGSERIAL    PRIMARY KEY,
    run_ts           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    stock_id         VARCHAR(20)  NOT NULL,
    feature_count    INTEGER      NOT NULL,
    rows_processed   INTEGER      NOT NULL,
    nan_filled       INTEGER,
    inf_cleared      INTEGER,
    duration_ms      INTEGER,
    status           VARCHAR(16)  NOT NULL,                  -- success / failed
    error_message    TEXT,
    cli_args         TEXT
);
CREATE INDEX IF NOT EXISTS idx_feature_log_stock_ts  ON feature_log(stock_id, run_ts DESC);
CREATE INDEX IF NOT EXISTS idx_feature_log_status_ts ON feature_log(status, run_ts DESC);

COMMENT ON TABLE feature_log IS '每次執行特徵更新時寫入一筆，追蹤資料清理與品質狀態';
"""

# 2. model_training_log — 訓練事件
DDL_MODEL_TRAINING_LOG = """
CREATE TABLE IF NOT EXISTS model_training_log (
    run_id              VARCHAR(64)  PRIMARY KEY,             -- e.g. '2026-05-07_2330_a3f4b2'
    stock_id            VARCHAR(20)  NOT NULL,
    job_type            VARCHAR(16)  NOT NULL,                 -- tuning / training / validation
    started_at          TIMESTAMPTZ  NOT NULL,
    finished_at         TIMESTAMPTZ,
    duration_sec        INTEGER,
    status              VARCHAR(16),                           -- success / failed / killed / running
    -- 訓練設定
    git_commit_hash     VARCHAR(40),
    feature_count       INTEGER,                                -- 防 P0-5 重演（< 175 警告）
    hyperparams         JSONB,                                  -- 最佳參數快照
    train_start_date    DATE,
    train_end_date      DATE,
    n_folds             INTEGER,
    -- OOF 結果
    oof_da              NUMERIC(5,4),
    oof_ic              NUMERIC(5,4),
    oof_sharpe_gross    NUMERIC(6,3),
    oof_sharpe_net      NUMERIC(6,3),                           -- 含交易成本
    oof_max_drawdown    NUMERIC(6,3),
    n_trades            INTEGER,
    -- 模型檔
    model_path          TEXT,
    promoted_to_prod    BOOLEAN      DEFAULT FALSE,
    error_message       TEXT
);
CREATE INDEX IF NOT EXISTS idx_mtl_stock_finished ON model_training_log(stock_id, finished_at DESC);
CREATE INDEX IF NOT EXISTS idx_mtl_promoted      ON model_training_log(promoted_to_prod, finished_at DESC);
CREATE INDEX IF NOT EXISTS idx_mtl_status        ON model_training_log(status, finished_at DESC);

COMMENT ON TABLE model_training_log IS '每次模型訓練一筆，提供 versioning + rollback + A/B 比較基礎';
"""

# 3. prediction_output — 模型原始輸出（pre-filter）
DDL_PREDICTION_OUTPUT = """
CREATE TABLE IF NOT EXISTS prediction_output (
    date                DATE         NOT NULL,
    stock_id            VARCHAR(20)  NOT NULL,
    model_run_id        VARCHAR(32)  NOT NULL,                 -- → model_training_log.run_id
    -- 多時程校準後機率（主結果）
    prob_up_15d         NUMERIC(6,5),
    prob_up_21d         NUMERIC(6,5),
    prob_up_30d         NUMERIC(6,5),
    horizon_consensus   INTEGER,                                -- 0~3，三時程一致性
    -- Ensemble 子模型輸出（歸因 / debug）
    prob_xgb            NUMERIC(6,5),
    prob_lgb            NUMERIC(6,5),
    pred_elasticnet     NUMERIC(8,5),
    pred_momentum       NUMERIC(8,5),
    prob_stacking       NUMERIC(6,5),                           -- meta-learner 輸出
    -- TFT 分位數（可選）
    quantile_10         NUMERIC(8,5),
    quantile_25         NUMERIC(8,5),
    quantile_50         NUMERIC(8,5),
    quantile_75         NUMERIC(8,5),
    quantile_90         NUMERIC(8,5),
    -- Regime 與不確定性
    regime              VARCHAR(16),                            -- low_vol / mid_vol / high_vol
    regime_confidence   NUMERIC(5,4),
    pred_std            NUMERIC(8,5),
    n_models_used       INTEGER,                                -- 5=全員到齊；少於=有 fallback
    -- 特徵快照（重現性）
    feature_count       INTEGER,
    feature_hash        VARCHAR(64),                            -- SHA256(feature_vector)
    -- SHAP 歸因（P3-3）
    shap_top_positive   TEXT,                                   -- JSON: [{feat, shap, value}, ...]
    shap_top_negative   TEXT,
    -- 推論時序
    inference_ts        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    inference_ms        INTEGER,                                -- 監控 P0-2 timeout（45s）
    PRIMARY KEY (date, stock_id, model_run_id)
);
CREATE INDEX IF NOT EXISTS idx_pred_date_stock ON prediction_output(date DESC, stock_id);
CREATE INDEX IF NOT EXISTS idx_pred_run        ON prediction_output(model_run_id);
CREATE INDEX IF NOT EXISTS idx_pred_horizon    ON prediction_output(date DESC, horizon_consensus DESC);

COMMENT ON TABLE prediction_output IS '模型原始輸出（pre-filter），可用於 signal_filter 反事實 replay 與 SHAP dashboard';
COMMENT ON COLUMN prediction_output.model_run_id IS '對應 model_training_log.run_id；同 (date,stock_id) 可有多筆，支援 A/B';
"""

# 4. signal_history — 過濾後決策
DDL_SIGNAL_HISTORY = """
CREATE TABLE IF NOT EXISTS signal_history (
    date              DATE         NOT NULL,
    stock_id          VARCHAR(20)  NOT NULL,
    model_run_id      VARCHAR(32),                              -- → prediction_output / model_training_log
    decision          VARCHAR(16),                              -- LONG / HOLD_CASH / WATCH
    overall_score     NUMERIC(5,2),
    prob_up           NUMERIC(6,5),                             -- signal_filter 實際看到的單一機率
    prob_up_15d       NUMERIC(6,5),
    prob_up_21d       NUMERIC(6,5),
    prob_up_30d       NUMERIC(6,5),
    horizon_consensus INTEGER,
    blocking_reasons  TEXT,                                     -- JSON list
    boosting_reasons  TEXT,                                     -- JSON list
    filter_version    VARCHAR(16),                              -- signal_filter 版本標記（調 threshold 時 +1）
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_sh_decision ON signal_history(decision, date DESC);
CREATE INDEX IF NOT EXISTS idx_sh_stock    ON signal_history(stock_id, date DESC);

COMMENT ON TABLE signal_history IS '每股每天的最終決策（post-filter），3 個月後可做反事實分析';
"""

# 5. stock_daily_status — 每日健康狀態（聚合視圖）
DDL_STOCK_DAILY_STATUS = """
CREATE TABLE IF NOT EXISTS stock_daily_status (
    date                    DATE         NOT NULL,
    stock_id                VARCHAR(20)  NOT NULL,
    fetch_completed_tables  INTEGER,
    fetch_missing_tables    TEXT,                                -- JSON: ['stock_per','price_adj']
    last_data_date          DATE,
    data_lag_days           INTEGER,
    feature_built           BOOLEAN,
    feature_nan_rate        NUMERIC(5,4),
    feature_count           INTEGER,
    prediction_generated    BOOLEAN,
    prob_up                 NUMERIC(5,4),
    decision                VARCHAR(16),
    model_version           VARCHAR(32),
    health_status           VARCHAR(16),                         -- green / yellow / red
    health_reasons          TEXT,
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_sds_health ON stock_daily_status(date DESC, health_status);
CREATE INDEX IF NOT EXISTS idx_sds_stock  ON stock_daily_status(stock_id, date DESC);

COMMENT ON TABLE stock_daily_status IS '每股每天的整條 pipeline 健康燈號，dashboard 主要查詢介面';
"""

# ─────────────────────────────────────────────
# 表清單（供 main 與 check 共用）
# ─────────────────────────────────────────────
TABLES = [
    ("fetch_log",             DDL_FETCH_LOG,             "事件 │ 抓取"),
    ("feature_log",           DDL_FEATURE_LOG,           "事件 │ 特徵工程"),
    ("model_training_log",    DDL_MODEL_TRAINING_LOG,    "事件 │ 訓練"),
    ("prediction_output",     DDL_PREDICTION_OUTPUT,     "事件 │ 預測 (pre-filter)"),
    ("signal_history",        DDL_SIGNAL_HISTORY,        "事件 │ 訊號 (post-filter)"),
    ("stock_daily_status",    DDL_STOCK_DAILY_STATUS,    "狀態 │ 每日聚合"),
]


# ─────────────────────────────────────────────
# 工具函式
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


def _table_row_count(cur, table_name: str) -> int:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]
    except Exception:
        return -1


def _table_size(cur, table_name: str) -> str:
    try:
        cur.execute("""
            SELECT pg_size_pretty(pg_total_relation_size(%s))
        """, (f"public.{table_name}",))
        row = cur.fetchone()
        return row[0] if row else "?"
    except Exception:
        return "?"


# ─────────────────────────────────────────────
# 主功能
# ─────────────────────────────────────────────
def ensure_all_tables(conn) -> dict[str, str]:
    """建立所有表（idempotent）。回傳 {table: status} 字典。"""
    result: dict[str, str] = {}
    with conn.cursor() as cur:
        for table_name, ddl, label in TABLES:
            existed = _table_exists(cur, table_name)
            try:
                cur.execute(ddl)
                conn.commit()
                if existed:
                    result[table_name] = "kept"
                    logger.info(f"  ✓ {label:30s} {table_name:24s} 已存在（保留現有資料）")
                else:
                    result[table_name] = "created"
                    logger.info(f"  ★ {label:30s} {table_name:24s} 已建立")
            except Exception as e:
                conn.rollback()
                result[table_name] = f"error: {e}"
                logger.error(f"  ✗ {label:30s} {table_name:24s} 建立失敗：{e}")
    return result


def check_all_tables(conn) -> None:
    """只檢查不建表，列出每張表的存在狀態、列數、大小。"""
    logger.info("=" * 78)
    logger.info(f"  {'類型 │ 用途':30s} {'資料表':24s} {'狀態':10s} {'列數':>10s}  大小")
    logger.info("-" * 78)
    with conn.cursor() as cur:
        for table_name, _, label in TABLES:
            if _table_exists(cur, table_name):
                rows = _table_row_count(cur, table_name)
                size = _table_size(cur, table_name)
                logger.info(
                    f"  {label:30s} {table_name:24s} {'✅ 存在':10s} {rows:>10,d}  {size}"
                )
            else:
                logger.info(
                    f"  {label:30s} {table_name:24s} {'❌ 不存在':10s} {'-':>10s}  -"
                )
    logger.info("=" * 78)


def dump_ddl() -> None:
    """印出全部 DDL，供 DBA review。"""
    print("-- ═══════════════════════════════════════════════════════════════")
    print("-- 量子金融藍圖 監控與可信度骨架 DDL")
    print("-- ═══════════════════════════════════════════════════════════════")
    for table_name, ddl, label in TABLES:
        print(f"\n-- ── {label} ── {table_name} ─────────────────────────────")
        print(ddl.strip())
    print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="建立監控與可信度骨架的全部資料表（idempotent）",
    )
    parser.add_argument("--check", action="store_true",
                        help="只檢查表是否存在，不建表")
    parser.add_argument("--dry-run", action="store_true",
                        help="只印 DDL 不執行（給 DBA review）")
    args = parser.parse_args()

    if args.dry_run:
        dump_ddl()
        return

    conn = _connect()
    try:
        if args.check:
            check_all_tables(conn)
        else:
            logger.info("═" * 78)
            logger.info("  開始建立監控骨架")
            logger.info("═" * 78)
            ensure_all_tables(conn)
            logger.info("─" * 78)
            check_all_tables(conn)
            logger.info("═" * 78)
            logger.info("  ✅ 完成")
            logger.info("═" * 78)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
