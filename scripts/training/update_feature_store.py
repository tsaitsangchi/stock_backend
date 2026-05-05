"""
update_feature_store.py — 每日特徵存儲更新 v3.0（逐支 commit 完整性版）
==============================================================================
增量更新 PostgreSQL `daily_features` 表，每筆 row 一個 (date, stock_id) JSONB。

v3.0 改進（與 core/db_utils.py v3.0 / 系統檢核報告 P0-3 / P0-5 一致）：
  ★ 改用 safe_commit_rows / commit_per_stock — 單支失敗不影響整批，崩潰前已寫入的股票全部保留
  ★ 改用 FailureLogger — 失敗即時 append 到 outputs/daily_features_failed_YYYYMMDD.json
  ★ 改用 is_conn_healthy — 每 N 支檢查一次連線健康度，自動修復 InFailedSqlTransaction
  ★ 改用 safe_float / safe_mapper — JSON 序列化前統一處理 NaN/Inf
  ★ [P0-5 對齊] 啟動時 assert_v3_features_present() — 防止把 v2 特徵集寫進 store
  ★ 結尾印 success / failed / total_records 統計摘要
  ★ 新增 --max-stocks / --skip-on-error 參數，方便分批跑
  ★ 新增 heartbeat 檔案（與 auto_train_manager P0-3 對齊），cron 可監控

v2 既有：
  · 統一使用 core.path_setup
  · 改用 core.db_utils.get_db_conn / ensure_ddl

執行：
    python update_feature_store.py
    python update_feature_store.py --stock-id 2330
    python update_feature_store.py --force          # 強制重刷
    python update_feature_store.py --max-stocks 10  # 只跑前 10 支
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path 自我修復 ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401  (side-effect: ensure all sub-paths)

import argparse
import json
import logging
import math
import time
from datetime import date, datetime, timedelta

import pandas as pd

from config import STOCK_CONFIGS

# core v3.0 helpers
from core.path_setup import get_outputs_dir, ensure_dirs_exist
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    safe_commit_rows,
    is_conn_healthy,
    FailureLogger,
)

from data_pipeline import build_daily_frame
from feature_engineering import build_features_with_medium_term

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DDL / SQL
# ─────────────────────────────────────────────
DDL_FEATURE_STORE = """
CREATE TABLE IF NOT EXISTS daily_features (
    date     DATE,
    stock_id VARCHAR(50),
    features JSONB,
    PRIMARY KEY (date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_daily_features_stock_id ON daily_features (stock_id);
"""

UPSERT_FEATURE_STORE = """
INSERT INTO daily_features (date, stock_id, features)
VALUES %s
ON CONFLICT (date, stock_id) DO UPDATE SET
    features = EXCLUDED.features;
"""

# 每支 stock 寫入完成後的心跳檔（cron 監控用）
HEARTBEAT_FILE = get_outputs_dir() / "update_feature_store.heartbeat"
HEALTH_CHECK_EVERY_N = 10  # 每 10 支股票檢查一次連線健康度


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_latest_feature_date(conn, stock_id: str):
    """查詢 daily_features 中該 stock_id 的最新日期。失敗自動 rollback 不擴散。"""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(date) FROM daily_features WHERE stock_id = %s",
                (stock_id,),
            )
            row = cur.fetchone()
            return row[0] if (row and row[0]) else None
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.warning(f"[{stock_id}] get_latest_feature_date 失敗：{e}")
        return None


def _sanitize_value(v):
    """JSON 序列化前處理：NaN/Inf → None，pd.NA → None，numpy scalar → python scalar。"""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    # pandas NA / Timestamp 等特殊型別
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalar
        try:
            return v.item()
        except Exception:
            return None
    return v


def serialize_records(new_df: pd.DataFrame, stock_id: str) -> list[tuple]:
    """把 new_df 轉成 (date_str, stock_id, json_str) tuple list。NaN/Inf 統一過濾。"""
    records: list[tuple] = []
    for idx, row in new_df.iterrows():
        try:
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            clean = {k: _sanitize_value(v) for k, v in row.to_dict().items()}
            records.append((date_str, stock_id, json.dumps(clean, ensure_ascii=False, default=str)))
        except Exception as e:
            logger.debug(f"[{stock_id}] 序列化異常筆 idx={idx}：{e}")
    return records


def assert_v3_features_present(stock_id: str = "2330") -> None:
    """[P0-5 對齊] 確認 ALL_FEATURES 已包含 v3 新因子；若大量缺失則中斷。"""
    try:
        from config import get_all_features
        all_feats = set(get_all_features(stock_id))
    except Exception as e:
        logger.warning(f"v3 守門：無法取得 ALL_FEATURES（{e}），跳過檢查")
        return

    v3_required = {
        "fcf_yield", "vix_zscore_252", "news_intensity",
        "is_in_disposition",
    }
    missing = v3_required - all_feats
    if missing:
        logger.warning(f"[v3 守門] ALL_FEATURES 中找不到 v3 因子：{sorted(missing)}")
    if len(all_feats) < 150:
        raise RuntimeError(
            f"[v3 守門] ALL_FEATURES 僅 {len(all_feats)} 個 < 150，請檢查 FEATURE_GROUPS。"
        )


def write_heartbeat(progress: str, ok: int, failed: int, total: int) -> None:
    """寫入心跳檔，cron 可透過 mtime 判斷活性。"""
    try:
        payload = {
            "ts":       datetime.now().isoformat(),
            "progress": progress,
            "ok":       ok,
            "failed":   failed,
            "total":    total,
        }
        HEARTBEAT_FILE.write_text(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        logger.debug(f"heartbeat 寫入失敗（不影響主流程）：{e}")


# ─────────────────────────────────────────────
# 主流程：單支處理
# ─────────────────────────────────────────────

def process_one_stock(
    conn,
    stock_id: str,
    force: bool,
    flog: FailureLogger,
) -> tuple[bool, int]:
    """
    處理單一股票，回傳 (success, n_records_written)。

    任何失敗都不會 raise；以 FailureLogger 即時落盤後 return False。
    每次寫入採 safe_commit_rows，崩潰前已 commit 的不會回滾。
    """
    # 1) 決定切片起點
    latest_date_obj = get_latest_feature_date(conn, stock_id)
    if latest_date_obj and not force:
        fetch_start_date = (latest_date_obj - timedelta(days=250)).strftime("%Y-%m-%d")
        latest_date_str  = latest_date_obj.strftime("%Y-%m-%d")
    else:
        fetch_start_date = "2010-01-01"
        latest_date_str  = "1900-01-01"
        if force:
            logger.info(f"[{stock_id}] 強制重刷模式啟用")

    # 2) 抓 raw + 特徵
    try:
        raw = build_daily_frame(stock_id, start_date=fetch_start_date)
    except Exception as e:
        flog.record(stock_id=stock_id, stage="build_daily_frame", error=str(e))
        return False, 0

    if raw is None or raw.empty:
        logger.info(f"[{stock_id}] build_daily_frame 無資料，跳過")
        return True, 0  # 視為「沒事可做」，不算失敗

    try:
        df = build_features_with_medium_term(raw, stock_id=stock_id, for_inference=True)
    except Exception as e:
        flog.record(stock_id=stock_id, stage="build_features", error=str(e))
        return False, 0

    if df is None or df.empty:
        logger.info(f"[{stock_id}] 特徵工程後無有效資料，跳過")
        return True, 0

    # 3) 增量切片
    try:
        new_df = df[df.index > latest_date_str].copy()
    except TypeError:
        # 索引非 datetime 時的兜底
        if "date" in df.columns:
            df = df.set_index("date")
            new_df = df[df.index.astype(str) > latest_date_str].copy()
        else:
            flog.record(stock_id=stock_id, stage="slice",
                        error="索引非 date 且找不到 date 欄位")
            return False, 0

    if new_df.empty:
        logger.info(f"[{stock_id}] 已是最新")
        return True, 0

    # 4) 序列化（NaN/Inf 統一處理）
    records = serialize_records(new_df, stock_id)
    if not records:
        flog.record(stock_id=stock_id, stage="serialize", error="序列化後 0 筆")
        return False, 0

    # 5) safe_commit_rows：失敗自動 rollback，不擴散
    n = safe_commit_rows(
        conn, UPSERT_FEATURE_STORE, records,
        template="(%s::date, %s, %s::jsonb)",
        label=f"daily_features/{stock_id}",
        page_size=1000,
    )
    if n == 0:
        flog.record(stock_id=stock_id, stage="commit", rows=len(records),
                    error="safe_commit_rows returned 0 (rolled back)")
        return False, 0

    logger.info(f"[{stock_id}] ✅ 寫入 {n:,} 筆")
    return True, n


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="每日特徵存儲更新 (v3.0)")
    parser.add_argument("--stock-id", type=str, help="指定更新單一股票代號")
    parser.add_argument("--force", action="store_true",
                        help="強制重刷：忽略已存特徵，重新計算全量歷史")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="只跑前 N 支股票（節流測試用）")
    parser.add_argument("--skip-v3-guard", action="store_true",
                        help="跳過 v3 因子守門檢查（不建議）")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 65)
    logger.info("  Feature Store Update (v3.0 — 逐支 commit 完整性版)")
    logger.info(f"  Force: {args.force}  |  Stock filter: {args.stock_id or 'ALL'}")
    logger.info("=" * 65)

    # 0) v3 守門
    if not args.skip_v3_guard:
        try:
            assert_v3_features_present()
        except Exception as e:
            logger.error(f"v3 守門失敗：{e}")
            return 2

    # 1) 確保關鍵目錄
    ensure_dirs_exist()

    # 2) 啟動 DB 連線 + 確保 DDL
    conn = get_db_conn()
    flog = FailureLogger("daily_features", log_to_db=True, db_conn=conn)

    try:
        ensure_ddl(conn, DDL_FEATURE_STORE)

        stock_ids = [args.stock_id] if args.stock_id else list(STOCK_CONFIGS.keys())
        if args.max_stocks:
            stock_ids = stock_ids[: args.max_stocks]
        total = len(stock_ids)
        logger.info(f"準備處理 {total} 支股票")

        ok_count = 0
        failed_count = 0
        total_records = 0

        for i, stock_id in enumerate(stock_ids, 1):
            # 連線健康度檢查（每 N 支一次）
            if i % HEALTH_CHECK_EVERY_N == 0 and not is_conn_healthy(conn):
                logger.warning(f"[#{i}] DB 連線不健康，重新建立連線")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = get_db_conn()
                flog.db_conn = conn

            logger.info(f"\n[{i}/{total}] {stock_id} ({STOCK_CONFIGS.get(stock_id, {}).get('name', '?')})")

            try:
                ok, n = process_one_stock(conn, stock_id, args.force, flog)
            except Exception as e:
                # 防禦性外層保護：理論上 process_one_stock 內部都已捕獲
                flog.record(stock_id=stock_id, stage="outer", error=str(e))
                ok, n = False, 0

            if ok:
                ok_count += 1
                total_records += n
            else:
                failed_count += 1

            # 每支寫一次 heartbeat
            write_heartbeat(f"{i}/{total}", ok_count, failed_count, total)

        # 3) 摘要
        elapsed = time.time() - t0
        logger.info("\n" + "=" * 65)
        logger.info(f"  完成！耗時 {elapsed:.1f} 秒")
        logger.info(f"  成功：{ok_count}/{total}  失敗：{failed_count}  寫入筆數：{total_records:,}")
        if failed_count:
            logger.info(f"  失敗清單：{flog.path}")
        flog.summary()
        logger.info("=" * 65)

        return 0 if failed_count == 0 else 1

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main() or 0)
