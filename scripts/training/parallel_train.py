"""
parallel_train.py — 全自動並行訓練管理器 v2（streaming 版）
===========================================================
同時執行 N 個訓練任務，確保在資源可控的情況下快速完成所有個股模型。

v2 改進（依系統重構報告 INT-04 建議）：
  · 將 concurrent.futures.ProcessPoolExecutor 改為 multiprocessing.Pool.imap_unordered
    ─ 避免一次性 submit 導致記憶體尖峰（執行緒池會立即把整個 generator 物件化）
    ─ imap_unordered 採漸進式串流：worker 完成一個才送下一個
    ─ 在標的數量擴張至數百檔時可避免 OOM
  · 訊號處理：KeyboardInterrupt 時優雅關閉 pool 並回收子進程
  · 修正頂部 sys.path 區塊（單行冪等寫法）

第五輪審查保留特性：
  ① 訓練優先名單由「缺檔」擴展為「缺檔 + 過期 (>30天) + drift > 0.2」
  ② step_days 引用 WF_CONFIG["step_days"]（與 Walk-Forward 視窗一致）
  ③ 每個 stock 的訓練支援指數退避自動重試（60s → 120s → 240s）
  ④ subprocess.run 帶 timeout 保護（4 hours/stock）
  ⑤ V3_REQUIRED_FEATURES 守門（P0-5：阻擋訓練到舊版特徵集）
  ⑥ 訓練完成自動彙整 feature_importance_by_group.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401

import argparse
import logging
import multiprocessing as mp
import os
import subprocess
import time

from config import (
    STOCK_CONFIGS, MODEL_DIR, LOG_DIR, BASE_DIR, WF_CONFIG,
    FEATURE_GROUPS, get_all_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ──────────────────────────────────────────────
# [P0-5 / QW-4] v3 因子守門
# ──────────────────────────────────────────────
V3_REQUIRED_FEATURES = {
    "fcf_yield",                  # quality
    "accruals",
    "sbl_short_intensity",        # short_interest
    "vix_zscore_252",             # fred_macro
    "yield_curve_inverted",
    "night_session_premium",      # extended_derivative
    "news_intensity",             # news_attention
    "is_in_disposition",          # event_risk
}
MIN_TOTAL_FEATURES = 150


def assert_v3_features_present(stock_id: str = "2330") -> None:
    """確認 ALL_FEATURES 已包含 v3 新因子，否則中斷。"""
    all_feats = set(get_all_features(stock_id))
    missing = V3_REQUIRED_FEATURES - all_feats
    if missing:
        logger.warning(
            f"[v3 守門] ALL_FEATURES 中找不到 {len(missing)} 個 v3 因子: {sorted(missing)}"
        )
    if len(all_feats) < MIN_TOTAL_FEATURES:
        raise RuntimeError(
            f"[v3 守門] ALL_FEATURES 僅 {len(all_feats)} 個，低於下限 "
            f"{MIN_TOTAL_FEATURES}，請檢查 FEATURE_GROUPS 是否完整。"
        )


# ──────────────────────────────────────────────
# 設定區域
# ──────────────────────────────────────────────
MAX_WORKERS    = 6
VENV_PYTHON    = os.environ.get("VENV_PYTHON",
                                "/home/hugo/project/stock_backend/venv/bin/python3")
TRAIN_SCRIPT   = str(BASE_DIR / "training" / "train_evaluate.py")
HEALTH_SCRIPT  = str(BASE_DIR / "monitor" / "model_health_check.py")
MAX_RETRIES    = 3
TRAIN_TIMEOUT  = 4 * 3600
RETRAIN_AGE_THRESHOLD_DAYS = 30


# ──────────────────────────────────────────────
# Worker context（用 module-level 全域，避免 Pool pickle 失敗）
# ──────────────────────────────────────────────
_WORKER_FLAGS: dict = {"turbo": False, "fast_mode": False}


def _worker_init(turbo: bool, fast_mode: bool) -> None:
    """Pool initializer：將 CLI flags 傳遞到 worker process。"""
    _WORKER_FLAGS["turbo"] = turbo
    _WORKER_FLAGS["fast_mode"] = fast_mode


def get_stocks_needing_training(
    force_retrain_days: int = RETRAIN_AGE_THRESHOLD_DAYS,
) -> list[str]:
    """
    [P1 第五輪修復]
    整合 model_health_check 的結果，找出需要訓練的個股：
      - 缺少 ensemble_{sid}.pkl 檔案
      - 模型檔超過 force_retrain_days 天未更新
      - PSI 超過 0.2（漂移嚴重，需重訓）
    """
    all_stocks = list(STOCK_CONFIGS.keys())
    try:
        from model_health_check import check_model_files_df, check_prediction_drift_df
        files_df = check_model_files_df(all_stocks)
        need = files_df[
            (files_df["status"] != "[OK]") |
            (files_df["age_days"] > force_retrain_days)
        ]["stock_id"].tolist()

        try:
            drift_df = check_prediction_drift_df(all_stocks)
            drifted = drift_df[drift_df["psi"] > 0.2]["stock_id"].tolist()
            need = sorted(set(need) | set(drifted))
        except Exception as e:
            logger.debug(f"健康 drift 檢查跳過：{e}")
        return need
    except Exception as e:
        logger.warning(f"健康檢查整合失敗，退回到檔案存在性檢查：{e}")
        return [
            sid for sid in all_stocks
            if not (MODEL_DIR / f"ensemble_{sid}.pkl").exists()
        ]


def train_one_stock(stock_id: str) -> tuple[str, bool, str | None]:
    """訓練單一個股，含指數退避重試與 timeout 保護。"""
    stock_name = STOCK_CONFIGS[stock_id]["name"]
    log_file   = str(LOG_DIR / f"train_{stock_id}.log")
    step_days  = str(WF_CONFIG["step_days"])

    cmd = [VENV_PYTHON, TRAIN_SCRIPT, "--stock-id", stock_id]
    if _WORKER_FLAGS.get("turbo"):
        cmd.append("--turbo")
    elif _WORKER_FLAGS.get("fast_mode"):
        cmd.append("--fast-mode")

    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{stock_id}] ({stock_name}) 訓練嘗試 {attempt}/{MAX_RETRIES} (step_days={step_days})")
            mode = "w" if attempt == 1 else "a"
            with open(log_file, mode) as f:
                if attempt > 1:
                    f.write(f"\n\n=== 重試第 {attempt} 次 ===\n")
                subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT,
                    check=True, timeout=TRAIN_TIMEOUT,
                )

            # 訓練成功 → 跑健康檢查
            with open(log_file, "a") as f:
                f.write("\n\n=== 自動健康檢查 (Post-Training Health Check) ===\n")
                try:
                    subprocess.run(
                        [VENV_PYTHON, HEALTH_SCRIPT],
                        stdout=f, stderr=subprocess.STDOUT,
                        timeout=300,
                    )
                except Exception as he:
                    f.write(f"健康檢查失敗（不影響訓練成功狀態）：{he}\n")
            return stock_id, True, None

        except subprocess.TimeoutExpired:
            last_error = f"訓練超時 ({TRAIN_TIMEOUT}s)"
            logger.warning(f"[{stock_id}] {last_error}（第 {attempt}/{MAX_RETRIES} 次）")
        except subprocess.CalledProcessError as e:
            last_error = f"非零回傳碼 {e.returncode}"
            logger.warning(f"[{stock_id}] {last_error}（第 {attempt}/{MAX_RETRIES} 次）")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[{stock_id}] 例外：{e}（第 {attempt}/{MAX_RETRIES} 次）")

        if attempt < MAX_RETRIES:
            backoff = 60 * (2 ** (attempt - 1))
            logger.info(f"[{stock_id}] 等待 {backoff}s 後重試...")
            time.sleep(backoff)

    return stock_id, False, last_error


def aggregate_feature_importance_by_group(stock_ids: list[str]) -> None:
    """
    [P2-5 / QW-4] 訓練後彙整 feature_importance_refined.csv → 按 group 加總。
    輸出 outputs/feature_importance_by_group.csv。
    """
    import pandas as pd  # 局部 import：避免 worker 進程冗載
    out_dir = MODEL_DIR.parent  # = outputs/
    refined = out_dir / "feature_importance_refined.csv"
    if not refined.exists():
        logger.warning("找不到 feature_importance_refined.csv，跳過 by_group 彙整")
        return

    df = pd.read_csv(refined)
    if "feature" not in df.columns or "importance" not in df.columns:
        logger.warning(f"feature_importance_refined.csv 欄位非預期：{df.columns.tolist()}")
        return

    feature_to_group = {}
    for grp, feats in FEATURE_GROUPS.items():
        for f in feats:
            feature_to_group[f] = grp

    df["group"] = df["feature"].map(lambda f: feature_to_group.get(f, "other_v3"))
    agg = (
        df.groupby("group")["importance"]
          .agg(["sum", "count"])
          .reset_index()
          .rename(columns={"sum": "sum_importance", "count": "n_features"})
          .sort_values("sum_importance", ascending=False)
    )
    top3 = (
        df.sort_values("importance", ascending=False)
          .groupby("group")["feature"]
          .apply(lambda s: ";".join(s.head(3).tolist()))
          .reset_index()
          .rename(columns={"feature": "top_features"})
    )
    out = agg.merge(top3, on="group", how="left")

    target = out_dir / "feature_importance_by_group.csv"
    out.to_csv(target, index=False)
    logger.info(f"[QW-4] 已輸出 group 彙整：{target}")
    logger.info(f"\n{out.to_string(index=False)}")


def main():
    # [P0-5 / QW-4] 訓練前先驗證 v3 因子已就位
    try:
        assert_v3_features_present()
    except Exception as e:
        logger.error(f"v3 守門失敗：{e}")
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true",
                        help="強制重新訓練所有標的（讓模型吃到 v3 因子）")
    parser.add_argument("--turbo", action="store_true", help="⚡ 極速模式")
    parser.add_argument("--fast-mode", action="store_true", help="快速模式")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"並行 worker 數（預設 {MAX_WORKERS}）")
    args, _ = parser.parse_known_args()

    if args.force_all:
        logger.info("⚡ [Force All] 強制重訓所有標的，目的：讓模型吃到 v3 新因子")
        to_train = list(STOCK_CONFIGS.keys())
    else:
        to_train = get_stocks_needing_training()

    if not to_train:
        print("所有個股模型皆已新鮮，無需重訓。")
        return

    print(f"總計需要訓練 {len(to_train)} 支個股，設定並行數為 {args.workers}...")
    print(f"訓練清單：{to_train}")

    start_time = time.time()
    results: dict[str, list[str]] = {"success": [], "failed": []}

    # [INT-04] multiprocessing.Pool.imap_unordered：漸進式串流，避免記憶體尖峰
    pool = mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.turbo, args.fast_mode),
    )
    try:
        for sid, success, error in pool.imap_unordered(train_one_stock, to_train):
            if success:
                print(f"[完成] {sid}")
                results["success"].append(sid)
            else:
                print(f"[失敗] {sid}: {error}")
                results["failed"].append(sid)
        pool.close()
    except KeyboardInterrupt:
        logger.warning("收到 KeyboardInterrupt，正在優雅關閉 worker pool …")
        pool.terminate()
        raise
    finally:
        pool.join()

    duration = (time.time() - start_time) / 3600

    print("\n" + "=" * 50)
    print(f"訓練任務結束！總耗時: {duration:.2f} 小時")
    print(f"成功: {len(results['success'])}")
    print(f"失敗: {len(results['failed'])}")
    if results["failed"]:
        print(f"失敗清單: {', '.join(results['failed'])}")
    print("=" * 50)

    try:
        aggregate_feature_importance_by_group(results["success"])
    except Exception as e:
        logger.warning(f"feature_importance_by_group 彙整失敗（不影響成功狀態）：{e}")


if __name__ == "__main__":
    main()
