"""
tune_hyperparameters.py — 使用 Optuna 為特定個股尋找最佳超參數 v3.1
================================================================
v3.1 改進（在 v3.0 基礎上）：
  ★ XGB / LGB 的 n_estimators 500 → 1000（搭配 ES 仍會早停，無代價）
  ★ LGB early_stopping(50, verbose=False) — 抑制每 trial 的雜訊輸出

v3.0 改進（與 core v3.0 helpers / 系統檢核報告 P3-2 / WF_CONFIG 對齊）：
  ★ 路徑改用 core.path_setup.get_outputs_dir() — 不再寫死 "scripts/outputs"
  ★ 結果改用 core.model_metadata.atomic_write_json + ModelMetadata —
    崩潰不留半份檔；附 git_hash / feature_count / horizon / seed
  ★ 加 --horizon 參數，預設讀 config.HORIZON（與其他模組一致）
  ★ 切分加入 WF_CONFIG["embargo_days"] 緩衝（防 train/val 邊界資訊洩漏）
  ★ XGB 改用 callbacks=[xgb.callback.EarlyStopping(rounds=50)]
  ★ XGB / LGB 各自獨立 sampler seed（避免兩 study 共用內部狀態）
  ★ 空 study 防呆：best_trial 不存在則不寫檔，回傳 exit 1
  ★ [P0-5 對齊] assert_v3_features_present()
  ★ 收尾 log Optuna 統計：completed / pruned / failed counts、best_trial.duration

執行：
    python tune_hyperparameters.py
    python tune_hyperparameters.py --stock-id 2330 --trials 50 --horizon 5
    python tune_hyperparameters.py --no-embargo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401  (side-effect: ensure all sub-paths)

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import joblib
import numpy as np

try:
    import optuna
except ModuleNotFoundError:
    sys.stderr.write(
        "\n[ERROR] 找不到 optuna。tune_hyperparameters.py 需要它。\n"
        "請在 venv 中執行：\n"
        "    pip install 'optuna>=3.5.0'\n"
        "或：\n"
        "    pip install -r requirements.txt\n\n"
    )
    sys.exit(127)

from config import HORIZON, TRAIN_START_DATE, WF_CONFIG, STOCK_CONFIGS, get_all_features

# core v3.0 helpers
from core.path_setup import get_outputs_dir, ensure_dirs_exist
from core.model_metadata import ModelMetadata, atomic_write_json

from data_pipeline import build_daily_frame
from feature_engineering import build_features

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# v3 因子守門（與 P0-5 / parallel_train 對齊）
# ─────────────────────────────────────────────
V3_REQUIRED = {"fcf_yield", "vix_zscore_252", "news_intensity", "is_in_disposition"}
MIN_TOTAL_FEATURES = 150


def assert_v3_features_present(stock_id: str) -> None:
    feats = set(get_all_features(stock_id))
    missing = V3_REQUIRED - feats
    if missing:
        logger.warning(f"[v3 守門] ALL_FEATURES 缺少 v3 因子：{sorted(missing)}")
    if len(feats) < MIN_TOTAL_FEATURES:
        raise RuntimeError(
            f"[v3 守門] ALL_FEATURES 僅 {len(feats)} 個 < {MIN_TOTAL_FEATURES}"
        )


def _git_hash() -> Optional[str]:
    """取得當前 git short hash，方便 metadata 追蹤。"""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return None


# ─────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────

def objective(trial, X_tr, y_tr, X_va, y_va, model_type: str = "xgb") -> float:
    """單次 trial：以 MSE 作為目標。"""
    if model_type == "xgb":
        import xgboost as xgb
        params = {
            "n_estimators":      1000,  # [v3.1] 500→1000
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 0.5),
            "random_state":      42,
            "n_jobs":            -1,
        }
        try:
            es_cb = xgb.callback.EarlyStopping(rounds=50, save_best=True, maximize=False)
            model = xgb.XGBRegressor(**params, callbacks=[es_cb])
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except (AttributeError, TypeError):
            model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)

    else:
        import lightgbm as lgb
        params = {
            "n_estimators":      1000,  # [v3.1] 500→1000
            "max_depth":         trial.suggest_int("max_depth", -1, 15),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state":      42,
            "n_jobs":            -1,
        }
        model = lgb.LGBMRegressor(**params, verbose=-1)
        # [v3.1] verbose=False 抑制 "Training until validation scores don't improve..." 雜訊
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        preds = model.predict(X_va)

    mse = float(np.mean((y_va - preds) ** 2))
    trial.report(mse, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return mse


# ─────────────────────────────────────────────
# 摘要：印 study 統計
# ─────────────────────────────────────────────

def _summarize_study(name: str, study: optuna.Study) -> dict:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    summary = {
        "n_trials":   len(study.trials),
        "completed":  len(completed),
        "pruned":     len(pruned),
        "failed":     len(failed),
    }
    try:
        bt = study.best_trial
        summary.update({
            "best_value":         bt.value,
            "best_trial_number":  bt.number,
            "best_duration_sec":  bt.duration.total_seconds() if bt.duration else None,
        })
    except (ValueError, RuntimeError):
        summary["best_value"] = None

    logger.info(
        f"[{name}] 統計：completed={len(completed)} / pruned={len(pruned)} "
        f"/ failed={len(failed)}, best_value={summary.get('best_value')}"
    )
    return summary


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def tune_one_stock(stock_id: str, args) -> int:
    """對單支股票執行調優，回傳 0=成功 / 1=失敗 / 2=守門失敗。"""
    logger.info("=" * 65)
    logger.info(f"  Optuna Hyperparameter Tuning v3.1 — {stock_id}")
    logger.info(f"  trials={args.trials}  horizon={args.horizon}d  seed={args.seed}")
    logger.info("=" * 65)

    if not args.skip_v3_guard:
        try:
            assert_v3_features_present(stock_id)
        except Exception as e:
            logger.error(f"v3 守門失敗：{e}")
            return 2

    raw = build_daily_frame(stock_id=stock_id, start_date=TRAIN_START_DATE)
    df  = build_features(raw, stock_id=stock_id)

    df = df.replace([np.inf, -np.inf], np.nan)

    target_col = f"target_{args.horizon}d"
    if target_col not in df.columns:
        if "target_30d" in df.columns:
            logger.warning(f"找不到 {target_col}，退回 target_30d")
            target_col = "target_30d"
        else:
            logger.error(f"❌ 找不到目標欄位 {target_col} 也找不到 target_30d")
            return 1

    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    df = df.dropna(axis=1, how="all")
    df = df.fillna(0)
    logger.info(f"資料清理完成：{initial_len} → {len(df)} 筆")

    if len(df) < 100:
        logger.warning(f"⚠️ {stock_id} 有效樣本過少 ({len(df)})，跳過調優")
        return 1

    all_features = get_all_features(stock_id)
    feat_cols = [c for c in all_features if c in df.columns]
    X = df[feat_cols].fillna(0)
    y = df[target_col]

    val_window = args.val_window
    embargo = 0 if args.no_embargo else WF_CONFIG.get("embargo_days", 45)
    val_start = -val_window
    train_end = val_start - embargo
    if abs(train_end) >= len(df):
        logger.error("資料量不足以套用 embargo + val_window，請降低 --val-window 或 --no-embargo")
        return 1
    X_tr, X_va = X.iloc[:train_end], X.iloc[val_start:]
    y_tr, y_va = y.iloc[:train_end], y.iloc[val_start:]
    logger.info(f"切分：train={len(X_tr)}  embargo={embargo}  val={len(X_va)}")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    sampler_xgb = optuna.samplers.TPESampler(seed=args.seed)
    sampler_lgb = optuna.samplers.TPESampler(seed=args.seed + 1)

    logger.info("\n[XGB] 開始調優...")
    t_xgb_start = time.time()
    study_xgb = optuna.create_study(direction="minimize", sampler=sampler_xgb, pruner=pruner)
    study_xgb.optimize(
        lambda t: objective(t, X_tr, y_tr, X_va, y_va, "xgb"),
        n_trials=args.trials,
    )
    xgb_summary = _summarize_study("XGB", study_xgb)
    xgb_summary["wall_sec"] = round(time.time() - t_xgb_start, 1)

    logger.info("\n[LGB] 開始調優...")
    t_lgb_start = time.time()
    study_lgb = optuna.create_study(direction="minimize", sampler=sampler_lgb, pruner=pruner)
    study_lgb.optimize(
        lambda t: objective(t, X_tr, y_tr, X_va, y_va, "lgb"),
        n_trials=args.trials,
    )
    lgb_summary = _summarize_study("LGB", study_lgb)
    lgb_summary["wall_sec"] = round(time.time() - t_lgb_start, 1)

    has_xgb = xgb_summary.get("best_value") is not None
    has_lgb = lgb_summary.get("best_value") is not None
    if not (has_xgb or has_lgb):
        logger.error("❌ XGB / LGB 兩個 study 都沒有 completed trial，不寫檔")
        return 1

    out_dir = get_outputs_dir() / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl  = out_dir / f"best_params_{stock_id}.pkl"
    out_json = out_dir / f"best_params_{stock_id}.json"
    out_meta = out_dir / f"best_params_{stock_id}.metadata.json"

    result = {
        "stock_id":     stock_id,
        "horizon":      args.horizon,
        "target_col":   target_col,
        "n_trials":     args.trials,
        "seed":         args.seed,
        "embargo":      embargo,
        "val_window":   val_window,
        "git_hash":     _git_hash(),
        "trained_at":   datetime.now().isoformat(),
        "feature_count": len(feat_cols),
        "xgb": {
            "best_params": study_xgb.best_params if has_xgb else None,
            "best_mse":    study_xgb.best_value if has_xgb else None,
            **xgb_summary,
        },
        "lgb": {
            "best_params": study_lgb.best_params if has_lgb else None,
            "best_mse":    study_lgb.best_value if has_lgb else None,
            **lgb_summary,
        },
    }

    atomic_write_json(out_json, result)
    logger.info(f"✅ JSON 已寫入：{out_json}")

    tmp_pkl = out_pkl.with_suffix(out_pkl.suffix + ".tmp")
    joblib.dump(result, tmp_pkl)
    os.replace(tmp_pkl, out_pkl)
    logger.info(f"✅ Pickle 已寫入：{out_pkl}")

    try:
        import hashlib
        feat_fp = hashlib.sha256(",".join(sorted(feat_cols)).encode()).hexdigest()[:16]
        try:
            train_end_str = (
                df.index[train_end].strftime("%Y-%m-%d")
                if hasattr(df.index[train_end], "strftime")
                else str(df.index[train_end])
            )
        except Exception:
            train_end_str = "unknown"
        notes_str = (
            f"Optuna tuning: XGB={xgb_summary.get('best_value')}, "
            f"LGB={lgb_summary.get('best_value')}"
        )
        meta = ModelMetadata(
            stock_id=stock_id,
            model_path=str(out_pkl.relative_to(out_dir.parent)),
            train_end_date=train_end_str,
            feature_count=len(feat_cols),
            feature_fingerprint=feat_fp,
            git_hash=result["git_hash"],
            horizon_days=args.horizon,
            calibration_method=None,
            calibrator_cv=None,
            notes=notes_str,
        )
        atomic_write_json(out_meta, meta.to_dict())
        logger.info(f"✅ Metadata 已寫入：{out_meta}")
    except Exception as e:
        logger.warning(f"ModelMetadata 寫入失敗（不影響主流程）：{e}")

    logger.info("\n" + "=" * 65)
    if has_xgb:
        logger.info(f"  Best XGB MSE: {study_xgb.best_value:.6f}  params: {study_xgb.best_params}")
    if has_lgb:
        logger.info(f"  Best LGB MSE: {study_lgb.best_value:.6f}  params: {study_lgb.best_params}")
    logger.info("=" * 65)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Optuna 超參數調優 (v3.1)")
    parser.add_argument("--stock-id", default="2330",
                        help="單支股票 ID（與 --all-stocks 互斥）")
    parser.add_argument("--all-stocks", action="store_true",
                        help="對 STOCK_CONFIGS 所有股票批次調優（依序執行）")
    parser.add_argument("--skip-existing", action="store_true",
                        help="--all-stocks 模式下跳過已有 best_params_<id>.json 的股票")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="Optuna sampler seed")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help=f"目標 horizon（預設 config.HORIZON={HORIZON}）")
    parser.add_argument("--val-window", type=int, default=252,
                        help="驗證窗口長度（交易日，預設 252）")
    parser.add_argument("--no-embargo", action="store_true",
                        help="關閉 train/val 之間的 embargo 緩衝")
    parser.add_argument("--skip-v3-guard", action="store_true",
                        help="跳過 v3 因子守門（不建議）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ensure_dirs_exist()

    if args.all_stocks:
        stock_ids = list(STOCK_CONFIGS.keys())
        out_dir = get_outputs_dir() / "tuning"
        out_dir.mkdir(parents=True, exist_ok=True)

        succeeded, failed, skipped = [], [], []
        total = len(stock_ids)

        for idx, sid in enumerate(stock_ids, 1):
            if args.skip_existing and (out_dir / f"best_params_{sid}.json").exists():
                logger.info(f"[{idx}/{total}] {sid} 已有結果，跳過")
                skipped.append(sid)
                continue

            logger.info(f"\n{'='*65}")
            logger.info(f"[{idx}/{total}] 開始處理 {sid}")
            logger.info(f"{'='*65}")
            try:
                rc = tune_one_stock(sid, args)
                if rc == 0:
                    succeeded.append(sid)
                else:
                    failed.append(sid)
            except Exception as e:
                logger.error(f"[{sid}] 未預期錯誤：{e}", exc_info=True)
                failed.append(sid)

        logger.info("\n" + "=" * 65)
        logger.info(f"  批次調優完成：成功={len(succeeded)}  失敗={len(failed)}  跳過={len(skipped)}")
        if failed:
            logger.warning(f"  失敗清單：{failed}")
        logger.info("=" * 65)
        return 0 if not failed else 1

    else:
        return tune_one_stock(args.stock_id, args)


if __name__ == "__main__":
    sys.exit(main() or 0)
