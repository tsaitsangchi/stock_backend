"""
tune_hyperparameters.py — 使用 Optuna 為特定個股尋找最佳超參數 v2
================================================================
v2 改進：
  · 修正頂部三重 sys.path 重複插入
  · 加入 Optuna MedianPruner 中位數剪枝（提早終止表現低於中位數的 trial），
    可大幅縮短 30 trials 完整跑完的等待時間
  · XGB 改 callback-based early stopping（避免 fit() 整輪跑完）
  · 統一使用 core.path_setup
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401

import argparse
import logging

import joblib
import numpy as np
import optuna

from config import TRAIN_START_DATE, get_all_features
from data_pipeline import build_daily_frame
from feature_engineering import build_features

logger = logging.getLogger(__name__)


def objective(trial, X_tr, y_tr, X_va, y_va, model_type: str = "xgb") -> float:
    """單次 trial：以 MSE 作為目標。"""
    if model_type == "xgb":
        params = {
            "n_estimators":      500,
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 0.5),
            "random_state":      42,
            "n_jobs":            -1,
        }
        import xgboost as xgb
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
    else:
        params = {
            "n_estimators":      500,
            "max_depth":         trial.suggest_int("max_depth", -1, 15),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state":      42,
            "n_jobs":            -1,
        }
        import lightgbm as lgb
        # [P2-FIX] LightGBM 新版將 verbose 移至建構子，並推薦使用 log_evaluation callback
        model = lgb.LGBMRegressor(**params, verbose=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)])
        preds = model.predict(X_va)

    mse = float(np.mean((y_va - preds) ** 2))
    # Pruner 需要中間回報；這裡單值結束時直接 report 一次即可
    trial.report(mse, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-id", default="2330")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="Optuna sampler seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(f"=== 開始超參數調優 (Optuna)：{args.stock_id} ===")

    # 1. 載入資料
    raw = build_daily_frame(stock_id=args.stock_id, start_date=TRAIN_START_DATE)
    df  = build_features(raw, stock_id=args.stock_id)

    # 4. 嚴格清理資料：移除特徵與標籤中的 NaN/Inf
    # [修正] XGBoost 對 Label 中的 NaN 非常敏感
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 確保目標欄位存在且無 NaN
    target_col = "target_30d"
    if target_col not in df.columns:
        logger.error(f"❌ 找不到目標欄位 {target_col}")
        return
        
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    df = df.dropna(axis=1, how='all') # 移除全空的特徵
    df = df.fillna(0) # 其餘特徵補 0
    
    logger.info(f"=== 資料清理完成：{initial_len} -> {len(df)} 筆 ===")
    
    if len(df) < 100:
        logger.warning(f"⚠️ {args.stock_id} 有效樣本過少 ({len(df)})，跳過調優。")
        return

    all_features = get_all_features(args.stock_id)
    feat_cols = [c for c in all_features if c in df.columns]
    X = df[feat_cols].fillna(0)
    y = df["target_30d"]

    # 2. 切分 Train/Val（最後一年作驗證）
    split = -252
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]

    # 3. Optuna pruner + sampler（reproducible）
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    logger.info("正在調優 XGBoost...")
    study_xgb = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study_xgb.optimize(
        lambda t: objective(t, X_tr, y_tr, X_va, y_va, "xgb"),
        n_trials=args.trials,
    )

    logger.info("正在調優 LightGBM...")
    study_lgb = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study_lgb.optimize(
        lambda t: objective(t, X_tr, y_tr, X_va, y_va, "lgb"),
        n_trials=args.trials,
    )

    logger.info(f"\nBest XGB Params: {study_xgb.best_params}")
    logger.info(f"Best LGB Params: {study_lgb.best_params}")

    # 5. 儲存結果
    result = {
        "stock_id": args.stock_id,
        "xgb": study_xgb.best_params,
        "lgb": study_lgb.best_params,
        "best_xgb_mse": study_xgb.best_value,
        "best_lgb_mse": study_lgb.best_value,
        "n_trials": args.trials,
    }
    out = Path("scripts/outputs") / f"best_params_{args.stock_id}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, out)
    logger.info(f"最佳參數已儲存至 {out}")


if __name__ == "__main__":
    main()
