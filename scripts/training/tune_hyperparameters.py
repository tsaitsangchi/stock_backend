import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))
import sys
from pathlib import Path
"""
tune_hyperparameters.py — 使用 Optuna 為特定個股尋找最佳超參數
"""

import argparse
import logging
import sys
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import log_loss, roc_auc_score

from config import STOCK_ID, TRAIN_START_DATE, STOCK_CONFIGS, get_all_features
from data_pipeline import build_daily_frame
from feature_engineering import build_features
from models.ensemble_model import RegimeEnsemble

logger = logging.getLogger(__name__)

def objective(trial, X_tr, y_tr, X_va, y_va, model_type="xgb"):
    if model_type == "xgb":
        params = {
            "n_estimators": 500,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "random_state": 42,
            "n_jobs": -1,
        }
        import xgboost as xgb
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
    else:
        params = {
            "n_estimators": 500,
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50)], verbose=-1)
        preds = model.predict(X_va)
    
    # 使用 MSE 或方向正確率作為目標
    # 這裡簡單使用 MSE
    mse = np.mean((y_va - preds)**2)
    return mse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-id", default="2330")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    logger.info(f"=== 開始超參數調優 (Optuna)：{args.stock_id} ===")
    
    # 1. 載入資料
    raw = build_daily_frame(stock_id=args.stock_id, start_date=TRAIN_START_DATE)
    df = build_features(raw, stock_id=args.stock_id)
    
    all_features = get_all_features(args.stock_id)
    feat_cols = [c for c in all_features if c in df.columns]
    X = df[feat_cols].fillna(0)
    y = df["target_30d"]
    
    # 2. 簡單切分 Train/Val (最後一年作為驗證)
    split = -252
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]
    
    # 3. 調優 XGB
    logger.info("正在調優 XGBoost...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda t: objective(t, X_tr, y_tr, X_va, y_va, "xgb"), n_trials=args.trials)
    
    # 4. 調優 LGB
    logger.info("正在調優 LightGBM...")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lambda t: objective(t, X_tr, y_tr, X_va, y_va, "lgb"), n_trials=args.trials)
    
    logger.info(f"\nBest XGB Params: {study_xgb.best_params}")
    logger.info(f"Best LGB Params: {study_lgb.best_params}")
    
    # 5. 儲存結果
    result = {
        "stock_id": args.stock_id,
        "xgb": study_xgb.best_params,
        "lgb": study_lgb.best_params
    }
    joblib.dump(result, f"scripts/outputs/best_params_{args.stock_id}.pkl")
    logger.info(f"最佳參數已儲存至 scripts/outputs/best_params_{args.stock_id}.pkl")

if __name__ == "__main__":
    main()
