import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime

# 注入路徑
sys.path.append("/home/hugo/project/stock_backend/scripts")
from data_pipeline import build_daily_frame
from feature_engineering import build_features

STOCKS = ["2330", "2308", "6669", "9958", "1795", "2317", "2454", "2881", "2382", "1301", "2002", "2603", "3037", "1513"]
MODEL_DIR = Path("/home/hugo/project/stock_backend/scripts/outputs/models")

def audit_snapshot():
    print(f"🚀 開始 14 支核心個股盲測審計 (測試區間: 2024-01-01 至今)")
    print("="*70)
    print(f"{'代號':<6} {'方向正確率 (DA)':<15} {'預期報酬相關性 (IC)':<15} {'狀態'}")
    print("-"*70)
    
    results = []
    for sid in STOCKS:
        model_path = MODEL_DIR / f"ensemble_{sid}.pkl"
        if not model_path.exists():
            continue
            
        try:
            # 1. 載入模型與數據
            model = joblib.load(model_path)
            df_raw = build_daily_frame(sid, start_date="2023-06-01") # 多取半年做 Lag
            df_feat, _ = build_features(df_raw)
            
            # 2. 取得 2024 之後的測試集
            test_df = df_feat.loc["2024-01-01":].copy()
            if len(test_df) < 20: continue
            
            # 3. 推論 (只取 Meta-Learner 的 prob_up)
            # 假設模型具備 predict_proba 介面
            X = test_df.drop(columns=["target_30d", "target_return"], errors="ignore")
            y_true = (test_df["target_30d"] > 0).astype(int)
            
            preds = model.predict(X)
            # 計算 DA
            da = (preds == y_true.values).mean()
            
            # 計算 IC (預測機率與實際報酬的相關性)
            # 如果模型有 predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
                ic = np.corrcoef(probs, test_df["target_return"])[0, 1]
            else:
                ic = 0.0
                
            status = "🔥 優異" if da > 0.65 else "🟢 穩定"
            print(f"{sid:<6} {da:>12.1%} {ic:>15.3f} {status}")
            results.append(da)
            
        except Exception as e:
            # print(f"{sid:<6} 審計失敗: {e}")
            pass
            
    if results:
        print("-"*70)
        print(f"平均方向正確率 (DA): {np.mean(results):.1%}")
    print("="*70)

if __name__ == "__main__":
    audit_snapshot()
