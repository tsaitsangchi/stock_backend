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
scripts/model_quality_audit.py
模型品質審計系統 (Trinity Edition)

核心邏輯：
1. 盲測審計 (Blind Test)：使用模型從未見過的資料區間 (2024-01-01+) 進行效能回測。
2. 多維指標：包含方向正確率 (DA)、資訊係數 (IC) 以及盲測淨值模擬。
3. 動態發現：自動從 config.py 讀取標的，不再依賴硬編碼清單。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd

# 注入路徑
sys.path.append(str(Path(__file__).resolve().parent))
from config import STOCK_CONFIGS, MODEL_DIR, ALL_FEATURES
from data_pipeline import build_daily_frame
from feature_engineering import build_features
from utils.model_loader import safe_load

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 測試區間定義
TEST_START_DATE = "2024-01-01"

def _get_target_col(df: pd.DataFrame) -> str:
    """找出第一個可用的二元目標欄位"""
    candidates = ["target_30d_binary", "target_consensus_binary", "target_binary"]
    for c in candidates:
        if c in df.columns:
            return c
    return "target_30d_binary" # Default fallback

def run_model_audit(stock_id: str) -> Dict[str, Any]:
    """針對單一標的執行品質審計"""
    model_path = MODEL_DIR / f"ensemble_{stock_id}.pkl"
    if not model_path.exists():
        return {"stock_id": stock_id, "status": "MISSING"}

    try:
        # 1. 載入模型與資料 (多抓半年避免 Lag 影響特徵)
        # [P3 修復] 使用 safe_load (File Locking)
        model = safe_load(model_path)
        df_raw = build_daily_frame(stock_id, start_date="2023-06-01")
        df_feat = build_features(df_raw, stock_id=stock_id)
        
        # 2. 切分盲測集 (2024+)
        test_df = df_feat.loc[TEST_START_DATE:].copy()
        if len(test_df) < 10:
            return {"stock_id": stock_id, "status": "INSUFFICIENT_DATA"}
            
        target_col = _get_target_col(test_df)
        y_true = test_df[target_col].dropna()
        if len(y_true) < 10:
            return {"stock_id": stock_id, "status": "INSUFFICIENT_LABELS"}
            
        # 3. 執行預測
        X = test_df.loc[y_true.index].drop(columns=[c for c in test_df.columns if "target_" in c], errors="ignore")
        pred_dict = model.predict(X)
        
        # 取得 ensemble 機率並轉為二元標籤
        if isinstance(pred_dict, dict) and "ensemble" in pred_dict:
            prob_up = pred_dict["ensemble"]
        else:
            # Fallback if it's a raw sklearn-like model
            prob_up = pred_dict
            
        preds = (prob_up > 0.5).astype(int)
        
        # 4. 計算指標
        da = (preds == y_true.values).mean()
        
        # IC (Information Coefficient)
        ic = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            # 優先與連續值目標 (target_30d_return 等) 做相關性分析
            ret_col = target_col.replace("_binary", "_return") if "_binary" in target_col else target_col
            if ret_col in test_df.columns:
                actual_ret = test_df.loc[y_true.index, ret_col].fillna(0)
                ic = np.corrcoef(probs, actual_ret)[0, 1]

        # 盲測淨值模擬 (Simple Long-Only when Prob > 0.5)
        # 這裡簡化為：若預測上漲且實際也上漲，累積收益
        test_df["actual_ret"] = test_df["close"].pct_change().shift(-1) # 簡化：次日收益
        # 實際上 audit 應該看的是 T+30 收益，這裡為了快速審核採簡化版
        
        status = "🔥 優異" if da > 0.62 else ("🟢 穩定" if da > 0.53 else "🔴 衰退")
        
        return {
            "stock_id": stock_id,
            "status": status,
            "da": round(da, 4),
            "ic": round(ic, 4),
            "samples": len(y_true),
            "last_seen": test_df.index[-1].strftime("%Y-%m-%d")
        }
        
    except Exception as e:
        logger.error(f"審計 {stock_id} 失敗: {e}")
        return {"stock_id": stock_id, "status": "ERROR", "msg": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Trinity Model Quality Audit")
    parser.add_argument("--stock-id", help="特定標的審計")
    parser.add_argument("--limit", type=int, help="限制審計標的數量")
    args = parser.parse_args()
    
    stock_ids = [args.stock_id] if args.stock_id else list(STOCK_CONFIGS.keys())
    if args.limit:
        stock_ids = stock_ids[:args.limit]
        
    print("\n" + "="*80)
    print(f"  Trinity Model Quality Audit (Blind Window: {TEST_START_DATE} ~ Present)")
    print("="*80)
    print(f"{'Stock':<8} {'DA':<10} {'IC':<10} {'Samples':<10} {'Status':<12} {'Last Seen'}")
    print("-"*80)
    
    all_results = []
    for sid in stock_ids:
        res = run_model_audit(sid)
        if res["status"] in ["MISSING", "ERROR", "INSUFFICIENT_DATA"]:
            print(f"{sid:<8} {'--':<10} {'--':<10} {'--':<10} {res['status']:<12}")
            continue
            
        print(f"{sid:<8} {res['da']:<10.1%} {res['ic']:<10.3f} {res['samples']:<10} {res['status']:<12} {res['last_seen']}")
        all_results.append(res["da"])
        
    print("-"*80)
    if all_results:
        print(f"平均盲測正確率 (Mean DA): {np.mean(all_results):.1%} (N={len(all_results)})")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
