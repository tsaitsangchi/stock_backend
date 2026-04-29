"""
audit_snapshot.py — 14 支核心個股盲測審計

修改摘要（第三輪審查修復）：
  [P0 2.1] 修正 build_features() 回傳值 tuple unpack 語法錯誤
           df_feat, _ = build_features(df_raw)  → df_feat = build_features(df_raw, stock_id=sid)
  [P0 2.1] 修正目標欄位：從 target_30d（回歸值）改為 target_consensus_binary（二元分類標籤）
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime

sys.path.append("/home/hugo/project/stock_backend/scripts")
from data_pipeline import build_daily_frame
from feature_engineering import build_features

STOCKS = [
    "2330", "2308", "6669", "9958", "1795", "2317", "2454",
    "2881", "2382", "1301", "2002", "2603", "3037", "1513",
]
MODEL_DIR = Path("/home/hugo/project/stock_backend/scripts/outputs/models")

# 目標欄位優先順序（依可用性依序嘗試）
TARGET_COLS = ["target_consensus_binary", "target_binary", "target_15d_binary"]


def _get_target_col(df: pd.DataFrame) -> str | None:
    """回傳 df 中第一個可用的二元目標欄位名稱，若無則回傳 None。"""
    for col in TARGET_COLS:
        if col in df.columns:
            return col
    return None


def audit_snapshot(stock_ids: list[str] | None = None):
    sids = stock_ids or STOCKS
    print(f"🚀 開始 {len(sids)} 支核心個股盲測審計 (測試區間: 2024-01-01 至今)")
    print("=" * 70)
    print(f"{'代號':<6} {'方向正確率 (DA)':<15} {'預期報酬相關性 (IC)':<20} {'狀態'}")
    print("-" * 70)

    results = []
    for sid in sids:
        model_path = MODEL_DIR / f"ensemble_{sid}.pkl"
        if not model_path.exists():
            continue

        try:
            # 1. 載入模型與資料
            model = joblib.load(model_path)
            df_raw = build_daily_frame(sid, start_date="2023-01-01")
            df_feat = build_features(df_raw, stock_id=sid) # 修正：build_features 只回傳一個 DF
            
            # 2. 切分盲測集 (2024+)
            test_df = df_feat.loc["2024-01-01":].copy()
            if test_df.empty:
                continue
                
            # 3. 執行預測與評估
            # 修正：使用正確的二元目標欄位
            target_col = _get_target_col(test_df)
            if target_col is None:
                print(f"{sid:<6} 找不到二元目標欄位，跳過")
                continue

            y_true = test_df[target_col].dropna()
            if len(y_true) < 20:
                continue

            # 3. 對齊索引後推論
            drop_cols = [c for c in test_df.columns if c.startswith("target_")]
            X = test_df.loc[y_true.index].drop(columns=drop_cols, errors="ignore")

            preds = model.predict(X)
            da = (preds == y_true.values).mean()

            # 計算 IC（預測機率與實際報酬的 Spearman 相關）
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
                # IC 對照 target_consensus（連續值），若無則退化為二元標籤
                ret_col = "target_consensus" if "target_consensus" in test_df.columns else target_col
                ic = np.corrcoef(probs, test_df.loc[y_true.index, ret_col])[0, 1]
            else:
                ic = 0.0

            status = "🔥 優異" if da > 0.65 else ("🟢 穩定" if da > 0.55 else "⚠️  待提升")
            print(f"{sid:<6} {da:>12.1%} {ic:>19.3f} {status}")
            results.append(da)

        except Exception as e:
            print(f"{sid:<6} 審計失敗: {e}")

    print("-" * 70)
    if results:
        print(f"平均方向正確率 (DA): {np.mean(results):.1%}  (樣本數: {len(results)} 支)")
    else:
        print("⚠️  尚無可驗證的模型（請先執行 train_evaluate.py）")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="盲測審計腳本")
    parser.add_argument("--stock-id", default=None, help="單一股票代號（留空=全部14支）")
    parser.add_argument("--limit", type=int, default=None, help="限制審計支數")
    args = parser.parse_args()

    target = [args.stock_id] if args.stock_id else None
    if args.limit and target is None:
        target = STOCKS[: args.limit]

    audit_snapshot(target)
