import os
import sys
import pandas as pd

sys.path.append(os.getcwd())

from scripts.data_pipeline import build_daily_frame
from scripts.feature_engineering import build_features

def test():
    raw = build_daily_frame(stock_id="2330")
    df = build_features(raw, stock_id="2330", for_inference=True)
    
    print("\n=== 康波長波與第一性特徵診斷 ===")
    cols = ["kwave_score", "gold_oil_ratio", "yield_cycle_pos", "foreign_flow_surprise", "adr_surprise"]
    
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].tail(10))

if __name__ == "__main__":
    test()
