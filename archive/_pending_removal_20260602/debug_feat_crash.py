import sys
from pathlib import Path

# ── 注入路徑 ──────────────────────────────────────
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor"]: 
    sys.path.append(str(base_dir / sub))
sys.path.append(str(base_dir))

import pandas as pd
import logging
from data_pipeline import build_daily_frame
from feature_engineering import build_features

logging.basicConfig(level=logging.INFO)

def debug_2881():
    stock_id = "2881"
    print(f"Testing features for {stock_id}...")
    try:
        raw = build_daily_frame(stock_id, start_date="2026-01-01")
        print(f"Raw shape: {raw.shape}")
        print(f"Raw columns: {[c for c in raw.columns if 'vti' in c.lower()]}")
        df = build_features(raw, stock_id=stock_id)
        print(f"Features shape: {df.shape}")
        print(f"VTI features: {[c for c in df.columns if 'vti' in c.lower()]}")
        
        from config import get_all_features
        all_f = get_all_features(stock_id)
        missing = [f for f in all_f if f not in df.columns]
        print(f"Missing from df: {missing}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_2881()
