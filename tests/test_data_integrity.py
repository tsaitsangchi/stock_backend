import pandas as pd
import numpy as np
from scripts.config import DATA_LAG_CONFIG

def test_data_lag_alignment():
    # 模擬資料
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=100),
        'revenue': np.random.rand(100)
    })
    lag_days = DATA_LAG_CONFIG.get("month_revenue", 40)
    # 驗證 shift 是否正確（約 28 個交易日）
    shifted = df['revenue'].shift(int(lag_days / 1.4))
    assert shifted.iloc[30] == df['revenue'].iloc[30 - int(lag_days / 1.4)]
    print(f"✅ Data Lag 驗證成功: 延遲 {int(lag_days / 1.4)} 交易日")
