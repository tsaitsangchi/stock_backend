import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

logger = logging.getLogger(__name__)

def lasso_feature_selection(X: pd.DataFrame, y: pd.Series, max_features: int = 30) -> list[str]:
    """
    使用 LASSO (L1 正則化) 進行特徵降維。
    """
    logger.info(f"  [LASSO] 開始特徵降維 (目標數: {max_features})...")
    # 預處理：填充缺失值與標準化
    X_tmp = X.fillna(X.median()).replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_tmp)
    
    # 訓練 LASSO
    lasso = LassoCV(cv=5, max_iter=3000, n_jobs=-1).fit(X_std, y)
    
    # 依據 L1 懲罰後的係數選取
    selector = SelectFromModel(lasso, prefit=True, max_features=max_features)
    mask = selector.get_support()
    selected_cols = X.columns[mask].tolist()
    
    logger.info(f"  [LASSO] 篩選完成：{len(X.columns)} -> {len(selected_cols)} 特徵")
    return selected_cols
