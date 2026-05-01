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
    
    if X.empty:
        logger.warning("  [LASSO] 輸入特徵矩陣為空，跳過篩選。")
        return []

    # 預處理：填充缺失值與標準化
    X_tmp = X.fillna(X.median()).replace([np.inf, -np.inf], 0)
    # 若全為 NaN 則填充為 0
    X_tmp = X_tmp.fillna(0)
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_tmp)
    
    # 訓練 LASSO (稍微調降 alpha_min 增加靈敏度)
    lasso = LassoCV(cv=5, max_iter=5000, n_jobs=-1, selection='random').fit(X_std, y)
    
    # 依據 L1 懲罰後的係數選取
    # [優化] threshold 設為非常小，只要有係數就選，直到達 max_features
    selector = SelectFromModel(lasso, prefit=True, max_features=max_features, threshold=-np.inf)
    mask = selector.get_support()
    selected_cols = X.columns[mask].tolist()
    
    # [Fallback] 如果 LASSO 還是沒選到任何特徵（全部係數為 0）
    if not selected_cols:
        logger.warning("  [LASSO] 嚴格篩選結果為空，改為選取與目標相關性最強的前 10 個特徵。")
        corrs = X_tmp.apply(lambda col: col.corr(y)).abs().sort_values(ascending=False)
        selected_cols = corrs.head(min(10, len(corrs))).index.tolist()
    
    logger.info(f"  [LASSO] 篩選完成：{len(X.columns)} -> {len(selected_cols)} 特徵")
    return selected_cols
