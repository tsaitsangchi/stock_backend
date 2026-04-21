# Copyright (c) 2026 Antigravity Quant Research. All rights reserved.
# Proprietary and Confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Written by Antigravity Virtual Colleague for the 80-Stock Predictive Project.
# Description: 統一的模型載入與快取工具，支援 ensemble 模型的自動化加載。
# Version: 2.0.0

"""
model_loader.py — 高效能模型載入器
市場邏輯：確保模型載入的一致性，並透過快取機制減少並行預測時的 I/O 損耗。
符合全域規則 v2.0：強型別、模組化、詳細日誌。
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional
from config import MODEL_DIR

logger = logging.getLogger(__name__)

# 使用快取避免重複載入大型模型
_MODEL_CACHE: dict[str, Any] = {}

def load_ensemble_model(stock_id: str) -> Optional[Any]:
    """
    載入指定個股的 Ensemble 模型 (XGB/LGB/TFT 合位元)。
    
    Args:
        stock_id: 台灣股市代碼 (str)
        
    Returns:
        載入的模型物件，若不存在則回傳 None。
    """
    global _MODEL_CACHE
    
    if stock_id in _MODEL_CACHE:
        logger.debug(f"Using cached model for {stock_id}")
        return _MODEL_CACHE[stock_id]
        
    model_path = Path(MODEL_DIR) / f"ensemble_{stock_id}.pkl"
    
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return None
        
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # 存入快取
        _MODEL_CACHE[stock_id] = model
        logger.info(f"Successfully loaded model for {stock_id} from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model for {stock_id}: {e}")
        return None

def clear_cache() -> None:
    """清除全域模型快取，釋放記憶體。"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("Model cache cleared.")
