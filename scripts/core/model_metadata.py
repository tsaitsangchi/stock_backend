"""
model_metadata.py v3.1 (Trinity Core Final)
================================================================================
模型元數據管理器 — 混合日誌整合版
負責定義 ModelMetadata 結構，並提供模型指標與超參數的持久化功能。

修訂歷程：
  v3.1 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v3.0 (2026-05-09):
    - [核心] 重構對接 evaluation_log 與 extra_metadata (JSONB)。

【執行範例說明】

1. 程式內引用與儲存模型指標：
   ------------------------------------------------------------
   from core.model_metadata import ModelMetadata, save_model_registry
   meta = ModelMetadata(stock_id="2330", oof_da=0.62, oof_sharpe=2.1)
   save_model_registry(meta)
   ------------------------------------------------------------

2. 業務紀錄查閱 (查看模型最新性能)：
   SELECT stock_id, oof_da, oof_sharpe, extra_metadata->>'feature_count' as features
   FROM evaluation_log 
   ORDER BY created_at DESC LIMIT 10;
"""

import sys
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR if _THIS_DIR.name == "scripts" else _THIS_DIR.parent
for _sub in ("", "core"):
    _p = (_SCRIPTS_DIR / _sub) if _sub else _SCRIPTS_DIR
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.db_utils import write_evaluation_log, write_pipeline_log
except ImportError:
    pass

@dataclass
class ModelMetadata:
    stock_id: str
    model_name: str = "Ensemble_XGB_LGBM"
    oof_da: float = 0.0
    oof_sharpe: float = 0.0
    feature_count: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self):
        return asdict(self)

def save_model_registry(meta: ModelMetadata):
    try:
        write_evaluation_log(
            stock_id=meta.stock_id,
            model_name=meta.model_name,
            sharpe=meta.oof_sharpe,
            mdd=0.0,
            ret=0.0,
            win_rate=meta.oof_da,
            start=None,
            end=None,
            extra=meta.to_json()
        )
        write_pipeline_log("model_registry", meta.stock_id, "success", "training")
        return True
    except Exception as e:
        print(f"[ERROR] Registry 失敗: {e}")
        return False