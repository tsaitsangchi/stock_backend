"""
model_metadata.py v3.4 (Trinity Core Final)
================================================================================
修訂歷程：
  v3.4 (2026-05-10): [修正] 強化路徑自癒 Bootstrap，解決 No module named 'core'。
"""
import sys, json, logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path

# ── 終極路徑自癒 Bootstrap ──
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS = None
for p in [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent]:
    if p.name == "scripts" or (p / "scripts").exists():
        _SCRIPTS = p if p.name == "scripts" else (p / "scripts")
        break
if _SCRIPTS:
    if str(_SCRIPTS) not in sys.path: sys.path.insert(0, str(_SCRIPTS))
    if str(_SCRIPTS.parent) not in sys.path: sys.path.insert(0, str(_SCRIPTS.parent))

try: from core.db_utils import write_pipeline_log
except ImportError:
    try: from db_utils import write_pipeline_log
    except ImportError: pass

@dataclass
class ModelMetadata:
    stock_id: str
    model_name: str = "Ensemble_XGB_LGBM"
    oof_da: float = 0.0
    oof_sharpe: float = 0.0
    feature_count: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    def to_json(self): return asdict(self)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = ModelMetadata(stock_id="2330")
    print(f"✅ Metadata initialized: {m}")