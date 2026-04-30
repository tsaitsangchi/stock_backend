import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
for sub in ["fetchers", "pipeline", "training", "monitor", "models", "utils"]:
    p = str(base_dir / sub)
    if p not in sys.path: sys.path.append(p)
if str(base_dir) not in sys.path: sys.path.append(str(base_dir))
