"""
path_setup.py v3.1 (Trinity Core Final)
================================================================================
系統路徑指揮官 — 環境標準化工具
負責自動偵測專案根目錄，並將 core, models, pipeline 等子目錄注入 sys.path。

修訂歷程：
  v3.1 (2026-05-10):
    - [文檔] 補齊極致詳細的執行範例說明。
  v3.0 (2026-05-09):
    - [核心] 實作 ensure_scripts_on_path，解決跨目錄匯入問題。

【執行範例說明】

在任何位於 scripts 子目錄下的腳本中引用（確保 import 不報錯）：
   ------------------------------------------------------------
   import sys
   from pathlib import Path
   
   # 自動修復路徑
   _THIS_DIR = Path(__file__).resolve().parent
   _CORE_DIR = _THIS_DIR.parent / "core"
   if str(_CORE_DIR) not in sys.path:
       sys.path.insert(0, str(_CORE_DIR))
       
   from path_setup import ensure_scripts_on_path
   ensure_scripts_on_path(__file__)
   ------------------------------------------------------------
"""

import sys
import os
from pathlib import Path

def ensure_scripts_on_path(current_file_path):
    """
    確保專案內部的 core, models, pipeline, evaluation, ingestion 等目錄都在 sys.path 中。
    """
    current_path = Path(current_file_path).resolve()
    # 尋找 scripts 目錄
    scripts_root = None
    temp = current_path
    for _ in range(5):
        if temp.name == "scripts":
            scripts_root = temp
            break
        if (temp / "scripts").exists():
            scripts_root = temp / "scripts"
            break
        temp = temp.parent
        
    if not scripts_root:
        return
        
    # 定義需要注入的子目錄
    sub_dirs = ["", "core", "models", "pipeline", "evaluation", "ingestion", "features", "training"]
    for sub in sub_dirs:
        p = (scripts_root / sub) if sub else scripts_root
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            
    # 設定 PYTHONPATH 環境變數以便子進程繼承
    os.environ["PYTHONPATH"] = str(scripts_root) + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")

if __name__ == "__main__":
    ensure_scripts_on_path(__file__)
    print("✅ 系統路徑已標準化。")
    for p in sys.path[:5]:
        print(f"  - {p}")