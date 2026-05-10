"""
path_setup.py v3.3 (Trinity Core Final)
================================================================================
系統路徑配置器 — 終極自癒版
負責自動將專案中的 scripts 目錄及其父目錄加入 sys.path，解決跨目錄執行時的 ModuleNotFoundError。

修訂歷程：
  v3.3 (2026-05-10): [修正] 強化對 scripts/ 目錄的偵測邏輯，支援從根目錄或子目錄啟動。
"""
import sys
from pathlib import Path

def ensure_scripts_on_path(current_file: str):
    """
    自癒邏輯：動態將 'scripts' 目錄加入 sys.path。
    """
    curr_p = Path(current_file).resolve()
    
    # 定義可能的 scripts 目錄路徑
    possible_scripts = []
    
    # 往上尋找名為 scripts 的目錄
    temp_p = curr_p
    for _ in range(4):
        if temp_p.name == "scripts":
            possible_scripts.append(temp_p)
        if (temp_p / "scripts").exists():
            possible_scripts.append(temp_p / "scripts")
        temp_p = temp_p.parent
        
    for s_dir in possible_scripts:
        if s_dir.exists():
            # 1. 加入 scripts/ 目錄本身 (支援 from core.xxx)
            s_path = str(s_dir)
            if s_path not in sys.path:
                sys.path.insert(0, s_path)
            
            # 2. 加入 scripts/ 的父目錄 (支援 from scripts.core.xxx)
            root_path = str(s_dir.parent)
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
            return True
    return False

if __name__ == "__main__":
    if ensure_scripts_on_path(__file__):
        print("✅ Path healing successful.")
        for p in sys.path[:3]: print(f" - {p}")
    else:
        print("❌ Could not find scripts directory.")