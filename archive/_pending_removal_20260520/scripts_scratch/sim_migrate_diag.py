
import sys, json
from pathlib import Path

# 模擬引導
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from core import get_root_dir
    
    def simulate_migrate_diag():
        print("🚀 [模擬實驗] 啟動資產配置診斷...")
        config_path = Path(get_root_dir()) / "config" / "stocks_config.json"
        
        if not config_path.exists():
            print(f"❌ 找不到設定檔 : {config_path}")
            return False
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        core_stocks = config.get("core_stocks", [])
        print(f"✅ 配置檔讀取成功！")
        print(f"📊 目前配置核心資產 : {len(core_stocks)} 檔")
        return True

    if __name__ == "__main__":
        simulate_migrate_diag()

except Exception as e:
    print(f"❌ 診斷失敗: {e}")
