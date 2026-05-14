
import sys
from pathlib import Path

# 模擬引導
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path: sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    import config
    
    def simulate_migrate_diag():
        print("🚀 [模擬實驗] 啟動資產配置診斷...")
        stock_configs = getattr(config, "STOCK_CONFIGS", {})
        
        if not stock_configs:
            print("❌ 找不到 STOCK_CONFIGS 或內容為空。")
            return False
            
        print(f"✅ 設定檔載入成功！")
        print(f"📊 目前配置核心資產 : {len(stock_configs)} 檔")
        return True

    if __name__ == "__main__":
        simulate_migrate_diag()

except Exception as e:
    print(f"❌ 診斷出錯: {e}")
