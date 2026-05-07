import os
import sys
import json
import logging
from datetime import datetime

# 加入專案路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from scripts.core.db_utils import get_db_conn
from scripts.core.finmind_client import finmind_get

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def enrich_metadata():
    conn = get_db_conn()
    cur = conn.cursor()

    try:
        # 1. 獲取所有核心股票清單
        cur.execute("SELECT stock_id, stock_name, industry FROM stocks WHERE is_core = TRUE")
        core_stocks = cur.fetchall()
        logger.info(f"開始為 {len(core_stocks)} 支核心標的豐富元數據...")

        # 2. 從 FinMind 獲取基礎市場資訊 (市場別)
        raw_info = finmind_get("TaiwanStockInfo", {})
        info_map = {item['stock_id']: item for item in raw_info} if raw_info else {}

        # 3. 定義規則引擎 (MBNRIC 與 美股對標)
        # MBNRIC: Medicine, Bio, Nano, Robot, Info, Cognitive
        # 加入 F (Finance) 作為傳統支柱
        
        rules = {
            "半導體業": {"tag": "I", "proxy": "TSM", "desc": "AI算力與半導體核心"},
            "IC設計": {"tag": "I", "proxy": "NVDA", "desc": "AI邏輯與架構"},
            "電腦及週邊設備業": {"tag": "R", "proxy": "MSFT", "desc": "具身智能與硬體組裝"},
            "生技醫療業": {"tag": "B", "proxy": "IBB", "desc": "第六波健康革命"},
            "金融保險業": {"tag": "F", "proxy": "XLF", "desc": "傳統信用與流動性"},
            "通信網路業": {"tag": "I", "proxy": "QQQ", "desc": "資訊傳輸基礎設施"},
            "電子零組件業": {"tag": "R", "proxy": "AAPL", "desc": "電子物理元件"},
            "塑膠工業": {"tag": "P", "proxy": "XLE", "desc": "傳產能化週期"},
            "鋼鐵工業": {"tag": "S", "proxy": "XLI", "desc": "實體建設基礎"},
            "航運業": {"tag": "L", "proxy": "SEA", "desc": "全球流動性物理鏈"}
        }

        # 特殊標的覆寫 (精確對標)
        manual_overrides = {
            "2330": {"proxy": "TSM", "tag": "I", "etf": ["0050", "0052"]},
            "2317": {"proxy": "AAPL", "tag": "R", "etf": ["0050"]},
            "2454": {"proxy": "AVGO", "tag": "I", "etf": ["0050", "00881"]},
            "2382": {"proxy": "SMCI", "tag": "R", "etf": ["0050", "0056"]},
            "3231": {"proxy": "DELL", "tag": "R", "etf": ["0050", "0056"]},
            "2881": {"proxy": "XLF", "tag": "F", "etf": ["0050"]},
            "2882": {"proxy": "XLF", "tag": "F", "etf": ["0050"]},
            "1301": {"proxy": "DOW", "tag": "P", "etf": ["0050"]}
        }

        updated_count = 0
        for stock_id, name, industry in core_stocks:
            market_type = info_map.get(stock_id, {}).get('market_type', 'TSE')
            
            # 根據行業別匹配規則
            rule = rules.get(industry, {"tag": "O", "proxy": "SPY", "desc": "其他類別"})
            mbnric_tag = rule['tag']
            us_proxy_id = rule['proxy']
            description = rule['desc']
            etf_tags = []

            # 套用手動覆寫
            if stock_id in manual_overrides:
                mo = manual_overrides[stock_id]
                mbnric_tag = mo.get('tag', mbnric_tag)
                us_proxy_id = mo.get('proxy', us_proxy_id)
                etf_tags = mo.get('etf', [])

            # 估算股本規模 (簡化版：0050 成分股為 Large)
            capital_scale = "Large" if "0050" in etf_tags else "Mid"
            if stock_id in ["2330", "2317", "2454"]: capital_scale = "Giant"

            # 執行更新
            cur.execute("""
                UPDATE stocks 
                SET market_type = %s,
                    mbnric_tag = %s,
                    us_proxy_id = %s,
                    etf_tags = %s,
                    capital_scale = %s,
                    description_zh = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE stock_id = %s
            """, (
                market_type, mbnric_tag, us_proxy_id, 
                json.dumps(etf_tags), capital_scale, description, 
                stock_id
            ))
            updated_count += 1

        conn.commit()
        logger.info(f"成功完成 {updated_count} 支核心標的之元數據同步。")

    except Exception as e:
        conn.rollback()
        logger.error(f"同步過程中發生錯誤: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    enrich_metadata()
