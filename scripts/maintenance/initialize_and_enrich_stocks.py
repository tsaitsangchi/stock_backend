"""
scripts/maintenance/initialize_and_enrich_stocks.py
====================================================
一次性初始化 stocks 核心資料表並填入所有元數據。
包含：DDL 建表 + 150支股票資料 + MBNRIC 規則引擎 + 美股對標

執行方式：
    ./venv/bin/python scripts/maintenance/initialize_and_enrich_stocks.py
"""
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.core.db_utils import get_db_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── DDL ───────────────────────────────────────────────────────
DDL = """
CREATE TABLE IF NOT EXISTS stocks (
    stock_id        TEXT PRIMARY KEY,
    stock_name      TEXT,
    industry        TEXT,
    us_chain        TEXT,
    is_core         BOOLEAN DEFAULT TRUE,
    market_type     TEXT,
    mbnric_tag      TEXT,
    us_proxy_id     TEXT,
    etf_tags        JSONB,
    capital_scale   TEXT,
    description_zh  TEXT,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# ─── MBNRIC 規則引擎（依 config.py 的 industry 標籤）──────────
RULES = {
    "Semiconductor":    ("I", "TSM",  "AI算力與半導體核心"),
    "AI_Hardware":      ("R", "NVDA", "具身智能與AI伺服器硬體"),
    "AI_Server":        ("R", "SMCI", "AI伺服器直接鏈"),
    "Biotech":          ("B", "IBB",  "第六波生技健康革命"),
    "Biotech_Yield":    ("B", "XBI",  "生技收息標的"),
    "Biotech_Drug":     ("B", "PFE",  "新藥開發鏈"),
    "Finance":          ("F", "XLF",  "傳統信用與流動性"),
    "Energy":           ("E", "TAN",  "潔淨能源與存儲"),
    "Materials":        ("P", "XLE",  "傳產與能化週期"),
    "Semi_Materials":   ("I", "AMAT", "半導體材料供應鏈"),
    "Robotics":         ("R", "ABB",  "自動化與機器人"),
    "Telecom":          ("I", "QQQ",  "資訊傳輸基礎設施"),
    "Consumer":         ("C", "VTI",  "消費與終端需求"),
    "Shipping":         ("L", "SEA",  "全球物流物理鏈"),
    "Innovation":       ("I", "QQQ",  "創新板新興標的"),
    "TIB_Innovation":   ("I", "QQQ",  "創新板生技/科技"),
    "TPEx_HighTurnover":("R", "VTI",  "高週轉率標的"),
    "Emerging":         ("O", "VTI",  "其他新興標的"),
    "Retail":           ("C", "VTI",  "零售消費"),
}

# ─── ETF 成分股（手動核心標的）────────────────────────────────
ETF_MAP = {
    "2330": ["0050", "0052", "00881"],
    "2317": ["0050"],
    "2454": ["0050", "00881"],
    "2382": ["0050", "0056"],
    "3231": ["0050", "0056"],
    "2308": ["0050"],
    "2881": ["0050", "0056"],
    "2882": ["0050", "0056"],
    "2886": ["0056"],
    "2891": ["0056"],
    "2884": ["0056"],
    "1301": ["0050"],
    "2303": ["0050"],
    "2002": ["0050"],
}

# ─── 股本規模（依報告第六波核心地位）────────────────────────────
GIANT = {"2330", "2317", "2454", "2308"}
LARGE_CAP = {"2382", "2303", "2881", "2882", "2886", "2891", "2002", "3711", "3037", "3231"}

# ─── 完整 150 支核心標的清單（來源：config.py STOCK_CONFIGS）──
STOCKS = {
    "1101": ("台泥",       "Materials",      ["VMC","MLM"],                  "TSE"),
    "1216": ("統一",       "Consumer",       ["VTI"],                        "TSE"),
    "1301": ("台塑",       "Materials",      ["DOW","LYB","XOM"],            "TSE"),
    "1503": ("士電",       "AI_Hardware",    ["ETN","PWR","GE"],             "TSE"),
    "1504": ("東元",       "Energy",         ["ABB","SIEGY"],                "TSE"),
    "1513": ("中興電",     "Energy",         ["ETN","PWR","GE"],             "TSE"),
    "1514": ("亞力",       "Energy",         ["ETN","PWR","GE"],             "TSE"),
    "1519": ("華城",       "AI_Hardware",    ["ETN","PWR","GE"],             "TSE"),
    "1529": ("樂事綠能",   "Energy",         ["TAN"],                        "OTC"),
    "1560": ("中砂",       "Semiconductor",  ["AMAT","ASML","TSM"],          "TSE"),
    "1565": ("精華",       "Biotech",        ["VTI"],                        "TSE"),
    "1597": ("直得",       "Robotics",       ["ABB","FANUY"],                "OTC"),
    "1707": ("葡萄王",     "Biotech",        ["VTI"],                        "TSE"),
    "1717": ("長興",       "Semi_Materials", ["VTI"],                        "TSE"),
    "1752": ("南光",       "Biotech",        ["IBB","XBI"],                  "OTC"),
    "1760": ("寶齡富錦",   "Biotech",        ["IBB","XBI","PFE"],            "OTC"),
    "1786": ("科妍",       "Biotech",        ["IBB","XBI"],                  "OTC"),
    "1795": ("美時",       "Biotech",        ["PFE","MRK"],                  "OTC"),
    "2002": ("中鋼",       "Materials",      ["X","NUE","STLD"],             "TSE"),
    "2049": ("上銀",       "Robotics",       ["ABB","FANUY","KUKA"],         "TSE"),
    "2249": ("湧盛",       "Emerging",       ["VTI"],                        "OTC"),
    "2301": ("光寶科",     "AI_Hardware",    ["MSFT","GOOGL","NVDA"],        "TSE"),
    "2303": ("聯電",       "Semiconductor",  ["UMC","INTC","TXN"],           "TSE"),
    "2308": ("台達電",     "AI_Hardware",    ["TSLA","NVDA","ENPH","SOXX"],  "TSE"),
    "2317": ("鴻海",       "AI_Hardware",    ["AAPL","HPE","MSFT"],          "TSE"),
    "2324": ("仁寶",       "AI_Hardware",    ["DELL","HPQ"],                 "TSE"),
    "2327": ("國巨",       "Semiconductor",  ["TEL","APH"],                  "TSE"),
    "2330": ("台積電",     "Semiconductor",  ["TSM","NVDA","AAPL","SOXX"],   "TSE"),
    "2337": ("旺宏",       "Semiconductor",  ["MU","WDC"],                   "TSE"),
    "2338": ("光罩",       "Semiconductor",  ["TSM","UMC"],                  "TSE"),
    "2344": ("華邦電",     "Semiconductor",  ["MU","WDC","STX"],             "TSE"),
    "2352": ("佳世達",     "AI_Hardware",    ["PHG","GE"],                   "TSE"),
    "2353": ("宏碁",       "AI_Hardware",    ["INTC","MSFT"],                "TSE"),
    "2356": ("英業達",     "AI_Server",      ["NVDA","AMD"],                 "TSE"),
    "2357": ("華碩",       "AI_Hardware",    ["INTC","NVDA","AMD"],          "TSE"),
    "2359": ("所羅門",     "Robotics",       ["NVDA","FANUY","ABB"],         "OTC"),
    "2360": ("致茂",       "Semiconductor",  ["NVDA","TSM"],                 "TSE"),
    "2368": ("金像電",     "AI_Hardware",    ["NVDA","MSFT"],                "TSE"),
    "2376": ("技嘉",       "AI_Hardware",    ["NVDA","AMD","SMCI"],          "TSE"),
    "2377": ("微星",       "AI_Hardware",    ["NVDA","INTC"],                "TSE"),
    "2379": ("瑞昱",       "Semiconductor",  ["QCOM","AVGO"],                "TSE"),
    "2382": ("廣達",       "AI_Hardware",    ["NVDA","MSFT","GOOGL","AMZN","SMCI"], "TSE"),
    "2383": ("台光電",     "AI_Hardware",    ["NVDA","AMD"],                 "TSE"),
    "2395": ("研華",       "Robotics",       ["HON","ROK","NVDA"],           "TSE"),
    "2401": ("凌陽",       "Semiconductor",  ["VTI"],                        "TSE"),
    "2404": ("漢唐",       "Semiconductor",  ["AMAT","TSM"],                 "OTC"),
    "2408": ("南亞科",     "Semiconductor",  ["MU","WDC","STX"],             "TSE"),
    "2409": ("友達",       "Semiconductor",  ["LPL","SONY"],                 "TSE"),
    "2421": ("建準",       "AI_Hardware",    ["NVDA","VRT"],                 "TSE"),
    "2436": ("偉詮電",     "Semiconductor",  ["VTI"],                        "OTC"),
    "2439": ("美律",       "AI_Hardware",    ["AAPL"],                       "TSE"),
    "2449": ("京元電子",   "Semiconductor",  ["NVDA","AMD","QCOM"],          "TSE"),
    "2454": ("聯發科",     "Semiconductor",  ["QCOM","ARM","SOXX","NVDA"],   "TSE"),
    "2455": ("全新",       "Semiconductor",  ["AVGO","QRVO","SWKS"],         "OTC"),
    "2474": ("可成",       "AI_Hardware",    ["AAPL"],                       "TSE"),
    "2492": ("華新科",     "Semiconductor",  ["TEL","APH"],                  "TSE"),
    "2603": ("長榮",       "Shipping",       ["SEA","ZIM"],                  "TSE"),
    "2609": ("陽明",       "Shipping",       ["SEA","ZIM"],                  "TSE"),
    "2610": ("華航",       "Shipping",       ["DAL","UAL"],                  "TSE"),
    "2615": ("萬海",       "Shipping",       ["SEA","ZIM"],                  "TSE"),
    "2618": ("長榮航",     "Shipping",       ["DAL","LUV"],                  "TSE"),
    "2881": ("富邦金",     "Finance",        ["XLF","KBE","TNX"],            "TSE"),
    "2882": ("國泰金",     "Finance",        ["XLF","KBE","VTI"],            "TSE"),
    "2884": ("玉山金",     "Finance",        ["XLF","VTI"],                  "TSE"),
    "2886": ("兆豐金",     "Finance",        ["XLF","KBE","TLT"],            "TSE"),
    "2891": ("中信金",     "Finance",        ["XLF","VTI"],                  "TSE"),
    "2892": ("第一金",     "Finance",        ["XLF","TLT"],                  "TSE"),
    "2912": ("統一超",     "Consumer",       ["VTI"],                        "TSE"),
    "3006": ("晶豪科",     "Semiconductor",  ["MU","WDC"],                   "OTC"),
    "3008": ("大立光",     "Semiconductor",  ["AAPL","LITE"],                "TSE"),
    "3013": ("晟銘電",     "AI_Hardware",    ["SMCI","NVDA"],                "OTC"),
    "3016": ("嘉晶",       "Semiconductor",  ["WOLF","ON"],                  "OTC"),
    "3017": ("奇鋐",       "AI_Hardware",    ["NVDA","VRT","AMD"],           "TSE"),
    "3019": ("亞光",       "Robotics",       ["AAPL","TSLA"],                "OTC"),
    "3030": ("德律",       "Semiconductor",  ["AMAT"],                       "TSE"),
    "3034": ("聯詠",       "Semiconductor",  ["AAPL","VTI"],                 "TSE"),
    "3035": ("智原",       "Semiconductor",  ["ARM","QCOM","SOXX"],          "OTC"),
    "3037": ("欣興",       "Semiconductor",  ["NVDA","AMD","INTC","SOXX"],   "TSE"),
    "3044": ("健鼎",       "AI_Hardware",    ["NVDA","MSFT"],                "TSE"),
    "3081": ("聯亞",       "Telecom",        ["LITE","COHR"],                "OTC"),
    "3088": ("艾訊",       "Robotics",       ["HON","ROK"],                  "OTC"),
    "3105": ("穩懋",       "Semiconductor",  ["QRVO","SWKS","AVGO"],         "OTC"),
    "3131": ("弘塑",       "Semiconductor",  ["AMAT","ASML"],                "OTC"),
    "3141": ("晶宏",       "Semiconductor",  ["AAPL"],                       "OTC"),
    "3227": ("原相",       "Semiconductor",  ["SONY","VTI"],                 "OTC"),
    "3231": ("緯創",       "AI_Hardware",    ["NVDA","SMCI","MSFT","DELL"],  "TSE"),
    "3264": ("欣銓",       "Semiconductor",  ["ASX","TXN"],                  "OTC"),
    "3293": ("鈊象",       "Consumer",       ["VTI"],                        "OTC"),
    "3324": ("雙鴻",       "AI_Hardware",    ["NVDA","VRT","AMD"],           "OTC"),
    "3376": ("新日興",     "AI_Hardware",    ["AAPL"],                       "OTC"),
    "3406": ("玉晶光",     "Robotics",       ["AAPL","META"],                "OTC"),
    "3443": ("創意",       "Semiconductor",  ["NVDA","AMD","SOXX"],          "OTC"),
    "3481": ("群創",       "Semiconductor",  ["LPL","SONY"],                 "TSE"),
    "3504": ("揚明光",     "Robotics",       ["AAPL","META"],                "OTC"),
    "3515": ("華擎",       "AI_Hardware",    ["AMD","NVDA","INTC"],          "OTC"),
    "3529": ("力旺",       "Semiconductor",  ["ARM","NVDA","SOXX"],          "OTC"),
    "3532": ("台勝科",     "Semiconductor",  ["SUMCO"],                      "OTC"),
    "3533": ("嘉澤",       "AI_Hardware",    ["INTC","AMD","NVDA"],          "OTC"),
    "3545": ("敦泰",       "Semiconductor",  ["VTI","SOXX"],                 "OTC"),
    "3576": ("聯合再生",   "Energy",         ["TAN","FSLR"],                 "TSE"),
    "3583": ("辛耘",       "Semiconductor",  ["AMAT","ASML"],                "OTC"),
    "3633": ("云光",       "Emerging",       ["VTI"],                        "OTC"),
    "3653": ("健策",       "AI_Hardware",    ["NVDA","AMD"],                 "OTC"),
    "3661": ("世芯-KY",    "Semiconductor",  ["NVDA","AMZN","MSFT"],         "OTC"),
    "3680": ("家登",       "Semiconductor",  ["ASML","TSM"],                 "OTC"),
    "3693": ("營邦",       "AI_Hardware",    ["SMCI","DELL"],                "OTC"),
    "3706": ("神達",       "AI_Hardware",    ["DELL","HPE"],                 "OTC"),
    "3711": ("日月光投控", "Semiconductor",  ["ASX","INTC","AMAT"],          "TSE"),
    "3712": ("永崴投控",   "Energy",         ["TAN","FSLR"],                 "OTC"),
    "4107": ("邦特",       "Biotech_Yield",  ["VTI"],                        "OTC"),
    "4126": ("太醫",       "Biotech_Yield",  ["VTI"],                        "OTC"),
    "4137": ("麗豐-KY",    "Biotech",        ["VTI"],                        "OTC"),
    "4147": ("中裕",       "Biotech",        ["IBB","XBI"],                  "OTC"),
    "4722": ("國精化",     "Semi_Materials", ["VTI"],                        "OTC"),
    "4743": ("合一",       "Biotech",        ["IBB","XBI"],                  "OTC"),
    "4749": ("新應材",     "Semi_Materials", ["AMAT","ASML"],                "OTC"),
    "4771": ("望隼",       "Biotech_Yield",  ["VTI"],                        "OTC"),
    "4772": ("台特化",     "Semi_Materials", ["AMAT","ASML"],                "OTC"),
    "5234": ("達興材料",   "Semi_Materials", ["AMAT"],                       "OTC"),
    "5269": ("祥碩",       "Semiconductor",  ["AMD","INTC","AAPL"],          "OTC"),
    "5274": ("信驊",       "Semiconductor",  ["NVDA","MSFT","AMZN"],         "OTC"),
    "5478": ("智冠",       "Consumer",       ["VTI"],                        "OTC"),
    "5871": ("中租-KY",    "Finance",        ["XLF"],                        "TSE"),
    "5880": ("合庫金",     "Finance",        ["XLF","TLT"],                  "TSE"),
    "6125": ("廣運",       "AI_Hardware",    ["NVDA","VRT"],                 "OTC"),
    "6138": ("茂達",       "Semiconductor",  ["TI","AVGO"],                  "OTC"),
    "6180": ("橘子",       "Consumer",       ["VTI"],                        "OTC"),
    "6182": ("合晶",       "Semiconductor",  ["SUMCO","TSM"],                "OTC"),
    "6187": ("萬潤",       "Semiconductor",  ["AMAT","ASML","TSM"],          "OTC"),
    "6223": ("旺矽",       "Semiconductor",  ["NVDA","AMAT"],                "OTC"),
    "6244": ("茂迪",       "Energy",         ["FSLR","TAN"],                 "TSE"),
    "6274": ("台燿",       "AI_Hardware",    ["NVDA","AMD"],                 "OTC"),
    "6285": ("啟碁",       "Telecom",        ["CSCO","VTI"],                 "OTC"),
    "6446": ("藥華藥",     "Biotech",        ["IBB","XBI","MRK"],            "OTC"),
    "6472": ("保瑞",       "Biotech",        ["PFE","MRK","AZN"],            "OTC"),
    "6488": ("環球晶",     "Semiconductor",  ["WOLF","ON","SUMCO"],          "TSE"),
    "6491": ("晶碩",       "Biotech_Yield",  ["VTI"],                        "OTC"),
    "6515": ("穎崴",       "Semiconductor",  ["NVDA","AMD"],                 "OTC"),
    "6523": ("達爾膚",     "Biotech_Yield",  ["VTI"],                        "OTC"),
    "6547": ("高端疫苗",   "Biotech",        ["PFE","MRNA"],                 "OTC"),
    "6669": ("緯穎",       "AI_Hardware",    ["MSFT","META","AMZN","NVDA"],  "TSE"),
    "6683": ("雍智科技",   "Semiconductor",  ["NVDA","AMD"],                 "OTC"),
    "6782": ("視陽",       "Biotech_Yield",  ["VTI"],                        "OTC"),
    "6805": ("富世達",     "AI_Hardware",    ["AAPL"],                       "OTC"),
    "6826": ("和淞",       "Semiconductor",  ["AMAT","ASML"],                "OTC"),
    "6919": ("康霈",       "Biotech",        ["IBB","XBI"],                  "OTC"),
    "7403": ("紐因科技",   "Innovation",     ["VTI"],                        "Innovation"),
    "7408": ("易得雲端",   "Innovation",     ["VTI"],                        "Innovation"),
    "7409": ("美格能",     "Innovation",     ["VTI"],                        "Innovation"),
    "7415": ("元澄半導體", "Innovation",     ["TSM","NVDA"],                 "Innovation"),
    "7695": ("宏潤生技",   "TIB_Innovation", ["IBB"],                        "Innovation"),
    "7769": ("鴻勁精密",   "Innovation",     ["AMAT","ASML"],                "Innovation"),
    "7799": ("禾榮科",     "Biotech_Drug",   ["IBB"],                        "Innovation"),
    "8046": ("南電",       "Semiconductor",  ["NVDA","AMD","INTC"],          "TSE"),
    "8098": ("慶康科技",   "Semiconductor",  ["AMAT","TSM"],                 "OTC"),
    "8210": ("勤誠",       "AI_Hardware",    ["NVDA","MSFT","SMCI"],         "OTC"),
    "8299": ("群聯",       "Semiconductor",  ["NVDA","MU"],                  "OTC"),
    "9904": ("寶成",       "Consumer",       ["NKE","VFC"],                  "TSE"),
    "2207": ("和泰車",     "Consumer",       ["TM","VTI"],                   "TSE"),
    "2412": ("中華電",     "Telecom",        ["VZ","T"],                     "TSE"),
}


def main():
    conn = get_db_conn()
    cur = conn.cursor()

    try:
        # 1. 建立資料表
        logger.info("建立 stocks 資料表...")
        cur.execute(DDL)
        conn.commit()
        logger.info("stocks 資料表已就緒。")

        # 2. 插入所有核心標的
        logger.info(f"開始寫入 {len(STOCKS)} 支核心標的...")
        inserted = 0

        for stock_id, (name, industry, us_tickers, default_market) in STOCKS.items():
            tag, proxy, desc = RULES.get(industry, ("O", "SPY", "其他類別"))
            us_chain = ",".join(us_tickers)
            etf_tags = ETF_MAP.get(stock_id, [])
            
            if stock_id in GIANT:
                scale = "Giant"
            elif stock_id in LARGE_CAP or "0050" in etf_tags:
                scale = "Large"
            else:
                scale = "Mid"

            cur.execute("""
                INSERT INTO stocks (
                    stock_id, stock_name, industry, us_chain, is_core,
                    market_type, mbnric_tag, us_proxy_id, etf_tags,
                    capital_scale, description_zh
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (stock_id) DO UPDATE SET
                    stock_name    = EXCLUDED.stock_name,
                    industry      = EXCLUDED.industry,
                    us_chain      = EXCLUDED.us_chain,
                    is_core       = EXCLUDED.is_core,
                    market_type   = EXCLUDED.market_type,
                    mbnric_tag    = EXCLUDED.mbnric_tag,
                    us_proxy_id   = EXCLUDED.us_proxy_id,
                    etf_tags      = EXCLUDED.etf_tags,
                    capital_scale = EXCLUDED.capital_scale,
                    description_zh= EXCLUDED.description_zh,
                    updated_at    = CURRENT_TIMESTAMP
            """, (
                stock_id, name, industry, us_chain, True,
                default_market, tag, proxy,
                json.dumps(etf_tags), scale, desc
            ))
            inserted += 1

        conn.commit()
        logger.info(f"✅ 成功寫入 {inserted} 支核心標的。")

        # 3. 驗證結果
        cur.execute("SELECT mbnric_tag, COUNT(*) FROM stocks GROUP BY mbnric_tag ORDER BY COUNT(*) DESC")
        rows = cur.fetchall()
        logger.info("=== MBNRIC 分布 ===")
        for tag, cnt in rows:
            logger.info(f"  {tag:3s}: {cnt} 支")

        cur.execute("SELECT capital_scale, COUNT(*) FROM stocks GROUP BY capital_scale ORDER BY COUNT(*) DESC")
        rows = cur.fetchall()
        logger.info("=== 股本規模分布 ===")
        for scale, cnt in rows:
            logger.info(f"  {scale:6s}: {cnt} 支")

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 執行失敗: {e}", exc_info=True)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
