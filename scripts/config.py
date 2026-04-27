"""
config.py — 全域設定：路徑、超參數、特徵分組
資料來源：PostgreSQL 17（連線設定在 data_pipeline.py）
"""
from pathlib import Path

# ─────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR  = OUTPUT_DIR / "models"
LOG_DIR    = OUTPUT_DIR / "logs"

for _d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 資料來源：PostgreSQL 17
# 對應資料表：
#   stock_price, stock_per, financial_statements, balance_sheet,
#   dividend, institutional_investors_buy_sell,
#   margin_purchase_short_sale, shareholding,
#   interest_rate, exchange_rate, total_return_index, month_revenue
# 連線設定在 data_pipeline.DB_CONFIG
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 核心參數
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 風險管理參數 (Risk Management)
# ─────────────────────────────────────────────
RISK_CONFIG = {
    # 槓鈴比例限制
    "target_core_ratio":    0.80,  # 核心防禦端目標比例
    "target_agg_ratio":     0.20,  # 進取凸性端目標比例
    "rebalance_threshold":  0.05,  # 偏離目標 5% 時觸發再平衡警告
    
    # 單一標的集中度限制 (Concentration Limit)
    "max_pos_core":         0.15,  # 單一核心標的上限 (如 2330)
    "max_pos_agg":          0.05,  # 單一進取標的上限 (防止單一 AI 股暴雷)
    
    # 流動性篩選 (Liquidity Screening)
    "min_avg_vol_twd":      50_000_000, # 最小日均成交量 5000 萬 TWD
    "max_vol_participation": 0.10,       # 交易量參與率上限 (單筆不超過日均量 10%)
    
    # 績效監控維度
    "target_payoff_ratio":  2.0,   # 目標損益比 (Avg Gain / Avg Loss)
    "min_expected_value":   0.01,  # 最小期望值門檻 (每筆交易預期獲利 > 1%)
}

# ─────────────────────────────────────────────
# 資料可用性時間表 (Data Availability Calendar)
# ─────────────────────────────────────────────
# 台股資料通常有嚴著的公告延遲，若直接使用原始日期會產生「未來資訊洩漏」。
# 此配置定義了資料從「統計區間末日」到「實際可被投資者讀取」的平移天數。
DATA_LAG_CONFIG = {
    # 月營收：次月 10 日前公告。
    # 若 FinMind 日期為月初 (如 2024-10-01)，平移 40 天 ≈ 次月 10 號。
    "month_revenue":        40,  
    
    # 季報 (Q1, Q2, Q3)：季末後 45 天內公告。
    "financial_statements": 45, 
    
    # 年報 (Q4)：次年 3 月底前公告。
    # 12/31 + 90 天 ≈ 次年 3/31。
    "annual_report":        90,
    
    # 籌碼資料 (法人/融資)：當日收盤後公告，次日開盤前可用。
    # 平移 1 天確保回測時使用的是「昨天已公告」的資料。
    "institutional_chip":   1,
}

# ─────────────────────────────────────────────
# 訓練策略配置 (Training Strategy)
# ─────────────────────────────────────────────
# 解決單一標的樣本不足 (n=2000) vs 特徵過多 (p=150) 的過擬合矛盾。
TRAINING_STRATEGY = {
    "use_global_backbone": True,    # 是否採用跨標的混合訓練 (Pooling)
    "finetune_local":      True,    # 訓練完大模型後，是否針對特定標的進行微調
    "feature_selection":   "robust_ic", # 特徵篩選策略: none / robust_ic (IC IR >= 0.5)
}

# 定義標的池 (Stock Pools)：相似產業或屬性的標的共同訓練大模型
SECTOR_POOLS = {
    "Semiconductor": ["2330", "2303", "2454", "3661", "3037", "3711"],
    "AI_Hardware":   ["2382", "2317", "6669", "2357", "3231", "2417"],
    "Finance":       ["2881", "2882", "2886", "2891", "5880"],
}

STOCK_ID       = "2330"    # 預設股票代號

# ─────────────────────────────────────────────
# 交易成本與市場衝擊 (Friction & Costs)
# ─────────────────────────────────────────────
# 台股實際交易成本估計，用於淨報酬計算與評估。
FRICTION_CONFIG = {
    "commission": 0.001425,       # 雙邊手續費 (買進與賣出各一次)
    "securities_tax": 0.003,      # 證交稅 (僅賣出時收取)
    "slippage_large_cap": 0.001,  # 大型股滑點估計 (如 2330, 市值 > 5000 億)
    "slippage_small_cap": 0.005,  # 中小型股滑點估計 (流動性較較低)
}

# 輔助：判定大型股標的
LARGE_CAP_TICKERS = ["2330", "2317", "2454", "2308", "2881", "2882", "2303"]
DEFAULT_STOCK_ID = "2330"

def calculate_net_return(gross_return: float, ticker: str) -> float:
    """
    計算扣除交易成本與滑點後的淨報酬。
    台股雙邊成本：手續費*2 + 證交稅 + 滑點
    """
    is_large_cap = ticker in LARGE_CAP_TICKERS
    slippage = FRICTION_CONFIG["slippage_large_cap"] if is_large_cap else FRICTION_CONFIG["slippage_small_cap"]
    
    # 雙邊總成本 (Round Trip) = commission*2 + tax + slippage (買入滑點 + 賣出滑點)
    total_cost = (FRICTION_CONFIG["commission"] * 2 + 
                  FRICTION_CONFIG["securities_tax"] + 
                  slippage * 2)
    return gross_return - total_cost

# ─────────────────────────────────────────────
# 個股客製化配置 (Multi-Stock Framework)
# ─────────────────────────────────────────────
STOCK_CONFIGS = {
    "2330": {
        "name": "台積電",
        "industry": "Semiconductor",
        "us_chain_tickers": ["TSM", "NVDA", "AAPL", "SOXX"],
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": True,
    },
    "2317": {
        "name": "鴻海",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["AAPL", "HPE", "MSFT"], # 加入微軟（AI 伺服器）
        "vol_low": 0.25,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "2454": {
        "name": "聯發科",
        "industry": "Semiconductor",
        "us_chain_tickers": ["QCOM", "ARM", "SOXX", "NVDA"],
        "vol_low": 0.30,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "2881": {
        "name": "富邦金",
        "industry": "Finance",
        "us_chain_tickers": ["XLF", "KBE", "TNX", "VTI"], # XLF/KBE 是金融 ETF, TNX 是 10Y 殖利率, VTI 是全市場
        "vol_low": 0.12,   # 金融股波動更低
        "vol_high": 0.25,
        "use_adr_premium": False,
    },
    "2382": {
        "name": "廣達",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["NVDA", "MSFT", "GOOGL", "AMZN", "SMCI", "SOXX"], # AI 供應鏈
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "1301": {
        "name": "台塑",
        "industry": "Materials",
        "us_chain_tickers": ["DOW", "LYB", "XOM", "CVX"], # 塑膠與石油巨頭
        "vol_low": 0.15,
        "vol_high": 0.35,
        "use_adr_premium": False,
    },
    "2002": {
        "name": "中鋼",
        "industry": "Materials",
        "us_chain_tickers": ["X", "NUE", "STLD", "MT"], # 鋼鐵巨頭 (X=US Steel, NUE=Nucor)
        "vol_low": 0.10,
        "vol_high": 0.30,
        "use_adr_premium": False,
    },
    "2603": {
        "name": "長榮",
        "industry": "Shipping",
        "us_chain_tickers": ["ZIM", "MATX", "SEA", "BDRY"], # ZIM 是全球同業, SEA 是航運 ETF, BDRY 是散裝/運價指標
        "vol_low": 0.30,
        "vol_high": 0.60,
        "use_adr_premium": False,
    },
    "3037": {
        "name": "欣興",
        "industry": "Semiconductor",
        "us_chain_tickers": ["NVDA", "AMD", "INTC", "SOXX"], # ABF 載板連動 (AI/PC 晶片)
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "3324": {
        "name": "雙鴻",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["NVDA", "VRT", "AMD"], # VRT 是全球散熱龍頭 Vertiv
        "vol_low": 0.35,
        "vol_high": 0.70,
        "use_adr_premium": False,
    },
    "1513": {
        "name": "中興電",
        "industry": "Energy",
        "us_chain_tickers": ["ETN", "PWR", "GE"], # ETN (Eaton), PWR (Quanta Services) 是重電龍頭
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "3008": {
        "name": "大立光",
        "industry": "Semiconductor",
        "us_chain_tickers": ["AAPL", "LITE"], # 蘋果供應鏈與光學通訊
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": False,
    },
    "2308": {
        "name": "台達電",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["TSLA", "NVDA", "ENPH", "SOXX"],
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": False,
    },
    "6669": {
        "name": "緯穎",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["MSFT", "META", "AMZN", "NVDA"],
        "vol_low": 0.30,
        "vol_high": 0.60,
        "use_adr_premium": False,
    },
    "9958": {
        "name": "世紀鋼",
        "industry": "Energy",
        "us_chain_tickers": ["X", "NUE", "STLD", "MT"], # 使用鋼鐵龍頭作為代理
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "1795": {
        "name": "美時",
        "industry": "BioTech",
        "us_chain_tickers": ["XLV", "PFE", "MRK", "TEVA"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "3661": {
        "name": "世芯-KY",
        "industry": "Semiconductor",
        "us_chain_tickers": ["NVDA", "AVGO", "AMD", "SOXX"],
        "vol_low": 0.35,
        "vol_high": 0.60,
        "use_adr_premium": False,
    },
    "3231": {
        "name": "緯創",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["DELL", "HPE", "SMCI", "NVDA"],
        "vol_low": 0.30,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "2303": {
        "name": "聯電",
        "industry": "Semiconductor",
        "us_chain_tickers": ["INTC", "TSM", "SOXX"],
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": True,
    },
    "2609": {
        "name": "陽明",
        "industry": "Shipping",
        "us_chain_tickers": ["ZIM", "MATX", "SEA"],
        "vol_low": 0.30,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "2882": {
        "name": "國泰金",
        "industry": "Finance",
        "us_chain_tickers": ["XLF", "JPM", "GS"],
        "vol_low": 0.15,
        "vol_high": 0.30,
        "use_adr_premium": False,
    },
    "1519": {
        "name": "華城",
        "industry": "Energy",
        "us_chain_tickers": ["ETN", "GE", "ABB", "PWR"],
        "vol_low": 0.35,
        "vol_high": 0.65,
        "use_adr_premium": False,
    },
    "3019": {
        "name": "亞光",
        "industry": "Domestic",
        "us_chain_tickers": ["TSLA", "MBLY", "AAPL"],
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "2618": {
        "name": "長榮航",
        "industry": "Shipping",
        "us_chain_tickers": ["DAL", "UAL", "AAL", "BA"],
        "vol_low": 0.25,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "1216": {
        "name": "統一",
        "industry": "Domestic",
        "us_chain_tickers": ["KO", "PEP", "COST", "WMT"],
        "vol_low": 0.10,
        "vol_high": 0.25,
        "use_adr_premium": False,
    },
    "6505": {
        "name": "台塑化",
        "industry": "Materials",
        "us_chain_tickers": ["XOM", "CVX", "XLE"],
        "vol_low": 0.15,
        "vol_high": 0.35,
        "use_adr_premium": False,
    },
    "2542": {
        "name": "興富發",
        "industry": "Domestic",
        "us_chain_tickers": ["LEN", "DHI", "ITB"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "1101": {
        "name": "台泥",
        "industry": "Materials",
        "us_chain_tickers": ["MLM", "VMC", "STLD"],
        "vol_low": 0.15,
        "vol_high": 0.35,
        "use_adr_premium": False,
    },
    "2412": {
        "name": "中華電",
        "industry": "Domestic",
        "us_chain_tickers": ["T", "VZ", "TMUS"],
        "vol_low": 0.05,
        "vol_high": 0.15,
        "use_adr_premium": False,
    },
    "2201": {
        "name": "裕隆",
        "industry": "Domestic",
        "us_chain_tickers": ["TSLA", "TM", "GM", "F"],
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "1476": {
        "name": "儒鴻",
        "industry": "Domestic",
        "us_chain_tickers": ["NKE", "LULU", "UAA"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "8454": {
        "name": "富邦媒",
        "industry": "Domestic",
        "us_chain_tickers": ["AMZN", "BABA", "MELI"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "3443": {
        "name": "創意",
        "industry": "Semiconductor",
        "us_chain_tickers": ["NVDA", "AVGO", "ARM", "SOXX"],
        "vol_low": 0.35,
        "vol_high": 0.65,
        "use_adr_premium": False,
    },
    "3711": {
        "name": "日月光投控",
        "industry": "Semiconductor",
        "us_chain_tickers": ["AMAT", "LRCX", "KLAC", "SOXX"],
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": True,
    },
    "2379": {
        "name": "瑞昱",
        "industry": "Semiconductor",
        "us_chain_tickers": ["QRVO", "SWKS", "MRVL"],
        "vol_low": 0.25,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "2885": {
        "name": "元大金",
        "industry": "Finance",
        "us_chain_tickers": ["GS", "MS", "SCHW"],
        "vol_low": 0.15,
        "vol_high": 0.30,
        "use_adr_premium": False,
    },
    "2634": {
        "name": "漢翔",
        "industry": "Domestic",
        "us_chain_tickers": ["BA", "LMT", "RTX"],
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": False,
    },
    "2395": {
        "name": "研華",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["MSFT", "INTC", "CSCO"],
        "vol_low": 0.15,
        "vol_high": 0.35,
        "use_adr_premium": False,
    },
    "6180": {
        "name": "橘子",
        "industry": "Domestic",
        "us_chain_tickers": ["MSFT", "ATVI", "SONY"],
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "2731": {
        "name": "雄獅",
        "industry": "Domestic",
        "us_chain_tickers": ["EXPE", "BKNG", "TRIP"],
        "vol_low": 0.30,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "8341": {
        "name": "日友",
        "industry": "Domestic",
        "us_chain_tickers": ["WM", "RSG", "WCN"],
        "vol_low": 0.10,
        "vol_high": 0.25,
        "use_adr_premium": False,
    },
    "5871": {
        "name": "中租-KY",
        "industry": "Finance",
        "us_chain_tickers": ["ALLY", "AXP", "DFS"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "9921": {
        "name": "巨大",
        "industry": "Domestic",
        "us_chain_tickers": ["PELO", "FOXF", "COLM"],
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": False,
    },
    "5483": {
        "name": "中美晶",
        "industry": "Semiconductor",
        "us_chain_tickers": ["WOLF", "ON", "FSLR"],
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "9917": {
        "name": "中保科",
        "industry": "Domestic",
        "us_chain_tickers": ["ADT", "ALLE", "VRT"],
        "vol_low": 0.05,
        "vol_high": 0.15,
        "use_adr_premium": False,
    },
    "2105": {
        "name": "正新",
        "industry": "Materials",
        "us_chain_tickers": ["GT", "CTB", "TSLA"],
        "vol_low": 0.15,
        "vol_high": 0.35,
        "use_adr_premium": False,
    },
    "1802": {
        "name": "台玻",
        "industry": "Materials",
        "us_chain_tickers": ["AA", "VMC", "MLM"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "1210": {
        "name": "大成",
        "industry": "Domestic",
        "us_chain_tickers": ["ADM", "BG", "TSN"],
        "vol_low": 0.10,
        "vol_high": 0.20,
        "use_adr_premium": False,
    },
    "2891": {
        "name": "中信金",
        "industry": "Finance",
        "us_chain_tickers": ["JPM", "C", "WFC"],
        "vol_low": 0.10,
        "vol_high": 0.25,
        "use_adr_premium": False,
    },
    "2912": {
        "name": "統一超",
        "industry": "Domestic",
        "us_chain_tickers": ["WMT", "TGT", "COST"],
        "vol_low": 0.05,
        "vol_high": 0.15,
        "use_adr_premium": False,
    },
    "4977": {
        "name": "眾達-KY",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["LITE", "FN", "AAOI"],
        "vol_low": 0.30,
        "vol_high": 0.60,
        "use_adr_premium": False,
    },
    "3034": {
        "name": "聯詠",
        "industry": "Semiconductor",
        "us_chain_tickers": ["QCOM", "MRVL", "SOXX"],
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": False,
    },
    "2357": {
        "name": "華碩",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["MSFT", "INTC", "AMD"],
        "vol_low": 0.20,
        "vol_high": 0.40,
        "use_adr_premium": False,
    },
    "5880": {
        "name": "合庫金",
        "industry": "Finance",
        "us_chain_tickers": ["JPM", "BAC", "WFC"],
        "vol_low": 0.10,
        "vol_high": 0.20,
        "use_adr_premium": False,
    },
    "4137": {
        "name": "大江",
        "industry": "BioTech",
        "us_chain_tickers": ["EL", "LRLCY", "PG"],
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "1402": {
        "name": "遠東新",
        "industry": "Materials",
        "us_chain_tickers": ["NKE", "ADS", "PVH"],
        "vol_low": 0.15,
        "vol_high": 0.30,
        "use_adr_premium": False,
    },
    "3131": {
        "name": "弘塑",
        "industry": "Semiconductor",
        "us_chain_tickers": ["ASML", "AMAT", "TSM", "KLAC"],
        "vol_low": 0.30,
        "vol_high": 0.60,
        "use_adr_premium": False,
    },
    "3583": {
        "name": "辛耘",
        "industry": "Semiconductor",
        "us_chain_tickers": ["ASML", "AMAT", "TSM", "LRCX"],
        "vol_low": 0.30,
        "vol_high": 0.60,
        "use_adr_premium": False,
    },
    "2376": {
        "name": "技嘉",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["NVDA", "AMD", "MSFT", "SMCI"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "3017": {
        "name": "奇鋐",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["NVDA", "VRT", "AMD", "HPE"],
        "vol_low": 0.30,
        "vol_high": 0.60,
        "use_adr_premium": False,
    },
    "6643": {
        "name": "M31",
        "industry": "Semiconductor",
        "us_chain_tickers": ["CDNS", "SNPS", "ARM", "TSM"],
        "vol_low": 0.35,
        "vol_high": 0.70,
        "use_adr_premium": False,
    },
    "6805": {
        "name": "富世達",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["AAPL", "GOOGL", "VTI"],
        "vol_low": 0.35,
        "vol_high": 0.75,
        "use_adr_premium": False,
    },
    "2049": {
        "name": "上銀",
        "industry": "Materials",
        "us_chain_tickers": ["CAT", "DE", "ABB", "VTI"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "2610": {
        "name": "華航",
        "industry": "Shipping",
        "us_chain_tickers": ["DAL", "UAL", "AAL", "LUV"],
        "vol_low": 0.25,
        "vol_high": 0.50,
        "use_adr_premium": False,
    },
    "1504": {
        "name": "東元",
        "industry": "Energy",
        "us_chain_tickers": ["GE", "ABB", "VRT", "ETN"],
        "vol_low": 0.15,
        "vol_high": 0.35,
        "use_adr_premium": False,
    },
    "6806": {
        "name": "森崴能源",
        "industry": "Energy",
        "us_chain_tickers": ["NEE", "ENPH", "FSLR", "VTI"],
        "vol_low": 0.30,
        "vol_high": 0.65,
        "use_adr_premium": False,
    },
    "2458": {
        "name": "義隆",
        "industry": "Semiconductor",
        "us_chain_tickers": ["AAPL", "MSFT", "HPQ", "DELL"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "2408": {
        "name": "南亞科",
        "industry": "Semiconductor",
        "us_chain_tickers": ["MU", "WDC", "STX", "TSM"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "8033": {
        "name": "雷虎",
        "industry": "Domestic",
        "us_chain_tickers": ["LMT", "RTX", "BA", "VTI"],
        "vol_low": 0.40,
        "vol_high": 0.85,
        "use_adr_premium": False,
    },
    "4763": {
        "name": "材料-KY",
        "industry": "Materials",
        "us_chain_tickers": ["PM", "MO", "VTI"],
        "vol_low": 0.30,
        "vol_high": 0.70,
        "use_adr_premium": False,
    },
    "1717": {
        "name": "長興",
        "industry": "Materials",
        "us_chain_tickers": ["DOW", "LYB", "VTI"],
        "vol_low": 0.15,
        "vol_high": 0.35,
        "use_adr_premium": False,
    },
    "2886": {
        "name": "兆豐金",
        "industry": "Finance",
        "us_chain_tickers": ["XLF", "JPM", "GS", "VTI"],
        "vol_low": 0.12,
        "vol_high": 0.25,
        "use_adr_premium": False,
    },
    "2345": {
        "name": "智邦",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["CSCO", "ANET", "NVDA"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "2353": {
        "name": "宏碁",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["MSFT", "INTC", "HPQ"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "2409": {
        "name": "友達",
        "industry": "Semiconductor",
        "us_chain_tickers": ["VTI", "AAPL", "TSLA"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "2313": {
        "name": "華通",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["VTI", "TSLA", "AAPL"],
        "vol_low": 0.30,
        "vol_high": 0.65,
        "use_adr_premium": False,
    },
    "2884": {
        "name": "玉山金",
        "industry": "Finance",
        "us_chain_tickers": ["XLF", "JPM", "VTI"],
        "vol_low": 0.10,
        "vol_high": 0.22,
        "use_adr_premium": False,
    },
    "6415": {
        "name": "矽力-KY",
        "industry": "Semiconductor",
        "us_chain_tickers": ["ADI", "TXN", "ON"],
        "vol_low": 0.35,
        "vol_high": 0.75,
        "use_adr_premium": False,
    },
    "1760": {
        "name": "寶齡富錦",
        "industry": "BioTech",
        "us_chain_tickers": ["XLV", "PFE", "MRK"],
        "vol_low": 0.30,
        "vol_high": 0.65,
        "use_adr_premium": False,
    },
    "3532": {
        "name": "台勝科",
        "industry": "Semiconductor",
        "us_chain_tickers": ["MU", "TSM", "WOLF"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "3035": {
        "name": "智原",
        "industry": "Semiconductor",
        "us_chain_tickers": ["ARM", "NVDA", "CDNS"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "2449": {
        "name": "京元電",
        "industry": "Semiconductor",
        "us_chain_tickers": ["TSM", "AMD", "NVDA"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "2301": {
        "name": "光寶科",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["GE", "VRT", "MSFT"],
        "vol_low": 0.20,
        "vol_high": 0.45,
        "use_adr_premium": False,
    },
    "1514": {
        "name": "亞力",
        "industry": "Energy",
        "us_chain_tickers": ["GE", "ETN", "ABB"],
        "vol_low": 0.25,
        "vol_high": 0.55,
        "use_adr_premium": False,
    },
    "6533": {
        "name": "晶心科",
        "industry": "Semiconductor",
        "us_chain_tickers": ["ARM", "NVDA", "MSFT"],
        "vol_low": 0.35,
        "vol_high": 0.75,
        "use_adr_premium": False,
    },
    "3702": {
        "name": "大聯大",
        "industry": "Semiconductor",
        "us_chain_tickers": ["VTI", "AVGO", "INTC"],
        "vol_low": 0.15,
        "vol_high": 0.30,
        "use_adr_premium": False,
    },
    "6214": {
        "name": "精誠",
        "industry": "Domestic",
        "us_chain_tickers": ["MSFT", "GOOGL", "VTI"],
        "vol_low": 0.15,
        "vol_high": 0.35,
        "use_adr_premium": False,
    },
    "4147": {
        "name": "中裕",
        "industry": "BioTech",
        "us_chain_tickers": ["XLV", "IBB", "PFE"],
        "vol_low": 0.30,
        "vol_high": 0.70,
        "use_adr_premium": False,
    },
    "4174": {
        "name": "浩鼎",
        "industry": "BioTech",
        "us_chain_tickers": ["XLV", "IBB", "MRK"],
        "vol_low": 0.35,
        "vol_high": 0.75,
        "use_adr_premium": False,
    },
    "2399": {
        "name": "映泰",
        "industry": "Web3_Hardware",
        "us_chain_tickers": ["NVDA", "AMD", "BTC"],
        "vol_low": 0.40,
        "vol_high": 0.90,
        "use_adr_premium": False,
    },
    "6150": {
        "name": "撼訊",
        "industry": "Web3_Hardware",
        "us_chain_tickers": ["AMD", "NVDA", "BTC"],
        "vol_low": 0.45,
        "vol_high": 1.00,
        "use_adr_premium": False,
    },
}

HORIZON        = 30        # 預測天數
LOOKBACK       = 252       # TFT encoder 序列長度（約 1 年交易日）

# ── 訓練起始日期（方向1：延長訓練期）────────────────────────
# 資料可用性：
#   股價/財報：1994~  月營收：2002~  PER：2005~  三大法人：2012~
# → 2010 以前缺三大法人特徵（fund_flow 欄位為 NaN → XGB/LGB 自動補 0）
# → 相較 2015 起，多出 ~1,300 天訓練資料，OOF 預計 +400 筆
TRAIN_START_DATE = "2010-01-01"

MIN_TRAIN_DAYS = 252 * 3   # Walk-Forward 最少訓練天數（3 年，配合延長期）
RETRAIN_FREQ   = 21        # 方向2：縮小 fold step（21 天≈1個月），OOF +43%

# ─────────────────────────────────────────────
# 特徵分組（對應第一性原則五大類）
# ─────────────────────────────────────────────
FEATURE_GROUPS = {
    # ① 技術動能：價格 / 成交量模式
    "technical": [
        "log_return_1d", "log_return_5d", "log_return_10d", "log_return_20d",
        "realized_vol_10d", "realized_vol_20d", "realized_vol_60d",
        "ma_20", "ma_50", "ma_120",
        "ma_cross_20_50", "ma_cross_50_120",
        "price_to_ma20", "price_to_ma50", "price_to_ma120",
        "rsi_14", "rsi_28",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_pct",
        "atr_14",
        "volume_ma_20", "volume_ratio_20",
        "price_volume_corr_20",
        "momentum_10d", "momentum_20d",
        "high_low_spread", "open_close_spread",
    ],
    # ② 資金流情緒：法人 & 散戶
    "fund_flow": [
        "foreign_net", "foreign_net_vol_ratio",
        "trust_net", "trust_net_vol_ratio",
        "dealer_net",              # 全局 OOF 零重要，但 Hold-Out 有貢獻，保留
        "foreign_net_ma5", "foreign_net_ma20",
        "foreign_holding_ratio", "foreign_holding_chg_5d",
        "margin_balance", "margin_balance_chg",
        "short_balance", "short_balance_chg",
        "margin_short_ratio",
        "retail_vs_inst",
    ],
    # ③ 基本面脈衝：季報 / 月營收
    "fundamental": [
        "revenue_yoy", "revenue_mom",
        "revenue_3m_avg_yoy",
        "gross_margin", "gross_margin_chg_qoq",
        "operating_income_margin",
        "eps_ttm", "eps_qoq", "eps_yoy",
        "roe_ttm",
        "current_ratio",
        "debt_ratio",
        "cash_ratio",
        "capex_ratio",
    ],
    # ④ 估值錨點：PER、PBR、殖利率
    "valuation": [
        "per", "per_pct_rank_252",
        "pbr", "pbr_pct_rank_252",
        "dividend_yield", "dy_pct_rank_252",
        "per_deviation_from_ma",
    ],
    # ⑤ 宏觀因子：利率、匯率、指數
    "macro": [
        "fed_rate", "fed_rate_chg_30d",
        "boj_rate", "ecb_rate",
        "usd_twd_spot", "usd_twd_chg_10d",
        "jpy_twd_spot", "jpy_twd_chg_10d",
        "eur_twd_spot", "eur_twd_chg_10d",
        "US10Y", "US2Y", "us_yield_spread",
        "taiex_ret_5d", "taiex_ret_20d",
        "tpex_ret_5d",
        "taiex_rel_strength",
    ],
    # ⑥ 事件驅動：股利、財報日程
    "event": [
        # days_to_next_ex_dividend: 零重要性，已移除
        "cash_dividend_ttm",
        "days_since_last_earnings",
        # dividend_ex_dummy: 零重要性，已移除
    ],
    # ⑦ 滾動高階統計
    "rolling_stats": [
        "skew_20d", "skew_60d",
        "kurt_20d", "kurt_60d",
        "autocorr_lag1_20d",
        "sharpe_20d", "sharpe_60d",
    ],
    # ⑧ 期貨籌碼（新增：台指期 TX + 台指選擇權 TFO）
    #   TX 與台積電相關性 > 0.85（台積電佔加權指數 ~30%）
    #   TFO PCR 是機構避險壓力的直接代理指標
    "futures_chip": [
        "tx_oi_chg_1d",   # 近月 OI 日變化（資金進出）
        "tx_oi_chg_5d",   # 近月 OI 5 日變化（周趨勢）
        "tx_basis",        # 期現貨基差（正=看多）
        "tx_basis_5d_chg", # 基差 5 日變化（轉折速度）
        "tx_vol_ma_ratio", # 台指期量能相對強度
        "tfo_pcr_volume",  # Put/Call 成交量比（恐慌指標）
        "tfo_pcr_oi",      # Put/Call 未平倉比（機構避險）
        # ★ 新增（中期信號）
        "tx_oi_direction_5d",# 台指期 OI 方向（+1=多方進場/-1=空方進場）
    ],
    # ⑨ 中期信號（Medium-term Signals）─────────────────────────
    # 補強未來 15~30 天預測的核心信號群
    "medium_term": [
        # ① 基本面動量
        "rev_yoy_positive_months",  # 月營收連續 YoY 正成長月數（動量確認）
        "rev_yoy_3m",               # 近 3 個月月營收 YoY 均值（短期動量）
        "gross_margin_qoq",         # 季毛利率 QoQ 實際變化值
        "gross_margin_qoq_dir",     # 季毛利率 QoQ 方向（+1/0/-1）
        "eps_accel_proxy",          # EPS 季加速度代理
        # ② 機構資金趨勢
        "foreign_net_weekly",       # 外資近 5 日累計淨買超（週化）
        "foreign_net_accel",        # 外資買超加速度（近週 vs 前週）
        "margin_chg_rate_5d",       # 融資餘額 5 日變化率（散戶擁擠度）
        "margin_chg_rate_20d",      # 融資餘額 20 日變化率（中期趨勢）
        "short_chg_rate_5d",        # 融券餘額 5 日變化率（空方動能）
        # ③ 市場結構信號
        "rs_line_20d",              # 個股 vs TAIEX 相對強弱（20日均）
        "rs_line_slope_5d",         # RS line 5 日斜率（是否在改善）
        "adr_premium",              # TSM ADR 折溢價（外資外部定價）
        "adr_premium_5d_chg",       # ADR 折溢價 5 日變化趨勢
        "adr_premium_ma5",          # ADR 折溢價 5 日均值
    ],
    "us_chain": [], # 佔位，由 get_all_features 動態填充
}

def get_all_features(stock_id: str = DEFAULT_STOCK_ID) -> list[str]:
    """
    依據 stock_id 動態生成特徵清單。
    """
    config = STOCK_CONFIGS.get(stock_id, STOCK_CONFIGS[DEFAULT_STOCK_ID])
    groups = FEATURE_GROUPS.copy()
    
    # 動態調整 us_chain 內容
    us_tickers = [t.lower().replace("^", "") for t in config["us_chain_tickers"]]
    us_chain_features = []
    if config.get("use_adr_premium", False):
        us_chain_features += ["tsm_premium", "tsm_premium_ma5"]
    
    for ticker in us_tickers:
        us_chain_features += [f"{ticker}_ret_1d", f"{ticker}_ret_5d", f"{ticker}_ret_20d"]
    
    groups["us_chain"] = us_chain_features
    return [f for grp in groups.values() for f in grp]

ALL_FEATURES = get_all_features(DEFAULT_STOCK_ID)


# ─────────────────────────────────────────────
# TFT 超參數
# ─────────────────────────────────────────────
TFT_PARAMS = {
    "hidden_size":           128,
    "lstm_layers":           2,
    "dropout":               0.1,
    "attention_head_size":   4,
    "max_encoder_length":    LOOKBACK,
    "max_prediction_length": HORIZON,
    "learning_rate":         1e-3,
    "batch_size":            64,
    "max_epochs":            100,
    "patience":              15,
    "gradient_clip_val":     0.1,
    "quantiles":             [0.1, 0.25, 0.5, 0.75, 0.9],
}

# CPU 快速模式參數（tft_model.py 在無 GPU 時自動套用，不需手動設定）
# hidden_size=32, lstm_layers=1, patience=5, max_epochs=30 → ~15 分鐘/fold
TFT_PARAMS_CPU_OVERRIDE = {
    "hidden_size":           32,
    "lstm_layers":           1,
    "attention_head_size":   1,
    "patience":              5,
    "max_epochs":            30,
}

# ─────────────────────────────────────────────
# XGBoost 超參數
# ─────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":          1000,
    "learning_rate":         0.02,
    "max_depth":             6,     # 恢復 6：depth=4 導致 OOF 集中於高端反而更差
    "min_child_weight":      5,     # 逸中值（預設 1）：减少極端樹叉不過度限制
    "gamma":                 0.1,   # 樹分賸最小減少量（leaf-purity 正則化）
    "subsample":             0.8,
    "colsample_bytree":      0.7,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "random_state":          42,
    "n_jobs":                -1,
    "early_stopping_rounds": 50,
}

# ─────────────────────────────────────────────
# LightGBM 超參數
# ─────────────────────────────────────────────
LGB_PARAMS = {
    "n_estimators":          1000,
    "learning_rate":         0.02,
    "num_leaves":            63,
    "max_depth":             -1,
    "subsample":             0.8,
    "colsample_bytree":      0.7,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "random_state":          42,
    "n_jobs":                -1,
    "early_stopping_rounds": 50,
    "verbose":               -1,
}

# ─────────────────────────────────────────────
# Walk-Forward CV 設定（Purged + Embargo）
# ─────────────────────────────────────────────
WF_CONFIG = {
    "train_window": 252 * 3,   # 訓練窗口：3年（配合 TRAIN_START_DATE 2010）
    "val_window":   126,       # 驗證窗口：半年（從 252 縮短，保留更多 fold）
    "embargo_days": 45,        # 禁區天數（考慮月報/季報公告延遲，由 30 提升至 45）
    "step_days":    RETRAIN_FREQ,  # fold 間距：21 天（方向2 優化）
    # ── test_window（修正 Fold DA std 38% 根本問題）──────────────
    # 每個 fold test 窗口大小，與 step_days 解耦。
    #   問題：test_window = step_days = 21 → 每 fold 只有 21 個二元樣本
    #         → DA 只有 k/21 共 22 種可能，理論 std 上限 ~50%，實測 38%。
    #   修正：test_window = 126（半年）→ 每 fold ≥ 100 個樣本
    #         → DA 理論 std 上限 = √(0.25/126) ≈ 4.5%（改善 8 倍）
    #   注意：test 窗口相鄰 fold 間有重疊（rolling window），這是合法的 ——
    #         重疊不等於洩漏（訓練集與 test 集仍嚴格分開），只是同一歷史日
    #         被多個 fold 評估，提升統計穩定性。
    "test_window":  126,
}

# ─────────────────────────────────────────────
# 評估目標
# ─────────────────────────────────────────────
EVAL_TARGETS = {
    "directional_accuracy": 0.65,
    "ic":                   0.05,
    "sharpe":               1.0,
}

# 組合層回測目標 (Portfolio Level)
PORTFOLIO_EVAL_TARGETS = {
    "portfolio_sharpe":   1.2,      # 考慮多元分散後的目標夏普
    "max_drawdown":      -0.15,     # 最大回撤限制 (15% 以內)
    "calmar_ratio":       2.0,      # 年化報酬 / 最大回撤
    "beta_to_taiex":      0.5,      # 對大盤的 Beta 暴露 (希望低於 0.5)
    "turnover_rate":      2.0,      # 年化換手率限制
    "worst_month_ret":   -0.08,     # 最差單月跌幅限制
}

# ─────────────────────────────────────────────
# Regime Detection 設定
# ─────────────────────────────────────────────
REGIME_CONFIG = {
    # realized_vol_20d（年化）閾值：低波動 / 高波動 / 極端波動
    "vol_low":    0.20,   # < 20%  → 低波動 regime（趨勢穩定）
    "vol_high":   0.40,   # > 40%  → 高波動 regime（震盪/危機）
    "train_split": 0.30,  # 新增：Regime 訓練切分點 (30% 波動率)
    # Hold-Out 長度（交易日）：2 年
    "oos_window": 252 * 2,
}

# ─────────────────────────────────────────────
# 國際標的清單 (由各個股配置自動彙整)
# ─────────────────────────────────────────────
def _get_international_watchlist():
    tickers = set()
    # 這裡 STOCK_CONFIGS 已經在上方定義好了
    for cfg in STOCK_CONFIGS.values():
        if "us_chain_tickers" in cfg:
            tickers.update(cfg["us_chain_tickers"])
    # 額外加入一些全局總經連動標的 (Index ETFs)
    tickers.update(["SPY", "QQQ", "SOXX", "DIA", "VTI", "TLT", "UUP", "TSM", "NVDA", "AAPL"])
    return sorted(list(tickers))

INTERNATIONAL_WATCHLIST = _get_international_watchlist()

# ─────────────────────────────────────────────
# 八二法則 (Pareto Principle) 設定
# ─────────────────────────────────────────────
PARETO_RATIO = 0.2  # 特徵層面：只保留前 20% 黃金特徵
CONFIDENCE_THRESHOLD = 0.75  # 訊號層面：極端高信心門檻
TIER_1_STOCKS = ["2330", "2317", "2454", "2382", "2881", "2412", "2308", "2882"] # 標的層面：核心權值股
