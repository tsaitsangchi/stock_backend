"""
config.py — 全域設定：路徑、超參數、特徵分組
資料來源：PostgreSQL 17（連線設定在下方 DB_CONFIG）

[P0-SECURITY 修正] 敏感資訊從 .env 檔案載入，不再硬編碼在原始碼中。
  1. 複製 .env.example → .env
  2. 填入 FINMIND_TOKEN（從 https://finmindtrade.com 取得）
  3. 將 .env 加入 .gitignore（切勿提交至版本控制）
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 載入 .env（必須在所有 os.environ 存取之前）
# ─────────────────────────────────────────────
load_dotenv()

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
# [P0-SECURITY] FinMind API Token
# 若 .env 中沒有設定，直接拋出 KeyError，防止靜默失敗
# ─────────────────────────────────────────────
FINMIND_TOKEN: str = os.environ["FINMIND_TOKEN"]

# ─────────────────────────────────────────────
# [P0] 統一的 PostgreSQL 連線設定（全系統唯一定義處）
# 所有 fetch_*.py 均從此處 import，不再各自重複定義
# ─────────────────────────────────────────────
DB_CONFIG: dict = {
    "dbname":   "stock",
    "user":     "stock",
    "password": os.environ.get("DB_PASSWORD", "stock"),
    "host":     os.environ.get("DB_HOST", "localhost"),
    "port":     os.environ.get("DB_PORT", "5432"),
}

# ─────────────────────────────────────────────
# 資料來源：PostgreSQL 17
# 對應資料表：
#   stock_price, stock_per, financial_statements, balance_sheet,
#   dividend, institutional_investors_buy_sell,
#   margin_purchase_short_sale, shareholding,
#   interest_rate, exchange_rate, total_return_index, month_revenue
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 風險管理參數 (Risk Management)
# ─────────────────────────────────────────────
RISK_CONFIG = {
    "target_core_ratio":    0.80,
    "target_agg_ratio":     0.20,
    "rebalance_threshold":  0.05,
    "max_pos_core":         0.15,
    "max_pos_agg":          0.05,
    "min_avg_vol_twd":      50_000_000,
    "max_vol_participation": 0.10,
    "target_payoff_ratio":  2.0,
    "min_expected_value":   0.01,
}

# ─────────────────────────────────────────────
# 資料可用性與完整性註冊表 (Table Registry)
# ─────────────────────────────────────────────
# 用於自動化監控、健康檢查與斷層癒合。
TABLE_REGISTRY = {
    # 核心價量
    "stock_price":                      {"type": "daily", "id_col": "stock_id", "lag": 1},
    "stock_per":                        {"type": "daily", "id_col": "stock_id", "lag": 1},
    "price_adj":                        {"type": "daily", "id_col": "stock_id", "lag": 1},
    "day_trading":                     {"type": "daily", "id_col": "stock_id", "lag": 1},
    "price_limit":                     {"type": "daily", "id_col": "stock_id", "lag": 1},
    
    # 籌碼面
    "institutional_investors_buy_sell": {"type": "daily", "id_col": "stock_id", "lag": 1},
    "margin_purchase_short_sale":       {"type": "daily", "id_col": "stock_id", "lag": 1},
    "shareholding":                    {"type": "daily", "id_col": "stock_id", "lag": 1},
    "securities_lending":              {"type": "daily", "id_col": "stock_id", "lag": 1},
    "daily_short_balance":             {"type": "daily", "id_col": "stock_id", "lag": 1},
    "eight_banks_buy_sell":            {"type": "daily", "id_col": "stock_id", "lag": 1},
    "sponsor_chip":                    {"type": "daily", "id_col": "stock_id", "lag": 2}, # 分點資料通常較慢
    
    # 基本面 (月/季)
    "month_revenue":                   {"type": "monthly", "id_col": "stock_id", "lag": 40},
    "financial_statements":            {"type": "quarterly", "id_col": "stock_id", "lag": 145},
    "balance_sheet":                   {"type": "quarterly", "id_col": "stock_id", "lag": 145},
    "cash_flows_statement":            {"type": "quarterly", "id_col": "stock_id", "lag": 145},
    "dividend":                        {"type": "event", "id_col": "stock_id", "lag": 365},
    
    # 市場層級與國際
    "total_margin_short":              {"type": "market", "id_col": None, "lag": 1},
    "total_inst_investors":            {"type": "market", "id_col": None, "lag": 1},
    "futures_inst_investors":          {"type": "market", "id_col": None, "lag": 1},
    "options_inst_investors":          {"type": "market", "id_col": None, "lag": 1},
    "us_stock_price":                  {"type": "daily", "id_col": "stock_id", "lag": 1},
    "exchange_rate":                  {"type": "daily", "id_col": "currency", "lag": 1},
    "interest_rate":                  {"type": "daily", "id_col": "country", "lag": 7},
}

DATA_LAG_CONFIG = {k: v["lag"] for k, v in TABLE_REGISTRY.items()}
DATA_LAG_CONFIG["annual_report"]    = 90
DATA_LAG_CONFIG["quarterly_report"] = 45

# ─────────────────────────────────────────────
# 訓練策略配置 (Training Strategy)
# ─────────────────────────────────────────────
TRAINING_STRATEGY = {
    "use_global_backbone": True,
    "finetune_local":      True,
    "feature_selection":   "robust_ic",
}

SECTOR_POOLS = {
    "Semiconductor": ["2330", "2303", "2454", "3661", "3037", "3711"],
    "AI_Hardware":   ["2382", "2317", "6669", "2357", "3231", "2417"],
    "Finance":       ["2881", "2882", "2886", "2891", "5880"],
}

STOCK_ID       = "2330"

# ─────────────────────────────────────────────
# 交易成本與市場衝擊 (Friction & Costs)
# ─────────────────────────────────────────────
FRICTION_CONFIG = {
    "commission":          0.001425,
    "securities_tax":      0.003,
    "slippage_large_cap":  0.001,
    "slippage_small_cap":  0.005,
}

LARGE_CAP_TICKERS = ["2330", "2317", "2454", "2308", "2881", "2882", "2303"]
DEFAULT_STOCK_ID  = "2330"

def calculate_net_return(gross_return: float, ticker: str) -> float:
    is_large_cap = ticker in LARGE_CAP_TICKERS
    slippage = FRICTION_CONFIG["slippage_large_cap"] if is_large_cap else FRICTION_CONFIG["slippage_small_cap"]
    total_cost = (FRICTION_CONFIG["commission"] * 2 +
                  FRICTION_CONFIG["securities_tax"] +
                  slippage * 2)
    return gross_return - total_cost

# ─────────────────────────────────────────────
# 個股客製化配置 (Multi-Stock Framework)
# ─────────────────────────────────────────────
STOCK_CONFIGS = {
    "1101": {'name': '台泥', 'industry': 'Materials', 'us_chain_tickers': ['VMC', 'MLM']},
    "1216": {'name': '統一', 'industry': 'Consumer', 'us_chain_tickers': ['VTI']},
    "1301": {'name': '台塑', 'industry': 'Materials', 'us_chain_tickers': ['DOW', 'LYB', 'XOM']},
    "1503": {'name': '士電', 'industry': 'AI_Hardware', 'us_chain_tickers': ['ETN', 'PWR', 'GE']},
    "1504": {'name': '東元', 'industry': 'Energy', 'us_chain_tickers': ['ABB', 'SIEGY']},
    "1513": {'name': '中興電', 'industry': 'Energy', 'us_chain_tickers': ['ETN', 'PWR', 'GE']},
    "1514": {'name': '亞力', 'industry': 'Energy', 'us_chain_tickers': ['ETN', 'PWR', 'GE']},
    "1519": {'name': '華城', 'industry': 'AI_Hardware', 'us_chain_tickers': ['ETN', 'PWR', 'GE']},
    "1529": {'name': '樂事綠能', 'industry': 'Energy', 'us_chain_tickers': ['TAN']},
    "1560": {'name': '中砂', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMAT', 'ASML', 'TSM']},
    "1565": {'name': '精華', 'industry': 'Biotech', 'us_chain_tickers': ['VTI']},
    "1597": {'name': '直得', 'industry': 'Robotics', 'us_chain_tickers': ['ABB', 'FANUY']},
    "1707": {'name': '葡萄王', 'industry': 'Biotech', 'us_chain_tickers': ['VTI']},
    "1717": {'name': '長興', 'industry': 'Semi_Materials', 'us_chain_tickers': ['VTI']},
    "1752": {'name': '南光', 'industry': 'Biotech', 'us_chain_tickers': ['IBB', 'XBI']},
    "1760": {'name': '寶齡富錦', 'industry': 'Biotech', 'us_chain_tickers': ['IBB', 'XBI', 'PFE']},
    "1786": {'name': '科妍', 'industry': 'Biotech', 'us_chain_tickers': ['IBB', 'XBI']},
    "1795": {'name': '美時', 'industry': 'Biotech', 'us_chain_tickers': ['PFE', 'MRK']},
    "2002": {'name': '中鋼', 'industry': 'Materials', 'us_chain_tickers': ['X', 'NUE', 'STLD']},
    "2049": {'name': '上銀', 'industry': 'Robotics', 'us_chain_tickers': ['ABB', 'FANUY', 'KUKA']},
    "2249": {'name': '湧盛', 'industry': 'Emerging', 'us_chain_tickers': ['VTI']},
    "2301": {'name': '光寶科', 'industry': 'AI_Hardware', 'us_chain_tickers': ['MSFT', 'GOOGL', 'NVDA']},
    "2303": {'name': '聯電', 'industry': 'Semiconductor', 'us_chain_tickers': ['UMC', 'INTC', 'TXN']},
    "2308": {'name': '台達電', 'industry': 'AI_Hardware', 'us_chain_tickers': ['TSLA', 'NVDA', 'ENPH', 'SOXX']},
    "2317": {'name': '鴻海', 'industry': 'AI_Hardware', 'us_chain_tickers': ['AAPL', 'HPE', 'MSFT']},
    "2324": {'name': '仁寶', 'industry': 'AI_Hardware', 'us_chain_tickers': ['DELL', 'HPQ']},
    "2327": {'name': '國巨', 'industry': 'Semiconductor', 'us_chain_tickers': ['TEL', 'APH']},
    "2330": {'name': '台積電', 'industry': 'Semiconductor', 'us_chain_tickers': ['TSM', 'NVDA', 'AAPL', 'SOXX'], 'use_adr_premium': True},
    "2337": {'name': '旺宏', 'industry': 'Semiconductor', 'us_chain_tickers': ['MU', 'WDC']},
    "2338": {'name': '光罩', 'industry': 'Semiconductor', 'us_chain_tickers': ['TSM', 'UMC']},
    "2344": {'name': '華邦電', 'industry': 'Semiconductor', 'us_chain_tickers': ['MU', 'WDC', 'STX']},
    "2352": {'name': '佳世達', 'industry': 'AI_Hardware', 'us_chain_tickers': ['PHG', 'GE']},
    "2353": {'name': '宏碁', 'industry': 'AI_Hardware', 'us_chain_tickers': ['INTC', 'MSFT']},
    "2356": {'name': '英業達', 'industry': 'AI_Server', 'us_chain_tickers': ['NVDA', 'AMD']},
    "2357": {'name': '華碩', 'industry': 'AI_Hardware', 'us_chain_tickers': ['INTC', 'NVDA', 'AMD']},
    "2359": {'name': '所羅門', 'industry': 'Robotics', 'us_chain_tickers': ['NVDA', 'FANUY', 'ABB']},
    "2360": {'name': '致茂', 'industry': 'Semiconductor', 'us_chain_tickers': ['NVDA', 'TSM']},
    "2368": {'name': '金像電', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'MSFT']},
    "2376": {'name': '技嘉', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'AMD', 'SMCI']},
    "2377": {'name': '微星', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'INTC']},
    "2379": {'name': '瑞昱', 'industry': 'Semiconductor', 'us_chain_tickers': ['QCOM', 'AVGO']},
    "2382": {'name': '廣達', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'SMCI']},
    "2383": {'name': '台光電', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'AMD']},
    "2395": {'name': '研華', 'industry': 'Robotics', 'us_chain_tickers': ['HON', 'ROK', 'NVDA']},
    "2401": {'name': '凌陽', 'industry': 'Semiconductor', 'us_chain_tickers': ['VTI']},
    "2404": {'name': '漢唐', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMAT', 'TSM']},
    "2408": {'name': '南亞科', 'industry': 'Semiconductor', 'us_chain_tickers': ['MU', 'WDC', 'STX']},
    "2409": {'name': '友達', 'industry': 'Semiconductor', 'us_chain_tickers': ['LPL', 'SONY']},
    "2421": {'name': '建準', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'VRT']},
    "2436": {'name': '偉詮電', 'industry': 'Semiconductor', 'us_chain_tickers': ['VTI']},
    "2439": {'name': '美律', 'industry': 'AI_Hardware', 'us_chain_tickers': ['AAPL']},
    "2449": {'name': '京元電子', 'industry': 'Semiconductor', 'us_chain_tickers': ['NVDA', 'AMD', 'QCOM']},
    "2454": {'name': '聯發科', 'industry': 'Semiconductor', 'us_chain_tickers': ['QCOM', 'ARM', 'SOXX', 'NVDA']},
    "2455": {'name': '全新', 'industry': 'Semiconductor', 'us_chain_tickers': ['AVGO', 'QRVO', 'SWKS']},
    "2474": {'name': '可成', 'industry': 'AI_Hardware', 'us_chain_tickers': ['AAPL']},
    "2492": {'name': '華新科', 'industry': 'Semiconductor', 'us_chain_tickers': ['TEL', 'APH']},
    "2881": {'name': '富邦金', 'industry': 'Finance', 'us_chain_tickers': ['XLF', 'KBE', 'TNX']},
    "2882": {'name': '國泰金', 'industry': 'Finance', 'us_chain_tickers': ['XLF', 'KBE', 'VTI']},
    "2884": {'name': '玉山金', 'industry': 'Finance', 'us_chain_tickers': ['XLF', 'VTI']},
    "2886": {'name': '兆豐金', 'industry': 'Finance', 'us_chain_tickers': ['XLF', 'KBE', 'TLT']},
    "2891": {'name': '中信金', 'industry': 'Finance', 'us_chain_tickers': ['XLF', 'VTI']},
    "2892": {'name': '第一金', 'industry': 'Finance', 'us_chain_tickers': ['XLF', 'TLT']},
    "3006": {'name': '晶豪科', 'industry': 'Semiconductor', 'us_chain_tickers': ['MU', 'WDC']},
    "3008": {'name': '大立光', 'industry': 'Semiconductor', 'us_chain_tickers': ['AAPL', 'LITE']},
    "3013": {'name': '晟銘電', 'industry': 'AI_Hardware', 'us_chain_tickers': ['SMCI', 'NVDA']},
    "3016": {'name': '嘉晶', 'industry': 'Semiconductor', 'us_chain_tickers': ['WOLF', 'ON']},
    "3017": {'name': '奇鋐', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'VRT', 'AMD']},
    "3019": {'name': '亞光', 'industry': 'Robotics', 'us_chain_tickers': ['AAPL', 'TSLA']},
    "3030": {'name': '德律', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMAT']},
    "3034": {'name': '聯詠', 'industry': 'Semiconductor', 'us_chain_tickers': ['AAPL', 'VTI']},
    "3035": {'name': '智原', 'industry': 'Semiconductor', 'us_chain_tickers': ['ARM', 'QCOM', 'SOXX']},
    "3037": {'name': '欣興', 'industry': 'Semiconductor', 'us_chain_tickers': ['NVDA', 'AMD', 'INTC', 'SOXX']},
    "3044": {'name': '健鼎', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'MSFT']},
    "3081": {'name': '聯亞', 'industry': 'Telecom', 'us_chain_tickers': ['LITE', 'COHR']},
    "3088": {'name': '艾訊', 'industry': 'Robotics', 'us_chain_tickers': ['HON', 'ROK']},
    "3105": {'name': '穩懋', 'industry': 'Semiconductor', 'us_chain_tickers': ['QRVO', 'SWKS', 'AVGO']},
    "3131": {'name': '弘塑', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMAT', 'ASML']},
    "3141": {'name': '晶宏', 'industry': 'Semiconductor', 'us_chain_tickers': ['EINK', 'AAPL']},
    "3227": {'name': '原相', 'industry': 'Semiconductor', 'us_chain_tickers': ['SONY', 'VTI']},
    "3231": {'name': '緯創', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'SMCI', 'MSFT', 'DELL']},
    "3264": {'name': '欣銓', 'industry': 'Semiconductor', 'us_chain_tickers': ['ASX', 'TXN']},
    "3324": {'name': '雙鴻', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'VRT', 'AMD']},
    "3376": {'name': '新日興', 'industry': 'AI_Hardware', 'us_chain_tickers': ['AAPL']},
    "3406": {'name': '玉晶光', 'industry': 'Robotics', 'us_chain_tickers': ['AAPL', 'META']},
    "3443": {'name': '創意', 'industry': 'Semiconductor', 'us_chain_tickers': ['NVDA', 'AMD', 'SOXX']},
    "3455": {'name': '由田', 'industry': 'TPEx_HighTurnover', 'us_chain_tickers': ['VTI']},
    "3481": {'name': '群創', 'industry': 'Semiconductor', 'us_chain_tickers': ['LPL', 'SONY']},
    "3504": {'name': '揚明光', 'industry': 'Robotics', 'us_chain_tickers': ['AAPL', 'META']},
    "3515": {'name': '華擎', 'industry': 'AI_Hardware', 'us_chain_tickers': ['AMD', 'NVDA', 'INTC']},
    "3529": {'name': '力旺', 'industry': 'Semiconductor', 'us_chain_tickers': ['ARM', 'NVDA', 'SOXX']},
    "3532": {'name': '台勝科', 'industry': 'Semiconductor', 'us_chain_tickers': ['SUMCO', 'Shin-Etsu']},
    "3533": {'name': '嘉澤', 'industry': 'AI_Hardware', 'us_chain_tickers': ['INTC', 'AMD', 'NVDA']},
    "3545": {'name': '敦泰', 'industry': 'Semiconductor', 'us_chain_tickers': ['VTI', 'SOXX']},
    "3576": {'name': '聯合再生', 'industry': 'Energy', 'us_chain_tickers': ['TAN', 'FSLR']},
    "3583": {'name': '辛耘', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMAT', 'ASML']},
    "3633": {'name': '云光', 'industry': 'Emerging', 'us_chain_tickers': ['VTI']},
    "3653": {'name': '健策', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'AMD']},
    "3661": {'name': '世芯-KY', 'industry': 'Semiconductor', 'us_chain_tickers': ['NVDA', 'AMZN', 'MSFT']},
    "3680": {'name': '家登', 'industry': 'Semiconductor', 'us_chain_tickers': ['ASML', 'TSM']},
    "3693": {'name': '營邦', 'industry': 'AI_Hardware', 'us_chain_tickers': ['SMCI', 'DELL']},
    "3706": {'name': '神達', 'industry': 'AI_Hardware', 'us_chain_tickers': ['DELL', 'HPE']},
    "3711": {'name': '日月光投控', 'industry': 'Semiconductor', 'us_chain_tickers': ['ASX', 'INTC', 'AMAT']},
    "3712": {'name': '永崴投控', 'industry': 'Energy', 'us_chain_tickers': ['TAN', 'FSLR']},
    "4107": {'name': '邦特', 'industry': 'Biotech_Yield', 'us_chain_tickers': ['VTI']},
    "4126": {'name': '太醫', 'industry': 'Biotech_Yield', 'us_chain_tickers': ['VTI']},
    "4137": {'name': '麗豐-KY', 'industry': 'Biotech', 'us_chain_tickers': ['VTI']},
    "4722": {'name': '國精化', 'industry': 'Semi_Materials', 'us_chain_tickers': ['VTI']},
    "4749": {'name': '新應材', 'industry': 'Semi_Materials', 'us_chain_tickers': ['AMAT', 'ASML']},
    "4771": {'name': '望隼', 'industry': 'Biotech_Yield', 'us_chain_tickers': ['VTI']},
    "4772": {'name': '台特化', 'industry': 'Semi_Materials', 'us_chain_tickers': ['AMAT', 'ASML']},
    "5234": {'name': '達興材料', 'industry': 'Semi_Materials', 'us_chain_tickers': ['AMAT']},
    "5269": {'name': '祥碩', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMD', 'INTC', 'AAPL']},
    "5274": {'name': '信驊', 'industry': 'Semiconductor', 'us_chain_tickers': ['ASPEED', 'NVDA', 'MSFT', 'AMZN']},
    "6187": {'name': '萬潤', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMAT', 'ASML', 'TSM']},
    "6223": {'name': '旺矽', 'industry': 'Semiconductor', 'us_chain_tickers': ['NVDA', 'AMAT']},
    "6274": {'name': '台燿', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'AMD']},
    "6285": {'name': '啟碁', 'industry': 'Telecom', 'us_chain_tickers': ['CSCO', 'VTI']},
    "6446": {'name': '藥華藥', 'industry': 'Biotech', 'us_chain_tickers': ['IBB', 'XBI', 'MRK']},
    "6472": {'name': '保瑞', 'industry': 'Biotech', 'us_chain_tickers': ['PFE', 'MRK', 'AZN']},
    "6488": {'name': '環球晶', 'industry': 'Semiconductor', 'us_chain_tickers': ['WOLF', 'ON', 'SUMCO']},
    "6491": {'name': '晶碩', 'industry': 'Biotech_Yield', 'us_chain_tickers': ['VTI']},
    "6515": {'name': '穎崴', 'industry': 'Semiconductor', 'us_chain_tickers': ['NVDA', 'AMD']},
    "6523": {'name': '達爾膚', 'industry': 'Biotech_Yield', 'us_chain_tickers': ['VTI']},
    "6669": {'name': '緯穎', 'industry': 'AI_Hardware', 'us_chain_tickers': ['MSFT', 'META', 'AMZN', 'NVDA']},
    "6683": {'name': '雍智科技', 'industry': 'Semiconductor', 'us_chain_tickers': ['NVDA', 'AMD']},
    "6782": {'name': '視陽', 'industry': 'Biotech_Yield', 'us_chain_tickers': ['VTI']},
    "6805": {'name': '富世達', 'industry': 'AI_Hardware', 'us_chain_tickers': ['AAPL']},
    "6826": {'name': '和淞', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMAT', 'ASML']},
    "6919": {'name': '康霈*', 'industry': 'Biotech', 'us_chain_tickers': ['IBB', 'XBI']},
    "6949": {'name': '沛爾生醫-創', 'industry': 'Biotech_Drug', 'us_chain_tickers': ['IBB']},
    "6977": {'name': '聯純', 'industry': 'Materials', 'us_chain_tickers': ['VTI']},
    "7403": {'name': '紐因科技', 'industry': 'Innovation', 'us_chain_tickers': ['VTI']},
    "7408": {'name': '易得雲端', 'industry': 'Innovation', 'us_chain_tickers': ['VTI']},
    "7409": {'name': '美格能', 'industry': 'Innovation', 'us_chain_tickers': ['VTI']},
    "7410": {'name': '萬寶隆', 'industry': 'TIB_Innovation', 'us_chain_tickers': ['VTI']},
    "7411": {'name': '澤聖擺渡', 'industry': 'Innovation', 'us_chain_tickers': ['VTI']},
    "7415": {'name': '元澄半導體', 'industry': 'Innovation', 'us_chain_tickers': ['TSM', 'NVDA']},
    "7416": {'name': '天工精密', 'industry': 'Innovation', 'us_chain_tickers': ['VTI']},
    "7695": {'name': '宏潤生技', 'industry': 'TIB_Innovation', 'us_chain_tickers': ['IBB']},
    "7698": {'name': '天龍材料', 'industry': 'Innovation', 'us_chain_tickers': ['VTI']},
    "7699": {'name': '瑞利光', 'industry': 'Innovation', 'us_chain_tickers': ['VTI']},
    "7769": {'name': '鴻勁精密', 'industry': 'Innovation', 'us_chain_tickers': ['AMAT', 'ASML']},
    "7799": {'name': '禾榮科', 'industry': 'Biotech_Drug', 'us_chain_tickers': ['IBB']},
    "7848": {'name': '騏億鑫', 'industry': 'Emerging', 'us_chain_tickers': ['VTI']},
    "7856": {'name': '漢測', 'industry': 'Emerging', 'us_chain_tickers': ['VTI']},
    "7861": {'name': '貝爾威勒', 'industry': 'Emerging', 'us_chain_tickers': ['VTI']},
    "7869": {'name': '宏于電機', 'industry': 'Innovation', 'us_chain_tickers': ['ABB', 'GE']},
    "7883": {'name': '饗賓', 'industry': 'Retail', 'us_chain_tickers': ['VTI']},
    "8098": {'name': '慶康科技', 'industry': 'Semiconductor', 'us_chain_tickers': ['AMAT', 'TSM']},
    "8210": {'name': '勤誠', 'industry': 'AI_Hardware', 'us_chain_tickers': ['NVDA', 'MSFT', 'SMCI']},
}

# ─────────────────────────────────────────────
# 國際標的清單 (由各個股配置自動彙整)
# ─────────────────────────────────────────────
def _get_international_watchlist():
    tickers = set()
    for cfg in STOCK_CONFIGS.values():
        if "us_chain_tickers" in cfg:
            tickers.update(cfg["us_chain_tickers"])
    tickers.update(["SPY", "QQQ", "SOXX", "DIA", "VTI", "TLT", "UUP", "TSM", "NVDA", "AAPL"])
    return sorted(list(tickers))

INTERNATIONAL_WATCHLIST = _get_international_watchlist()

# ─────────────────────────────────────────────
# 回測 / 訓練相關參數（保留原有值）
# ─────────────────────────────────────────────
LOOKBACK       = 60
HORIZON        = 5
RETRAIN_FREQ   = 21

PARETO_RATIO          = 0.2
CONFIDENCE_THRESHOLD  = 0.75
TIER_1_STOCKS         = ["2330", "2317", "2454", "2382", "2881", "2412", "2308", "2882"]

REGIME_CONFIG = {
    "vol_low":    0.20,
    "vol_high":   0.40,
    "train_split": 0.30,
    "oos_window": 252 * 2,
}

SYSTEM_STABILITY_CONFIG = {
    "inference_timeout":  45,
    "max_prob_threshold": 0.99,
    "min_prob_threshold": 0.01,
    "max_staleness_days": 3,
    "fallback_prob":      0.5,
}

WF_CONFIG = {
    "train_window": 252 * 3,
    "val_window":   126,
    "embargo_days": 45,
    "step_days":    RETRAIN_FREQ,
    "test_window":  126,
}

TRAIN_START_DATE = "2010-01-01"

EVAL_TARGETS = {
    "directional_accuracy": 0.65,
    "ic":                   0.05,
    "sharpe":               1.0,
}

PORTFOLIO_EVAL_TARGETS = {
    "portfolio_sharpe":  1.2,
    "max_drawdown":     -0.15,
    "calmar_ratio":      2.0,
    "beta_to_taiex":     0.5,
    "turnover_rate":     2.0,
    "worst_month_ret":  -0.08,
}

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

TFT_PARAMS_CPU_OVERRIDE = {
    "hidden_size":           32,
    "lstm_layers":           1,
    "attention_head_size":   1,
    "patience":              5,
    "max_epochs":            30,
}

XGB_PARAMS = {
    "n_estimators":          1000,
    "learning_rate":         0.02,
    "max_depth":             6,
    "min_child_weight":      5,
    "gamma":                 0.1,
    "subsample":             0.8,
    "colsample_bytree":      0.7,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "random_state":          42,
    "n_jobs":                -1,
    "early_stopping_rounds": 50,
}

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
# 特徵分組
# ─────────────────────────────────────────────
FEATURE_GROUPS: dict = {
    "price_volume": [
        "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
        "realized_vol_10d", "realized_vol_20d", "realized_vol_60d",
        "vol_ratio_20_60", "atr_14d", "atr_ratio",
        "ma5", "ma20", "ma60", "ma_cross_5_20", "ma_cross_20_60",
        "price_vs_ma20", "price_vs_ma60",
        "bb_width_20d", "bb_position_20d",
        "rsi_14d", "rsi_28d",
        "volume_ma_ratio_5d", "volume_ma_ratio_20d",
        "price_range_ratio_5d", "price_range_ratio_20d",
        "momentum_20d", "momentum_60d",
    ],
    "chip": [
        "foreign_net_ratio_5d", "foreign_net_ratio_20d",
        "investment_trust_net_ratio_5d",
        "dealer_net_ratio_5d",
        "three_inst_net_ratio_5d",
        "margin_balance_ratio", "short_balance_ratio",
        "margin_chg_rate_5d", "margin_chg_rate_20d",
        "short_chg_rate_5d",
        "foreign_holding_ratio", "foreign_holding_chg_20d",
    ],
    "fundamental": [
        "per", "per_pct_rank_252",
        "pbr", "pbr_pct_rank_252",
        "dividend_yield", "dy_pct_rank_252",
        "per_deviation_from_ma",
    ],
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
        # [P1 修復] kwave_score 不再是「全系統使用、無人定義」的幽靈特徵
        # 在 feature_engineering.add_kwave_regime_features 中明確計算（含 fallback）
        "kwave_score",
    ],
    "event": [
        "cash_dividend_ttm",
        "days_since_last_earnings",
    ],
    "rolling_stats": [
        "skew_20d", "skew_60d",
        "kurt_20d", "kurt_60d",
        "autocorr_lag1_20d",
        "sharpe_20d", "sharpe_60d",
    ],
    "futures_chip": [
        "tx_oi_chg_1d", "tx_oi_chg_5d",
        "tx_basis", "tx_basis_5d_chg",
        "tx_vol_ma_ratio",
        "tfo_pcr_volume", "tfo_pcr_oi",
        "tx_oi_direction_5d",
    ],
    "medium_term": [
        "rev_yoy_positive_months", "rev_yoy_3m",
        "gross_margin_qoq", "gross_margin_qoq_dir",
        "eps_accel_proxy",
        "foreign_net_weekly", "foreign_net_accel",
        "margin_chg_rate_5d", "margin_chg_rate_20d",
        "short_chg_rate_5d",
        "rs_line_20d", "rs_line_slope_5d",
        "adr_premium", "adr_premium_5d_chg", "adr_premium_ma5",
    ],
    "us_chain": [],
    "physics_signals": [
        "gravity_pull",
        "info_force_per_mass", 
        "singularity_dist",
        "market_entropy",
        "liquidity_quality",
        "smart_money_sync_buy",
        "kwave_score"
    ],
}

def get_all_features(stock_id: str = DEFAULT_STOCK_ID) -> list[str]:
    config = STOCK_CONFIGS.get(stock_id, STOCK_CONFIGS[DEFAULT_STOCK_ID])
    groups = FEATURE_GROUPS.copy()
    us_tickers = [t.lower().replace("^", "") for t in config["us_chain_tickers"]]
    us_chain_features = []
    if config.get("use_adr_premium", False):
        us_chain_features += ["tsm_premium", "tsm_premium_ma5"]
    for ticker in us_tickers:
        us_chain_features += [f"{ticker}_ret_1d", f"{ticker}_ret_5d", f"{ticker}_ret_20d"]
    groups["us_chain"] = us_chain_features
    all_f = [f for grp in groups.values() for f in grp]
    # 使用 dict.fromkeys 移除重複項並保留順序
    return list(dict.fromkeys(all_f))

ALL_FEATURES = get_all_features(DEFAULT_STOCK_ID)
