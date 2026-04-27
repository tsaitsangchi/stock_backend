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
# 資料可用性時間表 (Data Availability Calendar)
# ─────────────────────────────────────────────
DATA_LAG_CONFIG = {
    "month_revenue":        40,
    "financial_statements": 45,
    "annual_report":        90,
    "institutional_chip":   1,
}

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
    "2330": {
        "name": "台積電",
        "industry": "Semiconductor",
        "us_chain_tickers": ["TSM", "NVDA", "AAPL", "SOXX"],
        "vol_low": 0.20, "vol_high": 0.40, "use_adr_premium": True,
    },
    "2317": {
        "name": "鴻海",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["AAPL", "HPE", "MSFT"],
        "vol_low": 0.25, "vol_high": 0.45, "use_adr_premium": False,
    },
    "2454": {
        "name": "聯發科",
        "industry": "Semiconductor",
        "us_chain_tickers": ["QCOM", "ARM", "SOXX", "NVDA"],
        "vol_low": 0.30, "vol_high": 0.50, "use_adr_premium": False,
    },
    "2881": {
        "name": "富邦金",
        "industry": "Finance",
        "us_chain_tickers": ["XLF", "KBE", "TNX", "VTI"],
        "vol_low": 0.12, "vol_high": 0.25, "use_adr_premium": False,
    },
    "2382": {
        "name": "廣達",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["NVDA", "MSFT", "GOOGL", "AMZN", "SMCI", "SOXX"],
        "vol_low": 0.25, "vol_high": 0.50, "use_adr_premium": False,
    },
    "1301": {
        "name": "台塑",
        "industry": "Materials",
        "us_chain_tickers": ["DOW", "LYB", "XOM", "CVX"],
        "vol_low": 0.15, "vol_high": 0.35, "use_adr_premium": False,
    },
    "2002": {
        "name": "中鋼",
        "industry": "Materials",
        "us_chain_tickers": ["X", "NUE", "STLD", "MT"],
        "vol_low": 0.10, "vol_high": 0.30, "use_adr_premium": False,
    },
    "2603": {
        "name": "長榮",
        "industry": "Shipping",
        "us_chain_tickers": ["ZIM", "MATX", "SEA", "BDRY"],
        "vol_low": 0.30, "vol_high": 0.60, "use_adr_premium": False,
    },
    "3037": {
        "name": "欣興",
        "industry": "Semiconductor",
        "us_chain_tickers": ["NVDA", "AMD", "INTC", "SOXX"],
        "vol_low": 0.25, "vol_high": 0.50, "use_adr_premium": False,
    },
    "3324": {
        "name": "雙鴻",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["NVDA", "VRT", "AMD"],
        "vol_low": 0.35, "vol_high": 0.70, "use_adr_premium": False,
    },
    "1513": {
        "name": "中興電",
        "industry": "Energy",
        "us_chain_tickers": ["ETN", "PWR", "GE"],
        "vol_low": 0.25, "vol_high": 0.50, "use_adr_premium": False,
    },
    "3008": {
        "name": "大立光",
        "industry": "Semiconductor",
        "us_chain_tickers": ["AAPL", "LITE"],
        "vol_low": 0.20, "vol_high": 0.40, "use_adr_premium": False,
    },
    "2308": {
        "name": "台達電",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["TSLA", "NVDA", "ENPH", "SOXX"],
        "vol_low": 0.20, "vol_high": 0.40, "use_adr_premium": False,
    },
    "6669": {
        "name": "緯穎",
        "industry": "AI_Hardware",
        "us_chain_tickers": ["MSFT", "META", "AMZN", "NVDA"],
        "vol_low": 0.30, "vol_high": 0.60, "use_adr_premium": False,
    },
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
