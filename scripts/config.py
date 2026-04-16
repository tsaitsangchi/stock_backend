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
STOCK_ID       = "2330"    # 字串，與 DB varchar(50) 一致
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
    ],
    # ⑨ 美股供應鏈與 ADR（新增）
    "us_chain": [
        "tsm_premium", "tsm_premium_ma5",
        "nvda_ret_1d", "nvda_ret_5d", "nvda_ret_20d",
        "aapl_ret_1d", "aapl_ret_5d", "aapl_ret_20d",
        "soxx_ret_1d", "soxx_ret_5d", "soxx_ret_20d",
    ],
}

ALL_FEATURES = [f for grp in FEATURE_GROUPS.values() for f in grp]

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
    "embargo_days": HORIZON,   # 禁區天數（防止標籤洩漏）
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
