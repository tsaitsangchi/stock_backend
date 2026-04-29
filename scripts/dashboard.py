"""
scripts/dashboard.py
量子藍圖 — 系統監控儀表板 v1.0

使用方式：
    streamlit run scripts/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# 注入路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import STOCK_CONFIGS, MODEL_DIR
from data_pipeline import _query
from data_integrity_audit import IntegrityAuditor

# 頁面配置
st.set_page_config(
    page_title="量子藍圖 — 系統監控",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 套用 Premium 風格 CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 資料載入邏輯 (Cached)
# ─────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_integrity_matrix():
    auditor = IntegrityAuditor(days_window=60)
    return auditor.audit_coverage_matrix()

@st.cache_data(ttl=600)
def load_lag_report():
    auditor = IntegrityAuditor()
    return auditor.audit_announcement_lag()

@st.cache_data(ttl=600)
def load_model_status():
    from model_health_check import check_model_files_df
    return check_model_files_df(list(STOCK_CONFIGS.keys()))

@st.cache_data(ttl=600)
def load_performance_da():
    from model_health_check import evaluate_recent_performance_df
    return evaluate_recent_performance_df(list(STOCK_CONFIGS.keys()))

@st.cache_data(ttl=600)
def load_psi_drift():
    from model_health_check import check_prediction_drift_df
    return check_prediction_drift_df(list(STOCK_CONFIGS.keys()))

@st.cache_data(ttl=300)
def load_today_predictions():
    sql = """
        WITH latest_date AS (SELECT MAX(predict_date) FROM stock_forecast_daily)
        SELECT stock_id, prob_up, ensemble_price, current_close, confidence_level, warning_flag
        FROM stock_forecast_daily
        WHERE predict_date = (SELECT * FROM latest_date)
          AND day_offset = 30
    """
    df = _query(sql)
    if not df.empty:
        df["Expected Return"] = (df["ensemble_price"] / df["current_close"] - 1) * 100
    return df

@st.cache_data(ttl=60)
def load_training_failures():
    """模擬讀取管理器失敗日誌。"""
    try:
        from auto_train_manager import FAILURE_TRACKER
        return FAILURE_TRACKER
    except:
        return {}

# ─────────────────────────────────────────────
# Sidebar 側欄
# ─────────────────────────────────────────────

st.sidebar.title("🛡️ Antigravity Quant")
st.sidebar.info("核心模型監控系統 v1.0")
if st.sidebar.button("🔄 重新整理數據"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("系統狀態")
fresh_df = load_integrity_matrix()
# 從第一欄或已知主表獲取日期 (這裏簡化處理)
price_date = _query("SELECT MAX(date) FROM stock_price").iloc[0,0]
st.sidebar.metric("市場資料日期", str(price_date) if price_date else "N/A")

# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────

st.title("量子藍圖 — 系統監控儀表板")

# 第一排：系統三位一體指標 (Trinity Integrity Metrics)
col1, col2, col3, col4 = st.columns(4)

with col1:
    # 1. 資料完整度 (Data Integrity)
    avg_integrity = fresh_df.iloc[:, 1:].apply(lambda x: x.str.rstrip('%').astype(float)).mean().mean() / 100
    st.metric("1. 資料完整度", f"{avg_integrity:.1%}", 
              delta="核心資料庫" if avg_integrity > 0.9 else "需補件",
              delta_color="normal" if avg_integrity > 0.9 else "inverse")

with col2:
    # 2. 模型訓練完整度 (Model Training)
    model_df = load_model_status()
    trained_ratio = (model_df["status"] == "🟢 OK").sum() / len(model_df)
    st.metric("2. 模型訓練完整度", f"{trained_ratio:.1%}", 
              delta=f"已就緒 {(model_df['status'] == '🟢 OK').sum()} 檔")

with col3:
    # 3. 預測完整度 (Prediction)
    pred_df = load_today_predictions()
    pred_count = len(pred_df) if not pred_df.empty else 0
    total_stocks = len(STOCK_CONFIGS)
    pred_ratio = pred_count / total_stocks
    st.metric("3. 預測完整度", f"{pred_ratio:.1%}", 
              delta=f"今日生成 {pred_count} 檔")

with col4:
    # 4. 系統綜合效能 (Accuracy/Drift)
    perf_df = load_performance_da()
    drift_df = load_psi_drift() # 恢復漂移數據讀取
    avg_da = perf_df["da"].mean()
    st.metric("平均 30D 準確率 (DA)", f"{avg_da:.1%}", 
              delta=f"{avg_da-0.5:+.1%}" if avg_da else None)

st.markdown("---")

# 第二排：核心狀態矩陣 (Trinity Status Matrix)
st.subheader("🛡️ 全系統健康度矩陣")

# 準備矩陣數據
# 1. 資料狀態：從 fresh_df 計算平均，若 > 95% 為綠燈
def get_data_status(row):
    avg = row.iloc[1:].str.rstrip('%').astype(float).mean()
    return "🟢 完整" if avg > 95 else "🟡 部分" if avg > 80 else "🔴 缺漏"

matrix_df = pd.DataFrame({"stock_id": list(STOCK_CONFIGS.keys())})
matrix_df["資料狀態"] = fresh_df.apply(get_data_status, axis=1)

# 2. 模型狀態
matrix_df = pd.merge(matrix_df, model_df[["stock_id", "status"]], on="stock_id")
matrix_df.rename(columns={"status": "模型狀態"}, inplace=True)

# 3. 預測狀態
if not pred_df.empty:
    pred_ids = set(pred_df["stock_id"].tolist())
    matrix_df["預測狀態"] = matrix_df["stock_id"].apply(lambda x: "🟢 已產出" if x in pred_ids else "⚪ 待處理")
else:
    matrix_df["預測狀態"] = "⚪ 待處理"

# 樣式與顯示
def style_trinity(val):
    if "🟢" in str(val): color = '#2ea043'
    elif "🟡" in str(val): color = '#d29922'
    elif "🔴" in str(val): color = '#f85149'
    else: color = '#8b949e'
    return f'color: {color}'

st.dataframe(
    matrix_df.style.map(style_trinity),
    use_container_width=True,
    height=500
)

st.markdown("---")
st.caption("Quantum Blueprint Quant System Dashboard (Simplified) © 2026 Antigravity Research")

st.caption("Quantum Blueprint Quant System Dashboard © 2026 Antigravity Research")
