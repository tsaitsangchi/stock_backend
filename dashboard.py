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
def load_data_freshness():
    tables = ["stock_price", "stock_per", "institutional_investors_buy_sell", 
              "margin_purchase_short_sale", "shareholding", "stock_forecast_daily"]
    results = []
    for table in tables:
        try:
            df = _query(f"SELECT MAX(date) as last_date FROM {table}")
            last_date = df["last_date"].iloc[0] if not df.empty else None
            results.append({"Table": table, "Last Update": last_date})
        except:
            results.append({"Table": table, "Last Update": None})
    return pd.DataFrame(results)

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
        SELECT stock_id, prob_up, ensemble_price, current_close, confidence_level
        FROM stock_forecast_daily
        WHERE predict_date = (SELECT * FROM latest_date)
          AND day_offset = 30
    """
    df = _query(sql)
    if not df.empty:
        df["Expected Return"] = (df["ensemble_price"] / df["current_close"] - 1) * 100
    return df

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
fresh_df = load_data_freshness()
price_date = fresh_df[fresh_df["Table"]=="stock_price"]["Last Update"].iloc[0]
st.sidebar.metric("市場資料日期", str(price_date) if price_date else "N/A")

# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────

st.title("量子藍圖 — 系統監控儀表板")

# 第一排：核心指標
col1, col2, col3, col4 = st.columns(4)

with col1:
    model_df = load_model_status()
    ok_count = (model_df["status"] == "🟢 OK").sum()
    st.metric("模型就緒率", f"{ok_count}/{len(model_df)}", delta=None)

with col2:
    perf_df = load_performance_da()
    avg_da = perf_df["da"].mean()
    st.metric("平均 30D 準確率 (DA)", f"{avg_da:.1%}", delta=f"{avg_da-0.5:.1%}" if avg_da else None)

with col3:
    drift_df = load_psi_drift()
    warning_drift = (drift_df["psi"] > 0.1).sum()
    st.metric("分佈漂移警報", f"{warning_drift} 檔", delta=f"-{warning_drift}" if warning_drift > 0 else "0", delta_color="inverse")

with col4:
    pred_df = load_today_predictions()
    bull_count = (pred_df["prob_up"] > 0.5).sum() if not pred_df.empty else 0
    st.metric("今日多頭佔比", f"{bull_count}/{len(pred_df)}" if not pred_df.empty else "N/A")

st.markdown("---")

# 第二排：即時監控細節
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📊 模型健康度報告 (Performance & Drift)")
    # 合併效能與漂移數據
    health_report = pd.merge(perf_df, drift_df, on="stock_id")
    health_report = pd.merge(health_report, model_df[["stock_id", "status", "age_days"]], on="stock_id")
    
    # 重新排列欄位
    health_report = health_report[["stock_id", "status", "da", "psi", "perf_status", "drift_status", "age_days"]]
    
    # 樣式處理
    def color_status(val):
        if "🟢" in str(val) or "STABLE" in str(val) or "EXCELLENT" in str(val): color = '#2ea043'
        elif "🟡" in str(val) or "WARNING" in str(val): color = '#d29922'
        elif "🔴" in str(val) or "DEGRADED" in str(val) or "DRIFTED" in str(val): color = '#f85149'
        else: color = 'white'
        return f'color: {color}'

    st.dataframe(
        health_report.style.map(color_status, subset=['status', 'perf_status', 'drift_status'])
                           .background_gradient(subset=['da'], cmap="RdYlGn", vmin=0.45, vmax=0.65),
        use_container_width=True,
        height=400
    )

with right_col:
    st.subheader("⏳ 數據流鮮度")
    st.table(fresh_df)
    
    st.subheader("⚠️ 需注意清單")
    critical = health_report[(health_report["da"] < 0.52) | (health_report["psi"] > 0.15) | (health_report["status"] != "🟢 OK")]
    if not critical.empty:
        for _, row in critical.iterrows():
            st.warning(f"**{row['stock_id']}**: DA={row['da']:.1%}, PSI={row['psi']:.2f}, Status={row['status']}")
    else:
        st.success("目前無異常標的")

st.markdown("---")

# 第三排：預測分佈分析
st.subheader("📈 今日預測分佈分析 (Inference Distribution)")
if not pred_df.empty:
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        fig = px.histogram(pred_df, x="prob_up", nbins=20, 
                           title="上漲機率分佈 (Prob_up Distribution)",
                           color_discrete_sequence=['#58a6ff'])
        fig.add_vline(x=0.5, line_dash="dash", line_color="red")
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    with dist_col2:
        top_picks = pred_df.sort_values("prob_up", ascending=False).head(10)
        top_picks["Name"] = top_picks["stock_id"].map(lambda x: STOCK_CONFIGS.get(x, {}).get("name", "Unknown"))
        fig_bar = px.bar(top_picks, x="prob_up", y="Name", orientation='h',
                         title="今日高信心標的 (Top Prob_up)",
                         color="prob_up", color_continuous_scale="RdYlGn")
        fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("今日尚未生成推論數據。")

st.caption("Quantum Blueprint Quant System Dashboard © 2026 Antigravity Research")
