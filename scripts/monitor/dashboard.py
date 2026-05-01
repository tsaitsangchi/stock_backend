import sys
from pathlib import Path

# Setup paths - dashboard is in scripts/monitor/
current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir.parent
project_root = scripts_dir.parent

for sub in ["fetchers", "pipeline", "training", "monitor"]:
    sys.path.append(str(scripts_dir / sub))
sys.path.append(str(scripts_dir))
sys.path.append(str(project_root))
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
import json
import subprocess
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# 注入路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import STOCK_CONFIGS, MODEL_DIR
from data_pipeline import _query, load_exchange_rate
from data_integrity_audit import IntegrityAuditor
from portfolio_optimizer import PortfolioOptimizer

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

@st.cache_data(ttl=300)
def load_investment_advice(capital: float):
    """產出基於 v4.0 核心思想的投資建議矩陣。"""
    # 1. 取得最新訊號
    sql = """
        SELECT s.stock_id, s.decision, s.overall_score, s.prob_up, 
               s.boosting_reasons, s.blocking_reasons, p.close as current_price
        FROM signal_history s
        JOIN (
            SELECT stock_id, close, date 
            FROM stock_price 
            WHERE date = (SELECT MAX(date) FROM stock_price)
        ) p ON s.stock_id = p.stock_id
        WHERE s.date = (SELECT MAX(date) FROM signal_history)
    """
    df = _query(sql)
    if df.empty: return pd.DataFrame()

    # 2. 透過優化器計算權重
    optimizer = PortfolioOptimizer()
    # 模擬訊號格式給優化器
    signals = {}
    for _, row in df.iterrows():
        signals[row['stock_id']] = {
            'decision': row['decision'],
            'prob_up': float(row['prob_up']),
            'overall_score': float(row['overall_score'])
        }
    
    # 取得美元匯率
    usd_rate = load_exchange_rate()
    weights = optimizer.optimize_v4(signals, exchange_rate=usd_rate)
    
    # 3. 整合結果
    results = []
    for _, row in df.iterrows():
        sid = str(row['stock_id']).strip()
        w = weights.get(sid, 0)
        
        amount = capital * w
        shares = int(amount / row['current_price']) if row['current_price'] > 0 else 0
        
        # 信任度評級
        score = float(row['overall_score'])
        if score > 0.8: rating = "💎 極高 (物理共振)"
        elif score > 0.6: rating = "🟢 高 (結構溢價)"
        else: rating = "🟡 中 (觀察中)"
        
        stock_name = STOCK_CONFIGS.get(sid, {}).get('name', '未知標的')
        
        # 物理分析說明：若權重為 0，則顯示攔截原因
        if w > 0:
            analysis = f"✅ 建議配置\n核心優勢：{row['boosting_reasons']}"
        else:
            analysis = f"🛑 攔截原因：{row['blocking_reasons']}"

        results.append({
            "標的": f"{sid} {stock_name}",
            "機率": f"{float(row['prob_up']):.1%}",
            "建議權重": f"{w:.1%}",
            "預計金額": f"{amount:,.0f} TWD",
            "建議股數": f"{shares:,} 股",
            "信任度": rating,
            "物理分析/建議": analysis
        })
    
    # 加入現金部位
    cash_w = weights.get('CASH', 0)
    results.append({
        "標的": "💰 現金 (防禦端)",
        "機率": "-",
        "建議權重": f"{cash_w:.1%}",
        "預計金額": f"{capital * cash_w:,.0f} TWD",
        "建議股數": "-",
        "信任度": "🛡️ 槓鈴策略守護",
        "物理分析/建議": "80/20 規則：在市場熵值過高時保護本金。"
    })
    
    return pd.DataFrame(results)

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
st.sidebar.subheader("🚀 手動任務控制")
def run_manual_task(task_key):
    log_path = scripts_dir / "outputs" / "manual_action.log"
    cmd = [f"{scripts_dir.parent}/venv/bin/python3", str(scripts_dir / "monitor" / "action_runner.py"), task_key]
    with open(log_path, "a") as f:
        f.write(f"\n\n--- [{datetime.now()}] Triggered {task_key} via Dashboard ---\n")
    subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT)
    st.sidebar.success(f"已啟動 {task_key} 任務，請至 Log 查看。")

if st.sidebar.button("全系統資料抓取並審計"):
    run_manual_task("data")

if st.sidebar.button("全系統模型運算並審計"):
    run_manual_task("model")

if st.sidebar.button("全系統預測運算並審計"):
    run_manual_task("predict")

st.sidebar.markdown("---")
st.sidebar.subheader("投資設定")
capital = st.sidebar.number_input("投資本金 (TWD)", value=100000, step=10000)

st.sidebar.markdown("---")
st.sidebar.subheader("系統狀態")
fresh_df = load_integrity_matrix()
# 從第一欄或已知主表獲取日期 (這裏簡化處理)
price_date = _query("SELECT MAX(date) FROM stock_price").iloc[0,0]
st.sidebar.metric("市場資料日期", str(price_date) if price_date else "N/A")

# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 背景進程監控 (Managers Heartbeat)
# ─────────────────────────────────────────────
# 1. 自動訓練心跳
train_heartbeat_path = scripts_dir / "training" / "outputs" / "auto_train.heartbeat"
train_hb = {"status": "🔴 停止", "progress": "N/A", "running": [], "ts": "N/A"}
if train_heartbeat_path.exists():
    try:
        data = json.loads(train_heartbeat_path.read_text())
        diff = (datetime.now() - datetime.fromisoformat(data["ts"])).total_seconds()
        if diff < 120:
            train_hb["status"] = "🟢 運行中"
            train_hb["progress"] = data.get("progress", "N/A")
            train_hb["running"]  = data.get("running_ids", [])
            train_hb["ts"] = data["ts"]
    except: pass

# 2. 自動預測心跳
predict_heartbeat_path = scripts_dir / "training" / "outputs" / "auto_predict.heartbeat"
predict_hb = {"status": "🔴 停止", "processed": 0, "ts": "N/A"}
if predict_heartbeat_path.exists():
    try:
        data = json.loads(predict_heartbeat_path.read_text())
        diff = (datetime.now() - datetime.fromisoformat(data["ts"])).total_seconds()
        if diff < 120:
            predict_hb["status"] = "🟢 運行中"
            predict_hb["processed"] = data.get("processed_count", 0)
            predict_hb["ts"] = data["ts"]
    except: pass

st.title("量子藍圖 — 系統監控儀表板")

st.subheader("⚡ 背景服務狀態 (Background Services)")
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric("自動訓練管理器", train_hb["status"], f"進度: {train_hb['progress']}")
with m_col2:
    st.metric("自動預測管理器", predict_hb["status"], f"今日已推論: {predict_hb['processed']} 檔")
with m_col3:
    st.metric("執行中任務", f"{len(train_hb['running'])} 檔", f"{', '.join(train_hb['running'])}")

st.markdown("---")

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
    trained_ratio = (model_df["status"].str.contains("OK")).sum() / len(model_df)
    st.metric("2. 模型訓練完整度", f"{trained_ratio:.1%}", 
              delta=f"已就緒 {(model_df['status'].str.contains('OK')).sum()} 檔")

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
tab1, tab2 = st.tabs(["🛡️ 全系統健康度矩陣", "🎯 全系統投資建議矩陣"])

with tab1:
    st.subheader("全系統健康度矩陣")
    # 準備矩陣數據
    # 1. 資料狀態：從 fresh_df 計算平均，若 > 95% 為綠燈
    def get_data_status(row):
        # 排除 stock_id 欄位，計算其餘欄位的平均覆蓋率
        vals = row.iloc[1:].str.rstrip('%').astype(float)
        avg = vals.mean()
        # 關鍵指標：若 stock_price 為 100%，則至少為黃燈以上
        price_col = "stock_price" if "stock_price" in vals.index else None
        is_price_ok = (vals[price_col] >= 100) if price_col else False
        
        if avg > 90 and is_price_ok: return "🟢 完整"
        if avg > 70 and is_price_ok: return "🟡 部分"
        return "🔴 缺漏"

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
        height=400
    )

with tab2:
    st.subheader("全系統投資建議矩陣 (100k Barbell Strategy)")
    advice_df = load_investment_advice(capital)
    if not advice_df.empty:
        st.dataframe(
            advice_df,
            use_container_width=True,
            height=600,
            column_config={
                "標的": st.column_config.TextColumn("標的 (名稱)", width="medium"),
                "物理分析/建議": st.column_config.TextColumn("物理分析/建議 (Core v4.0)", width="large"),
                "預計金額": st.column_config.TextColumn("預計金額", width="small"),
                "建議股數": st.column_config.TextColumn("建議股數", width="small"),
            }
        )
        st.success(f"💡 目前配置建議：基於 {capital:,.0f} TWD 本金，系統採非對稱槓鈴策略，防禦端佔 {advice_df.iloc[-1]['建議權重']}。")
    else:
        st.warning("目前尚無有效推論資料，請等待重訓完成。")

st.markdown("---")

# 第三排：Log 導航員 (P2-2 修正)
st.subheader("📋 系統日誌導航 (Log Navigator)")

log_col1, log_col2 = st.columns([1, 3])

with log_col1:
    log_files = {
        "自動訓練管理器": scripts_dir / "training" / "outputs" / "manager.log",
        "自動預測管理器": scripts_dir / "training" / "outputs" / "predict_manager.log",
        "手動任務執行日誌": scripts_dir / "outputs" / "manual_action.log",
        "數據完整度審計": scripts_dir / "outputs" / "audit_data_integrity.log",
        "回測引擎審計": scripts_dir / "outputs" / "audit_backtest.log",
    }
    
    # 加入所有標的的訓練日誌
    for sid in STOCK_CONFIGS.keys():
        f = scripts_dir / "training" / "outputs" / f"train_{sid}.log"
        if f.exists():
            log_files[f"模型訓練: {sid}"] = f

    selected_log_name = st.selectbox("選擇日誌檔案", list(log_files.keys()))
    num_lines = st.slider("顯示行數", 10, 500, 100)
    refresh_log = st.button("🔄 刷新日誌")

with log_col2:
    selected_file = log_files[selected_log_name]
    if selected_file.exists():
        with open(selected_file, "r") as f:
            lines = f.readlines()
            # 顯示最後 N 行
            tail_lines = "".join(lines[-num_lines:])
            st.code(tail_lines, language="text")
            st.caption(f"檔案路徑: {selected_file}")
    else:
        st.warning(f"找不到檔案: {selected_file}")

st.markdown("---")
st.caption("Quantum Blueprint Quant System Dashboard (Simplified) © 2026 Antigravity Research")

st.caption("Quantum Blueprint Quant System Dashboard © 2026 Antigravity Research")
