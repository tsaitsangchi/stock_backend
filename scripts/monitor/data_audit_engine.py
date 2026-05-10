"""
data_audit_engine.py v5.5.26 (Trinity Core Final)
================================================================================
數據完整度稽核引擎 — 戰情看板核心
負責稽核 150 檔核心標的的資料完整度，並產出動態監控網頁 (dashboard.html)。

修訂歷程：
  v5.5.26 (2026-05-10):
    - [核心] 實作多維度數據稽核 (價格、籌碼、指標)。
    - [視覺] 產出極致美學的動態 HTML 戰情室。

【執行範例說明】

1. 直接從命令行執行（更新戰情網頁）：
   $ python scripts/monitor/data_audit_engine.py

2. 日誌查閱 (確認稽核任務狀態)：
   SELECT * FROM pipeline_execution_log WHERE task_name = 'dashboard_generation' ORDER BY created_at DESC LIMIT 5;
"""

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
for _sub in ("scripts", "scripts/core"):
    _p = _PROJECT_ROOT / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from core.db_utils import db_session, get_db_stock_ids, write_pipeline_log, db_transaction
except ImportError as e:
    print(f"[FATAL] 無法匯入核心組件: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def save_audit_to_db(data):
    """將稽核結果持久化至 data_audit_log"""
    with db_transaction() as cur:
        for d in data:
            cur.execute('''
                INSERT INTO data_audit_log (stock_id, missing_days, health_score, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (stock_id) DO UPDATE 
                SET missing_days = EXCLUDED.missing_days,
                    health_score = EXCLUDED.health_score,
                    status = EXCLUDED.status,
                    last_checked_at = CURRENT_TIMESTAMP
            ''', (d['id'], 0, d['score'], d['status']))

def audit_completeness():
    logger.info("📡 [Monitor] 正在啟動全核心個股資料完整度稽核...")
    stocks = get_db_stock_ids()
    report_data = []
    
    with db_session() as conn:
        with conn.cursor() as cur:
            for sid in stocks[:150]:
                cur.execute("""
                    SELECT 
                        COALESCE(i.stock_name, '未知') as name,
                        COUNT(p.*) as count,
                        (SELECT close FROM stock_price WHERE stock_id = %s ORDER BY date DESC LIMIT 1) as price,
                        (SELECT sharpe_ratio FROM evaluation_log WHERE stock_id = %s ORDER BY created_at DESC LIMIT 1) as sharpe,
                        (SELECT win_rate FROM evaluation_log WHERE stock_id = %s ORDER BY created_at DESC LIMIT 1) as win_rate,
                        (SELECT signal FROM predictions WHERE stock_id = %s ORDER BY created_at DESC LIMIT 1) as signal,
                        (SELECT confidence FROM predictions WHERE stock_id = %s ORDER BY created_at DESC LIMIT 1) as confidence
                    FROM (SELECT %s as sid) s
                    LEFT JOIN stock_info i ON s.sid = i.stock_id
                    LEFT JOIN stock_price p ON s.sid = p.stock_id
                    GROUP BY i.stock_name
                """, (sid, sid, sid, sid, sid, sid))
                res = cur.fetchone()
                
                price_count = res['count']
                stock_name = res['name']
                curr_price = float(res['price']) if res['price'] is not None else 0.0
                sharpe = float(res['sharpe']) if res['sharpe'] is not None else 0.0
                win_rate = float(res['win_rate']) if res['win_rate'] is not None else 0.0
                signal = res['signal'] or "HOLD"
                confidence = float(res['confidence']) if res['confidence'] is not None else 0.0
                
                # 計算分數
                score = min(100, round((price_count / 1000) * 100, 1))
                status = "Healthy" if score > 90 else "Warning" if score > 50 else "Critical"
                
                report_data.append({
                    "id": sid,
                    "name": stock_name,
                    "score": score,
                    "status": status,
                    "price": curr_price,
                    "sharpe": sharpe,
                    "win_rate": win_rate,
                    "signal": signal,
                    "conf": confidence,
                    "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
        
        save_audit_to_db(report_data)

    generate_html(report_data)
    generate_simulator_html(report_data)
    write_pipeline_log("dashboard_generation", "SYSTEM", "success", "sys")

def generate_simulator_html(data):
    # 此處邏輯為讀取 template 並取代 placeholder
    path = _PROJECT_ROOT / "monitor" / "portfolio_simulator.html"
    if not path.exists(): return
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    json_str = json.dumps(data)
    new_content = content.replace("const rawData = []; // [PLACEHOLDER_DATA]", f"const rawData = {json_str};")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    logger.info(f"🏆 [Monitor] 投資模擬器數據已同步: {path}")

def generate_html(data):
    html_path = _PROJECT_ROOT / "monitor" / "dashboard.html"
    html_path.parent.mkdir(exist_ok=True)
    
    # 統計摘要
    total = len(data)
    healthy = sum(1 for d in data if d['status'] == 'Healthy')
    warning = sum(1 for d in data if d['status'] == 'Warning')
    critical = sum(1 for d in data if d['status'] == 'Critical')

    json_data = json.dumps(data)
    
    html_template = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>Trinity Core - 數據戰情監控系統</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0a0e14;
            --card-bg: rgba(255, 255, 255, 0.03);
            --border: rgba(255, 255, 255, 0.1);
            --accent: #00f2ff;
            --healthy: #00ff88;
            --warning: #ffcc00;
            --critical: #ff4466;
        }}
        body {{
            background: var(--bg);
            color: #fff;
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 40px;
            overflow-x: hidden;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            text-align: center;
        }}
        .stat-value {{ font-size: 2.5rem; font-weight: 600; margin-bottom: 5px; }}
        .stat-label {{ color: rgba(255,255,255,0.5); font-size: 0.9rem; }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 15px;
        }}
        .stock-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        .stock-card:hover {{ transform: translateY(-5px); border-color: var(--accent); background: rgba(0, 242, 255, 0.05); }}
        .stock-id {{ font-weight: 600; font-size: 1.1rem; color: var(--accent); }}
        .stock-name {{ font-size: 0.85rem; opacity: 0.8; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .stock-score {{ font-size: 0.8rem; margin-top: 8px; font-family: monospace; }}
        .status-dot {{
            width: 8px; height: 8px; border-radius: 50%;
            display: inline-block; margin-right: 5px;
        }}
        .Healthy {{ color: var(--healthy); }}
        .Healthy .status-dot {{ background: var(--healthy); box-shadow: 0 0 10px var(--healthy); }}
        .Warning {{ color: var(--warning); }}
        .Warning .status-dot {{ background: var(--warning); }}
        .Critical {{ color: var(--critical); }}
        .Critical .status-dot {{ background: var(--critical); }}
        
        @keyframes fadeIn {{ from {{ opacity: 0; transform: scale(0.9); }} to {{ opacity: 1; transform: scale(1); }} }}
        .stock-card {{ animation: fadeIn 0.5s ease forwards; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size: 2rem; letter-spacing: 2px;">TRINITY <span style="color:var(--accent)">CORE</span></h1>
            <p style="margin:5px 0 0; opacity:0.6;">數據完整度實時監控系統 v5.5.26</p>
        </div>
        <div style="text-align:right">
            <div id="clock" style="font-size:1.2rem; font-weight:600;"></div>
            <div style="opacity:0.5; font-size:0.8rem;">最後更新: {datetime.now().strftime("%H:%M:%S")}</div>
        </div>
    </div>

    <div class="summary">
        <div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">監控總數</div></div>
        <div class="stat-card" style="color:var(--healthy)"><div class="stat-value">{healthy}</div><div class="stat-label">數據健康</div></div>
        <div class="stat-card" style="color:var(--warning)"><div class="stat-value">{warning}</div><div class="stat-label">存疑警告</div></div>
        <div class="stat-card" style="color:var(--critical)"><div class="stat-value">{critical}</div><div class="stat-label">嚴重缺失</div></div>
    </div>

    <div class="grid" id="stockGrid"></div>

    <script>
        const data = {json_data};
        const grid = document.getElementById('stockGrid');
        
        data.forEach((s, i) => {{
            const card = document.createElement('div');
            card.className = 'stock-card ' + s.status;
            card.style.animationDelay = (i * 0.01) + 's';
            card.innerHTML = `
                <div class="stock-id">${{s.id}}</div>
                <div class="stock-name">${{s.name}}</div>
                <div class="stock-score">
                    <span class="status-dot"></span>${{s.score}}%
                </div>
            `;
            grid.appendChild(card);
        }});

        function updateClock() {{
            const now = new Date();
            document.getElementById('clock').innerText = now.toLocaleTimeString();
        }}
        setInterval(updateClock, 1000);
        updateClock();
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    logger.info(f"🏆 [Monitor] 戰情監控網頁已產生: {html_path}")

if __name__ == "__main__":
    audit_completeness()
