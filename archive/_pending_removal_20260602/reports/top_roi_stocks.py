import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# ── 系統路徑修復 ──
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.append(str(_PROJECT_ROOT / "scripts"))

try:
    from core.db_utils import db_transaction, write_pipeline_log
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_top_sharpe_report():
    logger.info("📊 [Report] 正在產出高夏普值 Top 3 菁英戰報...")
    
    with db_transaction() as cur:
        # 取得夏普值前三名且具備名稱的標的
        cur.execute("""
            SELECT 
                e.stock_id, 
                i.stock_name, 
                e.sharpe_ratio, 
                e.total_return, 
                e.max_drawdown,
                e.win_rate
            FROM evaluation_log e
            JOIN stocks s ON e.stock_id = s.stock_id
            LEFT JOIN stock_info i ON e.stock_id = i.stock_id
            WHERE s.is_active = TRUE
            AND e.created_at > CURRENT_DATE - INTERVAL '7 days'
            ORDER BY e.sharpe_ratio DESC
            LIMIT 3
        """)
        top_stocks = cur.fetchall()

    if not top_stocks:
        logger.warning("⚠️ 查無回測數據，請先執行 run_batch_backtest.py")
        return

    html_path = _PROJECT_ROOT / "monitor" / "investment_report.html"
    
    html_template = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>Trinity Elite - Top 3 投資戰報</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #05070a;
            --gold: #d4af37;
            --gold-light: #f9e272;
            --card-bg: rgba(255, 255, 255, 0.05);
            --accent: #00f2ff;
        }}
        body {{
            background: var(--bg);
            color: #fff;
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 60px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .header {{ text-align: center; margin-bottom: 60px; }}
        .header h1 {{ 
            font-size: 3rem; 
            margin: 0; 
            background: linear-gradient(to bottom, var(--gold-light), var(--gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 5px;
            text-transform: uppercase;
        }}
        .podium {{
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 30px;
            width: 100%;
            max-width: 1200px;
        }}
        .stock-card {{
            background: var(--card-bg);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            backdrop-filter: blur(20px);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
        }}
        .rank-1 {{ width: 350px; border-color: var(--gold); box-shadow: 0 0 30px rgba(212, 175, 55, 0.2); z-index: 2; }}
        .rank-2, .rank-3 {{ width: 300px; opacity: 0.8; }}
        .stock-card:hover {{ transform: translateY(-15px) scale(1.05); opacity: 1; border-color: var(--gold); }}
        
        .crown {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .stock-id {{ font-size: 1.2rem; color: var(--gold); opacity: 0.8; }}
        .stock-name {{ font-size: 2rem; font-weight: 600; margin: 10px 0; }}
        .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 30px; }}
        .metric-box {{ text-align: left; padding: 10px; border-left: 2px solid var(--gold); }}
        .m-label {{ font-size: 0.7rem; opacity: 0.5; text-transform: uppercase; }}
        .m-value {{ font-size: 1.2rem; font-weight: 600; color: var(--gold-light); }}
        
        .back-btn {{
            margin-top: 50px;
            color: var(--accent);
            text-decoration: none;
            font-size: 0.9rem;
            border: 1px solid var(--accent);
            padding: 10px 25px;
            border-radius: 30px;
            transition: all 0.3s;
        }}
        .back-btn:hover {{ background: var(--accent); color: #000; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trinity Elite</h1>
        <p style="opacity:0.6; letter-spacing: 2px;">核心標的全集(dynamic per §14.7-BW)— 高夏普值 Top 3 戰報</p>
    </div>

    <div class="podium">
        <!-- Rank 2 -->
        <div class="stock-card rank-2">
            <div class="crown">🥈</div>
            <div class="stock-id">{top_stocks[1]['stock_id'] if len(top_stocks)>1 else '-'}</div>
            <div class="stock-name">{top_stocks[1]['stock_name'] if len(top_stocks)>1 else 'N/A'}</div>
            <div class="metric-grid">
                <div class="metric-box"><div class="m-label">Sharpe</div><div class="m-value">{top_stocks[1]['sharpe_ratio'] if len(top_stocks)>1 else '-'}</div></div>
                <div class="metric-box"><div class="m-label">ROI</div><div class="m-value">{f"{top_stocks[1]['total_return']*100:.1f}%" if len(top_stocks)>1 else '-'}</div></div>
            </div>
        </div>

        <!-- Rank 1 -->
        <div class="stock-card rank-1">
            <div class="crown">👑</div>
            <div class="stock-id">{top_stocks[0]['stock_id']}</div>
            <div class="stock-name">{top_stocks[0]['stock_name']}</div>
            <div class="metric-grid">
                <div class="metric-box"><div class="m-label">Sharpe Ratio</div><div class="m-value">{top_stocks[0]['sharpe_ratio']}</div></div>
                <div class="metric-box"><div class="m-label">Total Return</div><div class="m-value">{top_stocks[0]['total_return']*100:.1f}%</div></div>
                <div class="metric-box"><div class="m-label">Win Rate</div><div class="m-value">{top_stocks[0]['win_rate']*100:.1f}%</div></div>
                <div class="metric-box"><div class="m-label">Max Drawdown</div><div class="m-value">{top_stocks[0]['max_drawdown']*100:.1f}%</div></div>
            </div>
        </div>

        <!-- Rank 3 -->
        <div class="stock-card rank-3">
            <div class="crown">🥉</div>
            <div class="stock-id">{top_stocks[2]['stock_id'] if len(top_stocks)>2 else '-'}</div>
            <div class="stock-name">{top_stocks[2]['stock_name'] if len(top_stocks)>2 else 'N/A'}</div>
            <div class="metric-grid">
                <div class="metric-box"><div class="m-label">Sharpe</div><div class="m-value">{top_stocks[2]['sharpe_ratio'] if len(top_stocks)>2 else '-'}</div></div>
                <div class="metric-box"><div class="m-label">ROI</div><div class="m-value">{f"{top_stocks[2]['total_return']*100:.1f}%" if len(top_stocks)>2 else '-'}</div></div>
            </div>
        </div>
    </div>

    <a href="dashboard.html" class="back-btn">← 返回全核心監控室</a>
    <p style="margin-top:40px; font-size: 0.7rem; opacity: 0.3;">報告產出時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    logger.info(f"🏆 [Report] 投資戰報已產出: {html_path}")
    write_pipeline_log("elite_report_generation", "MARKET", "success", "report")

if __name__ == "__main__":
    generate_top_sharpe_report()
