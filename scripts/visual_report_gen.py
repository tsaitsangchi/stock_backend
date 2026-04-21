import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

from portfolio_optimizer import fetch_latest_predictions, optimize_portfolio
from config import STOCK_CONFIGS

def generate_report(budget=100000):
    # 1. 取得優化結果
    raw_preds = fetch_latest_predictions()
    if not raw_preds:
        print("Error: No predictions found in DB.")
        return

    df_preds = pd.DataFrame(raw_preds)
    
    def get_industry(sid):
        return STOCK_CONFIGS.get(sid, {}).get("industry", "Unknown")
    def get_name(sid):
        return STOCK_CONFIGS.get(sid, {}).get("name", "Unknown")
        
    df_preds["industry"] = df_preds["stock_id"].apply(get_industry)
    df_preds["name"] = df_preds["stock_id"].apply(get_name)
    
    df_final = optimize_portfolio(df_preds[df_preds["prob_up"] >= 0.55], budget)
    
    if df_final.empty:
        print("Error: Portfolio optimization yielded no results.")
        return

    # 2. 準備 JSON 數據
    fee_rate = 0.001425
    total_invested = 0
    stocks_data = []
    
    for _, row in df_final.sort_values("weight", ascending=False).iterrows():
        allocated = budget * row["weight"]
        shares = int((allocated / (1 + fee_rate)) / row["current_price"])
        if shares == 0: continue
        cost = shares * row["current_price"] * (1 + fee_rate)
        total_invested += cost
        
        stocks_data.append({
            "id": row["stock_id"],
            "name": row["name"],
            "industry": row["industry"],
            "weight": float(row["weight"]),
            "shares": shares,
            "cost": int(cost),
            "prob": float(row["prob_up"])
        })

    industries = df_final.groupby("industry")["weight"].sum().to_dict()
    
    report_json = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "utilization": round((total_invested / budget) * 100, 1),
        "industries": industries,
        "stocks": stocks_data
    }

    # 3. 讀取模板並注入
    template_path = Path("scripts/templates/report_template.html")
    output_path = Path("scripts/outputs/visual_report.html")
    
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    final_html = template.replace("{{REPORT_DATA_JSON}}", json.dumps(report_json, ensure_ascii=False))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"Success: Visual report generated at {output_path.absolute()}")
    return output_path.absolute()

if __name__ == "__main__":
    generate_report()
