# 《2026 量子金融藍圖》(2026 Quantum Finance Blueprint)
> **版本 v4.0 (Trinity Edition)**：融合「第一性原則、八二法則與康波週期」的財富重構框架。

本藍圖旨在利用量子物理動能、統計學過濾與 50 年經濟長波理論，在 2026 年歷史性技術奇點到來之際，為核心資產（2330, 2317, 2454 等）建立一套具備「反脆弱性」的決策實體。

---

## 系統架構

```
tsmc_predictor/
├── config.py               # 全域設定（路徑、超參數、個股客製化配置）
├── data_pipeline.py        # 資料層：從 PostgreSQL 讀取 → 每日寬格式 DataFrame
├── feature_engineering.py  # 特徵工程：5 大類 ~100 個通用特徵 + 動態產業特徵
├── train_evaluate.py       # 訓練核心：Purged Walk-Forward CV + 評估 + 模型儲存
├── parallel_train.py       # 並行管理器：全自動並行訓練 80+ 標的模型
├── parallel_fetch.py       # 並行抓取器：高速抓取全市場技術、籌碼、基本面數據
├── predict.py              # 推論核心：每日執行，輸出趨勢報告
├── model_health_check.py   # 健康檢查：監控資料鮮度與實戰 DA
├── automate_daily.py       # 自動化流水線：一鍵完成推論、檢查與組合優化
├── models/
```

---

## PostgreSQL 資料表對應

| 資料表 | 內容 | 更新頻率 |
|--------|------|---------|
| `stock_price` | 日線股價（open/max/min/close/volume）| 每日 |
| `stock_per` | PER / PBR / 股利殖利率 | 每日 |
| `institutional_investors_buy_sell` | 三大法人買賣超 | 每日 |
| `margin_purchase_short_sale` | 融資融券餘額 | 每日 |
| `shareholding` | 外資持股比率 | 每日 |
| `total_return_index` | TAIEX / TPEx 指數 | 每日 |
| `exchange_rate` | USD/TWD 等匯率 | 每日 |
| `interest_rate` | FED / BOJ / ECB 等央行利率 | 不定期 |
| `month_revenue` | 月營收（YoY / MoM）| 每月 |
| `financial_statements` | 損益表（EPS / 毛利率 / ROE）| 每季 |
| `balance_sheet` | 資產負債表（流動比率 / 負債比）| 每季 |
| `dividend` | 現金股利 / 除息日 | 每次公告 |

---

## 第一性原則特徵分組

| 類別 | 代表特徵 | 主要預測貢獻 |
|------|---------|------------|
| ① 技術動能 | RSI、MACD、BB%、realized_vol | 短期 momentum / 反轉 |
| ② 資金流情緒 | foreign_net、margin_balance | 法人 vs 散戶供需扭曲 |
| ③ 基本面脈衝 | revenue_yoy、gross_margin、eps_ttm | 長期內在價值定錨 |
| ④ 估值錨點 | per_pct_rank、dividend_yield | 資金吸引 / 排斥 |
| ⑤ 宏觀因子 | fed_rate_chg、usd_twd、taiex_rel | 折現率 / 風險偏好 |
| ⑥ 事件驅動 | days_to_ex_dividend、dividend_ex_dummy | 除息前後效應 |
| ⑦ 滾動統計 | skew、kurt、sharpe | 尾部風險 / 動量強度 |

---

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 確認 PostgreSQL 連線

修改 `data_pipeline.py` 中的 `DB_CONFIG`：
```python
DB_CONFIG = {
    "dbname":   "stock",
    "user":     "stock",
    "password": "stock",
    "host": "localhost",
    "port":     "5432",
}
```

連線診斷：
```bash
python data_pipeline.py
```
成功會印出 shape、date range、最後 5 筆關鍵欄及缺失率。

### 3. 資料抓取與更新

```bash
# [推薦] 並行高速抓取 (技術、籌碼、基本面、國際市場、期權)
python scripts/parallel_fetch.py

# 或是手動執行單項抓取 (範例：技術面資料)
python scripts/fetch_technical_data.py --start 2024-01-01
```

### 4. 訓練模型

```bash
# 完整訓練（含 TFT，首次約 2~4 小時）
python train_evaluate.py --start 2010-01-01

# 快速版（僅 XGBoost + LightGBM，約 10~30 分鐘）
python train_evaluate.py --start 2018-01-01 --no-tft

# 只做 Walk-Forward 評估，不儲存最終模型
python train_evaluate.py --no-tft --wf-only
```

訓練輸出：
- `outputs/models/ensemble_final.pkl` — 最終 Ensemble 模型
- `outputs/models/tft_final.ckpt` — TFT checkpoint
- `outputs/wf_fold_metrics.csv` — 各 Fold 評估結果
- `outputs/feature_importance.csv` — 特徵重要性排名
- `outputs/train.log` — 完整訓練日誌

### 4. 每日推論

```bash
# 標準推論（每日收盤後執行）
python predict.py

# 輸出 JSON 報告
python predict.py --output-json outputs/pred_$(date +%Y%m%d).json

# 加上資料漂移偵測
python predict.py --drift-check

# 跳過 TFT（速度較快）
python predict.py --no-tft
```

### 5. 自動化流水線與健康檢查

為了確保大規模生產環境（80+ 標的）的穩定性，系統提供了全自動化管線與監控腳本：

#### 執行全自動流水線
```bash
# 一鍵完成：批次推論 -> 健康檢查 -> 投資組合優化
python scripts/automate_daily.py
```

#### 單獨執行健康檢查
```bash
# 檢查數據鮮度、模型時效性與最近 30 天實戰 DA
python scripts/model_health_check.py
```

健康檢查報告包含：
- **數據流鮮度**：檢查 PostgreSQL 各表最新日期，預防爬蟲失效。
- **模型時效性**：監控各個股模型是否超過 30 天未重訓。
- **實戰準確度**：對比歷史預測軌跡與實際價格，計算真實方向正確率 (DA)。

---

## 預期輸出範例

```
══════════════════════════════════════════════════════════════
  台積電 (2330)  30 天趨勢預測報告
  基準日：2026-03-27  →  目標日：2026-04-26
══════════════════════════════════════════════════════════════

  當前收盤價 ：       1,820  TWD

  ▶ 方向預測   ：📈 上漲
  ▶ 上漲機率   ：67.3%
  ▶ 信心等級   ：🟢 高信心
  ▶ 模型離散度 ：σ=0.042  （越小→三模型越一致）

  預期報酬區間（q10 / q50 / q90）：
    低端   -3.2%  →    1,762  TWD
    中位   +4.1%  →    1,895  TWD  ← 點預測
    高端  +11.5%  →    2,029  TWD

  模型分解：
    ensemble    :  0.673  ██████████████░░░░░░
    xgb         :  0.651  █████████████░░░░░░░
    lgb         :  0.668  █████████████░░░░░░░
    tft         :  0.701  ██████████████░░░░░░

  驅動因子 (SHAP)：
    ↑ 支撐上漲：['revenue_yoy', 'foreign_net_ma5', 'rsi_14']
    ↓ 壓制下跌：['per_pct_rank_252', 'fed_rate_chg_30d']

  關鍵指標快照：
    RSI(14)=48.3  MACD hist=-0.52
    PER=27.48（歷史 68.0% 分位）
    外資淨買超 MA5=-3,061,107
    ...
```

---

## Walk-Forward 歷史表現目標

| 指標 | 目標 | 說明 |
|------|------|------|
| 方向正確率（DA） | ≥ 65% | 隨機基準 50%，台積電歷史可達 64~68% |
| IC（Rank IC） | ≥ 0.05 | Spearman 相關係數 |
| Sharpe Ratio | ≥ 1.0 | 30 天持有策略年化 |

---

## 模型整體架構

```
DB：PostgreSQL 17
    ↓
data_pipeline.build_daily_frame()        ← 12 張資料表 LEFT JOIN + ffill
    ↓
feature_engineering.build_features()    ← ~100 個特徵 + 目標變數
    ↓
    ┌───────────────────────────────────────────────────────┐
    │                  Purged Walk-Forward CV               │
    │                                                       │
    │  [TFT]          [XGBoost]      [LightGBM]            │
    │  序列 + 事件    表格特徵        表格特徵               │
    │  分位輸出       分類/回歸       分類/回歸              │
    │       └──────────────┬──────────────┘                │
    │                      ↓                               │
    │          [Level-2 Meta-Learner]                       │
    │          Logistic Regression                          │
    │                      ↓                               │
    │             上漲機率 + 信心區間 + SHAP                │
    └───────────────────────────────────────────────────────┘
```

---

## 注意事項

- **資料洩漏防護**：Walk-Forward 使用 `embargo_days=30` 禁區，確保含有未來目標的訓練樣本不滲入驗證集。
- **定期重訓**：建議每 30 天重新訓練，捕捉市場 regime change（可用 cron 排程）。
- **免責聲明**：本系統僅供研究參考，不構成投資建議。所有交易決策需結合人工判斷。

---

## 🌌 未來戰略：三位一體決策體系 (The Trinity Architecture)

本系統正朝向 **「微觀物理、中觀優化、宏觀導航」** 三位一體的智能體系演進：

### 1. 第一性原則 (Micro Logic)
*   **資訊物理學**：價格是位移，驚奇值 (Surprise) 是驅動力。系統將持續精煉「預期殘差」特徵。
*   **訂單流結構**：未來將引入微觀供需失衡監測，看透價格漲跌的底層物理。

### 2. 八二法則 (Optimization)
*   **槓鈴策略**：維持 80% 極度防禦與 20% 極度進取的資金配置，避開無意義的中等風險。
*   **20/60/20 濾波**：模型將針對「肥尾極端區」與「平穩期」自動切換權重邏輯。

### 3. 康波週期 (Macro Navigation)
*   **2026 導航點**：將 2026 年定義為第六波技術奇點 (AI/Robotics) 的關鍵共振期，引導長線權重配置。

### 🤖 多智能體協作 (Multi-Agent Roadmap)
我們正在構建「宏觀導航、價值偵探、風險防衛」三位一體的 Agent 協作機制，將本量化引擎升級為全自動的財富智慧實體。

---

## 💰 財富重構策略 (Wealth Construction Strategy)

針對 2026 年康波週期底部轉折，本系統實施以下戰略佈局：

### 1. 槓鈴部位管理 (Barbell Allocation)
*   **80% 安全底座 (Safety Base)**：分配給市值前 5% 的低波動權值股 (如 2330 核心持倉) 與國債，確保在 2026 前後的信貸出清期生存。
*   **20% 量子爆發位 (Quantum Upside)**：利用「量子衝量 (Impulse)」與「驚奇值 (Surprise)」特徵，動態捕捉 AI 與機器人賽道的非線性超額收益。

### 2. 2026 導航點執行 (2026 Pivot Execution)
*   系統將 2026 年 6 月定義為「第六波技術革命」的關鍵共振視窗。
*   **進入視窗前**：側重「熵值 (Entropy)」監控，防範舊週期的劇烈震盪。
*   **進入視窗後**：自動下調「量子過濾器」的門檻，並將資金權重向「高技術密度」標的傾斜。

### 3. 反脆弱機制 (Anti-fragility)
*   **肥尾保護**：當市場進入上下各 20% 的極端區間時，系統會自動啟動非對稱停損停利 (1 : 3)，確保損失有限而獲利無限。
