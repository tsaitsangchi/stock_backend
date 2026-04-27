# Antigravity Stock Predictive Engine 🚀

這是一個專為台股設計的**機構級量化預測與決策系統**。系統從最底層的原始資料抓取 (ETL) 開始，透過進階特徵工程 (Feature Engineering)，輸入至感知市場狀態的機器學習模型 (Regime-Aware Ensemble Model)，並最終由具備總經與大戶籌碼護欄的「決策濾網 (Signal Filter)」進行過濾，產出高勝率的交易訊號。

---

## 🏗️ 系統架構與模組

本系統主要分為四大核心模組：

### 1. 數據管線與 ETL (Data Pipeline)
負責從 FinMind 獲取數十種不同維度的歷史與每日更新資料，並存入 PostgreSQL。
- **基礎行情**：股價、本益比、市值比重。
- **基本面 (`fetch_fundamental_data.py`)**：三大財務報表、月營收、除權息。
- **籌碼面 (`fetch_chip_data.py`)**：三大法人買賣超、融資融券、外資持股比例。
- **機構進階籌碼 (`fetch_sponsor_chip_data.py`)**：股權分級 (大戶持股)、八大行庫買賣、期貨大額交易人未沖銷部位。
- **總經與情緒 (`fetch_macro_fundamental_data.py` & `fetch_derivative_data.py`)**：景氣對策信號、CNN恐懼貪婪指數、選擇權 Put/Call Ratio、美國指標股。
- **自動化排程 (`parallel_fetch.py` & `wait_and_train_and_full.sh`)**：支援多執行緒並行抓取，內建 402/Timeout 的指數退避重試機制。

### 2. 特徵工程 (`feature_engineering.py` & `data_pipeline.py`)
將散落在不同資料表的原始數據進行對齊 (`LEFT JOIN` 於時間軸)，並萃取出機器學習模型所需的 Alpha 特徵：
- **技術指標**：動能、RSI、MACD、波動率 (Realized Volatility)。
- **基本面動能**：營收 YoY/QoQ 成長率、EPS 加速指標。
- **進階籌碼特徵**：聰明錢同向買超 (`smart_money_sync_buy`)、近三月大戶持股變化 (`large_holder_change_3m`)。
- **總經特徵**：景氣燈號顏色轉換、極端貪婪/恐懼標記 (`is_extreme_fear/greed`)。

### 3. 機器學習預測模型 (`models/ensemble_model.py` & `train_evaluate.py`)
使用 XGBoost / LightGBM 建構分類模型。
- **Regime-Aware 策略**：模型訓練時會感知目前的市場狀態（牛市、熊市、盤整），動態調整權重。
- **多時程預測 (Multi-Horizon)**：針對 15天、21天、30天 等不同持有週期計算上漲機率 (`prob_up`)。

### 4. 決策濾網 (`signal_filter.py`)
模型給出的僅是「機率」，決策濾網負責擔任最終的**風險控管護欄**，以 5 大維度進行 100 分綜合評估：
1. **模型機率 (35%)**：XGBoost 產出的上漲機率。
2. **市場狀態 (20%)**：確認大盤趨勢與波動度。
3. **機構籌碼 (20%)**：外資、八大行庫是否同步買超？大戶籌碼是否集中？
4. **基本面 (15%)**：營收與毛利率是否支撐？
5. **情緒與宏觀 (10%)**：恐懼與貪婪指數、選擇權 PCR、景氣燈號。

**🚫 強制阻斷 (Hard Block) 機制**：
若偵測到「大戶近三月大舉倒貨超過 5%」，即使技術面極佳，也會強制攔截並輸出 `HOLD_CASH`。

---

## 🏃 執行順序與日常維運

系統的自動化執行標準作業流程 (SOP) 如下：

### Step 1: 資料抓取 (Daily Fetch)
每日收盤後（或初始化時），執行資料更新。
```bash
# 啟動自動排程腳本 (會先抓重點名單，再抓全市場)
nohup ./scripts/wait_and_train_and_full.sh > scripts/outputs/logs/wait_full.log 2>&1 &
```
*註：這會呼叫各個 `fetch_*.py` 腳本，透過 PostgreSQL 的 `ON CONFLICT DO UPDATE` 完成增量更新。*

### Step 2: 模型重訓 (Train & Evaluate)
當累積了足夠的新資料，或市場發生重大 Regime Shift 時，需重新訓練模型：
```bash
python scripts/train_evaluate.py --all
```
這會為 `config.py` 中的每一支股票生成專屬的 LightGBM/XGBoost 模型。

### Step 3: 每日決策預測 (Predict)
每日盤前執行預測腳本，生成當日的操作建議清單。
```bash
python scripts/predict_15d.py
```
這會載入最新模型與資料，通過 `SignalFilter` 產生 JSON 報告（包含 `LONG` / `HOLD_CASH` 決策以及 `Boosting` / `Blocking` 的理由）。

## 📊 系統監控與視覺化 (Monitoring & Visualization)

本系統內建基於 **Streamlit** 的輕量級監控儀表板，讓您無需翻閱日誌即可掌握全域狀態。

### 核心功能：
- **數據流鮮度**：即時監控 PostgreSQL 各資料表最新資料日期，防止抓取失效。
- **模型健康度**：計算過去 30 天實戰準確率 (DA)，自動標記效能衰退的模型。
- **今日訊號分析**：視覺化全市場機率分佈，偵測模型是否過擬合或產生偏見。
- **特徵漂移 (PSI)**：自動監控特徵穩定性指數，提醒何時應進行重訓練。

### 啟動方式：
```bash
# 在 venv 環境下執行
streamlit run scripts/dashboard.py
```
啟動後，系統會自動在瀏覽器打開網頁（預設通訊埠 8501）。

---

## 🔍 架構改進與未來發展建議 (Architecture Review)

雖然目前系統已經從單純的技術面預測，升級成具備宏觀與大戶護欄的強大引擎，但在「大型量化系統」的標準下，仍有以下幾個優化空間：

### 1. 特徵計算效能痛點：導入 Feature Store
- **現況**：目前 `data_pipeline.py` 是在每次訓練或預測時，從資料庫拉出原始資料並「即時計算 (On-the-fly)」所有特徵（例如 RSI、大戶持股增減）。
- **缺點**：當股票數量達到 1500+，且歷史長達 10 年時，每次重訓都會耗費大量 CPU 時間重複計算一模一樣的指標。
- **改進建議**：在資料庫建立一個實體的 `daily_features` 表格。每日 ETL 抓完原始資料後，寫一支腳本**將特徵計算好並存入 DB**。訓練時直接 `SELECT *`，速度將提升百倍。

### 2. 爬蟲管線的脆弱性：導入 Airflow 或 Celery
- **現況**：目前依賴 Bash (`wait_and_train.sh`) 或 Python 的 `while` 迴圈來處理重試、超時與依賴關係。
- **缺點**：如果伺服器重開機，或是 FinMind 突然更改 API 結構，Bash 腳本很難做到精確的錯誤追蹤與斷點續傳。
- **改進建議**：引入 **Apache Airflow** 等任務排程工具，將 `fetch -> feature_engineering -> train -> predict` 畫成有向無環圖 (DAG)。這能帶來圖形化監控與更穩定的重試機制。

### 3. 模型生命週期管理：導入 MLflow
- **現況**：模型直接儲存為 `.json` 覆蓋舊檔。
- **缺點**：無法追蹤「上個月的模型」跟「這個月的模型」哪個勝率較高？如果新加入了一個總經特徵導致過度擬合 (Overfitting)，我們很難回滾 (Rollback)。
- **改進建議**：使用 **MLflow** 記錄每一次訓練的 Hyperparameters、特徵重要性 (Feature Importance) 以及回測的 Sharpe Ratio，讓模型的演進有跡可循。

### 4. 回測引擎的深度整合
- **現況**：目前的決策邏輯 (`signal_filter.py`) 寫得很棒，但其門檻（例如 `> 60分才買入`、`貪婪指數 > 75 扣分`）依賴主觀經驗。
- **改進建議**：需要建立一套能夠完全模擬 `SignalFilter` 行為的事件驅動回測框架（如 Backtrader）。透過機器學習的網格搜索 (Grid Search)，自動找出最佳的「濾網權重配置」，而非人工硬編碼。

### 5. 即時性 (Intraday) 的匱乏
- **現況**：目前的系統是 EOD (End of Day) 系統，必須等晚上 FinMind 更新完資料才能給出明天的操作建議。
- **改進建議**：若想捕捉盤中快速爆發的機會，需串接 WebSocket 即時報價 API，並把架構改為 Streaming 處理模式。
