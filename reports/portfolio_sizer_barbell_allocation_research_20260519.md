# Portfolio Sizer / Portfolio Scripts 配置層研究記錄

**研究日期**: 2026-05-19
**研究對象**: 未來 `portfolio_sizer.py` / 現有 `scripts/pipeline/portfolio_strategy.py`、`scripts/pipeline/portfolio_optimizer.py`、`scripts/pipeline/portfolio_backtest.py`、`scripts/pipeline/faithful_portfolio_backtest.py`
**憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §0.2 / §0.2-D / §0.0-A / §8.4 / §9.2 / §14.7-U / §14.7-Y
**研究目的**: 說明八二法則與槓鈴策略如何落到 portfolio allocation / sizing 層，並定位現有 portfolio 腳本與未來 `portfolio_sizer.py` 的治權邊界。

---

## 1. 研究結論

目前 repo 尚未存在正式 `portfolio_sizer.py`。現有 portfolio 相關邏輯分散於：

```text
scripts/pipeline/portfolio_strategy.py
scripts/pipeline/portfolio_optimizer.py
scripts/pipeline/portfolio_backtest.py
scripts/pipeline/faithful_portfolio_backtest.py
scripts/evaluation/portfolio_strategy.py
scripts/evaluation/portfolio_optimizer.py
scripts/evaluation/portfolio_backtest.py
```

其中 `scripts/pipeline/portfolio_optimizer.py` 與 `scripts/pipeline/portfolio_strategy.py` 已具備八二法則 / 槓鈴策略雛形，但尚未達到 §9.2 正式 delivery 標準。

研究裁決：

1. **配置層是第五個主要工程轉換器**：它接在 `prediction_engine.py` 之後，把 formal prediction 轉成 position weights。
2. **八二法則在配置層不是再選 universe，而是資金權重治理**：只允許從已 committed prediction universe 中選 top-N / top 20% 攻擊端。
3. **槓鈴策略在配置層的合憲形式是「現金/防禦端 + 右尾攻擊端」**：不得把全部資金平均分配於中段。
4. **現有 pipeline optimizer 已有 20/60/20 與 attack_ratio=20% 的雛形**，但仍讀舊 `stock_forecast_daily`，未接上 `prediction_run` / `prediction_values`。
5. **現有 pipeline strategy 使用硬編碼 CORE_STOCKS / QUANTUM_STOCKS**，不符合已升級的 committed universe / prediction SSOT 治權。
6. **未來應新增或重構為 `portfolio_sizer.py`**：只讀唯一 committed prediction-backed run，輸出 allocation proposal，不反向修改 model / prediction。

---

## 2. 第五個轉換器定位

逐程式落地鏈應為：

```text
core_universe_builder.py
  -> feature_store_builder.py
  -> model_trainer.py
  -> prediction_engine.py
  -> portfolio_sizer.py
```

各層責任：

| 層級 | 程式 | 輸出 |
|---|---|---|
| Universe | `core_universe_builder.py` | core / convex / research / quarantine |
| Feature | `feature_store_builder.py` | feature matrix |
| Model | `model_trainer.py` | committed model artifact |
| Prediction | `prediction_engine.py` | `prediction_run` / `prediction_values` |
| Portfolio | 未來 `portfolio_sizer.py` | position weights / allocation proposal |

**配置層不得重選股票、不得重訓模型、不得重算 prediction。**

它只能回答：

```text
在唯一正式 prediction-backed run 已存在的前提下，
哪些標的可配置、各配置多少、保留多少現金、防守/攻擊比例是否合憲。
```

---

## 3. 八二法則在配置層的正確落點

§0.2 在配置層的語意不是「固定買 20% 股票」或「固定 80/20 報酬來源」，而是資源集中與下行隔離：

| 八二法則概念 | 配置層落點 |
|---|---|
| 右尾集中 | 只配置 top-ranked / high-confidence prediction names |
| 中段觀察 | 中段 prediction 不配置或低權重，不作平均分散 |
| 左尾隔離 | bottom-ranked / watch names 不配置 |
| 冪律不對稱 | 攻擊端可集中，但必須有 max position cap |
| 防禦端 | 未配置資金保留為 cash / safety sleeve |

現有 `portfolio_optimizer.py` 中：

```text
n_top = max(1, int(len(df) * 0.2))
attack_ratio = 0.20 * risk_gate_mult
weights["CASH"] = 1.0 - sum(weights)
```

這是八二法則與槓鈴策略的可用雛形。

但缺口是：

- `signals` 來源不是新治理表 `prediction_values`。
- `decision == 'LONG'` 來自舊 `SignalFilter` 語意，不是 §8.4 的 formal prediction label。
- 未明確區分 core / convex sleeve。
- 未檢查唯一 committed prediction-backed run。

---

## 4. 槓鈴策略在配置層的正確落點

槓鈴策略不是「全部持股 80% + 20% 高風險」，而是：

```text
防禦端：現金 / 高流動性 / 低波動 / 可等待
攻擊端：少數高 rank / 高 convexity / 受控上限
中段：避免投入主要資本
```

合憲配置形態：

| Sleeve | 建議語意 | 現有腳本狀態 |
|---|---|---|
| Safety / Cash | 保留未配置資金，降低錯誤模型下行 | `portfolio_optimizer.py` 已以 `CASH` 表達 |
| Core Attack | top-ranked core names，偏穩定右尾 | 尚未正式接上 core tier |
| Convex Attack | top-ranked convex names，允許小權重捕捉厚尾 | 尚未正式接上 convex tier |
| Quarantine / Watch | 不配置 | 現有腳本未接 committed tier |

`portfolio_strategy.py` 目前使用：

```text
CORE_STOCKS = ["2330", "2317", "2881", "2882", "0050"]
QUANTUM_STOCKS = [...]
safety_ratio = 0.80 or 0.85
kinetic_total_ratio = 0.20 or 0.15
```

概念符合槓鈴，但硬編碼清單已不符合現行 universe governance。未來應改為從 committed `core_universe_membership` / `prediction_values` join 取得 tier 與 rank。

---

## 5. 現有腳本研究

### 5.1 `scripts/pipeline/portfolio_optimizer.py`

定位：目前最接近未來 `portfolio_sizer.py` 的腳本。

已具備：

- top 20% selection。
- attack ratio 約 20%。
- market breadth risk gate。
- cash residual。
- MBNRIC industry multiplier 雛形。
- confidence multiplier 雛形。

主要缺口：

1. 讀舊 `stock_forecast_daily`，不讀 `prediction_run` / `prediction_values`。
2. `confidence_level` 與 `decision='LONG'` 語意來自舊 pipeline，不是新 prediction layer。
3. `mbnric_industries` 為硬編碼簡化清單，未接 §0.3 ThemeResonance / universe policy。
4. 無 max single-name cap 實際強制；`optimize_portfolio(..., max_stock_pct=0.05)` 參數未真正套用。
5. 無 sector exposure cap；已知 semiconductor concentration 風險尚未在配置層處理。

### 5.2 `scripts/pipeline/portfolio_strategy.py`

定位：舊式 barbell allocator / 命令列建議工具。

已具備：

- 80/20 或 85/15 safety/kinetic split。
- liquidity screen。
- core max position / aggressive max position 概念。
- core / quantum sleeve 概念。

主要缺口：

1. `CORE_STOCKS` / `QUANTUM_STOCKS` 硬編碼，與 committed universe 脫節。
2. 使用舊 per-stock model artifact `ensemble_<sid>.pkl`，與新 `model_registry` artifact 治權脫節。
3. `prob_up` 不是 `prediction_values.prediction_value` / rank。
4. 引入 `0050`，超出 core+convex 150 股票 universe 語意。
5. 檔案開頭有重複 sys.path 注入，顯示其仍屬舊 pipeline 遺留。

### 5.3 `scripts/pipeline/portfolio_backtest.py`

定位：組合層回測工具。

已具備：

- portfolio return。
- turnover。
- transaction cost。
- Sharpe / max drawdown / Calmar。
- beta to TAIEX。
- worst month。

主要缺口：

1. 使用 `prob > 0.75` 等權分配，不反映新 `prediction_rank`。
2. 不使用 formal prediction run。
3. 未實作 top-N / barbell allocation policy。
4. 可保留為未來 `portfolio_sizer.py` 的 backtest harness，但不應作正式 sizing SSOT。

### 5.4 `scripts/pipeline/faithful_portfolio_backtest.py`

定位：較接近真實策略鏈的舊回測工具。

已具備：

- OOF predictions。
- `SignalFilter`。
- `optimize_portfolio()`。
- turnover cost。

主要缺口：

1. 使用 OOF CSV 舊資料流，不是 committed `prediction_run`。
2. 以 mock report 組裝信號，與新 §8 prediction governance 不一致。
3. 可保留為過去 pipeline 參考，不宜直接升為 §9.2 正式工具。

---

## 6. 未來 `portfolio_sizer.py` 應有的正式契約

建議未來新增：

```text
scripts/core/portfolio_sizer.py
```

或若保留 pipeline 分層，至少需建立：

```text
scripts/core/portfolio_sizer.py      # SSOT sizing engine
scripts/pipeline/portfolio_backtest.py # backtest harness
```

正式輸入：

```text
prediction_run.status='committed'
prediction_values
core_universe_membership
core_universe_snapshot
optional: latest price / liquidity / sector map
```

正式輸出可先是 report / dry-run，不急著建新表：

```text
portfolio_allocation_proposal
stock_id
tier
prediction_rank
prediction_value
signal_label
target_weight
allocation_reason
risk_flags
```

後續若升 §9.2 強制契約，再新增治理表：

```text
portfolio_run
portfolio_weights
```

---

## 7. 建議 sizing policy v0.1

初版應保守，不做複雜 mean-variance optimizer。

建議規則：

1. 只讀唯一 committed prediction-backed run。
2. 只允許 `signal_label='long'` 或 top rank bucket 進入候選。
3. 攻擊端總權重上限 20%。
4. 現金 / safety sleeve 至少 80%，除非 §9.2 明確升級。
5. 單一股票上限 5%。
6. convex tier 單一股票上限 2% 或 3%。
7. sector 上限初版可設 40%，避免 semiconductor 過度集中。
8. bottom 20 / `watch` 永不配置。
9. 若 prediction coverage != 150，直接 FAIL。
10. 若 committed prediction-backed run != 1，直接 FAIL。

此設計能直接回應 §14.7-U 的研究結果：sector-neutral ranking 不應取代 model scoring，但 sector concentration 應在 portfolio/risk layer 處理。

---

## 8. 本研究裁決

portfolio 配置層是八二法則與槓鈴策略真正轉成資金行為的地方；它不應回頭改 universe、features、model 或 prediction。

現有 portfolio 腳本提供了重要雛形：

- `portfolio_optimizer.py`：最接近未來 `portfolio_sizer.py`。
- `portfolio_strategy.py`：保留槓鈴 sleeve 概念，但硬編碼清單需淘汰。
- `portfolio_backtest.py` / `faithful_portfolio_backtest.py`：可作回測 harness，不是正式 sizing SSOT。

下一步最佳工程動作不是直接用舊 portfolio 腳本出正式投資建議，而是先做一個小型、合憲、read-only 的 `portfolio_sizer.py` dry-run：

```text
input:  exactly one committed prediction_run + prediction_values
output: top-N allocation proposal report
policy: 80% cash/safety + <=20% attack, max 5% per stock, sector cap
```

如此才能讓 §0.2 八二法則、槓鈴策略、§8 prediction governance 與 §9.2 終極配置目標連成同一條可 audit 的工程鏈。
