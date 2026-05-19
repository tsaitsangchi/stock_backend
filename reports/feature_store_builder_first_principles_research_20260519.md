# feature_store_builder.py 第一性原理可觀測特徵研究記錄

**研究日期**: 2026-05-19
**研究對象**: `scripts/core/feature_store_builder.py`
**程式版本**: `feature_store_builder.py v0.1`
**憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §0.0-A / §0.1 / §0.4 / §8.2 / §8.5
**研究目的**: 說明第一性原理如何被轉成 as-of-strict、可觀測、可稽核的 Feature Store 特徵。

---

## 1. 研究結論

`feature_store_builder.py` 是「§0.1 第一性原理 → 可觀測 features」的第一個主要工程轉換器。它不重新選股、不訓練模型、不產生 prediction，而是把已 committed 的 `core_universe ∪ convex_universe` 轉成 27 個 feature values，寫入：

```text
feature_definition
feature_values
feature_store_snapshot
```

本程式的核心治理語意是：

```text
已鎖定 core+convex 150
  -> as-of-strict 讀取 raw DB
  -> 轉成 price / liquidity / fundamental / institutional / macro / theme features
  -> 寫入 Feature Store
  -> 供 model_trainer 使用
```

因此它是「治理 universe → 可訓練特徵矩陣」的橋。

---

## 2. 第一性原理對映

憲章 §0.1 的核心元素是：

```text
F = f(M, V) x Delta_lnP
```

在 `feature_store_builder.py` 中，這不是直接硬寫成單一公式，而是拆成可觀測 feature groups：

| 第一性元素 | Feature group | 實作特徵 | 說明 |
|---|---|---|---|
| `Delta_lnP` 價格位移 | `price` | `log_return_20d`, `log_return_60d`, `log_return_252d`, `ma_ratio_20`, `ma_ratio_60` | 價格路徑與位移，是最直接的市場狀態變數。 |
| 價格路徑風險 | `price` | `volatility_60d`, `volatility_252d`, `max_drawdown_252d` | 將價格非線性與下行風險轉為可觀測量。 |
| `M` 流動性質量 | `liquidity` | `avg_daily_value_log_60d`, `avg_daily_value_log_252d`, `turnover_mean_60d`, `zero_volume_ratio_252d` | 交易金額、週轉與零成交比例代表價格可承載資金的能力。 |
| `V` 內在價值密度 | `fundamental` | `revenue_yoy_12m`, `revenue_yoy_3m`, `eps_sum_4q`, `net_income_positive_ratio_8q` | 營收成長與獲利 proxy，對應 FundamentalGravity。 |
| 外部資訊力 / 資金力 | `institutional` | `foreign_net_20d`, `foreign_net_60d`, `trust_net_20d`, `trust_net_60d`, `margin_ratio_60d` | 外資、投信、融資融券代表外部力量與注意力集中。 |

研究裁決：本程式忠實遵守 §0.1-A 的限制，沒有把 `F = f(M, V) x Delta_lnP` 寫成單一硬公式，而是用可觀測 proxy 交給模型驗證。

---

## 3. 統合框架中的非第一性欄位

雖然本研究聚焦 §0.1，但 `feature_store_builder.py` 也承接 §0.0-A 統合框架：

| Feature group | 對應章節 | 語意 |
|---|---|---|
| `theme` | §0.3 | `theme_strength`, `theme_is_semiconductor` 將 MBNRIC / ThemeResonance 轉為股票層欄位。 |
| `macro` | §0.3 + §0.4 | `macro_dff_level`, `macro_vix_level`, `macro_t10y2y_level`, `macro_unrate_yoy` 將 FRED regime 轉為 as-of 可觀測背景。 |

重要邊界：macro features 是 broadcast 至每股的同值 regime 特徵，若沒有 stock-sensitive interaction，cross-sectional rank model 中可能貢獻為零；這與 §14.7-U 的 ablation 結果一致。

---

## 4. As-of-strict 與 anti-leakage

本程式最重要的正確性不是 feature 數量，而是時間邊界。

所有資料載入函式都使用：

```sql
WHERE date <= as_of_date
```

主要例子：

- `_load_price_series()`: price window 截止 `as_of_date`
- `_load_revenue()`: 月營收截止 `as_of_date`
- `_load_financial()`: 財報截止 `as_of_date`
- `_load_institutional()`: 法人買賣超截止 `as_of_date`
- `_load_theme()`: `TaiwanStockInfo` 取 `date <= as_of_date` 最新一筆
- `_load_macro()`: `FredData` 取 `date <= as_of_date` 最新值

`label_horizon` 僅寫入 `feature_store_snapshot`，不在本程式產生 label。這符合 §8.5 anti-leakage：Feature Store 只產生特徵，不接觸未來價格。

---

## 5. Feature Dictionary 結構

本程式定義 27 個 features：

| Group | Count | Null strategy | 主要資料源 |
|---|---:|---|---|
| `price` | 8 | `drop` | `TaiwanStockPriceAdj` |
| `liquidity` | 4 | `drop` | `TaiwanStockPriceAdj` |
| `fundamental` | 4 | `drop` / `zero_fill` | `TaiwanStockMonthRevenue`, `TaiwanStockFinancialStatements` |
| `institutional` | 5 | `zero_fill` | `TaiwanStockInstitutionalInvestorsBuySell`, `TaiwanStockMarginPurchaseShortSale` |
| `theme` | 2 | `zero_fill` | `TaiwanStockInfo` |
| `macro` | 4 | `drop` | `FredData` |

`drop` 的治理語意：資料不足時不寫該 feature value，讓下游 completeness audit 定位生命週期或觀察窗不足。
`zero_fill` 的治理語意：缺值可被解釋為「沒有觀測到該事件 / 該流量」，但必須標記 `is_null_imputed=True`。

---

## 6. Universe Lock

Feature Store 的股票範圍由最新 committed universe snapshot 決定：

```sql
core_tier IN ('core_universe', 'convex_universe')
```

這點很重要：`feature_store_builder.py` 不自己決定核心股，也不讀 research / quarantine。它完全承接 `core_universe_builder.py` 的治理結果。

裁決：此程式的治權邊界正確。若 feature coverage 不足，應由 completeness audit / source audit 定位，不應在 Feature Store 內偷偷擴大 universe。

---

## 7. 寫入順序與治理表

正式 `--commit` 時：

1. 先建立或重設 `feature_store_snapshot` 為 `draft`
2. 寫入 / upsert `feature_definition`
3. 刪除同 feature_set_id 舊 `feature_values`
4. 批次寫入新 `feature_values`
5. 更新 `feature_store_snapshot` 為 `committed`
6. 寫入 `data_audit_log`

這個流程確保下游只應讀取 committed snapshot；building 中的 feature set 不應被模型訓練使用。

---

## 8. 已符合憲章的地方

1. **第一性原理已拆成可觀測 proxy**
   價格、流動性、基本面、法人力皆有對應 feature group。

2. **沒有硬寫哲學公式**
   沒有直接計算 `F = f(M, V) x Delta_lnP`，符合 §0.1-A 禁令。

3. **時間邊界清楚**
   所有 raw data 均 `date <= as_of_date`，label 不在本程式產生。

4. **Universe lock 正確**
   僅讀 committed core+convex，不把 research/quarantine 偷渡進模型前置資料。

5. **Missing policy 可追蹤**
   `drop` 與 `zero_fill` 明確寫入 `feature_definition`，`zero_fill` 另標 `is_null_imputed`。

6. **下游邊界清楚**
   不保存 labels、model outputs、prediction signals。

---

## 9. 值得後續研究或改善的地方

### 9.1 財報與月營收的發布日語意

目前使用 `date <= as_of_date`。若 raw table 的 `date` 是財報季度日或營收月份，而非實際公告日，可能存在公告時點與資料可得時點差異。後續應研究：

- FinMind 欄位中是否有公告日或 create_time 可用。
- 月營收 `create_time` 是否應取代或輔助 `date`。
- 財報是否需要 filing lag rule，例如季報延後 N 日才可用。

### 9.2 Institutional features 使用絕對股數

`foreign_net_*` 與 `trust_net_*` 是絕對股數，容易偏向大型股。後續可研究：

- 以成交量或流通股數 normalize。
- 加入 rolling percentile / z-score。
- 分市場別或產業別 normalization。

### 9.3 Macro features 是 broadcast 常數

同一 `as_of_date` 下，macro 值對所有股票相同。若模型是 cross-sectional rank，單純 macro level 可能沒有區分力。後續可研究：

- macro × sector exposure interaction。
- macro × liquidity / volatility interaction。
- regime-conditioned model family，而非直接 broadcast feature。

### 9.4 Theme features 與 universe ThemeResonance 重複

`theme_strength` 與 `theme_is_semiconductor` 可能重複反映 core universe selection 中已使用的 `ThemeResonance`。後續可研究：

- 是否造成 selection bias 或 redundant signal。
- 是否應只作 audit/explain feature，不作 model input。
- 是否需加入 non-semiconductor MBNRIC 子支柱的更細分類。

### 9.5 Drop 策略需要定期 completeness report

`drop` 導致理論槽位少於實際 rows。這是合理設計，但必須持續區分：

- source-empty
- 合法生命週期缺口
- 觀察窗不足
- feature builder bug
- DB/API 漏抓

§14.7-M 的 completeness 定位已證明這條路徑正確，後續應延續成固定報告。

---

## 10. 本研究裁決

`feature_store_builder.py` 是「第一性原理 → 可觀測 feature matrix」的正確第二支研究程式。

它把：

- `Delta_lnP` 轉成 price return / volatility / moving-average / drawdown features；
- `M` 轉成 liquidity features；
- `V` 轉成 revenue / EPS / net-income features；
- external force 轉成 institutional / margin features；
- §0.3 regime 轉成 theme / macro features；
- §0.4 觀測主義轉成 as-of-strict、universe lock、null policy 與 committed snapshot；

最後產生可供 `model_trainer.py` 使用的 Feature Store。下一支逐程式研究應進入：

```text
scripts/core/model_trainer.py
```

研究重點應是：這些 features 是否真的轉化為可驗證的 rank-IC，而不是只停留在哲學上合理。
