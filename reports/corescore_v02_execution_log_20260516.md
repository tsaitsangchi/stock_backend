# CoreScore v0.2 六層評分實作執行記錄

**執行日期**: 2026-05-16  
**執行目的**: 實作 core_universe_builder.py v0.2 六層 CoreScore 正式評分，取代 v0.2-preflight metadata bootstrap  
**治權基準**: 系統架構大憲章_v5.4.22.md §6.4  
**執行環境**: venv/bin/python (Python 3.12)

---

## 實作變更摘要

### TOOL_VER 升版
- `v0.2-preflight` → `v0.2`
- `DEFAULT_POLICY_VERSION`: `core_universe_policy_v0.1` → `core_universe_policy_v0.2`

### 六層 CoreScore 評分公式

```
CoreScore = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP
```

| 層級 | 名稱 | 權重 | 資料來源 | 評分邏輯 |
|------|------|------|----------|---------|
| DQ | DataQuality | 25% | TaiwanStockPriceAdj + MonthRevenue + FinancialStatements | price_cov*40 + revenue_cov*30 + financial_cov*30 |
| LM | LiquidityMass | 25% | TaiwanStockPriceAdj | log10(avg_daily_value) 正規化 + continuity*15 |
| FG | FundamentalGravity | 20% | TaiwanStockMonthRevenue + FinancialStatements | YoY revenue growth + 獲利能力 |
| TR | ThemeResonance | 15% | TaiwanStockInfo.industry_category | THEME_KEYWORDS 主題共振 |
| IF | InstitutionalFlow | 10% | TaiwanStockInstitutionalInvestorsBuySell | 外資 + 投信淨買超 |
| VC | VolatilityControl | 5% | TaiwanStockPriceAdj | 收盤價變異係數 CV |
| RP | RiskPenalty | 扣分 | 多表 | 高波動低流動 + 資料缺口 + 排除條件 |

### 新增方法

- `_load_market_data()`: 批量讀取六張個股資料表，回傳 `(price_data, revenue_data, financial_data, institutional_data)` dict
- `_data_quality_score_v2()`: 基於實際資料覆蓋率的 DQ 評分
- `_liquidity_mass_score()`: log-scale 日均成交額 + 連續性
- `_fundamental_gravity_score()`: YoY 營收成長 + 淨利正負
- `_theme_resonance_score()`: 與原 `_theme_score()` 邏輯相同但重命名
- `_institutional_flow_score()`: Foreign_Investor + Investment_Trust 淨買超 (單位: 股)
- `_volatility_control_score()`: CV 反向評分

### 修正項目

1. **法人資料 SQL 名稱**: FinMind 資料庫使用英文名稱 (`Foreign_Investor`, `Investment_Trust`, `Dealer_self`, `Dealer_Hedging`) 而非中文，修正 LIKE 條件
2. **法人評分閾值**: 調整至符合實際股份數量級 (foreign_net > 100M → +25pt)

---

## 執行結果

### Dry-Run (Step 7A)

```
PREFLIGHT PASS/WARN/FAIL   : 7/0/0
V0.2 CONTRACT PASS/WARN/FAIL: 16/4/0
total_candidates : 2799
core_universe    : 120 | convex_universe : 30
research_universe: 2271 | quarantine     : 378
written_rows : 0 (dry-run)
總計耗時 : 1114.73 ms
主權判定 : WARNING (exit 0)
```

### Commit (Step 7B)

```
PREFLIGHT PASS/WARN/FAIL   : 7/0/0
V0.2 CONTRACT PASS/WARN/FAIL: 16/4/0
written_rows : 5601
總計耗時 : 1798.95 ms
主權判定 : WARNING (exit 0)
```

### 批量資料載入

```
price_data   : 147 stocks
revenue_data : 148 stocks
financial_data : 148 stocks
institutional_data : 147 stocks
```

---

## 評分結果樣本（前 10 名 core+convex）

| 股票 | 名稱 | Tier | Score | DQ | LM | FG | TR | IF | VC | RP |
|------|------|------|-------|----|----|----|----|----|----|-----|
| 2408 | 南亞科 | convex | 94.6 | 99 | 100 | 100 | 100 | 90 | 20 | 0 |
| 8299 | 群聯 | convex | 90.6 | 99 | 98 | 100 | 100 | 55 | 20 | 0 |
| 6223 | 旺矽 | convex | 90.2 | 99 | 90 | 100 | 100 | 70 | 20 | 0 |
| 3260 | 威剛 | convex | 89.6 | 99 | 92 | 100 | 100 | 60 | 20 | 0 |
| 2330 | 台積電 | convex | 89.6 | 99 | 100 | 90 | 100 | 45 | 50 | 0 |
| 2303 | 聯電 | convex | 88.9 | 99 | 95 | 83 | 100 | 65 | 50 | 0 |
| 5274 | 信驊 | convex | 88.8 | 99 | 88 | 100 | 100 | 60 | 20 | 0 |
| 3105 | 穩懋 | convex | 88.8 | 99 | 92 | 90 | 100 | 70 | 20 | 0 |
| 4991 | 環宇-KY | convex | 88.6 | 99 | 84 | 100 | 100 | 70 | 20 | 0 |
| 4749 | 新應材 | convex | 87.6 | 91 | 83 | 99 | 100 | 60 | 65 | 0 |

**備註**: VC=20 表示 CV>0.4 (高波動)，台灣半導體股 2 年區間波動較大屬正常。  
IF=45 (台積電) 反映 2024-2026 年外資有淨賣超現象。

---

## 剩餘 4 項 Warnings（已知限制，皆為預期）

1. `price_coverage_252d` market scope: zero=2652/2799（全市場未同步，非 core 範圍）
2. `revenue_coverage_24m` market scope: zero=2651/2799（同上）
3. `financial_coverage_8q` market scope: zero=2651/2799（同上）
4. `TaiwanStockInfo as-of candidates=65 < 150`：latest_registry_fallback 模式（符合設計）

---

## Snapshot 資訊

| 項目 | 值 |
|------|-----|
| 新 Snapshot ID | `core_universe_20260514_core_universe_policy_v0_2` |
| Policy Version | `core_universe_policy_v0.2` |
| 舊 Snapshot ID | `core_universe_20260514_core_universe_policy_v0_1` (v0.1 保留) |
| written_rows | 5601 (policy 1 + snapshot 1 + membership 2799 + scores 2799 + revision 1) |

---

## 系統現況小結（2026-05-16 CoreScore v0.2 實作後）

| 狀態 | 評估 |
|------|------|
| v0.2 六層 CoreScore | ✅ 正式實作完成 |
| v0.2 input contract preflight | ✅ 16/4/0 (4 warnings 皆預期) |
| v0.2 coverage 門檻 | ✅ price 97.3% / revenue 93.3% / financial 98.0% |
| Snapshot committed | ✅ `core_universe_policy_v0.2` |
| core_universe | ✅ 120 支 |
| convex_universe | ✅ 30 支 |
| Feature Store | ⏳ 待建立 |
| 模型訓練 | ⏳ 待啟動 |
