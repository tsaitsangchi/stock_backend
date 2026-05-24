# §0.1 第一性原理「資料證據鏈」Gap 報告

- **產出日期**: 2026-05-24 22:15
- **產出者**: Claude Code (Sonnet 4.5) session
- **觸發原因**: 用戶要求驗證「核心股選擇是否真的有第一性原理的資料來源支撐」
- **憲章版本**: v6.1.0
- **本報告位階**: 揭露性實證報告(非治權變更);治權層改動見配套的 §0.1.3 / §6.3 / §6.4 補登

---

## 一、執行摘要

| 變數 | 治權聲稱 | 實作落地 | 證據鏈狀態 |
|---|---|---|---|
| **M** Liquidity Mass | 流動性質量 | `Trading_money` + 法人籌碼 pivot | 🟢 **完整(已升級文件化 2026-05-24)** |
| **V** Intrinsic Value Density | 「毛利率 + ROE + FCF + EPS + ...」(§0.1.3 原文) | **僅 Revenue YoY + EPS 累計 + 淨利 binary 三訊號** | 🟡 **不完整(7+ 死資料 columns)** |
| **ΔlnP** log price change | 上行無限/下行有界之凸性 | **完全不算 log-return,只用 CV(STDDEV/AVG)** | 🔴 **結構性缺陷(對稱壓制上下行)** |

**整體判定**:§0.1 之 F = f(M, V) × ΔlnP 在治權層為 aspirational,實作層 CoreScore v0.2 為 best-effort 近似;**V 達成度約 15%,ΔlnP 達成度約 0%(僅有絕對波動率代理)**。

---

## 二、M(Liquidity Mass)— 已驗證完整

### 證據鏈 5 層全綠

| 層 | 證據 | 來源 |
|---|---|---|
| 1. DDL 定義 | `Trading_money NUMERIC(20,6)` | `data_schema.py:101` |
| 2. DB 實表 | `Trading_money | numeric(20,6)` | psql 確認 |
| 3. SQL 取數 | `AVG("Trading_money"::numeric) as avg_daily_value` | `core_universe_builder.py:612` |
| 4. Score 計算 | `log10(avg_val) * 25 → value_score` | `core_universe_builder.py:713-721` |
| 5. 權重套用 | `0.25 * lm` | `core_universe_builder.py:864` |

### 衍生路徑(InstitutionalFlow)
- TaiwanStockInstitutionalInvestorsBuySell.{buy, sell, name} 三 column raw
- 透過 SQL CASE pivot 為 `foreign_net` / `trust_net` / `prop_net`
- `_institutional_flow_score()` 用 threshold 打分,10% 權重

### 副產出:憲章 §6.3 / §6.4 lookback 窗口顯式化
- 252 trading days 窗口從 builder code 的隱式變數 → 治權層明文(commit `a4fa6f2` / tag `v6.1.0-charter-lookback-explicit`)

---

## 三、V(Intrinsic Value Density)— 7+ 死資料 columns

### 治權 §0.1.3 vs 實作對照

| §0.1.3 聲稱 V 應包含 | DB 是否有? | Builder 實用? | 影響 |
|---|---|---|---|
| **毛利率** | ✅ `GrossProfit` + `Revenue`(2337 stocks) | ❌ 沒用 | 獲利能力指標缺失 |
| **ROE** | ✅ `NetIncome` + `EquityAttributableToOwnersOfParent`(2147 stocks) | ❌ 沒用 | 資本效率指標缺失 |
| **FCF** | ❌ 沒有 | — | 真實 raw 缺(FinMind 未提供) |
| **EPS** | ✅ `EPS` type(165K rows) | ✅ 用 eps_sum | OK |
| **獲利率矩陣**(GP/OI/NI/PreTax)| ✅ 全部都有 | ❌ 只用 NI binary | 獲利結構分解缺失 |
| **估值矩陣**(PER/PBR)| ✅ `TaiwanStockPER` 7.3M rows / 2016 stocks | ❌ **整張表沒讀** | 估值合理性缺失 |
| **股利政策** | ✅ `TaiwanStockDividend` 22 columns | ❌ **整張表沒讀** | 現金回饋穩定性缺失 |

### Builder 實際 FG scoring 邏輯(core_universe_builder.py:724-750)

```python
def _fundamental_gravity_score(self, stock_id, revenue_data, financial_data):
    """FundamentalGravity (20%): revenue growth YoY + profitability"""
    score = 50.0
    # 訊號 1: Revenue YoY(±25/±15/±8/-10/-20)
    yoy = r.get("yoy_growth", 0.0)
    if yoy > 0.30: score += 25.0
    # ...
    # 訊號 2: 淨利 binary(+15) OR 訊號 3: EPS 累計(+10)
    if f.get("net_income_positive", False): score += 15.0
    elif f.get("eps_sum", 0) > 0: score += 10.0
    # 訊號 4: coverage(*10)
    coverage = (revenue_cov + financial_cov) / 2.0
    score += coverage * 10.0
    return score
```

**只 3 個內容訊號 + 1 個 coverage 訊號 → 20% 權重**

### 死資料分類(分為兩類)

#### Class 1:已 sync raw data 但 builder 完全沒讀
- TaiwanStockPER 整張(PER / PBR / dividend_yield × 7.3M rows)
- TaiwanStockDividend 整張(22 columns)

#### Class 2:同表存在但只用部分 type
- TaiwanStockFinancialStatements 30+ type,builder 只用 EPS + 淨利 LIKE match
- 缺:GrossProfit / OperatingIncome / EquityAttributableToOwnersOfParent / CostOfGoodsSold / OperatingExpenses / PreTaxIncome 等 5+ key types

---

## 四、ΔlnP(log price change)— 結構性缺陷

### 治權 §0.1 vs 實作對照

| §0.1 概念 | 實作 | Gap |
|---|---|---|
| **ΔlnP**(對數價變) | **不存在**(builder 全檔 zero log_return)| 100% 缺失 |
| **凸性**(上行無限/下行有界) | CV(STDDEV/AVG)對上下行對稱 | **結構性缺陷** |
| **時間序列特性**(skewness, kurtosis) | 不計算 | 缺失 |

### Builder VC 實作(core_universe_builder.py:789-807)

```python
def _volatility_control_score(self, stock_id, price_data):
    """VolatilityControl (5%): 收盤價變異係數 CV，越低越好"""
    cv = p.get("cv_close", 0.3)
    if cv <= 0.05: return 95.0
    elif cv <= 0.10: return 85.0
    # ... 越低越好(對稱壓制波動)
    return 20.0  # cv > 0.40
```

### SQL 計算 cv_close(L611-613)

```sql
STDDEV("close"::numeric) / NULLIF(AVG("close"::numeric), 0) as cv_close
```

**這是傳統統計學的 Coefficient of Variation,不是 §0.1 的 ΔlnP**。差異:
| 指標 | 公式 | 上行/下行 |
|---|---|---|
| CV(現有) | `STDDEV(price) / MEAN(price)` | 對稱 |
| ΔlnP(§0.1 要求) | `ln(P_t) - ln(P_{t-1})` | 不對稱,可分 upside/downside |

### 已有的 log-return 在哪裡?

- ✅ L2 Feature Store(`scripts/core/feature_store_builder.py`)
- ✅ L2 Feature Engineering(`scripts/pipeline/feature_engineering.py` + `scripts/features/feature_engineering.py`)
- ❌ L1 CoreScore 完全不知道它存在

**架構斷層**:`L1 CoreScore(防守)→ L2 Feature(分離)→ L3 Model(線性)` 三層對「凸性」邏輯不一致。

---

## 五、為何重要(impact assessment)

### 直接影響:V 表達力不足
- FG 20% 權重事實上只代理 3 個訊號
- 「巴菲特派(看獲利率/ROE)」與「葛拉漢派(看 PER/PBR)」的洞察全部缺席
- 雷達股(高 EPS 但低 ROE 的紙上獲利)無法被識別

### 結構性影響:凸性壓制
- CV 5% 權重 = 對稱波動懲罰
- 高上行波動股(右尾候選)被視為「不穩定」→ CoreScore 降低 → 進不了 core_universe
- 與 §0.2 槓鈴策略「攻擊端要凸性」直接衝突
- 已在 §0.0-C.3 入憲為「凸性系統性壓制」既知缺陷

### 哲學影響:§0.1 變成 aspirational
- 「F = f(M, V) × ΔlnP」在治權層是響亮的物理形式
- 在實作層 V 是 yoy_growth + eps,ΔlnP 是 STDDEV(close)
- **不是 bug,是 best-effort 工程妥協**;但需誠實揭露

---

## 六、修補路徑(本報告不主張立即執行,只列選項)

### Phase A:**揭露 + 入憲**(0 程式變動,本報告 + §0.1.3 補登)— 進行中
- ✅ 本報告 → 寫畢
- ✅ 憲章 §0.1.3 補入「V 工程落地實況(2026-05-24 揭露)」子節
- ✅ 憲章 §0.1 補入「ΔlnP 實作缺席揭露」子節

### Phase B:**V v0.3 補完毛利率 + ROE**(1-2 天)
- builder.py 加 SQL pivot 毛利率 + ROE
- _fundamental_gravity_score() 加兩 sub-scores
- 跑 IC 驗證 → policy 升 v0.3

### Phase C:**整合 PER/PBR**(1-2 天)
- builder.py 新增 SELECT TaiwanStockPER
- FG 加估值相對性 sub-score(stock_PER vs sector_median_PER)

### Phase D:**Dividend 整合**(可選,2-3 天)

### Phase E:**CoreScore 加 ΔlnP 第七層**(2-5 天,風險高)
- 新增 `PriceConvexity = upside_log_return − downside_log_return`
- 對應真正的 §0.1 ΔlnP
- 可能取代或補強 VC
- 需 IC 驗證 + ablation study

---

## 七、與既有治權的相容性

本報告不改:
- §0.1 / §0.1.1 / §0.1-A / §0.1-E 原文(治權層)
- §6.3 / §6.4 CoreScore 公式與權重
- §6.7 SQL SSOT / §7 三層防禦 / §8 / §9 強制契約
- 任何程式碼(Phase A 純文件)

本報告新增揭露:
- V 死資料 7+ columns 清單
- ΔlnP 結構性缺陷
- §0.1.3 落地度量化估計(~15% V,~0% ΔlnP)

---

## 八、結論

§0.1 第一性原理的資料證據鏈在實證上 **「M 完整 / V 三分之一 / ΔlnP 缺席」**。憲章原文未明示「§0.1 的 V = 表 X 之 columns Y1/Y2/Y3」之 DDL mapping,使後繼者可能誤以為 CoreScore v0.2 之 FG 即為 §0.1.3 完整實作。

**建議**:本報告產出後,於 §0.1.3 補入「工程落地實況」子節,明文揭露此 gap,提供未來 v6.2.0+ 升版之 baseline。

---

*Report generated 2026-05-24 22:15 by Claude Code Sonnet 4.5 session*
*Triggered by user's explicit verification request "核心股的第一性原理是否有具體的資料來源"*
*Source data: scripts/core/core_universe_builder.py / scripts/core/data_schema.py / live PostgreSQL DB query*
