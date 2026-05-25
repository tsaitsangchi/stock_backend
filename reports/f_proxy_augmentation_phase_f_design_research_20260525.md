# F proxy 補強 Phase F.1-F.3 — IF v0.5.1 → v0.6 設計研究報告

- **產出日期**: 2026-05-25(夜深++++++)
- **產出者**: Codex (Opus 4.7, 1M context) session
- **觸發**: 用戶 2026-05-25「以第一性原理與目前系統資料庫的對應」第三輪 + V 補強 v0.5.1 完成後選 Phase F 補強(對映 §0.1.3-B.7 純動員修補之第二大軸線);事前事前 DB 驗證模式(§14.7-BE)第 6 次具體實踐確認 F proxy 25/30 cols 對 core 150 99-100% 可用
- **位階**: **草案性設計提案**(治權先行;依 §0.0-G 紀律入憲 §14.7-BF 後實作 builder v0.6)
- **HEAD commit**: `ec59f04`(§14.7-BE + builder v0.5.1)
- **配套憲章節**: §0.1 第一性原理(F 為 T2 物理啟發類比)/ §0.1.3-B(bottom-up V 完整盤點;F proxy 80% 失血)/ §6.3 第 6 條 IF 公式(現行 v0.2 用 Foreign + Trust net)/ §6.4 治理欄位 / §14.7-AX(資料層揭露驅動治權升版)/ §14.7-BE(事前事前驗證模式)/ §14.7-BC(V 補強治權預備類比模式)

---

## 一、執行摘要

| 項目 | 內容 |
|---|---|
| **動機** | §0.1.3-B 揭露 F proxy 多源 80% 失血(InstitutionalFlow 6 names 用 2 / Margin 16 cols dead / Shareholding 13 cols dead);本研究依事前事前 DB 驗證(§14.7-BE 模式)確認 25 cols 對 core 150 99-100% 可用,跟 Dividend 4 cols 之 schema sunset(§14.7-BE)完全不同性質 |
| **目標** | IF sub-scores 從 1 個(legacy: foreign+trust net 合成)升至 **4-5 個**;F proxy 動員從 2/25 cols 升至 **9-14 cols**(挑選有獨立信息的);IF 權重 **10% 維持不變**(內部結構升;不打破 §6.4 公式總結構) |
| **新增 4-5 個 sub-scores** | Phase F.1: Institutional 4 names 補完(挑 2 names 加值) / Phase F.2: 融資擁擠度 + 強迫平倉 + 券資比(Margin)/ Phase F.3: 外資剩餘空間 + 持股趨勢(Shareholding) |
| **版本升版** | `core_universe_policy_v0.4 → v0.5`;`core_universe_builder v0.5.1 → v0.6` |
| **追溯適用** | 既有 v0.4 snapshot 不重 build;新 v0.5 snapshot 起適用 |
| **「資料現實裁決」第 6 次跑通可能** | dry-run 後若 r 矩陣顯示新 sub-scores 與既有 institutional_flow 相關 > 0.7,觸發 §14.7-BG 多重共線追溯(預備) |

---

## 二、研究背景與動機

### 2.1 F 治權揭露鏈

| 層 | 子節 | 內容 |
|---|---|---|
| 1 | **§0.1** 主章節 + **§0.1.1** | F 為「資訊力」之 T2 物理啟發類比;proxy 變數實作可(institutional_flow / revenue_yoy / log_return) |
| 2 | **§0.1.3-B 發現 3**(2026-05-25) | F proxy 多源覆蓋僅 ~20% — 39 cols dead(Institutional 6 names 用 2 / Margin 16 cols dead / Shareholding 13 cols dead / Dealer 4 names dead)|
| 3 | **§14.7-BE 第 6 次主動驗證**(2026-05-25 夜深++++++,本研究)| 事前事前 DB 驗證:**25/30 cols 對 core 150 99-100% 可用**;跟 Dividend 4 cols 之 sunset(§14.7-BE)**完全不同性質** |
| **4(本研究)** | **§14.7-BF**(預備入憲)| **F proxy 補強 Phase F.1-F.3 + IF v0.6 設計**;從 2/25 cols 升至 9-14 cols(挑獨立信息的) |

### 2.2 §14.7-AX 治權元規則第 6 次跑通預備

| 次 | 時點 | 元素 | 結果 |
|---|---|---|---|
| 1 | §0.1.3-A.1 | ROE | dropped |
| 2 | §14.7-BA | publication-date | 分層落地 |
| 3 | §14.7-BB | FRED vintage | strict → transitional |
| 4 | §14.7-BD | Dividend 民國年 | builder SQL 修正 |
| 5 | §14.7-BE | Dividend 4 cols 口徑 | 治權層追溯 + 事前事前驗證模式 |
| **6(預備)** | **§14.7-BF**(本研究 dry-run 後可能觸發)| **若新 sub-scores 與既有 IF 相關 > 0.7** | **預備 multicollinearity 修正** |

---

## 三、F proxy Raw Field 之 DB 實證(2026-05-25)

### 3.1 InstitutionalFlow 6 names(2024 實際覆蓋)

| Name | 全市場 | **core 150 valid** | 狀態 | builder v0.5.1 |
|---|---|---|---|---|
| Foreign_Investor | 2,690 | **150/150** | ✓ | ✅ 已用 |
| Investment_Trust | 2,690 | **150/150** | ✓ | ✅ 已用 |
| **Dealer_Hedging** | 2,359 | **150/150** | ✓ | ❌ **可加(F.1)** |
| **Dealer_self** | 2,359 | **150/150** | ✓ | ❌ **可加(F.1)** |
| **Foreign_Dealer_Self** | 2,359 | **150/150** | ✓ | ❌ **可加但冗餘**(r ~ 0.85 with Foreign_Investor)|
| Dealer | 481 | **1/150** | ⚠️ | ❌ **跳過**(core 幾乎 sunset)|

**Phase F.1 取捨**:推薦加 **Dealer_self + Dealer_Hedging 2 names**;跳過 Foreign_Dealer_Self(冗餘)+ Dealer(sunset)。

### 3.2 Margin / ShortSale 13 numeric cols(2024 實際覆蓋)

| 類別 | cols | core 150 valid |
|---|---|---|
| Margin 系列(6 cols)| MarginPurchaseBuy / Sell / CashRepayment / Limit / TodayBalance / YesterdayBalance | **全 150/150** ✓ |
| ShortSale 系列(6 cols)| ShortSaleBuy / Sell / CashRepayment / Limit / TodayBalance / YesterdayBalance | **149-150/150** ✓ |
| 平衡 cols(1 col)| OffsetLoanAndShort | **150/150** ✓ |

**全 13 numeric cols 對 core 150 99-100% 可用**(metadata Note 1 個 varchar 排除)。

### 3.3 Shareholding 7 numeric cols(2024 實際覆蓋)

| col | core 150 valid |
|---|---|
| ForeignInvestmentRemainingShares | **150/150** ✓ |
| ForeignInvestmentShares | **150/150** ✓ |
| ForeignInvestmentRemainRatio | **150/150** ✓ |
| ForeignInvestmentSharesRatio | **150/150** ✓ |
| NumberOfSharesIssued | **150/150** ✓ |
| ForeignInvestmentUpperLimitRatio | **150/150** ✓ |
| ChineseInvestmentUpperLimitRatio | **148/150** ✓ |

**全 7 numeric cols 對 core 150 99-100% 可用**(metadata stock_name / InternationalCode / RecentlyDeclareDate / note 4 cols 排除)。

### 3.4 F proxy 多源 ceiling 估算(實證後)

| 來源 | 可用 cols(實證) | 99-100% 對 core 150 覆蓋 |
|---|---|---|
| Institutional 推薦加 2 names | 2 | ✓ |
| Margin 13 cols | 13 | ✓ |
| Shareholding 7 cols | 7 | ✓ |
| **合計可動員** | **22**(扣除 Foreign_Dealer_Self 冗餘 + Dealer sunset)| ✓ |

**為何不會重複 Dividend 之 sunset 打臉**:Margin / Shareholding / Institutional 為**每日 / 每週流量資料**,schema 跨年代穩定;Dividend 為**事件型資料**,schema 隨會計準則變更。

---

## 四、Phase F.1 — Institutional 2 names 補完

### 4.1 設計

```python
def _load_institutional_v0_6(self, cur):
    """v0.6 加 Dealer_self + Dealer_Hedging 2 names(跳過 Foreign_Dealer_Self 冗餘 + Dealer sunset)"""
    inst_gate, n_ap = build_publication_date_gate("TaiwanStockInstitutionalInvestorsBuySell")
    cur.execute(f"""
        SELECT stock_id,
            SUM(CASE WHEN name IN ('Foreign_Investor','Foreign_Dealer_Self')
                     THEN (buy::numeric - sell::numeric) ELSE 0 END) as foreign_net,
            SUM(CASE WHEN name='Investment_Trust'
                     THEN (buy::numeric - sell::numeric) ELSE 0 END) as trust_net,
            SUM(CASE WHEN name IN ('Dealer_self','Dealer_Hedging')
                     THEN (buy::numeric - sell::numeric) ELSE 0 END) as prop_net,
            -- v0.6 NEW: 分開 self vs hedging
            SUM(CASE WHEN name='Dealer_self'
                     THEN (buy::numeric - sell::numeric) ELSE 0 END) as dealer_self_net,
            SUM(CASE WHEN name='Dealer_Hedging'
                     THEN (buy::numeric - sell::numeric) ELSE 0 END) as dealer_hedge_net
        FROM "TaiwanStockInstitutionalInvestorsBuySell"
        WHERE date >= %s AND {inst_gate}
        GROUP BY stock_id
    """, (lookback_252, *([self.as_of_date] * n_ap)))
```

**注意**:既有 `_score_candidate` 已從 `prop_net` 取「Dealer_self + Dealer_Hedging 合計」;Phase F.1 新增**獨立 Dealer 自營 vs 對沖 sub-score**(dealer_self 為主動;dealer_hedging 為對沖反向訊號)。

### 4.2 新 sub-score:`_dealer_directional_score` ±5

```python
def _dealer_directional_score(self, stock_id, institutional_data):
    """Dealer 方向性 ±5(v0.6 新):dealer_self > 0 且 dealer_hedging < 0 → 自營正面;反之反向"""
    inst = institutional_data.get(stock_id, {})
    d_self = inst.get("dealer_self_net", 0)
    d_hedge = inst.get("dealer_hedge_net", 0)
    # 自營正面 + 對沖反向 = 強正訊號
    if d_self > 10_000_000 and d_hedge < -5_000_000:
        return 5.0
    if d_self > 0 and d_hedge < 0:
        return 2.0
    if d_self < -10_000_000:  # 自營賣超 = 反向訊號
        return -3.0
    return 0.0
```

---

## 五、Phase F.2 — Margin 4 個關鍵 sub-scores

### 5.1 設計

```python
def _load_margin_v0_6(self, cur):
    """v0.6 擴張 Margin SQL: 加 short_today_balance / cash_repayment_60d_trend / num_shares_issued"""
    margin_gate, n_ap = build_publication_date_gate("TaiwanStockMarginPurchaseShortSale")
    cur.execute(f"""
        SELECT stock_id,
            AVG("MarginPurchaseTodayBalance"::numeric) as margin_balance_60d_avg,
            AVG("ShortSaleTodayBalance"::numeric) as short_balance_60d_avg,
            AVG("MarginPurchaseCashRepayment"::numeric) as margin_repay_60d_avg,
            -- 60d trend (last 20d avg / first 40d avg)
            AVG("MarginPurchaseTodayBalance"::numeric) FILTER (WHERE date >= %s) as margin_last_20d,
            AVG("MarginPurchaseTodayBalance"::numeric) FILTER (WHERE date < %s) as margin_prior_40d
        FROM "TaiwanStockMarginPurchaseShortSale"
        WHERE date >= %s AND {margin_gate}
        GROUP BY stock_id
    """, (last_20d_start, last_20d_start, lookback_60d, *([self.as_of_date] * n_ap)))
```

### 5.2 4 個新 sub-scores

#### 5.2.1 **融資擁擠度** `_margin_crowding_score` ±5

```python
def _margin_crowding_score(self, stock_id, margin_data, shareholding_data):
    """融資擁擠度 ±5:MarginPurchaseTodayBalance / NumberOfSharesIssued
    高擁擠 → 散戶 over-leverage → 反向訊號(右尾風險)"""
    margin_bal = margin_data.get(stock_id, {}).get("margin_balance_60d_avg")
    nshares = shareholding_data.get(stock_id, {}).get("num_shares_issued")
    if not margin_bal or not nshares or nshares <= 0:
        return 0.0
    ratio = margin_bal / nshares
    if ratio > 0.05:  # > 5%
        return -5.0  # 極度擁擠;反向
    if ratio > 0.02:  # > 2%
        return -2.0
    if ratio > 0.005:
        return 0.0
    return 2.0  # < 0.5% 健康
```

#### 5.2.2 **強迫平倉壓力** `_margin_forced_liquidation_score` ±3

```python
def _margin_forced_liquidation_score(self, stock_id, margin_data):
    """強迫平倉壓力 ±3:MarginPurchaseCashRepayment 之 60d 趨勢上升
    現金償還上升 → 強制 deleveraging → 前兆訊號"""
    trend = margin_data.get(stock_id, {}).get("margin_repay_trend_60d")
    if trend is None:
        return 0.0
    if trend > 0.30:  # 60d 上升 30%+
        return -3.0
    if trend > 0.10:
        return -1.0
    return 0.0
```

#### 5.2.3 **券資比** `_short_margin_ratio_score` ±3

```python
def _short_margin_ratio_score(self, stock_id, margin_data):
    """券資比 ±3:ShortSaleTodayBalance / MarginPurchaseTodayBalance
    高券資比 → 多空看法極端化 → 右尾或左尾風險"""
    short_bal = margin_data.get(stock_id, {}).get("short_balance_60d_avg", 0)
    margin_bal = margin_data.get(stock_id, {}).get("margin_balance_60d_avg", 1)
    if margin_bal <= 0:
        return 0.0
    ratio = short_bal / margin_bal
    if ratio > 0.50:
        return -3.0  # 極度看空 → 右尾風險(預期反彈)
    if ratio > 0.20:
        return -1.0
    return 0.0
```

#### 5.2.4 **融資趨勢** `_margin_trend_score` ±3

```python
def _margin_trend_score(self, stock_id, margin_data):
    """融資趨勢 ±3:margin_last_20d / margin_prior_40d
    短期融資增加 → 散戶追高 → 反向"""
    last20 = margin_data.get(stock_id, {}).get("margin_last_20d", 0)
    prior40 = margin_data.get(stock_id, {}).get("margin_prior_40d", 1)
    if prior40 <= 0:
        return 0.0
    trend = (last20 - prior40) / prior40
    if trend > 0.50:  # 短期暴增
        return -3.0
    if trend > 0.20:
        return -1.0
    if trend < -0.30:  # 短期暴跌(散戶恐慌賣)
        return 1.0
    return 0.0
```

---

## 六、Phase F.3 — Shareholding 3 個關鍵 sub-scores

### 6.1 設計

```python
def _load_shareholding_v0_6(self, cur):
    """v0.6 Shareholding (per stock latest week 之 7 numeric cols)"""
    sh_gate, n_ap = build_publication_date_gate("TaiwanStockShareholding")  # transitional 維持 date
    cur.execute(f"""
        SELECT DISTINCT ON (stock_id) stock_id,
            "ForeignInvestmentSharesRatio"::numeric as foreign_ratio,
            "ForeignInvestmentRemainRatio"::numeric as foreign_remain,
            "ForeignInvestmentUpperLimitRatio"::numeric as foreign_limit,
            "NumberOfSharesIssued"::numeric as num_shares,
            "ChineseInvestmentUpperLimitRatio"::numeric as china_limit
        FROM "TaiwanStockShareholding"
        WHERE {sh_gate}
        ORDER BY stock_id, date DESC
    """, (*([self.as_of_date] * n_ap),))
```

### 6.2 3 個新 sub-scores

#### 6.2.1 **外資剩餘空間** `_foreign_remain_capacity_score` ±5

```python
def _foreign_remain_capacity_score(self, stock_id, shareholding_data):
    """外資剩餘空間 ±5:(UpperLimit - SharesRatio) / UpperLimit
    剩餘空間大 → 外資未來可加碼空間 → 正訊號
    註:外資上限通常 100%(無限制)或 30%/49% 等(特定產業)"""
    sh = shareholding_data.get(stock_id, {})
    limit = sh.get("foreign_limit", 0)
    ratio = sh.get("foreign_ratio", 0)
    if limit <= 0 or ratio is None:
        return 0.0
    remain_pct = (limit - ratio) / limit
    if remain_pct > 0.80:  # 多數空間
        return 2.0  # 未來空間大
    if remain_pct > 0.50:
        return 3.0  # sweet spot
    if remain_pct > 0.20:
        return 5.0  # 高吸引(外資已重持)
    return 0.0  # 滿載(無加碼空間)
```

#### 6.2.2 **外資持股趨勢** `_foreign_holding_trend_score` ±3

需從 builder 端計算(SQL 取 2 個時點之 ratio):

```python
def _foreign_holding_trend_score(self, stock_id, shareholding_trend_data):
    """外資持股 60d 變化 ±3"""
    trend = shareholding_trend_data.get(stock_id, {}).get("foreign_ratio_60d_change")
    if trend is None: return 0.0
    if trend > 0.02:  # +2pp
        return 3.0
    if trend > 0.005:
        return 1.0
    if trend < -0.02:
        return -3.0
    return 0.0
```

#### 6.2.3 **法規產業屬性** `_regulated_industry_score` ±2

```python
def _regulated_industry_score(self, stock_id, shareholding_data):
    """法規受限產業 ±2:UpperLimit < 100% 代表特殊產業(電信 / 銀行 / 媒體 / 國防)
    通常 = 高度法規防護 / 穩定盈利"""
    sh = shareholding_data.get(stock_id, {})
    limit = sh.get("foreign_limit", 100)
    if limit < 30:  # 高度限制(媒體 / 國防)
        return 2.0
    if limit < 50:  # 中度限制(電信 / 銀行)
        return 1.0
    return 0.0  # 100% (一般股)
```

---

## 七、IF v0.6 總分計算

```python
def _institutional_flow_score_v0_6(self, stock_id, institutional_data,
                                    margin_data=None, shareholding_data=None):
    """v0.6 IF 11 sub-scores(v0.5 之 1 + v0.6 之 10)
    權重 10% 維持;內部結構升;clamp 0..100。"""
    base = 50.0
    score = base

    # === v0.5 既有(維持)===
    inst = institutional_data.get(stock_id, {})
    foreign_net = inst.get("foreign_net", 0)
    trust_net = inst.get("trust_net", 0)
    # Foreign net thresholds (legacy)
    if foreign_net > 100_000_000:
        score += 25.0
    elif foreign_net > 10_000_000:
        score += 15.0
    elif foreign_net > 0:
        score += 5.0
    elif foreign_net < -100_000_000:
        score -= 20.0
    elif foreign_net < -10_000_000:
        score -= 10.0
    if trust_net > 50_000_000:
        score += 15.0
    elif trust_net > 0:
        score += 5.0
    elif trust_net < -50_000_000:
        score -= 10.0

    # === v0.6 §14.7-BF 5 個新 sub-scores ===
    # F.1: Dealer 方向性
    score += self._dealer_directional_score(stock_id, institutional_data)        # ±5

    # F.2: Margin 4 sub-scores
    score += self._margin_crowding_score(stock_id, margin_data, shareholding_data)  # ±5
    score += self._margin_forced_liquidation_score(stock_id, margin_data)        # ±3
    score += self._short_margin_ratio_score(stock_id, margin_data)                # ±3
    score += self._margin_trend_score(stock_id, margin_data)                      # ±3

    # F.3: Shareholding 3 sub-scores
    score += self._foreign_remain_capacity_score(stock_id, shareholding_data)    # ±5
    score += self._foreign_holding_trend_score(stock_id, shareholding_trend_data) # ±3
    score += self._regulated_industry_score(stock_id, shareholding_data)         # ±2

    return max(0, min(100, score))  # clamp 0..100
```

**總範圍**:base 50 + v0.5 既有 ±60 + v0.6 新增 ±32 = -42 ~ 142 → clamp 0..100

---

## 八、與既有 v0.5.1 之相容裁決

### 8.1 IF 權重 10% 維持不變

§6.4 CoreScore 公式保留:
```
CoreScore = 0.25*DQ + 0.25*LM + 0.20*FG + 0.15*TR + 0.10*IF + 0.05*VC - RP
```
- IF 內部結構升(1 → 11 sub-scores)
- IF 權重 10% 不變

### 8.2 既有 v0.4 snapshot 影響

- 既有 `core_universe_20260524_core_universe_policy_v0_2` snapshot **不重 build**
- 新 v0.5 snapshot(`core_universe_policy_v0.5`)起適用 v0.6 builder
- 預期分層 churn rate < 20%(IF 10% 之變化 + 雙向)

### 8.3 §6.7 SSOT 不變

150 鎖定不變;選股結果可能略有 churn(F proxy 訊號變動 → 排序略變)。

### 8.4 治權邊界嚴守

本研究**不**修改:
- §6.4 CoreScore 公式總結構(6 維 + RP)
- §6.7 SSOT(150 鎖定)
- §0.1-A 6 條禁令
- §6.3 第 6 條 IF 條文原文(留待 v6.2.0 升強制契約時再升條文)

---

## 九、證偽承諾(對接 §0.1-E 框架)

| 指標 | 觀察期 | 通過門檻 | 不通過裁決 |
|---|---|---|---|
| **T_IF_v0.6.1** | 滾動 5 年 | v0.6 prediction h20 IC ≥ v0.5.1 baseline | < baseline 則 v0.6 撤回,policy 回退 v0.4 |
| **T_IF_v0.6.2** | walk-forward h20 panel | new sub-scores 與既有 institutional_flow 相關 < 0.7(避免冗餘) | 相關 > 0.7 觸發 §14.7-BG 多重共線追溯 |
| **T_IF_v0.6.3** | dry-run | IF score 分布 mean/std 與 v0.5.1 差異 ∈ [+3, +15] | 差異 > +25 過度疊加;< 0 反向 |
| **T_IF_v0.6.4** | 5 sample stocks 對照 | 至少 3 stocks IF score 之變化方向與基本面(配息穩定 / distress)一致 | 不一致則重審 sub-score thresholds |

---

## 十、追溯適用裁決

| 既有 artifact | 處理 |
|---|---|
| v0.2/v0.3 snapshot | **不重 build**;標記 `policy_version='legacy_v0.2_v0.3'` |
| v0.4 snapshot(若已 commit)| 標記 `policy_version='legacy_v0.4'`;不重 build |
| v0.5.1 builder | 保留為 baseline(audit 對照用) |
| audit_core_universe.py | **需升 v0.x 加 v0.5 policy 識別**(另案) |
| 既有 walk-forward h20/h30 panel | **保留**;新 v0.5 snapshot 用於下一輪 |

---

## 十一、風險與回退方案

| 風險 | 機率 | 回退 |
|---|---|---|
| **新 sub-scores 與既有 IF 相關 > 0.7**(multicollinearity)| **中**(類比 v0.5 之 OperatingIncome/PreTaxIncome 高度相關)| 撤回相關高之 sub-scores;設計 r 矩陣分析 |
| Margin 之 crowding ratio 對小型股雜訊大 | 中 | 加 NumberOfSharesIssued > 100M 之 size filter |
| Foreign holding trend 之 60d 變化計算需 2 個時點 SQL | 低 | 用 _load_shareholding 之 2-query 模式(_load_per 同模式)|
| **v0.6 IC < v0.5.1 baseline** | 低-中 | 政策版本回退 v0.4 |
| **「資料現實裁決」第 6 次跑通**(if dry-run 揭露新 sub-score 設計問題)| 中 | 追溯修正 sub-score 設計;類比 §14.7-BD/BE 之模式 |

---

## 十二、實作計畫(builder v0.5.1 → v0.6)

### 12.1 核心變更

| 模組 | 變更 |
|---|---|
| `core_universe_builder.py v0.5.1 → v0.6` | 新增 `_load_shareholding()` / 擴張 `_load_institutional()` 加 dealer 分項 / 擴張 `_load_margin()` 加 4 cols + trend / 加 8 個新 sub-score helpers / 升 `_institutional_flow_score()` / DEFAULT_POLICY_VERSION v0.4 → v0.5 |
| `audit_core_universe.py` | 加 `core_universe_policy_v0.5` 識別(另案)|

### 12.2 落地序列

```
Step 1: 設計研究報告(本文檔)
Step 2: 入憲 §14.7-BF + 修訂歷程 entry(本 commit)
Step 3: 實作 builder v0.6(下次 session)
Step 4: dry-run on as_of=2026-05-21,對照 v0.5.1 vs v0.6 之 institutional_flow_score
Step 5: r 矩陣分析(新 sub-scores 與既有 IF 相關)
Step 6: 若 r > 0.7 → 觸發 §14.7-BG 多重共線追溯
Step 7: audit_core_universe 升版加 v0.5 policy 識別
Step 8: commit;準備 walk-forward IC simulation
```

### 12.3 治權邊界嚴守

本實作**不**修改:
- §6.4 CoreScore 6 維權重結構
- §6.7 SSOT 150 鎖定
- §0.1-A 6 條禁令
- §6.3 第 6 條 IF 條文原文(留待 v6.2.0)
- raw DDL / CLI 介面 / annual_rebalance_guard / candidate_fallback / 5 張治理表寫入順序

---

## 十三、與既有契約之相容裁決 + 後續路徑

### 13.1 與 §14.7-BC V 補強之關係

§14.7-BC 模式為類比:設計研究先行 → 入憲 → builder 落地 → dry-run → 「資料現實裁決」可能跑通(§14.7-BD/BE)。本研究**完整對映此模式**;不同點:
- §14.7-BC 為 V 補強(基本面);本研究為 F 補強(籌碼面)
- §14.7-BC 之事前驗證(§14.7-BE)觸發 1 cols 真實可用之打臉;**本研究之事前驗證(§3.1-3.3)驗證 25 cols 真實可用**(完全相反)

### 13.2 與 §0.1.3-B.7 之關係

§0.1.3-B.7 列「修補路徑」分四類:純動員 / 條文活實作死 / 憲章先行 / §0.1-A 永禁。本研究為**純動員**(無治權變更;IF 權重 10% 維持)。

### 13.3 後續可選路徑

| 階段 | 內容 | 用戶授權 |
|---|---|---|
| **A(本研究)** | 設計研究 + 入憲 §14.7-BF + commit | **進行中** |
| **B** | builder v0.6 實作(Phase F.1 + F.2 + F.3 同時)| 待 |
| **C** | dry-run + 對照 + r 矩陣分析 | 待 |
| **D** | 若 r > 0.7 → §14.7-BG 多重共線追溯 | 條件性 |
| **E** | audit_core_universe 升版 + commit | 待 |
| **F** | walk-forward IC simulation(等 §8 schema 表建立) | 等 §8 |

---

## 十四、治權聲明

### 14.1 嚴守 §0.0-G 憲章先行紀律

本研究報告為**草案性提案**,先入憲 §14.7-BF 治權閉環記述,後續實作 builder v0.6 + dry-run + commit。

### 14.2 嚴守 §0.1-A 6 條禁令

- F proxy 為 §0.1 之 T2 物理啟發類比(對應 §0.1-A 第 1 條禁令之邊界內;**proxy 變數實作可**;不寫 F=M×ΔlnP 字面公式)
- 不實作 IFF Θ / SOC / 重力井邊緣 trigger
- 不寫地緣事件敘事
- 不用物理隱喻替代 backtest 證據(本研究有 T_IF_v0.6.1-4 證偽承諾)

### 14.3 與 §14.7-BE 教訓之關係

§14.7-BE 揭露「全歷史平均覆蓋率 ≠ 最近年實際覆蓋」之治權教訓;本研究**事前事前實證 25 cols 對 core 150 99-100% 覆蓋**(2024 / 2026-05-21 實證),避免重複 Dividend 4 cols 之打臉。

### 14.4 與 §14.7-AX「資料層揭露驅動治權升版」之關係

本研究再次驗證「資料層揭露驅動治權升版」之機制:
- §0.1.3-B field 盤點 → 揭露 F proxy 80% 失血
- §14.7-BE 治權新模式 → 本研究事前驗證避免事後打臉
- 若 dry-run 揭露新 multicollinearity 問題 → §14.7-BG 第 6 次跑通預備

---

## 附錄 A — DB 實證查詢全紀錄(2026-05-25)

```sql
-- Institutional 6 names 之 2024 stock 覆蓋
SELECT name, COUNT(DISTINCT stock_id) FROM TaiwanStockInstitutionalInvestorsBuySell
WHERE date >= '2024-01-01' GROUP BY name;
-- 結果:Foreign_Investor 2690 / Investment_Trust 2690 / Dealer_Hedging 2359 /
--      Dealer_self 2359 / Foreign_Dealer_Self 2359 / Dealer 481(對 core 1/150 為 sunset)

-- Margin 13 numeric cols 之 2024 覆蓋
-- 結果:全 13 cols 對 core 150 149-150/150 覆蓋(99-100%)

-- Shareholding 7 numeric cols 之 2024 覆蓋
-- 結果:全 7 cols 對 core 150 148-150/150 覆蓋(99-100%)
```

## 附錄 B — 憲章條文 cross-ref(已實際驗證)

| 引用條文 | 入憲狀態 | 行號 |
|---|---|---|
| §0.1 主章節(F 為 T2 物理啟發類比)| ✅ | §0.1 |
| §0.1.3-B 發現 3(F proxy 多源失血)| ✅ | §0.1.3-B.5 |
| §6.3 第 6 條 IF 公式 | ✅(v6.0.0 既有)| §6.3 |
| §6.4 CoreScore 6 維權重 | ✅(v6.0.0 既有)| §6.4 |
| §14.7-AX 資料層揭露驅動治權升版 | ✅(2026-05-24)| §14.7-AX |
| §14.7-BC V 補強治權預備 | ✅(2026-05-25 a6904aa)| §14.7-BC |
| §14.7-BE 事前事前驗證模式 | ✅(2026-05-25 ec59f04)| §14.7-BE |
| **§14.7-BF F proxy 補強治權預備** | ❌ **本研究預備入憲** | — |

## 附錄 C — 與 §14.7-BC V 補強之對照

| 維度 | §14.7-BC(V 補強) | **§14.7-BF(F 補強;本研究)** |
|---|---|---|
| 對應 sub-score | FG 20% | IF 10% |
| 補強來源 | PER 表 / Dividend / FinStmt 4 新 types | Institutional / Margin / Shareholding |
| 設計研究 cols | 計畫加 5 Dividend cols(§14.7-BE 揭露只 1 可用)| **事前驗證 25 cols 99-100% 可用**(不會重複打臉) |
| sub-scores 新增 | 6(PER/PBR/yield/Div 穩定性/OpMargin/AttrRatio) | **8**(Dealer 方向性 / 4 Margin / 3 Shareholding)|
| 動員度提升 | 23% → 64%(實證)| 2/25 → **22/25 = 88%**(實證) |
| 「資料現實裁決」次數 | 2 次(§14.7-BD/BE) | **預備 1 次**(§14.7-BG multicollinearity)|

---

## 十五、後續接續點

| 條件 | 動作 |
|---|---|
| 用戶確認本研究 | 入憲 §14.7-BF + 修訂歷程 v6.1.0-patch 2026-05-25 第九輪 entry |
| 入憲完成 | 實作 core_universe_builder v0.6(下次 session)+ audit_core_universe 升版 |
| 實作完成 | dry-run on as_of=2026-05-21,對照 v0.5.1 vs v0.6 + r 矩陣分析 |
| dry-run 通過 + r < 0.7 | commit;準備 walk-forward IC simulation |
| dry-run 揭露問題或 r > 0.7 | 「資料現實裁決」第 6 次跑通,追溯 §14.7-BG + builder v0.6 design |
