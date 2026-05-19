# Sub-Wave Recursive Decomposition (SWRD) Hypothesis — 詳細推導與系統應用研究

- **generated_at**: 2026-05-19 Asia/Taipei
- **constitution**: `reports/系統架構大憲章_v6.0.0.md` §0.3 / §0.3-A / §0.3-E / §0.3.6
- **scope**: 探討「康波週期 1/4 階梯遞迴分解」是否能作為系統各層視窗選擇之統一框架
- **verdict**: ACCEPTED_AS_HYPOTHESIS_NOT_LAW
- **production impact**: 0（純研究；不改任何程式或預設行為）

---

## 一、研究動機

§0.3 既有 Kondratiev / Schumpeter / Perez 之長波理論，採信「**第六波 MBNRIC 2026-2070**」作為戰略錨。但**長波如何分解到日常 / 月度 / 年度的可操作尺度**，現有 §0.3.3 嵌套表（K-wave / Kuznets / Juglar / Kitchin）採混合 1/3 ~ 1/5 比例，**未提供統一數學框架**。

本研究探討「**1/4 階梯遞迴分解**」假說：

$$
L_n = \frac{L_{n-1}}{4}
$$

從 L0 = 40-60 年（康波）開始，逐層分解至 L3-L4 之中短期週期。研究目標：

1. 推導 L0 → L5 完整階梯
2. 對照真實產業 / 經濟 / 股價週期
3. 評估與既有 §0.3.3 嵌套理論之相容性
4. 映射至系統 §6 / §8 / §9 各層
5. 提出治權邊界（避免越界至 §0.3-A 禁令範圍）

---

## 二、SWRD 階梯完整推導

### 2.1 數學定義

設 L0 之中心值為 50 年（K-wave 平均），則：

| Level | 期長下界 | 中心值 | 期長上界 | 倍數 |
|---|---|---|---|---|
| L0 | 40 年 | **50 年** | 60 年 | × 1 |
| L1 | 10 年 | **12.5 年** | 15 年 | × 1/4 |
| L2 | 30 月 (2.5 yr) | **37.5 月 (3.125 yr)** | 45 月 (3.75 yr) | × 1/4 |
| L3 | 225 天 (~7.5 月) | **281 天 (~9.3 月)** | 337 天 (~11.25 月) | × 1/4 |
| L4 | 56 天 (~1.9 月) | **70 天 (~2.3 月)** | 84 天 (~2.8 月) | × 1/4 |
| L5 | 14 天 | **17.5 天** | 21 天 | × 1/4 |
| L6 | 3.5 天 | **4.4 天** | 5.3 天 | × 1/4（已進入日內 swing 區）|

每階皆為前階之 25%，類似 dyadic（二進制）分解之兩個 octave。

### 2.2 為何是 1/4 而非 1/2 或 1/3？

| 比例 | 對應 | 結果 |
|---|---|---|
| 1/2 (octave) | 音樂八度 | L0 50 年 → L1 25 年 → L2 12.5 年 → L3 6.25 年 → ... 太慢，過早撞 Kuznets/Juglar |
| 1/3 | 三和弦 | L0 50 → L1 16.7 → L2 5.6 → L3 1.9 → L4 0.6 年 = 7 月 → ... 中段過度密集 |
| **1/4** | 兩 octave / 季度（quarter） | L0 50 → L1 12.5 → L2 3.1 → L3 0.78 = 9.3 月 → **完美對應市場結構** |
| 1/5 | 黃金比例近似 | 階梯過陡，跳過實證週期 |

**1/4 之優勢**：
- 對應「**季度（quarter）**」之自然分割 — 與企業財報週期一致
- 每階提供 ~4 個子週期統計樣本（如 L1 12.5 年 ≈ 50/4，L2 約有 4 個出現於 L1 內）
- 對應 **wavelet decomposition 之 dyadic factor**（信號處理理論基礎）

---

## 三、與既有經濟學嵌套週期之比對

### 3.1 §0.3.3 既有嵌套表 vs SWRD

| 既有週期 | 期長 | SWRD 對應 | 相容性 |
|---|---|---|---|
| K-wave | 45-60 yr | **L0** (40-60) | ✅ 完全對應 |
| Kuznets | 15-25 yr | L0 / L1 之間 (12.5-25) | ⚠️ Kuznets 偏長，部分跨入 L0 下界 |
| Juglar | 7-11 yr | L1 (10-15) / L2 之間 | ⚠️ Juglar 偏短，跨入 L1 下界 |
| Kitchin | 3-5 yr | L2 (2.5-3.75) | ✅ 良好對應 |

**結論**：
- L0 ↔ Kondratiev：**強對應** ✅
- L2 ↔ Kitchin：**強對應** ✅
- Kuznets / Juglar **不是** SWRD 純粹倍數，而是其上下界之**疊加**

SWRD 對 Kuznets / Juglar 的解釋為：「Kuznets 是 L0 之 1/3 〜 L1 之 2 倍（中間態）；Juglar 是 L1 之 2/3 〜 L2 之 2 倍（中間態）」。即 **Kuznets / Juglar 不是獨立週期，而是 L0/L1/L2 之耦合共振**。

這是一個**理論性的提升**：把混亂的嵌套理論統一為單一遞迴框架。

### 3.2 與 Carlota Perez 4 階段之關係

每個 K-wave (L0) 內部 4 階段（Installation / Crash / Deployment / Maturity，平均 12.5 年 ≈ L1）：

$$
\text{Perez 階段長度} \approx L_1 = L_0 / 4
$$

**驚人對應**：Perez 之 4 階段恰好對應 SWRD 之 L1。

進一步推論：
- 每個 L1 階段內部又有 4 子階段 → L2 (~3 年)
- 每個 L2 子階段內部又有 4 子子階段 → L3 (~9 月)

**SWRD 可視為 Perez 4 階段理論之嚴格遞迴化**。

---

## 四、實證對照：產業 / 經濟 / 股價週期

### 4.1 L0 (40-60 年) 強證據

| 領域 | 觀察週期 | 對應 |
|---|---|---|
| 技術範式 | 5 個 K-wave (§0.3.1) | ✅ |
| 帝國週期 | 大英 1815-1914 / 美國 1945-? | ✅ |
| 信用超週期 | 全球債務超循環（1980-2030?）| ✅ |
| 房地產超週期 | Harrison 18 年 × 2-3 倍 = 36-54 年 | ✅ |

### 4.2 L1 (10-15 年) 強證據

| 領域 | 觀察週期 | 對應 |
|---|---|---|
| 房地產 | Harrison 18 年 / 亞洲 ~12-15 年 | ✅ |
| 信用循環 | 1998 / 2008 / 2020 危機間距 ~10 年 | ✅ |
| 大宗商品超循環 | 1970s / 2000s / 2020s ~20 年 (2×L1) | ✅ |
| 生技 / 製藥 | 藥物研發週期 10-15 年 | ✅ |
| 半導體製程世代 | ~10 年（28nm → 14nm → 7nm → ...）| ✅ |
| 中國五年計劃 × 2 | 10 年 | ✅ |

### 4.3 L2 (30-45 月) 中-強證據

| 領域 | 觀察週期 | 對應 |
|---|---|---|
| **半導體 silicon cycle** | **3-4 年（DRAM、邏輯）** | **✅ 強對應** |
| Kitchin 庫存循環 | 3-5 年 | ✅ |
| 美國 mid-term election | 4 年 | ✅ |
| 中國五年計劃 | 5 年 | ⚠️ 略長於 L2 上界 |
| 比特幣減半週期（已禁區）| 4 年 | ✅（但屬 §9.4 禁區）|

### 4.4 L3 (225-337 天) 中度證據

| 領域 | 觀察週期 | 對應 |
|---|---|---|
| 季節性 | Sell in May, Halloween effect ~6 月 | ⚠️ 比 L3 短 |
| 財報週期 | 4 季 × 3 月 = 12 月 | ✅ 接近 L3 上界 |
| 企業預算週期 | 12 月 | ✅ |
| **中型技術股 swing** | **6-12 月成長期 + 修正期** | **✅ 強對應** |
| **央行政策落差** | **升降息對實體經濟傳導 ~6-12 月** | **✅ 強對應** |
| 期權月份系列 | 序列循環 ~9 月 | ✅ |

### 4.5 L4 (56-84 天) 弱-中證據

| 領域 | 觀察週期 | 對應 |
|---|---|---|
| 月度資金流 | 30 天 | ⚠️ 短於 L4 |
| 季度 rebalance | 90 天 | ⚠️ 接近 L4 上界 |
| 動量反轉學術 | 60 天 | ✅ |
| 期權季月循環 | 90 天 | ⚠️ |

**裁決**：L4 已接近**雜訊主導區**；雖有部分對應但非主導機制。

### 4.6 L5 (14-21 天) 弱證據 / 禁區

| 領域 | 觀察週期 | 對應 |
|---|---|---|
| 週度資金流 | 7 天 | ⚠️ |
| 月內波段 | 5-15 天 | ✅ |
| 美聯儲 FOMC 間距 | ~42 天 | ❌ |
| **§9.4 第 7 條** | 高頻禁區 | **🔴 永久禁止** |

---

## 五、對映系統各層

### 5.1 對映表

| Level | 期長 | 系統載體 | §0.3-A 治權層 | 現行使用 |
|---|---|---|---|---|
| L0 | 40-60 yr | §6.4 ThemeResonance / MBNRIC 字典 | **L1 Universe** | ✅ 已落地 |
| L1 | 10-15 yr | §6.8 年度重選 × 10 年 walk-forward | **L1 Universe** | ⚠️ 部分（year-by-year）|
| L2 | 30-45 mo | `core_universe_builder` `lookback_730/financial_start_1000` | **L1 Universe (前置)** | ✅ 已落地 |
| L3 | 225-337 天 | §9.1 horizon=30 × multi-month walk-forward；24-point panel 跨度 | **L2 Tactical** | ✅ 已落地（§14.7-T） |
| L4 | 56-84 天 | (未使用)；可作為 backtest sub-period | **L2/L3 邊界** | ⚠️ 不在當前 scope |
| L5 | 14-21 天 | **永久禁區** | - | 🔴 §9.4 第 7 條 |

### 5.2 對既有實作之解釋力

**為何 `lookback_252` 是合理選擇？**
- 252 trading days ≈ 365 calendar days = 1 年
- 1 年 = L3 上界（337 天）+ 緩衝
- → 對應「**從 L3 延伸至 L2 下界**」之自然甜蜜點

**為何 `lookback_730` 是合理選擇？**
- 730 days = 2 年 ≈ L2 下界（2.5 年）邊緣
- → 對應「**L2 中下界**」之 backtest 視窗

**為何 horizon=30 是合理選擇？**
- 30 trading days ≈ 42 calendar days
- 42 天 < L4 (56-84) 之 75%
- → **接近 L4 下界**；屬戰術層（§0.3-A L2）合理 horizon

**為何 horizon > 60 應避免？**
- 60 trading days ≈ 84 calendar days = L4 上界
- > 60 即跨入 L3 季節性區，**信號被季節週期污染**
- ✅ 對應 §9.1 之 horizon=30 設計選擇

**24-point walk-forward panel 跨度**
- 2024-04 → 2026-04 = 24 月 = L2 中下界
- → 對應「**完整 L2 週期 × 1**」之穩定性測試窗

**驚人結論**：當前系統所有視窗選擇**幾乎都對應 SWRD 之自然甜蜜點**，但這是**隱含**而非**明文**。SWRD 入憲後可使這些選擇變為**有原則的設計**而非任意 hardcoded。

---

## 六、SWRD 框架對未來升版之指引

### 6.1 v6.1.0 production-current（不變）

- horizon=30 對應 L4 邊界 → 合理
- 不引入 L1 / L0 訊號（§0.3-A 禁令 #2）

### 6.2 v6.2.0 horizon=30 contract（§9.1）

- horizon=30 trading days = ~42 calendar days = L4 中段
- SWRD 解釋：屬戰術層；不得擴展至 L3 否則違反禁令

### 6.3 v6.3.0 portfolio sizing（§9.2）

- rebalance frequency 建議：年度（L3 上界）或季度（L4 上界）
- **不建議**月度或週度（L5 禁區）

### 6.4 v6.4.0 backtest engine

- 最長 backtest 視窗：12.5 年（L1 中心）
- 標準 backtest 視窗：3 年（L2 中心）
- **不建議** > 25 年（跨入前一 K-wave，違反 §0.3-A 禁令）

---

## 七、治權邊界（重要）

### 7.1 SWRD 是「框架」不是「預測法則」

任何把 SWRD 之**期長參數**（如「L2 = 30-45 月」）寫入 `prediction_engine.py` / `portfolio_sizer.py` 之嘗試，**皆違反 §0.3-A 禁令 #2**。

### 7.2 SWRD 之合法用途

- ✅ 解釋既有視窗選擇之合理性
- ✅ 引導未來視窗常數宣告化（§5.5 第 4 條）
- ✅ 提供跨層 IC 一致性驗證之理論基礎（§0.3-E）
- ❌ 不得作為 prediction model 之輸入特徵
- ❌ 不得作為 portfolio sizing 之動態權重
- ❌ L4 / L5 永久禁區（§9.4 第 7 條延伸）

### 7.3 入憲後之 audit 機制

`audit_doctrine_compliance.py` 未來 v0.3 升版時，可在 P1 / P3 增加：

- **SWRD_LEVEL_LEAKAGE_CHECK**：掃 prediction / sizing 模組是否引用 SWRD 期長常數
- **SWRD_L4L5_FORBIDDEN_CHECK**：掃任何模組是否使用 < 56 天 (L4) / < 14 天 (L5) 視窗
- **SWRD_CROSS_LAYER_IC_CHECK**：L2 / L3 panel 之 IC 應呈現 L3 stdev < L2 stdev（更穩定）

---

## 八、§0.3-E 可證偽承諾延伸

SWRD 入憲後可加入下列**可證偽指標**，避免淪為敘事：

| 指標 | 通過門檻（2026-2036 觀察期）| 不通過則裁決 |
|---|---|---|
| L0 K-wave 第六波驗證 | §0.3-E 既有指標 | §0.3-E 既有裁決 |
| **L1 階梯實證** | 滾動 10 年中，semiconductor cycle 平均長度 ∈ [10, 15] 年 | 修訂 SWRD 假說；不取消 |
| **L2 階梯實證** | 滾動 5 年中，半導體 silicon cycle 平均長度 ∈ [2.5, 3.75] 年 | 同上 |
| **L3 階梯實證** | walk-forward IC 在 ~8-11 月跨度展現穩定性高於 L2 | 同上 |
| **L4 / L5 雜訊驗證** | walk-forward 在 < 84 天視窗 IC 不穩定（stdev > L3 之 2 倍）| 確認 §9.4 禁令合理 |

驗證結果每 5 年發佈於 `reports/swrd_validation_<YYYY>.md`，與 §0.3-E 之 `kondratiev_validation_<YYYY>.md` 並列。

---

## 九、結論與建議

### 9.1 SWRD 假說之地位

- **理論強度**：⭐⭐⭐⭐（具數學優雅性 + 部分實證對應）
- **實證強度**：⭐⭐⭐（L0/L1/L2/L3 有對應；L4/L5 為禁區）
- **系統適用性**：⭐⭐⭐⭐⭐（解釋現有視窗選擇之合理性）
- **治權風險**：⭐⭐（必須明文標註為**假說**，並重申禁令 #2）

### 9.2 入憲方式

採「**框架性入憲**」：

- 新增 §0.3.6 章節（7 子節，本研究之精簡版）
- 明文「Hypothesis 非 Law」
- 延伸 §0.3-A 禁令至 SWRD 期長參數
- 延伸 §0.3-E 可證偽承諾至 SWRD 跨層驗證

### 9.3 不立刻改任何程式

當前系統視窗選擇**已隱含對映 SWRD 甜蜜點**：

- `lookback_252` ≈ L3 上界 + L2 下界
- `lookback_730` ≈ L2 中下界
- `financial_start_1000` ≈ L2 中段
- `horizon=30` ≈ L4 下界
- 24-point panel 跨度 ≈ L2 × 1

無需立刻修改；SWRD 入憲之主要價值是**理論統一**，不是**行為變更**。

### 9.4 後續演進

- **v6.2.0 升版時**：可選擇性引入 SWRD 標籤至 `audit_doctrine_compliance.py` P1 檢查
- **v6.3.0 portfolio_sizer**：rebalance frequency 建議 L3 上界（年度）或 L4 上界（季度）
- **2031 SWRD 第一次正式驗證**：5 年實證後產出 `swrd_validation_2031.md`

---

## 十、參考文獻 / 延伸閱讀

1. Kondratiev, N. D. (1925). *The Major Economic Cycles*. Moscow.
2. Schumpeter, J. A. (1939). *Business Cycles: A Theoretical, Historical, and Statistical Analysis*.
3. Perez, C. (2002). *Technological Revolutions and Financial Capital*.
4. Mandelbrot, B. & Hudson, R. (2004). *The (Mis)Behavior of Markets* — fractal market hypothesis.
5. Peters, E. (1994). *Fractal Market Analysis*.
6. Harrison, F. (2005). *Boom Bust: House Prices, Banking and the Depression of 2010*.
7. 系統架構大憲章 v6.0.0 §0.3.0〜§0.3.5 + §0.3-A〜§0.3-E。

---

## 附錄：SWRD 階梯速查表

```
L0  40-60 年      K-wave         ThemeResonance / MBNRIC 字典
└─ L1  10-15 年   Sub-K-wave    年度重選 × 10 年 walk-forward
   └─ L2  30-45 月  Mid cycle    builder lookback_730 / 1000
      └─ L3  225-337 天 Short cyc walk-forward 24-point panel
         └─ L4  56-84 天  ⚠️ 邊界 backtest sub-period (未使用)
            └─ L5  14-21 天 🔴 禁區 §9.4 第 7 條
```
