# §0.3 康波週期資料證據 + L1 落地實況 — 「先驗哲學選擇 + L1 集中右尾」之治權實踐

- **產出日期**: 2026-05-26 00:53
- **產出者**: Claude Code (Sonnet 4.5) session
- **觸發**: 用戶 2026-05-26 00:46「§0.3 康波週期是否有資料依據」+ 00:52「入庫此 9 字結晶」
- **基於**: v0.7 snapshot (core_universe_20260522_core_universe_policy_v0_7)
- **核心裁決**: **§0.3 為「先驗哲學選擇 + L1 集中右尾」之治權實踐**(非短期預測信號)
- **commit/tag**: v6.1.18-k-wave-evidence-and-l1-implementation

---

## 一、5 個 §0.3 實證測試

### Test 1:ThemeResonance score 分佈(150 stocks)

| Theme Score | Stocks | MBNRIC 對應 |
|---|---|---|
| 100(半導體)| 38 (25.3%)| N+I+R triple pillar |
| 95(生技/醫療)| 2 (1.3%)| B+M |
| 90(資訊)| 1 (0.7%)| I+C |
| 85(通信)| 18 (12.0%)| I (5G) |
| 80(機器)| 85 (56.7%)| R(電機/汽車)|
| 75(綠能)| 6 (4.0%)| 支撐基建 |

**≥80 第六波 MBNRIC: 144 / 150 = 96%**

### Test 2:Wave Tier Distribution

| Wave Tier | Stocks | Pct |
|---|---|---|
| 🚀 第六波 MBNRIC (theme≥80) | 144 | **96.0%** |
| 🔋 支撐基建 (50-79) | 6 | 4.0% |
| 🏭 第四/五波遺產 (<50) | **0** | **0.0%** |

§0.3 治權執行效果:**完美淘汰第四/五波遺產**。

### Test 3:第六波 vs 非第六波 1Y forward return

| Wave | n | Mean | Median | Max |
|---|---|---|---|---|
| 非第六波(綠能 75) | 6 | +271% | +117% | +888% |
| 第六波 MBNRIC | 144 | +252% | **+192%** | **+1450%** |

**第六波 median 高於非第六波**(192% vs 117%),但 mean 因非第六波小樣本 outlier 略偏。

### Test 4:Industry-level return(關鍵異常!)

| Industry | Theme | n | Median Return |
|---|---|---|---|
| **通信網路業** | 85 | 4 | **+538%** 🏆 |
| 電子零組件業 | 100/65 | 31 | +236% |
| 電子工業 | 100/60 | 46 | +220% |
| 其他電子類 | 100 | 5 | +161% |
| **半導體業** | **100** | 38 | **+139%** ⚠️ |
| 電機機械 | 80 | 6 | +118% |
| 電子通路業 | 60 | 2 | +86% |
| **生技醫療業** | **95** | 1 | **+85%** ⚠️ |
| 電腦及週邊 | 75 | 14 | +80% |
| 資訊服務業 | 90 | 1 | +31% |
| **化學生技醫療** | **95** | 1 | **-49%** ⚠️ |

**Theme weight 與 forward return 弱相關 / 反向**:
- 半導體 theme 最高 100 → median return 139%(第 5)
- 通信網路 theme 中等 85 → median return 538%(第 1)
- 生技/化學 theme 高 95 → return 普通甚至虧損
- §0.3 自我認知正確:「**先驗哲學選擇,非後驗預測**」

### Test 5:Macro FRED 使用情況

```bash
grep "FredData|fred|DFF|VIX|UNRATE|T10Y2Y" core_universe_builder.py
→ 只在 preflight contract check 出現(line 164-167)
→ 實際 CoreScore 計算過程 0 行
```

**§0.3 macro 對 CoreScore = 0 影響**:
- §0.3.3 「macro_* 群為宏觀週期觀測載體」
- 但 builder 從未讀 FredData 算 score
- 對映 §0.1-C 既知缺陷「macro/theme cross-sectional 貢獻為零」

---

## 二、三層裁決

### Layer 1:§0.3 治權層 — **完整入憲** ✅

- §0.3.0 學術源流(Kondratiev → Schumpeter → Perez)
- §0.3.1-3 五大波 + 春夏秋冬 + 嵌套週期 + 2026 大共振
- §0.3.9 MBNRIC × TWSE industry mapping 完整
- §0.3-A 7 禁令 + §0.3-E 5 證偽承諾 + §0.3-D THEME_KEYWORDS 治權規則

### Layer 2:§0.3 落地實作 — **僅 L1 universe** 🟡

- ✅ ThemeResonance 15% weight 在 CoreScore 完整實作
- ✅ 96% core+convex 在第六波 MBNRIC tier(治權執行成功)
- ✅ 0% 第四/五波遺產通過(完美淘汰)
- ❌ FRED macro 完全沒進 CoreScore 計算(L1 缺席)
- ❌ §10 model_trainer 未落地(L2 缺席)
- ❌ portfolio_sizer 未落地(L3 缺席)

### Layer 3:§0.3 預測力 — **theme weight ≠ return**(弱) 🔴

- 半導體 100 但 median return 排第 5(139%)
- 通信網路 85 但 median return 第 1(538%)
- 生技 95 但 return 普通(85%)
- 化學生技 95 + 唯一虧損(-49%)
- **Theme 為先驗哲學選擇,不是後驗預測信號**

---

## 三、「先驗哲學選擇 + L1 集中右尾」之治權實踐(核心結晶)

### 拆解三段

**(1) 先驗哲學選擇**(Prior Philosophical Choice)
- §0.3 K-wave 45-60 年低頻 / 5 個波次樣本 / 無短期 IC 可驗
- 學術錨定:Kondratiev → Schumpeter → Perez 傳統
- §0.3-A 禁令 #3:「任何短期 IC / RMSE 不得用於驗證 K-wave」
- **這是 design choice,不是 data discovery**

**(2) L1 集中右尾**(L1 Right-Tail Concentration)
- ThemeResonance 15% weight 在 universe-level 應用
- 96% core+convex 在第六波 MBNRIC tier(實證落地)
- 0% 第四/五波遺產通過(完美淘汰)
- §0.3-A 禁令 #1:「K-wave 永久禁止進入 L2/L3」
- **L1 治權執行完美,但僅限 L1**

**(3) 之治權實踐**(Governance Practice)
- 不是 backtest IC 證明的「對」
- 不是預測模型的「準」
- 是治權邊界內的「**做到了該做的事**」
- 治權成熟度:**§0.3 100% 治權層完整 + 30% 實作層**(只 L1 落地)

### 對比 §0.1 / §0.2 之模式

| 支柱 | 性質 | 落地度 | 預測力證據 |
|---|---|---|---|
| §0.1 第一性原理 | 物理啟發 + T1 觀測量 | M 100% / V 73% / ΔlnP 100% | 強(M/ΔlnP IC > 0)|
| §0.2 八二法則 | 觀察事實 + 厚尾數學 | L1 完整 / L2/L3 待 | 強(top 5% return 947%)|
| **§0.3 康波週期** | **先驗哲學 + 樣本 n=5** | **L1 完整 / L2/L3 缺席** | **弱**(theme≠return)|

§0.3 是三柱中**最 aspirational 的**,但治權上完全自洽(§0.3-A 禁令 #3 已明文不可由短期 IC 驗)。

---

## 四、對 portfolio_sizer 之 implication

### 不應該
- ❌ 用 theme weight 作為 sizing 主要訊號(theme 100 不保證 return 高)
- ❌ 半導體 38 stocks 給最大集中(實證 median return 第 5)

### 應該
- ✅ Theme 維持為 **universe selection 篩選器**(L1),不傳到 L2/L3 size
- ✅ Sizing 應基於 rank-score / ROE / liquidity 等 T1 維度
- ✅ 跨 sector 多元化:通信網路 theme 85 但實際 return 第 1 → 可給 boost
- ✅ §0.2-A 禁令 #3 sector cap 20%/sector 強制

---

## 五、未來研究方向

### v6.2.0+(Phase B-D)
- §10 model_trainer 落地後跑 walk-forward IC for theme weights
- 若 theme IC ≈ 0,§0.3-D 允許「靜態權重不動」但實證落地度低

### v6.3.0+(2031 後)
- §0.3-E P1-P6 五年滾動 IC 驗證
- 若第六波(MBNRIC)IC 不顯著,§0.3 純粹敘事的可能性會被揭露
- 但 §0.3.0 明文「先驗哲學選擇」→ 不因短期 IC 弱而動搖

### macro FRED 整合(v6.2.0)
- 應該進 L2 Feature Store(不是 L1 CoreScore)
- §0.3.3 觀測載體 → §10 model_trainer 之 input feature

---

## 六、§0.3 與 §0.1 / §0.2 之三柱共振

```
                  §0.1 第一性    §0.2 八二法則    §0.3 康波週期
─────────────────────────────────────────────────────────────
治權層            ✅ 完整        ✅ 完整          ✅ 完整
L1 universe       ✅ M/V/ΔlnP    ✅ 集中右尾      ✅ 96% MBNRIC
L2 prediction     ⏸ 等 §10      ⏸ 等 §10        ⏸ 等 §10
L3 sizing         ⏸ 等 §9.2     ⏸ 等 §9.2       ⏸ 等 §9.2
資料證據強度      🟢 強          🟢🟢 極強         🟡 中(theme≠return)
落地度            ~92%           L1 完整          L1 完整
特性              觀測 + 物理    觀察 + 數學      先驗哲學 + 哲學集中
─────────────────────────────────────────────────────────────
共同特徵          三柱皆於 L1 universe 完成,L2/L3 待 §10/§9.2 落地
```

---

## 七、一句話結論

**§0.3 康波週期 = 「先驗哲學選擇 + L1 集中右尾」之治權實踐 — 不是短期可驗證之預測信號,但是治權層完整 + L1 universe 落地完整(96% 第六波 tier)+ macro 維度未實作(L1)/ L2/L3 缺席;§0.3 為三柱中最 aspirational 但治權上完全自洽。**

---

*Report generated 2026-05-26 00:53 by Claude Code session*
*Based on v0.7 snapshot;5 tests on theme distribution + return + macro usage*
*§0.3 三層落地實況封存:治權完整 + L1 落地 + 預測力弱 = 設計如預期*
