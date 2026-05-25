# F 升 §0.1 T1 治權決策研究 — 最終裁決報告

- **產出日期**: 2026-05-25 22:25
- **產出者**: Claude Code (Sonnet 4.5) session(獨立研究,Explore agent 拒做 synthesis)
- **觸發**: 用戶 2026-05-25 22:18「自己去研究出個最佳化結論」
- **決策性質**: 治權框架重構級(可能改 §0.1 第一性原理公式)
- **裁決**: ❌ **不立即升 T1** / ✅ **走 Phase A 研究路徑**(類比 §0.1.3 V 升 T1 之先例)
- **後續評估時點**: 2026-06-13(v6.1.1 production-current h20 gate 解除)後 + §10 walk-forward IC 完成

---

## 一、執行摘要(TL;DR)

| 維度 | 結論 |
|---|---|
| **F 是否應升 T1?** | **否(現在)** / 可(未來 walk-forward IC 通過後) |
| **IF 是否可升 T1?** | **是**(類比 §0.1.3 V 升 T1 — 都是 observable proxy → T1) |
| **§0.1 公式是否要改?** | **否**(維持 `F = f(M, V) × ΔlnP`)|
| **CoreScore 是否要動?** | **否**(IF 10% weight 維持) |
| **入憲建議** | 寫 design research(本檔)+ 預備 §14.7-BK / §0.1.4(待後續) |
| **核心發現(實證)** | r(IF, core_score) = **-0.091** → IF 對當前 CoreScore 影響極小,升 T1 即刻收益 ≈ 0 |
| **真實價值** | 為未來 §10 model_trainer 之 IF 作為 input feature 鋪治權路 |

---

## 二、§0.1 框架現狀與「升 T1」之治權含義

### §0.1 現行公式

```
F (Information Force) = f(M, V) × ΔlnP
```

- **F**:資訊力(T2,§0.1.1 明文「無第一性可觀測物理量」)
- **M**:流動性質量(T1 — `TaiwanStockPriceAdj.Trading_money`)
- **V**:內在價值密度(**T1**,§0.1.3 加入)
- **ΔlnP**:對數價格位移(T1)

### 「F 升 T1」之精確含義(用戶提問之解讀)

**澄清**:F 本身已是 T2(§0.1.1 明文);升 T1 之真實意圖是把 **IF(Institutional Flow)作為 T1 input 變數**。

```
若採此解讀,§0.1 公式變為:
   F = f(M, V, IF) × ΔlnP    ← Option A(類比 §0.1.3 加 V)

或:
   F = f(M, V) × IF × ΔlnP   ← Option B(IF 為資訊乘數)

或:
   F = f(M × IF, V) × ΔlnP   ← Option C(IF 為 M 之放大器)
```

**推薦 Option A**(若升):與 §0.1.3 V 加入完全對稱,治權路徑清晰。

### 治權位階深層含義(§0.1.3 V 之精確 mapping)

§0.1.3 對 V 之三層裁決樣板:

| V 等級 | 裁決 | 對應 IF 升 T1 之 implication |
|---|---|---|
| V 變數本身 | **T1 第一性** | 同樣:IF 變數本身(機構淨買賣 raw)可達 T1 |
| V 之計算方式(如 P/V 比率) | **T2 物理啟發類比** | 同樣:IF 之 12 sub-scores 加總公式為 T2 |
| V 之絕對「真值」 | **T3 操作隱喻** | 同樣:IF 之「資訊力真值」為 T3 |

**結論**:IF 升 T1 在治權邏輯上 **可行**,沿 §0.1.3 V 之先例。

---

## 三、5 條治權支持論點 🟢

### 支持 1:動員度實證強(IF 88% > V 64%)

```
IF: 12 sub-scores / 22 cols 動員 / 88% 覆蓋率
V:  12 sub-scores / 14 cols 動員 / 73% 覆蓋率(今晚 ROE 解鎖後)
```

若 V 已升 T1(§0.1.3),IF 動員度更高之 case 更應升。

### 支持 2:§0.1-C 已正式承認 IF 為 §0.1 物理基礎

§0.1-C(L1670-1690)明文:

> 「F 之 fundamental + institutional 為現有實作之物理基礎(**雖未明列於 §0.1 三變數模型**)」

且實證:`institutional` 群 ablation drop_minus_full = h20 **-0.0210** / h30 -0.0162(stable contribution)。

**意涵**:IF 之 §0.1 對應性已實證,只差「正式升 T1」之治權手續。

### 支持 3:§0.1.3 V 升 T1 之治權先例完美 mirror

| 步驟 | V(2026-05-19 完成) | IF(2026-05-25 候選) |
|---|---|---|
| 1. 揭露 ablation 實證 | §14.7-U fundamental drop_minus_full -0.0226 | §14.7-U institutional drop_minus_full -0.0210 |
| 2. 設計研究 | §0.1.3 加 V 之 4-variable model 補強 | 本檔(§0.1.4 加 IF 之 5-variable model 提案)|
| 3. T1 變數本身入憲 | V → T1 ✅ | IF → T1?(待決) |
| 4. 計算方式仍 T2 | V 計算為 P/V 比率屬 T2 | IF 12 sub-scores 加總為 T2 |
| 5. CoreScore 不改 | §6.4 FG 20% weight 維持 | §6.4 IF 10% weight 維持 |

**結構完全對稱**。

### 支持 4:§14.7-BF Phase F.1-F.3 已完整落地

2026-05-25 Codex 的 §14.7-BF 已將 IF sub-scores 從 1 → 12(+8 新),動員度 2/25 → 22/25 = 88%。**程式層已就緒**,只需治權層正式承認。

### 支持 5:與 §0.2 八二法則完美共振

機構資金流(IF)之分布為冪律(法人持股集中於少數標的)→ 對應 §0.2 八二法則 → §9.2 portfolio top-N 集中 → 全鏈一致。

升 IF 為 T1 強化 §0.1 ↔ §0.2 ↔ §9.2 之治權連續性。

---

## 四、5 條治權反對論點 🔴

### 反對 1:§0.1.1 明文裁定 F 為 T2

§0.1.1(L1340-1361)明文:

| 等級 | 元素 | 評語 |
|---|---|---|
| **T2 物理啟發類比** | **資訊力 F** | ⚠️ 概念存在,**但無第一性可觀測物理量**;計算依賴 proxy 變數 |

升 IF 為 T1 等於**部分覆寫 §0.1.1 T2 既有裁決**;需明文修訂 §0.1.1 表格,治權成本不低。

### 反對 2:IF 動員 = 多 sub-scores 加總 ≠「直接觀測物理量」

T1 元素標準:「可直接觀測;單位明確(TWD / 股)」。

IF 雖然底層 raw data 是直接觀測(`name='Foreign_Investor' buy/sell`),但 builder 之 IF sub-score 是 **12 個 derived metrics 之 weighted aggregation**(Dealer 方向 / Margin 4 項 / Shareholding 3 項...),並非單一直接觀測量。

對比 V:V 之 sub-scores 也是 derived(Revenue YoY / PER industry-relative...),但 V 升 T1 時 §0.1.3 明文裁定「**V 變數本身**為 T1,**V 計算方式**為 T2」。

IF 若升 T1 需明確區分「IF 變數本身(機構淨買賣 raw)」vs「IF 之 12 sub-scores 加總」— **治權上需做兩層裁決**,不是單純升一級。

### 反對 3:IF 對 CoreScore 影響極小(實證證據)

**今晚 v0.7 snapshot DB query 結果**:

```
r(IF, LM)         = 0.274    (略相關,acceptable < 0.7)
r(IF, FG)         = -0.038   (近 0)
r(IF, TR)         = -0.080   (近 0)
r(IF, VC)         = 0.002    (完全獨立)
r(IF, core_score) = -0.091   ⚠️ 反向!微弱負相關!
```

**極重要發現**:**IF 跟 core_score 反向相關**(r = -0.091)。意思是:IF 高的股票反而 core_score 略低。可能原因:
- IF 10% weight 被其他 90% weight 稀釋
- 法人重倉股(IF 高)往往是已被市場 priced-in(其他 4 軸偏低)

→ **即使升 IF 為 T1,對當前 universe selection 影響 ≈ 0**(因為 IF 對 core_score 排序影響本就極小)

→ 升 T1 之**即刻 ROI = 0**;真實價值在「未來 §10 model_trainer 用 IF as input feature 之治權正當性」。

### 反對 4:無 walk-forward IC 證據違 §0.1-A 禁令 #6

§0.1-A 禁令 #6:

> 「禁止把 §0.1 之物理隱喻作為『無法獲得 backtest 證據時的替代論據』」

升 T1 是永久承諾;若未通過 walk-forward IC 證偽就升,等於用治權層敘事替代 backtest 證據,**直接違禁令 #6**。

當前 §10 model_trainer 尚未落地 → walk-forward IC 框架不存在 → 無法做嚴格 backtest → **不能升 T1**。

### 反對 5:升 T1 不可逆,違 §0.0-I.7 / L26 治權

T1 化是永久承諾(per §0.1.1 / §0.0-I.7);升 T1 後若實證 IF 對預測力非正向,**無法 rollback**。當前無嚴格 IC 證據 → 升 T1 風險不可逆。

---

## 五、程式 / DB 層影響評估

| 元件 | 影響 |
|---|---|
| `_institutional_flow_score()` | **不需動**(已 12 sub-scores)|
| CoreScore 公式 | **不需動**(IF 10% weight 維持,跟治權層 T1/T2 獨立) |
| v0.2/v0.6/v0.7 snapshot | **0 影響** |
| audit_core_universe | **不需動**(POLICY_SCORE_SCOPE_MAP 維持) |
| feature_store / model_trainer / portfolio_sizer | **0 影響**(下游讀 CoreScore 結果) |
| `data_schema.py` | **不需動** |

**關鍵**:T1 化純治權層升版,**程式碼 0 行變動**。這是「治權升版前置成本低」的優勢,也是「即刻 ROI = 0」的根因。

---

## 六、治權升版風險評估

| 風險維度 | 評估 |
|---|---|
| **Rollback impossibility** | T1 化永久;只能補新治權子節「IF 為 T1 之 IC 證偽 → 降級」屬 messy |
| **§0.1-A 禁令 #6 違反** | **高**(無 backtest 證據就升)|
| **§0.0-G 治權位階(P0/P1/P2/P3)** | 應作為 P0 升版(對應 §0.0-G 6 步流程) |
| **與 §0.2 八二法則** | 強化(冪律 ↔ 集中)|
| **與 §0.3 康波週期** | 中性(日頻 vs 45 年低頻)|
| **與 §9.4 七條治權邊界** | 需重新審視第 7 條(horizon=30 與 IF 為 T1 是否衝突)|
| **§0.1-B 觀測主義張力** | 需更新表格(L1646-L1654 之 F 之可觀測性裁決)|

---

## 七、4 個 option 完整比較

| Option | 描述 | 治權成本 | 即刻 ROI | 長期 ROI | 推薦? |
|---|---|---|---|---|---|
| **A** | 立即升 T1(Option A 公式:F = f(M, V, IF) × ΔlnP)| **極高**(改 §0.1 / §0.1.1 / §0.1-B 等多處;違禁令 #6) | **0**(r(IF, core_score) = -0.091)| 中等(為 §10 鋪路)| ❌ |
| **B** | 不升 T1,維持 IF 為 T2 proxy | 0 | 0 | 0 | ⚠️(放棄路線) |
| **C** | **條件升 T1**(走 Phase A 設計研究,等 walk-forward IC 通過再升)| 中等(寫 design report) | 0(但治權準備就緒) | 高(IC 通過後 → 完整升 T1) | ✅ **本檔推薦** |
| **D** | 部分升 T1(只升 foreign_net 子集為 T1,其他 sub-scores 維持 T2) | 高(分割 IF;違 SSOT)| 0 | 低(模糊治權位階)| ❌ |

---

## 八、推薦執行藍圖(Option C 6 步)

```
Phase A — 設計研究(本檔已完成)
  Step 1. 揭露 IF 升 T1 之 5 支持 + 5 反對(本檔三/四節)✅
  Step 2. 實證 IF 對 CoreScore 影響(r 矩陣)✅
  Step 3. 對比 V 升 T1 之治權先例(類比分析)✅
  Step 4. Option A/B/C/D 比較(本檔七節)✅
  Step 5. 預備 §14.7-BK 治權閉環草案(待用戶授權後另案入憲)
  Step 6. 預備 §0.1.4 IF 第五變數補強草案(類比 §0.1.3 結構)

Phase B — 等待 §10 model_trainer 落地(預估 v6.2.0 / 2026-Q3)
  目標:walk-forward IC simulation 跑通

Phase C — IC ablation 驗證(§10 落地後)
  Step 7. 跑 v0.7+IF as T1 input vs v0.6 baseline 之 walk-forward IC
  Step 8. 通過門檻:IF IC 提升 ≥ 5%(sharpe / max_drawdown)
  Step 9. 若通過 → Phase D;若不通過 → 維持 IF 為 T2

Phase D — 正式升 T1(IC 證偽通過後)
  Step 10. 入憲 §0.1.4 「IF 第五變數補強:F = f(M, V, IF) × ΔlnP」
  Step 11. 入憲 §14.7-BK 治權閉環(類比 §14.7-BC V 補強)
  Step 12. 更新 §0.1.1 表格 (+ IF 為 T1 變數列) + §0.1-A 禁令審視
  Step 13. 更新 §0.1-B 表格 (+ IF 之可觀測性裁決)
  Step 14. 更新 §0.1-E 證偽承諾(+ T1.4 IF IC 正向性)
  Step 15. CoreScore weight 不動(維持 IF 10%)
```

---

## 九、§14.7-AX 元規則之第 9 次跑通候選

本決策對映「**治權框架重構類**」之新類別(非「資料現實裁決」第 1-3 類)。

§14.7-AX 元規則覆蓋面擴展(若採 Phase C 推薦並通過):

```
第 1 類:dataset 內部資料 mislabel/格式異常
第 2 類:dataset 內部公式漂移
第 3 類:外部資源 access 限制(今晚剛入憲)
第 4 類:治權框架重構(F/IF 升 T1 屬此類,候選 v6.2.0)  ← NEW (本檔提議)
```

---

## 十、最終裁決

### **不立即升 T1。執行 Phase A(設計研究)即停,等 Phase B/C 後再決定 Phase D。**

### 核心理由(3 條)

1. **無 backtest 證據時升 T1 直接違 §0.1-A 禁令 #6**(治權禁區)
2. **實證 IF 對 CoreScore 影響極小(r = -0.091)→ 升 T1 即刻 ROI = 0**(花成本沒回報)
3. **§14.7-BI/BJ 今晚實證之「外部資源驗證 protocol」要求永久決策前先 verify**(同邏輯適用)

### 建議入憲動作(明天決定後)

- **若你同意 Phase A 結論** → 本檔升為「**§14.7-BK F/IF 升 T1 治權預備設計研究**」(類比 §14.7-BC V 補強之治權預備模式);**不**改 §0.1 公式;**不**改任何程式碼;commit + tag `v6.1.15-f-promotion-research-phase-a`
- **若你想立即升 T1**(Option A) → 我需要你 override §0.1-A 禁令 #6,我會明文記述違禁令之 trade-off
- **若你想擱置** → 本檔留 untagged,等未來重新評估

### 重新評估時點

- **2026-06-13 後**(v6.1.1 production-current h20 gate 解除)
- **§10 model_trainer 落地後**(預估 v6.2.0 / 2026-Q3)
- **walk-forward IC simulation 完成後**

---

*Report generated 2026-05-25 22:25 by Claude Code session*
*Independent research(Explore agent declined synthesis)*
*Status: AWAITING USER DECISION(2026-05-26)*
