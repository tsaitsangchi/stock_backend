# portfolio_sizer 逆波動加權 (Inverse-Volatility Weighting) 落地設計提案

**性質**：設計提案(§四.2 計畫先行 / §二.4)。**非實作** —— 動 `portfolio_sizer.py`(§9.2 唯一治權載體)+ §9.2 charter 須用戶核可後才改 code/charter。
**日期**：2026-06-03
**動因**：(a) 風險感知權重 quarterly walk-forward 實證顯示 **inv_vol 合法 +0.21 Sharpe** → 用戶 directive「inv_vol 值得正式採納,納入 portfolio_sizer」。

> ❌ **裁決:已重驗 → gate 失敗 → 不採納 v0.4(2026-06-03)**。Phase-1 重驗(monthly/quarterly/annual × 3-seed median,XGBoost top-20)顯示 inv_vol **非跨 horizon 一致**:monthly ΔSh +0.07 / quarterly ΔSh **+0.20(複現 (a))** / **annual ΔSh −0.26(反而變差)**。預設 gate「各 horizon ΔSh>0 一致才採納」**未達**(annual 為負)→ **不實作 v0.4**。機制解讀:inv_vol 在月/季去噪有益,但 annual 降權了驅動年報酬的高波動成長股 → 反效果。production sizer 跑 30 日≈月 horizon(僅 +0.07 微利)且 annual(唯一穩健 production horizon)受害 → CP 值不足。本文件保留為「提案 → 重驗 → 否決」之完整誠實記錄(§一.8/§一.10)。重驗證據:`/tmp/inv_vol_reverify.log`。quarterly 要過門檻應走 **(b) K-wave 60d 訊號特徵**(攻 IC,非權重)。

---

## 🎯 零、白話說明

**現在**:portfolio_sizer 在攻擊端(20%)選股後,用「ROE 高低 × 預測值高低」調權重(v0.3 multiplier)。
**問題**:沒考慮**個股波動度** —— 同樣看好的兩檔,高波動那檔拉高組合風險、壓低 Sharpe。
**這次**:再加一個 **逆波動 multiplier** —— 波動越大、權重越低,波動越小、權重越高(在同一攻擊端內重分配,不改 80/20)。實證(quarterly)Sharpe **2.04→2.25**。

---

## 一、實證依據(a 實驗,2026-06-03)

XGBoost quarterly walk-forward(106 panels / 397 股 / as-of `volatility_60d`,單 seed,各方案同預測):

| 權重 | NET | Sharpe | Eff_t | Win |
|---|---|---|---|---|
| equal(baseline,≈validator 2.07 ✅ 忠實)| 22.7% | 2.04 | 5.54 | 76.2% |
| **inv_vol** | **24.6%** | **2.25(+0.21)** | 6.41 | 76.2% |
| inv_vol_sqrt | 23.8% | 2.14 | 6.03 | 76.2% |
| rank | 23.6% | 1.99 | 5.02 | 74.3% |

→ inv_vol 合法(標準風控、單一固定規則、as-of vol 無 look-ahead、walk-forward)補 +0.21 Sharpe;**機制可泛化**(降組合波動 → 拉 Sharpe),非 horizon-specific 過擬合。

## 二、現況(portfolio_sizer v0.3,§9.2)

- barbell:攻擊端 20% / 防護端(cash)80% / 單股 5% / convex 3% / sector 40%。
- **已是 multiplier-based 加權**:`weight = base × roe_multiplier(α=0.5) × value_multiplier(β=0.3)`,各 clamp([0.5,1.5] / [0.7,1.3])。
- pattern:`_load_roe_lookup()` → `_compute_roe_multiplier()` / `_compute_value_multiplier()` → `apply_policy` 套用 → G13 audit gate。
- **無波動感知**。inv_vol 完美套用此既有 pattern。

## 三、設計(v0.4 新增第三 multiplier)

### 3.1 加權式
```
weight = base × roe_mult(α) × value_mult(β) × inv_vol_mult(γ)   ← 新增
```
`inv_vol_mult` = 對攻擊端股池之 `volatility_60d` 取逆 → 標準化至**均值 1**(僅在 sleeve 內重分配,不改攻擊端總額 20%)→ clamp [γ_min, γ_max]。

### 3.2 新 DEFAULT_POLICY 參數(Tier-3 揭露,對齊既有 α/β 慣例)
```python
"inv_vol_weight_gamma": 0.5,        # 逆波動強度(0=關閉,等同 v0.3)
"inv_vol_multiplier_clamp_min": 0.7,
"inv_vol_multiplier_clamp_max": 1.3,
"inv_vol_feature": "volatility_60d",  # 波動來源(source-pure feature)
```

### 3.3 程式改動(mirror v0.3 ROE/value pattern,minimal-boundary §一.3)
| 改動 | 內容 |
|---|---|
| `_load_vol_lookup()` | 新 method,從 snapshot feature_values 讀 `volatility_60d`(類比 `_load_roe_lookup`)|
| `_compute_inv_vol_multiplier()` | 新 helper:`1/clip(vol)` → 標準化均值 1 → clamp(類比 `_compute_value_multiplier`)|
| `apply_policy` | proposed weight 乘上 `inv_vol_mult`(γ=0 時 no-op,完全相容 v0.3)|
| caps 重檢 | 既有 G3-G8 單股 5%/convex 3%/sector 40% 在乘後 renormalize 仍 enforce(不需新邏輯)|
| (可選)G16 audit | inv-vol 合規 gate(非必要;既有 caps 已 bound)|
| 標頭/版本 | TOOL_VER v0.3→v0.4 / DEFAULT_SIZING_POLICY_VERSION v0.3→v0.4 / 核心定義 +1 條 / 修訂歷程 v0.3 SUPERSEDED + v0.4 ACTIVE |

### 3.4 治權(§9.2 v0.4 + charter sub-clause)
- 新增 §9.2-J(或延伸 §9.2-I)「逆波動風險感知加權」條:source-pure(volatility_60d)+ Tier-3 揭露 γ/clamp + 攻擊端內重分配。
- **§0.2 barbell 不受影響**:inv_vol 是 sleeve **內**加權,不動 80/20 攻防比(§0.2-A 7 禁令不涉)。
- **反硬編相容**:vol 來自 source-pure feature;γ/clamp 同既有 α/β 為 Tier-3 disclosed param(非 feature 層硬編)。
- 雙層治權鎖:主憲章 §9.2-J + (若需)CLAUDE.md 對應條;或僅 §9.2 升版(§9.2 屬 §9 強制契約)。

## 四、⚠️ 採用前驗證要求(§一.10 誠實)
1. **+0.21 為 quarterly 單 seed 證據**;portfolio_sizer 跑 **§9.1 30-day production horizon** + barbell caps + 既有 α/β multiplier —— inv_vol 在**完整 sizer 構造下**的淨效益須**重驗**(3-seed walk-forward,含 caps 互動)。
2. 機制(降波動拉 Sharpe)泛化,但 caps + α/β 可能稀釋/放大效果 → 不可直接宣稱 production +0.21。
3. γ 預設 0.5 為起點,須以 ablation(γ ∈ {0, 0.3, 0.5, 0.8})walk-forward 選定,**不得**調到剛好過某門檻(過擬合紅線)。

## 五、風險 / 非目標
- **非目標**:不動 barbell 80/20、不動 §0.2 禁令、不動 model/prediction(sizer 層 only)。
- **風險**:(a) 與既有 ROE/value multiplier 交互作用未測 → 須 ablation;(b) inv_vol 可能與 value multiplier 部分共線(低波動股常為大型價值股)→ 須查相關性避免雙重計重。
- **相容**:γ=0 → 完全等同 v0.3(零破壞 default-off 可選)。

## 六、決策點(待核可才動工)
1. 做不做?(做 v0.4 / 先只重驗 inv_vol 於 production horizon 再決定 / 擱置)
2. 流程建議:先**重驗**(3-seed walk-forward,§9.1 30d horizon,含 caps)確認 inv_vol 在完整構造下仍 +Sharpe → 通過才修憲 §9.2-J + 改 code v0.4 → ablation 定 γ → commit。
3. ⚠️ 重驗 = 重訓類工作,觸發 §一.12 + §二.6;且目前背景 sync + torch 在跑,建議排隊(sync 完先)。

**證據基礎**:portfolio_sizer v0.3 現況出自 `scripts/core/portfolio_sizer.py` 標頭 + grep(barbell caps L121-135 / ROE-value multiplier pattern L130-135/L53-58 / `_load_roe_lookup`/`_compute_value_multiplier`);inv_vol 實證出自 `/tmp/qw_exp.log`(a 實驗,equal 2.04 ≈ validator 2.07 忠實 sanity check)。無 AI 幻像(§一.10)。
