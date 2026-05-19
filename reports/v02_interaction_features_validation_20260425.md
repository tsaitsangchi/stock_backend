# v0.2 Interaction Features 實證驗證報告

- **generated_at**: 2026-05-19 Asia/Taipei
- **scope**: 對齊 §0.0-D.6 升版條件 #1（macro × sector_exposure 交互特徵）之 ablation 驗證
- **as_of_date**: 2026-04-25
- **label_date**: 2026-05-15（horizon=20 calendar days）
- **feature_set_id**: `fs_20260425_feature_set_v0_2`（31 features）
- **model_id**: `mdl_20260425_lgbm_h20_f1102a9f_v0_1`
- **trainer**: `robust_rank_ic_baseline_v0.1`（linear weighted rank-IC）
- **verdict**: ❌ **DOCTRINE_UPGRADE_REJECTED_IN_LINEAR_BASELINE**
- **product impact**: 0（feature set 已 committed 但未進 production；無需 rollback）

---

## 一、執行步驟

### 1.1 v0.2 feature set commit

```bash
venv/bin/python scripts/core/feature_store_builder.py \
  --commit \
  --as-of-date 2026-04-25 \
  --feature-set-version feature_set_v0.2 \
  --label-horizon 20
```

**結果**：✅ PERFECT
- feature_set_id: `fs_20260425_feature_set_v0_2`
- features defined: 31（27 base + 4 interaction）
- value rows: 4575（150 stocks × ~30.5 avg）
- null imputed: 51
- source_cutoff: 2026-05-17 / as_of_date: 2026-04-25（as-of-strict 守住）

### 1.2 v0.2 baseline model 訓練

```bash
venv/bin/python scripts/core/model_trainer.py \
  --commit \
  --feature-set-id fs_20260425_feature_set_v0_2 \
  --label-horizon 20
```

**結果**：✅ PERFECT
- model_id: `mdl_20260425_lgbm_h20_f1102a9f_v0_1`
- rows trained: 147 stocks
- feature_count: 31
- **ic_mean: 0.3712**
- label range: -0.3243 ~ +0.6246（mean +0.0699，median +0.0363）

### 1.3 Group ablation

**Full model IC = 0.3712**

| Group | n features | IC w/o group | Δ vs full | 結論 |
|---|---:|---:|---:|---|
| liquidity | 4 | 0.3439 | **-0.0273** | ✅ HELPFUL (最強貢獻) |
| institutional | 5 | 0.3497 | **-0.0215** | ✅ HELPFUL |
| fundamental | 4 | 0.3674 | -0.0038 | ➖ neutral |
| price | 8 | 0.3702 | -0.0010 | ➖ neutral |
| macro | 4 | 0.3712 | 0.0000 | ➖ neutral（broadcast 失效） |
| theme | 2 | 0.3712 | 0.0000 | ➖ neutral |
| **interaction** | **4** | **0.3843** | **+0.0131** | **❌ HARMFUL** |

---

## 二、§0.0-D.6 升版條件 #1 驗證裁決

### 2.1 裁決：**否決（在 linear rank-IC baseline 下）**

評估報告 §8.2 P2 預測「macro × sector_exposure 交互特徵 → IC>0」**未能在 v0.1 trainer 中實現**。

**實證**：
- interaction 群 `drop_minus_full = +0.0131`（移除後 IC **上升** 0.0131）
- 即：interaction 群是**雜訊放大器**，不是訊號擷取器

### 2.2 結構性原因分析

**單時點 cross-sectional rank-IC 模型之根本限制**：

依本研究觀察到的數學機制，當時點固定時：

```text
macro_value 為常數 c
feature_x_macro = stock_feature × c

cross-sectional rank(feature_x_macro) ≡ cross-sectional rank(stock_feature)
（單調轉換）
```

因此：
- `feature_macro_vix_x_vol_60d` 在單一日期的排序 = `volatility_60d` 的排序
- `feature_macro_dff_x_eps_sum_4q` 的排序 = `eps_sum_4q` 的排序
- 等於**重複計算**已有特徵的訊號

`theme × stock_feature` 雖然有股票差異（theme 不是常數），但：
- 對非主題股 = 0，對主題股 = stock_feature
- 等於**對主題股 over-weight stock_feature**
- 在 linear weighted IC 中，這破壞了 cross-sectional 排序的中性性

### 2.3 證實了憲章 §0.3-D 之結構性失效

依憲章 §0.3-D 既有揭露：

```text
σ²_CS(macro_t) = 0  →  Rank-IC = Cov / (0 × σ_y) → NaN/0
```

**本實證進一步發現**：將 macro 與 stock-specific 特徵相乘**並未解除此失效**——
因為單時點下乘以常數對排序沒有影響，依然是 cross-sectional 重複。

**真正能解除 §0.3-D 失效的條件**：

| 條件 | 必要性 |
|---|---|
| 多時點訓練（regime-conditioned） | ✅ 必要 |
| 非線性模型（tree / NN / interaction trees） | ✅ 必要 |
| 真正的 stock-specific sector_exposure 特徵 | ✅ 必要（不是 broadcast macro × stock） |

---

## 三、誠實揭露：評估報告 §8.2 預測過於樂觀

《系統核心完整度評估報告.md》§8.2 P2 預期：

> 在 feature_store_builder.py 加入：
>   feature_macro_interest_stress = fed_rate × debt_to_equity
>   feature_theme_active_resonance = theme_strength × sector_exposure
> 預期效果：Feature Store 完整度升至 ≥95%；Model Trainer 實證有效性脫離 0%。

**本實證否決此預期**：在 v0.1 trainer（linear weighted rank-IC）下，純乘法交互特徵**不能**讓 §0.3 戰術層脫離 IC=0。

**評估報告 §6.3 已正確診斷三層凸性壓制的根源**——「穩健性優先於凸性」是設計偏好，**不是單純加幾個特徵能解決**。

---

## 四、修正後的升版路徑

### 4.1 §0.0-D.6 升版條件 #1 之**修正**版本

**原條件**（已否決）：

```text
"macro × sector_exposure 交互特徵落地" → §0.3 戰術層 IC > 0
```

**修正條件**（依本實證）：

```text
條件 #1a：建立真正的 stock-specific sector_exposure 特徵
          （非 broadcast macro × stock）
條件 #1b：將 trainer 升至非線性模型（如 LightGBM tree）
條件 #1c：walk-forward 至少 12 時點訓練 + macro regime 統計
條件 #1d：ablation IC impact > 0 才算達成
```

### 4.2 立即可執行的補強動作

1. **保留 v0.2 feature set**（無害，不入 production）
2. **不部署 v0.2 model 為 production-current**（因 IC 與 v0.1 baseline 無顯著差異）
3. **承認 §0.3 戰術層 IC=0 為設計選擇**（§0.3-A 治權禁令本就限制 K-wave 不入 L2）
4. **將本實證寫入 §0.3-D 結構性失效之延伸證據**

### 4.3 §0.3-A 治權禁令重新確認

依本實證，§0.3-A 第 2 條治權禁令——「K-wave 永久禁止進入 L2 tactical / L3 sizing」——**得到實證強化**：

- 即使透過工程技巧（乘法交互）試圖將 K-wave/macro 引入 L2，**結果在 linear baseline 中為負**
- 這證明 §0.3-A 的禁令**不是無因之禁**，而是有實證根據的物理限制

---

## 五、實證對其他升版條件的影響

| 升版條件 | 是否受影響 |
|---|:---:|
| §0.0-B.5 #1（portfolio_sizer.py 建立） | ✅ 不受影響（已達成） |
| §0.0-B.5 #2（macro/theme 交互特徵） | ❌ 否決（需修正條件） |
| §0.0-B.5 #3（prediction_engine 補丁） | ✅ 不受影響（已達成） |
| §0.0-C.6 #2（upside/downside vol 分離） | ⏳ 未測；應為獨立路徑 |
| §0.0-D.6 #1（macro × sector_exposure） | **❌ 否決**（本實證） |
| §0.0-D.6 #2（sector cap 落地） | ✅ 不受影響（portfolio_sizer v0.1 已含） |
| §0.0-D.6 #3（THEME_KEYWORDS 演化） | ⏳ 未測 |

---

## 六、治權邊界裁決

### 6.1 v0.2 feature set 之處置

- ✅ `fs_20260425_feature_set_v0_2` 保留 committed 狀態（無害且為實證基線）
- ✅ 4 個 interaction features 保留於程式碼（未來 nonlinear model 可能用到）
- ⚠️ 後續 production-current model 應使用 v0.1 feature_set_version（27 features）

### 6.2 v0.2 model 之處置

- ✅ `mdl_20260425_lgbm_h20_f1102a9f_v0_1` 保留為 historical evidence
- ⚠️ **不**升為 production-current（IC 0.3712 與 v0.1 baseline 0.3716 無顯著差異）
- ⚠️ 若未來重訓 production-current h20，建議用 v0.1 feature set 以維持已驗證之 IC stack

### 6.3 憲章修訂建議（不在本次範圍）

下次 v6.0.0-patch 應考慮：

1. **§0.3-D 結構性失效**新增本實證引用
2. **§0.0-D.6 升版條件 #1** 改為修正版本（4 個子條件）
3. **§0.3-A 治權禁令 #2** 加入「乘法交互特徵亦不得作為 K-wave 進入 L2 的後門」之延伸說明

---

## 七、結論

**§0.0-D.6 升版條件 #1 在 linear rank-IC baseline 下實證否決**。

評估報告 §8.2 P2 之預測——「加入 macro × sector_exposure 交互特徵即可解除 §0.3 戰術層 IC=0」——**未能在 v0.1 trainer 中實現**。本實證強化了憲章 §0.3-A / §0.3-D 之既有判定：**K-wave 與 §0.3 macro 訊號之 cross-sectional 失效是結構性的，需要非線性模型 + 多時點訓練才能擷取**。

當前最佳路徑：

1. ✅ 保留 v0.2 feature set 與 model 為歷史實證證據
2. ✅ 更新評估報告 §8.2 之預測為「需 nonlinear model 配合才能擷取」
3. ⏳ 真正解決方案：未來 v6.4.0+ 引入非線性 trainer 時再驗證

**§0.3 在當前系統中仍是「戰略哲學而非戰術武器」**——這完全符合 §0.3-A 治權宣告。本實證是憲章誠實性的最強證據之一。

---

## 八、附錄：實證重現指令

```bash
# Step 1: Commit v0.2 feature set
venv/bin/python scripts/core/feature_store_builder.py \
  --commit --as-of-date 2026-04-25 \
  --feature-set-version feature_set_v0.2 --label-horizon 20

# Step 2: Train v0.2 model
venv/bin/python scripts/core/model_trainer.py \
  --commit --feature-set-id fs_20260425_feature_set_v0_2 \
  --label-horizon 20

# Step 3: Run ablation (one-off script)
venv/bin/python scripts/maintenance/_oneoff_v02_ablation.py
```

**本報告為 audit trail 永久保留；feature set 與 model artifact 不刪除。**
