# P1 上行凸性 v0.1 設計研究報告（§14.7-AD 入憲依據）

- **generated_at**: 2026-05-20 Asia/Taipei
- **scope**: §0.0-E.6 P1 升版優先級「上行凸性三層修正」之第一階段設計：Feature Store 新增 4 個 upside/downside 分離特徵
- **憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §0.0-C.3 + §0.0-E.6 P1 + §9.9（本次入憲）
- **範圍裁決**: **只動 feature_store_builder.py**（避免 P2 interaction features「先實作後否決」之教訓重演）
- **依 §0.0-G.3 Level 1 流程 Step 2**：起草研究報告先於程式實作

---

## 一、設計動機

依憲章 §0.0-C.3 揭露：

> 「上行凸性系統性壓制」貫穿 L1 VolatilityControl（CV 壓抑雙向波動）→ Feature Store（上下行波動混合）→ Model Trainer（線性合成遺漏乘積效應），三層共同偏向穩健性而非凸性。

依 §0.0-E.6 升版優先級：

> **P1**（次高）：L1 VolatilityControl 改 upside/downside 分離設計，Feature Store 加入 upside capture ratio 特徵，Model Trainer 驗證 upside_vol 特徵 ablation IC > 0

依 §14.7-AA Part C 揭露 portfolio_sizer v0.1 之 100% 半導體集中問題之根因分析：

> 所有 top 20 long 訊號 100% 來自半導體業，使 sector cap 失去實質約束力。真正解除需 P1 上行凸性修正使其他產業有機會進入 long signals。

**本研究為實作前之憲章先行設計依據**。

---

## 二、漸進策略（P1 v0.1 限定範圍）

依 §0.0-D.6 #1 P2 interaction features「先實作後否決」之教訓（commit `b97c41f` 揭露 IC = +0.0131，移除反而上升），本次 P1 採**最小範圍 + 漸進驗證**：

| P1 階段 | 範圍 | 觸發升版條件 |
|---|---|---|
| **v0.1（本次）** | Feature Store 新增 4 個 upside/downside 特徵；不改 core_universe_builder | 入憲完成後可立即執行 |
| **v0.2** | model_trainer 重訓 + ablation IC 實證 | v0.1 落地後執行 |
| **v0.3** | 若新特徵 IC > 0 → 修改 core_universe_builder VolatilityControl 評分公式 | v0.2 ablation 通過 |
| **v0.4** | portfolio_sizer 配套（依新 universe 重組） | v0.3 完成 |

**v0.1 不直接改 core_universe_builder 的理由**：
1. 修改 VolatilityControl 評分會觸發 universe 重組（150 檔組成改變）
2. universe 重組需 §6.8 special restore 流程，影響 production-current
3. 在新特徵實證 IC > 0 前，先動 universe 是「先實作後否決」風險的最大來源

---

## 三、4 個新特徵之數學定義

### 3.1 上行/下行波動性（基於 lower partial moment）

對 60 日內 daily log returns $r_1, r_2, ..., r_{60}$，定義：

$$\text{upside\_volatility\_60d} = \sqrt{\frac{1}{N_+}\sum_{i: r_i > 0} r_i^2}$$

$$\text{downside\_volatility\_60d} = \sqrt{\frac{1}{N_-}\sum_{i: r_i < 0} r_i^2}$$

其中 $N_+ = |\{i: r_i > 0\}|$, $N_- = |\{i: r_i < 0\}|$。

**意義**：
- $upside\_vol$ 高 = 上行波動大 = 凸性右尾活躍
- $downside\_vol$ 高 = 下行波動大 = 風險暴露大

### 3.2 上行/下行 capture（基於分組平均）

$$\text{upside\_capture\_60d} = \frac{1}{N_+}\sum_{i: r_i > 0} r_i$$

$$\text{downside\_capture\_60d} = \frac{1}{N_-}\sum_{i: r_i < 0} |r_i|$$

**意義**：
- $upside\_capture$ 高 = 上漲日平均報酬大 = 凸性報酬潛力強
- $downside\_capture$ 高 = 下跌日平均跌幅大 = 下行衝擊強

### 3.3 與既有 `volatility_60d` 的關係

既有 `volatility_60d` = $\sqrt{\frac{1}{N-1}\sum_i (r_i - \bar{r})^2}$（標準差）

新特徵與既有不重疊：
- `volatility_60d` 衡量**雙向波動總強度**
- `upside_volatility_60d` + `downside_volatility_60d` 拆解為**方向性波動**

統計關係（粗略）：
$$\text{volatility\_60d}^2 \approx \frac{N_+}{N}\text{upside\_vol}^2 + \frac{N_-}{N}\text{downside\_vol}^2$$

**保留既有 volatility_60d**：不刪除既有特徵，避免 break v0.1 / v0.2 既有 model artifacts。

---

## 四、與 P2 interaction features 的根本區別

| 面向 | P2 interaction features（已否決）| P1 upside/downside（本次）|
|---|---|---|
| 計算來源 | `macro × stock_feature` 乘積 | `stock 自身 daily returns 之條件統計` |
| 橫截面方差 | 單時點下退化為單調轉換（無新資訊）| 每檔股票之 returns 序列不同 → 橫截面方差非零 |
| 數學風險 | 重複計算已有特徵 | 純衍生但非冗餘 |
| 與 §0.3-A 禁令 | 試圖透過工程技巧繞道 K-wave 入 L2 | 屬 §0.1 ΔlnP 之純粹 stock-specific 分解，不涉及 macro |
| Stock-specific 真實性 | ❌ 偽 stock-specific（macro 常數放大）| ✅ 真正 stock-specific（每檔股票 returns 不同）|

**結論**：P1 v0.1 之 4 個特徵與 P2 interaction 在數學本質上完全不同，不會重蹈「乘以常數 = 重複計算」的覆轍。

---

## 五、實作清單

### 5.1 feature_store_builder.py 變動

#### 5.1.1 新增 4 個 static method

```python
@staticmethod
def _upside_volatility(closes, n):
    """60 日內正報酬之 RMS (root mean squared positive returns)"""
    if len(closes) < n + 1:
        return None
    pos_rets = []
    for i in range(len(closes) - n, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            r = math.log(closes[i] / closes[i - 1])
            if r > 0:
                pos_rets.append(r)
    if len(pos_rets) < 5:  # min_observations_upside
        return None
    return math.sqrt(sum(r * r for r in pos_rets) / len(pos_rets))

@staticmethod
def _downside_volatility(closes, n):
    """60 日內負報酬之 RMS"""
    if len(closes) < n + 1:
        return None
    neg_rets = []
    for i in range(len(closes) - n, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            r = math.log(closes[i] / closes[i - 1])
            if r < 0:
                neg_rets.append(r)
    if len(neg_rets) < 5:
        return None
    return math.sqrt(sum(r * r for r in neg_rets) / len(neg_rets))

@staticmethod
def _upside_capture(closes, n):
    """60 日內正報酬之均值（衡量上行爆發力）"""
    if len(closes) < n + 1:
        return None
    pos_rets = []
    for i in range(len(closes) - n, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            r = math.log(closes[i] / closes[i - 1])
            if r > 0:
                pos_rets.append(r)
    if len(pos_rets) < 5:
        return None
    return sum(pos_rets) / len(pos_rets)

@staticmethod
def _downside_capture(closes, n):
    """60 日內負報酬絕對值均值"""
    if len(closes) < n + 1:
        return None
    neg_rets = []
    for i in range(len(closes) - n, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            r = math.log(closes[i] / closes[i - 1])
            if r < 0:
                neg_rets.append(abs(r))
    if len(neg_rets) < 5:
        return None
    return sum(neg_rets) / len(neg_rets)
```

#### 5.1.2 在 `_compute_price_features` 加入 4 個欄位

```python
def _compute_price_features(self, series):
    # ... 既有 8 個 price features 不變 ...
    f["volatility_60d"] = self._volatility(closes, 60)
    f["volatility_252d"] = self._volatility(closes, 252)
    # ↓ v0.3 新增 4 個
    f["upside_volatility_60d"] = self._upside_volatility(closes, 60)
    f["downside_volatility_60d"] = self._downside_volatility(closes, 60)
    f["upside_capture_60d"] = self._upside_capture(closes, 60)
    f["downside_capture_60d"] = self._downside_capture(closes, 60)
    # ... 其餘特徵 ...
```

#### 5.1.3 FEATURE_DEFINITIONS 加入 4 個 entry

```python
{"name": "upside_volatility_60d", "group": "price", "source": "TaiwanStockPriceAdj",
 "window": "60d", "vtype": "numeric", "null": "drop",
 "desc": "RMS of positive daily log returns over 60 days；§0.1 ΔlnP 上行凸性"},
{"name": "downside_volatility_60d", "group": "price", "source": "TaiwanStockPriceAdj",
 "window": "60d", "vtype": "numeric", "null": "drop",
 "desc": "RMS of negative daily log returns over 60 days；下行風險"},
{"name": "upside_capture_60d", "group": "price", "source": "TaiwanStockPriceAdj",
 "window": "60d", "vtype": "numeric", "null": "drop",
 "desc": "Mean of positive daily log returns over 60 days；上行爆發力"},
{"name": "downside_capture_60d", "group": "price", "source": "TaiwanStockPriceAdj",
 "window": "60d", "vtype": "numeric", "null": "drop",
 "desc": "Mean abs of negative daily log returns over 60 days；下行衝擊"},
```

#### 5.1.4 feature_set_version 預設值升級

```python
DEFAULT_FEATURE_SET_VERSION = "feature_set_v0.3"
```

**保留向後相容**：v0.1（27 features）與 v0.2（31 features 含 interaction，已否決）保留為歷史證據，不刪除。

---

## 六、預期 ablation IC 結果（理論）

### 6.1 正向預期

| Feature | 預期 IC 方向 | 機制 |
|---|---|---|
| upside_volatility_60d | **正** | 上行波動大者 forward return 較高（凸性捕捉）|
| upside_capture_60d | **正** | 上漲日平均報酬大者持續性高 |
| downside_volatility_60d | **負** | 下行波動大者 forward return 較低（風險懲罰）|
| downside_capture_60d | **負** | 下跌日平均跌幅大者後續再跌機率高 |

### 6.2 ablation impact 預期

- v0.3 full model IC 預期 ≥ v0.1 base model IC（27 features）
- 4 個新特徵 drop ablation 預期顯示 IC 下降（即 group HELPFUL）

### 6.3 與 v0.2 interaction features 比較

| 維度 | v0.2 interaction（已否決）| v0.3 upside/downside（本次）|
|---|---|---|
| Ablation IC | +0.0131（HARMFUL）| 預期 -0.01 〜 -0.03（HELPFUL）|
| 結構性風險 | broadcast 重複計算 | 真實 stock-specific 統計 |
| 治權強化 | n/a | 強化 §0.1 ΔlnP 之凸性表達 |

**保險條款**：若 v0.3 ablation 結果為 HARMFUL（drop_minus_full > 0），依 §0.0-G.5 違反裁決規則，v0.3 feature_set 不得升為 production-current，須維持 v0.1 為主力。

---

## 七、與既有契約的相容性

### 7.1 與 §8.2 Feature Store 治權邊界

✅ 完全相容：v0.3 為純衍生計算，不引入新 raw API 資料，不改寫入順序。

### 7.2 與 §9.1 Prediction Contract

✅ 完全相容：v0.3 不影響 prediction layer 之 schema；prediction_engine 自動讀取 committed feature_set。

### 7.3 與 §9.2 Portfolio Sizer Contract

✅ 完全相容：v0.3 不直接影響 portfolio_sizer；待 v0.4 重訓 model 後，prediction 變動才會傳至 sizer。

### 7.4 與 §0.0-A.5 五支落地鏈

✅ 完全相容：v0.3 為 §0.0-A.2 feature_store_builder 之內部升版，不改五支落地鏈拓撲。

---

## 八、執行步驟（依 §0.0-G.3 Level 1）

```text
✅ Step 1：起草 §9.9 Feature Store Upside/Downside Volatility Decomposition Contract v0.1（已入憲）
✅ Step 2：本研究報告（§14.7-AD）
⬜ Step 3：將兩節入憲（本次 commit 同時完成）
⬜ Step 4：實作 feature_store_builder.py v0.3 修改
         - 4 個 static method 新增
         - _compute_price_features 4 個欄位
         - FEATURE_DEFINITIONS 4 個 entry
         - DEFAULT_FEATURE_SET_VERSION 升至 v0.3
⬜ Step 5：撰寫實作驗證報告（§14.7-AE）
         - 語法 + AST 驗證
         - 4 個 static method 單元測試
         - 預期 v0.3 行為說明
⬜ Step 6：commit + push + tag v6.1.3-P1-upside-downside-features-landed
```

---

## 九、後續 P1 v0.2-v0.4 路徑

### v0.2: model_trainer 重訓 + ablation 實證

1. 執行 `python scripts/core/feature_store_builder.py --commit --as-of-date <DATE> --feature-set-version feature_set_v0.3`
2. 執行 `python scripts/core/model_trainer.py --commit --feature-set-id fs_<DATE>_feature_set_v0_3`
3. 執行 ablation：移除 4 個新特徵後 IC 比較
4. 入憲 ablation 結果為 §14.7-AF

### v0.3: VolatilityControl 評分公式修改（若 v0.2 通過）

依 v0.2 ablation IC > 0 之實證結果，起草 §9.10 VolatilityControl Upside/Downside Score Contract，修改 `core_universe_builder.py` 之 `_volatility_control_score()`：

```python
# v0.2 原公式：
score = sigmoid(-CV)  # CV 越低越高分

# v0.3 新公式（待 ablation 通過後落地）：
score = sigmoid(upside_vol - downside_vol) * 0.5 + sigmoid(upside_capture - downside_capture) * 0.5
```

### v0.4: portfolio_sizer 配套

若 universe 重組後 long signals 跨多個 sector，portfolio_sizer v0.2 之 G12 single_sector_count_max=5 自動生效，100% 半導體集中問題實質解除。

---

## 十、結論

本研究設計 P1 v0.1 漸進策略：

1. ✅ **最小範圍**：只動 feature_store_builder.py 4 個 static method + 4 個 FEATURE_DEFINITIONS
2. ✅ **避免 P2 教訓**：新特徵為真實 stock-specific 統計，非 broadcast 乘積
3. ✅ **保留向後相容**：既有 v0.1 base features 不變，v0.2 interaction 保留為歷史證據
4. ✅ **可實證驗證**：透過 v0.2 ablation IC 決定是否進入 v0.3 修改 VolatilityControl
5. ✅ **§0.0-H 通用模板第二支實例**：證明模板對非 portfolio_sizer 契約亦適用

下一步：實作 feature_store_builder.py v0.3 並產出 §14.7-AE 實作驗證報告。

---

**本研究入憲為 §14.7-AD**。
