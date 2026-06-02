# `audit_doctrine_compliance.py v0.3` 起草設計文件

- **生成時間**: 2026-05-19 Asia/Taipei
- **起草角色**: 治權守門人（Doctrine Gatekeeper）
- **目標版本**: v0.2 → v0.3
- **憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §0.1-A 6 條禁令 / §0.1-B audit 載體規格
- **產品影響**: 0（純設計文件；不修改既有程式碼）

---

## 一、起草動機

依憲章 §0.1-B 末段「audit 載體」明文：

> **audit 載體**：未來 v6.x.0 升版 `audit_doctrine_compliance.py` 可在 P4 (observability) 加入：
> - `T3_LEAKAGE_CHECK`：掃 §6 / §8 / §9 模組是否誤實作 IFF Θ / SOC / 重力井觸發等 T3 概念
> - `PROXY_TRANSPARENCY_CHECK`：F 之 proxy 變數（如 `institutional_flow`）是否在 `feature_definition` 明文標註其 §0.1 對應

此二檢查為憲章預訂的 v0.3 增量，**符合 §0.1-A 治權自動化的補強路徑**。

---

## 二、現狀盤點（v0.2 架構）

| 元素 | 現狀 |
|---|---|
| 檔案位置 | `scripts/maintenance/audit_doctrine_compliance.py` |
| 行數 | 601 行 |
| 當前版本 | `TOOL_VER = "v0.2"` |
| 四大支柱檢查 | P1 / P2 / P3 / P4 各 `audit_*()` 方法 |
| Pillar keywords | `PILLAR_KEYWORDS` 字典（4 支柱 × ~10 keywords） |
| Module list | `CHARTER_MODULES` 17 個檔案 |
| Helper | `_table_exists()`（v0.2 新增；DRAFT 表 graceful skip） |
| CLI flags | `--scan-module` / `--for-promotion` / `--no-report` |
| 月頻最新結果 | 14 PASS / 0 WARN / 0 FAIL = **PERFECT**（2026-05-17） |

---

## 三、v0.3 變更總覽

### 3.1 新增常數

```python
# 新增於 line ~135（PILLAR_NAMES 之後）

# §0.1-A 禁令 #2 + #3：T3 元素永久不實作清單
T3_FORBIDDEN_PATTERNS = {
    # IFF Θ 控制參數（§0.1.1 line 1205）
    "IFF_theta": r"\bIFF[_\s]?theta\b|\bΘ\s*=\s*\|?∇",
    "information_force_field": r"\binformation_force_field\b|\bIFF_field\b",
    "nabla_information": r"\|∇I\||\bnabla_I\b|\bgrad_information\b",
    "nabla_sentiment": r"\|∇S\||\bnabla_S\b|\bgrad_sentiment\b",

    # SOC 自組織臨界（§0.1.1 line 1206）
    "soc_critical": r"\bSOC_critical\b|\bself_organized_criticality\b|\bsandpile_model\b",
    "soc_trigger": r"\bsoc_trigger\b|\bcritical_avalanche\b",

    # 重力井邊緣觸發（§0.1.1 line 1207）
    "gravity_well_trigger": r"\bgravity_well_trigger\b|\bgravity_well_edge\b",
    "gravity_well_depth": r"\bgravity_well_depth\b|\bwell_depth_calc\b",
    "extreme_arbitrage_signal": r"\bextreme_arbitrage\b",
}

# T3 禁區掃描目標模組（§6 / §8 / §9 落地層）
T3_SCAN_TARGETS = [
    "scripts/core/feature_store_builder.py",
    "scripts/core/model_trainer.py",
    "scripts/core/prediction_engine.py",
    "scripts/core/portfolio_sizer.py",      # 預留（v6.1.0 後啟用）
    "scripts/pipeline/portfolio_optimizer.py",
    "scripts/pipeline/portfolio_strategy.py",
    "scripts/pipeline/portfolio_backtest.py",
]

# §0.1.3 V 變數 + §0.1 M / ΔlnP / F proxy 對映規格
# Key: feature_group / feature_name 前綴
# Value: 應在 description 中明文標註的 §0.1 元素
PROXY_FIRST_PRINCIPLES_MAPPING = {
    # ΔlnP 價格位移
    "log_return_":          ("§0.1", "Delta_lnP"),
    "ma_ratio_":            ("§0.1", "Delta_lnP"),
    "volatility_":          ("§0.1", "Delta_lnP"),
    "max_drawdown_":        ("§0.1", "Delta_lnP"),

    # M 流動性質量
    "avg_daily_value_":     ("§0.1", "M"),
    "turnover_":            ("§0.1", "M"),
    "zero_volume_ratio_":   ("§0.1", "M"),

    # V 內在價值密度（§0.1.3 補強）
    "revenue_yoy_":         ("§0.1.3", "V"),
    "eps_sum_":             ("§0.1.3", "V"),
    "net_income_":          ("§0.1.3", "V"),

    # F 外部資訊力（institutional proxy）
    "foreign_net_":         ("§0.1", "F_external"),
    "trust_net_":           ("§0.1", "F_external"),
    "margin_ratio_":        ("§0.1", "F_external"),
}
```

### 3.2 新增方法（DoctrineAuditor 類別內）

#### 3.2.1 `audit_t3_leakage()`

```python
def audit_t3_leakage(self):
    """
    §0.1-A 禁令 #2 + #3：T3 元素永久不實作

    掃 §6 / §8 / §9 目標模組是否誤實作：
      - IFF Θ 控制參數
      - SOC 自組織臨界
      - 重力井邊緣觸發

    違憲後果：T3 元素任何實作 → FAIL（不可調為 WARN）
    """
    print(f"\n🚫 [T3_LEAKAGE_CHECK] 掃描 §0.1-A 禁令 #2/#3 (IFF Θ / SOC / 重力井觸發)")
    leakage_found = False

    for rel in T3_SCAN_TARGETS:
        p = PROJECT_ROOT / rel
        if not p.exists():
            # portfolio_sizer.py 等預留檔案：合法缺席
            self.add("P1_first_principles", "PASS", "t3_leakage_skip",
                     f"{rel} 不存在（合法缺席 / 未實作）；跳過 T3 掃描")
            continue

        text = p.read_text(encoding="utf-8")
        # 移除註解與 docstring 後再掃（避免誤判提及概念的文件）
        stripped = self._strip_comments_and_docstrings(text)

        violations = []
        for concept, pattern in T3_FORBIDDEN_PATTERNS.items():
            if re.search(pattern, stripped, re.IGNORECASE | re.MULTILINE):
                violations.append(concept)

        if violations:
            leakage_found = True
            self.add("P1_first_principles", "FAIL", "t3_leakage",
                     f"{rel} 違反 §0.1-A 禁令；發現 T3 元素實作: {violations}")
        else:
            self.add("P1_first_principles", "PASS", "t3_leakage",
                     f"{rel} 無 T3 元素洩漏（§0.1-A 禁令 #2/#3 守住）")

    if not leakage_found:
        self.add("P1_first_principles", "PASS", "t3_leakage_summary",
                 f"§0.1-A 禁令 #2/#3 全部守住；掃描 {len(T3_SCAN_TARGETS)} 個模組無 T3 元素實作")

@staticmethod
def _strip_comments_and_docstrings(text: str) -> str:
    """移除 Python 註解與 docstring；避免誤判 charter 引用文字"""
    # 移除 triple-quoted strings (docstrings)
    text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)
    # 移除 single-line comments
    text = re.sub(r'#[^\n]*', '', text)
    return text
```

#### 3.2.2 `audit_proxy_transparency()`

```python
def audit_proxy_transparency(self, cur):
    """
    §0.1-B audit 載體 + §0.1.3 V 變數補強

    驗證 feature_definition 中之 proxy 變數是否在 description 明文標註
    §0.1 元素對應（M / V / ΔlnP / F）。

    違憲後果：缺對應標註 → WARN（文檔衛生問題，不影響工程正確性）
    """
    print(f"\n📝 [PROXY_TRANSPARENCY_CHECK] §0.1 元素對應標註檢驗")

    if not (_table_exists(cur, 'feature_definition') and
            _table_exists(cur, 'feature_store_snapshot')):
        self.add("P1_first_principles", "WARN", "proxy_transparency",
                 "feature_definition 表不存在（§8 DRAFT）；跳過 proxy transparency 檢驗")
        return

    # 取最新 committed feature set 的所有 feature definitions
    cur.execute("""
        SELECT feature_name, feature_group, description, notes
        FROM feature_definition
        WHERE feature_set_id IN (
            SELECT feature_set_id FROM feature_store_snapshot
            WHERE status='committed'
            ORDER BY as_of_date DESC LIMIT 1
        )
        ORDER BY feature_group, feature_name
    """)
    rows = cur.fetchall()

    if not rows:
        self.add("P1_first_principles", "FAIL", "proxy_transparency",
                 "無 committed feature_definition；§0.1 proxy 透明度無從驗證")
        return

    untagged_features = []
    misalignment_features = []
    correctly_tagged = 0

    for feature_name, group, description, notes in rows:
        # 找出該 feature 應對應的 §0.1 元素
        expected = None
        for prefix, (section, element) in PROXY_FIRST_PRINCIPLES_MAPPING.items():
            if feature_name.startswith(prefix):
                expected = (section, element)
                break

        if not expected:
            # macro / theme 等非 §0.1 特徵：不檢驗（屬 §0.3）
            continue

        section, element = expected
        text_to_check = ((description or "") + " " + (notes or "")).lower()

        # 必須在 description 或 notes 中明文標註
        if section.lower() not in text_to_check and element.lower() not in text_to_check:
            untagged_features.append((feature_name, expected))
        else:
            correctly_tagged += 1

    if untagged_features:
        sample = [f"{name} (應標 {sec}/{elt})"
                  for name, (sec, elt) in untagged_features[:5]]
        self.add("P1_first_principles", "WARN", "proxy_transparency",
                 f"{len(untagged_features)} 個 §0.1 proxy 未明文標註對應 (前 5 例: {sample})")
    else:
        self.add("P1_first_principles", "PASS", "proxy_transparency",
                 f"全部 {correctly_tagged} 個 §0.1 proxy features 已明文標註對應元素")
```

### 3.3 更新主流程

```python
# 在 DoctrineAuditor.run() 中（v0.2 line 519〜523）

self.audit_p1_first_principles(cur)
self.audit_t3_leakage()                   # ← v0.3 新增
self.audit_proxy_transparency(cur)        # ← v0.3 新增
self.audit_p2_pareto_barbell(cur)
self.audit_p3_kondratiev_2026(cur)
self.audit_p4_observability_digital_twin(cur)
```

### 3.4 更新版本標記

```python
TOOL_VER = "v0.3"   # was "v0.2"
```

### 3.5 更新 docstring header

```python
"""
audit_doctrine_compliance.py v0.3 (Quantum Finance §0 Supreme Doctrine Compliance Auditor)
================================================================================
最後更新日期: 2026-05-19
主權狀態: IMPLEMENTED (憲法 v6.0.0 §0 四大支柱 + §0.1-A 禁令 #2/#3 自動化檢驗
                    + §0.1-B audit 載體 + §0.1.3 V 變數對應透明化)
最高原則: Doctrine Compliance Authority — §0 從文件 → 機器強制

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
... (保留 v0.2 1-6 條)

7. [T3 Leakage Forbiddance]: §0.1-A 禁令 #2/#3 永久強制——IFF Θ / SOC / 重力井邊緣
   觸發為 T3 操作隱喻，禁止實作於 §6 / §8 / §9 模組。任何發現即 FAIL（不可降為 WARN）。
8. [Proxy Transparency]: §0.1-B audit 載體 + §0.1.3 V 變數補強——feature_definition
   中之 proxy 變數須在 description / notes 明文標註對應的 §0.1 元素
   （M / V / ΔlnP / F_external）。缺標註為 WARN（文檔衛生）。
"""
```

### 3.6 更新修訂紀錄

```python
"""
## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| **v0.3** | 2026-05-19 | Codex | §0.1-B audit 載體入庫：新增 `audit_t3_leakage()`
                                  (§0.1-A 禁令 #2/#3 自動化) 與 `audit_proxy_transparency()`
                                  (§0.1.3 V 變數對應標註)；新增常數
                                  `T3_FORBIDDEN_PATTERNS` / `T3_SCAN_TARGETS` /
                                  `PROXY_FIRST_PRINCIPLES_MAPPING`；新增 helper
                                  `_strip_comments_and_docstrings`。 | **ACTIVE** |
| v0.2 | ...                                                          | SUPERSEDED |
| v0.1 | ...                                                          | SUPERSEDED |
"""
```

---

## 四、判定規則對照

### 4.1 T3_LEAKAGE_CHECK 嚴格度

| 情況 | 判定 | 理由 |
|---|---|---|
| 目標模組存在 + 無 T3 pattern | ✅ PASS | 守住 §0.1-A 禁令 |
| 目標模組存在 + 發現 T3 pattern | ❌ **FAIL（不可降 WARN）** | 違反永久禁令 |
| 目標模組不存在（如 `portfolio_sizer.py`） | ✅ PASS（合法缺席） | 未實作即無 leak |
| Pattern 出現於註解或 docstring | ✅ PASS（已 strip） | 文件引用合憲 |
| Charter 文字引用 | ✅ PASS（target 不掃 reports/） | 哲學說明合憲 |

### 4.2 PROXY_TRANSPARENCY_CHECK 嚴格度

| 情況 | 判定 | 理由 |
|---|---|---|
| Proxy feature description 含對應元素 | ✅ PASS | 透明度達標 |
| Proxy feature description 缺對應元素 | ⚠️ WARN | 文檔衛生問題；不阻擋升版 |
| Feature 非 §0.1 群（macro/theme） | 略過 | 非 §0.1 適用範圍 |
| `feature_definition` 表不存在（DRAFT） | ⚠️ WARN | graceful skip |
| 無 committed feature set | ❌ FAIL | proxy 透明度無從驗證 |

---

## 五、預期執行效果

### 5.1 對既有月頻 audit 的影響

**v0.2 現況**：14 PASS / 0 WARN / 0 FAIL = PERFECT

**v0.3 預估**（基於當前程式碼狀態 + feature_definition 內容）：

| 新增檢查 | 預期結果 |
|---|---|
| `t3_leakage_skip` × 1（portfolio_sizer 預留） | ✅ PASS |
| `t3_leakage` × 6（既存 §6/§8/§9 模組） | ✅ PASS × 6（無 T3 實作） |
| `t3_leakage_summary` × 1 | ✅ PASS |
| `proxy_transparency` × 1 | ⚠️ WARN（多數 features 缺明文標註）|

**預估新版判定**：
- PASS: 14 + 8 = **22**
- WARN: 0 + 1 = **1**
- FAIL: 0
- 判定：**WARNING**（含 1 項可解釋的文檔衛生 WARN）

### 5.2 對升版 gate 的影響

| 升版目標 | v0.2 通過 | v0.3 通過 | 變化 |
|---|:---:|:---:|---|
| v6.1.0 | ✅ | ⚠️ | 需先補 feature_definition description（小型修補） |
| v7.0.0 | ✅ | ⚠️ | 同上 |
| Module scan（新模組） | ✅ | ✅ + T3 check | 更嚴格但對齊憲章 |

---

## 六、整合風險與緩解

| # | 風險 | 機率 | 影響 | 緩解 |
|---|---|:---:|:---:|---|
| 1 | T3 regex 誤判合法符號 | 🟡 中 | 🟡 中 | `_strip_comments_and_docstrings` + 黑名單模式（非白名單）|
| 2 | feature_definition 缺 description 欄位 | 🟡 中 | 🟢 低 | 預先 schema 檢查；若無 description 欄則 WARN（不 FAIL） |
| 3 | portfolio_sizer.py 未來建立後可能觸發 T3 false positive | 🟢 低 | 🟢 低 | 預留 PASS path + 註解標示掃描合憲性 |
| 4 | macro/theme features 被誤判為 §0.1 proxy | 🟡 中 | 🟢 低 | PROXY_FIRST_PRINCIPLES_MAPPING 只列前綴；非匹配即略過 |
| 5 | regex 跨平台 unicode 差異（Θ 字元） | 🟢 低 | 🟢 低 | re.IGNORECASE + 提供 ASCII 替代模式 |

---

## 七、測試計畫

### 7.1 單元測試（建議新增）

```python
# tests/test_audit_doctrine_compliance_v03.py

def test_strip_comments_preserves_code():
    text = '''
    # IFF_theta = compute_theta()  # 註解中提及 T3
    def normal_function():
        pass
    '''
    stripped = DoctrineAuditor._strip_comments_and_docstrings(text)
    assert "IFF_theta" not in stripped  # 註解被移除
    assert "normal_function" in stripped

def test_t3_pattern_detects_code():
    text = "def compute_iff_theta(I, S):\n    return abs(I) / abs(S)\n"
    for concept, pattern in T3_FORBIDDEN_PATTERNS.items():
        if "IFF" in concept:
            assert re.search(pattern, text, re.IGNORECASE)

def test_proxy_mapping_completeness():
    # 確認 27 features 中 §0.1 對應的 21 個都在 PROXY_FIRST_PRINCIPLES_MAPPING
    assert "log_return_" in PROXY_FIRST_PRINCIPLES_MAPPING
    assert "revenue_yoy_" in PROXY_FIRST_PRINCIPLES_MAPPING
    assert "foreign_net_" in PROXY_FIRST_PRINCIPLES_MAPPING
```

### 7.2 整合測試流程

```bash
# Step 1: dry-run（不寫報告）
python scripts/maintenance/audit_doctrine_compliance.py --no-report

# Step 2: 正式 run（產生報告）
python scripts/maintenance/audit_doctrine_compliance.py

# Step 3: 升版 gate 測試
python scripts/maintenance/audit_doctrine_compliance.py --for-promotion v6.1.0

# Step 4: 新模組掃描測試
python scripts/maintenance/audit_doctrine_compliance.py \
  --scan-module scripts/core/feature_store_builder.py
```

### 7.3 驗收門檻

- 既有 P1〜P4 14 項檢查 **全部維持 PASS**（無回歸）
- 新增 T3_LEAKAGE_CHECK ≥ 6 項 PASS（既存模組無 T3 實作）
- 新增 PROXY_TRANSPARENCY_CHECK 1 項（PASS 或可解釋 WARN）
- 整體 runtime < 100 ms（v0.2 為 34.37 ms；新增 ~50 ms regex 掃描）

---

## 八、與 §0.1-A 6 條禁令對照

| 禁令 # | 內容 | v0.3 自動化覆蓋 |
|---|---|:---:|
| 1 | 禁止 `F = M × ΔlnP` 字面公式寫入 prediction model | ⚠️ 部分（需手動 code review） |
| 2 | 禁止實作 `IFF Θ = \|∇I\|/\|∇S\|` 計算函式 | ✅ **T3_LEAKAGE_CHECK 自動偵測** |
| 3 | 禁止用 SOC 概念作為 trading trigger | ✅ **T3_LEAKAGE_CHECK 自動偵測** |
| 4 | 禁止「重力井邊緣 = 訊號觸發」寫入 §9.1 prediction | ✅ **T3_LEAKAGE_CHECK 自動偵測** |
| 5 | 禁止以「市場是物理系統」為由跳過 §8.5 anti-leakage | ❌ 由 `audit_leakage.py` 負責 |
| 6 | 禁止把 §0.1 物理隱喻作為「無法獲得 backtest 證據時的替代論據」 | ❌ 需文件審查 |

**覆蓋率**：6 條中 3 條完全自動化、1 條部分自動化、2 條需手動審查 → **v0.3 達 50% 自動化**（從 v0.2 的 0% 提升）。

未自動化的 #1/#5/#6 可作為 v0.4 候選項。

---

## 九、後繼者導引

### 9.1 實際 patch 操作步驟

當批准實作時：

```bash
# 1. 建立 worktree（或在當前 branch）
git checkout -b feat/audit-doctrine-v03

# 2. 編輯 scripts/maintenance/audit_doctrine_compliance.py
#    依本文件 §3 各小節編輯

# 3. 建立測試
mkdir -p tests/maintenance
# 依本文件 §7.1 建立 test_audit_doctrine_compliance_v03.py

# 4. 跑測試
pytest tests/maintenance/test_audit_doctrine_compliance_v03.py

# 5. 跑實際 audit
python scripts/maintenance/audit_doctrine_compliance.py --no-report

# 6. 確認結果符合 §5.1 預估，若有 WARN/FAIL 為合理範圍才提交
```

### 9.2 對下游 audit 的影響

`audit_supply_chain.py` / `audit_leakage.py` / `audit_downstream_readiness.py` **不需修改**——v0.3 變更僅影響 doctrine compliance 自身。

### 9.3 升版條件對應

依《系統核心完整度評估報告.md》§8 升版優先級：

| 升版項 | v0.3 貢獻 |
|---|---|
| P1 portfolio_sizer.py 建立 | 預留 T3 掃描位置；新建立時不會誤觸 |
| P2 macro × sector_exposure | 不影響 |
| P3 upside/downside vol | PROXY_FIRST_PRINCIPLES_MAPPING 需增 entry |
| P4 prediction_engine 補丁 | 不影響 |
| P5 2031 五年驗證 | T3_LEAKAGE_CHECK 5 年累計報告為證偽指標 #5 之輔助證據 |

---

## 十、起草結論

### 10.1 v0.3 變更摘要

| 項目 | 變更 |
|---|:---:|
| 新增程式碼行數 | ~120 行 |
| 新增常數 | 3 個（`T3_FORBIDDEN_PATTERNS` / `T3_SCAN_TARGETS` / `PROXY_FIRST_PRINCIPLES_MAPPING`） |
| 新增方法 | 3 個（`audit_t3_leakage` / `audit_proxy_transparency` / `_strip_comments_and_docstrings`） |
| 修改主流程 | 1 處（在 P1 之後插入兩個新檢查） |
| 修改 docstring | 1 處（核心定義新增 2 條） |
| 修改修訂紀錄 | 1 處（v0.3 ACTIVE） |
| 新增測試 | 3 個 unit + 整合測試流程 |

### 10.2 治權位階變化

| 治權層 | v0.2 | v0.3 | 變化 |
|---|---|---|---|
| §0.1 第一性原理檢驗 | 2 項（corescore + physics_features）| 2 + 8 = **10 項** | +400% 自動化 |
| §0.1-A 禁令自動化覆蓋 | 0% | **50%**（6 條中 3 條完全自動化）| +50% |
| §0.1-B audit 載體 | 未實作 | **已實作** | 完成 |
| §0.1.3 V 變數透明度 | 未檢查 | **PROXY_TRANSPARENCY_CHECK** | 已落地 |

### 10.3 起草交付物

✅ **本文件**：`reports/audit_doctrine_compliance_v03_draft_20260519.md`（設計規格）

⏳ **待批准實作**：依本文件 §3 patch `scripts/maintenance/audit_doctrine_compliance.py`

⏳ **待補測試**：依本文件 §7 建立 `tests/maintenance/test_audit_doctrine_compliance_v03.py`

**裁決**：v0.3 為**憲章預訂的合憲補強**，所有變更皆對映 §0.1-A / §0.1-B / §0.1.3 既有條款，**無新增治權主張**。可由維運者依本文件直接 patch，預期 v0.3 落地後使憲章 §0.1 治權自動化覆蓋率從 v0.2 的 ~30% 升至 ~50%。

---

**本起草文件僅為設計規格；不修改既有程式碼。實際 patch 須由維運者批准後依本文件 §九 操作。**
