# Portfolio Sizer v0.2 補強設計研究報告（§14.7-AB 入憲依據）

- **generated_at**: 2026-05-20 Asia/Taipei
- **scope**: 對 §14.7-AA Part D 揭露之 portfolio_sizer.py v0.1 4 項補強缺口（§9.2-F audit hooks 未獨立 / ConstitutionalViolationError 未用 / as_of_date 一致性未顯式 / 100% 半導體集中）之 v0.2 設計
- **憲章基準**: `reports/系統架構大憲章_v6.0.0.md` §9.2-A〜§9.2-H + §0.0-G + §0.0-H + §14.7-AA
- **目標**: v0.1 合規度 80% → v0.2 ≥95%
- **依 §0.0-G.3 Level 1 流程 Step 2**：起草研究報告先於程式實作

---

## 一、補強動機

依 §14.7-AA Part D.3 列出之 v0.2 升版建議：

| 缺口編號 | 內容 | v0.1 狀態 | v0.2 設計 |
|---|---|---|---|
| 1 | §9.2-F 4 個 audit hook 未獨立化 | inline 於 load_inputs / apply_policy | 抽出為 4 個 module-level function + class method |
| 2 | ConstitutionalViolationError 未使用 | self._detail("fail") 軟錯誤 | 定義例外類別 + 強制 raise |
| 3 | as_of_date 跨層一致性未顯式檢查 | 依賴 prediction_engine 隱式保證 | sizer 內顯式驗證 G11 |
| 4 | 100% 半導體集中無 sub-cap | sector_weight_max=0.40 字面合規但精神未實現 | 新增 G12 + single_sector_count_max=5 |

本研究為實作前之憲章先行設計依據。

---

## 二、ConstitutionalViolationError 設計

### 2.1 類別契約

```python
class ConstitutionalViolationError(Exception):
    """憲章 §0.0-G + §9.2-D 之違憲攔截例外。

    Attributes:
        gate_id (str): FAIL gate 編號（G1〜G12 或未來新增）
        message (str): 違憲具體訊息
        charter_ref (str): 對應憲章節（如 "§9.2-D / G7"）

    Example:
        raise ConstitutionalViolationError(
            gate_id="G7",
            message=f"sector 半導體業 total=0.42 > cap=0.40",
            charter_ref="§9.2-D / §14.7-Z",
        )
    """
    def __init__(self, gate_id: str, message: str, charter_ref: str):
        self.gate_id = gate_id
        self.message = message
        self.charter_ref = charter_ref
        super().__init__(f"[{gate_id}] {message} (依 {charter_ref})")
```

### 2.2 拋出時機

| Gate | 拋出位置 | 訊息範例 |
|---|---|---|
| G1 唯一 delivery | load_inputs Step 1 | `committed prediction-backed run count = 2, expected 1` |
| G2 覆蓋度 | load_inputs Step 2 | `prediction coverage = 149, expected 150` |
| G3 防守端 | apply_policy 終驗 | `cash weight = 0.78 < safety_min 0.80` |
| G4 攻擊端 | apply_policy 終驗 | `attack total = 0.21 > cap 0.20` |
| G5/G6 個股上限 | apply_policy 配置時 | `convex stock 6643 weight 0.04 > cap 0.03` |
| G7 sector cap | apply_policy 終驗 | `sector 半導體業 total=0.42 > cap=0.40` |
| G8 左尾隔離 | apply_policy 配置時 | `watch stock weight > 0` |
| G9/G10 治權純度 | static check（init 時） | `sizer 不可呼叫 prediction_run.update`（程式邏輯保證）|
| G11 as_of_date | load_inputs Step 4 | `prediction_run.as_of_date 2025-04-25 != feature_set.as_of_date 2025-04-24` |
| G12 single-sector | apply_policy 配置時 | `sector 半導體業 count=6 > max=5` |

### 2.3 CLI 層統一捕獲

```python
if __name__ == "__main__":
    try:
        main()
    except ConstitutionalViolationError as e:
        print(f"❌ 違憲攔截: {e}", file=sys.stderr)
        sys.exit(1)
```

---

## 三、Audit Hooks 獨立化設計

### 3.1 4 個獨立函式契約

#### audit_input_uniqueness
```python
def audit_input_uniqueness(
    prediction_runs: list[dict],
    prediction_rows: int,
    upstream_writes: list[str],
) -> tuple[bool, str]:
    """G1/G2/G9/G10: 唯一 delivery + coverage + read-only 邊界

    Args:
        prediction_runs: committed prediction-backed run 清單
        prediction_rows: 該 run 之 prediction_values rows
        upstream_writes: sizer 是否曾呼叫上游 write 操作之記錄

    Returns:
        (pass, message): pass=True 時 message 為 "OK"; pass=False 時為違憲詳情
    """
    if len(prediction_runs) != 1:
        return False, f"G1: committed run count = {len(prediction_runs)}, expected 1"
    if prediction_rows != 150:
        return False, f"G2: prediction rows = {prediction_rows}, expected 150"
    if upstream_writes:
        return False, f"G9/G10: sizer attempted upstream writes: {upstream_writes}"
    return True, "OK"
```

#### audit_constraint_satisfaction
```python
def audit_constraint_satisfaction(
    allocations: list[dict],
    policy: dict,
    sector_counts: dict[str, int],
) -> tuple[bool, str]:
    """G3-G8 + G12: 槓鈴 caps + sector cap + bottom 20 + single-sector count

    Args:
        allocations: 全部配置紀錄
        policy: DEFAULT_POLICY
        sector_counts: sector → 配置股票數

    Returns:
        (pass, message)
    """
    # G3/G4 槓鈴
    attack_total = sum(a["target_weight"] for a in allocations)
    if attack_total > policy["attack_total_weight_max"] + 0.0001:
        return False, f"G4: attack={attack_total:.4f} > cap"
    if (1.0 - attack_total) < policy["safety_total_weight_min"]:
        return False, f"G3: cash={1.0-attack_total:.4f} < safety_min"

    # G5/G6 個股 cap
    for a in allocations:
        cap = (policy["convex_tier_weight_max"]
               if a["tier"] == "convex_universe"
               else policy["single_stock_weight_max"])
        if a["target_weight"] > cap + 0.0001:
            return False, f"G5/G6: stock {a['stock_id']} weight > cap"

    # G7 sector cap
    sector_totals = {}
    for a in allocations:
        sector_totals[a["sector"]] = sector_totals.get(a["sector"], 0) + a["target_weight"]
    for sec, total in sector_totals.items():
        if total > policy["sector_weight_max"] + 0.0001:
            return False, f"G7: sector {sec} total={total:.4f} > cap"

    # G8 左尾隔離
    for a in allocations:
        if a["signal_label"] == "watch" and a["target_weight"] > 0:
            return False, f"G8: watch stock {a['stock_id']} has weight > 0"

    # G12 single-sector count
    for sec, count in sector_counts.items():
        if count > policy["single_sector_count_max"]:
            return False, f"G12: sector {sec} count={count} > max"

    return True, "OK"
```

#### audit_proposal_schema
```python
def audit_proposal_schema(
    proposal_rows: list[dict],
    required_fields: list[str],
) -> tuple[bool, str]:
    """§9.2-C 輸出 schema 9 欄位完整性"""
    if not proposal_rows:
        return True, "OK (empty proposal)"
    for i, row in enumerate(proposal_rows):
        missing = [f for f in required_fields if f not in row]
        if missing:
            return False, f"row {i} missing fields: {missing}"
    return True, "OK"
```

#### audit_log_observability
```python
def audit_log_observability(
    stats: dict,
    allocations: list[dict],
) -> tuple[bool, str]:
    """risk_flags / allocation_reason 完整記錄"""
    if "details" not in stats:
        return False, "stats missing 'details' key"
    for a in allocations:
        if a["target_weight"] > 0 and not a.get("allocation_reason"):
            return False, f"stock {a['stock_id']} missing allocation_reason"
    return True, "OK"
```

### 3.2 純函式設計理由

- **無 self / instance state**：可被 `audit_doctrine_compliance.py` 直接 import 並呼叫
- **依賴注入**：所有輸入為函式參數，易於單元測試
- **回傳 tuple[bool, str]**：統一格式，便於 audit chain 串接

---

## 四、G11 as_of_date 一致性設計

### 4.1 檢查位置

於 `load_inputs` Step 4 末尾（prediction × membership join 後）新增：

```python
# G11: as_of_date 跨層一致性
cur.execute(
    "SELECT as_of_date FROM feature_store_snapshot WHERE feature_set_id = %s",
    (self.run_meta["feature_set_id"],),
)
fs_row = cur.fetchone()
if fs_row is None:
    raise ConstitutionalViolationError(
        gate_id="G11",
        message=f"feature_set_id {self.run_meta['feature_set_id']} not found",
        charter_ref="§9.2-D / G11 / §8.5",
    )
fs_as_of_date = fs_row[0]
if fs_as_of_date != self.run_meta["as_of_date"]:
    raise ConstitutionalViolationError(
        gate_id="G11",
        message=(f"prediction_run.as_of_date={self.run_meta['as_of_date']} "
                 f"!= feature_set.as_of_date={fs_as_of_date}"),
        charter_ref="§9.2-D / G11 / §8.5",
    )
```

### 4.2 設計理由

- 即使 prediction_engine 已強制此一致性（§14.7-Y），sizer 仍應「**不信任上游、自我驗證**」
- 此檢查在 commit 時也可被 `audit_doctrine_compliance.py` 獨立呼叫
- 為未來 §6.7/§7/§8 之 audit hook 統一化奠基

---

## 五、G12 single-sector count cap 設計

### 5.1 參數選擇：count=5 vs weight-based

| 設計 | 優點 | 缺點 |
|---|---|---|
| `single_sector_weight_max = 0.075`（attack 內某 sector 最多 7.5%）| 直觀 | 與 sector_weight_max=0.40 重疊 |
| `single_sector_count_max = 5`（同 sector 最多 5 檔） | 簡潔、與 weight 邏輯互補 | 需與 convex_cap 3% 配合計算 |
| 二者並行 | 完整 | 規則複雜 |

**選擇 count=5**：

- 20% attack budget / 3% convex cap ≈ 6.7 檔上限，count=5 保守且可分散至 4 個 sector（5 × 4 = 20 檔候選空間）
- count 比 weight 更直觀，policy 文件更清晰
- 不與 sector_weight_max 重疊，是獨立維度的約束

### 5.2 觸發行為

```python
# 在 apply_policy 內，每次配置前檢查
sector = member.get("industry_category", "UNKNOWN")
if self.sector_counts[sector] >= policy["single_sector_count_max"]:
    self.allocations.append({
        ...
        "target_weight": 0.0,
        "allocation_reason": "single_sector_count_cap_reached",
        "risk_flags": [f"sector_{sector}_count_full"],
    })
    continue
# 配置成功後
self.sector_counts[sector] += 1
```

### 5.3 100% 半導體集中的預期解除效果

依 v0.1 首份 allocation proposal 實證（`reports/portfolio_allocation_proposal_2025-04-25.md`）：

| 階段 | v0.1 行為 | v0.2 預期行為 |
|---|---|---|
| Rank 1-5 半導體配置 | ✅ 配置至 attack budget 20% | ✅ 配置至 5 檔（count 滿） |
| Rank 6+ 半導體候選 | attack_budget 耗盡，全部歸 CASH | sector_count 滿，全部歸 CASH（即使 attack_budget 未滿）|
| 非半導體候選 | 從未進入 top 20 | 仍未進入（prediction 層問題，sizer 不解決） |
| 最終結果 | 100% 半導體 | 仍可能 100% 半導體（5 檔上限內）|

**重要裁決**：G12 v0.2 **無法解除** 100% 半導體集中之根因（即所有 long signals 來自單一產業），但可**降低集中風險**：5 檔 × 3% = 15% 而非 v0.1 的 5 檔 × (3%-5%) = 15-25%。真正解除需 P1 上行凸性修正使其他產業有機會進入 long signals。

---

## 六、v0.2 預期合規度

| §9.2 子節 | v0.1 評分 | v0.2 預期評分 |
|---|---|---|
| §9.2-A 識別 | 10/10 | 10/10 |
| §9.2-B 強制輸入 | 8/10 | **10/10**（補 G11） |
| §9.2-C 強制輸出 | 10/10 | 10/10 |
| §9.2-D FAIL Gate | 7/10 | **10/10**（補 G11/G12 + ConstitutionalViolationError）|
| §9.2-E Sizing Policy | 10/10 | 10/10 |
| §9.2-F Audit Hooks | 4/10 | **10/10**（4 hooks 獨立化）|
| §9.2-G 跨層影響 | 6/10 | **8/10**（解除 100% 集中之集中度，但不解除單一產業 long 訊號）|
| §9.2-H 違反處置 | 9/10 | **10/10**（補 ConstitutionalViolationError）|

**綜合**：v0.1 64/80 (80%) → v0.2 **78/80 (97.5%)**

---

## 七、跨層完整度預期影響

| 跨層基線 | v0.1 完整度 | v0.2 完整度（預期）| 變化 |
|---|---|---|---|
| §0.0-B Portfolio Sizer | ~60% | ~75% | +15% |
| §0.0-C Portfolio Sizer | ~60% | ~75% | +15% |
| §0.0-D Portfolio Sizer | ~65% | ~75% | +10% |

未達 80% 之根因：**100% 單一產業 long 訊號**不在 sizer 治權範圍，需 P1 上行凸性配套。

---

## 八、與 §0.0-H 通用模板的相容性驗證

v0.2 補強完全於 §9.2-A〜§9.2-H **既有八子節結構內**完成：

- §9.2-A：補註 v0.2 標記
- §9.2-D：補入 G11/G12 + §9.2-D.1 違憲例外契約
- §9.2-E：補入第 11/12 條 sizing policy
- §9.2-F：補入 §9.2-F.1 audit hooks 強制獨立化

**無新增子節 §9.2-I / §9.2-J**——驗證 §0.0-H 通用模板對「同一契約之版本升級」具備充足容納能力。

---

## 九、實作清單

實作 portfolio_sizer.py v0.2 需做以下 6 件事：

1. ✅ 定義 `ConstitutionalViolationError` 類別於模組頂端
2. ✅ 抽出 4 個 audit hook 為獨立函式（含對應簽名）
3. ✅ 補入 G11 as_of_date 一致性檢查於 load_inputs Step 4
4. ✅ 補入 G12 single-sector count cap 於 apply_policy
5. ✅ DEFAULT_POLICY 新增 `single_sector_count_max = 5`
6. ✅ CLI `__main__` 統一捕獲 ConstitutionalViolationError

---

## 十、結論

v0.2 補強為 **§0.0-G.3 Level 1 流程之第二次完整跑通**，驗證以下：

1. **既有 §9.2-A〜§9.2-H 模板可容納版本升級**（§0.0-H 模板穩定性）
2. **ConstitutionalViolationError 強化憲章 §0.0-G.5 違反裁決之執行**
3. **獨立函式 audit hooks 為 §6.8 同步治權審計與 audit_doctrine_compliance.py 鋪路**
4. **G12 single-sector count cap 是 v0.1 100% 半導體集中之直接回應**
5. **預期 v0.2 對齊度 97.5%**，剩餘 2.5% 缺口屬 sizer 治權範圍外（prediction layer 問題）

下一步：依本研究實作 portfolio_sizer.py v0.2 並產出實作驗證報告。

---

**本研究入憲為 §14.7-AB**。
