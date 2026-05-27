# v6.5.0 Native Gate Integration — 設計研究報告

**Phase**: A(設計研究)
**對齊**: charter §14.7-CF(Core Stock Selection Unified SSOT)
**作者**: Codex
**日期**: 2026-05-27
**目標**: 將現行 3-step post-process pipeline 整合為單一 `scripts/core/core_universe_builder.py` 原生實現 charter §14.7-CF 三 invariant

---

## §1 背景

### 1.1 現行 3-step pipeline(post-process gate 路徑)

| Step | 程式 | 治權 | Output |
|---|---|---|---|
| 1 | `scripts/maintenance/build_doctrine_gate_universe.py` v0.10 | §14.7-BV/BW pure doctrine | v0.10 N=1,862 |
| 2 | `scripts/maintenance/apply_feature_completeness_gate.py` | §14.7-CB feature completeness | v0.11 N=1,640 |
| 3 | `scripts/maintenance/apply_raw_data_completeness_gate.py` | §14.7-CD raw data completeness(新 SSOT) | v0.12 N=1,543(active committed) |

### 1.2 charter §14.7-CF 三 invariant(SSOT 終局)

1. **三基柱 source-existence prerequisite** — §0.1 ∧ §0.2 ∧ §0.3 對應 raw source 全到位為唯一前置條件
2. **N 動態 doctrine-derived** — 無 cap/floor/target/quota;N 為 gate 動態結果
3. **API-source uniqueness** — 全 raw data 必且僅可從 FinMind / FRED API 抓取

### 1.3 目前 disruption 來源

- charter §14.7-CD/CF 預告:`core_universe_builder` 應 native 整合 gate(取代 post-process script)
- 當前 `run_weekly_doctrine_recommit.py` Step 4 仍呼 build_doctrine_gate_universe v0.10(SUPERSEDED-IN-TRANSITION),Step 2/3 仍呼 compute_semi_supply_cycle_proxy(deprecated)
- 三 step 分離造成:(a) 重複 SQL audit(11 sources × 2 程式)/ (b) 治權碎片化(3 個 policy_version)/ (c) orchestrator 維護成本

---

## §2 整合架構決策

### 2.1 三選一決策樹

| 選項 | 描述 | 優點 | 缺點 | 採用? |
|---|---|---|---|---|
| **A. 全面 refactor `core_universe_builder.py`** | 替換 2,268 行 CoreScore-based builder 為 doctrine-gate-native | 真正 single canonical | 高風險 / CoreScore v0.7.1 邏輯需大量移除 | ❌ |
| **B. 新增 `DoctrineNativeGateBuilder` 類別於同檔 + 新 CLI mode** | 既有 `CoreUniverseBuilder` 保留(legacy / INFO display),新 class 為 §14.7-CF 原生 | 風險低 / 漸進遷移 / 向後相容 | 同檔兩 builder 並存 | **✅** |
| **C. 新建獨立檔 `core_universe_builder_v0_13.py`** | 全新檔案不汙染既有 | 完全隔離 | 違反用戶 directive「單一 core_universe_builder.py」 | ❌ |

**採用 Option B**:同檔內加 `DoctrineNativeGateBuilder` class + CLI flag `--mode doctrine-native`,既有 `CoreUniverseBuilder` 降為 legacy(`--mode legacy-corescore`,預設 fallback)。

### 2.2 新 class 之職責邊界

```
DoctrineNativeGateBuilder(§14.7-CF 三 invariant 原生實現)
├─ Stage 1: §0.3 K-wave macro prerequisite(13 FRED series 存在性 binary gate)
├─ Stage 2: §0.1 第一性原理 per-stock 8 raw sources × thresholds
├─ Stage 3: §0.2 八二法則 per-stock 3 raw sources × thresholds(Inst + Margin + Info)
├─ Stage 4: doctrine-pass union → core_universe(無 tier split / 無 cap / 無 floor)
└─ Stage 5: INSERT policy + snapshot + membership + revision_log(atomic supersede pattern,per §14.7-BX)
```

**關鍵差異 vs `build_doctrine_gate_universe.py` v0.10**:
- v0.10 Stage 2 為「per-stock 5-source EXISTENCE check」(SQL `LEFT JOIN DISTINCT stock_id`)— 太寬
- **新 v0.13 Stage 2/3 採用 `apply_raw_data_completeness_gate.py` 之 11 source × thresholds**(嚴格,e.g. PriceAdj 365d ≥ 200 / FinStmt 4Q / Institutional 60d ≥ 40 day)

### 2.3 Stage 整合對映表(舊 3 step → 新 single program)

| 舊 step | 舊邏輯 | 新 v0.13 Stage |
|---|---|---|
| build_doctrine_gate_universe Stage 1A/1B/1C | 13 FRED series 存在性 | **Stage 1**(完全沿用)|
| build_doctrine_gate_universe Stage 2(5-source exists) | 寬鬆 EXISTENCE check | **Stage 2+3**(升嚴為 11 source × threshold)|
| apply_feature_completeness_gate Step 2(90d recency) | PriceAdj 90d ≥ 1 row | **Stage 2 之 PriceAdj 365d ≥ 200**(更嚴格,蘊含 90d recency)|
| apply_feature_completeness_gate Step 3(37/37 features) | feature_values 表 37 features non-None | **暫不納入 v0.13**(feature_store 為 §8 下游,不應 block universe selection)|
| apply_raw_data_completeness_gate 11 sources × thresholds | per-stock 嚴格 SQL audit | **Stage 2(§0.1 8 sources)+ Stage 3(§0.2 3 sources)** |

**重要決策**:Step 3 feature completeness gate(37/37 features)**不納入 v0.13 native** — 它屬於 §8 下游 feature store 治權,模型訓練前再 enforce(per §0.0-E.6 priority + feature_store_builder 之 §14.7-CD 廢棄 fallback 後 feature 從此「真實或 None」,即 raw source 不全 → feature 為 None → 自然被 §10 model_trainer drop)。**新 v0.13 只 enforce raw layer**;feature layer 由 feature_store + model_trainer 自然 cascade。

### 2.4 N=1,543 等價性

新 v0.13 預期 output = v0.12 active snapshot(N=1,543):
- Stage 1(K-wave 13 series): pass(macro broadcast 全市場共享;與 v0.10 同)
- Stage 2+3(§0.1 8 sources + §0.2 3 sources × thresholds): 等價於 apply_raw_data_completeness_gate 之 11 thresholds
- v0.12 之 raw_data_completeness_gate 11 thresholds 之輸出 N=1,543 為 v0.13 之等價 target

**等價性驗證**:Phase D dry-run 應得 N=1,543,且 member set 與 v0.12 active snapshot 完全一致(0 diff)。

---

## §3 DB schema 相容性

| 表 | 變動 | 說明 |
|---|---|---|
| core_universe_snapshot | ✅ 零變動 | 新 policy_version `core_universe_policy_v0.13_doctrine_native_gate` |
| core_universe_membership | ✅ 零變動 | core_tier='core_universe',selection_reason='§14.7-CF doctrine-native gate verified' |
| core_universe_policy | ✅ 零變動 | 新 entry `v0.13_doctrine_native_gate` + active=TRUE |
| core_universe_scores | ⚠️ 選擇性寫入 | 新 builder 可選擇沿用 v0.7.1 CoreScore(INFO display)或不寫(scores 可由其他 session 補)|
| universe_revision_log | ✅ 零變動 | actor='core_universe_builder_doctrine_native',action_type='doctrine_native_gate' |

**§6.7 SSOT 相容性**:沿用 atomic supersede(舊 v0.12 → 'superseded',新 v0.13 → 'committed';任一時點 ≤ 1 committed)。

---

## §4 CLI / Interface 設計

### 4.1 新 CLI(於 `scripts/core/core_universe_builder.py` 內)

```bash
# 新原生模式(預計 v6.5.0 起為 production-current)
python scripts/core/core_universe_builder.py --mode doctrine-native --dry-run
python scripts/core/core_universe_builder.py --mode doctrine-native --commit

# 既有 CoreScore 模式(legacy / INFO display only)
python scripts/core/core_universe_builder.py --mode legacy-corescore --dry-run
python scripts/core/core_universe_builder.py --commit  # 預設(暫保 legacy-corescore 為 default;v6.6.0 起切換)
```

### 4.2 Backward compat

- v0.7.1 CoreScore builder 保留為 `CoreUniverseBuilder` class
- 新 `DoctrineNativeGateBuilder` class 為 v0.13
- 預設 mode 仍為 legacy-corescore(v6.5.0 過渡);v6.6.0 起切換預設為 doctrine-native
- charter §6.4 CoreScore 公式不撤銷(per §14.7-BW INFO display only)

### 4.3 與 `run_weekly_doctrine_recommit.py` 之整合

```
v6.5.0 過渡期(weekly orchestrator):
  Step 1: fetch FRED(不變)
  Step 2/3: compute_semi_supply_cycle_proxy(舊;待 v6.5.x 下架)
  Step 4: build_doctrine_gate_universe v0.10(舊;待 v6.5.x 切換)
  Step 5: audit(不變)

v6.5.x:
  Step 1: fetch FRED(不變)
  Step 4: python scripts/core/core_universe_builder.py --mode doctrine-native --commit(取代 Step 2/3/4)
  Step 5: audit_core_universe.py + 新增 v0.13 policy scope map
```

---

## §5 證偽承諾 T_CG-1〜7

- **T_CG-1**: v0.13 dry-run 結果 N 必與 v0.12 active snapshot N=1,543 完全一致(0 diff member set)
- **T_CG-2**: v0.13 commit 之 snapshot member set 與 v0.12 完全一致(per §14.7-CD 11 source thresholds 邏輯等價移植)
- **T_CG-3**: v0.13 不引入任何 hardcoded N(無 cap/floor/target;per §14.7-CF Invariant 2)
- **T_CG-4**: v0.13 寫入 snapshot status='committed' + 舊 v0.12 → 'superseded'(atomic supersede;per §6.7 SSOT)
- **T_CG-5**: v0.13 所有 SQL 讀的 raw tables 全在 9 FinMind + 1 fred_series 範圍(per §14.7-CF Invariant 3;0 system-computed source 引用)
- **T_CG-6**: `audit_core_universe.py` 升版識別 v0.13 後,跑 audit verdict PERFECT
- **T_CG-7**: charter §14.7-CB/CC/CD/CE/CF/CG 六節層次性 inscribed(treaty + proof + implementation 完整治權閉環)

---

## §6 Risk 評估

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 新 v0.13 N ≠ 1,543 | 低 | 高 | dry-run 必先比對 v0.12;若 diff > 5 stocks 須根因分析 |
| audit_core_universe 未識別 v0.13 | 中 | 中 | 同次升版 audit_core_universe POLICY_SCORE_SCOPE_MAP |
| run_weekly_doctrine_recommit 失效 | 中 | 中 | v6.5.0 過渡期 weekly orchestrator 不切;v6.5.x 切換時雙跑驗證 |
| 既有 CoreScore tests 失敗 | 低 | 低 | v0.7.1 CoreScore class 保留;test path 不變 |
| 用戶誤入 wrong mode | 低 | 中 | CLI 必須明示 `--mode doctrine-native`;預設仍為 legacy(v6.5.0 過渡期)|

---

## §7 Roadmap

| Phase | Scope | 預估 |
|---|---|---|
| Phase A | 探索 + 設計研究(本報告) | ✅ done |
| Phase B | charter §14.7-CG 入憲 | 30 min |
| Phase C | 程式落地:scripts/core/core_universe_builder.py 加 `DoctrineNativeGateBuilder` class | 2-3 hr |
| Phase D | 等價性驗證(dry-run N=1,543 + member set diff = 0) | 30 min |
| Phase E | 遷移:commit v0.13 snapshot + 更新 weekly orchestrator + deprecate 舊 3 script | 1 hr |

**總計**: ~5 人時(可單一 session 完成,跨 phase 用戶 check-in 可選)

---

## §8 結論

### 採用方案

- **Option B**: `scripts/core/core_universe_builder.py` 內新增 `DoctrineNativeGateBuilder` 類別 + CLI mode `--mode doctrine-native`
- 既有 `CoreUniverseBuilder` v0.7.1 保留為 legacy / INFO display(per §14.7-BW)
- 新 policy `core_universe_policy_v0.13_doctrine_native_gate`
- v0.13 snapshot 預期 N=1,543(與 v0.12 等價;0 diff)

### Native gate 治權閉環

```
§14.7-BW (N dynamic)
  → §14.7-CC (source authority)
  → §14.7-CD (raw completeness gate)
  → §14.7-CE (per-stock empirical proof)
  → §14.7-CF (unified SSOT)
  → §14.7-CG (native gate integration v6.5.0) ← 本次目標
```

### 同步配套

- charter §14.7-CG entry(Phase B)
- `audit_core_universe.py` 加 v0.13 policy scope map
- `run_weekly_doctrine_recommit.py` Step 4 切換(Phase E)
- 舊 3 script 標 DEPRECATED + v6.5.x 下架
