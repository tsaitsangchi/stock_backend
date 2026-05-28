# Portfolio Allocation Proposal (v0.3)

- **generated_at**: 2026-05-28 13:32:41 Asia/Taipei
- **tool**: portfolio_sizer.py v0.3
- **constitution**: 系統架構大憲章_v6.1.0.md §9.2-A〜§9.2-H / §0.2 / §0.0-A.5
- **prediction_run_id**: `pred_20260415_mdl_20260415_lgbm_h30_0b243a67_v0_1`
- **model_id**: `mdl_20260415_lgbm_h30_0b243a67_v0_1`
- **as_of_date**: 2026-04-15
- **universe_snapshot_id**: `core_universe_20260528_core_universe_policy_v0_15_feature_reasonableness_gate`
- **prediction_policy**: prediction_policy_v0.2
- **sizing_policy**: sizing_policy_v0.3

## 1. 配置摘要 (Allocation Summary)

| 指標 | 值 |
|---|---:|
| 攻擊端總權重 | 0.1500 (15.00%) |
| 防禦端 (CASH) | 0.8500 (85.00%) |
| 配置股票數 | 3 |
| 候選但未配置 | 17 |

### Tier 配置分布

| Tier | 總權重 | 股數 |
|---|---:|---:|
| core_universe | 0.1500 | 3 |
| **CASH (safety)** | **0.8500** | n/a |

### Sector 配置分布

| Sector | 總權重 | 股數 | weight cap (≤0.4) | count cap (≤3) |
|---|---:|---:|---:|---:|
| UNKNOWN | 0.1500 | 3 | ✅ | ✅ |

## 2. 配置明細 (Allocation Details)

| Rank | Stock | Name | Tier | Sector | Pred Value | Signal | Conf | Weight | Reason | Risk Flags |
|---:|---|---|---|---|---:|---|---:|---:|---|---|
| - | **CASH** | safety sleeve | safety | - | - | - | - | **0.8500** | §0.2 防禦端 (≥80%) | - |
| 1 | `8021` |  | core_universe | UNKNOWN | 0.8501 | long | 0.9982 | 0.0500 | tier=core_universe | cap=5% | roe_mult=1.00 | val_mult=1.30 | final=0.0500 | - |
| 2 | `8996` |  | core_universe | UNKNOWN | 0.8317 | long | 0.9964 | 0.0500 | tier=core_universe | cap=5% | roe_mult=1.00 | val_mult=1.30 | final=0.0500 | - |
| 3 | `3167` |  | core_universe | UNKNOWN | 0.8308 | long | 0.9946 | 0.0500 | tier=core_universe | cap=5% | roe_mult=1.00 | val_mult=1.30 | final=0.0500 | - |
| 4 | `2368` |  | core_universe | UNKNOWN | 0.8303 | long | 0.9929 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 5 | `6274` |  | core_universe | UNKNOWN | 0.8255 | long | 0.9911 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 6 | `8358` |  | core_universe | UNKNOWN | 0.8075 | long | 0.9893 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 7 | `2345` |  | core_universe | UNKNOWN | 0.8026 | long | 0.9875 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 8 | `3665` |  | core_universe | UNKNOWN | 0.8022 | long | 0.9857 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 9 | `6442` |  | core_universe | UNKNOWN | 0.7999 | long | 0.9839 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 10 | `3017` |  | core_universe | UNKNOWN | 0.7982 | long | 0.9822 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 11 | `6683` |  | core_universe | UNKNOWN | 0.7895 | long | 0.9804 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 12 | `4979` |  | core_universe | UNKNOWN | 0.7860 | long | 0.9786 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 13 | `5289` |  | core_universe | UNKNOWN | 0.7859 | long | 0.9768 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 14 | `8046` |  | core_universe | UNKNOWN | 0.7839 | long | 0.9750 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 15 | `3189` |  | core_universe | UNKNOWN | 0.7732 | long | 0.9732 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 16 | `3037` |  | core_universe | UNKNOWN | 0.7725 | long | 0.9715 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 17 | `2308` |  | core_universe | UNKNOWN | 0.7699 | long | 0.9697 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 18 | `3653` |  | core_universe | UNKNOWN | 0.7627 | long | 0.9679 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 19 | `6640` |  | core_universe | UNKNOWN | 0.7626 | long | 0.9661 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |
| 20 | `5439` |  | core_universe | UNKNOWN | 0.7618 | long | 0.9643 | 0.0000 | single_sector_count_cap_reached | sector_UNKNOWN_count_full_v02_G12 |

## 3. Sizing Policy v0.2 規則

- attack_total_weight_max: 0.2
- safety_total_weight_min: 0.8
- single_stock_weight_max: 0.05
- convex_tier_weight_max: 0.03
- sector_weight_max: 0.4
- **single_sector_count_max: 3** (v0.2 G12)
- required_coverage: dynamic=1121 (§14.7-BW)
- max_committed_runs: 1

## 4. 治權邊界宣告

- 本程式只讀 committed prediction_run / prediction_values / core_universe_membership；
  **不**重選 universe、**不**重訓 model、**不**重算 prediction、**不**修改任何 raw 表。
- 本配置為 dry-run / report proposal，**非**投資建議；signal_label='long' 為訊號標籤，
  不等於買賣指令。
- §0.1-A 永久禁令 #2/#3 守住：本程式未實作 IFF Θ / SOC / 重力井邊緣觸發。
- §0.2 槓鈴策略：攻擊端 ≤20% + 防禦端 ≥80% + 單股 ≤5% (convex ≤3%) + sector ≤40% + single-sector count ≤5。
- §0.0-G 憲章先行紀律對映：本程式為 §0.0-A.5 第五個轉換器；
  治權邊界明文限定於「formal prediction → allocation proposal」。
- §9.2-D.1 違憲例外契約 v0.2：所有 FAIL gate 觸發拋出 ConstitutionalViolationError。
- §9.2-F.1 Audit Hooks 獨立化 v0.2：4 個 hook 為 module-level function。

## 5. 執行紀錄

- ✅ [PASS] G1 committed prediction run located: pred_20260415_mdl_20260415_lgbm_h30_0b243a67_v0_1 (model=mdl_20260415_lgbm_h30_0b243a67_v0_1)
- ✅ [PASS] G2 prediction_values loaded: rows=1121
- ✅ [PASS] membership loaded: rows=1121 (snapshot=core_universe_20260528_core_universe_policy_v0_15_feature_reasonableness_gate)
- ✅ [PASS] G2 prediction × membership join complete (1121/1121)
- ✅ [PASS] G11 as_of_date consistency verified: prediction=2026-04-15 == feature_set=2026-04-15
- ⚠️ [WARN] G14 backward-compat: snapshot policy=core_universe_policy_v0.15_feature_reasonableness_gate not in v0.3 aligned set ['core_universe_policy_v0.6', 'core_universe_policy_v0.7']; ROE-weighted will fallback if no fg_roe available
- ⚠️ [WARN] G15 ROE coverage 0.0% < threshold 90% (0/1121 stocks have fg_roe); ROE-weighted will fallback for missing(對映 §14.7-BM 金融業 BS 對齊問題)
- ✅ [PASS] candidates filtered: 20 long signals
- ⚠️ [WARN] ROE-weighted disabled: only 0/20 candidates have ROE (fallback mult=1.0;對映 §14.7-BM 金融業 BS 對齊問題)
- ✅ [PASS] v0.3 multipliers computed: ROE-active=0/20 / value-active=20/20
- ✅ [PASS] sizing policy v0.2 applied: attack_total=0.1500, cash=0.8500, allocated_stocks=3, sector_counts={'UNKNOWN': 3}
- ✅ [PASS] audit_proposal_schema: OK
- ✅ [PASS] audit_log_observability: OK
