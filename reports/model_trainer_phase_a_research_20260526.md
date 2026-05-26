# §10 Model Trainer Formal Contract Phase A 設計研究 — v6.2.0 軌道最強優先項(治本 §14.7-AA Part C root cause)

- **產出日期**: 2026-05-26 evening
- **產出者**: Claude Sonnet 4.7 session
- **觸發**: 用戶選擇「§10 model_trainer 設計研究(治本 root cause;v6.2.0 軌道)— 最強優先」
- **scope**: Phase A 治權先行設計研究 — 不動程式;類比 §14.7-BC V 補強 + §14.7-BF F proxy + §14.7-BO CashFlow Phase A 模式
- **chapter slot**: §10(charter 內多處 forward reference 至 §10;本研究為 §10 升正式條文之 Phase A 預備)
- **對映 charter**: §0.0-A.3 五大轉換器之第三個 / §8.3 v0.1 草案升正式 / §14.7-X model_trainer 既有研究

---

## 一、觸發背景 — §10 為 v6.2.0 軌道之 root cause 治本主題

### 1.1 §10 在 charter 內之多處 forward references

| Charter 行 | 引用 |
|---|---|
| L5141 | 「100% 單一 sector 集中(upstream prediction)— **需 §10 model_trainer 治本**」 |
| L5144 | 「§10 model_trainer 治本另案(v6.2.0 軌道 2-3 週):training time enforce sector exposure cap + walk-forward IC > 0 + sharpe gate」 |
| L8795 | 「Phase H.3 walk-forward IC ablation 驗證(v6.2.0 強制 gate;**待 §10 model_trainer 落地**)」 |
| L8893 | 「當前 §10 model_trainer 未落地,無 walk-forward IC,升 T1 即違禁令」 |
| L8903 | 「Phase B — 等待 §10 model_trainer 落地(預估 v6.2.0 / 2026-Q3)」 |
| L9036 | 「v0.3 ROE-weighting 之實際 alpha 提升待 §10 model_trainer 落地後才能裁決」 |

→ **§10 是 v6.1.x 期間多 Phase B-D / 證偽承諾之 common gate**;治本最強優先項

### 1.2 §10 治本之 5 個 root causes(來自 v6.1.x 揭露)

| Root cause | 揭露來源 | §10 之治本路徑 |
|---|---|---|
| **100% 半導體 prediction candidates** | §14.7-AA Part C(L98)| training 加 sector-balanced loss / sector exposure cap |
| **portfolio_sizer v0.3 治標** | §9.2-I.7(L5141)| upstream prediction layer 多元化解決根因 |
| **walk-forward IC 缺自動 monitoring** | §14.7-BH Phase H.3 等(L8795)| §10-F 自動 IC monitoring 入 audit_doctrine_compliance |
| **F 升 T1 等 IC > 0 證據** | §14.7-BK Phase B(L8903)| walk-forward IC 之 sector-stratified ablation |
| **v0.3 ROE-weighting 證據空白** | §9.2-I 證偽 T_PS_v0.3-5(L9036)| walk-forward IC 證明 ROE-weighted alpha 提升 |

→ **§10 落地後可一次解 5 個 v6.1.x 之 root causes** — 治本不治標

---

## 二、§10 在五大轉換器中之位置

### 2.1 §0.0-A 五大轉換器鏈

```
1. core_universe_builder.py     — §6.7   ✅ v0.8 production
2. feature_store_builder.py     — §8.2   ✅ v0.5 production(§9.9 RMS 已對齊)
3. model_trainer.py             — §8.3   ⚠️ v0.1 robust_rank_ic_baseline(本研究升 §10)
4. prediction_engine.py         — §8.4/§9.1 ✅ v0.2 production(等 §10 升 §9.1 horizon=30)
5. portfolio_sizer.py           — §9.2   ✅ v0.3 production(等 §10 IC 證偽 T_PS_v0.3-5)
```

→ **§10 是五大轉換器之第三 + 最大瓶頸**;前 2 + 後 2 都已 v0.5+ production,只 §10 仍 v0.1 baseline

### 2.2 §10 跟 §9.1 prediction contract 之關係

```
§10 model_trainer   = TRAINING 階段(fit model on historical labels)
§9.1 prediction     = INFERENCE 階段(predict on as_of_date)

§10 之 output = model artifact + walk-forward IC evidence
§9.1 之 input = §10 之 committed model artifact
§9.1 之 output = prediction_values (rank / signal_label / confidence)
```

**治權邊界**:
- §10 不寫 prediction_values(那是 §9.1 之事)
- §9.1 不重訓 model(那是 §10 之事)
- 兩者透過 `model_registry` table SSOT 連結

---

## 三、§10 與 §8.3 既有 v0.1 草案之關係

### 3.1 §8.3 v0.1 草案內容(charter L169 入憲)

```
§8.3 Model Registry DDL 雛型:
  - model_registry table
  - model_training_run table
  - 訓練合約
  - model_trainer.py v0.1 模組規範
```

### 3.2 §10 與 §8.3 之治權升版關係

| 層 | §8.3 v0.1 (現行 DRAFT) | §10 (本研究 Phase A → Phase D) |
|---|---|---|
| 治權位階 | §8.3 (DRAFT;ACTIVE 但 DRAFT) | §10(formal contract;類比 §9.1/§9.2)|
| 規範密度 | DDL + 訓練合約 | §10-A~H 8 子節(對映 §0.0-H 模板)|
| FAIL gates | (草案級,~6 條)| **15+ FAIL gates**(類比 §9.2-A~H 12 + v0.3 G13-G15)|
| Walk-forward IC | (有 evidence;§8.8.7 panel)| **強制 auto monitoring** + IC > 0 gate |
| sector-balanced loss | (無)| **強制**(治本 §14.7-AA Part C)|
| Multi-model ensemble | (無)| **可選**(LGBM + XGBoost 雙模型;LSTM 試錯)|

→ **§10 是 §8.3 之 formal 升版**;§8.3 為 DRAFT 升 §10 為強制契約是 v6.2.0 主要任務

---

## 四、§10-A~H 八子節結構(對映 §0.0-H 模板)

### 4.1 §10-A Identity(對映 §9.2-A 模式)

```
硬編檔案路徑:scripts/core/model_trainer.py
類別位階:Type-2 治權契約層(類比 §9.1 / §9.2)
治權繼承:從 §8.3 v0.1 草案升正式條文
```

### 4.2 §10-B 強制輸入契約

| 輸入 | 強制來源 | FAIL 條件 |
|---|---|---|
| feature_store_snapshot | §8.2 committed | snapshot 非 committed 直接 FAIL |
| feature_values | 對應 feature_set_id | row count != 預期 → FAIL |
| as_of_date | §8.5 anti-leakage strict | label 之 forward 必須未實現 |
| label horizon | §9.1 強制 30 交易日 | horizon != 30 直接 FAIL |
| universe_snapshot_id | §6.7 core+convex 150 | 非 v0.6/v0.7 snapshot WARN |

**禁止讀取**:
- 任何 raw API tables(只讀 committed feature_store)
- 任何 hardcoded stock list
- 任何 prediction_values(那是 §9.1 之 output)

### 4.3 §10-C 強制輸出契約

**v0.2 階段**: model_artifact + model_training_run row + walk-forward IC report

| Output 欄位 | 型別 | 語意 |
|---|---|---|
| model_id | str | `mdl_{yyyymmdd}_{family}_h{horizon}_{sha1[:8]}_v{ver}` |
| model_family | str | `lgbm` / `xgboost` / `linear` / `lstm`(v0.3+) |
| feature_set_id | str | upstream §8.2 reference |
| ic_mean | float | walk-forward IC mean(必須 > 0)|
| ic_std | float | IC stdev |
| sharpe_ratio | float | annualized Sharpe(必須 ≥ 0.5)|
| sector_balance_score | float | top 5 sector weight 之 entropy(0=完美集中 / 1=完美分散)|
| training_time_seconds | float | 訓練耗時 |
| artifact_path | str | model artifact 之 file path |

### 4.4 §10-D 強制 FAIL Gate(15+ 條)

| Gate | 條件 | 依據 |
|---|---|---|
| **G1** 唯一 committed feature_set | feature_set_id 非 committed | §8.2 |
| **G2** universe 對齊 | universe_snapshot_id 不在 ALIGNED_UNIVERSE | §6.7 |
| **G3** label horizon = 30 | horizon != 30 | §9.1 |
| **G4** as_of_date anti-leakage | as_of_date + 30 > today | §8.5 第 1 條 |
| **G5** IC > 0 | walk-forward IC mean ≤ 0 | **治本 gate**(§10 核心)|
| **G6** IC std < 2 × IC mean | IC 不穩定 | walk-forward 穩定性 |
| **G7** sector entropy ≥ 0.5 | top 5 sector weight 過度集中 | **§14.7-AA Part C root cause** |
| **G8** Sharpe ≥ 0.5 | risk-adjusted return 不足 | §9.1 質量門檻 |
| **G9** training reproducible | random_seed 未鎖定 | §6.5 No-Hardcode + reproducibility |
| **G10** treatment consistency | train/inference transform 不對齊 | §14.7-X 揭露之 ROE rank 問題 |
| **G11** model artifact 完整 | artifact 缺 winsor bounds / encoder | §14.7-X v0.1 揭露 |
| **G12** sector-balanced loss enabled | training loss 未含 sector penalty | **治本核心(§14.7-AA Part C)** |
| **G13** walk-forward panel >= 8 points | panel 不足 robust 驗證 | §8.8.7 |
| **G14** label distribution sane | label mean/std outside [-3σ, +3σ] | §0.4 可觀察性 |
| **G15** model_registry single SSOT | 同 (date, family, horizon, fs_id) 多 model | §8.8.8 single-delivery |

### 4.5 §10-E 強制 Training Policy

```python
# v0.2 Training Policy(類比 §9.2-E sizing policy)
TRAINING_POLICY = {
    "label_horizon": 30,                    # 強制 §9.1
    "lookback_lookback": 252,               # §6.3 lookback
    "winsor_low": 0.05,                     # robust rank-IC baseline
    "winsor_high": 0.95,
    "ic_min_threshold": 0.05,               # G5(允許 baseline 過關)
    "sharpe_min_threshold": 0.5,            # G8
    "sector_entropy_min": 0.5,              # G7 治本
    "sector_penalty_weight": 0.3,           # G12 治本(loss = mse + λ × sector_penalty)
    "random_seed": 42,                      # G9 可重現
    "model_family_default": "lgbm",         # 主推 LGBM
    "ensemble_enabled": False,              # v0.2 baseline;v0.3 ensemble 開
    "walk_forward_panel_size": 8,           # G13
    "training_max_time_seconds": 3600,      # 1 hour timeout
}
```

### 4.6 §10-F 強制 Audit Hooks(4 個 module-level functions;類比 §9.2-F.1)

```python
def audit_model_input(feature_store_snapshot, universe_snapshot, as_of_date):
    """G1/G2/G3/G4 強制檢查"""

def audit_training_quality(ic_mean, ic_std, sharpe, sector_entropy):
    """G5/G6/G7/G8 強制檢查"""

def audit_training_artifact(model_artifact_path, expected_keys):
    """G10/G11 transform consistency + artifact 完整"""

def audit_walk_forward_panel(panel_runs, expected_size=8):
    """G13/G14 panel 完整性"""
```

每次 commit 應產出 `reports/model_trainer_audit_<model_id>.md`(類比 §9.2-F.1 模式)

### 4.7 §10-G 跨層影響

| 跨層基線 | 預期完整度提升 |
|---|---|
| §0.0-B 第一性原理 | Model Trainer 78% → **90%**(walk-forward IC 自動證偽)|
| §0.0-C 八二法則 | Model Trainer 75% → **90%**(sector-balanced loss 解 root cause)|
| §0.0-D 康波週期 | Model Trainer 60% → **85%**(MBNRIC 主題 IC 跨期驗證)|

### 4.8 §10-H 違反處置

任何違反 §10-A~F 之 model_trainer.py 版本:
- 直接判「非法變動」
- 不得通過 §6.8 同步治權審計
- 不得進入 production-current delivery
- 必須撤回或補正後重新審計

---

## 五、Walk-forward IC framework 設計

### 5.1 既有 §8.8.7 walk-forward h20 panel evidence

```
2024-09-30 → 2025-04-25(8 時點,每月一筆)
IC_mean 序列: [0.3429, 0.3180, 0.3581, 0.4226, 0.4934, 0.4829, 0.4260, 0.4901]
min = 0.3180 / max = 0.4934
trainer = robust_rank_ic_baseline_v0.1
horizon = 20(本研究升 30 per §9.1)
```

→ 既有 evidence 已 strong;§10 將其自動化 + sector-stratified

### 5.2 §10 自動化 walk-forward framework

```
v0.2 walk-forward 流程:
  for as_of in panel_dates:  # 8 個時點 by default
    fs = load_committed_feature_store(as_of)
    universe = load_universe_snapshot(as_of)
    labels = compute_forward_log_return(as_of, horizon=30)
    
    # 訓練(含 sector-balanced loss)
    model = train_with_sector_penalty(
        features=fs.values,
        labels=labels,
        sectors=universe.sectors,
        penalty_weight=POLICY["sector_penalty_weight"]
    )
    
    # IC 計算(per sector + overall)
    ic = compute_rank_ic(model.predict(fs.values), labels)
    sector_ic = {s: compute_rank_ic_in_sector(s) for s in universe.sectors}
    
    # 記錄
    panel_runs.append({...})
  
  # 驗收 G5/G6/G13
  audit_walk_forward_panel(panel_runs)
```

---

## 六、Sector-balanced loss(治本 §14.7-AA Part C 核心)

### 6.1 既有 v0.1 之 root cause(§14.7-AA Part C)

```
v0.1 robust_rank_ic_baseline 之 loss:
  loss = MSE(predicted_ranks, actual_ranks)  # 純 ranking loss
  
缺點: 無 sector awareness → 模型自動偏向「高 IC sector」(歷史上是半導體)
     → 100% top-20 long signals 來自半導體業
     → portfolio_sizer 之 sector_cap=0.40 失效
```

### 6.2 §10 v0.2 之 sector-balanced loss(治本)

```
v0.2 sector-balanced loss:
  loss = MSE(predicted_ranks, actual_ranks)
       + λ × sector_penalty
  
where:
  sector_penalty = -Σ_s entropy(sector_weight_s)
                  + γ × Σ_s |sector_weight_s - target_weight_s|
  
  target_weight_s = 1/N_sectors(uniform 4 主要 sectors)
  λ = 0.3(POLICY 預設;tunable via ablation)
  γ = 0.5(tunable)
```

→ Training 階段 model 被「punished」on sector 過度集中
→ Top-20 long signals 自動 sector-balanced(預期 4 sectors × 5 stocks 各 25%)

### 6.3 治本驗證(預期 walk-forward IC trade-off)

```
v0.1 baseline:        IC ≈ 0.4 / sector_entropy ≈ 0.2(極端集中)
v0.2 sector-balanced: IC ≈ 0.35(略降)/ sector_entropy ≈ 0.7(良好分散)

→ 預期 IC 降 0.05 但 sector 結構顯著改善
→ 對 portfolio_sizer 之治標 v0.3 提供「多 sector candidates」(§14.7-AA Part C 真治本)
```

---

## 七、Multi-model ensemble 設計(v0.3 可選)

### 7.1 v0.2 主推 LGBM(robust + fast)

```
為何 LGBM:
- 既有 robust_rank_ic_baseline 已用 LGBM ✅
- handle nonlinear interactions(治 §0.0-C.3「線性合成遺漏乘積效應」)
- sector-balanced loss 可透過 custom objective 實作
- training time < 5 min(per snapshot)
```

### 7.2 v0.3 可選 ensemble(LGBM + XGBoost + linear baseline)

```
ensemble formula:
  final_score = α × LGBM_score + β × XGBoost_score + γ × Linear_score
  
  α + β + γ = 1
  α, β, γ 由 walk-forward IC 自動 tune
  
default: α=0.5 / β=0.3 / γ=0.2
```

### 7.3 v0.4 試錯 LSTM(時序考慮)

```
LSTM for sequence-aware features:
- input: 252-day rolling features per stock
- output: 30-day forward log-return rank
- 可能 capture 跨期 trend / mean-reversion
- 但複雜度高 / training time > 30 min
- 列為 v0.4 試錯,非 v0.2 主推
```

---

## 八、§14.7-X 既有研究之整合

### 8.1 §14.7-X model_trainer_feature_predictive_power_research 既有結論

從 charter L335:
> 「`model_trainer.py` 是 §0.1 第一性原理能否通過 §0.4 可觀測性審判的第一個模型層工具」

從 L417 既有揭露:
> 「現行 `robust_rank_ic_baseline` 為 in-sample 估權重,缺 nonlinear interaction;無法捕捉 T2 元素之間的乘積效應(如 M × V)」

→ **§14.7-X 之既有揭露之 root cause 跟 §14.7-AA Part C 對齊**:皆 root cause 為「線性 + 缺 sector awareness」

### 8.2 §10 解決 §14.7-X 揭露之 4 個 gap

| §14.7-X gap | §10 解決路徑 |
|---|---|
| 缺 nonlinear interaction | LGBM tree-based 自動 capture nonlinear |
| 無乘積效應 | LGBM 之 leaf interaction 自動學 M × V × ΔlnP |
| in-sample 估權重 | walk-forward IC 強制 8 panel 證明 out-of-sample |
| 缺 sector awareness | sector-balanced loss(治本) |

---

## 九、對映 §9.1 prediction contract

### 9.1 §10 → §9.1 之 handshake

```
§10 model_trainer:
  - INPUT: feature_store_snapshot + universe_snapshot
  - OUTPUT: model_artifact + model_training_run row
  - 寫入: model_registry(SSOT for §9.1 之 query)

§9.1 prediction_engine:
  - INPUT: §10 committed model + 新 as_of_date 之 feature_values
  - OUTPUT: prediction_values(rank / signal_label / confidence)
  - 不重訓 model(那是 §10)
```

### 9.2 §9.1 升 horizon=30 之配套

```
現行 §9.1 v0.2 baseline: horizon=20(歷史 walk-forward 證據)
§10 落地後:強制 horizon=30(對齊 §9.1 終極預測契約)
配套:audit_downstream_readiness.py 升 horizon=30 之 readiness rule
```

---

## 十、Phase A-D 路線圖

### Phase A: 本研究(治權先行設計研究)

- ✅ §10-A~H 八子節結構設計
- ✅ Walk-forward IC framework 設計
- ✅ Sector-balanced loss 設計
- ✅ Multi-model ensemble v0.2/v0.3/v0.4 規劃
- ✅ §14.7-X / §8.3 整合裁決
- ✅ Phase A 之 commit + push + tag v6.1.24

### Phase B: 入憲 §10 formal contract(v6.2.0 軌道)

- ⏸ 起草 §10-A~H 8 子節 charter 條文
- ⏸ 起草 §14.7-BR(候選)v0.2 落地記述子節
- ⏸ 修訂歷程加 v6.2.0 entry
- ⏸ 預估 charter +400-500 行

### Phase C: 程式落地 model_trainer.py v0.1 → v0.2(v6.2.0 軌道)

- ⏸ 加 sector-balanced loss(custom objective)
- ⏸ 加 walk-forward 自動化 framework
- ⏸ 加 15 FAIL gates + 4 audit hooks
- ⏸ 標頭 docstring 重寫(per CLAUDE.md §四 #4 8-項)

### Phase D: smoke test + commit + tag v6.1.25/v6.2.0(v6.2.0 軌道 closure)

- ⏸ walk-forward 8 panel runs
- ⏸ sector_entropy 改善驗證
- ⏸ IC trade-off 接受度(預期 0.4 → 0.35)
- ⏸ commit + push + tag

---

## 十一、證偽承諾 T_MT_v0.2-1〜10

| ID | 證偽指標 | 通過門檻 |
|---|---|---|
| **T_MT_v0.2-1** | walk-forward IC mean > 0 | 8 panel runs 全 IC > 0 |
| **T_MT_v0.2-2** | walk-forward IC std < 2 × IC mean | 跨 panel 穩定 |
| **T_MT_v0.2-3** | sector_entropy ≥ 0.5 | top 20 long 不過度集中 |
| **T_MT_v0.2-4** | top-4 sector weight ≤ 0.5 | sector diversification |
| **T_MT_v0.2-5** | Sharpe ≥ 0.5 | annualized Sharpe gate |
| **T_MT_v0.2-6** | sector-balanced loss vs baseline IC 落差 < 0.10 | 治本不過度傷 IC |
| **T_MT_v0.2-7** | top 20 long 包含 ≥ 4 sectors | 解 100% 半導體集中 |
| **T_MT_v0.2-8** | training reproducible(seed=42) | bit-exact 同 model_id |
| **T_MT_v0.2-9** | train/inference transform consistency | G10 |
| **T_MT_v0.2-10** | portfolio_sizer v0.3 + §10 v0.2 端到端 sector_count ≥ 4 | 解 §14.7-AA Part C |

→ 10 個證偽承諾 / 5 為治本核心(T3/T4/T7/T6/T10)

---

## 十二、對既有治權影響

| 項目 | 影響 |
|---|---|
| 既有 v0.2 / v0.7 snapshot | **零**(本 Phase A 不動 universe)|
| §9.2-A~H 既有 12 FAIL gate | **零** |
| §0.1-A / §0.2-A / §0.3-A 治權禁令 | **零** |
| §6.4 CoreScore 公式 | **零** |
| §6.7 universe SSOT | **零** |
| §8.3 v0.1 草案 | 升 §10 為強制契約;§8.3 標 SUPERSEDED |
| §14.7-X 既有研究 | 整合至 §10-G 跨層影響表 |
| upstream prediction_engine | §9.1 升 horizon=30 配套(另案)|

---

## 十三、v6.2.0 升版觸發條件

依 §0.0-G.6 v6.2.0 升版條件:

| 條件 | 滿足? |
|---|---|
| §10 model_trainer v0.2 落地(walk-forward IC + sector-balanced loss)| ⏸ Phase D 落地 |
| ≥ 8 walk-forward IC panel runs(IC > 0)| ⏸ Phase D 跑出 |
| sector_entropy ≥ 0.5(治本驗證)| ⏸ Phase D 驗 |
| §9.1 升 horizon=30(配套)| ⏸ Phase C 之 prediction_engine 升版 |
| §9.2 portfolio_sizer v0.3(已 ready) | ✅ commit 262560d / v6.1.22 |
| ROE/CashFlow Phase B-D(可選 prereq) | ⏸ §14.7-BI / §14.7-BM / §14.7-BO Phase B-D |
| audit_doctrine_compliance v0.x 升版 | ⏸ Phase D 配套 |

→ **§10 Phase A-D 全完成是 v6.2.0 升版之核心 gate**

---

## 十四、預期 walk-forward IC 結果(theoretical projection)

```
v0.1 baseline (current):
  IC_mean = 0.4
  sector_entropy = 0.2(極端集中於半導體業)
  Sharpe ≈ 1.0(高但 sector 集中風險)

v0.2 sector-balanced:
  IC_mean ≈ 0.35(降 0.05;治本代價)
  sector_entropy ≈ 0.7(大幅改善)
  Sharpe ≈ 0.8(略降但 risk-adjusted 仍 acceptable)

v0.3 ensemble(LGBM + XGBoost + Linear):
  IC_mean ≈ 0.38(回升 0.03;ensemble 補回部分 IC)
  sector_entropy ≈ 0.7(維持)
  Sharpe ≈ 0.85
```

→ 預期 trade-off 合理;治本價值 >> IC 略降代價

---

## 十五、Cross-Reference 精確行號

| 項目 | 位置 |
|---|---|
| §10 forward references(charter)| L5141 / L5144 / L8795 / L8893 / L8903 / L9036 |
| §0.0-A 五大轉換器之 #3 | charter L249+ |
| §8.3 model_trainer v0.1 草案 | charter L169 |
| §14.7-X model_trainer_feature_predictive_power_research | charter L335 / L341 / L417 |
| §8.8.7 Walk-Forward H20 Historical Evidence Panel | charter L162(修訂歷程) |
| §9.1 prediction contract | charter §9.1 主章 |
| §9.2-A~H portfolio_sizer | charter L4901+ |
| 姊妹 Phase A: §14.7-BO CashFlow | reports/cashflow_sync_phase_a_research_20260526.md(commit 1b08d47) |
| 姊妹 Phase A: §14.7-BM 金融業 ROE | reports/financial_sector_roe_alignment_phase_a_research_20260526.md(commit 9f64755) |
| 本 §10 Phase A | reports/model_trainer_phase_a_research_20260526.md(本檔)|

---

## 十六、治權邊界嚴守 + Phase A 結論

### 本 §10 Phase A 不改:

- §6.4 / §6.7 / §0.1-A / §0.2-A / §0.3-A 治權禁令
- §9.10 / §9.9 既有強制契約
- §9.2-A~H 既有 12 FAIL gate + §9.2-I v0.3 補強
- §14.7-BI / §14.7-BM / §14.7-BN / §14.7-BO 既有 Phase A
- builder v0.8 / portfolio_sizer v0.3 / audit v0.2(已 commit)
- raw DDL / CLI 結構

### 本 §10 Phase A 新增(僅 reports/):

- 本研究 reports/ 之 §10 Phase A 設計研究文件
- §10-A~H 八子節結構規劃
- Walk-forward IC framework + sector-balanced loss 設計
- Multi-model ensemble v0.2/v0.3/v0.4 規劃
- 證偽承諾 T_MT_v0.2-1〜10

### 本 §10 Phase B-D 未來(v6.2.0 軌道):

- 入憲 §10-A~H 8 子節 charter 條文(+400-500 行)
- model_trainer.py v0.1 → v0.2 程式落地
- walk-forward 8 panel + sector_entropy 驗證
- 入憲 §14.7-BR(候選)v0.2 落地記述
- audit_doctrine_compliance v0.x 配套升版

---

## 十七、與其他 Phase A 之關係

| Phase A | 主題 | 對 §10 之關係 |
|---|---|---|
| §14.7-BK F 升 T1 Phase A | F/IF 升 §0.1 T1 | 等 §10 walk-forward IC 證據後 Phase B |
| §14.7-BM 金融業 ROE Phase A | Banking ROE | 等 §10 IC 證偽 banking sub-score 效益 |
| §14.7-BN portfolio_sizer v0.3 Phase B | sizer 補強 | 已 Phase C-D 完成;等 §10 IC 證偽 T_PS_v0.3-5 |
| §14.7-BO CashFlow Phase A | CFS sync | 等 §10 IC 證偽 CFS 7 sub-scores 效益 |
| **§10 model_trainer Phase A(本研究)** | **治本** | **為其他 4 個 Phase B-D 之 common gate** |

→ **§10 是所有 v6.1.x Phase A 之 common gate**;治本最強優先

---

## 十八、Phase A 結論

**§10 model_trainer Phase A** 為 v6.2.0 軌道之**治本最強優先項**,類比 §14.7-BC V 補強(13 章)/ §14.7-BF F proxy(15 章)/ §14.7-BO CashFlow(15 章)Phase A 模式但**規模更大、深度更深**(18 章 + 8 子節結構規劃 + 10 證偽承諾)。

**Phase A 治權成本**: ~3 小時 / 寫入 DB:0 / Charter 入憲:0(本研究不入憲;Phase B 才入)/ 程式變更:0

**Phase B 阻塞**: charter 入憲 §10 8 子節(+400-500 行)+ 修訂歷程 entry;預估 ~2 小時

**Phase C 阻塞**: 程式落地 model_trainer.py v0.2(sector-balanced loss + walk-forward auto);預估 ~3-5 天

**Phase D 阻塞**: walk-forward 8 panel runs(若 sync 完整可即時跑)+ sector_entropy 驗證 + commit;預估 ~1 天

**完整 v6.2.0 軌道 timeline**:Phase A-D 預估 **2-3 週**(per handoff §六 unfinished item #3)

未來 Phase B-D 啟動時,可基於本 Phase A 直接展開。§10 落地為 v6.2.0 升版之 single most important gate。

---

*Report generated 2026-05-26 evening by Claude Sonnet 4.7 session*
*基於 charter §10 多處 forward reference + §14.7-X 既有研究 + §14.7-AA Part C root cause 揭露*
*Phase B-D 為 v6.2.0 軌道之最強優先項;為其他 4 個 v6.1.x Phase A 之 common gate*
