# 跨機接續完整 Handoff:§14.7-DD Tree-family 12-PHASE 從零重建完成(2026-06-01 v6.26.0 sealed)

**Subject**:用戶換另一台電腦繼續本專案之 cross-machine session handoff
**Sealed checkpoint(tag)**:`v6.26.0-section14-7-DD-12phase-from-zero-complete-9trainers-9validators-20260601`
**HEAD commit**:`65ae071`(前序 `ea2b834` v6.25.1 全市場 DB↔API 對帳)
**Remote**:`https://github.com/tsaitsangchi/stock_backend`(master + tag 已 push,`git ls-remote` 驗證)
**Charter**:`reports/系統架構大憲章_v6.1.0.md`(§14.7-DD / §14.7-DC v0.16 / §0.3-A 多尺度循環修正案)
**AI 工具規則**:`CLAUDE.md`(§一.13 v0.16 / §一.14 / §一.15)

> 本文件全部數字 source = isolated venv psql READ-ONLY query 或 git / 程式 stdout(per CLAUDE.md §一.10);無記憶推測值。

---

## 一、本封存點核心狀態(DB / git 實證快照)

### 1.1 Git 封存狀態
- HEAD `65ae071`(master,與 origin/master 同步:ahead/behind = 0 0)
- 本 commit 落地:12 code 修正 + 13 新增(9 JSON + 2 driver + 2 報告),共 25 files / +1416 −25
- 最新 tag = `v6.26.0-...`(annotated,指向 65ae071)

### 1.2 Universe 狀態(committed,DB-verified)
```
snapshot_id      : core_universe_20260601_core_universe_policy_v0_18_source_pure_panhistorical_gate
policy_version   : core_universe_policy_v0.18_source_pure_panhistorical_gate
status           : committed   (created 2026-06-01 04:18:39)
total_candidates : 2,774
core_count       : 914   ⭐(core_universe tier)
quarantine_count : 213   (imputed / source-impure 強制排除)
research_count   : 0
```

### 1.3 Feature Store 狀態(DB-verified)⚠️ 含重要 nuance
- **feature_set_v0_5 在 DB 實為 44 features**(非 runbook / §一.14 所寫的 37)
  - 37 baseline(SPEC_37,amihud 移除後)**+ 7 個 §0.3-A K-wave 多尺度 source-pure 循環投影特徵**(v6.25.0 / task #35 加入 `feature_store_builder.py`):
    - `cycle_phase_5d` / `cycle_phase_20d` / `cycle_phase_60d` / `cycle_phase_252d`
    - `macro_beta_t10y2y` / `macro_beta_unrate` / `macro_beta_ipg3344s`
  - DB 實證:`fs_20260415_feature_set_v0_5` distinct features = 44,7 個 K-wave 特徵全在
- **v0_5 panel 總數 = 96**:1 個 current(`fs_20260601`,1,584 stocks)+ 95 個 historical walk-forward(各 914 stocks × 44 features,~40,200 rows/panel)

> ⚠️⚠️ **跨機關鍵未竟事項(§一.8 honest disclosure)**:7 個 K-wave 特徵**只建進 feature store,尚未接進 trainer / validator**。
> - `model_trainer_*.py` 與 `multi_cycle_*_validation.py` 的 `SPEC_43` 變數**實際 len = 37**(grep 驗證),**不含** 7 個 K-wave 特徵。
> - 故 **PHASE 10 模型 + PHASE 11 驗證結果均只用 37 features,K-wave 7 特徵對結果零影響**。
> - 若要讓模型實際使用 K-wave 特徵 → 須將 7 個 feature name 加入各 trainer/validator 的 SPEC list(目前命名 `SPEC_43`,內容 37)→ retrain + re-validate。此為 v6.27.x 候選工作,**未授權不 auto-run**。

### 1.4 Model 狀態(model_registry,DB-verified;PHASE 10 產出 2026-06-01 05:23–05:24)
```
model_family    rows   (latest created_at 2026-06-01)
xgboost          2
xgboost (ded)    └─ 含 dedicated;family 標記 xgboost
catboost         2
lgbm             9     (base model_trainer.py --walk-forward 8 panel + 1)
lightgbm         1
random_forest    1
extra_trees      1
ensemble_tree    1
TOTAL registry   17 rows
```
- artifacts 在 `data/models/`(gitignored,**不在 repo**,屬本機產物)
- 全部 train 於 v0_5 panel **37-feature 子集**(見 1.3 nuance)

### 1.5 PHASE 11 驗證 + T_CZ-6 裁決(來源:9 份 `reports/multi_cycle_*_20260601_*.json` + /tmp/phase11.log)
- 9/9 validator `exit=0`(`validator_pass=9 validator_fail=0`);seed 5422 / n_universe 914 / n_panels 95 / 全 (b) DB query
- 完整三閾值 **T_CZ-6(Eff t≥4.20 ∧ Sharpe≥2.40 ∧ Win≥79%)通過 3/36 格**:

| Validator | Horizon | Eff t | Sharpe | Win% |
|---|---|---:|---:|---:|
| xgboost | quarterly | +4.426 | +2.530 | 82.8% |
| xgboost | annual | +4.526 | +5.874 | 95.1% |
| xgboost_dedicated | annual | +4.628 | +5.937 | 95.1% |

> ⚠️ **finding B4(誠實揭露)**:T_CZ-6 **非 code-enforced**;validator 程式唯一硬判 `is_significant_p05 = abs(eff_t)>1.997`(p<0.05)。上表 PASS 為**依三閾值人工裁決**,非程式自動 gate。

---

## 二、§14.7-DD 12-PHASE 完成狀態

| PHASE | 階段 | 狀態 |
|---|---|---|
| 0 | 環境(venv + .env + PostgreSQL + import smoke) | ✅ |
| 1 | Schema DDL | ✅ |
| 2 | Genesis 名冊(`sovereign_sync_engine.py --seed`) | ✅ |
| 3 | Bootstrap 宇宙(`core_universe_builder.py --bootstrap`,§14.7-AM transitional research_universe) | ✅ |
| 4 | 全市場全歷史同步 + FRED | ✅(報告 `full_market_sync_20260601.md`) |
| 5 | Raw audit 對帳(DB↔API) | ✅(報告 `full_db_vs_api_reconciliation_20260601.md`) |
| 6 | 最終宇宙(real-data final build,policy v0.18 / 914 core) | ✅ |
| 7 | Feature Store(v0_5 / **44** features / current + 95 panels) | ✅ |
| 8 | 宇宙 source-pure gate 收緊(`--with-feature-gate`) | ✅ |
| 9 | Feature audit(IC / sign-stability / necessity) | ✅ |
| 10 | 模型訓練(base + 9 tree-family,17 registry rows) | ✅(僅 37-feature 子集) |
| 11 | 模型驗證(9 multi_cycle validators + universe_completeness MV) | ✅(9/9 pass) |

> PHASE 12(週度 doctrine recommit)為維運層,**不在從零序列內**(per §一.14)。

---

## 三、治權 doctrine 狀態(charter + CLAUDE.md 雙層鎖)

- **§14.7-DD**:Tree-family 12-PHASE 從零範本(本次完整執行之治權 SSOT)
- **§14.7-DC v0.16**:Source-Pure Universe Doctrine;T_DC-1 ~ T_DC-30 全 inscribed
  - T_DC-27:trainer/validator 載入 query 必補 `is_null_imputed IS NOT TRUE`(9 validator 本 commit 已補齊)
  - T_DC-28:amihud dead-feature 移除(SPEC_38 → 37)
  - T_DC-29:membership eligibility-flag binding(writer 明設 4×TRUE,不依賴 DEFAULT FALSE)
  - T_DC-30:K-wave 多尺度循環 per-stock source-pure 投影(4 strict 要件 + IC-gated)
- **§0.3-A 多尺度循環思想修正案**:K-wave reframe 為週(5d)/月(20d)/季(60d)/年(252d)循環思想(非 40-60 年特定值);7 features 已 build(見 1.3 nuance)
- **CLAUDE.md**:§一.13 v0.16(source-pure)/ §一.14(12-PHASE enforcement)/ §一.15(K-wave 投影)
- **DURABLE 紀律**:[[no-manual-data-fill]] — **永不手動 UPDATE/INSERT 補值**,只能 DELETE 過期列或改 writer code 重 build;[[core_stock_selection_doctrine]]

---

## 四、跨機重建路徑(CRITICAL — 新機器照此做)

### 4.1 取得 repo
```bash
git clone https://github.com/tsaitsangchi/stock_backend.git
cd stock_backend
git checkout v6.26.0-section14-7-DD-12phase-from-zero-complete-9trainers-9validators-20260601
# 或直接用 master HEAD 65ae071
```

### 4.2 環境前置(per §0.0-I.9 / CLAUDE.md §二.7)
- **OS 原生依賴**:
  - macOS:`brew install libomp postgresql@17`
  - Linux:`sudo apt-get install -y libgomp1 libpq-dev postgresql`
- **venv**:`python3 -m venv venv && ./venv/bin/pip install -r requirements.txt`
- **`.env`**(⚠️ **不在 repo;新機器須自建**;以下僅 key 名,VALUE 由你填):
  ```
  ENV / LOG_LEVEL / TZ / PROJECT_ROOT / MLFLOW_TRACKING_URI
  DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
  GEMINI_API_KEY / FINMIND_TOKEN / FRED_API_KEY / GITHUB_TOKEN
  ```
  - `PROJECT_ROOT` 必對齊新機器物理路徑(macOS `/Users/<user>/...`;path_setup.py 用 realpath 比對)
  - FINMIND_TOKEN 須 sponsor tier(全市場同步需要)
- **import smoke test(必過才往下)**:
  ```bash
  ./venv/bin/python -c "import psycopg2, pandas, polars, numpy, requests, sklearn, xgboost, lightgbm, catboost; print('✅ all imports OK')"
  ```

### 4.3 DB 重建(二選一)
DB 為**本機 PostgreSQL,13 GB / 32 public tables,不在 git**。新機器二選一:

- **路徑 A — pg_dump 搬遷(快,推薦若可存取舊機)**:
  ```bash
  # 舊機:  pg_dump -Fc -d <DB_NAME> -f stock_backend_v6.26.0.dump
  # 新機:  createdb <DB_NAME> && pg_restore -d <DB_NAME> stock_backend_v6.26.0.dump
  ```
  還原後即達本封存點 DB 狀態(policy v0.18 / 914 core / v0_5 96 panels / 17 models)。

- **路徑 B — 從零重建(完全可重現,~6-10 hr;PHASE 4 全市場同步最久)**:
  依 `reports/tree_based_from_zero_build_runbook_20260531.md` 跑 §14.7-DD PHASE 1→11。
  - ⚠️ runbook 寫「37 features」為 amihud 移除後 baseline;**現行 v0_5 = 44**(+7 K-wave),重 build PHASE 7 會自然產出 44(因 `feature_store_builder.py` 已含 7 特徵)
  - PHASE 4(≥30 min)→ §二.6 SHMM 強制;PHASE 4/10/11(≥5 min)→ §一.12 5-min 回報強制
  - driver 參考:`scripts/evaluation/_phase10_driver.sh`(PHASE 10)、`_phase11_driver.sh`(PHASE 11)

---

## 五、未竟事項 / 下一步(honest,皆**未授權不 auto-run**)

1. **🔴 K-wave 7 特徵未接入模型**(最重要):7 個 cycle_phase/macro_beta 已在 DB panel(44 features),但 trainer/validator 的 `SPEC_43`(實 len=37)未納入 → PHASE 10/11 未用到。若要啟用 → 各 trainer/validator SPEC list 補 7 名 → retrain + re-validate(v6.27.x 候選)。
2. **T_CZ-6 非 code-enforced(B4)**:validator 只硬判 p<0.05;完整三閾值 gate 須人工或補 code enforce。
3. **completeness_20260601 snapshot stale-FK 綁 superseded v0_13**:資料正確,**不手動 patch**;下次 `feature_store_builder.py` 對 v0.18 重跑自癒。
4. **既有 model artifacts 未 retrain**:任何 feature/universe 變更後 retrain 為獨立授權項。
5. **universe_completeness MV 已建**:5,484 列唯一 / 綁 v0.18 / 6 pillar × feature × 914 股全 100.00%。

---

## 六、新機器狀態驗證指令(到位後可跑以確認對齊本封存點)

```bash
# git 對齊
git log --oneline -1                      # 應為 65ae071
git tag --sort=-creatordate | head -1     # 應為 v6.26.0-...

# DB 對齊(committed universe)
./venv/bin/python -c "
import os,psycopg2; from dotenv import load_dotenv; load_dotenv('.env')
c=psycopg2.connect(host=os.environ['DB_HOST'],port=os.environ['DB_PORT'],dbname=os.environ['DB_NAME'],user=os.environ['DB_USER'],password=os.environ['DB_PASSWORD']); c.set_session(readonly=True); cur=c.cursor()
cur.execute(\"SELECT policy_version,core_count,quarantine_count FROM core_universe_snapshot WHERE status='committed' ORDER BY created_at DESC LIMIT 1\"); print('universe:',cur.fetchone())
cur.execute('SELECT count(DISTINCT feature_set_id) FROM feature_values WHERE feature_set_id LIKE %s',('%feature_set_v0_5',)); print('v0_5 panels:',cur.fetchone()[0])
cur.execute('SELECT count(*) FROM model_registry'); print('model_registry rows:',cur.fetchone()[0])
"
# 預期:universe ('core_universe_policy_v0.18_source_pure_panhistorical_gate', 914, 213) / v0_5 panels 96 / model_registry 17
```

---

**封存印記**:本 handoff = v6.26.0 sealed checkpoint 之跨機接力載體;承接 v6.25.1。三層治權鎖(主憲章 §14.7-DD/DC/0.3-A + CLAUDE.md §一.13/14/15 + 本文件)。無 DB 手動補值,既有 model 未 retrain — 皆 §一.8 honest disclosure。
