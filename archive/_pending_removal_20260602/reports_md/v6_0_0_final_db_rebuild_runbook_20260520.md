# v6.0.0-FINAL 完整 DB 重建 Runbook（含外部 API 接口驗證）

- **生成時間**: 2026-05-20 Asia/Taipei
- **基準**: HEAD `a0ba83c` / Tag `v6.0.0-FINAL-readme-aligned`
- **憲章基準**: `reports/系統架構大憲章_v6.0.0.md` v6.0.0-FINAL
- **配套文件**: `reports/v6_0_0_final_code_audit_20260520.md`（程式碼治權審計）
- **目的**:
  1. 驗證 v6.0.0-FINAL 治權結構之 schema 對齊
  2. 驗證程式碼與外部 API（FinMind / FRED）接口之**完美一致性**
  3. 從零到 production-current 之全流程可重現性實證
- **執行環境**: 主環境（venv + PostgreSQL + DB credentials）
- **預估時間**: 1-2 天（依 core 150 stocks vs 全市場 sync 選擇）

---

## 0. 前置條件

### 0.1 環境檢查

```bash
# 在主環境 /home/hugo/project/stock_backend/

# 1. venv 啟用
source venv/bin/activate
python -c "import psycopg2; print('psycopg2 OK')"

# 2. DB 連線環境變數
cat .env | grep -E "DB_HOST|DB_PORT|DB_NAME|DB_USER|FINMIND_TOKEN"

# 3. 確認分支與 commit
git branch --show-current  # 應為 master
git log --oneline -1       # 應為 v6.0.0-FINAL 相關
```

### 0.2 備份既有 DB（強烈建議）

```bash
# 完整 DB dump（時間: 30 分鐘-2 小時，依資料量）
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME \
  -F c -f backup_pre_rebuild_$(date +%Y%m%d_%H%M%S).dump

# 驗證 backup 完整性
pg_restore -l backup_pre_rebuild_*.dump | head -20
```

### 0.3 程式碼預清理（可選，依審計報告階段 A）

```bash
# 移除明確的歸檔/備份/暫存（55 個檔案）
git rm -r scripts/scratch/
git rm -r scripts/_patch_backup/
git rm -r scripts/archive/
git rm -r archive/backup_v5.2_pre_fix/
git rm -r archive/backup_v5.2_stable_core/
git rm -r archive/legacy_scripts/
git commit -m "chore: remove archive/backup/scratch files (v6.0.0-FINAL cleanup)"
git push origin master
```

---

## 一、空 DB 重建總流程（依憲章 §6.8 同步治權審計）

### 1.1 重建步驟矩陣

```text
Step 1   path_setup        路徑 SSOT 對齊        ~10 秒
Step 2   data_schema       建立 13 張 raw tables  ~1 分
Step 2B  core_universe_schema  建立 7 張治理表  ~1 分
Step 2C  db_utils          診斷連線              ~10 秒
Step 3   audit_supply_chain    供應鏈速率審計     ~1 分
Step 4   sovereign_sync_engine seed         初始化 seed data   ~5 分
Step 5   sovereign_sync_engine fred         FRED 4 序列同步     ~3 分
Step 6   sovereign_sync_engine market        全市場掃描         ~10 分
Step 7   sovereign_sync_engine core           Core 150 sync     ~6-8 小時
Step 7A  core 全天數補刷                                       ~2-4 小時
Step 8   audit_core_universe                  核心宇宙審計     ~1 分
Step 9   core_universe_builder --commit       150 universe 建立 ~5 分
Step 10  feature_store_builder --commit       v0.3 31 features  ~3 分
Step 11  model_trainer --commit               h20 / h30 訓練    ~5-10 分
Step 12  prediction_engine --commit           prediction-backed ~2 分
Step 13  portfolio_sizer --commit-report      allocation 產出   ~1 分
Step 14  audit_doctrine_compliance v0.3       治權合規審計     ~5 分
Step 15  audit_supply_chain final             最終供應鏈審計   ~1 分
```

**總計**：~12-20 小時（依 sync 範圍與機器規格）。

### 1.2 §6.8.7 全市場全天數限定治理例外

依憲章，全市場全天數同步為「限定治理例外」，需依 §6.8.7-B 規範執行：

```bash
# 全市場長跑持久化（建議 tmux + 30 分鐘監控）
tmux new -s full_market_rebuild_$(date +%Y%m%d)
python scripts/ingestion/sovereign_sync_engine.py --full-history --universe market \
  2>&1 | tee logs/full_market_sync_$(date +%Y%m%d_%H%M%S).log
# Ctrl+B D detach

# 30 分鐘監控腳本
./scripts/maintenance/full_market_sync_*_monitor.sh
```

---

## 二、外部 API 接口驗證（核心目的）

### 2.1 FinMind API 接口驗證

依 v6.0.0-FINAL 之 `scripts/core/finmind_client.py v4.46`，驗證 13 個 raw tables 之接口：

| 資料表 | FinMind dataset | 對應 ingest 工具 | 驗證項 |
|---|---|---|---|
| `TaiwanStockPrice` | `TaiwanStockPrice` | `ingest_technical_data.py` | 10 columns |
| `TaiwanStockPriceAdj` | `TaiwanStockPriceAdj` | `ingest_price_adj_data.py` | 10 columns |
| `TaiwanStockPER` | `TaiwanStockPER` | `ingest_fundamental_data.py` | 5 columns |
| `TaiwanStockInstitutionalInvestorsBuySell` | 同名 | `ingest_chip_data.py` | 5 columns |
| `TaiwanStockMarginPurchaseShortSale` | 同名 | `ingest_advanced_chip_data.py` | 16 columns |
| `TaiwanStockShareholding` | 同名 | `ingest_advanced_chip_data.py` | 13 columns |
| `TaiwanStockFinancialStatements` | 同名 | `ingest_fundamental_data.py` | 5 columns |
| `TaiwanStockMonthRevenue` | 同名 | `ingest_month_revenue.py` | 7 columns |
| `TaiwanStockTotalReturnIndex` | 同名 | `ingest_total_return_index.py` | 待確認 |
| `TaiwanStockInfo` | 同名 | `ingest_stock_info.py` | 8 columns |
| `TaiwanStockCashFlowsStatement` | 同名 | `ingest_cash_flows_data.py` | 待確認 |
| `TaiwanStockDerivative*` | 多個 | `ingest_derivative_*.py` | 待確認 |

**驗證指令**：

```bash
# 完整 API contract 審計（依 v6.0.0 之 data_schema v2.11）
python scripts/maintenance/audit_supply_chain.py --include-logs

# 預期輸出：API-Contract PASS=11+ / WARN=0 / FAIL=0
# 報告: reports/compliance_audit_<YYYYMMDD_HHMM>.md
```

**裁決規則**：

| 結果 | 判定 | 動作 |
|---|---|---|
| API-Contract PASS=All / FAIL=0 | ✅ 接口完美一致 | 繼續 |
| 任一 FinMind dataset FAIL | ❌ 接口不一致 | 修補 `data_schema.py` 或對應 ingest 工具 |
| API-Quota 超限 | ⚠️ 暫停並等配額重置 | 等 |

### 2.2 FRED API 接口驗證

依 v6.0.0-FINAL 之 ingest_fred_data.py，驗證 4 個 FRED 序列：

| 序列 | FRED ID | 用途 |
|---|---|---|
| Fed Funds Rate | DFF | macro_dff_level |
| VIX | VIXCLS | macro_vix_level |
| 10Y-2Y Spread | T10Y2Y | macro_t10y2y_level |
| Unemployment | UNRATE | macro_unrate_yoy |

**驗證指令**：

```bash
python scripts/ingestion/sovereign_sync_engine.py --seed --source fred

# 預期：FRED 4 序列合計 ~3000-4000 筆 upsert
# Audit log: reports/fred_sync_<DATE>.md（如有產出）
```

### 2.3 接口一致性實證裁決

執行以下三重驗證：

```bash
# 1. Schema 對齊
python scripts/core/data_schema.py --init --force
# 預期: PERFECT ALIGNMENT, PASS=11, WARN=0, FAIL=0

# 2. Supply chain 完整審計
python scripts/maintenance/audit_supply_chain.py
# 預期: 全 PASS

# 3. Source availability 審計
python scripts/maintenance/audit_source_availability.py
# 預期: 全 PASS
```

**三重 PASS = 接口完美一致**。

---

## 三、v6.0.0 治權結構壓力測試（新增子節驗證）

### 3.1 §9.9 P1 v0.1 feature_store v0.3 重建

```bash
# 重建 v0.3 feature set
python scripts/core/feature_store_builder.py \
  --commit --as-of-date 2026-04-25 \
  --feature-set-version feature_set_v0.3 --label-horizon 20

# 預期：
# - feature_set_id = fs_20260425_feature_set_v0_3
# - feature_definition rows = 31 (27 base + 4 upside/downside)
# - feature_values rows ~4500-4600
# - verdict = PERFECT
```

### 3.2 §9.2 v0.2 portfolio_sizer 重建

```bash
# 重建 prediction → sizing 流程
python scripts/core/model_trainer.py \
  --commit --feature-set-id fs_20260425_feature_set_v0_3 \
  --label-horizon 20

python scripts/core/prediction_engine.py \
  --as-of-date 2026-04-25 --commit

python scripts/core/portfolio_sizer.py --commit-report --as-of-date 2026-04-25

# 預期：
# - portfolio_sizer v0.2 通過全部 10 條 FAIL gate (G1-G10)
# - G11 as_of_date 一致性 PASS
# - G12 single_sector_count_max=5 觸發或不觸發（依資料）
# - ConstitutionalViolationError 未拋出
# - allocation proposal 產出於 reports/
```

### 3.3 P1 v0.2 ablation（v0.3 features 實證 IC）

```bash
# 編輯 _oneoff_v03_upside_downside_ablation.py 設定 MODEL_ID_V03
# 然後執行
python scripts/maintenance/_oneoff_v03_upside_downside_ablation.py

# 裁決規則（依 §14.7-AD）：
# - 4/4 HELPFUL → 升版條件達成 → 啟動 P1 v0.3
# - 1-3/4 HELPFUL → 部分達成
# - 0/4 HELPFUL → 否決（同 P2 教訓）
```

### 3.4 §9.1-A〜I 八子節驗證

```bash
# prediction_engine v0.2 之 §8.8.8 single-delivery 自動化驗證
python scripts/core/prediction_engine.py --as-of-date 2026-04-25 --commit
# 預期：exactly-one prediction-backed run maintained

# 驗證 single-delivery 自動化
psql -c "
  SELECT COUNT(*) FROM prediction_run
  WHERE status='committed'
    AND prediction_policy_version='prediction_policy_v0.1';"
# 預期：1
```

---

## 四、產出 audit 報告

### 4.1 v6.0.0-FINAL rebuild audit 報告範本

執行完成後產出 `reports/v6_0_0_final_rebuild_audit_<YYYYMMDD>.md`：

```markdown
# v6.0.0-FINAL DB 重建 Audit 報告

- **執行時間**: <YYYY-MM-DD HH:MM>
- **commit**: <git rev-parse HEAD>
- **rebuild 範圍**: <Core 150 / 全市場>
- **總執行時間**: <X 小時>

## 一、外部 API 接口驗證

### FinMind 13 dataset 接口
| Dataset | Status | Rows | Notes |
|---|---|---|---|
| <逐項填入> | | | |

### FRED 4 序列接口
| Series | Status | Rows | Notes |
|---|---|---|---|
| <逐項填入> | | | |

## 二、v6.0.0 治權結構壓力測試

### 五支落地鏈重建結果
| 程式 | Status | 產出 | Verdict |
|---|---|---|---|
| <逐項填入> | | | |

### 新治權結構驗證
- §9.9 v0.3 feature set: <PASS/FAIL>
- §9.2 v0.2 portfolio_sizer: <PASS/FAIL>
- §8.8.8 single-delivery: <PASS/FAIL>
- §9.1-A〜I 八子節: <PASS/FAIL>

## 三、P1 v0.2 ablation 結果

<填入 ablation JSON 結果>

## 四、裁決
- API 接口一致性: <PASS/FAIL>
- 治權結構壓力測試: <PASS/FAIL>
- P1 v0.2 ablation: <HELPFUL/HARMFUL/PART>

## 五、後續動作
<依結果列出>
```

### 4.2 入憲建議

執行完成後建議入憲為 **§14.7-AH**（程式碼治權審計 + DB 重建驗證合併報告）。

---

## 五、回滾安全網

### 5.1 若 rebuild 中途失敗

```bash
# 從 pre-rebuild backup 還原
pg_restore -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c \
  backup_pre_rebuild_<TIMESTAMP>.dump
```

### 5.2 若特定 dataset API 失敗

```bash
# 跳過失敗 dataset，繼續其他
python scripts/ingestion/sovereign_sync_engine.py --core \
  --skip-dataset TaiwanStockShareholding  # (範例)
```

### 5.3 若 audit 發現新 schema drift

```bash
# 不要強行繼續；回到憲章 §0.0-G Level 1 流程
# 起草 §X.x 修補契約 → 入憲 → 實作 → 驗證
```

---

## 六、驗收 checklist

執行完成後逐項勾選：

- [ ] schema bootstrap: data_schema + core_universe_schema PERFECT
- [ ] audit_supply_chain: 全 PASS
- [ ] sovereign_sync_engine seed + fred + core: 全成功
- [ ] core_universe_builder --commit: 150 stocks committed
- [ ] feature_store_builder --commit v0.3: 31 features × 150 stocks
- [ ] model_trainer --commit: model_registry 寫入
- [ ] prediction_engine --commit: exactly-one prediction-backed
- [ ] portfolio_sizer --commit-report: allocation proposal 產出
- [ ] audit_doctrine_compliance v0.3: 通過
- [ ] FinMind 13 dataset 全部 PASS
- [ ] FRED 4 序列全部 PASS
- [ ] G11 as_of_date 一致性 PASS
- [ ] G12 single_sector_count_max=5 行為符合預期
- [ ] ConstitutionalViolationError 未拋出
- [ ] 產出 reports/v6_0_0_final_rebuild_audit_<DATE>.md
- [ ] commit + push + tag `v6.0.0-FINAL-rebuild-validated`

---

## 七、與 Phase A 程式碼治權審計之配合

| Phase | 範圍 | 時點 |
|---|---|---|
| **Phase A 程式碼治權審計** | 靜態分析 + 識別無用程式 | ✅ 已完成（reports/v6_0_0_final_code_audit_20260520.md）|
| **Phase A 階段 1-5 移除** | 移除 55 個明確檔案 | ⬜ 待用戶確認 |
| **Phase B DB 重建** | 動態驗證 + API 接口 | ⬜ 主環境執行（本 runbook）|
| **Phase B 結果入憲** | §14.7-AH 入憲 | ⬜ 待 rebuild 完成 |

---

## 八、執行時機建議

### 8.1 立即可做（worktree 端）

- ✅ 移除 55 個明確檔案（Phase A 階段 1-5）
- ✅ commit + push + tag

### 8.2 主環境執行（時間充裕時）

- ⬜ Phase B DB 重建（本 runbook）
- 預估 1-2 天

### 8.3 完成後產出

- `reports/v6_0_0_final_rebuild_audit_<DATE>.md` 入憲為 §14.7-AH
- 新 git tag `v6.0.0-FINAL-rebuild-validated`

---

**本 runbook 為 v6.0.0-FINAL 之主環境執行指引，配套 `reports/v6_0_0_final_code_audit_20260520.md` 程式碼審計報告**。
