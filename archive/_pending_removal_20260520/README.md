# 🗑️ Pending Removal Quarantine（2026-05-20）

- **建立時間**: 2026-05-20
- **目的**: 暫存 v6.0.0-FINAL 程式碼治權審計識別為「**治權層不可達**」之歷史程式
- **狀態**: ⏳ **PENDING REMOVAL**（等待全系統重建驗證後刪除）
- **依據**: `reports/v6_0_0_final_code_audit_20260520.md` Phase A 階段 1-5

---

## 此目錄之治權地位

依憲章 §0.0-G.0 治權層級類型系統，本目錄為「**非治權層歸檔**」：

- ❌ 不在 v6.0.0-FINAL 治權核心（60 個檔案）內
- ❌ 不被 scripts/core/ 五支落地鏈 import
- ❌ 不被 scripts/maintenance/ audit 工具 import
- ❌ 不被 scripts/ingestion/ 工具 import
- ✅ 保留於此目錄供 audit trail 參考；待主環境重建驗證後刪除

---

## 來源與內容

| 子目錄 | 來源 | 檔案數 |
|---|---|---|
| `scripts_scratch/` | `scripts/scratch/`（暫存實驗）| 9 |
| `scripts__patch_backup/` | `scripts/_patch_backup/`（patch 備份）| 10 |
| `scripts_archive/` | `scripts/archive/`（既有歸檔）| 16 |
| `archive_backup_v5_2_pre_fix/` | `archive/backup_v5.2_pre_fix/`（v5.2 備份）| 6 |
| `archive_backup_v5_2_stable_core/` | `archive/backup_v5.2_stable_core/`（v5.2 stable 備份）| 6 |
| `archive_legacy_scripts/` | `archive/legacy_scripts/`（legacy 程式）| 6 |
| `root_legacy_md/` | 根目錄 v5.x 時代 audit / plan / 計畫檔（2026-05-20 加入）| 8 |
| `root_legacy_txt/` | 根目錄 v5.x 時代 health check / schema 稽核 / dependency raw dump（2026-05-20 加入）| 3 |
| `root_legacy_py/` | 根目錄 v5.x 時代 config / FastAPI auth / GitHub-Gemini 整合（2026-05-20 加入）| 3 |
| `root_legacy_sql/` | 根目錄 v5.x legacy DDL（憲章 §9.1-B 明文禁止讀取之 stock_forecast_daily 表）| 1 |
| `maintenance_v5x_legacy/` | maintenance v5.2 標準 + 治權 0 引用 + 確定 legacy（2026-05-20 加入；路徑丙 Y）| 13 |
| **總計** | | **81** |

> **注**：70 個 .py 檔案（54 既有 + 3 root + 13 maintenance）+ 8 個 .md 檔案 + 3 個 .txt 檔案 + 部分目錄之 __init__.py + README 等補檔
>
> **root_legacy_md/ 8 檔來源**（v5.x 之審查 / 計畫 / walkthrough，皆無治權核心引用）：
>
> - `System_Audit_Report_2026-04-30.md` / `系統檢核報告_2026-04-30.md`（v5.0 後檢核 31K × 2）
> - `第三輪優化審查報告.md`（24K，2026-04-27）
> - `第四輪優化審查報告_資料抓取層.md`（22K，2026-04-27 資料抓取層專題）
> - `系統架構全面審查報告.md`（31K，2026-04-27 v4.0 Trinity 全面審查）
> - `implementation_plan.md` / `task.md` / `walkthrough.md`（v5.x 優化計畫三件套）
>
> **root_legacy_txt/ 3 檔來源**（執行 artifact / 應屬 outputs/ 或 logs/ 但歷史落到 root）：
>
> - `health_check_output.txt`（6.9K，2026-05-01 v5.0 時代 Antigravity Quant health check 輸出）
> - `schema_check_report.txt`（5.6K，2026-05-10 schema 稽核日誌）
> - `requirements_raw.txt`（84B，依賴掃描 raw dump；含 stdlib `re/multiprocessing` 混雜，非權威清單）
>
> **root_legacy_py/ 3 檔來源**（v5.x 時代根目錄程式；migrate_stocks_config.py 已明文標註「config.py 已被列為過時(Legacy)數據源」）：
>
> - `config.py`（30K，全域設定 + STOCK_CONFIGS 硬編；v6.0 已用 stocks 表治權取代）
> - `main.py`（2.7K，FastAPI auth 服務；`from scripts.config import DB_CONFIG`；配套於 config.py）
> - `github_gemini_sync.py`（3.9K，GitHub + Gemini AI integration；standalone 工具）
>
> **root_legacy_sql/ 1 檔來源**（v5.x legacy DDL；憲章 §9.1-B 明文列為禁止輸入）：
>
> - `create_table.sql`（1.6K，定義 `stock_forecast_daily` 表 — 憲章 L405/L503/L4014/L4165 明文標註為「舊資料流」，§9.1-B 強制禁止讀取；v6.0 已用 `prediction_run` + `prediction_values` 治權取代）
>
> **maintenance_v5x_legacy/ 13 檔來源**（皆 0 憲章引用 + 0 完整度評估報告引用；路徑丙 Y 激進版隔離）：
>
> *Category C — v5.2 標準 / 內部循環（7 檔）*：
>
> - `check_data_integrity.py` v3.4（Quantum v5.2 標準）
> - `check_db_locks.py` v1.3（Quantum v5.2 標準；唯一引用方 fetchers/check_db_locks.py 亦為 legacy）
> - `check_finmind_datalist.py` v3.1（Quantum v5.2 標準；唯一引用方 fetchers/check_finmind_datalist.py 亦為 legacy）
> - `enrich_stocks_metadata.py` v2.3（Quantum v5.2 標準；唯一引用方為 initialize_and_enrich_stocks.py 內部循環）
> - `initialize_and_enrich_stocks.py` v1.2（Quantum v5.2 標準）
> - `log_analyzer.py` v1.1（Quantum v5.2 標準）
> - `test_block_trading.py` v1.3（Quantum v5.2 標準）
>
> *Category B — v6.0 PERFECT 標頭但憲章 0 引用（4 檔；標頭聲稱對齊但實證未被任何治權文件引用）*：
>
> - `check_finmind_quota.py` v1.46（標頭 PERFECT 全譜治權對齊，實證 0 引用）
> - `check_schema_consistency.py` v2.12（同上）
> - `check_system_health.py` v2.32（同上）
> - `verify_core_integrity.py` v1.82（同上）
>
> *Category E — 確定 Legacy（2 檔）*：
>
> - `sync_stocks_from_config.py`（`from config import STOCK_CONFIGS`；v6.0 已用 `migrate_stocks_config.py` 取代）
> - `test_finmind_raw.py`（無 docstring；ad-hoc 簡易測試）
>
> **可逆性說明**：若主環境執行時發現任何被隔離檔仍被治權核心呼叫，可由 `git mv archive/_pending_removal_20260520/<subdir>/<file> <original_path>/` 反向恢復，git history 完整保留。
>
> **maintenance 治權核心（隔離後殘留 8 檔）**：
>
> - `audit_supply_chain.py` v1.18（憲章 49 次引用）
> - `audit_core_universe.py` v0.1（43 次）
> - `audit_leakage.py` v0.2（35 次）
> - `audit_downstream_readiness.py` v0.2（31 次）
> - `audit_doctrine_compliance.py` v0.3（26 次）
> - `audit_source_availability.py` v0.1（11 次）
> - `_oneoff_v02_ablation.py`（§0.0-D.6 引用）
> - `_oneoff_v03_upside_downside_ablation.py`（§9.9 / §14.7-AD/AE 引用）

---

## 為什麼採「移到備份」而非「直接刪除」

依用戶判斷「先移到備份資料夾，待全系統重建完整後再進行移除」之策略：

### 安全網優勢

1. **可逆性**：若主環境重建發現任何被「意外引用」的歷史程式，可從本目錄恢復
2. **零治權風險**：審計已確認治權層不依賴這些檔案，但實證重建為最後驗證
3. **git 歷史保留**：使用 `git mv` 保留每個檔案之 commit 歷史
4. **明確標記**：本目錄名稱 `_pending_removal_` 清楚表達其狀態

### 刪除條件（**全部達成後才刪**）

- [ ] 主環境執行 `reports/v6_0_0_final_db_rebuild_runbook_20260520.md` 全流程
- [ ] schema bootstrap PASS
- [ ] sovereign_sync_engine（FinMind + FRED + market + core）全部 PASS
- [ ] 五支落地鏈 end-to-end 重建 PASS
- [ ] audit_doctrine_compliance v0.3 PASS
- [ ] 產出 `reports/v6_0_0_final_rebuild_audit_<DATE>.md` 入憲為 §14.7-AH
- [ ] 用戶確認可刪除

### 刪除指令

```bash
# 全部達成後執行
git rm -r archive/_pending_removal_20260520/
git commit -m "chore: remove pending_removal quarantine after v6.0.0-FINAL rebuild verification (§14.7-AH)"
git push
git tag v6.0.0-FINAL-cleanup-completed
```

---

## 重要警告

- ⚠️ **不要從本目錄 import 任何程式至 scripts/core/、scripts/maintenance/、scripts/ingestion/**
- ⚠️ **不要在本目錄內新增任何新檔案**
- ⚠️ **本目錄之檔案不受 v6.0.0-FINAL 治權結構保護**
- ⚠️ **本目錄即將被刪除**（依條件達成）

---

## 對照憲章

| 憲章節 | 治權地位 |
|---|---|
| §0.0-G.0 Type-3 實作層 | 本目錄為「非治權層歸檔」，明文排除 |
| §0.0-G.7 適用範圍 | 不適用 §0.0-G Level 1 流程 |
| §0.0-I 單一引用源 | 本目錄不被引用，無 single-source 衝突 |
| `v6_0_0_final_code_audit_20260520.md` Phase A 階段 1-5 | 本目錄為審計之「明確可移除清單」實現 |

---

**本目錄將於 v6.0.0-FINAL 完整重建驗證通過後刪除。**
