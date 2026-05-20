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
| **總計** | | **61** |

> **注**：54 個 .py 檔案 + 8 個 .md 檔案 + 部分目錄之 __init__.py + README 等補檔
>
> **root_legacy_md/ 8 檔來源**（v5.x 之審查 / 計畫 / walkthrough，皆無治權核心引用）：
>
> - `System_Audit_Report_2026-04-30.md` / `系統檢核報告_2026-04-30.md`（v5.0 後檢核 31K × 2）
> - `第三輪優化審查報告.md`（24K，2026-04-27）
> - `第四輪優化審查報告_資料抓取層.md`（22K，2026-04-27 資料抓取層專題）
> - `系統架構全面審查報告.md`（31K，2026-04-27 v4.0 Trinity 全面審查）
> - `implementation_plan.md` / `task.md` / `walkthrough.md`（v5.x 優化計畫三件套）

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
