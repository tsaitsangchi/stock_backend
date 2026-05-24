# POST-AUDIT 3 RUNBOOK(audit 3 RUN3 完成後 5 分鐘收尾流程)

> 用法:audit 3 RUN3(PID 42743)log 出現 `✅ audit completed` 或 process exit 後,逐步執行下列。

---

## 1. 確認 audit 3 結束(30 sec)

```bash
ps -p 42743 -o pid,etime,stat 2>&1 | head -3
# 若 process 已 exit:✅
# 若仍 alive:STOP — 等它跑完
```

## 2. 抽取 final stats(30 sec)

```bash
cd /Users/hugo/project/stock_backend
./scripts/maintenance/extract_audit3_stats.sh \
  reports/rebuild_logs/item3_v6.1.0_recursive/audit3_source_availability_RUN3.log \
  | tee reports/rebuild_logs/item3_v6.1.0_recursive/audit3_RUN3_FINAL_STATS.txt
```

**手抄這幾個關鍵數字**(填到 Phase 11 報告):
- `stocks=X/2771 (XX%)` ← 應為 `2771/2771 (100.0%)`
- `source_empty_ok=` / `time_drift_ok=` / `mismatch=` / `api_errors=` / `retries=`
- `elapsed=Xs` → 換算 h/m
- Throttle total_sleep / events / max / mean

## 3. 填寫 + rename Phase 11 報告(2 min)

```bash
cp reports/full_market_sync_TEMPLATE_v6.1.0_recursive_DRAFT.md \
   reports/full_market_sync_$(date +%Y%m%d_%H%M)_v6.1.0_recursive.md
```

編輯新報告檔,逐個替換 `[TODO]`:
- 一、Headline:Audit 3 列填 elapsed / errors
- 二、Phase 5-7 Audit 3 block 全填
- 三、§6.8.8-E.1 retries 數
- 四、§14.7-AW elapsed / errors 對比
- 五、Open Issues(預期 0;有再記)
- 六、Git commit Audit 3 RUN3 summary 行
- 刪掉檔頭 `> DRAFT TEMPLATE` 警告 block

## 4. Git 封存點(1 min)

```bash
cd /Users/hugo/project/stock_backend
git status                                     # 確認沒洩漏 .env
git add reports/full_market_sync_*_v6.1.0_recursive.md \
        reports/rebuild_logs/item3_v6.1.0_recursive/ \
        scripts/maintenance/extract_audit3_stats.sh

# 不要 add 整個 -A,避免 .env 或 venv 混入
git diff --cached --stat                       # 檢視

git commit -m "v6.1.0 recursive from-zero validation completed

- All 5 contracts validated (§7.4-A / §0.0-I.10 / §3.2A.H / §3.2A.I / §6.8.8-E.1)
- Cascade mitigation: 4→0 cascades, 43% speedup (7h54m → 4h30m)
- Audit performance: 23min → 1m49s (12.5× via BERNOULLI sampling)
- Audit 3 RUN3: 2771/2771 stocks, [TODO summary]
- SHMM §A.7 inscribed to global CLAUDE.md (HB-bound user reporting)

Co-Authored-By: Claude <noreply@anthropic.com>"

git tag v6.1.0-item3-validated
git push origin main --tags
```

## 5. SHMM 回滾(Task #39,1 min)

```bash
# (a) 砍 5-min test HB Monitor
#   TaskStop bze75h80z  ← via Agent runtime,非 CLI
#
# (b) 砍當前 watchdog(360s threshold)
#   TaskStop bws1sx3c5  ← via Agent runtime
#
# (c) 重掛 watchdog,threshold 1860s(31 min,憲章 spec)
#   重新 spawn Monitor:
#     watch 1m: if (now - mtime(/tmp/claude_loop_last_fire.txt)) > 1860: alert
#
# 保留 4 個 HB(15/20/25/30 min)+ 30-min backup A + watchdog 1860s
```

## 6. Task 狀態 transition(30 sec)

```
#36 in_progress → completed   (audit 3 + Item 3 整體 done)
#38 pending     → completed   (Phase 11 報告寫完)
#39 pending     → completed   (SHMM 回滾完)
#40 in_progress → 依用戶 API key 狀態:
    - 若 .env 已備:跑 --hello 收尾 → completed
    - 若還沒:保持 in_progress 等用戶
```

## 7. 用戶通知(15 sec)

簡訊式:
> 🏁 stock_backend v6.1.0 收尾完成
> - Audit 3 RUN3: 2771/2771 stocks / [TODO errors] errors / elapsed [TODO h]
> - Phase 11 報告:`reports/full_market_sync_<timestamp>_v6.1.0_recursive.md`
> - Git tag: `v6.1.0-item3-validated` 已推送
> - SHMM 回滾至憲章 spec
> - tsai_ai_assistant Phase 0:[等 .env 或 done]

---

**預估總耗時:5 分鐘**(全部已預備,只需執行)

*Runbook drafted 2026-05-24 17:34 by Claude Code session(audit 3 32.5% 之時的預先 staging)*
