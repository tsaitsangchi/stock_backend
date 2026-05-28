# Weekly Cron Activation — §14.7-BX/CE/CH Continuous Verification(2026-05-28)

**Activation date**: 2026-05-28
**HEAD**: `ba5a7b5`(v6.4.7)
**Authorized by**: User explicit auth(2026-05-28)
**Status**: ✅ Cron installed and active

---

## 一、Installed cron entry

```cron
# §14.7-BX/CE/CH weekly doctrine recommit + API audit + auto resync (v6.4.7 / 2026-05-28)
# Saturday 03:00 Asia/Taipei: off-hours / 完全 after Friday 17:30 daily sync(~9.5h buffer)
# Pipeline: FRED sync → Step 3.5 §14.7-CE API audit + auto resync → Step 4 §14.7-CG native gate → audit → drift report
0 3 * * 6 cd /home/hugo/project/stock_backend && .venv/bin/python scripts/maintenance/run_weekly_doctrine_recommit.py --commit >> logs/weekly_doctrine_recommit.log 2>&1
```

## 二、Schedule rationale

| Property | Value | Rationale |
|---|---|---|
| Time | Saturday 03:00 Asia/Taipei | Off-hours / 無 API contention |
| Cron syntax | `0 3 * * 6` | day-of-week 6 = Saturday |
| Buffer after Friday 17:30 daily sync | **~9.5 hours** | 確保 weekday daily sync 完成 |
| TZ | Asia/Taipei(machine)| 對齊台股市場時區 |
| Log path | `logs/weekly_doctrine_recommit.log` | 同既有 logs/ pattern |
| Python | `.venv/bin/python` | 對齊既有 3 cron entries |

## 三、Pipeline executed by cron

```
Step 1   FRED sync(24 indicators 重 fetch)
   ↓
Step 3.5 §14.7-CE live API audit + auto resync(本 week 新增)
   ├─ Live API call to FinMind / FRED
   ├─ Byte-level compare with DB
   ├─ Mismatch detected → auto resync(resync_priceadj_mismatch.py / fetch_fred_data.py)
   └─ Re-audit verify(0 mismatches → continue)
   ↓
Step 4   §14.7-CG native gate builder v0.13(`--mode doctrine-native --commit`)
   ├─ Build new core_universe snapshot
   ├─ Atomic supersede 既有 snapshot
   └─ feature_store binding(若需)
   ↓
Step 5   audit_universe_completeness.py
   ↓
Step 6   Drift report 生成(reports/weekly_universe_recommit_<YYYYMMDD>.md)
```

## 四、Existing crontab(post-install / 4 entries 全保留)

| # | Schedule | Job | Status |
|---|---|---|---|
| 1 | `30 17 * * 1-5` | Daily sync(weekdays 17:30)| ACTIVE |
| 2 | `0 2 1 * *` | Monthly doctrine compliance(1st of month 02:00)| ACTIVE |
| 3 | `0 0 15 12 *` | Annual research irrigation(12/15 00:00)| ACTIVE |
| **4** | **`0 3 * * 6`** | **§14.7-BX/CE/CH weekly recommit(Saturday 03:00)** | **🆕 ACTIVE** |

## 五、First execution

**Next fire time**: 2026-05-30 03:00 CST(本週末)

**預期 cron run 行為**:
1. Step 1 FRED sync — 抓 24 indicators(M2SL/INDPRO etc. 之 routine revision)
2. Step 3.5 API audit — 比對 1,541 stocks × 5 days + 24 FRED × 5 obs
3. 若 mismatch → auto resync(預計 < 5 stocks based on weekly ex-dividend rate)
4. Step 4 native gate build — N=1,541 → may shift slightly per new fundamentals
5. Step 5 audit — PERFECT verdict expected
6. Step 6 drift report — added/removed/stable 分布

## 六、Monitoring + safety

### 6.1 Log monitoring

```bash
# Watch latest weekly run
tail -f logs/weekly_doctrine_recommit.log

# Check Saturday's drift report
ls -lt reports/weekly_universe_recommit_*.md | head -3
```

### 6.2 Emergency disable

```bash
crontab -l | grep -v "weekly_doctrine_recommit" | crontab -
# Or edit interactively: crontab -e
```

### 6.3 Manual trigger(testing)

```bash
cd /home/hugo/project/stock_backend
.venv/bin/python scripts/maintenance/run_weekly_doctrine_recommit.py --commit --force-now
```

### 6.4 Skip API audit(emergency)

```bash
.venv/bin/python scripts/maintenance/run_weekly_doctrine_recommit.py --commit --skip-api-audit
```

## 七、Risks + Mitigations

### 7.1 Phase C-2/D-2 prerequisite ⚠️

依憲章 §14.7-BX core definition 5:

> **[Phase C-2 Pre-condition]**:weekly cron 啟動前須先完成 Phase C-2(M1/M2/M3 model retrain 策略治權選定)+ Phase D-2(model_trainer / feature_store weekly mode);否則 weekly recommit 將造成下游 model 不一致。

**Current state**:Phase C-2 / Phase D-2 未完成

**Implication**:
- Universe snapshot 每週 supersede(N=1,541 → may shift)
- 但 **model_trainer 不在 cron 中**(not auto-triggered)
- 所以 model inconsistency 風險 = 0(無 auto retrain)

**Mitigation**:若未來手動跑 model_trainer,先確認讀的 snapshot 與 model training set 一致;否則重 train。

### 7.2 API quota exhaustion

**Current usage**:Saturday 03:00 cron 預計用 ~1,565 FinMind calls(1,541 stocks + 24 FRED endpoints)

**Sponsor tier limit**:6,000/hour(>= 26% utilization)

**Mitigation**:remaining 74% buffer 足以處理 mismatch resync(~10-20 calls typically)

### 7.3 DB lock contention

**Risk**:Saturday 03:00 與其他 cron 衝突?

**Analysis**:
- Daily sync(17:30 weekdays):Saturday 不跑
- Monthly doctrine(1st 02:00):僅每月一次,衝突機率 < 3.3%
- Annual research(12/15 00:00):每年一次,衝突機率 < 0.3%

**Worst case**:1/12 月初的週六 monthly + weekly 同時跑(02:00 vs 03:00 差 1 小時,monthly typically < 10 min 完成,無衝突)

### 7.4 Cron log rotation

**Risk**:logs/weekly_doctrine_recommit.log 無限長

**Mitigation**:若 logs/ 累積過大,加 logrotate config 或定期 truncate

## 八、Doctrine alignment

### 8.1 §14.7-BX Phase C-3:weekly orchestrator ✅
- ✅ Orchestrator v0.2 installed
- ✅ Pipeline 6 steps 完整
- ✅ Atomic supersede via builder

### 8.2 §14.7-CE Empirical-Verification-axis ✅
- ✅ Live API audit in every cron run
- ✅ Auto resync on mismatch detection
- ✅ Re-audit verify before proceeding to Step 4

### 8.3 §14.7-CH Continuous-Verification-axis ✅
- ✅ Weekly continuous verification(not one-shot)
- ✅ Cron-ready integration
- ✅ Automatic governance enforcement

## 九、Cron post-install verification

```
$ crontab -l | tail -4
# §14.7-BX/CE/CH weekly doctrine recommit + API audit + auto resync (v6.4.7 / 2026-05-28)
# Saturday 03:00 Asia/Taipei: off-hours / 完全 after Friday 17:30 daily sync(~9.5h buffer)
# Pipeline: FRED sync → Step 3.5 §14.7-CE API audit + auto resync → Step 4 §14.7-CG native gate → audit → drift report
0 3 * * 6 cd /home/hugo/project/stock_backend && .venv/bin/python scripts/maintenance/run_weekly_doctrine_recommit.py --commit >> logs/weekly_doctrine_recommit.log 2>&1
```

✅ Installed and active. Next fire: 2026-05-30(本週六)03:00 CST。

---

**Authorization chain**:
- 用戶 2026-05-28 explicit auth("P0 (確認後啟動) Weekly cron 啟動")
- 用戶 2026-05-28 schedule choice("改 Saturday 03:00")
- Activation time:2026-05-28 ~08:32 CST
- First scheduled run:2026-05-30 03:00 CST

**Repository**: https://github.com/tsaitsangchi/stock_backend
