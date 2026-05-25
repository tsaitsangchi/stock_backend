# launchd 自動化 — stock_backend daily sync + weekly audit 3

對應 task #45 P0 自動化(2026-05-26 落地;憲章 §6.8.7 / §6.8.8 / §3.2A.I)

## 結構

```
deploy/launchd/
├── README.md                                              # 本檔
├── com.tsai.stock_backend.daily_sync.plist                # 每日 18:30 sync + audit 1+2
└── com.tsai.stock_backend.weekly_audit3.plist             # 週日 03:00 audit 3 全市場
scripts/maintenance/
├── daily_sync_and_audit.sh                                # daily 觸發腳本
└── weekly_audit3.sh                                       # weekly 觸發腳本
logs/
├── daily/                                                 # 每日 log 累積
└── weekly/                                                # 週末 log 累積
/tmp/
├── stock_backend_daily_last_run.txt                       # sentinel(SHMM-compatible)
└── stock_backend_weekly_audit3_last_run.txt
```

## 安裝步驟(一次性 setup)

### 1. 確認腳本可執行
```bash
chmod +x scripts/maintenance/daily_sync_and_audit.sh
chmod +x scripts/maintenance/weekly_audit3.sh
```

### 2. 複製 plist 至 LaunchAgents
```bash
cp deploy/launchd/com.tsai.stock_backend.daily_sync.plist ~/Library/LaunchAgents/
cp deploy/launchd/com.tsai.stock_backend.weekly_audit3.plist ~/Library/LaunchAgents/
```

### 3. 載入 launchd job
```bash
launchctl load ~/Library/LaunchAgents/com.tsai.stock_backend.daily_sync.plist
launchctl load ~/Library/LaunchAgents/com.tsai.stock_backend.weekly_audit3.plist
```

### 4. 驗證已載入
```bash
launchctl list | grep com.tsai.stock_backend
# 預期:
# -      0    com.tsai.stock_backend.daily_sync
# -      0    com.tsai.stock_backend.weekly_audit3
```

### 5. 手動測試一次(可選)
```bash
launchctl start com.tsai.stock_backend.daily_sync
# 等 5-10 min 看 logs/daily/ 是否有新檔
tail -30 logs/daily/daily_*.log | tail
```

## 卸載步驟(若要停用)

```bash
launchctl unload ~/Library/LaunchAgents/com.tsai.stock_backend.daily_sync.plist
launchctl unload ~/Library/LaunchAgents/com.tsai.stock_backend.weekly_audit3.plist
rm ~/Library/LaunchAgents/com.tsai.stock_backend.daily_sync.plist
rm ~/Library/LaunchAgents/com.tsai.stock_backend.weekly_audit3.plist
```

## 時程設定

| Job | 時點 | 內容 | 預估時長 |
|---|---|---|---|
| daily_sync | **每日 18:30**(收盤 + 30min buffer) | 1. core universe sync<br>2. FRED sync<br>3. audit 1 supply_chain<br>4. audit 2 schema(BERNOULLI sample) | ~10-15 min |
| weekly_audit3 | **週日 03:00**(凌晨無干擾) | audit 3 全市場 source_availability(sponsor 加速) | ~1.5-2h |

## 通知機制

腳本內建 macOS notification:
- ✅ 完成時跳通知「Done at HH:MM」
- ❌ 失敗時跳通知「FAIL at HH:MM」
- 任何 error 觸發 trap 自動 exit + notify

## SHMM 整合(cron 健康監控)

兩個 sentinel 檔:
```
/tmp/stock_backend_daily_last_run.txt          ← daily 完成時刻
/tmp/stock_backend_weekly_audit3_last_run.txt  ← weekly 完成時刻
```

可寫 watchdog Monitor 檢查 sentinel age:
- daily sentinel age > 26h → 警報(預期每日跑)
- weekly sentinel age > 8d → 警報(預期每週跑)

## 治權對齊

- ✅ 對齊憲章 §6.8.7 第 (4) 條(`--universe core` 為日常增量,`--universe full` 為週末全市場)
- ✅ 對齊憲章 §6.8.8 audit suite 完整性
- ✅ 對齊憲章 §3.2A.I parallel audit(workers=2)
- ✅ 對齊憲章 §3.2A.H BERNOULLI sample(audit 2 加速)
- ✅ 對齊憲章 §7.4-A 402 cascade mitigation(sponsor token + 5500/hr cap respected)
- ✅ 對齊 ~/.claude/CLAUDE.md §A SHMM(sentinel-compatible)

## 預期效益

- **消除每日手動 sync 時間**(原本 ~30 min/day → 0)
- **catch sync issues immediately**(每日 audit 1+2 確保 data integrity)
- **週末完整檢查**(audit 3 全市場 catch 任何 sourcing 問題)
- **不影響工作時間**(18:30 收盤後 + 凌晨)
