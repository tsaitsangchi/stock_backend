# §14.7-CE P0 Re-sync Closure(2026-05-28)

**Closure date**: 2026-05-28
**HEAD before**: `38eb805`(99.42% / 95.83% match)
**HEAD after**: TBD(100% / 100% match)
**Status**: 🎯 §14.7-CE absolute byte-level closure sealed

---

## 一、P0 Action Summary

依用戶 directive 執行 P0:**Re-sync 9 mismatch stocks 消除 PriceAdj retroactive adjustment 差異**。

### 1.1 Mismatch stocks identified(9 stocks)

| stock_id | 公司 | industry | Future ex-dividend date | Cause |
|---|---|---|---|---|
| 1215 | 卜蜂 | 食品工業 | 2026-06-02 | dividend ex-date forecast |
| 1568 | 倉佑 | 汽車工業 | 2026-06-02 | dividend ex-date forecast |
| 2426 | 鼎元 | 電子工業 | 2026-05-31 | dividend ex-date forecast |
| 2486 | 一詮 | 電子工業 | 2026-05-31 | dividend ex-date forecast |
| 3257 | — | — | — | retroactive close adj |
| 3413 | — | — | — | retroactive close adj |
| 3680 | — | — | — | retroactive close adj |
| 4995 | — | — | — | retroactive close adj |
| 6679 | — | — | — | retroactive close adj |

### 1.2 Re-sync execution

```bash
python /tmp/find_and_resync_mismatch.py
```

- Date range: **2024-01-01 ~ 2026-05-28**(wider window covers all retroactive history)
- Method: FinMind API direct call + UPSERT `(stock_id, date)` PK
- Result:**9 stocks × 578 rows each = 5,202 rows upserted**
- Elapsed: ~3 seconds

### 1.3 FRED M2SL re-sync

```bash
python scripts/fetchers/fetch_fred_data.py --ids M2SL --force
```

- M2SL 之 routine monthly revision 已對齊 latest FRED release
- Result: 2026-02-01 / 2025-12-01 之 value 已 update

---

## 二、Post-resync Verification(absolute 100% byte-level)

### 2.1 Re-run `audit_live_api_vs_db.py`

| Layer | Before P0 | After P0 |
|---|---|---|
| FinMind 1,541 stocks 100% byte-match | 1,532 (99.42%) | **1,541 (100.00%)** ✅ |
| FinMind total entries match | 7,660 | **7,705** ✅ |
| FinMind total mismatches | **45** | **0** ✅ |
| FRED 24 series 100% byte-match | 23 (95.83%) | **24 (100.00%)** ✅ |
| FRED total entries match | 104 | **106** ✅ |
| FRED total mismatches | **2** | **0** ✅ |

### 2.2 §14.7-CE Deep Verdict 最終結果

```
✅ §14.7-CE DEEP attestation:**PASS**
✅ 全 DB data = API origin / 0 system-generated value
```

---

## 三、治權閉環

```
1,541 active core stocks
  ↓ Live HTTP GET api.finmindtrade.com × 1,541
1,541/1,541 stocks 100% byte-match(7,705 entries)
  ↓
24 FRED series
  ↓ Live HTTP GET api.stlouisfed.org × 24
24/24 series 100% byte-match(106 entries)
  ↓
🎯 absolute attestation:
  - 0 system-generated values
  - 0 retroactive adjustment gaps
  - 0 routine revision lags
  - DB ≡ FinMind/FRED API(byte-level)
```

---

## 四、§14.7-CE 治權閉環四階段史

| Step | Tag | Method | Result |
|---|---|---|---|
| 1 | v6.4.4 | Structural audit(row existence)| 1,541/1,541 stocks 100% coverage |
| 2 | v6.4.5 | Live API call + byte-level | 1,532/1,541(99.42%);9 stocks mismatch due to FinMind retroactive adj |
| 3 | (本 P0) | Re-sync mismatch stocks + FRED M2SL | **1,541/1,541(100.00%) / 24/24 series 100% / 0 mismatches** 🎯 |
| 4 | v6.4.6 | 封存點 sealing tag | Absolute attestation |

---

## 五、後續 maintenance recommendation

### 5.1 Weekly automation

`audit_live_api_vs_db.py` 加入 `run_weekly_doctrine_recommit.py` 之 Step X:
- 每週末跑 audit(7,819 entries / ~80s)
- 若 mismatch > 0 → 自動 re-sync 對應 stocks
- 維持 100% byte-level attestation

### 5.2 Ex-date pre-/post-sync trigger

針對未來 ex-dividend dates 之 stocks 主動 re-sync:
```sql
SELECT DISTINCT stock_id FROM "TaiwanStockDividend"
WHERE "CashExDividendTradingDate" BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days';
```

- 對這些 stocks 在 ex-date 前 1 天 + ex-date 後 1 天分別 re-sync
- 消除 retroactive adjustment lag

### 5.3 FRED revision auto-fetch

對 routine-revised series(M2SL / INDPRO / UNRATE / CPIAUCSL)加 daily incremental re-fetch:
```bash
python scripts/fetchers/fetch_fred_data.py --ids M2SL INDPRO UNRATE CPIAUCSL --start 2025-01-01 --force
```

---

**Closure attested**: 2026-05-28
**Audit tool**: `scripts/audit/audit_live_api_vs_db.py`
**Re-sync tool**: `/tmp/find_and_resync_mismatch.py`(可移至 `scripts/maintenance/resync_priceadj_mismatch.py` 為 permanent SSOT)
**Repository**: https://github.com/tsaitsangchi/stock_backend
