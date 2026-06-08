"""
fetch_fred_data.py v3.3 (FRED 全球宏觀資料 Fetcher · §0.3 K-wave indicators · §14.7-DJ fred_series 改 generic provision_and_upsert · per CLAUDE.md §一.11 三段式入憲)
================================================================================
**最後更新日期**: 2026-06-08(§14.7-DJ:移除 hardcoded DDL_FRED/UPSERT_FRED,改 core.generic_schema.provision_and_upsert 建 fred_series;過濾 "."/null;per-series commit)
**主權狀態**: ACTIVE (§0.3 K-wave 13 indicators + §14.7-BY/BZ doctrine purity + §14.7-BG/BH 配套 + §一.11 三段式合規)
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:**FRED 全球宏觀資料抓取**(§0.3 康波週期指標):從 FRED API 抓 macro series 寫入 `fred_series`。

**輸入 → 輸出**:FRED API → fred_series(24 series 含 13 KWAVE)

**為什麼需要它**:核心股選拔的 K-wave gate 需要 fred_series;§14.7-DD PHASE 2b 前置。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[FRED Macro Data Authority]** (v3.2, §0.3 K-wave): 對齊憲章 §0.3 康波週期 13 indicators 之唯一 FRED API fetch 載體。
2. **[Per-Day Atomic Commit]** (v3.2): `commit_per_stock_per_day` — 每一天 × 每一指標獨立原子 commit;失敗 retry 不破壞已 commit。
3. **[Smart Retry/Gap-Fill]** (v3.2): `--retry-failed` + `--gap-fill` 依賴 fetch_log 補抓邏輯。
4. **[fetch_log Integration]** (v3.1+): 每次抓取(成功 / 失敗 / 跳過)寫入監控日誌。
5. **[Source Traceability]** (v3.2, §一.10): 全 (c) FRED API response + (b) DB write;0 AI memory。
6. **[Zero Hardcoded Verdict]** (v3.2, §5.6.3): success/failure 動態判定。
7. **[Sovereignty Declaration]** (v3.2, §3.1 ingestion 模組 / §0.3): 本程式為 **§0.3 康波週期 FRED 唯一 ingestion 載體**(§3.1 序列 ingestion;與 sovereign_sync_engine 之 FRED group 並列)。**治權邊界**:(a) §3.1 ingestion 載體;(b) **僅 fetch FRED**(不 fetch FinMind);(c) **不選股 / 不算 feature**;(d) 唯一職責:從 FRED API 抓 macro indicators + per-day atomic commit 至 fred_series table。
8. **[Historical Reference Authority]** (v3.2): `TOOL_VER = "v3.2"` 為記述快照;13 P0 indicators 為 §14.7-BY 治權範圍。
9. **[Idempotency]** (v3.2): `--force` 控制 re-fetch;預設 skip-if-exists;ON CONFLICT upsert 保護。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

### Group A. Default Fetch (預設 13 P0 indicators)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| A.1 11 P0 series | M2SL/CPIAUCSL/DGS10/DGS2/T10Y2Y/VIXCLS/INDPRO/IPG3344S/UMCSENT/WTISPLC/PALLFNFINDEXQ | §14.7-BY |
| A.2 2 Path E P1 series | BIS Credit / EIA Oil(per §14.7-BY Path E)| §14.7-BY Path E |
| A.3 Default 全 fetch | no flag | §0.3 K-wave |

### Group B. Selective Fetch (--ids)
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Specific series | `--ids T10Y2Y VIXCLS ...` | ad-hoc |
| B.2 Combined with --force | force re-fetch | maintenance |

### Group C. Resume / Gap-Fill
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| C.1 `--retry-failed` | fetch_log lookup | smart resume |
| C.2 `--gap-fill` | DB gap detection | data integrity |

### Group D. Error Handling + Retry
| 子項 | 對應方法 | 治權契約 |
| :--- | :--- | :--- |
| D.1 fred_get backoff | exponential retry | resilience |
| D.2 Per-day commit | atomic | data safety |
| D.3 fetch_log write | success/fail/skip | observability |

### 對齊憲章 §二 維運矩陣
| 場景 | 命令 |
| :--- | :--- |
| Default 全抓 | `python scripts/fetchers/fetch_fred_data.py` |
| 特定 indicator | `... --ids T10Y2Y VIXCLS` |
| 強制重抓 | `... --ids DGS10 --force` |
| 補抓失敗 | `... --retry-failed` |

### 不提供之旗標 (Intentionally Omitted)
- `--universe`:本程式僅 fetch FRED(macro),無 stock universe 概念。
- `--worker N`:FRED 單 endpoint serial fetch(rate limit 友善)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v3.2-§一.11 | 2026-05-29 | Codex | **§一.11 三段式標頭補正**。原 v3.2 邏輯不變。 | **ACTIVE** |
| v3.2 | (legacy)| — | **v3.2 改進**(配合 db_utils v3.0 / path_setup v2.0):導入 core.path_setup;完整實作 `--retry-failed` / `--gap-fill`;模組化抓取邏輯;強化 fred_get 退避重試。 | ARCHIVED(標頭格式)|
| v3.1 | (legacy)| — | **v3.1 既有**:整合 fetch_log(成功/失敗/跳過皆 log);效能追蹤;commit_per_stock_per_day 原子化。 | ARCHIVED |

# Legacy detail(原 docstring 內容延續):

v3.2 改進（配合 db_utils v3.0, path_setup v2.0）：
  ★ 導入 `core.path_setup` 統一處理路徑與確保目錄存在。
  ★ 完整實作 `--retry-failed` 與 `--gap-fill` 智慧補抓邏輯（依賴 fetch_log）。
  ★ 模組化抓取邏輯，與其他核心腳本架構對齊。
  ★ 強化 `fred_get` 錯誤處理與退避重試機制。

v3.1 既有：
  · 整合 fetch_log：每次抓取（無論成功、失敗或跳過）都會寫入監控日誌。
  · 效能追蹤：記錄各總經指標的 API 請求耗時。
  · 導入 commit_per_stock_per_day：每一天、每一指標獨立原子 commit。

執行範例（常規）：
    # 抓取預設的所有 FRED 總經指標
    python scripts/fetchers/fetch_fred_data.py
    
    # 針對特定指標抓取
    python scripts/fetchers/fetch_fred_data.py --ids T10Y2Y VIXCLS
    
    # 強制重抓特定指標
    python scripts/fetchers/fetch_fred_data.py --ids DGS10 --force
    
    # 強制重抓指定日期後的所有指標
    python scripts/fetchers/fetch_fred_data.py --start 2024-01-01 --force

執行範例（維運與模式切換）：
    # 重試最近 7 天失敗的指標
    python scripts/fetchers/fetch_fred_data.py --retry-failed 7

    # 補抓最近 30 天無成功紀錄的指標
    python scripts/fetchers/fetch_fred_data.py --gap-fill 30
"""

from __future__ import annotations

import sys
import logging
import os
import random
import time
from pathlib import Path
from datetime import date, datetime, timedelta
import argparse
import requests
from collections import defaultdict

# ── 1. 統一的環境與路徑設定 (path_setup v2.0) ──
_base_dir = Path(__file__).resolve().parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from core.path_setup import ensure_scripts_on_path
ensure_scripts_on_path(__file__)

# ── 2. 引入核心模組 ──
from core.db_utils import (
    get_db_conn,
    ensure_ddl,
    get_all_safe_starts,
    resolve_start_cached,
    FailureLogger,
    dedup_rows,
    DDL_FETCH_LOG
)
# §14.7-DJ (pure-generic):fred_series 改 generic auto-schema 自動建表(退役 hardcoded DDL_FRED/UPSERT_FRED)。
from core.generic_schema import provision_and_upsert

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_FRED_SERIES = [
    "T10Y2Y", "T10Y3M", "T10YIE", "VIXCLS", "BAMLH0A0HYM2",
    "DTWEXBGS", "M2SL", "DGS10", "DGS2", "DGS3MO",
    "UMCSENT", "INDPRO", "UNRATE", "CPIAUCSL",
    # §14.7-BY Phase C-1(2026-05-27):6 P0 K-wave-aligned indicators
    # Per Kondratiev/Schumpeter/Mensch/Perez 學派之 5 大驅動因素 SSOT
    "PATENTUSALLTOTAL",  # Tech: US Granted Patents Total(Schumpeter index proxy / annual / 85% K-wave correspondence)
    "B985RC1Q027SBEA",   # Tech: Private IP products investment(R&D + Software + Entertainment / quarterly / 80%)
    "TCMDO",             # Credit: US total credit market debt(quarterly / 75%)
    "LFWA64TTUSA647N",   # Demographics: US working-age population %(annual / 85%)
    "SPPOPDPNDOLUSA",    # Demographics: US old-age dependency ratio(annual / 80%)
    "PALLFNFINDEXQ",     # Commodity: CRB Global Price Index of All Commodities(quarterly / 75%)
    # §14.7-BY Phase E(2026-05-27):2 P1 K-wave-aligned indicators(補強 Credit + Energy class)
    "QUSPAM770A",        # Credit: BIS Total Credit to Private Non-Financial Sector(% of GDP / Q / 1947-2025 / 80%;真正 BIS Credit-to-GDP gap proxy)
    "WTISPLC",           # Energy: WTI Spot Crude Oil Price(M / 1946-2026 / 80 年深度 / 70%;補齊 §0.3 5 大驅動因素之 Energy class)
    # §14.7-CC(2026-05-27):Source Authority Doctrine — 2 §0.3.2 Multi-cycle indicators
    # 取代 system-computed kwave_supply_cycle_proxy(per「全資料來源須 FinMind / FRED API 直接抓取」原則)
    "IPG3344S",          # Semi Kitchin: US Industrial Production: Semiconductor (M / 1972~ / index 2017=100;取代 TW_SEMI_VWAP_YOY)
    "PCU4831114831115",  # Shipping Juglar: US Deep Sea Freight Transportation PPI (M / 1988~ / index Jun1988=100;取代 TW_SHIPPING_VWAP_YOY)
]

# §14.7-DJ (pure-generic):fred_series 由 generic auto-schema 自動建表(provision_and_upsert);
# 退役 hardcoded DDL_FRED/UPSERT_FRED。FRED 逐 series 樣本之 series_id 恆定 → KEY_CANDIDATES
# 之 series_id-before-date 順序正確推出複合 PK (series_id, date)。

_CLI_ARGS_STR = " ".join(sys.argv)

# ─────────────────────────────────────────────
# 日誌與 API 客戶端
# ─────────────────────────────────────────────
def _ensure_fetch_log_table(conn) -> None:
    try:
        ensure_ddl(conn, DDL_FETCH_LOG)
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.warning(f"[fetch_log] ensure DDL 失敗：{e}")

def _write_fetch_log(conn, **kwargs):
    """寫入 fetch_log，失敗不影響主流程。"""
    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO fetch_log (
                run_ts, table_name, stock_id, fetch_mode,
                fetch_date_from, fetch_date_to,
                rows_inserted, rows_updated, duration_ms,
                status, error_message, cli_args
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
            """
            cur.execute(sql, (
                kwargs.get("table_name"), kwargs.get("stock_id"), kwargs.get("fetch_mode", "per_stock"),
                kwargs.get("fetch_date_from"), kwargs.get("fetch_date_to"),
                kwargs.get("rows_inserted", 0), kwargs.get("duration_ms", 0),
                kwargs.get("status"), kwargs.get("error_message"), _CLI_ARGS_STR
            ))
        conn.commit()
    except Exception as e:
        try: conn.rollback()
        except: pass
        logger.debug(f"fetch_log 寫入失敗：{e}")

def fred_get(series_id: str, api_key: str, start: str, end: str, max_retries: int = 3) -> list:
    """從 FRED API 獲取資料，包含指數退避重試機制。"""
    params = {
        "series_id": series_id, 
        "api_key": api_key, 
        "file_type": "json", 
        "observation_start": start, 
        "observation_end": end
    }
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(FRED_API_URL, params=params, timeout=(10, 60))
            
            if resp.status_code == 200:
                return resp.json().get("observations", [])
            elif resp.status_code == 429:
                logger.warning(f"FRED API 配額受限 (429)，等待 60 秒後重試... (Attempt {attempt}/{max_retries})")
                time.sleep(60)
            else:
                resp.raise_for_status()
                
        except Exception as e:
            last_error = e
            jitter = random.uniform(0, 1)
            sleep_time = (2 ** attempt) + jitter
            logger.warning(f"FRED API 請求失敗 ({e})，{sleep_time:.1f} 秒後重試... (Attempt {attempt}/{max_retries})")
            time.sleep(sleep_time)
            
    raise RuntimeError(f"FRED API 請求失敗已達最大重試次數 ({max_retries}): {last_error}")

# ─────────────────────────────────────────────
# Fetcher Logic
# ─────────────────────────────────────────────
def fetch_fred_series(
    conn, series_ids: list[str], api_key: str, 
    start: str, end: str, delay: float, force: bool,
    fetch_mode_override: str | None = None
):
    logger.info("=== [fred_series] 開始 ===")
    # §14.7-DJ:generic auto-schema 於首次 provision_and_upsert 自動建表;不再 ensure_ddl(DDL_FRED)。
    latest = get_all_safe_starts(conn, "fred_series", id_col="series_id")
    flog = FailureLogger("fred_series", db_conn=conn, log_to_db=False)
    total_rows = 0
    fetch_mode = fetch_mode_override or "per_stock"

    for sid in series_ids:
        s = resolve_start_cached(sid, latest, start, "1990-01-01", force)
        if not s:
            _write_fetch_log(conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode, status="skipped", error_message="up_to_date")
            continue

        t0 = time.time()
        try:
            obs = fred_get(sid, api_key, s, end)
            dur = int((time.time() - t0) * 1000)

            if obs:
                # §14.7-DJ:轉 list[dict](series_id/date/value)。過濾 FRED 缺值標記 "." 與 null/空;
                # value 保留原始字串交 PG 精確 cast(generic [Source Authority],無 Python float 中介)。
                rows = []
                for o in obs:
                    v = o.get("value")
                    if v is None or (isinstance(v, str) and v.strip() in (".", "")):
                        continue
                    rows.append({"series_id": sid, "date": o.get("date"), "value": v})

                if rows:
                    rows = dedup_rows(rows, ("series_id", "date"))
                    cur = conn.cursor()
                    try:
                        n, _schema, _keys = provision_and_upsert(cur, "fred_series", rows)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                    finally:
                        cur.close()
                    total_rows += n
                    _write_fetch_log(
                        conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode,
                        fetch_date_from=s, fetch_date_to=end, rows_inserted=n,
                        duration_ms=dur, status="success" if n > 0 else "partial"
                    )
                else:
                    _write_fetch_log(
                        conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode,
                        fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                        duration_ms=dur, status="no_new_data"
                    )
            else:
                _write_fetch_log(
                    conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode,
                    fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                    duration_ms=dur, status="no_new_data"
                )
            time.sleep(delay)
            
        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            flog.add(sid, str(e), fetch_date_from=s, fetch_date_to=end)
            _write_fetch_log(
                conn, table_name="fred_series", stock_id=sid, fetch_mode=fetch_mode,
                fetch_date_from=s, fetch_date_to=end, rows_inserted=0, 
                duration_ms=dur, status="failed", error_message=str(e)
            )
            
    logger.info(f"  [fred_series] 總共寫入 {total_rows} 筆")
    if flog.has_failures():
        failed = flog.dump()
        logger.warning(f"  [fred_series] {len(failed)} 個 series 失敗：{[f.get('stock_id') for f in failed]}")


# ─────────────────────────────────────────────
# 依 fetch_log 反推目標：retry-failed / gap-fill
# ─────────────────────────────────────────────
def query_failed_targets(conn, days: int) -> list[str]:
    targets = []
    sql = """
    WITH recent AS (
        SELECT stock_id, status, run_ts,
               ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY run_ts DESC) AS rn
        FROM fetch_log
        WHERE table_name = 'fred_series' AND run_ts > NOW() - (%s || ' days')::interval
    )
    SELECT stock_id FROM recent WHERE rn = 1 AND status = 'failed';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (str(days),))
            targets = [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"[retry-failed] 查詢失敗：{e}")

    if targets:
        logger.info(f"  [retry-failed/fred_series] 發現 {len(targets)} 個目標 (例：{targets[:5]})")
    return targets

def query_gap_targets(conn, days: int, all_series_ids: list[str]) -> list[str]:
    targets = []
    sql = """
    SELECT DISTINCT stock_id FROM fetch_log 
    WHERE table_name = 'fred_series' AND status = 'success' 
      AND run_ts > NOW() - (%s || ' days')::interval AND stock_id = ANY(%s);
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (str(days), all_series_ids))
            have_success = {row[0] for row in cur.fetchall()}
        targets = [sid for sid in all_series_ids if sid not in have_success]
    except Exception as e:
        logger.error(f"[gap-fill/fred_series] 查詢失敗：{e}")

    if targets:
        logger.info(f"  [gap-fill/fred_series] 發現 {len(targets)} 個目標 (例：{targets[:5]})")
    return targets


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+", default=DEFAULT_FRED_SERIES, help="指定要抓取的 FRED 指標代號")
    p.add_argument("--start", default="1990-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", type=int, default=0, metavar="DAYS", help="重試 fetch_log 中近 N 天最後狀態為 failed 的目標")
    p.add_argument("--gap-fill", type=int, default=0, metavar="DAYS", help="補抓 fetch_log 中近 N 天無 success 紀錄的目標")
    args = p.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("❌ 嚴重錯誤：未設定環境變數 FRED_API_KEY")
        sys.exit(1)

    conn = get_db_conn()
    try:
        _ensure_fetch_log_table(conn)

        # 模式 A：retry-failed
        if args.retry_failed > 0:
            logger.info(f"═══ 模式：retry-failed（過去 {args.retry_failed} 天） ═══")
            targets = query_failed_targets(conn, args.retry_failed)
            if targets: 
                fetch_fred_series(conn, targets, api_key, args.start, args.end, args.delay, force=True, fetch_mode_override="retry")
            else: 
                logger.info("沒有找到需要重試的目標，結束。")
            return

        # 模式 B：gap-fill
        if args.gap_fill > 0:
            logger.info(f"═══ 模式：gap-fill（過去 {args.gap_fill} 天無 success） ═══")
            targets = query_gap_targets(conn, args.gap_fill, args.ids)
            if targets: 
                fetch_fred_series(conn, targets, api_key, args.start, args.end, args.delay, force=True, fetch_mode_override="gap_fill")
            else: 
                logger.info("沒有找到需要補抓的目標，結束。")
            return

        # 模式 C：常規抓取
        fetch_fred_series(conn, args.ids, api_key, args.start, args.end, args.delay, args.force)
            
    finally:
        conn.close()
        logger.info("全部完成")

if __name__ == "__main__":
    main()