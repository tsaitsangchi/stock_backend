"""
finmind_generic_ingest.py v0.2 (Generic Schema-Inferring FinMind Ingester · 任意 dataset 自動建表)
================================================================================
**最後更新日期**: 2026-06-08
**主權狀態**: GENERIC AUTO-SCHEMA INGESTION(從 FinMind 回應自動推導欄位/型別 + 自動建表)+ 不需 DATASET_REGISTRY 預定義 + §一.10 source-traceable(全資料來自 FinMind API)+ 與核心 11 表嚴格路徑(sovereign_sync_engine + DATASET_REGISTRY)隔離並存
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:給它任何一個 FinMind 的 dataset 名稱,它就自動去 FinMind 抓回來、看資料長什麼樣自動決定欄位型別、自動在資料庫建一張同名的表、把資料寫進去——不需要人先在 registry 定義 schema。

**它怎麼做(步驟)**:
1. 呼叫 FinMind `/api/v4/data`(帶 dataset / 可選 data_id / 日期範圍)。
2. **看回應自動推導每欄型別**:純數字→`NUMERIC`(至少 20,6,值很大就自動加大);其他→`VARCHAR`(至少 255,很長就 TEXT);`date`→`DATE`;`stock_id`/識別碼/`Time`→強制 `VARCHAR`(避免被當數字掉前導零)。
3. **自動偵測主鍵**(從 stock_id/date/Time/name/type… 等候選挑出能唯一識別列的組合)。
4. **自動建表**(`CREATE TABLE IF NOT EXISTS`,表名=dataset 名)+ upsert(`ON CONFLICT (主鍵) DO UPDATE`)。
5. 回報抓了幾列、寫了幾列、推導出的 schema。

**輸入 / 輸出**:輸入=FinMind dataset 名 + 選用參數;輸出=DB 內一張同名表 + 該 dataset 的資料。

**它不做的事(治權邊界)**:不碰核心 11 表的嚴格驗證路徑(那是 `sovereign_sync_engine` + `DATASET_REGISTRY` 的治權,§14.7-CD/CE 逐股比對);**自動建的表不自動進特徵/選股**——若要當特徵仍須過 §14.7-DC source-pure gate;**不收 intraday / 日以下 dataset(tick/分K/5秒)**——本系統以「日」為最小單位,intraday raw 預設拒絕(`INTRADAY_DATASETS` 守門;用戶 2026-06-08 directive);intraday source 之日級衍生值(如 5 秒委託統計收盤累積 → 每日大盤買均/賣均)須另以日級 derive 儲存,不存 raw。

**為什麼需要它**:解除「sync 只能抓 registry 預定義 11 個 dataset」的自我設限(用戶 2026-06-08 directive「不要在此系統設限制」),讓任意 FinMind 接口(大盤法人、委買委賣統計、現金流量…)都能落地探索,支撐 §14.7-DG regime/擇時等新方向。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)
1. **[Generic Auto-Schema]** (v0.1): 型別由 FinMind 回應動態推導,**零硬編 dataset schema**;字串 ≥ VARCHAR(255)、數字 ≥ NUMERIC(20,6)(用戶 directive 下限;2026-06-08 字串下限 100→255,實際下限由 `core.generic_schema.MIN_VARCHAR` 持有),值超界自動加大。
2. **[Sovereignty Declaration]** (v0.1, 憲法 §3.1 序列 ingestion 模組): 本程式為 §3.1 **generic ingestion 旁路**;治權邊界:(a) §3.1 ingestion;(b) 五套禁令(§0.1-A/§0.2-A/§0.3-A/§0.0-E.4/§6.8)不涉;(c) T1-T3 不分層;(d) §8.5 anti-leakage 不處理(raw ingestion);(e) **不選股**;(f) **不算特徵**;(g) **不碰核心 11 表 / DATASET_REGISTRY 嚴格路徑**(隔離並存);(h) **不收 intraday / 日以下 dataset**(日為最小單位,`INTRADAY_DATASETS` 預設拒絕);(i) 唯一職責:任意「日級(含)以上」FinMind dataset → 自動建表 → upsert。
3. **[Source Authority / No Synthetic]** (v0.1, §14.7-CC/§一.10): 全部資料**僅**來自 FinMind API(`api.finmindtrade.com`);不產生 synthetic / impute;NULL 即 NULL。
4. **[Zero Hardcoded Verdict]** (v0.1, §5.6.3): 成功/失敗依實際 API status + 寫入結果動態判定。
5. **[Idempotency]** (v0.1): `CREATE TABLE IF NOT EXISTS` + `ON CONFLICT (key) DO UPDATE`;同範圍重跑安全。
6. **[Historical Reference Authority]** (v0.1): 本檔 schema 推導為記述性,非權威 SSOT;核心 11 表權威仍為 `DATASET_REGISTRY`。

## 📊 二、全量維運指令總矩陣 (The Ultimate Operational Matrix)
### A. 主要運行模式
| 子項 | CLI | 治權契約 |
| :--- | :--- | :--- |
| A.1 抓市場級 dataset(無 data_id) | `python scripts/ingestion/finmind_generic_ingest.py --dataset TaiwanStockTotalInstitutionalInvestors --start-date 2010-01-01 --commit` | §14.7-CC |
| A.2 抓個股級 dataset | `... --dataset TaiwanStockCashFlowsStatement --data-id 2330 --start-date 2015-01-01 --commit` | §14.7-CC |
| A.3 dry-run(只推導 schema 不寫) | `... --dataset X`(省略 --commit) | §5.6.3 |
| A.4 指定日期範圍(巨量 dataset 必用) | `--start-date / --end-date` | 維運 |
### 不提供之旗標 (Intentionally Omitted)
- 無「--all-datasets 全史全抓」:intraday 巨量不可無腦全抓(需分段),故不提供。

## 📜 三、全修訂歷程 (Full Revision History)
| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-08 | Claude | 首版:通用 auto-schema FinMind ingester(從回應推導型別 + 自動建表 + upsert);字串≥VARCHAR(100)/數字≥NUMERIC(20,6);與核心 11 表嚴格路徑隔離並存。對映用戶 directive「取消 DEFAULT_FINMIND_DATASETS 寫死清單,不要在此系統設限制」。 | SUPERSEDED |
| v0.2 | 2026-06-08 | Claude | **去重至 §0.0-I 共用 SSOT**:infer_schema/detect_keys/ensure_table/upsert 移至 `core/generic_schema.py v1.0`,本檔改 `from core.generic_schema import ...`(不再持本地拷貝);commit 路徑改呼叫 `provision_and_upsert`(含 [Key Stability] 重用既有 PK + DATE 自動偵測 + NUMERIC 精確 cast 強化)。對映用戶 directive「全部的表都應是通用 ingester 建的」之共用機制提升。 | SUPERSEDED |
| v0.3 | 2026-06-08 | Claude | docstring 同步 `core.generic_schema v1.2` 之字串型別下限 100→255(用戶 directive「所有欄位字串型態最少要大於 varchar(255) 以上」);本檔**無 code 變更**(下限由所 import 之 `core.generic_schema.MIN_VARCHAR` 持有,自動繼承),僅 [Generic Auto-Schema] + 白話段標示更新。 | **ACTIVE** |
"""

from __future__ import annotations
import argparse
import logging
import os
import sys

import psycopg2
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# §0.0-I 單一引用源:schema 推導/建表/upsert 共用 core.generic_schema(不在此檔複製)
from core.generic_schema import infer_schema, detect_keys, provision_and_upsert  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
# Intraday / 日以下 dataset —— 本系統以「日」為最小單位,intraday raw 不進入(用戶 2026-06-08 directive)。
# 此類 source 若有日級可用值(如 5 秒委託統計之收盤累積 → 每日大盤買均/賣均),須另以「日級 derive」儲存,
# 不存 intraday raw。預設拒絕;--allow-intraday 為明示例外。
INTRADAY_DATASETS = {
    "TaiwanStockPriceTick", "TaiwanStockKBar", "TaiwanStockEvery5SecondsIndex",
    "TaiwanStockStatisticsOfOrderBookAndTrade", "TaiwanFuturesTick", "TaiwanOptionTick",
    "TaiwanFutOptTickInfo", "taiwan_stock_tick_snapshot",
    "taiwan_futures_snapshot", "taiwan_options_snapshot",
}


def fetch(dataset, data_id, start, end, token, date=None, securities_trader_id=None):
    params = {"dataset": dataset}
    if data_id:
        params["data_id"] = data_id
    if securities_trader_id:
        params["securities_trader_id"] = securities_trader_id
    if date:
        params["date"] = date
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end
    r = requests.get(FINMIND_URL, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    j = r.json()
    if j.get("status") != 200:
        raise RuntimeError(f"FinMind API status={j.get('status')} msg={j.get('msg')}")
    return j.get("data", [])


def main():
    p = argparse.ArgumentParser(description="Generic schema-inferring FinMind ingester v0.1")
    p.add_argument("--dataset", required=True, help="FinMind dataset 名(= 目標表名)")
    p.add_argument("--data-id", default=None, help="個股/標的代碼(市場級 dataset 省略)")
    p.add_argument("--securities-trader-id", default=None, help="券商代碼(分點型 dataset)")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--date", default=None, help="單日 date 參數(分點/單日型 dataset)")
    p.add_argument("--commit", action="store_true", help="實際建表+寫入(省略=dry-run 只推導 schema)")
    p.add_argument("--allow-intraday", action="store_true",
                   help="明示允許 intraday/日以下 dataset(預設拒絕;本系統以日為最小單位)")
    args = p.parse_args()

    if args.dataset in INTRADAY_DATASETS and not args.allow_intraday:
        logger.error(f"❌ '{args.dataset}' 為 intraday/日以下 dataset;本系統以「日」為最小單位,"
                     f"intraday raw 不進入(用戶 directive)。若需其日級衍生值(如大盤買均/賣均),"
                     f"請改用日級 derive 儲存;確需存 raw 才加 --allow-intraday。")
        sys.exit(2)

    token = os.getenv("FINMIND_TOKEN")
    if not token:
        logger.error("FINMIND_TOKEN 未設定"); sys.exit(1)

    logger.info(f"抓取 dataset={args.dataset} data_id={args.data_id} {args.start_date}~{args.end_date}")
    rows = fetch(args.dataset, args.data_id, args.start_date, args.end_date, token,
                 date=args.date, securities_trader_id=args.securities_trader_id)
    logger.info(f"  API 回傳 {len(rows)} 列")
    if not rows:
        logger.warning("無資料,結束。"); return

    schema = infer_schema(rows)
    keys = detect_keys(rows, schema)
    logger.info(f"  推導 schema({len(schema)} 欄):")
    for c, t in schema.items():
        logger.info(f"    \"{c}\" {t}")
    logger.info(f"  偵測主鍵: {keys}")

    if not args.commit:
        logger.info("  [DRY-RUN] 未寫入(加 --commit 實際建表+upsert)。")
        return

    conn = psycopg2.connect(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
                            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASSWORD"))
    try:
        cur = conn.cursor()
        n, _schema, eff_keys = provision_and_upsert(cur, args.dataset, rows)
        conn.commit()
        logger.info(f"  ✅ 已 upsert {n} 列 → 表 \"{args.dataset}\"(主鍵={eff_keys})")
    except Exception as e:
        conn.rollback(); logger.error(f"  ❌ 失敗(已 rollback): {e}"); raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
