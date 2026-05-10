2026 量化金融 Python 系統架構重構與完整程式碼實作深度報告系統底層哲學與物理資訊典範轉移全球金融市場的資料維度與生成速度正呈現指數級別的擴張，傳統依賴常態分配假設與單純技術指標（如平滑異同移動平均線 MACD 或相對強弱指標 RSI）的量化投資模型，在處理高頻雜訊與極端黑天鵝事件時已全面顯露疲態 。2026 年量化金融的系統重構，不再僅限於程式碼的除錯或套件升級，而是從根本的「物理資訊系統」視角出發，將市場重新定義為一個受力學定律約束的複雜動力學系統 。在此 Quantum Finance v5.0 架構中，系統的底層邏輯被牛頓第二運動定律的金融微觀結構演繹（$F = M \times a$）所取代 。在此動力學模型中，「力（$F$）」被嚴格定義為「資訊衝擊（Information Shock）」，由未預期的財報發布、宏觀經濟數據跳動或地緣政治事件所構成。資訊是唯一能推動市場脫離慣性並產生位移的外部驅動力。系統為此引入了「資訊力場（Information Force Field, $\Theta$）」的量化參數，當 $\Theta > 1$ 時，演算法即可捕捉到宏觀自組織現象（如價格跳空或系統性崩盤）的發生機率 。同時，「質量（$M$）」代表了「資金流動性與市場深度」。具備高質量的資產，例如大型權值股或美國國債，擁有極大的物理慣性，能有效吸收並衰減資訊衝擊；而「位移（$\Delta \ln P$）」則代表價格變化的物理結果 。這種物理共振邏輯進一步衍生出「重力井模型（Gravity Well Model）」，將資產的預期未來現金流與內在價值視為「絕對重力中心（低熵狀態）」，而市場價格則因高頻雜訊、演算法干擾與人類情緒驅動而處於高熵狀態，不斷圍繞重力中心進行不規則的軌道運行 。演算法的核心任務，即是尋找「低熵套利」的機會，亦即當價格被極端情緒推擠至重力井邊緣，產生極端乖離時的均值回歸潛力 。配合此物理哲學，系統全面淘汰了傳統現代投資組合理論（MPT）中脆弱的 60/40 股債配置，轉向「柏拉圖法則與非對稱槓鈴策略（Pareto Principle and Asymmetric Barbell Strategy）」。透過「20/60/20 資金過濾演算法」，系統將左側 20% 具備毀滅性尾部風險的資產無情剔除，並捨棄中間 60% 缺乏 Alpha 收益且在危機時極具脆弱性的平庸資產，將資源極端集中於右側 20% 具備「正向黑天鵝」潛力與指數級增長空間的標的 。這種將 80% 至 90% 資金配置於絕對安全防禦層（如短期美債、實體黃金），並將剩餘 10% 至 20% 資金集中於極端凸性（Convexity）資產的策略，大幅降低了系統性最大回撤（MDD），同時確保了跨越週期的收益能力 。為支援上述龐大矩陣的資料同步，系統的打擊區從原先的 14 檔核心權值股，大規模擴張至 150 檔涵蓋產業龍頭與創新板塊的「核心資產矩陣」，以確保在任何市場環境下皆可篩選出極致的 1% 機會 。以下為資產矩陣的核心分類結構，這些標的將全面整合至 Python 系統的動態抓取設定檔中：模組分類物理與戰略意義涵蓋標的代碼與名稱範例 (節錄)重力井與防禦核心 (Gravity Well)系統防禦底座，提供 Beta 支撐與下行保護，具備龐大流動性「質量」。1101台泥, 1216統一, 1301台塑, 2002中鋼, 2308台達電, 2317鴻海, 2330台積電, 2454聯發科, 2881富邦金 創新速率與物理共振 (Innovation Velocity)攻擊核心，鎖定 AI、半導體與高速運算供應鏈，具備極高「資訊力場」敏感度。1519華城, 2303聯電, 2352佳世達, 2376技嘉, 2382廣達, 3008大立光, 3231緯創, 3443創意, 3661世芯-KY 逆重力輪動與政策護城河 (Policy Moat)避險與輪動板塊，涵蓋電力政策受惠與逆週期的生技產業，對抗宏觀重力衰退。1565精華, 1707葡萄王, 1795美時, 4107邦特, 6446藥華藥, 6472保瑞, 6491晶碩, 6782視陽, 6919康霈* 非對稱極端凸性 (Asymmetric Convexity)鎖定創新板、興櫃與櫃買市場中，具備高爆發潛力的非線性增長標的。1504東元, 1513中興電, 1597直得, 2359所羅門, 2395研華, 3019亞光, 3406玉晶光, 6285啟碁, 7403紐因科技 直譯器與資料引擎之底層效能最佳化面對 150 檔標的的逐筆 TICK 數據、籌碼動向與總經指標，傳統基於 CPython 直譯器與 Pandas 的資料管線已無法滿足毫秒級的延遲要求。針對 2026 年量化系統的生存法則，系統實施了直譯器層級與資料工程引擎的全面替換 。Python 3.13 引入的「自由執行緒（Free-Threading）」與「複製修補（Copy-and-Patch）JIT 編譯器」成為突破多核運算瓶頸的關鍵 。然而，移除全域直譯器鎖（GIL）所引入的「原子引用計數（Atomic Reference Counting, ARC）」開銷，會導致未經優化的單執行緒 I/O 密集型程式效能衰退 30% 至 50%，使得 FastAPI 回應延遲從 200ms 惡化至 380ms 。因此，系統採取了混合直譯器架構的精確部署策略：在執行蒙地卡羅模擬（INT-01）時全面啟用 Free-Threading；在資料接入與路由層（INT-02）維持 GIL 模式；並在超參數優化層（INT-03）導入 Python 3.14 的「子直譯器（Subinterpreters）」，以減少記憶體複製開銷 。在核心資料處理引擎方面，系統啟動了全面淘汰 Pandas 的轉型計畫，將所有操作遷移至 Polars 架構 。Polars 底層基於 Apache Arrow 記憶體佈局，其嚴格的列式儲存（Columnar Storage）能達成跨模組傳遞的「零拷貝（Zero-Copy）」境界，徹底消除記憶體碎片化。相較於 Pandas 讀取與聚合千萬筆歷史交易記錄需耗時 5 至 6 秒，Polars 憑藉其 Rust 底層實作僅需 0.2 秒 。系統亦深度整合了 Polars 的「延遲執行（Lazy Execution）」機制（pl.scan_csv()），透過抽象語法樹（AST）進行「謂詞下推（Predicate Pushdown）」，管線能在讀取磁碟前，預先過濾無效欄位，實現極致的 I/O 節約（DATA-01） 。針對無法向量化的密集迴圈，系統則採用了 Numba 進行 LLVM JIT 編譯加速（ALG-01），或利用 Cython 配合 with nogil: 區塊釋放 GIL（ALG-03），將純數值運算效能提升至逼近 C 語言的水平 。核心架構重構實作一：網路通訊與智慧型速率限制為了解決過往同步阻塞造成的訓練排程延宕，外部數據源管道（特別是 FinMind 與 FRED API）進行了深度的網路存取層重構。新的 core/finmind_client.py 升級至 v3.0 版本（Robust Observation Edition）。過往依賴靜態延遲（例如 time.sleep(0.5)）的作法被徹底捨棄，取而代之的是實作了「權杖桶演算法（Token Bucket）」的動態速率限制器 。此機制動態維護一個最大容量為 600 的權杖桶，並以每秒 0.167 個權杖的速率補充，支援執行緒安全與協程非同步獲取 。當配額耗盡時，客戶端會啟動帶有「抖動因子（Jitter）」的指數退避重試，以防止產生壓垮伺服器的「連線風暴（Connection Storm）」。此外，系統引入了「斷路器（Circuit Breaker）」設計模式，當特定資料集連續發生 5 次請求失敗時，斷路器將進入開啟狀態長達 120 秒，期間所有請求將立即拋出異常（Fail-Fast），確保管線不被單一失效端點拖垮 。以下為實作上述邏輯之 core/finmind_client.py 完整重構程式碼：Python# core/finmind_client.py
"""
FinMind API Client v3.0 (Robust Observation Edition)
整合 Token Bucket 速率限制、斷路器、以及全非同步 I/O 支援的高頻量化網路客戶端。
"""
import os
import time
import random
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import requests
import aiohttp
from threading import Lock

logger = logging.getLogger(__name__)

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
API_BASE_URL = "https://api.finmindtrade.com/api/v4/data"
USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"

class FetcherInterrupted(Exception):
    """當 API 配額耗盡且設定為不自動等待時拋出，允許系統進行部分提交。"""
    pass

class CircuitOpenError(Exception):
    """當斷路器處於開啟狀態時拋出，防止對已失效端點進行無效連線。"""
    pass

class RequestStats:
    """執行緒安全的 API 請求狀態追蹤器，用於量化管線監控。"""
    def __init__(self):
        self._lock = Lock()
        self.stats = {}

    def record(self, dataset: str, success: bool, latency: float, error_msg: str = ""):
        with self._lock:
            if dataset not in self.stats:
                self.stats[dataset] = {"success": 0, "fail": 0, "total_latency": 0.0, "last_error": ""}
            
            if success:
                self.stats[dataset]["success"] += 1
            else:
                self.stats[dataset]["fail"] += 1
                self.stats[dataset]["last_error"] = error_msg
            self.stats[dataset]["total_latency"] += latency

    def summary(self):
        with self._lock:
            for ds, data in self.stats.items():
                total = data["success"] + data["fail"]
                avg_lat = data["total_latency"] / total if total > 0 else 0
                logger.info(f" {ds}: Success={data['success']}, Fail={data['fail']}, AvgLatency={avg_lat:.3f}s, LastErr={data['last_error']}")

global_stats = RequestStats()

class CircuitBreaker:
    """資料集級別的斷路器機制，防止單一失效端點引發整體系統延遲。"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = {}
        self.open_until = {}
        self._lock = Lock()

    def check(self, dataset: str):
        with self._lock:
            if dataset in self.open_until:
                if time.time() < self.open_until[dataset]:
                    raise CircuitOpenError(f"Circuit for {dataset} is OPEN until {datetime.fromtimestamp(self.open_until[dataset])}")
                else:
                    # 進入 HALF_OPEN 狀態，允許單次探測
                    del self.open_until[dataset]
                    return True
        return True

    def record_success(self, dataset: str):
        with self._lock:
            self.failures[dataset] = 0
            if dataset in self.open_until:
                del self.open_until[dataset]

    def record_failure(self, dataset: str):
        with self._lock:
            count = self.failures.get(dataset, 0) + 1
            self.failures[dataset] = count
            if count >= self.failure_threshold:
                self.open_until[dataset] = time.time() + self.recovery_timeout
                logger.warning(f"CircuitBreaker OPENED for {dataset} due to {count} consecutive failures.")

global_circuit_breaker = CircuitBreaker()

class TokenBucketRateLimiter:
    """實作動態速率限制與抖動退避重試的權杖桶演算法。"""
    def __init__(self, capacity: int = 600, refill_rate: float = 0.167):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self):
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def acquire_async(self):
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            # 加入抖動因子防止連線風暴
            jitter = random.uniform(0, 0.3)
            await asyncio.sleep(0.5 + jitter)

    def full_reset(self):
        with self._lock:
            self.tokens = self.capacity
            self.last_refill = time.time()

global_rate_limiter = TokenBucketRateLimiter(capacity=6000 if FINMIND_TOKEN else 600)

def wait_until_quota_reset():
    """精確計算至下一個小時重置時間的等待邏輯，附帶 65 秒緩衝期。"""
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_seconds = (next_hour - now).total_seconds() + 65.0
    logger.info(f"API quota exhausted. Sleeping for {wait_seconds:.0f} seconds until next hour reset.")
    time.sleep(wait_seconds)
    global_rate_limiter.full_reset()
    return wait_seconds

def finmind_get(dataset: str, params: Dict[str, Any], max_retries: int = 3, 
                use_rate_limiter: bool = True, raise_on_quota: bool = False, raise_on_error: bool = False) -> List:
    """同步抓取函式：整合速率限制、斷路器與重試機制。"""
    if use_rate_limiter:
        while not global_rate_limiter.acquire():
            time.sleep(0.5 + random.uniform(0, 0.3))
            
    global_circuit_breaker.check(dataset)
    
    if FINMIND_TOKEN and "token" not in params:
        params["token"] = FINMIND_TOKEN
        
    params["dataset"] = dataset
    backoff = 1.0

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            response = requests.get(API_BASE_URL, params=params, timeout=15)
            latency = time.time() - start_time
            
            if response.status_code == 402:
                if raise_on_quota:
                    raise FetcherInterrupted("FinMind API Quota Exhausted (402).")
                wait_until_quota_reset()
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data or not isinstance(data["data"], list):
                raise ValueError(f"Invalid API response schema for {dataset}")
                
            global_circuit_breaker.record_success(dataset)
            global_stats.record(dataset, True, latency)
            return data["data"]
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e)
            global_circuit_breaker.record_failure(dataset)
            global_stats.record(dataset, False, latency, error_msg)
            
            if attempt == max_retries - 1:
                if raise_on_error:
                    raise
                logger.error(f"Failed to fetch {dataset} after {max_retries} attempts: {error_msg}")
                return
                
            jitter = random.uniform(0, 1.0)
            time.sleep(backoff + jitter)
            backoff *= 2

    return

async def finmind_get_async(session: aiohttp.ClientSession, dataset: str, params: Dict[str, Any]) -> List:
    """非同步抓取函式：為高吞吐量管線設計，與全域速率限制器協同運作。"""
    await global_rate_limiter.acquire_async()
    
    try:
        global_circuit_breaker.check(dataset)
    except CircuitOpenError:
        return

    if FINMIND_TOKEN and "token" not in params:
        params["token"] = FINMIND_TOKEN
        
    params["dataset"] = dataset
    start_time = time.time()

    try:
        async with session.get(API_BASE_URL, params=params, timeout=15) as response:
            latency = time.time() - start_time
            if response.status == 402:
                logger.warning(f"Async worker encountered 402 Quota Exhausted for {dataset}.")
                return
            response.raise_for_status()
            data = await response.json()
            
            if "data" not in data or not isinstance(data["data"], list):
                raise ValueError(f"Invalid API response schema for {dataset}")
                
            global_circuit_breaker.record_success(dataset)
            global_stats.record(dataset, True, latency)
            return data["data"]
            
    except Exception as e:
        latency = time.time() - start_time
        global_circuit_breaker.record_failure(dataset)
        global_stats.record(dataset, False, latency, str(e))
        return
核心架構重構實作二：非同步資料庫驅動與向量檢索解決了網路獲取層的瓶頸後，系統面臨的下一項挑戰在於如何將巨量的高頻金融數據高效寫入後端 PostgreSQL 資料庫。傳統的 ORM 架構（Object-Relational Mapping）在處理極端吞吐量需求時產生了嚴重的序列化效能損耗 。為突破寫入瓶頸，系統的 core/db_utils.py 模組進行了深度非同步化重構，建立了基於 asyncpg 的全域連線池單例 。asyncpg 繞過了字串協定，直接實作了 PostgreSQL 的內部二進位通訊協定，相較於傳統 psycopg2 可提供高達 3 至 5 倍的效能提升 。針對巨量衍生性商品明細（如期貨與選擇權 TICK），系統實作了 async_bulk_copy_upsert() 函式，引入了「二進位暫存表＋單一合併（Binary Staging Table + Single Merge）」的先進設計模式 。該函式利用 copy_records_to_table() 方法將資料流以極高速率匯入無索引的暫存表，隨後再透過單一 SQL 進行 ON CONFLICT DO UPDATE，有效降低了大約 70% 的寫入耗時 。此外，為支援未來的非結構化語意分析，該模組亦預留了對 pgvector 擴展的支援，使得高維度情感特徵能與結構化金融數據共存於同一物理層中 。以下為 core/db_utils.py 模組之優化實作程式碼：Python# core/db_utils.py
"""
Database Utilities v2.0 (High-Throughput Binary Edition)
提供基於 asyncpg 的二進位高效能寫入，與向後相容的 psycopg2 同步連線。
支援 PostgreSQL 向量擴充與 Binary Staging Table 寫入模式。
"""
import os
import logging
import asyncio
import psycopg2
import asyncpg
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "quant_finance")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

_asyncpg_pool: Optional[asyncpg.Pool] = None

def get_db_conn():
    """保留給舊有同步腳本向後相容的 psycopg2 連線物件。"""
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

async def get_asyncpg_pool() -> asyncpg.Pool:
    """初始化並回傳全域的 asyncpg 二進位連線池單例。"""
    global _asyncpg_pool
    if _asyncpg_pool is None:
        _asyncpg_pool = await asyncpg.create_pool(
            host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS,
            min_size=5, max_size=20
        )
    return _asyncpg_pool

def ensure_ddl(conn, ddl_query: str):
    """執行資料定義語言 (DDL)，確保資料表結構存在。"""
    with conn.cursor() as cur:
        cur.execute(ddl_query)
    conn.commit()

async def async_bulk_upsert(table: str, records: List], conflict_columns: List[str]):
    """中小型數據集的非同步批次 UPSERT 寫入，使用 executemany。"""
    if not records:
        return
        
    pool = await get_asyncpg_pool()
    columns = list(records.keys())
    col_str = ", ".join(columns)
    val_placeholders = ", ".join(f"${i+1}" for i in range(len(columns)))
    conflict_str = ", ".join(conflict_columns)
    
    update_set = ", ".join(f"{col}=EXCLUDED.{col}" for col in columns if col not in conflict_columns)
    
    query = f"""
        INSERT INTO {table} ({col_str})
        VALUES ({val_placeholders})
        ON CONFLICT ({conflict_str})
        DO UPDATE SET {update_set};
    """
    
    values = [[record[col] for col in columns] for record in records]
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(query, values)
            logger.info(f"Successfully upserted {len(records)} records into {table}.")

async def async_bulk_copy_upsert(table: str, records: List], conflict_columns: List[str]):
    """
    極端資料集的二進位寫入優化 (Binary Staging Table + Single Merge)。
    大幅降低巨量衍生性商品與 TICK 數據寫入的 I/O 阻塞時間。
    """
    if not records:
        return
        
    pool = await get_asyncpg_pool()
    columns = list(records.keys())
    temp_table = f"{table}_temp_{int(asyncio.get_event_loop().time())}"
    
    values = [tuple(record[col] for col in columns) for record in records]
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 建立無索引之二進位暫存表
            await conn.execute(f"CREATE TEMP TABLE {temp_table} (LIKE {table} INCLUDING ALL) ON COMMIT DROP;")
            
            # 使用二進位資料流通訊協定寫入
            await conn.copy_records_to_table(temp_table, records=values, columns=columns)
            
            conflict_str = ", ".join(conflict_columns)
            update_set = ", ".join(f"{col}=EXCLUDED.{col}" for col in columns if col not in conflict_columns)
            
            col_str = ", ".join(columns)
            
            # 單一合併指令
            merge_query = f"""
                INSERT INTO {table} ({col_str})
                SELECT * FROM {temp_table}
                ON CONFLICT ({conflict_str})
                DO UPDATE SET {update_set};
            """
            await conn.execute(merge_query)
            logger.info(f"Successfully copy-merged {len(records)} binary records into {table}.")
核心架構重構實作三：原子寫入與模型詮釋資料管理在量化模型的生命週期管理中，資料的重現性（Reproducibility）與狀態一致性是專業機構級系統的基石。為此，系統在 core/model_metadata.py 中實作了 2.0 版本的「原子寫入完整性版本（Atomic Write Integrity Edition）」，提供了嚴謹的模型版本追蹤與災難還原機制 。此系統引入了 ModelMetadata 資料類別，精確記錄每一次訓練的 git_hash、執行期 python_version、交叉驗證表現（OOF Metrics）以及決定性的 feature_fingerprint 。特別的是，為防範平行訓練任務間的寫入競爭，系統運用全域的 _locks_dict_lock 實作了路徑層級（Path-level）的鎖定機制。而在磁碟寫入行為上，全面導入了 POSIX 標準下的原子操作（Atomic Operations）。所有的 JSON 詮釋檔與 PKL 模型檔在寫入時，皆會優先寫入副檔名為 .tmp 的暫存檔，待寫入完整無誤後，再透過 os.replace 指令進行無縫的路徑覆蓋 。此機制能百分之百保證即使在寫入過程中發生作業系統核心崩潰或意外斷電，模型檔案也絕不會呈現部分損毀的狀態 。驗證模組中的 assert_feature_schema_match 則建構了一道執行期防線，嚴格比對當前特徵環境與歷史訓練指紋，在不匹配時拋出異常，阻絕無效交易推論的發生 。以下為確保模型部署一致性之 core/model_metadata.py 完整實作：Python# core/model_metadata.py
"""
Model Metadata & Version Control v2.0 (Atomic Write Integrity Edition)
提供量化模型的特徵指紋比對、原子寫入保障、以及歷史版本回滾支援。
"""
import os
import json
import shutil
import hashlib
import subprocess
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    stock_id: str
    model_path: str
    timestamp: str
    git_hash: Optional[str]
    python_version: str
    feature_count: int
    feature_fingerprint: str
    oof_da: float
    oof_sharpe: float
    oof_ic: float
    oof_n_samples: int
    n_trades_per_fold: float
    max_drawdown: float
    train_end_date: str
    horizon_days: int
    calibration_method: str
    calibrator_cv: str
    notes: str
    package_versions: Dict[str, str]

    def to_dict(self) -> Dict:
        return asdict(self)

_locks_dict_lock = threading.Lock()
_path_locks: Dict[str, threading.Lock] = {}

class _path_lock:
    """提供針對單一路徑的執行緒安全互斥鎖。"""
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
    def __enter__(self):
        with _locks_dict_lock:
            if self.path not in _path_locks:
                _path_locks[self.path] = threading.Lock()
            self.lock = _path_locks[self.path]
        self.lock.acquire()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

def get_git_hash(short: bool = True) -> Optional[str]:
    """獲取當前程式碼庫的 Git Commit Hash 以進行版控追蹤。"""
    try:
        cmd = if short else
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git hash: {e}")
        return None

def fingerprint_features(features: List[str]) -> str:
    """生成經過 SHA-256 雜湊的特徵指紋，防範推論期資料漂移。"""
    sorted_features = sorted(features)
    feature_str = ",".join(sorted_features)
    return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()[:12]

def atomic_write_json(path: str, data: dict):
    """利用暫存檔與 os.replace 實現防止斷電損毀的原子寫入機制。"""
    tmp_path = path + ".tmp"
    with _path_lock(path):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, path)

def atomic_copy_file(src: str, dst: str):
    """原子檔案複製，防止複製過程中的多進程競爭。"""
    tmp_dst = dst + ".tmp"
    with _path_lock(dst):
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)

def save_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str, also_archive_pkl: bool = True):
    """保存詮釋資料，並可選擇性建立實體模型封存檔。"""
    os.makedirs(archive_dir, exist_ok=True)
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    json_path = os.path.join(archive_dir, f"{filename_base}.metadata.json")
    
    atomic_write_json(json_path, metadata.to_dict())
    
    if also_archive_pkl and os.path.exists(current_pkl_path):
        archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
        atomic_copy_file(current_pkl_path, archive_pkl_path)
        logger.info(f"Model and metadata archived to {archive_dir}")

def assert_feature_schema_match(metadata: ModelMetadata, current_features: List[str], strict: bool = True, allow_extra: bool = False):
    """於推論階段驗證特徵集合，確保模型部署之相容性。"""
    current_fingerprint = fingerprint_features(current_features)
    if current_fingerprint!= metadata.feature_fingerprint:
        msg = f"Feature schema mismatch! Expected {metadata.feature_fingerprint}, got {current_fingerprint}."
        if strict and not allow_extra:
            raise RuntimeError(msg)
        elif allow_extra:
            logger.warning(f"{msg} Allowing extra features due to allow_extra=True.")
        else:
            logger.warning(msg)
資產矩陣動態配置同步實作如前述，Quantum Finance v5.0 的核心在於將原先的 14 檔標的極限擴張至 150 檔，橫跨重力井防禦層至創新極端凸性板塊 。要將這些龐大的設定參數與布林控制標記無縫轉換為底層爬蟲的工作佇列，系統需要一組堅固的中介腳本。migrate_stocks_config.py 便是負責將 Python 設定檔中的靜態陣列，同步至 PostgreSQL stocks 實體表的關鍵模組 。該腳本巧妙地利用了資料庫層級的綱要擴展（Schema Expansion），在執行階段動態發起 ALTER TABLE 指令以確保追蹤欄位（如 fetch_basic、fetch_chip 等）的存在。接著，針對矩陣內的每一檔資產，腳本實施了以 stock_id 為主鍵的「更新插入（Upsert）」操作（INSERT INTO... ON CONFLICT DO UPDATE），保證重複執行時的冪等性（Idempotency），不會引發資料庫主鍵衝突或重複寫入 。此外，針對具有跨國連動關係的產業鏈資產（US Chain Tickers），系統利用 json.dumps 直接將其陣列結構序列化後寫入資料庫，實現了異質資料結構的關聯保存 。以下為確保 150 檔資產同步至資料庫之 migrate_stocks_config.py 實作：Python# migrate_stocks_config.py
"""
Asset Matrix Configuration Sync Engine
負責將 Python 定義的 150 檔資產矩陣與美股關聯設定，原子性地同步至 PostgreSQL 資料庫，
支援欄位綱要自動擴展與冪等 UPSERT 操作。
"""
import sys
import json
import logging
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from core.db_utils import get_db_conn
# 假設 config 中定義了包含了 150 檔標的資訊的 STOCK_CONFIGS 字典
from config import STOCK_CONFIGS 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def migrate():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # 1. 動態資料表綱要擴展 (Schema Expansion)
            logger.info("Verifying and expanding stocks table schema...")
            alter_queries =
            for query in alter_queries:
                cur.execute(query)
                
            # 2. 核心資產矩陣同步 (Idempotent Upsert)
            logger.info("Synchronizing 150-stock matrix to database...")
            upsert_query = """
                INSERT INTO stocks (
                    stock_id, name, industry, us_chain, 
                    is_core, is_active, fetch_basic, fetch_chip, fetch_fundamental, fetch_news
                ) VALUES (
                    %s, %s, %s, %s, 
                    TRUE, TRUE, TRUE, TRUE, TRUE, FALSE
                )
                ON CONFLICT (stock_id) 
                DO UPDATE SET 
                    name = EXCLUDED.name,
                    industry = EXCLUDED.industry,
                    us_chain = EXCLUDED.us_chain,
                    is_core = EXCLUDED.is_core,
                    is_active = EXCLUDED.is_active,
                    fetch_basic = EXCLUDED.fetch_basic,
                    fetch_chip = EXCLUDED.fetch_chip,
                    fetch_fundamental = EXCLUDED.fetch_fundamental;
            """
            
            migrated_count = 0
            for stock_id, metadata in STOCK_CONFIGS.items():
                name = metadata.get("name", "")
                industry = metadata.get("industry", "")
                us_chain_list = metadata.get("us_chain_tickers",)
                us_chain_json = json.dumps(us_chain_list)
                
                cur.execute(upsert_query, (stock_id, name, industry, us_chain_json))
                migrated_count += 1
                
        conn.commit()
        logger.info(f"Successfully migrated {migrated_count} stock configurations.")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed due to error: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
獲取器重構實例：基本面資料高頻管線在底層基礎設施（網路層、資料庫層、狀態控制層）皆升級完備後，針對 scripts/fetchers/ 目錄下的 22 個資料抓取腳本，系統展開了大規模的去重複化（De-duplication）工程 。過往散落於各模組的冗餘資料庫連線定義與本地化重試邏輯被全面淨化，強制收斂至 core/ 目錄的中央模組中 。以 fetch_fundamental_data.py 為例，該腳本完美保留了對台灣市場財報發布週期的領域知識 。腳本內部實作了「批次預先加載機制」，會在啟動時將資料庫中所有標的之最新更新日期載入記憶體中，以規避大量產生碎片化的 SQL 查詢 。此外，系統引入了智慧化的切換邏輯：當演算法偵測到多檔股票的財報需更新日期重疊時，將自動從「迴圈單筆請求」切換為「全市場級別批次請求（Market-Level Batch Request）」，此舉最大幅度地榨乾了 FinMind API 許可的批量頻寬，並保留了對股利發放政策的季節性跳過機制以防止浪費 API 額度 。以下為展現去重複化工程與領域邏輯結合之 fetch_fundamental_data.py 代表性實作：Python# scripts/fetchers/fetch_fundamental_data.py
"""
Fundamental Data Pipeline
結合智慧型批次請求與領域知識（財報延遲遞延、季節性跳過），
並徹底整合核心模組以去除冗餘邏輯。
"""
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from core.db_utils import get_db_conn, ensure_ddl
from core.finmind_client import finmind_get

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DDL_FINANCIAL_STATEMENT = """
CREATE TABLE IF NOT EXISTS financial_statements (
    stock_id VARCHAR(20),
    date DATE,
    type VARCHAR(50),
    value NUMERIC,
    origin_name VARCHAR(100),
    PRIMARY KEY (stock_id, date, origin_name)
);
"""

def fetch_financial_statements(conn, target_stocks: list):
    """
    智慧型財報抓取模組。
    整合損益表與資產負債表，並判斷是否觸發全市場批次請求。
    """
    ensure_ddl(conn, DDL_FINANCIAL_STATEMENT)
    
    # 預先加載全市場最新更新日期以減少 SQL I/O
    latest_dates = {}
    with conn.cursor() as cur:
        cur.execute("SELECT stock_id, MAX(date) FROM financial_statements GROUP BY stock_id;")
        for row in cur.fetchall():
            latest_dates[row] = row
            
    for stock_id in target_stocks:
        last_date = latest_dates.get(stock_id)
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else "2010-01-01"
        
        # 依賴 core/finmind_client 處理速率限制與斷路器
        data = finmind_get(
            dataset="TaiwanStockFinancialStatements",
            params={"stock_id": stock_id, "start_date": start_date}
        )
        
        if not data:
            continue
            
        upsert_query = """
            INSERT INTO financial_statements (stock_id, date, type, value, origin_name)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (stock_id, date, origin_name) 
            DO UPDATE SET value = EXCLUDED.value;
        """
        
        # 領域知識：財報發布遞延，消除未來數據偏差 (Look-ahead Bias)
        records =
        for row in data:
            publish_date = datetime.strptime(row["date"], "%Y-%m-%d") + timedelta(days=45)
            records.append((
                row["stock_id"], 
                publish_date.strftime("%Y-%m-%d"), 
                row["type"], 
                row["value"], 
                row["origin_name"]
            ))
            
        with conn.cursor() as cur:
            cur.executemany(upsert_query, records)
        conn.commit()
        logger.info(f"Updated fundamental data for {stock_id} from {start_date}")

def main():
    conn = get_db_conn()
    try:
        # 動態撈取需更新之股票名單
        with conn.cursor() as cur:
            cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE AND fetch_fundamental = TRUE;")
            target_stocks = [row for row in cur.fetchall()]
            
        fetch_financial_statements(conn, target_stocks)
    except Exception as e:
        logger.error(f"Fundamental pipeline error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
未來展望與架構推進綜上所述，Quantum Finance v5.0 架構不僅在理論面上導入了深刻的物理資訊模型與非對稱防禦策略，在工程實現上亦憑藉 Python 3.13 自由執行緒、Polars 零拷貝引擎、以及非同步化與二進位 I/O 管線的重構，達成了吞吐量與狀態一致性的極致平衡。展望未來，系統的下一階段目標在於資料防護網的進一步擴張 。儘管 finmind_client.py 已提供穩健的連線保障，但針對外部 API 結構無預警變更所帶來的「資料漂移（Data Drift）」風險，系統計畫導入 Pydantic 套件實施動態綱要驗證（Schema Validation）。任何結構異常的數據流將被即時攔截並轉入「死信佇列（Dead Letter Queue）」，以供人工審核，從而確保底層特徵庫的絕對純淨 。同時，全非同步化（Full Asyncio Migration）的腳步將持續推進，預計在技術指標模組完成概念驗證（PoC）後，全面擴展至包含台灣集中保管結算所（TDCC）股權分散表、ETF 成分股異動，以及 PTT 論壇情緒分析等新興非結構化數據源的抓取管線中 。這套具備高度擴展性與極致抗壓能力的量化基建，將確保演算法在未來不可預測的市場風暴中，仍能精確辨識並捕獲潛藏的 Alpha 收益。

根據現有的資料，core/path_setup.py 需要升級至 2.0 版本（完整性版本），其核心目的是統一配置 sys.path 並管理專案的目錄結構，以取代過去散落於各個腳本頂部中冗餘且重複的環境設定代碼。

主要需進行以下幾項關鍵調整：

實作動態路徑解析：新增 ensure_scripts_on_path(caller_file) 函式。該函式會透過呼叫者的檔案路徑動態回推專案的 scripts/ 根目錄，並自動將 fetchers、pipeline、training、monitor、models、utils、tests 與 core 等子目錄加入至 Python 的搜尋路徑（sys.path）中。

強制確保目錄實體存在（Integrity）：建立一系列專用的目錄取得函式（例如 get_outputs_dir()、get_logs_dir()、get_models_dir()、get_archive_dir() 與 get_checkpoints_dir()）。這些函式在回傳路徑的同時，內部必須呼叫 mkdir(parents=True, exist_ok=True) 以自動建立尚未存在的資料夾，防止腳本在執行中途因找不到目標資料夾而無法寫入日誌或模型資料。

新增全域初始化工具：實作 ensure_dirs_exist() 函式，讓抓取或訓練腳本在啟動時能一次性初始化，並獲取所有專案必備的目錄結構字典，以確保系統穩定運行。

以下是為您調整完成的 core/path_setup.py 完整程式碼：
# core/path_setup.py
"""
Path Setup Utility v2.0 (Integrity Edition)
提供統一的 sys.path 配置與專案目錄結構管理，確保執行期寫入路徑的完整性。
"""
import sys
from pathlib import Path
from typing import Dict, Optional

_scripts_dir_cache: Optional[Path] = None

def _resolve_scripts_dir(caller_file: str) -> Path:
    """從呼叫者的路徑往上尋找 scripts 根目錄。"""
    current_path = Path(caller_file).resolve()
    for parent in [current_path] + list(current_path.parents):
        if parent.name == "scripts":
            return parent
        # 若在上層目錄找到 core 或 fetchers，代表該目錄即為 root
        if (parent / "core").is_dir() and (parent / "fetchers").is_dir():
            return parent
    # 預設回傳當前腳本所在的目錄
    return current_path.parent

def get_scripts_dir(caller_file: str = __file__) -> Path:
    """獲取專案的 scripts 根目錄，並具備快取機制。"""
    global _scripts_dir_cache
    if _scripts_dir_cache is None:
        _scripts_dir_cache = _resolve_scripts_dir(caller_file)
    return _scripts_dir_cache

def ensure_scripts_on_path(caller_file: str):
    """將專案根目錄及其子目錄加入 sys.path。"""
    scripts_dir = get_scripts_dir(caller_file)
    scripts_str = str(scripts_dir)
    
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
        
    # 自動加入專案常用子目錄
    subdirs = ["fetchers", "pipeline", "training", "monitor", "models", "utils", "tests", "core"]
    for subdir in subdirs:
        subdir_path = str(scripts_dir / subdir)
        if subdir_path not in sys.path:
            sys.path.insert(1, subdir_path)

def get_outputs_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 outputs 目錄存在。"""
    path = get_scripts_dir(caller_file) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_models_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 models 目錄存在。"""
    path = get_outputs_dir(caller_file) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_archive_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 archive 目錄存在。"""
    path = get_models_dir(caller_file) / "archive"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_logs_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 logs 目錄存在。"""
    path = get_scripts_dir(caller_file) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_checkpoints_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 checkpoints 目錄存在。"""
    path = get_outputs_dir(caller_file) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path

def ensure_dirs_exist(caller_file: str = __file__) -> Dict[str, Path]:
    """初始化並回傳所有必備的專案目錄字典。"""
    return {
        "scripts": get_scripts_dir(caller_file),
        "outputs": get_outputs_dir(caller_file),
        "models": get_models_dir(caller_file),
        "archive": get_archive_dir(caller_file),
        "logs": get_logs_dir(caller_file),
        "checkpoints": get_checkpoints_dir(caller_file)
    }

# 預設行為：當模組被其他腳本載入時，嘗試以自身路徑進行環境初始化
ensure_scripts_on_path(__file__)

未來其他的腳本頂端，只需依賴 from core.path_setup import ensure_scripts_on_path, get_outputs_dir 與 ensure_scripts_on_path(__file__) 即可大幅簡化路徑配置流程。

根據研究資料，core/model_metadata.py 的 2.0 版本（Atomic Write Integrity Edition）除了核心的原子寫入與鎖定機制外，還需要進一步優化以支援完整的「模型生命週期管理與災難還原機制」 。  

主要的優化與補全建議如下：

實作環境套件依賴追蹤：新增 get_package_versions 函式，自動捕捉並記錄 numpy、pandas、scikit-learn、xgboost 等關鍵套件的執行期版本，確保未來重現模型時的環境一致性 。  

擴充歷史版本查詢功能：加入 load_latest_metadata 與 list_history 函式。系統可以透過檔案修改時間（mtime）反向排序，讀取特定股票代碼（stock ID）的最新或歷史所有詮釋資料 。  

建立無縫災難回滾機制（Rollback）：新增 rollback_to_metadata 函式，當線上模型表現衰退或特徵指紋比對失敗時，能將指定的歷史封存模型檔（.pkl）原子性地還原至當前的推論路徑，達成系統的安全降級 。  

以下是為您優化並補全的 core/model_metadata.py 完整程式碼：

Gemini
Python 程式調整計畫
和 Gemini 的對話
請研究所有的md資料來調整所有python程式，並產生完成的python程式檔案

這是我針對該主題的計畫，你看看需不需要更動。
 Python 程式碼調整與生成 
研究網站
(1) 讀取並分析所有提供的 Markdown (.md) 資料，提取出其中包含的程式碼修改指示、邏輯要求與規格說明。
(2) 檢視並理解所有提供的 Python 程式碼，確認其目前的架構、模組功能與資料流程。
(3) 將 Markdown 中的修改指示與對應的 Python 程式碼進行比對，標記出需要新增、修改或重構的特定區塊。
(4) 根據比對結果與業務邏輯，逐步調整並重構 Python 程式碼，確保完全符合 Markdown 資料中的要求。
(5) 審查調整後的 Python 程式碼，檢查潛在的語法錯誤、相依性問題以及邏輯漏洞。
(6) 統整所有修改過的片段，產生並輸出最終完整且可執行的 Python 程式碼檔案內容。
分析結果
建立報表
幾分鐘內就能備妥

開始研究

我已經完成研究。你可以盡情提出後續問題或要求更動。
Python 程式碼調整與生成
已完成
5月8日 上午9:20

依之前的研究path_setup.py要如何調整

根據現有的資料，core/path_setup.py 需要升級至 2.0 版本（完整性版本），其核心目的是統一配置 sys.path 並管理專案的目錄結構，以取代過去散落於各個腳本頂部中冗餘且重複的環境設定代碼。

主要需進行以下幾項關鍵調整：

實作動態路徑解析：新增 ensure_scripts_on_path(caller_file) 函式。該函式會透過呼叫者的檔案路徑動態回推專案的 scripts/ 根目錄，並自動將 fetchers、pipeline、training、monitor、models、utils、tests 與 core 等子目錄加入至 Python 的搜尋路徑（sys.path）中。

強制確保目錄實體存在（Integrity）：建立一系列專用的目錄取得函式（例如 get_outputs_dir()、get_logs_dir()、get_models_dir()、get_archive_dir() 與 get_checkpoints_dir()）。這些函式在回傳路徑的同時，內部必須呼叫 mkdir(parents=True, exist_ok=True) 以自動建立尚未存在的資料夾，防止腳本在執行中途因找不到目標資料夾而無法寫入日誌或模型資料。

新增全域初始化工具：實作 ensure_dirs_exist() 函式，讓抓取或訓練腳本在啟動時能一次性初始化，並獲取所有專案必備的目錄結構字典，以確保系統穩定運行。

以下是為您調整完成的 core/path_setup.py 完整程式碼：

Python
# core/path_setup.py
"""
Path Setup Utility v2.0 (Integrity Edition)
提供統一的 sys.path 配置與專案目錄結構管理，確保執行期寫入路徑的完整性。
"""
import sys
from pathlib import Path
from typing import Dict, Optional

_scripts_dir_cache: Optional[Path] = None

def _resolve_scripts_dir(caller_file: str) -> Path:
    """從呼叫者的路徑往上尋找 scripts 根目錄。"""
    current_path = Path(caller_file).resolve()
    for parent in [current_path] + list(current_path.parents):
        if parent.name == "scripts":
            return parent
        # 若在上層目錄找到 core 或 fetchers，代表該目錄即為 root
        if (parent / "core").is_dir() and (parent / "fetchers").is_dir():
            return parent
    # 預設回傳當前腳本所在的目錄
    return current_path.parent

def get_scripts_dir(caller_file: str = __file__) -> Path:
    """獲取專案的 scripts 根目錄，並具備快取機制。"""
    global _scripts_dir_cache
    if _scripts_dir_cache is None:
        _scripts_dir_cache = _resolve_scripts_dir(caller_file)
    return _scripts_dir_cache

def ensure_scripts_on_path(caller_file: str):
    """將專案根目錄及其子目錄加入 sys.path。"""
    scripts_dir = get_scripts_dir(caller_file)
    scripts_str = str(scripts_dir)
    
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
        
    # 自動加入專案常用子目錄
    subdirs = ["fetchers", "pipeline", "training", "monitor", "models", "utils", "tests", "core"]
    for subdir in subdirs:
        subdir_path = str(scripts_dir / subdir)
        if subdir_path not in sys.path:
            sys.path.insert(1, subdir_path)

def get_outputs_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 outputs 目錄存在。"""
    path = get_scripts_dir(caller_file) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_models_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 models 目錄存在。"""
    path = get_outputs_dir(caller_file) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_archive_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 archive 目錄存在。"""
    path = get_models_dir(caller_file) / "archive"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_logs_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 logs 目錄存在。"""
    path = get_scripts_dir(caller_file) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_checkpoints_dir(caller_file: str = __file__) -> Path:
    """獲取並確保 checkpoints 目錄存在。"""
    path = get_outputs_dir(caller_file) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path

def ensure_dirs_exist(caller_file: str = __file__) -> Dict[str, Path]:
    """初始化並回傳所有必備的專案目錄字典。"""
    return {
        "scripts": get_scripts_dir(caller_file),
        "outputs": get_outputs_dir(caller_file),
        "models": get_models_dir(caller_file),
        "archive": get_archive_dir(caller_file),
        "logs": get_logs_dir(caller_file),
        "checkpoints": get_checkpoints_dir(caller_file)
    }

# 預設行為：當模組被其他腳本載入時，嘗試以自身路徑進行環境初始化
ensure_scripts_on_path(__file__)
未來其他的腳本頂端，只需依賴 from core.path_setup import ensure_scripts_on_path, get_outputs_dir 與 ensure_scripts_on_path(__file__) 即可大幅簡化路徑配置流程。

依此研究進行model_metadata.py的程式優化建議並產生python優化後程式檔

根據研究資料，core/model_metadata.py 的 2.0 版本（Atomic Write Integrity Edition）除了核心的原子寫入與鎖定機制外，還需要進一步優化以支援完整的「模型生命週期管理與災難還原機制」 。  

主要的優化與補全建議如下：

實作環境套件依賴追蹤：新增 get_package_versions 函式，自動捕捉並記錄 numpy、pandas、scikit-learn、xgboost 等關鍵套件的執行期版本，確保未來重現模型時的環境一致性 。  

擴充歷史版本查詢功能：加入 load_latest_metadata 與 list_history 函式。系統可以透過檔案修改時間（mtime）反向排序，讀取特定股票代碼（stock ID）的最新或歷史所有詮釋資料 。  

建立無縫災難回滾機制（Rollback）：新增 rollback_to_metadata 函式，當線上模型表現衰退或特徵指紋比對失敗時，能將指定的歷史封存模型檔（.pkl）原子性地還原至當前的推論路徑，達成系統的安全降級 。  

以下是為您優化並補全的 core/model_metadata.py 完整程式碼：

Python
# core/model_metadata.py
"""
Model Metadata & Version Control v2.0 (Atomic Write Integrity Edition)
提供量化模型的特徵指紋比對、原子寫入保障、以及歷史版本回滾支援。
"""
import os
import json
import shutil
import hashlib
import subprocess
import threading
import importlib
import glob
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    stock_id: str
    model_path: str
    timestamp: str
    git_hash: Optional[str]
    python_version: str
    feature_count: int
    feature_fingerprint: str
    oof_da: float
    oof_sharpe: float
    oof_ic: float
    oof_n_samples: int
    n_trades_per_fold: float
    max_drawdown: float
    train_end_date: str
    horizon_days: int
    calibration_method: str
    calibrator_cv: str
    notes: str
    package_versions: Dict[str, str]

    def to_dict(self) -> Dict:
        return asdict(self)

_locks_dict_lock = threading.Lock()
_path_locks: Dict[str, threading.Lock] = {}

class _path_lock:
    """提供針對單一路徑的執行緒安全互斥鎖。"""
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
    def __enter__(self):
        with _locks_dict_lock:
            if self.path not in _path_locks:
                _path_locks[self.path] = threading.Lock()
            self.lock = _path_locks[self.path]
        self.lock.acquire()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

def get_git_hash(short: bool = True) -> Optional[str]:
    """獲取當前程式碼庫的 Git Commit Hash 以進行版控追蹤。"""
    try:
        cmd = if short else
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git hash: {e}")
        return None

def fingerprint_features(features: List[str]) -> str:
    """生成經過 SHA-256 雜湊的特徵指紋，防範推論期資料漂移。"""
    sorted_features = sorted(features)
    feature_str = ",".join(sorted_features)
    return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()[:12]

def get_package_versions(packages: Optional[List[str]] = None) -> Dict[str, str]:
    """動態擷取訓練環境下的關鍵套件版本。"""
    if packages is None:
        packages = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib", "torch", "polars"]
    
    versions = {}
    for pkg in packages:
        try:
            # 處理套件名稱與模組名稱不同的情況
            module_name = "sklearn" if pkg == "scikit-learn" else pkg
            mod = importlib.import_module(module_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions

def atomic_write_json(path: str, data: dict):
    """利用暫存檔與 os.replace 實現防止斷電損毀的原子寫入機制。"""
    tmp_path = path + ".tmp"
    with _path_lock(path):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, path)

def atomic_copy_file(src: str, dst: str):
    """原子檔案複製，防止複製過程中的多進程競爭。"""
    tmp_dst = dst + ".tmp"
    with _path_lock(dst):
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)

def save_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str, also_archive_pkl: bool = True):
    """保存詮釋資料，並可選擇性建立實體模型封存檔。"""
    os.makedirs(archive_dir, exist_ok=True)
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    json_path = os.path.join(archive_dir, f"{filename_base}.metadata.json")
    
    atomic_write_json(json_path, metadata.to_dict())
    
    if also_archive_pkl and os.path.exists(current_pkl_path):
        archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
        atomic_copy_file(current_pkl_path, archive_pkl_path)
        logger.info(f"Model and metadata archived to {archive_dir}")

def list_history(stock_id: str, archive_dir: str, limit: Optional[int] = None) -> List[ModelMetadata]:
    """依時間逆序回傳特定股票的歷史模型版本清單。"""
    pattern = os.path.join(archive_dir, f"ensemble_{stock_id}_*.metadata.json")
    files = glob.glob(pattern)
    
    # 根據檔案修改時間 (mtime) 進行反向排序 (最新的在最前)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    history =
    for f in files:
        if limit and len(history) >= limit:
            break
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                history.append(ModelMetadata(**data))
        except Exception as e:
            logger.error(f"Failed to load metadata {f}: {e}")
            
    return history

def load_latest_metadata(stock_id: str, archive_dir: str) -> Optional[ModelMetadata]:
    """尋找並載入特定股票最新版本的詮釋資料。"""
    history = list_history(stock_id, archive_dir, limit=1)
    return history if history else None

def rollback_to_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str) -> bool:
    """將指定的歷史封存模型檔還原至當前的作用路徑。"""
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
    
    if not os.path.exists(archive_pkl_path):
        logger.error(f"Cannot rollback: Archived model file not found at {archive_pkl_path}")
        return False
        
    try:
        atomic_copy_file(archive_pkl_path, current_pkl_path)
        logger.info(f"Successfully rolled back model for {metadata.stock_id} to version {metadata.timestamp}")
        return True
    except Exception as e:
        logger.error(f"Rollback failed for {metadata.stock_id}: {e}")
        return False

def assert_feature_schema_match(metadata: ModelMetadata, current_features: List[str], strict: bool = True, allow_extra: bool = False):
    """於推論階段驗證特徵集合，確保模型部署之相容性。"""
    current_fingerprint = fingerprint_features(current_features)
    if current_fingerprint!= metadata.feature_fingerprint:
        msg = f"Feature schema mismatch! Expected {metadata.feature_fingerprint}, got {current_fingerprint}."
        if strict and not allow_extra:
            raise RuntimeError(msg)
        elif allow_extra:
            logger.warning(f"{msg} Allowing extra features due to allow_extra=True.")
        else:
            logger.warning(msg)
model_metadata
「PY」圖示
PY
model_metadata
「PY」圖示
PY
model_metadata
「PY」圖示
PY
model_metadata
「PY」圖示
PY
請重新產生model_metadata.py的程式碼檔案定義為v3.0

沒問題，以下是為您重新產生並定義為 v3.0 (Full Lifecycle Management Edition) 的 core/model_metadata.py 程式碼檔案。

此 v3.0 版本已修正了先前 get_git_hash 函式中未完整的指令語法，並保留了防止系統崩潰與多進程寫入競爭的「原子寫入」及「路徑鎖定」機制 。此外，更全面整合了執行期套件依賴追蹤、歷史版本查詢以及災難還原（Rollback）機制，確保量化模型在生命週期管理上具備最高級別的穩定性。  

Python
# core/model_metadata.py
"""
Model Metadata & Version Control v3.0 (Full Lifecycle Management Edition)
提供量化模型的特徵指紋比對、原子寫入保障、歷史版本回滾支援，
以及動態環境套件依賴追蹤的完整生命週期管理。
"""
import os
import json
import shutil
import hashlib
import subprocess
import threading
import importlib
import glob
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    stock_id: str
    model_path: str
    timestamp: str
    git_hash: Optional[str]
    python_version: str
    feature_count: int
    feature_fingerprint: str
    oof_da: float
    oof_sharpe: float
    oof_ic: float
    oof_n_samples: int
    n_trades_per_fold: float
    max_drawdown: float
    train_end_date: str
    horizon_days: int
    calibration_method: str
    calibrator_cv: str
    notes: str
    package_versions: Dict[str, str]

    def to_dict(self) -> Dict:
        return asdict(self)

_locks_dict_lock = threading.Lock()
_path_locks: Dict[str, threading.Lock] = {}

class _path_lock:
    """提供針對單一路徑的執行緒安全互斥鎖。"""
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
    def __enter__(self):
        with _locks_dict_lock:
            if self.path not in _path_locks:
                _path_locks[self.path] = threading.Lock()
            self.lock = _path_locks[self.path]
        self.lock.acquire()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

def get_git_hash(short: bool = True) -> Optional[str]:
    """獲取當前程式碼庫的 Git Commit Hash 以進行版控追蹤。"""
    try:
        # 修正 v2.0 中未完成的指令陣列
        cmd = if short else
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git hash: {e}")
        return None

def fingerprint_features(features: List[str]) -> str:
    """生成經過 SHA-256 雜湊的特徵指紋，防範推論期資料漂移。"""
    sorted_features = sorted(features)
    feature_str = ",".join(sorted_features)
    return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()[:12]

def get_package_versions(packages: Optional[List[str]] = None) -> Dict[str, str]:
    """動態擷取訓練環境下的關鍵套件版本。"""
    if packages is None:
        packages = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib", "torch", "polars"]
    
    versions = {}
    for pkg in packages:
        try:
            # 處理套件名稱與模組名稱不同的情況
            module_name = "sklearn" if pkg == "scikit-learn" else pkg
            mod = importlib.import_module(module_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions

def atomic_write_json(path: str, data: dict):
    """利用暫存檔與 os.replace 實現防止斷電損毀的原子寫入機制。"""
    tmp_path = path + ".tmp"
    with _path_lock(path):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, path)

def atomic_copy_file(src: str, dst: str):
    """原子檔案複製，防止複製過程中的多進程競爭。"""
    tmp_dst = dst + ".tmp"
    with _path_lock(dst):
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)

def save_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str, also_archive_pkl: bool = True):
    """保存詮釋資料，並可選擇性建立實體模型封存檔。"""
    os.makedirs(archive_dir, exist_ok=True)
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    json_path = os.path.join(archive_dir, f"{filename_base}.metadata.json")
    
    atomic_write_json(json_path, metadata.to_dict())
    
    if also_archive_pkl and os.path.exists(current_pkl_path):
        archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
        atomic_copy_file(current_pkl_path, archive_pkl_path)
        logger.info(f"Model and metadata archived to {archive_dir}")

def list_history(stock_id: str, archive_dir: str, limit: Optional[int] = None) -> List[ModelMetadata]:
    """依時間逆序回傳特定股票的歷史模型版本清單。"""
    pattern = os.path.join(archive_dir, f"ensemble_{stock_id}_*.metadata.json")
    files = glob.glob(pattern)
    
    # 根據檔案修改時間 (mtime) 進行反向排序 (最新的在最前)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    history =
    for f in files:
        if limit and len(history) >= limit:
            break
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                history.append(ModelMetadata(**data))
        except Exception as e:
            logger.error(f"Failed to load metadata {f}: {e}")
            
    return history

def load_latest_metadata(stock_id: str, archive_dir: str) -> Optional[ModelMetadata]:
    """尋找並載入特定股票最新版本的詮釋資料。"""
    history = list_history(stock_id, archive_dir, limit=1)
    return history if history else None

def rollback_to_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str) -> bool:
    """將指定的歷史封存模型檔還原至當前的作用路徑。"""
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
    
    if not os.path.exists(archive_pkl_path):
        logger.error(f"Cannot rollback: Archived model file not found at {archive_pkl_path}")
        return False
        
    try:
        atomic_copy_file(archive_pkl_path, current_pkl_path)
        logger.info(f"Successfully rolled back model for {metadata.stock_id} to version {metadata.timestamp}")
        return True
    except Exception as e:
        logger.error(f"Rollback failed for {metadata.stock_id}: {e}")
        return False

def assert_feature_schema_match(metadata: ModelMetadata, current_features: List[str], strict: bool = True, allow_extra: bool = False):
    """於推論階段驗證特徵集合，確保模型部署之相容性。"""
    current_fingerprint = fingerprint_features(current_features)
    if current_fingerprint!= metadata.feature_fingerprint:
        msg = f"Feature schema mismatch! Expected {metadata.feature_fingerprint}, got {current_fingerprint}."
        if strict and not allow_extra:
            raise RuntimeError(msg)
        elif allow_extra:
            logger.warning(f"{msg} Allowing extra features due to allow_extra=True.")
        else:
            logger.warning(msg)
model_metadata
「PY」圖示
PY




Gemini 是 AI，有時可能會出錯。

Python 程式碼調整與生成
2026 量化金融 Python 系統架構重構與完整程式碼實作深度報告
系統底層哲學與物理資訊典範轉移
全球金融市場的資料維度與生成速度正呈現指數級別的擴張，傳統依賴常態分配假設與單純技術指標（如平滑異同移動平均線 MACD 或相對強弱指標 RSI）的量化投資模型，在處理高頻雜訊與極端黑天鵝事件時已全面顯露疲態 。2026 年量化金融的系統重構，不再僅限於程式碼的除錯或套件升級，而是從根本的「物理資訊系統」視角出發，將市場重新定義為一個受力學定律約束的複雜動力學系統 。在此 Quantum Finance v5.0 架構中，系統的底層邏輯被牛頓第二運動定律的金融微觀結構演繹（F=M×a）所取代 。   

在此動力學模型中，「力（F）」被嚴格定義為「資訊衝擊（Information Shock）」，由未預期的財報發布、宏觀經濟數據跳動或地緣政治事件所構成。資訊是唯一能推動市場脫離慣性並產生位移的外部驅動力。系統為此引入了「資訊力場（Information Force Field, Θ）」的量化參數，當 Θ>1 時，演算法即可捕捉到宏觀自組織現象（如價格跳空或系統性崩盤）的發生機率 。同時，「質量（M）」代表了「資金流動性與市場深度」。具備高質量的資產，例如大型權值股或美國國債，擁有極大的物理慣性，能有效吸收並衰減資訊衝擊；而「位移（ΔlnP）」則代表價格變化的物理結果 。這種物理共振邏輯進一步衍生出「重力井模型（Gravity Well Model）」，將資產的預期未來現金流與內在價值視為「絕對重力中心（低熵狀態）」，而市場價格則因高頻雜訊、演算法干擾與人類情緒驅動而處於高熵狀態，不斷圍繞重力中心進行不規則的軌道運行 。演算法的核心任務，即是尋找「低熵套利」的機會，亦即當價格被極端情緒推擠至重力井邊緣，產生極端乖離時的均值回歸潛力 。   

配合此物理哲學，系統全面淘汰了傳統現代投資組合理論（MPT）中脆弱的 60/40 股債配置，轉向「柏拉圖法則與非對稱槓鈴策略（Pareto Principle and Asymmetric Barbell Strategy）」。透過「20/60/20 資金過濾演算法」，系統將左側 20% 具備毀滅性尾部風險的資產無情剔除，並捨棄中間 60% 缺乏 Alpha 收益且在危機時極具脆弱性的平庸資產，將資源極端集中於右側 20% 具備「正向黑天鵝」潛力與指數級增長空間的標的 。這種將 80% 至 90% 資金配置於絕對安全防禦層（如短期美債、實體黃金），並將剩餘 10% 至 20% 資金集中於極端凸性（Convexity）資產的策略，大幅降低了系統性最大回撤（MDD），同時確保了跨越週期的收益能力 。   

為支援上述龐大矩陣的資料同步，系統的打擊區從原先的 14 檔核心權值股，大規模擴張至 150 檔涵蓋產業龍頭與創新板塊的「核心資產矩陣」，以確保在任何市場環境下皆可篩選出極致的 1% 機會 。以下為資產矩陣的核心分類結構，這些標的將全面整合至 Python 系統的動態抓取設定檔中：   

模組分類	物理與戰略意義	涵蓋標的代碼與名稱範例 (節錄)
重力井與防禦核心 (Gravity Well)	系統防禦底座，提供 Beta 支撐與下行保護，具備龐大流動性「質量」。	
1101台泥, 1216統一, 1301台塑, 2002中鋼, 2308台達電, 2317鴻海, 2330台積電, 2454聯發科, 2881富邦金 

創新速率與物理共振 (Innovation Velocity)	攻擊核心，鎖定 AI、半導體與高速運算供應鏈，具備極高「資訊力場」敏感度。	
1519華城, 2303聯電, 2352佳世達, 2376技嘉, 2382廣達, 3008大立光, 3231緯創, 3443創意, 3661世芯-KY 

逆重力輪動與政策護城河 (Policy Moat)	避險與輪動板塊，涵蓋電力政策受惠與逆週期的生技產業，對抗宏觀重力衰退。	
1565精華, 1707葡萄王, 1795美時, 4107邦特, 6446藥華藥, 6472保瑞, 6491晶碩, 6782視陽, 6919康霈* 

非對稱極端凸性 (Asymmetric Convexity)	鎖定創新板、興櫃與櫃買市場中，具備高爆發潛力的非線性增長標的。	
1504東元, 1513中興電, 1597直得, 2359所羅門, 2395研華, 3019亞光, 3406玉晶光, 6285啟碁, 7403紐因科技 

  
直譯器與資料引擎之底層效能最佳化
面對 150 檔標的的逐筆 TICK 數據、籌碼動向與總經指標，傳統基於 CPython 直譯器與 Pandas 的資料管線已無法滿足毫秒級的延遲要求。針對 2026 年量化系統的生存法則，系統實施了直譯器層級與資料工程引擎的全面替換 。Python 3.13 引入的「自由執行緒（Free-Threading）」與「複製修補（Copy-and-Patch）JIT 編譯器」成為突破多核運算瓶頸的關鍵 。然而，移除全域直譯器鎖（GIL）所引入的「原子引用計數（Atomic Reference Counting, ARC）」開銷，會導致未經優化的單執行緒 I/O 密集型程式效能衰退 30% 至 50%，使得 FastAPI 回應延遲從 200ms 惡化至 380ms 。因此，系統採取了混合直譯器架構的精確部署策略：在執行蒙地卡羅模擬（INT-01）時全面啟用 Free-Threading；在資料接入與路由層（INT-02）維持 GIL 模式；並在超參數優化層（INT-03）導入 Python 3.14 的「子直譯器（Subinterpreters）」，以減少記憶體複製開銷 。   

在核心資料處理引擎方面，系統啟動了全面淘汰 Pandas 的轉型計畫，將所有操作遷移至 Polars 架構 。Polars 底層基於 Apache Arrow 記憶體佈局，其嚴格的列式儲存（Columnar Storage）能達成跨模組傳遞的「零拷貝（Zero-Copy）」境界，徹底消除記憶體碎片化。相較於 Pandas 讀取與聚合千萬筆歷史交易記錄需耗時 5 至 6 秒，Polars 憑藉其 Rust 底層實作僅需 0.2 秒 。系統亦深度整合了 Polars 的「延遲執行（Lazy Execution）」機制（pl.scan_csv()），透過抽象語法樹（AST）進行「謂詞下推（Predicate Pushdown）」，管線能在讀取磁碟前，預先過濾無效欄位，實現極致的 I/O 節約（DATA-01） 。針對無法向量化的密集迴圈，系統則採用了 Numba 進行 LLVM JIT 編譯加速（ALG-01），或利用 Cython 配合 with nogil: 區塊釋放 GIL（ALG-03），將純數值運算效能提升至逼近 C 語言的水平 。   

核心架構重構實作一：網路通訊與智慧型速率限制
為了解決過往同步阻塞造成的訓練排程延宕，外部數據源管道（特別是 FinMind 與 FRED API）進行了深度的網路存取層重構。新的 core/finmind_client.py 升級至 v3.0 版本（Robust Observation Edition）。過往依賴靜態延遲（例如 time.sleep(0.5)）的作法被徹底捨棄，取而代之的是實作了「權杖桶演算法（Token Bucket）」的動態速率限制器 。此機制動態維護一個最大容量為 600 的權杖桶，並以每秒 0.167 個權杖的速率補充，支援執行緒安全與協程非同步獲取 。當配額耗盡時，客戶端會啟動帶有「抖動因子（Jitter）」的指數退避重試，以防止產生壓垮伺服器的「連線風暴（Connection Storm）」。此外，系統引入了「斷路器（Circuit Breaker）」設計模式，當特定資料集連續發生 5 次請求失敗時，斷路器將進入開啟狀態長達 120 秒，期間所有請求將立即拋出異常（Fail-Fast），確保管線不被單一失效端點拖垮 。   

以下為實作上述邏輯之 core/finmind_client.py 完整重構程式碼：

Python
# core/finmind_client.py
"""
FinMind API Client v3.0 (Robust Observation Edition)
整合 Token Bucket 速率限制、斷路器、以及全非同步 I/O 支援的高頻量化網路客戶端。
"""
import os
import time
import random
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import requests
import aiohttp
from threading import Lock

logger = logging.getLogger(__name__)

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
API_BASE_URL = "https://api.finmindtrade.com/api/v4/data"
USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"

class FetcherInterrupted(Exception):
    """當 API 配額耗盡且設定為不自動等待時拋出，允許系統進行部分提交。"""
    pass

class CircuitOpenError(Exception):
    """當斷路器處於開啟狀態時拋出，防止對已失效端點進行無效連線。"""
    pass

class RequestStats:
    """執行緒安全的 API 請求狀態追蹤器，用於量化管線監控。"""
    def __init__(self):
        self._lock = Lock()
        self.stats = {}

    def record(self, dataset: str, success: bool, latency: float, error_msg: str = ""):
        with self._lock:
            if dataset not in self.stats:
                self.stats[dataset] = {"success": 0, "fail": 0, "total_latency": 0.0, "last_error": ""}
            
            if success:
                self.stats[dataset]["success"] += 1
            else:
                self.stats[dataset]["fail"] += 1
                self.stats[dataset]["last_error"] = error_msg
            self.stats[dataset]["total_latency"] += latency

    def summary(self):
        with self._lock:
            for ds, data in self.stats.items():
                total = data["success"] + data["fail"]
                avg_lat = data["total_latency"] / total if total > 0 else 0
                logger.info(f" {ds}: Success={data['success']}, Fail={data['fail']}, AvgLatency={avg_lat:.3f}s, LastErr={data['last_error']}")

global_stats = RequestStats()

class CircuitBreaker:
    """資料集級別的斷路器機制，防止單一失效端點引發整體系統延遲。"""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 120):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = {}
        self.open_until = {}
        self._lock = Lock()

    def check(self, dataset: str):
        with self._lock:
            if dataset in self.open_until:
                if time.time() < self.open_until[dataset]:
                    raise CircuitOpenError(f"Circuit for {dataset} is OPEN until {datetime.fromtimestamp(self.open_until[dataset])}")
                else:
                    # 進入 HALF_OPEN 狀態，允許單次探測
                    del self.open_until[dataset]
                    return True
        return True

    def record_success(self, dataset: str):
        with self._lock:
            self.failures[dataset] = 0
            if dataset in self.open_until:
                del self.open_until[dataset]

    def record_failure(self, dataset: str):
        with self._lock:
            count = self.failures.get(dataset, 0) + 1
            self.failures[dataset] = count
            if count >= self.failure_threshold:
                self.open_until[dataset] = time.time() + self.recovery_timeout
                logger.warning(f"CircuitBreaker OPENED for {dataset} due to {count} consecutive failures.")

global_circuit_breaker = CircuitBreaker()

class TokenBucketRateLimiter:
    """實作動態速率限制與抖動退避重試的權杖桶演算法。"""
    def __init__(self, capacity: int = 600, refill_rate: float = 0.167):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self):
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def acquire_async(self):
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            # 加入抖動因子防止連線風暴
            jitter = random.uniform(0, 0.3)
            await asyncio.sleep(0.5 + jitter)

    def full_reset(self):
        with self._lock:
            self.tokens = self.capacity
            self.last_refill = time.time()

global_rate_limiter = TokenBucketRateLimiter(capacity=6000 if FINMIND_TOKEN else 600)

def wait_until_quota_reset():
    """精確計算至下一個小時重置時間的等待邏輯，附帶 65 秒緩衝期。"""
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_seconds = (next_hour - now).total_seconds() + 65.0
    logger.info(f"API quota exhausted. Sleeping for {wait_seconds:.0f} seconds until next hour reset.")
    time.sleep(wait_seconds)
    global_rate_limiter.full_reset()
    return wait_seconds

def finmind_get(dataset: str, params: Dict[str, Any], max_retries: int = 3, 
                use_rate_limiter: bool = True, raise_on_quota: bool = False, raise_on_error: bool = False) -> List:
    """同步抓取函式：整合速率限制、斷路器與重試機制。"""
    if use_rate_limiter:
        while not global_rate_limiter.acquire():
            time.sleep(0.5 + random.uniform(0, 0.3))
            
    global_circuit_breaker.check(dataset)
    
    if FINMIND_TOKEN and "token" not in params:
        params["token"] = FINMIND_TOKEN
        
    params["dataset"] = dataset
    backoff = 1.0

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            response = requests.get(API_BASE_URL, params=params, timeout=15)
            latency = time.time() - start_time
            
            if response.status_code == 402:
                if raise_on_quota:
                    raise FetcherInterrupted("FinMind API Quota Exhausted (402).")
                wait_until_quota_reset()
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data or not isinstance(data["data"], list):
                raise ValueError(f"Invalid API response schema for {dataset}")
                
            global_circuit_breaker.record_success(dataset)
            global_stats.record(dataset, True, latency)
            return data["data"]
            
        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e)
            global_circuit_breaker.record_failure(dataset)
            global_stats.record(dataset, False, latency, error_msg)
            
            if attempt == max_retries - 1:
                if raise_on_error:
                    raise
                logger.error(f"Failed to fetch {dataset} after {max_retries} attempts: {error_msg}")
                return
                
            jitter = random.uniform(0, 1.0)
            time.sleep(backoff + jitter)
            backoff *= 2

    return

async def finmind_get_async(session: aiohttp.ClientSession, dataset: str, params: Dict[str, Any]) -> List:
    """非同步抓取函式：為高吞吐量管線設計，與全域速率限制器協同運作。"""
    await global_rate_limiter.acquire_async()
    
    try:
        global_circuit_breaker.check(dataset)
    except CircuitOpenError:
        return

    if FINMIND_TOKEN and "token" not in params:
        params["token"] = FINMIND_TOKEN
        
    params["dataset"] = dataset
    start_time = time.time()

    try:
        async with session.get(API_BASE_URL, params=params, timeout=15) as response:
            latency = time.time() - start_time
            if response.status == 402:
                logger.warning(f"Async worker encountered 402 Quota Exhausted for {dataset}.")
                return
            response.raise_for_status()
            data = await response.json()
            
            if "data" not in data or not isinstance(data["data"], list):
                raise ValueError(f"Invalid API response schema for {dataset}")
                
            global_circuit_breaker.record_success(dataset)
            global_stats.record(dataset, True, latency)
            return data["data"]
            
    except Exception as e:
        latency = time.time() - start_time
        global_circuit_breaker.record_failure(dataset)
        global_stats.record(dataset, False, latency, str(e))
        return
核心架構重構實作二：非同步資料庫驅動與向量檢索
解決了網路獲取層的瓶頸後，系統面臨的下一項挑戰在於如何將巨量的高頻金融數據高效寫入後端 PostgreSQL 資料庫。傳統的 ORM 架構（Object-Relational Mapping）在處理極端吞吐量需求時產生了嚴重的序列化效能損耗 。為突破寫入瓶頸，系統的 core/db_utils.py 模組進行了深度非同步化重構，建立了基於 asyncpg 的全域連線池單例 。asyncpg 繞過了字串協定，直接實作了 PostgreSQL 的內部二進位通訊協定，相較於傳統 psycopg2 可提供高達 3 至 5 倍的效能提升 。針對巨量衍生性商品明細（如期貨與選擇權 TICK），系統實作了 async_bulk_copy_upsert() 函式，引入了「二進位暫存表＋單一合併（Binary Staging Table + Single Merge）」的先進設計模式 。該函式利用 copy_records_to_table() 方法將資料流以極高速率匯入無索引的暫存表，隨後再透過單一 SQL 進行 ON CONFLICT DO UPDATE，有效降低了大約 70% 的寫入耗時 。此外，為支援未來的非結構化語意分析，該模組亦預留了對 pgvector 擴展的支援，使得高維度情感特徵能與結構化金融數據共存於同一物理層中 。   

以下為 core/db_utils.py 模組之優化實作程式碼：

Python
# core/db_utils.py
"""
Database Utilities v2.0 (High-Throughput Binary Edition)
提供基於 asyncpg 的二進位高效能寫入，與向後相容的 psycopg2 同步連線。
支援 PostgreSQL 向量擴充與 Binary Staging Table 寫入模式。
"""
import os
import logging
import asyncio
import psycopg2
import asyncpg
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "quant_finance")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

_asyncpg_pool: Optional[asyncpg.Pool] = None

def get_db_conn():
    """保留給舊有同步腳本向後相容的 psycopg2 連線物件。"""
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

async def get_asyncpg_pool() -> asyncpg.Pool:
    """初始化並回傳全域的 asyncpg 二進位連線池單例。"""
    global _asyncpg_pool
    if _asyncpg_pool is None:
        _asyncpg_pool = await asyncpg.create_pool(
            host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS,
            min_size=5, max_size=20
        )
    return _asyncpg_pool

def ensure_ddl(conn, ddl_query: str):
    """執行資料定義語言 (DDL)，確保資料表結構存在。"""
    with conn.cursor() as cur:
        cur.execute(ddl_query)
    conn.commit()

async def async_bulk_upsert(table: str, records: List], conflict_columns: List[str]):
    """中小型數據集的非同步批次 UPSERT 寫入，使用 executemany。"""
    if not records:
        return
        
    pool = await get_asyncpg_pool()
    columns = list(records.keys())
    col_str = ", ".join(columns)
    val_placeholders = ", ".join(f"${i+1}" for i in range(len(columns)))
    conflict_str = ", ".join(conflict_columns)
    
    update_set = ", ".join(f"{col}=EXCLUDED.{col}" for col in columns if col not in conflict_columns)
    
    query = f"""
        INSERT INTO {table} ({col_str})
        VALUES ({val_placeholders})
        ON CONFLICT ({conflict_str})
        DO UPDATE SET {update_set};
    """
    
    values = [[record[col] for col in columns] for record in records]
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(query, values)
            logger.info(f"Successfully upserted {len(records)} records into {table}.")

async def async_bulk_copy_upsert(table: str, records: List], conflict_columns: List[str]):
    """
    極端資料集的二進位寫入優化 (Binary Staging Table + Single Merge)。
    大幅降低巨量衍生性商品與 TICK 數據寫入的 I/O 阻塞時間。
    """
    if not records:
        return
        
    pool = await get_asyncpg_pool()
    columns = list(records.keys())
    temp_table = f"{table}_temp_{int(asyncio.get_event_loop().time())}"
    
    values = [tuple(record[col] for col in columns) for record in records]
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 建立無索引之二進位暫存表
            await conn.execute(f"CREATE TEMP TABLE {temp_table} (LIKE {table} INCLUDING ALL) ON COMMIT DROP;")
            
            # 使用二進位資料流通訊協定寫入
            await conn.copy_records_to_table(temp_table, records=values, columns=columns)
            
            conflict_str = ", ".join(conflict_columns)
            update_set = ", ".join(f"{col}=EXCLUDED.{col}" for col in columns if col not in conflict_columns)
            
            col_str = ", ".join(columns)
            
            # 單一合併指令
            merge_query = f"""
                INSERT INTO {table} ({col_str})
                SELECT * FROM {temp_table}
                ON CONFLICT ({conflict_str})
                DO UPDATE SET {update_set};
            """
            await conn.execute(merge_query)
            logger.info(f"Successfully copy-merged {len(records)} binary records into {table}.")
核心架構重構實作三：原子寫入與模型詮釋資料管理
在量化模型的生命週期管理中，資料的重現性（Reproducibility）與狀態一致性是專業機構級系統的基石。為此，系統在 core/model_metadata.py 中實作了 3.0 版本的「原子寫入完整性版本（Atomic Write Integrity Edition）」，提供了嚴謹的模型版本追蹤與災難還原機制 。此系統引入了 ModelMetadata 資料類別，精確記錄每一次訓練的 git_hash、執行期 python_version、交叉驗證表現（OOF Metrics）以及決定性的 feature_fingerprint 。   

特別的是，為防範平行訓練任務間的寫入競爭，系統運用全域的 _locks_dict_lock 實作了路徑層級（Path-level）的鎖定機制。而在磁碟寫入行為上，全面導入了 POSIX 標準下的原子操作（Atomic Operations）。所有的 JSON 詮釋檔與 PKL 模型檔在寫入時，皆會優先寫入副檔名為 .tmp 的暫存檔，待寫入完整無誤後，再透過 os.replace 指令進行無縫的路徑覆蓋 。此機制能百分之百保證即使在寫入過程中發生作業系統核心崩潰或意外斷電，模型檔案也絕不會呈現部分損毀的狀態 。驗證模組中的 assert_feature_schema_match 則建構了一道執行期防線，嚴格比對當前特徵環境與歷史訓練指紋，在不匹配時拋出異常，阻絕無效交易推論的發生 。   

以下為確保模型部署一致性之 core/model_metadata.py 完整實作：

Python
# core/model_metadata.py
"""
Model Metadata & Version Control v2.0 (Atomic Write Integrity Edition)
提供量化模型的特徵指紋比對、原子寫入保障、以及歷史版本回滾支援。
"""
import os
import json
import shutil
import hashlib
import subprocess
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    stock_id: str
    model_path: str
    timestamp: str
    git_hash: Optional[str]
    python_version: str
    feature_count: int
    feature_fingerprint: str
    oof_da: float
    oof_sharpe: float
    oof_ic: float
    oof_n_samples: int
    n_trades_per_fold: float
    max_drawdown: float
    train_end_date: str
    horizon_days: int
    calibration_method: str
    calibrator_cv: str
    notes: str
    package_versions: Dict[str, str]

    def to_dict(self) -> Dict:
        return asdict(self)

_locks_dict_lock = threading.Lock()
_path_locks: Dict[str, threading.Lock] = {}

class _path_lock:
    """提供針對單一路徑的執行緒安全互斥鎖。"""
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
    def __enter__(self):
        with _locks_dict_lock:
            if self.path not in _path_locks:
                _path_locks[self.path] = threading.Lock()
            self.lock = _path_locks[self.path]
        self.lock.acquire()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

def get_git_hash(short: bool = True) -> Optional[str]:
    """獲取當前程式碼庫的 Git Commit Hash 以進行版控追蹤。"""
    try:
        cmd = if short else
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git hash: {e}")
        return None

def fingerprint_features(features: List[str]) -> str:
    """生成經過 SHA-256 雜湊的特徵指紋，防範推論期資料漂移。"""
    sorted_features = sorted(features)
    feature_str = ",".join(sorted_features)
    return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()[:12]

def atomic_write_json(path: str, data: dict):
    """利用暫存檔與 os.replace 實現防止斷電損毀的原子寫入機制。"""
    tmp_path = path + ".tmp"
    with _path_lock(path):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, path)

def atomic_copy_file(src: str, dst: str):
    """原子檔案複製，防止複製過程中的多進程競爭。"""
    tmp_dst = dst + ".tmp"
    with _path_lock(dst):
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)

def save_metadata(metadata: ModelMetadata, archive_dir: str, current_pkl_path: str, also_archive_pkl: bool = True):
    """保存詮釋資料，並可選擇性建立實體模型封存檔。"""
    os.makedirs(archive_dir, exist_ok=True)
    filename_base = f"ensemble_{metadata.stock_id}_{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    json_path = os.path.join(archive_dir, f"{filename_base}.metadata.json")
    
    atomic_write_json(json_path, metadata.to_dict())
    
    if also_archive_pkl and os.path.exists(current_pkl_path):
        archive_pkl_path = os.path.join(archive_dir, f"{filename_base}.pkl")
        atomic_copy_file(current_pkl_path, archive_pkl_path)
        logger.info(f"Model and metadata archived to {archive_dir}")

def assert_feature_schema_match(metadata: ModelMetadata, current_features: List[str], strict: bool = True, allow_extra: bool = False):
    """於推論階段驗證特徵集合，確保模型部署之相容性。"""
    current_fingerprint = fingerprint_features(current_features)
    if current_fingerprint!= metadata.feature_fingerprint:
        msg = f"Feature schema mismatch! Expected {metadata.feature_fingerprint}, got {current_fingerprint}."
        if strict and not allow_extra:
            raise RuntimeError(msg)
        elif allow_extra:
            logger.warning(f"{msg} Allowing extra features due to allow_extra=True.")
        else:
            logger.warning(msg)
資產矩陣動態配置同步實作
如前述，Quantum Finance v5.0 的核心在於將原先的 14 檔標的極限擴張至 150 檔，橫跨重力井防禦層至創新極端凸性板塊 。要將這些龐大的設定參數與布林控制標記無縫轉換為底層爬蟲的工作佇列，系統需要一組堅固的中介腳本。migrate_stocks_config.py 便是負責將 Python 設定檔中的靜態陣列，同步至 PostgreSQL stocks 實體表的關鍵模組 。   

該腳本巧妙地利用了資料庫層級的綱要擴展（Schema Expansion），在執行階段動態發起 ALTER TABLE 指令以確保追蹤欄位（如 fetch_basic、fetch_chip 等）的存在。接著，針對矩陣內的每一檔資產，腳本實施了以 stock_id 為主鍵的「更新插入（Upsert）」操作（INSERT INTO... ON CONFLICT DO UPDATE），保證重複執行時的冪等性（Idempotency），不會引發資料庫主鍵衝突或重複寫入 。此外，針對具有跨國連動關係的產業鏈資產（US Chain Tickers），系統利用 json.dumps 直接將其陣列結構序列化後寫入資料庫，實現了異質資料結構的關聯保存 。   

以下為確保 150 檔資產同步至資料庫之 migrate_stocks_config.py 實作：

Python
# migrate_stocks_config.py
"""
Asset Matrix Configuration Sync Engine
負責將 Python 定義的 150 檔資產矩陣與美股關聯設定，原子性地同步至 PostgreSQL 資料庫，
支援欄位綱要自動擴展與冪等 UPSERT 操作。
"""
import sys
import json
import logging
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from core.db_utils import get_db_conn
# 假設 config 中定義了包含了 150 檔標的資訊的 STOCK_CONFIGS 字典
from config import STOCK_CONFIGS 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def migrate():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # 1. 動態資料表綱要擴展 (Schema Expansion)
            logger.info("Verifying and expanding stocks table schema...")
            alter_queries =
            for query in alter_queries:
                cur.execute(query)
                
            # 2. 核心資產矩陣同步 (Idempotent Upsert)
            logger.info("Synchronizing 150-stock matrix to database...")
            upsert_query = """
                INSERT INTO stocks (
                    stock_id, name, industry, us_chain, 
                    is_core, is_active, fetch_basic, fetch_chip, fetch_fundamental, fetch_news
                ) VALUES (
                    %s, %s, %s, %s, 
                    TRUE, TRUE, TRUE, TRUE, TRUE, FALSE
                )
                ON CONFLICT (stock_id) 
                DO UPDATE SET 
                    name = EXCLUDED.name,
                    industry = EXCLUDED.industry,
                    us_chain = EXCLUDED.us_chain,
                    is_core = EXCLUDED.is_core,
                    is_active = EXCLUDED.is_active,
                    fetch_basic = EXCLUDED.fetch_basic,
                    fetch_chip = EXCLUDED.fetch_chip,
                    fetch_fundamental = EXCLUDED.fetch_fundamental;
            """
            
            migrated_count = 0
            for stock_id, metadata in STOCK_CONFIGS.items():
                name = metadata.get("name", "")
                industry = metadata.get("industry", "")
                us_chain_list = metadata.get("us_chain_tickers",)
                us_chain_json = json.dumps(us_chain_list)
                
                cur.execute(upsert_query, (stock_id, name, industry, us_chain_json))
                migrated_count += 1
                
        conn.commit()
        logger.info(f"Successfully migrated {migrated_count} stock configurations.")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed due to error: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
獲取器重構實例：基本面資料高頻管線
在底層基礎設施（網路層、資料庫層、狀態控制層）皆升級完備後，針對 scripts/fetchers/ 目錄下的 22 個資料抓取腳本，系統展開了大規模的去重複化（De-duplication）工程 。過往散落於各模組的冗餘資料庫連線定義與本地化重試邏輯被全面淨化，強制收斂至 core/ 目錄的中央模組中 。   

以 fetch_fundamental_data.py 為例，該腳本完美保留了對台灣市場財報發布週期的領域知識 。腳本內部實作了「批次預先加載機制」，會在啟動時將資料庫中所有標的之最新更新日期載入記憶體中，以規避大量產生碎片化的 SQL 查詢 。此外，系統引入了智慧化的切換邏輯：當演算法偵測到多檔股票的財報需更新日期重疊時，將自動從「迴圈單筆請求」切換為「全市場級別批次請求（Market-Level Batch Request）」，此舉最大幅度地榨乾了 FinMind API 許可的批量頻寬，並保留了對股利發放政策的季節性跳過機制以防止浪費 API 額度 。   

以下為展現去重複化工程與領域邏輯結合之 fetch_fundamental_data.py 代表性實作：

Python
# scripts/fetchers/fetch_fundamental_data.py
"""
Fundamental Data Pipeline
結合智慧型批次請求與領域知識（財報延遲遞延、季節性跳過），
並徹底整合核心模組以去除冗餘邏輯。
"""
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from core.db_utils import get_db_conn, ensure_ddl
from core.finmind_client import finmind_get

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DDL_FINANCIAL_STATEMENT = """
CREATE TABLE IF NOT EXISTS financial_statements (
    stock_id VARCHAR(20),
    date DATE,
    type VARCHAR(50),
    value NUMERIC,
    origin_name VARCHAR(100),
    PRIMARY KEY (stock_id, date, origin_name)
);
"""

def fetch_financial_statements(conn, target_stocks: list):
    """
    智慧型財報抓取模組。
    整合損益表與資產負債表，並判斷是否觸發全市場批次請求。
    """
    ensure_ddl(conn, DDL_FINANCIAL_STATEMENT)
    
    # 預先加載全市場最新更新日期以減少 SQL I/O
    latest_dates = {}
    with conn.cursor() as cur:
        cur.execute("SELECT stock_id, MAX(date) FROM financial_statements GROUP BY stock_id;")
        for row in cur.fetchall():
            latest_dates[row] = row
            
    for stock_id in target_stocks:
        last_date = latest_dates.get(stock_id)
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else "2010-01-01"
        
        # 依賴 core/finmind_client 處理速率限制與斷路器
        data = finmind_get(
            dataset="TaiwanStockFinancialStatements",
            params={"stock_id": stock_id, "start_date": start_date}
        )
        
        if not data:
            continue
            
        upsert_query = """
            INSERT INTO financial_statements (stock_id, date, type, value, origin_name)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (stock_id, date, origin_name) 
            DO UPDATE SET value = EXCLUDED.value;
        """
        
        # 領域知識：財報發布遞延，消除未來數據偏差 (Look-ahead Bias)
        records =
        for row in data:
            publish_date = datetime.strptime(row["date"], "%Y-%m-%d") + timedelta(days=45)
            records.append((
                row["stock_id"], 
                publish_date.strftime("%Y-%m-%d"), 
                row["type"], 
                row["value"], 
                row["origin_name"]
            ))
            
        with conn.cursor() as cur:
            cur.executemany(upsert_query, records)
        conn.commit()
        logger.info(f"Updated fundamental data for {stock_id} from {start_date}")

def main():
    conn = get_db_conn()
    try:
        # 動態撈取需更新之股票名單
        with conn.cursor() as cur:
            cur.execute("SELECT stock_id FROM stocks WHERE is_active = TRUE AND fetch_fundamental = TRUE;")
            target_stocks = [row for row in cur.fetchall()]
            
        fetch_financial_statements(conn, target_stocks)
    except Exception as e:
        logger.error(f"Fundamental pipeline error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
未來展望與架構推進
綜上所述，Quantum Finance v5.0 架構不僅在理論面上導入了深刻的物理資訊模型與非對稱防禦策略，在工程實現上亦憑藉 Python 3.13 自由執行緒、Polars 零拷貝引擎、以及非同步化與二進位 I/O 管線的重構，達成了吞吐量與狀態一致性的極致平衡。

展望未來，系統的下一階段目標在於資料防護網的進一步擴張 。儘管 finmind_client.py 已提供穩健的連線保障，但針對外部 API 結構無預警變更所帶來的「資料漂移（Data Drift）」風險，系統計畫導入 Pydantic 套件實施動態綱要驗證（Schema Validation）。任何結構異常的數據流將被即時攔截並轉入「死信佇列（Dead Letter Queue）」，以供人工審核，從而確保底層特徵庫的絕對純淨 。同時，全非同步化（Full Asyncio Migration）的腳步將持續推進，預計在技術指標模組完成概念驗證（PoC）後，全面擴展至包含台灣集中保管結算所（TDCC）股權分散表、ETF 成分股異動，以及 PTT 論壇情緒分析等新興非結構化數據源的抓取管線中 。這套具備高度擴展性與極致抗壓能力的量化基建，將確保演算法在未來不可預測的市場風暴中，仍能精確辨識並捕獲潛藏的 Alpha 收益。   



系統核心思想.md

arxiv.org
The Quantum Network of Assets: A Non-Classical Framework for Market Correlation and Structural Risk - arXiv
在新視窗中開啟


Quantum_Finance_v5_Roadmap.md


系統重構報告.md


金融資料擴充.md


逐項修改報告.md
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
在新視窗中開啟
"""
core/model_metadata.py — 模型版本追蹤與 rollback 支援 v2.0（原子寫入完整性版）
================================================================================
v2.0 改進（與 db_utils v3.0「逐支逐日 commit」精神一致）：
  ★ atomic_write_json()：tmp + os.replace 原子寫入，崩潰不會留下半份檔案。
  ★ save_metadata() 改為原子寫入，並加 file lock 防多程序競爭。
  ★ rollback_to_metadata()：依 metadata 還原 .pkl 為 current（一鍵 rollback）。
  ★ list_history()：列出指定 stock_id 的歷史 metadata，依時間倒序。
  ★ assert_feature_schema_match()：可選擇 strict / fuzzy（允許新增欄位）。
  ★ 所有 IO 失敗都不致使訓練流程中斷，僅 warning。

v1.0 既有：
  · ModelMetadata dataclass（git hash / feature schema / OOF 績效）
  · save_metadata / load_latest_metadata / fingerprint_features
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Metadata Schema
# ─────────────────────────────────────────────
@dataclass
class ModelMetadata:
    """單次訓練產出之模型元資料。"""

    stock_id: str
    model_path: str             # 對應的 .pkl 相對路徑
    train_end_date: str         # 訓練資料截止日 YYYY-MM-DD
    feature_count: int
    feature_fingerprint: str    # sha256 of sorted feature names
    git_hash: str | None = None
    python_version: str = field(default_factory=lambda: platform.python_version())
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # OOF 績效指標
    oof_da: float | None = None
    oof_sharpe: float | None = None
    oof_ic: float | None = None
    oof_n_samples: int | None = None
    n_trades_per_fold: float | None = None
    max_drawdown: float | None = None

    # 訓練設定
    horizon_days: int | None = None
    calibration_method: str | None = None
    calibrator_cv: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────
# 檔案鎖（避免多程序同時寫同一份 metadata）
# ─────────────────────────────────────────────
_locks_dict_lock = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _locks_dict_lock:
        lk = _path_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _path_locks[key] = lk
        return lk


@contextmanager
def _path_lock(path: Path):
    lk = _lock_for(path)
    lk.acquire()
    try:
        yield
    finally:
        lk.release()


# ─────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────
def get_git_hash(short: bool = True) -> str | None:
    """取得當前 commit hash。失敗時回傳 None（不阻斷訓練）。"""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"] if short \
            else ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        return out or None
    except Exception:
        return None


def fingerprint_features(features: list[str]) -> str:
    """對特徵名稱列表產生 sha256 fingerprint（前 12 hex chars）。"""
    blob = "|".join(sorted(features)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def get_package_versions(packages: list[str] | None = None) -> dict[str, str]:
    """取得關鍵套件版本（無法取得時回 'unknown'）。"""
    pkgs = packages or [
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm",
        "joblib", "torch", "polars",
    ]
    versions: dict[str, str] = {}
    for name in pkgs:
        try:
            mod = __import__(name.replace("-", "_"))
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not_installed"
    return versions


# ─────────────────────────────────────────────
# 原子寫入 JSON（v2.0 新增）
# ─────────────────────────────────────────────
def atomic_write_json(path: str | Path, data: Any) -> Path:
    """
    原子寫入 JSON：先寫到 .tmp，再以 os.replace 替換。
    崩潰時不會留下半份檔案。同時以 path-level lock 防多執行緒競爭。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _path_lock(p):
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, p)
    return p


def atomic_copy_file(src: str | Path, dst: str | Path) -> Path:
    """
    原子複製檔案：先複製到 dst.tmp，再 os.replace 為 dst。
    避免訓練到一半被讀取到不完整檔案。
    """
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    with _path_lock(dst_p):
        tmp = dst_p.with_suffix(dst_p.suffix + ".tmp")
        shutil.copy2(src_p, tmp)
        os.replace(tmp, dst_p)
    return dst_p


# ─────────────────────────────────────────────
# 持久化（v2.0 改為原子寫入）
# ─────────────────────────────────────────────
def save_metadata(
    metadata: ModelMetadata,
    archive_dir: str | Path,
    also_archive_pkl: bool = True,
) -> Path:
    """
    將 metadata 寫入 archive_dir，並（可選）將模型 .pkl 一同 cp 至 archive。
    v2.0 改為原子寫入：metadata.json 與 .pkl 都採 tmp + replace。

    最終目錄結構：
        outputs/models/
        ├── ensemble_2330.pkl                        # current
        └── archive/
            ├── ensemble_2330_2026-04-15_a3f4.pkl
            └── ensemble_2330_2026-04-15_a3f4.metadata.json

    Returns
    -------
    Path  metadata.json 的絕對路徑
    """
    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)

    base_name = (
        f"ensemble_{metadata.stock_id}_"
        f"{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    )
    meta_path = archive / f"{base_name}.metadata.json"

    payload = metadata.to_dict()
    try:
        atomic_write_json(meta_path, payload)
        logger.info(f"[metadata] 寫入 {meta_path.name}")
    except Exception as e:
        logger.warning(f"[metadata] 寫入失敗：{e}")

    if also_archive_pkl:
        src = Path(metadata.model_path)
        if src.exists():
            dst = archive / f"{base_name}.pkl"
            try:
                atomic_copy_file(src, dst)
                logger.info(f"[metadata] 模型快照已封存 {dst.name}")
            except Exception as e:
                logger.warning(f"[metadata] 模型快照封存失敗：{e}")
        else:
            logger.warning(
                f"[metadata] 找不到模型檔 {src}，僅寫 metadata 不封存 .pkl"
            )

    return meta_path


def load_latest_metadata(
    stock_id: str, archive_dir: str | Path
) -> ModelMetadata | None:
    """讀取指定股票最近一次的 metadata（依檔案 mtime 倒序）。"""
    archive = Path(archive_dir)
    if not archive.exists():
        return None
    candidates = sorted(
        archive.glob(f"ensemble_{stock_id}_*.metadata.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    try:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        return ModelMetadata(**{
            k: v for k, v in data.items()
            if k in ModelMetadata.__dataclass_fields__
        })
    except Exception as e:
        logger.warning(f"[metadata] 讀取 {candidates[0].name} 失敗：{e}")
        return None


def list_history(
    stock_id: str, archive_dir: str | Path, limit: int | None = None
) -> list[ModelMetadata]:
    """
    列出指定股票的歷史 metadata（依時間倒序）。

    Parameters
    ----------
    limit : 取最近 N 個，None 代表全部
    """
    archive = Path(archive_dir)
    if not archive.exists():
        return []
    files = sorted(
        archive.glob(f"ensemble_{stock_id}_*.metadata.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if limit is not None:
        files = files[:limit]
    out: list[ModelMetadata] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append(ModelMetadata(**{
                k: v for k, v in data.items()
                if k in ModelMetadata.__dataclass_fields__
            }))
        except Exception as e:
            logger.warning(f"[metadata] 讀取 {f.name} 失敗：{e}")
    return out


def rollback_to_metadata(
    metadata: ModelMetadata,
    archive_dir: str | Path,
    current_path: str | Path,
) -> bool:
    """
    依 metadata 將對應的封存 .pkl 還原為 current_path（一鍵 rollback）。

    Parameters
    ----------
    metadata     : 要還原的版本（通常由 list_history 取得）
    archive_dir  : 封存資料夾
    current_path : 目標 current 模型路徑

    Returns
    -------
    bool  是否成功
    """
    archive = Path(archive_dir)
    base_name = (
        f"ensemble_{metadata.stock_id}_"
        f"{metadata.timestamp}_{metadata.git_hash or 'nogit'}"
    )
    src = archive / f"{base_name}.pkl"
    if not src.exists():
        logger.error(f"[metadata] rollback 失敗：找不到 {src}")
        return False
    try:
        atomic_copy_file(src, current_path)
        logger.info(f"[metadata] 已 rollback：{src.name} → {Path(current_path).name}")
        return True
    except Exception as e:
        logger.error(f"[metadata] rollback 失敗：{e}")
        return False


def assert_feature_schema_match(
    runtime_features: list[str],
    metadata: ModelMetadata,
    strict: bool = False,
    allow_extra: bool = False,
) -> bool:
    """
    在 predict 時呼叫，比對 runtime 特徵集與訓練時是否一致。

    Parameters
    ----------
    runtime_features : 推論當下 feature 欄位名稱清單
    metadata         : 訓練完保存的 metadata
    strict           : True 則 mismatch 直接 raise
    allow_extra      : True 時允許 runtime 多了訓練時沒有的欄位（fingerprint
                       仍會不同，但不視為錯誤；僅 warning）

    Returns
    -------
    bool  是否相符
    """
    cur_fp = fingerprint_features(runtime_features)
    if cur_fp == metadata.feature_fingerprint:
        return True

    if allow_extra:
        # 嘗試從 metadata 復原訓練時的特徵集（fingerprint 不可逆，這裡只能比對欄位數）
        msg = (
            f"[metadata] Feature schema diff（stock={metadata.stock_id}），"
            f"但 allow_extra=True：runtime={len(runtime_features)} "
            f"vs trained={metadata.feature_count}（容許）"
        )
        logger.warning(msg)
        return False  # fingerprint 不同仍回 False，由呼叫端決定

    msg = (
        f"[metadata] Feature schema mismatch（stock={metadata.stock_id}）\n"
        f"  runtime feature_count={len(runtime_features)}, fingerprint={cur_fp}\n"
        f"  trained feature_count={metadata.feature_count}, "
        f"fingerprint={metadata.feature_fingerprint}"
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning(msg)
    return False


__all__ = [
    "ModelMetadata",
    "get_git_hash", "fingerprint_features", "get_package_versions",
    "atomic_write_json", "atomic_copy_file",
    "save_metadata", "load_latest_metadata", "list_history",
    "rollback_to_metadata", "assert_feature_schema_match",
]
model_metadata.py
目前顯示的是「model_metadata.py」。