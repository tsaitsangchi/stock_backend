"""
multi_cycle_ftransformer_validation.py v0.1 (FT-Transformer · Gorishniy et al. 2021 "Revisiting Deep Learning for Tabular Data" · Multi-Cycle Cross-Sectional Stock-Return Validation · per CLAUDE.md §一.11 三段式)
================================================================================
**最後更新日期**: 2026-06-04
**主權狀態**: FT-TRANSFORMER (GORISHNIY 2021) 4-HORIZON WALK-FORWARD CROSS-SECTIONAL VALIDATION + §14.7-CY HORIZON-DOCTRINE + §14.7-DC v0.17 SOURCE-PURE UNIVERSE(397 核心)+ §一.10 SOURCE-TRACEABLE(全 DB)+ §14.7-DE/DF §0.0-I 雙 helper(panel-date + metric SSOT)+ COMMON COMPARISON BASELINE + §一.11 三段式合規
**最高原則**: THE SUPREME AUTHORITY PRINCIPLE (最高權限原則)

## 🎯 零、這支程式在做什麼(白話說明,給人看的)

**一句話**:用 FT-Transformer(表格資料專用的 transformer:把每支股票的 37 個特徵各自轉成一個「token 向量」,再用 self-attention 讓特徵彼此互相參照)這套深度模型,實測「靠過去的股票特徵能不能預測未來、選股賺錢」,並選最看好的前 20 支股票。

**它怎麼做(步驟)**:
1. 取 397 支「乾淨核心股」+ 每股 37 個經驗證的特徵(估值、動能、籌碼、財務等;全來自真實 FinMind/FRED API,無 AI 亂補值)。
2. 把 2013-05 ~ 2026-06 切成月度時間點(panel)。
3. **逐點往前走(walk-forward)**:每個時間點「只用那之前的資料」訓練模型,預測「之後」每支股票的報酬 —— 不偷看未來(防資料洩漏)。
4. 模型怎麼算:**特徵 tokenizer** 把每個數值特徵 ×自己的權重 +偏置 變成一個 d 維小向量(token),再加一個可學的 [CLS] token → 變成 38 個 token;**transformer encoder**(多層 self-attention + 前饋網路)讓這些 token 互相交流;最後取 [CLS] token → 過 LayerNorm + Linear → 一個純量,即該股的預期報酬。
5. 訓練時讓「模型預測的報酬」盡量貼近「真實未來報酬」(MSE 最小化;特徵先用只看訓練資料配適的 StandardScaler 標準化)。
6. 預測下一期橫斷面每支股票的分數 → 挑最看好的 **前 20 支等權做多**,跟「全市場平均」比,算這一期賺賠。
7. 在 **4 種持有期**各做一遍:週(5 天)/ 月(20 天)/ 季(60 天)/ 年(252 天)。
8. 算每種週期的成績:**報酬率、Sharpe、勝率、預測命中率、跨次穩定度(可信度)、扣成本後年化報酬**(全經共用 helper `summarize_horizon_metrics`,與其他模型同指標 → 可比)。

**輸入**:資料庫(feature_values 特徵 + TaiwanStockPriceAdj 股價)。**輸出**:JSON(各週期成績)+ log。
**它不做的事**:不訓練正式上線模型、不改資料庫(純讀取評估;§3.2 evaluation 角色)。
**為什麼需要它**:FT-Transformer 是表格資料深度學習的代表性架構(Gorishniy et al. 2021);本程式把它納入「共同比較基準」,與 tree / 其他深度模型用同窗、同指標並排比較其精準度與信任度。

## 📜 一、核心定義說明 (Core Definitions / The Constitution)

1. **[Sovereignty Declaration]** (v0.1, 憲法 §3.2 橫切分析模組): 本程式為 evaluation 後處理工具,非序列落地鏈模組。
   **治權邊界**:(a) §3.2 evaluation 模組;(b) 不涉五套禁令(§0.1-A / §0.2-A / §0.3-A / §0.0-E.4 / §6.8);(c) T1-T3 不分層;
   (d) §8.5 anti-leakage:walk-forward 僅以「過去 panels」訓練(train [0..i-1] → predict i);StandardScaler 只 fit 訓練資料;**目標 panel 之 forward returns 從不進入訓練**(features 本身已 anti-leakage compliant per feature_store_builder);
   (e) **不訓練 production model**(不寫 model_registry);(f) **read-only**(不改 feature_values / TaiwanStockPriceAdj / universe);
   (g) 唯一職責:FT-Transformer 4-horizon walk-forward 橫斷面評估 + 計算 metrics + 持久化 JSON。
2. **[Real FT-Transformer — Gorishniy et al. 2021]** (v0.1, Gorishniy/Rubachev/Khrulkov/Babenko, NeurIPS 2021 "Revisiting Deep Learning for Tabular Data"):
   核心 = feature-tokenizer + transformer。**FEATURE TOKENIZER**:每個數值特徵各自一條學習線性嵌入(value × weight_j + bias_j)→ d 維 token;prepend 一個可學 [CLS] token → 38 token × d 序列;
   **TRANSFORMER ENCODER**:數層(N_LAYERS)標準 multi-head self-attention + FFN(pre-norm;n_heads=4;dropout~0.1);
   **HEAD**:取 [CLS] token 輸出 → LayerNorm → Linear → 純量 r̂。從頭以 torch.nn 實作(`rtdl` 套件無法安裝故自行實作);CPU + seeded;
   Adam(lr 1e-3 / weight_decay 1e-5)最小化 MSE(r̂, 真實 forward return)。d_token / n_layers / n_heads / epochs 為 module 常數。
3. **[Common Comparison Baseline]** (v0.1, reports/common_model_comparison_baseline_v1.md): 與全 tree / 其他深度模型套用**完全相同** protocol —
   397 source-pure 核心股 × canonical panels(get_canonical_panel_dates 資料驅動)× 真實 forward log returns(TaiwanStockPriceAdj)× 4 horizons(5/20/60/252d)×
   top-20 equal-weight long × 0.6% cost × summarize_horizon_metrics(§14.7-DF 單一 metric 來源)× T_CZ-6 gate。
   模型用各自 natural representation(tree=37 cross-sectional features 直接;FT-Transformer=同 37 features → per-feature token + self-attention),比較點在 OUTPUT 預測品質,非 input。
4. **[Source Traceability]** (v0.1, CLAUDE.md §一.10): 全數據 (b) DB query(feature_values + TaiwanStockPriceAdj + core_universe_*)+ (a) program output(本 JSON / log);
   **0 AI memory reuse**;**禁止從記憶 / 推測 / 估算**;source-pure 過濾 `is_null_imputed IS NOT TRUE`(§14.7-DC + §一.13 第四類幻像值排除)。
5. **[Anti-Leakage Walk-Forward]** (v0.1, 憲法 §8.5): 訓練只用 [0..i-1] panels 之 (X, forward-return) rows;StandardScaler 僅 fit 訓練 X;
   預測 panel i 時用該 panel 之特徵(已 anti-leakage)+ 訓練好的模型;**目標 panel 之 forward returns 從不用於訓練 / scaler / 任何擬合**(否則洩漏)。
6. **[Zero Hardcoded Verdict]** (v0.1, 憲法 §5.6.3): 4 horizons 之 significance verdict 由 helper 動態計算(`abs(eff_t) > 1.997`);
   T_CZ-6 gate(4.20/2.40/0.79)為 charter-mandated reference threshold(Tier 3 transparent disclosure,非 feature data,非硬編 verdict)。
7. **[§一.10 #3 Multi-Run]** (v0.1): stochastic(torch init / sgd shuffle / dropout)→ 須 ≥3 seeds(canonical 5422);
   single-run 不得作為 deterministic charter fact;median 為 inscription central estimate。
8. **[Schema-Compatible Output]** (v0.1): per-horizon JSON keys 由 summarize_horizon_metrics 產出 → 與全 tree validator 對齊 → 可直接 roll up 並比較。
9. **[Historical Reference Authority]** (v0.1): TOOL_VER = "v0.1" 為記述性快照,非權威來源。
10. **[Idempotency]** (v0.1): 不寫 DB(--output 寫 JSON,path 含 timestamp 不覆寫);可重跑;d_token / n_layers / n_heads / epochs / lr 為 module 常數。

## 📊 二、全量功能群矩陣 (The Ultimate Functional Group Matrix)

> 本程式作為 FT-Transformer 共同比較基準實作,依「驗證面向」拆分為 5 大功能群;每群對應憲章治權契約。

### Group A. Common Comparison Baseline DB loaders(與全 tree validator 逐字相同)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| A.1 Universe | 最新 committed snapshot core_tier='core_universe'(397 source-pure)| §14.7-DC v0.17 |
| A.2 Panels | get_canonical_panel_dates('feature_set_v0.6')| §14.7-DE / §0.0-I 單一引用源 |
| A.3 Features | load_features() → feature_values WHERE is_null_imputed IS NOT TRUE | §一.10 (b) DB / §14.7-DC |
| A.4 Forward returns | load_forward_returns() → 真實 log return(TaiwanStockPriceAdj)| §一.10 (b) DB |
| A.5 Metrics | summarize_horizon_metrics()(top-20 / 0.6% cost / Sharpe / Eff-t / T_CZ-6)| §14.7-DF / §14.7-CY / §14.7-CZ |

### Group B. FT-Transformer Model(feature tokenizer + transformer encoder + CLS head)

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| B.1 Feature tokenizer | `FTTransformer.tokenizer`(37 numeric → 38 token × d,含 [CLS])| Gorishniy et al. 2021 |
| B.2 Transformer encoder | `FTTransformer.encoder`(N_LAYERS × MHSA + FFN,pre-norm)| Gorishniy et al. 2021 |
| B.3 CLS head | `FTTransformer.head`([CLS] → LayerNorm → Linear → 純量)| Gorishniy et al. 2021 |
| B.4 Train per window | `train_ftt()` → StandardScaler(train-only)+ Adam + MSE(CPU)| §8.5 leakage-safe |
| B.5 Predict next panel | `predict_panel()` → forward(scaled Z_test) → r̂ | §8.5 leakage-safe |

### Group C. Walk-Forward Cross-Sectional Evaluation

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| C.1 Expanding window | `for i in range(1, len(panel_keys))`(train [0..i-1] → test i)| §14.7-CW T_CW-2 |
| C.2 Winsorization | `winsorize(arr, 0.01, 0.99)` train labels | label stability |
| C.3 (pred, actual) per panel | `panel_pa.append((pred_te, y_test))` | §14.7-DF 單一來源 |
| C.4 Significance verdict | summarize_horizon_metrics → `abs(eff_t) > 1.997` | §5.6.3 動態 verdict |

### Group D. Cross-Cycle Comparison + Persistence

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| D.1 Cross-cycle matrix | stdout 彙總 4 horizons | §14.7-CY T_CY-6 |
| D.2 JSON persistence | `reports/multi_cycle_ftransformer_<timestamp>.json` | §14.7-CY T_CY-1 |
| D.3 Meta context | `_meta`(tool_ver / torch_version / seed / d_token / run_at)| §一.10 source_traceability |

### Group E. Source Traceability + Anti-Hallucination

| 子項 | 對應方法 / 行為 | 治權契約 |
| :--- | :--- | :--- |
| E.1 Universe via DB | core_universe_membership + snapshot JOIN | §一.10 (b) DB query |
| E.2 Features via DB | feature_values WHERE is_null_imputed IS NOT TRUE | §一.10 (b) / §14.7-DC |
| E.3 Forward returns via DB | TaiwanStockPriceAdj LN(t1/t0)JOIN | §一.10 (b) DB query |
| E.4 0 AI memory reuse | All numbers from program output | §一.10 第 1 條 |

### 對齊憲章 §二 維運矩陣（標準場景索引）

| 場景 | 對應命令 |
| :--- | :--- |
| 日常 multi-cycle validation | `./venv/bin/python scripts/evaluation/multi_cycle_ftransformer_validation.py --dry-run` |
| 完整單 seed + 輸出 | `... --seed 5422 --output reports/multi_cycle_ftransformer_<ts>.json` |
| 3-run 教義全合規 | 對 {5422,7331,1009} 各跑一次 → 跨 seed 聚合 |

### 不提供之旗標 (Intentionally Omitted)

- `--cost-per-rebal X`:0.6% 為 §14.7-CY T_CY-5 conservative standard(與 baseline 對齊,不可變更否則破壞比較基準)。
- `--horizons`:固定 5/20/60/252(共同比較基準定義之一部分,不可變更)。
- `--write-model`:本程式為 evaluation only,不寫 `model_registry`(屬 model_trainer 治權;§3.2 橫切只讀)。

## 📜 三、全修訂歷程 (Full Revision History)

| 版本 | 日期 | 修訂者 | 修訂說明 | 治權狀態 |
| :--- | :--- | :--- | :--- | :--- |
| v0.1 | 2026-06-04 | Codex | **首版:FT-Transformer(Gorishniy et al. 2021 "Revisiting Deep Learning for Tabular Data")multi-cycle 橫斷面驗證 + 共同比較基準實作**。**功能 4 點**:(a) feature tokenizer(37 numeric → per-feature linear embedding + [CLS] → 38 token × d)+ N_LAYERS transformer encoder(MHSA + FFN,pre-norm,n_heads=4,dropout 0.1)+ CLS head(LayerNorm → Linear → 純量)end-to-end Adam/MSE(torch CPU seeded;`rtdl` 無法安裝故自行 torch.nn 實作);(b) walk-forward expanding window(train [0..i-1] → predict i);(c) anti-leakage StandardScaler 僅 fit 訓練 X,目標 panel forward returns 從不入訓練;(d) 4-horizon(5/20/60/252d)top-20 long / 0.6% cost。**治權邊界**:read-only(不寫 model_registry / 不改 feature_values);全 (b) DB query;0 AI memory(per §一.10);source-pure `is_null_imputed IS NOT TRUE`(§14.7-DC)。**單一引用源**:panel 窗 get_canonical_panel_dates(§14.7-DE)+ metric summarize_horizon_metrics(§14.7-DF)。**Output**:stdout + JSON。**FT-Transformer 常數**:d_token=32 / n_layers=3 / n_heads=4 / ffn_factor=2 / dropout=0.1 / epochs=40 / batch=512 / lr=1e-3 / seed 5422。 | **ACTIVE** |
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")  # NumPy log(NaN) + torch non-fatal warnings
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys, argparse, math, json, logging, time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent  # scripts/
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

import numpy as np
from core.db_utils import get_db_conn, get_canonical_panel_dates, summarize_horizon_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

CONSTITUTION_VER = "v6.1.0"
TOOL_VER = "v0.1"

DEFAULT_HORIZONS = [("weekly", 5), ("monthly", 20), ("quarterly", 60), ("annual", 252)]

# ── FT-Transformer constants (Gorishniy et al. 2021) ──
D_TOKEN = 32           # per-feature embedding dimension (token width)
N_LAYERS = 3           # transformer encoder layers
N_HEADS = 4            # multi-head self-attention heads (D_TOKEN % N_HEADS == 0)
FFN_FACTOR = 2         # FFN hidden = D_TOKEN * FFN_FACTOR
DROPOUT = 0.1          # attention + FFN dropout
EPOCHS = 40            # end-to-end training epochs (CPU, modest)
BATCH_SIZE = 512       # minibatch size
LR = 1e-3              # Adam learning rate
WEIGHT_DECAY = 1e-5    # L2 regularization

SPEC_43 = [
    "log_return_20d", "log_return_60d", "log_return_252d", "ma_ratio_20", "ma_ratio_60", "max_drawdown_252d",
    "upside_volatility_60d", "downside_volatility_60d", "convexity_60d", "volatility_60d", "volatility_252d",
    "upside_capture_60d", "downside_capture_60d",
    "avg_daily_value_log_60d", "avg_daily_value_log_252d", "turnover_mean_60d",
    "pe_ratio", "pb_ratio", "dividend_yield",
    "roe_ttm", "operating_margin_ttm", "eps_sum_4q", "revenue_yoy_3m_log", "asset_growth_yoy", "revenue_yoy_3m", "revenue_yoy_12m",
    "right_tail_returns_skew_252d",
    "liquidity_rank_pct_sector_60d", "size_log_zscore_sector",
    "foreign_net_20d", "foreign_net_60d", "margin_ratio_60d",
    # §14.7-DC v0.3 strict: theme_is_semiconductor + fitness_signal_60d + theme_strength all removed (hardcoded knowledge / transitively tainted = AI hallucination)
]
N_FEATURES = len(SPEC_43)


def load_features(cur, fs_id, universe):
    cur.execute("SELECT stock_id, feature_name, feature_value::numeric FROM feature_values WHERE feature_set_id=%s AND stock_id=ANY(%s) AND is_null_imputed IS NOT TRUE", (fs_id, list(universe)))
    feat_data = defaultdict(dict)
    for sid, fname, val in cur.fetchall():
        if val is not None and fname in SPEC_43:
            feat_data[sid][fname] = float(val)
    X, sids = [], []
    for sid in universe:
        if sid in feat_data:
            X.append([feat_data[sid].get(f, 0.0) for f in SPEC_43])
            sids.append(sid)
    return X, sids


def load_forward_returns(cur, as_of, horizon_days):
    cur.execute("""
        SELECT MIN(date) FROM "TaiwanStockPriceAdj"
        WHERE date >= (%s::date + INTERVAL '%s days')
          AND stock_id ~ '^[0-9]'
          AND date <= (%s::date + INTERVAL '%s days')
    """, (str(as_of), horizon_days, str(as_of), horizon_days + 14))
    r = cur.fetchone()
    label_date = r[0] if r and r[0] else None
    if not label_date: return {}, None
    cur.execute("""
        WITH t0 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0),
             t1 AS (SELECT stock_id, close FROM "TaiwanStockPriceAdj" WHERE date=%s AND close>0)
        SELECT t0.stock_id, LN(t1.close::numeric/t0.close::numeric)
        FROM t0 JOIN t1 ON t0.stock_id=t1.stock_id
    """, (str(as_of), str(label_date)))
    return {sid: float(r) for sid, r in cur.fetchall()}, label_date


def winsorize(arr, lo_q=0.01, hi_q=0.99):
    return np.clip(arr, np.quantile(arr, lo_q), np.quantile(arr, hi_q))


# ── FT-Transformer architecture (from-scratch torch.nn; Gorishniy et al. 2021) ──
def _build_model():
    import torch
    import torch.nn as nn

    class FeatureTokenizer(nn.Module):
        """Numerical feature tokenizer: each of N_FEATURES gets its own learned linear embedding
        token_{j} = value_{j} * weight_{j} + bias_{j}  (weight_j, bias_j ∈ R^d).
        A learnable [CLS] token is prepended → sequence of (N_FEATURES + 1) tokens × d."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(N_FEATURES, D_TOKEN))
            self.bias = nn.Parameter(torch.empty(N_FEATURES, D_TOKEN))
            self.cls_token = nn.Parameter(torch.empty(1, D_TOKEN))
            nn.init.normal_(self.weight, std=0.02)
            nn.init.normal_(self.bias, std=0.02)
            nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, x):                       # x:[B, N_FEATURES]
            tokens = x.unsqueeze(-1) * self.weight + self.bias  # [B, N_FEATURES, d]
            cls = self.cls_token.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B, 1, d]
            return torch.cat([cls, tokens], dim=1)  # [B, N_FEATURES+1, d]

    class FTTransformer(nn.Module):
        """Feature-tokenizer + transformer encoder + [CLS] head (Gorishniy et al. 2021).
        tokenizer : [B, N_FEATURES] → [B, N_FEATURES+1, d] (per-feature embedding + [CLS]).
        encoder   : N_LAYERS × pre-norm multi-head self-attention + FFN.
        head      : [CLS] token output → LayerNorm → Linear → scalar prediction r̂."""

        def __init__(self):
            super().__init__()
            self.tokenizer = FeatureTokenizer()
            layer = nn.TransformerEncoderLayer(
                d_model=D_TOKEN, nhead=N_HEADS,
                dim_feedforward=D_TOKEN * FFN_FACTOR,
                dropout=DROPOUT, activation="gelu",
                batch_first=True, norm_first=True,   # pre-norm
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
            self.head_norm = nn.LayerNorm(D_TOKEN)
            self.head_linear = nn.Linear(D_TOKEN, 1)

        def forward(self, x):                       # x:[B, N_FEATURES]
            tokens = self.tokenizer(x)              # [B, N_FEATURES+1, d]
            enc = self.encoder(tokens)              # [B, N_FEATURES+1, d]
            cls = enc[:, 0, :]                       # [B, d]  ([CLS] token output)
            return self.head_linear(self.head_norm(cls)).squeeze(-1)  # [B]

    return FTTransformer()


class _StandardScaler:
    """Minimal StandardScaler (fit on train only); degenerate columns → unit scale."""

    def fit(self, X):
        arr = np.asarray(X, dtype=float)
        self.mu = arr.mean(axis=0)
        self.sd = arr.std(axis=0)
        self.sd[self.sd < 1e-10] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mu) / self.sd


def train_ftt(X_train, y_train, seed, epochs):
    """Train FT-Transformer on pooled training rows (X_train: [N, N_FEATURES], y_train: [N]).
    StandardScaler is fit on TRAIN ONLY (anti-leakage); minibatch Adam + MSE on CPU, seeded.
    Returns (model, scaler) or (None, None) if insufficient data."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)

    if len(X_train) < 100:
        return None, None

    scaler = _StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(np.asarray(y_train, dtype=float), dtype=torch.float32)

    model = _build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.MSELoss()
    model.train()
    n = Xt.shape[0]
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = lossf(pred, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model, scaler


def predict_panel(model, scaler, X_test):
    """Predict expected returns for the target panel = FT-Transformer(scaled X_test).
    ANTI-LEAKAGE: scaler fit on training rows only; target panel forward returns never used."""
    import torch
    model.eval()
    Xs = scaler.transform(X_test)
    with torch.no_grad():
        pred = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    return pred  # [N]


def evaluate_horizon(cur, panels, horizon_days, universe, label, seed, epochs):
    logger.info(f"\n{'='*100}\nHorizon: {label}({horizon_days}d)\n{'='*100}")
    panel_data = {}
    t0 = time.monotonic()
    for fs_id, as_of in panels:
        X, sids = load_features(cur, fs_id, universe)
        if not X: continue
        returns, label_date = load_forward_returns(cur, as_of, horizon_days)
        if not returns: continue
        XX, yy, sids_matched = [], [], []
        for i, sid in enumerate(sids):
            if sid in returns:
                XX.append(X[i]); yy.append(returns[sid]); sids_matched.append(sid)
        if XX:
            panel_data[as_of] = (XX, yy, sids_matched, label_date)
    logger.info(f"  Loaded {len(panel_data)} panels({horizon_days}d forward / load: {time.monotonic()-t0:.1f}s)")

    panel_keys = sorted(panel_data.keys())
    panel_pa = []  # §14.7-DF: (pred, actual) per panel → 共用 helper(單一來源)

    for i in range(1, len(panel_keys)):
        test_key = panel_keys[i]
        X_test, y_test, _, _ = panel_data[test_key]

        # pooled training rows = all (X, forward-return) from panels [0..i-1].
        train_X, train_y = [], []
        for j in range(i):
            X_j, y_j, _, _ = panel_data[panel_keys[j]]
            train_X.extend(X_j); train_y.extend(y_j)
        X_tr = np.array(train_X); y_tr = winsorize(np.array(train_y))
        if len(X_tr) < 100: continue

        model, scaler = train_ftt(X_tr, y_tr, seed, epochs)
        if model is None:
            continue

        pred_te = predict_panel(model, scaler, np.array(X_test))
        panel_pa.append((pred_te, y_test))

    result = summarize_horizon_metrics(label, horizon_days, panel_pa)  # §14.7-DF Canonical Metric SSOT(單一來源)
    if result is None:
        return None

    logger.info(f"  {label}({horizon_days}d): Sharpe {result['sharpe']:+.3f} | Eff t {result['effective_t_stat']:+.3f} | Win {result['win_rate']*100:.1f}% | IC {result['mean_ic']:+.4f} | NET {result['annualized_simple_net']*100:+.1f}%/yr")
    logger.info(f"    precision: hit {result['precision_directional_hit_rate']*100:.1f}% | top-20 overlap {result['precision_top20_actual_overlap']*100:.1f}% | RMSE {result['precision_rmse']:.4f} | reliability IC-CoV {result['reliability_ic_stability_cov']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=f"Multi-Cycle FT-Transformer Validation {TOOL_VER}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--commit", action="store_true")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"end-to-end training epochs (default {EPOCHS}, CPU)")
    parser.add_argument("--seed", type=int, default=5422,
                        help="§一.10 #3 multi-run seed(canonical 5422;≥3 distinct seeds → min/median/max/mean)")
    args = parser.parse_args()
    if not args.dry_run and not args.commit: args.dry_run = True

    horizon_days_list = [int(d.strip()) for d in args.horizons.split(",")]
    horizon_labels = []
    for d in horizon_days_list:
        if d <= 7: horizon_labels.append(("weekly", d))
        elif d <= 30: horizon_labels.append(("monthly", d))
        elif d <= 90: horizon_labels.append(("quarterly", d))
        else: horizon_labels.append(("annual", d))

    import torch
    logger.info("="*100)
    logger.info(f"Multi-Cycle FT-Transformer Validation {TOOL_VER}(Gorishniy et al. 2021)")
    logger.info("="*100)
    logger.info(f"  Horizons: {horizon_labels}")
    logger.info(f"  torch ver: {torch.__version__} (device=cpu)")
    logger.info(f"  FT-Transformer: d_token={D_TOKEN} n_layers={N_LAYERS} n_heads={N_HEADS} ffn_factor={FFN_FACTOR} dropout={DROPOUT} epochs={args.epochs} batch={BATCH_SIZE} lr={LR} seed={args.seed}")
    logger.info(f"  Mode:     {'COMMIT' if args.commit else 'DRY-RUN'}")

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("""SELECT m.stock_id FROM core_universe_membership m
                       JOIN core_universe_snapshot s ON s.snapshot_id=m.snapshot_id
                       WHERE s.status='committed' AND m.core_tier='core_universe'
                       AND s.snapshot_id=(SELECT snapshot_id FROM core_universe_snapshot
                                          WHERE status='committed' ORDER BY created_at DESC LIMIT 1)""")
        universe = list({r[0] for r in cur.fetchall()})
        logger.info(f"  Universe: {len(universe)} stocks")

        panels = get_canonical_panel_dates("feature_set_v0.6")  # §14.7-DE / §0.0-I 單一引用源
        logger.info(f"  Panels:   {len(panels)} ({panels[0][1]} ~ {panels[-1][1]}, data-driven §14.7-DE)")

        results = {}
        t_global = time.monotonic()
        for label, days in horizon_labels:
            r = evaluate_horizon(cur, panels, days, universe, label, args.seed, args.epochs)
            if r: results[label] = r

        logger.info(f"\n{'='*100}\nCross-Cycle Comparison Matrix(FT-Transformer)\n{'='*100}")
        logger.info(f"  {'Horizon':10} {'N':>4} {'n_eff':>6} {'Eff t':>7} {'Sig?':>5} {'Sharpe':>7} {'Win':>6} {'NetAnn':>9}")
        for label, r in results.items():
            sig = "✅" if r["is_significant_p05"] else "❌"
            logger.info(f"  {label:10} {r['n_panels']:>4} {r['n_effective']:>6.1f} {r['effective_t_stat']:>+7.3f} {sig:>5} {r['sharpe']:>+7.3f} {r['win_rate']*100:>5.1f}% {r['annualized_simple_net']*100:>+8.2f}%")

        logger.info(f"\n  Total elapsed: {time.monotonic()-t_global:.1f}s")

        if args.output or args.commit:
            output_path = args.output or f"reports/multi_cycle_ftransformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_full = Path(_base_dir).parent / output_path
            output_full.parent.mkdir(parents=True, exist_ok=True)
            output_results = {label: {k: v for k, v in r.items()} for label, r in results.items()}
            output_results["_meta"] = {
                "tool": "multi_cycle_ftransformer_validation.py",
                "tool_ver": TOOL_VER,
                "model_family": "ft_transformer",
                "model": "FT-Transformer (Gorishniy et al. 2021, from-scratch torch.nn, CPU)",
                "torch_version": torch.__version__,
                "run_at": datetime.now().isoformat(),
                "constitution_ver": CONSTITUTION_VER,
                "seed": args.seed,
                "d_token": D_TOKEN,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "ffn_factor": FFN_FACTOR,
                "dropout": DROPOUT,
                "epochs": args.epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "horizons": horizon_days_list,
                "n_universe": len(universe),
                "n_panels_input": len(panels),
                "source_traceability": "per CLAUDE.md §一.10 — all (b) DB query (feature_values is_null_imputed IS NOT TRUE + TaiwanStockPriceAdj)",
            }
            with open(output_full, "w") as f:
                json.dump(output_results, f, indent=2, default=str)
            logger.info(f"  Results persisted: {output_full}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
