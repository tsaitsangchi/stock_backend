from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.path_setup  # noqa: F401

"""
compute_stock_dynamics.py — 計算 stock_dynamics_registry (P0-4)
=====================================================================
從歷史價量 / 籌碼 / 基本面資料計算 7 個 dynamics 因子並寫入
stock_dynamics_registry。

signal_filter._load_dynamics_registry 會讀取此表，過去因為
「沒有任何 INSERT」導致全程靜默 fallback、動態門檻邏輯形同虛設。

7 個因子定義：
  1) info_sensitivity      ─ 對外資/法人籌碼變化的敏感度（β to chip flow）
  2) gravity_elasticity    ─ 偏離均線後的回歸速度（半衰期反推）
  3) fat_tail_index        ─ 報酬尖峰厚尾程度（kurtosis - 3）
  4) convexity_score       ─ 上漲/下跌不對稱性（右尾 / 左尾比）
  5) tail_risk_score       ─ 5% VaR 的尾部風險評分（負值越大風險越大）
  6) wave_track            ─ 主題賽道分類（依產業 + 動量）
  7) innovation_velocity   ─ 創新速度，由「20 日報酬 vs 大盤」估算

每個股都會更新一筆，PRIMARY KEY (stock_id) 採 UPSERT。

執行：
    python compute_stock_dynamics.py                 # 全部標的
    python compute_stock_dynamics.py --stock-id 2330
    python compute_stock_dynamics.py --window 504    # 用 2 年資料
    python compute_stock_dynamics.py --dry-run       # 不寫 DB
"""

import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import STOCK_CONFIGS, SECTOR_POOLS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────
DDL_REGISTRY = """
CREATE TABLE IF NOT EXISTS stock_dynamics_registry (
    stock_id              VARCHAR(50) PRIMARY KEY,
    info_sensitivity      NUMERIC(8,4),
    gravity_elasticity    NUMERIC(8,4),
    fat_tail_index        NUMERIC(8,4),
    convexity_score       NUMERIC(8,4),
    tail_risk_score       NUMERIC(8,4),
    wave_track            VARCHAR(50),
    innovation_velocity   NUMERIC(8,4),
    sample_size           INTEGER,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_dynamics_track ON stock_dynamics_registry (wave_track);
"""

UPSERT_SQL = """
INSERT INTO stock_dynamics_registry (
    stock_id, info_sensitivity, gravity_elasticity, fat_tail_index,
    convexity_score, tail_risk_score, wave_track, innovation_velocity,
    sample_size, updated_at
) VALUES (
    %(stock_id)s, %(info_sensitivity)s, %(gravity_elasticity)s, %(fat_tail_index)s,
    %(convexity_score)s, %(tail_risk_score)s, %(wave_track)s, %(innovation_velocity)s,
    %(sample_size)s, CURRENT_TIMESTAMP
)
ON CONFLICT (stock_id) DO UPDATE SET
    info_sensitivity    = EXCLUDED.info_sensitivity,
    gravity_elasticity  = EXCLUDED.gravity_elasticity,
    fat_tail_index      = EXCLUDED.fat_tail_index,
    convexity_score     = EXCLUDED.convexity_score,
    tail_risk_score     = EXCLUDED.tail_risk_score,
    wave_track          = EXCLUDED.wave_track,
    innovation_velocity = EXCLUDED.innovation_velocity,
    sample_size         = EXCLUDED.sample_size,
    updated_at          = CURRENT_TIMESTAMP
"""


# ─────────────────────────────────────────────
# 資料載入
# ─────────────────────────────────────────────

def _load_price(stock_id: str, window: int) -> pd.DataFrame:
    """從 stock_price 取最近 window 個交易日的收盤價、成交量。"""
    from data_pipeline import _query
    sql = """
        SELECT date, close::float, trading_volume::bigint
        FROM stock_price
        WHERE stock_id = %s
        ORDER BY date DESC
        LIMIT %s
    """
    df = _query(sql, (stock_id, window))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ret_1d"] = df["close"].pct_change()
    return df


def _load_chip_flow(stock_id: str, window: int) -> Optional[pd.Series]:
    """法人 net buy / volume 比例，作為 info flow proxy。"""
    from data_pipeline import _query
    sql = """
        SELECT date,
               (COALESCE(foreign_investor_buy, 0) - COALESCE(foreign_investor_sell, 0))::float
                  AS foreign_net
        FROM institutional_investors_buy_sell
        WHERE stock_id = %s
        ORDER BY date DESC
        LIMIT %s
    """
    try:
        df = _query(sql, (stock_id, window))
    except Exception as e:
        logger.debug(f"[{stock_id}] chip flow load 失敗：{e}")
        return None
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")["foreign_net"]
    return df


def _load_taiex(window: int) -> Optional[pd.Series]:
    """大盤 TAIEX 收盤，用於 innovation_velocity 比較基準。"""
    from data_pipeline import _query
    for sql, params in [
        ("SELECT date, close::float FROM stock_price WHERE stock_id = 'TAIEX' "
         "ORDER BY date DESC LIMIT %s", (window,)),
        ("SELECT date, close::float FROM total_return_index ORDER BY date DESC LIMIT %s",
         (window,)),
    ]:
        try:
            df = _query(sql, params)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").set_index("date")["close"]
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────
# 7 個 dynamics 因子計算
# ─────────────────────────────────────────────

def compute_info_sensitivity(returns: pd.Series, chip: Optional[pd.Series]) -> float:
    """β_chip = cov(ret, chip_flow_zscore) / var(chip_flow_zscore)，無資料時回傳 0.5（中性）。"""
    if chip is None or chip.empty:
        return 0.5
    df = pd.concat([returns, chip], axis=1, join="inner").dropna()
    if len(df) < 60:
        return 0.5
    chip_z = (df.iloc[:, 1] - df.iloc[:, 1].rolling(60).mean()) / \
             (df.iloc[:, 1].rolling(60).std() + 1e-9)
    valid = pd.concat([df.iloc[:, 0], chip_z], axis=1).dropna()
    if len(valid) < 30:
        return 0.5
    cov = valid.iloc[:, 0].cov(valid.iloc[:, 1])
    var = valid.iloc[:, 1].var()
    if var <= 0:
        return 0.5
    beta = cov / var
    # 把 beta 壓到 [0, 1]：用 tanh，0 表完全不敏感
    return float(np.clip(0.5 + np.tanh(beta * 50), 0.0, 1.0))


def compute_gravity_elasticity(close: pd.Series) -> float:
    """
    偏離 60 日均線後的回歸速度，用半衰期推算 elasticity = ln(2) / half_life。
    half_life 透過 AR(1) 估計：ret = ρ * ret_lag1，half_life = ln(0.5) / ln(|ρ|)
    """
    if len(close) < 100:
        return 0.05
    ma60 = close.rolling(60, min_periods=30).mean()
    dev = (close - ma60) / (ma60 + 1e-9)
    dev = dev.dropna()
    if len(dev) < 60:
        return 0.05
    rho = dev.autocorr(lag=1)
    if rho is None or rho <= 0 or rho >= 1:
        return 0.05
    half_life = np.log(0.5) / np.log(abs(rho))
    if half_life <= 0:
        return 0.05
    elasticity = np.log(2) / half_life
    return float(np.clip(elasticity, 0.001, 1.0))


def compute_fat_tail_index(returns: pd.Series) -> float:
    """超額峰度（kurt - 3）。常態 = 0；> 3 屬重尾。"""
    r = returns.dropna()
    if len(r) < 60:
        return 3.0
    return float(np.clip(r.kurtosis(), -2.0, 30.0))


def compute_convexity_score(returns: pd.Series) -> float:
    """
    凸性：右尾報酬 / 左尾報酬 比。
      score = mean(top_5%) / |mean(bottom_5%)|
    > 1 表正凸（願意買 OTM call 的標的）。
    """
    r = returns.dropna()
    if len(r) < 100:
        return 0.0
    upper = r.quantile(0.95)
    lower = r.quantile(0.05)
    top_avg = r[r >= upper].mean()
    bot_avg = r[r <= lower].mean()
    if bot_avg == 0 or pd.isna(bot_avg):
        return 0.0
    score = top_avg / abs(bot_avg)
    return float(np.clip(score - 1.0, -3.0, 3.0))


def compute_tail_risk_score(returns: pd.Series) -> float:
    """5% VaR + Expected Shortfall，回傳負值；越負代表風險越大。"""
    r = returns.dropna()
    if len(r) < 60:
        return -1.0
    var5 = r.quantile(0.05)
    es5 = r[r <= var5].mean()
    if pd.isna(es5):
        return -1.0
    return float(np.clip(es5 * 100, -20.0, 0.0))  # 轉百分點，截斷至 [-20%, 0%]


def compute_wave_track(stock_id: str, returns: pd.Series, taiex_ret: Optional[pd.Series]) -> str:
    """
    依「所屬產業 + 是否跑贏大盤 60 日」分類：
      AI / Semi 領頭 + 跑贏大盤 → 'AI_INNOVATION_WAVE'
      Finance + 穩定 → 'INCOME_TRACK'
      其他 → 'STRAT_SECTOR'  (有跑贏大盤者) 或 'LEGACY_IT'
    """
    cfg = STOCK_CONFIGS.get(stock_id, {})
    industry = cfg.get("industry", "Unknown")

    outperform = False
    if taiex_ret is not None and len(taiex_ret) >= 60:
        try:
            r60 = returns.tail(60).sum()
            t60 = taiex_ret.tail(60).sum()
            outperform = bool(r60 > t60)
        except Exception:
            outperform = False

    if industry in ("Semiconductor", "AI_Hardware") and outperform:
        return "AI_INNOVATION_WAVE"
    if industry == "Finance":
        return "INCOME_TRACK"
    if outperform:
        return "STRAT_SECTOR"
    return "LEGACY_IT"


def compute_innovation_velocity(returns: pd.Series, taiex_ret: Optional[pd.Series]) -> float:
    """
    過去 60 日累積超額報酬（vs TAIEX），轉成 [0.5, 2.0] 倍數。
    """
    if len(returns) < 60:
        return 1.0
    r60 = returns.tail(60).sum()
    if taiex_ret is None or len(taiex_ret) < 60:
        return float(np.clip(1.0 + r60 * 5, 0.5, 2.0))
    t60 = taiex_ret.tail(60).sum()
    excess = r60 - t60
    return float(np.clip(1.0 + excess * 8, 0.5, 2.5))


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def compute_for_stock(stock_id: str, window: int = 504) -> Optional[Dict]:
    """計算單一個股的 dynamics dict。"""
    df = _load_price(stock_id, window)
    if df.empty or len(df) < 60:
        logger.warning(f"[{stock_id}] 資料不足 ({len(df)} 筆)，跳過")
        return None

    close = df.set_index("date")["close"]
    returns = df.set_index("date")["ret_1d"].dropna()

    chip = _load_chip_flow(stock_id, window)
    taiex = _load_taiex(window)
    taiex_ret = taiex.pct_change().dropna() if taiex is not None else None

    out = {
        "stock_id":            stock_id,
        "info_sensitivity":    round(compute_info_sensitivity(returns, chip), 4),
        "gravity_elasticity":  round(compute_gravity_elasticity(close), 4),
        "fat_tail_index":      round(compute_fat_tail_index(returns), 4),
        "convexity_score":     round(compute_convexity_score(returns), 4),
        "tail_risk_score":     round(compute_tail_risk_score(returns), 4),
        "wave_track":          compute_wave_track(stock_id, returns, taiex_ret),
        "innovation_velocity": round(compute_innovation_velocity(returns, taiex_ret), 4),
        "sample_size":         int(len(returns)),
    }
    return out


def upsert_registry(records: list[Dict], dry_run: bool = False) -> int:
    if not records:
        return 0
    if dry_run:
        logger.info(f"[DRY-RUN] 跳過 DB 寫入，共 {len(records)} 筆")
        for r in records:
            logger.info(f"  {r}")
        return len(records)

    import psycopg2.extras
    from core.db_utils import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for stmt in [s.strip() for s in DDL_REGISTRY.split(";") if s.strip()]:
                cur.execute(stmt)
            psycopg2.extras.execute_batch(cur, UPSERT_SQL, records, page_size=200)
        conn.commit()
    finally:
        conn.close()
    return len(records)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    p = argparse.ArgumentParser(description="計算並寫入 stock_dynamics_registry")
    p.add_argument("--stock-id", default=None, help="只處理單一股票")
    p.add_argument("--window", type=int, default=504, help="使用最近 N 個交易日（預設 504 ≈ 2 年）")
    p.add_argument("--dry-run", action="store_true", help="試跑，不寫入 DB")
    args = p.parse_args()

    targets = [args.stock_id] if args.stock_id else list(STOCK_CONFIGS.keys())
    logger.info(f"準備計算 {len(targets)} 個股的 dynamics（window={args.window}）")

    records = []
    for sid in targets:
        try:
            rec = compute_for_stock(sid, window=args.window)
            if rec:
                logger.info(f"[{sid}] {rec['wave_track']:24s}  "
                            f"info={rec['info_sensitivity']:.2f}  "
                            f"elast={rec['gravity_elasticity']:.3f}  "
                            f"tail={rec['tail_risk_score']:.2f}  "
                            f"conv={rec['convexity_score']:+.2f}  "
                            f"v={rec['innovation_velocity']:.2f}")
                records.append(rec)
        except Exception as e:
            logger.error(f"[{sid}] 計算失敗：{e}", exc_info=True)

    n = upsert_registry(records, dry_run=args.dry_run)
    logger.info(f"\n✨ 寫入完成：{n}/{len(targets)} 筆")


if __name__ == "__main__":
    main()
