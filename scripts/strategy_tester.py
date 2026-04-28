"""
strategy_tester.py — 物理特徵策略統一測試框架（[P3 修復 2.11]）

合併 asymmetric_simulator.py 與 singularity_layout_simulator.py 為單一框架。
原兩個模組差異僅在「閾值計算方式」（固定 vs 動態），且重複了讀資料 → 建倉
→ 出場 → 計算報酬的全部流程。本框架以策略物件抽象化兩者：

  PhysicsStrategy(entry_method=..., exit_method=...)

執行：
    # 動態閾值（取代 asymmetric_simulator.py）
    python strategy_tester.py --stock-id 2330 --entry dynamic

    # 固定閾值（取代 singularity_layout_simulator.py）
    python strategy_tester.py --stock-id 2330 --entry fixed --pull-thresh -10

    # 批量跑第六波賽道：
    python strategy_tester.py --batch sixth_wave
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import build_daily_frame
from feature_engineering import build_features
from config import FRICTION_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 策略參數
# ─────────────────────────────────────────────


@dataclass
class StrategyParams:
    """策略可調參數，所有預設值對齊原兩支腳本的「實際使用值」。"""
    entry_method: str = "dynamic"          # "dynamic" | "fixed"
    exit_method:  str = "gravity_zero"     # "gravity_zero" | "days_held" | "stop_loss_take_profit"
    # 動態閾值：mean ± std_mult × std
    pull_std_mult:  float = 1.0
    force_std_mult: float = 1.0
    # 固定閾值（fixed entry 時使用；對齊 singularity_layout_simulator.py）
    pull_thresh_fixed:  float = -10.0
    force_thresh_fixed: float = 0.0
    # 出場：days_held / stop_loss_take_profit 共用
    max_days_held:    int   = 30
    stop_loss_pct:    float = 0.10
    take_profit_pct:  float = 0.20
    # 倉位
    position_pct:    float = 0.20          # 每次入場用 20% 資金
    # 起始觀察窗（以避免 rolling-stat warmup 期）
    warmup_days:     int   = 126


# ─────────────────────────────────────────────
# 進場條件 — 各種 entry_method 的實作
# ─────────────────────────────────────────────


def _dynamic_entry(row: pd.Series, df: pd.DataFrame, params: StrategyParams) -> bool:
    """
    重力拉力低於 mean - std_mult×std 且資訊衝擊高於 mean + std_mult×std 即進場。
    對齊 asymmetric_simulator.py 的「寬鬆版物理套利」邏輯。
    """
    pull_mean,  pull_std  = df["gravity_pull"].mean(),         df["gravity_pull"].std()
    force_mean, force_std = df["info_force_per_mass"].mean(),  df["info_force_per_mass"].std()
    pull_th  = pull_mean  - params.pull_std_mult  * pull_std
    force_th = force_mean + params.force_std_mult * force_std
    return bool(row["gravity_pull"] < pull_th and row["info_force_per_mass"] > force_th)


def _fixed_entry(row: pd.Series, df: pd.DataFrame, params: StrategyParams) -> bool:
    """
    對齊 singularity_layout_simulator.py：用絕對閾值（-10/0）。
    """
    return bool(row["gravity_pull"] < params.pull_thresh_fixed
                and row["info_force_per_mass"] > params.force_thresh_fixed)


ENTRY_REGISTRY: dict[str, Callable[[pd.Series, pd.DataFrame, StrategyParams], bool]] = {
    "dynamic": _dynamic_entry,
    "fixed":   _fixed_entry,
}


# ─────────────────────────────────────────────
# 出場條件
# ─────────────────────────────────────────────


def _gravity_zero_exit(row: pd.Series, ctx: dict, params: StrategyParams) -> bool:
    """重力拉力回到 ≥ 0 即出場（對齊原兩支腳本）。"""
    return bool(row["gravity_pull"] > 0)


def _days_held_exit(row: pd.Series, ctx: dict, params: StrategyParams) -> bool:
    """達到最大持有天數即出場。"""
    return ctx["days_held"] >= params.max_days_held


def _sltp_exit(row: pd.Series, ctx: dict, params: StrategyParams) -> bool:
    """停損 / 停利 + 重力歸零三條件取或。"""
    pnl_pct = row["close"] / ctx["entry_price"] - 1
    if pnl_pct <= -params.stop_loss_pct:
        return True
    if pnl_pct >= params.take_profit_pct:
        return True
    return _gravity_zero_exit(row, ctx, params)


EXIT_REGISTRY: dict[str, Callable[[pd.Series, dict, StrategyParams], bool]] = {
    "gravity_zero":          _gravity_zero_exit,
    "days_held":             _days_held_exit,
    "stop_loss_take_profit": _sltp_exit,
}


# ─────────────────────────────────────────────
# 主策略類別
# ─────────────────────────────────────────────


@dataclass
class TradeRecord:
    entry_date:  pd.Timestamp
    exit_date:   Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price:  float = 0.0
    gross_ret:   float = 0.0
    net_ret:     float = 0.0


@dataclass
class PhysicsStrategy:
    """物理特徵驅動的進出場策略（取代 asymmetric / singularity 兩個模組）。"""
    stock_id:        str
    initial_capital: float = 1_000_000.0
    params:          StrategyParams = field(default_factory=StrategyParams)

    # ─── 成本（賣出含完整手續費 + 證交稅，買入只含手續費）───
    @property
    def buy_cost_rate(self) -> float:
        return 1.0 + FRICTION_CONFIG["commission"]

    @property
    def sell_cost_rate(self) -> float:
        return 1.0 - (FRICTION_CONFIG["commission"] + FRICTION_CONFIG["securities_tax"])

    @property
    def round_trip_cost(self) -> float:
        return FRICTION_CONFIG["commission"] * 2 + FRICTION_CONFIG["securities_tax"]

    # ─── 主要回測流程 ───
    def run(self, start_date: str = "2022-01-01") -> dict:
        logger.info(f"=== 物理策略回測：{self.stock_id} "
                    f"(entry={self.params.entry_method}, exit={self.params.exit_method}) ===")

        raw = build_daily_frame(stock_id=self.stock_id, start_date=start_date)
        df = build_features(raw, stock_id=self.stock_id, for_inference=True)
        if not {"gravity_pull", "info_force_per_mass"}.issubset(df.columns):
            raise RuntimeError("缺少必要的物理特徵欄位（gravity_pull / info_force_per_mass）")

        entry_fn = ENTRY_REGISTRY[self.params.entry_method]
        exit_fn = EXIT_REGISTRY[self.params.exit_method]

        capital = self.initial_capital
        position_shares = 0.0
        equity_curve: list[float] = []
        trades: list[TradeRecord] = []
        ctx: dict = {}

        for i in range(self.params.warmup_days, len(df)):
            row = df.iloc[i]
            date = df.index[i]
            price = float(row["close"])

            # ── 出場 ──
            if position_shares > 0:
                ctx["days_held"] = (date - ctx["entry_date"]).days
                if exit_fn(row, ctx, self.params):
                    capital = position_shares * price * self.sell_cost_rate
                    gross = price / ctx["entry_price"] - 1
                    net = gross - self.round_trip_cost
                    trades[-1].exit_date = date
                    trades[-1].exit_price = price
                    trades[-1].gross_ret = gross
                    trades[-1].net_ret = net
                    position_shares = 0.0
                    ctx = {}
                    logger.info(f"[{date.date()}] EXIT  price={price:.1f} gross={gross:.2%} net={net:.2%}")

            # ── 進場 ──
            if position_shares == 0 and entry_fn(row, df, self.params):
                deploy = capital * self.params.position_pct
                position_shares = deploy / (price * self.buy_cost_rate)
                capital -= deploy
                ctx = {"entry_date": date, "entry_price": price, "days_held": 0}
                trades.append(TradeRecord(entry_date=date, entry_price=price))
                logger.info(f"[{date.date()}] ENTRY price={price:.1f} pull={row['gravity_pull']:.2f}")

            equity_curve.append(capital + position_shares * price)

        final_equity = equity_curve[-1] if equity_curve else self.initial_capital
        completed = [t for t in trades if t.exit_date is not None]
        win_rate = (sum(1 for t in completed if t.net_ret > 0) / len(completed)) if completed else 0.0

        eq = pd.Series(equity_curve)
        max_dd = float((eq / eq.cummax() - 1).min()) if len(eq) > 0 else 0.0

        report = {
            "stock_id":      self.stock_id,
            "entry_method":  self.params.entry_method,
            "exit_method":   self.params.exit_method,
            "total_return":  final_equity / self.initial_capital - 1,
            "max_drawdown":  max_dd,
            "n_trades":      len(completed),
            "win_rate":      win_rate,
            "final_equity":  final_equity,
            "trades":        completed,
        }
        self._print_report(report)
        return report

    @staticmethod
    def _print_report(r: dict) -> None:
        print("\n" + "=" * 60)
        print(f" Physics Strategy Report — {r['stock_id']}")
        print(f"  entry: {r['entry_method']}    exit: {r['exit_method']}")
        print("-" * 60)
        print(f"  Total Return : {r['total_return']:.2%}")
        print(f"  Max Drawdown : {r['max_drawdown']:.2%}")
        print(f"  Trades       : {r['n_trades']}")
        print(f"  Win Rate     : {r['win_rate']:.2%}")
        print(f"  Final Equity : {r['final_equity']:,.0f}")
        print("=" * 60)


# ─────────────────────────────────────────────
# 預設批次：第六波賽道
# ─────────────────────────────────────────────


SIXTH_WAVE_DEFAULT = ["2330", "2308", "2454", "3661"]


def _run_batch(stock_ids: list[str], params: StrategyParams,
               start_date: str = "2022-01-01") -> pd.DataFrame:
    rows = []
    for sid in stock_ids:
        try:
            r = PhysicsStrategy(stock_id=sid, params=params).run(start_date=start_date)
            rows.append({k: v for k, v in r.items() if k != "trades"})
        except Exception as e:
            logger.error(f"  {sid} 回測失敗：{e}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="物理特徵策略統一測試框架")
    p.add_argument("--stock-id", default="2330")
    p.add_argument("--entry", choices=list(ENTRY_REGISTRY.keys()), default="dynamic")
    p.add_argument("--exit",  choices=list(EXIT_REGISTRY.keys()),  default="gravity_zero")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--pull-thresh", type=float, default=-10.0,
                   help="fixed-entry 模式的重力拉力門檻（預設 -10.0）")
    p.add_argument("--batch", choices=["", "sixth_wave"], default="",
                   help="批量模式：sixth_wave = 2330/2308/2454/3661")
    return p.parse_args()


def main():
    args = parse_args()
    params = StrategyParams(
        entry_method      = args.entry,
        exit_method       = args.exit,
        pull_thresh_fixed = args.pull_thresh,
    )
    if args.batch == "sixth_wave":
        report_df = _run_batch(SIXTH_WAVE_DEFAULT, params, start_date=args.start)
        print("\n=== Batch summary ===")
        print(report_df.to_string(index=False))
    else:
        PhysicsStrategy(stock_id=args.stock_id, params=params).run(start_date=args.start)


if __name__ == "__main__":
    main()
