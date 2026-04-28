"""
signal_filter.py — 實務訊號過濾層
===================================
核心理念：模型負責「進場篩選」，不負責「精確預測價格」。
只在多個條件同時成立時才建議交易，其餘時間持現金等待。

五大過濾維度：
  ① 模型機率門檻   : prob_up > 0.65（訊號顯著高於隨機）
  ② 波動率 Regime  : low_vol / mid_vol（高波動不進場）
  ③ 趨勢 Regime    : bull / sideways（熊市不做多）
  ④ 機構籌碼確認   : 外資連續買超 N 週 + 買超加速度正向
  ⑤ 基本面動量     : 月營收連續 YoY 正成長 + 毛利率 QoQ 未惡化

停損停利設定：
  進場後設定 -5% 停損 / +12% 停利（非對稱 risk-reward = 1:2.4）

使用方式：
    from signal_filter import SignalFilter, FilterResult
    sf = SignalFilter()
    result = sf.evaluate(report, df_feat)
    print(result.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import CONFIDENCE_THRESHOLD, DB_CONFIG
from utils.db import get_db_connection

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 過濾條件設定
# ─────────────────────────────────────────────

FILTER_CONFIG = {
    # ① 模型機率門檻
    "prob_up_threshold":     0.65,   # 上漲機率 > 65% 才考慮進場
    "prob_down_threshold":   0.35,   # 下跌機率 < 35%（做多時）

    # ② 模型一致性門檻
    "min_model_agreement":   0.60,   # XGB/LGB/TFT 方向一致性

    # ③ 波動率 Regime 允許進場的等級
    "allowed_vol_regimes":   {"low_vol", "mid_vol"},   # 高波動不進場

    # ④ 趨勢 Regime 允許做多的等級
    "allowed_trend_for_long": {"bull", "sideways"},   # 熊市不做多

    # ⑤ 外資籌碼門檻
    "min_foreign_net_weekly":  0,      # 外資近週淨買超 ≥ 0（不能淨賣）
    "min_foreign_net_accel":  -1e9,   # 外資買超加速度（可為負，但不能崩潰）
    "min_foreign_bullish_months": 0,  # 外資連續買超月數（放寬：不強制要求）

    # ⑥ 基本面動量門檻
    "min_rev_yoy_positive_months": 0, # 連續 YoY 正成長月數（可為 0，資料補齊後調高）
    "min_rev_yoy_3m":         -0.30,  # 近 3 月 YoY 均值 ≥ -30%（避免急跌）
    "gross_margin_qoq_min":   -0.10,  # 毛利率 QoQ 降幅不超過 -10%

    # ⑦ 宏觀護欄
    "block_on_macro_shock":   True,   # 宏觀衝擊時強制不交易

    # ⑧ 停損停利（建議值，供 report 顯示）
    "stop_loss_pct":   -0.05,         # -5% 停損
    "take_profit_pct": +0.12,         # +12% 停利
}

# 各維度的權重（用於綜合評分，0~100）
FILTER_WEIGHTS = {
    "prob":            35,   # 模型機率（最重要）
    "regime":          20,   # Regime 適合度
    "chip":            20,   # 機構籌碼
    "fundamental":     15,   # 基本面動量
    "sentiment_macro": 10,   # 恐懼貪婪與大額選擇權
}


# ─────────────────────────────────────────────
# 過濾結果資料類別
# ─────────────────────────────────────────────

@dataclass
class FilterDimension:
    name:     str
    passed:   bool
    score:    float          # 0.0~1.0（此維度的得分）
    detail:   str            # 說明文字
    value:    float = 0.0    # 實際數值


@dataclass
class FilterResult:
    """
    訊號過濾完整結果。

    decision:
      "LONG"      → 建議做多（所有必要條件通過）
      "HOLD_CASH" → 建議持現金（條件不足）
      "WATCH"     → 接近門檻，持續觀察
    """
    decision:      str                          # LONG / HOLD_CASH / WATCH
    overall_score: float                        # 0~100 的綜合評分
    dimensions:    list[FilterDimension] = field(default_factory=list)
    stop_loss:     float = FILTER_CONFIG["stop_loss_pct"]
    take_profit:   float = FILTER_CONFIG["take_profit_pct"]
    blocking_reasons: list[str] = field(default_factory=list)
    boosting_reasons: list[str] = field(default_factory=list)

    @property
    def is_tradeable(self) -> bool:
        return self.decision == "LONG"

    def summary(self) -> str:
        icon = {"LONG": "🟢", "HOLD_CASH": "🔴", "WATCH": "🟡"}.get(self.decision, "⚪")
        lines = [
            "═" * 55,
            f"  訊號過濾結果：{icon} {self.decision}  （綜合評分：{self.overall_score:.0f}/100）",
            "─" * 55,
        ]
        for dim in self.dimensions:
            mark = "✅" if dim.passed else "❌"
            lines.append(f"  {mark} {dim.name:20s} {dim.detail}")

        if self.blocking_reasons:
            lines.append("\n  🚫 阻斷條件：")
            for r in self.blocking_reasons:
                lines.append(f"    ・{r}")
        if self.boosting_reasons:
            lines.append("\n  ⭐ 強化條件：")
            for r in self.boosting_reasons:
                lines.append(f"    ・{r}")

        if self.decision == "LONG":
            lines.append(f"\n  建議停損：{self.stop_loss*100:.0f}%  |  建議停利：{self.take_profit*100:.0f}%")
            rr = abs(self.take_profit / self.stop_loss)
            lines.append(f"  風險報酬比：1 : {rr:.1f}")
        elif self.decision == "WATCH":
            lines.append("\n  📋 建議：接近進場門檻，明日繼續觀察")
        else:
            lines.append("\n  📋 建議：持現金等待更強訊號出現")

        lines.append("═" * 55)
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 主過濾類別
# ─────────────────────────────────────────────

class SignalFilter:
    """
    多維度訊號過濾器。

    設計原則：
      - Hard Block: 宏觀衝擊、高波動熊市 → 直接阻斷，不論機率多高
      - Soft Score: 各維度評分加權，綜合分 ≥ 60 才觸發 LONG
      - Watch Zone: 綜合分 45~60 → WATCH（接近門檻，持續觀察）
    """

    def __init__(self, config: dict | None = None):
        self.cfg = {**FILTER_CONFIG, **(config or {})}
        self.dynamics_registry = {} # 快取個股動力學參數


    def _load_dynamics_registry(self, stock_id: str):
        """ 從資料庫讀取個股動力學 DNA """
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT info_sensitivity, gravity_elasticity, fat_tail_index, convexity_score, tail_risk_score, wave_track, innovation_velocity FROM stock_dynamics_registry WHERE stock_id = %s", (stock_id,))
            res = cur.fetchone()
            cur.close()
            conn.close()
            if res:
                return {
                    "sensitivity": res[0] or 0.5,
                    "elasticity": res[1] or 0.05,
                    "tail": res[2] or 3.0,
                    "convexity": res[3] or 0.0,
                    "tail_risk": res[4] or 0.0,
                    "wave_track": res[5] or "LEGACY_IT",
                    "innovation_velocity": res[6] or 1.0
                }
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────
    # 維度評估函式
    # ─────────────────────────────────────────────

    def _eval_prob(self, report: dict) -> FilterDimension:
        prob_up     = report.get("prob_up", 0.5)
        agreement   = report.get("model_agreement", 0.0)
        
        # ── 2026 量子金融藍圖：動態門檻調整 ───────────────────
        # 根據康波分數調整基準門檻
        kwave_score = report.get("kwave_score", 0.0)
        entropy_delta = report.get("entropy_delta", 0.0)
        
        base_threshold = self.cfg["prob_up_threshold"]
        
        # 如果康波分數過高 (風險大) 或 熵值劇變 (不穩定)，門檻自動提高
        dynamic_threshold = base_threshold
        if kwave_score > 0.5: dynamic_threshold += 0.05
        if entropy_delta > 0.01: dynamic_threshold += 0.03
        
        min_agree    = self.cfg["min_model_agreement"]
        prob_passed  = prob_up >= dynamic_threshold
        agree_passed = agreement >= min_agree
        passed       = prob_passed and agree_passed

        # 評分：線性映射 [threshold, 1.0] → [0.5, 1.0]
        # 八二法則優化：如果進入擊球區 (prob_up >= CONFIDENCE_THRESHOLD)，直接給予滿分
        if prob_up >= CONFIDENCE_THRESHOLD:
            prob_score = 1.0
        else:
            prob_score = min(1.0, max(0.0, (prob_up - 0.5) / 0.5))
            
        agree_score = min(1.0, max(0.0, agreement))
        score       = (prob_score * 0.7 + agree_score * 0.3)

        detail = f"prob={prob_up:.2f} (動態門檻={dynamic_threshold:.2f})  agreement={agreement:.0%}"
        if prob_up >= CONFIDENCE_THRESHOLD:
            detail += " 🔥核心擊球區"
        if kwave_score < -0.5:
            detail += " 🚀康波順風"
            
        return FilterDimension("①動力學模型機率", passed, score, detail, prob_up)

    def _eval_regime(self, report: dict, df_feat: pd.DataFrame) -> FilterDimension:
        """② 波動率 + 趨勢 Regime 維度"""
        latest = df_feat.iloc[-1]

        # 波動率 Regime
        vol_20d = float(latest.get("realized_vol_20d", 0.25))
        if vol_20d < 0.20:
            vol_regime = "low_vol"
        elif vol_20d < 0.40:
            vol_regime = "mid_vol"
        else:
            vol_regime = "high_vol"

        # 趨勢 Regime（優先使用特徵工程已計算的欄位）
        trend_regime = str(latest.get("trend_regime", "sideways"))
        trend_int    = int(latest.get("trend_regime_int", 0))

        allowed_vol   = self.cfg["allowed_vol_regimes"]
        allowed_trend = self.cfg["allowed_trend_for_long"]

        vol_ok   = vol_regime in allowed_vol
        trend_ok = trend_regime in allowed_trend
        passed   = vol_ok and trend_ok

        # 評分
        vol_score   = 1.0 if vol_regime == "low_vol" else (0.6 if vol_regime == "mid_vol" else 0.0)
        trend_score = 1.0 if trend_regime == "bull" else (0.5 if trend_regime == "sideways" else 0.0)
        score       = vol_score * 0.5 + trend_score * 0.5

        detail = f"vol={vol_20d:.0%}({vol_regime})  trend={trend_regime}"
        return FilterDimension("②市場Regime", passed, score, detail, vol_20d)

    def _eval_chip(self, df_feat: pd.DataFrame) -> FilterDimension:
        """③ 機構籌碼維度"""
        latest = df_feat.iloc[-1]

        foreign_weekly = float(latest.get("foreign_net_weekly", 0))
        foreign_accel  = float(latest.get("foreign_net_accel", 0))
        # 修正：移除錯誤的基本面代理指標，改用籌碼面指標
        foreign_bullish_months = float(latest.get("foreign_bullish_months_proxy", 0)) 

        min_weekly = self.cfg["min_foreign_net_weekly"]
        weekly_ok  = foreign_weekly >= min_weekly
        accel_ok   = foreign_accel >= self.cfg["min_foreign_net_accel"]
        passed     = weekly_ok and accel_ok

        smart_sync = int(latest.get("smart_money_sync_buy", 0))

        # 評分：週淨買超正負 + 加速度方向 + 聰明錢同步
        weekly_score = 1.0 if foreign_weekly > 1e8 else (0.5 if foreign_weekly > 0 else 0.0)
        accel_score  = 1.0 if foreign_accel > 0 else 0.3
        score = (weekly_score * 0.5 + accel_score * 0.3 + smart_sync * 0.2)

        detail = (f"外資週淨買超={foreign_weekly/1e8:.1f}億  "
                  f"加速度={'↑' if foreign_accel > 0 else '↓'}")
        return FilterDimension("③機構籌碼", passed, score, detail, foreign_weekly)

    def _eval_fundamental(self, df_feat: pd.DataFrame) -> FilterDimension:
        """④ 基本面動量維度"""
        latest = df_feat.iloc[-1]

        rev_pos_months = float(latest.get("rev_yoy_positive_months", np.nan))
        rev_yoy_3m     = float(latest.get("rev_yoy_3m", np.nan))
        gm_qoq         = float(latest.get("gross_margin_qoq", np.nan))
        eps_accel      = float(latest.get("eps_accel_proxy", np.nan))

        min_months = self.cfg["min_rev_yoy_positive_months"]
        min_yoy3m  = self.cfg["min_rev_yoy_3m"]
        min_gm_qoq = self.cfg["gross_margin_qoq_min"]

        # 各子條件（NaN 視為通過，資料補齊前不阻斷）
        months_ok = np.isnan(rev_pos_months) or (rev_pos_months >= min_months)
        yoy3m_ok  = np.isnan(rev_yoy_3m)    or (rev_yoy_3m >= min_yoy3m)
        gm_ok     = np.isnan(gm_qoq)        or (gm_qoq >= min_gm_qoq)
        passed    = months_ok and yoy3m_ok and gm_ok

        # 評分
        months_score = min(1.0, max(0.0, rev_pos_months / 12)) if not np.isnan(rev_pos_months) else 0.5
        yoy_score    = min(1.0, max(0.0, (rev_yoy_3m + 0.3) / 0.6)) if not np.isnan(rev_yoy_3m) else 0.5
        gm_score     = 1.0 if (not np.isnan(gm_qoq) and gm_qoq >= 0) else (0.5 if np.isnan(gm_qoq) else 0.0)
        score = months_score * 0.3 + yoy_score * 0.4 + gm_score * 0.3

        rev_str = f"YoY連正{rev_pos_months:.0f}月" if not np.isnan(rev_pos_months) else "YoY=N/A"
        gm_str  = f"GM_QoQ={gm_qoq:+.1%}" if not np.isnan(gm_qoq) else "GM_QoQ=N/A"
        detail  = f"{rev_str}  {gm_str}"
        return FilterDimension("④基本面動量", passed, score, detail, rev_pos_months)

    def _eval_sentiment_macro(self, df_feat: pd.DataFrame) -> FilterDimension:
        """⑤ 情緒與宏觀維度"""
        latest = df_feat.iloc[-1]
        
        fg_score = float(latest.get("fear_greed_score", 50))
        pc_ratio = float(latest.get("put_call_large_ratio", 1.0))
        macro_color = str(latest.get("macro_monitoring_color", "N/A"))
        
        # 簡單評估：沒有極端貪婪，且 Put/Call 比率沒有極端看空 (<1.2)
        passed = (fg_score <= 75) and (pc_ratio < 1.2)
        
        # 評分
        score = 0.5
        if fg_score < 25: score += 0.3
        elif fg_score > 75: score -= 0.3
        
        if macro_color == 'blue': score += 0.2
        elif macro_color == 'red': score -= 0.2
        
        score = min(1.0, max(0.0, score))
        detail = f"FG={fg_score:.0f}  P/C={pc_ratio:.2f}  Macro={macro_color}"
        return FilterDimension("⑤情緒與宏觀", passed, score, detail, fg_score)

    # ─────────────────────────────────────────────
    # 主評估函式
    # ─────────────────────────────────────────────

    def evaluate(
        self,
        report:   dict,
        df_feat:  pd.DataFrame,
    ) -> FilterResult:
        """
        執行完整的多維度訊號過濾。
        """
        stock_id = report.get("stock_id", "2330")
        dynamics = self._load_dynamics_registry(stock_id)
        
        blocking_reasons = []
        boosting_reasons = []
        dimensions       = []
        
        # ── 動力學 DNA 注入 ───────────────────────────────────────
        if dynamics:
            # 根據敏感度調整門檻：敏感度越低，門檻越高
            sensitivity_bias = (0.5 - dynamics["sensitivity"]) * 0.1
            self.cfg["prob_up_threshold"] += max(-0.05, min(0.05, sensitivity_bias))
            boosting_reasons.append(f"🧬 已載入個股動態 DNA (Sensitivity={dynamics['sensitivity']:.2f})")

        # ── Hard Block 1：宏觀衝擊 ──────────────────────────────
        macro_shock = report.get("warnings", {}).get("macro_shock", False)
        if macro_shock and self.cfg["block_on_macro_shock"]:
            blocking_reasons.append("宏觀衝擊偵測（FED/匯率/指數急變），強制暫停交易")
            return FilterResult(
                decision="HOLD_CASH",
                overall_score=0.0,
                dimensions=[],
                blocking_reasons=blocking_reasons,
            )

        # ── Hard Block 2：高波動熊市 ─────────────────────────────
        latest = df_feat.iloc[-1]
        vol_20d      = float(latest.get("realized_vol_20d", 0.25))
        trend_regime = str(latest.get("trend_regime", "sideways"))
        if vol_20d >= 0.40 and trend_regime == "bear":
            blocking_reasons.append(f"高波動({vol_20d:.0%}) + 熊市，風險過高")
            return FilterResult(
                decision="HOLD_CASH",
                overall_score=5.0,
                dimensions=[],
                blocking_reasons=blocking_reasons,
            )

        # ── 各維度評估 ───────────────────────────────────────────
        dim_prob  = self._eval_prob(report)
        dim_regime = self._eval_regime(report, df_feat)
        dim_chip  = self._eval_chip(df_feat)
        dim_fund  = self._eval_fundamental(df_feat)
        dim_smacro = self._eval_sentiment_macro(df_feat)
        dimensions = [dim_prob, dim_regime, dim_chip, dim_fund, dim_smacro]

        # ── 加權綜合評分（0~100）────────────────────────────────
        w = FILTER_WEIGHTS
        overall = (
            dim_prob.score   * w["prob"]         +
            dim_regime.score * w["regime"]       +
            dim_chip.score   * w["chip"]         +
            dim_fund.score   * w["fundamental"]  +
            dim_smacro.score * w["sentiment_macro"]
        )

        # ── 阻斷與強化條件 ───────────────────────────────────────
        prob_up   = report.get("prob_up", 0.5)
        agreement = report.get("model_agreement", 0.0)

        if not dim_prob.passed:
            blocking_reasons.append(
                f"模型機率不足 ({prob_up:.2f} < {self.cfg['prob_up_threshold']})"
            )
        if agreement < self.cfg["min_model_agreement"]:
            blocking_reasons.append(
                f"模型一致性不足 ({agreement:.0%} < {self.cfg['min_model_agreement']:.0%})"
            )
        if not dim_regime.passed:
            blocking_reasons.append(f"市場 Regime 不適合進場 ({trend_regime} / vol={vol_20d:.0%})")

        # 懲罰條件
        is_extreme_greed = int(latest.get("is_extreme_greed", 0))
        if is_extreme_greed:
            blocking_reasons.append(f"極度貪婪 (Greed > 75) — 追高風險 (扣減綜合分數)")

        large_holder_change_3m = float(latest.get("large_holder_change_3m", 0))
        if large_holder_change_3m < -0.05:
            blocking_reasons.append(f"大戶籌碼顯著流失 ({large_holder_change_3m:+.1%}) — 建議觀望")

        # 強化條件（加分項）
        foreign_weekly = float(latest.get("foreign_net_weekly", 0))
        if foreign_weekly > 5e8:
            boosting_reasons.append(f"外資大量買超 ({foreign_weekly/1e8:.0f}億/週)")
        if prob_up >= CONFIDENCE_THRESHOLD:
            boosting_reasons.append(f"⭐ 核心擊球區：高勝率機會 (prob={prob_up:.1%})")
            
        if trend_regime == "bull" and vol_20d < 0.20:
            boosting_reasons.append("低波動牛市 — 最佳進場環境")
        eps_accel = float(latest.get("eps_accel_proxy", np.nan))
        if not np.isnan(eps_accel) and eps_accel > 0.1:
            boosting_reasons.append(f"EPS 加速成長 ({eps_accel:+.0%})")
            
        is_extreme_fear = int(latest.get("is_extreme_fear", 0))
        if is_extreme_fear:
            boosting_reasons.append("極度恐懼 (Fear < 25) — 逢低反向佈局時機")
            
        smart_money_sync = int(latest.get("smart_money_sync_buy", 0))
        if smart_money_sync:
            boosting_reasons.append("聰明錢護航 (外資與八大行庫同步買超)")
            
        # 🚀 動力學能量釋放 (Impulse/Energy Boost)
        kinetic_momentum = float(latest.get("kinetic_momentum", 0))
        if kinetic_momentum > 0:
            # 根據創新速度加乘
            velocity_multiplier = dynamics.get("innovation_velocity", 1.0) if dynamics else 1.0
            boosting_reasons.append(f"🌀 動力學動量正向 (Mass x Disp) — 動能釋放 (Velocity={velocity_multiplier:.2f})")
            overall += (3 * velocity_multiplier)
            
        # 🧪 技術奇點與結構性溢價 (Structural Premium)
        structural_premium = float(latest.get("structural_premium", 0))
        if structural_premium > 0.5 or (dynamics and dynamics.get("wave_track") != "LEGACY_IT"):
            track_name = dynamics.get("wave_track", "STRAT_SECTOR") if dynamics else "STRAT_SECTOR"
            boosting_reasons.append(f"✨ 結構性溢價：新興賽道 ({track_name}) 領導者效應")
            overall += 7
            
        # 🌌 重力井套利偵測 (Gravity Well Arbitrage)
        # 核心原則：偏離邊端時，引力最強，套利空間最大
        gravity_pull = float(latest.get("gravity_pull", 0))
        info_force = float(latest.get("info_force_per_mass", 0))
        
        # 情況 A：超跌引力反彈 (極端負偏離 + 正向資訊力注入)
        if gravity_pull < -0.1 and info_force > 0:
            boosting_reasons.append(f"🌌 重力井共振：價格處於超跌邊緣且資訊力注入 — 強引力回歸預期")
            overall += 10 # 提供重大加分
            
        # 情況 B：超漲重力警告 (極端正偏離 + 資訊力衰竭)
        if gravity_pull > 0.1 and info_force < 0:
            blocking_reasons.append(f"⚠️ 重力井預警：價格偏離重力中心過遠且動能衰竭 — 存在回歸壓力")
            overall -= 10

        # ── 最終決策 ─────────────────────────────────────────────
        # 必要條件：模型機率 + 波動率/趨勢 Regime 都必須通過
        must_pass = dim_prob.passed and dim_regime.passed
        
        # ── 80/20 風險掃描儀 (Risk Scanner) ────────────────────
        # 針對可能引發 80% 虧損的 20% 關鍵因子
        is_risk_zone = False
        
        # 1. 熵值激增 (Entropy Spike) -> 市場進入極度混亂
        entropy_delta = float(latest.get("entropy_delta", 0))
        if entropy_delta > 0.02:
            blocking_reasons.append(f"🔴 風險掃描：市場熵值激增 ({entropy_delta:+.3f})，系統進入不確定性紅區")
            is_risk_zone = True
            
        # 2. 康波風險 (K-Wave Risk)
        kwave_score = float(latest.get("kwave_score", 0))
        if kwave_score > 1.5:
            blocking_reasons.append(f"🔴 風險掃描：康波分數極高 ({kwave_score:.2f})，長波修正風險極大")
            is_risk_zone = True
            
        # 3. 資訊力崩潰 (Force Collapse)
        info_force = float(latest.get("total_info_force", 0))
        if info_force < -1.5:
            blocking_reasons.append(f"🔴 風險掃描：資訊力崩潰 ({info_force:.2f})，市場正遭受負向衝擊")
            is_risk_zone = True
            
        # 4. 2026 泡沫清算共振 (2026 Resonance Risk)
        bubble_risk = float(latest.get("bubble_crash_risk", 0))
        if bubble_risk > 0.7:
             blocking_reasons.append(f"💀 2026 共振預警：信用泡沫出清風險極大 (Risk={bubble_risk:.2f})")
             is_risk_zone = True

        # 如果處於風險紅區，即便機率高也強制轉為 WATCH 或 HOLD_CASH
        if is_risk_zone:
            must_pass = False

        # ── 80/20 戰略決策：淘汰平庸 ──────────────────────────
        # 核心原則：0% 隱藏危險區 (捨棄中等風險/報酬標的)
        if must_pass and overall >= 65:
            # 右側 20%：極端正向尾部
            if dynamics and dynamics.get("convexity", 0) > 1.0:
                boosting_reasons.append(f"💎 右側 20%：高凸性資產 (Convexity={dynamics['convexity']:.2f})")
                decision = "LONG"
            else:
                decision = "LONG"
        elif must_pass and overall >= 50:
            # 中間 60%：穩定核心但缺乏超額報酬 -> 徹底捨棄
            decision = "HOLD_CASH"
            blocking_reasons.append("⚠️ 0% 隱藏危險區：標的處於平庸中間帶，缺乏獲利凸性，拒絕交易")
        else:
            decision = "HOLD_CASH"
            
        # 左側 20%：尾部風險熔斷
        if dynamics and dynamics.get("tail_risk", 0) < -5.0:
             blocking_reasons.append(f"💀 左側 20%：毀滅性風險預警 (TailRisk={dynamics['tail_risk']:.2f})")
             decision = "HOLD_CASH"

        return FilterResult(
            decision         = decision,
            overall_score    = overall,
            dimensions       = dimensions,
            blocking_reasons = blocking_reasons,
            boosting_reasons = boosting_reasons,
        )

    def backtest_filter(
        self,
        df_feat:   pd.DataFrame,
        oof_preds: pd.Series,
    ) -> pd.DataFrame:
        """
        歷史回測：對每個歷史日期套用過濾規則，輸出每日決策表。

        用於 backtest_audit.py，比較：
          全部訊號 vs 只交易「通過過濾」的訊號

        回傳 DataFrame，含 decision / overall_score / passed 欄位
        """
        results = []
        cfg = self.cfg

        for dt in oof_preds.dropna().index:
            if dt not in df_feat.index:
                continue
            sub = df_feat.loc[:dt]
            if len(sub) < 5:
                continue
            latest = sub.iloc[-1]
            prob_up   = float(oof_preds.loc[dt])
            agreement = float(latest.get("model_agreement_hist", 0.5))

            vol_20d      = float(latest.get("realized_vol_20d", 0.25))
            trend_regime = str(latest.get("trend_regime", "sideways"))
            foreign_weekly = float(latest.get("foreign_net_weekly", 0))
            gm_qoq       = float(latest.get("gross_margin_qoq", np.nan))

            # 簡化版過濾（僅用可量化條件）
            prob_ok  = prob_up >= cfg["prob_up_threshold"]
            vol_ok   = vol_20d < 0.40
            trend_ok = trend_regime in cfg["allowed_trend_for_long"]
            chip_ok  = foreign_weekly >= cfg["min_foreign_net_weekly"]
            gm_ok    = np.isnan(gm_qoq) or (gm_qoq >= cfg["gross_margin_qoq_min"])

            # 評分
            score = (
                min(1.0, max(0.0, (prob_up - 0.5) / 0.5)) * 35 +
                (1.0 if vol_20d < 0.20 else 0.5 if vol_ok else 0.0) * 12.5 +
                (1.0 if trend_regime == "bull" else 0.5 if trend_ok else 0.0) * 12.5 +
                (1.0 if foreign_weekly > 0 else 0.3) * 25 +
                (1.0 if (not np.isnan(gm_qoq) and gm_qoq >= 0) else 0.5) * 15
            )

            must_pass = prob_ok and vol_ok and trend_ok
            if must_pass and score >= 60:
                decision = "LONG"
            elif must_pass and score >= 45:
                decision = "WATCH"
            else:
                decision = "HOLD_CASH"

            results.append({
                "date":          dt,
                "prob_up":       prob_up,
                "vol_20d":       vol_20d,
                "trend_regime":  trend_regime,
                "foreign_weekly": foreign_weekly,
                "overall_score": score,
                "decision":      decision,
                "passed":        decision == "LONG",
            })

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).set_index("date")
