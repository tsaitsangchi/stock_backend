import numpy as np
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from typing import Optional

logger = logging.getLogger(__name__)

def directional_accuracy(y_true: pd.Series, y_pred_signal: pd.Series) -> float:
    """計算方向準確率（漲跌一致性）。"""
    y_true_sign = (y_true > 0).astype(int)
    y_pred_sign = (y_pred_signal > 0).astype(int)
    correct = (y_true_sign == y_pred_sign).sum()
    return float(correct / len(y_true)) if len(y_true) > 0 else 0.0

def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """計算秩相關係數 (Rank IC)。"""
    if len(y_true) < 2:
        return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic)

def calculate_net_return(gross_return: float, ticker: str = "2330") -> float:
    """
    計算單次交易淨報酬（扣除摩擦成本）。
    """
    is_etf = ticker.startswith("00")
    tax = 0.001 if is_etf else 0.003
    fee = 0.001425 * 2 * 0.6 
    friction = tax + fee
    return gross_return - friction

def simulate_sharpe(y_true: pd.Series, prob_up: pd.Series, ticker: str = "2330") -> dict:
    """
    模擬夏普比率與期望值。
    """
    returns = y_true[prob_up > 0.5]
    if len(returns) == 0:
        return {"sharpe": 0.0, "expectancy": 0.0, "win_rate": 0.0}
    
    net_returns = returns.apply(lambda x: calculate_net_return(x, ticker))
    
    avg = net_returns.mean()
    std = net_returns.std()
    sharpe = (avg / std * np.sqrt(252)) if std > 1e-6 else 0.0
    win_rate = (net_returns > 0).mean()
    
    return {
        "sharpe": float(sharpe),
        "expectancy": float(avg),
        "win_rate": float(win_rate)
    }

def evaluate_fold(y_true: pd.Series, prob_up: pd.Series, stock_id: str = "2330") -> dict:
    """彙整單一 Fold 的全套指標（含 single-class 保護）。"""
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(prob_up, dtype=float)
    y_s   = pd.Series(y_arr)
    p_s   = pd.Series(p_arr)

    da  = directional_accuracy(y_s, p_s - 0.5)

    gross_returns = y_s[p_s > 0.5]
    if len(gross_returns) > 0:
        avg_gross = float(gross_returns.mean())
        avg_net = calculate_net_return(avg_gross, stock_id)
    else:
        avg_net = 0.0

    y_binary = (y_arr > 0).astype(int)
    n_classes = len(np.unique(y_binary))
    if n_classes < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_binary, p_arr)

    ic  = information_coefficient(y_s, p_s)
    sim = simulate_sharpe(y_s, p_s, ticker=stock_id)
    return {"directional_accuracy": da, "auc": auc, "ic": ic, "avg_net_return": avg_net, **sim}

def regime_analysis(
    df:       pd.DataFrame,
    oof_pred: pd.Series,
    regime_config: dict,
) -> dict:
    """
    依市場波動 regime（低波動 / 中波動 / 高波動）+ 趨勢 regime 分組評估 OOF 預測表現。
    """
    vol_col = "realized_vol_20d"
    if vol_col not in df.columns:
        logger.warning("  [Regime] realized_vol_20d 欄位不存在，跳過 regime 分析")
        return {}

    vol_low  = regime_config["vol_low"]
    vol_high = regime_config["vol_high"]

    valid_idx = oof_pred.dropna().index
    if len(valid_idx) == 0:
        return {}

    vol   = df.loc[valid_idx, vol_col].fillna(df[vol_col].median())
    y_reg = df.loc[valid_idx, "target_30d"]
    pred  = oof_pred.loc[valid_idx]

    vol_regimes = {
        f"低波動（vol < {vol_low:.0%}）": vol < vol_low,
        f"中波動（{vol_low:.0%} ≤ vol < {vol_high:.0%})":
            (vol >= vol_low) & (vol < vol_high),
        f"高波動（vol ≥ {vol_high:.0%}）": vol >= vol_high,
    }

    results = {}
    logger.info("\n=== Regime 分析（波動率分群）===")
    for label, mask in vol_regimes.items():
        n = mask.sum()
        if n < 20:
            logger.info(f"  {label}：樣本不足（{n} 筆），略過")
            continue
        try:
            m = evaluate_fold(y_reg[mask], pred[mask])
            results[label] = m
            _auc_str = f"{m['auc']:.3f}" if not np.isnan(m['auc']) else " NaN"
            _ic_str  = f"{m['ic']:.3f}"  if not np.isnan(m['ic'])  else " NaN"
            logger.info(
                f"  {label}（n={n:4d}）｜"
                f"DA={m['directional_accuracy']:.3f}  "
                f"AUC={_auc_str}  "
                f"IC={_ic_str}  Sharpe={m['sharpe']:.2f}  "
            )
        except Exception as e:
            logger.warning(f"  {label} 評估失敗：{e}")

    if "trend_regime" in df.columns:
        trend_col = df.loc[valid_idx, "trend_regime"]
        trend_regimes = {
            "牛市（bull）": trend_col == "bull",
            "熊市（bear）": trend_col == "bear",
            "整理期（sideways）": trend_col == "sideways",
        }
        logger.info("\n=== Regime 分析（趨勢分群）===")
        for label, mask in trend_regimes.items():
            n = mask.sum()
            if n < 20: continue
            try:
                m = evaluate_fold(y_reg[mask], pred[mask])
                results[f"trend_{label}"] = m
                logger.info(f"  {label}（n={n:4d}）｜DA={m['directional_accuracy']:.3f}  Sharpe={m['sharpe']:.2f}")
            except: pass

    # 高波動 vs 低波動衰退量
    low_key  = [k for k in results if "低波動" in k]
    high_key = [k for k in results if "高波動" in k]
    if low_key and high_key:
        decay = results[low_key[0]]["directional_accuracy"] - results[high_key[0]]["directional_accuracy"]
        results["_da_decay_high_vs_low"] = decay
        logger.info(f"\n  【關鍵】高波動 vs 低波動 DA 衰退：{decay:+.3f}")

    return results
