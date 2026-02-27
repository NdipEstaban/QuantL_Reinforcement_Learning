"""
evaluation/metrics.py
======================
Performance metrics for backtesting.
"""

import numpy as np
import pandas as pd


def compute_metrics(portfolio_values: np.ndarray, trades: list = None) -> dict:
    """
    Compute standard trading performance metrics.

    Parameters
    ----------
    portfolio_values : array of portfolio value at each step
    trades           : optional list of (entry, exit, pnl) tuples

    Returns
    -------
    dict of metrics
    """
    pv      = np.array(portfolio_values)
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-8)

    # Total return
    total_return = (pv[-1] - pv[0]) / pv[0]

    # Annualised return (assume 252 trading days, 24 for forex)
    n_periods = len(pv)
    periods_per_year = 252 * 24   # for H1; adjust for M15: 252*96
    ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # Sharpe ratio
    std = np.std(returns)
    sharpe = (np.mean(returns) / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    # Max drawdown
    peak    = np.maximum.accumulate(pv)
    dd      = (pv - peak) / np.maximum(peak, 1e-8)
    max_dd  = float(dd.min())

    # Calmar ratio
    calmar  = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate (from trades if available)
    win_rate    = None
    profit_factor = None
    avg_win     = None
    avg_loss    = None

    if trades:
        pnls     = [t["pnl"] for t in trades if "pnl" in t]
        if pnls:
            wins     = [p for p in pnls if p > 0]
            losses   = [p for p in pnls if p <= 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
            gross_win  = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf
            avg_win  = np.mean(wins)  if wins   else 0
            avg_loss = np.mean(losses) if losses else 0

    return {
        "total_return_pct"  : round(total_return * 100, 2),
        "ann_return_pct"    : round(ann_return * 100, 2),
        "sharpe_ratio"      : round(sharpe, 3),
        "max_drawdown_pct"  : round(max_dd * 100, 2),
        "calmar_ratio"      : round(calmar, 3),
        "final_portfolio"   : round(pv[-1], 2),
        "win_rate_pct"      : round(win_rate * 100, 2) if win_rate is not None else None,
        "profit_factor"     : round(profit_factor, 3) if profit_factor is not None else None,
        "avg_win"           : round(avg_win, 4)  if avg_win  is not None else None,
        "avg_loss"          : round(avg_loss, 4) if avg_loss is not None else None,
        "n_trades"          : len(trades) if trades else None,
    }


def print_metrics(metrics: dict, title: str = "Performance Metrics"):
    print(f"\n{'═'*45}")
    print(f"  {title}")
    print(f"{'═'*45}")
    print(f"  Total Return       : {metrics['total_return_pct']:>8.2f}%")
    print(f"  Annualised Return  : {metrics['ann_return_pct']:>8.2f}%")
    print(f"  Sharpe Ratio       : {metrics['sharpe_ratio']:>8.3f}")
    print(f"  Max Drawdown       : {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Calmar Ratio       : {metrics['calmar_ratio']:>8.3f}")
    print(f"  Final Portfolio    : ${metrics['final_portfolio']:>10,.2f}")
    if metrics["win_rate_pct"] is not None:
        print(f"  Win Rate           : {metrics['win_rate_pct']:>8.2f}%")
    if metrics["profit_factor"] is not None:
        print(f"  Profit Factor      : {metrics['profit_factor']:>8.3f}")
    if metrics["n_trades"] is not None:
        print(f"  Total Trades       : {metrics['n_trades']:>8}")
    print(f"{'═'*45}")
