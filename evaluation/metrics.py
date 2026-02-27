"""
evaluation/metrics.py
======================
Performance metrics for backtesting.

Key design decisions:
  - Sharpe is computed on DAILY returns, not candle-by-candle returns.
    Per-candle Sharpe is meaningless when 80%+ of candles are flat (std → 0).
  - Annualised return is only shown when the episode covers >= 30 days.
    Extrapolating 20 days to annual figures is misleading.
  - Win rate and profit factor are the primary metrics — they come from
    actual trade-level data and are not affected by flat periods.
"""

import numpy as np
import pandas as pd


# M15: 4 candles/hour × 24 hours × 252 trading days
M15_CANDLES_PER_DAY = 96
M15_CANDLES_PER_YEAR = 252 * M15_CANDLES_PER_DAY  # 24,192


def compute_metrics(
    portfolio_values: list,
    trades: list = None,
    candles_per_day: int = M15_CANDLES_PER_DAY,
) -> dict:
    """
    Compute trading performance metrics.

    Parameters
    ----------
    portfolio_values : portfolio value at each candle step
    trades           : list of trade dicts with 'pnl' key
    candles_per_day  : 96 for M15, 24 for H1, 1440 for M1
    """
    pv = np.array(portfolio_values, dtype=float)
    if len(pv) < 2:
        return {}

    # ── Total return ─────────────────────────────────────────────
    total_return = (pv[-1] - pv[0]) / max(pv[0], 1e-8)

    # ── Days covered by this episode ─────────────────────────────
    n_candles = len(pv)
    n_days = n_candles / candles_per_day  # e.g. 2000/96 ≈ 20.8 days

    # ── Annualised return (only meaningful if >= 30 days) ─────────
    if n_days >= 30:
        ann_return = (1 + total_return) ** (252.0 / n_days) - 1
    else:
        # Episode too short to annualise reliably — show None
        ann_return = None

    # ── Sharpe on DAILY returns (not candle returns) ──────────────
    # Group candle-level portfolio values into daily buckets.
    # This removes the 80%+ zero-return candles that destroy candle Sharpe.
    n_full_days = int(n_candles // candles_per_day)
    sharpe = 0.0
    daily_returns = None

    if n_full_days >= 10:
        # Reshape into daily chunks, take end-of-day portfolio value
        daily_pv = pv[: n_full_days * candles_per_day].reshape(
            n_full_days, candles_per_day
        )[
            :, -1
        ]  # last candle of each day
        daily_returns = np.diff(daily_pv) / np.maximum(daily_pv[:-1], 1e-8)
        std = np.std(daily_returns)
        if std > 0:
            sharpe = float(np.mean(daily_returns) / std * np.sqrt(252))

    # ── Max drawdown (candle-level — this is correct) ────────────
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / np.maximum(peak, 1e-8)
    max_dd = float(dd.min())

    # ── Calmar (only if annualised return available) ──────────────
    if ann_return is not None and max_dd != 0:
        calmar = ann_return / abs(max_dd)
    else:
        calmar = None

    # ── Trade-level metrics ───────────────────────────────────────
    win_rate = None
    profit_factor = None
    avg_win_pct = None
    avg_loss_pct = None
    expectancy = None

    if trades:
        pnls = [t["pnl"] for t in trades if "pnl" in t]
        if pnls:
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            win_rate = len(wins) / len(pnls)

            gross_win = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

            avg_win_pct = float(np.mean(wins)) if wins else 0.0
            avg_loss_pct = float(np.mean(losses)) if losses else 0.0

            # Expectancy per trade as % of position
            expectancy = win_rate * avg_win_pct + (1 - win_rate) * avg_loss_pct

    return {
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct": (
            round(ann_return * 100, 2) if ann_return is not None else None
        ),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3) if calmar is not None else None,
        "final_portfolio": round(float(pv[-1]), 2),
        "win_rate_pct": round(win_rate * 100, 2) if win_rate is not None else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "avg_win_pct": round(avg_win_pct * 100, 4) if avg_win_pct is not None else None,
        "avg_loss_pct": (
            round(avg_loss_pct * 100, 4) if avg_loss_pct is not None else None
        ),
        "expectancy_pct": (
            round(expectancy * 100, 4) if expectancy is not None else None
        ),
        "n_trades": len(trades) if trades else 0,
        "n_days": round(n_days, 1),
    }


def print_metrics(metrics: dict, title: str = "Performance Metrics"):
    print(f"\n{'═'*47}")
    print(f"  {title}")
    print(f"{'═'*47}")
    print(f"  Total Return       : {metrics['total_return_pct']:>8.2f}%")

    if metrics.get("ann_return_pct") is not None:
        print(f"  Annualised Return  : {metrics['ann_return_pct']:>8.2f}%")
    else:
        print(f"  Annualised Return  : {'n/a (<30 days)':>14}")

    print(f"  Sharpe (daily)     : {metrics['sharpe_ratio']:>8.3f}")
    print(f"  Max Drawdown       : {metrics['max_drawdown_pct']:>8.2f}%")

    if metrics.get("calmar_ratio") is not None:
        print(f"  Calmar Ratio       : {metrics['calmar_ratio']:>8.3f}")

    print(f"  Final Portfolio    : ${metrics['final_portfolio']:>10,.2f}")
    print(f"  Episode length     : {metrics.get('n_days', '?'):>6.1f} trading days")

    if metrics.get("win_rate_pct") is not None:
        print(f"  Win Rate           : {metrics['win_rate_pct']:>8.2f}%")
    if metrics.get("profit_factor") is not None:
        pf = metrics["profit_factor"]
        pf_str = f"{pf:>8.3f}" if pf != float("inf") else "       ∞"
        print(f"  Profit Factor      : {pf_str}")
    if metrics.get("n_trades") is not None:
        print(f"  Total Trades       : {metrics['n_trades']:>8}")
    if metrics.get("avg_win_pct") is not None:
        print(f"  Avg Win            : {metrics['avg_win_pct']:>8.4f}%")
    if metrics.get("avg_loss_pct") is not None:
        print(f"  Avg Loss           : {metrics['avg_loss_pct']:>8.4f}%")
    if metrics.get("expectancy_pct") is not None:
        print(f"  Expectancy/trade   : {metrics['expectancy_pct']:>8.4f}%")

    print(f"{'═'*47}")
