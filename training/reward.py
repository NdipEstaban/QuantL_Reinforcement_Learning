"""
training/reward.py
===================
Reward functions for gym-trading-env.

account_flip_aggressive:
    Designed for 10x monthly account flipping.

    Core philosophy change from previous version:
    - REMOVE selectivity bonus (was teaching model to stay flat)
    - ADD inaction penalty (staying flat = missing opportunity)
    - Use LOG PORTFOLIO GROWTH as base (captures compounding naturally)
    - Exponential win bonus for 3R+ trades
    - Hard inaction penalty after N flat candles in a row
    - Drawdown penalty is asymmetric — small DD tolerated, large DD catastrophic
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def reward_log_return(history: dict) -> float:
    pv = history["portfolio_valuation"]
    if len(pv) < 2 or pv[-2] <= 0:
        return 0.0
    return float(np.log(pv[-1] / pv[-2]))


def reward_sharpe(history: dict) -> float:
    pv = np.array(history["portfolio_valuation"])
    if len(pv) < 5:
        return 0.0
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-8)
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std)


def reward_sharpe_drawdown(history: dict) -> float:
    pv = np.array(history["portfolio_valuation"])
    if len(pv) < 5:
        return 0.0
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-8)
    std = np.std(returns)
    sharpe = float(np.mean(returns) / std) if std > 0 else 0.0
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / np.maximum(peak, 1e-8)
    penalty = float(dd[-1]) * config.DRAWDOWN_PENALTY_WEIGHT
    return sharpe + penalty


def reward_account_flip(history: dict) -> float:
    """Legacy — kept for comparison."""
    pv = np.array(history["portfolio_valuation"])
    if len(pv) < 3:
        return 0.0
    prev = pv[-2]
    curr = pv[-1]
    if prev <= 0:
        return 0.0
    step_return = (curr - prev) / prev
    rrr = getattr(config, "RRR_MULTIPLIER", 3.0)
    asymmetric_ret = step_return * rrr if step_return > 0 else step_return
    asymmetric_component = asymmetric_ret * getattr(config, "ASYMMETRIC_WEIGHT", 1.0)
    window = getattr(config, "SORTINO_WINDOW", 50)
    window_pv = pv[-min(window, len(pv)) :]
    sortino = 0.0
    if len(window_pv) >= 3:
        rets = np.diff(window_pv) / np.maximum(window_pv[:-1], 1e-8)
        neg_rets = rets[rets < 0]
        if len(neg_rets) > 1:
            downside_std = np.std(neg_rets)
            sortino = np.mean(rets) / downside_std if downside_std > 0 else 0.01
        else:
            sortino = 0.01
    sortino_component = sortino * getattr(config, "SORTINO_WEIGHT", 0.5)
    peak = np.maximum.accumulate(pv)
    dd_frac = min((curr - peak[-1]) / np.maximum(peak[-1], 1e-8), 0.0)
    dd_penalty = -(dd_frac**2) * getattr(config, "DD_PENALTY_WEIGHT", 2.0)
    selectivity_bonus = 0.0
    try:
        current_pos = history["position", -1]
    except Exception:
        current_pos = 0
    if current_pos == 0:
        selectivity_bonus = getattr(config, "SELECTIVITY_BONUS", 0.0002)
    total = asymmetric_component + sortino_component + dd_penalty + selectivity_bonus
    return float(np.tanh(total))


def reward_account_flip_aggressive(history: dict) -> float:
    """
    Aggressive account-flip reward targeting 10x monthly returns.

    Core insight from failure analysis:
    ────────────────────────────────────
    Previous version had inaction_penalty = -0.0001 per candle.
    Over 2000 candles that costs only -0.20 total.
    One good trade returns tanh(large) ≈ +1.0.
    Model learned: make 1 trade, stay flat 1999 candles = free money.

    Fix:
    ────
    Inaction penalty must EXCEED the amortized value of cherry-picking.
    If best possible trade = +1.0 reward:
        1999 flat candles must cost MORE than +1.0 to prevent 1-trade strategy.
        Base penalty -0.001 × 2000 = -2.0 > any single trade reward.

    But penalty must not be so large that ANY trade is better than flat.
    Bad trades must still cost more than inaction.
    Solution: escalating penalty that scales with consecutive flat candles,
    combined with a per-trade cap so no single trade dominates the episode.

    Components:
        1. ASYMMETRIC LOG RETURN     — wins amplified 3x, losses face value
        2. ESCALATING INACTION       — flat penalty doubles every FLAT_WINDOW candles
        3. TRADE FREQUENCY BONUS     — bonus proportional to trade count this episode
        4. DRAWDOWN PENALTY          — exponential above free zone, zero below
        5. WIN STREAK BONUS          — consecutive profitable trades get multiplier
    """
    pv = np.array(history["portfolio_valuation"])
    if len(pv) < 3:
        return 0.0

    prev = pv[-2]
    curr = pv[-1]
    init = pv[0]

    if prev <= 0 or init <= 0:
        return 0.0

    step_return = (curr - prev) / prev
    log_step    = float(np.log(curr / prev)) if curr > 0 and prev > 0 else 0.0

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 1: ASYMMETRIC LOG RETURN
    # Wins get RRR_MULTIPLIER boost, losses taken at face value.
    # Log return captures compounding — growing $10k→$11k and
    # $100k→$110k both give the same log signal.
    # ══════════════════════════════════════════════════════════════
    rrr       = getattr(config, "FLIP_RRR_MULTIPLIER", 3.0)
    log_scale = getattr(config, "FLIP_LOG_SCALE", 10.0)

    if log_step > 0:
        base_component = log_step * rrr * log_scale
    else:
        base_component = log_step * log_scale

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 2: GRADUATED INACTION PENALTY
    #
    # Key change: penalty is much smaller in early episode candles
    # (model needs time to observe features before trading)
    # and only escalates after a reasonable observation window.
    #
    # Tier structure:
    #   Flat 0-10 candles:    0.0      (observation window, free)
    #   Flat 11-30 candles:  -0.0002   (mild — setup forming)
    #   Flat 31-60 candles:  -0.0005   (moderate — should have entered)
    #   Flat 61-100 candles: -0.001    (strong — missing clear setups)
    #   Flat 100+ candles:   -0.002    (severe — unacceptable inaction)
    # ══════════════════════════════════════════════════════════════
    inaction_penalty = 0.0
    base_inaction = getattr(config, "FLIP_INACTION_PENALTY", -0.0002)
    flat_window = getattr(config, "FLIP_FLAT_WINDOW", 20)

    try:
        current_pos = history["position", -1]
    except Exception:
        current_pos = 0

    if current_pos == 0:
        consecutive_flat = 0
        check_len = min(120, len(pv) - 1)
        for k in range(-1, -check_len - 1, -1):
            try:
                p = history["position", k]
                if p == 0:
                    consecutive_flat += 1
                else:
                    break
            except Exception:
                break

        if consecutive_flat <= 10:
            inaction_penalty = 0.0  # free observation window
        elif consecutive_flat <= 30:
            inaction_penalty = base_inaction * 0.2
        elif consecutive_flat <= 60:
            inaction_penalty = base_inaction * 0.5
        elif consecutive_flat <= 100:
            inaction_penalty = base_inaction * 1.0
        else:
            inaction_penalty = base_inaction * 2.0

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 3: TRADE FREQUENCY BONUS
    # Logarithmic — rewards having trades but with diminishing returns.
    # 5 trades: bonus ≈ 0.0009
    # 20 trades: bonus ≈ 0.0015
    # 100 trades: bonus ≈ 0.0023
    # Prevents "trade 500 times randomly" from dominating.
    # ══════════════════════════════════════════════════════════════
    freq_bonus = 0.0
    freq_weight = getattr(config, "FLIP_FREQ_BONUS_WEIGHT", 0.0005)
    check_window = min(500, len(pv) - 1)
    trade_count = 0
    prev_p = 0

    for k in range(-check_window, 0):
        try:
            p = history["position", k]
            if p != 0 and prev_p == 0:
                trade_count += 1
            prev_p = p
        except Exception:
            break

    if trade_count > 0:
        freq_bonus = freq_weight * np.log1p(trade_count)

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 4: DRAWDOWN PENALTY
    # Free zone: drawdowns below DD_FREE_ZONE ignored entirely.
    # Allows model to accept normal 5-15% trade risk.
    # Above free zone: quadratic penalty — protects against blowup.
    # ══════════════════════════════════════════════════════════════
    dd_penalty = 0.0
    peak       = np.maximum.accumulate(pv)
    dd_frac    = (curr - peak[-1]) / np.maximum(peak[-1], 1e-8)
    dd_frac    = min(dd_frac, 0.0)
    dd_depth   = abs(dd_frac)

    dd_free    = getattr(config, "FLIP_DD_FREE_ZONE", 0.05)
    dd_weight  = getattr(config, "FLIP_DD_PENALTY_WEIGHT", 5.0)

    if dd_depth > dd_free:
        excess_dd  = dd_depth - dd_free
        dd_penalty = -(excess_dd ** 2) * dd_weight

    # ══════════════════════════════════════════════════════════════
    # TOTAL
    # ══════════════════════════════════════════════════════════════
    total = (
        base_component
        + inaction_penalty
        + freq_bonus
        + dd_penalty
    )

    return float(np.tanh(total))


# ── Registry ─────────────────────────────────────────────────────

_REWARD_FUNCTIONS = {
    "log_return": reward_log_return,
    "sharpe": reward_sharpe,
    "sharpe_drawdown": reward_sharpe_drawdown,
    "account_flip": reward_account_flip,
    "account_flip_aggressive": reward_account_flip_aggressive,
}


def get_reward_function(name: str):
    if name not in _REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Choose from: {list(_REWARD_FUNCTIONS.keys())}"
        )
    return _REWARD_FUNCTIONS[name]
