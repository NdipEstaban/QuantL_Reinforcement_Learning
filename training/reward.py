"""
training/reward.py
===================
Reward functions for gym-trading-env.

Version history:
  v1  Asymmetric log return + graduated inaction + freq bonus + DD penalty.
      Problem: freq bonus rewarded direct flips (+1→-1) over going through 0.

  v2  Added intra-trade DD penalty, quality exit bonus, consecutive loss penalty.
      Problem: freq bonus still present, model still never used action=0.

  v3  RECOMMENDED — teaches the model to use action=0 (flat) correctly.
      · REMOVED  frequency bonus (root cause of no-0 behaviour)
      · ADDED    direct flip penalty (+1→-1 or -1→+1 without passing through 0)
      · ENHANCED quality exit bonus — now RRR-scaled, weight raised 0.05→0.15
      · RETAINED all v2 components (intra-DD, consecutive loss, DD penalty)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════


def _get_pos(history: dict, offset: int) -> int:
    """Safe position accessor — returns 0 on any error."""
    try:
        return int(history["position", offset])
    except Exception:
        return 0


def _scan_entry_pv(
    history: dict, pv: np.ndarray, start_offset: int = -2, scan_limit: int = 500
):
    """
    Walk backwards from start_offset looking for the last flat→position
    transition. Returns the portfolio value at that entry candle, or None.
    """
    for k in range(start_offset, -scan_limit - 1, -1):
        try:
            p = history["position", k]
            if p == 0:
                idx = len(pv) + k + 1
                if 0 <= idx < len(pv):
                    return float(pv[idx])
                return None
        except Exception:
            return None
    return None
# ══════════════════════════════════════════════════════════════════════
# SAFE EXTRACTORS (Multiprocessing / VecEnv safe)
# ══════════════════════════════════════════════════════════════════════


def _safe_array(x):
    """Always return a 1D numpy array."""
    if isinstance(x, np.ndarray):
        return np.asarray(x).reshape(-1)
    return np.asarray([x], dtype=np.float64)


def _safe_last_two(pv):
    """Return (prev, curr) safely or (None, None)."""
    pv = _safe_array(pv)
    if pv.size < 2:
        return None, None
    return float(pv[-2]), float(pv[-1])


# ══════════════════════════════════════════════════════════════════════
# LEGACY (kept for comparison)
# ══════════════════════════════════════════════════════════════════════


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
    return float(np.mean(returns) / std) if std > 0 else 0.0


def reward_account_flip(history: dict) -> float:
    """Legacy v0."""
    pv = np.array(history["portfolio_valuation"])
    if len(pv) < 3:
        return 0.0
    prev, curr = pv[-2], pv[-1]
    if prev <= 0:
        return 0.0
    step_return = (curr - prev) / prev
    rrr = getattr(config, "RRR_MULTIPLIER", 3.0)
    asymmetric = step_return * rrr if step_return > 0 else step_return
    peak = np.maximum.accumulate(pv)
    dd_frac = min((curr - peak[-1]) / np.maximum(peak[-1], 1e-8), 0.0)
    dd_penalty = -(dd_frac**2) * getattr(config, "DD_PENALTY_WEIGHT", 2.0)
    try:
        cp = history["position", -1]
    except Exception:
        cp = 0
    sel = getattr(config, "SELECTIVITY_BONUS", 0.0002) if cp == 0 else 0.0
    return float(np.tanh(asymmetric + dd_penalty + sel))


def reward_account_flip_aggressive(history: dict) -> float:
    """V1 — kept for reference."""
    pv = np.array(history["portfolio_valuation"])
    if len(pv) < 3:
        return 0.0
    prev, curr, init = pv[-2], pv[-1], pv[0]
    if prev <= 0 or init <= 0:
        return 0.0
    log_step = float(np.log(curr / prev)) if curr > 0 and prev > 0 else 0.0
    rrr = getattr(config, "FLIP_RRR_MULTIPLIER", 3.0)
    log_scale = getattr(config, "FLIP_LOG_SCALE", 10.0)
    base = log_step * rrr * log_scale if log_step > 0 else log_step * log_scale
    base_in = getattr(config, "FLIP_INACTION_PENALTY", -0.000001)
    inaction = 0.0
    try:
        cp = history["position", -1]
    except Exception:
        cp = 0
    if cp == 0:
        cf = 0
        for k in range(-1, -min(120, len(pv) - 1) - 1, -1):
            try:
                p = history["position", k]
                if p == 0:
                    cf += 1
                else:
                    break
            except Exception:
                break
        if cf <= 10:
            inaction = 0.0
        elif cf <= 30:
            inaction = base_in * 0.2
        elif cf <= 60:
            inaction = base_in * 0.5
        elif cf <= 100:
            inaction = base_in * 1.0
        else:
            inaction = base_in * 2.0
    fw = getattr(config, "FLIP_FREQ_BONUS_WEIGHT", 0.0005)
    tc, pp = 0, 0
    for k in range(-min(500, len(pv) - 1), 0):
        try:
            p = history["position", k]
            if p != 0 and pp == 0:
                tc += 1
            pp = p
        except Exception:
            break
    freq = fw * np.log1p(tc) if tc > 0 else 0.0
    peak = np.maximum.accumulate(pv)
    dd_frac = min((curr - peak[-1]) / np.maximum(peak[-1], 1e-8), 0.0)
    dd_free = getattr(config, "FLIP_DD_FREE_ZONE", 0.05)
    dd_w = getattr(config, "FLIP_DD_PENALTY_WEIGHT", 5.0)
    dd_pen = -((abs(dd_frac) - dd_free) ** 2) * dd_w if abs(dd_frac) > dd_free else 0.0
    return float(np.tanh(base + inaction + freq + dd_pen))


def reward_account_flip_simple(history: dict) -> float:
    pv = np.asarray(history["portfolio_valuation"])
    if len(pv) < 3:
        return 0.0

    prev, curr = pv[-2], pv[-1]
    if prev <= 0:
        return 0.0

    # ─────────────────────────────────────────────
    # Shared state
    # ─────────────────────────────────────────────
    log_step = np.log(curr / prev)
    risk = getattr(config, "RISK_PER_TRADE", 0.15)

    pos = _get_pos(history, -1)
    prev_pos = _get_pos(history, -2)

    rrr_mult = getattr(config, "FLIP_RRR_MULTIPLIER", 4.0)
    log_scale = getattr(config, "FLIP_LOG_SCALE", 10.0)

    # ─────────────────────────────────────────────
    # 1. Asymmetric log return
    # ─────────────────────────────────────────────
    base = log_step * log_scale
    if log_step > 0:
        base *= rrr_mult

    # ─────────────────────────────────────────────
    # 2. Mild flat penalty (constant, not laddered)
    # ─────────────────────────────────────────────
    flat_pen = 0.0
    if pos == 0:
        flat_pen = getattr(config, "FLIP_INACTION_PENALTY", -1e-6)

    # ─────────────────────────────────────────────
    # 3. Portfolio drawdown penalty (quadratic)
    # ─────────────────────────────────────────────
    peak = np.max(pv)
    dd = min((curr - peak) / peak, 0.0)
    dd_free = getattr(config, "FLIP_DD_FREE_ZONE", 0.08)

    dd_pen = 0.0
    if abs(dd) > dd_free:
        excess = abs(dd) - dd_free
        dd_pen = -(excess**2) * getattr(config, "FLIP_DD_PENALTY_WEIGHT", 2.0)

    # ─────────────────────────────────────────────
    # 4. Intra-trade deep loss penalty
    # ─────────────────────────────────────────────
    intra_pen = 0.0
    if pos != 0:
        entry = _scan_entry_pv(history, pv, -1, 300)
        if entry:
            loss = max((entry - curr) / entry, 0.0)
            threshold = risk * getattr(config, "FLIP_INTRA_DD_THRESHOLD", 0.8)
            if loss > threshold:
                excess = loss - threshold
                intra_pen = -(excess**2) * getattr(config, "FLIP_INTRA_DD_WEIGHT", 2.0)

    # ─────────────────────────────────────────────
    # 5. Quality exit bonus (RRR scaled)
    # ─────────────────────────────────────────────
    exit_bonus = 0.0
    if pos == 0 and prev_pos != 0:
        entry = _scan_entry_pv(history, pv, -2, 300)
        if entry and curr > entry:
            profit = (curr - entry) / entry
            trade_rrr = profit / risk
            weight = getattr(config, "FLIP_QUALITY_EXIT_BONUS", 0.15)
            exit_bonus = weight * np.tanh(trade_rrr)

    # ─────────────────────────────────────────────
    # 6. Direct flip penalty
    # ─────────────────────────────────────────────
    flip_pen = 0.0
    if (prev_pos == 1 and pos == -1) or (prev_pos == -1 and pos == 1):
        flip_pen = getattr(config, "FLIP_DIRECT_FLIP_PENALTY", -0.05)

    total = base + flat_pen + dd_pen + intra_pen + exit_bonus + flip_pen
    return float(np.tanh(total))


def reward_custom(history: dict) -> float:
    try:
        pv_raw = history["portfolio_valuation"]
    except Exception:
        return 0.0

    pv = _safe_array(pv_raw)
    prev, curr = _safe_last_two(pv)

    if prev is None or prev <= 0:
        return 0.0

    log_step = np.log(curr / prev)

    pos = _get_pos(history, -1)
    prev_pos = _get_pos(history, -2)

    risk = getattr(config, "RISK_PER_TRADE", 0.15)
    rrr_mult = getattr(config, "FLIP_RRR_MULTIPLIER", 4.0)
    log_scale = getattr(config, "FLIP_LOG_SCALE", 10.0)

    # 1. Asymmetric return
    base = log_step * log_scale
    if log_step > 0:
        base *= rrr_mult

    # 2. Mild flat penalty
    flat_pen = 0.0
    if pos == 0:
        flat_pen = getattr(config, "FLIP_INACTION_PENALTY", -1e-6)

    # 3. Portfolio DD penalty
    peak = np.max(pv)
    dd = min((curr - peak) / max(peak, 1e-8), 0.0)

    dd_free = getattr(config, "FLIP_DD_FREE_ZONE", 0.08)
    dd_pen = 0.0
    if abs(dd) > dd_free:
        excess = abs(dd) - dd_free
        dd_pen = -(excess**2) * getattr(config, "FLIP_DD_PENALTY_WEIGHT", 2.0)

    # 4. Intra-trade deep loss penalty
    intra_pen = 0.0
    if pos != 0:
        entry = _scan_entry_pv(history, pv, -1, 300)
        if entry:
            loss = max((entry - curr) / entry, 0.0)
            threshold = risk * getattr(config, "FLIP_INTRA_DD_THRESHOLD", 0.8)
            if loss > threshold:
                excess = loss - threshold
                intra_pen = -(excess**2) * getattr(config, "FLIP_INTRA_DD_WEIGHT", 2.0)

    # 5. Quality exit bonus
    exit_bonus = 0.0
    if pos == 0 and prev_pos != 0:
        entry = _scan_entry_pv(history, pv, -2, 300)
        if entry and curr > entry:
            profit = (curr - entry) / entry
            trade_rrr = profit / max(risk, 1e-8)
            weight = getattr(config, "FLIP_QUALITY_EXIT_BONUS", 0.15)
            exit_bonus = weight * np.tanh(trade_rrr)

    # 6. Direct flip penalty
    flip_pen = 0.0
    if (prev_pos == 1 and pos == -1) or (prev_pos == -1 and pos == 1):
        flip_pen = getattr(config, "FLIP_DIRECT_FLIP_PENALTY", -0.05)

    total = base + flat_pen + dd_pen + intra_pen + exit_bonus + flip_pen
    return float(np.tanh(total))


def reward_institutional(history) -> float:
    try:
        pv_raw = history["portfolio_valuation"]
    except Exception:
        return 0.0

    pv = _safe_array(pv_raw)
    if pv.size < 5:
        return 0.0

    prev = float(pv[-2])
    curr = float(pv[-1])

    if prev <= 0:
        return 0.0

    pos = _get_pos(history, -1)
    prev_pos = _get_pos(history, -2)

    # 1. Log return
    log_ret = np.log(curr / prev)

    # 2. Rolling volatility
    returns = np.diff(pv) / np.maximum(pv[:-1], 1e-8)
    if returns.size == 0:
        return 0.0

    window = returns[-20:] if returns.size >= 20 else returns
    vol = np.std(window)

    vol_pen = -vol * getattr(config, "INST_VOL_WEIGHT", 2.0)

    # 3. Drawdown penalty
    peak = np.maximum.accumulate(pv)
    dd = (curr - peak[-1]) / max(peak[-1], 1e-8)

    dd_pen = 0.0
    if dd < 0:
        dd_pen = -(dd**2) * getattr(config, "INST_DD_WEIGHT", 8.0)

    # 4. Direct flip penalty
    flip_pen = 0.0
    if (prev_pos == 1 and pos == -1) or (prev_pos == -1 and pos == 1):
        flip_pen = getattr(config, "INST_FLIP_PENALTY", -0.05)

    total = (
        log_ret * getattr(config, "INST_RETURN_SCALE", 10.0)
        + vol_pen
        + dd_pen
        + flip_pen
    )

    return float(np.tanh(total))


# ══════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════

_REWARD_FUNCTIONS = {
    "log_return": reward_log_return,
    "sharpe": reward_sharpe,
    "account_flip": reward_account_flip,
    "account_flip_aggressive": reward_account_flip_aggressive,
    "account_flip_simple": reward_account_flip_simple,
    "custom": reward_custom,
    "institutional": reward_institutional,
}


def get_reward_function(name: str):
    if name not in _REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Choose from: {list(_REWARD_FUNCTIONS.keys())}"
        )
    return _REWARD_FUNCTIONS[name]
