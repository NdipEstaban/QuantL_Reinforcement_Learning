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


def reward_account_flip_aggressive_v2(history: dict) -> float:
    """V2 — kept for reference. Freq bonus still present."""
    pv = np.array(history["portfolio_valuation"])
    if len(pv) < 3:
        return 0.0
    prev, curr, init = pv[-2], pv[-1], pv[0]
    if prev <= 0 or init <= 0:
        return 0.0
    scan_limit = min(500, len(pv) - 1)
    risk_per_trade = getattr(config, "RISK_PER_TRADE", 0.15)
    log_step = float(np.log(curr / prev)) if curr > 0 and prev > 0 else 0.0
    current_pos = _get_pos(history, -1)
    prev_pos = _get_pos(history, -2)
    rrr = getattr(config, "FLIP_RRR_MULTIPLIER", 4.0)
    log_scale = getattr(config, "FLIP_LOG_SCALE", 10.0)
    base = log_step * rrr * log_scale if log_step > 0 else log_step * log_scale
    base_in = getattr(config, "FLIP_INACTION_PENALTY", -0.000001)
    inaction = 0.0
    if current_pos == 0:
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
    for k in range(-scan_limit, 0):
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
    dd_free = getattr(config, "FLIP_DD_FREE_ZONE", 0.08)
    dd_w = getattr(config, "FLIP_DD_PENALTY_WEIGHT", 2.0)
    dd_pen = -((abs(dd_frac) - dd_free) ** 2) * dd_w if abs(dd_frac) > dd_free else 0.0
    intra_dd_pen = 0.0
    if current_pos != 0:
        thr = getattr(config, "FLIP_INTRA_DD_THRESHOLD", 0.80)
        intra_w = getattr(config, "FLIP_INTRA_DD_WEIGHT", 2.0)
        entry_pv = _scan_entry_pv(history, pv, -1, scan_limit)
        if entry_pv is not None and entry_pv > 0:
            loss = max((entry_pv - curr) / entry_pv, 0.0)
            allowed = risk_per_trade * thr
            if loss > allowed:
                intra_dd_pen = -((loss - allowed) ** 2) * intra_w
    quality_exit = 0.0
    if current_pos == 0 and prev_pos != 0:
        exit_w = getattr(config, "FLIP_QUALITY_EXIT_BONUS", 0.05)
        entry_pv = _scan_entry_pv(history, pv, -2, scan_limit)
        if entry_pv is not None and curr > entry_pv:
            profit_frac = (curr - entry_pv) / entry_pv
            trade_rrr = profit_frac / max(risk_per_trade, 1e-8)
            quality_exit = exit_w * np.tanh(trade_rrr)
    consec_pen = 0.0
    if current_pos != 0:
        st = getattr(config, "FLIP_CONSEC_LOSS_THRESHOLD", 2)
        pv2 = getattr(config, "FLIP_CONSEC_LOSS_PENALTY", -0.02)
        results, pp2, e2 = [], 0, None
        for k in range(-scan_limit, 0):
            idx = len(pv) + k
            if not (0 <= idx < len(pv)):
                continue
            try:
                p = history["position", k]
            except Exception:
                break
            if pp2 == 0 and p != 0:
                e2 = pv[idx]
            elif pp2 != 0 and p == 0 and e2 is not None:
                results.append(pv[idx] - e2)
                e2 = None
            pp2 = p
        streak = 0
        for r in reversed(results):
            if r < 0:
                streak += 1
            else:
                break
        if streak >= st:
            consec_pen = pv2
    total = base + inaction + freq + dd_pen + intra_dd_pen + quality_exit + consec_pen
    return float(np.tanh(total))


# ══════════════════════════════════════════════════════════════════════
# V3 — RECOMMENDED
# ══════════════════════════════════════════════════════════════════════


def reward_account_flip_aggressive_v3(history: dict) -> float:
    """
    V3 — teaches the model to use action=0 (flat) correctly.

    Root cause fix:
        The frequency bonus in v1/v2 made direct flips (+1→-1) rational
        because they maintained trade count and avoided the inaction timer.
        The model never needed to use 0. V3 fixes this with three changes:

    Change 1 — FREQUENCY BONUS REMOVED
        No reward for trade count. The model now has no incentive to flip
        directly. The only way to earn the exit bonus is to go through flat.

    Change 2 — DIRECT FLIP PENALTY (new)
        A penalty fires whenever the model transitions +1→-1 or -1→+1
        without an intervening 0. Makes bypassing flat explicitly costly.
        Config: FLIP_DIRECT_FLIP_PENALTY  (default -0.05)

    Change 3 — RRR-SCALED QUALITY EXIT BONUS (enhanced)
        Bonus for exiting to flat in profit now scales with trade RRR:
            1R exit  →  tanh(1.0) × weight  ≈  0.76 × weight
            2R exit  →  tanh(2.0) × weight  ≈  0.96 × weight
            3R+ exit →  tanh(3.0) × weight  ≈  1.00 × weight
        Weight raised from 0.05 to 0.15 — profitable flat exits are now
        clearly more valuable than any direct flip.
        Config: FLIP_QUALITY_EXIT_BONUS  (default 0.15)

    Retained from v2:
        · Asymmetric log return (wins amplified by RRR multiplier)
        · Graduated inaction penalty (escalates with consecutive flat candles)
        · Portfolio drawdown penalty (quadratic above free zone)
        · Intra-trade deep drawdown penalty (fires at 80% of risked amount)
        · Consecutive loss streak penalty (gentle, not punitive)

    Config keys:
        FLIP_RRR_MULTIPLIER          4.0   win amplification
        FLIP_LOG_SCALE              10.0   log return scale
        FLIP_INACTION_PENALTY   -0.000001  base per-step flat penalty
        FLIP_DD_FREE_ZONE            0.08  DD below this is free
        FLIP_DD_PENALTY_WEIGHT       2.0   DD penalty multiplier
        FLIP_INTRA_DD_THRESHOLD      0.80  intra-trade loss threshold
        FLIP_INTRA_DD_WEIGHT         2.0   intra-trade DD penalty scale
        FLIP_QUALITY_EXIT_BONUS      0.15  max bonus for profitable flat exit
        FLIP_CONSEC_LOSS_THRESHOLD   2     losses in a row before penalty
        FLIP_CONSEC_LOSS_PENALTY    -0.02  per-step penalty during streak
        FLIP_DIRECT_FLIP_PENALTY    -0.05  penalty for +1→-1 or -1→+1
    """
    pv = np.array(history["portfolio_valuation"])
    if len(pv) < 3:
        return 0.0

    prev, curr, init = pv[-2], pv[-1], pv[0]
    if prev <= 0 or init <= 0:
        return 0.0

    # ── Shared state (computed once, used across all components) ──
    scan_limit = min(500, len(pv) - 1)
    risk_per_trade = getattr(config, "RISK_PER_TRADE", 0.15)
    log_step = float(np.log(curr / prev)) if curr > 0 and prev > 0 else 0.0
    current_pos = _get_pos(history, -1)
    prev_pos = _get_pos(history, -2)

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 1: ASYMMETRIC LOG RETURN
    # Winning steps amplified by RRR multiplier — teaches the model
    # that profitable candles are worth more than losing ones cost.
    # Log return captures compounding: $10k→$11k = $100k→$110k.
    # ══════════════════════════════════════════════════════════════
    rrr = getattr(config, "FLIP_RRR_MULTIPLIER", 4.0)
    log_scale = getattr(config, "FLIP_LOG_SCALE", 10.0)
    base = log_step * rrr * log_scale if log_step > 0 else log_step * log_scale

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 2: GRADUATED INACTION PENALTY
    # Escalates with consecutive flat candles to prevent the model
    # from learning "stay flat the whole episode" as a strategy.
    # Free 10-candle observation window — model needs time to read
    # features before entering. Penalty starts mild and escalates.
    # ══════════════════════════════════════════════════════════════
    inaction = 0.0
    base_in = getattr(config, "FLIP_INACTION_PENALTY", -0.000001)
    if current_pos == 0:
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

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 3: PORTFOLIO DRAWDOWN PENALTY
    # Drawdowns below DD_FREE_ZONE are ignored — the model needs to
    # accept normal trade risk without being penalised for it.
    # Above the free zone: quadratic penalty protects against blowup.
    # ══════════════════════════════════════════════════════════════
    peak = np.maximum.accumulate(pv)
    dd_frac = min((curr - peak[-1]) / np.maximum(peak[-1], 1e-8), 0.0)
    dd_free = getattr(config, "FLIP_DD_FREE_ZONE", 0.08)
    dd_w = getattr(config, "FLIP_DD_PENALTY_WEIGHT", 2.0)
    dd_pen = -((abs(dd_frac) - dd_free) ** 2) * dd_w if abs(dd_frac) > dd_free else 0.0

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 4: INTRA-TRADE DEEP DRAWDOWN PENALTY
    # Fires when unrealised loss on an open trade exceeds
    # FLIP_INTRA_DD_THRESHOLD × RISK_PER_TRADE (default 80%).
    # Teaches the model that deep floating losses are costly even
    # if they recover — aligns with live risk management behaviour.
    # Example: risk 15%, threshold 80% → penalty fires at 12% loss.
    # ══════════════════════════════════════════════════════════════
    intra_dd_pen = 0.0
    if current_pos != 0:
        thr = getattr(config, "FLIP_INTRA_DD_THRESHOLD", 0.80)
        intra_w = getattr(config, "FLIP_INTRA_DD_WEIGHT", 2.0)
        entry_pv = _scan_entry_pv(history, pv, -1, scan_limit)
        if entry_pv is not None and entry_pv > 0:
            loss = max((entry_pv - curr) / entry_pv, 0.0)
            allowed = risk_per_trade * thr
            if loss > allowed:
                intra_dd_pen = -((loss - allowed) ** 2) * intra_w

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 5: RRR-SCALED QUALITY EXIT BONUS  (enhanced in v3)
    # Fires when model closes to flat (action=0) with a net profit.
    # Scales with how large the move was relative to risk:
    #   1R → tanh(1) ≈ 0.76 × weight
    #   2R → tanh(2) ≈ 0.96 × weight
    #   3R → tanh(3) ≈ 1.00 × weight
    # Weight 0.15 (up from 0.05 in v2) makes flat exits clearly
    # more rewarding than any direct flip could be.
    # ══════════════════════════════════════════════════════════════
    quality_exit = 0.0
    if current_pos == 0 and prev_pos != 0:
        exit_w = getattr(config, "FLIP_QUALITY_EXIT_BONUS", 0.15)
        entry_pv = _scan_entry_pv(history, pv, -2, scan_limit)
        if entry_pv is not None and curr > entry_pv:
            profit_frac = (curr - entry_pv) / entry_pv
            trade_rrr = profit_frac / max(risk_per_trade, 1e-8)
            quality_exit = exit_w * np.tanh(trade_rrr)

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 6: DIRECT FLIP PENALTY  (new in v3)
    # Penalises transitions +1→-1 or -1→+1 without passing through 0.
    # This is the structural fix that forces the model to use flat
    # as the exit action rather than bypassing it.
    # Without this: flip directly = free, go through flat = costs inaction timer.
    # With this:    flip directly = costs flip_pen, go through flat = free.
    # ══════════════════════════════════════════════════════════════
    flip_pen = 0.0
    direct_flip = (prev_pos == 1 and current_pos == -1) or (
        prev_pos == -1 and current_pos == 1
    )
    if direct_flip:
        flip_pen = getattr(config, "FLIP_DIRECT_FLIP_PENALTY", -0.05)

    # ══════════════════════════════════════════════════════════════
    # COMPONENT 7: CONSECUTIVE LOSS STREAK PENALTY
    # Gentle per-step nudge while model has N+ losses in a row.
    # Kept small — normal losing streaks should not cause the model
    # to stop trading entirely. Just enough to discourage chasing
    # the market after repeated losses.
    # ══════════════════════════════════════════════════════════════
    consec_pen = 0.0
    if current_pos != 0:
        streak_t = getattr(config, "FLIP_CONSEC_LOSS_THRESHOLD", 2)
        streak_p = getattr(config, "FLIP_CONSEC_LOSS_PENALTY", -0.02)
        results, pp2, e2 = [], 0, None
        for k in range(-scan_limit, 0):
            idx = len(pv) + k
            if not (0 <= idx < len(pv)):
                continue
            try:
                p = history["position", k]
            except Exception:
                break
            if pp2 == 0 and p != 0:
                e2 = pv[idx]
            elif pp2 != 0 and p == 0 and e2 is not None:
                results.append(pv[idx] - e2)
                e2 = None
            pp2 = p
        streak = 0
        for r in reversed(results):
            if r < 0:
                streak += 1
            else:
                break
        if streak >= streak_t:
            consec_pen = streak_p

    # ══════════════════════════════════════════════════════════════
    # TOTAL
    # Note: FREQUENCY BONUS intentionally absent — it was the root
    # cause of the model never using action=0.
    # ══════════════════════════════════════════════════════════════
    total = (
        base  # C1: asymmetric log return
        + inaction  # C2: graduated flat penalty
        + dd_pen  # C3: portfolio drawdown
        + intra_dd_pen  # C4: intra-trade deep loss
        + quality_exit  # C5: RRR-scaled profitable flat exit
        + flip_pen  # C6: direct flip penalty (NEW)
        + consec_pen  # C7: consecutive loss streak
    )

    return float(np.tanh(total))


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


# ══════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════

_REWARD_FUNCTIONS = {
    "log_return": reward_log_return,
    "sharpe": reward_sharpe,
    "account_flip": reward_account_flip,
    "account_flip_aggressive": reward_account_flip_aggressive,
    "account_flip_aggressive_v2": reward_account_flip_aggressive_v2,
    "account_flip_aggressive_v3": reward_account_flip_aggressive_v3,
    "account_flip_simple": reward_account_flip_simple,
}


def get_reward_function(name: str):
    if name not in _REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Choose from: {list(_REWARD_FUNCTIONS.keys())}"
        )
    return _REWARD_FUNCTIONS[name]
