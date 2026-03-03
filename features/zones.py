"""
features/zones.py
==================
Fair Value Gap, Order Block, Breaker Block, and Liquidity Pool detection.

FVG (Fair Value Gap)
────────────────────
Bullish FVG: candle[i-2].high < candle[i].low  (gap up — unfilled space)
Bearish FVG: candle[i-2].low  > candle[i].high (gap down — unfilled space)
Filled when price closes through the entire gap.

Order Block
────────────────────
OBs are identified AT THE TIP of a confirmed IT or LT swing.

Bullish OB (+OB): identified when an ITL or LTL is confirmed.
    = the last DOWN-close candle at the swing LOW tip with the
      highest body-to-wick ratio (most body, least wick).
    = the candle representing the final institutional selling
      before the bullish displacement that formed the swing low.

Bearish OB (-OB): identified when an ITH or LTH is confirmed.
    = the last UP-close candle at the swing HIGH tip with the
      highest body-to-wick ratio.
    = the final institutional buying before bearish displacement.

Mitigated when price closes THROUGH the entire OB candle body.
    Bullish OB mitigated: close < ob_low
    Bearish OB mitigated: close > ob_high

Breaker Block
────────────────────
The last series of opposing candles that failed to hold price after:
1. Price first took out a previous swing high/low (Stop Run)
2. Then shifted structure in the opposite direction (after a BOS)

A bullish breaker was originally a bearish OB that got broken above.
A bearish breaker was originally a bullish OB that got broken below.

NEW: Liquidity Pool Detection
────────────────────────────────────
Equal highs / equal lows within a tolerance band signal clustered stops.
Price sweeping into these pools creates high-probability reversal setups.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ══════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ══════════════════════════════════════════════════════


def add_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal columns: _fvg_top, _fvg_bottom, _fvg_direction, _fvg_filled_pct
    Feature columns:
        feature_fvg_active          0/1
        feature_fvg_direction       +1/-1
        feature_fvg_rebalance_pct   0.0→1.0 (critical for ITH/ITL qualification)
        feature_fvg_position        0.0→1.0 (where in FVG is price)
        feature_dist_to_fvg         ATR-normalised distance to FVG midpoint
    """
    df = df.copy()
    n = len(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr = df["_atr"].values

    _fvg_top = np.full(n, np.nan)
    _fvg_bot = np.full(n, np.nan)
    _fvg_dir = np.zeros(n)
    _fvg_fill = np.full(n, np.nan)

    fvg_active = np.zeros(n)
    fvg_dir_feat = np.zeros(n)
    fvg_rebal = np.full(n, 0.0)
    fvg_pos = np.full(n, np.nan)
    dist_fvg = np.full(n, np.nan)

    active = []

    for i in range(2, n):

        if low[i] > high[i - 2]:  # bullish gap
            active.append([low[i], high[i - 2], 1, 0.0])
        if high[i] < low[i - 2]:  # bearish gap
            active.append([low[i - 2], high[i], -1, 0.0])

        still_active = []
        for fvg in active:
            t, b, d, _ = fvg
            gap = t - b
            if gap <= 0:
                continue
            if d == 1:
                filled = np.clip((t - close[i]) / gap, 0.0, 1.0)
            else:
                filled = np.clip((close[i] - b) / gap, 0.0, 1.0)
            fvg[3] = filled
            if filled < 1.0:
                still_active.append(fvg)
        active = still_active

        if not active:
            continue

        best = active[-1]
        t, b, d, filled = best

        _fvg_top[i] = t
        _fvg_bot[i] = b
        _fvg_dir[i] = d
        _fvg_fill[i] = filled

        fvg_active[i] = 1
        fvg_dir_feat[i] = d
        fvg_rebal[i] = filled

        gap = t - b
        if gap > 0:
            fvg_pos[i] = np.clip((close[i] - b) / gap, 0.0, 1.0)

        mid = (t + b) / 2
        if atr[i] > 0:
            dist_fvg[i] = (mid - close[i]) / atr[i]

    df["_fvg_top"] = _fvg_top
    df["_fvg_bottom"] = _fvg_bot
    df["_fvg_direction"] = _fvg_dir
    df["_fvg_filled_pct"] = _fvg_fill

    df["feature_fvg_active"] = fvg_active
    df["feature_fvg_direction"] = fvg_dir_feat
    df["feature_fvg_rebalance_pct"] = fvg_rebal
    df["feature_fvg_position"] = fvg_pos
    df["feature_dist_to_fvg"] = dist_fvg

    return df


# ══════════════════════════════════════════════════════
# ORDER BLOCKS
# ══════════════════════════════════════════════════════


def _find_ob_at_swing_tip(
    tip_idx: int,
    direction: int,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    lookback: int = 10,
) -> tuple:
    """
    At the confirmed swing tip (tip_idx), walk backwards and collect the
    CONSECUTIVE run of opposing candles immediately before the tip.

    For a swing HIGH tip (bearish OB):
        Opposing = UP-close candles
        ob_high = highest CLOSE, ob_low = lowest OPEN

    For a swing LOW tip (bullish OB):
        Opposing = DOWN-close candles
        ob_high = highest OPEN, ob_low = lowest CLOSE

    Returns (ob_high, ob_low, start_candle_idx) or (nan, nan, -1).
    """
    search_start = max(0, tip_idx - lookback)
    collected = []

    for j in range(tip_idx, search_start - 1, -1):

        if direction == 1:
            is_opposing = close_arr[j] > open_arr[j]
        else:
            is_opposing = close_arr[j] < open_arr[j]

        if is_opposing:
            collected.append(j)
        else:
            break

    if not collected:
        return np.nan, np.nan, -1

    if direction == 1:
        ob_high = max(close_arr[j] for j in collected)
        ob_low = min(open_arr[j] for j in collected)
    else:
        ob_high = max(open_arr[j] for j in collected)
        ob_low = min(close_arr[j] for j in collected)

    if ob_high <= ob_low:
        return np.nan, np.nan, -1

    start_idx = collected[-1]
    return ob_high, ob_low, start_idx


def add_order_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order Block detection anchored to confirmed IT and LT swing tips.

    Internal: _ob_high, _ob_low
    Features:
        feature_ob_active       0/1
        feature_ob_direction    +1 (bullish) / -1 (bearish)
        feature_ob_position     0.0→1.0 within OB
        feature_ob_50_dist      |position - 0.5|
        feature_ob_correct_side 1=price approaching from correct side
        feature_ob_dist_atr     distance from close to nearest OB edge / ATR
                                NEW: how far price is from actually touching the OB
    """
    required = ["_ith_flag", "_itl_flag", "_ith_price", "_itl_price"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing {r} — run structure detection first.")

    df = df.copy()
    n = len(df)
    open_ = df["open"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["_atr"].values

    ith_flag = df["_ith_flag"].values.astype(bool)
    itl_flag = df["_itl_flag"].values.astype(bool)
    lth_flag = df.get(
        "_lth_flag", pd.Series(np.zeros(n, dtype=bool), index=df.index)
    ).values.astype(bool)
    ltl_flag = df.get(
        "_ltl_flag", pd.Series(np.zeros(n, dtype=bool), index=df.index)
    ).values.astype(bool)

    _ob_high_arr = np.full(n, np.nan)
    _ob_low_arr = np.full(n, np.nan)

    ob_active = np.zeros(n)
    ob_dir_feat = np.zeros(n)
    ob_pos = np.full(n, np.nan)
    ob_50_dist = np.full(n, np.nan)
    ob_correct = np.zeros(n)
    ob_dist_atr = np.full(n, np.nan)  # NEW

    active_obs = []

    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        # When ITH confirmed → Bearish OB at that swing tip
        if ith_flag[i] or lth_flag[i]:
            oh, ol, ob_idx = _find_ob_at_swing_tip(
                tip_idx=i,
                direction=1,
                open_arr=open_,
                close_arr=close,
                high_arr=high,
                low_arr=low,
            )
            if not np.isnan(oh) and oh > ol:
                active_obs.append([oh, ol, -1])

        # When ITL confirmed → Bullish OB at that swing tip
        if itl_flag[i] or ltl_flag[i]:
            oh, ol, ob_idx = _find_ob_at_swing_tip(
                tip_idx=i,
                direction=-1,
                open_arr=open_,
                close_arr=close,
                high_arr=high,
                low_arr=low,
            )
            if not np.isnan(oh) and oh > ol:
                active_obs.append([oh, ol, 1])

        # Mitigate OBs
        still_active = []
        for ob in active_obs:
            oh, ol, od = ob
            if od == 1 and close[i] < ol:
                continue
            if od == -1 and close[i] > oh:
                continue
            still_active.append(ob)
        active_obs = still_active

        if not active_obs:
            continue

        oh, ol, od = active_obs[-1]

        _ob_high_arr[i] = oh
        _ob_low_arr[i] = ol
        ob_active[i] = 1
        ob_dir_feat[i] = od

        rng = oh - ol
        if rng > 0:
            pos = np.clip((close[i] - ol) / rng, 0.0, 1.0)
            ob_pos[i] = pos
            ob_50_dist[i] = abs(pos - 0.5)

        if od == 1:
            ob_correct[i] = 1 if close[i] <= oh else 0
            # Distance from close to bullish OB high (entry edge) — positive = price above OB
            ob_dist_atr[i] = (close[i] - oh) / atr[i]
        else:
            ob_correct[i] = 1 if close[i] >= ol else 0
            # Distance from close to bearish OB low — positive = price below OB
            ob_dist_atr[i] = (ol - close[i]) / atr[i]

    df["_ob_high"] = _ob_high_arr
    df["_ob_low"] = _ob_low_arr
    df["feature_ob_active"] = ob_active
    df["feature_ob_direction"] = ob_dir_feat
    df["feature_ob_position"] = ob_pos
    df["feature_ob_50_dist"] = ob_50_dist
    df["feature_ob_correct_side"] = ob_correct
    df["feature_ob_dist_atr"] = ob_dist_atr  # NEW

    return df


# ══════════════════════════════════════════════════════
# BREAKER BLOCKS
# ══════════════════════════════════════════════════════


def add_breaker_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Breaker Block: former OB zone that price broke through, now acts as
    support (bullish breaker) or resistance (bearish breaker) on return.

    Internal: _bb_high, _bb_low
    Features:
        feature_bb_active       0/1
        feature_bb_direction    +1 / -1
        feature_bb_position     0.0→1.0 within breaker zone
        feature_bb_50_dist      |position - 0.5|
        feature_bb_correct_side 1=price returning from correct side
    """
    if "_ob_high" not in df.columns:
        raise ValueError(
            "Run add_order_block_features before add_breaker_block_features."
        )

    df = df.copy()
    n = len(df)
    close = df["close"].values
    bos_bull = df.get(
        "feature_bos_bull_it", pd.Series(np.zeros(n), index=df.index)
    ).values
    bos_bear = df.get(
        "feature_bos_bear_it", pd.Series(np.zeros(n), index=df.index)
    ).values

    _bb_high_arr = np.full(n, np.nan)
    _bb_low_arr = np.full(n, np.nan)
    bb_active = np.zeros(n)
    bb_dir_feat = np.zeros(n)
    bb_pos = np.full(n, np.nan)
    bb_50_dist = np.full(n, np.nan)
    bb_correct = np.zeros(n)

    active_breakers = []
    former_obs = []
    seen_ob_keys = set()

    ob_high_vals = df["_ob_high"].values
    ob_low_vals = df["_ob_low"].values
    ob_dir_vals = df["feature_ob_direction"].values

    for i in range(1, n):
        c = close[i]

        # Record newly formed OBs
        prev_oh = ob_high_vals[i - 1]
        curr_oh = ob_high_vals[i]
        if not np.isnan(curr_oh) and (np.isnan(prev_oh) or curr_oh != prev_oh):
            ol = ob_low_vals[i]
            od = ob_dir_vals[i]
            key = (round(curr_oh, 6), round(ol, 6), int(od))
            if key not in seen_ob_keys:
                seen_ob_keys.add(key)
                former_obs.append([curr_oh, ol, od])

        # Check if former OB broken → becomes breaker
        remaining = []
        for ob in former_obs:
            oh, ol, od = ob
            if od == -1 and bos_bull[i] == 1 and c > oh:
                active_breakers.append([oh, ol, 1])
            elif od == 1 and bos_bear[i] == 1 and c < ol:
                active_breakers.append([oh, ol, -1])
            else:
                remaining.append(ob)
        former_obs = remaining

        # Remove stale breakers
        active_breakers = [
            b
            for b in active_breakers
            if not (b[2] == 1 and c < b[1] - 2 * df["_atr"].values[i])
            and not (b[2] == -1 and c > b[0] + 2 * df["_atr"].values[i])
        ]

        if not active_breakers:
            continue

        bh, bl, bd = active_breakers[-1]
        _bb_high_arr[i] = bh
        _bb_low_arr[i] = bl
        bb_active[i] = 1
        bb_dir_feat[i] = bd

        rng = bh - bl
        if rng > 0:
            pos = np.clip((c - bl) / rng, 0.0, 1.0)
            bb_pos[i] = pos
            bb_50_dist[i] = abs(pos - 0.5)

        if bd == 1:
            bb_correct[i] = 1 if c <= bh else 0
        else:
            bb_correct[i] = 1 if c >= bl else 0

    df["_bb_high"] = _bb_high_arr
    df["_bb_low"] = _bb_low_arr
    df["feature_bb_active"] = bb_active
    df["feature_bb_direction"] = bb_dir_feat
    df["feature_bb_position"] = bb_pos
    df["feature_bb_50_dist"] = bb_50_dist
    df["feature_bb_correct_side"] = bb_correct

    return df


# ══════════════════════════════════════════════════════
# NEW: LIQUIDITY POOL DETECTION
# ══════════════════════════════════════════════════════


def add_liquidity_pool_features(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Detects clustered equal highs (buy-side liquidity) and equal lows
    (sell-side liquidity) within a rolling window.

    WHY THIS MATTERS FOR 5RRR:
        Liquidity pools ARE targets. Price doesn't move randomly — it moves
        to sweep the stops sitting at equal highs/lows. If you can see that
        there is a large liquidity pool N ATRs above entry, the model has
        explicit information that the trade has room to run.

        - Pool above price = buy stops (short sellers SLs, breakout buyer entries)
        - Pool below price = sell stops (long holders SLs, breakout seller entries)

        When bias is bullish and there is a large buy-side pool above:
            → Price will likely target it → hold the long trade to that level
        When bias is bearish and a sell-side pool exists below:
            → Price will likely sweep it → hold the short trade to that level

    Features:
        feature_liq_above_count   Number of equal highs in recent window.
                                  Higher = larger pool = stronger magnet.
        feature_liq_below_count   Number of equal lows in recent window.

        feature_liq_above_dist    (pool_level - close) / ATR.
                                  Positive = pool is above current price.
                                  Large positive = target is far away (high RRR potential).
        feature_liq_below_dist    (close - pool_level) / ATR.
                                  Positive = pool is below current price.

        feature_liq_above_viable  1 if there is a pool > 3R above (long trade has room).
        feature_liq_below_viable  1 if there is a pool > 3R below (short trade has room).

    Equal high/low tolerance: 1.5× ATR percentage (adaptive to price level).
    A cluster requires at least 2 touches within tolerance to be a valid pool.
    """
    df = df.copy()
    n = len(df)
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    atr = df["_atr"].values

    liq_above_count = np.zeros(n)
    liq_below_count = np.zeros(n)
    liq_above_dist = np.full(n, np.nan)
    liq_below_dist = np.full(n, np.nan)
    liq_above_viable = np.zeros(n)
    liq_below_viable = np.zeros(n)

    # Tolerance: adaptive — use ATR-relative tolerance so it works across
    # different price levels and volatility regimes
    # At EURUSD M15 ATR ≈ 0.0012, tolerance ≈ 0.0003 (3 pips) which is correct
    tol_atr_fraction = getattr(config, "LIQ_POOL_TOL_ATR", 0.25)
    min_rrr_viable = getattr(config, "MIN_RR_RATIO", 3.0)

    for i in range(lookback, n):
        a = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 1e-8
        tol = a * tol_atr_fraction

        window_h = h[i - lookback : i]
        window_l = l[i - lookback : i]

        # Find the most prominent equal-high cluster in the window
        # Strategy: for each candidate high, count how many others are within tol
        best_above_count = 0
        best_above_level = np.nan

        # Only look at highs ABOVE current close (liquidity targets above)
        above_mask = window_h > c[i]
        if above_mask.any():
            candidates = window_h[above_mask]
            for cand in candidates:
                count = int(np.sum(np.abs(window_h - cand) <= tol))
                if count > best_above_count:
                    best_above_count = count
                    best_above_level = cand

        liq_above_count[i] = best_above_count if best_above_count >= 2 else 0
        if best_above_count >= 2 and not np.isnan(best_above_level):
            liq_above_dist[i] = (best_above_level - c[i]) / a
            one_r = c[i] * getattr(config, "RISK_PER_TRADE", 0.0015)
            if one_r > 0:
                r_dist = (best_above_level - c[i]) / one_r
                liq_above_viable[i] = float(r_dist >= min_rrr_viable)

        # Equal-low cluster BELOW current close
        best_below_count = 0
        best_below_level = np.nan

        below_mask = window_l < c[i]
        if below_mask.any():
            candidates = window_l[below_mask]
            for cand in candidates:
                count = int(np.sum(np.abs(window_l - cand) <= tol))
                if count > best_below_count:
                    best_below_count = count
                    best_below_level = cand

        liq_below_count[i] = best_below_count if best_below_count >= 2 else 0
        if best_below_count >= 2 and not np.isnan(best_below_level):
            liq_below_dist[i] = (c[i] - best_below_level) / a
            one_r = c[i] * getattr(config, "RISK_PER_TRADE", 0.0015)
            if one_r > 0:
                r_dist = (c[i] - best_below_level) / one_r
                liq_below_viable[i] = float(r_dist >= min_rrr_viable)

    df["feature_liq_above_count"] = liq_above_count
    df["feature_liq_below_count"] = liq_below_count
    df["feature_liq_above_dist"] = liq_above_dist
    df["feature_liq_below_dist"] = liq_below_dist
    df["feature_liq_above_viable"] = liq_above_viable
    df["feature_liq_below_viable"] = liq_below_viable

    return df


# ══════════════════════════════════════════════════════
# MASTER
# ══════════════════════════════════════════════════════


def add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [zones] Detecting Fair Value Gaps...")
    df = add_fvg_features(df)
    print("  [zones] Detecting Order Blocks...")
    df = add_order_block_features(df)
    print("  [zones] Detecting Breaker Blocks...")
    df = add_breaker_block_features(df)
    print("  [zones] Detecting Liquidity Pools...")
    df = add_liquidity_pool_features(df)
    return df
