"""
features/zones.py
==================
Fair Value Gap, Order Block, and Breaker Block detection.

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
    df    = df.copy()
    n     = len(df)
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    atr   = df["_atr"].values

    # Internal arrays
    _fvg_top   = np.full(n, np.nan)
    _fvg_bot   = np.full(n, np.nan)
    _fvg_dir   = np.zeros(n)
    _fvg_fill  = np.full(n, np.nan)

    # Feature arrays
    fvg_active   = np.zeros(n)
    fvg_dir_feat = np.zeros(n)
    fvg_rebal    = np.full(n, 0.0)
    fvg_pos      = np.full(n, np.nan)
    dist_fvg     = np.full(n, np.nan)

    # Active FVG list: [top, bottom, direction, filled_pct]
    active = []

    for i in range(2, n):

        # ── Detect new FVG ──────────────────────────────
        if low[i] > high[i - 2]:           # bullish gap
            active.append([low[i], high[i - 2], 1, 0.0])
        if high[i] < low[i - 2]:           # bearish gap
            active.append([low[i - 2], high[i], -1, 0.0])

        # ── Update fill on all active FVGs ──────────────
        still_active = []
        for fvg in active:
            t, b, d, _ = fvg
            gap = t - b
            if gap <= 0:
                continue
            if d == 1:    # bullish: track how much price has come back down
                filled = np.clip((t - close[i]) / gap, 0.0, 1.0)
            else:          # bearish: track how much price has come back up
                filled = np.clip((close[i] - b) / gap, 0.0, 1.0)
            fvg[3] = filled
            if filled < 1.0:
                still_active.append(fvg)
        active = still_active

        if not active:
            continue

        # Use the most recently created active FVG
        best = active[-1]
        t, b, d, filled = best

        _fvg_top[i]  = t
        _fvg_bot[i]  = b
        _fvg_dir[i]  = d
        _fvg_fill[i] = filled

        fvg_active[i]   = 1
        fvg_dir_feat[i] = d
        fvg_rebal[i]    = filled

        gap = t - b
        if gap > 0:
            fvg_pos[i] = np.clip((close[i] - b) / gap, 0.0, 1.0)

        mid = (t + b) / 2
        if atr[i] > 0:
            dist_fvg[i] = (mid - close[i]) / atr[i]

    df["_fvg_top"]             = _fvg_top
    df["_fvg_bottom"]          = _fvg_bot
    df["_fvg_direction"]       = _fvg_dir
    df["_fvg_filled_pct"]      = _fvg_fill

    df["feature_fvg_active"]        = fvg_active
    df["feature_fvg_direction"]     = fvg_dir_feat
    df["feature_fvg_rebalance_pct"] = fvg_rebal
    df["feature_fvg_position"]      = fvg_pos
    df["feature_dist_to_fvg"]       = dist_fvg

    return df


# ══════════════════════════════════════════════════════
# ORDER BLOCKS
# ══════════════════════════════════════════════════════

def _find_ob_at_swing_tip(
    tip_idx   : int,
    direction : int,        # +1 = swing HIGH tip → bearish OB (look for up-close candles)
                            # -1 = swing LOW  tip → bullish OB (look for down-close candles)
    open_arr  : np.ndarray,
    close_arr : np.ndarray,
    high_arr  : np.ndarray,
    low_arr   : np.ndarray,
    lookback  : int = 10,
) -> tuple:
    """
    At the confirmed swing tip (tip_idx), walk backwards and collect the
    CONSECUTIVE run of opposing candles immediately before the tip.

    Rule:
        If there are MULTIPLE consecutive opposing candles at the tip
        → the entire series forms the OB zone
          (ob_high = highest body top, ob_low = lowest body bottom)

        If there is only ONE opposing candle at the tip
        → that single candle is the OB zone

    "Consecutive" means an unbroken run of same-direction candles
    walking back from the tip with no interruption.

    For a swing HIGH tip (bearish OB):
        Opposing candles = UP-close candles (close > open)
        Walk back from tip_idx and collect every UP-close candle
        until the run is broken by a DOWN-close candle.
        OB zone:
            ob_high = highest CLOSE among all collected candles (top of bodies)
            ob_low  = lowest  OPEN  among all collected candles (bottom of bodies)

    For a swing LOW tip (bullish OB):
        Opposing candles = DOWN-close candles (close < open)
        Walk back from tip_idx and collect every DOWN-close candle
        until the run is broken by an UP-close candle.
        OB zone:
            ob_high = highest OPEN  among all collected candles (top of bodies)
            ob_low  = lowest  CLOSE among all collected candles (bottom of bodies)

    Returns (ob_high, ob_low, start_candle_idx) or (nan, nan, -1) if not found.

    Example — bearish OB with 3 consecutive up-close candles at swing HIGH:
        candle i-3: close > open  ← start of run (oldest)
        candle i-2: close > open
        candle i-1: close > open  ← end of run (most recent, at tip)
        candle i  : the tip itself (down-close that caused the high)
        → OB zone spans all 3 candles
        → ob_high = max(close[i-3], close[i-2], close[i-1])
        → ob_low  = min(open[i-3],  open[i-2],  open[i-1])

    Example — bullish OB with 1 down-close candle at swing LOW:
        candle j-1: close > open  ← different direction, breaks run
        candle j  : close < open  ← only this one qualifies
        candle j+1: the tip low
        → OB = just candle j
        → ob_high = open[j], ob_low = close[j]
    """
    search_start = max(0, tip_idx - lookback)
    collected    = []   # indices of consecutive opposing candles

    # Walk backwards from the tip collecting consecutive opposing candles
    for j in range(tip_idx, search_start - 1, -1):

        if direction == 1:      # swing HIGH → look for UP-close candles
            is_opposing = close_arr[j] > open_arr[j]
        else:                   # swing LOW  → look for DOWN-close candles
            is_opposing = close_arr[j] < open_arr[j]

        if is_opposing:
            collected.append(j)
        else:
            # Run broken — stop immediately
            # (we only want the CONSECUTIVE run right at the tip)
            break

    if not collected:
        return np.nan, np.nan, -1

    # ── Compute OB zone boundaries from the collected candles ──
    if direction == 1:      # bearish OB — bodies of UP-close candles
        ob_high = max(close_arr[j] for j in collected)   # top of bodies
        ob_low  = min(open_arr[j]  for j in collected)   # bottom of bodies
    else:                   # bullish OB — bodies of DOWN-close candles
        ob_high = max(open_arr[j]  for j in collected)   # top of bodies
        ob_low  = min(close_arr[j] for j in collected)   # bottom of bodies

    # Safety check
    if ob_high <= ob_low:
        return np.nan, np.nan, -1

    start_idx = collected[-1]   # oldest candle in the run
    return ob_high, ob_low, start_idx


def add_order_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order Block detection anchored to confirmed IT and LT swing tips.

    TRIGGER: when an ITH, ITL, LTH, or LTL is confirmed, we immediately
    look back at the candles that formed that swing tip and identify the
    Order Block there.

    Bullish OB (+OB):
        Confirmed at an ITL or LTL tip.
        = last down-close candle at the swing LOW with best body-to-wick ratio.
        This is the candle that represents the last institutional selling
        before the bullish displacement that formed the swing low.

    Bearish OB (-OB):
        Confirmed at an ITH or LTH tip.
        = last up-close candle at the swing HIGH with best body-to-wick ratio.
        This is the last institutional buying before the bearish displacement.

    Mitigated: when price closes THROUGH the entire OB candle body.
        Bullish OB mitigated: close < ob_low
        Bearish OB mitigated: close > ob_high

    Internal: _ob_high, _ob_low
    Features:
        feature_ob_active       0/1
        feature_ob_direction    +1 (bullish) / -1 (bearish)
        feature_ob_position     0.0 (at OB low) → 1.0 (at OB high)
        feature_ob_50_dist      |position - 0.5|, near 0.0 = ideal entry
        feature_ob_correct_side 1 = price approaching from correct side
    """
    required = ["_ith_flag", "_itl_flag", "_ith_price", "_itl_price"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing {r} — run structure detection first.")

    df    = df.copy()
    n     = len(df)
    open_ = df["open"].values
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    atr   = df["_atr"].values

    # Structure flags — OBs are anchored to these confirmations
    ith_flag = df["_ith_flag"].values.astype(bool)
    itl_flag = df["_itl_flag"].values.astype(bool)
    lth_flag = df.get("_lth_flag", pd.Series(np.zeros(n, dtype=bool), index=df.index)).values.astype(bool)
    ltl_flag = df.get("_ltl_flag", pd.Series(np.zeros(n, dtype=bool), index=df.index)).values.astype(bool)

    _ob_high_arr = np.full(n, np.nan)
    _ob_low_arr  = np.full(n, np.nan)

    ob_active    = np.zeros(n)
    ob_dir_feat  = np.zeros(n)
    ob_pos       = np.full(n, np.nan)
    ob_50_dist   = np.full(n, np.nan)
    ob_correct   = np.zeros(n)

    # Active OBs: list of [ob_high, ob_low, direction (+1/-1)]
    active_obs = []

    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        # ── Step 1: Check if a new IT or LT swing tip was confirmed ──
        # When ITH confirmed → Bearish OB at that swing tip
        if ith_flag[i] or lth_flag[i]:
            oh, ol, ob_idx = _find_ob_at_swing_tip(
                tip_idx   = i,
                direction = 1,          # swing HIGH → bearish OB
                open_arr  = open_,
                close_arr = close,
                high_arr  = high,
                low_arr   = low,
            )
            if not np.isnan(oh) and oh > ol:
                active_obs.append([oh, ol, -1])   # -1 = bearish OB

        # When ITL confirmed → Bullish OB at that swing tip
        if itl_flag[i] or ltl_flag[i]:
            oh, ol, ob_idx = _find_ob_at_swing_tip(
                tip_idx   = i,
                direction = -1,         # swing LOW → bullish OB
                open_arr  = open_,
                close_arr = close,
                high_arr  = high,
                low_arr   = low,
            )
            if not np.isnan(oh) and oh > ol:
                active_obs.append([oh, ol, 1])    # +1 = bullish OB

        # ── Step 2: Mitigate OBs that price closed through ──────────
        # Bullish OB: mitigated when close BELOW the OB low (institution failed)
        # Bearish OB: mitigated when close ABOVE the OB high
        still_active = []
        for ob in active_obs:
            oh, ol, od = ob
            if od == 1  and close[i] < ol:  continue   # bullish OB mitigated
            if od == -1 and close[i] > oh:  continue   # bearish OB mitigated
            still_active.append(ob)
        active_obs = still_active

        if not active_obs:
            continue

        # ── Step 3: Use the most recently formed active OB ──────────
        oh, ol, od = active_obs[-1]

        _ob_high_arr[i] = oh
        _ob_low_arr[i]  = ol
        ob_active[i]    = 1
        ob_dir_feat[i]  = od

        rng = oh - ol
        if rng > 0:
            pos = np.clip((close[i] - ol) / rng, 0.0, 1.0)
            ob_pos[i]     = pos
            ob_50_dist[i] = abs(pos - 0.5)

        # Correct side:
        #   Bullish OB (+1): price should be BELOW or ENTERING from below
        #   Bearish OB (-1): price should be ABOVE or ENTERING from above
        if od == 1:
            ob_correct[i] = 1 if close[i] <= oh else 0
        else:
            ob_correct[i] = 1 if close[i] >= ol else 0

    df["_ob_high"]                = _ob_high_arr
    df["_ob_low"]                 = _ob_low_arr
    df["feature_ob_active"]       = ob_active
    df["feature_ob_direction"]    = ob_dir_feat
    df["feature_ob_position"]     = ob_pos
    df["feature_ob_50_dist"]      = ob_50_dist
    df["feature_ob_correct_side"] = ob_correct

    return df


# ══════════════════════════════════════════════════════
# BREAKER BLOCKS
# ══════════════════════════════════════════════════════

def add_breaker_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Breaker Block: the last series of opposing candles that failed to hold
    price after a stop run (sweeping a previous high/low) followed by
    a structure shift (BOS in the opposite direction).

    Specifically:
    - A bullish breaker was originally a bearish OB zone that price
      broke ABOVE, turning it into support on a return.
    - A bearish breaker was originally a bullish OB zone that price
      broke BELOW, turning it into resistance on a return.

    This happens after a BOS — so we look for former OB zones that
    price has since breached in the opposite direction.

    Internal: _bb_high, _bb_low
    Features:
        feature_bb_active       0/1
        feature_bb_direction    +1 (bullish breaker) / -1 (bearish breaker)
        feature_bb_position     0.0→1.0 within breaker zone
        feature_bb_50_dist      |position - 0.5|
        feature_bb_correct_side 1=price returning from correct side
    """
    if "_ob_high" not in df.columns:
        raise ValueError("Run add_order_block_features before add_breaker_block_features.")

    df    = df.copy()
    n     = len(df)
    close = df["close"].values
    bos_bull = df.get("feature_bos_bull_it", pd.Series(np.zeros(n), index=df.index)).values
    bos_bear = df.get("feature_bos_bear_it", pd.Series(np.zeros(n), index=df.index)).values

    _bb_high_arr = np.full(n, np.nan)
    _bb_low_arr  = np.full(n, np.nan)
    bb_active    = np.zeros(n)
    bb_dir_feat  = np.zeros(n)
    bb_pos       = np.full(n, np.nan)
    bb_50_dist   = np.full(n, np.nan)
    bb_correct   = np.zeros(n)

    # Track former OBs that have been broken → become breakers
    # Format: [high, low, direction_as_breaker]
    active_breakers = []
    former_obs      = []   # OBs we'll monitor for being broken

    ob_high_vals = df["_ob_high"].values
    ob_low_vals  = df["_ob_low"].values
    ob_dir_vals  = df["feature_ob_direction"].values

    # Track which OB zones have been seen — use a dict keyed by
    # (rounded high, rounded low, direction) to avoid float equality issues
    # and prevent the same OB being added on every active candle
    seen_ob_keys = set()

    for i in range(1, n):
        c = close[i]

        # ── Record newly formed OBs — only on transition candles ──
        # A new OB appears when _ob_high changes value from previous candle
        if i >= 1:
            prev_oh = ob_high_vals[i - 1]
            curr_oh = ob_high_vals[i]
            # New OB if high changed OR first appearance
            if not np.isnan(curr_oh) and (np.isnan(prev_oh) or curr_oh != prev_oh):
                ol = ob_low_vals[i]
                od = ob_dir_vals[i]
                # Round to 6 decimal places for stable key
                key = (round(curr_oh, 6), round(ol, 6), int(od))
                if key not in seen_ob_keys:
                    seen_ob_keys.add(key)
                    former_obs.append([curr_oh, ol, od])

        # ── Check if any former OB has been broken (becomes breaker) ──
        remaining = []
        for ob in former_obs:
            oh, ol, od = ob
            if od == -1 and bos_bull[i] == 1 and c > oh:
                # Bearish OB broken to upside → bullish breaker
                active_breakers.append([oh, ol, 1])
            elif od == 1 and bos_bear[i] == 1 and c < ol:
                # Bullish OB broken to downside → bearish breaker
                active_breakers.append([oh, ol, -1])
            else:
                remaining.append(ob)
        former_obs = remaining

        # ── Remove breakers price has moved far past ──
        active_breakers = [
            b for b in active_breakers
            if not (b[2] == 1  and c < b[1] - 2 * df["_atr"].values[i])
            and not (b[2] == -1 and c > b[0] + 2 * df["_atr"].values[i])
        ]

        if not active_breakers:
            continue

        bh, bl, bd = active_breakers[-1]
        _bb_high_arr[i] = bh
        _bb_low_arr[i]  = bl
        bb_active[i]    = 1
        bb_dir_feat[i]  = bd

        rng = bh - bl
        if rng > 0:
            pos = np.clip((c - bl) / rng, 0.0, 1.0)
            bb_pos[i]     = pos
            bb_50_dist[i] = abs(pos - 0.5)

        if bd == 1:   # bullish breaker — price should be at or below it
            bb_correct[i] = 1 if c <= bh else 0
        else:
            bb_correct[i] = 1 if c >= bl else 0

    df["_bb_high"]               = _bb_high_arr
    df["_bb_low"]                = _bb_low_arr
    df["feature_bb_active"]      = bb_active
    df["feature_bb_direction"]   = bb_dir_feat
    df["feature_bb_position"]    = bb_pos
    df["feature_bb_50_dist"]     = bb_50_dist
    df["feature_bb_correct_side"]= bb_correct

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
    return df
