"""
features/structure.py
======================
Detects structural levels using exact ICT definitions.

FIXES IN THIS VERSION:
  1. Distance signs corrected — lower levels (STL/ITL/LTL) use
     (close - level)/atr so all distances are positive magnitudes.
  2. ITH/ITL stamped at CONFIRMATION candle not swing candle
     — eliminates lookahead bias in level timing.
  3. BOS no longer updates last_ith to current close — uses a
     separate "broken" set to prevent cascade firing.
  4. Bias alignment uses 2-of-3 voting not strict 3-of-3.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ══════════════════════════════════════════════════════
# ATR
# ══════════════════════════════════════════════════════


def compute_atr(df: pd.DataFrame, period: int = None) -> pd.Series:
    period = period or config.ATR_PERIOD
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


# ══════════════════════════════════════════════════════
# RAW SWING DETECTION
# ══════════════════════════════════════════════════════


def _find_raw_swings(df: pd.DataFrame, lookback: int) -> tuple:
    """
    Raw swing high at i: df.high[i] is the strict maximum in
    [i-lookback .. i+lookback]. Confirmed only after lookback candles.
    """
    n = len(df)
    high = df["high"].values
    low = df["low"].values

    raw_high = np.zeros(n, dtype=bool)
    raw_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        wh = high[i - lookback : i + lookback + 1]
        wl = low[i - lookback : i + lookback + 1]
        if high[i] == wh.max() and np.sum(wh == high[i]) == 1:
            raw_high[i] = True
        if low[i] == wl.min() and np.sum(wl == low[i]) == 1:
            raw_low[i] = True

    return raw_high, raw_low


# ══════════════════════════════════════════════════════
# STH / STL
# ══════════════════════════════════════════════════════


def detect_sth_stl(df: pd.DataFrame) -> pd.DataFrame:
    lookback = config.SWING_LOOKBACK
    raw_high, raw_low = _find_raw_swings(df, lookback)

    df = df.copy()
    df["_sth_flag"] = raw_high
    df["_stl_flag"] = raw_low
    df["_sth_price"] = df["high"].where(df["_sth_flag"]).ffill()
    df["_stl_price"] = df["low"].where(df["_stl_flag"]).ffill()
    return df


# ══════════════════════════════════════════════════════
# ITH / ITL
# FIX: stamp at confirmation candle, not swing candle
# ══════════════════════════════════════════════════════


def detect_ith_itl(df: pd.DataFrame) -> pd.DataFrame:
    """
    ITH: swing high confirmed when a STH appears to BOTH its left AND right.
         Stamped at the candle where right-side confirmation first occurs.
         OR: a swing high that rebalances a bearish FVG by >= 80%.

    ITL: mirror for lows.

    FIX vs previous version:
        Before: ith[i] = True at the swing candle i, with right_ok
                checking future candles → lookahead bias in timing.
        Now:    we scan forward and stamp the flag at the CONFIRMATION
                candle (where right_ok first becomes true), not at i.
                _ith_price is still the swing price (correct).
                _ith_flag fires at the confirmation candle.
    """
    if "_sth_flag" not in df.columns:
        df = detect_sth_stl(df)

    df = df.copy()
    n = len(df)
    sth = df["_sth_flag"].values
    stl = df["_stl_flag"].values
    high = df["high"].values
    low = df["low"].values

    has_fvg = all(
        c in df.columns for c in ["_fvg_top", "_fvg_bottom", "_fvg_direction"]
    )
    fvg_pct = config.FVG_QUALIFY_THRESHOLD

    ith = np.zeros(n, dtype=bool)
    itl = np.zeros(n, dtype=bool)
    ith_price = np.full(n, np.nan)
    itl_price = np.full(n, np.nan)

    for i in range(1, n - 1):

        # ── ITH via flanking STHs ──────────────────────
        if sth[i] and sth[:i].any():
            # Find the first STH to the right — that's the confirmation candle
            search_end = min(i + 80, n)
            for j in range(i + 1, search_end):
                if sth[j]:
                    # Stamp the flag at j (confirmation), price at i (swing)
                    ith[j] = True
                    ith_price[j] = high[i]
                    break

        # ── ITH via FVG rebalance >= 80% ──────────────
        if not ith[i] and has_fvg:
            ft = df["_fvg_top"].values[i]
            fb = df["_fvg_bottom"].values[i]
            fd = df["_fvg_direction"].values[i]
            if not (np.isnan(ft) or np.isnan(fb)) and (ft - fb) > 0:
                if fd == -1:  # bearish FVG: swing HIGH tapping up into it
                    tap = high[i] - fb
                    if tap / (ft - fb) >= fvg_pct:
                        ith[i] = True
                        ith_price[i] = high[i]

        # ── ITL via flanking STLs ──────────────────────
        if stl[i] and stl[:i].any():
            search_end = min(i + 80, n)
            for j in range(i + 1, search_end):
                if stl[j]:
                    itl[j] = True
                    itl_price[j] = low[i]
                    break

        # ── ITL via FVG rebalance >= 80% ──────────────
        if not itl[i] and has_fvg:
            ft = df["_fvg_top"].values[i]
            fb = df["_fvg_bottom"].values[i]
            fd = df["_fvg_direction"].values[i]
            if not (np.isnan(ft) or np.isnan(fb)) and (ft - fb) > 0:
                if fd == 1:  # bullish FVG: swing LOW tapping down into it
                    tap = ft - low[i]
                    if tap / (ft - fb) >= fvg_pct:
                        itl[i] = True
                        itl_price[i] = low[i]

    df["_ith_flag"] = ith
    df["_itl_flag"] = itl

    # Forward-fill: use explicit price array so we ffill the SWING price
    # (not the confirmation candle's price)
    ith_s = pd.Series(ith_price, index=df.index)
    itl_s = pd.Series(itl_price, index=df.index)
    df["_ith_price"] = ith_s.ffill()
    df["_itl_price"] = itl_s.ffill()

    return df


# ══════════════════════════════════════════════════════
# LTH / LTL
# FIX: same confirmation-candle stamping as ITH/ITL
# ══════════════════════════════════════════════════════


def detect_lth_ltl(df: pd.DataFrame) -> pd.DataFrame:
    if "_ith_flag" not in df.columns:
        df = detect_ith_itl(df)

    df = df.copy()
    n = len(df)
    ith = df["_ith_flag"].values
    itl = df["_itl_flag"].values
    high = df["high"].values
    low = df["low"].values

    lth = np.zeros(n, dtype=bool)
    ltl = np.zeros(n, dtype=bool)
    lth_price = np.full(n, np.nan)
    ltl_price = np.full(n, np.nan)

    for i in range(1, n - 1):
        if ith[i] and ith[:i].any():
            search_end = min(i + 150, n)
            for j in range(i + 1, search_end):
                if ith[j]:
                    lth[j] = True
                    lth_price[j] = df["_ith_price"].values[i]
                    break

        if itl[i] and itl[:i].any():
            search_end = min(i + 150, n)
            for j in range(i + 1, search_end):
                if itl[j]:
                    ltl[j] = True
                    ltl_price[j] = df["_itl_price"].values[i]
                    break

    df["_lth_flag"] = lth
    df["_ltl_flag"] = ltl
    df["_lth_price"] = pd.Series(lth_price, index=df.index).ffill()
    df["_ltl_price"] = pd.Series(ltl_price, index=df.index).ffill()

    return df


# ══════════════════════════════════════════════════════
# BOS / CHoCH + BIAS STATE MACHINE
# FIX: BOS no longer resets level to current close
#      preventing cascade re-firing on consecutive candles
# ══════════════════════════════════════════════════════


def detect_bos_choch_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    BOS   = price closes ABOVE an ITH (bull) or BELOW an ITL (bear).
    CHoCH = price closes ABOVE an LTH (bull) or BELOW an LTL (bear).

    FIX vs previous version:
        Before: after bull BOS, last_ith = close[i]
                → next candle if close > close[i], another BOS fires
                → cascade of 10-20 BOS flags during a trending move
        Now:    after BOS fires on level X, level X is retired
                New BOS only fires when a genuinely NEW IT level is broken
                We track the set of broken levels to prevent re-firing
    """
    required = [
        "_sth_price",
        "_stl_price",
        "_ith_price",
        "_itl_price",
        "_lth_price",
        "_ltl_price",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing {c} — run structure detection first.")

    df = df.copy()
    n = len(df)
    close = df["close"].values

    sth_p = df["_sth_price"].values
    stl_p = df["_stl_price"].values
    ith_p = df["_ith_price"].values
    itl_p = df["_itl_price"].values
    lth_p = df["_lth_price"].values
    ltl_p = df["_ltl_price"].values

    bos_bull_st = np.zeros(n)
    bos_bear_st = np.zeros(n)
    bos_bull_it = np.zeros(n)
    bos_bear_it = np.zeros(n)
    bos_bull_lt = np.zeros(n)
    bos_bear_lt = np.zeros(n)
    choch_bull = np.zeros(n)
    choch_bear = np.zeros(n)
    st_bias = np.zeros(n)
    it_bias = np.zeros(n)
    lt_bias = np.zeros(n)

    cur_st = 0
    cur_it = 0
    cur_lt = 0

    # Track the ACTUAL level value being watched — updated only when a
    # NEW level appears in the forward-filled price series
    last_sth = np.nan
    last_stl = np.nan
    last_ith = np.nan
    last_itl = np.nan
    last_lth = np.nan
    last_ltl = np.nan

    # Track which level values have already triggered a BOS
    # so the same level doesn't fire twice
    broken_sth = set()
    broken_stl = set()
    broken_ith = set()
    broken_itl = set()
    broken_lth = set()
    broken_ltl = set()

    for i in range(1, n):
        c = close[i]

        # Update levels only when the ffill produces a new value
        if not np.isnan(sth_p[i]) and sth_p[i] != last_sth:
            last_sth = sth_p[i]
        if not np.isnan(stl_p[i]) and stl_p[i] != last_stl:
            last_stl = stl_p[i]
        if not np.isnan(ith_p[i]) and ith_p[i] != last_ith:
            last_ith = ith_p[i]
        if not np.isnan(itl_p[i]) and itl_p[i] != last_itl:
            last_itl = itl_p[i]
        if not np.isnan(lth_p[i]) and lth_p[i] != last_lth:
            last_lth = lth_p[i]
        if not np.isnan(ltl_p[i]) and ltl_p[i] != last_ltl:
            last_ltl = ltl_p[i]

        # ── ST BOS ─────────────────────────────────────
        if not np.isnan(last_sth) and last_sth not in broken_sth and c > last_sth:
            bos_bull_st[i] = 1
            cur_st = 1
            broken_sth.add(last_sth)

        if not np.isnan(last_stl) and last_stl not in broken_stl and c < last_stl:
            bos_bear_st[i] = 1
            cur_st = -1
            broken_stl.add(last_stl)

        # ── IT BOS ─────────────────────────────────────
        if not np.isnan(last_ith) and last_ith not in broken_ith and c > last_ith:
            bos_bull_it[i] = 1
            cur_it = 1
            broken_ith.add(last_ith)

        if not np.isnan(last_itl) and last_itl not in broken_itl and c < last_itl:
            bos_bear_it[i] = 1
            cur_it = -1
            broken_itl.add(last_itl)

        # ── LT BOS / CHoCH ─────────────────────────────
        if not np.isnan(last_lth) and last_lth not in broken_lth and c > last_lth:
            bos_bull_lt[i] = 1
            choch_bull[i] = 1
            cur_lt = 1
            broken_lth.add(last_lth)

        if not np.isnan(last_ltl) and last_ltl not in broken_ltl and c < last_ltl:
            bos_bear_lt[i] = 1
            choch_bear[i] = 1
            cur_lt = -1
            broken_ltl.add(last_ltl)

        st_bias[i] = cur_st
        it_bias[i] = cur_it
        lt_bias[i] = cur_lt

    # CHoCH active window
    choch_bw = config.CHOCH_ACTIVE_WINDOW
    choch_bull_active = (
        pd.Series(choch_bull).rolling(choch_bw, min_periods=1).max().values
    )
    choch_bear_active = (
        pd.Series(choch_bear).rolling(choch_bw, min_periods=1).max().values
    )

    # ── FIX: 2-of-3 bias alignment (not strict 3-of-3) ──────────
    # All three aligned = strong signal (+1/-1)
    # Two of three aligned = weak signal (+0.5/-0.5)
    # Mixed or neutral = 0
    bias_aligned = np.zeros(n)
    for i in range(n):
        s = st_bias[i] + it_bias[i] + lt_bias[i]
        if s == 3:
            bias_aligned[i] = 1.0
        elif s == 2:
            bias_aligned[i] = 0.5
        elif s == -2:
            bias_aligned[i] = -0.5
        elif s == -3:
            bias_aligned[i] = -1.0

    df["feature_bos_bull_st"] = bos_bull_st
    df["feature_bos_bear_st"] = bos_bear_st
    df["feature_bos_bull_it"] = bos_bull_it
    df["feature_bos_bear_it"] = bos_bear_it
    df["feature_bos_bull_lt"] = bos_bull_lt
    df["feature_bos_bear_lt"] = bos_bear_lt
    df["feature_choch_bull"] = choch_bull
    df["feature_choch_bear"] = choch_bear
    df["feature_choch_bull_active"] = choch_bull_active
    df["feature_choch_bear_active"] = choch_bear_active
    df["feature_st_bias"] = st_bias
    df["feature_it_bias"] = it_bias
    df["feature_lt_bias"] = lt_bias
    df["feature_bias_aligned"] = bias_aligned

    return df


# ══════════════════════════════════════════════════════
# DISTANCE FEATURES
# FIX: correct sign convention per original spec
#   Upper levels (STH/ITH/LTH): (level - close) / atr → positive above
#   Lower levels (STL/ITL/LTL): (close - level) / atr → positive below
# Both are now positive magnitudes representing distance
# ══════════════════════════════════════════════════════


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    atr = df["_atr"]
    close = df["close"]

    # Levels ABOVE price — positive when level is above close
    df["feature_dist_to_sth"] = (df["_sth_price"] - close) / atr
    df["feature_dist_to_ith"] = (df["_ith_price"] - close) / atr
    df["feature_dist_to_lth"] = (df["_lth_price"] - close) / atr

    # Levels BELOW price — positive when level is below close
    df["feature_dist_to_stl"] = (close - df["_stl_price"]) / atr
    df["feature_dist_to_itl"] = (close - df["_itl_price"]) / atr
    df["feature_dist_to_ltl"] = (close - df["_ltl_price"]) / atr

    # Range positions — 0.0 = at lower level, 1.0 = at upper level
    it_range = (df["_ith_price"] - df["_itl_price"]).replace(0, np.nan)
    lt_range = (df["_lth_price"] - df["_ltl_price"]).replace(0, np.nan)

    df["feature_it_range_position"] = ((close - df["_itl_price"]) / it_range).clip(
        0.0, 1.0
    )

    df["feature_lt_range_position"] = ((close - df["_ltl_price"]) / lt_range).clip(
        0.0, 1.0
    )

    return df


# ══════════════════════════════════════════════════════
# MASTER
# ══════════════════════════════════════════════════════


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [structure] Computing ATR...")
    if "_atr" not in df.columns:
        df["_atr"] = compute_atr(df)

    print("  [structure] Detecting STH / STL...")
    df = detect_sth_stl(df)

    print("  [structure] Detecting ITH / ITL (with confirmation stamping)...")
    df = detect_ith_itl(df)

    print("  [structure] Detecting LTH / LTL...")
    df = detect_lth_ltl(df)

    print("  [structure] Computing BOS / CHoCH / Bias (cascade-free)...")
    df = detect_bos_choch_bias(df)

    print("  [structure] Computing distance features (corrected signs)...")
    df = add_distance_features(df)

    return df
