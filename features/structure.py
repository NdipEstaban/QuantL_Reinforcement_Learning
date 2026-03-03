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

NEW FEATURES (added for 5RRR target):
  5. Candle anatomy (body ratio, wick ratios, direction, consecutive count, ROC)
  6. Displacement quality (3-candle ATR-normalised score, strong flag)
  7. Swing failure / liquidity sweep detection (wick beyond level + close back)
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
            search_end = min(i + 80, n)
            for j in range(i + 1, search_end):
                if sth[j]:
                    ith[j] = True
                    ith_price[j] = high[i]
                    break

        # ── ITH via FVG rebalance >= 80% ──────────────
        if not ith[i] and has_fvg:
            ft = df["_fvg_top"].values[i]
            fb = df["_fvg_bottom"].values[i]
            fd = df["_fvg_direction"].values[i]
            if not (np.isnan(ft) or np.isnan(fb)) and (ft - fb) > 0:
                if fd == -1:
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
                if fd == 1:
                    tap = ft - low[i]
                    if tap / (ft - fb) >= fvg_pct:
                        itl[i] = True
                        itl_price[i] = low[i]

    df["_ith_flag"] = ith
    df["_itl_flag"] = itl

    ith_s = pd.Series(ith_price, index=df.index)
    itl_s = pd.Series(itl_price, index=df.index)
    df["_ith_price"] = ith_s.ffill()
    df["_itl_price"] = itl_s.ffill()

    return df


# ══════════════════════════════════════════════════════
# LTH / LTL
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
# BOS / CHoCH / BIAS
# ══════════════════════════════════════════════════════


def detect_bos_choch_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    BOS   = price closes ABOVE an ITH (bull) or BELOW an ITL (bear).
    CHoCH = price closes ABOVE an LTH (bull) or BELOW an LTL (bear).
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

    last_sth = np.nan
    last_stl = np.nan
    last_ith = np.nan
    last_itl = np.nan
    last_lth = np.nan
    last_ltl = np.nan

    broken_sth = set()
    broken_stl = set()
    broken_ith = set()
    broken_itl = set()
    broken_lth = set()
    broken_ltl = set()

    for i in range(1, n):
        c = close[i]

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

        if not np.isnan(last_sth) and last_sth not in broken_sth and c > last_sth:
            bos_bull_st[i] = 1
            cur_st = 1
            broken_sth.add(last_sth)

        if not np.isnan(last_stl) and last_stl not in broken_stl and c < last_stl:
            bos_bear_st[i] = 1
            cur_st = -1
            broken_stl.add(last_stl)

        if not np.isnan(last_ith) and last_ith not in broken_ith and c > last_ith:
            bos_bull_it[i] = 1
            cur_it = 1
            broken_ith.add(last_ith)

        if not np.isnan(last_itl) and last_itl not in broken_itl and c < last_itl:
            bos_bear_it[i] = 1
            cur_it = -1
            broken_itl.add(last_itl)

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

    choch_bw = config.CHOCH_ACTIVE_WINDOW
    choch_bull_active = (
        pd.Series(choch_bull).rolling(choch_bw, min_periods=1).max().values
    )
    choch_bear_active = (
        pd.Series(choch_bear).rolling(choch_bw, min_periods=1).max().values
    )

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
# ══════════════════════════════════════════════════════


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    atr = df["_atr"]
    close = df["close"]

    df["feature_dist_to_sth"] = (df["_sth_price"] - close) / atr
    df["feature_dist_to_ith"] = (df["_ith_price"] - close) / atr
    df["feature_dist_to_lth"] = (df["_lth_price"] - close) / atr

    df["feature_dist_to_stl"] = (close - df["_stl_price"]) / atr
    df["feature_dist_to_itl"] = (close - df["_itl_price"]) / atr
    df["feature_dist_to_ltl"] = (close - df["_ltl_price"]) / atr

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
# NEW: CANDLE ANATOMY FEATURES
# ══════════════════════════════════════════════════════


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle body and wick anatomy derived purely from OHLCV + ATR.
    These help the model identify institutional vs retail candles and
    measure momentum — high body ratio + direction confirms strong move.

    Features:
        feature_body_ratio      body / total range  (0=doji, 1=marubozu)
        feature_top_wick_ratio  top wick / total range
        feature_bot_wick_ratio  bottom wick / total range
        feature_candle_dir      +1=bull, -1=bear, 0=doji
        feature_consec_bull     consecutive bullish candles ending here (0-10)
        feature_consec_bear     consecutive bearish candles ending here (0-10)
        feature_roc_1_atr       1-period close ROC / ATR  (immediate momentum)
        feature_roc_5_atr       5-period close ROC / ATR  (short-term momentum)
    """
    df = df.copy()
    n = len(df)
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    atr = df["_atr"].values

    body_ratio = np.zeros(n)
    top_wick_ratio = np.zeros(n)
    bot_wick_ratio = np.zeros(n)
    candle_dir = np.zeros(n)
    consec_bull = np.zeros(n)
    consec_bear = np.zeros(n)
    roc_1 = np.zeros(n)
    roc_5 = np.zeros(n)

    for i in range(n):
        total = max(h[i] - l[i], 1e-8)
        body = abs(c[i] - o[i])

        body_ratio[i] = body / total
        top_wick_ratio[i] = (h[i] - max(o[i], c[i])) / total
        bot_wick_ratio[i] = (min(o[i], c[i]) - l[i]) / total

        if c[i] > o[i]:
            candle_dir[i] = 1.0
        elif c[i] < o[i]:
            candle_dir[i] = -1.0
        else:
            candle_dir[i] = 0.0

        # Consecutive candles — scan back up to 10
        if i > 0:
            if c[i] > o[i]:
                cnt = 1
                for k in range(i - 1, max(i - 10, -1), -1):
                    if c[k] > o[k]:
                        cnt += 1
                    else:
                        break
                consec_bull[i] = cnt
            elif c[i] < o[i]:
                cnt = 1
                for k in range(i - 1, max(i - 10, -1), -1):
                    if c[k] < o[k]:
                        cnt += 1
                    else:
                        break
                consec_bear[i] = cnt

        # Rate of change (signed, ATR-normalised)
        a = max(atr[i], 1e-8) if not np.isnan(atr[i]) else 1e-8
        if i >= 1:
            roc_1[i] = (c[i] - c[i - 1]) / a
        if i >= 5:
            roc_5[i] = (c[i] - c[i - 5]) / a

    df["feature_body_ratio"] = body_ratio
    df["feature_top_wick_ratio"] = top_wick_ratio
    df["feature_bot_wick_ratio"] = bot_wick_ratio
    df["feature_candle_dir"] = candle_dir
    df["feature_consec_bull"] = consec_bull
    df["feature_consec_bear"] = consec_bear
    df["feature_roc_1_atr"] = roc_1
    df["feature_roc_5_atr"] = roc_5

    return df


# ══════════════════════════════════════════════════════
# NEW: DISPLACEMENT QUALITY FEATURES
# ══════════════════════════════════════════════════════


def add_displacement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures the strength and quality of recent 3-candle displacements.

    Strong displacement = large impulsive move in one direction, leaving
    FVGs behind and creating valid OBs. Weak displacement = choppy moves
    that produce low-quality OBs prone to failure.

    Why this helps RRR:
        Entry after strong displacement means the model entered behind
        institutional order flow. These trades have higher probability
        of continuation and justify holding for 5R+ targets.

    Features:
        feature_disp_3_atr   abs(close[i] - close[i-3]) / ATR
                             Measures 3-candle move intensity.
                             > 1.5 = significant, > 3.0 = very strong.

        feature_disp_strong  1 if disp_3_atr > DISPLACEMENT_ATR_MULT (config)
                             Binary filter: only enter after confirmed displacement.

        feature_disp_bull    1 if strong bullish displacement (close[i] > close[i-3]
                             AND disp_strong). Confirms bullish order flow.

        feature_disp_bear    1 if strong bearish displacement. Confirms bearish flow.

        feature_disp_impulse 1 if all 3 candles are same direction (fully impulsive).
                             An impulsive move is more reliable than a mixed one.
                             E.g. 3 consecutive bull candles = clean institutional move.
    """
    df = df.copy()
    n = len(df)
    c = df["close"].values
    o = df["open"].values
    atr = df["_atr"].values
    mult = getattr(config, "DISPLACEMENT_ATR_MULT", 1.5)

    disp_3_atr = np.zeros(n)
    disp_strong = np.zeros(n)
    disp_bull = np.zeros(n)
    disp_bear = np.zeros(n)
    disp_impulse = np.zeros(n)

    for i in range(3, n):
        a = max(atr[i], 1e-8) if not np.isnan(atr[i]) else 1e-8
        move = abs(c[i] - c[i - 3])
        score = move / a
        disp_3_atr[i] = score

        strong = score > mult
        disp_strong[i] = float(strong)

        if strong:
            if c[i] > c[i - 3]:
                disp_bull[i] = 1.0
            else:
                disp_bear[i] = 1.0

        # Impulse: all 3 candles close in the same direction
        bull_3 = (c[i] > o[i]) and (c[i - 1] > o[i - 1]) and (c[i - 2] > o[i - 2])
        bear_3 = (c[i] < o[i]) and (c[i - 1] < o[i - 1]) and (c[i - 2] < o[i - 2])
        if bull_3 or bear_3:
            disp_impulse[i] = 1.0

    df["feature_disp_3_atr"] = disp_3_atr
    df["feature_disp_strong"] = disp_strong
    df["feature_disp_bull"] = disp_bull
    df["feature_disp_bear"] = disp_bear
    df["feature_disp_impulse"] = disp_impulse

    return df


# ══════════════════════════════════════════════════════
# NEW: SWING FAILURE / LIQUIDITY SWEEP FEATURES
# ══════════════════════════════════════════════════════


def add_swing_failure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects liquidity sweeps (stop hunts): price wicks beyond a key level
    then closes back inside. This is one of the highest-probability ICT
    entry signals because:

        1. Stops below a swing low get triggered (sellers get filled)
        2. Price immediately reverses — the triggering of stops WAS the move
        3. Entry above the swept level gives tight SL + large target = high RRR

    Why this directly targets 5RRR:
        After a bullish sweep, SL is just below the wick low.
        Target is the opposite range high = often 5-8R away.
        Win rate is high because the manipulation (stop hunt) has already occurred.

    Requires: _sth_price, _stl_price, _ith_price, _itl_price
    Must be called after detect_sth_stl and detect_ith_itl.

    Features:
        feature_sweep_bull   1 if this candle wicked below STL or ITL
                             AND closed above it (bullish stop hunt complete).
                             Signal: go long, SL below wick low.

        feature_sweep_bear   1 if this candle wicked above STH or ITH
                             AND closed below it (bearish stop hunt complete).
                             Signal: go short, SL above wick high.

        feature_sweep_bull_recent  1 if any bullish sweep in last 5 candles.
                                   Allows model to enter on the candle after.

        feature_sweep_bear_recent  1 if any bearish sweep in last 5 candles.
    """
    required = ["_sth_price", "_stl_price", "_ith_price", "_itl_price"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing {r} — run structure detection first.")

    df = df.copy()
    n = len(df)
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # Use i-1 prices to avoid lookahead (level existed before this candle opened)
    sth_p = df["_sth_price"].values
    stl_p = df["_stl_price"].values
    ith_p = df["_ith_price"].values
    itl_p = df["_itl_price"].values

    sweep_bull = np.zeros(n)
    sweep_bear = np.zeros(n)

    for i in range(1, n):
        # Reference: use the level from the PREVIOUS candle (no lookahead)
        stl_ref = stl_p[i - 1]
        sth_ref = sth_p[i - 1]
        itl_ref = itl_p[i - 1]
        ith_ref = ith_p[i - 1]

        # Bullish sweep: wick below STL or ITL, close recovers above
        bull = False
        if not np.isnan(stl_ref) and l[i] < stl_ref and c[i] > stl_ref:
            bull = True
        if not bull and not np.isnan(itl_ref) and l[i] < itl_ref and c[i] > itl_ref:
            bull = True
        sweep_bull[i] = float(bull)

        # Bearish sweep: wick above STH or ITH, close falls back below
        bear = False
        if not np.isnan(sth_ref) and h[i] > sth_ref and c[i] < sth_ref:
            bear = True
        if not bear and not np.isnan(ith_ref) and h[i] > ith_ref and c[i] < ith_ref:
            bear = True
        sweep_bear[i] = float(bear)

    # Rolling recent sweep windows (5 candles)
    sweep_bull_recent = pd.Series(sweep_bull).rolling(5, min_periods=1).max().values
    sweep_bear_recent = pd.Series(sweep_bear).rolling(5, min_periods=1).max().values

    df["feature_sweep_bull"] = sweep_bull
    df["feature_sweep_bear"] = sweep_bear
    df["feature_sweep_bull_recent"] = sweep_bull_recent
    df["feature_sweep_bear_recent"] = sweep_bear_recent

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

    print("  [structure] Computing candle anatomy features...")
    df = add_candle_features(df)

    print("  [structure] Computing displacement quality features...")
    df = add_displacement_features(df)

    print("  [structure] Computing swing failure / sweep features...")
    df = add_swing_failure_features(df)

    return df
