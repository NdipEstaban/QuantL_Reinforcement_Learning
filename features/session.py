"""
features/session.py
====================
Session classification, killzone detection, and day-boundary reference levels.

Sessions (UTC):
    Asian   : 00:00 – 08:00
    London  : 07:00 – 16:00  (overlaps with NY)
    New York: 12:00 – 21:00

Killzones (high-probability ICT windows):
    London Open KZ : 07:00 – 10:00 UTC
    NY Open KZ     : 12:00 – 15:00 UTC

NEW: Day Reference Levels
    Previous Day High (PDH) — sell-side liquidity target / resistance
    Previous Day Low  (PDL) — buy-side liquidity target / support
    Asian Session Range     — manipulation zone during London/NY sessions
    Midnight Open           — 00:00 UTC price (ICT reference level)

WHY PDH/PDL MATTER FOR 5RRR:
    The previous day's high and low are the most reliable intraday targets
    on M15. Institutional algorithms programme entries at OBs/FVGs with
    targets at PDH/PDL because large players know exactly where retail stops
    are clustered. Distance from entry to PDH/PDL is often 4-8R on M15.

    When bias is bullish and price is near PDL → price is at a buy zone
    near a target level (PDH above). This directly produces high-RRR setups.

Features:
    feature_session              1=Asian / 2=London / 3=NewYork / 0=off
    feature_killzone_active      1=inside London or NY killzone window
    feature_pdh_dist             (PDH - close) / ATR  — dist to prev day high
    feature_pdl_dist             (close - PDL) / ATR  — dist to prev day low
    feature_prev_day_range_pct   where close sits in prev day range (0=PDL, 1=PDH)
    feature_asian_high_dist      (Asian_H - close) / ATR
    feature_asian_low_dist       (close - Asian_L) / ATR
    feature_asian_range_pos      where close sits in Asian range (0=low, 1=high)
    feature_midnight_open_dist   (midnight_open - close) / ATR  (signed)
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires df to have a UTC DatetimeIndex and _atr column.
    If the index has no timezone info, it is assumed UTC.
    """
    df = df.copy()

    if not hasattr(df.index, "hour"):
        print(
            "  [session] WARNING: index is not DatetimeIndex — skipping session features."
        )
        df["feature_session"] = 0
        df["feature_killzone_active"] = 0
        df["feature_pdh_dist"] = np.nan
        df["feature_pdl_dist"] = np.nan
        df["feature_prev_day_range_pct"] = np.nan
        df["feature_asian_high_dist"] = np.nan
        df["feature_asian_low_dist"] = np.nan
        df["feature_asian_range_pos"] = np.nan
        df["feature_midnight_open_dist"] = np.nan
        return df

    hours = df.index.hour
    n = len(df)

    session = np.zeros(n, dtype=int)
    killzone = np.zeros(n, dtype=int)

    for i, h in enumerate(hours):
        # Session classification
        if config.ASIAN_START <= h < config.ASIAN_END:
            session[i] = 1
        elif config.LONDON_KZ_START <= h < 16:
            session[i] = 2
        elif 12 <= h < 21:
            session[i] = 3

        # Killzone
        london_kz = config.LONDON_KZ_START <= h < config.LONDON_KZ_END
        ny_kz = config.NY_KZ_START <= h < config.NY_KZ_END
        killzone[i] = 1 if (london_kz or ny_kz) else 0

    df["feature_session"] = session
    df["feature_killzone_active"] = killzone

    # ── Day Reference Levels ────────────────────────────────────────────────
    # We compute PDH/PDL by grouping OHLCV by UTC date, then forward-filling
    # the PREVIOUS day's values into the current day.
    # This is entirely lookahead-free: at any candle, we only use completed days.

    atr = df["_atr"].values
    close = df["close"].values

    # Extract UTC date for grouping
    dates = df.index.normalize()  # midnight UTC for each row

    # Daily OHLC
    daily = (
        df["close"]
        .groupby(dates)
        .agg(
            high_=("max"),
            low_=("min"),
            open_=("first"),
        )
        .rename(columns={"high_": "d_high", "low_": "d_low", "open_": "d_open"})
    )

    # Rebuild properly
    daily_high = df["high"].groupby(dates).max()
    daily_low = df["low"].groupby(dates).min()
    daily_open = df["close"].groupby(dates).first()  # use close[0] as midnight open

    # Shift by 1 day to get PREVIOUS day's values (strict no-lookahead)
    pdh_series = daily_high.shift(1)
    pdl_series = daily_low.shift(1)
    pdo_series = daily_open.shift(1)  # prev day open

    # Reindex back to M15 frequency (forward fill within each day)
    pdh = pdh_series.reindex(dates).values
    pdl = pdl_series.reindex(dates).values

    # Asian session range: high/low between 00:00 and 08:00 UTC on the SAME date
    # For each M15 candle, report the Asian range that completed before London open
    # We compute it per date, then attach to all candles on that same date
    asian_mask = (hours >= config.ASIAN_START) & (hours < config.ASIAN_END)

    asian_df = df[asian_mask].copy()
    if len(asian_df) > 0:
        asian_dates = asian_df.index.normalize()
        asian_high_by_date = asian_df["high"].groupby(asian_dates).max()
        asian_low_by_date = asian_df["low"].groupby(asian_dates).min()
        asian_open_by_date = (
            asian_df["open"].groupby(asian_dates).first()
        )  # 00:00 open = midnight open
    else:
        # Fallback if no Asian data
        asian_high_by_date = pd.Series(dtype=float)
        asian_low_by_date = pd.Series(dtype=float)
        asian_open_by_date = pd.Series(dtype=float)

    # Reindex to full M15 index — forward fill so London/NY candles see today's Asian range
    asian_h_aligned = asian_high_by_date.reindex(dates).ffill().values
    asian_l_aligned = asian_low_by_date.reindex(dates).ffill().values
    midnight_open = asian_open_by_date.reindex(dates).ffill().values

    # ── Compute distance features ───────────────────────────────────────────
    pdh_dist = np.full(n, np.nan)
    pdl_dist = np.full(n, np.nan)
    prev_day_range_pct = np.full(n, np.nan)
    asian_high_dist = np.full(n, np.nan)
    asian_low_dist = np.full(n, np.nan)
    asian_range_pos = np.full(n, np.nan)
    midnight_dist = np.full(n, np.nan)

    for i in range(n):
        a = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 1e-8
        c = close[i]

        # PDH / PDL (only valid after first full trading day in dataset)
        if not np.isnan(pdh[i]) and not np.isnan(pdl[i]):
            pdh_dist[i] = (pdh[i] - c) / a  # positive = PDH above
            pdl_dist[i] = (c - pdl[i]) / a  # positive = PDL below
            day_rng = pdh[i] - pdl[i]
            if day_rng > 0:
                prev_day_range_pct[i] = np.clip((c - pdl[i]) / day_rng, 0.0, 1.0)

        # Asian range
        if not np.isnan(asian_h_aligned[i]) and not np.isnan(asian_l_aligned[i]):
            asian_high_dist[i] = (asian_h_aligned[i] - c) / a
            asian_low_dist[i] = (c - asian_l_aligned[i]) / a
            ar = asian_h_aligned[i] - asian_l_aligned[i]
            if ar > 0:
                asian_range_pos[i] = np.clip((c - asian_l_aligned[i]) / ar, 0.0, 1.0)

        # Midnight open (signed — positive = price below midnight open)
        if not np.isnan(midnight_open[i]):
            midnight_dist[i] = (midnight_open[i] - c) / a

    df["feature_pdh_dist"] = pdh_dist
    df["feature_pdl_dist"] = pdl_dist
    df["feature_prev_day_range_pct"] = prev_day_range_pct
    df["feature_asian_high_dist"] = asian_high_dist
    df["feature_asian_low_dist"] = asian_low_dist
    df["feature_asian_range_pos"] = asian_range_pos
    df["feature_midnight_open_dist"] = midnight_dist

    return df
