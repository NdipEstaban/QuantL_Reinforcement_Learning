"""
features/session.py
====================
Session classification and killzone detection.

Sessions (UTC):
    Asian   : 00:00 – 08:00
    London  : 07:00 – 16:00  (overlaps with NY)
    New York: 12:00 – 21:00

Killzones (high-probability ICT windows):
    London Open KZ : 07:00 – 10:00 UTC
    NY Open KZ     : 12:00 – 15:00 UTC

Features:
    feature_session          1=Asian / 2=London / 3=NewYork / 0=off
    feature_killzone_active  1=inside London or NY killzone window
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires df to have a UTC DatetimeIndex.
    If the index has no timezone info, it is assumed UTC.
    """
    df = df.copy()

    if not hasattr(df.index, "hour"):
        print("  [session] WARNING: index is not DatetimeIndex — skipping session features.")
        df["feature_session"]        = 0
        df["feature_killzone_active"] = 0
        return df

    hours = df.index.hour
    n     = len(df)

    session  = np.zeros(n, dtype=int)
    killzone = np.zeros(n, dtype=int)

    for i, h in enumerate(hours):
        # Session classification
        if config.ASIAN_START <= h < config.ASIAN_END:
            session[i] = 1
        elif config.LONDON_KZ_START <= h < 16:
            session[i] = 2
        elif 12 <= h < 21:
            session[i] = 3

        # Killzone: London open or NY open window
        london_kz = config.LONDON_KZ_START <= h < config.LONDON_KZ_END
        ny_kz     = config.NY_KZ_START <= h < config.NY_KZ_END
        killzone[i] = 1 if (london_kz or ny_kz) else 0

    df["feature_session"]         = session
    df["feature_killzone_active"] = killzone

    return df
