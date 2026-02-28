"""
features/pipeline.py
=====================
Master feature pipeline. Runs all feature modules in the correct order
and saves the processed DataFrame to data/processed/.

Run directly:
    python -m features.pipeline

Or import:
    from features.pipeline import add_all_features, load_and_process
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from features.structure import add_structure_features
from features.zones import add_zone_features
from features.session import add_session_features


# ══════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════


def load_raw_data(path: str = None) -> pd.DataFrame:
    """
    Load raw OHLCV CSV. Handles column renaming, timezone conversion,
    and missing volume column gracefully.
    """
    path = path or config.DATA_FILE
    print(f"Loading data from: {path}")

    df = pd.read_csv(path)

    # Rename columns to lowercase standard names
    rename = {
        config.DATE_COLUMN: "date",
        config.OPEN_COLUMN: "open",
        config.HIGH_COLUMN: "high",
        config.LOW_COLUMN: "low",
        config.CLOSE_COLUMN: "close",
    }
    if config.VOLUME_COLUMN and config.VOLUME_COLUMN in df.columns:
        rename[config.VOLUME_COLUMN] = "volume"

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Ensure volume column exists
    if "volume" not in df.columns:
        df["volume"] = 1.0

    # Parse datetime and set as index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates()

    # Ensure OHLCV validity
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    # Convert timezone to UTC
    if df.index.tz is None:
        try:
            import pytz

            broker_tz = pytz.timezone(config.BROKER_TIMEZONE)
            df.index = df.index.tz_localize(broker_tz).tz_convert("UTC")
        except Exception as e:
            print(f"  [pipeline] Timezone conversion failed ({e}) — assuming UTC.")
            df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
        

    print(f"  Loaded {len(df):,} candles | {df.index[0]} → {df.index[-1]}")
    return df


# ══════════════════════════════════════════════════════
# FEATURE PIPELINE
# ══════════════════════════════════════════════════════


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature modules in the correct dependency order:

    1. FVG first  (needed by ITH/ITL qualification in structure)
    2. Structure  (STH/STL → ITH/ITL → LTH/LTL → BOS/CHoCH/Bias → Distances)
    3. OB/BB      (needs structure levels + BOS flags)
    4. Session    (needs DatetimeIndex)
    5. Clean up   (drop NaN rows, drop internal _ columns from final output)
    """
    print("\n=== Feature Pipeline ===")

    # ATR must be computed first — every module depends on it
    print("\n[0/4] Computing ATR...")
    from features.structure import compute_atr

    df["_atr"] = compute_atr(df)

    # Step 1: FVG (needed for ITH/ITL FVG-qualification path)
    print("\n[1/4] Fair Value Gaps...")
    from features.zones import add_fvg_features

    df = add_fvg_features(df)

    # Step 2: Full structure (uses _fvg_* columns from step 1)
    print("\n[2/4] Market Structure (STH/STL/ITH/ITL/LTH/LTL/BOS/CHoCH/Bias)...")
    df = add_structure_features(df)

    # Step 3: OB + BB (needs _ith_price, _itl_price, BOS flags)
    print("\n[3/4] Order Blocks + Breaker Blocks...")
    from features.zones import add_order_block_features, add_breaker_block_features

    df = add_order_block_features(df)
    df = add_breaker_block_features(df)

    # Step 4: Sessions
    print("\n[4/4] Session / Killzone...")
    df = add_session_features(df)

    # ── Final cleanup ────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    internal_cols = [c for c in df.columns if c.startswith("_")]

    print(f"\n{'═'*55}")
    print(f"✅ Pipeline complete")
    print(f"   Candles after cleaning : {len(df):,}")
    print(f"   Feature columns        : {len(feature_cols)}")
    print(f"   Internal columns       : {len(internal_cols)} (kept for reference)")
    print(f"{'═'*55}")
    print("\nFeatures fed to model:")
    for c in sorted(feature_cols):
        print(f"   {c}")

    return df


# ══════════════════════════════════════════════════════
# LOAD + PROCESS + SAVE
# ══════════════════════════════════════════════════════


def load_and_process(data_path: str = None, save: bool = True) -> pd.DataFrame:
    """
    Full pipeline: load CSV → add features → optionally save processed file.
    Returns the processed DataFrame.
    """
    df = load_raw_data(data_path)
    df = add_all_features(df)

    if save:
        os.makedirs("data/processed", exist_ok=True)
        out_path = "data/processed/features.pkl"
        df.to_pickle(out_path)
        print(f"\nSaved processed data to: {out_path}")

    return df


def load_processed(path: str = "data/processed/features.pkl") -> pd.DataFrame:
    """Load a previously processed DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No processed data found at {path}.\n" "Run: python -m features.pipeline"
        )
    df = pd.read_pickle(path)
    print(
        f"Loaded processed data: {len(df):,} candles, {len([c for c in df.columns if c.startswith('feature_')])} features"
    )
    return df


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    df = load_and_process(save=True)
    print(f"\nSample (last 3 rows):")
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    print(df[["close"] + feature_cols[:8]].tail(3).to_string())
