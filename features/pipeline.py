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
    if df.index.tzinfo is None:
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
# LOAD + PROCESS + SAVE  (with per-CSV cache)
# ══════════════════════════════════════════════════════


def _cache_path_for(data_path: str) -> str:
    """
    Derive the .pkl cache path from the source CSV path.

    Examples:
        data/raw/EURUSD_M15_2010-2016.csv  →  data/processed/EURUSD_M15_2010-2016.pkl
        data/unseen/EURUSD_2024_M15.csv    →  data/processed/EURUSD_2024_M15.pkl
    """
    stem = os.path.splitext(os.path.basename(data_path))[
        0
    ]  # filename without extension
    return os.path.join("data", "processed", f"{stem}.pkl")


def _cache_is_valid(csv_path: str, cache_path: str) -> bool:
    """
    Return True if:
      1. The cache file exists, AND
      2. The cache is newer than the source CSV (CSV hasn't been modified since)

    This means: if you update the CSV or change feature code and delete the
    cache, the pipeline will reprocess automatically next run.
    """
    if not os.path.exists(cache_path):
        return False
    csv_mtime = os.path.getmtime(csv_path)
    cache_mtime = os.path.getmtime(cache_path)
    return cache_mtime >= csv_mtime


def load_and_process(
    data_path: str = None, save: bool = True, force: bool = False
) -> pd.DataFrame:
    """
    Full pipeline with per-CSV caching.

    Behaviour:
      - If a valid cache exists for this CSV, load it directly (fast).
      - If the CSV has been modified since the cache was written, reprocess.
      - If no cache exists yet, process and save one.
      - Pass force=True to always reprocess and overwrite the cache.

    Cache location:  data/processed/<csv_stem>.pkl
    Example:
      data/raw/EURUSD_M15_2010-2016.csv  →  data/processed/EURUSD_M15_2010-2016.pkl

    Args:
        data_path : path to source CSV (defaults to config.DATA_FILE)
        save      : whether to save the cache after processing (default True)
        force     : if True, ignore existing cache and reprocess (default False)
    """
    data_path = data_path or config.DATA_FILE
    cache_path = _cache_path_for(data_path)

    if not force and _cache_is_valid(data_path, cache_path):
        print(f"\n[pipeline] Cache hit — loading from: {cache_path}")
        print(f"           (delete this file to force reprocessing)")
        df = pd.read_pickle(cache_path)
        n_feat = len([c for c in df.columns if c.startswith("feature_")])
        print(f"           {len(df):,} candles | {n_feat} features")
        return df

    # Cache miss or forced reprocess
    if force:
        print(f"\n[pipeline] force=True — reprocessing {data_path}")
    elif os.path.exists(cache_path):
        print(f"\n[pipeline] Cache stale (CSV newer than cache) — reprocessing")
    else:
        print(f"\n[pipeline] No cache found — processing for the first time")

    df = load_raw_data(data_path)
    df = add_all_features(df)

    if save:
        os.makedirs("data/processed", exist_ok=True)
        df.to_pickle(cache_path)
        print(f"\n[pipeline] Saved cache → {cache_path}")

    return df


def load_processed(path: str = None) -> pd.DataFrame:
    """
    Load a previously processed DataFrame from a cache file.

    Args:
        path: explicit .pkl path, or None to use config.DATA_FILE's cache path.
    """
    if path is None:
        path = _cache_path_for(config.DATA_FILE)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No processed cache found at: {path}\n"
            f"Run load_and_process() first to generate it."
        )
    df = pd.read_pickle(path)
    n_feat = len([c for c in df.columns if c.startswith("feature_")])
    print(f"[pipeline] Loaded cache: {len(df):,} candles | {n_feat} features | {path}")
    return df


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    df = load_and_process(save=True)
    print(f"\nSample (last 3 rows):")
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    print(df[["close"] + feature_cols[:8]].tail(3).to_string())
