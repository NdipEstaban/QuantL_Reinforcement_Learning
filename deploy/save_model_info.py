"""
deploy/save_model_info.py
==========================
One-time utility: saves model_info.json alongside each model.

model_info.json stores:
  - obs_features : exact ordered list of observation feature columns
  - positions    : position values the model was trained with
  - algorithm    : algorithm class name
  - n_features   : observation space dimension
  - trained_at   : ISO timestamp

Run once after training (or after copying a model to deploy):
    python deploy/save_model_info.py

The file is then read by predictor.py at load time to guarantee
the observation space matches training exactly.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

# ── Project root ───────────────────────────────────────────────────────────────
BOT_ROOT = Path(os.getenv("BOT_ROOT", Path(__file__).parent.parent)).resolve()
sys.path.insert(0, str(BOT_ROOT))

import config
import gymnasium as gym
import gym_trading_env  # noqa: F401
from features.pipeline import load_processed


def get_obs_features_from_env(df) -> list[str]:
    """
    Create a TradingEnv instance to read the exact obs_features list.
    This is the ONLY reliable way to get the correct column order.
    """
    env = gym.make(
        "TradingEnv",
        name="INFO_PROBE",
        df=df.copy(),
        positions=config.POSITIONS,
        trading_fees=config.TRADING_FEES,
        borrow_interest_rate=config.BORROW_INTEREST_RATE,
        portfolio_initial_value=config.PORTFOLIO_INITIAL_VALUE,
        max_episode_duration=100,
        verbose=0,
    )
    raw = env.unwrapped
    if hasattr(raw, "_obs_features"):
        features = list(raw._obs_features)
    elif hasattr(raw, "obs_features"):
        features = list(raw.obs_features)
    else:
        features = [c for c in raw.df.columns if c.startswith("feature_")]
    env.close()
    # Ensure both gym-trading-env internal columns are present
    if "feature_close_pct" not in features:
        features.append("feature_close_pct")
    if "feature_portfolio_distribution" not in features:
        features.append("feature_portfolio_distribution")
    return features


def save_model_info(model_dir: Path, algorithm: str, obs_features: list[str]) -> None:
    info = {
        "algorithm": algorithm,
        "obs_features": obs_features,
        "n_features": len(obs_features),
        "positions": list(config.POSITIONS),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "obs_features order is critical — must match gym-trading-env column order. "
            "Do NOT sort. Regenerate this file if the feature pipeline changes."
        ),
    }
    out = model_dir / "model_info.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: {out}")
    print(
        f"  Features ({len(obs_features)}): {obs_features[:5]} ... {obs_features[-3:]}"
    )


def main():
    print("Loading processed data to probe feature columns...")
    df = load_processed()

    print("Creating TradingEnv to read obs_features list...")
    obs_features = get_obs_features_from_env(df)
    print(f"  Got {len(obs_features)} obs features")
    print(f"  First 5: {obs_features[:5]}")
    print(f"  Last 5:  {obs_features[-5:]}")

    models_dir = BOT_ROOT / "models"
    found = 0

    for algo_dir in sorted(models_dir.iterdir()):
        if not algo_dir.is_dir():
            continue
        # Check for versioned layout
        for version_dir in sorted(algo_dir.iterdir()):
            if version_dir.is_dir() and (version_dir / "final_model.zip").exists():
                algo = algo_dir.name.upper().replace("-", "").replace("_", "")
                alg_class = "RecurrentPPO" if algo in ("RPPO", "RECURRENTPPO") else algo
                print(f"\nProcessing: {algo_dir.name}/{version_dir.name}")
                save_model_info(version_dir, alg_class, obs_features)
                found += 1
        # Legacy flat layout
        if (algo_dir / "final_model.zip").exists():
            print(f"\nProcessing legacy: {algo_dir.name} (→ rppo/v1)")
            save_model_info(algo_dir, "RecurrentPPO", obs_features)
            found += 1

    if found == 0:
        print("\nNo models found. Check models/ directory structure.")
    else:
        print(f"\n✅ Saved model_info.json for {found} model(s)")


if __name__ == "__main__":
    main()
