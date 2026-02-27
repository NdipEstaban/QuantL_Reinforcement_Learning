"""
training/train.py
==================
Main training script with curriculum-aware resume logic.

When resuming from a checkpoint, hyperparameters from config.py are
ALWAYS applied — they override whatever was saved in the .zip file.
This is essential for curriculum learning phases where you intentionally
change learning_rate, clip_range, and ent_coef between phases.

Usage:
    python run.py train
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from features.pipeline import load_processed, load_and_process
from env.trading_env import make_env
from training.callbacks import get_callbacks


def split_data(df: pd.DataFrame) -> tuple:
    """Split into train and test sets (chronological — never shuffle)."""
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"  Train: {len(train_df):,} candles | Test: {len(test_df):,} candles")
    return train_df, test_df


def get_algorithm(name: str):
    from stable_baselines3 import PPO, A2C, SAC

    algs = {"PPO": PPO, "A2C": A2C, "SAC": SAC}
    if name not in algs:
        raise ValueError(f"Unknown algorithm '{name}'. Choose: {list(algs.keys())}")
    return algs[name]


def _find_latest_checkpoint() -> str | None:
    """Return path to latest checkpoint or final_model, or None if none exist."""
    ckpt_dir = config.MODEL_SAVE_PATH

    # Prefer final_model
    for name in ["final_model.zip", "final_model"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p

    # Fall back to numbered checkpoints
    pattern = os.path.join(ckpt_dir, "*.zip")
    zips = sorted(glob.glob(pattern))
    return zips[-1] if zips else None


def _apply_config_hyperparams(model, ppo_config: dict) -> None:
    """
    Force config hyperparameters onto a loaded model.

    PPO.load() restores hyperparameters from the .zip file, which means
    changes made in config.py between training phases are silently ignored.
    This function overrides them with the current config values.

    This is critical for curriculum learning — each phase intentionally
    uses different learning_rate, clip_range, and ent_coef.
    """
    lr = ppo_config.get("learning_rate")
    clip = ppo_config.get("clip_range")
    ent = ppo_config.get("ent_coef")
    gamma = ppo_config.get("gamma")
    n_epochs = ppo_config.get("n_epochs")

    changed = []

    # SB3 stores lr and clip_range as callables (schedule functions).
    # Setting them as floats causes TypeError on model.learn().
    # We must wrap them in a lambda that accepts progress_remaining.

    if lr is not None:
        model.learning_rate = lambda _: lr
        # Also patch the optimizer directly — SB3 bakes the old LR into
        # optimizer.param_groups at model creation. The schedule change
        # alone only affects future _update_learning_rate() calls, but
        # SB3 calls that at the start of each train() iteration, so this
        # is belt-and-suspenders for the very first iteration.
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for pg in model.policy.optimizer.param_groups:
                pg["lr"] = lr
        changed.append(f"learning_rate={lr}")

    if clip is not None:
        model.clip_range = lambda _: clip
        changed.append(f"clip_range={clip}")

    if ent is not None:
        model.ent_coef = ent  # ent_coef is stored as a plain float in SB3
        changed.append(f"ent_coef={ent}")

    if gamma is not None:
        model.gamma = gamma
        changed.append(f"gamma={gamma}")

    if n_epochs is not None:
        model.n_epochs = n_epochs
        changed.append(f"n_epochs={n_epochs}")

    if changed:
        print(f"  Hyperparams overridden from config: {', '.join(changed)}")


def train(data_path: str = None):
    """
    Full training run with automatic resume and config hyperparameter override.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

    # ── Load data ────────────────────────────────────
    print("\n=== Loading Data ===")
    try:
        df = load_processed()
    except FileNotFoundError:
        print("No processed data found — running feature pipeline...")
        df = load_and_process(data_path, save=True)

    # ── Split ────────────────────────────────────────
    print("\n=== Splitting Data ===")
    train_df, test_df = split_data(df)

    # ── Environments ─────────────────────────────────
    print(f"\n=== Creating {config.N_ENVS} Parallel Environments ===")
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.monitor import Monitor

    def _make():
        return make_env(train_df, mode="train")

    train_env = VecMonitor(SubprocVecEnv([_make for _ in range(config.N_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv([lambda: make_env(test_df, mode="eval")]))

    print(f"  Train envs : {config.N_ENVS} parallel processes")
    print(f"  Eval env   : 1 (deterministic)")

    # ── Model ─────────────────────────────────────────
    print(f"\n=== Building PPO Model ===")
    device = "cpu"
    AlgorithmClass = get_algorithm(config.ALGORITHM)
    ppo_kwargs = {k: v for k, v in config.PPO_CONFIG.items() if k != "verbose"}

    checkpoint = _find_latest_checkpoint()

    if checkpoint:
        print(f"  Resuming from: {checkpoint}")
        model = AlgorithmClass.load(
            checkpoint,
            env=train_env,
            device=device,
        )
        # ← KEY FIX: override saved hyperparams with current config
        _apply_config_hyperparams(model, config.PPO_CONFIG)
    else:
        print(f"  Starting from scratch")
        model = AlgorithmClass(
            "MlpPolicy",
            train_env,
            device=device,
            tensorboard_log=config.LOG_DIR,
            verbose=1,
            **ppo_kwargs,
        )

    n_features = len([c for c in df.columns if c.startswith("feature_")])
    steps_per_iter = config.N_ENVS * config.PPO_CONFIG.get("n_steps", 2048)
    print(f"  Algorithm       : {config.ALGORITHM}")
    print(f"  Features        : {n_features}")
    print(f"  Parallel envs   : {config.N_ENVS}")
    print(f"  Device          : {device}")
    print(f"  Effective steps : {config.TOTAL_TIMESTEPS:,}")
    print(f"  Steps per iter  : {steps_per_iter:,}")

    # ── Callbacks ────────────────────────────────────
    callbacks = get_callbacks(eval_env=eval_env)

    # ── Train ─────────────────────────────────────────
    print(f"\n=== Training ===")
    print("TensorBoard: tensorboard --logdir logs/")

    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=False,  # keep global step counter
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    # ── Save ──────────────────────────────────────────
    final_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
    model.save(final_path)
    print(f"\n✅ Model saved to: {final_path}.zip")

    return model, train_df, test_df


if __name__ == "__main__":
    train()
