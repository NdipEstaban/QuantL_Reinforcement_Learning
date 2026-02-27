import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from features.pipeline import load_processed, load_and_process
from env.trading_env import make_env
from training.callbacks import get_callbacks
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env_fn(df: pd.DataFrame, mode: str = "train"):
    """Factory function — required by SubprocVecEnv."""

    def _init():
        return make_env(df, mode=mode)

    return _init


def split_data(df: pd.DataFrame) -> tuple:
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"  Train: {len(train_df):,} candles | Test: {len(test_df):,} candles")
    return train_df, test_df


def get_algorithm(name: str):
    from stable_baselines3 import PPO, A2C, SAC

    return {"PPO": PPO, "A2C": A2C, "SAC": SAC}[name]


def train(data_path: str = None):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    print("\n=== Loading Data ===")
    try:
        df = load_processed()
    except FileNotFoundError:
        print("No processed data — running feature pipeline...")
        df = load_and_process(data_path, save=True)

    print("\n=== Splitting Data ===")
    train_df, test_df = split_data(df)

    # ── Parallel environments ─────────────────────────────────────
    # SubprocVecEnv spawns N independent env processes.
    # Each process steps forward one candle simultaneously.
    # Throughput scales almost linearly with N_ENVS up to core count.
    #
    # With 16 cores: keep 2 free for OS + main process = 14 envs max.
    # Practically 8-12 is the sweet spot (memory and overhead).
    n_envs = getattr(config, "N_ENVS", 8)
    print(f"\n=== Creating {n_envs} Parallel Environments ===")

    # Train: N parallel envs
    train_env = SubprocVecEnv(
        [make_env_fn(train_df, mode="train") for _ in range(n_envs)],
        start_method="spawn",  # required on Windows
    )
    train_env = VecMonitor(train_env)  # wraps all envs with Monitor

    # Eval: single env (deterministic, no need to parallelize)
    eval_env = VecMonitor(DummyVecEnv([lambda: make_env(test_df, mode="eval")]))

    print(f"  Train envs : {n_envs} parallel processes")
    print(f"  Eval env   : 1 (deterministic)")

    # ── Model with GPU ────────────────────────────────────────────
    print(f"\n=== Building {config.ALGORITHM} Model ===")
    AlgorithmClass = get_algorithm(config.ALGORITHM)

    # Check CUDA availability
    import torch

    device = "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device : GPU ({gpu_name}, {vram:.1f}GB VRAM)")
    else:
        print("  Device : CPU (CUDA not available)")

    # Check if resuming
    final_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
    ckpt_dir = config.MODEL_SAVE_PATH
    checkpoints = (
        sorted(
            [
                f
                for f in os.listdir(ckpt_dir)
                if f.startswith("checkpoint") and f.endswith(".zip")
            ]
        )
        if os.path.exists(ckpt_dir)
        else []
    )

    if os.path.exists(final_path + ".zip"):
        print(f"  Resuming from: {final_path}.zip")
        model = AlgorithmClass.load(final_path, env=train_env, device=device)
    elif checkpoints:
        latest = os.path.join(ckpt_dir, checkpoints[-1])
        print(f"  Resuming from checkpoint: {latest}")
        model = AlgorithmClass.load(latest, env=train_env, device=device)
    else:
        print("  Starting from scratch")
        model = AlgorithmClass(
            "MlpPolicy",
            train_env,
            device=device,
            tensorboard_log=config.LOG_DIR,
            **{k: v for k, v in config.PPO_CONFIG.items() if k != "verbose"},
            verbose=1,
        )

    n_features = len([c for c in df.columns if c.startswith("feature_")])
    effective_steps = config.TOTAL_TIMESTEPS
    print(f"  Algorithm       : {config.ALGORITHM}")
    print(f"  Features        : {n_features}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Device          : {device}")
    print(f"  Effective steps : {effective_steps:,}")
    print(f"  Steps per iter  : {n_envs * config.PPO_CONFIG['n_steps']:,}")

    # ── Train ─────────────────────────────────────────────────────
    callbacks = get_callbacks(eval_env=eval_env)

    print(f"\n=== Training ===")
    print("TensorBoard: tensorboard --logdir logs/\n")

    try:
        model.learn(
            total_timesteps=effective_steps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")

    final_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
    model.save(final_path)
    print(f"\n✅ Model saved to: {final_path}.zip")

    train_env.close()
    eval_env.close()
    return model, train_df, test_df


if __name__ == "__main__":
    train()
