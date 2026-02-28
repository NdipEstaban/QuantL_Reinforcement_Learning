"""
training/train.py
==================
Training script supporting both PPO and RecurrentPPO.

RecurrentPPO (RPPO) adds an LSTM layer to the policy so the model can
carry memory across timesteps — remembering that a BOS fired 30 candles
ago, or that price has been trending in one direction for 2 hours.

Key differences from standard PPO:
  - Policy class : "MlpLstmPolicy" instead of "MlpPolicy"
  - policy_kwargs: lstm_hidden_size, n_lstm_layers
  - inference    : model.predict() requires lstm_states + episode_starts
                   (handled in backtest.py run_episode)
  - Training     : otherwise identical to PPO — same envs, callbacks, etc.

Install sb3-contrib once:
    pip install sb3-contrib

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
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"  Train: {len(train_df):,} candles | Test: {len(test_df):,} candles")
    return train_df, test_df


def get_algorithm(name: str):
    """Return algorithm class. Supports PPO and RecurrentPPO."""
    if name == "RecurrentPPO":
        try:
            from sb3_contrib import RecurrentPPO

            return RecurrentPPO
        except ImportError:
            raise ImportError(
                "sb3-contrib not installed.\n" "Run: pip install sb3-contrib"
            )
    from stable_baselines3 import PPO, A2C, SAC

    algs = {"PPO": PPO, "A2C": A2C, "SAC": SAC}
    if name not in algs:
        raise ValueError(
            f"Unknown algorithm '{name}'. Choose: PPO, RecurrentPPO, A2C, SAC"
        )
    return algs[name]


def _find_latest_checkpoint() -> str | None:
    ckpt_dir = config.MODEL_SAVE_PATH
    for name in ["final_model.zip", "final_model"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    zips = sorted(glob.glob(os.path.join(ckpt_dir, "*.zip")))
    return zips[-1] if zips else None


def _apply_config_hyperparams(model, ppo_config: dict) -> None:
    """
    Force config hyperparameters onto a loaded model.
    PPO.load() / RecurrentPPO.load() restore saved hyperparams from the .zip,
    ignoring config.py. This is critical for curriculum learning phases.
    """
    changed = []
    lr = ppo_config.get("learning_rate")
    clip = ppo_config.get("clip_range")
    ent = ppo_config.get("ent_coef")
    gamma = ppo_config.get("gamma")
    n_ep = ppo_config.get("n_epochs")

    if lr is not None:
        # Patch the schedule callable
        model.learning_rate = lambda _: lr
        # Patch the lr_schedule attribute SB3 uses internally
        if hasattr(model, "lr_schedule"):
            model.lr_schedule = lambda _: lr
        # Patch optimizer directly as belt-and-suspenders
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for pg in model.policy.optimizer.param_groups:
                pg["lr"] = lr
        changed.append(f"learning_rate={lr}")

    if clip is not None:
        model.clip_range = lambda _: clip
        changed.append(f"clip_range={clip}")

    if ent is not None:
        model.ent_coef = ent
        changed.append(f"ent_coef={ent}")

    if gamma is not None:
        model.gamma = gamma
        changed.append(f"gamma={gamma}")

    if n_ep is not None:
        model.n_epochs = n_ep
        changed.append(f"n_epochs={n_ep}")

    if changed:
        print(f"  Hyperparams overridden from config: {', '.join(changed)}")


def train(data_path: str = None):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

    # ── Load data ─────────────────────────────────────
    print("\n=== Loading Data ===")
    try:
        df = load_processed()
    except FileNotFoundError:
        print("No processed data found — running feature pipeline...")
        df = load_and_process(data_path, save=True)

    print("\n=== Splitting Data ===")
    train_df, test_df = split_data(df)

    # ── Environments ──────────────────────────────────
    print(f"\n=== Creating {config.N_ENVS} Parallel Environments ===")
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    is_recurrent = config.ALGORITHM == "RecurrentPPO"

    if is_recurrent:
        # RecurrentPPO requires SubprocVecEnv but the LSTM state must be
        # tracked per-env. SB3-contrib handles this automatically — we just
        # need to ensure envs are created the same way as standard PPO.
        print("  Mode: RecurrentPPO (LSTM policy)")
    else:
        print("  Mode: PPO (MLP policy)")

    def _make_train():
        return make_env(train_df, mode="train")

    def _make_eval():
        return make_env(test_df, mode="eval")

    train_env = VecMonitor(SubprocVecEnv([_make_train for _ in range(config.N_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv([_make_eval]))

    print(f"  Train envs : {config.N_ENVS} parallel processes")
    print(f"  Eval env   : 1 (deterministic)")

    # ── Model ─────────────────────────────────────────
    print(f"\n=== Building {config.ALGORITHM} Model ===")
    AlgorithmClass = get_algorithm(config.ALGORITHM)
    ppo_kwargs = {k: v for k, v in config.PPO_CONFIG.items() if k != "verbose"}

    # Policy class and kwargs depend on algorithm
    if is_recurrent:
        policy_class = "MlpLstmPolicy"
        policy_kwargs = dict(
            lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
            n_lstm_layers=config.LSTM_N_LAYERS,
            shared_lstm=False,  # separate LSTM for actor and critic
            enable_critic_lstm=True,
        )
    else:
        policy_class = "MlpPolicy"
        policy_kwargs = {}

    checkpoint = _find_latest_checkpoint()

    if checkpoint:
        print(f"  Resuming from: {checkpoint}")
        model = AlgorithmClass.load(
            checkpoint,
            env=train_env,
            device="cpu",
            policy_kwargs=policy_kwargs if is_recurrent else None,
        )
        _apply_config_hyperparams(model, config.PPO_CONFIG)
    else:
        print(f"  Starting from scratch")
        model = AlgorithmClass(
            policy_class,
            train_env,
            device="cpu",
            tensorboard_log=config.LOG_DIR,
            verbose=1,
            policy_kwargs=policy_kwargs,
            **ppo_kwargs,
        )

    n_features = len([c for c in df.columns if c.startswith("feature_")])
    steps_per_iter = config.N_ENVS * config.PPO_CONFIG.get("n_steps", 2048)
    lstm_info = (
        f"  LSTM hidden: {config.LSTM_HIDDEN_SIZE} × {config.LSTM_N_LAYERS} layers"
        if is_recurrent
        else ""
    )
    print(f"  Algorithm       : {config.ALGORITHM}")
    print(f"  Features        : {n_features}")
    print(f"  Parallel envs   : {config.N_ENVS}")
    print(f"  Device          : cpu")
    print(f"  Effective steps : {config.TOTAL_TIMESTEPS:,}")
    print(f"  Steps per iter  : {steps_per_iter:,}")
    if lstm_info:
        print(lstm_info)

    # ── Callbacks ─────────────────────────────────────
    callbacks = get_callbacks(eval_env=eval_env)

    # ── Train ──────────────────────────────────────────
    print(f"\n=== Training ===")
    print("TensorBoard: tensorboard --logdir logs/")

    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    # ── Save ───────────────────────────────────────────
    final_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
    model.save(final_path)
    print(f"\n✅ Model saved to: {final_path}.zip")

    return model, train_df, test_df


if __name__ == "__main__":
    train()
