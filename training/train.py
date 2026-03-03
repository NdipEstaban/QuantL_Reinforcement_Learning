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

import json
from datetime import datetime, timezone


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


def train_phase(phase: dict) -> None:
    """
    Train one phase from config.TRAINING_PHASES.
    Each phase fully specifies algorithm, data, timesteps, and save path.
    """
    name = phase["name"]
    algo_name = phase.get("algorithm", config.ALGORITHM)
    data_file = phase["data_file"]
    timesteps = phase["timesteps"]
    resume_path = phase.get("resume_from")
    save_dir = phase.get("save_to", config.MODEL_SAVE_PATH)
    skip = phase.get("skip", False)

    if skip:
        print(f"\n{'─'*55}")
        print(f"  SKIPPING: {name}")
        print(f"{'─'*55}")
        return

    print(f"\n{'═'*55}")
    print(f"  PHASE: {name}")
    print(f"  {phase.get('description', '')}")
    print(f"  Algorithm  : {algo_name}")
    print(f"  Data       : {data_file}")
    print(f"  Timesteps  : {timesteps:,}")
    print(f"  Resume     : {resume_path or 'fresh'}")
    print(f"  Save to    : {save_dir}")
    print(f"{'═'*55}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    from features.pipeline import load_and_process

    # force_reprocess=True in phase dict to bypass cache (e.g. after changing features)
    force = phase.get("force_reprocess", False)
    df = load_and_process(data_file, save=True, force=force)
    _, test_df = split_data(df)

    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from env.trading_env import make_env

    train_env = VecMonitor(
        SubprocVecEnv(
            [lambda: make_env(df, mode="train") for _ in range(config.N_ENVS)]
        )
    )
    eval_env = VecMonitor(SubprocVecEnv([lambda: make_env(test_df, mode="eval")]))

    AlgorithmClass = get_algorithm(algo_name)
    is_recurrent = algo_name == "RecurrentPPO"

    # Per-phase ppo_config overrides the global config.PPO_CONFIG.
    # Only keys present in the phase dict are overridden — missing keys
    # fall through to config.PPO_CONFIG so you never have to repeat defaults.
    phase_ppo = phase.get("ppo_config", {})
    merged_ppo = {**config.PPO_CONFIG, **phase_ppo}  # phase wins on overlap
    ppo_kwargs = {k: v for k, v in merged_ppo.items() if k != "verbose"}

    print(f"  PPO config (merged):")
    for k, v in sorted(ppo_kwargs.items()):
        changed = " ← phase override" if k in phase_ppo else ""
        print(f"    {k:<18} = {v}{changed}")

    if is_recurrent:
        policy_class = "MlpLstmPolicy"
        policy_kwargs = dict(
            lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
            n_lstm_layers=config.LSTM_N_LAYERS,
            shared_lstm=False,
            enable_critic_lstm=True,
        )
    else:
        policy_class, policy_kwargs = "MlpPolicy", {}

    if resume_path and os.path.exists(resume_path):
        print(f"  Loading: {resume_path}")
        model = AlgorithmClass.load(
            resume_path,
            env=train_env,
            device="cpu",
            policy_kwargs=policy_kwargs if is_recurrent else None,
        )
        # Apply the merged (phase-overridden) hyperparams onto the loaded model
        _apply_config_hyperparams(model, merged_ppo)
    else:
        model = AlgorithmClass(
            policy_class,
            train_env,
            device="cpu",
            tensorboard_log=config.LOG_DIR,
            verbose=1,
            policy_kwargs=policy_kwargs,
            **ppo_kwargs,
        )

    callbacks = get_callbacks(eval_env=eval_env)
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=(resume_path is None),
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print(f"\n  Interrupted — saving partial checkpoint for {name}")

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"  ✅ Model: {final_path}.zip")

    _save_model_info(save_dir, algo_name, df)

    train_env.close()
    eval_env.close()


def _save_model_info(save_dir: str, algorithm: str, df) -> None:
    """Save model_info.json — read by deploy/predictor.py for obs feature order."""
    import gymnasium as gym

    try:
        import gym_trading_env  # noqa: F401

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
        obs_features = list(
            getattr(
                raw,
                "_obs_features",
                getattr(
                    raw,
                    "obs_features",
                    [c for c in raw.df.columns if c.startswith("feature_")],
                ),
            )
        )
        env.close()
    except Exception as e:
        print(f"  ⚠  TradingEnv probe failed ({e}) — using pipeline fallback")
        obs_features = [c for c in df.columns if c.startswith("feature_")]
        if "feature_close_pct" not in obs_features:
            obs_features.append("feature_close_pct")

    info = {
        "algorithm": algorithm,
        "obs_features": obs_features,
        "n_features": len(obs_features),
        "positions": list(config.POSITIONS),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    info_path = os.path.join(save_dir, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  ✅ model_info.json: {info_path} ({len(obs_features)} features)")


def run_all_phases(phases: list = None) -> None:
    """
    Run all training phases defined in config.TRAINING_PHASES.
    Called automatically by: python run.py train
    """
    phases = phases or getattr(config, "TRAINING_PHASES", [])

    if not phases:
        print("No TRAINING_PHASES in config.py — running single train()")
        train()
        return

    active_count = sum(1 for p in phases if not p.get("skip", False))
    print(f"\n{'═'*55}")
    print(f"  AUTOMATED TRAINING — {active_count}/{len(phases)} phases active")
    for i, p in enumerate(phases):
        status = "SKIP" if p.get("skip") else f"phase {i+1}"
        overrides = p.get("ppo_config", {})
        override_str = ""
        if overrides:
            override_str = (
                "  [" + ", ".join(f"{k}={v}" for k, v in overrides.items()) + "]"
            )
        print(f"    [{status:>7}]  {p['name']}{override_str}")
    print(f"{'═'*55}")

    for phase in phases:
        train_phase(phase)

    print(f"\n{'═'*55}")
    print(f"  ✅ All training complete")
    print(f"{'═'*55}")
