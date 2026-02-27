"""
training/callbacks.py
======================
Stable-Baselines3 training callbacks.
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TrainingProgressCallback(BaseCallback):
    """
    Logs training progress to console every N steps.
    Shows: timestep, mean reward, portfolio performance.
    """

    def __init__(self, log_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq   = log_freq
        self.rewards    = []
        self.ep_rewards = []

    def _on_step(self) -> bool:
        # Collect reward
        if "rewards" in self.locals:
            self.rewards.extend(self.locals["rewards"].tolist())

        # Log periodically
        if self.num_timesteps % self.log_freq == 0 and self.rewards:
            mean_r = np.mean(self.rewards[-1000:])
            if self.verbose:
                print(
                    f"  Step {self.num_timesteps:>8,} | "
                    f"Mean reward (last 1k): {mean_r:+.6f}"
                )

        return True   # True = continue training


def get_callbacks(eval_env=None):
    """
    Build callback list for training.

    Includes:
        - Checkpoint: saves model every CHECKPOINT_FREQ steps
        - Progress:   logs to console every 10k steps
        - Eval:       evaluates on eval_env if provided
    """
    callbacks = []

    # Checkpoint
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    callbacks.append(CheckpointCallback(
        save_freq   = config.CHECKPOINT_FREQ,
        save_path   = config.MODEL_SAVE_PATH,
        name_prefix = "checkpoint",
        verbose     = 0,
    ))

    # Progress logger
    callbacks.append(TrainingProgressCallback(log_freq=10_000, verbose=1))

    # Eval callback (optional)
    if eval_env is not None:
        callbacks.append(EvalCallback(
            eval_env         = eval_env,
            best_model_save_path = config.MODEL_SAVE_PATH,
            log_path         = config.LOG_DIR,
            eval_freq        = 50_000,
            n_eval_episodes  = 3,
            deterministic    = True,
            verbose          = 1,
        ))

    return callbacks
