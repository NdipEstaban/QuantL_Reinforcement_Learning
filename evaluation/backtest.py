"""
evaluation/backtest.py
=======================
Walk-forward backtesting and model evaluation.

Usage:
    python run.py backtest
    python run.py backtest --wfo
"""

import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from features.pipeline import load_processed
from env.trading_env import make_env
from evaluation.metrics import compute_metrics, print_metrics
from training.reward import get_reward_function

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


# ── helpers ───────────────────────────────────────────────────────


def _make_single_env(df, mode="eval"):
    """Single env wrapped correctly for SB3 model.predict()."""
    return Monitor(make_env(df, mode=mode))


def _make_vec_env(df, mode="train"):
    """Single-process VecEnv for fine-tuning (no multiprocess overhead)."""
    return VecMonitor(DummyVecEnv([lambda: make_env(df, mode=mode)]))


def _make_eval_env(df, episode_len):
    """
    Create a deterministic eval env with an explicit episode length.
    gym-trading-env requires len(df) > episode_len to pick a start index.
    This function sets max_episode_duration directly so we control it.
    """
    reward_fn = get_reward_function(config.REWARD_FUNCTION)
    env = gym.make(
        "TradingEnv",
        name="FOREX_EVAL",
        df=df,
        positions=config.POSITIONS,
        trading_fees=config.TRADING_FEES,
        borrow_interest_rate=config.BORROW_INTEREST_RATE,
        portfolio_initial_value=config.PORTFOLIO_INITIAL_VALUE,
        initial_position=0,
        max_episode_duration=episode_len,
        reward_function=reward_fn,
        verbose=0,
    )
    return Monitor(env)


# ── episode runner ────────────────────────────────────────────────


def run_episode(model, env, deterministic: bool = True) -> dict:
    """
    Run one episode. env must be a plain (non-vec) Monitor-wrapped env.

    Supports both PPO (stateless) and RecurrentPPO (stateful LSTM).
    For RecurrentPPO, lstm_states are carried across timesteps and reset
    at episode boundaries via episode_starts flag.
    """
    obs, info = env.reset()
    done = False
    truncated = False
    portfolio_values = []
    positions = []
    trades = []
    entry_price = None
    entry_pos = 0

    # LSTM state tracking — None for standard PPO (ignored), initialised
    # automatically by RecurrentPPO on first predict() call
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)  # True = start of episode

    while not done and not truncated:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        episode_starts = np.zeros((1,), dtype=bool)  # not start after first step
        obs, reward, done, truncated, info = env.step(action)

        pv = info.get("portfolio_valuation", config.PORTFOLIO_INITIAL_VALUE)
        pos = info.get("position", 0)
        portfolio_values.append(float(pv))
        positions.append(float(pos))

        price = info.get("data_close", info.get("close", info.get("price", None)))

        if price is not None:
            price = float(price)
            pos = float(pos)
            if entry_pos == 0 and pos != 0:
                entry_price = price
                entry_pos = pos
            elif entry_pos != 0 and (pos == 0 or pos != entry_pos):
                if entry_price is not None:
                    pnl_pct = (price - entry_price) / entry_price * entry_pos
                    trades.append(
                        {
                            "entry": entry_price,
                            "exit": price,
                            "pnl": pnl_pct,
                            "direction": entry_pos,
                        }
                    )
                entry_price = price if pos != 0 else None
                entry_pos = pos

    return {
        "portfolio_values": portfolio_values,
        "positions": positions,
        "trades": trades,
    }


# ── full dataset runner ───────────────────────────────────────────


def run_full_dataset(model, df, mode="eval") -> dict:
    """
    Run the model across the ENTIRE dataset by running consecutive episodes.

    Portfolio values are NOT compounded across episodes.
    POSITIONS=[-1,0,1] means 100% deployment per trade. At 60% win rate
    with ~300 trades/episode, each episode returns ~13%. Compounding 68
    episodes gives 1.13^68 = $55M — a meaningless number.

    Instead:
      - Each episode resets to $10,000 (independent return measurement)
      - Portfolio curve is additive (P&L stacked), not multiplicative
      - Per-episode median/mean return printed for consistency check
      - All trades collected for aggregate win rate / profit factor

    crash fix: chunk_size = ep_len * 2 so gym-trading-env reset() has
    room to pick a start index (needs len(df) > max_episode_duration).
    """
    ep_len = config.MAX_EPISODE_DURATION
    chunk_size = ep_len * 2
    step_size = ep_len

    starts = list(range(0, len(df) - chunk_size, step_size))
    if not starts:
        starts = [0]
        chunk_size = len(df)

    print(
        f"  Running {len(starts)} consecutive episodes "
        f"covering {min(len(starts) * step_size, len(df)):,} "
        f"of {len(df):,} candles..."
    )

    init = config.PORTFOLIO_INITIAL_VALUE
    all_trades = []
    all_positions = []
    episode_returns = []
    episode_max_dds = []
    additive_pv = []  # P&L stacked additively — no exponential compounding

    for start in starts:
        chunk = df.iloc[start : start + chunk_size]
        env = _make_eval_env(chunk, episode_len=ep_len)
        ep = run_episode(model, env, deterministic=True)
        env.close()

        pv = np.array(ep["portfolio_values"])
        if len(pv) > 0:
            ep_return = (pv[-1] - pv[0]) / max(pv[0], 1e-8)
            episode_returns.append(ep_return)

            peak = np.maximum.accumulate(pv)
            dd = (pv - peak) / np.maximum(peak, 1e-8)
            episode_max_dds.append(float(dd.min()))

            # Additive stacking: shift episode P&L to start from last value
            # pv - pv[0] = P&L of this episode starting at 0
            # + offset   = continue from previous endpoint
            offset = additive_pv[-1] if additive_pv else init
            ep_pnl = pv - pv[0]  # P&L curve starting at 0
            ep_shifted = ep_pnl + offset  # shift to current base
            additive_pv.extend(ep_shifted.tolist())

        all_trades.extend(ep["trades"])
        all_positions.extend(ep["positions"])

    if episode_returns:
        median_ret = float(np.median(episode_returns)) * 100
        mean_ret = float(np.mean(episode_returns)) * 100
        pos_eps = sum(1 for r in episode_returns if r > 0)
        print(f"  Per-episode median return : {median_ret:>+.2f}%")
        print(f"  Per-episode mean return   : {mean_ret:>+.2f}%")
        print(f"  Profitable episodes       : {pos_eps}/{len(episode_returns)}")

    return {
        "portfolio_values": additive_pv,
        "positions": all_positions,
        "trades": all_trades,
        "episode_returns": episode_returns,
        "episode_max_dds": episode_max_dds,
    }


# ── simple backtest ───────────────────────────────────────────────


def simple_backtest(model_path: str = None) -> dict:
    df = load_processed()
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")

    for path in [model_path, model_path + ".zip"]:
        if os.path.exists(path):
            model_path = path
            break
    else:
        raise FileNotFoundError(
            f"No model found at {model_path}. Run: python run.py train"
        )

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    results = {}
    for name, subset in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\nRunning {name} ({len(subset):,} candles)...")
        episode = run_full_dataset(model, subset, mode="eval")
        metrics = compute_metrics(episode["portfolio_values"], episode["trades"])
        print_metrics(metrics, title=f"{name} Results")

        bh = (
            (subset["close"].iloc[-1] - subset["close"].iloc[0])
            / subset["close"].iloc[0]
            * 100
        )
        n_trades = len(episode["trades"])
        n_candles = len(episode["positions"])
        print(f"  Buy & Hold         : {bh:>8.2f}%")
        print(f"  Trades taken       : {n_trades}")
        print(f"  Trade rate         : {n_trades/max(n_candles,1)*100:.1f}% of candles")

        results[name.lower()] = {
            "metrics": metrics,
            "episode": episode,
            "df": subset,
        }

    return results


# ── walk-forward backtest ─────────────────────────────────────────


def walk_forward_backtest(model_path: str = None) -> list:
    """
    Walk-forward validation.
    Each fold fine-tunes from the base model for 200k steps,
    then evaluates on the next out-of-sample period.
    """
    df = load_processed()
    fold_size = len(df) // config.WFO_FOLDS
    all_metrics = []

    base_path = model_path or os.path.join(config.MODEL_SAVE_PATH, "final_model")
    for candidate in [base_path, base_path + ".zip"]:
        if os.path.exists(candidate):
            base_path = candidate
            break
    else:
        base_path = None

    print(f"\n=== Walk-Forward Backtest ({config.WFO_FOLDS} folds) ===")
    if base_path:
        print(f"  Base model : {base_path}")
        print(f"  Strategy   : fine-tune 200k steps per fold")
    else:
        print("  WARNING: No trained model found. Run python run.py train first.")

    for fold in range(config.WFO_FOLDS - 1):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, len(df))

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        print(f"\nFold {fold + 1}/{config.WFO_FOLDS - 1}")
        print(
            f"  Train : {train_df.index[0].date()} → "
            f"{train_df.index[-1].date()} ({len(train_df):,} candles)"
        )
        print(
            f"  Test  : {test_df.index[0].date()} → "
            f"{test_df.index[-1].date()} ({len(test_df):,} candles)"
        )

        train_env = _make_vec_env(train_df, mode="train")

        if base_path:
            print(f"  Fine-tuning for 200,000 steps...")
            model = PPO.load(base_path, env=train_env, device="cpu")
            model.learn(
                total_timesteps=200_000,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        else:
            scratch_steps = min(200_000, len(train_df) * 5)
            print(f"  Training from scratch for {scratch_steps:,} steps...")
            ppo_kwargs = {k: v for k, v in config.PPO_CONFIG.items() if k != "verbose"}
            model = PPO("MlpPolicy", train_env, device="cpu", verbose=0, **ppo_kwargs)
            model.learn(total_timesteps=scratch_steps, progress_bar=False)

        train_env.close()

        # Evaluate across full fold using chained episodes
        episode = run_full_dataset(model, test_df, mode="eval")
        metrics = compute_metrics(episode["portfolio_values"], episode["trades"])
        print_metrics(metrics, title=f"Fold {fold + 1} Out-of-Sample")

        n_trades = len(episode["trades"])
        n_candles = len(episode["positions"])
        bh = (
            (test_df["close"].iloc[-1] - test_df["close"].iloc[0])
            / test_df["close"].iloc[0]
            * 100
        )
        print(f"  Trades taken : {n_trades}")
        print(f"  Trade rate   : {n_trades/max(n_candles,1)*100:.1f}% of candles")
        print(f"  Buy & Hold   : {bh:.2f}%")

        all_metrics.append(metrics)

    print(f"\n{'═'*47}")
    print("  Walk-Forward Summary")
    print(f"{'═'*47}")
    avg_return = np.mean([m["total_return_pct"] for m in all_metrics])
    avg_sharpe = np.mean([m["sharpe_ratio"] for m in all_metrics])
    avg_dd = np.mean([m["max_drawdown_pct"] for m in all_metrics])
    consistent = sum(1 for m in all_metrics if m["total_return_pct"] > 0)
    print(f"  Avg Return    : {avg_return:>8.2f}%")
    print(f"  Avg Sharpe    : {avg_sharpe:>8.3f}")
    print(f"  Avg Max DD    : {avg_dd:>8.2f}%")
    print(f"  Profitable    : {consistent}/{len(all_metrics)} folds")
    print(f"{'═'*47}")

    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wfo", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.wfo:
        walk_forward_backtest(args.model)
    else:
        simple_backtest(args.model)
