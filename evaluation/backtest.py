"""
evaluation/backtest.py
=======================
Walk-forward backtesting and model evaluation.

Usage:
    python run.py backtest                              # Run backtest
    python run.py backtest --wfo                        # Walk-forward backtest
    python run.py backtest --render                     # Backtest + save render logs
    python run.py forwardtest --data X.csv --render     # Forward test + render logs

Render workflow:
    Step 1 — Run with --render flag:
        python run.py backtest --render
        Saves .pkl files to:  render_logs/train/   render_logs/test/

    Step 2 — Launch the viewer:
        python run.py render --render-dir render_logs/train
        python run.py render --render-dir render_logs/test
        python run.py render --render-dir render_logs/forwardtest
        Opens http://127.0.0.1:5000  (press Ctrl+C to stop)
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

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


# ══════════════════════════════════════════════════════════════════
# MODEL LOADER  (RecurrentPPO-aware)
# ══════════════════════════════════════════════════════════════════


def _load_model(model_path: str):
    """
    Load model using the correct algorithm class.

    Critical: PPO.load() on a RecurrentPPO checkpoint silently drops every
    LSTM weight and falls back to a blank MLP policy with no error or warning.
    Always use RecurrentPPO.load() when config.ALGORITHM == "RecurrentPPO".
    """
    algo = getattr(config, "ALGORITHM", "PPO")
    if algo == "RecurrentPPO":
        try:
            from sb3_contrib import RecurrentPPO

            return RecurrentPPO.load(model_path, device="cpu")
        except ImportError:
            print("  ⚠  sb3-contrib not installed — falling back to PPO.load().")
            print("     LSTM weights will be lost. Fix: pip install sb3-contrib")
    from stable_baselines3 import PPO

    return PPO.load(model_path, device="cpu")


# ══════════════════════════════════════════════════════════════════
# ENV HELPERS
# ══════════════════════════════════════════════════════════════════


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

    Returns Monitor(TradingEnv).
    Access the raw TradingEnv via env.unwrapped — needed for save_for_render().
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


# ══════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════


def run_episode(
    model,
    env,
    deterministic: bool = True,
    save_render: bool = False,
    render_dir: str = "render_logs",
) -> dict:
    """
    Run one episode. env must be a plain (non-vec) Monitor-wrapped env.

    Supports both PPO (stateless) and RecurrentPPO (stateful LSTM).
    For RecurrentPPO, lstm_states are carried across timesteps and reset
    at episode boundaries via the episode_starts flag.

    Parameters
    ----------
    save_render : When True, saves a render snapshot after the episode ends.
                  gym-trading-env's save_for_render() writes a numbered .pkl
                  to render_dir/ that the Renderer can load and display as an
                  interactive candlestick + portfolio chart in the browser.

                  Must be called on env.unwrapped (the raw TradingEnv), NOT
                  on the Monitor wrapper. Gymnasium's .unwrapped property
                  traverses the full wrapper chain transparently.

                  Requirement: the DataFrame passed to TradingEnv must contain
                  open, high, low, close columns. Our pipeline always keeps
                  these intact, so save_for_render will never assert-fail. ✓

                  Multiple episodes writing to the same render_dir are
                  auto-numbered by the library — nothing overwrites.

    render_dir  : Directory to write .pkl snapshot files into.
                  Use different subdirs per run (train/test/forwardtest) so
                  the Renderer can be pointed at each independently.
    """
    obs, info = env.reset()
    done = False
    truncated = False
    portfolio_values = []
    positions = []
    trades = []
    entry_price = None
    entry_pos = 0

    # LSTM state — None for plain PPO (silently ignored by SB3).
    # RecurrentPPO auto-initialises on the first predict() call.
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)  # True = first step of episode

    while not done and not truncated:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        episode_starts = np.zeros((1,), dtype=bool)  # False for all subsequent steps
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

    # ── Render snapshot ───────────────────────────────────────────
    if save_render:
        try:
            os.makedirs(render_dir, exist_ok=True)
            
            # Store original index if it exists
            if hasattr(env.unwrapped, 'df') and env.unwrapped.df is not None:
                original_index = env.unwrapped.df.index.copy()
                # Convert to naive if it has timezone
                if original_index.tz is not None:
                    env.unwrapped.df.index = original_index.tz_localize(None)
            
            env.unwrapped.save_for_render(dir=render_dir)
            
            # Restore original index if we changed it
            if hasattr(env.unwrapped, 'df') and env.unwrapped.df is not None:
                if 'original_index' in locals():
                    env.unwrapped.df.index = original_index
                    
        except AssertionError as e:
            print(f"  ⚠  save_for_render skipped (OHLC columns missing?): {e}")
        except Exception as e:
            print(f"  ⚠  save_for_render error — {type(e).__name__}: {e}")

    return {
        "portfolio_values": portfolio_values,
        "positions": positions,
        "trades": trades,
    }


# ══════════════════════════════════════════════════════════════════
# FULL DATASET RUNNER
# ══════════════════════════════════════════════════════════════════


def run_full_dataset(
    model,
    df,
    mode: str = "eval",
    save_render: bool = False,
    render_dir: str = "render_logs",
) -> dict:
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

    When save_render=True, each episode writes a numbered .pkl to render_dir/.
    The Renderer loads all of them and lets you navigate between episodes
    using arrows in the browser UI.

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
    if save_render:
        print(f"  Render logs → {os.path.abspath(render_dir)}/")

    init = config.PORTFOLIO_INITIAL_VALUE
    all_trades = []
    all_positions = []
    episode_returns = []
    episode_max_dds = []
    additive_pv = []

    for start in starts:
        chunk = df.iloc[start : start + chunk_size]
        env = _make_eval_env(chunk, episode_len=ep_len)
        ep = run_episode(
            model,
            env,
            deterministic=True,
            save_render=save_render,
            render_dir=render_dir,
        )
        env.close()

        pv = np.array(ep["portfolio_values"])
        if len(pv) > 0:
            ep_return = (pv[-1] - pv[0]) / max(pv[0], 1e-8)
            episode_returns.append(ep_return)

            peak = np.maximum.accumulate(pv)
            dd = (pv - peak) / np.maximum(peak, 1e-8)
            episode_max_dds.append(float(dd.min()))

            offset = additive_pv[-1] if additive_pv else init
            ep_pnl = pv - pv[0]
            ep_shifted = ep_pnl + offset
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


# ══════════════════════════════════════════════════════════════════
# SIMPLE BACKTEST
# ══════════════════════════════════════════════════════════════════


def simple_backtest(
    model_path: str = None,
    save_render: bool = False,
    render_dir: str = "render_logs",
) -> dict:
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
    model = _load_model(model_path)

    results = {}
    for name, subset in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\nRunning {name} ({len(subset):,} candles)...")

        # Each split gets its own subdir so train and test .pkl files
        # stay separate — the Renderer can be pointed at either one.
        split_render_dir = os.path.join(render_dir, name.lower())

        episode = run_full_dataset(
            model,
            subset,
            mode="eval",
            save_render=save_render,
            render_dir=split_render_dir,
        )
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
            "render_dir": split_render_dir if save_render else None,
        }

    if save_render:
        print(f"\n✅ Render logs saved.")
        print(f"   Train  →  python run.py render --render-dir {render_dir}/train")
        print(f"   Test   →  python run.py render --render-dir {render_dir}/test")

    return results


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════


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
            model = _load_model(base_path)
            model.set_env(train_env)
            model.learn(
                total_timesteps=200_000,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        else:
            from stable_baselines3 import PPO

            scratch_steps = min(200_000, len(train_df) * 5)
            print(f"  Training from scratch for {scratch_steps:,} steps...")
            ppo_kwargs = {k: v for k, v in config.PPO_CONFIG.items() if k != "verbose"}
            model = PPO("MlpPolicy", train_env, device="cpu", verbose=0, **ppo_kwargs)
            model.learn(total_timesteps=scratch_steps, progress_bar=False)

        train_env.close()

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


# ══════════════════════════════════════════════════════════════════
# CLI (direct usage)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wfo", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--render", action="store_true", help="Save render logs after running"
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default="render_logs",
        dest="render_dir",
        help="Directory to save render .pkl files (default: render_logs)",
    )
    args = parser.parse_args()

    if args.wfo:
        walk_forward_backtest(args.model)
    else:
        simple_backtest(
            model_path=args.model,
            save_render=args.render,
            render_dir=args.render_dir,
        )
