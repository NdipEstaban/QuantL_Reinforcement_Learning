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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from features.pipeline import load_processed
from env.trading_env import make_env
from evaluation.metrics import compute_metrics, print_metrics

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


# ── episode runner ────────────────────────────────────────────────


def run_episode(model, env, deterministic: bool = True) -> dict:
    """
    Run one episode.  env must be a plain (non-vec) Monitor-wrapped env.

    gym-trading-env info key reference:
        portfolio_valuation  -- current portfolio value
        position             -- current position value (-1, 0, 1)
        data_close           -- close price (all df cols come back as data_{col})
    """
    obs, info = env.reset()
    done = False
    truncated = False

    portfolio_values = []
    positions = []
    trades = []
    entry_price = None
    entry_pos = 0
    _keys_printed = False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)

        # Print info keys on first step so we can verify correct extraction
        if not _keys_printed and info:
            numeric_keys = sorted(
                [
                    k
                    for k, v in info.items()
                    if isinstance(v, (int, float)) and not k.startswith("date_")
                ]
            )
            # print(f"  [info keys]: {numeric_keys[:15]}")
            _keys_printed = True

        pv = info.get("portfolio_valuation", config.PORTFOLIO_INITIAL_VALUE)
        pos = info.get("position", 0)
        portfolio_values.append(float(pv))
        positions.append(float(pos))

        # gym-trading-env returns df columns as data_{colname}
        price = info.get("data_close", info.get("close", info.get("price", None)))

        if price is not None:
            price = float(price)
            pos = float(pos)
            if entry_pos == 0 and pos != 0:
                entry_price = price
                entry_pos = pos
            elif entry_pos != 0 and (pos == 0 or pos != entry_pos):
                if entry_price is not None:
                    raw_pnl = (price - entry_price) / entry_price
                    pnl_pct = raw_pnl * entry_pos
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


# ── simple backtest ───────────────────────────────────────────────


def simple_backtest(model_path: str = None) -> dict:
    df = load_processed()
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")

    # Try with and without .zip
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
        env = _make_single_env(subset, mode="eval")
        episode = run_episode(model, env, deterministic=True)
        metrics = compute_metrics(episode["portfolio_values"], episode["trades"])
        print_metrics(metrics, title=f"{name} Results")

        bh = (
            (subset["close"].iloc[-1] - subset["close"].iloc[0])
            / subset["close"].iloc[0]
            * 100
        )
        print(f"  Buy & Hold         : {bh:>8.2f}%")

        n_trades = len(episode["trades"])
        n_candles = len(episode["positions"])
        print(f"  Trades taken       : {n_trades}")
        print(f"  Trade rate         : {n_trades/max(n_candles,1)*100:.1f}% of candles")

        results[name.lower()] = {"metrics": metrics, "episode": episode, "df": subset}

    return results


# ── walk-forward backtest ─────────────────────────────────────────


def walk_forward_backtest(model_path: str = None) -> list:
    """
    Walk-forward validation.

    If a trained base model exists, each fold fine-tunes from it (200k steps).
    If no model exists, trains from scratch per fold (warns that results
    will be unreliable — run train first).
    """
    df = load_processed()
    fold_size = len(df) // config.WFO_FOLDS
    all_metrics = []

    # Resolve base model path
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
        print("  WARNING: No trained model found.")
        print("  Run 'python run.py train' first for reliable results.")
        print("  Continuing with scratch training (46k steps) per fold...\n")

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

        # Build training env (single DummyVecEnv — multiprocess not needed
        # for short fine-tune runs, avoids spawn overhead on Windows)
        train_env = _make_vec_env(train_df, mode="train")

        if base_path:
            finetune_steps = 200_000
            print(f"  Fine-tuning for {finetune_steps:,} steps...")
            model = PPO.load(base_path, env=train_env, device="cpu")
            model.learn(
                total_timesteps=finetune_steps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        else:
            scratch_steps = min(200_000, len(train_df) * 5)
            print(f"  Training from scratch for {scratch_steps:,} steps...")
            ppo_kwargs = {k: v for k, v in config.PPO_CONFIG.items() if k != "verbose"}
            model = PPO(
                "MlpPolicy",
                train_env,
                device="cpu",
                verbose=0,
                **ppo_kwargs,
            )
            model.learn(
                total_timesteps=scratch_steps,
                progress_bar=False,
            )

        train_env.close()

        # Evaluate on out-of-sample fold
        test_env = _make_single_env(test_df, mode="eval")
        episode = run_episode(model, test_env, deterministic=True)
        metrics = compute_metrics(episode["portfolio_values"], episode["trades"])
        print_metrics(metrics, title=f"Fold {fold + 1} Out-of-Sample")

        n_trades = len(episode["trades"])
        n_candles = len(episode["positions"])
        bh = (
            (test_df["close"].iloc[-1] - test_df["close"].iloc[0])
            / test_df["close"].iloc[0]
            * 100
        )
        print(f"  Trades taken       : {n_trades}")
        print(f"  Trade rate         : {n_trades/max(n_candles,1)*100:.1f}% of candles")
        print(f"  Buy & Hold         : {bh:.2f}%")

        all_metrics.append(metrics)

    # Summary
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
