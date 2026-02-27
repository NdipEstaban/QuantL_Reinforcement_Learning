"""
run.py — Single Entry Point
============================
Run any part of the project from one place.

Usage:
    python run.py features                        # Process raw CSV → features
    python run.py train                           # Train the model
    python run.py backtest                        # Run backtest + print metrics
    python run.py backtest --wfo                  # Walk-forward backtest
    python run.py dashboard                       # Open interactive Dash dashboard
    python run.py all                             # features → train → backtest

    python run.py verify                          # Feature chart, 200 candles from index 200
    python run.py verify --candles 400            # Show 400 candles
    python run.py verify --candles 300 --start 1000   # Start at candle 1000
    python run.py verify --section structure      # Show only structure levels
    python run.py verify --section zones          # Show only FVG/OB/BB zones
    python run.py verify --section bias           # Show only bias subplot
    python run.py verify --section bos            # Show only BOS/CHoCH events
    python run.py verify --stats                  # Print feature coverage report only
    python run.py verify --save chart.html        # Save to HTML instead of opening browser
"""

import sys
import os
import argparse


def cmd_features(args):
    from features.pipeline import load_and_process

    df = load_and_process(save=True)
    print(
        f"\nDone. {len(df):,} candles, "
        f"{len([c for c in df.columns if c.startswith('feature_')])} features."
    )


def cmd_train(args):
    from training.train import train

    train()


def cmd_backtest(args):
    from evaluation.backtest import simple_backtest, walk_forward_backtest

    if args.wfo:
        walk_forward_backtest()
    else:
        simple_backtest()


def cmd_dashboard(args):
    from visualization.dashboard import create_app

    print("Dashboard: http://127.0.0.1:8050")
    app = create_app()
    app.run(debug=False, port=8050)


def cmd_verify(args):
    from visualization.verify_features import (
        load_data,
        build_chart,
        print_feature_stats,
    )

    candles = getattr(args, "candles", 200)
    start = getattr(args, "start", 200)
    section = getattr(args, "section", "all")
    stats = getattr(args, "stats", False)
    save = getattr(args, "save", None)

    print(f"\n=== Feature Verification Chart ===")
    print(f"  Candles : {candles}")
    print(f"  Start   : candle index {start} (skips ATR/structure warmup)")
    print(f"  Section : {section}")

    print("\nLoading processed features...")
    df = load_data(start, candles)

    # Always print the coverage report — catches broken features immediately
    print_feature_stats(df)

    if stats:
        return  # stats-only mode, no chart

    print("Building interactive chart...")
    fig = build_chart(df, section=section)

    if save:
        path = save if save.endswith(".html") else save + ".html"
        fig.write_html(path)
        print(f"\n✅ Chart saved to: {path}")
        print("   Open in any browser to view interactively.")
    else:
        print("\nOpening chart in browser...")
        print("Tips:")
        print("  • Scroll to zoom, drag to pan")
        print("  • Click legend items to toggle features on/off")
        print("  • Hover over any candle to see all feature values")
        print("  • Double-click legend item to isolate it\n")
        fig.show()


def cmd_forwardtest(args):
    """
    Evaluate model on a completely new CSV file never seen during training.
    Runs feature pipeline on the new data, then backtests without any training.
    """
    import config
    from features.pipeline import load_and_process
    from evaluation.backtest import run_episode, simple_backtest
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from env.trading_env import make_env
    from evaluation.metrics import compute_metrics, print_metrics
    import pandas as pd

    if not args.data:
        print("ERROR: Provide path to new CSV with --data path/to/new_data.csv")
        return

    print(f"\n=== Forward Test on Unseen Data ===")
    print(f"  Data file : {args.data}")

    # Process features on the new data — same pipeline, no fitting
    print("\nRunning feature pipeline on new data...")
    df = load_and_process(
        args.data, save=False
    )  # save=False — don't overwrite training data
    print(f"  Candles   : {len(df):,}")
    print(f"  Period    : {df.index[0]} → {df.index[-1]}")
    print(f"  Features  : {len([c for c in df.columns if c.startswith('feature_')])}")

    # Load trained model
    model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
    for candidate in [model_path, model_path + ".zip"]:
        if os.path.exists(candidate):
            model_path = candidate
            break
    else:
        print("ERROR: No trained model found. Run: python run.py train")
        return

    print(f"\nLoading model : {model_path}")
    model = PPO.load(model_path, device="cpu")

    # Run episode on entire new dataset
    print("\nRunning forward test...")
    env = Monitor(make_env(df, mode="eval"))
    episode = run_episode(model, env, deterministic=True)
    metrics = compute_metrics(episode["portfolio_values"], episode["trades"])

    print_metrics(metrics, title="FORWARD TEST — Unseen Data")

    n_trades = len(episode["trades"])
    n_candles = len(episode["positions"])
    bh = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    print(f"  Buy & Hold         : {bh:>8.2f}%")
    print(f"  Trades taken       : {n_trades}")
    print(
        f"  Trade rate         : {n_trades / max(n_candles, 1) * 100:.1f}% of candles"
    )

    if n_trades > 0:
        wins = [t for t in episode["trades"] if t["pnl"] > 0]
        losses = [t for t in episode["trades"] if t["pnl"] <= 0]
        avg_win = sum(t["pnl"] for t in wins) / max(len(wins), 1) * 100
        avg_loss = sum(t["pnl"] for t in losses) / max(len(losses), 1) * 100
        print(f"  Avg win            : {avg_win:>8.3f}%")
        print(f"  Avg loss           : {avg_loss:>8.3f}%")
        print(
            f"  Expectancy/trade   : {(metrics.get('win_rate',0)/100*avg_win + (1-metrics.get('win_rate',0)/100)*avg_loss):>8.3f}%"
        )


def cmd_all(args):
    print("=== Step 1: Feature Pipeline ===")
    cmd_features(args)
    print("\n=== Step 2: Training ===")
    cmd_train(args)
    print("\n=== Step 3: Backtest ===")
    cmd_backtest(args)


COMMANDS = {
    "features": cmd_features,
    "train": cmd_train,
    "backtest": cmd_backtest,
    "dashboard": cmd_dashboard,
    "verify": cmd_verify,
    "forwardtest" : cmd_forwardtest,
    "all": cmd_all,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forex RL Bot",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=list(COMMANDS.keys()),
        help="What to run",
    )

    # backtest flags
    parser.add_argument(
        "--wfo", action="store_true", help="Walk-forward backtest (use with: backtest)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw CSV file (use with: features)",
    )

    # verify flags
    parser.add_argument(
        "--candles",
        type=int,
        default=200,
        help="Number of candles to display (use with: verify, default 200)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=200,
        help="Start candle index (use with: verify, default 200)",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="all",
        choices=["all", "structure", "zones", "bias", "session", "bos"],
        help="Feature group to show (use with: verify, default all)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print feature coverage report only, no chart (use with: verify)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save chart to HTML file (use with: verify)",
    )

    args = parser.parse_args()

    if args.data:
        import config

        config.DATA_FILE = args.data

    COMMANDS[args.command](args)
