"""
run.py — Single Entry Point
============================
Run any part of the project from one place.

Usage:
    python run.py features                        # Process raw CSV → features
    python run.py train                           # Train the model
    python run.py backtest                        # Run backtest + print metrics
    python run.py backtest --wfo                  # Walk-forward backtest
    python run.py backtest --render               # Backtest + save render logs
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

    python run.py forwardtest --data 2025.csv         # Forward test on unseen data
    python run.py forwardtest --data 2025.csv --render  # Forward test + render logs

    python run.py render --render-dir render_logs/train     # View train episodes
    python run.py render --render-dir render_logs/test      # View test episodes
    python run.py render --render-dir render_logs/forwardtest  # View forward test

Render workflow (no retraining needed):
    Step 1 — Backtest or forward test with --render:
        python run.py backtest --render
        python run.py forwardtest --data 2025.csv --render

    Step 2 — View results in browser:
        python run.py render --render-dir render_logs/test
        Opens http://127.0.0.1:5000
        Navigate between saved episodes with the arrow buttons.
        Press Ctrl+C to stop the server.
"""

import sys
import os
import argparse

from evaluation.backtest import run_full_dataset


def cmd_features(args):
    from features.pipeline import load_and_process

    df = load_and_process(save=True)
    print(
        f"\nDone. {len(df):,} candles, "
        f"{len([c for c in df.columns if c.startswith('feature_')])} features."
    )


def cmd_train(args):
    from training.train import run_all_phases

    run_all_phases()


def cmd_backtest(args):
    from evaluation.backtest import simple_backtest, walk_forward_backtest

    if args.wfo:
        walk_forward_backtest()
    else:
        simple_backtest(
            save_render=args.render,
            render_dir=args.render_dir,
        )


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
    Runs feature pipeline on the new data, then evaluates without any training.
    Optionally saves render logs with --render for visual episode inspection.
    """
    import config
    from features.pipeline import load_and_process
    from evaluation.backtest import _load_model, run_full_dataset
    from evaluation.metrics import compute_metrics, print_metrics

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

    # Load trained model — RecurrentPPO-aware
    model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model")
    for candidate in [model_path, model_path + ".zip"]:
        if os.path.exists(candidate):
            model_path = candidate
            break
    else:
        print("ERROR: No trained model found. Run: python run.py train")
        return

    print(f"\nLoading model : {model_path}")
    model = _load_model(model_path)

    # Render dir for forward test
    render_dir = os.path.join(args.render_dir, "forwardtest")

    print("\nRunning forward test...")
    episode = run_full_dataset(
        model,
        df,
        mode="eval",
        save_render=args.render,
        render_dir=render_dir,
    )
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
        wr = (metrics.get("win_rate_pct") or 0) / 100
        print(f"  Avg win            : {avg_win:>8.3f}%")
        print(f"  Avg loss           : {avg_loss:>8.3f}%")
        print(f"  Expectancy/trade   : {wr * avg_win + (1 - wr) * avg_loss:>8.3f}%")

    if args.render:
        print(f"\n✅ Render logs saved.")
        print(f"   View with:  python run.py render --render-dir {render_dir}")


def cmd_render(args):
    """
    Launch the gym-trading-env Renderer to visually inspect saved episodes.

    Opens http://127.0.0.1:5000 showing for each saved episode:
      - Interactive candlestick chart (zoom/pan)
      - Portfolio valuation curve
      - Position timeline (long/flat/short colouring)
      - Custom metrics panel (win rate, profit factor, drawdown, RR, etc.)

    Navigate between episodes using the arrow buttons in the top-right.
    Press Ctrl+C to stop the server.

    Usage:
        python run.py render --render-dir render_logs/train
        python run.py render --render-dir render_logs/test
        python run.py render --render-dir render_logs/forwardtest
    """
    import pandas as pd
    from gym_trading_env.renderer import Renderer

    render_dir = args.render_dir
    if not os.path.isdir(render_dir):
        print(f"ERROR: '{render_dir}' does not exist.")
        print(f"Generate render logs first with the --render flag:")
        print(f"  python run.py backtest --render")
        print(f"  python run.py forwardtest --data X.csv --render")
        return

    pkl_files = [f for f in os.listdir(render_dir) if f.endswith(".pkl")]
    if not pkl_files:
        print(f"ERROR: No .pkl episode files found in '{render_dir}'.")
        print("Generate them with:  python run.py backtest --render")
        return

    print(f"\n=== gym-trading-env Renderer ===")
    print(f"  Episodes  : {len(pkl_files)}")
    print(f"  Source    : {os.path.abspath(render_dir)}")
    print(f"  URL       : http://127.0.0.1:5000")
    print(f"  Stop      : Ctrl+C\n")

    renderer = Renderer(render_logs_dir=render_dir)

    # ── Custom metrics ────────────────────────────────────────────
    # Each function receives the episode's full DataFrame, which contains:
    #   open, high, low, close     — original OHLC data
    #   portfolio_valuation        — portfolio $ value at each step
    #   position                   — held position at each step (-1 / 0 / +1)
    #   feature_*                  — all ICT/SMC feature columns
    #   DatetimeIndex              — UTC timestamps
    #
    # The function must return a string — whatever you return is shown
    # verbatim in the metrics panel next to the chart.
    # ─────────────────────────────────────────────────────────────

    def _trades_from_df(df):
        """Extract (entry_price, exit_price, direction, pnl) tuples from position column."""
        pos = df["position"].values
        prices = df["close"].values
        trades = []
        entry_price = None
        entry_dir = 0
        for i in range(1, len(pos)):
            prev, curr = pos[i - 1], pos[i]
            if prev == 0 and curr != 0:  # entry
                entry_price = prices[i]
                entry_dir = curr
            elif prev != 0 and (curr == 0 or curr != prev):  # exit
                if entry_price is not None:
                    pnl = (prices[i] - entry_price) / entry_price * entry_dir
                    trades.append((entry_price, prices[i], entry_dir, pnl))
                entry_price = prices[i] if curr != 0 else None
                entry_dir = curr
        return trades

    def metric_total_return(df):
        pv = df["portfolio_valuation"]
        ret = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
        return f"{ret:+.2f}%"

    def metric_annualised_return(df):
        pv = df["portfolio_valuation"]
        ret = pv.iloc[-1] / pv.iloc[0]
        try:
            span = df.index[-1] - df.index[0]
            if span.total_seconds() < 86400:
                return "n/a (< 1 day)"
            years = span / pd.Timedelta(days=365)
            ann = (ret ** (1.0 / years) - 1) * 100
            return f"{ann:+.2f}%"
        except Exception:
            return "n/a"

    def metric_max_drawdown(df):
        pv = df["portfolio_valuation"].values
        peak = pd.Series(pv).cummax().values
        dd = (pv - peak) / peak
        return f"{dd.min() * 100:.2f}%"

    def metric_win_rate(df):
        trades = _trades_from_df(df)
        if not trades:
            return "n/a (no trades)"
        wins = sum(1 for *_, pnl in trades if pnl > 0)
        return f"{wins / len(trades) * 100:.1f}%  ({wins}/{len(trades)})"

    def metric_profit_factor(df):
        trades = _trades_from_df(df)
        if not trades:
            return "n/a"
        gross_win = sum(pnl for *_, pnl in trades if pnl > 0)
        gross_loss = abs(sum(pnl for *_, pnl in trades if pnl <= 0))
        if gross_loss == 0:
            return "∞" if gross_win > 0 else "n/a"
        return f"{gross_win / gross_loss:.3f}"

    def metric_avg_rr(df):
        trades = _trades_from_df(df)
        if not trades:
            return "n/a"
        wins = [pnl for *_, pnl in trades if pnl > 0]
        losses = [pnl for *_, pnl in trades if pnl <= 0]
        if not wins or not losses:
            return "n/a"
        avg_w = sum(wins) / len(wins)
        avg_l = abs(sum(losses) / len(losses))
        return f"{avg_w / avg_l:.2f}R" if avg_l > 0 else "n/a"

    def metric_trade_count(df):
        trades = _trades_from_df(df)
        n_candles = len(df)
        pct = len(trades) / n_candles * 100 if n_candles > 0 else 0
        return f"{len(trades)}  ({pct:.1f}% of candles)"

    def metric_market_return(df):
        ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        return f"{ret:+.2f}%"

    def metric_killzone_rate(df):
        """What % of the model's trades entered during a killzone window."""
        trades = _trades_from_df(df)
        if not trades:
            return "n/a"
        # Re-detect entry candles where position flipped from 0
        pos = df["position"].values
        kz_col = "feature_killzone_active"
        has_kz = kz_col in df.columns
        kz_entries = 0
        total = 0
        for i in range(1, len(pos)):
            if pos[i - 1] == 0 and pos[i] != 0:
                total += 1
                if has_kz and df[kz_col].iloc[i] == 1:
                    kz_entries += 1
        if total == 0:
            return "n/a"
        if not has_kz:
            return f"{total} trades (killzone col missing)"
        return f"{kz_entries / total * 100:.1f}%  ({kz_entries}/{total})"

    # Register all metrics — they appear in the order added
    renderer.add_metric(name="Total Return", function=metric_total_return)
    renderer.add_metric(name="Annualised Return", function=metric_annualised_return)
    renderer.add_metric(name="Max Drawdown", function=metric_max_drawdown)
    renderer.add_metric(name="Win Rate", function=metric_win_rate)
    renderer.add_metric(name="Profit Factor", function=metric_profit_factor)
    renderer.add_metric(name="Avg RR Ratio", function=metric_avg_rr)
    renderer.add_metric(name="Trades Taken", function=metric_trade_count)
    renderer.add_metric(name="Market Return (B&H)", function=metric_market_return)
    renderer.add_metric(name="Killzone Entry Rate", function=metric_killzone_rate)

    renderer.run()


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
    "forwardtest": cmd_forwardtest,
    "render": cmd_render,
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

    # ── backtest / forwardtest flags ──────────────────────────────
    parser.add_argument(
        "--wfo",
        action="store_true",
        help="Walk-forward backtest (use with: backtest)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw CSV file (use with: forwardtest, features)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help=(
            "Save gym-trading-env render logs after running.\n"
            "Use with: backtest, forwardtest\n"
            "Then view with: python run.py render --render-dir <dir>"
        ),
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default="render_logs",
        dest="render_dir",
        help=(
            "Root directory for render .pkl files (default: render_logs)\n"
            "  backtest saves to:     <render-dir>/train/  and  <render-dir>/test/\n"
            "  forwardtest saves to:  <render-dir>/forwardtest/\n"
            "  render reads from:     the exact path you pass"
        ),
    )

    # ── verify flags ──────────────────────────────────────────────
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
