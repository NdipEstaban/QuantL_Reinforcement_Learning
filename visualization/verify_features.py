"""
visualization/verify_features.py
==================================
Interactive candlestick chart with ALL ICT features overlaid.

Run via run.py:
    python run.py verify
    python run.py verify --candles 300
    python run.py verify --candles 300 --start 500
    python run.py verify --section structure
    python run.py verify --section zones
    python run.py verify --section bias
    python run.py verify --section bos
    python run.py verify --stats
    python run.py verify --save chart.html
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

C = {
    "bg": "#0d1117",
    "bg2": "#161b22",
    "grid": "#21262d",
    "text": "#c9d1d9",
    "text_dim": "#8b949e",
    "bull": "#26a65b",
    "bear": "#e74c3c",
    "sth": "#58a6ff",
    "stl": "#ff7b72",
    "ith": "#ffa657",
    "itl": "#d2a8ff",
    "lth": "#ffffff",
    "ltl": "#f0e68c",
    "bos_bull": "#3fb950",
    "bos_bear": "#f85149",
    "choch_bull": "#79c0ff",
    "choch_bear": "#ffa198",
    "ob_bull_fill": "rgba(38,166,91,0.20)",
    "ob_bull_line": "#26a65b",
    "ob_bear_fill": "rgba(231,76,60,0.20)",
    "ob_bear_line": "#e74c3c",
    "bb_bull_fill": "rgba(88,166,255,0.15)",
    "bb_bull_line": "#58a6ff",
    "bb_bear_fill": "rgba(255,123,114,0.15)",
    "bb_bear_line": "#ff7b72",
    "fvg_bull_fill": "rgba(63,185,80,0.10)",
    "fvg_bull_line": "#3fb950",
    "fvg_bear_fill": "rgba(248,81,73,0.10)",
    "fvg_bear_line": "#f85149",
    "killzone": "rgba(255,166,87,0.12)",
    "session_lon": "rgba(88,166,255,0.05)",
    "session_ny": "rgba(210,168,255,0.05)",
}


def load_data(start: int, n_candles: int) -> pd.DataFrame:
    from features.pipeline import load_processed

    df = load_processed()
    end = min(start + n_candles, len(df))
    start = max(0, end - n_candles)
    subset = df.iloc[start:end].copy()
    print(f"  Loaded {len(subset):,} candles: {subset.index[0]} to {subset.index[-1]}")
    return subset


def add_candlesticks(fig, df, row=1):
    x = list(range(len(df)))
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"].tolist(),
            high=df["high"].tolist(),
            low=df["low"].tolist(),
            close=df["close"].tolist(),
            name="Price",
            increasing_line_color=C["bull"],
            decreasing_line_color=C["bear"],
            increasing_fillcolor=C["bull"],
            decreasing_fillcolor=C["bear"],
            whiskerwidth=0.4,
            showlegend=True,
        ),
        row=row,
        col=1,
    )
    n = len(df)
    step = max(1, n // 15)
    tvs = list(range(0, n, step))
    tts = [str(df.index[i])[:13] for i in tvs]
    fig.update_xaxes(tickvals=tvs, ticktext=tts, tickangle=-45, row=row, col=1)


def add_structure_levels(fig, df, row=1):
    n = len(df)
    configs = [
        ("_sth_flag", "_sth_price", "STH", C["sth"], "dot", 1.2),
        ("_stl_flag", "_stl_price", "STL", C["stl"], "dot", 1.2),
        ("_ith_flag", "_ith_price", "ITH", C["ith"], "dash", 1.8),
        ("_itl_flag", "_itl_price", "ITL", C["itl"], "dash", 1.8),
        ("_lth_flag", "_lth_price", "LTH", C["lth"], "solid", 2.2),
        ("_ltl_flag", "_ltl_price", "LTL", C["ltl"], "solid", 2.2),
    ]
    shown = set()
    for flag_col, price_col, label, color, dash, lwidth in configs:
        if flag_col not in df.columns:
            continue
        flags = df[flag_col].values
        prices = df[price_col].values
        flag_idxs = [i for i in range(n) if flags[i]]
        for k, fi in enumerate(flag_idxs):
            p = prices[fi]
            if np.isnan(p):
                continue
            end_i = flag_idxs[k + 1] if k + 1 < len(flag_idxs) else n - 1
            show_leg = label not in shown
            shown.add(label)
            fig.add_trace(
                go.Scatter(
                    x=[fi, end_i],
                    y=[p, p],
                    mode="lines",
                    line=dict(color=color, width=lwidth, dash=dash),
                    name=label,
                    showlegend=show_leg,
                    hovertemplate=f"<b>{label}</b>: {p:.5f}<extra></extra>",
                    legendgroup=label,
                ),
                row=row,
                col=1,
            )


def add_bos_choch(fig, df, row=1):
    n = len(df)
    atr = df["_atr"].values
    events = [
        ("feature_bos_bull_st", "ST BOS up", C["bos_bull"], True, 0.5),
        ("feature_bos_bear_st", "ST BOS dn", C["bos_bear"], False, 0.5),
        ("feature_bos_bull_it", "IT BOS up", C["bos_bull"], True, 1.5),
        ("feature_bos_bear_it", "IT BOS dn", C["bos_bear"], False, 1.5),
        ("feature_bos_bull_lt", "LT BOS up", C["choch_bull"], True, 2.5),
        ("feature_bos_bear_lt", "LT BOS dn", C["choch_bear"], False, 2.5),
        ("feature_choch_bull", "CHoCH up", "#ffd700", True, 3.5),
        ("feature_choch_bear", "CHoCH dn", "#ffd700", False, 3.5),
    ]
    shown = set()
    for col, label, color, is_bull, mult in events:
        if col not in df.columns:
            continue
        idxs = [i for i in range(n) if df[col].iloc[i] == 1]
        if not idxs:
            continue
        if is_bull:
            ys = [df["high"].iloc[i] + atr[i] * mult for i in idxs]
            sym = "triangle-up"
        else:
            ys = [df["low"].iloc[i] - atr[i] * mult for i in idxs]
            sym = "triangle-down"
        size = 8 if "ST" in label else 11 if "IT" in label else 13
        show_leg = label not in shown
        shown.add(label)
        fig.add_trace(
            go.Scatter(
                x=idxs,
                y=ys,
                mode="markers",
                marker=dict(
                    symbol=sym,
                    color=color,
                    size=size,
                    line=dict(color="white", width=0.5),
                ),
                name=label,
                showlegend=show_leg,
                hovertemplate=f"<b>{label}</b> candle %{{x}}<extra></extra>",
                legendgroup=label,
            ),
            row=row,
            col=1,
        )


def add_fvg_zones(fig, df, row=1):
    if "_fvg_top" not in df.columns:
        return
    n = len(df)
    fvg_top = df["_fvg_top"].values
    fvg_bot = df["_fvg_bottom"].values
    fvg_dir = df["_fvg_direction"].values
    fvg_fill = df["_fvg_filled_pct"].fillna(0).values
    added_bull = False
    added_bear = False
    i = 0
    while i < n:
        if np.isnan(fvg_top[i]):
            i += 1
            continue
        t = fvg_top[i]
        b = fvg_bot[i]
        d = int(fvg_dir[i])
        j = i + 1
        while j < n and not np.isnan(fvg_top[j]) and fvg_top[j] == t:
            j += 1
        fill = fvg_fill[j - 1]
        alpha = max(0.04, 0.15 * (1 - fill))
        if d == 1:
            fc = f"rgba(63,185,80,{alpha:.2f})"
            lc = C["fvg_bull_line"]
            lbl = "FVG Bull"
            show = not added_bull
            added_bull = True
        else:
            fc = f"rgba(248,81,73,{alpha:.2f})"
            lc = C["fvg_bear_line"]
            lbl = "FVG Bear"
            show = not added_bear
            added_bear = True
        fig.add_trace(
            go.Scatter(
                x=[i, j, j, i, i],
                y=[t, t, b, b, t],
                fill="toself",
                fillcolor=fc,
                line=dict(color=lc, width=0.8, dash="dot"),
                mode="lines",
                name=lbl,
                showlegend=show,
                legendgroup=lbl,
                hovertemplate=f"<b>{lbl}</b> T:{t:.5f} B:{b:.5f} Fill:{fill*100:.0f}%<extra></extra>",
            ),
            row=row,
            col=1,
        )
        i = j


def _draw_box(
    fig,
    x0,
    x1,
    high,
    low,
    fill_color,
    line_color,
    label,
    show_legend,
    row,
    legend_group,
    dash="solid",
):
    mid = (high + low) / 2
    fig.add_trace(
        go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[high, high, low, low, high],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=1.2, dash=dash),
            mode="lines",
            name=label,
            showlegend=show_legend,
            legendgroup=legend_group,
            hovertemplate=f"<b>{label}</b> H:{high:.5f} L:{low:.5f} 50%:{mid:.5f}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[mid, mid],
            mode="lines",
            line=dict(color=line_color, width=0.8, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=legend_group,
        ),
        row=row,
        col=1,
    )


def _draw_blocks(
    fig,
    df,
    high_col,
    low_col,
    dir_col,
    act_col,
    bull_fill,
    bull_line,
    bear_fill,
    bear_line,
    bull_label,
    bear_label,
    row,
    dash="solid",
):
    if high_col not in df.columns:
        return
    n = len(df)
    highs = df[high_col].values
    lows = df[low_col].values
    dirs = df[dir_col].values
    acts = df[act_col].values
    added_b = False
    added_be = False
    i = 0
    while i < n:
        if np.isnan(highs[i]) or acts[i] == 0:
            i += 1
            continue
        oh = highs[i]
        ol = lows[i]
        od = int(dirs[i])
        j = i + 1
        while j < n and not np.isnan(highs[j]) and highs[j] == oh and acts[j] == 1:
            j += 1
        if od == 1:
            _draw_box(
                fig,
                i,
                j - 1,
                oh,
                ol,
                bull_fill,
                bull_line,
                bull_label,
                not added_b,
                row,
                bull_label,
                dash,
            )
            added_b = True
        else:
            _draw_box(
                fig,
                i,
                j - 1,
                oh,
                ol,
                bear_fill,
                bear_line,
                bear_label,
                not added_be,
                row,
                bear_label,
                dash,
            )
            added_be = True
        i = j


def add_order_blocks(fig, df, row=1):
    _draw_blocks(
        fig,
        df,
        "_ob_high",
        "_ob_low",
        "feature_ob_direction",
        "feature_ob_active",
        C["ob_bull_fill"],
        C["ob_bull_line"],
        C["ob_bear_fill"],
        C["ob_bear_line"],
        "OB Bull (+)",
        "OB Bear (-)",
        row,
    )


def add_breaker_blocks(fig, df, row=1):
    _draw_blocks(
        fig,
        df,
        "_bb_high",
        "_bb_low",
        "feature_bb_direction",
        "feature_bb_active",
        C["bb_bull_fill"],
        C["bb_bull_line"],
        C["bb_bear_fill"],
        C["bb_bear_line"],
        "BB Bull",
        "BB Bear",
        row,
        dash="dot",
    )


def add_session_bands(fig, df, row=1):
    if not hasattr(df.index, "hour"):
        return
    n = len(df)
    hours = df.index.hour
    added_kz = False
    i = 0
    while i < n:
        h = hours[i]
        in_kz = (7 <= h < 10) or (12 <= h < 15)
        if in_kz:
            j = i + 1
            while j < n and ((7 <= hours[j] < 10) or (12 <= hours[j] < 15)):
                j += 1
            fig.add_vrect(
                x0=i - 0.5,
                x1=j - 0.5,
                fillcolor=C["killzone"],
                line_width=0,
                layer="below",
            )
            if not added_kz:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color="rgba(255,166,87,0.6)", width=8),
                        name="Killzone",
                        showlegend=True,
                    ),
                    row=row,
                    col=1,
                )
                added_kz = True
            i = j
            continue
        if 7 <= h < 16:
            j = i + 1
            while j < n and 7 <= hours[j] < 16:
                j += 1
            fig.add_vrect(
                x0=i - 0.5,
                x1=j - 0.5,
                fillcolor=C["session_lon"],
                line_width=0,
                layer="below",
            )
            i = j
            continue
        if 12 <= h < 21:
            j = i + 1
            while j < n and 12 <= hours[j] < 21:
                j += 1
            fig.add_vrect(
                x0=i - 0.5,
                x1=j - 0.5,
                fillcolor=C["session_ny"],
                line_width=0,
                layer="below",
            )
            i = j
            continue
        i += 1


def add_bias_subplot(fig, df, row=2):
    n = list(range(len(df)))
    for col, label, color, width, dash in [
        ("feature_lt_bias", "LT Bias", C["lth"], 2.5, "solid"),
        ("feature_it_bias", "IT Bias", C["ith"], 2.0, "solid"),
        ("feature_st_bias", "ST Bias", C["sth"], 1.5, "solid"),
        ("feature_bias_aligned", "Bias Aligned", "#3fb950", 1.5, "dot"),
    ]:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=n,
                y=df[col].tolist(),
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash=dash),
                showlegend=True,
                legendgroup=label,
            ),
            row=row,
            col=1,
        )
    for yv, col in [(0, C["text_dim"]), (1, C["bos_bull"]), (-1, C["bos_bear"])]:
        fig.add_hline(
            y=yv, line_color=col, line_width=0.5, line_dash="dot", row=row, col=1
        )


def add_feature_subplot(fig, df, row=3):
    n = list(range(len(df)))
    for col, label, color, dash, width in [
        ("feature_it_range_position", "IT Range Pos", C["ith"], "solid", 2.0),
        ("feature_lt_range_position", "LT Range Pos", C["lth"], "dot", 1.8),
        ("feature_ob_position", "OB Position", C["ob_bull_line"], "dash", 1.5),
        ("feature_fvg_rebalance_pct", "FVG Fill %", C["fvg_bull_line"], "dashdot", 1.5),
        ("feature_ob_correct_side", "OB Correct", "#ffa657", "solid", 1.2),
    ]:
        if col not in df.columns:
            continue
        series = df[col].ffill().fillna(0)
        fig.add_trace(
            go.Scatter(
                x=n,
                y=series.tolist(),
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash=dash),
                showlegend=True,
                legendgroup=label,
            ),
            row=row,
            col=1,
        )
    for yv in [0.0, 0.5, 1.0]:
        fig.add_hline(
            y=yv, line_color=C["grid"], line_width=0.6, line_dash="dot", row=row, col=1
        )


def build_chart(df: pd.DataFrame, section: str = "all") -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=[
            "Price + ICT Features",
            "ST / IT / LT Bias",
            "Range Position & Zone Fill",
        ],
    )

    print("  Candlesticks...")
    add_candlesticks(fig, df, row=1)

    if section in ("all", "session"):
        print("  Session bands...")
        add_session_bands(fig, df, row=1)

    if section in ("all", "zones"):
        print("  FVG zones...")
        add_fvg_zones(fig, df, row=1)
        print("  Order Blocks...")
        add_order_blocks(fig, df, row=1)
        print("  Breaker Blocks...")
        add_breaker_blocks(fig, df, row=1)

    if section in ("all", "structure"):
        print("  Structure levels...")
        add_structure_levels(fig, df, row=1)

    if section in ("all", "bos", "structure"):
        print("  BOS / CHoCH...")
        add_bos_choch(fig, df, row=1)

    print("  Bias subplot...")
    add_bias_subplot(fig, df, row=2)
    print("  Feature values subplot...")
    add_feature_subplot(fig, df, row=3)

    # Lock y-axis to actual price range — this is the critical fix
    low_min = df["low"].min()
    high_max = df["high"].max()
    pad = (high_max - low_min) * 0.05
    # Extra top padding for BOS markers stacked above candles
    fig.update_yaxes(range=[low_min - pad, high_max + pad * 8], row=1, col=1)
    fig.update_yaxes(range=[-1.3, 1.3], row=2, col=1)
    fig.update_yaxes(range=[-0.1, 1.1], row=3, col=1)

    fig.update_layout(
        title=dict(
            text=(
                f"ICT Feature Verification — {len(df)} candles  "
                f"[{str(df.index[0])[:13]} to {str(df.index[-1])[:13]}]"
            ),
            font=dict(color=C["text"], size=14, family="monospace"),
            x=0.01,
        ),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg2"],
        font=dict(color=C["text"], family="monospace", size=11),
        height=980,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(13,17,23,0.90)",
            bordercolor=C["grid"],
            borderwidth=1,
            font=dict(size=10),
            orientation="v",
            x=1.01,
            y=1.0,
            tracegroupgap=4,
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=70, r=200, t=60, b=60),
    )

    for r in [1, 2, 3]:
        fig.update_xaxes(
            gridcolor=C["grid"],
            showgrid=True,
            zeroline=False,
            showspikes=True,
            spikethickness=1,
            spikecolor=C["text_dim"],
            row=r,
            col=1,
        )
        fig.update_yaxes(
            gridcolor=C["grid"],
            showgrid=True,
            zeroline=False,
            showspikes=True,
            spikethickness=1,
            spikecolor=C["text_dim"],
            row=r,
            col=1,
        )

    for ann in fig.layout.annotations:
        ann.font.color = C["text_dim"]
        ann.font.size = 11

    return fig


def print_feature_stats(df: pd.DataFrame):
    feat_cols = sorted([c for c in df.columns if c.startswith("feature_")])
    print(f"\n{'='*70}")
    print(f"  FEATURE COVERAGE  ({len(df):,} candles, {len(feat_cols)} features)")
    print(f"{'='*70}")
    print(f"  {'FEATURE':<40} {'ACTIVE%':>7}  {'NaN':>5}  {'MIN':>8}  {'MAX':>8}")
    print(f"  {'-'*40} {'-'*7}  {'-'*5}  {'-'*8}  {'-'*8}")
    for col in feat_cols:
        s = df[col]
        nz_pct = (s != 0).sum() / max(len(s), 1) * 100
        nan_ct = s.isna().sum()
        mn = s.min() if not s.isna().all() else float("nan")
        mx = s.max() if not s.isna().all() else float("nan")
        flag = ""
        if s.isna().all():
            flag = "  BROKEN: all NaN"
        elif (s == 0).all():
            flag = "  BROKEN: all zero"
        elif nz_pct < 1.0:
            flag = "  WARNING: rarely active"
        print(f"  {col:<40} {nz_pct:>6.1f}%  {nan_ct:>5}  {mn:>8.4f}  {mx:>8.4f}{flag}")
    print(f"{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ICT Feature Verification Chart")
    parser.add_argument("--candles", type=int, default=200)
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument(
        "--section",
        type=str,
        default="all",
        choices=["all", "structure", "zones", "bias", "session", "bos"],
    )
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    print(f"\n=== ICT Feature Verification ===")
    df = load_data(args.start, args.candles)
    print_feature_stats(df)
    if args.stats:
        return
    print("Building chart...")
    fig = build_chart(df, section=args.section)
    if args.save:
        path = args.save if args.save.endswith(".html") else args.save + ".html"
        fig.write_html(path)
        print(f"Saved to: {path}")
    else:
        print("Opening in browser... (scroll=zoom, drag=pan, click legend=toggle)")
        fig.show()


if __name__ == "__main__":
    main()
