"""
visualization/charts.py
========================
Plotly chart functions for model evaluation.

Charts available:
    equity_curve(results)         — Portfolio vs Buy-and-Hold
    drawdown_chart(results)       — Drawdown over time
    price_with_trades(df, episode) — Price candles + trade markers
    structure_chart(df)           — Price with STH/STL/ITH/ITL/LTH/LTL overlaid
    feature_heatmap(df)           — Feature correlation / activity heatmap
    position_timeline(episode)    — Position over time
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


COLORS = {
    "bull"       : "#26a69a",   # teal green
    "bear"       : "#ef5350",   # red
    "neutral"    : "#90a4ae",   # grey
    "portfolio"  : "#42a5f5",   # blue
    "buyhold"    : "#ffca28",   # amber
    "drawdown"   : "#ef5350",   # red
    "bg"         : "#1a1a2e",   # dark navy
    "grid"       : "#2d2d44",
    "text"       : "#e0e0e0",
    "sth"        : "#64b5f6",   # light blue
    "stl"        : "#f48fb1",   # pink
    "ith"        : "#ffa726",   # orange
    "itl"        : "#ab47bc",   # purple
    "lth"        : "#ffffff",   # white
    "ltl"        : "#ffee58",   # yellow
    "ob_bull"    : "rgba(38,166,154,0.3)",
    "ob_bear"    : "rgba(239,83,80,0.3)",
    "fvg_bull"   : "rgba(38,166,154,0.15)",
    "fvg_bear"   : "rgba(239,83,80,0.15)",
    "bb"         : "rgba(171,71,188,0.25)",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor = COLORS["bg"],
    plot_bgcolor  = COLORS["bg"],
    font          = dict(color=COLORS["text"], family="monospace"),
    xaxis         = dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis         = dict(gridcolor=COLORS["grid"], showgrid=True),
    legend        = dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=COLORS["grid"]),
    margin        = dict(l=60, r=40, t=60, b=40),
)


def _apply_layout(fig, title=""):
    fig.update_layout(title=title, **LAYOUT_DEFAULTS)
    return fig


# ══════════════════════════════════════════════════════
# 1. EQUITY CURVE
# ══════════════════════════════════════════════════════

def equity_curve(results: dict, title: str = "Equity Curve") -> go.Figure:
    """
    Portfolio value vs buy-and-hold for train and test periods.

    Parameters
    ----------
    results : dict returned by backtest.simple_backtest()
              expected keys: 'train', 'test', each with 'episode' and 'df'
    """
    fig = go.Figure()

    for split, color_p, color_bh in [
        ("train", "#42a5f5", "#ffca28"),
        ("test",  "#26a69a", "#ffa726"),
    ]:
        if split not in results:
            continue

        ep = results[split]["episode"]
        df = results[split]["df"]
        pv = ep["portfolio_values"]
        initial = pv[0] if pv else 10_000

        x = list(range(len(pv)))

        # Portfolio line
        fig.add_trace(go.Scatter(
            x    = x,
            y    = pv,
            mode = "lines",
            name = f"Portfolio ({split})",
            line = dict(color=color_p, width=2),
        ))

        # Buy-and-hold line
        close = df["close"].values[:len(pv)]
        bh    = initial * close / close[0]
        fig.add_trace(go.Scatter(
            x    = x,
            y    = bh.tolist(),
            mode = "lines",
            name = f"Buy & Hold ({split})",
            line = dict(color=color_bh, width=1.5, dash="dot"),
        ))

    fig.add_hline(
        y=results.get("train", {}).get("episode", {}).get("portfolio_values", [10000])[0],
        line_dash="dash", line_color=COLORS["neutral"], opacity=0.5,
        annotation_text="Initial Capital"
    )

    return _apply_layout(fig, title)


# ══════════════════════════════════════════════════════
# 2. DRAWDOWN CHART
# ══════════════════════════════════════════════════════

def drawdown_chart(results: dict, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure()

    for split, color in [("train", "#42a5f5"), ("test", "#26a69a")]:
        if split not in results:
            continue

        pv   = np.array(results[split]["episode"]["portfolio_values"])
        peak = np.maximum.accumulate(pv)
        dd   = (pv - peak) / np.maximum(peak, 1e-8) * 100

        fig.add_trace(go.Scatter(
            x    = list(range(len(dd))),
            y    = dd.tolist(),
            mode = "lines",
            name = f"Drawdown ({split})",
            fill = "tozeroy",
            line = dict(color=COLORS["drawdown"], width=1),
            fillcolor = "rgba(239,83,80,0.2)",
        ))

    fig.add_hline(y=0, line_color=COLORS["neutral"], opacity=0.5)

    return _apply_layout(fig, title)


# ══════════════════════════════════════════════════════
# 3. PRICE CHART WITH STRUCTURE LEVELS
# ══════════════════════════════════════════════════════

def structure_chart(df: pd.DataFrame, n_candles: int = 200,
                    title: str = "Market Structure") -> go.Figure:
    """
    Candlestick chart with STH/STL/ITH/ITL/LTH/LTL levels overlaid.
    Shows the last n_candles of the DataFrame.
    """
    subset = df.tail(n_candles).copy()
    x      = list(range(len(subset)))

    fig = make_subplots(rows=1, cols=1)

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x     = x,
        open  = subset["open"].tolist(),
        high  = subset["high"].tolist(),
        low   = subset["low"].tolist(),
        close = subset["close"].tolist(),
        name  = "Price",
        increasing_line_color = COLORS["bull"],
        decreasing_line_color = COLORS["bear"],
    ))

    # Structure level markers
    level_config = [
        ("_sth_flag", "_sth_price", "STH", COLORS["sth"],  "triangle-up",   8),
        ("_stl_flag", "_stl_price", "STL", COLORS["stl"],  "triangle-down", 8),
        ("_ith_flag", "_ith_price", "ITH", COLORS["ith"],  "triangle-up",   12),
        ("_itl_flag", "_itl_price", "ITL", COLORS["itl"],  "triangle-down", 12),
        ("_lth_flag", "_lth_price", "LTH", COLORS["lth"],  "star",          14),
        ("_ltl_flag", "_ltl_price", "LTL", COLORS["ltl"],  "star",          14),
    ]

    for flag_col, price_col, label, color, symbol, size in level_config:
        if flag_col not in subset.columns or price_col not in subset.columns:
            continue
        mask = subset[flag_col].fillna(False).astype(bool)
        if mask.any():
            idx    = [i for i, m in enumerate(mask) if m]
            prices = subset[price_col][mask].tolist()
            fig.add_trace(go.Scatter(
                x      = idx,
                y      = prices,
                mode   = "markers+text",
                name   = label,
                text   = [label] * len(idx),
                textposition = "top center" if "H" in label else "bottom center",
                marker = dict(symbol=symbol, size=size, color=color),
            ))

    # FVG zones
    if "_fvg_top" in subset.columns:
        in_fvg = False
        for i, (idx_val, row) in enumerate(subset.iterrows()):
            ft = row.get("_fvg_top", np.nan)
            fb = row.get("_fvg_bottom", np.nan)
            fd = row.get("_fvg_direction", 0)
            if not np.isnan(ft) and not np.isnan(fb):
                fill_color = COLORS["fvg_bull"] if fd == 1 else COLORS["fvg_bear"]
                fig.add_hrect(
                    y0=fb, y1=ft,
                    fillcolor=fill_color,
                    line_width=0,
                    annotation_text="FVG" if i % 10 == 0 else "",
                    annotation_position="top right",
                )

    # OB zones
    if "_ob_high" in subset.columns:
        for i, row in subset.iterrows():
            oh = row.get("_ob_high", np.nan)
            ol = row.get("_ob_low", np.nan)
            od = row.get("feature_ob_direction", 0)
            if not (np.isnan(oh) or np.isnan(ol)):
                fill_color = COLORS["ob_bull"] if od == 1 else COLORS["ob_bear"]
                fig.add_hrect(y0=ol, y1=oh, fillcolor=fill_color, line_width=0)

    fig.update_xaxes(rangeslider_visible=False)
    return _apply_layout(fig, title)


# ══════════════════════════════════════════════════════
# 4. PRICE WITH TRADE ENTRY/EXIT MARKERS
# ══════════════════════════════════════════════════════

def price_with_trades(df: pd.DataFrame, episode: dict,
                      title: str = "Trades") -> go.Figure:
    """
    Candlestick chart with trade entry/exit markers.
    """
    n   = len(episode["portfolio_values"])
    sub = df.tail(n).copy()
    x   = list(range(len(sub)))

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x     = x,
        open  = sub["open"].tolist(),
        high  = sub["high"].tolist(),
        low   = sub["low"].tolist(),
        close = sub["close"].tolist(),
        name  = "Price",
        increasing_line_color = COLORS["bull"],
        decreasing_line_color = COLORS["bear"],
    ))

    # Position background shading
    positions = episode.get("positions", [])
    for i, pos in enumerate(positions):
        if pos == 1:
            fig.add_vrect(x0=i, x1=i+1, fillcolor="rgba(38,166,154,0.1)", line_width=0)
        elif pos == -1:
            fig.add_vrect(x0=i, x1=i+1, fillcolor="rgba(239,83,80,0.1)", line_width=0)

    # Trade markers
    for trade in episode.get("trades", []):
        if "entry" in trade and "exit" in trade:
            color = COLORS["bull"] if trade.get("direction", 1) == 1 else COLORS["bear"]
            pnl_str = f"PnL: {trade['pnl']*100:.2f}%"
            # Entry
            fig.add_annotation(
                x=0, y=trade["entry"],
                text="▶",
                showarrow=False,
                font=dict(color=color, size=10),
            )

    fig.update_xaxes(rangeslider_visible=False)
    return _apply_layout(fig, title)


# ══════════════════════════════════════════════════════
# 5. FEATURE ACTIVITY HEATMAP
# ══════════════════════════════════════════════════════

def feature_heatmap(df: pd.DataFrame, n_candles: int = 500,
                    title: str = "Feature Activity") -> go.Figure:
    """
    Shows recent feature values as a heatmap.
    Binary features (0/1) show as on/off.
    Continuous features show their value.
    """
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feature_cols:
        return go.Figure()

    subset = df.tail(n_candles)[feature_cols].copy()

    # Normalise each feature to 0-1 for display
    for col in feature_cols:
        vmin = subset[col].min()
        vmax = subset[col].max()
        if vmax > vmin:
            subset[col] = (subset[col] - vmin) / (vmax - vmin)

    fig = go.Figure(data=go.Heatmap(
        z         = subset.values.T.tolist(),
        y         = feature_cols,
        colorscale= "RdYlGn",
        showscale = True,
        xgap      = 0.5,
        ygap      = 0.5,
    ))

    fig.update_layout(
        title  = title,
        height = max(400, len(feature_cols) * 18),
        **LAYOUT_DEFAULTS,
    )
    return fig


# ══════════════════════════════════════════════════════
# 6. METRICS SUMMARY BAR CHART
# ══════════════════════════════════════════════════════

def metrics_bar(train_metrics: dict, test_metrics: dict,
                title: str = "Train vs Test Performance") -> go.Figure:
    """
    Side-by-side bar chart comparing train and test metrics.
    """
    keys = ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "calmar_ratio"]
    labels = ["Total Return %", "Sharpe Ratio", "Max Drawdown %", "Calmar Ratio"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name   = "Train",
        x      = labels,
        y      = [train_metrics.get(k, 0) for k in keys],
        marker_color = COLORS["portfolio"],
    ))
    fig.add_trace(go.Bar(
        name   = "Test",
        x      = labels,
        y      = [test_metrics.get(k, 0) for k in keys],
        marker_color = COLORS["bull"],
    ))

    fig.update_layout(barmode="group")
    return _apply_layout(fig, title)
