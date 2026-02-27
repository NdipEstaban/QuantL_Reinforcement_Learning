"""
visualization/dashboard.py
===========================
Interactive Dash dashboard for model evaluation.

Run:
    python -m visualization.dashboard

Then open: http://127.0.0.1:8050

Tabs:
    1. Overview   — equity curve, drawdown, metrics summary
    2. Structure  — price chart with ICT structure levels
    3. Trades     — price with trade entry/exit markers
    4. Features   — feature activity heatmap over time
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go

from visualization.charts import (
    equity_curve, drawdown_chart, structure_chart,
    price_with_trades, feature_heatmap, metrics_bar, COLORS, LAYOUT_DEFAULTS
)
from evaluation.metrics import compute_metrics


# ══════════════════════════════════════════════════════
# LOAD DATA + RESULTS
# ══════════════════════════════════════════════════════

def load_dashboard_data():
    """Load processed features and model results for the dashboard."""
    from features.pipeline import load_processed

    try:
        df = load_processed()
    except FileNotFoundError:
        print("No processed data found. Run: python -m features.pipeline")
        return None, None

    # Try to load backtest results
    results = None
    model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model.zip")
    if os.path.exists(model_path):
        try:
            from evaluation.backtest import simple_backtest
            results = simple_backtest(model_path)
        except Exception as e:
            print(f"Could not load backtest results: {e}")

    return df, results


# ══════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════

def build_layout(df, results):
    """Build the Dash app layout."""

    has_results = results is not None

    tab_style        = {"backgroundColor": "#1a1a2e", "color": "#90a4ae", "padding": "8px 16px"}
    tab_selected     = {"backgroundColor": "#2d2d44", "color": "#e0e0e0", "padding": "8px 16px"}

    # Metrics cards
    def metric_card(label, value, color="#42a5f5"):
        return html.Div([
            html.Div(label, style={"color": "#90a4ae", "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(str(value), style={"color": color, "fontSize": "22px", "fontWeight": "bold"}),
        ], style={
            "backgroundColor": "#2d2d44",
            "padding": "16px",
            "borderRadius": "8px",
            "textAlign": "center",
            "minWidth": "140px",
        })

    # Build metric cards from results
    cards = []
    if has_results and "test" in results:
        m = results["test"]["metrics"]
        ret_color = COLORS["bull"] if m["total_return_pct"] > 0 else COLORS["bear"]
        cards = [
            metric_card("Test Return", f"{m['total_return_pct']}%", ret_color),
            metric_card("Sharpe Ratio", m["sharpe_ratio"]),
            metric_card("Max Drawdown", f"{m['max_drawdown_pct']}%", COLORS["bear"]),
            metric_card("Calmar Ratio", m["calmar_ratio"]),
        ]
        if m["win_rate_pct"] is not None:
            cards.append(metric_card("Win Rate", f"{m['win_rate_pct']}%"))
        if m["profit_factor"] is not None:
            cards.append(metric_card("Profit Factor", m["profit_factor"]))
    else:
        cards = [html.Div(
            "Train model first: python -m training.train",
            style={"color": COLORS["neutral"], "padding": "20px"}
        )]

    n_features = len([c for c in df.columns if c.startswith("feature_")]) if df is not None else 0
    n_candles  = len(df) if df is not None else 0

    layout = html.Div([

        # Header
        html.Div([
            html.H2("🤖 Forex RL Bot — ICT/SMC Dashboard",
                    style={"margin": "0", "color": COLORS["text"]}),
            html.Div(
                f"Data: {n_candles:,} candles | Features: {n_features} | "
                f"Pair: EURUSD | TF: {config.BASE_TF}",
                style={"color": COLORS["neutral"], "fontSize": "12px", "marginTop": "4px"}
            ),
        ], style={
            "backgroundColor": "#0d0d1a",
            "padding": "16px 24px",
            "borderBottom": f"1px solid {COLORS['grid']}",
        }),

        # Metric cards row
        html.Div(cards, style={
            "display": "flex",
            "gap": "12px",
            "padding": "16px 24px",
            "flexWrap": "wrap",
            "backgroundColor": "#12122a",
        }),

        # Tabs
        dcc.Tabs(
            id="tabs",
            value="tab-overview",
            style={"backgroundColor": "#1a1a2e"},
            children=[
                dcc.Tab(label="📈 Equity Curve",   value="tab-overview",
                        style=tab_style, selected_style=tab_selected),
                dcc.Tab(label="🏗️ Structure",       value="tab-structure",
                        style=tab_style, selected_style=tab_selected),
                dcc.Tab(label="🎯 Trades",          value="tab-trades",
                        style=tab_style, selected_style=tab_selected),
                dcc.Tab(label="🔥 Features",        value="tab-features",
                        style=tab_style, selected_style=tab_selected),
            ],
        ),

        html.Div(id="tab-content", style={
            "backgroundColor": COLORS["bg"],
            "minHeight": "600px",
            "padding": "16px",
        }),

    ], style={
        "backgroundColor": COLORS["bg"],
        "fontFamily": "monospace",
        "minHeight": "100vh",
    })

    return layout


# ══════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════

def register_callbacks(app, df, results):

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
    )
    def render_tab(tab):

        if tab == "tab-overview":
            figs = []
            if results:
                ec  = equity_curve(results, "Portfolio vs Buy & Hold")
                dd  = drawdown_chart(results, "Drawdown")
                mb  = metrics_bar(
                    results.get("train", {}).get("metrics", {}),
                    results.get("test",  {}).get("metrics", {}),
                )
                figs = [
                    dcc.Graph(figure=ec,  style={"height": "350px"}),
                    dcc.Graph(figure=dd,  style={"height": "250px"}),
                    dcc.Graph(figure=mb,  style={"height": "350px"}),
                ]
            else:
                figs = [html.Div(
                    "No backtest results. Run: python -m evaluation.backtest",
                    style={"color": COLORS["neutral"], "padding": "40px", "textAlign": "center"}
                )]
            return html.Div(figs)

        elif tab == "tab-structure":
            if df is not None:
                fig = structure_chart(df, n_candles=300, title="ICT Market Structure")
                return dcc.Graph(figure=fig, style={"height": "700px"})
            return html.Div("No data loaded.")

        elif tab == "tab-trades":
            if results and "test" in results:
                ep  = results["test"]["episode"]
                dft = results["test"]["df"]
                fig = price_with_trades(dft, ep, title="Test Period — Trades")
                return dcc.Graph(figure=fig, style={"height": "700px"})
            return html.Div(
                "No trade data. Run backtest first.",
                style={"color": COLORS["neutral"], "padding": "40px", "textAlign": "center"}
            )

        elif tab == "tab-features":
            if df is not None:
                fig = feature_heatmap(df, n_candles=400, title="Feature Activity (last 400 candles)")
                return dcc.Graph(figure=fig, style={"height": "800px"})
            return html.Div("No data loaded.")

        return html.Div("Select a tab.")


# ══════════════════════════════════════════════════════
# APP FACTORY
# ══════════════════════════════════════════════════════

def create_app() -> Dash:
    df, results = load_dashboard_data()

    app = Dash(
        __name__,
        title="Forex RL Bot Dashboard",
        suppress_callback_exceptions=True,
    )

    app.layout = build_layout(df, results)
    register_callbacks(app, df, results)

    return app


if __name__ == "__main__":
    print("Starting dashboard at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop\n")
    app = create_app()
    app.run(debug=True, port=8050)
