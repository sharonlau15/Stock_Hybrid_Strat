"""
dashboard/app.py
================
Dash dashboard for Stock Hybrid Strategy.

Tabs
----
  Portfolio Types   — compare all portfolio constructors (heatmap, cumret, metrics)
  Strategies        — compare all 6 strategies (cumret, rolling Sharpe, drawdown)
  Trading Progress  — live NAV, positions, trade log
  Backtesting       — custom run: pick strategies + portfolio, see results

Run
---
  python dashboard/app.py
  # open http://localhost:8050
"""

import sys
import json
import threading
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from dashboard.data_loader import (
    load_strategy_metrics, load_strategy_returns,
    load_comparison_metrics, load_comparison_returns,
    load_custom_metrics, load_custom_returns,
    load_split_metrics, load_split_returns, load_split_summary,
    load_constant_weight_metrics, load_constant_weight_returns,
    load_constant_strategy_metrics, load_constant_strategy_returns,
    load_live_state, metrics_to_df, comparison_pivot,
)

# ── App ────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Stock Hybrid Strategy",
)
server = app.server

CHART_TEMPLATE = "plotly_dark"
CHART_BG       = "rgba(0,0,0,0)"
COLORS         = px.colors.qualitative.Plotly  # 10 distinct hex colors

PORTFOLIO_NAMES = [
    "max_sharpe", "equal_weight", "min_variance", "risk_parity", "signal_weighted",
]
STRATEGY_NAMES = [
    "momentum", "mean_reversion", "risk_parity",
    "cross_sectional_momentum", "vol_breakout", "ml_signal",
    "exhaustion_fade", "sma_brownian", "seasonal_exhaustion_fade",
    "seasonal_exhaustion_fade_long", "macro_regime",
]

# ── Background job state ───────────────────────────────────────────────────────

_jobs: dict = {}
_job_lock   = threading.Lock()


def _job_set(job_id: str, status: str, error: str = "") -> None:
    with _job_lock:
        _jobs[job_id] = {"status": status, "error": error}


def _job_get(job_id: str) -> dict:
    with _job_lock:
        return _jobs.get(job_id, {})


# ── Chart / table helpers ──────────────────────────────────────────────────────

_TABLE_STYLE = dict(
    style_table      = {"overflowX": "auto"},
    style_header     = {"backgroundColor": "#343a40", "fontWeight": "bold", "color": "#f8f9fa"},
    style_data       = {"backgroundColor": "#212529", "color": "#f8f9fa"},
    style_data_conditional = [{"if": {"row_index": "odd"}, "backgroundColor": "#2c3136"}],
    style_cell       = {"textAlign": "center", "padding": "8px", "minWidth": "90px",
                        "fontSize": "13px"},
)


def _empty_fig(msg: str = "No data yet — run a backtest first") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font={"size": 14, "color": "#666"})
    fig.update_layout(template=CHART_TEMPLATE, paper_bgcolor=CHART_BG,
                      plot_bgcolor=CHART_BG, height=350)
    return fig


def _status_badge(job_id: str) -> dbc.Badge:
    job = _job_get(job_id)
    status = job.get("status", "idle")
    if status == "running":
        return dbc.Badge("Running…", color="warning",  pill=True, className="ms-2")
    if status == "done":
        return dbc.Badge("Done",     color="success",  pill=True, className="ms-2")
    if status == "error":
        err = job.get("error", "")[:50]
        return dbc.Badge(f"Error: {err}", color="danger", pill=True, className="ms-2")
    return dbc.Badge("Ready", color="secondary", pill=True, className="ms-2")


def _cumret_fig(returns: pd.DataFrame, title: str = "Cumulative Returns") -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        cum = (1 + returns[col].dropna()).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=(cum * 100).values, mode="lines",
            name=col, line=dict(color=COLORS[i % len(COLORS)], width=1.8),
        ))
    fig.update_layout(
        template=CHART_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        title=title, yaxis_title="Return (%)", xaxis_title="Date",
        hovermode="x unified", height=380,
        legend=dict(orientation="h", y=-0.28, font_size=11),
    )
    return fig


def _rolling_sharpe_fig(returns: pd.DataFrame, window: int = 63) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        r  = returns[col].dropna()
        rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values, mode="lines", name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.6)
    fig.update_layout(
        template=CHART_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        title=f"Rolling Sharpe ({window}D)", yaxis_title="Sharpe", height=320,
        hovermode="x unified", legend=dict(orientation="h", y=-0.35, font_size=11),
    )
    return fig


def _drawdown_fig(returns: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        cum  = (1 + returns[col].dropna()).cumprod()
        dd   = (cum - cum.cummax()) / cum.cummax() * 100
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name=col,
            line=dict(color=color, width=1.2),
            fill="tozeroy", fillcolor=color.replace(")", ",0.12)").replace("rgb(", "rgba("),
        ))
    fig.update_layout(
        template=CHART_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        title=title, yaxis_title="Drawdown (%)", height=320,
        hovermode="x unified", legend=dict(orientation="h", y=-0.35, font_size=11),
    )
    return fig


def _metrics_table(df: pd.DataFrame) -> tuple[list, list]:
    """Return (columns, data) for a DataTable, formatted for display."""
    if df.empty:
        return [], []
    display = ["label", "sharpe", "sortino", "calmar", "cagr", "max_drawdown",
               "annual_vol", "win_rate", "total_return", "final_capital"]
    cols    = [c for c in display if c in df.columns]
    sub     = df[cols].copy()

    pct_cols = {"cagr", "max_drawdown", "annual_vol", "win_rate", "total_return"}
    for c in pct_cols.intersection(sub.columns):
        sub[c] = sub[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "–")
    for c in {"sharpe", "sortino", "calmar"}.intersection(sub.columns):
        sub[c] = sub[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    if "final_capital" in sub.columns:
        sub["final_capital"] = sub["final_capital"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "–"
        )

    rename = {
        "label": "Name", "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar",
        "cagr": "CAGR", "max_drawdown": "Max DD", "annual_vol": "Ann Vol",
        "win_rate": "Win Rate", "total_return": "Total Ret", "final_capital": "Final $",
    }
    columns = [{"name": rename.get(c, c), "id": c} for c in cols]
    return columns, sub.to_dict("records")


def _split_metrics_table(metrics: dict) -> tuple[list, list]:
    """Build period-by-period DataTable from split_metrics dict (strategy/period keys)."""
    import math
    strats = sorted({
        key.rsplit("/", 1)[0] for key in metrics
        if "/" in key and key.rsplit("/", 1)[1] in ("train", "val", "test")
    })
    if not strats:
        return [], []

    rows = []
    for strat in strats:
        def _fmt(period: str, key: str, pct: bool = False) -> str:
            v = metrics.get(f"{strat}/{period}", {}).get(key, float("nan"))
            if not isinstance(v, (int, float)) or math.isnan(float(v)):
                return "–"
            return f"{float(v)*100:.1f}%" if pct else f"{float(v):.3f}"

        tr_s = float(metrics.get(f"{strat}/train", {}).get("sharpe", float("nan")))
        va_s = float(metrics.get(f"{strat}/val",   {}).get("sharpe", float("nan")))
        te_s = float(metrics.get(f"{strat}/test",  {}).get("sharpe", float("nan")))
        consistent = (
            not math.isnan(va_s) and not math.isnan(te_s)
            and va_s > 0 and te_s > 0
            and abs(tr_s - va_s) < 0.5 and abs(tr_s - te_s) < 0.5
        )
        rows.append({
            "Strategy":     strat,
            "Train Sharpe": _fmt("train", "sharpe"),
            "Val Sharpe":   _fmt("val",   "sharpe"),
            "Test Sharpe":  _fmt("test",  "sharpe"),
            "Train CAGR":   _fmt("train", "cagr",         pct=True),
            "Val CAGR":     _fmt("val",   "cagr",         pct=True),
            "Test CAGR":    _fmt("test",  "cagr",         pct=True),
            "Train MaxDD":  _fmt("train", "max_drawdown", pct=True),
            "Val MaxDD":    _fmt("val",   "max_drawdown", pct=True),
            "Test MaxDD":   _fmt("test",  "max_drawdown", pct=True),
            "Consistent":   "✓" if consistent else "✗",
        })

    cols = [{"name": k, "id": k} for k in rows[0].keys()]
    return cols, rows


def _split_cumret_fig(returns: pd.DataFrame, strat_filter: str = None) -> go.Figure:
    """Cumulative returns with shaded Train / Val / Test regions."""
    fig = go.Figure()
    if returns.empty:
        return _empty_fig("No split results — enable split mode and run backtest")

    strat_names = sorted({
        col.rsplit("/", 1)[0] for col in returns.columns
        if "/" in col and col.rsplit("/", 1)[1] in ("train", "val", "test")
    })
    show_strats = [strat_filter] if strat_filter and strat_filter in strat_names else strat_names
    color_map   = {s: COLORS[i % len(COLORS)] for i, s in enumerate(strat_names)}

    # Determine period boundary dates from any available strategy
    train_end = val_end = series_start = series_end = None
    for strat in strat_names:
        for period in ("train", "val", "test"):
            col = f"{strat}/{period}"
            if col not in returns.columns:
                continue
            r = returns[col].dropna()
            if r.empty:
                continue
            if series_start is None or r.index[0] < series_start:
                series_start = r.index[0]
            if series_end is None or r.index[-1] > series_end:
                series_end = r.index[-1]
            if period == "train" and train_end is None:
                train_end = r.index[-1]
            if period == "val" and val_end is None:
                val_end = r.index[-1]
        if train_end and val_end:
            break

    for strat in show_strats:
        parts = [
            returns[f"{strat}/{p}"].dropna()
            for p in ("train", "val", "test")
            if f"{strat}/{p}" in returns.columns and not returns[f"{strat}/{p}"].dropna().empty
        ]
        if not parts:
            continue
        full_r = pd.concat(parts).sort_index()
        cum    = (1 + full_r).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=(cum * 100).values, mode="lines", name=strat,
            line=dict(color=color_map[strat], width=1.8),
        ))

    if series_start and train_end:
        fig.add_vrect(x0=series_start, x1=train_end,
                      fillcolor="rgba(30,100,255,0.07)", line_width=0,
                      annotation_text="Train", annotation_position="top left",
                      annotation_font=dict(size=10, color="#6699ff"))
    if train_end and val_end:
        fig.add_vrect(x0=train_end, x1=val_end,
                      fillcolor="rgba(255,165,0,0.07)", line_width=0,
                      annotation_text="Val", annotation_position="top left",
                      annotation_font=dict(size=10, color="orange"))
    if val_end and series_end:
        fig.add_vrect(x0=val_end, x1=series_end,
                      fillcolor="rgba(0,200,80,0.07)", line_width=0,
                      annotation_text="Test", annotation_position="top left",
                      annotation_font=dict(size=10, color="#00cc66"))

    fig.update_layout(
        template=CHART_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        title="Cumulative Returns — Train / Val / Test",
        yaxis_title="Return (%)", xaxis_title="Date",
        hovermode="x unified", height=400,
        legend=dict(orientation="h", y=-0.28, font_size=11),
    )
    return fig


# ── Tab layouts ────────────────────────────────────────────────────────────────

def _card(children, **kwargs) -> dbc.Card:
    return dbc.Card(dbc.CardBody(children), className="mb-3", **kwargs)


def _portfolio_tab() -> html.Div:
    return html.Div([
        dcc.Interval(id="port-interval", interval=3000, n_intervals=0),
        dbc.Row([
            dbc.Col(dbc.Button("Run Portfolio Comparison", id="run-port-btn",
                               color="primary", size="sm"), width="auto"),
            dbc.Col(html.Div(id="port-status"), width="auto", className="align-self-center"),
            dbc.Col(html.Small(id="port-updated", className="text-muted"),
                    className="align-self-center"),
        ], className="mb-3 g-2 align-items-center"),

        dbc.Row([
            dbc.Col(_card(dcc.Graph(id="port-heatmap")), md=6),
            dbc.Col(_card([
                dbc.Row([
                    dbc.Col(html.Label("Filter by strategy:", className="text-muted small"), width="auto"),
                    dbc.Col(dcc.Dropdown(id="port-strat-filter", options=[], value=None,
                                        placeholder="All strategies",
                                        style={"color": "#000", "fontSize": "13px"})),
                ], className="mb-2 align-items-center"),
                dcc.Graph(id="port-cumret"),
            ]), md=6),
        ]),

        dbc.Row([
            dbc.Col(_card(dcc.Graph(id="port-bar")), md=5),
            dbc.Col(_card(
                dash_table.DataTable(id="port-table", sort_action="native",
                                     page_size=12, **_TABLE_STYLE)
            ), md=7),
        ]),
    ], className="p-3")


def _strategies_tab() -> html.Div:
    return html.Div([
        dcc.Interval(id="strat-interval", interval=3000, n_intervals=0),
        dbc.Row([
            dbc.Col(dbc.Button("Run Strategy Backtest", id="run-strat-btn",
                               color="primary", size="sm"), width="auto"),
            dbc.Col(
                dcc.Dropdown(id="strat-port-select",
                             options=[{"label": p, "value": p} for p in PORTFOLIO_NAMES],
                             value="max_sharpe", clearable=False,
                             style={"color": "#000", "minWidth": "165px", "fontSize": "13px"}),
                width="auto",
            ),
            dbc.Col(html.Div(id="strat-status"), width="auto", className="align-self-center"),
            dbc.Col(html.Small(id="strat-updated", className="text-muted"),
                    className="align-self-center"),
        ], className="mb-3 g-2 align-items-center"),

        _card(dash_table.DataTable(id="strat-table", sort_action="native",
                                   page_size=10, **_TABLE_STYLE)),

        dbc.Row([
            dbc.Col(_card(dcc.Graph(id="strat-cumret")), md=6),
            dbc.Col(_card(dcc.Graph(id="strat-rolling")), md=6),
        ]),
        _card(dcc.Graph(id="strat-drawdown")),
    ], className="p-3")


_POS_TABLE_STYLE = dict(
    style_table      = {"overflowX": "auto"},
    style_header     = {"backgroundColor": "#343a40", "fontWeight": "bold", "color": "#f8f9fa"},
    style_data       = {"backgroundColor": "#212529", "color": "#f8f9fa"},
    style_data_conditional = [
        {"if": {"row_index": "odd"},          "backgroundColor": "#2c3136"},
        {"if": {"filter_query": "{P&L $} > 0", "column_id": "P&L $"},  "color": "#28a745"},
        {"if": {"filter_query": "{P&L $} < 0", "column_id": "P&L $"},  "color": "#dc3545"},
        {"if": {"filter_query": "{P&L %} > 0", "column_id": "P&L %"},  "color": "#28a745"},
        {"if": {"filter_query": "{P&L %} < 0", "column_id": "P&L %"},  "color": "#dc3545"},
    ],
    style_cell = {"textAlign": "center", "padding": "6px 10px", "minWidth": "80px",
                  "fontSize": "13px"},
)

_TRADES_TABLE_STYLE = dict(
    style_table  = {"overflowX": "auto"},
    style_header = {"backgroundColor": "#343a40", "fontWeight": "bold", "color": "#f8f9fa"},
    style_data   = {"backgroundColor": "#212529", "color": "#f8f9fa"},
    style_data_conditional = [
        {"if": {"row_index": "odd"},                                   "backgroundColor": "#2c3136"},
        {"if": {"filter_query": '{side} = "buy"',  "column_id": "side"}, "color": "#28a745"},
        {"if": {"filter_query": '{side} = "sell"', "column_id": "side"}, "color": "#dc3545"},
        {"if": {"filter_query": '{side} = "BUY"',  "column_id": "side"}, "color": "#28a745"},
        {"if": {"filter_query": '{side} = "SELL"', "column_id": "side"}, "color": "#dc3545"},
    ],
    style_cell = {"textAlign": "center", "padding": "6px 10px", "minWidth": "80px",
                  "fontSize": "13px"},
)


def _trading_tab() -> html.Div:
    return html.Div([
        dcc.Interval(id="trading-interval", interval=30_000, n_intervals=0),

        # ── Connection / mode status row ──────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div(id="alpaca-connection-badge"), width="auto",
                    className="align-self-center"),
            dbc.Col(html.Div(id="trading-mode-badge"), width="auto",
                    className="align-self-center"),
            dbc.Col(html.Small(id="alpaca-account-info", className="text-muted"),
                    className="align-self-center"),
            dbc.Col(dbc.Button("Refresh", id="refresh-trading-btn",
                               color="secondary", size="sm"), width="auto",
                    className="ms-auto"),
            dbc.Col(html.Small(id="trading-updated", className="text-muted"),
                    className="align-self-center"),
        ], className="mb-3 g-2 align-items-center"),

        # ── 5 KPI cards ───────────────────────────────────────────────────────
        dbc.Row(id="trading-kpis", className="mb-3"),

        # ── Active strategy + current weights panel ───────────────────────────
        _card([
            html.H6("Active Strategy & Portfolio Weights", className="text-muted small mb-2"),
            html.Div(id="trading-strategy-panel"),
        ]),

        # ── NAV chart ─────────────────────────────────────────────────────────
        _card(dcc.Graph(id="trading-nav")),

        # ── Positions (with live P&L) + Recent Trades ─────────────────────────
        dbc.Row([
            dbc.Col(_card([
                html.H6("Open Positions", className="text-muted small mb-2"),
                dash_table.DataTable(id="positions-table", page_size=10, **_POS_TABLE_STYLE),
            ]), md=8),
            dbc.Col(_card([
                html.H6("Recent Trades", className="text-muted small mb-2"),
                dash_table.DataTable(id="trades-table", page_size=10, **_TRADES_TABLE_STYLE),
            ]), md=4),
        ]),
    ], className="p-3")


def _risk_tab() -> html.Div:
    return html.Div([
        dcc.Interval(id="risk-interval", interval=10_000, n_intervals=0),

        dbc.Row([
            # ── Kill switch ────────────────────────────────────────────────────
            dbc.Col(_card([
                html.H5("Kill Switch", className="mb-3"),
                dbc.Row([
                    dbc.Col(html.Span("Status: ", className="text-muted"), width="auto",
                            className="align-self-center"),
                    dbc.Col(html.Div(id="kill-switch-badge"), width="auto",
                            className="align-self-center"),
                ], className="mb-3 align-items-center"),
                html.Label("Mode", className="small text-muted"),
                dbc.RadioItems(
                    id="kill-switch-mode",
                    options=[
                        {"label": "Halt New Orders", "value": "halt"},
                        {"label": "Close All Positions + Halt", "value": "close_all"},
                    ],
                    value="halt",
                    className="mb-3",
                ),
                html.Label("Note (optional)", className="small text-muted"),
                dbc.Input(id="kill-switch-note", type="text",
                          placeholder="Reason for halting…",
                          className="mb-3",
                          style={"backgroundColor": "#2c3136", "color": "#f8f9fa",
                                 "border": "1px solid #495057"}),
                dbc.Row([
                    dbc.Col(dbc.Button("ACTIVATE", id="kill-activate-btn",
                                       color="danger", className="me-2"), width="auto"),
                    dbc.Col(dbc.Button("DEACTIVATE", id="kill-deactivate-btn",
                                       color="success"), width="auto"),
                ], className="mb-3"),
                html.Div(id="kill-switch-msg", className="text-info small"),
            ]), md=6),

            # ── Connection detail ──────────────────────────────────────────────
            dbc.Col(_card([
                html.H5("Connection Status", className="mb-3"),
                html.Div(id="risk-connection-detail"),
            ]), md=6),
        ]),
    ], className="p-3")


def _backtesting_tab() -> html.Div:
    return html.Div([
        dcc.Interval(id="bt-interval", interval=3000, n_intervals=0),
        dbc.Row([
            # ── Controls ──────────────────────────────────────────────────────
            dbc.Col(_card([
                html.H6("Custom Backtest", className="text-muted small mb-3"),

                html.Label("Strategies", className="small"),
                dcc.Dropdown(
                    id="bt-strat-select",
                    options=[{"label": s, "value": s} for s in STRATEGY_NAMES],
                    value=STRATEGY_NAMES, multi=True,
                    style={"color": "#000", "fontSize": "12px"},
                    className="mb-3",
                ),

                html.Label("Portfolio", className="small"),
                dcc.Dropdown(
                    id="bt-port-select",
                    options=[{"label": p, "value": p} for p in PORTFOLIO_NAMES],
                    value="max_sharpe", clearable=False,
                    style={"color": "#000", "fontSize": "12px"},
                    className="mb-3",
                ),

                html.Label("Backtest Mode", className="small"),
                dcc.Dropdown(
                    id="bt-mode",
                    options=[
                        {"label": "Walk-Forward (dynamic)",   "value": "walk_forward"},
                        {"label": "Constant Weights (frozen)", "value": "constant_weight"},
                        {"label": "Constant Strategy (frozen signal)", "value": "constant_strategy"},
                    ],
                    value="walk_forward", clearable=False,
                    style={"color": "#000", "fontSize": "12px"},
                    className="mb-3",
                ),

                dbc.Row([
                    dbc.Col(html.Label("Train / Val / Test split", className="small"),
                            width="auto", className="align-self-center"),
                    dbc.Col(dbc.Switch(id="bt-run-split", value=True), width="auto"),
                ], className="mb-2 align-items-center"),

                dbc.Row([
                    dbc.Col(html.Label("Train %", className="small"), width="auto",
                            className="align-self-center"),
                    dbc.Col(dbc.Input(id="bt-train-frac", type="number",
                                      value=70, min=50, max=85, step=5,
                                      size="sm"), width=4),
                    dbc.Col(html.Label("Val %", className="small"), width="auto",
                            className="align-self-center"),
                    dbc.Col(dbc.Input(id="bt-val-frac", type="number",
                                      value=15, min=5, max=25, step=5,
                                      size="sm"), width=4),
                ], className="mb-3 align-items-center g-1"),

                dbc.Row([
                    dbc.Col(html.Label("Use cache", className="small"), width="auto",
                            className="align-self-center"),
                    dbc.Col(dbc.Switch(id="bt-use-cache", value=True), width="auto"),
                ], className="mb-3 align-items-center"),

                dbc.Button("Run Backtest", id="run-bt-btn", color="success",
                           size="sm", className="w-100 mb-2"),
                html.Div(id="bt-status"),
            ]), md=3),

            # ── Results ───────────────────────────────────────────────────────
            dbc.Col([
                # Period selector (only shown when split results exist)
                dbc.Row([
                    dbc.Col(html.Label("View strategy:", className="small text-muted"),
                            width="auto", className="align-self-center"),
                    dbc.Col(dcc.Dropdown(id="bt-strat-view", options=[], value=None,
                                         placeholder="All strategies",
                                         style={"color": "#000", "fontSize": "12px"})),
                ], className="mb-2 align-items-center"),

                # Split metrics table
                _card(dash_table.DataTable(id="bt-split-table", sort_action="native",
                                           page_size=12, **_TABLE_STYLE)),

                # Cumulative return chart with Train/Val/Test shading
                _card(dcc.Graph(id="bt-cumret")),

                dbc.Row([
                    dbc.Col(_card(dcc.Graph(id="bt-drawdown")), md=6),
                    dbc.Col(_card(dcc.Graph(id="bt-rolling-sharpe")), md=6),
                ]),
            ], md=9),
        ]),
    ], className="p-3")


# ── Full layout ────────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        html.H4("Stock Hybrid Strategy", className="mb-0 fw-bold"),
        html.Small("Multi-Strategy Backtest & Portfolio Management Dashboard",
                   className="text-muted"),
    ]), className="py-3 border-bottom mb-1")),

    dbc.Tabs([
        dbc.Tab(_portfolio_tab(),  label="Portfolio Types",  tab_id="tab-portfolio",
                label_style={"fontWeight": "500"}),
        dbc.Tab(_strategies_tab(), label="Strategies",       tab_id="tab-strategies",
                label_style={"fontWeight": "500"}),
        dbc.Tab(_trading_tab(),    label="Trading Progress", tab_id="tab-trading",
                label_style={"fontWeight": "500"}),
        dbc.Tab(_risk_tab(),       label="Risk Controls",    tab_id="tab-risk",
                label_style={"fontWeight": "500"}),
        dbc.Tab(_backtesting_tab(),label="Backtesting",      tab_id="tab-backtesting",
                label_style={"fontWeight": "500"}),
    ], id="main-tabs", active_tab="tab-portfolio"),
], fluid=True, className="px-4")


# ── Pipeline runners (called from background threads) ──────────────────────────

def _run_pipeline(portfolio_name: str, use_cache: bool = True) -> None:
    from utils.logger import setup_logger
    from main import run_backtest_pipeline
    setup_logger()
    run_backtest_pipeline(use_cache=use_cache, portfolio_name=portfolio_name)


def _run_custom(strategy_names: list, portfolio_name: str, use_cache: bool = True) -> None:
    """Run backtest for a subset of strategies and save to custom_backtest_* files."""
    import json
    from utils.logger import setup_logger
    from data.ingestion import get_universe_ohlcv, build_close_matrix, build_return_matrix
    from strategies import get_all_strategies
    from backtest.engine import run_all_backtests
    from portfolio import get_portfolio
    from config.settings import RESULT_DIR, PORTFOLIO_PARAMS

    setup_logger()

    universe_data = get_universe_ohlcv(use_cache=use_cache)
    close         = build_close_matrix(universe_data)
    returns       = build_return_matrix(close)
    high_df       = pd.DataFrame({s: universe_data[s]["high"]   for s in universe_data})
    low_df        = pd.DataFrame({s: universe_data[s]["low"]    for s in universe_data})
    volume_df     = pd.DataFrame({s: universe_data[s]["volume"] for s in universe_data})

    all_strats = get_all_strategies()
    strategies = [s for s in all_strats if s.name in strategy_names]
    if not strategies:
        raise ValueError(f"No valid strategies in {strategy_names}")

    signals_dict = {}
    for strat in strategies:
        signals_dict[strat.name] = strat.run(
            close=close, returns=returns, high=high_df, low=low_df, volume=volume_df,
        )

    portfolio = get_portfolio(portfolio_name,
                              params=PORTFOLIO_PARAMS.get(portfolio_name))
    results = run_all_backtests(
        strategies=strategies, signals_dict=signals_dict,
        close=close, returns=returns, optimizer=portfolio,
    )

    metrics = {name: r.metrics for name, r in results.items()}
    with open(RESULT_DIR / "custom_backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    rets_df = pd.DataFrame({name: r.portfolio_returns for name, r in results.items()})
    rets_df.to_csv(RESULT_DIR / "custom_backtest_returns.csv")


def _run_custom_split(
    strategy_names: list, portfolio_name: str,
    use_cache: bool = True, train_frac: float = 0.70, val_frac: float = 0.15,
) -> None:
    """Run split backtest for a subset of strategies and save to split_*.* files."""
    from utils.logger import setup_logger
    from data.ingestion import get_universe_ohlcv, build_close_matrix, build_return_matrix
    from strategies import get_all_strategies
    from backtest.engine import run_all_backtests_with_splits
    from portfolio import get_portfolio
    from config.settings import RESULT_DIR, PORTFOLIO_PARAMS
    from utils.reporting import save_split_results

    setup_logger()

    universe_data = get_universe_ohlcv(use_cache=use_cache)
    close         = build_close_matrix(universe_data)
    returns       = build_return_matrix(close)
    high_df       = pd.DataFrame({s: universe_data[s]["high"]   for s in universe_data})
    low_df        = pd.DataFrame({s: universe_data[s]["low"]    for s in universe_data})
    volume_df     = pd.DataFrame({s: universe_data[s]["volume"] for s in universe_data})

    all_strats = get_all_strategies()
    strategies = [s for s in all_strats if s.name in strategy_names]
    if not strategies:
        raise ValueError(f"No valid strategies in {strategy_names}")

    signals_dict = {}
    for strat in strategies:
        signals_dict[strat.name] = strat.run(
            close=close, returns=returns, high=high_df, low=low_df, volume=volume_df,
        )

    portfolio = get_portfolio(portfolio_name, params=PORTFOLIO_PARAMS.get(portfolio_name))
    split_results = run_all_backtests_with_splits(
        strategies=strategies, signals_dict=signals_dict,
        close=close, returns=returns, optimizer=portfolio,
        train_frac=train_frac, val_frac=val_frac,
    )
    save_split_results(split_results)


def _run_custom_constant(
    strategy_names: list, portfolio_name: str, mode: str, use_cache: bool = True,
) -> None:
    """Run constant_weight or constant_strategy backtest for a subset of strategies."""
    from utils.logger import setup_logger
    from data.ingestion import get_universe_ohlcv, build_close_matrix, build_return_matrix
    from strategies import get_all_strategies
    from backtest.engine import run_constant_weight_backtests, run_constant_strategy_backtests
    from portfolio import get_portfolio
    from config.settings import PORTFOLIO_PARAMS
    from utils.reporting import save_results

    setup_logger()

    universe_data = get_universe_ohlcv(use_cache=use_cache)
    close         = build_close_matrix(universe_data)
    returns       = build_return_matrix(close)
    high_df       = pd.DataFrame({s: universe_data[s]["high"]   for s in universe_data})
    low_df        = pd.DataFrame({s: universe_data[s]["low"]    for s in universe_data})
    volume_df     = pd.DataFrame({s: universe_data[s]["volume"] for s in universe_data})

    all_strats = get_all_strategies()
    strategies = [s for s in all_strats if s.name in strategy_names]
    if not strategies:
        raise ValueError(f"No valid strategies in {strategy_names}")

    signals_dict = {}
    for strat in strategies:
        signals_dict[strat.name] = strat.run(
            close=close, returns=returns, high=high_df, low=low_df, volume=volume_df,
        )

    portfolio = get_portfolio(portfolio_name, params=PORTFOLIO_PARAMS.get(portfolio_name))
    runner = run_constant_weight_backtests if mode == "constant_weight" else run_constant_strategy_backtests
    results = runner(
        strategies=strategies, signals_dict=signals_dict,
        close=close, returns=returns, optimizer=portfolio,
    )
    save_results(results, prefix=mode)


# ── Callbacks ─────────────────────────────────────────────────────────────────

# ─── Portfolio Types tab ───────────────────────────────────────────────────────

@app.callback(
    Output("port-heatmap",    "figure"),
    Output("port-bar",        "figure"),
    Output("port-cumret",     "figure"),
    Output("port-table",      "data"),
    Output("port-table",      "columns"),
    Output("port-strat-filter","options"),
    Output("port-status",     "children"),
    Output("port-updated",    "children"),
    Input("run-port-btn",     "n_clicks"),
    Input("port-interval",    "n_intervals"),
    Input("port-strat-filter","value"),
    prevent_initial_call=False,
)
def _portfolio_cb(n_clicks, n_intervals, selected_strat):
    if ctx.triggered_id == "run-port-btn" and n_clicks:
        def _t():
            try:
                _job_set("port", "running")
                _run_pipeline("all", use_cache=True)
                _job_set("port", "done")
            except Exception as e:
                _job_set("port", "error", str(e))
        threading.Thread(target=_t, daemon=True).start()

    badge   = _status_badge("port")
    metrics = load_comparison_metrics()
    updated = datetime.now().strftime("%H:%M:%S")

    if not metrics:
        e = _empty_fig("Click 'Run Portfolio Comparison' to generate data")
        return e, e, e, [], [], [], badge, updated

    pivot   = comparison_pivot(metrics)
    returns = load_comparison_returns()
    df      = metrics_to_df(metrics)

    strat_opts = [{"label": s, "value": s} for s in pivot.index]

    # Heatmap
    heatmap = go.Figure(go.Heatmap(
        x=pivot.columns.tolist(), y=pivot.index.tolist(),
        z=pivot.values, colorscale="RdYlGn",
        text=pivot.round(3).values, texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="Sharpe", thickness=14),
        zmid=0,
    ))
    heatmap.update_layout(
        template=CHART_TEMPLATE, paper_bgcolor=CHART_BG,
        title="Sharpe: Strategy × Portfolio", height=360,
        xaxis_title="Portfolio", yaxis_title="Strategy",
    )

    # Grouped bar chart (portfolio on x, strategies as groups)
    bar = go.Figure()
    for i, strat in enumerate(pivot.index):
        bar.add_trace(go.Bar(
            name=strat, x=pivot.columns.tolist(), y=pivot.loc[strat].tolist(),
            marker_color=COLORS[i % len(COLORS)],
        ))
    bar.update_layout(
        template=CHART_TEMPLATE, paper_bgcolor=CHART_BG, barmode="group",
        title="Sharpe by Portfolio Type", yaxis_title="Sharpe",
        height=360, legend=dict(orientation="h", y=-0.3, font_size=11),
    )

    # Cumulative returns — filtered by selected strategy
    if not returns.empty:
        if selected_strat:
            cols = [c for c in returns.columns if c.startswith(f"{selected_strat}/")]
        else:
            cols = returns.columns.tolist()
        ret_sub = returns[cols] if cols else returns
        cumret  = _cumret_fig(ret_sub,
                              f"Cumulative Returns — {selected_strat or 'All Strategies'}")
    else:
        cumret = _empty_fig()

    cols, data = _metrics_table(df)
    return heatmap, bar, cumret, data, cols, strat_opts, badge, updated


# ─── Strategies tab ────────────────────────────────────────────────────────────

@app.callback(
    Output("strat-cumret",  "figure"),
    Output("strat-rolling", "figure"),
    Output("strat-drawdown","figure"),
    Output("strat-table",   "data"),
    Output("strat-table",   "columns"),
    Output("strat-status",  "children"),
    Output("strat-updated", "children"),
    Input("run-strat-btn",  "n_clicks"),
    Input("strat-interval", "n_intervals"),
    State("strat-port-select","value"),
    prevent_initial_call=False,
)
def _strategies_cb(n_clicks, n_intervals, portfolio_name):
    if ctx.triggered_id == "run-strat-btn" and n_clicks:
        pname = portfolio_name or "max_sharpe"
        def _t():
            try:
                _job_set("strat", "running")
                _run_pipeline(pname, use_cache=True)
                _job_set("strat", "done")
            except Exception as e:
                _job_set("strat", "error", str(e))
        threading.Thread(target=_t, daemon=True).start()

    badge   = _status_badge("strat")
    metrics = load_strategy_metrics()
    returns = load_strategy_returns()
    updated = datetime.now().strftime("%H:%M:%S")

    df           = metrics_to_df(metrics)
    cols, data   = _metrics_table(df)

    if returns.empty:
        e = _empty_fig("Click 'Run Strategy Backtest' to generate data")
        return e, e, e, data, cols, badge, updated

    cumret  = _cumret_fig(returns, "Cumulative Returns — All Strategies")
    rolling = _rolling_sharpe_fig(returns)
    dd      = _drawdown_fig(returns, "Drawdown — All Strategies")

    return cumret, rolling, dd, data, cols, badge, updated


# ─── Trading Progress tab ─────────────────────────────────────────────────────

def _kpi_card(title: str, value: str, color: str = "light") -> dbc.Col:
    return dbc.Col(dbc.Card([
        dbc.CardHeader(title, className="text-muted", style={"fontSize": "12px"}),
        dbc.CardBody(html.H5(value, className=f"text-{color} mb-0")),
    ], className="text-center h-100"), md=3, className="mb-2")


def _check_alpaca() -> tuple[bool, str, float, float, str]:
    """Try connecting to Alpaca and return (connected, mode, equity, buying_power, error)."""
    try:
        from config.client import get_trading_client
        from config.settings import PAPER_TRADING
        client  = get_trading_client()
        account = client.get_account()
        mode    = "PAPER" if PAPER_TRADING else "LIVE"
        return True, mode, float(account.equity), float(account.buying_power), ""
    except Exception as e:
        from config.settings import PAPER_TRADING
        mode = "PAPER" if PAPER_TRADING else "LIVE"
        return False, mode, 0.0, 0.0, str(e)


def _fetch_live_prices() -> dict:
    """Fetch latest Alpaca trade prices. Returns {} on failure."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        from config.settings import (PAPER_TRADING, PAPER_API_KEY, PAPER_API_SECRET,
                                     LIVE_API_KEY, LIVE_API_SECRET, UNIVERSE)
        key    = PAPER_API_KEY if PAPER_TRADING else LIVE_API_KEY
        secret = PAPER_API_SECRET if PAPER_TRADING else LIVE_API_SECRET
        client = StockHistoricalDataClient(key, secret)
        trades = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=UNIVERSE))
        return {sym: float(t.price) for sym, t in trades.items()}
    except Exception:
        return {}


@app.callback(
    Output("alpaca-connection-badge",  "children"),
    Output("trading-mode-badge",       "children"),
    Output("alpaca-account-info",      "children"),
    Output("trading-kpis",             "children"),
    Output("trading-strategy-panel",   "children"),
    Output("trading-nav",              "figure"),
    Output("positions-table",          "data"),
    Output("positions-table",          "columns"),
    Output("trades-table",             "data"),
    Output("trades-table",             "columns"),
    Output("trading-updated",          "children"),
    Input("refresh-trading-btn", "n_clicks"),
    Input("trading-interval",    "n_intervals"),
    prevent_initial_call=False,
)
def _trading_cb(n_clicks, n_intervals):
    from config.settings import STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRAILING_STOP_PCT, USE_TRAILING_STOP
    from db.engine_controls import load_engine_controls

    connected, mode, equity, buying_power, err = _check_alpaca()
    live_prices = _fetch_live_prices() if connected else {}
    state       = load_live_state()
    controls    = load_engine_controls()
    updated     = datetime.now().strftime("%H:%M:%S")

    # ── Connection badge ──────────────────────────────────────────────────────
    conn_badge = (
        dbc.Badge("Alpaca: Connected",    color="success", pill=True) if connected else
        dbc.Badge("Alpaca: Disconnected", color="danger",  pill=True, title=err)
    )

    # ── Mode badge ────────────────────────────────────────────────────────────
    mode_badge = dbc.Badge(f"{mode} Mode",
                           color="warning" if mode == "LIVE" else "info", pill=True)

    # ── Account info line ─────────────────────────────────────────────────────
    if connected:
        last_run = (state or {}).get("last_run")
        active   = (state or {}).get("active_strategy", "–")
        acct_info = (
            f"Equity: ${equity:,.2f}  |  Buying power: ${buying_power:,.2f}"
            f"  |  Strategy: {active}"
            + (f"  |  Last run: {last_run}" if last_run else "")
        )
    else:
        acct_info = f"Could not reach Alpaca — {err[:80]}"

    # ── Kill switch engine status ─────────────────────────────────────────────
    kill_active = controls.get("kill_switch", False)

    if not state:
        engine_badge = dbc.Badge("HALTED", color="danger") if kill_active else dbc.Badge("IDLE", color="secondary")
        placeholder_kpis = [
            _kpi_card("NAV", "–"), _kpi_card("Cash", "–"),
            _kpi_card("Total P&L", "–"), _kpi_card("Open Positions", "–"),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Engine", className="text-muted", style={"fontSize": "12px"}),
                dbc.CardBody(engine_badge),
            ], className="text-center h-100"), md=3, className="mb-2"),
        ]
        return (conn_badge, mode_badge, acct_info, placeholder_kpis,
                html.Span("No live state — start engine first", className="text-muted"),
                _empty_fig("No live state — run  python main.py --mode live  first"),
                [], [], [], [], updated)

    cash      = state.get("cash_usd", 0)
    positions = {s: q for s, q in state.get("positions", {}).items() if q != 0}
    nav_hist  = state.get("nav_history", [])
    active    = state.get("active_strategy", "–")
    trades    = state.get("trade_log", [])
    entries   = state.get("position_entries", {})
    weights   = state.get("current_weights", {})

    # Compute NAV using live prices if available
    if live_prices:
        pos_value = sum(qty * live_prices.get(sym, 0) for sym, qty in positions.items())
        cur_nav   = cash + pos_value
    else:
        cur_nav = nav_hist[-1]["nav"] if nav_hist else cash

    start   = nav_hist[0]["nav"] if nav_hist else cash
    pnl     = cur_nav - start
    pnl_pct = pnl / start * 100 if start else 0
    pnl_col = "success" if pnl >= 0 else "danger"

    engine_badge = dbc.Badge("HALTED", color="danger") if kill_active else dbc.Badge("LIVE", color="success")

    kpis = [
        _kpi_card("NAV",              f"${cur_nav:,.2f}",                 "info"),
        _kpi_card("Cash",             f"${cash:,.2f}",                    "light"),
        _kpi_card("Total P&L",        f"${pnl:+,.2f} ({pnl_pct:+.2f}%)", pnl_col),
        _kpi_card("Open Positions",   str(len(positions)),                 "light"),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Engine", className="text-muted", style={"fontSize": "12px"}),
            dbc.CardBody(engine_badge),
        ], className="text-center h-100"), md=3, className="mb-2"),
    ]

    # ── Active strategy + weights panel ──────────────────────────────────────
    if active and active != "–":
        strat_label = html.Span(
            active.replace("_", " ").title(),
            className="fw-bold fs-6 me-3",
            style={"color": "#ffc107"},
        )
        weight_items = []
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            if w > 0.001:
                weight_items.append(
                    dbc.Badge(
                        f"{sym} {w*100:.1f}%",
                        color="primary" if w >= 0.10 else "secondary",
                        className="me-1 mb-1",
                        style={"fontSize": "12px"},
                    )
                )
        strategy_panel = html.Div([strat_label] + weight_items)
    else:
        strategy_panel = html.Span("No active strategy yet", className="text-muted")

    # ── NAV history chart ─────────────────────────────────────────────────────
    if nav_hist:
        dates   = [e["date"] for e in nav_hist]
        navs    = [e["nav"]  for e in nav_hist]
        nav_fig = go.Figure([
            go.Scatter(
                x=dates, y=navs, mode="lines",
                fill="tozeroy",
                line=dict(color="#00b4d8", width=2),
                fillcolor="rgba(0,180,216,0.12)",
            ),
        ])
        nav_fig.add_hline(y=start, line_dash="dot", line_color="gray",
                          annotation_text="Start NAV")
        nav_fig.update_layout(
            template=CHART_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            title="Portfolio NAV History", yaxis_title="NAV (USD)", height=320,
            showlegend=False,
        )
    else:
        nav_fig = _empty_fig("No NAV history recorded yet")

    # ── Positions table (with live P&L, SL, TP) ──────────────────────────────
    pos_rows = []
    for sym, qty in positions.items():
        entry_info  = entries.get(sym, {})
        entry_px    = entry_info.get("entry_price", entry_info.get("price", 0))
        peak_px     = entry_info.get("peak_price",  entry_info.get("peak", entry_px))
        current_px  = live_prices.get(sym, 0)
        pnl_d       = (current_px - entry_px) * qty if current_px and entry_px else 0
        pnl_p       = (current_px / entry_px - 1) * 100 if entry_px and current_px else 0
        sl_px       = entry_px * (1 - STOP_LOSS_PCT)
        tp_px       = entry_px * (1 + TAKE_PROFIT_PCT)
        trail_px    = peak_px  * (1 - TRAILING_STOP_PCT) if USE_TRAILING_STOP else None
        row = {
            "Symbol":   sym,
            "Qty":      round(qty, 4),
            "Entry $":  round(entry_px, 2)   if entry_px  else "–",
            "Current $":round(current_px, 2) if current_px else "–",
            "P&L $":    round(pnl_d, 2),
            "P&L %":    round(pnl_p, 2),
            "SL $":     round(sl_px, 2)      if entry_px  else "–",
            "TP $":     round(tp_px, 2)      if entry_px  else "–",
        }
        if USE_TRAILING_STOP and trail_px:
            row["Trail $"] = round(trail_px, 2)
        pos_rows.append(row)

    pos_cols = [{"name": k, "id": k} for k in (pos_rows[0].keys() if pos_rows else
                ["Symbol", "Qty", "Entry $", "Current $", "P&L $", "P&L %", "SL $", "TP $"])]

    # ── Trade log (most recent 20, newest first, buy/sell colored) ────────────
    recent  = list(reversed(trades[-20:]))
    tr_cols = [{"name": k, "id": k} for k in (recent[0].keys() if recent else
               ["time", "symbol", "side", "qty", "price", "reason"])]

    return (conn_badge, mode_badge, acct_info, kpis, strategy_panel,
            nav_fig, pos_rows, pos_cols, recent, tr_cols, updated)


# ─── Risk Controls tab ────────────────────────────────────────────────────────

@app.callback(
    Output("kill-switch-badge",   "children"),
    Output("risk-connection-detail", "children"),
    Input("kill-activate-btn",    "n_clicks"),
    Input("kill-deactivate-btn",  "n_clicks"),
    Input("risk-interval",        "n_intervals"),
    State("kill-switch-mode",     "value"),
    State("kill-switch-note",     "value"),
    prevent_initial_call=False,
)
def _risk_cb(activate_clicks, deactivate_clicks, n_intervals, ks_mode, ks_note):
    from db.engine_controls import load_engine_controls, save_engine_controls

    trigger = ctx.triggered_id
    if trigger == "kill-activate-btn" and activate_clicks:
        close_all = (ks_mode == "close_all")
        save_engine_controls(
            kill_switch=True,
            kill_mode="halt",
            close_all_triggered=close_all,
            note=ks_note or "",
        )
    elif trigger == "kill-deactivate-btn" and deactivate_clicks:
        save_engine_controls(
            kill_switch=False,
            kill_mode="halt",
            close_all_triggered=False,
            note="",
        )

    controls     = load_engine_controls()
    kill_active  = controls.get("kill_switch", False)
    kill_note    = controls.get("note", "")

    if kill_active:
        badge = dbc.Badge(
            "ACTIVE" + (f" — {kill_note}" if kill_note else ""),
            color="danger", pill=True,
        )
    else:
        badge = dbc.Badge("INACTIVE", color="success", pill=True)

    # Connection status detail
    connected, mode, equity, buying_power, err = _check_alpaca()
    if connected:
        conn_detail = html.Div([
            dbc.Row([
                dbc.Col(html.Span("Alpaca:", className="text-muted"), width="auto"),
                dbc.Col(dbc.Badge("CONNECTED", color="success"), width="auto"),
            ], className="mb-2 align-items-center"),
            html.Small(f"Mode: {mode}  |  Equity: ${equity:,.2f}  |  "
                       f"Buying power: ${buying_power:,.2f}", className="text-muted"),
        ])
    else:
        conn_detail = html.Div([
            dbc.Row([
                dbc.Col(html.Span("Alpaca:", className="text-muted"), width="auto"),
                dbc.Col(dbc.Badge("DISCONNECTED", color="danger"), width="auto"),
            ], className="mb-2 align-items-center"),
            html.Small(err[:120], className="text-danger"),
        ])

    return badge, conn_detail


@app.callback(
    Output("kill-switch-msg", "children"),
    Input("kill-activate-btn",   "n_clicks"),
    Input("kill-deactivate-btn", "n_clicks"),
    State("kill-switch-mode",    "value"),
    prevent_initial_call=True,
)
def _kill_msg(activate, deactivate, mode_val):
    trigger = ctx.triggered_id
    if trigger == "kill-activate-btn" and activate:
        if mode_val == "close_all":
            return "Kill switch ACTIVATED — engine will close all positions on next monitor cycle."
        return "Kill switch ACTIVATED — new orders halted."
    if trigger == "kill-deactivate-btn" and deactivate:
        return "Kill switch DEACTIVATED — engine resumed."
    return ""


# ─── Backtesting tab ──────────────────────────────────────────────────────────

@app.callback(
    Output("bt-cumret",        "figure"),
    Output("bt-drawdown",      "figure"),
    Output("bt-rolling-sharpe","figure"),
    Output("bt-split-table",   "data"),
    Output("bt-split-table",   "columns"),
    Output("bt-strat-view",    "options"),
    Output("bt-status",        "children"),
    Input("run-bt-btn",        "n_clicks"),
    Input("bt-interval",       "n_intervals"),
    State("bt-strat-select",   "value"),
    State("bt-port-select",    "value"),
    State("bt-use-cache",      "value"),
    State("bt-run-split",      "value"),
    State("bt-train-frac",     "value"),
    State("bt-val-frac",       "value"),
    State("bt-strat-view",     "value"),
    State("bt-mode",           "value"),
    prevent_initial_call=False,
)
def _backtesting_cb(
    n_clicks, n_intervals, strategies, portfolio, use_cache,
    run_split, train_frac, val_frac, strat_view, bt_mode,
):
    run_split  = bool(run_split)
    train_frac = (train_frac or 70) / 100
    val_frac   = (val_frac   or 15) / 100
    bt_mode    = bt_mode or "walk_forward"

    if ctx.triggered_id == "run-bt-btn" and n_clicks:
        strats = strategies or STRATEGY_NAMES
        pname  = portfolio or "max_sharpe"
        if run_split:
            tf, vf = train_frac, val_frac
            def _t():
                try:
                    _job_set("bt", "running")
                    _run_custom_split(strats, pname, use_cache=bool(use_cache),
                                      train_frac=tf, val_frac=vf)
                    _job_set("bt", "done")
                except Exception as e:
                    _job_set("bt", "error", str(e))
        elif bt_mode == "constant_weight":
            def _t():
                try:
                    _job_set("bt", "running")
                    _run_custom_constant(strats, pname, "constant_weight", bool(use_cache))
                    _job_set("bt", "done")
                except Exception as e:
                    _job_set("bt", "error", str(e))
        elif bt_mode == "constant_strategy":
            def _t():
                try:
                    _job_set("bt", "running")
                    _run_custom_constant(strats, pname, "constant_strategy", bool(use_cache))
                    _job_set("bt", "done")
                except Exception as e:
                    _job_set("bt", "error", str(e))
        else:
            def _t():
                try:
                    _job_set("bt", "running")
                    _run_custom(strats, pname, use_cache=bool(use_cache))
                    _job_set("bt", "done")
                except Exception as e:
                    _job_set("bt", "error", str(e))
        threading.Thread(target=_t, daemon=True).start()

    badge = _status_badge("bt")

    if run_split:
        metrics = load_split_metrics()
        returns = load_split_returns()

        if not metrics or returns.empty:
            e = _empty_fig("Enable split mode and click 'Run Backtest'")
            return e, e, e, [], [], [], badge

        strat_names = sorted({
            key.rsplit("/", 1)[0] for key in metrics
            if "/" in key and key.rsplit("/", 1)[1] in ("train", "val", "test")
        })
        strat_opts = [{"label": s, "value": s} for s in strat_names]

        cumret  = _split_cumret_fig(returns, strat_filter=strat_view)

        show = [strat_view] if strat_view and strat_view in strat_names else strat_names
        combined = pd.DataFrame()
        for strat in show:
            parts = [
                returns[f"{strat}/{p}"].dropna()
                for p in ("train", "val", "test")
                if f"{strat}/{p}" in returns.columns
            ]
            if parts:
                combined[strat] = pd.concat(parts).sort_index()

        dd      = _drawdown_fig(combined, "Drawdown") if not combined.empty else _empty_fig()
        rolling = _rolling_sharpe_fig(combined, 63)   if not combined.empty else _empty_fig()
        cols, data = _split_metrics_table(metrics)
        return cumret, dd, rolling, data, cols, strat_opts, badge

    else:
        if bt_mode == "constant_weight":
            metrics = load_constant_weight_metrics()
            returns = load_constant_weight_returns()
        elif bt_mode == "constant_strategy":
            metrics = load_constant_strategy_metrics()
            returns = load_constant_strategy_returns()
        else:
            metrics = load_custom_metrics()
            returns = load_custom_returns()
        df      = metrics_to_df(metrics)

        strat_opts = (
            [{"label": row["label"], "value": row["label"]} for _, row in df.iterrows()]
            if not df.empty else []
        )
        t_cols, t_data = _metrics_table(df)

        if returns.empty:
            e = _empty_fig("Configure and click 'Run Backtest' to see results")
            return e, e, e, t_data, t_cols, strat_opts, badge

        sub     = returns[[strat_view]] if strat_view and strat_view in returns.columns else returns
        cumret  = _cumret_fig(sub,  "Custom Backtest — Cumulative Returns")
        dd      = _drawdown_fig(sub, "Custom Backtest — Drawdown")
        rolling = _rolling_sharpe_fig(sub, 63)

        return cumret, dd, rolling, t_data, t_cols, strat_opts, badge


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting dashboard at http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
