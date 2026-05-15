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


def _trading_tab() -> html.Div:
    return html.Div([
        dcc.Interval(id="trading-interval", interval=60_000, n_intervals=0),
        dbc.Row([
            dbc.Col(dbc.Button("Refresh", id="refresh-trading-btn",
                               color="secondary", size="sm"), width="auto"),
            dbc.Col(html.Small(id="trading-updated", className="text-muted"),
                    className="align-self-center"),
        ], className="mb-3 g-2 align-items-center"),

        dbc.Row(id="trading-kpis", className="mb-3"),
        _card(dcc.Graph(id="trading-nav")),

        dbc.Row([
            dbc.Col(_card([
                html.H6("Open Positions", className="text-muted small mb-2"),
                dash_table.DataTable(id="positions-table", page_size=10, **_TABLE_STYLE),
            ]), md=6),
            dbc.Col(_card([
                html.H6("Recent Trades", className="text-muted small mb-2"),
                dash_table.DataTable(id="trades-table", page_size=10, **_TABLE_STYLE),
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
                _card(dash_table.DataTable(id="bt-table", sort_action="native",
                                           page_size=10, **_TABLE_STYLE)),
                dbc.Row([
                    dbc.Col(_card(dcc.Graph(id="bt-cumret")), md=6),
                    dbc.Col(_card(dcc.Graph(id="bt-drawdown")), md=6),
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
    high_df       = pd.DataFrame({s: universe_data[s]["high"] for s in universe_data})
    low_df        = pd.DataFrame({s: universe_data[s]["low"]  for s in universe_data})

    all_strats = get_all_strategies()
    strategies = [s for s in all_strats if s.name in strategy_names]
    if not strategies:
        raise ValueError(f"No valid strategies in {strategy_names}")

    signals_dict = {}
    for strat in strategies:
        signals_dict[strat.name] = strat.run(
            close=close, returns=returns, high=high_df, low=low_df,
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


@app.callback(
    Output("trading-kpis",    "children"),
    Output("trading-nav",     "figure"),
    Output("positions-table", "data"),
    Output("positions-table", "columns"),
    Output("trades-table",    "data"),
    Output("trades-table",    "columns"),
    Output("trading-updated", "children"),
    Input("refresh-trading-btn","n_clicks"),
    Input("trading-interval",  "n_intervals"),
    prevent_initial_call=False,
)
def _trading_cb(n_clicks, n_intervals):
    state   = load_live_state()
    updated = datetime.now().strftime("%H:%M:%S")

    if not state:
        placeholder = [
            _kpi_card("NAV", "–"), _kpi_card("Cash", "–"),
            _kpi_card("Total P&L", "–"), _kpi_card("Active Strategy", "–"),
        ]
        return (placeholder,
                _empty_fig("No live state — run  python main.py --mode live  first"),
                [], [], [], [], updated)

    cash     = state.get("cash_usd", 0)
    positions = {s: q for s, q in state.get("positions", {}).items() if q != 0}
    nav_hist  = state.get("nav_history", [])
    active    = state.get("active_strategy", "–")
    trades    = state.get("trade_log", [])
    entries   = state.get("position_entries", {})

    cur_nav = nav_hist[-1]["nav"] if nav_hist else cash
    start   = nav_hist[0]["nav"]  if nav_hist else cash
    pnl     = cur_nav - start
    pnl_pct = pnl / start * 100 if start else 0
    pnl_col = "success" if pnl >= 0 else "danger"

    kpis = [
        _kpi_card("NAV",             f"${cur_nav:,.2f}",                 "info"),
        _kpi_card("Cash",            f"${cash:,.2f}",                     "light"),
        _kpi_card("Total P&L",       f"${pnl:+,.2f} ({pnl_pct:+.2f}%)", pnl_col),
        _kpi_card("Active Strategy", active,                               "warning"),
    ]

    # NAV history chart
    if nav_hist:
        dates = [e["date"] for e in nav_hist]
        navs  = [e["nav"]  for e in nav_hist]
        nav_fig = go.Figure([
            go.Scatter(x=dates, y=navs, mode="lines+markers",
                       line=dict(color="#0d6efd", width=2),
                       marker=dict(size=4)),
        ])
        nav_fig.add_hline(y=start, line_dash="dot", line_color="gray",
                          annotation_text="Start NAV")
        nav_fig.update_layout(
            template=CHART_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            title="Portfolio NAV History", yaxis_title="NAV (USD)", height=340,
        )
    else:
        nav_fig = _empty_fig("No NAV history recorded yet")

    # Positions table
    pos_rows = [
        {"Symbol": sym, "Qty": qty,
         "Entry Price": f"${entries.get(sym, {}).get('price', 0):,.2f}",
         "Entry Date":  entries.get(sym, {}).get("date", "–"),
         "Peak":        f"${entries.get(sym, {}).get('peak', 0):,.2f}"}
        for sym, qty in positions.items()
    ]
    pos_cols = [{"name": k, "id": k} for k in (pos_rows[0].keys() if pos_rows else [])]

    # Trade log (most recent 20, newest first)
    recent = list(reversed(trades[-20:]))
    tr_cols = [{"name": k, "id": k} for k in (recent[0].keys() if recent else [])]

    return kpis, nav_fig, pos_rows, pos_cols, recent, tr_cols, updated


# ─── Backtesting tab ──────────────────────────────────────────────────────────

@app.callback(
    Output("bt-cumret",  "figure"),
    Output("bt-drawdown","figure"),
    Output("bt-table",   "data"),
    Output("bt-table",   "columns"),
    Output("bt-status",  "children"),
    Input("run-bt-btn",  "n_clicks"),
    Input("bt-interval", "n_intervals"),
    State("bt-strat-select","value"),
    State("bt-port-select", "value"),
    State("bt-use-cache",   "value"),
    prevent_initial_call=False,
)
def _backtesting_cb(n_clicks, n_intervals, strategies, portfolio, use_cache):
    if ctx.triggered_id == "run-bt-btn" and n_clicks:
        strats = strategies or STRATEGY_NAMES
        pname  = portfolio or "max_sharpe"
        def _t():
            try:
                _job_set("bt", "running")
                _run_custom(strats, pname, use_cache=bool(use_cache))
                _job_set("bt", "done")
            except Exception as e:
                _job_set("bt", "error", str(e))
        threading.Thread(target=_t, daemon=True).start()

    badge   = _status_badge("bt")
    metrics = load_custom_metrics()
    returns = load_custom_returns()

    df         = metrics_to_df(metrics)
    cols, data = _metrics_table(df)

    if returns.empty:
        e = _empty_fig("Configure and click 'Run Backtest' to see results")
        return e, e, data, cols, badge

    cumret = _cumret_fig(returns, "Custom Backtest — Cumulative Returns")
    dd     = _drawdown_fig(returns, "Custom Backtest — Drawdown")

    return cumret, dd, data, cols, badge


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting dashboard at http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
