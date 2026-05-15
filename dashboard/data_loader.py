"""
dashboard/data_loader.py
========================
Read pre-computed results from the results/ directory.

File sources
------------
  results/strategy_metrics.json          — single-portfolio backtest run
  results/portfolio_returns.csv          — single-portfolio returns time series
  results/portfolio_comparison_metrics.json — --portfolio all comparison run
  results/portfolio_comparison_returns.csv  — comparison returns time series
  results/custom_backtest_metrics.json   — dashboard Backtesting tab run
  results/custom_backtest_returns.csv    — Backtesting tab returns
  results/live_state.json               — live trading engine state
"""

import json
import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results"


# ── Raw loaders ────────────────────────────────────────────────────────────────

def _load_json(filename: str) -> dict:
    f = RESULT_DIR / filename
    if not f.exists():
        return {}
    with open(f) as fp:
        return json.load(fp)


def _load_csv(filename: str) -> pd.DataFrame:
    f = RESULT_DIR / filename
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f, index_col=0, parse_dates=True)


def load_strategy_metrics()     -> dict:        return _load_json("strategy_metrics.json")
def load_strategy_returns()     -> pd.DataFrame: return _load_csv("portfolio_returns.csv")
def load_comparison_metrics()   -> dict:        return _load_json("portfolio_comparison_metrics.json")
def load_comparison_returns()   -> pd.DataFrame: return _load_csv("portfolio_comparison_returns.csv")
def load_custom_metrics()       -> dict:        return _load_json("custom_backtest_metrics.json")
def load_custom_returns()       -> pd.DataFrame: return _load_csv("custom_backtest_returns.csv")
def load_live_state()           -> dict:        return _load_json("live_state.json")


# ── Parsers ────────────────────────────────────────────────────────────────────

def metrics_to_df(metrics: dict) -> pd.DataFrame:
    """Flatten a {label: {metric: value}} dict into a DataFrame row per label."""
    rows = []
    for label, m in metrics.items():
        if isinstance(m, dict) and "error" not in m:
            rows.append({"label": label, **m})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def comparison_pivot(metrics: dict, value: str = "sharpe") -> pd.DataFrame:
    """
    Build a strategy × portfolio pivot table from comparison metrics.
    Keys must have the form "strategy/portfolio".
    """
    data = []
    for key, m in metrics.items():
        if "/" not in key or not isinstance(m, dict):
            continue
        strategy, portfolio = key.split("/", 1)
        data.append({"strategy": strategy, "portfolio": portfolio,
                     **{k: m.get(k) for k in ("sharpe", "cagr", "max_drawdown", "annual_vol")}})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    return df.pivot_table(index="strategy", columns="portfolio", values=value)
