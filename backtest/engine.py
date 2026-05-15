"""
backtest/engine.py
==================
Walk-forward backtester with T+1 execution timing.

Execution rule
--------------
  Signal computed from data up to close of day T
  → Portfolio rebalanced at open/close of T+1  (shift(1) in base.py)
  → Returns measured from T+1 to T+2

Uses 252-day annualization (US stock market, not 365 like crypto).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger

from config.settings import (
    TRANSACTION_COST_BP, SLIPPAGE_BP,
    RISK_LOOKBACK_DAYS, MAX_WEIGHT_SUM,
    LONG_SHORT, TRADING_DAYS,
)


@dataclass
class BacktestResult:
    strategy_name:     str
    portfolio_returns: pd.Series
    weights_history:   pd.DataFrame
    signal_history:    pd.DataFrame
    metrics:           dict = field(default_factory=dict)

    def __post_init__(self):
        self.metrics = compute_metrics(self.portfolio_returns, self.strategy_name)


def compute_metrics(returns: pd.Series, name: str = "", initial_capital: float = 10_000) -> dict:
    """Standard performance metrics. Annualized over 252 trading days."""
    r = returns.dropna()
    if len(r) < 30:
        return {"error": "insufficient data"}

    ann    = TRADING_DAYS
    mu     = r.mean() * ann
    vol    = r.std()  * np.sqrt(ann)
    sharpe = mu / vol if vol > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(ann)
    sortino  = mu / downside if downside > 0 else np.nan

    cum         = (1 + r).cumprod()
    rolling_max = cum.cummax()
    dd          = (cum - rolling_max) / rolling_max
    max_dd      = dd.min()
    calmar      = mu / abs(max_dd) if max_dd != 0 else np.nan

    total_r = cum.iloc[-1] - 1
    n_years = len(r) / ann
    cagr    = (1 + total_r) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    win_rate     = (r > 0).mean()
    avg_win      = r[r > 0].mean() if (r > 0).any() else 0
    avg_loss     = r[r < 0].mean() if (r < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

    pnl_series     = r * initial_capital
    final_capital  = initial_capital * cum.iloc[-1]

    return {
        "strategy":       name,
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "calmar":         round(calmar, 3),
        "cagr":           round(cagr, 4),
        "annual_vol":     round(vol, 4),
        "max_drawdown":   round(max_dd, 4),
        "total_return":   round(total_r, 4),
        "win_rate":       round(win_rate, 4),
        "profit_factor":  round(profit_factor, 3),
        "n_days":         len(r),
        "initial_capital": round(initial_capital, 2),
        "final_capital":   round(final_capital, 2),
        "total_pnl":       round(pnl_series.sum(), 2),
        "avg_daily_pnl":   round(pnl_series.mean(), 2),
        "best_day_pnl":    round(pnl_series.max(), 2),
        "worst_day_pnl":   round(pnl_series.min(), 2),
    }


def apply_transaction_costs(
    returns:  pd.Series,
    weights:  pd.DataFrame,
    cost_bp:  int = TRANSACTION_COST_BP,
    slip_bp:  int = SLIPPAGE_BP,
) -> pd.Series:
    """Deduct round-trip costs proportional to daily turnover."""
    total_bp = (cost_bp + slip_bp) / 10_000
    turnover = weights.diff().abs().sum(axis=1)
    cost     = turnover * total_bp
    return returns - cost.reindex(returns.index, fill_value=0)


class WalkForwardBacktester:
    """
    Walk-forward portfolio construction with T+1 execution.

    Parameters
    ----------
    signals   : pre-computed signal matrix (already shifted by base.py)
    close     : close price matrix
    returns   : log return matrix
    optimizer : callable(signals_row, cov_matrix) → weights dict
    """

    def __init__(self, signals, close, returns, optimizer, rebal_freq="1D"):
        self.signals    = signals
        self.close      = close
        self.returns    = returns
        self.optimizer  = optimizer
        self.rebal_freq = rebal_freq

    def run(self, strategy_name: str = "unnamed") -> BacktestResult:
        logger.info(f"Backtesting: {strategy_name}")

        weights_history = pd.DataFrame(0.0, index=self.close.index, columns=self.close.columns)

        rebal_dates = (
            self.close.index[RISK_LOOKBACK_DAYS:]
            if self.rebal_freq == "1D"
            else self.close.resample(self.rebal_freq).last().index
        )

        current_weights = pd.Series(0.0, index=self.close.columns)

        for dt in rebal_dates:
            if dt not in self.signals.index:
                continue

            # Use .loc to select returns up to and including dt, then take
            # the last RISK_LOOKBACK_DAYS rows. This is unambiguous regardless
            # of index alignment between close and returns.
            cov_matrix = (
                self.returns.loc[:dt]
                .iloc[-RISK_LOOKBACK_DAYS:]
                .cov() * TRADING_DAYS
            )

            signal_row = self.signals.loc[dt]
            try:
                new_weights = self.optimizer(
                    signals    = signal_row,
                    cov        = cov_matrix,
                    long_short = LONG_SHORT,
                )
                current_weights = pd.Series(new_weights).reindex(self.close.columns, fill_value=0)
            except Exception as e:
                logger.warning(f"Optimizer failed on {dt}: {e} — holding previous weights")

            weights_history.loc[dt] = current_weights

        weights_history = weights_history.replace(0, np.nan).ffill().fillna(0)

        fwd_returns  = self.returns.shift(-1)
        port_returns = (weights_history * fwd_returns).sum(axis=1)
        port_returns = port_returns.iloc[RISK_LOOKBACK_DAYS:-1]
        port_returns = apply_transaction_costs(port_returns, weights_history.loc[port_returns.index])

        return BacktestResult(
            strategy_name     = strategy_name,
            portfolio_returns  = port_returns,
            weights_history   = weights_history,
            signal_history    = self.signals,
        )


def run_all_backtests(strategies, signals_dict, close, returns, optimizer) -> dict[str, BacktestResult]:
    results = {}
    for strategy in strategies:
        name = strategy.name
        if name not in signals_dict:
            logger.warning(f"No signals for {name} — skipping")
            continue
        bt = WalkForwardBacktester(
            signals   = signals_dict[name],
            close     = close,
            returns   = returns,
            optimizer = optimizer,
        )
        results[name] = bt.run(strategy_name=name)
        logger.success(
            f"{name}: Sharpe={results[name].metrics.get('sharpe')} "
            f"| CAGR={results[name].metrics.get('cagr')}"
        )
    return results


def run_portfolio_comparison(
    strategies,
    signals_dict: dict,
    close,
    returns,
    portfolios: list,
) -> dict[str, dict[str, BacktestResult]]:
    """
    Run every (strategy, portfolio) combination and return a nested dict.

    Returns
    -------
    dict: strategy_name → {portfolio_name → BacktestResult}

    Example
    -------
    results["momentum"]["min_variance"].metrics["sharpe"]
    """
    results: dict[str, dict[str, BacktestResult]] = {}

    for strategy in strategies:
        sname = strategy.name
        if sname not in signals_dict:
            logger.warning(f"No signals for {sname} — skipping")
            continue

        results[sname] = {}
        for portfolio in portfolios:
            pname = portfolio.name
            label = f"{sname}/{pname}"
            logger.info(f"Backtesting: {label}")
            bt = WalkForwardBacktester(
                signals   = signals_dict[sname],
                close     = close,
                returns   = returns,
                optimizer = portfolio,
            )
            result = bt.run(strategy_name=label)
            results[sname][pname] = result
            logger.success(
                f"{label}: Sharpe={result.metrics.get('sharpe')} "
                f"| CAGR={result.metrics.get('cagr')}"
            )

    return results
