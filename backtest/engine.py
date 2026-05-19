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


@dataclass
class SplitBacktestResult:
    """Three-way chronological split of a single strategy backtest."""
    strategy_name: str
    train_end:     pd.Timestamp
    val_end:       pd.Timestamp
    train:         BacktestResult
    val:           BacktestResult
    test:          BacktestResult


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

    def run_with_splits(
        self,
        strategy_name: str   = "unnamed",
        train_frac:    float  = 0.70,
        val_frac:      float  = 0.15,
    ) -> "SplitBacktestResult":
        """
        Run the full backtest once, then slice the return series into
        Train / Validation / Test periods.

        Splitting AFTER the full run is the only correct approach.
        Splitting BEFORE and re-running each period separately would cause the
        optimizer to use a shorter covariance history on val/test, changing
        both weights and returns — invalidating the cross-period comparison.
        Here, the covariance at any date always uses all returns up to that
        date, exactly as it would in live trading.
        """
        full = self.run(strategy_name)
        ret  = full.portfolio_returns

        if len(ret) < 60:
            raise ValueError(
                f"{strategy_name}: only {len(ret)} observations after warmup — "
                "need ≥ 60 to split into three periods."
            )

        train_end, val_end = compute_split_dates(ret.index, train_frac, val_frac)

        def _slice(label: str, mask) -> "BacktestResult":
            r   = ret.loc[mask]
            wh  = full.weights_history.reindex(r.index, fill_value=0)
            sig = full.signal_history.reindex(r.index, fill_value=0)
            return BacktestResult(
                strategy_name    = label,
                portfolio_returns = r,
                weights_history  = wh,
                signal_history   = sig,
            )

        return SplitBacktestResult(
            strategy_name = strategy_name,
            train_end     = train_end,
            val_end       = val_end,
            train = _slice(f"{strategy_name}/train", ret.index <= train_end),
            val   = _slice(f"{strategy_name}/val",   (ret.index > train_end) & (ret.index <= val_end)),
            test  = _slice(f"{strategy_name}/test",   ret.index > val_end),
        )


# ── Train / Validation / Test split helpers ────────────────────────────────────

def compute_split_dates(
    index:      pd.DatetimeIndex,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute chronological split boundary dates from a time-series index.

    Splits are always contiguous and non-overlapping:
        [0, train_end]       — training period   (train_frac of bars)
        (train_end, val_end] — validation period  (val_frac of bars)
        (val_end,   end]     — test period        (remaining bars)

    Parameters
    ----------
    index      : the portfolio-returns DatetimeIndex (post-warmup)
    train_frac : fraction of bars assigned to training   (default 0.70)
    val_frac   : fraction of bars assigned to validation (default 0.15)

    Returns
    -------
    (train_end, val_end) — both dates inclusive upper bounds for their period
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")
    n         = len(index)
    train_end = index[int(n * train_frac) - 1]
    val_end   = index[int(n * (train_frac + val_frac)) - 1]
    return train_end, val_end


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


def run_all_backtests_with_splits(
    strategies,
    signals_dict: dict,
    close,
    returns,
    optimizer,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> dict[str, SplitBacktestResult]:
    """
    Run every strategy through run_with_splits and return a dict of
    SplitBacktestResult keyed by strategy name.
    """
    results: dict[str, SplitBacktestResult] = {}
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
        try:
            sr = bt.run_with_splits(name, train_frac=train_frac, val_frac=val_frac)
            results[name] = sr
            logger.success(
                f"{name}: "
                f"Train={sr.train.metrics.get('sharpe', 'N/A')} | "
                f"Val={sr.val.metrics.get('sharpe', 'N/A')} | "
                f"Test={sr.test.metrics.get('sharpe', 'N/A')} (Sharpe)"
            )
        except Exception as e:
            logger.error(f"Split backtest failed for {name}: {e}")
    return results
