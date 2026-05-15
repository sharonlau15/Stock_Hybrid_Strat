"""
portfolio/min_variance.py
=========================
Minimize portfolio variance (w^T Σ w) subject to weight constraints.
Signals are used only as a mask: only positive-signal assets are eligible.

This portfolio ignores expected returns and focuses purely on
diversification / low-volatility. It often outperforms max-Sharpe
when signal quality is poor.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger

from config.settings import MAX_WEIGHT_SUM, MAX_POSITION_SIZE, LONG_SHORT
from portfolio.base import BasePortfolio


def _equal_weight_fallback(eligible, max_w_sum, max_pos) -> dict:
    if eligible.empty:
        return {}
    w_each = min(max_w_sum / len(eligible), max_pos)
    return {sym: w_each for sym in eligible.index}


class MinVariancePortfolio(BasePortfolio):
    """Minimum-variance portfolio constrained to positive-signal stocks."""

    def __init__(self, params: dict = None):
        super().__init__("min_variance", params)

    def compute_weights(
        self,
        signals:    pd.Series,
        cov:        pd.DataFrame,
        long_short: bool = LONG_SHORT,
    ) -> dict:
        max_w_sum = self.params.get("max_w_sum", MAX_WEIGHT_SUM)
        max_pos   = self.params.get("max_pos",   MAX_POSITION_SIZE)

        # Filter to investable universe (positive signal only for long-only)
        pos = signals[signals > 0]
        if pos.empty:
            return {}

        symbols = pos.index.tolist()
        n       = len(symbols)
        cov_arr = cov.reindex(index=symbols, columns=symbols).fillna(0).values
        cov_arr += np.eye(n) * 1e-6

        bounds      = [(0.0, max_pos)] * n
        constraints = [{"type": "ineq", "fun": lambda w: max_w_sum - w.sum()}]

        # Seed with equal weight
        w0 = np.full(n, min(max_w_sum / n, max_pos))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
            result = minimize(
                fun         = lambda w: float(w @ cov_arr @ w),
                x0          = w0,
                method      = "SLSQP",
                bounds      = bounds,
                constraints = constraints,
                options     = {"maxiter": 1000, "ftol": 1e-10},
            )

        if not result.success:
            logger.debug(f"MinVariance failed: {result.message} — equal-weight fallback")
            return _equal_weight_fallback(pos, max_w_sum, max_pos)

        weights = pd.Series(result.x, index=symbols)
        weights[weights.abs() < 0.001] = 0.0
        return weights.to_dict()
