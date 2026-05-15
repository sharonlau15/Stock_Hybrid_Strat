"""
portfolio/max_sharpe.py
=======================
Maximize Sharpe ratio via SLSQP.  Signal acts as expected-return proxy.

Constraints
-----------
  - annualized vol ≥ min_vol  (prevents degenerate low-risk solutions)
  - sum(|w|) ≤ max_w_sum
  - each |position| ≤ max_pos
  - long-only or long-short depending on long_short flag

Fallback: equal-weight across positive-signal stocks if optimizer fails.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger

from config.settings import MIN_ANNUALIZED_VOL, MAX_WEIGHT_SUM, MAX_POSITION_SIZE, LONG_SHORT
from portfolio.base import BasePortfolio


def _portfolio_stats(w, mu, cov):
    port_ret = float(w @ mu)
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(max(port_var, 0))
    sharpe   = port_ret / port_vol if port_vol > 1e-8 else 0.0
    return port_ret, port_vol, sharpe


def _equal_weight_fallback(signals, long_short, max_w_sum, max_pos) -> dict:
    pos = signals[signals > 0]
    neg = signals[signals < 0]
    w   = pd.Series(0.0, index=signals.index)
    if not pos.empty:
        w[pos.index] = min(max_w_sum / len(pos), max_pos)
    if long_short and not neg.empty:
        w[neg.index] = -min(max_w_sum / len(neg), max_pos)
    return w.to_dict()


class MaxSharpePortfolio(BasePortfolio):
    """Maximize Sharpe ratio (signal = μ proxy, covariance from returns history)."""

    def __init__(self, params: dict = None):
        super().__init__("max_sharpe", params)

    def compute_weights(
        self,
        signals:    pd.Series,
        cov:        pd.DataFrame,
        long_short: bool = LONG_SHORT,
    ) -> dict:
        min_vol   = self.params.get("min_vol",   MIN_ANNUALIZED_VOL)
        max_w_sum = self.params.get("max_w_sum", MAX_WEIGHT_SUM)
        max_pos   = self.params.get("max_pos",   MAX_POSITION_SIZE)

        symbols = signals.index.tolist()
        n       = len(symbols)
        sig     = signals.fillna(0).values.astype(float)
        mu      = sig.copy()
        cov_arr = cov.reindex(index=symbols, columns=symbols).fillna(0).values
        cov_arr += np.eye(n) * 1e-6

        bounds = [(-max_pos, max_pos)] * n if long_short else [(0.0, max_pos)] * n

        def neg_sharpe(w):
            _, _, sharpe = _portfolio_stats(w, mu, cov_arr)
            return -sharpe

        constraints = [
            {"type": "ineq", "fun": lambda w: max_w_sum - np.abs(w).sum()},
            {"type": "ineq", "fun": lambda w: float(np.sqrt(max(0.0, float(w @ cov_arr @ w)))) - min_vol},
        ]

        w0 = sig / (np.abs(sig).sum() + 1e-8) * max_w_sum
        w0 = np.clip(w0, *(bounds[0]))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
            result = minimize(
                fun         = neg_sharpe,
                x0          = w0,
                method      = "SLSQP",
                bounds      = bounds,
                constraints = constraints,
                options     = {"maxiter": 1000, "ftol": 1e-9},
            )

        if not result.success:
            logger.debug(f"MaxSharpe failed: {result.message} — equal-weight fallback")
            return _equal_weight_fallback(signals, long_short, max_w_sum, max_pos)

        weights = pd.Series(result.x, index=symbols)
        weights[weights.abs() < 0.001] = 0.0
        return weights.to_dict()


# ── Backwards-compatible module-level function ─────────────────────────────────
def max_sharpe_optimize(
    signals:    pd.Series,
    cov:        pd.DataFrame,
    long_short: bool  = LONG_SHORT,
    min_vol:    float = MIN_ANNUALIZED_VOL,
    max_w_sum:  float = MAX_WEIGHT_SUM,
    max_pos:    float = MAX_POSITION_SIZE,
) -> dict:
    return MaxSharpePortfolio(
        params={"min_vol": min_vol, "max_w_sum": max_w_sum, "max_pos": max_pos}
    ).compute_weights(signals, cov, long_short)
