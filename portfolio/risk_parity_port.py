"""
portfolio/risk_parity_port.py
=============================
Equal Risk Contribution (ERC) portfolio, also called "risk parity".

Each asset contributes the same fraction of total portfolio risk:
    RC_i = w_i * (Σw)_i  →  RC_i == RC_j  ∀ i,j

Implementation
--------------
Minimize sum_i (w_i * MRC_i - target_RC)^2
where MRC_i = (Σw)_i  (marginal risk contribution).

Only positive-signal assets are eligible (long-only mode).
Covariance is required; signals are used as a filter only.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger

from config.settings import MAX_WEIGHT_SUM, MAX_POSITION_SIZE, LONG_SHORT
from portfolio.base import BasePortfolio


def _inv_vol_fallback(symbols, cov_arr, max_w_sum, max_pos) -> dict:
    """Inverse-volatility weighting as ERC approximation when optimizer fails."""
    vols    = np.sqrt(np.diag(cov_arr)).clip(min=1e-8)
    inv_vol = 1.0 / vols
    raw     = inv_vol / inv_vol.sum() * max_w_sum
    capped  = np.minimum(raw, max_pos)
    # Reallocate leftover from capped positions
    slack = max_w_sum - capped.sum()
    if slack > 1e-6:
        uncapped = capped < max_pos
        if uncapped.any():
            capped[uncapped] += slack / uncapped.sum()
            capped = np.minimum(capped, max_pos)
    return dict(zip(symbols, capped))


class RiskParityPortfolio(BasePortfolio):
    """Equal Risk Contribution (ERC) portfolio restricted to positive-signal assets."""

    def __init__(self, params: dict = None):
        super().__init__("risk_parity", params)

    def compute_weights(
        self,
        signals:    pd.Series,
        cov:        pd.DataFrame,
        long_short: bool = LONG_SHORT,
    ) -> dict:
        max_w_sum = self.params.get("max_w_sum", MAX_WEIGHT_SUM)
        max_pos   = self.params.get("max_pos",   MAX_POSITION_SIZE)

        pos = signals[signals > 0]
        if pos.empty:
            return {}

        symbols = pos.index.tolist()
        n       = len(symbols)
        cov_arr = cov.reindex(index=symbols, columns=symbols).fillna(0).values
        cov_arr += np.eye(n) * 1e-6

        target_rc = 1.0 / n  # equal fractional contribution

        def erc_objective(w):
            port_var = float(w @ cov_arr @ w)
            if port_var < 1e-12:
                return 0.0
            mrc = cov_arr @ w          # marginal risk contribution (unnormalized)
            rc  = w * mrc / port_var   # fractional risk contribution
            return float(np.sum((rc - target_rc) ** 2))

        bounds      = [(0.0, max_pos)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - max_w_sum}]
        w0          = np.full(n, max_w_sum / n)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
            result = minimize(
                fun         = erc_objective,
                x0          = w0,
                method      = "SLSQP",
                bounds      = bounds,
                constraints = constraints,
                options     = {"maxiter": 2000, "ftol": 1e-10},
            )

        if not result.success or result.fun > 0.01:
            logger.debug(f"RiskParity ERC failed: {result.message} — inv-vol fallback")
            return _inv_vol_fallback(symbols, cov_arr, max_w_sum, max_pos)

        weights = pd.Series(result.x, index=symbols)
        weights[weights.abs() < 0.001] = 0.0
        return weights.to_dict()
