"""
portfolio/optimizer.py
======================
Backwards-compatibility shim.

The optimizer logic has moved to portfolio/max_sharpe.py.
This module re-exports max_sharpe_optimize and _portfolio_stats so that
any existing imports continue to work unchanged.
"""

from portfolio.max_sharpe import (  # noqa: F401  (re-export)
    MaxSharpePortfolio,
    max_sharpe_optimize,
    _portfolio_stats,
    _equal_weight_fallback,
)

__all__ = [
    "MaxSharpePortfolio",
    "max_sharpe_optimize",
    "_portfolio_stats",
    "_equal_weight_fallback",
]
