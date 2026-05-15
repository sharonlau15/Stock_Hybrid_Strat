"""
portfolio/__init__.py
=====================
Registry for all portfolio constructors.

Add a new portfolio type by:
  1. Creating portfolio/my_portfolio.py with a class that extends BasePortfolio
  2. Importing it here and adding it to REGISTRY
  No other files need to change.

Usage
-----
  from portfolio import get_portfolio, get_all_portfolios, REGISTRY

  # Get one by name
  port = get_portfolio("max_sharpe")

  # Get all for a comparison run
  portfolios = get_all_portfolios()
"""

from portfolio.max_sharpe     import MaxSharpePortfolio
from portfolio.equal_weight   import EqualWeightPortfolio
from portfolio.min_variance   import MinVariancePortfolio
from portfolio.risk_parity_port import RiskParityPortfolio
from portfolio.signal_weighted import SignalWeightedPortfolio

REGISTRY: list[type] = [
    MaxSharpePortfolio,
    EqualWeightPortfolio,
    MinVariancePortfolio,
    RiskParityPortfolio,
    SignalWeightedPortfolio,
]

# name → class mapping (keyed by instance name, not class name)
_NAME_MAP: dict[str, type] = {cls().__class__.__name__: cls for cls in REGISTRY}
_NAME_MAP.update({cls().name: cls for cls in REGISTRY})


def get_portfolio(name: str, params: dict = None):
    """Return an instantiated portfolio by name (e.g. 'max_sharpe')."""
    cls = _NAME_MAP.get(name)
    if cls is None:
        available = [cls().name for cls in REGISTRY]
        raise ValueError(f"Unknown portfolio '{name}'. Available: {available}")
    return cls(params=params)


def get_all_portfolios(params_override: dict = None) -> list:
    """Return one instance of every registered portfolio type."""
    params_override = params_override or {}
    return [
        cls(params=params_override.get(cls().name))
        for cls in REGISTRY
    ]
