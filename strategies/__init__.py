"""
strategies/__init__.py
======================
Strategy registry — the single place to add or remove strategies.

To add a new strategy
---------------------
  1. Create  strategies/your_strategy.py  with a class that extends BaseStrategy
  2. Import it below
  3. Add the class to REGISTRY

That's it. Nothing else in the codebase needs to change.
"""

from strategies.momentum         import MomentumStrategy
from strategies.mean_reversion   import MeanReversionStrategy
from strategies.risk_parity      import RiskParityStrategy
from strategies.cross_sectional  import CrossSectionalMomentumStrategy
from strategies.vol_breakout     import VolBreakoutStrategy
from strategies.ml_signal        import MLSignalStrategy
from strategies.exhaustion_fade  import ExhaustionFadeStrategy
from strategies.sma_brownian              import SMABrownianStrategy
from strategies.seasonal_exhaustion_fade  import SeasonalExhaustionFadeStrategy
from strategies.macro_regime              import MacroRegimeStrategy

# ── Registry ───────────────────────────────────────────────────────────────────
REGISTRY: list[type] = [
    MomentumStrategy,
    MeanReversionStrategy,
    RiskParityStrategy,
    CrossSectionalMomentumStrategy,
    VolBreakoutStrategy,
    MLSignalStrategy,
    ExhaustionFadeStrategy,
    SMABrownianStrategy,
    SeasonalExhaustionFadeStrategy,
    MacroRegimeStrategy,
]


def get_all_strategies() -> list:
    """Instantiate and return every registered strategy."""
    return [cls() for cls in REGISTRY]


def get_strategy(name: str):
    """Return a single strategy instance by its name string."""
    for cls in REGISTRY:
        instance = cls()
        if instance.name == name:
            return instance
    available = [cls().name for cls in REGISTRY]
    raise KeyError(f"No strategy named '{name}'. Available: {available}")
