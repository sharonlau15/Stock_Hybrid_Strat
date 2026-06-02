"""
strategies/coin_flip.py
=======================
Coin Flip Strategy — pure random signals as a benchmark baseline.

Each day, each stock gets +1 (buy) or 0 (flat) with equal probability.
Long-only to match the rest of the system. Seeded per-run for
reproducibility — change the seed to get a different random path.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class CoinFlipStrategy(BaseStrategy):

    def __init__(self, params: dict = None):
        defaults = STRATEGY_PARAMS.get("coin_flip", {})
        super().__init__("coin_flip", {**defaults, **(params or {})})

    def generate_signals(
        self,
        close:   pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        seed = self.params.get("seed", 42)
        rng  = np.random.default_rng(seed)

        # Each cell independently 0 or +1 with equal probability
        flips = rng.integers(0, 2, size=close.shape).astype(float)
        return pd.DataFrame(flips, index=close.index, columns=close.columns)
