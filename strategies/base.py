"""
strategies/base.py
==================
Abstract base class for all strategies.
Every strategy must implement generate_signals() and returns a
signal DataFrame with values in [-1, +1] (negative = short conviction).
"""

import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, name: str, params: dict):
        self.name   = name
        self.params = params

    @abstractmethod
    def generate_signals(self, close: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute signal matrix.

        Returns
        -------
        pd.DataFrame — same shape as close, values in [-1, +1]
            Positive = long conviction, negative = short conviction, 0 = flat.
        """

    def run(self, close: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate signals and apply T+1 shift (no look-ahead)."""
        signals = self.generate_signals(close, returns, **kwargs)
        return signals.shift(1).fillna(0)
