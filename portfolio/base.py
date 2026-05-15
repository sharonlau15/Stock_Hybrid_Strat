"""
portfolio/base.py
=================
Abstract base class for all portfolio constructors.

Every portfolio class must implement compute_weights(), which converts a
signal row and a covariance matrix into a symbol → weight dict.  Because
BasePortfolio implements __call__, any instance is also a drop-in callable
for the existing WalkForwardBacktester.optimizer interface.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BasePortfolio(ABC):
    def __init__(self, name: str, params: dict = None):
        self.name   = name
        self.params = params or {}

    @abstractmethod
    def compute_weights(
        self,
        signals:    pd.Series,
        cov:        pd.DataFrame,
        long_short: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        signals    : Series[float] — signal per symbol, values in [-1, +1]
        cov        : DataFrame — annualized covariance matrix (symbols × symbols)
        long_short : bool — allow short positions if True

        Returns
        -------
        dict: symbol → weight  (weights may not sum to exactly 1)
        """
        ...

    def __call__(
        self,
        signals:    pd.Series,
        cov:        pd.DataFrame,
        long_short: bool = False,
    ) -> dict:
        return self.compute_weights(signals, cov, long_short)
