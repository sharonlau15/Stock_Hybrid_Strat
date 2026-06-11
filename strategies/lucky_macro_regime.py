"""
strategies/lucky_macro_regime.py
=================================
Lucky Macro Regime — macro regime gate + coin flip stock selection.

Purpose
-------
Diagnostic / benchmark strategy that answers the question:
  "Is macro regime timing itself valuable, or only in combination with
   smart stock selection?"

Logic
-----
  1. Compute the same 5-indicator macro composite as MacroRegimeStrategy.
  2. Risk-on day  (composite > 0): randomly assign +1 or 0 per stock (coin flip).
  3. Risk-off day (composite ≤ 0): all signals = 0 (stay in cash).

If this strategy outperforms pure CoinFlip, the regime gate is doing real work.
If it underperforms MacroRegimeStrategy, smart allocation still adds value on top.
"""

import numpy as np
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy
from strategies.macro_regime import MacroRegimeStrategy
from config.settings import STRATEGY_PARAMS


class LuckyMacroRegimeStrategy(BaseStrategy):

    def __init__(self, params: dict = None):
        defaults = STRATEGY_PARAMS.get("lucky_macro_regime", {})
        super().__init__("lucky_macro_regime", {**defaults, **(params or {})})
        self._macro = MacroRegimeStrategy()

    def _composite(self, close: pd.DataFrame, returns: pd.DataFrame, **kwargs) -> pd.Series:
        """Re-derive the macro composite score (same logic as MacroRegimeStrategy)."""
        p = self._macro
        spy     = close["SPY"]
        spy_ret = returns["SPY"]

        signal_close = kwargs.get("signal_close")
        if signal_close is None:
            etf_cols     = {"SPY", "QQQ"}
            signal_close = close[[c for c in close.columns if c not in etf_cols]]

        components = [
            p._vix_proxy(spy_ret),
            p._equity_momentum(spy),
            p._market_breadth(signal_close),
            p._vol_regime(spy_ret),
            p._cross_momentum(signal_close),
        ]
        return (
            pd.concat(components, axis=1)
            .mean(axis=1)
            .clip(-1.0, 1.0)
            .reindex(close.index)
            .fillna(0.0)
        )

    def generate_signals(
        self,
        close:   pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        if "SPY" not in close.columns:
            logger.warning("LuckyMacroRegime: SPY not in universe — returning flat")
            return pd.DataFrame(0.0, index=close.index, columns=close.columns)

        seed      = self.params.get("seed", 99)
        composite = self._composite(close, returns, **kwargs)
        risk_on   = composite > 0

        rng    = np.random.default_rng(seed)
        flips  = rng.integers(0, 2, size=close.shape).astype(float)
        flips_df = pd.DataFrame(flips, index=close.index, columns=close.columns)

        # Zero out all risk-off days
        signals = flips_df.where(risk_on, other=0.0)

        risk_on_frac = risk_on.mean()
        logger.info(
            f"LuckyMacroRegime: risk-on {risk_on_frac:.0%} of days "
            f"| avg daily positions {signals.sum(axis=1)[risk_on].mean():.1f} stocks"
        )
        return signals
