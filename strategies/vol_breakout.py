import pandas as pd
from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class VolBreakoutStrategy(BaseStrategy):
    """
    ATR-based channel breakout.
    Signal = daily move / ATR half-band, clipped to [-1, +1].
    """

    def __init__(self):
        super().__init__("vol_breakout", STRATEGY_PARAMS["vol_breakout"])

    @staticmethod
    def _atr(high, low, close, period):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def generate_signals(self, close, _returns, **kwargs):
        high = kwargs.get("high", close)
        low  = kwargs.get("low",  close)
        p    = self.params

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for sym in close.columns:
            atr       = self._atr(high[sym], low[sym], close[sym], p["atr_period"])
            half_band = (p["atr_multiplier"] * atr).replace(0, None)
            signals[sym] = ((close[sym] - close[sym].shift(1)) / half_band).clip(-1, 1).fillna(0)

        return signals
