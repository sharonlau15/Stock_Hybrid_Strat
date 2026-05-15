import numpy as np
from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class MeanReversionStrategy(BaseStrategy):
    """
    Rolling z-score of price vs its own moving average.
    Negative z (oversold) → positive signal. Clipped at ±3σ.
    """

    def __init__(self):
        super().__init__("mean_reversion", STRATEGY_PARAMS["mean_reversion"])

    def generate_signals(self, close, _returns, **_kwargs):
        p  = self.params
        w  = p["zscore_window"]
        ez = p["entry_z"]

        roll_mean = close.rolling(w).mean()
        roll_std  = close.rolling(w).std()
        zscore    = (close - roll_mean) / roll_std.replace(0, np.nan)

        return (-zscore / (ez * 3)).clip(-1, 1).fillna(0)
