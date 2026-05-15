import numpy as np
from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Rank all stocks by N-day return each day, rescale rank to [-1, +1].
    Top-ranked → +1 (long), bottom-ranked → -1 (short).
    """

    def __init__(self):
        super().__init__("cross_sectional_momentum", STRATEGY_PARAMS["cross_sectional_momentum"])

    def generate_signals(self, close, _returns, **_kwargs):
        lb     = self.params["lookback"]
        method = self.params["rank_method"]

        cum_ret = close / close.shift(lb) - 1
        ranked  = cum_ret.rank(axis=1, ascending=True, method=method)
        n_valid = cum_ret.notna().sum(axis=1)
        signals = (ranked.sub(1, axis=0)).div(n_valid - 1, axis=0) * 2 - 1
        return signals.where(cum_ret.notna(), np.nan)
