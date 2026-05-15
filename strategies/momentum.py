import pandas as pd
from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class MomentumStrategy(BaseStrategy):
    """
    12-1 month time-series momentum.
    Ranks stocks by [lookback_long - lookback_short] cumulative return.
    Top N → long. Long-only by default (bottom_n = 0 in settings).
    """

    def __init__(self):
        super().__init__("momentum", STRATEGY_PARAMS["momentum"])

    def generate_signals(self, close, _returns, **_kwargs):
        p     = self.params
        top_n = p["top_n"]
        bot_n = p["bottom_n"]

        score   = close / close.shift(p["lookback_long"]) - close / close.shift(p["lookback_short"])
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for dt in score.index:
            row = score.loc[dt].dropna()
            if len(row) < top_n + bot_n:
                continue
            ranked = row.rank(ascending=True)
            n      = len(ranked)
            signals.loc[dt, ranked[ranked >= n - top_n + 1].index] = 1.0
            if bot_n:
                signals.loc[dt, ranked[ranked <= bot_n].index] = -1.0

        return signals
