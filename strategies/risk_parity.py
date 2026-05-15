import numpy as np
from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class RiskParityStrategy(BaseStrategy):
    """
    Equal risk contribution: weights ∝ 1 / rolling_vol.
    Long-only, always fully invested.
    """

    def __init__(self):
        super().__init__("risk_parity", STRATEGY_PARAMS["risk_parity"])

    def generate_signals(self, close, returns, **_kwargs):
        lb      = self.params["lookback"]
        inv_vol = (1.0 / returns.rolling(lb).std()).replace([np.inf, -np.inf], np.nan)
        row_sum = inv_vol.sum(axis=1, skipna=True)
        signals = inv_vol.div(row_sum.replace(0, np.nan), axis=0).fillna(0)
        return signals.reindex(close.index, fill_value=0).ffill().fillna(0)
