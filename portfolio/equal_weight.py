"""
portfolio/equal_weight.py
=========================
Distribute capital equally across all positive-signal stocks.
No covariance required — pure signal filtering.

Long-only: equal share among top signals.
Long-short: equal share long for positive signals, equal share short for negative.
"""

import pandas as pd

from config.settings import MAX_WEIGHT_SUM, MAX_POSITION_SIZE, LONG_SHORT
from portfolio.base import BasePortfolio


class EqualWeightPortfolio(BasePortfolio):
    """Naive equal-weight baseline — useful as a performance lower bound."""

    def __init__(self, params: dict = None):
        super().__init__("equal_weight", params)

    def compute_weights(
        self,
        signals:    pd.Series,
        cov:        pd.DataFrame,
        long_short: bool = LONG_SHORT,
    ) -> dict:
        max_w_sum = self.params.get("max_w_sum", MAX_WEIGHT_SUM)
        max_pos   = self.params.get("max_pos",   MAX_POSITION_SIZE)

        pos = signals[signals > 0]
        neg = signals[signals < 0]
        w   = pd.Series(0.0, index=signals.index)

        if not pos.empty:
            w[pos.index] = min(max_w_sum / len(pos), max_pos)

        if long_short and not neg.empty:
            w[neg.index] = -min(max_w_sum / len(neg), max_pos)

        return w.to_dict()
