"""
portfolio/signal_weighted.py
=============================
Weight proportional to signal magnitude — no optimizer needed.

w_i = signal_i / sum(signal_j for j > 0)  * max_w_sum

Iteratively caps positions at max_pos and reallocates excess weight
to uncapped positions (same logic as the live engine's signal_to_weights).
This is the fastest portfolio constructor and makes a good baseline
for signal quality: if signal_weighted beats optimized portfolios, it
suggests the signal itself is the dominant source of alpha.
"""

import pandas as pd
import numpy as np

from config.settings import MAX_WEIGHT_SUM, MAX_POSITION_SIZE, LONG_SHORT
from portfolio.base import BasePortfolio


class SignalWeightedPortfolio(BasePortfolio):
    """Weight ∝ signal strength with iterative position caps."""

    def __init__(self, params: dict = None):
        super().__init__("signal_weighted", params)

    def compute_weights(
        self,
        signals:    pd.Series,
        cov:        pd.DataFrame,
        long_short: bool = LONG_SHORT,
    ) -> dict:
        max_w_sum = self.params.get("max_w_sum", MAX_WEIGHT_SUM)
        max_pos   = self.params.get("max_pos",   MAX_POSITION_SIZE)

        weights = pd.Series(0.0, index=signals.index)

        pos = signals[signals > 0]
        if not pos.empty:
            raw = pos / pos.sum() * max_w_sum
            weights[pos.index] = _cap_and_reallocate(raw, max_pos)

        if long_short:
            neg = signals[signals < 0].abs()
            if not neg.empty:
                raw = neg / neg.sum() * max_w_sum
                weights[neg.index] = -_cap_and_reallocate(raw, max_pos)

        return weights.to_dict()


def _cap_and_reallocate(raw: pd.Series, max_pos: float) -> pd.Series:
    """Iteratively cap at max_pos and reallocate excess to uncapped positions."""
    w = raw.copy()
    for _ in range(len(w)):
        excess   = (w - max_pos).clip(lower=0).sum()
        w        = w.clip(upper=max_pos)
        uncapped = w[w < max_pos]
        if excess < 1e-8 or uncapped.empty:
            break
        w[uncapped.index] += excess / len(uncapped)
    return w
