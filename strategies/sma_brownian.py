"""
strategies/sma_brownian.py
==========================
SMA crossover filtered and scaled by Geometric Brownian Motion drift.

Concept
-------
Under GBM a stock's log-price follows:
    d(ln S) = (μ - σ²/2) dt + σ dW

The term (μ - σ²/2) is the *realised drift* of log-returns and is a
forward-looking edge estimate: positive drift → expect upward movement,
negative → downward.

Signal construction
-------------------
1. Direction  — fast SMA above slow SMA → +1 (bullish), else −1 (bearish).
2. Drift      — rolling (μ − σ²/2) estimated from recent daily log-returns,
                normalised to [−1, +1] by dividing by 2× its rolling std.
3. Agreement  — only emit a signal when direction and drift agree in sign:
                  signal = direction × clip(direction × drift_norm, 0, 1)
                Conflicting regimes (trend says up but drift says down, or
                vice-versa) return 0 (flat).

This combination means we ride momentum only when the underlying GBM
process also supports the move — filtering out whipsaw crossovers in
mean-reverting, low-drift regimes.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from config.settings import STRATEGY_PARAMS


class SMABrownianStrategy(BaseStrategy):

    def __init__(self):
        super().__init__("sma_brownian", STRATEGY_PARAMS.get("sma_brownian", {}))
        p = self.params
        self.fast_window  = p.get("fast_window",  20)   # fast SMA bars
        self.slow_window  = p.get("slow_window",  60)   # slow SMA bars
        self.drift_window = p.get("drift_window", 63)   # GBM estimation window (≈ 1 quarter)
        self.min_signal   = p.get("min_signal",   0.10) # ignore very weak signals

    def generate_signals(self, close: pd.DataFrame, returns: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for ticker in close.columns:
            px  = close[ticker].dropna()
            ret = returns[ticker].reindex(px.index)

            if len(px) < self.slow_window + self.drift_window:
                continue

            # ── 1. SMA crossover direction ─────────────────────────────────────
            fast_sma  = px.rolling(self.fast_window).mean()
            slow_sma  = px.rolling(self.slow_window).mean()
            direction = np.sign(fast_sma - slow_sma)   # +1 / −1 / 0

            # ── 2. GBM drift: μ − σ²/2 (Itô's correction) ────────────────────
            # Annualise so the magnitude is comparable across tickers and dates.
            mu     = ret.rolling(self.drift_window).mean() * 252
            sigma2 = ret.rolling(self.drift_window).var()  * 252
            drift  = mu - 0.5 * sigma2

            # Normalise drift: z-score capped at ±2σ → scaled to [−1, +1]
            drift_std  = drift.rolling(self.drift_window).std().clip(lower=1e-8)
            drift_norm = (drift / (2 * drift_std)).clip(-1.0, 1.0)

            # ── 3. Combine — emit signal only when direction and drift agree ───
            # clip(direction × drift_norm, 0, 1) gives the agreement strength:
            #   direction=+1, drift_norm=+0.7  → signal = +0.7  (long, strong)
            #   direction=+1, drift_norm=−0.3  → signal =  0.0  (flat, conflict)
            #   direction=−1, drift_norm=−0.5  → signal = −0.5  (short, moderate)
            raw = direction * (direction * drift_norm).clip(0, 1)

            # Mute very weak signals below the threshold
            raw = raw.where(raw.abs() >= self.min_signal, 0.0)

            # Zero out bars where SMAs or drift are not yet formed
            valid = slow_sma.notna() & drift_norm.notna()
            raw   = raw.where(valid, 0.0)

            signals[ticker] = raw.reindex(signals.index, fill_value=0.0)

        return signals
